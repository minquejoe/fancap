"""Subtitle source sites: search & download subtitles for anime."""
import os
import re
import httpx

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


# ---------------------------------------------------------------- assrt (官方 API)

def assrt_search(query: str, token: str = "", limit: int = 10):
    """assrt.net official API. token: free token from assrt.net/user/profile."""
    params = {"q": query, "page": 1}
    if token:
        params["token"] = token
    r = httpx.get("https://api.assrt.net/v1/sub/search", params=params,
                  headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    d = r.json()
    if d.get("status") != 0:
        return []
    out = []
    subs = (d.get("sub") or {}).get("subs") or []
    for s in subs[:limit]:
        out.append({
            "source": "assrt",
            "title": s.get("name", ""),
            "id": s.get("id"),
            "lang": ",".join(s.get("lang", {}).get("desc", []) or []),
            "url": f"https://assrt.net/sub/{s.get('id')}.html",
        })
    return out


def assrt_download(sub_id: str, token: str = "") -> bytes | None:
    """Download subtitle archive from assrt. Needs token for download URLs."""
    if not token:
        return None
    r = httpx.get(f"https://api.assrt.net/v1/sub/detail",
                  params={"id": sub_id, "token": token},
                  headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    d = r.json()
    subs = (d.get("sub") or {}).get("subs") or []
    if not subs:
        return None
    files = subs[0].get("filelist") or []
    if not files:
        return None
    url = "https://dl.assrt.net/download/" + files[0].get("url", "").lstrip("/")
    if not files[0].get("url"):
        return None
    r = httpx.get(url, headers={"User-Agent": UA}, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------- subtitlecat (无 key)

def subtitlecat_search(query: str, limit: int = 10):
    """subtitlecat.com: scrape search results (no key, no heavy anti-bot).
    Result links point at subs/<id>/<title>.<lang>.html pages; the srt download
    link is inside that page."""
    import time
    r = None
    for attempt in range(3):
        try:
            r = httpx.get("https://www.subtitlecat.com/index.php",
                          params={"search": query},
                          headers={"User-Agent": UA}, timeout=30, follow_redirects=True)
            r.raise_for_status()
            break
        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt == 2:
                raise
            time.sleep(3)
    out = []
    # relative links: subs/<id>/<name>.<lang>.html (no leading slash on real pages)
    for m in re.finditer(r'<a href="(subs/[^"]+\.html)"[^>]*>([^<]+)</a>'
                         r'(?:\s*\((translated from [^)]+)\))?', r.text):
        href, title, tr = m.group(1), m.group(2), m.group(3) or ""
        url = f"https://www.subtitlecat.com/{href}"
        # chinese detection from title suffix (.sc/.chs/.cht/.zh) or "(translated from Chinese)"
        zh = ("Chinese" in tr or is_chinese_subtitle(title))
        out.append({"source": "subtitlecat", "title": title.strip(), "url": url,
                    "lang": "zh" if zh else "foreign", "translated_from": tr})
        if len(out) >= limit:
            break
    return out


def subtitlecat_download(url: str, prefer_zh: bool = True) -> tuple[bytes, str]:
    """Download srt from a result page: fetch page, pick best srt link.
    Page lists per-language links: /subs/<id>/<name>-<lang>.srt."""
    import time
    r = None
    for attempt in range(3):
        try:
            r = httpx.get(url, headers={"User-Agent": UA}, timeout=30, follow_redirects=True)
            r.raise_for_status()
            break
        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt == 2:
                raise
            time.sleep(3)
    links = re.findall(r'<a[^>]+href="(/subs/[^"]+\.srt)"[^>]*>', r.text)
    if not links:
        links = re.findall(r'<a[^>]+href="(subs/[^"]+\.srt)"', r.text)
    if not links:
        raise RuntimeError("no srt link on result page: " + url)
    # pick: zh variants first (chs/zh/sc-cn), else first
    def _zh(l):
        return bool(re.search(r'[-.](chs|cht|zh|sc|cn|zh-hans|zh-hant)\.srt$', l, re.I))
    zh_links = [l for l in links if _zh(l)]
    pick = (zh_links if (prefer_zh and zh_links) else links)[0]
    dl = "https://www.subtitlecat.com" + (pick if pick.startswith("/") else "/" + pick)
    for attempt in range(3):
        try:
            r = httpx.get(dl, headers={"User-Agent": UA}, timeout=60, follow_redirects=True)
            r.raise_for_status()
            return r.content, dl
        except httpx.ConnectError:
            if attempt == 2:
                raise
            time.sleep(3)


# ---------------------------------------------------------------- 统一接口

def search_all(query: str, assrt_token: str = "", limit: int = 10) -> list[dict]:
    """Search all sources, merge results. assrt first when token present."""
    results = []
    if assrt_token:
        try:
            results += assrt_search(query, assrt_token, limit)
        except Exception:
            pass
    try:
        results += subtitlecat_search(query, limit)
    except Exception:
        pass
    return results


def is_chinese_subtitle(title: str) -> bool:
    """Heuristic: title mentions Chinese."""
    t = title.lower()
    return any(k in t for k in ("chinese", "chs", "cht", "zh", "简体", "繁體", "繁体", "中字", "中英"))


def looks_chinese_content(text: str) -> bool:
    return len(re.findall(r'[一-鿿]', text)) > len(text) * 0.05
