"""Jellyfin API client for fancap: browse series/episodes, stream audio, upload subtitles."""
import httpx

ZH_LANGS = ("chi", "zh", "zho", "zh-cn", "chs", "zh-hans", "chs")


class Jellyfin:
    def __init__(self, base_url: str, api_key: str):
        self.base = base_url.rstrip("/")
        self.key = api_key
        # Jellyfin 10.10+/12: X-Emby-Token is rejected; new style is
        # Authorization: MediaBrowser Token="..."
        self.headers = {"Authorization": f'MediaBrowser Token="{api_key}"'}
        self.user_id = None

    def ping(self) -> bool:
        r = httpx.get(f"{self.base}/System/Info/Public", timeout=10)
        return r.status_code == 200

    def get_user_id(self) -> str:
        if self.user_id:
            return self.user_id
        r = httpx.get(f"{self.base}/Users", headers=self.headers, timeout=10)
        r.raise_for_status()
        users = r.json()
        if not users:
            raise RuntimeError("no users on jellyfin server")
        self.user_id = users[0]["Id"]
        return self.user_id

    def series_list(self):
        uid = self.get_user_id()
        r = httpx.get(
            f"{self.base}/Users/{uid}/Items",
            params={"IncludeItemTypes": "Series", "Recursive": "true", "SortBy": "SortName"},
            headers=self.headers, timeout=20,
        )
        r.raise_for_status()
        return [{"id": i["Id"], "name": i.get("Name", "?"),
                 "year": i.get("ProductionYear") or ""}
                for i in r.json().get("Items", [])]

    def seasons(self, series_id: str):
        uid = self.get_user_id()
        r = httpx.get(f"{self.base}/Shows/{series_id}/Seasons",
                      params={"userId": uid}, headers=self.headers, timeout=20)
        r.raise_for_status()
        return [{"id": i["Id"], "name": i.get("Name", "?")}
                for i in r.json().get("Items", [])]

    def episodes(self, series_id: str, season_id: str = None):
        uid = self.get_user_id()
        params = {"userId": uid, "Fields": "MediaStreams"}
        if season_id:
            params["seasonId"] = season_id
        r = httpx.get(f"{self.base}/Shows/{series_id}/Episodes",
                      params=params, headers=self.headers, timeout=20)
        r.raise_for_status()
        out = []
        for i in r.json().get("Items", []):
            streams = i.get("MediaStreams") or []
            subs = [{"idx": s.get("Index"), "lang": s.get("Language") or "?",
                     "disp": s.get("DisplayTitle") or ""}
                    for s in streams if s.get("Type") == "Subtitle"]
            has_zh = any(s["lang"].lower() in ZH_LANGS for s in subs)
            out.append({"id": i["Id"], "name": i.get("Name", "?"),
                        "ep": i.get("IndexNumber"), "has_zh": has_zh, "subs": subs})
        return out

    def stream_url(self, item_id: str) -> str:
        """Original-container stream URL for ffmpeg (api_key in query)."""
        return f"{self.base}/Videos/{item_id}/stream?static=true&api_key={self.key}"

    def upload_subtitle(self, item_id: str, srt_text: str,
                        language: str = "chi", is_forced: bool = False) -> int:
        """Upload srt via POST /Videos/{itemId}/Subtitles (JSON body, base64 Data)."""
        import base64
        body = {
            "Language": language,
            "Format": "srt",
            "IsForced": is_forced,
            "IsHearingImpaired": False,
            "Data": base64.b64encode(srt_text.encode("utf-8")).decode("ascii"),
        }
        r = httpx.post(f"{self.base}/Videos/{item_id}/Subtitles",
                       json=body, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.status_code
