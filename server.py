#!/usr/bin/env python3
"""fancap web UI - FastAPI server: settings, task submit, progress, Jellyfin browser."""
import os
import queue
import subprocess
import sys
import threading
import uuid
import yaml
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

import fancap

WORK = Path("/home/orangepi/fancap-work")
CONFIG_PATH = Path.home() / ".config/fancap/config.yaml"
app = FastAPI(title="fancap")

_tasks = {}          # id -> {status, stage, log[], files, error}
_queue = queue.Queue()


# ---------------------------------------------------------------- config

def load_cfg() -> fancap.Config:
    cfg = fancap.Config()
    if CONFIG_PATH.exists():
        d = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        for k, v in (d.get("translate") or {}).items():
            setattr(cfg, k, v)
        cfg.asr_backend = (d.get("asr") or {}).get("backend", cfg.asr_backend)
        cfg.model_dir = (d.get("asr") or {}).get("model_dir", cfg.model_dir)
    return cfg


def save_cfg(d: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(d, allow_unicode=True))


def _load_yaml() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {}


# ---------------------------------------------------------------- worker

def _worker():
    while True:
        tid = _queue.get()
        t = _tasks[tid]
        try:
            t["status"] = "running"

            def log(msg):
                t["log"].append(msg)
                t["stage"] = msg.split("]")[0].lstrip("[") if msg.startswith("[") else t["stage"]

            cfg = load_cfg()
            prefix = str(WORK / "output" / tid)
            os.makedirs(WORK / "output", exist_ok=True)
            # run pipeline in an isolated subprocess: RKNN context + big model
            # mmap is unreliable inside a long-lived uvicorn process
            script = (
                "import sys; sys.path.insert(0, %r)\n"
                "import fancap\n"
                "cfg = fancap.Config()\n"
                "cfg.model_dir = %r\n"
                "cfg.base_url = %r\n"
                "cfg.api_key = %r\n"
                "cfg.model = %r\n"
                "cfg.prompt = %r\n"
            ) % (str(WORK), cfg.model_dir, cfg.base_url, cfg.api_key, cfg.model,
                 getattr(cfg, "prompt", ""))
            if t["mode"] == "jf":
                script += (
                    "fancap.pipeline_jf(%r, %r, %r, %r, cfg, log=print)\n"
                ) % (t["jf_url"], t["jf_key"], t["item_id"], t.get("ep_name", ""))
            else:
                script += "fancap.pipeline_%s(%r, %r, cfg, log=print)\n" % (
                    t["mode"], t["input"], prefix)
            proc = subprocess.run(
                [sys.executable, "-"], input=script,
                capture_output=True, text=True, timeout=3600,
            )
            for line in (proc.stdout + proc.stderr).splitlines():
                if line.strip():
                    log(line)
            if proc.returncode != 0:
                raise RuntimeError(f"pipeline exited {proc.returncode}")
            t["status"] = "done"
            t["files"] = [prefix + ".srt", prefix + ".ass"] if t["mode"] != "jf" else []
        except Exception as e:
            t["status"] = "error"
            t["error"] = str(e)
            t["log"].append(f"ERROR: {e}")
        finally:
            _queue.task_done()


threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------- api

@app.post("/api/task")
async def submit_task(
    mode: str = Form(...),                  # transcribe | translate | auto
    video: UploadFile | None = File(None),
    subtitle: UploadFile | None = File(None),
):
    if mode not in ("transcribe", "translate", "auto"):
        raise HTTPException(400, "bad mode")
    src = subtitle if mode == "translate" else video
    if src is None:
        raise HTTPException(400, "need file")
    tid = uuid.uuid4().hex[:8]
    dest = WORK / "input" / f"{tid}-{src.filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        while chunk := await src.read(1 << 20):
            f.write(chunk)
    _tasks[tid] = {"status": "queued", "stage": "-", "log": [], "files": [],
                   "mode": mode, "input": str(dest), "error": None}
    _queue.put(tid)
    return {"id": tid}


@app.get("/api/task/{tid}")
async def task_status(tid: str):
    if tid not in _tasks:
        raise HTTPException(404)
    return _tasks[tid]


@app.get("/api/tasks")
async def all_tasks():
    return {k: {k2: v.get(k2) for k2 in ("status", "stage", "mode", "error")}
            for k, v in _tasks.items()}


@app.get("/api/config")
async def get_config():
    return _load_yaml() or {"jellyfin": {"base_url": "", "api_key": ""}}


@app.post("/api/config")
async def set_config(d: dict):
    save_cfg(d)
    return {"ok": True}


@app.get("/download/{tid}/{name}")
async def download(tid: str, name: str):
    path = WORK / "output" / f"{tid}.{name}"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, filename=path.name)


# ---------------------------------------------------------------- jellyfin

def _jf_client():
    jd = _load_yaml().get("jellyfin") or {}
    base = jd.get("base_url", "")
    key = jd.get("api_key", "")
    if not base or not key:
        raise HTTPException(400, "jellyfin not configured")
    from jellyfin import Jellyfin
    return Jellyfin(base, key)


@app.get("/api/jf/series")
async def jf_series():
    jf = _jf_client()
    try:
        return jf.series_list()
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/jf/episodes/{series_id}")
async def jf_episodes(series_id: str, season_id: str = None):
    jf = _jf_client()
    try:
        return jf.episodes(series_id, season_id)
    except Exception as e:
        raise HTTPException(502, str(e))


@app.post("/api/jf/queue")
async def jf_queue(items: list[dict]):
    """items: [{id, name}] of episode item ids to process."""
    jd = _load_yaml().get("jellyfin") or {}
    base = jd.get("base_url", "")
    key = jd.get("api_key", "")
    if not base or not key:
        raise HTTPException(400, "jellyfin not configured")
    queued = []
    for it in items:
        tid = uuid.uuid4().hex[:8]
        _tasks[tid] = {"status": "queued", "stage": "-", "log": [], "files": [],
                       "mode": "jf", "error": None,
                       "jf_url": base, "jf_key": key,
                       "item_id": it["id"], "ep_name": it.get("name", "")}
        _queue.put(tid)
        queued.append(tid)
    return {"queued": queued}


@app.get("/api/prompt")
async def get_prompt():
    d = _load_yaml()
    tr = d.get("translate") or {}
    return {"prompt": tr.get("prompt", ""), "default": fancap.DEFAULT_TRANSLATE_PROMPT}


@app.post("/api/prompt")
async def set_prompt(body: dict):
    d = _load_yaml()
    tr = d.get("translate") or {}
    tr["prompt"] = body.get("prompt", "")
    d["translate"] = tr
    save_cfg(d)
    return {"ok": True}


@app.post("/api/test-translate")
async def test_translate():
    """Test the configured translation endpoint with a tiny request."""
    cfg = load_cfg()
    if not cfg.api_key:
        raise HTTPException(400, "未配置 API Key")
    from openai import OpenAI
    try:
        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=20)
        t0 = __import__("time").time()
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": "把「今日はいい天気ですね」翻译成中文，只输出译文"}],
            temperature=0,
        )
        dt = __import__("time").time() - t0
        text = resp.choices[0].message.content.strip()
        return {"ok": True, "latency": round(dt, 2), "sample": text[:100],
                "model": cfg.model, "base_url": cfg.base_url}
    except Exception as e:
        return JSONResponse(status_code=502, content={
            "ok": False, "error": str(e)[:400],
            "base_url": cfg.base_url, "model": cfg.model})


# ---------------------------------------------------------------- ui

@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>fancap · 番剧中文字幕</title>
<style>
:root{
 --bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--border:#2d3748;
 --fg:#e6edf3;--muted:#8b949e;--accent:#6366f1;--accent2:#8b5cf6;
 --ok:#34d399;--err:#f87171;--run:#60a5fa;
}
*{box-sizing:border-box}
body{font-family:"Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;min-height:100vh}
.wrap{max-width:920px;margin:0 auto;padding:1.5rem 1rem 4rem}
header{display:flex;align-items:center;gap:.7rem;padding:.6rem 0 1.2rem}
.logo{width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent2));
 display:flex;align-items:center;justify-content:center;font-size:1.3rem}
h1{font-size:1.25rem;margin:0;letter-spacing:.5px}
h1 small{display:block;font-size:.72rem;color:var(--muted);font-weight:400}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:1.1rem;margin-bottom:1rem}
.card h3{margin:0 0 .8rem;font-size:.95rem;display:flex;align-items:center;gap:.5rem}
.card h3 .dot{width:8px;height:8px;border-radius:50%;background:var(--accent)}
label{display:block;font-size:.75rem;color:var(--muted);margin:.6rem 0 .25rem;text-transform:uppercase;letter-spacing:.5px}
input,select,textarea{width:100%;background:var(--panel2);border:1px solid var(--border);color:var(--fg);
 border-radius:8px;padding:.55rem .7rem;font-size:.85rem;outline:none}
input:focus,textarea:focus{border-color:var(--accent)}
textarea{font-family:ui-monospace,monospace;font-size:.78rem;resize:vertical;min-height:110px;line-height:1.5}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:.55rem 1.1rem;
 font-size:.85rem;cursor:pointer;transition:.15s;margin:.35rem .35rem 0 0}
button:hover{filter:brightness(1.15)}
button.ghost{background:var(--panel2);border:1px solid var(--border);color:var(--fg)}
button.small{padding:.35rem .8rem;font-size:.78rem}
.task{display:flex;align-items:center;gap:.6rem;padding:.55rem .2rem;border-bottom:1px solid var(--border);font-size:.85rem}
.task:last-child{border-bottom:0}
.task b{color:var(--muted);font-weight:600;font-family:ui-monospace,monospace;font-size:.75rem}
.task .name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.badge{padding:.15rem .55rem;border-radius:99px;font-size:.7rem;font-weight:600}
.done{color:var(--ok);background:rgba(52,211,153,.12)}
.error{color:var(--err);background:rgba(248,113,113,.12)}
.running{color:var(--run);background:rgba(96,165,250,.12)}
.queued{color:var(--muted);background:rgba(139,148,158,.12)}
.tag-zh{color:var(--ok);font-size:.72rem}
.tag-no{color:var(--err);font-size:.72rem}
.eprow{display:flex;align-items:center;gap:.55rem;padding:.42rem .2rem;border-bottom:1px solid var(--border);font-size:.84rem}
.eprow input{width:auto}
.eprow .name{flex:1}
.hint{font-size:.75rem;color:var(--muted);margin-top:.4rem}
a{color:var(--run);text-decoration:none}
.empty{color:var(--muted);font-size:.82rem;padding:1rem 0;text-align:center}
</style></head><body><div class="wrap">
<header>
 <div class="logo">🎬</div>
 <h1>fancap <small>RK3588 NPU · 番剧中文字幕生成</small></h1>
</header>

<div class="grid">
<section class="card">
 <h3><span class="dot"></span>本地任务</h3>
 <label>模式</label>
 <select id="mode">
  <option value="auto">自动（识别 + 翻译）</option>
  <option value="translate">仅翻译（已有字幕）</option>
 </select>
 <label>文件（视频 / 音频 / 字幕）</label>
 <input type="file" id="file">
 <button onclick="submit()">提交任务</button>
</section>

<section class="card">
 <h3><span class="dot"></span>Jellyfin 连接</h3>
 <label>服务器 URL</label><input id="jf_url" placeholder="http://192.168.1.x:8096">
 <label>API Key</label><input id="jf_key" type="password" placeholder="Jellyfin 控制台生成">
 <button onclick="saveJf()">保存</button>
 <button class="ghost" onclick="testJf()">测试连接</button>
 <button class="ghost" onclick="loadSeries()">浏览媒体库</button>
 <div id="jf_browse"></div>
</section>
</div>

<section class="card">
 <h3><span class="dot"></span>任务队列</h3>
 <div id="tasks"><div class="empty">加载中…</div></div>
</section>

<div class="grid">
<section class="card">
 <h3><span class="dot"></span>翻译端点</h3>
 <label>Base URL</label><input id="base_url" placeholder="https://api.deepseek.com/v1">
 <label>API Key</label><input id="api_key" type="password">
 <label>模型</label><input id="model" placeholder="deepseek-chat">
 <button onclick="saveCfg()">保存</button>
 <button class="ghost" onclick="testTrans()">测试连通性</button>
 <div class="hint">未填写 API Key 时只输出原文，不翻译。</div>
</section>

<section class="card">
 <h3><span class="dot"></span>翻译 Prompt</h3>
 <textarea id="prompt_ta" placeholder="留空使用默认提示词"></textarea>
 <div class="hint">可用变量：{src}（源语言）、{lines}（字幕行）。留空 = 默认。</div>
 <button onclick="savePrompt()">保存 Prompt</button>
 <button class="ghost small" onclick="resetPrompt()">恢复默认</button>
</section>
</div>
</div>

<script>
const $=id=>document.getElementById(id);
async function submit(){
  const f=$('file').files[0];if(!f){flash('请选择文件',true);return;}
  const fd=new FormData();
  fd.append('mode',$('mode').value);
  fd.append('video',f);fd.append('subtitle',f);
  await fetch('/api/task',{method:'POST',body:fd});
  $('file').value='';refresh();
}
async function refresh(){
  const d=await(await fetch('/api/tasks')).json();
  $('tasks').innerHTML=Object.entries(d).map(([id,t])=>{
    const st=t.status==='done'?(t.mode==='jf'?'已上传 Jellyfin':(t.stage||'完成')):t.status;
    return `<div class="task"><b>${id}</b><span class="name">${t.mode}${t.error?' — '+t.error.slice(0,60):''}</span>
      <span class="badge ${t.status}">${st}</span></div>`;}).join('')
    ||'<div class="empty">暂无任务 — 提交视频或在 Jellyfin 里勾选剧集</div>';
}
async function loadCfg(){
  const d=await(await fetch('/api/config')).json();const tr=d.translate||{};
  $('base_url').value=tr.base_url||'';$('api_key').value=tr.api_key||'';$('model').value=tr.model||'';
  const jf=d.jellyfin||{};$('jf_url').value=jf.base_url||'';$('jf_key').value=jf.api_key||'';
  const p=await(await fetch('/api/prompt')).json();
  $('prompt_ta').value=p.prompt||'';$('prompt_ta').placeholder=(p.default||'').slice(0,60)+'…';
}
async function saveCfg(){
  const d=await(await fetch('/api/config')).json();
  d.translate={base_url:$('base_url').value,api_key:$('api_key').value,model:$('model').value,prompt:$('prompt_ta').value};
  await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  flash('翻译端点已保存');
}
async function savePrompt(){
  const d=await(await fetch('/api/config')).json();
  d.translate=Object.assign(d.translate||{},{prompt:$('prompt_ta').value});
  await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  flash('Prompt 已保存');
}
async function resetPrompt(){
  const p=await(await fetch('/api/prompt')).json();
  $('prompt_ta').value=p.default;savePrompt();
}
async function testTrans(){
  await saveCfg();
  flash('测试中…');
  const r=await fetch('/api/test-translate',{method:'POST'});
  const d=await r.json();
  if(d.ok){flash(`连通 ✓ ${d.latency}s — 示例: ${d.sample}`);}
  else{flash('失败: '+d.error.slice(0,150),true);}
}
async function saveJf(){
  const d=await(await fetch('/api/config')).json();
  d.jellyfin={base_url:$('jf_url').value,api_key:$('jf_key').value};
  await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  flash('Jellyfin 已保存');
}
async function testJf(){
  const r=await fetch('/api/jf/series');
  if(r.ok){flash('连接成功，找到 '+((await r.json()).length)+' 个剧库');}
  else{flash('连接失败: '+(await r.text()).slice(0,120),true);}
}
let jfSeries=[];
async function loadSeries(){
  const r=await fetch('/api/jf/series');if(!r.ok)return;
  jfSeries=await r.json();
  $('jf_browse').innerHTML=jfSeries.map(s=>
    `<div class="task"><span class="name"><a href="#" onclick="loadEps('${s.id}','${(s.name||'').replace(/'/g,'')}');return false">${s.name} ${s.year||''}</a></span></div>`).join('')
    ||'<div class="empty">无剧库</div>';
}
async function loadEps(sid,name){
  const r=await fetch('/api/jf/episodes/'+sid);const eps=await r.json();
  $('jf_browse').innerHTML=`<div class="task"><span class="name"><b>${name}</b></span>
    <button class="ghost small" onclick="loadSeries()">返回</button>
    <button class="small" onclick="queueSel()">处理选中</button></div>`+
   eps.map(e=>`<div class="eprow">
     <input type="checkbox" class="epcb" data-id="${e.id}" data-name="${(e.name||'').replace(/'/g,'')}" ${e.has_zh?'':'checked'}>
     <span class="name">${e.ep!==undefined?('第'+e.ep+'集 · '):''}${e.name}</span>
     ${e.has_zh?'<span class="tag-zh">✓ 有中文</span>':'<span class="tag-no">无中文</span>'}
   </div>`).join('');
}
async function queueSel(){
  const items=[...document.querySelectorAll('.epcb:checked')].map(c=>({id:c.dataset.id,name:c.dataset.name}));
  if(!items.length){flash('未勾选任何集',true);return;}
  const r=await fetch('/api/jf/queue',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(items)});
  if(r.ok){flash('已入队 '+items.length+' 集');refresh();}
}
function flash(msg,err){
  const b=document.createElement('div');
  b.style.cssText='position:fixed;bottom:1.2rem;left:50%;transform:translateX(-50%);'+
   'padding:.6rem 1.2rem;border-radius:10px;font-size:.85rem;z-index:99;'+
   (err?'background:#7f1d1d;color:#fecaca':'background:#14532d;color:#bbf7d0');
  b.textContent=msg;document.body.appendChild(b);setTimeout(()=>b.remove(),2600);
}
refresh();loadCfg();setInterval(refresh,2000);
</script></body></html>"""
