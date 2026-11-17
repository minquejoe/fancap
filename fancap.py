#!/usr/bin/env python3
"""fancap - anime subtitle generation tool.

Pipeline: video/audio -> ffmpeg extract -> Silero VAD segment -> SenseVoice ASR
(sherpa-onnx, RKNN on RK3588 / int8 ONNX fallback) -> translation via OpenAI-compatible
endpoint -> .srt/.ass output.

Scenes:
  1. no subtitle: full pipeline (transcribe + translate)
  2. foreign soft subtitle: extract + translate only
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass

import pysubs2
import sherpa_onnx
import soundfile as sf
import numpy as np


# ---------------------------------------------------------------- config

@dataclass
class Config:
    # ASR
    asr_backend: str = "auto"          # auto | rknn | onnx-cpu
    model_dir: str = "/home/orangepi/fancap-work/models/sherpa-onnx-rk3588-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    num_threads: int = 4
    # translation (OpenAI-compatible endpoint)
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    concurrent: int = 8
    prompt: str = ""                   # custom prompt template; empty = default
    # VAD
    vad_threshold: float = 0.25
    max_seg_sec: float = 9.5           # RKNN fixed 10s input, keep margin
    min_seg_sec: float = 0.2


# ---------------------------------------------------------------- audio

def extract_audio(video_path: str, out_wav: str) -> None:
    """Extract 16k mono wav from any video/audio file or HTTP URL."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000",
         "-vn", "-f", "wav", out_wav],
        check=True, capture_output=True,
    )


def load_audio(wav_path: str):
    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000, f"expect 16k, got {sr}"
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


# ---------------------------------------------------------------- VAD

def segments_by_vad(audio: np.ndarray, cfg: Config):
    """Silero VAD via sherpa-onnx; split at speech boundaries, hard-cap at max_seg_sec."""
    vad = sherpa_onnx.VoiceActivityDetector(
        sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(_find_vad_model()),
                threshold=cfg.vad_threshold,
                min_silence_duration=0.25,
                min_speech_duration=0.25,
                max_speech_duration=cfg.max_seg_sec,
            ),
            sample_rate=16000,
            num_threads=1,
        )
    )
    sample_rate = 16000
    window = 512
    segments = []
    # streaming: feed windows, drain queue as segments appear, flush at end
    pos = 0
    while pos < len(audio):
        chunk = audio[pos:pos + window]
        vad.accept_waveform(chunk)
        while not vad.empty():
            segments.extend(_take_segment(vad, sample_rate, cfg))
        pos += window
    vad.flush()
    while not vad.empty():
        segments.extend(_take_segment(vad, sample_rate, cfg))
    return segments


def _take_segment(vad, sample_rate: int, cfg: Config):
    """Pop one VAD segment, split over-long ones on silence-free boundaries."""
    seg = vad.front
    start_s = max(0.0, seg.start / sample_rate)
    end_s = start_s + len(seg.samples) / sample_rate
    vad.pop()
    out = []
    while end_s - start_s > cfg.max_seg_sec:
        cut = start_s + cfg.max_seg_sec
        out.append((start_s, cut))
        start_s = cut
    out.append((start_s, end_s))
    return out


def _find_vad_model() -> str:
    for cand in (
        os.path.expanduser("~/.cache/sherpa-onnx/silero_vad/silero_vad.onnx"),
        "silero_vad.onnx",
        "/home/orangepi/fancap-work/models/silero_vad.onnx",
    ):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("silero_vad.onnx not found; download from sherpa-onnx releases")


# ---------------------------------------------------------------- ASR

def build_recognizer(cfg: Config):
    """SenseVoice recognizer. RKNN via provider='rknn' when model.rknn present, else int8 onnx."""
    d = cfg.model_dir
    rknn = os.path.join(d, "model.rknn")
    onnx_i8 = os.path.join(d, "model.int8.onnx")
    tokens = os.path.join(d, "tokens.txt")
    if cfg.asr_backend in ("auto", "rknn") and os.path.exists(rknn):
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=rknn, tokens=tokens, provider="rknn",
            num_threads=1,  # RKNN: 1 = all 3 cores auto
        ), "rknn-npu"
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=onnx_i8, tokens=tokens, num_threads=cfg.num_threads,
    ), "cpu-onnx"


def recognize_segments(segments, audio, cfg: Config):
    """Return [(start, end, text)] using CPU onnx recognizer (RKNN runner separate)."""
    recognizer, backend = build_recognizer(cfg)
    stream = recognizer.create_stream()
    results = []
    for (s, e) in segments:
        a0, a1 = int(s * 16000), int(e * 16000)
        # SenseVoice onnx expects <=10s; our segments are capped
        stream.accept_waveform(16000, audio[a0:a1])
        recognizer.decode_stream(stream)
        results.append((s, e, stream.result.text if hasattr(stream.result, "text") else str(stream.result)))
        stream = recognizer.create_stream()
    return results, backend


# ---------------------------------------------------------------- translation

DEFAULT_TRANSLATE_PROMPT = """你是番剧字幕翻译。把下面的{src}字幕翻译成简体中文。
要求：
- 逐条翻译，保持行数一致，每行一条
- 保持口语化、符合番剧语气，人称/口癖前后一致
- 只输出翻译结果，不要解释

字幕：
{lines}
"""


def translate_lines(lines: list[str], src_lang: str, cfg: Config) -> list[str]:
    """Batch translate via OpenAI-compatible /v1/chat/completions. No api_key -> passthrough."""
    if not cfg.api_key:
        return list(lines)
    from openai import OpenAI
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    out = []
    B = 40  # lines per batch
    template = cfg.prompt.strip() if (cfg.prompt or "").strip() else DEFAULT_TRANSLATE_PROMPT
    for i in range(0, len(lines), B):
        batch = lines[i:i + B]
        prompt = template.format(src=src_lang, lines="\n".join(batch))
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        translated = resp.choices[0].message.content.strip().splitlines()
        if len(translated) != len(batch):
            # pad/truncate defensively
            translated = (translated + [""] * len(batch))[:len(batch)]
        out.extend(translated)
    return out


# ---------------------------------------------------------------- subtitle IO

def write_srt(segments_text, path: str) -> None:
    subs = pysubs2.SSAFile()
    for (s, e, text) in segments_text:
        subs.events.append(pysubs2.SSAEvent(
            start=int(s * 1000), end=int(e * 1000), text=text))
    subs.save(path, format_="srt")


def write_ass(segments_text, path: str) -> None:
    subs = pysubs2.SSAFile()
    style = pysubs2.SSAStyle()
    style.fontname = "Noto Sans CJK SC"
    style.fontsize = 52
    style.outline = 2
    style.shadow = 1
    subs.styles["Default"] = style
    for (s, e, text) in segments_text:
        subs.events.append(pysubs2.SSAEvent(
            start=int(s * 1000), end=int(e * 1000), text=text, style="Default"))
    subs.save(path, format_="ass")


def read_subtitle(path: str):
    """Read existing soft subtitle, return [(start, end, text)] and pysubs2 object."""
    subs = pysubs2.load(path)
    out = []
    for ev in subs.events:
        if ev.is_comment:
            continue
        out.append((ev.start / 1000, ev.end / 1000, ev.plaintext))
    return out, subs


# ---------------------------------------------------------------- pipeline

def pipeline_transcribe(video: str, out_prefix: str, cfg: Config, log=print):
    log("[1/4] 提取音频…")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    tmp = tempfile.mktemp(suffix=".wav")
    extract_audio(video, tmp)
    audio = load_audio(tmp)
    log(f"    时长 {len(audio)/16000:.1f}s")

    log("[2/4] VAD 切分…")
    segs = segments_by_vad(audio, cfg)
    log(f"    {len(segs)} 段")

    log("[3/4] ASR 识别…")
    results, backend = recognize_segments(segs, audio, cfg)
    log(f"    后端 {backend}")

    src_lang = "日语" if _looks_ja(results) else "外语"
    log(f"[4/4] 翻译（{src_lang}→中文）…")
    texts = [t for _, _, t in results]
    zh = translate_lines(texts, src_lang, cfg)
    final = [(s, e, z) for (s, e, _), z in zip(results, zh)]

    write_srt(final, out_prefix + ".srt")
    write_ass(final, out_prefix + ".ass")
    os.remove(tmp)
    log(f"完成: {out_prefix}.srt / .ass")
    return final


def pipeline_translate(subtitle_in: str, out_prefix: str, cfg: Config, log=print):
    segs, subs = read_subtitle(subtitle_in)
    texts = [t for _, _, t in segs]
    src_lang = "英语" if _looks_en(texts) else "外语"
    log(f"翻译 {len(texts)} 条（{src_lang}→中文）…")
    zh = translate_lines(texts, src_lang, cfg)
    final = [(s, e, z) for (s, e, _), z in zip(segs, zh)]
    # keep original styles: clone input file, replace text
    for ev, z in zip(subs.events, zh):
        ev.text = z
    subs.save(out_prefix + ".srt")
    subs.save(out_prefix + ".ass")
    log(f"完成: {out_prefix}.srt / .ass")
    return final


def pipeline_jf(jf_url: str, jf_key: str, item_id: str, ep_name: str, cfg: Config, log=print):
    """Jellyfin episode: stream audio -> NPU ASR -> translate -> upload srt back."""
    from jellyfin import Jellyfin
    jf = Jellyfin(jf_url, jf_key)
    log(f"[JF] 处理: {ep_name}")
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        log("[1/4] 流式抽取音频…")
        extract_audio(jf.stream_url(item_id), tmp)
        audio = load_audio(tmp)
        log(f"    时长 {len(audio)/16000/60:.1f} 分钟")

        log("[2/4] VAD 切分…")
        segs = segments_by_vad(audio, cfg)
        log(f"    {len(segs)} 段")

        log("[3/4] ASR 识别…")
        results, backend = recognize_segments(segs, audio, cfg)
        log(f"    后端 {backend}")
        src_lang = "日语" if _looks_ja(results) else "外语"
        log(f"[4/4] 翻译（{src_lang}→中文）并上传…")
        texts = [t for _, _, t in results]
        zh = translate_lines(texts, src_lang, cfg)
        final = [(s, e, z) for (s, e, _), z in zip(results, zh)]
        srt_path = tempfile.mktemp(suffix=".srt")
        write_srt(final, srt_path)
        srt_text = open(srt_path, encoding="utf-8").read()
        os.remove(srt_path)
        if not cfg.api_key:
            log("    未配置翻译端点，上传原文")
        else:
            log(f"    {len(zh)} 条已翻译")
        status = jf.upload_subtitle(item_id, srt_text)
        log(f"    已上传 Jellyfin (HTTP {status})")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True


def pipeline_sub_search(query: str, out_prefix: str, cfg: Config, log=print,
                        upload_to_jf: str = None):
    """Search subtitle sites for the query, download best Chinese/foreign sub,
    translate if needed, write srt/ass. Optionally upload to a Jellyfin item."""
    from sources import search_all, subtitlecat_download, looks_chinese_content
    log(f"[SS] 搜索字幕: {query}")
    results = search_all(query, limit=10)
    if not results:
        raise RuntimeError("没有找到匹配的字幕")
    # prefer Chinese subs first, then any
    zh = [r for r in results if r.get("lang") == "zh" or is_chinese_subtitle(r["title"])]
    ordered = zh + [r for r in results if r not in zh]
    log(f"    {len(results)} 条结果，中文 {len(zh)} 条")

    from sources import subtitlecat_download as _dl
    raw = None
    chosen = None
    for cand in ordered:
        if cand.get("source") != "subtitlecat":
            continue  # assrt needs token; skip when unconfigured
        try:
            raw, dl_url = _dl(cand["url"])
            chosen = cand
            log(f"    下载: {cand['title']}")
            break
        except Exception as e:
            log(f"    跳过 {cand['title'][:40]}: {str(e)[:60]}")
    if raw is None:
        raise RuntimeError("所有候选字幕下载失败（assrt 需要 token 配置）")

    import pysubs2
    import tempfile
    tmp = tempfile.mktemp(suffix=".srt")
    with open(tmp, "wb") as f:
        f.write(raw)
    try:
        subs = pysubs2.load(tmp, encoding="utf-8")
    except Exception:
        subs = pysubs2.load(tmp, encoding="gbk")
    segs = [(e.start / 1000, e.end / 1000, e.plaintext)
            for e in subs.events if not e.is_comment]
    log(f"    {len(segs)} 条字幕")

    content = "\n".join(t for _, _, t in segs)
    if looks_chinese_content(content):
        log("    已是中文字幕，直接使用")
    else:
        src_lang = "日语" if _looks_ja([(0, 0, t) for t in content.splitlines()]) else "外语"
        log(f"    外语字幕，翻译（{src_lang}→中文）…")
        zh_texts = translate_lines([t for _, _, t in segs], src_lang, cfg)
        segs = [(s, e, z) for (s, e, _), z in zip(segs, zh_texts)]

    write_srt(segs, out_prefix + ".srt")
    write_ass(segs, out_prefix + ".ass")
    os.remove(tmp)
    log(f"完成: {out_prefix}.srt / .ass")

    if upload_to_jf:
        from jellyfin import Jellyfin
        jf = Jellyfin(upload_to_jf[0], upload_to_jf[1])
        srt_text = open(out_prefix + ".srt", encoding="utf-8").read()
        st = jf.upload_subtitle(upload_to_jf[2], srt_text)
        log(f"    已上传 Jellyfin (HTTP {st})")
    return segs


def _looks_ja(results) -> bool:
    import re
    text = " ".join(t for _, _, t in results)
    return bool(re.search(r'[ぁ-んァ-ヶ一]', text))


def _looks_en(texts) -> bool:
    import re
    text = " ".join(texts)
    ascii_ratio = sum(c.isascii() for c in text) / max(1, len(text))
    return ascii_ratio > 0.8


if __name__ == "__main__":
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["transcribe", "translate"])
    ap.add_argument("input")
    ap.add_argument("out_prefix")
    ap.add_argument("--config", default="~/.config/fancap/config.yaml")
    args = ap.parse_args()
    cfg = Config()
    p = os.path.expanduser(args.config)
    if os.path.exists(p):
        with open(p) as f:
            d = yaml.safe_load(f)
        for k, v in (d.get("translate") or {}).items():
            setattr(cfg, k, v)
        cfg.asr_backend = d.get("asr", {}).get("backend", cfg.asr_backend)
    if args.mode == "transcribe":
        pipeline_transcribe(args.input, args.out_prefix, cfg)
    else:
        pipeline_translate(args.input, args.out_prefix, cfg)
