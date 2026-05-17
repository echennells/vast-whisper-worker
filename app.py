"""FastAPI service for faster-whisper on vast.ai GPU instances.

Exposes POST /transcribe accepting multipart audio. Returns timestamped segments
with optional word-level timings. Model is loaded once at startup and held in
GPU memory.

Auth: Bearer token in Authorization header, compared against WORKER_AUTH_TOKEN
env var. If unset, auth is disabled (intended for local dev only).

Env vars:
  WORKER_AUTH_TOKEN   required in production — shared secret with controller
  WHISPER_MODEL       default: large-v3-turbo
  WHISPER_DEVICE      default: cuda
  WHISPER_COMPUTE     default: int8_float16 (best speed/quality on 3090+)
  WHISPER_VAD         default: 1 (enable voice activity detection)
  BEAM_SIZE           default: 5
  WORD_TIMESTAMPS     default: 1
"""
import os
import subprocess
import tempfile
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn


YTDLP_TIMEOUT = int(os.environ.get("YTDLP_TIMEOUT", "900"))
MIN_AUDIO_BYTES = int(os.environ.get("MIN_AUDIO_BYTES", "20480"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("worker")


MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8_float16")
USE_VAD = os.environ.get("WHISPER_VAD", "1") == "1"
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "5"))
WORD_TIMESTAMPS = os.environ.get("WORD_TIMESTAMPS", "1") == "1"
AUTH_TOKEN = os.environ.get("WORKER_AUTH_TOKEN", "").strip()


_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    from faster_whisper import WhisperModel
    log.info("loading model name=%s device=%s compute=%s", MODEL_NAME, DEVICE, COMPUTE_TYPE)
    t0 = time.time()
    _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    log.info("model loaded in %.1fs", time.time() - t0)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(lifespan=lifespan, title="vast-whisper-worker", version="1.0")


def _check_auth(authorization: Optional[str]):
    if not AUTH_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if authorization[7:].strip() != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


@app.get("/healthz")
async def healthz(authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    return {
        "ok": True,
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "vad": USE_VAD,
        "model_loaded": _model is not None,
    }


def _run_whisper(tmp_path: str, language, initial_prompt, use_word_ts, use_vad, bs):
    """Shared whisper core. Returns the JSON dict for the HTTP response."""
    model = _load_model()
    t0 = time.time()
    segments_iter, info = model.transcribe(
        tmp_path,
        language=language,
        beam_size=bs,
        vad_filter=use_vad,
        vad_parameters={"min_silence_duration_ms": 500} if use_vad else None,
        word_timestamps=use_word_ts,
        initial_prompt=initial_prompt,
        condition_on_previous_text=False,
    )

    segments = []
    for seg in segments_iter:
        s = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": (seg.text or "").strip(),
        }
        if use_word_ts and seg.words:
            s["words"] = [
                {
                    "start": float(w.start) if w.start is not None else None,
                    "end": float(w.end) if w.end is not None else None,
                    "word": w.word,
                    "probability": float(w.probability) if w.probability is not None else None,
                }
                for w in seg.words
            ]
        segments.append(s)

    elapsed = time.time() - t0
    text = " ".join(s["text"] for s in segments).strip()

    log.info("done segs=%d elapsed=%.2fs audio_dur=%.2fs realtime=%.1fx",
             len(segments), elapsed, info.duration, info.duration / max(elapsed, 0.01))

    return {
        "ok": True,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "transcription_time": elapsed,
        "realtime_factor": info.duration / max(elapsed, 0.01),
        "model": MODEL_NAME,
        "compute_type": COMPUTE_TYPE,
        "vad_used": use_vad,
        "word_timestamps": use_word_ts,
        "text": text,
        "segments": segments,
    }


@app.post("/transcribe")
async def transcribe(
    request: Request,
    authorization: Optional[str] = Header(None),
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
    word_timestamps: Optional[str] = Form(None),
    vad: Optional[str] = Form(None),
    beam_size: Optional[int] = Form(None),
):
    _check_auth(authorization)

    use_word_ts = WORD_TIMESTAMPS if word_timestamps is None else (word_timestamps == "1")
    use_vad = USE_VAD if vad is None else (vad == "1")
    bs = beam_size or BEAM_SIZE

    suffix = os.path.splitext(audio.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        total = 0
        while True:
            chunk = await audio.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
            total += len(chunk)

    log.info("transcribe size=%d file=%s lang=%s vad=%s word_ts=%s beam=%d",
             total, audio.filename, language, use_vad, use_word_ts, bs)

    try:
        return JSONResponse(_run_whisper(tmp_path, language, initial_prompt, use_word_ts, use_vad, bs))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class TranscribeUrlReq(BaseModel):
    url: str
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    language: Optional[str] = None
    initial_prompt: Optional[str] = None
    word_timestamps: Optional[bool] = None
    vad: Optional[bool] = None
    beam_size: Optional[int] = None
    proxy_url: Optional[str] = None


def _ytdlp_fetch(url: str, start: Optional[float], end: Optional[float],
                 dest_dir: str, proxy_url: Optional[str]) -> str:
    """Download audio (optionally a [start,end] range) into dest_dir via yt-dlp.
    Returns the audio file path. Raises RuntimeError on failure.
    """
    out_template = os.path.join(dest_dir, "out.%(ext)s")
    cmd = ["yt-dlp", "--no-warnings"]
    if proxy_url:
        cmd += ["--proxy", proxy_url]
    cmd += [
        "-f", "140/ba[ext=m4a]/bestaudio[ext=m4a]/bestaudio",
        "--extractor-args", "youtube:player_client=android_vr,web_safari,tv_embedded,mweb",
        "--socket-timeout", "60",
        "--retries", "5",
        "--fragment-retries", "10",
        "-o", out_template,
        "--no-progress",
    ]
    if start is not None and end is not None:
        cmd += ["--download-sections", f"*{int(start)}-{int(end)}"]
    cmd.append(url)

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=YTDLP_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"yt-dlp timeout after {YTDLP_TIMEOUT}s")
    if r.returncode != 0:
        err = (r.stderr or b"").decode("utf-8", "replace").strip()[:800]
        raise RuntimeError(f"yt-dlp failed: {err}")

    files = [f for f in os.listdir(dest_dir) if f.startswith("out.")]
    if not files:
        raise RuntimeError("yt-dlp produced no output file")
    path = os.path.join(dest_dir, files[0])
    size = os.path.getsize(path)
    if size < MIN_AUDIO_BYTES:
        raise RuntimeError(f"audio too small: {size} bytes (likely truncated)")
    return path


@app.post("/transcribe-url")
async def transcribe_url(
    req: TranscribeUrlReq,
    authorization: Optional[str] = Header(None),
):
    """Pull audio (optionally a [start,end] range) directly via yt-dlp on this
    GPU box, then transcribe. Removes the load on the orchestrator and
    parallelizes YouTube fetches across worker IPs.
    """
    _check_auth(authorization)

    use_word_ts = WORD_TIMESTAMPS if req.word_timestamps is None else bool(req.word_timestamps)
    use_vad = USE_VAD if req.vad is None else bool(req.vad)
    bs = req.beam_size or BEAM_SIZE

    log.info("transcribe-url url=%s start=%s end=%s lang=%s vad=%s",
             req.url, req.start_seconds, req.end_seconds, req.language, use_vad)

    with tempfile.TemporaryDirectory(prefix="ytdlp_") as td:
        t_fetch = time.time()
        try:
            tmp_path = _ytdlp_fetch(req.url, req.start_seconds, req.end_seconds, td, req.proxy_url)
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail={"stage": "fetch", "error": str(e)})
        fetch_elapsed = time.time() - t_fetch
        fetch_size = os.path.getsize(tmp_path)
        log.info("fetch ok size=%d elapsed=%.1fs", fetch_size, fetch_elapsed)

        result = _run_whisper(tmp_path, req.language, req.initial_prompt, use_word_ts, use_vad, bs)
        result["fetch_seconds"] = fetch_elapsed
        result["fetch_size_bytes"] = fetch_size
        return JSONResponse(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
