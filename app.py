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
import tempfile
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import uvicorn

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

    model = _load_model()

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

    t0 = time.time()
    try:
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

        return JSONResponse({
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
        })
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
