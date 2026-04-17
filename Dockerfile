# syntax=docker/dockerfile:1.6
#
# vast-whisper worker: faster-whisper + FastAPI, model pre-baked for
# sub-10s startup on GPU host once image is pulled.
#
# Base: CUDA 12.2 runtime + cuDNN8 (faster-whisper / CTranslate2 needs cuDNN).
# We intentionally use -runtime (not -devel) for size. CTranslate2 ships its
# own CUDA kernels so we don't need nvcc.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# CTranslate2 CPU prefetch wheel (only needed during build for prefetch_model.py
# to load the model once to warm the HF cache; runtime uses GPU).
RUN python3 -m pip install --upgrade pip wheel setuptools

# Keep deps minimal. faster-whisper pulls CTranslate2 + tokenizers + HF hub.
RUN python3 -m pip install \
        faster-whisper==1.0.3 \
        fastapi==0.115.2 \
        uvicorn[standard]==0.31.1 \
        python-multipart==0.0.12

# Pre-download the model into /models so no network is needed at runtime.
# Use int8 compute for the prefetch load (CPU); the actual runtime will load
# with int8_float16 on GPU and CTranslate2 will convert as needed.
COPY prefetch_model.py /opt/prefetch_model.py
RUN mkdir -p /models && \
    WHISPER_MODEL=large-v3-turbo python3 /opt/prefetch_model.py

COPY app.py /opt/app.py

EXPOSE 8000

# Health check hits /healthz (no auth header = fine if WORKER_AUTH_TOKEN unset
# at runtime; with token, liveness still returns 401 which curl treats as
# responsive — use --fail-with-body or accept-any-code).
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/healthz | grep -qE "^(200|401)$" || exit 1

CMD ["python3", "/opt/app.py"]
