# vast-whisper-worker

faster-whisper FastAPI worker for vast.ai GPU instances. Built by the
`build.yml` workflow and published to `ghcr.io/echennells/vast-whisper:latest`.

Controller lives in the n8n server's ytdlp-sidecar and routes `timestamps:true`
transcription jobs here when a warm GPU worker exists.

## Run locally (CPU, for testing only)

    docker run --rm -p 8000:8000 \
      -e WHISPER_DEVICE=cpu \
      -e WHISPER_COMPUTE=int8 \
      -e WORKER_AUTH_TOKEN=dev \
      ghcr.io/echennells/vast-whisper:latest

## API

`GET /healthz` — liveness, returns model metadata.

`POST /transcribe` — multipart audio upload. Returns `{segments:[{start,end,text,words:[...]}], language, duration, ...}`.
