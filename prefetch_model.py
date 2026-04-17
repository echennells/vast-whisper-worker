"""Download the faster-whisper model + Silero VAD during image build so the
container can start without network access. This prevents per-boot model
downloads and makes cold-start deterministic."""
import os
import sys

from faster_whisper import WhisperModel

model_name = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
compute = os.environ.get("WHISPER_COMPUTE", "int8_float16")

print(f"prefetching model={model_name} compute={compute}", flush=True)

m = WhisperModel(model_name, device="cpu", compute_type="int8")
print("model cached", flush=True)

try:
    import torch
    torch.hub.set_dir(os.environ.get("TORCH_HOME", "/root/.cache/torch"))
except Exception:
    pass

try:
    from faster_whisper.vad import get_vad_model
    get_vad_model()
    print("vad cached", flush=True)
except Exception as e:
    print(f"vad prefetch skipped: {e}", flush=True, file=sys.stderr)

print("prefetch complete", flush=True)
