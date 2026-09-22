"""One-line GPU report for groove-start; exits non-zero when CUDA is unusable.

Also names the engine BACKEND=auto would pick on this card (docs/GGUF_ENGINE.md §3), so the
container log and the pod launcher show it before the first generation."""

import os
import sys

import torch

from yue2_groove import gguf_engine

if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False — no NVIDIA GPU visible to the container")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
flash = torch.backends.cuda.is_flash_attention_available() and cap >= (8, 0)
backend = (os.environ.get("YUE2_GROOVE_BACKEND") or "auto").strip().lower()
if backend == "auto":  # the app's own rule, on the card as the app reads it
    backend = "auto→" + gguf_engine.auto_backend("cuda", gguf_engine.cuda_total_vram_gib())[0]
print(
    f"{name} {total:.0f} GiB sm{cap[0]}{cap[1]} bf16={torch.cuda.is_bf16_supported()} "
    f"flash={flash} torch={torch.__version__} yue2.cpp={gguf_engine.engine_version() or 'none'} "
    f"backend={backend}"
)
