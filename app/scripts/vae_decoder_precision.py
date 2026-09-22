"""What half-precision decoding costs: the fp32 VAE decode as baseline, two ways of going lower.

The official configuration decodes audio in fp32 and upstream enforces it —
`YuE2VAE._latent` raises `VAE decoder weights must remain FP32`, so a port that stores the VAE in
half precision has to bypass that guard on purpose.  This script measures what it buys:

  `--mode weights`   bf16/fp16 **weights + activations** (the guard bypassed for measurement only)
  `--mode autocast`  fp32 weights, convolutions autocast to bf16/fp16

Both decode the *same* latent (`--source/latent.npy`) with the same tiling as a run, and report
SNR against the fp32 decode of that latent, plus the spectral side effects.  Every variant loads a
**fresh VAE instance**: a half-precision run leaves state behind that shifts a later fp32-weight
autocast run by ~2 dB (38.6 instead of 40.9 for bf16, 43.2 instead of 59.1 for fp16), and re-typing
weights that a previous half-precision pass already rounded reads fp16 as 43.4 dB instead of
54.3 dB.  One instance per row is what makes the numbers reproducible.

Numbers printed for `20260915-142716-Something_True_CFG15/latent.npy` on an M4 Pro (MPS, torch
2.14.0, standard `YuE2-Vae`, core 1024 / halo 16):

| mode | bf16 | fp16 |
|---|---|---|
| weights + activations | 35.3 dB | 54.3 dB |
| autocast (fp32 weights) | 40.9 dB | 59.1 dB |

For scale: a different PyTorch build of the fp32 stack moves the same render 119 dB, a bf16
decoder 35.3 dB, and the model's own 32 → 48 solver-step change 28.0 dB (`CROSS_PLATFORM.md` §9.3).
The latents themselves are not the lossy part — they already carry bf16 precision, so rounding
them costs nothing (`scripts/ode_steps.py --bf16-latent`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from yue2.modeling_vae import YuE2VAE

from yue2_groove import adapter


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("source", help="run directory holding the latent.npy to decode")
    ap.add_argument(
        "--vae",
        default="m-a-p/YuE2-Vae",
        help="VAE directory or Hub id (default: standard YuE2-Vae)",
    )
    ap.add_argument("--device", default="mps")
    ap.add_argument("--mode", default="both", choices=["weights", "autocast", "both"])
    ap.add_argument("--core-frames", type=int, default=1024)
    ap.add_argument("--halo-frames", type=int, default=16)
    ap.add_argument(
        "--out",
        default="vae_decoder_precision.json",
        help="where to write the JSON summary (default: ./vae_decoder_precision.json)",
    )
    return ap.parse_args(argv)


def spectral(x: np.ndarray) -> tuple[float, float]:
    m = x.mean(1)
    s = np.abs(np.fft.rfft(m))
    f = np.fft.rfftfreq(len(m), 1 / 48000)
    return float((f * s).sum() / s.sum()), float(s[f > 8000].sum() / s.sum() * 100)


def compare(ref: np.ndarray, new: np.ndarray) -> dict:
    n = min(len(ref), len(new))
    d = new[:n] - ref[:n]
    rms_ref, rms_d = np.sqrt((ref[:n] ** 2).mean()), np.sqrt((d**2).mean())
    centroid, above8k = spectral(new[:n])
    return {
        "max_abs_delta": float(np.abs(d).max()),
        "rms_delta": float(rms_d),
        "snr_db": float(20 * np.log10(rms_ref / rms_d)) if rms_d > 0 else float("inf"),
        "centroid_hz": centroid,
        "magnitude_above_8k_pct": above8k,
        "nonfinite": int((~np.isfinite(new)).sum()),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    src = Path(args.source).resolve()
    device = torch.device(args.device)
    vae_dir = adapter.resolve_model(args.vae, local_files_only=True)
    z = torch.as_tensor(
        np.load(src / "latent.npy", allow_pickle=False), dtype=torch.float32
    ).T.unsqueeze(0)

    def decode(vae: YuE2VAE) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            audio = vae.decode_tiled(
                z, core_frames=args.core_frames, halo_frames=args.halo_frames, output_device="cpu"
            )
        if device.type == "mps":
            torch.mps.synchronize()
        return audio[0].float().clamp(-1, 1).T.contiguous().numpy(), time.perf_counter() - t0

    def load(mode: str | None, dtype: torch.dtype) -> YuE2VAE:
        """A fresh fp32 VAE with one measurement patch installed (None = untouched upstream path)."""
        vae = YuE2VAE.from_pretrained(
            vae_dir, decoder_only=True, device=str(device), local_files_only=True
        )
        if mode is None or dtype == torch.float32:
            return vae
        if mode == "weights":

            def latent_nocheck(latent):
                latent = torch.as_tensor(latent)
                if latent.ndim != 3 or latent.shape[1] != vae.config.latent_dim:
                    raise ValueError("Expected nonempty [B,latent_dim,T] latents")
                if not torch.isfinite(latent).all():
                    raise ValueError("VAE latents contain non-finite values")
                return latent

            def decode_half(latent):
                value = vae._latent(latent)
                with torch.autocast(device_type=device.type, enabled=False):
                    return vae.decoder(value.to(device=device, dtype=dtype)).float()

            vae._latent = latent_nocheck  # the guard the half-precision port has to drop
            vae.decoder.to(dtype)
        else:

            def decode_half(latent):
                # The fp32 cast must stay outside the autocast context, or autocast lowers it again.
                value = vae._latent(latent).to(device=device, dtype=torch.float32)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
                    return vae.decoder(value).float()

        vae.decode = decode_half
        return vae

    ref, ref_seconds = decode(load(None, torch.float32))
    centroid, above8k = spectral(ref)
    print(
        f"[vae] fp32 reference decode: {ref_seconds:.1f}s, centroid {centroid:.1f} Hz, "
        f">8k {above8k:.3f}%  ({src.name})",
        flush=True,
    )

    results = {}
    modes = ["weights", "autocast"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for dtype in (torch.bfloat16, torch.float16):
            out, elapsed = decode(load(mode, dtype))
            stats = compare(ref, out)
            results.setdefault(mode, {})[str(dtype)[6:]] = stats
            print(
                f"[vae] {mode:8s} {str(dtype)[6:]:8s} | decode {elapsed:.1f}s | "
                f"vs fp32: max|Δ|={stats['max_abs_delta']:.3e} rmsΔ={stats['rms_delta']:.3e} "
                f"SNR={stats['snr_db']:.1f} dB | centroid {stats['centroid_hz']:.1f} Hz "
                f"(Δ{stats['centroid_hz'] - centroid:+.1f}), >8k "
                f"{stats['magnitude_above_8k_pct']:.3f}% | nonfinite={stats['nonfinite']}",
                flush=True,
            )

    out_path = Path(args.out).expanduser().resolve()
    out_path.write_text(
        json.dumps(
            {
                "source": str(src),
                "latent": "latent.npy",
                "vae": str(vae_dir),
                "device": str(device),
                "torch": torch.__version__,
                "core_frames": args.core_frames,
                "halo_frames": args.halo_frames,
                "fp32_reference": {
                    "centroid_hz": centroid,
                    "magnitude_above_8k_pct": above8k,
                    "seconds": ref_seconds,
                },
                "variants": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[vae] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
