"""NAR solver steps below the default: re-render a saved run's tokens at another step count.

One variable at a time (`CROSS_PLATFORM.md` §9.3, `LINUX_CUDA.md` §5): prefix, semantic tokens
and the CPU-generated noise are taken from `--source` unchanged, and only the midpoint step
count differs.  Each render is compared against

  `--baseline`  a 32-step render of the same tokens **on this machine** — isolates the step count
  `--source`    the run's own audio, i.e. the reference render — adds the platform difference

Writes `<runs>/<stamp>-rerender-nar<steps>-<source id>/` with `audio.flac`, `latent.npy` and
`rerender.json`, then finalizes it (`request.json` / `config.json` / `result.json`) so `04 //
LIBRARY` and `05 // TOOLS` accept the directory like any other run.  `--bf16-latent` additionally
reports what rounding the 32-step latent to bf16 before the fp32 VAE costs.

Measured on an M4 Pro (bf16, MPS, torch 2.14.0), 32-step baseline `…040751-rerender-nar-…`,
source run `20260915-142716-Something_True_CFG15`:

| steps | latent RMS Δ / std | latent corr | NAR | vs this machine's 32-step render |
|---|---|---|---|---|
| 32 | — | — | 629 s | reference |
| 16 | 5.6 % | 0.99842 | 323 s | 21.1 dB |
| 8 | 9.4 % | 0.99557 | 164 s | 16.6 dB |

NAR time is linear in the step count.  The bf16 latent round trip is bit-identical (0 of 455 872
values change), and its 133 dB residual against the stored render is the PCM_24 container's own
quantisation, not a decode difference.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerender import compare_audio, compare_latents, finalize, sha256

from yue2_groove import adapter, config


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "source",
        help="reference run directory: prefix.npy, semantic.npy, config.json, request.json, audio.flac",
    )
    ap.add_argument(
        "--baseline", help="32-step render of the same tokens on this machine (default: none)"
    )
    ap.add_argument(
        "--steps", default="16,8", help="step counts to render, comma separated (default: 16,8)"
    )
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument(
        "--runs",
        default=str(config.runs_dir()),
        help="run directory root (default: $YUE2_GROOVE_RUNS or ./runs, like the app)",
    )
    ap.add_argument(
        "--model", default=config.default_model(), help="local model directory or Hub id"
    )
    ap.add_argument("--vae", default=config.default_vae(), help="local decoder directory or Hub id")
    ap.add_argument(
        "--bf16-latent",
        action="store_true",
        help="also measure the baseline's 32-step latent rounded to bf16 before the VAE",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src = Path(args.source).resolve()
    base = Path(args.baseline).resolve() if args.baseline else None
    runs = Path(args.runs)
    cfg = json.loads((src / "config.json").read_text())
    req = json.loads((src / "request.json").read_text())
    src_result = (
        json.loads((src / "result.json").read_text()) if (src / "result.json").exists() else {}
    )
    gen, seed = cfg["generation"], int(req["seed"])
    steps_list = [int(s) for s in args.steps.replace(" ", "").split(",") if s]

    print(
        f"[ode] source={src.name} device={args.device} dtype={args.dtype} seed={seed} "
        f"context={gen['context']} vae_core_frames={cfg['vae_core_frames']} steps={steps_list}",
        flush=True,
    )
    if base is None:
        print("[ode] no --baseline: comparisons are against the source render only")

    model_dir = adapter.resolve_model(args.model, local_files_only=True)
    vae_dir = adapter.resolve_model(args.vae, local_files_only=True)
    if not any(model_dir.glob("*.safetensors")):
        print(
            f"[ode] no safetensors in {model_dir} — pass --model/--vae (or set YUE2_GROOVE_MODEL / "
            "YUE2_GROOVE_VAE) to a local model directory"
        )
        return 2
    pipe, dtype_loaded = adapter.load_pipeline(
        model_dir,
        vae=vae_dir,
        device=args.device,
        dtype=args.dtype,
        backend="torch",
        quantization="none",
        offload_ar=False,
        memory_budget_gib=float(cfg.get("memory_budget_gib", 24)),
        ode_steps=int(gen["ode_steps"]),
        vae_core_frames=int(cfg["vae_core_frames"]),
        revision="",
        vae_revision="",
        local_files_only=True,
    )

    # The weights must be the ones that produced the source run, or the render is not comparable.
    for name in ("mot", "vae"):
        want = (
            (((src_result.get("weights") or {}).get(name) or {}).get("files") or {})
            .get("model.safetensors", {})
            .get("sha256")
        )
        have = (
            ((pipe.weights.get(name) or {}).get("files") or {})
            .get("model.safetensors", {})
            .get("sha256")
        )
        if want and have and want != have:
            print(f"[ode] ABORT: {name} weights differ (source {want[:16]}…, local {have[:16]}…)")
            return 2

    prefix = np.load(src / "prefix.npy", allow_pickle=False).tolist()
    codec = np.load(src / "semantic.npy", allow_pickle=False).tolist()
    src_audio, sr = sf.read(src / "audio.flac", dtype="float32", always_2d=True)
    base_lat = np.load(base / "latent.npy", allow_pickle=False) if base is not None else None
    base_audio = (
        sf.read(base / "audio.flac", dtype="float32", always_2d=True)[0]
        if base is not None
        else None
    )

    def report(name: str, lat: np.ndarray, audio: np.ndarray) -> dict:
        """Compare one render against the source and (when given) the same-machine baseline."""
        out = {}
        line = f"[ode] {name:26s}"
        if base_audio is not None:
            out["vs_baseline"] = compare_audio(base_audio, audio, sr)
            line += f" vs baseline {out['vs_baseline']['snr_db']:5.1f} dB"
        out["vs_source"] = compare_audio(src_audio, audio, sr)
        line += f" | vs source {out['vs_source']['snr_db']:5.1f} dB"
        if base_lat is not None:
            out["latent_vs_baseline"] = compare_latents(base_lat, lat)
            lat_vs = out["latent_vs_baseline"]
            line += (
                f" | latent Δ/std {lat_vs['rms_delta_over_ref_std'] * 100:.1f} %"
                f" corr {lat_vs['correlation']:.5f}"
            )
        line += (
            f" | centroid {out['vs_source']['new']['centroid_hz']:.0f} Hz"
            f" >8k {out['vs_source']['new']['magnitude_above_8k_pct']:.2f} %"
        )
        print(line, flush=True)
        return out

    from yue2.nar import synthesize

    for steps in steps_list:
        model = pipe._load_model(
            for_nar=True
        )  # pipe.decode() parks the model on the CPU after each decode
        t0 = time.perf_counter()
        lat = (
            synthesize(
                model,
                prefix,
                codec,
                seed,
                steps=steps,
                context=int(gen["context"]),
                attention="sdpa",
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        t_nar = time.perf_counter() - t0
        audio = adapter.decode(pipe, lat, full=False)
        print(f"[ode] {steps} steps: NAR {t_nar:.0f}s", flush=True)
        comparison = report(f"NAR {steps} steps", lat, audio)

        out = (
            runs
            / f"{time.strftime('%Y%m%d-%H%M%S')}-rerender-nar{steps}-{src.name.split('-', 2)[-1]}"
        )
        out.mkdir(parents=True, exist_ok=False)
        sf.write(out / "audio.flac", audio, 48000, subtype="PCM_24")
        np.save(out / "latent.npy", lat.astype(np.float32))
        (out / "rerender.json").write_text(
            json.dumps(
                {
                    "operation": "rerender_nar",
                    "stage_label": f"nar{steps}",
                    "source_run": str(src),
                    "source_id": src.name,
                    "source_device": cfg.get("device"),
                    "source_dtype": cfg.get("model_dtype"),
                    "source_inputs_sha256": {
                        "prefix.npy": sha256(src / "prefix.npy"),
                        "semantic.npy": sha256(src / "semantic.npy"),
                    },
                    "source_audio_sha256": sha256(src / "audio.flac"),
                    "seed": seed,
                    "ode_steps": steps,
                    "context": gen["context"],
                    "vae_core_frames": cfg["vae_core_frames"],
                    "vae_decode": "halo_crop",
                    "device": str(pipe.device),
                    "dtype": dtype_loaded,
                    "torch": torch.__version__,
                    "yue2": adapter.yue2_version(),
                    "machine": platform.machine(),
                    "weights": pipe.weights,
                    "runtime_sha256": getattr(pipe, "runtime_sha256", None),
                    "timing": {"nar_seconds": t_nar},
                    "outputs_sha256": {
                        "audio.flac": sha256(out / "audio.flac"),
                        "latent.npy": sha256(out / "latent.npy"),
                    },
                    "comparison_vs_source": {"audio": comparison["vs_source"]},
                    "comparison_vs_baseline": {
                        "audio": comparison.get("vs_baseline"),
                        "latent": comparison.get("latent_vs_baseline"),
                    },
                },
                indent=2,
            )
            + "\n"
        )
        (out / "local_env.json").write_text(
            json.dumps(
                {
                    "tool": "yue2_groove",
                    "dtype": dtype_loaded,
                    "device": str(pipe.device),
                    "torch": torch.__version__,
                    "yue2": adapter.yue2_version(),
                    "note": f"rerender nar{steps} of {src.name} (see rerender.json)",
                },
                indent=2,
            )
            + "\n"
        )
        finalize(out)
        print(f"[ode] wrote {out}\n", flush=True)

    if args.bf16_latent:
        if base_lat is None:
            print("[ode] --bf16-latent needs --baseline (the 32-step latent to round)")
        else:
            lat_bf16 = torch.as_tensor(base_lat).to(torch.bfloat16).float().numpy()
            print(
                f"[ode] 32-step latent -> bf16 -> fp32: {int((lat_bf16 != base_lat).sum())} of {base_lat.size} "
                "values changed",
                flush=True,
            )
            report(
                "32 steps, latent->bf16->VAE", lat_bf16, adapter.decode(pipe, lat_bf16, full=False)
            )

    pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
