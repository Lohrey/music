"""Re-render a saved run's tokens or latents on this machine (CROSS_PLATFORM.md §9.3).

  --stage vae   decode SOURCE/latent.npy with the local VAE (same tiling as the run)
  --stage nar   synthesize latents from SOURCE/prefix.npy + SOURCE/semantic.npy with the
                local model (same seed -> the same CPU-generated noise), then decode

Writes <runs>/<stamp>-rerender-<stage>-<source id>/ with audio.flac, latent.npy,
rerender.json (provenance + comparison) and local_env.json, then prints a sample-level
comparison against SOURCE/audio.flac.

The two measurements quoted in CROSS_PLATFORM.md §9.3 came from this script: the VAE-only
re-render differs from the CUDA reference by 109.8 dB, the NAR + VAE re-render by 28.6 dB
(against 28.0 dB for CUDA's own ODE 32 -> 48 change).

--finalize DIR completes a directory written by another harness (for example
scripts/ode_steps.py): it copies the source run's plan artifacts and writes request.json /
config.json / result.json out of that directory's rerender.json, so 04 // LIBRARY and the
05 // TOOLS comparison page accept it like any other run.  A `stage_label` field in
rerender.json (e.g. "nar16") names the variant in the request id and the result records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from yue2_groove import adapter, config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectral(x: np.ndarray, sr: int):
    m = x.mean(1) if x.ndim == 2 else x
    S = np.abs(np.fft.rfft(m))
    f = np.fft.rfftfreq(len(m), 1 / sr)
    return {
        "centroid_hz": float((f * S).sum() / S.sum()),
        "magnitude_above_8k_pct": float(S[f > 8000].sum() / S.sum() * 100),
        "rms_mono": float(np.sqrt((m**2).mean())),
    }


def compare_audio(ref: np.ndarray, new: np.ndarray, sr: int):
    n = min(len(ref), len(new))
    a, b = ref[:n], new[:n]
    d = a - b
    rms_a = float(np.sqrt((a**2).mean()))
    rms_d = float(np.sqrt((d**2).mean()))
    return {
        "samples_compared": int(n),
        "length_ref": len(ref),
        "length_new": len(new),
        "max_abs_delta": float(np.abs(d).max()),
        "rms_delta": rms_d,
        "snr_db": float(20 * np.log10(rms_a / rms_d)) if rms_d > 0 else float("inf"),
        "fraction_identical_samples": float((a == b).mean()),
        "ref": spectral(a, sr),
        "new": spectral(b, sr),
    }


def compare_latents(ref: np.ndarray, new: np.ndarray):
    n = min(len(ref), len(new))
    a, b = ref[:n].astype(np.float64), new[:n].astype(np.float64)
    d = a - b
    return {
        "frames_compared": int(n),
        "max_abs_delta": float(np.abs(d).max()),
        "rms_delta": float(np.sqrt((d**2).mean())),
        "rms_delta_over_ref_std": float(np.sqrt((d**2).mean()) / a.std()),
        "correlation": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
        "identical": bool(np.array_equal(ref[:n], new[:n])),
    }


COPY_FROM_SOURCE = (
    "score.abc",
    "plan.json",
    "plan_manifest.json",
    "prefix.npy",
    "semantic.npy",
    "abc_tokens.npy",
)


def finalize(out: Path) -> None:
    """Make *out* a complete run directory: request.json / config.json / result.json alongside
    the re-rendered audio, so LIBRARY and the 05 // TOOLS comparison page accept it."""
    import shutil

    meta = json.loads((out / "rerender.json").read_text())
    src = Path(meta["source_run"])
    stage = meta.get("stage_label") or meta["operation"].split("_", 1)[1]
    for name in COPY_FROM_SOURCE:
        if (src / name).exists() and not (out / name).exists():
            shutil.copyfile(src / name, out / name)
    request = json.loads((src / "request.json").read_text())
    request["id"] = f"{request.get('id') or src.name}-rerender-{stage}"
    (out / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    config = json.loads((src / "config.json").read_text())
    config["device"] = meta["device"]
    config["runtime_sha256"] = meta.get("runtime_sha256") or config.get("runtime_sha256")
    config["rerender"] = {
        "stage": stage,
        "source_run": src.name,
        "source_device": meta["source_device"],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    audio_info = sf.info(out / "audio.flac")
    artifacts = {
        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "result.json"
    }
    identity = hashlib.sha256(
        json.dumps(
            {"source": src.name, "stage": stage, "audio": artifacts["audio.flac"]["sha256"]}
        ).encode()
    ).hexdigest()
    timing = dict(meta["timing"])
    timing["e2e_seconds"] = sum(
        v for k, v in timing.items() if k.endswith("_seconds") and k != "e2e_seconds"
    )
    result = {
        "status": "complete",
        "identity": identity,
        "truncated": {"abc": False, "semantic": False},
        "sample_rate": audio_info.samplerate,
        "audio_seconds": audio_info.duration,
        "weights": meta["weights"],
        "timing": timing,
        "artifacts": artifacts,
        "rerender": {
            "stage": stage,
            "source_run": src.name,
            "source_device": meta["source_device"],
            "source_inputs_sha256": meta["source_inputs_sha256"],
            "comparison_vs_source": meta["comparison_vs_source"],
        },
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if not (out / "local_env.json").exists():
        (out / "local_env.json").write_text(
            json.dumps(
                {
                    "tool": "yue2_groove",
                    "dtype": meta.get("dtype"),
                    "device": meta.get("device"),
                    "torch": meta.get("torch"),
                    "yue2": meta.get("yue2"),
                    "note": f"rerender {stage} of {src.name} (see rerender.json)",
                },
                indent=2,
            )
            + "\n"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "source",
        help="run directory with prefix.npy / semantic.npy / latent.npy / config.json / request.json; "
        "with --finalize, an existing rerender output directory",
    )
    ap.add_argument(
        "--finalize",
        action="store_true",
        help="only write request/config/result.json into an existing output directory",
    )
    ap.add_argument("--stage", choices=["vae", "nar"])
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
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if args.finalize:
        finalize(Path(args.source).resolve())
        print(f"[rerender] finalized {args.source}")
        return 0
    if not args.stage:
        ap.error("--stage is required")

    src = Path(args.source).resolve()
    cfg = json.loads((src / "config.json").read_text())
    req = json.loads((src / "request.json").read_text())
    src_result = (
        json.loads((src / "result.json").read_text()) if (src / "result.json").exists() else {}
    )
    gen = cfg["generation"]
    seed = int(req["seed"])
    print(
        f"[rerender] source={src.name} stage={args.stage} device={args.device} dtype={args.dtype} "
        f"seed={seed} ode_steps={gen['ode_steps']} context={gen['context']} vae_core_frames={cfg['vae_core_frames']}",
        flush=True,
    )

    t_load = time.perf_counter()
    model_dir = adapter.resolve_model(args.model, local_files_only=True)
    vae_dir = adapter.resolve_model(args.vae, local_files_only=True)
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
    load_seconds = time.perf_counter() - t_load
    print(
        f"[rerender] pipeline loaded in {load_seconds:.0f}s, model dtype {dtype_loaded}", flush=True
    )

    # Weights must be the ones that produced the source run.
    src_weights = src_result.get("weights") or {}
    for name in ("mot", "vae"):
        want = (
            ((src_weights.get(name) or {}).get("files") or {})
            .get("model.safetensors", {})
            .get("sha256")
        )
        have = (
            ((pipe.weights.get(name) or {}).get("files") or {})
            .get("model.safetensors", {})
            .get("sha256")
        )
        if want and have and want != have:
            print(
                f"[rerender] ABORT: {name} weights differ (source {want[:16]}…, local {have[:16]}…)"
            )
            return 2
        print(
            f"[rerender] {name} weights {have[:16] if have else '?'}… {'== source' if want == have else '(source hash unavailable)'}",
            flush=True,
        )
    if pipe.generation_config.context != gen["context"]:
        print(
            f"[rerender] ABORT: context {pipe.generation_config.context} != source {gen['context']}"
        )
        return 2

    inputs = {}
    timing = {"load_seconds": load_seconds}
    if args.stage == "nar":
        prefix = np.load(src / "prefix.npy", allow_pickle=False).tolist()
        codec = np.load(src / "semantic.npy", allow_pickle=False).tolist()
        inputs = {
            "prefix.npy": sha256(src / "prefix.npy"),
            "semantic.npy": sha256(src / "semantic.npy"),
        }
        from yue2.nar import synthesize

        model = pipe._load_model(for_nar=True)
        last = {"t": time.perf_counter()}

        def on_progress(done, total):
            now = time.perf_counter()
            if done == total or now - last["t"] > 30:
                last["t"] = now
                print(f"[rerender] nar {done}/{total} steps ({now - t_nar:.0f}s)", flush=True)

        t_nar = time.perf_counter()
        latents = synthesize(
            model,
            prefix,
            codec,
            seed,
            steps=int(gen["ode_steps"]),
            context=int(gen["context"]),
            attention="sdpa",
            offload_ar=False,
            on_progress=on_progress,
        )
        latents = latents.detach().float().cpu().numpy()
        timing["nar_seconds"] = time.perf_counter() - t_nar
        print(f"[rerender] nar done: {latents.shape} in {timing['nar_seconds']:.0f}s", flush=True)
    else:
        latents = np.load(src / "latent.npy", allow_pickle=False).astype(np.float32)
        inputs = {"latent.npy": sha256(src / "latent.npy")}

    t_vae = time.perf_counter()
    audio = adapter.decode(pipe, latents, full=False)
    timing["vae_seconds"] = time.perf_counter() - t_vae
    print(
        f"[rerender] vae done: {len(audio) / 48000:.1f}s audio in {timing['vae_seconds']:.0f}s",
        flush=True,
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = args.label or f"rerender-{args.stage}-{src.name.split('-', 2)[-1]}"
    out = Path(args.runs) / f"{stamp}-{label}"
    out.mkdir(parents=True, exist_ok=False)
    sf.write(out / "audio.flac", audio, 48000, subtype="PCM_24")
    np.save(out / "latent.npy", latents.astype(np.float32))

    comparison = {}
    if (src / "audio.flac").exists():
        ref, sr = sf.read(src / "audio.flac", dtype="float32", always_2d=True)
        new, _ = sf.read(out / "audio.flac", dtype="float32", always_2d=True)
        comparison["audio"] = compare_audio(ref, new, sr)
    if args.stage == "nar" and (src / "latent.npy").exists():
        comparison["latent"] = compare_latents(
            np.load(src / "latent.npy", allow_pickle=False), latents
        )

    payload = {
        "operation": f"rerender_{args.stage}",
        "source_run": str(src),
        "source_id": src.name,
        "source_device": cfg.get("device"),
        "source_dtype": cfg.get("model_dtype"),
        "source_inputs_sha256": inputs,
        "source_audio_sha256": sha256(src / "audio.flac")
        if (src / "audio.flac").exists()
        else None,
        "seed": seed,
        "ode_steps": gen["ode_steps"],
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
        "timing": timing,
        "outputs_sha256": {
            "audio.flac": sha256(out / "audio.flac"),
            "latent.npy": sha256(out / "latent.npy"),
        },
        "comparison_vs_source": comparison,
    }
    (out / "rerender.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "local_env.json").write_text(
        json.dumps(
            {
                "tool": "yue2_groove",
                "dtype": dtype_loaded,
                "device": str(pipe.device),
                "torch": torch.__version__,
                "yue2": adapter.yue2_version(),
                "note": f"rerender {args.stage} of {src.name} (see rerender.json)",
            },
            indent=2,
        )
        + "\n"
    )
    finalize(out)
    print(f"[rerender] wrote {out}")
    print(json.dumps(comparison, indent=2))
    pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
