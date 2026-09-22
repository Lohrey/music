"""The kernel the tabs share: the one pipeline, the one job, the run directory.

Module state on purpose — one process serves one model: ``RUNS`` (the works
directory, set once by the CLI), the cached pipeline behind ``load_pipeline`` /
``get_pipe`` / ``unload_pipeline``, the ``RUNNING`` job slot and the ``CANCEL``
flag every generator handler honours.  ``run_generation`` is the single path
from a request to a saved run (with the panic-safe ``pending.json`` protocol),
and ``run_dir`` / ``slug`` / ``artifact_files`` name what it writes.  Handlers
reach all of it as ``runtime.<name>`` so a test can patch one place.
"""

from __future__ import annotations

import contextlib

try:
    import fcntl  # Unix only; optional macOS F_FULLFSYNC in _fsync_fd
except ImportError:  # Windows (and any host without the module)
    fcntl = None
import importlib.util
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import torch

from .. import adapter, config, gguf_engine

# Where generated works are stored; main() may override it with --runs.
RUNS = config.runs_dir()
log = logging.getLogger("yue2_groove")

_PIPE = None
_PIPE_KEY = None
_LOCK = threading.Lock()
RUNNING = threading.Lock()
CANCEL = threading.Event()

DTYPE_CHOICES = [
    ("bfloat16 (checkpoint dtype; default on CUDA/MPS)", "bfloat16"),
    ("float32 (cast at load; slower, 2x memory)", "float32"),
]

# vLLM is upstream's optional Linux/CUDA backend (``yue2-infer[fast]``).  Offer it
# only when the package is importable, so the dropdown never promises a backend
# that ends in ImportError at generate time (macOS, Windows, or a plain install).
# ``gguf`` is the yue2.cpp engine (yue2_groove.gguf_engine): always listed, because
# BACKEND=auto may select it at startup and the rail must be able to show that; loading
# it without the binaries fails with the install hint rather than a stack trace.
BACKEND_CHOICES = (
    ["torch", "torch-eager"] + (["vllm"] if importlib.util.find_spec("vllm") else []) + ["gguf"]
)
BACKEND_MODES = ["auto", *BACKEND_CHOICES]  # what --backend / YUE2_GROOVE_BACKEND accept


def resolve_backend(mode: str, device: str, *, device_explicit: bool = False) -> tuple[str, str]:
    """The concrete BACKEND for a launch: ``auto`` applies the VRAM rule (``(backend, note)``).

    An explicit ``--device cpu`` / ``mps`` is respected: the rule that routes a GPU torch cannot
    see to the GGUF engine only applies when the device itself was chosen automatically."""
    mode = (mode or "auto").strip().lower()
    if mode != "auto":
        return (mode if mode in BACKEND_CHOICES else "torch"), ""
    if device_explicit and device != "cuda":
        return "torch", ""
    return gguf_engine.auto_backend(device, gguf_engine.cuda_total_vram_gib())


# The SEED fields: -1 (the GENERATE and COVER default) draws a fresh seed for every run — the
# convention of most generation UIs — and the drawn seed is what the request carries, so the
# run records it, STATUS shows it and TRY ANOTHER SEED / a retype repeat the take exactly.
RANDOM_SEED = -1

# Song length: upstream's default semantic budget is 9 000 tokens (6:00 at 25 tokens/s); the
# model's real limit is its 24 576-token context, shared by the lyrics, the score and the
# song, so the sliders reach 12:00 and `fit_semantic_budget` trims a budget that would not
# fit at run time (upstream refuses one rather than shortening it).
SEMANTIC_TOKENS_MAX = 18000  # 12:00
ABC_TOKENS_MAX = 8192  # a 10-minute score is ~6 000 tokens (LINUX_CUDA §3.7: 600 / min)
TOKENS_PER_SECOND = 25

ABC_DEFAULTS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 30,
    "repetition_penalty": 1.005,
    "penalty_window": 100,
    "min_tokens": 32,
    "max_tokens": 4096,
}
SEM_DEFAULTS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 100,
    "repetition_penalty": 1.2,
    "penalty_window": 50,
    "min_tokens": 200,
    "max_tokens": 9000,
}


# ─────────────────────────── helpers ───────────────────────────


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_vae(choice: str, custom: str) -> tuple[str, str]:
    """Return (path_or_hub_id, display_name)."""
    if choice == "standard":
        return config.default_vae(), "standard (YuE2-Vae)"
    if choice == "legacy":
        return config.default_vae_legacy(), "legacy (benchmark)"
    if not (custom or "").strip():
        raise gr.Error("Custom VAE requires a path or Hugging Face ID")
    return custom.strip(), "custom"


def random_seed() -> int:
    return secrets.randbelow(2**31)


def resolve_seed(value) -> int:
    """The request seed for a SEED field: ``-1`` (or an empty field) draws one, anything else
    is taken as typed.  Raises ValueError for text that is not a whole number."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return random_seed()
    seed = int(float(value)) if isinstance(value, str) else int(value)
    return random_seed() if seed < 0 else seed


def fit_semantic_budget(pipe, request, abc_sampling, semantic_sampling):
    """``(semantic_sampling, note)`` with ``max_tokens`` trimmed to what the model's context
    leaves after the prompt (exact for an external score, the ABC budget's worst case for a
    model-written one); *note* says so when it trimmed, else ``""``.  Upstream refuses a
    prefix plus budget beyond the context instead of shortening it — after the ABC stage."""
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return semantic_sampling, ""
    prefix, capped = gguf_engine.semantic_budget(
        request, tokenizer, abc_sampling, semantic_sampling, gguf_engine.CONTEXT
    )
    if capped >= semantic_sampling.max_tokens:
        return semantic_sampling, ""
    fields = {
        name: getattr(semantic_sampling, name)
        for name in adapter.sampling_fields(semantic_sampling)
    }
    fields["max_tokens"] = capped
    fields["min_tokens"] = min(fields["min_tokens"], capped)
    note = (
        f"semantic max_tokens {semantic_sampling.max_tokens} → {capped} "
        f"({capped / TOKENS_PER_SECOND / 60:.1f} min): the lyrics and the score budget "
        f"({prefix} tokens at most) share the model's {gguf_engine.CONTEXT}-token context"
    )
    return adapter.sampling(**fields), note


def sampling(temp, top_p, top_k, rep, window, min_tokens, max_tokens, label: str):
    """One phase's sampling parameters from the seven slider values."""
    try:
        return adapter.sampling(
            temperature=float(temp),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(rep),
            penalty_window=int(window),
            min_tokens=int(min_tokens),
            max_tokens=int(max_tokens),
        )
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"{label} sampling parameters invalid: {exc}") from exc


def sampling_pair(*values):
    """(abc, semantic) sampling from the 14 ADVANCED // SAMPLING slider values, in
    the order the sliders are wired: seven for the ABC phase, seven for the semantic
    phase.  This is the one place that knows that order."""
    if len(values) != 14:
        raise gr.Error(f"expected 14 sampling values, got {len(values)}")
    return sampling(*values[:7], "ABC phase"), sampling(*values[7:], "semantic phase")


@dataclass(frozen=True)
class RuntimeSettings:
    """The settings rail as one value, in the order its components are wired.

    ``key`` is what identifies a loaded pipeline: the resolved device, the backend
    after the FlashAttention fallback and the VAE path, so a rail change that does
    not change the key reuses the loaded model.
    """

    device: str
    dtype: str
    backend: str
    quantization: str
    offload_ar: bool
    budget: float
    ode_steps: int
    vae_core_frames: str
    model: str
    vae_choice: str
    vae_custom: str
    revision: str
    vae_revision: str
    offline: bool

    @property
    def cores(self) -> int | None:
        return None if self.vae_core_frames == "auto" else int(self.vae_core_frames)

    @property
    def key(self) -> tuple:
        device = pick_device(self.device)
        backend, _ = effective_backend(device, self.backend)
        vae_path, _ = resolve_vae(self.vae_choice, self.vae_custom)
        return (
            device,
            self.dtype,
            backend,
            self.quantization,
            bool(self.offload_ar),
            float(self.budget),
            int(self.ode_steps),
            self.cores,
            self.model,
            vae_path,
            self.revision or "",
            self.vae_revision or "",
            bool(self.offline),
        )


FLASH_FALLBACK_NOTE = (
    "this PyTorch build or GPU cannot run FlashAttention; using upstream's "
    "eager decoder instead of CUDA graphs (slower per token, same model)"
)


def effective_backend(device: str, backend: str) -> tuple[str, str]:
    """Route a CUDA host that cannot run upstream's FlashAttention decode kernel to torch-eager.

    Only the ``torch`` backend on a CUDA device is ever remapped.  A Linux host with a
    FlashAttention-capable torch build and an Ampere-or-newer GPU keeps the CUDA-graph path
    untouched; MPS and CPU never consult the probe.  Returns ``(backend, reason)``.

    Temporary: works around yue2-v0.1.6 selecting FlashAttention by op presence alone
    (upstream PR #166 fixes it and keeps CUDA graphs via cuDNN/SDPA).  The contract test
    ``test_cuda_graph_still_selects_flash_attention_by_op_presence_only`` fails once the
    installed upstream carries that fix — remove this function and its callers then.
    """
    if device == "cuda" and backend == "torch" and not adapter.cuda_flash_attention_usable():
        return "torch-eager", FLASH_FALLBACK_NOTE
    return backend, ""


def load_pipeline(settings: RuntimeSettings, progress=None):
    """(Re)load the pipeline. Reuses the existing one when settings are unchanged."""
    global _PIPE, _PIPE_KEY
    device = pick_device(settings.device)
    backend, fallback = effective_backend(device, settings.backend)
    fallback = f" ({fallback})" if fallback else ""
    vae_path, vae_name = resolve_vae(settings.vae_choice, settings.vae_custom)
    key = settings.key
    if _PIPE is not None and key == _PIPE_KEY:
        if isinstance(_PIPE, gguf_engine.GgufPipeline):
            return (
                _PIPE,
                f"Model ready: engine=yue2.cpp ({_PIPE.quant} GGUF) backend=gguf vae={vae_name}",
            )
        return _PIPE, (
            f"Model ready: device={device} dtype={settings.dtype} "
            f"backend={backend}{fallback} vae={vae_name}"
        )
    if backend == "gguf":  # DTYPE / QUANTIZATION / OFFLOAD / BUDGET do not apply: no checks on them
        return _load_gguf(settings, vae_path, vae_name, key, progress)
    if backend == "vllm" and device != "cuda":
        raise gr.Error(
            "vLLM backend requires NVIDIA CUDA; use torch here (MPS falls back to eager)"
        )
    if settings.quantization == "fp8" and device != "cuda":
        raise gr.Error("FP8 quantization requires NVIDIA CUDA (sm89+)")

    unload_pipeline()
    if progress is not None:
        progress(0.05, desc="Loading model (first run downloads ~7.3 GB)…")
    with _LOCK:
        pipe, used_dtype = adapter.load_pipeline(
            settings.model,
            vae=vae_path,
            device=device,
            dtype=settings.dtype,
            backend=backend,
            quantization=settings.quantization,
            offload_ar=settings.offload_ar,
            memory_budget_gib=settings.budget,
            ode_steps=settings.ode_steps,
            vae_core_frames=settings.cores,
            revision=settings.revision,
            vae_revision=settings.vae_revision,
            local_files_only=settings.offline,
        )
        _PIPE, _PIPE_KEY = pipe, key
    note = (
        f"Loaded: device={device} dtype={used_dtype} backend={backend}{fallback} "
        f"vae={vae_name} ode_steps={settings.ode_steps} cores={settings.cores or 'auto'}"
    )
    return _PIPE, note


def _load_gguf(settings: RuntimeSettings, vae_path: str, vae_name: str, key, progress):
    """The yue2.cpp engine: nothing resident, but the GGUF files must exist (prepared once
    from the checkpoints already on disk) and the tokenizer needs the checkpoint directory.
    DTYPE / QUANTIZATION / OFFLOAD / MEMORY BUDGET do not apply and are ignored."""
    global _PIPE, _PIPE_KEY
    if settings.vae_choice != "standard":
        raise gr.Error(
            "the GGUF engine uses the standard YuE2-Vae; pick STANDARD or the torch backend"
        )
    if not gguf_engine.available():
        raise gr.Error(gguf_engine.INSTALL_HINT)
    unload_pipeline()
    if progress is not None:
        progress(
            0.05, desc="Preparing the GGUF engine (first run downloads ~7.3 GB; see the terminal)…"
        )

    def say(line: str) -> None:
        log.info("gguf: %s", line)
        if progress is not None:
            progress(0.05, desc=line[:120])

    with _LOCK:
        model_dir = adapter.resolve_model(
            settings.model, revision=settings.revision or None, local_files_only=settings.offline
        )
        vae_dir = adapter.resolve_model(
            vae_path, revision=settings.vae_revision or None, local_files_only=settings.offline
        )
        pipe = gguf_engine.GgufPipeline.open(
            model_dir=Path(model_dir),
            vae_dir=Path(vae_dir),
            ode_steps=settings.ode_steps,
            vae_core_frames=settings.cores,
            log=say,
        )
        _PIPE, _PIPE_KEY = pipe, key
    note = (
        f"Loaded: engine=yue2.cpp ({pipe.quant} GGUF, {pipe.backbone.name}) backend=gguf "
        f"vae={vae_name} ode_steps={settings.ode_steps} cores={settings.cores or 'auto'}"
        + (f" {pipe.max_seq_reason}" if pipe.max_seq else "")
        + " — not the reference configuration: a different take for the same seed"
    )
    return _PIPE, note


def unload_pipeline() -> None:
    global _PIPE, _PIPE_KEY
    with _LOCK:
        if _PIPE is not None:
            with contextlib.suppress(Exception):  # closing must never raise
                adapter.close_pipeline(_PIPE)
        _PIPE, _PIPE_KEY = None, None
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_pipe(settings: RuntimeSettings, progress=None):
    """The loaded pipeline for these settings, loading it first when they changed."""
    if _PIPE is None or settings.key != _PIPE_KEY:
        return load_pipeline(settings, progress)
    return _PIPE, "Model ready"


def write_local_env(directory: Path, pipe, note: str = "", *, budget_note: str = "") -> None:
    """Record what actually ran in this run directory (``local_env.json``).

    Upstream's ``config.json`` hardcodes ``"model_dtype": "bfloat16"``, so an
    explicit float32 cast from this UI (or any future override) would otherwise
    be misreported.  Best-effort only: a sidecar, never a rewrite of upstream
    artifacts.  The Library shows it when it disagrees with ``config.json``.
    """
    try:
        payload = {
            "tool": "yue2_groove",
            "dtype": adapter.model_dtype(pipe),
            "device": str(getattr(pipe, "device", "")),
            "torch": torch.__version__,
            "yue2": adapter.yue2_version(),
            "note": note or None,
            "semantic_budget_note": budget_note or None,
        }
        if isinstance(pipe, gguf_engine.GgufPipeline):
            payload.update(gguf_engine.local_env(pipe))
        (Path(directory) / "local_env.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:  # noqa: BLE001, S110 — provenance must never fail a run
        pass


def run_dir(*parts: str) -> Path:
    """A fresh, time-stamped directory name under RUNS: 20260919-213000-<parts…>."""
    return RUNS / "-".join((time.strftime("%Y%m%d-%H%M%S"), *parts))


def slug(text: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "-" for c in (text or "")[:40].strip())
    return out.strip("-") or "song"


def artifact_files(directory: Path, score: bool):
    names = [
        "audio.flac",
        "request.json",
        "config.json",
        "result.json",
        "latent.npy",
        "semantic.npy",
    ] + (["score.abc"] if score else [])
    return [str(directory / n) for n in names if (directory / n).exists()]


def _looks_like_run(path: Path) -> bool:
    return any(
        (path / name).exists() for name in ("latent.npy", "result.json", "plan.json", "decode.json")
    )


def scan_runs():
    if not RUNS.is_dir():
        return []
    return [
        str(p)
        for p in sorted(RUNS.iterdir(), reverse=True)
        if p.is_dir() and not p.name.startswith(".") and _looks_like_run(p)
    ]


def cancel_run() -> str:
    CANCEL.set()
    return "Cancel requested — will stop after the current token / ODE step"


# ── the one job at a time ──────────────────────────────────────────────────
# Every generator handler claims the slot, yields its updates inside a try and
# releases it in a finally; a fast double click that beats the disabled button
# is refused with BUSY_MESSAGE rather than queued.
BUSY_MESSAGE = "Another job is already running — wait for it to finish"


def try_start_job() -> bool:
    """Claim the job slot (and clear a stale cancel); False while another job runs."""
    CANCEL.clear()
    return RUNNING.acquire(blocking=False)


def end_job() -> None:
    RUNNING.release()


def failure_text(exc: BaseException, what: str = "Generation") -> str:
    """The status line for a job that did not finish (a cancel is not a failure)."""
    if isinstance(exc, InterruptedError):
        return f"Cancelled: {exc}"
    return f"{what} failed: {type(exc).__name__}: {exc}"


# ───────────────────── run durability (panic-safe writes) ─────────────────────
# save_artifacts() closes the files but never fsyncs, and the whole point of the
# 2026-09-13 panics was that a run could be listened to and still vanish when the
# kernel died before APFS flushed it.  A run is only "done" once its files and
# directory are flushed; ".pending" is the marker the Library shows when it is not.
PENDING_FILE = "pending.json"


def _fsync_fd(fd: int) -> None:
    with contextlib.suppress(OSError):
        os.fsync(fd)
    # macOS: fsync only reaches the drive cache, F_FULLFSYNC reaches the media.
    # fcntl is absent on Windows and the flag is macOS-only; skip the extra flush.
    if fcntl is not None:
        full_fsync = getattr(fcntl, "F_FULLFSYNC", None)
        if full_fsync is not None:
            with contextlib.suppress(OSError):
                fcntl.fcntl(fd, full_fsync)


def _fsync_path(path) -> None:
    try:
        fd = os.open(Path(path), os.O_RDONLY)
    except OSError:
        return
    try:
        _fsync_fd(fd)
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def _fsync_tree(directory) -> None:
    """Flush every file in *directory* and the directory itself."""
    directory = Path(directory)
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.is_file():
            _fsync_path(entry)
    _fsync_path(directory)


def _write_pending(directory, status: str, error: str = "") -> None:
    """Mark *directory* as an unfinished run, durably.

    Written before generation starts and removed only after every artifact is
    flushed, so a run that never finishes stays visible in the Library instead of
    disappearing silently.
    """
    directory = Path(directory)
    payload = {
        "schema": "yue2-groove-pending-v1",
        "status": status,
        "pid": os.getpid(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if error:
        payload["error"] = error[:500]
    path = directory / PENDING_FILE
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _fsync_path(path)
        _fsync_path(directory)
    except OSError:
        pass


def _clear_pending(directory) -> None:
    directory = Path(directory)
    try:
        (directory / PENDING_FILE).unlink()
        _fsync_path(directory)
    except OSError:
        pass


def run_generation(
    pipe, request, outdir, *, abc_sampling, semantic_sampling, progress, note, extra_manifest=None
):
    """Generate one song into *outdir* and save its artifacts.

    Shared by 01 GENERATE and 03 EDIT so both flows report progress identically.
    ``extra_manifest`` (a dict) is written next to the run as ``edit_manifest.json``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_pending(outdir, "running")
    log.info("run start: %s", outdir)
    counts = {"abc": 0, "semantic": 0}
    semantic_sampling, budget_note = fit_semantic_budget(
        pipe, request, abc_sampling, semantic_sampling
    )
    if budget_note:
        log.info("run %s: %s", outdir.name, budget_note)
    abc_budget = abc_sampling.max_tokens if request.cot != "off" else 0
    sem_budget = semantic_sampling.max_tokens

    def on_token(phase, token, count=1):  # the GGUF engine reports in 100-token strides
        counts[phase] = counts.get(phase, 0) + count
        if request.cot == "off":
            frac = 0.55 * min(1.0, counts["semantic"] / max(1, sem_budget))
        else:
            frac = 0.15 * min(1.0, counts["abc"] / max(1, abc_budget)) + 0.40 * min(
                1.0, counts["semantic"] / max(1, sem_budget)
            )
        progress(min(0.55, frac), desc=f"Generating {phase}: {counts[phase]} tokens")

    def on_progress(stage, done, total):
        if stage == "nar":
            progress(
                0.55 + 0.40 * min(1.0, done / max(1, total)),
                desc=f"Synthesizing audio: step {done}/{total}",
            )
        elif stage == "vae":
            progress(
                0.95 + 0.05 * min(1.0, done / max(1, total)),
                desc=f"Decoding audio: chunk {done}/{total}",
            )

    t0 = time.perf_counter()
    try:
        song = adapter.generate(
            pipe,
            request,
            abc_sampling=abc_sampling,
            semantic_sampling=semantic_sampling,
            cancelled=CANCEL.is_set,
            on_token=on_token,
            on_progress=on_progress,
        )
        progress(1.0, desc="Saving artifacts…")
        result = song.save_artifacts(outdir)
        if extra_manifest is not None:
            (Path(outdir) / "edit_manifest.json").write_text(
                json.dumps(extra_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        write_local_env(outdir, pipe, note, budget_note=budget_note)
        result["budget_note"] = budget_note  # for the status line; local_env.json keeps it
    except Exception as exc:
        _write_pending(
            outdir,
            "cancelled" if isinstance(exc, InterruptedError) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        log.warning("run unfinished: %s — %s: %s", outdir, type(exc).__name__, exc)
        raise
    # flush the artifacts first, then remove the marker: the run only stops being
    # "pending" once it is actually on disk
    _fsync_tree(outdir)
    _clear_pending(outdir)
    log.info("run done: %s", outdir)
    return song, result, time.perf_counter() - t0


def generation_status(song, result, outdir, elapsed, request, note):
    budget = result.get("budget_note") if isinstance(result, dict) else None
    return (
        f"Done: {result['audio_seconds']:.1f}s audio in {elapsed:.0f}s\n"
        f"truncated={result['truncated']}  seed={request.seed}  cfg={request.guidance}\n"
        f"NAR={song.timing['nar_seconds']:.0f}s  VAE={song.timing['vae_seconds']:.0f}s  "
        f"semantic={song.timing['semantic'].get('output_tps', 0):.1f} tok/s  "
        f"ABC={song.timing['abc'].get('output_tokens', 0)} tokens\n"
        + (f"{budget}\n" if budget else "")
        + f"run directory: {outdir}\n{note}"
    )


def _load_project_example():
    """Example request that feeds the placeholders and the "fill example" buttons."""
    return (
        (
            "English, warm piano pop, expressive female voice, acoustic piano, "
            "rounded bass and light drums, 88 BPM"
        ),
        (
            "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n\n"
            "[Chorus]\nLet the day come into view\nEvery road begins with you"
        ),
    )


EXAMPLE_STYLE, EXAMPLE_LYRICS = _load_project_example()


def request_texts(style, lyrics, *, fallback: bool = False):
    """Resolve STYLE/LYRICS; empty fields are an error unless *fallback* is set.

    The repository example is never substituted silently: it stays a placeholder
    and one E-button click away, but the request that runs is what the user typed.
    """
    style, lyrics = (style or "").strip(), (lyrics or "").strip()
    if fallback:
        return style or EXAMPLE_STYLE, lyrics or EXAMPLE_LYRICS
    missing = [name for name, value in (("STYLE", style), ("LYRICS", lyrics)) if not value]
    if missing:
        raise gr.Error(
            f"{' and '.join(missing)} empty — type a target, or click E to fill the "
            f"repository example"
        )
    return style, lyrics
