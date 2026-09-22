"""The GGUF engine: YuE2 through yue2.cpp (GGML) instead of PyTorch.

A second, clearly labelled engine for cards the reference configuration does not fit —
12 GB and 8 GB NVIDIA cards, and Windows hosts whose PyTorch build has no FlashAttention.
It runs ``yue-synth`` / ``yue-plan`` / ``neural-codec`` from the pinned yue2.cpp release
as child processes (the same boundary the SheetSage2 environment uses: JSON in, files
out, nothing imported), and writes the same run directory the PyTorch engine writes, so
04 LIBRARY / 03 EDIT / 06 DECODE / the comparison page accept a GGUF run like any other.

What it is not: the reference configuration.  Even the BF16 GGUF does not reproduce the
PyTorch engine bit for bit, and a quantized backbone samples a *different take* for the
same seed (any perturbation forks the token stream — see docs/CROSS_PLATFORM.md).  Runs
record ``engine`` / GGUF hashes in ``config.json`` and ``result.json`` so the two never mix.

Measured (docs/GGUF_ENGINE.md): with the reference run's tokens and noise, Q8_0's rendering
lands 24 dB from the CUDA reference — the same size as a CUDA → MPS platform change — and
was not distinguishable from it in a blind ABX (6/12).  Q6_K and below degrade measurably.

Selection: ``YUE2_GROOVE_BACKEND=auto`` (the default) picks this engine on a CUDA host whose
card has less than ``YUE2_GROOVE_GGUF_VRAM_GIB`` (16) GiB and a yue2.cpp binary installed;
otherwise the PyTorch engine.  BACKEND in the settings rail overrides either way.

Layout: binaries in ``YUE2_GROOVE_YUE2CPP`` (default ``<repo>/bin/yue2cpp``, else PATH);
GGUF files in ``YUE2_GROOVE_GGUF`` (default ``<repo>/models/gguf``), prepared on first use
from the already-downloaded checkpoints with yue2.cpp's own converter (vendored,
byte-identical) and ``quantize`` — the local Q8_0 is tensor-for-tensor identical to the
published ``Serveurperso/YuE2-GGUF`` file.  ``YUE2_GROOVE_GGUF_QUANT`` (Q8_0) picks the
variant; ``YUE2_GROOVE_GGUF_MAX_SEQ`` caps the KV cache for 8 GB cards.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import config

# The yue2.cpp revision this engine is written against (request JSON fields, CLI flags,
# log lines parsed below).  The CI workflow builds exactly this commit; bump both together.
YUE2CPP_PIN = "e8b39f7"
YUE2CPP_REPO = "https://github.com/ServeurpersoCom/yue2.cpp"

QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "BF16")
DEFAULT_QUANT = "Q8_0"
DEFAULT_VRAM_THRESHOLD_GIB = 16.0
CONTEXT = 24576
CODEC_OFFSET = 151853
SAMPLE_RATE = 48000

BINARIES = ("yue-synth", "yue-plan", "neural-codec", "quantize")

INSTALL_HINT = (
    "yue2.cpp binaries not found. Download the yue2cpp-<platform> asset of this release "
    "(github.com/deadjoe/yue2_groove/releases) into <repo>/bin/yue2cpp, or set "
    "YUE2_GROOVE_YUE2CPP to a directory holding yue-synth / yue-plan / neural-codec / "
    f"quantize (built from {YUE2CPP_REPO} at {YUE2CPP_PIN})."
)

StageProgress = Callable[[str, int, int], None]


# ── locating things ──────────────────────────────────────────────────────────


def repo_root() -> Path:
    return config.PACKAGE_DIR.parent


def binary_dir() -> Path | None:
    """Directory holding the yue2.cpp binaries, or None when none is installed."""
    explicit = (os.environ.get("YUE2_GROOVE_YUE2CPP") or "").strip()
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.append(repo_root() / "bin" / "yue2cpp")
    for candidate in candidates:
        if _binary(candidate, "yue-synth") is not None:
            return candidate
    on_path = shutil.which("yue-synth")
    return Path(on_path).parent if on_path else None


def _binary(directory: Path, name: str) -> Path | None:
    for suffix in ("", ".exe"):
        candidate = directory / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def binary(name: str) -> Path:
    directory = binary_dir()
    found = _binary(directory, name) if directory is not None else None
    if found is None:
        raise RuntimeError(INSTALL_HINT)
    return found


def available() -> bool:
    return binary_dir() is not None


_VERSION_LINE = re.compile(r"yue2\.cpp (\S+) \((\d{4}-\d\d-\d\d)\)")


def engine_version(path: Path | None = None) -> str:
    """What the installed ``yue-synth`` says it is (``e8b39f7 (2026-09-18)``), or ``""``.

    The binaries print their commit on the usage banner; the run records this next to the
    pin the app was written against, so an archive never claims a version it did not run.
    """
    try:
        out = subprocess.run(
            [str(path or binary("yue-synth"))],
            capture_output=True,
            timeout=30,
            check=False,
            **config.SUBPROCESS_TEXT,
        )
    except Exception:  # noqa: BLE001 — unrunnable binary: the generate path reports it properly
        return ""
    m = _VERSION_LINE.search((out.stdout or "") + (out.stderr or ""))
    return f"{m.group(1)} ({m.group(2)})" if m else ""


def exact_ids_wanted() -> bool:
    """``YUE2_GROOVE_GGUF_EXACT_IDS=1``: read the AR ids the engine actually used (yue2.cpp's
    ``--dump``, several hundred MB of scratch per song) instead of re-tokenizing the score.
    Off by default: in 28 of 28 archived model-written scores the two were identical."""
    return (os.environ.get("YUE2_GROOVE_GGUF_EXACT_IDS") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def gguf_dir() -> Path:
    explicit = (os.environ.get("YUE2_GROOVE_GGUF") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    models = (os.environ.get("YUE2_GROOVE_MODELS") or "").strip()
    if models:
        return Path(models).expanduser() / "gguf"
    return repo_root() / "models" / "gguf"


def quant() -> str:
    value = (os.environ.get("YUE2_GROOVE_GGUF_QUANT") or DEFAULT_QUANT).strip().upper()
    return value if value in QUANTS else DEFAULT_QUANT


SMALL_CARD_GB = 8  # cards up to this size get the context cap by default
SMALL_CARD_MAX_SEQ = 12288  # 2 KV sets ≈ 2.7 GB instead of 5.4: ~5.5 GB peak, fits 8 GB


def max_seq() -> int | None:
    """``YUE2_GROOVE_GGUF_MAX_SEQ``: an explicit KV-cache cap (``--max-seq``), or None."""
    raw = (os.environ.get("YUE2_GROOVE_GGUF_MAX_SEQ") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if 0 < value < CONTEXT else None


def effective_max_seq(card_gb: int | None) -> tuple[int | None, str]:
    """``(max_seq, reason)``: the environment's cap when set, else the small-card default.

    Measured on a 16 GB card (docs/GGUF_ENGINE.md): the full 24 576 context peaks at 8.2 GB, of
    which two KV sets are 5.4 GB — an 8 GB card cannot hold that, 12 288 brings it to ~5.5 GB.
    """
    explicit = max_seq()
    if explicit is not None:
        return explicit, f"max_seq={explicit} (YUE2_GROOVE_GGUF_MAX_SEQ)"
    if card_gb is not None and card_gb <= SMALL_CARD_GB:
        return (
            SMALL_CARD_MAX_SEQ,
            f"max_seq={SMALL_CARD_MAX_SEQ} ({card_gb} GB card: context capped)",
        )
    return None, ""


def semantic_budget(
    request, tokenizer, abc_sampling, semantic_sampling, cap: int
) -> tuple[int, int]:
    """``(prefix_tokens, max_tokens)`` that fit *cap*: yue2.cpp refuses a prefix plus semantic
    budget that its KV cache cannot hold (and upstream one beyond the model's context), so
    the budget is derived from the cap up front — the app applies it with ``CONTEXT`` for
    both engines (``runtime.fit_semantic_budget``) and this engine again with its ``max_seq``.

    With an external score the prefix is exact; with a model-written one it is the worst
    case (the ABC stage's own ``max_tokens``), so a full-length request stays valid whatever
    the model writes.  The ABC stage itself always fits: its prefix is a few hundred tokens.
    """
    from . import adapter

    if request.cot == "off" or request.abc is not None:
        abc_ids = tokenizer.encode(request.abc) if request.abc is not None else None
        prefix_tokens = len(adapter.token_prefixes(request, tokenizer, abc_ids))
    else:
        base = len(adapter.token_prefixes(request, tokenizer))  # EOD + text + ABC_START
        prefix_tokens = base + int(abc_sampling.max_tokens) + 2  # + score + ABC_END, MUSIC_START
    budget = cap - prefix_tokens - 1  # MUSIC_END
    if budget < 250:  # ten seconds of audio: below that the request is not worth running
        raise RuntimeError(
            f"the prompt ({prefix_tokens} tokens) leaves no room for a song in a {cap}-token "
            "context: shorten the lyrics or the ABC budget"
            + (", or raise YUE2_GROOVE_GGUF_MAX_SEQ" if cap < CONTEXT else "")
        )
    return prefix_tokens, min(int(semantic_sampling.max_tokens), budget)


def vram_threshold_gib() -> float:
    try:
        return float(os.environ.get("YUE2_GROOVE_GGUF_VRAM_GIB") or DEFAULT_VRAM_THRESHOLD_GIB)
    except ValueError:
        return DEFAULT_VRAM_THRESHOLD_GIB


def backbone_name(quant_label: str) -> str:
    return f"YuE2-3B-{quant_label}.gguf"


VAE_NAME = "YuE2-Vae-F32.gguf"


def auto_backend(device: str, total_vram_gib: float | None, *, installed: bool | None = None):
    """``(backend, reason)`` for BACKEND=auto.

    The GGUF engine is chosen only for a CUDA card below the threshold with the binaries
    installed; every other host keeps the reference PyTorch engine.  *reason* says why, for
    the status line and the log — including why a small card did **not** get it.  Cards are
    compared by their marketed size: a "16 GB" card reports 15.99 GiB and must count as 16.
    """
    installed = available() if installed is None else installed
    threshold = vram_threshold_gib()
    if total_vram_gib is None:
        return "torch", ""
    card_gb = round(total_vram_gib)
    if device != "cuda":
        # an NVIDIA card torch cannot see (a CPU-only torch build, e.g. PyPI's Windows wheel):
        # the reference engine would run on the CPU for hours; yue2.cpp drives the card itself
        if device == "cpu" and installed:
            return "gguf", (
                f"auto: NVIDIA card ({card_gb} GB) present but torch has no CUDA "
                f"→ GGUF {quant()} engine on the GPU"
            )
        if device == "cpu":
            return "torch", (
                f"NVIDIA card ({card_gb} GB) present but torch has no CUDA, so generation runs "
                "on the CPU (hours per song). Install the GGUF engine binaries to use the card "
                "(see docs/GGUF_ENGINE.md), or a CUDA build of torch."
            )
        return "torch", ""
    if card_gb >= threshold:
        return "torch", ""
    if not installed:
        return "torch", (
            f"{card_gb} GB card: the GGUF engine would fit better, but no yue2.cpp "
            "binaries are installed (see docs/GGUF_ENGINE.md)"
        )
    return (
        "gguf",
        f"auto: {card_gb} GB card < {threshold:.0f} GB → GGUF {quant()} engine",
    )


def nvidia_total_vram_gib() -> float | None:
    """Total memory of the first NVIDIA GPU from ``nvidia-smi``, or None without one.

    Asked first because ``torch.cuda.get_device_properties`` initialises a CUDA context —
    a few hundred MB of the very VRAM this rule is about, held by a process that, with the
    GGUF engine, never needs one.  Works with a CPU-only torch as well (the PyPI Windows
    wheel), where the card is invisible to torch but usable by yue2.cpp.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            timeout=15,
            check=False,
            **config.SUBPROCESS_TEXT,
        )
        first = out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 else ""
        return float(first) / 1024 if first else None  # MiB → GiB
    except Exception:  # noqa: BLE001 — a broken driver is the same as no card for this rule
        return None


def nvidia_compute_capability() -> float | None:
    """Compute capability of the first NVIDIA GPU (e.g. 6.1 for a GTX 1070), or None."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, timeout=15, check=False, **config.SUBPROCESS_TEXT,
        )
        first = out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 else ""
        return float(first) if first else None
    except Exception:  # noqa: BLE001 — old driver without compute_cap: assume a modern card
        return None


# The release binaries are built for CUDA sm_75 (RTX 20 series) and newer.  Older cards
# (GTX 10 series / Pascal, GTX 16 series is 7.5 and fine) run the same binaries on the
# Vulkan backend instead: hiding the CUDA devices from the child makes ggml pick Vulkan.
MIN_CUDA_CC = 7.5


def engine_env() -> dict[str, str]:
    env = config.child_env()
    wanted = os.environ.get("YUE2_GROOVE_GGUF_DEVICE", "auto").strip().lower()
    cc = nvidia_compute_capability() if wanted == "auto" else None
    if wanted == "vulkan" or (cc is not None and cc < MIN_CUDA_CC):
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def cuda_total_vram_gib() -> float | None:
    """Total VRAM in GiB: ``nvidia-smi`` first (no CUDA context), then torch."""
    vram = nvidia_total_vram_gib()
    if vram is not None:
        return vram
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:  # noqa: BLE001 — no torch / no driver: the rule simply does not apply
        return None


# ── hashes ───────────────────────────────────────────────────────────────────


def sha256_file(path: Path, *, cache: bool = True) -> str:
    """SHA-256 of a (large) file; cached in ``<file>.sha256`` keyed by size + mtime."""
    path = Path(path)
    stat = path.stat()
    side = path.with_name(path.name + ".sha256")
    if cache and side.is_file():
        try:
            cached = json.loads(side.read_text(encoding="utf-8"))
            if cached.get("bytes") == stat.st_size and cached.get("mtime") == stat.st_mtime:
                return cached["sha256"]
        except (OSError, ValueError, KeyError):
            pass
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if cache:
        with contextlib.suppress(OSError):
            side.write_text(
                json.dumps({"sha256": digest, "bytes": stat.st_size, "mtime": stat.st_mtime}),
                encoding="utf-8",
            )
    return digest


# ── preparing the GGUF files ─────────────────────────────────────────────────


def _hash_cache_path(out_dir: Path) -> Path:
    return out_dir / ".hashes.json"


def cached_sha256(path: Path, out_dir: Path) -> str:
    """SHA-256 of *path*, remembered in ``<out_dir>/.hashes.json`` by size + mtime — the
    7 GB checkpoint is hashed once, and nothing is ever written into the checkpoint directory."""
    path = Path(path)
    stat = path.stat()
    key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    cache_file = _hash_cache_path(out_dir)
    cache: dict = {}
    with contextlib.suppress(OSError, ValueError):
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    digest = cache.get(key)
    if isinstance(digest, str) and len(digest) == 64:
        return digest
    digest = sha256_file(path, cache=False)
    cache[key] = digest
    with contextlib.suppress(OSError):
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return digest


def converter_version(path: Path | None = None) -> str:
    """SHA-256 of the vendored converter: a converter change is a conversion-input change.

    Hashed with CRLF folded to LF — a Windows checkout under Git's ``autocrlf`` reads the
    same converter, and must name the same GGUF file, as a Linux or macOS one.
    """
    path = path or config.PACKAGE_DIR / "vendor" / "yue2cpp_convert.py"
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def checkpoint_identity(checkpoint_dir: Path, out_dir: Path) -> dict:
    """Everything the conversion reads from a checkpoint, by actual content.

    ``files``: every ``*.safetensors`` hashed (verified against ``weights_manifest.json`` when
    the checkpoint ships one — a mismatch is an integrity failure, not a cache miss);
    ``config.json`` and ``qwen.tiktoken`` (both embedded in the GGUF); and the converter
    itself.  ``identity`` folds them into one digest that names the GGUF file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    weights = sorted(checkpoint_dir.glob("*.safetensors"))
    if not weights:
        raise RuntimeError(f"no safetensors weights in {checkpoint_dir}")
    files = {w.name: cached_sha256(w, out_dir) for w in weights}
    manifest = checkpoint_dir / "weights_manifest.json"
    if manifest.is_file():
        with contextlib.suppress(OSError, ValueError, TypeError):
            expected = json.loads(manifest.read_text(encoding="utf-8")).get("files") or {}
            for name, digest in files.items():
                wanted = (expected.get(name) or {}).get("sha256")
                if wanted and wanted != digest:
                    raise RuntimeError(
                        f"weight integrity failed: {checkpoint_dir / name} does not match "
                        f"weights_manifest.json ({digest[:12]} vs {wanted[:12]})"
                    )
    extras = {}
    for name in ("config.json", "qwen.tiktoken"):
        path = checkpoint_dir / name
        if path.is_file():
            extras[name] = sha256_file(path, cache=False)
    record = {"files": files, **extras, "converter": converter_version()}
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return {"identity": identity, **record}


def gguf_names(quant_label: str, model_identity: str, vae_identity: str) -> tuple[str, str]:
    """``YuE2-3B-<quant>-<id12>.gguf`` and ``YuE2-Vae-F32-<id12>.gguf``."""
    return (
        f"YuE2-3B-{quant_label}-{model_identity[:12]}.gguf",
        f"YuE2-Vae-F32-{vae_identity[:12]}.gguf",
    )


def ready_made(out_dir: Path, quant_label: str) -> tuple[Path, Path] | None:
    """Plain-named files a user placed in an explicitly configured ``YUE2_GROOVE_GGUF``
    (downloaded from the published repository): used as they are, provenance untied."""
    if not (os.environ.get("YUE2_GROOVE_GGUF") or "").strip():
        return None
    backbone, vae = out_dir / backbone_name(quant_label), out_dir / VAE_NAME
    return (backbone, vae) if backbone.is_file() and vae.is_file() else None


@dataclass
class Prepared:
    backbone: Path
    vae: Path
    provenance: str  # "converted" | "ready-made"
    model: dict | None = None  # checkpoint_identity() records when converted
    vae_source: dict | None = None


def prepare(
    model_dir: Path,
    vae_dir: Path,
    out_dir: Path | None = None,
    *,
    quant_label: str | None = None,
    log: Callable[[str], None] | None = None,
) -> Prepared:
    """Make sure the GGUF files for *these* checkpoints exist.

    Files are named after :func:`checkpoint_identity` (weights, config, tokenizer, converter),
    so a change of model, revision, VAE or converter converts afresh instead of silently
    reusing an older file.  The whole conversion — yue2.cpp's converter (a byte-identical
    vendored copy, driven in a child interpreter) and the release's ``quantize`` — works in
    a private ``.partial-*`` directory, including the 7.2 GB BF16 intermediate; only the
    finished file is moved into place.  So a concurrent preparation can neither clobber
    another nor delete a file another process is using, and an interrupted one never leaves
    a truncated .gguf that looks finished.  Idempotent: existing files are kept.
    """
    out_dir = Path(out_dir or gguf_dir())
    quant_label = quant_label or quant()
    say = log or (lambda _msg: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    found = ready_made(out_dir, quant_label)
    if found is not None:
        say(f"using ready-made GGUF files in {out_dir} (external: source not verified)")
        return Prepared(found[0], found[1], "ready-made")
    model = checkpoint_identity(model_dir, out_dir)
    vae_source = checkpoint_identity(vae_dir, out_dir)
    backbone_file, vae_file = gguf_names(quant_label, model["identity"], vae_source["identity"])
    backbone, vae = out_dir / backbone_file, out_dir / vae_file
    if backbone.is_file() and vae.is_file():
        return Prepared(backbone, vae, "converted", model, vae_source)

    _sweep_stale_partials(out_dir)
    partial = Path(tempfile.mkdtemp(prefix=".partial-", dir=str(out_dir)))
    try:
        components = {}
        if not backbone.is_file():
            components["backbone"] = str(Path(model_dir).resolve())
        if not vae.is_file():
            components["vae"] = str(Path(vae_dir).resolve())
        say(f"Converting checkpoints to GGUF ({', '.join(components)}) — one-time, under a minute…")
        _run_converter(components, partial, say)
        if "vae" in components:
            os.replace(partial / VAE_NAME, vae)
        if "backbone" in components:
            native = partial / backbone_name("BF16")
            if quant_label == "BF16":
                os.replace(native, backbone)
            else:
                say(f"Quantizing to {quant_label} — one-time…")
                staged = partial / backbone.name
                _run(
                    [
                        child_path(binary("quantize")),
                        child_path(native),
                        child_path(staged),
                        quant_label,
                    ],
                    say,
                )
                os.replace(staged, backbone)
    finally:
        shutil.rmtree(partial, ignore_errors=True)  # takes the BF16 intermediate with it
    for path in (backbone, vae):
        if not path.is_file():
            raise RuntimeError(f"GGUF preparation did not produce {path}")
    # files of an earlier identity (another revision, an older converter) are left in place —
    # nothing here deletes what another process may be reading — but named, with their size
    superseded = sorted(
        old
        for pattern in (f"YuE2-3B-{quant_label}-*.gguf", "YuE2-Vae-F32-*.gguf")
        for old in out_dir.glob(pattern)
        if old not in (backbone, vae)
    )
    if superseded:
        say(
            "no longer used, can be deleted: "
            + ", ".join(f"{old.name} ({old.stat().st_size / 1e9:.1f} GB)" for old in superseded)
        )
    return Prepared(backbone, vae, "converted", model, vae_source)


def _sweep_stale_partials(out_dir: Path, older_than_seconds: float = 86400) -> None:
    """Leftovers of a preparation that died more than a day ago (a live one is never touched)."""
    now = time.time()
    for stale in out_dir.glob(".partial-*"):
        with contextlib.suppress(OSError):
            if now - stale.stat().st_mtime > older_than_seconds:
                shutil.rmtree(stale, ignore_errors=True)


def _run_converter(components: dict[str, str], out_dir: Path, say) -> None:
    """The vendored converter in a child interpreter: paths patched, memory released on exit."""
    if importlib.util.find_spec("gguf") is None:
        raise RuntimeError(
            "GGUF conversion needs the `gguf` package: uv pip install gguf   "
            "(or point YUE2_GROOVE_GGUF at a directory with ready-made GGUF files)"
        )
    code = (
        "import json, sys\n"
        "import yue2_groove.vendor.yue2cpp_convert as cv\n"
        "spec = json.loads(sys.argv[1])\n"
        "cv.CHECKPOINT_DIR = ''\n"
        "cv.OUTPUT_DIR = spec['out']\n"
        "cv.COMPONENTS.update(spec['components'])\n"
        "for name in spec['components']:\n"
        "    cv.convert(name)\n"
    )
    spec = json.dumps({"out": str(out_dir), "components": components})
    _run([sys.executable, "-c", code, spec], say)


def _run(argv: list[str], say) -> None:
    """A preparation child (the converter, ``quantize``): its stderr goes to *say*; any early
    exit — a raising *say*, a KeyboardInterrupt — stops and reaps the child before the
    private directory it was writing into is removed."""
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=engine_env(),
        **config.SUBPROCESS_TEXT,
    )
    tail: list[str] = []
    try:
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.rstrip()
            if line:
                tail = (tail + [line])[-20:]
                say(line)
        code = proc.wait()
    except BaseException:
        _stop_child(proc)
        raise
    finally:
        with contextlib.suppress(OSError):
            if proc.stderr is not None:
                proc.stderr.close()
    if code != 0:
        raise RuntimeError(f"{Path(argv[0]).name} failed (exit {code}):\n" + "\n".join(tail))


# ── the request ──────────────────────────────────────────────────────────────


def _sampling_dict(sampling) -> dict:
    return {
        "temperature": float(sampling.temperature),
        "top_p": float(sampling.top_p),
        "top_k": int(sampling.top_k),
        "repetition_penalty": float(sampling.repetition_penalty),
        "penalty_window": int(sampling.penalty_window),
        "min_tokens": int(sampling.min_tokens),
        "max_tokens": int(sampling.max_tokens),
    }


_SEMANTIC_DEFAULTS = SimpleNamespace(  # the checkpoint preset; only a placeholder for yue-plan
    temperature=1.0,
    top_p=0.95,
    top_k=100,
    repetition_penalty=1.2,
    penalty_window=50,
    min_tokens=200,
    max_tokens=9000,
)


def build_request(request, *, abc_sampling, semantic_sampling, steps: int, semantic_tokens=None):
    """yue2.cpp's request JSON from the app's ``SongRequest`` and two ``Sampling`` values.

    Field for field the same protocol: ``cot`` / ``abc`` / ``cfg_scale`` (``-1`` = protocol
    default, exactly like ``cfg_scale=None`` upstream), both seeds from the one request seed
    (upstream runs both stages from it), ``duration`` left at the semantic cap.
    """
    payload = {
        "style": request.style,
        "lyrics": request.lyrics,
        "abc": request.abc or "",
        "cot": request.cot,
        "lm_seed": int(request.seed),
        "seed": int(request.seed),
        "steps": int(steps),
        "cfg_scale": -1.0 if request.cfg_scale is None else float(request.cfg_scale),
        "output_format": "wav32",
        "abc_sampling": _sampling_dict(abc_sampling),
        "semantic_sampling": _sampling_dict(semantic_sampling),
    }
    if semantic_tokens is not None:
        payload["semantic_tokens"] = ",".join(str(int(t)) for t in semantic_tokens)
    return payload


# ── the child process ────────────────────────────────────────────────────────

# yue2.cpp labels its two AR phases "Score" and "Semantic"; the app calls them abc / semantic
_PHASE = {"Score": "abc", "Semantic": "semantic"}
_AR_STEP = re.compile(r"^\[AR\] (Score|Semantic) (\d+)/(\d+)$")
_AR_SONG = re.compile(r"^\[AR\] (Score|Semantic) song 0: (\d+) tokens( \(truncated\))?$")
_AR_DONE = re.compile(
    r"^\[AR\] (Score|Semantic): (\d+) tokens over \d+ songs, (\d+) steps, ([\d.]+) s \(([\d.]+) ms/step\)$"
)
_NAR_STEP = re.compile(r"^\[NAR\] Step (\d+)/(\d+), (\d+) ms$")
_VAE_DONE = re.compile(r"^\[VAE\] Tiled decode done: .* (\d+) ms$")
_VAE_ONE = re.compile(r"^\[VAE\] Decoded: .* (\d+) ms$")  # a song of one tile decodes untiled
_PIPE_DONE = re.compile(r"^\[Pipeline\] Done: .* in ([\d.]+) s")
_BACKEND = re.compile(r"^\[Load\] (?:LM|NAR|VAE|KV) backend: (\S+)")
_LOAD = re.compile(r"^\[Store\] Load (LM|NAR|VAE): (\d+) ms$")
_FATAL = re.compile(r"FATAL: (.*)$")


@dataclass
class ChildLog:
    """What the parser pulled out of a yue2.cpp stderr stream."""

    backend: str = ""
    timing: dict = field(default_factory=dict)
    truncated: dict = field(default_factory=lambda: {"abc": False, "semantic": False})
    tokens: dict = field(default_factory=lambda: {"abc": 0, "semantic": 0})
    fatal: str = ""
    lines: list[str] = field(default_factory=list)


def _advance(log: ChildLog, phase: str, count: int, on_token) -> None:
    """The AR stage reports every 100 tokens; ``on_token(phase, None, count=stride)`` once per report."""
    delta = count - log.tokens[phase]
    if delta > 0:
        log.tokens[phase] = count
        if on_token is not None:
            on_token(phase, None, count=delta)


def parse_line(
    line: str, log: ChildLog, *, on_token=None, on_progress: StageProgress | None = None
):
    """Fold one stderr line into *log*, forwarding progress to the app's callbacks.

    ``on_token(phase, token, count=)`` is the app's per-token callback with a stride: the
    child prints ``[AR] Score 100/4096`` every 100 tokens, not one line per token."""
    log.lines = (log.lines + [line])[-40:]
    m = _BACKEND.match(line)
    if m and not log.backend:
        log.backend = m.group(1)
        return
    m = _AR_STEP.match(line)
    if m:
        _advance(log, _PHASE[m.group(1)], int(m.group(2)), on_token)
        return
    m = _AR_SONG.match(line)
    if m:
        phase = _PHASE[m.group(1)]
        log.truncated[phase] = bool(m.group(3))
        _advance(log, phase, int(m.group(2)), on_token)
        return
    m = _AR_DONE.match(line)
    if m:
        label, count, steps, seconds, ms = m.groups()
        log.timing[_PHASE[label]] = {
            "seconds": float(seconds),
            "output_tokens": int(count),
            "steps": int(steps),
            "ms_per_step": float(ms),
            "output_tps": int(count) / float(seconds) if float(seconds) > 0 else 0.0,
        }
        return
    m = _NAR_STEP.match(line)
    if m:
        done, total, ms = int(m.group(1)), int(m.group(2)), int(m.group(3))
        log.timing["nar_seconds"] = log.timing.get("nar_seconds", 0.0) + ms / 1000
        if on_progress is not None:
            on_progress("nar", done, total)
        return
    m = _VAE_DONE.match(line) or _VAE_ONE.match(line)
    if m:
        log.timing["vae_seconds"] = int(m.group(1)) / 1000
        if on_progress is not None:
            on_progress("vae", 1, 1)
        return
    m = _LOAD.match(line)
    if m:
        log.timing.setdefault("load", {})[f"{m.group(1).lower()}_load_seconds"] = (
            int(m.group(2)) / 1000
        )
        return
    m = _PIPE_DONE.match(line)
    if m:
        log.timing["e2e_seconds"] = float(m.group(1))
        return
    m = _FATAL.search(line)
    if m and not log.fatal:
        log.fatal = m.group(1).strip()


def run_child(
    argv: list[str],
    *,
    cancelled=None,
    on_token=None,
    on_progress: StageProgress | None = None,
    cwd: Path | None = None,
) -> ChildLog:
    """Run a yue2.cpp binary, streaming its stderr through :func:`parse_line`.

    ``cancelled()`` is polled twice a second; a cancel terminates the child and raises
    ``InterruptedError`` (the app treats that as "cancelled", not "failed").
    """
    log = ChildLog()
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        env=engine_env(),
        **config.SUBPROCESS_TEXT,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def pump():  # the reader only queues; parsing (and the callbacks) stay on the caller's thread
        assert proc.stderr is not None
        for raw in proc.stderr:
            lines.put(raw.rstrip("\r\n"))
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    try:
        while True:
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                line = ""
            else:
                if line is None:
                    break
            if line:
                parse_line(line, log, on_token=on_token, on_progress=on_progress)
            if cancelled is not None and cancelled() and proc.poll() is None:
                raise InterruptedError("yue2.cpp stopped")
        code = proc.wait()
    except BaseException:
        # whatever left the loop early — a cancel, a callback that raised, a KeyboardInterrupt —
        # the child must not outlive it (the next generation would compete with it for VRAM)
        _stop_child(proc)
        raise
    finally:
        with contextlib.suppress(OSError):
            if proc.stderr is not None:
                proc.stderr.close()
    if code != 0:
        detail = log.fatal or "\n".join(log.lines[-8:])
        raise RuntimeError(f"{Path(argv[0]).name} failed (exit {code}): {detail}")
    return log


def _stop_child(proc: subprocess.Popen) -> None:
    """Terminate, then kill, and always reap: a stopped child is never left as a zombie."""
    if proc.poll() is not None:
        return
    with contextlib.suppress(OSError):
        proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(10)


# ── paths the child sees ─────────────────────────────────────────────────────
# yue2.cpp opens files with fopen / CreateFileA: on Windows that is the ANSI code page, so a
# path with characters outside it (a user name in another script, an emoji) cannot be opened.
# Two measures: the working files live next to the GGUF files (one root to keep ASCII), and
# on Windows a non-ASCII path is handed over as its 8.3 short name when the volume has one.


def child_path(path: Path) -> str:
    text = str(path)
    if os.name != "nt" or text.isascii():
        return text
    try:
        import ctypes

        # the deepest existing ancestor gets shortened; an output file that does not exist
        # yet keeps its (ASCII) name under it
        existing, rest = Path(text), []
        while not existing.exists() and existing.parent != existing:
            rest.insert(0, existing.name)
            existing = existing.parent
        buffer = ctypes.create_unicode_buffer(1024)
        length = ctypes.windll.kernel32.GetShortPathNameW(str(existing), buffer, 1024)  # type: ignore[attr-defined]
        short = str(Path(buffer.value).joinpath(*rest)) if length else text
        if short.isascii():
            return short
    except Exception:  # noqa: BLE001, S110 — best effort; the error below names the path
        pass
    raise RuntimeError(
        f"yue2.cpp cannot open a path with non-ASCII characters on Windows: {text}\n"
        "Move the app (or set YUE2_GROOVE_GGUF) to a folder whose path is plain ASCII."
    )


@contextlib.contextmanager
def _work_directory(root: Path):
    """A throwaway directory under *root* (next to the GGUF files) for one child run."""
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="work-", dir=str(root)))
    try:
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ── the pipeline object the app holds ────────────────────────────────────────


@dataclass
class GgufPipeline:
    """Stands in for the loaded PyTorch pipeline in ``runtime._PIPE``.

    Nothing is resident: every generation is one ``yue-synth`` process (Metal pays a
    ~20 s shader compile per process; CUDA a second or two of loads from the page cache).
    ``adapter.generate / plan / decode / model_dtype / close_pipeline`` dispatch on this type.
    """

    backbone: Path
    vae: Path
    model_dir: Path
    vae_dir: Path
    quant: str
    ode_steps: int = 32
    vae_core_frames: int | None = None
    max_seq: int | None = None
    max_seq_reason: str = ""
    device: str = "gguf"  # replaced by the child's backend name (MTL0 / CUDA0 / …) after a run
    weights: dict = field(default_factory=dict)
    version: str = ""  # what the binary reports; compare with YUE2CPP_PIN
    binary_sha256: str = ""
    ready_made: bool = False  # external GGUF files: the local checkpoints are not their source
    _plan_ids: tuple = ("retokenized", None)  # set per generate(); read by config_dict()
    _budget_note: dict | None = None  # set per generate() when max_seq capped the semantic budget

    engine = "yue2.cpp"

    @classmethod
    def open(
        cls,
        *,
        model_dir: Path,
        vae_dir: Path,
        ode_steps: int,
        vae_core_frames: int | None,
        quant_label: str | None = None,
        log: Callable[[str], None] | None = None,
    ) -> GgufPipeline:
        synth = binary("yue-synth")  # fail early, with the install hint
        version = engine_version(synth)
        vram = cuda_total_vram_gib()
        cap, reason = effective_max_seq(round(vram) if vram is not None else None)
        if reason and log is not None:
            log(reason)
        if version and not version.startswith(YUE2CPP_PIN) and log is not None:
            log(f"note: installed yue2.cpp is {version}, the app was written against {YUE2CPP_PIN}")
        quant_label = quant_label or quant()
        if log is not None and not _hash_cache_path(gguf_dir()).is_file():
            log("Hashing the checkpoints (first time only)…")
        prepared = prepare(model_dir, vae_dir, quant_label=quant_label, log=log)
        backbone, vae = prepared.backbone, prepared.vae
        if log is not None and not backbone.with_name(backbone.name + ".sha256").is_file():
            log("Hashing the GGUF files (first time only)…")
        # what the GGUF files came from — only what was actually verified: the checkpoints'
        # hashed content for a conversion, an honest "unknown" for ready-made files
        external = {"provenance": "ready-made: external GGUF files, source not verified"}
        weights = {
            "mot": {
                "files": {
                    backbone.name: {
                        "sha256": sha256_file(backbone),
                        "bytes": backbone.stat().st_size,
                    }
                },
                "source": {"provenance": "converted", **prepared.model}
                if prepared.model
                else external,
            },
            "vae": {
                "files": {vae.name: {"sha256": sha256_file(vae), "bytes": vae.stat().st_size}},
                "source": {"provenance": "converted", **prepared.vae_source}
                if prepared.vae_source
                else external,
            },
        }
        return cls(
            backbone=backbone,
            vae=vae,
            model_dir=Path(model_dir),
            vae_dir=Path(vae_dir),
            quant=quant_label,
            ode_steps=int(ode_steps),
            vae_core_frames=vae_core_frames,
            max_seq=cap,
            max_seq_reason=reason,
            weights=weights,
            version=version,
            binary_sha256=sha256_file(synth, cache=False),
            ready_made=prepared.provenance == "ready-made",
        )

    def _workdir(self):
        return _work_directory(self.backbone.parent / ".work")

    # the attributes the tabs read off the PyTorch pipeline
    @property
    def dtype_label(self) -> str:
        return f"{self.quant.lower()} (gguf)"

    def _common_flags(self) -> list[str]:
        flags = []
        if self.max_seq:
            flags += ["--max-seq", str(self.max_seq)]
        return flags

    def _synth_flags(self) -> list[str]:
        flags = self._common_flags()
        if self.vae_core_frames:
            flags += ["--vae-core", str(self.vae_core_frames)]
        return flags

    def config_dict(self, request, abc_sampling, semantic_sampling) -> dict:
        """``config.json`` for a GGUF run: the keys the Library reads, plus ``engine``."""
        guidance = request.guidance
        return {
            "generation": {
                "abc": _sampling_dict(abc_sampling),
                "semantic": _sampling_dict(semantic_sampling),
                "ode_steps": self.ode_steps,
                "ode_method": "midpoint",
                "context": CONTEXT,
                "version": "yue2-native-v1",
            },
            "overrides": {} if request.cfg_scale is None else {"cfg_scale": guidance},
            "cot": request.cot,
            "cfg_scale": guidance,
            "cfg_negative": "instruction_only"
            if request.cot == "off"
            else "same_instruction_and_exact_abc",
            "backend": "gguf",
            "quantization": self.quant,
            "model_dtype": self.dtype_label,
            "vae_dtype": "float32",
            "vae_decode": "halo_crop",
            "vae_core_frames": self.vae_core_frames or 512,
            "vae_halo_frames": 16,
            "device": self.device,
            "memory_budget_gib": None,
            "max_seq": self.max_seq,
            "max_seq_reason": self.max_seq_reason or None,
            "semantic_budget_cap": self._budget_note,
            "offload_ar": False,
            "engine": {
                "name": "yue2.cpp",
                "pin": YUE2CPP_PIN,
                "version": self.version or None,  # what the binary itself reported
                "binary_sha256": self.binary_sha256 or None,
                "backbone_gguf": self.backbone.name,
                "vae_gguf": self.vae.name,
            },
            "plan_ids_provenance": self._plan_ids[0],  # what this run actually recorded
            "plan_ids_match_retokenized": self._plan_ids[1],
            "decoder_release": self._decoder_release(),
            "validation_status": "unvalidated",
        }

    def _decoder_release(self):
        """``release_variant`` of the VAE checkpoint, as upstream records it (the listening page shows it)."""
        if self.ready_made:
            return None  # the local checkpoint is not what the external VAE GGUF came from
        try:
            return json.loads((self.vae_dir / "config.json").read_text(encoding="utf-8")).get(
                "release_variant"
            )
        except (OSError, ValueError):
            return None

    def generate(
        self,
        request,
        *,
        abc_sampling,
        semantic_sampling,
        cancelled=None,
        on_token=None,
        on_progress: StageProgress | None = None,
    ) -> GgufSong:
        from . import adapter  # tokenizer + plan objects come through the one yue2 door

        tokenizer = adapter.text_tokenizer(self.model_dir)
        self._budget_note = None
        if self.max_seq:
            prefix_tokens, capped = semantic_budget(
                request, tokenizer, abc_sampling, semantic_sampling, self.max_seq
            )
            if capped < int(semantic_sampling.max_tokens):
                semantic_sampling = dataclasses.replace(
                    semantic_sampling,
                    max_tokens=capped,
                    min_tokens=min(int(semantic_sampling.min_tokens), capped),
                )
                self._budget_note = {
                    "max_seq": self.max_seq,
                    "prefix_tokens_assumed": prefix_tokens,
                    "semantic_max_tokens": capped,
                }
        payload = build_request(
            request,
            abc_sampling=abc_sampling,
            semantic_sampling=semantic_sampling,
            steps=self.ode_steps,
        )
        started = time.perf_counter()
        with self._workdir() as work:
            (work / "request.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            argv = [
                str(binary("yue-synth")),
                "--model",
                child_path(self.backbone),
                "--vae",
                child_path(self.vae),
                "--request",
                child_path(work / "request.json"),
                "--out",
                child_path(work / "audio.wav"),
                "--tokens",
                child_path(work / "tokens.csv"),
                "--latent",
                child_path(work / "latent.vae"),
                "--score",
                child_path(work / "score.abc"),
                *self._synth_flags(),
            ]
            exact = exact_ids_wanted()
            if exact:
                (work / "dump").mkdir()  # yue2.cpp writes into it but does not create it
                argv += ["--dump", child_path(work / "dump")]
            log = run_child(argv, cancelled=cancelled, on_token=on_token, on_progress=on_progress)
            if log.backend:
                self.device = log.backend
            audio = _read_wav32(work / "audio.wav")
            tokens = [
                int(t) for t in (work / "tokens.csv").read_text().strip().split(",") if t.strip()
            ]
            latents = np.fromfile(work / "latent.vae", dtype=np.float32).reshape(-1, 64)
            score = (
                (work / "score.abc").read_text(encoding="utf-8")
                if (work / "score.abc").is_file()
                else None
            )
            engine_ids = _engine_ar_ids(work / "dump" / "ar_ids.bin", tokens) if exact else None
        if request.cot == "off":
            score = None
        elif request.abc is not None:
            score = request.abc  # the exact text the prefix was built from
        else:
            score = score or ""  # the model wrote nothing: an empty plan, still a plan
        abc_ids = tokenizer.encode(score) if score is not None else []
        prefix = adapter.token_prefixes(request, tokenizer, abc_ids if score is not None else None)
        ids_match = None
        if engine_ids is not None:
            # the ids the engine actually sampled and fed to the semantic stage win over the
            # re-tokenization of their text (identical in every archived run, but this is the truth)
            engine_prefix, engine_abc = engine_ids
            ids_match = engine_prefix == prefix
            if not ids_match:
                prefix, abc_ids = engine_prefix, engine_abc
        self._plan_ids = ("engine", ids_match) if engine_ids is not None else ("retokenized", None)
        plan_timing = (
            {"seconds": 0.0, "output_tokens": 0, "external_prefix_tokens": len(abc_ids)}
            if request.abc is not None
            else log.timing.get("abc", {"seconds": 0.0, "output_tokens": 0})
        )
        plan = adapter.symbolic_plan(
            request, score, abc_ids, prefix, plan_timing, log.truncated["abc"]
        )
        timing = {
            "abc": plan_timing,
            "semantic": log.timing.get("semantic", {"seconds": 0.0, "output_tokens": len(tokens)}),
            "nar_seconds": log.timing.get("nar_seconds", 0.0),
            "vae_seconds": log.timing.get("vae_seconds", 0.0),
            "load": log.timing.get("load", {}),
            "e2e_seconds": time.perf_counter() - started,
            "engine_seconds": log.timing.get("e2e_seconds"),
        }
        cfg = self.config_dict(request, abc_sampling, semantic_sampling)
        identity = adapter.identity(
            {"request": request.to_dict(), "config": cfg, "weights": self.weights}
        )
        return GgufSong(
            audio=audio,
            plan=plan,
            tokens=tokens,
            latents=latents,
            config=cfg,
            weights=self.weights,
            timing=timing,
            request_identity=identity,
            truncated={"abc": log.truncated["abc"], "semantic": log.truncated["semantic"]},
            child_tail=list(log.lines),
        )

    def plan(self, request, *, abc_sampling, cancelled=None):
        from . import adapter

        tokenizer = adapter.text_tokenizer(self.model_dir)
        if request.cot == "off":
            return adapter.symbolic_plan(
                request, None, [], adapter.token_prefixes(request, tokenizer), {}, False
            )
        if request.abc is not None:
            ids = tokenizer.encode(request.abc)
            return adapter.symbolic_plan(
                request,
                request.abc,
                ids,
                adapter.token_prefixes(request, tokenizer, ids),
                {"seconds": 0.0, "output_tokens": 0, "external_prefix_tokens": len(ids)},
                False,
            )
        payload = build_request(
            request,
            abc_sampling=abc_sampling,
            semantic_sampling=_SEMANTIC_DEFAULTS,  # yue-plan never reaches the semantic stage
            steps=self.ode_steps,
        )
        with self._workdir() as work:
            (work / "request.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            argv = [
                str(binary("yue-plan")),
                "--model",
                child_path(self.backbone),
                "--request",
                child_path(work / "request.json"),
                "--out",
                child_path(work / "score.abc"),
                *self._common_flags(),
            ]
            log = run_child(argv, cancelled=cancelled)
            score = (work / "score.abc").read_text(encoding="utf-8")
        ids = tokenizer.encode(score)
        timing = log.timing.get("abc", {"seconds": 0.0, "output_tokens": len(ids)})
        return adapter.symbolic_plan(
            request,
            score,
            ids,
            adapter.token_prefixes(request, tokenizer, ids),
            timing,
            log.truncated["abc"],
        )

    def decode(self, latents, *, full=False, vae=None, on_progress=None):
        """``[T,64]`` latents → ``[N,2]`` float audio through ``neural-codec --decode``."""
        if vae is not None:
            raise RuntimeError(
                "the GGUF engine decodes with its own VAE GGUF; VAE overrides need the torch backend"
            )
        with self._workdir() as work:
            np.ascontiguousarray(np.asarray(latents, dtype=np.float32)).tofile(work / "latent.vae")
            argv = [
                str(binary("neural-codec")),
                "--vae",
                child_path(self.vae),
                "--decode",
                "-i",
                child_path(work / "latent.vae"),
                "-o",
                child_path(work / "audio.wav"),
                "--format",
                "wav32",
            ]
            if not full and self.vae_core_frames:
                argv += ["--vae-core", str(self.vae_core_frames)]
            if full:
                argv += ["--vae-core", str(max(1, int(np.asarray(latents).shape[0])))]
            run_child(argv)
            audio = _read_wav32(work / "audio.wav")
        if on_progress is not None:
            on_progress(1, 1)
        return audio

    def close(self) -> None:
        return None


def _engine_ar_ids(dump: Path, tokens: list[int]) -> tuple[list[int], list[int]]:
    """``(prefix, abc_ids)`` from yue2.cpp's ``ar_ids.bin``.

    The engine dumps the sequence the *first acoustic chunk* attends to: the prefix, that
    chunk's codes (all of them for a one-chunk song, a leading slice otherwise) and
    ``MUSIC_END``.  Parsed by the protocol's markers and checked against the returned
    semantic stream; exact ids were asked for, so anything short of them is an error.
    """
    import struct

    from . import adapter

    if not dump.is_file():
        raise RuntimeError(
            "YUE2_GROOVE_GGUF_EXACT_IDS is set but yue2.cpp wrote no ar_ids.bin (see the log above)"
        )
    raw = np.fromfile(dump, dtype=np.float32)
    ndim = struct.unpack("i", struct.pack("f", raw[0]))[0]
    ids = [int(v) for v in raw[1 + ndim :]]
    p = adapter.protocol_ids()
    if p["MUSIC_START"] not in ids or not ids or ids[-1] != p["MUSIC_END"]:
        raise RuntimeError(
            "yue2.cpp's ar_ids.bin does not follow the prefix / codes / MUSIC_END layout"
        )
    cut = ids.index(p["MUSIC_START"]) + 1
    prefix, chunk = ids[:cut], ids[cut:-1]
    codes = [c - p["CODEC_OFFSET"] for c in chunk]
    if not chunk or codes != tokens[: len(codes)]:
        raise RuntimeError(
            "yue2.cpp's ar_ids.bin codes do not match the semantic stream it returned "
            f"({len(codes)} dumped, {len(tokens)} returned)"
        )
    if p["ABC_START"] in prefix and p["ABC_END"] in prefix:
        abc_ids = prefix[prefix.index(p["ABC_START"]) + 1 : prefix.index(p["ABC_END"])]
    else:
        abc_ids = []
    return prefix, abc_ids


def _read_wav32(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise RuntimeError(f"unexpected sample rate {rate} from yue2.cpp")
    return np.clip(audio, -1.0, 1.0)  # upstream clamps its decoder output the same way


@dataclass
class GgufSong:
    """The ``SongResult`` shape ``runtime.run_generation`` consumes."""

    audio: np.ndarray
    plan: Any  # upstream's SymbolicPlan (adapter.symbolic_plan)
    tokens: list[int]
    latents: np.ndarray
    config: dict
    weights: dict
    timing: dict
    request_identity: str
    truncated: dict
    child_tail: list[str]
    sample_rate: int = SAMPLE_RATE

    @property
    def abc(self):
        return self.plan.abc

    def save_artifacts(self, directory):
        import soundfile as sf

        from . import adapter

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.plan.save(directory)
        sf.write(directory / "audio.flac", self.audio, self.sample_rate, subtype="PCM_24")
        np.save(directory / "semantic.npy", np.asarray(self.tokens, dtype=np.int32))
        np.save(directory / "latent.npy", self.latents.astype(np.float32))
        _write_json(directory / "request.json", self.plan.request.to_dict())
        _write_json(directory / "config.json", self.config)
        result = {
            "status": "complete",
            "identity": self.request_identity,
            "engine": self.config.get("engine"),
            "truncated": self.truncated,
            "sample_rate": self.sample_rate,
            "audio_seconds": len(self.audio) / self.sample_rate,
            "weights": self.weights,
            "timing": self.timing,
            "artifacts": adapter.collect_hashes(directory),
        }
        _write_json(directory / "result.json", result)
        return result


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def local_env(pipe: GgufPipeline) -> dict:
    """The ``local_env.json`` sidecar fields for a GGUF run."""
    return {
        "engine": "yue2.cpp",
        "engine_pin": YUE2CPP_PIN,
        "quantization": pipe.quant,
        "backbone_gguf": pipe.backbone.name,
        "platform": platform.platform(),
    }


# ── installing the binaries / preparing the models from the command line ─────
# `python -m yue2_groove.gguf_engine install` fetches this platform's yue2cpp-<pin>-<platform>
# asset from the app's GitHub release into <repo>/bin/yue2cpp (what the launcher's install
# step calls); `prepare` converts + quantizes the GGUF files ahead of the first generation;
# `check` prints what would be used.

RELEASES = "https://github.com/deadjoe/yue2_groove/releases"
RELEASES_API = "https://api.github.com/repos/deadjoe/yue2_groove/releases"


def platform_asset() -> str:
    """The release asset for this machine, or a RuntimeError naming why there is none."""
    system, machine = platform.system(), platform.machine().lower()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return f"yue2cpp-{YUE2CPP_PIN}-macos-arm64-metal.tar.gz"
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return f"yue2cpp-{YUE2CPP_PIN}-linux-x64.tar.gz"
    if system == "Windows" and machine in ("amd64", "x86_64"):
        return f"yue2cpp-{YUE2CPP_PIN}-windows-x64.zip"
    raise RuntimeError(f"no prebuilt yue2.cpp for {system}/{machine}; build it from {YUE2CPP_REPO}")


def app_release_tag() -> str:
    try:
        from importlib.metadata import version

        return "v" + version("yue2-groove")
    except Exception:  # noqa: BLE001 — not installed as a distribution
        return "latest"


def release_with_asset(asset: str) -> str | None:
    """The newest release that lists *asset*, or None (no such release, or no API).

    The asset name carries the yue2.cpp pin, so any release listing it has the right binaries:
    a release just published has none for the first two hours (the workflow is still building),
    and an install run in that window takes them from the release before.  ``GH_TOKEN`` /
    ``GITHUB_TOKEN``, when set, lifts the API's anonymous rate limit (CI builds).
    """
    request = urllib.request.Request(
        f"{RELEASES_API}?per_page=30",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "yue2-groove"},
    )
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            releases = json.loads(response.read().decode("utf-8"))
        for release in releases:  # newest first
            if any(item.get("name") == asset for item in release.get("assets") or ()):
                return release.get("tag_name") or None
    except (OSError, ValueError, AttributeError, TypeError):  # offline, rate-limited, odd JSON
        return None
    return None


def installed_version(directory: Path) -> str:
    """The commit named in an install's VERSION file (``yue2.cpp <commit> <platform> …``), or ``""``."""
    try:
        words = (directory / "VERSION").read_text(encoding="utf-8").split()
        return words[1] if len(words) > 1 and words[0] == "yue2.cpp" else ""
    except OSError:
        return ""


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def install_binaries(
    dest: Path | None = None, *, tag: str | None = None, say=None, force: bool = False
) -> Path:
    """Download and unpack this platform's binaries; returns the directory.

    Idempotent: an install already at ``YUE2CPP_PIN`` is kept (the launcher runs this on every
    update — no 700 MB download for nothing); ``force`` replaces it anyway."""
    import io
    import stat
    import tarfile
    import zipfile

    say = say or (lambda _m: None)
    dest = Path(dest or (repo_root() / "bin" / "yue2cpp"))
    if not force and installed_version(dest) == YUE2CPP_PIN and _binary(dest, "yue-synth"):
        say(f"already installed: yue2.cpp {YUE2CPP_PIN} in {dest}")
        return dest
    asset = platform_asset()
    tag = tag or app_release_tag()
    url = (
        f"{RELEASES}/latest/download/{asset}"
        if tag == "latest"
        else f"{RELEASES}/download/{tag}/{asset}"
    )
    say(f"downloading {url}")
    try:
        data = _fetch(url)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise RuntimeError(f"{url}: HTTP {exc.code}") from exc
        # not on that release (yet): any release listing the pinned asset name will do
        other = release_with_asset(asset)
        if other is None or other == tag:
            raise RuntimeError(
                f"{url}: HTTP 404 — no release lists {asset}. The binaries are attached to a "
                "release about two hours after it is published (and releases before the GGUF "
                "engine carry none); try again later, or `--tag` a release that lists them."
            ) from exc
        url = f"{RELEASES}/download/{other}/{asset}"
        say(f"not on {tag} (yet) — taking {asset} from {other}: {url}")
        data = _fetch(url)
    if (dest / "VERSION").is_file():  # ours from an earlier install: no stale libraries left behind
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            archive.extractall(dest)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            try:
                archive.extractall(dest, filter="data")
            except TypeError:  # Python < 3.12 (3.10 / 3.11 without the backport)
                archive.extractall(dest)
    if os.name != "nt":
        for name in (*BINARIES, "yue-server", "yue-transcribe", "mp3-codec"):
            path = dest / name
            if path.is_file():
                path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    version_file = dest / "VERSION"
    say(version_file.read_text().strip() if version_file.is_file() else f"unpacked into {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    import argparse

    config.load_env()  # the same .env the app reads: YUE2_GROOVE_YUE2CPP / _GGUF / _MODELS …
    parser = argparse.ArgumentParser(
        prog="python -m yue2_groove.gguf_engine", description="GGUF engine (yue2.cpp) helpers"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inst = sub.add_parser("install", help="download this platform's yue2.cpp binaries")
    inst.add_argument("--dest", default=None, help="target directory (default <repo>/bin/yue2cpp)")
    inst.add_argument(
        "--tag", default=None, help="app release tag (default: this version; or latest)"
    )
    inst.add_argument(
        "--force", action="store_true", help="replace an install that is already at the pin"
    )
    prep = sub.add_parser("prepare", help="convert + quantize the GGUF files from the checkpoints")
    prep.add_argument("--model", default=config.default_model())
    prep.add_argument("--vae", default=config.default_vae())
    prep.add_argument("--quant", default=None, choices=list(QUANTS))
    sub.add_parser("check", help="print what the engine would use")
    args = parser.parse_args(argv)

    say = lambda message: print(message, file=sys.stderr, flush=True)  # noqa: E731, T201
    if args.command == "install":
        install_binaries(
            Path(args.dest) if args.dest else None, tag=args.tag, say=say, force=args.force
        )
        return 0
    if args.command == "prepare":
        from . import adapter

        model_dir = Path(adapter.resolve_model(args.model))
        vae_dir = Path(adapter.resolve_model(args.vae))
        prepared = prepare(model_dir, vae_dir, quant_label=args.quant, log=say)
        backbone = prepared.backbone
        say(
            f"ready ({prepared.provenance}): {backbone} ({backbone.stat().st_size / 1e9:.2f} GB), {prepared.vae}"
        )
        return 0
    directory = binary_dir()
    say(
        f"binaries: {directory or 'not found'}"
        + (f" ({INSTALL_HINT})" if directory is None else "")
    )
    say(f"gguf dir: {gguf_dir()}  quant: {quant()}  max_seq: {max_seq() or 'full context'}")
    vram = cuda_total_vram_gib()
    backend, note = auto_backend("cuda" if vram is not None else "other", vram)
    say(f"auto backend: {backend}" + (f" — {note}" if note else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
