"""Paths and defaults, overridable through environment variables.

* ``YUE2_GROOVE_RUNS``    directory for generated works (default: ``./runs`` in the
                          current working directory — the service script runs from the
                          repository root, so that is ``<repo>/runs``)
* ``YUE2_GROOVE_MODEL``   Hugging Face id or local directory of the 3B model
* ``YUE2_GROOVE_VAE``     Hugging Face id or local directory of the listening decoder
* ``YUE2_GROOVE_VAE_LEGACY``  same for the benchmark decoder
* ``YUE2_GROOVE_MODELS``  optional directory holding ``YuE2-3B`` / ``YuE2-Vae`` /
                          ``YuE2-Vae-legacy`` / ``SheetSage2`` subdirectories (e.g. the
                          ``models/`` folder of a YuE checkout); a subdirectory that exists
                          wins over the id

SheetSage2 (audio → ABC transcription) runs in its own virtual environment because its
pinned torch/transformers conflict with YuE2's.  These settings tell the UI where that
environment is; the environment itself is optional — without it the COVER tab reports a
clear configuration error and the rest of the UI keeps working.

* ``YUE2_GROOVE_SHEETSAGE_PYTHON``  python interpreter of the SheetSage2 venv (required
                                    to transcribe; e.g. ``../YuE/.venv-sheetsage2/bin/python``)
* ``YUE2_GROOVE_SHEETSAGE_MODEL``   Hugging Face id or local snapshot directory
                                    (default: ``m-a-p/SheetSage2``)
* ``YUE2_GROOVE_SHEETSAGE_BASE_MODEL``  optional local path of the parent MERT-v2-FullSong
                                    snapshot for offline adapter loads
* ``YUE2_GROOVE_SHEETSAGE_DEVICE``  device for transcription: ``auto`` / ``cuda`` / ``mps`` /
                                    ``cpu`` (default: ``auto``)
* ``YUE2_GROOVE_SHEETSAGE_KEEP_WARM``  ``1`` keeps a resident SheetSage2 worker alive between
                                    transcriptions (faster repeats, holds the model in memory;
                                    default: off / one process per transcription)
* ``YUE2_GROOVE_SHEETSAGE_IDLE_SECONDS``  how long a resident worker may sit idle before the
                                    background reaper stops it (default: 900)
* ``YUE2_GROOVE_TRANSCRIPTIONS``    where transcription outputs are written (default:
                                    ``<runs>/transcriptions``)

The GGUF engine (``yue2_groove.gguf_engine``: YuE2 through yue2.cpp, for cards under
16 GB) is selected and located by:

* ``YUE2_GROOVE_BACKEND``       ``auto`` (default: the VRAM rule) / ``torch`` / ``torch-eager`` /
                                ``gguf``; the same as ``--backend``
* ``YUE2_GROOVE_GGUF_VRAM_GIB`` the rule's threshold: a CUDA card below it gets ``gguf`` (16)
* ``YUE2_GROOVE_YUE2CPP``       directory of the yue2.cpp binaries (default ``<repo>/bin/yue2cpp``,
                                then PATH)
* ``YUE2_GROOVE_GGUF``          directory of the GGUF files (default ``<repo>/models/gguf``, or
                                ``$YUE2_GROOVE_MODELS/gguf``); prepared there on first use
* ``YUE2_GROOVE_GGUF_QUANT``    ``Q8_0`` (default) / ``Q6_K`` / ``Q5_K_M`` / ``BF16``
* ``YUE2_GROOVE_GGUF_MAX_SEQ``  KV-cache cap (``--max-seq``); cards of 8 GB or less get 12288 by
                                default, the semantic budget is trimmed to fit
* ``YUE2_GROOVE_GGUF_EXACT_IDS`` ``1`` records the score ids the engine sampled (from its debug
                                dump) instead of re-tokenizing the score text

The repository-root ``.env`` (see README) is also read by the app itself via
:func:`load_env`, before the CLI defaults are resolved — so a direct
``python -m yue2_groove`` gets the same settings as ``scripts/serve.sh``, including
the ``PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO`` memory guard.  An already-exported
value always wins over the file.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
EXAMPLES_DIR = PACKAGE_DIR / "examples"


def static_text(name: str) -> str:
    """A bundled frontend file (stylesheet / script) as text."""
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def load_env(path=None, environ=None) -> int:
    """Load a repo-root ``.env`` into the process environment.

    ``scripts/serve.sh`` sources ``.env`` before launching, but a direct
    ``python -m yue2_groove`` / ``yue2-groove`` does not.  Loading it here makes
    ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` (the MPS memory guard that keeps the
    allocator from driving the whole machine into swap) and ``YUE2_GROOVE_*``
    settings apply to every launch.  Existing environment variables are never
    overridden, so ``serve.sh`` and the command line still win.

    Returns the number of keys set.  Missing file, unreadable file and malformed
    lines are ignored.
    """
    target = os.environ if environ is None else environ
    if path is None:
        candidates = [Path.cwd() / ".env", PACKAGE_DIR.parent / ".env"]
        env_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if env_path is None:
            return 0
    else:
        env_path = Path(path).expanduser()
    try:
        text = env_path.read_text(encoding="utf-8").lstrip("\ufeff")
    except OSError:
        return 0
    count = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in target:
            target[key] = value
            count += 1
    return count


# Text-mode keyword arguments for every child process the app spawns; pair with
# :func:`child_env`.  Without an explicit encoding, Python on Windows decodes a
# pipe with the ANSI code page, which cannot hold most non-ASCII paths or ABC text.
SUBPROCESS_TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}


def child_env() -> dict[str, str]:
    """Environment for child interpreters: the current one plus UTF-8 stdio.

    The SheetSage2 driver and ``yue2 doctor`` print JSON with ``ensure_ascii=False``
    (paths, warnings).  A child Python on Windows encodes stdout with the ANSI code
    page unless told otherwise, so a path outside that page raises
    ``UnicodeEncodeError`` in the child before a byte reaches us.  Pinning both ends
    — the child here, the parent through :data:`SUBPROCESS_TEXT` — keeps the pipe
    readable on every platform and under every locale.
    """
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


HUB_MODEL = "m-a-p/YuE2-3B"
HUB_VAE = "m-a-p/YuE2-Vae"
HUB_VAE_LEGACY = "m-a-p/YuE2-Vae-legacy"
HUB_SHEETSAGE = "m-a-p/SheetSage2"


def runs_dir() -> Path:
    return Path(os.environ.get("YUE2_GROOVE_RUNS") or (Path.cwd() / "runs")).expanduser().resolve()


def _local_or_hub(subdir: str, hub_id: str, env: str) -> str:
    explicit = os.environ.get(env)
    if explicit:
        return explicit
    models = os.environ.get("YUE2_GROOVE_MODELS")
    if models:
        candidate = Path(models).expanduser() / subdir
        if candidate.is_dir():
            return str(candidate)
    return hub_id


def default_model() -> str:
    return _local_or_hub("YuE2-3B", HUB_MODEL, "YUE2_GROOVE_MODEL")


def default_vae() -> str:
    return _local_or_hub("YuE2-Vae", HUB_VAE, "YUE2_GROOVE_VAE")


def default_vae_legacy() -> str:
    return _local_or_hub("YuE2-Vae-legacy", HUB_VAE_LEGACY, "YUE2_GROOVE_VAE_LEGACY")


def sheetsage_python() -> str:
    """Interpreter of the separate SheetSage2 environment (empty when not configured)."""
    return (os.environ.get("YUE2_GROOVE_SHEETSAGE_PYTHON") or "").strip()


def default_sheetsage_model() -> str:
    return _local_or_hub("SheetSage2", HUB_SHEETSAGE, "YUE2_GROOVE_SHEETSAGE_MODEL")


def default_sheetsage_base_model() -> str:
    return (os.environ.get("YUE2_GROOVE_SHEETSAGE_BASE_MODEL") or "").strip()


def default_sheetsage_device() -> str:
    value = (os.environ.get("YUE2_GROOVE_SHEETSAGE_DEVICE") or "auto").strip().lower()
    return value if value in ("auto", "cuda", "mps", "cpu") else "auto"


def sheetsage_keep_warm() -> bool:
    """True when a resident SheetSage2 worker should be reused between runs."""
    value = (os.environ.get("YUE2_GROOVE_SHEETSAGE_KEEP_WARM") or "").strip().lower()
    return value in ("1", "true", "yes", "on")


def sheetsage_idle_seconds() -> float:
    """Idle lifetime of the resident worker before the background reaper stops it."""
    try:
        return max(0.0, float(os.environ.get("YUE2_GROOVE_SHEETSAGE_IDLE_SECONDS") or 900))
    except ValueError:
        return 900.0


def transcriptions_dir(base=None) -> Path:
    """Where transcription outputs go.

    ``YUE2_GROOVE_TRANSCRIPTIONS`` wins when set; otherwise ``<base>/transcriptions``,
    where *base* defaults to the configured runs directory.  The web UI passes its
    ``--runs`` override as *base* so both stay in sync.
    """
    override = os.environ.get("YUE2_GROOVE_TRANSCRIPTIONS")
    if override:
        return Path(override).expanduser().resolve()
    return (Path(base) if base is not None else runs_dir()) / "transcriptions"
