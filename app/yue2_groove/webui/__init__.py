"""YUE2 // GROOVE — unofficial Gradio web UI for YuE2 song generation.

Built on top of the official ``yue2`` CLI/Python API; every human-operable control of
the CLI/API is reachable from the browser so you never have to fall back to a terminal
for one parameter.

  01 GENERATE   style + lyrics (+ optional ABC) → editable score plan → 48 kHz stereo song;
                all sampling parameters of both the ABC and the semantic phase; plan mode
                full/melody/off, seed, cfg_scale, custom id, cancel, live progress; ALL MODES
                runs the same text request as full + melody + off and compares them
  02 COVER      source audio → SheetSage2 transcription (separate venv) → editable ABC →
                chord strip → SEND TO GENERATE or generate right on the tab
  03 EDIT       load a work as a frozen baseline, edit its ABC, check exact melody/meter
                invariants, regenerate from the edited score, compare baseline vs edit
  04 LIBRARY    every generated work: sort, select, rename, confirmed delete; details with a
                player (spectrum + transport), style / lyrics / ABC / score and run tables
  05 TOOLS      ABC validation / event export, chord stripping (cover melodies), edit
                invariant check, environment doctor, listening-comparison page
  06 DECODE     re-decode a saved latent.npy (source / standard / legacy / custom VAE,
                full or tiled) without generating again; single .npy upload supported
  07 BATCH      one JSON request per line (the equivalent of ``yue2 batch``), run in order
  Settings rail model & runtime: device / dtype / backend / quantization / offload_ar /
                memory budget / ODE steps / VAE core frames / revisions / offline; load & unload

Two views, one kernel.  **SONG** is the producer-facing director: it follows the
current work, states its stage (DRAFT / SCORE / AUDIO / REVISE / DONE) and offers
the next actions, all of which reuse the Studio handlers; its score is read-only
and it holds no editable component.  **STUDIO** is the full 7-tab gear room
above.  The switch is a class on ``<html>`` and both roots stay mounted, so the
score SVG and the player survive it (see ``docs/VIEW_SWITCH_PREFLIGHT.md``).

SheetSage2 (COVER) runs in its own virtual environment; set ``YUE2_GROOVE_SHEETSAGE_PYTHON``
to its interpreter (see README, 'Cover from audio').  Nothing in this process imports
``transformers``; the subprocess boundary lives in ``sheetsage_adapter.py``.

Apple Silicon (MPS): bfloat16 works with torch >= 2.11.  The upstream pin (torch 2.10.0)
hits pytorch/pytorch#174861 — the single-query SDPA kernel corrupts once the KV cache
passes 1024 tokens — so install with the override file in ``overrides/`` (see README) and
run ``scripts/mps_sdpa_check.py`` to verify.  vLLM / FP8 need NVIDIA CUDA.

Usage:
  python -m yue2_groove --port 7860             # opens the browser, SONG view
  python -m yue2_groove --view studio           # force the STUDIO view
  python -m yue2_groove --tab 1                 # force STUDIO on tab 0..6
  python -m yue2_groove --host 0.0.0.0 --auth user:pass   # LAN access (set a password)
  bash scripts/serve.sh start|stop|restart|status|log     # background service

The default view is SONG; ``YUE2_GROOVE_VIEW=song|studio`` or ``--view`` forces
it for a launch, and the last choice is remembered per browser otherwise.

Visual language: Bearbone Design System v0.2 (warm near-black ground family + ivory ink,
1px strokes, no shadows/gradients, monospace), with a dark and a bright scene.

Model weights are CC BY-NC 4.0 (non-commercial); this UI is not affiliated with the
YuE2 authors.
"""

from __future__ import annotations

# import order follows the dependency direction: kernel, page, tabs, wiring, CLI
from . import (
    cli,
    cover_tab,
    edit_tab,
    frontend,
    generate_tab,
    layout,
    library_tab,
    runtime,
    song_view,
    theme,
    tools_tab,
)
from .cli import main
from .layout import build_ui

__all__ = [
    "build_ui",
    "cli",
    "cover_tab",
    "edit_tab",
    "frontend",
    "generate_tab",
    "layout",
    "library_tab",
    "main",
    "runtime",
    "song_view",
    "theme",
    "tools_tab",
]
