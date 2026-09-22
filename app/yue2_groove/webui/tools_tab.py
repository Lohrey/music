"""05 TOOLS handlers: ABC checks, chord stripping, the doctor, the listening page."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import gradio as gr

from .. import adapter, config
from ..vendor import abc_tools
from . import runtime

# ─────────────────────────── tools ───────────────────────────


def abc_inspect(text):
    if not (text or "").strip():
        raise gr.Error("Paste an ABC score first")
    try:
        tools = abc_tools
        report = tools.report(tools.parse_abc(text))
    except ValueError as exc:
        raise gr.Error(f"Invalid ABC: {exc}") from exc
    return json.dumps(report, ensure_ascii=False, indent=2, default=tools.json_value)


def abc_strip_chords(text, keep_voice):
    if not (text or "").strip():
        raise gr.Error("Paste an ABC score first")
    try:
        return abc_tools.strip_chords(text, keep_voice=keep_voice)
    except ValueError as exc:
        raise gr.Error(f"Processing failed: {exc}") from exc


def abc_compare(before, after, voices, allow_tempo):
    tools = abc_tools
    if not (before or "").strip() or not (after or "").strip():
        raise gr.Error("Provide both the original and edited ABC")
    try:
        result = tools.compare(
            tools.parse_abc(before),
            tools.parse_abc(after),
            names=tools.VOICES if voices == "both" else (voices,),
            allow_tempo_change=bool(allow_tempo),
        )
    except ValueError as exc:
        raise gr.Error(f"Comparison failed: {exc}") from exc
    return json.dumps(result, ensure_ascii=False, indent=2)


def run_doctor(model, vae_choice, vae_custom, revision, vae_revision, offline, verify):
    vae_path, _ = runtime.resolve_vae(vae_choice, vae_custom)
    cmd = adapter.doctor_command(
        model,
        vae_path,
        revision=revision,
        vae_revision=vae_revision,
        offline=bool(offline),
        verify=bool(verify),
    )
    res = subprocess.run(
        cmd,
        capture_output=True,
        timeout=1800,
        check=False,
        env=config.child_env(),
        **config.SUBPROCESS_TEXT,
    )
    return res.stdout.strip() or res.stderr.strip()


def make_comparison(paths_text, progress=gr.Progress()):
    sources = [p.strip() for p in (paths_text or "").splitlines() if p.strip()]
    if not sources:
        raise gr.Error("List one saved run directory per line")
    for src in sources:
        if not (Path(src) / "result.json").is_file():
            raise gr.Error(f"Not a valid YuE2 run directory (result.json missing): {src}")
    outdir = runtime.run_dir("comparison")
    progress(0.2, desc="Building listening comparison…")
    cmd = [sys.executable, "-m", "yue2_groove.vendor.listen", *sources, "--output", str(outdir)]
    res = subprocess.run(
        cmd,
        capture_output=True,
        timeout=1800,
        check=False,
        env=config.child_env(),
        **config.SUBPROCESS_TEXT,
    )
    if res.returncode not in (0, 1):
        raise gr.Error(f"Build failed: {res.stderr.strip()}")
    html_path = outdir / "index.html"
    served = f"/gradio_api/file={html_path}"
    link = f'<a href="{served}" target="_blank" rel="noopener">OPEN COMPARISON PAGE ↗</a>'
    status = (
        f"{res.stdout.strip()}\n"
        f"Click the link above to open it in a new tab, or paste one of:\n"
        f"  {served}\n"
        f"  file://{html_path}"
    )
    return str(html_path), link, status
