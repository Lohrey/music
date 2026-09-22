"""Suno-style edit tools for finished songs: every edit writes a NEW song (MP3).

Quick edits run with ffmpeg (seconds, no GPU): crop, remove section, reverse, speed,
fade in / out. AI edits use ACE-Step (extend, replace section, remaster) and Demucs
(stems: vocals + instrumental).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("yue2_groove")

SUFFIX = {
    "crop": "gekürzt",
    "remove": "ohne Teil",
    "reverse": "rückwärts",
    "speed": "Tempo",
    "fadein": "Fade-In",
    "fadeout": "Fade-Out",
    "extend": "verlängert",
    "replace": "neuer Teil",
    "remaster": "Remaster",
}


def ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    from . import youtube  # installs imageio-ffmpeg on first use

    youtube._ensure_tools()
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: list[str]) -> None:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.run(  # noqa: S603
        [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True, text=True, timeout=900, creationflags=flags,
    )
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg: " + (proc.stderr or "").strip()[-300:])


def _num(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _atempo_chain(factor: float) -> str:
    parts = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor /= 0.5
    parts.append(f"atempo={factor:.4f}")
    return ",".join(parts)


def quick_edit(op: str, src: Path, dst: Path, params: dict, duration: float) -> None:
    """ffmpeg edits. *dst* is an .mp3 path."""
    src, dst = str(src), str(dst)
    out = ["-c:a", "libmp3lame", "-b:a", "320k", dst]
    start = max(0.0, _num(params.get("start"), 0.0))
    end = _num(params.get("end"), duration)
    end = min(max(end, start + 0.5), duration or end)
    if op == "crop":
        length = end - start
        fade = min(0.05, length / 4)
        af = f"afade=t=in:d={fade},afade=t=out:st={max(0, length - fade)}:d={fade}"
        _run(["-ss", f"{start}", "-t", f"{length}", "-i", src, "-af", af, *out])
    elif op == "remove":
        if start <= 0.05 and end >= duration - 0.05:
            raise RuntimeError("Das wäre der ganze Song.")
        if start <= 0.05:
            _run(["-ss", f"{end}", "-i", src, "-af", "afade=t=in:d=0.05", *out])
        elif end >= duration - 0.05:
            _run(["-t", f"{start}", "-i", src, "-af", f"afade=t=out:st={max(0, start - 0.3)}:d=0.3", *out])
        else:
            xf = min(0.4, start / 2, (duration - end) / 2)
            fc = (
                f"[0:a]atrim=0:{start},asetpts=PTS-STARTPTS[a];"
                f"[0:a]atrim=start={end},asetpts=PTS-STARTPTS[b];"
                f"[a][b]acrossfade=d={xf}:c1=tri:c2=tri[o]"
            )
            _run(["-i", src, "-filter_complex", fc, "-map", "[o]", *out])
    elif op == "reverse":
        _run(["-i", src, "-af", "areverse", *out])
    elif op == "speed":
        factor = min(3.0, max(0.25, _num(params.get("factor"), 1.0)))
        if params.get("pitch"):  # like a record player: pitch follows the speed
            af = f"aresample=44100,asetrate={44100 * factor:.0f},aresample=44100"
            _run(["-i", src, "-af", af, *out])
        else:
            _run(["-i", src, "-af", _atempo_chain(factor), *out])
    elif op == "fadein":
        sec = min(max(0.2, _num(params.get("seconds"), 4.0)), max(0.5, duration))
        _run(["-i", src, "-af", f"afade=t=in:st=0:d={sec}", *out])
    elif op == "fadeout":
        sec = min(max(0.2, _num(params.get("seconds"), 6.0)), max(0.5, duration))
        _run(["-i", src, "-af", f"afade=t=out:st={max(0.0, duration - sec)}:d={sec}", *out])
    elif op == "master":  # loudness + gentle glue, used after an AI remaster
        af = "acompressor=threshold=-18dB:ratio=2:attack=20:release=250,loudnorm=I=-12:TP=-1:LRA=9"
        _run(["-i", src, "-af", af, *out])
    else:
        raise RuntimeError(f"Unbekannte Bearbeitung: {op}")


# ─────────────────────────────── stems (Demucs) ───────────────────────────────

_STEM_SCRIPT = r'''
import sys, numpy as np, soundfile as sf, torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
src, out_dir = sys.argv[1], sys.argv[2]
model = get_model("htdemucs"); model.eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)
import subprocess, os
wav_path = os.path.join(out_dir, "_in.wav")
subprocess.run([sys.argv[3], "-y", "-loglevel", "error", "-i", src, "-ac", "2", "-ar",
                str(model.samplerate), wav_path], check=True)
data, rate = sf.read(wav_path, dtype="float32", always_2d=True)
wav = torch.from_numpy(data.T.copy())
ref = wav.mean(0); wav = (wav - ref.mean()) / (ref.std() + 1e-8)
with torch.no_grad():
    sources = apply_model(model, wav[None], device=dev, shifts=1, split=True, overlap=0.25, progress=False)[0]
sources = sources * (ref.std() + 1e-8) + ref.mean()
names = model.sources
vocals = sources[names.index("vocals")]
rest = sum(sources[i] for i, n in enumerate(names) if n != "vocals")
sf.write(os.path.join(out_dir, "vocals.wav"), vocals.cpu().numpy().T, rate)
sf.write(os.path.join(out_dir, "instrumental.wav"), rest.cpu().numpy().T, rate)
print("ok")
'''


def _stem_python() -> str:
    """ACE-Step's venv (has CUDA torch + torchaudio); Demucs is added there without touching
    any installed package. Falls back to our own interpreter."""
    from . import ace_engine

    root = ace_engine.install_dir()
    if root is not None:
        py = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if py.is_file():
            return str(py)
    return sys.executable


def _ensure_demucs(py: str) -> None:
    check = subprocess.run([py, "-c", "import demucs.pretrained"], capture_output=True)  # noqa: S603
    if check.returncode == 0:
        return
    log.info("stems: installing demucs into %s", py)
    freeze = subprocess.run(  # noqa: S603
        [py, "-m", "pip", "freeze"], capture_output=True, text=True
    )
    pins = ""
    if freeze.returncode != 0:
        uvf = shutil.which("uv")
        if uvf:
            freeze = subprocess.run([uvf, "pip", "freeze", "--python", py], capture_output=True, text=True)  # noqa: S603
    pins = "\n".join(
        ln for ln in (freeze.stdout or "").splitlines() if "==" in ln and not ln.startswith("-e")
    )
    cons = Path(tempfile.gettempdir()) / "yue2_demucs_constraints.txt"
    cons.write_text(pins, encoding="utf-8")
    attempts = [[py, "-m", "pip", "install", "-q", "demucs", "-c", str(cons)]]
    uv = shutil.which("uv")
    if uv:
        attempts.append([uv, "pip", "install", "--python", py, "demucs", "-c", str(cons)])
    last = ""
    for cmd in attempts:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # noqa: S603
        if proc.returncode == 0:
            return
        last = (proc.stderr or proc.stdout or "")[-300:]
    raise RuntimeError("Stems: Demucs ließ sich nicht installieren. " + last)


def stems(src: Path, work: Path) -> tuple[Path, Path]:
    """Split *src* into (vocals.wav, instrumental.wav) inside *work*."""
    py = _stem_python()
    _ensure_demucs(py)
    work.mkdir(parents=True, exist_ok=True)
    script = work / "_stems.py"
    script.write_text(_STEM_SCRIPT, encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.run(  # noqa: S603
        [py, str(script), str(src), str(work), ffmpeg()],
        capture_output=True, text=True, timeout=3600, creationflags=flags,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode != 0:
        raise RuntimeError("Stems: " + (proc.stderr or proc.stdout or "").strip()[-400:])
    return work / "vocals.wav", work / "instrumental.wav"


def to_mp3(src: Path, dst: Path) -> None:
    _run(["-i", str(src), "-c:a", "libmp3lame", "-b:a", "320k", str(dst)])


def write_meta(out_dir: Path, request: dict, duration: float | None, note: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    req = {**(request or {}), "edit": note}
    (out_dir / "request.json").write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(
        json.dumps({"status": "complete", "audio_seconds": duration, "edit": note}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
