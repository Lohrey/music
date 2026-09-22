"""ACE-Step 1.5 as a second engine (50+ languages, fast) — via its own REST API.

ACE-Step is installed by its own Pinokio launcher (``~/pinokio/api/ace-step.pinokio.git``).
This module starts that installation's ``acestep-api`` server on demand, sends jobs
to it and copies the finished MP3s into our runs folder, so the song list, player,
sharing and the ⋯ menu treat ACE songs like every other song.

GPU: ACE-Step and YuE2 share one card, so :func:`stop` frees ACE's VRAM before a
YuE2 job, and the caller unloads YuE2 before an ACE job.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
import uuid
from pathlib import Path

log = logging.getLogger("yue2_groove")

PORT = int(os.environ.get("YUE2_ACE_PORT", "8019"))
BASE = f"http://127.0.0.1:{PORT}"
_PROC: subprocess.Popen | None = None
_LOCK = threading.Lock()


def _candidates() -> list[Path]:
    env = os.environ.get("ACESTEP_DIR", "").strip()
    out = [Path(env)] if env else []
    here = Path(__file__).resolve()
    for parent in here.parents:  # …/pinokio/api/<this app>/app/yue2_groove/webui
        if parent.name == "api":
            out.append(parent / "ace-step.pinokio.git" / "app")
            break
    out += [
        Path("C:/pinokio/api/ace-step.pinokio.git/app"),
        Path.home() / "pinokio" / "api" / "ace-step.pinokio.git" / "app",
    ]
    return out


def install_dir() -> Path | None:
    for cand in _candidates():
        if (cand / ".venv").is_dir() and (cand / "acestep").is_dir():
            return cand
    return None


def available() -> bool:
    return install_dir() is not None


def _server_cmd(root: Path) -> list[str]:
    scripts = root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")
    exe = scripts / ("acestep-api.exe" if sys.platform == "win32" else "acestep-api")
    if exe.is_file():
        return [str(exe), "--port", str(PORT)]
    py = scripts / ("python.exe" if sys.platform == "win32" else "python")
    return [str(py), "-m", "acestep.api_server", "--port", str(PORT)]


def _get(path: str, timeout: float = 10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """Encode form fields + audio files (the API refuses absolute paths, so we upload)."""
    boundary = "----yue2ace" + uuid.uuid4().hex
    out = bytearray()
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            value = json.dumps(value)
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n".encode()
        out += str(value).encode("utf-8") + b"\r\n"
    for key, path in files.items():
        path = Path(path)
        out += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"; "
            f"filename=\"source{path.suffix or '.mp3'}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        out += path.read_bytes() + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _post(path: str, payload: dict, timeout: float = 30, files: dict | None = None) -> dict:
    if files:
        data, ctype = _multipart(payload, files)
        timeout = max(timeout, 300)
    else:
        data, ctype = json.dumps(payload).encode(), "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            raw = exc.read().decode("utf-8", "replace")
            try:
                j = json.loads(raw)
                detail = j.get("detail") or j.get("error") or raw
            except ValueError:
                detail = raw
        except Exception:  # noqa: BLE001, S110
            pass
        raise RuntimeError(f"ACE-Step {exc.code}: {str(detail)[:300] or exc.reason}") from None
    if isinstance(body, dict) and body.get("code") not in (None, 200):
        raise RuntimeError(f"ACE-Step: {body.get('error') or body}")
    return body.get("data", body) if isinstance(body, dict) else body


def _healthy() -> bool:
    try:
        _get("/health", timeout=3)
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_server(progress=None, timeout: float = 1800) -> None:
    """Start ACE-Step's API server if needed and wait until it answers."""
    global _PROC
    with _LOCK:
        if _healthy():
            return
        root = install_dir()
        if root is None:
            raise RuntimeError(
                "ACE-Step ist noch nicht installiert (Pinokio → ACE-Step 1.5 → Install)."
            )
        if _PROC is None or _PROC.poll() is not None:
            log.info("ace-step: starting API server on port %s", PORT)
            env = {**os.environ, "ACESTEP_API_HOST": "127.0.0.1", "PYTHONIOENCODING": "utf-8"}
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            logf = open(root / "yue2_ace_api.log", "ab")  # noqa: SIM115
            _PROC = subprocess.Popen(  # noqa: S603
                _server_cmd(root), cwd=str(root), env=env, stdout=logf, stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        start = time.time()
        while time.time() - start < timeout:
            if _PROC.poll() is not None:
                raise RuntimeError(
                    "ACE-Step-Server ist abgestürzt – Details in ace-step…/app/yue2_ace_api.log"
                )
            if _healthy():
                return
            if progress:
                progress(0.02, "⏳ ACE-Step wird geladen …")
            time.sleep(2)
        raise RuntimeError("ACE-Step-Server startet nicht (Zeitüberschreitung).")


def stop() -> None:
    """Free the GPU: stop the ACE-Step server we started (no-op otherwise)."""
    global _PROC
    with _LOCK:
        if _PROC is not None and _PROC.poll() is None:
            log.info("ace-step: stopping API server (freeing VRAM)")
            _PROC.terminate()
            try:
                _PROC.wait(timeout=20)
            except subprocess.TimeoutExpired:
                _PROC.kill()
        _PROC = None
        if sys.platform == "win32" and _healthy():  # left over from an earlier app run
            _kill_port_windows()


def _kill_port_windows() -> None:
    try:
        out = subprocess.run(  # noqa: S603, S607
            ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout
        pids = {
            ln.split()[-1] for ln in out.splitlines()
            if len(ln.split()) >= 5 and ln.split()[1].endswith(f":{PORT}") and "LISTEN" in ln.upper()
        }
        for pid in pids:
            if pid.isdigit() and pid != "0":
                log.info("ace-step: stopping leftover API server (pid %s)", pid)
                subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True,  # noqa: S603, S607
                               timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as exc:  # noqa: BLE001
        log.warning("ace-step: could not stop leftover server: %s", exc)


def generate(
    payload: dict,
    out_dirs: list[Path],
    progress=None,
    cancelled=None,
    timeout: float = 3600,
) -> list[dict]:
    """Run one ACE-Step job (``batch_size`` = len(out_dirs)) and store each result as
    ``<out_dir>/audio.mp3`` + request/result JSON. Returns the per-song metadata."""
    ensure_server(progress)
    payload = {**payload, "batch_size": len(out_dirs), "audio_format": "mp3"}
    payload.setdefault("lm_backend", "pt" if sys.platform == "win32" else "vllm")
    files = {}
    for key, form_key in (("src_audio_path", "src_audio"), ("reference_audio_path", "ref_audio")):
        if payload.get(key):
            files[form_key] = payload.pop(key)
    task = _post("/release_task", payload, files=files or None)
    task_id = task["task_id"] if isinstance(task, dict) else task
    start = time.time()
    items: list[dict] = []
    while True:
        if cancelled and cancelled():
            raise InterruptedError("abgebrochen")
        if time.time() - start > timeout:
            raise RuntimeError("ACE-Step: Zeitüberschreitung")
        data = _post("/query_result", {"task_id_list": [task_id]})
        entry = (data or [{}])[0]
        status = int(entry.get("status", 0))
        result = entry.get("result") or "[]"
        items = json.loads(result) if isinstance(result, str) else result
        if status == 1:
            break
        if status == 2:
            err = (items[0].get("error") if items else None) or entry.get("progress_text") or "?"
            raise RuntimeError(f"ACE-Step-Fehler: {err}")
        frac = float((items[0].get("progress") if items else 0) or 0)
        if progress:
            progress(0.05 + 0.9 * max(0.0, min(1.0, frac)), "🎤 Gesang & Musik entstehen")
        time.sleep(2)

    songs = []
    for out_dir, item in zip(out_dirs, [i for i in items if i.get("file")]):
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "audio.mp3"
        ref = str(item["file"])
        local = ref
        if "/v1/audio" in ref and "path=" in ref:  # the API returns a download URL
            qs = urllib.parse.urlparse(ref).query
            local = (urllib.parse.parse_qs(qs).get("path") or [""])[0]
        src = Path(local) if local else None
        if src is not None and src.is_file():
            shutil.copyfile(src, target)
        elif ref.startswith("/"):
            target.write_bytes(_get(ref if "/v1/audio" in ref else
                                    "/v1/audio?path=" + urllib.parse.quote(ref), timeout=120))
        elif ref.startswith("http"):
            with urllib.request.urlopen(ref, timeout=120) as resp:  # noqa: S310
                target.write_bytes(resp.read())
        else:
            target.write_bytes(_get("/v1/audio?path=" + urllib.parse.quote(ref), timeout=120))
        meta = item.get("metas") or {}
        duration = meta.get("duration")
        try:
            import soundfile as sf

            duration = sf.info(str(target)).duration
        except Exception:  # noqa: BLE001, S110
            pass
        request = {
            "engine": "ace-step",
            "style": payload.get("prompt") or payload.get("sample_query") or "",
            "lyrics": item.get("lyrics") or payload.get("lyrics") or "",
            "vocal_language": payload.get("vocal_language"),
            "task_type": payload.get("task_type", "text2music"),
            "seed": payload.get("seed"),
        }
        (out_dir / "request.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "result.json").write_text(
            json.dumps(
                {"weights": "ACE-Step 1.5", "audio_seconds": duration, "status": "complete",
                 "caption": item.get("prompt"), "bpm": meta.get("bpm"),
                 "keyscale": meta.get("keyscale")},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        songs.append({"dir": out_dir, "duration": duration})
    if not songs:
        raise RuntimeError("ACE-Step hat keine Audiodatei geliefert.")
    return songs
