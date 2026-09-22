"""The only groove module that talks to SheetSage2.

SheetSage2 is *not* a pip package: it is a Transformers model with remote code
(``AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True)``) whose
pinned torch/transformers versions collide with YuE2's.  It therefore runs in its
own virtual environment and this module talks to it over a subprocess boundary:
``config.sheetsage_python()`` selects the interpreter, ``sheetsage_driver.py`` is
the script executed inside it, and audio/ABC/results are exchanged through files
and stdout progress lines.  No ``transformers`` import ever happens in the groove
process, so an unconfigured SheetSage2 cannot break the rest of the UI.

Kept deliberately narrow — the adapter counterpart of ``adapter.py``:

* ``resolve_python`` / ``probe``  — is the second environment usable?
* ``build_command`` / ``transcribe`` — run one transcription, with progress and
  cooperative cancel; returns the driver's ``result.json`` (``abc``, ``warnings``,
  ``abc_error``, ``output_dir``, ...).

The model *load* is part of the subprocess lifecycle by default: every transcription
starts a fresh interpreter, loads the model (weights stay in the shared Hugging Face
cache) and frees it on exit.  Setting ``YUE2_GROOVE_SHEETSAGE_KEEP_WARM=1`` (or
``keep_warm=True`` per call) instead keeps one resident driver process (``--serve``)
whose model is reused until ``stop_worker()`` / the UI's UNLOAD button or the idle
timeout (``YUE2_GROOVE_SHEETSAGE_IDLE_SECONDS``); a cancel or crash terminates it and
the next call starts a fresh one.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import config

DRIVER = Path(__file__).resolve().parent / "sheetsage_driver.py"
PROGRESS_PREFIX = "@@PROGRESS "
READY_PREFIX = "@@READY "
RESULT_PREFIX = "@@RESULT "

TASKS = ("melody-vocal", "melody-full", "full")

Progress = Callable[[float | None, str], None]
Cancelled = Callable[[], bool]


class SheetsageNotConfigured(RuntimeError):
    """The SheetSage2 virtual environment is missing or misconfigured."""


class SheetsageFailed(RuntimeError):
    """The transcription subprocess failed."""


def resolve_python(explicit: str | None = None) -> str:
    """Absolute path of the SheetSage2 interpreter, or a clear configuration error."""
    value = (explicit or config.sheetsage_python()).strip()
    if not value:
        raise SheetsageNotConfigured(
            "SheetSage2 is not configured: create its separate virtual environment "
            "(see README, 'Cover from audio') and set "
            "YUE2_GROOVE_SHEETSAGE_PYTHON=/path/to/.venv-sheetsage2/bin/python"
        )
    path = Path(value).expanduser()
    if not path.is_file():
        raise SheetsageNotConfigured(f"SheetSage2 python not found: {path}")
    if not os.access(path, os.X_OK):
        raise SheetsageNotConfigured(f"SheetSage2 python is not executable: {path}")
    return str(path)


def _python_fragment() -> str:
    return (
        "import json, importlib.metadata, importlib.util, sys;"
        "mods={n: importlib.util.find_spec(n) is not None for n in "
        "('torch','transformers','huggingface-hub','torchaudio')};"
        "versions={};"
        "[versions.__setitem__(n, importlib.metadata.version(n)) "
        "for n in ('torch','transformers','huggingface-hub') "
        "if mods.get(n)];"
        "print(json.dumps({'python': sys.version.split()[0], 'executable': sys.executable,"
        " 'packages': mods, 'versions': versions}))"
    )


def probe(python: str | None = None, *, timeout: float = 180) -> dict:
    """Check the SheetSage2 environment without loading any model.

    Returns ``{"python", "packages", "versions", "ffmpeg", "ok"}``; raises
    ``SheetsageNotConfigured`` when the interpreter itself is missing.
    """
    interpreter = resolve_python(python)
    try:
        result = subprocess.run(
            [interpreter, "-c", _python_fragment()],
            capture_output=True,
            timeout=timeout,
            check=False,
            env=config.child_env(),
            **config.SUBPROCESS_TEXT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SheetsageFailed(
            f"Could not run the SheetSage2 python ({interpreter}): {exc}"
        ) from exc
    if result.returncode != 0:
        raise SheetsageFailed(
            f"SheetSage2 environment check failed: {result.stderr.strip()[-800:]}"
        )
    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise SheetsageFailed(
            f"Unexpected check output from {interpreter}: {result.stdout[-300:]!r}"
        ) from exc
    packages = info.get("packages") or {}
    info["ffmpeg"] = shutil.which("ffmpeg")
    info["ok"] = bool(packages.get("torch") and packages.get("transformers"))
    return info


def format_probe(info: dict) -> str:
    lines = [f"python: {info.get('python')}  ({info.get('executable')})"]
    versions = info.get("versions") or {}
    packages = info.get("packages") or {}
    for name in ("torch", "transformers", "huggingface_hub", "torchaudio"):
        key = name.replace("_", "-")
        mark = "ok" if packages.get(key) else "missing"
        version = versions.get(key) or ""
        lines.append(f"  {name}: {mark}{(' ' + version) if version else ''}")
    lines.append(f"ffmpeg: {info.get('ffmpeg') or 'not on PATH (SheetSage2 may need it)'}")
    lines.append(
        "ready" if info.get("ok") else "not ready — install the SheetSage2 requirements first"
    )
    return "\n".join(lines)


def build_command(
    audio_path,
    output_dir,
    *,
    python=None,
    task: str = "melody-full",
    model: str | None = None,
    revision: str | None = None,
    base_model: str | None = None,
    offline: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    preset: str = "default",
    max_seconds: float | None = None,
    threads: int = 4,
) -> list[str]:
    """argv for one transcription; the driver resolves ``melody_only`` from ``task``."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {', '.join(TASKS)}, got {task!r}")
    cmd = [
        resolve_python(python),
        str(DRIVER),
        str(Path(audio_path).expanduser()),
        "--output",
        str(Path(output_dir).expanduser()),
        "--task",
        task,
        "--model",
        model or config.default_sheetsage_model(),
        "--device",
        device,
        "--dtype",
        dtype,
        "--preset",
        preset,
    ]
    if revision:
        cmd += ["--revision", revision]
    if base_model:
        cmd += ["--base-model", base_model]
    if offline:
        cmd += ["--offline"]
    if max_seconds is not None:
        cmd += ["--max-seconds", str(float(max_seconds))]
    if threads:
        cmd += ["--threads", str(int(threads))]
    return cmd


def build_serve_command(
    *,
    python=None,
    model: str | None = None,
    revision: str | None = None,
    base_model: str | None = None,
    offline: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    preset: str = "default",
    threads: int = 4,
) -> list[str]:
    """argv for the resident worker (``--serve``); requests carry audio/task."""
    cmd = [
        resolve_python(python),
        str(DRIVER),
        "--serve",
        "--model",
        model or config.default_sheetsage_model(),
        "--device",
        device,
        "--dtype",
        dtype,
        "--preset",
        preset,
    ]
    if revision:
        cmd += ["--revision", revision]
    if base_model:
        cmd += ["--base-model", base_model]
    if offline:
        cmd += ["--offline"]
    if threads:
        cmd += ["--threads", str(int(threads))]
    return cmd


def _unique_dir(directory: Path) -> Path:
    if not directory.exists():
        return directory
    for index in range(2, 1000):
        candidate = directory.with_name(f"{directory.name}-{index}")
        if not candidate.exists():
            return candidate
    raise SheetsageFailed(f"Could not find a free directory next to {directory}")


def _default_output_dir(audio: Path) -> Path:
    stem = (
        "".join(c if c.isalnum() or c in "-_" else "-" for c in audio.stem)[:40].strip("-")
        or "audio"
    )
    return config.transcriptions_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}"


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:  # pragma: no cover - Windows
            process.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:  # pragma: no cover - Windows
            process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _read_failure(output_dir: Path, stderr_tail: str) -> str:
    record = {}
    with contextlib.suppress(OSError, ValueError):
        record = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    message = str(record.get("error") or "").strip()
    warnings = record.get("warnings") or []
    if warnings:
        message += f" (warnings: {'; '.join(str(w) for w in warnings[:3])})"
    if not message:
        message = stderr_tail.strip()[-1200:] or "no error output"
    return message


class _Worker:
    """One resident SheetSage2 driver process (``sheetsage_driver.py --serve``).

    The model is loaded once at ``start``; each ``transcribe`` writes one JSON
    request to its stdin and waits for the matching ``@@RESULT`` line.  A crash or
    a cancel terminates the process; the next transcription starts a fresh one.
    """

    def __init__(self, cmd: list[str], key: tuple):
        self.cmd = cmd
        self.key = key
        self.process: subprocess.Popen | None = None
        self.last_used = 0.0
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: list[str] = []
        self._lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def _pump(self, stream, sink) -> None:
        try:
            for line in stream:
                if sink is None:
                    self._stderr.append(line)
                else:
                    sink.put(line)
        except (OSError, ValueError):
            pass
        finally:
            if sink is not None:
                sink.put(None)

    def start(self, timeout: float = 900.0) -> dict:
        self.process = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            start_new_session=(os.name == "posix"),
            env=config.child_env(),
            **config.SUBPROCESS_TEXT,
        )
        threading.Thread(
            target=self._pump, args=(self.process.stdout, self._lines), daemon=True
        ).start()
        threading.Thread(target=self._pump, args=(self.process.stderr, None), daemon=True).start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=0.5)
            except queue.Empty:
                if not self.alive():
                    raise SheetsageFailed(
                        "SheetSage2 worker exited while loading: " + self.stderr_tail()
                    ) from None
                continue
            if line is None:
                raise SheetsageFailed(
                    "SheetSage2 worker exited while loading: " + self.stderr_tail()
                )
            if line.startswith(READY_PREFIX):
                try:
                    return json.loads(line[len(READY_PREFIX) :])
                except ValueError as exc:
                    raise SheetsageFailed(f"Unreadable worker ready line: {line[:200]!r}") from exc
        self.stop()
        raise SheetsageFailed(f"SheetSage2 worker did not report ready within {timeout:.0f}s")

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def busy(self) -> bool:
        """True while a request is being served (never reap a busy worker)."""
        return self._lock.locked()

    def stderr_tail(self, limit: int = 800) -> str:
        return "".join(self._stderr).strip()[-limit:]

    def stop(self, timeout: float = 15.0) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write(json.dumps({"op": "stop"}) + "\n")
                    process.stdin.flush()
                    process.wait(timeout=timeout)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    _terminate(process)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass

    # ── one request ─────────────────────────────────────────────────────────
    def transcribe(
        self,
        request: dict,
        *,
        cancelled: Cancelled | None = None,
        progress: Progress | None = None,
        timeout: float | None = None,
    ) -> dict:
        if not self.alive():
            raise SheetsageFailed("SheetSage2 worker is not running")
        with self._lock:
            self.last_used = time.monotonic()  # checked out: never reap mid-request
            try:
                return self._request(
                    request, cancelled=cancelled, progress=progress, timeout=timeout
                )
            finally:
                # Idle time starts when the request *ends*: a long transcription must
                # not look expired the moment it finishes.
                self.last_used = time.monotonic()

    def _request(
        self,
        request: dict,
        *,
        cancelled: Cancelled | None,
        progress: Progress | None,
        timeout: float | None,
    ) -> dict:
        try:
            assert self.process is not None and self.process.stdin is not None
            self.process.stdin.write(json.dumps({"op": "transcribe", **request}) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise SheetsageFailed(
                "SheetSage2 worker closed its input: " + self.stderr_tail()
            ) from exc
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            try:
                line = self._lines.get(timeout=0.25)
            except queue.Empty:
                line = ""
            if line is None:
                self.process = None
                raise SheetsageFailed(
                    "SheetSage2 worker stopped unexpectedly: " + self.stderr_tail()
                )
            if line.startswith(PROGRESS_PREFIX):
                try:
                    event = json.loads(line[len(PROGRESS_PREFIX) :])
                except ValueError:
                    event = {}
                done, total = event.get("done"), event.get("total")
                fraction = (done / total) if (isinstance(done, (int, float)) and total) else None
                if progress is not None:
                    # a UI callback must never kill the job
                    with contextlib.suppress(Exception):
                        progress(fraction, "Transcribing…" + (f" {done}/{total}" if total else ""))
            elif line.startswith(RESULT_PREFIX):
                try:
                    reply = json.loads(line[len(RESULT_PREFIX) :])
                except ValueError as exc:
                    raise SheetsageFailed(f"Unreadable worker reply: {line[:200]!r}") from exc
                if reply.get("ok"):
                    return reply
                raise SheetsageFailed(str(reply.get("error") or "SheetSage2 worker failed"))
            if cancelled is not None and self.alive() and cancelled():
                _terminate(self.process)
                self.process = None
                raise InterruptedError("Transcription cancelled")
            if deadline is not None and time.monotonic() > deadline:
                _terminate(self.process)
                self.process = None
                raise SheetsageFailed(f"Transcription timed out after {timeout:.0f}s")
            if not self.alive() and self._lines.empty():
                raise SheetsageFailed("SheetSage2 worker exited: " + self.stderr_tail())


_WORKER: _Worker | None = None
_WORKER_LOCK = threading.Lock()
_REAPER: threading.Thread | None = None
_REAPER_TICK = 5.0  # how often the background reaper checks the idle timeout


def _reap_idle_worker() -> bool:
    """Stop the resident worker when it has been idle past the configured timeout.

    Called by the background reaper (so memory is actually released without waiting
    for the next transcription) and directly by tests.  A busy worker is never
    reaped; an expired worker is stopped and ``None`` is returned by status.
    """
    global _WORKER
    with _WORKER_LOCK:
        worker = _WORKER
        if worker is None or not worker.alive():
            return False
        idle = config.sheetsage_idle_seconds()
        if not idle or worker.busy or (time.monotonic() - worker.last_used) <= idle:
            return False
        _WORKER = None
    worker.stop()
    return True


def _reaper_loop() -> None:
    while True:
        time.sleep(_REAPER_TICK)
        with contextlib.suppress(Exception):  # the reaper must never kill the process
            _reap_idle_worker()


def _ensure_worker(cmd: list[str], idle_seconds: float) -> _Worker:
    """Return the resident worker for *cmd*, starting/replacing it when needed."""
    global _WORKER, _REAPER
    key = tuple(cmd)
    with _WORKER_LOCK:
        if _WORKER is not None:
            expired = bool(idle_seconds) and (time.monotonic() - _WORKER.last_used) > idle_seconds
            if not _WORKER.alive() or _WORKER.key != key or expired:
                _WORKER.stop()
                _WORKER = None
        if _WORKER is None:
            worker = _Worker(cmd, key)
            worker.start()
            _WORKER = worker
            if _REAPER is None or not _REAPER.is_alive():
                _REAPER = threading.Thread(
                    target=_reaper_loop, daemon=True, name="sheetsage-reaper"
                )
                _REAPER.start()
        # mark as just used so the reaper cannot race a request that is starting
        _WORKER.last_used = time.monotonic()
        return _WORKER


def worker_status() -> dict | None:
    """The resident worker's state, or ``None`` when nothing is loaded."""
    with _WORKER_LOCK:
        if _WORKER is None or not _WORKER.alive():
            return None
        return {
            "pid": _WORKER.process.pid if _WORKER.process else None,
            "idle_seconds": round(time.monotonic() - _WORKER.last_used, 1),
            "command": _WORKER.cmd.copy(),
        }


def stop_worker() -> bool:
    """Stop the resident worker if there is one; returns whether it existed."""
    global _WORKER
    with _WORKER_LOCK:
        worker, _WORKER = _WORKER, None
    if worker is None:
        return False
    worker.stop()
    return True


def transcribe(
    audio_path,
    *,
    output_dir=None,
    task: str = "melody-full",
    melody_only: bool | None = None,
    model: str | None = None,
    revision: str | None = None,
    base_model: str | None = None,
    offline: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    preset: str = "default",
    max_seconds: float | None = None,
    threads: int = 4,
    python: str | None = None,
    keep_warm: bool | None = None,
    cancelled: Cancelled | None = None,
    progress: Progress | None = None,
    timeout: float | None = None,
) -> dict:
    """Transcribe one audio file and return the driver's result record.

    ``task`` selects vocal-melody / full-melody / full-score transcription; the
    optional ``melody_only`` flips between the melody tasks and ``full`` so callers
    that only think in the boolean can still be explicit.  ``progress`` receives
    ``(fraction_or_None, description)`` events; ``cancelled`` is polled while the
    subprocess runs and terminates it (raising ``InterruptedError``).
    """
    audio = Path(audio_path).expanduser()
    if not audio.is_file():
        raise SheetsageFailed(f"Audio file not found: {audio}")
    task = task if task in TASKS else "melody-full"
    if melody_only is True and task == "full":
        task = "melody-full"
    elif melody_only is False and task != "full":
        task = "full"

    output = Path(output_dir).expanduser() if output_dir else _default_output_dir(audio)
    output = _unique_dir(output)

    def report(fraction, description):
        if progress is not None:
            with contextlib.suppress(Exception):  # a UI callback must never kill the job
                progress(fraction, description)

    warm = config.sheetsage_keep_warm() if keep_warm is None else bool(keep_warm)
    if warm:
        serve_cmd = build_serve_command(
            python=python,
            model=model,
            revision=revision,
            base_model=base_model,
            offline=offline,
            device=device,
            dtype=dtype,
            preset=preset,
            threads=threads,
        )
        report(None, "Loading the SheetSage2 worker…")
        worker = _ensure_worker(serve_cmd, config.sheetsage_idle_seconds())
        reply = worker.transcribe(
            {
                "audio": str(audio),
                "output_dir": str(output),
                "task": task,
                "max_seconds": max_seconds,
            },
            cancelled=cancelled,
            progress=progress,
            timeout=timeout,
        )
        try:
            record = json.loads((output / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SheetsageFailed(
                f"SheetSage2 worker replied but wrote no readable result.json in {output}"
            ) from exc
        record.setdefault("output_dir", str(output))
        record["task"] = task
        record["melody_only"] = bool(record.get("melody_only", task != "full"))
        record["command"] = serve_cmd
        record["worker"] = {"pid": worker.process.pid if worker.process else None, "resident": True}
        if not reply.get("abc") and not record.get("abc"):
            raise SheetsageFailed(str(reply.get("error") or "SheetSage2 worker returned no ABC"))
        return record

    # One-shot: build the command first so a configuration error touches nothing.
    cmd = build_command(
        audio,
        output,
        python=python,
        task=task,
        model=model,
        revision=revision,
        base_model=base_model,
        offline=offline,
        device=device,
        dtype=dtype,
        preset=preset,
        max_seconds=max_seconds,
        threads=threads,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report(None, "Starting the SheetSage2 environment…")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        stdin=subprocess.DEVNULL,
        start_new_session=(os.name == "posix"),
        env=config.child_env(),
        **config.SUBPROCESS_TEXT,
    )
    assert process.stdout is not None and process.stderr is not None  # both are PIPEs
    stdout, stderr = process.stdout, process.stderr
    lines: queue.Queue[str | None] = queue.Queue()
    stderr_lines: list[str] = []

    def pump_stdout():
        try:
            for line in stdout:
                lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    def pump_stderr():
        with contextlib.suppress(OSError, ValueError):  # the pipe closes when the child exits
            stderr_lines.extend(stderr)

    stdout_thread = threading.Thread(target=pump_stdout, daemon=True)
    stderr_thread = threading.Thread(target=pump_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    deadline = time.monotonic() + timeout if timeout else None
    cancelled_at = None
    stream_done = False
    while True:
        try:
            line = lines.get(timeout=0.25)
        except queue.Empty:
            line = ""
        if line is None:
            stream_done = True
        elif line.startswith(PROGRESS_PREFIX):
            try:
                event = json.loads(line[len(PROGRESS_PREFIX) :])
            except ValueError:
                event = {}
            done, total = event.get("done"), event.get("total")
            fraction = (done / total) if (isinstance(done, (int, float)) and total) else None
            report(fraction, "Transcribing…" + (f" {done}/{total}" if total else ""))
        if cancelled is not None and process.poll() is None and cancelled():
            cancelled_at = time.monotonic()
            _terminate(process)
            break
        if deadline is not None and time.monotonic() > deadline:
            _terminate(process)
            raise SheetsageFailed(
                f"Transcription timed out after {timeout:.0f}s; the audio may be too long for this machine"
            )
        if process.poll() is not None and (stream_done or lines.empty()):
            break

    process.wait()
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    if cancelled_at is not None:
        raise InterruptedError("Transcription cancelled")
    if process.returncode != 0:
        raise SheetsageFailed(
            f"SheetSage2 transcription failed (exit {process.returncode}): "
            f"{_read_failure(output, ''.join(stderr_lines))}"
        )
    try:
        record = json.loads((output / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SheetsageFailed(
            f"SheetSage2 finished but wrote no readable result.json in {output}: "
            f"{''.join(stderr_lines).strip()[-600:]}"
        ) from exc
    record.setdefault("output_dir", str(output))
    record["task"], record["melody_only"] = task, bool(record.get("melody_only", task != "full"))
    record["command"] = cmd
    return record
