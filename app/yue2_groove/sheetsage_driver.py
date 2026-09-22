#!/usr/bin/env python3
"""Standalone SheetSage2 transcription driver (runs in the SheetSage2 env).

This file is executed by the *SheetSage2* virtual environment's Python, which
pins different torch/transformers versions than YuE2.  ``yue2_groove`` never
imports ``transformers`` in its own process; ``sheetsage_adapter.py`` builds the
command line and runs this script with ``subprocess``.  Keep the two modules in
sync — together they are the SheetSage2↔groove boundary (the counterpart of
``adapter.py`` for ``yue2``).

Only the standard library is imported at module load, so tests can import this
file in the YuE2 environment without touching the model code.

Interface
---------
One-shot (default): the adapter passes audio/--output/task flags and reads
``<output_dir>/result.json`` / ``failure.json``.
Resident (``--serve``): the model loads once, then each stdin line is a JSON
request (``transcribe`` / ``ping`` / ``stop``); replies are ``@@READY`` once and
``@@RESULT`` per request.  Progress events are ``@@PROGRESS`` lines in both modes;
everything else on stdout/stderr is diagnostic text the adapter ignores.

Adapted from the YuE2 skill helper ``skills/yue2-music/scripts/transcribe.py``
(Apache 2.0, Copyright (c) 2026 the YuE2 authors); see NOTICE.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path

PROGRESS_PREFIX = "@@PROGRESS "
READY_PREFIX = "@@READY "
RESULT_PREFIX = "@@RESULT "

TASKS = ("melody-vocal", "melody-full", "full")
BASE_PROMPTS = ("timestamp", "downbeat_meter", "structure", "key")

# Best-effort import of the vendored portable checker when this file is run from
# the package directory (python yue2_groove/sheetsage_driver.py).  Absent when
# the driver is copied elsewhere; the checks that need it are skipped then.
try:  # pragma: no cover - exercised only in the SheetSage2 environment
    from vendor.abc_tools import parse_abc  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from yue2_groove.vendor.abc_tools import parse_abc  # type: ignore
    except ImportError:
        parse_abc = None


class DriverError(RuntimeError):
    """A transcription failure the adapter can present verbatim."""


def task_settings(task: str) -> tuple[list[str], bool]:
    """(prompts, melody_only) for a task name.

    ``melody-vocal`` keeps only the vocal melody, ``melody-full`` keeps the vocal
    and instrumental melodies (both without chord symbols / chord playback), and
    ``full`` is the default chord + melody transcription.
    """
    if task == "full":
        return list(BASE_PROMPTS) + ["chord_full", "melody_full"], False
    if task == "melody-full":
        return list(BASE_PROMPTS) + ["melody_full"], True
    if task == "melody-vocal":
        return list(BASE_PROMPTS) + ["melody_vocal"], True
    raise DriverError(f"Unknown transcription task {task!r}; use one of {', '.join(TASKS)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def fresh_directory(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def progress_callback():
    """A callback safe for any signature the model uses; prints one stdout line."""

    def report(*args, **kwargs):
        numbers, stage = [], None
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                if stage is None and len(value) <= 40:
                    stage = value
                continue
            if isinstance(value, (int, float)):
                numbers.append(int(value) if float(value).is_integer() else round(float(value), 4))
        event = {"done": numbers[0] if numbers else None}
        if len(numbers) > 1:
            event["total"] = numbers[1]
        if stage:
            event["stage"] = stage
        print(PROGRESS_PREFIX + json.dumps(event, ensure_ascii=False), flush=True)

    return report


def check_melody_only_interface(model) -> None:
    """Refuse to guess: ``melody_only`` must exist as a keyword on ``transcribe``."""
    try:
        parameter = inspect.signature(model.transcribe).parameters.get("melody_only")
    except (TypeError, ValueError) as exc:
        raise DriverError(
            "Cannot verify SheetSage2's melody_only interface; refresh the model "
            "code to a reviewed revision exposing melody_only explicitly"
        ) from exc
    if parameter is None or parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
        raise DriverError(
            "This SheetSage2 revision does not expose melody_only; refresh the model "
            "and remote code to a reviewed revision supporting melody_only=True"
        )


def verify_melody_only_abc(abc: str) -> None:
    """A melody-only transcription must not contain quoted chord symbols."""
    if parse_abc is None:
        return
    try:
        score = parse_abc(abc)
    except ValueError as exc:
        raise DriverError(f"Melody-only transcription did not parse as native ABC: {exc}") from exc
    if any(voice.chords for voice in score.voices.values()):
        raise DriverError("Melody transcription contains unexpected chord symbols")


def snapshot_hashes(snapshot: Path) -> dict:
    files = {}
    if snapshot.is_dir():
        for path in sorted(snapshot.iterdir()):
            if path.is_file() and (
                path.suffix in {".py", ".safetensors"} or path.name == "config.json"
            ):
                files[path.name] = sha256(path)
    return files


def resolve_device_dtype(requested_device: str, requested_dtype: str, torch):
    device = requested_device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dtype = requested_dtype
    if dtype == "auto":
        dtype = "bf16" if device == "cuda" else "fp32"
    return device, dtype


def sync_remote_code(model: str) -> None:
    """Copy every ``*.py`` of a local SheetSage2 snapshot into transformers' module cache.

    ``trust_remote_code`` only copies the files it finds as static relative imports.
    SheetSage2 also opens sibling files at run time (e.g. ``chord_spelling_sheetsage2.py``
    next to ``__file__``), which then are missing from
    ``HF_MODULES_CACHE/transformers_modules/<name>/`` and the transcription fails with
    FileNotFoundError.  Copying the whole code set up front fixes that; best effort.
    """
    try:
        import shutil

        from transformers.utils import HF_MODULES_CACHE

        source = Path(model).expanduser()
        if not source.is_dir():
            try:
                from huggingface_hub import snapshot_download

                source = Path(snapshot_download(model, allow_patterns=["*.py"]))
            except Exception:  # noqa: BLE001
                return
        target = Path(HF_MODULES_CACHE) / "transformers_modules" / source.name
        target.mkdir(parents=True, exist_ok=True)
        for file in source.glob("*.py"):
            dest = target / file.name
            if not dest.is_file() or dest.stat().st_size != file.stat().st_size:
                shutil.copy2(file, dest)
    except Exception as exc:  # noqa: BLE001
        print(f"sync_remote_code skipped: {exc}", file=sys.stderr)


def load_model(args):
    """Import torch/transformers (deliberately late) and load the model once.

    Returns ``(model, device, dtype)``.  Shared by the one-shot ``run`` and the
    resident ``serve`` loop, so a warm worker loads exactly what the CLI would.
    """
    import torch  # deliberately late: this module must import without the SheetSage2 env
    from transformers import AutoModel  # deliberately late as well

    device, dtype = resolve_device_dtype(args.device, args.dtype, torch)
    if args.threads:
        torch.set_num_threads(args.threads)
    loader = {"trust_remote_code": True, "local_files_only": bool(args.offline)}
    if args.revision:
        loader.update(revision=args.revision, code_revision=args.revision)
    if args.base_model:
        loader["base_model_path"] = args.base_model
    sync_remote_code(args.model)
    model = AutoModel.from_pretrained(args.model, **loader).eval().to(device)
    return model, device, dtype


def provenance(args, model, device: str, dtype: str) -> dict:
    snapshot = Path(getattr(model, "_source_snapshot", args.model))
    return {
        "model": args.model,
        "requested_revision": args.revision,
        "device": device,
        "dtype": dtype,
        "config": model.config.to_dict(),
        "snapshot_sha256": snapshot_hashes(snapshot),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "huggingface-hub")
        },
    }


def prepare_request(args, request: dict):
    """Validate one request and create its fresh output directory + input.json.

    Returns ``(audio, output, task, max_seconds, prompts, melody_only, record)``.
    """
    audio = Path(request["audio"]).expanduser()
    if not audio.is_file():
        raise DriverError(f"Audio file not found: {audio}")
    task = request.get("task") or args.task
    max_seconds = request.get("max_seconds", args.max_seconds)
    if max_seconds is not None and max_seconds <= 0:
        raise DriverError("--max-seconds must be positive and explicitly crops the input")
    prompts, melody_only = task_settings(task)
    output = fresh_directory(Path(request["output_dir"]).expanduser())
    write_json(
        output / "input.json",
        {
            "source_name": audio.name,
            "source_audio_sha256": sha256(audio),
            "model": args.model,
            "revision": args.revision,
            "offline": args.offline,
            "base_model_path": args.base_model,
            "prompts": prompts,
            "task": task,
            "melody_only": melody_only,
            "preset": args.preset,
            "max_seconds": max_seconds,
            "device": args.device,
            "dtype": args.dtype,
        },
    )
    record = {
        "status": "failed",
        "task": task,
        "melody_only": melody_only,
        "prompts": prompts,
        "abc": None,
        "abc_error": None,
        "warnings": [],
        "num_events": None,
        "output_dir": str(output),
        "source_audio": str(audio),
        "score_path": None,
    }
    return audio, output, task, max_seconds, prompts, melody_only, record


def write_failure(output: Path, record: dict, exc: Exception) -> None:
    output = Path(output)
    if not (output / "failure.json").exists():
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "type": type(exc).__name__,
                "error": str(exc),
                "abc_error": record.get("abc_error"),
                "warnings": record.get("warnings", []),
            },
        )
    if not (output / "result.json").exists():
        write_json(output / "result.json", record)


def run_transcription(
    model, args, device: str, dtype: str, model_provenance: dict, request: dict
) -> dict:
    """Transcribe one request with an already loaded model into a fresh directory."""
    audio, output, _task, max_seconds, prompts, melody_only, record = prepare_request(args, request)
    write_json(output / "model_provenance.json", model_provenance)
    record["device"], record["dtype"] = device, dtype
    try:
        if melody_only:
            check_melody_only_interface(model)
        options = {"melody_only": True} if melody_only else {}
        result = model.transcribe(
            str(audio),
            output_dir=str(output),
            prompts=prompts,
            dtype=dtype,
            preset=args.preset,
            max_seconds=max_seconds,
            progress=progress_callback(),
            **options,
        )
        if not isinstance(result, dict):
            raise DriverError(f"SheetSage2 returned {type(result).__name__}, expected a dict")
        abc = result.get("abc")
        abc_error = result.get("abc_error")
        warnings = list(result.get("warnings") or [])
        record.update(
            {
                "abc": abc,
                "abc_error": abc_error,
                "warnings": warnings,
                "num_events": result.get("num_events"),
            }
        )
        if abc_error or not abc:
            write_json(
                output / "failure.json",
                {
                    "status": "failed",
                    "type": "DriverError",
                    "error": f"Transcription produced no usable ABC: {abc_error}",
                    "abc_error": abc_error,
                    "warnings": warnings,
                },
            )
            write_json(output / "result.json", record)
            raise DriverError(f"Transcription produced no usable ABC: {abc_error}")
        if melody_only:
            verify_melody_only_abc(abc)
        score_path = output / "score.abc"
        if not score_path.is_file():
            # Keep the contract even when the model implementation only returned
            # the text: write the score the adapter is going to display.
            score_path.write_text(abc, encoding="utf-8")
        record.update({"status": "complete", "score_path": str(score_path)})
        write_json(output / "result.json", record)
        return record
    except Exception as exc:
        write_failure(output, record, exc)
        raise


def run(args) -> dict:
    """One-shot mode: load, transcribe, exit (the safe default)."""
    request = {
        "audio": str(args.audio),
        "output_dir": str(args.output),
        "task": args.task,
        "max_seconds": args.max_seconds,
    }
    model, device, dtype = load_model(args)
    return run_transcription(
        model, args, device, dtype, provenance(args, model, device, dtype), request
    )


def serve(args) -> int:
    """Resident mode: load the model once, then one JSON request per stdin line.

    Requests::

        {"op": "transcribe", "audio": "...", "output_dir": "...", "task": "...", "max_seconds": 30}
        {"op": "ping"}
        {"op": "stop"}

    Replies (stdout, one per line)::

        @@READY  {"model", "device", "dtype"}                     once, after the load
        @@RESULT {"ok": true,  "output_dir", "abc", "warnings", ...}
        @@RESULT {"ok": false, "error", "type", "output_dir"}

    Model code may print to stdout/stderr; callers parse only the @@-prefixed lines.
    A crash in a request is reported and the loop keeps serving.
    """
    model, device, dtype = load_model(args)
    model_provenance = provenance(args, model, device, dtype)
    print(
        READY_PREFIX
        + json.dumps({"model": args.model, "device": device, "dtype": dtype}, ensure_ascii=False),
        flush=True,
    )
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            print(
                RESULT_PREFIX + json.dumps({"ok": False, "error": f"bad request JSON: {exc}"}),
                flush=True,
            )
            continue
        op = request.get("op")
        if op == "stop":
            return 0
        if op == "ping":
            print(RESULT_PREFIX + json.dumps({"ok": True, "pong": True}), flush=True)
            continue
        if op != "transcribe":
            print(
                RESULT_PREFIX + json.dumps({"ok": False, "error": f"unknown op {op!r}"}), flush=True
            )
            continue
        try:
            record = run_transcription(model, args, device, dtype, model_provenance, request)
            print(
                RESULT_PREFIX
                + json.dumps(
                    {
                        "ok": True,
                        "output_dir": record["output_dir"],
                        "abc": record["abc"],
                        "task": record["task"],
                        "melody_only": record["melody_only"],
                        "warnings": record["warnings"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep the worker alive for the next request
            print(
                RESULT_PREFIX
                + json.dumps(
                    {
                        "ok": False,
                        "type": type(exc).__name__,
                        "error": str(exc),
                        "output_dir": request.get("output_dir"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, help="Fresh output directory")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Stay resident: load once, then serve JSON requests on stdin",
    )
    parser.add_argument("--task", choices=TASKS, default="melody-full")
    parser.add_argument("--model", default="m-a-p/SheetSage2")
    parser.add_argument("--revision", help="Pin model and remote code to the same commit")
    parser.add_argument(
        "--base-model", help="Verified MERT-v2-FullSong snapshot for an offline adapter load"
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp32"))
    parser.add_argument("--preset", choices=("default", "paper"), default="default")
    parser.add_argument(
        "--max-seconds", type=float, help="Explicitly crop the input; omitted processes it whole"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.serve:
            return serve(args)
        if args.audio is None or args.output is None:
            parser.error("audio and --output are required unless --serve")
        record = run(args)
        print(f"Saved {record['score_path']}; warnings: {record['warnings']}")
        return 0
    except Exception as exc:  # noqa: BLE001 - the CLI must report every failure clearly
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
