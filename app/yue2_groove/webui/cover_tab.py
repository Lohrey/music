"""02 COVER handlers (see sheetsage_adapter.py for the subprocess, cover.py for the request)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr

from .. import adapter, config, cover, edit_flow, library, sheetsage_adapter
from . import library_tab, runtime


def sampling_summary_pair(*args):
    """``demo.load`` wrapper: one summary string per mirror output.

    Gradio requires exactly one return value per output component, and the page
    load updates both the EDIT and the COVER mirror.
    """
    text = sampling_summary(*args)
    return text, text


def sampling_summary(
    abc_temp,
    abc_p,
    abc_k,
    abc_rep,
    abc_win,
    abc_min,
    abc_max,
    sem_temp,
    sem_p,
    sem_k,
    sem_rep,
    sem_win,
    sem_min,
    sem_max,
) -> str:
    """One-line view of the sampling parameters shared with 01 GENERATE."""
    return (
        f"ABC  t={float(abc_temp):g}  p={float(abc_p):g}  k={int(abc_k)}  "
        f"rep={float(abc_rep):g}  win={int(abc_win)}  min={int(abc_min)}  max={int(abc_max)}\n"
        f"SEM  t={float(sem_temp):g}  p={float(sem_p):g}  k={int(sem_k)}  "
        f"rep={float(sem_rep):g}  win={int(sem_win)}  min={int(sem_min)}  max={int(sem_max)}"
    )


def _transcription_files(directory) -> list[str]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [str(p) for p in sorted(directory.rglob("*")) if p.is_file()]


def cover_check_environment():
    """CHECK ENVIRONMENT: probe the SheetSage2 venv without loading any model."""
    try:
        return sheetsage_adapter.format_probe(sheetsage_adapter.probe())
    except (sheetsage_adapter.SheetsageNotConfigured, sheetsage_adapter.SheetsageFailed) as exc:
        return f"SheetSage2 environment not ready: {exc}"


def cover_unload_worker():
    """Drop the resident SheetSage2 worker (freeing its memory) if there is one."""
    stopped = sheetsage_adapter.stop_worker()
    return (
        "SheetSage2 worker unloaded; the next transcription loads the model again."
        if stopped
        else "No resident SheetSage2 worker."
    )


def cover_transcribe(
    audio_path,
    task,
    max_seconds,
    model,
    device,
    dtype,
    revision,
    base_model,
    keep_warm,
    offline,
    progress=gr.Progress(),
):
    """Generator: transcription disables the TRANSCRIBE button while running."""
    if not (audio_path or "").strip():
        raise gr.Error("Upload a source audio file first")
    if task not in sheetsage_adapter.TASKS:
        raise gr.Error(f"Unknown transcription task: {task}")
    if not runtime.try_start_job():
        yield (
            gr.update(),
            gr.update(),
            runtime.BUSY_MESSAGE,
            *((gr.update(),) * 7),
            gr.update(),
            gr.update(),
            gr.skip(),
        )
        return
    controls = (gr.update(interactive=False),) * 7
    idle = (gr.update(interactive=True),) * 7
    try:
        yield (
            gr.update(),
            gr.update(),
            "Starting SheetSage2 transcription…",
            *controls,
            gr.update(),
            gr.update(),
            gr.skip(),
        )
        outdir = (
            config.transcriptions_dir(runtime.RUNS)
            / f"{time.strftime('%Y%m%d-%H%M%S')}-{runtime.slug(Path(audio_path).stem)}"
        )

        def on_progress(value, text):
            progress(value, desc=text)

        record = sheetsage_adapter.transcribe(
            audio_path,
            output_dir=outdir,
            task=task,
            model=(model or "").strip() or None,
            revision=(revision or "").strip() or None,
            base_model=(base_model or "").strip() or None,
            keep_warm=bool(keep_warm),
            offline=bool(offline),
            device=device,
            dtype=dtype,
            max_seconds=float(max_seconds) if max_seconds else None,
            cancelled=runtime.CANCEL.is_set,
            progress=on_progress,
        )
        abc = record.get("abc") or ""
        warnings = record.get("warnings") or []
        lines = [
            f"Transcribed (task={record.get('task')}) → {record['output_dir']}",
            (
                f"device={record.get('device')} dtype={record.get('dtype')}  "
                f"melody_only={record.get('melody_only')}"
            ),
            "Review the ABC before covering; transcription can contain musical errors.",
        ]
        if warnings:
            lines.append("warnings: " + "; ".join(str(w) for w in warnings))
        worker = sheetsage_adapter.worker_status()
        if worker:
            lines.append(
                f"SheetSage2 worker resident (pid {worker['pid']}) — reused by the next "
                f"transcription until UNLOAD or the idle timeout."
            )
        # surface the direct-generation path now that there is a score to use,
        # and select the new transcription as the reusable source
        _items, choices = library_tab.library_choices(library_tab.library_mode("time", "desc"))
        known = {value for _label, value in choices}
        try:
            rel = (
                Path(record["output_dir"]).resolve().relative_to(runtime.RUNS.resolve()).as_posix()
            )
        except (ValueError, OSError):
            rel = None
        source_update = (
            gr.update(choices=choices, value=rel) if rel in known else gr.update(choices=choices)
        )
        yield (
            abc,
            _transcription_files(record["output_dir"]),
            "\n".join(lines),
            *idle,
            gr.update(open=True),
            source_update,
            record["output_dir"],
        )
    except InterruptedError as exc:
        note = (
            " The resident SheetSage2 worker was stopped; the next transcription reloads it."
            if keep_warm
            else ""
        )
        yield (
            gr.update(),
            gr.update(),
            f"Cancelled: {exc}.{note}",
            *idle,
            gr.update(),
            gr.update(),
            gr.skip(),
        )
    except Exception as exc:  # noqa: BLE001
        yield (
            gr.update(),
            gr.update(),
            f"Transcription failed: {type(exc).__name__}: {exc}",
            *idle,
            gr.update(),
            gr.update(),
            gr.skip(),
        )
    finally:
        runtime.end_job()


def cover_choices():
    """Choices for the COVER source dropdown (any saved work or transcription with an ABC)."""
    _items, choices = library_tab.library_choices(
        library_tab.library_mode("time", "desc"), include_pending=False
    )
    return gr.update(choices=choices)


def cover_load(rel):
    """Fill COVER ABC (and STYLE/LYRICS when the source has them) from a saved work."""
    if not (rel or "").strip():
        raise gr.Error("Pick a SOURCE WORK first (press REFRESH if the list is empty)")
    item, det = library.load(runtime.RUNS, rel)
    if item is None or det is None:
        raise gr.Error("That work no longer exists — press REFRESH")
    abc = (det.get("abc") or "").strip()
    if not abc:
        raise gr.Error(f"{item['name']} has no ABC score to load")
    request = det.get("request") or {}
    style = request.get("style") or request.get("tags") or ""
    lyrics = request.get("lyrics") or ""
    return (
        gr.update(value=abc),
        gr.update(value=style) if style.strip() else gr.update(),
        gr.update(value=lyrics) if lyrics.strip() else gr.update(),
        (
            f"Loaded {item['name']} ({item['kind']}) — {len(abc)} chars. Continue here or "
            f"press SEND TO EDIT."
        ),
    )


def cover_send_to_edit(abc_text, rel, style, lyrics):
    """Hand the current COVER score to 03 EDIT, keeping the source as the freeze target."""
    if not (abc_text or "").strip():
        raise gr.Error("Transcribe or load a score first")
    if not (rel or "").strip():
        raise gr.Error(
            "LOAD a source work (or transcribe) first — 03 EDIT freezes that source "
            "before it can check the edit"
        )
    item, det = library.load(runtime.RUNS, rel)
    if item is None or det is None:
        raise gr.Error("That source work no longer exists — refresh 02 COVER")
    source_abc = (det.get("abc") or "").strip()
    if not source_abc:
        raise gr.Error(f"{item['name']} has no ABC to freeze as the edit baseline")
    try:
        current = edit_flow.validate_edited_abc(abc_text)
    except ValueError as exc:
        raise gr.Error(f"Cannot send this score to 03 EDIT: {exc}") from exc
    request = det.get("request") or {}
    style_value = (style or "").strip() or request.get("style") or request.get("tags") or ""
    lyrics_value = (lyrics or "").strip() or request.get("lyrics") or ""
    return (
        gr.update(value=current),  # EDITED ABC (what you see in COVER)
        source_abc,  # baseline ABC for CHECK INVARIANTS
        gr.update(value=style_value) if style_value else gr.update(),
        gr.update(value=lyrics_value) if lyrics_value else gr.update(),
        rel,  # SOURCE WORK in 03 EDIT (state)
        {},  # check state reset
        None,  # not frozen yet: FREEZE BASELINE is required
        f"From 02 COVER: baseline = {rel}. Press FREEZE BASELINE, then CHECK INVARIANTS.",
        "Loaded from 02 COVER — FREEZE BASELINE is required before CHECK / GENERATE EDITED.",
        gr.update(
            choices=[
                value
                for _label, value in library_tab.library_choices(
                    library_tab.library_mode("time", "desc")
                )[1]
            ],
            value=rel,
        ),
        str((runtime.RUNS / rel).resolve()),
        gr.update(selected="edit"),
    )


def cover_generate(
    style,
    lyrics,
    abc_text,
    task,
    keep_voice,
    seed,
    cfg_scale,
    abc_temp,
    abc_p,
    abc_k,
    abc_rep,
    abc_win,
    abc_min,
    abc_max,
    sem_temp,
    sem_p,
    sem_k,
    sem_rep,
    sem_win,
    sem_min,
    sem_max,
    device,
    dtype,
    backend,
    quantization,
    offload_ar,
    budget,
    ode_steps,
    vae_core_frames,
    model,
    vae_choice,
    vae_custom,
    revision,
    vae_revision,
    offline,
    progress=gr.Progress(),
):
    """Generator: generate directly from the score on 02 COVER (no tab detour).

    Uses the same ``cover.build_cover_request`` as SEND TO GENERATE (melody tasks get a
    chord-free score, full tasks keep the harmony) and the shared generation core.
    """
    if not (abc_text or "").strip():
        raise gr.Error(
            "Transcribe (or paste) an ABC score first — GENERATE COVER is "
            "score-conditioned and never plans a fresh melody"
        )
    style, lyrics = runtime.request_texts(style, lyrics)
    try:
        request = cover.build_cover_request(
            style,
            lyrics,
            abc_text,
            task=task,
            seed=runtime.resolve_seed(seed),
            cfg_scale=cfg_scale,
            keep_voice=keep_voice,
            request_factory=adapter.song_request,
        )
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"Invalid cover request: {exc}") from exc
    abc_sampling, sem_sampling = runtime.sampling_pair(
        abc_temp,
        abc_p,
        abc_k,
        abc_rep,
        abc_win,
        abc_min,
        abc_max,
        sem_temp,
        sem_p,
        sem_k,
        sem_rep,
        sem_win,
        sem_min,
        sem_max,
    )

    settings = runtime.RuntimeSettings(
        device,
        dtype,
        backend,
        quantization,
        offload_ar,
        budget,
        ode_steps,
        vae_core_frames,
        model,
        vae_choice,
        vae_custom,
        revision,
        vae_revision,
        offline,
    )
    if not runtime.try_start_job():
        yield (
            runtime.BUSY_MESSAGE,
            gr.update(),
            gr.update(),
            gr.update(),
            *((gr.update(),) * 7),
            gr.skip(),
        )
        return
    controls = (gr.update(interactive=False),) * 7
    idle = (gr.update(interactive=True),) * 7
    try:
        yield (
            "Starting cover generation…",
            gr.update(),
            gr.update(),
            gr.update(),
            *controls,
            gr.skip(),
        )
        pipe, note = runtime.get_pipe(settings, progress)
        outdir = runtime.run_dir(
            "cover", runtime.slug(request.id if request.id != "song" else style)
        )
        song, result, elapsed = runtime.run_generation(
            pipe,
            request,
            outdir,
            abc_sampling=abc_sampling,
            semantic_sampling=sem_sampling,
            progress=progress,
            note=note,
        )
        yield (
            runtime.generation_status(song, result, outdir, elapsed, request, note),
            str(outdir / "audio.flac"),
            (song.abc or ""),
            runtime.artifact_files(outdir, bool(song.abc)),
            *idle,
            str(outdir),
        )
    except Exception as exc:  # noqa: BLE001 — the UI reports, never crashes
        yield runtime.failure_text(exc), gr.update(), gr.update(), gr.update(), *idle, gr.skip()
    finally:
        runtime.end_job()


def cover_strip(text, keep_voice):
    try:
        stripped = cover.prepare_cover_abc(text, keep_voice=keep_voice)
    except ValueError as exc:
        raise gr.Error(f"Chord strip failed: {exc}") from exc
    return stripped, (
        f"Chords removed (kept: {keep_voice}). Melody, meter and tempo were verified unchanged."
    )


def cover_send_to_generate(text, task, style, lyrics, keep_voice):
    """One click into 01 GENERATE: ABC, plan mode, and any target style/lyrics typed here."""
    try:
        cot = cover.cot_for_task(task)
        if cot == "melody":
            prepared = cover.prepare_cover_abc(text, keep_voice=keep_voice)
        else:
            cover.inspect_abc(text)  # validate before handing it over
            prepared = cover.clean_abc(text)
        if not prepared.strip():
            raise ValueError("There is no ABC score to send")
    except ValueError as exc:
        raise gr.Error(f"Cannot send this score to GENERATE: {exc}") from exc
    copied = []
    style_update = gr.update()
    lyrics_update = gr.update()
    if (style or "").strip():
        style_update = gr.update(value=style.strip())
        copied.append("STYLE")
    if (lyrics or "").strip():
        lyrics_update = gr.update(value=lyrics.strip())
        copied.append("LYRICS")
    what = f" and the {', '.join(copied)} typed here" if copied else ""
    detail = "chord-free (melody mode)" if cot == "melody" else "full score (with chords)"
    status = (
        f"Sent to 01 GENERATE: PLAN MODE={cot.upper()}, the {detail} ABC{what}. "
        f"SCORE INPUT is open so you can review it; add anything still missing, "
        f"then press GENERATE."
    )
    return (
        gr.update(value=prepared),
        gr.update(value=cot),
        style_update,
        lyrics_update,
        gr.update(open=True),
        gr.update(selected="gen"),
        status,
    )


def _sheetsage_python_candidates() -> list[Path]:
    repo = Path(__file__).resolve().parent.parent
    return [
        repo / ".venv-sheetsage2" / "bin" / "python",
        repo.parent / "YuE" / ".venv-sheetsage2" / "bin" / "python",
    ]


def cover_detect_python():
    """Look for a SheetSage2 venv in the documented locations and use it for this session."""
    found = next((path for path in _sheetsage_python_candidates() if path.is_file()), None)
    if found is None:
        return (
            "No SheetSage2 venv found at ./.venv-sheetsage2 or ../YuE/.venv-sheetsage2 — "
            "install one (README, 'Cover from audio') or set YUE2_GROOVE_SHEETSAGE_PYTHON."
        )
    os.environ["YUE2_GROOVE_SHEETSAGE_PYTHON"] = str(found)
    return f"Using {found} for this session — add it to .env (or --sheetsage-python) to persist."
