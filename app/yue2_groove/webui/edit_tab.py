"""03 EDIT handlers (see edit_flow.py for the baseline / invariant logic)."""

from __future__ import annotations

import json

import gradio as gr

from .. import adapter, edit_flow, library
from . import library_tab, runtime, tools_tab


def edit_choices():
    _items, choices = library_tab.library_choices(
        library_tab.library_mode("time", "desc"), include_pending=False
    )
    return gr.update(choices=choices)


def edit_load(rel):
    """Load one saved work as the edit source; its ABC becomes the baseline."""
    if not (rel or "").strip():
        raise gr.Error("Pick a source work first")
    item, det = library.load(runtime.RUNS, rel)
    if item is None or det is None:
        raise gr.Error("That work no longer exists — press REFRESH")
    abc = (det.get("abc") or "").strip()
    if not abc:
        raise gr.Error(f"{item['name']} has no ABC score to edit")
    request = det.get("request") or {}
    style = request.get("style") or request.get("tags") or ""
    info = (
        f"{item['name']} · {rel}\n"
        "baseline not frozen yet — FREEZE BASELINE is required before CHECK INVARIANTS "
        "and GENERATE EDITED (the original run directory is never modified)."
    )
    return (
        gr.update(value=style),
        gr.update(value=request.get("lyrics", "") or ""),
        gr.update(value=abc),
        abc,
        rel,
        {},
        None,
        info,
        f"Loaded {item['name']} as the edit source.",
    )


def edit_freeze(source_rel, visible_rel=""):
    if not (source_rel or "").strip():
        raise gr.Error("Load a source work first")
    if (visible_rel or "").strip() and visible_rel.strip() != source_rel.strip():
        raise gr.Error(
            f"SOURCE WORK now shows “{visible_rel.strip()}” but “{source_rel.strip()}” "
            f"is loaded — press LOAD first so the baseline you freeze is the one you "
            f"see"
        )
    try:
        record = edit_flow.freeze_baseline(runtime.RUNS, source_rel)
    except (ValueError, OSError) as exc:
        raise gr.Error(f"Freeze failed: {exc}") from exc
    info = (
        f"{source_rel} → {record['baseline']['rel']}\n"
        f"recorded {len(record['hashes'])} file hash(es) + copies of score.abc/request.json; "
        "the original run directory is untouched."
    )
    return record, info, f"Baseline frozen: {record['baseline']['rel']}"


def edit_check(baseline_abc, abc_text, voices, allow_tempo, baseline_state, contract, allow_meter):
    if not (baseline_abc or "").strip():
        raise gr.Error("Load a source work first — the check compares against its baseline ABC")
    if not baseline_state:
        raise gr.Error(
            "Freeze the baseline first (FREEZE BASELINE) — the check is recorded "
            "against that frozen record"
        )
    try:
        result = edit_flow.check_invariants(
            baseline_abc,
            abc_text,
            voices=voices,
            allow_tempo_change=bool(allow_tempo),
            allow_meter_change=bool(allow_meter),
            contract=contract or "exact",
        )
    except ValueError as exc:
        raise gr.Error(f"Invariant check failed: {exc}") from exc
    state = {
        "sha256": edit_flow.sha256_text(edit_flow.clean_abc(abc_text)),
        "match": bool(result["match"]),
        "result": result,
        "voices": voices,
        "allow_tempo_change": bool(allow_tempo),
        "allow_meter_change": bool(allow_meter),
        "contract": result["contract"],
    }
    return json.dumps(result, ensure_ascii=False, indent=2), state


def edit_generate(
    style,
    lyrics,
    cot,
    seed,
    cfg_scale,
    abc_text,
    baseline_abc,
    source_rel,
    check_state,
    allow_changes,
    baseline_state,
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
    """Generator: regenerate from the edited score; never silently drops the edit."""
    try:
        abc = edit_flow.validate_edited_abc(abc_text)
    except ValueError as exc:
        raise gr.Error(f"The edited ABC cannot be used: {exc}") from exc
    if not (baseline_abc or "").strip():
        raise gr.Error(
            "Load a source work first — 03 EDIT regenerates a saved work from its "
            "edited score; use 01 GENERATE for a fresh song"
        )
    if not baseline_state:
        raise gr.Error(
            "Freeze the baseline first (FREEZE BASELINE) — the edit manifest must "
            "point at the frozen source"
        )
    if not check_state or check_state.get("sha256") != edit_flow.sha256_text(abc):
        raise gr.Error("Run CHECK INVARIANTS on the current edited ABC before generating")
    if not check_state.get("match") and not allow_changes:
        differences = "; ".join((check_state.get("result") or {}).get("differences", [])[:3])
        raise gr.Error(
            "CHECK INVARIANTS did not pass: "
            + (differences or "scores differ")
            + " — enable ALLOW MELODY/RHYTHM CHANGES if the change is intentional"
        )
    style, lyrics = runtime.request_texts(style, lyrics)
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
    try:
        seed = runtime.resolve_seed(seed)  # the request and the manifest carry the same seed
        request = edit_flow.build_edit_request(
            style,
            lyrics,
            abc,
            cot=cot,
            seed=seed,
            cfg_scale=cfg_scale,
            request_factory=adapter.song_request,
        )
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"Invalid request: {exc}") from exc

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
            gr.update(),
            gr.update(),
            gr.update(),
            runtime.BUSY_MESSAGE,
            *((gr.update(),) * 6),
            gr.update(),
        )
        return
    busy = (gr.update(interactive=False),) * 6
    idle = (gr.update(interactive=True),) * 6
    try:
        yield gr.update(), gr.update(), gr.update(), "Starting edit generation…", *busy, gr.update()
        pipe, note = runtime.get_pipe(settings, progress)
        outdir = runtime.run_dir(
            "edit", runtime.slug(request.id if request.id != "song" else style)
        )
        manifest = edit_flow.build_edit_manifest(
            source_rel=source_rel or "",
            before_abc=baseline_abc or "",
            after_abc=abc,
            cot=cot,
            seed=seed,
            cfg_scale=float(cfg_scale) if cfg_scale else None,
            invariants=(check_state or {}).get("result"),
            voices=(check_state or {}).get("voices", "both"),
            allow_tempo_change=bool((check_state or {}).get("allow_tempo_change")),
            allow_meter_change=bool((check_state or {}).get("allow_meter_change")),
            contract=(check_state or {}).get("contract", "exact"),
            allow_changes=bool(allow_changes),
            baseline=baseline_state,
        )
        song, result, elapsed = runtime.run_generation(
            pipe,
            request,
            outdir,
            abc_sampling=abc_sampling,
            semantic_sampling=sem_sampling,
            progress=progress,
            note=note,
            extra_manifest=manifest,
        )
        status = (
            runtime.generation_status(song, result, outdir, elapsed, request, note)
            + "\nedit_manifest.json records the source/edit hashes and the invariant result."
        )
        yield (
            str(outdir / "audio.flac"),
            (song.abc or ""),
            runtime.artifact_files(outdir, bool(song.abc)),
            status,
            *idle,
            str(outdir),
        )
    except Exception as exc:  # noqa: BLE001 — the UI reports, never crashes
        yield gr.update(), gr.update(), gr.update(), runtime.failure_text(exc), *idle, gr.update()
    finally:
        runtime.end_job()


def edit_compare(source_rel, last_run):
    if not (source_rel or "").strip():
        raise gr.Error("Load a source work first")
    if not (last_run or "").strip():
        raise gr.Error("Generate an edited version first")
    _item, det = library.load(runtime.RUNS, source_rel)
    if det is None:
        raise gr.Error("The source work no longer exists")
    return tools_tab.make_comparison(f"{det['path']}\n{last_run}")
