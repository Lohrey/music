"""01 GENERATE, 06 DECODE and 07 BATCH handlers (plus ALL MODES and PLAN ONLY)."""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from .. import adapter
from . import runtime, tools_tab


def update_run_choices():
    return gr.update(choices=runtime.scan_runs())


def scan_batches():
    """[(label, batch_dir)] for batch or all-modes groups with saved songs."""
    out = []
    if not runtime.RUNS.is_dir():
        return out
    for d in sorted(runtime.RUNS.iterdir(), reverse=True):
        if not d.is_dir() or not ("batch" in d.name or "-allmodes-" in d.name):
            continue
        songs = [p for p in sorted(d.iterdir()) if p.is_dir() and (p / "result.json").is_file()]
        if songs:
            out.append((f"{d.name}  ({len(songs)} songs)", str(d)))
    return out


def update_batch_choices():
    return gr.update(choices=[c for c, _ in scan_batches()])


def fill_from_batch(label):
    """Fill the comparison input with every saved song of the chosen batch run."""
    if not label:
        raise gr.Error("Pick a batch run first")
    for choice, path in scan_batches():
        if choice == label:
            songs = [
                str(p)
                for p in sorted(Path(path).iterdir())
                if p.is_dir() and (p / "result.json").is_file()
            ]
            return "\n".join(songs)
    raise gr.Error("That batch directory is gone; press REFRESH LIST")


def reset_sampling_values():
    a, s = runtime.ABC_DEFAULTS, runtime.SEM_DEFAULTS
    return (
        a["temperature"],
        a["top_p"],
        a["top_k"],
        a["repetition_penalty"],
        a["penalty_window"],
        a["min_tokens"],
        a["max_tokens"],
        s["temperature"],
        s["top_p"],
        s["top_k"],
        s["repetition_penalty"],
        s["penalty_window"],
        s["min_tokens"],
        s["max_tokens"],
        "Protocol defaults (full)",
    )


def duration_text(tokens):
    seconds = max(0, int(tokens)) / runtime.TOKENS_PER_SECOND
    minutes = seconds / 60.0
    text = (
        f"**Estimated audio length:** up to ≈ {seconds:.0f} s "
        f"({minutes:.1f} min) at {int(tokens)} semantic tokens — the song ends when the score ends"
    )
    if int(tokens) > runtime.SEM_DEFAULTS["max_tokens"]:
        text += (
            "; a long song needs long lyrics and a larger ABC budget (the *Long song* preset "
            "sets both), and lyrics + score + song must fit the model's 24 576-token context"
        )
    return text


def generate(
    style,
    lyrics,
    cot,
    seed,
    cfg_scale,
    abc_text,
    out_id,
    preset,
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
    """Generator: disables the action buttons until the run finishes."""
    style, lyrics = runtime.request_texts(style, lyrics)  # empty input is an error
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
    kwargs = {}
    if (out_id or "").strip():
        kwargs["id"] = out_id.strip()
    if cfg_scale:
        kwargs["cfg_scale"] = float(cfg_scale)
    if (abc_text or "").strip():
        if cot == "off":
            raise gr.Error("An ABC score requires cot=full or cot=melody")
        # Pasted scores often carry the chat/TUI code-block indentation; the model
        # expects clean ABC lines ("X:1", "V: Vocal", ...). Strip the common indent.
        kwargs["abc"] = textwrap.dedent(abc_text).strip()
    try:
        request = adapter.song_request(
            style=style, lyrics=lyrics, cot=cot, seed=runtime.resolve_seed(seed), **kwargs
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
        # A fast double-click can queue a second run before the button disables.
        # Keep the buttons as they are (the running job owns them).
        yield (
            gr.update(),
            gr.update(),
            runtime.BUSY_MESSAGE,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.skip(),
        )
        return
    busy = (gr.update(interactive=False), gr.update(interactive=False))
    idle = (gr.update(interactive=True), gr.update(interactive=True))
    try:
        yield gr.update(), gr.update(), "Starting generation…", gr.update(), *busy, gr.skip()
        pipe, note = runtime.get_pipe(settings, progress)
        progress(0.02, desc="Starting generation…")
        outdir = runtime.run_dir(request.id if request.id != "song" else runtime.slug(style))
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
            str(outdir / "audio.flac"),
            (song.abc or ""),
            runtime.generation_status(song, result, outdir, elapsed, request, note),
            runtime.artifact_files(outdir, bool(song.abc)),
            *idle,
            str(outdir),
        )
    except Exception as exc:  # noqa: BLE001 — the UI reports, never crashes
        yield gr.update(), gr.update(), runtime.failure_text(exc), gr.update(), *idle, gr.skip()
    finally:
        runtime.end_job()


ALL_MODES = ("full", "melody", "off")


def generate_all_modes(
    style,
    lyrics,
    seed,
    cfg_scale,
    abc_text,
    out_id,
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
    """Generator: the same text request as full / melody / off, then a comparison.

    Mirrors upstream's ``all-modes``: text-only input (``cot=off`` cannot accept an
    ABC), one fresh directory per mode, a ``run.json`` summary at the group root,
    failures retained as ``failure.json``, and a listening bundle of the modes that
    completed.  Buttons are disabled while running; CANCEL stops after the current
    mode.
    """
    if (abc_text or "").strip():
        raise gr.Error(
            "ALL MODES is text-only: cot=off cannot accept an ABC score. Clear "
            "ABC SCORE, or generate the modes one by one with their own ABC."
        )
    style, lyrics = runtime.request_texts(style, lyrics)
    base_id = (out_id or "").strip() or runtime.slug(style)
    try:
        seed = runtime.resolve_seed(seed)  # one seed for all three modes
        adapter.song_request(style=style, lyrics=lyrics, cot="full", seed=seed, id=base_id)
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"Invalid request: {exc}") from exc
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
            gr.update(),
            gr.update(),
            gr.skip(),
        )
        return
    busy = (gr.update(interactive=False),) * 3
    idle = (gr.update(interactive=True),) * 3
    try:
        yield (
            "Starting ALL MODES (full → melody → off)…",
            gr.update(),
            gr.update(),
            *busy,
            gr.skip(),
        )
        pipe, note = runtime.get_pipe(settings, progress)
        root = runtime.run_dir("allmodes", runtime.slug(base_id))
        results, files, done_dirs = [], [], []
        for position, mode in enumerate(ALL_MODES):
            mode_dir = root / mode
            if runtime.CANCEL.is_set():
                results.append({"mode": mode, "status": "cancelled"})
                break
            kwargs = {
                "style": style,
                "lyrics": lyrics,
                "cot": mode,
                "seed": seed,
                "id": f"{base_id}_{mode}",
            }
            if cfg_scale:
                kwargs["cfg_scale"] = float(cfg_scale)
            request = adapter.song_request(**kwargs)

            def scoped(value, desc=None, position=position, mode=mode):
                fraction = position + (0.0 if value is None else min(1.0, float(value)))
                progress(
                    min(1.0, fraction / len(ALL_MODES)),
                    desc=f"[{mode}] {desc}" if desc else f"[{mode}]",
                )

            try:
                mode_dir.mkdir(parents=True, exist_ok=True)
                (mode_dir / "input.json").write_text(
                    json.dumps(request.to_dict(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                song, receipt, elapsed = runtime.run_generation(
                    pipe,
                    request,
                    mode_dir,
                    abc_sampling=abc_sampling,
                    semantic_sampling=sem_sampling,
                    progress=scoped,
                    note=note,
                )
                results.append(
                    {
                        "mode": mode,
                        "status": "complete",
                        "audio_seconds": receipt["audio_seconds"],
                        "truncated": receipt["truncated"],
                        "seconds": round(elapsed, 2),
                    }
                )
                files += runtime.artifact_files(mode_dir, bool(song.abc))
                done_dirs.append(str(mode_dir))
            except InterruptedError as exc:
                record = {"mode": mode, "status": "cancelled", "error": str(exc)}
                (mode_dir / "failure.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                results.append(record)
                break
            except Exception as exc:  # noqa: BLE001 — keep going; the failure is retained
                record = {
                    "mode": mode,
                    "status": "failed",
                    "type": type(exc).__name__,
                    "error": str(exc),
                }
                mode_dir.mkdir(parents=True, exist_ok=True)
                (mode_dir / "failure.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                results.append(record)
            progress((position + 1) / len(ALL_MODES), desc=f"{mode} finished")

        root.mkdir(parents=True, exist_ok=True)
        summary = {
            "action": "all-modes",
            "request": {
                "style": style,
                "lyrics": lyrics,
                "seed": seed,
                "cfg_scale": float(cfg_scale) if cfg_scale else None,
                "id": base_id,
            },
            "results": results,
        }
        (root / "run.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        link, comparison = "", ""
        if len(done_dirs) >= 2:
            progress(0.98, desc="Building listening comparison…")
            try:
                _html, link, comparison = tools_tab.make_comparison("\n".join(done_dirs))
            except gr.Error as exc:
                comparison = f"comparison failed: {exc}"

        lines = [f"ALL MODES finished: {root}", ""]
        for record in results:
            if record["status"] == "complete":
                lines.append(
                    f"  {record['mode']:<7} complete  {record['audio_seconds']:.1f}s "
                    f"audio in {record['seconds']:.0f}s  truncated={record['truncated']}"
                )
            else:
                lines.append(f"  {record['mode']:<7} {record['status']}  {record.get('error', '')}")
        lines += ["", note]
        if comparison:
            lines.append(comparison)
        if len(done_dirs) < 2:
            lines.append("(the comparison needs at least two completed modes)")
        yield "\n".join(lines), files, link, *idle, str(root)
    except Exception as exc:  # noqa: BLE001
        yield (
            f"ALL MODES failed: {type(exc).__name__}: {exc}",
            gr.update(),
            gr.update(),
            *idle,
            gr.skip(),
        )
    finally:
        runtime.end_job()


def plan_only(
    style,
    lyrics,
    cot,
    seed,
    cfg_scale,
    out_id,
    abc_temp,
    abc_p,
    abc_k,
    abc_rep,
    abc_win,
    abc_min,
    abc_max,
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
    """Generator: disables the action buttons until planning finishes."""
    style, lyrics = runtime.request_texts(style, lyrics)  # empty input is an error
    abc_sampling = runtime.sampling(
        abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max, "ABC phase"
    )
    kwargs = {}
    if (out_id or "").strip():
        kwargs["id"] = out_id.strip()
    if cfg_scale:
        kwargs["cfg_scale"] = float(cfg_scale)
    try:
        request = adapter.song_request(
            style=style, lyrics=lyrics, cot=cot, seed=runtime.resolve_seed(seed), **kwargs
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
        yield gr.update(), runtime.BUSY_MESSAGE, gr.update(), gr.update(), gr.update(), gr.skip()
        return
    busy = (gr.update(interactive=False), gr.update(interactive=False))
    idle = (gr.update(interactive=True), gr.update(interactive=True))
    try:
        yield gr.update(), "Planning score…", gr.update(), *busy, gr.skip()
        pipe, note = runtime.get_pipe(settings, progress)
        progress(0.05, desc="Planning score…")
        plan = adapter.plan(
            pipe, request, abc_sampling=abc_sampling, cancelled=runtime.CANCEL.is_set
        )
        outdir = runtime.run_dir(
            request.id if request.id != "song" else runtime.slug(style), "plan"
        )
        plan.save(outdir)
        files = [
            str(outdir / n)
            for n in ("score.abc", "plan.json", "abc_tokens.npy", "prefix.npy")
            if (outdir / n).exists()
        ]
        yield (plan.abc or ""), f"Plan saved: {outdir}\n{note}", files, *idle, str(outdir)
    except Exception as exc:  # noqa: BLE001 — the UI reports, never crashes
        yield gr.update(), runtime.failure_text(exc, "Planning"), gr.update(), *idle, gr.skip()
    finally:
        runtime.end_job()


def decode_run(
    source_dir,
    latent_file,
    dec_vae_choice,
    dec_vae_custom,
    dec_vae_revision,
    full_decode,
    device,
    dtype,
    backend,
    quantization,
    offload_ar,
    budget,
    ode_steps,
    vae_core_frames,
    model,
    gen_vae_choice,
    gen_vae_custom,
    revision,
    gen_vae_revision,
    offline,
    progress=gr.Progress(),
):
    """Generator: re-decodes saved latents; disables the decode button while running."""
    path = None
    if (latent_file or "").strip():
        path = Path(latent_file.strip())
    elif (source_dir or "").strip():
        path = Path(source_dir.strip()) / "latent.npy"
    if path is None or not path.is_file():
        raise gr.Error("Provide latent.npy (upload a file or point at a saved run directory)")
    latents = np.load(path, allow_pickle=False)
    if latents.ndim != 2 or latents.shape[1] != 64:
        raise gr.Error(f"Latent shape must be [T,64], got {latents.shape}")

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
        gen_vae_choice,
        gen_vae_custom,
        revision,
        gen_vae_revision,
        offline,
    )
    if not runtime.try_start_job():
        yield gr.update(), runtime.BUSY_MESSAGE, gr.update(), gr.skip()
        return
    try:
        yield gr.update(), "Decoding…", gr.update(interactive=False), gr.skip()
        pipe, note = runtime.get_pipe(settings, progress)
        override = None
        if dec_vae_choice != "keep":
            override, vae_name = runtime.resolve_vae(dec_vae_choice, dec_vae_custom)
        else:
            vae_name = "source-generation VAE"
        if dec_vae_choice != "keep" and dec_vae_revision:
            override = str(
                adapter.resolve_model(
                    override, revision=dec_vae_revision, local_files_only=bool(offline)
                )
            )
        progress(0.1, desc=f"Decoding {latents.shape[0]} frames ({vae_name})…")
        t0 = time.perf_counter()

        def on_progress(done, total):
            progress(
                0.1 + 0.85 * min(1.0, done / max(1, total)),
                desc=f"Decoding audio: chunk {done}/{total}",
            )

        audio = adapter.decode(
            pipe, latents, full=bool(full_decode), vae=override, on_progress=on_progress
        )
        seconds = time.perf_counter() - t0
        outdir = runtime.run_dir("decode", runtime.slug(vae_name))
        outdir.mkdir(parents=True, exist_ok=True)
        sf.write(outdir / "audio.flac", audio, 48000, subtype="PCM_24")
        np.save(outdir / "latent.npy", latents.astype(np.float32))
        (outdir / "decode.json").write_text(
            json.dumps(
                {
                    "operation": "decode_cached_latents",
                    "source_latent": str(path),
                    "vae": override or str(pipe.vae_dir),
                    "full_decode": bool(full_decode),
                    "core_frames": None if full_decode else pipe.vae_core_frames,
                    "seconds": seconds,
                    "device": str(pipe.device),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        runtime.write_local_env(outdir, pipe, note)
        status = (
            f"Decoded {len(audio) / 48000:.1f}s audio in {seconds:.0f}s\n"
            f"VAE={vae_name}  mode={'full' if full_decode else 'tiled'}  "
            f"source={path}\nrun directory: {outdir}\n{note}"
        )
        yield str(outdir / "audio.flac"), status, gr.update(interactive=True), str(outdir)
    except Exception as exc:  # noqa: BLE001 — the UI reports, never crashes
        yield (
            gr.update(),
            runtime.failure_text(exc, "Decode"),
            gr.update(interactive=True),
            gr.skip(),
        )
    finally:
        runtime.end_job()


def _row_sampling(base, overrides: dict | None, label: str):
    """A batch row's sampling: the shared sliders with the row's overrides on top."""
    v = {k: (overrides or {}).get(k, getattr(base, k)) for k in adapter.sampling_fields(base)}
    return runtime.sampling(
        v["temperature"],
        v["top_p"],
        v["top_k"],
        v["repetition_penalty"],
        v["penalty_window"],
        v["min_tokens"],
        v["max_tokens"],
        label,
    )


def batch_generate(
    jsonl_text,
    jsonl_file,
    out_id,
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
    """Generator: runs a JSONL queue; disables the batch button while running."""
    text = ""
    base = Path.cwd()
    if jsonl_file:
        base = Path(jsonl_file).parent
        text = Path(jsonl_file).read_text(encoding="utf-8")
    elif jsonl_text.strip():
        text = jsonl_text
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise gr.Error("Provide JSONL (one request per line) or upload a .jsonl file")
    ids = [r.get("id") for r in rows]
    if any(x is None for x in ids) or len(set(ids)) != len(ids):
        raise gr.Error("Every line needs a unique id")
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
        yield gr.update(), runtime.BUSY_MESSAGE, gr.update(), gr.skip()
        return
    try:
        yield gr.update(), "Starting batch…", gr.update(interactive=False), gr.skip()
        pipe, note = runtime.get_pipe(settings, progress)
        outdir = runtime.run_dir("batch", runtime.slug(out_id or "batch"))
        outdir.mkdir(parents=True, exist_ok=True)
        allowed = {"style", "tags", "lyrics", "cot", "seed", "abc", "cfg_scale", "id"}
        results, failures = [], 0
        batch_start = time.perf_counter()
        for index, row in enumerate(rows, 1):
            if runtime.CANCEL.is_set():
                results.append([row.get("id"), "cancelled", "", "", ""])
                break
            row_start = time.perf_counter()
            kwargs = {k: v for k, v in row.items() if k in allowed and k != "id"}
            if isinstance(kwargs.get("seed"), int) and kwargs["seed"] < 0:
                kwargs["seed"] = runtime.random_seed()  # -1 = draw one per row
            if "abc_path" in row:
                kwargs["abc"] = (base / row["abc_path"]).read_text(encoding="utf-8")
            try:
                request = adapter.song_request(id=row["id"], **kwargs)
            except (ValueError, TypeError) as exc:
                results.append([row["id"], f"invalid: {exc}", "", "", ""])
                failures += 1
                continue
            a_s = _row_sampling(abc_sampling, row.get("abc_sampling"), "ABC phase")
            s_s = _row_sampling(sem_sampling, row.get("semantic_sampling"), "semantic phase")

            def on_token(phase, token, count=1, index=index):
                progress((index - 1 + 0.5) / len(rows), desc=f"Song {index}/{len(rows)}: {phase}")

            def on_progress(stage, done, total, index=index):
                span = 1.0 / len(rows)
                if stage == "nar":
                    inner = 0.55 + 0.40 * min(1.0, done / max(1, total))
                    desc = f"Song {index}/{len(rows)}: synthesizing {done}/{total}"
                else:
                    inner = 0.95 + 0.05 * min(1.0, done / max(1, total))
                    desc = f"Song {index}/{len(rows)}: decoding {done}/{total}"
                progress(min(1.0, (index - 1) * span + span * inner), desc=desc)

            try:
                song = adapter.generate(
                    pipe,
                    request,
                    abc_sampling=a_s,
                    semantic_sampling=s_s,
                    cancelled=runtime.CANCEL.is_set,
                    on_token=on_token,
                    on_progress=on_progress,
                )
                receipt = song.save_artifacts(outdir / row["id"])
                runtime.write_local_env(outdir / row["id"], pipe, note)
                results.append(
                    [
                        row["id"],
                        "complete",
                        f"{receipt['audio_seconds']:.1f}s",
                        f"{time.perf_counter() - row_start:.0f}s",
                        str(outdir / row["id"]),
                    ]
                )
            except InterruptedError:
                results.append(
                    [row["id"], "cancelled", "", f"{time.perf_counter() - row_start:.0f}s", ""]
                )
                break
            except Exception as exc:  # noqa: BLE001
                failures += 1
                results.append(
                    [
                        row["id"],
                        f"failed: {type(exc).__name__}: {exc}",
                        "",
                        f"{time.perf_counter() - row_start:.0f}s",
                        "",
                    ]
                )
            progress(index / len(rows), desc=f"Completed {index}/{len(rows)}")
        total = time.perf_counter() - batch_start
        status = (
            f"Batch finished: {len(results)} rows, {failures} failed in {total:.0f}s "
            f"({total / max(1, len(results)):.0f}s per song)\n"
            f"run directory: {outdir}\n{note}"
        )
        yield results, status, gr.update(interactive=True), str(outdir)
    except Exception as exc:  # noqa: BLE001 — the UI reports, never crashes
        yield (
            gr.update(),
            runtime.failure_text(exc, "Batch"),
            gr.update(interactive=True),
            gr.skip(),
        )
    finally:
        runtime.end_job()


def apply_preset(name):
    presets = {
        "Protocol defaults (full)": (32, 4096, 200, 9000),
        # 15 000 = 10:00; a 10-minute score is ~6 000 ABC tokens, and 6 144 + typical lyrics
        # still leave that much of the context (runtime.fit_semantic_budget)
        "Long song (~10 min)": (32, 6144, 200, 15000),
        "Preview (~1–1.5 min song)": (32, 700, 64, 2200),
        "Quick test (~20 s)": (16, 256, 32, 512),
    }
    a_min, a_max, s_min, s_max = presets[name]
    return (
        gr.update(value=a_min),
        gr.update(value=a_max),
        gr.update(value=s_min),
        gr.update(value=s_max),
    )
