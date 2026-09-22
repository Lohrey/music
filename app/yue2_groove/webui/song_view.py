"""The SONG view: the director that follows the current work (see workflow.py).

It only states facts and offers actions that map to existing Studio handlers;
the current work travels through a hidden bridge textbox that the client
mirrors to localStorage (``mirror_current`` / ``publish_current``).
"""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from .. import workflow
from . import cover_tab, frontend, library_tab, runtime, tools_tab


def mirror_current(value):
    """Forward a *_last_run state into the hidden current-work bridge."""
    return (value or "").strip() or gr.update()


def _abs_of_run(value) -> str:
    """Absolute path of a run given an absolute path, a Library rel or empty."""
    if not (value or "").strip():
        return ""
    return str((runtime.RUNS / rel_of_run(value)).resolve())


def publish_current(path):
    """Band for the hidden current-work bridge; clears an invalid stored path once."""
    text = workflow.band(runtime.RUNS, path)
    if (path or "").strip() and not text:
        return "", gr.update(value="")  # (band, bridge): stale entry — clear both
    return text, gr.update()  # band rendered, bridge untouched


def rel_of_run(value) -> str:
    """Accept a run directory (absolute) or a Library rel and return the rel."""
    text = (value or "").strip()
    if not text:
        raise gr.Error("Pick or generate a work first")
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(runtime.RUNS.resolve()).as_posix()
        except (ValueError, OSError):
            raise gr.Error("That run is outside the runs directory") from None
    return text


# ───────────────── SONG view (director; see workflow.py for identity) ─────────────────
# The SONG view only states facts and offers actions that map to existing Studio
# handlers — it holds no editable component (no textbox / radio / slider / number /
# checkbox), so there is no second copy of STYLE / LYRICS / ABC / sampling state.
VIEW_CHOICES = ("song", "studio")
SONG_ACTIONS = ("listen", "render", "edit", "retry", "check", "send", "library")
SONG_STAGES = (
    ("draft", "DRAFT"),
    ("score", "SCORE"),
    ("audio", "AUDIO"),
    ("revise", "REVISE"),
    ("done", "DONE"),
)


def resolve_view(cli_view=None, cli_tab=None, env_view=None) -> tuple[str, int]:
    """Resolve the startup view: ``(view_mode, tab)``.

    ``view_mode`` is ``auto`` (the last stored choice, else the SONG default),
    ``song`` or ``studio``.  An explicit ``--tab N`` forces the Studio view on
    that tab; ``--view`` / ``YUE2_GROOVE_VIEW`` force a view for a fresh session.
    """
    if cli_tab is not None:
        return "studio", int(cli_tab)
    cli = (cli_view or "").strip().lower()
    if cli in VIEW_CHOICES:
        return cli, 0
    env = (env_view or "").strip().lower()
    if env in VIEW_CHOICES:
        return env, 0
    return "auto", 0


def _song_work(active):
    """Identify the current work from a bridge value, or ``None``."""
    if not (active or "").strip():
        return None
    try:
        return workflow.identify(runtime.RUNS, _abs_of_run(active))
    except gr.Error:
        return None


def _song_artifact(work: dict, name: str):
    """One artifact of a work, looking into the children of a group run."""
    paths = [Path(work["path"])]
    paths += [runtime.RUNS / child for child in work.get("children") or []]
    for base in paths:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _song_audio(work: dict):
    path = _song_artifact(work, "audio.flac")
    return str(path) if path is not None else None


def _song_abc(work: dict) -> str:
    path = _song_artifact(work, "score.abc")
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _song_stage_track(stage: str) -> str:
    """DRAFT → SCORE → AUDIO → REVISE → DONE, with the current stage highlighted.

    DONE is only ever reached through ``finished.json`` (``workflow.identify``),
    so the last cell stays unpainted unless the artifact says the work is done.
    """
    keys = [key for key, _label in SONG_STAGES]
    current = keys.index(stage) if stage in keys else 0
    cells = []
    for index, (_key, label) in enumerate(SONG_STAGES):
        classes = ["bb-stage"]
        if index == current:
            classes.append("bb-stage-current")
        elif index < current:
            classes.append("bb-stage-past")
        cells.append(f'<span class="{" ".join(classes)}">{label}</span>')
    return '<div id="bb-song-stage">' + '<span class="bb-stage-sep">→</span>'.join(cells) + "</div>"


def _song_identity(work: dict) -> str:
    return (
        '<div id="bb-song-identity">'
        f'<div class="bb-song-title">{html.escape(work["title"])}</div>'
        '<div class="bb-song-meta">'
        f"<span>KIND <b>{html.escape(work['kind_label'])}</b></span>"
        f"<span>STAGE <b>{html.escape(work['stage_label'])}</b></span>"
        f"<span>LAST <b>{html.escape(workflow.last_event(work))}</b></span>"
        f"<span>RUN <b>{html.escape(work['rel'])}</b></span>"
        "</div></div>"
    )


def _song_family(entries) -> str:
    if not entries:
        return ""
    rows = []
    for entry in entries:
        confidence = entry.get("confidence", "heuristic")
        tag = "exact" if confidence == "exact" else "possibly related"
        relations = html.escape(", ".join(entry.get("relations") or []))
        rows.append(
            '<div class="bb-family-item">'
            f'<b class="bb-family-hit" data-bb-run="{html.escape(entry["path"], quote=True)}">'
            f"{html.escape(entry['title'])}</b> "
            f'<span class="bb-family-rel">[{tag}] {relations}</span></div>'
        )
    return (
        '<div id="bb-song-family"><div class="bb-score-title">FAMILY</div>'
        '<div class="bb-family-list">' + "".join(rows) + "</div></div>"
    )


def render_song(active):
    """Everything the SONG view shows for the current work, or the empty state."""
    work = _song_work(active)
    if work is None:
        return (
            gr.update(visible=True),  # song_empty
            gr.update(visible=False),  # song_work
            "",
            "",
            "",  # identity, stage, family
            gr.update(value=None, visible=False),  # song_player
            frontend.score_panel("SONG SCORE", "No current work.", abc=""),
            *[gr.update(visible=False) for _ in SONG_ACTIONS],
            gr.update(visible=False),  # song_studio_btn
            gr.update(visible=False),  # song_compare_btn
        )
    actions = workflow.next_actions(work)
    ids = {action["id"] for action in actions}
    updates = []
    for key in SONG_ACTIONS:
        if key == "retry":
            seed = next((a.get("seed") for a in actions if a["id"] == "retry"), None)
            label = f"TRY SEED {seed}" if isinstance(seed, int) else "TRY ANOTHER SEED"
            updates.append(gr.update(visible=key in ids, value=label))
        else:
            updates.append(gr.update(visible=key in ids))
    audio = _song_audio(work)
    family = workflow.family(runtime.RUNS, work)
    comparable = bool(work.get("source_rel") or work.get("baseline_rel"))
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        _song_identity(work),
        _song_stage_track(work["stage"]),
        _song_family(family),
        gr.update(value=audio, visible=bool(audio)),
        frontend.score_panel("SONG SCORE", "No score for this work.", abc=_song_abc(work)),
        *updates,
        gr.update(visible=True),
        gr.update(visible=comparable),
    )


def song_render_action(active):
    """RENDER: open the Studio surface that renders this work's score.

    A plan goes to 01 GENERATE with the score attached; a transcription goes to
    02 COVER, where score-conditioned generation lives.  Nothing is generated
    silently.  The two branches return the same 12 outputs (the unused Studio
    fields are left untouched) so one Gradio event can serve both kinds.
    """
    work = _song_work(active)
    if work is None:
        raise gr.Error("That work no longer exists — refresh SONG")
    noop = gr.update()
    current = str(Path(work["path"]).resolve())
    if work["kind"] == "transcription":
        abc, style, lyrics, status, choices, _current, tabs = library_tab.library_use_in_cover(
            work["rel"]
        )
        return (
            noop,
            noop,
            noop,
            noop,
            abc,
            style,
            lyrics,
            status,
            choices,
            current,
            tabs,
            gr.update(value="studio"),
        )
    request = work.get("request") or {}
    return (
        gr.update(value=_song_abc(work)),  # abc
        gr.update(open=True),  # score_input_accordion
        gr.update(value=request.get("style") or ""),  # style
        gr.update(value=request.get("lyrics") or ""),  # lyrics
        noop,
        noop,
        noop,
        noop,  # cover abc/style/lyrics/status
        noop,  # cover_source
        current,  # current_bridge
        gr.update(selected="gen"),  # tabs
        gr.update(value="studio"),  # view_bridge
    )


def song_retry(active):
    """TRY ANOTHER SEED: prefill 01 GENERATE with the same request and seed + 1."""
    work = _song_work(active)
    if work is None:
        raise gr.Error("That work no longer exists — refresh SONG")
    request = work.get("request") or {}
    seed = request.get("seed")
    next_seed = seed + 1 if isinstance(seed, int) else 831001
    cot = request.get("cot") if request.get("cot") in ("full", "melody", "off") else "full"
    return (
        gr.update(value=request.get("style") or ""),  # style
        gr.update(value=request.get("lyrics") or ""),  # lyrics
        gr.update(value=cot),  # cot
        gr.update(value=next_seed),  # seed
        str(Path(work["path"]).resolve()),  # current_bridge
        gr.update(selected="gen"),  # tabs
        gr.update(value="studio"),  # view_bridge
    )


def song_send(active):
    """SEND TO GENERATE: the existing cover → generate handoff, with no cover UI."""
    work = _song_work(active)
    if work is None:
        raise gr.Error("That work no longer exists — refresh SONG")
    abc_text = _song_abc(work)
    if not abc_text.strip():
        raise gr.Error("That transcription has no score to send")
    request = work.get("request") or {}
    task = request.get("task") or "melody-full"
    abc, cot, style, lyrics, accordion, tabs, status = cover_tab.cover_send_to_generate(
        abc_text, task, request.get("style") or "", request.get("lyrics") or "", "both"
    )
    return (abc, cot, style, lyrics, accordion, tabs, status, gr.update(value="studio"))


def song_open_edit(active):
    """EDIT / CHECK: the existing library → edit handoff (baseline + invariant section)."""
    return (*library_tab.library_open_in_edit(active), gr.update(value="studio"))


def song_open_library(active):
    """OPEN IN LIBRARY: the existing library detail + player."""
    return (*library_tab.open_last_in_library(active), gr.update(value="studio"))


def song_open_studio(active):
    """OPEN IN STUDIO: the last surface that makes sense for this work."""
    work = _song_work(active)
    if work is None:
        return gr.update(selected="gen"), gr.update(value="studio")
    if work["kind"] == "transcription":
        tab = "cover"
    elif work["kind"] == "plan":
        tab = "gen"
    else:
        tab = "library"
    return gr.update(selected=tab), gr.update(value="studio")


def song_compare(active):
    """BUILD COMPARISON: source/baseline vs the current work, via the compare helper."""
    work = _song_work(active)
    if work is None:
        raise gr.Error("That work no longer exists — refresh SONG")
    other = work.get("source_rel") or work.get("baseline_rel")
    if not other:
        raise gr.Error("No baseline or source to compare against")
    source = (runtime.RUNS / other).resolve()
    if not source.is_dir():
        raise gr.Error("The source work for this comparison is missing")
    _path, link, status = tools_tab.make_comparison(f"{source}\n{Path(work['path']).resolve()}")
    # stay in SONG: the link opens in a new tab and the status is readable here
    return link, status
