"""04 LIBRARY handlers and the LIBRARY → EDIT / COVER handoffs (see library.py)."""

from __future__ import annotations

import gradio as gr

from .. import library
from . import cover_tab, edit_tab, runtime, song_view


def library_open_in_edit(active):
    """Load the viewed/generated work into 03 EDIT and switch there."""
    rel = song_view.rel_of_run(active)
    (style, lyrics, abc, baseline_abc, source_rel, check_state, baseline_state, info, status) = (
        edit_tab.edit_load(rel)
    )
    choices = gr.update(
        choices=[value for _label, value in library_choices(library_mode("time", "desc"))[1]],
        value=rel,
    )
    return (
        style,
        lyrics,
        abc,
        baseline_abc,
        source_rel,
        check_state,
        baseline_state,
        info,
        status,
        choices,
        str((runtime.RUNS / rel).resolve()),
        gr.update(selected="edit"),
    )


def library_use_in_cover(active):
    """Load the viewed/generated work into 02 COVER and switch there."""
    rel = song_view.rel_of_run(active)
    abc, style, lyrics, status = cover_tab.cover_load(rel)
    choices = gr.update(
        choices=[value for _label, value in library_choices(library_mode("time", "desc"))[1]],
        value=rel,
    )
    return (
        abc,
        style,
        lyrics,
        status,
        choices,
        str((runtime.RUNS / rel).resolve()),
        gr.update(selected="cover"),
    )


def open_last_in_library(active):
    """Show a freshly generated run in 04 LIBRARY (list, details, player)."""
    rel = song_view.rel_of_run(active)
    _items, choices = library_choices(library_mode("time", "desc"))
    info, style, lyrics, abc, rename_box, rename_btn, status = _library_details([rel])
    return (
        gr.update(choices=choices, value=[rel]),
        info,
        style,
        lyrics,
        abc,
        rename_box,
        rename_btn,
        status,
        gr.update(selected="library"),
        rel,
        str((runtime.RUNS / rel).resolve()),
    )


# ───────────────── library tab (see library.py) ─────────
def library_mode(sort_key, sort_dir):
    key = "name" if str(sort_key) == "name" else "time"
    direction = "asc" if str(sort_dir) == "asc" else "desc"
    return f"{key}_{direction}"


def library_choices(sort_mode, include_pending=True):
    items = library.sort_items(library.scan(runtime.RUNS), sort_mode)
    if not include_pending:
        items = [item for item in items if not item.get("pending")]
    return items, [(library.label(item), item["rel"]) for item in items]


def _library_details(selected):
    """(info html, style, lyrics, abc, rename box, rename button, status)."""
    selected = list(selected or [])
    if not selected:
        return (
            library.render_empty_html("Select one work to see its details."),
            "",
            "",
            "",
            gr.update(value="", interactive=False),
            gr.update(interactive=False),
            "",
        )
    if len(selected) > 1:
        return (
            library.render_multi_html(selected),
            "",
            "",
            "",
            gr.update(value="", interactive=False),
            gr.update(interactive=False),
            f"{len(selected)} selected — pick one to view details, or delete the selection.",
        )
    item, det = library.load(runtime.RUNS, selected[0])
    if item is None or det is None:
        return (
            library.render_empty_html("That work no longer exists — refresh the list."),
            "",
            "",
            "",
            gr.update(value="", interactive=False),
            gr.update(interactive=False),
            "Not found.",
        )
    request = det.get("request") or {}
    return (
        library.render_info_html(item, det),
        request.get("style", "") or "",
        request.get("lyrics", "") or "",
        det.get("abc", "") or "",
        gr.update(value=item["name"], interactive=True),
        gr.update(interactive=True),
        f"{item['name']} · {library.format_seconds(item.get('duration'))}",
    )


def library_refresh(sort_key, sort_dir, selected=(), active=""):
    items, choices = library_choices(library_mode(sort_key, sort_dir))
    rels = {item["rel"] for item in items}
    keep = [rel for rel in (selected or ()) if rel in rels]
    keep_active = active if active in rels else ""
    info, style, lyrics, abc, rename_box, rename_btn, _ = _library_details(
        [keep_active] if keep_active else []
    )
    summary = (
        f"{len(items)} work(s) · {library.summarize(items)}"
        if items
        else "No works yet — generate something, then refresh."
    )
    return (
        gr.update(choices=choices, value=keep),
        info,
        style,
        lyrics,
        abc,
        rename_box,
        rename_btn,
        summary,
    )


def library_view(active):
    """Row click: show details only. Selection (checkboxes) is a separate state."""
    info, style, lyrics, abc, rename_box, rename_btn, _ = _library_details(
        [active] if active else []
    )
    return info, style, lyrics, abc, rename_box, rename_btn


def library_rename(active, new_name, sort_key, sort_dir, selected=()):
    if not active:
        info, style, lyrics, abc, rename_box, rename_btn, _ = _library_details([])
        return (
            gr.update(),
            info,
            style,
            lyrics,
            abc,
            rename_box,
            rename_btn,
            "Click a work first, then rename it.",
            gr.update(),
        )
    ok, message, new_rel = library.rename(runtime.RUNS, active, new_name)
    items, choices = library_choices(library_mode(sort_key, sort_dir))
    existing = {item["rel"] for item in items}
    target = new_rel if (ok and new_rel) else active
    # renaming must not disturb the checkboxes: keep the other selections,
    # and follow the renamed work if it was checked
    keep = []
    for rel in selected or ():
        if rel == active:
            if target in existing and target not in keep:
                keep.append(target)
        elif rel in existing and rel not in keep:
            keep.append(rel)
    info, style, lyrics, abc, rename_box, rename_btn, _ = _library_details([target])
    return (
        gr.update(choices=choices, value=keep),
        info,
        style,
        lyrics,
        abc,
        rename_box,
        rename_btn,
        message,
        gr.update(value=target),
    )


def library_delete_prepare(selected, sort_key, sort_dir):
    items, _ = library_choices(library_mode(sort_key, sort_dir))
    by_rel = {item["rel"]: item for item in items}
    chosen = [by_rel[rel] for rel in (selected or []) if rel in by_rel]
    if not chosen:
        return (
            library.render_confirm_html([]),
            [],
            gr.update(interactive=False),
            "Nothing selected.",
        )
    return (
        library.render_confirm_html(chosen),
        [item["rel"] for item in chosen],
        gr.update(interactive=True),
        f"{len(chosen)} item(s) ready to delete — confirm below.",
    )


def library_delete_confirm(pending, sort_key, sort_dir, active=""):
    _, message = library.delete(runtime.RUNS, pending or [])
    items, choices = library_choices(library_mode(sort_key, sort_dir))
    rels = {item["rel"] for item in items}
    keep_active = active if (active and active in rels) else ""
    if keep_active:
        info, style, lyrics, abc, rename_box, rename_btn, _ = _library_details([keep_active])
    else:
        empty = (
            "Select one work to see its details."
            if items
            else "No works yet — generate something, then refresh."
        )
        info, style, lyrics, abc = library.render_empty_html(empty), "", "", ""
        rename_box, rename_btn = (
            gr.update(value="", interactive=False),
            gr.update(interactive=False),
        )
    return (
        gr.update(choices=choices, value=[]),
        info,
        style,
        lyrics,
        abc,
        rename_box,
        rename_btn,
        library.render_confirm_html([]),
        [],
        gr.update(interactive=False),
        message,
        gr.update(value=keep_active),
    )


def library_delete_cancel():
    return library.render_confirm_html([]), [], gr.update(interactive=False), "Delete cancelled."
