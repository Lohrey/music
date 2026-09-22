"""Current-work identity and lifecycle stage for the producer-facing view.

Phase A of the SONG / STUDIO design: a pure, stdlib-only module that answers two
questions from the artifacts already on disk — *which work is this?* and *which
stage is it at?*  It never imports ``yue2`` and never writes anything; the UI
keeps the selected path in a hidden bridge (per browser, mirrored to
localStorage) and renders a one-line status band from :func:`band`.

Stages are derived from artifacts only, never inferred from activity:

    DRAFT    nothing selected / no known artifacts (shown only as "no work yet")
    SCORE    a plan or a transcription: ``score.abc`` without ``audio.flac``
    AUDIO    a song or a decode with ``audio.flac``
    REVISE   an edit attempt (``edit_manifest.json``) or a work a frozen
             baseline points at — including the "checked but not yet rendered"
             preparation state the producer is sitting in
    DONE     explicit ``finished.json`` marker (Phase C) — never guessed

"Listen" is deliberately **not** a stage: there is no reliable artifact for
"I listened".  The player and the comparison pages are actions available on
AUDIO / REVISE, not a state the system can claim.

Phase B constraints (stated here, enforced there): the SONG container must not
contain textboxes / radios / sliders / numbers / checkboxes (no second copy of
any editable state), and the status band must state facts only.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import textwrap
from datetime import datetime
from pathlib import Path

from . import library

STAGES = ("draft", "score", "audio", "revise", "done")
STAGE_LABELS = {
    "draft": "DRAFT",
    "score": "SCORE",
    "audio": "AUDIO",
    "revise": "REVISE",
    "done": "DONE",
}
KINDS = ("song", "plan", "transcription", "decode", "group")
KIND_LABELS = {
    "song": "SONG",
    "plan": "PLAN",
    "transcription": "TRANSCRIPTION",
    "decode": "DECODE",
    "group": "GROUP",
}

_ARTIFACTS = ("result.json", "audio.flac", "audio.mp3", "score.abc", "decode.json")
_STAMP_RE = re.compile(r"^(\d{8})-(\d{6})-(.*)$")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _request(path: Path) -> dict:
    request = _read_json(path / "request.json")
    if request is None:
        plan = _read_json(path / "plan.json")
        if isinstance(plan, dict) and isinstance(plan.get("request"), dict):
            request = plan["request"]
    return request if isinstance(request, dict) else {}


def _created(path: Path) -> datetime:
    match = _STAMP_RE.match(path.name)
    if match:
        try:
            return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.fromtimestamp(0)


def _abc_digest(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    return hashlib.sha256(textwrap.dedent(text).strip().encode("utf-8")).hexdigest()


def _baseline_for(root: Path, path: Path):
    """The frozen-baseline record that points at *path*, if any."""
    baselines = root / "baselines"
    if not baselines.is_dir():
        return None
    try:
        candidates = sorted(baselines.iterdir(), reverse=True)
    except OSError:
        return None
    for entry in candidates:
        record = _read_json(entry / "baseline.json")
        if not isinstance(record, dict):
            continue
        source = record.get("source") or {}
        if source.get("path") == str(path) or source.get("rel") == path.name:
            return record
    return None


def identify(root, path) -> dict | None:
    """Identify one run directory under *root*, or ``None`` when it is not a work.

    The returned dict is intentionally plain data (JSON-serialisable except for
    nothing — datetimes are pre-formatted) so the UI can cache or log it.
    """
    root = Path(root).resolve()
    path = Path(path).resolve()
    if path == root or root not in path.parents or not path.is_dir():
        return None

    result = _read_json(path / "result.json") or {}
    decode = _read_json(path / "decode.json")
    edit = _read_json(path / "edit_manifest.json")
    has_audio = (path / "audio.flac").is_file() or (path / "audio.mp3").is_file()
    has_score = (path / "score.abc").is_file()
    children: list[Path] = []
    if not any((path / name).exists() for name in _ARTIFACTS):
        # a batch / all-modes group: run.json plus per-mode subdirectories
        if (path / "run.json").is_file():
            try:
                children = [
                    entry
                    for entry in sorted(path.iterdir())
                    if entry.is_dir() and any((entry / name).exists() for name in _ARTIFACTS)
                ]
            except OSError:
                children = []
        if not children:
            return None

    if children:
        kind = "group"
        has_audio = any((child / "audio.flac").is_file() or (child / "audio.mp3").is_file() for child in children)
        has_score = any((child / "score.abc").is_file() for child in children)
    elif decode is not None or (has_audio and not result and (path / "latent.npy").is_file()):
        kind = "decode"
    elif result.get("task") and not result.get("weights"):
        kind = "transcription"
    elif result or has_audio:
        kind = "song"
    elif has_score:
        kind = "plan"
    else:
        return None

    baseline = _baseline_for(root, path)
    if (path / "finished.json").is_file():
        stage = "done"
    elif edit is not None or baseline is not None:
        stage = "revise"
    elif has_audio:
        stage = "audio"
    else:
        stage = "score"

    request = _request(path)
    created = _created(path)
    rel = path.relative_to(root).as_posix()
    source_rel = None
    if isinstance(edit, dict):
        source = edit.get("source") or {}
        source_rel = source.get("rel")
    return {
        "path": str(path),
        "rel": rel,
        "title": library.display_name(path.name),
        "kind": kind,
        "kind_label": KIND_LABELS[kind],
        "stage": stage,
        "stage_label": STAGE_LABELS[stage],
        "created": created.isoformat(timespec="seconds"),
        "created_label": created.strftime("%Y-%m-%d %H:%M"),
        "request": request,
        "family_key": str(request.get("id") or rel.split("/")[0]),
        "source_rel": source_rel,
        "baseline_rel": (baseline.get("baseline") or {}).get("rel") if baseline else None,
        "has_audio": has_audio,
        "has_score": has_score,
        "edited": edit is not None,
        "children": [child.relative_to(root).as_posix() for child in children],
    }


def last_event(work: dict) -> str:
    """What the artifact says happened last — a fact, not an inference."""
    label = work["created_label"]
    if work["stage"] == "done":
        return f"finished {label}"
    if work["edited"]:
        return f"edited {label}"
    return {
        "song": f"generated {label}",
        "decode": f"decoded {label}",
        "transcription": f"transcribed {label}",
        "plan": f"planned {label}",
        "group": f"created {label}",
    }.get(work["kind"], f"created {label}")


def band(root, path) -> str:
    """One factual line for the status band; empty when there is no current work."""
    work = identify(root, path)
    if work is None:
        return ""
    cells = [
        '<span class="bb-eyebrow">CURRENT</span>',
        f"<b>{html.escape(work['title'])}</b>",
        html.escape(work["stage_label"]),
        html.escape(work["kind_label"].lower()),
        html.escape(last_event(work)),
    ]
    return '<div id="bb-current-band">' + " · ".join(cells) + "</div>"


def family(root, run: dict, limit: int = 12) -> list[dict]:
    """Runs that plausibly belong to the same creative session as *run*.

    Heuristics, not identity: a shared request id and an explicit edit/cover link
    are marked ``exact``; same-group and ABC-digest matches are ``heuristic`` and
    the UI is expected to label them "possibly related".
    """
    root = Path(root).resolve()
    run_abc = _abc_digest((run.get("request") or {}).get("abc"))
    group = run["rel"].split("/")[0] if "/" in run["rel"] else ""
    out: list[dict] = []
    for item in library.scan(root):
        if item["rel"] == run["rel"]:
            continue
        candidate = identify(root, item["path"])
        if candidate is None:
            continue
        relations, exact = [], False
        if candidate["family_key"] and candidate["family_key"] == run.get("family_key"):
            relations.append("same request id")
            exact = True
        if candidate.get("source_rel") == run["rel"] or run.get("source_rel") == candidate["rel"]:
            relations.append("edit source / attempt")
            exact = True
        if candidate["kind"] == "transcription" and run_abc:
            score = Path(candidate["path"]) / "score.abc"
            if score.is_file() and _abc_digest(score.read_text(encoding="utf-8")) == run_abc:
                relations.append("cover source")
        if group and candidate["rel"].split("/")[0] == group:
            relations.append("same group")
        if relations:
            out.append(
                {
                    **candidate,
                    "relations": relations,
                    "confidence": "exact" if exact else "heuristic",
                }
            )
    out.sort(key=lambda entry: entry["created"], reverse=True)
    return out[:limit]


def next_actions(run: dict) -> list[dict]:
    """Actions the SONG view may offer for *run* (Phase B renders these).

    Every entry maps to an existing Studio handler; no action duplicates an
    editable control.
    """
    stage, kind = run.get("stage"), run.get("kind")
    if stage in (None, "draft"):
        return []
    actions: list[dict] = []
    if stage in ("audio", "revise", "done"):
        actions.append({"id": "listen", "label": "LISTEN"})
    if stage == "score":
        actions.append({"id": "render", "label": "RENDER IN STUDIO"})
    if stage in ("audio", "revise"):
        actions.append({"id": "edit", "label": "EDIT WORK"})
        seed = (run.get("request") or {}).get("seed")
        if isinstance(seed, int):
            actions.append({"id": "retry", "label": "TRY ANOTHER SEED", "seed": seed + 1})
    if stage == "revise":
        actions.insert(0, {"id": "check", "label": "OPEN CHECK IN STUDIO"})
    if kind == "transcription":
        actions.append({"id": "send", "label": "SEND TO GENERATE"})
    actions.append({"id": "library", "label": "OPEN IN LIBRARY"})
    return actions
