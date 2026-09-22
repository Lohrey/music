"""Library backend for YUE2 // GROOVE.

This module is deliberately self-contained: it reads the artifact directories
written by the pipeline **by file convention** (``request.json``,
``result.json``, ``config.json``, ``score.abc``, ``audio.flac``, ``plan.json``,
``decode.json``) and never imports from the ``yue2`` package.  Upstream
refactors therefore cannot break the Library; if an artifact name ever changes,
the affected field simply renders as "—".

Everything here is plain Python returning strings/structures.  The Gradio
wiring lives in ``webui.py``; the pane's stylesheet and the player / score
script are ``static/library.css`` and ``static/library.js``.
"""

from __future__ import annotations

import contextlib
import html
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

# a directory that holds any of these is a Library entry
ARTIFACTS = ("result.json", "audio.flac", "audio.mp3", "score.abc")
# written at the start of a run and removed only once every artifact is flushed;
# a directory that still has one is a run that did not finish
PENDING = "pending.json"
# freeze records keep a copy of score.abc; they are provenance, not works
SKIP_DIRS = ("baselines",)
STAMP_RE = re.compile(r"^(\d{8})-(\d{6})-(.*)$")
SORTS = ("time_desc", "time_asc", "name_asc", "name_desc")
MAX_NAME = 80


# ─────────────────────────────────────────────────────────────── helpers ──
def format_seconds(value) -> str:
    try:
        total = round(float(value))
    except (TypeError, ValueError):
        return "—"
    if total <= 0:
        return "—"
    return f"{total // 60}:{total % 60:02d}"


def format_bytes(value) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return "—"


def display_name(folder: str) -> str:
    match = STAMP_RE.match(folder)
    return match.group(3) if match else folder


def _created(path: Path) -> datetime:
    match = STAMP_RE.match(path.name)
    if match:
        try:
            return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.fromtimestamp(0)


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                with contextlib.suppress(OSError):
                    total += entry.stat().st_size
    except OSError:
        pass
    return total


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _request_of(path: Path):
    """request.json, else the request embedded in plan.json (PLAN ONLY runs)."""
    request = _read_json(path / "request.json")
    if request is None:
        plan = _read_json(path / "plan.json")
        if isinstance(plan, dict) and isinstance(plan.get("request"), dict):
            request = plan["request"]
    return request


# ──────────────────────────────────────────────────────────────── scan ──
def _kind(result: dict, has_audio: bool, has_score: bool) -> str:
    if result and (result.get("weights") or has_audio):
        return "song"
    if result and result.get("task"):  # SheetSage2 transcription result.json
        return "transcription"
    if result:
        return "song"
    if has_audio:
        return "decode"
    if has_score:
        return "plan"
    return "data"


def _item(root: Path, path: Path) -> dict:
    request = _request_of(path) or {}
    result = _read_json(path / "result.json") or {}
    pending = _read_json(path / PENDING) or {}
    has_audio = (path / "audio.flac").is_file() or (path / "audio.mp3").is_file()
    has_score = (path / "score.abc").is_file()
    kind = _kind(result, has_audio, has_score)
    if pending and not result:
        kind = "incomplete"  # audio without result.json is a run, not a decode
    truncated = result.get("truncated")
    if isinstance(truncated, dict):
        truncated = any(bool(v) for v in truncated.values())
    return {
        "rel": path.relative_to(root).as_posix(),
        "path": str(path),
        "folder": path.name,
        "name": display_name(path.name),
        "created": _created(path),
        "created_label": _created(path).strftime("%Y-%m-%d %H:%M"),
        "duration": result.get("audio_seconds"),
        "status": result.get("status") or pending.get("status", ""),
        "pending": bool(pending),
        "pending_error": pending.get("error", ""),
        "truncated": bool(truncated),
        "cot": request.get("cot", ""),
        "seed": request.get("seed"),
        "cfg": request.get("cfg_scale"),
        "kind": kind,
        "has_audio": has_audio,
        "has_score": has_score or bool(request.get("abc")),
        "size": _dir_bytes(path),
    }


def scan(root) -> list[dict]:
    """Every artifact directory under *root* (one level of nesting for batches)."""
    root = Path(root).resolve()
    items: list[dict] = []
    if not root.is_dir():
        return items

    def visit(directory: Path, depth: int) -> None:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for path in entries:
            if not path.is_dir() or path.name.startswith(".") or path.name in SKIP_DIRS:
                continue
            if (path / "index.html").is_file() and (path / "manifest.json").is_file():
                continue  # a listening comparison bundle, not a work
            if any((path / name).exists() for name in ARTIFACTS) or (path / PENDING).is_file():
                items.append(_item(root, path))
                continue  # never descend into a work directory
            if depth < 2:
                visit(path, depth + 1)

    visit(root, 1)
    return items


def sort_items(items, mode: str = "time_desc") -> list[dict]:
    if mode == "time_asc":
        return sorted(items, key=lambda i: (i["created"], i["name"].lower()))
    if mode == "name_asc":
        return sorted(items, key=lambda i: (i["name"].lower(), i["created"]))
    if mode == "name_desc":
        return sorted(items, key=lambda i: (i["name"].lower(), i["created"]), reverse=True)
    return sorted(items, key=lambda i: (i["created"], i["name"].lower()), reverse=True)


def label(item: dict) -> str:
    parts = [item["name"], item["created"].strftime("%m-%d %H:%M")]
    if item.get("pending"):
        parts.append("INCOMPLETE")
    if item["kind"] == "plan":
        parts.append("PLAN")
    elif item["kind"] == "decode":
        parts.append("DECODE")
    elif item["kind"] == "transcription":
        parts.append("TRANSCRIPTION")
    else:
        parts.append(format_seconds(item.get("duration")))
    if item.get("cot"):
        parts.append(str(item["cot"]).upper())
    cfg = item.get("cfg")
    if isinstance(cfg, (int, float)) and float(cfg) != 1.0:
        parts.append(f"cfg {float(cfg):g}")
    if item.get("truncated"):
        parts.append("TRUNCATED")
    elif item.get("status") and item["status"] != "complete":
        parts.append(str(item["status"]).upper())
    return " · ".join(str(p) for p in parts if p not in (None, "", "—"))


def summarize(items) -> str:
    songs = sum(1 for i in items if i["kind"] == "song")
    plans = sum(1 for i in items if i["kind"] == "plan")
    decodes = sum(1 for i in items if i["kind"] == "decode")
    transcriptions = sum(1 for i in items if i["kind"] == "transcription")
    total = sum(int(i.get("size") or 0) for i in items)
    incomplete = sum(1 for i in items if i.get("pending"))
    bits = [f"{len(items)} item(s)"]
    for count, word in (
        (songs, "song"),
        (plans, "plan"),
        (decodes, "decode"),
        (transcriptions, "transcription"),
    ):
        if count:
            bits.append(f"{count} {word}(s)")
    if incomplete:
        bits.append(f"{incomplete} incomplete")
    bits.append(format_bytes(total))
    return " · ".join(bits)


def load(root, rel: str):
    """(item, details) for a relative directory name, or (None, None)."""
    root = Path(root).resolve()
    path = (root / rel).resolve()
    if root not in path.parents or not path.is_dir():
        return None, None
    return _item(root, path), details(path)


# ───────────────────────────────────────────────────────────── details ──
def details(path) -> dict:
    path = Path(path)
    request = _request_of(path) or {}
    result = _read_json(path / "result.json") or {}
    config = _read_json(path / "config.json") or {}
    local_env = _read_json(path / "local_env.json") or {}
    abc = _read_text(path / "score.abc") or str(request.get("abc") or "")
    audio = path / "audio.flac"
    if not audio.is_file() and (path / "audio.mp3").is_file():
        audio = path / "audio.mp3"
    files = []
    try:
        for entry in sorted(path.iterdir()):
            if entry.is_file():
                with contextlib.suppress(OSError):
                    files.append({"name": entry.name, "bytes": entry.stat().st_size})
    except OSError:
        pass
    return {
        "path": str(path),
        "request": request,
        "result": result,
        "config": config,
        "local_env": local_env,
        "abc": abc,
        "audio": str(audio) if audio.is_file() else None,
        "files": files,
    }


def player_html(audio_path: str) -> str:
    src = "/gradio_api/file=" + quote(str(audio_path))
    return (
        '<div class="bb-player" data-bb-player>'
        f'<audio preload="metadata" src="{html.escape(src, quote=True)}"></audio>'
        '<div class="bb-prow">'
        '<button type="button" class="bb-pback" data-bb-back'
        ' aria-label="Restart from the beginning" title="Restart from the beginning"></button>'
        '<button type="button" class="bb-pjump" data-bb-jump="-15"'
        ' aria-label="Back 15 seconds" title="Back 15 seconds">15</button>'
        '<button type="button" class="bb-pbtn" data-bb-play'
        ' aria-label="Play" title="Play"></button>'
        '<button type="button" class="bb-pjump bb-fwd" data-bb-jump="15"'
        ' aria-label="Forward 15 seconds" title="Forward 15 seconds">15</button>'
        '<button type="button" class="bb-pend" data-bb-end'
        ' aria-label="Jump to the end" title="Jump to the end"></button>'
        '<span class="bb-ptime" data-bb-time>0:00 / 0:00</span>'
        "</div>"
        '<div class="bb-pseek" data-bb-seek>'
        '<canvas class="bb-pviz" data-bb-viz aria-hidden="true"></canvas>'
        '<div class="bb-pfill"></div>'
        "</div>"
        "</div>"
    )


def _table(title: str, rows) -> str:
    rows = [(name, v) for name, v in rows if v not in (None, "", "—", {})]
    if not rows:
        return ""
    out = [
        f'<div class="bb-lib-section">{html.escape(title)}</div>',
        '<table class="bb-lib-table"><tbody>',
    ]
    for name, value in rows:
        out.append(f"<tr><td>{html.escape(str(name))}</td><td>{html.escape(str(value))}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def render_info_html(item: dict, det: dict) -> str:
    request = det.get("request") or {}
    result = det.get("result") or {}
    config = det.get("config") or {}
    local_env = det.get("local_env") or {}
    generation = config.get("generation") or {}
    abc_sampling = generation.get("abc") or {}
    sem_sampling = generation.get("semantic") or {}
    timing = result.get("timing") or {}
    semantic = timing.get("semantic") or {}

    out = [
        '<div class="bb-lib-card">',
        f'<div class="bb-lib-title">{html.escape(item["name"])}</div>',
        '<div class="bb-lib-sub">'
        + " · ".join(
            html.escape(str(b))
            for b in (
                item.get("created_label"),
                format_seconds(item.get("duration")),
                "INCOMPLETE" if item.get("pending") else item.get("kind", "").upper(),
            )
            if b
        )
        + "</div>",
        (
            f'<div class="bb-lib-sub bb-lib-dim">{html.escape(item.get("rel", ""))}'
            f" · {html.escape(format_bytes(item.get('size')))}</div>"
        ),
    ]

    if item.get("pending"):
        detail = item.get("pending_error") or item.get("status") or "unfinished"
        out.append(
            '<div class="bb-lib-note bb-lib-incomplete">INCOMPLETE — this run did not '
            "finish, so it was never saved as a song. The artifacts below are partial."
            f'<span class="bb-lib-dim"> ({html.escape(str(detail))})</span></div>'
        )

    chips = []
    if request.get("cot"):
        chips.append("cot " + str(request["cot"]).upper())
    if request.get("seed") is not None:
        chips.append("seed " + str(request["seed"]))
    if isinstance(request.get("cfg_scale"), (int, float)):
        chips.append(f"cfg {float(request['cfg_scale']):g}")
    if config.get("device"):
        chips.append(f"{config['device']} / {config.get('model_dtype', '')}".strip(" /"))
    if semantic.get("output_tokens"):
        chips.append(f"{int(semantic['output_tokens']):,} tok")
    if semantic.get("output_tps"):
        chips.append(f"{float(semantic['output_tps']):.1f} tok/s")
    if item.get("truncated"):
        chips.append("TRUNCATED")
    if chips:
        out.append(
            '<div class="bb-lib-chips">'
            + "".join(f'<span class="bb-chip">{html.escape(c)}</span>' for c in chips)
            + "</div>"
        )

    if det.get("audio"):
        out.append(player_html(det["audio"]))
    elif item.get("pending"):
        out.append(
            '<div class="bb-lib-note">No audio yet — the run was interrupted before '
            "the song was written.</div>"
        )
    else:
        out.append('<div class="bb-lib-note">No audio in this directory — score-only plan.</div>')

    # The styled score below must render even while the (lazily mounted) ABC
    # accordion is closed, so the source travels with the card itself.
    if det.get("abc"):
        out.append(f'<pre id="bb-lib-abc-src" hidden>{html.escape(det["abc"])}</pre>')

    out.append(
        _table(
            "request",
            [
                ("PLAN MODE", request.get("cot")),
                ("SEED", request.get("seed")),
                ("CFG SCALE", request.get("cfg_scale")),
                ("REQUEST ID", request.get("id")),
                ("EXTERNAL ABC", "yes" if request.get("abc") else "no"),
            ],
        )
    )
    out.append(
        _table(
            "sampling",
            [
                ("ABC temperature", abc_sampling.get("temperature")),
                ("ABC top_p", abc_sampling.get("top_p")),
                ("ABC top_k", abc_sampling.get("top_k")),
                ("ABC max_tokens", abc_sampling.get("max_tokens")),
                ("SEMANTIC temperature", sem_sampling.get("temperature")),
                ("SEMANTIC top_p", sem_sampling.get("top_p")),
                ("SEMANTIC top_k", sem_sampling.get("top_k")),
                ("SEMANTIC max_tokens", sem_sampling.get("max_tokens")),
            ],
        )
    )
    ode = " · ".join(
        str(x) for x in (generation.get("ode_steps"), generation.get("ode_method")) if x
    )
    # local_env.json (a sidecar written by the web UI) records the device/dtype that
    # actually ran; upstream's config.json hardcodes model_dtype=bfloat16, so the two
    # disagree after an explicit float32 cast -- shown only when they differ.
    actual = None
    if local_env:
        bits = []
        if local_env.get("dtype") and local_env["dtype"] != config.get("model_dtype"):
            bits.append(
                f"{local_env['dtype']} (cast at load; config.json records "
                f"{config.get('model_dtype') or '—'})"
            )
        if local_env.get("device") and local_env["device"] != config.get("device"):
            bits.append(f"device {local_env['device']}")
        actual = " · ".join(bits) or None
    out.append(
        _table(
            "runtime",
            [
                ("DEVICE", config.get("device")),
                ("MODEL DTYPE", config.get("model_dtype")),
                ("WEBUI ACTUAL", actual),
                ("VAE DTYPE", config.get("vae_dtype")),
                ("BACKEND", config.get("backend")),
                ("QUANTIZATION", config.get("quantization")),
                ("ODE", ode),
                ("CONTEXT", generation.get("context")),
                ("VAE CORE FRAMES", config.get("vae_core_frames")),
                ("VAE DECODE", config.get("vae_decode")),
                ("OFFLOAD AR", config.get("offload_ar")),
                ("MEMORY BUDGET (GiB)", config.get("memory_budget_gib")),
            ],
        )
    )
    out.append(
        _table(
            "timing",
            [
                ("AUDIO LENGTH", format_seconds(result.get("audio_seconds"))),
                (
                    "SAMPLE RATE",
                    f"{int(result['sample_rate']):,} Hz" if result.get("sample_rate") else None,
                ),
                (
                    "SEMANTIC TOKENS",
                    f"{int(semantic['output_tokens']):,}"
                    if semantic.get("output_tokens")
                    else None,
                ),
                (
                    "SEMANTIC tok/s",
                    f"{float(semantic['output_tps']):.2f}" if semantic.get("output_tps") else None,
                ),
                ("ABC SECONDS", (timing.get("abc") or {}).get("seconds")),
                ("NAR SECONDS", timing.get("nar_seconds")),
                ("VAE SECONDS", timing.get("vae_seconds")),
                ("TOTAL SECONDS", timing.get("e2e_seconds")),
                ("STATUS", result.get("status")),
            ],
        )
    )
    provenance = []
    for key, name in (("mot", "MODEL"), ("vae", "VAE")):
        for filename, meta in (
            ((result.get("weights") or {}).get(key) or {}).get("files") or {}
        ).items():
            info = meta or {}
            provenance.append(
                (
                    f"{name} WEIGHT",
                    f"{filename} · {format_bytes(info.get('bytes'))} · {str(info.get('sha256', ''))[:12]}",
                )
            )
    if config.get("runtime_sha256"):
        provenance.append(("RUNTIME SHA", str(config["runtime_sha256"])[:12]))
    if config.get("validation_status"):
        provenance.append(("VALIDATION", config["validation_status"]))
    out.append(_table("provenance", provenance))

    files = det.get("files") or []
    if files:
        out.append(
            '<div class="bb-lib-section">files</div>'
            '<table class="bb-lib-table bb-lib-filetable"><tbody>'
        )
        out.extend(
            f"<tr><td>{html.escape(entry['name'])}</td>"
            f"<td>{html.escape(format_bytes(entry['bytes']))}</td></tr>"
            for entry in files
        )
        out.append("</tbody></table>")
    out.append("</div>")
    return "".join(out)


def render_empty_html(message: str = "No work selected.") -> str:
    return f'<div class="bb-lib-card"><div class="bb-lib-note">{html.escape(message)}</div></div>'


def render_multi_html(rels) -> str:
    rows = "".join(f"<li>{html.escape(str(r))}</li>" for r in rels)
    return (
        f'<div class="bb-lib-card"><div class="bb-lib-section">{len(rels)} works selected</div>'
        f'<ul class="bb-lib-list-plain">{rows}</ul>'
        '<div class="bb-lib-note">Select a single work to see its details.</div></div>'
    )


def render_confirm_html(items) -> str:
    if not items:
        return ""
    rows = "".join(
        f'<li>{html.escape(i["name"])}<span class="bb-lib-dim"> · {html.escape(i["rel"])}</span></li>'
        for i in items
    )
    return (
        '<div class="bb-lib-confirm">'
        f'<div class="bb-lib-section">delete {len(items)} item(s) — this cannot be undone</div>'
        f'<ul class="bb-lib-list-plain">{rows}</ul></div>'
    )


# ─────────────────────────────────────────────────────────── mutations ──
def _safe_rel(root: Path, rel: str) -> Path | None:
    try:
        path = (Path(root).resolve() / rel).resolve()
    except (OSError, ValueError):
        return None
    if path == Path(root).resolve() or Path(root).resolve() not in path.parents:
        return None
    return path


def sanitize_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"[^\w.\-]+", "-", name, flags=re.UNICODE)
    name = re.sub(r"-{2,}", "-", name).strip("-._")
    return name[:MAX_NAME]


def rename(root, rel: str, new_name: str):
    """Rename one work; the ``YYYYMMDD-HHMMSS-`` prefix is preserved."""
    root = Path(root).resolve()
    source = _safe_rel(root, rel)
    if source is None or not source.is_dir():
        return False, "That work no longer exists.", None
    clean = sanitize_name(new_name)
    if not clean:
        return False, "Enter a name (letters, digits, dashes, dots).", None
    match = STAMP_RE.match(source.name)
    prefix = f"{match.group(1)}-{match.group(2)}-" if match else ""
    target = source.parent / (prefix + clean)
    if target == source:
        return True, "Name unchanged.", source.relative_to(root).as_posix()
    if target.exists():
        return False, f"“{target.name}” already exists.", None
    try:
        source.rename(target)
    except OSError as exc:
        return False, f"Rename failed: {exc}", None
    return True, f"Renamed to “{clean}”.", target.relative_to(root).as_posix()


def delete(root, rels):
    """Delete the given work directories (refuses anything outside the root)."""
    root = Path(root).resolve()
    removed, errors = [], []
    for rel in rels or []:
        path = _safe_rel(root, rel)
        if path is None or path == root or not path.is_dir():
            errors.append(f"{rel}: not a work directory")
            continue
        if not any((path / name).exists() for name in ARTIFACTS + (PENDING,)):
            errors.append(f"{rel}: no known artifacts")
            continue
        try:
            shutil.rmtree(path)
            removed.append(rel)
        except OSError as exc:
            errors.append(f"{rel}: {exc}")
    if removed and errors:
        return False, f"Deleted {len(removed)} item(s); failed: " + "; ".join(errors)
    if errors:
        return False, "Nothing deleted — " + "; ".join(errors)
    return True, f"Deleted {len(removed)} item(s)."
