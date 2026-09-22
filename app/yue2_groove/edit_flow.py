"""Edit-iteration helpers: freeze a baseline, check invariants, build manifests.

The workflow these functions support is symbolic-first:

1. **freeze** a generated work (a copy of its small text artifacts plus hashes;
   the original run directory is never touched and stays the listening baseline);
2. **edit** the ABC (and style/lyrics) in the UI;
3. **check** exact sounding-note / meter invariants with the vendored portable
   checker, explicitly permitting a tempo change when that is intended;
4. **generate** again from the edited score — the ABC is always submitted, so an
   edit can never silently degrade into a fresh plan;
5. **compare** baseline and new render with ``vendor/listen.py``.

Everything here is pure Python; the Gradio wiring lives in ``webui.py``.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from . import adapter, library
from .vendor import abc_tools

BASELINE_DIR = "baselines"
#: text/sidecar files recorded in a baseline snapshot (audio/latents are not copied)
BASELINE_FILES = ("score.abc", "request.json", "plan.json", "result.json", "edit_manifest.json")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def snapshot_hashes(directory) -> dict[str, str]:
    """SHA-256 of the artifacts that identify a work (missing files are absent)."""
    directory = Path(directory)
    hashes = {}
    for name in BASELINE_FILES + ("audio.flac", "latent.npy", "semantic.npy"):
        digest = sha256_file(directory / name)
        if digest:
            hashes[name] = digest
    return hashes


def clean_abc(text: str) -> str:
    return textwrap.dedent(text or "").strip()


def validate_edited_abc(text: str) -> str:
    """Return the cleaned ABC or raise ``ValueError``; an empty edit is refused."""
    source = clean_abc(text)
    if not source:
        raise ValueError(
            "The edited ABC is empty; generation would discard the edit and plan a new score"
        )
    abc_tools.parse_abc(source)
    return source


CONTRACTS = ("exact", "pitch", "free")


def _compare_pitch(before, after, names) -> dict:
    """Same ordered pitch sequence per voice; rhythm, meter and tempo are not gated."""
    differences, notes = [], []
    for name in names:
        a, b = before.voices[name], after.voices[name]
        pitches_a = [pitch for _onset, pitch, _duration in a.notes]
        pitches_b = [pitch for _onset, pitch, _duration in b.notes]
        if pitches_a != pitches_b:
            common = min(len(pitches_a), len(pitches_b))
            first = next((i for i in range(common) if pitches_a[i] != pitches_b[i]), common)
            differences.append(f"{name}: pitch sequence differs starting at note {first + 1}")
            continue
        rhythm_a = [(onset, duration) for onset, _pitch, duration in a.notes]
        rhythm_b = [(onset, duration) for onset, _pitch, duration in b.notes]
        if rhythm_a != rhythm_b:
            notes.append(f"{name}: same pitches, rhythm changed (permitted by the pitch contract)")
    return {
        "match": not differences,
        "compared_voices": list(names),
        "differences": differences,
        "notes": notes,
        "scope": "ordered pitch sequence per voice; rhythm, meter and tempo are not gated",
    }


def _compare_exact_without_meter(before, after, names, allow_tempo_change: bool) -> dict:
    """Exact notes/tempo comparison with the bar-time-grid check reported, not gated.

    Local implementation instead of filtering the vendored checker's message text,
    so ``ALLOW METER CHANGE`` cannot silently break when upstream rewords a string.
    """
    differences, allowed = [], []
    if before.bpm != after.bpm and not allow_tempo_change:
        differences.append("quarter-note tempo differs")
    for name in names:
        a, b = before.voices[name], after.voices[name]
        if a.notes != b.notes:
            common = min(len(a.notes), len(b.notes))
            first = next((i for i in range(common) if a.notes[i] != b.notes[i]), common)
            differences.append(
                f"{name}: sounding notes differ starting at note {first + 1} "
                f"(pitch, onset or duration)"
            )
        if a.bars != b.bars:
            allowed.append(f"{name}: meter/time grid differs (permitted by ALLOW METER CHANGE)")
    return {
        "match": not differences,
        "compared_voices": list(names),
        "differences": differences,
        "allowed_differences": allowed,
        "scope": "sounding notes and meter-reported; bar-grid differences permitted",
    }


def check_invariants(
    before_abc: str,
    after_abc: str,
    *,
    voices: str = "both",
    allow_tempo_change: bool = False,
    allow_meter_change: bool = False,
    contract: str = "exact",
) -> dict:
    """Invariant check between the baseline and the edit under an explicit contract.

    ``exact`` (notes + meter grid, tempo optional), ``pitch`` (ordered pitch
    sequence only) and ``free`` (differences are reported, nothing is gated).
    """
    before_text, after_text = clean_abc(before_abc), clean_abc(after_abc)
    if not before_text or not after_text:
        raise ValueError("Both the baseline ABC and the edited ABC are required for a check")
    if contract not in CONTRACTS:
        raise ValueError(f"contract must be one of {', '.join(CONTRACTS)}")
    names = abc_tools.VOICES if voices == "both" else (voices,)
    before = abc_tools.parse_abc(before_text)
    after = abc_tools.parse_abc(after_text)
    if contract == "free":
        # report, do not gate: the same comparison as exact, with the verdict forced true
        reported = abc_tools.compare(before, after, names=names, allow_tempo_change=True)
        result = {
            "match": True,
            "compared_voices": list(names),
            "differences": reported["differences"],
            "scope": "free adaptation: differences are listed for the record, nothing is gated",
        }
    elif contract == "pitch":
        result = _compare_pitch(before, after, names)
    elif allow_meter_change:
        result = _compare_exact_without_meter(before, after, names, bool(allow_tempo_change))
    else:
        result = abc_tools.compare(
            before, after, names=names, allow_tempo_change=bool(allow_tempo_change)
        )
    result.setdefault("tempo_change_allowed", bool(allow_tempo_change))
    result["meter_change_allowed"] = bool(allow_meter_change)
    result["contract"] = contract
    result["before_sha256"] = sha256_text(before_text)
    result["after_sha256"] = sha256_text(after_text)
    result["checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return result


def build_edit_request(
    style: str,
    lyrics: str,
    abc_text: str,
    *,
    cot: str = "full",
    seed: int = 831001,
    cfg_scale: float | None = None,
    id: str | None = None,
    request_factory: Callable | None = None,
):
    """YuE2 request for an edited score; ``abc`` is always present."""
    if cot not in ("full", "melody"):
        raise ValueError("An edited score requires cot=full (melody+harmony) or cot=melody")
    text = validate_edited_abc(abc_text)
    kwargs = {"style": style, "lyrics": lyrics, "cot": cot, "seed": int(seed), "abc": text}
    if cfg_scale:
        kwargs["cfg_scale"] = float(cfg_scale)
    if id and id.strip():
        kwargs["id"] = id.strip()
    factory = request_factory or adapter.song_request
    return factory(**kwargs)


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Could not find a free directory next to {path}")


def freeze_baseline(root, rel: str, *, baseline_root=None, now=None) -> dict:
    """Record an immutable reference to one work.

    Writes ``<baseline_root>/<stamp>-<name>-baseline/`` containing ``baseline.json``
    plus copies of the source's ``score.abc`` and ``request.json`` (small files only;
    audio and latents stay in the original run directory).  The source directory is
    never modified.  Returns the baseline record.
    """
    root = Path(root).resolve()
    source = (root / rel).resolve()
    if source == root or root not in source.parents or not source.is_dir():
        raise ValueError(f"Not a work directory under {root}: {rel}")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    name = library.display_name(source.name)[:60] or "work"
    baselines = Path(baseline_root).expanduser() if baseline_root else root / BASELINE_DIR
    destination = _unique_dir(baselines / f"{stamp}-{name}-baseline")
    destination.mkdir(parents=True, exist_ok=False)

    record = {
        "schema": "yue2-groove-baseline-v1",
        "created": datetime.fromtimestamp(now if now is not None else time.time())
        .astimezone()
        .isoformat(timespec="seconds"),
        "source": {"rel": rel, "path": str(source)},
        "baseline": {
            "rel": destination.relative_to(root).as_posix()
            if destination.is_relative_to(root)
            else str(destination),
            "path": str(destination),
        },
        "hashes": snapshot_hashes(source),
    }
    (destination / "baseline.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for copied in ("score.abc", "request.json"):
        if (source / copied).is_file():
            (destination / copied).write_text(
                (source / copied).read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
            )
    record["abc"] = (
        (source / "score.abc").read_text(encoding="utf-8")
        if (source / "score.abc").is_file()
        else ""
    )
    return record


def build_edit_manifest(
    *,
    source_rel: str,
    before_abc: str,
    after_abc: str,
    cot: str,
    seed: int,
    cfg_scale,
    invariants: dict | None,
    voices: str,
    allow_tempo_change: bool,
    allow_changes: bool,
    allow_meter_change: bool = False,
    contract: str = "exact",
    baseline: dict | None = None,
    now=None,
) -> dict:
    """The ``edit_manifest.json`` written next to a regenerated song."""
    before_text, after_text = clean_abc(before_abc), clean_abc(after_abc)
    return {
        "schema": "yue2-groove-edit-v1",
        "created": datetime.fromtimestamp(now if now is not None else time.time())
        .astimezone()
        .isoformat(timespec="seconds"),
        "source": {"rel": source_rel, "frozen": baseline is not None, "baseline": baseline or None},
        "abc": {
            "before_sha256": sha256_text(before_text) if before_text else None,
            "after_sha256": sha256_text(after_text),
        },
        "request": {"cot": cot, "seed": int(seed), "cfg_scale": cfg_scale},
        "invariants": invariants or None,
        "permitted": {
            "compared_voices": ["Vocal", "Ins"] if voices == "both" else [voices],
            "contract": contract,
            "tempo_change": bool(allow_tempo_change),
            "meter_change": bool(allow_meter_change),
            "changes_override": bool(allow_changes),
        },
    }
