"""Cover-workflow helpers: SheetSage2 ABC → YuE2 cover request.

Pure functions, no Gradio and no model: the transcription itself lives in
``sheetsage_adapter``, and these helpers turn its ABC into a request that can be
submitted to the existing generation path (``adapter.generate``).

The one rule this module enforces: a cover is *score-conditioned*.  The ABC must
be present after preparation — generation with an empty score would make YuE2
plan a new melody, silently discarding the transcription.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable

from . import adapter
from .vendor import abc_tools

#: transcription task → YuE2 conditioning mode (``cot``)
COT_FOR_TASK = {"melody-vocal": "melody", "melody-full": "melody", "full": "full"}


def cot_for_task(task: str) -> str:
    """YuE2 plan mode for a SheetSage2 task; covers never use ``off``."""
    if task not in COT_FOR_TASK:
        raise ValueError(
            f"Unknown transcription task {task!r}; use one of {', '.join(COT_FOR_TASK)}"
        )
    return COT_FOR_TASK[task]


def clean_abc(text: str) -> str:
    """Drop chat/code-block indentation and surrounding blank lines."""
    return textwrap.dedent(text or "").strip()


def inspect_abc(text: str) -> dict:
    """Validate an ABC score and return the portable structural report."""
    source = clean_abc(text)
    if not source:
        raise ValueError("There is no ABC score to inspect")
    return abc_tools.report(abc_tools.parse_abc(source))


def prepare_cover_abc(text: str, *, keep_voice: str = "both") -> str:
    """Validate a transcription and remove chord symbols for a cover melody.

    ``keep_voice`` follows ``abc_tools.strip_chords``: ``both`` (default) keeps
    vocal and instrumental melodies, ``Vocal`` / ``Ins`` silences the other voice.
    Scores that have no chord symbols are returned unchanged after validation.
    """
    source = clean_abc(text)
    if not source:
        raise ValueError("The transcription has no ABC score to prepare")
    score = abc_tools.parse_abc(source)
    if any(voice.chords for voice in score.voices.values()):
        return abc_tools.strip_chords(source, keep_voice=keep_voice)
    return source


def build_cover_request(
    style: str,
    lyrics: str,
    abc_text: str,
    *,
    task: str = "melody-full",
    seed: int = 831001,
    cfg_scale: float | None = None,
    id: str | None = None,
    keep_voice: str = "both",
    request_factory: Callable | None = None,
):
    """Build the YuE2 request for a cover, always carrying the ABC explicitly.

    Melody-conditioned covers get a chord-free score (``cot="melody"`` does not
    drop chord symbols by itself); ``keep_voice`` is forwarded to the strip step.
    """
    text = clean_abc(abc_text)
    if not text:
        raise ValueError(
            "A cover requires an ABC score; an empty score would let YuE2 plan a "
            "new melody and discard the transcription"
        )
    cot = cot_for_task(task)
    if cot == "melody":
        text = prepare_cover_abc(text, keep_voice=keep_voice)
    kwargs = {"style": style, "lyrics": lyrics, "cot": cot, "seed": int(seed), "abc": text}
    if cfg_scale:
        kwargs["cfg_scale"] = float(cfg_scale)
    if id and id.strip():
        kwargs["id"] = id.strip()
    factory = request_factory or adapter.song_request
    return factory(**kwargs)
