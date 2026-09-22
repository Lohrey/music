"""Look up the original lyrics of a song for covers (LRCLIB first, lyrics.ovh second).

Lyrics are copyrighted; they are fetched on demand for the user's private covers,
shown in the editable lyrics field and never published by this app on its own.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request

log = logging.getLogger("yue2_groove")

_UA = {"User-Agent": "YuE2-Musik/1.0 (private cover tool)"}
_NOISE = re.compile(
    r"\s*[\(\[【][^\)\]】]*(official|video|audio|lyric|lyrics|visualizer|hd|4k|remaster|live|"
    r"musikvideo|clip|mv|explicit|prod\.?)[^\)\]】]*[\)\]】]",
    re.I,
)


def parse_title(raw: str) -> tuple[str, str]:
    """'01 - Andreas Bourani - Auf Uns (Lyric Video).mp3' -> ('Andreas Bourani', 'Auf Uns')."""
    text = re.sub(r"\.(mp3|wav|flac|m4a|ogg|webm|mp4|opus|aac)$", "", raw or "", flags=re.I)
    text = re.sub(r"^[0-9a-f]{20,}-", "", text)  # our Drive cache prefix
    text = re.sub(r"^\d{8,}-", "", text)  # our timestamp prefix
    text = text.replace("_", " ")
    text = _NOISE.sub("", text)
    text = re.sub(r"\s*(ft\.?|feat\.?|featuring)\s.+$", "", text, flags=re.I)
    text = re.sub(r"^\s*\d{1,3}\s*[-.)]\s*", "", text)  # track numbers
    parts = [p.strip(" -–—|") for p in re.split(r"\s+[-–—|]\s+|\s-\s|\s{2,}-+\s*", text) if p.strip(" -–—|")]
    if len(parts) >= 2:
        return parts[0], " ".join(parts[1:])
    return "", text.strip()


def _json(url: str):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8", "replace"))


def _lrclib(artist: str, title: str, duration: float | None) -> str | None:
    q = urllib.parse.urlencode({"q": f"{artist} {title}".strip()})
    try:
        hits = _json(f"https://lrclib.net/api/search?{q}")
    except Exception as exc:  # noqa: BLE001
        log.info("lyrics: lrclib failed: %s", exc)
        return None
    best, best_score = None, -1e9
    for h in hits or []:
        text = h.get("plainLyrics") or ""
        if not text or h.get("instrumental"):
            continue
        score = 0.0
        name = (h.get("trackName") or "").lower()
        if title and title.lower() in name:
            score += 5
        if artist and artist.lower() in (h.get("artistName") or "").lower():
            score += 5
        if duration and h.get("duration"):
            diff = abs(float(h["duration"]) - duration)
            score += 6 if diff < 5 else (2 if diff < 15 else -6)
        if score > best_score:
            best, best_score = text, score
    return best if best_score > 0 else None


def _ovh(artist: str, title: str) -> str | None:
    if not artist or not title:
        return None
    url = "https://api.lyrics.ovh/v1/{}/{}".format(
        urllib.parse.quote(artist), urllib.parse.quote(title)
    )
    try:
        text = (_json(url) or {}).get("lyrics") or ""
    except Exception as exc:  # noqa: BLE001
        log.info("lyrics: lyrics.ovh failed: %s", exc)
        return None
    text = re.sub(r"^Paroles de la chanson.*\n", "", text)
    return text.strip() or None


def tag_sections(text: str) -> str:
    """Plain lyrics -> YuE2 / ACE-Step structure: repeated stanzas become [chorus]."""
    text = re.sub(r"\r\n?", "\n", text or "").strip()
    text = re.sub(r"^\s*[\[\(]?(chorus|refrain|hook|verse|strophe|bridge|pre-chorus|outro|intro)"
                  r"[^\n]*[\]\)]?\s*:?\s*$", "", text, flags=re.I | re.M)
    stanzas = [s.strip() for s in re.split(r"\n\s*\n+", text) if s.strip()]
    if len(stanzas) <= 1:  # no blank lines: cut every 4 lines
        lines = [ln for ln in text.split("\n") if ln.strip()]
        stanzas = ["\n".join(lines[i:i + 4]) for i in range(0, len(lines), 4)]
    norm = [re.sub(r"\W+", " ", s.lower()).strip() for s in stanzas]
    repeated = {n for n in norm if norm.count(n) > 1}
    out = []
    for i, (s, n) in enumerate(zip(stanzas, norm)):
        if n in repeated:
            tag = "[chorus]"
        elif i == len(stanzas) - 1 and len(stanzas) > 3:
            tag = "[outro]"
        else:
            tag = "[verse]"
        out.append(f"{tag}\n{s}")
    return "\n\n".join(out)


def find(raw_title: str, duration: float | None = None) -> tuple[str | None, str]:
    """(tagged lyrics or None, 'Artist – Title' that was searched)."""
    artist, title = parse_title(raw_title)
    label = f"{artist} – {title}" if artist else title
    if not title:
        return None, label
    text = _lrclib(artist, title, duration) or _ovh(artist, title)
    if not text and not artist:
        text = _lrclib("", title, duration)
    return (tag_sections(text) if text else None), label
