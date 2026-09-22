"""EINFACH — a minimal, Suno-style front page for YUE2 // GROOVE.

One screen, two big tabs (NEUER SONG / COVER), one big button each, and a list of
"Meine Songs" you click to play.  Every knob of the Studio keeps its default; the
full Studio still runs next door (port + 1) for power users.

It reuses the Studio's kernel only (runtime.run_generation, cover.build_cover_request,
sheetsage_adapter.transcribe, library.scan) — no generation logic lives here.
"""

from __future__ import annotations

import colorsys
import functools
import hashlib
import html
import json
import logging
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import gradio as gr

from .. import adapter, config, cover, library, sheetsage_adapter
from . import ace_engine, edits, lyrics as lyrics_db, runtime, youtube

log = logging.getLogger("yue2_groove")

# ─────────────────────────────── defaults ───────────────────────────────

LENGTHS = {"Kurz": 3000, "Normal": 6000, "Lang": 9000}  # semantic ceiling (25 tok/s)

INSTRUMENTAL_LYRICS = (
    "[intro]\n\n\n\n[verse]\n\n\n\n[chorus]\n\n\n\n[verse]\n\n\n\n[chorus]\n\n\n\n[outro]\n\n\n"
)

TAGS = [
    # Genre
    "Pop", "Rock", "Schlager", "Hip-Hop", "Rap", "R&B", "Soul", "Funk", "Elektro", "House",
    "Techno", "Dance", "Jazz", "Blues", "Klassik", "Orchester", "Filmmusik", "Country", "Folk",
    "Reggae", "Latin", "Metal", "Punk", "Indie", "Lo-Fi", "Ambient", "Kinderlied", "Weihnachten",
    # Instrumente
    "Akustik", "Klavier", "Gitarre", "E-Gitarre", "Geige", "Saxophon", "Synthesizer",
    # Stimme & Sprache
    "Deutsch", "Englisch",
    # Stimmung & Tempo
    "Fröhlich", "Traurig", "Romantisch", "Ruhig", "Energiegeladen", "Episch", "Party",
    "Langsam", "Schnell",
]
MAX_VERSIONS = 4
VOICES = [("Egal", ""), ("👩 Frau", "Frauenstimme"), ("👨 Mann", "Männerstimme"),
          ("👫 Duett", "Duett"), ("👥 Chor", "Chor")]


def _with_voice(tags, voice) -> list:
    tags = [t for t in (tags or []) if t not in ("Frauenstimme", "Männerstimme", "Duett", "Chor")]
    return tags + [voice] if voice else tags

# what the model gets for each German tag (YuE2 styles are English)
TAG_PROMPT = {
    "Elektro": "electronic", "Klassik": "classical", "Orchester": "orchestral",
    "Filmmusik": "cinematic film score", "Kinderlied": "children's song", "Weihnachten": "Christmas",
    "Akustik": "acoustic", "Klavier": "piano", "Gitarre": "acoustic guitar",
    "E-Gitarre": "electric guitar", "Geige": "violin", "Saxophon": "saxophone",
    "Frauenstimme": "female vocal", "Männerstimme": "male vocal", "Chor": "choir",
    "Duett": "male and female duet vocals",
    "Deutsch": "German", "Englisch": "English", "Fröhlich": "happy, uplifting",
    "Traurig": "sad, melancholic", "Romantisch": "romantic", "Ruhig": "calm, mellow",
    "Energiegeladen": "energetic", "Episch": "epic", "Langsam": "slow tempo", "Schnell": "fast tempo",
    "Kinder": "children",
}


def _style_with(text, tags) -> str:
    parts = [(text or "").strip().strip(",")] + [TAG_PROMPT.get(t, t) for t in (tags or []) if t]
    return ", ".join(p for p in parts if p)


_STAGES = (
    ("Generating abc", "🎼 Melodie wird komponiert"),
    ("Generating semantic", "🎤 Gesang & Musik entstehen"),
    ("Synthesizing", "🎚️ Klang wird erzeugt"),
    ("Decoding", "🔊 Audio wird fertiggestellt"),
    ("Saving", "💾 Wird gespeichert"),
    ("Starting", "⏳ Wird vorbereitet"),
    ("Loading", "⏳ KI-Modell wird geladen"),
)


class _Friendly:
    """Wraps gr.Progress: German stage names, 'Version x von n', overall percentage."""

    def __init__(self, progress, index: int = 1, total: int = 1, start: float = 0.0):
        self.progress, self.index, self.total, self.start = progress, index, total, start

    def __call__(self, frac, desc=None, **_kw):
        text = str(desc or "")
        for key, label in _STAGES:
            if text.startswith(key):
                text = label
                break
        span = (1.0 - self.start) / self.total
        overall = self.start + span * (self.index - 1) + span * max(0.0, min(1.0, float(frac or 0)))
        if self.total > 1:
            text = f"Version {self.index} von {self.total} · {text}"
        self.progress(min(0.999, overall), desc=text)

_SETTINGS: runtime.RuntimeSettings | None = None


def _settings() -> runtime.RuntimeSettings:
    assert _SETTINGS is not None, "build_simple_ui() sets the runtime settings"
    return _SETTINGS


def _sampling(length: str):
    a, s = runtime.ABC_DEFAULTS, runtime.SEM_DEFAULTS
    sem_max = LENGTHS.get(length, LENGTHS["Normal"])
    abc = runtime.sampling(
        a["temperature"],
        a["top_p"],
        a["top_k"],
        a["repetition_penalty"],
        a["penalty_window"],
        a["min_tokens"],
        a["max_tokens"],
        "ABC",
    )
    sem = runtime.sampling(
        s["temperature"],
        s["top_p"],
        s["top_k"],
        s["repetition_penalty"],
        s["penalty_window"],
        s["min_tokens"],
        sem_max,
        "semantic",
    )
    return abc, sem


# ─────────────────────── optional: AI lyrics (Claude) ───────────────────────


def _auto_lyrics(description: str) -> str | None:
    """Write lyrics with the Claude API when ANTHROPIC_API_KEY is set (app/.env)."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("YUE2_LYRICS_MODEL", "claude-sonnet-4-5")
    prompt = (
        "Schreibe einen kurzen, singbaren Songtext zu dieser Beschreibung: "
        f"«{description}». Nutze genau diese Abschnitts-Markierungen in eckigen Klammern: "
        "[verse], [chorus], [verse], [chorus], [outro]. Pro Abschnitt 4 kurze Zeilen. "
        "Sprache: die Sprache der Beschreibung, außer sie nennt eine andere. "
        "Antworte NUR mit dem Songtext, ohne Titel und ohne Erklärungen."
    )
    body = json.dumps(
        {"model": model, "max_tokens": 800, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return text.strip() or None
    except Exception as exc:  # noqa: BLE001
        log.warning("auto lyrics failed: %s", exc)
        return None


# ─────────────────────────────── songs list ───────────────────────────────


def _songs() -> list[dict]:
    items = [
        i
        for i in library.scan(runtime.RUNS)
        if i["kind"] == "song" and i["has_audio"] and not i.get("pending")
    ]
    return library.sort_items(items, "time_desc")


def _title(item_or_name) -> str:
    name = item_or_name["name"] if isinstance(item_or_name, dict) else item_or_name
    if name.startswith("cover-"):
        name = name[len("cover-") :]
    name = " ".join(name.replace("-", " ").replace("_", " ").split()) or "Song"
    name = re.sub(r"\s+v([1-9])$", r" · Version \1", name)
    return name.replace(" female vocal", "").replace(" male vocal", "").replace(" instrumental", "")


def _style_of(path) -> str:
    request = library._request_of(Path(path)) or {}
    return str(request.get("style") or request.get("tags") or "")


def _art(seed_text: str) -> str:
    h = int(hashlib.md5(seed_text.encode()).hexdigest()[:6], 16) / 0xFFFFFF  # noqa: S324
    r1, g1, b1 = colorsys.hls_to_rgb(h, 0.55, 0.75)
    r2, g2, b2 = colorsys.hls_to_rgb((h + 0.15) % 1, 0.35, 0.8)
    c1 = f"rgb({int(r1 * 255)},{int(g1 * 255)},{int(b1 * 255)})"
    c2 = f"rgb({int(r2 * 255)},{int(g2 * 255)},{int(b2 * 255)})"
    return f"background:linear-gradient(135deg,{c1},{c2})"


# ─────────────────────── playlists & published songs ───────────────────────


def _store_path() -> Path:
    return runtime.RUNS / "simple_store.json"


def _store() -> dict:
    try:
        data = json.loads(_store_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("playlists", {})  # name -> [rel, ...]
    data.setdefault("published", {})  # token -> rel
    return data


def _save_store(data: dict) -> None:
    path = _store_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _token_for(rel: str, create: bool = False) -> str | None:
    data = _store()
    for token, r in data["published"].items():
        if r == rel:
            return token
    if not create:
        return None
    token = hashlib.sha1(f"{rel}{time.time()}".encode()).hexdigest()[:10]  # noqa: S324
    data["published"][token] = rel
    _save_store(data)
    return token


def _rel_moved(old: str, new: str | None) -> None:
    """Keep playlists / published links in sync after a rename (new) or delete (None)."""
    data = _store()
    for name, rels in data["playlists"].items():
        data["playlists"][name] = [new if r == old else r for r in rels if r != old or new]
    for token, r in list(data["published"].items()):
        if r == old:
            if new:
                data["published"][token] = new
            else:
                del data["published"][token]
    _save_store(data)


def songs_html(active: str = "") -> str:
    songs = _songs()
    if not songs:
        return (
            '<div class="yy-empty">Noch keine Songs.<br>'
            "Schreib oben, was du hören willst, und drück <b>Song erstellen</b>.</div>"
        )
    store = _store()
    lists_of: dict[str, list[str]] = {}
    for name, rels in store["playlists"].items():
        for r in rels:
            lists_of.setdefault(r, []).append(name)
    published = set(store["published"].values())
    names = sorted(store["playlists"])
    rows = []
    if names:
        opts = "".join(f'<option value="{html.escape(n)}">❤ {html.escape(n)}</option>' for n in names)
        rows.append(
            '<div class="yy-plbar"><select class="yy-pl-filter" aria-label="Playlist">'
            f'<option value="">Alle Songs</option>{opts}</select></div>'
        )
    for item in songs[:300]:
        title = html.escape(_title(item))
        style = html.escape(_style_of(item["path"])[:90])
        dur = library.format_seconds(item.get("duration"))
        is_cover = item["name"].startswith("cover-")
        badge = '<span class="yy-badge">Cover</span>' if is_cover else ""
        cls = "yy-song yy-active" if item["rel"] == active else "yy-song"
        rel = html.escape(item["rel"])
        pls = html.escape("|".join(lists_of.get(item["rel"], [])))
        is_pub = item["rel"] in published
        if is_pub:
            badge += '<span class="yy-badge yy-pub">🌐</span>'
        rows.append(
            f'<div class="{cls}" data-rel="{rel}" data-pl="{pls}" title="Abspielen">'
            f'<div class="yy-art" style="{_art(item["folder"])}"><span>▶</span></div>'
            f'<div class="yy-meta"><div class="yy-title">{title} {badge}</div>'
            f'<div class="yy-style">{style}</div></div>'
            f'<div class="yy-dur">{html.escape(str(dur))}</div>'
            f'<button class="yy-more" data-more="{rel}" data-title="{title}" '
            f'data-pub="{1 if is_pub else 0}" data-dur="{float(item.get("duration") or 0):.1f}" '
            'title="Mehr" aria-label="Mehr">⋯</button></div>'
        )
    return '<div class="yy-list">' + "".join(rows) + "</div>"


def cover_choices():
    return [(_title(i), i["rel"]) for i in _songs()]


# ─────────────────────────────── player ───────────────────────────────


def _mp3_for(audio: Path) -> str:
    """An MP3 next to the FLAC for the download button (FLAC if the encoder is missing)."""
    audio = Path(audio)
    if audio.suffix.lower() == ".mp3":
        return str(audio)
    mp3 = audio.with_suffix(".mp3")
    if mp3.is_file():
        return str(mp3)
    try:
        import soundfile as sf

        data, rate = sf.read(str(audio))
        tmp = mp3.with_suffix(".tmp.mp3")
        sf.write(str(tmp), data, rate, format="MP3")
        tmp.replace(mp3)
        return str(mp3)
    except Exception:  # noqa: BLE001
        return str(audio)


def show_song(pick):
    rel = (pick or "").split("|")[0].strip()
    hidden = (
        gr.update(visible=False),
        None,
        "",
        gr.update(value=None),
        "",
        songs_html(),
        gr.update(visible=False),
    )
    if not rel:
        return hidden
    item, det = library.load(runtime.RUNS, rel)
    if item is None or not det or not det.get("audio"):
        return hidden
    request = det.get("request") or {}
    style = html.escape(str(request.get("style") or ""))
    lyrics = str(request.get("lyrics") or "").strip()
    if lyrics == INSTRUMENTAL_LYRICS.strip():
        lyrics = "(ohne Gesang)"
    info = (
        f'<div class="yy-now"><div class="yy-now-art" style="{_art(item["folder"])}"></div>'
        f'<div><div class="yy-now-title">{html.escape(_title(item))}</div>'
        f'<div class="yy-style">{style}</div></div></div>'
    )
    mp3 = _mp3_for(Path(det["audio"]))
    src = "/gradio_api/file=" + urllib.parse.quote(mp3)
    info += (
        f'<div class="yy-mini-data" hidden data-src="{html.escape(src)}" '
        f'data-title="{html.escape(_title(item))}" data-sub="{style}" '
        f'data-art="{html.escape(_art(item["folder"]))}" data-rel="{html.escape(item["rel"])}" '
        f'data-nonce="{time.time()}"></div>'
    )
    if lyrics:
        info += (
            '<details class="yy-lyrics"><summary>Songtext anzeigen</summary>'
            f"<pre>{html.escape(lyrics)}</pre></details>"
        )
    return (
        gr.update(visible=True),
        det["audio"],
        info,
        gr.update(value=mp3),
        rel,
        songs_html(rel),
        gr.update(visible=False),
    )


# ─────────────────────────────── generation ───────────────────────────────


def _busy(n_outputs_after_status: int):
    return ("⏳ Es läuft gerade schon ein Song. Bitte kurz warten.",) + (gr.update(),) * (
        n_outputs_after_status
    )


def _to_mp3_only(outdir) -> None:
    """Keep every new song as MP3 only (320 kbit/s); the FLAC is removed once the MP3 exists."""
    flac = Path(outdir) / "audio.flac"
    if not flac.is_file():
        return
    mp3 = flac.with_suffix(".mp3")
    try:
        import soundfile as sf

        data, rate = sf.read(str(flac))
        tmp = mp3.with_suffix(".tmp.mp3")
        try:
            sf.write(str(tmp), data, rate, format="MP3", compression_level=0.0)
        except TypeError:
            sf.write(str(tmp), data, rate, format="MP3")
        tmp.replace(mp3)
    except Exception:  # noqa: BLE001 — fall back to the bundled ffmpeg
        try:
            import subprocess

            import imageio_ffmpeg

            subprocess.run(  # noqa: S603
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(flac),
                 "-b:a", "320k", str(mp3)],
                check=True, timeout=600,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("mp3 conversion failed, keeping FLAC: %s", exc)
    if mp3.is_file() and mp3.stat().st_size > 0:
        try:
            flac.unlink()
        except OSError:
            pass


def _run(request, outdir, length, progress):
    ace_engine.stop()  # one model on the GPU at a time
    abc_s, sem_s = _sampling(length)
    pipe, note = runtime.get_pipe(_settings(), progress)
    runtime.run_generation(
        pipe,
        request,
        outdir,
        abc_sampling=abc_s,
        semantic_sampling=sem_s,
        progress=progress,
        note=note,
    )
    _to_mp3_only(outdir)
    return outdir


# ─────────────────────────────── ACE-Step engine ───────────────────────────────

ENGINES = [("🌍 ACE-Step – alle Sprachen (auch Deutsch), schnell", "ace"),
           ("🎼 YuE2 – Englisch, lange Songs", "yue")]
ACE_SECONDS = {"Kurz": 90, "Normal": 180, "Lang": 240}
_DE_WORDS = set("ich du und der die das nicht ist mit mein dein wir ihr sie auf für wie noch nur "
                "sich auch ein eine einen kein liebe herz nacht immer schon mal".split())
_EN_WORDS = set("i you and the not is with my your we they on for how only love heart night "
                "always just a an no me it this that baby".split())


def _vocal_language(tags, *texts) -> str:
    tags = tags or []
    if "Deutsch" in tags:
        return "de"
    if "Englisch" in tags:
        return "en"
    words = re.findall(r"[a-zäöüß']+", " ".join(t or "" for t in texts).lower())
    de = sum(w in _DE_WORDS for w in words) + 2 * sum(any(c in w for c in "äöüß") for w in words)
    en = sum(w in _EN_WORDS for w in words)
    if de == en == 0:
        return "unknown"
    return "de" if de >= en else "en"


def _ace_run(payload: dict, out_dirs: list[Path], progress) -> list[dict]:
    runtime.unload_pipeline()  # free YuE2's VRAM for ACE-Step
    return ace_engine.generate(
        payload, out_dirs, progress=lambda f, t: progress(f, desc=t), cancelled=runtime.CANCEL.is_set
    )


def _with_stop(fn):
    """Append the visibility of the ABBRECHEN button: shown only while a job runs."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for out in fn(*args, **kwargs):
            running = isinstance(out[0], str) and "wird erstellt" in out[0]
            yield (*out, gr.update(visible=running))

    return wrapper


def _pick_value(outdir: Path) -> str:
    rel = Path(outdir).resolve().relative_to(runtime.RUNS.resolve()).as_posix()
    return f"{rel}|{time.time()}"


@_with_stop
def create_song(
    style, tags, lyrics, instrumental, length, versions, engine="yue", progress=gr.Progress()
):
    """NEUER SONG. Outputs: status, button, pick."""
    name_hint = _style_with(style, None) or ", ".join(tags or []) or "Song"
    description = _style_with(style, None)
    style = _style_with(style, tags)
    versions = max(1, min(MAX_VERSIONS, int(versions or 1)))
    lyrics = (lyrics or "").strip()
    if not style and not lyrics:
        yield "✋ Bitte schreib zuerst, was für ein Song es werden soll.", gr.update(), gr.update()
        return
    if not runtime.try_start_job():
        yield _busy(2)
        return
    try:
        yield "🎵 Song wird erstellt … (das dauert ein paar Minuten)", gr.update(
            interactive=False
        ), gr.update()
        if engine == "ace":
            lang = _vocal_language(tags, description, lyrics)
            payload = {
                "task_type": "text2music",
                "vocal_language": lang,
                "audio_duration": ACE_SECONDS.get(length, 180),
                "use_random_seed": True,
            }
            if instrumental:
                payload.update(prompt=(style + ", instrumental").strip(", "), lyrics="[Instrumental]")
            elif lyrics:
                payload.update(prompt=style or "Pop", lyrics=lyrics)
            else:  # ACE-Step's own language model writes the lyrics from the description
                payload.update(sample_mode=True, sample_query=(
                    style + (", Songtext auf Deutsch" if lang == "de" else "")) or "Pop")
            dirs = [
                runtime.run_dir(runtime.slug(name_hint) + (f"-v{v}" if versions > 1 else ""))
                for v in range(1, versions + 1)
            ]
            songs = _ace_run(payload, dirs, _Friendly(progress))
            done = "✅ Fertig! Dein Song läuft." if len(songs) == 1 else f"✅ Alle {len(songs)} Versionen fertig!"
            yield done, gr.update(interactive=True), _pick_value(songs[0]["dir"])
            return
        if instrumental:
            lyrics = INSTRUMENTAL_LYRICS
            style = (style + ", instrumental").strip(", ")
        elif not lyrics:
            progress(0.01, desc="Songtext wird geschrieben …")
            lyrics = _auto_lyrics(style) or INSTRUMENTAL_LYRICS
            if lyrics == INSTRUMENTAL_LYRICS:
                style = (style + ", instrumental").strip(", ")
        if not style:
            style = "Pop"
        outdir = None
        for v in range(1, versions + 1):
            request = adapter.song_request(
                style=style, lyrics=lyrics, cot="full", seed=runtime.random_seed()
            )
            outdir = runtime.run_dir(runtime.slug(name_hint) + (f"-v{v}" if versions > 1 else ""))
            _run(request, outdir, length, _Friendly(progress, v, versions))
            if v < versions:
                yield (
                    f"✅ Version {v} fertig – Version {v + 1} von {versions} wird erstellt …",
                    gr.update(),
                    _pick_value(outdir),
                )
        done = "✅ Fertig! Dein Song läuft." if versions == 1 else f"✅ Alle {versions} Versionen fertig!"
        if lyrics == INSTRUMENTAL_LYRICS and not instrumental:
            done += " (Ohne Songtext wird er ohne Gesang – schreib einen Text, dann wird gesungen.)"
        yield done, gr.update(interactive=True), _pick_value(outdir)
    except Exception as exc:  # noqa: BLE001
        yield (
            "❌ Das hat nicht geklappt: " + runtime.failure_text(exc),
            gr.update(interactive=True),
            gr.update(),
        )
    finally:
        runtime.end_job()


@_with_stop
def again(current, progress=gr.Progress()):
    """NOCHMAL: same style + lyrics, new take. Outputs: status, pick."""
    item, det = library.load(runtime.RUNS, current or "")
    if item is None or not det:
        yield "Bitte zuerst einen Song anklicken.", gr.update()
        return
    request = det.get("request") or {}
    if not runtime.try_start_job():
        yield _busy(1)
        return
    try:
        yield "🎵 Neue Version wird erstellt …", gr.update()
        if request.get("engine") == "ace-step":
            payload = {
                "task_type": "text2music",
                "prompt": request.get("style") or "Pop",
                "lyrics": request.get("lyrics") or "[Instrumental]",
                "vocal_language": request.get("vocal_language") or "unknown",
                "audio_duration": (item.get("duration") or 180),
                "use_random_seed": True,
            }
            outdir = runtime.run_dir(runtime.slug(_title(item)))
            _ace_run(payload, [outdir], _Friendly(progress))
            yield "✅ Neue Version fertig!", _pick_value(outdir)
            return
        kwargs = {
            "style": request.get("style") or "Pop",
            "lyrics": request.get("lyrics") or INSTRUMENTAL_LYRICS,
            "cot": request.get("cot") or "full",
            "seed": runtime.random_seed(),
        }
        if request.get("abc"):
            kwargs["abc"] = request["abc"]
        if request.get("cfg_scale"):
            kwargs["cfg_scale"] = request["cfg_scale"]
        new = adapter.song_request(**kwargs)
        outdir = runtime.run_dir(runtime.slug(_title(item)))
        _run(new, outdir, "Lang", _Friendly(progress))
        yield "✅ Neue Version fertig!", _pick_value(outdir)
    except Exception as exc:  # noqa: BLE001
        yield "❌ Das hat nicht geklappt: " + runtime.failure_text(exc), gr.update()
    finally:
        runtime.end_job()


def _mark_original_lyrics(outdir) -> None:
    """Note in request.json that this cover sings the original (copyrighted) lyrics."""
    path = Path(outdir) / "request.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["original_lyrics"] = True
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass


def _audio_seconds(path) -> float | None:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001
        return None


@_with_stop
def create_cover(
    source, upload, yt_url, folder_file, pick_rel, style, tags, lyrics, versions,
    engine="yue", progress=gr.Progress(),
):
    """COVER. Outputs: status, button, pick."""
    style_text = _style_with(style, None)
    style = _style_with(style, tags)
    versions = max(1, min(MAX_VERSIONS, int(versions or 1)))
    lyrics = (lyrics or "").strip()
    from_yt = source == "link"
    from_upload = source == "file" or from_yt
    if from_yt and not youtube.looks_like_link(yt_url):
        yield "✋ Bitte einen Link einfügen (YouTube, Google Drive, Dropbox …).", gr.update(), gr.update()
        return
    if source == "file" and not upload:
        yield "✋ Bitte zuerst eine Musikdatei hochladen oder etwas vorsingen.", gr.update(), gr.update()
        return
    if not from_upload and not pick_rel:
        yield "✋ Bitte einen deiner Songs auswählen.", gr.update(), gr.update()
        return
    if not runtime.try_start_job():
        yield _busy(2)
        return
    try:
        yield "🎤 Cover wird erstellt … (das dauert ein paar Minuten)", gr.update(
            interactive=False
        ), gr.update()
        name = "cover"
        yt_title = ""
        if from_yt and youtube.drive_folder_id(yt_url):
            if not folder_file:
                raise ValueError("Bitte im Ordner zuerst ein Lied auswählen.")
            file_id, _, fname = str(folder_file).partition("|")
            progress(0.01, desc="Datei wird geladen …")
            upload = youtube.download_drive_file(
                file_id, fname, runtime.RUNS / "gdrive", progress=lambda v, t: progress(v, desc=t)
            )
            yt_title = Path(fname).stem
        elif from_yt:
            progress(0.01, desc="Datei wird geladen …")
            upload, yt_title = youtube.download_link(
                yt_url,
                runtime.RUNS / "youtube",
                progress=lambda v, t: progress(v, desc=t),
            )
        note = ""
        if from_upload:
            upload = youtube.as_audio(upload)
            name = yt_title or Path(upload).stem
            if not lyrics:  # original lyrics from LRCLIB / lyrics.ovh
                progress(0.01, desc="Originaltext wird gesucht …")
                found, label = lyrics_db.find(name, _audio_seconds(upload))
                if found:
                    lyrics, note = found, f" 📝 Originaltext von „{label}“ verwendet."
        else:
            item, det = library.load(runtime.RUNS, pick_rel)
            if item is None or not det:
                raise RuntimeError("Der Song existiert nicht mehr.")
            req = det.get("request") or {}
            lyrics = lyrics or str(req.get("lyrics") or "")
            style = style or str(req.get("style") or "")
            name = _title(item)
            upload = det.get("audio")
        if engine == "ace":
            if not upload:
                raise RuntimeError("Keine Audiodatei für das Cover gefunden.")
            lang = _vocal_language(tags, style_text, lyrics)
            payload = {
                "task_type": "cover",
                "src_audio_path": str(Path(upload).resolve()),
                "prompt": style or "Pop",
                "lyrics": lyrics or "[Instrumental]",
                "vocal_language": lang,
                "audio_cover_strength": float(os.environ.get("YUE2_ACE_COVER_STRENGTH", "0.6")),
                "use_random_seed": True,
            }
            dirs = [
                runtime.run_dir("cover", runtime.slug(f"{name} {style_text or style}")
                                + (f"-v{v}" if versions > 1 else ""))
                for v in range(1, versions + 1)
            ]
            songs = _ace_run(payload, dirs, _Friendly(progress))
            if note:
                for sdir in dirs:
                    _mark_original_lyrics(sdir)
            done = "✅ Cover fertig!" if len(songs) == 1 else f"✅ Alle {len(songs)} Cover-Versionen fertig!"
            yield done + note, gr.update(interactive=True), _pick_value(songs[0]["dir"])
            return
        if from_upload:
            progress(0.01, desc="Melodie wird herausgehört …")
            outdir_t = (
                config.transcriptions_dir(runtime.RUNS)
                / f"{time.strftime('%Y%m%d-%H%M%S')}-{runtime.slug(Path(upload).stem)}"
            )
            record = sheetsage_adapter.transcribe(
                upload,
                output_dir=outdir_t,
                task="melody-full",
                device=config.default_sheetsage_device(),
                cancelled=runtime.CANCEL.is_set,
                progress=lambda v, t: progress(min(0.3, float(v) * 0.3), desc=t),
            )
            abc = record.get("abc") or ""
        else:
            abc = det.get("abc") or ""
            if not abc.strip():
                raise RuntimeError(
                    "Dieses Lied hat keine Noten für YuE2 – bitte oben „ACE-Step“ auswählen."
                )
        lyrics = lyrics or INSTRUMENTAL_LYRICS
        if not abc.strip():
            raise RuntimeError("Aus dieser Datei konnte keine Melodie gelesen werden.")
        outdir = None
        start = 0.3 if from_upload else 0.0
        for v in range(1, versions + 1):
            request = cover.build_cover_request(
                style or "Pop",
                lyrics,
                abc,
                task="melody-full",
                seed=runtime.random_seed(),
                request_factory=adapter.song_request,
            )
            suffix = f"-v{v}" if versions > 1 else ""
            outdir = runtime.run_dir("cover", runtime.slug(f"{name} {style}") + suffix)
            _run(request, outdir, "Lang", _Friendly(progress, v, versions, start))
            if v < versions:
                yield (
                    f"✅ Version {v} fertig – Version {v + 1} von {versions} wird erstellt …",
                    gr.update(),
                    _pick_value(outdir),
                )
            if note:
                _mark_original_lyrics(outdir)
        done = "✅ Cover fertig!" if versions == 1 else f"✅ Alle {versions} Cover-Versionen fertig!"
        yield done + note, gr.update(interactive=True), _pick_value(outdir)
    except sheetsage_adapter.SheetsageNotConfigured:
        yield (
            "❌ Für Cover aus Dateien fehlt das Zusatz-Modul (SheetSage2). "
            "Cover von deinen eigenen Songs geht trotzdem.",
            gr.update(interactive=True),
            gr.update(),
        )
    except Exception as exc:  # noqa: BLE001
        yield (
            "❌ Das hat nicht geklappt: " + runtime.failure_text(exc),
            gr.update(interactive=True),
            gr.update(),
        )
    finally:
        runtime.end_job()


# ─────────────────────────────── Suno-style edits ───────────────────────────────

_EDIT_LABEL = {
    "crop": "✂️ Wird gekürzt", "remove": "✂️ Teil wird entfernt", "reverse": "⏪ Wird umgedreht",
    "speed": "⏩ Tempo wird geändert", "fadein": "🔉 Fade-In", "fadeout": "🔉 Fade-Out",
    "extend": "➕ Song wird verlängert", "replace": "🔄 Teil wird neu gemacht",
    "stems": "🎚 Gesang & Musik werden getrennt", "remaster": "✨ Remaster",
}


def _ace_edit_payload(op, params, request, src, duration):
    style = (params.get("style") or "").strip() or str(request.get("style") or "Pop")
    old_lyrics = str(request.get("lyrics") or "").strip()
    if old_lyrics.lower().startswith("[instrumental") or old_lyrics == INSTRUMENTAL_LYRICS.strip():
        old_lyrics = ""
    new_lyrics = (params.get("lyrics") or "").strip()
    lang = request.get("vocal_language") or _vocal_language([], old_lyrics, new_lyrics)
    base = {"prompt": style, "vocal_language": lang, "src_audio_path": str(src), "use_random_seed": True}
    if op == "extend":
        extra = min(240.0, max(10.0, float(params.get("seconds") or 30)))
        lyr = (old_lyrics + "\n\n" + new_lyrics).strip() if new_lyrics else (old_lyrics or "[Instrumental]")
        return {**base, "task_type": "repaint", "lyrics": lyr,
                "repainting_start": max(0.0, duration - 2.0), "repainting_end": duration + extra}
    if op == "replace":
        start = max(0.0, float(params.get("start") or 0))
        end = min(duration, max(start + 3.0, float(params.get("end") or duration)))
        return {**base, "task_type": "repaint", "lyrics": new_lyrics or old_lyrics or "[Instrumental]",
                "repainting_start": start, "repainting_end": end, "repaint_mode": "balanced",
                "repaint_strength": 0.7}
    # remaster: same song, re-rendered close to the original
    return {**base, "task_type": "cover", "lyrics": old_lyrics or "[Instrumental]",
            "audio_cover_strength": 0.85}


@_with_stop
def act_edit(value, progress=gr.Progress()):
    """value = JSON {rel, op, …params, n}. Every edit becomes a new song. Outputs: status, pick."""
    try:
        params = json.loads(value or "{}")
    except ValueError:
        yield gr.update(), gr.update()
        return
    rel, op = params.get("rel") or "", params.get("op") or ""
    item, det = library.load(runtime.RUNS, rel)
    if item is None or not det or not det.get("audio"):
        yield "Bitte zuerst einen Song anklicken.", gr.update()
        return
    src = Path(det["audio"])
    request = det.get("request") or {}
    duration = _audio_seconds(src) or float(item.get("duration") or 0) or 180.0
    title = re.sub(r" · Version \d+$", "", _title(item))[:24].strip()
    if not runtime.try_start_job():
        yield _busy(1)
        return
    try:
        yield f"{_EDIT_LABEL.get(op, '✏️ Bearbeitung')} – wird erstellt …", gr.update()
        if op in ("crop", "remove", "reverse", "speed", "fadein", "fadeout"):
            outdir = runtime.run_dir(runtime.slug(f"{title} {edits.SUFFIX[op]}"))
            outdir.mkdir(parents=True, exist_ok=True)
            edits.quick_edit(op, src, outdir / "audio.mp3", params, duration)
            edits.write_meta(outdir, request, _audio_seconds(outdir / "audio.mp3"), op)
            yield "✅ Fertig – als neuer Song gespeichert.", _pick_value(outdir)
            return
        if op == "stems":
            ace_engine.stop()
            runtime.unload_pipeline()
            work = Path(tempfile.gettempdir()) / f"yue2_stems_{int(time.time())}"
            progress(0.1, desc="🎚 Gesang & Musik werden getrennt")
            voc, inst = edits.stems(src, work)
            first = None
            for wav, label in ((voc, "Gesang"), (inst, "Instrumental")):
                outdir = runtime.run_dir(runtime.slug(f"{title} {label}"))
                outdir.mkdir(parents=True, exist_ok=True)
                edits.to_mp3(wav, outdir / "audio.mp3")
                edits.write_meta(outdir, request, _audio_seconds(outdir / "audio.mp3"), "stem-" + label)
                first = first or outdir
            shutil.rmtree(work, ignore_errors=True)
            yield "✅ Fertig – „Gesang“ und „Instrumental“ sind jetzt in deiner Liste.", _pick_value(first)
            return
        if op in ("extend", "replace", "remaster"):
            if not ace_engine.available():
                if op == "remaster":
                    outdir = runtime.run_dir(runtime.slug(f"{title} {edits.SUFFIX[op]}"))
                    outdir.mkdir(parents=True, exist_ok=True)
                    edits.quick_edit("master", src, outdir / "audio.mp3", params, duration)
                    edits.write_meta(outdir, request, _audio_seconds(outdir / "audio.mp3"), op)
                    yield "✅ Fertig – als neuer Song gespeichert.", _pick_value(outdir)
                    return
                raise RuntimeError("Dafür wird ACE-Step gebraucht (Pinokio → ACE-Step 1.5 → Install).")
            outdir = runtime.run_dir(runtime.slug(f"{title} {edits.SUFFIX[op]}"))
            payload = _ace_edit_payload(op, params, request, src, duration)
            _ace_run(payload, [outdir], _Friendly(progress))
            if op == "remaster":
                mp3 = outdir / "audio.mp3"
                tmp = outdir / "_raw.mp3"
                mp3.replace(tmp)
                try:
                    edits.quick_edit("master", tmp, mp3, {}, duration)
                    tmp.unlink()
                except Exception:  # noqa: BLE001
                    tmp.replace(mp3)
            if request.get("original_lyrics"):
                _mark_original_lyrics(outdir)
            yield "✅ Fertig – als neuer Song gespeichert.", _pick_value(outdir)
            return
        yield "❌ Unbekannte Bearbeitung.", gr.update()
    except Exception as exc:  # noqa: BLE001
        log.warning("edit %s failed: %s", op, exc)
        yield "❌ Das hat nicht geklappt: " + str(exc)[:400], gr.update()
    finally:
        runtime.end_job()


def act_rename(value):
    """value = rel|new name|nonce"""
    rel, _, rest = (value or "").partition("|")
    new_name = rest.rpartition("|")[0].strip()
    if not rel or not new_name:
        return gr.update(), gr.update()
    ok, msg, new_rel = library.rename(runtime.RUNS, rel, new_name)
    if ok and new_rel and new_rel != rel:
        _rel_moved(rel, new_rel)
    return (f"✏️ Umbenannt in „{new_name}“." if ok else "❌ " + msg), songs_html(new_rel or rel)


def act_playlist(value):
    """value = rel|playlist name|nonce ; an existing entry is removed (toggle)."""
    rel, _, rest = (value or "").partition("|")
    name = rest.rpartition("|")[0].strip()[:40]
    if not rel or not name:
        return gr.update(), gr.update()
    data = _store()
    rels = data["playlists"].setdefault(name, [])
    if rel in rels:
        rels.remove(rel)
        msg = f"➖ Aus „{name}“ entfernt."
        if not rels:
            del data["playlists"][name]
    else:
        rels.append(rel)
        msg = f"❤ Zu „{name}“ hinzugefügt."
    _save_store(data)
    return msg, songs_html(rel)


def act_publish(value):
    """value = rel|nonce — toggles the public page."""
    rel = (value or "").split("|")[0]
    if not rel:
        return gr.update(), gr.update()
    token = _token_for(rel)
    if token:
        data = _store()
        data["published"].pop(token, None)
        _save_store(data)
        return "🔒 Nicht mehr veröffentlicht.", songs_html(rel)
    _token_for(rel, create=True)
    _item, det = library.load(runtime.RUNS, rel)
    warn = ""
    if det and (det.get("request") or {}).get("original_lyrics"):
        warn = " ⚠️ Achtung: Dieses Cover singt den Originaltext – öffentlich teilen kann Urheberrechte verletzen."
    return "🌐 Veröffentlicht – über ⋯ → Teilen bekommst du den Link." + warn, songs_html(rel)


def act_reuse(value):
    rel = (value or "").split("|")[0]
    item, det = library.load(runtime.RUNS, rel)
    if item is None or not det:
        return gr.update(), gr.update(), gr.update(), gr.update()
    req = det.get("request") or {}
    lyrics = str(req.get("lyrics") or "")
    if lyrics.strip() == INSTRUMENTAL_LYRICS.strip():
        lyrics = ""
    return gr.update(selected="new"), str(req.get("style") or ""), lyrics, gr.update(value=[])


def go_cover(current):
    return (
        gr.update(selected="cover"),
        gr.update(value="library"),
        gr.update(choices=cover_choices(), value=current or None, visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def delete_song(current):
    ok, msg = library.delete(runtime.RUNS, [current] if current else [])
    if ok and current:
        _rel_moved(current, None)
    return (
        "🗑 Gelöscht." if ok else msg,
        gr.update(visible=False),
        None,
        "",
        songs_html(),
        gr.update(visible=False),
    )


def stop():
    runtime.cancel_run()
    return "⏹ Wird gestoppt …"


def _source_toggle(source):
    return (
        gr.update(visible=source == "file"),
        gr.update(visible=source == "link"),
        gr.update(visible=source == "library", choices=cover_choices()),
    )


# ─────────────────────── public song page & download route ───────────────────────

_PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<meta property="og:title" content="{title}"><meta property="og:description" content="{sub}">
<meta property="og:type" content="music.song"><meta property="og:audio" content="{audio}">
<style>body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#0e0e12;color:#f3f1ee;font-family:Inter,system-ui,sans-serif}}
.c{{width:min(420px,92vw);background:#17171d;border:1px solid #26262e;border-radius:22px;padding:22px;text-align:center}}
.a{{width:100%;aspect-ratio:1;border-radius:16px;{art};margin-bottom:18px}}
h1{{font-size:24px;margin:0 0 4px}}p{{color:#9a98a3;margin:0 0 16px}}audio{{width:100%;margin-bottom:14px}}
a.b{{display:inline-block;margin:4px;padding:12px 18px;border-radius:999px;text-decoration:none;font-weight:700}}
.p{{background:linear-gradient(90deg,#ff5c8a,#ff9a3c);color:#fff}}.s{{background:#26262e;color:#f3f1ee}}</style></head>
<body><div class="c"><div class="a"></div><h1>{title}</h1><p>{sub}</p>
<audio controls preload="metadata" src="{audio}"></audio><br>
<a class="b s" href="{audio}?download=1">⬇ Herunterladen</a><a class="b p" href="/">🎵 Eigene Musik machen</a>
</div></body></html>"""


def register_routes(app) -> None:
    """Add /lied/<token> (public page), /lied/<token>.mp3 and /yy/dl?rel=… to the FastAPI app."""
    from fastapi import Request
    from fastapi.responses import FileResponse, HTMLResponse, Response
    from starlette.routing import Route

    def _file_response(rel: str, download: bool):
        item, det = library.load(runtime.RUNS, rel)
        if item is None or not det or not det.get("audio"):
            return Response("Nicht gefunden", status_code=404)
        mp3 = _mp3_for(Path(det["audio"]))
        name = runtime.slug(_title(item)) + Path(mp3).suffix
        return FileResponse(
            mp3, filename=name, content_disposition_type="attachment" if download else "inline"
        )

    async def page(request: Request):
        token = request.path_params["token"]
        if token.endswith(".mp3"):
            rel = _store()["published"].get(token[:-4])
            if not rel:
                return Response("Nicht gefunden", status_code=404)
            return _file_response(rel, request.query_params.get("download") == "1")
        rel = _store()["published"].get(token)
        item, det = library.load(runtime.RUNS, rel) if rel else (None, None)
        if item is None or not det:
            return HTMLResponse("<h1>Dieses Lied ist nicht (mehr) veröffentlicht.</h1>", status_code=404)
        req = det.get("request") or {}
        return HTMLResponse(
            _PAGE.format(
                title=html.escape(_title(item)),
                sub=html.escape(str(req.get("style") or "")),
                art=_art(item["folder"]),
                audio=f"/lied/{token}.mp3",
            )
        )

    async def dl(request: Request):
        return _file_response(
            request.query_params.get("rel", ""), request.query_params.get("inline") != "1"
        )

    async def publish_api(request: Request):
        rel = request.query_params.get("rel", "")
        item, _det = library.load(runtime.RUNS, rel)
        if item is None:
            return Response("Nicht gefunden", status_code=404)
        token = _token_for(rel, create=True)
        return Response(json.dumps({"url": f"/lied/{token}", "title": _title(item)}),
                        media_type="application/json")

    routes = [
        Route("/lied/{token}", page, methods=["GET"]),
        Route("/yy/dl", dl, methods=["GET"]),
        Route("/yy/publish", publish_api, methods=["GET"]),
    ]
    app.router.routes[0:0] = routes  # before Gradio's own routes


# ─────────────────────────────── page ───────────────────────────────

CSS = """
:root, .dark { --yy-bg:#0e0e12; --yy-card:#17171d; --yy-line:#26262e; --yy-ink:#f3f1ee;
  --yy-dim:#9a98a3; --yy-accent:#ff5c8a; --yy-accent2:#ff9a3c; }
body, gradio-app, .gradio-container { background: var(--yy-bg) !important; color: var(--yy-ink); }
.gradio-container { max-width: 1100px !important; margin: 0 auto !important;
  font-size: 17px; font-family: Inter, system-ui, sans-serif !important; }
footer { display:none !important; }
#yy-head h1 { font-size: 30px; margin: 18px 0 2px; font-weight: 800; letter-spacing:-.5px }
#yy-head h1, #yy-head h1 span { color: var(--yy-ink) !important; }
#yy-head h1 span { color: var(--yy-accent) !important; }
#yy-head p { color: var(--yy-dim); margin: 0 0 8px; }
.yy-box { background: var(--yy-card) !important; border:1px solid var(--yy-line) !important;
  border-radius: 18px !important; padding: 18px !important; }
.yy-box textarea, .yy-box input { font-size: 17px !important; border-radius: 12px !important; }
#yy-create, #yy-cover-btn { background: linear-gradient(90deg,var(--yy-accent),var(--yy-accent2)) !important;
  color:#fff !important; font-size: 20px !important; font-weight: 800 !important; border:0 !important;
  border-radius: 999px !important; min-height: 60px !important; }
#yy-status { font-size: 17px; min-height: 28px; }
.yy-chips { display:flex; flex-wrap:wrap; gap:8px; margin: 2px 0 6px; }
.yy-chip { border:1px solid var(--yy-line); background:#1f1f27; color:var(--yy-ink); border-radius:999px;
  padding:6px 13px; font-size:15px; cursor:pointer; user-select:none; }
.yy-chip:hover { border-color: var(--yy-accent); }
.yy-hide { display:none !important; }
.yy-list { display:flex; flex-direction:column; gap:6px; max-height: 70vh; overflow:auto; }
.yy-song { display:flex; align-items:center; gap:14px; padding:8px; border-radius:14px; cursor:pointer; }
.yy-song:hover, .yy-active { background:#22222b; }
.yy-art { width:56px; height:56px; border-radius:12px; flex:none; display:flex; align-items:center;
  justify-content:center; color:#fff; font-size:20px; }
.yy-meta { flex:1; min-width:0; }
.yy-title { font-weight:700; font-size:17px; }
.yy-style { color:var(--yy-dim); font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.yy-dur { color:var(--yy-dim); font-size:14px; }
.yy-badge { font-size:11px; background:#2b2b35; color:var(--yy-accent2); padding:2px 7px;
  border-radius:999px; vertical-align:middle; }
#yy-head { position: relative; }
.yy-lang { position: absolute; right: 0; top: 22px; display: flex; gap: 2px; background: #1f1f27;
  border: 1px solid var(--yy-line); border-radius: 999px; padding: 3px; }
.yy-lang a { color: var(--yy-dim); text-decoration: none; font-size: 13px; font-weight: 700;
  padding: 4px 10px; border-radius: 999px; }
.yy-lang a.on { background: var(--yy-accent); color: #fff; }
#yy-folder-list .wrap { max-height: 320px; overflow-y: auto; flex-direction: column; flex-wrap: nowrap !important; }
#yy-folder-list label { width: 100%; }
.yy-more { background: transparent; border: 0; color: var(--yy-ink); font-size: 22px; cursor: pointer; line-height: 1;
  padding: 6px 12px; border-radius: 10px; font-weight: 900; letter-spacing: 1px; }
.yy-more:hover { background: #2b2b35; }
.yy-plbar { margin: 0 0 8px; }
.yy-pl-filter { background: #1f1f27; color: var(--yy-ink); border: 1px solid var(--yy-line); border-radius: 999px;
  padding: 8px 14px; font-size: 15px; }
.yy-pub { margin-left: 4px; }
.yy-trash { background: transparent; border: 0; color: var(--yy-dim); font-size: 18px; cursor: pointer;
  padding: 8px 10px; border-radius: 10px; opacity: .55; }
.yy-trash:hover { opacity: 1; background: #2b2b35; color: #ff7a7a; }
/* readable progress card */
#yy-status { min-height: 0; border-radius: 16px; }
#yy-status .wrap.hide { display: none !important; }
#yy-status .wrap:not(.hide) { position: relative !important; inset: auto !important;
  display: flex !important; flex-direction: column-reverse; align-items: stretch; gap: 10px;
  background: var(--yy-card) !important; border: 1px solid var(--yy-line); border-radius: 16px;
  padding: 16px 18px !important; margin-top: 10px; opacity: 1 !important; min-height: 92px; }
#yy-status .wrap .progress-text, #yy-status .wrap .meta-text, #yy-status .wrap .meta-text-center {
  position: static !important; font-size: 17px !important; font-weight: 700; color: var(--yy-ink) !important;
  font-family: Inter, system-ui, sans-serif !important; line-height: 1.4;
  text-align: left; transform: none !important; }
#yy-status .wrap .progress-level { width: 100%; display: flex; flex-direction: column; gap: 8px; }
#yy-status .wrap .progress-level-inner { font-family: Inter, system-ui, sans-serif !important; font-size: 16px !important; color: var(--yy-ink) !important;
  font-weight: 600; text-align: left !important; }
#yy-status .wrap .progress-bar-wrap { width: 100% !important; height: 14px !important; border-radius: 999px;
  background: #26262e !important; border: 0 !important; overflow: hidden; }
#yy-status .wrap .progress-bar { background: linear-gradient(90deg,var(--yy-accent),var(--yy-accent2)) !important;
  border-radius: 999px; }
#yy-status .wrap .eta-bar { display: none !important; }
#yy-status .wrap svg, #yy-status .wrap .loader { display: none !important; }
#yy-status p { font-size: 17px; }
.yy-versions input[type=range] { accent-color: var(--yy-accent); }
.yy-empty { color:var(--yy-dim); text-align:center; padding:40px 10px; line-height:1.7 }
.yy-now { display:flex; gap:14px; align-items:center; }
.yy-now-art { width:72px; height:72px; border-radius:14px; flex:none; }
.yy-now-title { font-size:22px; font-weight:800; }
.yy-lyrics summary { cursor:pointer; color:var(--yy-dim); margin-top:8px; }
.yy-lyrics pre { white-space:pre-wrap; font-family:inherit; font-size:15px; }
.yy-voice .wrap { gap: 6px !important; }
.yy-voice label { padding: 6px 10px !important; }
.yy-actions button { flex-direction:row !important; white-space:nowrap; gap:6px; }
#yy-player .yy-actions + div button, #yy-player button.stop { min-height: 44px; font-size: 16px !important; }
.yy-actions button { border-radius:999px !important; font-size:16px !important; min-height:46px; }
#yy-studio { text-align:center; color:var(--yy-dim); font-size:13px; margin:18px 0; }
#yy-studio a { color:var(--yy-dim); }
.tab-nav button, button[role=tab] { font-size:18px !important; font-weight:700 !important; }
"""

ICON_180 = "data:image/png;base64,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"

HEAD = """
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black">
<meta name="apple-mobile-web-app-title" content="Musik machen">
<meta name="theme-color" content="#0e0e12">
<style>
#yy-menu-bg { position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.35); }
#yy-menu { position: fixed; z-index: 10001; background: #1c1c23; border: 1px solid #2c2c36; color: #f3f1ee;
  font-family: Inter, system-ui, sans-serif; box-shadow: 0 18px 50px rgba(0,0,0,.5); padding: 6px; }
#yy-menu.pop { width: 290px; border-radius: 16px; }
#yy-menu.sheet { left: 0; right: 0; bottom: 0; border-radius: 20px 20px 0 0; padding: 10px 10px calc(14px + env(safe-area-inset-bottom)); }
#yy-menu .yy-menu-title { font-weight: 800; font-size: 15px; padding: 10px 12px 8px; color: #9a98a3;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#yy-menu .yy-menu-item { display: flex; align-items: center; gap: 12px; width: 100%; background: transparent; border: 0;
  color: inherit; font-size: 16px; padding: 12px; border-radius: 12px; cursor: pointer; text-align: left; font-family: inherit; }
#yy-menu .yy-menu-item:hover { background: #2a2a33; }
#yy-menu .yy-menu-item .i { width: 24px; text-align: center; }
#yy-menu .yy-menu-item.danger { color: #ff7a7a; }
#yy-menu.sheet { max-height: 85vh; overflow-y: auto; }
#yy-menu.center { left: 50%; top: 50%; transform: translate(-50%,-50%); width: min(440px, 92vw); max-height: 88vh; overflow-y: auto; border-radius: 18px; padding: 10px 14px 14px; }
#yy-menu.yy-edit .yy-menu-title { white-space: normal; color: #f3f1ee; font-size: 17px; }
.yy-edit-body { padding: 0 8px; }
.yy-edit-body p, .yy-edit-body li { color: #c9c7d1; font-size: 15px; line-height: 1.45; }
.yy-edit-body label { display: block; margin: 14px 0 6px; font-weight: 700; font-size: 15px; }
.yy-edit-body label b { color: #ff9a3c; margin-left: 6px; }
.yy-edit-body input[type=range] { width: 100%; accent-color: #ff5c8a; height: 28px; }
.yy-edit-body textarea, .yy-edit-body input[type=text] { width: 100%; box-sizing: border-box; background: #121217; color: #f3f1ee;
  border: 1px solid #2c2c36; border-radius: 12px; padding: 10px; font-size: 15px; font-family: inherit; }
.yy-edit-body .yy-edit-check { font-weight: 500; display: flex; gap: 8px; align-items: center; }
.yy-edit-listen { margin-top: 12px; background: #2a2a33; color: #f3f1ee; border: 0; border-radius: 999px; padding: 10px 16px; font-size: 15px; cursor: pointer; }
.yy-edit-btns { display: flex; gap: 10px; padding: 16px 8px 4px; }
.yy-edit-btns button { flex: 1; min-height: 48px; border-radius: 999px; border: 0; font-size: 16px; font-weight: 800; cursor: pointer; font-family: inherit; }
.yy-edit-cancel { background: #2a2a33; color: #f3f1ee; }
.yy-edit-ok { background: linear-gradient(90deg,#ff5c8a,#ff9a3c); color: #fff; }
#yy-toast { position: fixed; left: 50%; bottom: 96px; transform: translateX(-50%) translateY(20px); opacity: 0;
  background: #f3f1ee; color: #0e0e12; padding: 10px 18px; border-radius: 999px; font-weight: 700; z-index: 10002;
  transition: all .25s; pointer-events: none; font-family: Inter, system-ui, sans-serif; }
#yy-toast.on { opacity: 1; transform: translateX(-50%) translateY(0); }
#yy-mini { position: fixed; left: 0; right: 0; bottom: 0; z-index: 9999; background: rgba(20,20,26,.97);
  backdrop-filter: blur(10px); border-top: 1px solid #26262e; color: #f3f1ee;
  padding-bottom: env(safe-area-inset-bottom); font-family: Inter, system-ui, sans-serif; }
#yy-mini .yy-mini-prog { position: relative; height: 22px; margin: -9px 0 -9px; cursor: pointer; touch-action: none; }
#yy-mini .yy-mini-track { position: absolute; left: 0; right: 0; top: 9px; height: 4px; background: #2b2b35; }
#yy-mini .yy-mini-prog:hover .yy-mini-track, #yy-mini .yy-mini-prog.drag .yy-mini-track { height: 6px; top: 8px; }
#yy-mini .yy-mini-fill { position: relative; height: 100%; width: 0; background: linear-gradient(90deg,#ff5c8a,#ff9a3c); }
#yy-mini .yy-mini-fill::after { content: ""; position: absolute; right: -7px; top: 50%; width: 14px; height: 14px;
  margin-top: -7px; border-radius: 50%; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.5); }
#yy-mini .yy-mini-nav { width: 40px; height: 40px; border-radius: 50%; border: 0; flex: none; cursor: pointer;
  background: transparent; color: #f3f1ee; font-size: 18px; }
#yy-mini .yy-mini-nav:hover { background: #2a2a33; }
#yy-mini .yy-mini-row { display: flex; align-items: center; gap: 12px; padding: 8px 14px; max-width: 1100px; margin: 0 auto; }
#yy-mini .yy-mini-art { width: 40px; height: 40px; border-radius: 9px; flex: none; }
#yy-mini .yy-mini-meta { flex: 1; min-width: 0; cursor: pointer; }
#yy-mini .yy-mini-title { font-weight: 700; font-size: 15px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#yy-mini .yy-mini-sub { color: #9a98a3; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#yy-mini .yy-mini-time { color: #9a98a3; font-size: 12px; font-variant-numeric: tabular-nums; }
#yy-mini .yy-mini-btn { width: 44px; height: 44px; border-radius: 50%; border: 0; flex: none; cursor: pointer;
  background: #f3f1ee; color: #0e0e12; font-size: 16px; font-weight: 900; }
body.yy-has-mini { padding-bottom: 74px !important; }
@media (max-width: 640px) { #yy-mini .yy-mini-time { display: none; } }
@media (max-width: 640px) {
  .gradio-container, .gradio-container .main, .gradio-container .wrap, .gradio-container main {
    min-width: 0 !important; width: 100% !important; max-width: 100vw !important; box-sizing: border-box; }
  .gradio-container { padding: 0 8px !important; font-size: 16px; }
  .yy-box { padding: 12px !important; }
  #yy-head h1 { font-size: 26px; }
  .yy-art { width: 46px; height: 46px; }
  .yy-dur { display: none; }
  .yy-lang { position: static !important; width: max-content; margin: 12px 0 -8px auto; }
  html, body { overflow-x: hidden; }
}
</style>
<link rel="apple-touch-icon" href="__ICON__">
<link rel="icon" type="image/png" href="__ICON__">
<script>
(function(){
  function setBox(id, value){
    const el = document.querySelector('#'+id+' textarea, #'+id+' input');
    if(!el) return;
    el.value = value;
    el.dispatchEvent(new Event('input', {bubbles:true}));
  }
  document.addEventListener('click', function(e){
    const more = e.target.closest('[data-more]');
    if(more){ e.stopPropagation(); e.preventDefault(); openMenu(more); return; }
    const del = e.target.closest('[data-del]');
    if(del){
      e.stopPropagation(); e.preventDefault();
      setBox('yy-del', del.dataset.del + '|' + Date.now());
      setTimeout(function(){
        const p = document.querySelector('#yy-player'); if(p) p.scrollIntoView({behavior:'smooth', block:'start'});
      }, 400);
      return;
    }
    const song = e.target.closest('[data-rel]');
    if(song){
      setBox('yy-pick', song.dataset.rel + '|' + Date.now());
      return;
    }
    const chip = e.target.closest('[data-chip]');
    if(chip){
      const box = document.querySelector('#'+chip.dataset.target+' textarea');
      if(!box) return;
      const cur = box.value.trim();
      box.value = cur ? (cur.replace(/,\\s*$/,'') + ', ' + chip.dataset.chip) : chip.dataset.chip;
      box.dispatchEvent(new Event('input', {bubbles:true}));
    }
  });
  // ── ⋯ song menu ──
  function L(de, en){ var l='de'; try{ l = localStorage.getItem('yy-lang') || 'de'; }catch(e){} return l==='en' ? en : de; }
  function toast(msg){
    var t = document.getElementById('yy-toast');
    if(!t){ t = document.createElement('div'); t.id = 'yy-toast'; document.body.appendChild(t); }
    t.textContent = msg; t.classList.add('on'); clearTimeout(window.__yyToast);
    window.__yyToast = setTimeout(function(){ t.classList.remove('on'); }, 2600);
  }
  function closeMenu(){ var m = document.getElementById('yy-menu'); if(m) m.remove(); var b = document.getElementById('yy-menu-bg'); if(b) b.remove(); }
  function share(rel, title){
    fetch('/yy/publish?rel=' + encodeURIComponent(rel)).then(function(r){ return r.json(); }).then(function(j){
      var url = location.origin + j.url;
      if(navigator.share){ navigator.share({title: j.title, text: '🎵 ' + j.title, url: url}).catch(function(){}); }
      else if(navigator.clipboard){ navigator.clipboard.writeText(url).then(function(){ toast(L('🔗 Link kopiert','🔗 Link copied')); }); }
      else { prompt(L('Link zum Teilen:','Link to share:'), url); }
    }).catch(function(){ toast(L('Teilen ging nicht','Sharing failed')); });
  }
  function openMenu(btn){
    closeMenu();
    var rel = btn.dataset.more, title = btn.dataset.title, pub = btn.dataset.pub === '1';
    var items = [
      ['▶', L('Abspielen','Play'), function(){ setBox('yy-pick', rel + '|' + Date.now()); }],
      ['🔁', L('Remix (neue Version)','Remix (new version)'), function(){ setBox('yy-act-remix', rel + '|' + Date.now()); toast(L('🔁 Remix startet …','🔁 Remix starting …')); }],
      ['✂️', L('Bearbeiten (Kürzen, Tempo, Stems …)','Edit (crop, speed, stems …)'), function(){ openEdit(btn); }],
      ['🎤', L('Als Cover verwenden','Use as cover'), function(){ setBox('yy-act-cover', rel + '|' + Date.now()); window.scrollTo({top:0, behavior:'smooth'}); }],
      ['📝', L('Text & Stil ändern, neu erstellen','Change lyrics & style, recreate'), function(){ setBox('yy-act-reuse', rel + '|' + Date.now()); window.scrollTo({top:0, behavior:'smooth'}); }],
      ['✏️', L('Umbenennen','Rename'), function(){ var n = prompt(L('Neuer Name:','New name:'), title); if(n && n.trim()) setBox('yy-act-rename', rel + '|' + n.trim() + '|' + Date.now()); }],
      ['❤', L('Zur Playlist hinzufügen / entfernen','Add to / remove from playlist'), function(){
        var names = Array.from(document.querySelectorAll('.yy-pl-filter option')).map(function(o){ return o.value; }).filter(Boolean);
        var last = ''; try{ last = localStorage.getItem('yy-pl-last') || ''; }catch(e){}
        var n = prompt(L('Name der Playlist','Playlist name') + (names.length ? ' (' + names.join(', ') + ')' : '') + ':', last || names[0] || L('Favoriten','Favourites'));
        if(n && n.trim()){ try{ localStorage.setItem('yy-pl-last', n.trim()); }catch(e){} setBox('yy-act-pl', rel + '|' + n.trim() + '|' + Date.now()); } }],
      ['🔗', L('Teilen','Share'), function(){ share(rel, title); }],
      ['🌐', pub ? L('Veröffentlichung beenden','Unpublish') : L('Veröffentlichen','Publish'), function(){ setBox('yy-act-pub', rel + '|' + Date.now()); }],
      ['⬇', L('Herunterladen (MP3)','Download (MP3)'), function(){ window.location.href = '/yy/dl?rel=' + encodeURIComponent(rel); }],
      ['🗑', L('Löschen','Delete'), function(){ setBox('yy-del', rel + '|' + Date.now());
        setTimeout(function(){ var p = document.querySelector('#yy-player'); if(p) p.scrollIntoView({behavior:'smooth', block:'start'}); }, 400); }, 'danger']
    ];
    showSheet(title, items, btn);
  }
  function showSheet(title, items, btn){
    closeMenu();
    var bg = document.createElement('div'); bg.id = 'yy-menu-bg'; bg.addEventListener('click', closeMenu); document.body.appendChild(bg);
    var m = document.createElement('div'); m.id = 'yy-menu';
    var h = document.createElement('div'); h.className = 'yy-menu-title'; h.textContent = title; m.appendChild(h);
    items.forEach(function(it){
      var b = document.createElement('button'); b.className = 'yy-menu-item' + (it[3] ? ' ' + it[3] : '');
      b.innerHTML = '<span class="i"></span><span class="t"></span>';
      b.querySelector('.i').textContent = it[0]; b.querySelector('.t').textContent = it[1];
      b.addEventListener('click', function(e){ e.stopPropagation(); closeMenu(); it[2](); });
      m.appendChild(b);
    });
    document.body.appendChild(m);
    if(window.innerWidth >= 700){
      var r = btn.getBoundingClientRect(), mh = m.offsetHeight;
      var top = r.bottom + 6; if(top + mh > window.innerHeight - 10) top = Math.max(10, r.top - mh - 6);
      m.style.top = top + 'px'; m.style.left = Math.max(10, r.right - m.offsetWidth) + 'px'; m.classList.add('pop');
    } else { m.classList.add('sheet'); }
  }
  // ── Suno-style edit tools ──
  function mmss(t){ t = Math.max(0, Math.round(t)); return Math.floor(t/60) + ':' + ('0' + (t%60)).slice(-2); }
  function openEdit(btn){
    var rel = btn.dataset.more, title = btn.dataset.title, dur = parseFloat(btn.dataset.dur || '0') || 180;
    function go(op, fields, info){ return function(){ editForm(rel, title, dur, op, fields, info); }; }
    var items = [
      ['➕', L('Verlängern','Extend'), go('extend', ['seconds','lyrics','style'], L('Die KI schreibt den Song weiter.','The AI continues the song.'))],
      ['✂️', L('Kürzen (Ausschnitt behalten)','Crop'), go('crop', ['range'], L('Nur der gewählte Teil bleibt.','Only the chosen part stays.'))],
      ['🧽', L('Teil entfernen','Remove section'), go('remove', ['range'], L('Der gewählte Teil wird herausgeschnitten.','The chosen part is cut out.'))],
      ['🔄', L('Teil neu machen','Replace section'), go('replace', ['range','lyrics','style'], L('Die KI singt/spielt den gewählten Teil neu.','The AI redoes the chosen part.'))],
      ['⏪', L('Rückwärts','Reverse'), go('reverse', [], L('Der ganze Song läuft rückwärts.','The whole song plays backwards.'))],
      ['⏩', L('Tempo ändern','Adjust speed'), go('speed', ['factor','pitch'], '')],
      ['🔉', L('Einblenden (Fade In)','Fade in'), go('fadein', ['fade'], '')],
      ['🔈', L('Ausblenden (Fade Out)','Fade out'), go('fadeout', ['fade'], '')],
      ['🎚', L('Gesang & Musik trennen (Stems)','Get stems'), go('stems', [], L('Du bekommst 2 neue Songs: nur Gesang und nur Musik.','You get 2 new songs: vocals only and music only.'))],
      ['✨', L('Remaster (besserer Klang)','Remaster'), go('remaster', [], L('Die KI rendert den Song neu und macht ihn lauter und klarer.','The AI re-renders the song louder and clearer.'))],
      ['🙋', L('Hilfe','Get help'), function(){ editHelp(); }]
    ];
    showSheet('✂️ ' + title, items, btn);
  }
  function editHelp(){
    var rows = [
      [L('Verlängern','Extend'), L('hängt ein neues Stück an – optional mit neuem Text.','adds a new part – optionally with new lyrics.')],
      [L('Kürzen','Crop'), L('behält nur den Teil zwischen Start und Ende.','keeps only the part between start and end.')],
      [L('Teil entfernen','Remove section'), L('schneidet einen Teil heraus.','cuts a part out.')],
      [L('Teil neu machen','Replace section'), L('lässt die KI einen Teil neu erfinden (z.B. anderer Text).','lets the AI redo a part (e.g. new lyrics).')],
      [L('Tempo','Speed'), L('schneller oder langsamer.','faster or slower.')],
      [L('Fade','Fade'), L('sanft ein- oder ausblenden.','fade in or out smoothly.')],
      ['Stems', L('trennt Gesang und Musik (z.B. für Karaoke).','splits vocals and music (e.g. for karaoke).')],
      ['Remaster', L('neu gerendert, lauter und klarer.','re-rendered, louder and clearer.')]
    ];
    var body = '<p>' + L('Jede Bearbeitung wird als <b>neuer Song</b> gespeichert – das Original bleibt.','Every edit is saved as a <b>new song</b> – the original stays.') + '</p><ul>' +
      rows.map(function(r){ return '<li><b>' + r[0] + '</b> – ' + r[1] + '</li>'; }).join('') + '</ul>';
    modal(L('🙋 Hilfe zum Bearbeiten','🙋 Editing help'), body, null);
  }
  function modal(title, bodyHtml, onOk, okLabel){
    closeMenu();
    var bg = document.createElement('div'); bg.id = 'yy-menu-bg'; bg.addEventListener('click', closeMenu); document.body.appendChild(bg);
    var m = document.createElement('div'); m.id = 'yy-menu'; m.className = 'yy-edit ' + (window.innerWidth >= 700 ? 'center' : 'sheet');
    m.innerHTML = '<div class="yy-menu-title"></div><div class="yy-edit-body">' + bodyHtml + '</div>' +
      '<div class="yy-edit-btns"><button class="yy-edit-cancel"></button>' + (onOk ? '<button class="yy-edit-ok"></button>' : '') + '</div>';
    m.querySelector('.yy-menu-title').textContent = title;
    m.querySelector('.yy-edit-cancel').textContent = onOk ? L('Abbrechen','Cancel') : L('OK','OK');
    m.querySelector('.yy-edit-cancel').addEventListener('click', closeMenu);
    if(onOk){ var ok = m.querySelector('.yy-edit-ok'); ok.textContent = okLabel || L('✅ Los','✅ Go');
      ok.addEventListener('click', function(){ var v = onOk(m); if(v !== false) closeMenu(); }); }
    document.body.appendChild(m);
    return m;
  }
  function editForm(rel, title, dur, op, fields, info){
    var h = info ? '<p class="yy-edit-info">' + info + '</p>' : '';
    var a = Math.round(dur * 0.25), b = Math.round(dur * 0.5);
    if(op === 'crop'){ a = 0; b = Math.round(Math.min(dur, 60)); }
    if(fields.indexOf('range') >= 0){
      h += '<label>' + L('Start','Start') + ' <b class="v-a">' + mmss(a) + '</b></label><input type="range" class="f-a" min="0" max="' + Math.floor(dur) + '" step="1" value="' + a + '">' +
           '<label>' + L('Ende','End') + ' <b class="v-b">' + mmss(b) + '</b></label><input type="range" class="f-b" min="0" max="' + Math.floor(dur) + '" step="1" value="' + b + '">' +
           '<button type="button" class="yy-edit-listen">▶ ' + L('Anhören','Preview') + '</button>';
    }
    if(fields.indexOf('seconds') >= 0){
      h += '<label>' + L('Wie viel länger?','How much longer?') + ' <b class="v-s">30 s</b></label><input type="range" class="f-s" min="10" max="120" step="5" value="30">';
    }
    if(fields.indexOf('factor') >= 0){
      h += '<label>' + L('Tempo','Speed') + ' <b class="v-f">1.00×</b></label><input type="range" class="f-f" min="0.5" max="2" step="0.05" value="1.1">' +
           '<label class="yy-edit-check"><input type="checkbox" class="f-p"> ' + L('Tonhöhe mitändern (wie Plattenspieler)','Change pitch too (like a record)') + '</label>';
    }
    if(fields.indexOf('fade') >= 0){
      h += '<label>' + L('Dauer','Length') + ' <b class="v-d">5 s</b></label><input type="range" class="f-d" min="1" max="20" step="1" value="5">';
    }
    if(fields.indexOf('lyrics') >= 0){
      h += '<label>' + L('Neuer Text (optional)','New lyrics (optional)') + '</label><textarea class="f-l" rows="4" placeholder="' + L('leer = alter Text','empty = old lyrics') + '"></textarea>';
    }
    if(fields.indexOf('style') >= 0){
      h += '<label>' + L('Stil (optional)','Style (optional)') + '</label><input type="text" class="f-st" placeholder="' + L('leer = wie bisher','empty = same as before') + '">';
    }
    var names = {extend: L('➕ Verlängern','➕ Extend'), crop: L('✂️ Kürzen','✂️ Crop'), remove: L('🧽 Teil entfernen','🧽 Remove section'),
      replace: L('🔄 Teil neu machen','🔄 Replace section'), reverse: L('⏪ Rückwärts','⏪ Reverse'), speed: L('⏩ Tempo ändern','⏩ Adjust speed'),
      fadein: L('🔉 Einblenden','🔉 Fade in'), fadeout: L('🔈 Ausblenden','🔈 Fade out'), stems: L('🎚 Stems','🎚 Stems'), remaster: '✨ Remaster'};
    var m = modal(names[op] + ' – ' + title, h, function(m){
      var p = {rel: rel, op: op, n: Date.now()};
      var fa = m.querySelector('.f-a'), fb = m.querySelector('.f-b');
      if(fa){ p.start = Math.min(+fa.value, +fb.value); p.end = Math.max(+fa.value, +fb.value);
        if(p.end - p.start < 1){ toast(L('Bitte einen längeren Teil wählen','Please pick a longer part')); return false; } }
      var q;
      if((q = m.querySelector('.f-s'))) p.seconds = +q.value;
      if((q = m.querySelector('.f-f'))) p.factor = +q.value;
      if((q = m.querySelector('.f-p'))) p.pitch = q.checked;
      if((q = m.querySelector('.f-d'))) p.seconds = +q.value;
      if((q = m.querySelector('.f-l'))) p.lyrics = q.value;
      if((q = m.querySelector('.f-st'))) p.style = q.value;
      setBox('yy-act-edit', JSON.stringify(p));
      toast(L('⏳ Bearbeitung startet …','⏳ Edit starting …'));
      var st = document.querySelector('#yy-status'); if(st) st.scrollIntoView({behavior:'smooth', block:'center'});
    });
    function bind(cls, lab, fmt){ var i = m.querySelector(cls), v = m.querySelector(lab); if(!i || !v) return;
      var f = function(){ v.textContent = fmt(+i.value); }; i.addEventListener('input', f); f(); }
    bind('.f-a', '.v-a', mmss); bind('.f-b', '.v-b', mmss);
    bind('.f-s', '.v-s', function(x){ return '+' + x + ' s'; });
    bind('.f-f', '.v-f', function(x){ return x.toFixed(2) + '×'; });
    bind('.f-d', '.v-d', function(x){ return x + ' s'; });
    var lb = m.querySelector('.yy-edit-listen');
    if(lb){ var au = null; lb.addEventListener('click', function(){
      if(au && !au.paused){ au.pause(); lb.textContent = '▶ ' + L('Anhören','Preview'); return; }
      var s0 = Math.min(+m.querySelector('.f-a').value, +m.querySelector('.f-b').value), s1 = Math.max(+m.querySelector('.f-a').value, +m.querySelector('.f-b').value);
      au = au || new Audio('/yy/dl?inline=1&rel=' + encodeURIComponent(rel));
      au.currentTime = s0; au.play(); lb.textContent = '⏸ ' + L('Stopp','Stop');
      au.ontimeupdate = function(){ if(au.currentTime >= s1){ au.pause(); lb.textContent = '▶ ' + L('Anhören','Preview'); } };
    }); }
  }
  function applyFilter(){
    var sel = document.querySelector('.yy-pl-filter'); var v = '';
    try{ v = localStorage.getItem('yy-pl-filter') || ''; }catch(e){}
    if(sel){ if(!Array.from(sel.options).some(function(o){ return o.value === v; })) v = ''; sel.value = v; }
    document.querySelectorAll('.yy-song').forEach(function(c){
      c.style.display = (!v || (c.dataset.pl || '').split('|').indexOf(v) >= 0) ? '' : 'none';
    });
  }
  document.addEventListener('change', function(e){
    if(e.target.classList && e.target.classList.contains('yy-pl-filter')){
      try{ localStorage.setItem('yy-pl-filter', e.target.value); }catch(x){} applyFilter();
    }
  });
  new MutationObserver(function(){ clearTimeout(window.__yyF); window.__yyF = setTimeout(applyFilter, 50); })
    .observe(document.documentElement, {subtree:true, childList:true});
  document.addEventListener('keydown', function(e){ if(e.key === 'Escape') closeMenu(); });

  // ── docked mini player (Suno-style) ──
  var lastNonce = null, bar = null, audio = null;
  function fmt(t){ if(!isFinite(t)) return '0:00'; var m=Math.floor(t/60), s=Math.floor(t%60); return m+':'+(s<10?'0':'')+s; }
  function buildBar(){
    if(bar) return bar;
    bar = document.createElement('div'); bar.id = 'yy-mini';
    bar.innerHTML = '<div class="yy-mini-prog"><div class="yy-mini-track"><div class="yy-mini-fill"></div></div></div>' +
      '<div class="yy-mini-row"><div class="yy-mini-art"></div>' +
      '<div class="yy-mini-meta"><div class="yy-mini-title"></div><div class="yy-mini-sub"></div></div>' +
      '<div class="yy-mini-time">0:00</div>' +
      '<button class="yy-mini-nav yy-mini-prev" aria-label="Zurück">⏮</button>' +
      '<button class="yy-mini-btn" aria-label="Play/Pause">▶</button>' +
      '<button class="yy-mini-nav yy-mini-next" aria-label="Nächstes Lied">⏭</button></div>';
    audio = document.createElement('audio'); audio.preload = 'auto'; bar.appendChild(audio);
    document.body.appendChild(bar); document.body.classList.add('yy-has-mini');
    var btn = bar.querySelector('.yy-mini-btn'), fill = bar.querySelector('.yy-mini-fill'), time = bar.querySelector('.yy-mini-time');
    btn.addEventListener('click', function(e){ e.stopPropagation(); if(audio.paused) audio.play(); else audio.pause(); });
    audio.addEventListener('play', function(){ btn.textContent = '❚❚'; });
    audio.addEventListener('pause', function(){ btn.textContent = '▶'; });
    audio.addEventListener('ended', function(){ btn.textContent = '▶'; step(1); });
    var dragging = false, prog = bar.querySelector('.yy-mini-prog');
    function paint(){ if(dragging) return;
      fill.style.width = (audio.duration ? 100*audio.currentTime/audio.duration : 0) + '%';
      time.textContent = fmt(audio.currentTime) + ' / ' + fmt(audio.duration); }
    audio.addEventListener('timeupdate', paint); audio.addEventListener('loadedmetadata', paint);
    function frac(e){ var r = prog.getBoundingClientRect(); return Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); }
    prog.addEventListener('pointerdown', function(e){ e.preventDefault(); e.stopPropagation(); dragging = true; prog.classList.add('drag');
      try{ prog.setPointerCapture(e.pointerId); }catch(x){} var f = frac(e); fill.style.width = 100*f + '%';
      time.textContent = fmt(f*(audio.duration||0)) + ' / ' + fmt(audio.duration); });
    prog.addEventListener('pointermove', function(e){ if(!dragging) return; var f = frac(e); fill.style.width = 100*f + '%';
      time.textContent = fmt(f*(audio.duration||0)) + ' / ' + fmt(audio.duration); });
    function release(e){ if(!dragging) return; dragging = false; prog.classList.remove('drag');
      if(audio.duration) audio.currentTime = audio.duration * frac(e); paint(); }
    prog.addEventListener('pointerup', release); prog.addEventListener('pointercancel', function(){ dragging = false; prog.classList.remove('drag'); paint(); });
    bar.querySelector('.yy-mini-next').addEventListener('click', function(e){ e.stopPropagation(); step(1); });
    bar.querySelector('.yy-mini-prev').addEventListener('click', function(e){ e.stopPropagation();
      if(audio.currentTime > 3) audio.currentTime = 0; else step(-1); });
    bar.querySelector('.yy-mini-meta').addEventListener('click', function(){
      var p = document.querySelector('#yy-player'); if(p) p.scrollIntoView({behavior:'smooth', block:'start'});
    });
    return bar;
  }
  var curRel = '';
  function step(dir){
    var list = Array.from(document.querySelectorAll('.yy-song[data-rel]')).filter(function(c){ return c.style.display !== 'none'; });
    if(!list.length) return;
    var i = list.findIndex(function(c){ return c.dataset.rel === curRel; });
    var n = list[(i + dir + list.length) % list.length];
    if(n) setBox('yy-pick', n.dataset.rel + '|' + Date.now());
  }
  function syncMini(){
    var d = document.querySelector('.yy-mini-data'); if(!d || d.dataset.nonce === lastNonce) return;
    lastNonce = d.dataset.nonce; buildBar(); curRel = d.dataset.rel || '';
    bar.querySelector('.yy-mini-title').textContent = d.dataset.title;
    bar.querySelector('.yy-mini-sub').textContent = d.dataset.sub;
    bar.querySelector('.yy-mini-art').setAttribute('style', d.dataset.art);
    audio.src = d.dataset.src; var p = audio.play(); if(p && p.catch) p.catch(function(){});
  }
  new MutationObserver(function(){ syncMini(); }).observe(document.documentElement, {subtree:true, childList:true});
  function studio(){
    const a = document.querySelector('#yy-studio a');
    const local = ['127.0.0.1','localhost'].includes(location.hostname);
    if(a && !local){ a.parentElement.style.display='none'; return; }
    if(a && a.dataset.port){ a.href = location.protocol+'//'+location.hostname+':'+a.dataset.port+'/'; }
    else setTimeout(studio, 500);
  }
  studio();
})();
</script>
<script>
(function(){
  // German is the source language; English is an optional view layer.
  var EXACT = {
    "Musik machen":"Make music",
    "🎵 Neuer Song":"🎵 New song","🎤 Cover":"🎤 Cover",
    "Was für ein Song?":"What kind of song?",
    "Genre & Stimmung (mehrere möglich)":"Genre & mood (pick several)",
    "Stimme":"Voice","Egal":"Any","👩 Frau":"👩 Female","👨 Mann":"👨 Male","👫 Duett":"👫 Duet","👥 Chor":"👥 Choir",
    "Antippen und auswählen – oder oben frei beschreiben":"Tap to choose – or describe freely above",
    "Songtext (kannst du leer lassen)":"Lyrics (optional)",
    "Ohne Gesang":"Instrumental","Länge":"Length","Kurz":"Short","Normal":"Normal","Lang":"Long",
    "Wie viele Versionen?":"How many versions?",
    "✨ Song erstellen":"✨ Create song","🎤 Cover erstellen":"🎤 Create cover",
    "📚 Aus meinen Songs":"📚 From my songs","📁 Datei oder 🎙️ Vorsingen":"📁 File or 🎙️ sing",
    "🔗 Link (YouTube, Google Drive …)":"🔗 Link (YouTube, Google Drive …)","Link einfügen":"Paste link",
    "Musikdatei hochladen – oder aufs Mikrofon tippen und vorsingen / summen":"Upload audio – or tap the mic and sing / hum",
    "YouTube, Google-Drive-Datei oder -Ordner („Jeder mit dem Link“), Dropbox, OneDrive oder direkte MP3":"YouTube, Google Drive file or folder (anyone with the link), Dropbox, OneDrive or a direct MP3",
    "Lieder im Ordner – antippen zum Anhören":"Songs in the folder – tap to listen","Vorhören":"Preview",
    "KI-Modell":"AI model","🌍 ACE-Step – alle Sprachen (auch Deutsch), schnell":"🌍 ACE-Step – all languages, fast",
    "🎼 YuE2 – Englisch, lange Songs":"🎼 YuE2 – English, long songs","🔎 Originaltext suchen":"🔎 Find original lyrics",
    "Leer lassen = die KI schreibt den Text (bei ACE-Step).":"Leave empty = the AI writes the lyrics (ACE-Step).",
    "Leer lassen = Originaltext wird automatisch gesucht.":"Leave empty = original lyrics are looked up automatically.",
    "Musikdatei":"Audio file","Welcher Song?":"Which song?","Neuer Stil":"New style",
    "Neuer Songtext (kannst du leer lassen)":"New lyrics (optional)",
    "Meine Songs":"My songs","🔁 Nochmal":"🔁 Again","⬇ Speichern":"⬇ Save","🗑 Löschen":"🗑 Delete",
    "Diesen Song wirklich löschen?":"Really delete this song?","Ja, löschen":"Yes, delete","Nein":"No",
    "⏹ Abbrechen":"⏹ Cancel","Songtext anzeigen":"Show lyrics","Profi-Modus (alle Einstellungen)":"Pro mode (all settings)",
    "Noch keine Songs.":"No songs yet.","Einstellungen":"Settings","Sprache":"Language",
    "Fröhlich":"Happy","Traurig":"Sad","Romantisch":"Romantic","Ruhig":"Calm","Energiegeladen":"Energetic",
    "Episch":"Epic","Langsam":"Slow","Schnell":"Fast","Frauenstimme":"Female voice","Männerstimme":"Male voice",
    "Chor":"Choir","Deutsch":"German","Englisch":"English","Klassik":"Classical","Orchester":"Orchestra",
    "Filmmusik":"Film score","Kinderlied":"Children's song","Weihnachten":"Christmas","Akustik":"Acoustic",
    "Klavier":"Piano","Gitarre":"Guitar","E-Gitarre":"Electric guitar","Geige":"Violin","Saxophon":"Saxophone",
    "Elektro":"Electronic","(ohne Gesang)":"(instrumental)","Alle Songs":"All songs"
  };
  var PHRASES = [
    ["Dein Song läuft.","Your song is playing."],["Fertig!","Done!"],
    ["Song wird erstellt … (das dauert ein paar Minuten)","Creating song … (takes a few minutes)"],
    ["Cover wird erstellt … (das dauert ein paar Minuten)","Creating cover … (takes a few minutes)"],
    ["Neue Version wird erstellt …","Creating a new version …"],["Neue Version fertig!","New version done!"],
    ["Cover fertig!","Cover done!"],["Alle ","All "],[" Cover-Versionen fertig!"," cover versions done!"],
    [" Versionen fertig!"," versions done!"],[" wird erstellt …"," is being created …"],
    [" fertig – "," done – "],["Version ","Version "],[" von "," of "],
    ["Melodie wird komponiert","Composing melody"],["Gesang & Musik entstehen","Creating vocals & music"],
    ["Klang wird erzeugt","Rendering sound"],["Audio wird fertiggestellt","Finishing audio"],
    ["Wird gespeichert","Saving"],["Wird vorbereitet","Preparing"],["KI-Modell wird geladen","Loading AI model"],
    ["Melodie wird herausgehört …","Listening for the melody …"],["YouTube-Audio wird geladen …","Loading YouTube audio …"],
    ["Wird in MP3 umgewandelt …","Converting to MP3 …"],["Songtext wird geschrieben …","Writing lyrics …"],
    ["Das hat nicht geklappt: ","That didn't work: "],["Es läuft gerade schon ein Song. Bitte kurz warten.","A song is already being made. Please wait."],
    ["Bitte schreib zuerst, was für ein Song es werden soll.","Please describe the song first."],
    ["Bitte zuerst eine Musikdatei hochladen.","Please upload an audio file first."],
    ["Bitte einen deiner Songs auswählen.","Please pick one of your songs."],
    ["Bitte einen Link einfügen (YouTube, Google Drive, Dropbox …).","Please paste a link (YouTube, Google Drive, Dropbox …)."],
    ["Datei wird geladen …","Downloading file …"],[" Lieder gefunden – eins antippen zum Anhören und Auswählen."," songs found – tap one to listen and select."],
    ["In diesem Ordner sind keine Musikdateien (oder er ist nicht freigegeben).","No audio files in this folder (or it isn't shared)."],
    ["Bitte im Ordner zuerst ein Lied auswählen.","Please pick a song from the folder first."],["Vorhören: ","Preview: "],["hat keine Audiodatei geliefert. Ist die Datei für „Jeder mit dem Link“ freigegeben?","returned no audio. Is the file shared with “anyone with the link”?"],
    ["Bitte zuerst eine Musikdatei hochladen oder etwas vorsingen.","Please upload an audio file or sing something first."],["oder","or"],
    ["Bitte zuerst einen Song anklicken.","Please tap a song first."],["Gelöscht.","Deleted."],
    ["Wird gestoppt …","Stopping …"],["Nicht mehr veröffentlicht.","Unpublished."],
    ["Veröffentlicht – über ⋯ → Teilen bekommst du den Link.","Published – use ⋯ → Share to get the link."],
    [" hinzugefügt."," added."],[" entfernt."," removed."],["Zu „","To “"],["Aus „","From “"],["Renamed to","Renamed to"],
    ["(Ohne Songtext wird er ohne Gesang – schreib einen Text, dann wird gesungen.)","(Without lyrics it's instrumental – add lyrics to get vocals.)"],
    ["Schreib oben, was du hören willst, und drück","Describe what you want to hear above and press"],
    ["Song erstellen","Create song"]
  ];
  var PH = {
    "z. B. fröhlicher Schlager über den Sommer, Frauenstimme":"e.g. happy summer pop song, female voice",
    "Leer lassen = die KI schreibt ihn.":"Leave empty = the AI writes them.",
    "Hier deinen Text eintippen. Leer = ohne Gesang.":"Type your lyrics here. Empty = instrumental.",
    "z. B. Rock mit Männerstimme":"e.g. rock with male voice",
    "Leer lassen = die KI schreibt den Text (bei ACE-Step).":"Leave empty = the AI writes the lyrics (ACE-Step).",
    "Leer lassen = Originaltext wird automatisch gesucht.":"Leave empty = original lyrics are looked up automatically.",
    "Leer lassen = alter Text bleibt (bei Datei/YouTube: ohne Gesang).":"Leave empty = keep old lyrics (file/YouTube: instrumental)."
  };
  var DEUI = {
    "Drop Audio Here":"Musik hierher ziehen","- or -":"– oder –","Click to Upload":"Zum Auswählen tippen",
    "Record":"Aufnehmen","Stop":"Stopp","Stop recording":"Aufnahme stoppen","Pause":"Pause","Resume":"Weiter",
    "Drop File Here":"Datei hierher ziehen","Upload file":"Datei hochladen","Clear":"Löschen","Download":"Herunterladen",
    "Share":"Teilen","Select":"Auswählen","Loading...":"Lädt …","Error":"Fehler","Close":"Schließen",
    "Microphone":"Mikrofon","Upload":"Hochladen","Trim":"Kürzen","Undo":"Rückgängig"
  };
  function lang(){ try { return localStorage.getItem('yy-lang') || 'de'; } catch(e){ return 'de'; } }
  var busy = false;
  function trDe(node){
    var walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, null), t;
    while((t = walker.nextNode())){
      var k = t.nodeValue.trim();
      if(k && DEUI[k]) t.nodeValue = t.nodeValue.replace(k, DEUI[k]);
    }
    node.querySelectorAll && node.querySelectorAll('[aria-label],[title]').forEach(function(el){
      ['aria-label','title'].forEach(function(a){ var v = el.getAttribute(a); if(v && DEUI[v]) el.setAttribute(a, DEUI[v]); });
    });
  }
  function tr(node){
    if(lang() !== 'en') return trDe(node);
    var walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, null);
    var t;
    while((t = walker.nextNode())){
      if(t.parentElement && /^(SCRIPT|STYLE|TEXTAREA)$/.test(t.parentElement.tagName)) continue;
      var v = t.nodeValue, k = v.trim();
      if(!k) continue;
      if(EXACT[k]){ t.nodeValue = v.replace(k, EXACT[k]); continue; }
      if(!/[a-zäöüß]/i.test(k)) continue;
      var n = v;
      for(var i=0;i<PHRASES.length;i++){ if(n.indexOf(PHRASES[i][0])>=0) n = n.split(PHRASES[i][0]).join(PHRASES[i][1]); }
      if(n !== v) t.nodeValue = n;
    }
    node.querySelectorAll && node.querySelectorAll('textarea[placeholder],input[placeholder]').forEach(function(el){
      var p = el.getAttribute('placeholder'); if(PH[p]) el.setAttribute('placeholder', PH[p]);
    });
  }
  function run(){ if(busy) return; busy = true; try { tr(document.body); } finally { busy = false; } }
  new MutationObserver(function(){ clearTimeout(window.__yyT); window.__yyT = setTimeout(run, 30); })
    .observe(document.documentElement, {subtree:true, childList:true, characterData:true});
  window.yySetLang = function(l){ try { localStorage.setItem('yy-lang', l); } catch(e){} location.reload(); };
  function mark(){
    var b = document.querySelectorAll('[data-lang]');
    if(!b.length) return setTimeout(mark, 300);
    b.forEach(function(x){ x.classList.toggle('on', x.dataset.lang === lang()); });
    document.documentElement.lang = lang();
    run();
  }
  document.addEventListener('click', function(e){
    var b = e.target.closest('[data-lang]'); if(b){ e.preventDefault(); window.yySetLang(b.dataset.lang); }
  });
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mark); else mark();
})();
</script>
"""
HEAD = HEAD.replace("__ICON__", ICON_180)


def _chips(target: str) -> str:
    return (
        '<div class="yy-chips">'
        + "".join(
            f'<span class="yy-chip" data-chip="{html.escape(c)}" data-target="{target}">+ {c}</span>'
            for c in CHIPS
        )
        + "</div>"
    )


def theme():
    return gr.themes.Soft(primary_hue="pink", neutral_hue="zinc").set(
        body_background_fill="#0e0e12",
        body_background_fill_dark="#0e0e12",
        body_text_color="#f3f1ee",
        body_text_color_dark="#f3f1ee",
        block_background_fill="#17171d",
        block_background_fill_dark="#17171d",
        block_border_color="#26262e",
        block_border_color_dark="#26262e",
        input_background_fill="#1f1f27",
        input_background_fill_dark="#1f1f27",
        block_label_text_color="#9a98a3",
        block_label_text_color_dark="#9a98a3",
        block_title_text_color="#f3f1ee",
        block_title_text_color_dark="#f3f1ee",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_title_background_fill="transparent",
        block_title_background_fill_dark="transparent",
        button_secondary_background_fill="#26262e",
        button_secondary_background_fill_dark="#26262e",
        button_secondary_background_fill_hover="#33333d",
        button_secondary_background_fill_hover_dark="#33333d",
        button_secondary_text_color="#f3f1ee",
        button_secondary_text_color_dark="#f3f1ee",
        button_secondary_border_color="#33333d",
        button_secondary_border_color_dark="#33333d",
        checkbox_label_background_fill="#1f1f27",
        checkbox_label_background_fill_dark="#1f1f27",
        checkbox_label_text_color="#f3f1ee",
        checkbox_label_text_color_dark="#f3f1ee",
        checkbox_label_background_fill_hover="#26262e",
        checkbox_label_background_fill_hover_dark="#26262e",
        checkbox_label_border_color="#33333d",
        checkbox_label_border_color_dark="#33333d",
    )


def build_simple_ui(defaults: dict, studio_port: int | None = None) -> gr.Blocks:
    global _SETTINGS
    _SETTINGS = runtime.RuntimeSettings(
        defaults["device"],
        defaults["dtype"],
        defaults.get("backend", "torch"),
        "none",
        False,
        24,
        32,
        "auto",
        defaults["model"],
        defaults.get("vae", "standard"),
        "",
        "",
        "",
        False,
    )

    with gr.Blocks(title="YuE2 Musik") as demo:
        gr.HTML(
            '<div id="yy-head"><div class="yy-lang" title="Sprache / Language">'
            '<a href="#" data-lang="de">DE</a><a href="#" data-lang="en">EN</a></div>'
            '<h1>🎵 <span>Musik machen</span></h1></div>'
        )
        with gr.Row(equal_height=False):
            # ─────────── left: create ───────────
            with gr.Column(scale=5, min_width=260):
                ace_ok = ace_engine.available()
                engine = gr.Radio(
                    ENGINES,
                    value="ace" if ace_ok else "yue",
                    label="KI-Modell",
                    info=None if ace_ok else "ACE-Step wird gerade installiert – bis dahin YuE2.",
                    elem_id="yy-engine",
                )
                with gr.Tabs() as tabs:
                    with gr.Tab("🎵 Neuer Song", id="new"):
                        with gr.Column(elem_classes=["yy-box"]):
                            style = gr.Textbox(
                                label="Was für ein Song?",
                                placeholder="z. B. fröhlicher Schlager über den Sommer, Frauenstimme",
                                lines=2,
                                elem_id="yy-style",
                            )
                            with gr.Row(elem_classes=["yy-genre-row"]):
                              tags = gr.Dropdown(
                                TAGS, scale=3,
                                multiselect=True,
                                label="Genre & Stimmung (mehrere möglich)",
                                info="Antippen und auswählen – oder oben frei beschreiben",
                                elem_id="yy-tags",
                              )
                              voice = gr.Radio(VOICES, value="", label="Stimme", scale=2,
                                               elem_classes=["yy-voice"])
                            lyrics = gr.Textbox(
                                label="Songtext (kannst du leer lassen)",
                                placeholder=(
                                    "Leer lassen = die KI schreibt den Text (bei ACE-Step)."
                                ),
                                lines=6,
                            )
                            with gr.Row():
                                instrumental = gr.Checkbox(label="Ohne Gesang", value=False)
                                length = gr.Radio(list(LENGTHS), value="Normal", label="Länge")
                            versions = gr.Slider(
                                1, MAX_VERSIONS, value=2, step=1, label="Wie viele Versionen?",
                                elem_classes=["yy-versions"],
                            )
                            create_btn = gr.Button("✨ Song erstellen", elem_id="yy-create")
                    with gr.Tab("🎤 Cover", id="cover"):
                        with gr.Column(elem_classes=["yy-box"]):
                            source = gr.Radio(
                                [
                                    ("📚 Aus meinen Songs", "library"),
                                    ("📁 Datei oder 🎙️ Vorsingen", "file"),
                                    ("🔗 Link (YouTube, Google Drive …)", "link"),
                                ],
                                value="library",
                                show_label=False,
                            )
                            upload = gr.Audio(
                                label="Musikdatei hochladen – oder aufs Mikrofon tippen und vorsingen / summen",
                                sources=["upload", "microphone"],
                                type="filepath",
                                visible=False,
                            )
                            yt_url = gr.Textbox(
                                label="Link einfügen",
                                info="YouTube, Google-Drive-Datei oder -Ordner („Jeder mit dem Link“), Dropbox, OneDrive oder direkte MP3",
                                placeholder="https://…",
                                lines=1,
                                visible=False,
                            )
                            folder_list = gr.Radio(
                                [], label="Lieder im Ordner – antippen zum Anhören",
                                visible=False, elem_id="yy-folder-list",
                            )
                            folder_preview = gr.Audio(
                                label="Vorhören", interactive=False, visible=False, type="filepath",
                            )
                            pick_song = gr.Dropdown(
                                label="Welcher Song?", choices=cover_choices(), interactive=True
                            )
                            cover_style = gr.Textbox(
                                label="Neuer Stil",
                                placeholder="z. B. Rock mit Männerstimme",
                                lines=2,
                                elem_id="yy-cover-style",
                            )
                            with gr.Row(elem_classes=["yy-genre-row"]):
                              cover_tags = gr.Dropdown(
                                TAGS, scale=3,
                                multiselect=True,
                                label="Genre & Stimmung (mehrere möglich)",
                                elem_id="yy-cover-tags",
                              )
                              cover_voice = gr.Radio(VOICES, value="", label="Stimme", scale=2,
                                                     elem_classes=["yy-voice"])
                            cover_lyrics = gr.Textbox(
                                label="Neuer Songtext (kannst du leer lassen)",
                                placeholder="Leer lassen = Originaltext wird automatisch gesucht.",
                                lines=4,
                            )
                            find_lyrics_btn = gr.Button("🔎 Originaltext suchen", size="sm")
                            cover_versions = gr.Slider(
                                1, MAX_VERSIONS, value=2, step=1, label="Wie viele Versionen?",
                                elem_classes=["yy-versions"],
                            )
                            cover_btn = gr.Button("🎤 Cover erstellen", elem_id="yy-cover-btn")
                status = gr.Markdown("", elem_id="yy-status")
                stop_btn = gr.Button("⏹ Abbrechen", size="sm", visible=False)

            # ─────────── right: my songs ───────────
            with gr.Column(scale=6, min_width=260):
                with gr.Column(visible=False, elem_classes=["yy-box"], elem_id="yy-player") as player_box:
                    now = gr.HTML("")
                    # playback happens in the docked mini player (JS); this one stays hidden
                    player = gr.Audio(
                        show_label=False, interactive=False, autoplay=False, type="filepath",
                        visible=False,
                    )
                    with gr.Row(elem_classes=["yy-actions"]):
                        again_btn = gr.Button("🔁 Nochmal", min_width=90)
                        to_cover_btn = gr.Button("🎤 Cover", min_width=90)
                        download = gr.DownloadButton("⬇ Speichern", min_width=90)
                        del_btn = gr.Button("🗑 Löschen", min_width=90)
                    with gr.Row(visible=False) as confirm_row:
                        confirm_text = gr.Markdown("**Diesen Song wirklich löschen?**")
                        yes_btn = gr.Button("Ja, löschen", variant="stop", size="sm")
                        no_btn = gr.Button("Nein", size="sm")
                gr.Markdown("### Meine Songs")
                songs = gr.HTML(songs_html())
        gr.HTML(
            f'<div id="yy-studio"><a data-port="{studio_port or ""}" target="_blank">'
            "Profi-Modus (alle Einstellungen)</a></div>"
            if studio_port
            else ""
        )

        pick = gr.Textbox(elem_id="yy-pick", elem_classes=["yy-hide"], show_label=False)
        del_pick = gr.Textbox(elem_id="yy-del", elem_classes=["yy-hide"], show_label=False)
        act_remix_box = gr.Textbox(elem_id="yy-act-remix", elem_classes=["yy-hide"], show_label=False)
        act_cover_box = gr.Textbox(elem_id="yy-act-cover", elem_classes=["yy-hide"], show_label=False)
        act_rename_box = gr.Textbox(elem_id="yy-act-rename", elem_classes=["yy-hide"], show_label=False)
        act_pl_box = gr.Textbox(elem_id="yy-act-pl", elem_classes=["yy-hide"], show_label=False)
        act_pub_box = gr.Textbox(elem_id="yy-act-pub", elem_classes=["yy-hide"], show_label=False)
        act_reuse_box = gr.Textbox(elem_id="yy-act-reuse", elem_classes=["yy-hide"], show_label=False)
        act_edit_box = gr.Textbox(elem_id="yy-act-edit", elem_classes=["yy-hide"], show_label=False)
        current = gr.State("")

        show_outputs = [player_box, player, now, download, current, songs, confirm_row]
        pick.change(show_song, inputs=[pick], outputs=show_outputs)

        def ask_delete(value):
            out = list(show_song(value))
            if isinstance(out[2], str):
                out[2] = re.sub(r'<div class="yy-mini-data"[^>]*></div>', "", out[2])
            out[-1] = gr.update(visible=bool((value or "").split("|")[0]))
            return tuple(out)

        del_pick.change(ask_delete, inputs=[del_pick], outputs=show_outputs)

        def remix_from_menu(value, progress=gr.Progress()):
            yield from again((value or "").split("|")[0], progress)

        act_remix_box.change(
            remix_from_menu, inputs=[act_remix_box], outputs=[status, pick, stop_btn],
            show_progress_on=[status],
        )
        act_cover_box.change(
            lambda v: go_cover((v or "").split("|")[0]),
            inputs=[act_cover_box],
            outputs=[tabs, source, pick_song, upload, yt_url],
        )
        act_rename_box.change(act_rename, inputs=[act_rename_box], outputs=[status, songs])
        act_pl_box.change(act_playlist, inputs=[act_pl_box], outputs=[status, songs])
        act_pub_box.change(act_publish, inputs=[act_pub_box], outputs=[status, songs])
        act_edit_box.change(
            act_edit, inputs=[act_edit_box], outputs=[status, pick, stop_btn], show_progress_on=[status]
        )
        act_reuse_box.change(act_reuse, inputs=[act_reuse_box], outputs=[tabs, style, lyrics, tags])

        def create_song_v(style_v, tags_v, voice_v, *rest, progress=gr.Progress()):
            yield from create_song(style_v, _with_voice(tags_v, voice_v), *rest, progress=progress)

        def create_cover_v(*args, progress=gr.Progress()):
            args = list(args)
            voice_v = args.pop(7)
            args[6] = _with_voice(args[6], voice_v)
            yield from create_cover(*args, progress=progress)

        create_btn.click(
            create_song_v,
            inputs=[style, tags, voice, lyrics, instrumental, length, versions, engine],
            outputs=[status, create_btn, pick, stop_btn],
            show_progress_on=[status],
        )
        cover_btn.click(
            create_cover_v,
            inputs=[
                source, upload, yt_url, folder_list, pick_song, cover_style, cover_tags, cover_voice,
                cover_lyrics,
                cover_versions, engine,
            ],
            outputs=[status, cover_btn, pick, stop_btn],
            show_progress_on=[status],
        )
        again_btn.click(
            again, inputs=[current], outputs=[status, pick, stop_btn], show_progress_on=[status]
        )
        to_cover_btn.click(
            go_cover,
            inputs=[current],
            outputs=[tabs, source, pick_song, upload, yt_url],
        )
        source.change(_source_toggle, inputs=[source], outputs=[upload, yt_url, pick_song])

        def find_original(src, up, url, folder_value, rel):
            try:
                duration = None
                if src == "library":
                    item, det = library.load(runtime.RUNS, rel or "")
                    if item is None or not det:
                        return gr.update(), "✋ Bitte zuerst einen Song auswählen."
                    text = str((det.get("request") or {}).get("lyrics") or "").strip()
                    if not text or text == INSTRUMENTAL_LYRICS.strip():
                        return gr.update(), "📝 Dieser Song hat keinen Songtext."
                    return text, "📝 Songtext des Liedes übernommen."
                if src == "file":
                    if not up:
                        return gr.update(), "✋ Bitte zuerst eine Datei hochladen."
                    name, duration = Path(up).name, _audio_seconds(up)
                elif youtube.drive_folder_id(url or ""):
                    if not folder_value:
                        return gr.update(), "✋ Bitte im Ordner zuerst ein Lied auswählen."
                    name = str(folder_value).partition("|")[2]
                elif youtube.looks_like_youtube(url or ""):
                    name = youtube.video_title(url)
                else:
                    name = Path(urllib.parse.urlparse(url or "").path).name
                found, label = lyrics_db.find(name, duration)
            except Exception as exc:  # noqa: BLE001
                return gr.update(), "❌ Suche fehlgeschlagen: " + str(exc)
            if not found:
                return gr.update(), f"🔎 Kein Songtext gefunden für „{label}“ – du kannst ihn selbst eintippen."
            return found, f"📝 Originaltext für „{label}“ gefunden – du kannst ihn noch ändern."

        find_lyrics_btn.click(
            find_original,
            inputs=[source, upload, yt_url, folder_list, pick_song],
            outputs=[cover_lyrics, status],
        )

        def open_folder(url):
            if not youtube.drive_folder_id(url):
                return gr.update(visible=False, choices=[], value=None), gr.update(visible=False, value=None), gr.update()
            try:
                files = youtube.list_drive_folder(url)
            except Exception as exc:  # noqa: BLE001
                return (
                    gr.update(visible=False, choices=[], value=None),
                    gr.update(visible=False, value=None),
                    "❌ " + str(exc),
                )
            if not files:
                return (
                    gr.update(visible=False, choices=[], value=None),
                    gr.update(visible=False, value=None),
                    "📂 In diesem Ordner sind keine Musikdateien (oder er ist nicht freigegeben).",
                )
            choices = [(f"🎵 {f['path']}", f"{f['id']}|{f['name']}") for f in files]
            return (
                gr.update(visible=True, choices=choices, value=None),
                gr.update(visible=False, value=None),
                f"📂 {len(files)} Lieder gefunden – eins antippen zum Anhören und Auswählen.",
            )

        def preview_folder_file(value):
            if not value:
                return gr.update(visible=False, value=None)
            file_id, _, fname = str(value).partition("|")
            try:
                path = youtube.download_drive_file(file_id, fname, runtime.RUNS / "gdrive")
                path = youtube.as_audio(path)
            except Exception:  # noqa: BLE001
                return gr.update(visible=False, value=None)
            return gr.update(visible=True, value=path, label=f"Vorhören: {fname}")

        source.change(
            lambda src, url: (
                (gr.update(), gr.update()) if src == "link" and youtube.drive_folder_id(url)
                else (gr.update(visible=False), gr.update(visible=False))
            ),
            inputs=[source, yt_url],
            outputs=[folder_list, folder_preview],
        )
        yt_url.change(open_folder, inputs=[yt_url], outputs=[folder_list, folder_preview, status])
        folder_list.change(
            preview_folder_file, inputs=[folder_list], outputs=[folder_preview],
            show_progress_on=[folder_preview],
        )
        del_btn.click(lambda: gr.update(visible=True), outputs=[confirm_row])
        no_btn.click(lambda: gr.update(visible=False), outputs=[confirm_row])
        yes_btn.click(
            delete_song,
            inputs=[current],
            outputs=[status, player_box, player, now, songs, confirm_row],
        )
        stop_btn.click(stop, outputs=[status])
        demo.load(lambda: (songs_html(), gr.update(choices=cover_choices())), outputs=[songs, pick_song])
    return demo
