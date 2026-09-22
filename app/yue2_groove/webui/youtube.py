"""YouTube link → audio file for the COVER tab (yt-dlp + a bundled ffmpeg).

The tools are installed into the app's environment on first use:
``yt-dlp[default]`` (with the YouTube JS-challenge solver), ``imageio-ffmpeg``
(a static ffmpeg, no system install needed) and ``deno`` (the JavaScript runtime
yt-dlp needs for YouTube).  Only use recordings you are allowed to use.
"""

from __future__ import annotations

import importlib
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("yue2_groove")

MAX_SECONDS = 12 * 60
PACKAGES = ["yt-dlp[default]", "imageio-ffmpeg", "deno"]
_URL_RE = re.compile(r"^https?://(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/", re.I)


def looks_like_youtube(url: str) -> bool:
    return bool(_URL_RE.match((url or "").strip()))


_GDRIVE_RE = re.compile(r"(?:drive|docs)\.google\.com/(?:file/d/|open\?id=|uc\?(?:.*&)?id=)([\w-]{10,})")


def looks_like_link(url: str) -> bool:
    return bool(re.match(r"^https?://", (url or "").strip(), re.I))


def _direct_url(url: str) -> tuple[str, str]:
    """(download url, kind) for Google Drive / Dropbox / OneDrive share links."""
    m = _GDRIVE_RE.search(url)
    if m:
        return (
            f"https://drive.usercontent.google.com/download?id={m.group(1)}&export=download&confirm=t",
            "Google Drive",
        )
    if "dropbox.com" in url:
        base = re.sub(r"[?&]dl=[01]", "", url)
        return base + ("&" if "?" in base else "?") + "dl=1", "Dropbox"
    if "1drv.ms" in url or "onedrive.live.com" in url:
        return url + ("&" if "?" in url else "?") + "download=1", "OneDrive"
    return url, "Link"


AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma", ".mp4", ".webm", ".mkv", ".mov", ".aiff", ".aif")
_FOLDER_RE = re.compile(r"(?:drive\.google\.com/(?:drive/(?:u/\d+/)?folders/|embeddedfolderview\?(?:.*&)?id=))([\w-]{10,})")


def drive_folder_id(url: str) -> str | None:
    m = _FOLDER_RE.search(url or "")
    return m.group(1) if m else None


def list_drive_folder(url_or_id: str, depth: int = 2, limit: int = 300) -> list[dict]:
    """Audio files of a public ("anyone with the link") Google Drive folder, sub-folders
    included up to *depth* levels: [{"id", "name", "path"}]."""
    import html as _html
    import urllib.request

    folder = drive_folder_id(url_or_id) or url_or_id
    found: list[dict] = []

    def visit(fid: str, prefix: str, level: int) -> None:
        if len(found) >= limit:
            return
        req = urllib.request.Request(
            f"https://drive.google.com/embeddedfolderview?id={fid}",
            headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "de,en"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                page = resp.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            if level == 0:
                raise ValueError(
                    "Ordner nicht lesbar. Ist er für „Jeder mit dem Link“ freigegeben?"
                ) from exc
            return
        for chunk in page.split('class="flip-entry"')[1:]:
            m_id = re.search(r'id="entry-([\w-]+)"', chunk)
            m_href = re.search(r'<a href="([^"]+)"', chunk)
            m_name = re.search(r'flip-entry-title">([^<]*)<', chunk)
            if not (m_id and m_href and m_name):
                continue
            name = _html.unescape(m_name.group(1)).strip()
            if "/folders/" in m_href.group(1):
                if level < depth:
                    visit(m_id.group(1), f"{prefix}{name}/", level + 1)
            elif name.lower().endswith(AUDIO_EXT):
                found.append({"id": m_id.group(1), "name": name, "path": prefix + name})
                if len(found) >= limit:
                    return

    visit(folder, "", 0)
    return found


def download_drive_file(file_id: str, name: str, target_dir, progress=None) -> str:
    """Download one Drive file into *target_dir* (cached by id); returns the path."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\- ]+", "_", name)[:80] or "audio.mp3"
    out = target_dir / f"{file_id}-{safe}"
    if out.is_file() and out.stat().st_size > 0:
        return str(out)
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    return _fetch(url, out, "Google Drive", progress)


def _fetch(url: str, out: Path, kind: str, progress=None) -> str:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 YuEMusik"})
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        if "text/html" in resp.headers.get("Content-Type", ""):
            raise ValueError(
                f"{kind} hat keine Audiodatei geliefert. Ist die Datei für "
                "„Jeder mit dem Link“ freigegeben?"
            )
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if done > 400 * 1024 * 1024:
                    raise ValueError("Die Datei ist zu groß (über 400 MB).")
                if progress and total:
                    progress(0.02 + 0.08 * done / total, f"{kind}: Datei wird geladen …")
    tmp.replace(out)
    return str(out)


def download_link(url: str, target_dir, progress=None) -> tuple[str, str]:
    """YouTube via yt-dlp, anything else (Google Drive, Dropbox, OneDrive, direct MP3)
    via a plain download. Returns (audio path, title)."""
    url = (url or "").strip()
    if looks_like_youtube(url):
        return download_audio(url, target_dir, progress)
    if not looks_like_link(url):
        raise ValueError("Bitte einen Link einfügen (YouTube, Google Drive, Dropbox …).")
    import urllib.request
    from email.message import Message

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    direct, kind = _direct_url(url)
    if progress:
        progress(0.02, f"{kind}: Datei wird geladen …")
    req = urllib.request.Request(direct, headers={"User-Agent": "Mozilla/5.0 YuEMusik"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise ValueError(
                f"{kind} hat keine Audiodatei geliefert. Ist die Datei für "
                "„Jeder mit dem Link“ freigegeben?"
            )
        msg = Message()
        msg["content-disposition"] = resp.headers.get("Content-Disposition", "")
        name = msg.get_filename() or Path(urllib.request.urlparse(direct).path).name or "audio"
        name = re.sub(r"[^\w.\- ]+", "_", name)[:80] or "audio"
        if "." not in name:
            name += ".mp3"
        out = target_dir / f"{int(__import__('time').time())}-{name}"
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(out, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if done > 400 * 1024 * 1024:
                    raise ValueError("Die Datei ist zu groß (über 400 MB).")
                if progress and total:
                    progress(0.02 + 0.08 * done / total, f"{kind}: Datei wird geladen …")
    return str(out), Path(name).stem


def _pip_install(packages) -> None:
    for cmd in (
        [sys.executable, "-m", "pip", "install", "-q", "-U", *packages],
        ["uv", "pip", "install", "--python", sys.executable, "-q", "-U", *packages],
    ):
        try:
            subprocess.run(cmd, check=True, timeout=900)  # noqa: S603
            importlib.invalidate_caches()
            return
        except Exception:  # noqa: BLE001, S112
            continue
    raise RuntimeError("Konnte yt-dlp nicht installieren (keine Internetverbindung?)")


def _ensure_tools():
    try:
        import imageio_ffmpeg  # noqa: F401
        import yt_dlp  # noqa: F401
    except ImportError:
        log.info("youtube: installing yt-dlp / ffmpeg / deno …")
        _pip_install(PACKAGES)
    import imageio_ffmpeg
    import yt_dlp

    return yt_dlp, imageio_ffmpeg.get_ffmpeg_exe()


def _js_runtimes() -> dict:
    """deno from the venv (pip package) and any node on PATH / in Pinokio."""
    runtimes: dict = {}
    scripts = Path(sys.executable).parent
    for name in ("deno", "node"):
        found = shutil.which(name) or shutil.which(name, path=str(scripts))
        if not found and name == "node":
            for guess in (Path("C:/pinokio/bin"), Path.home() / "pinokio" / "bin"):
                hits = list(guess.glob("**/node.exe"))[:1] if guess.is_dir() else []
                if hits:
                    found = str(hits[0])
                    break
        if found:
            runtimes[name] = {"path": found}
    return runtimes or {"deno": {}}


def download_audio(url: str, target_dir, progress=None) -> tuple[str, str]:
    """Download *url*'s audio as MP3 into *target_dir*; returns (path, title)."""
    url = (url or "").strip()
    if not looks_like_youtube(url):
        raise ValueError("Das ist kein YouTube-Link.")
    yt_dlp, ffmpeg = _ensure_tools()
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    def hook(d):
        if progress is None:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            frac = (d.get("downloaded_bytes", 0) / total) if total else 0.0
            progress(0.02 + 0.08 * min(1.0, frac), "YouTube-Audio wird geladen …")
        elif d.get("status") == "finished":
            progress(0.1, "Wird in MP3 umgewandelt …")

    base = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "js_runtimes": _js_runtimes(),
        "remote_components": ["ejs:github"],
        "ffmpeg_location": ffmpeg,
    }
    with yt_dlp.YoutubeDL(base) as ydl:
        info = ydl.extract_info(url, download=False)
    duration = info.get("duration") or 0
    if duration and duration > MAX_SECONDS:
        raise ValueError(f"Das Video ist zu lang ({duration // 60} Min). Maximal 12 Minuten.")
    title = str(info.get("title") or "YouTube")
    opts = {
        **base,
        "format": "bestaudio/best",
        "outtmpl": str(target_dir / "%(id)s.%(ext)s"),
        "progress_hooks": [hook],
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        result = ydl.extract_info(url, download=True)
    downloads = result.get("requested_downloads") or []
    path = downloads[0].get("filepath") if downloads else None
    if not path or not Path(path).is_file():
        hits = sorted(target_dir.glob(f"{result.get('id', '')}*.mp3"))
        path = str(hits[0]) if hits else None
    if not path:
        raise RuntimeError("Download fertig, aber keine MP3-Datei gefunden.")
    return path, title


READABLE = {".mp3", ".wav", ".flac", ".ogg"}


def as_audio(path: str) -> str:
    """Convert anything ffmpeg can read (m4a, mp4, webm, aac, opus …) to MP3 next to it."""
    src = Path(path)
    if src.suffix.lower() in READABLE:
        return str(src)
    try:
        import imageio_ffmpeg
    except ImportError:
        _pip_install(["imageio-ffmpeg"])
        import imageio_ffmpeg
    out = src.with_suffix(".converted.mp3")
    subprocess.run(  # noqa: S603
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
         "-vn", "-ac", "2", "-b:a", "192k", str(out)],
        check=True,
        timeout=600,
    )
    return str(out)


def video_title(url: str) -> str:
    """Title of a YouTube video without downloading it (for the lyrics lookup)."""
    yt_dlp, _ffmpeg = _ensure_tools()
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "js_runtimes": _js_runtimes(), "remote_components": ["ejs:github"]}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return str(info.get("title") or "")
