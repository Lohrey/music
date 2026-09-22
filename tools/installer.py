"""Musik machen – Installer, Starter, Update und Backup ohne Pinokio.

Wird von den .bat-Dateien (Windows) bzw. .sh-Dateien (Linux / Docker) aufgerufen:

    python tools/installer.py install     # alles installieren (mehrfach ausführbar)
    python tools/installer.py start       # Server starten + Browser öffnen
    python tools/installer.py update      # neuen Code holen + Pakete nachziehen
    python tools/installer.py backup      # alle Songs in Backups/Songs-<Datum>.zip
    python tools/installer.py restore [zip]

Nur Standardbibliothek: läuft mit jeder Python-Version, die uv mitbringt.
Grafikkarte wird erkannt: RTX 20xx und neuer -> CUDA 12.8, ältere (z. B. GTX 1070) ->
CUDA-12.6-Pakete für die App, ältere torch-Version für ACE-Step, Vulkan für den
YuE2-GGUF-Motor.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
ACE = ROOT / "ace-step"
SETTINGS = ROOT / "einstellungen.txt"
SETTINGS_EXAMPLE = ROOT / "einstellungen.beispiel.txt"
BACKUPS = ROOT / "Backups"
WIN = sys.platform == "win32"

# ── feste Versionen (so wie auf dem Shadow PC getestet) ──
REPO_URL = "https://github.com/Lohrey/music.git"
YUE2_TAG = "yue2-v0.1.6"
YUE2_GIT = "https://github.com/multimodal-art-projection/YuE"
ACE_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
ACE_GIT = "https://github.com/ace-step/ACE-Step-1.5"
TORCH_MAIN = "2.10.0"
DEFAULT_PORT = 42003


# ─────────────────────────────── Ausgabe ───────────────────────────────

def say(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"    [OK] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"    [!] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"\n[FEHLER] {msg}\n", flush=True)
    sys.exit(1)


def run(cmd: list, cwd: Path | None = None, check: bool = True, env: dict | None = None,
        quiet: bool = False) -> subprocess.CompletedProcess:
    shown = " ".join(str(c) for c in cmd)
    print(f"    $ {shown[:220]}", flush=True)
    proc = subprocess.run(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
        env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None,
        text=True, encoding="utf-8", errors="replace",
    )
    if check and proc.returncode != 0:
        if quiet and proc.stdout:
            print(proc.stdout[-3000:])
        die(f"Befehl fehlgeschlagen ({proc.returncode}): {shown[:160]}")
    return proc


def out(cmd: list, cwd: Path | None = None) -> str:
    try:
        return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=120).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# ─────────────────────────────── Werkzeuge ───────────────────────────────

def uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    for cand in (Path.home() / ".local" / "bin" / ("uv.exe" if WIN else "uv"),
                 Path.home() / ".cargo" / "bin" / ("uv.exe" if WIN else "uv")):
        if cand.is_file():
            return str(cand)
    die("uv fehlt. Bitte installieren.bat erneut starten (die installiert uv automatisch).")
    return ""


def git() -> str | None:
    found = shutil.which("git")
    if found:
        return found
    if WIN:
        for cand in (Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "cmd" / "git.exe",
                     Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Git" / "cmd" / "git.exe",
                     Path.home() / "AppData" / "Local" / "Programs" / "Git" / "cmd" / "git.exe"):
            if cand.is_file():
                # frisch per winget installiert: dieses Fenster kennt den PATH noch nicht
                os.environ["PATH"] = str(cand.parent) + os.pathsep + os.environ.get("PATH", "")
                return str(cand)
    return None


def ensure_git() -> str | None:
    g = git()
    if g or not WIN:
        return g
    if shutil.which("winget"):
        say("Git wird installiert (für Updates) …")
        run(["winget", "install", "--id", "Git.Git", "-e", "--silent",
             "--accept-package-agreements", "--accept-source-agreements"], check=False)
    return git()


def py(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if WIN else "bin/python")


def download_zip(url: str, target: Path) -> None:
    """GitHub-Archiv laden und ohne den obersten Ordner nach *target* entpacken."""
    print(f"    ↓ {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "musik-machen-installer"})
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310
        data = resp.read()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.infolist():
            parts = member.filename.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            dest = target / parts[1]
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(member))


# ─────────────────────────────── Grafikkarte ───────────────────────────────

def gpu() -> dict:
    smi = shutil.which("nvidia-smi")
    if not smi and WIN:
        cand = Path(r"C:\Windows\System32\nvidia-smi.exe")
        smi = str(cand) if cand.is_file() else None
    info = {"name": None, "vram_gb": None, "cc": None}
    if not smi:
        return info
    line = out([smi, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"])
    if not line:
        line = out([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    parts = [p.strip() for p in line.splitlines()[0].split(",")] if line else []
    if parts:
        info["name"] = parts[0]
        try:
            info["vram_gb"] = round(float(parts[1]) / 1024, 1)
        except (IndexError, ValueError):
            pass
        try:
            info["cc"] = float(parts[2])
        except (IndexError, ValueError):
            pass
    return info


def torch_index(g: dict) -> str:
    """cu126 für ältere Karten (Pascal/Volta/…) – cu128-Pakete können dort nicht rechnen."""
    if g.get("cc") is not None and g["cc"] < 7.5:
        return "cu126"
    return "cu128"


# ─────────────────────────────── Einstellungen ───────────────────────────────

def read_settings() -> dict:
    data = {}
    if SETTINGS.is_file():
        for raw in SETTINGS.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def set_setting(key: str, value: str) -> None:
    lines = SETTINGS.read_text(encoding="utf-8").splitlines() if SETTINGS.is_file() else []
    done = False
    for i, raw in enumerate(lines):
        s = raw.strip().lstrip("#").strip()
        if s.startswith(key + "="):
            lines[i] = f"{key}={value}"
            done = True
            break
    if not done:
        lines.append(f"{key}={value}")
    SETTINGS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_settings(interactive: bool) -> None:
    if not SETTINGS.is_file():
        shutil.copyfile(SETTINGS_EXAMPLE, SETTINGS)
        # vom alten Pinokio-Setup übernehmen, falls vorhanden
        for old in pinokio_dirs():
            f = old / "app" / "handy-zugang.txt"
            if f.is_file():
                for raw in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    s = raw.strip()
                    if s and not s.startswith("#") and "=" in s:
                        k, v = s.split("=", 1)
                        set_setting(k.strip(), v.strip())
                ok(f"Einstellungen aus {f} übernommen")
                break
    cfg = read_settings()
    if interactive and not cfg.get("NGROK_AUTHTOKEN"):
        print("\n  Handy-Zugang (optional): ngrok-Token von https://dashboard.ngrok.com")
        print("  (einfach Enter drücken = überspringen, geht später in einstellungen.txt)")
        try:
            token = input("  ngrok-Token: ").strip()
        except EOFError:
            token = ""
        if token:
            set_setting("NGROK_AUTHTOKEN", token)
            try:
                domain = input("  Feste ngrok-Adresse (z. B. xyz.ngrok-free.dev, Enter = keine): ").strip()
            except EOFError:
                domain = ""
            if domain:
                set_setting("NGROK_DOMAIN", domain.replace("https://", "").strip("/"))


def pinokio_dirs() -> list[Path]:
    cands = [Path(r"C:\pinokio\api\yue2-groove-pinokio.git"),
             Path.home() / "pinokio" / "api" / "yue2-groove-pinokio.git"]
    return [c for c in cands if c.is_dir()]


# ─────────────────────────────── Installation ───────────────────────────────

def install_main(g: dict) -> None:
    say("App-Umgebung (Python 3.10) …")
    env = APP / "env"
    u = uv()
    if not py(env).is_file():
        run([u, "venv", env, "--python", "3.10", "--seed"])
    p = py(env)
    # YuE2 als ZIP-Archiv: braucht kein Git (sonst: "Git executable not found").
    # Fällt auf git zurück, falls der ZIP-Download scheitert.
    sources = [os.environ.get("MUSIK_YUE2_URL") or f"{YUE2_GIT}/archive/refs/tags/{YUE2_TAG}.zip"]
    if git():
        sources.append(f"git+{YUE2_GIT}@{YUE2_TAG}")
    for i, src in enumerate(sources):
        r = run([u, "pip", "install", "--python", p, "-e", f"{APP}[gguf]", f"yue2-infer @ {src}",
                 "yt-dlp[default]", "imageio-ffmpeg", "pyngrok",
                 "--overrides", APP / "overrides" / "linux.txt"],
                env={"GIT_TERMINAL_PROMPT": "0"}, check=i == len(sources) - 1)
        if r.returncode == 0:
            break
        warn("Download fehlgeschlagen – versuche anderen Weg …")
    check = run([p, "-c", "from pathlib import Path; import gradio; p=Path(gradio.__file__).parent/"
                 "'templates'/'frontend'/'index.html'; assert p.is_file(), p"], check=False, quiet=True)
    if check.returncode != 0:
        run([u, "pip", "install", "--python", p, "--reinstall-package", "gradio", "gradio>=6,<7"])
    ok("App-Pakete installiert")

    if g.get("name"):
        idx = torch_index(g)
        say(f"PyTorch mit CUDA ({idx}) für {g['name']} …")
        want = f"+{idx}"
        have = out([p, "-c", "import torch;print(torch.__version__)"])
        if want not in have:
            extra = ["--no-deps"] if WIN else []
            run([u, "pip", "install", "--python", p, f"torch=={TORCH_MAIN}",
                 "--index-url", f"https://download.pytorch.org/whl/{idx}", "--force-reinstall", *extra])
        probe = out([p, "-c", "import torch;print(torch.cuda.is_available(), "
                                "torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '', "
                                "torch.cuda.get_arch_list())"])
        print(f"    torch: {probe}")
        cc = g.get("cc")
        if cc and f"sm_{int(round(cc * 10))}" not in probe:
            warn("torch kennt diese Karte nicht direkt – YuE2 nutzt dann den GGUF-Motor (Vulkan).")
        ok("PyTorch bereit")
    else:
        warn("Keine NVIDIA-Grafikkarte gefunden – die App läuft dann nur sehr langsam (CPU).")

    say("YuE2-GGUF-Motor (für Karten unter 16 GB) …")
    r = run([p, "-m", "yue2_groove.gguf_engine", "install", "--tag", "v1.0.3"], cwd=APP, check=False)
    if r.returncode != 0:
        r = run([p, "-m", "yue2_groove.gguf_engine", "install", "--tag", "latest"], cwd=APP, check=False)
    if r.returncode == 0:
        ok("GGUF-Motor installiert")
    else:
        warn("GGUF-Motor nicht installiert – YuE2 läuft dann nur mit 16 GB+ Grafikspeicher")


def install_sheetsage(g: dict) -> None:
    say("Cover-Werkzeug SheetSage2 (Melodie raushören für YuE2-Cover) …")
    u = uv()
    venv = APP / ".venv-sheetsage2"
    if not py(venv).is_file():
        run([u, "venv", venv, "--python", "3.10", "--seed"])
    p = py(venv)
    run([u, "pip", "install", "--python", p, "huggingface-hub==0.36.0"])
    if "+cu126" not in out([p, "-c", "import torch;print(torch.__version__)"]):
        if g.get("name") or not WIN:
            run([u, "pip", "install", "--python", p, "torch==2.8.0", "torchaudio==2.8.0",
                 "--index-url", "https://download.pytorch.org/whl/cu126"])
        else:
            run([u, "pip", "install", "--python", p, "torch==2.8.0", "torchaudio==2.8.0"])
    models = APP / "models" / "SheetSage2"
    if not (models / "config.json").is_file():
        run([p, "-c", "from huggingface_hub import snapshot_download; "
             f"snapshot_download('m-a-p/SheetSage2', local_dir=r'{models}')"])
    run([u, "pip", "install", "--python", p, "-r", models / "requirements.txt"])
    if (g.get("name") or not WIN) and "+cu" not in out([p, "-c", "import torch;print(torch.__version__)"]):
        run([u, "pip", "install", "--python", p, "torch==2.8.0", "torchaudio==2.8.0", "--force-reinstall",
             "--no-deps", "--index-url", "https://download.pytorch.org/whl/cu126"])
    run([p, "-c", "import json; from huggingface_hub import snapshot_download; "
         f"c=json.load(open(r'{models / 'config.json'}', encoding='utf-8')); "
         "snapshot_download(c['base_model_name_or_path'], revision=c['base_model_revision'])"])
    ok("SheetSage2 bereit")


def install_yue_models() -> None:
    say("YuE2-Modelle laden (einmalig ca. 8 GB) …")
    p = py(APP / "env")
    for repo in ("m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"):
        run([p, "-c", f"from huggingface_hub import snapshot_download; print(snapshot_download('{repo}'))"])
    ok("YuE2-Modelle geladen")


def install_ace(g: dict) -> None:
    say("ACE-Step 1.5 (alle Sprachen, schnell) …")
    u = uv()
    if not (ACE / "acestep").is_dir():
        if ACE.exists():
            shutil.rmtree(ACE, ignore_errors=True)
        # ZIP statt git clone: klappt auch ohne Git
        try:
            download_zip(os.environ.get("MUSIK_ACE_ZIP") or f"{ACE_GIT}/archive/{ACE_COMMIT}.zip", ACE)
        except Exception as exc:  # noqa: BLE001
            g_exe = git()
            if not g_exe:
                die(f"ACE-Step konnte nicht geladen werden ({exc}). Internet prüfen und neu starten.")
            warn(f"ZIP-Download fehlgeschlagen ({exc}) – nehme git")
            shutil.rmtree(ACE, ignore_errors=True)
            run([g_exe, "clone", ACE_GIT, ACE])
            run([g_exe, "checkout", "-q", ACE_COMMIT], cwd=ACE)
    p = py(ACE / ".venv")
    stamp = ACE / ".venv" / "musik-machen-commit.txt"
    if not p.is_file() or not stamp.is_file() or stamp.read_text().strip() != ACE_COMMIT:
        run([u, "sync", "--python", "3.11"], cwd=ACE, env={"UV_PROJECT_ENVIRONMENT": str(ACE / ".venv")})
        stamp.write_text(ACE_COMMIT)
    # ältere Karten (z. B. GTX 1070 / Pascal): ACE-Steps eigener Kompatibilitäts-Fix
    probe = run([p, "-c", "import os,sys; sys.path.insert(0, os.getcwd()); "
                 "from acestep.launcher_compat import legacy_torch_fix_probe_exit_code; "
                 "raise SystemExit(legacy_torch_fix_probe_exit_code())"], cwd=ACE, check=False, quiet=True)
    if probe.returncode == 42:
        warn("Ältere NVIDIA-Karte erkannt – installiere passende torch-Version für ACE-Step")
        run([u, "pip", "install", "--python", p, "--force-reinstall", "--index-url",
             "https://download.pytorch.org/whl/cu121", "torch==2.5.1+cu121",
             "torchvision==0.20.1+cu121", "torchaudio==2.5.1+cu121"])
        run([u, "pip", "install", "--python", p, "--force-reinstall", "torchao==0.11.0"], check=False)
    say("ACE-Step-Modelle laden (einmalig) …")
    dl = ACE / ".venv" / ("Scripts/acestep-download.exe" if WIN else "bin/acestep-download")
    if dl.is_file():
        run([dl], cwd=ACE, check=False)
    ok("ACE-Step bereit")


def migrate_songs() -> None:
    runs = APP / "runs"
    if runs.is_dir() and any(runs.iterdir()):
        return
    for old in pinokio_dirs():
        src = old / "app" / "runs"
        if src.is_dir() and any(src.iterdir()):
            say(f"Songs aus der Pinokio-Installation übernehmen ({src}) …")
            shutil.copytree(src, runs, dirs_exist_ok=True)
            ok("Songs übernommen")
            return


def desktop_shortcut() -> None:
    if not WIN:
        return
    ico = ROOT / "tools" / "musik.ico"
    target = ROOT / "starten.bat"
    ps = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut("
        "[Environment]::GetFolderPath('Desktop')+'\\Musik machen.lnk');"
        f"$s.TargetPath='{target}';$s.WorkingDirectory='{ROOT}';"
        + (f"$s.IconLocation='{ico}';" if ico.is_file() else "")
        + "$s.Save()"
    )
    r = run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], check=False, quiet=True)
    if r.returncode == 0:
        ok("Verknüpfung „Musik machen“ auf dem Desktop angelegt")


def windows_prereqs() -> None:
    """Microsoft-VC++-Laufzeit (braucht PyTorch) und Git (nur für Updates)."""
    sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    if not (sys32 / "vcruntime140_1.dll").is_file() and shutil.which("winget"):
        say("Microsoft Visual C++ Laufzeit wird installiert …")
        run(["winget", "install", "--id", "Microsoft.VCRedist.2015+.x64", "-e", "--silent",
             "--accept-package-agreements", "--accept-source-agreements"], check=False)
    ensure_git()
    if len(str(ROOT)) > 60:
        warn(f"Langer Ordnerpfad ({ROOT}). Falls etwas mit 'Dateiname zu lang' scheitert: "
             "Ordner nach C:\\MusikMachen verschieben und neu starten.")


def cmd_install(args) -> None:
    t0 = time.time()
    print("=" * 64)
    print("  Musik machen – Installation (dauert beim ersten Mal 20–60 Minuten)")
    print("=" * 64)
    g = gpu()
    if g["name"]:
        ok(f"Grafikkarte: {g['name']} · {g['vram_gb']} GB · Compute {g['cc']}")
        if g["vram_gb"] and g["vram_gb"] < 7.5:
            warn("Unter 8 GB: YuE2 läuft nicht, ACE-Step im Sparmodus.")
    ensure_settings(interactive=not args.yes)
    if WIN:
        windows_prereqs()
    install_main(g)
    if not args.skip_models:
        install_yue_models()
    if not args.skip_sheetsage:
        install_sheetsage(g)
    if not args.skip_ace:
        install_ace(g)
    migrate_songs()
    desktop_shortcut()
    mins = (time.time() - t0) / 60
    say(f"Fertig in {mins:.0f} Minuten. Starten mit „starten.bat“ oder dem Desktop-Symbol.")


# ─────────────────────────────── Start ───────────────────────────────

def cmd_start(args) -> None:
    cfg = read_settings()
    port = int(args.port or cfg.get("MUSIK_PORT") or DEFAULT_PORT)
    host = args.host or cfg.get("MUSIK_HOST") or "127.0.0.1"
    p = py(APP / "env")
    if not p.is_file():
        die("Noch nicht installiert – bitte zuerst installieren.bat ausführen.")
    env = {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "YUE2_GROOVE_VIEW": "song",
        "YUE2_GROOVE_HOST": host,
        "YUE2_GROOVE_MODELS": str(APP / "models"),
        "YUE2_GROOVE_RUNS": str(APP / "runs"),
        "ACESTEP_DIR": str(ACE),
    }
    ss = py(APP / ".venv-sheetsage2")
    cmd = [p, "-m", "yue2_groove", "--host", host, "--port", str(port), "--no-preload"]
    if ss.is_file():
        cmd += ["--sheetsage-python", ss]
    url = f"http://127.0.0.1:{port}/"
    if not args.no_browser:
        def _open():
            for _ in range(600):
                try:
                    urllib.request.urlopen(url, timeout=2)  # noqa: S310
                    webbrowser.open(url)
                    return
                except Exception:  # noqa: BLE001
                    time.sleep(1)
        import threading
        threading.Thread(target=_open, daemon=True).start()
    print("=" * 64)
    print(f"  Musik machen läuft gleich unter {url}")
    print("  Dieses Fenster offen lassen. Beenden: Fenster schließen oder Strg+C.")
    print("=" * 64, flush=True)
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(APP), env={**os.environ, **env})
    try:
        sys.exit(proc.wait())
    except KeyboardInterrupt:
        proc.terminate()


# ─────────────────────────────── Update ───────────────────────────────

def cmd_update(args) -> None:
    g_exe = ensure_git()
    if not g_exe:
        die("Git fehlt. Neuen Stand als ZIP von GitHub laden und über diesen Ordner entpacken.")
    if not (ROOT / ".git").is_dir():
        if "OWNER" in REPO_URL:
            die("Repo-Adresse unbekannt.")
        say("Ordner mit GitHub verbinden …")
        run([g_exe, "init", "-q"], cwd=ROOT)
        run([g_exe, "remote", "add", "origin", REPO_URL], cwd=ROOT, check=False)
        run([g_exe, "fetch", "origin", "main"], cwd=ROOT)
        run([g_exe, "reset", "--hard", "origin/main"], cwd=ROOT)
        run([g_exe, "branch", "--set-upstream-to=origin/main"], cwd=ROOT, check=False)
    else:
        say("Neuen Code holen …")
        run([g_exe, "pull", "--ff-only"], cwd=ROOT)
    args.skip_models = args.skip_sheetsage = False
    args.skip_ace = False
    args.yes = True
    cmd_install(args)


# ─────────────────────────────── Backup ───────────────────────────────

def cmd_backup(args) -> None:
    runs = APP / "runs"
    if not runs.is_dir():
        die("Keine Songs gefunden.")
    BACKUPS.mkdir(exist_ok=True)
    name = BACKUPS / f"Songs-{_dt.datetime.now():%Y-%m-%d_%H-%M}.zip"
    say(f"Songs sichern nach {name} …")
    n = 0
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in runs.rglob("*"):
            if f.is_file() and f.suffix.lower() not in (".npy", ".pt"):
                zf.write(f, f.relative_to(runs))
                n += 1
    ok(f"{n} Dateien gesichert ({name.stat().st_size / 1e6:.0f} MB)")
    print("    Tipp: die ZIP-Datei in Google Drive / OneDrive legen – dann ist sie sicher.")
    if WIN:
        os.startfile(BACKUPS)  # noqa: S606


def cmd_restore(args) -> None:
    zips = [Path(args.zip)] if args.zip else sorted(BACKUPS.glob("Songs-*.zip"))[-1:]
    if not zips or not zips[0].is_file():
        die("Keine Backup-ZIP gefunden (Backups-Ordner leer). ZIP-Datei auf songs-zurueckholen.bat ziehen.")
    runs = APP / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    say(f"Songs zurückholen aus {zips[0]} …")
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(runs)
    ok("Fertig – beim nächsten Start sind die Songs da.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Musik machen – Installer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install")
    i.add_argument("--yes", action="store_true", help="keine Fragen stellen")
    i.add_argument("--skip-models", action="store_true", help="YuE2-Modelle erst beim ersten Song laden")
    i.add_argument("--skip-sheetsage", action="store_true")
    i.add_argument("--skip-ace", action="store_true")
    s = sub.add_parser("start")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--no-browser", action="store_true")
    sub.add_parser("update")
    sub.add_parser("backup")
    r = sub.add_parser("restore")
    r.add_argument("zip", nargs="?")
    args = ap.parse_args()
    {"install": cmd_install, "start": cmd_start, "update": cmd_update,
     "backup": cmd_backup, "restore": cmd_restore}[args.cmd](args)


if __name__ == "__main__":
    main()
