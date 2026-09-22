# 🎵 Musik machen

Eigener „Suno“ zum Selbst-Hosten: Songs aus einer Beschreibung, Cover von YouTube/Drive/Datei,
Vorsingen, Bearbeiten (Kürzen, Verlängern, Stems, Remaster …), Handy-App.
Zwei KI-Modelle: **ACE-Step 1.5** (alle Sprachen, schnell) und **YuE2** (Englisch, lange Songs).

Kein Pinokio nötig.

---

## Auf einem neuen Windows-PC installieren

**Voraussetzung:** NVIDIA-Grafikkarte mit mind. 8 GB (z. B. GTX 1070, RTX 2070, RTX 3060 …),
aktueller NVIDIA-Treiber, ca. 60 GB freier Speicher.

1. Auf GitHub oben rechts **Code → Download ZIP**, entpacken, z. B. nach `C:\MusikMachen`
   (kurzer Pfad ohne Leerzeichen ist am sichersten).
2. **`installieren.bat`** doppelklicken. Läuft beim ersten Mal 20–60 Minuten.
   Fragt nur einmal nach dem ngrok-Token (Enter = überspringen).
3. **`starten.bat`** doppelklicken (oder das Desktop-Symbol „Musik machen“). Der Browser öffnet sich.

Das war's. Auf demselben PC mit alter Pinokio-Installation werden Songs und
Einstellungen automatisch übernommen.

| Datei | Wofür |
|---|---|
| `installieren.bat` | Alles installieren. Kann jederzeit erneut gestartet werden (repariert). |
| `starten.bat` | App starten. Fenster offen lassen, solange die App laufen soll. |
| `aktualisieren.bat` | Neusten Stand von GitHub holen (fragt beim ersten Mal nach dem GitHub-Login). |
| `songs-sichern.bat` | Alle Songs als ZIP in den Ordner `Backups` – am besten in Google Drive legen. |
| `songs-zurueckholen.bat` | Backup-ZIP drauf ziehen → Songs sind wieder da. |
| `einstellungen.txt` | ngrok-Token, Passwort usw. (wird beim Installieren angelegt, nie hochgeladen). |

## Umzug auf einen anderen PC

1. Alter PC: `songs-sichern.bat` → ZIP in Google Drive.
2. Neuer PC: installieren (oben), `einstellungen.txt` vom alten PC rüberkopieren
   (oder beim Installieren den ngrok-Token neu eingeben).
3. ZIP auf `songs-zurueckholen.bat` ziehen → `starten.bat`.

## Handy / ngrok

Der Server läuft zu Hause auf dem PC. ngrok baut einen Tunnel ins Internet, damit das Handy
(App oder Browser) den PC erreicht – ohne Router-Einstellungen.

- Der **ngrok-Token** in `einstellungen.txt` verbindet den PC mit deinem ngrok-Konto.
- Zu jedem kostenlosen ngrok-Konto gehört **eine feste Adresse** (z. B. `…ngrok-free.dev`).
  Welcher PC den Token benutzt, ist egal: **gleicher Token = gleiche Adresse**. Die Handy-App
  muss nach einem Umzug also nicht geändert werden.
- Es kann immer nur **ein PC gleichzeitig** diese Adresse nutzen. Alten PC vorher ausschalten
  (oder dort die App beenden), sonst meldet ngrok beim neuen PC einen Fehler.
- Token/Adresse: https://dashboard.ngrok.com (Your Authtoken / Domains).

## Grafikkarten

Die Installation erkennt die Karte und stellt alles passend ein:

| Karte | ACE-Step (Deutsch & alle Sprachen) | YuE2 (Englisch, lang) |
|---|---|---|
| 16 GB+ (RTX 4080, 5080 …) | ✅ schnell | ✅ volle Länge |
| 12 GB (RTX 3060 12G, 4070 …) | ✅ | ✅ über GGUF-Motor |
| 8 GB RTX 20/30/40 (2070, 3070, 4060 …) | ✅ Sparmodus | ✅ GGUF-Motor, Songs bis ca. 4:50 |
| 8 GB **GTX 10xx** (1070, 1080) | ✅ Sparmodus, langsamer | ⚠️ GGUF-Motor über Vulkan – ungetestet, langsam |
| unter 8 GB | ⚠️ nur ACE-Step ohne Text-KI | ❌ |

GTX-10-Karten (Pascal) bekommen automatisch passende ältere CUDA-Pakete
(App: CUDA 12.6, ACE-Step: torch 2.5.1/CUDA 12.1). Der YuE2-GGUF-Motor ist nur für RTX-Karten
mit CUDA gebaut; auf GTX-Karten weicht er automatisch auf Vulkan aus
(`YUE2_GROOVE_GGUF_DEVICE=auto`). Für Deutsch ist ACE-Step ohnehin die bessere Wahl.

## Ohne eigenen PC: Cloud-GPU (Linux / Docker)

```bash
docker compose -f docker/docker-compose.yml up -d --build
# → http://<server>:42003
```

Songs, Modelle und `einstellungen.txt` liegen im Docker-Volume `musik-daten`.
**Wichtig:** Auf einem Cloud-Server ist die App direkt im Internet → in `einstellungen.txt`
ein Passwort setzen (`YUE2_GROOVE_AUTH=name:passwort`).

Linux ohne Docker: `./installieren.sh` und `./starten.sh`.

## Aufbau

```
app/            die App (Python-Paket yue2_groove, Oberfläche in yue2_groove/webui/simple_ui.py)
  runs/         deine Songs (nicht im Repo)
ace-step/       ACE-Step 1.5 (wird installiert, nicht im Repo)
android/        Handy-App: Quellcode, fertige APK und Signatur-Schlüssel (für App-Updates)
docker/         Docker für Cloud-GPUs
tools/          installer.py – macht Installation, Start, Update, Backup
```

## Herkunft & Lizenzen

- App-Grundlage: [deadjoe/yue2_groove](https://github.com/deadjoe/yue2_groove) (Apache-2.0,
  siehe `app/LICENSE` und `app/NOTICE`), stark umgebaut (einfache Oberfläche, ACE-Step,
  Cover-Links, Bearbeiten, Handy).
- YuE2-GGUF-Motor: Programmdateien kommen aus den Releases von deadjoe/yue2_groove.
- Modelle: YuE2 / SheetSage2 / MERT sind **CC BY-NC 4.0** (nicht kommerziell),
  ACE-Step 1.5 ist MIT.
- Covers mit Originaltext nur privat nutzen – öffentlich teilen kann Urheberrechte verletzen.
