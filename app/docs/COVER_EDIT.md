# Cover and edit workflows (02 // COVER, 03 // EDIT)

This is the practical manual for the two tabs added on top of the generation flow:
**audio → ABC → cover** (SheetSage2) and **freeze → edit → check → regenerate → listen**.
It also lists the manual end-to-end checks (`C1`, `C2`, `E1`, `X`) and what hardware they need.

The UI runs both models in **separate Python environments** because their dependency pins
conflict; they exchange files, not imports. Nothing here changes the upstream YuE2 clone.

## 1. Two environments

| Role | Environment | What the UI needs to know |
|---|---|---|
| Song generation | the groove venv (`yue2-groove` + `yue2-infer`), see the main README | nothing new |
| Audio → ABC | a **separate SheetSage2 venv** (Python 3.10/3.11, its own torch/transformers) | `YUE2_GROOVE_SHEETSAGE_PYTHON=/path/to/.venv-sheetsage2/bin/python` |

Set up SheetSage2 next to a YuE checkout (adjust paths). FFmpeg 6.1+ must be on `PATH`:

```bash
cd /path/to/YuE

# Linux / NVIDIA CUDA (the configuration SheetSage2 validates)
python3.11 -m venv .venv-sheetsage2
.venv-sheetsage2/bin/python -m pip install huggingface-hub==0.36.0
.venv-sheetsage2/bin/hf download m-a-p/SheetSage2 --local-dir models/SheetSage2
.venv-sheetsage2/bin/python -m pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
.venv-sheetsage2/bin/python -m pip install -r models/SheetSage2/requirements.txt
```

macOS / Apple Silicon: the same commands without the CUDA index
(`pip install torch==2.8.0 torchaudio==2.8.0`) and `device=mps`; this path is untested and
CPU (`device=cpu`) always works but is slow. The model card is the authority on versions.

Loading SheetSage2 automatically downloads/loads its parent encoder
**MERT-v2-FullSong** — do not install or select it manually.

Point the UI at the environment (`.env` in the groove repository root, so `serve.sh`
picks it up; env vars and the `--sheetsage-python` flag are equivalent):

```bash
YUE2_GROOVE_SHEETSAGE_PYTHON=/path/to/YuE/.venv-sheetsage2/bin/python
# optional:
YUE2_GROOVE_SHEETSAGE_MODEL=/path/to/YuE/models/SheetSage2   # defaults to m-a-p/SheetSage2
YUE2_GROOVE_SHEETSAGE_BASE_MODEL=/path/to/MERT-v2-FullSong  # parent encoder snapshot, offline loads
YUE2_GROOVE_SHEETSAGE_DEVICE=auto                            # auto | cuda | mps | cpu
YUE2_GROOVE_SHEETSAGE_KEEP_WARM=1                            # reuse one resident worker (faster repeats)
YUE2_GROOVE_SHEETSAGE_IDLE_SECONDS=900                       # resident worker idle lifetime
YUE2_GROOVE_TRANSCRIPTIONS=/path/to/transcriptions           # defaults to <runs>/transcriptions
```

`YUE2_GROOVE_SHEETSAGE_BASE_MODEL` is passed to the model loader as `base_model_path`, which is
what makes `OFFLINE` work against a locally downloaded MERT-v2-FullSong snapshot instead of the
Hub cache. `YUE2_GROOVE_TRANSCRIPTIONS` wins over the default `<runs>/transcriptions` when set;
the 02 COVER UI writes every transcription into a fresh timestamped directory there. By default
every transcription is a fresh subprocess (dependencies isolated, all memory returned on exit);
`KEEP SHEETSAGE2 WARM` keeps one resident worker whose loaded model is reused — much faster
repeats, at the cost of holding the weights in memory. UNLOAD SHEETSAGE2 frees it immediately;
a background reaper also stops it once the idle timeout passes (no need to wait for another
run), and CANCEL terminates a busy worker (the following run starts a fresh one).

The **CHECK ENVIRONMENT** button in 02 COVER probes that interpreter (`torch`,
`transformers`, ffmpeg) without loading weights, and **AUTO-DETECT VENV** looks for
`./.venv-sheetsage2` / `../YuE/.venv-sheetsage2` and uses it for the session. If SheetSage2 is not configured, the
tab reports a configuration error and every other tab keeps working.

## 2. 02 COVER — audio to a cover song

1. **SOURCE AUDIO** — upload a reference recording (wav/mp3/flac; decoded to mono 24 kHz).
   Already have a score? SKIP the audio: **SOURCE WORK → REFRESH → LOAD ABC** fills COVER ABC
   (and STYLE/LYRICS when the source has them) from any saved work or transcription, so a
   transcription survives a page reload; **SEND TO EDIT** hands the current score to 03 EDIT with
   that source as the freeze target.
2. **TRANSCRIPTION TASK**
   - `MELODY // VOCAL` — only the sung line; best when the cover should keep the vocal melody;
   - `MELODY // VOCAL+INST` — vocal and instrumental melodies, chord-free (the default);
   - `FULL SCORE // + CHORDS` — adds chord symbols; use this to regenerate with `cot=full`.
3. **TRANSCRIBE** — runs the driver in the SheetSage2 venv. The status pane shows the output
   directory under `runs/transcriptions/…`, the device/dtype, and any model warnings. CANCEL
   terminates the subprocess. The tab stays usable for a new attempt.
4. **Review the ABC** in the editor (and the rendered score below it). Transcription can contain
   musical errors; fix them before covering. `STRIP CHORDS` re-serializes the score without
   chord symbols and verifies that melody, meter and tempo are unchanged.
5. **SEND TO GENERATE** — fills `01 GENERATE` with the score and sets PLAN MODE to `MELODY`
   (chord-free input) or `FULL` (full-score input), then switches to that tab. Add the target
   STYLE and LYRICS there and press GENERATE. The existing `adapter.generate`/`YuE2Pipeline`
   path is used unchanged; artifacts land in `runs/<stamp>-<id>/`.
6. **GENERATE COVER** (optional, in the `GENERATE COVER // direct from this score` accordion) —
   the same score-conditioned request without leaving 02 COVER: type the target STYLE/LYRICS
   there, press the button, and the result appears with its own RESULT audio, RESULT ABC (the
   score actually submitted — chord-stripped for melody tasks) and GENERATED FILES. Both routes
   run the identical `cover.build_cover_request` + generation core.
7. **SEND TO EDIT** — pushes the score shown in COVER ABC into 03 EDIT and selects the source
   work in the SOURCE WORK dropdown (LOAD a work first, or transcribe; EDIT needs that source to
   freeze the baseline). EDIT then requires FREEZE BASELINE + CHECK INVARIANTS as usual.

While a transcription runs, every action button on the tab (including STRIP CHORDS and the two
GENERATE buttons) is disabled; CANCEL stays active and terminates the subprocess.

`cot="melody"` does not remove chords by itself — both generation routes strip them for you. A
cover supplies a symbolic melody condition; it does not preserve the source singer's identity or
waveform.

## 3. 03 EDIT — iterate on a score

The flow follows the upstream skill: a frozen baseline, an explicit edit contract, an exact
symbolic invariant check, regeneration from the edited score, and a listening comparison.

1. **SOURCE WORK → REFRESH → LOAD** — pick a saved work that contains `score.abc` (a generated
   plan, a full song, or an earlier edit). Its style/lyrics/ABC load into the editor and the
   ABC becomes the baseline.
2. **FREEZE BASELINE** — writes `runs/baselines/<stamp>-<name>-baseline/` with `baseline.json`
   (SHA-256 of the source artifacts) and copies of `score.abc` / `request.json`. The original run
   directory is never touched; audio and latents stay where they are and remain the listening
   baseline. **CHECK INVARIANTS and GENERATE EDITED require the frozen record** — this is what
   ties a verification result to an immutable source. `ALLOW MELODY/RHYTHM CHANGES` only lets a
   *failing* check through for intentional adaptations — FREEZE and CHECK are always required.
3. **Edit** the ABC (and style/lyrics if the arrangement changes). Keep the native dialect:
   the checker rejects unsupported notation rather than guessing.
4. **CHECK INVARIANTS** — compares baseline and edit per voice under the chosen **CONTRACT**:
   `EXACT` (notes + meter grid; `ALLOW TEMPO CHANGE` / `ALLOW METER CHANGE` relax those),
   `PITCH` (ordered pitch sequence only, rhythm may change) or `FREE` (differences are
   recorded, nothing is gated). Chord-only edits pass under EXACT.
5. **GENERATE EDITED** — refused unless the check passed on *exactly* the current ABC, and the
   edited score is always submitted (an edit can never silently turn into a fresh plan). For
   intentional melody/rhythm changes, tick `ALLOW MELODY/RHYTHM CHANGES`. Every attempt is a new
   run directory with `edit_manifest.json`: source and edit hashes, the invariant result,
   permitted changes, request fields, and whether the baseline was frozen. The score that was
   actually generated appears in a separate **RESULT ABC** box (with its own score view) — the
   `EDITED ABC` editor is never overwritten, so the next CHECK INVARIANTS still compares what you
   wrote. Sampling parameters are the shared sliders of `01 GENERATE → ADVANCED // SAMPLING`; the
   EDIT tab shows a read-only mirror and never keeps a second copy. While a generation runs, the
   tab's action buttons (FREEZE / LOAD / CHECK / COMPARISON / GENERATE) are disabled; CANCEL
   remains active.
6. **BUILD COMPARISON // baseline vs edit** — builds a local listening page from the original
   run and the new one (same player as 05 TOOLS → LISTENING COMPARISON). Listen to the whole
   song and to the changed passage; the symbolic check is not listening evidence. **OPEN IN
   LIBRARY** shows the finished attempt, and 04 LIBRARY can send any work back with **OPEN IN
   03 EDIT** / **USE IN 02 COVER**.

The check is symbolic: passing it does not guarantee the audio realizes the score. Treat
transcription, symbolic checks and listening as three separate pieces of evidence.

## 4. Manual end-to-end checks

Use one 30–60 s vocal reference for the cover cases and an existing generated work for the edit
case. `runs/` below means `YUE2_GROOVE_RUNS` (default `./runs`).

### C1 — cover from a recording

1. `bash scripts/serve.sh start` with `YUE2_GROOVE_SHEETSAGE_PYTHON` set.
2. 02 COVER → upload the reference → task `MELODY // VOCAL+INST` → TRANSCRIBE.
   *Expect:* status lists `runs/transcriptions/<stamp>-<name>`; `score.abc`, `events.json`,
   melody MIDI and `result.json` are written; the ABC editor and score view fill in; no chord
   symbols in the ABC.
3. STRIP CHORDS (no-op on an already chord-free score) → SEND TO GENERATE.
   *Expect:* 01 GENERATE shows the ABC, PLAN MODE=MELODY.
4. Type target STYLE/LYRICS → GENERATE.
   *Expect:* `runs/<stamp>-<id>/` contains `audio.flac`, `score.abc`, `request.json`,
   `config.json`, `result.json`, `latent.npy`, `semantic.npy`, `local_env.json`; 04 LIBRARY plays it.

### C2 — full-score conditioning

1. 02 COVER → task `FULL SCORE // + CHORDS` → TRANSCRIBE → SEND TO GENERATE.
   *Expect:* PLAN MODE=FULL and the chord symbols still present.
2. Generate and confirm `request.json` in the run directory records `"cot": "full"`.

### E1 — edit iteration

1. 03 EDIT → REFRESH → pick a work with audio → LOAD → FREEZE BASELINE.
   *Expect:* `runs/baselines/<stamp>-<name>-baseline/baseline.json` plus copies of `score.abc`
   and `request.json`; the source directory's files are unchanged.
2. Reharmonize some chords (keep every melody note) → CHECK INVARIANTS.
   *Expect:* `"match": true` and unchanged note lists in the JSON report.
3. Change one melody pitch, press CHECK INVARIANTS, then GENERATE EDITED.
   *Expect:* match is false and generation is refused with the difference listed; enabling
   `ALLOW MELODY/RHYTHM CHANGES` and pressing again proceeds.
4. With the check green, GENERATE EDITED.
   *Expect:* a new `runs/<stamp>-edit-<name>/` with audio plus `edit_manifest.json` containing
   the baseline/edit hashes and the invariant result.
5. BUILD COMPARISON → open the link → listen to baseline vs edit.
6. Clear the EDITED ABC and press GENERATE EDITED.
   *Expect:* refusal — an empty edit cannot silently become a fresh plan.

### X — failures stay understandable

- No `YUE2_GROOVE_SHEETSAGE_PYTHON`: 02 COVER reports the missing configuration; the UI does not
  crash. A bad interpreter path / missing torch or transformers is reported the same way.
- CANCEL during transcription terminates the SheetSage2 subprocess and re-enables the button;
  CANCEL during edit generation follows the existing cooperative-cancel path.
- A transcription that yields no usable ABC shows the driver's error and warnings
  (`failure.json` is written in the output directory).

## 5. Hardware and known limits

**Verified during development** (macOS, Apple M4 Pro, 69 GB; YuE2 venv torch 2.14.0, MPS
bfloat16; SheetSage2 venv Python 3.11, torch 2.8.0, transformers 4.45.2):

- **E1** was exercised with the locally cached YuE2 weights — the chord-only edit passed the
  invariant check, generation from the edited ABC completed in 27 s at 16 ODE steps / 512
  semantic tokens (20.5 s of 48 kHz stereo `audio.flac`), and the run directory contained every
  artifact plus `edit_manifest.json` with the differing source/edit hashes, the invariant result
  and the baseline pointer.
- **C1** was exercised end-to-end through the running UI: a 30 s vocal excerpt was transcribed
  by SheetSage2 on MPS (`melody-vocal`, `melody_only=True`) into a chord-free ABC (65 sounding
  notes, 32 s, Fm / 105 BPM), SEND TO GENERATE set `cot=melody`, and YuE2 produced a complete
  20.5 s 48 kHz stereo run whose `request.json` carries the transcribed melody. The same setup
  was probed with CHECK ENVIRONMENT and a second transcription (10 s) confirmed repeatability.
  A fully offline transcription (`offline=True` plus the local MERT-v2-FullSong snapshot wired
  through `base_model_path`) also completed in 10 s on MPS.
- **ALL MODES** (see the README section) completed `full` / `melody` / `off` back to back on
  the same loaded pipeline and produced the three run directories plus the automatic listening
  bundle (`run.json`, per-mode `input.json`, comparison `index.html` + `manifest.json`).

C2 differs from C1 only in the transcription task and plan mode; the commands above are
sufficient. SheetSage2 is CUDA-validated upstream; the MPS numbers here are this machine's, not
a support guarantee.

- SheetSage2 runs in its own process. By default **one subprocess per transcription**: the
  interpreter, remote code and MERT-v2-FullSong parent are loaded fresh and all memory is
  returned when the process exits (dependency isolation; nothing left reserved when YuE2
  loads). `KEEP SHEETSAGE2 WARM` switches to **one resident worker** whose model is reused
  until UNLOAD / the idle timeout; a background reaper stops it once idle, and CANCEL or a
  crash kills it immediately (the next run restarts it).
  Measured on this machine (MPS fp32, 30 s excerpt, warm OS cache): 9.7 s cold vs 6.0 s warm
  on the second run — the saving grows when the weights are not in the page cache.
- YuE2 generation: the upstream baseline (24 GB NVIDIA, BF16) or Apple Silicon with the
  documented overrides (see the main README and `docs/MACOS_MPS.md`). One model at a time is
  the supported configuration; each transcription is a fresh process, so SheetSage2's weights
  are loaded per call (the Hugging Face cache is shared, so disk is not duplicated).
- SheetSage2: CUDA is the validated path; `mps`/`cpu` are offered but untested. Long inputs take
  longer and use more memory; use MAX SECONDS only when cropping is intended.
- Model weights (YuE2, SheetSage2, MERT-v2-FullSong) are **CC BY-NC 4.0 (non-commercial)**.
  This repository ships no weights and is not affiliated with the model authors.
- Transcription accuracy, symbolic score checks and listening quality are separate claims; do
  not report a passing check as proof that the generated audio realized the score.
