# YUE2 // GROOVE

<p align="center">
  <a href="https://pinokio.co/apps/github-com-deadjoe-yue2-groove-pinokio"><img src="https://img.shields.io/badge/one--click-Pinokio-F4A261" alt="Install with Pinokio"></a>
  <a href="https://github.com/deadjoe/yue2-groove-pinokio"><img src="https://img.shields.io/badge/launcher-yue2--groove--pinokio-7C3AED?logo=github&logoColor=white" alt="Pinokio launcher repo"></a>
  <a href="https://github.com/multimodal-art-projection/YuE"><img src="https://img.shields.io/badge/model-Yue2-0A9396" alt="Yue2 model"></a>
</p>

YUE2 // GROOVE is the latest music studio built on the open-source Yue2 model and its inference stack. Generate high-quality full songs from style and lyrics with an editable score plan — powered by the latest YuE model — cover from audio with SheetSage2 and MERT2, refine and compare edits, and keep your works in a reusable, easy-to-manage library. You get high-quality creation with real creative control.

Two views, one kernel. **SONG** (the default) is the producer-facing director: one work at a time, its stage
(`DRAFT → SCORE → AUDIO → REVISE → DONE`), a read-only score, playback, and the next actions — no knobs.
**STUDIO** is the full 7-tab gear room. Switch anytime (top right); your work stays intact.

- **SONG** — start cards for NEW SONG / COVER A RECORDING / EDIT A WORK; then stage + next actions
  (`LISTEN` / `EDIT WORK` / `TRY ANOTHER SEED` / `OPEN IN LIBRARY`, …)
- **01 // GENERATE** — style + lyrics (+ optional ABC) → editable plan → full song (plan mode, seed, CFG, sampling)
- **02 // COVER** — recording → SheetSage2 / MERT2 transcription → polish ABC → generate a cover
- **03 // EDIT** — freeze baseline → edit score → check invariants → generate edited → compare
- **04 // LIBRARY** — all works: play, rename, open in EDIT / COVER, durable run history
- **05 // TOOLS** · **06 // DECODE** · **07 // BATCH** — utilities, re-decode, batch jobs
- **Settings rail** — device, dtype, backend (PyTorch or the optional GGUF engine), memory budget, ODE steps, models

The UI is a thin layer over the `yue2` package: it never modifies upstream code, and the
single file that imports `yue2` (`yue2_groove/adapter.py`) is covered by contract tests
that fail loudly when an upstream release changes something the UI depends on.

> **Platforms.** macOS / Apple Silicon (developed here), Linux + NVIDIA (validated end to
> end), Windows + NVIDIA (best effort, through the GGUF engine), Docker and cloud GPUs — the
> table under [Requirements](#requirements) says what each needs and how far it was tested.

## Screenshots

**SONG** — empty start (NEW SONG / COVER / EDIT):

<img src="docs/images/song-start-dark.png" alt="SONG view: start cards for new song, cover, and edit" width="100%">

**SONG** — after generate (audio + score):

<img src="docs/images/song-audio-dark.png" alt="SONG view: Grand_Piano_CFG15 with waveform and score" width="100%">

**02 // COVER** — transcribe a recording with SheetSage2:

<img src="docs/images/cover-dark.png" alt="02 COVER: source audio upload and transcription tasks" width="100%">

**03 // EDIT** — freeze a baseline and revise the score:

<img src="docs/images/edit-dark.png" alt="03 EDIT: freeze baseline and style fields" width="100%">

**04 // LIBRARY** — works, player, and run tables:

<img src="docs/images/library-dark.png" alt="04 LIBRARY: work list, player, request and sampling tables" width="100%">

## Requirements

Python 3.10 or newer, [uv](https://docs.astral.sh/uv/) (`pip` also works, see below), about
8 GB of disk for the model weights (`m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae`, downloaded from Hugging
Face on first use; licensed **CC BY-NC 4.0**, non-commercial, by the YuE2 project) — and one of
these machines. The app has two engines: **PyTorch**, upstream's reference configuration, and the
optional **GGUF engine** (the same model, 8-bit, through yue2.cpp) for cards the reference does not
fit; BACKEND=auto picks per machine as the table says.

| Machine | Memory | Engine the app picks | How far it was tested | Read |
|---|---|---|---|---|
| **macOS, Apple Silicon** | 32 GB+ unified memory (developed on 64 GB) | PyTorch on MPS | Developed and tested here, every release | [docs/MACOS_MPS.md](docs/MACOS_MPS.md) |
| **Linux + NVIDIA, 16 GB or more** | 24 GB is upstream's validated configuration; 16 GB runs everything the app can produce | PyTorch on CUDA | Validated end to end — generation, Cover, memory budgets, reproducibility — in two sessions on L4 24 GB cards | [docs/LINUX_CUDA.md](docs/LINUX_CUDA.md) |
| **Linux + NVIDIA, 8 – 15 GB** | 12 GB: full-length songs at any CFG; 8 GB: songs up to ~5 min (automatic context cap) | GGUF engine, once its binaries are installed | Run through the app on a 16 GB card (RTX A4000) and on an 8 GB card (RTX 2070, under Windows); no physical 12 GB card yet | [docs/GGUF_ENGINE.md](docs/GGUF_ENGINE.md), [docs/LOW_VRAM.md](docs/LOW_VRAM.md) |
| **Windows + NVIDIA** | as Linux | GGUF engine under 16 GB (PyTorch's Windows build has no FlashAttention; the GGUF engine brings its own); PyTorch, slower, on bigger cards | Best effort: install, Cover and one full-length song (RTX 2070 8 GB) verified through the Pinokio launcher | [docs/LOW_VRAM.md §4](docs/LOW_VRAM.md) |
| **Docker** — Linux + NVIDIA, or Windows through Docker Desktop / WSL2 | as Linux | Both engines are in the image | The image is the Linux environment above, prebuilt; it is what the RunPod launcher runs | [deploy/docker/README.md](deploy/docker/README.md) |
| **Cloud GPU** (RunPod, …) | 16 GB is enough | as Linux | The Linux validation ran on rented L4 and A4000 pods | same image |

Not supported: Intel Macs, machines without a GPU (a song takes hours on a CPU), AMD / Intel GPUs
(the GGUF engine's Vulkan build exists but has not been run through the app).

*Optional, for covers:* a separate SheetSage2 environment (Python 3.10/3.11, its own
torch/transformers, FFmpeg 6.1+) — see [Cover from audio](#cover-from-audio-sheetsage2).
Everything except 02 COVER works without it.


## Run in Pinokio (easiest)

Prefer not to set up Python by hand? Use the one-click launcher:

**→ [YUE2 // GROOVE on Pinokio](https://pinokio.co/apps/github-com-deadjoe-yue2-groove-pinokio)**  
**→ Launcher repo: [deadjoe/yue2-groove-pinokio](https://github.com/deadjoe/yue2-groove-pinokio)**

<img src="docs/images/pinokio-app-page.png" alt="YUE2 // GROOVE on Pinokio Explore — Install" width="100%">

1. Install **[Pinokio Desktop](https://pinokio.computer)**
2. Open the app page above (or **Explore** → search `YUE2 // GROOVE`) → **Install** → **Start**
3. Alternate: Discover → **Download from URL** → paste `https://github.com/deadjoe/yue2-groove-pinokio`

That path installs YuE2 + Cover (SheetSage2 / MERT2) for you. The manual steps below remain for developers who want a local clone.

## Quick start

1. **Install uv** (skip if you have it):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone YuE2 and this repository side by side:**

   ```bash
   git clone https://github.com/multimodal-art-projection/YuE.git
   git clone https://github.com/deadjoe/yue2_groove.git
   cd yue2_groove
   ```

   Already have a YuE2 clone somewhere else? Skip the first line and use its path
   wherever `../YuE` appears below.

3. **Create a virtual environment and install both packages into it.**
   Pick the override file for your platform (it pins two dependency versions; see
   [Why the override file?](#why-the-override-file)):

   ```bash
   uv venv .venv --python 3.12

   # macOS / Apple Silicon
   uv pip install --python .venv/bin/python ../YuE -e . --overrides overrides/macos.txt

   # Linux / NVIDIA CUDA
   uv pip install --python .venv/bin/python ../YuE -e . --overrides overrides/linux.txt
   ```

   Prefer not to clone YuE2 at all? The `yue2` extra pulls the pinned upstream tag
   straight from GitHub: `uv pip install --python .venv/bin/python -e ".[yue2]" --overrides overrides/macos.txt`
   (CI pins the same revision explicitly, and also runs the suite against upstream `main`).

4. **(macOS) Check the attention kernel** — takes a few seconds, loads no weights:

   ```bash
   .venv/bin/python scripts/mps_sdpa_check.py      # expect: "defect not reproduced"
   ```

5. **Start the service:**

   ```bash
   bash scripts/serve.sh start
   ```

   The script prints the local and LAN URLs (default `http://127.0.0.1:7860/`). The model
   loads in the background after the page is up; the first start also downloads the
   weights (about 8 GB) to your Hugging Face cache, so the first generation waits for that.

   Already have the weights on disk (for example in `../YuE/models/YuE2-3B` and
   `../YuE/models/YuE2-Vae`)? Create `.env` **before** the first start so nothing is
   downloaded again:

   ```bash
   echo "YUE2_GROOVE_MODELS=/absolute/path/to/YuE/models" > .env
   ```

6. **Stop, restart, inspect:**

   ```bash
   bash scripts/serve.sh status
   bash scripts/serve.sh log          # follow the log
   bash scripts/serve.sh restart --port 7861
   bash scripts/serve.sh stop
   ```

Generated works land in `runs/` inside this repository (one directory per song with
`audio.flac`, `score.abc`, `request.json`, `config.json`, `result.json`, `latent.npy`,
`semantic.npy` — the same artifacts the official CLI writes, so they stay compatible with
upstream's tools).

## Configuration

Create a `.env` file in the repository root (ignored by git). Both `scripts/serve.sh`
and the app itself read it, so the settings apply however you launch:

```bash
YUE2_GROOVE_AUTH=alice:my-secret     # login for the web UI; set this before exposing it on a LAN
YUE2_GROOVE_PORT=7860
YUE2_GROOVE_HOST=0.0.0.0             # 127.0.0.1 keeps it local-only
YUE2_GROOVE_RUNS=/path/to/runs       # where works are stored (default: ./runs)
YUE2_GROOVE_MODEL=m-a-p/YuE2-3B      # Hugging Face id or a local directory
YUE2_GROOVE_VAE=m-a-p/YuE2-Vae
YUE2_GROOVE_VAE_LEGACY=m-a-p/YuE2-Vae-legacy
YUE2_GROOVE_MODELS=/path/to/models   # optional: a folder holding YuE2-3B/, YuE2-Vae/, YuE2-Vae-legacy/, SheetSage2/ (and gguf/)
YUE2_GROOVE_BACKEND=auto             # auto | torch | gguf — the engine (see "GGUF engine" below; its knobs are in docs/GGUF_ENGINE.md)
YUE2_GROOVE_SHEETSAGE_PYTHON=/path/to/.venv-sheetsage2/bin/python   # 02 COVER (separate env)
YUE2_GROOVE_SHEETSAGE_MODEL=m-a-p/SheetSage2   # or a local SheetSage2 snapshot
YUE2_GROOVE_SHEETSAGE_BASE_MODEL=/path/to/MERT-v2-FullSong   # offline parent encoder snapshot
YUE2_GROOVE_SHEETSAGE_DEVICE=auto              # auto | cuda | mps | cpu
YUE2_GROOVE_SHEETSAGE_KEEP_WARM=1             # reuse a resident SheetSage2 worker (faster repeats)
YUE2_GROOVE_SHEETSAGE_IDLE_SECONDS=900        # resident worker idle lifetime
YUE2_GROOVE_TRANSCRIPTIONS=/path/to/transcriptions   # wins over <runs>/transcriptions

# Apple Silicon memory guard: cap PyTorch's MPS allocator so it cannot drive the
# machine into swap.  Defaults are 1.7 (HIGH) / 1.4 (LOW) of the recommended
# working set; 1.7 can exceed physical RAM on a 64 GB Mac (1.7 x 51.8 GiB = 88 GiB)
# and has caused a kernel panic.  Set BOTH: PyTorch rejects a HIGH below the
# default LOW unless LOW is lowered too.  Never use HIGH=0.0 on a shared machine:
# it *disables* the cap (it does not mean zero memory).
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
```

On Apple Silicon, run `bash scripts/watch_memory.sh` (or `--once` for one
snapshot) in a second terminal while a generation runs: it shows free memory,
kernel pressure, swap, compressor pages and swapfiles, and turns WARN/CRIT well
before the panic state. Keep SWAP and COMPR near zero.

The same things are available as command-line flags:

```bash
.venv/bin/python -m yue2_groove --help
.venv/bin/python -m yue2_groove --host 127.0.0.1 --port 7860 --tab 0 \
    --device auto --dtype auto --model m-a-p/YuE2-3B --runs runs --auth alice:secret
bash scripts/serve.sh start --port 7861 --device mps --dtype bfloat16 --no-preload
bash scripts/serve.sh start -f          # foreground, Ctrl-C to stop
```

`--tab` selects the start tab (`0..6`, order: GENERATE, COVER, EDIT, LIBRARY, TOOLS,
DECODE, BATCH) and forces the Studio view.  `--view song|studio` (or
`YUE2_GROOVE_VIEW`) forces a view for the launch without touching the tab; without
either flag the app opens in SONG and remembers your last choice per browser.
`--sheetsage-python` points at the SheetSage2 interpreter (same as
`YUE2_GROOVE_SHEETSAGE_PYTHON`).

`--device auto` picks CUDA → MPS → CPU; `--dtype auto` is bfloat16 on CUDA/MPS (the
checkpoint dtype) and float32 on CPU.

**CUDA without FlashAttention.** Upstream's CUDA-graph decoder (`BACKEND = torch`) calls the
FlashAttention kernel directly and only checks that the operator exists — so a PyTorch build
compiled without FlashAttention (the Windows CUDA wheels, ROCm) or a pre-Ampere GPU (RTX 20xx)
fails at the first decode step with `RuntimeError: USE_FLASH_ATTENTION was not enabled for
build`. The app asks torch first (`torch.backends.cuda.is_flash_attention_available()` and the
GPU's compute capability) and on such hosts runs upstream's own eager decoder instead —
`BACKEND = torch-eager`, the same code path MPS uses: same model and weights, no CUDA graphs.
Measured on an L4: 1.8× slower per semantic token and about 1.5× end to end, ~700 MiB more
VRAM, and a different take for the same seed (different kernels). The status line and each
run's `local_env.json` say when this happened. Linux hosts with a FlashAttention-capable build
and an Ampere-or-newer GPU are not affected.

**Pre-downloading the weights** (optional, faster than letting the first generation do it):

```bash
uv pip install --python .venv/bin/python hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download m-a-p/YuE2-3B
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download m-a-p/YuE2-Vae
```

If you keep weights in local directories instead (for example `../YuE/models/YuE2-3B`),
set `YUE2_GROOVE_MODELS=../YuE/models` or pass `--model /path/to/YuE2-3B`.

## Why the override file?

Two dependency pins conflict, and `uv`'s `--overrides` resolves both without touching
upstream code:

| Pin | Reason |
|---|---|
| `torch==2.14.0` (macOS only) | YuE2 pins `torch==2.10.0`, whose MPS single-query attention kernel reads out of bounds once the KV cache passes 1024 tokens ([pytorch/pytorch#174861](https://github.com/pytorch/pytorch/issues/174861)) — in bfloat16 the values turn to garbage and NaN well before a minute of audio. Fixed in torch 2.11. `scripts/mps_sdpa_check.py` reproduces the defect on 2.10 and passes on 2.11+. |
| `huggingface-hub==0.36.2` | Gradio asks for `huggingface-hub>=1.16`, YuE2 pins `0.36.2` (transformers 4.57.x needs it). Keeping upstream's pin is safe; Gradio works with it. |

With plain `pip` there is no override mechanism; install in this order and accept the
version warning:

```bash
python -m venv .venv && . .venv/bin/activate
pip install ../YuE
pip install -e .
pip install "huggingface-hub==0.36.2"            # both platforms
pip install "torch==2.14.0"                      # macOS only
```

More on the Apple Silicon story, with measurements: [docs/MACOS_MPS.md](docs/MACOS_MPS.md).
The Linux + NVIDIA story, with measured VRAM, memory budget and FP8 numbers:
[docs/LINUX_CUDA.md](docs/LINUX_CUDA.md).
Why the same seed produces a different song on macOS than on CUDA — five controlled runs,
two experiments, the sampler-level cause, and a blind listening test on a fixed composition
that finds no platform deficit: [docs/CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md).

## Cover from audio (SheetSage2)

**02 // COVER** turns a recording into a cover: upload audio → SheetSage2 transcribes it to ABC
(melody-only or full score) → review/edit the score, strip chords → **SEND TO GENERATE** fills
the ABC and the right plan mode, or **GENERATE COVER** generates right on the COVER tab with its
own style/lyrics → the normal YuE2 generation path runs unchanged. A previously saved score or
transcription can be reused after a page reload with **SOURCE WORK → LOAD ABC** (optionally then
**SEND TO EDIT**).

SheetSage2 pins different torch/transformers versions than YuE2, so it runs in its **own
virtual environment** and the UI talks to it over a subprocess boundary: nothing in the groove
process imports `transformers`, and an unconfigured SheetSage2 cannot break the rest of the UI.
Install it next to the YuE checkout (Linux/CUDA shown; see the model card for others):

```bash
cd /path/to/YuE
python3.11 -m venv .venv-sheetsage2
.venv-sheetsage2/bin/python -m pip install huggingface-hub==0.36.0
.venv-sheetsage2/bin/hf download m-a-p/SheetSage2 --local-dir models/SheetSage2
.venv-sheetsage2/bin/python -m pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
.venv-sheetsage2/bin/python -m pip install -r models/SheetSage2/requirements.txt

# in yue2_groove/.env (or pass --sheetsage-python on the command line)
YUE2_GROOVE_SHEETSAGE_PYTHON=/path/to/YuE/.venv-sheetsage2/bin/python
```

FFmpeg 6.1+ must be on `PATH`. MERT-v2-FullSong, SheetSage2's parent encoder, downloads
automatically — do not install it separately. On macOS use the same commands without the CUDA
index and `device=mps` (untested) or `device=cpu` (works, slow). The **CHECK ENVIRONMENT**
button probes the second venv without loading weights and **AUTO-DETECT VENV** finds
`./.venv-sheetsage2` or `../YuE/.venv-sheetsage2` for the session; the TRANSCRIBE task picks
vocal-only, vocal+instrumental, or full-score (with chords) output, and **MELODY VOICES**
decides which melodies a cover keeps. **KEEP SHEETSAGE2 WARM** reuses one
resident worker between transcriptions (fast repeats; **UNLOAD SHEETSAGE2** or the idle reaper
frees it), while the default is a fresh process per transcription that returns all memory on
exit. Running both models sequentially on one GPU is the supported setup.

Manual walkthrough (`C1`/`C2`), failure behavior and hardware expectations:
[docs/COVER_EDIT.md](docs/COVER_EDIT.md).

## Edit a work and compare

**03 // EDIT** is the iteration loop from the upstream skill: load a saved work → **FREEZE
BASELINE** (hashes + copies of `score.abc`/`request.json`; the original run directory is never
modified) → edit the ABC → **CHECK INVARIANTS** (exact sounding-note/meter comparison, per
voice, with an explicit allow-tempo flag) → **GENERATE EDITED**.

CHECK INVARIANTS runs under an explicit contract: **EXACT** keeps sounding notes and the
meter grid (tempo/meter changes can be permitted), **PITCH** keeps only the ordered pitch
sequence so rhythm may change, **FREE** records differences without gating. The contract and
permissions are written to `edit_manifest.json`.

Generation refuses to run unless the check passed on exactly the current ABC: the edited score
is always submitted, so an edit can never silently fall back to a fresh plan. `ALLOW
MELODY/RHYTHM CHANGES` exists for intentional adaptations: it only lets a *failing* check
through — FREEZE BASELINE and CHECK INVARIANTS are always required. Sampling parameters are
shared with 01 GENERATE (the EDIT tab mirrors them read-only). Each attempt is a new run directory
with `edit_manifest.json` (source/edit hashes, frozen flag, invariant result, permitted changes),
and **BUILD COMPARISON // baseline vs edit** creates a local listening page from both runs.

Manual walkthrough (`E1`) and the failure cases (`X`):
[docs/COVER_EDIT.md](docs/COVER_EDIT.md).

## Compare the three plan modes

**ALL MODES** (01 GENERATE) runs the same text request three times — `full`, `melody`, `off` —
like upstream's `all-modes`: text-only input (an ABC would be invalid for `off`), one fresh
directory per mode under `runs/<stamp>-allmodes-<id>/`, a `run.json` summary at the group root,
and a retained `failure.json` when one mode fails while the others keep running. Each mode
appears in 04 LIBRARY like any other work, and when at least two modes complete the UI also
builds the same local listening bundle as 05 TOOLS with the link shown under the buttons.

## GGUF engine (optional): cards under 16 GB and Windows

A second engine for the same model: YuE2 through [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp)
(a GGML port, MIT) with an 8-bit (Q8_0) backbone.

- **Why.** The reference engine needs ~11 GB of VRAM for a full-length song and, on Windows, runs
  on a PyTorch build without FlashAttention. The GGUF engine has 3.8 GB of weights, peaks at
  8.2 GB for a full-length song (measured on a 16 GB card; 6.1 GB on an 8 GB card with the
  automatic context cap), brings its own FlashAttention on every platform, and composes (the AR
  stage) 2.5–6× faster; the rendering stage is not faster.
- **What it costs.** It is **not the reference configuration**: the same seed gives a *different
  take*. The rendering itself is as close to the reference as the CUDA → Apple Silicon difference
  the project already accepts (24 dB SNR with the reference's own tokens and noise), and a blind
  ABX could not tell the two apart (6/12). Every run records which engine made it; GGUF runs are
  never mixed with reference runs in a comparison.
- **When it is used.** `--backend auto` (the default) picks it on a CUDA card under 16 GB — or an
  NVIDIA card that a CPU-only torch cannot see — once its binaries are installed. Never
  automatically on Apple Silicon or on 16 GB and up; BACKEND → **gguf** in the settings rail
  switches by hand anywhere (a Mac composes faster with it too).
- **Getting it.** Pinokio installs it with Update and the Docker image carries it. By hand:

```bash
.venv/bin/python -m yue2_groove.gguf_engine install     # this platform's binaries → bin/yue2cpp
uv pip install --python .venv/bin/python -e ".[gguf]"    # the converter's gguf writer
.venv/bin/python -m yue2_groove.gguf_engine prepare      # optional: make the GGUF files now (under a minute, +4.3 GB on disk)
```

The GGUF files are made from the weights already on disk at the first GGUF generation (no
download). Measurements, the selection rule, environment variables and limits:
[docs/GGUF_ENGINE.md](docs/GGUF_ENGINE.md).

## Tips: creativity knobs and instrumental / vocals

- **[Generation creativity knobs](docs/GENERATION_CREATIVITY_KNOBS.md)** — producer guide: three ideas (CFG / temperature+seed / plan mode), recipes, then a short glossary. SEED `-1` (the default) draws a new seed per run and records it; song length follows the lyrics, with a ceiling of 6:00 by default and up to 12:00 (the *Long song* preset; the app trims a ceiling that would not fit the model's context and says so).
- **[Exclude / instrumental research](docs/EXCLUDE_INSTRUMENTAL_RESEARCH.md)** — YuE2 has no Suno-style exclude API; upstream issues, community workarounds (empty lyric sections, ABC `V: Vocal` rests), and what groove can/cannot do.

## Keeping up with upstream

YuE2 moves quickly. This UI pins nothing about upstream at the code level; the compatibility
matrix is:

| yue2-groove | YuE2 (`yue2-infer`) | Notes |
|---|---|---|
| 0.8.x | `yue2-v0.1.6` | NAR/VAE progress via an internal hook; a [pull request](https://github.com/multimodal-art-projection/YuE/pull/173) adds a public `on_progress` callback that the UI uses automatically once merged |

SheetSage2 is used only by 02 COVER, through its Transformers interface
(`AutoModel.from_pretrained(..., trust_remote_code=True)` then `transcribe(..., melody_only=True)`).
The driver verifies that `melody_only` is an explicit keyword and refuses older revisions
instead of guessing; it never imports SheetSage2's code into this process.

To upgrade YuE2:

```bash
git -C ../YuE pull
uv pip install --python .venv/bin/python ../YuE --overrides overrides/macos.txt   # or linux.txt
.venv/bin/python -m pytest -q          # contract tests: green means the UI still matches upstream
```

If `tests/test_contract.py` fails, upstream changed something the UI relies on; the fix
belongs in `yue2_groove/adapter.py`, the only module that imports `yue2`.

## Development

```bash
uv pip install --python .venv/bin/python -e ".[test]" --overrides overrides/macos.txt
.venv/bin/ruff check . && .venv/bin/ruff format --check .   # lint + formatting (config in pyproject.toml)
.venv/bin/pyright                                          # types, basic mode, the package only (pyrightconfig.json)
.venv/bin/python -m compileall -q yue2_groove tests scripts
.venv/bin/python -m pytest -q                              # node on PATH also syntax-checks the bundled JS
```

These are the CI gates (`.github/workflows/tests.yml`); `ruff format .` fixes formatting.
`yue2_groove/vendor/` is excluded from all of them because it is an upstream
byte-identical copy (see NOTICE).  Conventions and module boundaries: `CONTRIBUTING.md`.

- `yue2_groove/webui/` — the Gradio app: `runtime.py` (the one pipeline, the one job, the
  run directory), one `*_tab.py` per Studio tab, `song_view.py` (the SONG director),
  `theme.py` + `frontend.py` (stylesheet assembly, page head, bundled scripts),
  `layout.py` (`build_ui`), `cli.py` (`main`).
- `yue2_groove/static/` — the bundled frontend: the stylesheet and every client script
  (served through `allowed_paths`; `tests/test_static_assets.py` is their gate).
- `yue2_groove/library.py` — the Library backend; standard library only, reads run
  directories by file convention and never imports `yue2`.
- `yue2_groove/adapter.py` — every call into `yue2`; keep it that way.
- `yue2_groove/sheetsage_adapter.py` — the only module that runs SheetSage2; subprocess,
  progress, cancel. It never imports `transformers` in this process.
- `yue2_groove/sheetsage_driver.py` — the stdlib-only script executed inside the SheetSage2
  venv (adapted from upstream's `skills/yue2-music/scripts/transcribe.py`).
- `yue2_groove/cover.py` — transcription task → plan mode, chord stripping, cover request.
- `yue2_groove/edit_flow.py` — baseline freeze, invariant check, edit manifest.
- `yue2_groove/vendor/` — `abc_tools.py` and `listen.py` copied from upstream's
  `skills/yue2-music/scripts` (Apache 2.0; the wheel does not ship them).
- `scripts/serve.sh` — service manager; `scripts/mps_sdpa_check.py` — MPS kernel guard.
- `docs/COVER_EDIT.md` — 02 COVER / 03 EDIT manual, manual E2E checks C1/C2/E1/X.
- `docs/GENERATION_CREATIVITY_KNOBS.md` — producer-facing GENERATE creativity guide (recipes + glossary).
- `docs/EXCLUDE_INSTRUMENTAL_RESEARCH.md` — upstream research: no exclude API; instrumental workarounds.
- `yue2_groove/gguf_engine.py` — the GGUF engine: yue2.cpp as a child process, upstream's run layout, the BACKEND=auto rule; `vendor/yue2cpp_convert.py` is yue2.cpp's converter (MIT, byte-identical).
- `docs/LINUX_CUDA.md` — Linux + NVIDIA CUDA validation: measured VRAM budget and boundaries, ODE steps, FP8 cost.
- `docs/LOW_VRAM.md` — quick reference for 12 / 16 GB cards and Windows hosts; `docs/GGUF_ENGINE.md` — the GGUF engine: measurements (Q8_0 vs the reference rendering, blind ABX), install, selection, limits.
- `docs/CROSS_PLATFORM.md` — CUDA vs Apple MPS: why the same seed differs (sampler RNG device), what was ruled out, measured performance ratios.

## Credits and license

- [YuE2](https://github.com/multimodal-art-projection/YuE) by the Multimodal Art Projection
  team — the model and the `yue2` runtime (Apache 2.0). Model weights: CC BY-NC 4.0.
- [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) and
  [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) — optional audio→score
  models for 02 COVER, loaded from their public snapshots. Weights: CC BY-NC 4.0.
- [abcjs](https://github.com/paulrosen/abcjs) renders the scores (MIT).
- This repository: Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
  Not affiliated with or endorsed by the YuE2 authors.
