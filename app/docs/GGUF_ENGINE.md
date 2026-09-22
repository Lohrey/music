# The GGUF engine — YuE2 through yue2.cpp, for cards under 16 GB

**Status:** rendering measured on an M1 Max (Metal) against the CUDA reference run; full generations
run through the app on the M1 Max, on an RTX A4000 16 GB (Linux, CUDA) and on an RTX 2070 8 GB
(Windows 11, Pinokio install), the CUDA runs with VRAM sampled every second (2026-09-20/21); no
physical 12 GB card yet. **Not the reference configuration** — see §4 before relying on it.

YUE2 // GROOVE's reference configuration is upstream's: the unmodified BF16 model in PyTorch, validated
at 24 GB and measured down to a 12 GB budget ([LINUX_CUDA.md](LINUX_CUDA.md)). Below that, and on
Windows hosts whose PyTorch build has no FlashAttention, the reference path is slow or does not fit.
This engine runs the same model through [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp) (a
GGML port, MIT) with an 8-bit backbone: 3.8 GB of weights instead of 7.3, its own FlashAttention on
every platform, and — measured below — a rendering the ear could not tell from the reference.

## 1. What it changes, in numbers

The rendering test removes the sampling stage (which forks on any perturbation) and renders the
**reference run's own score, semantic tokens and seeded noise** (the CUDA run of
[LINUX_CUDA.md](LINUX_CUDA.md)) through each stack. Both inputs were verified bit-identical before
the runs. Metrics as in `scripts/rerender.py`.

| Same tokens + noise rendered by | latent Δ / std | latent corr | audio SNR |
|---|---|---|---|
| PyTorch BF16 · MPS (M4 Pro) — CROSS_PLATFORM §9.3 | 2.23 % | 0.99975 | 28.6 dB |
| yue2.cpp **BF16** GGUF · Metal | 3.15 % | 0.99951 | 25.9 dB |
| yue2.cpp **Q8_0** GGUF · Metal | 3.68 % | 0.99933 | 24.0 dB |
| PyTorch · ODE 16 steps (LINUX_CUDA §5) | 5.46 % | 0.99851 | 21.3 dB |
| yue2.cpp Q6_K GGUF · Metal | 7.89 % | 0.99691 | 17.7 dB |
| PyTorch · ODE 8 steps | 9.4 % | — | 16.6 dB |
| yue2.cpp Q5_K_M GGUF · Metal | 11.17 % | 0.99374 | 14.6 dB |

Quantization alone (Q8_0 against the BF16 GGUF on the same engine): 2.30 % / 27.3 dB — the same size as
the CUDA → MPS platform change the project already accepts. **Blind listening:** two passes over the
eight takes found no reliable difference; an ABX of the reference against Q8_0 (instant switching, the
song's quietest 15 s, 12 trials) scored **6/12 — chance**. Q6_K and below degrade measurably; the port's
own note, "an audio code LM breaks below Q5", agrees. Q8_0 is therefore the only quant the app offers by
default (`YUE2_GROOVE_GGUF_QUANT` can pick the others for experiments).

Speed: the AR stage is bandwidth-bound and is where the 8-bit weights pay; the NAR stage is
compute-bound and does not. The reference request (the full-length song of LINUX_CUDA.md, CFG 1.5,
cot full), generated end to end through the app:

| Host | Semantic | NAR | End to end | Reference (PyTorch, same request) |
|---|---|---|---|---|
| RTX A4000 16 GB, CUDA | 99.8 tok/s | 132 s | **229 s** for 272 s of audio | L4 24 GB: 39 tok/s, 69 s, 325 s |
| RTX 2070 8 GB, Windows, CUDA | 96.7 tok/s | 194 s | **351 s** for 291 s of audio (context capped, §3) | — (does not fit) |
| M1 Max 64 GB, Metal | 69.8 tok/s | 613 s | **785 s** for 265 s of audio | M4 Pro: 11 tok/s, 793 s, 1 231 s |

Memory: on the A4000, `nvidia-smi` sampled every second put the full-length CFG 1.5 song at a
**peak of 8 239 MiB** at the full 24 576 context — 2.3 GiB under the PyTorch reference's
10.5–10.8 GiB (LINUX_CUDA §3), so a 12 GB card keeps ~3.8 GB for the desktop. On the RTX 2070 the
same song with the automatic cap (§3) peaked at **6 125 MiB including the Windows desktop's
~0.8 GB** — about 5.4 GB for the engine, 2 GB of headroom on an 8 GB card.

## 2. Install

Three pieces, all optional — without them the app behaves as before (a CUDA card under 16 GB gets
one line in the log and STATUS saying the engine would fit better; nothing else changes):

1. **Binaries** — built by `.github/workflows/yue2cpp.yml` at the pinned yue2.cpp commit
   (`gguf_engine.YUE2CPP_PIN`) for macOS arm64 (Metal), Linux x64 (CUDA 12.8 + Vulkan + CPU) and
   Windows x64 (CUDA 12.8 + Vulkan + CPU), attached to each app release. Fetch this platform's:

   ```bash
   .venv/bin/python -m yue2_groove.gguf_engine install        # → <repo>/bin/yue2cpp
   ```

   or unpack the asset yourself and point `YUE2_GROOVE_YUE2CPP` at the directory. The archives carry
   the CUDA runtime, so an NVIDIA driver is enough; macOS needs nothing else. (A build from source is
   `cmake -S . -B build && cmake --build build` in a yue2.cpp checkout at the pinned commit.)
2. **`gguf` package** — `uv pip install --python .venv/bin/python -e ".[gguf]"` — the writer the
   converter uses. Skip it if you place ready-made GGUF files in `YUE2_GROOVE_GGUF`.
3. **GGUF files** — prepared automatically on the first GGUF generation (under a minute, no
   download): the checkpoints already on disk are converted with yue2.cpp's own converter (a
   byte-identical vendored copy, `yue2_groove/vendor/yue2cpp_convert.py`) and quantized with the
   release's `quantize`. Ahead of time: `python -m yue2_groove.gguf_engine prepare`. The result is
   **tensor-for-tensor identical to the published `Serveurperso/YuE2-GGUF` Q8_0** (627/627 tensors),
   so provenance stays with the weights you verified. The VAE is never quantized (F32). The files are
   named after everything the conversion reads — the weights (hashed once, remembered in
   `.hashes.json`, and checked against `weights_manifest.json`: a mismatch is an integrity error),
   `config.json`, `qwen.tiktoken` and the converter itself (line endings folded, so a Windows
   checkout names the same file) — so a different MODEL, MODEL REVISION, VAE, config or converter
   is converted afresh, never served an older file; the older file stays (nothing deletes what
   another process may be reading) and the log names it as deletable. The 7.2 GB BF16
   intermediate lives in a private `.partial-*` directory for the duration of one preparation and
   is never shared, so two preparations cannot disturb each other. Plain-named files
   (`YuE2-3B-Q8_0.gguf`, `YuE2-Vae-F32.gguf`) downloaded from the published repository are used as
   they are only from an explicitly set `YUE2_GROOVE_GGUF`; runs then record their source as
   *external, not verified* rather than pretending the local checkpoint produced them.

`python -m yue2_groove.gguf_engine check` prints what would be used and what BACKEND=auto decides.

## 3. Selection

- `--backend auto` (the default; also `YUE2_GROOVE_BACKEND`, and `scripts/serve.sh start --backend …`):
  a **CUDA card under 16 GB** (by marketed size: a "16 GB" card reports 15.99 GiB and counts as 16)
  with the binaries installed gets the GGUF engine; everything else keeps the PyTorch engine. The
  threshold is `YUE2_GROOVE_GGUF_VRAM_GIB`; 16 is where
  LINUX_CUDA.md measured that the reference configuration runs everything the app can produce. A
  small card without binaries stays on PyTorch and says so in the log and STATUS. The card is read
  with `nvidia-smi` (no CUDA context in the app's process), so an NVIDIA card next to a **CPU-only
  torch** (PyPI's Windows wheel) also gets the GGUF engine instead of hours on the CPU — and without
  the binaries, the log and STATUS say that installing them would use the card.
- Apple Silicon is never switched automatically (the reference path is what the app is developed on),
  but BACKEND → **gguf** in the settings rail works there too and is much faster for the AR stage.
- The rail's DTYPE / QUANTIZATION / OFFLOAD AR / MEMORY BUDGET do not apply to this engine; ODE STEPS
  and VAE CORE FRAMES do.
- **8 GB cards** get a context cap by default: `max_seq 12288` (two KV sets ≈ 2.7 GB instead of
  5.4 → about 5.5 GB peak instead of 8.2). yue2.cpp refuses a prompt plus semantic budget its cache
  cannot hold, so the engine trims the semantic `max_tokens` up front — from the exact prefix for an
  external score, from the worst case (the ABC budget) for a model-written one: a full-length
  request with default lyrics keeps ~4.9 minutes. STATUS shows the cap, `config.json` records it
  (`max_seq`, `semantic_budget_cap`). `YUE2_GROOVE_GGUF_MAX_SEQ` sets the cap on any card. Measured
  on an RTX 2070 8 GB: the reference request ended by itself at 4:51, one second under the cap,
  at 6.1 GB peak with the desktop.

- DEVICE does not apply either: yue2.cpp picks the best backend it finds (Metal / CUDA / Vulkan /
  CPU). An explicit `--device cpu` or `mps` does keep BACKEND=auto on the PyTorch engine — the
  "torch cannot see the card" rule only fires when the device was chosen automatically.

Runs from this engine record `"backend": "gguf"`, the quant, the GGUF names and hashes, the source
checkpoints' weight hashes, and the yue2.cpp version — both the pin the app was written against and
what the installed binary itself reports on its banner, with the binary's SHA-256 — in `config.json`
/ `result.json` / `local_env.json`; the Library shows them. `prefix.npy`, `abc_tokens.npy` and
`plan.json` are rebuilt with upstream's own tokenizer and prefix rule (`adapter.text_tokenizer` /
`token_prefixes` / `symbolic_plan`), so 03 EDIT, 06 DECODE and the comparison page accept a GGUF
run like any other. 06 DECODE decodes with yue2.cpp's VAE (`neural-codec`); VAE overrides need the
torch backend.

**The score ids.** yue2.cpp feeds the semantic stage the ids it sampled; the app re-tokenizes the
score *text* it returns (`config.json: "plan_ids_provenance": "retokenized"`). In every
model-written score from the validation runs (28 of 28, CUDA and MPS) the two are identical — the
model writes canonical tokenizations — and an external score is re-tokenized by upstream itself. For an exact record set
`YUE2_GROOVE_GGUF_EXACT_IDS=1`: the engine's own ids are read from its debug dump (a few hundred MB
of scratch per song, deleted afterwards), parsed at the protocol's markers (the dump holds the first
acoustic chunk, which is the whole song unless `max_seq` splits it) and checked against the returned
semantic stream; the run then records `"engine"` and whether they matched the re-tokenization. If
the exact ids were asked for and cannot be obtained, the run fails rather than recording a guess.

## 4. What it is not

- **Not the same song.** Even the BF16 GGUF differs from PyTorch at the ε level, and any ε forks the
  sampled token stream within a few dozen draws ([CROSS_PLATFORM.md](CROSS_PLATFORM.md)). The same
  seed reproduces on one machine with one engine — two GGUF runs here were byte-identical — but not
  across engines. Seeded A/B comparisons stay within one engine.
- **Not validated end to end on a 12 GB card.** 16 GB and 8 GB are measured (§1); 12 GB sits between
  them with the A4000's 8.2 GB peak at the full context. Reports from real 12 GB cards are welcome.
- **Not upstream.** yue2.cpp is a young, single-maintainer project on a patched GGML fork. It is pinned
  by commit, run behind the same process boundary as SheetSage2 (JSON in, files out, nothing imported),
  and can be removed without touching the reference path.
- **The AR stage's quality distribution is unmeasured.** The rendering test above is exact; whether
  the quantized sampler writes songs of the same quality is a distributional question only a blind
  test over many generations answers.

## 5. Platforms

| Platform | Backend | Status |
|---|---|---|
| macOS, Apple Silicon | Metal | run through the app on an M1 Max with the release package: install, prepare, GENERATE |
| Linux, NVIDIA | CUDA (Vulkan and CPU in the same package) | run through the app on an RTX A4000 16 GB with the release package: fresh host, prepare, GENERATE, COVER, Library; the [Docker image](../deploy/docker/README.md) carries this package, so a container on a small card gets the engine by itself |
| Windows, NVIDIA / AMD | CUDA / Vulkan | run through the app on an RTX 2070 8 GB (Windows 11) by way of the Pinokio launcher's Update: install, prepare, GENERATE with the automatic context cap; Vulkan (AMD) built and smoke-tested only |

Each yue-synth process on macOS compiles the Metal shader library (~20 s) before it starts; CUDA
loads in a second or two from the page cache.

**Windows paths.** yue2.cpp opens files through the ANSI code page. The engine keeps its working
files next to the GGUF files and hands non-ASCII paths over as 8.3 short names, but the safe setup
is an app folder (and `YUE2_GROOVE_GGUF`) whose path is plain ASCII — which Pinokio's is.
