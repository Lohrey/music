# YuE2 on Linux + NVIDIA CUDA — memory and performance validation

**Date:** 2026-09-15 · **Revised:** 2026-09-18 with a second L4 session (§3.6–§3.8, §7) · **Status:** end-to-end Linux/CUDA validation of this app, two sessions on two different L4 hosts
**Companion docs:** [MACOS_MPS.md](MACOS_MPS.md) covers Apple Silicon;
[CROSS_PLATFORM.md](CROSS_PLATFORM.md) compares the two platforms run-for-run (why the same seed differs).

Upstream YuE2 documents a single validated platform: *Linux · Python 3.12 · NVIDIA GPU with BF16 support and 24 GB VRAM*. This report turns that guideline into measured numbers for this app: what actually runs, what it actually needs in VRAM, where the failure boundary is, and what the experimental FP8 mode does and costs.

All runs below were executed on rented cloud GPUs (RunPod, NVIDIA L4) with one fixed song request so that single variables could be changed per run — session 1 (2026-09-15, driver 595.91.07) through the real UI, session 2 (2026-09-17, a different L4 host, driver 570.195.03) headlessly through the same app functions the UI's batch tab calls. Raw artifacts (timings, hashes, audio) are recorded in each run's `result.json`; session-2 peaks come from `nvidia-smi` sampled every second alongside each run.

---

## 1. Tested configuration

### Hardware

| | |
|---|---|
| GPU | NVIDIA L4, 24 GB GDDR6 (Ada Lovelace, compute capability **8.9**) |
| VRAM visible to PyTorch | **22.04 GiB** (`torch.cuda.get_device_properties`) |
| Driver / CUDA | 595.91.07 / CUDA 13.2 |
| Host | RunPod Secure Cloud, EU (Iceland), $0.49/hr |
| CPU | AMD EPYC 7702, 128 threads |
| System RAM | 503 GiB (no swap) |
| Disk | 80 GB container volume |

### Software

| Package | Version |
|---|---|
| OS | Ubuntu 24.04.3 LTS (kernel 6.8.0) |
| Python | 3.12.3 (`uv` venv) |
| torch | 2.10.0+cu128 |
| transformers | 4.57.6 |
| huggingface-hub | 0.36.2 |
| safetensors / tiktoken / soundfile | 0.7.0 / 0.12.0 / 0.13.1 |
| Models | `m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae` (BF16, ~7.8 GB on disk) |

`yue2 doctor` on this host reports `dependencies_ready: true`, CUDA BF16 support `true`, and `compute_capability [8, 9]`.

### The test request

One request was used for every run unless a row says otherwise:

| | |
|---|---|
| style | grunge / hard-edged 90s alternative rock, 116.8 BPM, 4/4, C#/Db major |
| lyrics | "Something True" — a full song (~50 lines, sections from `[Intro]` to `[Outro]`) |
| `cot` | full |
| seed | 831001 |
| cfg_scale | 1.5 (the author's standard test value; default is 1.0) |
| ODE steps | 32 (protocol default) unless noted |
| dtype | bfloat16 · backend `torch` (CUDA graphs when available) |

---

## 2. Result: Linux + CUDA works

The full pipeline ran end-to-end on Linux/CUDA — install, model load, symbolic plan, semantic generation, acoustic synthesis, and VAE decode — and produced a complete 48 kHz stereo song with no truncation.

| Metric | Value |
|---|---|
| Audio produced | 284.9 s (4:45), 48 kHz stereo, FLAC 54 MB |
| End-to-end | **324.8 s** (5:25) → real-time factor ≈ **1.14** |
| Truncation | none (`truncated.abc: false`, `truncated.semantic: false`) |
| Peak VRAM observed | 10.5 GiB |

### Phase breakdown (reference run)

| Stage | Time | Detail |
|---|---|---|
| ABC plan | 59.7 s | 3 074 tokens · 51.5 tok/s · `execution: cuda_graph` · `attention: flash` |
| Semantic AR | 181.3 s | 7 124 tokens · 39.3 tok/s · 2 CFG branches (cfg 1.5) · `cuda_graph` |
| NAR synthesis | 69.4 s | 32 ODE steps (midpoint) |
| VAE decode | 14.4 s | 1 024-frame window, 16-frame halo |
| Model load | ~6.8 s | weight resolve + integrity check + transfer to GPU |

The optimized CUDA paths are active: **CUDA Graphs** for both AR stages and **Flash Attention** throughout. GPU utilization held 98–100 % at the card's 72 W power cap for the duration (~56–59 °C).

---

## 3. Memory: what the app budgets vs. what the song needs

### 3.1 The budget mechanism

`MEMORY BUDGET` (Settings → RUNTIME) is not a target — it is a hard cap:

```
allowed = min(budget_setting − 2 GiB, total_visible_VRAM − 2 GiB)
torch.cuda.set_per_process_memory_fraction(allowed / total_visible_VRAM)
```

The 2 GiB reserve exists so that the driver, CUDA context and non-PyTorch allocations are never squeezed. Exceeding the cap raises a clean `OutOfMemoryError` for that run; the app records the failure and keeps running (the next run reloads and retries).

At budget ≤ 12 GiB the pipeline additionally **halves the VAE decode window** (1 024 → 512 frames) to shrink the decoder's working set.

### 3.2 Controlled memory runs

Same request, same seed, only the memory setting changed:

| Run | dtype | Memory budget | Effective cap¹ | Result | Peak VRAM observed² |
|---|---|---|---|---|---|
| reference | BF16 | 24 | 20.0 GiB | ✅ complete | 10.53 GiB |
| budget 16 | BF16 | **16** | 14.0 GiB | ✅ complete | 10.81 GiB |
| budget 12 | BF16 | **12** | 10.0 GiB | ❌ OOM at ~60 s | 10.25 GiB at failure |
| budget 12, repeated | BF16 | 12 | 10.0 GiB | ❌ OOM at ~60 s (reproduced) | 10.25 GiB at failure |
| FP8 at budget 12 | **FP8** | 12 | 10.0 GiB | ✅ complete | 9.92 GiB |

¹ On this L4: `min(budget − 2, 22.04 − 2)` GiB.
² Sampled every 5 s with `nvidia-smi`; brief spikes between samples are possible, so treat peaks as lower bounds (the OOM runs give an independent lower bound of ≥ 10.25 GiB for the BF16 working set).

### 3.3 Where the boundary is, and why

The BF16 failure is reproducible and precise: both budget-12 runs die exactly **≈ 60 s in**, at the hand-off from the ABC plan to the semantic AR stage — the first phase that runs with two CFG branches:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB.
GPU 0 has a total capacity of 22.04 GiB of which 11.78 GiB is free.
This process has 10.25 GiB memory in use. 10.00 GiB allowed; ...
```

So the BF16 working set for this song is **slightly above 10.25 GiB** (measured peak 10.5–10.8 GiB across passing runs), and the binding constraint is the semantic stage with `cfg_scale 1.5` (two branches, batched).

### 3.4 The budget is a pure cap — proven by hashes

Two runs with different budgets produced **bit-identical output**, which shows the mechanism only limits allocation and does not change any computation:

| Artifact | budget 24 | budget 16 | Match |
|---|---|---|---|
| `score.abc` (SHA-256, 16 hex) | `db2d1467174a7d71` | `db2d1467174a7d71` | ✅ identical |
| `semantic.npy` | `e8a09373bc99f11e` | `e8a09373bc99f11e` | ✅ identical |
| `audio.flac` | `07f934932f357eca` | `07f934932f357eca` | ✅ identical |

### 3.5 What hardware can actually run this

Measured working set for this song (285 s, CFG 1.5, BF16): **~10.5–10.8 GiB peak**.
Measured FP8 working set: **~9.9–10.0 GiB peak**.

Applying the app's own budget arithmetic to card classes:

| Card class | VRAM visible to PyTorch¹ | App cap (total − 2 GiB) | BF16, CFG 1.5 (default) | BF16 with CFG 1.0 or a length cap (§3.6) | FP8 |
|---|---|---|---|---|---|
| 24 GB (3090 / 4090) | ~24 GiB | ~22 GiB | ✅ comfortable | ✅ | ✅ |
| **16 GB (4060 Ti 16G, 5070 Ti, …)** | ~16 GiB | **~14 GiB** | ✅ ~3 GiB headroom, including the longest song at the default ceiling (§3.7) | ✅ | ✅ |
| 12 GB (3060 12G, 4070, …) | ~12 GiB | **~10 GiB** | ❌ fails at ~10.25 GiB | ✅ measured: CFG 1.0 → 9.6 GiB; CFG 1.5 with a semantic cap → 10.3 GiB (tight) or a ≤ 3-minute song → 9.5 GiB | ✅ ~4× slower |
| 8 GB | ~8 GiB | ~6 GiB | ❌ | ❌ | ❌ |

¹ Inference, not measured on those cards. Every row is the L4 under the corresponding allocation cap (14 GiB and 10 GiB), on two different L4 hosts; no physical 16 GB or 12 GB card has been tested. The cap is applied inside PyTorch's allocator; the `nvidia-smi` peaks quoted below include the CUDA context (~300 MB) on top of it.

**Notes for users**

- Cards the reference configuration does not fit — 12 GB at CFG 1.5 full length, anything under —
  have a second option since the GGUF engine ([GGUF_ENGINE.md](GGUF_ENGINE.md)): the same model
  through yue2.cpp with an 8-bit backbone, measured at 8.2 GB peak for this song. Not the reference
  configuration; a different take for the same seed.
- The upstream "24 GB" guideline is conservative for this workload: the model itself is ~7.3 GB in BF16, and a ~5-minute song with the default 1 024-frame VAE window peaks at ~10.8 GiB.
- Longer songs use more context tokens (protocol context is 24 576); this song used ~10 200 tokens. The 24 GB guideline likely covers maximal-length songs — treat 16 GB as the tested floor for typical songs, and expect heavy songs to need more.
- `cfg_scale 1.5` means two CFG branches and roughly doubles the AR-stage memory and time versus `cfg_scale 1.0`. This is the main setting that pushes a run over a memory limit — and, as §3.6 shows, the cleanest way to fit 12 GB.

### 3.6 Session 2: what fits in 12 GB without quantization

The BF16 failure at budget 12 (§3.3) is the semantic stage's KV cache: `StaticKVCache` / `GraphAR` pre-allocate `prefix + max_tokens` positions per CFG branch at 112 KiB per position (28 layers × 8 KV heads × 128 × bf16 × K and V), i.e. 2 × 12 959 × 112 KiB ≈ 2.8 GiB for the default 9 000-token ceiling and two branches. Three lossless ways to shrink that were measured on the second L4 (budget 12 → 10 GiB cap):

| Run | Setting | Result | Peak (`nvidia-smi`, 1 s) | Take |
|---|---|---|---|---|
| E4 | **CFG 1.0** (upstream's default), no other change | ✅ complete, 301 s | **9 576 MiB** | different from the 24 GB take — a different sampling distribution |
| E3 | CFG 1.5, semantic `max_tokens` 7 200 | ✅ complete, 327 s | **10 338 MiB** (tight) | different from the 24 GB take (first differing semantic token 22): the smaller KV buffers change FlashAttention's tiling, and any numeric perturbation flips a token within a few dozen draws |
| E5 | CFG 1.5, a 2-minute song (a solo-piano request), cap 5 000 | ✅ complete, 123 s | **9 480 MiB** | — |

So a 12 GB-class card runs the unquantized BF16 model when the song is short or the CFG is 1.0; at the default CFG 1.5 a full-length song needs a length cap and lands within ~100 MiB of the limit. The cap is "lossless" in the sense of full precision — but it is **not** the same take as the uncapped run, for the same reason every other perturbation in [CROSS_PLATFORM.md](CROSS_PLATFORM.md) is not: the sampled tokens diverge early and stay diverged. FP8 (§4) remains the fallback for full-length CFG 1.5 songs on 12 GB.

### 3.7 Session 2: the longest song at the default ceiling, at budget 16

A request with 207 lines of lyrics drove every stage to its ceiling: the score reached 3 969 tokens (cap 4 096), the semantic stage hit its **9 000-token cap** (6:00 of audio, `truncated.semantic = true`; the ceiling the sliders allowed at the time — they now reach 18 000, at ~330 MB more per extra minute at CFG 1.5), the semantic prefix was 5 456 tokens and the NAR ran over ~14 500 tokens of context. Under the 16 GB budget (14 GiB cap) it completed at **11 568 MiB** peak (E8, 447 s). Upstream's model card quotes 14.08 GiB for its own maximum-context test; on this stack the app's maximum stays ~2.4 GiB below the 16 GB-class cap.

### 3.8 Session 2: AR offload and the eager decoder

| Run | Setting | Result | Peak | Output |
|---|---|---|---|---|
| E6 | budget 24, **AR offload on** | ✅ 330 s | 10 962 MiB | **byte-identical** to the reference (`semantic.npy`, `latent.npy`, `audio.flac`) |
| E7 | budget 24, **BACKEND = torch-eager** | ✅ 498 s | 11 478 MiB | a different take from the ABC stage on (different kernels) |

AR offload is lossless but does **not** lower this workload's peak — the peak is the semantic stage (two CFG branches + KV), and offload only parks the AR weights during the acoustic stage. The eager decoder — the path the app falls back to when a torch build cannot run FlashAttention — costs **1.8× per semantic token** (22.0 vs 39.0 tok/s) and about **1.5× end to end**, and uses ~700 MiB more than the CUDA-graph path.

---

## 4. FP8 quantization: what it does and what it costs

### 4.1 What FP8 is here

`QUANTIZATION = fp8` (Settings → RUNTIME) quantizes **AR linear layers only** (per-tensor E4M3 weights, dynamic activations, `torch._scaled_mm`); the NAR stack and the VAE always run in BF16/FP32. It requires compute capability ≥ 8.9, i.e. RTX 40-series and newer (and this L4). Upstream labels it experimental with **no quality or speed claim**.

### 4.2 Measured result

Same request, same seed, `MEMORY BUDGET = 12`:

| | BF16 (budget 16) | **FP8 (budget 12)** | Ratio |
|---|---|---|---|
| Status | complete | **complete** | — |
| Audio | 284.9 s | 313.8 s | (different take) |
| ABC stage | 63.4 s · 3 074 tok · 48.5 tok/s | **227.7 s · 3 312 tok · 14.5 tok/s** | **3.6× slower per token** |
| Semantic stage | 183.9 s · 7 124 tok · 38.7 tok/s | **1 077.7 s · 7 847 tok · 7.3 tok/s** | **5.4× slower per token** |
| NAR | 69.3 s | 81.0 s | ~1.2× (unaffected by design) |
| VAE | 13.7 s | 11.5 s | ~same |
| **End-to-end** | **330.4 s** | **1 398.0 s (23:18)** | **4.2× slower** |
| Peak VRAM | 10.8 GiB | **9.9 GiB** | −0.9 GiB |
| AR execution | `cuda_graph` | **`eager`** | — |

**FP8 does unlock the 10 GiB budget that BF16 cannot fit** — but at ~4× the wall-clock time.

### 4.3 Why it is slower

Two mechanisms, both visible in the code:

1. **FP8 disables CUDA Graphs.** The sampling path activates graphs only when the AR model is unquantized (`graph_enabled = … and not getattr(model, "_yue2_fp8_originals", {})`), so quantized runs fall back to eager execution and expose per-token kernel-launch overhead. This is the dominant cost.
2. **Single-token decode is padded for FP8 GEMM.** Decode rows are padded to a multiple of 16 before `torch._scaled_mm`, so a batch-of-one step computes 16 rows' worth of work.

NAR (flow matching) and VAE are untouched by FP8, which is why their times barely move.

### 4.4 FP8 changes the song, not just its rendering

With the same seed, FP8 produced a **different plan and different tokens** — not a re-rendering of the same music:

| Artifact | BF16 | FP8 | Match |
|---|---|---|---|
| `score.abc` | `db2d1467174a7d71` | `e1bd6badaa6fc6a8` | ❌ different |
| `semantic.npy` | `e8a09373bc99f11e` | `fa8193769086779a` | ❌ different |
| `audio.flac` | `07f934932f357eca` | `41756563b76434f3` | ❌ different |
| ABC / semantic tokens | 3 074 / 7 124 | 3 312 / 7 847 | different |

Different numerics shift the sampling distribution, and the autoregressive loop amplifies that into a different composition. (This is the same effect that makes `seed` non-portable across platforms — see Appendix A.)

### 4.5 When to use FP8

- **Only** as a memory rescue for cards in the 10–12 GB class, and only on Ada-or-newer hardware.
- Expect ~4× longer generations (≈ 23 minutes for a ~5-minute song on this L4).
- Expect a **different take** than the same seed in BF16 — for A/B comparisons, change one thing at a time.
- Not recommended at 16 GB and above: BF16 runs faster and its output is the reference behavior.

---

## 5. Side finding: ODE steps

Same request, seed, and budget (24), changing only the ODE solver steps:

| ODE steps | ABC | Semantic | NAR | End-to-end | AR hash match vs. 32 |
|---|---|---|---|---|---|
| 32 (default) | 59.7 s | 181.3 s | 69.4 s | 324.8 s | — |
| 48 | 59.2 s | 181.4 s | **103.4 s** | 356.6 s | ✅ plan + tokens identical |

- ODE steps affect **only NAR**, linearly (~2.2 s per step on this L4): +49 % NAR time for 32 → 48.
- The AR output is **bit-identical**, so this is a clean A/B of the acoustic solver alone.
- Listening impression (single listener, not blind): 48 steps was *slightly* better than 32 for a ~10 % total runtime cost — diminishing returns. The protocol default of 32 is a reasonable production setting; 48–64 is a "final render" choice.

Below the default, measured on an M4 Pro by re-rendering this run's tokens with the same noise and
comparing against that machine's own 32-step render (`scripts/ode_steps.py`; method and decoder-side
numbers in [CROSS_PLATFORM.md](CROSS_PLATFORM.md) §9.3):

| ODE steps | NAR | latent RMS Δ / std | latent corr | SNR vs that machine's 32-step render |
|---|---|---|---|---|
| 32 | 629 s | — | — | — |
| 16 | 323 s | 5.6 % | 0.99842 | **21.1 dB** |
| 8 | 164 s | 9.4 % | 0.99557 | **16.6 dB** |

Time stays linear in the step count (164 / 323 / 629 s for 8 / 16 / 32 here; ~2.2 s per step on
the L4 above). The two directions are not symmetric in effect: 32 → 48 was a change the listener
rated *slightly* better, while 32 → 16 and 32 → 8 land 21.1 and 16.6 dB away from the same
32-step render — a different render, not merely a cheaper one.

---

## 6. Artifacts and reproduction

Each run directory contains `result.json` (timings, model identities, artifact SHA-256s), `config.json` (effective settings), `request.json`, `score.abc`, `semantic.npy`, `latent.npy` and `audio.flac`. Every number in this report comes from those files: session 1 is the five budget runs of §3 (the reference, budget 16, budget 12 twice, FP8 at budget 12) plus the ODE-48 run of §5 and its listening page; session 2 is E1–E10 below.

Setup on a fresh Linux + CUDA host (mirrors the launcher's steps, see the Pinokio repo for the GUI path):

```bash
git clone https://github.com/deadjoe/yue2_groove.git app && cd app
uv venv env --python 3.12
uv pip install --python env/bin/python torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128          # first: the cu128 wheel, from PyTorch's index
uv pip install --python env/bin/python -e ".[yue2]" --overrides overrides/linux.txt
env/bin/hf download m-a-p/YuE2-3B && env/bin/hf download m-a-p/YuE2-Vae
env/bin/python -m yue2_groove --host 0.0.0.0 --port 7860 --no-preload --auth user:pass
```

Install torch **before** the app: otherwise the app's dependency resolution pulls PyPI's torch (with its
separate `nvidia-*` CUDA wheels, several GB) only for the cu128 wheel to replace it. On one cloud host
with ~4 MB/s to PyPI that first, wasted install took 26 minutes. The same environment is available
pre-built as a container image — `ghcr.io/deadjoe/yue2_groove`, see [deploy/docker/README.md](../deploy/docker/README.md) —
which turns this whole setup into an image pull. The session-2 runs, as named in this report:

```
E1   the reference request on a second L4 host — bit-identical to session 1      §7
E2   the session-1 reference latent re-decoded — bit-identical audio             §7
E3   budget 12, CFG 1.5, semantic cap 7 200                                       §3.6
E4   budget 12, CFG 1.0                                                           §3.6
E5   budget 12, CFG 1.5, a 2-minute solo-piano song, cap 5 000                    §3.6
E6   budget 24, AR offload on                                                     §3.8
E7   budget 24, BACKEND = torch-eager                                             §3.8
E8   budget 16, the longest song at the 9 000-token default ceiling                §3.7
E9   Cover: SheetSage2 transcription on CUDA                                      §7
E10  five performances of the reference score, seeds 831001–831005               CROSS_PLATFORM §9.1
```

---

## 7. Session 2: reproducibility across hosts, and Cover on Linux

**Bit-exact across L4 hosts and driver versions.** The reference request, re-run from a fresh install on a
different L4 (driver 570.195.03 vs 595.91.07 in session 1), reproduced `abc_tokens.npy`, `semantic.npy`,
`latent.npy` **and `audio.flac`** byte for byte; re-decoding the session-1 reference latent reproduced the
reference audio; and the reference score fed back as an exact ABC with seed 831001 reproduced
`semantic.npy` exactly (the CUDA-side equivalent of the Mac validation row in
[CROSS_PLATFORM.md](CROSS_PLATFORM.md) §9.1). Same GPU model + same torch build ⇒ same bits, on
different machines.

**Cover (SheetSage2) works on Linux + CUDA.** Session 1 had validated Cover only on Windows. In session 2
the second environment was built exactly as the README describes (`uv venv .venv-sheetsage2 --python 3.11`,
`torch==2.8.0+cu126`, `huggingface-hub==0.36.0`, `m-a-p/SheetSage2` into `models/SheetSage2`, the pinned
MERT-v2-FullSong snapshot) and `sheetsage_driver.py` transcribed the first 60 s of the reference song with
`--device cuda`: `result.json` records `device: cuda, dtype: bf16`, the full score export (ABC, MIDI, lab
files) was written with no warnings, and the score came back in **F minor at 128 BPM** — the song it was
transcribing is Fm / 130. FFmpeg 6.1.1 was present on the image. (E9.)

---

## Appendix A — cross-platform reference (not a controlled comparison)

Recorded on the author's Macs against **different songs**, so the numbers are a speed reference only — do not read them as a benchmark:

| Host | Backend | Song | End-to-end | Semantic AR | NAR |
|---|---|---|---|---|---|
| Apple M4 Pro 64 GB | MPS, BF16, eager | jazz, 259 s | 1 515 s | 11.2 tok/s | 792.7 s |
| **NVIDIA L4 (this report)** | **CUDA, BF16, graphs+flash** | **rock, 285 s** | **330 s** | **39.3 tok/s** | **69.4 s** |

Also note: the same seed on a different backend, GPU or quantization setting produces a **different song** (upstream: *"Separate GPUs, runtime versions, or sampling settings can change a seeded generation"*). A torch build change alone on this same L4 (2.10 → 2.14) was bit-exact through the NAR latents — see [CROSS_PLATFORM.md](CROSS_PLATFORM.md) Exp A — but seeded A/B comparisons are still only valid on one machine with one configuration.

---

## Appendix B — scope and limits of this validation

- **One GPU model** (NVIDIA L4, Ada) on two cloud hosts; other architectures were not tested here.
- **One song** (≈ 285 s, ~10 200 of the 24 576 context tokens) at one CFG value (1.5) — memory numbers scale with song length and CFG; the 16 GB / 12 GB card rows are arithmetic extrapolations, not measurements.
- **Peak VRAM** was sampled externally every 5 s; true peaks may be slightly higher. The OOM runs bound the BF16 requirement from below.
- **Listening notes** are a single non-blind listener, one take per setting.
- Session 1 covered **generation through the UI**; session 2 ran headlessly through the same app functions and added Cover on Linux (§7). Cover on Linux was exercised once, on one 60-second clip.
