# Cross-platform analysis: NVIDIA CUDA vs Apple MPS

**Date:** 2026-09-16 · **Revised:** 2026-09-16 (re-analysis of the archived runs: token-level divergence positions, latent comparison, audio metrics recomputed, FP8 run added as a calibration point) · 2026-09-18 (follow-up experiments §9.1–9.3 and the blind listening test §9.5 completed) · **Evidence:** 5 controlled generation runs across 3 machines + 1 CUDA FP8 run, then 1 repeat run, 5 re-renders, 6 fixed-score takes and an 8-item blind listening test · **Status:** the divergence is explained; the quality question was tested on a fixed composition and no platform deficit was found (single listener)

This document records a controlled investigation into a reported **quality difference between
the same model on NVIDIA CUDA and on Apple Silicon (MPS)**: identical model, identical
request, identical settings, but outputs that listeners describe as clearly different in
quality. It also records the experiments that were run to explain the difference — including
two findings that change how the comparison must be interpreted:

> **The sampler uses a different random-number generator device per platform** (CUDA device
> RNG on CUDA, CPU RNG on MPS). The same seed therefore produces **different random draws**
> on the two platforms. This alone guarantees different songs.

> **The platform is confounded with the composition.** Every CUDA take of this request lands in
> one composition family (F minor, 130 BPM) and every MPS take in another (E major, 126 BPM),
> because the two RNG streams part at the very first genuinely random token — the tempo. Any
> "CUDA vs MPS" difference measured here is therefore also an "Fm/130 vs E/126" difference.
> The original five-take experiment therefore **cannot** attribute a quality difference to the
> platform. The design that can — the same composition performed on both platforms, rated
> blind — was run afterwards (§9.1, §9.5): **no platform deficit was found**; the take-to-take
> variation of the draw itself dominates everything the platform contributes.

See also: [LINUX_CUDA.md](LINUX_CUDA.md) (memory/performance validation on CUDA; source of the
FP8 run used here) and [MACOS_MPS.md](MACOS_MPS.md) (MPS support notes and the SDPA defect).

---

## Summary

| # | Finding | Evidence |
|---|---|---|
| 1 | **The sampler's RNG device differs by platform — by code design.** CUDA samples with a CUDA-device generator; MPS samples on the **CPU** with a CPU generator. PyTorch generators are device-specific algorithms (CPU: MT19937, CUDA: Philox), and `torch.multinomial` has different CPU and CUDA implementations, so the same seed yields **different random streams**. | `yue2/sampling.py` (`rng_device = device if device.type in {"cpu","cuda"} else torch.device("cpu")`); local test: same seed, CPU vs MPS generator → different draws, each reproducible. |
| 2 | The same seed therefore produces **different compositions** on CUDA vs MPS — and this is *sufficient* to explain the divergence; no numerical difference in the model is required. | The **first 26 ABC tokens are identical on every platform** (the near-deterministic ABC header); the streams part at **token 27, the tempo** (`Q:1/4=130` on CUDA, `126` on MPS). Two CUDA runs (different torch builds) are byte-identical; chord-set overlap CUDA↔MPS = **0.00**. |
| 3 | **torch version is not a cause — and Exp A turned out to be a zero-perturbation test.** | Exp A: CUDA, torch 2.10 vs 2.14 → `abc_tokens.npy`, `semantic.npy` **and `latent.npy` byte-identical** (AR *and* NAR bit-exact). Only `audio.flac` differs: max \|Δ\| = 6.4 × 10⁻⁶ full-scale, SNR 119 dB — last-bit differences in the fp32 VAE convolutions, far below audibility. Exp A therefore says nothing about how robust sampling is to numeric perturbation. |
| 4 | **Numeric precision changes the take but not the family — and the fp32 take was never evaluated by ear.** | Exp B: MPS, float32 instead of bfloat16 → diverges from the bf16 take at **ABC token 122**, stays in the E/126 family, **1.65× slower**. Its spectral balance is the closest of all Mac takes to the CUDA reference (§4.4). "No quality gain" is *not* established. |
| 5 | **Any real numeric perturbation flips a sampled token within a few hundred ABC tokens, on both platforms.** MPS is not special in this respect; the same-seed takes are *not* reproducible across Macs. | First divergence (ABC token index): FP8 vs BF16 on the **same L4** = 225; M1 Max vs M4 Pro (both bf16) = 122; M4 bf16 vs fp32 = 122; M1 bf16 vs M4 fp32 = 313. Score hashes differ on every pair. |
| 6 | **The five-take comparison cannot establish whether MPS quality is worse** — platform is perfectly confounded with composition family (Fm/130 vs E/126), and the measurable differences track the draw, not the platform. **Tested afterwards on a fixed composition, blind: no platform deficit.** Five Mac performances of the CUDA reference's score were rated 5 / 4 / 3 / 2 / 2 against 3 for the CUDA performance (its two re-renders 4 / 4); the best-rated take was a Mac performance. | The **FP8 CUDA take is nearly as bright as the Mac takes** (centroid 3 373 Hz vs 3 463–3 551) and the **M4 fp32 take is as dark as the CUDA reference** (3 010 vs 2 895) (§4.4); five CUDA performances of the fixed score span 2 895–4 298 Hz, two of them brighter than any Mac take (§9.1). The "static harmony" is a three-chord I–♭VII–IV rock song with a one-chord riff intro — a stylistic outcome, not a defect (§4.1). Blind ratings and the 1-point noise floor: §9.5. |
| 7 | The platforms also use **different attention/execution paths by design** — now bounded for the acoustic stage: re-rendering the CUDA tokens on the M4 Pro gives **SNR 110 dB for the VAE** and **28.6 dB for NAR + VAE**, the latter essentially the same as CUDA's own ODE 32 → 48 change (28.0 dB), with the spectrum unchanged (§9.3); blind, the Mac render of the same performance was indistinguishable from the CUDA render (§9.5). The AR stage remains numerically unquantified (§9.4), but its output was rated blind without a platform deficit. | CUDA: `execution: cuda_graph`, `attention: flash`, both CFG branches in one batch. MPS: `execution: eager`, `attention: sdpa`, `repeat_interleave` K/V expansion, CFG branches as two batch-1 forwards. |

**Practical consequences**

- Same seed ≠ same song across platforms — expect that, and use it deliberately. It is not
  fixable: different RNG algorithms, different `multinomial` implementations, and reduced-
  precision kernel differences amplified by autoregressive sampling.
- Same seed ≠ same song **across Macs** either (M1 Max vs M4 Pro share only 122 ABC tokens).
  On **one** Mac the pipeline is bit-exact end to end: a repeat run on the M4 Pro reproduced every
  artifact, `audio.flac` included (§9.2).
- torch version alignment is **tested and has no effect** (bit-exact AR/NAR). float32 on MPS
  **changes the take** (like any perturbation) and was **not evaluated by ear**; there is no
  evidence here that it is a quality mode, and it costs 1.65×.
- Whether MPS *systematically* produces worse music was tested by holding the composition
  fixed (§9.1) and rating blind (§9.5): **no deficit found** — the Mac performances span the
  whole rating scale around the CUDA one, and the Mac render of the CUDA performance is
  indistinguishable from the CUDA render. What remains open is only a distribution-level
  comparison (the CUDA side has one performance) and the numeric gap itself (§9.4).

---

## 1. Background

The model (YuE2-3B, unquantized BF16) is trained and validated on Linux + NVIDIA. This UI
also supports Apple Silicon through PyTorch's MPS backend, with a torch override for a known
MPS attention defect (see [MACOS_MPS.md](MACOS_MPS.md)).

During routine testing, the same song request produced outputs that were judged by ear to be
**clearly better on CUDA than on two different Macs**. The Macs' outputs were *similar to each
other*, which motivated a controlled comparison rather than a "one bad roll of the dice"
explanation. The investigation found that the platform difference has a code-level cause
(§6.1). A re-analysis of the archived runs then showed that the Macs' similarity is a shared
*prefix* (the two Mac takes part at ABC token 122), and that the listening comparison is
confounded: one composition family per platform (§4.1, §5, §8). The follow-up that removes the
confound — the CUDA composition performed five times on the Mac and rated blind alongside the
CUDA original and its Mac re-render — found no platform deficit (§9.1, §9.5).

## 2. Method

### 2.1 What is held fixed

| Variable | Value |
|---|---|
| Model weights | `m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae` — **SHA-256 verified identical in all runs** |
| Upstream runtime | `yue2-infer` 0.1.6 — **`runtime_sha256` identical in all runs** (hash of the upstream `yue2` runtime files) |
| App | `yue2-groove` 0.8.0 — repo commit `524ac85` (reference and FP8 calibration runs, first cloud session) or `238ff9a` (Exp A, both Controls, Exp B); those two commits differ in **documentation only**, so the generation path is identical |
| Request | same style, lyrics, `seed=831001`, `cfg_scale=1.5`, `cot=full` — the 884-token request prefix (first 884 entries of `prefix.npy`) is byte-identical in all six runs |
| Generation config | `ode_steps=32`, `ode_method=midpoint`, `context=24576`, protocol sampling defaults (ABC: T 0.7 / top-k 30 / top-p 0.9, no CFG; semantic: T 1.0 / top-k 100 / top-p 0.95, CFG 1.5) |
| Runtime config | `backend=torch`, `quantization=none`, `offload_ar=false`, `memory_budget=24`, `vae_core_frames=1024` (FP8 calibration run: `quantization=fp8`, budget 12, `vae_core_frames=512`) |

### 2.2 What is deliberately varied

| Run | Platform | torch | dtype | Purpose |
|---|---|---|---|---|
| Reference | NVIDIA L4 (CUDA) | 2.10.0+cu128 | bfloat16 | the CUDA baseline |
| **Exp A** | NVIDIA L4 (CUDA) | **2.14.0+cu130** | bfloat16 | isolate torch version |
| Control 1 | Apple M1 Max (MPS) | 2.14.0 | bfloat16 | isolate the platform |
| Control 2 | Apple M4 Pro (MPS) | 2.14.0 | bfloat16 | isolate the GPU generation |
| **Exp B** | Apple M4 Pro (MPS) | 2.14.0 | **float32** | isolate numeric precision |
| *Calibration* | NVIDIA L4 (CUDA) | 2.10.0+cu128 | bf16 weights **quantized to FP8**, eager + SDPA | a known, large numeric perturbation on the CUDA RNG stream (from [LINUX_CUDA.md](LINUX_CUDA.md) §4) |

The original design: if torch version mattered, Exp A would diverge from the reference; if the
GPU generation mattered, the two Macs would diverge from each other; if precision mattered,
Exp B would diverge from Control 2. What actually happened: Exp A did not diverge **at all**
(bit-exact through NAR), while the two Macs *and* Exp B diverged from each other at the token
level (ABC token 122) yet stayed in one composition family. §4.1 explains why "family" — not
"same take" — is the right unit, and why the FP8 run is needed to calibrate what a token-level
divergence means.

A platform difference that is *not* varied but is inherent to the code is the **sampling RNG
device** (§6.1): it is CUDA on CUDA and CPU on MPS, so the same seed yields different random
draws on the two platforms no matter what else is configured.

### 2.3 Measurements

- **Divergence** — index of the first differing token between two runs' `abc_tokens.npy` /
  `semantic.npy`; first differing line of `score.abc`. In an autoregressive sampler a single
  flipped token changes every subsequent distribution, so this index — not the similarity of
  what follows — is the measure of how far two runs agree.
- **Composition** — parsed from `score.abc`: key, tempo, section list, chord sequence per
  section, chord-set similarity, and vocal-melody repetition over *sung* bars (full-rest bars
  excluded, chord symbols stripped).
- **Tokens** — `semantic.npy` / `abc_tokens.npy` (count, diversity, entropy, repetition).
- **Latents** — `latent.npy` (NAR output; byte comparison and RMS difference where the token
  streams match).
- **Audio** — `audio.flac`, decoded to PCM. Every spectral and level measure uses the mono mix
  (mean of the two channels): peak, RMS and crest factor (peak/RMS in dB); spectral centroid and share of spectral magnitude above 8 kHz from one magnitude
  spectrum of the whole mono mix; power share per band from 1-s Hann windows; stereo
  correlation; loudness dynamics as the standard deviation and P10–P95 range of 3-s RMS in dB.
  Sample-level comparison where the token streams match.
- **Performance** — per-stage timings from `result.json` (each run records them itself).
- **Integrity** — SHA-256 of every artifact (`result.json` records them; all re-verified
  against the archived files for this revision).
- **Listening** — one experienced listener (the author), non-blind, comparing takes on the same
  playback setup (§4.7). The follow-up in §9.5 is blind: takes served under random codes with
  every provenance field scrubbed, the key opened only after rating.

---

## 3. Test environments

### 3.1 NVIDIA L4 (reference, Exp A and the FP8 calibration run)

| | |
|---|---|
| GPU | NVIDIA L4, 24 GB GDDR6, Ada (compute capability 8.9), 72 W cap |
| Host | RunPod Secure Cloud, EU, Ubuntu 24.04.3, AMD EPYC 7702 (128 threads), 503 GiB RAM |
| Driver | 595.91.07 (CUDA 13.2 capable) |
| Python | 3.12.3 (`uv` venv) |
| torch | reference: **2.10.0+cu128** · Exp A: **2.14.0+cu130** |
| Other pins | transformers 4.57.6, huggingface-hub 0.36.2, safetensors 0.7.0, tiktoken 0.12.0, soundfile 0.13.1 |
| Attention / execution | bf16: `attention: flash`, `execution: cuda_graph` · FP8: `attention: sdpa`, `execution: eager` |
| Sampling RNG device | **CUDA device generator** |

### 3.2 Apple M1 Max (Control 1)

| | |
|---|---|
| Chip | Apple M1 Max, **32-core GPU**, 64 GB unified memory |
| OS | macOS 26.6.2 (Metal 4) |
| Python / torch | 3.12 · **torch 2.14.0** (macOS override) |
| Attention / execution | `attention: sdpa`, `execution: eager` |
| Sampling RNG device | **CPU generator** (MPS falls back to CPU sampling) |
| App path | Pinokio launcher (`--view song`), MPS watermark guard from `.env` (0.8 / 0.5) |

### 3.3 Apple M4 Pro (Control 2 and Exp B)

| | |
|---|---|
| Chip | Apple M4 Pro, **20-core GPU**, 64 GB unified memory |
| OS | macOS 26.6.2 |
| Python / torch | 3.12 · **torch 2.14.0** |
| Attention / execution | `attention: sdpa`, `execution: eager` |
| Sampling RNG device | **CPU generator** |
| App path | `scripts/serve.sh` service manager (repo checkout) |

### 3.4 The fixed request

| | |
|---|---|
| seed / cfg_scale / cot | `831001` / `1.5` / `full` |
| style | 790-character grunge / 90s-alternative prompt, `116, 8 BPM`, 4/4, C#/Db major, "quiet-loud dynamics … loud-quiet-loud dynamic architecture" |
| lyrics | "Something True" — a full song: 111 lines including section tags and bracketed performance directions (`[Intro]` … `[Outro]`) |
| Exact text | preserved in each run's `request.json` |

> **Data-provenance note.** `config.json` contains a `model_dtype` field that upstream
> hardcodes to `bfloat16` (it is a protocol label, not the runtime dtype) — the fp32 run
> also shows `bfloat16` there. The authoritative runtime dtype for each run is in the app's
> `local_env.json` (`"dtype": "float32"` for Exp B). This was verified directly.
> Run directory names carry each machine's **local** time; the two cloud runs are UTC.
> Neither platform honoured the prompt's key or tempo hints (C#/Db major, 116 BPM): CUDA drew
> Fm/130, MPS drew E/126. That is a model property, not a platform one.

---

## 4. Results

### 4.1 Composition — the headline difference

| Take | Key | BPM | Chords | Distinct | Top chord % | Longest run of one chord | Sung vocal bars | Distinct sung bars | Longest run of identical sung bars |
|---|---|---|---|---|---|---|---|---|---|
| L4 · torch 2.10 · bf16 *(reference)* | **Fm** | **130** | 155 | 5 | 27 % | 4 | 121 | **49 %** | 2 |
| L4 · torch 2.14 · bf16 *(Exp A)* | **Fm** | **130** | 155 | 5 | 27 % | 4 | 121 | **49 %** | 2 |
| L4 · FP8 *(calibration)* | **Fm** | **130** | 182 | 5 | 32 % | 5 | 117 | 45 % | 2 |
| M1 Max · bf16 *(Control 1)* | **E** | **126** | 143 | 4 | 47 % | 13 | 124 | 35 % | 5 |
| M4 Pro · bf16 *(Control 2)* | **E** | **126** | 144 | 5 | 53 % | 19 | 117 | 35 % | 3 |
| M4 Pro · fp32 *(Exp B)* | **E** | **126** | 151 | 4 | 60 % | 14 | 122 | 39 % | 3 |

*Sung vocal bars* are bars of the vocal voice that contain at least one note; "distinct" and
"longest run" are computed on the bar text with chord symbols stripped, so a one-chord intro of
rests does not count as repetition.

#### Where the runs part — first differing token

| Pair | `abc_tokens.npy` (of ~3 000) | `score.abc` | `semantic.npy` |
|---|---|---|---|
| L4 2.10 vs L4 2.14 (Exp A) | **identical** | identical | **identical** |
| L4 2.10 vs L4 ODE 48 / budget 16 | identical | identical | identical |
| L4 bf16 vs **L4 FP8** (same RNG stream, quantized weights, eager path) | **225** | line 17 (second `V: Ins` line of the intro) | 2 |
| **M1 Max vs M4 Pro** (same RNG stream, both bf16) | **122** | line 13 (first `V: Ins` line of the intro) | 4 |
| M4 Pro bf16 vs **fp32** (same RNG stream, same chip) | **122** (the M4 bf16 take is the one that flips; M1 bf16 and M4 fp32 still agree there) | line 13 | 4 |
| M1 Max bf16 vs M4 Pro fp32 | 313 | line 24 | 21 |
| L4 vs any MPS take (different RNG streams) | **26** | line 5: `Q:1/4=130` vs `Q:1/4=126` | 1 |

The semantic stage is conditioned on the (already different) score and re-seeds with the same
seed, so its first-divergence index is near zero for every pair except Exp A; the ABC index is
the informative one.

First 24 chord symbols of each take (the intro is where the divergence is already visible):

```
L4 (both torch versions): Fm Fm Ab Eb Eb7 Fm Ab Eb Eb7 Fm Ab Eb Eb7 Fm Ab Eb Eb7 Fm Ab Eb Eb7 Fm Ab Eb
M1 Max  bf16:             E  E  E  E  E  E  E  E  E  E  E  E  E  D  D  Amaj7 Amaj7 E  E  E  E  D  D  A
M4 Pro  bf16:             E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  D  A  E  E  E
M4 Pro  fp32:             E  E  E  E  E  E  E  E  E  E  E  E  E  D  D  A  A  E  E  E  E  D  D  A
```

Chord progression by section (consecutive duplicates collapsed):

| Section | L4 (Fm/130) | M1 Max | M4 Pro bf16 | M4 Pro fp32 |
|---|---|---|---|---|
| intro | Fm Ab Eb Eb7 ×4 | E | E | E |
| verse | Fm Ab Eb Eb7 → Fm Ab Eb Db | E D A(maj7) ×4 | E D A ×4 | E D A ×4 |
| pre-chorus | Db Eb Db Eb Db | E D A ×2 | C#m B C#m B A ×2 | C#m A E D ×2 |
| chorus | Fm Ab Eb Db ×… | E D A ×… | E D A ×… | E D A ×… |
| interlude | Fm Ab Eb Db ×2 | *(none)* | E | E |
| bridge | Fm Ab Eb Db ×4 | E D A ×3 | E D A ×2 | E D A ×2 |
| outro | Fm Ab Eb Db ×4 | E D A E | E | E |

Section order: reference, M4 Pro bf16 and FP8 = intro · verse · pre-chorus · chorus · interlude ·
verse · pre-chorus · chorus · bridge · chorus · outro. M1 Max has no interlude; M4 Pro fp32
places the interlude after the second chorus. Every take realises the requested
verse / pre-chorus / chorus / bridge / outro form.

Complete chord inventories (symbol × occurrences):

| Take | Chords |
|---|---|
| L4 (both torch versions) | Db ×42, Eb ×38, Fm ×35, Ab ×34, Eb7 ×6 |
| M1 Max bf16 | E ×67, A ×38, D ×36, Amaj7 ×2 |
| M4 Pro bf16 | E ×76, A ×40, D ×20, C#m ×4, B ×4 |
| M4 Pro fp32 | E ×91, D ×28, A ×28, C#m ×4 |

Chord-set similarity (Jaccard index on the set of distinct chord symbols):

|  | L4 2.10 | L4 2.14 | M1 Max | M4 Pro bf16 | M4 Pro fp32 |
|---|---|---|---|---|---|
| **L4 2.10** | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **L4 2.14** | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **M1 Max** | 0.00 | 0.00 | 1.00 | 0.50 | 0.60 |
| **M4 Pro bf16** | 0.00 | 0.00 | 0.50 | 1.00 | 0.80 |
| **M4 Pro fp32** | 0.00 | 0.00 | 0.60 | 0.80 | 1.00 |

**Reading.**

1. *Two families, decided at token 27.* The ABC header (`X:1 T: M:4/4 L:1/32 Q:1/4=1`) is
   near-deterministic and identical on every platform; the tempo digit is the first genuinely
   random decision, and the CUDA and CPU streams make it differently. Everything downstream —
   key, progression, melody style — is conditioned on that decision. The seed therefore picks a
   *family*, and every take of a given platform with this seed lives in that platform's family.
2. *The MPS takes are not near-copies.* They share a prefix of 122–313 ABC tokens (the header,
   the vocal-intro rests and the first riff bar) and then diverge completely; their score hashes
   all differ. Their remaining similarity (same key/tempo, same E–D–A progression, Jaccard
   0.5–0.8) is inherited from the shared prefix, exactly as the FP8 CUDA take inherits Fm/130
   and the Fm–Ab–Eb progression from its 225-token shared prefix with the reference.
3. *A token-level divergence is the normal response to any numeric perturbation.* FP8 weight
   quantization on the same GPU and RNG stream flips a token at 225; a different Apple GPU
   generation flips one at 122; float32 on the same chip flips one at 122. Single samples of a
   heavy-tailed quantity cannot rank these perturbations, but they put them in one regime:
   small, non-zero, and amplified by the autoregressive loop. Exp A (identical tokens) is the
   only pair with a *zero* perturbation.
4. *"Harmonically static" is a description of the E/126 family, not a defect.* The CUDA take is
   a four-chord i–♭III–♭VII–♭VI progression in F minor (Fm–Ab–Eb–Db, with Eb7 as ♭VII7 in
   the intro/verse) and a distinct pre-chorus; the MPS family is a three-chord
   I–♭VII–IV rock song in E over a one-chord riff intro of 9–13 bars — both idiomatic for the
   requested style. Top-chord share and longest chord run mostly measure the riff intro.
5. *The one composition-level measure that separates the platforms and matches the listening
   notes is vocal-melody repetition*: 49 % / 45 % distinct sung bars on the two CUDA draws vs
   35 % / 35 % / 39 % on the three MPS draws, with longer runs of identical bars (5 / 3 / 3 vs
   2 / 2). Because the MPS draws share the E/126 prefix, this cannot be attributed to the
   platform rather than to the family (§8).

### 4.2 Token level

| Take | semantic tokens | unique | entropy (bits) | repeated-token share¹ | abc tokens | unique abc |
|---|---|---|---|---|---|---|
| L4 · 2.10 bf16 | 7 123 | 4 682 | 11.91 | 53.0 % | 3 073 | 81 |
| L4 · 2.14 bf16 | 7 123 | 4 682 | 11.91 | 53.0 % | 3 073 | 81 |
| L4 · FP8 | 7 846 | 5 116 | 12.04 | 53.9 % | 3 311 | 84 |
| M1 Max bf16 | 6 665 | 4 452 | 11.85 | 51.6 % | 2 686 | 78 |
| M4 Pro bf16 | 6 677 | 4 468 | 11.86 | 51.9 % | 3 017 | 79 |
| M4 Pro fp32 | 7 155 | 4 673 | 11.92 | 53.8 % | 2 839 | 79 |

¹ share of tokens whose ID occurs more than once in the stream.

All latent tensors are finite on every platform (no NaN/Inf anywhere). Token entropy is
essentially equal across platforms — **the difference is not a degenerate or corrupted token
stream; it is a different, comparably-rich composition.** Exactly repeated 4-grams are
essentially absent from every semantic stream (0–1 per run), and the longest run of one
repeated semantic token is 2 everywhere.

### 4.3 Artifact hashes

| Take | `score.abc` | `semantic.npy` | `latent.npy` | `audio.flac` |
|---|---|---|---|---|
| L4 · 2.10 bf16 | `db2d1467174a7d71…` | `e8a09373bc99f11e…` | `f3e90f8968bbd16d…` | `07f934932f357eca…` |
| L4 · 2.14 bf16 | `db2d1467174a7d71…` | `e8a09373bc99f11e…` | `f3e90f8968bbd16d…` | `a824641c442d9e9e…` |
| L4 · FP8 | `e1bd6badaa6fc6a8…` | `fa8193769086779a…` | `6132b6e06b7b3f7b…` | `41756563b76434f3…` |
| M1 Max bf16 | `04eb164b3d73782c…` | `6a33a837f9777a3f…` | `d670a9c5a98c2996…` | `57201bbd19154f43…` |
| M4 Pro bf16 | `5ba38106abfe35ce…` | `99d780bef2169f4f…` | `d79e37053643c9ba…` | `6ed915a459baf8ff…` |
| M4 Pro fp32 | `d996dd11c604f0c0…` | `0bf1dac96dc78b68…` | `51cac367a5d0d925…` | `745c85d4a623ba73…` |

Full SHA-256 values are in Appendix A. Note the CUDA pair: **identical plan, tokens *and*
latents, different audio** — the torch build changed nothing in the AR or NAR stages; the
difference is confined to the fp32 VAE decode (cuDNN cu128 vs cu130) and is measured in §4.4.

### 4.4 Audio-level measurements

All values recomputed from the archived `audio.flac` files for this revision (estimators in
§2.3). Earlier versions of this table used a different estimator and mis-stated the fp32 take.

| Take | Duration | Peak | RMS (mono mix) | Crest | Spectral centroid | Magnitude > 8 kHz | Stereo corr. | 3-s RMS σ | P10–P95 range |
|---|---|---|---|---|---|---|---|---|---|
| L4 · 2.10 bf16 | 284.9 s | 1.000 | 0.155 | 16.2 dB | 2 895 Hz | 10.4 % | 0.845 | 5.5 dB | 14.1 dB |
| L4 · 2.14 bf16 | 284.9 s | 1.000 | 0.155 | 16.2 dB | 2 895 Hz | 10.4 % | 0.845 | 5.5 dB | 14.1 dB |
| L4 · FP8 | 313.8 s | 1.000 | 0.141 | 17.0 dB | **3 373 Hz** | 12.4 % | 0.804 | 4.1 dB | 11.3 dB |
| M1 Max bf16 | 266.6 s | 1.000 | 0.143 | 16.9 dB | 3 463 Hz | 15.3 % | 0.867 | 2.7 dB | 6.0 dB |
| M4 Pro bf16 | 267.1 s | 1.000 | 0.142 | 17.0 dB | 3 551 Hz | 15.8 % | 0.857 | 5.4 dB | 12.3 dB |
| M4 Pro fp32 | 286.2 s | 1.000 | 0.165 | 15.7 dB | **3 010 Hz** | 10.8 % | 0.845 | 2.7 dB | 4.2 dB |

Power share per band (whole song, mono): bass 60–150 Hz = 46.6 % (L4) · 52.5 % (FP8) ·
37.7 % (M1) · 37.5 % (M4 bf16) · 45.7 % (M4 fp32); mid 400 Hz–1 kHz = 17.9 % · 15.5 % ·
24.2 % · 21.8 % · 21.2 %.

Loudness envelope, full 10-s windows only (a trailing partial window is dropped), dB relative
to each song's own RMS (the prompt asks for "quiet-loud dynamics"):

```
L4 ref   -14 -11 -13  -6  -2  +0  -1  +0  +1  +2  +1  +0  +1  +1  +0  -1  +1  +1  +2  -0  -1  +1  +2  +2  +2  +1  +1 -13   (28 windows)
L4 FP8   -12 -11  +0  +1  +0  +1  +1  +1  +0  +1  +1  +1  -2  -9  -0  +0  +0  +0  +1  +1  +1  -3  +0  +1  +1  +1  +1  +2  -6  -2  +1   (31 windows)
M1 bf16   -9  -3  -1  +0  -1  -4  +1  +1  +1  +1  -1  +0  -0  -3  +0  +2  +0  +0  +1  -1  -1  +0  +2  +2  +2  -0   (26 windows)
M4 bf16  -14 -10  -6  -2  +0  +1  +2  +2  +2  -4  -1  +0  +0  +0  +1  +2  +2  +1  -6  +0  +1  -2  +2  +2  +2  -0   (26 windows)
M4 fp32   -9  -4  -2  -0  -0  -0  +1  +1  +0  -2  -1  -1  +0  +0  +0  +1  +2  +2  -1  +1  -2  -1  +1  +1  +2  +2  -1  +0   (28 windows)
```

The dropped tails are 6.6 s (M1 Max, −5 dB), 7.1 s (M4 Pro bf16, **−16 dB** — its quiet outro) and
6.2 s (M4 Pro fp32, −2 dB); the two CUDA takes end on a full window.

Sample-level comparisons where the token streams are identical:

| Pair | max \|Δ\| (full-scale) | RMS Δ | SNR | Note |
|---|---|---|---|---|
| L4 2.10 vs L4 2.14 (identical latents) | 6.4 × 10⁻⁶ | 1.8 × 10⁻⁷ | **119 dB** | 43 % of samples bit-identical; below 16-bit quantisation noise — inaudible by construction |
| L4 ODE 32 vs ODE 48 (identical tokens, different solver) | 0.33 | 6.4 × 10⁻³ | 28 dB | latent RMS Δ = 2.3 % of latent std — the "audible but subtle" reference point |

**Reading.** No clipping anywhere; levels are comparable. The two bf16 Mac takes are brighter
and bass-lighter than the CUDA reference — but the **FP8 take generated on the same L4 is just
as bright**, and the **M4 fp32 take is as dark and as bass-heavy as the reference**. Brightness
and bass share therefore vary between *draws* on both platforms; they are not a platform
rendering signature. The same holds
for dynamics: the M4 Pro bf16 take reproduces the reference's quiet intro / quiet outro
architecture almost exactly (P10–P95 12.3 vs 14.1 dB), while the M1 Max and M4 fp32 takes are
flat (6.0 / 4.2 dB). Note that the listener ranked M4 bf16 *with* M1 Max, not with the
reference — so the non-blind verdict was not tracking macro-dynamics or harmonic structure
either, which pointed at the performance and rendering layers rather than the score. The
follow-ups then cleared both layers of a platform effect (§9.3, §9.5), leaving the draw
itself — which composition and which performance the RNG produced — as the only variable
that the verdict can have been responding to.

### 4.5 Performance

| Take | ABC | Semantic AR | NAR | VAE | End-to-end | RTF¹ |
|---|---|---|---|---|---|---|
| L4 · 2.10 bf16 | 59.7 s · 51.5 tok/s | 181.3 s · 39.3 tok/s | 69.4 s | 14.4 s | **324.8 s** | **1.14** |
| L4 · 2.14 bf16 | 59.7 s · 51.5 tok/s | 181.2 s · 39.3 tok/s | 70.0 s | 20.6 s | 331.5 s | 1.16 |
| M1 Max bf16 | 105.6 s · 25.4 tok/s | 589.9 s · 11.3 tok/s | 817.9 s | 12.2 s | 1 525.6 s | 5.72 |
| M4 Pro bf16 | 94.8 s · 31.8 tok/s | 550.7 s · 12.1 tok/s | 575.5 s | 10.4 s | 1 231.4 s | 4.61 |
| M4 Pro fp32 | 142.8 s · 19.9 tok/s | 1 037.7 s · 6.9 tok/s | 842.5 s | 12.5 s | 2 036.1 s | 7.11 |

¹ Real-time factor = end-to-end seconds per second of generated audio.

Ratios versus the CUDA reference:

| Take | End-to-end | NAR alone | Semantic AR (per token) | ABC (per token) |
|---|---|---|---|---|
| M1 Max bf16 | **4.70× slower** | **11.8× slower** | 3.5× slower | 2.0× slower |
| M4 Pro bf16 | **3.79× slower** | **8.3× slower** | 3.2× slower | 1.6× slower |
| M4 Pro fp32 | **6.27× slower** | **12.1× slower** | 5.7× slower | 2.6× slower |

Observations:

- **NAR (flow-matching ODE synthesis) is the worst stage on MPS** — 8–12× slower — while VAE
  decode is equal or faster on Apple GPUs (10.4–12.5 s vs 14.4 s).
- **fp32 costs 1.65×** end-to-end on the M4 Pro (1 231 s → 2 036 s) and changes nothing about
  the family (§5).
- The M4 Pro is ~1.24× faster than the M1 Max overall and 1.42× faster in NAR, consistent
  with newer GPU generations having better Metal kernels for this workload.

### 4.6 Memory (observed peaks)

| Take | Working memory observed | Where |
|---|---|---|
| L4 · CUDA (bf16) | ≈ 10.5 GiB peak VRAM (10 780 MiB) | `nvidia-smi`, 5 s sampling |
| M1 Max (bf16) | ≈ 10.3 GB GPU allocation | `ioreg` "In use system memory" |
| M4 Pro (bf16) | ≈ 10.5 GB GPU allocation | `ioreg` |
| M4 Pro (fp32) | 11.8–20.9 GB GPU allocation | `ioreg` |

fp32 doubles the weights (7.3 GB → 14.6 GB) and visibly increases the reserve the MPS
allocator holds. See [LINUX_CUDA.md](LINUX_CUDA.md) for the full VRAM-budget analysis on CUDA
(including the 12 GiB OOM boundary and FP8).

### 4.7 Listening evaluation

Single experienced listener (the author), non-blind, same playback setup. The test song is
one this listener has used repeatedly for cross-platform quality assessment.

| Take | Verdict (as reported) |
|---|---|
| L4 · 2.10 bf16 (reference) | clearly better than the Mac takes; prompted the investigation |
| L4 · 2.14 bf16 (Exp A) | *"very good — vocals, music, lyrics, arrangement, instruments and narrative direction all correct; high quality"* |
| M1 Max bf16 | *"clearly worse than the Linux/CUDA version — obvious by ear"* (vocals, music, arrangement, instruments) |
| M4 Pro bf16 | *"about the same as the M1 Max take; clearly worse than Linux/CUDA"* |
| M4 Pro fp32 (Exp B) | **not evaluated by ear** |
| L4 · FP8 | not part of the listening comparison |

The listening verdicts were reported in Chinese and are translated here.

**How much does this prove?** Less than it appears to, for a structural reason: the listener
compared **one composition family against another**. Every CUDA take is Fm/130 with the
Fm–Ab–Eb progression; every MPS take is E/126 with E–D–A. The verdict "CUDA better" is
indistinguishable from "this listener prefers the Fm/130 realisation of this song", and the
verdict "the two Macs sound alike" is expected from their shared 122-token prefix regardless of
platform. The only measurable difference that lines up with the notes on *vocals* is the
vocal-melody repetition in §4.1 — and it, too, is inherited by all three MPS takes from the same
prefix. The experiment that separates platform from family is §9.1; its blind ratings are in
§9.5, and they do not reproduce a Mac deficit once the composition is held fixed.

---

## 5. Hypothesis testing

### Exp A — "the torch version explains it" → **rejected (zero perturbation)**

Same GPU (L4), same dtype, same request; only torch changes (2.10.0+cu128 → 2.14.0+cu130,
i.e. the Linux pin vs the macOS pin):

| Artifact | torch 2.10 | torch 2.14 | Result |
|---|---|---|---|
| `abc_tokens.npy` / `score.abc` | `73e6a905…` / `db2d1467…` | same | **byte-identical** |
| `semantic.npy` | `e8a09373…` | same | **byte-identical** |
| `latent.npy` (NAR output) | `f3e90f89…` | same | **byte-identical** |
| `audio.flac` | `07f93493…` | `a824641c…` | PCM max \|Δ\| 6.4 × 10⁻⁶, SNR 119 dB (§4.4) |

The composition is **torch-version independent on CUDA**, and so is the acoustic latent.
Upstream pins deterministic kernels (`cudnn.deterministic`, TF32 off, `float32_matmul_precision
= "highest"`), and on the same GPU the two builds evidently selected bit-equivalent kernels for
the AR and NAR paths. What Exp A does **not** show is that CUDA sampling is robust to numeric
perturbation: the perturbation it applied to the token path was zero. The FP8 calibration run
below is the CUDA data point for a real perturbation.

### Exp B — "numeric precision explains it" → **rejected as an explanation of the divergence; not evaluated as a quality lever**

Same machine (M4 Pro), same request; only dtype changes (bfloat16 → float32), verified in
`local_env.json` and by the doubled GPU allocation:

| | M4 Pro bf16 | M4 Pro fp32 |
|---|---|---|
| Key / tempo | E / 126 | **E / 126** |
| First differing ABC token | — | **122** |
| Distinct chords / top share | 5 / 53 % | 4 / 60 % |
| Chord-set similarity vs bf16 | — | 0.80 |
| Spectral centroid / bass share | 3 551 Hz / 37.5 % | **3 010 Hz / 45.7 %** (closest to the CUDA reference) |
| End-to-end | 1 231 s | 2 036 s (1.65×) |
| Listening | "clearly worse than CUDA" | **not evaluated** |

With the RNG stream held constant (both runs sample on CPU), fp32 flips a token at ABC index
122 and produces a different take of the same family — the same behaviour as every other
non-zero perturbation in this study. Precision is therefore not what puts MPS in the E/126
family (the tempo token at index 27 does that). Whether fp32 *sounds* better was not tested;
its audio measurements (§4.4) happen to be the closest of the Mac takes to the CUDA reference,
which is a reason to include it in the blind comparison of §9.5, not a conclusion.

### Cross-check — "the GPU generation explains it" → **rejected as an explanation of the divergence**

M1 Max (32-core, 2021) and M4 Pro (20-core, 2024) are different GPU generations with
different Metal capabilities. On the same RNG stream they agree for **122 ABC tokens** and then
part; their scores share key, tempo, the E–D–A progression and a 0.50 chord-set overlap because
those were decided inside the shared prefix. They are two different takes of one family, not
"the same take with small numeric variations".

### Calibration — FP8 on the reference GPU (from LINUX_CUDA.md)

Same L4, same RNG stream, weights quantized to FP8 and the eager + SDPA execution path
(CUDA graphs are disabled for FP8): first differing ABC token **225**; Fm/130 with the same
Fm–Ab–Eb–Eb7 intro; a different chorus (Fm–Db–Ab–Eb); centroid 3 373 Hz, i.e. nearly as
bright as the Mac takes (3 463–3 551 Hz) against the reference's 2 895 Hz. This is what a large, known numeric perturbation does on CUDA,
and it is qualitatively identical to what a different Apple GPU or a different dtype does on
MPS. Token-level divergence is therefore not evidence of an MPS-specific defect.

### What remains, and what is still open

- **Explained:** the *divergence* between CUDA and MPS. The platforms sample from different
  random streams by design (§6.1); the streams part at the first random token (tempo), which
  places the platforms in different composition families.
- **Explained:** why the Mac takes resemble each other — a shared prefix on a shared RNG stream,
  not small numerics. Numerics on MPS are large enough to flip a token within ~120 draws, as
  FP8 is on CUDA within ~225.
- **Answered afterwards, as far as one listener can answer it:** whether MPS draws are *worse
  on average*. This five-take design cannot answer it (platform and family are confounded, §8);
  the fixed-composition blind test can, and found no deficit (§9.1, §9.5). The numerical path
  differences in §6.2–6.3 are bounded for the acoustic stage (§9.3) and remain unmeasured for
  the AR stage (§9.4) — but their perceptual effect, if any, is below this listener's 1-point
  noise floor.

---

## 6. Platform differences in the code path

### 6.1 Sampling RNG device — **code-verified, and sufficient to explain the divergence**

```python
# yue2/sampling.py — generate_tokens()
rng_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
generator = torch.Generator(device=rng_device).manual_seed(seed)
...
if sampling.temperature == 0:
    next_id = scores.argmax(-1, keepdim=True)
else:
    probabilities = scores.softmax(-1)
    if device.type == "mps":
        next_id = torch.multinomial(probabilities.cpu(), 1, generator=generator).to(device)
    else:
        next_id = torch.multinomial(probabilities, 1, generator=generator)
```

| | CUDA | MPS |
|---|---|---|
| Generator device | CUDA device generator | **CPU generator** (probabilities moved to CPU for the draw) |
| Same seed gives | CUDA RNG stream (Philox) | CPU RNG stream (MT19937) |
| `torch.multinomial` implementation | CUDA kernel | CPU kernel (different consumption of the stream) |
| Streams equal? | **No** | |

Measured on the M1 Max (torch 2.14.0) — same seed, same probability vector, eight draws:

```
CPU generator : 239, 131, 556, 728, 260, 385,  69, 707
MPS generator : 186, 753, 776, 703, 229, 980, 485, 545     # different stream
CPU generator re-run: identical to the first CPU list      # reproducible
```

Note that the **NAR stage is different**: its initial noise is drawn with a CPU generator on
every platform (`yue2/nar.py`, `torch.Generator(device="cpu")`), so for a given token stream
the acoustic stage has RNG parity across platforms. This is what makes §9.3 a clean test.

**Consequences**

1. The same seed **cannot** produce the same song on CUDA and on MPS. This is a property of
   the sampler's RNG-device selection, not of the model.
2. The first 26 ABC tokens agree anyway, because the ABC header is near-deterministic; the
   streams part at the first genuinely random token (the tempo), which puts each platform in
   its own composition family (§4.1).
3. Two Macs on the same CPU stream agree until their kernel numerics first flip a sampled
   token (ABC index 122 here) and then diverge completely — same family, different take.
4. An apples-to-apples *platform* comparison therefore cannot rely on a shared seed at all; it
   has to hold the composition fixed explicitly (§9.1).

### 6.2 Attention and execution path — different by design (secondary, unquantified)

Upstream's `sdpa()` wrapper (`yue2/modeling_yue2.py`) selects different implementations per
device:

```python
def sdpa(query, key, value, *, attn_mask=None, is_causal=False):
    grouped = query.shape[1] != key.shape[1]
    if grouped and query.device.type == "mps":
        # PyTorch's MPS attention does not implement enable_gqa on every release.
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        grouped = False
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=grouped
    )
```

| | CUDA (bf16) | MPS |
|---|---|---|
| Grouped-query attention | native (`enable_gqa=True`) | K/V expanded with `repeat_interleave`, then ungrouped |
| AR execution (as recorded) | `cuda_graph` — `GraphAR`, native variable-length FlashAttention, both CFG branches in one batch (`GraphAR`'s optional projection fusion is off by default and never enabled by the sampler) | `eager` — `StaticKVCache`, MPS SDPA, CFG branches as two batch-1 forwards |
| NAR attention | one SDPA call over the whole query block | queries processed in 256-row blocks |

The math is equivalent; the numerics are not. The model runs ~24 000-token contexts through
this path, and the AR decode on MPS goes through the single-query half-precision kernel whose
out-of-bounds defect (fixed in torch 2.11) is documented in [MACOS_MPS.md](MACOS_MPS.md).
The project's `scripts/mps_sdpa_check.py` is a **gross-error and determinism check** (tolerance
0.05 against an fp64 reference, key lengths up to 2 048, random inputs); it does not establish
precision parity with the CUDA kernels at the model's real context lengths. Nothing in this
study measures the magnitude of these differences at the logit level; §9.4 does.

### 6.3 Kernel libraries

cuBLAS / cuDNN / Flash-Attention vs Metal Performance Shaders / MPSGraph: different reduction
orders, tiling, and algorithm selection for bf16 and fp32. Not directly measured here; listed
for completeness.

### 6.4 What has been excluded as a mechanism

- **Model weights** — SHA-256 verified identical in all runs.
- **Tokenizer / text handling** — same request text, same `yue2-infer` runtime hash, byte-identical 884-token request prefix in every run.
- **Degenerate sampling on MPS** — token entropy and repetition are comparable across
  platforms (§4.2); no NaN/Inf in any latent tensor.
- **torch build** — tested, bit-exact through NAR (§5).
- **A platform-specific rendering signature in brightness or dynamics** — the FP8 CUDA take
  and the M4 fp32 take break the pattern in both directions (§4.4), and re-rendering the CUDA
  tokens on the M4 Pro leaves the spectrum unchanged to four significant figures (§9.3).
- **The VAE** — cross-platform decode of identical latents differs at −110 dB (§9.3).
- **The NAR as a large effect** — cross-platform synthesis from identical tokens and noise
  differs by the same amount as CUDA's own ODE 32 → 48 change (§9.3); anything beyond
  "subtle" is excluded — and blind, the Mac render was indistinguishable from the CUDA render
  (§9.5).
- **A platform deficit in the semantic stage large enough to hear** — five Mac performances of
  the CUDA composition were rated across the whole scale around the CUDA performance, best take
  included (§9.5). A deficit smaller than one rating point on a 5-point scale is not excluded.

---

## 7. Practical implications

### For macOS users of this UI

- **Same seed ≠ same song on a Mac vs on CUDA — by design.** The Mac samples with the CPU
  RNG; CUDA samples with the CUDA RNG. Use this deliberately: a seed you like on one platform
  does not reproduce on the other.
- **Same seed ≠ same song across Macs either.** The M1 Max and M4 Pro takes share only 122 ABC
  tokens. Treat a seed as reproducible only on one machine with one torch build and one dtype —
  there it *is* reproducible, bit-exact through the FLAC (verified on the M4 Pro, §9.2).
- **Seed re-rolling is the practical lever** — the first random token decides the family, so
  different seeds explore genuinely different songs.
- **`float32` is not a documented quality mode.** It costs ~1.65× time, produces a different
  take of the same family, and has not been evaluated by ear. Do not reach for it to "fix"
  quality on the strength of this document.
- **MPS is slower, most of all in NAR** (8–12× vs CUDA; a ~4.5-minute song takes ~20 minutes
  on an M4 Pro, ~25 minutes on an M1 Max). Memory is modest: ~10.5 GB of unified memory at
  bf16.
- If you want to compare platforms *like for like*, fix the score (§9.1) — a same-seed
  comparison compares two families, not one song rendered twice.
- **A Mac take is not a worse take.** With the composition fixed, Mac performances were rated
  across the whole scale around the CUDA one, and the Mac rendering of a CUDA performance was
  indistinguishable from the CUDA rendering (§9.5). If a Mac take disappoints, re-roll the seed;
  do not switch platforms expecting a systematic gain.

### For the project

- Document macOS as **supported, with a different sampling stream by design**; do not promise
  cross-platform or cross-machine seed parity.
- The comparison tooling (`05 // TOOLS`) is only meaningful **within** one machine, or across
  machines with an explicitly fixed score.
- Record the sampling RNG device in `result.json` so runs are self-describing (it is currently
  implicit in the device).
- A "first differing token" comparison between two runs' `abc_tokens.npy` / `semantic.npy`
  would have exposed the M1 ≠ M4 divergence immediately; it is a cheap addition to `05 // TOOLS`.
- If cross-platform *sampling parity* were ever wanted, the `rng_device` choice in
  `yue2/sampling.py` is the single place to change — but note that it would only align the
  stream, not the kernels, so takes would still part at the first numerically close call.
- The ODE-step A/B finding in [LINUX_CUDA.md](LINUX_CUDA.md) is unaffected: on one platform
  and one dtype, changing ODE steps leaves the plan and tokens byte-identical and changes only
  the acoustic solver (SNR 28 dB vs the 32-step take).

---

## 8. Limitations

1. **Platform is confounded with composition family.** One request, one seed; the CUDA stream
   puts every CUDA take in Fm/130 and the CPU stream puts every MPS take in E/126. No
   comparison in this document — listening, harmony, brightness, dynamics, melody repetition —
   can separate "MPS" from "the E/126 realisation of this song". More takes of this seed would
   not help (they stay in their family); different seeds would, but only in aggregate, with
   enough of them to average over families — fixing the score (§9.1) removes the confound
   outright, and that is the follow-up that was run (§9.1, §9.5). This limitation applies to
   the five-take comparison in §4, not to the fixed-score result.
2. **The three MPS takes are not independent samples.** They share a 122–313-token ABC prefix
   (header, tempo, key, intro riff) on one RNG stream. After the prefix they are independent
   continuations, but of the same family.
3. **The §4.7 ranking is one listener, not blind, and the fp32 take was not listened to.** It
   was made knowing which take came from where; the wider impression from routine Mac use was
   not recorded under controlled conditions. The §9.5 test is blind but still one listener in
   one session, with a measured repeatability of ±1 point on a 5-point scale (identical audio
   rated 3 and 4); it can rule out a large platform deficit, not a small one. The fp32 take
   belongs to the E/126 family and was not part of the fixed-score bundle.
4. **Output-level, not op-level.** No per-token logit comparison was performed, so §6.2–6.3
   remain candidates rather than a measured mechanism for any residual numeric effect — a
   question that is now academic for quality (§9.5) but still open as a measurement (§9.4).
5. **The blind ratings so far compare one CUDA performance against five Mac performances.**
   The rated bundle has a single CUDA-sampled performance (the reference, rated once directly
   and twice through its re-renders) against five Mac-sampled ones — a point against a
   distribution. Five CUDA seeds on the same score now exist (§9.1) and are packed with the five
   Mac takes as a second blind bundle, not yet rated; at the signal level the two sets are
   indistinguishable (§9.1).
6. **Exp A is a null test of numeric robustness.** It shows torch 2.10 and 2.14 are bit-exact
   on this GPU, not that sampling tolerates perturbation; the FP8 run is the only CUDA
   perturbation data point, and it is a single sample of a heavy-tailed quantity (first flip
   index).
7. **Same-machine reproducibility on MPS was verified once, on one machine.** A repeat of the
   fixed request on the M4 Pro reproduced every artifact byte for byte (§9.2). The M1 Max has
   not been repeated, and other torch builds are untested.
8. **Two Apple GPUs, one macOS version, one torch build on the Mac side.** macOS 26.6.2 and
   torch 2.14.0 throughout; other combinations are untested.
9. **Memory numbers** come from periodic sampling (5 s on CUDA, 20 s on the Macs), so brief
   peaks may be missed; treat them as working-set figures, not exact maxima.
10. **The CUDA-vs-CPU random-stream difference is inferred** from the code path, from a
   locally measured CPU-vs-MPS generator difference, and from documented PyTorch behaviour
   (device-specific generator algorithms); it is corroborated by the divergence at token 27.
   No CUDA-device RNG measurement was taken.
11. **Not tested:** macOS with torch 2.10 + the documented zero-mask workaround; MPS attention
    pinned to the math backend; any patch-level intervention.

---

## 9. Follow-up experiments (in order of value — 9.1, 9.2, 9.3 and 9.5 have been run)

### 9.1 Fixed-score comparison — the design that removes the confound → **done on both sides**

`request.abc` bypasses the ABC sampling stage entirely (`yue2/pipeline.py`, "Using provided
score"; the UI exposes it as the optional ABC input in `01 // GENERATE`, via `03 // EDIT`, and
as `abc_path` in `07 // BATCH`). Feed the **reference CUDA `score.abc`** as the exact score and
generate with several seeds: composition, key, tempo, progression and melody are then identical
everywhere; only the performance (semantic stage) and the rendering (NAR/VAE) are drawn — which
is exactly the layer the listening notes ("vocals, instruments, arrangement") point at.

**Run (M4 Pro, one `07 // BATCH` job, settings identical to Control 2).** Six rows:

- *Validation row* — Control 2's own `score.abc` with seed 831001. Its semantic prefix
  re-tokenised to Control 2's exactly (3 017 ABC ids, 3 903-token prefix), and the run reproduced
  Control 2 **byte for byte** (`abc_tokens.npy`, `prefix.npy`, `semantic.npy`, `latent.npy`,
  `audio.flac`), with the ABC stage skipped (`external_prefix_tokens: 3017`, 0 s). The
  fixed-score path is therefore equivalent to the generated path.
- *Five takes* — the reference's `score.abc` with seeds 831001–831005. Every one carries the
  reference's exact 3 959-token semantic prefix (`prefix.npy` = `41941da1…`); each is a new
  performance draw (first differing semantic token 0–2 against the reference).

| Take | semantic tokens | unique | entropy | Duration | RMS (mono) | Crest | Centroid | > 8 kHz | Bass 60–150 Hz | 3-s RMS σ | P10–P95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A · CUDA reference (CUDA performance, CUDA render) | 7 123 | 4 682 | 11.91 | 284.9 s | 0.155 | 16.2 dB | 2 895 Hz | 10.4 % | 46.6 % | 5.5 dB | 14.1 dB |
| B · Mac re-render of A (§9.3) | 7 123 | 4 682 | 11.91 | 284.9 s | 0.155 | 16.2 dB | 2 894 Hz | 10.4 % | 46.6 % | 5.5 dB | 14.1 dB |
| M · s831001 | 7 139 | 4 868 | 11.99 | 285.6 s | 0.163 | 15.8 dB | 3 956 Hz | 17.4 % | 40.0 % | 3.2 dB | 8.2 dB |
| M · s831002 | 7 157 | 4 824 | 11.98 | 286.3 s | 0.206 | 13.7 dB | 3 347 Hz | 14.0 % | 41.3 % | 3.6 dB | 8.4 dB |
| M · s831003 | 7 141 | 4 753 | 11.95 | 285.6 s | 0.128 | 17.9 dB | 3 576 Hz | 14.7 % | 42.8 % | 4.0 dB | 10.9 dB |
| M · s831004 | 7 108 | 4 762 | 11.96 | 284.3 s | 0.149 | 16.5 dB | 3 369 Hz | 12.4 % | 45.4 % | 3.2 dB | 5.1 dB |
| M · s831005 | 7 131 | 4 605 | 11.87 | 285.2 s | 0.194 | 14.2 dB | 3 032 Hz | 11.2 % | 55.5 % | 6.4 dB | 17.0 dB |

Estimators as in §2.3. Semantic timings were 594–600 s per take (≈ 12 tok/s) and NAR 629–633 s.

**Reading.** Token-level, the Mac performances are the reference's equals (length, diversity,
entropy; no truncation). Five of five are brighter than A — but A is the darkest take in the
whole pool of this document (the FP8 CUDA take sits at 3 373 Hz, inside the Mac range), and
under the null hypothesis that A and the five Mac takes are exchangeable, A being the darkest
has probability 1/6. Loudness and dynamics vary widely between Mac performances (RMS 0.128 to
0.206, P10–P95 5 to 17 dB): the semantic stage decides how dense the mix is, and it decides it
differently every draw. None of this is a quality judgement; that is §9.5.

**CUDA side (L4, second session, 2026-09-17; LINUX_CUDA.md's E10, seeds 831001–831005).**
The same five seeds with the same exact score on an L4. Seed 831001 reproduced the reference
`semantic.npy` / `latent.npy` / `audio.flac` **byte for byte** — the CUDA-side validation row
(same prefix, same RNG stream, same kernels; ABC stage skipped). The other four are new CUDA
performances of the composition:

| Take | semantic tokens | unique | entropy | Duration | RMS (mono) | Crest | Centroid | > 8 kHz | Bass 60–150 Hz | 3-s RMS σ | P10–P95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CUDA · s831001 (= A) | 7 123 | 4 682 | 11.91 | 284.9 s | 0.155 | 16.2 dB | 2 895 Hz | 10.4 % | 46.6 % | 5.5 dB | 14.1 dB |
| CUDA · s831002 | 7 145 | 4 770 | 11.95 | 285.8 s | 0.184 | 14.7 dB | **4 121 Hz** | 19.0 % | 45.3 % | 5.0 dB | 14.2 dB |
| CUDA · s831003 | 7 183 | 4 737 | 11.93 | 287.3 s | 0.146 | 16.7 dB | **4 298 Hz** | 19.7 % | 36.9 % | 2.8 dB | 7.1 dB |
| CUDA · s831004 | 7 152 | 4 785 | 11.98 | 286.1 s | 0.129 | 17.7 dB | 3 142 Hz | 11.9 % | 38.3 % | 4.2 dB | 9.5 dB |
| CUDA · s831005 | 7 087 | 4 839 | 11.98 | 283.5 s | 0.139 | 17.1 dB | 3 232 Hz | 12.4 % | 47.6 % | 5.5 dB | 15.0 dB |

**This closes the brightness question.** Five CUDA performances of the same score span
2 895–4 298 Hz; two of them are brighter than any of the five Mac performances (3 032–3 956 Hz).
Means: CUDA 3 538 Hz, Mac 3 456 Hz. Loudness, crest and dynamic range overlap the same way
(P10–P95: CUDA 7–15 dB, Mac 5–17 dB). Brightness, density and dynamics are properties of the
draw; the reference A was simply the darkest draw in either set. Nothing in these ten takes
separates the platforms at the token or signal level.

*Reproduction note.* An exact score must reach the tokenizer **byte for byte**: `score.abc`
ends with a newline and that newline is part of the last ABC token (7360 in the reference
run), so `encode(score.abc)` reproduces the original `abc_tokens.npy` and the original
semantic prefix only from the unmodified file. `07 // BATCH` with `abc_path` reads the file
raw and does reproduce it (verified against both the reference and Control 2); the ABC text
box in `01 // GENERATE` applies `strip()` and therefore yields a different last token and a
different take. Copy the file with `cp`, not through an editor.

### 9.2 Same-machine repeat — is MPS reproducible at all? → **done: bit-exact**

The fixed request was re-run on the M4 Pro with identical settings (`request.json` and
`config.json` identical to Control 2 apart from the run id). Every artifact matched Control 2
byte for byte:

| Artifact | Control 2 | Repeat |
|---|---|---|
| `abc_tokens.npy` | `cec10145…` | `cec10145…` |
| `semantic.npy` | `99d780be…` | `99d780be…` |
| `latent.npy` | `d79e3705…` | `d79e3705…` |
| `audio.flac` | `6ed915a4…` | `6ed915a4…` |

Timings agreed within 1 % (e2e 1 236 s vs 1 231 s). So on one Mac with one torch build and
one dtype the whole pipeline — AR sampling, NAR, and the VAE — is deterministic; a seed is a
valid label for a take there. The M1 ≠ M4 divergence in §4.1 is therefore a cross-chip kernel
difference, not run-to-run noise.

### 9.3 Cross-platform re-rendering — isolates the acoustic stage → **done: VAE cleared, NAR within an ODE-step of CUDA**

The NAR noise is CPU-generated on every platform (§6.1), so rendering the CUDA run's tokens
on a Mac uses the *same* noise and the *same* tokens; only kernel numerics differ. Both levels
were run on the M4 Pro (bf16, MPS, torch 2.14.0, weights verified against the reference run,
`ode_steps` / `context` / `vae_core_frames` read from the reference `config.json`) with a
stand-alone script; outputs are full run directories (`…035647-rerender-vae-…`,
`…040751-rerender-nar-…`) usable in `04 // LIBRARY` and `05 // TOOLS`.

| Level | Input from the CUDA reference | Sample-level vs reference `audio.flac` | Spectrum (ref → Mac) |
|---|---|---|---|
| **VAE only** | `latent.npy` | max \|Δ\| 5.6 × 10⁻⁵ · RMS Δ 5.2 × 10⁻⁷ · **SNR 109.8 dB** · 26.7 % samples identical | centroid 2 894.6 → 2 894.6 Hz · >8 kHz 10.389 → 10.389 % |
| **NAR + VAE** | `prefix.npy` + `semantic.npy`, seed 831001 | max \|Δ\| 0.45 · RMS Δ 6.0 × 10⁻³ · **SNR 28.6 dB** | centroid 2 894.6 → 2 894.0 Hz · >8 kHz 10.389 → 10.383 % · RMS 0.15458 → 0.15459 |
| *(calibration)* CUDA ODE 32 → 48 | same tokens, same GPU | RMS Δ 6.4 × 10⁻³ · **SNR 28.0 dB** | — |

Latents, NAR + VAE level: RMS Δ = 2.2 % of the reference latent's std, correlation 0.99975 —
against 2.3 % / 0.99973 for CUDA's own ODE 32 → 48 change.

**Reading.**

- The **VAE** differs across platforms at the −110 dB level (cuDNN vs Metal fp32 convolutions;
  cu128 vs cu130 on one GPU was −119 dB). It is excluded as a source of any audible difference.
- The **NAR** on MPS differs from CUDA by the same magnitude as changing the ODE step count
  on CUDA — a perturbation the listener rated as *"slightly better"* for 48 vs 32 steps
  (LINUX_CUDA.md §5), i.e. subtle. Its spectral balance is unchanged to four significant
  figures, so **the Mac takes' brightness (§4.4) cannot come from the acoustic stage**.
- ~28 dB is *expected* to be the floor any non-bit-exact NAR will show — a mechanism inference
  from two data points, not a measurement: 32 midpoint steps turn tiny velocity differences
  into phase differences that the sample-level SNR counts in full. If so, the metric cannot
  rank platform against solver-step; listening can, and for the first time the comparison is
  on identical music — reference vs `…040751-rerender-nar-…` in `05 // TOOLS`.
- SNR and max \|Δ\| are not additive across stages: "VAE high, NAR + VAE at 28 dB" means the NAR
  is the dominant contributor, not that the difference is exactly the NAR's.

**Solver steps and decoder precision (M4 Pro, one change at a time).** Two follow-on
measurements from the same harness — same tokens, same CPU-generated noise, seed 831001 — produced
by `scripts/ode_steps.py` (steps) and `scripts/vae_decoder_precision.py` (decoder):

| Change, against that machine's own 32-step render | Latent Δ / corr | vs the Mac 32-step render | vs the CUDA reference |
|---|---|---|---|
| NAR 32 → 16 steps | RMS Δ 5.6 % of std · corr 0.99842 | **21.1 dB** | 21.3 dB |
| NAR 32 → 8 steps | RMS Δ 9.4 % of std · corr 0.99557 | **16.6 dB** | 16.8 dB |
| 32-step latent rounded to bf16 before the fp32 VAE | **bit-identical** — 0 of 455 872 values | **133 dB** | 28.6 dB |

Re-render runs `…155954-rerender-nar8-…` and `…162741-rerender-nar16-…` — complete run
directories, finalized like the 2a/2b re-renders so `04 // LIBRARY` and `05 // TOOLS` accept them;
their NAR time is 164 s and 323 s against 629 s for 32 steps, i.e. linear in the step count. Decoder
precision was measured separately, holding the latent fixed (the reference run's
`latent.npy`, decoded by the standard fp32 VAE at core 1024 / halo 16; baseline = the fp32
decode of that same latent), with `scripts/vae_decoder_precision.py`:

| Decoder | SNR vs the fp32 decode of the same latent |
|---|---|
| fp16 autocast, fp32 weights | 59.1 dB |
| fp16 weights + activations | 54.3 dB |
| bf16 autocast, fp32 weights | 40.9 dB |
| bf16 weights + activations | 35.3 dB |

Every row needs a **freshly loaded** VAE. A half-precision pass leaves the module in a state that
shifts a later fp32-weight autocast run by ~2 dB (38.6 / 43.2 dB instead of 40.9 / 59.1), and
re-typing weights that a previous pass already rounded to bf16 reads an fp16 decoder as 43.4 dB
instead of 54.3 dB — the one number here that a one-pass measurement gets wrong.

Three things follow. The **latent is not a precision lever**: the NAR output already carries bf16
precision, so rounding it changes nothing — and the 133 dB residual is the PCM_24 container's own
quantisation (step 2⁻²³ against an RMS of 0.16 ≈ 133 dB), not a decode difference: two decodes of
the same latent in one process are bit-identical. The **decoder is the sensitive end** — bf16
costs 35.3 dB, and it is where a port that stores the VAE in half precision changes the audio in
exchange for a few hundred MB. And **halving the solver is a bigger change than the model's own
knob**: 16 and 8 steps move the render 21.1 and 16.6 dB, against 28.0 dB for 32 → 48 (measured on
the L4 with the CUDA 32-step render as baseline, so the two are not the same yardstick — but both
are one-variable changes against a same-machine or same-GPU baseline, and the ordering below
28 dB is unambiguous: 8 steps moved further than 16, which moved further than 48).

**Listening.** A first, non-blind A/B of the reference against the NAR + VAE re-render —
identical composition, identical performance tokens, identical noise — came out in favour of
the **Mac render**. The blind test in §9.5 then rated the reference 3, its Mac re-render 4 and
its VAE-only re-decode (identical audio to within 110 dB) also 4: the "preference" is the same
size as the listener's own repeat noise, so the honest statement is that **the Mac rendering is
indistinguishable from the CUDA rendering at this listener's resolution** — not better, and
certainly not worse. The reverse direction (Mac tokens re-rendered on CUDA) remains untested
and would need a CUDA machine.

Consequence: whatever made the E/126 Mac takes sound "clearly worse" is not rendering. §9.1
and §9.5 then tested the remaining candidate, the semantic stage, on a fixed composition.

### 9.4 Teacher-forced logit comparison — quantifies the numeric gap without sampling (open; no longer a quality question)

Force the CUDA run's token sequence through the model on each platform via the same
incremental `StaticKVCache` path the sampler uses (not a single prefill — the MPS decode kernel
differs from the prefill kernel), and record per-position top-k probabilities. Report
KL(p_CUDA ‖ p_MPS), argmax agreement and entropy difference **as a function of position**. A
KL that is small and flat means the platforms differ only by chaotic amplification of
negligible noise; a KL that grows with context length is the signature of an attention-
precision effect and would be a mechanism worth fixing. After §9.5 this is a measurement of
the numeric gap for its own sake — the perceptual question it was meant to serve is answered.

### 9.5 Blind listening on the fixed score → **done: no platform deficit**

**Bundle.** Eight takes — the seven of §9.1 (A, the CUDA reference; B, its Mac NAR + VAE
re-render; the five Mac performances) plus C, A's Mac VAE-only re-decode from §9.3, identical
to A to within 110 dB and included as a decoy — were copied under random codes with every provenance field scrubbed
(`device`, timings, `rerender` blocks, the ABC text, run ids) and served through the app's own
listening page (`yue2_groove.vendor.listen`). The code → source key was kept on another
machine and opened only after the ratings were handed in. One listener (the author), one
session, repeated listening, one overall rating per take on a 1–5 scale. The composition is
identical in every case, so the rating can only respond to performance and rendering.

| Code | Rating | Identity |
|---|---|---|
| T70 | **5** | M · Mac performance, seed 831003 |
| T25 | 4 | **C** · VAE-only re-decode of A (identical audio, decoy) |
| T66 | 4 | **B** · Mac NAR + VAE re-render of A (CUDA performance) |
| T81 | 4 | M · Mac performance, seed 831005 |
| T78 | 3 | **A** · CUDA reference (CUDA performance, CUDA render) |
| T98 | 3 | M · Mac performance, seed 831004 |
| T60 | 2 | M · Mac performance, seed 831002 |
| T87 | 2 | M · Mac performance, seed 831001 |

**Reading.**

1. *Noise floor: 1 point.* A and C are the same audio to within 110 dB and were rated 3 and 4.
   A one-point difference between any two takes in this table is therefore not a signal; two
   points or more may be.
2. *Rendering: indistinguishable.* The Mac render of the CUDA performance (B, 4) scored above
   the CUDA render (A, 3) — by exactly the amount the identical decoy did (C, 4). The 28.6 dB
   acoustic-stage difference of §9.3 is below this listener's repeatability; the non-blind
   "preference" for the Mac render in the first A/B is not reproduced as a preference, only as
   "no difference".
3. *Performance: no platform deficit.* The five Mac-sampled performances of the CUDA
   composition were rated 5, 4, 3, 2, 2; the single CUDA-sampled performance 3 (directly) and
   4, 4 (through its re-renders). The best-rated take in the bundle is a Mac performance, the
   two worst are Mac performances, and the CUDA performance sits in the middle of the Mac
   spread. The means (Mac 3.2, CUDA 3.7) differ by less than the noise floor. With the
   composition fixed, the platform explains nothing visible in these ratings; the draw does.
4. *What the ratings track, tentatively (n = 5).* Ratings rise with dynamic range —
   P10–P95 10.9 dB → 5, 17.0 → 4, 5.1 → 3, 8.4 → 2, 8.2 → 2 — and the loudest, most
   compressed performance (s831002, crest 13.7 dB) and the brightest (s831001, 3 956 Hz) are the
   two rated 2. The prompt asks for "quiet-loud dynamics"; the listener rewarded performances
   that delivered it. Brightness on its own does not predict the rating (s831003 at 3 576 Hz
   was the favourite).

**Conclusion of the investigation.** Step by step: the same seed lands on different
composition families by RNG design (§6.1); rendering the same performance on MPS is
indistinguishable from CUDA (§9.3, §9.5); sampling the same composition on MPS produces
performances rated across the whole scale around the CUDA one (§9.1, §9.5). What was heard
as "clearly worse on the Mac" was the E/126 draw of this song — one composition the listener
liked less — judged non-blind. Nothing measured or heard in this document points at the
platform.

**Limits.** One listener, one session, a 1-point noise floor, one CUDA performance against
five Mac performances (§8.5); the test rules out a large deficit, not a small one. A second
listener would sharpen it. The five CUDA performances of the same score (§9.1) are packed with
the five Mac performances as a second blind bundle (ten takes, random codes, key kept apart);
rating it gives the symmetric five-vs-five comparison and, since the Mac five were rated once
already, a test-retest estimate for the listener. It is built and awaiting a listening session.

### 9.6 Common-RNG run (demoted)

Forcing CPU sampling on CUDA aligns only the random stream, not the kernels. Given that every
non-zero perturbation flipped a token within a few hundred draws, a common-RNG CUDA run would
almost certainly produce a *third* take, and whether it lands in Fm/130 or E/126 would depend
on whether the first flip precedes token 27. It answers "are the logits bit-equal" (no), not
"do the numerics matter for quality" — §9.5 answered that by listening, and §9.4 would measure
the gap itself.

---

## 10. Reproduction

Environment setup is documented in [LINUX_CUDA.md](LINUX_CUDA.md) §6 (CUDA) and
[MACOS_MPS.md](MACOS_MPS.md) (MPS). The only deviations for this document:

```bash
# Exp A — CUDA with torch 2.14 instead of the upstream pin 2.10
uv pip install --python env/bin/python torch==2.14.0 --force-reinstall

# Exp B — macOS: set DTYPE = float32 in Settings → RUNTIME (or --dtype float32)

# FP8 calibration — CUDA: QUANTIZATION = fp8, MEMORY BUDGET = 12 (see LINUX_CUDA.md §4)
```

Then run the fixed request (seed 831001, cfg 1.5, cot full, ODE 32) and compare:

```bash
# composition
grep -E '^(K|Q):' score.abc                     # key / tempo
grep -o '"[A-G][#b]?[a-z0-9]*"' score.abc       # chord stream

# where two runs part
python3 - <<'PY'
import numpy as np, sys
a, b = (np.load(f"{d}/abc_tokens.npy") for d in ("RUN_A", "RUN_B"))
n = min(len(a), len(b)); d = np.nonzero(a[:n] != b[:n])[0]
print("first differing ABC token:", int(d[0]) if len(d) else "none", "| lengths", len(a), len(b))
PY

# integrity
sha256sum score.abc abc_tokens.npy semantic.npy latent.npy audio.flac
jq '.weights, .truncated, .timing' result.json  # hashes, truncation, per-stage timings
```

Verification checklist for a comparable run: same `weights.*.sha256`, same
`runtime_sha256`, same `config.json` settings except the variable under test, and
`local_env.json` recording the intended `device`/`dtype`. For cross-platform comparisons,
also record the **sampling RNG device** (§6.1) — it is currently implicit in the device — and
compare `abc_tokens.npy` before comparing anything downstream.

**Fixed-score takes and the blind bundle (§9.1, §9.5).** In `07 // BATCH`, one JSONL row per
seed with the same `style` / `lyrics` / `cot` / `cfg_scale` as the reference `request.json`
and `"abc_path"` pointing (absolute path) at a byte-exact copy of the reference `score.abc`;
include one row with the *Mac* control's own `score.abc` and its seed as a validation row — it
must reproduce that control byte for byte. Check every take's `prefix.npy` hash against the
reference before listening. The blind bundle was a one-off script: copy each take's
`audio.flac` under a random code, write `request.json` / `config.json` / `result.json` with
`device`, timings, `rerender` blocks, the ABC text and run ids removed, build the page with
`python -m yue2_groove.vendor.listen <cases> --output <dir>`, and keep the code → source key
elsewhere until the ratings are in.

---

## Appendix A — full artifact hashes (SHA-256)

```
L4 2.10 bf16 (reference)
  score.abc      db2d1467174a7d7166207f03a5e0dc146ea79e33a3af5719cccac2af4f92c7b8
  abc_tokens.npy 73e6a90596086c9f6798d39e624e9c306d24e322f8c405fee11524cc2b4e706c
  semantic.npy   e8a09373bc99f11ed3c931b95f8d6c76d6e4765b233c6fc37362fcdebda8aaed
  latent.npy     f3e90f8968bbd16d07d8e6410d91052206576708b1901dd39c083c3e0c0fb82d
  audio.flac     07f934932f357ecab3ad7d200eead58135fbad72a66fe1ac2f7fad25f3cb68ec

L4 2.14 bf16 (Exp A)
  score.abc      db2d1467174a7d7166207f03a5e0dc146ea79e33a3af5719cccac2af4f92c7b8
  abc_tokens.npy 73e6a90596086c9f6798d39e624e9c306d24e322f8c405fee11524cc2b4e706c
  semantic.npy   e8a09373bc99f11ed3c931b95f8d6c76d6e4765b233c6fc37362fcdebda8aaed
  latent.npy     f3e90f8968bbd16d07d8e6410d91052206576708b1901dd39c083c3e0c0fb82d
  audio.flac     a824641c442d9e9ed5f22871d58a5118c942ab9e0672867a08379722aec0418f

L4 FP8 (calibration; the FP8 row of LINUX_CUDA.md §3)
  score.abc      e1bd6badaa6fc6a87888903978211d9bfa90dbf7e35768e5e65c6c5e05c806cc
  abc_tokens.npy 9f30f6a6f4ec50aad31e12a7ea933c886520c0fa06c29f2ee54e1a96d57b62af
  semantic.npy   fa8193769086779a3c672fde9b1a845be3bd23c26de7270bf9cb11d5015c6cd1
  latent.npy     6132b6e06b7b3f7bc7e2c34c39e8592db299b31ae5c5e1a8cd348ff040858212
  audio.flac     41756563b76434f3a050a59bb0444a78a44a7dabc9b0637f8caa0220eadf7146

M1 Max bf16 (Control 1)
  score.abc      04eb164b3d73782cdba65e1aa0ccb30337853a7018f5b1f67500902fdc613137
  abc_tokens.npy 6b314234d2da3569109b508693d7c6ea4dc968c9b0779919877a90af2c517e80
  semantic.npy   6a33a837f9777a3f6b877969edef00a705a32d27e52f2beb4caafe5711121406
  latent.npy     d670a9c5a98c2996cd636fa649a283ec11b51ef8b0cce86b8cef68c842b75684
  audio.flac     57201bbd19154f431f13857d7ff0186b89f832db149679f4c710e8c9ee0fe34a

M4 Pro bf16 (Control 2)
  score.abc      5ba38106abfe35ce7aafca0fb363f4605a91ed7c4107f992cf954d1202e12c33
  abc_tokens.npy cec1014596cc2e2ba6cb68d07f4d8b332f88c4d103037e67f0a19a4b874ac6d0
  semantic.npy   99d780bef2169f4f90aef61924a79181b41f02739c93827f9eb692cbd7ad9d27
  latent.npy     d79e37053643c9ba2de01ed73a6e4f45c530ccfb6e5f1fdb0136f4db2563185b
  audio.flac     6ed915a459baf8ff1a48d10a0aa8352e215fd53a8e186fd99edca698fb18f729

M4 Pro fp32 (Exp B)
  score.abc      d996dd11c604f0c08d6ab76f28445c125450d979820a9960200fda0462136406
  abc_tokens.npy a6f0b6beb8256ed851904b9e155640f09ccee043ead6042685cb7a67ea7225a7
  semantic.npy   0bf1dac96dc78b68d72c9fa8f0a35fadef43140b851637f20d94209ba568a7b1
  latent.npy     51cac367a5d0d9250a696fa10cd96fc96ed2385015bb67f57b1f009cc87dc266
  audio.flac     745c85d4a623ba73e480d613700ac62c7f70f78a7aea70b6d4b6d54d9f48d386

Weights (identical in all runs)
  YuE2-3B model.safetensors   1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9
  YuE2-Vae model.safetensors  807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346

Upstream runtime (identical in all runs)
  runtime_sha256              17963197ec6c7a87f7519cda132ad70c9843acecf65eeb0d58e41506a744b3ed
```

## Appendix B — machine-readable summary

```json
{
  "request": {"seed": 831001, "cfg_scale": 1.5, "cot": "full", "ode_steps": 32},
  "rng": {"cuda": "CUDA device generator", "mps": "CPU generator", "nar_noise": "CPU generator on all platforms", "same_seed_same_stream": false},
  "runs": [
    {"run": "Control 1 — L4, torch 2.10",  "platform": "cuda/L4",   "torch": "2.10.0+cu128", "dtype": "bfloat16", "key": "Fm", "bpm": 130, "abc_tokens": 3073, "semantic_tokens": 7123, "e2e_s": 324.8,  "rtf": 1.14},
    {"run": "L4, torch 2.14",              "platform": "cuda/L4",   "torch": "2.14.0+cu130", "dtype": "bfloat16", "key": "Fm", "bpm": 130, "abc_tokens": 3073, "semantic_tokens": 7123, "e2e_s": 331.5,  "rtf": 1.16},
    {"run": "L4, FP8 (calibration)",       "platform": "cuda/L4",   "torch": "2.10.0+cu128", "dtype": "bfloat16", "quantization": "fp8", "key": "Fm", "bpm": 130, "abc_tokens": 3311, "semantic_tokens": 7846, "e2e_s": 1398.0},
    {"run": "M1 Max, bf16",                "platform": "mps/M1Max", "torch": "2.14.0",       "dtype": "bfloat16", "key": "E",  "bpm": 126, "abc_tokens": 2686, "semantic_tokens": 6665, "e2e_s": 1525.6, "rtf": 5.72},
    {"run": "Control 2 — M4 Pro, bf16",    "platform": "mps/M4Pro", "torch": "2.14.0",       "dtype": "bfloat16", "key": "E",  "bpm": 126, "abc_tokens": 3017, "semantic_tokens": 6677, "e2e_s": 1231.4, "rtf": 4.61},
    {"run": "M4 Pro, fp32",                "platform": "mps/M4Pro", "torch": "2.14.0",       "dtype": "float32",  "key": "E",  "bpm": 126, "abc_tokens": 2839, "semantic_tokens": 7155, "e2e_s": 2036.1, "rtf": 7.11}
  ],
  "first_differing_abc_token": {
    "L4_2.10 vs L4_2.14": null,
    "L4_bf16 vs L4_fp8": 225,
    "M1Max_bf16 vs M4Pro_bf16": 122,
    "M4Pro_bf16 vs M4Pro_fp32": 122,
    "M1Max_bf16 vs M4Pro_fp32": 313,
    "L4 vs any MPS": 26
  },
  "audio_pcm_snr_db": {"L4_2.10 vs L4_2.14": 119, "L4_ODE32 vs L4_ODE48": 28},
  "follow_up": {
    "same_machine_repeat": {"run": "M4 Pro repeat of Control 2", "artifacts_identical": true},
    "rerender_of_reference_on_M4Pro": {
      "vae_only": {"snr_db_vs_cuda": 109.8},
      "nar_vae_32": {"snr_db_vs_cuda": 28.6, "latent_rms_delta_over_std": 0.022},
      "nar_vae_16": {"snr_db_vs_mac32": 21.1},
      "nar_vae_8":  {"snr_db_vs_mac32": 16.6}
    },
    "fixed_score_batch": {"run": "M4 Pro batch on Control 1's score.abc", "platform": "mps/M4Pro",
                          "validation_row_reproduces_control2": true, "seeds": [831001, 831002, 831003, 831004, 831005]},
    "blind_listening": {"listener": "author", "scale": "1-5 overall", "noise_floor_points": 1,
                        "ratings": {"A_cuda_reference": 3, "B_mac_rerender_of_A": 4, "C_vae_only_redecode_of_A": 4,
                                    "M_s831001": 2, "M_s831002": 2, "M_s831003": 5, "M_s831004": 3, "M_s831005": 4}},
    "fixed_score_batch_cuda": {"run": "LINUX_CUDA.md E10, seeds 831001..5", "platform": "cuda/L4 (second host)",
                               "seed_831001_reproduces_reference": true, "centroid_hz": [2895, 4121, 4298, 3142, 3232]},
    "blind_listening_2": {"status": "built, not rated", "cases": 10, "composition": "reference score", "cuda": 5, "mac": 5}
  }
}
```

## Appendix C — raw data

The run directories (each containing `result.json`, `config.json`, `request.json`,
`local_env.json`, `score.abc`, `semantic.npy`, `abc_tokens.npy`, `prefix.npy`, `latent.npy`,
`audio.flac`) are kept by the author with the other validation artifacts. Every number in this
document is derived from those files; `result.json` additionally carries the per-stage timings
and artifact hashes recorded by the app at generation time. All artifact hashes were re-verified
against those files for this revision, including the follow-up runs of §9 (the M4 Pro repeat, the
re-renders, the fixed-score batch and the blind bundle with its key).
