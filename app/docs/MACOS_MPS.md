# YuE2 on macOS (Apple Silicon / MPS)

YuE2 officially validates Linux + NVIDIA (24 GB, bfloat16). It also runs on Apple Silicon
through PyTorch's MPS backend — with one caveat that this project works around at the
dependency level. These notes record what was verified and how.

> **Companion analysis:** [CROSS_PLATFORM.md](CROSS_PLATFORM.md) measures what differs between
> CUDA and MPS for the same request. Note in particular that the sampler draws from a
> platform-specific RNG device (CUDA generator on CUDA, CPU generator on MPS), so **the same
> seed does not reproduce across platforms** — and not across Macs either: an M1 Max and an
> M4 Pro on the same CPU stream agreed for 122 ABC tokens and then diverged, because any kernel
> numeric difference eventually flips a sampled token. On one Mac, however, a repeat run is
> bit-exact end to end (verified on the M4 Pro, `audio.flac` included). Quality-wise, a blind
> test on a fixed composition found no MPS deficit: Mac performances were rated across the whole
> scale around the CUDA one, and the Mac rendering of a CUDA performance was indistinguishable
> from the CUDA rendering (CROSS_PLATFORM.md §9.5).

## Tested configuration

| | |
|---|---|
| Machine | Apple M4 Pro, 64 GB unified memory, macOS 26.6 |
| Python | 3.12 (`uv`) |
| PyTorch | 2.14.0 via `overrides/macos.txt` (upstream pins 2.10.0) |
| YuE2 | `yue2-v0.1.6` |
| Weights | `m-a-p/YuE2-3B` (7.3 GB bf16) + `m-a-p/YuE2-Vae` (0.5 GB) |

## The MPS attention defect (and why torch is overridden)

With the upstream pin (torch 2.10.0), long generations on MPS fail part-way with

```
RuntimeError: probability tensor contains either inf, nan or element < 0
```

The cause is **not** a bfloat16 range problem (bf16 has the same exponent width as fp32).
It is [pytorch/pytorch#174861](https://github.com/pytorch/pytorch/issues/174861): the MPS
2-pass "vector" attention kernel used for single-query (incremental KV-cache) decoding in
half precision reads out of bounds once the key length reaches 1024. Outputs become
nondeterministic garbage, sometimes inf/NaN, which then propagates through the residual
stream and RMSNorm. Prefill (multi-query) uses a different kernel and is unaffected, so
generation looks fine for the first several hundred tokens and then breaks.

The defect was introduced after torch 2.7.1 and fixed in **2.11.0**. `scripts/mps_sdpa_check.py`
reproduces it without loading any weights, mirroring the model's exact call shape
(16 query heads, 8 KV heads, head dim 128, sliced cache view, `repeat_interleave` for GQA):

| torch | kv < 1024 | kv ≥ 1024 | verdict |
|---|---|---|---|
| 2.10.0 (upstream pin) | OK | wrong, non-reproducible | `DEFECT PRESENT`, exit 1 |
| 2.11.0 | OK | OK | exit 0 |
| 2.14.0 (this project) | OK | OK | exit 0 |

The check also confirms that on the defective version an explicit zero attention mask, a
float32 attention, or a hand-written softmax all give correct results — i.e. only the fused
half-precision single-query kernel is broken. Nothing in the model or in upstream's code
needs to change; a torch with the fix is the whole solution.

`enable_gqa=True` is not usable on MPS in any of these versions (it raises); upstream's
`repeat_interleave` fallback in `modeling_yue2.sdpa` remains necessary.

## Performance (M4 Pro, bf16, `cot="full"`, 32 ODE steps)

Same request, seed and score (192 s song, CFG 1.5), comparing dtypes:

| dtype | semantic tokens | semantic tok/s | NAR | VAE | end to end | weights |
|---|---|---|---|---|---|---|
| bfloat16 (torch 2.14) | 4793 | 13.5 | 343 s | 7.7 s | 11.8 min | 7.3 GB |
| float32 (torch 2.10, fp32 workaround) | 4808 | 4.8 | 645 s | 14.7 s | 27.8 min | 14.6 GB |

A short song (48 s of audio) generates 1191 semantic tokens at ~35 tok/s; throughput
falls with context length (KV cache), which is why the long run averages 13.5 tok/s.
Rule of thumb: a 1-minute song takes 2–3 minutes; a full 3–4 minute song 10–15 minutes.
The NAR stage grows fastest with length (attention is roughly quadratic).

## Memory

bf16 weights need about 7.3 GB plus the KV cache and NAR working set; a 32 GB machine is
comfortable, 64 GB was used for the numbers above. `--dtype float32` doubles the weight
memory and roughly halves the speed; it exists for debugging, not for everyday use.

## If you must stay on torch 2.10

Wrap `yue2.modeling_yue2.sdpa` at runtime so that MPS + half precision + single-query calls
pass an explicit all-zero `attn_mask` (which routes around the fused kernel). This was
verified to avoid the defect at a small cost, but it is not shipped here: upgrading torch
is simpler and fixes the root cause.
