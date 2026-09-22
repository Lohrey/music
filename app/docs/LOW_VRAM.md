# Less than 24 GB of VRAM — quick reference

The requirement stays: an NVIDIA GPU with **24 GB** (YuE2's validated setup) or an Apple Silicon
Mac with 32 GB+. This page is for people running on smaller cards anyway. Linux numbers are
measured ([LINUX_CUDA.md](LINUX_CUDA.md)); Windows numbers are estimates (§4).

## 1. Your card

| GPU | Works? | What you get | What to set |
|---|---|---|---|
| **24 GB+** (RTX 3090, 4090, 5090, …) | Yes | Everything | Nothing |
| **16 GB** (RTX 4060 Ti 16G, 4070 Ti Super, 4080, 5060 Ti 16G, 5070 Ti, 5080, …) | Yes | Everything at the default length ceiling (6 min) at any CFG; longer ceilings (§2) add ~330 MB per minute at CFG 1.5 | Nothing |
| **12 GB** (RTX 3060 12G, 4070, 4070 Super, 4070 Ti, 5070, …) — Linux or Docker | With limits | Songs of ~5 min at CFG 1.0 · songs ≤ 3 min at CFG 1.5 · longer at CFG 1.5: §3 | MEMORY BUDGET **12**, CFG SCALE **empty** |
| **12 GB** — Windows | Borderline | Songs ≤ 3 min at CFG 1.0 to start · CFG 1.5: FP8 or Docker (§3) | MEMORY BUDGET **12**, CFG SCALE **empty**, max_tokens **4 500** |
| **12 GB**, any OS, with the [GGUF engine](GGUF_ENGINE.md) installed | Yes | Full-length songs at any CFG (measured peak 8.2 GB on a 16 GB card), faster AR stage; a different take for the same seed | Nothing — BACKEND=auto picks it |
| **8–11 GB** (RTX 4060, 3070, 3080 10G, all RTX 20 series, …) | **No** on the reference engine | — | The [GGUF engine](GGUF_ENGINE.md): an 8 GB card gets a context cap automatically (songs up to ~4.9 min; measured on an RTX 2070 8 GB under Windows: a 4:51 song at 6.1 GB peak), or a cloud GPU (RunPod, …) with the [Docker image](../deploy/docker/README.md); 16 GB there is enough |

## 2. Where to set it

1. Top right → **STUDIO** (the SONG view has no settings).
2. Settings rail (right) → **MEMORY BUDGET** → `12`. **Repeat after every launch** — it resets to 24.
   The next GENERATE applies it (or click LOAD / APPLY).
3. **01 // GENERATE** → **CFG SCALE** → leave **empty** (= 1.0).
4. Song length → **ADVANCED // SAMPLING** → semantic phase → **max_tokens**; the *Estimated audio
   length* line updates. Shorten the lyrics to match, or the song is cut off at the limit.

| max_tokens | 2 200 (*Preview* preset) | 4 500 | 6 000 | 7 200 | 9 000 (default) | 15 000 (*Long song* preset) |
|---|---|---|---|---|---|---|
| Song length | ≈ 1:30 | 3:00 | 4:00 | 4:48 | 6:00 | 10:00 |

   Above 9 000 the original engine reserves more memory (≈ 330 MB per extra minute at CFG 1.5,
   half that at CFG 1.0): on a 12 GB card stay at the default; the GGUF engine reserves the
   same either way. The song still ends when the score ends, so long songs need long lyrics.

## 3. 12 GB: what you want → what to set

Start at the top; go down one row only if that row is not what you want.

| You want | Set | Result |
|---|---|---|
| ~5-min song, CFG 1.0 | MEMORY BUDGET 12, nothing else | Works (measured, ~5 min per song on an NVIDIA L4) |
| CFG 1.5, song ≤ 3 min | + max_tokens **4 500**, shorter lyrics | Works (measured with a 2-min song) |
| CFG 1.5, song ≤ 4:45 | + max_tokens **7 200** | Works with almost no margin; long lyrics or a long score can push it over |
| CFG 1.5, full length | QUANTIZATION → **fp8** (RTX 40 series or newer; not RTX 30) | Works, **~4× slower** (23 min instead of 5½ on an L4) |
| CFG 1.5, full length, any card | Install the [GGUF engine](GGUF_ENGINE.md) (`python -m yue2_groove.gguf_engine install`) | Works, faster than FP8; not the reference configuration — a different take for the same seed |
| Windows: the Linux rows above | [Docker image](../deploy/docker/README.md) under Docker Desktop + WSL2 | Linux numbers and speed |

Every one of these changes (CFG, length, FP8, Windows vs Linux) turns the same seed into a
**different song**. That is normal.

## 4. Windows users, read this

Windows is not a supported platform. Install, launch and Cover were verified once (Windows 11,
RTX 2070 8 GB), and one full song was generated there — on the GGUF engine, which that card
gets automatically; the reference engine has not been run on Windows by the maintainer, so
the Windows numbers on this page are the Linux measurements plus the known Windows overhead.
Reports are welcome.

| What you see | Why | What to do |
|---|---|---|
| Slower than the Linux figures (~1.5×); STATUS shows `backend=torch-eager (… cannot run FlashAttention …)` | Windows builds of PyTorch have no FlashAttention, so the app uses the slower decoder — it also needs ~0.7 GB more VRAM, which is why 12 GB is "borderline" | Expected. For Linux speed and memory, run the Docker image under Docker Desktop + WSL2 |
| Crash at the first generate step: `USE_FLASH_ATTENTION was not enabled for build` | App older than 0.9.0 | Update (Pinokio: **Update**). 0.9.0+ switches decoder by itself |
| Crash at start (`fcntl`), or Cover fails with non-English folder names or lyrics | App older than 0.9.0 | Update |
| A song takes hours; STATUS shows `device=cpu` | PyPI's Windows torch is CPU-only | Pinokio installs the CUDA build for you. Manual install, in the app's venv: `pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps` |
| Out of memory on 12 GB with the §3 settings | The desktop, browser and other apps share the card | Close other GPU apps. If the CPU has built-in graphics, plug the monitor into the motherboard |

## 5. Out of memory?

The app keeps running. Change one thing, press GENERATE again: on 12 GB, go one row down in §3;
on Windows, see the last row of §4.

## 6. Leave these alone

| Setting | Why |
|---|---|
| OFFLOAD AR WEIGHTS | Does not lower the peak (measured) |
| ODE STEPS | Changes render time and sound, not memory |
| DTYPE, BACKEND, VAE CORE FRAMES | Leave what the app picked: `auto` / the launch default / `auto`. BACKEND → `gguf` is the one deliberate change, see §1 |
| MEMORY BUDGET below your card's size | Saves nothing; runs fail sooner |
