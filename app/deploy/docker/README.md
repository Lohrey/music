# YUE2 // GROOVE in Docker (Linux + NVIDIA)

One image, three uses: a desktop or server with an NVIDIA GPU (Linux, or Windows through
Docker Desktop / WSL2), a RunPod pod, and the launcher that creates such pods. It is the
environment `docs/LINUX_CUDA.md` validates — `torch 2.10.0+cu128`, `yue2-infer 0.1.6`, the
SheetSage2 Cover environment — plus the [GGUF engine](../docs/GGUF_ENGINE.md)'s yue2.cpp
binaries (the release's own `linux-x64` asset), baked once. **Code only:** the YuE2, SheetSage2
and MERT-v2-FullSong weights are CC BY-NC 4.0 and are downloaded by you at first start into
`/data` (mount it so that happens once); the GGUF files are made there from them.

Not for macOS: Docker Desktop cannot pass Metal/MPS into a Linux container — Mac users take
the Pinokio or native path.

## Requirements

- NVIDIA GPU: 16 GB comfortably runs everything the app can produce on the reference engine;
  on a card under 16 GB the app switches to the GGUF engine by itself (BACKEND=auto,
  `docs/GGUF_ENGINE.md` §3): 12 GB then runs full-length songs at any CFG, 8 GB gets a context cap
  (songs up to ~4.9 min; a full song measured at 6.1 GB peak). `YUE2_GROOVE_BACKEND=torch` keeps
  the reference engine on a small card (12 GB: CFG 1.0 or shorter songs, `docs/LINUX_CUDA.md` §3).
- A driver that supports CUDA 12.8 (R570 or newer on Linux; the matching WSL driver on Windows).
- Docker with GPU support: NVIDIA Container Toolkit on Linux; Docker Desktop on Windows with
  WSL2 (GPU support is built in).
- ~25 GB of disk: the image (~16 GB) plus ~8 GB of weights in `/data`, and another 4 GB there
  when the GGUF engine is used (its files, made once from the weights).

## Run

```bash
docker run --gpus all -p 7860:7860 \
  -v groove-data:/data \
  -e YUE2_GROOVE_AUTH=alice:my-secret \
  ghcr.io/deadjoe/yue2_groove:latest
```

Then open http://localhost:7860. The first start downloads the weights (`weights` step in the
log), verifies their hashes and the GPU (`verify` — its line ends in the engine this card gets,
`backend=auto→torch` or `auto→gguf`), on a card that gets the GGUF engine converts the weights
once (`gguf`), and starts the app (`start` → `ready`). Works, downloaded weights and GGUF files
live in the `groove-data` volume (`/data/runs`, `/data/hf`, `/data/models`) and survive image
updates. `docker compose` equivalent:

```yaml
services:
  groove:
    image: ghcr.io/deadjoe/yue2_groove:latest
    ports: ["7860:7860"]
    volumes: ["groove-data:/data"]
    environment:
      YUE2_GROOVE_AUTH: alice:my-secret
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
volumes:
  groove-data: {}
```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `YUE2_GROOVE_AUTH` | (none) | `user:password` for the web UI — set it on anything reachable from a network |
| `YUE2_GROOVE_PORT` | `7860` | port the app listens on inside the container |
| `HF_HOME` | `/data/hf` | Hugging Face cache (YuE2 weights, MERT snapshot) |
| `YUE2_GROOVE_RUNS` | `/data/runs` | where works are stored |
| `YUE2_GROOVE_MODELS` | `/data/models` | SheetSage2 snapshot directory; the GGUF files go in its `gguf/` |
| `YUE2_GROOVE_BACKEND` | `auto` | `auto` (the VRAM rule), `torch`, `gguf` — `docs/GGUF_ENGINE.md` §3; the engine's other knobs (`YUE2_GROOVE_GGUF_*`) pass through the same way |
| `GROOVE_PROGRESS_URL`, `GROOVE_PROGRESS_TOKEN` | (none) | when set, every start step is POSTed as JSON to that URL with a bearer token — used by the pod launcher, ignored otherwise |

## On RunPod

Create a pod from this image with port `7860/http` exposed; RunPod's proxy URL
`https://<pod-id>-7860.proxy.runpod.net` is the app. The base image's `/start.sh` stays the
entrypoint (sshd with `PUBLIC_KEY`, Jupyter with `JUPYTER_PASSWORD`, as on any RunPod pod);
it calls `/pre_start.sh`, which starts `groove-start`. Weights download from Hugging Face in
about half a minute inside RunPod's datacenters (31 s measured, `hf_transfer`); a network volume mounted at `/data` makes it
instant on later pods in the same datacenter.

[yue2_groove_pod](https://github.com/deadjoe/yue2_groove_pod) is a phone-sized launcher for
exactly this: a Cloudflare Worker that creates the cheapest in-stock pod from this image,
follows the progress protocol below, hands back the URL and deletes the pod on a time limit.

## Progress protocol (for launchers)

`groove-start` reports `{"step", "status", "message", "ts", "pod"}` with
`step ∈ {weights, verify, gguf, start, ready}` and `status ∈ {started, done, failed}`; `ready/done`
carries the URL in `message` (the RunPod proxy URL when `RUNPOD_POD_ID` is set, else
`http://localhost:<port>`). `gguf` appears only on a card that gets the GGUF engine and never
fails the start (a failed preparation is reported in its `done` message and retried by the
app). A `failed` status ends the start; the container stays up so the log can be read. The
same lines are printed to the container log.

## Build

`.github/workflows/image.yml` builds and pushes `ghcr.io/deadjoe/yue2_groove` on every `v*`
tag (also tagged `latest`) and on manual dispatch (tag `main` by default). The yue2.cpp binaries
come from a release asset (`yue2cpp-<pin>-linux-x64.tar.gz`, built by `yue2cpp.yml`); the
workflow picks the newest release that lists the pinned name and, right after a pin bump, waits
for it. Local build:

```bash
docker build -f deploy/docker/Dockerfile -t yue2-groove .
# after a pin bump, name a release that already lists the new asset:
docker build -f deploy/docker/Dockerfile --build-arg YUE2CPP_PIN=<pin> --build-arg YUE2CPP_RELEASE=v<x.y.z> -t yue2-groove .
```
