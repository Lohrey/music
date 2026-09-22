# Musik machen auf einer Linux-GPU (eigener Server oder Cloud-GPU wie RunPod / Vast.ai).
#   docker build -t musik-machen -f docker/Dockerfile .
#   docker run --gpus all -p 42003:42003 -v musik-daten:/data musik-machen
# Modelle und Songs liegen im Volume /data und überleben Updates.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:$PATH \
    HF_HOME=/data/hf \
    UV_CACHE_DIR=/data/uv-cache \
    UV_LINK_MODE=copy \
    PYTHONUTF8=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libsndfile1 libvulkan1 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /opt/musik-machen
COPY . .

# Pakete ins Image, große Modelle erst beim ersten Start ins Volume
RUN uv run --no-project --python 3.11 python tools/installer.py install --yes --skip-models --skip-sheetsage \
    || echo "Installation im Build unvollständig – wird beim Start nachgeholt"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 42003
VOLUME ["/data"]
ENTRYPOINT ["/entrypoint.sh"]
