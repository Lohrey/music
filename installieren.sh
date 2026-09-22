#!/usr/bin/env bash
# Linux (Ubuntu o. ä.) mit NVIDIA-Treiber: ./installieren.sh
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv run --no-project --python 3.11 python tools/installer.py install "$@"
