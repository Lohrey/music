#!/usr/bin/env bash
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
exec uv run --no-project --python 3.11 python tools/installer.py start "$@"
