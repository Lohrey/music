#!/usr/bin/env bash
set -e
cd /opt/musik-machen
mkdir -p /data/runs /data/models /data/ace-checkpoints
# Songs, Modelle und Einstellungen dauerhaft im Volume /data
[ -L app/runs ] || { rm -rf app/runs; ln -s /data/runs app/runs; }
[ -L app/models ] || { rm -rf app/models; ln -s /data/models app/models; }
[ -f /data/einstellungen.txt ] || cp einstellungen.beispiel.txt /data/einstellungen.txt
ln -sf /data/einstellungen.txt einstellungen.txt
export ACESTEP_CHECKPOINTS_DIR=/data/ace-checkpoints
# fehlende Teile (Modelle, SheetSage2) beim ersten Start nachladen – danach schnell
uv run --no-project --python 3.11 python tools/installer.py install --yes
exec uv run --no-project --python 3.11 python tools/installer.py start --host 0.0.0.0 --no-browser
