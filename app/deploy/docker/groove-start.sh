#!/bin/bash
# Bring YUE2 // GROOVE up inside the container: weights → verification → (GGUF files) → app,
# reporting each step on stdout and, when GROOVE_PROGRESS_URL is set, to that URL as JSON
# (bearer GROOVE_PROGRESS_TOKEN). Steps: weights | verify | gguf | start | ready | failed.
set -uo pipefail
cd /app
# never run twice in one container (a second /start.sh hook, a manual start over ssh, …)
PIDFILE=/run/groove-start.pid
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[groove-start] already running (pid $(cat "$PIDFILE"))"; exit 0
fi
echo $$ > "$PIDFILE"
PORT="${YUE2_GROOVE_PORT:-7860}"
GROOVE_PROGRESS_URL="${GROOVE_PROGRESS_URL:-}"
GROOVE_PROGRESS_TOKEN="${GROOVE_PROGRESS_TOKEN:-}"

progress() { # step status message
  local step="$1" status="$2" message="${3:-}"
  echo "[groove-start] $(date -u +%FT%TZ) $step $status $message"
  if [[ -n "$GROOVE_PROGRESS_URL" ]]; then
    curl -sS -m 10 -o /dev/null -X POST "$GROOVE_PROGRESS_URL" \
      -H "content-type: application/json" \
      ${GROOVE_PROGRESS_TOKEN:+-H "authorization: Bearer $GROOVE_PROGRESS_TOKEN"} \
      --data "$(printf '{"step":"%s","status":"%s","message":%s,"ts":"%s","pod":"%s"}' \
        "$step" "$status" "$(printf '%s' "$message" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
        "$(date -u +%FT%TZ)" "${RUNPOD_POD_ID:-}")" || true
  fi
}
fail() { progress "$1" failed "$2"; exit 1; }

mkdir -p "$HF_HOME" "$YUE2_GROOVE_RUNS" "$YUE2_GROOVE_MODELS"

# 1. weights (cached in /data across restarts)
progress weights started "YuE2-3B, YuE2-Vae, SheetSage2, MERT-v2-FullSong"
env/bin/hf download m-a-p/YuE2-3B > /dev/null || fail weights "YuE2-3B download failed"
env/bin/hf download m-a-p/YuE2-Vae > /dev/null || fail weights "YuE2-Vae download failed"
if [[ ! -f "$YUE2_GROOVE_MODELS/SheetSage2/config.json" ]]; then
  .venv-sheetsage2/bin/hf download m-a-p/SheetSage2 --local-dir "$YUE2_GROOVE_MODELS/SheetSage2" > /dev/null || fail weights "SheetSage2 download failed"
fi
.venv-sheetsage2/bin/python - <<'PY' || fail weights "MERT-v2-FullSong download failed"
import json, os
from huggingface_hub import snapshot_download
c = json.load(open(os.path.join(os.environ["YUE2_GROOVE_MODELS"], "SheetSage2", "config.json"), encoding="utf-8"))
snapshot_download(c["base_model_name_or_path"], revision=c["base_model_revision"])
PY
progress weights done

# 2. verification: weight hashes, CUDA, bf16, FlashAttention
progress verify started
env/bin/python -m yue2.cli doctor --model m-a-p/YuE2-3B --vae m-a-p/YuE2-Vae --verify-hashes > /tmp/doctor.json 2>/tmp/doctor.err \
  || fail verify "yue2 doctor failed: $(tail -c 300 /tmp/doctor.err)"
GPU_INFO=$(env/bin/python /usr/local/bin/groove-gpu-check 2>&1) || fail verify "$GPU_INFO"
progress verify done "$GPU_INFO"

# 3. the GGUF engine's files, when that is the engine this card gets (the check above ends in
#    backend=gguf or backend=auto→gguf, docs/GGUF_ENGINE.md §3): converted and quantized once
#    from the weights, into $YUE2_GROOVE_MODELS/gguf, so the first generation does not spend
#    the minute on it. Not fatal: the app repeats the attempt at the first GGUF generation and
#    reports there, and BACKEND=torch stays available.
if [[ "$GPU_INFO" == *"backend="*gguf ]]; then
  progress gguf started "Q8_0 GGUF files from the YuE2 weights (once)"
  if env/bin/python -m yue2_groove.gguf_engine prepare 2>&1 | sed 's/^/[gguf] /'; then
    progress gguf done
  else
    progress gguf done "preparation failed — the app will try again at the first GGUF generation"
  fi
fi

# 4. the app
progress start started "port $PORT"
AUTH_ARGS=()
[[ -n "${YUE2_GROOVE_AUTH:-}" ]] && AUTH_ARGS=(--auth "$YUE2_GROOVE_AUTH")
env/bin/python -m yue2_groove --host 0.0.0.0 --port "$PORT" \
  --sheetsage-python "$YUE2_GROOVE_SHEETSAGE_PYTHON" "${AUTH_ARGS[@]}" &
APP_PID=$!
for _ in $(seq 1 120); do
  if ! kill -0 "$APP_PID" 2>/dev/null; then fail start "the app exited during startup"; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" || true)
  if [[ "$code" == "200" || "$code" == "401" ]]; then
    URL=""
    [[ -n "${RUNPOD_POD_ID:-}" ]] && URL="https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net"
    progress ready done "${URL:-http://localhost:$PORT}"
    wait "$APP_PID"
    fail start "the app exited (code $?)"
  fi
  sleep 2
done
fail start "the app did not answer on port $PORT within 240 s"
