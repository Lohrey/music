#!/usr/bin/env bash
# YUE2 // GROOVE service manager: start / stop / restart / status / log
#
# Binds 0.0.0.0 by default so other machines on your LAN can open
# http://<this-machine-ip>:7860. The UI has no user system of its own: set a
# password when you expose it on a network (see below).
#
# Usage:
#   bash scripts/serve.sh start [--port 7860] [--host 0.0.0.0] [--view song|studio]
#                               [--tab 0..6]
#                               [--device auto|mps|cuda|cpu] [--dtype auto|bfloat16|float32]
#                               [--backend auto|torch|torch-eager|gguf]
#                               [--model ID_OR_DIR] [--runs DIR]
#                               [--auth user:password] [--sheetsage-python PATH]
#                               [--no-preload] [-f]
#
# The app opens in the SONG view by default.  --view forces SONG or STUDIO for
# this launch (or set YUE2_GROOVE_VIEW in .env); an explicit --tab forces the
# Studio view on that tab.  Without either flag the last choice is remembered
# per browser (localStorage).
#   bash scripts/serve.sh stop
#   bash scripts/serve.sh restart [same options as start]
#   bash scripts/serve.sh status
#   bash scripts/serve.sh log
#
# Optional local configuration (not tracked by git): create `.env` in the
# repository root, for example
#   YUE2_GROOVE_AUTH=alice:my-secret
#   YUE2_GROOVE_PORT=7860
#   YUE2_GROOVE_MODEL=m-a-p/YuE2-3B
#   YUE2_GROOVE_RUNS=/somewhere/runs
#   YUE2_GROOVE_SHEETSAGE_PYTHON=/path/to/.venv-sheetsage2/bin/python
# The script sources that file automatically.
#
# The Python interpreter defaults to the repository's .venv; override with
# YUE2_GROOVE_PYTHON=/path/to/python.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY="${YUE2_GROOVE_PYTHON:-$ROOT/.venv/bin/python}"
RUN_DIR="${YUE2_GROOVE_RUNS:-$ROOT/runs}"

HOST="${YUE2_GROOVE_HOST:-0.0.0.0}"
PORT="${YUE2_GROOVE_PORT:-7860}"
TAB=""
VIEW=""
FOREGROUND=0
EXTRA_ARGS=()

# optional local configuration
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
  HOST="${YUE2_GROOVE_HOST:-$HOST}"
  PORT="${YUE2_GROOVE_PORT:-$PORT}"
  RUN_DIR="${YUE2_GROOVE_RUNS:-$RUN_DIR}"
  PY="${YUE2_GROOVE_PYTHON:-$PY}"
fi

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

cmd="${1:-}"
case "$cmd" in
  start|stop|restart|status|log) shift ;;
  *) usage ;;
esac

while [ $# -gt 0 ]; do
  case "$1" in
    --port)     PORT="$2"; shift 2 ;;
    --host)     HOST="$2"; shift 2 ;;
    --view)     VIEW="$2"; shift 2 ;;
    --tab)      TAB="$2"; shift 2 ;;
    --device)   EXTRA_ARGS+=("--device" "$2"); shift 2 ;;
    --dtype)    EXTRA_ARGS+=("--dtype" "$2"); shift 2 ;;
    --backend)  EXTRA_ARGS+=("--backend" "$2"); shift 2 ;;
    --model)    EXTRA_ARGS+=("--model" "$2"); shift 2 ;;
    --runs)     RUN_DIR="$2"; shift 2 ;;
    --auth)     YUE2_GROOVE_AUTH="$2"; shift 2 ;;
    --sheetsage-python) EXTRA_ARGS+=("--sheetsage-python" "$2"); shift 2 ;;
    --no-preload) EXTRA_ARGS+=("--no-preload"); shift ;;
    -f|--foreground) FOREGROUND=1; shift ;;
    -h|--help)  usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

PID_FILE="$RUN_DIR/.server.pid"
PORT_FILE="$RUN_DIR/.server.port"
LOG_FILE="$RUN_DIR/.server.log"

if [ ! -x "$PY" ]; then
  echo "Python interpreter not found: $PY" >&2
  echo "Create the virtual environment first (see README) or set YUE2_GROOVE_PYTHON." >&2
  exit 1
fi

is_running() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

current_port() { cat "$PORT_FILE" 2>/dev/null || echo "$PORT"; }

lan_ips() {
  local ip iface out=()
  if command -v ipconfig >/dev/null 2>&1; then           # macOS
    for iface in en0 en1 en2; do
      ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
      [ -n "$ip" ] && out+=("$ip")
    done
  fi
  if [ "${#out[@]}" -eq 0 ]; then                        # Linux
    for ip in $(hostname -I 2>/dev/null || true); do
      out+=("$ip")
    done
  fi
  if [ "${#out[@]}" -eq 0 ] && command -v ifconfig >/dev/null 2>&1; then
    ip="$(ifconfig 2>/dev/null | awk '/inet / && $2 != "127.0.0.1" {print $2; exit}')"
    [ -n "$ip" ] && out+=("$ip")
  fi
  if [ "${#out[@]}" -gt 0 ]; then
    printf '%s\n' "${out[@]}"
  fi
}

auth_label() {
  # never print the password itself
  if [ -n "${YUE2_GROOVE_AUTH:-}" ]; then
    echo "${YUE2_GROOVE_AUTH%%:*}:***"
  else
    echo "no password set (anyone on the LAN can use it)"
  fi
}

print_urls() {
  local port="$1"
  echo "  local:  http://127.0.0.1:${port}/"
  while IFS= read -r ip; do
    [ -n "$ip" ] && echo "  LAN:    http://${ip}:${port}/"
  done < <(lan_ips)
  echo "  login:  $(auth_label)"
}

start_service() {
  mkdir -p "$RUN_DIR"
  if is_running; then
    echo "Already running (PID $(cat "$PID_FILE"))."
    status_service
    return 0
  fi
  rm -f "$PID_FILE"

  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port ${PORT} is already in use:"
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | tail -n +1
    echo "Pick another port: bash scripts/serve.sh start --port 7861"
    return 1
  fi

  # The login is handed to the app through the environment (YUE2_GROOVE_AUTH), not
  # as a command-line argument, so it does not show up in `ps` or in this output.
  export YUE2_GROOVE_AUTH="${YUE2_GROOVE_AUTH:-}"
  local args=(-m yue2_groove --host "$HOST" --port "$PORT" --runs "$RUN_DIR")
  # only pass the view/tab flags when they were asked for: the app's own default
  # is SONG, and a bare --tab 0 would force the Studio view
  if [ -n "$VIEW" ]; then
    args+=(--view "$VIEW")
  fi
  if [ -n "$TAB" ]; then
    args+=(--tab "$TAB")
  fi
  # bash 3.2 treats an empty array as unbound under set -u, hence the length check
  if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    args+=("${EXTRA_ARGS[@]}")
  fi

  if [ "$FOREGROUND" = "1" ]; then
    echo "Foreground (Ctrl-C to stop): $PY ${args[*]}"
    exec "$PY" "${args[@]}"
  fi

  # Keep the log across restarts: a panic used to be followed by a truncation that
  # erased the run directory it had just printed.  Append a start banner, and only
  # rotate when the file grows past ~5 MB.
  if [ -f "$LOG_FILE" ] && [ "$(wc -c < "$LOG_FILE" | tr -d ' ')" -gt 5242880 ]; then
    mv -f "$LOG_FILE" "$LOG_FILE.1"
  fi
  printf '\n===== %s · serve.sh start =====\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
  echo "Starting: $PY ${args[*]}"
  PYTHONUNBUFFERED=1 nohup "$PY" "${args[@]}" >> "$LOG_FILE" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "$PORT" > "$PORT_FILE"

  local i
  for i in $(seq 1 90); do
    if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${PORT}/"; then
      echo "Started (PID ${pid})."
      print_urls "$PORT"
      echo "  stop:   bash scripts/serve.sh stop"
      echo "  log:    bash scripts/serve.sh log"
      echo "Note: the model keeps loading in the background after the page is up;"
      echo "      the first generation waits for it (or downloads the weights first)."
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Start failed; last lines of the log:"
      tail -n 30 "$LOG_FILE" || true
      rm -f "$PID_FILE"
      return 1
    fi
    sleep 1
  done
  echo "Timed out after 90 s; last lines of the log:"
  tail -n 30 "$LOG_FILE" || true
  return 1
}

stop_service() {
  if ! is_running; then
    echo "Not running."
    rm -f "$PID_FILE" "$PORT_FILE"
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  echo "Stopping (PID ${pid})..."
  kill "$pid" 2>/dev/null || true
  local i
  for i in $(seq 1 40); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE" "$PORT_FILE"
      echo "Stopped."
      return 0
    fi
    sleep 0.5
  done
  echo "Graceful shutdown timed out; killing."
  kill -9 "$pid" 2>/dev/null || true
  sleep 1
  rm -f "$PID_FILE" "$PORT_FILE"
  echo "Stopped (killed)."
}

status_service() {
  if is_running; then
    local pid port
    pid="$(cat "$PID_FILE")"
    port="$(current_port)"
    echo "Running: PID ${pid}"
    print_urls "$port"
    echo "  log:    ${LOG_FILE}"
  else
    echo "Not running."
    return 1
  fi
}

case "$cmd" in
  start)   start_service ;;
  stop)    stop_service ;;
  restart) stop_service; start_service ;;
  status)  status_service ;;
  log)
    if [ -f "$LOG_FILE" ]; then
      tail -n 100 -f "$LOG_FILE" || true      # Ctrl-C ends the tail; that is not an error
    else
      echo "No log yet (the service has not been started)."; exit 1
    fi ;;
esac
