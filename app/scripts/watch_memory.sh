#!/usr/bin/env bash
# YUE2 // GROOVE memory guard — a live view of the numbers that killed this Mac.
#
# On 2026-09-13 22:29 this machine panicked while a generation was running:
#
#   panic: watchdog timeout: no checkins from watchdogd in 91 seconds
#   Compressor Info: 100% of segments limit (BAD) with 73 swapfiles
#
# The kernel spent so much time compressing and swapping that even watchdogd was
# starved — no hardware fault, just memory.  The root cause was that PyTorch's
# MPS allocator was allowed to grow to 88 GiB on a 64 GiB machine (its default
# PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.7 of a 51.8 GiB "recommended max"), on top
# of everything else running here.  `.env` now caps it at 0.8 ≈ 41.5 GiB.
#
# Run this in a second terminal while a generation runs, and watch the two
# columns that matter: SWAP and COMPR.  Both should stay near zero.  If COMPR
# keeps climbing while FREE% falls, stop the generation yourself — do not let
# the machine reach the state that panicked it.
#
# Usage:
#   bash scripts/watch_memory.sh                 # every 5 s, until stopped
#   bash scripts/watch_memory.sh --interval 2    # every 2 s
#   bash scripts/watch_memory.sh --count 60      # stop after 60 samples (~5 min)
#   bash scripts/watch_memory.sh --once          # one snapshot, then exit
#   bash scripts/watch_memory.sh --quiet         # only print when the state changes
#   bash scripts/watch_memory.sh --log           # also flag MPS OOM lines in the run log
#
# Stopping: Ctrl-C in the foreground.  If you put it in the background with a
# trailing `&`, the shell makes the job ignore SIGINT, so Ctrl-C will not reach
# it — use `kill <pid>` (SIGTERM) instead; that is trapped and exits cleanly.
#
# Exit status with --once / --count: 0 healthy, 1 warning, 2 critical.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

INTERVAL=5
ONCE=0
WATCH_LOG=0
QUIET=0
MAX_SAMPLES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --count)    MAX_SAMPLES="$2"; shift 2 ;;
    --once)     ONCE=1; shift ;;
    --quiet)    QUIET=1; shift ;;
    --log)      WATCH_LOG=1; shift ;;
    -h|--help)  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# Thresholds, chosen against what the panic log actually showed (73 swapfiles,
# compressor segments exhausted).  Warn well before that point, not at it.
FREE_WARN_PCT=25
FREE_CRIT_PCT=12
SWAP_WARN_MIB=8192
SWAP_CRIT_MIB=16384
COMPR_WARN_MIB=8192
COMPR_CRIT_MIB=16384
HEARTBEAT=12        # --quiet prints a heartbeat line every this many samples

RUN_DIR="${YUE2_GROOVE_RUNS:-$ROOT/runs}"
LOG_FILE="$RUN_DIR/.server.log"

# Optional local configuration: display the watermark caps that serve.sh will export.
WATERMARK_HIGH="unset (PyTorch default 1.7)"
WATERMARK_LOW="unset (PyTorch default 1.4)"
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
  WATERMARK_HIGH="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-$WATERMARK_HIGH}"
  WATERMARK_LOW="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-$WATERMARK_LOW}"
  RUN_DIR="${YUE2_GROOVE_RUNS:-$RUN_DIR}"
  LOG_FILE="$RUN_DIR/.server.log"
fi

use_color=0
[ -t 1 ] && use_color=1
paint() {  # paint COLOR TEXT
  if [ "$use_color" = "1" ]; then printf '\033[%sm%s\033[0m' "$1" "$2"; else printf '%s' "$2"; fi
}
num() {    # num VALUE FALLBACK — integers only; anything odd becomes the fallback
  case "${1:-}" in
    ''|*[!0-9]*) printf '%s' "$2" ;;
    *) printf '%s' "$1" ;;
  esac
}

# ── raw readings ─────────────────────────────────────────────────────────────

read_free_pct() {   # system-wide free percentage, integer
  memory_pressure -Q 2>/dev/null \
    | awk '/free percentage/{gsub(/[^0-9]/, "", $NF); print $NF; exit}' || echo "?"
}

read_pressure() {   # 1 normal, 2 warning, 4 critical (kernel memorystatus)
  sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo "0"
}

read_swap_mib() {   # swap used, MiB
  sysctl -n vm.swapusage 2>/dev/null | awk '{
    v = $6; u = substr(v, length(v)); n = substr(v, 1, length(v) - 1) + 0;
    if (u == "G") n *= 1024;
    printf "%d", n;
  }' || echo 0
}

read_compr_mib() {  # pages held by the compressor, MiB
  local ps pages
  ps="$(sysctl -n hw.pagesize || echo 16384)"
  pages="$(vm_stat 2>/dev/null | awk '/occupied by compressor/{gsub(/[^0-9]/, "", $5); print $5 + 0; exit}' || echo 0)"
  echo $(( pages * ps / 1048576 ))
}

read_swapfiles() {
  find /private/var/vm -maxdepth 1 -name 'swapfile*' 2>/dev/null | wc -l | tr -d ' ' || true
}

read_top_proc() {   # biggest RSS process, "name 1234M"
  ps -Ao rss,comm -m 2>/dev/null | awk 'NR == 2 {
    name = $2; for (i = 3; i <= NF; i++) name = name " " $i;
    n = split(name, parts, "/");
    printf "%s %dM", parts[n], $1 / 1024;
  }'
}

# ── verdict ──────────────────────────────────────────────────────────────────

verdict() {  # verdict FREE PRESSURE SWAP_MIB COMPR_MIB  → echoes "OK"|"WARN"|"CRIT"
  # anything that is not an integer degrades to a benign default, so this can be
  # called with whatever the sensors returned (or nothing at all) without
  # spraying "integer expression expected" from the numeric tests below
  local free pressure swap compr
  free="$(num "${1:-?}" '?')"
  pressure="$(num "${2:-0}" 0)"
  swap="$(num "${3:-0}" 0)"
  compr="$(num "${4:-0}" 0)"
  if [ "$free" != "?" ] && [ "$free" -le "$FREE_CRIT_PCT" ]; then echo CRIT; return; fi
  if [ "$pressure" -ge 4 ]; then echo CRIT; return; fi
  if [ "$swap" -ge "$SWAP_CRIT_MIB" ] || [ "$compr" -ge "$COMPR_CRIT_MIB" ]; then echo CRIT; return; fi
  if [ "$free" != "?" ] && [ "$free" -le "$FREE_WARN_PCT" ]; then echo WARN; return; fi
  if [ "$pressure" -ge 2 ]; then echo WARN; return; fi
  if [ "$swap" -ge "$SWAP_WARN_MIB" ] || [ "$compr" -ge "$COMPR_WARN_MIB" ]; then echo WARN; return; fi
  echo OK
}

tag() {  # tag VERDICT → colored short tag
  case "$1" in
    OK)   paint 32 "OK  " ;;
    WARN) paint 33 "WARN" ;;
    CRIT) paint 31 "CRIT" ;;
    *)    printf '%s' "$1" ;;
  esac
}

oom_in_log() {
  [ "$WATCH_LOG" = "1" ] && [ -f "$LOG_FILE" ] && grep -q "MPS backend out of memory" "$LOG_FILE"
}

# ── main ─────────────────────────────────────────────────────────────────────

stopped=0
sleep_pid=""
stop() {
  [ "$stopped" = "1" ] && exit 0
  stopped=1
  # the tick's `sleep` is a child; drop it so the shell can exit at once
  [ -n "$sleep_pid" ] && kill "$sleep_pid" 2>/dev/null || true
  printf '\nstopped.\n'
  exit 0
}
# INT covers Ctrl-C in the foreground; TERM/HUP cover `kill <pid>`, which is the
# only way to stop a backgrounded copy (see the note at the top).
trap stop INT TERM HUP

printf 'YUE2 // GROOVE memory guard\n'
printf '  interval      : %ss · Ctrl-C to stop (background: kill <pid>)\n' "$INTERVAL"
if [ "$QUIET" = "1" ]; then
  printf '  output        : changes only, heartbeat every %s samples\n' "$HEARTBEAT"
fi
printf '  warn / crit   : FREE < %s%% / < %s%%   swap > %s MiB / %s MiB   compr > %s MiB / %s MiB\n' \
  "$FREE_WARN_PCT" "$FREE_CRIT_PCT" "$SWAP_WARN_MIB" "$SWAP_CRIT_MIB" "$COMPR_WARN_MIB" "$COMPR_CRIT_MIB"
printf '  MPS watermark : HIGH=%s  LOW=%s   (from .env; unset means PyTorch default)\n' \
  "$WATERMARK_HIGH" "$WATERMARK_LOW"
printf '  run log       : %s\n' "${LOG_FILE/#$HOME/~}"
if [ "$WATCH_LOG" = "1" ]; then
  if oom_in_log; then
    printf '  %s MPS backend out of memory already present in the log — the guard fired, not the kernel.\n' \
      "$(paint 33 'NOTE')"
  else
    printf '  log watch     : flagging "MPS backend out of memory" as it appears\n'
  fi
fi
printf '\n'
printf '%-8s  %6s  %8s  %8s  %10s  %11s  %-6s  %s\n' \
  TIME FREE% PRESSURE SWAP COMPR SWAPFILES STATE "TOP PROCESS (RSS)"

last_state=""
samples=0
while :; do
  free_pct="$(read_free_pct)"
  pressure="$(read_pressure)"
  swap_mib="$(read_swap_mib)"
  compr_mib="$(read_compr_mib)"
  swapfiles="$(read_swapfiles)"
  top_proc="$(read_top_proc)"
  # coerce every reading to an integer (or "?" for FREE%) so a weird value can
  # never reach a numeric test and spray "integer expression expected" errors
  free_pct="$(num "$free_pct" '?')"
  pressure="$(num "$pressure" 0)"
  swap_mib="$(num "$swap_mib" 0)"
  compr_mib="$(num "$compr_mib" 0)"
  state="$(verdict "$free_pct" "$pressure" "$swap_mib" "$compr_mib")"

  line="$(printf '%s  %5s%%  %8s  %6sMiB  %8sMiB  %10s  [%s]  %s' \
    "$(date +%T)" "$free_pct" "$pressure" "$swap_mib" "$compr_mib" "$swapfiles" \
    "$(tag "$state")" "$top_proc")"

  samples=$(( samples + 1 ))
  # --quiet keeps the terminal usable during a long generation: the opening
  # sample, every transition, and a slow heartbeat — nothing else.
  if [ "$QUIET" != "1" ] \
     || [ "$samples" = "1" ] \
     || [ "$state" != "$last_state" ] \
     || [ $(( samples % HEARTBEAT )) -eq 0 ]; then
    printf '%s\n' "$line"
  fi

  # Only speak up when the picture changes, and only when it gets worse: a wall
  # of repeated WARN lines would be noise, one line per transition is a signal.
  if [ "$state" != "$last_state" ]; then
    case "$state" in
      WARN) echo "  → memory pressure rising: FREE ${free_pct}%, swap ${swap_mib}MiB, compressor ${compr_mib}MiB." ;;
      CRIT) echo "  → STOP the generation now (serve.sh stop or Ctrl-C in the UI). This is the state that panicked the machine." ;;
    esac
    last_state="$state"
  fi

  if [ "$WATCH_LOG" = "1" ] && oom_in_log && [ "${oom_seen:-0}" = "0" ]; then
    echo "  → 'MPS backend out of memory' appeared in ${LOG_FILE/#$HOME/~}: the watermark cap did its job."
    oom_seen=1
  fi

  if [ "$ONCE" = "1" ] \
     || { [ "$MAX_SAMPLES" -gt 0 ] && [ "$samples" -ge "$MAX_SAMPLES" ]; }; then
    case "$state" in
      OK) exit 0 ;;
      WARN) exit 1 ;;
      *) exit 2 ;;
    esac
  fi
  # Sleeping in the background and `wait`ing for it is what makes Ctrl-C / kill
  # take effect immediately: bash only runs a trap once the foreground command
  # finishes, so a plain `sleep 5` would delay the shutdown by up to 5 s.
  sleep "$INTERVAL" &
  sleep_pid=$!
  wait "$sleep_pid" 2>/dev/null || true
  sleep_pid=""
done
