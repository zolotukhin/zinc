#!/usr/bin/env bash
# loop_watchdog.sh — keep an autonomous perf loop alive: restart its driver on
# EXIT *or* HANG (driver tee-LOG stale > STALE_SECS). Plain restart-on-exit misses
# hangs — a wedged claude -p blocks the driver without exiting, idling the GPU for
# hours. v2: derives LOG from EFFORT (no env-propagation risk); a stat FAILURE now
# counts as STALE → restart (v1's `|| echo now` made a failed stat look fresh →
# never restarted, the silent-failure bug); writes a heartbeat STATE file each
# check so you can verify it is actually watching.
#
#   WORKTREE=/path/to/dedicated-worktree DRIVER=run_perf_effort_4090.sh \
#   EFFORT=loops/efforts/MULTI_HOUR_EFFORT_27_CUDA_4090_DECODE.md \
#   nohup bash scripts/loop_watchdog.sh >/tmp/e27_watchdog.log 2>&1 &
#
# Env: WORKTREE, DRIVER (under loops/), EFFORT (path arg). LOG defaults to the
#   driver tee-log derived from EFFORT. STALE_SECS (2700=45m — keep > slowest legit
#   silent cycle), MAX_CYCLES (200), CHECK_SECS (300). Heartbeat:
#   /tmp/watchdog_<effort>.state (read it to confirm it's alive + the age it sees).
#   Stop: pkill -f loop_watchdog (then kill the driver PID it last logged).
set -u
WORKTREE="${WORKTREE:?}"; DRIVER="${DRIVER:?}"; EFFORT="${EFFORT:?}"
LOG="${LOG:-/tmp/perf_effort_$(basename "$EFFORT" .md).log}"
STALE_SECS="${STALE_SECS:-2700}"; MAX_CYCLES="${MAX_CYCLES:-200}"; CHECK_SECS="${CHECK_SECS:-300}"
STATE="/tmp/watchdog_$(basename "$EFFORT" .md).state"
wlog(){ echo "[watchdog $(date '+%m-%d %H:%M:%S')] $*"; }
start_driver() {
  ( cd "$WORKTREE" \
      && git fetch origin -q 2>/dev/null \
      && git checkout -q -f --detach origin/main 2>/dev/null
    MAX_CYCLES="$MAX_CYCLES" exec bash "loops/$DRIVER" "$EFFORT" ) &
  echo $!
}
PID=$(start_driver); wlog "started driver pid $PID (worktree $WORKTREE, watching $LOG, stale>${STALE_SECS}s)"
while true; do
  sleep "$CHECK_SECS"
  now=$(date +%s)
  if mt=$(stat -f %m "$LOG" 2>/dev/null); then age=$((now - mt)); else age=$((STALE_SECS + 1)); fi
  alive=0; kill -0 "$PID" 2>/dev/null && alive=1
  echo "$(date '+%m-%d %H:%M:%S') driver_pid=$PID alive=$alive log_age=${age}s thresh=${STALE_SECS}s log=$LOG" > "$STATE"
  if [ "$alive" -eq 0 ] || [ "$age" -gt "$STALE_SECS" ]; then
    wlog "RESTART (alive=$alive log_age=${age}s > ${STALE_SECS}s) — killing $PID + children"
    pkill -P "$PID" 2>/dev/null; kill -9 "$PID" 2>/dev/null; sleep 5
    PID=$(start_driver); wlog "new driver pid $PID"
  fi
done
