#!/usr/bin/env bash
# Swap the running sweep driver for the code staged in ~/clef-quant/bench-staged WITHOUT losing finished work. Run ON
# the Spark, detached (nohup setsid ... &): the wait for the unit boundary can take as long as one unit (~10 min, 27B ref).
#   1. touch sweep-state/STOP: the worker finishes its current unit and exits at the unit boundary, the scheduler merges
#      what is complete, the driver exits (DRIVER_STOP) and removes STOP. Finished sets and finished units stay; the
#      relaunched driver re-runs only the interrupted unit (driver.py docstring), and re-queues any config whose outputs
#      are missing (an abandoned one included: the failure counter is in-memory, sched.py).
#   2. the watchdog timer is paused from STOP until the driver is back, so the driver-down window is not reported as an
#      outage; a trap restarts the timer on every exit path.
#   3. rsync bench-staged/ -> bench/ (no --delete), then the optional post-sync check command. If it fails the driver is
#      NOT relaunched (the old code is gone from bench/, so a relaunch would run unverified code) and the message says so.
#   4. relaunch, log before/after progress, one Signal message via alerts/watchdog.py.
# Usage:  sweep/swap_driver.sh "<reason, goes into the Signal message>" [post-sync check command...]
# Log:    ~/clef-quant/logs/swap-driver.log
set -uo pipefail
cd ~/clef-quant
LOG=~/clef-quant/logs/swap-driver.log
REASON="${1:?reason for the swap}"; shift || true
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
driver_pid() { pgrep -f "python3 -m sweep.driver run" | head -1 || true; }
progress() { (cd ~/clef-quant/bench && timeout 300 python3 -m sweep.status --jobs sweep/jobs.jsonl 2>/dev/null | head -1); }
signal() {  # one message; a send failure is logged, never fatal
  (cd ~/clef-quant/alerts && python3 -c '
import json, sys
import watchdog as w
cfg = json.loads(open("config.json").read())
w.send(cfg, sys.argv[1], False)' "$1") >>"$LOG" 2>&1 || log "signal send failed (see above)"
}
trap 'systemctl --user start clef-watchdog.timer' EXIT

before="$(progress)"
log "swap requested: $REASON"
log "before: $before"
pid="$(driver_pid)"
if [ -z "$pid" ]; then log "no driver running; nothing to swap"; exit 1; fi
systemctl --user stop clef-watchdog.timer
fails_before=$(grep -c " FAIL " logs/sweep.log || true)
log "watchdog paused; touching STOP; waiting for the unit boundary (driver pid $pid)"
touch sweep-state/STOP
while [ -n "$(driver_pid)" ]; do sleep 15; done
log "driver exited; last sweep.log line: $(tail -1 logs/sweep.log)"

rsync -a --exclude __pycache__ --exclude .pytest_cache bench-staged/ bench/
log "code synced from bench-staged"
if [ $# -gt 0 ]; then
  if out="$("$@" 2>&1 | tail -3)"; then
    log "post-sync check OK: $out"
  else
    log "post-sync check FAILED: $out"
    log "driver NOT relaunched. Fix bench/, then: cd ~/clef-quant/bench && nohup setsid python3 -m sweep.driver run --jobs sweep/jobs.jsonl >> ~/clef-quant/logs/sweep-driver.out 2>&1 < /dev/null &"
    signal "[clef-bench] driver swap FAILED its post-sync check; driver NOT relaunched. $REASON
$(tail -3 "$LOG")"
    exit 1
  fi
fi

cd ~/clef-quant/bench
nohup setsid python3 -m sweep.driver run --jobs sweep/jobs.jsonl >> ~/clef-quant/logs/sweep-driver.out 2>&1 < /dev/null &
sleep 90   # let STAGE_START / WORKER_START land in sweep.log
pid="$(driver_pid)"
log "relaunched driver pid ${pid:-NONE}: $(tail -1 ~/clef-quant/logs/sweep.log)"
after="$(progress)"
log "after: $after"
if [ -z "$pid" ]; then
  signal "[clef-bench] driver swap: relaunch did NOT come up. $REASON
$(tail -4 "$LOG")"
  exit 1
fi
# FAIL lines logged between STOP and the relaunch are the OLD code's last attempts. The paused watchdog reports them on
# its next tick, after this message; say so here, or that alert reads as a failure of the new code (2026-10-18 07:15).
fails_after=$(grep -c " FAIL " ~/clef-quant/logs/sweep.log || true)
stale=""
if [ "$fails_after" -gt "$fails_before" ]; then
  stale="
Note: $((fails_after - fails_before)) FAIL line(s) were logged between STOP and this relaunch, by the code being replaced. The watchdog will also report them on its next tick; they predate the new code."
fi
log "pre-swap FAIL lines the watchdog will still report: $((fails_after - fails_before))"
signal "[clef-bench] driver swapped at a unit boundary. $REASON
before: $before
after:  $after
sweep.log: $(tail -2 ~/clef-quant/logs/sweep.log | cut -c1-200)$stale"
