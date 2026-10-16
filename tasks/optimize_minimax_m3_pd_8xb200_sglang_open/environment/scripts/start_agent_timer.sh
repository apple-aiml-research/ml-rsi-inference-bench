#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Expose the agent's remaining time budget as files, on EVERY backend.
#
# The instruction tells the agent:
#
#     You have a fixed time budget: `cat /app/.timer/remaining_secs` shows the seconds left
#     (also `/app/.timer/elapsed_secs`, and `test -f /app/.timer/alert_5min` appears as the
#     deadline nears). Pace yourself ...
#
# The timer is a TASK affordance, not an agent affordance: Pier runs a task's
# `[environment.healthcheck]` command inside the container after the environment starts and before
# agent setup, on every backend (pier/trial/trial.py: _setup_environment -> run_healthcheck ->
# _setup_agent -> _execute_agent). That is the one generic hook that gets us there without touching
# the vendored Pier and without changing a single word of the instruction.
#
# Contract: <workspace>/.timer/{remaining_secs,elapsed_secs} hold an integer plus a newline, written
# atomically (tmp file + mv), and alert_30min / alert_10min / alert_5min appear as empty files when
# the remaining budget drops to 30 / 10 / 5 minutes.
#
# Idempotent and single-writer: re-running it (healthcheck retries, or a second step) does not start
# a second countdown, and the pid file makes that true on every backend.
set -u

TIMER_DIR="${INFB_WORKSPACE:-/app}/.timer"
PID_FILE="$TIMER_DIR/writer.pid"

# The budget: the SMALLEST positive candidate, not the first one found. Every deadline-ish variable
# that is present votes and the earliest deadline wins, because over-reporting is the harmful
# direction: an agent that thinks it has hours starts work it cannot finish and can leave an
# untested commit as its final HEAD, which is a scored outcome. Under-reporting only makes it
# finish early.
#   INFB_AGENT_TIMER_BUDGET_SEC   the PUBLIC override, described below
#   INFB_TIMEOUT_SEC              an older name for the agent timeout
#   INFB_AGENT_TIMEOUT_SEC        baked by the TASK, for backends where no launcher injects any
budget=""
for name in INFB_AGENT_TIMER_BUDGET_SEC INFB_TIMEOUT_SEC INFB_AGENT_TIMEOUT_SEC; do
    value="$(printenv "$name" 2>/dev/null || true)"
    case "$value" in
        ''|*[!0-9]*) continue ;;
    esac
    [ "$value" -le 0 ] && continue
    if [ -z "$budget" ] || [ "$value" -lt "$budget" ]; then
        budget="$value"
        chosen="$name"
    fi
done
if [ -n "${budget:-}" ]; then
    echo "[timer] budget=${budget}s from ${chosen} (smallest of the candidates present)"
    case "$chosen" in
        INFB_AGENT_TIMEOUT_SEC)
            echo "[timer] NOTE: this is the task's declared default; a pier --override-timeout-sec / --max-timeout-sec /"
            echo "[timer]       --agent-timeout-multiplier is NOT visible in the container. Set"
            echo "[timer]       INFB_AGENT_TIMER_BUDGET_SEC in [environment.env] to align the timer with an override."
            ;;
    esac
fi

# No budget is not a failure: a step-scoped or untimed run legitimately has none. Exit 0 so the
# healthcheck passes — this script must never be the reason a trial does not start.
if [ -z "$budget" ] || [ "$budget" -le 0 ]; then
    echo "[timer] no positive budget in the environment; not starting a countdown"
    exit 0
fi

# Already running? Yield. `kill -0` on the recorded pid is the liveness test; a stale pid file from a
# previous container start is overwritten below.
if [ -f "$PID_FILE" ]; then
    existing="$(cat "$PID_FILE" 2>/dev/null || true)"
    case "$existing" in
        ''|*[!0-9]*) : ;;
        *) if kill -0 "$existing" 2>/dev/null; then
               echo "[timer] already running (pid $existing); nothing to do"
               exit 0
           fi ;;
    esac
fi

umask 022
mkdir -p "$TIMER_DIR"

# The countdown itself, detached so it outlives this exec. setsid where available (the writer must
# not die with the healthcheck's process group), nohup otherwise.
writer() {
    started="$(date +%s)"
    while :; do
        now="$(date +%s)"
        elapsed=$(( now - started ))
        [ "$elapsed" -lt 0 ] && elapsed=0
        remaining=$(( budget - elapsed ))
        [ "$remaining" -lt 0 ] && remaining=0

        printf '%s\n' "$remaining" > "$TIMER_DIR/.remaining_secs.tmp.$$"
        mv -f "$TIMER_DIR/.remaining_secs.tmp.$$" "$TIMER_DIR/remaining_secs"
        printf '%s\n' "$elapsed" > "$TIMER_DIR/.elapsed_secs.tmp.$$"
        mv -f "$TIMER_DIR/.elapsed_secs.tmp.$$" "$TIMER_DIR/elapsed_secs"

        [ "$remaining" -le 1800 ] && { [ -e "$TIMER_DIR/alert_30min" ] || : > "$TIMER_DIR/alert_30min"; }
        [ "$remaining" -le 600 ]  && { [ -e "$TIMER_DIR/alert_10min" ] || : > "$TIMER_DIR/alert_10min"; }
        [ "$remaining" -le 300 ]  && { [ -e "$TIMER_DIR/alert_5min" ]  || : > "$TIMER_DIR/alert_5min"; }

        [ "$remaining" -le 0 ] && break
        sleep "${INFB_TIMER_TICK_SEC:-10}"
    done
}

# Write the first tick SYNCHRONOUSLY, before backgrounding: the agent's very first round can read
# the file, and a healthcheck that returns before the file exists would leave a window in which the
# instruction is false — the exact defect this script removes.
printf '%s\n' "$budget" > "$TIMER_DIR/.remaining_secs.tmp.$$"
mv -f "$TIMER_DIR/.remaining_secs.tmp.$$" "$TIMER_DIR/remaining_secs"
printf '%s\n' 0 > "$TIMER_DIR/.elapsed_secs.tmp.$$"
mv -f "$TIMER_DIR/.elapsed_secs.tmp.$$" "$TIMER_DIR/elapsed_secs"

if command -v setsid >/dev/null 2>&1; then
    setsid bash -c "$(declare -f writer); TIMER_DIR='$TIMER_DIR' budget='$budget' writer" \
        >/dev/null 2>&1 &
else
    nohup bash -c "$(declare -f writer); TIMER_DIR='$TIMER_DIR' budget='$budget' writer" \
        >/dev/null 2>&1 &
fi
printf '%s\n' "$!" > "$PID_FILE"
echo "[timer] countdown started: budget=${budget}s dir=$TIMER_DIR pid=$!"
exit 0
