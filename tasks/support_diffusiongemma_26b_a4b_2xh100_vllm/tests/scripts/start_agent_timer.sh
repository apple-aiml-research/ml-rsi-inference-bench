#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -u

TIMER_DIR="${INFB_WORKSPACE:-/app}/.timer"
TELEMETRY_DIR="${INFB_ARTIFACTS_DIR:-/logs/artifacts}/timer"
mkdir -p "$TELEMETRY_DIR" 2>/dev/null || TELEMETRY_DIR=""
[ -n "$TELEMETRY_DIR" ] && [ -w "$TELEMETRY_DIR" ] || TELEMETRY_DIR=""
PID_FILE="$TIMER_DIR/writer.pid"

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
            echo "[timer] NOTE: this is the task's declared default; a pier --agent-timeout-multiplier is"
            echo "[timer]       NOT visible in the container. Set"
            echo "[timer]       INFB_AGENT_TIMER_BUDGET_SEC in [environment.env] to align the timer with an override."
            ;;
    esac
fi

if [ -z "$budget" ] || [ "$budget" -le 0 ]; then
    echo "[timer] no positive budget in the environment; not starting a countdown"
    exit 0
fi

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

        _cg="${INFB_CGROUP_ROOT:-/sys/fs/cgroup}"
        _cg_proc="${INFB_PROC_CGROUP:-/proc/self/cgroup}"
        _cg_self_pid="${INFB_SELF_PID:-$$}"
        _cg_leaf=""
        _cg_scope="fallback"
        _cg_tried=""
        if [ -r "$_cg_proc" ]; then
            _cg_tried="$(awk -F: '$1=="0"{print $3; exit}' "$_cg_proc" 2>/dev/null)"
        fi
        for _cand in "$_cg"/*/*/*/ "$_cg"/*/*/ "$_cg"/*/; do
            [ -d "$_cand" ] || continue
            _cand="${_cand%/}"
            [ -r "$_cand/cgroup.procs" ] || continue
            if grep -qx "$_cg_self_pid" "$_cand/cgroup.procs" 2>/dev/null; then
                _cg_leaf="$_cand"
                break
            fi
        done
        if [ -z "$_cg_leaf" ] && [ -n "$_cg_tried" ]; then
            _cg_try="$_cg_tried"
            _cg_guard=0
            while [ -n "$_cg_try" ] && [ "$_cg_try" != "/" ] && [ "$_cg_guard" -lt 24 ]; do
                if [ -r "$_cg$_cg_try/memory.current" ]; then
                    _cg_leaf="$_cg$_cg_try"
                    break
                fi
                _cg_rest="${_cg_try#/}"
                case "$_cg_rest" in
                    */*) _cg_try="/${_cg_rest#*/}" ;;
                    *) _cg_try="" ;;
                esac
                _cg_guard=$((_cg_guard + 1))
            done
        fi
        if [ -n "$_cg_leaf" ]; then
            _cg="$_cg_leaf"
            _cg_scope="leaf"
        fi
        if [ -r "$_cg/memory.current" ]; then
            _mem_cur="$(cat "$_cg/memory.current" 2>/dev/null || echo)"
            _mem_max="$(cat "$_cg/memory.max" 2>/dev/null || echo)"
            _mem_peak="$(cat "$_cg/memory.peak" 2>/dev/null || echo)"
        elif [ -r "$_cg/memory/memory.usage_in_bytes" ]; then
            _mem_cur="$(cat "$_cg/memory/memory.usage_in_bytes" 2>/dev/null || echo)"
            _mem_max="$(cat "$_cg/memory/memory.limit_in_bytes" 2>/dev/null || echo)"
            _mem_peak="$(cat "$_cg/memory/memory.max_usage_in_bytes" 2>/dev/null || echo)"
        else
            _mem_cur=""; _mem_max=""; _mem_peak=""
        fi
        if [ -n "$_mem_cur" ]; then
            _mem_line="$(printf '%s current=%s peak=%s limit=%s scope=%s cgproc=%s' \
                "$elapsed" "$_mem_cur" "${_mem_peak:-unknown}" "${_mem_max:-unknown}" \
                "$_cg_scope" "${_cg_tried:-none}")"
            printf '%s\n' "$_mem_line" >> "$TIMER_DIR/memory_samples"
            printf '%s\n' "${_mem_peak:-$_mem_cur}" > "$TIMER_DIR/.memory_peak.tmp.$$"
            mv -f "$TIMER_DIR/.memory_peak.tmp.$$" "$TIMER_DIR/memory_peak"
            if [ -n "${TELEMETRY_DIR:-}" ]; then
                printf '%s\n' "$_mem_line" >> "$TELEMETRY_DIR/memory_samples" 2>/dev/null || true
                printf '%s\n' "${_mem_peak:-$_mem_cur}" > "$TELEMETRY_DIR/memory_peak" 2>/dev/null || true
            fi
        fi

        [ "$remaining" -le 0 ] && break
        sleep "${INFB_TIMER_TICK_SEC:-10}"
    done
}

printf '%s\n' "$budget" > "$TIMER_DIR/.remaining_secs.tmp.$$"
mv -f "$TIMER_DIR/.remaining_secs.tmp.$$" "$TIMER_DIR/remaining_secs"
printf '%s\n' 0 > "$TIMER_DIR/.elapsed_secs.tmp.$$"
mv -f "$TIMER_DIR/.elapsed_secs.tmp.$$" "$TIMER_DIR/elapsed_secs"

if command -v setsid >/dev/null 2>&1; then
    setsid bash -c "$(declare -f writer); TIMER_DIR='$TIMER_DIR' TELEMETRY_DIR='$TELEMETRY_DIR' budget='$budget' writer" \
        >/dev/null 2>&1 &
else
    nohup bash -c "$(declare -f writer); TIMER_DIR='$TIMER_DIR' TELEMETRY_DIR='$TELEMETRY_DIR' budget='$budget' writer" \
        >/dev/null 2>&1 &
fi
printf '%s\n' "$!" > "$PID_FILE"
echo "[timer] countdown started: budget=${budget}s dir=$TIMER_DIR pid=$!"
exit 0
