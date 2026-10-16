#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -uo pipefail
export INFB_VERIFIER_T0=$(date +%s)   # the verifier's clock starts here
trap 'if [ ! -f /logs/verifier/reward.json ] && [ ! -f /logs/verifier/reward.txt ]; then mkdir -p /logs/verifier; echo -1 > /logs/verifier/reward.txt; fi' EXIT
log() { echo "[verifier] $*"; }
cd /app || { mkdir -p /logs/verifier; exit 6; }

python3 /tests/grader.py prepare || exit $?
[ -f /logs/verifier/reward.json ] && exit 0   # model.patch didn't apply -> graded 0

export RUN_LOG=/logs/verifier/run.log
: > "$RUN_LOG" 2>/dev/null || true
run_log() { echo "+ $*" >> "$RUN_LOG" 2>/dev/null; "$@" 2>&1 | tee -a "$RUN_LOG"; return "${PIPESTATUS[0]}"; }

# >>> RUN TESTS (task-specific) <<<
export PATH="/opt/vllm/bin:/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 CARGO_NET_OFFLINE=true
PY=/opt/vllm/bin/python
BASE="$(python3 -c 'import json; print(json.load(open("/tests/config.json"))["base_commit"])')"
cfg() { python3 -c 'import json,sys; v=json.load(open("/tests/config.json"))["infb"]; [v := v[k] for k in sys.argv[1:]]; print(" ".join(v) if isinstance(v, list) else v)' "$@"; }
# Each bounded group's exit, for the reward's disposition: its name, `timeout`'s rc, its bound, whether that bound
# is a composed wall or one operation's own public limit, and the junit report it writes.
: > /logs/verifier/groups.jsonl
group_record() { printf '{"group": "%s", "rc": %d, "bound_s": %d, "bound": "%s", "junit": "%s"}\n' "$1" "$2" "$3" "$4" "$5" >> /logs/verifier/groups.jsonl; }
reap_guard() {  # reap_guard <group> <argv...>: exit 0 when nothing matches, 1 on a match, 2 when the pattern is unreadable
  "$PY" -I - "$@" <<'REAPGUARD'
import os, re, sys
m = re.search(r'^REAP_PATTERN = re\.compile\(r"([^"]+)"\)', open("/tests/infb_efficiency/nodes.py").read(), re.M)
if not m:
    print("[reap-guard] REAP_PATTERN is not readable in tests/infb_efficiency/nodes.py")
    sys.exit(2)
pattern, group, argv = re.compile(m.group(1)), sys.argv[1], sys.argv[2:]
hits = ["its command line: " + " ".join(argv)[:300]] if pattern.search(" ".join(argv)) else []
pid, seen = os.getppid(), set()
while pid > 1 and pid not in seen:
    seen.add(pid)
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
        ppid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        break
    if pattern.search(cmd):
        hits.append(f"ancestor {pid}: {cmd[:300]}")
    pid = ppid
if hits:
    print(f"[reap-guard] {group}: {pattern.pattern!r} matches " + " | ".join(hits))
    sys.exit(1)
print(f"[reap-guard] {group}: {pattern.pattern!r} matches no command line of this session")
REAPGUARD
}
REAP_FAULT="the reap guard found a command line of this verifier session that the IPC reap would kill, or could not read its pattern: a verifier fault; no reward written (sentinel -1)"
ARGS=/logs/verifier/args
mkdir -p "$ARGS"
printf '%s\n' "--deselect=infb_verify/test_hidden.py::test_served_configuration" > "$ARGS/session.args"
printf '%s\n' "/tests/infb_verify/test_hidden.py::test_served_configuration" \
    "/tests/infb_verify/test_sparse_indexer.py::test_sparse_indexer_serving_binding" > "$ARGS/runtime.args"
printf '%s\n' "--deselect=infb_verify/test_sparse_indexer.py::test_sparse_indexer_serving_binding" > "$ARGS/sparse-components.args"

# 1. The tests come from the verifier image, never from the agent's tree. grader.py prepare applied the
#    candidate's model.patch (which may touch anything) and then test.patch; restore upstream's test package
#    to the base blobs, drop anything the candidate added there, and re-apply the vendored upstream tests.
#    `git checkout <base> -- tests` alone is not a reset (it leaves files a patch ADDED), so the tree is
#    removed first.
rm -rf tests 2>>"$RUN_LOG"
git checkout -q "$BASE" -- tests 2>>"$RUN_LOG" || { log "tests reset failed"; exit 4; }
git clean -fdq -- tests 2>>"$RUN_LOG" || true
git apply --whitespace=nowarn --allow-empty /tests/test.patch 2>>"$RUN_LOG" || { log "test.patch re-apply failed"; exit 4; }
git status --porcelain -- tests >> "$RUN_LOG" 2>&1 || true
rm -rf eval/deepseek_v41_flash 2>>"$RUN_LOG"
git checkout -q "$BASE" -- eval/deepseek_v41_flash 2>>"$RUN_LOG" || { log "evaluation materials reset failed"; exit 4; }
git clean -fdqx -- eval/deepseek_v41_flash 2>>"$RUN_LOG" || true
git status --porcelain -- eval >> "$RUN_LOG" 2>&1 || true

if unshare -r -n python3 -I /tests/infb_verify/netblock.py -- true 2>>"$RUN_LOG"; then
    NB=(unshare -r -n python3 -I /tests/infb_verify/netblock.py --)
elif python3 -I /tests/infb_verify/netblock.py --probe 2>>"$RUN_LOG"; then
    NB=()
else
    log "void: the outside is reachable and the runtime refuses a network namespace; no reward written (sentinel -1)"
    exit 3
fi
log "network: $([ ${#NB[@]} -gt 0 ] && echo 'blocked in a namespace per phase' || echo "blocked by the backend")"

run_log timeout --kill-after="$(cfg preflight kill_after_s)" "$(cfg preflight budget_s)" "${NB[@]}" "$PY" /tests/preflight.py \
    || { log "preflight failed or overran its bound; no reward written (sentinel -1)"; exit 3; }

SPARSE_JIT_ROOT=/tmp/infb-sparse-jit
unset DSV41_SPARSE_JIT_CACHE_DIR
rm -rf "$SPARSE_JIT_ROOT" "${HOME:-/root}/.dj" 2>>"$RUN_LOG"
export INFB_SPARSE_JIT_ROOT="$SPARSE_JIT_ROOT"

mkdir -p /logs/verifier/efficiency/test_hidden
python3 /tests/infb_verify/host_load.py sample --out /logs/verifier/efficiency/test_hidden/host-load.jsonl \
    >> "$RUN_LOG" 2>&1 &
HOST_LOAD_PID=$!

run_log timeout --kill-after="$(cfg reference_phase kill_after_s)" "$(cfg reference_phase budget_s)" \
    "${NB[@]}" "$PY" -I /tests/infb_efficiency/reference.py --config /tests/config.json \
    || { log "the reference could not be served or measured on this host within its bound; no reward written (sentinel -1)"; exit 3; }

L0=( "$PY" -m pytest -p no:cacheprovider --rootdir=/tests -q /tests/infb_verify/test_l0.py \
    --junitxml=/logs/verifier/junit-l0.xml )
run_log reap_guard l0 "${L0[@]}" || { log "l0: $REAP_FAULT"; exit 3; }
run_log env INFB_REBUILD_TIMEOUT_SEC="$(cfg l0 rebuild_timeout_s)" \
    timeout --kill-after="$(cfg l0 kill_after_s)" "$(cfg l0 budget_s)" "${NB[@]}" "${L0[@]}"
group_record l0 $? "$(cfg l0 budget_s)" wall junit-l0.xml

SPARSE_ID=$(python3 -I -c 'import importlib.util as u, pathlib as p; s=u.spec_from_file_location("infb_native", "/tests/eval/deepseek_v41_flash/tests/native.py"); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.sparse_source_id(p.Path("/app")))' 2>>"$RUN_LOG") || SPARSE_ID=unreadable
export DSV41_SPARSE_JIT_CACHE_DIR="$SPARSE_JIT_ROOT/candidate/$SPARSE_ID/$(cfg efficiency reference sparse_build arch)"
mkdir -p "$DSV41_SPARSE_JIT_CACHE_DIR"
log "sparse components: the candidate's sparse source id $SPARSE_ID; JIT cache $DSV41_SPARSE_JIT_CACHE_DIR"
cd /tests || { log "no /tests"; exit 5; }
SPARSE=( "$PY" -m pytest -p no:cacheprovider --rootdir=/tests -q -s /tests/infb_verify/test_sparse_indexer.py \
    "@$ARGS/sparse-components.args" --junitxml=/logs/verifier/junit-sparse-components.xml )
run_log reap_guard sparse-components "${SPARSE[@]}" || { log "sparse-components: $REAP_FAULT"; exit 3; }
run_log timeout --kill-after=60 "$(cfg sparse_indexer group_bounds components_s)" "${NB[@]}" "${SPARSE[@]}"
group_record sparse-components $? "$(cfg sparse_indexer group_bounds components_s)" wall junit-sparse-components.xml
cd /app || { log "no /app"; exit 5; }

cd /app || { log "no /app"; exit 5; }
# The deployment node runs in its own group, so the session deselects it by its node id, which pytest matches
#    as a prefix of ids relative to --rootdir: `infb_verify/...`, never `/tests/infb_verify/...` (an absolute path
#    deselects nothing, and the node then fails the session on setup beside the session's server). The id
AFTER=$(python3 -c 'import json; x=json.load(open("/tests/config.json"))["infb"]; p=x["preservation"]; print(int(x["runtime_evidence"]["group_budget_s"]) + 60 + len(p["files"]) * (int(p["budget_per_file_s"]) + 60))')
END_BY=$("$PY" -I /tests/infb_efficiency/deadline.py end-by --config /tests/config.json --after "$AFTER") \
    || { log "no verifier clock in config.json: the relaunch cannot run (skipped_no_verifier_clock)"; END_BY=""; }
RUNNER=( "$PY" -I /tests/infb_efficiency/deadline.py run --group test_hidden \
    --budget-s "$(cfg session budget_s)" --config /tests/config.json ${END_BY:+--end-by "$END_BY"} \
    --junit /logs/verifier/junit-hidden.xml \
    --results /logs/verifier/efficiency/test_hidden/session-results.jsonl \
    --record /logs/verifier/efficiency/test_hidden/efficiency.json \
    --summary /logs/verifier/efficiency/test_hidden/deadline-summary.json \
    --expect-prefix infb_verify.test_hidden.test_0 --expect-prefix infb_verify.test_hidden.test_gsm8k \
    --expect-prefix infb_verify.test_fidelity. --expect-prefix infb_verify.test_text_semantics. \
    --cuda-devices 0,1,2,3 --always-reap -- )
SESSION=( "$PY" -m pytest -p infb_efficiency.deadline -p no:cacheprovider --rootdir=/tests -q -s \
    /tests/infb_verify/test_hidden.py /tests/infb_verify/test_fidelity.py /tests/infb_verify/test_text_semantics.py \
    "@$ARGS/session.args" --junitxml=/logs/verifier/junit-hidden.xml )
run_log reap_guard session "${RUNNER[@]}" "${SESSION[@]}" || { log "session: $REAP_FAULT"; exit 3; }
run_log "${NB[@]}" env PYTHONPATH=/tests "${RUNNER[@]}" "${SESSION[@]}"
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >> "$RUN_LOG" 2>&1 || true
kill "$HOST_LOAD_PID" 2>/dev/null || true
run_log python3 /tests/infb_verify/host_load.py summarize \
    --samples /logs/verifier/efficiency/test_hidden/host-load.jsonl \
    --sidecar /logs/verifier/efficiency/test_hidden/requests.jsonl.gz \
    --out /logs/verifier/efficiency/test_hidden/host-load-by-draw.json || true
run_log python3 -I /tests/infb_verify/slow_host.py --config /tests/config.json \
    --results /logs/verifier/efficiency/test_hidden/session-results.jsonl \
    --record /logs/verifier/efficiency/test_hidden/efficiency.json \
    --out /logs/verifier/efficiency/test_hidden/slow-host.json
SLOW_HOST_RC=$?
if [ "$SLOW_HOST_RC" = 3 ]; then log "the slow-host rule: a host fault; no reward written (sentinel -1)"; exit 3; fi
[ "$SLOW_HOST_RC" = 0 ] || log "the slow-host check failed (rc $SLOW_HOST_RC); grading goes on as before the rule"

cd /app || { log "no /app"; exit 5; }
RUNTIME=( "$PY" -m pytest -p no:cacheprovider --rootdir=/tests -q -s "@$ARGS/runtime.args" \
    --junitxml=/logs/verifier/junit-runtime.xml )
run_log reap_guard runtime "${RUNTIME[@]}" || { log "runtime: $REAP_FAULT"; exit 3; }
run_log timeout --kill-after=60 "$(cfg runtime_evidence group_budget_s)" "${NB[@]}" "${RUNTIME[@]}"
group_record runtime $? "$(cfg runtime_evidence group_budget_s)" wall junit-runtime.xml
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true

cd /app || { log "no /app"; exit 5; }
PRES_FILES=$(python3 -c 'import json; print(" ".join(sorted(json.load(open("/tests/config.json"))["infb"]["preservation"]["files"])))')
k=0
for f in $PRES_FILES; do
  mapfile -t PRES_IDS < <(python3 -c 'import json,sys; print("\n".join(json.load(open("/tests/config.json"))["infb"]["preservation"]["files"][sys.argv[1]]))' "$f")
  PRES=( "$PY" -m pytest -p no:cacheprovider --rootdir=/app -q "${PRES_IDS[@]}" \
      --junitxml="/logs/verifier/junit-preservation-$k.xml" )
  run_log reap_guard "preservation-$k" "${PRES[@]}" || { log "preservation-$k: $REAP_FAULT"; exit 3; }
  run_log timeout --kill-after=60 "$(cfg preservation budget_per_file_s)" "${NB[@]}" "${PRES[@]}"
  group_record "preservation-$k" $? "$(cfg preservation budget_per_file_s)" operation_limit "junit-preservation-$k.xml"
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  k=$((k + 1))
done

run_log python3 -I /tests/infb_verify/disposition.py --config /tests/config.json --verifier-dir /logs/verifier \
    || { log "the reward's disposition could not be computed: a verifier fault, no reward (sentinel -1)"; echo -1 > /logs/verifier/reward.txt; }
# >>> END RUN TESTS <<<

_seen=""
for _rl in "$RUN_LOG" /logs/verifier/*_run.log /logs/verifier/*-run.log /logs/verifier/*-mocha.log /logs/verifier/*.log /logs/verifier/*.out; do
  [ -f "$_rl" ] && [ -s "$_rl" ] || continue
  case " $_seen " in *" $_rl "*) continue ;; esac
  case "${_rl##*/}" in *convert*.log|ctrf*.log|junit*.log) continue ;; esac
  _seen="$_seen $_rl"
  echo "===== raw suite output: ${_rl##*/} ====="
  cat "$_rl"
done 2>/dev/null
echo "===== grade ====="

python3 /tests/grader.py grade
log "reward.json=$(cat /logs/verifier/reward.json 2>/dev/null)"

mkdir -p /logs/verifier/reports 2>/dev/null
for _f in /logs/verifier/*; do
  case "${_f##*/}" in
    reward.json|reward.txt|ctrf.json|run.log|test-stdout.txt|reports) continue ;;
  esac
  [ -f "$_f" ] && mv -f "$_f" /logs/verifier/reports/ 2>/dev/null
done
