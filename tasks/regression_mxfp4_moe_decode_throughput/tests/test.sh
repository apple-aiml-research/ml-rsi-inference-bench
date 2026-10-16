#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

workspace="${INFB_WORKSPACE:-/app}"
verifier_dir="${INFB_VERIFIER_DIR:-/logs/verifier}"
artifacts_dir="${INFB_ARTIFACTS_DIR:-/logs/artifacts}"
tests_dir="${INFB_TESTS_DIR:-/tests}"
mkdir -p "$verifier_dir" "$artifacts_dir/raw"

python3 - <<'REAP' || echo "[test.sh] reap failed; continuing (the bench reports its own memory error)"
import os, signal, subprocess, time

REQUIRED_FRACTION, TIMEOUT_SEC = 0.92, 300.0


def _smi(query):
    out = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    return [l.strip() for l in out.stdout.splitlines() if l.strip()] if out.returncode == 0 else []


mine = {os.getpid(), os.getppid()}
pids = set()
for row in _smi("compute-apps=pid"):
    try:
        pids.add(int(row.split(",")[0]))
    except ValueError:
        pass
# Never signal our own process or our parent: this runs INSIDE the trial, and killing the shell that
# invoked it would abort the verification it exists to protect.
victims = sorted(pids - mine)
print(f"[reap] compute apps holding GPU memory: {sorted(pids)}; killing {victims}")
for pid in victims:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break
        except PermissionError:
            print(f"[reap] pid {pid}: not ours, leaving it")
            break
        time.sleep(3)

deadline = time.time() + TIMEOUT_SEC
while True:
    pairs = []
    for row in _smi("gpu=memory.free,memory.total"):
        try:
            free_mb, total_mb = (float(x) for x in row.split(",")[:2])
            pairs.append((free_mb / 1024.0, total_mb / 1024.0))
        except ValueError:
            pass
    if not pairs:
        print("[reap] nvidia-smi reported no devices; leaving the bench to say so")
        break
    worst_free, worst_total = min(pairs, key=lambda p: p[0] / p[1] if p[1] else 0.0)
    need = REQUIRED_FRACTION * worst_total
    if worst_free >= need:
        print(f"[reap] all {len(pairs)} GPUs free: min {worst_free:.1f} of {worst_total:.1f} GiB "
              f">= {need:.1f} GiB")
        break
    if time.time() >= deadline:
        print(f"[reap] TIMEOUT after {TIMEOUT_SEC:.0f}s: min free {worst_free:.1f} GiB < {need:.1f} GiB "
              f"-- the engine load below will fail and the trial will void, which is the honest outcome")
        break
    time.sleep(10)
REAP




artifact_cache="${INFB_ARTIFACT_CACHE:-}"
if [[ -z "${INFB_MODEL_PATH:-}" && ! -e "${workspace}/model" && -n "${artifact_cache}" ]]; then
  snap_dir=$(ls -1d "${artifact_cache}/hf/hub/models--openai--gpt-oss-20b/snapshots/"*/ 2>/dev/null | tail -1 | sed 's|/$||')
  if [[ -n "${snap_dir}" && -d "${snap_dir}" ]]; then
    ln -sfn "${snap_dir}" "${workspace}/model"
  elif [[ -d "${artifact_cache}/models/gpt-oss-20b" ]]; then
    ln -sfn "${artifact_cache}/models/gpt-oss-20b" "${workspace}/model"
  fi
fi
stage="${tests_dir}/stage_engine_extensions.sh"
engine_ext="${INFB_VLLM_ENGINE_EXT:-/opt/inference-bench/engine-ext}"
overlay_extensions() {
  local tree="$1"
  if [[ ! -f "${stage}" ]]; then
    echo "[test.sh] NOTE: no trusted overlay script at ${stage}; skipping the compiled-artefact" >&2
    echo "  overlay for ${tree} rather than running an agent-writable copy as root." >&2
    return 0
  fi
  if [[ -d "${engine_ext}/sglang" && -d "${tree}/python/sglang" ]]; then
    bash "${stage}" overlay "${tree}" "${engine_ext}"
  fi
}
overlay_extensions "${workspace}/repo"


source_info="${workspace}/real_engine_source.json"
baseline_repo="${workspace}/baseline_repo"
baseline_ready=0
if [[ -f "${source_info}" ]]; then
  orig_source_dir=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get('source_dir', ''))
" "${source_info}" 2>/dev/null || true)
  if [[ -n "${orig_source_dir}" && -d "${orig_source_dir}/python/sglang" ]]; then
    echo "[test.sh] staging baseline source ${orig_source_dir} → ${baseline_repo}"
    rm -rf "${baseline_repo}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --exclude .git --exclude __pycache__ --exclude '*.pyc' \
        "${orig_source_dir}/" "${baseline_repo}/" 2>/dev/null \
        || cp -a "${orig_source_dir}/." "${baseline_repo}/"
    else
      cp -a "${orig_source_dir}/." "${baseline_repo}/"
    fi
    overlay_extensions "${baseline_repo}"
    digest_rc=0
    python3 "$tests_dir/_pristine_digest.py" "${baseline_repo}" \
      --expect "$tests_dir/pristine_tree.sha256" \
      --out "${artifacts_dir}/raw/pristine_tree_check.json" || digest_rc=$?
    if [[ "${digest_rc}" -ne 0 ]]; then
      echo "[test.sh] FATAL: the tree staged as the PRISTINE baseline does not match the sealed" >&2
      echo "  content digest (rc=${digest_rc}; see ${artifacts_dir}/raw/pristine_tree_check.json)." >&2
      echo "  Refusing to produce a baseline number from it: it is the denominator of the scored" >&2
      echo "  ratio, so an unverified tree cannot be allowed to set the bar. This run cannot be" >&2
      echo "  scored and will fail closed as pristine_failed." >&2
    else
      baseline_ready=1
    fi
  else
    echo "[test.sh] WARNING: could not resolve original source dir for baseline bench" >&2
  fi
fi

if [[ "${baseline_ready}" -eq 1 ]]; then
  kernel_pack_rc=0
  PYTHONPATH="${workspace}/repo/python:${PYTHONPATH:-}" \
    "${INFB_VLLM_PYTHON:-python3}" "$tests_dir/assert_kernel_pack_is_pinned.py" \
    >"${artifacts_dir}/raw/kernel_pack_check.log" 2>&1 || kernel_pack_rc=$?
  cat "${artifacts_dir}/raw/kernel_pack_check.log"
  if [[ "${kernel_pack_rc}" -ne 0 ]]; then
    echo "[test.sh] FATAL: this environment cannot express the defect this task measures" >&2
    echo "  (rc=${kernel_pack_rc}; see ${artifacts_dir}/raw/kernel_pack_check.log). Both arms would" >&2
    echo "  bench the same code and the candidate would be scored as having fixed nothing. Refusing" >&2
    echo "  to produce a number: this run fails closed as pristine_failed, which is an INFRA zero." >&2
    baseline_ready=0
  fi
fi

rm -f "${artifacts_dir}/raw"/vllm_bench_latency.json \
      "${artifacts_dir}/raw"/vllm_bench_throughput.json \
      "${artifacts_dir}/raw"/sglang_bench_serving.json \
      "${artifacts_dir}/raw"/correctness_probe.json \
      "${artifacts_dir}/raw"/vllm_bench_manifest.json \
      "${artifacts_dir}/raw"/vllm_bench_stdout.txt \
      "${artifacts_dir}/raw"/vllm_bench_stderr.txt \
      "${artifacts_dir}/raw"/vllm_target_env.json \
      "${artifacts_dir}/raw"/vllm_target_env_stderr.txt
rm -rf "${artifacts_dir}/rounds" "${artifacts_dir}/warmup" "${artifacts_dir}/bench_rounds.json"
rm -f "${artifacts_dir}/baseline/raw"/sglang_bench_serving.json \
      "${artifacts_dir}/baseline/raw"/correctness_probe.json \
      "${artifacts_dir}/baseline/raw"/vllm_bench_manifest.json
if [[ "${baseline_ready}" -eq 1 ]]; then
  echo "[test.sh] warmed + rotated bench rounds (pristine vs candidate)"
  python3 "$tests_dir/_bench_rounds.py" \
    --mode serve \
    --pristine "${baseline_repo}" \
    --candidate "${workspace}/repo" \
    --artifacts "${artifacts_dir}" \
    --raw-name sglang_bench_serving.json \
    --manifest-name vllm_bench_manifest.json \
    --metric-source sglang \
    --metric-key throughput.output_tokens_per_sec \
    --rounds 2 \
    --min-warm 2 \
    --warmup-cap 4 \
    --budget-sec 900 \
    --bench-timeout-sec 1500 \
    >"${artifacts_dir}/raw/bench_rounds.log" 2>&1 \
    || echo "[test.sh] bench rounds returned non-zero; the verifier will record it"
else
  echo "[test.sh] no pristine arm; candidate-only bench, the ratio will fail closed"
  python3 "$tests_dir/_candidate_runner.py" serve \
    --repo "${workspace}/repo" --raw-dir "${artifacts_dir}/raw" \
    >"${artifacts_dir}/raw/test_sh_bench.log" 2>&1 \
    || echo "[test.sh] candidate bench returned non-zero; verifier will record this"
fi

python "$tests_dir/verify_regression_real.py" \
  --workspace "$workspace" \
  --tests-dir "$tests_dir" \
  --reward "$verifier_dir/reward.json" \
  --metrics "$artifacts_dir/metrics.json" \
  --raw-dir "$artifacts_dir/raw"
