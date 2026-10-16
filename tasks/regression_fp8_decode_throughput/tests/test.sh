#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

overlay_third_party() {  # <tree root, e.g. /app/repo>
  local tree="${1:?tree required}" src_root="${2:?harvest required}"
  local pkg base f
  for pkg in flashmla triton_kernels; do
    [[ -d "${src_root}/third_party/${pkg}" ]] || continue
    mkdir -p "${tree}/vllm/third_party/${pkg}"
    for f in "${src_root}/third_party/${pkg}"/*.py; do
      [[ -e "${f}" ]] || continue
      base=$(basename "${f}")
      [[ -f "${tree}/vllm/third_party/${pkg}/${base}" ]] && continue
      ln -sfn "${f}" "${tree}/vllm/third_party/${pkg}/${base}"
    done
  done
  if [[ -d "${src_root}/third_party" && ! -e "${tree}/vllm/third_party/flashmla/flash_mla_interface.py" ]]; then
    echo "[test.sh] FATAL: ${tree} has no third_party/flashmla/flash_mla_interface.py after the" >&2
    echo "  overlay. The engine will die at import in platforms/cuda.py, 40 minutes from now, in a" >&2
    echo "  traceback that names neither this script nor the harvest." >&2
    return 1
  fi
}

resolve_engine_ext() {
  local venv="${1:-}"
  local d=/opt/infb/engine-ext/vllm
  if [[ ! -d "${d}" && -n "${venv}" ]]; then
    d=$(echo "${venv}"/lib/python*/site-packages/vllm 2>/dev/null | head -1)
  fi
  if [[ ! -d "${d}" ]]; then
    echo "[test.sh] FATAL: no compiled engine extensions at /opt/infb/engine-ext/vllm, and none" >&2
    echo "  under ${venv:-<no venv resolved>}. Both the candidate and the baseline tree would then" >&2
    echo "  import a Python-only engine, and the run would score real_engine_ready 0 with no" >&2
    echo "  other error printed anywhere -- which has happened." >&2
    return 1
  fi
  printf '%s' "${d}"
}

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
  snap_dir=$(ls -1d "${artifact_cache}/hf/hub/models--nvidia--DeepSeek-R1-0528-NVFP4/snapshots/"*/ 2>/dev/null | tail -1 | sed 's|/$||') || true
  if [[ -n "${snap_dir}" && -d "${snap_dir}" ]]; then
    ln -sfn "${snap_dir}" "${workspace}/model"
  elif [[ -d "${artifact_cache}/models/DeepSeek-R1-0528-NVFP4" ]]; then
    ln -sfn "${artifact_cache}/models/DeepSeek-R1-0528-NVFP4" "${workspace}/model"
  fi
fi
venv_dir="${INFB_VENV:-}"
if [[ -d "${workspace}/repo/vllm" ]]; then
  venv_vllm=$(resolve_engine_ext "${venv_dir}")
  if [[ -d "${venv_vllm}" ]]; then
    for f in "${venv_vllm}"/*.so "${venv_vllm}"/_version.py; do
      [[ -e "${f}" ]] || continue
      ln -sfn "${f}" "${workspace}/repo/vllm/$(basename "${f}")"
    done
    if [[ -d "${venv_vllm}/vllm_flash_attn" && -d "${workspace}/repo/vllm/vllm_flash_attn" ]]; then
      for f in "${venv_vllm}"/vllm_flash_attn/*.so; do
        [[ -e "${f}" ]] || continue
        ln -sfn "${f}" "${workspace}/repo/vllm/vllm_flash_attn/$(basename "${f}")"
      done
    fi
    overlay_third_party "${workspace}/repo" "${venv_vllm}" || exit 2
  fi
fi

if [[ -z "${INFB_SYNTH_MODEL_DIR:-}" && -e "${workspace}/model" ]]; then
  export INFB_SYNTH_MODEL_DIR="${workspace}/model"
fi

baseline_ready=0
baseline_repo="${workspace}/baseline_repo"
source_info="${workspace}/real_engine_source.json"
if [[ -f "${source_info}" ]]; then
  orig_source_dir=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get('source_dir', ''))
" "${source_info}" 2>/dev/null || true)
  if [[ -n "${orig_source_dir}" && -d "${orig_source_dir}/vllm" ]]; then
    echo "[test.sh] staging baseline source ${orig_source_dir} → ${baseline_repo}"
    rm -rf "${baseline_repo}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --exclude .git --exclude __pycache__ --exclude '*.pyc' \
        "${orig_source_dir}/" "${baseline_repo}/" 2>/dev/null \
        || cp -a "${orig_source_dir}/." "${baseline_repo}/"
    else
      cp -a "${orig_source_dir}/." "${baseline_repo}/"
    fi
    if [[ -d "${baseline_repo}/vllm" ]]; then
      venv_vllm=$(resolve_engine_ext "${venv_dir}")
      if [[ -d "${venv_vllm}" ]]; then
        for f in "${venv_vllm}"/*.so "${venv_vllm}"/_version.py; do
          [[ -e "${f}" ]] || continue
          ln -sfn "${f}" "${baseline_repo}/vllm/$(basename "${f}")"
        done
        if [[ -d "${venv_vllm}/vllm_flash_attn" && -d "${baseline_repo}/vllm/vllm_flash_attn" ]]; then
          for f in "${venv_vllm}"/vllm_flash_attn/*.so "${venv_vllm}"/vllm_flash_attn/*.py; do
            [[ -e "${f}" ]] || continue
            fa_base=$(basename "${f}")
            if [[ "${fa_base}" == *.py && -f "${baseline_repo}/vllm/vllm_flash_attn/${fa_base}" ]]; then
              continue
            fi
            ln -sfn "${f}" "${baseline_repo}/vllm/vllm_flash_attn/${fa_base}"
          done
          if [[ -d "${venv_vllm}/vllm_flash_attn/cute" && ! -e "${baseline_repo}/vllm/vllm_flash_attn/cute" ]]; then
            ln -sfn "${venv_vllm}/vllm_flash_attn/cute" "${baseline_repo}/vllm/vllm_flash_attn/cute"
          fi
          if [[ ! -e "${baseline_repo}/vllm/vllm_flash_attn/__init__.py" ]]; then
            echo "[test.sh] FATAL: baseline_repo vllm_flash_attn has no __init__.py after overlay" >&2
            exit 2
          fi
        fi
        overlay_third_party "${baseline_repo}" "${venv_vllm}" || exit 2
      fi
      baseline_ready=1
      if [[ -f "${tests_dir}/pristine_tree.sha256" ]]; then
        if ! python3 "${tests_dir}/_pristine_digest.py" "${baseline_repo}" \
               --expect "${tests_dir}/pristine_tree.sha256" \
               --out "${artifacts_dir}/raw/pristine_tree_precheck.json"; then
          echo "[test.sh] FATAL: the staged pristine tree does not match the sealed digest." >&2
          echo "  Refusing to bench it: it is the denominator of the scored ratio and the no-op gate." >&2
          rm -rf "${baseline_repo}"
          baseline_ready=0
        else
          echo "[test.sh] pristine tree verified against the sealed digest"
        fi
      fi
      echo "[test.sh] pristine tree staged at ${baseline_repo}"
    fi
  else
    echo "[test.sh] WARNING: could not resolve original source dir for baseline bench" >&2
  fi
fi

rm -f "${artifacts_dir}/raw"/vllm_bench_latency.json \
      "${artifacts_dir}/raw"/vllm_bench_throughput.json \
      "${artifacts_dir}/raw"/vllm_bench_serve.json \
      "${artifacts_dir}/raw"/vllm_bench_manifest.json \
      "${artifacts_dir}/raw"/vllm_bench_stdout.txt \
      "${artifacts_dir}/raw"/vllm_bench_stderr.txt \
      "${artifacts_dir}/raw"/vllm_target_env.json \
      "${artifacts_dir}/raw"/vllm_target_env_stderr.txt \
      "${artifacts_dir}/raw"/vllm_serve.log \
      "${artifacts_dir}/raw"/correctness_recording.json \
      "${artifacts_dir}/raw"/sample_vllm_bench.json \
      "${artifacts_dir}/raw"/exercised_probe*.json \
      "${artifacts_dir}/raw"/exercised_probe*.log
rm -rf "${artifacts_dir}/rounds" "${artifacts_dir}/warmup" "${artifacts_dir}/bench_rounds.json"
rm -f "${artifacts_dir}/baseline/raw"/vllm_bench_serve.json \
      "${artifacts_dir}/baseline/raw"/vllm_bench_manifest.json \
      "${artifacts_dir}/baseline/raw"/vllm_bench_stdout.txt \
      "${artifacts_dir}/baseline/raw"/vllm_bench_stderr.txt \
      "${artifacts_dir}/baseline/raw"/vllm_serve.log \
      "${artifacts_dir}/baseline/raw"/correctness_recording.json
export INFB_BOOT_TIMEOUT_SEC="${INFB_BOOT_TIMEOUT_SEC:-1800}"
if [[ "${baseline_ready}" -eq 1 ]]; then
  echo "[test.sh] warmed + rotated bench rounds (pristine vs candidate)"
  python3 "$tests_dir/_bench_rounds.py" \
    --mode serve \
    --pristine "${baseline_repo}" \
    --candidate "${workspace}/repo" \
    --artifacts "${artifacts_dir}" \
    --raw-name vllm_bench_serve.json \
    --manifest-name vllm_bench_manifest.json \
    --metric-source vllm \
    --metric-key latency_ms.tpot.mean \
    --lower-is-better \
    --rounds 2 \
    --min-warm 1 \
    --warmup-cap 2 \
    --budget-sec 600 \
    --bench-timeout-sec 2700 \
    >"${artifacts_dir}/raw/bench_rounds.log" 2>&1 \
    || echo "[test.sh] bench rounds returned non-zero; the verifier will record it"
  cat "${artifacts_dir}/raw/bench_rounds.log" || true
else
  echo "[test.sh] no staged pristine tree; candidate-only bench, the score fails closed"
  python3 "$tests_dir/_candidate_runner.py" serve \
    --repo "${workspace}/repo" --raw-dir "${artifacts_dir}/raw" \
    >"${artifacts_dir}/raw/test_sh_bench.log" 2>&1 \
    || echo "[test.sh] candidate bench returned non-zero; the verifier will record it"
fi

probe_arm() {  # $1 = pristine | candidate; prints the probe's return code
  local arm="$1" rc=0
  local -a logs=("${artifacts_dir}"/rounds/*/"${arm}"/raw/vllm_serve.log)
  if [[ "${arm}" == "candidate" ]]; then
    logs+=("${artifacts_dir}/raw/vllm_serve.log")
  else
    logs+=("${artifacts_dir}/baseline/raw/vllm_serve.log")
  fi
  "${INFB_VLLM_PYTHON:-python3}" "$tests_dir/assert_defect_path_is_exercised.py" \
    --logs "${logs[@]}" \
    --model-config "${INFB_MODEL_DIR:-/app/model}/config.json" \
    --out "${artifacts_dir}/raw/exercised_probe_${arm}.json" \
    >"${artifacts_dir}/raw/exercised_probe_${arm}.log" 2>&1 || rc=$?
  printf '%s' "${rc}"
}
pristine_probe_rc=$(probe_arm pristine)
candidate_probe_rc=$(probe_arm candidate)
cat "${artifacts_dir}/raw/exercised_probe_pristine.log" "${artifacts_dir}/raw/exercised_probe_candidate.log" || true
echo "[test.sh] exercised probe rc: pristine=${pristine_probe_rc} candidate=${candidate_probe_rc} (0=defect path live, 4=inert configuration, 2=probe unusable)"
if ! python3 - "${artifacts_dir}/raw" "${pristine_probe_rc}" "${candidate_probe_rc}" <<'PY'
import json
import pathlib
import sys

raw = pathlib.Path(sys.argv[1])
out = {}
for arm, rc in (("pristine", sys.argv[2]), ("candidate", sys.argv[3])):
    try:
        verdict = json.loads((raw / f"exercised_probe_{arm}.json").read_text())
    except (OSError, ValueError):
        verdict = None
    out[arm] = {"rc": int(rc) if rc.lstrip("-").isdigit() else None, "verdict": verdict}
(raw / "exercised_probe.json").write_text(json.dumps(out, indent=2) + "\n")
PY
then
  echo "[test.sh] could not write raw/exercised_probe.json; the verifier reads its absence as 'not exercised'"
fi

if [[ -n "${baseline_repo:-}" && -d "${baseline_repo}" && -f "${tests_dir}/_pristine_digest.py" ]]; then
  python3 "${tests_dir}/_pristine_digest.py" "${baseline_repo}" \
      --out "${artifacts_dir}/raw/pristine_tree_check.json" \
    || echo "[test.sh] pristine digest failed; continuing (it is recorded, not enforced)"
else
  echo "[test.sh] no staged baseline_repo to digest (baseline_repo=${baseline_repo:-<unset>})"
fi

python3 "$tests_dir/verify_regression_real.py" \
  --workspace "$workspace" \
  --tests-dir "$tests_dir" \
  --reward "$verifier_dir/reward.json" \
  --metrics "$artifacts_dir/metrics.json" \
  --raw-dir "$artifacts_dir/raw"
