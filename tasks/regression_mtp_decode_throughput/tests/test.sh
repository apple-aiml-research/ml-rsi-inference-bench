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


jit_root="${artifacts_dir}/raw/jit"
rm -rf "${jit_root}"
mkdir -p "${jit_root}/baseline/triton" "${jit_root}/baseline/vllm" \
         "${jit_root}/candidate/triton" "${jit_root}/candidate/vllm" \
         "${jit_root}/reference_again/triton" "${jit_root}/reference_again/vllm"

{
  echo "=== nvidia-smi -L"
  nvidia-smi -L 2>&1 || echo "(nvidia-smi unavailable)"
  echo "=== driver"
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>&1 || true
  echo "=== torch / nvml as the engine sees them"
  python3 - <<'PYGPU' 2>&1 || true
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
          "device_count", torch.cuda.device_count())
except Exception as exc:
    print("torch probe failed:", type(exc).__name__, exc)
try:
    import pynvml
    pynvml.nvmlInit()
    print("nvml device_count", pynvml.nvmlDeviceGetCount(),
          "driver", pynvml.nvmlSystemGetDriverVersion())
    pynvml.nvmlShutdown()
except Exception as exc:
    print("NVML FAILED:", type(exc).__name__, exc,
          "<- this is exactly what makes vLLM say 'Failed to infer device type'")
PYGPU
} > "${artifacts_dir}/raw/gpu_visibility.txt" 2>&1
echo "[test.sh] gpu visibility recorded -> raw/gpu_visibility.txt"
if grep -qE 'NVML FAILED|device_count 0|nvidia-smi unavailable' "${artifacts_dir}/raw/gpu_visibility.txt"; then
  echo "[test.sh] WARNING: no usable GPU on this host -- the bench will die in seconds with 'Failed to infer device type', and the trial is an INFRA VOID, not a model result"
fi

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
  snap_dir=$(ls -1d "${artifact_cache}/hf/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct/snapshots/"*/ 2>/dev/null | tail -1 | sed 's|/$||') || true
  if [[ -n "${snap_dir}" && -d "${snap_dir}" ]]; then
    ln -sfn "${snap_dir}" "${workspace}/model"
  elif [[ -d "${artifact_cache}/models/Qwen3-Next-80B-A3B-Instruct" ]]; then
    ln -sfn "${artifact_cache}/models/Qwen3-Next-80B-A3B-Instruct" "${workspace}/model"
  fi
fi
venv_dir="${INFB_VENV:-}"
if [[ -n "${venv_dir}" && -d "${workspace}/repo/vllm" ]]; then
  venv_vllm=/opt/infb/engine-ext/vllm
  if [[ ! -d "${venv_vllm}" ]]; then
    venv_vllm=$(echo "${venv_dir}"/lib/python*/site-packages/vllm 2>/dev/null | head -1)
    echo "[test.sh] WARNING: no harvested extensions at /opt/infb/engine-ext/vllm;" \
         "falling back to the venv copy at ${venv_vllm}, which carries a second," \
         "readable engine tree" >&2
  fi
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
  fi
fi

if [[ -z "${INFB_SYNTH_MODEL_DIR:-}" && -e "${workspace}/model" ]]; then
  export INFB_SYNTH_MODEL_DIR="${workspace}/model"
fi

baseline_ready=0
baseline_repo=""
source_info="${workspace}/real_engine_source.json"
if [[ -f "${source_info}" ]]; then
  orig_source_dir=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get('source_dir', ''))
" "${source_info}" 2>/dev/null || true)
  if [[ -n "${orig_source_dir}" && -d "${orig_source_dir}/vllm" ]]; then
    baseline_repo="${workspace}/baseline_repo"
    echo "[test.sh] staging baseline source ${orig_source_dir} → ${baseline_repo}"
    rm -rf "${baseline_repo}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --exclude .git --exclude __pycache__ --exclude '*.pyc' \
        "${orig_source_dir}/" "${baseline_repo}/" 2>/dev/null \
        || cp -a "${orig_source_dir}/." "${baseline_repo}/"
    else
      cp -a "${orig_source_dir}/." "${baseline_repo}/"
    fi
    if [[ -n "${venv_dir}" && -d "${baseline_repo}/vllm" ]]; then
      venv_vllm=/opt/infb/engine-ext/vllm
      if [[ ! -d "${venv_vllm}" ]]; then
        venv_vllm=$(echo "${venv_dir}"/lib/python*/site-packages/vllm 2>/dev/null | head -1)
        echo "[test.sh] WARNING: no harvested extensions at /opt/infb/engine-ext/vllm;" \
             "falling back to the venv copy at ${venv_vllm}, which carries a second," \
         "readable engine tree" >&2
      fi
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
          for d in "${venv_vllm}"/vllm_flash_attn/*/; do
            [[ -d "${d}" ]] || continue
            fa_dir=$(basename "${d}")
            [[ "${fa_dir}" == "__pycache__" ]] && continue
            [[ -e "${baseline_repo}/vllm/vllm_flash_attn/${fa_dir}" ]] && continue
            ln -sfn "${d%/}" "${baseline_repo}/vllm/vllm_flash_attn/${fa_dir}"
          done
          missing_fa=""
          for src in "${venv_vllm}"/vllm_flash_attn/*; do
            [[ -e "${src}" ]] || continue
            fa_base=$(basename "${src}")
            [[ "${fa_base}" == "__pycache__" ]] && continue
            [[ -e "${baseline_repo}/vllm/vllm_flash_attn/${fa_base}" ]] || missing_fa="${missing_fa} ${fa_base}"
          done
          if [[ -n "${missing_fa}" ]]; then
            echo "[test.sh] FATAL: baseline_repo vllm_flash_attn is missing after the overlay:${missing_fa}" >&2
            exit 2
          fi
        fi
      fi

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
  else
    echo "[test.sh] WARNING: could not resolve original source dir for baseline bench" >&2
  fi
else
  echo "[test.sh] WARNING: no ${source_info}; there is no pristine arm to bench" >&2
fi

rm -f "${artifacts_dir}/raw"/vllm_bench_latency.json \
      "${artifacts_dir}/raw"/vllm_bench_throughput.json \
      "${artifacts_dir}/raw"/vllm_bench_serve.json \
      "${artifacts_dir}/raw"/vllm_bench_manifest.json \
      "${artifacts_dir}/raw"/vllm_bench_stdout.txt \
      "${artifacts_dir}/raw"/vllm_bench_stderr.txt \
      "${artifacts_dir}/raw"/vllm_target_env.json \
      "${artifacts_dir}/raw"/vllm_target_env_stderr.txt \
      "${artifacts_dir}/raw"/gdn_jit_probe.json \
      "${artifacts_dir}/raw"/equivalence.json \
      "${artifacts_dir}/raw"/equivalence_compare.json
rm -f "${artifacts_dir}/baseline/raw"/vllm_bench_latency.json \
      "${artifacts_dir}/baseline/raw"/vllm_bench_throughput.json \
      "${artifacts_dir}/baseline/raw"/vllm_bench_serve.json \
      "${artifacts_dir}/baseline/raw"/vllm_bench_manifest.json \
      "${artifacts_dir}/baseline/raw"/vllm_bench_stdout.txt \
      "${artifacts_dir}/baseline/raw"/vllm_bench_stderr.txt \
      "${artifacts_dir}/baseline/raw"/equivalence.json
rm -rf "${artifacts_dir}/rounds" "${artifacts_dir}/warmup" "${artifacts_dir}/bench_rounds.json" \
       "${artifacts_dir}/equivalence"
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
    --metric-key throughput.total_tokens_per_sec \
    --rounds 2 \
    --min-warm 1 \
    --warmup-cap 2 \
    --budget-sec 1200 \
    --bench-timeout-sec 3600 \
    >"${artifacts_dir}/raw/bench_rounds.log" 2>&1 \
    || echo "[test.sh] bench rounds returned non-zero; the verifier will record it"
  rm -rf "${artifacts_dir}"/warmup/*/raw/bench_jit "${artifacts_dir}"/warmup/*/raw/wl/*/bench_jit
else
  echo "[test.sh] no staged pristine arm; candidate-only bench, the verifier will fail closed"
  python3 "$tests_dir/_candidate_runner.py" serve \
    --repo "${workspace}/repo" --raw-dir "${artifacts_dir}/raw" \
    >"${artifacts_dir}/raw/test_sh_bench.log" 2>&1 \
    || echo "[test.sh] candidate bench returned non-zero; verifier will record this"
fi

gdn_probe_rc=0
python3 "$tests_dir/assert_gdn_kernel_recompiles.py" \
  --baseline-cache "${artifacts_dir}/rounds/0/pristine/raw/bench_jit" \
  --candidate-cache "${artifacts_dir}/rounds/0/candidate/raw/bench_jit" \
  --out "${artifacts_dir}/raw/gdn_jit_probe.json" \
  >"${artifacts_dir}/raw/gdn_jit_probe.log" 2>&1 || gdn_probe_rc=$?
cat "${artifacts_dir}/raw/gdn_jit_probe.log" || true
echo "[test.sh] gdn jit probe rc=${gdn_probe_rc} (0=exercised, candidate reduced it; 5=exercised, candidate did not; 4=baseline never recompiled; 2=probe unusable)"

if [[ -n "${baseline_repo:-}" && -d "${baseline_repo:-/nonexistent}" ]]; then
  TRITON_CACHE_DIR="${jit_root}/baseline/triton" VLLM_CACHE_ROOT="${jit_root}/baseline/vllm" \
  python3 "$tests_dir/_equivalence_probe.py" \
    --repo "${baseline_repo}" --model "${workspace}/model" \
    --out "${artifacts_dir}/baseline/raw/equivalence.json" \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    >"${artifacts_dir}/raw/equivalence_baseline.log" 2>&1 || true
fi
TRITON_CACHE_DIR="${jit_root}/candidate/triton" VLLM_CACHE_ROOT="${jit_root}/candidate/vllm" \
python3 "$tests_dir/_equivalence_probe.py" \
  --repo "${workspace}/repo" --model "${workspace}/model" \
  --out "${artifacts_dir}/raw/equivalence.json" \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  >"${artifacts_dir}/raw/equivalence_candidate.log" 2>&1 || true
if [[ -n "${baseline_repo:-}" && -d "${baseline_repo:-/nonexistent}" ]]; then
  mkdir -p "${artifacts_dir}/equivalence/reference_again"
  TRITON_CACHE_DIR="${jit_root}/reference_again/triton" \
  VLLM_CACHE_ROOT="${jit_root}/reference_again/vllm" \
  python3 "$tests_dir/_equivalence_probe.py" \
    --repo "${baseline_repo}" --model "${workspace}/model" \
    --out "${artifacts_dir}/equivalence/reference_again/equivalence.json" \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    >"${artifacts_dir}/equivalence/reference_again/equivalence.log" 2>&1 || true
fi
python3 "$tests_dir/compare_arm_outputs.py" \
  --baseline "${artifacts_dir}/baseline/raw/equivalence.json" \
  --candidate "${artifacts_dir}/raw/equivalence.json" \
  --out "${artifacts_dir}/raw/equivalence_compare.json" \
  >>"${artifacts_dir}/raw/equivalence_candidate.log" 2>&1 || true
echo "[test.sh] output equivalence: $(python3 -c "
import json,sys
try: print(json.load(open('${artifacts_dir}/raw/equivalence_compare.json'))['verdict'])
except Exception as e: print('not recorded:', e)
" 2>/dev/null)"

if [[ -n "${baseline_repo:-}" && -d "${baseline_repo}" && -f "${tests_dir}/_pristine_digest.py" ]]; then
  python3 "${tests_dir}/_pristine_digest.py" "${baseline_repo}" \
      --out "${artifacts_dir}/raw/pristine_tree_check.json" \
    || echo "[test.sh] pristine digest failed; continuing (it is recorded, not enforced)"
else
  echo "[test.sh] no staged baseline_repo to digest (baseline_repo=${baseline_repo:-<unset>})"
fi

if [[ -d "${jit_root}" || -d "${artifacts_dir}/rounds" ]]; then
  printf '{"jit_root_files": %s, "round_cache_files": %s, "candidate_only_cache_files": %s, "kib": %s, "note": "%s"}\n' \
    "$(find "${jit_root}" -type f 2>/dev/null | wc -l | tr -d ' ')" \
    "$(find "${artifacts_dir}/rounds" -path '*/bench_jit/*' -type f 2>/dev/null | wc -l | tr -d ' ')" \
    "$(find "${artifacts_dir}/raw" -path '*/bench_jit/*' -type f 2>/dev/null | wc -l | tr -d ' ')" \
    "$(du -sk "${jit_root}" "${artifacts_dir}"/rounds/*/*/raw/bench_jit "${artifacts_dir}"/rounds/*/*/raw/wl/*/bench_jit "${artifacts_dir}/raw/bench_jit" "${artifacts_dir}"/raw/wl/*/bench_jit 2>/dev/null | awk '{s+=$1} END {print s+0}')" \
    "pruned after the probes read them; per-kernel counts are in raw/gdn_jit_probe.json" \
    > "${artifacts_dir}/raw/jit_summary.json"
  rm -rf "${jit_root}"
  rm -rf "${artifacts_dir}"/rounds/*/*/raw/bench_jit "${artifacts_dir}"/rounds/*/*/raw/wl/*/bench_jit
  rm -rf "${artifacts_dir}/raw/bench_jit" "${artifacts_dir}"/raw/wl/*/bench_jit
  echo "[test.sh] pruned the compile caches out of artifacts/ (summary in raw/jit_summary.json)"
fi

python3 "$tests_dir/verify_regression_real.py" \
  --workspace "$workspace" \
  --tests-dir "$tests_dir" \
  --reward "$verifier_dir/reward.json" \
  --metrics "$artifacts_dir/metrics.json" \
  --raw-dir "$artifacts_dir/raw"
