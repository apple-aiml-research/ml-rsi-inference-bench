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
  snap_dir=$(ls -1d "${artifact_cache}/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/"*/ 2>/dev/null | tail -1 | sed 's|/$||') || true
  if [[ -n "${snap_dir}" && -d "${snap_dir}" ]]; then
    ln -sfn "${snap_dir}" "${workspace}/model"
  elif [[ -d "${artifact_cache}/models/Qwen3.5-4B" ]]; then
    ln -sfn "${artifact_cache}/models/Qwen3.5-4B" "${workspace}/model"
  fi
fi

baseline_repo="${workspace}/baseline_repo"
source_info="${workspace}/real_engine_source.json"
staged_from=""
extract_dir=""

read_json_key() {
  python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
v = d.get(sys.argv[2])
print(v if isinstance(v, str) else '')
" "$1" "$2" 2>/dev/null || true
}

stage_baseline_from() {
  rm -rf "${baseline_repo}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude .git --exclude __pycache__ --exclude '*.pyc' \
      "$1/" "${baseline_repo}/" 2>/dev/null \
      || cp -a "$1/." "${baseline_repo}/"
  else
    mkdir -p "${baseline_repo}"
    cp -a "$1/." "${baseline_repo}/"
  fi
}

mount_dir=""
archive=""
if [[ -f "${source_info}" ]]; then
  mount_dir="$(read_json_key "${source_info}" source_dir)"
  archive="$(read_json_key "${source_info}" source_archive)"
fi
[[ -n "${archive}" ]] || archive="${INFB_ENGINE_SRC_ARCHIVE:-/opt/inference-bench/engine-src.tar.gz}"

if [[ -n "${mount_dir}" && -d "${mount_dir}/sglang" ]]; then
  echo "[test.sh] staging the pristine baseline from the runtime mount ${mount_dir}"
  stage_baseline_from "${mount_dir}"
  staged_from="runtime_mount"
elif [[ -f "${archive}" ]]; then
  want_archive="$(awk 'NR==1{print $1}' "$tests_dir/engine_src_archive.sha256" 2>/dev/null || true)"
  got_archive="$(gzip -dc "${archive}" 2>/dev/null | sha256sum | awk '{print $1}' || true)"
  if [[ -z "${want_archive}" || "${want_archive}" == "UNPINNED" ]]; then
    echo "[test.sh] NOTE: tests/engine_src_archive.sha256 is UNPINNED, so the archive is not"
    echo "  checked before unpacking; the staged tree is still checked against the sealed content"
    echo "  digest below, which is this task's authority."
    unpack_archive=1
  elif [[ "${want_archive}" != "${got_archive}" ]]; then
    echo "[test.sh] FATAL: ${archive} does not match the sealed digest — want '${want_archive}'," >&2
    echo "  have '${got_archive}'. Refusing to unpack it as the pristine baseline." >&2
    unpack_archive=0
  else
    echo "[test.sh] pristine archive verified against the sealed digest"
    unpack_archive=1
  fi
  if [[ "${unpack_archive:-0}" == "1" ]]; then
    echo "[test.sh] staging the pristine baseline from the image archive ${archive}"
    extract_dir="$(mktemp -d)"
    if tar xzf "${archive}" -C "${extract_dir}" \
       && [[ -d "${extract_dir}/engine-src/python/sglang" ]]; then
      stage_baseline_from "${extract_dir}/engine-src/python"
      staged_from="image_archive"
    else
      echo "[test.sh] WARNING: ${archive} did not unpack into a usable tree" >&2
    fi
    rm -rf "${extract_dir}"
  fi
else
  echo "[test.sh] WARNING: no pristine source (no runtime mount, no image archive)" >&2
fi

baseline_ready=0
if [[ -n "${staged_from}" ]]; then
  want="$(awk 'NR==1{print $1}' "$tests_dir/pristine_tree.sha256" 2>/dev/null || true)"
  got="$(python3 "$tests_dir/_pristine_digest.py" "${baseline_repo}" 2>/dev/null || true)"
  if [[ -z "${want}" || "${want}" != "${got}" ]]; then
    echo "[test.sh] FATAL: the staged pristine tree (${staged_from}) does not match the sealed" >&2
    echo "  digest — want '${want}', have '${got}'. Refusing to bench it as the baseline: the" >&2
    echo "  ratio's denominator and the no-op gate both depend on this tree being the pinned one." >&2
    rm -rf "${baseline_repo}"
  else
    echo "[test.sh] pristine tree verified against the sealed digest (${staged_from})"
    baseline_ready=1
  fi
fi

export INFB_BENCH_SEED="${INFB_BENCH_SEED:-$(date +%s)}"
echo "[test.sh] bench seed ${INFB_BENCH_SEED}"

rm -f "${artifacts_dir}/raw"/sglang_bench_serving.json \
      "${artifacts_dir}/raw"/sglang_bench_throughput.json \
      "${artifacts_dir}/raw"/sglang_bench_manifest.json \
      "${artifacts_dir}/raw"/sglang_bench_stdout.txt \
      "${artifacts_dir}/raw"/sglang_bench_stderr.txt \
      "${artifacts_dir}/raw"/sglang_server.log
rm -rf "${artifacts_dir}/raw/wl" "${artifacts_dir}/rounds" "${artifacts_dir}/warmup" \
       "${artifacts_dir}/bench_rounds.json"
rm -f "${artifacts_dir}/baseline/raw"/sglang_bench_serving.json \
      "${artifacts_dir}/baseline/raw"/sglang_bench_throughput.json \
      "${artifacts_dir}/baseline/raw"/sglang_bench_manifest.json
rm -rf "${artifacts_dir}/baseline/raw/wl"
if [[ "${baseline_ready}" -eq 1 ]]; then
  echo "[test.sh] warmed + rotated bench rounds (pristine vs candidate)"
  python3 "$tests_dir/_bench_rounds.py" \
    --mode serve \
    --pristine "${baseline_repo}" \
    --candidate "${workspace}/repo" \
    --artifacts "${artifacts_dir}" \
    --raw-name sglang_bench_serving.json \
    --manifest-name sglang_bench_manifest.json \
    --metric-source sglang \
    --metric-key latency_ms.ttft.mean \
    --lower-is-better \
    --guard-band repeat_img2_n120=0.93 \
    --rounds 2 \
    --min-warm 1 \
    --warmup-cap 4 \
    --budget-sec 1200 \
    --bench-timeout-sec 1500 \
    >"${artifacts_dir}/raw/bench_rounds.log" 2>&1 \
    || echo "[test.sh] bench rounds returned non-zero; the verifier will record it"
else
  echo "[test.sh] no verified pristine arm; candidate-only bench, the gate will fail closed"
  python3 "$tests_dir/_candidate_runner.py" serve \
    --repo "${workspace}/repo" --raw-dir "${artifacts_dir}/raw" --workloads all \
    >"${artifacts_dir}/raw/test_sh_bench.log" 2>&1 \
    || echo "[test.sh] candidate bench returned non-zero; verifier will record this"
fi

python3 "$tests_dir/verify_regression_real.py" \
  --workspace "$workspace" \
  --tests-dir "$tests_dir" \
  --reward "$verifier_dir/reward.json" \
  --metrics "$artifacts_dir/metrics.json" \
  --raw-dir "$artifacts_dir/raw"
