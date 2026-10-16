#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

workspace="${INFB_WORKSPACE:-/app}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="${script_dir}/padded_draft_seq_lens_fix.patch"
target_file="${workspace}/repo/vllm/v1/spec_decode/eagle.py"

if [[ ! -f "${target_file}" ]]; then
  prepare_script="${workspace}/scripts/prepare_target_env.sh"
  if [[ ! -f "${prepare_script}" ]]; then
    prepare_script="/app/scripts/prepare_target_env.sh"
  fi
  if [[ -f "${prepare_script}" ]]; then
    echo "Running prepare_target_env.sh to sync the vLLM source tree..."
    bash "${prepare_script}"
  fi
fi

if [[ ! -f "${target_file}" ]]; then
  echo "Oracle target file not found: ${target_file}" >&2
  exit 2
fi

cd "${workspace}/repo"

if command -v patch >/dev/null 2>&1; then
  patch -p1 --batch --forward < "${patch_file}"
elif [[ -d .git ]] && command -v git >/dev/null 2>&1; then
  git apply --whitespace=nowarn "${patch_file}"
else
  echo "Neither 'patch' nor 'git' available -- cannot apply the oracle fix." >&2
  exit 2
fi

# Assert the change landed in all three files rather than trusting the exit code: `patch --forward`
# exits 0 when a hunk was ALREADY applied, so a tree left half-patched by an earlier attempt would
# otherwise look clean. The Triton kernel is one of the three and is the easiest to lose.
for f in vllm/v1/spec_decode/eagle.py vllm/v1/spec_decode/utils.py vllm/v1/worker/gpu_model_runner.py; do
  if ! grep -q "num_rejected_tokens_gpu" "${f}"; then
    echo "Oracle fix did not land in ${f}" >&2
    exit 2
  fi
done

echo "Oracle fix applied across three files; anchor at $(realpath "${target_file}")"
