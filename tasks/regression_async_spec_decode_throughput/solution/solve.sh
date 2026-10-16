#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

workspace="${INFB_WORKSPACE:-/app}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="${script_dir}/uniform_decode_scheduling_fix.patch"
target_file="${workspace}/repo/vllm/v1/core/sched/scheduler.py"

if [[ ! -f "${target_file}" ]]; then
  prepare_script="${workspace}/scripts/prepare_target_env.sh"
  if [[ ! -f "${prepare_script}" ]]; then
    prepare_script="/app/scripts/prepare_target_env.sh"
  fi
  if [[ -f "${prepare_script}" ]]; then
    echo "Running prepare_target_env.sh to sync vLLM source tree..."
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
  echo "Neither 'patch' nor 'git' available — cannot apply oracle fix." >&2
  exit 2
fi

echo "Oracle fix applied at $(realpath "${target_file}")"
