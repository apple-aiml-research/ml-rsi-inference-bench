#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Oracle solution. No-op: applies an empty patch so speedup ≈ 1.0× and correctness is disabled.

set -euo pipefail

workspace="${INFB_WORKSPACE:-/app}"
repo="${workspace}/repo"

if [[ ! -d "$repo" ]]; then
    echo "[oracle] $repo missing — running prepare_target_env.sh first" >&2
    if   [[ -f "${workspace}/scripts/prepare_target_env.sh" ]]; then
        bash "${workspace}/scripts/prepare_target_env.sh"
    elif [[ -f "${INFB_TASK_DIR:-}/environment/scripts/prepare_target_env.sh" ]]; then
        bash "${INFB_TASK_DIR}/environment/scripts/prepare_target_env.sh"
    else
        echo "[oracle] cannot find prepare_target_env.sh" >&2
        exit 2
    fi
fi

patch_file="$(dirname "${BASH_SOURCE[0]}")/known_good.patch"
if [[ -f "${patch_file}" && -s "${patch_file}" ]]; then
    echo "[oracle] applying known_good.patch ..."
    git -C "${repo}" apply --3way "${patch_file}" || true
else
    echo "[oracle] known_good.patch empty or missing — no-op run (speedup ≈ 1.0×)"
fi

echo "[oracle] done"
