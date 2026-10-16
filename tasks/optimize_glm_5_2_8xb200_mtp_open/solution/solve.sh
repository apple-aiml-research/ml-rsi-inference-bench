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

patch_file="$(dirname "${BASH_SOURCE[0]}")/${INFB_ORACLE_PATCH:-known_good.patch}"
if [[ "${INFB_ORACLE_PATCH:-}" == "__csrc_noop__" ]]; then
    tb="${repo}/csrc/libtorch_stable/torch_bindings.cpp"
    [[ -f "${tb}" ]] || tb="${repo}/csrc/torch_bindings.cpp"
    if [[ -f "${tb}" ]]; then
        printf '\n// infb csrc rebuild oracle no-op (forces _C recompile; no behavior change)\n' >> "${tb}"
        echo "[oracle] __csrc_noop__: appended byte-noop comment to csrc/torch_bindings.cpp"
    else
        echo "[oracle] __csrc_noop__: ${tb} not found — csrc tree layout differs; cannot validate" >&2
        exit 2
    fi
elif [[ -f "${patch_file}" ]] && grep -q '^diff --git' "${patch_file}"; then
    echo "[oracle] applying known_good.patch ..."
    git -C "${repo}" apply --3way "${patch_file}" || true
else
    echo "[oracle] known_good.patch missing or contains no git diff — no-op run (speedup ≈ 1.0×)"
fi

echo "[oracle] done"
