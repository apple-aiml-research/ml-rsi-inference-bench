#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

set -euo pipefail

workspace="${INFB_WORKSPACE:-/app}"
repo="${workspace}/repo"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$repo" ]]; then
    echo "[oracle] $repo missing — running prepare_target_env.sh first" >&2
    if   [[ -f "${workspace}/scripts/prepare_target_env.sh" ]]; then
        bash "${workspace}/scripts/prepare_target_env.sh"
    elif [[ -f "${INFB_TASK_DIR:-}/environment/scripts/prepare_target_env.sh" ]]; then
        bash "${INFB_TASK_DIR}/environment/scripts/prepare_target_env.sh"
    else
        echo "[oracle] cannot find prepare_target_env.sh (looked under workspace/scripts and INFB_TASK_DIR/environment/scripts)" >&2
        exit 2
    fi
fi

if [[ ! -d "$repo/.git" ]]; then
    echo "[oracle] $repo/.git missing; cannot apply a patch as a commit." >&2
    exit 2
fi

oracle_patch="${INFB_ORACLE_PATCH:-__sglang_noop__}"

if [[ "$oracle_patch" == "__sglang_noop__" ]]; then
    cd "$repo"
    noop_target="$(find . -maxdepth 4 -path '*/sglang/__init__.py' 2>/dev/null | head -1)"
    if [[ -z "$noop_target" ]]; then
        echo "[oracle] FATAL: no */sglang/__init__.py under $repo — the prepare flatten/overlay is broken, and a real agent run would have nothing to edit either." >&2
        find . -maxdepth 3 -name '__init__.py' 2>/dev/null | head -20 >&2
        exit 2
    fi
    printf '\n# inference-bench llada2 block-diffusion oracle no-op (baseline reproduce)\n' >> "$noop_target"
    git -c user.email='infb@example.com' -c user.name='infb' add -A
    git -c user.email='infb@example.com' -c user.name='infb' \
        commit -q -m "oracle: llada2 sglang no-op (baseline reproduce)"
    echo "[oracle] no-op marker applied (1 comment line in ${noop_target}) — baseline reproduce." >&2
    echo "[oracle] EXPECTED: correctness_ok=true, speedup_geomean ~= 1.000. Anything else is a" >&2
    echo "[oracle] harness defect, not a failed optimisation:" >&2
    echo "[oracle]   * correctness_ok=false -> the sequence-equality gate disagrees with itself" >&2
    echo "[oracle]     across two boots; token_match_floor is wrong, not the candidate." >&2
    echo "[oracle]   * a void throughput phase -> the bench could not read the committed GSM8K" >&2
    echo "[oracle]     bank, or the server never entered the dllm path." >&2
    echo "[oracle]   * speedup_geomean far from 1.0 -> the two sides are not paired." >&2
    git log --oneline inference_base..HEAD
    exit 0
fi

if [[ "$oracle_patch" == "__sglang_sglkernel_csrc_noop__" ]]; then
    cd "$repo"
    sglk_target="$(find sgl-kernel/csrc -name '*.cu' -not -path '*/3rdparty/*' 2>/dev/null | sort | head -1)"
    if [[ -z "$sglk_target" ]]; then
        sglk_target="$(find sgl-kernel -maxdepth 4 -name '*.cu' -not -path '*/3rdparty/*' 2>/dev/null | sort | head -1)"
    fi
    if [[ -z "$sglk_target" ]]; then
        echo "[oracle] FATAL: no .cu under repo/sgl-kernel/csrc to edit -- the sgl-kernel staging block in prepare_target_env.sh did not run, so the surface this task declares is not actually open." >&2
        find sgl-kernel -maxdepth 3 -name '*.cu' 2>/dev/null | head -20 >&2
        exit 2
    fi
    printf '\n// inference-bench llada2 sgl-kernel csrc oracle: byte-noop marker (forces cmake/nvcc rebuild; behaviour unchanged)\n' >> "$sglk_target"
    git -c user.email='infb@example.com' -c user.name='infb' add -A
    git -c user.email='infb@example.com' -c user.name='infb' \
        commit -q -m "oracle: llada2 sgl-kernel csrc byte-noop (${sglk_target}; rebuild validation)"
    echo "[oracle] sgl-kernel csrc no-op applied (trailing comment in ${sglk_target}) — rebuild validation" >&2
    git log --oneline inference_base..HEAD
    exit 0
fi

patch="${script_dir}/${oracle_patch}"
[[ -f "$patch" ]] || patch="$oracle_patch"
if [[ ! -f "$patch" ]]; then
    echo "[oracle] FATAL: INFB_ORACLE_PATCH=${oracle_patch} resolves to no file (looked in ${script_dir} and as a path)." >&2
    exit 2
fi
echo "[oracle] applying $patch" >&2
git -C "$repo" apply --index "$patch"
git -C "$repo" -c user.name=oracle -c user.email=oracle@infb \
    commit -q -m "oracle: $(basename "$patch")"
echo "[oracle] applied; HEAD=$(git -C "$repo" rev-parse --short HEAD)" >&2
