#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

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
        echo "[oracle] cannot find prepare_target_env.sh (looked under workspace/scripts and INFB_TASK_DIR/environment/scripts)" >&2
        exit 2
    fi
fi

if [[ ! -d "$repo/.git" ]]; then
    echo "[oracle] $repo/.git missing; cannot apply patch as a commit." >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

series_dir="${INFB_ORACLE_PATCH_SERIES_DIR:-}"
if [[ -n "$series_dir" ]]; then
    [[ "$series_dir" = /* ]] || series_dir="${script_dir}/${series_dir}"
    if [[ ! -d "$series_dir" ]] || ! ls "$series_dir"/*.patch >/dev/null 2>&1; then
        echo "[oracle] patch series dir missing or empty: $series_dir" >&2
        exit 2
    fi
    cd "$repo"
    total=$(ls "$series_dir"/*.patch | wc -l)
    echo "[oracle] reconstructing stack from $total patches in $series_dir (git apply per-patch)" >&2
    applied=0; skipped=0
    tmp_diff="${workspace}/_series_cur.diff"
    for p in $(ls "$series_dir"/*.patch | sort); do
        if ! grep -q '^diff --git' "$p"; then
            skipped=$((skipped+1)); continue   # empty metadata/result commit
        fi
        # raw diff = from first `diff --git` to EOF, minus the trailing git signature
        awk '/^diff --git/{f=1} f' "$p" | sed '/^-- $/,$d' > "$tmp_diff"
        if ! git apply --whitespace=nowarn "$tmp_diff" 2>"${workspace}/_series_apply.err"; then
            echo "[oracle] git apply FAILED on $(basename "$p"):" >&2
            tail -20 "${workspace}/_series_apply.err" >&2 || true
            exit 2
        fi
        applied=$((applied+1))
    done
    echo "[oracle] applied $applied diff-bearing patches ($skipped empty/metadata skipped)" >&2
    git -c user.email='infb@example.com' -c user.name='infb' add -A
    git -c user.email='infb@example.com' -c user.name='infb' \
        commit -q -m "oracle: reconstructed intermediate stack ($applied patches from $(basename "$series_dir"))"
    echo "[oracle] reconstruction committed. Net code diff vs base (sglang/ only):" >&2
    git diff --stat inference_base..HEAD -- sglang/ 2>/dev/null | tail -20 >&2
    exit 0
fi

oracle_patch="${INFB_ORACLE_PATCH:-__sglang_noop__}"
patch_file="${script_dir}/${oracle_patch}"

if [[ "$oracle_patch" == "__sglang_noop__" ]]; then
    cd "$repo"
    echo "[oracle] __sglang_noop__: repo top-level: $(ls -1 2>/dev/null | head -20 | tr '\n' ' ')" >&2
    noop_target="$(find . -maxdepth 4 -path '*/sglang/__init__.py' 2>/dev/null | head -1)"
    if [[ -z "$noop_target" ]]; then
        echo "[oracle] FATAL: no */sglang/__init__.py under $repo — the prepare flatten/overlay is broken (a real agent run would fail too)." >&2
        find . -maxdepth 3 -name '__init__.py' 2>/dev/null | head -20 >&2
        exit 2
    fi
    printf '\n# inference-bench sglang oracle no-op (baseline reproduce)\n' >> "$noop_target"
    git -c user.email='infb@example.com' -c user.name='infb' add -A
    git -c user.email='infb@example.com' -c user.name='infb' \
        commit -q -m "oracle: sglang no-op (baseline reproduce)"
    echo "[oracle] sglang no-op applied (1-line comment in ${noop_target}) — baseline reproduce" >&2
    git log --oneline inference_base..HEAD
    exit 0
fi

if [[ "$oracle_patch" == "__sglang_sglkernel_csrc_noop__" ]]; then
    cd "$repo"
    if [[ ! -d sgl-kernel ]]; then
        echo "[oracle] FATAL: repo/sgl-kernel not staged. Set INFB_EDIT_POLICY_CSRC_SGLKERNEL=1 so" >&2
        echo "  prepare_target_env.sh stages the monorepo sgl-kernel/ source EDITABLE." >&2
        exit 2
    fi
    sglk_target=""
    for pref in \
        sgl-kernel/csrc/common_extension.cc \
        sgl-kernel/csrc/common_extension.cu \
        sgl-kernel/csrc/elementwise/activation.cu \
        sgl-kernel/csrc/elementwise/rope.cu; do
        if [[ -f "$pref" ]]; then sglk_target="$pref"; break; fi
    done
    if [[ -z "$sglk_target" ]]; then
        sglk_target="$(find sgl-kernel/csrc -type f \( -name '*.cu' -o -name '*.cc' \) \
                        -not -path '*/3rdparty/*' 2>/dev/null | sort | head -1)"
    fi
    if [[ -z "$sglk_target" ]]; then
        echo "[oracle] FATAL: no .cu/.cc under repo/sgl-kernel/csrc to edit." >&2
        find sgl-kernel -maxdepth 3 -name '*.cu' 2>/dev/null | head -20 >&2
        exit 2
    fi
    printf '\n// inference-bench sglang sgl-kernel csrc oracle: byte-noop marker (forces cmake/nvcc rebuild; behaviour unchanged) _infb_oracle\n' >> "$sglk_target"
    git -c user.email='infb@example.com' -c user.name='infb' add -A
    git -c user.email='infb@example.com' -c user.name='infb' \
        commit -q -m "oracle: sglang sgl-kernel csrc byte-noop (${sglk_target}; rebuild validation)"
    echo "[oracle] sgl-kernel csrc no-op applied (trailing comment in ${sglk_target}) — rebuild validation" >&2
    git log --oneline inference_base..HEAD
    exit 0
fi

if [[ ! -f "$patch_file" ]]; then
    echo "[oracle] $patch_file missing." >&2
    exit 2
fi

# Apply the patch as a real commit so the verifier's git diff against
# inference_base captures it.
cd "$repo"
git apply --3way "$patch_file" 2>&1 || {
    rc=$?
    echo "[oracle] git apply failed (rc=$rc). Trying with -p0 / unidiff fallback..." >&2
    git apply --3way -p0 "$patch_file" 2>&1 || true
    if ! git diff --quiet inference_base 2>/dev/null; then
        echo "[oracle] (some diff produced; continuing with whatever applied)" >&2
    else
        echo "[oracle] no changes applied; oracle may have drifted vs current SGLang." >&2
        exit 3
    fi
}

git -c user.email='infb@example.com' -c user.name='infb' \
    add -A
git -c user.email='infb@example.com' -c user.name='infb' \
    commit -q -m "oracle (broad): no-op placeholder (pipeline integration test)"

echo "[oracle] Applied known-good broad patch:"
git log --oneline inference_base..HEAD
