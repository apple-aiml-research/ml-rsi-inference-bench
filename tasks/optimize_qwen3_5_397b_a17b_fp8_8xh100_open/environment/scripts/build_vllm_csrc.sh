#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${INFB_PIP_INDEX_URL:=https://pypi.org/simple}"

workspace="${INFB_WORKSPACE:-/app}"
repo="${workspace}/repo"
build_dir="${repo}/.csredit/build"
install_prefix="${repo}/.csredit/install"   # cmake writes to ${install_prefix}/vllm/<name>.abi3.so

if [[ "${1:-}" == "--restore" ]]; then
    if [[ ! -f "${repo}/.csredit/venv_symlinks.json" ]]; then
        echo "[build_vllm_csrc] --restore: no .csredit/venv_symlinks.json — nothing was swapped, nothing to undo."
        exit 0
    fi
    _rs_here="${_SCRIPT_DIR}"
    _rs_helper=""
    for _c in "${_rs_here}/_paired_helper.py" "${_rs_here}/../_paired_helper.py" \
              "${_rs_here}/../../_paired_helper.py"; do
        if [[ -f "${_c}" ]]; then _rs_helper="${_c}"; break; fi
    done
    if [[ -z "${_rs_helper}" ]]; then
        echo "[build_vllm_csrc] FATAL: --restore needs _paired_helper.py near ${_rs_here}, not found." >&2
        exit 2
    fi
    if [[ -n "${INFB_VLLM_PYTHON:-}" && -x "${INFB_VLLM_PYTHON}" ]]; then
        _rs_py="${INFB_VLLM_PYTHON}"
    elif [[ -x /opt/venv/bin/python3 ]]; then
        _rs_py=/opt/venv/bin/python3
    elif command -v python3 >/dev/null 2>&1; then
        _rs_py=$(command -v python3)
    else
        echo "[build_vllm_csrc] FATAL: --restore found no python3." >&2
        exit 2
    fi
    _rs_venv="$(cd "$(dirname "$(dirname "${_rs_py}")")" && pwd)"
    echo "[build_vllm_csrc] --restore: putting repo/vllm/ back on the venv .so (venv=${_rs_venv})"
    "${_rs_py}" - "${_rs_helper}" "${repo}" "${_rs_venv}" <<'PYRESTORE'
import importlib.util, pathlib, sys
helper, repo, venv = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("_paired_helper_for_restore", helper)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
try:
    mod.swap_so_to_venv(pathlib.Path(repo), venv=pathlib.Path(venv))
except TypeError:
    # helper predates the venv-derived restore; the recorded paths are all it has
    mod.swap_so_to_venv(pathlib.Path(repo))
PYRESTORE
    echo "[build_vllm_csrc] --restore: done. Rebuild any time; the swap is idempotent."
    exit 0
fi

if [[ ! -f "${repo}/CMakeLists.txt" ]]; then
    echo "[build_vllm_csrc] FATAL: CMakeLists.txt not at ${repo} (workspace=${workspace})" >&2
    exit 2
fi


declare -a TARGETS
if [[ $# -eq 0 ]]; then
    TARGETS=(_C _moe_C)
else
    TARGETS=("$@")
    for t in "${TARGETS[@]}"; do
        case "${t}" in
            _C|_moe_C|_vllm_fa2_C|_vllm_fa3_C) ;;
            *) echo "[build_vllm_csrc] FATAL: unknown target '${t}'. Allowed: _C, _moe_C, _vllm_fa2_C, _vllm_fa3_C, or --restore." >&2; exit 2;;
        esac
    done
fi

_fa_target() { case "$1" in _vllm_fa2_C|_vllm_fa3_C) return 0;; *) return 1;; esac; }
_so_relpath() { if _fa_target "$1"; then echo "vllm/vllm_flash_attn/$1.abi3.so"; else echo "vllm/$1.abi3.so"; fi; }
_need_fa_src=0
for t in "${TARGETS[@]}"; do _fa_target "${t}" && _need_fa_src=1; done
if [[ "${_need_fa_src}" == 1 ]]; then
    if [[ -z "${INFB_FA_SRC_DIR:-}" && -d "${repo}/.fa_src" ]]; then
        INFB_FA_SRC_DIR="${repo}/.fa_src"
    fi
    if [[ -z "${INFB_FA_SRC_DIR:-}" || ! -d "${INFB_FA_SRC_DIR}" ]]; then
        echo "[build_vllm_csrc] FATAL: FA target requested but INFB_FA_SRC_DIR unset/missing (stage the flash-attention submodule source first)." >&2
        exit 2
    fi
    export VLLM_FLASH_ATTN_SRC_DIR="${INFB_FA_SRC_DIR}"
    echo "[build_vllm_csrc] FA rebuild: VLLM_FLASH_ATTN_SRC_DIR=${VLLM_FLASH_ATTN_SRC_DIR}"
fi

if [[ -n "${INFB_VLLM_PYTHON:-}" && -x "${INFB_VLLM_PYTHON}" ]]; then
    PYTHON="${INFB_VLLM_PYTHON}"
elif [[ -x /opt/venv/bin/python3 ]]; then
    PYTHON=/opt/venv/bin/python3
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=$(command -v python3)
else
    echo "[build_vllm_csrc] FATAL: no python3 found." >&2
    exit 2
fi

if ! command -v cmake >/dev/null 2>&1; then
    venv_bin_cmake="${PYTHON%/*}/cmake"
    if [[ -x "${venv_bin_cmake}" ]]; then
        export PATH="${PYTHON%/*}:${PATH}"
        echo "[build_vllm_csrc] cmake at ${venv_bin_cmake}; prepended ${PYTHON%/*} to PATH (pre-install)."
    else
        echo "[build_vllm_csrc] cmake not on PATH and not at ${venv_bin_cmake}; installing cmake+ninja ..."
        _csrc_tools_prefix=""
        if "${PYTHON}" -m pip install --no-cache-dir \
                --index-url "${INFB_PIP_INDEX_URL}" \
                'cmake>=3.26' ninja 2>&1 | tail -3; then
            echo "[build_vllm_csrc] pip install cmake+ninja into venv: ok"
        else
            _csrc_tools_prefix="${INFB_CSRC_TOOLS_PREFIX:-/tmp/infb_csrc_tools}"
            echo "[build_vllm_csrc] venv install failed (read-only venv?); installing cmake+ninja to ${_csrc_tools_prefix} ..."
            if "${PYTHON}" -m pip install --no-cache-dir \
                    --index-url "${INFB_PIP_INDEX_URL}" \
                    --prefix="${_csrc_tools_prefix}" --ignore-installed \
                    'cmake>=3.26' ninja 2>&1 | tail -5; then
                export PATH="${_csrc_tools_prefix}/bin:${PYTHON%/*}:${PATH}"
                for _sp in "${_csrc_tools_prefix}"/lib/python*/site-packages; do
                    [[ -d "${_sp}" ]] && export PYTHONPATH="${_sp}:${PYTHONPATH:-}"
                done
                echo "[build_vllm_csrc] cmake+ninja installed to ${_csrc_tools_prefix}; PATH+PYTHONPATH updated."
            else
                echo "[build_vllm_csrc] FATAL: pip install cmake failed (venv read-only AND writable-prefix install failed; network blocked?)." >&2
                exit 2
            fi
        fi
        hash -r 2>/dev/null || true
        if ! command -v cmake >/dev/null 2>&1; then
            if [[ -x "${venv_bin_cmake}" ]]; then
                export PATH="${PYTHON%/*}:${PATH}"
                echo "[build_vllm_csrc] cmake at ${venv_bin_cmake}; prepended ${PYTHON%/*} to PATH (post-install)."
            fi
        fi
    fi
    if ! command -v cmake >/dev/null 2>&1; then
        echo "[build_vllm_csrc] FATAL: cmake still not findable after install + PATH fix." >&2
        echo "  PATH=${PATH}" >&2
        ls -la "${PYTHON%/*}/cmake" 2>&1 | sed 's/^/  /' >&2 || true
        exit 2
    fi
fi
for tool in cmake ninja; do
    command -v "${tool}" >/dev/null 2>&1 || {
        echo "[build_vllm_csrc] FATAL: '${tool}' not on PATH after install attempts; runtime image missing build deps." >&2
        exit 2
    }
done
NVCC="$(command -v nvcc 2>/dev/null || true)"
if [[ -z "${NVCC}" && -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    NVCC="${CUDA_HOME}/bin/nvcc"
fi
if [[ -z "${NVCC}" ]]; then
    echo "[build_vllm_csrc] FATAL: nvcc not on PATH (CUDA_HOME=${CUDA_HOME:-unset})." >&2
    exit 2
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"
export MAX_JOBS="${INFB_CSRC_BUILD_MAX_JOBS:-$(nproc)}"
declare -a CCACHE_ARGS
if command -v ccache >/dev/null 2>&1; then
    CCACHE_ARGS=(-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi

echo "════════════════════════════════════════════════════════════════"
echo "  build_vllm_csrc: targets=${TARGETS[*]}"
echo "════════════════════════════════════════════════════════════════"
echo "  repo                  = ${repo}"
echo "  build_dir             = ${build_dir}"
echo "  install_prefix        = ${install_prefix}"
echo "  PYTHON                = ${PYTHON}"
echo "  NVCC                  = ${NVCC}"
echo "  TORCH_CUDA_ARCH_LIST  = ${TORCH_CUDA_ARCH_LIST}"
echo "  MAX_JOBS              = ${MAX_JOBS}"
echo "  ccache                = ${CCACHE_ARGS[*]:-(none)}"
echo ""

mkdir -p "${build_dir}"
cd "${build_dir}"

echo "[cmake] configure ..."
_INFB_CSRC_DEPS_ROOT="${INFB_CSRC_DEPS_DIR:-/opt/csrc-deps}"

_infb_cutlass_version() {
    sed -nE 's/#define CUTLASS_(MAJOR|MINOR|PATCH) +([0-9]+).*/\2/p' "${1}/include/cutlass/version.h" \
        2>/dev/null | paste -sd. -
}

_infb_resolve_local_cutlass() {
    [[ -n "${VLLM_CUTLASS_SRC_DIR:-}" ]] && { echo "[build_vllm_csrc] CUTLASS: caller-supplied ${VLLM_CUTLASS_SRC_DIR}"; return 0; }
    local want cand ver vendored
    want=$(sed -n 's/.*set(CUTLASS_REVISION *"v\([0-9.]*\)").*/\1/p' "${repo}/CMakeLists.txt" | head -1)
    [[ -z "$want" ]] && { echo "[build_vllm_csrc] CUTLASS: no CUTLASS_REVISION in CMakeLists; leaving FetchContent to it"; return 0; }

    local -a candidates=()
    [[ -d "${repo}/.csrc_src/cutlass" ]] && candidates+=("${repo}/.csrc_src/cutlass")
    [[ -d "${_INFB_CSRC_DEPS_ROOT}/cutlass" ]] && candidates+=("${_INFB_CSRC_DEPS_ROOT}/cutlass")
    vendored=$("${PYTHON}" - <<'PYCUT' 2>/dev/null
import os.path, sys
try:
    import flashinfer
except Exception:
    sys.exit(0)
d = os.path.join(os.path.dirname(flashinfer.__file__), "data", "cutlass")
print(d if os.path.isdir(os.path.join(d, "include", "cutlass")) else "")
PYCUT
)
    [[ -n "$vendored" && -d "$vendored" ]] && candidates+=("$vendored")
    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "[build_vllm_csrc] CUTLASS: no local tree found; configure will need network for cutlass"
        return 0
    fi

    for cand in "${candidates[@]}"; do
        ver=$(_infb_cutlass_version "$cand")
        if [[ "$ver" != "$want" ]]; then
            echo "[build_vllm_csrc] CUTLASS: ${cand} is ${ver:-unknown} but this source wants ${want} — skipping it" >&2
            continue
        fi
        if [[ ! -f "${cand}/CMakeLists.txt" ]]; then
            echo "[build_vllm_csrc] CUTLASS: ${cand} matches v${ver} but has NO CMakeLists.txt, so" \
                 "FetchContent_MakeAvailable cannot define CUTLASS_INCLUDE_DIR and every cutlass-including" \
                 "translation unit would fail to compile — skipping it" >&2
            continue
        fi
        export VLLM_CUTLASS_SRC_DIR="$cand"
        [[ -d "${cand}/python" ]] && export PYTHONPATH="${cand}/python${PYTHONPATH:+:${PYTHONPATH}}"
        echo "[build_vllm_csrc] CUTLASS: local v${ver} at ${cand} (defines CUTLASS_INCLUDE_DIR; cutlass_library from ${cand}/python)"
        return 0
    done
    echo "[build_vllm_csrc] CUTLASS: no usable local tree (version or CMakeLists mismatch); configure will need network" >&2
}
_infb_resolve_local_cutlass

_infb_resolve_csrc_deps() {
    local manifest="${_INFB_CSRC_DEPS_ROOT}/knobs.tsv"
    [[ -s "$manifest" ]] || { echo "[build_vllm_csrc] csrc deps: none staged in this image"; return 0; }
    local name knob subdir declared path staged=0 skipped=0
    while IFS=$'\t' read -r name knob subdir declared; do
        [[ -n "$name" && -n "$knob" ]] || continue
        if [[ -n "${!knob:-}" ]]; then
            echo "[build_vllm_csrc] csrc deps: ${knob} already set by the caller — left alone"
            continue
        fi
        local repo_tree="${repo}/.csrc_src/${name}"
        if [[ -d "${repo_tree}" ]]; then
            path="${repo_tree}"
            [[ -n "$subdir" && "$subdir" != "-" ]] && path="${path}/${subdir}"
            if [[ -d "$path" ]]; then
                export "${knob}=${path}"
                echo "[build_vllm_csrc] csrc deps: ${knob} → ${path} (agent-editable copy)"
                staged=$((staged + 1))
                continue
            fi
            echo "[build_vllm_csrc] csrc deps: ${repo_tree} exists but ${path} does not — falling back to the image tree" >&2
        fi
        path="${_INFB_CSRC_DEPS_ROOT}/${name}"
        [[ -n "$subdir" && "$subdir" != "-" ]] && path="${path}/${subdir}"
        if [[ -d "$path" ]]; then
            export "${knob}=${path}"
            staged=$((staged + 1))
        else
            echo "[build_vllm_csrc] csrc deps: ${knob} NOT set — ${path} is not in this image"
            skipped=$((skipped + 1))
        fi
    done < "$manifest"
    echo "[build_vllm_csrc] csrc deps: ${staged} knob(s) pointed at staged trees, ${skipped} missing"
}
_INFB_FC_ARGS=()
if [ -d /opt/fetchcontent ] && [ -n "$(ls -A /opt/fetchcontent 2>/dev/null)" ]; then
    _INFB_FC_ARGS=(-DFETCHCONTENT_BASE_DIR=/opt/fetchcontent -DFETCHCONTENT_FULLY_DISCONNECTED=ON)
    echo "[build_vllm_csrc] FetchContent: reusing $(ls -d /opt/fetchcontent/*-src 2>/dev/null | wc -l) pre-populated tree(s), fully disconnected"
else
    echo "[build_vllm_csrc] FetchContent: no pre-populated cache in this image; the per-dep knobs are all there is"
fi
_infb_resolve_csrc_deps

for _v in VLLM_FLASH_ATTN_SRC_DIR TRITON_KERNELS_SRC_DIR FLASH_MLA_SRC_DIR QUTLASS_SRC_DIR \
          DEEPGEMM_SRC_DIR FMHA_SM100_SRC_DIR; do
    [[ -z "${!_v:-}" ]] && echo "[build_vllm_csrc] NOTE: ${_v} unset — if this engine version declares that dependency, configure will try to clone it from github, which fails under allow_internet=false"
done

cmake -G Ninja \
      "${_INFB_FC_ARGS[@]}" \
      -DVLLM_PYTHON_EXECUTABLE="${PYTHON}" \
      -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
      -DCMAKE_BUILD_TYPE=Release \
      "${CCACHE_ARGS[@]}" \
      "${repo}"
echo ""

_ninja_has() { grep -qE "^build $1: " build.ninja 2>/dev/null; }
if [[ -f build.ninja ]]; then
    _resolved_targets=()
    for t in "${TARGETS[@]}"; do
        if _ninja_has "${t}"; then
            _resolved_targets+=("${t}")
        elif _ninja_has "${t}_stable_libtorch"; then
            echo "[build_vllm_csrc] target '${t}' absent (ROCm-only in this vLLM); using '${t}_stable_libtorch' (stable-libtorch ABI, CUDA)."
            _resolved_targets+=("${t}_stable_libtorch")
        else
            _resolved_targets+=("${t}")  # leave as-is; the build step errors clearly if truly absent
        fi
    done
    TARGETS=("${_resolved_targets[@]}")
fi

for t in "${TARGETS[@]}"; do
    echo "[cmake] build  --target ${t} ..."
    cmake --build . --target "${t}" -j "${MAX_JOBS}"
    echo "[cmake] install --component ${t} ..."
    cmake --install . --component "${t}"

    so="${install_prefix}/$(_so_relpath "${t}")"
    if [[ ! -f "${so}" ]]; then
        echo "[build_vllm_csrc] FATAL: expected ${so} after install, not found." >&2
        exit 3
    fi
    if [[ -L "${so}" ]]; then
        echo "[build_vllm_csrc] FATAL: ${so} is a symlink — install_prefix should never be inside the venv overlay tree." >&2
        ls -l "${so}" >&2
        exit 3
    fi
    sz=$(stat -c%s "${so}")
    echo "[build_vllm_csrc] ${t}: built + installed -> ${so} (${sz} bytes)"
    echo ""
done

_swap_rebuilt_so_into_repo() {
    if [[ "${INFB_CSRC_NO_SWAP:-0}" == "1" ]]; then
        echo "[build_vllm_csrc] INFB_CSRC_NO_SWAP=1 -> installed only, repo/vllm/ left alone."
        return 0
    fi
    case "${INFB_EDIT_POLICY_CSRC:-}" in
        1|true|TRUE|True|yes|YES|Yes) ;;
        *)
            echo "[build_vllm_csrc] csrc surface is CLOSED on this task (INFB_EDIT_POLICY_CSRC unset)"
            echo "[build_vllm_csrc]   -> installed only; the running engine keeps the baseline .so,"
            echo "[build_vllm_csrc]      exactly as instruction.md states."
            return 0 ;;
    esac
    local _here helper="" c
    _here="${_SCRIPT_DIR}"
    for c in "${_here}/_paired_helper.py" "${_here}/../_paired_helper.py" \
             "${_here}/../../_paired_helper.py"; do
        if [[ -f "${c}" ]]; then helper="${c}"; break; fi
    done
    if [[ -z "${helper}" ]]; then
        echo "[build_vllm_csrc] WARN: _paired_helper.py not found near ${_here} -> NOT swapping." >&2
        echo "[build_vllm_csrc] WARN: the .so is installed but nothing loads it. The verifier's" >&2
        echo "[build_vllm_csrc] WARN: own per-side swap still applies, so SCORING is" >&2
        echo "[build_vllm_csrc] WARN: unaffected -- but YOUR OWN benchmarks from here on would compare" >&2
        echo "[build_vllm_csrc] WARN: the baseline kernel against itself. Failing loudly instead." >&2
        return 3
    fi
    "${PYTHON}" - "${helper}" "${repo}" "${install_prefix}" <<'PY'
import importlib.util, pathlib, sys
helper, repo, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
# spec_from_file_location, not sys.path.insert: this runs in the agent's own
# container and must not change what anything else there imports.
spec = importlib.util.spec_from_file_location("_paired_helper_for_swap", helper)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.swap_so_to_built(pathlib.Path(repo), pathlib.Path(prefix))
PY
}
_swap_rebuilt_so_into_repo

echo "[build_vllm_csrc] done."
