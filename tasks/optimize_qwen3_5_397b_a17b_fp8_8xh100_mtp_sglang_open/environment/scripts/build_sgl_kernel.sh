#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

: "${INFB_PIP_INDEX_URL:=https://pypi.org/simple}"

workspace="${INFB_WORKSPACE:-/app}"
repo="${workspace}/repo"
sgl_src="${INFB_SGLKERNEL_SRC:-${repo}/sgl-kernel}"
_build_root="$(dirname "${sgl_src}")"
build_dir="${INFB_SGLKERNEL_BUILD_DIR:-${_build_root}/.sglkredit/build}"
install_prefix="${INFB_SGLKERNEL_INSTALL_PREFIX:-${_build_root}/.sglkredit/install}"

if [[ "${INFB_EDIT_POLICY_CSRC_SGLKERNEL:-}" != "1" ]]; then
    echo "[build_sgl_kernel] NOTE: the sgl-kernel surface is CLOSED on this task"
    echo "[build_sgl_kernel]   (INFB_EDIT_POLICY_CSRC_SGLKERNEL is '${INFB_EDIT_POLICY_CSRC_SGLKERNEL:-unset}')."
    echo "[build_sgl_kernel]   Building anyway, but NOTHING will load the result: the bench, the"
    echo "[build_sgl_kernel]   correctness check and the verifier all gate the overlay on that flag, so"
    echo "[build_sgl_kernel]   your edit will read ~1.00x and score as unchanged -- exactly as"
    echo "[build_sgl_kernel]   instruction.md states for a closed surface. This costs ~30-90 minutes."
fi


echo "════════════════════════════════════════════════════════════════"
echo "  build_sgl_kernel: rebuild SGLang sgl-kernel from edited source"
echo "════════════════════════════════════════════════════════════════"
echo "  repo            = ${repo}"
echo "  sgl-kernel src  = ${sgl_src}"
echo "  build_dir       = ${build_dir}"
echo "  install_prefix  = ${install_prefix}"

if [[ ! -d "${sgl_src}" ]]; then
    echo "[build_sgl_kernel] FATAL: ${sgl_src} not present. prepare_target_env.sh must" >&2
    echo "  stage the monorepo sgl-kernel/ into repo/ under INFB_EDIT_POLICY_CSRC_SGLKERNEL=1." >&2
    exit 2
fi
if [[ ! -f "${sgl_src}/CMakeLists.txt" && ! -f "${sgl_src}/pyproject.toml" ]]; then
    echo "[build_sgl_kernel] FATAL: no CMakeLists.txt/pyproject.toml under ${sgl_src}" >&2
    ls -la "${sgl_src}" 2>&1 | sed 's/^/  /' >&2 || true
    exit 2
fi

_art="${INFB_VLLM_TARGET_ENV_ARTIFACT:-${INFB_SGLANG_TARGET_ENV_ARTIFACT:-}}"
if [[ -n "${_art}" && -d "${_art}/venv" ]]; then
    for _sp in "${_art}"/venv/lib/python*/site-packages "${_art}"/venv/lib64/python*/site-packages; do
        [[ -d "${_sp}" ]] && export PYTHONPATH="${_sp}${PYTHONPATH:+:${PYTHONPATH}}"
    done
    echo "  PYTHONPATH (artifact venv site-packages prepended for torch) = ${PYTHONPATH}"
fi
if [[ -n "${INFB_SGLKERNEL_PYTHON:-}" && -x "${INFB_SGLKERNEL_PYTHON}" ]]; then
    PYTHON="${INFB_SGLKERNEL_PYTHON}"
elif [[ -n "${INFB_VLLM_PYTHON:-}" && -x "${INFB_VLLM_PYTHON}" ]]; then
    PYTHON="${INFB_VLLM_PYTHON}"
elif [[ -x /opt/venv/bin/python3 ]]; then
    PYTHON=/opt/venv/bin/python3
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=$(command -v python3)
else
    echo "[build_sgl_kernel] FATAL: no python3 found." >&2
    exit 2
fi
echo "  PYTHON          = ${PYTHON}"

torch_cuda="$("${PYTHON}" -c 'import torch,sys; sys.stdout.write(torch.version.cuda or "")' 2>/dev/null || true)"
torch_ver="$("${PYTHON}" -c 'import torch,sys; sys.stdout.write(torch.__version__)' 2>/dev/null || true)"
echo "  torch           = ${torch_ver:-MISSING} (cuda=${torch_cuda:-?})"
if [[ -z "${torch_ver}" ]]; then
    echo "[build_sgl_kernel] FATAL: ${PYTHON} cannot import torch — wrong interpreter? The" >&2
    echo "  rebuild MUST use the artifact venv python (its torch 2.9.1+cu12 is the kernel ABI)." >&2
    exit 2
fi
_serving_python=""
for _cand in "${_art:+${_art}/venv/bin/python3}" /opt/venv/bin/python3; do
    [[ -n "${_cand}" && -x "${_cand}" ]] && { _serving_python="${_cand}"; break; }
done
if [[ -z "${_serving_python}" ]]; then
    echo "[build_sgl_kernel] WARN: cannot locate the serving venv python (looked at" >&2
    echo "  \${_art}/venv/bin/python3 and /opt/venv/bin/python3), so the build-vs-serve torch" >&2
    echo "  check is SKIPPED. A kernel built against a different torch ImportErrors at serve" >&2
    echo "  time; if that happens, this is the check that was not run." >&2
elif [[ "${_serving_python}" != "${PYTHON}" ]]; then
    _serving_torch="$("${_serving_python}" -c 'import torch,sys; sys.stdout.write(f"{torch.__version__}|{torch.version.cuda or \"\"}")' 2>/dev/null || true)"
    if [[ -n "${_serving_torch}" && "${_serving_torch}" != "${torch_ver}|${torch_cuda}" ]]; then
        echo "[build_sgl_kernel] FATAL: the build interpreter and the SERVING venv disagree about torch." >&2
        echo "  build  ${PYTHON}: ${torch_ver} (cuda=${torch_cuda:-?})" >&2
        echo "  serve  ${_serving_python}: ${_serving_torch/|/ (cuda=}${_serving_torch:+)}" >&2
        echo "  A kernel built against one and loaded by the other ImportErrors at serve time." >&2
        echo "  Point INFB_SGLKERNEL_PYTHON at the serving venv's python." >&2
        exit 2
    fi
fi

if [[ -x /opt/infb/csrc-tools/bin/cmake ]]; then
    export PATH="/opt/infb/csrc-tools/bin:${PATH}"
    echo "[build_sgl_kernel] cmake: using the image's staged $(/opt/infb/csrc-tools/bin/cmake --version | head -1 | awk '{print $3}')"
fi

if ! command -v cmake >/dev/null 2>&1; then
    venv_bin_cmake="${PYTHON%/*}/cmake"
    if [[ -x "${venv_bin_cmake}" ]]; then
        export PATH="${PYTHON%/*}:${PATH}"
        echo "[build_sgl_kernel] cmake at ${venv_bin_cmake}; prepended ${PYTHON%/*} to PATH."
    else
        echo "[build_sgl_kernel] cmake not found; pip-installing into the venv ..."
        "${PYTHON}" -m pip install --no-cache-dir --index-url "${INFB_PIP_INDEX_URL}" \
            'cmake>=3.26' 2>&1 | tail -3 \
            || { echo "[build_sgl_kernel] FATAL: pip install cmake failed (network?)." >&2; exit 2; }
        hash -r 2>/dev/null || true
        [[ -x "${venv_bin_cmake}" ]] && export PATH="${PYTHON%/*}:${PATH}"
    fi
    command -v cmake >/dev/null 2>&1 \
        || { echo "[build_sgl_kernel] FATAL: cmake still not findable." >&2; exit 2; }
fi
if ! command -v ninja >/dev/null 2>&1; then
    venv_bin_ninja="${PYTHON%/*}/ninja"
    if [[ -x "${venv_bin_ninja}" ]]; then
        export PATH="${PYTHON%/*}:${PATH}"
    else
        "${PYTHON}" -m pip install --no-cache-dir --index-url "${INFB_PIP_INDEX_URL}" ninja 2>&1 | tail -2 \
            || { echo "[build_sgl_kernel] FATAL: pip install ninja failed." >&2; exit 2; }
        hash -r 2>/dev/null || true
        [[ -x "${venv_bin_ninja}" ]] && export PATH="${PYTHON%/*}:${PATH}"
    fi
fi
command -v ninja >/dev/null 2>&1 || { echo "[build_sgl_kernel] FATAL: ninja not findable." >&2; exit 2; }

NVCC="$(command -v nvcc 2>/dev/null || true)"
if [[ -z "${NVCC}" && -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"; NVCC="${CUDA_HOME}/bin/nvcc"
fi
if [[ -z "${NVCC}" ]]; then
    for d in /usr/local/cuda*/bin /opt/usr/local/cuda*/bin; do
        if [[ -x "${d}/nvcc" ]]; then export PATH="${d}:${PATH}"; NVCC="${d}/nvcc"; break; fi
    done
fi
[[ -n "${NVCC}" ]] || { echo "[build_sgl_kernel] FATAL: nvcc not on PATH (CUDA_HOME=${CUDA_HOME:-unset})." >&2; exit 2; }

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"   # H100 sm90; B200 caller sets 10.0+PTX
_sglk_default_max_jobs() {
    local cg="${INFB_CGROUP_ROOT:-/sys/fs/cgroup}" meminfo="${INFB_PROC_MEMINFO:-/proc/meminfo}"
    local num='^[1-9][0-9]{0,17}$' q="" p="" lim="" tot="" k v rest
    _sglk_per="${INFB_SGLKERNEL_BUILD_GIB_PER_JOB:-10}"
    [[ "${_sglk_per}" =~ ^[1-9][0-9]{0,3}$ ]] || _sglk_per="invalid(${_sglk_per})"
    _sglk_nproc="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 1)"
    [[ "${_sglk_nproc}" =~ ^[1-9][0-9]*$ ]] || _sglk_nproc=1
    _sglk_jobs="${_sglk_nproc}"; _sglk_quota="none"; _sglk_mem_gib="none"
    if [[ -r "${cg}/cpu.max" ]]; then
        read -r q p < "${cg}/cpu.max" || true
    elif [[ -r "${cg}/cpu/cpu.cfs_quota_us" && -r "${cg}/cpu/cpu.cfs_period_us" ]]; then
        read -r q < "${cg}/cpu/cpu.cfs_quota_us" || true
        read -r p < "${cg}/cpu/cpu.cfs_period_us" || true
    fi
    if [[ "${q}" =~ ${num} && "${p}" =~ ${num} ]]; then
        _sglk_quota=$(( (q + p - 1) / p ))
        if (( _sglk_quota < _sglk_jobs )); then _sglk_jobs="${_sglk_quota}"; fi
    fi
    if [[ -r "${cg}/memory.max" ]]; then
        read -r lim < "${cg}/memory.max" || true
    elif [[ -r "${cg}/memory/memory.limit_in_bytes" ]]; then
        read -r lim < "${cg}/memory/memory.limit_in_bytes" || true
    fi
    [[ "${lim}" =~ ${num} ]] || lim=""
    if [[ -r "${meminfo}" ]]; then
        while read -r k v rest; do
            if [[ "${k}" == "MemTotal:" && "${v}" =~ ^[1-9][0-9]{0,14}$ ]]; then tot=$(( v * 1024 )); break; fi
        done < "${meminfo}"
    fi
    if [[ -n "${tot}" ]] && { [[ -z "${lim}" ]] || (( tot < lim )); }; then lim="${tot}"; fi
    if [[ -n "${lim}" ]]; then _sglk_mem_gib=$(( lim / (1 << 30) )); fi
    if [[ -n "${lim}" && "${_sglk_per}" =~ ^[0-9]+$ ]]; then
        v=$(( lim / (_sglk_per * (1 << 30)) ))
        if (( v < _sglk_jobs )); then _sglk_jobs="${v}"; fi
    fi
    if (( _sglk_jobs < 1 )); then _sglk_jobs=1; fi
}
if [[ -n "${INFB_SGLKERNEL_BUILD_MAX_JOBS:-}" ]]; then
    export MAX_JOBS="${INFB_SGLKERNEL_BUILD_MAX_JOBS}"
    _sglk_jobs_why="INFB_SGLKERNEL_BUILD_MAX_JOBS"
else
    _sglk_default_max_jobs
    export MAX_JOBS="${_sglk_jobs}"
    _sglk_jobs_why="min(nproc=${_sglk_nproc}, cpu quota=${_sglk_quota}, memory=${_sglk_mem_gib} GiB / ${_sglk_per} GiB per job)"
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS}}"
echo "  NVCC            = ${NVCC}"
echo "  TORCH_CUDA_ARCH = ${TORCH_CUDA_ARCH_LIST}"
echo "  MAX_JOBS        = ${MAX_JOBS}   <- ${_sglk_jobs_why}"
echo "  CMAKE_BUILD_PARALLEL_LEVEL = ${CMAKE_BUILD_PARALLEL_LEVEL}   (ninja -j)"
echo ""

src_clone="${INFB_SGLANG_SOURCE_DIR:-${INFB_VLLM_TARGET_ENV_ARTIFACT:-}/source}"
if [[ -d "${sgl_src}/3rdparty" ]]; then
    missing_sub=0
    for sub in cutlass flashinfer DeepGEMM deepgemm; do
        d="${sgl_src}/3rdparty/${sub}"
        if [[ -d "${d}" && -z "$(ls -A "${d}" 2>/dev/null)" ]]; then missing_sub=1; fi
    done
    if [[ "${missing_sub}" -eq 1 ]]; then
        echo "[build_sgl_kernel] 3rdparty submodules empty — populating from source clone ${src_clone}"
        if [[ -d "${src_clone}/sgl-kernel/3rdparty" ]]; then
            cp -a "${src_clone}/sgl-kernel/3rdparty/." "${sgl_src}/3rdparty/" 2>/dev/null || true
        fi
        if [[ -d "${src_clone}/.git" ]]; then
            ( cd "${src_clone}" && git submodule update --init --recursive --depth 1 2>&1 | tail -15 ) || true
            cp -a "${src_clone}/sgl-kernel/3rdparty/." "${sgl_src}/3rdparty/" 2>/dev/null || true
        fi
    fi
fi

_cutlass_dir="${sgl_src}/3rdparty/cutlass"
if [[ ! -e "${_cutlass_dir}/include/cutlass/cutlass.h" ]]; then
    echo "[build_sgl_kernel] CUTLASS 3rdparty empty (github unreachable) — sourcing from the nvidia-cutlass pip package"
    _cut_tmp="$(mktemp -d)"
    if "${PYTHON}" -m pip download nvidia-cutlass --no-deps -d "${_cut_tmp}" 2>&1 | tail -3; then
        _whl="$(ls "${_cut_tmp}"/nvidia_cutlass-*.whl 2>/dev/null | head -1)"
        if [[ -n "${_whl}" ]]; then
            ( cd "${_cut_tmp}" && { command -v unzip >/dev/null 2>&1 && unzip -q "${_whl}" -d x || "${PYTHON}" -m zipfile -e "${_whl}" x; } ) || true  # portable: the image may lack unzip
            _src="$(find "${_cut_tmp}/x" -type d -path '*cutlass_library/source' 2>/dev/null | head -1)"
            if [[ -n "${_src}" && -f "${_src}/include/cutlass/cutlass.h" ]]; then
                mkdir -p "${_cutlass_dir}"
                cp -a "${_src}/." "${_cutlass_dir}/" 2>/dev/null || true
                echo "[build_sgl_kernel] CUTLASS populated from pip $(basename "${_whl}") → cutlass.h $([[ -f "${_cutlass_dir}/include/cutlass/cutlass.h" ]] && echo OK || echo MISSING)"
            else
                echo "[build_sgl_kernel] WARNING: pip nvidia-cutlass wheel had no cutlass_library/source — CUTLASS fallback failed"
            fi
        fi
    else
        echo "[build_sgl_kernel] WARNING: pip download nvidia-cutlass failed — CUTLASS still empty (build may be partial)"
    fi
    rm -rf "${_cut_tmp}" 2>/dev/null || true
fi

mkdir -p "${install_prefix}" "${build_dir}"

export CMAKE_ARGS="${CMAKE_ARGS:+${CMAKE_ARGS} }-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:+${SKBUILD_CMAKE_ARGS};}-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
echo "[build_sgl_kernel] CMake-4 back-compat: -DCMAKE_POLICY_VERSION_MINIMUM=3.5 (clears <3.5 submodules e.g. dlpack)"

case "${TORCH_CUDA_ARCH_LIST}" in
    9.*|1[0-9].*)
        export CMAKE_ARGS="${CMAKE_ARGS:+${CMAKE_ARGS} }-DENABLE_BELOW_SM90=OFF"
        export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:+${SKBUILD_CMAKE_ARGS};}-DENABLE_BELOW_SM90=OFF"
        echo "[build_sgl_kernel] arch: -DENABLE_BELOW_SM90=OFF (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}; sm_80/sm_89 cubins cannot load here)"
        ;;
    *)
        echo "[build_sgl_kernel] arch: keeping below-SM90 gencode (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST})"
        ;;
esac

_infb_populate_editsrc_from_image() {
    local root="$1" deps="${INFB_CSRC_DEPS_DIR:-/opt/csrc-deps}"
    [[ -d "$root" ]] && return 0
    [[ -s "${deps}/knobs.tsv" ]] || return 0
    local n k s d copied=0
    while IFS=$'\t' read -r n k s d; do
        [[ -n "$n" && -n "$d" && "$d" != "-" ]] || continue
        [[ -d "${deps}/${n}" ]] || continue
        mkdir -p "${root}/${d}"
        cp -a "${deps}/${n}/." "${root}/${d}/" 2>/dev/null || continue
        copied=$((copied + 1))
    done < "${deps}/knobs.tsv"
    if [[ "$copied" -gt 0 ]]; then
        echo "[build_sgl_kernel] _editsrc: populated ${copied} dep(s) from ${deps} (agent-editable copies)"
    fi
}
_infb_populate_editsrc_from_image "${sgl_src}/_editsrc"

_editsrc_root="${sgl_src}/_editsrc"
if [[ -d "${_editsrc_root}" ]]; then
    for _d in "${_editsrc_root}"/*/; do
        [[ -d "${_d}" ]] || continue
        _decl="$(basename "${_d%/}")"
        _abs="$(cd "${_d}" && pwd)"
        _var="FETCHCONTENT_SOURCE_DIR_${_decl^^}"   # repo-flash-attention → REPO-FLASH-ATTENTION
        export CMAKE_ARGS="${CMAKE_ARGS:+${CMAKE_ARGS} }-D${_var}=${_abs}"
        export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:+${SKBUILD_CMAKE_ARGS};}-D${_var}=${_abs}"
        echo "[build_sgl_kernel] FetchContent redirect (editable kernel source): -D${_var}=${_abs}"
    done
fi

_artifact="${INFB_VLLM_TARGET_ENV_ARTIFACT:-${INFB_SGLANG_TARGET_ENV_ARTIFACT:-}}"
_syslibs=""
for _cand in "${_artifact}/syslibs" "${INFB_SGLANG_SYSLIBS:-}"; do
    if [[ -n "${_cand}" && -d "${_cand}" ]]; then _syslibs="${_cand}"; break; fi
done
if [[ -n "${_syslibs}" ]]; then
    _numa_inc=""; _numa_lib=""
    [[ -e "${_syslibs}/include/numa.h" ]] && _numa_inc="${_syslibs}/include"
    for _l in "${_syslibs}/libnuma.so" "${_syslibs}/libnuma.so.1"; do
        [[ -e "${_l}" ]] && { _numa_lib="${_l}"; break; }
    done
    if [[ -n "${_numa_inc}" ]]; then
        export CPATH="${_numa_inc}${CPATH:+:${CPATH}}"
        export C_INCLUDE_PATH="${_numa_inc}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
        export CPLUS_INCLUDE_PATH="${_numa_inc}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
    fi
    if [[ -n "${_numa_lib}" ]]; then
        export LIBRARY_PATH="${_syslibs}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
        export LD_LIBRARY_PATH="${_syslibs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    _cmake_numa=""
    [[ -n "${_numa_inc}" ]] && _cmake_numa="${_cmake_numa} -DNUMA_INCLUDE_DIRS=${_numa_inc} -DNUMA_INCLUDE_DIR=${_numa_inc}"
    [[ -n "${_numa_lib}" ]] && _cmake_numa="${_cmake_numa} -DNUMA_LIBRARIES=${_numa_lib} -DNUMA_LIBRARY=${_numa_lib}"
    export CMAKE_ARGS="${CMAKE_ARGS:-}${_cmake_numa}"
    export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:-}${_cmake_numa// /;}"
    echo "[build_sgl_kernel] NUMA build dep: syslibs=${_syslibs} numa.h=${_numa_inc:-MISSING} libnuma=${_numa_lib:-MISSING}"
    echo "[build_sgl_kernel] CMAKE_ARGS += ${_cmake_numa} (NUMA-only; CMAKE_PREFIX_PATH left to scikit-build torch auto-inject)"
    if [[ -z "${_numa_inc}" ]]; then
        echo "[build_sgl_kernel] WARN: numa.h not in ${_syslibs}/include — mscclpp FindNUMA may fail. Rebuild the env artifact with libnuma-dev." >&2
    fi
else
    echo "[build_sgl_kernel] WARN: no artifact syslibs dir found (INFB_VLLM_TARGET_ENV_ARTIFACT/syslibs); NUMA build dep may be unsatisfied -> mscclpp FindNUMA failure." >&2
fi

_torch_cmake_prefix="$("${PYTHON}" -c 'import torch,sys; sys.stdout.write(torch.utils.cmake_prefix_path)' 2>/dev/null || true)"
if [[ -n "${_torch_cmake_prefix}" ]]; then
    _torch_dir="${_torch_cmake_prefix}/Torch"
    echo "[build_sgl_kernel] Torch cmake prefix = ${_torch_cmake_prefix} (Torch_DIR=${_torch_dir}, exists=$([[ -e "${_torch_dir}/TorchConfig.cmake" ]] && echo yes || echo NO))"
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DTorch_DIR=${_torch_dir}"
    export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:-};-DTorch_DIR=${_torch_dir}"
else
    echo "[build_sgl_kernel] WARN: could not resolve torch.utils.cmake_prefix_path from ${PYTHON} — find_package(Torch) relies on scikit-build auto-inject." >&2
fi

pyproject="${sgl_src}/pyproject.toml"
build_reqs=""
if [[ -f "${pyproject}" ]]; then
    build_reqs="$("${PYTHON}" - "${pyproject}" <<'PY'
import sys
try:
    import tomllib  # py3.11+
    with open(sys.argv[1], "rb") as f:
        data = tomllib.load(f)
except Exception:
    # Fallback: crude [build-system].requires extractor (no tomllib).
    import re
    txt = open(sys.argv[1], encoding="utf-8").read()
    m = re.search(r"\[build-system\].*?requires\s*=\s*\[(.*?)\]", txt, re.DOTALL)
    reqs = re.findall(r'["\']([^"\']+)["\']', m.group(1)) if m else []
    print(" ".join(reqs)); sys.exit(0)
reqs = (data.get("build-system", {}) or {}).get("requires", []) or []
print(" ".join(str(r) for r in reqs))
PY
)"
fi
if [[ -z "${build_reqs}" ]]; then
    echo "[build_sgl_kernel] WARN: could not read [build-system].requires from ${pyproject}; using default backend set." >&2
    build_reqs="scikit-build-core cmake ninja"
fi
echo "[build_sgl_kernel] [build-system].requires = ${build_reqs}"
_sglk_backend_ready() { "${PYTHON}" -c "import scikit_build_core.build" >/dev/null 2>&1; }
if ! _sglk_backend_ready; then
    for _prefix in "${INFB_SGL_BUILD_BACKEND_PREFIX:-}" /opt/infb/buildbackend; do
        [[ -n "${_prefix}" ]] || continue
        _site="$(ls -d "${_prefix}"/lib/python*/site-packages 2>/dev/null | head -1 || true)"
        [[ -n "${_site}" ]] || continue
        export PYTHONPATH="${_site}${PYTHONPATH:+:${PYTHONPATH}}"
        if _sglk_backend_ready; then
            echo "[build_sgl_kernel] build backend from the image at ${_site} (no network needed)"
            break
        fi
    done
fi
if _sglk_backend_ready; then
    echo "[build_sgl_kernel] build backend already importable — skipping the pip install"
    build_reqs=""
fi
if [[ -n "${build_reqs}" ]]; then
echo "[build_sgl_kernel] pre-installing build backend (normal install — build tools only, NOT the kernel) ..."
# shellcheck disable=SC2086
if ! "${PYTHON}" -m pip install --no-cache-dir --index-url "${INFB_PIP_INDEX_URL}" ${build_reqs} 2>&1 | tail -20; then
    echo "[build_sgl_kernel] FATAL: failed to pip-install the build backend (${build_reqs}) from ${INFB_PIP_INDEX_URL}." >&2
    exit 2
fi
fi
hash -r 2>/dev/null || true
if ! "${PYTHON}" -c "import scikit_build_core.build" 2>/dev/null; then
    echo "[build_sgl_kernel] FATAL: 'import scikit_build_core.build' still fails after installing the backend." >&2
    echo "  build-system.requires was: ${build_reqs}" >&2
    "${PYTHON}" -c "import scikit_build_core.build" 2>&1 | sed 's/^/    /' >&2 || true
    exit 2
fi
echo "[build_sgl_kernel] build backend ready (scikit_build_core.build imports)."

_cache_root=""
for _c in "${_artifact}/sglkredit_cache" "${INFB_SGLKERNEL_CACHE_DIR:-}"; do
    if [[ -n "${_c}" && -d "${_c}" ]]; then _cache_root="${_c}"; break; fi
done
if [[ -z "${_cache_root}" ]]; then
    _cache_root="${repo}/.sglkredit/cache"
fi
_ro_ccache="${_cache_root}/ccache"
_ccache_writable=0
if [[ -d "${_ro_ccache}" ]] && ( : > "${_ro_ccache}/.infb_wtest" ) 2>/dev/null; then
    rm -f "${_ro_ccache}/.infb_wtest" 2>/dev/null || true
    _ccache_writable=1
fi
if [[ "${_ccache_writable}" == 1 ]]; then
    export CCACHE_DIR="${_ro_ccache}"
    mkdir -p "${CCACHE_DIR}" 2>/dev/null || true
else
    export CCACHE_DIR="${repo}/.sglkredit/ccache_rw"
    mkdir -p "${CCACHE_DIR}" 2>/dev/null || true
    if [[ -d "${_ro_ccache}" ]]; then
        echo "[build_sgl_kernel] warm ccache not writable (${_ro_ccache}) — seeding writable copy at ${CCACHE_DIR}"
        cp -a "${_ro_ccache}/." "${CCACHE_DIR}/" 2>/dev/null || true
    fi
fi
export CCACHE_TEMPDIR="${CCACHE_DIR}/tmp"
mkdir -p "${CCACHE_TEMPDIR}" 2>/dev/null || true
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
if [[ -x "${_cache_root}/bin/ccache" ]]; then
    export PATH="${_cache_root}/bin:${PATH}"
    echo "[build_sgl_kernel] using bundled ccache: ${_cache_root}/bin/ccache"
fi
if ccache --version >/dev/null 2>&1; then
    echo "[build_sgl_kernel] ccache OK (loads + runs): CCACHE_DIR=${CCACHE_DIR} (maxsize=${CCACHE_MAXSIZE}); sgl-kernel's CMakeLists wires it as the compiler launcher."
    ccache -z >/dev/null 2>&1 || true   # zero stats so the post-build summary is per-build
else
    echo "[build_sgl_kernel] NOTE: ccache not runnable (absent or missing shared lib) — stripping bundled ccache from PATH; build will be cold (plain nvcc). CCACHE_DIR=${CCACHE_DIR} reserved." >&2
    if [[ -n "${_cache_root:-}" && -d "${_cache_root}/bin" ]]; then
        PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${_cache_root}/bin" | paste -sd: -)"
        export PATH
    fi
fi

if [[ "${INFB_SGLKERNEL_INCREMENTAL:-1}" =~ ^(1|true|yes)$ ]]; then
    _persist_build="${INFB_SGLKERNEL_SKBUILD_DIR:-${build_dir}/skbuild}"
    _stamp="${_persist_build}/.infb_build_stamp"
    _cur_sig="src=${sgl_src}|arch=${TORCH_CUDA_ARCH_LIST}"
    if [[ -f "${_stamp}" ]] && [[ "$(cat "${_stamp}" 2>/dev/null || true)" != "${_cur_sig}" ]]; then
        echo "[build_sgl_kernel] persistent build dir config changed (was '$(cat "${_stamp}" 2>/dev/null)', now '${_cur_sig}') — wiping ${_persist_build} for a clean reconfigure."
        rm -rf "${_persist_build}" 2>/dev/null || true
    fi
    mkdir -p "${_persist_build}" 2>/dev/null || true
    printf '%s' "${_cur_sig}" > "${_stamp}" 2>/dev/null || true
    export SKBUILD_BUILD_DIR="${_persist_build}"
    echo "[build_sgl_kernel] INCREMENTAL: SKBUILD_BUILD_DIR=${_persist_build} (persistent CMake binary dir → stable _deps paths + ninja incremental)"
    if command -v ccache >/dev/null 2>&1; then
        export CCACHE_BASEDIR="${INFB_SGLKERNEL_CCACHE_BASEDIR:-${_build_root}}"
        export CCACHE_NOHASHDIR="${CCACHE_NOHASHDIR:-true}"
        export CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-time_macros,include_file_mtime,include_file_ctime,locale,pch_defines}"
        echo "[build_sgl_kernel] INCREMENTAL: CCACHE_BASEDIR=${CCACHE_BASEDIR} CCACHE_NOHASHDIR=${CCACHE_NOHASHDIR} (path-independent hashing)"
    fi
else
    echo "[build_sgl_kernel] INCREMENTAL disabled (INFB_SGLKERNEL_INCREMENTAL=0) — cold /tmp build dir (legacy path)."
fi

export CMAKE_VERBOSE_MAKEFILE="${CMAKE_VERBOSE_MAKEFILE:-ON}"
export VERBOSE="${VERBOSE:-1}"  # ninja honors VERBOSE=1

echo "[build_sgl_kernel] pip wheel build (scikit-build, no isolation, VERBOSE — streaming) ..."
echo "[build_sgl_kernel]   final CMAKE_ARGS = ${CMAKE_ARGS}"
t0=$(date +%s)
wheel_out="${build_dir}/wheelhouse"
mkdir -p "${wheel_out}"
build_log="${INFB_SGLKERNEL_PIP_LOG:-${build_dir}/pip_wheel.log}"
if "${PYTHON}" -m pip wheel -v --no-build-isolation --no-deps \
        -w "${wheel_out}" "${sgl_src}" 2>&1 | tee "${build_log}"; then
    wheel="$(ls -1 "${wheel_out}"/*.whl 2>/dev/null | grep -iE 'kernel' | head -1 || true)"
    if [[ -z "${wheel}" ]]; then
        echo "[build_sgl_kernel] FATAL: pip wheel reported success but no sgl_kernel wheel in ${wheel_out}" >&2
        ls -la "${wheel_out}" >&2 || true
        exit 3
    fi
    echo "[build_sgl_kernel] built wheel: ${wheel}"
    command -v ccache >/dev/null 2>&1 && { echo "[build_sgl_kernel] ccache stats:"; ccache -s 2>/dev/null | sed 's/^/  /' || true; }
    "${PYTHON}" - "$wheel" "$install_prefix" <<'PY'
import sys, zipfile, pathlib
wheel, prefix = sys.argv[1], pathlib.Path(sys.argv[2])
prefix.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(wheel) as z:
    for n in z.namelist():
        # extract package dirs (sgl_kernel/...), skip the .dist-info metadata
        if n.split("/", 1)[0].endswith(".dist-info"):
            continue
        z.extract(n, prefix)
print(f"[build_sgl_kernel] unpacked wheel package tree -> {prefix}")
PY
else
    echo "[build_sgl_kernel] FATAL: pip wheel build of sgl-kernel FAILED (full log: ${build_log}; last 60 lines above)." >&2
    echo "  Likely causes: missing 3rdparty submodules, compile-time > wall, torch-ABI, or sm flags." >&2
    command -v ccache >/dev/null 2>&1 && { echo "[build_sgl_kernel] ccache stats at failure:"; ccache -s 2>/dev/null | sed 's/^/  /' || true; }
    exit 3
fi
elapsed=$(( $(date +%s) - t0 ))
echo "[build_sgl_kernel] build elapsed=${elapsed}s"

if [[ ! -d "${install_prefix}/sgl_kernel" ]]; then
    echo "[build_sgl_kernel] FATAL: ${install_prefix}/sgl_kernel/ missing after install." >&2
    ls -la "${install_prefix}" >&2 || true
    exit 3
fi
so_count="$(find "${install_prefix}/sgl_kernel" -name '*.so' 2>/dev/null | wc -l)"
if [[ "${so_count}" -eq 0 ]]; then
    echo "[build_sgl_kernel] FATAL: no .so under ${install_prefix}/sgl_kernel/ (build produced no kernel)." >&2
    find "${install_prefix}/sgl_kernel" -maxdepth 3 2>/dev/null | head -40 >&2 || true
    exit 3
fi
echo "[build_sgl_kernel] OK: ${install_prefix}/sgl_kernel/ has ${so_count} .so file(s)."
echo "[build_sgl_kernel] ${install_prefix} is now prepended to the CANDIDATE PYTHONPATH (before venv"
echo "[build_sgl_kernel]   site-packages) by ALL THREE consumers, so \`import sgl_kernel\` resolves here:"
echo "[build_sgl_kernel]     - your own paired bench      (scripts/run_benchmark.py)"
echo "[build_sgl_kernel]     - your own correctness check (scripts/run_correctness.py)"
echo "[build_sgl_kernel]     - the scored verifier"
echo "[build_sgl_kernel]   The pristine/baseline side keeps the venv kernel in every one of them."
echo "[build_sgl_kernel]   Re-run this after any further sgl-kernel edit: the consumers warn when this"
echo "[build_sgl_kernel]   dir is older than your source, but they will not rebuild it for you."
echo "[build_sgl_kernel] done."
