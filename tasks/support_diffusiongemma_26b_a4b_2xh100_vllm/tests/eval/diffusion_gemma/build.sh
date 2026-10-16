#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Rebuild the engine in place after changing CUDA, C++ or Rust sources, including the compiled
# kernels of the FlashAttention fork under /app/third_party/vllm-flash-attn:
#   bash /app/eval/diffusion_gemma/build.sh
#
# This is the SAME sequence the image was built with and the same one the verifier runs on the
# submitted tree, offline and without build isolation. Pure-Python edits need no rebuild, the
# fork's CuTe-DSL sources included: the install is editable, and vllm/vllm_flash_attn/cute is a link
# to /app/third_party/vllm-flash-attn/flash_attn/cute. CMake makes that link in setup.py's build
# directory, and setup.py then copies the sources through it into the tree, where the copies would
# not import; so the copy is removed before the install and replaced by the link after it.
set -euo pipefail
cd "${INFB_WORKSPACE:-/app}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS="${MAX_JOBS:-48}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-48}"
export CARGO_NET_OFFLINE=true
# The CMake build directory persists in the image at /app/build/temp, so a rebuild recompiles only
# what changed: a Python-only change takes seconds, a C++/CUDA change its own objects. A change to
# the build configuration (setup.py, pyproject.toml, requirements/) re-runs the editable install
# first, which is a full compile that the install clock does not cover.
BASE="$(cat /opt/infb/prepared-base/PREPARED_BASE_SHA 2>/dev/null || true)"
if [ ! -f build/temp/CMakeCache.txt ] || [ -z "$BASE" ] \
   || ! git diff --quiet "$BASE" -- setup.py pyproject.toml requirements/; then
    rm -rf vllm/vllm_flash_attn/cute
    /opt/vllm/bin/pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e .
fi
rm -rf vllm/vllm_flash_attn/cute
/opt/vllm/bin/python setup.py build_ext --inplace --build-temp /app/build/temp
rm -rf vllm/vllm_flash_attn/cute
ln -s ../../third_party/vllm-flash-attn/flash_attn/cute vllm/vllm_flash_attn/cute
case "$(readlink -f vllm/vllm_flash_attn/cute)" in
    "$(pwd -P)/third_party/vllm-flash-attn/flash_attn/cute") ;;
    *) echo "FATAL: vllm/vllm_flash_attn/cute -> $(readlink -f vllm/vllm_flash_attn/cute)" >&2; exit 7 ;;
esac
/opt/vllm/bin/python -c "import vllm; print('vllm', vllm.__version__, 'from', vllm.__file__)"
/opt/vllm/bin/python -c "import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print('cuda custom ops ok')"
/opt/vllm/bin/python -c "import vllm.vllm_flash_attn._vllm_fa2_C, vllm.vllm_flash_attn._vllm_fa3_C; print('flash-attention FA2/FA3 ok')"
/opt/vllm/bin/python -c "import vllm.vllm_flash_attn.cute.interface as i; print('FA4 CuTe-DSL interface ok:', i.__file__)"
echo "FA4 CuTe-DSL sources: $(readlink -f vllm/vllm_flash_attn/cute)"
