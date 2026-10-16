#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Rebuild the engine in place after changing CUDA, C++ or Rust sources:
#   bash /app/eval/qwen38/build.sh
#
# This is the SAME command the image was built with and the same one the verifier runs on the
# submitted tree, offline and without build isolation. Pure-Python edits need no rebuild -- the
# install is editable.
set -euo pipefail
cd "${INFB_WORKSPACE:-/app}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0a}"
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS="${MAX_JOBS:-64}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-64}"
export CARGO_NET_OFFLINE=true
/opt/vllm/bin/pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e .
/opt/vllm/bin/python -c "import vllm; print('vllm', vllm.__version__, 'from', vllm.__file__)"
# The CUDA custom ops of THIS vLLM. Not `vllm._C`: at this commit setup.py adds that name only on
# the ROCm and CPU paths (`ext_modules`, `if _build_custom_ops()`).
/opt/vllm/bin/python -c "import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print('cuda custom ops ok')"
