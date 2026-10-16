#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Rebuild the engine in place after changing CUDA or C++ sources.
#
# This is the SAME command the image was built with and the same one the verifier runs on the
# submitted tree, offline and without build isolation: a different build path for the reference
# solution or for the candidate would be a defect, not a convenience. Pure-Python edits need no
# rebuild -- the install is editable.
set -euo pipefail
cd "${INFB_WORKSPACE:-/app}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS="${MAX_JOBS:-64}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-64}"
/opt/vllm/bin/pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e .
/opt/vllm/bin/python -c "import vllm; print('vllm', vllm.__version__, 'from', vllm.__file__)"
/opt/vllm/bin/python -c "import vllm._C, vllm._moe_C; print('cuda custom ops ok')"
