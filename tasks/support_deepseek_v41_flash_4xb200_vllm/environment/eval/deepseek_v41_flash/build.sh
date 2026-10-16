#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Rebuild the engine in place after changing CUDA, C++ or Rust sources or the build's configuration
# (tests/native.py's `compiled_sources_touched` lists what counts), then the sparse MQA-logits runtime
# (third_party/deepgemm_v41_sparse -> vllm/third_party/deep_gemm_v41_sparse; sparse_indexer/README.md).
#
# This is the same command the image was built with and the same one the evaluation runs on a submitted tree
# that changes compiled sources, offline and without build isolation. Pure-Python, Triton and TileLang edits
# need no rebuild: the install is editable.
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
# The sparse MQA-logits runtime: its own extension and JIT include tree, from the committed source tree.
/opt/vllm/bin/python tools/build_deepgemm_v41_sparse.py third_party/deepgemm_v41_sparse \
  vllm/third_party/deep_gemm_v41_sparse /opt/vllm/bin/python
