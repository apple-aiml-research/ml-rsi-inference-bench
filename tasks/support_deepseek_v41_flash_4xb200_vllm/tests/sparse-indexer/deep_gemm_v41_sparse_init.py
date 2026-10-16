#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The supplied MXFP4 sparse MQA-logits runtime for DeepSeek-V4.1's two-level indexer.

Its source is the editable tree ``third_party/deepgemm_v41_sparse`` at the repository root. The compiled extension
(``_C``) and the JIT include tree (``include/``) next to this file are build products of that tree, written by
``tools/build_deepgemm_v41_sparse.py`` (``build-manifest.json`` records which source they were built from); rebuild
after changing the tree. The package is separate from the engine's ordinary DeepGEMM provider
(``vllm.third_party.deep_gemm``): its own extension, its own JIT include root, and its own configuration prefix
``DSV41_SPARSE`` (for example ``DSV41_SPARSE_JIT_CACHE_DIR``).

Importing it initializes no CUDA context; device state is created on the first operation.
"""

import os

import torch  # noqa: F401  (the extension links against torch's libraries)

from . import _C

_C.init(os.path.dirname(os.path.abspath(__file__)))
_C.use_deterministic_algorithms(True)

from ._C import (  # noqa: E402
    fp8_fp4_paged_sparse_mqa_logits,
    fp8_fp4_sparse_mqa_logits,
    get_paged_sparse_mqa_logits_metadata,
    get_sparse_mqa_logits_metadata,
)

__all__ = [
    "get_sparse_mqa_logits_metadata",
    "get_paged_sparse_mqa_logits_metadata",
    "fp8_fp4_sparse_mqa_logits",
    "fp8_fp4_paged_sparse_mqa_logits",
]
