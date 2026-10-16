#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The component-measurement entry points of DeepSeek-V4.1's two-level sparse indexer.

Each call performs one step of the model's four candidate-consuming indexers (layers 24, 28, 32 and 36), which
share one set of index keys and one candidate pool, with their own queries and head weights: from quantized
inputs to the four request-local selection buffers. They are the interface through which the complete indexer
pipeline is measured and checked; they call the same implementation native serving uses. The contract (shapes,
dtypes, coordinates, numerical rules and workspace lifetime) is
``/app/eval/deepseek_v41_flash/sparse_indexer/README.md``.

Both entry points write request-local positions into ``out`` (int32 ``[4, R, 512]``, -1 padded) and return an
opaque workspace that the caller may pass back as ``workspace=`` on a later call.
"""

from typing import Any

import torch


def prefill(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    candidate_blocks: torch.Tensor,
    out: torch.Tensor,
    *,
    workspace: Any = None,
) -> Any:
    """Packed prefill: four consumers' top-512 selections over one packed K workspace.

    q: int8 [4, R, 32, 64]; q_scale: int32 [4, R, 32]; k_quant: int8 [N, 64]; k_scale: int32 [N];
    weights: bf16 [4, R, 32]; k_start, k_end: int32 [R] (absolute ``[ks, ke)`` in the K workspace);
    candidate_blocks: int32 [R, 2048] (request-local block ids, -1 padded); out: int32 [4, R, 512].
    """
    raise NotImplementedError("sparse_indexer_eval.prefill is not implemented")


def decode(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    row_indices: torch.Tensor,
    candidate_blocks: torch.Tensor,
    out: torch.Tensor,
    *,
    workspace: Any = None,
) -> Any:
    """Paged decode and target verification: four consumers' top-512 selections over one paged index cache.

    q: int8 [4, R, 1, 32, 64]; q_scale: int32 [4, R, 1, 32]; kv_cache: uint8 [pages, page_kv, 1, 68];
    weights: bf16 [4, R, 32]; context_lens, row_indices: int32 [R]; block_table: int32
    [R, ceil(max_visible_length / page_kv)]; candidate_blocks: int32 [R, 2048]; out: int32 [4, R, 512].
    """
    raise NotImplementedError("sparse_indexer_eval.decode is not implemented")
