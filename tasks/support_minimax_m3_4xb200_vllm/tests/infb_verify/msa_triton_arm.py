#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The verifier-owned comparator arm of the MSA component timing; never in the agent image.

The contract's six entry points over the reference tree's own Triton MSA operators: selection by the landing's
Triton block scoring and top-k (prefill) and fused decode scoring and top-k, attention by the landing's Triton
block-sparse prefill kernel (its non-SM100 path) and split-K decode kernel. It runs only in the isolated reference
phase, with the reference tree first on the path; nothing of it reaches the agent image.
"""

from __future__ import annotations

import math

import torch

from vllm.models.minimax_m3.common.ops.index_topk import (
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)

HEAD_DIM = 128
NUM_HEADS = 16
TOPK = 16
SCALE = 1.0 / math.sqrt(HEAD_DIM)


class _Cache:
    __slots__ = ("kv", "index")

    def __init__(self, kv, index):
        self.kv, self.index = kv, index


def create_cache(num_pages: int, kv_dtype: torch.dtype, device):
    kv = torch.zeros((num_pages, 2, SPARSE_BLOCK_SIZE, 1, HEAD_DIM), dtype=kv_dtype, device=device)
    index = torch.zeros((num_pages, SPARSE_BLOCK_SIZE, HEAD_DIM), dtype=torch.bfloat16, device=device)
    return _Cache(kv, index)


def cache_views(cache):
    return cache.kv[:, 0, :, 0, :], cache.kv[:, 1, :, 0, :], cache.index


def _max(t):
    return int(t.max().item()) if t.numel() else 0


def select_prefill(cache, index_q, block_table, query_start_loc, seq_lens, context_lens):
    q_lens = query_start_loc[1:] - query_start_loc[:-1]
    max_q, max_s = _max(q_lens), _max(seq_lens)
    score = minimax_m3_index_score(index_q.view(-1, 1, HEAD_DIM), cache.index, block_table, query_start_loc,
                                   seq_lens, context_lens, max_q, max_s, 1, SCALE)
    return minimax_m3_index_topk(score, query_start_loc, context_lens, max_q, TOPK, 0, 1)[0]


def select_decode(cache, index_q, block_table, seq_lens):
    return minimax_m3_index_decode(index_q.view(-1, 1, HEAD_DIM), cache.index, block_table, seq_lens,
                                   _max(seq_lens), TOPK, 0, 1, 1, SCALE, 1)[0]


def attend_prefill(cache, q, selection, block_table, query_start_loc, seq_lens, context_lens):
    q_lens = query_start_loc[1:] - query_start_loc[:-1]
    out = torch.empty((q.shape[0], NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    minimax_m3_sparse_attn(q, cache.kv, selection.unsqueeze(0), block_table, query_start_loc, seq_lens,
                           context_lens, _max(q_lens), 1, SCALE, out)
    return out


def attend_decode(cache, q, selection, block_table, seq_lens):
    out = torch.empty((q.shape[0], NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    minimax_m3_sparse_attn_decode(q, cache.kv, selection.unsqueeze(0), block_table, seq_lens, 1, SCALE, out, 1)
    return out
