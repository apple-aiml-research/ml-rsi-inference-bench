#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Plain PyTorch statement of the four MiniMax Sparse Attention operators of this task's contract."""

from __future__ import annotations

import math

import torch

BLOCK = 128
TOPK = 16
HEAD_DIM = 128
NUM_HEADS = 16
SCALE = 1.0 / math.sqrt(HEAD_DIM)
CHUNK = 256


def token_rows(pages: torch.Tensor, block_table_row: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """The first ``n_tokens`` token rows of one request, gathered from a ``[pages, 128, 128]`` view."""
    n_blocks = (n_tokens + BLOCK - 1) // BLOCK
    phys = block_table_row[:n_blocks].long()
    if pages.element_size() == 1:  # FP8: gather the bytes, then read the values
        rows = pages.view(torch.uint8).index_select(0, phys).view(pages.dtype)
    else:
        rows = pages.index_select(0, phys)
    return rows.reshape(n_blocks * BLOCK, pages.shape[-1])[:n_tokens]


def block_scores(index_q: torch.Tensor, index_k: torch.Tensor, positions: torch.Tensor,
                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[Q, n_blocks]`` max-pooled index scores; ``-inf`` where a block has no visible token.

    ``index_q`` ``[Q, 128]``, ``index_k`` ``[S, 128]`` (the request's tokens), ``positions`` ``[Q]``.
    """
    s = index_k.shape[0]
    n_blocks = (s + BLOCK - 1) // BLOCK
    scores = index_q.to(dtype) @ index_k.to(dtype).T  # [Q, S]
    future = torch.arange(s, device=scores.device)[None, :] > positions[:, None]
    scores = scores.masked_fill(future, float("-inf"))
    pad = n_blocks * BLOCK - s
    if pad:
        scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    return scores.view(-1, n_blocks, BLOCK).amax(dim=-1)


def select_rows(scores: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Selections ``[Q, 16]`` from block scores: the local block first, then the best other visible blocks.

    Ties are broken towards the lower block index, which is one admissible order among many (the contract accepts any
    selection within the published score margin).
    """
    q = scores.shape[0]
    out = torch.full((q, TOPK), -1, dtype=torch.int32, device=scores.device)
    local = (positions // BLOCK).long()
    boosted = scores.clone()
    boosted[torch.arange(q, device=scores.device), local] = float("inf")
    n_blocks = scores.shape[1]
    k = min(TOPK, n_blocks)
    # Stable descending order by score, then by ascending block index.
    order = torch.argsort(-boosted, dim=-1, stable=True)[:, :k]
    vals = boosted.gather(1, order)
    idx = torch.where(vals == float("-inf"), torch.full_like(order, -1), order)
    out[:, :k] = idx.to(torch.int32)
    return out


def select_prefill(index_k_pages, index_q, block_table, query_start_loc, seq_lens, context_lens,
                   *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    total = index_q.shape[0]
    out = torch.full((total, TOPK), -1, dtype=torch.int32, device=index_q.device)
    for r in range(seq_lens.numel()):
        q0, q1 = int(query_start_loc[r]), int(query_start_loc[r + 1])
        if q1 == q0:
            continue
        s = int(seq_lens[r])
        keys = token_rows(index_k_pages, block_table[r], s)
        for c0 in range(q0, q1, CHUNK):
            c1 = min(q1, c0 + CHUNK)
            pos = int(context_lens[r]) + torch.arange(c0 - q0, c1 - q0, device=index_q.device)
            out[c0:c1] = select_rows(block_scores(index_q[c0:c1], keys, pos, dtype), pos)
    return out


def select_decode(index_k_pages, index_q, block_table, seq_lens, *, dtype: torch.dtype = torch.float32):
    rows = seq_lens.numel()
    out = torch.full((rows, TOPK), -1, dtype=torch.int32, device=index_q.device)
    for r in range(rows):
        s = int(seq_lens[r])
        if s == 0:
            continue
        keys = token_rows(index_k_pages, block_table[r], s)
        pos = torch.tensor([s - 1], device=index_q.device)
        out[r:r + 1] = select_rows(block_scores(index_q[r:r + 1], keys, pos, dtype), pos)
    return out


def gather_attended(k: torch.Tensor, v: torch.Tensor, selection: torch.Tensor, positions: torch.Tensor):
    """The selected blocks' keys and values [Q, 16*128, 128] and the attended mask [Q, 16*128].

    Token j of a selected block is attended iff j <= p; -1 slots and tokens past the request's length are not.
    """
    s = k.shape[0]
    q = selection.shape[0]
    sel = selection.long()
    offsets = torch.arange(BLOCK, device=selection.device)
    tok = (sel.clamp(min=0)[:, :, None] * BLOCK + offsets[None, None, :]).reshape(q, TOPK * BLOCK)
    valid = (sel >= 0).repeat_interleave(BLOCK, dim=1) & (tok < s) & (tok <= positions[:, None])
    tok = tok.clamp(max=s - 1)
    return k[tok], v[tok], valid


def attend_rows(q, k, v, selection, positions, *, dtype=torch.float32, bf16_operands: bool = False):
    """[Q, 16, 128] attention of q [Q, 16, 128] over one request's k/v [S, 128].

    bf16_operands=True is the rounding-envelope reference of the BF16 policy: BF16 score matmul (the scale
    applied to BF16 queries), softmax computed in FP32 and rounded to BF16, BF16 P.V matmul (the upcast=False
    reference of FlashAttention's and MSA's tests).
    """
    kk, vv, valid = gather_attended(k, v, selection, positions)
    if bf16_operands:
        qq = (q.to(torch.bfloat16) * SCALE).to(torch.bfloat16)
        scores = torch.einsum("qhd,qsd->qhs", qq, kk.to(torch.bfloat16))
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        p = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
        return torch.einsum("qhs,qsd->qhd", p, vv.to(torch.bfloat16))
    scores = torch.einsum("qhd,qsd->qhs", q.to(dtype) * SCALE, kk.to(dtype))
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("qhs,qsd->qhd", p, vv.to(dtype))


def attend_prefill(k_pages, v_pages, q, selection, block_table, query_start_loc, seq_lens, context_lens,
                   *, dtype: torch.dtype = torch.float32, bf16_operands: bool = False) -> torch.Tensor:
    out_dtype = torch.bfloat16 if bf16_operands else dtype
    out = torch.zeros((q.shape[0], NUM_HEADS, HEAD_DIM), dtype=out_dtype, device=q.device)
    for r in range(seq_lens.numel()):
        q0, q1 = int(query_start_loc[r]), int(query_start_loc[r + 1])
        if q1 == q0:
            continue
        s = int(seq_lens[r])
        work = torch.bfloat16 if bf16_operands else dtype
        k = token_rows(k_pages, block_table[r], s).to(work)
        v = token_rows(v_pages, block_table[r], s).to(work)
        for c0 in range(q0, q1, CHUNK):
            c1 = min(q1, c0 + CHUNK)
            pos = int(context_lens[r]) + torch.arange(c0 - q0, c1 - q0, device=q.device)
            out[c0:c1] = attend_rows(q[c0:c1], k, v, selection[c0:c1], pos, dtype=dtype,
                                     bf16_operands=bf16_operands)
    return out


def attend_decode(k_pages, v_pages, q, selection, block_table, seq_lens, *, dtype: torch.dtype = torch.float32,
                  bf16_operands: bool = False) -> torch.Tensor:
    out_dtype = torch.bfloat16 if bf16_operands else dtype
    out = torch.zeros((q.shape[0], NUM_HEADS, HEAD_DIM), dtype=out_dtype, device=q.device)
    for r in range(seq_lens.numel()):
        s = int(seq_lens[r])
        if s == 0:
            continue
        work = torch.bfloat16 if bf16_operands else dtype
        k = token_rows(k_pages, block_table[r], s).to(work)
        v = token_rows(v_pages, block_table[r], s).to(work)
        pos = torch.tensor([s - 1], device=q.device)
        out[r:r + 1] = attend_rows(q[r:r + 1], k, v, selection[r:r + 1], pos, dtype=dtype,
                                   bf16_operands=bf16_operands)
    return out
