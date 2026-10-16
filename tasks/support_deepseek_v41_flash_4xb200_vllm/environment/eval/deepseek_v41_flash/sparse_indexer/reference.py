#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The sparse-indexer contract's mathematics, independently of any kernel: MXFP4 unpacking, the fused paged cache,
the FP32 score, the candidate coordinates and the comparison rules of README.md.

Everything here is plain PyTorch in FP32 with TF32 and reduced-precision reductions disabled, evaluated in chunks so
that no ``[rows, heads, keys, dim]`` broadcast is ever formed. It is the arithmetic the contract defines, not an
implementation to copy: it is slow on purpose and makes no attempt to be efficient.

PyTorch is imported inside the functions, so the module imports with the standard library alone.
"""
from __future__ import annotations

import math

#: The model's geometry (README.md, "Geometry").
NUM_HEADS = 32
HEAD_DIM = 128
PACKED_BYTES = HEAD_DIM // 2  # two E2M1 values per byte
SCALE_GROUP = 32  # coordinates per UE8M0 scale byte
CANDIDATE_BLOCK = 8
NUM_CANDIDATES = 2048
TOPK = 512
MAX_VISIBLE = 32768
NUM_CONSUMERS = 4
PAGE_RECORD_BYTES = PACKED_BYTES + 4  # 64 value bytes and one 4-byte scale word per token

#: E2M1 magnitudes for the low three bits; bit 3 is the sign.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: The public numerical allowance (README.md, "Comparison rules").
LOGIT_P99 = 0.01
LOGIT_MAX = 0.02
SOURCE_DELTA = 0.02


def _torch():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return torch


def unpack_e2m1(packed):
    """int8/uint8 ``[..., 64]`` -> float32 ``[..., 128]``: low nibble first, bit 3 the sign."""
    torch = _torch()
    b = packed.view(torch.uint8).to(torch.int32)
    lo, hi = b & 0xF, (b >> 4) & 0xF
    codes = torch.stack((lo, hi), dim=-1).flatten(-2)
    table = torch.tensor(E2M1, dtype=torch.float32, device=packed.device)
    mag = table[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -mag, mag)


def unpack_scales(words):
    """int32 ``[...]`` scale words -> float32 ``[..., 4]``: byte i (low byte first) scales coordinates
    ``[32 i, 32 i + 32)`` by ``2 ** (byte - 127)``."""
    torch = _torch()
    b = words.contiguous().view(torch.uint8).view(*words.shape, 4).to(torch.int32)
    return torch.pow(2.0, (b - 127).to(torch.float32))


def dequantize(packed, words):
    """int8 ``[..., 64]`` and int32 ``[...]`` -> float32 ``[..., 128]``."""
    torch = _torch()
    values = unpack_e2m1(packed)
    scales = unpack_scales(words)
    return (values.view(*values.shape[:-1], 4, SCALE_GROUP) * scales.unsqueeze(-1)).flatten(-2)


def page_tokens(kv_cache, pages, length: int):
    """The first ``length`` tokens of one request from the fused paged cache: ``(values int8 [length, 64],
    words int32 [length])``. Each page stores ``page_kv * 64`` value bytes and then ``page_kv * 4`` scale bytes (not
    interleaved 68-byte records); its stride may exceed that and is read from the tensor."""
    torch = _torch()
    num_pages, page_kv = kv_cache.shape[0], kv_cache.shape[1]
    stride = kv_cache.stride(0)
    flat = torch.as_strided(kv_cache, (num_pages, stride), (stride, 1))
    values, words = [], []
    for i in range(math.ceil(length / page_kv) if length else 0):
        raw = flat[int(pages[i])]
        n = min(page_kv, length - i * page_kv)
        values.append(raw[: page_kv * PACKED_BYTES].view(page_kv, PACKED_BYTES)[:n])
        words.append(raw[page_kv * PACKED_BYTES : page_kv * PAGE_RECORD_BYTES].view(torch.int32)[:n])
    if not values:
        return (torch.empty((0, PACKED_BYTES), dtype=torch.uint8, device=kv_cache.device),
                torch.empty((0,), dtype=torch.int32, device=kv_cache.device))
    return torch.cat(values).view(torch.int8), torch.cat(words)


def scores(q_deq, weights, k_deq, chunk: int = 4096):
    """FP32 ``score[r, t] = sum_h w[r, h] * max(q[r, h] . k[t], 0)`` for q ``[R, 32, 128]``, weights ``[R, 32]``
    and keys ``[N, 128]``, all float32, in key chunks."""
    torch = _torch()
    w = weights.to(torch.float32)
    out = torch.empty((q_deq.shape[0], k_deq.shape[0]), dtype=torch.float32, device=q_deq.device)
    for n0 in range(0, k_deq.shape[0], chunk):
        k = k_deq[n0 : n0 + chunk]
        dots = torch.einsum("rhd,nd->rhn", q_deq, k).clamp_min_(0)
        out[:, n0 : n0 + k.shape[0]] = torch.einsum("rhn,rh->rn", dots, w)
    return out


def candidate_positions(candidate_row, visible: int):
    """Sorted request-local positions a consumer may select for one row: the positions ``8 b + o < visible`` of each
    valid (non-negative) candidate block ``b``. Padding and invisible positions are excluded."""
    torch = _torch()
    blocks = candidate_row[candidate_row >= 0].to(torch.int64).unique()
    pos = (blocks[:, None] * CANDIDATE_BLOCK + torch.arange(CANDIDATE_BLOCK, device=blocks.device)).flatten()
    return pos[pos < visible]


def row_error(got, want):
    """Per-row normalized error of scores over valid candidates: ``|got - want| / max|want|`` (an all-zero reference
    row requires zero scores); returns ``(p99, max)`` for one row."""
    torch = _torch()
    got, want = got.to(torch.float32), want.to(torch.float32)
    if want.numel() == 0:
        return 0.0, 0.0
    scale = want.abs().max()
    if scale == 0:
        err = (got != 0).to(torch.float32)
    else:
        err = (got - want).abs() / scale
    if not torch.isfinite(got).all():
        return math.inf, math.inf
    return float(torch.quantile(err, 0.99)), float(err.max())


def selection_problems(selected, allowed, visible: int, k: int = TOPK) -> list[str]:
    """Structural problems of one output row: int32 request-local positions, distinct, each allowed and visible,
    exactly ``min(k, |allowed|)`` valid entries, -1 padding after nothing else."""
    torch = _torch()
    out = []
    sel = selected.to(torch.int64)
    valid = sel[sel >= 0]
    if (sel < -1).any():
        out.append("an entry below -1")
    if valid.unique().numel() != valid.numel():
        out.append("a position selected twice")
    if (valid >= visible).any():
        out.append("a position beyond the row's visible length")
    if valid.numel() and not torch.isin(valid, allowed).all():
        out.append("a position outside the candidate pool")
    want = min(k, allowed.numel())
    if valid.numel() != want:
        out.append(f"{valid.numel()} valid entries, the contract requires {want}")
    return out


def topk_matches(selected, allowed, row_scores, k: int = TOPK) -> bool:
    """Whether the valid entries of ``selected`` are a top-``k`` of ``row_scores`` over ``allowed`` (positions),
    accepting any choice among equal scores: the selected multiset of scores equals the top-``k`` multiset."""
    torch = _torch()
    sel = selected.to(torch.int64)
    sel = sel[sel >= 0]
    pool = row_scores[allowed]
    want = torch.sort(torch.topk(pool, min(k, pool.numel())).values).values
    got = torch.sort(row_scores[sel]).values
    return got.shape == want.shape and bool((got == want).all())


def within_selection_slack(selected, allowed, row_scores, delta: float, k: int = TOPK) -> bool:
    """For a selection whose scores are not materialized: no excluded allowed position may score more than
    ``2 * delta`` above the smallest selected score."""
    torch = _torch()
    sel = selected.to(torch.int64)
    sel = sel[sel >= 0]
    if sel.numel() == 0:
        return allowed.numel() == 0
    excluded = allowed[~torch.isin(allowed, sel)]
    if excluded.numel() == 0:
        return True
    return float(row_scores[excluded].max()) <= float(row_scores[sel].min()) + 2 * delta
