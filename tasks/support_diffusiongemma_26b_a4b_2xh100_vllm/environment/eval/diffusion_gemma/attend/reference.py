#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The `attend` operator in plain PyTorch: the contract's semantics, stated executably.

For a sequence with n keys and m query rows (q = a row's absolute position, p = a key's):

* encode:  row q sees p <= q; on sliding layers also p > q - W. If q lies in an image span [s, e)
           of the sequence, it also sees every p in [s, e), whatever the window.
* denoise: the rows are the canvas [c, n), c = n - m. Every row sees the whole canvas. Committed
           keys: all of [0, c) on full layers; on sliding layers one window shared by every row,
           clipped at 0: [c - (W - 1), c) (reading "w1") or [c - W, c) (reading "w").
* values:  weighted by the softmax, over the visible keys, of q.k at scale 1.0 (no softcap).

`attend_reference` works per sequence and in chunks of 256 rows, in FP32 by default or FP64, from
the stored BF16 keys and values. `attend_bf16` evaluates the same operator the way a BF16 kernel
rounds: BF16 score matmul, FP32 softmax rounded to BF16, BF16 P.V (the tolerance's yardstick).
"""
from __future__ import annotations

import torch

W = 1024  # the model's sliding window
CHUNK = 256
READINGS = ("w1", "w")


def visible_interval(kind: str, q: int, n: int, m: int, layer_kind: str, reading: str = "w1") -> tuple[int, int]:
    """The contiguous [lo, hi) of keys row q sees before image spans are added."""
    if kind == "encode":
        lo = max(q - (W - 1), 0) if layer_kind == "sliding" else 0
        return lo, q + 1
    c = n - m
    if layer_kind == "full":
        return 0, n
    shift = W - 1 if reading == "w1" else W
    return max(c - shift, 0), n


def visible_mask(kind: str, n: int, m: int, spans, layer_kind: str, reading: str = "w1",
                 device=None) -> torch.Tensor:
    """Bool [m, n]: row i (absolute position n - m + i) sees key p."""
    rows = torch.arange(n - m, n, device=device)
    keys = torch.arange(n, device=device)
    if kind == "encode":
        mask = keys[None, :] <= rows[:, None]
        if layer_kind == "sliding":
            mask &= keys[None, :] > rows[:, None] - W
        for s, e in spans:
            if e <= s:
                continue
            in_rows = (rows >= s) & (rows < e)
            in_keys = (keys >= s) & (keys < e)
            mask |= in_rows[:, None] & in_keys[None, :]
        return mask
    lo, hi = visible_interval("denoise", n - 1, n, m, layer_kind, reading)
    mask = torch.zeros(m, n, dtype=torch.bool, device=device)
    mask[:, lo:hi] = True
    return mask


def _gather(k_cache: torch.Tensor, block_table_row: torch.Tensor, n: int) -> torch.Tensor:
    """Keys (or values) 0..n-1 of one sequence from a paged cache [blocks, block_size, heads, dim]."""
    bs = k_cache.shape[1]
    nb = (n + bs - 1) // bs
    idx = block_table_row[:nb].long()
    return k_cache[idx].reshape(nb * bs, k_cache.shape[2], k_cache.shape[3])[:n]


def attend_reference(k_cache, v_cache, q, block_table, seq_lens, query_start_loc, encode, image_ranges,
                     layer_kind: str, dtype=torch.float32, reading: str = "w1") -> torch.Tensor:
    """The operator, per sequence and in chunks of 256 rows, at `dtype` (FP32 default, FP64 option)."""
    T, hq, d = q.shape
    hk = k_cache.shape[2]
    group = hq // hk
    out = torch.zeros(T, hq, d, dtype=dtype, device=q.device)
    qsl = query_start_loc.tolist()
    for r in range(len(qsl) - 1):
        a, b = qsl[r], qsl[r + 1]
        m, n = b - a, int(seq_lens[r])
        if m == 0:
            continue
        kind = "encode" if int(encode[r]) else "denoise"
        spans = [(int(s), int(e)) for s, e in image_ranges[r].tolist() if e > s] if image_ranges is not None else []
        k = _gather(k_cache, block_table[r], n).to(dtype)
        v = _gather(v_cache, block_table[r], n).to(dtype)
        mask = visible_mask(kind, n, m, spans, layer_kind, reading, device=q.device)
        for s0 in range(0, m, CHUNK):
            s1 = min(s0 + CHUNK, m)
            qs = q[a + s0:a + s1].to(dtype)
            for h in range(hq):
                sc = qs[:, h] @ k[:, h // group].T
                sc = sc.masked_fill(~mask[s0:s1], float("-inf"))
                out[a + s0:a + s1, h] = torch.softmax(sc, dim=-1) @ v[:, h // group]
    return out


def attend_bf16(k_cache, v_cache, q, block_table, seq_lens, query_start_loc, encode, image_ranges,
                layer_kind: str, reading: str = "w1") -> torch.Tensor:
    """The same operator with BF16 rounding where a BF16 kernel rounds."""
    T, hq, d = q.shape
    hk = k_cache.shape[2]
    group = hq // hk
    out = torch.zeros(T, hq, d, dtype=torch.bfloat16, device=q.device)
    qsl = query_start_loc.tolist()
    for r in range(len(qsl) - 1):
        a, b = qsl[r], qsl[r + 1]
        m, n = b - a, int(seq_lens[r])
        if m == 0:
            continue
        kind = "encode" if int(encode[r]) else "denoise"
        spans = [(int(s), int(e)) for s, e in image_ranges[r].tolist() if e > s] if image_ranges is not None else []
        k = _gather(k_cache, block_table[r], n).to(torch.bfloat16)
        v = _gather(v_cache, block_table[r], n).to(torch.bfloat16)
        mask = visible_mask(kind, n, m, spans, layer_kind, reading, device=q.device)
        for s0 in range(0, m, CHUNK):
            s1 = min(s0 + CHUNK, m)
            qs = q[a + s0:a + s1].to(torch.bfloat16)
            for h in range(hq):
                sc = (qs[:, h] @ k[:, h // group].T).float()
                sc = sc.masked_fill(~mask[s0:s1], float("-inf"))
                p = torch.softmax(sc, dim=-1).to(torch.bfloat16)
                out[a + s0:a + s1, h] = p @ v[:, h // group]
    return out
