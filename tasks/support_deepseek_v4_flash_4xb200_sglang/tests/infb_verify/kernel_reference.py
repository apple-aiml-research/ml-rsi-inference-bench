#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The private float64 references of the five kernel
interfaces the task assigns. Verifier-only: never imported by, or shipped as, agent material. Each reference is
independent of every implementation under test -- ordinary PyTorch, float64 reductions, its own packed-cache decoder
and address arithmetic -- and computes in bounded chunks so no queries x heads x top-k x dimension tensor exists."""
from __future__ import annotations

import math

import torch

D_MODEL1 = 512
D_NOPE = 448
D_ROPE = 64
GROUP = 64
N_GROUPS = 7
DATA_ROW = 576
SCALE_ROW = 8
INDEXER_D = 128
FP8_MAX = 448.0


# ── packed representations ───────────────────────────────────────────────────────────────────────


def model1_page_bytes(page_size: int) -> int:
    return math.ceil(584 * page_size / DATA_ROW) * DATA_ROW


def indexer_page_bytes(page_size: int) -> int:
    return (INDEXER_D + 4) * page_size


def _fp8_to_f64(u8: torch.Tensor) -> torch.Tensor:
    return u8.contiguous().view(torch.float8_e4m3fn).to(torch.float64)


def _e8m0_to_f64(u8: torch.Tensor) -> torch.Tensor:
    e = u8.to(torch.int64)
    out = torch.pow(torch.tensor(2.0, dtype=torch.float64), (e - 127).to(torch.float64))
    return torch.where(e == 255, torch.full_like(out, float("nan")), out)


def decode_model1(cache: torch.Tensor, page_size: int, token_index: torch.Tensor) -> torch.Tensor:
    """The float64 values of MODEL1 entries at physical token indices (page * page_size + offset), from the bytes.

    `cache` is the pool as bytes, [num_pages, >= page stride]; `token_index` any integer shape; result [..., 512]."""
    raw = cache.reshape(cache.shape[0], -1).to(torch.uint8) if cache.dtype != torch.uint8 else cache.reshape(cache.shape[0], -1)
    idx = token_index.reshape(-1).to(torch.int64).cpu()
    page, off = idx // page_size, idx % page_size
    raw = raw.cpu()
    data_start = off * DATA_ROW
    cols = torch.arange(DATA_ROW, dtype=torch.int64)
    data = raw[page.unsqueeze(1), data_start.unsqueeze(1) + cols.unsqueeze(0)]            # [n, 576]
    scol = torch.arange(N_GROUPS, dtype=torch.int64)
    sstart = DATA_ROW * page_size + off * SCALE_ROW
    scales = _e8m0_to_f64(raw[page.unsqueeze(1), sstart.unsqueeze(1) + scol.unsqueeze(0)])  # [n, 7]
    nope = _fp8_to_f64(data[:, :D_NOPE]).view(-1, N_GROUPS, GROUP) * scales.unsqueeze(-1)
    rope = data[:, D_NOPE:].contiguous().view(torch.bfloat16).to(torch.float64)            # [n, 64]
    out = torch.cat([nope.reshape(-1, D_NOPE), rope], dim=-1)
    return out.view(*token_index.shape, D_MODEL1)


def decode_indexer(cache: torch.Tensor, page_size: int, token_index: torch.Tensor) -> torch.Tensor:
    """The float64 values of indexer entries (128 E4M3 values per token, then a 4-byte FP32 scale per token)."""
    raw = cache.reshape(cache.shape[0], -1).cpu()
    raw = raw if raw.dtype == torch.uint8 else raw.view(torch.uint8)
    idx = token_index.reshape(-1).to(torch.int64).cpu()
    page, off = idx // page_size, idx % page_size
    cols = torch.arange(INDEXER_D, dtype=torch.int64)
    data = raw[page.unsqueeze(1), (off * INDEXER_D).unsqueeze(1) + cols.unsqueeze(0)]
    scol = torch.arange(4, dtype=torch.int64)
    sbytes = raw[page.unsqueeze(1), (INDEXER_D * page_size + off * 4).unsqueeze(1) + scol.unsqueeze(0)]
    scale = sbytes.contiguous().view(torch.float32).to(torch.float64).view(-1, 1)
    return (_fp8_to_f64(data) * scale).view(*token_index.shape, INDEXER_D)


# ── the store formula (pinned SGLang store.cuh), and its exact bytes ─────────────────────────────────


def _e8m0_ceil(raw: torch.Tensor) -> torch.Tensor:
    """The E8M0 byte of the smallest power of two >= raw (raw > 0, float32 semantics of the kernel)."""
    r = raw.to(torch.float64)
    e = torch.ceil(torch.log2(r))
    # log2 of an exact power of two is exact; guard the neighbours explicitly
    e = torch.where(torch.pow(2.0, e - 1) >= r, e - 1, e)
    e = torch.where(torch.pow(2.0, e) < r, e + 1, e)
    return (e + 127).to(torch.int64)


def store_flashmla_expected(x: torch.Tensor) -> dict:
    """Expected MODEL1 row bytes of float32 inputs x [n, 512]: 576 data bytes and 7 scale bytes per row."""
    x32 = x.to(torch.float32).cpu()
    n = x32.shape[0]
    groups = x32[:, :D_NOPE].view(n, N_GROUPS, GROUP)
    raw = torch.clamp_min(groups.abs().amax(dim=-1), 1e-4) / FP8_MAX                     # float32 like the kernel
    sbyte = _e8m0_ceil(raw)                                                              # [n, 7]
    inv = torch.pow(torch.tensor(2.0, dtype=torch.float64), -(sbyte - 127).to(torch.float64)).to(torch.float32)
    q = (groups * inv.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8).view(n, D_NOPE)
    rope = x32[:, D_NOPE:].to(torch.bfloat16).contiguous().view(torch.uint8).view(n, 2 * D_ROPE)
    return {"data": torch.cat([q, rope], dim=-1), "scale": sbyte.to(torch.uint8)}


def store_indexer_expected(x: torch.Tensor) -> dict:
    """Expected indexer bytes of float32 inputs x [n, 128]: 128 FP8 bytes and the FP32 scale per row."""
    x32 = x.to(torch.float32).cpu()
    scale = torch.clamp_min(x32.abs().amax(dim=-1, keepdim=True), 1e-4) / FP8_MAX
    q = (x32 * (1.0 / scale)).to(torch.float8_e4m3fn).view(torch.uint8)
    return {"data": q, "scale": scale.view(-1)}


def read_model1_rows(cache: torch.Tensor, page_size: int, token_index: torch.Tensor) -> dict:
    """The raw data and scale bytes of MODEL1 rows (padding byte excluded)."""
    raw = cache.reshape(cache.shape[0], -1).cpu()
    raw = raw if raw.dtype == torch.uint8 else raw.view(torch.uint8)
    idx = token_index.reshape(-1).to(torch.int64).cpu()
    page, off = idx // page_size, idx % page_size
    cols = torch.arange(DATA_ROW, dtype=torch.int64)
    data = raw[page.unsqueeze(1), (off * DATA_ROW).unsqueeze(1) + cols.unsqueeze(0)]
    scol = torch.arange(N_GROUPS, dtype=torch.int64)
    scale = raw[page.unsqueeze(1), (DATA_ROW * page_size + off * SCALE_ROW).unsqueeze(1) + scol.unsqueeze(0)]
    return {"data": data, "scale": scale}


def read_indexer_rows(cache: torch.Tensor, page_size: int, token_index: torch.Tensor) -> dict:
    raw = cache.reshape(cache.shape[0], -1).cpu()
    raw = raw if raw.dtype == torch.uint8 else raw.view(torch.uint8)
    idx = token_index.reshape(-1).to(torch.int64).cpu()
    page, off = idx // page_size, idx % page_size
    cols = torch.arange(INDEXER_D, dtype=torch.int64)
    data = raw[page.unsqueeze(1), (off * INDEXER_D).unsqueeze(1) + cols.unsqueeze(0)]
    scol = torch.arange(4, dtype=torch.int64)
    sb = raw[page.unsqueeze(1), (INDEXER_D * page_size + off * 4).unsqueeze(1) + scol.unsqueeze(0)]
    return {"data": data, "scale": sb.contiguous().view(torch.float32).view(-1)}


def row_byte_spans_model1(page_size: int, token_index: torch.Tensor) -> list[tuple[int, int, int]]:
    """(page, start, end) byte spans a set of MODEL1 rows occupies (data and scale rows, the padding byte included)."""
    spans = []
    for t in token_index.reshape(-1).tolist():
        page, off = t // page_size, t % page_size
        spans.append((page, off * DATA_ROW, off * DATA_ROW + DATA_ROW))
        spans.append((page, DATA_ROW * page_size + off * SCALE_ROW, DATA_ROW * page_size + (off + 1) * SCALE_ROW))
    return spans


def row_byte_spans_indexer(page_size: int, token_index: torch.Tensor) -> list[tuple[int, int, int]]:
    spans = []
    for t in token_index.reshape(-1).tolist():
        page, off = t // page_size, t % page_size
        spans.append((page, off * INDEXER_D, (off + 1) * INDEXER_D))
        spans.append((page, INDEXER_D * page_size + off * 4, INDEXER_D * page_size + (off + 1) * 4))
    return spans


# ── attention ─────────────────────────────────────────────────────────────────────────────────────


def sparse_attention(q: torch.Tensor, keys: torch.Tensor, valid: torch.Tensor, sink: torch.Tensor | None,
                     scale: float, d_v: int = D_MODEL1, chunk: int = 16) -> dict:
    """float64 attention over per-query key lists.

    q [N, H, D]; keys [N, K, D] (the gathered selected entries, duplicates kept); valid [N, K] bool; sink [H] or None.
    Returns out [N, H, d_v], lse [N, H] (sink excluded; +inf when empty), max_logits [N, H] (-inf when empty)."""
    n, h, _ = q.shape
    out = torch.empty(n, h, d_v, dtype=torch.float64)
    lse = torch.empty(n, h, dtype=torch.float64)
    mx = torch.empty(n, h, dtype=torch.float64)
    s64 = None if sink is None else sink.to(torch.float64).cpu()
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        qq = q[a:b].to(torch.float64).cpu()
        kk = keys[a:b].to(torch.float64).cpu()
        vv = valid[a:b].cpu()
        z = scale * torch.einsum("nhd,nkd->nhk", qq, kk)
        z = z.masked_fill(~vv.unsqueeze(1), float("-inf"))
        zmax = z.amax(dim=-1)                                                   # [n, h], -inf when empty
        l = torch.logsumexp(z, dim=-1)                                          # -inf when empty
        m = zmax if s64 is None else torch.maximum(zmax, s64.view(1, h).expand_as(zmax))
        m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        w = torch.exp(z - m_safe.unsqueeze(-1))                                 # 0 at masked entries
        num = torch.einsum("nhk,nkd->nhd", w, kk[..., :d_v])
        den = w.sum(dim=-1)
        if s64 is not None:
            den = den + torch.exp(s64.view(1, h) - m_safe)
        o = torch.where(den.unsqueeze(-1) > 0, num / den.clamp_min(1e-300).unsqueeze(-1), torch.zeros_like(num))
        empty = ~vv.any(dim=-1)                                                 # [n]
        o[empty] = 0.0
        out[a:b] = o
        lse[a:b] = torch.where(torch.isfinite(l), l, torch.full_like(l, float("inf")))
        mx[a:b] = zmax
    return {"out": out, "lse": lse, "max_logits": mx}


# ── compression, post-processing ───────────────────────────────────────────────────────────────────


def compress_emissions(kv_score: torch.Tensor, ape: torch.Tensor, ratio: int, head_dim: int,
                       positions: list[int]) -> torch.Tensor:
    """float64 compressed rows at the emitting positions `positions` of one request's logical history.

    kv_score [L, (2 or 4) * D] float32: C4 rows [kv_overlap, kv_fresh, score_overlap, score_fresh], C128 rows
    [kv, score]; ape [8, D] (C4) or [128, D] (C128); every position p must satisfy (p + 1) % ratio == 0."""
    x = kv_score.to(torch.float64).cpu()
    bias = ape.to(torch.float64).cpu()
    d = head_dim
    rows = []
    for p in positions:
        if (p + 1) % ratio:
            raise ValueError(f"position {p} does not emit at ratio {ratio}")
        if ratio == 4:
            if p < 7:
                kv_ov = torch.zeros(4, d, dtype=torch.float64)
                sc_ov = torch.full((4, d), float("-inf"), dtype=torch.float64)
            else:
                kv_ov = x[p - 7:p - 3, :d]
                sc_ov = x[p - 7:p - 3, 2 * d:3 * d]
            kv_fr = x[p - 3:p + 1, d:2 * d]
            sc_fr = x[p - 3:p + 1, 3 * d:4 * d]
            kv = torch.cat([kv_ov, kv_fr], dim=0)
            sc = torch.cat([sc_ov, sc_fr], dim=0) + bias
        elif ratio == 128:
            kv = x[p - 127:p + 1, :d]
            sc = x[p - 127:p + 1, d:2 * d] + bias
        else:
            raise ValueError(f"ratio {ratio}")
        rows.append((kv * torch.softmax(sc, dim=0)).sum(dim=0))
    return torch.stack(rows) if rows else torch.empty(0, d, dtype=torch.float64)


def norm_rope(x: torch.Tensor, weight: torch.Tensor, eps: float, freqs_real: torch.Tensor,
              positions: list[int]) -> torch.Tensor:
    """float64 RMSNorm then RoPE on the last 64 coordinates at `positions` (the group-start positions).

    freqs_real [max_pos, 64]: per position the interleaved (cos, sin) pairs, as `torch.view_as_real(freqs_cis)`."""
    v = x.to(torch.float64).cpu()
    w = weight.to(torch.float64).cpu()
    y = v * torch.rsqrt((v * v).mean(dim=-1, keepdim=True) + eps) * w
    f = freqs_real.to(torch.float64).cpu()[torch.tensor(positions, dtype=torch.int64)].view(len(positions), D_ROPE // 2, 2)
    r = y[:, -D_ROPE:].reshape(len(positions), D_ROPE // 2, 2)
    xr, xi, fr, fi = r[..., 0], r[..., 1], f[..., 0], f[..., 1]
    rot = torch.stack([xr * fr - xi * fi, xr * fi + xi * fr], dim=-1).reshape(len(positions), D_ROPE)
    return torch.cat([y[:, :-D_ROPE], rot], dim=-1)
