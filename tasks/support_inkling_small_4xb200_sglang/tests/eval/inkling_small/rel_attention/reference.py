#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Plain-PyTorch reference for the relative-attention operator contract (README.md)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

PAGE = 128
CHUNK = 128


@dataclass
class Case:
    """One call of the contract: its inputs and keyword arguments (README.md, 'Entry point')."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    rel_bias: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    window_size: tuple = (None, None)
    softmax_scale: float = 1.0 / 128
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_k: int | None = None
    page_table: torch.Tensor | None = None
    seqused_k: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)

    @property
    def paged(self) -> bool:
        return self.page_table is not None

    def requests(self):
        """Per request: (query rows [lo, hi), key length k_len)."""
        cq = self.cu_seqlens_q.tolist()
        if self.paged:
            kl = self.seqused_k.tolist()
        else:
            ck = self.cu_seqlens_k.tolist()
            kl = [ck[b + 1] - ck[b] for b in range(len(cq) - 1)]
        return [(cq[b], cq[b + 1], kl[b]) for b in range(len(cq) - 1)]

    def keys(self, b: int, lo: int, hi: int):
        """Keys and values of request b at key positions [lo, hi), shaped [n, 2, 128]."""
        if self.paged:
            p0, p1 = lo // PAGE, (hi - 1) // PAGE + 1
            pages = self.page_table[b, p0:p1].long()
            kk = self.k[pages].reshape(-1, *self.k.shape[2:])
            vv = self.v[pages].reshape(-1, *self.v.shape[2:])
            off = lo - p0 * PAGE
            return kk[off: off + hi - lo], vv[off: off + hi - lo]
        base = int(self.cu_seqlens_k[b])
        return self.k[base + lo: base + hi], self.v[base + lo: base + hi]


def window(case: Case) -> int | None:
    left = case.window_size[0]
    return None if left is None or left < 0 else int(left)


def _chunk_logits(case: Case, b: int, t0: int, t1: int, p0: int, k_len: int, dtype, bf16_pipeline: bool):
    """Logits [8, n_q, n_k] of queries rows t0..t1 (key positions p0..) with their gathered keys."""
    W = window(case)
    n_q = t1 - t0
    p_last = p0 + n_q - 1
    j_lo = 0 if W is None else max(0, p0 - W)
    j_hi = p_last + 1
    kk, vv = case.keys(b, j_lo, j_hi)
    h_q, h_kv = case.q.shape[1], kk.shape[1]
    g = h_q // h_kv
    q = case.q[t0:t1]
    kk = kk.repeat_interleave(g, dim=1)
    vv = vv.repeat_interleave(g, dim=1)
    pos = torch.arange(p0, p0 + n_q, device=q.device)
    jj = torch.arange(j_lo, j_hi, device=q.device)
    d = pos[:, None] - jj[None, :]
    allowed = d >= 0
    if W is not None:
        allowed &= d <= W
    E = case.rel_bias.shape[-1]
    idx = d.clamp(0, E - 1)
    bias = case.rel_bias[t0:t1]
    gathered = torch.gather(bias.permute(1, 0, 2), 2, idx.unsqueeze(0).expand(h_q, -1, -1))
    in_extent = (d >= 0) & (d < E)
    if bf16_pipeline:
        s = torch.einsum("thd,shd->hts", q, kk * case.softmax_scale)
        s = s.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        s = s + torch.where(in_extent.unsqueeze(0), gathered, torch.zeros_like(gathered))
        return s, vv, allowed
    qf, kf = q.to(dtype), kk.to(dtype)
    s = torch.einsum("thd,shd->hts", qf, kf) * case.softmax_scale
    s = s + torch.where(in_extent.unsqueeze(0), gathered.to(dtype), torch.zeros((), dtype=dtype, device=q.device))
    s = s.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    return s, vv.to(dtype), allowed


@torch.no_grad()
def reference(case: Case, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The contract's law, exactly, in `dtype`: returns [total_q, 8, 128]."""
    out = torch.empty(case.q.shape[0], case.q.shape[1], case.v.shape[-1], dtype=dtype, device=case.q.device)
    for b, (lo, hi, k_len) in enumerate(case.requests()):
        q_len = hi - lo
        for c0 in range(0, q_len, CHUNK):
            c1 = min(q_len, c0 + CHUNK)
            s, vv, _ = _chunk_logits(case, b, lo + c0, lo + c1, k_len - q_len + c0, k_len, dtype, False)
            p = torch.softmax(s, dim=-1)
            out[lo + c0: lo + c1] = torch.einsum("hts,shd->thd", p, vv)
    return out


@torch.no_grad()
def reference_bf16(case: Case) -> torch.Tensor:
    """The same attention with FlashAttention-4's test-pipeline operand precisions: bf16 [total_q, 8, 128]."""
    out = torch.empty(case.q.shape[0], case.q.shape[1], case.v.shape[-1], dtype=torch.bfloat16, device=case.q.device)
    for b, (lo, hi, k_len) in enumerate(case.requests()):
        q_len = hi - lo
        for c0 in range(0, q_len, CHUNK):
            c1 = min(q_len, c0 + CHUNK)
            s, vv, _ = _chunk_logits(case, b, lo + c0, lo + c1, k_len - q_len + c0, k_len, None, True)
            p = torch.softmax(s, dim=-1).to(torch.bfloat16)
            out[lo + c0: lo + c1] = torch.einsum("hts,shd->thd", p, vv)
    return out


def valid_rows(case: Case) -> torch.Tensor:
    """Boolean [total_q]: every query row of a request with at least one query is valid."""
    m = torch.zeros(case.q.shape[0], dtype=torch.bool, device=case.q.device)
    for lo, hi, _ in case.requests():
        m[lo:hi] = True
    return m
