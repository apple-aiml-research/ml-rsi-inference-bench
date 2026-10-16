#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Torch implementations of the FP8 references (W8A8 emulation, W8A16 reference, act-quant signature).

Tensor references for block-scaled E4M3 weights and per-token activation groups.
The verifier evaluates dequantized operands in float32 or float64, independently
of the candidate's quantized kernel.

  dequant_block(wq, scale)              float32 weight, W ~= decode(wq) * scale per 128x128 block
  quant_act_groups(x)                   E4M3 codes (torch.float8_e4m3fn) + float32 scales, per token per 128 group
  dequant_act_groups(xq, s)             float32 activations
  linear_w8a8(x, wq, scale, bias)       the W8A8 emulation, float32 (or float64) matmul of dequantised operands
  linear_w8a16(x, wq, scale, bias)      FP8 storage, full-precision activations
  act_quant_signature(y, y_w8a16, y_w8a8)  projection onto the activation-quantization delta
"""
from __future__ import annotations

import torch

FP8_MAX = 448.0
BLOCK = 128
ACT_EPS = 1e-10


def dequant_block(wq: torch.Tensor, scale: torch.Tensor, block: int = BLOCK, dtype=torch.float32) -> torch.Tensor:
    n, k = wq.shape
    s = scale.to(torch.float32).to(dtype).repeat_interleave(block, 0).repeat_interleave(block, 1)[:n, :k]
    return wq.to(torch.float32).to(dtype) * s


def quant_act_groups(x: torch.Tensor, block: int = BLOCK, eps: float = ACT_EPS):
    """x [..., K] -> (codes float8_e4m3fn [..., K], scales float32 [..., ceil(K/block)])."""
    shp = x.shape
    k = shp[-1]
    g = -(-k // block)
    x2 = x.reshape(-1, k).to(torch.float32)
    if g * block != k:
        x2 = torch.nn.functional.pad(x2, (0, g * block - k))
    xg = x2.view(x2.shape[0], g, block)
    amax = xg.abs().amax(dim=2).clamp(min=eps)
    s = amax / FP8_MAX
    q = (xg / s[:, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q = q.view(x2.shape[0], g * block)[:, :k]
    return q.reshape(shp), s.reshape(*shp[:-1], g)


def dequant_act_groups(q: torch.Tensor, s: torch.Tensor, block: int = BLOCK, dtype=torch.float32) -> torch.Tensor:
    k = q.shape[-1]
    return q.to(torch.float32).to(dtype) * s.to(dtype).repeat_interleave(block, -1)[..., :k]


def linear_w8a8(x, wq, scale, bias=None, block: int = BLOCK, dtype=torch.float32, w_deq=None):
    q, s = quant_act_groups(x, block)
    xd = dequant_act_groups(q, s, block, dtype)
    w = w_deq if w_deq is not None else dequant_block(wq, scale, block, dtype)
    y = xd @ w.t()
    if bias is not None:
        y = y + bias.to(dtype)
    return y


def linear_w8a16(x, wq, scale, bias=None, block: int = BLOCK, dtype=torch.float32, w_deq=None):
    w = w_deq if w_deq is not None else dequant_block(wq, scale, block, dtype)
    y = x.to(dtype) @ w.t()
    if bias is not None:
        y = y + bias.to(dtype)
    return y


def act_quant_signature(y_cand, y_w8a16, y_w8a8) -> dict:
    """Project the candidate delta from W8A16 onto W8A8-W8A16 in float64.

    Return the projection coefficient and residual norm relative to that reference delta.
    """
    c = y_cand.detach().double().flatten()
    a = y_w8a16.detach().double().flatten()
    b = y_w8a8.detach().double().flatten()
    d = b - a
    dd = float(d @ d)
    if dd == 0.0:
        raise ValueError("act_quant_signature: the W8A8 and W8A16 references coincide")
    e = c - a
    beta = float(e @ d) / dd
    resid = float(torch.linalg.vector_norm(e - beta * d)) / dd ** 0.5
    return {"beta": beta, "orthogonal_residual_over_d": resid, "d_rel": dd ** 0.5 / float(torch.linalg.vector_norm(a))}


def rel_l2(a, b) -> float:
    a = a.detach().double().flatten()
    b = b.detach().double().flatten()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))
