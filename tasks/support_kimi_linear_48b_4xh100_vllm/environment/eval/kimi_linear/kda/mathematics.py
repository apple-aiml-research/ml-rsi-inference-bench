#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Independent, serial PyTorch equations; no optimized KDA or engine imports."""

import torch
from torch.nn import functional as F

from recurrence import naive_recurrent_kda


def normalize_qk(x):
    """Release epsilon and rounding: FP32 sum/rsqrt, then activation dtype."""
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)


def decay_gate(raw, a_log, dt_bias):
    return -a_log.float().reshape(1, -1, 1).exp() * F.softplus(
        raw.float() + dt_bias.float().reshape(1, raw.shape[-2], raw.shape[-1]),
        beta=1.0, threshold=20.0,
    )


def gated_rmsnorm(x, gate, weight, eps=1e-5):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
            * weight.float() * gate.float().sigmoid()).to(x.dtype)


def packed_recurrence(q, k, v, g, beta, starts, initial_state):
    """[T,H,D] packed inputs and [N,H,K,V] FP32 states; both results are independent."""
    outputs, states = [], []
    q, k = normalize_qk(q), normalize_qk(k)
    for i, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        out, state = naive_recurrent_kda(
            q[None, start:end], k[None, start:end], v[None, start:end],
            g[None, start:end], beta[None, start:end],
            initial_state=initial_state[i:i + 1], output_final_state=True,
        )
        outputs.append(out[0])
        states.append(state[0])
    return torch.cat(outputs), torch.stack(states)


def convolution(x, weight, initial, starts):
    """Depthwise causal width-four convolution, SiLU, BF16 result and raw input history.

    Adapted from the original engine test's independent causal_conv1d_ref. State uses
    [N,3,H*D], the engine cache orientation; arithmetic is ordinary PyTorch FP32.
    """
    weight = weight.reshape(x.shape[-1], -1).float()
    out, states = [], []
    for i, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        joined = torch.cat((initial[i], x[start:end]), dim=0)
        y = F.conv1d(joined.T[None].float(), weight[:, None], groups=x.shape[-1])
        out.append(F.silu(y[0].T).to(x.dtype))
        states.append(joined[-(weight.shape[-1] - 1):].clone())
    return torch.cat(out), torch.stack(states)


@torch.inference_mode()
def layer_reference(hidden, weights, caches, starts, indices, has_initial, heads=8, dim=128, eps=1e-5):
    """Complete native KDA layer before TP reduction; output plus all four updated caches.

    weights is a semantic tensor mapping, not candidate callables. The enclosing comparison
    includes projections and convolution so fusion can cross the attention boundary.
    Caller performs an independent TP sum on the local projected output.
    """
    new = [x.clone() for x in caches]
    selected = [x[indices].clone() for x in caches]
    for state in selected:
        state[~has_initial] = 0
    qkv = []
    for i, name in enumerate(('q', 'k', 'v')):
        projected = F.linear(hidden, weights[name + '_proj'])
        y, state = convolution(projected, weights[name + '_conv'], selected[i], starts)
        qkv.append(y.reshape(-1, heads, dim))
        new[i][indices] = state
    beta = F.linear(hidden, weights['beta_proj']).float().sigmoid()
    raw = F.linear(F.linear(hidden, weights['decay_a']), weights['decay_b']).reshape(-1, heads, dim)
    gate = decay_gate(raw, weights['a_log'], weights['dt_bias'])
    attention, state = packed_recurrence(*qkv, gate, beta, starts, selected[3])
    new[3][indices] = state
    out_gate = F.linear(F.linear(hidden, weights['gate_a']), weights['gate_b']).reshape(-1, heads, dim)
    normalized = gated_rmsnorm(attention, out_gate, weights['norm'], eps)
    projected = F.linear(normalized.flatten(1), weights['out_proj'])
    return projected, new, attention


def error(actual, expected):
    """Finite checks precede reductions and the absolute-error escape."""
    if actual.shape != expected.shape:
        raise AssertionError(f'shape {tuple(actual.shape)} != {tuple(expected.shape)}')
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise AssertionError('non-finite expected or actual tensor')
    delta = actual.float() - expected.float()
    rmse = delta.square().mean().sqrt()
    scale = expected.float().square().mean().sqrt()
    return {'max_abs': float(delta.abs().max()), 'relative_rmse': float(rmse / (scale + 1e-8))}
