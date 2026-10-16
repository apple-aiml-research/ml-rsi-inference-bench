#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent PLE mathematics over one request's logical history.

The n-gram reference is adapted from the vLLM project's independent tensor tests (initial model
support); gating and the convolution express the same mathematical and state contract in small
sequential operations over logical sequences. No page, slot, metadata object or physical state
layout appears here; state is the request's committed history. It imports no engine code and is
slow on purpose. The numerical allowance below is that of `scope.json`.

For a row at position p of a request (tokens t_0..t_p committed or current):

* n-gram rows: for each order 2..ngram_size, the token and its preceding order-1 tokens (EOS for
  positions before the request's start; a shift never crosses an EOS: once an EOS is met, earlier
  positions read as EOS) are mixed as XOR of token·multiplier in two's-complement int64, and each
  of the order's heads reads row (mixed mod head vocabulary size) + head offset of the table;
* embedding: those table rows, dequantized (FP8 value times the per-tensor scale, rounded to
  BF16), concatenated over heads;
* gating: key = grouped_norm(key_proj(embedding)), query = grouped_norm(hidden) per HC stream;
  gate = sigmoid(sign(d)·sqrt(max(|d|, 1e-6))) with d = key·query / sqrt(hidden) rounded to the
  hidden dtype; gated = gate · value_proj(embedding) per stream;
* convolution: conv_in = grouped_norm(gated); a depthwise causal convolution of kernel K and
  dilation D over each channel's conv_in history (zeros before the request's start), float32
  accumulation, rounded to the hidden dtype; output = gated + silu(convolution).

Rejected speculative rows are not history: only committed rows feed later n-grams and convolutions.
"""

import math

import torch
from torch.nn import functional as F

OUTPUT_ALLOWANCE = {"atol": 0.01, "rtol": 0.016}


def int64(value: int) -> int:
    """Two's-complement int64 wraparound, as the engine's int64 tensor arithmetic."""
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= 1 << 63 else value


def ngram_rows(context, tokens, *, multipliers, sizes, offsets, heads_per_ngram, context_len, eos):
    """Hashed n-gram table rows for `tokens`, given the `context_len` committed tokens before them
    (EOS where the request has none); shifts do not cross an EOS."""
    history = list(context) + list(tokens)
    rows = []
    for pos in range(len(context), len(history)):
        shifted, crossed = [history[pos]], False
        for shift in range(1, context_len + 1):
            token = eos if crossed else history[pos - shift]
            shifted.append(token)
            crossed |= token == eos
        row = []
        mixed = shifted[0] * int(multipliers[0])
        for order in range(2, context_len + 2):
            mixed ^= shifted[order - 1] * int(multipliers[order - 1])
            start = (order - 2) * heads_per_ngram
            for head in range(start, start + heads_per_ngram):
                row.append(int64(mixed) % int(sizes[head]) + int(offsets[head]))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


def dequantize(rows, scale):
    """FP8 table rows times the per-tensor scale, rounded to BF16."""
    return (rows.float() * scale.float()).to(torch.bfloat16)


def grouped_norm(x, weight, groups, eps):
    grouped = x.float().unflatten(-1, (groups, x.shape[-1] // groups))
    value = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    return (value.flatten(-2) * (1 + weight.float())).to(x.dtype)


def gate(key, value, hidden, *, norm_key, norm_query, norm_conv, streams, eps):
    """(gated, conv_in) of rows: key/value from the embedding's projections, hidden per HC stream."""
    width = hidden.shape[-1] // streams
    k = grouped_norm(key, norm_key, streams, eps).unflatten(-1, (streams, width))
    q = grouped_norm(hidden, norm_query, streams, eps).unflatten(-1, (streams, width))
    dot = ((k * q).sum(-1, keepdim=True) / math.sqrt(width)).to(hidden.dtype)
    gating = torch.sigmoid(dot.sign() * dot.abs().clamp_min(1e-6).sqrt()).to(hidden.dtype)
    gated = (gating * value.unsqueeze(-2)).flatten(-2)
    return gated, grouped_norm(gated, norm_conv, streams, eps)


def convolution_output(gated, conv_in, history_conv, weight, *, dilation):
    """Rows' outputs given the committed conv inputs before them (`history_conv`, oldest first,
    at least (K - 1)·D rows; zeros before the request's start). `weight` is (channels, K)."""
    kernel = weight.shape[-1]
    full = torch.cat((history_conv.to(conv_in.dtype), conv_in), 0)
    start = history_conv.shape[0]
    out = gated.clone()
    w = weight.float()
    for j in range(conv_in.shape[0]):
        p = start + j
        window = torch.stack([full[p - dilation * (kernel - 1 - t)] for t in range(kernel)], -1)
        conv = (window.float() * w).sum(-1).to(conv_in.dtype)
        out[j] = out[j] + F.silu(conv)
    return out
