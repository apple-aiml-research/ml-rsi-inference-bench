#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent QSA mathematics over one request's logical history.

Adapted from the vLLM project's independent QSA tensor reference tests (initial model support),
rewritten over logical sequences: every function takes a request's tokens in order and no page
table, cache layout, metadata object or selection buffer. It imports no engine code and is slow on
purpose. Grading's enclosing-region comparator applies these functions to the checkpoint's own
weights; the numerical allowances below are those of `scope.json`.

The layer, for a query row at position p of a request whose history is rows 0..p:

* queries: q (per head, gemma-normed, NeoX rope on the rotary prefix) and the output gate come from
  q_proj; index queries from the first part of index_qk_proj, gemma-normed and roped;
* keys and values: k_proj (gemma-normed, roped) and v_proj of every history row; raw index keys
  from the second part of index_qk_proj;
* compressed index keys: the mean over each complete group of `ratio` consecutive raw index keys,
  gemma-normed and roped at the group's first position; a row sees the groups completed at or
  before it, (p + 1) // ratio of them;
* selection: of the visible groups, the `budget // ratio` with the highest index score
  (relu of the per-head dot product, summed over heads, divided by sqrt(index head dim)); the row
  attends to the tokens of its selected groups and to the open causal tail after the last complete
  group. Order does not matter, and near ties are admissible within `SELECTION_ALLOWANCE`
  (`admissible`): no selection buffer or ordering is part of the contract;
* output: softmax attention per head (grouped-query heads share their KV head), rounded to BF16,
  times sigmoid(gate), through o_proj.
"""

import math

import torch

OUTPUT_ALLOWANCE = {"atol": 0.02, "rtol": 0.016}
SELECTION_ALLOWANCE = {"atol": 0.05, "rtol": 0.016}


def gemma_norm(x, weight, eps):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (value * (1 + weight.float())).to(x.dtype)


def rope_tables(positions, *, head_dim, partial_rotary_factor, rope_theta):
    """cos/sin over the rotary dimensions at text positions, rounded like a BF16 model cache."""
    rotary = int(head_dim * partial_rotary_factor)
    inv = 1.0 / (rope_theta ** (torch.arange(0, rotary, 2, dtype=torch.float32, device=positions.device) / rotary))
    freqs = positions.to(torch.float32)[:, None] * inv[None, :]
    return freqs.cos().to(torch.bfloat16).float(), freqs.sin().to(torch.bfloat16).float()


def rope(x, cos, sin):
    """NeoX half rotation of the leading rotary dimensions; the remainder is unchanged."""
    half = cos.shape[-1]
    while cos.ndim < x.ndim:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    left, right = x[..., :half].float(), x[..., half:2 * half].float()
    rotated = torch.cat((left * cos - right * sin, right * cos + left * sin), -1)
    return torch.cat((rotated.to(x.dtype), x[..., 2 * half:]), -1)


def compressed_keys(raw_index_keys, weight, *, ratio, eps, tables):
    """Compressed index keys of every complete group of `ratio` raw index keys (history order).

    `tables(positions)` returns the rope cos/sin at the given positions."""
    complete = raw_index_keys.shape[0] // ratio
    width = raw_index_keys.shape[-1]
    pooled = raw_index_keys[:complete * ratio].float().view(complete, ratio, width).mean(1).to(raw_index_keys.dtype)
    cos, sin = tables(torch.arange(complete, device=raw_index_keys.device) * ratio)
    return rope(gemma_norm(pooled, weight, eps), cos, sin)


def visible_groups(position, ratio):
    """Complete compressed groups a row at `position` sees."""
    return (position + 1) // ratio


def index_scores(index_query, compressed, visible):
    """One row's score for each visible group: relu(q_h · k) summed over index heads / sqrt(d)."""
    return torch.relu(torch.einsum("hd,gd->gh", index_query.float(), compressed[:visible].float())).sum(-1) \
        / math.sqrt(index_query.shape[-1])


def select(scores, *, budget_groups):
    """This reference's own top-k selection of groups (one admissible selection; order illustrative)."""
    want = min(budget_groups, scores.shape[0])
    return torch.topk(scores, want).indices if want else torch.empty(0, dtype=torch.long, device=scores.device)


def admissible(scores, selected, allowance=SELECTION_ALLOWANCE):
    """(bool, margin): the weakest selected group scores at least the strongest omitted one minus
    atol + rtol·|strongest omitted|, and exactly min(budget, visible) groups are selected."""
    chosen = torch.zeros_like(scores, dtype=torch.bool)
    chosen[selected] = True
    if bool(chosen.all()) or not bool(chosen.any()):
        return True, None
    weakest = float(scores[chosen].min())
    strongest = float(scores[~chosen].max())
    margin = weakest - strongest
    return margin >= -(allowance["atol"] + allowance["rtol"] * abs(strongest)), margin


def attended_tokens(groups, position, ratio):
    """Token indices a row attends to: its selected groups' tokens and the open causal tail."""
    span = torch.arange(ratio, device=groups.device)
    chosen = (groups[:, None] * ratio + span).flatten()
    tail = torch.arange(visible_groups(position, ratio) * ratio, position + 1, device=groups.device)
    return torch.cat((chosen, tail))


def attend(query, keys, values, tokens, *, scale):
    """Softmax attention of one row's heads over `tokens` of its history (float32, per head)."""
    heads, kv = query.shape[0], keys.shape[1]
    kk = keys[tokens].float().repeat_interleave(heads // kv, dim=1)
    vv = values[tokens].float().repeat_interleave(heads // kv, dim=1)
    weights = torch.softmax(torch.einsum("hd,shd->hs", query.float(), kk) * scale, -1)
    return torch.einsum("hs,shd->hd", weights, vv)


def gated_output(attended, gate, o_weight):
    """Rows' attention outputs rounded to BF16, times sigmoid(gate), through o_proj."""
    flat = attended.to(torch.bfloat16).flatten(1)
    return torch.nn.functional.linear(flat * torch.sigmoid(gate.flatten(1)), o_weight)
