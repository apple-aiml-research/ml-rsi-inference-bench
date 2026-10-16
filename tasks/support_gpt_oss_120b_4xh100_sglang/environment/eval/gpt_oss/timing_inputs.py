#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Inputs for the declared complete-expert timing workloads: balanced and skewed routing."""

import torch

FAMILIES = ("balanced", "skewed")


def inputs(count, hidden, experts, family, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(count, hidden, generator=g, device="cuda", dtype=torch.bfloat16)
    if family == "balanced":
        ids = (torch.arange(count * 4, device="cuda").reshape(count, 4) + seed) % experts
    elif family == "skewed":
        ids = torch.stack((torch.zeros(count, device="cuda", dtype=torch.int64),
                           1 + torch.arange(count, device="cuda") % (experts-3),
                           torch.full((count,), experts-2, device="cuda", dtype=torch.int64),
                           torch.full((count,), experts-1, device="cuda", dtype=torch.int64)), dim=-1)
    else:
        raise ValueError(f"unknown timing routing family {family}")
    weights = torch.randn(count, 4, generator=g, device="cuda").softmax(-1)
    return x, ids, weights
