#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The pair protocols' shared parts (public): the pair statistic and its positions, reads on token ids,
seed canvases and the prefix-cache hit counter."""
from __future__ import annotations

import math
import random

from . import reads as R

EOS = (1, 106, 50)


def cut(targets) -> int:
    """The statistic's positions for a canvas generated here: up to and including its first end-of-sequence token."""
    for i, t in enumerate(targets):
        if t in EOS:
            return i + 1
    return len(targets)


def pair(a: dict, b: dict, upto: int) -> float:
    """The pair statistic between two readings of one read (mean k3 over the first `upto` positions)."""
    if not (a["structural"] and b["structural"]):
        return math.inf
    return R.k3(a["target_logprob"][:upto], b["target_logprob"][:upto])


def read(srv, prompt_ids, seed, targets, steps=1) -> dict:
    return R.parse(srv.post("/v1/completions", R.text_read(prompt_ids, seed, targets, steps), timeout=1800), targets)


def seed_of(targets, phi, key) -> list[int]:
    """A seed canvas: a seeded fraction phi of the targets replaced by uniform vocabulary draws."""
    rng = random.Random(key)
    out = list(targets)
    for i in rng.sample(range(len(out)), int(round(phi * len(out)))):
        out[i] = rng.randrange(262144)
    return out


def prefix_hits(srv) -> float:
    return srv.metric("vllm:prefix_cache_hits_total") or srv.metric("vllm:prefix_cache_hits")
