#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Identity sets of an `attend` implementation (the attribution rule of README.md), public."""
from __future__ import annotations

import re

import torch

EXCLUDE = re.compile(r"memcpy|memset|Memcpy|Memset|copy_|CopyKernel|fill_|FillFunctor|elementwise_kernel.*(copy|fill)"
                     r"|at::native::", re.I)


def kernels_of(fn) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = set()
    for ev in prof.events():
        if getattr(ev, "device_type", None) is not None and str(ev.device_type).endswith("CUDA"):
            if not EXCLUDE.search(ev.name):
                names.add(ev.name)
    return sorted(names)


def identity_sets(calls: dict) -> dict:
    """calls: {(family, layer_kind): zero-argument callable} -> {family: {layer_kind: [kernel names]}}."""
    out: dict = {}
    for (family, layer_kind), fn in calls.items():
        out.setdefault(family, {})[layer_kind] = kernels_of(fn)
    return out


def exclusions() -> str:
    """The exclusion rule, as the identity record states it."""
    return EXCLUDE.pattern
