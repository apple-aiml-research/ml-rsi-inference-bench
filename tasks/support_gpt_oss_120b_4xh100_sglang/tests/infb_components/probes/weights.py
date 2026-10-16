#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private expert weight constructors for the held-out numerical cases.

Every case must lie in the admitted operation domain (`gpt_oss.expert_math.admit`);
a constructor never clamps or drops a case to make it admissible.
"""

import torch

from gpt_oss.expert_math import Weights


def generated_weights(hidden=2880, intermediate=2880, experts=128, seed=271828):
    """Finite data; every nibble, scale block and logical tail participates."""
    if hidden % 32 or intermediate % 32:
        raise ValueError("logical dimensions must contain complete quantization blocks")
    g = torch.Generator().manual_seed(seed)
    def packed(shape):
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
    def scales(shape):
        return torch.randint(119, 124, shape, generator=g, dtype=torch.uint8)
    return Weights(
        packed((experts, 2*intermediate, hidden//2)),
        scales((experts, 2*intermediate, hidden//32)),
        packed((experts, hidden, intermediate//2)),
        scales((experts, hidden, intermediate//32)),
        (torch.randn(experts, 2*intermediate, generator=g) * 3).bfloat16().float(),
        (torch.randn(experts, hidden, generator=g) * .25).bfloat16().float(),
    )


def diagnostic_weights(family, hidden=2880, intermediate=2880, experts=128, seed=271828):
    if family == "generated":
        return generated_weights(hidden, intermediate, experts, seed)
    w = Weights(torch.zeros(experts, 2*intermediate, hidden//2, dtype=torch.uint8),
                torch.full((experts, 2*intermediate, hidden//32), 127, dtype=torch.uint8),
                torch.zeros(experts, hidden, intermediate//2, dtype=torch.uint8),
                torch.full((experts, hidden, intermediate//32), 127, dtype=torch.uint8),
                torch.zeros(experts, 2*intermediate), torch.zeros(experts, hidden))
    if family in ("negative_gate", "positive_clamps", "plus_one"):
        # Constant gate/up values isolate activation semantics from W1 rounding.
        gate, up = {"negative_gate": (-10, 8), "positive_clamps": (10, -10),
                    "plus_one": (1, -1)}[family]
        w.gate_up_bias[:, 0::2] = gate
        w.gate_up_bias[:, 1::2] = up
        w.down.fill_(0x22)  # two +1 E2M1 values, low nibble first
    elif family == "tp_blocks":
        w.gate_up_bias.fill_(1)
        # Every logical quantization block has its own output channel. Losing
        # a TP tail or crossing a scale boundary cannot cancel in an aggregate.
        for block in range(intermediate // 32):
            w.down[:, block, block*16:(block+1)*16] = 0x22
    elif family == "bias":
        w.down_bias.copy_((torch.arange(experts)[:, None] % 5 - 2).expand(-1, hidden) * .25)
    else:
        raise ValueError(f"unknown expert weight family {family}")
    return w
