#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Independent packed GPT-OSS expert mathematics, with a streamed FP32 reference."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch

HIDDEN = 2880
INTERMEDIATE = 2880
EXPERTS = 128
TOP_K = 4
MAX_TOKENS = 4096
ROUTE_SUM_TOLERANCE = 8 * torch.finfo(torch.float32).eps


class AuthorDataFault(ValueError):
    """An expert input or reference result outside the admitted operation domain."""


@dataclass
class Weights:
    gate_up: torch.Tensor
    gate_up_scale: torch.Tensor
    down: torch.Tensor
    down_scale: torch.Tensor
    gate_up_bias: torch.Tensor
    down_bias: torch.Tensor


def unpack(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("packed values and E8M0 scales must be uint8")
    if blocks.shape[:-1] != scales.shape[:-1] or blocks.shape[-1] != scales.shape[-1] * 16:
        raise ValueError("one scale is required for every 32 logical values")
    if (scales == 255).any():
        raise AuthorDataFault("E8M0 scale byte 255 (NaN) is outside the finite contract")
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                         device=blocks.device, dtype=torch.float32)
    codes = torch.stack((blocks & 15, blocks >> 4), dim=-1).flatten(-2)
    values = table[codes.long()]
    exponent = scales.to(torch.int32) - 127
    factors = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    return values * factors.repeat_interleave(32, dim=-1)


def activation(z: torch.Tensor) -> torch.Tensor:
    gate = z[..., 0::2].clamp(max=7)
    up = z[..., 1::2].clamp(min=-7, max=7)
    return gate * torch.sigmoid(1.702 * gate) * (up + 1)


def routes(logits: torch.Tensor, count=TOP_K) -> tuple[torch.Tensor, torch.Tensor]:
    selected, ids = logits.float().topk(count, dim=-1)
    return ids, selected.softmax(dim=-1)


def _finite(tensor, what):
    if not torch.isfinite(tensor).all():
        raise AuthorDataFault(f"{what} is not finite; the case is outside the admitted domain")


def reference(x, ids, probabilities, weights: Weights, *, intermediate_dtype=None):
    """Stream one expert at a time; full logical output includes every TP tail.

    The mathematical default uses FP32 throughout. A separately reported cast
    comparison can set intermediate_dtype to the native activation dtype; it
    must not substitute for reporting the mathematical comparison. Every decoded
    weight, intermediate and result must be finite (`AuthorDataFault` otherwise).
    """
    if x.ndim != 2 or ids.shape != probabilities.shape or ids.shape[0] != x.shape[0]:
        raise ValueError("invalid expert input shapes")
    if not torch.isfinite(x).all() or not torch.isfinite(probabilities).all():
        raise AuthorDataFault("expert inputs must be finite")
    if (probabilities < 0).any() or not torch.allclose(probabilities.sum(-1),
                                                    torch.ones(x.shape[0], device=x.device)):
        raise AuthorDataFault("selected-route weights must be normalized and nonnegative")
    if (ids < 0).any() or (ids >= weights.gate_up.shape[0]).any():
        raise AuthorDataFault("expert ID out of range")
    if ids.shape[1] > 1 and (ids.sort(-1).values[:, 1:] == ids.sort(-1).values[:, :-1]).any():
        raise AuthorDataFault("native top-k selects distinct experts within a token")
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in ids.unique().tolist():
        tokens, slots = torch.where(ids == expert)
        w1 = unpack(weights.gate_up[expert].to(x.device), weights.gate_up_scale[expert].to(x.device))
        w2 = unpack(weights.down[expert].to(x.device), weights.down_scale[expert].to(x.device))
        _finite(w1, f"expert {expert} decoded gate/up weights")
        _finite(w2, f"expert {expert} decoded down weights")
        z = x[tokens].float() @ w1.T + weights.gate_up_bias[expert].to(x.device)
        _finite(z, f"expert {expert} gate/up projection")
        h = activation(z)
        if intermediate_dtype is not None:
            h = h.to(intermediate_dtype).float()
            _finite(h, f"expert {expert} activation at its declared {intermediate_dtype} boundary")
        y = h @ w2.T + weights.down_bias[expert].to(x.device)
        _finite(y, f"expert {expert} down projection")
        out.index_add_(0, tokens, y * probabilities[tokens, slots, None])
    _finite(out, "the complete expert output")
    return out


def admit(x, ids, probabilities, weights: Weights) -> dict:
    """Check that one complete expert invocation lies in the admitted operation domain."""
    shapes = {"gate_up": (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
              "gate_up_scale": (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32),
              "down": (EXPERTS, HIDDEN, INTERMEDIATE // 2),
              "down_scale": (EXPERTS, HIDDEN, INTERMEDIATE // 32)}
    for name, shape in shapes.items():
        tensor = getattr(weights, name)
        if tensor.dtype != torch.uint8 or tuple(tensor.shape) != shape:
            raise AuthorDataFault(f"{name} must be uint8 {shape}, not {tensor.dtype} {tuple(tensor.shape)}")
    for name in ("gate_up_scale", "down_scale"):
        if (getattr(weights, name) == 255).any():
            raise AuthorDataFault(f"{name} contains the invalid E8M0 byte 255")
    for name, shape in (("gate_up_bias", (EXPERTS, 2 * INTERMEDIATE)), ("down_bias", (EXPERTS, HIDDEN))):
        bias = getattr(weights, name)
        if bias.dtype != torch.float32 or tuple(bias.shape) != shape:
            raise AuthorDataFault(f"{name} must be float32 {shape}, not {bias.dtype} {tuple(bias.shape)}")
        _finite(bias, name)
        if not torch.equal(bias, bias.bfloat16().float()):
            raise AuthorDataFault(f"{name} holds values that are not BF16-representable")
    if x.ndim != 2 or x.shape[1] != HIDDEN or not 1 <= x.shape[0] <= MAX_TOKENS:
        raise AuthorDataFault(f"activations must be [1..{MAX_TOKENS}, {HIDDEN}], not {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise AuthorDataFault(f"activations must be BF16, not {x.dtype}")
    _finite(x, "activations")
    if tuple(ids.shape) != (x.shape[0], TOP_K) or ids.dtype.is_floating_point or ids.dtype == torch.bool:
        raise AuthorDataFault(f"routes must be integer [{x.shape[0]}, {TOP_K}], not {ids.dtype} {tuple(ids.shape)}")
    if (ids < 0).any() or (ids >= EXPERTS).any():
        raise AuthorDataFault("route IDs must name one of the 128 experts")
    ordered = ids.sort(-1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise AuthorDataFault("the four routed experts of a token must be distinct")
    if tuple(probabilities.shape) != tuple(ids.shape) or probabilities.dtype != torch.float32:
        raise AuthorDataFault("route probabilities must be float32 with one value per route")
    _finite(probabilities, "route probabilities")
    if (probabilities < 0).any():
        raise AuthorDataFault("route probabilities must be nonnegative")
    deviation = (probabilities.double().sum(-1) - 1).abs().max().item()
    if deviation > ROUTE_SUM_TOLERANCE:
        raise AuthorDataFault(f"route probabilities must sum to one; the largest deviation is {deviation:.3g}")
    return {"tokens": int(x.shape[0]), "route_sum_deviation": deviation,
            "experts_used": int(ids.unique().numel())}


def checkpoint_weights(model_path, layer):
    from safetensors import safe_open
    root = Path(model_path)
    mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    names = ["gate_up_proj_blocks", "gate_up_proj_scales", "down_proj_blocks",
             "down_proj_scales", "gate_up_proj_bias", "down_proj_bias"]
    values = []
    for name in names:
        key = f"model.layers.{layer}.mlp.experts.{name}"
        with safe_open(root / mapping[key], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
        if name.endswith("blocks"):
            tensor = tensor.flatten(-2)
        elif name.endswith("bias"):
            tensor = tensor.float()
        values.append(tensor)
    return Weights(*values)


def checkpoint_router(model_path, layer):
    """The released router of one layer: weight [experts, hidden] and bias [experts]."""
    from safetensors import safe_open
    root = Path(model_path)
    mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = []
    for name in ("weight", "bias"):
        key = f"model.layers.{layer}.mlp.router.{name}"
        with safe_open(root / mapping[key], framework="pt", device="cpu") as handle:
            tensors.append(handle.get_tensor(key))
    return tensors


def example(model_path="/model", layer=0, tokens=8, seed=271828, device="cpu"):
    """A small checkpoint-backed case: the released layer's experts and router, BF16
    activations, and the router's own normalized top-4 routes."""
    weights = checkpoint_weights(model_path, layer)
    weight, bias = checkpoint_router(model_path, layer)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16)
    logits = x.float() @ weight.to(device).float().T + bias.to(device).float()
    ids, probabilities = routes(logits)
    return weights, x, ids, probabilities
