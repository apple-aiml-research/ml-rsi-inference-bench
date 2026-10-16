#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'The independent dense-delta reference for SD3 transformer LoRA, and the request-lifecycle model.'
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safetensors_np as stn  # noqa: E402

PREFIX = "transformer."
METADATA_KEY = "lora_adapter_metadata"
_KEY_RE = re.compile(r"^(?P<mod>.+)\.lora_(?P<ab>[AB])\.weight$")


@dataclass
class ModuleLoRA:
    A: np.ndarray
    B: np.ndarray
    r: int
    alpha: float
    rslora: bool = False
    alpha_default: float | None = None

    @property
    def scaling(self) -> float:
        return self.alpha / (float(np.sqrt(self.r)) if self.rslora else self.r)

    def scaling_with(self, alpha: float | None = None, rslora: bool | None = None) -> float:
        """The scaling this module would carry with another alpha or rsLoRA reading (a neighbour of the grid)."""
        a = self.alpha if alpha is None else float(alpha)
        rs = self.rslora if rslora is None else bool(rslora)
        return a / (float(np.sqrt(self.r)) if rs else self.r)

    def delta(self, scale: float = 1.0) -> np.ndarray:
        return float(scale) * self.scaling * (self.B @ self.A)


@dataclass
class Adapter:
    name: str
    modules: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    ignored_keys: list = field(default_factory=list)
    unsupported_keys: list = field(default_factory=list)
    variant: str | None = None


_VARIANT_MARKERS = (("dora", ("lora_magnitude_vector", "dora_scale"), "use_dora"),
                    ("lora_bias", ("lora_B.bias",), "lora_bias"),
                    ("qalora", (), "use_qalora"))


def _meta_obj(metadata: dict) -> dict:
    raw = metadata.get(METADATA_KEY)
    return json.loads(raw) if raw else {}


def _pattern_value(pattern: dict, mod: str):
    """peft's rank_pattern / alpha_pattern lookup: the first key K (in order) with re.match(rf"(.*\\.)?({K})$", mod)."""
    for key, val in pattern.items():
        if re.match(rf"(.*\.)?({key})$", mod):
            return val
    return None


def parse_adapter(tensors: dict, metadata: dict, name: str = "adapter", prefix: str = PREFIX,
                  module_shapes: dict | None = None) -> Adapter:
    """Group lora_A / lora_B (and optional .alpha) tensors by transformer module and resolve r / alpha."""
    meta = _meta_obj(metadata)
    m_alpha = meta.get(f"{prefix}lora_alpha")
    m_r = meta.get(f"{prefix}r")
    alpha_pattern = meta.get(f"{prefix}alpha_pattern") or {}
    rank_pattern = meta.get(f"{prefix}rank_pattern") or {}
    rslora = bool(meta.get(f"{prefix}use_rslora", meta.get("use_rslora", False)))
    pairs: dict = {}
    kohya_alpha: dict = {}
    ad = Adapter(name=name, metadata=meta)
    for variant, _markers, flag in _VARIANT_MARKERS:
        if meta.get(f"{prefix}{flag}") or meta.get(flag):
            ad.variant = ad.variant or variant
    for key, val in tensors.items():
        if not key.startswith(prefix):
            ad.ignored_keys.append(key)
            continue
        rest = key[len(prefix):]
        hit = next((v for v, markers, _ in _VARIANT_MARKERS if any(m in rest for m in markers)), None)
        if hit:
            ad.unsupported_keys.append(key)
            ad.variant = ad.variant or hit
            continue
        if rest.endswith(".alpha"):
            kohya_alpha[rest[: -len(".alpha")]] = float(np.asarray(val).reshape(-1)[0])
            continue
        m = _KEY_RE.match(rest)
        if not m:
            raise ValueError(f"{key}: not a lora_A/lora_B weight key")
        pairs.setdefault(m["mod"], {})[m["ab"]] = np.asarray(val, dtype=np.float64)
    for mod, ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            raise ValueError(f"{mod}: has {sorted(ab)} but needs both lora_A and lora_B")
        A, B = ab["A"], ab["B"]
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"{mod}: LoRA factors must be 2-D, got A{A.shape} B{B.shape}")
        r = A.shape[0]
        if B.shape[1] != r:
            raise ValueError(f"{mod}: rank mismatch, A is [{A.shape[0]}, *] but B is [*, {B.shape[1]}] "
                             f"(a transposed factor?)")
        pr = _pattern_value(rank_pattern, mod)
        want_r = pr if pr is not None else m_r
        if want_r is not None and int(want_r) != r:
            raise ValueError(f"{mod}: A/B rank {r} disagrees with the metadata's r={want_r}")
        if module_shapes is not None:
            if mod not in module_shapes:
                raise ValueError(f"{mod}: not a module of the target model")
            out_f, in_f = module_shapes[mod]
            if B.shape[0] != out_f or A.shape[1] != in_f:
                raise ValueError(f"{mod}: B@A is [{B.shape[0]}, {A.shape[1]}] but the module weight is "
                                 f"[{out_f}, {in_f}] (a transposed delta or a wrong mapping)")
        pa = _pattern_value(alpha_pattern, mod)
        if m_alpha is not None:
            default = float(m_alpha)
        elif mod in kohya_alpha:
            default = kohya_alpha[mod]
        else:
            default = float(r)
        alpha = float(pa) if pa is not None else default
        ad.modules[mod] = ModuleLoRA(A=A, B=B, r=r, alpha=alpha, rslora=rslora, alpha_default=default)
    return ad


def load_adapter(path, name: str | None = None, module_shapes: dict | None = None) -> Adapter:
    tensors, metadata = stn.load(path)
    return parse_adapter(tensors, metadata, name=name or Path(path).parent.name, module_shapes=module_shapes)


def merged_weight(W, lora: ModuleLoRA | None, scale: float = 1.0) -> np.ndarray:
    """W + scale * (alpha/r) * B @ A in float64. lora=None or scale=0 returns W exactly (as float64)."""
    W64 = np.asarray(W, dtype=np.float64)
    if lora is None or scale == 0:
        return W64.copy()
    d = lora.delta(scale)
    if d.shape != W64.shape:
        raise ValueError(f"delta {d.shape} does not match weight {W64.shape}")
    return W64 + d


def linear_ref(x, W, bias=None, lora: ModuleLoRA | None = None, scale: float = 1.0) -> np.ndarray:
    """y = x @ merged(W).T + bias, float64 (nn.Linear convention: W is [out, in])."""
    y = np.asarray(x, dtype=np.float64) @ merged_weight(W, lora, scale).T
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64)
    return y


def unmerged_linear_ref(x, W, bias=None, lora: ModuleLoRA | None = None, scale: float = 1.0) -> np.ndarray:
    """The runtime (unmerged) form: x @ W.T + scale * (alpha/r) * (x @ A.T) @ B.T. Equal to linear_ref in exact
    arithmetic; kept separate so a test can show a merged and an unmerged implementation are the same contract."""
    x64 = np.asarray(x, dtype=np.float64)
    y = x64 @ np.asarray(W, dtype=np.float64).T
    if lora is not None and scale != 0:
        y = y + float(scale) * lora.scaling * ((x64 @ lora.A.T) @ lora.B.T)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64)
    return y


def shard_delta(delta: np.ndarray, kind: str, rank: int, tp: int, parts: int = 1) -> np.ndarray:
    """The rank-local slice of a [out, in] delta under Megatron-style TP.

    kind="column": rows (output features) split in tp contiguous blocks; with parts>1 (a fused QKV / gate-up
    layer whose out dim is `parts` equal sub-matrices stacked), each part's row block is taken and the parts are
    stacked again. kind="row": columns (input features) split in tp contiguous blocks.
    """
    d = np.asarray(delta)
    out_f, in_f = d.shape
    if kind == "column":
        if out_f % (tp * parts):
            raise ValueError(f"out={out_f} is not divisible by tp*parts={tp * parts}")
        sub = out_f // parts
        shard = sub // tp
        return np.concatenate([d[p * sub + rank * shard: p * sub + (rank + 1) * shard] for p in range(parts)], axis=0)
    if kind == "row":
        if in_f % tp:
            raise ValueError(f"in={in_f} is not divisible by tp={tp}")
        shard = in_f // tp
        return d[:, rank * shard:(rank + 1) * shard]
    raise ValueError(f"unknown shard kind {kind!r}")


class Lifecycle:
    """The adapter state every request must observe, replayed from an operation sequence.

    ops (dicts): {"op": "base"}                       generate with whatever is active (no change)
                 {"op": "set", "adapter": "A", "scale": 1.0}   activate one adapter, replacing any other
                 {"op": "set", "adapter": ["A", "C"], "scale": [1.0, 0.5]}   activate several at once (v4), each at
                                                      its own scale and with its own scaling convention
                 {"op": "disable"}                    deactivate (base model)
                 {"op": "scale", "scale": 0.5}        change the active (single) adapter's scale
                 any op with "refused": true          the contract says the engine must refuse it (a DoRA or LoRA-bias
                                                      file, v4): the state does NOT change
    Each op is followed by one request; `run` returns the per-request effective state {"adapter", "scale"}
    (adapter None = base; a list for several adapters).
    """

    def __init__(self):
        self.adapter = None
        self.scale = 1.0

    def apply(self, op: dict) -> dict:
        kind = op["op"]
        if op.get("refused"):
            pass
        elif kind == "base":
            pass
        elif kind == "set":
            a = op["adapter"]
            if isinstance(a, (list, tuple)):
                sc = op.get("scale", [1.0] * len(a))
                self.adapter, self.scale = list(a), [float(x) for x in sc]
            else:
                self.adapter, self.scale = a, float(op.get("scale", 1.0))
        elif kind == "disable":
            self.adapter = None
        elif kind == "scale":
            if self.adapter is None or isinstance(self.adapter, list):
                raise ValueError("scale op with no single active adapter")
            self.scale = float(op["scale"])
        else:
            raise ValueError(f"unknown lifecycle op {kind!r}")
        if self.adapter is None:
            return {"adapter": None, "scale": 0.0}
        if isinstance(self.adapter, list):
            if not any(self.scale):
                return {"adapter": None, "scale": 0.0}
            return {"adapter": list(self.adapter), "scale": list(self.scale)}
        if self.scale == 0:
            return {"adapter": None, "scale": 0.0}
        return {"adapter": self.adapter, "scale": self.scale}

    def run(self, ops) -> list[dict]:
        return [self.apply(op) for op in ops]


STATE_VARIANTS = ("pattern_ignored", "rslora_ignored", "first_at_last_scale")


def state_terms(state: dict, adapters: dict) -> dict:
    """{module: [(coef, ModuleLoRA), ...]} of one state -- the delta is sum(coef * B @ A) -- for a single or a multi
    adapter state, optionally under one of STATE_VARIANTS (state["variant"])."""
    names = state.get("adapter")
    if names is None:
        return {}
    scales = state.get("scale")
    if isinstance(names, str):
        names, scales = [names], [scales]
    variant = state.get("variant")
    if variant is not None and variant not in STATE_VARIANTS:
        raise ValueError(f"unknown state variant {variant!r}")
    shared = set.intersection(*(set(adapters[n].modules) for n in names)) if len(names) > 1 else set()
    last = adapters[names[-1]]
    out: dict = {}
    for n, s in zip(names, scales):
        if not s:
            continue
        for m, lo in adapters[n].modules.items():
            if variant == "pattern_ignored":
                c = lo.scaling_with(alpha=lo.alpha_default if lo.alpha_default is not None else lo.alpha)
            elif variant == "rslora_ignored":
                c = lo.scaling_with(rslora=False)
            elif variant == "first_at_last_scale" and m in shared and n != names[-1]:
                c = last.modules[m].scaling
            else:
                c = lo.scaling
            out.setdefault(m, []).append((float(s) * c, lo))
    return out


def state_deltas(state: dict, adapters: dict) -> dict:
    """{module: delta [out, in] float64} of one state (state_terms summed); {} for the base model."""
    return {m: sum(coef * (lo.B @ lo.A) for coef, lo in terms) for m, terms in state_terms(state, adapters).items()}


def effective_deltas(state: dict, adapters: dict) -> dict:
    """{module: ModuleLoRA-scaled delta} for one request's effective state; {} for the base model."""
    return state_deltas(state, adapters)


def expected_equalities(states: list[dict]) -> list[tuple[int, int]]:
    """Pairs of request indices whose outputs must be IDENTICAL under a correct implementation (same effective
    state), in the order they occur. The verifier compares these pairs at bit or tolerance level; everything
    else it compares against the merged-weight reference."""
    pairs = []
    for j, sj in enumerate(states):
        for i in range(j):
            if states[i] == sj:
                pairs.append((i, j))
                break
    return pairs
