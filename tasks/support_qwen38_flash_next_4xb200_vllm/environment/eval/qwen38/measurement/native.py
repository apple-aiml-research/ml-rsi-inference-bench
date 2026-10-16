#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Native regions: the optional declaration, discovery in the loaded model, capture and replay.

A *region* is one loaded native module method whose calls compute a complete QSA or PLE layer
(or a contiguous enclosing region around it). The default declaration names the checkpoint's
layers by module path; `/app/qwen38-adapter.json` may name other paths, a different method, the
positions of its hidden-state input and output, and an enclosing region. It names native values
only: never a case, verdict, replacement callable or constructed metadata.

Capture wraps the region at the instance level while the engine's own runner executes a schedule
(`measurement.schedule`). Each call is recorded with its arguments, a deep copy of the forward
context the runner built (attention metadata included), its output and the rows of every cache
tensor bound under the region before and after the call. Replay re-enters that forward context and
calls the native method again; the observer never constructs metadata or state.
"""
from __future__ import annotations

import copy
import dataclasses
import fnmatch
import hashlib
import inspect
import json
import os
import re
from pathlib import Path

SCHEMA = "qwen38-native-regions/1"
COMPONENTS = ("qsa", "ple")
# The contiguous checkpoint operations a region may enclose, in execution order: the complete QSA
# layer; the complete PLE layer, optionally with the residual addition of its output to the layer's
# hidden streams. A narrower native region is measured over a wider requested enclosure by composing
# the missing trailing operations (`COMPOSABLE`), so reference and candidate time the same work.
ENCLOSING = {
    "qsa": (("qsa",),),
    "ple": (("ple",), ("ple", "ple_residual")),
}
COMPOSABLE = ("ple_residual",)
DEFAULT_REGIONS = [
    {"component": "qsa", "module": "*.layers.*.self_attn", "method": "forward",
     "hidden_input": "hidden_states", "output": "return",
     "checkpoint_layer": r"\.layers\.(\d+)\.", "encloses": ["qsa"]},
    {"component": "ple", "module": "*.layers.*.ple", "method": "forward",
     "hidden_input": 0, "output": "return",
     "checkpoint_layer": r"\.layers\.(\d+)\.", "encloses": ["ple"]},
]
FIELDS = {"component", "module", "method", "hidden_input", "output", "checkpoint_layer", "encloses"}


class DeclarationError(ValueError):
    """An invalid or unresolvable native-region declaration."""


def engine_root() -> Path:
    return Path(os.environ.get("QWEN38_ENGINE_ROOT", "/app")).resolve()


def declaration(root: str | os.PathLike | None = None) -> list[dict]:
    """The committed declaration's regions, or the defaults when no file is committed."""
    path = Path(root or engine_root()) / "qwen38-adapter.json"
    if not path.is_file():
        return [dict(r) for r in DEFAULT_REGIONS]
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or set(data) - {"schema", "regions"}:
        raise DeclarationError(f"qwen38-adapter.json must be {{'schema': {SCHEMA!r}, 'regions': [...]}}")
    regions = data.get("regions")
    if not isinstance(regions, list) or not regions:
        raise DeclarationError("regions must be a nonempty list")
    out = []
    for row in regions:
        out.append(validate_region(row))
    if sorted(r["component"] for r in out) != sorted(set(r["component"] for r in out)):
        raise DeclarationError("declare at most one region per component")
    for component in COMPONENTS:
        if component not in {r["component"] for r in out}:
            out.append(dict(next(r for r in DEFAULT_REGIONS if r["component"] == component)))
    return out


def validate_region(row) -> dict:
    if not isinstance(row, dict) or set(row) - FIELDS:
        raise DeclarationError(f"unknown region fields: {set(row) - FIELDS if isinstance(row, dict) else row!r}")
    base = next((dict(r) for r in DEFAULT_REGIONS if r["component"] == row.get("component")), None)
    if base is None:
        raise DeclarationError(f"component must be one of {COMPONENTS}")
    base.update(row)
    if not isinstance(base["module"], str) or not base["module"]:
        raise DeclarationError("module must be a nonempty glob over model.named_modules() paths")
    if not isinstance(base["method"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", base["method"]) \
            or base["method"].startswith("__"):
        raise DeclarationError("method must name a native method")
    if not (isinstance(base["hidden_input"], str) or (type(base["hidden_input"]) is int and base["hidden_input"] >= 0)):
        raise DeclarationError("hidden_input is an argument name or a nonnegative position")
    out = base["output"]
    if not (out == "return" or (isinstance(out, dict) and set(out) == {"argument"}
                                and (isinstance(out["argument"], str) or type(out["argument"]) is int))):
        raise DeclarationError('output is "return" or {"argument": <name or position>}')
    try:
        if re.compile(base["checkpoint_layer"]).groups != 1:
            raise DeclarationError("checkpoint_layer needs exactly one capturing group")
    except re.error as exc:
        raise DeclarationError(f"invalid checkpoint_layer pattern: {exc}") from exc
    if tuple(base["encloses"]) not in ENCLOSING[base["component"]]:
        raise DeclarationError(f"encloses must be one of {ENCLOSING[base['component']]}")
    base["encloses"] = list(base["encloses"])
    return base


def effective(region: dict, requested=None) -> dict:
    """The region measured over `requested` enclosed operations: its declared ones plus composed
    trailing operations (`compose`). Requesting less than a region natively encloses is an error."""
    declared = list(region["encloses"])
    if requested is None or list(requested) == declared:
        return dict(region, compose=[])
    requested = list(requested)
    if tuple(requested) not in ENCLOSING[region["component"]] or requested[:len(declared)] != declared \
            or any(op not in COMPOSABLE for op in requested[len(declared):]):
        raise DeclarationError(f"{region['component']}: a region enclosing {declared} cannot be measured as "
                               f"{requested}")
    return dict(region, encloses=requested, compose=requested[len(declared):])


def compose(region: dict, hidden, value):
    """The region's logical output: the native value with the composed trailing operations applied."""
    for op in region.get("compose") or ():
        if op == "ple_residual":
            value = hidden + value
        else:
            raise DeclarationError(f"cannot compose {op}")
    return value


def resolve(model, region: dict, static_context: dict | None = None) -> list[dict]:
    """The loaded modules a region names, ordered by checkpoint layer, with their source identity."""
    root = engine_root()
    rows = []
    for name, module in model.named_modules():
        if not fnmatch.fnmatchcase(name, region["module"]):
            continue
        method = getattr(module, region["method"], None)
        if method is None or not callable(method):
            raise DeclarationError(f"{name} has no method {region['method']!r}")
        source = inspect.getsourcefile(inspect.unwrap(getattr(type(module), region["method"], method)))
        if not source or not Path(source).resolve().is_relative_to(root):
            raise DeclarationError(f"{name}.{region['method']} is not engine source under {root}")
        match = re.search(region["checkpoint_layer"], "." + name + ".")
        if not match:
            raise DeclarationError(f"{name}: checkpoint_layer pattern does not identify a layer")
        path = Path(source).resolve()
        rows.append({"name": name, "module": module, "layer": int(match.group(1)), "static_context": static_context,
                     "class": f"{type(module).__module__}.{type(module).__qualname__}",
                     "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not rows:
        raise DeclarationError(f"no loaded module matches {region['module']!r} for {region['component']}")
    rows.sort(key=lambda r: r["layer"])
    if len({r["layer"] for r in rows}) != len(rows):
        raise DeclarationError(f"{region['component']}: two loaded modules claim one checkpoint layer")
    return rows


def argument(args, kwargs, spec, signature=None):
    """The value at a declared argument name or position."""
    if isinstance(spec, str):
        if spec in kwargs:
            return kwargs[spec]
        if signature is not None:
            names = [p for p in signature.parameters if p != "self"]
            if spec in names and names.index(spec) < len(args):
                return args[names.index(spec)]
        raise DeclarationError(f"declared input {spec!r} is not an argument of this call")
    if spec < len(args):
        return args[spec]
    raise DeclarationError(f"declared input position {spec} is beyond this call's {len(args)} arguments")


def clone_tree(value, memo=None):
    """Copy every tensor inside nested containers, dataclasses and engine objects; share the rest."""
    import torch
    memo = {} if memo is None else memo
    key = id(value)
    if key in memo:
        return memo[key]
    if isinstance(value, torch.Tensor):
        out = value.detach().clone()
    elif isinstance(value, dict):
        out = {k: clone_tree(v, memo) for k, v in value.items()}
    elif isinstance(value, list):
        out = [clone_tree(v, memo) for v in value]
    elif isinstance(value, tuple) and not hasattr(value, "_fields"):
        out = tuple(clone_tree(v, memo) for v in value)
    elif isinstance(value, (torch.nn.Module, type)) or callable(value) and not hasattr(value, "__dict__"):
        out = value
    elif (dataclasses.is_dataclass(value) or hasattr(value, "__dict__")) and \
            type(value).__module__.split(".")[0] in ("vllm", "flashinfer"):
        out = copy.copy(value)
        memo[key] = out
        names = [f.name for f in dataclasses.fields(value)] if dataclasses.is_dataclass(value) else list(vars(value))
        for name in names:
            try:
                object.__setattr__(out, name, clone_tree(getattr(value, name), memo))
            except (AttributeError, TypeError):
                pass
        return out
    else:
        out = value
    memo[key] = out
    return out


def tensors_of(value, seen=None) -> list:
    """Every tensor reachable from captured call inputs (the same traversal as `clone_tree`)."""
    import torch
    seen = set() if seen is None else seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return [value]
    out = []
    if isinstance(value, dict):
        for v in value.values():
            out += tensors_of(v, seen)
    elif isinstance(value, (list, tuple)):
        for v in value:
            out += tensors_of(v, seen)
    elif (dataclasses.is_dataclass(value) or hasattr(value, "__dict__")) and \
            type(value).__module__.split(".")[0] in ("vllm", "flashinfer") and \
            not isinstance(value, (torch.nn.Module, type)):
        names = [f.name for f in dataclasses.fields(value)] if dataclasses.is_dataclass(value) else list(vars(value))
        for name in names:
            out += tensors_of(getattr(value, name, None), seen)
    return out


class Inputs:
    """Pristine copies of a captured call's inputs (arguments and attention metadata), restored before
    every replay, so a native method that writes into its inputs replays identically."""

    def __init__(self, call):
        live = tensors_of([call["args"], call["kwargs"], call["attn_metadata"], call["context"].get("slot_mapping")])
        self.pairs = [(t, t.clone()) for t in live]

    def restore(self):
        for live, saved in self.pairs:
            live.copy_(saved)


def bound_state(module, static_context: dict | None = None) -> list[tuple[str, object, str | None]]:
    """Every cache tensor the engine bound to the region's module tree (its native KV binding), with
    the engine layer name it is bound under (`static_forward_context`), which names its KV group."""
    import torch
    by_tensor = {}
    for layer_name, layer in (static_context or {}).items():
        kv = getattr(layer, "kv_cache", None)
        for t in ([kv] if isinstance(kv, torch.Tensor) else list(kv) if isinstance(kv, (list, tuple)) else []):
            if isinstance(t, torch.Tensor):
                by_tensor.setdefault(id(t), layer_name)
    seen, rows = set(), []
    for name, sub in module.named_modules():
        kv = getattr(sub, "kv_cache", None)
        tensors = [kv] if isinstance(kv, torch.Tensor) else list(kv) if isinstance(kv, (list, tuple)) else []
        for i, t in enumerate(tensors):
            if isinstance(t, torch.Tensor) and t.numel() and t.device.type == "cuda" and id(t) not in seen:
                seen.add(id(t))
                rows.append((f"{name}.kv_cache[{i}]" if name else f"kv_cache[{i}]", t, by_tensor.get(id(t))))
    return rows


class StateRows:
    """Rows (cache blocks) of bound tensors, saved for exact restoration.

    Every engine cache view is indexed by block along dimension 0. Each tensor is saved over the
    blocks its own KV group assigned to the schedule's requests (`blocks(layer_name)`), so a
    snapshot never includes rows another group owns inside the shared cache buffer."""

    def __init__(self, tensors, blocks):
        import torch
        self.tensors, self.index, self.saved = [], [], []
        for name, t, layer_name in tensors:
            rows = blocks(layer_name) if callable(blocks) else blocks
            if not rows:
                continue
            index = torch.tensor(sorted(set(rows)), dtype=torch.long, device=t.device)
            self.tensors.append((name, t, layer_name))
            self.index.append(index)
            self.saved.append(t.index_select(0, index).clone())

    def restore(self):
        for (_, t, _), index, saved in zip(self.tensors, self.index, self.saved):
            t.index_copy_(0, index, saved)

    def current(self):
        return [t.index_select(0, index) for (_, t, _), index in zip(self.tensors, self.index)]

    def equal(self, other_rows) -> list[bool]:
        """Bitwise equality of the saved rows with other rows."""
        return [bitwise_equal(a, b) for a, b in zip(self.saved, other_rows)]

    def nbytes(self) -> int:
        return sum(s.numel() * s.element_size() for s in self.saved)


def _int_view(t):
    import torch
    return {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]


def bitwise_equal(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        (a.contiguous().view(-1).view(_int_view(a)) == b.contiguous().view(-1).view(_int_view(b))).all())


_MISSING = object()


class Capture:
    """Record each call of a region's native method on one loaded module.

    In ``history`` mode only the region's hidden-state input is kept (the logical history a
    comparator needs); in ``full`` mode the call's arguments, a deep copy of the forward context the
    runner built, and the native output are kept for replay. ``intervene``, when set, perturbs the
    region's output of full-mode calls (a use witness; never during capture for timing). The
    wrapper is installed on the instance and removed on exit."""

    def __init__(self, entry: dict, region: dict):
        self.entry, self.region = entry, region
        self.module = entry["module"]
        self.state = bound_state(self.module, entry.get("static_context"))
        self.calls = []
        self.mode = None          # None (not recording), "history" or "full"
        self.intervene = None
        # (target list, block rows): saved at the next call, after the runner prepared the step
        # (fresh blocks zeroed) and before the region writes anything.
        self.pending = None
        self._installed = False

    def __enter__(self):
        from vllm.forward_context import get_forward_context
        method = self.region["method"]
        original = getattr(self.module, method)
        signature = inspect.signature(getattr(type(self.module), method))

        def observed(*args, **kwargs):
            if self.mode is None:
                return original(*args, **kwargs)
            if self.mode == "history":
                hidden = clone_tree(argument(args, kwargs, self.region["hidden_input"], signature))
                result = original(*args, **kwargs)
                self.calls.append({"phase": "history", "hidden": hidden})
                return result
            if self.pending is not None:
                target, rows = self.pending
                self.pending = None
                if rows and self.state:
                    target.append(StateRows(self.state, rows))
            ctx = get_forward_context()
            record = {"phase": "full",
                      "args": [clone_tree(a) for a in args],
                      "kwargs": {k: clone_tree(v) for k, v in kwargs.items()},
                      "attn_metadata": clone_tree(ctx.attn_metadata),
                      "context": {"slot_mapping": clone_tree(getattr(ctx, "slot_mapping", None)),
                                  "batch_descriptor": getattr(ctx, "batch_descriptor", None),
                                  "cudagraph_runtime_mode": getattr(ctx, "cudagraph_runtime_mode", None)}}
            record["hidden"] = argument(record["args"], record["kwargs"], self.region["hidden_input"], signature)
            result = original(*args, **kwargs)
            value = output_value(self.region, args, kwargs, result, signature)
            import torch
            if not isinstance(value, torch.Tensor):
                raise DeclarationError(f"{self.entry['name']}: the declared output is not a tensor")
            record["output"] = compose(self.region, record["hidden"], value.detach()).clone()
            if self.intervene is not None:
                self.intervene(value)  # in place: the native tensor the enclosing computation consumes
            self.calls.append(record)
            return result

        # Restore exactly what was there: an instance attribute another observer installed, or the class method.
        self._previous = vars(self.module).get(method, _MISSING)
        setattr(self.module, method, observed)
        self._installed = True
        return self

    def __exit__(self, *exc):
        if self._installed:
            method = self.region["method"]
            if self._previous is _MISSING:
                delattr(self.module, method)
            else:
                setattr(self.module, method, self._previous)
            self._installed = False
        return False


def output_value(region, args, kwargs, result, signature=None):
    spec = region["output"]
    if spec == "return":
        return result
    return argument(args, kwargs, spec["argument"], signature)


def replay(vllm_config, entry: dict, region: dict, call: dict):
    """Call the native method again inside the captured forward context; return its output."""
    from vllm.forward_context import set_forward_context
    ctx = call["context"]
    method = getattr(entry["module"], region["method"])
    signature = inspect.signature(getattr(type(entry["module"]), region["method"]))
    with set_forward_context(call["attn_metadata"], vllm_config,
                             cudagraph_runtime_mode=ctx.get("cudagraph_runtime_mode"),
                             batch_descriptor=ctx.get("batch_descriptor"),
                             slot_mapping=ctx.get("slot_mapping")):
        result = method(*call["args"], **call["kwargs"])
    return compose(region, call["hidden"], output_value(region, call["args"], call["kwargs"], result, signature))


def identity(entry: dict, region: dict) -> dict:
    """The loaded region's identity as bound into every record."""
    return {"component": region["component"], "owner": entry["name"], "layer": entry["layer"],
            "class": entry["class"], "method": region["method"], "output": region["output"], "source": entry["source"],
            "source_sha256": entry["source_sha256"], "encloses": list(region["encloses"]),
            "composed": list(region.get("compose") or []),
            "state": [f"{name}@{layer}" for name, _, layer in bound_state(entry["module"], entry.get("static_context"))]}
