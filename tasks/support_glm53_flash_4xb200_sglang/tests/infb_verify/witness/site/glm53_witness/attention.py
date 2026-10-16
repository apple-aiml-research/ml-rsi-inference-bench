#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""GLM-7: the complete attention sublayer of the first and last DSA layers of the released config, observed where it is
invoked in the served model, in every rank of the (untimed) witness launch."""
from __future__ import annotations

import copy
import os
import threading

from . import FEATURES, after_import, control, emit, error, store

ON = "attention" in FEATURES
REEXEC = 3
GRAPH_BRANCHES = os.environ.get("INFB_GLM53_WITNESS_GRAPH_BRANCHES", "1") != "0"
_local = threading.local()
_st: dict = {"wrapped": [], "flags": [], "bufs": {}, "next_bufs": {}, "mlp_bufs": {}, "calls": 0}
_pf: dict = {"runners": {}}


def _nbytes(t) -> int:
    return int(t.numel()) * int(t.element_size())


def _static_buffer(t, slot: tuple):
    """Where the in-graph copy of `t` lands while a graph is captured. On the decode runner (no prefill capture in
    progress) a fresh buffer per captured shape, exactly as v50 to v56 allocate it. Inside a prefill-runner capture (v62)
    the leading rows of that runner's shared buffer for `slot`, allocated once, inside the capture (from the graph's
    pool), and sized from the runner's own largest captured size; a tensor that is not laid out over the captured rows (a
    0-d scale), has more rows than that size, or has another trailing shape or dtype than the slot's buffer gets a fresh
    buffer, as on decode, counted in `fresh`. A failed allocation stops prefill observation on that runner, with the
    reason logged, and raises into the calling site's own guard (an observer error; the served call goes on)."""
    import torch
    pf = getattr(_local, "prefill", None)
    if pf is None:
        return torch.empty_like(t)
    rows = pf["rows"]
    buf = pf["shared"].get(slot) if t.dim() > 0 and t.shape[0] <= rows else None
    try:
        if t.dim() == 0 or t.shape[0] > rows or (
                buf is not None and (tuple(buf.shape[1:]) != tuple(t.shape[1:]) or buf.dtype != t.dtype)):
            fresh = torch.empty_like(t)
            pf["fresh"] += 1
            pf["fresh_bytes"] += _nbytes(fresh)
            return fresh
        if buf is None:
            buf = torch.empty((rows, *t.shape[1:]), dtype=t.dtype, device=t.device)
            pf["shared"][slot] = buf
            pf["bytes"] += _nbytes(buf)
    except Exception as exc:  # noqa: BLE001 - the observer's own allocation failed: no prefill rows from here on
        pf["failed"] = f"a static buffer for {slot} ({rows} rows) could not be allocated: {exc!r}"[:400]
        emit("prefill_observer", stage="stopped", installed=False, runner=pf["runner"], rows=rows, reason=pf["failed"])
        _local.capture_key = None
        _local.prefill = None
        raise
    return buf[: t.shape[0]]


def designated_layers(runner) -> list[int]:
    cfg = getattr(getattr(runner, "model_config", None), "hf_config", None)
    text = getattr(cfg, "text_config", None) or cfg
    types = list(getattr(text, "layer_types", None) or [])
    dsa = [i for i, t in enumerate(types) if t == "deepseek_sparse_attention"]
    return sorted({dsa[0], dsa[-1]}) if dsa else []


def _locate_child(model, layer: int, needle: str):
    """The child of decoder layer `layer` (the module at `<...>.layers.<layer>`) whose name contains `needle`: the
    checkpoint's `self_attn` (`attn`) or `mlp`."""
    for name, mod in model.named_modules():
        if name.endswith(f"layers.{layer}") and name.rsplit(".", 2)[-2] == "layers":
            for child_name, child in mod.named_children():
                if needle in child_name:
                    return child, f"{name}.{child_name}"
    return None, None


def _locate_layer(model, index: int):
    for name, mod in model.named_modules():
        if name.endswith(f"layers.{index}") and name.rsplit(".", 2)[-2] == "layers":
            return mod, name
    return None, None


def _branch():
    return getattr(_local, "branch", None)


def _split(hidden):
    """(values, scale or None) of an input as it arrived."""
    if isinstance(hidden, tuple):
        return hidden[0], (hidden[1] if len(hidden) > 1 else None)
    return hidden, None


def _rebuild(x, items):
    """`items` (a list, or a dict for a mapping) as a container of `x`'s own type. v56: v50 rebuilt every tuple
    as a plain tuple, so an implementation's NamedTuple layer state lost its fields inside the served forward. A plain
    tuple, list or dict is rebuilt as itself; a NamedTuple by its `_make`; a dict subclass as a copy of itself holding
    the new values (its own state kept: an OrderedDict's order, a defaultdict's factory); any other tuple or list
    subclass by its own constructor from the items (torch.Size, a structseq)."""
    t = type(x)
    if t in (tuple, list, dict):
        return t(items)
    if isinstance(x, tuple) and callable(getattr(t, "_make", None)):
        return t._make(items)
    if isinstance(x, dict):
        c = copy.copy(x)
        c.clear()
        c.update(items)
        return c
    return t(items)


def _clone(x):
    """A deep copy of every tensor in `x` (tuples, lists, dicts), keeping each tensor's Python attributes (for example
    a deferred-reduction mark) and each container's own type (`_rebuild`); anything else is passed through."""
    import torch
    if isinstance(x, torch.Tensor):
        c = x.clone()
        for k, v in getattr(x, "__dict__", {}).items():
            try:
                setattr(c, k, v)
            except Exception:  # noqa: BLE001
                pass
        return c
    if isinstance(x, (tuple, list)):
        return _rebuild(x, [_clone(v) for v in x])
    if isinstance(x, dict):
        return _rebuild(x, {k: _clone(v) for k, v in x.items()})
    return x


def _zeros(x):
    """Zeros shaped like every tensor in `x`, each container rebuilt as its own type (`_rebuild`); v56: a dict too, so a
    branch never receives the served result's own tensors."""
    import torch
    if isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    if isinstance(x, (tuple, list)):
        return _rebuild(x, [_zeros(v) for v in x])
    if isinstance(x, dict):
        return _rebuild(x, {k: _zeros(v) for k, v in x.items()})
    return x


def _is_positions(v) -> bool:
    """A positions vector: a 1-D integer tensor."""
    import torch
    return isinstance(v, torch.Tensor) and v.dim() == 1 and not v.is_floating_point() and v.dtype != torch.bool


def _is_hidden(v) -> bool:
    """Hidden states: a floating tensor of two or more dimensions, or an FP8 `(values, scale)` tuple whose values are
    one (v56: v50 took any tensor, so a positions vector could pass for hidden states)."""
    import torch
    if isinstance(v, tuple) and v:
        v = v[0]
    return isinstance(v, torch.Tensor) and v.dim() >= 2 and v.is_floating_point()


def _call_args(args, kwargs):
    """(positions, hidden_states, forward_batch) of a call, read by name or by type, never by position alone: each by
    its keyword; otherwise the first positional argument, then the first other keyword value, of its type -- a 1-D
    integer tensor for the positions, a floating tensor of two or more dimensions (or an FP8 `(values, scale)` tuple)
    for the hidden states, an object with a forward mode for the batch. A call that carries no positions is read at
    the forward batch's own `positions` (the tensor runtime.py reads). Reading the first two positional slots as
    positions and hidden states would misread a `self_attn(hidden_states, forward_batch)` signature as positions =
    the hidden states and no hidden states, and its attention would never be recorded."""
    positions, hidden, fb = kwargs.get("positions"), kwargs.get("hidden_states"), kwargs.get("forward_batch")
    rest = [*args, *(v for k, v in kwargs.items() if k not in ("positions", "hidden_states", "forward_batch"))]
    if positions is None:
        positions = next((v for v in rest if _is_positions(v)), None)
    if hidden is None:
        hidden = next((v for v in rest if _is_hidden(v)), None)
    if fb is None:
        fb = next((v for v in rest if hasattr(v, "forward_mode")), None)
    if positions is None and fb is not None:
        positions = getattr(fb, "positions", None)
    return positions, hidden, fb


def _layer_args(args, kwargs):
    """(positions, hidden_states, forward_batch, residual) of a decoder layer's forward call (`_call_args`); the
    residual by keyword, or the first later positional Tensor of the hidden states' shape -- an argument of another
    type (an allocator, a flag) is never read as the residual."""
    import torch
    positions, hidden, fb = _call_args(args, kwargs)
    residual = kwargs.get("residual")
    if residual is not None and not isinstance(residual, torch.Tensor):
        residual = None
    h = hidden[0] if isinstance(hidden, tuple) else hidden
    if residual is None and isinstance(h, torch.Tensor):
        for v in args[3:]:
            if isinstance(v, torch.Tensor) and v.shape == h.shape:
                residual = v
                break
    return positions, hidden, fb, residual


def _stream(hidden, residual):
    """The residual stream a decoder layer receives: its hidden states, plus the carried residual if there is one."""
    h = hidden[0] if isinstance(hidden, tuple) else hidden
    if residual is not None:
        h = h + residual
    return h.reshape(h.shape[0], -1)


def _meta(fb) -> dict:
    out = {"mode": getattr(getattr(fb, "forward_mode", None), "name", None), "bs": getattr(fb, "batch_size", None)}
    for key in ("seq_lens_cpu", "extend_seq_lens_cpu", "extend_prefix_lens_cpu"):
        v = getattr(fb, key, None)
        if v is not None:
            try:
                out[key] = v.tolist() if hasattr(v, "tolist") else list(v)
            except Exception:  # noqa: BLE001
                pass
    return out


def _windowed(kind: str, name: str, positions, h, window, fields: dict) -> None:
    """Keep the rows of `h` whose position lies in `window`; one safetensors file and one `kind` event. Rows that do
    not align with the layer's positions (a scattered layout) are kept whole and marked `aligned: false`: the branches
    of one call share that layout, so they stay comparable row by row."""
    import torch
    lo, hi = window
    if h is None:
        return
    if positions is None or positions.shape[0] != h.shape[0]:
        if h.shape[0] == 0:
            return
        _st["calls"] += 1
        n = _st["calls"]
        file = store(f"{name}-c{n:06d}", {"h": h})
        emit(kind, call=n, file=file, rows=int(h.shape[0]), aligned=False,
             positions_rows=int(positions.shape[0]) if positions is not None else None, dtype=str(h.dtype),
             width=int(h.shape[-1]), **fields)
        return
    keep = (positions >= lo) & (positions < hi)
    idx = torch.nonzero(keep).flatten()
    if idx.numel() == 0:
        return
    _st["calls"] += 1
    n = _st["calls"]
    t = {"positions": positions.index_select(0, idx).to(torch.int64), "h": h.index_select(0, idx)}
    file = store(f"{name}-c{n:06d}", t)
    emit(kind, call=n, file=file, rows=int(idx.numel()), aligned=True, positions_first=int(t["positions"][0]),
         positions_last=int(t["positions"][-1]), dtype=str(t["h"].dtype), width=int(t["h"].shape[-1]), **fields)


def _wrap_next(mod, layer: int, path: str, stream: str) -> None:
    """Record the stream entering decoder layer `mod` in the served call: designated layer `layer` itself (`in`) or the
    one after it (`next`)."""
    import torch
    orig = mod.forward

    def forward(*args, **kwargs):
        if _branch() is None:
            try:
                positions, hidden, fb, residual = _layer_args(args, kwargs)
                if positions is not None and hidden is not None and positions.numel() > 0:
                    h = _stream(hidden, residual)
                    deferred = bool(getattr(hidden, "_sglang_needs_allreduce_fusion", False))
                    if torch.cuda.is_current_stream_capturing():
                        key = getattr(_local, "capture_key", None)
                        if key is not None:
                            bufs = {"positions": _static_buffer(positions, ("next", layer, stream, "positions")),
                                    "h": _static_buffer(h, ("next", layer, stream, "h"))}
                            bufs["positions"].copy_(positions)
                            bufs["h"].copy_(h)
                            _st["next_bufs"].setdefault(key, []).append((layer, path, bufs, deferred, stream))
                    else:
                        window = (control().get("args") or {}).get("capture_next")
                        if window:
                            _windowed("next_input", f"N{layer}{stream}", positions.detach(), h.detach(), window,
                                      {"layer": layer, "stream": stream, "module": path, "where": "eager",
                                       "deferred_allreduce": deferred, **_meta(fb)})
            except Exception as exc:  # noqa: BLE001
                error(f"next-layer input L{layer}", exc)
        return orig(*args, **kwargs)

    mod.forward = forward
    emit("next_wrapped", layer=layer, stream=stream, module=path, cls=type(mod).__name__)


def _record(layer: int, path: str, positions, x, scale, y, meta: dict, where: str, rows: int | None = None) -> None:
    import torch
    _st["calls"] += 1
    n = _st["calls"]
    t = {"positions": positions[:rows] if rows is not None else positions,
         "x": x[:rows] if rows is not None else x, "y": y[:rows] if rows is not None else y}
    if scale is not None:
        t["x_scale"] = scale[:rows] if rows is not None and scale.dim() > 0 and scale.shape[0] >= rows else scale
    t = {k: (v.to(torch.int64) if k == "positions" else v) for k, v in t.items()}
    file = store(f"L{layer}-c{n:06d}", t)
    emit("attention_call", call=n, layer=layer, module=path, where=where, file=file,
         shapes={k: list(v.shape) for k, v in t.items()}, dtypes={k: str(v.dtype) for k, v in t.items()},
         positions_first=int(t["positions"][0]) if t["positions"].numel() else None,
         positions_last=int(t["positions"][-1]) if t["positions"].numel() else None, **meta)


def _flag(device):
    import torch
    t = torch.ones((), device=device, dtype=torch.bfloat16)
    _st["flags"].append(t)
    return t


def _wrap(mod, layer: int, path: str) -> None:
    import torch
    orig = mod.forward
    flag = _flag(next(mod.parameters()).device)

    def forward(*args, **kwargs):
        branch = _branch()
        if branch is not None:
            cached = getattr(_local, "attn_out", {}).get(layer)
            if cached is None:
                raise RuntimeError(f"no served attention result of layer {layer} to substitute in branch {branch}")
            if branch != "Z":
                return _clone(cached)
            return _rebuild(cached, [_zeros(cached[0]), *cached[1:]]) if isinstance(cached, tuple) else _zeros(cached)
        out = orig(*args, **kwargs)
        try:
            y = (out[0] if isinstance(out, tuple) else out) * flag
            out = _rebuild(out, [y, *out[1:]]) if isinstance(out, tuple) else y
            if getattr(_local, "triple", None) == layer:
                _local.attn_out[layer] = _clone(out)
            positions, hidden, fb = _call_args(args, kwargs)
            x, scale = _split(hidden)
            if torch.cuda.is_current_stream_capturing():
                key = getattr(_local, "capture_key", None)
                if key is not None and positions is not None and x is not None:
                    bufs = {"positions": _static_buffer(positions, ("attn", layer, "positions")),
                            "x": _static_buffer(x, ("attn", layer, "x")), "y": _static_buffer(y, ("attn", layer, "y"))}
                    if scale is not None and hasattr(scale, "shape"):
                        bufs["x_scale"] = _static_buffer(scale, ("attn", layer, "x_scale"))
                    bufs["positions"].copy_(positions)
                    bufs["x"].copy_(x)
                    bufs["y"].copy_(y)
                    if "x_scale" in bufs:
                        bufs["x_scale"].copy_(scale)
                    _st["bufs"].setdefault(key, []).append((layer, path, bufs))
            elif (control().get("args") or {}).get("capture_attention") and positions is not None and x is not None:
                _record(layer, path, positions.detach().clone(), x.detach().clone(),
                        scale.detach().clone() if scale is not None and hasattr(scale, "clone") else None,
                        y.detach().clone(), _meta(fb), "eager")
        except Exception as exc:  # noqa: BLE001
            error(f"attention forward L{layer}", exc)
        return out

    mod.forward = forward
    _st["wrapped"].append((layer, path, type(mod).__name__))
    emit("attention_wrapped", layer=layer, module=path, cls=type(mod).__name__,
         params=sorted(n for n, _ in mod.named_parameters())[:64])


def _wrap_mlp(mod, layer: int, path: str) -> None:
    """The MLP child of designated layer `layer`: in U it runs as served (its output's shape kept for the substitutes);
    in Z and the re-executions it returns zeros of that shape. In every branch its input is the observable."""
    import torch
    orig = mod.forward

    def forward(*args, **kwargs):
        branch = _branch()
        triple = getattr(_local, "triple", None) == layer
        if triple:
            try:
                hidden = kwargs.get("hidden_states")
                if hidden is None and len(args) > 0 and _is_hidden(args[0]):
                    hidden = args[0]
                x, _ = _split(hidden)
                name = branch or "U"
                positions = getattr(_local, "positions", {}).get(layer)
                if torch.cuda.is_current_stream_capturing():
                    key = getattr(_local, "capture_key", None)
                    if key is not None and x is not None and positions is not None:
                        bufs = {"positions": _static_buffer(positions, ("mlp", layer, name, "positions")),
                                "h": _static_buffer(x, ("mlp", layer, name, "h"))}
                        bufs["positions"].copy_(positions)
                        bufs["h"].copy_(x)
                        _st["mlp_bufs"].setdefault(key, []).append((layer, path, name, bufs))
                else:
                    window = (control().get("args") or {}).get("capture_next")
                    if window and x is not None and positions is not None and positions.numel() > 0:
                        _windowed("mlp_input", f"M{layer}{name}", positions.detach(), x.detach(), window,
                                  {"layer": layer, "branch": name, "module": path, "where": "eager",
                                   **_meta(getattr(_local, "fb", {}).get(layer))})
            except Exception as exc:  # noqa: BLE001
                error(f"mlp input L{layer}", exc)
        if branch is not None:
            shape = getattr(_local, "mlp_out", {}).get(layer)
            if shape is None:
                raise RuntimeError(f"no served MLP result of layer {layer} to shape branch {branch}'s zeros")
            return _zeros(shape)
        out = orig(*args, **kwargs)
        if triple:
            _local.mlp_out[layer] = out
        return out

    mod.forward = forward
    emit("mlp_wrapped", layer=layer, module=path, cls=type(mod).__name__)


def _wrap_triple(mod, layer: int, path: str) -> None:
    """Designated decoder layer `layer`: U (served, returned as clones), then Z and the re-executions on clones of its
    arguments."""
    import torch
    orig = mod.forward

    def forward(*args, **kwargs):
        capturing = torch.cuda.is_current_stream_capturing()
        armed = (capturing and GRAPH_BRANCHES) or (not capturing and bool((control().get("args") or {}).get("consumption")))
        if not armed or _branch() is not None:
            return orig(*args, **kwargs)
        for slot in ("attn_out", "mlp_out", "positions", "fb"):
            if not hasattr(_local, slot):
                setattr(_local, slot, {})
        try:
            positions, _, fb, _ = _layer_args(args, kwargs)
            args_z, kwargs_z = _clone(args), _clone(kwargs)
            reexec = [(_clone(args), _clone(kwargs)) for _ in range(REEXEC)]
        except Exception as exc:  # noqa: BLE001
            error(f"triple call L{layer}: the branches' arguments", exc)
            return orig(*args, **kwargs)
        _local.triple = layer
        _local.positions[layer], _local.fb[layer] = positions, fb
        try:
            served = orig(*args, **kwargs)
        except BaseException:
            _local.triple = None
            raise
        try:
            out = _clone(served)
        except Exception as exc:  # noqa: BLE001 - v56: no branch runs, so the served result itself is returned
            error(f"triple call L{layer}: the served result", exc)
            _local.triple = None
            _local.attn_out.pop(layer, None)
            _local.mlp_out.pop(layer, None)
            return served
        try:
            for name, a_, k_ in (("Z", args_z, kwargs_z), *[(f"U{j}", a, k) for j, (a, k) in enumerate(reexec, 2)]):
                _local.branch = name
                orig(*a_, **k_)
        except Exception as exc:  # noqa: BLE001
            error(f"triple call L{layer} branch {_branch()}", exc)
        finally:
            _local.branch = None
            _local.triple = None
            _local.attn_out.pop(layer, None)
            _local.mlp_out.pop(layer, None)
        return out

    mod.forward = forward
    emit("triple_wrapped", layer=layer, module=path, cls=type(mod).__name__)


def install_on_model(runner) -> None:
    if not ON or getattr(runner, "is_draft_worker", False):
        return
    layers = designated_layers(runner)
    emit("attention_designated", layers=layers)
    for layer in layers:
        mod, path = _locate_child(runner.model, layer, "attn")
        mlp, mpath = _locate_child(runner.model, layer, "mlp")
        if mod is None or mlp is None:
            emit("attention_wrapped", layer=layer, module=path, mlp=mpath,
                 error="no attention or MLP child found at this layer index")
            continue
        _wrap(mod, layer, path)
        _wrap_mlp(mlp, layer, mpath)
        for index, stream in ((layer, "in"), (layer + 1, "next")):
            dec, dpath = _locate_layer(runner.model, index)
            if dec is None:
                emit("next_wrapped", layer=layer, stream=stream, module=None, error=f"no decoder layer {index}")
            else:
                _wrap_next(dec, layer, dpath, stream)
        dec, dpath = _locate_layer(runner.model, layer)
        _wrap_triple(dec, layer, dpath)


def _read_back(key, rows, forward_batch, runner: str) -> None:
    """After one replay of the graph captured under `key`, what was captured into it, read back for the rows the replay
    served (`rows`: None reads the whole static buffer, 0 reads nothing): the next-layer and MLP inputs inside the
    phase's `capture_next` window (the MLP inputs only while `consumption` is armed), and the attention calls while
    `capture_attention` is on. Every row is copied out at once, on the replay's stream and before anything else is
    enqueued, and no emitted row is a view of a static buffer: a prefill runner's buffers are shared by all its shapes,
    so its next replay overwrites them (v62)."""
    args_ = control().get("args") or {}
    try:
        window = args_.get("capture_next")
        if window and rows != 0:
            meta = {**_meta(forward_batch), "graph_key": list(key), "where": "graph", "graph_runner": runner}
            for layer, path, bufs, deferred, stream in _st["next_bufs"].get(key, []):
                n_rows = rows if rows is not None else bufs["positions"].shape[0]
                _windowed("next_input", f"N{layer}{stream}", bufs["positions"][:n_rows].clone(),
                          bufs["h"][:n_rows].clone(), window,
                          {"layer": layer, "stream": stream, "module": path, "deferred_allreduce": deferred,
                           **meta})
            if args_.get("consumption"):
                for layer, path, name, bufs in _st["mlp_bufs"].get(key, []):
                    n_rows = rows if rows is not None else bufs["positions"].shape[0]
                    _windowed("mlp_input", f"M{layer}{name}", bufs["positions"][:n_rows].clone(),
                              bufs["h"][:n_rows].clone(), window,
                              {"layer": layer, "branch": name, "module": path, **meta})
    except Exception as exc:  # noqa: BLE001
        error("next-layer and MLP input graph read-back", exc)
    if args_.get("capture_attention"):
        try:
            for layer, path, bufs in (_st["bufs"].get(key, []) if rows != 0 else []):
                _record(layer, path, bufs["positions"].clone(), bufs["x"].clone(),
                        bufs["x_scale"].clone() if "x_scale" in bufs else None, bufs["y"].clone(),
                        {**_meta(forward_batch), "graph_key": list(key), "graph_runner": runner}, "graph", rows)
        except Exception as exc:  # noqa: BLE001
            error("attention graph read-back", exc)


def _patch_graph_runner(mod) -> None:
    cls = getattr(mod, "DecodeCudaGraphRunner", None)
    if cls is None:
        return
    orig_capture, orig_execute = cls.capture_one_shape, cls.execute

    def capture_one_shape(self, size, *args, **kwargs):
        _local.capture_key = (id(self), int(size))
        try:
            return orig_capture(self, size, *args, **kwargs)
        finally:
            _local.capture_key = None

    def execute(self, forward_batch, *args, **kwargs):
        out = orig_execute(self, forward_batch, *args, **kwargs)
        try:
            key = (id(self), int(self._replay_graph_key.size))
            raw = getattr(self, "raw_num_token", None)
            rows = int(raw) if raw is not None else None
        except Exception as exc:  # noqa: BLE001
            error("graph read-back", exc)
            return out
        _read_back(key, rows, forward_batch, "decode")
        return out

    cls.capture_one_shape = capture_one_shape
    cls.execute = execute


PREFILL_RUNNER_MODULE = "sglang.srt.model_executor.runner.prefill_cuda_graph_runner"
_PREFILL_SURFACE = {"capture_one_shape": ("size",), "execute": ("forward_batch",),
                    "_shape_key": ("num_tokens", "forward_batch")}
_PREFILL_BACKENDS = ("full", "breakable")


def _prefill_surface(mod) -> str | None:
    """Why the engine's prefill graph runner cannot be observed by this patch, or None: the class must exist, each of
    its methods in `_PREFILL_SURFACE` must take the parameters the patch reads, by those names and in that order after
    `self`, and the capture must take `prefix_num_chunks` by keyword, as the engine's capture loop passes it."""
    import inspect
    cls = getattr(mod, "PrefillCudaGraphRunner", None)
    if cls is None:
        return "the engine defines no PrefillCudaGraphRunner"
    for name, reads in _PREFILL_SURFACE.items():
        fn = getattr(cls, name, None)
        if not callable(fn):
            return f"PrefillCudaGraphRunner has no {name}"
        try:
            params = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError) as exc:
            return f"PrefillCudaGraphRunner.{name}'s signature cannot be read: {exc!r}"
        named = [p.name for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)][1:1 + len(reads)]
        if named != list(reads):
            return f"PrefillCudaGraphRunner.{name} takes ({', '.join(named)}) after self, not ({', '.join(reads)})"
    params = inspect.signature(cls.capture_one_shape).parameters
    if "prefix_num_chunks" not in params and not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return "PrefillCudaGraphRunner.capture_one_shape takes no prefix_num_chunks"
    return None


def _patch_prefill_runner(mod) -> None:
    """v62: the engine's prefill CUDA graph runner, observed as the decode runner is (module docstring).
    Nothing here raises into the server: a missing or changed surface skips the patch with its reason, a runner whose
    sizes or backend the patch cannot use is skipped with its reason at its first capture, and every wrapper calls the
    engine's own method with the arguments it was given. Each decision is a `prefill_observer` event (`stage`: `patch`,
    `runner`, `captured`, `stopped`), which the driver copies into the launch record."""
    try:
        reason = _prefill_surface(mod)
    except Exception as exc:  # noqa: BLE001
        reason = f"the prefill runner's surface could not be read: {exc!r}"[:400]
    if reason:
        emit("prefill_observer", stage="patch", installed=False, reason=reason)
        return
    cls = mod.PrefillCudaGraphRunner
    orig_capture, orig_execute, orig_shape_key = cls.capture_one_shape, cls.execute, cls._shape_key
    orig_capture_all = getattr(cls, "capture", None)
    variant_of = getattr(mod, "_chunked_prefix_variant", None)

    def _variant(n):
        """The chunked-prefix variant label, by the engine's own rule (`capture_one_shape` builds its ShapeKey so)."""
        if not n:
            return None
        return variant_of(int(n)) if callable(variant_of) else f"chunked_prefix:{int(n)}"

    def _state(self) -> dict:
        """The runner's observation state, made at its first capture: the engine sets `capture_num_tokens` before it
        captures (and `raw_num_tokens` only after, so that is read at each replay)."""
        st = _pf["runners"].get(id(self))
        if st is None:
            st = {"runner": id(self), "rows": None, "sizes": None, "shared": {}, "bytes": 0, "fresh": 0,
                  "fresh_bytes": 0, "failed": None, "shapes": 0}
            backend = getattr(self, "prefill_backend_name", None)
            try:
                sizes = sorted(int(s) for s in getattr(self, "capture_num_tokens"))
                if not sizes or sizes[0] <= 0:
                    raise ValueError(f"captured sizes {sizes}")
                st["rows"], st["sizes"] = sizes[-1], len(sizes)
            except Exception as exc:  # noqa: BLE001
                st["failed"] = f"the runner's captured sizes could not be read: {exc!r}"[:400]
            if st["failed"] is None and str(backend) not in _PREFILL_BACKENDS:
                st["failed"] = (f"the prefill backend is {backend!r}; only {' and '.join(_PREFILL_BACKENDS)} replay the "
                                f"captured body inside execute (TC_PIECEWISE is not claimed)")
            if st["failed"] is None and getattr(self, "enable_cp_v2_bcg_capture", False):
                st["failed"] = "context-parallel breakable capture is not claimed"
            _pf["runners"][id(self)] = st
            emit("prefill_observer", stage="runner", installed=st["failed"] is None, runner=id(self),
                 rows=st["rows"], sizes=st["sizes"], max_num_tokens=getattr(self, "max_num_tokens", None),
                 backend=str(backend), draft=bool(getattr(getattr(self, "model_runner", None), "is_draft_worker",
                                                          False)), reason=st["failed"])
        return st

    def capture_one_shape(self, size, *args, **kwargs):
        st = None
        try:
            st = _state(self)
            if st["failed"] is None:
                _local.capture_key = ("prefill", id(self), int(size), _variant(kwargs.get("prefix_num_chunks")))
                _local.prefill = st
        except Exception as exc:  # noqa: BLE001
            error("prefill graph capture key", exc)
            _local.capture_key, _local.prefill = None, None
        try:
            return orig_capture(self, size, *args, **kwargs)
        finally:
            _local.capture_key, _local.prefill = None, None
            if st is not None and st["failed"] is None:
                st["shapes"] += 1

    def _shape_key(self, *args, **kwargs):
        sk = orig_shape_key(self, *args, **kwargs)
        try:
            _local.prefill_replay_key = (int(sk.size), getattr(sk, "variant_label", None))
        except Exception:  # noqa: BLE001
            _local.prefill_replay_key = None
        return sk

    def execute(self, forward_batch, *args, **kwargs):
        _local.prefill_replay_key = None
        out = orig_execute(self, forward_batch, *args, **kwargs)
        try:
            st = _pf["runners"].get(id(self))
            replayed = getattr(_local, "prefill_replay_key", None)
            if st is None or st["failed"] is not None or replayed is None:
                return out
            if getattr(getattr(forward_batch, "forward_mode", None), "name", None) == "IDLE":
                return out
            raw = getattr(self, "raw_num_tokens", None)
            if raw is None:
                st["failed"] = "raw_num_tokens was not readable after a replay"
                emit("prefill_observer", stage="stopped", installed=False, runner=id(self), rows=st["rows"],
                     reason=st["failed"])
                return out
            key, rows = ("prefill", id(self), *replayed), int(raw)
        except Exception as exc:  # noqa: BLE001
            error("prefill graph read-back key", exc)
            return out
        _read_back(key, rows, forward_batch, "prefill")
        return out

    cls.capture_one_shape = capture_one_shape
    cls.execute = execute
    cls._shape_key = _shape_key
    if callable(orig_capture_all):
        def capture(self, *args, **kwargs):
            try:
                return orig_capture_all(self, *args, **kwargs)
            finally:
                st = _pf["runners"].get(id(self))
                if st is not None:
                    emit("prefill_observer", stage="captured", installed=st["failed"] is None, runner=id(self),
                         rows=st["rows"], sizes=st["sizes"], shapes=st["shapes"], buffers=len(st["shared"]),
                         bytes=st["bytes"], fresh=st["fresh"], fresh_bytes=st["fresh_bytes"], reason=st["failed"])

        cls.capture = capture
    emit("prefill_observer", stage="patch", installed=True)


def _on_control(c: dict) -> None:
    args = c.get("args") or {}
    value = 0.0 if args.get("zero_attention") else 1.0
    for t in _st["flags"]:
        t.fill_(value)
    if args.get("zero_attention") is not None or _st["flags"]:
        emit("attention_flag", value=value, flags=len(_st["flags"]))


if ON:
    from . import on_control
    after_import("sglang.srt.model_executor.runner.decode_cuda_graph_runner", _patch_graph_runner)
    after_import(PREFILL_RUNNER_MODULE, _patch_prefill_runner)
    on_control(_on_control)
