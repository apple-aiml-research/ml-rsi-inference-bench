#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Verifier-owned recorder for native component execution, inputs, outputs and interventions."""


import json
import os
import sys
import time

_OUT = os.environ.get("INFB_SD3_COMPONENT_RECORD_FILE")


_TAP = os.environ.get("INFB_SD3_COMPONENT_TAP_DIR") if _OUT else None
_HERE = os.path.dirname(os.path.abspath(__file__))

_LOADER_MODULE = "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader"
_PIPELINE_MODULE = "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base"
_TRANSFORMERS_MODULE = "transformers.modeling_utils"
_DIFFUSERS_MODULE = "diffusers.models.modeling_utils"
_ATTN_BACKEND_MODULE = "sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend"
_ATTN_ARM = "ATTN_ARM"
_ATTN_MAX_SIGS = 8
_ATTN_MAX_RERUNS = 96


_MODEL_COMPONENTS = ("text_encoder", "text_encoder_2", "text_encoder_3", "vae", "transformer")
_FOREIGN_PREFIXES = ("transformers.", "diffusers.")
_MAX_EVENTS = 20000
_MAX_MRO = 16
_MAX_TREE = 50000
_MAX_FOREIGN_KINDS = 16


_WHOLE_MODEL_MODULES = ("diffusers.models.transformers", "diffusers.models.autoencoders", "diffusers.models.unets",
                        "diffusers.pipelines", "transformers")
_MAX_WHOLE_KINDS = 64

_state = {"events": 0, "seq": 0, "window": [], "tapped": set()}


_attn = {"depth": 0, "summ": None, "mode": None, "plan": None, "target": -1, "delta": 0.0, "probe_done": False,
         "wrapped": 0}
_TAP_ENCODERS = ("text_encoder", "text_encoder_2", "text_encoder_3")


def _emit(rec):
    """One line per record, in ONE write() on an O_APPEND descriptor, so lines from several processes never
    interleave. O_NOFOLLOW: a symlink planted at the record path is refused, not written through."""
    if not _OUT:
        return
    try:
        _state["events"] += 1
        if _state["events"] > _MAX_EVENTS:
            if _state["events"] == _MAX_EVENTS + 1:
                rec = {"event": "truncated", "max_events": _MAX_EVENTS}
            else:
                return
        rec.setdefault("pid", os.getpid())
        rec.setdefault("t", time.time())
        line = (json.dumps(rec, separators=(",", ":"), default=str) + "\n").encode("utf-8")
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(_OUT, flags, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:  # noqa: BLE001 - the recorder must never take a process down
        pass


def _name(cls):
    return [getattr(cls, "__module__", None) or "", getattr(cls, "__qualname__", None)
            or getattr(cls, "__name__", "?")]


def _mro(cls):
    try:
        return [_name(k) for k in cls.__mro__[:_MAX_MRO]]
    except Exception:  # noqa: BLE001
        return [_name(cls)]


def _is_torch(mod):
    return mod == "torch" or mod.startswith("torch.")


def _is_fsdp_base(k):
    """torch's FSDPModule, recognised by where torch defines it (torch 2.9.1 reports `torch.distributed.fsdp`)."""
    mod = getattr(k, "__module__", "") or ""
    return getattr(k, "__qualname__", None) == "FSDPModule" and mod.startswith("torch.distributed.fsdp")


def _impl(cls):
    """The class that IMPLEMENTS `cls`: the first class in its MRO that is neither torch's nor FSDP2's wrapper.

    FSDP2's `fully_shard` swaps a module's __class__ for `type(f"FSDP{name}", (FSDPModule, cls), {...})`. That
    dict carries no __module__, so Python takes the module of the frame that builds the class: torch's own
    module for a plain class, but `abc` for an ABC-derived one, because ABCMeta.__new__ is the frame that runs
    type.__new__. An ABC-derived encoder can therefore appear as `abc.FSDPCLIPTextModelWithProjection`.
    The wrapper is recognised STRUCTURALLY, never by its module: it
    is the entry directly before FSDPModule in the MRO. torch's own classes (nn.Module, FSDPModule) are
    skipped by module. Must agree with component_probe.impl_of (tested)."""
    mro = list(getattr(cls, "__mro__", (cls,)))
    for i, k in enumerate(mro):
        mod = getattr(k, "__module__", "") or ""
        if mod == "builtins":
            break
        if _is_torch(mod):
            continue
        if i + 1 < len(mro) and _is_fsdp_base(mro[i + 1]):
            continue
        return k
    return cls


def _unwrap(obj):
    """Look through torch's wrapper OBJECTS (torch.compile's OptimizedModule keeps the model in `_orig_mod`)."""
    chain = []
    for _ in range(4):
        mod = getattr(type(obj), "__module__", "") or ""
        inner = getattr(obj, "_orig_mod", None) if _is_torch(mod) else None
        if inner is None:
            break
        chain.append(_name(type(obj)))
        obj = inner
    return obj, chain


def _whole_base(cls):
    """`module.qualname` of the first class in cls's MRO defined in a whole-model module, or None. Never raises."""
    try:
        for k in getattr(cls, "__mro__", (cls,)):
            mod = getattr(k, "__module__", "") or ""
            if any(mod == p or mod.startswith(p + ".") for p in _WHOLE_MODEL_MODULES):
                return mod + "." + (getattr(k, "__qualname__", None) or getattr(k, "__name__", "?"))
    except Exception:  # noqa: BLE001
        pass
    return None


def _defining_function(cls, meth):
    """[module, qualname] of the FUNCTION a class attribute holds, looking through staticmethod/classmethod and
    functools.wraps chains. `forward = <diffusers' forward>` on an SGLang class is owned by that class but defined by
    diffusers; this is what tells them apart. None when there is no plain function to read."""
    try:
        fn = cls.__dict__[meth]
        fn = getattr(fn, "__func__", fn)
        for _ in range(8):
            inner = getattr(fn, "__wrapped__", None)
            if inner is None:
                break
            fn = inner
        mod, qual = getattr(fn, "__module__", None), getattr(fn, "__qualname__", None)
        return [mod, qual] if isinstance(mod, str) and isinstance(qual, str) else None
    except Exception:  # noqa: BLE001
        return None


def _objinfo(obj, tree):
    if obj is None:
        return None
    inner, wrappers = _unwrap(obj)
    cls = type(inner)
    info = {"raw": _name(type(obj)), "wrappers": wrappers, "mro": _mro(cls)}
    owners, fns = {}, {}
    for meth in ("forward", "encode", "decode"):
        for k in getattr(cls, "__mro__", ()):
            if meth in getattr(k, "__dict__", {}):
                owners[meth] = _name(k)
                fn = _defining_function(k, meth)
                if fn is not None:
                    fns[meth] = fn
                break
    info["method_owner"] = owners
    info["method_fn"] = fns
    walk = getattr(inner, "named_modules", None)
    if tree and callable(walk):
        foreign, n, truncated = {}, 0, False
        whole, whole_via = {}, {}
        try:
            for _, m in walk():
                n += 1
                if n > _MAX_TREE:
                    truncated = True
                    break
                k = _impl(type(m))
                if (getattr(k, "__module__", "") or "").startswith(_FOREIGN_PREFIXES):
                    key = k.__module__ + "." + k.__qualname__
                    foreign[key] = foreign.get(key, 0) + 1
                wb = _whole_base(type(m))
                if wb is not None:
                    whole[wb] = whole.get(wb, 0) + 1
                    if wb not in whole_via:
                        whole_via[wb] = (getattr(k, "__module__", "") or "") + "." + (
                            getattr(k, "__qualname__", None) or "?")
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            info["tree_error"] = f"{type(exc).__name__}: {exc}"[:200]
        top = sorted(foreign.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = sorted(whole)[:_MAX_WHOLE_KINDS]
        info["tree"] = {"n_modules": n, "truncated": truncated, "foreign_total": sum(foreign.values()),
                        "foreign": dict(top[:_MAX_FOREIGN_KINDS]), "foreign_kinds": len(foreign),
                        "whole": {w: whole[w] for w in kept}, "whole_kinds": len(whole),
                        "whole_via": {w: whole_via[w] for w in kept}}
    return info


def _window():
    return _state["window"][-1] if _state["window"] else None


def _record_instantiation(self, via):
    try:
        cls = type(self)


        _emit({"event": "instantiated", "via": via, "cls": _name(cls), "mro": _mro(cls),
               "whole": _whole_base(cls), "component": _window()})
    except Exception:  # noqa: BLE001
        pass


def _hook_init(cls, via, target, module):
    """Wrap `cls.__init__` so that EVERY subclass constructed records its concrete class first."""
    import functools

    own = cls.__dict__.get("__init__")
    if getattr(own, "_infb_recorded", False):
        _emit({"event": "patch_skipped", "target": target, "why": "already wrapped"})
        return
    if own is not None:
        @functools.wraps(own)
        def __init__(self, *args, **kwargs):
            _record_instantiation(self, via)
            return own(self, *args, **kwargs)
    else:
        inherited = cls.__init__

        def __init__(self, *args, **kwargs):
            _record_instantiation(self, via)
            return super(cls, self).__init__(*args, **kwargs)


        functools.update_wrapper(__init__, inherited)
        __init__.__qualname__ = cls.__qualname__ + ".__init__"
    __init__._infb_recorded = True
    cls.__init__ = __init__
    _emit({"event": "patched", "target": target, "file": getattr(module, "__file__", None),
           "own_init": own is not None})


def _patch_transformers(module):
    cls = getattr(module, "PreTrainedModel", None)
    if not isinstance(cls, type):
        _emit({"event": "patch_skipped", "target": _TRANSFORMERS_MODULE + ".PreTrainedModel.__init__",
               "why": "no PreTrainedModel class"})
        return
    _hook_init(cls, "transformers.PreTrainedModel", _TRANSFORMERS_MODULE + ".PreTrainedModel.__init__", module)


def _patch_diffusers(module):
    cls = getattr(module, "ModelMixin", None)
    if not isinstance(cls, type):
        _emit({"event": "patch_skipped", "target": _DIFFUSERS_MODULE + ".ModelMixin.__init__",
               "why": "no ModelMixin class"})
        return
    _hook_init(cls, "diffusers.ModelMixin", _DIFFUSERS_MODULE + ".ModelMixin.__init__", module)


def _patch_loader(module):
    import functools

    target = _LOADER_MODULE + ".PipelineComponentLoader.load_component"
    cls = getattr(module, "PipelineComponentLoader", None)
    raw = cls.__dict__.get("load_component") if isinstance(cls, type) else None
    original = raw.__func__ if isinstance(raw, staticmethod) else None
    if original is None or getattr(original, "_infb_recorded", False):
        _emit({"event": "patch_skipped", "target": target,
               "why": "already wrapped" if original is not None else "no static PipelineComponentLoader.load_component"})
        return

    @functools.wraps(original)
    def load_component(*args, **kwargs):
        name = kwargs.get("component_name", args[0] if args else None)
        library = kwargs.get("transformers_or_diffusers", args[2] if len(args) > 2 else None)
        _state["seq"] += 1
        seq = _state["seq"]
        _emit({"event": "load_begin", "component": name, "library": library, "seq": seq})
        _state["window"].append(name)
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            if _state["window"]:
                _state["window"].pop()
            _emit({"event": "load_end", "component": name, "seq": seq, "ok": False,
                   "error": f"{type(exc).__name__}: {exc}"[:300]})
            raise
        if _state["window"]:
            _state["window"].pop()
        try:
            comp = result[0] if isinstance(result, tuple) and result else result
            _emit({"event": "load_end", "component": name, "seq": seq, "ok": True,
                   "obj": _objinfo(comp, tree=False)})
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "record_error", "where": "load_end", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return result

    load_component._infb_recorded = True
    cls.load_component = staticmethod(load_component)
    _emit({"event": "patched", "target": target, "file": getattr(module, "__file__", None)})


def _armed():
    try:
        return bool(_TAP) and os.path.exists(os.path.join(_TAP, "ARM"))
    except Exception:  # noqa: BLE001
        return False


def _cpu(x):
    """A tensor as a detached, contiguous CPU copy; lists/tuples elementwise; anything else unchanged."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        value = x.detach().to("cpu").contiguous()


        return value if type(value) is torch.Tensor else value.as_subclass(torch.Tensor)
    if isinstance(x, (list, tuple)):
        return [_cpu(v) for v in x]
    return x


def _saveable(x):
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        return _cpu(x)
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, (list, tuple)) and all(torch is not None and isinstance(v, torch.Tensor) for v in x):
        return [_cpu(v) for v in x]
    return repr(x)[:200]


def _dit_inputs(module, args, kwargs):
    """Read the executed call without requiring parameter names or argument order.

    Bind inspectable names first, including transparent Torch/decorator wrappers.
    For renamed or opaque inputs, SD3's image latent, token conditioning, pooled
    conditioning and timestep have distinct shapes. Use that only when each role
    has one compatible object and the batch sizes agree. Never guess by position
    or pick the first of several matches. Callers report ambiguous/missing input
    evidence without modifying the candidate call or awarding it credit.
    """
    import inspect

    forward = module.forward
    try:
        signature = inspect.signature(forward)

        inner, _ = _unwrap(module)
        if inner is not module and all(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                                       for p in signature.parameters.values()):
            signature = inspect.signature(inner.forward)
    except (TypeError, ValueError):
        values = dict(kwargs)
    else:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = {}
        for name, value in bound.arguments.items():
            kind = signature.parameters[name].kind
            if kind == inspect.Parameter.VAR_KEYWORD:
                for key, extra in value.items():


                    values.setdefault(key, extra)
            elif kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            else:
                values[name] = value
    names = ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep")
    if all(name in values for name in names):
        return values
    torch = sys.modules.get("torch")
    if torch is None:
        raise ValueError("DiT inputs need inspectable names or unambiguous tensor shapes")


    pool = {id(value): value for value in (*args, *kwargs.values())}
    used = {id(values[name]) for name in names if name in values}

    def resolve(name, compatible):
        if name in values:
            return values[name]
        candidates = [v for key, v in pool.items() if key not in used and compatible(v)]
        if len(candidates) != 1:
            why = "ambiguous" if candidates else "missing or incompatible"
            raise ValueError(f"DiT {name}: {why} input evidence ({len(candidates)} shape matches)")
        values[name] = candidates[0]
        used.add(id(candidates[0]))
        return candidates[0]


    x = resolve("hidden_states", lambda v: isinstance(v, torch.Tensor) and v.is_floating_point() and
                ((v.ndim == 4 and v.shape[1] == 16) or
                 (v.ndim == 5 and v.shape[1] == 16 and v.shape[2] == 1) or
                 (v.ndim == 3 and v.shape[-1] == 64)) and all(d > 0 for d in v.shape))
    if not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[0] < 1:
        raise ValueError("DiT hidden_states: missing or incompatible batch dimension")
    rows = x.shape[0]
    resolve("encoder_hidden_states", lambda v: isinstance(v, torch.Tensor) and v.is_floating_point() and
            v.ndim == 3 and v.shape[0] == rows and v.shape[-1] == 4096 and v.shape[1] > 0)
    resolve("pooled_projections", lambda v: isinstance(v, torch.Tensor) and v.is_floating_point() and
            v.ndim == 2 and v.shape == (rows, 2048))


    step = resolve("timestep", lambda v: (isinstance(v, torch.Tensor) and v.ndim <= 1 and
                                          not v.is_complex() and v.dtype != torch.bool) or
                   _dit_scalar_timestep(v))
    if isinstance(step, torch.Tensor) and step.numel() not in (1, rows):
        raise ValueError("DiT timestep: incompatible batch dimension")
    return values


def _dit_scalar_timestep(value):
    import math
    import numbers
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _dit_capture_inputs(module, args, kwargs):
    """Canonical component-replay payload, without changing any runtime operand.

    A scalar timestep and a one-element tensor describe the same call. Diffusers'
    replay expects a 1-D tensor; normalise only the saved evidence, retaining the
    scalar's precision. All other inputs keep their original shape and dtype.
    """
    values = _dit_inputs(module, args, kwargs)
    torch = sys.modules.get("torch")
    ts = values["timestep"]
    if torch is not None:
        if _dit_scalar_timestep(ts):
            values["timestep"] = torch.as_tensor(ts).reshape(1)
            if isinstance(ts, float):
                values["timestep"] = torch.tensor([ts], dtype=torch.float64)
        elif isinstance(ts, torch.Tensor) and ts.ndim == 0:
            values["timestep"] = ts.reshape(1)
    return values


def _tap_save(name, payload):
    """Save one captured call as `<tap>/<name>.pt` (torch.save, atomic) and record that it happened."""
    torch = sys.modules.get("torch")
    if torch is None or name in _state["tapped"]:
        return
    _state["tapped"].add(name)
    path = os.path.join(_TAP, name + ".pt")
    try:
        torch.save(payload, path + ".tmp")
        os.replace(path + ".tmp", path)
        _emit({"event": "tap", "name": name, "file": path,
               "shapes": {k: list(v.shape) for k, v in payload.items() if hasattr(v, "shape")},
               "dtypes": {k: str(v.dtype) for k, v in payload.items() if hasattr(v, "dtype")}})
    except Exception as exc:  # noqa: BLE001 - a failed capture loses the capture, never the request
        _emit({"event": "tap_error", "name": name, "error": f"{type(exc).__name__}: {exc}"[:300]})


def _encoder_outputs(out):
    """What an encoder returned, by the attribute names SGLang's BaseEncoderOutput and transformers share."""
    d = {}
    for key in ("last_hidden_state", "pooler_output", "text_embeds"):
        v = getattr(out, key, None)
        if v is not None and hasattr(v, "shape"):
            d[key] = _cpu(v)
    hs = getattr(out, "hidden_states", None)
    if isinstance(hs, (list, tuple)) and len(hs) >= 2 and hasattr(hs[-2], "shape"):
        d["hidden_states_m2"] = _cpu(hs[-2])
    return d


def _install_taps(mods):
    """Pass-through taps on the three text encoders, the DiT and the VAE's decode.

    Each saves the FIRST call made while `<tap>/ARM` exists and then goes quiet: every text encoder's positive
    prompt pass, the DiT's first forward (step 1, the conditional pass -- SGLang runs the CFG branches as
    separate passes, the positive first), and the final decode. Every tap returns nothing and catches its own
    errors, so the forward's result is untouched; they stay inert through every scored request, because the
    verifier creates ARM only for one dedicated request after those.
    """
    installed = []

    def _hook_for(name, pack):
        def hook(module, args, kwargs, output):
            try:
                if name not in _state["tapped"] and _armed():
                    _tap_save(name, pack(module, args, kwargs, output))
            except Exception as exc:  # noqa: BLE001
                _emit({"event": "tap_error", "name": name, "error": f"{type(exc).__name__}: {exc}"[:300]})
            return None
        return hook

    def _pack_encoder(module, args, kwargs, output):
        ids = kwargs.get("input_ids", args[0] if args else None)
        return {"input_ids": _cpu(ids), **_encoder_outputs(output)}

    def _pack_dit(module, args, kwargs, output):
        out = output[0] if isinstance(output, (list, tuple)) else getattr(output, "sample", output)
        payload = {k: _saveable(v) for k, v in _dit_capture_inputs(module, args, kwargs).items()}
        payload["args"] = [_saveable(v) for v in args]
        payload["output"] = _cpu(out)
        return payload

    for name, pack in [(n, _pack_encoder) for n in _TAP_ENCODERS] + [("transformer", _pack_dit)]:
        m = mods.get(name)
        reg = getattr(m, "register_forward_hook", None)
        if not callable(reg):
            continue
        try:
            reg(_hook_for(name, pack), with_kwargs=True)
            installed.append(name)
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "tap_error", "name": name, "error": f"{type(exc).__name__}: {exc}"[:300]})
    vae = mods.get("vae")
    orig = getattr(vae, "decode", None)
    if callable(orig):
        def decode(*args, **kwargs):
            out = orig(*args, **kwargs)
            try:
                if "vae_decode" not in _state["tapped"] and _armed():
                    z = kwargs.get("z", args[0] if args else None)
                    sample = getattr(out, "sample", out)
                    if isinstance(sample, (list, tuple)):
                        sample = sample[0]
                    _tap_save("vae_decode", {"z": _cpu(z), "sample": _cpu(sample)})
            except Exception as exc:  # noqa: BLE001
                _emit({"event": "tap_error", "name": "vae_decode", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return out
        try:
            vae.decode = decode
            installed.append("vae_decode")
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "tap_error", "name": "vae_decode", "error": f"{type(exc).__name__}: {exc}"[:300]})
    _emit({"event": "tap_installed", "components": installed, "dir": _TAP})


def _impl_name(obj):
    k = type(obj)
    return (getattr(k, "__module__", "") or "") + "." + (getattr(k, "__qualname__", None) or k.__name__)


def _shape(t):
    try:
        return [int(x) for x in t.shape]
    except Exception:  # noqa: BLE001
        return None


_CENSUS = bool(_OUT and os.environ.get("INFB_SD3_PARAM_CENSUS"))
_CENSUS_MAX_PARAMS = 4000


def _rank():
    """This process's distributed rank and world size: torch.distributed when it is initialised, else the env the
    launcher set. Never imports torch (the recorder is stdlib-only); reads the module only if the process has it."""
    out = {"pid": os.getpid()}
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        if os.environ.get(key) is not None:
            out[key.lower()] = os.environ.get(key)
    try:
        dist = getattr(sys.modules.get("torch"), "distributed", None)
        if dist is not None and dist.is_available() and dist.is_initialized():
            out["dist_rank"], out["dist_world"] = int(dist.get_rank()), int(dist.get_world_size())
    except Exception:  # noqa: BLE001
        pass
    return out


def _rank_int():
    """This process's distributed rank as an int (torch.distributed's, else the launcher's RANK, else 0)."""
    r = _rank()
    try:
        return int(r.get("dist_rank", r.get("rank", 0)))
    except Exception:  # noqa: BLE001
        return 0


def _local(p):
    """Return the parameter tensor held by this rank, using a DTensor's local shard.

    FSDP2 wraps the encoders when hsdp_shard_dim > 1. A DTensor's shape is global;
    `to_local()` returns its local tensor without communication.
    """
    to_local = getattr(p, "to_local", None)
    if callable(to_local):
        try:
            return to_local(), True
        except Exception:  # noqa: BLE001 - an unreadable shard is recorded as the parameter itself
            pass
    return p, False


def _census(mods):
    """One `param_census` line per model component the pipeline holds: every parameter's name and LOCAL shape on
    THIS rank, plus the totals. A TP-sharded weight shows half its checkpoint rows or columns on each rank, and so does
    an FSDP2 shard (a DTensor: its global shape goes to `global_shapes`); a replicated one shows the full shape on both.
    Storage only: whether the COMPUTE is split is the compute census's question."""
    for key in _MODEL_COMPONENTS:
        obj = mods.get(key)
        named = getattr(obj, "named_parameters", None)
        if not callable(named):
            continue
        try:
            shapes, global_shapes, numel, dtypes, n, n_dt = {}, {}, 0, set(), 0, 0
            for name, p in named():
                n += 1
                loc, is_dt = _local(p)
                shp = _shape(loc)
                if is_dt:
                    n_dt += 1
                    if len(global_shapes) < _CENSUS_MAX_PARAMS:
                        global_shapes[str(name)] = _shape(p)
                if shp is not None:
                    c = 1
                    for d in shp:
                        c *= d
                    numel += c
                dtypes.add(str(getattr(p, "dtype", "")))
                if len(shapes) < _CENSUS_MAX_PARAMS:
                    shapes[str(name)] = shp
            _emit({"event": "param_census", "component": key, "cls": _name(type(obj)), "rank": _rank(),
                   "n_params": n, "numel": numel, "dtypes": sorted(dtypes), "shapes": shapes,
                   "n_dtensor": n_dt, "global_shapes": global_shapes, "truncated": n > _CENSUS_MAX_PARAMS})
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "record_error", "where": f"param_census {key}", "error": f"{type(exc).__name__}: {exc}"[:300]})


_PAR = bool(_OUT and os.environ.get("INFB_SD3_PARALLEL_EVIDENCE"))
try:
    _DROP_RANK = int(os.environ["INFB_SD3_RANK_DROP"]) if _PAR and os.environ.get("INFB_SD3_RANK_DROP") else None
except ValueError:
    _DROP_RANK = None


def _parse_stall(raw):
    """INFB_SD3_RANK_STALL=<rank>:<seconds> -> (rank, seconds), else None (malformed or dormant)."""
    try:
        rank, _, secs = str(raw).partition(":")
        rank, secs = int(rank), float(secs)
        return (rank, secs) if rank >= 0 and secs > 0 else None
    except (TypeError, ValueError):
        return None


_STALL = _parse_stall(os.environ.get("INFB_SD3_RANK_STALL")) if _PAR and os.environ.get("INFB_SD3_RANK_STALL") \
    else None


_FAULT_DIR = os.environ.get("INFB_SD3_FAULT_DIR") if _OUT else None
_FAULT_ARM = "FAULT"


def _maybe_fault(rows=None, phase="before_dit"):
    """If `<fault dir>/FAULT` exists, claim it with an atomic rename -- exactly one forward fires per arm, whichever
    process gets there first -- and raise RuntimeError carrying the marker and the arm's text. The one place this
    recorder raises on purpose: it is the T2 verifier's injected execution failure, armed for one request of the
    verifier's own engine, never during a scored request."""
    path = os.path.join(_FAULT_DIR, _FAULT_ARM)
    try:
        if not os.path.exists(path):
            return
        claimed = f"{path}.fired.{os.getpid()}"
        os.rename(path, claimed)
        with open(claimed, encoding="utf-8") as fh:
            text = fh.read().strip()[:200]
    except OSError:
        return
    _emit({"event": "fault_fired", "text": text, "rank": _rank(), "rows": rows, "phase": phase})
    raise RuntimeError(f"INFB injected fault: {text or 'the verifier failed this DiT forward'}")


_SLOW_ARM = "SLOW"
_slow = {"s": 0.0}
_prec_now = {"v": None, "env_done": False}


def _maybe_slow():
    """While `<fault dir>/SLOW` exists, sleep the seconds it holds (0-5) before this DiT forward, and remember them for
    the forward's `dit_call` line. Never raises: an unreadable arm is no delay."""
    s = 0.0
    try:
        with open(os.path.join(_FAULT_DIR, _SLOW_ARM), encoding="utf-8") as fh:
            s = min(max(float(fh.read().strip() or 0.0), 0.0), 5.0)
    except (OSError, ValueError):
        s = 0.0
    _slow["s"] = s
    if s > 0.0:
        time.sleep(s)
_PAR_COLL = ("all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter_tensor", "all_to_all_single",
             "all_to_all", "broadcast")
_PAR_FT = ("all_to_all_single", "all_gather_tensor", "all_reduce", "reduce_scatter_tensor")
_PAR_INPUT = {"all_gather": (1, "tensor"), "all_gather_into_tensor": (1, "input_tensor"),
              "reduce_scatter_tensor": (1, "input"), "all_to_all_single": (1, "input")}
_PAR_MAX_CALLS = 2000
_par = {"depth": 0, "coll": None, "installed": False, "drop": False, "calls": None, "census_done": False,
        "hooks": [], "x": None, "stall": False, "stalled": False, "late": None}


def _par_zero(name, functional, a, k):
    """This rank's contribution to one collective replaced by zeros (the rank-drop probe)."""
    a = list(a)

    def z(t):
        return t.new_zeros(t.shape) if hasattr(t, "new_zeros") else t

    if functional:
        if a:
            a[0] = z(a[0])
        elif "self" in k:
            k["self"] = z(k["self"])
    elif name == "all_reduce":
        t = a[0] if a else k.get("tensor")
        if hasattr(t, "zero_"):
            t.zero_()
    elif name in _PAR_INPUT:
        i, key = _PAR_INPUT[name]
        if len(a) > i:
            a[i] = z(a[i])
        elif key in k:
            k[key] = z(k[key])
    elif name == "all_to_all":
        lst = a[1] if len(a) > 1 else k.get("input_tensor_list")
        if lst is not None:
            new = [z(t) for t in lst]
            if len(a) > 1:
                a[1] = new
            else:
                k["input_tensor_list"] = new
    return tuple(a), k


def _par_install():
    """Wrap the collectives once, at the first DiT forward (torch.distributed is loaded by then)."""
    if _par["installed"]:
        return
    _par["installed"] = True
    torch = sys.modules.get("torch")
    dist = getattr(torch, "distributed", None) if torch is not None else None
    if dist is None:
        return
    try:
        me = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else None
    except Exception:  # noqa: BLE001
        me = None
    _par["drop"] = _DROP_RANK is not None and me == _DROP_RANK
    _par["stall"] = _STALL is not None and me == _STALL[0]
    ft = sys.modules.get("torch.distributed._functional_collectives")
    targets = [(dist, n, n, False) for n in _PAR_COLL]
    targets += [(ft, n, "functional." + n, True) for n in _PAR_FT] if ft is not None else []
    for mod, name, key, functional in targets:
        orig = getattr(mod, name, None)
        if orig is None or getattr(orig, "_infb_par", False):
            continue

        def wrapped(*a, _o=orig, _n=name, _key=key, _f=functional, **k):
            coll = _par["coll"]
            if _par["depth"] > 0 and coll is not None:
                coll[_key] = coll.get(_key, 0) + 1
                if _par["drop"] and _n != "broadcast":
                    try:
                        a, k = _par_zero(_n, _f, a, k)
                    except Exception:  # noqa: BLE001 - the recorder must never break the collective
                        pass
                late = _par.get("late")
                if late is not None and _n != "broadcast":
                    try:
                        a, k = _par_scale(_n, _f, a, k, late["scale"])
                        late["collectives"] = late.get("collectives", 0) + 1
                    except Exception:  # noqa: BLE001 - the recorder must never break the collective
                        pass
                if _par["stall"] and not _par["stalled"] and _n != "broadcast":
                    _par["stalled"] = True
                    _emit({"event": "stall_begin", "rank": _rank(), "op": _key, "seconds": _STALL[1],
                           "t": time.time()})
                    time.sleep(_STALL[1])
                    _emit({"event": "stall_end", "rank": _rank(), "op": _key, "t": time.time()})
            return _o(*a, **k)

        wrapped._infb_par = True
        try:
            setattr(mod, name, wrapped)
        except Exception:  # noqa: BLE001
            pass
    _emit({"event": "patched", "target": "parallel_evidence collectives", "file": getattr(dist, "__file__", None),
           "drop_rank": _DROP_RANK, "drop_here": _par["drop"], "stall": list(_STALL) if _STALL else None,
           "stall_here": _par["stall"], "rank": _rank()})


def _par_census_begin(module):
    """Forward hooks on every submodule with a 2-D weight, for this one forward: the executed matmul widths."""
    calls = []
    _par["calls"] = calls
    named = getattr(module, "named_modules", None)
    if not callable(named):
        return
    for name, sub in named():
        shp = _shape(getattr(sub, "weight", None))
        reg = getattr(sub, "register_forward_hook", None)
        if shp is None or len(shp) != 2 or not callable(reg):
            continue

        def hook(m, args, out, _name=name):
            try:
                if len(calls) >= _PAR_MAX_CALLS:
                    return None
                xs = _shape(args[0]) if args else None
                ys = _shape(out[0] if isinstance(out, (tuple, list)) else out)
                if xs and ys and xs[-1] > 0:
                    n = 1
                    for d in xs:
                        n *= d
                    calls.append([str(_name), n // xs[-1], xs[-1], ys[-1]])
            except Exception:  # noqa: BLE001
                pass
            return None

        try:
            _par["hooks"].append(reg(hook))
        except Exception:  # noqa: BLE001
            pass


def _par_census_end():
    calls = _par["calls"]
    if calls is None or _par["census_done"]:
        return
    _par["census_done"] = True
    for h in _par["hooks"]:
        try:
            h.remove()
        except Exception:  # noqa: BLE001
            pass
    _par["hooks"] = []
    _emit({"event": "compute_census", "rank": _rank(), "x": _par["x"],
           "macs": sum(r * i * o for _, r, i, o in calls), "n_calls": len(calls),
           "truncated": len(calls) >= _PAR_MAX_CALLS, "calls": calls[:600]})


def _par_begin(module, x=None):
    _par_install()


    arm = os.environ.get("INFB_SD3_PARALLEL_DROP_FILE")
    if arm:
        try:
            with open(arm, encoding="utf-8") as fh:
                target = int(fh.read().strip())
        except (OSError, ValueError):
            target = None
        me = _rank().get("dist_rank")
        _par["drop"] = target is not None and me == target
        if _par["drop"]:
            _emit({"event": "rank_drop_active", "rank": _rank(), "drop_here": True, "drop_rank": target})
    if _par["depth"] == 0:
        _par["late"] = None
        late = _late_arm(_batch.get("now"))
        if late is not None and late["kind"] == "collective":
            _par["late"] = {"kind": "collective", "scale": 1.0 + float(late["delta"]), "delta": float(late["delta"]),
                            "t_max": float(late["t_max"]), "rank": late.get("rank"), "key": late.get("key")}
            _emit({"event": "late_arm_applied", "rank": _rank(), "n": (_batch.get("now") or {}).get("n"),
                   "timesteps": (_batch.get("now") or {}).get("timesteps"), **_par["late"]})
    _par["depth"] += 1
    if _par["depth"] == 1:
        _par["coll"] = {}
        if os.environ.get("INFB_SD3_COMPUTE_EVERY_FORWARD"):
            _par["census_done"] = False
            _par["calls"] = None
        if not _par["census_done"] and _par["calls"] is None:
            _par["x"] = x
            _par_census_begin(module)


def _par_end():
    if _par["depth"] <= 0:
        return None
    _par["depth"] -= 1
    if _par["depth"] > 0:
        return None
    coll, _par["coll"] = _par["coll"], None
    _par_census_end()
    return coll


def _attn_plan():
    """The probe's plan while `<tap>/ATTN_ARM` exists, else None. Read once per DiT forward (one stat)."""
    if not _TAP:
        return None
    path = os.path.join(_TAP, _ATTN_ARM)
    try:
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            plan = json.load(fh)
        return plan if isinstance(plan, dict) else {}
    except Exception:  # noqa: BLE001 - an unreadable plan is a default plan, never a crash
        return {}


def _attn_begin():
    st = _attn
    st["depth"] += 1
    if st["depth"] == 1:
        st["summ"] = {"n": 0, "impls": {}, "sigs": {}, "k_seq": {}, "q_seq": {}}
        if st["mode"] is None and not st["probe_done"]:
            plan = _attn_plan()
            if plan is not None:
                st["mode"], st["plan"] = "capture", plan


def _attn_end():
    st = _attn
    if st["depth"] <= 0:
        return None
    st["depth"] -= 1
    if st["depth"] > 0:
        return None
    summ, st["summ"] = st["summ"], None
    if summ is None:
        return None
    return {"n": summ["n"], "impls": summ["impls"], "sigs": list(summ["sigs"].values()),
            "k_seq": summ["k_seq"], "q_seq": summ["q_seq"], "probe": st["mode"] == "capture"}


def _attn_capture(impl, idx, q, k, v, out):
    """Save rows (evenly spaced over the whole sequence, so it does not depend on the token order) and three heads
    of q and of the output, and those heads of k and v in full, as float32 .npz. Never raises."""
    try:
        np = sys.modules.get("numpy")
        if np is None:
            raise RuntimeError("numpy is not loaded in this process")
        plan = _attn["plan"] or {}
        n_rows = int(plan.get("rows", 64))
        s_q, heads_n = int(q.shape[1]), int(q.shape[2])
        rows = sorted({round(i * (s_q - 1) / max(n_rows - 1, 1)) for i in range(n_rows)})
        heads = sorted({0, heads_n // 2, heads_n - 1})

        def sub(t, r=None):
            x = t[:, r] if r is not None else t
            return x[:, :, heads].detach().float().cpu().numpy()

        path = os.path.join(_TAP, f"attn_call_{idx}.npz")
        np.savez(path + ".tmp.npz", q_rows=sub(q, rows), k=sub(k), v=sub(v), out_rows=sub(out, rows),
                 rows=np.asarray(rows), heads=np.asarray(heads), q_shape=np.asarray(_shape(q)),
                 k_shape=np.asarray(_shape(k)))
        os.replace(path + ".tmp.npz", path)
        _emit({"event": "attn_capture", "call": idx, "file": path, "impl": _impl_name(impl),
               "q_shape": _shape(q), "k_shape": _shape(k), "out_shape": _shape(out),
               "softmax_scale": getattr(impl, "softmax_scale", None), "causal": getattr(impl, "causal", None),
               "rows": len(rows), "heads": heads})
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "attn_probe_error", "where": f"capture {idx}", "error": f"{type(exc).__name__}: {exc}"[:300]})


def _wrap_impl_forward(cls):
    """Pass-through wrapper on one AttentionImpl subclass's OWN forward (inherited ones are already wrapped)."""
    import functools

    fn = cls.__dict__.get("forward")
    if not callable(fn) or getattr(fn, "_infb_recorded", False):
        return

    @functools.wraps(fn)
    def forward(self, *args, **kwargs):
        st = _attn
        summ = st["summ"]
        if st["depth"] <= 0 or summ is None:
            if _PW and _pw["stack"]:
                out = fn(self, *args, **kwargs)
                try:
                    _pw_impl_note(self, args[0] if len(args) > 0 else kwargs.get("query"),
                                  args[1] if len(args) > 1 else kwargs.get("key"),
                                  args[2] if len(args) > 2 else kwargs.get("value"), out)
                except Exception:  # noqa: BLE001
                    pass
                return out
            return fn(self, *args, **kwargs)
        idx = summ["n"]
        summ["n"] = idx + 1
        q = args[0] if len(args) > 0 else kwargs.get("query")
        k = args[1] if len(args) > 1 else kwargs.get("key")
        v = args[2] if len(args) > 2 else kwargs.get("value")
        try:
            name = _impl_name(self)
            summ["impls"][name] = summ["impls"].get(name, 0) + 1
            qs, ks = _shape(q), _shape(k)
            sig = [qs, ks, name, getattr(self, "softmax_scale", None), getattr(self, "causal", None)]
            key = json.dumps(sig, default=str)
            if key not in summ["sigs"] and len(summ["sigs"]) < _ATTN_MAX_SIGS:
                summ["sigs"][key] = sig
            if ks and len(ks) >= 2:
                summ["k_seq"][str(ks[1])] = summ["k_seq"].get(str(ks[1]), 0) + 1
            if qs and len(qs) >= 2:
                summ["q_seq"][str(qs[1])] = summ["q_seq"].get(str(qs[1]), 0) + 1
        except Exception:  # noqa: BLE001
            pass
        out = fn(self, *args, **kwargs)
        if _PW and _pw["stack"]:
            try:
                _pw_impl_note(self, q, k, v, out)
            except Exception:  # noqa: BLE001
                pass
        mode = st["mode"]
        if mode is None or isinstance(out, tuple):
            return out
        if mode == "capture" and idx in set((st["plan"] or {}).get("capture", ())):
            _attn_capture(self, idx, q, k, v, out)
        elif mode == "perturb" and idx == st["target"]:
            return out * (1.0 + st["delta"])
        return out

    forward._infb_recorded = True
    cls.forward = forward
    _attn["wrapped"] += 1


def _patch_attention_backend(module):
    """Wrap forward on every AttentionImpl subclass: the ones already defined, and each one defined later."""
    base = getattr(module, "AttentionImpl", None)
    target = _ATTN_BACKEND_MODULE + ".AttentionImpl"
    if not isinstance(base, type):
        _emit({"event": "patch_skipped", "target": target, "why": "no AttentionImpl class"})
        return
    if getattr(base, "_infb_attn_recorded", False):
        _emit({"event": "patch_skipped", "target": target, "why": "already wrapped"})
        return
    prev = base.__dict__.get("__init_subclass__")

    def __init_subclass__(cls, **kwargs):
        if prev is not None:
            prev.__func__(cls, **kwargs)
        else:
            super(base, cls).__init_subclass__(**kwargs)
        _wrap_impl_forward(cls)

    base.__init_subclass__ = classmethod(__init_subclass__)
    base._infb_attn_recorded = True
    todo = list(base.__subclasses__())
    while todo:
        k = todo.pop()
        _wrap_impl_forward(k)
        todo.extend(k.__subclasses__())
    _emit({"event": "patched", "target": target, "file": getattr(module, "__file__", None)})


def _dit_out(output):
    return output[0] if isinstance(output, (list, tuple)) else getattr(output, "sample", output)


def _rel(a, b):
    d = float((a.double() - b.double()).norm())
    n = float(b.double().norm())
    return d / max(n, 1e-30)


def _attn_probe(module, args, kwargs, output, summ):
    """The armed forward has returned: re-run the DiT on the same inputs, unchanged once and once per attention call
    with that call's output scaled by (1 + delta); record how far each re-run's output moved. Never raises."""
    st = _attn
    st["probe_done"] = True
    plan = st["plan"] or {}
    delta = float(plan.get("delta", 0.5))
    rec = {"event": "attn_probe", "n": summ.get("n"), "impls": summ.get("impls"), "delta": delta}
    fwd = getattr(module, "forward", None)
    try:
        base = _dit_out(output)
        n = int(summ.get("n") or 0)

        def rerun(mode, target=-1):
            st["mode"], st["target"], st["delta"] = mode, target, delta
            st["depth"] += 1
            st["summ"] = {"n": 0, "impls": {}, "sigs": {}, "k_seq": {}, "q_seq": {}}
            try:
                out = _dit_out(fwd(*args, **kwargs))
                return out, st["summ"]["n"]
            finally:
                st["depth"] -= 1
                st["summ"], st["mode"], st["target"] = None, None, -1

        t0 = time.time()
        rep, rep_n = rerun("plain")
        rec["repeat_rel"] = _rel(rep, base)
        rec["repeat_calls"] = rep_n
        per = []
        for j in range(min(n, _ATTN_MAX_RERUNS - 1)):
            oj, _ = rerun("perturb", j)
            per.append(_rel(oj, base))
        rec["per_call_rel"] = per
        rec["reruns"] = 1 + len(per)
        rec["rerun_s"] = round(time.time() - t0, 3)
    except Exception as exc:  # noqa: BLE001 - a failed probe loses the probe, never the request
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        st["mode"], st["plan"], st["target"], st["summ"] = None, None, -1, None
    _emit(rec)


_BATCH = bool(_OUT and os.environ.get("INFB_SD3_BATCH_OBSERVE"))
_BATCH_ARM = "BATCH_ARM"
_BATCH_MAX_LINES = 4000
_batch = {"n": 0, "now": None}
_batch_capture = {"count": 0, "bytes": 0}


_COMMON_INPUT_DIR = os.environ.get("INFB_SD3_COMMON_INPUT_DIR") if _OUT else None


def _batch_capture_common_input(module, args, kwargs, output, rec):
    """Read-only first-step snapshots for independent numeric attribution.

    Only an explicit verifier marker enables this. Bounded per worker; neither
    this record nor its later numeric replay changes any scored verdict.
    """
    base = _COMMON_INPUT_DIR or _TAP
    if not _BATCH or not base or not rec.get("completed"):
        return
    marker = os.path.join(base, "BATCH_CAPTURE")
    if not os.path.isfile(marker):
        return
    try:
        plan = json.load(open(marker, encoding="utf-8"))
        phase = plan.get("phase")
        if phase not in ("solo", "burst", "parallel") or not rec.get("timesteps"):
            return
        if any(t != 1000.0 for t in rec["timesteps"]):
            return
        torch = sys.modules["torch"]
        names = ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep")
        inputs = _dit_capture_inputs(module, args, kwargs)
        vals = {name: inputs[name] for name in names}
        vals["output"] = _dit_out(output)
        if not all(isinstance(t, torch.Tensor) for t in vals.values()):
            raise ValueError("common-input capture requires every tensor")
        nbytes = sum(t.numel() * t.element_size() for t in vals.values())
        if _batch_capture["count"] >= 32 or _batch_capture["bytes"] + nbytes > 256 * 1024 * 1024:
            _emit({"event": "batch_capture_limit", "rank": _rank(), "n": rec["n"]})
            return
        frozen = {}
        for name, t in vals.items():
            value = t.detach().cpu().clone()

            frozen[name] = value if type(value) is torch.Tensor else value.as_subclass(torch.Tensor)
        frozen.update(phase=phase, forward_n=rec["n"], rank=_rank_int())
        if phase == "parallel":
            frozen.update(mode=str(plan.get("mode")), request_key=str(plan.get("key")))
        root = os.path.join(base, "common_inputs")
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, "rank%d_n%d.pt" % (_rank_int(), rec["n"]))
        with open(path, "xb") as fh:
            torch.save(frozen, fh)
        _batch_capture["count"] += 1
        _batch_capture["bytes"] += nbytes
        _emit({"event": "batch_capture", "path": path, "phase": phase, "rank": _rank(),
               "n": rec["n"], "tensor_bytes": nbytes})
    except Exception as exc:
        _emit({"event": "batch_capture_error", "error": str(exc)[:300], "n": rec.get("n")})


_MS_MAX_BYTES = 6 * 1024 ** 3
_MS_MAX_FILES = 40000
_MS_CHUNK = 8 << 20
_ms = {"bytes": 0, "files": 0, "stopped": False, "errors": 0, "calls": {}, "known": {}}


class _MsLimit(Exception):
    pass


def _ms_plan():
    """The verifier's marker when it asks for every step, else None. Read per call; never raises."""
    base = _COMMON_INPUT_DIR or _TAP
    if not _OUT or not base or _ms["stopped"]:
        return None
    try:
        with open(os.path.join(base, "BATCH_CAPTURE"), encoding="utf-8") as fh:
            plan = json.load(fh)
    except (OSError, ValueError):
        return None
    return plan if isinstance(plan, dict) and plan.get("multistep") is True else None


def _ms_store(t, sha=None):
    """One tensor, persisted once by content: {dtype, shape, sha256, bytes}. A tensor whose sha256 the observer
    already computed and this process already stored is not copied again."""
    import hashlib
    if sha is not None and sha in _ms["known"]:
        return _ms["known"][sha]
    torch = sys.modules["torch"]
    value = t.detach()
    if type(value) is not torch.Tensor:
        value = value.as_subclass(torch.Tensor)
    dense = torch.empty(tuple(value.shape), dtype=value.dtype)
    dense.copy_(value)
    raw = dense.reshape(-1).view(torch.uint8).numpy()
    digest = hashlib.sha256(raw).hexdigest()
    desc = {"dtype": str(dense.dtype).replace("torch.", ""), "shape": list(dense.shape), "sha256": digest,
            "bytes": int(raw.nbytes)}
    if digest not in _ms["known"]:
        root = os.path.join(_COMMON_INPUT_DIR or _TAP, "multistep", "blobs")
        path = os.path.join(root, digest + ".bin")
        if not os.path.exists(path):
            if _ms["bytes"] + raw.nbytes > _MS_MAX_BYTES or _ms["files"] >= _MS_MAX_FILES:
                raise _MsLimit()
            os.makedirs(root, exist_ok=True)
            tmp = "%s.%d.tmp" % (path, os.getpid())
            view = memoryview(raw)
            with open(tmp, "wb") as fh:
                for i in range(0, len(view), _MS_CHUNK):
                    fh.write(view[i:i + _MS_CHUNK])
            os.replace(tmp, path)
            _ms["bytes"] += int(raw.nbytes)
            _ms["files"] += 1
        _ms["known"][digest] = desc
    return desc


def _ms_rows(t, fps=None):
    """Per-row descriptors of a batch tensor (rows on dim 0); `fps` are the observer's per-row fingerprints."""
    out = []
    for i in range(int(t.shape[0])):
        fp = (fps[i] or {}).get("fp") if fps and i < len(fps) and isinstance(fps[i], dict) else None
        sha = fp.rsplit("sha256:", 1)[-1] if isinstance(fp, str) and "sha256:" in fp else None
        out.append(_ms_store(t[i], sha))
    return out


def _ms_index(kind, plan, extra, tensors):
    """Persist the tensors of one call and append its index line. Never raises."""
    if _ms["stopped"]:
        return
    try:
        descs = {}
        for name, (value, rows, fps) in tensors.items():
            torch = sys.modules["torch"]
            if not isinstance(value, torch.Tensor):
                continue
            descs[name] = _ms_rows(value, fps) if rows and value.ndim >= 1 else _ms_store(value)
        counter = (str(plan.get("key")), kind)
        _ms["calls"][counter] = _ms["calls"].get(counter, 0) + 1
        line = {"schema": "sd3-qfe-multistep-index/1", "kind": kind, "phase": plan.get("phase"),
                "mode": plan.get("mode"), "key": plan.get("key"), "members": plan.get("members"),
                "rank": _rank_int(), "pid": os.getpid(), "call": _ms["calls"][counter], "t": time.time(),
                **extra, "tensors": descs}
        root = os.path.join(_COMMON_INPUT_DIR or _TAP, "multistep")
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "index.rank%d.pid%d.jsonl" % (_rank_int(), os.getpid())), "a",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
    except _MsLimit:
        _ms["stopped"] = True
        _emit({"event": "multistep_capture_limit", "rank": _rank(), "bytes": _ms["bytes"], "files": _ms["files"]})
    except Exception as exc:  # noqa: BLE001 - a failed capture loses the capture, never the call
        _ms["errors"] += 1
        if _ms["errors"] <= 20:
            _emit({"event": "multistep_capture_error", "kind": kind, "error": f"{type(exc).__name__}: {exc}"[:300]})


def _ms_dit(module, args, kwargs, returned, rec):
    """The marked request's DiT call: per-row inputs, the timestep and the output the pipeline actually received."""
    plan = _ms_plan()
    if plan is None or not rec.get("completed"):
        return
    try:
        vals = _dit_capture_inputs(module, args, kwargs)
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "multistep_capture_error", "kind": "dit", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return
    _ms_index("dit", plan, {"n": rec.get("n"), "timesteps": rec.get("timesteps"), "rows": rec.get("rows"),
                            "perturbed": bool(rec.get("perturbed")), "late_arm": rec.get("late_arm")},
              {"hidden_states": (vals.get("hidden_states"), True, rec.get("x")),
               "encoder_hidden_states": (vals.get("encoder_hidden_states"), True, rec.get("c")),
               "pooled_projections": (vals.get("pooled_projections"), True, rec.get("p")),
               "timestep": (vals.get("timestep"), False, None),
               "output": (_dit_out(returned), True, None)})


def _ms_float(v):
    try:
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(v, torch.Tensor):
            return [float(x) for x in v.detach().reshape(-1)[:8].cpu()]
        return [float(v)]
    except Exception:  # noqa: BLE001
        return None


def _ms_cpu(t):
    """A host copy taken before a call that might reuse its argument's storage."""
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(t, torch.Tensor):
        return None
    value = t.detach()
    if type(value) is not torch.Tensor:
        value = value.as_subclass(torch.Tensor)
    dense = torch.empty(tuple(value.shape), dtype=value.dtype)
    dense.copy_(value)
    return dense


def _ms_scheduler_before(step_fn, instance, args, kwargs):
    """The marked request's scheduler step inputs (sample, model_output, timestep, the sigma it steps from)."""
    try:
        plan = _ms_plan()
        if plan is None:
            return None
        import inspect
        got = inspect.signature(step_fn).bind(instance, *args, **kwargs).arguments
        ts = got.get("timestep")
        sig, idx = None, getattr(instance, "step_index", None)
        try:
            sigmas = getattr(instance, "sigmas", None)
            if sigmas is not None and idx is not None:
                sig = [float(sigmas[int(idx)]), float(sigmas[int(idx) + 1])]
        except Exception:  # noqa: BLE001
            sig = None
        return {"plan": plan, "sample": _ms_cpu(got.get("sample")), "model_output": _ms_cpu(got.get("model_output")),
                "timestep": _ms_cpu(ts), "timestep_value": _ms_float(ts), "step_index": idx, "sigmas": sig,
                "scheduler": type(instance).__module__ + "." + type(instance).__name__}
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "multistep_capture_error", "kind": "scheduler", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return None


def _ms_scheduler_after(ms, instance, result):
    try:
        prev = result[0] if isinstance(result, (tuple, list)) else getattr(result, "prev_sample", result)
        ms["sigmas_from"] = "before_step" if ms.get("sigmas") is not None else None
        if ms.get("sigmas") is None:


            try:
                idx, sigmas = getattr(instance, "step_index", None), getattr(instance, "sigmas", None)
                if idx is not None and sigmas is not None and int(idx) >= 1:
                    ms["sigmas"] = [float(sigmas[int(idx) - 1]), float(sigmas[int(idx)])]
                    ms["sigmas_from"] = "after_step"
            except Exception:  # noqa: BLE001
                pass
        _ms_index("scheduler", ms["plan"], {k: ms.get(k) for k in ("timestep_value", "step_index", "sigmas", "scheduler",
                                                                   "sigmas_from")},
                  {"sample": (ms["sample"], False, None), "model_output": (ms["model_output"], False, None),
                   "timestep": (ms["timestep"], False, None), "prev_sample": (prev, False, None)})
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "multistep_capture_error", "kind": "scheduler", "error": f"{type(exc).__name__}: {exc}"[:300]})


def _ms_vae_before(args, kwargs):
    try:
        plan = _ms_plan()
        if plan is None:
            return None
        return {"plan": plan, "z": _ms_cpu(_first_float(args, kwargs))}
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "multistep_capture_error", "kind": "vae", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return None


def _ms_vae_after(ms, out):
    try:
        sample = getattr(out, "sample", out)
        if isinstance(sample, (tuple, list)):
            sample = sample[0]
        _ms_index("vae", ms["plan"], {}, {"z": (ms["z"], False, None), "sample": (sample, False, None)})
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "multistep_capture_error", "kind": "vae", "error": f"{type(exc).__name__}: {exc}"[:300]})


_LATE_ARM_FILE = os.environ.get("INFB_SD3_LATE_ARM_FILE") if _OUT else None


def _late_arm(rec):
    """The armed plan when it applies to this forward, else None. Never raises."""
    if not _LATE_ARM_FILE or rec is None:
        return None
    try:
        with open(_LATE_ARM_FILE, encoding="utf-8") as fh:
            arm = json.load(fh)
        if not isinstance(arm, dict) or arm.get("kind") not in ("output", "collective"):
            return None
        ts = rec.get("timesteps") or []
        if not ts or any(t is None or float(t) > float(arm["t_max"]) for t in ts):
            return None
        if arm.get("rank") is not None and _rank_int() != int(arm["rank"]):
            return None
        if arm.get("key") is not None:
            base = _COMMON_INPUT_DIR or _TAP
            with open(os.path.join(base or "", "BATCH_CAPTURE"), encoding="utf-8") as fh:
                if json.load(fh).get("key") != arm["key"]:
                    return None
        return arm
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _par_scale(name, functional, a, k, scale):
    """This rank's contribution to one collective scaled by `scale` (the late collective arm)."""
    a = list(a)

    def s(t):
        return t * scale if hasattr(t, "new_zeros") else t

    if functional:
        if a:
            a[0] = s(a[0])
        elif "self" in k:
            k["self"] = s(k["self"])
    elif name == "all_reduce":
        t = a[0] if a else k.get("tensor")
        if hasattr(t, "mul_"):
            t.mul_(scale)
    elif name in _PAR_INPUT:
        i, key = _PAR_INPUT[name]
        if len(a) > i:
            a[i] = s(a[i])
        elif key in k:
            k[key] = s(k[key])
    elif name == "all_to_all":
        lst = a[1] if len(a) > 1 else k.get("input_tensor_list")
        if lst is not None:
            new = [s(t) for t in lst]
            if len(a) > 1:
                a[1] = new
            else:
                k["input_tensor_list"] = new
    return tuple(a), k


def _batch_moments(t):
    """Full logical tensor bytes for exact noise identity; sampled moments
    only for approximate conditioning attribution. Moments/rounded decimals
    cannot establish bit equality: unsampled elements may differ freely.
    Shape and dtype are included; tensor strides do not alter logical values.
    None means absent/failed instrumentation, never an equality witness.
    """
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(t, torch.Tensor):
        return None
    try:
        flat = t.detach().reshape(-1)[::61].double()
        w = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
        m = [float(flat.sum()), float((flat * flat).sum()), float((flat * w).sum()), float(flat.abs().max())]
        import hashlib
        raw = t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return {"m": m, "fp": "%s|%s|sha256:%s" %
                (tuple(t.shape), str(t.dtype).replace("torch.", ""), hashlib.sha256(raw).hexdigest())}
    except Exception:  # noqa: BLE001
        return None


def _batch_rows(module, args, kwargs):
    """{rows, x, c, p, t}: the batch rows a DiT forward was handed and, per row, the fingerprints of hidden_states,
    encoder_hidden_states and pooled_projections and the row's timestep (the SD3 DiT's arguments, by keyword or
    signature binding or unambiguous tensor structure). None when there is no tensor input.
    Binding errors are reported by the enclosing observer."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    vals = _dit_inputs(module, args, kwargs)
    x = vals["hidden_states"]
    if not isinstance(x, torch.Tensor) or x.ndim < 1:
        return None
    rows = int(x.shape[0])
    out = {"rows": rows, "x": [], "c": [], "p": [], "timesteps": []}
    ts = vals["timestep"]
    for i in range(rows):
        out["x"].append(_batch_moments(x[i]))
        for key, name in (("c", "encoder_hidden_states"), ("p", "pooled_projections")):
            v = vals[name]
            ok = isinstance(v, torch.Tensor) and v.ndim >= 1 and int(v.shape[0]) == rows
            out[key].append(_batch_moments(v[i]) if ok else None)
        try:
            if isinstance(ts, torch.Tensor):
                flat = ts.reshape(-1)
                out["timesteps"].append(float(flat[i if flat.numel() == rows else 0]))
            else:
                out["timesteps"].append(float(ts) if ts is not None else None)
        except Exception:  # noqa: BLE001
            out["timesteps"].append(None)
    return out


def _batch_arm():
    ' batch arm.'
    try:
        with open(os.path.join(_TAP or "", _BATCH_ARM), encoding="utf-8") as fh:
            plan = json.loads(fh.read() or "{}")
        if "selectors" in plan:
            return {"selectors": plan["selectors"], "delta": float(plan.get("delta", .25)),
                    "match_rel": float(plan["match_rel"]), "margin": float(plan["margin"])}
        return {"row": int(plan.get("row", 0)), "delta": float(plan.get("delta", 0.25))}
    except (OSError, ValueError, TypeError):
        return None


def _batch_out_rows(output):
    """Per-row fingerprints of a DiT forward's output tensor (the sample), or None."""
    torch = sys.modules.get("torch")
    y = _dit_out(output)
    if torch is None or not isinstance(y, torch.Tensor) or y.ndim < 1:
        return None
    return [_batch_moments(y[i]) for i in range(int(y.shape[0]))]


def _batch_selected_plan(plan, rec):
    """Follow unique conditioning, irrespective of split/concatenated CFG.
    Ambiguous empty-negative rows are not perturbed. The returned actual row
    is recorded, so the reader checks ownership independently after execution.
    """
    if "selectors" not in plan:
        return plan
    import math

    def distance(a, b):
        if not a or not b or len(a) != len(b) or not all(math.isfinite(v) for v in (*a, *b)):
            return float("inf")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / (math.sqrt(sum(x*x for x in b)) or 1e-30)

    selected = []
    for s in plan["selectors"]:
        ds = sorted((max(distance((rec["c"][i] or {}).get("m"), s["c"]),
                         distance((rec["p"][i] or {}).get("m"), s["p"])), i)
                    for i in range(rec["rows"]))
        if ds and ds[0][0] <= plan["match_rel"] and (len(ds) == 1 or
                ds[1][0] >= plan["margin"] * max(ds[0][0], 1e-12)):
            selected.append(ds[0][1])
    if len(set(selected)) != 1:
        return None
    return {"row": selected[0], "delta": plan["delta"], "source": "observed_conditioning"}


def _batch_perturb(output, plan, *, min_rows=2):
    """The armed probe: `output` with row plan["row"] scaled by (1 + delta), as a NEW tensor of the same type (the
    forward's own result is not written in place). Only for a forward handed at least two rows. Returns (output, done)."""
    torch = sys.modules.get("torch")
    y = _dit_out(output)
    if torch is None or not isinstance(y, torch.Tensor) or y.ndim < 1 or int(y.shape[0]) < min_rows:
        return output, False
    k = plan["row"]
    if not 0 <= k < int(y.shape[0]):
        return output, False
    scale = torch.ones((int(y.shape[0]),) + (1,) * (y.ndim - 1), dtype=y.dtype, device=y.device)
    scale[k] = 1.0 + plan["delta"]
    y2 = y * scale
    if isinstance(output, torch.Tensor):
        return y2, True
    if isinstance(output, tuple):
        return (y2,) + tuple(output[1:]), True
    if isinstance(output, list):
        return [y2] + list(output[1:]), True
    try:
        output.sample = y2
        return output, True
    except Exception:  # noqa: BLE001
        return output, False


def _batch_enter(module, args, kwargs):
    if not _BATCH:
        return
    _batch["now"] = None
    try:
        rec = _batch_rows(module, args, kwargs)
        if rec is not None:
            _batch["n"] += 1
            rec.update(n=_batch["n"], rank=_rank(), t_start=time.time())
            _batch["now"] = rec
            _emit({"event": "batch_dit_start", **rec})
    except Exception as exc:
        _emit({"event": "batch_observer_error", "where": "enter", "error": str(exc)[:300]})


def _batch_finish(module, args, kwargs, output):
    if not _BATCH:
        return None
    rec, _batch["now"] = _batch.get("now"), None
    if rec is None:
        return None
    changed, late_changed, out = False, False, output
    try:
        rec = dict(rec, y=_batch_out_rows(output), completed=output is not None)
        _batch_capture_common_input(module, args, kwargs, output, rec)
        native_plan = None
        try:
            with open(os.path.join(_TAP or "", "NATIVE_DIT_ARM"), encoding="utf-8") as f:
                native_plan = json.load(f)
        except (OSError, ValueError, TypeError):
            pass
        if native_plan is not None and _batch_arm() is not None:
            raise RuntimeError("native and batching interventions overlap")
        if output is not None and native_plan is not None:
            out, changed = _batch_perturb(output, {"row": 0, "delta": float(native_plan["delta"])}, min_rows=1)
            rec["native_perturbed"] = changed
        plan = _batch_arm()
        if plan is not None:
            plan = _batch_selected_plan(plan, rec)
        if output is not None and plan is not None:
            out, changed = _batch_perturb(output, plan)
        late = _late_arm(rec)
        if output is not None and late is not None and late["kind"] == "output":
            if changed:
                _emit({"event": "late_arm_skipped", "why": "another intervention already changed this forward",
                       "n": rec.get("n")})
            else:
                out, late_changed = _batch_perturb(output, {"row": int(late.get("row") or 0),
                                                            "delta": float(late["delta"])}, min_rows=1)
                if late_changed:
                    rec["late_arm"] = {"kind": "output", "row": int(late.get("row") or 0),
                                       "delta": float(late["delta"]), "t_max": float(late["t_max"]),
                                       "key": late.get("key")}
                    _emit({"event": "late_arm_applied", "rank": _rank(), "n": rec.get("n"),
                           "timesteps": rec.get("timesteps"), **rec["late_arm"]})
        rec.update(perturbed=changed, perturb_plan=plan if changed and plan is not None else None)
        if changed or late_changed:


            rec["y_returned"] = _batch_out_rows(out)
        if _par.get("late") is not None:
            rec["late_arm"] = dict(_par["late"])
        _ms_dit(module, args, kwargs, out, rec)
        _emit({"event": "batch_dit", **rec})
    except Exception as exc:
        _emit({"event": "batch_observer_error", "where": "finish", "error": str(exc)[:300]})
    if output is not None and _FAULT_DIR:
        _maybe_fault(rows=rec.get("rows"), phase="after_dit")
    return out if changed or late_changed else None


def _count_dit_calls(transformer):
    """Observe each forward of the pipeline's DiT, with request-row and intervention evidence."""
    reg = getattr(transformer, "register_forward_hook", None)
    if not callable(reg):
        return
    pre = getattr(transformer, "register_forward_pre_hook", None)

    def pre_hook(module, args, kwargs):
        _batch_enter(module, args, kwargs)
        if _FAULT_DIR:
            if not _BATCH:
                _maybe_fault()
            _maybe_slow()
        x = None
        _prec_now["v"] = None
        try:
            inp = _dit_inputs(module, args, kwargs)
            x = inp["hidden_states"]
            _prec_now["v"] = {"in": _dtype_name(getattr(x, "dtype", None)), "autocast": _autocast_of(x)}
            if _PW:
                _joint_dit(inp.get("encoder_hidden_states"))
            if not _prec_now["env_done"]:
                _prec_now["env_done"] = True
                _precision_env()
                _emit({"event": "precision", "component": "transformer", "params": _param_census(module)})
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "record_error", "where": "dit_inputs", "error": str(exc)[:300]})
        try:
            _attn_begin()
        except Exception:  # noqa: BLE001
            pass
        if _PAR:
            try:
                _par_begin(module, _shape(x) if x is not None else None)
            except Exception:  # noqa: BLE001
                pass
        if _PW:
            while _pw["dit"]:
                _pw_exit("transformer", _pw["dit"].pop(), failed=True)
            _pw["dit"].append(_pw_enter("transformer", module))
        return None

    def pw_close(module, args, kwargs, output):
        """FIRST post-hook, also after a raising forward (always_call): the witness closes before any hook runs."""
        try:
            _pw_exit("transformer", _pw["dit"].pop() if _pw["dit"] else None, failed=output is None)
        except Exception:  # noqa: BLE001
            pass
        return None

    def pw_probe(module, args, kwargs, output):
        """LAST post-hook: the consumption probe, after this forward's attention window has closed."""
        try:
            fwd = module.forward
            _pw_maybe_probe("transformer", module, lambda: fwd(*args, **kwargs), output)
        except Exception:  # noqa: BLE001
            pass
        return None

    def hook(module, args, kwargs, output):
        try:
            summ = _attn_end()
            extra = {}
            if _CENSUS or _PAR:
                extra = {"rank": _rank()}
                try:
                    extra["x"] = _shape(_dit_inputs(module, args, kwargs)["hidden_states"])
                except Exception as exc:
                    extra["x"] = None
                    _emit({"event": "record_error", "where": "dit_inputs", "error": str(exc)[:300]})
            if _PAR:
                extra["coll"] = _par_end()
            if _FAULT_DIR and _slow["s"] > 0.0:
                extra["slow_s"] = _slow["s"]
            if _prec_now.get("v"):
                extra["prec"] = dict(_prec_now["v"], out=_out_dtype(output))
            _emit({"event": "dit_call", **({"attn": summ} if summ is not None else {}), **extra})
            if summ is not None and summ.get("probe"):
                _attn_probe(module, args, kwargs, output, summ)
        except Exception:  # noqa: BLE001
            pass
        return None

    try:
        if callable(pre):
            pre(pre_hook, with_kwargs=True)
        if _PW:
            reg(pw_close, with_kwargs=True, always_call=True)
        reg(hook, with_kwargs=True)
        if _PW:
            reg(pw_probe, with_kwargs=True)
        if _BATCH:
            reg(_batch_finish, with_kwargs=True, always_call=True)
            _emit({"event": "batch_observer_installed", "rank": _rank(), "pre_hook": callable(pre)})
        _emit({"event": "dit_counter_installed", "attn_window": callable(pre)})
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "record_error", "where": "dit_counter", "error": f"{type(exc).__name__}: {exc}"[:300]})


_PREC_MAX_CALLS = 2


def _dtype_name(x):
    return None if x is None else str(x).replace("torch.", "")


def _first_float(args, kwargs):
    """The first floating-point tensor handed to a call (the activations; token ids are skipped)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    for v in list(args) + list((kwargs or {}).values()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            return v
    return None


def _autocast_of(t):
    """{device: autocast dtype or None} for the device a tensor sits on, read while the call runs."""
    torch = sys.modules.get("torch")
    dev = getattr(getattr(t, "device", None), "type", None) or "cuda"
    try:
        on = bool(torch.is_autocast_enabled(dev))
        return {"device": dev, "dtype": _dtype_name(torch.get_autocast_dtype(dev)) if on else None}
    except Exception:  # noqa: BLE001 - an older torch without the device-typed query
        return {"device": dev, "dtype": None, "unknown": True}


_JOINT_MAX_T5_ROWS = 4096
_JOINT_MAX_DIT = 4096
_JOINT_IDENT_MAX = 64
_JOINT_T5_DEFAULT_LEN = 256
_joint = {"t5_rows": 0, "dit": 0, "calls": 0, "t5_lens": set(), "seen": set(), "ident": {}, "overflow": {}}


def _joint_rows(t):
    """[rows, tokens, hidden] view of a token tensor, or None."""
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.dim() < 2:
        return None
    t = t.detach()
    return t.reshape(-1, *t.shape[-2:]) if t.dim() >= 3 else t.reshape(1, *t.shape)


def _joint_sha(t, path):
    import hashlib
    torch = sys.modules.get("torch")
    for d in path:
        t = t.to(getattr(torch, d))
    return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


_JOINT_T5_PATHS = {"float32": ("float32",), "bfloat16": ("bfloat16",), "float16>float32": ("float16", "float32"),
                   "float16>bfloat16": ("float16", "bfloat16"), "bfloat16>float32": ("bfloat16", "float32")}


def _joint_ids_sha(ids):
    """The canonical identity of one tokenized prompt: its token ids without padding (0) and the end token (1), so a
    different pad length, padding side or end-token setting names the same prompt. The judge's copy is
    sd3_precision_profiles.canonical_ids_sha; the two must agree."""
    import hashlib
    core = [int(i) for i in ids]
    while core and core[-1] in (0, 1):
        core.pop()
    while core and core[0] == 0:
        core.pop(0)
    return hashlib.sha256(",".join(str(i) for i in core).encode()).hexdigest(), len(core)


def _joint_token_ids(args, kwargs, module=None):
    """Per row, the canonical ids of the token tensor a T5 forward was handed, or None. Read the way _dit_inputs reads
    the DiT: bound to the forward's own signature when it names input_ids; otherwise the one integer tensor handed, or
    the one that is not a 0/1 mask (an empty prompt's ids are 0/1 too, so a mask is only discarded when it leaves
    exactly one). Several candidates or none: None -- the judge reports the rows' prompt as unidentified, never guesses."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    is_ids = lambda a: (isinstance(a, torch.Tensor) and not a.is_floating_point() and a.dtype != torch.bool
                        and a.dim() in (1, 2) and a.numel() > 0)
    ids = (kwargs or {}).get("input_ids")
    if not is_ids(ids) and module is not None:
        try:
            import inspect
            bound = inspect.signature(module.forward).bind_partial(*args, **(kwargs or {}))
            ids = bound.arguments.get("input_ids")
        except (TypeError, ValueError):
            ids = None
    if not is_ids(ids):
        cands = [a for a in list(args) + [v for k, v in (kwargs or {}).items() if "mask" not in k] if is_ids(a)]
        if len(cands) > 1:
            cands = [a for a in cands if int(a.max()) > 1]
        ids = cands[0] if len(cands) == 1 else None
    if ids is None:
        return None
    ids = ids.reshape(1, -1) if ids.dim() == 1 else ids
    return [_joint_ids_sha(row) for row in ids.tolist()]


def _joint_overflow(kind, n):
    o = _joint["overflow"]
    o[kind] = o.get(kind, 0) + n
    _emit({"event": "precision_joint_overflow", "kind": kind, "dropped": o[kind],
           "max": _JOINT_MAX_T5_ROWS if kind == "t5_rows" else _JOINT_MAX_DIT})


def _joint_t5(output, args=(), kwargs=None, module=None):
    """precision_joint_t5: each row of a T5 output (last_hidden_state), hashed under every judged cast and detour, with
    the canonical token ids that produced it. Every T5 forward of the worker, up to the resource guard."""
    try:
        o = getattr(output, "last_hidden_state", None)
        if o is None and isinstance(output, (tuple, list)) and output:
            o = output[0]
        x = _joint_rows(o if o is not None else output)
        if x is None:
            _emit({"event": "precision_joint_t5", "status": "unreadable"})
            return
        if _joint["t5_rows"] + x.shape[0] > _JOINT_MAX_T5_ROWS:
            _joint_overflow("t5_rows", int(x.shape[0]))
            return
        ids = _joint_token_ids(args, kwargs, module)
        if ids is not None and len(ids) != x.shape[0]:
            ids = None
        _joint["t5_rows"] += int(x.shape[0])
        _joint["t5_lens"].add(int(x.shape[-2]))
        _emit({"event": "precision_joint_t5", "status": "ok", "dtype": _dtype_name(x.dtype), "shape": list(x.shape),
               "rows": [dict({k: _joint_sha(x[i], v) for k, v in _JOINT_T5_PATHS.items()},
                             ids=ids[i][0] if ids else None, ntok=ids[i][1] if ids else None)
                        for i in range(x.shape[0])]})
    except Exception as exc:  # noqa: BLE001 - an unreadable output is a missing reading, never a crash
        _emit({"event": "precision_witness_error", "component": "text_encoder_3", "where": "joint",
               "error": f"{type(exc).__name__}: {exc}"[:200]})


def _joint_whole(ehs):
    """The conditioning's whole-tensor hash, cached for the same live object at the same version (SD3 hands the DiT
    the same prompt_embeds object at every step): hashed once per distinct object, never per step. A tensor without a
    version counter -- an inference tensor (made under torch.inference_mode), whose in-place edits nothing tracks, and
    whose ._version raises -- is never cached: it is hashed at every forward."""
    import weakref
    try:
        ver = None if ehs.is_inference() else ehs._version
    except Exception:  # noqa: BLE001 - no readable version: no cache
        ver = None
    ident = _joint["ident"]
    if ver is not None:
        hit = ident.get(id(ehs))
        if hit is not None and hit[0]() is ehs and hit[1] == ver:
            return hit[2]
    whole = _joint_sha(ehs.detach(), (_dtype_name(ehs.dtype),))
    if ver is not None:
        while ident and len(ident) >= _JOINT_IDENT_MAX:
            ident.pop(next(iter(ident)))
        try:
            ident[id(ehs)] = (weakref.ref(ehs), ver, whole)
        except TypeError:
            pass
    return whole


def _joint_dit(ehs):
    """precision_joint_dit + precision_joint_call. Every DiT forward writes one call line naming the conditioning it
    consumed (w = whole-tensor hash) or why none could be named (status); each distinct conditioning is written once,
    with the T5 part of every row (the last L tokens, for every T5 length L this worker produced and SD3's 256) hashed
    at fp32 and bf16 -- what the DiT's first GEMM consumes at either compute precision."""
    try:
        _joint["calls"] += 1
        c = _joint["calls"]
        x = _joint_rows(ehs)
        if x is None:
            _emit({"event": "precision_joint_call", "c": c, "status": "no_conditioning"})
            return
        whole = _joint_whole(ehs)
        w = whole[:16]
        if whole not in _joint["seen"]:
            if _joint["dit"] >= _JOINT_MAX_DIT:
                _joint_overflow("dit", 1)
                _emit({"event": "precision_joint_call", "c": c, "w": w, "status": "overflow"})
                return
            _joint["seen"].add(whole)
            _joint["dit"] += 1
            lens = sorted(L for L in (_joint["t5_lens"] | {_JOINT_T5_DEFAULT_LEN}) if L <= int(x.shape[-2]))
            _emit({"event": "precision_joint_dit", "n": _joint["dit"], "w": w, "dtype": _dtype_name(x.dtype),
                   "shape": list(x.shape), "lens": lens,
                   "rows": [{str(L): {d: _joint_sha(x[i, -L:], (d,)) for d in ("float32", "bfloat16")} for L in lens}
                            for i in range(x.shape[0])]})
        _emit({"event": "precision_joint_call", "c": c, "w": w, "b": int(x.shape[0]), "status": "ok"})
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "precision_witness_error", "component": "transformer", "where": "joint",
               "error": f"{type(exc).__name__}: {exc}"[:200]})


def _param_census(module):
    out = {}
    try:
        for q in module.parameters():
            k = _dtype_name(q.dtype)
            out[k] = out.get(k, 0) + int(q.numel())
    except Exception:  # noqa: BLE001
        pass
    return out


def _out_dtype(o):
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    for cand in (o, getattr(o, "sample", None), getattr(o, "last_hidden_state", None),
                 o[0] if isinstance(o, (tuple, list)) and o else None):
        if isinstance(cand, torch.Tensor):
            return _dtype_name(cand.dtype)
    return None


def _precision_env():
    torch = sys.modules.get("torch")
    rec = {"event": "precision_env", "rank": _rank(),
           "env": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")}}
    try:
        rec["tf32"] = {"matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                       "cudnn": bool(torch.backends.cudnn.allow_tf32),
                       "float32_matmul_precision": str(torch.get_float32_matmul_precision())}
    except Exception as exc:  # noqa: BLE001
        rec["tf32"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    _emit(rec)


_PW = bool(_OUT and os.environ.get("INFB_SD3_PRECISION_WITNESS"))
_PW_ARM = "PREC_ARM"
_PW_MAX_KEYS = 48
_PW_MAX_SEQ = 50000
_PW_MAX_TARGETS = 16
_PW_MAX_ALL = 512
_PW_MAX_OUTS = 64
_PW_SHORT = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "float64": "fp64",
             "float8_e4m3fn": "fp8e4m3", "float8_e5m2": "fp8e5m2"}


_pw = {"cls": None, "stack": [], "calls": {}, "dit": [], "enc": {}, "probe_done": set(), "modstack": []}


def _pw_short(t):
    d = str(getattr(t, "dtype", None)).replace("torch.", "")
    return _PW_SHORT.get(d, d if getattr(t, "dtype", None) is not None else "-")


def _pw_table(torch):
    """{OpOverloadPacket: (kind, operand positions)}: the MAIN ops, whose operands decide a component's compute
    precision. Both the composite forms (seen under inference_mode, where autograd does not decompose them: `linear`,
    `scaled_dot_product_attention`, `conv2d`) and the ones they decompose into (`addmm`, `convolution`, the fused SDPA
    kernels) are listed: a call reaches this mode in exactly one of them."""
    aten = torch.ops.aten
    spec = (("gemm", (0, 1), ("mm", "bmm", "mv", "dot", "matmul", "linear", "_scaled_mm")),
            ("gemm", (1, 2), ("addmm", "baddbmm", "addbmm", "addmv", "_addmm_activation")),
            ("conv", (0, 1), ("convolution", "_convolution", "conv1d", "conv2d", "conv3d", "conv_transpose1d",
                              "conv_transpose2d", "conv_transpose3d", "cudnn_convolution", "mkldnn_convolution",
                              "convolution_overrideable")),
            ("attn", (0, 1, 2), ("scaled_dot_product_attention", "_scaled_dot_product_flash_attention",
                                 "_scaled_dot_product_efficient_attention", "_scaled_dot_product_cudnn_attention",
                                 "_scaled_dot_product_flash_attention_for_cpu",
                                 "_scaled_dot_product_fused_attention_overrideable", "_scaled_dot_product_attention_math",
                                 "_flash_attention_forward", "_efficient_attention_forward")))
    table = {}
    for kind, pos, names in spec:
        for n in names:
            try:
                table[getattr(aten, n)] = (kind, pos)
            except Exception:  # noqa: BLE001 - an op this torch does not have
                continue
    try:
        table[aten.einsum] = ("gemm", "list1")
    except Exception:  # noqa: BLE001
        pass
    return table


def _pw_macs(kind, ops, o0):
    """Multiply-accumulates of one main op, from its operand and output shapes (diagnostic weight of the op)."""
    try:
        if kind == "gemm" and ops and ops[0].dim() >= 1:
            return int(o0.numel()) * int(ops[0].shape[-1])
        if kind == "conv" and len(ops) >= 2:
            k = 1
            for d in ops[1].shape[1:]:
                k *= int(d)
            return int(o0.numel()) * k
        if kind == "attn" and len(ops) >= 3:
            d, dv = int(ops[0].shape[-1]), int(ops[2].shape[-1])
            return int(ops[0].numel()) // max(d, 1) * int(ops[1].shape[-2]) * (d + dv)
    except Exception:  # noqa: BLE001
        pass
    return 0


def _pw_perturb(torch, out, delta):
    """The probe's perturbation of one op's result: every other feature (last dim) scaled by 1 + delta, the rest by
    1 - delta -- a normalisation downstream cannot undo it, as it could a uniform scale."""
    def p(t):
        n = int(t.shape[-1]) if t.dim() else 1
        sign = torch.ones(n, dtype=t.dtype, device=t.device)
        sign[1::2] = -1.0
        return t * (1.0 + delta * sign)

    if isinstance(out, torch.Tensor) and out.is_floating_point():
        return p(out)
    if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor) and out[0].is_floating_point():
        return (p(out[0]),) + tuple(out[1:])
    return out


def _pw_note(acc, spec, func, args, out):
    """One main op: its operand dtypes, output dtype and MACs into the call's tally; the parameters it read; the probe's
    sequence and perturbation. Returns the op's result (the perturbed one only for the probe's target)."""
    torch = sys.modules["torch"]
    kind, pos = spec
    if pos == "list1":
        ops = list(args[1]) if len(args) > 1 and isinstance(args[1], (list, tuple)) else []
    else:
        ops = [args[i] for i in pos if i < len(args)]
    ops = [o for o in ops if isinstance(o, torch.Tensor)]
    o0 = out[0] if isinstance(out, (tuple, list)) and out else out
    key = f"{kind}:{func.overloadpacket.__name__}:{'/'.join(_pw_short(o) for o in ops)}>{_pw_short(o0)}"
    macs = _pw_macs(kind, ops, o0)
    tally = acc["main"]
    ent = tally.get(key)
    if ent is None:
        if len(tally) >= _PW_MAX_KEYS:
            key = "overflow"
        ent = tally.setdefault(key, [0, 0])
    ent[0] += 1
    ent[1] += macs
    i = acc["n_main"]
    acc["n_main"] = i + 1
    acc["macs"] += macs
    pm = acc["pmap"]
    if pm:
        for o in ops:
            try:
                bp = o.untyped_storage().data_ptr()
            except Exception:  # noqa: BLE001 - a tensor without a plain storage (a subclass) reads nothing here
                continue
            if bp in pm:
                acc["read"].add(bp)
    seq = acc["seq"]
    if seq is not None and len(seq) < _PW_MAX_SEQ:
        ms = _pw["modstack"]
        seq.append([key, macs, ms[-1] if ms else None])
    if i == acc["target"]:
        try:
            functional = not func._schema.is_mutable
        except Exception:  # noqa: BLE001
            functional = False
        if functional:
            out = _pw_perturb(torch, out, acc["delta"])
            acc["perturbed"] = key
    return out


def _pw_mode_cls():
    """The witness's TorchDispatchMode class, built once torch is loaded (this file imports only the stdlib)."""
    if _pw["cls"] is None:
        torch = sys.modules["torch"]

        disp = sys.modules.get("torch.utils._python_dispatch")
        if disp is None or not hasattr(disp, "TorchDispatchMode"):
            raise RuntimeError("torch.utils._python_dispatch is not loaded: no TorchDispatchMode to witness with")
        TorchDispatchMode = disp.TorchDispatchMode
        table = _pw_table(torch)

        class _InfbPrecisionWitness(TorchDispatchMode):
            """Pass-through witness of one served call: every op runs unchanged, below autocast (so the operands are
            the ones the kernel really gets), and each MAIN op's operand dtypes are noted. The probe's perturbed
            re-runs are the only place a result is altered, and the pipeline never receives those."""

            def __init__(self, acc):
                super().__init__()
                self._infb_acc = acc

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                acc = self._infb_acc
                acc["n_ops"] += 1
                spec = table.get(getattr(func, "overloadpacket", None))
                if spec is not None:
                    try:
                        out = _pw_note(acc, spec, func, args, out)
                    except Exception:  # noqa: BLE001 - a failed note loses the note (counted), never the op
                        acc["errors"] += 1
                return out

        _pw["cls"] = _InfbPrecisionWitness
    return _pw["cls"]


def _pw_params(module):
    """{storage pointer: elements} of the component's parameters of rank >= 2 (its GEMM / conv weights), read at the
    call so that a weight moved since the build is found where it now is."""
    pm = {}
    params = getattr(module, "parameters", None)
    if not callable(params):
        return pm
    try:
        for p in params():
            try:
                if p.dim() >= 2:
                    pm[p.untyped_storage().data_ptr()] = int(p.numel())
            except Exception:  # noqa: BLE001 - a sharded / subclass parameter is simply not in the map
                continue
    except Exception:  # noqa: BLE001
        pass
    return pm


def _pw_open(comp, module, seq=None, target=-1, delta=0.0):
    acc = {"comp": comp, "n_ops": 0, "n_main": 0, "macs": 0, "main": {}, "impl": {}, "errors": 0,
           "pmap": _pw_params(module), "read": set(), "seq": seq, "target": target, "delta": delta, "perturbed": None,
           "t0": time.time()}
    mode = _pw_mode_cls()(acc)
    mode.__enter__()
    _pw["stack"].append((mode, acc))
    return acc


def _pw_close(acc):
    """Exit this call's mode (and any a raising call left open above it)."""
    st = _pw["stack"]
    while st:
        mode, a = st.pop()
        try:
            mode.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        if a is acc:
            break


def _pw_enter(comp, module):
    """Open the witness for one served call of `comp`. Never raises; None when it could not open (recorded)."""
    if not _PW:
        return None
    try:
        if sys.modules.get("torch") is None:
            return None
        n = _pw["calls"].get(comp, 0)
        _pw["calls"][comp] = n + 1
        acc = _pw_open(comp, module)
        acc["call"] = n
        return acc
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "precision_witness_error", "component": comp, "where": "enter",
               "error": f"{type(exc).__name__}: {exc}"[:300]})
        return None


def _pw_exit(comp, acc, failed=False):
    """Close one served call's witness and write its `precision_ops` line. Never raises."""
    if acc is None:
        return
    try:
        _pw_close(acc)
        tot = sum(acc["pmap"].values())
        _emit({"event": "precision_ops", "component": comp, "call": acc.get("call"), "n_ops": acc["n_ops"],
               "n_main": acc["n_main"], "macs": acc["macs"], "main": acc["main"], "impl": acc["impl"],
               "weights_read": {"read": len(acc["read"]), "of": len(acc["pmap"]),
                                "numel_share": (sum(acc["pmap"][p] for p in acc["read"]) / tot) if tot else None},
               "errors": acc["errors"], "failed": bool(failed), "s": round(time.time() - acc["t0"], 4)})
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "precision_witness_error", "component": comp, "where": "exit",
               "error": f"{type(exc).__name__}: {exc}"[:300]})


def _pw_impl_note(impl, q, k, v, out):
    """An engine AttentionImpl call inside a witnessed call: its q / k / v / output dtypes at the kernel's own boundary
    (a fused kernel outside the dispatcher -- FA3 -- is seen here, not by the mode)."""
    if not _pw["stack"]:
        return
    acc = _pw["stack"][-1][1]
    key = (f"attn_impl:{type(impl).__name__}:{'/'.join(_pw_short(t) for t in (q, k, v))}>"
           f"{_pw_short(out[0] if isinstance(out, (tuple, list)) and out else out)}")
    macs = _pw_macs("attn", [t for t in (q, k, v) if t is not None], None) if all(
        t is not None for t in (q, k, v)) else 0
    ent = acc["impl"].setdefault(key, [0, 0]) if len(acc["impl"]) < _PW_MAX_KEYS else acc["impl"].setdefault(
        "overflow", [0, 0])
    ent[0] += 1
    ent[1] += macs


def _pw_plan():
    """The consumption probe's plan while `<tap>/PREC_ARM` exists, else None."""
    if not _TAP:
        return None
    path = os.path.join(_TAP, _PW_ARM)
    try:
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            plan = json.load(fh)
        return plan if isinstance(plan, dict) else {}
    except Exception:  # noqa: BLE001 - an unreadable plan is a default plan
        return {}


def _pw_outs(o, clone=False):
    """The floating-point tensors a component call returned (a tensor, a tuple, or an output object's fields)."""
    torch = sys.modules.get("torch")
    found = []

    def add(t):
        if torch is not None and isinstance(t, torch.Tensor) and t.is_floating_point() and len(found) < _PW_MAX_OUTS:
            found.append(t.detach().clone() if clone else t.detach())

    if torch is not None and isinstance(o, torch.Tensor):
        add(o)
    elif isinstance(o, (list, tuple)):
        for v in o:
            for w in (v if isinstance(v, (list, tuple)) else (v,)):
                add(w)
    else:
        for key in ("sample", "last_hidden_state", "pooler_output", "text_embeds", "image_embeds"):
            add(getattr(o, key, None))
        hs = getattr(o, "hidden_states", None)
        if isinstance(hs, (list, tuple)):
            for v in hs:
                add(v)
    return found


def _pw_rel(a, b):
    """||a - b|| / ||b|| over every tensor pair, float64; None when the structures differ."""
    if not a or len(a) != len(b):
        return None
    num = den = 0.0
    for x, y in zip(a, b):
        if tuple(x.shape) != tuple(y.shape):
            return None
        dx = x.double() - y.double()
        num += float((dx * dx).sum())
        den += float((y.double() * y.double()).sum())
    return (num ** 0.5) / max(den ** 0.5, 1e-30)


def _pw_unit(seed, s):
    """A deterministic number in [0, 1) from (seed, s): splitmix64 (this file imports only the stdlib core)."""
    m = 0xFFFFFFFFFFFFFFFF
    z = (int(seed) * 0x9E3779B97F4A7C15 + (int(s) + 1) * 0xD1B54A32D192ED03) & m
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
    return (z ^ (z >> 31)) / float(1 << 64)


def _pw_targets(seq, k, seed):
    """Up to k main-op indices of a call's op sequence, sampled by MULTIPLY-ACCUMULATES: one per equal stratum of the
    call's cumulative MACs, at a seeded random point inside it -- an op is drawn in proportion to the work it does, so
    whatever carries the bulk of the compute is what gets perturbed (ops without a MAC count weigh 1)."""
    w = [max(int(e[1] or 0), 1) for e in seq]
    total = sum(w)
    if not w or k <= 0:
        return []
    picks, acc_w, j = set(), 0, 0
    points = sorted(((s + _pw_unit(seed, s)) * total / k) for s in range(k))
    for i, wi in enumerate(w):
        acc_w += wi
        while j < len(points) and points[j] < acc_w:
            picks.add(i)
            j += 1
    return sorted(picks)


def _pw_indexed(names):
    """{container path: its highest index} over a module tree's names ('a.layers.11.mlp' -> {'a.layers': 11})."""
    out = {}
    for n in names:
        parts = str(n).split(".")
        for i, part in enumerate(parts):
            if part.isdigit() and i:
                c = ".".join(parts[:i])
                out[c] = max(out.get(c, -1), int(part))
    return dict(sorted(out.items())[:32])


def _pw_maybe_probe(comp, module, call, output):
    """CONSUMPTION PROBE: while `<tap>/PREC_ARM` exists, the first served call of `comp` that returns is re-run on the
    same inputs -- once unchanged (the repeat; it also labels each main op with its module), then once per main op
    sampled by its MACs (`_pw_targets`; every op with the plan's exhaustive "all" option) with THAT op's result perturbed -- and
    how far each re-run's output moved is recorded (`precision_probe`). A main op whose perturbation does not move the
    output is not consumed. The pipeline keeps the original output; the re-runs' outputs are dropped. Never raises."""
    if not _PW or comp in _pw["probe_done"]:
        return
    plan = _pw_plan()
    if plan is None:
        return
    _pw["probe_done"].add(comp)
    delta = float(plan.get("delta", 0.5))
    k = min(int(plan.get("k", 8)), _PW_MAX_TARGETS)
    rec = {"event": "precision_probe", "component": comp, "delta": delta, "k": k, "seed": plan.get("seed", 0)}
    t0 = time.time()
    hooks = []
    try:
        torch = sys.modules["torch"]
        base = _pw_outs(output, clone=True)
        rec["n_out"] = len(base)


        names = {id(m): n for n, m in module.named_modules()}
        rec["indexed"] = _pw_indexed(names.values())
        ms = _pw["modstack"]
        ms.clear()

        def gpre(mod, args):
            ms.append(names.get(id(mod)))

        def gpost(mod, args, out):
            if ms:
                ms.pop()

        mm = torch.nn.modules.module
        hooks = [mm.register_module_forward_pre_hook(gpre), mm.register_module_forward_hook(gpost, always_call=True)]
        seq = []
        acc = _pw_open(comp, module, seq=seq)
        try:
            rep = call()
        finally:
            _pw_close(acc)
            for h in hooks:
                h.remove()
            hooks = []
            ms.clear()
        rec["repeat_rel"] = _pw_rel(_pw_outs(rep), base)
        rec["n_main"] = len(seq)
        rec["main_macs"] = sum(int(e[1] or 0) for e in seq)
        rec["repeat_main"] = acc["main"]
        per = []
        targets = (list(range(len(seq)))[:_PW_MAX_ALL] if plan.get("all")
                   else _pw_targets(seq, k, int(plan.get("seed", 0) or 0)))
        for j in targets:
            acc = _pw_open(comp, module, target=j, delta=delta)
            try:
                oj = call()
            finally:
                _pw_close(acc)
            per.append({"i": j, "key": seq[j][0], "macs": seq[j][1], "module": seq[j][2],
                        "perturbed": acc["perturbed"] is not None, "rel": _pw_rel(_pw_outs(oj), base)})
        rec["targets"] = per
        rec["reruns"] = 1 + len(per)
    except Exception as exc:  # noqa: BLE001 - a failed probe loses the probe, never the request
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        _pw["modstack"].clear()
    rec["s"] = round(time.time() - t0, 3)
    _emit(rec)


def _observe_precision(mods):
    """Pass-through precision observers on the served text encoders, the VAE's decode and the scheduler's step; with
    the precision witness on (profile mode), each served call of the five components also runs inside it."""
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        m = mods.get(name)
        reg = getattr(m, "register_forward_hook", None)
        if not callable(reg):
            continue
        if _PW:
            try:
                opened = _pw["enc"].setdefault(name, [])

                def pw_pre(module, args, kwargs, _name=name, _opened=opened):
                    _opened.append(_pw_enter(_name, module))
                    return None

                def pw_post(module, args, kwargs, output, _name=name, _opened=opened):
                    try:
                        _pw_exit(_name, _opened.pop() if _opened else None, failed=output is None)
                        if output is not None:
                            fwd = module.forward
                            _pw_maybe_probe(_name, module, lambda: fwd(*args, **kwargs), output)
                    except Exception:  # noqa: BLE001
                        pass
                    return None

                m.register_forward_pre_hook(pw_pre, with_kwargs=True)


                reg(pw_post, with_kwargs=True, always_call=True, prepend=True)
            except Exception as exc:  # noqa: BLE001
                _emit({"event": "precision_witness_error", "component": name, "where": "install",
                       "error": f"{type(exc).__name__}: {exc}"[:200]})
        seen = {"n": 0}

        def hook(module, args, kwargs, output, _name=name, _seen=seen):
            try:
                if _seen["n"] < _PREC_MAX_CALLS:
                    _seen["n"] += 1
                    x = _first_float(args, kwargs)
                    _emit({"event": "precision", "component": _name, "params": _param_census(module),
                           "in": _dtype_name(getattr(x, "dtype", None)), "out": _out_dtype(output),
                           "autocast": _autocast_of(x if x is not None else next(module.parameters(), None))})
            except Exception:  # noqa: BLE001
                pass
            if _PW and _name == "text_encoder_3":
                _joint_t5(output, args, kwargs, module)
            return None

        try:
            reg(hook, with_kwargs=True)
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "record_error", "where": f"precision:{name}", "error": f"{type(exc).__name__}: {exc}"[:200]})
    vae = mods.get("vae")
    dec = getattr(vae, "decode", None)
    if callable(dec) and not getattr(dec, "_infb_precision", False):
        seen_v = {"n": 0}

        def decode(*args, **kwargs):
            ms = _ms_vae_before(args, kwargs)
            acc = _pw_enter("vae", vae) if _PW else None
            try:
                out = dec(*args, **kwargs)
            except BaseException:
                _pw_exit("vae", acc, failed=True)
                raise
            _pw_exit("vae", acc)
            if ms is not None:
                _ms_vae_after(ms, out)
            if acc is not None:
                _pw_maybe_probe("vae", vae, lambda: dec(*args, **kwargs), out)
            try:
                if seen_v["n"] < _PREC_MAX_CALLS:
                    seen_v["n"] += 1
                    x = _first_float(args, kwargs)
                    _emit({"event": "precision", "component": "vae", "params": _param_census(vae),
                           "in": _dtype_name(getattr(x, "dtype", None)), "out": _out_dtype(out),
                           "autocast": _autocast_of(x)})
            except Exception:  # noqa: BLE001
                pass
            return out

        decode._infb_precision = True
        try:
            vae.decode = decode
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "record_error", "where": "precision:vae", "error": f"{type(exc).__name__}: {exc}"[:200]})
    _observe_scheduler_precision(mods.get("scheduler"))


def _observe_scheduler_precision(sch):
    """Observe the receiving scheduler, including request-local shallow/deep copies.

    Instance closures over bound methods survive deepcopy unchanged: a copied
    scheduler would prepare/step the original object's state. Class descriptors
    bind the actual receiver instead. Only observed instances and their copies
    opt in; per-instance bookkeeping is external to the copied scheduler state.
    """
    import functools
    import inspect
    import weakref

    step, set_ts = getattr(sch, "step", None), getattr(sch, "set_timesteps", None)
    if not callable(step) or not callable(set_ts):
        return
    enabled = "_infb_scheduler_precision_enabled"
    try:
        if getattr(step, "_infb_precision", False):
            setattr(sch, enabled, True)
            return
        cls = type(sch)


        if not all(inspect.ismethod(fn) and fn.__self__ is sch and
                   getattr(cls, name) is fn.__func__
                   for name, fn in (("step", step), ("set_timesteps", set_ts))):
            raise TypeError("scheduler observer needs receiver-bound step/set_timesteps methods")
        step_fn, set_fn = step.__func__, set_ts.__func__
        fresh = weakref.WeakKeyDictionary()

        def state(instance):
            return fresh.setdefault(instance, {"v": True, "n": 0})

        @functools.wraps(set_fn)
        def set_timesteps(instance, *args, **kwargs):
            try:
                if getattr(instance, enabled, False):
                    state(instance)["v"] = True
            except Exception:  # noqa: BLE001 - recording cannot alter scheduling
                pass
            return set_fn(instance, *args, **kwargs)

        @functools.wraps(step_fn)
        def step_(instance, *args, **kwargs):
            try:
                if getattr(instance, enabled, False):
                    current = state(instance)
                    if current["v"] and current["n"] < 500:
                        current["v"], current["n"] = False, current["n"] + 1
                        got = inspect.signature(step_fn).bind(instance, *args, **kwargs).arguments
                        sample = got.get("sample")
                        rec = {"event": "precision_latents", "sample": _dtype_name(getattr(sample, "dtype", None)),
                               "model_output": _dtype_name(getattr(got.get("model_output"), "dtype", None))}
                        if _PW and _joint_rows(sample) is not None:
                            rec["sample_sha256"] = _joint_sha(sample.detach(), (_dtype_name(sample.dtype),))
                            rec["sample_shape"] = list(sample.shape)
                        _emit(rec)
            except Exception:  # noqa: BLE001
                pass
            ms = _ms_scheduler_before(step_fn, instance, args, kwargs)
            result = step_fn(instance, *args, **kwargs)
            if ms is not None:
                _ms_scheduler_after(ms, instance, result)
            return result

        step_._infb_precision = True
        cls.step, cls.set_timesteps = step_, set_timesteps
        setattr(sch, enabled, True)
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "record_error", "where": "precision:scheduler",
               "error": f"{type(exc).__name__}: {exc}"[:200]})


_VAE_ARM = "VAE_ARM"
_VAE_TAP_MAX = 8
_VAE_MAX_LINES = 64
_vt = {"depth": 0, "cur": None, "n": 0, "taps": 0}


def _hw(t):
    """[H, W] of a tensor's last two dims (SGLang may carry an image latent as [B, C, 1, H, W])."""
    shape = getattr(t, "shape", None)
    try:
        return [int(shape[-2]), int(shape[-1])] if shape is not None and len(shape) >= 2 else None
    except Exception:  # noqa: BLE001
        return None


def _num_attr(obj, name, kind):
    try:
        v = getattr(obj, name, None)
        return None if v is None else kind(v)
    except Exception:  # noqa: BLE001
        return None


def _in_precision_probe():
    """True while one of the precision witness's consumption-probe re-runs is open (_pw_maybe_probe opens its plain
    re-run with an op sequence and each perturbed re-run with a target index; a served call opens neither). Reads
    the witness state only."""
    try:
        return any(a.get("seq") is not None or (isinstance(a.get("target"), int) and a["target"] >= 0)
                   for _, a in _pw["stack"])
    except Exception:  # noqa: BLE001
        return False


def _observe_vae_tiling(mods):
    ' observe vae tiling.'
    torch = sys.modules.get("torch")
    vae = mods.get("vae")
    dec = getattr(vae, "decode", None)
    if vae is None or not callable(dec) or getattr(dec, "_infb_vae_tiling", False):
        return
    counted = False
    reg = getattr(getattr(vae, "decoder", None), "register_forward_hook", None)
    if callable(reg):
        def _tile_hook(module, args, kwargs, output):
            try:
                cur = _vt["cur"]
                if cur is not None:
                    cur["probe_passes" if _in_precision_probe() else "tiles"] += 1
            except Exception:  # noqa: BLE001
                pass
            return None

        try:
            reg(_tile_hook, with_kwargs=True)
            counted = True
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "record_error", "where": "vae_tiling:decoder", "error": f"{type(exc).__name__}: {exc}"[:200]})
    wrapped = []
    for name in ("tiled_decode", "blend_v", "blend_h"):
        fn = getattr(vae, name, None)
        if not callable(fn) or getattr(fn, "_infb_vae_tiling", False):
            continue

        def _wrapper(*args, _fn=fn, _name=name, **kwargs):
            try:
                cur = _vt["cur"]
                if cur is not None and _in_precision_probe():
                    if _name != "tiled_decode":
                        cur["probe_blends"] += 1
                elif cur is not None:
                    if _name == "tiled_decode":
                        cur["tiled"] = True
                    else:
                        cur["blends"] += 1
                        ext = kwargs.get("blend_extent", args[2] if len(args) > 2 else None)
                        if ext is not None:
                            cur["extents"].append(int(ext))
            except Exception:  # noqa: BLE001
                pass
            return _fn(*args, **kwargs)

        _wrapper._infb_vae_tiling = True
        try:
            setattr(vae, name, _wrapper)
            wrapped.append(name)
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "record_error", "where": f"vae_tiling:{name}", "error": f"{type(exc).__name__}: {exc}"[:200]})
    def decode(*args, **kwargs):
        z = kwargs.get("z", args[0] if args else None)
        outer = _vt["depth"] == 0
        cur = {"tiled": False, "tiles": 0, "blends": 0, "extents": [], "probe_passes": 0, "probe_blends": 0}
        dev = getattr(getattr(z, "device", None), "type", None)
        mem0 = peak0 = None
        if outer:
            _vt["cur"] = cur
            if torch is not None and dev == "cuda":
                try:
                    mem0 = int(torch.cuda.memory_allocated(z.device))
                    peak0 = int(torch.cuda.max_memory_allocated(z.device))
                except Exception:  # noqa: BLE001
                    pass
        _vt["depth"] += 1
        t0 = time.time()
        try:
            out = dec(*args, **kwargs)
        finally:
            _vt["depth"] -= 1
            if outer:
                _vt["cur"] = None
        if outer:
            try:
                _vt["n"] += 1
                n = _vt["n"]
                if n <= _VAE_MAX_LINES:
                    peak = note = None
                    if torch is not None and dev == "cuda" and peak0 is not None:
                        try:
                            p1 = int(torch.cuda.max_memory_allocated(z.device))
                            if p1 > peak0:
                                peak = p1
                            else:
                                note = ("the decode did not raise the allocator's running peak (%d bytes before it): "
                                        "its own peak is not readable without a reset, which this file never does" % peak0)
                        except Exception:  # noqa: BLE001
                            pass
                    sample = getattr(out, "sample", out)
                    if isinstance(sample, (list, tuple)):
                        sample = sample[0] if sample else None

                    cfg = getattr(vae, "config", None)
                    cfg = getattr(cfg, "arch_config", cfg)
                    replay_config = {}
                    for key in ("in_channels", "out_channels", "latent_channels", "block_out_channels",
                                "down_block_types", "up_block_types", "layers_per_block", "act_fn",
                                "norm_num_groups", "sample_size", "scaling_factor", "shift_factor",
                                "force_upcast", "use_quant_conv", "use_post_quant_conv"):
                        value = cfg.get(key) if isinstance(cfg, dict) else getattr(cfg, key, None)
                        if isinstance(value, tuple):
                            value = list(value)
                        if isinstance(value, (str, bool, int, float, list)):
                            replay_config[key] = value
                    _emit({"event": "vae_decode", "n": n, "rank": _rank(), "tiled": cur["tiled"],
                           "tiles": cur["tiles"] if counted else None, "blends": cur["blends"],
                           "blend_extent": min(cur["extents"]) if cur["extents"] else None,
                           "tile_latent": _num_attr(vae, "tile_latent_min_size", int),
                           "tile_sample": _num_attr(vae, "tile_sample_min_size", int),
                           "overlap": _num_attr(vae, "tile_overlap_factor", float),
                           "use_tiling": _num_attr(vae, "use_tiling", bool),
                           "latent_hw": _hw(z), "out_hw": _hw(sample),
                           "in_dtype": _dtype_name(getattr(z, "dtype", None)), "out_dtype": _out_dtype(out),
                           "vae_config": replay_config,
                           "device": dev, "peak_mem_bytes": peak, "mem_before_bytes": mem0,
                           "peak_before_bytes": peak0, "peak_note": note,
                           "peak_delta_bytes": (peak - mem0) if peak is not None and mem0 is not None else None,
                           "probe_passes": cur["probe_passes"] if counted else None,
                           "probe_blends": cur["probe_blends"], "s": round(time.time() - t0, 3)})
                    if _TAP and _vt["taps"] < _VAE_TAP_MAX and os.path.exists(os.path.join(_TAP, _VAE_ARM)):
                        _vt["taps"] += 1
                        _tap_save("vae_decode_%d" % n, {"z": _cpu(z), "sample": _cpu(sample)})
            except Exception as exc:  # noqa: BLE001 - a lost line, never a lost decode
                _emit({"event": "record_error", "where": "vae_tiling", "error": f"{type(exc).__name__}: {exc}"[:200]})
        return out

    decode._infb_vae_tiling = True
    try:
        vae.decode = decode
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "record_error", "where": "vae_tiling:decode", "error": f"{type(exc).__name__}: {exc}"[:200]})
        return
    _emit({"event": "vae_tiling_installed", "decoder_counted": counted, "wrapped": wrapped,
           "cls": _name(type(vae)), "tap": bool(_TAP)})


_OFF = bool(_OUT and os.environ.get("INFB_SD3_OFFLOAD_OBSERVE"))

_OFF_DEVICE = os.environ.get("INFB_SD3_OFFLOAD_OBSERVE_DEVICE", "cuda") if _OFF else "cuda"


_OFF_MAX_LINES = 4096
_off = {"n": 0, "blocks": None, "list": None, "hooked": False, "cur": None, "depth": 0}


def _off_bytes(mods_iter, seen):
    """Bytes of the tensors (parameters and buffers) of the given modules whose storage is on _OFF_DEVICE, each
    storage counted once across `seen` (the manager maps a layer's parameters onto ONE flat GPU buffer, and parks an
    offloaded parameter on a shared one-element placeholder: both are one storage)."""
    n = 0
    for mod in mods_iter:
        for t in list(mod.parameters()) + list(mod.buffers()):
            try:
                dev = t.device
                if dev.type != _OFF_DEVICE:
                    continue
                st = t.untyped_storage()
                key = (int(st.data_ptr()), dev.index)
                if key in seen:
                    continue
                seen.add(key)
                n += int(st.nbytes())
            except Exception:  # noqa: BLE001 - an unreadable tensor is not resident evidence
                continue
    return n


def _off_is_list(child):
    """A ModuleList / Sequential (torch loaded, as in every engine), else -- stand-in objects of the CPU unit test -- any
    indexable, sized child that is not a mapping."""
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "nn"):
        return isinstance(child, (torch.nn.ModuleList, torch.nn.Sequential))
    return hasattr(child, "__getitem__") and hasattr(child, "__len__") and not hasattr(child, "keys")


def _off_block_list(transformer):
    """The DiT's block list: its direct child ModuleList (or Sequential) with the most entries (>= 2), and its name."""
    best, name = None, None
    try:
        for nm, child in transformer.named_children():
            if _off_is_list(child) and len(child) >= 2:
                if best is None or len(child) > len(best):
                    best, name = child, nm
    except Exception:  # noqa: BLE001
        return None, None
    return best, name


def _off_mem():
    torch = sys.modules.get("torch")
    if torch is None or _OFF_DEVICE != "cuda":
        return None
    try:
        return int(torch.cuda.memory_allocated())
    except Exception:  # noqa: BLE001
        return None


def _off_hook_blocks(blocks):
    """Per-block forward pre-hooks, registered at the DiT's FIRST forward -- after any offload manager registered its
    own (those are built between the pipeline and the first request), so this one runs after the manager's and sees
    the block as the manager left it to run. Pass-through: returns nothing."""
    def make(i):
        def pre(module, args):
            try:
                cur = _off["cur"]
                if cur is None:
                    return None
                own = _off_bytes([module], set())
                cur["self"].append(own)
                cur["total"].append(_off_bytes(list(blocks), set()))
                m = _off_mem()
                if m is not None:
                    cur["mem_max"] = m if cur["mem_max"] is None else max(cur["mem_max"], m)
            except Exception:  # noqa: BLE001
                pass
            return None
        return pre

    for i, blk in enumerate(blocks):
        blk.register_forward_pre_hook(make(i))


def _observe_dit_residency(mods):
    transformer = mods.get("transformer")
    reg = getattr(transformer, "register_forward_hook", None)
    pre = getattr(transformer, "register_forward_pre_hook", None)
    if not _OFF or not callable(reg) or not callable(pre) or getattr(transformer, "_infb_offload_observed", False):
        return
    blocks, name = _off_block_list(transformer)
    _off["blocks"], _off["list"] = blocks, name

    def dit_pre(module, args, kwargs):
        try:
            if _off["depth"] == 0:
                if not _off["hooked"] and _off["blocks"] is not None:
                    _off["hooked"] = True
                    _off_hook_blocks(_off["blocks"])
                blocks_ = list(_off["blocks"]) if _off["blocks"] is not None else []
                _off["cur"] = {"self": [], "total": [], "mem_max": None, "t0": time.time(),
                               "start_total": _off_bytes(blocks_, set()) if blocks_ else None, "mem_start": _off_mem()}
            _off["depth"] += 1
        except Exception:  # noqa: BLE001
            pass
        return None

    def dit_post(module, args, kwargs, output):
        try:
            _off["depth"] = max(0, _off["depth"] - 1)
            if _off["depth"] != 0:
                return None
            cur, _off["cur"] = _off["cur"], None
            if cur is None:
                return None
            _off["n"] += 1
            if _off["n"] > _OFF_MAX_LINES:
                if _off["n"] == _OFF_MAX_LINES + 1:
                    _emit({"event": "dit_residency_truncated", "n": _off["n"],
                           "max_lines": _OFF_MAX_LINES, "rank": _rank_int()})
                return None
            blocks_ = list(_off["blocks"]) if _off["blocks"] is not None else []
            _emit({"event": "dit_residency", "n": _off["n"], "rank": _rank_int(), "list": _off["list"],
                   "n_blocks": len(blocks_), "self": cur["self"], "total": cur["total"],
                   "start_total": cur["start_total"], "end_total": _off_bytes(blocks_, set()) if blocks_ else None,
                   "mem_start": cur["mem_start"], "mem_max": cur["mem_max"], "mem_end": _off_mem(),
                   "device": _OFF_DEVICE, "t0": cur["t0"], "t1": time.time()})
        except Exception as exc:  # noqa: BLE001 - a lost line, never a lost forward
            _emit({"event": "record_error", "where": "dit_residency", "error": f"{type(exc).__name__}: {exc}"[:200]})
        return None

    try:
        pre(dit_pre, with_kwargs=True)
        reg(dit_post, with_kwargs=True, always_call=True)
        transformer._infb_offload_observed = True
        _emit({"event": "dit_residency_installed", "list": name, "n_blocks": len(blocks) if blocks is not None else 0,
               "cls": _name(type(transformer)), "device": _OFF_DEVICE})
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "record_error", "where": "dit_residency:install", "error": f"{type(exc).__name__}: {exc}"[:200]})


_CFG = bool(_OUT and os.environ.get("INFB_SD3_CFG_OBSERVE"))
_GROUP_MODULE = "sglang.multimodal_gen.runtime.distributed.group_coordinator"
_PSTATE_MODULE = "sglang.multimodal_gen.runtime.distributed.parallel_state"
_CFG_COLLECTIVES = ("all_reduce", "all_gather", "broadcast", "gather", "reduce_scatter", "send", "recv",
                    "broadcast_object", "send_object", "recv_object", "all_to_all_4D")
_CFG_MAX_LINES = 400
_cfgo = {"n": 0, "coll": 0}


def _cfg_fp(t):
    """A fingerprint of a tensor for telling CFG branches apart: shape, dtype and four moments of every 61st value (in
    float64, in order), as a string. Equal tensors give equal strings; two branches' conditionings differ in all four
    (stdlib only: no hashing module, the worker's torch does the arithmetic). None for a non-tensor."""
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(t, torch.Tensor):
        return None
    try:
        flat = t.detach().reshape(-1)[::61].double()
        w = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
        m = [float(flat.sum()), float((flat * flat).sum()), float((flat * w).sum()), float(flat.abs().max())]
        return "%s|%s|%s" % (tuple(t.shape), str(t.dtype).replace("torch.", ""), ",".join("%.12g" % x for x in m))
    except Exception:  # noqa: BLE001
        return None


def _cfg_conditioning_rows(hidden_states, encoder_hidden_states, pooled_projections):
    """Read each real forward row; no candidate IDs or self-reported layout."""
    torch = sys.modules.get("torch")
    tensors = (hidden_states, encoder_hidden_states, pooled_projections)
    if torch is None or not all(isinstance(t, torch.Tensor) and t.ndim > 0 for t in tensors):
        return None
    count = int(hidden_states.shape[0])
    if count < 1 or any(int(t.shape[0]) != count for t in tensors[1:]):
        return None
    return [{"ehs": _cfg_fp(encoder_hidden_states[i:i + 1]),
             "pooled": _cfg_fp(pooled_projections[i:i + 1])} for i in range(count)]


def _cfg_rank():
    """The CFG-group rank, read from parallel_state if the worker loaded it (no import of the engine)."""
    try:
        fn = getattr(sys.modules.get(_PSTATE_MODULE), "get_classifier_free_guidance_rank", None)
        return int(fn()) if callable(fn) else None
    except Exception:  # noqa: BLE001
        return None


def _patch_group_coordinator(module):
    """Count the collectives made on the CFG group (pass-through): each listed GroupCoordinator method, wrapped on the
    class, adds one when `self` is parallel_state's _CFG object at call time. With the CLIP observer armed the
    same wrapper also counts, inside a CLIP text encoder's forward, every listed call on ANY group, by method name."""
    if not (_CFG or _CLIP):
        return
    cls = getattr(module, "GroupCoordinator", None)
    if not isinstance(cls, type):
        return
    wrapped = []
    for name in _CFG_COLLECTIVES:
        fn = cls.__dict__.get(name)
        if not callable(fn) or getattr(fn, "_infb_cfg", False):
            continue

        def counted(self, *args, _fn=fn, _name=name, **kwargs):
            try:
                if _CFG and getattr(sys.modules.get(_PSTATE_MODULE), "_CFG", None) is self:
                    _cfgo["coll"] += 1
                if _CLIP and _clipo["depth"] > 0 and _clipo["gc"] is not None:
                    _clipo["gc"][_name] = _clipo["gc"].get(_name, 0) + 1
            except Exception:  # noqa: BLE001
                pass
            return _fn(self, *args, **kwargs)

        counted._infb_cfg = True
        setattr(cls, name, counted)
        wrapped.append(name)
    _emit({"event": "patched", "target": _GROUP_MODULE + ".GroupCoordinator", "cfg_collectives": wrapped})


def _observe_cfg(mods):
    transformer = mods.get("transformer")
    reg = getattr(transformer, "register_forward_hook", None)
    if not _CFG or not callable(reg) or getattr(transformer, "_infb_cfg_observed", False):
        return

    def hook(module, args, kwargs, output):
        try:
            _cfgo["n"] += 1
            if _cfgo["n"] > _CFG_MAX_LINES:
                return None
            inputs = _dit_inputs(module, args, kwargs)
            hs = inputs["hidden_states"]
            _emit({"event": "dit_cfg", "n": _cfgo["n"], "rank": _rank_int(), "cfg_rank": _cfg_rank(),
                   "rows": int(hs.shape[0]) if getattr(hs, "shape", None) is not None else None,
                   "ehs": _cfg_fp(inputs["encoder_hidden_states"]),
                   "pooled": _cfg_fp(inputs["pooled_projections"]),
                   "conditioning_rows": _cfg_conditioning_rows(hs, inputs["encoder_hidden_states"],
                                                              inputs["pooled_projections"]),
                   "cfg_coll": _cfgo["coll"],
                   "t": time.time()})
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "record_error", "where": "dit_cfg", "error": f"{type(exc).__name__}: {exc}"[:200]})
        return None

    try:
        reg(hook, with_kwargs=True)
        transformer._infb_cfg_observed = True
        _emit({"event": "dit_cfg_installed", "rank": _rank_int(), "cls": _name(type(transformer))})
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "record_error", "where": "dit_cfg:install", "error": f"{type(exc).__name__}: {exc}"[:200]})


_CLIP = bool(_OUT and os.environ.get("INFB_SD3_CLIP_OBSERVE"))
_CLIP_DIR = os.environ.get("INFB_SD3_CLIP_OBSERVE_DIR") if _CLIP else None
_CLIP_ENCODERS = ("text_encoder", "text_encoder_2")
_CLIP_DIST_OPS = ("all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter_tensor", "reduce_scatter",
                  "all_to_all_single", "all_to_all", "broadcast")
_CLIP_MAX_CALLS = 1200
_CLIP_MAX_LINES = 64
_clipo = {"depth": 0, "enc": None, "gc": None, "dist": None, "n": {}, "lines": 0, "census": None, "hooks": [],
          "done": set(), "dist_installed": False}


def _clip_dist_install():
    """Wrap torch.distributed's collectives once (pass-through), counting the calls made inside a CLIP forward."""
    if _clipo["dist_installed"]:
        return
    _clipo["dist_installed"] = True
    dist = getattr(sys.modules.get("torch"), "distributed", None)
    if dist is None:
        return
    wrapped = []
    for name in _CLIP_DIST_OPS:
        orig = getattr(dist, name, None)
        if orig is None or getattr(orig, "_infb_clip", False):
            continue

        def counted(*a, _o=orig, _n=name, **k):
            try:
                if _clipo["depth"] > 0 and _clipo["dist"] is not None:
                    _clipo["dist"][_n] = _clipo["dist"].get(_n, 0) + 1
            except Exception:  # noqa: BLE001
                pass
            return _o(*a, **k)

        counted._infb_clip = True
        try:
            setattr(dist, name, counted)
            wrapped.append(name)
        except Exception:  # noqa: BLE001
            pass
    _emit({"event": "patched", "target": "clip_observer torch.distributed", "ops": wrapped, "rank": _rank_int()})


def _clip_is_embedding(sub):
    return "Embedding" in type(sub).__name__


def _clip_census_begin(module):
    calls = []
    _clipo["census"] = calls
    named = getattr(module, "named_modules", None)
    if not callable(named):
        return
    for name, sub in named():
        shp = _shape(getattr(sub, "weight", None))
        reg = getattr(sub, "register_forward_hook", None)
        if shp is None or len(shp) != 2 or not callable(reg) or _clip_is_embedding(sub):
            continue

        def hook(m, args, out, _name=name):
            try:
                if len(calls) >= _CLIP_MAX_CALLS:
                    return None
                xs = _shape(args[0]) if args else None
                ys = _shape(out[0] if isinstance(out, (tuple, list)) else out)
                if xs and ys and xs[-1] > 0:
                    n = 1
                    for d in xs:
                        n *= d
                    calls.append([str(_name), n // xs[-1], xs[-1], ys[-1]])
            except Exception:  # noqa: BLE001
                pass
            return None

        try:
            _clipo["hooks"].append(reg(hook))
        except Exception:  # noqa: BLE001
            pass


def _clip_census_end(enc):
    calls = _clipo["census"]
    _clipo["census"] = None
    for h in _clipo["hooks"]:
        try:
            h.remove()
        except Exception:  # noqa: BLE001
            pass
    _clipo["hooks"] = []
    if calls is None:
        return
    _emit({"event": "clip_compute", "encoder": enc, "rank": _rank_int(), "macs": sum(r * i * o for _, r, i, o in calls),
           "n_calls": len(calls), "truncated": len(calls) >= _CLIP_MAX_CALLS, "calls": calls[:300]})


def _clip_capture(enc, args, kwargs, output):
    torch = sys.modules.get("torch")
    if torch is None or not _CLIP_DIR:
        return
    path = os.path.join(_CLIP_DIR, "clip_%s_rank%d.pt" % (enc, _rank_int()))
    try:
        ids = kwargs.get("input_ids", args[0] if args else None)
        payload = {"input_ids": _cpu(ids), **_encoder_outputs(output)}
        torch.save(payload, path + ".tmp")
        os.replace(path + ".tmp", path)
        _emit({"event": "clip_capture", "encoder": enc, "rank": _rank_int(), "file": path,
               "shapes": {k: list(v.shape) for k, v in payload.items() if hasattr(v, "shape")},
               "dtypes": {k: str(v.dtype) for k, v in payload.items() if hasattr(v, "dtype")}})
    except Exception as exc:  # noqa: BLE001 - a failed capture loses the capture, never the request
        _emit({"event": "record_error", "where": "clip_capture " + enc, "error": f"{type(exc).__name__}: {exc}"[:300]})


def _clip_local_shapes(module, needle, limit=24):
    out = {}
    named = getattr(module, "named_parameters", None)
    if not callable(named):
        return out
    try:
        for pname, p in named():
            if needle in pname and len(out) < limit:
                t, _ = _local(p)
                shp = _shape(t)
                if shp is not None and len(shp) == 2:
                    out[pname] = shp
    except Exception:  # noqa: BLE001
        pass
    return out


def _clip_dims(module):
    cfg = getattr(module, "config", None)
    arch = getattr(cfg, "arch_config", cfg)
    out = {}
    for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers", "projection_dim"):
        v = getattr(arch, k, None)
        if isinstance(v, int):
            out[k] = v
    return out


def _observe_clip(mods):
    if not _CLIP:
        return
    scope = {}
    for key, needle in (("text_encoder_3", "block.0."), ("transformer", "transformer_blocks.0.")):
        if mods.get(key) is not None:
            scope[key] = _clip_local_shapes(mods.get(key), needle, limit=8)
    for enc in _CLIP_ENCODERS:
        te = mods.get(enc)
        reg = getattr(te, "register_forward_hook", None)
        pre_reg = getattr(te, "register_forward_pre_hook", None)
        if not callable(reg) or not callable(pre_reg) or getattr(te, "_infb_clip_observed", False):
            continue

        def pre(module, args, kwargs, _enc=enc):
            try:
                _clip_dist_install()
                _clipo["depth"] += 1
                if _clipo["depth"] == 1:
                    _clipo["enc"], _clipo["gc"], _clipo["dist"] = _enc, {}, {}
                    if _enc not in _clipo["done"]:
                        _clip_census_begin(module)
            except Exception as exc:  # noqa: BLE001
                _emit({"event": "record_error", "where": "clip_pre", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return None

        def post(module, args, kwargs, output, _enc=enc):
            try:
                if _clipo["depth"] <= 0:
                    return None
                _clipo["depth"] -= 1
                if _clipo["depth"] > 0:
                    return None
                first = _enc not in _clipo["done"]
                if first:
                    _clipo["done"].add(_enc)
                    _clip_census_end(_enc)
                    _clip_capture(_enc, args, kwargs, output)
                n = _clipo["n"][_enc] = _clipo["n"].get(_enc, 0) + 1
                _clipo["lines"] += 1
                if _clipo["lines"] <= _CLIP_MAX_LINES:
                    ids = kwargs.get("input_ids", args[0] if args else None)
                    _emit({"event": "clip_call", "encoder": _enc, "n": n, "rank": _rank_int(), "first": first,
                           "gc": dict(_clipo["gc"] or {}), "dist": dict(_clipo["dist"] or {}), "ids": _shape(ids)})
                _clipo["gc"], _clipo["dist"], _clipo["enc"] = None, None, None
            except Exception as exc:  # noqa: BLE001
                _emit({"event": "record_error", "where": "clip_post", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return None

        try:
            pre_reg(pre, with_kwargs=True)
            reg(post, with_kwargs=True)
            te._infb_clip_observed = True
            _emit({"event": "clip_layout", "encoder": enc, "rank": _rank_int(), "cls": _name(type(te)),
                   "dims": _clip_dims(te), "layer0": _clip_local_shapes(te, "layers.0."), "scope": scope})
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "record_error", "where": "clip:install", "error": f"{type(exc).__name__}: {exc}"[:200]})


def _record_served(pipeline):
    try:
        mods = getattr(pipeline, "modules", None)
        out = {}
        if isinstance(mods, dict):
            for key, obj in list(mods.items())[:64]:
                try:
                    out[str(key)] = _objinfo(obj, tree=str(key) in _MODEL_COMPONENTS)
                except Exception as exc:  # noqa: BLE001
                    out[str(key)] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        _emit({"event": "served", "pipeline": _name(type(pipeline)), "modules": out,
               "modules_is_dict": isinstance(mods, dict)})
        if isinstance(mods, dict):
            _count_dit_calls(mods.get("transformer"))
            _observe_precision(mods)
            _observe_vae_tiling(mods)
            _observe_dit_residency(mods)
            _observe_cfg(mods)
            _observe_clip(mods)
            if _CENSUS:
                _census(mods)
        if _TAP and isinstance(mods, dict):
            _install_taps(mods)
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "record_error", "where": "served", "error": f"{type(exc).__name__}: {exc}"[:300]})


def _patch_pipeline(module):
    import functools

    target = _PIPELINE_MODULE + ".ComposedPipelineBase.__init__"
    cls = getattr(module, "ComposedPipelineBase", None)
    original = cls.__dict__.get("__init__") if isinstance(cls, type) else None
    if original is None or getattr(original, "_infb_recorded", False):
        _emit({"event": "patch_skipped", "target": target,
               "why": "already wrapped" if original is not None else "no ComposedPipelineBase.__init__"})
        return

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _record_served(self)

    __init__._infb_recorded = True
    cls.__init__ = __init__
    _emit({"event": "patched", "target": target, "file": getattr(module, "__file__", None)})


_PATCHERS = {
    _TRANSFORMERS_MODULE: _patch_transformers,
    _DIFFUSERS_MODULE: _patch_diffusers,
    _LOADER_MODULE: _patch_loader,
    _PIPELINE_MODULE: _patch_pipeline,
    _ATTN_BACKEND_MODULE: _patch_attention_backend,
}
if _CFG:
    _PATCHERS[_GROUP_MODULE] = _patch_group_coordinator
if _CLIP:
    _PATCHERS[_GROUP_MODULE] = _patch_group_coordinator


def _patch(fullname, module):
    try:
        _PATCHERS[fullname](module)
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "patch_error", "target": fullname, "error": f"{type(exc).__name__}: {exc}"[:300]})


class _PostImportPatcher:
    """A meta-path finder for the five target names only. It asks the REST of sys.meta_path for the spec, so
    each module resolves exactly as it would without this file (an editable install's finder included), then
    wraps that spec's exec_module to patch the module once it has run."""

    def __init__(self):
        self._inflight = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _PATCHERS or fullname in self._inflight:
            return None
        self._inflight.add(fullname)
        try:
            spec = None
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(fullname, path, target)
                if spec is not None:
                    break
        finally:
            self._inflight.discard(fullname)
        loader = getattr(spec, "loader", None)
        real_exec = getattr(loader, "exec_module", None)
        if spec is None or real_exec is None:
            if spec is not None:
                _emit({"event": "patch_skipped", "target": fullname, "why": "spec has no exec_module"})
            return None

        def exec_module(module):
            real_exec(module)
            if getattr(module, "__name__", None) == fullname:
                _patch(fullname, module)

        loader.exec_module = exec_module
        return spec


def _chain_shadowed_sitecustomize():
    """Run the sitecustomize this file shadows, if there is one, so prepending this directory changes nothing else."""
    import importlib.util
    seen = getattr(sys, "_infb_sitecustomize_seen", None)
    if seen is None:
        seen = set()
        sys._infb_sitecustomize_seen = seen
    seen.add(os.path.realpath(__file__))

    for entry in sys.path:
        try:
            if not entry or os.path.abspath(entry) == _HERE:
                continue
            candidate = os.path.join(entry, "sitecustomize.py")
            if os.path.isfile(candidate) and os.path.realpath(candidate) not in seen:
                spec = importlib.util.spec_from_file_location("_infb_shadowed_sitecustomize", candidate)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return candidate
        except Exception:  # noqa: BLE001 - someone else's start-up hook failing must not stop this one
            return None
    return None


if _OUT and os.environ.get("INFB_SD3_REQUEST_OBSERVE"):
    try:
        import request_wire_observer
        request_wire_observer.install(_emit)
    except Exception as exc:
        _emit({"event": "request_wire_error", "where": "install", "error": str(exc)[:250]})

if _OUT:
    _emit({"event": "installed", "ppid": os.getppid(), "python": sys.executable,
           "argv": [str(a)[:160] for a in (getattr(sys, "orig_argv", None) or [])[:4]],
           "targets": sorted(_PATCHERS)})
    sys.meta_path.insert(0, _PostImportPatcher())
    for _name_already in [n for n in _PATCHERS if n in sys.modules]:
        _emit({"event": "late_patch", "target": _name_already})
        _patch(_name_already, sys.modules[_name_already])
_CHAINED = _chain_shadowed_sitecustomize()
if _OUT and _CHAINED:
    _emit({"event": "chained", "file": _CHAINED})
