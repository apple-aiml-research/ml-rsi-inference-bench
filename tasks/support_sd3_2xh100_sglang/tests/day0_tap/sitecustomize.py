#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 sitecustomize utilities.'


import hashlib
import json
import os
import sys
import time

_DIR = os.environ.get("INFB_SD3_DAY0_TAP_DIR")
_LIB = os.environ.get("INFB_SD3_DAY0_LIB")
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_MODULE = "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base"
_TRITON_MODULE = "triton.runtime.jit"
_FLASHINFER_MODULE = "flashinfer.gemm.gemm_base"
_FLASHINFER_ENTRY = "fp8_blockscale_gemm_sm90"
_FLASHINFER_JIT_MODULE = "flashinfer.jit.core"
_FI_MANIFEST = os.environ.get("INFB_SD3_DAY0_FI_MANIFEST")
_MAX_EVENTS = 50000
_state = {"events": 0, "arm_key": None, "arm": None, "labels": {}, "fp8": None, "fp8_done": set()}


def _emit(rec):
    if not _DIR:
        return
    try:
        _state["events"] += 1
        if _state["events"] > _MAX_EVENTS:
            return
        rec.setdefault("pid", os.getpid())
        rec.setdefault("t", time.time())
        line = (json.dumps(rec, separators=(",", ":"), default=str) + "\n").encode("utf-8")
        fd = os.open(os.path.join(_DIR, "events.jsonl"),
                     os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:  # noqa: BLE001 - the tap must never take a process down
        pass


def _arm():
    """The current ARM record, or None. Re-read only when the file changes."""
    try:
        st = os.stat(os.path.join(_DIR, "ARM"))
    except OSError:
        _state["arm_key"], _state["arm"] = None, None
        return None
    key = (st.st_mtime_ns, st.st_size, st.st_ino)
    if key != _state["arm_key"]:
        try:
            with open(os.path.join(_DIR, "ARM"), encoding="utf-8") as fh:
                _state["arm"] = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "arm_unreadable", "error": f"{type(exc).__name__}: {exc}"[:200]})
            _state["arm"] = None
        _state["arm_key"] = key
    return _state["arm"]


def _cpu(x):
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().to("cpu").contiguous()
    if isinstance(x, (list, tuple)):
        return [_cpu(v) for v in x]
    return x


def _saveable(x):
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        return _cpu(x)
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, (list, tuple)) and torch is not None and all(isinstance(v, torch.Tensor) for v in x):
        return [_cpu(v) for v in x]
    return repr(x)[:200]


def _first_tensor(obj):
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for o in obj:
            if isinstance(o, torch.Tensor):
                return o
    return getattr(obj, "sample", None) if not isinstance(obj, (list, tuple)) else None


def _timestep_of(args, kwargs):
    t = kwargs.get("timestep")
    if t is None:
        t = kwargs.get("t")
    torch = sys.modules.get("torch")
    try:
        if torch is not None and isinstance(t, torch.Tensor):
            return float(t.detach().reshape(-1)[0].to("cpu", torch.float64))
        if isinstance(t, (int, float)):
            return float(t)
    except Exception:  # noqa: BLE001
        pass
    return None


def _rank_world():
    """(rank, world size) of this process: torch.distributed when it is initialised, else RANK / WORLD_SIZE."""
    torch = sys.modules.get("torch")
    try:
        dist = getattr(torch, "distributed", None) if torch is not None else None
        if dist is not None and dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
    except Exception:  # noqa: BLE001
        pass
    try:
        return int(os.environ.get("RANK", "0") or 0), int(os.environ.get("WORLD_SIZE", "1") or 1)
    except ValueError:
        return 0, 1


def _not_the_recording_rank():
    rank, world = _rank_world()
    if world > 1 and rank != 0:
        if not _state.get("rank_skip_emitted"):
            _state["rank_skip_emitted"] = True
            _emit({"event": "rank_skip", "rank": rank, "world": world})
        return True
    return False


def _position(label, ts):
    """(step index, pass index) of a forward under `label`, from its timestep."""
    lab = _state["labels"].setdefault(label, {"timesteps": [], "passes": {}, "calls": 0})
    lab["calls"] += 1
    if ts is None:
        step = lab["calls"] - 1
    else:
        key = round(ts, 6)
        if key not in lab["timesteps"]:
            lab["timesteps"].append(key)
        step = lab["timesteps"].index(key)
    p = lab["passes"].get(step, 0)
    lab["passes"][step] = p + 1
    return step, p


def _sha(t):
    return hashlib.sha256(t.contiguous().view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes()).hexdigest()


def _dtype_name(dt):
    """'bfloat16' for torch.bfloat16; None for anything that is not a torch dtype (a port without orig_dtype)."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(dt, torch.dtype):
        return str(dt).replace("torch.", "")
    return None


def _store_census(dit):
    """sha256 of every 2-D float8 tensor the DiT holds: as stored, transposed, and cut into 1536-row slabs (a fused
    QKV holds three of the asset's tensors side by side). Hashing happens on CPU copies."""
    import torch
    out = []
    seen = set()
    for mname, mod in dit.named_modules():
        for pname, t in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
            if not isinstance(t, torch.Tensor) or t.dim() != 2 or t.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                continue
            if id(t) in seen:
                continue
            seen.add(id(t))
            ptr = int(t.data_ptr())
            c = t.detach().to("cpu")
            rec = {"name": f"{mname}.{pname}" if mname else pname, "shape": list(c.shape), "dtype": str(c.dtype),
                   "sha256": _sha(c), "sha256_t": _sha(c.t()), "data_ptr": ptr}
            if c.shape[0] % 1536 == 0 and c.shape[0] > 1536:
                rec["slabs"] = [_sha(c[i:i + 1536]) for i in range(0, c.shape[0], 1536)]
            out.append(rec)
    return out


def _scale_census(dit):
    """sha256 of every 2-D float32 tensor a module holds next to a 2-D float8 weight whose 128x128 block-scale shape
    it has (either orientation): the block scales the layer computes with, found by shape, never by name."""
    import torch
    out = []
    for mname, mod in dit.named_modules():
        tens = [(n, t) for n, t in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False))
                if isinstance(t, torch.Tensor) and t.dim() == 2]
        wshapes = {tuple(t.shape) for _, t in tens if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)}
        want = {(-(-a // 128), -(-b // 128)) for r, c in wshapes for a, b in ((r, c), (c, r))}
        for pname, t in tens:
            if t.dtype == torch.float32 and tuple(t.shape) in want:
                c = t.detach().to("cpu")
                out.append({"name": f"{mname}.{pname}" if mname else pname, "shape": list(c.shape),
                            "sha256": _sha(c), "sha256_t": _sha(c.t())})
    return out


def _fp8_start(dit, arm, label):
    import torch
    if _LIB and _LIB not in sys.path:
        sys.path.insert(0, _LIB)
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        import fp8_evidence as E
    finally:
        sys.dont_write_bytecode = prev
    ctx = {"label": label, "t0": time.time(), "layers_io": {}, "fired": set(), "hooks": [], "triton": [],
           "flashinfer": [], "fi_layer": None, "fi_out": None, "fi_snap": None, "fi_mark": {}, "fi_layers": {},
           "cur_layer": None, "fp8_events": [], "ev_mark": {}, "direct": {}, "layer_w": {}}
    stride = int(arm.get("fp8_layer_stride", 3))
    rows = int(arm.get("fp8_rows", 64))

    layers = E.find_fp8_layers(dit, E.q_shapes_for(tp=max(1, _rank_world()[1]), fused_qkv=True))


    ctx["layers_found"] = [{"module": L["module"], "weight": L["weight"], "keep": i % stride == 0,
                            "orig_dtype": _dtype_name(getattr(L["obj"], "orig_dtype", None))}
                           for i, L in enumerate(layers)]
    for L in layers:
        w = getattr(L["obj"], L["weight"], None)
        ctx["layer_w"][L["module"]] = int(w.data_ptr()) if isinstance(w, torch.Tensor) else None
    for i, L in enumerate(layers):
        keep = (i % stride == 0)

        def pre_hook(mod, args, name=L["module"]):
            ctx["fi_layer"], ctx["fi_out"], ctx["fi_snap"] = name, None, None
            ctx["fi_mark"][name] = len(ctx["flashinfer"])
            ctx["cur_layer"] = name
            ctx["ev_mark"][name] = len(ctx["fp8_events"])

        def hook(mod, args, out, name=L["module"], nks=tuple(L["nk_candidates"]), keep=keep,
                 orig=_dtype_name(getattr(L["obj"], "orig_dtype", None))):
            try:
                ctx["fired"].add(name)
                _fi_provenance(ctx, name, mod, out)
                _direct_provenance(ctx, name, mod, args, out)
                if not keep or name in ctx["layers_io"]:
                    return
                x, y = _first_tensor(args), _first_tensor(out)
                if x is None or y is None or (y.shape[-1], x.shape[-1]) not in nks:
                    return
                x2, y2 = x.detach().reshape(-1, x.shape[-1]), y.detach().reshape(-1, y.shape[-1])
                idx = torch.tensor(sample_positions(x2.shape[0], rows), dtype=torch.long, device=x2.device)
                ctx["layers_io"][name] = {"x": x2[idx].to("cpu", torch.float32), "y": y2[idx].to("cpu", torch.float32),
                                          "x_dtype": _dtype_name(x.dtype), "y_dtype": _dtype_name(y.dtype),
                                          "orig_dtype": orig}
            except Exception as exc:  # noqa: BLE001
                _emit({"event": "fp8_layer_hook_error", "module": name, "error": f"{type(exc).__name__}: {exc}"[:200]})
        ctx["hooks"].append(L["obj"].register_forward_pre_hook(pre_hook))
        ctx["hooks"].append(L["obj"].register_forward_hook(hook))
    ctx["recorder"] = E.op_recorder(on_fp8=lambda op, tens, res: _fp8_event(ctx, "dispatch", op, tens, res))
    ctx["recorder"].__enter__()


    _emit({"event": "fp8_started", "label": label, "layers_found": len(layers)})
    try:
        from torch.profiler import ProfilerActivity, profile
        torch.cuda.synchronize()
        ctx["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        ctx["prof"].__enter__()
    except Exception as exc:  # noqa: BLE001
        ctx["prof"] = None
        _emit({"event": "fp8_profiler_unavailable", "error": f"{type(exc).__name__}: {exc}"[:200]})
    _state["fp8"] = ctx


def _fp8_finish(dit, raised=False):
    import torch
    ctx, _state["fp8"] = _state["fp8"], None
    rec = {"label": ctx["label"], "layers_found": ctx["layers_found"], "layers_fired": sorted(ctx["fired"]),
           "forward_raised": bool(raised)}
    try:
        ctx["recorder"].__exit__(None, None, None)
        rec["dispatch_records"] = ctx["recorder"].records()
    except Exception as exc:  # noqa: BLE001
        rec["dispatch_error"] = f"{type(exc).__name__}: {exc}"[:300]
    for h in ctx["hooks"]:
        h.remove()
    if ctx.get("prof") is not None:
        try:
            torch.cuda.synchronize()
            ctx["prof"].__exit__(None, None, None)
            kernels = {}
            for e in ctx["prof"].events():
                if "CUDA" in str(getattr(e, "device_type", "")):
                    kernels[e.name] = kernels.get(e.name, 0) + 1
            rec["kernels"] = kernels
            rec["kernels_empty"] = not kernels
        except Exception as exc:  # noqa: BLE001
            rec["kernels_error"] = f"{type(exc).__name__}: {exc}"[:300]
    rec["triton_launches"] = ctx["triton"]
    rec["fp8_direct"] = {"layers": ctx["direct"], "events_total": len(ctx["fp8_events"]),
                         "events_outside_layers": sum(1 for e in ctx["fp8_events"] if e.get("layer") is None),
                         "sample_positions": _SAMPLE_N}
    rec["flashinfer_calls"] = ctx["flashinfer"]
    rec["flashinfer_layers"] = ctx["fi_layers"]
    fi = sys.modules.get("flashinfer")
    rec["flashinfer_version"] = getattr(fi, "__version__", None) if fi is not None else None
    rec["flashinfer_patched"] = bool(_state.get("fi_patched"))
    rec["flashinfer_trust"] = _state.get("fi_trust")
    rec["flashinfer_jit_loads"] = list(_state.get("fi_jit_loads") or [])
    try:
        rec["flashinfer_bindings"] = _fi_bindings()
    except Exception as exc:  # noqa: BLE001
        rec["flashinfer_bindings"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    try:
        rec["store"] = _store_census(dit)
    except Exception as exc:  # noqa: BLE001
        rec["store_error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        rec["store_scales"] = _scale_census(dit)
    except Exception as exc:  # noqa: BLE001
        rec["store_scales_error"] = f"{type(exc).__name__}: {exc}"[:300]
    rec["rank"], rec["world"] = _rank_world()
    base = os.path.join(_DIR, ctx["label"])
    try:
        torch.save(ctx["layers_io"], base + ".fp8_layers.pt.tmp")
        os.replace(base + ".fp8_layers.pt.tmp", base + ".fp8_layers.pt")
        with open(base + ".fp8.json.tmp", "w", encoding="utf-8") as fh:
            json.dump(rec, fh, default=str)
        os.replace(base + ".fp8.json.tmp", base + ".fp8.json")
        _emit({"event": "fp8_saved", "label": ctx["label"], "layers_found": len(ctx["layers_found"]),
               "layers_fired": len(ctx["fired"]), "layers_io": len(ctx["layers_io"]),
               "dispatch": len(rec.get("dispatch_records") or []), "triton": len(ctx["triton"]),
               "flashinfer": len(ctx["flashinfer"]),
               "kernels": len(rec.get("kernels") or {})})
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "fp8_save_error", "error": f"{type(exc).__name__}: {exc}"[:300]})


def _patch_triton(module):
    """Record, while an FP8 capture is active, every Triton launch with its tensor arguments' dtypes."""
    cls = getattr(module, "JITFunction", None)
    run = getattr(cls, "run", None) if cls is not None else None
    if run is None or getattr(run, "_infb_day0", False):
        return
    import functools

    @functools.wraps(run)
    def wrapped(self, *args, **kwargs):
        ctx = _state.get("fp8")
        tens = []
        if ctx is not None:
            try:
                torch = sys.modules.get("torch")
                tens = [a for a in list(args) + list(kwargs.values()) if torch is not None and isinstance(a, torch.Tensor)]
                dts = [str(a.dtype).replace("torch.", "") for a in tens]
                ctx["triton"].append({"fn": getattr(getattr(self, "fn", None), "__name__", "?"), "dtypes": dts})
            except Exception:  # noqa: BLE001
                tens = []
        res = run(self, *args, **kwargs)
        if ctx is not None and tens:
            try:
                name = "triton." + str(getattr(getattr(self, "fn", None), "__name__", "?"))
                if _E_fp8_gemm_like(name, tens):
                    _fp8_event(ctx, "triton", name, tens, None)
            except Exception:  # noqa: BLE001
                pass
        return res

    wrapped._infb_day0 = True
    cls.run = wrapped
    _emit({"event": "patched", "target": _TRITON_MODULE + ".JITFunction.run"})


def _tensor_meta(t):
    return {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "data_ptr": int(t.data_ptr())}


def _fi_provenance(ctx, name, mod, out):
    """After an FP8 layer's forward: did it call the pinned FlashInfer entry, and is its output that call's result?

    Read BY VALUE: the layer's output at the positions sampled from the call's result when the call
    returned -- 'equal', 'equal_plus_bias' (the layer's bias added in any of the dtypes _provenance_of tries), else
    'different'; 'no_output' when either side is missing. The buffer relation is recorded apart (`storage`, a
    diagnostic): 'same_storage' alone proved nothing, since the output buffer can be rewritten in place after the call.
    `cross` keeps both samples at the same positions for a TP row-parallel layer, whose output is the ranks' results
    summed -- the replay judges it across ranks (fp8_evidence.tp_flashinfer_row_sum)."""
    try:
        mark = ctx["fi_mark"].pop(name, None)
        calls = ctx["flashinfer"][mark:] if mark is not None else []
        fo, snap = ctx.get("fi_out"), ctx.get("fi_snap")
        ctx["fi_out"], ctx["fi_snap"], ctx["fi_layer"] = None, None, None
        if not calls:
            return
        torch = sys.modules.get("torch")
        y = _first_tensor(out)
        storage = None
        if y is not None and fo is not None:
            storage = "same_storage" if y.data_ptr() == fo.data_ptr() and y.numel() == fo.numel() else "own_storage"
        bias = getattr(mod, "bias", None)
        prov = _provenance_of(y, [snap] if snap else [], bias)
        cross = None
        if y is not None and snap and snap["numel"] == int(y.numel()):
            ix = torch.tensor(snap["idx"], dtype=torch.long)
            fy = y.detach().reshape(-1)
            bv = None
            if isinstance(bias, torch.Tensor) and int(bias.numel()) == int(snap["cols"]):
                bv = bias.detach().reshape(-1)[(ix % int(snap["cols"])).to(bias.device)].to("cpu", torch.float32).tolist()
            cross = {"channel": "flashinfer", "idx": snap["idx"], "c": snap["vals"], "c_dtype": snap["dtype"],
                     "y": fy[ix.to(fy.device)].to("cpu", torch.float32).tolist(), "y_dtype": _dtype_name(y.dtype),
                     "bias": bv}
        ctx["fi_layers"][name] = {"calls": len(calls), "provenance": prov, "storage": storage, "cross": cross}
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "fi_provenance_error", "module": name, "error": f"{type(exc).__name__}: {exc}"[:200]})


_SAMPLE_N = 64
_OUT_DTYPES = ("bfloat16", "float16", "float32")


def _E_fp8_gemm_like(name, tens):
    E = sys.modules.get("fp8_evidence")
    return bool(E is not None and E.fp8_gemm_like(name, tens))


def sample_positions(n: int, k: int = _SAMPLE_N) -> list:
    'k flat positions spread evenly over [0, n) -- round(i (n - 1) / (k - 1)) -- in EXACT integer arithmetic, so the'


    n, k = int(n), min(int(k), int(n))
    if k <= 1:
        return [0] * max(k, 0)
    return [(2 * i * (n - 1) + (k - 1)) // (2 * (k - 1)) for i in range(k)]


def _out_sample(t):
    """A fixed sample of a result tensor right after the GEMM wrote it: _SAMPLE_N flat positions spread over it. The
    positions are checked on the HOST before any device op: an index the device rejects is an asynchronous CUDA assert
    that poisons the candidate's context, which no try / except around this call can undo."""
    torch = sys.modules.get("torch")
    flat = t.detach().reshape(-1)
    n = int(flat.numel())
    pos = sample_positions(n)
    if pos and not (0 <= pos[0] and pos[-1] < n):
        raise ValueError(f"sample positions [{pos[0]}, {pos[-1]}] outside a tensor of {n} elements")
    idx = torch.tensor(pos, dtype=torch.long, device=flat.device)
    return {"ptr": int(t.data_ptr()), "numel": n, "shape": list(t.shape), "dtype": _dtype_name(t.dtype),
            "cols": int(t.shape[-1]) if t.dim() else 1, "idx": pos,
            "vals": flat[idx].to("cpu", torch.float32).tolist()}


def _fp8_event(ctx, channel, op, tens, res):
    """One FP8 GEMM on a directly observed channel: the layer whose forward made it, its two float8 operands, and a
    sample of every tensor that can hold its result -- what it returned, and any 16/32-bit [M, N] argument it wrote
    into (DeepGEMM's and Triton's output buffers)."""
    torch = sys.modules.get("torch")
    f8 = [t for t in tens if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and t.dim() == 2]
    if len(f8) < 2:
        return
    a, b = f8[0], f8[1]
    m, ns = int(a.shape[0]), {int(b.shape[0]), int(b.shape[1])}
    cands = [res] if isinstance(res, torch.Tensor) else [r for r in res if isinstance(r, torch.Tensor)] \
        if isinstance(res, (tuple, list)) else []
    cands += [t for t in tens if _dtype_name(t.dtype) in _OUT_DTYPES and t.dim() == 2 and int(t.shape[0]) == m
              and int(t.shape[1]) in ns]
    outs, seen = [], set()
    for t in cands:
        if id(t) in seen or t.numel() == 0 or _dtype_name(t.dtype) not in _OUT_DTYPES:
            continue
        seen.add(id(t))
        outs.append(_out_sample(t))
    ctx["fp8_events"].append({"channel": channel, "op": op, "layer": ctx.get("cur_layer"),
                              "fp8": [{"ptr": int(t.data_ptr()), "shape": list(t.shape)} for t in (a, b)],
                              "out": outs})


def _provenance_of(y, outs, bias):
    """'equal': y holds, at an event result's sampled positions, exactly the values that result held right after the
    call; 'equal_plus_bias': those values plus the layer's bias (added in the result's dtype, in y's, or in float32);
    'different' otherwise; 'no_output' when there is nothing to compare."""
    torch = sys.modules.get("torch")
    if y is None or not outs:
        return "no_output"
    flat = y.detach().reshape(-1)
    for o in outs:
        if o["numel"] != int(flat.numel()):
            continue
        idx = torch.tensor(o["idx"], dtype=torch.long)
        yv = flat[idx.to(flat.device)].to("cpu")
        dt_o = getattr(torch, o.get("dtype") or "", None)
        s = torch.tensor(o["vals"], dtype=torch.float32)
        s_o = s.to(dt_o) if isinstance(dt_o, torch.dtype) else s
        if torch.equal(yv, s_o.to(yv.dtype)):
            return "equal"
        if isinstance(bias, torch.Tensor) and int(bias.numel()) == int(o["cols"]):
            bv = bias.detach().reshape(-1)[(idx % int(o["cols"])).to(bias.device)].to("cpu")
            for cand in ((s_o + bv.to(s_o.dtype)).to(yv.dtype), s_o.to(yv.dtype) + bv.to(yv.dtype),
                         (s + bv.float()).to(yv.dtype)):
                if torch.equal(yv, cand):
                    return "equal_plus_bias"
    return "different"


def _direct_provenance(ctx, name, mod, args, out):
    """After a quantised layer's forward: did one FP8 GEMM it made on a directly observed channel use THIS layer's
    float8 weight, take THIS layer's token rows, and produce what the layer returned?"""
    try:
        mark = ctx["ev_mark"].pop(name, None)
        ctx["cur_layer"] = None
        evs = ctx["fp8_events"][mark:] if mark is not None else []
        if not evs:
            return
        x, y = _first_tensor(args), _first_tensor(out)
        rows = int(x.numel() // x.shape[-1]) if x is not None and x.dim() >= 1 and x.shape[-1] else None
        wptr = ctx["layer_w"].get(name)
        bias = getattr(mod, "bias", None)
        readings, consumed, cross = [], None, None
        for e in evs:
            on_w = wptr is not None and any(f["ptr"] == wptr for f in e["fp8"])
            on_rows = rows is not None and any(f["shape"][0] == rows for f in e["fp8"] if f["ptr"] != wptr)
            prov = _provenance_of(y, e["out"], bias)
            readings.append({"channel": e["channel"], "op": e["op"], "on_layer_weight": on_w,
                             "on_layer_rows": on_rows, "provenance": prov})
            if consumed is None and on_w and on_rows and prov in ("equal", "equal_plus_bias"):
                consumed = e["channel"]
            if cross is None and on_w and on_rows and e["out"] and y is not None and \
                    e["out"][0]["numel"] == int(y.numel()):


                torch = sys.modules.get("torch")
                o = e["out"][0]
                ix = torch.tensor(o["idx"], dtype=torch.long)
                fy = y.detach().reshape(-1)
                bv = None
                if isinstance(bias, torch.Tensor) and int(bias.numel()) == int(o["cols"]):
                    bv = bias.detach().reshape(-1)[(ix % int(o["cols"])).to(bias.device)].to("cpu", torch.float32).tolist()
                cross = {"channel": e["channel"], "idx": o["idx"], "c": o["vals"], "c_dtype": o["dtype"],
                         "y": fy[ix.to(fy.device)].to("cpu", torch.float32).tolist(), "y_dtype": _dtype_name(y.dtype),
                         "bias": bv}
        ctx["direct"][name] = {"events": len(evs), "weight_ptr": wptr, "rows": rows, "consumed": consumed is not None,
                               "consumed_channel": consumed, "readings": readings[:8], "cross": cross,


                               "io_seen": x is not None and y is not None}
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "direct_provenance_error", "module": name, "error": f"{type(exc).__name__}: {exc}"[:200]})


def _sha256_path(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tree_digest(root):
    'sha256 over "relpath NUL sha256 LF" of every regular file under root, sorted; __pycache__ and .pyc skipped'

    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for f in filenames:
            if f.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, f)
            if os.path.isfile(full) and not os.path.islink(full):
                lines.append(f"{os.path.relpath(full, root)}\0{_sha256_path(full)}\n")
    h = hashlib.sha256()
    for line in sorted(lines):
        h.update(line.encode("utf-8"))
    return h.hexdigest()


def _code_named(code, name):
    """The code object of `def name` inside a module's code (breadth first; the first match)."""
    todo = [code]
    while todo:
        c = todo.pop(0)
        for k in c.co_consts:
            if hasattr(k, "co_code"):
                if k.co_name == name:
                    return k
                todo.append(k)
    return None


def _fi_trust(module):
    """At bind time -- the moment gemm_base has executed, before any candidate code can call the entry: are the
    FlashInfer files beside the executed module the PINNED bytes (the verifier's manifest, INFB_SD3_DAY0_FI_MANIFEST:
    digests from the pinned wheel, never the package's own RECORD or __version__), and is the live entry's code
    object the code those verified bytes compile to (a planted .pyc, or a function swapped in before this hook ran,
    fails here)? A tap that cannot tell (no manifest, an unreadable file) is an INSTRUMENT fault, kept apart."""
    out = {"manifest": _FI_MANIFEST, "verified": False, "fault": None, "mismatch": [], "files_checked": 0,
           "trees_checked": 0, "entry_code_is_pinned": None, "runner_getter_code_is_pinned": None}
    try:
        if not _FI_MANIFEST:
            out["fault"] = "no pinned manifest (INFB_SD3_DAY0_FI_MANIFEST unset)"
            return out
        with open(_FI_MANIFEST, encoding="utf-8") as fh:
            man = json.load(fh)
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        root = os.path.dirname(os.path.dirname(origin))
        out["package_dir"] = root
        for rel, want in sorted((man.get("files") or {}).items()):
            got = _sha256_path(os.path.join(root, rel)) if os.path.isfile(os.path.join(root, rel)) else None
            out["files_checked"] += 1
            if got != want:
                out["mismatch"].append({"file": rel, "got": got, "want": want})
        for rel, want in sorted((man.get("trees") or {}).items()):
            got = _tree_digest(os.path.join(root, rel)) if os.path.isdir(os.path.join(root, rel)) else None
            out["trees_checked"] += 1
            if got != want:
                out["mismatch"].append({"tree": rel, "got": got, "want": want})
        entry = man["entry"]
        with open(os.path.join(root, entry["module_file"]), "rb") as fh:
            trusted = compile(fh.read(), origin, "exec", dont_inherit=True)
        fn = getattr(module, entry["function"], None)
        tcode = _code_named(trusted, entry["function"])
        out["entry_code_is_pinned"] = bool(tcode is not None and getattr(fn, "__code__", None) == tcode)
        getter = getattr(module, entry.get("runner_getter") or "", None)
        gcode = _code_named(trusted, entry.get("runner_getter") or "")
        g_inner = getattr(getter, "__wrapped__", getter)
        out["runner_getter_code_is_pinned"] = bool(gcode is not None and getattr(g_inner, "__code__", None) == gcode)
        out["jit_modules"] = list(man.get("jit_modules") or [])
        out["verified"] = (not out["mismatch"] and out["entry_code_is_pinned"] and out["runner_getter_code_is_pinned"])
    except Exception as exc:  # noqa: BLE001 - the instrument could not tell: a fault, never a verdict on the candidate
        out["fault"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def _patch_flashinfer(module):
    """Record, while an FP8 capture is active, every call of the pinned FlashInfer FP8 entry -- and whether each call
    ran the VERIFIED code: the function object bound at verification still carries the code object that matched the
    pinned bytes (identity, per call), the globals that code resolves are the ones present at verification, the
    runner getter is unchanged, and a profile hook saw the verified code object execute during the call. A Python
    hook in the candidate's process is not an unbypassable boundary."""
    fn = getattr(module, _FLASHINFER_ENTRY, None)
    if fn is None or getattr(fn, "_infb_day0", False):
        _emit({"event": "patch_skipped", "target": f"{_FLASHINFER_MODULE}.{_FLASHINFER_ENTRY}", "present": fn is not None})
        return
    import functools
    trust = _fi_trust(module)
    _state["fi_trust"] = trust
    live_code = getattr(fn, "__code__", None)
    getter_name = "get_fp8_blockscale_gemm_runner_sm90"
    getter_at_bind = module.__dict__.get(getter_name)
    names = [n for n in (getattr(live_code, "co_names", ()) or ()) if n in module.__dict__]
    globals_at_bind = {n: id(module.__dict__[n]) for n in names}

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        ctx = _state.get("fp8")
        if ctx is None:
            return fn(*args, **kwargs)
        seen = {"trusted_frame": False, "frames": 0}

        def prof(frame, event, arg):
            if event == "call":
                seen["frames"] += 1
                if frame.f_code is live_code:
                    seen["trusted_frame"] = True

        check = {"at_bind": bool(trust.get("verified")),
                 "code_unaltered": getattr(fn, "__code__", None) is live_code,
                 "globals_unaltered": all(id(module.__dict__.get(n)) == i for n, i in globals_at_bind.items()),
                 "runner_getter_unaltered": module.__dict__.get(getter_name) is getter_at_bind}
        prev = sys.getprofile()
        sys.setprofile(prof)
        try:
            out = fn(*args, **kwargs)
        finally:
            sys.setprofile(prev)
        check["trusted_frame"] = seen["trusted_frame"]
        try:
            torch = sys.modules.get("torch")
            rec = {"entry": f"{_FLASHINFER_MODULE}.{_FLASHINFER_ENTRY}", "layer": ctx.get("fi_layer"),
                   "args": [dict(_tensor_meta(a), i=i) for i, a in enumerate(args) if isinstance(a, torch.Tensor)],
                   "kwargs": {k: _tensor_meta(v) for k, v in kwargs.items() if isinstance(v, torch.Tensor)},
                   "scalars": {k: repr(v)[:40] for k, v in kwargs.items() if not isinstance(v, torch.Tensor)},
                   "trusted": check}
            o = _first_tensor(out)
            rec["out"] = _tensor_meta(o) if o is not None else None
            ctx["flashinfer"].append(rec)
            if ctx.get("fi_layer") is not None:
                ctx["fi_out"] = o


                ctx["fi_snap"] = _out_sample(o) if o is not None and o.is_floating_point() and o.numel() else None
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "fi_record_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
        return out

    wrapped._infb_day0 = True
    setattr(module, _FLASHINFER_ENTRY, wrapped)


    for alias in ("flashinfer.gemm", "sglang.srt.layers.quantization.fp8_utils"):
        m = sys.modules.get(alias)
        if m is not None and getattr(m, _FLASHINFER_ENTRY, None) is fn:
            setattr(m, _FLASHINFER_ENTRY, wrapped)
    _state["fi_patched"] = True
    _state["fi_wrapper"] = wrapped
    _emit({"event": "patched", "target": f"{_FLASHINFER_MODULE}.{_FLASHINFER_ENTRY}",
           "trusted_at_bind": bool(trust.get("verified")), "trust_fault": trust.get("fault"),
           "trust_mismatch": len(trust.get("mismatch") or [])})


def _patch_flashinfer_jit(module):
    """Record every JIT module FlashInfer loads (JitSpec.load): its name, the .so's path and sha256, and whether it
    lies in the fresh workspace the verifier created for this run (FLASHINFER_WORKSPACE_BASE; INFB_SD3_DAY0_FI_T0 is
    that directory's creation time) -- a module built in this run from the verified sources, not a cached binary."""
    cls = getattr(module, "JitSpec", None)
    load = getattr(cls, "load", None) if cls is not None else None
    if load is None or getattr(load, "_infb_day0", False):
        return
    import functools

    @functools.wraps(load)
    def wrapped(self, *args, **kwargs):
        res = load(self, *args, **kwargs)
        try:
            path = str(args[0] if args else kwargs.get("so_path", ""))
            base = os.environ.get("FLASHINFER_WORKSPACE_BASE")
            t0 = float(os.environ.get("INFB_SD3_DAY0_FI_T0") or 0)
            real = os.path.realpath(path)
            fresh = bool(base and real.startswith(os.path.realpath(base) + os.sep) and t0
                         and os.path.getmtime(real) >= t0)
            _state.setdefault("fi_jit_loads", []).append(
                {"name": getattr(self, "name", None), "path": path, "sha256": _sha256_path(real),
                 "fresh_in_verifier_workspace": fresh})
        except Exception as exc:  # noqa: BLE001
            _state.setdefault("fi_jit_loads", []).append({"error": f"{type(exc).__name__}: {exc}"[:200]})
        return res

    wrapped._infb_day0 = True
    cls.load = wrapped
    _emit({"event": "patched", "target": _FLASHINFER_JIT_MODULE + ".JitSpec.load"})


def _fi_bindings():
    """Which of the known names of the entry are the recorder at this moment (diagnostic: a legal wrapper may rebind
    an alias around the recorder; only the per-layer record decides credit)."""
    w = _state.get("fi_wrapper")
    out = {}
    for mod, attr in ((_FLASHINFER_MODULE, _FLASHINFER_ENTRY), ("flashinfer.gemm", _FLASHINFER_ENTRY),
                      ("sglang.srt.layers.quantization.fp8_utils", _FLASHINFER_ENTRY)):
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, attr):
            out[f"{mod}.{attr}"] = getattr(m, attr) is w
    return out


def _hook_transformer(pipeline):
    mods = getattr(pipeline, "modules", None)
    dit = mods.get("transformer") if isinstance(mods, dict) else None
    if dit is None or not callable(getattr(dit, "register_forward_hook", None)):
        _emit({"event": "no_transformer", "pipeline": type(pipeline).__name__})
        return

    def pre(module, args, kwargs):
        try:
            arm = _arm()
            if not arm or (not arm.get("all_ranks") and _not_the_recording_rank()):
                return None
            label = str(arm.get("label"))
            if arm.get("all_ranks"):
                rank, world = _rank_world()
                if world > 1:
                    label = f"{label}.r{rank}"
            ts = _timestep_of(args, kwargs)
            step, p = _position(label, ts)
            module._infb_day0_pos = (label, step, p, ts)
            fs = arm.get("fp8_step")
            if fs is not None and step == int(fs) and p == 0 and label not in _state["fp8_done"] and _state["fp8"] is None:
                _state["fp8_done"].add(label)
                _fp8_start(module, arm, label)
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "pre_hook_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return None

    def post(module, args, kwargs, output):
        try:
            pos = getattr(module, "_infb_day0_pos", None)
            module._infb_day0_pos = None
            if pos is None:
                return None
            label, step, p, ts = pos


            raised = output is None
            if raised:
                _emit({"event": "forward_raised", "label": label, "step": step, "pass": p})
            if _state["fp8"] is not None and _state["fp8"]["label"] == label:
                _fp8_finish(module, raised=raised)
            if raised:
                return None
            arm = _arm() or {}
            out = _first_tensor(output)
            shapes = {k: list(v.shape) for k, v in kwargs.items() if hasattr(v, "shape")}
            with open(os.path.join(_DIR, f"{label}.calls.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"step": step, "pass": p, "timestep": ts, "shapes": shapes,
                                     "out_shape": list(out.shape) if out is not None else None}) + "\n")
            if step in set(int(s) for s in arm.get("save_steps", [])):
                payload = {k: _saveable(v) for k, v in kwargs.items()}
                payload["args"] = [_saveable(v) for v in args]
                payload["output"] = _cpu(out) if out is not None else None
                torch = sys.modules.get("torch")
                path = os.path.join(_DIR, f"{label}.s{step}.p{p}.pt")
                torch.save(payload, path + ".tmp")
                os.replace(path + ".tmp", path)
                _emit({"event": "saved", "label": label, "step": step, "pass": p, "timestep": ts})
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "post_hook_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return None

    try:
        dit.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            dit.register_forward_hook(post, with_kwargs=True, always_call=True)
        except TypeError:
            dit.register_forward_hook(post, with_kwargs=True)
        _emit({"event": "hooked", "pipeline": type(pipeline).__name__,
               "transformer": [type(dit).__module__, type(dit).__qualname__],
               "n_modules": sum(1 for _ in dit.modules())})
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "hook_error", "error": f"{type(exc).__name__}: {exc}"[:300]})


def _patch_pipeline(module):
    import functools
    cls = getattr(module, "ComposedPipelineBase", None)
    original = cls.__dict__.get("__init__") if isinstance(cls, type) else None
    if original is None or getattr(original, "_infb_day0", False):
        _emit({"event": "patch_skipped", "target": _PIPELINE_MODULE})
        return

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _hook_transformer(self)

    __init__._infb_day0 = True
    cls.__init__ = __init__
    _emit({"event": "patched", "target": _PIPELINE_MODULE + ".ComposedPipelineBase.__init__",
           "file": getattr(module, "__file__", None)})


_PATCHERS = {_PIPELINE_MODULE: _patch_pipeline, _TRITON_MODULE: _patch_triton, _FLASHINFER_MODULE: _patch_flashinfer,
             _FLASHINFER_JIT_MODULE: _patch_flashinfer_jit}


class _PostImportPatcher:
    """A meta-path finder for the target names only: it asks the rest of sys.meta_path for the spec (so each module
    resolves as it would without this file) and wraps that spec's exec_module to patch the module once it ran."""

    def __init__(self):
        self._inflight = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _PATCHERS or fullname in self._inflight:
            return None
        self._inflight.add(fullname)
        try:
            spec = None
            for finder in list(sys.meta_path):
                if finder is self or getattr(finder, "find_spec", None) is None:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    break
        finally:
            self._inflight.discard(fullname)
        loader = getattr(spec, "loader", None)
        real_exec = getattr(loader, "exec_module", None)
        if spec is None or real_exec is None:
            return None

        def exec_module(module):
            real_exec(module)
            if getattr(module, "__name__", None) == fullname:
                try:
                    _PATCHERS[fullname](module)
                except Exception as exc:  # noqa: BLE001
                    _emit({"event": "patch_error", "target": fullname, "error": f"{type(exc).__name__}: {exc}"[:300]})

        loader.exec_module = exec_module
        return spec


def _chain_shadowed_sitecustomize():
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
                spec = importlib.util.spec_from_file_location("_infb_day0_shadowed_sitecustomize", candidate)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return candidate
        except Exception:  # noqa: BLE001
            return None
    return None


if _DIR:
    _emit({"event": "installed", "ppid": os.getppid(), "python": sys.executable, "targets": sorted(_PATCHERS)})
    sys.meta_path.insert(0, _PostImportPatcher())
    for _name in [n for n in _PATCHERS if n in sys.modules]:
        _PATCHERS[_name](sys.modules[_name])
_CHAINED = _chain_shadowed_sitecustomize()
if _DIR and _CHAINED:
    _emit({"event": "chained", "file": _CHAINED})
