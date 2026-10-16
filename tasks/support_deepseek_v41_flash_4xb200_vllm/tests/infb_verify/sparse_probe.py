#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The trusted sparse-indexer observer inside the diagnostic child's workers (the sparse-indexer contract)."""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

AUDIT_DIR = Path(os.environ.get("INFB_SPARSE_AUDIT_DIR", "/logs/verifier/sparse-indexer/binding"))
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
CONTROL = AUDIT_DIR / "active"
SLOTS = 16            # rows per observed call: up to 8 decode rows and 8 prefill rows
TOPK = 512
NCAND = 2048
HEADS = 32
QBYTES = 64
RESTRICTIVE = 16384   # a row with more visible compressed positions than the pool can hold
KEEP_PER_REQUEST = 2  # restrictive rows kept with their keys per layer, path and request (and as many others)
MAX_PAGES = 32768 // 16 + 1  # page-table entries kept per row: enough for 32,768 positions at 16 per page


def _cfg() -> dict:
    try:
        return json.loads((TESTS_DIR / "config.json").read_text())["infb"]["sparse_indexer"]
    except (OSError, ValueError, KeyError):
        return {}


def _rank() -> int:
    try:
        import torch.distributed as dist  # noqa: PLC0415

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        pass
    for k in ("RANK", "LOCAL_RANK"):
        if os.environ.get(k, "").isdigit():
            return int(os.environ[k])
    return -1


def _source_identity(obj) -> dict:
    """Where an observed callable comes from, for provenance (never a name allowlist)."""
    try:
        fn = getattr(obj, "forward", obj)
        path = inspect.getsourcefile(fn) or ""
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None
        return {"qualname": getattr(type(obj), "__qualname__", None) if not inspect.isfunction(obj)
                else obj.__qualname__, "module": getattr(fn, "__module__", None), "file": path, "sha256": digest}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}




def view_span_bytes(x) -> int:
    """The bytes from a tensor view's first element to the end of its last: a strided view (the attention output sliced
    to the local heads before the projection) spans more than its elements."""
    if x.numel() == 0:
        return 0
    return (sum((s - 1) * st for s, st in zip(x.shape, x.stride())) + 1) * x.element_size()


def select_rows(torch, keys, vis, limit: int):
    """The last two rows of each request (rows of one request are consecutive, visibility rising), at most `limit`,
    in row order, as fixed-size device tensors: (rows, valid). Capture-safe: no host synchronization."""
    T = keys.shape[0]
    dev = keys.device
    idx = torch.arange(T, device=dev)
    nxt = torch.cat([keys[1:], torch.full((1,), -2, dtype=keys.dtype, device=dev)])
    last = keys != nxt
    prev = torch.cat([last[1:] & (keys[:-1] == keys[1:]), torch.zeros(1, dtype=torch.bool, device=dev)])
    mask = (last | prev) & (keys >= 0) & (vis > 0)
    order = torch.sort(torch.where(mask, idx, idx + T)).values
    if T < limit:
        order = torch.cat([order, torch.full((limit - T,), 2 * T, dtype=order.dtype, device=dev)])
    rows = order[:limit]
    valid = rows < T
    return torch.where(valid, rows, torch.zeros_like(rows)), valid


class Buffers:
    """One observed layer's fixed-size snapshot buffers (device), filled by copies and read after the step."""

    def __init__(self, torch, device, pages: int, combined: int, max_tokens: int):
        z = lambda *s, dt=torch.int32: torch.zeros(s, dtype=dt, device=device)  # noqa: E731
        self.fields = {
            "rows": z(SLOTS, dt=torch.int64), "valid": z(SLOTS, dt=torch.bool), "key": z(SLOTS, dt=torch.int64),
            "vis": z(SLOTS, dt=torch.int64), "decode": z(SLOTS, dt=torch.bool),
            "q": z(SLOTS, HEADS, QBYTES, dt=torch.uint8), "q_scale": z(SLOTS, HEADS),
            "weights": z(SLOTS, HEADS, dt=torch.float32), "pool": z(SLOTS, NCAND), "selection": z(SLOTS, TOPK),
            "table": z(SLOTS, pages), "proj": z(SLOTS, HEADS, dt=torch.float32),
            "fused_out": z(SLOTS, HEADS, dt=torch.float32),
            "att_dec_idx": z(SLOTS, TOPK), "att_dec_len": z(SLOTS), "att_dec_table": z(SLOTS, pages),
            "att_dec_valid": z(SLOTS, dt=torch.bool),
            "att_pre_idx": z(SLOTS, combined), "att_pre_len": z(SLOTS), "att_pre_pos": z(SLOTS, dt=torch.int64),
            "att_pre_region": z(SLOTS, dt=torch.int64), "att_pre_wbase": z(SLOTS, dt=torch.int64),
            "att_pre_layout": z(SLOTS, 3, dt=torch.int64),
            "att_pre_valid": z(SLOTS, dt=torch.bool),
            "counter": z(4, dt=torch.int64),  # indexer calls, fused calls, decode attention, prefill attention
            # host facts a captured graph cannot report through Python: each hook writes them with fill_ (a constant the
            # graph replays): the index-key cache and the call's rows; the attention layer's output buffer, the
            # FlashMLA outputs and the projection's input; the decode path's compressed page size
            "facts": z(8, dt=torch.int64),
        }
        self.proj_full = z(max_tokens, HEADS, dt=torch.float32)
        self.fused_full = z(max_tokens, HEADS, dt=torch.float32)
        self.bytes = sum(v.numel() * v.element_size() for v in self.fields.values()) + 2 * self.proj_full.numel() * 4


class Observer:
    def __init__(self, torch, runner, model, bindings: dict, rank: int):
        self.torch, self.runner, self.model, self.rank = torch, runner, model, rank
        self.b = bindings
        self.dir = AUDIT_DIR / f"rank{rank}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.kept: dict = {}
        self.keys_read: dict = {}
        self.local = threading.local()
        self.events: list = []
        self.capture_events: list = []
        self.buffers: dict[int, Buffers] = {}
        self.written: set = set()
        self.attention_calls: dict = {}
        self.fused_meta: dict = {}
        self.o_proj_inputs: set = set()
        self.last_counters: dict = {}
        cfg = _cfg()
        self.max_tokens = int(getattr(getattr(runner, "scheduler_config", None), "max_num_batched_tokens", 0) or 0) \
            or int(getattr(getattr(getattr(runner, "vllm_config", None), "scheduler_config", None),
                           "max_num_batched_tokens", 0) or 0) or 65536
        max_len = int(cfg.get("geometry", {}).get("max_visible", 32768))
        pages = MAX_PAGES
        combined = -(-(TOPK + int(bindings.get("window", 128)) + 127) // 128) * 128 + 128
        device = torch.device("cuda", torch.cuda.current_device())
        for layer in [bindings["source"]["layer"], *[c["layer"] for c in bindings["consumers"]]]:
            self.buffers[layer] = Buffers(torch, device, pages, combined, self.max_tokens)
        self.shared_proj = torch.zeros((self.max_tokens, HEADS), dtype=torch.float32, device=device)
        self.shared_fused = torch.zeros_like(self.shared_proj)
        self.shared_fused_meta: dict | None = None

    def _ctx(self):
        from vllm.forward_context import get_forward_context  # noqa: PLC0415

        try:
            return get_forward_context()
        except Exception:  # noqa: BLE001
            return None

    def _capturing(self) -> bool:
        return bool(self.torch.cuda.is_current_stream_capturing())

    def _event(self, what: str, layer: int, **extra):
        rec = {"what": what, "layer": layer, "capturing": self._capturing(), **extra}
        (self.capture_events if rec["capturing"] else self.events).append(rec)

    def wrap_indexer(self, layer: int, role: str, op) -> None:
        orig = op.forward
        obs = self

        def forward(hidden_states, q_quant, k, weights, *a, **kw):
            out = orig(hidden_states, q_quant, k, weights, *a, **kw)
            try:
                views = obs._default_views(op, q_quant, weights, int(hidden_states.shape[0]))
                if views is not None:
                    obs._fill(layer, role, views)
                    obs._event("indexer", layer, role=role, tokens=int(hidden_states.shape[0]),
                               k_cache_ptr=int(op.k_cache.kv_cache.data_ptr()), weights_ptr=int(weights.data_ptr()))
            except Exception as exc:  # noqa: BLE001  (the observer's own fault; the call itself is untouched)
                obs._fault(f"layer {layer} {role}: {exc}")
            return out

        op.forward = forward

    def _rows(self, op, T: int):
        """[T] request keys (the request's first index-cache page: unique among live requests), [T] visible compressed
        lengths, [T] decode flags, and the segments that give a row's page-table row -- (start, end, table, row ->
        table-row index) -- from the indexer metadata (the engine's convention)."""
        torch = self.torch
        ctx = self._ctx()
        meta = ctx.attn_metadata[op.k_cache.prefix]
        dev = op.topk_indices_buffer.device
        key = torch.full((T,), -1, dtype=torch.int64, device=dev)
        vis = torch.zeros(T, dtype=torch.int64, device=dev)
        dec = torch.zeros(T, dtype=torch.bool, device=dev)
        segments = []
        nd = int(getattr(meta, "num_decode_tokens", 0) or 0)
        if nd > 0:
            sd = getattr(meta, "sparse_decode", None)
            if sd is not None and getattr(sd, "block_table", None) is not None:
                n = sd.row_ke.shape[0]
                bt = sd.block_table[:n]
                key[:n] = bt[:, 0].to(torch.int64)
                vis[:n] = sd.row_ke[:n].to(torch.int64)
                dec[:n] = True
                segments.append((0, n, bt, torch.arange(n, device=dev)))
            else:
                d = meta.decode
                nr = d.block_table.shape[0]
                req = torch.repeat_interleave(torch.arange(nr, device=dev), d.decode_lens[:nr], output_size=nd)
                first = torch.cumsum(d.decode_lens[:nr], 0) - d.decode_lens[:nr]
                off = torch.arange(nd, device=dev) - first[req]
                key[:nd] = d.block_table[req, 0].to(torch.int64)
                vis[:nd] = (d.seq_lens[req] - d.decode_lens[req] + 1 + off).to(torch.int64)
                dec[:nd] = True
                segments.append((0, nd, d.block_table, req))
        if int(getattr(meta, "num_prefills", 0) or 0) > 0 and getattr(meta, "prefill", None) is not None:
            for chunk in meta.prefill.chunks:
                s, e = int(chunk.token_start), int(chunk.token_end)
                if e <= s:
                    continue
                seq = chunk.token_to_seq[: e - s].to(torch.int64)
                key[s:e] = chunk.block_table[seq, 0].to(torch.int64)
                vis[s:e] = (chunk.cu_seqlen_ke[: e - s] - chunk.cu_seqlen_ks[: e - s]).to(torch.int64)
                segments.append((s, e, chunk.block_table, seq))
        return key, vis, dec, segments, nd

    def _tables(self, rows, segments, P: int):
        """[SLOTS, P] page-table rows of the chosen rows, gathered from the segment each row lies in."""
        torch = self.torch
        out = torch.zeros((rows.shape[0], P), dtype=torch.int32, device=rows.device)
        for s, e, table, index in segments:
            inside = (rows >= s) & (rows < e)
            local = torch.where(inside, rows - s, torch.zeros_like(rows)).clamp(max=max(e - s - 1, 0))
            got = torch.zeros_like(out)
            w = min(P, table.shape[1])
            got[:, :w] = table[index[local], :w].to(torch.int32)
            out = torch.where(inside[:, None], got, out)
        return out

    def _default_views(self, op, q_quant, weights, T: int) -> dict | None:
        """The views of one indexer call under the default mapping (the audit schema's names), after the call: the rows
        from the indexer metadata, the call's inputs, the shared key cache and its page tables, the pool in the op's
        candidate buffer (the source wrote it; a consumer read it and leaves it as it was), the selection it wrote."""
        torch = self.torch
        ctx = self._ctx()
        if T == 0 or ctx is None or not isinstance(getattr(ctx, "attn_metadata", None), dict):
            return None
        key, vis, dec, segments, nd = self._rows(op, T)
        q, q_scale = q_quant if isinstance(q_quant, (tuple, list)) else (q_quant, None)
        return {"rows": torch.stack([key, vis, dec.to(torch.int64)], 1), "q": q, "q_scale": q_scale,
                "weights": weights, "k_cache": op.k_cache.kv_cache, "k_table": segments,
                "candidate_pool": op.candidate_blocks[:T], "selection": op.topk_indices_buffer[:T, :TOPK]}

    def _fill(self, layer: int, role: str, views: dict) -> None:
        """Copy one producer call's views into the layer's buffers: the rows chosen on the device (decode and prefill
        separately), then their inputs, page tables, pool and selection. Capture-safe."""
        torch = self.torch
        rows3 = views["rows"]
        T = int(rows3.shape[0])
        if T == 0:
            return
        key, vis, dec = rows3[:, 0].to(torch.int64), rows3[:, 1].to(torch.int64), rows3[:, 2].to(torch.bool)
        half = SLOTS // 2
        r1, v1 = select_rows(torch, torch.where(dec, key, torch.full_like(key, -1)), vis, half)
        r2, v2 = select_rows(torch, torch.where(dec, torch.full_like(key, -1), key), vis, half)
        rows, valid = torch.cat([r1, r2]), torch.cat([v1, v2])
        buf = self.buffers[layer]
        f = buf.fields
        f["rows"].copy_(rows)
        f["valid"].copy_(valid)
        f["key"].copy_(key[rows])
        f["vis"].copy_(vis[rows])
        f["decode"].copy_(dec[rows] & valid)
        P = f["table"].shape[1]
        kt = views.get("k_table")
        if isinstance(kt, list):
            f["table"].copy_(self._tables(rows, kt, P))
        elif kt is not None:
            tab = kt[0][kt[1][rows]] if isinstance(kt, tuple) else kt[rows]
            got = torch.zeros_like(f["table"])
            got[:, : min(P, tab.shape[1])] = tab[:, :P].to(torch.int32)
            f["table"].copy_(got)
        q = views["q"][rows].reshape(SLOTS, HEADS, -1)[..., :QBYTES]
        f["q"].copy_(q.view(torch.uint8) if q.dtype != torch.uint8 else q)
        if views.get("q_scale") is not None:
            qs = views["q_scale"][rows]
            if qs.dtype != torch.int32:
                qs = qs.contiguous().view(torch.int32)
            f["q_scale"].copy_(qs.reshape(SLOTS, -1)[:, :HEADS])
        f["weights"].copy_(views["weights"][rows].reshape(SLOTS, HEADS).to(torch.float32))
        if views.get("projection") is not None:
            f["proj"].copy_(views["projection"][rows].reshape(SLOTS, HEADS).to(torch.float32))
            self.fused_meta[layer] = {"softmax_scale": float(views["weight_scale"]), "head_scale": 1.0,
                                      "declared": True}
        elif self.b.get("mapping") == "declared":
            f["proj"].copy_(self.shared_proj[rows])
            f["fused_out"].copy_(self.shared_fused[rows])
            if self.shared_fused_meta is not None:
                self.fused_meta[layer] = dict(self.shared_fused_meta)
        else:
            f["proj"].copy_(buf.proj_full[rows])
            f["fused_out"].copy_(buf.fused_full[rows])
        f["pool"].copy_(views["candidate_pool"][rows].to(torch.int32))
        if views.get("selection") is not None:
            f["selection"].copy_(views["selection"][rows, :TOPK].to(torch.int32))
        if role == "consumer":
            for name in ("att_dec_valid", "att_pre_valid"):
                f[name].zero_()
            f["att_pre_idx"].fill_(-1)
            f["facts"][2:].zero_()
        f["counter"][0].add_(1)
        f["facts"][0].fill_(int(views["k_cache"].data_ptr()))
        f["facts"][1].fill_(T)
        if role == "source":
            self.k_cache = views["k_cache"]
        self.local.sel = getattr(self.local, "sel", {})
        self.local.sel[layer] = (rows, valid, dec[rows] & valid)
        self.local.views = getattr(self.local, "views", {})
        self.local.views[layer] = views
        self.written.add(layer)

    def wrap_fused(self, layers: dict) -> list[str]:
        """Wrap the supplied fused query call where the observed layers' indexers look it up; the wrapper copies its
        projection input and its weights output into the layer's full buffers (inside the captured work)."""
        patched = []
        mods = {type(ind).__module__ for ind in layers.values()}
        obs = self
        for modname in mods:
            mod = sys.modules.get(modname)
            fn = getattr(mod, "fused_indexer_q_rope_quant", None)
            if fn is None or getattr(fn, "_infb_sparse", False):
                continue

            def wrapped(*a, _fn=fn, **kw):
                out = _fn(*a, **kw)
                try:
                    layer = getattr(obs.local, "indexer_layer", None)
                    if layer in obs.buffers:
                        names = list(inspect.signature(_fn).parameters)
                        bound = dict(zip(names, a)) | kw
                        proj = bound.get("index_weights")
                        w = out[1] if isinstance(out, tuple) else None
                        buf = obs.buffers[layer]
                        if proj is not None and w is not None and proj.dim() == 2:
                            n = min(proj.shape[0], buf.proj_full.shape[0])
                            buf.proj_full[:n].copy_(proj[:n].to(obs.torch.float32))
                            buf.fused_full[:n].copy_(w[:n].reshape(n, -1)[:, :HEADS].to(obs.torch.float32))
                            buf.fields["counter"][1].add_(1)
                            obs.fused_meta[layer] = {
                                "softmax_scale": float(bound.get("index_weights_softmax_scale", float("nan"))),
                                "head_scale": float(bound.get("index_weights_head_scale", float("nan"))),
                                "weights_dtype": str(w.dtype), "weights_ptr": int(w.data_ptr())}
                            obs._event("fused_query", layer, tokens=int(n), weights_ptr=int(w.data_ptr()))
                except Exception as exc:  # noqa: BLE001
                    obs._fault(f"fused query: {exc}")
                return out

            wrapped._infb_sparse = True
            setattr(mod, "fused_indexer_q_rope_quant", wrapped)
            patched.append(modname)
        for layer, ind in layers.items():
            orig = ind.forward

            def fwd(*a, _orig=orig, _layer=layer, **kw):
                obs.local.indexer_layer = _layer
                try:
                    return _orig(*a, **kw)
                finally:
                    obs.local.indexer_layer = None

            ind.forward = fwd
        return patched

    def wrap_attention(self, layer: int, attn) -> None:
        obs = self
        orig = attn.forward_mqa

        def forward_mqa(q, kv, positions, output, *a, **kw):
            obs.local.att = {"layer": layer, "q_ptr": int(q.data_ptr()), "q_row_bytes": int(q.stride(0) * q.element_size()),
                             "attn": attn}
            obs._event("forward_mqa", layer, output_ptr=int(output.data_ptr()), output_shape=list(output.shape),
                       output_bytes=int(output.numel() * output.element_size()))
            with contextlib.suppress(Exception):
                facts = obs.buffers[layer].fields["facts"]
                facts[2].fill_(int(output.data_ptr()))
                facts[3].fill_(int(output.numel() * output.element_size()))
            try:
                return orig(q, kv, positions, output, *a, **kw)
            finally:
                obs.local.att = None

        attn.forward_mqa = forward_mqa
        if hasattr(attn, "_o_proj"):
            o_orig = attn._o_proj

            def o_proj(o, *a, **kw):
                obs.o_proj_inputs.add((layer, int(o.data_ptr())))
                with contextlib.suppress(Exception):
                    obs.buffers[layer].fields["facts"][6].fill_(int(o.data_ptr()))
                obs._event("o_proj", layer, input_ptr=int(o.data_ptr()), input_shape=list(o.shape))
                return o_orig(o, *a, **kw)

            attn._o_proj = o_proj

    def _replace_function(self, fn, wrapped) -> int:
        """Replace every module-level reference to `fn` by `wrapped`; how many were replaced."""
        n = 0
        for mod in list(sys.modules.values()):
            d = getattr(mod, "__dict__", None)
            if not isinstance(d, dict):
                continue
            for k, v in list(d.items()):
                if v is fn:
                    d[k] = wrapped
                    n += 1
        return n

    def _wrap(self, target, before=None, after=None, finally_=None) -> str:
        """Observe the calls of `target`: a module's forward, a bound method on its object, or a function where
        modules look it up. `before(args, kwargs)`, `after(args, kwargs, result)` and `finally_()` run around each
        call; an exception in them is the observer's own fault and never reaches the engine."""
        obs = self

        def run(orig, a, kw):
            if before:
                obs._guard(before, a, kw)
            try:
                out = orig(*a, **kw)
            finally:
                if finally_:
                    obs._guard(finally_)
            if after:
                obs._guard(after, a, kw, out)
            return out

        if isinstance(target, self.torch.nn.Module):
            orig = target.forward
            target.forward = lambda *a, **kw: run(orig, a, kw)
            return f"module {type(target).__qualname__}.forward"
        if inspect.ismethod(target):
            orig = target
            setattr(target.__self__, target.__func__.__name__, lambda *a, **kw: run(orig, a, kw))
            return f"method {type(target.__self__).__qualname__}.{target.__func__.__name__}"
        if callable(target):
            n = self._replace_function(target, lambda *a, **kw: run(target, a, kw))
            return f"function {getattr(target, '__qualname__', target)} ({n} reference(s))"
        raise TypeError(f"not observable: {type(target).__name__}")

    def _guard(self, fn, *a):
        try:
            fn(*a)
        except Exception as exc:  # noqa: BLE001
            self._fault(f"{getattr(fn, '__name__', 'hook')}: {exc}")

    def wrap_declared(self, entry: dict) -> list[str]:
        """Observe one declared binding: its producer call through its views, and (a consumer) its attention call
        site as the context of the native FlashMLA calls, its output use for the projection's input."""
        b, layer, role = entry["binding"], entry["layer"], entry["role"]

        def produced(a, kw, out):
            views = b.views(a, kw, out)
            if not views:
                return
            self._fill(layer, role, views)
            self._event("indexer", layer, role=role, declared=True)

        done = [self._wrap(b.call, after=produced)]
        if role == "consumer":
            def enter(a, kw):
                self.local.att = {"layer": layer, "declared": b}

            def leave():
                self.local.att = None

            done.append(self._wrap(b.attention, before=enter, finally_=leave))
            if b.output_use is not None:
                def used(a, kw):
                    x = next((v for v in list(a) + list(kw.values()) if isinstance(v, self.torch.Tensor)), None)
                    if x is not None:
                        facts = self.buffers[layer].fields["facts"]
                        facts[6].fill_(int(x.data_ptr()))
                        facts[2].fill_(int(x.data_ptr()))
                        facts[3].fill_(view_span_bytes(x))

                done.append(self._wrap(b.output_use, before=used))
        return done

    def wrap_fused_everywhere(self) -> str:
        """The supplied fused query call wherever the model looks it up (a declared description that does not give the
        weights' projection itself): its input projection and BF16 output, copied inside the captured work, and its
        scales."""
        from vllm.models.deepseek_v4.common.ops import fused_indexer_q as fq  # noqa: PLC0415

        fn = getattr(fq, "fused_indexer_q_rope_quant", None)
        if fn is None or getattr(fn, "_infb_sparse", False):
            return "no supplied fused query call"
        obs = self
        names = list(inspect.signature(fn).parameters)

        def wrapped(*a, **kw):
            out = fn(*a, **kw)
            try:
                bound = dict(zip(names, a)) | kw
                proj, w = bound.get("index_weights"), out[1] if isinstance(out, tuple) else None
                if proj is not None and w is not None and proj.dim() == 2:
                    n = min(proj.shape[0], obs.shared_proj.shape[0])
                    obs.shared_proj[:n].copy_(proj[:n].to(obs.torch.float32))
                    obs.shared_fused[:n].copy_(w[:n].reshape(n, -1)[:, :HEADS].to(obs.torch.float32))
                    obs.shared_fused_meta = {
                        "softmax_scale": float(bound.get("index_weights_softmax_scale", float("nan"))),
                        "head_scale": float(bound.get("index_weights_head_scale", float("nan"))),
                        "weights_dtype": str(w.dtype)}
            except Exception as exc:  # noqa: BLE001
                obs._fault(f"fused query: {exc}")
            return out

        wrapped._infb_sparse = True
        return f"fused_indexer_q_rope_quant: {self._replace_function(fn, wrapped)} reference(s)"

    def wrap_leaves_everywhere(self) -> list[str]:
        """The native FlashMLA calls, wherever the model looks them up (a declared description names only the call
        site they run in)."""
        import vllm.v1.attention.ops.flashmla as fm  # noqa: PLC0415

        out = []
        for name, handler in (("flash_mla_with_kvcache", self._on_decode), ("flash_mla_sparse_fwd", self._on_prefill)):
            fn = getattr(fm, name, None)
            if fn is None or getattr(fn, "_infb_sparse", False):
                continue
            obs = self

            def wrapped(*a, _fn=fn, _h=handler, **kw):
                res = _fn(*a, **kw)
                try:
                    _h(_fn, a, kw, res)
                except Exception as exc:  # noqa: BLE001
                    obs._fault(f"{_fn.__name__}: {exc}")
                return res

            wrapped._infb_sparse = True
            out.append(f"{name}: {self._replace_function(fn, wrapped)} reference(s)")
        return out

    def _declared_rows(self, coords: dict, rows, want):
        """The leaf row each chosen producer row appears in, from a declared description's coordinates."""
        torch = self.torch
        crow = coords["rows"].to(torch.int64).to(rows.device)
        match = (crow[None, :] == rows[:, None]) & want[:, None]
        return match.any(1), match.to(torch.int64).argmax(1)

    def wrap_leaves(self, modules: set) -> list[str]:
        """Replace the FlashMLA leaf functions and the prefill combine helper where the consumer layers' modules
        look them up (the calls themselves are unchanged)."""
        patched = []
        obs = self
        for modname in modules:
            mod = sys.modules.get(modname)
            if mod is None:
                continue
            for name, handler in (("flash_mla_with_kvcache", self._on_decode), ("flash_mla_sparse_fwd", self._on_prefill),
                                  ("combine_topk_swa_indices", self._on_combine)):
                fn = getattr(mod, name, None)
                if fn is None or getattr(fn, "_infb_sparse", False):
                    continue

                def wrapped(*a, _fn=fn, _h=handler, **kw):
                    out = _fn(*a, **kw)
                    try:
                        _h(_fn, a, kw, out)
                    except Exception as exc:  # noqa: BLE001
                        obs._fault(f"{_fn.__name__}: {exc}")
                    return out

                wrapped._infb_sparse = True
                setattr(mod, name, wrapped)
                patched.append(f"{modname}.{name}")
        return patched

    def _bound(self, fn, a, kw) -> dict:
        names = list(inspect.signature(fn).parameters)
        return dict(zip(names, a)) | kw

    def _current(self):
        att = getattr(self.local, "att", None)
        if not att or att["layer"] not in self.buffers:
            return None, None
        sel = getattr(self.local, "sel", {}).get(att["layer"])
        return att, sel

    def _on_combine(self, fn, a, kw, out):
        att, _ = self._current()
        if att is None:
            return
        b = self._bound(fn, a, kw)
        self.local.combine = {"M": int(b["M"]), "N": int(b["N"]), "window": int(b["window_size"]),
                              "cr": int(b["compress_ratio"]), "topk": int(b["topk"]),
                              "qsl": b["query_start_loc"], "seq_lens": b["seq_lens"], "gather_lens": b["gather_lens"]}

    def _on_decode(self, fn, a, kw, out):
        att, sel = self._current()
        if att is None or sel is None:
            return
        torch = self.torch
        b = self._bound(fn, a, kw)
        rows, valid, dec = sel
        f = self.buffers[att["layer"]].fields
        ext = b.get("extra_indices_in_kvcache")
        if ext is None:
            return
        n = ext.shape[0]
        declared = att.get("declared")
        if declared is not None:
            coords = declared.to_attention(self.local.views[att["layer"]], fn.__name__, b)
            ok, r = self._declared_rows(coords, rows, dec)
        else:
            ok = dec & (rows < n)
            r = torch.where(ok, rows, torch.zeros_like(rows))
        okc = ok[:, None]
        f["att_dec_idx"].copy_(torch.where(okc, ext.reshape(n, -1)[r, :TOPK].to(torch.int32), f["att_dec_idx"]))
        lens = b.get("extra_topk_length")
        if lens is not None:
            f["att_dec_len"].copy_(torch.where(ok, lens.reshape(-1)[r].to(torch.int32), f["att_dec_len"]))
        if declared is not None:
            bt, block_size = coords["block_table"][r], int(coords["block_size"])
        else:
            ctx = self._ctx()
            a_layer = att["attn"]
            fl = ctx.attn_metadata.get(a_layer.compressed_cache_prefix)
            swa = ctx.attn_metadata.get(a_layer.swa_cache_layer.prefix)
            req = swa.token_to_req_indices[r].to(torch.int64)
            bt, block_size = fl.block_table[req], int(fl.block_size // a_layer.compress_ratio)
        P = f["att_dec_table"].shape[1]
        new_t = torch.zeros_like(f["att_dec_table"])
        new_t[:, : min(P, bt.shape[1])] = bt[:, :P].to(torch.int32)
        f["att_dec_table"].copy_(torch.where(okc, new_t, f["att_dec_table"]))
        f["att_dec_valid"].copy_(f["att_dec_valid"] | ok)
        f["counter"][2].add_(1)
        f["facts"][7].fill_(block_size)
        if b.get("out") is not None:
            f["facts"][4].fill_(int(b["out"].data_ptr()))
        self._event("flash_mla_with_kvcache", att["layer"], out_ptr=int(b["out"].data_ptr()) if b.get("out") is not None
                    else None, compressed_block_size=block_size, rows=int(n))

    def _on_prefill(self, fn, a, kw, out):
        att, sel = self._current()
        if att is not None and sel is not None and att.get("declared") is not None:
            return self._on_prefill_declared(fn, a, kw, att, sel)
        comb = getattr(self.local, "combine", None)
        if att is None or sel is None or comb is None:
            return
        torch = self.torch
        b = self._bound(fn, a, kw)
        rows, valid, dec = sel
        q = b["q"]
        n = q.shape[0]
        # this call's rows are a slice of the layer's query rows: its first row's offset, in rows
        first = (int(q.data_ptr()) - att["q_ptr"]) // max(att["q_row_bytes"], 1)
        local = rows - first
        ok = valid & ~dec & (local >= 0) & (local < n)
        lr = torch.where(ok, local, torch.zeros_like(local))
        f = self.buffers[att["layer"]].fields
        idx = b["indices"].reshape(n, -1)
        width = min(idx.shape[1], f["att_pre_idx"].shape[1])
        # a prefill step may call once per chunk: each call fills only the slots whose rows it holds
        okc = ok[:, None]
        new_idx = torch.full_like(f["att_pre_idx"], -1)
        new_idx[:, :width] = idx[lr, :width].to(torch.int32)
        f["att_pre_idx"].copy_(torch.where(okc, new_idx, f["att_pre_idx"]))
        f["att_pre_len"].copy_(torch.where(ok, b["topk_length"].reshape(-1)[lr].to(torch.int32), f["att_pre_len"]))
        # the native combined layout of this chunk: request j's compressed region at j*M, its window at j*M + N
        qsl = comb["qsl"].to(torch.int64)
        base = qsl[0]
        starts = qsl[:-1] - base
        ends = qsl[1:] - base
        j = torch.searchsorted(ends, lr, right=True).clamp(max=max(starts.shape[0] - 1, 0))
        seq = comb["seq_lens"].to(torch.int64)
        gl = comb["gather_lens"].to(torch.int64)
        qlen = ends[j] - starts[j]
        # the native layout: request j's compressed region at j*M, its window's absolute position p at j*M + N + p -
        # (seq_len - gather_len)
        region = comb["M"] * j
        f["att_pre_pos"].copy_(torch.where(ok, seq[j] - qlen + (lr - starts[j]), f["att_pre_pos"]))
        f["att_pre_region"].copy_(torch.where(ok, region, f["att_pre_region"]))
        f["att_pre_wbase"].copy_(torch.where(ok, region + comb["N"] - (seq[j] - gl[j]), f["att_pre_wbase"]))
        layout = self._layout(ok.device, comb["window"], comb["cr"], comb["topk"])
        f["att_pre_layout"].copy_(torch.where(okc, layout, f["att_pre_layout"]))
        f["att_pre_valid"].copy_(f["att_pre_valid"] | ok)
        f["counter"][3].add_(1)
        f["facts"][5].fill_(int(b["out"].data_ptr()))
        self.attention_calls.setdefault(att["layer"], []).append({"M": comb["M"], "N": comb["N"], "window": comb["window"],
                                                                   "cr": comb["cr"], "topk": comb["topk"], "first_row": first})
        self._event("flash_mla_sparse_fwd", att["layer"], out_ptr=int(b["out"].data_ptr()), rows=int(n),
                    first_row=int(first), M=comb["M"], N=comb["N"])

    def _layout(self, device, window: int, cr: int, topk: int):
        """[SLOTS, 3] (window, compress ratio, top-k width), by fills (capture-safe: no host-to-device copy)."""
        out = self.torch.empty((SLOTS, 3), dtype=self.torch.int64, device=device)
        out[:, 0].fill_(int(window))
        out[:, 1].fill_(int(cr))
        out[:, 2].fill_(int(topk))
        return out

    def _on_prefill_declared(self, fn, a, kw, att, sel):
        torch = self.torch
        b = self._bound(fn, a, kw)
        rows, valid, dec = sel
        coords = att["declared"].to_attention(self.local.views[att["layer"]], fn.__name__, b)
        ok, r = self._declared_rows(coords, rows, valid & ~dec)
        f = self.buffers[att["layer"]].fields
        idx = b["indices"].reshape(b["indices"].shape[0], -1)
        width = min(idx.shape[1], f["att_pre_idx"].shape[1])
        okc = ok[:, None]
        new_idx = torch.full_like(f["att_pre_idx"], -1)
        new_idx[:, :width] = idx[r, :width].to(torch.int32)
        f["att_pre_idx"].copy_(torch.where(okc, new_idx, f["att_pre_idx"]))
        f["att_pre_len"].copy_(torch.where(ok, b["topk_length"].reshape(-1)[r].to(torch.int32), f["att_pre_len"]))
        for name, key in (("att_pre_pos", "positions"), ("att_pre_region", "region"), ("att_pre_wbase", "window_base")):
            f[name].copy_(torch.where(ok, coords[key].to(torch.int64)[r], f[name]))
        layout = self._layout(ok.device, int(coords["window"]), int(coords["compress_ratio"]), TOPK)
        f["att_pre_layout"].copy_(torch.where(okc, layout, f["att_pre_layout"]))
        f["att_pre_valid"].copy_(f["att_pre_valid"] | ok)
        f["counter"][3].add_(1)
        f["facts"][5].fill_(int(b["out"].data_ptr()))
        self._event("flash_mla_sparse_fwd", att["layer"], out_ptr=int(b["out"].data_ptr()), declared=True)

    def _fault(self, what: str) -> None:
        self.events.append({"what": "observer_fault", "detail": what[:500], "trace": traceback.format_exc()[-1500:]})

    def after_step(self) -> None:
        """Read the step's snapshots while the diagnostic control file exists; keep at most two rows per layer, path
        and request with their keys."""
        self.step += 1
        events, self.events = self.events, []
        if not CONTROL.exists():
            self.written.clear()
            return
        torch = self.torch
        torch.cuda.synchronize()
        for layer, buf in self.buffers.items():
            c = tuple(int(x) for x in buf.fields["counter"].cpu().tolist())
            if self.last_counters.get(layer) != c:
                self.written.add(layer)
            self.last_counters[layer] = c
        if not self.written:
            return
        rec: dict = {"step": self.step, "rank": self.rank, "time": time.time(), "events": events, "layers": {},
                     "keys": {}}
        ctx_mode = None
        with contextlib.suppress(Exception):
            ctx_mode = str(getattr(self._ctx(), "cudagraph_runtime_mode", None))
        rec["cudagraph_runtime_mode_after"] = ctx_mode
        need_keys: dict = {}
        for layer in sorted(self.written):
            f = {k: v.detach().cpu() for k, v in self.buffers[layer].fields.items()}
            keep = []
            for s in range(SLOTS):
                if not bool(f["valid"][s]):
                    continue
                path = "decode" if bool(f["decode"][s]) else "prefill"
                restrictive = int(f["vis"][s]) > RESTRICTIVE
                k = (layer, path, int(f["key"][s]), restrictive)
                if self.kept.get(k, 0) >= KEEP_PER_REQUEST:
                    continue
                self.kept[k] = self.kept.get(k, 0) + 1
                keep.append(s)
                kk = (path, int(f["key"][s]))
                if int(f["vis"][s]) > need_keys.get(kk, (0, None))[0]:
                    need_keys[kk] = (int(f["vis"][s]), f["table"][s])
            if keep:
                rec["layers"][layer] = {"slots": keep, **{k: v[keep] for k, v in f.items() if k not in ("counter", "facts")},
                                        "counter": f["counter"], "facts": f["facts"], "fused": self.fused_meta.get(layer),
                                        "attention_calls": self.attention_calls.get(layer, [])[-4:]}
        self.written.clear()
        self.attention_calls.clear()
        if not rec["layers"]:
            return
        cache = getattr(self, "k_cache", None)
        if cache is None:
            cache = self.b["source"]["op"].k_cache.kv_cache
        page_kv = int(cache.shape[1])
        for (path, key), (vis, table) in need_keys.items():
            if self.keys_read.get((path, key), 0) >= 2 * KEEP_PER_REQUEST:
                continue
            self.keys_read[(path, key)] = self.keys_read.get((path, key), 0) + 1
            n = -(-vis // page_kv)
            pages = table[:n].to(torch.int64).to(cache.device)
            page_bytes = int(cache.stride(0))
            raw = torch.as_strided(cache, (cache.shape[0], page_kv * 68), (page_bytes, 1))
            rec["keys"][f"{path}:{key}"] = {"visible": vis, "pages": table[:n].clone(),
                                            "raw": raw[pages].cpu(), "page_kv": page_kv,
                                            "cache_shape": list(cache.shape), "page_bytes": page_bytes}
        rec["o_proj_inputs"] = sorted(self.o_proj_inputs)
        rec["capture_events"] = self.capture_events[-400:]
        torch.save(rec, self.dir / f"step{self.step:06d}.pt")


def default_bindings(model) -> dict:
    """The default mapping from the prepared engine's DeepSeek-V4-family conventions: in `...layers.<n>`, the attention
    module -- the one with `forward_mqa` and an `indexer` whose `indexer_op` is set on the index layers -- whatever its
    attribute name (`attn` in the prepared engine's V4 decoder layer); the configured source and consumers."""
    cfg = _cfg()
    layers = cfg.get("audit_layers", {"source": 20, "consumers": [24, 28, 32, 36]})
    found: dict = {}
    for name, mod in model.named_modules():
        parts = name.split(".")
        if "layers" not in parts or not hasattr(mod, "forward_mqa") or getattr(mod, "indexer", None) is None:
            continue
        try:
            lid = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue
        found.setdefault(lid, mod)
    out: dict = {"mapping": "default", "consumers": []}
    for lid, role in [(layers["source"], "source"), *[(c, "consumer") for c in layers["consumers"]]]:
        attn = found.get(lid)
        ind = getattr(attn, "indexer", None)
        op = getattr(ind, "indexer_op", None)
        if attn is None or op is None:
            raise LookupError(f"layer {lid}: no attention module with indexer.indexer_op in the model")
        entry = {"layer": lid, "attn": attn, "indexer": ind, "op": op, "identity": _source_identity(op),
                 "attention_identity": _source_identity(attn)}
        if role == "source":
            out["source"] = entry
        else:
            out["consumers"].append(entry)
    out["window"] = int(getattr(out["consumers"][0]["attn"], "window_size", 128) or 128)
    return out


BINDING_FIELDS = ("layer_id", "role", "call", "views", "attention", "to_attention", "output_use")


def declared_bindings(bindings) -> dict:
    """A candidate's description (the public schema, duck-typed: a copy of the dataclass serves) as the observer's
    bindings, when it describes the configured source and every consumer; otherwise a ValueError naming what is
    missing."""
    cfg = _cfg()
    layers = cfg.get("audit_layers", {"source": 20, "consumers": [24, 28, 32, 36]})
    by = {}
    for x in bindings:
        missing = [k for k in BINDING_FIELDS if not hasattr(x, k)]
        if missing:
            raise ValueError(f"a binding without {missing}")
        if not callable(x.views) or (x.role == "consumer" and (x.attention is None or not callable(x.to_attention))):
            raise ValueError(f"layer {x.layer_id}: views, attention or to_attention unusable")
        by[(int(x.layer_id), x.role)] = x
    want = [(int(layers["source"]), "source"), *[(int(c), "consumer") for c in layers["consumers"]]]
    absent = [w for w in want if w not in by]
    if absent:
        raise ValueError(f"no binding for {absent}")
    entries = [{"layer": lid, "role": role, "binding": by[(lid, role)], "identity": _source_identity(by[(lid, role)].call)}
               for lid, role in want]
    return {"mapping": "declared", "source": entries[0], "consumers": entries[1:],
            "describe": [{"layer": e["layer"], "role": e["role"], "call": e["identity"]} for e in entries]}


def install(runner) -> None:
    """Called by the startup hook after the runner's load_model: resolve the bindings, wrap the observed calls and the
    runner's step. Every failure is recorded (`install.json`) and leaves the runner as it was."""
    rank = _rank()
    rec: dict = {"rank": rank, "pid": os.getpid(), "time": time.time(), "tests_dir": str(TESTS_DIR)}
    (AUDIT_DIR / f"rank{rank}").mkdir(parents=True, exist_ok=True)
    try:
        import torch  # noqa: PLC0415

        model = runner.get_model() if hasattr(runner, "get_model") else runner.model
        model = model.unwrap() if hasattr(model, "unwrap") and not hasattr(model, "named_modules") else model
        declared = None
        try:
            import importlib  # noqa: PLC0415

            mod = importlib.import_module("vllm.model_executor.layers.sparse_indexer_audit")
            declared = getattr(mod, "get_bindings", None)
        except ImportError:
            pass
        rec["declared_bindings"] = declared is not None
        b = None
        if declared is not None:
            try:
                b = declared_bindings(declared(model))
                rec["declared"] = b["describe"]
            except Exception as exc:  # noqa: BLE001  (a description the observer cannot use is recorded, not failed)
                rec["declared_problem"] = f"{type(exc).__name__}: {exc}"[:1000]
                b = None
        if b is None:
            b = default_bindings(model)
        obs = Observer(torch, runner, model, b, rank)
        if b["mapping"] == "declared":
            rec["wrapped"] = [w for e in [b["source"], *b["consumers"]] for w in obs.wrap_declared(e)]
            rec["fused_patched"] = [obs.wrap_fused_everywhere()]
        else:
            obs.wrap_indexer(b["source"]["layer"], "source", b["source"]["op"])
            for c in b["consumers"]:
                obs.wrap_indexer(c["layer"], "consumer", c["op"])
                obs.wrap_attention(c["layer"], c["attn"])
            rec["fused_patched"] = obs.wrap_fused({e["layer"]: e["indexer"] for e in [b["source"], *b["consumers"]]})
            rec["leaves_patched"] = obs.wrap_leaves({type(c["attn"]).__module__ for c in b["consumers"]})
        rec["leaves_everywhere"] = obs.wrap_leaves_everywhere()
        rec["observer_bytes"] = sum(buf.bytes for buf in obs.buffers.values())
        rec["compress_ratios"] = {e["layer"]: int(getattr(e.get("attn"), "compress_ratio", -1))
                                  for e in [b["source"], *b["consumers"]]}
        rec["bindings"] = {"mapping": b["mapping"], "source": {"layer": b["source"]["layer"],
                                                               "identity": b["source"].get("identity")},
                           "consumers": [{"layer": c["layer"], "identity": c.get("identity"),
                                          "attention": c.get("attention_identity")} for c in b["consumers"]]}
        orig = runner.execute_model

        def execute_model(*a, **kw):
            out = orig(*a, **kw)
            try:
                obs.after_step()
            except Exception as exc:  # noqa: BLE001
                (obs.dir / f"after-step-fault-{obs.step}.txt").write_text(traceback.format_exc())
            return out

        runner.execute_model = execute_model
        runner._infb_sparse_observer = obs
        rec["status"] = "installed"
    except Exception as exc:  # noqa: BLE001
        rec.update(status="not_installed", error=f"{type(exc).__name__}: {exc}"[:2000],
                   traceback=traceback.format_exc()[-4000:])
    (AUDIT_DIR / f"rank{rank}" / f"install-{os.getpid()}.json").write_text(json.dumps(rec, indent=1, default=str))
