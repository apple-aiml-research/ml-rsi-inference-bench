#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""R-RUNTIME and the shared bookkeeping: rank identity, forward-path records and the graph witness, in every engine process."""
from __future__ import annotations

import os
import threading

from . import after_import, control, emit, error, set_rank

_local = threading.local()
_MAX_IDS = 16384
_prof: dict = {"armed_seq": None, "left": 0, "prof": None, "forwards": []}


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _rank_facts(runner) -> dict:
    import torch
    facts: dict = {"gpu_id": getattr(runner, "gpu_id", None), "is_draft_worker": getattr(runner, "is_draft_worker", None),
                   "cuda_current_device": _safe(torch.cuda.current_device),
                   "device_uuid": _safe(lambda: str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        from sglang.srt.distributed import parallel_state as ps
        for name in ("get_tensor_model_parallel_rank", "get_tensor_model_parallel_world_size",
                     "get_moe_expert_parallel_rank", "get_moe_expert_parallel_world_size",
                     "get_pipeline_model_parallel_rank", "get_pipeline_model_parallel_world_size"):
            if hasattr(ps, name):
                facts[name.replace("get_", "")] = _safe(getattr(ps, name))
    except Exception as exc:  # noqa: BLE001
        facts["parallel_state_error"] = repr(exc)[:200]
    try:
        from sglang.srt.layers import dp_attention as dpa
        for name in ("get_attention_dp_rank", "get_attention_dp_size", "get_attention_tp_rank", "get_attention_tp_size"):
            if hasattr(dpa, name):
                facts[name.replace("get_", "")] = _safe(getattr(dpa, name))
    except Exception as exc:  # noqa: BLE001
        facts["dp_attention_error"] = repr(exc)[:200]
    ps_obj = getattr(runner, "ps", None)
    if ps_obj is not None:
        fields = {}
        for name in ("tp_rank", "tp_size", "pp_rank", "pp_size", "dp_rank", "dp_size", "attn_tp_rank", "attn_tp_size",
                     "attn_dp_rank", "attn_dp_size", "attn_cp_rank", "attn_cp_size", "moe_ep_rank", "moe_ep_size",
                     "moe_dp_rank", "moe_dp_size", "gpu_id"):
            v = getattr(ps_obj, name, "<absent>")
            if v == "<absent>" or isinstance(v, (int, bool, str, type(None))):
                fields[name] = v
        facts["ps"] = fields
    return facts


def _native_facts(runner) -> dict:
    out: dict = {}
    out["attn_backend"] = type(getattr(runner, "attn_backend", None)).__name__
    pool = getattr(runner, "token_to_kv_pool", None)
    out["kv_pool"] = type(pool).__name__ if pool is not None else None
    out["kv_cache_dtype"] = str(getattr(runner, "kv_cache_dtype", None))
    out["kv_pool_dtype"] = str(getattr(pool, "dtype", None)) if pool is not None else None
    out["kv_pool_store_dtype"] = str(getattr(pool, "store_dtype", None)) if pool is not None else None
    graph = getattr(runner, "decode_cuda_graph_runner", None)
    out["decode_graph_runner"] = type(graph).__name__ if graph is not None else None
    for attr in ("capture_bs", "cuda_graph_bs", "graph_bs_list"):
        if graph is not None and hasattr(graph, attr):
            out["decode_graph_bs"] = _safe(lambda: list(getattr(graph, attr)))
            break
    out["prefill_graph_runner"] = type(getattr(runner, "prefill_cuda_graph_runner", None)).__name__
    sa = getattr(runner, "server_args", None)
    for key in ("moe_runner_backend", "moe_a2a_backend", "dsa_prefill_backend", "dsa_decode_backend",
                "attention_backend", "kv_cache_dtype", "enable_dp_attention", "speculative_algorithm",
                "speculative_num_steps", "speculative_adaptive", "disable_cuda_graph", "cuda_graph_max_bs"):
        out[f"server_args.{key}"] = repr(_safe(lambda: getattr(sa, key)))[:80]
    return out


def _count_layer_calls(model) -> None:
    """Count Python-level invocations of every decoder layer of the loaded model (a replayed graph runs none)."""
    layers = None
    for name, mod in model.named_modules():
        if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) > 8:
            layers, where = mod, name
            break
    if layers is None:
        emit("layer_counter", installed=False)
        return

    def pre_hook(module, args):
        _local.layer_calls = getattr(_local, "layer_calls", 0) + 1

    for layer in layers:
        layer.register_forward_pre_hook(pre_hook)
    emit("layer_counter", installed=True, path=where, n_layers=len(layers))


def _inside(t, bufs: list) -> bool | None:
    """Whether tensor `t`'s storage lies inside one of `bufs` (None when there is nothing to compare)."""
    try:
        if t is None or not bufs:
            return None
        for buf in bufs:
            start = buf.data_ptr()
            if start <= t.data_ptr() < start + buf.numel() * buf.element_size():
                return True
        return False
    except Exception:  # noqa: BLE001
        return None


def _graph_output_buffers(runner) -> list:
    """The static logits buffers a replayed decode graph writes its output into: the decode graph runner's
    `next_token_logits_buffer`, a view of the runner's shared graph output (`graph_shared_output`)."""
    out = []
    graph = getattr(runner, "decode_cuda_graph_runner", None)
    buf = getattr(getattr(graph, "buffers", None), "next_token_logits_buffer", None)
    if buf is not None:
        out.append(buf)
    shared = getattr(runner, "graph_shared_output", None)
    out += [b for b in (getattr(shared, "_logits_buffers", None) or {}).values() if b is not None]
    return out


def _maybe_start_profile(c: dict) -> None:
    n = int((c.get("args") or {}).get("profile_forwards") or 0)
    if n <= 0 or _prof["armed_seq"] == c.get("seq") or _prof["prof"] is not None:
        return
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
        _prof.update(armed_seq=c.get("seq"), left=n, forwards=[])
        torch.cuda.synchronize()
        _prof["prof"] = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=False)
        _prof["prof"].__enter__()
        emit("profile_started", forwards=n)
    except Exception as exc:  # noqa: BLE001
        _prof["prof"] = None
        error("profile start", exc)


def _stop_profile() -> None:
    import torch
    prof, _prof["prof"] = _prof["prof"], None
    try:
        torch.cuda.synchronize()
        prof.__exit__(None, None, None)
        from . import _dir
        path = os.path.join(_dir("profiles", str(os.getpid())), f"trace-{_prof['armed_seq']}.json")
        prof.export_chrome_trace(path)
        emit("profile_stopped", trace=path, forwards=_prof["forwards"], summary=_summarize(path))
    except Exception as exc:  # noqa: BLE001
        error("profile stop", exc)


def _summarize(path: str) -> dict:
    """Graph launches, the GPU kernels correlated with them, and kernels launched outside graphs, from the trace."""
    import json as _json
    events = _json.load(open(path)).get("traceEvents", [])
    runtime = [e for e in events if e.get("cat") == "cuda_runtime"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    graph_corr = {e.get("args", {}).get("correlation") for e in runtime if "GraphLaunch" in str(e.get("name"))}
    launch_corr = {e.get("args", {}).get("correlation") for e in runtime
                   if str(e.get("name")).startswith(("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchCooperativeKernel"))}
    in_graph = [k for k in kernels if k.get("args", {}).get("correlation") in graph_corr]
    eager = [k for k in kernels if k.get("args", {}).get("correlation") in launch_corr]
    streams = sorted({k.get("args", {}).get("stream") for k in in_graph})
    names: dict = {}
    for k in in_graph:
        names[k.get("name", "")[:80]] = names.get(k.get("name", "")[:80], 0) + 1
    return {"graph_launches": len(graph_corr - {None}), "kernels_in_graphs": len(in_graph),
            "kernels_outside_graphs": len(eager), "graph_kernel_streams": streams,
            "top_graph_kernels": sorted(names.items(), key=lambda kv: -kv[1])[:24]}


def _row_positions(fb, n_rows: int):
    """The position of the input token each next-token logits row follows: every token's own for decode and verify,
    each request's last token for an extend."""
    import torch
    pos = fb.positions
    mode = fb.forward_mode
    if mode.is_extend() and not mode.is_target_verify():
        lens = getattr(fb, "extend_seq_lens", None)
        ends = torch.cumsum(lens.to(torch.long), 0) - 1 if lens is not None else torch.tensor([pos.shape[0] - 1])
        return pos[ends.to(pos.device)][:n_rows]
    return pos[:n_rows]


def _force(logits, rows_pos, force_ids: list[int]) -> None:
    """Teacher forcing of the probe (the consumption control only): each row whose next position lies in the fixed
    sequence gets that position's token as its maximum, so the served continuation is the fixed one whatever the
    logits; the recorded logits are copied before this."""
    import torch
    key = (id(force_ids), len(force_ids), str(logits.device))
    if getattr(_local, "force_key", None) != key:
        _local.force_key, _local.force_ids = key, torch.tensor(force_ids, dtype=torch.long, device=logits.device)
    fixed = _local.force_ids
    nxt = rows_pos.to(torch.long) + 1
    ok = nxt < fixed.shape[0]
    if not bool(ok.any()):
        return
    rows = torch.nonzero(ok).flatten()
    tok = fixed[nxt[rows]]
    logits[rows, tok] = logits[rows].max(dim=-1).values + 1.0e4


def _patch_logits_processor(mod) -> None:
    """The consumption control's eager observable: the full logits of the prefill positions a request asks input
    log-probabilities for (`logprob_start_len`), as the logits processor computes them (never during graph capture)."""
    cls = mod.LogitsProcessor
    orig = cls._get_logits

    def _get_logits(self, hidden_states, lm_head, logits_metadata, *args, **kwargs):
        out = orig(self, hidden_states, lm_head, logits_metadata, *args, **kwargs)
        try:
            import torch
            mode = getattr(logits_metadata, "forward_mode", None)
            if (control().get("args") or {}).get("capture_logits") and getattr(self, "_glm53_target", False) \
                    and not torch.cuda.is_current_stream_capturing() and mode is not None and mode.is_extend() \
                    and not mode.is_target_verify() and getattr(logits_metadata, "extend_return_logprob", False) \
                    and out is not None and out.dim() == 2 and out.shape[0] > 1:
                from . import store
                c = control()
                _local.pruned_calls = getattr(_local, "pruned_calls", 0) + 1
                path = store(f"pruned-c{c.get('seq')}-{_local.pruned_calls:05d}", {"logits": out.float()})
                emit("pruned_logits", file=path, rows=int(out.shape[0]), call=_local.pruned_calls)
        except Exception as exc:  # noqa: BLE001
            error("pruned logits capture", exc)
        return out

    cls._get_logits = _get_logits


def _patch_model_runner(mod) -> None:
    cls = mod.ModelRunner
    orig_init, orig_forward, orig_load = cls.__init__, cls.forward, cls.load_model

    def load_model(self, *args, **kwargs):
        r = orig_load(self, *args, **kwargs)
        try:
            if not getattr(self, "is_draft_worker", False):
                _count_layer_calls(self.model)
                for m in self.model.modules():
                    if type(m).__name__ == "LogitsProcessor":
                        m._glm53_target = True
                from . import attention as _att  # noqa: PLC0415 - does nothing unless the feature is on
                _att.install_on_model(self)
        except Exception as exc:  # noqa: BLE001
            error("model_runner_load_model", exc)
        return r

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            facts = _rank_facts(self)
            if not facts.get("is_draft_worker"):
                set_rank(**{k: v for k, v in facts.items() if not isinstance(v, dict)})
            emit("rank_facts", **facts)
        except Exception as exc:  # noqa: BLE001
            error("model_runner_init", exc)

    def forward(self, forward_batch, *args, **kwargs):
        c = control()
        rec = bool((c.get("args") or {}).get("record_forwards"))
        target = not getattr(self, "is_draft_worker", False)
        if target:
            _maybe_start_profile(c)
        if getattr(self, "_glm53_native_seq", None) != (c.get("seq") if rec else "startup"):
            try:
                self._glm53_native_seq = c.get("seq") if rec else "startup"
                emit("native_facts", draft=bool(getattr(self, "is_draft_worker", False)), **_native_facts(self))
            except Exception as exc:  # noqa: BLE001
                error("native_facts", exc)
        _local.layer_calls = 0
        out = orig_forward(self, forward_batch, *args, **kwargs)
        if target and _prof["prof"] is not None:
            _prof["forwards"].append(getattr(getattr(forward_batch, "forward_mode", None), "name", None))
            _prof["left"] -= 1
            if _prof["left"] <= 0:
                _stop_profile()
        args = c.get("args") or {}
        if target and (args.get("capture_logits") or args.get("force_ids")):
            try:
                logits = getattr(getattr(out, "logits_output", None), "next_token_logits", None)
                if logits is not None and logits.shape[0] > 0:
                    rows_pos = _row_positions(forward_batch, logits.shape[0])
                    mode = getattr(getattr(forward_batch, "forward_mode", None), "name", None)
                    if args.get("capture_logits"):
                        from . import store
                        _local.logit_calls = getattr(_local, "logit_calls", 0) + 1
                        path = store(f"logits-c{c.get('seq')}-{_local.logit_calls:05d}",
                                     {"logits": logits.float(), "positions": rows_pos.to("cpu")})
                        ids = getattr(forward_batch, "input_ids", None)
                        emit("logits", file=path, mode=mode, shape=list(logits.shape),
                             can_run_graph=getattr(out, "can_run_graph", None),
                             positions=rows_pos.tolist() if rows_pos.numel() <= 4096 else None,
                             input_ids=ids.tolist() if ids is not None and ids.numel() <= _MAX_IDS else None)
                    if args.get("force_ids"):
                        _force(logits, rows_pos, args["force_ids"])
            except Exception as exc:  # noqa: BLE001
                error("logits capture", exc)
        if rec:
            try:
                fb = forward_batch
                ids = getattr(fb, "input_ids", None)
                pos = getattr(fb, "positions", None)
                logits = getattr(getattr(out, "logits_output", None), "next_token_logits", None)
                emit("forward", draft=bool(getattr(self, "is_draft_worker", False)),
                     mode=getattr(getattr(fb, "forward_mode", None), "name", None),
                     bs=getattr(fb, "batch_size", None), n_tokens=int(ids.shape[0]) if ids is not None else None,
                     can_run_graph=getattr(out, "can_run_graph", None), layer_calls=getattr(_local, "layer_calls", None),
                     input_ids=ids.tolist() if ids is not None and ids.numel() <= _MAX_IDS else None,
                     positions=pos.tolist() if pos is not None and pos.numel() <= _MAX_IDS else None,
                     logits_in_graph_buffer=_inside(logits, _graph_output_buffers(self)),
                     forward_pass_id=getattr(self, "forward_pass_id", None))
            except Exception as exc:  # noqa: BLE001
                error("forward", exc)
        return out

    cls.__init__ = __init__
    cls.forward = forward
    cls.load_model = load_model


after_import("sglang.srt.model_executor.model_runner", _patch_model_runner)
after_import("sglang.srt.layers.logits_processor", _patch_logits_processor)
