#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private, untimed observations of the loaded expert modules and ordinary GPU execution.

Installed by `infb_components.probes.observe_launch` in every TP worker. It records
raw evidence; private gates supply every limit. A binding identifies native modules
and may adapt a different organization, but supplies no verdict, clock or byte count.

Per rank, `rank-<r>.jsonl` receives:
  loaded                      binding identities and storage at load (before KV pools)
  graphs                      the batch sizes the engine captured as CUDA graphs
  gpu_execution               a profiled forward: kernels, and which ran inside graph launches
  target_execution            each distinct (mode, tokens, batch, graph) execution
  resident_after_execution    storage after a profiled forward (lazy expansion is persistent)
  forward                     while a workload window is open: every forward with its live
                              and padded batch, graph replays and eager native-module calls
"""
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import time

import torch

from gpt_oss.native_experts import NativeExperts, implementation_identity, live_storages, source_identity

_installed = False
_rank = None
_states = {}
_counters = {"replays": 0, "native_calls": 0}
_window = {"mtime": None, "name": None}


def emit(kind, **record):
    root = Path(os.environ["GPT_OSS_OBSERVE"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"rank-{_rank}.jsonl").open("a") as output:
        output.write(json.dumps({"kind": kind, "rank": _rank, "pid": os.getpid(),
                                 "time_ns": time.time_ns(), **record}, allow_nan=False) + "\n")


def binding():
    name = os.environ.get("GPT_OSS_BINDING")
    if not name:
        return NativeExperts
    module, attribute = name.split(":")
    return getattr(importlib.import_module(module), attribute)


def window():
    """The open workload window's name, or None. The coverage client writes the file."""
    path = Path(os.environ["GPT_OSS_OBSERVE"]) / "window.json"
    try:
        mtime = path.stat().st_mtime_ns
    except FileNotFoundError:
        _window.update(mtime=None, name=None)
        return None
    if mtime != _window["mtime"]:
        try:
            _window.update(mtime=mtime, name=json.loads(path.read_text()).get("name"))
        except (OSError, ValueError):
            _window.update(mtime=None, name=None)
    return _window["name"]


def graph_attribution(trace_path):
    """Kernels launched by cudaGraphLaunch versus kernels launched directly, from the trace."""
    events = json.loads(Path(trace_path).read_text()).get("traceEvents", [])
    launches = {e.get("args", {}).get("correlation") for e in events
                if e.get("name") == "cudaGraphLaunch" and e.get("ph") == "X"}
    launches.discard(None)
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    inside = [e for e in kernels if e.get("args", {}).get("correlation") in launches]
    return {"graph_launches": len(launches), "graph_kernels": len(inside),
            "graph_kernel_names": len({e.get("name") for e in inside}),
            "direct_kernels": len(kernels) - len(inside)}


def _attach(runner, before):
    global _rank
    _rank = runner.tp_rank
    model = runner.model
    if getattr(model.config, "model_type", None) != "gpt_oss":
        return
    modules = binding().locate(model)
    assert len(modules) == model.config.num_hidden_layers, "binding must identify every target layer"
    state = {"model": model, "modules": modules, "before": before, "invocations": [],
             "seen": set(), "traced": set(), "executed": set(), "window_traced": set()}
    _states[id(runner)] = state
    identities = []
    for layer_id, module in modules:
        identity = implementation_identity(module)
        identities.append({"layer": layer_id, "identity": identity, "storage": live_storages(module)})
        def before_call(native, args, kwargs, layer=layer_id):
            x = kwargs.get("hidden_states", args[0] if args else None)
            assert isinstance(x, torch.Tensor), "native binding has no activation input"
            capturing = torch.cuda.is_current_stream_capturing()
            if not capturing:
                _counters["native_calls"] += 1
            key = (layer, tuple(x.shape), capturing)
            if key not in state["seen"]:
                state["seen"].add(key)
                state["invocations"].append({"layer": layer, "shape": list(x.shape), "capturing": capturing})
        module.register_forward_pre_hook(before_call, with_kwargs=True)
        if str(layer_id) == os.environ.get("GPT_OSS_PERTURB_LAYER"):
            def perturb(native, args, kwargs, output):
                return output * 0
            module.register_forward_hook(perturb, with_kwargs=True)
    torch.cuda.synchronize()
    emit("loaded", identity=source_identity(model), bindings=identities,
         expert_storage=live_storages([module for _, module in modules]),
         all_model_storage=live_storages(model),
         allocator_live_delta=torch.cuda.memory_allocated()-before["live"],
         allocator_reserved_delta=torch.cuda.memory_reserved()-before["reserved"],
         device_used_delta=before["free"]-torch.cuda.mem_get_info()[0])


def install():
    global _installed
    if _installed or not os.environ.get("GPT_OSS_OBSERVE"):
        return
    _installed = True
    original_replay = torch.cuda.CUDAGraph.replay
    def replay(self, *args, **kwargs):
        _counters["replays"] += 1
        return original_replay(self, *args, **kwargs)
    torch.cuda.CUDAGraph.replay = replay
    from sglang.srt.model_executor.model_runner import ModelRunner
    original_load = ModelRunner.load_model
    def load(self, *args, **kwargs):
        torch.cuda.synchronize()
        before = {"live": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                  "free": torch.cuda.mem_get_info()[0]}
        result = original_load(self, *args, **kwargs)
        _attach(self, before)
        return result
    ModelRunner.load_model = load
    original_graphs = ModelRunner.init_cuda_graphs
    def init_cuda_graphs(self, *args, **kwargs):
        result = original_graphs(self, *args, **kwargs)
        if id(self) in _states:
            runner = getattr(self, "cuda_graph_runner", None)
            emit("graphs", captured=sorted(int(bs) for bs in getattr(runner, "graphs", {}) or {}),
                 capture_bs=[int(bs) for bs in getattr(runner, "capture_bs", []) or []],
                 max_bs=int(getattr(runner, "max_bs", 0) or 0), runner=runner is not None)
        return result
    ModelRunner.init_cuda_graphs = init_cuda_graphs
    original_forward = ModelRunner.forward
    def forward(self, fb, *args, **kwargs):
        state = _states.get(id(self))
        if state is None:
            return original_forward(self, fb, *args, **kwargs)
        count = fb.input_ids.numel()
        mode = str(fb.forward_mode.name)
        label = mode + ("-large" if count >= 128 else "-small")
        name = window()
        tracing = label not in state["traced"] and mode != "IDLE"
        if name is not None and (name, mode) not in state["window_traced"] and mode == "DECODE":
            # One profiled decode forward per workload window: warm-request evidence.
            tracing = True
            state["window_traced"].add((name, mode))
        replays, calls = _counters["replays"], _counters["native_calls"]
        t0 = time.time_ns()
        if tracing:
            state["traced"].add(label)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as trace:
                output, graph = original_forward(self, fb, *args, **kwargs)
                torch.cuda.synchronize()
            suffix = f"-{name}" if name is not None else ""
            path = Path(os.environ["GPT_OSS_OBSERVE"]) / f"rank-{_rank}-{label}{suffix}.trace.json"
            trace.export_chrome_trace(str(path))
            kernels = Counter(event.name for event in trace.events()
                              if event.device_type == torch.autograd.DeviceType.CUDA)
            emit("gpu_execution", mode=mode, tokens=count, graph=bool(graph), window=name,
                 batch_size=fb.batch_size, kernels=dict(kernels), trace=str(path),
                 attribution=graph_attribution(path), invocations=state["invocations"])
            state["invocations"] = []
        else:
            output, graph = original_forward(self, fb, *args, **kwargs)
        t1 = time.time_ns()
        runner = getattr(self, "cuda_graph_runner", None)
        if name is not None:
            emit("forward", window=name, mode=mode, tokens=count, live_bs=fb.batch_size,
                 graph=bool(graph), padded_bs=int(runner.bs) if graph and runner is not None else None,
                 raw_bs=int(runner.raw_bs) if graph and runner is not None else None,
                 replays=_counters["replays"]-replays, native_calls=_counters["native_calls"]-calls,
                 t0_ns=t0, t1_ns=t1, profiled=tracing)
        execution = (mode, count, fb.batch_size, bool(graph))
        if execution not in state['executed']:
            state['executed'].add(execution)
            emit('target_execution', mode=mode, tokens=count, batch_size=fb.batch_size, graph=bool(graph))
        if tracing:
            # Observe after execution too: lazy expansion is persistent storage.
            # Remove only actual request/KV tensors, excluding any alias of a
            # model tensor. Global live allocation still catches retained copies
            # outside the model; free allocator reservation is reported separately.
            torch.cuda.synchronize()
            model = state['model']
            pool_bytes = live_storages([self.req_to_token_pool, self.token_to_kv_pool],
                                       excluding=model)['bytes']
            before = state['before']
            emit('resident_after_execution', mode=mode, graph=bool(graph), window=name,
                 expert_storage=live_storages([module for _, module in state['modules']]),
                 all_model_storage=live_storages(model), pool_storage_bytes=pool_bytes,
                 allocator_live_delta=torch.cuda.memory_allocated()-before['live']-pool_bytes,
                 allocator_reserved_delta=torch.cuda.memory_reserved()-before['reserved'],
                 device_used_delta=before['free']-torch.cuda.mem_get_info()[0])
        return output, graph
    ModelRunner.forward = forward
