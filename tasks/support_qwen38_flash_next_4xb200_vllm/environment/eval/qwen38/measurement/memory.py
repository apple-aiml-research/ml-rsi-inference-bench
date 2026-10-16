#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""PLE auxiliary memory: attributable storage over the combined allocator and driver timeline.

For a captured case, one traced draw per rank (never a timing draw) restores the starting state and
executes the measured sequence eagerly while two timelines are recorded:

* the PyTorch caching allocator's own event history (block allocations and frees), and
* CUDA driver and runtime allocation activity (CUPTI callbacks on `cuMemAlloc*`, `cuMemCreate`,
  `cudaMalloc*` and their frees), for storage outside that allocator. Allocations the PyTorch
  allocator makes for its own segments are identified by address and not counted twice.

The charged total, per rank, is

    persistent auxiliary storage reachable from the region (outside parameters and bound state)
  + bound-state capacity beyond the logical live state of the case's requests
  + the peak, over the merged timeline, of storage allocated during the draw and still live
    (outputs retained across the measured sequence included).

Parameters, the region's mathematical inputs and the logical live state (per request `C` channels
by `H` positions in the state dtype) are excluded. Storage ranges are deduplicated. Unused reserved
pool capacity is reported separately and is not working memory.
"""
from __future__ import annotations

import ctypes
import gc
import os
import re
import sys
import threading
import time
from pathlib import Path

CUPTI_CB_DOMAIN_DRIVER_API = 1
CUPTI_CB_DOMAIN_RUNTIME_API = 2
CUPTI_API_EXIT = 1

DRIVER = ("cuMemAlloc_v2", "cuMemFree_v2", "cuMemAllocAsync", "cuMemFreeAsync", "cuMemAllocFromPoolAsync",
          "cuMemCreate", "cuMemRelease", "cuMemAllocManaged", "cuMemAllocPitch_v2")
RUNTIME = ("cudaMalloc_v3020", "cudaFree_v3020", "cudaMallocAsync_v11020", "cudaFreeAsync_v11020",
           "cudaMallocManaged_v6000", "cudaMallocPitch_v3020", "cudaMallocFromPoolAsync_v11020")


class MemoryObserverError(RuntimeError):
    """The observer could not trace allocation activity; the case has no valid memory reading."""


class _CallbackData(ctypes.Structure):
    _fields_ = [("callbackSite", ctypes.c_int), ("functionName", ctypes.c_char_p),
                ("functionParams", ctypes.c_void_p), ("functionReturnValue", ctypes.c_void_p),
                ("symbolName", ctypes.c_char_p), ("context", ctypes.c_void_p),
                ("contextUid", ctypes.c_uint32), ("correlationData", ctypes.c_void_p),
                ("correlationId", ctypes.c_uint32)]


_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(_CallbackData))


def _cupti_root() -> Path:
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    for base in (spec.submodule_search_locations if spec else []):
        for sub in ("cu13", "cuda_cupti"):
            root = Path(base) / sub
            if (root / "include" / "cupti_driver_cbid.h").is_file():
                return root
    raise MemoryObserverError("CUPTI headers/library are not installed in the engine environment")


def _cbids(header: Path, prefix: str, names) -> dict:
    text = header.read_text()
    out = {}
    for name in names:
        m = re.search(rf"{prefix}{re.escape(name)}\s*=\s*(\d+)", text)
        if m:
            out[int(m.group(1))] = name
    return out


class DriverTrace:
    """CUPTI callback subscription recording allocation and free activity with host timestamps."""

    def __init__(self):
        root = _cupti_root()
        libs = sorted((root / "lib").glob("libcupti.so*"))
        if not libs:
            raise MemoryObserverError(f"no libcupti under {root / 'lib'}")
        self.lib = ctypes.CDLL(str(libs[0]))
        self.driver = _cbids(root / "include" / "cupti_driver_cbid.h", "CUPTI_DRIVER_TRACE_CBID_", DRIVER)
        self.runtime = _cbids(root / "include" / "cupti_runtime_cbid.h", "CUPTI_RUNTIME_TRACE_CBID_", RUNTIME)
        self.events = []
        self.lock = threading.Lock()
        self.subscriber = ctypes.c_void_p()
        self._cb = _CALLBACK(self._on)

    def _on(self, userdata, domain, cbid, data):
        try:
            d = data.contents
            if d.callbackSite != CUPTI_API_EXIT:
                return
            name = (self.driver if domain == CUPTI_CB_DOMAIN_DRIVER_API else self.runtime).get(cbid)
            if name is None:
                return
            ret = ctypes.cast(d.functionReturnValue, ctypes.POINTER(ctypes.c_int)).contents.value \
                if d.functionReturnValue else 0
            if ret != 0:
                return
            p = d.functionParams
            u64 = ctypes.POINTER(ctypes.c_uint64)
            size = None
            if name in ("cuMemAlloc_v2", "cuMemAllocAsync", "cuMemAllocFromPoolAsync", "cuMemAllocManaged",
                        "cudaMalloc_v3020", "cudaMallocAsync_v11020", "cudaMallocManaged_v6000",
                        "cudaMallocFromPoolAsync_v11020"):
                ptr = ctypes.cast(p, ctypes.POINTER(ctypes.c_void_p)).contents.value
                addr = ctypes.cast(ptr, u64).contents.value
                size = ctypes.cast(p + 8, u64).contents.value
                event = ("alloc", addr, size)
            elif name in ("cuMemAllocPitch_v2", "cudaMallocPitch_v3020"):
                ptr = ctypes.cast(p, ctypes.POINTER(ctypes.c_void_p)).contents.value
                addr = ctypes.cast(ptr, u64).contents.value
                pitch = ctypes.cast(ctypes.cast(p + 8, ctypes.POINTER(ctypes.c_void_p)).contents.value, u64).contents.value
                height = ctypes.cast(p + 24, u64).contents.value
                event = ("alloc", addr, pitch * height)
            elif name == "cuMemCreate":
                handle_ptr = ctypes.cast(p, ctypes.POINTER(ctypes.c_void_p)).contents.value
                handle = ctypes.cast(handle_ptr, u64).contents.value
                event = ("create", handle, ctypes.cast(p + 8, u64).contents.value)
            elif name == "cuMemRelease":
                event = ("release", ctypes.cast(p, u64).contents.value, None)
            else:  # frees take the pointer by value as their first parameter
                event = ("free", ctypes.cast(p, u64).contents.value, None)
            with self.lock:
                self.events.append((time.perf_counter_ns(), name) + event)
        except Exception:  # never raise through a C callback
            with self.lock:
                self.events.append((time.perf_counter_ns(), "observer-error", "error", 0, None))

    def __enter__(self):
        rc = self.lib.cuptiSubscribe(ctypes.byref(self.subscriber), self._cb, None)
        if rc != 0:
            raise MemoryObserverError(f"cuptiSubscribe failed ({rc}); another CUPTI subscriber may be active")
        for domain, table in ((CUPTI_CB_DOMAIN_DRIVER_API, self.driver), (CUPTI_CB_DOMAIN_RUNTIME_API, self.runtime)):
            for cbid in table:
                rc = self.lib.cuptiEnableCallback(1, self.subscriber, domain, cbid)
                if rc != 0:
                    raise MemoryObserverError(f"cuptiEnableCallback({domain}, {cbid}) failed ({rc})")
        return self

    def __exit__(self, *exc):
        self.lib.cuptiUnsubscribe(self.subscriber)
        return False


def _storage(t):
    s = t.untyped_storage()
    return s.data_ptr(), s.nbytes()


def _merge(ranges):
    out = []
    for start, end in sorted(ranges):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def persistent(module, excluded_ranges) -> dict:
    """Non-parameter CUDA storage reachable from the region's module tree and the globals of the
    modules that define its classes, deduplicated by storage range, outside excluded ranges."""
    import torch
    seen, found = set(), {}

    def walk(value, path, depth=0):
        if isinstance(value, torch.Tensor):
            if value.device.type == "cuda" and value.untyped_storage().nbytes():
                start, size = _storage(value)
                found.setdefault((start, start + size), path)
            return
        if id(value) in seen or depth > 6:
            return
        seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            for name, child in vars(value).items():
                if name in ("_parameters",):
                    continue
                walk(child, f"{path}.{name}", depth + 1)
        elif isinstance(value, dict):
            for k, child in list(value.items()):
                walk(child, f"{path}[{k!r}]", depth + 1)
        elif isinstance(value, (list, tuple, set)):
            for i, child in enumerate(list(value)):
                walk(child, f"{path}[{i}]", depth + 1)

    walk(module, "region")
    root = Path(os.environ.get("QWEN38_ENGINE_ROOT", "/app")).resolve()
    sources = {type(m).__module__ for m in module.modules()}
    for name in sorted(sources):
        mod = sys.modules.get(name)
        path = getattr(mod, "__file__", None)
        if isinstance(path, str) and Path(path).resolve().is_relative_to(root):
            for key, value in list(vars(mod).items()):
                if isinstance(value, (torch.Tensor, dict, list, tuple)):
                    walk(value, f"global:{name}.{key}", 1)
    excluded = _merge(excluded_ranges)

    def outside(start, end):
        """Bytes of [start, end) not covered by excluded ranges."""
        covered = 0
        for a, b in excluded:
            lo, hi = max(a, start), min(b, end)
            if lo < hi:
                covered += hi - lo
        return (end - start) - covered

    rows = []
    for (start, end) in [tuple(r) for r in _merge(found.keys())]:
        extra = outside(start, end)
        if extra > 0:
            rows.append({"start": start, "bytes": extra, "path": next(p for (a, b), p in found.items() if a >= start and b <= end)})
    return {"bytes": sum(r["bytes"] for r in rows), "allocations": rows[:64]}


def account(worker, stored: dict, request: dict) -> dict:
    """Memory evidence for one captured PLE case on this rank (no verdict)."""
    import torch
    from . import native
    entry, region = stored["entry"], stored["region"]
    module = entry["module"]
    case = stored["case"]
    snapshots = stored["result"]["snapshots"]
    state = native.bound_state(module, entry.get("static_context"))
    group_blocks = stored["result"].get("group_blocks", {})
    # Exclusions: parameters and the region's bound cache storage (its logical and physical state).
    excluded = [(s, s + n) for s, n in (_storage(p) for p in module.parameters())]
    for _, t, _ in state:
        start, nbytes = _storage(t)
        excluded.append((start, start + nbytes))
    kept = persistent(module, excluded)
    # Bound-state capacity beyond the logical live state of the case's requests.
    requests = len(case["lengths"])
    channels, positions = request["channels"], request["positions"]
    logical = 0
    physical = 0
    for name, t, layer in state:
        rows = len(group_blocks.get(layer, [])) if layer else 0
        row_bytes = t[0].numel() * t.element_size() if t.dim() else 0
        physical += rows * row_bytes
        logical += requests * channels * positions * t.element_size()
    excess = max(0, physical - logical)
    # The traced draw.
    sequence = worker._sequence(stored)
    for rows in snapshots["pre"][0]:
        rows.restore()
    for inputs in stored.get("inputs", [])[:stored["measured"]]:
        inputs.restore()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base_allocated = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    trace = DriverTrace()
    torch.cuda.memory._record_memory_history(enabled="all", context=None, stacks="python", max_entries=1_000_000)
    try:
        with trace:
            # A marker allocation aligns the allocator's clock with the host clock of driver events.
            marker_host = time.perf_counter_ns()
            marker = torch.empty(1 << 16, dtype=torch.uint8, device="cuda")
            marker_addr = marker.data_ptr()
            del marker
            begun = time.perf_counter_ns()
            outputs = sequence()
            torch.cuda.synchronize()
            ended = time.perf_counter_ns()
        snap = torch.cuda.memory._snapshot()
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    peak_reserved = torch.cuda.max_memory_reserved()
    device = torch.cuda.current_device()
    events = [e for e in (snap.get("device_traces") or [[]] * (device + 1))[device]]
    segments = {e["addr"] for e in events if e.get("action") == "segment_alloc"}
    for seg in snap.get("segments", []):
        segments.add(seg.get("address"))
    # Allocator timeline (its own µs clock), aligned to the host clock by the marker allocation.
    marks = [e for e in events if e.get("action") == "alloc" and e.get("addr") == marker_addr]
    if not marks:
        raise MemoryObserverError("the allocator history did not record the alignment marker")
    offset = marker_host - marks[0]["time_us"] * 1000
    alloc = [(e["time_us"] * 1000 + offset, +e["size"] if e["action"] == "alloc" else -e["size"])
             for e in events if e.get("action") in ("alloc", "free_requested") and e.get("addr") != marker_addr
             and e["time_us"] * 1000 + offset >= begun]
    external, live, sizes = [], {}, {}
    errors = [e for e in trace.events if e[1] == "observer-error"]
    for when, name, kind, key, size in trace.events:
        if kind == "alloc":
            # A runtime allocation is also reported by the driver call it makes: count an address once.
            if key in segments or key in live:
                continue
            live[key] = size
            external.append((when, size))
        elif kind == "create":
            sizes[key] = size
            external.append((when, size))
        elif kind == "free" and key in live:
            external.append((when, -live.pop(key)))
        elif kind == "release" and key in sizes:
            external.append((when, -sizes.pop(key)))
    timeline = sorted([(t, d, "allocator") for t, d in alloc] + [(t, d, "driver") for t, d in external])
    current = peak = 0
    peak_at = None
    for when, delta, source in timeline:
        current += delta
        if current > peak:
            peak, peak_at = current, when
    del outputs
    transient = max(0, peak)
    total = kept["bytes"] + excess + transient
    return {"persistent_auxiliary_bytes": kept["bytes"], "persistent_allocations": kept["allocations"],
            "state_physical_bytes": physical, "state_logical_bytes": logical, "state_excess_bytes": excess,
            "transient_peak_bytes": transient, "total_bytes": total,
            "allocator_events": len(alloc), "driver_events": len(external), "observer_errors": len(errors),
            "reserved_growth_bytes": max(0, peak_reserved - base_reserved),
            "baseline_allocated_bytes": base_allocated, "draw_ns": ended - begun,
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
