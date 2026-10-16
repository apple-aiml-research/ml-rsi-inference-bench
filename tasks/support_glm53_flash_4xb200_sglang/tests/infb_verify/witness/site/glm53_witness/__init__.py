#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The private witness observer, loaded into every engine process of the untimed witness launch (never the graded one).

It observes the engine's existing interfaces and records what executed; it never supplies model behavior. Pieces:
- an event sink: each process appends JSON lines to `<witness dir>/events/<pid>.jsonl` (buffered, flushed by a
  background thread and at exit), every event carrying the process's rank identity once it is known;
- post-import patches: a `sys.meta_path` finder that runs a patch function right after a named engine module executes,
  so the engine's own classes are wrapped where they are defined, in every process;
- a control channel: `<witness dir>/control.json` (a phase name, a sequence number and arguments) is read by a
  background thread every 0.25 s; each process acknowledges the sequence number it applied in `acks/<pid>.json`, so the
  verifier sends a phase's requests only after every rank has armed it;
- tensor stores: `store(name, tensor_dict)` writes safetensors files under `<witness dir>/tensors/<pid>/`.

Which pieces are active is `INFB_GLM53_WITNESS_FEATURES` (comma-separated: `adaptive`, `runtime`, `attention`).
Anything that goes wrong inside the observer is an `observer_error` event; the engine call it wraps proceeds as it would
without the observer.
"""
from __future__ import annotations

import atexit
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time
import traceback
from typing import Any, Callable

ROOT = os.environ.get("INFB_GLM53_WITNESS_DIR", "")
FEATURES = {f.strip() for f in os.environ.get("INFB_GLM53_WITNESS_FEATURES", "").split(",") if f.strip()}
PID = os.getpid()

_lock = threading.Lock()
_buffer: list[str] = []
_seq = 0
_rank: dict = {}
_control: dict = {"seq": 0, "phase": "idle", "args": {}}
_control_listeners: list[Callable[[dict], None]] = []
_patches: dict[str, list[Callable]] = {}
_installed = False


def _dir(*parts: str) -> str:
    path = os.path.join(ROOT, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def set_rank(**facts: Any) -> None:
    """The process's rank identity (tp/dp/ep/attn-tp ranks and sizes, device, role); later events carry it."""
    with _lock:
        _rank.update({k: v for k, v in facts.items() if v is not None})
    emit("rank", **facts)


def rank() -> dict:
    return dict(_rank)


def emit(kind: str, **fields: Any) -> None:
    global _seq
    with _lock:
        _seq += 1
        rec = {"k": kind, "n": _seq, "t": time.time(), "pid": PID, "phase": _control.get("phase"),
               "cseq": _control.get("seq"), **({"rank": dict(_rank)} if _rank else {}), **fields}
        try:
            _buffer.append(json.dumps(rec, default=_jsonable))
        except Exception as exc:  # noqa: BLE001
            _buffer.append(json.dumps({"k": "observer_error", "n": _seq, "pid": PID, "where": f"emit {kind}",
                                       "error": repr(exc)[:300]}))


def _jsonable(v: Any) -> Any:
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return v.detach().to("cpu").tolist() if v.numel() <= 4096 else f"<tensor {tuple(v.shape)} {v.dtype}>"
    except Exception:  # noqa: BLE001
        pass
    return repr(v)[:200]


def error(where: str, exc: BaseException) -> None:
    emit("observer_error", where=where, error=repr(exc)[:500], tb=traceback.format_exc()[-2000:])


def flush() -> None:
    with _lock:
        lines, _buffer[:] = list(_buffer), []
    if not lines:
        return
    try:
        with open(os.path.join(_dir("events"), f"{PID}.jsonl"), "a") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError:
        pass


def control() -> dict:
    return _control


def on_control(fn: Callable[[dict], None]) -> None:
    _control_listeners.append(fn)


def _read_control() -> None:
    path = os.path.join(ROOT, "control.json")
    try:
        doc = json.load(open(path))
    except (OSError, ValueError):
        return
    if not isinstance(doc, dict) or doc.get("seq") == _control.get("seq"):
        return
    _control.clear()
    _control.update({"seq": doc.get("seq"), "phase": doc.get("phase", "idle"), "args": doc.get("args") or {}})
    emit("control", control=dict(_control))
    for fn in list(_control_listeners):
        try:
            fn(dict(_control))
        except Exception as exc:  # noqa: BLE001
            error(f"control listener {getattr(fn, '__name__', fn)}", exc)
    try:
        ack = {"pid": PID, "seq": _control["seq"], "phase": _control["phase"], "rank": dict(_rank), "t": time.time()}
        tmp = os.path.join(_dir("acks"), f".{PID}.tmp")
        with open(tmp, "w") as fh:
            json.dump(ack, fh)
        os.replace(tmp, os.path.join(_dir("acks"), f"{PID}.json"))
    except OSError:
        pass


def _loop() -> None:
    while True:
        try:
            _read_control()
            flush()
        except Exception as exc:  # noqa: BLE001
            error("background loop", exc)
        time.sleep(0.25)


def store(name: str, tensors: dict) -> str:
    """Write CPU copies of `tensors` to `<dir>/tensors/<pid>/<name>.safetensors`; returns the path."""
    from safetensors.torch import save_file
    path = os.path.join(_dir("tensors", str(PID)), f"{name}.safetensors")
    save_file({k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}, path)
    return path




def after_import(module: str, fn: Callable) -> None:
    """Run `fn(module_object)` right after `module` executes (or now, if it already has)."""
    if module in sys.modules:
        _run_patch(module, fn, sys.modules[module])
    else:
        _patches.setdefault(module, []).append(fn)


def _run_patch(name: str, fn: Callable, mod: Any) -> None:
    try:
        fn(mod)
        emit("patched", module=name, patch=getattr(fn, "__name__", repr(fn)))
    except Exception as exc:  # noqa: BLE001
        error(f"patch {name}:{getattr(fn, '__name__', fn)}", exc)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader, name: str):
        self.inner, self.name = inner, name

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        for fn in _patches.pop(self.name, []):
            _run_patch(self.name, fn, module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _patches:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader, fullname)
                return spec
        return None


def install() -> None:
    global _installed
    if _installed or not ROOT:
        return
    _installed = True
    sys.meta_path.insert(0, _Finder())
    emit("process", argv=sys.argv[:6], features=sorted(FEATURES), python=sys.version.split()[0],
         cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    threading.Thread(target=_loop, name="glm53-witness", daemon=True).start()
    atexit.register(flush)
    from . import runtime  # noqa: F401 - rank facts and forward bookkeeping, needed by every feature
    if "adaptive" in FEATURES:
        from . import adaptive  # noqa: F401
    if "attention" in FEATURES:
        from . import attention  # noqa: F401
