#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""R-ADAPTIVE: the adaptive controller's updates, decisions and their consumption, observed in the worker processes."""
from __future__ import annotations

import copy
import threading

from . import after_import, control, emit, error, on_control

_local = threading.local()
_state: dict = {"substitute": None, "armed_snapshot": {}, "instances": {}, "snapshots": {}, "orig": {}}
_REPLAY_FLOOR = 100
_REPLAY_INTERVALS = 10


def _state_copy(obj):
    """A deep copy of a controller's state in which tensors, modules and the engine's own workers stay shared: a
    controller that keeps a reference to heavy engine objects is copied without copying them (they are not its
    decision state)."""
    memo: dict = {}
    try:
        import torch
        atoms = (torch.Tensor, torch.nn.Module)
    except Exception:  # noqa: BLE001
        atoms = ()
    seen: set = set()

    def seed(o, depth: int) -> None:
        if depth > 3 or id(o) in seen:
            return
        seen.add(id(o))
        values = list(o.values()) if isinstance(o, dict) else list(getattr(o, "__dict__", {}).values())
        for v in values:
            mod = type(v).__module__ or ""
            if (atoms and isinstance(v, atoms)) or mod.startswith(("sglang.srt.speculative.eagle", "sglang.srt.model_executor")):
                memo[id(v)] = v
            elif isinstance(v, dict) or hasattr(v, "__dict__"):
                seed(v, depth + 1)
    seed(obj, 0)
    return copy.deepcopy(obj, memo)


def _slot_view(params) -> dict:
    """The live configuration and state of every slot, read from the object (whatever its update formula)."""
    out = {}
    slots = getattr(params, "_slots", None) or {}
    for bs, slot in sorted(slots.items()):
        out[str(bs)] = {k: getattr(slot, k, None) for k in ("candidate_steps", "current_steps", "ema_accept_len",
                                                            "_batch_count", "warmup_batches", "update_interval",
                                                            "ema_alpha", "up_hysteresis", "down_hysteresis",
                                                            "ceiling_coeff")}
    return out


def _route_key(params, batch_size: int):
    try:
        slots = params._slots
        for bs, slot in slots.items():
            if slot is params._route(batch_size):
                return bs
    except Exception:  # noqa: BLE001 - a refactored controller may route differently; the tier is still recorded
        return None
    return None


def _patch_params(mod) -> None:
    cls = mod.AdaptiveSpeculativeParams
    orig_init, orig_get, orig_verify = cls.__init__, cls.get_steps_for_batch, cls.on_verify_complete
    orig_set_bs = getattr(cls, "set_cuda_graph_bs", None)
    _state["orig"].update(get=orig_get, verify=orig_verify)

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            _state["instances"][id(self)] = self
            emit("adaptive_init", obj=id(self), args=[repr(a)[:80] for a in args],
                 kwargs={k: repr(v)[:120] for k, v in kwargs.items()}, slots=_slot_view(self),
                 candidate_steps=list(getattr(self, "candidate_steps", []) or []))
        except Exception as exc:  # noqa: BLE001
            error("adaptive_init", exc)

    def set_cuda_graph_bs(self, cuda_graph_bs):
        r = orig_set_bs(self, cuda_graph_bs)
        try:
            emit("adaptive_cuda_graph_bs", obj=id(self), cuda_graph_bs=list(cuda_graph_bs or []))
        except Exception as exc:  # noqa: BLE001
            error("adaptive_cuda_graph_bs", exc)
        return r

    def _first_use(self):
        if id(self) not in _state["snapshots"]:
            try:
                _state["snapshots"][id(self)] = _state_copy(self)
                emit("adaptive_snapshot", obj=id(self), slots=_slot_view(self))
            except Exception as exc:  # noqa: BLE001
                error("adaptive_snapshot", exc)
                _state["snapshots"][id(self)] = None

    def get_steps_for_batch(self, batch_size):
        _first_use(self)
        chosen = orig_get(self, batch_size)
        returned = chosen
        sub = _state["substitute"]
        if sub is not None:
            returned = sub
        try:
            _local.batch = getattr(_local, "batch", 0) + 1
            emit("adaptive_choose", obj=id(self), bs=batch_size, slot=_route_key(self, batch_size), chosen=chosen,
                 returned=returned, substituted=sub is not None, batch=_local.batch)
        except Exception as exc:  # noqa: BLE001
            error("adaptive_choose", exc)
        return returned

    def on_verify_complete(self, num_correct_drafts_per_req, batch_size):
        _first_use(self)
        try:
            before = _slot_view(self).get(str(_route_key(self, batch_size)))
        except Exception:  # noqa: BLE001
            before = None
        decision = orig_verify(self, num_correct_drafts_per_req, batch_size)
        returned = decision
        sub = _state["substitute"]
        if sub is not None and decision is not None:
            returned = sub
        try:
            after = _slot_view(self).get(str(_route_key(self, batch_size)))
            emit("adaptive_feedback", obj=id(self), bs=batch_size, slot=_route_key(self, batch_size),
                 feedback=list(num_correct_drafts_per_req), before=before, after=after, decision=decision,
                 returned=returned, substituted=sub is not None)
        except Exception as exc:  # noqa: BLE001
            error("adaptive_feedback", exc)
        return returned

    cls.__init__ = __init__
    cls.get_steps_for_batch = get_steps_for_batch
    cls.on_verify_complete = on_verify_complete
    if orig_set_bs is not None:
        cls.set_cuda_graph_bs = set_cuda_graph_bs


def _patch_controller(mod) -> None:
    cls = mod.AdaptiveController
    orig = cls._activate

    def _activate(self, speculative_num_steps):
        before = getattr(self.worker, "speculative_num_steps", None)
        r = orig(self, speculative_num_steps)
        try:
            emit("adaptive_apply", requested=speculative_num_steps, worker_before=before,
                 worker_after=getattr(self.worker, "speculative_num_steps", None),
                 draft_tokens_after=getattr(self.worker, "speculative_num_draft_tokens", None))
        except Exception as exc:  # noqa: BLE001
            error("adaptive_apply", exc)
        return r

    cls._activate = _activate


def _patch_worker(mod) -> None:
    draft_cls = getattr(mod, "EagleDraftWorker", None)
    worker_cls = getattr(mod, "EAGLEWorkerV2", None)
    if draft_cls is not None and hasattr(draft_cls, "draft"):
        orig_draft = draft_cls.draft

        def draft(self, batch):
            steps = getattr(self, "speculative_num_steps", None)
            out = orig_draft(self, batch)
            try:
                draft_tokens = getattr(out, "draft_token", None)
                emit("spec_draft", steps=steps, bs=len(getattr(batch, "reqs", []) or []),
                     draft_token_num=getattr(out, "draft_token_num", None),
                     draft_token_shape=list(draft_tokens.shape) if draft_tokens is not None else None,
                     runner=type(getattr(self, "cuda_graph_runner", None)).__name__,
                     runner_steps=getattr(getattr(self, "cuda_graph_runner", None), "speculative_num_steps", None))
            except Exception as exc:  # noqa: BLE001
                error("spec_draft", exc)
            return out

        draft_cls.draft = draft
    if worker_cls is not None and hasattr(worker_cls, "verify"):
        orig_verify = worker_cls.verify

        def verify(self, batch, *args, **kwargs):
            tokens = getattr(self, "speculative_num_draft_tokens", None)
            out = orig_verify(self, batch, *args, **kwargs)
            try:
                accept = getattr(out, "accept_lens", None)
                emit("spec_verify", num_draft_tokens=tokens, steps=getattr(self, "speculative_num_steps", None),
                     bs=len(getattr(batch, "reqs", []) or []),
                     accept_lens=accept.tolist() if accept is not None and accept.numel() <= 4096 else None)
            except Exception as exc:  # noqa: BLE001
                error("spec_verify", exc)
            return out

        worker_cls.verify = verify


def _replay_one(snapshot, bs: int, history: str, n: int) -> dict:
    """One counterfactual history on a fresh copy of the pre-use state, through the controller's own (unwrapped)
    methods: each batch proposes the tier the controller chooses, and reports zero or all of its drafts accepted."""
    get, verify = _state["orig"]["get"], _state["orig"]["verify"]
    params = _state_copy(snapshot)
    tiers = []
    for _ in range(n):
        k = get(params, bs)
        tiers.append(k)
        verify(params, [0] * bs if history == "zero" else [k] * bs, bs)
    final = get(params, bs)
    return {"final": final, "first_tiers": tiers[:12], "last_tiers": tiers[-12:], "n": n}


def _on_control(c: dict) -> None:
    phase, args = c.get("phase"), c.get("args") or {}
    if phase == "adaptive_substitute":
        if _state["substitute"] is None:
            _state["armed_snapshot"] = {k: _state_copy(v.__dict__) for k, v in _state["instances"].items()}
        _state["substitute"] = int(args["tier"])
        emit("adaptive_substitute_armed", tier=_state["substitute"])
    elif phase == "adaptive_restore":
        _state["substitute"] = None
        for key, saved in _state["armed_snapshot"].items():
            obj = _state["instances"].get(key)
            if obj is not None:
                obj.__dict__.clear()
                obj.__dict__.update(saved)
        emit("adaptive_restored", instances=len(_state["armed_snapshot"]),
             slots={str(k): _slot_view(v) for k, v in _state["instances"].items()})
        _state["armed_snapshot"] = {}
    elif phase == "adaptive_replay":
        sub, _state["substitute"] = _state["substitute"], None
        try:
            for key, obj in _state["instances"].items():
                snap = _state["snapshots"].get(key)
                if not snap:
                    emit("adaptive_replay", obj=key, error="no pre-use snapshot of this controller")
                    continue
                slots = {}
                for bs in [int(b) for b in args.get("slots", [])]:
                    view = _slot_view(snap).get(str(_route_key(snap, bs))) or {}
                    warm, interval = view.get("warmup_batches"), view.get("update_interval")
                    n = max(_REPLAY_FLOOR, int(warm or 0) + _REPLAY_INTERVALS * int(interval or 0))
                    slots[str(bs)] = {"slot": _route_key(snap, bs), "config": view, "length": n,
                                      "length_rule": f"max({_REPLAY_FLOOR}, warmup {warm} + {_REPLAY_INTERVALS} x "
                                                     f"update_interval {interval})",
                                      "zero": _replay_one(snap, bs, "zero", n), "all": _replay_one(snap, bs, "all", n)}
                emit("adaptive_replay", obj=key, slots=slots)
        except Exception as exc:  # noqa: BLE001
            error("adaptive_replay", exc)
        finally:
            _state["substitute"] = sub


after_import("sglang.srt.speculative.adaptive_spec_params", _patch_params)
after_import("sglang.srt.speculative.adaptive_runtime_state", _patch_controller)
after_import("sglang.srt.speculative.eagle_worker_v2", _patch_worker)
on_control(_on_control)
_ = control
