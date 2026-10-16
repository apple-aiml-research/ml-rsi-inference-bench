#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Real-runner coverage of history-carrying steps during the MTP3 state workloads.

Installed on the diagnostic launch only (never during timing), it wraps the runner's base
`execute_model(scheduler_output)` at the instance level and counts, from the engine's own scheduler
outputs, requests scheduled with prior history, prefills continuing a nonzero history (a restored
prefix or a later chunk) and steps that mix speculative verification with a prefill. It reads no
candidate-specific field, metadata class or tensor name: whether the candidate used that history
correctly is judged by the state workloads' outcomes and by the component checks, which run the
same runner over native histories.
"""
from __future__ import annotations

import copy
import json


def _computed(scheduler_output) -> dict[str, tuple[int, int]]:
    """req_id -> (tokens computed before this step, tokens scheduled in it)."""
    scheduled = dict(scheduler_output.num_scheduled_tokens)
    out = {}
    for new in scheduler_output.scheduled_new_reqs:
        out[new.req_id] = (int(new.num_computed_tokens), int(scheduled.get(new.req_id, 0)))
    cached = scheduler_output.scheduled_cached_reqs
    for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
        out[req_id] = (int(computed), int(scheduled.get(req_id, 0)))
    if set(out) != set(scheduled):
        raise AssertionError(f"scheduled requests {sorted(scheduled)} differ from their records {sorted(out)}")
    return out


def install(worker):
    if hasattr(worker, "_qwen38_lifecycle"):
        raise ValueError("lifecycle observer already installed")
    runner = worker.model_runner
    native = runner.execute_model
    report = {"schema": "qwen38-runner-coverage/1", "steps": 0, "scheduled_rows": 0, "nonempty_history_rows": 0,
              "prefill_rows": 0, "restored_prefill_rows": 0, "spec_rows": 0, "mixed_spec_prefill_steps": 0,
              "failure_count": 0, "failures": [], "examples": []}
    prompts: dict[str, int] = {}

    def observe(scheduler_output):
        for new in scheduler_output.scheduled_new_reqs:
            prompts[new.req_id] = len(new.prompt_token_ids or [])
        rows = _computed(scheduler_output)
        spec = {k: len(v) for k, v in (scheduler_output.scheduled_spec_decode_tokens or {}).items() if v}
        if not rows:
            return
        report["steps"] += 1
        prefills = 0
        for req_id, (computed, scheduled) in rows.items():
            report["scheduled_rows"] += 1
            report["nonempty_history_rows"] += int(computed > 0)
            if req_id in spec:
                report["spec_rows"] += 1
            elif computed < prompts.get(req_id, 0):
                prefills += 1
                report["prefill_rows"] += 1
                if computed > 0:
                    report["restored_prefill_rows"] += 1
                    if len(report["examples"]) < 8:
                        report["examples"].append({"computed": computed, "scheduled": scheduled,
                                                   "prompt": prompts[req_id]})
        report["mixed_spec_prefill_steps"] += int(bool(spec) and prefills > 0)
        for req_id in scheduler_output.finished_req_ids or ():
            prompts.pop(req_id, None)

    def execute_model(scheduler_output, *args, **kwargs):
        try:
            observe(scheduler_output)
        except Exception as exc:  # the observer never changes serving
            report["failure_count"] += 1
            if len(report["failures"]) < 16:
                report["failures"].append(repr(exc))
        return native(scheduler_output, *args, **kwargs)

    runner.execute_model = execute_model
    worker._qwen38_lifecycle = (runner, report)
    return report


def control(worker, *, action="read"):
    if action == "install":
        return copy.deepcopy(install(worker))
    if action not in ("read", "remove") or not hasattr(worker, "_qwen38_lifecycle"):
        raise ValueError("lifecycle observer is not installed or action is invalid")
    runner, report = worker._qwen38_lifecycle
    result = copy.deepcopy(report)
    if action == "remove":
        del runner.execute_model
        del worker._qwen38_lifecycle
    return result


class LifecycleWorker:
    def qwen38_lifecycle(self, serialized):
        import torch.distributed as dist
        request = json.loads(serialized)
        identity = {"rank": dist.get_rank(), "request_id": request.get("request_id")}
        try:
            if set(request) - {"request_id", "action"}:
                raise ValueError("unknown lifecycle request fields")
            value = control(self, action=request.get("action", "read"))
            return {**identity, "status": "complete", "value": value}
        except Exception as exc:
            return {**identity, "status": "failed", "error": repr(exc)}
