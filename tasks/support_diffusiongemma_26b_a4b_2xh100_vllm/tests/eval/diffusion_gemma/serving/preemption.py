#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Preemption and resume, public."""
from __future__ import annotations

from .common import completion_tokens, gather
from .recipes import SERVED_NAME


def preemption_resume(srv, prompts: list[list[int]], max_tokens: int, timeout: float = 1800.0) -> dict:
    before = srv.metric("vllm:num_preemptions_total")
    bodies = [{"model": SERVED_NAME, "prompt": list(p), "max_tokens": int(max_tokens), "ignore_eos": True, "temperature": 1.0}
              for p in prompts]
    outs = gather(lambda b: srv.post("/v1/completions", b, timeout=timeout), bodies, concurrency=len(bodies))
    after = srv.metric("vllm:num_preemptions_total")
    toks = [completion_tokens(o) for o in outs]
    probe = srv.post("/v1/completions", {"model": SERVED_NAME, "prompt": "Hello", "max_tokens": 8})
    return {"load": {"requests": len(prompts), "prompt_len": [len(p) for p in prompts], "max_tokens": int(max_tokens)},
            "limit_s": timeout,
            "completion_tokens": toks, "statuses": [o.get("_status") for o in outs],
            "preemptions": (after - before) if after is not None and before is not None else None,
            "probe": probe.get("_status"), "walls": [o.get("_wall_s") for o in outs],
            "pass": all(t == int(max_tokens) for t in toks) and probe.get("_status") == 200}
