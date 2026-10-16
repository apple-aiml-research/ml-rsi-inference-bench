#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The served configuration, public: what a launch resolved, from its log and /v1/models."""
from __future__ import annotations

import json
import re

from .recipes import SERVED_NAME

#: Between a 256-token canvas per step (block diffusion) and one token per step (an autoregressive decode).
MIN_TOKENS_PER_STEP = 16
STEP_METRICS = ("vllm:iteration_tokens_total_sum", "vllm:iteration_tokens_total_count")


def scheduling(srv) -> dict:
    """One 512-token generation alone (greedy, ignore_eos): the tokens the server scheduled per engine step."""
    before = {n: srv.metric(n) for n in STEP_METRICS}
    r = srv.post("/v1/completions", {"model": SERVED_NAME, "prompt": "Write a long story about a lighthouse keeper.",
                                     "max_tokens": 512, "ignore_eos": True, "temperature": 0.0})
    d = {n: srv.metric(n) - before[n] for n in STEP_METRICS}
    steps, tokens = d["vllm:iteration_tokens_total_count"], d["vllm:iteration_tokens_total_sum"]
    per_step = tokens / steps if steps else None
    return {"status": r.get("_status"), "steps": steps, "tokens": tokens, "tokens_per_step": per_step,
            "min_tokens_per_step": MIN_TOKENS_PER_STEP,
            "pass": r.get("_status") == 200 and per_step is not None and per_step >= MIN_TOKENS_PER_STEP}


def served_configuration(srv, deployment: str) -> dict:
    log = srv.log_path.read_text(errors="replace")
    status, models = srv.get("/v1/models")
    rec = {"models_status": status}
    try:
        data = json.loads(models)["data"][0]
        rec["served_name"], rec["max_model_len"] = data.get("id"), data.get("max_model_len")
    except Exception:  # noqa: BLE001
        rec["served_name"], rec["max_model_len"] = None, None
    engine = next((l for l in log.splitlines() if "Initializing a V1 LLM engine" in l), "")

    def cfg(key):
        m = re.search(rf"{key}=([^,]+)", engine)
        return m.group(1).strip() if m else None

    m = re.search(r"max_num_batched_tokens=(\d+)", log)
    rec.update(tp=cfg("tensor_parallel_size"), enforce_eager=cfg("enforce_eager"), trust_remote_code=cfg("trust_remote_code"),
               enable_prefix_caching=cfg("enable_prefix_caching"), chunked_prefill="Chunked prefill is enabled" in log,
               max_num_batched_tokens=int(m.group(1)) if m else None,
               async_scheduling="Asynchronous scheduling is enabled" in log,
               cudagraph_mode=(re.search(r"'cudagraph_mode': <CUDAGraphMode\.([A-Z_]+)", engine) or [None, None])[1],
               block_size=(re.search(r"block_size[=:] ?(\d+)", engine) or [None, None])[1])
    args_line = next((l for l in log.splitlines() if "non-default args" in l), "")
    rec["max_num_seqs_4"] = "'max_num_seqs': 4" in args_line
    rec["mm_limits"] = "'limit_mm_per_prompt': {'image': 7}" in args_line and "'max_soft_tokens': 1120" in args_line
    want_tp = "2" if deployment == "tp2" else "1"
    rec["pass"] = bool(rec["served_name"] == SERVED_NAME and rec["max_model_len"] == 262144 and rec["tp"] == want_tp
                       and rec["max_num_seqs_4"] and rec["enable_prefix_caching"] == "True" and rec["chunked_prefill"]
                       and rec["enforce_eager"] == "False" and rec["async_scheduling"] and rec["trust_remote_code"] == "False"
                       and rec["mm_limits"])
    if deployment == "tp1":
        rec["scheduling"] = scheduling(srv)
        rec["pass"] = rec["pass"] and rec["scheduling"]["pass"]
    return rec
