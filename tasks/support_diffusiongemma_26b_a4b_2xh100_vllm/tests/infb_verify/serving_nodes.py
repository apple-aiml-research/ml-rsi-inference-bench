#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The verifier's side of the served checks shared by both deployments: the
private inputs and bars around the public protocols of eval/diffusion_gemma/serving/ (the grading
copy under /tests/eval is identical to the agent's /app/eval). Each function takes the live server
and returns the protocol's record, whose "pass" is the node's verdict; the test modules call them
in the verifier's session order and write the records."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from eval.diffusion_gemma.serving import accuracy, api, chat, configuration, long_context as LC, preemption, repairs
from eval.diffusion_gemma.serving import images as IM
from eval.diffusion_gemma.serving.common import choice, content
from eval.diffusion_gemma.serving.common import tokenizer as tok  # noqa: F401 -- used by sibling modules
from eval.diffusion_gemma.serving.recipes import SERVED_NAME

from infb_verify import common, preemption_rule, readsuite


def prose(n: int, offset: int) -> list[int]:
    """n token ids of the hidden prose pool from `offset` (wrapping)."""
    ids = common.hidden("prose-pool.json.xz")["ids"]
    while len(ids) < offset + n:
        ids = ids + ids
    return ids[offset:offset + n]


def served_configuration(srv, deployment: str) -> dict:
    return configuration.served_configuration(srv, deployment)


def api_probes(srv) -> dict:
    """The public probes, with the image chat judged as the text chat is judged: status 200 and
    the response's token ids present, its content recorded -- the checkpoint can answer an image prompt with
    thinking disabled by an immediate end of sequence. The public probe's own image-chat record is kept beside it."""
    rec = api.api_probes(srv)
    img = IM.data_url(IM.image(7))
    r = srv.post("/v1/chat/completions", {"model": SERVED_NAME, "max_tokens": 128, "return_token_ids": True,
                                          "chat_template_kwargs": {"enable_thinking": False},
                                          "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                                                                                    {"type": "text", "text": "What shapes do you see?"}]}]})
    rec["image_chat_public"] = rec.pop("image_chat")
    rec["image_chat"] = {"status": r.get("_status"), "has_ids": bool(choice(r).get("token_ids")), "content": content(r)[:200]}
    ok = True
    for name in ("completion_text", "completion_ids"):
        row = rec[name]
        ok &= (row["status"] == 200 and row["n"] == row["positions"] and row["not_top1"] == 0
               and row["mass_over_1"] == 0 and row["bad_values"] == 0)
    ok &= rec["stream_completion"]["status"] == 200 and rec["stream_completion"]["chunks"] > 0
    ok &= rec["chat"]["status"] == 200 and rec["chat"]["has_ids"]
    ok &= rec["chat_stream"]["status"] == 200 and rec["chat_stream"]["chunks"] > 0
    ok &= rec["image_chat"]["status"] == 200 and rec["image_chat"]["has_ids"]
    ok &= all(st == 200 for st in rec["concurrent8"])
    rec["public_pass"] = rec["pass"]
    rec["pass"] = bool(ok)
    return rec


def repair_structured_outputs(srv) -> dict:
    return repairs.structured_outputs(srv)


def repair_long_output(srv) -> dict:
    return repairs.long_output(srv, prose(1000, 200_000))


def repair_first_token(srv, cases: list[dict]) -> dict:
    return repairs.first_token(srv, cases)


def concurrent_logprobs(srv) -> dict:
    return repairs.concurrent_logprobs(srv)


def chat_reasoning(srv, cases: list[dict]) -> dict:
    """The public protocol, with one exclusion: a thinking-disabled stream draw whose raw output
    opens a thought block is recorded -- it counts in the thought-block rate -- and its split is not judged. The
    base's Gemma4 reasoning parser splits such a block in the plain form and passes it through as content in the
    stream form (the prompt already closed the thought channel): an engine limitation the reference and every
    candidate share. The record names the verifier version, whose request bodies the case selection
    must be read from."""
    rec = chat.chat_reasoning(srv, cases)
    ok = True
    for x in rec["cases"]:
        x["judged"] = not (x["form"] == "stream" and not x["thinking"] and x["opens_thought_block"])
        if x["judged"]:
            ok &= x["pass"]
    rec["public_pass"] = rec["pass"]
    rec["pass"] = bool(ok)
    rec["verifier_version"] = common.verifier_version()
    return rec


def reads_node(srv, read_set: str, deployment: str, bars: dict | None, chat: bool = False) -> dict:  # noqa: ARG001
    data = common.hidden("reads.json.xz")
    reads = readsuite.select(data, read_set, deployment)
    return readsuite.verdict(srv, data, reads, bars)


def step1_node(srv, deployment: str, bars: dict | None, increment_bars: dict | None) -> dict:
    """The step-1 node: the two-step reads against the author's step 1 under the band verdict, and beside
    them the step-1 increment over the same first draw (readsuite.increment), whose author one-step values
    are the recording run's own step 0 for these canvases (tests/hidden/step1_step0.json). Both must
    pass."""
    data = common.hidden("reads.json.xz")
    reads = readsuite.select(data, "step1", deployment)
    rec = readsuite.verdict(srv, data, reads, bars, keep_lp=True)
    rec["increment"] = readsuite.increment(srv, data, reads, rec["draws"]["1"], common.hidden("step1_step0.json")["step0"],
                                           increment_bars)
    rec["reads_pass"] = rec["pass"]
    rec["pass"] = bool(rec["pass"] and rec["increment"]["pass"])
    return rec


def long_context(srv) -> dict:
    return LC.near_limit(srv, prose(LC.NEAR_LIMIT_PROMPT, 0))


def gsm8k(srv, floor: float | None) -> dict:
    rec = accuracy.gsm8k(srv)
    rec["floor"] = floor
    rec["pass"] = rec["statuses"] == [200] and (floor is None and common.CALIBRATION or (floor is not None and rec["accuracy"] >= floor))
    return rec


#: The preemption load: at most four sequences run (--max-num-seqs 4), so the load's prompts plus
#: outputs must together overflow the recipe's pool while they run.
PREEMPTION_LOAD = {"requests": 4, "prompt_len": 190_000, "max_tokens": 16_384}


def preemption_load() -> dict:
    """The calibrated load; INFB_PREEMPTION_LOAD (a JSON object) overrides it for calibration runs."""
    path = common.HIDDEN / "preemption_load.json"
    load = {**PREEMPTION_LOAD, **(common.hidden("preemption_load.json") if path.exists() else {})}
    return {**load, **json.loads(os.environ.get("INFB_PREEMPTION_LOAD") or "{}")}


#: The per-request limit when no frozen bar applies: long enough to measure walls uncut.
CALIBRATION_PREEMPTION_LIMIT_S = 7200


def preemption_resume(srv, load: dict | None = None, limit_s: float | None = None) -> dict:
    """limit_s: the frozen per-request limit (policy.json "preemption", budgets.json preemption_workload);
    None in a calibration launch."""
    load = load or preemption_load()
    prompts = [prose(int(load["prompt_len"]), 20_000 * k) for k in range(int(load["requests"]))]
    return preemption.preemption_resume(srv, prompts, int(load["max_tokens"]),
                                        timeout=float(limit_s or CALIBRATION_PREEMPTION_LIMIT_S))


POOL_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def kv_pool_tokens(srv) -> list[int]:
    """The launch's KV pool in tokens, as its startup log reports it (the load rule reads it at calibration)."""
    try:
        text = Path(srv.log_path).read_text(errors="replace")
    except (OSError, TypeError):
        return []
    return [int(m.group(1).replace(",", "")) for m in POOL_RE.finditer(text)]


def preemption_node(srv, policy) -> dict:
    """The preemption node: the load of record, then, only when that run passes and records
    no preemption, one re-run at the heavier load (preemption_rule). Each load's per-request limit is the policy's,
    max(1,800 s, 4 W_ref) over the reference's walls at that load (budgets.json preemption_workload and
    preemption_rerun). A calibration launch runs both loads uncut and records both. The top-level walls and count are
    the load of record's."""
    load = preemption_load()
    heavier = {**load, "max_tokens": int(load["heavier_max_tokens"])}
    if policy is None:
        runs = [preemption_resume(srv, load), preemption_resume(srv, heavier)]
        verdict = {"pass": all(r["pass"] for r in runs), "reason": "calibration: both loads recorded"}
    else:
        bars = policy["preemption"]
        runs = [preemption_resume(srv, load, bars["limit_s"])]
        if preemption_rule.rerun(runs[0]):
            runs.append(preemption_resume(srv, heavier, bars["heavier_limit_s"]))
        verdict = preemption_rule.verdict(runs)
    return {"runs": runs, "walls": runs[0]["walls"], "preemptions": runs[0]["preemptions"], "load": runs[0]["load"],
            "limit_s": runs[0]["limit_s"], "reran": len(runs) > 1 and policy is not None,
            "kv_pool_tokens": kv_pool_tokens(srv),
            "demand_tokens": [r["load"]["requests"] * (max(r["load"]["prompt_len"]) + r["load"]["max_tokens"]) for r in runs],
            **verdict}
