#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The twelve serving nodes of one deployment, shared by `test_bf16kv.py` and `test_fp8kv.py`.

Every node takes the deployment's launch (`serve.Session`), checks the server is alive first (a node after a dead
server is `blocked_by` the node that saw it die), records every attempt before any verdict (`records.py`) and
writes its full observation under /logs/verifier/<deployment>/. The protocols are the trusted public copies under
/tests/eval; the inputs, the calibrated bars and the case selections are private (/tests/hidden).
"""
from __future__ import annotations

import contextlib
import functools
import json
import lzma
import math
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
for _path in (str(TESTS_DIR / "eval"), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from minimax_m3.serving import accuracy, api, chat, configuration, fidelity, http, replay  # noqa: E402
from infb_verify import binding  # noqa: E402
from infb_verify.records import Blocked, Recorder, write_json  # noqa: E402

H = TESTS_DIR / "hidden"
LOGS = Path(os.environ.get("INFB_SERVED_DIR", "/logs/verifier"))
KV = {"bf16kv": "bf16", "fp8kv": "fp8"}


@functools.cache
def policy() -> dict:
    return json.loads((H / "policy.json").read_text())


@functools.cache
def replay_inputs() -> dict:
    return json.loads((H / "replay.json").read_text())


@functools.cache
def recorder(deployment: str) -> Recorder:
    return Recorder(f"serving-{deployment}")


def out(session, name: str) -> Path:
    return LOGS / session.deployment / f"{name}.json"


@contextlib.contextmanager
def node(session, name: str, **workload):
    """The dead-server rule around one node's attempt."""
    with recorder(session.deployment).attempt(name, deployment=session.deployment, **workload) as attempt:
        try:
            session.require(name)
            yield attempt
        finally:
            session.after(name)


# ------------------------------------------------------------------------------------------------ efficiency

def efficiency(session, axis: str, recipe: str) -> None:
    """Whole-serving efficiency: the task-owned client, paired against the reference phase's draws."""
    from infb_efficiency.nodes import run_axis  # noqa: PLC0415

    with node(session, f"test_0_efficiency_{axis}", axis=axis, recipe=recipe) as a:
        ok, why, record = run_axis(session.url, axis, recipe)
        a.set(result={"ok": ok, "why": why}, record=str(record) if record else None)
        assert ok, why


# ------------------------------------------------------------------------------------------------ configuration

def served_configuration(session) -> None:
    """The served configuration read from the launch log, /v1/models and the cache-config metric."""
    with node(session, "test_served_configuration", launch_log=str(session.launch_log)) as a:
        result = configuration.served_configuration(session.launch_log.read_text(errors="replace"),
                                                    session.deployment, base_url=session.url)
        write_json(out(session, "served_configuration"), result)
        a.set(departures=result["departures"])
        assert not result["departures"], "the served configuration departs from the recipe: " + "; ".join(
            result["departures"])


def api_probes(session) -> None:
    """Structure only -- completions from text and token IDs with top-5 log-probabilities and prompt
    log-probabilities, streamed completions, chat and streamed chat, eight concurrent requests."""
    ids = replay_inputs()["groups"][0]["prompts"][0]["token_ids"][:512]
    with node(session, "test_api_probes", prompt_tokens=len(ids)) as a:
        result = api.probes(session.url, prompt_ids=ids)
        write_json(out(session, "api_probes"), result)
        a.set(observed=sorted(result))


# ------------------------------------------------------------------------------------------------ chat and tools

@functools.cache
def chat_cases() -> dict:
    cases = json.loads((H / "chat_cases.json").read_text())
    if not cases.get("selected"):
        raise Blocked("tests/hidden/chat_cases.json selection (not frozen)")
    return cases


def _selected(kind: str) -> list[dict]:
    cases = chat_cases()
    pools = cases[kind]
    wanted = cases["selected"][kind]
    by_id = {c["id"]: dict(c, group=g) for g, cands in pools.items() for c in cands}
    return [by_id[i] for i in wanted]


#: Adaptive-mode streaming is not graded (a reference limitation: the serving layer's prompt scan meets the chat
#: template's literal markers); adaptive is graded non-streaming only, enabled and disabled in both modes.
GRADED_STREAMING = {"enabled": (False, True), "disabled": (False, True), "adaptive": (False,)}


def chat_reasoning(session) -> None:
    """Two prompts per thinking mode (enabled and disabled non-streaming and streaming, adaptive
    non-streaming); the split must match the raw text."""
    rows, failed = [], []
    with node(session, "test_chat_reasoning") as a:
        for case in _selected("reasoning"):
            for streaming in GRADED_STREAMING[case["group"]]:
                resp = chat.chat(session.url, case["messages"], mode=case["group"], streaming=streaming)
                verdict = chat.check_reasoning(resp, case["group"])
                rows.append({"id": case["id"], "mode": case["group"], "streaming": streaming, "response": resp,
                             **verdict})
                if not verdict["pass"]:
                    failed.append(f"{case['id']}/{'stream' if streaming else 'full'}: {verdict['problems']}")
        write_json(out(session, "chat_reasoning"), rows)
        a.set(cases=len(rows), failed=failed)
        assert not failed, "reasoning split: " + "; ".join(failed)


def tool_calls(session) -> None:
    """Four tool requests, each non-streaming and streaming; served calls must equal the calls parsed from the
    generated text, validate against the schema, and leave no namespace markup in content."""
    rows, failed = [], []
    with node(session, "test_tool_calls") as a:
        for case in _selected("tools"):
            for streaming in (False, True):
                resp = chat.chat(session.url, case["messages"], mode=case.get("thinking_mode"), streaming=streaming,
                                 tools=case["tools"])
                verdict = chat.check_tools(resp, case["tools"])
                rows.append({"id": case["id"], "streaming": streaming, "response": resp, **verdict})
                if not verdict["pass"]:
                    failed.append(f"{case['id']}/{'stream' if streaming else 'full'}: {verdict['problems']}")
        write_json(out(session, "tool_calls"), rows)
        a.set(cases=len(rows), failed=failed)
        assert not failed, "tool calls: " + "; ".join(failed)


# ------------------------------------------------------------------------------------------------ replay

def _replay_verdict(rows: list[dict], bars: dict) -> list[str]:
    return [f"{r['id']}: mean k3 {r['k3_mean']:.6g} (bar {bars['T_mean']}), max |d| {r['max_abs_d']:.6g} "
            f"(bar {bars['T_max']})" for r in rows
            if not (r["k3_mean"] <= bars["T_mean"] and r["max_abs_d"] <= bars["T_max"])]


def generation_replay(session) -> None:
    """17 prompts in C=1/3/5 groups generate 132 tokens each; after the cache reset each prompt plus its own
    continuation is teacher-forced; per sequence mean k3 <= T_mean and max |d| <= T_max. No numerical retry."""
    R = replay_inputs()
    bars = policy()["replay"][session.deployment]
    with node(session, "test_generation_replay", prompts=sum(len(g["prompts"]) for g in R["groups"])) as a:
        rows = replay.generation_replay(session.url, R["groups"], R["generate"])
        write_json(out(session, "generation_replay"), rows)
        failed = _replay_verdict(rows, bars)
        a.set(bars=bars, rows=[{k: r[k] for k in ("id", "prompt_tokens", "k3_mean", "max_abs_d")} for r in rows],
              failed=failed)
        assert not failed, "generated-continuation replay: " + "; ".join(failed)


def prefix_branch_replay(session) -> None:
    """Per source and shared-prefix length, A, B, A, B warm with reuse evidence (requests 2-4 each at least
    128*floor(S/128) hit tokens), then a cold replay of all four continuations under the node's own bars."""
    R = replay_inputs()
    bars = policy()["prefix_branch"][session.deployment]
    results, failed = [], []
    with node(session, "test_prefix_branch_replay", groups=len(R["prefix_branches"])) as a:
        for g in R["prefix_branches"]:
            res = replay.prefix_branches(session.url, g["prefix"], g["suffixes"], R["generate"])
            need = 128 * (g["shared_prefix"] // 128)
            results.append({"source": g["source"], "shared_prefix": g["shared_prefix"], **res})
            for r in res["requests"][1:]:
                if not r["prefix_cache_hit_tokens"] >= need:
                    failed.append(f"{g['source']}/S={g['shared_prefix']} request {r['request']}: "
                                  f"{r['prefix_cache_hit_tokens']} hit tokens < {need} (prefix caching not active)")
            failed += _replay_verdict([{**r, "id": f"{g['source']}/S={g['shared_prefix']}/request{r['request']}"}
                                       for r in res["requests"]], bars)
        write_json(out(session, "prefix_branch_replay"), results)
        a.set(bars=bars, failed=failed)
        assert not failed, "prefix-branch replay: " + "; ".join(failed)


# ------------------------------------------------------------------------------------------------ fidelity

# The per-sequence tail forms: (i) the interim rule, each sequence's mean k3 with its single worst position
# removed against C_b; (ii) no per-sequence bar, the band medians alone; (iii) two parts, the capped mean
# (no position contributes more than KAPPA / n) and the flip rate, each against its own bar. The frozen policy names
# the form in force.
TAIL_FORMS = {"i": "k3_mean_without_worst", "ii": None, "iii": ("k3_capped", "flips_per_1k")}
# The per-sequence statistics of engine agreement, by the form the frozen policy names: "mean" is the
# mean k3 <= T; "iii" is the capped mean and flip rate, each within its bar.
EA_FORMS = {"mean": ("k3_mean",), "iii": ("k3_capped", "flips_per_1k")}


def author_fidelity(session) -> None:
    """32 hidden sequences against the model author's log-probabilities under the frozen band policy, the
    per-sequence tail read in the form that policy names."""
    with lzma.open(H / "fidelity.json.xz", "rt") as fh:
        seqs = json.load(fh)["sequences"]
    form = policy()["fidelity"]["tail_form"]
    band_policy = policy()["fidelity"][session.deployment]
    with node(session, "test_author_fidelity", sequences=len(seqs), tail_form=form) as a:
        res = fidelity.evaluate(session.url, seqs, policy=band_policy, statistic=TAIL_FORMS[form])
        write_json(out(session, "author_fidelity"), res)
        a.set(policy=band_policy, tail_form=form, statistic=res["statistic"], band_medians=res["band_medians"],
              band_redraws=res["band_redraws"], sequence_resends=res["sequence_resends"],
              passed_band_medians=res["passed_band_medians"], passed_sequences=res["passed_sequences"],
              passed=res["passed"])
        assert res["passed"], (f"author fidelity: band medians {res['band_medians']}, redraws {res['band_redraws']}, "
                               f"re-sends {res['sequence_resends']}")


def engine_agreement(session) -> None:
    """Two 262,143-token sequences against the reference engine's own series for this deployment; per sequence
    the statistics of the form the frozen policy names within their bars (form `mean`: mean k3 <= T; form `iii`:
    the capped mean and the flip rate), re-sent once (fails only if both readings fail)."""
    R = replay_inputs()
    with lzma.open(H / "engine_agreement.json.xz", "rt") as fh:
        ref = {r["id"]: r for r in json.load(fh)["series"][session.deployment]}
    ea = policy()["engine_agreement"]
    form = ea.get("tail_form", "mean")
    names = EA_FORMS[form]
    bars = {"k3_mean": ea[session.deployment]["T"]} if form == "mean" else ea[session.deployment]["C"]
    rows, failed = [], []
    with node(session, "test_engine_agreement_262k", sequences=len(R["engine_agreement"]), tail_form=form) as a:
        for s in R["engine_agreement"]:
            recorded = {"id": s["id"], "token_ids": s["token_ids"], "reference_logprobs": ref[s["id"]]["logprobs"]}
            readings = []
            for _ in range(2):
                served = fidelity.prompt_logprobs(session.url, s["token_ids"], timeout=1800)
                readings.append(fidelity.sequence_statistics(recorded, served))
                if not fidelity.over_bars(readings[-1], names, bars):
                    break
            rows.append({"id": s["id"], "readings": readings})
            if all(fidelity.over_bars(r, names, bars) for r in readings):
                failed.append(f"{s['id']}: {[{n: round(r[n], 6) for n in names} for r in readings]} over {bars}")
        write_json(out(session, "engine_agreement"), rows)
        a.set(tail_form=form, bars=bars, rows=rows, failed=failed)
        assert not failed, "engine agreement: " + "; ".join(failed)


# ------------------------------------------------------------------------------------------------ GSM8K

def gsm8k(session) -> None:
    """All 1,319 GSM8K test questions, chat with thinking, greedy, 4,096 tokens, 64 in flight."""
    floor = policy()["gsm8k"][session.deployment]["floor"]
    with node(session, "test_gsm8k_chat_thinking", floor=floor) as a:
        res = accuracy.evaluate(session.url)
        write_json(out(session, "gsm8k"), res)
        a.set(accuracy=res["accuracy"], questions=res["questions"], wall_s=res["wall_s"])
        assert res["questions"] == 1319, res["questions"]
        assert res["accuracy"] >= floor, f"GSM8K accuracy {res['accuracy']:.4f} < floor {floor}"


# ------------------------------------------------------------------------------------------------ witness

def execution_witness(session) -> None:
    """Last on the launch, the server's profiler over one 6,144-token prompt then three concurrent
    3,000-token prompts (32 tokens each, a fresh cache salt so nothing is served from the prefix cache); every rank
    must show each entry point's identity kernels, the attend_decode kernel at least 57 x 31 times and CUDA-graph
    replay."""
    identity = binding.identity_record(session.deployment)
    W = json.loads((H / "witness.json").read_text())
    with node(session, "test_execution_witness", prompts=[len(p) for p in [W["single"], *W["concurrent"]]]) as a:
        http.request(session.url, "/start_profile", {}, timeout=300, method="POST")
        try:
            salt = uuid.uuid4().hex

            def gen(p):
                return http.request(session.url, "/v1/completions", {
                    "model": fidelity.MODEL, "prompt": p, "max_tokens": 32, "temperature": 0.0, "ignore_eos": True,
                    "cache_salt": salt}, timeout=900)

            gen(W["single"])
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(gen, W["concurrent"]))
        finally:
            http.request(session.url, "/stop_profile", {}, timeout=300, method="POST")
        observed = binding.observe_serving(identity, session.trace_dir, KV[session.deployment])
        write_json(out(session, "execution_witness"), observed)
        a.set(verdict_detail=observed["verdict"])
        assert observed["verdict"]["pass"], "execution witness: " + "; ".join(observed["verdict"]["problems"])
