#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The served nodes: the graded launch serves the recipe the instruction states, its chat endpoint renders the
reasoning effort it is asked for and returns the final channel alone as the content, and its decode path agrees with
its own prefill path."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import pathlib
import re
import statistics
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, "/tests/public_eval")
sys.path.insert(0, "/tests")

CONFIG = pathlib.Path(os.environ.get("INFB_CONFIG", "/tests/config.json"))
FIDELITY_DIR = pathlib.Path(os.environ.get("INFB_FIDELITY_DIR", "/tests/hidden/fidelity"))
# The public final-channel check. In the verifier image /public is built from tests/public/, the verifier's own
# copy, held byte-identical to the environment/public/ the agent reads.
FINAL_CHANNEL_PROBE = pathlib.Path(os.environ.get("INFB_FINAL_CHANNEL_PROBE", "/tests/public_eval/gpt_oss/final_channel/probe.py"))
# The public tool-call check, beside it and loaded the same way.
TOOL_CALL_PROBE = pathlib.Path(os.environ.get("INFB_TOOL_CALL_PROBE", "/tests/public_eval/gpt_oss/basic_tool_calls/probe.py"))
# The public decode-batch check, loaded the same way.
DECODE_BATCH_PROBE = pathlib.Path(os.environ.get("INFB_DECODE_BATCH_PROBE", "/tests/public_eval/gpt_oss/decode_batch/probe.py"))
RECIPE_ARGS = ["--tp", "4", "--cuda-graph-max-bs", "200", "--mem-fraction-static", "0.93"]
ATTENTION_BACKEND = "triton"  # the backend the instruction names: the only backend at this base that carries the attention sink
TP = 4
DECODE_PREFIXES = (256, 512)
DECODE_NEW_TOKENS = 64
# The largest-graph node: held-out prompts of this many tokens, past the 128-token window.
LARGEST_GRAPH_PROMPT_TOKENS = 256

# Sixteen short questions with an answer to reason towards. Fixed text, written for this node; the statistic is a
# mean completion length per level, not accuracy, so the questions' own difficulty does not enter the bar.
PROMPTS = [
    "A train leaves at 3:40 pm and arrives at 6:15 pm. How long is the journey in minutes?",
    "What is the sum of the first 20 odd numbers?",
    "Solve for x: 3x + 7 = 25.",
    "How many positive divisors does 360 have?",
    "A rectangle has perimeter 36 and area 80. What are its side lengths?",
    "Is 221 a prime number? Explain briefly.",
    "What is 17% of 240?",
    "Two fair dice are rolled. What is the probability that the sum is 8?",
    "Write 0.375 as a fraction in lowest terms.",
    "In how many ways can 5 different books be arranged on a shelf?",
    "A car travels 150 km in 2.5 hours. What is its average speed in metres per second?",
    "What is the least common multiple of 12, 18 and 30?",
    "A shirt costs $40 after a 20% discount. What was the original price?",
    "What is the next number in the sequence 2, 6, 12, 20, 30, ...?",
    "How many degrees is each interior angle of a regular hexagon?",
    "Solve y^2 - 5y + 6 = 0.",
]
# Twelve questions, each needing one of the public probe's three tools. Fixed text, written for this node and disjoint
# from the probe's public questions.
TOOL_QUESTIONS = [
    {"question": "What is the weather like in Paris right now?", "tool": "get_weather"},
    {"question": "Is it raining in Tokyo at the moment?", "tool": "get_weather"},
    {"question": "How cold is it in Oslo today, in celsius?", "tool": "get_weather"},
    {"question": "What's the temperature in Chicago right now in fahrenheit?", "tool": "get_weather"},
    {"question": "Convert 150 US dollars to euros.", "tool": "convert_currency"},
    {"question": "How many Japanese yen is 20 British pounds today?", "tool": "convert_currency"},
    {"question": "I have 500 Swiss francs. How much is that in Canadian dollars?", "tool": "convert_currency"},
    {"question": "What is 75 euros in US dollars at today's rate?", "tool": "convert_currency"},
    {"question": "Is flight BA117 on time?", "tool": "lookup_flight"},
    {"question": "Has flight UA901 landed yet?", "tool": "lookup_flight"},
    {"question": "What is the status of flight LH400?", "tool": "lookup_flight"},
    {"question": "Can you check whether flight AF22 is delayed?", "tool": "lookup_flight"},
]
LEVELS = ("low", "medium", "high")
MAX_TOKENS = 2048


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def _final_channel_probe():
    """The public final-channel check, loaded from FINAL_CHANNEL_PROBE without writing bytecode beside it."""
    spec = importlib.util.spec_from_file_location("infb_final_channel_probe", FINAL_CHANNEL_PROBE)
    module = importlib.util.module_from_spec(spec)
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    return module


def _tool_call_probe():
    """The public tool-call check, loaded from TOOL_CALL_PROBE the same way."""
    spec = importlib.util.spec_from_file_location("infb_tool_call_probe", TOOL_CALL_PROBE)
    module = importlib.util.module_from_spec(spec)
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    return module


def _decode_batch_probe():
    """The public decode-batch check, loaded from DECODE_BATCH_PROBE the same way."""
    spec = importlib.util.spec_from_file_location("infb_decode_batch_probe", DECODE_BATCH_PROBE)
    module = importlib.util.module_from_spec(spec)
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    return module


from gpt_oss.served import (effort_lengths as _effort_lengths, recipe_findings as _recipe_findings,
                            decode_prefill_deltas)


def effort_lengths(base_url):
    return _effort_lengths(base_url, PROMPTS, model=_config()["infb"]["model_repo"], max_tokens=MAX_TOKENS)


def recipe_findings(info, log_text):
    return _recipe_findings(info, log_text, model=_config()["infb"]["model_repo"])


def decode_prefixes() -> dict:
    """The eight held-out prefixes: the four `broad` fidelity sequences, cut at each of DECODE_PREFIXES tokens."""
    seqs = [json.loads(p.read_text()) for p in sorted(FIDELITY_DIR.glob("broad_*.json"))]
    assert len(seqs) == 4, f"the decode node reads the four broad fidelity sequences, found {len(seqs)} in {FIDELITY_DIR}"
    return {f"{s['id']}@{n}": s["token_ids"][:n] for s in seqs for n in DECODE_PREFIXES}


def recipe_maximum() -> int:
    """The recipe's `--cuda-graph-max-bs`, from its launch words: the coverage workloads' largest batch."""
    return int(RECIPE_ARGS[RECIPE_ARGS.index("--cuda-graph-max-bs") + 1])


def coverage_workloads() -> dict:
    """The fixed coverage workloads: one request and the serving concurrency must
    replay the model on CUDA graphs; the recipe's maximum batch is a served-correctness and capacity
    workload, which the engine may decode without a graph (the largest captured batch, derived from the
    observed captures, carries the replay witness instead)."""
    concurrency = _config()["infb"]["efficiency"]["recipes"]["TestHiddenAcceptance"]["C"]
    return {"c1": {"n": 1, "graph_replay": True},
            f"c{concurrency}": {"n": concurrency, "graph_replay": True},
            f"c{recipe_maximum()}": {"n": recipe_maximum(), "graph_replay": False}}


def launch_observed(model: str, base_url: str, other_args: list[str], log, observe: pathlib.Path):
    """The recipe's server through the private production observer, healthy within the launch limit."""
    import subprocess
    import time
    limit = _config()["infb"]["components"]["operation_limits_sec"]["launch"]
    _, host, port = base_url.split(":")
    env = dict(os.environ, GPT_OSS_OBSERVE=str(observe),
               PYTHONPATH=os.pathsep.join(["/tests/public_eval", "/tests", os.environ.get("PYTHONPATH", "")]))
    argv = [sys.executable, "-m", "infb_components.probes.observe_launch", "--model-path", model, *other_args,
            "--host", host[2:], "--port", port]
    process = subprocess.Popen(argv, stdout=log, stderr=log, env=env)
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the server exited {process.returncode} before becoming healthy")
        try:
            if requests.get(f"{base_url}/health_generate", timeout=5).status_code == 200:
                return process
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(f"the server did not become healthy within the {limit} s launch limit")


def largest_graph_prompts(n: int) -> list[list[int]]:
    """n held-out prompts of LARGEST_GRAPH_PROMPT_TOKENS tokens: consecutive, non-overlapping windows of the fidelity
    set's natural-text bands (broad, plain and long; the Harmony conversations and the public examples excluded), in
    file order. 204 windows exist, so any graph up to the recipe's batch 200 is covered. Natural text, as the decode
    node's prefixes are: a Harmony conversation cut at an arbitrary token reads |decode - prefill| of 0.05-0.17 whether
    the implementation is correct or not, which is the content and not the defect."""
    windows = []
    for p in sorted(FIDELITY_DIR.glob("*.json")):
        if not p.name.startswith(("broad_", "plain_", "long_")):
            continue
        ids = json.loads(p.read_text())["token_ids"]
        W = LARGEST_GRAPH_PROMPT_TOKENS
        windows += [ids[s:s + W] for s in range(0, len(ids) - W + 1, W)]
    assert len(windows) >= n, f"the largest-graph node needs {n} held-out prompts and the fidelity set yields {len(windows)}"
    return windows[:n]


class TestServed(unittest.TestCase):
    """Launches the candidate's server once at the recipe; every node reads it."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils import kill_process_tree
        from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

        cls._kill = kill_process_tree
        cls.base_url = DEFAULT_URL_FOR_TEST
        log_dir = pathlib.Path(os.environ.get("INFB_SERVED_LOG_DIR", "/logs/verifier"))
        if not log_dir.is_dir():
            log_dir = pathlib.Path(tempfile.mkdtemp())
        cls.log_path = log_dir / "served-server.txt"
        cls.observe = log_dir / "served-observations"
        cls._log = open(cls.log_path, "w")
        cls.process = launch_observed(_config()["infb"]["model_repo"], cls.base_url, RECIPE_ARGS,
                                      cls._log, cls.observe)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            cls._kill(cls.process.pid)
        if getattr(cls, "_log", None) is not None:
            cls._log.close()

    def test_decode_agrees_with_prefill(self):
        T = _config()["infb"]["bars"]["decode_prefill_abs_median_T"]
        per_prefix = {}
        worst = {}
        for key, prefix in decode_prefixes().items():
            d = decode_prefill_deltas(self.base_url, prefix)
            per_prefix[key] = statistics.median(d)
            worst[key] = max(d)
        stat = statistics.median(per_prefix.values())
        print(f"[INFB] decode_prefill_abs_median={stat:.6g} T={T:.6g} n={len(per_prefix)} "
              f"worst_prefix={max(per_prefix, key=per_prefix.get)}:{max(per_prefix.values()):.6g}")
        print(f"[INFB] decode_prefill_per_prefix={json.dumps({k: round(v, 6) for k, v in per_prefix.items()})} "
              f"max_abs={json.dumps({k: round(v, 4) for k, v in worst.items()})}")
        self.assertLess(stat, T, f"the decode path's log-probabilities are {stat:.6g} away from the same tokens' "
                                 f"prefill log-probabilities (median over prefixes of the median |delta|), not below "
                                 f"T={T:.6g}: decode and prefill do not compute the same model; per prefix {per_prefix}")

    def test_decode_at_the_largest_graph(self):
        from infb_components import gates
        from infb_components.probes.graph_coverage import coverage
        T = _config()["infb"]["bars"]["largest_graph_decode_worst_request_T"]
        workloads = coverage_workloads()
        record = coverage(self.base_url, self.observe, pathlib.Path("/tests/public_eval"),
                          largest_graph_prompts(max(spec["n"] for spec in workloads.values())),
                          workloads=workloads, recipe_max=recipe_maximum(), bar=T)
        path = self.log_path.with_name("served-coverage.json")
        path.write_text(json.dumps(record, indent=2) + "\n")
        largest = record["largest_graph"]
        summary = {name: [{"worst": d.get("worst"), "valid": d["structurally_valid"]} for d in entry["draws"]]
                   for name, entry in [*record["workloads"].items(), ("largest_graph", largest)]}
        print(f"[INFB] largest_graph_captured_sizes={json.dumps([r['captured'] for r in record['ranks']])} "
              f"n={largest['n']} T={T:.6g} readings={json.dumps(summary)} record={path}")
        gates.coverage_numerical(record, bar=T)
        concurrency = _config()["infb"]["efficiency"]["recipes"]["TestHiddenAcceptance"]["C"]
        gates.coverage_runtime(record, concurrency=concurrency, layers=36)

    def test_final_channel_is_the_content(self):
        probe = _final_channel_probe()
        rows = probe.probe(self.base_url, PROMPTS, model=_config()["infb"]["model_repo"], max_tokens=MAX_TOKENS)
        bad = {i: r["reasons"] for i, r in enumerate(rows) if r["reasons"]}
        print(f"[INFB] final_channel_clean={len(rows) - len(bad)}/{len(rows)} "
              f"with_reasoning_content={sum(1 for r in rows if r['reasoning_content_chars'])} not_clean={json.dumps(bad)}")
        print(f"[INFB] final_channel_content_heads={json.dumps([r['content'][:60] for r in rows])}")
        self.assertEqual(bad, {}, f"{len(bad)} of {len(rows)} answers at low effort carry Harmony channel text in their "
                                  f"content, so the chat endpoint does not return the final channel alone: {bad}")

    def test_tool_calls_are_returned(self):
        probe = _tool_call_probe()
        rows = probe.probe(self.base_url, TOOL_QUESTIONS, model=_config()["infb"]["model_repo"])
        bad = {f"{i % len(TOOL_QUESTIONS)}{'s' if r['stream'] else ''}": r["reasons"] for i, r in enumerate(rows)
               if r["reasons"]}
        plain = [r for r in rows if not r["stream"]]
        print(f"[INFB] tool_calls_returned={len(rows) - len(bad)}/{len(rows)} "
              f"plain={sum(1 for r in plain if not r['reasons'])}/{len(plain)} "
              f"streamed={sum(1 for r in rows if r['stream'] and not r['reasons'])}/{len(rows) - len(plain)} "
              f"not_tool_calls={json.dumps(bad)}")
        print(f"[INFB] tool_calls_named={json.dumps([[c['name'] for c in r['calls']] for r in rows])}")
        self.assertEqual(bad, {}, f"{len(bad)} of {len(rows)} answers to questions that need a tool did not come back as "
                                  f"that tool call under the recipe's launch: {bad}")

    def test_recipe_conformance(self):
        r = requests.get(f"{self.base_url}/get_server_info", timeout=300)
        self.assertEqual(r.status_code, 200, f"/get_server_info returned {r.status_code}")
        self._log.flush()
        served, bad = recipe_findings(r.json(), self.log_path.read_text(errors="replace"))
        print(f"[INFB] served_configuration={json.dumps(served, default=str)}")
        self.assertEqual(bad, [], "the graded launch does not serve the recipe the instruction states: " + "; ".join(bad))

    def test_reasoning_effort_orders_length(self):
        k = _config()["infb"]["bars"]["reasoning_effort_step_ratio"]
        m = effort_lengths(self.base_url)
        steps = {f"{lo}->{hi}": m[hi]["mean"] / m[lo]["mean"] for lo, hi in zip(LEVELS, LEVELS[1:])}
        print(f"[INFB] reasoning_effort_mean_completion_tokens={json.dumps({e: round(v['mean'], 2) for e, v in m.items()})} "
              f"step_ratios={json.dumps({s: round(r, 4) for s, r in steps.items()})} bar={k}")
        print(f"[INFB] reasoning_effort_median_completion_tokens={json.dumps({e: v['median'] for e, v in m.items()})} "
              f"at_max_tokens={json.dumps({e: v['at_max_tokens'] for e, v in m.items()})}")
        print(f"[INFB] reasoning_effort_per_prompt={json.dumps({e: v['per_prompt'] for e, v in m.items()})}")
        for step, ratio in steps.items():
            self.assertGreaterEqual(ratio, k,
                                    f"mean completion length rises only {ratio:.4f}x over {step}, below the bar {k}x, "
                                    "so the endpoint does not render the requested reasoning level")


if __name__ == "__main__":
    unittest.main()
