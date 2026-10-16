#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The served-API nodes: the paired efficiency gate, then the hidden GSM8K slice through vLLM's own evaluator."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from infb_verify import _trusted, serve

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
BARS = X.get("bars", {})
SLICE = json.loads((TESTS_DIR / "hidden" / "gsm8k_slice.json").read_text())
DATASETS = Path(X["dataset_root"]) / "gsm8k"
#: The quality protocol the agent is given (`tests/quality.py`), the grading copy: the evaluator's call, its settings,
#: the offline data and the acceptance-length reading. This node adds the hidden slice and the floor.
_QUALITY = _trusted.load("tests/quality.py")


# The paired efficiency gate, scored. test_00_efficiency_measure measures both axes of the
# session server's first launch before every other node, on a server whose GPUs have seen the same
# history as the reference's (served and measured earlier in this same trial, on the same GPUs and
# the same hidden draw). test_0_efficiency_relaunch runs after every other node and, where an axis's
# ratio lies near the bar or the reference reads low for the recipe, launches both sides once more
# within the time the verifier's clock leaves. The two scored nodes judge from the record: each side
# keeps its better valid launch, and an axis passes iff the record's verdict is `pass`, the
# candidate's value at least rho = 0.8 of the reference's, in exact arithmetic. decode_c1 is the
# median of five draws on both sides, throughput the median of three candidate draws against the
# reference's median of three.
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from infb_efficiency.nodes import measure_first_launch, scored, settle  # noqa: E402

#: The key of the one scored recipe in `config.json` `infb.efficiency.recipes`: the last component
#: of the junit classname of the nodes that score it. The task's acceptance has no registered upstream
#: CLASS to mix `EfficiencyMixin` into (the maintainers' obligations here are unit tests and a plain
#: evaluator), so the nodes are module-level functions and the key is this module's name.
RECIPE = "test_hidden"


def _the_recipe_is_what_the_kit_launches() -> None:
    """Checked at import. The verifier serves the reference from `launch.argv` (a static argv: there is no class
    whose setUpClass it could intercept), so that argv must BE the recipe this module's server fixture
    launches for the candidate -- upstream's RemoteOpenAIServer argv with the declared port. A drift
    would make every ratio a ratio against a different recipe, and nothing else would notice."""
    assert Path(__file__).stem == RECIPE, (Path(__file__).stem, RECIPE)
    eff = X.get("efficiency", {})
    recipe = (eff.get("recipes") or {}).get(RECIPE)
    assert recipe, f"config.json infb.efficiency.recipes has no entry {RECIPE!r}"
    want = ["vllm", "serve", X["model_id"], *X["recipe"]["server_args"],
            "--port", str(eff["reference"]["port"]), "--seed", "0"]
    assert recipe["launch"]["argv"] == want, (recipe["launch"]["argv"], want)


_the_recipe_is_what_the_kit_launches()


def test_00_efficiency_measure(server):
    """Both efficiency axes of the session server's first launch, measured before every other node;
    each axis's outcome is recorded for its scored node. It fails only where a measurement raised."""
    measure_first_launch(server.url_root, RECIPE)


def test_0_efficiency_relaunch(server):
    """After every other node of the session: the relaunch of both sides where an axis's
    first launch reads near the bar or the reference reads low for the recipe (config.json `infb.efficiency.relaunch`),
    with the second launch read as `test_served_configuration` reads the first (`serve.relaunch_check`). Not graded: a
    relaunch the verifier's clock leaves no time for, or one cut off by its deadline, leaves the first launch deciding."""
    first_argv = serve.launch_argv(server)
    settle(RECIPE, lambda: serve.stop(server), lambda: serve.relaunch(server),
           check=lambda _url: serve.relaunch_check(server, first_argv))


def test_0_efficiency_decode_c1(server):
    """S1: per-user decode speed at concurrency 1, at least 0.8 of the reference's: the median of five draws on both
    sides, each side's better launch."""
    ok, why, _ = scored(server.url_root, "decode_c1", RECIPE)
    assert ok, why


def test_0_efficiency_throughput(server):
    """S_C: output throughput at C = 16 (the LowLatency class's load), at least 0.8 of
    the reference's: the median of three draws on both sides, each side's better launch."""
    ok, why, _ = scored(server.url_root, "throughput", RECIPE)
    assert ok, why


def _acceptance_length(server) -> float:
    import requests  # noqa: PLC0415

    r = requests.get(server.url_for("metrics"), timeout=60)
    r.raise_for_status()
    return _QUALITY.acceptance_length(r.text)


def test_gsm8k_hidden(server):
    """GSM8K on the hidden 200-question slice at or above the measured floor."""
    if why := serve.down(server):
        pytest.fail(serve.blocked_message(why))
    ge = _QUALITY.evaluator(APP_DIR)
    host, port = server.url_root.rsplit(":", 1)
    result, n = _QUALITY.run_gsm8k(ge, host, int(port), DATASETS / "train.jsonl", DATASETS / "test.jsonl",
                                   SLICE["indices"])
    test = SLICE["indices"]
    acceptance = _acceptance_length(server)
    record = _QUALITY.result_record(result, n)
    record.update({"slice": SLICE["name"], "dspark_acceptance_length": acceptance})
    print("INFB_GSM8K " + json.dumps(record))
    floor = BARS.get("gsm8k_floor")
    if floor is None:
        pytest.skip("gsm8k_floor is not published yet; the measurement is printed above")
    assert float(result["accuracy"]) >= float(floor), (
        f"GSM8K on the hidden {len(test)}-question slice is {result['accuracy']:.4f} "
        f"({record['correct']} of {len(test)}), below the floor {floor} "
        f"(invalid rate {result['invalid_rate']:.3f}; DSpark acceptance length {acceptance:.3f})"
    )


# ── the deployment: its configuration facts and its runtime evidence, scored ───────────────


def test_served_configuration(instrumented_server):
    """The deployment the recipe states, as configuration facts and as runtime evidence.

    Configuration facts: every rank's launch log of the timed session's server and of the instrumented launch, read
    against the recipe (`recipe.RESOLVED`: TP4 with no pipeline, data or expert parallelism, automatic MoE backend
    selection, CompilationMode.NONE with CUDA graphs FULL_AND_PIECEWISE, the tokenizer mode, the context, the seed and
    DSpark with five speculative tokens; the sparse-MLA attention backend family on every rank). Runtime evidence, on the
    instrumented launch (`runtime_evidence.py`, run in its own group after the timed session): CUDA-graph replay of
    useful model kernels during warm C1 and C16 requests on all four ranks, real draft work with accepted draft tokens,
    and two distinct warm inputs whose generations agree with the server's own rescoring. A launch log or a runtime
    fact that is missing fails; nothing here depends on a log line's spelling.
    """
    from infb_verify import runtime_evidence as rt  # noqa: PLC0415

    record: dict = {"node": "test_served_configuration"}
    session_text = serve.LAUNCH_LOG.read_text(errors="replace") if serve.LAUNCH_LOG.exists() else ""
    instrumented_text = rt.LAUNCH_LOG.read_text(errors="replace") if rt.LAUNCH_LOG.exists() else ""
    record["configuration"] = {"session": serve.served_configuration(session_text),
                               "instrumented": serve.served_configuration(instrumented_text)}
    runtime = instrumented_server.observe()
    record["runtime"] = runtime
    problems = []
    for which, text in (("session", session_text), ("instrumented", instrumented_text)):
        report = record["configuration"][which]
        if not text:
            problems.append(f"the {which} launch log is missing or empty: a harness fault")
        elif not report["engine_config_lines"]:
            problems.append(f"no engine configuration line in the {which} launch log")
        problems += [f"{which} launch: {d}" for d in report["departures"]]
    problems += runtime["problems"]
    record["status"] = "pass" if not problems else ("blocked_by:" + runtime["blocked_by"] if runtime.get("blocked_by")
                                                    else "candidate_fail")
    rt.persist(record)
    print("INFB_SERVED_CONFIGURATION " + json.dumps({"configuration": {k: {"departures": v["departures"],
          "attention_backends": v["attention_backends"]} for k, v in record["configuration"].items()},
          "runtime": {k: runtime.get(k) for k in ("summary", "problems")}}, sort_keys=True, default=str))
    assert not problems, ((record["status"] + ": " if record["status"].startswith("blocked_by:") else "")
                          + "the deployment departs from the recipe or its runtime evidence is missing: "
                          + "; ".join(problems))
