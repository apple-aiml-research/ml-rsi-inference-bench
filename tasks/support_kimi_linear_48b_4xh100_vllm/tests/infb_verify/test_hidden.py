#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Served at the recipe: the maintainers' accuracy evaluation, and the behaviour probes.

Both run against ONE server launched at the recipe by the session fixture in conftest.py.

The candidate checks are the evaluation tree's own (`tests/acceptance.py`), loaded from the grading copy
`/tests/eval/kimi_linear/`, byte-identical to the supplied `/app/eval/kimi_linear/`; this module adds the
held-out inputs and the acceptance values.

`test_gsm8k_accuracy` runs **vLLM's own evaluator**, `tests/evals/gsm8k/gsm8k_eval.py`, vendored
byte-identically under `eval/kimi_linear/tests/upstream/`.
It is upstream's protocol unchanged -- all 1,319 test questions, 5-shot from the first five
`train.jsonl` rows, `max_tokens` 256, temperature 0, seed 42, the stop list
`["Question", "Assistant:", "<|separator|>"]`, last-number extraction -- against a server at **this
task's recipe** rather than under the evaluator's own pytest wrapper, which forces
`--enforce-eager --max-model-len 4096`. That substitution is the one adaptation. Nothing is hidden in GSM8K: the whole published set is run, and lookup is defended by
the fidelity gate, not by hiding questions.

`test_api_probes` is the API-correctness half of "serves correctly", kept minimal, with **batch
invariance** as the one mechanism probe. The probe inputs live in `/tests/hidden/probes.json`, which is
read at import time: a verifier image built without it fails at COLLECTION and takes the whole suite
with it, which is why the recipe's verifier tail asserts the file is non-empty.
"""

from __future__ import annotations

import json
import os
from importlib import util
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
CFG = json.loads((TESTS_DIR / "config.json").read_text())
X = CFG["infb"]
HIDDEN = json.loads((TESTS_DIR / "hidden" / "probes.json").read_text())
EVAL = TESTS_DIR / "eval/kimi_linear"


def _load(name: str, path: Path):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACCEPTANCE = _load("infb_kimi_acceptance", EVAL / "tests/acceptance.py")
#: The evaluator caches its two dataset files at /tmp/<basename> and returns early when they exist,
#: so seeding them there is the whole of running it offline. No edit to upstream's file. The directory
#: is the restored evaluation tree's (a tracked symlink onto the image's sha256-checked copies).
DATASET_DIR = Path(os.environ.get("INFB_GSM8K_DATASET_DIR", X["dataset_root"]))


# ── the paired efficiency gate, scored ──────────────────────────
# The two nodes are DEFINED FIRST in this module: pytest runs a module's functions in definition order,
# and the session fixture's server is launched for the first of them, so the candidate is measured on a
# fresh server, as the reference was by the reference phase earlier in this trial, on the same GPUs
# and the same hidden draw. `run_axis` measures the candidate with the grader's client in its own process
# and passes iff the record's verdict for that axis is `pass`: at least rho = 0.8 of the reference, in
# exact arithmetic; on the single-request axis, the best of up to c1_attempts draws.
import sys as _sys  # noqa: E402

if str(TESTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(TESTS_DIR))
from infb_efficiency.nodes import run_axis  # noqa: E402
from infb_verify import records  # noqa: E402

RECIPE = "test_hidden"


def _the_recipe_key_and_the_launch_argv_hold() -> None:
    """Checked at import, because a mismatch found at grading time costs a whole reference phase.

    1. the module's name is the recipe key the config declares;
    2. the reference's static launch argv is THIS task's recipe: the model, then serve.py's SERVER_ARGS in
       order, then the kit's port and seed. The reference and the candidate must be served alike, or
       the ratio measures the difference between two recipes."""
    from infb_verify.serve import SERVER_ARGS  # noqa: PLC0415

    recipes = (X.get("efficiency") or {}).get("recipes") or {}
    assert Path(__file__).stem == RECIPE and RECIPE in recipes, (
        f"module {Path(__file__).stem!r}, recipe key {RECIPE!r}, config recipes {sorted(recipes)}")
    argv = recipes[RECIPE]["launch"]["argv"]
    want = ["vllm", "serve", X["model_id"], *SERVER_ARGS]
    assert argv[: len(want)] == want and argv[len(want):] == ["--port", "31000", "--seed", "0"], (
        f"the reference's launch argv {argv} is not this task's recipe {want} + the kit's port and seed")


_the_recipe_key_and_the_launch_argv_hold()


def test_0_efficiency_decode_c1(server):
    """S1: per-user decode speed at concurrency 1, at least 0.8 of the reference's (best of up to 3 draws)."""
    ok, why, _ = run_axis(server.url_root, "decode_c1", RECIPE)
    assert ok, why


def test_0_efficiency_throughput(server):
    """S_C: output throughput at the recipe's load C = 64, at least 0.8 of the reference's (one draw)."""
    ok, why, _ = run_axis(server.url_root, "throughput", RECIPE)
    assert ok, why


def _kda_check(check: str, run, meta: dict):
    """A KDA node's verdict under a record: its problems, persisted before the assertion."""
    def body(record: dict):
        record["problems"] = problems = run()
        return not problems, problems

    problems = records.run(check, meta, body)
    assert not problems, problems


def test_kda_mathematics(server):
    from infb_verify import kda
    _kda_check("test_kda_mathematics", lambda: kda.check_math(CFG, "candidate", server.url_root),
               {"evidence": str(kda.RECORDS / "candidate-math.json"), "policy": "infb.kda.acceptance.kda"})


def test_kda_prefill_efficiency(server):
    from infb_verify import kda
    _kda_check("test_kda_prefill_efficiency", lambda: kda.check_timing(CFG, server.url_root, "prefill"),
               {"evidence": str(kda.RECORDS / "candidate-prefill.json"), "families": [0, 1, 2, 3]})


def test_kda_decode_efficiency(server):
    from infb_verify import kda
    _kda_check("test_kda_decode_efficiency", lambda: kda.check_timing(CFG, server.url_root, "decode"),
               {"evidence": str(kda.RECORDS / "candidate-decode.json"), "families": [4, 5, 6, 7]})


def test_gsm8k_accuracy(server):
    """Upstream's evaluator, upstream's protocol, this task's recipe."""
    port = int(server.url_for("v1/completions").split(":")[2].split("/")[0])

    def body(record: dict):
        result = ACCEPTANCE.accuracy(
            port,
            dataset_dir=DATASET_DIR,
            evaluator=EVAL / "tests/upstream/gsm8k_eval.py",
            num_questions=X["gsm8k"]["num_questions"],
            num_shots=X["gsm8k"]["num_shots"],
            max_tokens=X["gsm8k"]["max_tokens"],
            seed=X["gsm8k"]["seed"],
        )
        record["result"] = result
        return (result["accuracy"] >= X["gsm8k"]["floor"]
                and result["invalid_rate"] <= X["gsm8k"]["max_invalid_rate"]), result

    result = records.run("test_gsm8k_accuracy", {"protocol": X["gsm8k"]}, body)
    floor = X["gsm8k"]["floor"]
    assert result["accuracy"] >= floor, (
        f"GSM8K accuracy {result['accuracy']:.4f} is below the published floor {floor}; "
        f"invalid_rate {result['invalid_rate']:.4f}, {result['num_questions']} questions in "
        f"{result['latency']:.1f} s"
    )
    # A high invalid rate with a passing score would mean the scorer is reading numbers out of
    # something other than an answer, which is the finding this protocol exists to avoid.
    assert result["invalid_rate"] <= X["gsm8k"]["max_invalid_rate"], result


def test_api_probes(server):
    """Serving correctly, in the smallest set of checks that can tell it apart from not.

    Batch invariance is the mechanism probe: the same greedy prompts, served one at a time and then
    concurrently in groups whose sizes are NOT CUDA-graph capture sizes, must give the same tokens.
    At this recipe the capture sizes are 1, 2, 4, then 8-248 in steps of 8, then 256-512 in steps of
    16, so a batch of 3 or 5 is padded and the padded rows reach the recurrent kernel -- the exact
    path the post-landing crash report is about. Token ids are compared, never decoded text: it was
    measured that text comparison passes a fault which log-probabilities reject.

    Exact token equality between a request served alone and the same request served in a batch is NOT
    what a correct vLLM server guarantees: batch composition changes reduction order, and greedy
    decoding then flips wherever two tokens are nearly tied. What a correct server DOES guarantee is that every
    divergence is a near-tie: at the first position where the two runs differ, each run's token is in
    the other's top-5, within NEAR_TIE_NATS of the other's top-1. A broken batch -- state leaking between requests, padded rows written into real
    ones, an unwritten buffer -- diverges by many nats or outside the top-5, and fails here.
    """
    def body(record: dict):
        record["batch"] = ACCEPTANCE.api_probes(server.url_for, HIDDEN, model=X["served_model_name"], top_k=TOP_K,
                                                near_tie_nats=NEAR_TIE_NATS)
        return True, record["batch"]

    records.run("test_api_probes", {"top_k": TOP_K, "near_tie_nats": NEAR_TIE_NATS}, body)


def test_recipe_conformance(server):
    """The graded launch serves the recipe as the instruction states it (instruction item 5).
    Read from the graded server's own output (`serve.SERVER_LOG`), and from every rank's own lines
    where the engine prints per rank:
      * the arguments the recipe names, as the server parsed them: tensor-parallel size 4, max-model-len
        1,048,576, trust-remote-code and long-prefill-token-threshold 2,048, and nothing beyond the harness's own
        model, port, seed and the declared verification worker extension;
      * the defaults the instruction says stay, as the engine configured itself: one pipeline stage and one
        data-parallel rank, no quantization, not eager, BF16 weights and KV cache, prefix caching off, chunked
        prefill on with max_num_batched_tokens 8,192, compilation mode VLLM_COMPILE with CUDA graphs in
        FULL_AND_PIECEWISE mode, and FlashMLA for the MLA layers on all four ranks;
      * and the model compiled: the compilation monitor's `torch.compile takes ... s in total`, which the engine
        prints only when it compiled the model. A model the engine cannot compile (no `@support_torch_compile`)
        still shows the default mode in its configuration line, serves decode steps in FULL graphs and runs
        every mixed batch eagerly; one agent's served tree was such a model, at 0.868 and 0.925 of
        the reference.
    The reference passes by construction: it is the reference implementation at the recipe. A reference tree
    without the compile decorator does not conform, which is what this check reads.
    """
    from infb_verify import serve

    assert serve.SERVER_LOG.is_file(), f"the graded server's output was not captured at {serve.SERVER_LOG}"
    def body(record: dict):
        problems, summary = ACCEPTANCE.recipe_problems(serve.SERVER_LOG.read_text(errors="replace"),
                                                       max_model_len=X["max_model_len"])
        record.update(problems=problems, summary=summary)
        return not problems, (problems, summary)

    problems, summary = records.run("test_recipe_conformance", {"log": str(serve.SERVER_LOG)}, body)
    assert not problems, "the graded server does not serve the recipe as stated:\n  " + "\n  ".join(problems)
    print("INFB_RECIPE " + json.dumps(summary))


CONCURRENCY = json.loads((TESTS_DIR / "hidden" / "concurrency-prompts.json").read_text())


def test_concurrent_state_isolation(server):
    """Concurrent requests are isolated from one another (instruction item 1).

    Sixteen hidden natural-text prompts are each decoded greedily for 256 tokens, first one at a time and then all
    sixteen at once. At once, the scheduler mixes their prefill chunks and decode steps in the same batches, so every
    KDA layer carries sixteen recurrent states side by side and each request must read and write only its own. The
    reading of one prompt is the probes' gap (test_api_probes step 5) at the first position where its two answers
    differ, from the top-20 log-probabilities, the most the recipe returns: how far each answer's token is from the
    other answer's top-1, the larger of the two. Equal answers read 0, and a token outside the other's top-20 reads
    infinity. The verdict is taken over two bands: **inside**, eight prompts of 1,024 to 2,000
    tokens, each within one 2,048-token prefill chunk; and **crossing**, eight of 3,000 to 12,000, each crossing chunk
    boundaries. Each band's median must be below its bar, T_b = max(2.0, 3 x the oracle's band median over its
    draws); 2.0 is the probes' near-tie bar (config.json `concurrency.per_band`). A band that fails is re-drawn once,
    the whole batch sent at once again, and fails only if both draws do; both are printed.
    """
    cfg = X["concurrency"]
    bars = cfg["T_per_band"]
    prompts = CONCURRENCY["prompts"]
    by_band = {band: [p for p in prompts if p["band"] == band] for band in bars}
    assert len(prompts) == cfg["n_prompts"] and all(len(v) == cfg["n_prompts"] // len(bars) for v in by_band.values()), (
        f"the concurrency set has {[(b, len(v)) for b, v in by_band.items()]} prompts by band, not "
        f"{cfg['n_prompts'] // len(bars)} in each of {sorted(bars)}"
    )
    def body(record: dict):
        out = ACCEPTANCE.concurrency(server.url_for, prompts, list(bars), model=X["served_model_name"],
                                     max_tokens=cfg["max_tokens"], top_k=cfg["top_k"], bars=bars)
        record.update(draws=out["draws"], passed=out["passed"])
        return all(out["passed"].values()), out

    out = records.run("test_concurrent_state_isolation", {"T_per_band": bars, "top_k": cfg["top_k"]}, body)
    draws, passed = out["draws"], out["passed"]
    print("INFB_CONCURRENCY " + json.dumps({"T_per_band": bars, "top_k": cfg["top_k"], "passed": passed,
                                           "band_medians": [d["medians"] for d in draws], "draws": draws}))
    failures = []
    for band in bars:
        if not passed[band]:
            readings = " then ".join(format(d["medians"][band], ".6g") for d in draws)
            worst = sorted((r["gap"], i) for d in draws for i, r in d["rows"].items()
                           if i in {p["id"] for p in by_band[band]})[-3:]
            failures.append(f"the {band} band's median gap is {readings} nats, not below {bars[band]:.6g} "
                            f"(its largest readings: {worst})")
    assert not failures, (
        "requests served at once do not give the answers they give alone, beyond a correct server's near-ties: "
        + "; ".join(failures)
    )


NEAR_TIE_NATS = X["api_probes"]["near_tie_nats"]
TOP_K = X["api_probes"]["top_k"]
