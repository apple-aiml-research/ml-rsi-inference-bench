#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The served-API nodes — behaviour probes, and the prefix-reuse pair.

All of these run against **one** server launched at the recipe by `serve.launch()` — the
session-scoped fixture in `conftest.py`, shared with the fidelity module, so the two cost one extra
launch per trial between them beside the frozen accuracy test's own.

Prefix caching is **on by default** at this recipe (`CacheConfig.enable_prefix_caching` is True, the
hybrid model reports `is_prefix_caching_supported()`, and `MambaModelConfig.verify_and_update_config`
switches `mamba_cache_mode` to `"align"`), so the maintainers' own accuracy run already exercised
reuse through its shared few-shot prefix. What is added here is **observation**:

  * `test_prefix_reuse_witness` — cache hits actually occur on a repeated long prefix, read from
    `/metrics` rather than inferred;
  * `test_prefix_reuse_agreement` — generation from a **warm** prefix agrees with the same
    generation served **cold**, the scored token sitting 2 to 8 tokens past the cache boundary.

**Why the agreement node reads generated-token log-probabilities rather than `prompt_logprobs`.**
`SamplingParams.__post_init__` sets `skip_reading_prefix_cache = prompt_logprobs is not None`, and
`KVCacheManager.get_computed_blocks` then returns no hit, so a teacher-forced probe never reads the
cache and cannot carry this case at all. Teacher-forced fidelity is measured on the fresh path;
reuse is measured here, through the log-probabilities of tokens the model generates.

"Cold" is a copy of the same request carrying a distinct `cache_salt` — a field of both the chat and
completion protocols at the reference.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from infb_verify import serve, served

TESTS_DIR_ = Path(os.environ.get("TESTS_DIR", "/tests"))
for _path in (str(TESTS_DIR_ / "eval"), str(TESTS_DIR_)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from qwen38.measurement import budget  # noqa: E402
from qwen38.measurement import observations as obs  # noqa: E402
from qwen38.serving import api, fidelity, mtp  # noqa: E402
from infb_verify.qwen38_private import stateful  # noqa: E402

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
BARS = X.get("bars", {})
CASES = json.loads((TESTS_DIR / "hidden" / "prefix_reuse.json").read_text())["cases"]
WITNESS = json.loads((TESTS_DIR / "hidden" / "prefix_witness.json").read_text())
FIDELITY = json.loads((TESTS_DIR / "hidden" / "fidelity.json").read_text())["sequences"]

MODEL = X["model_id"]

# The two scored efficiency nodes. `/tests` on sys.path is how the efficiency package is reached; its
# own modules put their package directory on the path themselves.
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from infb_efficiency.nodes import run_axis  # noqa: E402

#: The key of this task's one scored recipe in `config.json` `infb.efficiency.recipes`. A recipe is keyed
#: by the last component of the junit classname of the nodes that score it. Every other Day-0 task mixes
#: `EfficiencyMixin` into a registered upstream CLASS and is keyed by that hidden class's name; this task
#: has no class to mix into, because upstream's acceptance for this model is the plain module-level
#: function `tests/evals/qwen4_exp/test_accuracy.py::test_qwen4_exp_accuracy[Qwen3.8-Flash-Next-FP8]`,
#: which launches through `tests/utils.py`'s RemoteOpenAIServer inside its own body. So the two nodes are
#: module-level functions here and the key is this module's name -- the form `nodes.py` provides for
#: exactly this case.
RECIPE = "test_hidden"


def _module_name_matches_the_recipe_key() -> None:
    """A name mismatch here would make the nodes look for a recipe that does not exist, at grading
    time, after the reference has already been served for an hour. Checked at import instead."""
    assert Path(__file__).stem == RECIPE, (
        f"this module is {Path(__file__).stem!r} but RECIPE is {RECIPE!r}: the kit keys a recipe by the "
        f"last component of the scoring nodes' junit classname, which is this module's name"
    )
    assert RECIPE in (X.get("efficiency", {}).get("recipes") or {}), (
        f"config.json infb.efficiency.recipes has no entry {RECIPE!r}"
    )


_module_name_matches_the_recipe_key()


def _client(server):
    return server.get_client()


# The paired efficiency gate.
#
# These two run FIRST in this session -- `test_0_` sorts before every other name here, and the session
# is the first graded step after the reference phase -- so the candidate is measured on a server whose
# GPUs have seen the same history the reference's did, rather than one that follows an hour-long
# accuracy evaluation.
#
# The reference's half of each comparison was measured earlier in this same trial, by the reference
# phase, on the same GPUs and the same hidden draw; `run_axis` measures the candidate with the same
# client in its own process and passes iff the record's verdict for that axis is `pass` -- the
# candidate's value at least rho = 0.8 of the reference's, in exact arithmetic.


def test_0_efficiency_decode_c1(server):
    """S1: per-user decode speed at concurrency 1, at least 0.8 of the reference's."""
    ok, why, _ = run_axis(server.url_root, "decode_c1", RECIPE)
    assert ok, why


def test_0_efficiency_throughput(server):
    """S_C: output throughput at the recipe's load C = 32, at least 0.8 of the reference's.

    C is EvalScope's own `eval_batch_size` from the frozen acceptance config, which is the concurrency
    the maintainers' own accuracy run uses.
    """
    ok, why, _ = run_axis(server.url_root, "throughput", RECIPE)
    assert ok, why




def test_served_configuration(server):
    """The server runs at the configuration the instruction states, read from every rank's launch log."""
    text = serve.LAUNCH_LOG.read_text(errors="replace") if serve.LAUNCH_LOG.exists() else ""
    report = serve.served_configuration(text)
    print("INFB_SERVED_CONFIGURATION " + json.dumps(report, sort_keys=True))
    assert text, f"the session server's launch log {serve.LAUNCH_LOG} is missing or empty: a harness fault"
    assert report["engine_config_lines"], (
        "no engine configuration line ('Initializing a V1 LLM engine … with config:') is in the launch "
        "log, so the served configuration cannot be read"
    )
    assert not report["departures"], (
        "the served configuration departs from the one the instruction states: "
        + "; ".join(report["departures"])
    )


def test_api_probes(server):
    """Four cheap properties of a correct server, each one a way a wrong one shows itself: two
    identical greedy chat requests agree, streaming agrees with non-streaming, the length cap
    reports itself, and eight concurrent identical requests give the serial first token."""
    result = served.api_probes(server, "ordinary")
    assert result["holds"], {k: result[k] for k in ("repeat", "stream", "length", "isolation")}






def _case_ids() -> list[list[int]]:
    """Every agreement case's prefix as the checkpoint's own token ids, each its recorded length."""
    from tokenizers import Tokenizer  # noqa: PLC0415  (the kit's client reads the tokenizer the same way)

    tok = Tokenizer.from_file(X["efficiency"]["tokenizer"])
    ids = [tok.encode(c["prefix"], add_special_tokens=False).ids for c in CASES]
    wrong = [(c["id"], len(i), c["realized_tokens"]) for c, i in zip(CASES, ids) if len(i) != c["realized_tokens"]]
    assert not wrong, f"the tokenizer does not reproduce the cases' recorded lengths (id, got, recorded): {wrong}"
    return ids


def _boundary_cut(server, ids: list[list[int]], i: int) -> tuple[list[int], int]:
    """Case i as the served cache sees it: (the prompt, the boundary it restores from).

    The case's prefix is sent, then sent again: the repeat's prefix-cache hits are the tokens it read from the cache,
    the deepest block boundary before its end, whatever the block size. Align-mode registration is not always one
    request deep, so up to two repeats. The prompt is that boundary plus the case's tail (2 to 8 tokens), so its
    first generated token sits inside the PLE convolution's ten-token field past a boundary the cache really restores.
    A prefix no repeat reads from is doubled, continuing into the following cases' prefixes, since a block may be
    longer than the case; if even all eight together are never read from, the node fails, saying so."""
    case = CASES[i]
    stream = [t for j in range(len(ids)) for t in ids[(i + j) % len(ids)]]
    n = case["realized_tokens"]
    while True:
        probe = stream[:n]
        _complete(server, probe, salt=None, n=1, context=f"{case['id']}/prepare")
        for attempt in range(2):
            before = stateful.cache_reading(server.url_root)
            _complete(server, probe, salt=None, n=1, context=f"{case['id']}/repeat-{attempt}")
            hit = stateful.cache_delta(before, stateful.cache_reading(server.url_root))["hits"]
            if hit > 0:
                assert hit + case["tail_tokens"] <= len(stream), f"{case['id']}: a boundary at {hit:g} leaves no tail"
                return stream[: int(hit) + case["tail_tokens"]], int(hit)
        assert n < len(stream), (
            f"{case['id']}: no repeat of up to {n} tokens read anything from the prefix cache, so there is no cache "
            f"boundary for this node to test at"
        )
        n = min(len(stream), 2 * n)


def _complete(server, prompt: str | list[int], *, salt: str | None, n: int = 64, forced: bool = False,
              context: str = "completion") -> dict:
    """A validated greedy completion with top-five probabilities, as integer token IDs.

    `forced` keeps generating for exactly `n` tokens. Without it the two sides can stop at different lengths and
    there is nothing to compare; `ignore_eos` removes it. Every request's timeout is what remains of its
    operation's allowance.
    """
    body = {"model": MODEL, "prompt": prompt, "temperature": 0.0, "max_tokens": n, "seed": 0, "logprobs": 5,
            "return_tokens_as_token_ids": True}
    if salt:
        body["cache_salt"] = salt
    if forced:
        body["ignore_eos"] = True
    response = mtp.request(server.url_root, "/v1/completions", body, timeout=600)
    return obs.completion(response, context, requested=n, ignore_eos=forced, vocabulary=stateful.vocabulary(),
                          prompt_tokens=len(prompt) if isinstance(prompt, list) else None)


def test_prefix_reuse_witness(server):
    """Cache hits actually occur when a long prefix is repeated.

    Hit registration on this model's align-mode cache is not one request deep, so each prefix is sent three
    times and the counter delta is read across the whole repeat. Every prefix is at least
    `WITNESS["min_tokens"]` (4,096) tokens, so any cache block of up to 2,048 tokens is covered at least twice.
    """
    assert all(c["realized_tokens"] >= WITNESS["min_tokens"] for c in WITNESS["cases"])
    observed = []
    with budget.operation("prefix_reuse_witness"):
        for case in WITNESS["cases"]:
            before = stateful.cache_reading(server.url_root)
            for attempt in range(3):
                _complete(server, case["prefix"], salt=None, n=1, context=f"{case['id']}/{attempt}")
            delta = stateful.cache_delta(before, stateful.cache_reading(server.url_root))
            observed.append({"id": case["id"], "tokens": case["realized_tokens"],
                             "hits": delta["hits"], "queries": delta["queries"]})
    print("INFB_REUSE_WITNESS " + json.dumps(observed))
    silent = [o for o in observed if o["hits"] <= 0]
    assert not silent, (
        "no prefix-cache hit was registered for "
        + ", ".join(f"{o['id']} ({o['tokens']} tokens, {o['queries']} queries)" for o in silent)
        + " — a server that never reuses a repeated prefix is not serving this recipe, whose "
        "cache is on by default in align mode"
    )


def test_prefix_reuse_agreement(server):
    """Generation from a WARM prefix agrees with the same generation served COLD.

    This is the node that rejects a solution which is only correct on fresh requests: the PLE
    convolution state and the two-token n-gram context have to be restored from the aligned block on
    a hit, and the QSA side caches have to line up, or the warm generation drifts from the cold one.

    **The case lengths are the whole design.** vLLM reuses whole 784-token pages, so a prefix of L
    tokens takes 784*floor(L/784) from the cache and recomputes `L mod 784`. The PLE stream state is a
    depthwise convolution of kernel 4 and dilation 3 -- a ten-token receptive field -- so a wrong
    restored state hundreds of tokens behind the scored position washes out. The cases are therefore
    `784 * pages + tail` with `tail` between 2 and 8, so the first generated token lies inside that
    window, and each case's boundary is read from the cache itself (`_boundary_cut`) so a tree with
    another page size is measured at its own boundary. The warm request must itself read that boundary
    from the cache, or the node fails saying so.

    The two sides are the same request; only `cache_salt` differs, so the cold twin cannot hit.
    """
    t2 = BARS.get("prefix_reuse_T2")
    ids = _case_ids()
    per_case = []
    with budget.operation("prefix_agreement"):
        for i, case in enumerate(CASES):
            prompt, boundary = _boundary_cut(server, ids, i)  # the prefix is warm from here
            before = stateful.cache_reading(server.url_root)
            warm = _complete(server, prompt, salt=None, forced=True, context=f"{case['id']}/warm")
            hits = stateful.cache_delta(before, stateful.cache_reading(server.url_root))["hits"]
            cold = _complete(server, prompt, salt=f"infb-cold-{case['id']}", forced=True, context=f"{case['id']}/cold")
            per_case.append(_compare(case["id"], warm, cold)
                            | {"tokens": len(prompt), "boundary": boundary, "warm_hits": hits})
    print("INFB_REUSE_AGREEMENT " + json.dumps(per_case, allow_nan=False))

    cold_warm = [c for c in per_case if c["warm_hits"] != c["boundary"]]
    assert not cold_warm, (
        "the warm request did not read its prefix from the cache up to the boundary a repeat of the same prefix had "
        "just read, so its comparison would be cold against cold: "
        + ", ".join(f"{c['id']} (boundary {c['boundary']}, warm read {c['warm_hits']:g})" for c in cold_warm)
    )
    undefined = [c for c in per_case if c["pos0_terms"] == 0]
    assert not undefined, (
        "neither side's chosen token at position 0 appears in the other's top-5 for "
        + ", ".join(c["id"] for c in undefined)
        + " — the prompts are identical there, so that is not a tie-break"
    )
    if t2 is None:
        pytest.skip("prefix_reuse_T2 is not published yet; the statistic is printed above")
    median = statistics.median(obs.k3_value(c["k3_pos0"]) for c in per_case)
    assert median <= t2, (
        f"warm-versus-cold k3 median at position 0 is {median:.6g}, over T2 {t2:.6g}: {per_case}"
    )


def _compare(case_id: str, warm: dict, cold: dict) -> dict:
    """k3 at position 0, on the tokens each side actually CHOSE (validated records).

    Three properties this has to have, each learned by measuring the reference rather than by design:

    * **Comparable.** Position 0 is the only position where the warm and cold prompts are identical --
      one served from the cache, one recomputed. From position 1 the two are conditioned on whatever
      each emitted, so once the argmax flips at a near-tie they are decoding different texts.
    * **Always defined.** Averaging k3 over the agreed leading run returned `inf` for the case whose
      argmax flipped at position 0.
    * **On the head, not the tail.** k3 over the tokens two top-5 lists happen to share measures the
      instability of a tail, not the agreement of a model.

    So the statistic is k3 on each side's own argmax at position 0, evaluated under both
    distributions, averaged over the two. Both terms are available whenever the top-5 rule holds,
    which is asserted separately. An exponent overflow is a numerical +infinity, never clipped.
    """
    n = min(len(warm["tokens"]), len(cold["tokens"]))
    agreed, first_div, violation = 0, None, False
    for i in range(n):
        if warm["tokens"][i] == cold["tokens"][i]:
            agreed += 1
            continue
        first_div = i
        violation = not obs.near_tie(warm, cold, i)
        break
    w0 = dict(warm["top_logprobs"][0]) | {warm["tokens"][0]: warm["token_logprobs"][0]}
    c0 = dict(cold["top_logprobs"][0]) | {cold["tokens"][0]: cold["token_logprobs"][0]}
    terms = [obs.k3(c0[tok] - w0[tok]) for tok in {warm["tokens"][0], cold["tokens"][0]} if tok in w0 and tok in c0]
    values = [obs.k3_value(t) for t in terms]
    mean = math.fsum(v / len(values) for v in values) if values and all(math.isfinite(v) for v in values) else (
        math.inf if values else math.nan)
    return {"id": case_id, "compared": n, "agreed_run": agreed, "pos0_terms": len(terms),
            "k3_terms": terms, "k3_pos0": obs.json_number(mean) if values else None,
            "first_divergence": first_div, "top5_violation": violation}


# Fresh-request state.
#
# A request's output must depend on its own tokens only. The base engine did not meet that for a
# fresh one-token prompt: its shared GDN metadata builder counted it as a decode, and the decode
# kernels read the slot's recurrent and convolution state as they stood -- another request's, or
# whatever start-up left there. Upstream repaired it in #51565 (`79468c20e`, merged 2026-09-22,
# inside the landing's four-week window). This node is the obligation's check. It reads outcomes only:
#   * a one-token prompt served alone, after unrelated requests have run and finished;
#   * the same prompt again after more of them, whose continuation must agree with the first, by the
#     probes' near-tie rule;
#   * a one-token prompt arriving while sixteen unrelated requests decode.
# The statistic is the fidelity node's k3, decode against prefill: each generated token's
# log-probability as the server produced it, against the log-probability the same server assigns that
# token when the prompt and the generated tokens are sent back as one prefill (`prompt_logprobs`,
# which never reads the prefix cache).

FRESH = json.loads((TESTS_DIR / "hidden" / "fresh_state.json").read_text())
FRESH_BAR = BARS["fresh_state"]


def _fresh_turnover_prompts() -> tuple[list[list[int]], list[list[int]]]:
    """Unrelated prompts, from the hidden fidelity set: 16 turnover prompts of 1,024 tokens, and 16 background
    prompts of 200 tokens cut from the same sequences just after them."""
    by_id = {s["id"]: s for s in FIDELITY}
    seqs = [by_id[i]["token_ids"] for i in FRESH["turnover_sequences"]]
    n, m = FRESH["turnover_tokens"], FRESH["background_tokens"]
    return [s[:n] for s in seqs], [s[n:n + m] for s in seqs]


def _fresh_generate(server, ids: list[int], n: int, context: str) -> dict:
    """A validated greedy fixed-length continuation with each position's five most likely tokens."""
    return stateful.generate(server.url_root, MODEL, ids, n, case_id=context)


def _fresh_k3(server, ids: list[int], gen: dict, context: str) -> float:
    """The mean over generated positions of k3(prefill - decode), from validated records on both sides;
    an overflow is +infinity."""
    pre = fidelity.prompt_logprobs(server.url_root, MODEL, ids + gen["tokens"], vocabulary=stateful.vocabulary())
    ref = pre[len(ids) - 1: len(ids) - 1 + len(gen["tokens"])]
    result = obs.k3_mean(ref, gen["token_logprobs"], context)
    return math.inf if result["overflow"] else result["mean"]


def _fresh_load(server, prompt: list[int], n: int, *, ignore_eos: bool, context: str) -> dict:
    """One unrelated request, validated like every other observation (it is load, not a statistic)."""
    body = {"model": MODEL, "prompt": prompt, "max_tokens": n, "temperature": 0.0, "seed": 0, "logprobs": 1,
            "return_tokens_as_token_ids": True, "cache_salt": f"infb-fresh-load-{uuid.uuid4().hex}"}
    if ignore_eos:
        body["ignore_eos"] = True
    response = mtp.request(server.url_root, "/v1/completions", body, timeout=600)
    return obs.completion(response, context, requested=n, ignore_eos=ignore_eos, top_k=1,
                          vocabulary=stateful.vocabulary(), prompt_tokens=len(prompt))


def _fresh_turnover(server, prompts: list[list[int]]) -> None:
    """Unrelated requests that run and finish, so freed state slots are handed out again."""
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        list(pool.map(lambda item: _fresh_load(server, item[1], FRESH["turnover_max_tokens"], ignore_eos=False,
                                               context=f"turnover-{item[0]}"), enumerate(prompts)))


def _fresh_phase(server, name: str, run_one) -> dict:
    """One condition over every witness, drawn again once whole if its median is at or over the bar."""
    draws = []
    for _ in range(2):
        rows = [dict(run_one(w), witness=w["id"]) for w in FRESH["witnesses"]]
        med = statistics.median(r["k3"] for r in rows)
        draws.append({"median": obs.json_number(med), "max": obs.json_number(max(r["k3"] for r in rows)),
                      "k3": [obs.json_number(r["k3"]) for r in rows]})
        if med < FRESH_BAR["T"]:
            break
    return {"phase": name, "draws": draws, "passed": obs.k3_value(draws[-1]["median"]) < FRESH_BAR["T"]}


def test_fresh_request_state(server):
    """A fresh request is computed from its own tokens: a one-token prompt served alone after unrelated
    requests, again after more of them (the same continuation, up to a near-tie), and while sixteen others
    decode, each agrees with the same server's prefill of the same tokens (k3 below the bar)."""
    turnover, background = _fresh_turnover_prompts()
    n = FRESH["generated_tokens"]
    first: dict[int, dict] = {}

    def alone(w):
        _fresh_turnover(server, turnover)
        g = _fresh_generate(server, [w["id"]], n, f"alone-{w['id']}")
        first.setdefault(w["id"], g)
        return {"k3": _fresh_k3(server, [w["id"]], g, f"alone-{w['id']}")}

    def joining(w):
        _fresh_turnover(server, turnover)
        with ThreadPoolExecutor(max_workers=len(background)) as pool:
            futs = [pool.submit(_fresh_load, server, p, FRESH["background_max_tokens"], ignore_eos=True,
                                context=f"background-{i}") for i, p in enumerate(background)]
            time.sleep(FRESH["join_after_s"])
            g = _fresh_generate(server, [w["id"]], n, f"joining-{w['id']}")
            for f in futs:
                f.result()
        return {"k3": _fresh_k3(server, [w["id"]], g, f"joining-{w['id']}")}

    with budget.operation("fresh_state_evaluation"):
        a = _fresh_phase(server, "alone", alone)
        # the slot's history: the same prompt again, after more unrelated requests, against its first continuation
        repeats = []
        for _ in range(2):
            rows = []
            for w in FRESH["witnesses"]:
                _fresh_turnover(server, turnover)
                g = _fresh_generate(server, [w["id"]], n, f"repeat-{w['id']}")
                agreement = api.agreement(first[w["id"]], g)
                rows.append({"witness": w["id"], "divergence": agreement, "consistent": agreement["near_tie"]})
            repeats.append(rows)
            if all(r["consistent"] for r in rows):
                break
        b = {"phase": "repeat", "draws": repeats, "passed": all(r["consistent"] for r in repeats[-1])}
        c = _fresh_phase(server, "joining", joining)

    record = {"T": FRESH_BAR["T"], "generated_tokens": n, "alone": a,
              "repeat": {"consistent": [sum(r["consistent"] for r in d) for d in repeats],
                         "divergences": [[r["divergence"] for r in d] for d in repeats], "passed": b["passed"]},
              "joining": c}
    print("INFB_FRESH_STATE " + json.dumps(record, allow_nan=False))
    failed = [p for p, ok in (("alone", a["passed"]), ("repeat", b["passed"]), ("joining", c["passed"])) if not ok]
    assert not failed, (
        f"a fresh request is not computed from its own tokens alone ({', '.join(failed)}): a one-token prompt's "
        f"decoded log-probabilities depart from the same server's prefill of the same tokens (median k3 over the "
        f"bar {FRESH_BAR['T']}), or its continuation changes with what the server served before it. "
        f"{json.dumps(record, allow_nan=False)[:2500]}"
    )




def test_greedy_series(server):
    """Diagnostic, not scored: this tree's greedy continuations of the MTP class's sixteen prompts WITHOUT
    speculation, for `test_mtp.test_mtp_equivalence` to compare its MTP session against. Written to
    `mtp_series.SERIES_FILE`, which is removed first so nothing earlier can stand in for it; a request that fails
    is recorded there and fails that node's structural validation, not this one."""
    from infb_verify import mtp_series  # noqa: PLC0415

    mtp_series.SERIES_FILE.unlink(missing_ok=True)
    with budget.operation("comparison_series_recording"):
        s = mtp_series.series(server.url_root, int(BARS["mtp"]["series_tokens"]))
    mtp_series.SERIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    mtp_series.SERIES_FILE.write_text(json.dumps(s))
    errors = [c["id"] for c in s["cases"] if c.get("error")]
    print("INFB_GREEDY_SERIES " + json.dumps({"cases": len(s["cases"]), "errors": errors,
                                              "seconds": [c.get("seconds") for c in s["cases"]]}))


from infb_verify import expanded  # noqa: E402


@pytest.fixture(scope='session')
def component_records(server):
    return expanded.components(server, 'ordinary')


def test_qsa_mathematics_and_state(component_records):
    expanded.mathematics(component_records, 'qsa')


def test_qsa_prefill_efficiency(component_records):
    expanded.speed(component_records, 'qsa', 'prefill')


def test_qsa_decode_efficiency(component_records):
    expanded.speed(component_records, 'qsa', 'decode')


def test_ple_mathematics_and_state(component_records):
    expanded.mathematics(component_records, 'ple')


def test_ple_prefill_efficiency(component_records):
    expanded.speed(component_records, 'ple', 'prefill')


def test_ple_decode_efficiency(component_records):
    expanded.speed(component_records, 'ple', 'decode')


def test_ple_workspace_memory(component_records):
    expanded.memory(component_records)


def test_native_component_binding(component_records):
    expanded.native_binding(component_records)
