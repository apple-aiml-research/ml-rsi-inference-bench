#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Generated-history replay for DeepSeek-V4-Flash on SGLang: a greedy generation's output log-probabilities compared,
token by token, with cold teacher-forced replays of the same history. The protocol and the metrics the grader uses,
shared byte for byte with it.

    python /app/eval/deepseek_v4_flash/fidelity/generated_replay.py --base-url http://127.0.0.1:30000 \
        --example fresh --output /tmp/replay.json          # or --example prefix

A serving engine is not bit-reproducible: two identical cold prefills of the same history, a minute apart in the same
server, disagree at contested tokens by as much as a generation disagrees with its replay. So the history is replayed
**R = 3 times**, and the generation is judged against the spread the engine itself produces, not against one sample.

For one observation with prompt P and N generated tokens:

  1. GENERATE: `/generate` with `input_ids` P, greedy (`temperature` 0), `max_new_tokens` N, `ignore_eos`, and
     `return_logprob`. The answer must carry exactly N `output_ids`, and exactly one `[logprob, token_id, text]` entry
     of `meta_info.output_token_logprobs` per generated token, carrying that token's id and a finite log-probability.
     Every token id is a non-negative integer, a boolean is not (`True == 1` numerically). The engine's own counts
     (`completion_tokens`, `cached_tokens`, `spec_verify_ct`, `spec_accepted_drafts`) are recorded as it reports them.
  2. Requests generated together (a concurrent batch, or the warm requests of a prefix series) all finish first.
  3. FLUSH: `POST /flush_cache` must succeed, so the replay is cold.
  4. REPLAY, R = 3 times: the fidelity call (`fidelity.py`) on P + the generated ids with `max_new_tokens` 0 and
     `logprob_start_len` = len(P) - 1, parsed by `fidelity.parse_input_logprobs`: the entry for the prompt's last
     position carries no value, then exactly N finite values for the N generated positions, with the same ids in order.
     A missing, shifted, duplicated, null or non-finite entry is a structural failure before any number is formed.
     Each pass is preceded by its own flush, and every history of the group is replayed in the same order each pass, so
     every pass is cold and the passes are comparable.
  5. COMPARE, per generated token j, in float64. With g the generated log-probability and r1, r2, r3 the three replays:
       m    = median(r1, r2, r3);  lo = min(r1, r2, r3);  hi = max(r1, r2, r3)
       e    = max(0, lo - g, g - hi)      the excess: how far g lies OUTSIDE the engine's own replay range
       d    = r1 - g                      the first replay's difference, reported
       k3   = expm1(d) - d                reported
       the token is DECISIVE when exp(g) >= 0.9 (at temperature 0, g is the generation's own top-1 log-probability)
  6. THE FOUR STATISTICS, per node and run, pooled over every generated token of the node's whole schedule:
       S1' = median over decisive tokens of |g - m|,  divided by  median over decisive tokens of |r2 - r1|
             (the denominator is floored at 1e-6 nats). A correct run reads about 1: the generation sits in the
             replays' own spread. A state fault moves the generation and not the cold replays, so the ratio grows.
       S2' = the number of tokens with e >= 1 nat.
       S3' = the largest e.
       S4  = the decisive tokens' share of all the node's tokens, a guard so that S1' cannot be emptied.
     Nothing is trimmed and there is no best-of retry. Both sides of every comparison come from the same server, host
     and driver within the same minute, so a host's or a driver's own nondeterminism enters both and cancels.

The candidate's own generated history is what is replayed: the generated text need not equal anyone else's. The
comparison is between two engine paths over the same history -- speculative, cached, graph-replayed generation against
a cold prefill -- which is consistency, not independent model truth; `check.py` and the kernel checks supply that.
The supplied examples are one 2,048-token prompt (`fresh`) and one 2,048-token prefix shared by two continuations
(`prefix`: flushed once, both warm generations reuse the prefix, then each is replayed cold R times), 64 generated
tokens each. The command reports the metrics and every token's values; it applies no pass threshold, and passing these
examples does not establish acceptance on held-out histories. Each request is one operation with `fidelity.py`'s limit;
a flush is an action (`../limits.json`).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent

#: How many cold teacher-forced replays of each history the protocol takes (step 4). The generation is judged against
#: the spread of these, never against one of them: a serving engine's repeated prefills of the same history disagree
#: at contested tokens by as much as the generation disagrees with a replay.
REPLAYS = 3
#: A generated token is decisive when its own probability is at least this (step 5). At temperature 0 the generated
#: log-probability is the top-1's, so this selects the positions where the model is not choosing between near-ties.
DECISIVE_P = 0.9
#: S2' counts the tokens whose excess over the replays' range reaches this many nats (step 6).
EXCESS_NAT = 1.0
#: S1's denominator floor, in nats: a host whose repeated replays agreed exactly on most decisive tokens would
#: otherwise divide by zero and fail a correct run.
DENOMINATOR_FLOOR = 1e-6


def _fidelity():
    """The sibling fidelity core, loaded by path under the grader's module name (one instance in a process)."""
    name = "deepseek_v4_flash_eval_fidelity"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "fidelity.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


fidelity = _fidelity()
StructuralFailure = fidelity.StructuralFailure
ServedRequestFailed = fidelity.ServedRequestFailed
ACTION_LIMIT_S = fidelity.ACTION_LIMIT_S
_is_token_id = fidelity._is_token_id  # noqa: SLF001 - the sibling core's definition, shared


def _finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_generation(prompt_ids: list[int], body, max_new: int, label: str) -> dict:
    """The generated ids and their log-probabilities from one `/generate` answer, after the structural checks of step 1;
    `StructuralFailure` names the first departure."""
    meta = body.get("meta_info") if isinstance(body, dict) else None
    out = body.get("output_ids") if isinstance(body, dict) else None
    lps = meta.get("output_token_logprobs") if isinstance(meta, dict) else None
    if not isinstance(out, list) or len(out) != max_new:
        raise StructuralFailure(f"{label}: {len(out) if isinstance(out, list) else 'no'} output ids, expected {max_new}")
    bad = next((j for j, t in enumerate(out) if not _is_token_id(t)), None)
    if bad is not None:
        raise StructuralFailure(f"{label}: output id {out[bad]!r} at generated position {bad} is not a token id")
    if not isinstance(lps, list) or len(lps) != len(out):
        raise StructuralFailure(f"{label}: {len(lps) if isinstance(lps, list) else 'no'} output log-probabilities for "
                                f"{len(out)} generated tokens")
    values = []
    for j, (entry, tok) in enumerate(zip(lps, out)):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise StructuralFailure(f"{label}: the output entry at generated position {j} is not "
                                    f"[logprob, token_id, text]: {entry!r:.120}")
        value, eid = entry[0], entry[1]
        if not _is_token_id(eid) or eid != tok:
            raise StructuralFailure(f"{label}: the output entry at generated position {j} carries token id {eid!r}, "
                                    f"the answer generated {tok}")
        if not _finite_number(value):
            raise StructuralFailure(f"{label}: the output log-probability at generated position {j} is {value!r}, "
                                    f"not a finite number")
        values.append(float(value))
    counts = {k: meta.get(k) for k in ("completion_tokens", "cached_tokens", "spec_verify_ct", "spec_accepted_drafts",
                                       "prompt_tokens")}
    return {"ids": list(out), "logprobs": values, "meta": counts}


def generate(base_url: str, prompt_ids: list[int], max_new: int, label: str) -> dict:
    """One greedy generation (step 1): the checked ids and log-probabilities, the engine's counts and the timing."""
    limit = fidelity.operation_limit_s(len(prompt_ids) + max_new)
    t0 = time.monotonic()
    try:
        r = requests.post(f"{base_url}/generate", json={
            "input_ids": prompt_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new, "ignore_eos": True},
            "return_logprob": True, "logprob_start_len": len(prompt_ids) - 1, "return_text_in_logprobs": False},
            timeout=(limit, limit))
    except requests.Timeout as exc:
        raise ServedRequestFailed(f"{label}: no complete generation within its {limit} s limit") from exc
    elapsed = time.monotonic() - t0
    if r.status_code != 200:
        raise ServedRequestFailed(f"{label}: generation answered HTTP {r.status_code}: {r.text[:300]}")
    try:
        body = r.json()
    except ValueError as exc:
        raise ServedRequestFailed(f"{label}: the generation answer is not JSON: {r.text[:300]!r}") from exc
    if elapsed > limit:
        raise ServedRequestFailed(f"{label}: the generation took {elapsed:.1f} s, over its {limit} s limit")
    got = check_generation(prompt_ids, body, max_new, label)
    got.update({"elapsed_s": round(elapsed, 3), "limit_s": limit, "t0": time.time() - elapsed})
    return got


def generate_together(base_url: str, prompts: list[tuple[str, list[int]]], max_new: int) -> list[dict]:
    """Several generations submitted concurrently (step 2): every one finishes before this returns; the first failure is
    raised after all have ended."""
    if len(prompts) == 1:
        return [generate(base_url, prompts[0][1], max_new, prompts[0][0])]
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futs = [pool.submit(generate, base_url, ids, max_new, label) for label, ids in prompts]
        cf.wait(futs)
    return [f.result() for f in futs]


#: How long the engine may defer a flush until its scheduler is idle (the endpoint's own `timeout` parameter). A
#: generation's HTTP answer can arrive while the scheduler still holds the finished request for one overlapped step;
#: without a deferral the engine refuses the flush ("pending requests"), which says nothing about the history.
FLUSH_WAIT_S = 60


def flush(base_url: str) -> float:
    """SGLang's own cache flush once no request is in flight (step 3): `/flush_cache?timeout=` lets the engine wait
    for its scheduler to go idle before flushing. Its seconds; a refusal is the server's."""
    t0 = time.monotonic()
    try:
        r = requests.post(f"{base_url}/flush_cache", params={"timeout": FLUSH_WAIT_S}, timeout=ACTION_LIMIT_S)
    except requests.Timeout as exc:
        raise ServedRequestFailed(f"/flush_cache: no answer within its {ACTION_LIMIT_S} s limit") from exc
    if r.status_code != 200:
        raise ServedRequestFailed(f"/flush_cache answered HTTP {r.status_code}: {r.text[:200]}")
    time.sleep(2)
    return round(time.monotonic() - t0, 3)


def replay(base_url: str, prompt_ids: list[int], generated_ids: list[int], label: str) -> dict:
    """One cold teacher-forced replay of P + ids (step 4): the N values for the generated positions, checked."""
    history = list(prompt_ids) + list(generated_ids)
    start = len(prompt_ids) - 1
    got = fidelity.served_logprobs(base_url, history, sequence_id=f"{label} (replay)", start=start)
    if got["positions"] != len(generated_ids):
        raise StructuralFailure(f"{label}: the replay scored {got['positions']} positions for {len(generated_ids)} "
                                f"generated tokens")
    return got


def compare(generated: list[float], replays: list[list[float]], first_position: int) -> dict:
    """Step 5, in float64: per token the three replays, their median and range, the excess e, the first replay's d and
    k3, and whether the token is decisive. `replays` is the R passes in the order they were taken."""
    if not replays:
        raise ValueError("the comparison needs at least one replay pass")
    for k, rep in enumerate(replays, start=1):
        if len(rep) != len(generated) or not generated:
            raise ValueError(f"the comparison needs series of the same nonzero length, got {len(generated)} generated "
                             f"and {len(rep)} in replay pass {k}")
    tokens, total, worst, worst_e = [], 0.0, 0.0, 0.0
    for j, g in enumerate(generated):
        rs = [float(rep[j]) for rep in replays]
        srt = sorted(rs)
        lo, hi, m = srt[0], srt[-1], statistics.median(srt)
        e = max(0.0, lo - g, g - hi) if all(math.isfinite(v) for v in srt) else math.inf
        d = rs[0] - float(g)
        term = math.inf if not math.isfinite(d) or d > 700.0 else math.expm1(d) - d
        pos = first_position + j
        tokens.append({"position": pos, "c4": pos % 4, "c128": pos % 128, "generated": g, "replays": rs,
                       "replayed": rs[0], "median": m, "lo": lo, "hi": hi, "d": d, "abs_d": abs(d), "k3": term,
                       "excess": e, "decisive": math.exp(min(float(g), 0.0)) >= DECISIVE_P})
        total += term
        worst = max(worst, abs(d)) if math.isfinite(d) else math.inf
        worst_e = max(worst_e, e) if math.isfinite(e) else math.inf
    return {"tokens": tokens, "n": len(tokens), "replays": len(replays), "mean_k3": total / len(tokens),
            "max_abs_d": worst, "max_excess": worst_e}


def observe(label: str, prompt_ids: list[int], gen: dict, replays: list[dict]) -> dict:
    """The observation record for one finished generation and its R cold replays (the replays already taken, in the
    order of their passes). No request is made here: the caller owns the flush between passes, because every history
    of a group is replayed in the same order in each pass."""
    cmp_ = compare(gen["logprobs"], [r["values"] for r in replays], first_position=len(prompt_ids))
    return {"label": label, "prompt_tokens": len(prompt_ids), "generated_tokens": len(gen["ids"]),
            "prompt_digest": fidelity.ids_digest(prompt_ids), "generated_digest": fidelity.ids_digest(gen["ids"]),
            "engine_counts": gen["meta"], "generate_s": gen["elapsed_s"],
            "replay_s": [r["elapsed_s"] for r in replays], "replay_passes": len(replays),
            "mean_k3": cmp_["mean_k3"], "max_abs_d": cmp_["max_abs_d"], "max_excess": cmp_["max_excess"],
            "tokens": cmp_["tokens"], "generated_ids": gen["ids"]}


def node_statistics(observations: list[dict]) -> dict:
    """Step 6: S1', S2', S3' and S4 for one node and run, pooled over every generated token of its whole schedule."""
    dev_g, dev_rr, excesses, n, decisive = [], [], [], 0, 0
    passes = min((o.get("replay_passes", 1) for o in observations), default=0)
    for o in observations:
        for t in o["tokens"]:
            n += 1
            excesses.append(t["excess"])
            if t["decisive"]:
                decisive += 1
                dev_g.append(abs(t["generated"] - t["median"]))
                if len(t["replays"]) >= 2:
                    dev_rr.append(abs(t["replays"][1] - t["replays"][0]))
    denominator = max(statistics.median(dev_rr), DENOMINATOR_FLOOR) if dev_rr else DENOMINATOR_FLOOR
    return {"S1p": (statistics.median(dev_g) / denominator) if dev_g else math.inf,
            "S2p": sum(1 for e in excesses if e >= EXCESS_NAT),
            "S3p": max(excesses) if excesses else 0.0,
            "S4": (decisive / n) if n else 0.0,
            "tokens": n, "decisive_tokens": decisive, "replay_passes": passes,
            "decisive_median_abs_g_minus_m": statistics.median(dev_g) if dev_g else None,
            "decisive_median_abs_r2_minus_r1": statistics.median(dev_rr) if dev_rr else None}


def replay_group(base_url: str, items: list[tuple[str, list[int], dict]], passes: int = REPLAYS, op=None) -> tuple:
    """Steps 3 and 4 for a group of finished generations: `passes` cold passes, each after its own flush, every
    history replayed in the same order within each pass. Returns (the passes per item in item order, the flush
    seconds)."""
    ctx = op if op is not None else (lambda _kind, _label: contextlib.nullcontext())
    out: list[list[dict]] = [[] for _ in items]
    flush_s = []
    for p in range(1, passes + 1):
        with ctx("action", f"flush before cold replay pass {p}"):
            flush_s.append(flush(base_url))
        for i, (label, prompt_ids, gen) in enumerate(items):
            with ctx("http_request", f"{label}: cold replay pass {p}"):
                out[i].append(replay(base_url, prompt_ids, gen["ids"], f"{label} (pass {p})"))
    return out, flush_s




def run_fresh(base_url: str, example: dict) -> dict:
    """The `fresh` protocol: flush, generate, then R cold replays, each after its own flush."""
    n = int(example["max_new_tokens"])
    flush_s = [flush(base_url)]
    gen = generate(base_url, example["prompt_ids"], n, example["id"])
    passes, more = replay_group(base_url, [(example["id"], example["prompt_ids"], gen)])
    flush_s += more
    obs = [observe(example["id"], example["prompt_ids"], gen, passes[0])]
    return {"protocol": "fresh", "flush_s": flush_s, "replay_passes": REPLAYS, "observations": obs,
            "statistics": node_statistics(obs)}


def run_prefix(base_url: str, example: dict) -> dict:
    """The `prefix` protocol: flush once; each continuation generated in turn on the shared prefix, without a flush
    between them (the later ones reuse the prefix, `cached_tokens` recorded); after all have finished, R cold replay
    passes, each after its own flush, with both histories replayed in the same order in every pass."""
    n = int(example["max_new_tokens"])
    prefix = example["prefix_ids"]
    flush_s = [flush(base_url)]
    items = []
    for cont in example["continuations"]:
        prompt = prefix + cont["ids"]
        label = f"{example['id']}/{cont['id']}"
        items.append((label, prompt, generate(base_url, prompt, n, label)))
    passes, more = replay_group(base_url, items)
    flush_s += more
    obs = [observe(label, prompt, gen, passes[i]) for i, (label, prompt, gen) in enumerate(items)]
    return {"protocol": "prefix", "flush_s": flush_s, "prefix_tokens": len(prefix), "replay_passes": REPLAYS,
            "cached_tokens": [o["engine_counts"].get("cached_tokens") for o in obs], "observations": obs,
            "statistics": node_statistics(obs)}


def load_example(name: str) -> dict:
    ex = json.loads((HERE / "examples" / "replay" / f"{name}.json").read_text())
    ids = ex.get("prompt_ids") or ex.get("prefix_ids")
    if not isinstance(ids, list) or not all(_is_token_id(t) for t in ids):
        raise fidelity.ReferenceFault(f"replay example {name}: its token ids are not a list of token ids")
    return ex


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="generated-history replay of a running server (measured, no pass mark)")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--example", choices=("fresh", "prefix"), required=True)
    ap.add_argument("--output", default=None, help="write the complete record here as JSON")
    args = ap.parse_args(argv)
    ex = load_example(args.example)
    rec = {"example": ex["id"], "input_digest": hashlib.sha256(json.dumps(ex, sort_keys=True).encode()).hexdigest(),
           "base_url": args.base_url}
    try:
        info = requests.get(f"{args.base_url}/get_server_info", timeout=ACTION_LIMIT_S)
        rec["server"] = {k: v for k, v in (info.json() if info.status_code == 200 else {}).items()
                         if k in ("version", "model_path", "tp_size", "speculative_algorithm", "kv_cache_dtype",
                                  "attention_backend", "moe_runner_backend")}
    except (requests.RequestException, ValueError):
        rec["server"] = None
    t0 = time.monotonic()
    try:
        rec.update(run_fresh(args.base_url, ex) if args.example == "fresh" else run_prefix(args.base_url, ex))
        rec["structural"] = "valid"
    except (StructuralFailure, ServedRequestFailed) as exc:
        rec.update({"structural": "invalid", "error": str(exc)})
    rec["elapsed_s"] = round(time.monotonic() - t0, 3)
    if args.output:
        Path(args.output).write_text(json.dumps(rec, indent=1))
    if rec["structural"] != "valid":
        print(f"{ex['id']}: invalid answer: {rec['error']}")
        return 1
    for o in rec["observations"]:
        print(f"{o['label']}: {o['generated_tokens']} tokens after {o['prompt_tokens']}, {o['replay_passes']} cold "
              f"replays, mean k3 {o['mean_k3']:.6g}, max |d| {o['max_abs_d']:.6g}, max excess {o['max_excess']:.6g}, "
              f"cached_tokens {o['engine_counts'].get('cached_tokens')}, "
              f"spec_verify_ct {o['engine_counts'].get('spec_verify_ct')}")
    s = rec["statistics"]
    print(f"S1' {s['S1p']:.4g}  S2' {s['S2p']}  S3' {s['S3p']:.6g}  S4 {s['S4']:.4g}  "
          f"({s['decisive_tokens']} decisive of {s['tokens']} tokens)")
    print("measured; no pass threshold is applied to these examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
