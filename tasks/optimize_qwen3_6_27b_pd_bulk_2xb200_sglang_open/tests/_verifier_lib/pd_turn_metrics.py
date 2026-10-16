#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Turn-group latency percentiles from `benchmark_serving_multi_turn.py --stats-json-output`.

WHY THE SPLIT IS THE WHOLE POINT. Disaggregation moves first turns and continuation turns in OPPOSITE
directions: a first turn pays the full prefill plus a KV transfer it did not previously need, so it gets
slower; a continuation turn runs on a decode pool with no prefill work landing in its batch, so it gets
faster. Reporting one pooled p95 averages a regression against an improvement and can read as "no effect"
when both effects are large. So the scored metric is the CONTINUATION group, and the first-turn group is
reported beside it -- expected worse, and a run where it is not is a run to distrust.

The client's `--stats-json-output` writes `[s._asdict() for s in client_metrics]`, and despite the name
`client_metrics: list[RequestStats]` is per REQUEST. Fields used here, all verified present in the v0.20.1
client (identical set to v0.24.0): `latency_ms`, `ttft_ms`, `tpot_ms`, `input_num_turns`,
`input_num_tokens`, `output_num_tokens`, `approx_cached_percent`, `conversation_id`.

`input_num_turns` counts the turns in the request's message list, so 1 means "first user message, nothing
before it". Anything >1 is a continuation. That is the split, and it needs no client change.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import sys
from pathlib import Path
from typing import Any

def expected_counts(pd_meta: dict) -> dict[str, int]:
    """How many requests the conversation spec asks for, per arm.

    `conversation_turns` counts USER turns and their replies together -- the pd table's own comment says
    "8 = 4 user turns and their replies" -- so the client issues `conversation_turns // 2` requests per
    conversation, and all but the first are continuation turns (the scored group).

    Checked against the shipped spec rather than reasoned about: with conversations=16 and
    conversation_turns=8 the client issues 16*4 = 64 requests, of which 16*3 = 48 are
    continuation turns (the scored group).
    """
    conversations = int(pd_meta["conversations"])
    user_turns = int(pd_meta["conversation_turns"]) // 2
    if conversations < 1 or user_turns < 2:
        raise SystemExit(
            f"FATAL: the conversation spec asks for {conversations} conversation(s) of {user_turns} user "
            f"turn(s); the scored metric is a percentile over CONTINUATION turns, so a spec with fewer "
            f"than two user turns per conversation has no scored group at all."
        )
    return {
        "expected_requests": conversations * user_turns,
        "expected_continuations": conversations * (user_turns - 1),
    }


# When the scored percentile is standing on the edge of a stall cluster, so that a tenth of a
# percentage point of drift moves it many-fold. Two conditions, both required: the upper tail has a
# real CLIFF (some adjacent pair of occupied buckets differs by >= this factor), and the mass ABOVE
# that cliff sits within this distance of the 99th percentile. Neither alone works.
ITL_CLIFF_GAP = 3.0
ITL_CLIFF_MASS_TOLERANCE_PP = 0.4

MIN_REQUESTS = 20
MIN_PER_GROUP = 8


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, so p95 of 20 points is not just "the 19th point"."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(pos)]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


_ITL_BUCKET_FACTOR = 2.0 ** (1.0 / 32.0)
_ITL_BUCKET_MIN_MS = 0.1


def _itl_bucket_upper_ms(index: int) -> float:
    return _ITL_BUCKET_MIN_MS * (_ITL_BUCKET_FACTOR ** index)


def pooled_itl_percentile(rows: list[dict[str, Any]], q: float) -> float | None:
    """The percentile over EVERY inter-token gap in an arm, by summing the per-request histograms.

    Why pooled rather than a percentile of per-request percentiles: the p95 of per-request mean TPOT
    is owned by a single straggler, so its run-to-run spread is an order of magnitude wider than this
    pooled percentile's. Same notion of "tail", three orders of magnitude more samples, and the
    difference between a metric that can carry a score and one that cannot.

    Returns the bucket's UPPER edge: an upper bound at 2.2% resolution, which is stated here because two
    arms differing by 6% are three buckets apart.
    """
    total: dict[int, int] = {}
    for r in rows:
        flat = r.get("itl_hist") or ()
        for i in range(0, len(flat) - 1, 2):
            b = int(flat[i])
            total[b] = total.get(b, 0) + int(flat[i + 1])
    n = sum(total.values())
    if not n:
        return None
    target, seen = q * n, 0
    for b in sorted(total):
        seen += total[b]
        if seen >= target:
            return _itl_bucket_upper_ms(b)
    return _itl_bucket_upper_ms(max(total))


def _summary(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    lat = [float(r["latency_ms"]) for r in rows]
    out: dict[str, Any] = {
        f"{label}_count": len(rows),
        f"{label}_p50_latency_ms": _percentile(lat, 0.50),
        f"{label}_p95_latency_ms": _percentile(lat, 0.95),
        f"{label}_p99_latency_ms": _percentile(lat, 0.99),
        f"{label}_mean_latency_ms": sum(lat) / len(lat) if lat else float("nan"),
    }
    for field in ("ttft_ms", "tpot_ms", "approx_cached_percent", "input_num_tokens",
                  "output_num_tokens"):
        vals = [float(r[field]) for r in rows if r.get(field) is not None]
        if vals:
            out[f"{label}_mean_{field}"] = sum(vals) / len(vals)
            out[f"{label}_p95_{field}"] = _percentile(vals, 0.95)
    return out


def extract(stats_path: Path, *, min_requests: int = MIN_REQUESTS,
            min_per_group: int = MIN_PER_GROUP,
            expected_requests: int | None = None,
            expected_continuations: int | None = None) -> dict[str, Any]:
    rows = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"FATAL: {stats_path} is not a list of per-request records")
    if len(rows) < min_requests:
        raise SystemExit(
            f"FATAL: {len(rows)} requests in {stats_path.name}; a p95 over fewer than {min_requests} is "
            f"not a measurement. Either the run died early or the client wrote a partial file."
        )
    if expected_requests is not None and len(rows) != expected_requests:
        raise SystemExit(
            f"FATAL: {stats_path.name} has {len(rows)} requests, and the conversation spec asks for "
            f"{expected_requests}. A percentile over a SUBSET is not the measurement this task is scored "
            f"on, and the subset is not random -- the requests that disappear are the ones the "
            f"deployment refused, so what is left flatters it. Check the pool logs for 400s (a prompt "
            f"that outgrew `context_length` is the measured cause) before re-running."
        )
    missing = sorted({"latency_ms", "input_num_turns"} - set(rows[0]))
    if missing:
        raise SystemExit(f"FATAL: the stats records lack {missing}; the client's schema changed")

    first = [r for r in rows if int(r["input_num_turns"]) <= 1]
    cont = [r for r in rows if int(r["input_num_turns"]) > 1]
    if len(cont) < min_per_group:
        raise SystemExit(
            f"FATAL: only {len(cont)} continuation-turn requests. The scored metric is a p95 over that "
            f"group, so a workload that produces almost none is measuring the wrong thing -- check that "
            f"the conversation spec really asks for multiple turns."
        )

    if expected_continuations is not None and len(cont) != expected_continuations:
        raise SystemExit(
            f"FATAL: {stats_path.name} has {len(cont)} continuation-turn requests against the "
            f"{expected_continuations} the spec asks for. That group IS the scored metric, so this is not "
            f"a measurement of this task."
        )

    _hist: dict[int, int] = {}
    for _r in cont:
        _h = _r.get("itl_hist") or []
        for _i in range(0, len(_h) - 1, 2):
            _hist[int(_h[_i])] = _hist.get(int(_h[_i]), 0) + int(_h[_i + 1])
    _cliff_mass = _cliff_gap = None
    if len(_hist) >= 3:
        _idx = sorted(_hist)
        _tot = sum(_hist.values())
        _cum = 0
        _p90i = _idx[-1]
        for _i in _idx:
            _cum += _hist[_i]
            if _cum >= 0.90 * _tot:
                _p90i = _i
                break
        _best_ratio, _best_lo = 0.0, None
        for _a, _b in zip(_idx, _idx[1:]):
            if _a < _p90i:
                continue
            _ratio = _itl_bucket_upper_ms(_b) / _itl_bucket_upper_ms(_a)
            if _ratio > _best_ratio:
                _best_ratio, _best_lo = _ratio, _a
        if _best_lo is not None:
            _cliff_gap = _best_ratio
            _cliff_mass = sum(c for i2, c in _hist.items() if i2 > _best_lo) / _tot

    result: dict[str, Any] = {"total_requests": len(rows)}
    if _cliff_mass is not None:
        result["pd_itl_cliff_gap"] = round(_cliff_gap, 2)
        result["pd_itl_cliff_mass_pct"] = round(_cliff_mass * 100, 3)
        _on_edge = (_cliff_gap >= ITL_CLIFF_GAP
                    and abs(_cliff_mass * 100 - 1.0) < ITL_CLIFF_MASS_TOLERANCE_PP)
        result["pd_itl_p99_on_cliff_edge"] = _on_edge
        if _on_edge:
            print(
                f"[pd-metrics] WARNING: the scored pooled p99 ITL is standing on the edge of a stall "
                f"cluster. {_cliff_mass * 100:.2f}% of inter-token gaps sit above a {_cliff_gap:.1f}x "
                f"jump in the distribution, and the 99th percentile is exactly that boundary -- so which "
                f"side of the cluster p99 lands on is decided by a tenth of a percentage point. In that "
                f"state the p99 can move by an order of magnitude from one repeat to the next while the "
                f"continuation p95, ttft and tpot all stay within a few percent. Do not read this run's "
                f"ITL term as a measurement of the deployment.",
                flush=True,
            )
    result.update(_summary(cont, "continuation"))
    if len(first) >= min_per_group:
        result.update(_summary(first, "first_turn"))
    else:
        result["first_turn_count"] = len(first)
        result["first_turn_note"] = (
            f"only {len(first)} first-turn requests, below the {min_per_group} floor -- reported count "
            f"only, no percentiles"
        )

    result["continuation_p95_latency_ms"] = result["continuation_p95_latency_ms"]
    result["first_turn_p95_latency_ms"] = result.get("first_turn_p95_latency_ms")

    cached = [float(r["approx_cached_percent"]) for r in cont if r.get("approx_cached_percent") is not None]
    if cached:
        result["continuation_cached_percent_mean"] = sum(cached) / len(cached)
        result["continuation_cached_percent_p05"] = _percentile(cached, 0.05)
    for q, name in ((0.50, "p50"), (0.95, "p95"), (0.99, "p99")):
        v = pooled_itl_percentile(cont, q)
        if v is not None:
            result[f"pooled_{name}_itl_ms"] = v
    p99_itl = result.get("pooled_p99_itl_ms")
    lat_p95 = result.get("continuation_p95_latency_ms")

    if p99_itl and lat_p95:
        # THE SCORED VALUE, a geometric mean of the two on purpose. compute_primary_score takes ONE
        # metric per workload and forms pristine/candidate, so a two-term score has to be one number
        # whose RATIO equals what we want. sqrt(a*b) does exactly that:
        # sqrt(a1*b1)/sqrt(a2*b2) == sqrt((a1/a2)*(b1/b2)), i.e. the geomean of the two ratios, with no
        # change to the reward machinery. Two terms because one alone is answerable by making the other
        # worse: continuation p95 is what an agent rollout feels; pooled P99 ITL is what
        # disaggregation is FOR.
        result["pd_latency_itl_geomean_ms"] = (lat_p95 * p99_itl) ** 0.5

    conversations = {r.get("conversation_id") for r in rows}
    result["conversations"] = len(conversations - {None})
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats", type=Path, nargs="+", help="one or more --stats-json-output files")
    ap.add_argument("--label", action="append", default=[],
                    help="a label per stats file, in order (e.g. --label aggregated --label pd)")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    labels = args.label or [p.stem for p in args.stats]
    if len(labels) != len(args.stats):
        raise SystemExit(f"FATAL: {len(labels)} labels for {len(args.stats)} files")

    out: dict[str, Any] = {}
    for label, path in zip(labels, args.stats):
        out[label] = extract(path)
        r = out[label]
        print(f"\n=== {label} ({path}) ===")
        print(f"  requests {r['total_requests']}  conversations {r['conversations']}")
        print(f"  continuation n={r['continuation_count']:<5d} "
              f"p50 {r['continuation_p50_latency_ms']:>9.1f} ms  "
              f"p95 {r['continuation_p95_latency_ms']:>9.1f} ms  "
              f"p99 {r['continuation_p99_latency_ms']:>9.1f} ms")
        if r.get("first_turn_p95_latency_ms") is not None:
            print(f"  first turn   n={r['first_turn_count']:<5d} "
                  f"p50 {r['first_turn_p50_latency_ms']:>9.1f} ms  "
                  f"p95 {r['first_turn_p95_latency_ms']:>9.1f} ms")
        else:
            print(f"  first turn   n={r.get('first_turn_count')} -- {r.get('first_turn_note')}")
        if "continuation_cached_percent_mean" in r:
            print(f"  cached%      mean {r['continuation_cached_percent_mean']:.1f} "
                  f"p05 {r['continuation_cached_percent_p05']:.1f}   "
                  f"(prefix reuse across turns)")
        print(f"  mean ttft {r.get('continuation_mean_ttft_ms', float('nan')):.1f} ms   "
              f"mean tpot {r.get('continuation_mean_tpot_ms', float('nan')):.2f} ms")

    if len(args.stats) == 2:
        a, b = labels
        pa = out[a]["continuation_p95_latency_ms"]
        pb = out[b]["continuation_p95_latency_ms"]
        print(f"\n=== {b} vs {a}, on the SCORED axis (continuation p95, lower is better) ===")
        print(f"  {a:>12s}: {pa:9.1f} ms")
        print(f"  {b:>12s}: {pb:9.1f} ms")
        print(f"  ratio {a}/{b} = {pa / pb:.4f}   "
              f"reward-shaped max(0, ratio-1) = {max(0.0, pa / pb - 1):.4f}")
        fa = out[a].get("first_turn_p95_latency_ms")
        fb = out[b].get("first_turn_p95_latency_ms")
        if fa and fb:
            print(f"  first-turn p95 {a} {fa:.1f} -> {b} {fb:.1f} ms "
                  f"({'worse, as predicted' if fb > fa else 'BETTER -- unexpected, distrust the run'})")

    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())




def aggregate_profiles(per_profile_scores: dict[str, list[float]], profiles: list[dict]) -> list[float]:
    """Reduce {profile_name: [score per rep]} to [aggregated score per rep], geomean across profiles.

    Returns one entry per measurement ROUND, not per profile: round `i` gets the geomean of every scored
    profile's score at index `i`. That is what preserves the paired-noise-cancelling property the
    single-point path had -- candidate round i and pristine round i are still compared as rounds.

    A round is DROPPED if any scored profile failed to produce a score for it. Reporting a geomean over a
    subset of the points would silently score a different axis than the one declared.

    `scored: false` profiles are MEASURED but excluded, and `n_meas` is taken over the SCORED profiles alone
    -- copied from the existing reducer for its stated reason: were it taken over all of them, one failed
    guardrail bench would drop a round whose scored points all succeeded, so a guardrail could zero a valid
    score.
    """
    scored = [p for p in profiles if p.get("scored", True)]
    if not scored:
        raise SystemExit(
            "FATAL: every PD profile is scored=false, so the workload has no score at all. At least one "
            "concurrency point must be scored."
        )
    missing = [p["name"] for p in scored if p["name"] not in per_profile_scores]
    if missing:
        raise SystemExit(
            f"FATAL: the declared scored profile(s) {missing} produced no entry at all. The declared axis and "
            f"the measured one must match; a geomean over what happened to run is a different measurement."
        )
    n_meas = min((len(per_profile_scores[p["name"]]) for p in scored), default=0)
    out: list[float] = []
    for index in range(n_meas):
        at_index = [per_profile_scores[p["name"]][index] for p in scored]
        if all(value > 0 for value in at_index):
            out.append(math.exp(sum(math.log(value) for value in at_index) / len(at_index)))
    return out


PROFILE_DIVISORS = (4, 2, 1)


def _override_profiles(profiles: list[dict]) -> list[dict]:
    """Regenerate the scored concurrency points when INFB_PD_CONVERSATIONS moves the conversation count.

    THE OTHER HALF of the knob applied in `pd_topology.apply_workload_overrides`, and the half that is
    easy to leave out: with only the table side applied, the conversation count moves and the baked
    points do not, so `get_client_config` raises `max_active_conversations > len(input_conv)` or the run
    scores a concurrency it never ran. A knob in that state is worse than no knob.

    Both halves read the same variable, which is what makes them unable to disagree.

    Inert unset: returns the list it was given, unchanged and BY IDENTITY. `profile_points` rebuilds its
    own list after validating, so the identity contract is this function's, not its caller's.
    """
    raw = os.environ.get("INFB_PD_CONVERSATIONS", "").strip()
    if not raw or not profiles:
        return profiles
    try:
        conversations = int(raw)
    except ValueError:
        raise SystemExit(
            f"FATAL: INFB_PD_CONVERSATIONS={raw!r} is not an integer; it sets both the conversation count "
            f"and the three scored concurrency points."
        ) from None
    est_requests = conversations * 4
    if est_requests < MIN_REQUESTS:
        raise SystemExit(
            f"FATAL: INFB_PD_CONVERSATIONS={conversations} yields about {est_requests} requests and "
            f"`extract()` refuses a p95 over fewer than {MIN_REQUESTS} -- the run completes and then throws "
            f"the measurement away. The smallest workable value is "
            f"{-(-MIN_REQUESTS // 4)} (below it, hours of setup and benching are discarded)."
        )
    points = sorted({max(1, conversations // d) for d in PROFILE_DIVISORS})
    if len(points) != len(PROFILE_DIVISORS):
        raise SystemExit(
            f"FATAL: INFB_PD_CONVERSATIONS={conversations} collapses the scored points to {points} -- "
            f"{len(PROFILE_DIVISORS)} distinct concurrencies are required and this yields {len(points)}. "
            f"The smallest workable value is {min(d for d in PROFILE_DIVISORS if d > 1) * 2}."
        )
    out = [{**dict(profiles[0]), "name": f"c{c}", "concurrency": c} for c in points]
    print(f"[pd-metrics] scored concurrency points regenerated {[p.get('concurrency') for p in profiles]} "
          f"-> {points} (INFB_PD_CONVERSATIONS={conversations}). Experiment control; no task declares it.",
          flush=True)
    return out


def profile_points(workload: dict) -> list[dict]:
    """The declared concurrency points, validated. One entry per point, `name` and `concurrency` required.

    A workload with no `profiles` is REFUSED rather than defaulted to its single `max_concurrency`. A silent
    fallback is how a task ends up declaring a sweep and running one point -- reporting a number for an axis
    it never measured.
    """
    profiles = _override_profiles(workload.get("profiles") or [])
    if not profiles:
        raise SystemExit(
            f"FATAL: workload {workload.get('name')!r} declares no `profiles`, and the PD score is a geomean "
            f"over three concurrency points. They are published with the task, from the pd table."
        )
    seen = set()
    for entry in profiles:
        for key in ("name", "concurrency"):
            if key not in entry:
                raise SystemExit(f"FATAL: PD profile {entry!r} is missing {key!r}")
        if int(entry["concurrency"]) < 1:
            raise SystemExit(f"FATAL: PD profile {entry['name']!r} has concurrency {entry['concurrency']}")
        if entry["name"] in seen:
            raise SystemExit(f"FATAL: PD profile name {entry['name']!r} is declared twice")
        seen.add(entry["name"])
    return list(profiles)
