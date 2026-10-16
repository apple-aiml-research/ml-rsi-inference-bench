#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Warmed, rotated bench rounds for the scored ratio.

WHAT THIS REPLACES. `tests/test.sh` used to call `_candidate_runner.py` exactly twice: pristine, then
candidate, one sample each, in a cold container. Measured with a candidate tree BYTE-IDENTICAL to pristine, that estimator
reported a speedup of roughly ten percent for no change at all, because the first bench in a
container costs far more than the second for identical work. The artefact was larger than the
reference repair's own effect, so no threshold on that estimator could separate a real repair
from doing nothing. The two arms ran identical work -- same seed, the same verify and
accept-length counters, zero accept-length drift in every replicate. Only wall clock differed.

WHAT THIS DOES, and why it is two corrections rather than one (see _verifier_lib/bench_stats.py for
the full argument):

  1. ONE OR MORE DISCARDED WARMUP BENCHES on the pristine tree, adaptive against a plateau rather than
     a fixed count. This removes the MAGNITUDE: after it, no scored bench is the first in the
     container.
  2. SCORED ROUNDS WITH ROTATED ARM ORDER. This removes the ASYMMETRY: the residual first-in-round
     cost lands on each arm equally instead of always on `pristine`, the shared denominator. A warmup
     alone leaves a smaller bias of the SAME SIGN, which is leniently wrong rather than noisily wrong.

Rounds must be EVEN (a multiple of the arm count) or round 0's identity ordering leaves one arm
leading more often -- `rotated_arms` documents this.

COST, priced off the same runs: today 196.7 s; +warmup ~267 s; +warmup+rotation ~421 s, which is
7.8% of a 5400 s verifier budget and 4 benches against a per-bench cap the worst arm used 6.4% of.
`--budget-sec` is nevertheless enforced on the WARMUP loop, because the magnitudes above are measured
on ONE task and the family's other tasks are unmeasured: a task whose bench is 20x slower must degrade
to fewer warmups rather than spend the whole verifier budget and leave no baseline at all.

FAILURE POSTURE. This script never fails the shell -- it exits 0 and records what happened, exactly
like `_candidate_runner.py`. The verifier decides. If it produces no usable rounds it also leaves the
legacy single-bench artefacts absent, and `check_performance` then fails the declared gate closed as
`baseline_tps_unavailable` -> `pristine_failed`, which is an infra zero and a rerun, never a pass.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _verifier_lib import regression_scoring as _scoring  # noqa: E402
from _verifier_lib.bench_stats import (  # noqa: E402
    first_run_off_plateau,
    load_matrix_spec,
    matrix_section,
    ratio_with_stdev,
    read_point_series,
    rotated_arms,
    summarise_arm,
    warmup_plateau_state,
)
from _verifier_lib.metrics import normalize_engine_metrics  # noqa: E402

ARMS = ("pristine", "candidate")


def _read_metric(raw_dir: Path, raw_name: str, metric_source: str, metric_key: str) -> float | None:
    """The scored scalar one bench produced, or None.

    Goes through `normalize_engine_metrics` -- the SAME normaliser the verifier uses -- rather than
    reaching into the engine's raw JSON with its own key names. A second reader with its own idea of
    where throughput lives is how an aggregate silently stops matching the number the gate compares.
    """
    bench = raw_dir / raw_name
    if not bench.exists():
        return None
    try:
        metrics = normalize_engine_metrics(
            bench, raw_dir / "_round_metrics.json", metric_source=metric_source
        )
    except Exception:  # noqa: BLE001 -- a malformed bench result is "no number", not a crash
        return None
    node: object = metrics.get("summary", {})
    for part in metric_key.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return float(node) if isinstance(node, (int, float)) and node > 0 else None


def _bench(runner: Path, mode: str, repo: Path, raw_dir: Path, timeout: float | None,
           workloads: str = "anchor") -> int:
    """One runner invocation: one engine session over `--workloads` (the whole matrix, warmups included)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(runner), mode, "--repo", str(repo), "--raw-dir", str(raw_dir),
           "--workloads", workloads]
    log = raw_dir / "_runner.log"
    try:
        with log.open("w", encoding="utf-8") as handle:
            return subprocess.run(
                cmd, stdout=handle, stderr=subprocess.STDOUT, env=dict(os.environ),
                timeout=timeout,
            ).returncode
    except subprocess.TimeoutExpired:
        # The runner already caps its own bench subprocess; this outer timeout only exists so a
        # wedged runner cannot outlive the round loop. Recorded, never raised.
        return 124


def _publish(src: Path, dest: Path, names: list[str]) -> list[str]:
    """Copy the NAMED bench artefacts of one round into the legacy location the verifier reads.

    NAMED, not the whole directory, on purpose: `tests/test.sh` writes `pristine_tree.json` into the
    baseline raw dir before any bench runs, and `load_pristine_manifest()` reads it to tell a
    digest-mismatch infra zero apart from a missing baseline. A recursive copy would clobber it.
    """
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in names:
        source = src / name
        if source.is_file():
            shutil.copy2(source, dest / name)
            copied.append(name)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, help="the mode arg _candidate_runner.py expects")
    parser.add_argument("--pristine", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--artifacts", required=True, help="INFB_ARTIFACTS_DIR")
    parser.add_argument("--raw-name", required=True, help="the runner's result JSON basename")
    parser.add_argument("--manifest-name", required=True)
    parser.add_argument("--metric-source", required=True, choices=["vllm", "sglang"])
    parser.add_argument(
        "--metric-key", default="throughput.total_tokens_per_sec",
        help="dotted path under metrics['summary'] -- MUST be the key the verifier divides",
    )
    parser.add_argument(
        "--rounds", type=int, default=2,
        help="scored rounds per arm; MUST be even or rotation is only partial",
    )
    parser.add_argument(
        "--min-warm", type=int, default=2,
        help="plateau floor. Measured on the A/B rig: vLLM needed 2, sglang 1.",
    )
    parser.add_argument("--warmup-cap", type=int, default=4)
    parser.add_argument(
        "--budget-sec", type=float, default=0.0,
        help="stop warming once cumulative warmup elapsed exceeds this (0 = unlimited)",
    )
    parser.add_argument("--bench-timeout-sec", type=float, default=0.0)
    parser.add_argument(
        "--workloads-json", default=None,
        help="the trusted tests/workloads.json (default: beside this file). When it declares more "
             "than one scored point or any guard point, every scored round benches the whole matrix "
             "in one engine session and the record carries a per-point `workloads` section.",
    )
    parser.add_argument(
        "--lower-is-better", action="store_true",
        help="the scored metric is a cost (mean TPOT, TTFT): the per-point direction default",
    )
    parser.add_argument(
        "--guard-band", action="append", default=[], metavar="NAME=RATIO",
        help="override one guard's min_ratio (e.g. repeat_img2_n120=0.93): the band a task earns from its "
             "legs when its trusted workloads.json cannot carry it (the two copies must stay identical)",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    runner = here / "_candidate_runner.py"
    artifacts = Path(args.artifacts)
    trees = {"pristine": Path(args.pristine), "candidate": Path(args.candidate)}
    timeout = args.bench_timeout_sec or None
    artefact_names = [
        args.raw_name, args.manifest_name,
        f"{args.metric_source}_bench_stdout.txt", f"{args.metric_source}_bench_stderr.txt",
        "bench_stdout.txt", "bench_manifest.json",
    ]

    spec = load_matrix_spec(args.workloads_json or (here / "workloads.json"))
    is_matrix = len(spec["scored"]) > 1 or bool(spec["guards"])
    for item in args.guard_band:
        gname, _, gval = item.partition("=")
        if gname not in spec["guards"]:
            raise SystemExit(f"--guard-band names {gname!r}, which is not a guard of the matrix: {spec['guards']}")
        spec["points"][gname]["min_ratio"] = float(gval)

    def _point_reader(raw_dir: Path, metric_key: str) -> float | None:
        return _read_metric(raw_dir, args.raw_name, args.metric_source, metric_key)

    point_series: dict[str, dict[str, list[float]]] = {
        name: {arm: [] for arm in ARMS} for name in spec["points"]
    }

    record: dict[str, object] = {
        "schema_version": 2 if is_matrix else 1,
        "design": "discarded warmup + rotated scored rounds"
                  + (" + workload matrix" if is_matrix else ""),
        "mode": args.mode,
        "rounds_requested": args.rounds,
        "min_warm": args.min_warm,
        "metric_key": args.metric_key,
        "notes": [],
    }
    if args.rounds % len(ARMS) != 0:
        record["notes"].append(
            f"rounds={args.rounds} is not a multiple of {len(ARMS)} arms, so round 0's identity "
            f"ordering leaves one arm leading more often -- the rotation correction is PARTIAL"
        )

    warmup: dict[str, dict[str, object]] = {}
    per_arm_budget = (args.budget_sec / len(ARMS)) if args.budget_sec else 0.0
    for arm in ARMS:
        warm_series: list[float | None] = []
        plateau: list[float] = []
        tol = 0.01
        warm_elapsed = 0.0
        for attempt in range(1, args.warmup_cap + 1):
            raw_dir = artifacts / "warmup" / arm / str(attempt) / "raw"
            started = time.time()
            rc = _bench(runner, args.mode, trees[arm], raw_dir, timeout,
                        workloads="all" if is_matrix else "anchor")
            took = time.time() - started
            warm_elapsed += took
            value = _read_metric(raw_dir, args.raw_name, args.metric_source, args.metric_key)
            warm_series.append(value)
            is_warm, plateau, tol = warmup_plateau_state(warm_series, args.min_warm)
            print(f"[bench_rounds] warmup {arm} {attempt} DISCARDED rc={rc} metric={value} "
                  f"({took:.0f}s) plateau={len(plateau)}/{args.min_warm + 1} tol={tol:.2%}",
                  flush=True)
            if is_warm:
                break
            if len(warm_series) >= 2 and warm_series[-1] is None and warm_series[-2] is None:
                record["notes"].append(
                    f"two consecutive {arm} warmups produced no metric; stopped after {attempt} of "
                    f"{args.warmup_cap}. The scored rounds will start COLD on that arm -- read "
                    f"first_scored_run_vs_warmup_plateau and the per-round spread with that in mind."
                )
                break
            if per_arm_budget and warm_elapsed > per_arm_budget:
                record["notes"].append(
                    f"{arm} warmup budget {per_arm_budget:.0f}s exceeded after {attempt} run(s) "
                    f"({warm_elapsed:.0f}s); stopping so the scored rounds and the baseline still fit"
                )
                break
        else:
            record["notes"].append(
                f"no confirmed plateau after {args.warmup_cap} warmups on the {arm} arm "
                f"({warm_series}); the per-round spread and the first-scored-run backstop are what "
                f"to read on this run"
            )
        entry: dict[str, object] = {"runs": warm_series, "elapsed_sec": round(warm_elapsed, 1)}
        if plateau:
            entry["plateau"] = {
                "n": len(plateau),
                "mean": round(sum(plateau) / len(plateau), 4),
                "tol": round(tol, 5),
            }
        warmup[arm] = entry

    record["warmup"] = warmup
    record["warmup_runs"] = warmup["pristine"]["runs"]
    record["warmup_elapsed_sec"] = warmup["pristine"]["elapsed_sec"]
    if "plateau" in warmup["pristine"]:
        record["warmup_plateau"] = warmup["pristine"]["plateau"]

    series: dict[str, list[float]] = {arm: [] for arm in ARMS}
    order_log: list[list[str]] = []
    published: dict[str, Path] = {}
    for index in range(args.rounds):
        order = rotated_arms(list(ARMS), index)
        order_log.append(order)
        for position, arm in enumerate(order):
            raw_dir = artifacts / "rounds" / str(index) / arm / "raw"
            started = time.time()
            rc = _bench(runner, args.mode, trees[arm], raw_dir, timeout,
                        workloads="all" if is_matrix else "anchor")
            took = time.time() - started
            value = _read_metric(raw_dir, args.raw_name, args.metric_source, args.metric_key)
            if is_matrix:
                for name, point_value in read_point_series(
                        spec, raw_dir, _point_reader, default_key=args.metric_key,
                        default_lower=args.lower_is_better).items():
                    if point_value is not None:
                        point_series[name][arm].append(point_value)
                    else:
                        record["notes"].append(f"round {index} {arm}: point {name} produced no metric")
            print(f"[bench_rounds] round {index} {arm:<9} (position {position}) rc={rc} "
                  f"metric={value} ({took:.0f}s)", flush=True)
            record.setdefault("elapsed_sec", {}).setdefault(arm, []).append(round(took, 1))
            if value is not None:
                series[arm].append(value)
                published[arm] = raw_dir
            elif rc != 0:
                record["notes"].append(f"round {index} {arm} produced no metric (rc={rc})")
    record["arm_order"] = order_log

    record["arms"] = {arm: summarise_arm(series[arm]) for arm in ARMS}
    ratio, ratio_stdev = ratio_with_stdev(record["arms"]["pristine"], record["arms"]["candidate"])
    record["ratio"] = ratio
    record["ratio_stdev"] = ratio_stdev
    if is_matrix:
        record["workloads"] = matrix_section(spec, point_series, default_key=args.metric_key,
                                             default_lower=args.lower_is_better)
        record["scored_workloads"] = list(spec["scored"])
        record["guard_workloads"] = list(spec["guards"])
        record["anchor"] = spec["anchor"]
        resolved = _scoring.resolve_matrix_ratio(record)
        record["anchor_ratio"] = ratio
        record["ratio"] = resolved.get("speedup")
        record["matrix"] = {key: resolved.get(key) for key in (
            "estimator", "reason", "speedup_per_workload", "worst_workload_speedup",
            "best_workload_speedup", "guard_ratios", "guards_ok", "guard_unmeasured",
            "unusable_workloads")}
    first_arm = order_log[0][0] if order_log else ARMS[0]
    first_scored = (series.get(first_arm) or [None])[0]
    first_plateau = (warmup.get(first_arm) or {}).get("plateau") or {}
    plateau_mean = first_plateau.get("mean")
    plateau_tol = first_plateau.get("tol", 0.01)
    deviation = first_run_off_plateau(
        first_scored, [plateau_mean] if plateau_mean is not None else [], plateau_tol
    )
    if deviation is not None:
        record["first_scored_run_vs_warmup_plateau"] = round(deviation, 5)
        if abs(deviation) > plateau_tol:
            record["notes"].append(
                f"the first scored bench ({first_arm}) sits {deviation:+.2%} off that arm's warmup "
                f"plateau (tolerance {plateau_tol:.2%}) -- it may still have been warming. Rotation "
                f"keeps this out of the RATIO, but a no-op control could not have seen it at all."
            )

    record["published"] = {}
    for arm, dest in (("candidate", artifacts / "raw"),
                      ("pristine", artifacts / "baseline" / "raw")):
        source = published.get(arm)
        if source is not None:
            record["published"][arm] = {
                "from": str(source), "to": str(dest),
                "files": _publish(source, dest, artefact_names),
            }

    out = artifacts / "bench_rounds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[bench_rounds] ratio={ratio} stdev={ratio_stdev} "
          f"pristine={record['arms']['pristine']['runs']} "
          f"candidate={record['arms']['candidate']['runs']} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
