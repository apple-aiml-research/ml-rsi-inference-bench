#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The execution witness's private verdict over the public attribution protocol (`msa/binding.py`).

Component side: the MSA group's correctness nodes profile one call of each entry-point family on the candidate's
implementation and record the identity sets, with the tree and build-product digests
(`/logs/verifier/msa/identity.json`). Serving side: the node profiles a fixed prompt set on the running server;
each tensor-parallel rank writes its own trace. Pass iff on every one of the four ranks: each of the four entry
points has at least one identity-set kernel in the trace (attention sets of the deployment's KV dtype); the most
frequent kernel of the `attend_decode` set occurs at least 57 x 31 times (57 sparse layers, 31 decode steps of the
single request alone); `cudaGraphLaunch` occurs. Identity comes from observation, never from fixed names.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
for _path in (str(TESTS_DIR / "eval"), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from infb_verify.records import Blocked  # noqa: E402

MSA_LOGS = Path(os.environ.get("INFB_MSA_DIR", "/logs/verifier/msa"))
FAMILIES = ("select_prefill", "select_decode", "attend_prefill_bf16", "attend_prefill_fp8", "attend_decode_bf16",
            "attend_decode_fp8")
SPARSE_LAYERS = 57
DECODE_STEPS = 31
RANKS = 4
TRACE_WAIT_S = 300


def identity_record(deployment: str) -> dict:
    """The component identity sets, or blocked: the witness cannot be judged without them."""
    path = MSA_LOGS / "identity.json"
    if not path.exists():
        raise Blocked("infb_verify.test_msa component identity record (/logs/verifier/msa/identity.json)")
    rec = json.loads(path.read_text())
    fams = rec.get("families") or {}
    missing = [f for f in FAMILIES if (fams.get(f) or {}).get("status") != "observed"]
    if missing:
        raise Blocked(f"infb_verify.test_msa component identity sets for {missing}")
    return rec


def _trace_files(trace_dir: Path) -> list[Path]:
    return sorted(p for p in trace_dir.rglob("*") if p.is_file() and p.name.endswith((".json", ".json.gz")))


def observe_serving(identity: dict, trace_dir: Path, kv: str) -> dict:
    """Wait for the ranks' traces (the profiler writes them after /stop_profile returns), observe each, decide."""
    from minimax_m3.msa import binding as public  # noqa: PLC0415 - imports torch

    families = {f: identity["families"][f] for f in FAMILIES}
    deadline = time.monotonic() + TRACE_WAIT_S
    observed: dict[str, dict] = {}
    while True:
        files = _trace_files(trace_dir)
        observed = {}
        for p in files:
            try:
                trace = public.load_trace(p)
            except (OSError, ValueError, EOFError):
                continue  # still being written
            kernels, _ = public.trace_counts(trace)
            if kernels:  # a worker's trace (the API server's front-end trace has no GPU kernels)
                observed[str(p)] = public.observe_rank(families, trace, kv)
        if len(observed) >= RANKS or time.monotonic() > deadline:
            break
        time.sleep(5)
    problems = []
    if len(observed) != RANKS:
        problems.append(f"{len(observed)} rank traces with GPU kernels, expected {RANKS}")
    need = SPARSE_LAYERS * DECODE_STEPS
    for name, row in observed.items():
        rank = Path(name).name
        absent = [e for e, ok in row["entry_points_observed"].items() if not ok]
        if absent:
            problems.append(f"{rank}: no identity-set kernel of {absent} in the trace")
        if row["attend_decode_top_count"] < need:
            problems.append(f"{rank}: the attend_decode kernel ran {row['attend_decode_top_count']} times "
                            f"(need {need})")
        if row["cuda_graph_launches"] < 1:
            problems.append(f"{rank}: no cudaGraphLaunch")
    return {"trace_dir": str(trace_dir), "ranks": observed, "kv": kv,
            "verdict": {"pass": not problems, "problems": problems}}
