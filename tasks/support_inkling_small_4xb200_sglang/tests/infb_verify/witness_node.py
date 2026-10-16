#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""`test_execution_witness`: the served model runs the component's GPU computations, on every rank.

Expected verdicts, stated first: the reference passes; a correct operator the served model does not run fails;
a renamed or differently fused implementation passes, because identity comes from observation.

1. Component side: the operator phase profiled the committed entry point (`rel_attention/binding.py`) and wrote the
   four identity sets -- `prefill_full`, `prefill_local`, `decode_full`, `decode_local` -- to `INFB_BINDING_IDS`
   (default `/logs/verifier/records/binding.json`).
2. Serving side, last on the production launch: `/start_profile` (CPU and GPU activities, a fresh directory), one
   6,144-token prose prompt generating 32 tokens, then three concurrent 3,000-token prompts generating 32 tokens each,
   then `/stop_profile`; the four ranks' traces are read.
3. Pass iff on every rank: each prefill identity set has at least one kernel in the trace; the most frequent served
   kernel of `decode_full` occurs at least 7 x 31 times and that of `decode_local` at least 35 x 31 times (42 x 31 if the
   two are the same kernel); `cudaGraphLaunch` occurs.

Kernel identity is the observed name. The CuTe DSL names carry the argument layouts after `_object_at_`; the record
also reports each match on the name up to that point, so a layout-only difference between the component's calls and the
served calls is visible as such.
"""
from __future__ import annotations

import collections
import concurrent.futures
import gzip
import json
import lzma
import os
import re
import time
from pathlib import Path

import requests

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
DECODE_STEPS = 31
FULL_LAYERS, LOCAL_LAYERS = 7, 35
TRACE_ROOT = Path(os.environ.get("INFB_WITNESS_DIR", "/logs/verifier/witness"))


def normalized(name: str) -> str:
    return name.split("_object_at_", 1)[0]


def identity_sets() -> dict[str, list[str]]:
    path = Path(os.environ.get("INFB_BINDING_IDS", "/logs/verifier/records/binding.json"))
    doc = json.loads(path.read_text())
    return doc.get("identity_sets", doc)


def prompts() -> tuple[list[int], list[list[int]]]:
    seqs = json.load(lzma.open(TESTS_DIR / "hidden" / "fidelity.json.xz", "rt"))["sequences"]
    by = {(s["family"], s["band"]): s for s in seqs}
    long = by[("prose", 8192)]["token_ids"][:6144]
    short = [by[(f, 8192)]["token_ids"][:3000] for f in ("code", "dialogue", "records")]
    return long, short


def _generate(base_url: str, ids: list[int], timeout: float) -> None:
    r = requests.post(f"{base_url}/generate", json={"input_ids": ids, "sampling_params": {
        "temperature": 0.0, "max_new_tokens": 32, "ignore_eos": True}}, timeout=timeout)
    r.raise_for_status()


def read_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return json.load(fh).get("traceEvents", [])


def observe(base_url: str, server_pid: int | None, timeout: float = 900.0, wait_s: float = 180.0):
    ids = identity_sets()
    out = TRACE_ROOT / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out.mkdir(parents=True, exist_ok=True)
    long, short = prompts()
    body = {"output_dir": str(out), "activities": ["CPU", "GPU"], "with_stack": False, "record_shapes": False}
    r = requests.post(f"{base_url}/start_profile", json=body, timeout=300)
    r.raise_for_status()
    try:
        _generate(base_url, long, timeout)
        with concurrent.futures.ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda p: _generate(base_url, p, timeout), short))
    finally:
        requests.post(f"{base_url}/stop_profile", timeout=300)
    deadline = time.monotonic() + wait_s
    files: list[Path] = []
    while time.monotonic() < deadline:
        files = sorted(p for p in out.rglob("*.json*") if re.search(r"TP-\d+", p.name))
        if len({re.search(r"TP-(\d+)", p.name).group(1) for p in files}) >= 4:
            break
        time.sleep(3)
    failures, ranks = [], {}
    if len({re.search(r"TP-(\d+)", p.name).group(1) for p in files}) < 4:
        failures.append(f"traces for {len(files)} file(s), not four ranks, under {out}")
    same = sorted(ids.get("decode_full", [])) == sorted(ids.get("decode_local", []))
    for path in files:
        rank = re.search(r"TP-(\d+)", path.name).group(1)
        counts, graph = collections.Counter(), 0
        for ev in read_trace(path):
            if ev.get("cat") == "kernel":
                counts[ev.get("name", "")] += 1
            elif ev.get("name") == "cudaGraphLaunch":
                graph += 1
        ncounts = collections.Counter()
        for n, c in counts.items():
            ncounts[normalized(n)] += c
        rec = {"trace": str(path), "cudaGraphLaunch": graph, "families": {}}
        for fam, names in ids.items():
            exact = {n: counts[n] for n in names}
            norm = {normalized(n): ncounts[normalized(n)] for n in names}
            rec["families"][fam] = {"exact": exact, "normalized": norm}
        fr = rec["families"]
        for fam in ("prefill_full", "prefill_local"):
            if not any(v > 0 for v in fr[fam]["exact"].values()):
                failures.append(f"rank {rank}: no {fam} kernel observed (by name up to the layout: "
                                f"{any(v > 0 for v in fr[fam]['normalized'].values())})")
        need = {"decode_full": FULL_LAYERS * DECODE_STEPS, "decode_local": LOCAL_LAYERS * DECODE_STEPS}
        if same:
            need = {"decode_full": (FULL_LAYERS + LOCAL_LAYERS) * DECODE_STEPS}
        for fam, n in need.items():
            top = max(fr[fam]["exact"].values(), default=0)
            if top < n:
                failures.append(f"rank {rank}: the most frequent {fam} kernel occurs {top} times, fewer than {n} "
                                f"(by name up to the layout: {max(fr[fam]['normalized'].values(), default=0)})")
        if graph == 0:
            failures.append(f"rank {rank}: no cudaGraphLaunch during the profiled decode")
        ranks[rank] = rec
    facts = {"trace_dir": str(out), "server_pid": server_pid, "same_decode_kernel": same, "ranks": ranks,
             "identity_sets": ids, "prompts": {"single": len(long), "concurrent": [len(s) for s in short]}}
    return failures, facts


def summary(facts: dict) -> dict:
    return {r: {"cudaGraphLaunch": v["cudaGraphLaunch"],
                **{f: {"exact_max": max(d["exact"].values(), default=0),
                       "normalized_max": max(d["normalized"].values(), default=0)} for f, d in v["families"].items()}}
            for r, v in facts.get("ranks", {}).items()}
