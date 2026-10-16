#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Execution attribution, the serving side, public."""
from __future__ import annotations

import concurrent.futures as cf
import gzip
import json
import time
from collections import Counter
from pathlib import Path

from .recipes import SERVED_NAME

CANVASES_W1 = 4


def traces(root: Path, since: float) -> list[Path]:
    """The engine ranks' kernel traces written since `since` (the API server's CPU trace excluded)."""
    return sorted(p for p in Path(root).rglob("*.json*") if p.stat().st_mtime >= since and "trace" in p.name
                  and "async_llm" not in p.name and "rank" in p.name)


def events(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw).get("traceEvents", [])


def window(srv, root: Path, work, ranks: int, trace_wait_s: float) -> tuple[dict[str, Counter], dict]:
    """Profile `work()`; (per-rank kernel counts, the export timing)."""
    t0 = time.time()
    srv.post("/start_profile", {})
    work()
    t_stop = time.monotonic()
    srv.post("/stop_profile", {}, timeout=trace_wait_s)
    files: list[Path] = []
    while time.monotonic() < t_stop + trace_wait_s:
        files = traces(root, t0)
        if len(files) >= ranks:
            time.sleep(5)  # let every rank finish writing
            files = traces(root, t0)
            break
        time.sleep(2)
    timing = {"export_s": round(time.monotonic() - t_stop, 1), "ranks_found": len(files),
              "bytes": sum(f.stat().st_size for f in files)}
    per_rank: dict[str, Counter] = {}
    for f in files:
        c: Counter = Counter()
        for ev in events(f):
            cat = ev.get("cat", "")
            if cat == "kernel":
                c[ev.get("name", "")] += 1
            elif cat in ("cuda_runtime", "cuda_driver") and "GraphLaunch" in ev.get("name", ""):
                c["cudaGraphLaunch"] += 1
        per_rank[f.name] = c
    return per_rank, timing


def witness(srv, root: Path, identity: dict, window1_prompts: tuple[list[int], list[list[int]]], read_body: dict,
            ranks: int, trace_wait_s: float = 300) -> dict:
    """identity: {family: {layer_kind: [kernel names]}, "exclusions": pattern}; window1_prompts: (the 6,144-token
    prompt, three 2,000-token prompts); read_body: one /v1/completions read at c = 4,096."""
    long_prompt, short_prompts = window1_prompts

    def w1():
        srv.post("/v1/completions", {"model": SERVED_NAME, "prompt": list(long_prompt), "max_tokens": 256,
                                     "ignore_eos": True, "temperature": 1.0})
        with cf.ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda p: srv.post("/v1/completions", {"model": SERVED_NAME, "prompt": list(p), "max_tokens": 256,
                                                               "ignore_eos": True, "temperature": 1.0}), short_prompts))

    win1, t1 = window(srv, root, w1, ranks, trace_wait_s)
    win2, t2 = window(srv, root, lambda: srv.post("/v1/completions", read_body), ranks, trace_wait_s)
    fam = {f: {k for kinds in identity[f].values() for k in kinds} for f in ("encode", "denoise", "mixed")}
    rec = {"ranks_w1": len(win1), "ranks_w2": len(win2), "trace_timing": [t1, t2], "trace_wait_s": trace_wait_s,
           "identity_set_exclusions": identity.get("exclusions"), "window1": [], "window2": []}
    ok = len(win1) == ranks and len(win2) == ranks and all(fam.values())
    for rank, c in win1.items():
        present = {f: sum(c[k] for k in ks) for f, ks in fam.items()}
        top = max((c[k] for k in fam["denoise"]), default=0)
        row = {"rank": rank, "present": present, "top_denoise": top, "graph_launch": c["cudaGraphLaunch"],
               "pass": all(v > 0 for v in present.values()) and top >= 30 * CANVASES_W1 and c["cudaGraphLaunch"] > 0}
        rec["window1"].append(row)
        ok &= row["pass"]
    for rank, c in win2.items():
        n = sum(c[k] for k in fam["denoise"])
        rec["window2"].append({"rank": rank, "denoise_kernels": n, "pass": n >= 30})
        ok &= n >= 30
    rec["pass"] = bool(ok)
    return rec
