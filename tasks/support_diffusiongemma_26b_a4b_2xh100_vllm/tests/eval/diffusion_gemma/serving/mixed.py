#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Mixed-batch reads, public."""
from __future__ import annotations

import concurrent.futures as cf
import math
import threading
import time

from ._pairs import cut, pair, read
from .recipes import SERVED_NAME


def mixed(srv, data: dict, reads: list[dict], load_prompts: list[list[int]], T: float | None = None) -> dict:
    stop = threading.Event()

    def load(prompt):
        while not stop.is_set():
            srv.post("/v1/completions", {"model": SERVED_NAME, "prompt": prompt, "max_tokens": 1024, "ignore_eos": True,
                                         "temperature": 1.0}, timeout=1800)

    loaded = []
    with cf.ThreadPoolExecutor(len(load_prompts)) as ex:
        futs = [ex.submit(load, p) for p in load_prompts]
        time.sleep(5)
        for r in reads:
            p = data["prompts"][r["prompt"]]
            loaded.append(read(srv, p["ids"], r["seed"], r["targets"], r["steps"]))
        stop.set()
        for f in futs:
            f.result()
    rows, ok = [], True
    for r, a in zip(reads, loaded):
        p = data["prompts"][r["prompt"]]
        alone = read(srv, p["ids"], r["seed"], r["targets"], r["steps"])
        stat = pair(a, alone, r.get("stat_positions", cut(r["targets"])))
        good = math.isfinite(stat) and (T is None or stat <= T)
        rows.append({"id": r["id"], "k3": stat, "pass": good})
        ok &= good
    return {"rows": rows, "T": T, "pass": bool(ok)}
