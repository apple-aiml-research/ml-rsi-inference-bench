#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Output shape: is the measured window real work, or degenerate continuation?"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

_NAME = re.compile(
    r"^bench_(?P<workload>.+?)_(?P=workload)_(?P<side>cand|pristine)"
    r"(?:_(?P<profile>[^_]+))?_meas(?P<meas>\d+)\.json$"
)

DEFAULT_K = 10


def _texts(path: Path) -> list[str] | None:
    """generated_texts, or None when --save-detailed was absent."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    got = doc.get("generated_texts")
    if not isinstance(got, list) or not got:
        return None
    return [t if isinstance(t, str) else "" for t in got]


def _grams(text: str, k: int) -> list[str]:
    w = text.split()
    if len(w) < k + 1:
        return []
    return [" ".join(w[i:i + k]) for i in range(len(w) - k + 1)]


def measure_one(text: str, k: int = DEFAULT_K) -> dict | None:
    """Repetition rate and distinct-k for a single completion."""
    g = _grams(text, k)
    if not g:
        return None
    seen: set[str] = set()
    repeated = 0
    for x in g:
        if x in seen:
            repeated += 1
        else:
            seen.add(x)
    return {
        "words": len(text.split()),
        "kgrams": len(g),
        "repetition_rate": round(repeated / len(g), 4),
        "distinct_k": round(len(seen) / len(g), 4),
    }


def _agg(vals: list[float]) -> dict:
    v = sorted(vals)
    n = len(v)

    def q(p):
        return round(v[min(n - 1, int(p / 100 * n))], 4)

    return {"n": n, "p25": q(25), "p50": q(50), "p75": q(75), "p90": q(90),
            "mean": round(st.mean(v), 4), "max": q(100)}


def read_from_raw_dir(raw_dir, workload_names, k: int = DEFAULT_K) -> dict:
    """{workload: {profile: {side: {...aggregates}}}} over every measured bench file."""
    wanted = set(workload_names)
    out: dict = {}
    for p in sorted(Path(raw_dir).glob("bench_*_meas*.json")):
        m = _NAME.match(p.name)
        if not m or m.group("workload") not in wanted:
            continue
        texts = _texts(p)
        node = (out.setdefault(m.group("workload"), {})
                   .setdefault(m.group("profile") or "_all", {})
                   .setdefault(m.group("side"), {"files": 0, "blind_files": 0,
                                                 "_rep": [], "_dis": [], "_wrd": []}))
        node["files"] += 1
        if texts is None:
            node["blind_files"] += 1
            continue
        for t in texts:
            one = measure_one(t, k)
            if one is None:
                continue
            node["_rep"].append(one["repetition_rate"])
            node["_dis"].append(one["distinct_k"])
            node["_wrd"].append(one["words"])

    for wl, byprof in out.items():
        for prof, bysides in byprof.items():
            for side, node in bysides.items():
                rep, dis, wrd = node.pop("_rep"), node.pop("_dis"), node.pop("_wrd")
                node["k"] = k
                node["completions"] = len(rep)
                if rep:
                    node["repetition_rate"] = _agg(rep)
                    node["distinct_k"] = _agg(dis)
                    node["words"] = _agg([float(x) for x in wrd])
                    node["share_above_30pct_repetition"] = round(
                        sum(1 for x in rep if x > 0.30) / len(rep), 4)
                else:
                    node["status"] = ("blind -- no generated_texts (is --save-detailed set?)"
                                      if node["blind_files"] else "no completions long enough for k")
    return out


def summarise(per_workload: dict) -> dict:
    """One flat block per (workload, profile, side), plus the reading a human needs."""
    flat: dict = {}
    worst = 0.0
    blind = 0
    for wl, byprof in per_workload.items():
        for prof, bysides in byprof.items():
            for side, node in bysides.items():
                key = f"{wl}/{prof}/{side}"
                rr = node.get("repetition_rate")
                flat[key] = {
                    "completions": node.get("completions", 0),
                    "blind_files": node.get("blind_files", 0),
                    "repetition_p50": (rr or {}).get("p50"),
                    "repetition_p90": (rr or {}).get("p90"),
                    "repetition_mean": (rr or {}).get("mean"),
                    "distinct_k_p50": (node.get("distinct_k") or {}).get("p50"),
                    "words_p50": (node.get("words") or {}).get("p50"),
                    "share_above_30pct_repetition": node.get("share_above_30pct_repetition"),
                    "status": node.get("status"),
                }
                blind += node.get("blind_files", 0)
                if rr and rr.get("mean") is not None:
                    worst = max(worst, rr["mean"])
    return {
        "per_point": flat,
        "worst_mean_repetition_rate": round(worst, 4) if flat else None,
        "blind_files": blind,
        "gates": "none -- recorded only",
        "interpretation": (
            "These numbers say how much of the --ignore-eos window is degenerate continuation rather "
            "than task work. They exist to CHOOSE output_len: a high mean repetition rate means the "
            "window is longer than the model has anything to say, so acceptance length and throughput "
            "are both being measured on looping text -- and an acceptance length read over a window "
            "the model finished answering early in will sit well below the figure its publisher "
            "reports, because the positions being scored are no longer task work. No threshold is "
            "applied here -- picking one before the distribution is known is the same error as "
            "fixing an acceptance tolerance before that metric's run-to-run noise is characterised."
        ),
    }
