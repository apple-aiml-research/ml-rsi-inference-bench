#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Multi-token equivalence: what the teacher-forced correctness gate cannot see."""

from __future__ import annotations

import json
from pathlib import Path


def _texts(path: Path) -> list[str] | None:
    """The per-request generated texts, or None when the run was not saved detailed."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    got = doc.get("generated_texts")
    if not isinstance(got, list) or not got:
        return None
    return [t if isinstance(t, str) else "" for t in got]


def _first_divergence(a: str, b: str) -> int | None:
    """Index of the first differing character, or None when the strings are equal."""
    if a == b:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def compare(left: list[str], right: list[str]) -> dict:
    """Agreement between two aligned lists of generated texts."""
    n = min(len(left), len(right))
    if n == 0:
        return {"n": 0}
    identical = 0
    fracs: list[float] = []
    first_idxs: list[int] = []
    for a, b in zip(left[:n], right[:n]):
        idx = _first_divergence(a, b)
        if idx is None:
            identical += 1
            fracs.append(1.0)
        else:
            first_idxs.append(idx)
            denom = max(len(a), len(b), 1)
            fracs.append(idx / denom)
    out = {
        "n": n,
        "identical": identical,
        "identical_frac": round(identical / n, 4),
        "prefix_match_frac_mean": round(sum(fracs) / len(fracs), 4),
    }
    if first_idxs:
        out["first_divergence_chars_min"] = min(first_idxs)
        out["first_divergence_chars_median"] = sorted(first_idxs)[len(first_idxs) // 2]
    if len(left) != len(right):
        out["length_mismatch"] = [len(left), len(right)]
    return out


def read_from_raw_dir(raw_dir, workload_names, num_runs: int = 3) -> dict:
    """{workload: {profile: {cand_vs_pristine: [...], pristine_vs_self: [...]}}}."""
    raw = Path(raw_dir)
    out: dict = {}
    for wl in workload_names:
        profiles = sorted({
            p.name.rsplit("_meas", 1)[0].split(f"{wl}_cand_", 1)[-1]
            for p in raw.glob(f"bench_{wl}_{wl}_cand_*_meas*.json")
        })
        for prof in profiles:
            cand = [_texts(raw / f"bench_{wl}_{wl}_cand_{prof}_meas{i}.json") for i in range(num_runs)]
            pris = [_texts(raw / f"bench_{wl}_{wl}_pristine_{prof}_meas{i}.json") for i in range(num_runs)]
            cross = [compare(c, p) for c, p in zip(cand, pris) if c and p]
            self_ = []
            have = [t for t in pris if t]
            for i in range(len(have) - 1):
                self_.append(compare(have[i], have[i + 1]))
            if cross or self_:
                out.setdefault(wl, {})[prof] = {
                    "cand_vs_pristine": cross,
                    "pristine_vs_self": self_,
                }
    return out


def summarise(per_workload: dict) -> dict:
    """A short verdict-shaped summary that says what it does NOT establish."""
    lines: dict = {}
    for wl, profs in per_workload.items():
        for prof, d in profs.items():
            cross = d.get("cand_vs_pristine") or []
            self_ = d.get("pristine_vs_self") or []
            lines[f"{wl}/{prof}"] = {
                "cand_vs_pristine_identical_frac": [c.get("identical_frac") for c in cross],
                "pristine_vs_self_identical_frac": [c.get("identical_frac") for c in self_],
                "interpretable": prof == "c1",
                "note": (
                    "c1 has no batching, so a disagreement here can be about the patch"
                    if prof == "c1" else
                    "c>1: batch composition varies between runs and a hybrid model's recurrent "
                    "state depends on it, so a disagreement here is not attributable"
                ),
            }
    return {
        "per_point": lines,
        "gating": "records only -- a threshold needs the pristine-vs-self floor measured first",
    }
