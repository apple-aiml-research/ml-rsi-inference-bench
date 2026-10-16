#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 cfg gates.'
from __future__ import annotations

import math

POLICY = {


    "COMBINED_PSNR_DB": 40.0,

    "GUIDANCE1_PSNR_DB": 45.0,
}
POLICY_STATE = "specified"
POLICY_BASIS = {'COMBINED_PSNR_DB': 'Required bound for COMBINED_PSNR_DB.', 'GUIDANCE1_PSNR_DB': 'Required bound for GUIDANCE1_PSNR_DB.'}


def psnr(a, b) -> float | None:
    'uint8 psnr (100.0 for identical images); None -- no PSNR credit -- when either image is missing, empty or'


    if a is None or b is None:
        return None
    import numpy as np

    try:
        x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if x.shape != y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean((x - y) ** 2))
    if not math.isfinite(mse):
        return None
    return 100.0 if mse == 0.0 else min(100.0, 10.0 * math.log10(255.0 ** 2 / mse))


def lines_in_window(events: list, t0: float, t1: float) -> list:
    return [e for e in events if e.get("event") == "dit_cfg" and e.get("t") is not None and t0 <= float(e["t"]) <= t1]


def _by_rank(lines: list) -> dict:
    out: dict = {}
    for ln in lines:
        out.setdefault(ln.get("rank"), []).append(ln)
    return out


def _cond(ln) -> tuple:
    return (ln.get("ehs"), ln.get("pooled"))


def _conditionings(line):
    ' conditionings.'
    count = line.get("rows")
    rows = line.get("conditioning_rows")
    if isinstance(count, int) and count > 0 and isinstance(rows, list) and len(rows) == count:
        if all(isinstance(row, dict) and row.get("ehs") is not None and
               row.get("pooled") is not None for row in rows):
            return tuple((row["ehs"], row["pooled"]) for row in rows)
        return ()
    if count == 1 and line.get("ehs") is not None and line.get("pooled") is not None:
        return (_cond(line),)
    return ()


def cfg_parallel_is_real(p7: list, s7: list, images: dict, steps: int, extra: dict | None = None) -> dict:
    """`p7` / `s7`: the dit_cfg lines of the guidance-7 request on P and on S; `images`: p7 / s7 / p1 / s1 -> uint8
    arrays or None; `steps`: the request's denoising steps."""
    P = POLICY
    d: dict = {"policy_state": POLICY_STATE, "steps": steps, **(extra or {})}
    reasons: list = []
    ranks = _by_rank(p7 or [])
    d["p7_forwards_per_rank"] = {str(r): len(v) for r, v in ranks.items()}
    d["s7_forwards"] = len(s7 or [])
    if len(ranks) < 2 or not s7:
        return {"verdict": "not_evaluated", "reasons": [
            f"recorded: the CFG-parallel request ran on {len(ranks)} rank(s) and the serial one recorded "
            f"{len(s7 or [])} forward(s) -- the profile did not occur (two CFG ranks, a serial reference)"],
                "details": d}
    clauses: dict = {}
    serial_set = {cond for ln in s7 for cond in _conditionings(ln)}
    per_rank = {r: {cond for ln in v for cond in _conditionings(ln)} for r, v in ranks.items()}
    missing_rows = any(not _conditionings(ln) for ln in [*p7, *s7])
    serial_work = sum(len(_conditionings(ln)) for ln in s7)
    d["serial_rows"] = serial_work
    first_serial = next(iter(_conditionings(s7[0])), None)
    d["serial_first_branch_rank"] = next((r for r, c in per_rank.items() if c == {first_serial}), None)
    d["conditionings_per_rank"] = {str(r): sorted(map(str, c)) for r, c in per_rank.items()}
    d["serial_conditionings"] = sorted(map(str, serial_set))
    one_each = all(len(c) == 1 for c in per_rank.values()) and all(
        len(_conditionings(ln)) == 1 for ln in p7)
    union = set().union(*per_rank.values())
    distinct = one_each and len(union) == len(per_rank)
    counts_ok = all(len(v) == steps for v in ranks.values())
    matches_serial = union == serial_set
    clauses["split"] = None if missing_rows else bool(one_each and distinct and counts_ok and matches_serial and serial_work == 2 * steps)
    if missing_rows:
        reasons.append("split: not_evaluated (conditioning rows were not independently observed)")
    if clauses["split"] is False:
        why = []
        if not one_each:
            why.append("a rank computed more than one CFG branch (replicated compute)")
        if one_each and not distinct:
            why.append("both ranks computed the same branch")
        if not counts_ok:
            why.append(f"forwards per rank {d['p7_forwards_per_rank']} != {steps} steps")
        if serial_work != 2 * steps:
            why.append(f"serial CFG row work {serial_work} != {2 * steps}")
        if not matches_serial:
            why.append("the ranks' conditionings are not the serial run's two branches")
        reasons.append("split: " + "; ".join(why))
    if one_each and distinct and len(serial_set) == 2:

        first = _conditionings(s7[0])
        d["serial_first_row_rank"] = next((r for r, c in per_rank.items()
                                           if first and first[0] in c), None)

    rises = {}
    for r, v in ranks.items():
        vals = [ln.get("cfg_coll") for ln in v if ln.get("cfg_coll") is not None]
        rises[str(r)] = (max(vals) - min(vals)) if vals else None
    d["cfg_collectives_per_rank"] = rises
    if any(v is None for v in rises.values()):
        clauses["collective"] = None
        reasons.append("collective: not_evaluated (no CFG-group collective count recorded)")
    else:
        clauses["collective"] = all(v >= steps - 1 for v in rises.values())
        if not clauses["collective"]:
            reasons.append(f"collective: the CFG group carried {rises} collectives per rank across the request's "
                           f"forwards (< {steps - 1}, one per step): the branches were not combined across the group")

    m7 = psnr((images or {}).get("p7"), (images or {}).get("s7"))
    m1 = psnr((images or {}).get("p1"), (images or {}).get("s1"))
    d["psnr"] = {"p7_vs_s7": m7, "p1_vs_s1": m1}
    clauses["combined"] = None if m7 is None else m7 >= P["COMBINED_PSNR_DB"]
    if clauses["combined"] is None:
        reasons.append("combined: not_evaluated (an image is missing, empty or non-finite: no PSNR credit)")
    elif not clauses["combined"]:
        reasons.append(f"combined: the CFG-parallel image reads {m7:.2f} dB against the serial one (< "
                       f"{P['COMBINED_PSNR_DB']}): a branch is lost or mis-scaled in the combine")
    if (images or {}).get("p1") is None:
        clauses["guidance1"] = False
        reasons.append("guidance1: the request without guidance returned no image on the CFG-parallel server")
    elif m1 is None:
        clauses["guidance1"] = None
        reasons.append("guidance1: not_evaluated (an image of the pair is empty or non-finite, or the serial one is "
                       "missing: no PSNR credit)")
    else:
        clauses["guidance1"] = m1 >= P["GUIDANCE1_PSNR_DB"]
        if not clauses["guidance1"]:
            reasons.append(f"guidance1: {m1:.2f} dB against the serial guidance-1 image (< {P['GUIDANCE1_PSNR_DB']})")
    from reference_verdict import independent_clauses
    reference_clauses, reference_reasons = independent_clauses(
        d.get("reference_checks"), ("p7", "p1", "s7", "s1"), "bf16")
    clauses.update(reference_clauses)
    reasons.extend(reference_reasons)
    d["clauses"] = clauses
    if any(v is False for v in clauses.values()):
        return {"verdict": "fail", "reasons": reasons, "details": d}
    if any(v is None for v in clauses.values()):
        return {"verdict": "not_evaluated", "reasons": reasons, "details": d}
    return {"verdict": "pass", "reasons": reasons or ["every clause held"], "details": d}
