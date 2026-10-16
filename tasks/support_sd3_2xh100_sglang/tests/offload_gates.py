#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 offload gates utilities.'


from __future__ import annotations

import math

POLICY = {

    "INSTRUMENT_MIN_FRAC": 0.99,


    "MAX_RESIDENT_FRAC": 0.5,

    "STREAM_MIN_FRAC": 0.99,

    "PEAK_SAVING_FRAC": 0.5,


    "MATCH_PSNR_DB": 50.0,
    "REPEAT_PSNR_DB": 50.0,
}

POLICY_STATE = "specified"
POLICY_BASIS = {'MAX_RESIDENT_FRAC': 'Required bound for MAX_RESIDENT_FRAC.', 'STREAM_MIN_FRAC': 'Required bound for STREAM_MIN_FRAC.', 'PEAK_SAVING_FRAC': 'Required bound for PEAK_SAVING_FRAC.', 'MATCH_PSNR_DB': 'Required bound for MATCH_PSNR_DB.', 'REPEAT_PSNR_DB': 'Required bound for REPEAT_PSNR_DB.'}


def psnr(a, b) -> float | None:
    """uint8 psnr (100.0 for identical images), None when either is missing or the shapes differ."""
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
    """Physical records and relevant recorder gaps for this request window.

    A per-worker cap stays in effect after its first overflow. Keep its receipt
    even if this window has some complete forward records, so that a prefix of
    a request can never stand in for the whole request. An explicitly foreign
    process/rank or a cap reached only after the window does not taint it.
    """
    out = [e for e in events if e.get("event") == "dit_residency" and e.get("t0") is not None
           and t0 <= float(e["t0"]) <= t1]
    rows = sorted(out, key=lambda e: (e.get("rank") or 0, e.get("n") or 0))
    for event in events:
        if event.get("event") not in ("dit_residency_truncated", "truncated"):
            continue
        try:
            at = float(event["t"])
        except (KeyError, TypeError, ValueError, OverflowError):
            at = None
        if at is not None and math.isfinite(at) and at > t1:
            continue


        def possibly_same_worker(row):
            for key in ("pid", "rank"):
                x, y = event.get(key), row.get(key)
                if type(x) is int and type(y) is int and x != y:
                    return False
            return True
        if not out or any(possibly_same_worker(row) for row in out):
            rows.append(event)
    return rows


def _byte_count(value) -> bool:
    """Recorder byte counts are finite, nonnegative integers (including exact JSON floats)."""
    if type(value) is int:
        return value >= 0
    return type(value) is float and math.isfinite(value) and value >= 0 and value.is_integer()


def _record_problem(lines: list, n_blocks: int | None = None) -> str | None:
    """No forward may silently disappear from the per-block comparison.

    A partial hook record or corrupt byte count is insufficient instrument
    evidence. It cannot establish offload and is not an algorithm failure.
    """
    if not lines:
        return "no dit_residency reading for the request"
    for index, ln in enumerate(lines):
        if not isinstance(ln, dict):
            return f"forward {index}: residency record is not an object"
        if ln.get("event") in ("dit_residency_truncated", "truncated"):
            return (f"instrument: {ln['event']} for pid={ln.get('pid')} rank={ln.get('rank')}; "
                    "complete physical residency for this request is not established")
        n = ln.get("n_blocks")
        if type(n) is not int or n < 2:
            return f"forward {index}: the DiT's block list was not found (n_blocks < 2 or invalid)"
        if n_blocks is None:
            n_blocks = n
        if n != n_blocks:
            return f"forward {index}: n_blocks {n} differs from the resident engine's {n_blocks}"
        for key in ("self", "total"):
            values = ln.get(key)
            if not isinstance(values, (list, tuple)) or len(values) != n:
                return f"forward {index}: {key} does not record all {n} block starts"
            if not all(_byte_count(v) for v in values):
                return f"forward {index}: {key} contains invalid residency byte counts"
        if any(total < own for own, total in zip(ln["self"], ln["total"])):
            return f"forward {index}: total residency is smaller than a running block's own residency"
        peak = ln.get("mem_max")
        if peak is not None and not _byte_count(peak):
            return f"forward {index}: mem_max is not a valid allocator byte count"
    return None


def _full_size(b_lines: list) -> tuple[list | None, int | None]:
    """Per-block resident bytes on the resident engine (its first complete forward) and their sum."""
    for ln in b_lines:
        s = ln.get("self") or []
        if len(s) == (ln.get("n_blocks") or 0) and len(s) >= 2:
            return list(s), int(sum(s))
    return None, None


def _max_frac(lines: list, full: int) -> float | None:
    vals = [t for ln in lines for t in (ln.get("total") or [])]
    if not vals or not full:
        return None
    return max(vals) / full


def _peak(lines: list) -> int | None:
    vals = [ln.get("mem_max") for ln in lines if ln.get("mem_max") is not None]
    return max(vals) if vals else None


def dit_layerwise_offload_is_real(a1: list, a2: list, b1: list, images: dict, extra: dict | None = None) -> dict:
    """The verdict. `a1` / `a2` / `b1`: the dit_residency lines of each request's window; `images`: {"a1", "a2",
    "b1"} -> uint8 arrays (or None). Returns {"verdict": pass|fail|not_evaluated, "reasons": [...], "details": {...}}."""
    P = POLICY
    d: dict = {"policy_state": POLICY_STATE, **(extra or {})}
    reasons: list = []


    n_blocks = None
    for key, lines in (("b1", b1), ("a1", a1), ("a2", a2)):
        problem = _record_problem(lines, n_blocks)
        if problem is not None:
            return {"verdict": "not_evaluated", "reasons": [f"instrument: {key}: {problem}"], "details": d}
        if key == "b1":
            n_blocks = lines[0]["n_blocks"]
    per_block, full = _full_size(b1)
    d["block_bytes_resident_engine"] = full
    if not full:
        return {"verdict": "not_evaluated", "reasons": ["the resident engine's block bytes were not measured"],
                "details": d}
    b_frac_min = min((t / full for ln in b1 for t in (ln.get("total") or [])), default=None)
    d["instrument_min_frac_resident_engine"] = b_frac_min
    if b_frac_min is None or b_frac_min < P["INSTRUMENT_MIN_FRAC"]:
        return {"verdict": "not_evaluated", "reasons": [
            f"instrument: the resident engine's block starts saw {b_frac_min} of the block bytes resident (< "
            f"{P['INSTRUMENT_MIN_FRAC']}); the recorder cannot see where this DiT keeps its weights"], "details": d}
    clauses: dict = {}

    fa1, fa2 = _max_frac(a1, full), _max_frac(a2 or [], full)
    d["max_resident_frac"] = {"a1": fa1, "a2": fa2}
    clauses["offloaded"] = fa1 is not None and fa1 <= P["MAX_RESIDENT_FRAC"]
    if not clauses["offloaded"]:
        reasons.append(f"offloaded: at a block start of the offload request {fa1} of the block bytes were resident "
                       f"(> {P['MAX_RESIDENT_FRAC']}) -- the switch did not offload, or layers were never released")

    worst = None
    for ln in a1:
        s = ln.get("self") or []
        if len(s) != len(per_block):
            continue
        for own, ref in zip(s, per_block):
            if ref:
                worst = own / ref if worst is None else min(worst, own / ref)
    d["stream_min_frac"] = worst
    clauses["streamed"] = worst is not None and worst >= P["STREAM_MIN_FRAC"]
    if not clauses["streamed"]:
        reasons.append(f"streamed: a block started with {worst} of its own weights resident (< {P['STREAM_MIN_FRAC']})"
                       " -- it ran from a placeholder or off the GPU" if worst is not None else
                       "streamed: no offload forward recorded every block (block lists differ between the engines)")

    pa, pb = _peak(a1), _peak(b1)
    d["sampled_peak_bytes"] = {"offload": pa, "resident": pb,
                               "saving": (pb - pa) if pa is not None and pb is not None else None}
    if pa is None or pb is None:
        clauses["memory"] = None
        reasons.append("memory: not_evaluated (no allocator reading on one of the engines)")
    else:
        clauses["memory"] = pa <= pb - P["PEAK_SAVING_FRAC"] * full
        if not clauses["memory"]:
            reasons.append(f"memory: the offload request's sampled peak {pa} is not {P['PEAK_SAVING_FRAC']} x the block "
                           f"bytes ({full}) under the resident one's {pb}")

    m, r = psnr((images or {}).get("a1"), (images or {}).get("b1")), psnr((images or {}).get("a2"), (images or {}).get("a1"))
    d["psnr"] = {"a1_vs_b1": m, "a2_vs_a1": r}
    clauses["matches"] = None if m is None else m >= P["MATCH_PSNR_DB"]
    if clauses["matches"] is None:
        reasons.append("matches: not_evaluated (an image is missing)")
    elif not clauses["matches"]:
        reasons.append(f"matches: the offload image reads {m:.2f} dB against the resident engine's (< "
                       f"{P['MATCH_PSNR_DB']}) -- moving the weights changed the result")
    rep_off = fa2 is not None and fa2 <= P["MAX_RESIDENT_FRAC"]
    clauses["repeat"] = None if r is None else (r >= P["REPEAT_PSNR_DB"] and rep_off)
    if clauses["repeat"] is None:
        reasons.append("repeat: not_evaluated (the repeated request's image is missing)")
    elif not clauses["repeat"]:
        reasons.append(f"repeat: the repeated offload request reads {r:.2f} dB against the first and offloaded="
                       f"{rep_off} -- the offload state is not restored between requests")
    from reference_verdict import independent_clauses
    reference_clauses, reference_reasons = independent_clauses(
        d.get("reference_checks"), ("a1", "a2", "b1"), "bf16")
    clauses.update(reference_clauses)
    reasons.extend(reference_reasons)
    d["clauses"] = clauses
    if any(v is False for v in clauses.values()):
        return {"verdict": "fail", "reasons": reasons, "details": d}
    if any(v is None for v in clauses.values()):
        return {"verdict": "not_evaluated", "reasons": reasons, "details": d}
    return {"verdict": "pass", "reasons": reasons or ["every clause held"], "details": d}
