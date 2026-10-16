#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The read protocols: denoise, image and step-1 reads against the author's
recordings; commit-encode replay, prefix branches and mixed batches against the candidate itself.

Structural checks come first (every read returns exactly L positions whose top log-probabilities
hold the target and the emitted token, all finite and <= 1e-6).

Each band's median of per-read means must be <= T_b, and each read's mean <= C_b, or <= the read's
own C_b,r where the policy carries one. The bars are frozen in policy.json.
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import statistics

from eval.diffusion_gemma.serving import reads as R

CONCURRENCY = 4
TP2_BANDS = (1024, 4096, 16384, 65536)


def select(data: dict, read_set: str, deployment: str) -> list[dict]:
    out = [r for r in data["reads"] if r["set"] == read_set]
    if deployment == "tp2" and read_set == "reads":
        out = [r for r in out if r["band"] in TP2_BANDS]
    return out


def band_key(row: dict) -> str:
    """A read's band for the band verdict: its committed-length band; the image reads are one band."""
    return "image" if str(row["id"]).startswith("image/") else str(row["band"])


def run(srv, data: dict, reads: list[dict], keep_lp: bool = False) -> list[dict]:
    """The public read protocol (eval/diffusion_gemma/serving/reads.py) at the node's concurrency; keep_lp keeps each
    read's served target log-probabilities in its row."""
    if not keep_lp:
        return R.run(srv, data, reads, concurrency=CONCURRENCY)
    with cf.ThreadPoolExecutor(CONCURRENCY) as ex:
        results = list(ex.map(lambda rd: R.send(srv, data["prompts"][rd["prompt"]], rd), reads))
    return [{"id": rd["id"], "band": rd["band"], "k3": res["k3"], "k3_all": res["k3_all"],
             "stat_positions": res.get("stat_positions"), "structural": res["structural"],
             "prompt_ids_match": res.get("prompt_ids_match"), "problems": res["problems"], "wall_s": res["wall_s"],
             "target_logprob": res.get("target_logprob")}
            for rd, res in zip(reads, results)]


def verdict(srv, data: dict, reads: list[dict], bars: dict | None, keep_lp: bool = False) -> dict:
    """The band verdict. `bars` = {band: {"T": float, "C": float}} or None (calibration: record only)."""
    first = run(srv, data, reads, keep_lp)
    by_band: dict[str, list[dict]] = {}
    for row in first:
        by_band.setdefault(band_key(row), []).append(row)
    rec = {"draws": {"1": first}, "bands": {}, "structural_failures": [r for r in first if not r["structural"]]}
    ok = not rec["structural_failures"]
    for band, rows in sorted(by_band.items(), key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0)):
        med = statistics.median(r["k3"] for r in rows)
        entry = {"n": len(rows), "median": med, "max": max(r["k3"] for r in rows),
                 "median_all_positions": statistics.median(r["k3_all"] for r in rows)}
        if bars is not None:
            T, C = bars[band]["T"], bars[band]["C"]
            c_read = bars[band].get("C_read") or {}
            resample = bars[band].get("resample", True)
            entry.update(T=T, C=C, resample=resample)
            if med > T and resample:  # one band redraw
                redo = run(srv, data, [rd for rd in reads if band_key(rd) == band])
                entry["redraw_median"] = statistics.median(r["k3"] for r in redo)
                entry["band_pass"] = entry["redraw_median"] <= T
                rec["draws"].setdefault("2", []).extend(redo)
            else:
                entry["band_pass"] = med <= T
            over = [r for r in rows if r["k3"] > c_read.get(r["id"], C)]
            entry["over"] = [{"id": r["id"], "k3": r["k3"], "C": c_read.get(r["id"], C)} for r in over]
            resent = []
            for r in over if resample else ():  # one re-send per failing read, in the scope only
                rd = next(x for x in reads if x["id"] == r["id"])
                again = R.send(srv, data["prompts"][rd["prompt"]], rd)
                resent.append({"id": r["id"], "k3": again["k3"], "pass": again["k3"] <= c_read.get(r["id"], C)})
            entry["resent"] = resent
            entry["read_pass"] = all(x["pass"] for x in resent) if resample else not over
            ok &= entry["band_pass"] and entry["read_pass"]
        rec["bands"][band] = entry
    rec["pass"] = bool(ok)
    return rec


def _increment_rows(srv, data: dict, reads: list[dict], author0: dict, two: list[dict] | None = None) -> list[dict]:
    """Per read, mean over its statistic positions of |(lp2 - lp1) served - (lp2 - lp1) author|: lp2 the two-step
    read (`two`, the node's own draw, or sent here), lp1 the one-step read of the same canvas, the author's lp2 the
    recording's step-1 values and lp1 its step-0 values (author0)."""
    two = two if two is not None else run(srv, data, reads, keep_lp=True)
    one = run(srv, data, [dict(rd, steps=1, author=author0[rd["id"]]) for rd in reads], keep_lp=True)
    rows = []
    for rd, r2, r1 in zip(reads, two, one):
        n, ok = int(rd.get("stat_positions", len(rd["targets"]))), r2["structural"] and r1["structural"]
        a2, a1 = rd["author"], author0[rd["id"]]
        lp2, lp1 = r2["target_logprob"], r1["target_logprob"]
        inc = sum(abs((lp2[i] - lp1[i]) - (a2[i] - a1[i])) for i in range(n)) / max(n, 1) if ok else math.inf
        rows.append({"id": rd["id"], "band": rd["band"], "n": n, "structural": ok, "increment": inc,
                     "k3_step1": r2["k3"], "k3_step0": r1["k3"], "problems": r2["problems"] + r1["problems"],
                     "lp2": lp2, "lp1": lp1})
    return rows


def increment(srv, data: dict, reads: list[dict], two: list[dict], author0: dict, bars: dict | None) -> dict:
    """The step-1 increment, beside the step-1 reads' own verdict: each canvas's one-step read
    through the same interface, and per read the increment statistic (_increment_rows), whose shared numerics offset
    cancels and whose self-conditioning pathway does not. The band verdict with the increment's own bars
    ({band: {"T", "C", "resample"}}, or None in calibration: record only); in the resample scope a failing band's two
    reads per canvas are re-drawn once and a failing read's are re-sent once."""
    rows = _increment_rows(srv, data, reads, author0, two)
    rec = {"draws": {"1": rows}, "bands": {}, "structural_failures": [r["id"] for r in rows if not r["structural"]]}
    ok = not rec["structural_failures"]
    by_band: dict[str, list[dict]] = {}
    for row in rows:
        by_band.setdefault(str(row["band"]), []).append(row)
    for band, rs in sorted(by_band.items(), key=lambda kv: int(kv[0])):
        med = statistics.median(r["increment"] for r in rs)
        entry = {"n": len(rs), "median": med, "max": max(r["increment"] for r in rs)}
        if bars is not None:
            T, C, resample = bars[band]["T"], bars[band]["C"], bars[band].get("resample", True)
            entry.update(T=T, C=C, resample=resample)
            if med > T and resample:  # one band redraw
                redo = _increment_rows(srv, data, [rd for rd in reads if str(rd["band"]) == band], author0)
                entry["redraw_median"] = statistics.median(r["increment"] for r in redo)
                entry["band_pass"] = entry["redraw_median"] <= T
                rec["draws"].setdefault("2", []).extend(redo)
            else:
                entry["band_pass"] = med <= T
            over = [r for r in rs if r["increment"] > C]
            entry["over"] = [{"id": r["id"], "increment": r["increment"], "C": C} for r in over]
            resent = []
            for r in over if resample else ():  # one re-send per failing read, in the scope only
                again = _increment_rows(srv, data, [rd for rd in reads if rd["id"] == r["id"]], author0)[0]
                resent.append({"id": r["id"], "increment": again["increment"], "pass": again["increment"] <= C})
            entry["resent"] = resent
            entry["read_pass"] = all(x["pass"] for x in resent) if resample else not over
            ok &= entry["band_pass"] and entry["read_pass"]
        rec["bands"][band] = entry
    rec["pass"] = bool(ok)
    return rec
