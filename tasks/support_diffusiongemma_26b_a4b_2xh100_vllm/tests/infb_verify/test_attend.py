#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The `attend` group: two collections, then six verdict nodes.

Correctness collection: every private case of tests/hidden/attend_cases.json through the candidate's
/app/diffusion_gemma_attend.py, with the policies of the attend contract and the witness. Timing and
identity collection: the timing families (measure.py) and the identity sets (binding.py). The
comparator's timings come from the reference phase (fa4_arm: the oracle adapter over the isolated
reference tree), recorded before any candidate code was imported.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from infb_verify import common

IMPL = os.environ.get("INFB_ATTEND_IMPL", "/app/diffusion_gemma_attend.py")
REF_TIMING = Path(os.environ.get("INFB_ATTEND_REFERENCE", "/logs/verifier/efficiency/attend_reference.json"))
RATIO = 0.8
_CACHE: dict = {}


def correctness() -> dict:
    if "correctness" not in _CACHE:
        import torch
        from eval.diffusion_gemma.attend import check

        impl = check.load_impl(IMPL)
        cases = common.hidden("attend_cases.json")["cases"]
        records = []
        for case in cases:
            try:
                records.append(check.run_case(case, impl, torch.device("cuda")))
            except (SystemExit, Exception) as exc:  # noqa: BLE001 -- any failure is the case's, recorded
                records.append({"id": case["id"], "pass": False, "error": f"{type(exc).__name__}: {exc}"[:2000]})
            torch.cuda.empty_cache()
        _CACHE["correctness"] = {"records": records, "cases": {c["id"]: c for c in cases}}
        common.record("attend_correctness", {"records": records})
    return _CACHE["correctness"]


def timing() -> dict:
    if "timing" not in _CACHE:
        from eval.diffusion_gemma.attend import binding, check, measure
        import torch

        impl = check.load_impl(IMPL)
        ns = json.loads(REF_TIMING.read_text()).get("n") if REF_TIMING.exists() else None
        coll = measure.collection(impl, ns)
        calls = {}
        for fam in ("encode", "denoise", "mixed"):
            for lk in ("sliding", "full"):
                x = check.Inputs(measure.case(fam, 4096, lk), impl, torch.device("cuda"))
                calls[(fam, lk)] = x.call
        ids = binding.identity_sets(calls)
        common.record("attend_binding", {**ids, "exclusions": binding.exclusions()})
        _CACHE["timing"] = coll
        common.record("attend_timing", coll)
    return _CACHE["timing"]


def _cases(kind: str) -> list[dict]:
    c = correctness()
    out = []
    for rec in c["records"]:
        case = c["cases"][rec["id"]]
        kinds = {s["kind"] for s in case["sequences"]}
        if kind == "encode" and kinds == {"encode"}:
            out.append(rec)
        elif kind == "denoise_mixed" and "denoise" in kinds:
            out.append(rec)
    return out


def test_attend_mask_witness():
    c = correctness()
    probes = [r for r in c["records"] if "witness" in r]
    readings = {rd for r in probes for rd in r["witness"]["fits"]}
    consistent = any(all(rd in r["witness"]["fits"] for r in probes) for rd in ("w1", "w"))
    rec = {"probe_cases": len(probes), "failing": [r["id"] for r in probes if not r["witness"]["pass"]],
           "readings_seen": sorted(readings), "pass": bool(probes) and consistent}
    common.record("test_attend_mask_witness", rec)
    assert rec["pass"], rec


def _values(kind):
    recs = _cases(kind)
    failing = [r["id"] for r in recs if not r["pass"] and "witness" not in r]
    failing += [r["id"] for r in recs if "witness" in r and not r.get("accuracy")]
    rec = {"cases": len(recs), "failing": failing, "pass": bool(recs) and not failing}
    return rec


def test_attend_values_encode():
    rec = _values("encode")
    common.record("test_attend_values_encode", rec)
    assert rec["pass"], rec


def test_attend_values_denoise_mixed():
    rec = _values("denoise_mixed")
    common.record("test_attend_values_denoise_mixed", rec)
    assert rec["pass"], rec


def _efficiency(family: str):
    coll = timing()
    ref = json.loads(REF_TIMING.read_text())["collection"] if REF_TIMING.exists() else None
    rows, ok = [], True
    for key, cand in coll.items():
        if not key.startswith(family + "/"):
            continue
        if ref is None:
            rows.append({"context": key, "candidate": cand, "reference": None})
            ok &= common.CALIBRATION
            continue
        r = ref[key]
        g, w = r["gpu_s"] / cand["gpu_s"], r["wall_s"] / cand["wall_s"]
        rows.append({"context": key, "gpu_ratio": g, "wall_ratio": w, "pass": g >= RATIO and w >= RATIO})
        ok &= g >= RATIO and w >= RATIO
    return {"rows": rows, "pass": bool(ok)}


def test_attend_encode_efficiency():
    rec = _efficiency("encode")
    common.record("test_attend_encode_efficiency", rec)
    assert rec["pass"], rec


def test_attend_denoise_efficiency():
    rec = _efficiency("denoise")
    common.record("test_attend_denoise_efficiency", rec)
    assert rec["pass"], rec


def test_attend_mixed_efficiency():
    rec = _efficiency("mixed")
    common.record("test_attend_mixed_efficiency", rec)
    assert rec["pass"], rec
