#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The fidelity statistic's hidden bands and their draws, shared by both classes' `test_fidelity_kl` nodes."""
from __future__ import annotations

import glob
import gzip
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from glm53_protocols.fidelity import (  # noqa: E402
    _k3_term,  # noqa: F401 - the one statistic term, re-exported for the guards
    k3_per_sequence,
    parse_answer,
    reference_problems,
    request_logprobs,
    served_continuation_logprobs,  # noqa: F401 - the public callers' form, re-exported for the guards
)
from infb_verify.records import compact, write_record  # noqa: E402

FIDELITY_DIR = TESTS_DIR / "hidden" / "fidelity"
LONG_DIR = TESTS_DIR / "hidden" / "fidelity-long"
SERVED_DIR = Path(os.environ.get("INFB_FIDELITY_SERVED_DIR", "/logs/verifier/fidelity"))


def load_reference(directory: Path = FIDELITY_DIR) -> list[dict]:
    files = sorted(glob.glob(str(directory / "*.json")))
    refs = [json.load(open(f)) for f in files if not f.endswith("provenance.json")]
    assert refs, f"no reference sequences under {directory}"
    return refs


def store_served(label: str, refs: list[dict], served: list[list[float] | None], request: str,
                 append: bool = False) -> str:
    """One JSON row per sequence, full-precision floats, the request named. A record, never a verdict."""
    try:
        out_dir = SERVED_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "served-logprobs.jsonl.gz"
        with gzip.open(path, "at" if append else "wt", encoding="utf-8") as fh:
            for r, s in zip(refs, served):
                fh.write(json.dumps({"request": request, "id": r["id"], "kind": r["kind"],
                                     "cont_len": r["cont_len"], "served_logprobs": s}) + "\n")
        return str(path)
    except Exception as exc:  # noqa: BLE001 - a record
        return f"not written: {exc!r}"[:300]


def _read(base_url: str, refs: list[dict], routed_dp_rank: int | None = None) -> dict:
    """One draw: the request, its structural reading and, when every sequence is well formed, its statistic."""
    t0 = time.time()
    out: dict = {"started_at": t0, "request": {"endpoint": "/generate", "n_sequences": len(refs),
                                               "routed_dp_rank": routed_dp_rank,
                                               "sequence_ids": [r["id"] for r in refs]}}
    try:
        answer = request_logprobs(base_url, refs, routed_dp_rank)
    except requests.Timeout as exc:
        out.update(structural="not_answered", error=f"past the HTTP operation limit: {exc!r}"[:500])
    except requests.RequestException as exc:
        out.update(structural="not_answered", error=f"{type(exc).__name__}: {exc}"[:500])
    except ValueError as exc:
        out.update(structural="malformed", error=f"the answer is not JSON: {exc}"[:500])
    else:
        readings = parse_answer(refs, answer)
        bad = [{"id": r["id"], "problems": r["problems"], "expected_span": r["expected_span"],
                "expected_entries": r["expected_entries"], "observed_entries": r["observed_entries"]}
               for r in readings if r["problems"]]
        out["spans"] = [{"id": r["id"], "expected_span": r["expected_span"], "expected_entries": r["expected_entries"],
                         "observed_entries": r["observed_entries"]} for r in readings]
        out["values"] = [r["values"] for r in readings]
        if bad:
            out.update(structural="malformed", malformed=bad, raw_first_malformed=compact(
                next((a for r, a in zip(readings, answer if isinstance(answer, list) else []) if r["problems"]),
                     answer)))
        else:
            out["structural"] = "valid"
            k3 = k3_per_sequence(refs, out["values"])
            out["k3"] = {r["id"]: v for r, v in zip(refs, k3)}
            out["median_k3"] = statistics.median(k3)
            out["max_k3"] = max(k3)
            out["per_kind"] = {kind: statistics.median([v for r, v in zip(refs, k3) if r["kind"] == kind])
                               for kind in sorted({r["kind"] for r in refs})}
    out["elapsed_s"] = round(time.time() - t0, 3)
    return out


def fidelity_band(base_url: str, label: str, band: str, directory: Path, bar: float,
                  dp_ranks: tuple[int, ...] = ()) -> dict:
    """One band under the fixed draw policy. Returns the band's verdict with every reading; each reading is persisted
    before the verdict is formed."""
    refs = load_reference(directory)
    ref_bad = {r.get("id"): p for r in refs if (p := reference_problems(r))}
    if ref_bad:
        rec = {"status": "reference_fault", "band": band, "reference_problems": ref_bad}
        rec["record"] = write_record("fidelity", label, f"{band}-references", rec)
        return {"band": band, "status": "reference_fault", "reason": f"malformed reference records: {ref_bad}",
                "readings": []}
    readings = []
    status, reason = "candidate_fail", None
    for attempt in ("first", "redraw"):
        reading = _read(base_url, refs)
        reading.update(band=band, attempt=attempt, bar=bar, expected_positions="the last cont_len of each sequence")
        if reading["structural"] == "valid":
            reading["numerical"] = "pass" if float(reading["median_k3"]) < float(bar) else "above_bar"
            reading["status"] = "pass" if reading["numerical"] == "pass" else "candidate_fail"
        else:
            reading["status"] = "candidate_fail"
        reading["served_record"] = store_served(label if attempt == "first" else f"{label}-{attempt}", refs,
                                                reading.get("values") or [None] * len(refs), f"{band}:{attempt}")
        values = reading.pop("values", None)
        reading["record"] = write_record("fidelity", label, f"{band}-{attempt}", reading)
        reading["values_kept"] = values is not None
        readings.append(reading)
        if reading["structural"] != "valid":
            status = "candidate_fail"
            reason = (f"the {band} band's {attempt} draw is not a valid reading ({reading['structural']}): "
                      + (json.dumps(reading.get("malformed", [])[:2]) if reading.get("malformed")
                         else reading.get("error", "")))[:900]
            break
        if reading["numerical"] == "pass":
            status, reason = "pass", None
            break
        reason = f"the {band} band's {attempt} draw reads median k3 {reading['median_k3']:.6g}, not below its bar"
    if status != "pass" and len(readings) == 2 and readings[1]["structural"] == "valid":
        reason = (f"the {band} band: both draws fail its bar: first {readings[0]['median_k3']:.6g}, "
                  f"re-draw {readings[1]['median_k3']:.6g}")
    diagnostics = {}
    if readings and readings[0]["structural"] != "not_answered":
        for rank in dp_ranks:
            d = _read(base_url, refs, rank)
            diagnostics[f"dp_rank_{rank}"] = {k: d.get(k) for k in ("structural", "median_k3", "elapsed_s", "error")}
            store_served(label, refs, d.get("values") or [None] * len(refs), f"{band}:dp_rank_{rank}", append=True)
    return {"band": band, "status": status, "reason": reason, "readings": readings, "diagnostics": diagnostics}
