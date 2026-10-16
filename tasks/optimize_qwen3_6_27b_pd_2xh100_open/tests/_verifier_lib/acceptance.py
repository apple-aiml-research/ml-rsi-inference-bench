#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Acceptance length: read it from the serve logs, and gate on the half that needs no threshold."""

from __future__ import annotations

import os
import re
from pathlib import Path

_SPEC_LINE = re.compile(r"SpecDecoding metrics:")
_MEAN = re.compile(r"Mean acceptance length:\s*([0-9.]+)")
_ACCEPTED = re.compile(r"Accepted:\s*(\d+)\s*tokens")
_DRAFTED = re.compile(r"Drafted:\s*(\d+)\s*tokens")
_PER_POS = re.compile(r"Per-position acceptance rate:\s*([0-9.,\s]+?),\s*Avg")

_LOG_NAME = re.compile(
    r"^vllm_serve_(?P<workload>.+)_(?P<side>cand|pristine)_(?P<port>\d+)\.log$"
)

_SGLANG_LOG_NAME = re.compile(
    r"^sglang_serve_(?P<workload>.+)_(?P<side>cand|pristine)_(?P<port>\d+)\.log$"
)
_SGLANG_ACCEPT_LEN = re.compile(r"accept len:\s*([0-9.]+)")
_SGLANG_ACCEPT_RATE = re.compile(r"accept rate:\s*([0-9.]+)")


def parse_log(path: Path) -> dict:
    """Exact aggregate over every SpecDecoding interval in one serve log."""
    accepted_total = 0
    drafted_total = 0
    drafts_total = 0.0
    per_pos_last: list[float] = []
    per_pos_weighted: list[float] = []
    intervals = 0
    malformed = 0

    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not _SPEC_LINE.search(line):
            continue
        m_mean, m_acc, m_drf = _MEAN.search(line), _ACCEPTED.search(line), _DRAFTED.search(line)
        if not (m_mean and m_acc and m_drf):
            malformed += 1
            continue
        mean = float(m_mean.group(1))
        accepted = int(m_acc.group(1))
        drafted = int(m_drf.group(1))
        intervals += 1
        accepted_total += accepted
        drafted_total += drafted
        m_pos = _PER_POS.search(line)
        if m_pos:
            per_pos_last = [float(x) for x in m_pos.group(1).split(",") if x.strip()]
        if mean > 1.0:
            _nd = accepted / (mean - 1.0)
        elif drafted > 0 and per_pos_last:
            _nd = drafted / len(per_pos_last)
        else:
            _nd = 0.0
        drafts_total += _nd
        if m_pos and _nd > 0:
            if len(per_pos_weighted) < len(per_pos_last):
                per_pos_weighted.extend([0.0] * (len(per_pos_last) - len(per_pos_weighted)))
            for _i, _r in enumerate(per_pos_last):
                per_pos_weighted[_i] += _r * _nd

    out: dict = {
        "intervals": intervals,
        "malformed_lines": malformed,
        "accepted_tokens": accepted_total,
        "drafted_tokens": drafted_total,
        "num_drafts": round(drafts_total, 1),
        "per_position_acceptance_rate": (
            [round(w / drafts_total, 4) for w in per_pos_weighted]
            if (per_pos_weighted and drafts_total > 0) else per_pos_last
        ),
        "per_position_acceptance_rate_last_interval": per_pos_last,
        "k_from_per_position": len(per_pos_weighted or per_pos_last) or None,
    }
    if drafts_total > 0:
        out["mean_acceptance_length"] = round(1.0 + accepted_total / drafts_total, 4)
        out["acceptance_rate"] = (
            round(accepted_total / drafted_total, 4) if drafted_total else None
        )
    return out


def parse_sglang_log(path: Path) -> dict:
    """Aggregate over every `Decode batch` line that carries an acceptance reading."""
    lengths: list[float] = []
    rates: list[float] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m_len = _SGLANG_ACCEPT_LEN.search(line)
        if not m_len:
            continue
        lengths.append(float(m_len.group(1)))
        m_rate = _SGLANG_ACCEPT_RATE.search(line)
        if m_rate:
            rates.append(float(m_rate.group(1)))

    out: dict = {
        "intervals": len(lengths),
        "malformed_lines": 0,
        "num_drafts": float(len(lengths)),
        "reading_source": "sglang_decode_batch_accept_len",
    }
    if lengths:
        out["mean_acceptance_length"] = round(sum(lengths) / len(lengths), 4)
    if rates:
        out["acceptance_rate"] = round(sum(rates) / len(rates), 4)
    return out


def read_from_raw_dir(raw_dir, workload_names) -> dict:
    """{workload: {side: parsed}} for the named workloads only."""
    out: dict = {}
    wanted = set(workload_names)
    for path in sorted(Path(raw_dir).glob("vllm_serve_*.log")):
        m = _LOG_NAME.match(path.name)
        if not m or m.group("workload") not in wanted:
            continue
        out.setdefault(m.group("workload"), {})[m.group("side")] = parse_log(path)
    for path in sorted(Path(raw_dir).glob("sglang_serve_*.log")):
        m = _SGLANG_LOG_NAME.match(path.name)
        if not m or m.group("workload") not in wanted:
            continue
        out.setdefault(m.group("workload"), {}).setdefault(
            m.group("side"), parse_sglang_log(path)
        )
    return out


def gate(per_workload: dict, workload_names, *, speculative: bool) -> dict:
    """Verdict + the numbers behind it. Never raises: a broken instrument must not fail a run."""
    rtol_raw = os.environ.get("INFB_ACCEPT_RTOL", "").strip()
    try:
        rtol = float(rtol_raw) if rtol_raw else None
    except ValueError:
        rtol = None

    detail: dict = {}
    hard_failures: list[str] = []
    pristine_failures: list[str] = []
    soft_failures: list[str] = []
    unknown: list[str] = []

    for name in workload_names:
        sides = per_workload.get(name) or {}
        cand, pris = sides.get("cand"), sides.get("pristine")
        entry: dict = {
            "cand": cand,
            "pristine": pris,
            "ratio": None,
        }
        if cand is None or pris is None:
            entry["status"] = "log_missing"
            unknown.append(f"{name}:{'cand' if cand is None else 'pristine'}")
        elif not speculative:
            entry["status"] = "not_a_speculative_task"
        elif pris["num_drafts"] <= 0:
            entry["status"] = "pristine_no_drafts"
            pristine_failures.append(
                f"{name}: pristine num_drafts={pris['num_drafts']} "
                f"(cand={cand['num_drafts']})"
            )
        elif cand["num_drafts"] <= 0:
            entry["status"] = "no_drafts"
            hard_failures.append(
                f"{name}: cand num_drafts={cand['num_drafts']}, "
                f"pristine num_drafts={pris['num_drafts']}"
            )
        else:
            c = cand.get("mean_acceptance_length")
            p = pris.get("mean_acceptance_length")
            entry["ratio"] = round(c / p, 4) if (c and p) else None
            entry["status"] = "measured"
            if rtol is not None and entry["ratio"] is not None and entry["ratio"] < (1.0 - rtol):
                soft_failures.append(
                    f"{name}: acceptance ratio {entry['ratio']} < {1.0 - rtol:.4f}"
                )
        detail[name] = entry

    verdict: dict = {
        "per_workload": detail,
        "rtol": rtol,
        "rtol_source": "INFB_ACCEPT_RTOL" if rtol is not None else "unset -- recorded, not judged",
        "unknown": unknown,
        "failed": False,
        "reason": None,
    }
    if pristine_failures:
        verdict["failed"] = True
        verdict["reason"] = "acceptance_pristine_no_drafts"
        verdict["failures"] = pristine_failures
    elif hard_failures:
        verdict["failed"] = True
        verdict["reason"] = "acceptance_no_drafts"
        verdict["failures"] = hard_failures
    elif soft_failures:
        verdict["failed"] = True
        verdict["reason"] = "acceptance_length_below_floor"
        verdict["failures"] = soft_failures
    return verdict
