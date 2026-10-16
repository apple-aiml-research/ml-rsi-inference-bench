#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Client for the native-region measurement protocol against a running server.

`collect` runs a recipe's cases through the server's `MeasurementWorker`: capture every case, then
`timing.MATRICES` complete matrices of timed draws in the fixed case order, then allocation evidence
for PLE cases and use evidence once per component. An untimed case (`"timed": false`, a joining case)
is captured and checked only. Every TP rank's answer is kept verbatim with its
identity; a missing, stale or failed rank answer fails the case (the raw answers are retained). The
record is rewritten after every answer, so an interrupted collection keeps what was observed.

Budgets (`budgets.json`): the whole collection has `component_collection` seconds and each matrix
`component_matrix` seconds; every request receives the remaining allowance of both. Work left when
an allowance is spent is recorded as not run (`budget_exhausted`), never skipped silently.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from . import timing
from .worker import protocol_identity

RPC_TIMEOUT_S = 900


class CollectionError(RuntimeError):
    """A rank answer was missing, stale or failed."""


class BudgetExhausted(CollectionError):
    """An operation's allowance was spent before this request could run."""


def budgets() -> dict:
    data = json.loads((Path(__file__).resolve().parents[1] / "budgets.json").read_text())
    return {name: row["seconds"] for name, row in data["operations"].items()}


def rpc(base_url: str, request: dict, *, timeout: float = RPC_TIMEOUT_S) -> list[dict]:
    request = dict(request, protocol=protocol_identity(), request_id=uuid.uuid4().hex)
    payload = {"method": "qwen38_measurement", "timeout": timeout, "args": [json.dumps(request)]}
    req = urllib.request.Request(base_url.rstrip("/") + "/collective_rpc", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout + 60) as response:
        rows = json.load(response)["results"]
    if len(rows) != 4 or {row.get("rank") for row in rows} != {0, 1, 2, 3}:
        raise CollectionError(f"{request['action']}: missing or duplicate TP rank answers")
    if any(row.get("request_id") != request["request_id"] or row.get("protocol") != request["protocol"]
           for row in rows):
        raise CollectionError(f"{request['action']}: stale or mismatched answer")
    failed = [row for row in rows if row.get("status") != "complete"]
    if failed:
        raise CollectionError(f"{request['action']}: rank {failed[0]['rank']} failed: {failed[0].get('error')}")
    sources = {row.get("source") for row in rows}
    if len(sources) != 1 or None in sources:
        raise CollectionError(f"{request['action']}: ranks report different candidate sources")
    return rows


def collect(base_url: str, recipe: str, cases: list[dict], tokens: dict, *, output: str | Path,
            vocabulary: int, channels: int = 10240, positions: dict | None = None,
            matrices: int = timing.MATRICES, use_components=("qsa", "ple"),
            use_tolerance: dict | None = None, budget_s: float | None = None,
            matrix_budget_s: float | None = None, enclosing: dict | None = None) -> dict:
    """`tokens[case_id] = {"prompts": {req_id: [...]}, "streams": {req_id: [...]}}`, optionally with
    `"case"`: the case as sent (per-request planned acceptance). `enclosing` optionally names, per
    component, the enclosed operations to measure (a wider enclosure than the loaded region's own is
    composed; `native.ENCLOSING`)."""
    limits = budgets()
    budget_s = float(budget_s if budget_s is not None else limits["component_collection"])
    matrix_budget_s = float(matrix_budget_s if matrix_budget_s is not None else limits["component_matrix"])
    begun = time.monotonic()
    window = {"end": None}

    def allowance() -> float:
        now = time.monotonic()
        left = begun + budget_s - now
        if window["end"] is not None:
            left = min(left, window["end"] - now)
        if left <= 1.0:
            raise BudgetExhausted("the operation's allowance is spent")
        return min(RPC_TIMEOUT_S, left)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("case IDs must be unique and nonempty")
    positions = positions or {"ordinary": 9, "mtp3": 12}
    report = {"schema": "qwen38-native-measurement/1", "protocol": protocol_identity(), "recipe": recipe,
              "matrices": matrices, "warmups": timing.WARMUPS, "draws": timing.DRAWS,
              "budgets": {"collection_s": budget_s, "matrix_s": matrix_budget_s}, "matrix_seconds": [],
              "enclosing": enclosing,
              "cases": {c["id"]: {"case": c, "status": "running"} for c in cases}, "use": {},
              "status": "running", "started": time.time()}

    def save():
        output.write_text(json.dumps(report, indent=1, allow_nan=False))

    def attempt(key, action, row, **request):
        try:
            answers = rpc(base_url, {"action": action, **request}, timeout=allowance())
            row.setdefault(key, []).append(answers)
            return answers
        except Exception as exc:
            row["status"] = "failed"
            row.setdefault("errors", []).append({"action": action, "error": repr(exc)[:4000]})
            if isinstance(exc, BudgetExhausted):
                row["errors"][-1]["budget_exhausted"] = True
                report["budget_exhausted"] = True
            if isinstance(exc, urllib.error.HTTPError):
                row["errors"][-1]["response"] = exc.read().decode(errors="replace")[:4000]
            return None
        finally:
            save()

    report["identity"] = rpc(base_url, {"action": "identity"}, timeout=allowance())
    report["source"] = report["identity"][0]["source"]
    save()
    for case in cases:
        row = report["cases"][case["id"]]
        given = tokens[case["id"]]
        answers = attempt("capture", "capture", row, id=case["id"], case=given.get("case", case),
                          vocabulary=vocabulary, prompts=given["prompts"], streams=given["streams"],
                          enclosing=enclosing)
        if answers is not None and any(a["recipe"] != recipe for a in answers):
            row["status"] = "failed"
            row.setdefault("errors", []).append({"action": "capture", "error": "served recipe differs"})
    for matrix in range(matrices):
        started = time.monotonic()
        window["end"] = started + matrix_budget_s
        for case in cases:
            row = report["cases"][case["id"]]
            if row["status"] != "failed" and case.get("timed", True):
                attempt("matrices", "draws", row, id=case["id"], matrix=matrix)
        window["end"] = None
        report["matrix_seconds"].append(time.monotonic() - started)
        save()
    for case in cases:
        row = report["cases"][case["id"]]
        if case["component"] == "ple" and case.get("timed", True) and row["status"] != "failed":
            attempt("memory", "memory", row, id=case["id"], channels=channels, positions=positions[recipe])
    for component in use_components:
        first = next((c for c in cases if c["component"] == component and c.get("timed", True)
                      and report["cases"][c["id"]]["status"] != "failed"), None)
        if first is not None:
            holder = report["use"].setdefault(component, {"case": first["id"], "status": "running"})
            given = tokens[first["id"]]
            attempt("answers", "use", holder, id=first["id"], vocabulary=vocabulary,
                    prompts=given["prompts"], streams=given["streams"],
                    tolerance=(use_tolerance or {}).get(component))
            if holder["status"] != "failed":
                holder["status"] = "complete"
    for case in cases:
        row = report["cases"][case["id"]]
        try:
            # Releasing captured state is cleanup, bounded by the RPC timeout even when the
            # operation's allowance is spent.
            rpc(base_url, {"action": "release", "id": case["id"]}, timeout=120)
        except Exception as exc:
            row.setdefault("errors", []).append({"action": "release", "error": repr(exc)[:2000]})
        if row["status"] != "failed":
            # An untimed case (a joining case) is complete once captured and checked.
            complete = len(row.get("capture", [])) == 1 and (not case.get("timed", True) or (
                len(row.get("matrices", [])) == matrices and (case["component"] != "ple" or len(row.get("memory", [])) == 1)))
            row["status"] = "complete" if complete else "failed"
    report["status"] = "complete" if all(r["status"] == "complete" for r in report["cases"].values()) and all(
        u["status"] == "complete" for u in report["use"].values()) else "failed"
    report["finished"] = time.time()
    save()
    return report
