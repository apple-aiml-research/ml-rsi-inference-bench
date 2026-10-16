#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The task's reward disposition, before grading -- a void is not a zero."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

VOID = Path("/logs/verifier/void.json")
SERVICE = Path("/logs/verifier/service")
NON_CANDIDATE = ("verifier_fault", "reference_fault", "environment_fault")
RUNNER_BUDGET = ("the group was killed at its wall-clock budget", "reached its wall-clock budget")
RECORDS = Path(os.environ.get("INFB_RECORD_DIR", "/logs/verifier/records"))
WITNESS_CHECKS = {"test_00_recipe_conformance": "recipe_conformance", "test_zz_adaptive_schedule": "adaptive",
                  "test_attention_semantics": "attention_semantics"}
_MARKER = re.compile(r"^(?:E\s+)?(?:\w+(?:Error|Exception): |Failed: )?"
                     r"(blocked_by:|verifier_fault: censored|verifier_fault:|reference_fault:|environment_fault:)")


def _cases(reports: list[str]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for rep in reports:
        try:
            root = ET.parse(rep).getroot()
        except (OSError, ET.ParseError):
            continue
        for tc in root.iter("testcase"):
            nid = f"{tc.get('classname', '')}.{tc.get('name', '')}"
            bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
            if bad is not None:
                out[nid] = ("failed", (bad.get("message") or "") + "\n" + (bad.text or ""))
            elif tc.find("skipped") is not None:
                out[nid] = ("skipped", (tc.find("skipped").get("message") or ""))
            else:
                out[nid] = ("passed", "")
    return out


def _witness_status(nid: str) -> str | None:
    """A witness-judged node's status from its own verdict record (`records/<check>/<class>/verdict.json`)."""
    parts = nid.split(".")
    check = WITNESS_CHECKS.get(parts[-1]) if len(parts) >= 2 else None
    if not check:
        return None
    try:
        status = str(json.loads((RECORDS / check / parts[-2] / "verdict.json").read_text()).get("status") or "")
    except (OSError, ValueError):
        return None
    if status.startswith("blocked"):
        return "blocked"
    if status == "censored" or status.startswith("verifier_fault: censored"):
        return "censored"
    if status in NON_CANDIDATE or status == "candidate_fail":
        return status
    return None


def classify(outcome: str, message: str, nid: str | None = None) -> str:
    if outcome == "passed":
        return "pass"
    if nid:
        status = _witness_status(nid)
        if status:
            return status
    text = message.strip()
    if any(b in text for b in RUNNER_BUDGET):
        return "censored"
    for line in text.splitlines():
        m = _MARKER.match(line.strip())
        if m:
            tag = m.group(1)
            if tag == "blocked_by:":
                return "blocked"
            if tag == "verifier_fault: censored":
                return "censored"
            return tag[:-1]
    return "candidate_fail"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    cases = _cases(cfg["grade"]["reports"])
    scored = list(cfg.get("f2p_node_ids", [])) + list(cfg.get("p2p_node_ids", []))
    statuses: dict[str, str] = {}
    for nid in scored:
        outcome, msg = cases.get(nid, ("missing", "missing from every report"))
        statuses[nid] = "candidate_fail" if outcome in ("missing", "skipped") else classify(outcome, msg, nid)
    deaths = []
    for f in sorted(SERVICE.glob("*.json")):
        try:
            died = json.loads(f.read_text()).get("died_in")
        except (OSError, ValueError):
            continue
        if died:
            deaths.append(died)
    counts: dict[str, int] = {}
    for s in statuses.values():
        counts[s] = counts.get(s, 0) + 1
    summary = {"counts": counts, "server_deaths": deaths,
               "non_pass": {n: s for n, s in statuses.items() if s != "pass"}}
    if deaths or counts.get("candidate_fail"):
        print("[disposition] " + json.dumps({"reward": "graded", **summary}))
        return 0
    voided = {n: s for n, s in statuses.items() if s not in ("pass", "candidate_fail")}
    if voided:
        rec = {"void": True, "reason": "no valid candidate failure; scored nodes were censored, blocked or faulted",
               "censored_by": sorted(n for n, s in voided.items() if s == "censored"),
               "blocked_by": sorted(n for n, s in voided.items() if s == "blocked"),
               "faults": {n: s for n, s in voided.items() if s not in ("censored", "blocked")}, **summary}
        VOID.parent.mkdir(parents=True, exist_ok=True)
        VOID.write_text(json.dumps(rec, indent=1))
        print("[disposition] " + json.dumps(rec))
        return 3
    print("[disposition] " + json.dumps({"reward": "graded", **summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
