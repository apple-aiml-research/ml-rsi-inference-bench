#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The reward disposition, before grading (standard library only).

Run by tests/test.sh as `python3 -I /tests/infb_verify/disposition.py` after every group and before
the shared grader:

  1. A scored node with a valid candidate failure -- including an operation over its own budget on a
     healthy host -- leaves the grader's reward (0).
  2. Otherwise a scored node that was censored (its group's wall expired while the node was within its
     own deadline) or blocked (a prerequisite record it depends on was unavailable for a reason other
     than a candidate failure) voids the trial: no reward, the -1 sentinel, with `censored_by` /
     `blocked_by` in /logs/verifier/disposition.json.
  3. Otherwise the grader's reward stands.

Exit 0 lets the grader run; exit 3 is the void.
"""
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS = Path(os.environ.get("TESTS_DIR", "/tests"))
VERIFIER = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier"))
WORST = {"passed": 0, "skipped": 1, "failed": 2}


def junit_cases(path):
    """classname.name -> (outcome, message) for one report; absent or unreadable is empty."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return {}
    cases = {}
    for case in root.iter("testcase"):
        key = f"{case.get('classname')}.{case.get('name')}"
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        outcome = "failed" if problem is not None else ("skipped" if case.find("skipped") is not None else "passed")
        message = ((problem.get("message") or "") + "\n" + (problem.text or "")) if problem is not None else ""
        if key not in cases or WORST[outcome] > WORST[cases[key][0]]:
            cases[key] = (outcome, message)
    return cases


def censored_from_summaries():
    names = set()
    for path in VERIFIER.glob("*-deadline-summary.json"):
        try:
            names.update(json.loads(path.read_text()).get("censored_nodes") or [])
        except (OSError, ValueError):
            continue
    return names


def disposition():
    config = json.loads((TESTS / "config.json").read_text())
    scored = list(config["f2p_node_ids"]) + list(config["p2p_node_ids"])
    cases = {}
    for report in config["grade"]["reports"]:
        cases.update(junit_cases(report))
    censored = censored_from_summaries()
    statuses = {}
    for node in scored:
        outcome, message = cases.get(node, ("missing", ""))
        head = message.lstrip()
        if node in censored or head.startswith("censored:"):
            statuses[node] = "censored"
        elif "blocked_by:" in head.splitlines()[0] if head else False:
            statuses[node] = "blocked_by:" + head.split("blocked_by:", 1)[1].splitlines()[0][:300]
        elif outcome in ("failed", "missing", "skipped"):
            statuses[node] = "candidate_fail"
        else:
            statuses[node] = "pass"
    failures = sorted(n for n, s in statuses.items() if s == "candidate_fail")
    censored_nodes = sorted(n for n, s in statuses.items() if s == "censored")
    blocked = sorted(n for n, s in statuses.items() if s.startswith("blocked_by:"))
    verdict = "graded" if failures or not (censored_nodes or blocked) else "void"
    return {"schema": "diffusion-gemma-disposition/1", "verdict": verdict, "candidate_failures": failures,
            "censored_by": censored_nodes, "blocked_by": [f"{n} ({statuses[n]})" for n in blocked],
            "statuses": statuses}


def main():
    record = disposition()
    VERIFIER.mkdir(parents=True, exist_ok=True)
    (VERIFIER / "disposition.json").write_text(json.dumps(record, indent=1) + "\n")
    if record["verdict"] == "void":
        print(f"[verifier] void: no candidate failure, and {record['censored_by'] or record['blocked_by']} "
              f"could not be judged; no reward (sentinel -1)", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
