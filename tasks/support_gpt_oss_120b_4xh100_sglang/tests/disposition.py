#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The reward disposition, computed before grading.

Run by tests/test.sh as /usr/bin/python3 -I -S /tests/disposition.py after every step and before the
shared grader. Standard library only.

  1. A scored node with a valid candidate failure -- including an operation over its own cap on a
     healthy host, and a component speed or numerical failure -- leaves the grader's reward (0).
  2. Otherwise, a scored node that was censored (a node or group wall expired while every operation
     stayed within its own cap) or blocked (a dependent observation whose prerequisite, such as the
     production server or a valid identity, was unavailable) voids the trial: no reward, the -1
     sentinel, recorded in /logs/verifier/disposition.json as censored_by:<node> / blocked_by:<node>.
  3. Otherwise the grader's reward stands.

Exit 0 lets the grader run; exit 3 is the void.
"""
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

TESTS = Path(os.environ.get("TESTS_DIR", "/tests"))
VERIFIER = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier"))
COMPONENT_CLASS = "infb_components.test_contract.TestContract"


def junit_cases(path):
    """classname.name -> (outcome, message, properties) for one report; absent or unreadable is empty."""
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
        message = (problem.get("message") or "") + "\n" + (problem.text or "") if problem is not None else ""
        properties = {p.get("name"): p.get("value") for p in case.iter("property")}
        worst = {"passed": 0, "skipped": 1, "failed": 2}
        if key not in cases or worst[outcome] > worst[cases[key][0]]:
            cases[key] = (outcome, message, properties)
    return cases


def step_walls():
    walls = {}
    path = VERIFIER / "step-walls.jsonl"
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            walls[row.get("step")] = row
    return walls


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
    walls = step_walls()
    component_wall = (walls.get("components") or {}).get("rc") in (124, 137)
    rows = {}
    for node in scored:
        outcome, message, properties = cases.get(node, ("missing", "", {}))
        status = properties.get("status", "")
        if node in censored or message.lstrip().startswith("censored:") or \
                (component_wall and node.startswith(COMPONENT_CLASS) and outcome == "missing"):
            rows[node] = "censored"
        elif status.startswith("blocked_by:"):
            rows[node] = status
        elif outcome in ("failed", "missing", "skipped"):
            rows[node] = "candidate_fail"
        else:
            rows[node] = "pass"
    failures = sorted(n for n, s in rows.items() if s == "candidate_fail")
    censored_nodes = sorted(n for n, s in rows.items() if s == "censored")
    blocked = sorted(n for n, s in rows.items() if s.startswith("blocked_by:"))
    if failures:
        verdict = "graded"
    elif censored_nodes or blocked:
        verdict = "void"
    else:
        verdict = "graded"
    return {"schema": "gpt-oss-disposition/1", "verdict": verdict, "candidate_failures": failures,
            "censored_by": censored_nodes, "blocked_by": [f"{n} ({rows[n]})" for n in blocked],
            "statuses": rows}


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
