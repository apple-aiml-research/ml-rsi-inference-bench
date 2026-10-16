#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The reward disposition before grading.

    python3 -I /tests/attribution.py      # in test.sh, after every suite and before `grader.py grade`

Every scored id (config.json f2p/p2p) that did not pass is attributed, in this order:

  * `censored_by:<node>` when the deadline plugin ended the node at its own bound (the junit message names it
    censored) -- a node wall that expired while every operation stayed within its published limit (a request, RPC
    or build that exhausts its own limit fails inside the node first, as the candidate's); `blocked_by:<node>` for
    an id that failed on that same expiry, as every node of a session does when its shared launch was censored;
    a setup or teardown expiry is named `<node>:setup` or `<node>:teardown`;
  * `censored_by:<group>` for the node the deadline runner found running when it killed its group at the group's
    budget, and `blocked_by:<group>` for the nodes that had not reported;
  * the status its check's record carries (`/logs/verifier/records/<check>.json`, written by
    infb_verify/records.py): `candidate_fail`, `reference_fault`, `environment_fault`, `verifier_fault` or
    `blocked_by:<check>`;
  * when the id's step ended at its bound (`tests/bounded_step.py`): `censored_by:<group>` for a group of several
    operations; for a group that is one operation (`infb.attribution.groups[*].single_operation`) the bound is that operation's own limit, so its expiry is the candidate's
    resource failure;
  * `candidate_fail` otherwise: a failed, errored, skipped or absent scored id with no other cause.

Disposition: (1) any `candidate_fail` -> the grader's reward, as before; (2) otherwise any censored, blocked or
non-candidate status -> no reward: exit 3, so test.sh stops before grading and the frame writes its -1 void
sentinel, which `attribution.json` distinguishes from the sentinel's other uses; (3) otherwise the grader's reward
stands. `/logs/verifier/attribution.json` is written in every case. Standard library only.
"""
from __future__ import annotations

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS = Path(os.environ.get("TESTS_DIR", "/tests"))
VERIFIER = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier"))
RECORDS = Path(os.environ.get("INFB_RECORDS_DIR", str(VERIFIER / "records")))
EXIT_VOID = 3


def junit_outcomes(paths: list[Path]) -> dict[str, dict]:
    """classname.name -> {outcome, message, report} over every readable report (worst outcome wins)."""
    rank = {"passed": 0, "skipped": 1, "failed": 2}
    out: dict[str, dict] = {}
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            continue
        for case in root.iter("testcase"):
            nid = ".".join(x for x in (case.get("classname"), case.get("name")) if x)
            outcome, message = "passed", ""
            for tag in ("failure", "error"):
                node = case.find(tag)
                if node is not None:
                    outcome, message = "failed", (node.get("message") or "") + "\n" + (node.text or "")
            if outcome == "passed" and case.find("skipped") is not None:
                outcome = "skipped"
            cur = out.get(nid)
            if cur is None or rank[outcome] > rank[cur["outcome"]]:
                out[nid] = {"outcome": outcome, "message": message[-2000:], "report": path.name}
    return out


def bounded_steps(run_log: Path) -> dict[str, dict]:
    """group -> its INFB_BOUNDED record (the last one, if a group ran twice)."""
    steps: dict[str, dict] = {}
    if run_log.is_file():
        for line in run_log.read_text(errors="replace").splitlines():
            if line.startswith("INFB_BOUNDED "):
                try:
                    row = json.loads(line[len("INFB_BOUNDED "):])
                except json.JSONDecodeError:
                    continue
                steps[row.get("group")] = row
    return steps


def records(root: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out[row.get("check", path.stem)] = row
    return out


def group_of(nid: str, groups: dict) -> str | None:
    for name, group in groups.items():
        if any(nid.startswith(p) for p in group["prefixes"]):
            return name
    return None


#: tests/infb_efficiency/deadline.py's alarm and its runner's kill, as they write them into the junit report.
PLUGIN_CENSORED = re.compile(r"censored: (.+?) \((setup|call|teardown)\) reached its node bound")
RUNNER_CENSORED = "censored: the group "
RUNNER_BLOCKED = "blocked: the group "


def node_name(nid: str) -> str:
    """A check's name from a junit id (`classname.name[params]`) or a pytest node id (`path::name[params]`)."""
    return re.split(r"::|\.", nid.split("[", 1)[0])[-1]


def status_of(name: str, seen: dict | None, group: str | None, groups: dict, step: dict | None,
              record: dict | None) -> str:
    message = seen["message"] if seen else ""
    censored = PLUGIN_CENSORED.search(message)
    if censored:
        owner, phase = node_name(censored.group(1)), censored.group(2)
        cause = owner if phase == "call" else f"{owner}:{phase}"
        return f"censored_by:{cause}" if owner == name else f"blocked_by:{cause}"
    if RUNNER_CENSORED in message:
        return f"censored_by:{group}"
    if RUNNER_BLOCKED in message:
        return f"blocked_by:{group}"
    if record is not None and record.get("status") not in (None, "running", "pass"):
        return record["status"]
    if step is not None and step.get("reason") == "bound":
        return "candidate_fail" if groups[group].get("single_operation") else f"censored_by:{group}"
    return "candidate_fail"


def attribute(cfg: dict, outcomes: dict, steps: dict, recs: dict) -> dict:
    groups = cfg["infb"]["attribution"]["groups"]
    scored = [(nid, "f2p") for nid in cfg["f2p_node_ids"]] + [(nid, "p2p") for nid in cfg["p2p_node_ids"]]
    rows = []
    for nid, bucket in scored:
        seen = outcomes.get(nid)
        if seen is not None and seen["outcome"] == "passed":
            continue
        name = node_name(nid)
        group = group_of(nid, groups)
        status = status_of(name, seen, group, groups, steps.get(group) if group else None, recs.get(name))
        rows.append({"id": nid, "bucket": bucket, "outcome": seen["outcome"] if seen else "absent",
                     "group": group, "status": status})
    candidate = [r for r in rows if r["status"] == "candidate_fail"]
    other = [r for r in rows if r["status"] != "candidate_fail"]
    # (1) a valid candidate failure decides the conjunction; (2) otherwise a censored, blocked or non-candidate
    # cause voids the trial; (3) otherwise every scored id passed and the grader's reward stands.
    decision = "void" if other and not candidate else "grade"
    return {"decision": decision, "not_passed": rows,
            "causes": sorted({r["status"] for r in other}), "candidate_failures": len(candidate)}


def main() -> int:
    cfg = json.loads((TESTS / "config.json").read_text())
    reports = [Path(p) for p in cfg["grade"]["reports"]]
    report = attribute(cfg, junit_outcomes(reports), bounded_steps(VERIFIER / "run.log"), records(RECORDS))
    VERIFIER.mkdir(parents=True, exist_ok=True)
    (VERIFIER / "attribution.json").write_text(json.dumps(report, indent=1) + "\n")
    print("INFB_ATTRIBUTION " + json.dumps({k: report[k] for k in ("decision", "causes", "candidate_failures")}))
    return EXIT_VOID if report["decision"] == "void" else 0


if __name__ == "__main__":
    sys.exit(main())
