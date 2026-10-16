#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The reward's disposition under censoring before the frame grades.

    python3 -I /tests/infb_verify/disposition.py [--config /tests/config.json] [--verifier-dir /logs/verifier]

test.sh runs this after every group and before the frame's `grader.py grade`, which is shared and untouched. It
reads the reports the grader will read, with the grader's own junit parser, and the groups' exit records
(`groups.jsonl`, written by test.sh). It classifies every scored node: config.json's F2P and P2P ids (the diagnostics
are not scored). It then applies the censoring rule:

  1. a scored node with a valid candidate failure leaves the grader's reward as it is (0), whatever else was censored
     or blocked. A candidate failure is a check that failed, an operation over its own public limit on a healthy host,
     or the candidate's server dying (the primary cause where it is first seen);
  2. otherwise a scored node that was censored, blocked or faulted by a cause that is not the candidate's gives no
     reward. Censored means a node or group wall expired while every operation was within its own limit: a verifier
     fault. The result is the task's -1 sentinel in `reward.txt`, which Pier reads before `reward.json`. The record
     beside it, `disposition.json` (`void`, `censored_by`, `blocked_by`), tells a void apart from the other verifier
     faults that use the same sentinel: preflight, the restore and the reference's host faults end test.sh before
     this step runs and leave no disposition;
  3. otherwise the grader's reward stands.

Every operation enforces its own limit (each request, measurement invocation, import, the rebuild, each unit-test
file's process and each launch), so a wall that expires had every operation within its limit. A failure is read from
its report as follows:
  * censored: a node's call alarm (`DeadlineExceeded: ... (call)`), the session runner's kill at its budget or at the
    verifier clock's end, or a wall-bounded group (L0, the component group, the runtime group) whose `timeout`
    expired before it reported;
  * the candidate's: a setup alarm (the session's launch at its startup limit; the slow-host rule has already turned a
    slow host's into a host fault), a teardown alarm (the teardown limit), a unit-test file's process over its limit,
    any failed check or unreported node, and `blocked_by:server_unreachable`: the candidate's server died, and the
    node where that was first seen names it;
  * void, other than a wall: `blocked_by:<any other cause>`, `reference_fault`, `verifier_fault`, `environment_fault`.
A censored node keeps a failing junit case whose message is prefixed `censored (verifier_fault): `, so the partial
scores the grader writes stay readable and name it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

CENSORED_PREFIX = "censored (verifier_fault): "
#: Messages that mean a wall expired: the per-node alarm in a node's call, and the session runner's kill.
_CALL_ALARM = re.compile(r"DeadlineExceeded:.*\(call\)")
_WALL_TEXTS = ("reached the verifier clock's end", "the group was killed at its", "before this node reported")
#: A setup or teardown alarm: the launch's or the teardown's own limit (launch_s and teardown_s are those limits).
_OPERATION_ALARM = re.compile(r"DeadlineExceeded:.*\((setup|teardown)\)")
_BLOCKED = re.compile(r"blocked_by:([A-Za-z0-9_.:-]+?)(?::\s|\s|$)")
_FAULTS = ("reference_fault", "verifier_fault", "environment_fault")
TIMEOUT_RCS = (124, 137)  # `timeout` at its bound (TERM), and after its kill-after (KILL)


def junit_names(nodeid: str) -> tuple[str, str]:
    """(classname, name) as pytest's junitxml derives them from a node id (`deadline.junit_names`)."""
    path, bracket, params = nodeid.partition("[")
    names = path.split("::")
    names[0] = re.sub(r"\.py$", "", names[0].replace("/", "."))
    names[-1] += bracket + params
    return ".".join(names[:-1]), names[-1]


def _split(jid: str) -> tuple[str, str]:
    """A junit id `classname.name` split at the last dot outside a parametrization."""
    head, bracket, params = jid.partition("[")
    classname, _, name = head.rpartition(".")
    return classname, name + bracket + params


def _load_grader(tests_dir: Path):
    spec = importlib.util.spec_from_file_location("infb_disposition_grader", tests_dir / "grader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def groups_of(cfg: dict) -> dict[str, str]:
    """Each scored id's group: `l0`, `sparse-components`, `runtime`, `session`, or `preservation-<k>` in the order test.sh
    runs the files."""
    out = {}
    files = sorted((cfg["infb"].get("preservation") or {}).get("files", {}))
    for k, f in enumerate(files):
        for pid in cfg["infb"]["preservation"]["files"][f]:
            out[".".join(junit_names(pid))] = f"preservation-{k}"
    for jid in cfg["f2p_node_ids"] + cfg["p2p_node_ids"]:
        if jid in out:
            continue
        if jid.startswith("infb_verify.test_l0."):
            out[jid] = "l0"
        elif jid in ("infb_verify.test_hidden.test_served_configuration",
                     "infb_verify.test_sparse_indexer.test_sparse_indexer_serving_binding"):
            out[jid] = "runtime"
        elif jid.startswith("infb_verify.test_sparse_indexer."):
            out[jid] = "sparse-components"
        elif jid.startswith("infb_verify."):
            out[jid] = "session"
        else:
            out[jid] = "unknown"
    return out


def classify(status: str | None, message: str, group: dict | None) -> tuple[str, str]:
    """(class, why) for one scored node: class is `pass`, `candidate`, `censored`, `blocked` or `fault`."""
    if status == "passed":
        return "pass", ""
    first = message or ""
    if status is None:
        if group and group.get("rc") in TIMEOUT_RCS:
            if group.get("bound") == "wall":
                return "censored", (f"the {group['group']} group's bound of {group['bound_s']} s expired (rc "
                                    f"{group['rc']}) before it reported")
            return "candidate", (f"the {group['group']} process exceeded its {group['bound_s']} s operation limit "
                                 f"(rc {group['rc']}): a resource failure")
        return "candidate", ("in no report (rc " + str(group.get("rc")) + ")" if group else "in no report")
    if _CALL_ALARM.search(first) or any(t in first for t in _WALL_TEXTS):
        return "censored", first.splitlines()[0][:300]
    if _OPERATION_ALARM.search(first):
        return "candidate", first.splitlines()[0][:300]
    m = _BLOCKED.search(first)
    if m:
        cause = m.group(1)
        if cause == "server_unreachable":
            return "candidate", "blocked_by:server_unreachable: the candidate's server died (its primary cause)"
        return "blocked", f"blocked_by:{cause}"
    for fault in _FAULTS:
        if re.search(rf"(^|\W){fault}:", first):
            return "fault", fault
    if status == "skipped":
        return "candidate", "skipped (a scored node that did not run to a pass)"
    return "candidate", first.splitlines()[0][:300] if first else "failed"


def _failure_messages(path: Path) -> dict[str, str]:
    """{junit id: the failure's message attribute} for one report: the exception's own text, where pytest.fail, the
    deadline runner and this module put their markers. Not the element's text, which is the traceback."""
    out: dict[str, str] = {}
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return out
    for tc in root.iter("testcase"):
        cn, nm = (tc.get("classname") or "").strip(), (tc.get("name") or "").strip()
        if not nm:
            continue
        for ch in tc:
            if ch.tag.rsplit("}", 1)[-1] in ("failure", "error"):
                out.setdefault(f"{cn}.{nm}" if cn else nm, (ch.get("message") or "").strip())
    return out


def _annotate(path: Path, censored: set[str]) -> int:
    """Prefix the failure message of each censored case in one junit report; the number annotated."""
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError):
        return 0
    n = 0
    for tc in tree.getroot().iter("testcase"):
        cn, nm = (tc.get("classname") or "").strip(), (tc.get("name") or "").strip()
        if (f"{cn}.{nm}" if cn else nm) not in censored:
            continue
        for ch in tc:
            if ch.tag.rsplit("}", 1)[-1] in ("failure", "error") and not (ch.get("message") or "").startswith(CENSORED_PREFIX):
                ch.set("message", CENSORED_PREFIX + (ch.get("message") or ""))
                n += 1
    if n:
        tree.write(path, encoding="utf-8", xml_declaration=True)
    return n


def _write_junit(path: Path, cases: list[tuple[str, str]], suite: str) -> None:
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<testsuites>",
             f'<testsuite name={quoteattr(suite)} tests="{len(cases)}" failures="{len(cases)}" errors="0" skipped="0">']
    for jid, msg in cases:
        cn, nm = _split(jid)
        lines.append(f"<testcase classname={quoteattr(cn)} name={quoteattr(nm)} time=\"0.000\">"
                     f"<failure message={quoteattr(msg[:500])}>{escape(msg)}</failure></testcase>")
    lines += ["</testsuite>", "</testsuites>", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def dispose(cfg: dict, verifier_dir: Path, tests_dir: Path) -> dict:
    grader = _load_grader(tests_dir)
    reports = [Path(p) for p in cfg["grade"]["reports"]]
    seen: dict[str, tuple[str, str]] = {}
    messages: dict[str, str] = {}
    for rep in reports:
        for nid, (st, msg) in grader.parse_junit(str(verifier_dir / rep.name), cfg["grade"]).items():
            grader.add(seen, nid, st, msg)
        for nid, message in _failure_messages(verifier_dir / rep.name).items():
            messages.setdefault(nid, message)
    groups = {}
    gpath = verifier_dir / "groups.jsonl"
    if gpath.exists():
        for line in gpath.read_text(errors="replace").splitlines():
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            groups[g["group"]] = g
    group_of = groups_of(cfg)
    rows = []
    for bucket in ("f2p", "p2p"):
        for jid in cfg[f"{bucket}_node_ids"]:
            st, _ = seen.get(jid, (None, ""))
            g = group_of.get(jid)
            # the failure's own message, never its traceback: pytest prints the failing test's source there, which
            # names every marker the test could raise
            cls, why = classify(st, messages.get(jid, ""), groups.get(g) if g else None)
            rows.append({"id": jid, "bucket": bucket, "group": g, "status": st or "missing", "class": cls, "why": why})
    by = {c: [r for r in rows if r["class"] == c] for c in ("pass", "candidate", "censored", "blocked", "fault")}
    if by["candidate"]:
        rule, void = 1, False
    elif by["censored"] or by["blocked"] or by["fault"]:
        rule, void = 2, True
    else:
        rule, void = 3, False
    # the censored cases keep a failing junit case that names them censored; a wall-bounded group killed before it
    # reported, and a unit-test file's process over its limit, get the report the grader would otherwise find missing
    censored = {r["id"] for r in by["censored"]}
    annotated = {rep.name: _annotate(verifier_dir / rep.name, censored) for rep in reports}
    written = []
    for name, g in groups.items():
        rep = verifier_dir / g.get("junit", "")
        if not g.get("junit") or rep.exists() or g.get("rc") not in TIMEOUT_RCS:
            continue
        cases = [(r["id"], (CENSORED_PREFIX if r["class"] == "censored" else "") + r["why"])
                 for r in rows if r["group"] == name and r["status"] == "missing"]
        if cases:
            _write_junit(rep, cases, f"{name} (written by the reward's disposition)")
            written.append(rep.name)
    record = {"rule": rule, "void": void,
              "censored_by": [r["id"] for r in by["censored"]],
              "blocked_by": [{"node": r["id"], "cause": r["why"]} for r in by["blocked"]],
              "faulted_by": [{"node": r["id"], "fault": r["why"]} for r in by["fault"]],
              "candidate_failures": [{"node": r["id"], "why": r["why"]} for r in by["candidate"]],
              "counts": {c: len(v) for c, v in by.items()}, "groups": list(groups.values()),
              "junit_annotated": annotated, "junit_written": written, "rows": rows}
    if void:
        (verifier_dir / "reward.txt").write_text("-1\n")
        record["reward_txt"] = -1
    (verifier_dir / "disposition.json").write_text(json.dumps(record, indent=1))
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=os.environ.get("TESTS_DIR", "/tests") + "/config.json")
    ap.add_argument("--verifier-dir", default=os.environ.get("VERIFIER_DIR", "/logs/verifier"))
    a = ap.parse_args(argv)
    cfg_path = Path(a.config)
    rec = dispose(json.loads(cfg_path.read_text()), Path(a.verifier_dir), cfg_path.parent)
    print("INFB_DISPOSITION " + json.dumps({k: rec[k] for k in ("rule", "void", "censored_by", "blocked_by",
                                                              "faulted_by", "counts")}), flush=True)
    for r in rec["candidate_failures"][:20]:
        print(f"[disposition] candidate: {r['node']}: {r['why']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
