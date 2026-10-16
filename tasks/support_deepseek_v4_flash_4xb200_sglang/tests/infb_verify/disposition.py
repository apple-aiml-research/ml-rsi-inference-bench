#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The trial's disposition before grading: whether the grader's binary reward stands or the trial is void because a
scored result was censored."""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

VOID = 10


def junit_outcomes(paths: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    rank = {"passed": 0, "skipped": 1, "failed": 2}
    for path in paths:
        if not path.is_file():
            continue
        for case in ET.parse(path).iter("testcase"):
            nid = f"{case.get('classname', '')}.{case.get('name', '')}".strip(".")
            fail = case.find("failure") if case.find("failure") is not None else case.find("error")
            outcome = "failed" if fail is not None else ("skipped" if case.find("skipped") is not None else "passed")
            msg = (fail.get("message") or fail.text or "") if fail is not None else ""
            if nid not in out or rank[outcome] > rank[out[nid]["outcome"]]:
                out[nid] = {"outcome": outcome, "message": msg}
    return out


def plugin_attributions(path: Path) -> dict[str, dict]:
    """classname.name -> the attribution of its failed call, where the plugin recorded one."""
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("outcome") == "failed" and r.get("attribution"):
            out[f"{r['classname']}.{r['name']}"] = r["attribution"]
    return out


def classify(nid: str, outcome: dict | None, attribution: dict | None, l3_wall: dict | None,
             kernel_faults: dict | None = None) -> str:
    """`candidate`, `censored:<node>` or `blocked:<node>` for one scored id that did not pass."""
    if kernel_faults and nid in kernel_faults:
        return f"censored:{nid}"
    if attribution:
        return "candidate" if attribution.get("status") == "candidate_fail" else f"censored:{nid}"
    msg = (outcome or {}).get("message", "")
    if msg.startswith("censored"):
        return f"censored:{nid}"
    if msg.startswith("blocked_by:"):
        return "blocked:" + msg.split(":", 2)[1].strip()
    if l3_wall and nid in set(l3_wall.get("not_run", [])):
        return "blocked:" + str(l3_wall.get("censored_by"))
    if l3_wall and nid in set(l3_wall.get("censored_ids", [])):
        return f"censored:{l3_wall.get('censored_by')}"
    return "candidate"


def decide(config: dict, verifier_dir: Path) -> dict:
    reports = [Path(p.replace("/logs/verifier", str(verifier_dir), 1)) for p in config["grade"]["reports"]]
    outcomes = junit_outcomes(reports)
    atts = plugin_attributions(verifier_dir / "efficiency" / "TestHiddenDSV4" / "l2-results.jsonl")
    l3_wall = json.loads((verifier_dir / "l3-wall.json").read_text()) if (verifier_dir / "l3-wall.json").is_file() \
        else None
    kfaults = json.loads((verifier_dir / "verifier-faults.json").read_text()) \
        if (verifier_dir / "verifier-faults.json").is_file() else None
    scored = list(config["f2p_node_ids"]) + list(config["p2p_node_ids"])
    cls = {}
    for nid in scored:
        o = outcomes.get(nid)
        if o is not None and o["outcome"] == "passed":
            continue
        cls[nid] = classify(nid, o, atts.get(nid), l3_wall, kfaults)
    candidate = sorted(n for n, c in cls.items() if c == "candidate")
    censored = sorted(n for n, c in cls.items() if c.startswith("censored:"))
    blocked = sorted(n for n, c in cls.items() if c.startswith("blocked:"))
    if candidate:
        decision = {"decision": "grade", "why": "a valid candidate failure", "candidate_failures": candidate}
    elif censored or blocked:
        first = (censored or blocked)[0]
        key = "censored_by" if censored else "blocked_by"
        decision = {"decision": "void", key: cls[first].split(":", 1)[1], "why": "a scored result was censored, and no "
                    "candidate failure was observed"}
    else:
        decision = {"decision": "grade", "why": "no scored id failed"}
    return {**decision, "censored": censored, "blocked": blocked, "not_passed": len(cls), "scored": len(scored)}


def main(argv: list[str]) -> int:
    config = json.loads(Path(argv[1]).read_text())
    vdir = Path(argv[argv.index("--verifier-dir") + 1]) if "--verifier-dir" in argv else Path("/logs/verifier")
    d = decide(config, vdir)
    print(json.dumps(d, sort_keys=True))
    if d["decision"] == "void":
        (vdir / "void.json").write_text(json.dumps(d, indent=1, sort_keys=True))
        return VOID
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
