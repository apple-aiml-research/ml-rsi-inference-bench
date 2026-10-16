#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The verification DAG's bounds, composed mechanically. This task's own; nothing else reads it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DISPATCH_S = 60
VERIFIER_FINAL_S = 600
JOB_CLEANUP_S = 600


def _caps(tests_dir: Path) -> dict:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(tests_dir / "public"))
    from glm53_protocols.limits import LIMITS  # noqa: PLC0415 - the one source of the public values
    return dict(LIMITS)


def node_bound(node: dict, caps: dict) -> float:
    cap = float(caps[node["largest_op"]])
    r = float(node.get("R_s") or 0.0)
    return (1 + int(node.get("extra", 0))) * max(3.0 * r, cap + DISPATCH_S)


def bounds(cfg: dict, tests_dir: Path = HERE.parent) -> dict:
    dag = cfg["infb"]["deadline_dag"]
    caps = _caps(tests_dir)
    groups, nodes = {}, {}
    for g in dag["groups"]:
        total = 0.0
        for n in g["nodes"]:
            b = node_bound(n, caps)
            nodes[f"{g['name']}/{n['name']}"] = b
            total += b
        groups[g["name"]] = total + DISPATCH_S
    root = sum(groups.values()) + DISPATCH_S
    verifier = root + VERIFIER_FINAL_S
    job = float(dag["agent_s"]) + float(dag["environment_build_s"]) + float(dag["verifier_build_s"]) + verifier \
        + JOB_CLEANUP_S
    return {"groups": groups, "nodes": nodes, "root_s": root, "verifier_s": verifier, "job_s": job,
            "rule": "B = (1 + extra) x max(3 x R, largest op cap + 60 s); group = sum B + 60 s; root = sum groups + "
                    "60 s; verifier = root + 600 s; job = agent + builds + verifier + 600 s"}


def alarms(cfg: dict, b: dict) -> dict:
    """The per-node alarms of the graded groups (`infb.efficiency.deadlines.nodes`): each node's own B."""
    out = {}
    for key, value in b["nodes"].items():
        group, node = key.split("/", 1)
        if group.startswith("graded:") and not node.startswith("_"):
            out[f"{group.split(':', 1)[1]}.{node}"] = value
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--print", dest="key")
    a = ap.parse_args(argv)
    path = Path(a.config)
    cfg = json.loads(path.read_text())
    b = bounds(cfg, path.parent)
    if a.key:
        print(f"{b['groups'][a.key]:g}")
        return 0
    if a.write:
        cfg["infb"]["bounds"] = b
        d = cfg["infb"]["efficiency"]["deadlines"]
        d["nodes"] = alarms(cfg, b)
        d["launch_s"] = b["nodes"]["graded:TestHiddenHighThroughput/_launch"]
        for axis in ("decode_c1", "throughput"):
            d[f"reference_{axis}_s"] = b["nodes"][f"reference/TestHiddenHighThroughput.{axis}"]
        d["teardown_s"] = float(_caps(path.parent)["teardown_s"])
        path.write_text(json.dumps(cfg, indent=1) + "\n")
    print(json.dumps({k: b[k] for k in ("groups", "root_s", "verifier_s", "job_s")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
