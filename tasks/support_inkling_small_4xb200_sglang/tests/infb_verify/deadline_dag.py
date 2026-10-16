#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The verifier's execution DAG and its generated bounds. The DAG itself is data, `config.json` `infb.deadline_dag`; this module computes every bound from
it and writes or checks the scalar values the callers read.

Rules:

  * every operation keeps its public limit (`eval/inkling_small/limits.json`) and is enforced as one operation;
  * a node's wall is B(node) = max(margin x R(node), Cmax(node) + dispatch), R(node) the reference's measured wall for the
    node's complete prescribed procedure (no retries) on the declared hardware, Cmax(node) the largest limit among its
    operations; a node whose policy permits whole-node re-draws, retries or relaunches adds one B per permitted
    repetition (per-request resends inside a node are covered by the margin);
  * a serial group is the sum of its members' bounds plus one dispatch allowance; the job lifetime
    is agent + build allowances + verifier + `final_s`.

    python3 deadline_dag.py <task dir>            # print the generated values and the calculation
    python3 deadline_dag.py <task dir> --check    # exit 1 unless every caller carries the generated values
    python3 deadline_dag.py <task dir> --write    # write the generated values into config.json
"""
from __future__ import annotations

import json
import math
import re
import sys
import tomllib
from pathlib import Path


def limits(task: Path) -> dict:
    return json.loads((task / "tests" / "eval" / "deepseek_v4_flash" / "limits.json").read_text())


def node_bound(node: dict, caps: dict[str, float], margin: float, dispatch: float) -> dict:
    """B for one node, with the terms that produced it."""
    ops = node["ops"]
    if not ops or any(k not in caps for k in ops):
        raise ValueError(f"node operations {ops} name a kind limits.json does not define")
    cmax = max(caps[k] for k in ops)
    r = float(node["R_s"])
    b = math.ceil(max(margin * r, cmax + dispatch))
    reps = int(node.get("repetitions", 0))
    return {"R_s": r, "Cmax_s": cmax, "B_s": b, "repetitions": reps, "bound_s": b * (1 + reps),
            "from": "margin x R" if margin * r > cmax + dispatch else "Cmax + dispatch"}


def generate(task: Path) -> dict:
    cfg = json.loads((task / "tests" / "config.json").read_text())
    dag = cfg["infb"]["deadline_dag"]
    lim = limits(task)
    caps = {k: float(v["limit_s"]) for k, v in lim["operations"].items()}
    margin, dispatch, final = float(dag["margin_factor"]), float(dag["dispatch_s"]), float(dag["final_s"])
    l2 = dag["l2"]
    parts = {"launch": node_bound(l2["launch"], caps, margin, dispatch),
             "teardown": node_bound(l2["teardown"], caps, margin, dispatch)}
    nodes = {name: node_bound(n, caps, margin, dispatch) for name, n in l2["nodes"].items()}
    l2_bound = math.ceil(parts["launch"]["bound_s"] + sum(n["bound_s"] for n in nodes.values())
                         + parts["teardown"]["bound_s"] + dispatch)
    steps = {}
    for s in dag["steps"]:
        steps[s["name"]] = ({"bound_s": l2_bound, "group": s["group"]} if "group" in s
                            else node_bound(s, caps, margin, dispatch))
    root = math.ceil(sum(s["bound_s"] for s in steps.values()) + dispatch)
    build = float(dag["verifier_build_s"])
    verifier = math.ceil(root + final + build)
    toml = tomllib.loads((task / "task.toml").read_text())
    agent = float(toml["agent"]["timeout_sec"])
    env_build = float(toml["environment"]["build_timeout_sec"])
    # the verifier image's build is already inside the verifier's allowance (verifier_build_s), so the job adds only
    # the task image's
    lifetime = math.ceil(agent + env_build + verifier + final)
    return {"caps": caps, "margin_factor": margin, "dispatch_s": dispatch, "final_s": final,
            "l2": {"launch": parts["launch"], "teardown": parts["teardown"], "nodes": nodes, "bound_s": l2_bound},
            "steps": steps, "root_s": root, "verifier_build_s": build, "verifier_timeout_sec": verifier,
            "job_lifetime_s": lifetime}


def scalars(gen: dict) -> dict:
    """The values each caller reads, from the generated DAG."""
    return {
        "config.efficiency.deadlines": {"launch_s": gen["l2"]["launch"]["bound_s"],
                                        "teardown_s": gen["l2"]["teardown"]["bound_s"],
                                        "nodes": {k: v["bound_s"] for k, v in gen["l2"]["nodes"].items()}},
        "test.sh": {"l2_budget_s": gen["l2"]["bound_s"],
                    **{f"{k}_s": int(v["bound_s"]) for k, v in gen["steps"].items() if k != "l2"}},
        "task.toml.verifier.timeout_sec": gen["verifier_timeout_sec"],
        "job_lifetime_s": gen["job_lifetime_s"],
    }


def check(task: Path) -> list[str]:
    """Every caller against the generated values; one line per difference."""
    gen = generate(task)
    want = scalars(gen)
    out = []
    cfg = json.loads((task / "tests" / "config.json").read_text())
    have = cfg["infb"]["efficiency"]["deadlines"]
    w = want["config.efficiency.deadlines"]
    for k in ("launch_s", "teardown_s"):
        if have.get(k) != w[k]:
            out.append(f"config.json infb.efficiency.deadlines.{k} is {have.get(k)!r}, the DAG gives {w[k]}")
    if have.get("nodes") != w["nodes"]:
        out.append(f"config.json infb.efficiency.deadlines.nodes differ from the DAG: have {have.get('nodes')}, "
                   f"want {w['nodes']}")
    if "node_default_s" in have:
        out.append("config.json infb.efficiency.deadlines.node_default_s: every node is bounded by name")
    if cfg["infb"]["deadline_dag"].get("generated") != want:
        out.append("config.json infb.deadline_dag.generated is not the generator's output")
    toml = tomllib.loads((task / "task.toml").read_text())
    if toml["verifier"]["timeout_sec"] != want["task.toml.verifier.timeout_sec"]:
        out.append(f"task.toml [verifier] timeout_sec is {toml['verifier']['timeout_sec']}, the DAG gives "
                   f"{want['task.toml.verifier.timeout_sec']}")
    sh = (task / "tests" / "test.sh").read_text()
    for key, val in want["test.sh"].items():
        m = re.search(rf"^{key.upper()}=(\d+)\b", sh, flags=re.M)
        if not m or int(m.group(1)) != val:
            out.append(f"tests/test.sh {key.upper()} is {m.group(1) if m else 'absent'}, the DAG gives {val}")
    return out


def write(task: Path) -> dict:
    gen = generate(task)
    want = scalars(gen)
    path = task / "tests" / "config.json"
    cfg = json.loads(path.read_text())
    d = cfg["infb"]["efficiency"]["deadlines"]
    d.pop("node_default_s", None)
    d.update(want["config.efficiency.deadlines"])
    cfg["infb"]["deadline_dag"]["generated"] = want
    path.write_text(json.dumps(cfg, indent=1) + "\n")
    return want


def main(argv: list[str]) -> int:
    task = Path(argv[1]).resolve()
    if "--write" in argv:
        print(json.dumps(write(task), indent=1))
        return 0
    if "--check" in argv:
        problems = check(task)
        print("\n".join(problems) if problems else "every caller carries the DAG's generated values")
        return 1 if problems else 0
    print(json.dumps(generate(task), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
