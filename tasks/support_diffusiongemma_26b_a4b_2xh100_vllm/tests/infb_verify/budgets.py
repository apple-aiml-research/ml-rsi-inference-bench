#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The verifier's deadlines, derived from the public operation limits.

`eval/diffusion_gemma/budgets.json` is the one source of every bound in this verifier: pytest alarms,
client and subprocess timeouts, group budgets and the task's verifier timeout. This module holds the
private part -- the map from this verifier's nodes to those operations, and the phase DAG that
composes them:

  * node bound B(node) = max(3R, Cmax + 60 s), R the reference's worst measured wall for the node
    over the confirmation runs (tests/hidden/reference_walls.json once confirmed; before R exists
    every node takes Cmax + 60 s), Cmax the node's largest operation cap;
  * a reads node permits one band redraw, which adds one B(node); the step-1 node's increment draws the
    one-step read of each canvas with its own redraw, which doubles that;
  * launches, cleanups and collections are phases of their group, bounded like nodes; so is the preemption
    node's permitted re-run at the heavier load, which adds its own cap + 60 s;
  * a serial section is the sum of its bounds plus 60 s; [verifier] timeout_sec = root + 600 s.

    python3 -I budgets.py               the complete arithmetic (JSON)
    python3 -I budgets.py group NAME    a group's budget, for `deadline.py run --budget-s`
    python3 -I budgets.py outer NAME    that budget plus the report/cleanup allowance, for `timeout`
    python3 -I budgets.py phase NAME    a frame phase's bound (the reference phase)
    python3 -I budgets.py op NAME       one operation's published cap
    python3 -I budgets.py check [TOML [CONFIG]]
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TESTS_DIR", str(HERE.parent)))
BUDGETS = ROOT / "eval/diffusion_gemma/budgets.json"
WALLS = ROOT / "hidden/reference_walls.json"

SECTION_S = 60             # every serial section adds 60 s
NODE_MARGIN_S = 60         # Cmax + 60 s
GROUP_REPORT_CLEANUP_S = 300
KILL_GRACE_S = 30
VERIFIER_MARGIN_S = 600    # [verifier] timeout_sec = composed root + 600 s
#: The shared frame's own fixed bounds (test.sh): `grader.py prepare` and the trusted restore (300 s
#: each, 10 s kill grace), preflight (600 s, 30 s), grading (110 s, 10 s).
FRAME = {"prepare": 300 + 10, "trusted_restore": 300 + 10, "preflight": 600 + 30, "grading": 110 + 10}

READS = ("test_denoise_reads", "test_image_reads", "test_step1_reads")
#: reads nodes that also draw the one-step read of each canvas, with its own redraw.
INCREMENT = ("test_step1_reads",)
#: operations a node carries as phases of its group (bounded like nodes, added to the node's bound):
#: a session's launch and the previous session's cleanup, an attend collection, and the preemption node's
#: permitted re-run at the heavier load.
PHASES = ("server_launch", "server_cleanup", "component_invocation", "preemption_rerun")

#: node (junit short classname + name) -> the operations its call performs. A session's first node
#: also carries the launch (SESSIONS starts servers lazily) and, after another session, its cleanup.
TP1 = {
    "test_0_efficiency_decode_c1_tp1": ["server_launch", "measurement_invocation"],
    "test_0_efficiency_throughput_tp1": ["measurement_invocation"],
    "test_denoise_policy_fixed_half": ["http_workload"],
    "test_served_configuration_tp1": ["server_cleanup", "server_launch", "identity_action", "http_workload"],
    "test_api_probes_tp1": ["http_workload"],
    "test_repair_structured_outputs": ["http_workload"],
    "test_repair_long_output": ["http_workload"],
    "test_repair_first_token": ["http_workload"],
    "test_concurrent_logprobs": ["http_workload"],
    "test_chat_reasoning": ["http_workload"],
    "test_tool_calls": ["http_workload"],
    "test_denoise_reads_tp1": ["http_workload_long"],
    "test_image_reads_tp1": ["http_workload"],
    "test_step1_reads_tp1": ["http_workload"],
    "test_commit_encode_replay_tp1": ["http_workload"],
    "test_stale_phase_leak_tp1": ["http_workload"],
    "test_prefix_branches": ["http_workload"],
    "test_mixed_batch_reads_tp1": ["http_workload"],
    "test_semantic_majority_tp1": ["http_workload"],
    "test_denoise_policy": ["http_workload"],
    "test_long_context_tp1": ["http_workload_long"],
    "test_gsm8k": ["gsm8k_run"],
    "test_execution_witness_tp1": ["http_workload"],
    "test_preemption_resume": ["preemption_workload", "preemption_rerun"],
}
TP2 = {
    "test_0_efficiency_decode_c1_tp2": ["server_launch", "measurement_invocation"],
    "test_0_efficiency_throughput_tp2": ["measurement_invocation"],
    "test_served_configuration_tp2": ["identity_action"],
    "test_api_probes_tp2": ["http_workload"],
    "test_denoise_reads_tp2": ["http_workload_long"],
    "test_image_reads_tp2": ["http_workload"],
    "test_step1_reads_tp2": ["http_workload"],
    "test_commit_encode_replay_tp2": ["http_workload"],
    "test_stale_phase_leak_tp2": ["http_workload"],
    "test_mixed_batch_reads_tp2": ["http_workload"],
    "test_semantic_majority_tp2": ["http_workload"],
    "test_long_context_tp2": ["http_workload_long"],
    "test_execution_witness_tp2": ["http_workload"],
}
ATTEND = {
    "test_attend_mask_witness": ["component_invocation", "component_verdict"],  # the correctness collection
    "test_attend_values_encode": ["component_verdict"],
    "test_attend_values_denoise_mixed": ["component_verdict"],
    "test_attend_encode_efficiency": ["component_invocation", "component_verdict"],  # the timing collection
    "test_attend_denoise_efficiency": ["component_verdict"],
    "test_attend_mixed_efficiency": ["component_verdict"],
}
NODES = {
    **{f"test_l0.{n}": ops for n, ops in {"test_build_editable_install": ["offline_rebuild", "identity_action"],
                                          "test_registry_resolves_native_class": ["identity_action"],
                                          "test_config_parses": ["identity_action"]}.items()},
    **{f"test_attend.{n}": ops for n, ops in ATTEND.items()},
    "test_parsers.test_gemma4_tool_parser_whole_canvas_streaming": ["identity_action"],
    **{f"test_tp1.{n}": ops for n, ops in TP1.items()},
    **{f"test_tp2.{n}": ops for n, ops in TP2.items()},
    "test_ar.test_ar_model_serves": ["server_launch", "http_workload"],
}
#: pytest groups in test.sh order; each group's phases beyond its nodes (a final cleanup).
GROUPS = {
    "l0": {"prefix": "test_l0.", "phases": []},
    "attend": {"prefix": "test_attend.", "phases": []},
    "parsers": {"prefix": "test_parsers.", "phases": []},
    "l3": {"prefix": None, "phases": [("l3_regression_group", 1)]},
    "tp1": {"prefix": "test_tp1.", "phases": [("server_cleanup", 1)], "sections": 2},  # two sessions in series
    "tp2": {"prefix": "test_tp2.", "phases": [("server_cleanup", 1)]},
    "ar": {"prefix": "test_ar.", "phases": [("server_cleanup", 1)]},
}
#: the reference phase: the component comparator arm, then each deployment's fixed
#: configuration launched with its C1 and C4 draws.
REFERENCE = [("component_invocation", 1),
             ("server_launch", 1), ("measurement_invocation", 2), ("server_cleanup", 1),   # tp1_fixed
             ("server_launch", 1), ("measurement_invocation", 2), ("server_cleanup", 1)]   # tp2_fixed


def operations(path: Path | None = None) -> dict[str, int]:
    data = json.loads(Path(path or BUDGETS).read_text())
    if data.get("schema") != "diffusion-gemma-budgets/1":
        raise ValueError(f"{path or BUDGETS}: unexpected budgets schema")
    out = {}
    for name, row in data["operations"].items():
        cap = row["cap_s"]
        if type(cap) is not int or cap <= 0:
            raise ValueError(f"{name}: a cap is a positive integer number of seconds")
        out[name] = cap
    return out


def seconds(operation: str) -> int:
    return operations()[operation]


def reference_walls() -> dict[str, float]:
    return json.loads(WALLS.read_text()) if WALLS.exists() else {}


def node_bound(node: str, ops: dict | None = None, walls: dict | None = None) -> int:
    """B(node) = max(3R, Cmax + 60), plus one more B for a reads node's band redraw; the phases a node
    carries (PHASES) are bounded like nodes and add their own cap + 60. R, when recorded, is the
    node's verdict wall without its phases."""
    ops = ops or operations()
    walls = reference_walls() if walls is None else walls
    work = [o for o in NODES[node] if o not in PHASES]
    phases = [o for o in NODES[node] if o in PHASES]
    cmax = max(ops[o] for o in work)
    r = walls.get(node)
    bound = max(3 * r, cmax + NODE_MARGIN_S) if r else cmax + NODE_MARGIN_S
    if node.split(".", 1)[1].rsplit("_tp", 1)[0] in READS:
        bound *= 2
    if node.split(".", 1)[1].rsplit("_tp", 1)[0] in INCREMENT:
        bound *= 2
    return int(math.ceil(bound + sum(ops[p] + NODE_MARGIN_S for p in phases)))


def deadlines() -> dict:
    """The resolved deadlines in the form deadline.py's plugin reads."""
    ops = operations()
    return {"source": str(BUDGETS), "launch_s": ops["server_launch"] + NODE_MARGIN_S,
            "teardown_s": ops["server_cleanup"] + NODE_MARGIN_S,
            "node_default_s": ops["component_verdict"] + NODE_MARGIN_S,
            "nodes": {k: node_bound(k, ops) for k in NODES}, "setup": {}}


def group_budget(name: str, ops: dict | None = None) -> dict:
    ops = ops or operations()
    group = GROUPS[name]
    rows = [{"operation": op, "count": n, "seconds": (ops[op] + NODE_MARGIN_S) * n} for op, n in group["phases"]]
    if group["prefix"]:
        rows += [{"node": k, "operations": NODES[k], "seconds": node_bound(k, ops)} for k in NODES
                 if k.startswith(group["prefix"])]
    total = sum(r["seconds"] for r in rows) + SECTION_S * group.get("sections", 1)
    return {"group": name, "phases": rows, "budget_s": total, "outer_s": total + GROUP_REPORT_CLEANUP_S,
            "kill_grace_s": KILL_GRACE_S, "composed_s": total + GROUP_REPORT_CLEANUP_S + KILL_GRACE_S}


def reference_phase(ops: dict | None = None) -> dict:
    ops = ops or operations()
    rows = [{"operation": op, "count": n, "seconds": (ops[op] + NODE_MARGIN_S) * n} for op, n in REFERENCE]
    bound = sum(r["seconds"] for r in rows) + SECTION_S
    return {"phases": rows, "bound_s": bound, "kill_grace_s": KILL_GRACE_S, "composed_s": bound + KILL_GRACE_S}


def composition() -> dict:
    ops = operations()
    groups = {name: group_budget(name, ops) for name in GROUPS}
    reference = reference_phase(ops)
    parts = {**{f"frame:{k}": v for k, v in FRAME.items()}, "reference_phase": reference["composed_s"],
             **{f"group:{k}": g["composed_s"] for k, g in groups.items()}}
    root = sum(parts.values()) + SECTION_S
    return {"schema": "diffusion-gemma-deadline-composition/1", "budgets": str(BUDGETS), "operations": ops,
            "walls": str(WALLS) if WALLS.exists() else None, "frame": FRAME, "reference_phase": reference,
            "groups": groups, "parts": parts, "composed_s": root,
            "rule": "B(node) = max(3R, Cmax + 60); serial sections add 60; timeout = root + 600",
            "verifier_timeout_s": root + VERIFIER_MARGIN_S}


def check(task_toml: Path | None = None, config: Path | None = None) -> dict:
    result = composition()
    ops = result["operations"]
    missing = sorted({o for ops_ in NODES.values() for o in ops_ if o not in ops} |
                     {op for g in GROUPS.values() for op, _ in g["phases"] if op not in ops} |
                     {op for op, _ in REFERENCE if op not in ops})
    if missing:
        raise ValueError(f"operations missing from budgets.json: {missing}")
    if config is not None:
        cfg = json.loads(Path(config).read_text())
        scored = [i for i in cfg["f2p_node_ids"] + cfg["p2p_node_ids"] if i.startswith("infb_verify.")]
        unnamed = sorted(i for i in scored if i.split(".", 1)[1] not in NODES)
        if unnamed:
            raise ValueError(f"scored nodes without an operation: {unnamed}")
    if task_toml is not None:
        import tomllib
        declared = tomllib.loads(Path(task_toml).read_text())["verifier"]["timeout_sec"]
        if declared != result["verifier_timeout_s"]:
            raise ValueError(f"task.toml verifier timeout {declared} != composed {result['verifier_timeout_s']}")
    return result


def main(argv: list[str]) -> int:
    if not argv:
        print(json.dumps(composition(), indent=1))
        return 0
    what, name = argv[0], (argv[1] if len(argv) > 1 else None)
    if what == "group":
        print(group_budget(name)["budget_s"])
    elif what == "outer":
        print(group_budget(name)["outer_s"])
    elif what == "phase" and name == "reference":
        print(reference_phase()["bound_s"])
    elif what == "op":
        print(seconds(name))
    elif what == "node":
        print(node_bound(name))
    elif what == "check":
        check(Path(argv[1]) if len(argv) > 1 else None, Path(argv[2]) if len(argv) > 2 else None)
        print("ok")
    else:
        raise SystemExit("usage: budgets.py [group|outer NAME | phase reference | op NAME | node NAME | check [TOML [CONFIG]]]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
