#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The verifier's deadlines, derived from the public workload budgets.

`eval/qwen38/budgets.json` is the one source of every bound in this verifier: pytest alarms, client
and subprocess timeouts, group budgets and the task's verifier timeout. This module holds the
private part: the mapping from this verifier's nodes to those semantic operations, and the phase DAG
that composes them. Serial phases add; nothing here runs concurrently, so no maximum is taken. Every
allowed attempt, reread and redraw is inside its operation's budget (budgets.json says so for each).

    python3 -I budgets.py               the complete arithmetic (JSON)
    python3 -I budgets.py group NAME    a group's budget, for `deadline.py run --budget-s`
    python3 -I budgets.py outer NAME    that budget plus the report/cleanup allowance, for `timeout`
    python3 -I budgets.py phase NAME    a frame phase's bound (the reference phase)

`check()` fails when a scored node has no operation, a group names an unknown operation or the
task's verifier timeout differs from `max(42000, 60*ceil(1.10*composed/60))`.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TESTS_DIR", str(HERE.parent)))
BUDGETS = ROOT / "eval/qwen38/budgets.json"

#: Allowances of the composition itself (not workloads): pytest start, imports, collection and junit
#: writing inside a group's budget; the report/cleanup allowance and kill grace outside it; the
#: reference phase's process allowance and kill grace.
GROUP_PROCESS_S = 300
GROUP_REPORT_CLEANUP_S = 300
KILL_GRACE_S = 30
REFERENCE_PROCESS_S = 150
#: The shared verifier frame's own fixed bounds (test.sh, synced from the frame): `grader.py prepare`
#: and the trusted restore (300 s each, 10 s kill grace), preflight (600 s, 30 s), grading (110 s, 10 s).
FRAME = {"prepare": 300 + 10, "trusted_restore": 300 + 10, "preflight": 600 + 30, "grading": 110 + 10}
VERIFIER_IMAGE_BUILD_S = 3600
TIMEOUT_FLOOR_S = 42000
OUTER_MARGIN = 1.10
REFERENCE_DRAWS = 3
SUPPLIED_IMPORTS = 2

COMPONENT_NODES = {
    "ordinary": ("test_qsa_mathematics_and_state", "test_qsa_prefill_efficiency", "test_qsa_decode_efficiency",
                 "test_ple_mathematics_and_state", "test_ple_prefill_efficiency", "test_ple_decode_efficiency",
                 "test_ple_workspace_memory", "test_native_component_binding"),
    # MTP3 QSA prefill timing is excluded: no node, no slot.
    "mtp3": ("test_qsa_mathematics_and_state", "test_qsa_decode_efficiency",
             "test_ple_mathematics_and_state", "test_ple_prefill_efficiency", "test_ple_decode_efficiency",
             "test_ple_workspace_memory", "test_native_component_binding"),
}

#: node (junit short classname + name) -> the operations its call phase performs.
NODES = {
    "test_hidden.test_0_efficiency_decode_c1": ["ordinary_c1_draw"],
    "test_hidden.test_0_efficiency_throughput": ["ordinary_c32_draw"],
    "test_hidden.test_served_configuration": ["served_configuration_read"],
    "test_hidden.test_api_probes": ["api_observation_batch"],
    "test_hidden.test_prefix_reuse_witness": ["prefix_reuse_witness"],
    "test_hidden.test_prefix_reuse_agreement": ["prefix_agreement"],
    "test_hidden.test_fresh_request_state": ["fresh_state_evaluation"],
    "test_hidden.test_greedy_series": ["comparison_series_recording"],
    **{f"test_hidden.{n}": ["component_verdict"] for n in COMPONENT_NODES["ordinary"]},
    "test_fidelity.test_fidelity_kl": ["fidelity_evaluation"],
    "test_mtp.test_served_configuration": ["served_configuration_read"],
    "test_mtp.test_0_efficiency_decode_c1": ["mtp3_c1_draw"],
    "test_mtp.test_0_efficiency_throughput": ["mtp3_c32_draw"],
    "test_mtp.test_api_probes": ["api_observation_batch"],
    "test_mtp.test_fidelity_kl": ["fidelity_evaluation"],
    "test_mtp.test_mtp_equivalence": ["mtp_equivalence"],
    "test_mtp.test_mtp_acceptance": ["mtp_acceptance"],
    **{f"test_mtp.{n}": ["component_verdict"] for n in COMPONENT_NODES["mtp3"]},
    "test_native.test_production_and_state[mtp3]": ["mtp3_production_state"],
    "test_accuracy.test_qwen4_exp_accuracy[Qwen3.8-Flash-Next-FP8]": ["accuracy_including_launch"],
}
#: nodes whose setup builds the session's component record (collection runs in fixture setup).
COLLECTING = {f"test_hidden.{n}" for n in COMPONENT_NODES["ordinary"]} | \
             {f"test_mtp.{n}" for n in COMPONENT_NODES["mtp3"]}

#: pytest groups in test.sh order: launches, fixture phases, node calls and teardowns, all serial.
GROUPS = {
    "l0": {"phases": [("offline_rebuild", 1), ("post_build_import", 1), ("registry_resolution", 1),
                      ("config_parse", 1), ("supplied_import", SUPPLIED_IMPORTS)], "nodes": []},
    "l3": {"phases": [("l3_regression_group", 1)], "nodes": [], "process_allowance": False},
    "test_hidden": {"phases": [("server_launch", 1), ("component_collection", 1), ("server_cleanup", 1)],
                    "nodes": [k for k in NODES if k.startswith(("test_hidden.", "test_fidelity."))]},
    "test_mtp": {"phases": [("server_launch", 1), ("component_collection", 1), ("server_cleanup", 1)],
                 "nodes": [k for k in NODES if k.startswith("test_mtp.")]},
    "native": {"phases": [("server_launch", 1), ("server_cleanup", 1)],
               "nodes": ["test_native.test_production_and_state[mtp3]"]},
    "l2_accuracy": {"phases": [], "nodes": ["test_accuracy.test_qwen4_exp_accuracy[Qwen3.8-Flash-Next-FP8]"]},
}
REFERENCE = {
    "test_hidden": [("server_launch", 1), ("ordinary_c1_draw", REFERENCE_DRAWS), ("ordinary_c32_draw", REFERENCE_DRAWS),
                    ("component_collection", 1), ("server_cleanup", 1)],
    "test_mtp": [("server_launch", 1), ("mtp3_c1_draw", REFERENCE_DRAWS), ("mtp3_c32_draw", REFERENCE_DRAWS),
                 ("component_collection", 1), ("server_cleanup", 1)],
}
AXES = {"test_hidden": ("ordinary_c1_draw", "ordinary_c32_draw"), "test_mtp": ("mtp3_c1_draw", "mtp3_c32_draw")}


def operations(path: Path | None = None) -> dict[str, int]:
    data = json.loads(Path(path or BUDGETS).read_text())
    if data.get("schema") != "qwen38-budgets/1":
        raise ValueError(f"{path or BUDGETS}: unexpected budgets schema")
    out = {}
    for name, row in data["operations"].items():
        seconds = row["seconds"]
        if type(seconds) is not int or seconds <= 0:
            raise ValueError(f"{name}: a budget is a positive integer number of seconds")
        out[name] = seconds
    return out


def seconds(operation: str) -> int:
    return operations()[operation]


def node_seconds(node: str, ops: dict | None = None) -> int:
    ops = ops or operations()
    return sum(ops[o] for o in NODES[node])


def deadlines(recipe: str | None = None) -> dict:
    """The resolved deadlines in the form every reader takes (plugin, kit, reference phase, L0)."""
    ops = operations()
    out = {"source": str(BUDGETS), "launch_s": ops["server_launch"], "teardown_s": ops["server_cleanup"],
           "node_default_s": ops["component_verdict"],
           "components_s": ops["component_collection"], "component_matrix_s": ops["component_matrix"],
           "nodes": {k: node_seconds(k, ops) for k in NODES},
           "setup": {k: ops["server_launch"] + ops["component_collection"] for k in COLLECTING},
           "l0": {"build_editable_install": ops["offline_rebuild"], "import_after_build": ops["post_build_import"],
                  "registry_resolves": ops["registry_resolution"], "config_parses": ops["config_parse"],
                  "supplied_kernels_import": ops["supplied_import"]}}
    c1, c32 = AXES.get(recipe or "test_hidden", AXES["test_hidden"])
    out["reference_decode_c1_s"], out["reference_throughput_s"] = ops[c1], ops[c32]
    return out


def group_budget(name: str, ops: dict | None = None) -> dict:
    ops = ops or operations()
    group = GROUPS[name]
    rows = [{"operation": op, "count": n, "seconds": ops[op] * n} for op, n in group["phases"]]
    rows += [{"node": node, "operations": NODES[node], "seconds": node_seconds(node, ops)} for node in group["nodes"]]
    total = sum(r["seconds"] for r in rows)
    allowance = GROUP_PROCESS_S if group.get("process_allowance", True) else 0
    budget = total + allowance
    return {"group": name, "phases": rows, "phases_s": total, "process_allowance_s": allowance, "budget_s": budget,
            "outer_s": budget + GROUP_REPORT_CLEANUP_S, "kill_grace_s": KILL_GRACE_S,
            "composed_s": budget + GROUP_REPORT_CLEANUP_S + KILL_GRACE_S}


def reference_phase(ops: dict | None = None) -> dict:
    ops = ops or operations()
    recipes = {name: [{"operation": op, "count": n, "seconds": ops[op] * n} for op, n in phases]
               for name, phases in REFERENCE.items()}
    total = sum(r["seconds"] for rows in recipes.values() for r in rows)
    bound = total + REFERENCE_PROCESS_S
    return {"recipes": recipes, "phases_s": total, "process_allowance_s": REFERENCE_PROCESS_S, "bound_s": bound,
            "kill_grace_s": KILL_GRACE_S, "composed_s": bound + KILL_GRACE_S}


def composition() -> dict:
    """The complete verifier DAG and its arithmetic."""
    ops = operations()
    groups = {name: group_budget(name, ops) for name in GROUPS}
    reference = reference_phase(ops)
    parts = {**{f"frame:{k}": v for k, v in FRAME.items()}, "reference_phase": reference["composed_s"],
             **{f"group:{k}": g["composed_s"] for k, g in groups.items()},
             "verifier_image_build": VERIFIER_IMAGE_BUILD_S}
    composed = sum(parts.values())
    timeout = max(TIMEOUT_FLOOR_S, 60 * math.ceil(OUTER_MARGIN * composed / 60))
    return {"schema": "qwen38-deadline-composition/1", "budgets": str(BUDGETS), "operations": ops,
            "frame": FRAME, "reference_phase": reference, "groups": groups, "parts": parts,
            "composed_s": composed,
            "rule": "verifier timeout = max(42000, 60*ceil(1.10*composed_s/60)); every phase serial",
            "verifier_timeout_s": timeout}


def check(task_toml: Path | None = None, config: Path | None = None) -> dict:
    """Static validation of the composition against the task's own files."""
    result = composition()
    ops = result["operations"]
    missing = [o for node in NODES for o in NODES[node] if o not in ops]
    missing += [op for g in GROUPS.values() for op, _ in g["phases"] if op not in ops]
    missing += [op for rows in REFERENCE.values() for op, _ in rows if op not in ops]
    if missing:
        raise ValueError(f"operations missing from budgets.json: {sorted(set(missing))}")
    grouped = [n for g in GROUPS.values() for n in g["nodes"]]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(NODES):
        raise ValueError("every node belongs to exactly one group")
    if config is not None:
        cfg = json.loads(Path(config).read_text())
        scored = [i for i in cfg["f2p_node_ids"] + cfg["p2p_node_ids"] if i.startswith("infb_verify.") and
                  not i.startswith("infb_verify.test_l0.")]
        named = {f"{i.split('.')[1]}.{i.split('.', 2)[2]}" for i in scored}
        unnamed = sorted(n for n in named if n not in NODES)
        if unnamed:
            raise ValueError(f"scored nodes without an operation: {unnamed}")
        prefix = "tests.evals.qwen4_exp.test_accuracy."
        for i in cfg["f2p_node_ids"]:
            if i.startswith("tests.evals.") and f"test_accuracy.{i.removeprefix(prefix)}" not in NODES:
                raise ValueError(f"accuracy node without an operation: {i}")
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
    elif what == "check":
        check(Path(argv[1]) if len(argv) > 1 else None, Path(argv[2]) if len(argv) > 2 else None)
        print("ok")
    else:
        raise SystemExit(f"usage: budgets.py [group|outer NAME | phase reference | check TOML CONFIG]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
