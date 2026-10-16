#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The verifier's deadlines: the operation limits composed over the verification DAG.

`eval/minimax_m3/limits.json` (public; restored from /tests/eval before any candidate code runs) holds the operation
caps. This module holds the private part: which operations each scored node and each phase performs, the serial DAG
that composes them, and the measured reference walls `R(node)` (`config.json`
`infb.deadlines.measured`, each with the run it came from).

  node bound   B(node) = max(3 R(node), Cmax(node) + 60 s) once R is measured, else Σ caps(node) + 60 s;
               a node whose policy permits whole-node repetitions adds one B per repetition
  group        serial: Σ phases + Σ B + 60 s
  verifier     root (frame + reference phase + groups) + 600 s
  job          agent + build allowances + verifier + 600 s

  python3 -I deadline_dag.py               the complete arithmetic (JSON)
  python3 -I deadline_dag.py group NAME    a group's budget, for `deadline.py run --budget-s`
  python3 -I deadline_dag.py outer NAME    that budget plus the report/cleanup allowance, for `timeout`
  python3 -I deadline_dag.py phase reference
  python3 -I deadline_dag.py check TOML CONFIG
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TESTS_DIR", str(HERE.parent)))
LIMITS = ROOT / "eval/minimax_m3/limits.json"
CONFIG = ROOT / "config.json"

DISPATCH_S = 60                # per node and per serial section, dispatch and recording
GROUP_REPORT_CLEANUP_S = 300   # the runner's report writing and reaping outside a group's budget
KILL_GRACE_S = 30
VERIFIER_FINAL_S = 600         # final artifact and result handling
JOB_FINAL_S = 600
REFERENCE_DRAWS = 3
#: The shared frame's fixed bounds in test.sh: `grader.py prepare` and the trusted restore (300 s + 10 s grace each),
#: preflight (600 s + 30 s), the disposition (60 s), grading (110 s + 10 s).
FRAME = {"prepare": 310, "trusted_restore": 310, "preflight": 630, "disposition": 60, "grading": 120}

L3_FILES = (
    "tests/v1/core/test_kv_cache_utils.py", "tests/v1/core/test_prefix_caching.py",
    "tests/v1/core/test_single_type_kv_cache_manager.py", "tests/v1/core/test_scheduler.py",
    "tests/v1/worker/test_gpu_input_batch.py", "tests/kernels/core/test_activation.py",
    "tests/kernels/test_fp32_router_gemm.py", "tests/kernels/core/test_fused_allreduce_gemma_rms_norm.py",
    "tests/kernels/moe/test_cutlass_mxfp8_grouped_mm.py", "tests/kernels/moe/test_deepgemm.py",
    "tests/kernels/moe/test_block_fp8.py", "tests/quantization/test_fp8.py",
    "tests/v1/attention/test_attention_backends.py", "tests/models/test_registry.py", "tests/reasoning/",
    "tests/tool_parsers/", "tests/v1/spec_decode/test_eagle.py",
)
RUST_CRATES = ("vllm-reasoning-parser", "vllm-tool-parser", "vllm-chat")
DEPLOYMENTS = ("bf16kv", "fp8kv")
#: Component families timed by each efficiency node, and the sparsity pairs (msa/measure.py, msa/sparsity.py).
PREFILL_FAMILIES = ("select_prefill", "attend_prefill_bf16", "attend_prefill_fp8")
DECODE_FAMILIES = ("select_decode", "attend_decode_bf16", "attend_decode_fp8")
SPARSITY_WORKLOADS = 4
#: The poisoned-cache geometries (msa/sparsity.py POISON), one component invocation each, when the policy names them.
POISON_WORKLOADS = 6


def limits(path: Path | None = None) -> dict[str, int]:
    data = json.loads(Path(path or LIMITS).read_text())
    if data.get("schema") != "infb-m3-limits/1":
        raise ValueError(f"{path or LIMITS}: unexpected limits schema")
    out = {k: v for k, v in data.items() if k.endswith("_s")}
    for name, seconds in out.items():
        if type(seconds) is not int or seconds <= 0:
            raise ValueError(f"{name}: a limit is a positive integer number of seconds")
    return out


def _msa_case_counts() -> dict[str, int]:
    """Private case count per MSA correctness node, from the recorded case identities (tests/hidden/msa_cases.json),
    so the composition needs no torch."""
    rec = json.loads((ROOT / "hidden" / "msa_cases.json").read_text())
    return {node: len(rows) for node, rows in rec["nodes"].items()}


def node_operations(lim: dict[str, int]) -> dict[str, dict]:
    """junit short classname + name -> {"ops": [cap, ...], "repetitions": n}."""
    H, HL, A, C, T = (lim["http_workload_s"], lim["http_workload_long_s"], lim["action_s"],
                      lim["component_invocation_s"], lim["test_file_s"])
    cases = _msa_case_counts()
    nodes = {
        "test_l0.test_build_editable_install": {"ops": [lim["rebuild_s"], A]},
        "test_l0.test_supplied_kernels_import": {"ops": [C] * 3},
        "test_l0.test_registry_resolves_native_class": {"ops": [A]},
        "test_l0.test_config_parses": {"ops": [A]},
        "test_msa.test_select_prefill_correctness": {"ops": [C] * cases["test_select_prefill_correctness"]},
        "test_msa.test_select_decode_correctness": {"ops": [C] * cases["test_select_decode_correctness"]},
        "test_msa.test_attend_prefill_correctness": {"ops": [C] * cases["test_attend_prefill_correctness"]},
        "test_msa.test_attend_decode_correctness": {"ops": [C] * cases["test_attend_decode_correctness"]},
        "test_msa.test_prefill_efficiency": {"ops": [C] * len(PREFILL_FAMILIES)},
        "test_msa.test_decode_efficiency": {"ops": [C] * len(DECODE_FAMILIES)},
        "test_msa.test_sparsity_realized": {"ops": [C] * (SPARSITY_WORKLOADS + POISON_WORKLOADS)},
        **{f"test_parsers.{n}": {"ops": [A]} for n in ("test_reasoning_parser", "test_reasoning_parser_streaming",
                                                        "test_tool_parser", "test_tool_parser_streaming")},
    }
    for dep in DEPLOYMENTS:
        m = f"test_{dep}"
        nodes.update({
            # best of up to three attempts: two whole-node repetitions
            f"{m}.test_0_efficiency_decode_c1": {"ops": [C], "repetitions": 2},
            f"{m}.test_0_efficiency_throughput": {"ops": [C] * 3},
            f"{m}.test_served_configuration": {"ops": [A]},
            # text, token ids, streamed, chat, streamed chat, one batch of eight
            f"{m}.test_api_probes": {"ops": [H] * 6},
            f"{m}.test_chat_reasoning": {"ops": [H] * 12},
            f"{m}.test_tool_calls": {"ops": [H] * 8},
            # nine C=1 generations, the C=3 and C=5 batches, the reset, 17 replays
            f"{m}.test_generation_replay": {"ops": [H] * 11 + [A] + [H] * 17},
            # per group: reset, four generations, reset, four replays (with the metric reads)
            f"{m}.test_prefix_branch_replay": {"ops": ([A] + [H] * 4 + [A] + [H] * 4) * 4},
            # 24 sequences below 64,000 tokens and 8 of 65,536; a band may be re-drawn once (the whole node at worst)
            f"{m}.test_author_fidelity": {"ops": [H] * 24 + [HL] * 8, "repetitions": 1},
            f"{m}.test_engine_agreement_262k": {"ops": [HL] * 2},
            f"{m}.test_gsm8k_chat_thinking": {"ops": [lim["gsm8k_run_s"]]},
            # start the profiler, one 6,144-token request, one batch of three, stop, read four ranks' traces
            f"{m}.test_execution_witness": {"ops": [A, H, H, A, A]},
        })
    return nodes


def groups(lim: dict[str, int]) -> dict[str, dict]:
    """pytest groups in test.sh order; phases are operations outside any node."""
    nodes = node_operations(lim)
    return {
        "l0": {"phases": [], "nodes": [n for n in nodes if n.startswith("test_l0.")]},
        "msa": {"phases": [], "nodes": [n for n in nodes if n.startswith("test_msa.")]},
        "parsers": {"phases": [], "nodes": [n for n in nodes if n.startswith("test_parsers.")]},
        "l3": {"phases": [("l3_file", lim["test_file_s"], len(L3_FILES))], "nodes": []},
        "rust": {"phases": [("rust_crate", lim["test_file_s"], len(RUST_CRATES))], "nodes": []},
        **{dep: {"phases": [("server_launch", lim["startup_s"], 1), ("server_cleanup", lim["teardown_s"], 1)],
                 "nodes": [n for n in nodes if n.startswith(f"test_{dep}.")]} for dep in DEPLOYMENTS},
    }


def reference_phase(lim: dict[str, int]) -> list[tuple[str, int, int]]:
    """The reference phase before any candidate code: the Triton arm's component timing on GPU 0 (the six timing
    families the efficiency nodes compare against; sparsity is graded against the frozen bounds, not a reference
    reading), then per deployment one reference launch serving three draws per whole-serving metric."""
    rows = [("triton_arm_component_timing", lim["component_invocation_s"], len(PREFILL_FAMILIES) + len(DECODE_FAMILIES))]
    for dep in DEPLOYMENTS:
        rows += [(f"{dep}:server_launch", lim["startup_s"], 1),
                 (f"{dep}:decode_c1_draw", lim["component_invocation_s"], REFERENCE_DRAWS),
                 (f"{dep}:throughput_draw", lim["component_invocation_s"], REFERENCE_DRAWS),
                 (f"{dep}:server_cleanup", lim["teardown_s"], 1)]
    return rows


def measured(config: Path | None = None) -> dict[str, dict]:
    path = Path(config or CONFIG)
    if not path.is_file():
        return {}
    return ((json.loads(path.read_text()).get("infb") or {}).get("deadlines") or {}).get("measured") or {}


def node_bound(name: str, spec: dict, meas: dict) -> dict:
    ops = spec["ops"]
    cmax, total = max(ops), sum(ops)
    reps = spec.get("repetitions", 0)
    row = meas.get(name)
    if row is not None:
        once = max(math.ceil(3 * row["R_s"]), cmax + DISPATCH_S)
        rule = "max(3R, Cmax + 60)"
    else:
        once = total + DISPATCH_S
        rule = "sum of caps + 60 (R not measured)"
    return {"node": name, "operations": len(ops), "Cmax_s": cmax, "caps_s": total, "R_s": (row or {}).get("R_s"),
            "R_run": (row or {}).get("run"), "repetitions": reps, "once_s": once, "bound_s": once * (1 + reps),
            "rule": rule}


def group_budget(name: str, lim: dict | None = None, meas: dict | None = None) -> dict:
    lim = lim or limits()
    meas = measured() if meas is None else meas
    g = groups(lim)[name]
    ops = node_operations(lim)
    phases = [{"phase": p, "cap_s": s, "count": n, "seconds": s * n} for p, s, n in g["phases"]]
    rows = [node_bound(n, ops[n], meas) for n in g["nodes"]]
    budget = sum(p["seconds"] for p in phases) + sum(r["bound_s"] for r in rows) + DISPATCH_S
    return {"group": name, "phases": phases, "nodes": rows, "budget_s": budget,
            "outer_s": budget + GROUP_REPORT_CLEANUP_S, "composed_s": budget + GROUP_REPORT_CLEANUP_S + KILL_GRACE_S}


def reference_bound(lim: dict | None = None) -> dict:
    lim = lim or limits()
    rows = [{"operation": p, "cap_s": s, "count": n, "seconds": s * n} for p, s, n in reference_phase(lim)]
    bound = sum(r["seconds"] for r in rows) + DISPATCH_S
    return {"rows": rows, "bound_s": bound, "composed_s": bound + KILL_GRACE_S}


def deadlines() -> dict:
    """The resolved deadlines in the form every reader takes (the pytest plugin, L0, the serving groups)."""
    lim = limits()
    meas = measured()
    ops = node_operations(lim)
    nodes = {n: node_bound(n, spec, meas)["once_s"] for n, spec in ops.items()}
    return {"source": str(LIMITS), "limits": lim, "launch_s": lim["startup_s"], "teardown_s": lim["teardown_s"],
            "node_default_s": lim["action_s"], "nodes": nodes,
            # one reference draw of a whole-serving metric is one measurement invocation
            "reference_decode_c1_s": lim["component_invocation_s"],
            "reference_throughput_s": lim["component_invocation_s"],
            "l0": {"build_editable_install": lim["rebuild_s"], "import_after_build": lim["action_s"],
                   "registry_resolves": lim["action_s"], "config_parses": lim["action_s"],
                   "supplied_kernels_import": lim["component_invocation_s"]}}


def composition(config: Path | None = None) -> dict:
    lim = limits()
    meas = measured(config)
    gs = {name: group_budget(name, lim, meas) for name in groups(lim)}
    ref = reference_bound(lim)
    parts = {**{f"frame:{k}": v for k, v in FRAME.items()}, "reference_phase": ref["composed_s"],
             **{f"group:{k}": g["composed_s"] for k, g in gs.items()}}
    root = sum(parts.values()) + DISPATCH_S
    return {"schema": "infb-m3-deadline-composition/1", "limits": lim, "frame": FRAME, "reference_phase": ref,
            "groups": gs, "parts": parts, "root_s": root, "verifier_timeout_s": root + VERIFIER_FINAL_S,
            "rule": "B(node) = max(3R, Cmax+60) (R measured) else caps+60; groups and root serial (+60); "
                    "verifier = root + 600; job = agent + build allowances + verifier + 600"}


def job_lifetime(agent_s: int, build_allowances_s: int, config: Path | None = None) -> int:
    return agent_s + build_allowances_s + composition(config)["verifier_timeout_s"] + JOB_FINAL_S


def check(task_toml: Path | None = None, config: Path | None = None) -> dict:
    result = composition(config)
    lim = result["limits"]
    ops = node_operations(lim)
    grouped = [n for g in groups(lim).values() for n in g["nodes"]]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(ops):
        raise ValueError("every node belongs to exactly one group")
    if config is not None:
        cfg = json.loads(Path(config).read_text())
        scored = [i for i in cfg["f2p_node_ids"] + cfg["p2p_node_ids"] if i.startswith("infb_verify.")]
        named = {i.removeprefix("infb_verify.") for i in scored}
        missing = sorted(n for n in named if n not in ops)
        if missing:
            raise ValueError(f"scored nodes without operations: {missing}")
    if task_toml is not None:
        import tomllib  # noqa: PLC0415
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
        print(reference_bound()["bound_s"])
    elif what == "check":
        check(Path(argv[1]) if len(argv) > 1 else None, Path(argv[2]) if len(argv) > 2 else None)
        print("ok")
    else:
        raise SystemExit("usage: deadline_dag.py [group|outer NAME | phase reference | check TOML CONFIG]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
