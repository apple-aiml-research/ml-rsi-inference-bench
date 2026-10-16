#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The component phases around `kernel_worker.py` and `kernel_verdict.py`.

    python -I kernel_phase.py reference      # test.sh step reference_kernels, before any candidate code runs
    python kernel_phase.py fault <node id> <reason>   # test.sh: a kernel node's wall expired within its limits

The reference phase runs the restored human reference (the tree snapshot.py built into the verifier image, its
vendored FlashMLA built in place) through every correctness node and every efficiency family, on the device and with
the JIT cache root the reference arm uses for serving. The reference must pass every private case; a reference that
fails one, or that cannot be measured, is a verifier/host fault (exit 3: no reward, the -1 sentinel), never a
candidate result. The candidate's nodes (`test_kernels.py`) call `candidate_node`.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = Path(os.environ.get("TESTS_DIR", str(HERE.parent)))
sys.path.insert(0, str(HERE))
import kernel_cases as kc  # noqa: E402
import kernel_verdict as kv  # noqa: E402

VERIFIER_FAULT = 3
FAULTS = Path(os.environ.get("INFB_VERIFIER_DIR", "/logs/verifier")) / "verifier-faults.json"


def config() -> dict:
    return json.loads((TESTS / "config.json").read_text())


def kcfg() -> dict:
    return config()["infb"]["kernels"]


def op_limit(kind: str) -> int:
    ops = json.loads((TESTS / "eval" / "deepseek_v4_flash" / "limits.json").read_text())["operations"]
    return int(ops[kind]["limit_s"])


def worker_limit(node: str) -> int:
    """The worker's limit: one component_check operation for a correctness node (loading the arm and every case); for
    an efficiency node one invocation per family (each also enforced inside the worker) and one action for loading the
    arm."""
    if node in kc.CORRECTNESS_NODES:
        return op_limit("component_check")
    return op_limit("invocation") * len(kc._timing_families(kc.EFFICIENCY_NODES[node])) + op_limit("action")


def paths(arm: str, node: str) -> dict:
    k = kcfg()
    return {"out": Path(k["work"]) / arm / node, "records": Path(k["records"]) / arm / f"{node}.jsonl",
            "log": Path(k["records"]) / arm / f"{node}.log", "verdict": Path(k["records"]) / arm / f"{node}.verdict.json"}


def run_worker(arm: str, node: str, tree: str, env_extra: dict) -> dict:
    p = paths(arm, node)
    for d in (p["out"], p["records"].parent):
        d.mkdir(parents=True, exist_ok=True)
    for f in p["out"].glob("*.pt"):
        f.unlink()
    p["records"].unlink(missing_ok=True)
    env = dict(os.environ)
    env.update(env_extra)
    env["CUDA_VISIBLE_DEVICES"] = str(kcfg()["device"])
    cmd = [sys.executable, "-I", str(HERE / "kernel_worker.py"), "--tree", tree, "--node", node, "--out", str(p["out"]),
           "--records", str(p["records"]), "--manifest", str(kcfg()["manifest"])]
    limit = worker_limit(node)
    t0 = time.monotonic()
    with p["log"].open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)

        def stop_worker(signum, frame):  # noqa: ARG001 - the node's wall (test.sh's timeout) stops the worker too
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise SystemExit(128 + signum)

        previous = signal.signal(signal.SIGTERM, stop_worker)
        try:
            rc = proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc = "timeout"
        finally:
            signal.signal(signal.SIGTERM, previous)
    return {**p, "rc": rc, "elapsed_s": round(time.monotonic() - t0, 3), "limit_s": limit}


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")


def judge(arm: str, node: str, w: dict) -> dict:
    if w["rc"] == "timeout":
        recs = kv.read_records(w["records"])
        last = next((r for r in reversed(recs) if "case" in r or "family" in r), {})
        return {"status": "fail", "problems": [f"the node's operation exceeded its {w['limit_s']} s limit and was "
                                               f"stopped (last recorded: {last.get('case') or last.get('family')})"]}
    if w["rc"] == VERIFIER_FAULT:
        faults = [r.get("failure", "") for r in kv.read_records(w["records"]) if r.get("status") == "verifier_fault"]
        return {"status": "fault", "problems": faults or [f"the worker exited {w['rc']} without a record"]}
    if node in kc.CORRECTNESS_NODES:
        manifest = json.loads(Path(kcfg()["manifest"]).read_text())
        return kv.correctness_verdict(node, w["out"], w["records"], manifest)
    return kv.efficiency_verdict(node, w["records"], paths("reference", node)["records"], float(kcfg()["rho"]))


def write_component_map(arm: str) -> dict:
    """The arm's component map (check.component_map) from whatever efficiency records it has: the kernels each scoped
    role's public calls launched, which the served-recipe witness looks for in the served traces."""
    profiles = []
    for node, phase in kc.EFFICIENCY_NODES.items():
        for r in kv.read_records(paths(arm, node)["records"]):
            if r.get("status") == "profiled":
                profiles.append({"family": r["profile"], "phase": phase, "role": r["role"], "kernels": r["kernels"]})
    cmap = {"arm": arm, "roles": kv.chk.component_map(profiles), "profiles": len(profiles)}
    out = Path(kcfg()["records"]) / arm / "component-map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(out, cmap)
    return cmap


def node_id(node: str) -> str:
    return f"infb_verify.test_kernels.{node}"


def add_fault(nid: str, reason: str) -> None:
    faults = json.loads(FAULTS.read_text()) if FAULTS.is_file() else {}
    faults[nid] = reason[:2000]
    FAULTS.parent.mkdir(parents=True, exist_ok=True)
    FAULTS.write_text(json.dumps(faults, indent=1, sort_keys=True) + "\n")


def candidate_node(node: str) -> dict:
    w = run_worker("candidate", node, "/app", {})
    v = judge("candidate", node, w)
    v.update(node=node, arm="candidate", worker_rc=w["rc"], worker_elapsed_s=w["elapsed_s"], limit_s=w["limit_s"])
    if node in kc.EFFICIENCY_NODES:
        v["component_map_roles"] = {k: r["base_names"] for k, r in write_component_map("candidate")["roles"].items()}
    _write(w["verdict"], v)
    if v["status"] == "fault":
        add_fault(node_id(node), "; ".join(v["problems"])[:2000])
    return v


def assert_node(v: dict) -> None:
    if v["status"] == "pass":
        return
    if v["status"] == "fault":
        raise AssertionError("censored: a verifier fault, not the candidate's result: " + "; ".join(v["problems"])[:3000])
    raise AssertionError("; ".join(v["problems"][:12])[:6000])


def reference_phase() -> int:
    k = kcfg()
    sys.path.insert(0, str(TESTS))
    from infb_efficiency.reference import private_jit_caches  # noqa: PLC0415 - the reference arm's own cache root

    env: dict = {}
    jit = private_jit_caches(env, Path(config()["infb"]["efficiency"]["reference"]["root"]) / "jit-cache")
    tree = str(k["reference_tree"])
    summary = {"tree": tree, "jit_cache": jit, "nodes": {}}
    status = 0
    for node in (*kc.CORRECTNESS_NODES, *kc.EFFICIENCY_NODES):
        w = run_worker("reference", node, tree, env)
        if node in kc.CORRECTNESS_NODES:
            v = judge("reference", node, w)
        elif w["rc"] != 0:
            v = {"status": "fault", "problems": [f"the reference worker exited {w['rc']}"]}
        else:
            fams = {r["family"]: r for r in kv.read_records(w["records"]) if "family" in r and r.get("status") != "started"}
            bad = [f"{f}: {fams.get(f, {}).get('status', 'missing')}" for f in kc._timing_families(kc.EFFICIENCY_NODES[node])
                   if fams.get(f, {}).get("status") != "measured"]
            v = {"status": "pass" if not bad else "fail", "problems": bad,
                 "families": {f: {"wall_s": r.get("wall_s"), "gpu_s": r.get("gpu_s")} for f, r in fams.items()}}
        v.update(node=node, arm="reference", worker_rc=w["rc"], worker_elapsed_s=w["elapsed_s"])
        _write(w["verdict"], v)
        summary["nodes"][node] = {"status": v["status"], "elapsed_s": w["elapsed_s"], "problems": v["problems"][:20]}
        print(f"[reference_kernels] {node}: {v['status']} in {w['elapsed_s']} s" +
              (f" -- {'; '.join(v['problems'][:5])[:1500]}" if v["status"] != "pass" else ""), flush=True)
        if v["status"] != "pass":
            status = VERIFIER_FAULT
    summary["component_map"] = {k: r["base_names"] for k, r in write_component_map("reference")["roles"].items()}
    _write(Path(k["records"]) / "reference" / "summary.json", summary)
    if status:
        print("[reference_kernels] the restored reference did not pass every private case or could not be measured: "
              "reference-blocked on this host (no reward, the -1 sentinel)", flush=True)
    return status


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "reference":
        return reference_phase()
    if len(argv) == 4 and argv[1] == "fault":
        add_fault(argv[2], argv[3])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
