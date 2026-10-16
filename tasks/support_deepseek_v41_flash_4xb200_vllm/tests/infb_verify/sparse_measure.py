#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The sparse indexer's component efficiency (the sparse-indexer contract): the fixed workload matrix, measured by the
public protocol (the grading copy of `sparse_indexer/measure.py`), paired reference-first on one GPU, private.

    python -I sparse_measure.py record --role reference|candidate --phase prefill|decode --out FILE --expect-root DIR
                                        [--counts REFERENCE_RECORD] [--device cuda:0]
    python -I sparse_measure.py judge --reference FILE --candidate FILE --out FILE
    python -I sparse_measure.py phase --role candidate --phase prefill|decode --out FILE --gpu 0 --counts FILE

`record` runs in its own process per role and phase, with the role's tree on the interpreter's path (the reference tree
on PYTHONPATH; the candidate's installed tree), and measures every case of the phase: ten warm work units (compile and
warm time kept separately), a repetition count by doubling to a 200 ms pilot block or 2,048 (the reference's; the
candidate reads the reference's counts), three blocks timed by wall clock around device synchronization and by CUDA
events, over four preallocated input sets per case rotated between work units. The outputs of the last work unit of
each input set are then checked against the contract (structure on every row, the selection rule on sampled rows).

`judge` computes, per case, median(reference) / median(candidate) for wall and for GPU time; a case passes when both are
at least rho and its correctness held; a phase is the conjunction of its cases. No timing is redrawn.

`phase` is one role's phase as the verifier runs it: a canary reading, the recording in its own process on the assigned
GPU, a canary reading. The candidate's recording runs with `python -I` from outside /app (its installed tree); the
reference's is started by `infb_efficiency/reference.py` with the reference server's own environment (the restored tree
on PYTHONPATH, its private JIT cache), which `-I` would discard. `verdict` reads both roles' phase records into one node
outcome: a correctness failure of the measured outputs is the candidate's on any host; a slow case or a recording over
its operation limit is the candidate's only on a host the canary read healthy, and otherwise voids the node
(`blocked_by:host_unhealthy`); a reference that did not record is `blocked_by:reference`.

Operation limits (the public README's): the warm work of a phase (every case's ten warm units, JIT compilation
included) within the rebuild cap, its measurement (pilots and blocks) within the measurement-invocation cap; the process
itself is killed at their sum plus a minute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
sys.path.insert(0, str(TESTS_DIR))

from infb_verify import _trusted  # noqa: E402

CONFIG = json.loads((TESTS_DIR / "config.json").read_text())["infb"]["sparse_indexer"]


def _modules():
    measure = _trusted.load("sparse_indexer/measure.py")
    contract = _trusted.load("sparse_indexer/contract.py")
    reference = _trusted.load("sparse_indexer/reference.py")
    return measure, contract, reference


def workloads(phase: str) -> list[dict]:
    return [dict(w) for w in CONFIG["workloads"][phase]]


def case_seed(case_id: str) -> int:
    """The private seed for one case (a separate stream per case id)."""
    h = hashlib.sha256(f"{CONFIG['seeds']['private']}:{case_id}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**62)


def structure_problems(torch, out, pools, visible) -> list[str]:
    """Vectorized structure checks of one consumer's [R, 512] output against [R, 2048] pools and [R] visible lengths:
    int32; -1 padding after the valid prefix is not required, but every valid entry is a distinct visible position of
    a pool block; the valid count is min(512, selectable)."""
    problems = []
    R = out.shape[0]
    B = 8
    valid = out >= 0
    if bool((out < -1).any()):
        problems.append("an entry below -1")
    vis = visible.to(torch.int64)
    nblocks_row = (vis + B - 1) // B
    pv = pools.to(torch.int64)
    pool_ok = (pv >= 0) & (pv < nblocks_row[:, None])
    # the selectable count per row: the visible positions of each valid pool block (8, or fewer in the newest block)
    per_block = (vis[:, None] - pv * B).clamp(0, B)
    want = torch.where(pool_ok, per_block, 0).sum(1).clamp(max=512)
    got = valid.sum(1)
    bad = (got != want).nonzero().flatten()
    if bad.numel():
        r = int(bad[0])
        problems.append(f"{bad.numel()} row(s) with a wrong valid count (row {r}: {int(got[r])}, want {int(want[r])})")
    pos = out.to(torch.int64)
    blocks = torch.where(valid, pos // B, 0)
    beyond = valid & (pos >= vis[:, None])
    if bool(beyond.any()):
        problems.append(f"{int(beyond.any(1).sum())} row(s) select a position beyond the visible length")
    # membership in the pool: a [R, max_blocks] mask
    mb = int(nblocks_row.max()) + 1 if R else 1
    member = torch.zeros((R, mb), dtype=torch.bool, device=out.device)
    rows = torch.arange(R, device=out.device)[:, None].expand_as(pv)
    member[rows[pool_ok], pv[pool_ok]] = True
    outside = valid & ~member.gather(1, blocks.clamp(max=mb - 1))
    if bool(outside.any()):
        problems.append(f"{int(outside.any(1).sum())} row(s) select a position outside the candidate pool")
    srt = torch.sort(torch.where(valid, pos, torch.iinfo(torch.int64).max), dim=1).values
    dup = (srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] != torch.iinfo(torch.int64).max)
    if bool(dup.any()):
        problems.append(f"{int(dup.any(1).sum())} row(s) select a position twice")
    return problems


def check_outputs(torch, ref, contract, kind: str, step, out, sample: int = 6) -> dict:
    """The measured outputs of one input set: structure on every row of every consumer, the selection rule on sampled
    rows."""
    problems: list[str] = []
    R = step.rows
    if kind == "packed":
        visible = (step.k_end - step.k_start).to(torch.int64)
    else:
        visible = step.context_lens.to(torch.int64)
    for c in range(contract.NUM_CONSUMERS):
        problems += [f"consumer {c}: {p}" for p in structure_problems(torch, out[c], step.candidate_blocks, visible)]
    rows = sorted({0, R - 1, *range(0, R, max(1, R // sample))})[: sample + 2]
    if kind == "packed":
        kd = ref.dequantize(step.k_quant, step.k_scale)
    for r in rows:
        v = int(visible[r])
        if kind == "packed":
            keys = kd[int(step.k_start[r]) : int(step.k_end[r])]
        else:
            values, words = ref.page_tokens(step.kv_cache, step.block_table[r], v)
            keys = ref.dequantize(values, words)
        allowed = ref.candidate_positions(step.candidate_blocks[r], v)
        for c in range(contract.NUM_CONSUMERS):
            if kind == "packed":
                qd = ref.dequantize(step.q[c, r : r + 1], step.q_scale[c, r : r + 1])
            else:
                qd = ref.dequantize(step.q[c, r : r + 1, 0], step.q_scale[c, r : r + 1, 0])
            s = ref.scores(qd, step.weights[c, r : r + 1], keys)[0]
            delta = ref.SOURCE_DELTA * float(s.abs().max()) if s.numel() else 0.0
            if not ref.within_selection_slack(out[c, r], allowed, s, delta):
                problems.append(f"consumer {c} row {r}: an excluded selectable position scores more than 2 delta above "
                                f"the smallest selected one")
    return {"rows_checked_structure": R * contract.NUM_CONSUMERS, "rows_checked_scores": len(rows) * contract.NUM_CONSUMERS,
            "problems": problems[:20], "n_problems": len(problems)}


def _inside(path: str | None, root: str) -> bool:
    if not path:
        return False
    rp, rr = os.path.realpath(path), os.path.realpath(root)
    return rp == rr or rp.startswith(rr.rstrip("/") + "/")


def record(role: str, phase: str, out_path: Path, counts: dict | None, device: str, expect_root: str) -> int:
    import torch

    measure, contract, ref = _modules()
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    kind = "packed" if phase == "prefill" else "paged"
    rec = {"schema": "sparse-indexer-component/1", "role": role, "phase": phase, "device": torch.cuda.get_device_name(device),
           "mode": CONFIG["measurement_mode"], "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "python": sys.executable, "sys_path0": sys.path[:3], "cases": []}
    try:
        import vllm  # noqa: F401,PLC0415
        rec["vllm_file"] = getattr(sys.modules["vllm"], "__file__", None)
    except Exception as exc:  # noqa: BLE001
        rec["vllm_import_error"] = f"{type(exc).__name__}: {exc}"[:500]
    rec["expect_root"] = expect_root
    if "vllm_file" in rec and not _inside(rec["vllm_file"], expect_root):
        # the process imported another tree than the role's: the verifier's own environment is wrong
        rec.update(status="verifier_fault", error=f"the engine imported from {rec['vllm_file']}, not inside {expect_root}")
        out_path.write_text(json.dumps(rec, indent=1) + "\n")
        return 0
    caps = CONFIG["operation_caps_s"]
    rec["totals"] = {"warm_s": 0.0, "measurement_s": 0.0, "caps_s": {"warm": caps["warm"], "measurement": caps["measurement"]}}
    t_all = time.perf_counter()
    for case in workloads(phase):
        entry = {"id": case["id"], **{k: v for k, v in case.items() if k != "id"}}
        t0 = time.perf_counter()
        if counts is not None and case["id"] not in counts:
            entry.update(status="blocked_by:reference", error="the reference record holds no repetition count for this case")
            rec["cases"].append(entry)
            continue
        try:
            seed = case_seed(case["id"])
            sets = (measure.packed_sets if kind == "packed" else measure.paged_sets)(case, seed, device)
            bad = [p for s in sets for p in (contract.packed_problems(s) if kind == "packed" else contract.paged_problems(s))]
            if bad:
                raise RuntimeError(f"the generated inputs fall outside the interface: {sorted(set(bad))}")
            torch.cuda.reset_peak_memory_stats(device)
            unit = measure.unit_for(phase, sets, device)
            reps = None if counts is None else int(counts[case["id"]])
            t_m = time.perf_counter()
            entry.update(measure.measure(unit, torch, repetitions=reps))
            rec["totals"]["warm_s"] += entry["warm_s"]
            rec["totals"]["measurement_s"] += time.perf_counter() - t_m - entry["warm_s"]
            entry["peak_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
            # the last work unit of each input set wrote that set's output buffer
            entry["correctness"] = [check_outputs(torch, ref, contract, kind, s, o) for s, o in zip(sets, unit.outputs)]
            entry["status"] = "measured"
        except NotImplementedError as exc:
            entry["status"] = "candidate_fail" if role == "candidate" else "reference_fault"
            entry["error"] = f"NotImplementedError: {exc}"[:800]
        except Exception as exc:  # noqa: BLE001 - recorded with its cause; the judge attributes it by role
            entry["status"] = "candidate_fail" if role == "candidate" else "reference_fault"
            entry["error"] = f"{type(exc).__name__}: {exc}"[:800]
        entry["seconds"] = round(time.perf_counter() - t0, 3)
        rec["cases"].append(entry)
        out_path.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"[sparse_measure] {role} {case['id']}: {entry.get('status')} "
              f"{entry.get('median_wall_s', float('nan')) * 1e6:.1f} us wall", flush=True)
    rec["seconds"] = round(time.perf_counter() - t_all, 3)
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    over = [k for k in ("warm", "measurement") if rec["totals"][f"{k}_s"] > caps[k]]
    if over:
        rec["over_limit"] = [f"{k} {rec['totals'][f'{k}_s']:.1f} s over its {caps[k]} s limit" for k in over]
    rec["complete"] = True
    out_path.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


def _positive(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def judge(ref_rec: dict, cand_rec: dict) -> dict:
    rho = float(CONFIG["rho"])
    cand = {c["id"]: c for c in cand_rec.get("cases", [])}
    out = {"phase": ref_rec.get("phase"), "rho": rho, "cases": [], "status": "pass"}
    for r in ref_rec.get("cases", []):
        c = cand.get(r["id"])
        row = {"id": r["id"], "status": "pass"}
        if r.get("status") != "measured" or not all(_positive(r.get(k)) for k in ("median_wall_s", "median_gpu_s")):
            row.update(status="reference_fault", why=f"the reference did not measure this case: {r.get('error') or r.get('status')}")
        elif c is None:
            row.update(status="candidate_fail", why="no candidate record for this case")
        elif c.get("status") != "measured":
            row.update(status=c.get("status", "candidate_fail"), why=c.get("error") or "not measured")
        elif not all(_positive(c.get(k)) for k in ("median_wall_s", "median_gpu_s")):
            row.update(status="candidate_fail", why="missing, nonfinite or nonpositive timings")
        else:
            row["ratio_wall"] = r["median_wall_s"] / c["median_wall_s"]
            row["ratio_gpu"] = r["median_gpu_s"] / c["median_gpu_s"]
            row["repetitions"] = c.get("repetitions")
            problems = [p for chk in (c.get("correctness") or []) for p in chk.get("problems", [])]
            if c.get("correctness") is None:
                problems = ["the measured outputs were not checked"]
            if problems:
                row.update(status="candidate_fail", why=f"correctness: {problems[:3]}")
            elif row["ratio_wall"] < rho or row["ratio_gpu"] < rho:
                row.update(status="candidate_fail", why=f"ratios wall {row['ratio_wall']:.4f}, GPU {row['ratio_gpu']:.4f}")
        out["cases"].append(row)
    for row in out["cases"]:
        if row["status"] == "candidate_fail":
            out["status"] = "candidate_fail"
            break
        if row["status"] != "pass" and out["status"] == "pass":
            out["status"] = row["status"]
    return out


def canary_reading(env: dict) -> dict:
    """One reading of the task's environment canary (infb_efficiency/canary.py read) on every visible GPU."""
    canary = _canary()
    return canary.take([sys.executable, "-I", str(TESTS_DIR / "infb_efficiency" / "canary.py"), "read"], env)


def _canary():
    import importlib.util  # noqa: PLC0415

    name = "infb_sparse_canary"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TESTS_DIR / "infb_efficiency" / "canary.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def phase(role: str, phase_name: str, out: Path, gpu: str, counts: Path | None, python: str, *,
          env: dict | None = None, cwd: str | None = None, expect_root: str = "/app") -> dict:
    """One role's phase: a canary reading, the recording in its own process on the assigned GPU, a canary reading. The
    phase record (written to `out`, and returned) names the recording, the process outcome and the readings.

    `env` is the recording's environment (default: this process's); the candidate's recording runs isolated (`-I`), the
    reference's with `env` as given (the reference server's, the restored tree on PYTHONPATH)."""
    import subprocess  # noqa: PLC0415

    caps = CONFIG["operation_caps_s"]
    rec_path = out.with_suffix(".cases.json")
    base = dict(os.environ if env is None else env)
    rec = {"schema": "sparse-indexer-phase/1", "role": role, "phase": phase_name, "gpu": gpu, "record": str(rec_path),
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "canary": {"before": canary_reading(base)}}
    argv = [python, *(["-I"] if role == "candidate" else []), str(Path(__file__).resolve()), "record", "--role", role,
            "--phase", phase_name, "--out", str(rec_path), "--device", "cuda:0", "--expect-root", expect_root]
    if counts:
        argv += ["--counts", str(counts)]
    child = dict(base, CUDA_VISIBLE_DEVICES=gpu, TESTS_DIR=str(TESTS_DIR))
    kill_at = caps["warm"] + caps["measurement"] + 60
    rec["argv"], rec["kill_at_s"] = argv, kill_at
    t0 = time.monotonic()
    try:
        p = subprocess.run(argv, env=child, cwd=cwd or str(out.parent), capture_output=True, text=True, timeout=kill_at)
        rec.update(rc=p.returncode, stdout=p.stdout[-4000:], stderr=p.stderr[-4000:])
    except subprocess.TimeoutExpired as exc:
        err = exc.stderr or b""
        rec.update(rc=None, timed_out=True, stderr=(err.decode(errors="replace") if isinstance(err, bytes) else err)[-4000:])
    rec["seconds"] = round(time.monotonic() - t0, 3)
    rec["canary"]["after"] = canary_reading(base)
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[sparse_measure] {role} {phase_name}: rc {rec.get('rc')} after {rec['seconds']} s", flush=True)
    return rec


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def verdict(ref_phase_path: Path, cand_phase_path: Path) -> dict:
    """One component-efficiency node's outcome from both roles' phase records: {"status", "why", "judge", "canary"}.
    `status` is `pass`, `candidate_fail`, or `blocked_by:<cause>`."""
    out: dict = {"reference_phase": str(ref_phase_path), "candidate_phase": str(cand_phase_path)}
    rp, cp = _read(ref_phase_path), _read(cand_phase_path)
    rr = _read(Path(rp["record"])) if rp else None
    if rp is None or rr is None or not rr.get("complete") or rp.get("rc") != 0:
        why = ("no reference phase record" if rp is None else
               f"the reference recording did not complete (rc {rp.get('rc')}): {(rp.get('stderr') or '')[-300:]}")
        return {**out, "status": "blocked_by:reference", "why": why}
    if rr.get("status") == "verifier_fault" or rr.get("over_limit"):
        return {**out, "status": "blocked_by:reference", "why": rr.get("error") or f"the reference ran {rr['over_limit']}"}
    if cp is None:
        return {**out, "status": "blocked_by:verifier", "why": "no candidate phase record"}
    cr = _read(Path(cp["record"])) or {}
    if cr.get("status") == "verifier_fault":
        return {**out, "status": "blocked_by:verifier", "why": cr.get("error")}
    canary = canary_verdict([rp, cp])
    out["canary"] = canary
    healthy = not canary["unhealthy"]
    j = judge(rr, cr)
    out["judge"] = j
    wrong = [c for c in j["cases"] if c["status"] == "candidate_fail" and str(c.get("why", "")).startswith("correctness")]
    if wrong:
        return {**out, "status": "candidate_fail", "why": f"{wrong[0]['id']}: {wrong[0]['why']}"}
    resource = []
    if cp.get("timed_out"):
        resource.append(f"the candidate's recording was killed at {cp['kill_at_s']} s")
    elif cp.get("rc") not in (0, None) or not cr.get("complete"):
        done = [c["id"] for c in cr.get("cases", [])]
        return {**out, "status": "candidate_fail",
                "why": f"the candidate's recording died (rc {cp.get('rc')}) after {done}: {(cp.get('stderr') or '')[-400:]}"}
    errors = [c for c in j["cases"] if c["status"] == "candidate_fail" and not str(c.get("why", "")).startswith("ratios")]
    if errors:  # an exception, a missing case or a nonfinite timing: the candidate's on any host
        return {**out, "status": "candidate_fail", "why": f"{errors[0]['id']}: {errors[0]['why']}"}
    resource += cr.get("over_limit") or []
    slow = [c for c in j["cases"] if c["status"] == "candidate_fail"]
    if (resource or slow) and not healthy:
        return {**out, "status": "blocked_by:host_unhealthy",
                "why": f"the canary read the host unhealthy ({canary['unhealthy']}) while "
                       + "; ".join(resource + [f"{c['id']}: {c['why']}" for c in slow])[:600]}
    if resource:
        return {**out, "status": "candidate_fail", "why": "over its operation limit: " + "; ".join(resource)}
    if j["status"] != "pass":
        first = next(c for c in j["cases"] if c["status"] != "pass")
        status = "candidate_fail" if j["status"] == "candidate_fail" else "blocked_by:reference"
        return {**out, "status": status, "why": f"{first['id']}: {first.get('why')}"}
    return {**out, "status": "pass", "why": ""}


def canary_verdict(phases: list[dict]) -> dict:
    """The component phases' canary readings judged as the task's canary judges them: each reading
    against the running upper median of the readings so far, the GPUs at 0.1 and the CPU at 0.25. An invalid
    reading marks the host unhealthy for the component gate: its nodes are voided, never redrawn."""
    canary = _canary()
    tol, cpu_tol = CONFIG["canary"]["tolerance"], CONFIG["canary"]["cpu_tolerance"]
    readings = [(f"{p['role']}.{k}", p["canary"][k]) for p in phases for k in ("before", "after")]
    seen, out = [], []
    for name, r in readings:
        seen.append(r)
        cmp_ = canary.compare(canary.baseline(seen), r, tol, cpu_tol)
        out.append({"reading": name, **cmp_})
    return {"judged": out, "unhealthy": [o["reading"] for o in out if o.get("valid") is False]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--role", choices=("reference", "candidate"), required=True)
    r.add_argument("--phase", choices=("prefill", "decode"), required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--counts", type=Path)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--expect-root", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--reference", type=Path, required=True)
    j.add_argument("--candidate", type=Path, required=True)
    j.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("phase")
    f.add_argument("--role", choices=("reference", "candidate"), required=True)
    f.add_argument("--phase", choices=("prefill", "decode"), required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--gpu", default="0")
    f.add_argument("--counts", type=Path)
    f.add_argument("--python", default=sys.executable)
    f.add_argument("--expect-root", default="/app")
    a = ap.parse_args(argv)
    if a.cmd == "phase":
        a.out.parent.mkdir(parents=True, exist_ok=True)
        phase(a.role, a.phase, a.out, a.gpu, a.counts, a.python, expect_root=a.expect_root)
        return 0
    if a.cmd == "record":
        counts = None
        if a.counts:
            counts = {c["id"]: c["repetitions"] for c in json.loads(a.counts.read_text())["cases"]
                      if c.get("status") == "measured"}
        return record(a.role, a.phase, a.out, counts, a.device, a.expect_root)
    verdict = judge(json.loads(a.reference.read_text()), json.loads(a.candidate.read_text()))
    a.out.write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps({"phase": verdict["phase"], "status": verdict["status"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
