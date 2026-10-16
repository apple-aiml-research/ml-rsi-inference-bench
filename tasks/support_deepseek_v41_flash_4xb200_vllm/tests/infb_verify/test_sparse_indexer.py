#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The seven sparse-indexer nodes (the sparse-indexer contract).

Six run in the component group, after L0 rebuilt the candidate's tree and before the served session: the private
correctness corpus in four families (`sparse_cases.py`), then the paired component efficiency of prefill and decode
(`sparse_measure.py`), against the reference's recordings, which `infb_efficiency/reference.py` took before any candidate
code ran. The seventh, `test_sparse_indexer_serving_binding`, runs in the runtime group on the instrumented launch the
deployment node shares (`runtime_evidence.observe_sparse_indexer`).

Every node writes one record under `/logs/verifier/sparse-indexer/<node>.json`, even on an exception, and fails unless
its evidence is complete: a skipped, absent, duplicate, malformed or nonfinite required result never passes. A failure
names its class the way the reward's disposition (`disposition.py`) reads it: a candidate failure plainly, a cause that
is not the candidate's as `blocked_by:<cause>` (`reference`, `host_unhealthy`, `gpu_busy`, `verifier`).
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
S = X["sparse_indexer"]
OUT = Path(os.environ.get("INFB_SPARSE_RECORD_DIR", S["records"]))
PY = os.environ.get("INFB_ENGINE_PYTHON", sys.executable)
GPU = str(S.get("gpu", "0"))
CAPS = S["operation_caps_s"]
FAMILIES = {
    "test_two_level_candidate_semantics": "two_level",
    "test_sparse_indexer_packed_correctness": "packed",
    "test_sparse_indexer_paged_correctness": "paged",
    "test_sparse_indexer_replay_and_spec_rows": "replay_spec",
}


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@contextlib.contextmanager
def node_record(name: str):
    """The node's record, written when the node ends however it ends; an exception the node did not classify is the
    verifier's own (a `blocked_by:verifier` record), and the node fails with it."""
    rec: dict = {"node": name, "started_at": _utc(), "status": "incomplete"}
    try:
        yield rec
    except BaseException as exc:
        if not isinstance(exc, (pytest.fail.Exception, pytest.skip.Exception)):
            rec.update(status="blocked_by:verifier", why=f"verifier_fault: {type(exc).__name__}: {exc}"[:2000],
                       traceback=traceback.format_exc()[-4000:])
        raise
    finally:
        rec["finished_at"] = _utc()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")


def finish(rec: dict) -> None:
    """Fail the node unless its record passed; the message leads with the class the disposition reads."""
    if rec.get("status") != "pass":
        status = str(rec.get("status"))
        lead = status + ": " if status.startswith("blocked_by:") else ""
        pytest.fail(lead + str(rec.get("why") or status)[:1500], pytrace=False)


def gpus_busy() -> str | None:
    """None when no compute process holds any of the task's GPUs (the component gate's condition: no server and no
    timing process beside the candidate's component checks); otherwise what holds them."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"nvidia-smi could not be read: {exc}"
    if p.returncode != 0:
        return f"nvidia-smi exited {p.returncode}: {p.stderr.strip()[:300]}"
    held = [line.strip() for line in p.stdout.splitlines() if line.strip()]
    return ("GPU compute processes before the component checks: " + "; ".join(held)[:600]) if held else None


def run_bounded(argv: list[str], env: dict, bound_s: float, cwd: Path) -> dict:
    t0 = time.monotonic()
    out: dict = {"argv": argv, "bound_s": bound_s}
    try:
        p = subprocess.run(argv, env=env, cwd=str(cwd), capture_output=True, text=True, timeout=bound_s)
        out.update(rc=p.returncode, stdout=p.stdout[-3000:], stderr=p.stderr[-6000:])
    except subprocess.TimeoutExpired as exc:
        err = exc.stderr or b""
        out.update(rc=None, timed_out=True, stderr=(err.decode(errors="replace") if isinstance(err, bytes) else err)[-6000:])
    out["seconds"] = round(time.monotonic() - t0, 3)
    return out


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def judge_family(family: str, proc: dict, data: dict | None, expected: list[str]) -> dict:
    """A family's outcome from its process and its record (a pure function; the logic controls call it)."""
    if data is None:
        return {"status": "blocked_by:verifier",
                "why": f"verifier_fault: the {family} family wrote no record (rc {proc.get('rc')}): "
                       f"{(proc.get('stderr') or '')[-600:]}"}
    cases = data.get("cases") or []
    ids = [c.get("case") for c in cases]
    bad = [c for c in cases if c.get("status") == "candidate_fail"]
    if bad:
        c = bad[0]
        return {"status": "candidate_fail", "why": f"{len(bad)} case(s) failed; {c['case']}: {c.get('problems', [])[:3]}"}
    if not data.get("complete"):
        last = ids[-1] if ids else "(none)"
        if proc.get("timed_out"):
            return {"status": "candidate_fail",
                    "why": f"the {family} family's process exceeded its {proc['bound_s']:.0f} s operation limit after "
                           f"case {last}: a resource failure"}
        return {"status": "candidate_fail",
                "why": f"the {family} family's process died (rc {proc.get('rc')}) after case {last}, inside the "
                       f"implementation under test: {(proc.get('stderr') or '')[-600:]}"}
    faults = [c for c in cases if c.get("status") == "verifier_fault"]
    if faults:
        return {"status": "blocked_by:verifier", "why": f"verifier_fault in {faults[0]['case']}: {faults[0].get('problems')}"}
    if len(ids) != len(set(ids)) or sorted(ids) != sorted(expected):
        return {"status": "blocked_by:verifier",
                "why": f"verifier_fault: the {family} record holds cases {sorted(ids)}, config.json names {sorted(expected)}"}
    if any(c.get("status") != "pass" for c in cases):
        return {"status": "blocked_by:verifier", "why": "verifier_fault: a case with an unknown status"}
    return {"status": "pass", "why": ""}


def run_family(name: str) -> dict:
    family = FAMILIES[name]
    with node_record(name) as rec:
        rec["family"] = family
        busy = gpus_busy()
        if busy:
            rec.update(status="blocked_by:gpu_busy", why=busy)
            return rec
        OUT.mkdir(parents=True, exist_ok=True)
        out = OUT / f"cases-{family}.json"
        out.unlink(missing_ok=True)
        argv = [PY, "-I", str(TESTS_DIR / "infb_verify" / "sparse_cases.py"), family, "--out", str(out),
                "--device", "cuda:0"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, TESTS_DIR=str(TESTS_DIR))
        rec["process"] = run_bounded(argv, env, float(CAPS["correctness_family"]), OUT)
        rec["record"] = str(out)
        rec.update(judge_family(family, rec["process"], _read(out), S["case_ids"][family]))
        print(f"INFB_SPARSE {name}: {rec['status']} " + str(rec.get("why"))[:300], flush=True)
    return rec


def run_efficiency(name: str, phase: str) -> dict:
    """The candidate's component phase on the assigned GPU, then its verdict against the reference's phase."""
    with node_record(name) as rec:
        rec["phase"] = phase
        ref_phase = OUT / f"reference-{phase}.json"
        cand_phase = OUT / f"candidate-{phase}.json"
        ref = _read(ref_phase)
        if ref is None:
            rec.update(status="blocked_by:reference", why=f"no reference component recording at {ref_phase}")
            return rec
        busy = gpus_busy()
        if busy:
            rec.update(status="blocked_by:gpu_busy", why=busy)
            return rec
        from infb_verify import sparse_measure as sm  # noqa: PLC0415

        sm.phase("candidate", phase, cand_phase, GPU, Path(ref["record"]), PY, cwd=str(OUT), expect_root="/app")
        rec.update(sm.verdict(ref_phase, cand_phase))
        cases = (rec.get("judge") or {}).get("cases") or []
        print(f"INFB_SPARSE {name}: {rec['status']} "
              + json.dumps({c["id"]: [round(c.get("ratio_wall") or 0, 4), round(c.get("ratio_gpu") or 0, 4)]
                            for c in cases}), flush=True)
    return rec


def test_two_level_candidate_semantics():
    """The source indexer's pool (the newest block forced, candidate membership, the strict excluded-block witness)
    and the consumers' own scores within it, through the four-consumer entry points."""
    finish(run_family("test_two_level_candidate_semantics"))


def test_sparse_indexer_packed_correctness():
    """Packed score math, [ks, ke) offsets with start residues 0/1/5/7, expansion, top-k and remap, the compute
    schedule's row counts, and guarded bounds."""
    finish(run_family("test_sparse_indexer_packed_correctness"))


def test_sparse_indexer_paged_correctness():
    """The fused paged cache's byte layout, page-table addressing through permuted pages with holes, per-row
    visibility and logical positions, and guarded bounds."""
    finish(run_family("test_sparse_indexer_paged_correctness"))


def test_sparse_indexer_replay_and_spec_rows():
    """Verification groups of one to six rows, the accepted-prefix continuation with a poisoned tail, same-step reuse
    and next-step refresh of metadata, and CUDA-graph replay with changed inputs."""
    finish(run_family("test_sparse_indexer_replay_and_spec_rows"))


def test_sparse_indexer_prefill_efficiency():
    """Every packed workload (P1-P4) passes the paired complete-pipeline protocol: both ratios at least rho and the
    measured outputs correct."""
    finish(run_efficiency("test_sparse_indexer_prefill_efficiency", "prefill"))


def test_sparse_indexer_decode_efficiency():
    """Every paged workload (D1-D8) passes the paired complete-pipeline protocol."""
    finish(run_efficiency("test_sparse_indexer_decode_efficiency", "decode"))


def test_sparse_indexer_serving_binding(instrumented_server):
    """The production dataflow on the instrumented launch: the layer-20 source's pool and the four consumers'
    own selections under the comparison rules, the positions the native FlashMLA call consumes on every rank after the
    native coordinate transform, the attention output's use, graph replay, and the consumers' BF16 weights rounded
    once (weight rounding)."""
    with node_record("test_sparse_indexer_serving_binding") as rec:
        rec.update(instrumented_server.observe_sparse_indexer())
    finish(rec)
