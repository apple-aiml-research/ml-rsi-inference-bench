#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The MSA group: four correctness nodes over the private cases, two component
efficiency nodes against the reference phase's Triton-arm timing, and the sparsity node.

Runs on GPU 0 with no server resident. Each node's units -- cases, timing families, sparsity
pairs -- run in a worker process (`msa_worker.py`) that imports the candidate's `/app/minimax_m3_msa.py`; the parent
holds every unit to the 1,800-second component-invocation limit (a unit over it, or a crash inside it, is that unit's
failure) and resumes the worker after it, so every case has a record. The correctness nodes also collect the
component identity sets the execution witness reads (`/logs/verifier/msa/identity.json`).
"""

from __future__ import annotations

import functools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from infb_verify import deadline_dag, records, serve
from infb_verify.records import Blocked, Recorder, write_json

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
CONTRACT = APP_DIR / "minimax_m3_msa.py"
WORKER = Path(__file__).resolve().parent / "msa_worker.py"
LOGS = Path(os.environ.get("INFB_MSA_DIR", "/logs/verifier/msa"))
POLICY = TESTS_DIR / "hidden" / "policy.json"
PY = sys.executable
RHO = 0.8
NODE_FAMILIES = {
    "test_select_prefill_correctness": ("select_prefill",),
    "test_select_decode_correctness": ("select_decode",),
    "test_attend_prefill_correctness": ("attend_prefill_bf16", "attend_prefill_fp8"),
    "test_attend_decode_correctness": ("attend_decode_bf16", "attend_decode_fp8"),
}
EFFICIENCY = {"test_prefill_efficiency": ("select_prefill", "attend_prefill_bf16", "attend_prefill_fp8"),
              "test_decode_efficiency": ("select_decode", "attend_decode_bf16", "attend_decode_fp8")}


@functools.cache
def recorder() -> Recorder:
    return Recorder("msa")


def cap_s() -> int:
    return deadline_dag.limits()["component_invocation_s"]


def gpu0() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return visible.split(",")[0].strip() or "0"


def run_units(mode: str, units: list[str], *, contract: Path = CONTRACT, node: str | None = None, arm: str = "candidate",
              tag: str, python: str | None = None, env: dict | None = None, isolated: bool = True) -> list[dict]:
    """Run `units` in worker processes, one unit at a time; returns one row per unit: the worker's record, or the
    parent's record of a unit that exceeded its cap, crashed its worker or never ran.

    The candidate arm runs `python -I` (nothing but the trusted copies and the installed engine on the path); the
    reference arm (the reference phase) passes its own environment, whose PYTHONPATH puts the isolation hook and the
    reference tree first, and `isolated=False` so that path is honoured."""
    out = LOGS / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ if env is None else env, CUDA_VISIBLE_DEVICES=gpu0(), **serve.arm_env(arm))
    py = python or PY
    rows: dict[int, dict] = {}
    start = 0
    while start < len(units):
        offset = out.stat().st_size if out.exists() else 0
        cmd = [py, *(["-I"] if isolated else []), str(WORKER), "--contract", str(contract), "--mode", mode,
               "--start", str(start),
               "--out", str(out), "--seed-file", str(POLICY)]
        cmd += ["--node", node] if mode == "cases" else ["--units", ",".join(units)]
        with open(LOGS / f"{tag}.worker.log", "a") as log:
            proc = subprocess.Popen(cmd, env=env, cwd="/tmp", start_new_session=True, stdout=log,
                                    stderr=subprocess.STDOUT)
        state = {"current": None, "began": time.monotonic(), "import_error": None, "ended": False, "seen": 0,
                 "ready": False}

        def consume() -> None:
            events = _events(out, offset)
            for ev in events[state["seen"]:]:
                kind = ev.get("event")
                if kind == "start":
                    state["current"], state["began"] = ev["index"], time.monotonic()
                elif kind == "done":
                    rows[ev["index"]] = {"unit": ev["unit"], "seconds": ev["seconds"], **ev["record"]}
                    state["current"] = None
                elif kind == "import_error":
                    state["import_error"] = ev["error"]
                elif kind == "end":
                    state["ended"] = True
                elif kind == "ready":
                    state["ready"] = True
            state["seen"] = len(events)

        while True:
            consume()
            if state["current"] is not None and time.monotonic() - state["began"] > cap_s():
                _kill(proc)
                i = state["current"]
                rows[i] = {"unit": units[i], "status": "candidate_fail", "pass": False,
                           "resource": f"exceeded the {cap_s()} s component-invocation limit"}
                start = i + 1
                break
            if proc.poll() is not None:
                consume()
                if not state["ready"]:  # the verifier's own setup failed before any code under test ran
                    raise records.VerifierFault(f"the MSA worker exited (rc {proc.returncode}) before it was ready; "
                                                f"see {LOGS / (tag + '.worker.log')}")
                if state["import_error"] is not None:
                    for i in range(start, len(units)):
                        rows.setdefault(i, {"unit": units[i], "status": "candidate_fail", "pass": False,
                                            "exception": "/app/minimax_m3_msa.py does not import: "
                                                         + state["import_error"]})
                    start = len(units)
                elif state["current"] is not None:
                    i = state["current"]
                    rows[i] = {"unit": units[i], "status": "candidate_fail", "pass": False,
                               "exception": f"the worker died (rc {proc.returncode}) inside this unit"}
                    start = i + 1
                elif state["ended"]:
                    start = len(units)
                elif start in rows:  # died between units: resume after the last recorded one
                    start = max(rows) + 1
                else:  # died before its first unit began: that unit has no record of its own
                    rows[start] = {"unit": units[start], "status": "candidate_fail", "pass": False,
                                   "exception": f"the worker died (rc {proc.returncode}) before this unit began"}
                    start += 1
                break
            time.sleep(1.0)
    return [rows.get(i, {"unit": units[i], "status": "verifier_fault", "pass": False, "error": "no record"})
            for i in range(len(units))]


def _events(path: Path, offset: int) -> list[dict]:
    """Every event of the worker log after `offset` bytes (the current worker's own lines)."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", errors="replace") as fh:
        fh.seek(offset)
        out = []
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=60)


def correctness(node: str) -> None:
    from infb_verify import msa_cases  # noqa: PLC0415

    ids = msa_cases.case_ids()[node]
    with recorder().attempt(node, cases=len(ids)) as a:
        rows = run_units("cases", ids, node=node, tag=node)
        failed = [f"{r['unit']}: {r.get('structure') or r.get('exception') or r.get('resource') or r.get('policy')}"
                  for r in rows if not r.get("pass")]
        faults = [r["unit"] for r in rows if r.get("status") == "verifier_fault"]
        identity = run_units("identity", list(NODE_FAMILIES[node]), tag=f"{node}.identity")
        merge_identity(dict(zip(NODE_FAMILIES[node], identity)))
        write_json(LOGS / f"{node}.json", {"cases": rows, "identity": identity})
        a.set(cases=len(rows), failed=[f[:300] for f in failed], verifier_faults=faults)
        if faults and not failed:
            raise records.VerifierFault(f"cases without a judgeable record: {faults}")
        assert not failed, f"{len(failed)} of {len(rows)} cases failed: " + "; ".join(f[:500] for f in failed[:8])


def merge_identity(new: dict) -> None:
    path = LOGS / "identity.json"
    cur = json.loads(path.read_text()) if path.exists() else {"families": {}}
    cur["families"].update(new)
    cur["identities"] = records.identities()
    cur["build_products"] = build_products()
    write_json(path, cur)


@functools.cache
def build_products() -> dict:
    return {str(p.relative_to(APP_DIR)): records.sha256_file(p) for p in sorted((APP_DIR / "vllm").rglob("*.so"))}


def reference_timing() -> dict:
    path = LOGS / "reference-timing.json"
    if not path.exists():
        raise Blocked("the reference phase's Triton-arm component timing (/logs/verifier/msa/reference-timing.json)")
    return json.loads(path.read_text())


def efficiency(node: str) -> None:
    ref = reference_timing()
    families = list(EFFICIENCY[node])
    with recorder().attempt(node, families=families) as a:
        rows = run_units("measure", families, tag=node)
        verdicts, failed = {}, []
        for fam, cand in zip(families, rows):
            r = ref["families"][fam]
            if cand.get("status") != "measured":
                failed.append(f"{fam}: {cand.get('status')} {cand.get('structure') or cand.get('resource') or ''}")
                continue
            gpu, wall = r["gpu_ms_median"] / cand["gpu_ms_median"], r["wall_ms_median"] / cand["wall_ms_median"]
            verdicts[fam] = {"gpu_ratio": gpu, "wall_ratio": wall, "reference": r, "candidate": cand,
                             "pass": gpu >= RHO and wall >= RHO}
            if not verdicts[fam]["pass"]:
                failed.append(f"{fam}: GPU ratio {gpu:.4f}, wall ratio {wall:.4f} (need both >= {RHO})")
        write_json(LOGS / f"{node}.json", {"families": verdicts, "rows": rows})
        a.set(verdicts={f: {k: v[k] for k in ("gpu_ratio", "wall_ratio", "pass")} for f, v in verdicts.items()},
              failed=failed)
        assert not failed, "component efficiency: " + "; ".join(failed)


def test_select_prefill_correctness():
    correctness("test_select_prefill_correctness")


def test_select_decode_correctness():
    correctness("test_select_decode_correctness")


def test_attend_prefill_correctness():
    correctness("test_attend_prefill_correctness")


def test_attend_decode_correctness():
    correctness("test_attend_decode_correctness")


def test_prefill_efficiency():
    efficiency("test_prefill_efficiency")


def test_decode_efficiency():
    efficiency("test_decode_efficiency")


def test_sparsity_realized():
    """Sparsity is realized, against the frozen bounds in `tests/hidden/policy.json`: R_pre (GPU-time median,
    BF16 KV, the reference selections) within the frozen bound F_pre; R_dec within F_dec when the policy marks
    decode scored (`sparsity.decode.scored`), otherwise recorded as a diagnostic with no reward; and, when the
    policy names poisoned geometries (`sparsity.poisoned.geometries`), every one of them passed: the unselected
    pages' K and V poisoned, the call still meets its case's correctness policy (msa/sparsity.py)."""
    bounds = json.loads(POLICY.read_text())["sparsity"]
    decode_scored = bool(bounds.get("decode", {}).get("scored"))
    geometries = list(bounds.get("poisoned", {}).get("geometries", []))
    with recorder().attempt("test_sparsity_realized", bounds=bounds) as a:
        rows = dict(zip(("decode", "prefill"), run_units("sparsity", ["decode", "prefill"], tag="test_sparsity_realized")))
        poisoned = dict(zip(geometries, run_units("poison", geometries, tag="test_sparsity_realized-poison"))) \
            if geometries else {}
        write_json(LOGS / "test_sparsity_realized.json", {**rows, "poisoned": poisoned})
        failed = []
        r = rows["prefill"]
        if r.get("status") != "measured":
            failed.append(f"prefill: {r.get('status')} {r.get('structure') or r.get('resource') or ''}")
        elif not r["ratio"] <= bounds["F_pre"]:
            failed.append(f"R_pre {r['ratio']:.4f} > F_pre {bounds['F_pre']}")
        decode = rows["decode"]
        if decode_scored:
            if decode.get("status") != "measured":
                failed.append(f"decode: {decode.get('status')} {decode.get('structure') or decode.get('resource') or ''}")
            elif not decode["ratio"] <= bounds["F_dec"]:
                failed.append(f"R_dec {decode['ratio']:.4f} > F_dec {bounds['F_dec']}")
        for name, rec in poisoned.items():
            if not rec.get("pass"):
                failed.append(f"poisoned {name}: {rec.get('status')} "
                              f"{rec.get('structure') or rec.get('exception') or rec.get('resource') or ''}")
        a.set(ratios={w: rows[w].get("ratio") for w in rows},
              decode={"status": decode.get("status"), "R_dec": decode.get("ratio"), "scored": decode_scored},
              poisoned={n: {"status": p.get("status"), "pass": p.get("pass"), "poisoned_pages": p.get("poisoned_pages")}
                        for n, p in poisoned.items()},
              failed=failed)
        assert not failed, "sparsity realized: " + "; ".join(failed)
