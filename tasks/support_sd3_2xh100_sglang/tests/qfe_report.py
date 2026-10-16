#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"sd3-qfe-v1: the Q / F / E report of one SD3 trial, from the verifier's own records. Stdlib only, CPU only."
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

SCHEMA = "sd3-qfe-report/1"
VERSION = "sd3-qfe-v1"
SIDECAR = "sd3_qfe_report.json"
PROFILES = ("bf16", "fp32", "mixed")
Q_GATES = {p: f"image_similarity_full_{p}" for p in PROFILES}
FIXTURES_PER_PROFILE = 10
METRICS = ("psnr", "ssim", "clip_img", "clip_txt")
F_GATES = ("tensor_parallel_dit", "sequence_parallel_dit", "sp2_x_cfg_x_nondivisible", "dynamic_batching_is_real",
           "dynamic_batching_is_real_tp2", "batching_x_multioutput_isolation", "batching_x_multioutput_isolation_tp2")


QUALITY_COUPLED = {
    "cfg_parallel_is_real": "independent_p7 / p1 / s7 / s1 reference clauses (bf16)",
    "dit_layerwise_offload_is_real": "independent_a1 / a2 / b1 reference clauses (bf16)",
    "attention_backend_selection": "torch_sdpa engine image against the fixture's bars",
}
STATES = ("pass", "fail", "not_evaluated")
MANIFEST_FILES = ("qfe_report.py", "qfe_typed.py", "config.json", "image_thresholds.json")


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def finite(x) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def load_records(gates_dir: Path) -> tuple[dict, list]:
    """gate id -> record; a file whose `gate` differs from its name, an unreadable file, or two files claiming one id
    is an integrity problem: that id gets no record (fails closed)."""
    records, problems, seen = {}, [], {}
    for p in sorted(gates_dir.glob("*.gate.json")) if gates_dir.is_dir() else []:
        name = p.name[: -len(".gate.json")]
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"{p.name}: unreadable ({type(exc).__name__})")
            continue
        gid = rec.get("gate") if isinstance(rec, dict) else None
        if gid != name:
            problems.append(f"{p.name}: gate field {gid!r} differs from the file name")
            continue
        seen.setdefault(gid, []).append(rec)
    for gid, recs in seen.items():
        if len(recs) == 1:
            records[gid] = recs[0]
        else:
            problems.append(f"{gid}: {len(recs)} conflicting records")
    return records, problems


def verdict_of(rec) -> str:
    'verdict of.'
    if not isinstance(rec, dict):
        return "not_evaluated"
    passed = rec.get("passed")
    if passed is True:
        return "pass"
    if passed is False:
        details = rec.get("details") if isinstance(rec.get("details"), dict) else {}
        return "not_evaluated" if details.get("not_evaluated") is True else "fail"
    return "invalid"


def q_axis(records: dict) -> dict:
    out = {}
    for profile, gid in Q_GATES.items():
        rec = records.get(gid)
        details = rec.get("details") if isinstance(rec, dict) and isinstance(rec.get("details"), dict) else {}
        per = [x for x in details.get("per") or [] if isinstance(x, dict) and x.get("scored") is not False]
        fixtures = {}
        for item in per:
            fid = item.get("id")
            if not isinstance(fid, str) or fid in fixtures:
                continue
            ok_metrics = all(finite(item.get(m)) for m in METRICS)
            fixtures[fid] = ("pass" if item.get("match") is True and ok_metrics else
                             "fail" if item.get("match") is False and ok_metrics else "not_evaluated")
        passed = sum(1 for v in fixtures.values() if v == "pass")
        status = verdict_of(rec)
        if status == "pass" and (len(fixtures) != FIXTURES_PER_PROFILE or passed != FIXTURES_PER_PROFILE):
            status = "invalid"
        out[profile] = {"gate": gid, "status": status if rec is not None else "not_evaluated",
                        "fixtures_passed": passed, "fixtures_total": FIXTURES_PER_PROFILE,
                        "fixtures_recorded": len(fixtures), "fixtures": dict(sorted(fixtures.items()))}
    return out


def qfe_block(rec) -> dict | None:
    details = rec.get("details") if isinstance(rec, dict) and isinstance(rec.get("details"), dict) else {}
    for candidate in (details.get("qfe"), ((details.get("verdict") or {}).get("details") or {}).get("qfe")
                      if isinstance(details.get("verdict"), dict) else None):
        if isinstance(candidate, dict) and candidate.get("schema") == "sd3-qfe-typed/1":
            return candidate
    return None


def f_axis(records: dict) -> dict:
    out = {}
    for gid in F_GATES:
        rec = records.get(gid)
        legacy = verdict_of(rec) if rec is not None else "not_evaluated"
        blk = qfe_block(rec)
        if blk is None:
            out[gid] = {"status": "not_evaluated", "legacy": legacy, "reason": "no typed sd3-qfe sub-verdicts in the record",
                        "structural": {}, "numerical": None, "e": "not_evaluated", "pairs": {}}
            continue
        status = blk.get("status") if blk.get("status") in STATES else "invalid"
        structural = blk.get("structural") or {}
        problems = []
        if status == "pass" and legacy != "pass":
            problems.append("typed pass contradicts the legacy verdict")
        if legacy == "pass" and (status != "pass" or blk.get("e") != "pass"):
            problems.append("legacy pass contradicts the typed sub-verdicts")
        if any(v not in STATES for v in structural.values()):
            problems.append("malformed structural sub-verdict")
        if problems:
            status = "invalid"
        pair_counts = {}
        for pr in blk.get("pairs") or []:
            pair_counts[pr.get("status")] = pair_counts.get(pr.get("status"), 0) + 1
        out[gid] = {"status": status, "legacy": legacy, "structural": structural, "numerical": blk.get("numerical"),
                    "e": blk.get("e") if blk.get("e") in STATES else "invalid",
                    "obligations": len(blk.get("obligations") or []), "comparisons": len(blk.get("comparisons") or []),
                    "pairs": dict(sorted(pair_counts.items())), "problems": problems}
    return out


def legacy_axis(reward, config: dict, records: dict) -> dict:
    f2p, p2p = config.get("f2p_node_ids") or [], config.get("p2p_node_ids") or []
    counted = {"f2p_passed": sum(verdict_of(records.get(g)) == "pass" for g in f2p),
               "p2p_passed": sum(verdict_of(records.get(g)) == "pass" for g in p2p)}
    problems = []
    if not isinstance(reward, dict) or reward.get("reward") not in (0, 1) or type(reward.get("reward")) is bool:
        problems.append("reward.json missing, void or malformed: not a valid trial")
    else:
        for k in ("f2p_passed", "p2p_passed"):
            if reward.get(k) != counted[k]:
                problems.append(f"reward.json {k} {reward.get(k)} != {counted[k]} counted from the gate records")
        if reward.get("reward") == 1 and (counted["f2p_passed"] != len(f2p) or counted["p2p_passed"] != len(p2p)):
            problems.append("reward 1 without every scored record passing")
    valid = not problems
    return {"reward": reward.get("reward") if isinstance(reward, dict) else None,
            "f2p_passed": reward.get("f2p_passed") if isinstance(reward, dict) else None, "f2p_total": len(f2p),
            "p2p_passed": reward.get("p2p_passed") if isinstance(reward, dict) else None, "p2p_total": len(p2p),
            "partial": reward.get("partial") if isinstance(reward, dict) else None,
            "valid_trial": valid, "problems": problems,
            "binary_ready": 1 if valid and reward.get("reward") == 1 else 0}


def report(verifier_dir: Path, tests_dir: Path) -> dict:
    records, integrity = load_records(verifier_dir / "gates")
    try:
        reward = json.loads((verifier_dir / "reward.json").read_text())
    except (OSError, ValueError):
        reward = None
    return report_from(records, integrity, reward, json.loads((tests_dir / "config.json").read_text()), tests_dir)


def report_from(records: dict, integrity: list, reward, config: dict, tests_dir: Path) -> dict:
    """The report from already-loaded records (the live tail, or a replay of recorded trials)."""
    q, f = q_axis(records), f_axis(records)
    scored = list(config.get("f2p_node_ids") or []) + list(config.get("p2p_node_ids") or [])
    other = {g: {"status": verdict_of(records.get(g)) if g in records else "not_evaluated",
                 "p2p": g in (config.get("p2p_node_ids") or []),
                 **({"quality_coupled": QUALITY_COUPLED[g]} if g in QUALITY_COUPLED else {})}
             for g in scored if g not in Q_GATES.values() and g not in F_GATES}
    e_ready = all(v.get("e") == "pass" for v in f.values())
    return {
        "schema": SCHEMA, "version": VERSION, "status": "ok",
        "threshold_sha256": sha256(tests_dir / "image_thresholds.json"),
        "manifest": {name: sha256(tests_dir / name) for name in MANIFEST_FILES},
        "Q": {"profiles_passed": sum(v["status"] == "pass" for v in q.values()), "profiles_total": len(PROFILES),
              "fixtures_passed": sum(v["fixtures_passed"] for v in q.values()),
              "fixtures_total": FIXTURES_PER_PROFILE * len(PROFILES), "profiles": q},
        "F": {"scope": "the seven TP2 / SP2 / batching feature gates only; not a count of all SD3 capabilities",
              "passed": sum(v["status"] == "pass" for v in f.values()),
              "not_evaluated": sum(v["status"] == "not_evaluated" for v in f.values()),
              "failed": sum(v["status"] in ("fail", "invalid") for v in f.values()), "total": len(F_GATES),
              "gates": f},
        "E": {"ready": e_ready, "gates": {g: v.get("e") for g, v in f.items()}},
        "other": {"total": len(other), "passed": sum(v["status"] == "pass" for v in other.values()), "conditions": other},
        "legacy": legacy_axis(reward, config, records),
        "integrity_problems": integrity,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="sd3-qfe-v1 Q/F/E report (CPU, stdlib)")
    ap.add_argument("--verifier-dir", default=os.environ.get("VERIFIER_DIR") or os.environ.get("INFB_VERIFIER_DIR")
                    or "/logs/verifier")
    ap.add_argument("--tests-dir", default=os.environ.get("TESTS_DIR") or str(Path(__file__).resolve().parent))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    verifier_dir, tests_dir = Path(args.verifier_dir), Path(args.tests_dir)
    out = Path(args.out) if args.out else verifier_dir / SIDECAR
    try:
        doc = report(verifier_dir, tests_dir)
    except Exception as exc:  # noqa: BLE001 -- visible, never a passing score
        doc = {"schema": SCHEMA, "version": VERSION, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}
        print(f"[sd3-qfe] aggregation error: {doc['error']}", file=sys.stderr)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(f"[sd3-qfe] {doc['status']}: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
