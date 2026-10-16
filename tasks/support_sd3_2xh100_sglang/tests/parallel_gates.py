#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 parallel gates.'
from __future__ import annotations

import math
from batching_gates import psnr
import qfe_typed
import reference_contract

POLICY = {"MAX_COMPUTE_FRACTION": 0.75, "DROP_MAX_DB": 20.0,
          "REPEAT_PSNR_DB": 45.0}
POLICY_STATE = "specified"
POLICY_BASIS = {'compute_drop': 'Required bound for compute_drop.', 'images': 'Required bound for images.', 'reference': 'Required bound for reference.', 'repeat': 'Required bound for repeat.'}


def rank_of(e):
    r = e.get("rank")
    return r.get("dist_rank") if isinstance(r, dict) else r


def window(run, key, event="dit_call", rank=None):
    q = (run.get("readings") or {}).get(key) or {}
    if not q.get("t0") or not q.get("t1"):
        return []
    return [e for e in run.get("events", []) if e.get("event") == event
            and (rank is None or rank_of(e) == rank)
            and q["t0"] <= float(e.get("t", 0)) <= q["t1"]]


def census(run):


    pid_ranks = {}
    for e in run.get("events", []):
        if e.get("pid") is not None and rank_of(e) is not None:
            pid_ranks.setdefault(e["pid"], set()).add(rank_of(e))
    ranks = {}
    for e in run.get("events", []):
        rank = rank_of(e)
        if rank is None and e.get("event") == "served":
            options = pid_ranks.get(e.get("pid"), set())
            rank = next(iter(options)) if len(options) == 1 else None
        if rank is None:
            continue
        out = ranks.setdefault(rank, {"compute": [], "params": [], "served": []})
        if e.get("event") == "compute_census":
            out["compute"].append(e)
        elif e.get("event") == "param_census" and e.get("component") == "transformer":
            out["params"].append(e)
        elif e.get("event") == "served":
            out["served"].append(e)
    return ranks


def result(clauses, details):
    failed = [k for k, v in clauses.items() if v == "fail"]
    missing = [k for k, v in clauses.items() if v == "not_evaluated"]
    return {"verdict": "fail" if failed else "not_evaluated" if missing else "pass",
            "reasons": [f"{k}: {clauses[k]}" for k in failed + missing],
            "details": {"clauses": clauses, **details}}


def comparable_inputs(dense_shape, rank_shapes, mode):
    """Accept an SP spatial split before DiT as well as one inside DiT.

    This only establishes comparable work, not proof of sharding. Executed
    MACs, full SP parameter storage, communication, consumed rank output and
    independent images remain separate required clauses.
    """
    shapes = [dense_shape, *rank_shapes]
    if len(rank_shapes) != 2 or any(not isinstance(s, list) or not s or
            any(type(n) is not int or n <= 0 for n in s) for s in shapes):
        return False
    if all(s == dense_shape for s in rank_shapes):
        return True
    if mode != 'sp2' or any(len(s) != 4 for s in shapes):
        return False
    for axis in (2, 3):
        if not all(all(s[i] == dense_shape[i] for i in range(4) if i != axis)
                   for s in rank_shapes):
            continue
        full = dense_shape[axis]
        local = [s[axis] for s in rank_shapes]


        if (all(n in (full // 2, (full + 1) // 2) for n in local)
                and sum(local) in (full, full + full % 2)):
            return True
    return False


def topology(dense, parallel, mode, policy=POLICY):
    dr, pr = census(dense), census(parallel)
    problems, missing, fractions = [], [], {}
    dc = (dr.get(0) or {}).get("compute", [])
    dp = (dr.get(0) or {}).get("params", [])
    if len(dc) != 1 or len(dp) != 1:
        return "not_evaluated", {"missing": ["dense compute/parameter census"]}
    d, params = dc[0], dp[0]
    if not d.get("macs") or d.get("truncated") or not params.get("shapes"):
        return "not_evaluated", {"missing": ["incomplete dense census"]}
    if set(pr) != {0, 1}:
        missing.append("both worker ranks were not recorded")
    rank_shapes = [(pr[r]['compute'][0].get('x') if r in pr and len(pr[r]['compute']) == 1 else None)
                   for r in (0, 1)]
    comparable = comparable_inputs(d.get('x'), rank_shapes, mode)
    for rank in (0, 1):
        r = pr.get(rank) or {}
        if not r.get("served") or len(r.get("compute", [])) != 1 or len(r.get("params", [])) != 1:
            missing.append(f"rank {rank}: incomplete served/compute/parameter record")
            continue
        c, p = r["compute"][0], r["params"][0]
        macs = c.get("macs")
        if not isinstance(macs, (int, float)) or not math.isfinite(macs) or macs <= 0 or c.get("truncated"):
            missing.append(f"rank {rank}: incomplete executed work")
            continue
        if not comparable:
            missing.append(f"rank {rank}: incomparable boundary input shape")
            continue
        fractions[str(rank)] = macs / d["macs"]
        if fractions[str(rank)] > policy["MAX_COMPUTE_FRACTION"]:
            problems.append(f"rank {rank}: replicated rather than sharded compute")


        def elements(c):
            if c.get("truncated") or not c.get("shapes"):
                return None
            try:
                values = [math.prod(shape) for shape in c["shapes"].values()]
                return sum(values) if all(v > 0 for v in values) else None
            except (TypeError, ValueError):
                return None
        dn, pn = elements(params), elements(p)
        if not dn or not pn:
            missing.append(f"rank {rank}: incomplete parameter inventory")
        elif mode == "tp2" and not 0 < pn < dn:
            problems.append(f"rank {rank}: tensor parameters were not sharded")
        elif mode == "sp2" and pn != dn:
            problems.append(f"rank {rank}: SP parameter volume differs from the complete model")
    return ("fail" if problems else "not_evaluated" if missing else "pass"), {
        "compute_fractions": fractions, "missing": missing, "problems": problems}


def parallel_is_real(reading, request_ids, policy=POLICY):
    mode, specs = reading["mode"], reading["requests"]
    dense, parallel, dropped = (reading.get(k) or {} for k in ("dense", "parallel", "drop"))
    clauses, detail = {}, {}
    clauses["sharded"], detail["topology"] = topology(dense, parallel, mode, policy)
    clauses["recorded"] = "pass"
    clauses["collectives"] = "pass"
    clauses["images"] = "pass"
    clauses["reference"] = "pass"
    detail["images"] = {}
    run_ids = {"dense": dense.get("verifier_run_id"), "parallel": parallel.get("verifier_run_id")}
    refs = reading.get("reference") or {}
    for key in request_ids:
        spec = specs[key]
        expected = int(spec["num_inference_steps"]) * (2 if spec["guidance_scale"] > 1 else 1)
        for rank in (0, 1):
            calls = window(parallel, key, rank=rank)
            boundary = window(parallel, key, event="batch_dit", rank=rank)
            if not calls:
                clauses["recorded"] = "not_evaluated"
            elif boundary:


                from collections import Counter
                n = int(spec.get("num_outputs_per_prompt", 1))
                branches = 2 if spec["guidance_scale"] > 1 else 1
                work = Counter()
                valid = len(boundary) == len(calls)
                for event in boundary:
                    ts, rows = event.get("timesteps") or [], event.get("rows", 0)
                    if (event.get("completed") is not True or rows not in {n, n * branches} or
                            len(ts) != rows or len(set(ts)) != 1 or
                            any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in ts)):
                        valid = False
                    else:
                        work[ts[0]] += rows
                if not valid or len(work) != int(spec["num_inference_steps"]) or any(v != n * branches for v in work.values()):
                    clauses["recorded"] = "fail"
            elif len(calls) != expected:
                clauses["recorded"] = "fail"
            if any(not isinstance(c.get("coll"), dict) for c in calls):
                clauses["collectives"] = "not_evaluated"
            elif any(not any(n > 0 for op, n in c["coll"].items() if op != "broadcast") for c in calls):
                clauses["collectives"] = "fail"
        a, b, ref = ((parallel.get("images") or {}).get(key), (dense.get("images") or {}).get(key), refs.get(key))
        same, ar, br = psnr(a, b), psnr(a, ref), psnr(b, ref)
        aq = reference_contract.image_check(a, ref, spec)
        bq = reference_contract.image_check(b, ref, spec)
        detail["images"][key] = {"parallel_vs_dense": same, "parallel_vs_reference": aq,
                                "dense_vs_reference": bq}
        for check in (aq,bq):
            if check["verdict"] == "fail":
                clauses["images"] = clauses["reference"] = "fail"
            elif check["verdict"] == "not_evaluated":
                for name in ("images","reference"):
                    if clauses[name] != "fail":
                        clauses[name] = "not_evaluated"


    armed = any((e.get("event") == "rank_drop_active" or (e.get("event") == "patched" and e.get("target") == "parallel_evidence collectives"))
                and rank_of(e) == 1 and e.get("drop_here") is True for e in dropped.get("events", []))
    drop_key = reading["drop_key"]
    value = psnr((dropped.get("images") or {}).get(drop_key), (parallel.get("images") or {}).get(drop_key))
    clauses["consumed"] = "not_evaluated" if not armed else (
        "pass" if value is not None and value < policy["DROP_MAX_DB"] else "fail")
    detail["drop"] = {"rank1_armed": armed, "psnr": value}
    if not parallel.get("built"):
        clauses["engine"] = "fail"
    out = result(clauses, detail)
    out["details"]["qfe"] = qfe_from_detail(clauses, detail, specs, request_ids, run_ids, out["verdict"])
    return out


def qfe_from_detail(clauses, detail, specs, request_ids, run_ids, verdict):
    """sd3-qfe-v1 block of one parallel gate, from exactly what the gate stored (`clauses`, `detail["images"]`,
    `detail["drop"]`), so the live gate and a replay of a recorded gate build the same block.

    structural = the causal clauses only; images/reference are E: the parallel output and its same-run dense anchor,
    each against the independent reference (the rank-drop image is a causal control, never an E check). Paired drift
    (parallel vs dense) is a readout, never credit."""
    comparisons, obligations, pairs, readouts = [], [], [], {}
    for key in request_ids:
        reading = (detail.get("images") or {}).get(key) or {}
        aq, bq = reading.get("parallel_vs_reference") or {}, reading.get("dense_vs_reference") or {}
        deployment, anchor = qfe_typed.row(aq, "parallel", key, 0), qfe_typed.row(bq, "dense_anchor", key, 0)
        comparisons += [deployment, anchor]
        obligations += [qfe_typed.obligation("parallel", key, ["image"], "checked", deployment["verdict"], 1),
                        qfe_typed.obligation("dense_anchor", key, ["image"], "checked", anchor["verdict"], 1)]
        pairs.append(qfe_typed.pair(deployment, anchor, reading.get("parallel_vs_dense"),
                                    qfe_typed.binding((specs or {}).get(key), 0, aq.get("profile") or bq.get("profile"),
                                                      run_ids)))
        readouts[f"{key}.parallel_vs_dense_psnr_db"] = reading.get("parallel_vs_dense")
    readouts["drop_psnr_db"] = (detail.get("drop") or {}).get("psnr")
    structural = {k: clauses[k] for k in ("sharded", "recorded", "collectives", "consumed", "engine") if k in clauses}
    ids = run_ids or {}
    if ids.get("dense") != ids.get("parallel"):
        structural["same_run_anchor"] = "fail" if ids.get("dense") and ids.get("parallel") else "not_evaluated"
    return qfe_typed.block(structural, comparisons, obligations, pairs, verdict, readouts)


def cancellation_stop(run, request):
    """Actual start work per rank, rather than a cancellation return message."""
    from t2_gates import POLICY as CANCEL_POLICY
    q = (run.get('readings') or {}).get('cancel') or {}
    abort = run.get('abort') or {}
    t0, t1, ta = q.get('t0'), q.get('t1'), abort.get('t')
    if (not all(type(t) in (int, float) and math.isfinite(t) for t in (t0, t1, ta))
            or not t0 <= ta <= t1 or not abort.get('active_forward')):
        return 'not_evaluated', {'reason': 'cancel did not overlap an observed native request'}
    events = run.get('events') or []
    calls = window(run, 'cancel', 'batch_dit_start')
    counts, missing, failed = {}, [], []
    steps = request.get('num_inference_steps')
    if type(steps) is not int or steps <= 0:
        return 'not_evaluated', {'reason': 'cancel step contract absent'}
    for rank in (0, 1):
        starts = [e for e in calls if rank_of(e) == rank]
        pids = {e.get('pid') for e in starts}
        installed = {e.get('pid') for e in events if e.get('event') == 'batch_observer_installed'
                     and e.get('pre_hook') is True and type(e.get('t')) in (int, float)
                     and math.isfinite(e['t']) and e['t'] <= t0}
        errors = [e for e in events if e.get('event') in ('batch_observer_error', 'truncated', 'record_error')
                  and e.get('pid') in pids and type(e.get('t')) in (int, float)
                  and math.isfinite(e['t']) and t0 <= e['t'] <= t1]
        if not starts or None in pids or not pids <= installed or errors:
            missing.append(f'rank {rank}: native row recorder absent/incomplete or failed')
            continue
        after = [e for e in starts if e['t'] >= ta]
        counts[str(rank)] = {'total': len(starts), 'after_abort': len(after)}
        if len(starts) >= steps or len(after) > CANCEL_POLICY['CANCEL_MAX_CALLS_AFTER_ABORT']:
            failed.append(f'rank {rank}: cancelled request kept executing')
    return ('fail' if failed else 'not_evaluated' if missing else 'pass'), {
        'counts': counts, 'missing': missing, 'failures': failed,
        'max_calls_after_abort': CANCEL_POLICY['CANCEL_MAX_CALLS_AFTER_ABORT']}


def lifecycle_under_parallel(reading, policy=POLICY):
    run = reading.get("parallel") or {}
    images, rd = run.get("images") or {}, run.get("readings") or {}
    clauses, values = {}, {}
    for key in ("a_repeat", "after_invalid", "after_fault", "after_cancel"):
        value = psnr(images.get(key), images.get("a"))
        values[key] = value
        clauses[key] = "pass" if value is not None and value >= policy["REPEAT_PSNR_DB"] else "fail"
    clauses["invalid_refused"] = "pass" if rd.get("invalid", {}).get("raised") is True else "fail"
    fired = window(run, "fault", "fault_fired")
    fault_error = str(rd.get("fault", {}).get("error") or "")
    clauses["fault"] = "not_evaluated" if not fired else (
        "pass" if rd.get("fault", {}).get("raised") and "INFB injected fault" in fault_error else "fail")
    cancelled = rd.get("cancel", {})
    error = str(cancelled.get("error") or "").lower()
    reached = run.get("abort") or {}
    clauses["cancel"] = "not_evaluated" if not reached.get("active_forward") else (
        "pass" if (cancelled.get("no_output") is True or
                   (cancelled.get("raised") is True and any(x in error for x in ("abort", "cancel", "499"))))
        and cancelled.get("t1", 0) >= reached.get("t", float("inf")) else "fail")
    clauses['cancel_stop'], stop = cancellation_stop(run, (reading.get('requests') or {}).get('a') or {})
    for key in ("a", "b", "a_repeat", "after_invalid", "after_fault", "after_cancel"):
        if any(not window(run, key, rank=rank) for rank in (0, 1)):
            clauses["worker_continuity"] = "not_evaluated"
    clauses["teardown"] = "pass" if run.get("shutdown_ok") is True else "fail"
    return result(clauses, {"repeat_psnr": values, "abort": reached, "fault_events": fired, 'cancel_stop': stop})
