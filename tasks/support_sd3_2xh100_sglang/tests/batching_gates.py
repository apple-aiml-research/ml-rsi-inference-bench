#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 batching gates utilities.'
from __future__ import annotations

import math
import qfe_typed
import reference_contract

POLICY = {

    "STATE_PSNR_DB": 40.0,


    "CHANGED_MAX_DB": 40.0,
    "UNCHANGED_MIN_DB": 50.0,
    "PERTURB_DELTA": 0.25,

    "DISTINCT_MAX_DB": 40.0,


    "COND_MATCH_REL": 1e-2,
    "COND_MARGIN": 10.0,

    "FAULT_MARKER": "INFB injected fault",
}
POLICY_STATE = "specified"
POLICY_BASIS = {'images': 'Required bound for images.', 'STATE_PSNR_DB': 'Required bound for STATE_PSNR_DB.', 'CHANGED_MAX_DB / UNCHANGED_MIN_DB': 'Required bound for CHANGED_MAX_DB / UNCHANGED_MIN_DB.'}


def psnr(a, b) -> float | None:
    """uint8 HWC psnr, 100 for identical; None if either is missing or the shapes differ."""
    if a is None or b is None:
        return None
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse <= 1e-9 else 10.0 * math.log10(255.0 ** 2 / mse)


def _result(verdict: str, reasons: list, details: dict) -> dict:
    return {"verdict": verdict, "reasons": reasons, "details": details}


def _rel(a, b) -> float:
    if not a or not b or len(a) != len(b) or not all(math.isfinite(v) for v in (*a, *b)):
        return float("inf")
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(x * x for x in b)) or 1e-30
    return num / den


def tp2_execution(tp: dict, dense: dict | None) -> dict:
    'Every fused DiT forward must retain the two-rank computation contract.'
    if not dense or set((tp or {}).get("ranks", {})) != {"0", "1"}:
        return _result("not_evaluated", ["TP2 batching lacks current-trial dense/rank evidence"], {})
    base = (dense.get("ranks") or {}).get("0") or {}
    dc = base.get("compute") or []
    if not dc or len(dc) != len(base.get("forwards") or []) or any(e.get("truncated") for e in dc):
        return _result("not_evaluated", ["incomplete dense fused compute record"], {})
    total = sum(e.get("macs", 0) for e in dc)
    if not total or not math.isfinite(total):
        return _result("not_evaluated", ["invalid dense fused compute record"], {})
    bad, missing, fractions = [], [], {}
    for rank, r in tp["ranks"].items():
        comp, forwards, batches = (r.get(k) or [] for k in ("compute", "forwards", "batch"))
        if not comp or not (len(comp) == len(forwards) == len(batches)) or any(e.get("truncated") for e in comp):
            missing.append(f"rank {rank}: incomplete fused compute/forward record")
            continue
        def row_work(events):
            from collections import Counter
            counts = Counter()
            for event in events:
                shape = event.get("x")
                if not shape or not all(isinstance(n, int) and n > 0 for n in shape):
                    return None
                counts[tuple(shape[1:])] += shape[0]
            return counts
        if not row_work(comp) or row_work(comp) != row_work(dc):
            missing.append(f"rank {rank}: fused input shapes differ from dense")
        work = sum(e.get("macs", 0) for e in comp)
        fractions[rank] = work / total
        if not math.isfinite(work) or work <= 0:
            missing.append(f"rank {rank}: no finite executed work")
        elif fractions[rank] > 0.75:
            bad.append(f"rank {rank}: replicated fused computation")
        if any(not isinstance(e.get("coll"), dict) for e in forwards):
            missing.append(f"rank {rank}: missing collective instrumentation")
        elif any(not any(n > 0 for op, n in e["coll"].items() if op != "broadcast") for e in forwards):
            bad.append(f"rank {rank}: fused forward without participating collectives")
        if any(e.get("completed") is not True for e in batches):
            bad.append(f"rank {rank}: incomplete fused outputs")
    return _result("fail" if bad else "not_evaluated" if missing else "pass", bad + missing,
                   {"compute_fraction_by_rank": fractions, "max_compute_fraction": 0.75})


def solo_rows(solo: dict) -> list:
    """[(key, output index, x fp, c moments)] from each request's solo FIRST forward (its first-step rows)."""
    out = []
    for key, s in sorted((solo or {}).items()):
        first = (s or {}).get("first") or {}
        for i in range(int(first.get("rows") or 0)):
            x = (first.get("x") or [None] * (i + 1))[i] or {}
            c = (first.get("c") or [None] * (i + 1))[i] or {}
            p = (first.get("p") or [None] * (i + 1))[i] or {}
            out.append((key, i, x.get("fp"), c.get("m"), p.get("m")))
    return out


def attribute(forward: dict, solo: dict, policy: dict = POLICY) -> list:
    """Each row of `forward` -> (key, output index) of the solo request whose first-step latents it carries
    BIT-IDENTICALLY, the conditioning deciding between requests that share a seed; None for a row no solo request
    explains (or explains ambiguously)."""
    cands = solo_rows(solo)
    out = []
    for i in range(int((forward or {}).get("rows") or 0)):
        x = ((forward.get("x") or [None] * (i + 1))[i] or {}).get("fp")
        c = ((forward.get("c") or [None] * (i + 1))[i] or {}).get("m")
        same = [k for k in cands if x is not None and k[2] == x]
        if not same or c is None:
            out.append(None)
            continue
        pooled = ((forward.get("p") or [None] * (i + 1))[i] or {}).get("m")
        dist = sorted((max(_rel(c, k[3]), _rel(pooled, k[4])), (k[0], k[1]))
                      for k in same if k[3] is not None and k[4] is not None and pooled is not None)
        if dist and dist[0][0] <= policy["COND_MATCH_REL"] and (
                len(dist) == 1 or dist[1][0] >= policy["COND_MARGIN"] * max(dist[0][0], 1e-12)):
            out.append(dist[0][1])
        else:
            out.append(None)
    return out


def _anchors(s: dict, key: str) -> tuple:
    """All first-step branches, with outputs identified by distinct solo noise.

    An empty negative prompt can be identical between requests. It is valid
    evidence for branch multiplicity, but cannot identify a perturbation owner.
    """
    fw = s.get("forwards") or []
    times = list(dict.fromkeys(t for f in fw for t in f.get("timesteps", [])
                               if isinstance(t, (int, float)) and math.isfinite(t)))
    if not times:
        return [], []
    noise, anchors = [], []
    for f in fw:
        if f.get("timesteps") != [times[0]] * f.get("rows", 0):
            continue
        for i in range(f.get("rows", 0)):
            try:
                x, c, p = (f[field][i] or {} for field in ("x", "c", "p"))
                fp = x["fp"]
                if fp not in noise:
                    noise.append(fp)
                anchors.append({"owner": (key, noise.index(fp)), "x": fp,
                                "c": c["m"], "p": p["m"],
                                "identity": (c["fp"], p["fp"])})
            except (KeyError, IndexError, TypeError):
                return [], times
    branches = 2 if float(s.get("guidance", 1)) > 1 else 1
    n = len(s.get("images") or [])
    if len(noise) != n or len(anchors) != n * branches:
        return [], times
    return anchors, times


def _choices(f, i, anchors, initial, policy):
    try:
        x, c, p = (f[field][i] or {} for field in ("x", "c", "p"))
        distances = sorted((max(_rel(c.get("m"), a["c"]), _rel(p.get("m"), a["p"])), j)
                           for j, a in enumerate(anchors) if not initial or a["x"] == x.get("fp"))
    except (KeyError, IndexError, TypeError):
        return []
    if not distances or distances[0][0] > policy["COND_MATCH_REL"]:
        return []
    best, j = distances[0]


    identity = anchors[j]["identity"]
    if any(d < policy["COND_MARGIN"] * max(best, 1e-12) and anchors[k]["identity"] != identity
           for d, k in distances[1:]):
        return []
    return [k for d, k in distances if d <= policy["COND_MATCH_REL"] and anchors[k]["identity"] == identity]


def cancelled_rows_evidence(forwards, solo, key, abort_t, done_t, policy=POLICY):
    """Independent conditioning attribution before/after the public abort.

    In-flight work may finish; only forwards starting after abort are charged.
    Missing/ambiguous attribution never counts as the row having disappeared.
    """
    import t2_gates
    anchors = [a for owner, run in sorted(solo.items()) for a in _anchors(run, owner)[0]]
    if not anchors or not all(type(t) in (int, float) and math.isfinite(t) for t in (abort_t, done_t)) or done_t < abort_t:
        return {'measured': False, 'reason': 'cancel anchors or time window missing'}
    if not any(a['owner'][0] == key for a in anchors):
        return {'measured': False, 'reason': 'cancelled member has no solo anchor'}
    present, after, missing = False, 0, []
    for f in forwards:
        owners = []
        if type(f.get('rows')) is not int or f['rows'] <= 0:
            missing.append('cancel forward row count missing')
            continue
        for i in range(f['rows']):
            choices = _choices(f, i, anchors, False, policy)
            resolved = {anchors[j]['owner'][0] for j in choices}
            if len(resolved) != 1:
                missing.append('ambiguous cancel row')
            else:
                owners.append(next(iter(resolved)))
        start = f.get('t_start', f.get('t'))
        if type(start) not in (int, float) or not math.isfinite(start):
            missing.append('cancel forward start missing')
            continue
        if start <= abort_t and key in owners and f.get('rows', 0) >= 2:
            present = True
        if abort_t < start <= done_t:
            after += owners.count(key)
    limit = t2_gates.POLICY['CANCEL_MAX_CALLS_AFTER_ABORT']
    return {'measured': present and not missing, 'stopped': after <= limit,
            'rows_after_abort': after, 'max_rows_after_abort': limit, 'missing': missing}


def _observed_runs(forwards, solo, expected, policy):
    """Check physical forwards, counting every request/output/CFG branch at
    every observed timestep. Split N-row and concatenated 2N-row CFG are both
    legal; neither can hide extra serial work or a missing/repeated branch.
    """
    from collections import Counter
    specs, out = {}, []
    for name, e in expected.items():
        anchors, times, errors = [], None, []
        for k in e["members"]:
            a, ts = _anchors(solo.get(k) or {}, k)
            if not a or (times is not None and ts != times):
                errors.append("missing/incompatible solo branch anchors or timesteps")
            anchors.extend(a)
            times = ts if times is None else times
        g = {"observed_layout": True, "attribution": sorted({a["owner"] for a in anchors}),
             "rows": e["rows"], "members": sorted(e["members"]), "forwards": 0,
             "row_counts": [], "timesteps": times or [], "layout_errors": errors,
             "perturbed_owners": [], "perturb_errors": [], "records": []}
        specs[name] = (e, anchors, g)
        out.append(g)
    for f in sorted(forwards or [], key=lambda f: float(f.get("t") or 0)):
        names = [name for name, (e, a, _g) in specs.items()
                 if f.get("rows") in {e["rows"], e["rows"] * e["branches"]}
                 and all(_choices(f, i, a, False, policy) for i in range(f["rows"]))]
        if len(names) != 1:
            out.append({"attribution": None, "rows": f.get("rows"), "members": [],
                        "forwards": 1, "row_counts": [f.get("rows")]})
            continue
        _e, a, g = specs[names[0]]
        g["records"].append(f)
        g["forwards"] += 1
        g["row_counts"].append(f["rows"])
        if f.get("perturbed"):
            row = (f.get("perturb_plan") or {}).get("row")
            choices = _choices(f, row, a, False, policy) if isinstance(row, int) and 0 <= row < f["rows"] else []
            owners = {a[j]["owner"] for j in choices}
            if len(owners) != 1:
                g["perturb_errors"].append("perturbed row has no unique observed owner")
            else:
                g["perturbed_owners"].append(next(iter(owners)))
    for e, anchors, g in specs.values():
        records = g.pop("records")
        times = g["timesteps"]
        if not records:
            g["attribution"] = None
        if any(f.get("completed") is not True or not f.get("timesteps") or
               len(set(f["timesteps"])) != 1 or f["timesteps"][0] not in times or
               len(f["timesteps"]) != f["rows"] for f in records):
            g["layout_errors"].append("incomplete forward or unexpected per-row timestep")
        for step, t in enumerate(times):
            fs = [f for f in records if f.get("timesteps") == [t] * f["rows"]]
            rows = [(f, i) for f in fs for i in range(f["rows"])]
            choices = [_choices(f, i, anchors, step == 0, policy) for f, i in rows]
            if (len(rows) != len(anchors) or not anchors or
                    not (len(fs) == e["branches"] and all(f["rows"] == e["rows"] for f in fs) or
                         len(fs) == 1 and fs[0]["rows"] == e["rows"] * e["branches"])):
                g["layout_errors"].append(f"step {step}: missing or extra CFG work")
                continue
            assigned = []
            def match(i, used):
                if i == len(rows):
                    offset = 0
                    for f in fs:
                        counts = Counter(anchors[j]["owner"] for j in assigned[offset:offset + f["rows"]])
                        if counts != Counter({a["owner"]: f["rows"] // e["rows"] for a in anchors}):
                            return False
                        offset += f["rows"]
                    return True
                for j in choices[i]:
                    if j not in used:
                        assigned.append(j)
                        if match(i + 1, used | {j}):
                            return True
                        assigned.pop()
                return False
            if not match(0, set()):
                g["layout_errors"].append(f"step {step}: noise/conditioning/branch multiplicity mismatch")
    return out


def consumption_plan(forwards, solo, expected, policy=POLICY):
    """Select a uniquely attributable conditioning row in each group, using
    observed inputs only. An empty negative-prompt row is never chosen by an
    arbitrary index. The hook follows this conditioning through all steps.
    """
    selectors = []
    for e in expected.values():
        anchors = [a for key in e["members"] for a in _anchors(solo[key], key)[0]]
        selected = None
        for f in forwards:
            if f.get("rows") not in {e["rows"], e["rows"] * e["branches"]}:
                continue
            for i in range(f["rows"]):
                js = _choices(f, i, anchors, True, policy)
                if len(js) == 1:
                    a = anchors[js[0]]
                    if sum(max(_rel(b["c"], a["c"]), _rel(b["p"], a["p"])) <= policy["COND_MATCH_REL"]
                           for b in anchors) == 1:
                        selected = {"c": a["c"], "p": a["p"]}
                        break
            if selected:
                break
        if selected:
            selectors.append(selected)
    return {"selectors": selectors, "delta": policy["PERTURB_DELTA"],
            "match_rel": policy["COND_MATCH_REL"], "margin": policy["COND_MARGIN"]}


def runs(forwards: list, solo: dict, policy: dict = POLICY, expected: dict | None = None) -> list:
    """The window's forwards cut into GROUPS: a group starts at a forward whose rows are attributable at the first step
    (and differs from the group running), and holds every forward until the next start. [{start, rows, members,
    forwards}] in time order."""
    if expected is not None and any("forwards" in s for s in solo.values()):
        return _observed_runs(forwards, solo, expected, policy)
    out = []
    for f in sorted(forwards or [], key=lambda e: float(e.get("t") or 0.0)):
        att = attribute(f, solo, policy)
        starts = bool(att) and all(a is not None for a in att)
        if starts and (not out or att != out[-1]["attribution"] or out[-1]["steps_seen"] > 1):
            out.append({"attribution": att, "rows": int(f.get("rows") or 0), "forwards": 0, "steps_seen": 0,
                        "members": sorted({a[0] for a in att}), "row_counts": []})
        if not out:
            out.append({"attribution": None, "rows": int(f.get("rows") or 0), "forwards": 0, "steps_seen": 0,
                        "members": [], "row_counts": []})
        g = out[-1]
        g["forwards"] += 1
        g["row_counts"].append(int(f.get("rows") or 0))
        if not starts:
            g["steps_seen"] += 1
    for g in out:
        g.pop("steps_seen", None)
    return out


def _group_of(groups: list, key: str):
    return next((g for g in groups if key in g["members"]), None)


def dynamic_batching_is_real(r: dict, policy: dict = POLICY) -> dict:
    """r: the verifier's readings (verify_dynamic_batching.collect). Clauses:
      recorded   the observer's lines exist for the solo round and both bursts, every solo request has a first forward;
      merged     every expected group of the burst shares ONE first forward carrying every member's rows;
      shared     each group runs exactly steps x branches forwards, each handed the group's full rows, and the window holds
                 no other forward (no serial re-run beside the fused one);
      matches    every burst and solo image independently clears its fixed diffusers image policy;
      consumed   the armed burst: the request owning the perturbed row moves below CHANGED_MAX_DB, every other member stays
                 >= UNCHANGED_MIN_DB (not_evaluated if the armed burst's grouping differs from the plain one's);
      reference  complete independent references and all four image metrics, for both solo and burst;
      grouping   no forward carries rows of two incompatible requests (different guidance), and the guidance-1 group runs
                 one branch per step;
      cancel     a member aborted while its group runs returns no image and an abort; every other member returns its solo
                 image; the next request is served as before (not_evaluated if the abort did not land inside a fused run);
      failure    a fault injected into a fused forward: every member returns its solo image or an error naming the fault --
                 never a missing image without one, never another member's image; the next request is served as before
                 (not_evaluated if the fault did not fire in a forward of two rows or more)."""
    clauses, reasons, not_eval = {}, [], []
    steps = int(r.get("steps") or 0)
    solo = r.get("solo") or {}
    exp = r.get("groups") or {}
    plain, armed = r.get("burst") or {}, r.get("armed") or {}

    def clause(name, ok, why=None, ne=False):
        clauses[name] = "not_evaluated" if ne else ("pass" if ok else "fail")
        if ne:
            not_eval.append(f"{name}: {why}")
        elif not ok:
            reasons.append(f"{name}: {why}")

    missing_first = sorted(k for k, s in solo.items() if not (s or {}).get("first"))
    rec_ok = bool(solo) and not missing_first and bool(plain.get("forwards")) and bool(armed.get("forwards"))
    clause("recorded", rec_ok, f"no observer lines (solo without a first forward: {missing_first}; burst forwards "
                               f"{len(plain.get('forwards') or [])}, armed {len(armed.get('forwards') or [])})")
    groups = runs(plain.get("forwards") or [], solo, policy, exp) if rec_ok else []
    details = {"groups": groups, "steps": steps}

    m_bad, s_bad, g_bad = [], [], []
    for name, e in sorted(exp.items()):
        matching = [x for x in groups if x["attribution"] and sorted(x["members"]) == sorted(e["members"])]
        g = matching[0] if len(matching) == 1 else None
        if g is None:
            m_bad.append(f"group {name} ({e['members']}) never shared a first forward; groups seen "
                         f"{[x['members'] for x in groups]}")
            continue
        if g["rows"] != e["rows"]:
            m_bad.append(f"group {name} was handed {g['rows']} rows, expected {e['rows']}")
        want = steps * int(e["branches"])
        if g.get("observed_layout"):
            s_bad.extend(f"group {name}: {err}" for err in g["layout_errors"])
            if len(g["timesteps"]) != steps:
                s_bad.append(f"group {name}: observed {len(g['timesteps'])} timesteps, expected {steps}")
            continue
        if g["forwards"] != want or any(n != e["rows"] for n in g["row_counts"]):
            s_bad.append(f"group {name}: {g['forwards']} forwards (expected {want} = {steps} steps x {e['branches']} "
                         f"branches), rows per forward {sorted(set(g['row_counts']))} (expected {e['rows']})")
        if int(e["branches"]) == 1 and g["forwards"] != steps:
            g_bad.append(f"the guidance-1 group {name} ran {g['forwards']} forwards for {steps} steps")
    extra = [x for x in groups if not x["attribution"] or not any(sorted(x["members"]) == sorted(e["members"])
                                                                  for e in exp.values())]
    for x in extra:
        mixed = {solo.get(k, {}).get("guidance") for k in x["members"]}
        if len(mixed) > 1:
            g_bad.append(f"one forward carried rows of requests with different guidance: {x['members']}")
        s_bad.append(f"a run of {x['forwards']} forward(s) outside every expected group (members {x['members']})")
    if rec_ok:
        clause("merged", not m_bad, "; ".join(m_bad))
        clause("shared", not s_bad, "; ".join(s_bad))
        clause("grouping", not g_bad, "; ".join(g_bad))
    else:
        for c in ("merged", "shared", "grouping"):
            clause(c, False, "nothing recorded", ne=True)


    required = sorted({k for group in exp.values() for k in group["members"]})
    rf_read, checks = {}, []

    q = {"rows": [], "obl": [], "pairs": [], "solo": {}}
    run_ids = {"run": r.get("run_id")}
    reqs = r.get("requests") or {}

    def outputs_of(key):
        return int((reqs.get(key) or {}).get("num_outputs_per_prompt", 1))

    def deployment(role, key, images, check, anchor_images, legal=("image",), outcome=None):
        """E rows for one deployment output set, paired with the same-run solo anchor of the same request."""
        n = outputs_of(key)
        rows = qfe_typed.output_rows(check, role, key, n)
        q["rows"] += rows
        q["obl"] += [qfe_typed.obligation(role, key, list(legal), outcome or f"{len(images or [])} image(s)",
                                          x["verdict"], n) for x in rows]
        anchors = q["solo"].get(key) or []
        for i, d_row in enumerate(rows):
            drift = (psnr(images[i], anchor_images[i])
                     if images and anchor_images and i < len(images) and i < len(anchor_images) else None)
            q["pairs"].append(qfe_typed.pair(d_row, anchors[i] if i < len(anchors) else None, drift,
                                             qfe_typed.binding(reqs.get(key), i, d_row.get("profile"), run_ids)))

    for key in required:
        single = (solo.get(key) or {}).get("images") or []
        burst_images = ((plain.get("requests") or {}).get(key) or {}).get("images") or []
        sq = reference_contract.check_outputs(r, key, single)
        bq = reference_contract.check_outputs(r, key, burst_images)
        checks.extend((sq, bq))
        rf_read[key] = {"solo": sq, "burst": bq,
                       "burst_vs_solo_psnr": [psnr(a, b) for a, b in zip(burst_images, single)]}
        srows = qfe_typed.output_rows(sq, "solo_anchor", key, outputs_of(key))
        q["solo"][key] = srows
        q["rows"] += srows
        q["obl"] += [qfe_typed.obligation("solo_anchor", key, ["image"], f"{len(single)} image(s)", x["verdict"],
                                          outputs_of(key)) for x in srows]
        deployment("burst", key, burst_images, bq, single)
    details["reference"] = rf_read
    bad = [x["reason"] for x in checks if x["verdict"] == "fail"]
    missing = [x["reason"] for x in checks if x["verdict"] == "not_evaluated"]
    clause("matches", bool(required) and not bad and not missing,
           "; ".join(bad + missing), ne=bool(missing) and not bad)
    clause("reference", bool(required) and not bad and not missing,
           "; ".join(bad + missing), ne=bool(missing) and not bad)

    a_groups = runs(armed.get("forwards") or [], solo, policy, exp) if rec_ok else []
    plan = armed.get("plan") or {}
    same_shape = [sorted(g["members"]) for g in a_groups if g["attribution"]] == \
                 [sorted(g["members"]) for g in groups if g["attribution"]]
    c_bad, c_read = [], {}
    if not rec_ok or not same_shape or not a_groups:
        clause("consumed", False, "the armed burst did not group as the plain burst did", ne=True)
    else:
        if not any(e.get("perturbed") is True for e in armed.get("forwards", [])):
            c_bad.append("the armed observer never perturbed a fused result")
        for g in a_groups:
            if not g["attribution"] or g["rows"] < 2:
                continue
            row = int(plan.get("row", 0))
            if not 0 <= row < len(g["attribution"]):
                c_bad.append("the armed row is outside the observed group")
                continue
            owner = g["attribution"][row]
            if g.get("observed_layout"):
                owners = set(g["perturbed_owners"])
                c_bad.extend(g["layout_errors"] + g["perturb_errors"])
                if len(owners) != 1:
                    c_bad.append("each group needs exactly one uniquely observed perturbed owner")
                    continue
                owner = next(iter(owners))
            for key in g["members"]:
                imgs_a = ((armed.get("requests") or {}).get(key) or {}).get("images") or []
                imgs_c = ((plain.get("requests") or {}).get(key) or {}).get("images") or []
                for i in range(max(len(imgs_a), len(imgs_c))):
                    p = psnr(imgs_a[i] if i < len(imgs_a) else None, imgs_c[i] if i < len(imgs_c) else None)
                    c_read[f"{key}[{i}]"] = p
                    is_owner = (key, i) == tuple(owner)
                    if p is None:
                        c_bad.append(f"{key}[{i}]: an image is missing")
                    elif is_owner and p >= policy["CHANGED_MAX_DB"]:
                        c_bad.append(f"{key}[{i}] owns the perturbed row and did not move ({p:.2f} dB): the fused "
                                     f"forward's result is not what it consumed")
                    elif not is_owner and p < policy["UNCHANGED_MIN_DB"]:
                        c_bad.append(f"{key}[{i}] moved ({p:.2f} dB) when another request's row was perturbed")
        details["consumed"] = c_read
        clause("consumed", not c_bad, "; ".join(c_bad))

    cancel = r.get("cancel") or {}
    if not cancel.get("fused_at_abort"):
        clause("cancel", False, "the abort did not land inside a fused run", ne=True)
        q["cancel_semantics"] = "not_evaluated"
        for key in list(cancel.get("expected_survivors") or ()) + [cancel.get("after_key")]:
            q["obl"].append(qfe_typed.obligation("cancel_survivor" if key != cancel.get("after_key") else "cancel_after",
                                                 key, ["image"], "branch not reached", "not_evaluated", outputs_of(key)))
    else:
        bad = []
        st = cancel.get("aborted_state") or {}
        error = str(st.get("error") or "").lower()
        if not (st.get("state") == "no_output" or
                (st.get("state") == "refused" and any(x in error for x in ("abort", "cancel", "499")))):
            bad.append("cancelled member returned an image or an unrelated failure")
        coverage = cancel.get("observed_stop") or {}
        if not coverage.get("measured"):
            clause("cancel_stop", False, "cancelled conditioning row stop was not recorded", ne=True)
        elif not coverage.get("stopped"):
            bad.append("the cancelled member kept contributing native rows after the abort")
        if set(cancel.get("survivors") or {}) != set(cancel.get("expected_survivors") or ()):
            bad.append("cancel survivor readings are incomplete")


        q["cancel_semantics"] = "fail" if bad else "pass"
        checks = [reference_contract.check_outputs(r, key, imgs)
                  for key, imgs in sorted((cancel.get("survivors") or {}).items())]
        checks.append(reference_contract.check_outputs(r, cancel.get("after_key"), [cancel.get("after")]))
        for (key, imgs), chk in zip(sorted((cancel.get("survivors") or {}).items()), checks):
            deployment("cancel_survivor", key, imgs, chk, (solo.get(key) or {}).get("images") or [])
        deployment("cancel_after", cancel.get("after_key"), [cancel.get("after")], checks[-1],
                   [cancel.get("after_solo")])
        for key in sorted(set(cancel.get("expected_survivors") or ()) - set(cancel.get("survivors") or {})):
            q["obl"].append(qfe_typed.obligation("cancel_survivor", key, ["image"], "missing", "fail", outputs_of(key)))
        bad.extend(x["reason"] for x in checks if x["verdict"] == "fail")
        missing = [x["reason"] for x in checks if x["verdict"] == "not_evaluated"]
        details["cancel_reference"] = checks
        clause("cancel", not bad and not missing, "; ".join(bad + missing), ne=bool(missing) and not bad)

    fail = r.get("failure") or {}
    if not fail.get("fired_rows") or int(fail["fired_rows"]) < 2:
        clause("failure", False, f"the fault did not fire in a fused forward (rows {fail.get('fired_rows')})", ne=True)
        q["failure_semantics"] = "not_evaluated"
        for key in list(fail.get("expected_members") or ()) + [fail.get("after_key")]:
            q["obl"].append(qfe_typed.obligation("failure_member" if key != fail.get("after_key") else "failure_after",
                                                 key, ["image", "refused_with_fault_marker"] if key != fail.get("after_key")
                                                 else ["image"], "branch not reached", "not_evaluated", outputs_of(key)))
    else:
        bad = []
        if set(fail.get("members") or {}) != set(fail.get("expected_members") or ()):
            bad.append("fused failure member readings are incomplete")
        checks = []
        for key, m in sorted((fail.get("members") or {}).items()):
            if m.get("state") == "image":
                chk = reference_contract.check_outputs(r, key, m.get("images") or [])
                checks.append(chk)
                deployment("failure_member", key, m.get("images") or [], chk, (solo.get(key) or {}).get("images") or [],
                           legal=("image", "refused_with_fault_marker"), outcome="image")
            elif m.get("state") == "refused":
                if policy["FAULT_MARKER"] not in str(m.get("error") or ""):
                    bad.append(f"{key} was refused for another reason: {str(m.get('error'))[:160]}")
                    q["obl"].append(qfe_typed.obligation("failure_member", key, ["image", "refused_with_fault_marker"],
                                                         "refused for another reason", "fail", outputs_of(key)))
                else:

                    q["obl"].append(qfe_typed.obligation("failure_member", key, ["image", "refused_with_fault_marker"],
                                                         "refused_with_fault_marker", "pass", outputs_of(key)))
            else:
                bad.append(f"{key} came back {m.get('state')} (no image and no error)")
                q["obl"].append(qfe_typed.obligation("failure_member", key, ["image", "refused_with_fault_marker"],
                                                     f"{m.get('state')} (no image and no error)", "fail", outputs_of(key)))
        for key in sorted(set(fail.get("expected_members") or ()) - set(fail.get("members") or {})):
            q["obl"].append(qfe_typed.obligation("failure_member", key, ["image", "refused_with_fault_marker"],
                                                 "missing", "fail", outputs_of(key)))
        q["failure_semantics"] = "fail" if bad else "pass"
        checks.append(reference_contract.check_outputs(r, fail.get("after_key"), [fail.get("after")]))
        deployment("failure_after", fail.get("after_key"), [fail.get("after")], checks[-1], [fail.get("after_solo")])
        bad.extend(x["reason"] for x in checks if x["verdict"] == "fail")
        missing = [x["reason"] for x in checks if x["verdict"] == "not_evaluated"]
        details["failure_reference"] = checks
        clause("failure", not bad and not missing, "; ".join(bad + missing), ne=bool(missing) and not bad)
    verdict = "fail" if reasons else ("not_evaluated" if not_eval else "pass")
    details["clauses"] = clauses

    structural = {k: clauses[k] for k in ("recorded", "merged", "shared", "grouping", "consumed") if k in clauses}
    structural["cancel_semantics"] = q.get("cancel_semantics", "not_evaluated")
    if "cancel_stop" in clauses:
        structural["cancel_stop"] = clauses["cancel_stop"]
    structural["failure_semantics"] = q.get("failure_semantics", "not_evaluated")
    details["qfe"] = qfe_typed.block(structural, q["rows"], q["obl"], q["pairs"], verdict,
                                     {"burst_vs_solo_psnr_db": {k: v["burst_vs_solo_psnr"] for k, v in rf_read.items()}})
    return _result(verdict, reasons + not_eval, details)


def batching_x_multioutput_isolation(r: dict, policy: dict = POLICY, *, require_adapter_fence: bool = True) -> dict:
    """Clauses:
      multioutput  the multi-output member's outputs in the burst: both present, each passing its independent request reference, and the
                   two different images;
      seeds        each burst row carries the first-step latents of its own request's seed + output index -- the rows of
                   the multi-output member are two DIFFERENT solo rows (their own seeds), never one seed twice;
      ownership    two members with the SAME seed and different prompts: their rows are told apart by conditioning, and
                   each burst image passes its own independent request reference;
      fence        a request sent before an adapter change and one sent after it never share a forward, and each equals
                   its solo image under its own adapter state (not_evaluated if the second request was not yet queued
                   when the first started, or if the adapter does not change the second request's image)."""
    clauses, reasons, not_eval = {}, [], []
    solo = r.get("solo") or {}
    plain = r.get("burst") or {}
    roles = r.get("roles") or {}

    def clause(name, ok, why=None, ne=False):
        clauses[name] = "not_evaluated" if ne else ("pass" if ok else "fail")
        if ne:
            not_eval.append(f"{name}: {why}")
        elif not ok:
            reasons.append(f"{name}: {why}")

    groups = runs(plain.get("forwards") or [], solo, policy, r.get("groups"))
    mo = roles.get("multioutput")
    got = ((plain.get("requests") or {}).get(mo) or {}).get("images") or []
    want = (solo.get(mo) or {}).get("images") or []
    bad, mo_struct = [], []
    quality = multioutput_quality = reference_contract.check_outputs(r, mo, got)
    if len(want) != 2 or len(got) != 2:
        bad.append(f"{mo}: expected exactly two independent outputs")
        mo_struct.append("count")
    if quality["verdict"] == "fail":
        bad.append(quality["reason"])
    if len(got) == 2:
        d = psnr(got[0], got[1])
        if d is None or d >= policy["DISTINCT_MAX_DB"]:
            bad.append(f"{mo}'s two outputs are not distinct valid images ({d} dB)")
            mo_struct.append("distinct")

    q_struct = {"multioutput_outputs": "fail" if mo_struct else "pass"}
    clause("multioutput", not bad and quality["verdict"] == "pass", "; ".join(bad) or quality["reason"],
           ne=not bad and quality["verdict"] == "not_evaluated")
    g = _group_of(groups, mo)
    if g is None or not g["attribution"]:
        clause("seeds", False, f"no fused first forward carries {mo}'s rows")
    else:
        rows = [a for a in g["attribution"] if a and a[0] == mo]
        clause("seeds", sorted(i for _k, i in rows) == [0, 1] and not g.get("layout_errors"),
               f"{mo}'s rows carried first-step latents of outputs {sorted(i for _k, i in rows)} (expected [0, 1]: its "
               f"own seed and seed + 1)")
    a, b = roles.get("same_seed") or (None, None)
    bad = []
    ga = _group_of(groups, a)
    if not (ga and ga["attribution"] and b in ga["members"]):
        bad.append(f"{a} and {b} (one seed, two prompts) were not told apart in one fused forward")

    q_struct["ownership_separation"] = "fail" if bad else "pass"
    ownership_checks = []
    for k in (a, b):
        images = ((plain.get("requests") or {}).get(k) or {}).get("images") or []
        quality = reference_contract.check_outputs(r, k, images)
        ownership_checks.append(quality)
        if quality["verdict"] == "fail":
            bad.append(quality["reason"])
    missing = [x["reason"] for x in ownership_checks if x["verdict"] == "not_evaluated"]
    clause("ownership", not bad and not missing, "; ".join(bad + missing), ne=bool(missing) and not bad)
    if require_adapter_fence:
        fence = r.get("fence") or {}
        if not fence.get("queued_behind"):
            clause("fence", False, "the request after the adapter change was not queued before the first one started", ne=True)
        elif fence.get("adapter_effect_db") is None or fence["adapter_effect_db"] >= policy["CHANGED_MAX_DB"]:
            clause("fence", False, f"the adapter does not change the request ({fence.get('adapter_effect_db')} dB)", ne=True)
        else:
            bad = []
            if fence.get("shared_forward"):
                bad.append("the requests before and after the adapter change shared a forward")
            for k in ("before", "after"):
                p = psnr(fence.get(f"{k}_image"), fence.get(f"{k}_solo"))
                if p is None or p < policy["STATE_PSNR_DB"]:
                    bad.append(f"the request {k} the change: {p} dB to its solo image under its own adapter state")
            clause("fence", not bad, "; ".join(bad))
    else:
        clauses["fence"] = "excluded_by_frozen_contract"
    verdict = "fail" if reasons else ("not_evaluated" if not_eval else "pass")


    q_struct["seeds"] = clauses.get("seeds", "not_evaluated")
    if clauses.get("fence") not in (None, "excluded_by_frozen_contract"):
        q_struct["fence"] = clauses["fence"]
    reqs = r.get("requests") or {}
    run_ids = {"run": r.get("run_id")}
    q_rows, q_obl, q_pairs = [], [], []
    for role, key, check, images in [("multioutput", mo, multioutput_quality, got)] + [
            ("ownership", k, c, ((plain.get("requests") or {}).get(k) or {}).get("images") or [])
            for k, c in zip((a, b), ownership_checks)]:
        n = int((reqs.get(key) or {}).get("num_outputs_per_prompt", 1))
        rows = qfe_typed.output_rows(check, role, key, n)
        q_rows += rows
        q_obl += [qfe_typed.obligation(role, key, ["image"], f"{len(images)} image(s)", x["verdict"], n) for x in rows]
        anchor_images = (solo.get(key) or {}).get("images") or []
        for i, d_row in enumerate(rows):
            drift = psnr(images[i], anchor_images[i]) if i < len(images) and i < len(anchor_images) else None
            q_pairs.append(qfe_typed.pair(d_row, None, drift,
                                          qfe_typed.binding(reqs.get(key), i, d_row.get("profile"), run_ids)))
    qfe = qfe_typed.block(q_struct, q_rows, q_obl, q_pairs, verdict)
    return _result(verdict, reasons + not_eval, {"clauses": clauses, "groups": groups, "multioutput_reference": multioutput_quality,
                                               "ownership_reference": ownership_checks, "qfe": qfe})
