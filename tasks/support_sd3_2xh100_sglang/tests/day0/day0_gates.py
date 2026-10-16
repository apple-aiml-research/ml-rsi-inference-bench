#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 day0 gates utilities.'
from __future__ import annotations

import math

import day0_policy as P

VERDICTS = ("pass", "fail", "insufficient", "not_evaluated")


def _state_key(expected: dict) -> str:
    """'base', 'A@1', or for several adapters at once (request sequence v4) 'A+C@1,0.5'."""
    a, s = expected.get("adapter"), expected.get("scale")
    if a is None:
        return "base"
    if isinstance(a, (list, tuple)):
        sc = [float(x or 0.0) for x in (s if isinstance(s, (list, tuple)) else [s] * len(a))]
        return "base" if not any(sc) else "+".join(str(x) for x in a) + "@" + ",".join(f"{x:g}" for x in sc)
    s = float(s or 0.0)
    return "base" if s == 0.0 else f"{a}@{s:g}"


def _rms(values: list[float]) -> float | None:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else None


GRID = ("base", "half", "state", "double")


def lora_reference_verdict(readings: list[dict], api_errors: list | None = None, margin_max: float | None = None,
                           floor_ratio_max: float | None = None, power_margin: float | None = None,
                           scored_points=None, expected: list | None = None,
                           refusal_failures: list | None = None) -> dict:
    """Classify served adapter strength and measure its distance from the independent reference."""


    mmax = P.LORA_MARGIN_MAX if margin_max is None else margin_max
    cmax = P.LORA_FLOOR_RATIO_MAX if floor_ratio_max is None else floor_ratio_max
    pmar = P.LORA_POWER_MARGIN if power_margin is None else power_margin
    default_points = {tuple(p) for p in (P.LORA_SCORED_POINTS if scored_points is None else scored_points)}
    out = {"verdict": "not_evaluated", "reasons": [], "statistic": "v2: served strength + legal-floor ratio, per request",
           "margin_max": mmax, "floor_ratio_max": cmax, "power_margin": pmar, "points": [], "requests": {},
           "coverage": "frozen request set" if expected is not None else "not given"}
    if api_errors:
        out["api_errors"] = list(api_errors)
    if refusal_failures:
        out["refusal_failures"] = list(refusal_failures)
    spec = None if expected is None else {int(e["request"]): e for e in expected}
    serve_errors = [x for e in (spec or {}).values() for x in (e.get("serve_errors") or [])]
    if not readings:
        if api_errors or refusal_failures or serve_errors:
            out["verdict"] = "fail"
            out["reasons"].append("no DiT reading was captured, and the candidate's engine failed: "
                                  + str((list(api_errors or []) + list(refusal_failures or []) + serve_errors)[0]))
        else:
            out["reasons"].append("no DiT reading was captured")
        return out
    by_req: dict[int, list[dict]] = {}
    ignored = set()
    for r in readings:
        k = int(r["request"])
        if spec is not None and k not in spec:
            ignored.add(k)
            continue
        by_req.setdefault(k, []).append(r)
    if ignored:
        out["ignored_requests"] = sorted(ignored)
    req_ids = sorted(spec) if spec is not None else sorted(by_req)

    def state_of(k):
        if spec is not None:
            return _state_key(spec[k].get("expected") or {})
        return _state_key(by_req[k][0]["expected"])

    base_reqs = [k for k in req_ids if state_of(k) == "base"]
    unjudged: dict = {}
    for k in req_ids:
        state = state_of(k)
        e = spec.get(k) if spec is not None else None
        pts = {tuple(p) for p in e.get("points") or []} if e is not None else default_points
        grid = tuple(e.get("grid") or ()) if e is not None else (() if state == "base" else GRID)
        if state != "base" and not ("state" in grid and "base" in grid):
            raise ValueError(f"r{k} ({state}): a grid needs base and state, got {list(grid)}")
        rec = {"state": state, "scored_points": 0, "grid": list(grid)}
        if e is not None and e.get("serve_errors"):
            rec["serve_errors"] = list(e["serve_errors"])
        errs, floors, rows, rows16, seen = [], [], [], [], set()
        for r in by_req.get(k, []):
            where = (r["fixture"], int(r["step"]))
            scored = where in pts
            pt = {"request": k, "point": f"{r['fixture']}@{int(r['step'])}.p{int(r.get('pass', 0))}", "state": state,
                  "scored": scored}
            f16 = r.get("floor16")
            if state == "base":
                err = r.get("err_state")
                pt.update(err=err, floor16=f16, C=(err / f16) if err is not None and f16 else None)
                if scored and err is not None and f16:
                    errs.append(err)
                    floors.append(f16)
                    seen.add(where)
            else:
                d, d16 = r.get("dist") or {}, r.get("dist16") or {}
                if any(d.get(t) is None for t in grid) or any(d16.get(t) is None for t in grid):
                    pt["status"] = "not_evaluated: the replay did not record the strength grid here"
                    out["points"].append(pt)
                    continue
                d, d16 = {t: d[t] for t in grid}, {t: d16[t] for t in grid}
                pt.update(dist=d, dist16=d16, served=min(grid, key=lambda t: d[t]),
                          C=d["state"] / d16["state"] if d16["state"] else None)
                if scored:
                    rows.append(d)
                    rows16.append(d16)
                    seen.add(where)
            out["points"].append(pt)
            rec["scored_points"] += int(scored)
        missing = sorted(pts - seen)
        if missing:
            rec["missing_points"] = [f"{f}@{s}" for f, s in missing]
        if state == "base":
            if errs:
                rec["C"] = _rms(errs) / _rms(floors)
        elif rows:
            D = {t: _rms([x[t] for x in rows]) for t in grid}
            D16 = {t: _rms([x[t] for x in rows16]) for t in grid}
            powered = [t for t in grid if t != "state" and D16[t] > 0 and D16["state"] / D16[t] <= pmar]
            rec.update(D=D, D16=D16, served=min(grid, key=lambda t: D[t]), powered=powered,
                       M16={t: D16["state"] / D16[t] if D16[t] else None for t in grid if t != "state"},
                       C=D["state"] / D16["state"] if D16["state"] else None)


            den = min(D[t] for t in powered) if powered else None
            rec["M"] = (None if den is None else D["state"] / den if den > 0
                        else (math.inf if D["state"] > 0 else 1.0))
        if (rec.get("C") is None or missing
                or (state != "base" and rec.get("M") is None and rec.get("served") == "state")):
            unjudged[k] = (f"r{k} ({state})" + (f" no reading at {rec['missing_points']}" if missing else ""))
        out["requests"][str(k)] = rec
    c0 = out["requests"].get(str(base_reqs[0]), {}).get("C") if base_reqs else None
    lim = cmax * max(1.0, c0 or 0.0)
    out["C_0"], out["floor_ratio_limit"] = c0, lim
    failing = []
    for ks, rec in out["requests"].items():
        k = int(ks)
        why = []
        if rec.get("serve_errors"):
            why.append(f"the engine did not serve it: {rec['serve_errors'][0]}")
        if rec.get("C") is not None and rec["C"] > lim:
            why.append(f"{rec['C']:.2f} legal floors > {lim:.2f}")
        if rec["state"] != "base" and rec.get("served"):
            if rec["served"] != "state":
                why.append(f"served {_grid_name(rec['state'], rec['served'])}, not {rec['state']}")
            elif rec.get("M") is not None and rec["M"] > mmax:
                why.append(f"margin over the resolved neighbours {rec['M']:.2f} > {mmax}")
        rec["status"] = "fail" if why else ("pass" if k not in unjudged else "not_evaluated")
        if why:
            failing.append(f"r{k} ({rec['state']}): " + "; ".join(why))
    if api_errors:
        out["verdict"] = "fail"
        out["reasons"].append(f"the LoRA API refused {len(api_errors)} op(s): {api_errors[0]}")
    if refusal_failures:
        out["verdict"] = "fail"
        out["reasons"].append(f"{len(refusal_failures)} op(s) the contract says must be refused were accepted: "
                              f"{refusal_failures[0]}")
    if failing:
        out["verdict"] = "fail"
        out["reasons"].append(f"{len(failing)} request(s) off the reference: " + " | ".join(failing))
    if out["verdict"] != "fail":
        if spec is None:
            out["reasons"].append("the frozen request set was not given: which requests and points had to be read "
                                  "is unknown, so a missing one cannot be told from a smaller sequence")
        elif c0 is None:
            out["reasons"].append("no base-request reading with a legal floor, so no C_0")
        elif unjudged:
            out["reasons"].append(f"not evaluated: {list(unjudged.values())} (no scored reading, a point of the "
                                  f"frozen set not read, no legal floor, or no neighbour the legal port resolves)")
        else:
            out["verdict"] = "pass"
    return out


def _grid_name(state: str, tag: str) -> str:
    """A grid tag as the state it stands for: base / the state / X@s/2 / X@2s, or a named neighbour of v4
    (rslora_ignored, pattern_ignored, first_only, last_only, first_at_last_scale) as `<state> [<tag>]`."""
    if tag in ("state", "base"):
        return state if tag == "state" else "base"
    if tag in ("half", "double") and "@" in state and "+" not in state:
        ad, s = state.split("@")
        return f"{ad}@{float(s) * (0.5 if tag == 'half' else 2.0):g}"
    return f"{state} [{tag}]"


def lora_reference_v1_readings(readings: list[dict], api_errors: list | None = None, coef: float | None = None,
                               discrimination_min: float | None = None, scored_points=None) -> dict:
    """Diagnostic v1 statistic: compare each point's error with a multiple of the candidate's
    base-request error at that point. These readings do not determine the active verdict.

    readings: one dict per (request, fixture, step, pass) the replay measured --
        {"request", "fixture", "step", "pass", "expected": {"adapter", "scale"}, "err_state", "err_base",
         "effect_ref", "beta"}
    err_state = rel_l2(candidate DiT output, reference DiT output for the request's expected state) on the candidate's
    own captured inputs; err_base the same against the base reference; effect_ref = rel_l2(ref_state, ref_base).

    The floor at a reading point is the candidate's own base request's err_state there (request 0): its measured
    precision. A reading passes iff err_state <= coef x floor. An adapter reading counts only where effect_ref >=
    discrimination_min x floor. Every expected state of the sequence needs at least one counted SCORED reading."""
    coef = P.LORA_COEF if coef is None else coef
    dmin = P.LORA_DISCRIMINATION_MIN if discrimination_min is None else discrimination_min
    scored_points = {tuple(p) for p in (P.LORA_SCORED_POINTS if scored_points is None else scored_points)}
    out = {"verdict": "not_evaluated", "reasons": [], "coef": coef, "discrimination_min": dmin,
           "floors": {}, "readings": [], "states": {}}
    if api_errors:
        out["api_errors"] = list(api_errors)
    if not readings:
        out["reasons"].append("no DiT reading was captured")
        return out
    floors = {}
    for r in readings:
        if r.get("request") == 0 and r.get("err_state") is not None:
            floors[(r["fixture"], int(r["step"]), int(r.get("pass", 0)))] = float(r["err_state"])
    out["floors"] = {f"{f}@{s}.p{p}": v for (f, s, p), v in sorted(floors.items())}
    needed = {_state_key(r["expected"]) for r in readings}
    counted = {k: 0 for k in needed}
    failing = []
    for r in readings:
        key = (r["fixture"], int(r["step"]), int(r.get("pass", 0)))
        floor = floors.get(key)
        state = _state_key(r["expected"])
        scored = (r["fixture"], int(r["step"])) in scored_points
        rec = {"request": r["request"], "point": f"{key[0]}@{key[1]}.p{key[2]}", "state": state, "scored": scored,
               "err_state": r.get("err_state"), "beta": r.get("beta"), "effect_ref": r.get("effect_ref")}
        if floor is None or r.get("err_state") is None:
            rec["status"] = "not_evaluated: no base floor at this point" if floor is None else "not_evaluated: no err"
            out["readings"].append(rec)
            continue
        f = max(floor, P.LORA_FLOOR_MIN)
        rec["ratio"] = float(r["err_state"]) / f
        if state != "base":
            eff = r.get("effect_ref")
            rec["effect_over_floor"] = None if eff is None else float(eff) / f
            if eff is None or float(eff) < dmin * f:
                rec["status"] = "not_evaluated: the adapter effect does not rise above the floor here"
                out["readings"].append(rec)
                continue
        ok = float(r["err_state"]) <= coef * f
        rec["status"] = "pass" if ok else "fail"
        out["readings"].append(rec)
        if scored:
            counted[state] += 1
            if not ok:
                failing.append(rec)
    out["states"] = counted
    if api_errors:
        out["verdict"] = "fail"
        out["reasons"].append(f"the LoRA API refused {len(api_errors)} op(s): {api_errors[0]}")
    if failing:
        out["verdict"] = "fail"
        worst = max(failing, key=lambda x: x["ratio"])
        out["reasons"].append(f"{len(failing)} scored reading(s) off the reference; worst {worst['point']} request "
                              f"{worst['request']} ({worst['state']}): {worst['ratio']:.2f}x the floor > {coef}")
    if out["verdict"] != "fail":
        missing = sorted(k for k, n in counted.items() if n == 0)
        if not floors:
            out["reasons"].append("no base-request reading, so no floor")
        elif missing:
            out["reasons"].append(f"no counted scored reading for state(s) {missing}")
        else:
            out["verdict"] = "pass"
    return out


def lora_isolation_verdict(levels: dict, repeat_coef: float | None = None, psnr_margin_db: float | None = None,
                           ssim_margin: float | None = None, required: dict | None = None) -> dict:
    'levels: {"<fixture>/<level>": {"repeat": {"identical": bool, "psnr", "ssim", "rel_l2": float},'


    k = P.LORA_REPEAT_COEF if repeat_coef is None else repeat_coef
    pm = P.LORA_ISO_PSNR_MARGIN_DB if psnr_margin_db is None else psnr_margin_db
    sm = P.LORA_ISO_SSIM_MARGIN if ssim_margin is None else ssim_margin
    out = {"verdict": "not_evaluated", "reasons": [], "repeat_coef": k, "pixel": {"psnr_margin_db": pm,
                                                                                  "ssim_margin": sm},
           "levels": {}, "coverage": "frozen pair set" if required is not None else "not given"}
    missing = []
    for prefix, pks in sorted((required or {}).items()):
        names = [n for n in levels if n == prefix or n.startswith(prefix + ".")]
        if not names:
            missing.append(f"{prefix} (no level measured)")
        for n in names:
            got = (levels[n] or {}).get("pairs") or {}
            missing += [f"{n}:{pk} (not measured)" for pk in pks if got.get(pk) is None]
    if not levels:
        out["reasons"].append("no equal-pair measurement")
        return out
    failing = []
    for name, lv in sorted(levels.items()):
        rep, pairs = lv.get("repeat"), lv.get("pairs") or {}
        res = {"repeat": rep, "pairs": {}}
        if not rep or not pairs:
            missing.append(name)
            out["levels"][name] = res
            continue
        pixel = name.endswith("/image")
        for pk, pv in sorted(pairs.items()):
            if pv is None:
                missing.append(f"{name}:{pk}")
                continue
            rel_ok = float(pv.get("rel_l2", math.inf)) <= k * float(rep.get("rel_l2", 0.0))
            if rep.get("identical"):
                ok = bool(pv.get("identical"))
            elif pixel:
                if None in (pv.get("psnr"), pv.get("ssim"), rep.get("psnr"), rep.get("ssim")):
                    missing.append(f"{name}:{pk} (no PSNR / SSIM)")
                    res["pairs"][pk] = dict(pv, ok=None, rel_l2_rule_diagnostic=rel_ok)
                    continue
                ok = (float(pv["psnr"]) >= float(rep["psnr"]) - pm and float(pv["ssim"]) >= float(rep["ssim"]) - sm)
            else:
                ok = rel_ok
            res["pairs"][pk] = dict(pv, ok=ok, **({"rel_l2_rule_diagnostic": rel_ok} if pixel else {}))
            if not ok:
                failing.append(f"{name}:{pk}")
        out["levels"][name] = res
    if failing:
        out["verdict"] = "fail"
        out["reasons"].append(f"equal pairs that differ beyond the repeat noise: {failing}")
    elif missing:
        out["reasons"].append(f"not measured: {missing}")
    elif required is None:
        out["reasons"].append("the frozen pair set was not given: which pairs had to be measured is unknown")
    else:
        out["verdict"] = "pass"
    return out


_LORA_ORDER = {"fail": 0, "not_evaluated": 1, "pass": 2}


def lora_multi_adapter_verdict(readings: list[dict], levels: dict, api_errors: list | None = None,
                               expected: list | None = None, required: dict | None = None, **bars) -> dict:
    """lora_multi_adapter_composition (capability lora_multi_adapter, request sequence v4 r12-r13): several adapters at
    once, each at its own scale and scaling (alpha / r, alpha / sqrt(r) under rsLoRA).
      requests  lora_reference_verdict's v2 statistic and bars on each multi request's grid -- base / state /
                first_only / last_only / first_at_last_scale (the sequence's named neighbours) -- with C_0 from the first
                base request, whose readings ride along in `readings` / `expected`
      pair      the same request sent twice (r12 / r13) at the repeat pair's level (lora_isolation_verdict on the
                multi_equal_pairs levels)
    The verdict is the worse of the two (fail < not_evaluated < pass); the API refusing the multi op fails it."""
    ref = lora_reference_verdict(readings, api_errors, expected=expected, **bars)
    iso = lora_isolation_verdict(levels, required=required)
    out = {"verdict": min((ref["verdict"], iso["verdict"]), key=lambda x: _LORA_ORDER[x]), "reasons": [],
           "requests": ref["requests"], "C_0": ref.get("C_0"), "floor_ratio_limit": ref.get("floor_ratio_limit"),
           "coverage": {"requests": ref["coverage"], "pairs": iso["coverage"]},
           "reference": {"verdict": ref["verdict"], "reasons": ref["reasons"]},
           "pair": {"verdict": iso["verdict"], "reasons": iso["reasons"], "levels": iso["levels"]}}
    for name, v in (("requests", ref), ("pair", iso)):
        if v["verdict"] != "pass":
            out["reasons"] += [f"{name}: {r}" for r in v["reasons"]]
    return out


def fp8_execution_verdict(ev: dict, marlin: str | None = None, kernel_names: str | None = None,
                          flashinfer: str | None = None, consumption: str | None = None) -> dict:
    'ev (one armed DiT forward of the FP8 engine, from the worker tap and the replay):'


    marlin = P.MARLIN_POLICY if marlin is None else marlin
    kernel_names = P.KERNEL_NAME_POLICY if kernel_names is None else kernel_names
    flashinfer = P.FLASHINFER_BOUNDARY_POLICY if flashinfer is None else flashinfer
    consumption = P.RESULT_CONSUMPTION_POLICY if consumption is None else consumption
    if (marlin not in P.MARLIN_POLICIES or kernel_names not in P.KERNEL_NAME_POLICIES
            or flashinfer not in P.FLASHINFER_BOUNDARY_POLICIES or consumption not in P.RESULT_CONSUMPTION_POLICIES):
        raise ValueError(f"unknown policy {marlin!r} / {kernel_names!r} / {flashinfer!r} / {consumption!r}")
    out = {"verdict": "not_evaluated", "reasons": [], "checks": {}, "execution_type": None,
           "policy": dict(P.fp8_policy(), marlin=marlin, kernel_names=kernel_names, flashinfer_boundary=flashinfer,
                          result_consumption=consumption)}
    if not ev or not ev.get("captured"):
        if ev and ev.get("capture_started_not_finished"):

            out["verdict"] = "fail"
            out["checks"]["served"] = "fail"
            out["reasons"].append("the armed DiT forward raised inside the candidate: an FP8 capture started and never "
                                  "finished, so the engine does not serve with the FP8 asset")
            return out
        out["reasons"].append("the worker tap captured no FP8 forward")
        return out
    fails = []

    st = ev.get("store") or {}
    exp_q, got_q = st.get("q_modules_expected"), st.get("q_modules_matched")
    if exp_q is None or got_q is None:
        out["checks"]["store"] = "not_evaluated"
        out["reasons"].append("no storage census")
        return out
    out["checks"]["store"] = "pass" if got_q == exp_q else "fail"
    if got_q != exp_q:
        fails.append(f"store: {got_q} of {exp_q} quantised weights are held as the asset's float8 bytes")

    exp_s, got_s = st.get("scales_expected"), st.get("scales_matched")
    if exp_s is not None and got_s is not None:
        out["checks"]["scales"] = "pass" if got_s >= exp_s else "fail"
        if got_s < exp_s:
            fails.append(f"scales: {got_s} of {exp_s} quantised layers hold the asset's block scales "
                         "(weight_scale_inv) for what they hold")

    unknown: list = []

    dc = ev.get("dispatch")
    if dc is None:
        out["checks"]["dispatch"] = "not_evaluated"
        out["reasons"].append("no dispatch census")
        return out
    bad = int(dc.get("nonfp8_q_shape_token_calls", 0))
    unparsed = {str(k): int(v) for k, v in (dc.get("unparsed") or {}).items()}
    out["checks"]["dispatch"] = "fail" if bad else ("unknown" if unparsed else "pass")
    if bad:
        fails.append(f"dispatch: {bad} token-level 16/32-bit GEMM(s) ran with a quantised layer's (N, K)")
    elif unparsed:
        unknown.append(f"dispatch: {sum(unparsed.values())} GEMM-like call(s) the census could not size "
                       f"({sorted(unparsed)[:3]}): whether one ran a quantised layer's (N, K) in 16/32 bits is unknown")

    betas = [float(b["beta"]) for b in (ev.get("beta") or []) if b.get("beta") is not None and math.isfinite(b["beta"])]
    if len(betas) < P.BETA_MIN_LAYERS:
        out["checks"]["beta"] = "not_evaluated"
        if not fails:
            out["reasons"].append(f"beta read on {len(betas)} layer(s) < {P.BETA_MIN_LAYERS}")
            return out
    else:
        w8a8 = all(b >= P.BETA_W8A8_MIN for b in betas)
        w8a16 = all(b <= P.BETA_W8A16_MAX for b in betas)
        out["checks"]["beta"] = {"n": len(betas), "min": min(betas), "max": max(betas),
                                 "signature": "w8a8" if w8a8 else ("w8a16" if w8a16 else "mixed")}
        if consumption == "R1":


            exp_b = ev.get("beta_expected")
            out["checks"]["beta"]["expected"] = exp_b
            if exp_b is None:
                unknown.append("beta: the record does not say how many layers the tap sampled, so a missing reading "
                               "cannot be told from a smaller sample")
            elif len(betas) < int(exp_b):
                unknown.append(f"beta: read on {len(betas)} of the {int(exp_b)} layers the tap sampled "
                               f"({len(ev.get('beta_errors') or [])} replay error(s))")
        if w8a16:
            out["execution_type"] = "w8a16 (FP8 weight-only)"
            if marlin == "M1":
                fails.append("beta: the layers carry no activation quantisation (W8A16, FP8 weight-only); under M1 "
                             "that is not this profile's FP8 execution, whatever its numerics")
        elif w8a8:
            out["execution_type"] = "w8a8"
        else:
            fails.append(f"beta: per-layer signatures between W8A16 and W8A8 (min {min(betas):.3f}, "
                         f"max {max(betas):.3f}); neither the declared activation quantisation nor none")

    fb = ev.get("flashinfer")
    if fb and fb.get("calls"):
        out["checks"]["flashinfer_boundary"] = dict(fb, policy=flashinfer)
        if flashinfer == "F1":
            if fb.get("sixteen_bit_weight_calls"):
                fails.append(f"flashinfer: {fb['sixteen_bit_weight_calls']} call(s) of the FP8 entry consumed a 16-bit "
                             "weight for a quantised layer (dequantise, then quantise again)")
            if fb.get("provenance_bad_layers"):
                fails.append(f"flashinfer: {len(fb['provenance_bad_layers'])} quantised layer(s) returned something other "
                             "than the FP8 entry's output")
            if fb.get("code_altered_calls"):
                fails.append(f"flashinfer: {fb['code_altered_calls']} call(s) ran the pinned entry's function object with "
                             "its code object replaced in memory after verification (the file bytes unchanged)")

    dr = ev.get("direct")
    if dr is not None:
        out["checks"]["result_consumption"] = dict(dr, policy=consumption)
        if consumption == "R1" and dr.get("show_layers"):
            fails.append(f"consumption: {len(dr['show_layers'])} quantised layer(s) ran an FP8-operand GEMM whose "
                         "result is not what the layer returned (called for show, on its own operands or on others); "
                         f"e.g. {dr['show_layers'][:3]}")
    if fails:
        out["verdict"] = "fail"
        out["reasons"].extend(fails)
        return out
    if out["execution_type"] == "w8a16 (FP8 weight-only)":


        out["verdict"] = "insufficient"
        out["reasons"].append("M2 would accept FP8 weight-only as weight-only, but its only reading is the beta "
                              "signature of the sampled layers: no operand or consumption evidence of a weight-only "
                              "GEMM is recorded, and a signature alone never passes; no W8A8 credit either")
        out["reasons"].extend(unknown)
        return out

    fired = int(ev.get("q_layers_fired") or 0)
    found = ev.get("q_layers_found")
    fi_layers = int(fb.get("evidenced_layers", 0)) if fb and fb.get("pinned") else 0
    pos = {"dispatch_fp8_operand_gemm": int(dc.get("fp8_q_shape_calls", 0)),
           "triton_fp8_operand_launch": int((ev.get("triton") or {}).get("fp8_operand_launches", 0)),
           "ncu_fp8_tensor_pipe": 1 if int((ev.get("ncu") or {}).get("fp8_tensor_pipe_inst", 0)) > 0 else 0,
           "flashinfer_pinned_entry": fi_layers if flashinfer == "F1" else 0}
    if consumption == "R1":


        needed = int(st.get("q_modules_expected") or 0)
        names = set((dr or {}).get("evidenced_layer_names") or [])
        cover, unmapped = set(), []
        if dr is not None:
            if "evidenced_asset_modules" in dr:
                cover |= set(dr["evidenced_asset_modules"])
            elif dr.get("evidenced_layers"):
                unmapped.append(f"{dr['evidenced_layers']} layer(s) on the direct channels")
        if flashinfer == "F1" and fb and fb.get("pinned"):
            names |= set(fb.get("evidenced_layer_names") or [])
            if "evidenced_asset_modules" in fb:
                cover |= set(fb["evidenced_asset_modules"])
            elif fi_layers:
                unmapped.append(f"{fi_layers} layer(s) at the pinned FlashInfer entry")
        pos["direct_fp8_gemm_consumed"] = int((dr or {}).get("evidenced_layers", 0))
        unseen = list((dr or {}).get("unseen_layers") or [])
        out["checks"]["positive_evidence"] = dict(
            pos, needed=needed, needed_is="the asset's quantised modules (the frozen Q-set)", covered=len(cover),
            evidenced_layers=len(names), fired=fired, found=found, unseen_layers=len(unseen),
            flashinfer_pinned_entry_recorded=fi_layers, ncu_credited=False,
            counted="asset Q-set modules computed by a consumed FP8 GEMM of their own bytes (R1)")
        complete = needed > 0 and len(cover) >= needed
    else:
        out["checks"]["positive_evidence"] = dict(pos, needed=fired, flashinfer_pinned_entry_recorded=fi_layers,
                                                  counted="calls (R0, historical: it cannot certify)")
        visible = pos["dispatch_fp8_operand_gemm"] + pos["triton_fp8_operand_launch"] + pos["flashinfer_pinned_entry"]
        complete = bool(fired and (visible >= fired or pos["ncu_fp8_tensor_pipe"]))
    if complete:
        if unknown:
            out["verdict"] = "not_evaluated"
            out["reasons"].extend(unknown)
            return out
        out["verdict"] = "pass"
        return out
    if kernel_names == "K1" and pos["dispatch_fp8_operand_gemm"] < max(fired, 1):
        out["verdict"] = "fail"
        out["reasons"].append(f"K1: {pos['dispatch_fp8_operand_gemm']} dispatcher-visible FP8-operand GEMM(s) for "
                              f"{fired} quantised layer(s)")
        return out
    out["verdict"] = "insufficient"
    if consumption == "R1":
        out["reasons"].append(f"store, dispatch and beta hold, but only {len(cover)} of the asset's {needed} quantised "
                              "module(s) are computed by a consumed FP8 GEMM of their own bytes: an unknown or fused "
                              "kernel, or layers the instruments did not see; a close beta alone never passes "
                              "-- gather more evidence")
        if unmapped:
            out["reasons"].append(f"the record names no asset module per evidenced layer ({'; '.join(unmapped)}): "
                                  "its coverage of the asset's "
                                  "Q-set is unknown")
        if isinstance(found, int) and fired < found:
            out["reasons"].append(f"{found - fired} of the {found} float8 layers found never fired in the armed "
                                  "forward: their compute was not observed")
        if unseen:
            out["reasons"].append(f"{len(unseen)} layer(s) made FP8 GEMMs the tap could not trace to their output (it "
                                  f"did not see the layer's input or output tensor): instrument gap, e.g. {unseen[:3]}")
    else:
        out["reasons"].append(f"store, dispatch and beta hold, but only {visible} FP8-operand GEMM(s) are visible for "
                              f"{fired} quantised layer(s): an unknown or fused kernel; a close beta alone never "
                              "passes -- gather more evidence")
    if consumption == "R1" and dr is None and (pos["dispatch_fp8_operand_gemm"] or pos["triton_fp8_operand_launch"]):
        out["reasons"].append(f"the record carries no result-consumption provenance: its "
                              f"{pos['dispatch_fp8_operand_gemm'] + pos['triton_fp8_operand_launch']} "
                              "FP8-operand call(s) earn nothing under R1")
    if fi_layers and flashinfer == "F2":
        out["reasons"].append(f"the pinned FlashInfer entry's record would evidence {fi_layers} layer(s) under F1; "
                              "F2 does not count it")
    out["reasons"].extend(_flashinfer_no_credit_reasons(fb, fired))
    out["reasons"].extend(unknown)
    return out


_ORDER = {"fail": 0, "not_evaluated": 1, "insufficient": 2, "pass": 3}


def fp8_tp_verdict(ev: dict, **policies) -> dict:
    """The TP profile (T4 tp2_x_fp8): fp8_execution_verdict on EVERY rank's evidence -- its store census against the
    rank's own slabs of the asset, its dispatch census at the shard shapes, beta (row-parallel layers on the ranks'
    inputs joined), result consumption with row-parallel layers judged across ranks -- and the profile's verdict is
    the worst rank's: a defect on one rank fails it, whatever the other reads. Reading rank 0 alone is the defect this
    exists against (fp8_tp_rank1_dequant)."""
    tp = int(ev.get("tp") or 0)
    ranks = ev.get("ranks") or []
    out = {"verdict": "not_evaluated", "reasons": [], "tp": tp, "ranks": [], "policy": None}
    if not ev.get("captured") or tp < 2 or len(ranks) != tp:
        if ev.get("capture_started_not_finished"):
            out["verdict"] = "fail"
            out["reasons"].append("the armed DiT forward raised inside the candidate on some rank")
        else:
            out["reasons"].append(ev.get("reason") or f"{len(ranks)} rank record(s) for TP={tp}")
        return out
    per = [fp8_execution_verdict(r, **policies) for r in ranks]
    out["policy"] = per[0]["policy"]
    for i, v in enumerate(per):
        out["ranks"].append({"rank": i, "verdict": v["verdict"], "checks": v["checks"], "reasons": v["reasons"][:4],
                             "execution_type": v["execution_type"]})
        if v["verdict"] != "pass":
            out["reasons"] += [f"rank {i}: {r}" for r in v["reasons"][:3]]
    out["verdict"] = min((v["verdict"] for v in per), key=lambda x: _ORDER[x])
    return out


def fp8_e4_verdict(why: str, control: dict | None, armed_built: bool) -> dict:
    'E4 with its control.'


    c = control or {}
    if c.get("served"):
        return {"verdict": "not_evaluated", "attribution": "instrument", "checks": {"served": "instrument_crash"},
                "reasons": [f"instrument: {why}; the same request on a fresh engine WITHOUT the tap served an image "
                            "-- the tap broke a forward the candidate serves, so this is the instrument's crash, never "
                            "the candidate's result"[:600]]}
    if c.get("built") or (not armed_built and c.get("built") is False):
        return {"verdict": "fail", "attribution": "candidate", "checks": {"served": "fail"},
                "reasons": [f"{why} (the control without the tap did not serve either: {c.get('why')})"[:600]]}
    return {"verdict": "not_evaluated", "attribution": "undetermined", "checks": {"served": "undetermined"},
            "reasons": [f"{why}; the control without the tap could not run ({c.get('why')}), so whose failure it is "
                        "was not measured"[:600]]}


def _flashinfer_no_credit_reasons(fb: dict | None, fired: int) -> list[str]:
    """Why FlashInfer records earned no credit, each cause named separately: an
    instrument fault, an entry that is not the pinned library code, a JIT module not built fresh, calls that did not
    run the verified code, and layers the recorder never saw -- the last is a missing record, not read as a bypass."""
    if not fb or not fb.get("calls"):
        return []
    out = []
    if fb.get("instrument_fault"):
        out.append(f"flashinfer: instrument fault ({fb['instrument_fault']}): no FlashInfer credit, not attributed "
                   "to the candidate")
    elif not fb.get("trusted_at_bind"):
        what = [m.get("file") or m.get("tree") for m in fb.get("trust_mismatch") or []]
        out.append("flashinfer: the entry the process holds is not the pinned library code "
                   f"(files / trees differing: {what or 'none'}; entry code object pinned: "
                   f"{fb.get('entry_code_is_pinned')}): no credit")
    if fb.get("jit_modules_required") and not fb.get("jit_ok"):
        out.append(f"flashinfer: JIT module(s) {fb['jit_modules_required']} not loaded fresh from the verifier's "
                   "workspace: no credit")
    for key, what in (("not_executed_calls", "did not run the verified code object"),
                      ("globals_altered_calls", "ran with the entry's module globals changed since verification"),
                      ("runner_getter_altered_calls", "ran with the runner getter changed since verification")):
        if fb.get(key):
            out.append(f"flashinfer: {fb[key]} call(s) {what}: no credit")
    unread = fb.get("provenance_unread_layers") or []
    if unread:
        out.append(f"flashinfer: {len(unread)} layer(s) whose output could not be read against the call's result by "
                   "value (no sample; or, for a TP row-parallel layer, not every rank's sample): no credit for them, "
                   "and nothing observed to fail them")
    miss = fb.get("unrecorded_fired_layers")
    if miss:
        out.append(f"flashinfer: {miss} of {fired} quantised layer(s) made no call the recorder saw: no credit for "
                   "them (a missing record is not read as a bypass)")
    return out


def gate_details(capability: str, layer: str, verdict: dict, policy: dict, **extra) -> dict:
    'gate details.'
    v = verdict.get("verdict")
    d = {"capability": capability, "layer": layer, "diagnostic": True,
         "diagnostic_why": "not calibrated: needs an oracle pass, a legal-variant pass and a targeted-negative "
                           "failure, each with evidence that the instrument ran",
         "verdict": v, "reasons": verdict.get("reasons", []), "policy": policy,
         "not_evaluated": v == "not_evaluated", "insufficient": v == "insufficient"}
    d.update(extra)
    return d
