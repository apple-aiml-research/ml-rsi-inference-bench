#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'SD3 clip gates utilities.'


from __future__ import annotations

import math
import os

ENCODERS = ("text_encoder", "text_encoder_2")


_COEF = float(os.environ.get("SD3_ALIGN_REL_L2_COEF", "1.5"))
_ANCHOR = {
    ("text_encoder", "hidden_m2"): float(os.environ.get("SD3_ALIGN_LEGAL_CLIP_L_HIDDEN", "0.002815689575877673")),
    ("text_encoder", "pooled"): float(os.environ.get("SD3_ALIGN_LEGAL_CLIP_L_POOLED", "0.008176863242576545")),
    ("text_encoder_2", "hidden_m2"): float(os.environ.get("SD3_ALIGN_LEGAL_CLIP_G_HIDDEN", "0.01757848938436147")),
    ("text_encoder_2", "pooled"): float(os.environ.get("SD3_ALIGN_LEGAL_CLIP_G_POOLED", "0.009419541681727173")),
}
POLICY = {


    "SHARD_MAX_FRAC": 0.75,
    **{f"REF_REL_L2[{e}.{k}]": a * _COEF for (e, k), a in _ANCHOR.items()},
}
POLICY_STATE = "specified"
POLICY_BASIS = {'SHARD_MAX_FRAC': 'Required bound for SHARD_MAX_FRAC.', 'REF_REL_L2': 'Required bound for REF_REL_L2.'}
_KINDS = (("hidden_m2", "penultimate hidden state"), ("pooled", "projected pooled vector"))


def _finite_pos(x) -> bool:
    """A census count the judge may divide by or into: a finite number > 0 (a NaN, an infinity, zero or a negative count
    is a broken instrument reading, never evidence)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def _finite_nonneg(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v >= 0


def parse(events: list) -> dict:
    """Recorder lines -> {rank: {encoder: {layout, compute, capture, calls}}} (the client process carries no lines)."""
    out: dict = {}
    for e in events or []:
        ev = e.get("event")
        if ev not in ("clip_layout", "clip_compute", "clip_capture", "clip_call"):
            continue
        r, enc = e.get("rank"), e.get("encoder")
        if r is None or enc not in ENCODERS:
            continue
        d = out.setdefault(int(r), {}).setdefault(enc, {"layout": None, "compute": None, "capture": None, "calls": []})
        if ev == "clip_layout":
            d["layout"] = {"cls": e.get("cls"), "dims": e.get("dims"), "layer0": e.get("layer0"),
                           "scope": e.get("scope")}
        elif ev == "clip_compute":
            d["compute"] = {"macs": e.get("macs"), "n_calls": e.get("n_calls"), "truncated": e.get("truncated")}
        elif ev == "clip_capture":
            d["capture"] = {"file": e.get("file"), "shapes": e.get("shapes"), "dtypes": e.get("dtypes")}
        else:
            d["calls"].append({"n": e.get("n"), "first": e.get("first"), "gc": e.get("gc") or {},
                               "dist": e.get("dist") or {}, "ids": e.get("ids")})
    return out


def _reductions(calls: list) -> list:
    """Per forward: the larger of the two levels' all-reduce counts (B0's GroupCoordinator.all_reduce may or may not go
    through torch.distributed.all_reduce; a raw ProcessGroup goes through torch.distributed only)."""
    return [max(int(c["gc"].get("all_reduce", 0)), int(c["dist"].get("all_reduce", 0))) for c in calls]


def clip_encoder_parallel_is_real(events: list, reference: dict | None, ranks=(0, 1), extra: dict | None = None) -> dict:
    """`events`: the recorder lines of the SP engine; `reference`: the isolated child's result
    {encoder: {"dense_macs": int, "ranks": {"0": {"hidden_m2": {"rel_l2": ..}, "pooled": {..}}, ...}}, "error": ..}."""
    P = POLICY
    d: dict = {"policy_state": POLICY_STATE, **(extra or {})}
    per = parse(events)
    ref = reference or {}
    d["ranks_seen"] = sorted(per)
    missing = []
    for r in ranks:
        for enc in ENCODERS:
            x = (per.get(r) or {}).get(enc) or {}
            if not x.get("compute") or not _finite_pos(x.get("compute", {}).get("macs")):
                missing.append(f"rank {r} {enc}: no usable compute census (missing, zero, negative or non-finite)")
            if not x.get("capture"):
                missing.append(f"rank {r} {enc}: no output capture")
    for enc in ENCODERS:
        er = ref.get(enc) or {}
        if not _finite_pos(er.get("dense_macs")):
            missing.append(f"reference {enc}: no usable dense MACs ({er.get('error') or ref.get('error') or 'not run'})")
        for r in ranks:
            if not ((er.get("ranks") or {}).get(str(r)) or {}).get("hidden_m2"):
                missing.append(f"reference {enc}: rank {r}'s output not compared")

    d["collectives"] = {str(r): {enc: {"all_reduce_per_forward": _reductions(((per.get(r) or {}).get(enc) or {})
                                                                               .get("calls", [])),
                                       "first_forward": next((c for c in ((per.get(r) or {}).get(enc) or {})
                                                              .get("calls", []) if c.get("first")), None)}
                                 for enc in ENCODERS} for r in ranks}
    d["layout"] = {str(r): {enc: ((per.get(r) or {}).get(enc) or {}).get("layout") for enc in ENCODERS} for r in ranks}
    d["cross_rank"] = {enc: (ref.get(enc) or {}).get("cross_rank") for enc in ENCODERS}
    if missing:
        return {"verdict": "not_evaluated", "reasons": ["recorded: " + "; ".join(missing)], "details": d}
    clauses, reasons = {}, []
    frac = {}
    for r in ranks:
        for enc in ENCODERS:
            frac[f"{r}.{enc}"] = round(float(per[r][enc]["compute"]["macs"]) / float(ref[enc]["dense_macs"]), 4)
    d["executed_frac"] = frac
    over = {k: v for k, v in frac.items() if v > P["SHARD_MAX_FRAC"]}
    clauses["sharded"] = not over
    if over:
        reasons.append(f"sharded: executed MACs over the dense forward's {over} (> {P['SHARD_MAX_FRAC']}): the encoder "
                       "ran whole on that GPU (replicated, or its weights gathered back for compute)")
    devs, bad, unread = {}, [], []
    for enc in ENCODERS:
        for r in ranks:
            got = ref[enc]["ranks"][str(r)]
            for kind, label in _KINDS:
                m = got.get(kind) or {}
                v = m.get("rel_l2")
                bar = P[f"REF_REL_L2[{enc}.{kind}]"]
                devs[f"{r}.{enc}.{kind}"] = v


                if v is None:
                    unread.append(f"rank {r} {enc} {label}: no reading")
                elif m.get("ref_finite") is False:
                    unread.append(f"rank {r} {enc} {label}: the fp32 reference is non-finite (instrument)")
                elif m.get("cand_finite") is False:
                    bad.append(f"rank {r} {enc} {label}: the candidate's output is non-finite")
                elif not _finite_nonneg(v):
                    unread.append(f"rank {r} {enc} {label}: deviation {v!r} is not a finite reading")
                elif float(v) > bar:
                    bad.append(f"rank {r} {enc} {label} rel_l2 {float(v):.3g} > {bar:.3g}")
    d["reference_rel_l2"] = devs
    if not bad and unread:
        clauses["reference"] = None
        reasons.append("reference: not_evaluated (" + "; ".join(unread) + ")")
    else:
        clauses["reference"] = not bad
        if bad:
            reasons.append("reference: " + "; ".join(bad + unread) + " -- the encoder's output is not the fp32 "
                           "reference's on the same ids (a share lost or mis-combined)")
    d["clauses"] = clauses
    if any(v is False for v in clauses.values()):
        return {"verdict": "fail", "reasons": reasons, "details": d}
    if any(v is None for v in clauses.values()):
        return {"verdict": "not_evaluated", "reasons": reasons, "details": d}
    return {"verdict": "pass", "reasons": reasons or ["every clause held"], "details": d}
