#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'Typed Q/F/E sub-verdicts, emitted at the source by the seven feature gates (sd3-qfe-v1).'
from __future__ import annotations

SCHEMA = "sd3-qfe-typed/1"
VERSION = "sd3-qfe-v1"
STATES = ("pass", "fail", "not_evaluated")
PAIR_STATES = ("both_pass", "deployment_only_fail", "base_only_fail", "both_fail", "not_evaluated")
NUMERICAL_ORIGINAL = "complete combined gate passed; numerical credit uses that combined verdict"
NUMERICAL_MISSING = ("no common-input independent numerical proof or validated paired-preservation criterion; "
                     "structure and paired drift are reported, not credited")
BINDING_KEYS = ("prompt", "negative_prompt", "seed", "guidance_scale", "num_inference_steps", "height", "width",
                "num_outputs_per_prompt")


def state(value) -> str:
    'state.'
    if value in STATES:
        return value
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "not_evaluated"


def row(check, role: str, key, index=None) -> dict:
    """One comparison row from a `reference_contract.image_check` result."""
    check = check if isinstance(check, dict) else {}
    fixture = check.get("fixture") if isinstance(check.get("fixture"), dict) else {}
    return {"role": role, "key": key, "index": index, "fixture": fixture.get("id"), "profile": check.get("profile"),
            "verdict": state(check.get("verdict")),
            "metrics": check.get("metrics") if isinstance(check.get("metrics"), dict) else {},
            "bars": check.get("bars") if isinstance(check.get("bars"), dict) else {}}


def output_rows(check, role: str, key, n: int | None = None) -> list:
    """Rows for a `reference_contract.check_outputs` result: one per expected output. A check without per-output
    results (missing spec/reference, wrong count) yields n rows (or one) carrying the check's own verdict."""
    check = check if isinstance(check, dict) else {}
    per = check.get("per") if isinstance(check.get("per"), list) else []
    if per:
        return [row(item, role, key, i) for i, item in enumerate(per)]
    count = n if isinstance(n, int) and n > 0 else 1
    return [{"role": role, "key": key, "index": i, "fixture": None, "profile": None,
             "verdict": state(check.get("verdict")), "metrics": {}, "bars": {}} for i in range(count)]


def binding(spec, index, profile, run_ids=None) -> dict:
    spec = spec if isinstance(spec, dict) else {}
    return {**{k: spec.get(k) for k in BINDING_KEYS}, "output_index": index, "profile": profile,
            "run_ids": dict(run_ids or {})}


def pair_status(deployment: str, anchor: str | None) -> str:
    if deployment not in ("pass", "fail") or anchor not in ("pass", "fail"):
        return "not_evaluated"
    return {("pass", "pass"): "both_pass", ("fail", "pass"): "deployment_only_fail",
            ("pass", "fail"): "base_only_fail", ("fail", "fail"): "both_fail"}[(deployment, anchor)]


def pair(deployment_row: dict, anchor_row: dict | None, drift_db, bind: dict) -> dict:
    """A deployment comparison next to its matched same-run anchor. `anchor_row` None: the original code measures no
    anchor for this output, so attribution is not_evaluated (no new reference check is added for attribution)."""
    anchor = anchor_row["verdict"] if anchor_row else None
    return {"role": deployment_row["role"], "key": deployment_row["key"], "index": deployment_row["index"],
            "deployment": deployment_row["verdict"], "anchor": anchor if anchor_row else "not_measured",
            "status": pair_status(deployment_row["verdict"], anchor),
            "paired_drift_psnr_db": drift_db, "binding": bind}


def obligation(role: str, key, legal: list, outcome: str, satisfied: str, n: int | None = None) -> dict:
    return {"role": role, "key": key, "outputs": n, "legal_outcomes": list(legal), "outcome": outcome,
            "satisfied": state(satisfied)}


def numerical(legacy_verdict: str | None) -> dict:
    if legacy_verdict == "pass":
        return {"status": "pass", "basis": NUMERICAL_ORIGINAL}
    return {"status": "not_evaluated", "basis": NUMERICAL_MISSING}


def feature_status(structural: dict, num: dict) -> str:
    values = [state(v) for v in structural.values()]
    if not values:
        return "not_evaluated"
    if "fail" in values:
        return "fail"
    if all(v == "pass" for v in values) and (num or {}).get("status") == "pass":
        return "pass"
    return "not_evaluated"


def e_ready(obligations: list, comparisons: list) -> str:
    """E for one gate: every obligation satisfied and every required comparison passing."""
    states = [o["satisfied"] for o in obligations] + [c["verdict"] for c in comparisons]
    if not obligations:
        return "not_evaluated"
    if "fail" in states:
        return "fail"
    return "pass" if all(s == "pass" for s in states) else "not_evaluated"


def block(structural: dict, comparisons: list, obligations: list, pairs: list, legacy_verdict: str | None,
          readouts: dict | None = None) -> dict:
    structural = {k: state(v) for k, v in structural.items()}
    num = numerical(legacy_verdict)
    return {"schema": SCHEMA, "version": VERSION, "structural": structural, "numerical": num,
            "comparisons": list(comparisons), "obligations": list(obligations), "pairs": list(pairs),
            "e": e_ready(obligations, comparisons), "readouts": dict(readouts or {}),
            "status": feature_status(structural, num)}


def add_structural(qfe: dict, name: str, value, legacy_verdict: str | None = None) -> dict:
    """Add a structural requirement and recompute the feature and combined verdicts."""
    if not isinstance(qfe, dict):
        return qfe
    qfe.setdefault("structural", {})[name] = state(value)
    if legacy_verdict is not None:
        qfe["numerical"] = numerical(legacy_verdict)
    qfe["status"] = feature_status(qfe["structural"], qfe.get("numerical") or {})
    return qfe
