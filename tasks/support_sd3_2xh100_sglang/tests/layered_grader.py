#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Layered side record of one verifier run: layered_score.json, next to reward.json."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

SCHEMA = "infb-tstar-layered-score/3"
MAP_SCHEMA = "infb-tstar-capability-map/3"
GRADER_VERSION = "3"
OUTPUT_NAME = "layered_score.json"
MAP_NAME = "capabilities.json"
GATE_SUFFIX = ".gate.json"
LAYERS = ("T1", "T2", "T3", "T4")
STATUSES = ("certified", "diagnostic", "planned")
MERGE_RULES = ("all_of",)


MAPPING_STATUSES = ("pre_day0", "day0")

FORGERY_KINDS = ("hidden_reference_read", "scoring_path_edit", "planted_score_file")


UNMEASURED_MARKERS = ("not_evaluated", "skipped", "uncalibrated", "verifier_defect", "infra_void")


ID_RE = re.compile(r"^(?:[a-z][a-z0-9_]*|C[1-4]\.[1-9][0-9]*)$")
LEDGER_ID_RE = re.compile(r"^C([1-4])\.[1-9][0-9]*$")
CTRF_NAME_RE = re.compile(r"^\[(f2p|p2p)\] (.+)$")
MESSAGE_CHARS = 500

MAP_KEYS = frozenset({"schema_version", "task", "map_version", "mapping_status", "score_profile",
                      "layers_published_to_agent", "ledger_source", "note", "capabilities", "layer_conditions",
                      "integrity", "unmapped"})
CAPABILITY_KEYS = frozenset({"id", "title", "layer", "status", "evidence", "merge", "requires", "report_only",
                             "certified_by", "why"})
CONDITION_KEYS = frozenset({"node_id", "layer", "capability", "ledger_id", "why"})
INTEGRITY_KEYS = frozenset({"id", "title", "detects", "evidence", "why"})
UNMAPPED_KEYS = frozenset({"node_id", "reason"})
PROVENANCE_KEYS = ("task", "score_profile", "map_version", "mapping_status", "layers_published_to_agent",
                   "ledger_source")


def log(msg: str) -> None:
    print(f"[layered] {msg}", flush=True)


def load_ids(cfg: dict, key: str) -> list[str]:
    """A whitelist exactly as grader.py reads it: stripped, empties dropped, first occurrence kept."""
    ids: list[str] = []
    seen: set[str] = set()
    for line in cfg.get(key) or []:
        s = str(line).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        ids.append(s)
    return ids


def _nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def counted(cap: dict) -> bool:
    """Whether a capability counts toward its layer: every capability does, except a report-only one."""
    return not cap.get("report_only", False)


def validate_map(cmap: object, cfg: dict) -> list[str]:
    """Every reason `cmap` cannot layer runs graded under `cfg`. [] means valid.

    The rule that matters most: every scored id (f2p and p2p) is placed EXACTLY ONCE -- in a certified capability's
    evidence, a layer condition, an integrity rule, or `unmapped` with a reason. So a new f2p id with no capability is
    an error here. The status rules keep a
    capability honest about what backs it: certified <=> scored evidence and a certification record, diagnostic <=>
    unscored (diagnostic) gates, planned <=> no gate at all.
    """
    if not isinstance(cmap, dict):
        return ["the capability map is not a JSON object"]
    errors: list[str] = []
    if cmap.get("schema_version") != MAP_SCHEMA:
        errors.append(f"schema_version is {cmap.get('schema_version')!r}, expected {MAP_SCHEMA!r}")
    unknown = sorted(set(cmap) - MAP_KEYS)
    if unknown:
        errors.append(f"unknown top-level key(s) {unknown} -- a misspelt section would be silently ignored")
    for key in ("task", "map_version", "score_profile"):
        if not _nonempty_str(cmap.get(key)):
            errors.append(f"{key!r} must be a non-empty string")
    if cmap.get("mapping_status") not in MAPPING_STATUSES:
        errors.append(f"'mapping_status' is {cmap.get('mapping_status')!r}, not one of {list(MAPPING_STATUSES)}")
    published = cmap.get("layers_published_to_agent")
    if not isinstance(published, bool):
        errors.append("'layers_published_to_agent' must be true or false: a reader must know whether the agent "
                      "was told the target, the scoring scope and the statuses")
    elif cmap.get("mapping_status") == "day0" and not published:
        errors.append("a day0 map sets 'layers_published_to_agent' true: a day-0 version publishes its target, its "
                      "scoring scope and every capability's status, uncertified ones included")
    if "ledger_source" in cmap and not _nonempty_str(cmap.get("ledger_source")):
        errors.append("'ledger_source', when given, must name the day-0 ledger the map mirrors")

    f2p = load_ids(cfg, "f2p_node_ids")
    p2p = load_ids(cfg, "p2p_node_ids")
    f2p_set, p2p_set = set(f2p), set(p2p)
    diagnostic = set(load_ids(cfg, "diagnostic_node_ids"))
    for nid in sorted(f2p_set & p2p_set):
        errors.append(f"{nid!r} is in both f2p_node_ids and p2p_node_ids")
    placements: dict[str, list[str]] = {}
    diagnostic_uses: dict[str, list[str]] = {}

    def describe(nid: str) -> str:
        if nid in f2p_set:
            return "a scored f2p id"
        if nid in p2p_set:
            return "a scored p2p id"
        if nid in diagnostic:
            return "a DIAGNOSTIC id config.json does not score"
        return "an id config.json does not have"

    def place(nid: object, allowed: set[str], wanted: str, where: str) -> None:
        if not _nonempty_str(nid):
            errors.append(f"{where}: a node id must be a non-empty string, got {nid!r}")
            return
        nid = str(nid)
        placements.setdefault(nid, []).append(where)
        if nid not in allowed:
            errors.append(f"{where} names {nid!r}, {describe(nid)}; only {wanted} ids belong there")

    def entries(key: str) -> list:
        value = cmap.get(key, [])
        if not isinstance(value, list):
            errors.append(f"{key!r} must be a list")
            return []
        bad = [i for i, e in enumerate(value) if not isinstance(e, dict)]
        if bad:
            errors.append(f"{key!r} entries {bad} are not objects")
        return [e for e in value if isinstance(e, dict)]

    def check_keys(entry: dict, allowed: frozenset, where: str) -> None:
        extra = sorted(set(entry) - allowed)
        if extra:
            errors.append(f"{where}: unknown key(s) {extra}")

    def evidence_list(entry: dict, where: str, *, required: bool) -> list:
        evidence = entry.get("evidence", [])
        if not isinstance(evidence, list):
            errors.append(f"{where}: 'evidence' must be a list of node ids")
            return []
        if required and not evidence:
            errors.append(f"{where}: 'evidence' must be a non-empty list of node ids")
        dups = sorted({str(n) for n in evidence if evidence.count(n) > 1})
        if dups:
            errors.append(f"{where}: evidence lists {dups} more than once")
        return list(dict.fromkeys(evidence))

    def check_id(value: object, where: str) -> bool:
        if isinstance(value, str) and ID_RE.match(value):
            return True
        errors.append(f"{where}: 'id' must be a ledger id C<layer>.<n> or snake_case, got {value!r}")
        return False

    capabilities = entries("capabilities")
    if not capabilities:
        errors.append("'capabilities' must be a non-empty list")
    by_id: dict[str, dict] = {}
    for i, cap in enumerate(capabilities):
        cid = cap.get("id")
        where = f"capability {cid!r}" if _nonempty_str(cid) else f"capabilities[{i}]"
        check_keys(cap, CAPABILITY_KEYS, where)
        if check_id(cid, where):
            if cid in by_id:
                errors.append(f"{where}: duplicate capability id")
            else:
                by_id[cid] = cap
        layer = cap.get("layer")
        if layer not in LAYERS:
            errors.append(f"{where}: 'layer' is {layer!r}, not one of {list(LAYERS)}")
        elif isinstance(cid, str) and LEDGER_ID_RE.match(cid) and f"T{LEDGER_ID_RE.match(cid).group(1)}" != layer:
            errors.append(f"{where}: a ledger id names its layer, and {cid} is not on {layer}")
        if not _nonempty_str(cap.get("title")):
            errors.append(f"{where}: 'title' must name the capability")
        if cap.get("merge", "all_of") not in MERGE_RULES:
            errors.append(f"{where}: 'merge' is {cap.get('merge')!r}; this grader implements only {list(MERGE_RULES)}")
        if not isinstance(cap.get("requires", []), list):
            errors.append(f"{where}: 'requires' must be a list of capability ids")
        if not isinstance(cap.get("report_only", False), bool):
            errors.append(f"{where}: 'report_only' must be true or false")
        if not _nonempty_str(cap.get("why")):
            errors.append(f"{where}: 'why' must say why this layer")
        status = cap.get("status")
        certified_by = cap.get("certified_by")
        if status not in STATUSES:
            errors.append(f"{where}: 'status' is {status!r}, not one of {list(STATUSES)}")
            continue
        if status == "certified":
            evidence = evidence_list(cap, where, required=True)
            for nid in evidence:
                place(nid, f2p_set, "scored f2p", where)
            if not (isinstance(certified_by, list) and certified_by and all(_nonempty_str(c) for c in certified_by)):
                errors.append(f"{where}: a certified capability names what certified it ('certified_by': the oracle, "
                              f"legal-variant and targeted-negative runs)")
            if cap.get("report_only"):
                errors.append(f"{where}: a report-only capability has no scored evidence; it is diagnostic or planned")
        else:
            if certified_by is not None:
                errors.append(f"{where}: 'certified_by' on a {status} capability -- only a certified one has a "
                              f"certification record")
            evidence = evidence_list(cap, where, required=(status == "diagnostic"))
            if status == "planned" and evidence:
                errors.append(f"{where}: a planned capability has no gate yet, so no evidence; a gate that exists "
                              f"makes it diagnostic")
            for nid in evidence if status == "diagnostic" else []:
                if not _nonempty_str(nid):
                    errors.append(f"{where}: a node id must be a non-empty string, got {nid!r}")
                    continue
                diagnostic_uses.setdefault(str(nid), []).append(where)
                if nid not in diagnostic:
                    errors.append(f"{where} names {nid!r}, {describe(str(nid))}; a diagnostic capability reads only "
                                  f"diagnostic_node_ids (a scored gate belongs to a certified capability)")

    for nid, where in sorted(diagnostic_uses.items()):
        if len(where) > 1:
            errors.append(f"diagnostic id {nid!r} evidences {len(where)} capabilities ({'; '.join(where)}); one")

    edges: dict[str, list[str]] = {}
    for cid, cap in by_id.items():
        requires = cap.get("requires", [])
        edges[cid] = []
        if not isinstance(requires, list):
            continue
        for rid in requires:
            if rid == cid:
                errors.append(f"capability {cid!r} requires itself")
            elif rid not in by_id:
                errors.append(f"capability {cid!r} requires {rid!r}, which is not a capability of this map")
            else:
                edges[cid].append(rid)
                if by_id[rid].get("report_only"):
                    errors.append(f"capability {cid!r} requires the report-only {rid!r}, which is never credited")
                if (cap.get("layer") in LAYERS and by_id[rid].get("layer") in LAYERS
                        and LAYERS.index(by_id[rid]["layer"]) > LAYERS.index(cap["layer"])):
                    errors.append(f"capability {cid!r} ({cap['layer']}) requires {rid!r} of the HIGHER layer "
                                  f"{by_id[rid]['layer']}; a prerequisite must sit on the same or a lower layer")


    done: set[str] = set()
    stack: list[str] = []

    def visit(cid: str) -> None:
        stack.append(cid)
        for rid in edges[cid]:
            if rid in stack:
                errors.append(f"'requires' has a cycle: {' -> '.join(stack[stack.index(rid):] + [rid])}")
            elif rid not in done:
                visit(rid)
        stack.pop()
        done.add(cid)

    for cid in by_id:
        if cid not in done:
            visit(cid)


    condition_ledger_ids: dict[str, set[str]] = {}
    for i, cond in enumerate(entries("layer_conditions")):
        where = f"layer condition {cond.get('node_id')!r}" if _nonempty_str(cond.get("node_id")) \
            else f"layer_conditions[{i}]"
        check_keys(cond, CONDITION_KEYS, where)
        layer = cond.get("layer")
        if layer not in LAYERS:
            errors.append(f"{where}: 'layer' is {layer!r}, not one of {list(LAYERS)}")
        protected = cond.get("capability")
        if not _nonempty_str(protected):
            errors.append(f"{where}: 'capability' must name the certified capability this p2p protects -- a p2p "
                          f"conditions the capability it protects, on that capability's layer")
        else:
            target = by_id.get(protected)
            if target is None:
                errors.append(f"{where}: 'capability' {protected!r} is not a capability of this map")
            elif target.get("status") != "certified" or target.get("report_only"):
                errors.append(f"{where}: protects {protected!r}, which is not a certified, counted capability -- a "
                              f"scored p2p must condition something that can be credited (else it is `unmapped`)")
            elif target.get("layer") != layer:
                errors.append(f"{where}: protects {protected!r} on {target.get('layer')}, but says {layer}; a "
                              f"condition sits on the layer of the capability it protects")
        if "ledger_id" in cond:
            lid = cond.get("ledger_id")
            if not (isinstance(lid, str) and ID_RE.match(lid)):
                errors.append(f"{where}: 'ledger_id' must be the day-0 ledger item's id (C<layer>.<n> or "
                              f"snake_case), got {lid!r}")
            else:
                m = LEDGER_ID_RE.match(lid)
                if m and layer in LAYERS and f"T{m.group(1)}" != layer:
                    errors.append(f"{where}: ledger id {lid} names T{m.group(1)}, but the condition sits on {layer}; "
                                  f"the ledger item must use the layer of the capability it protects")
                condition_ledger_ids.setdefault(lid, set()).add(str(layer))
        if not _nonempty_str(cond.get("why")):
            errors.append(f"{where}: 'why' must say what this p2p protects in that capability")
        place(cond.get("node_id"), p2p_set, "scored p2p", where)


    rule_ids: set[str] = set()
    for i, rule in enumerate(entries("integrity")):
        rid = rule.get("id")
        where = f"integrity rule {rid!r}" if _nonempty_str(rid) else f"integrity[{i}]"
        check_keys(rule, INTEGRITY_KEYS, where)
        if check_id(rid, where):
            if rid in rule_ids or rid in by_id:
                errors.append(f"{where}: id already used by another rule or capability")
            else:
                rule_ids.add(rid)
        if not _nonempty_str(rule.get("title")):
            errors.append(f"{where}: 'title' must name the rule")
        if not _nonempty_str(rule.get("why")):
            errors.append(f"{where}: 'why' must say what the rule protects")
        detects = rule.get("detects")
        if not (isinstance(detects, list) and detects):
            errors.append(f"{where}: 'detects' must list the scoring forgery it detects, from {list(FORGERY_KINDS)}: "
                          f"integrity is scoring forgery only, and a gate that detects none of them is "
                          f"capability evidence")
        else:
            outside = [d for d in detects if d not in FORGERY_KINDS]
            if outside:
                errors.append(f"{where}: 'detects' names {outside}, which is not a scoring forgery "
                              f"{list(FORGERY_KINDS)}; a native or wrapper check is T1 capability evidence")
            if len({json.dumps(d) for d in detects}) != len(detects):
                errors.append(f"{where}: 'detects' lists a kind twice")
        for nid in evidence_list(rule, where, required=True):
            place(nid, f2p_set, "scored f2p", where)

    for lid, layers_of in sorted(condition_ledger_ids.items()):
        if lid in by_id or lid in rule_ids:
            errors.append(f"ledger id {lid!r} is a condition's ledger_id and also a capability or integrity id; one "
                          f"ledger item has one place in the map")
        if len(layers_of) > 1:
            errors.append(f"the conditions of ledger item {lid!r} sit on {sorted(layers_of)}; one item, one layer")

    for i, item in enumerate(entries("unmapped")):
        where = f"unmapped {item.get('node_id')!r}" if _nonempty_str(item.get("node_id")) else f"unmapped[{i}]"
        check_keys(item, UNMAPPED_KEYS, where)
        if not _nonempty_str(item.get("reason")):
            errors.append(f"{where}: an unmapped scored id needs a 'reason'")
        place(item.get("node_id"), f2p_set | p2p_set, "scored (f2p or p2p)", where)

    for nid in f2p + p2p:
        where = placements.get(nid, [])
        if not where:
            bucket = "f2p" if nid in f2p_set else "p2p"
            errors.append(f"scored {bucket} id {nid!r} is placed nowhere: put it in a certified capability's "
                          f"evidence, a layer condition (p2p), an integrity rule, or `unmapped` with a reason")
        elif len(where) > 1:
            errors.append(f"scored id {nid!r} is placed {len(where)} times ({'; '.join(where)}); exactly once")
    if capabilities and "T1" not in {c.get("layer") for c in capabilities if counted(c)}:
        errors.append("no T1 capability: highest_continuous_layer could never leave 0")
    return errors


_RANK = {"passed": 0, "skipped": 1}


def _worse(a: str, b: str) -> str:
    return a if _RANK.get(a, 2) >= _RANK.get(b, 2) else b


def node_verdicts(ctrf: dict | None, f2p: list[str], p2p: list[str],
                  apply_failed: bool) -> dict[tuple[str, str], tuple[str, str]]:
    """(bucket, id) -> (status, message), from grader.py's ctrf.json. An id it does not carry is failed."""
    if apply_failed:
        why = "model.patch did not apply (reward.json apply_failed=1); nothing was graded"
        return {(b, n): ("failed", why) for b, ids in (("f2p", f2p), ("p2p", p2p)) for n in ids}
    seen: dict[tuple[str, str], tuple[str, str]] = {}
    tests = ((ctrf or {}).get("results") or {}).get("tests") or []
    for tc in tests if isinstance(tests, list) else []:
        if not isinstance(tc, dict):
            continue
        m = CTRF_NAME_RE.match(str(tc.get("name") or "").strip())
        if not m:
            continue
        key = (m.group(1), m.group(2).strip())
        status = str(tc.get("status") or "").strip().lower() or "failed"
        message = str(tc.get("message") or "").strip()
        if key in seen:
            status = _worse(seen[key][0], status)
            message = seen[key][1] or message
        seen[key] = (status, message if status != "passed" else "")
    out = {}
    for bucket, ids in (("f2p", f2p), ("p2p", p2p)):
        for nid in ids:
            out[(bucket, nid)] = seen.get((bucket, nid), ("failed", "absent from ctrf.json"))
    return out


def gate_records(verifier_dir: Path) -> dict[str, list[dict]]:
    """gate id -> the `details` of every readable gate file naming it, named the way gates_to_ctrf.py names a node."""
    found: dict[str, list[dict]] = {}
    gdir = verifier_dir / "gates"
    if not gdir.is_dir():
        return found
    for path in sorted(gdir.glob("*" + GATE_SUFFIX)):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        name = str(doc.get("gate") or path.name[: -len(GATE_SUFFIX)]).strip()
        if name:
            details = doc.get("details")
            found.setdefault(name, []).append(details if isinstance(details, dict) else {})
    return found


def _unmeasured(details: dict) -> str | None:
    """What a gate's own record says about a PASS that measured nothing, or None when it says nothing of the kind."""
    marks = (["measured=false"] if details.get("measured") is False else []) + [
        key for key in UNMEASURED_MARKERS if details.get(key)]
    if not marks:
        return None
    why = next((details[k].strip() for k in ("skipped", "why") if _nonempty_str(details.get(k))), "")
    return f"not evaluated ({', '.join(marks)})" + (f": {why}" if why else "")


def _pass_unmeasured(nid: str, gates: dict[str, list[dict]]) -> tuple[str | None, object]:
    """(why this PASS is no measurement, the gate's charged_to) -- (None, ...) when its one record shows a measurement.

    Fail closed: a pass is a measurement only when exactly one gate record names the gate and it carries no
    could-not-evaluate marker. No record, or two, and nothing shows what was measured.
    """
    records = gates.get(nid, [])
    if not records:
        return "not evaluated: no gate file names this gate, so nothing shows it measured anything", None
    if len(records) > 1:
        return f"not evaluated: {len(records)} gate files name this gate, so no one record says what it measured", None
    return _unmeasured(records[0]), records[0].get("charged_to")


def _row(nid: str, verdict: tuple[str, str], gates: dict[str, list[dict]], failed_scored: set[str]) -> dict:
    """One scored node as this record reads it: grader.py's verdict, except that a PASS whose gate measured nothing
    is `not_evaluated`. Such a row is `charged` only when its
    charged_to names ANOTHER scored gate that did not pass in this trial; either way it credits nothing."""
    status, message = verdict
    if status != "passed":
        row = {"node_id": nid, "status": status}
        if message:
            row["message"] = message[:MESSAGE_CHARS]
        return row
    why, charged_to = _pass_unmeasured(nid, gates)
    if why is None:
        return {"node_id": nid, "status": "passed"}
    row = {"node_id": nid, "status": "not_evaluated", "grader_status": "passed", "message": why[:MESSAGE_CHARS]}
    if _nonempty_str(charged_to):
        row["charged_to"] = charged_to.strip()
    row["charged"] = row.get("charged_to") in failed_scored and row.get("charged_to") != nid
    return row


def diagnostic_rows(cfg: dict, verifier_dir: Path, ids: list[str], apply_failed: bool,
                    gates: dict[str, list[dict]]) -> dict[str, dict]:
    """id -> a DIAGNOSTIC gate's reading: passed / failed / skipped / absent / unread / not_evaluated.

    grader.py scores only the whitelists, so an unscored gate's status is taken from the report it read (the same
    worst-status-wins rule; report names relative to the verifier dir, as grader.py runs there), and a PASS its own
    gate record marks as unmeasured reads `not_evaluated`, as a scored one does. Informational: never credits.
    """
    if not ids:
        return {}
    grade_cfg = cfg.get("grade") or {}
    if apply_failed or grade_cfg.get("format", "ctrf") != "ctrf":
        return {nid: {"node_id": nid, "status": "unread"} for nid in ids}
    seen: dict[str, str] = {}
    for rep in grade_cfg.get("reports") or []:
        try:
            doc = json.loads((verifier_dir / str(rep)).read_text(encoding="utf-8"))
            tests = (doc.get("results") or {}).get("tests") or []
        except (OSError, ValueError, AttributeError):
            continue
        for tc in tests if isinstance(tests, list) else []:
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("name") or "").strip()
            raw = str(tc.get("status") or "").strip().lower()
            status = "passed" if raw == "passed" else ("skipped" if raw in ("skipped", "pending", "other") else "failed")
            seen[name] = _worse(seen[name], status) if name in seen else status
    rows: dict[str, dict] = {}
    for nid in ids:
        rows[nid] = {"node_id": nid, "status": seen.get(nid, "absent")}
        if rows[nid]["status"] == "passed":
            why, _charged_to = _pass_unmeasured(nid, gates)
            if why is not None:
                rows[nid] = {"node_id": nid, "status": "not_evaluated", "message": why[:MESSAGE_CHARS]}
    return rows


def _result(own: list[dict], requires_met: bool) -> str:
    """A certified capability's outcome in this run: `credited`, or why not. Keep a failure (some own gate or
    condition did not pass) apart from a missing measurement (none failed, some measured nothing)."""
    statuses = {r["status"] for r in own}
    if statuses - {"passed", "not_evaluated"}:
        return "failed"
    if "not_evaluated" in statuses:
        return "not_measured"
    return "credited" if requires_met else "blocked"


def grade(cmap: dict, cfg: dict, reward: dict, ctrf: dict | None, diagnostics: dict[str, dict],
          gates: dict[str, list[dict]]) -> dict:
    """The layered body for a VALID map (validate_map(cmap, cfg) == []) and a published reward."""
    f2p, p2p = load_ids(cfg, "f2p_node_ids"), load_ids(cfg, "p2p_node_ids")
    apply_failed = bool(reward.get("apply_failed"))
    verdicts = node_verdicts(ctrf, f2p, p2p, apply_failed)
    grader_status = {nid: verdicts[("f2p", nid)] for nid in f2p}
    grader_status.update({nid: verdicts[("p2p", nid)] for nid in p2p})


    recomputed = 1 if (f2p and all(verdicts[("f2p", n)][0] == "passed" for n in f2p)
                       and all(verdicts[("p2p", n)][0] == "passed" for n in p2p)) else 0
    failed_scored = {nid for nid, (status, _message) in grader_status.items() if status != "passed"}
    rows = {nid: _row(nid, verdict, gates, failed_scored) for nid, verdict in grader_status.items()}
    gaps = [nid for nid in dict.fromkeys(f2p + p2p)
            if rows[nid]["status"] == "not_evaluated" and not rows[nid]["charged"]]

    capabilities = cmap["capabilities"]
    by_id = {c["id"]: c for c in capabilities}
    conditions = cmap.get("layer_conditions", [])
    linked: dict[str, list[dict]] = {}
    for cond in conditions:
        linked.setdefault(cond["capability"], []).append(dict(rows[cond["node_id"]]))

    def own_rows(c: dict) -> list[dict]:
        return [rows[n] for n in c["evidence"]] + linked.get(c["id"], [])

    def own_pass(c: dict) -> bool:
        return c["status"] == "certified" and all(r["status"] == "passed" for r in own_rows(c))

    memo: dict[str, bool] = {}

    def credited(cid: str) -> bool:
        if cid not in memo:
            c = by_id[cid]
            memo[cid] = (counted(c) and own_pass(c)
                         and all(credited(r) for r in c.get("requires", [])))
        return memo[cid]

    cap_rows = []
    for c in capabilities:
        requires = list(c.get("requires", []))
        row = {
            "id": c["id"],
            "title": c["title"],
            "layer": c["layer"],
            "status": c["status"],
            "report_only": bool(c.get("report_only", False)),
            "merge": c.get("merge", "all_of"),
            "requires": requires,
            "credited": credited(c["id"]),
        }
        if c["status"] == "certified":
            row["evidence"] = [dict(rows[n]) for n in c["evidence"]]
            row["conditions"] = [dict(r) for r in linked.get(c["id"], [])]
            row["evidence_passed"] = all(r["status"] == "passed" for r in row["evidence"])
            row["requires_met"] = all(credited(r) for r in requires)
            row["passed"] = own_pass(c) and row["requires_met"]
            row["result"] = _result(row["evidence"] + row["conditions"], row["requires_met"])
        elif c["status"] == "diagnostic":
            row["evidence"] = [dict(diagnostics.get(n, {"node_id": n, "status": "unread"})) for n in c["evidence"]]
            statuses = {r["status"] for r in row["evidence"]}
            row["reading"] = ("passed" if statuses == {"passed"} else
                              "failed" if statuses & {"failed", "skipped"} else "not_measured")
        else:
            row["evidence"] = []
        cap_rows.append(row)

    layers = {}
    for layer in LAYERS:
        caps = [r for r in cap_rows if r["layer"] == layer and not r["report_only"]]
        conds = [dict(rows[c["node_id"]], capability=c["capability"],
                      **({"ledger_id": c["ledger_id"]} if "ledger_id" in c else {}))
                 for c in conditions if c["layer"] == layer]
        by_status = {s: sum(1 for r in caps if r["status"] == s) for s in STATUSES}
        credited_n = sum(1 for r in caps if r["credited"])
        conds_ok = sum(1 for r in conds if r["status"] == "passed")
        certified_layer = bool(caps) and by_status["certified"] == len(caps)
        layers[layer] = {
            "declared": bool(caps),
            "certified": certified_layer,
            "complete": certified_layer and credited_n == len(caps) and conds_ok == len(conds),
            "capabilities_total": len(caps),
            "certified_total": by_status["certified"],
            "diagnostic_total": by_status["diagnostic"],
            "planned_total": by_status["planned"],
            "credited": credited_n,
            "not_certified": [r["id"] for r in caps if r["status"] != "certified"],
            "conditions": conds,
            "conditions_passed": conds_ok,
            "conditions_total": len(conds),
        }
    highest = 0
    for layer in LAYERS:
        if not layers[layer]["complete"]:
            break
        highest += 1

    rules = []
    for rule in cmap.get("integrity", []):
        ev = [dict(rows[n]) for n in rule["evidence"]]
        rules.append({"id": rule["id"], "title": rule["title"], "detects": list(rule["detects"]),
                      "ok": all(r["status"] == "passed" for r in ev), "evidence": ev})
    integrity_ok = all(r["ok"] for r in rules)

    return {
        "binary": {"reward": reward.get("reward"), "recomputed": recomputed, "apply_failed": apply_failed},
        "capabilities": cap_rows,
        "capability_vector": {r["id"]: 1 if r["credited"] else 0 for r in cap_rows},
        "capability_status": {r["id"]: r["status"] for r in cap_rows},
        "capabilities_by_status": {s: sum(1 for r in cap_rows if r["status"] == s) for s in STATUSES},
        "scoring_scope": [r["id"] for r in cap_rows if r["status"] == "certified" and not r["report_only"]],
        "layers": layers,
        "highest_continuous_layer": highest,
        "integrity": {"ok": integrity_ok, "rules": rules, "not_ok": [r["id"] for r in rules if not r["ok"]]},
        "credited_layer": highest if integrity_ok else 0,
        "measurement_gaps": gaps,
        "unmapped": [dict(rows[u["node_id"]], reason=u["reason"]) for u in cmap.get("unmapped", [])],
    }


def _read_json(path: Path, errors: list[str], what: str) -> tuple[object, bytes | None]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        errors.append(f"no {path.name}: {what}")
        return None, None
    except OSError as exc:
        errors.append(f"{path.name} unreadable ({type(exc).__name__}: {exc})")
        return None, None
    try:
        return json.loads(raw.decode("utf-8")), raw
    except (UnicodeDecodeError, ValueError) as exc:
        errors.append(f"{path.name} is not JSON ({exc})")
        return None, raw


def void_state(verifier_dir: Path, errors: list[str]) -> dict | None:
    """The run's void when the verifier withheld its reward, else None. Contradictions are appended to `errors`.

    A void is not a candidate result: no reward.json means grader.py published nothing -- it withheld the reward on
    infrastructure evidence (and said so in infra_void.json), or the verifier stopped first. Either way pier banks
    the verifier EXIT trap's reward.txt = -1 (or, without the trap, raises RewardFileNotFoundError), never 0.
    """
    declaration = verifier_dir / "infra_void.json"
    txt = verifier_dir / "reward.txt"
    if (verifier_dir / "reward.json").exists():
        if declaration.exists():
            errors.append("infra_void.json and reward.json are both present: a void withholds the reward, so the "
                          "verifier's own outputs contradict each other")
        if txt.exists():
            errors.append("reward.txt is present beside reward.json: pier banks reward.txt first, so reward.json is "
                          "not the number this trial scored")
        return None
    if txt.exists():
        try:
            text = txt.read_text(encoding="utf-8").strip()
            banked = float(text)
        except (OSError, ValueError):
            text, banked = "(unreadable)", None
        if banked != -1.0:
            errors.append(f"reward.txt holds {text[:40]!r} and there is no reward.json: that is not this verifier's "
                          f"output, whose only reward.txt is the EXIT trap's -1")
            return None
    if not declaration.exists():
        return {"declared": False,
                "reason": "no reward.json and no infra_void.json: the verifier stopped before publishing a reward"}
    try:
        doc = json.loads(declaration.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"declared": True, "reason": "infra_void.json is present but unreadable"}
    reason = doc.get("reason") if isinstance(doc, dict) else None
    return {"declared": True,
            "reason": reason.strip()[:MESSAGE_CHARS] if _nonempty_str(reason) else "infra_void.json names no reason"}


def build_record(tests_dir: Path, verifier_dir: Path) -> dict:
    """The whole layered record for one verifier run. Never raises: a failure is a `valid: false` record."""
    errors: list[str] = []
    record: dict = {"schema_version": SCHEMA, "grader_version": GRADER_VERSION, "capability_map_sha256": None,
                    "config_sha256": None, **{key: None for key in PROVENANCE_KEYS}}
    try:
        cmap, map_raw = _read_json(tests_dir / MAP_NAME, errors, "this task ships no capability map")
        cfg, cfg_raw = _read_json(tests_dir / "config.json", errors, "the scored whitelists are missing")
        for key in PROVENANCE_KEYS:
            record[key] = cmap.get(key) if isinstance(cmap, dict) else None
        record["capability_map_sha256"] = hashlib.sha256(map_raw).hexdigest() if map_raw is not None else None
        record["config_sha256"] = hashlib.sha256(cfg_raw).hexdigest() if cfg_raw is not None else None
        if cmap is not None and not isinstance(cfg, dict):
            errors.append("config.json is not a JSON object")
        elif cmap is not None:
            errors += [f"capability map: {e}" for e in validate_map(cmap, cfg)]

        void = void_state(verifier_dir, errors)
        if void is not None:
            record.update(valid=False, outcome="void", void=void, errors=errors, binary={"reward": None})
            return record
        reward, _ = _read_json(verifier_dir / "reward.json", errors, "the verifier published no reward")
        if reward is not None and not (isinstance(reward, dict) and isinstance(reward.get("reward"), (int, float))
                                       and not isinstance(reward.get("reward"), bool)):
            errors.append("reward.json carries no numeric 'reward'")
            reward = None
        ctrf = None
        if isinstance(reward, dict) and not reward.get("apply_failed"):
            ctrf, _ = _read_json(verifier_dir / "ctrf.json", errors,
                                 "grader.py's per-node verdicts are missing, so no capability can be read")
        if errors:
            record.update(valid=False, outcome="invalid", errors=errors,
                          binary={"reward": reward.get("reward") if isinstance(reward, dict) else None})
            return record
        gates = gate_records(verifier_dir)
        diagnostic_ids = [n for c in cmap["capabilities"] if c.get("status") == "diagnostic" for n in c["evidence"]]
        diagnostics = diagnostic_rows(cfg, verifier_dir, diagnostic_ids, bool(reward.get("apply_failed")), gates)
        body = grade(cmap, cfg, reward, ctrf if isinstance(ctrf, dict) else None, diagnostics, gates)
        body["binary"]["consistent"] = float(body["binary"]["reward"]) == float(body["binary"]["recomputed"])
        if not body["binary"]["consistent"]:
            record.update(valid=False, outcome="invalid", binary=body["binary"], errors=[
                f"reward.json says reward={body['binary']['reward']} but grader.py's own per-node verdicts give "
                f"{body['binary']['recomputed']}: the two files do not describe the same grading"])
            return record
        record.update(valid=True, outcome="graded", errors=[], **body)
        return record
    except Exception as exc:  # noqa: BLE001 -- a bug here must not cost the run its record
        record.pop("void", None)
        record.update(valid=False, outcome="invalid",
                      errors=errors + [f"layered_grader.py failed: {type(exc).__name__}: {exc}"],
                      binary=record.get("binary") or {"reward": None})
        return record


def render(record: dict) -> str:
    return json.dumps(record, indent=1, sort_keys=True) + "\n"


def summary(record: dict) -> str:
    if record.get("outcome") == "void":
        void = record.get("void") or {}
        return (f"VOID ({'declared' if void.get('declared') else 'undeclared'}: {void.get('reason')}) -- not a "
                f"candidate result: nothing credited, nothing failed"
                + (f"; also: {'; '.join(record['errors'])}" if record.get("errors") else ""))
    if not record.get("valid"):
        return f"NOT VALID -- {'; '.join(record.get('errors') or [])}"
    parts = []
    for layer in LAYERS:
        info = record["layers"][layer]
        if not info["declared"]:
            parts.append(f"{layer} -")
            continue
        state = "complete" if info["complete"] else ("incomplete" if info["certified"] else "not certified")
        conds = f", conditions {info['conditions_passed']}/{info['conditions_total']}" if info["conditions_total"] else ""
        parts.append(f"{layer} {state} (credited {info['credited']}/{info['capabilities_total']}, certified "
                     f"{info['certified_total']}{conds})")
    integ = "ok" if record["integrity"]["ok"] else f"NOT ok {record['integrity']['not_ok']}"
    gaps = f"; MEASUREMENT GAPS {record['measurement_gaps']}" if record["measurement_gaps"] else ""
    return (f"{record.get('task')} {record.get('map_version')}: {', '.join(parts)}; "
            f"highest {record['highest_continuous_layer']}, credited {record['credited_layer']}; integrity {integ}; "
            f"reward {record['binary']['reward']} (recomputed {record['binary']['recomputed']}){gaps}")


def main() -> int:
    tests_dir = Path(os.environ.get("TESTS_DIR", "/tests"))
    verifier_dir = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier"))
    record = build_record(tests_dir, verifier_dir)
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / OUTPUT_NAME).write_text(render(record), encoding="utf-8")
    log(summary(record))
    return 0 if record.get("valid") else 1


if __name__ == "__main__":
    sys.exit(main())
