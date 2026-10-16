#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""R-ATTENTION after the witness launch: the captured sublayer histories assembled per input and layer, the author replay on
them, the E of every compared tensor, and the consumption control's logits compared -- all as records for the judge."""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

MODEL = "/model/zai-org/GLM-5.3-Flash"
PREFILL_ROWS, DECODE_ROWS = 32, 16
REDUCED_REL = 2.0 ** -6


def _rank_layout(per: dict) -> str:
    """How the ranks observed one position's attention result: `single` (one rank), `reduced` (every rank holds the
    same reduced sublayer output -- each within REDUCED_REL relative RMS of the lowest rank's) or `partials` (each rank
    holds its tensor-parallel partial, to be summed). The public boundary is after the reduction; an implementation may
    hand either layout to the rest of the layer (attention-numerics.md)."""
    ys = [per[r][1].float() for r in sorted(per)]
    if len(ys) == 1:
        return "single"
    ref = ys[0]
    scale = float(ref.pow(2).mean().sqrt())
    if all(float((y - ref).pow(2).mean().sqrt()) <= REDUCED_REL * scale for y in ys[1:]):
        return "reduced"
    return "partials"


def _events(wdir: Path) -> list[dict]:
    out = []
    for f in glob.glob(str(wdir / "events" / "*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return sorted(out, key=lambda e: (e.get("pid"), e.get("n")))


def _forward_after(events_by_pid: dict, call: dict) -> dict | None:
    """The target forward whose record follows this attention call in its process (the call ran inside it)."""
    for e in events_by_pid.get(call["pid"], []):
        if e["n"] > call["n"] and e["k"] == "forward" and not e.get("draft"):
            return e
    return None


def assemble_input(events: list[dict], cseq: int, layer: int, final_ids: list[int]):
    import torch
    from safetensors.torch import load_file
    by_pid: dict = {}
    for e in events:
        by_pid.setdefault(e["pid"], []).append(e)
    calls = [e for e in events if e["k"] == "attention_call" and e.get("cseq") == cseq and e["layer"] == layer]
    rows: dict = {}
    token_checks = {"kept": 0, "left_out_other_token": [], "left_out_unmatched": 0}
    for c in sorted(calls, key=lambda e: (e["pid"], e["call"])):
        t = load_file(c["file"])
        fwd = _forward_after(by_pid, c)
        fwd_pos = fwd.get("positions") if fwd else None
        fwd_ids = fwd.get("input_ids") if fwd else None
        for j, p in enumerate(t["positions"].tolist()):
            if fwd_pos is None or fwd_ids is None or j >= len(fwd_pos) or fwd_pos[j] != p:
                token_checks["left_out_unmatched"] += 1
                continue
            if p >= len(final_ids) or fwd_ids[j] != final_ids[p]:
                token_checks["left_out_other_token"].append(p)
                continue
            token_checks["kept"] += 1
            rows.setdefault(p, {})[c["pid"]] = (t["x"][j], t["y"][j], c["where"], c["call"])
    ranks = sorted({pid for r in rows.values() for pid in r})
    if not rows:
        return None, {"problem": "no captured rows for this input and layer"}
    S = max(rows) + 1
    missing = [p for p in range(S) if p not in rows]
    if missing:
        return None, {"problem": f"{len(missing)} positions of the history were never observed (first {missing[:5]})"}
    ref_rank = min(r for r in ranks if all(r in rows[p] for p in range(S))) if any(
        all(r in rows[p] for p in range(S)) for r in ranks) else None
    if ref_rank is None:
        return None, {"problem": "no rank observed the whole history"}
    x = torch.stack([rows[p][ref_rank][0] for p in range(S)])
    inputs_identical = all(torch.equal(rows[p][r][0], rows[p][ref_rank][0]) for p in range(S) for r in rows[p])
    layouts = {p: _rank_layout(rows[p]) for p in range(S)}
    kinds = sorted({v for v in layouts.values() if v != "single"})
    counts = {k: sum(1 for v in layouts.values() if v == k) for k in ("single", "partials", "reduced")}
    cand, per_rank, used = {}, {}, {}
    for p in range(S):
        if layouts[p] == "reduced":
            cand[p] = rows[p][min(rows[p])][1].to(torch.bfloat16)
        else:
            acc = None
            for r, (xr, yr, where, call) in rows[p].items():
                acc = yr.float() if acc is None else acc + yr.float()
            cand[p] = acc.to(torch.bfloat16)
        per_rank[p] = sorted(rows[p])
        used[p] = sorted({rows[p][r][2] for r in rows[p]})
    info = {"history_len": S, "ranks": ranks, "reference_rank_for_inputs": ref_rank, "inputs_identical_across_ranks":
            inputs_identical, "rank_layout": "mixed" if len(kinds) > 1 else (kinds[0] if kinds else "single"),
            "rank_layouts": counts,
            "token_checks": {**token_checks, "left_out_other_token":
                                                  sorted(set(token_checks["left_out_other_token"]))[:16]},
            "paths": {str(p): used[p] for p in list(used)[-(PREFILL_ROWS + DECODE_ROWS):]}}
    return (x, cand, per_rank), info


def replay_all(wdir: Path, cls: str, attention_phase: dict, inputs: list[dict] | None = None) -> dict:
    import torch
    from glm53_protocols.attention import E_FLOOR, allowance, relative_rms_error, within_policy
    from infb_verify import attention_replay as R
    out: dict = {"representation": R.REPRESENTATION, "fp8_kv": True, "readings": list(R.READINGS), "tensors": [],
                 "consumption": None, "started": time.time()}
    if not attention_phase.get("completed"):
        out["problem"] = f"the attention phase did not complete: {attention_phase.get('error')}"
        return out
    events = _events(wdir)
    cfg = json.loads(Path(MODEL, "config.json").read_text())["text_config"]
    types = cfg["layer_types"]
    layers = sorted({[i for i, t in enumerate(types) if t == "deepseek_sparse_attention"][k] for k in (0, -1)})
    weights = {L: R.load_layer(MODEL, L) for L in layers}
    prompts = {i["id"]: i["prompt_ids"] for i in (inputs or [])}
    for inp in attention_phase.get("inputs", []):
        prompt_len = inp["total"] - DECODE_ROWS
        final_ids = list(prompts.get(inp["id"], [])) + list(inp.get("output_ids", []))
        for L in layers:
            rec: dict = {"input": inp["id"], "total": inp["total"], "layer": L}
            if len(prompts.get(inp["id"], [])) != prompt_len:
                rec.update(status="invalid", problem="the input's prompt ids are not the capture's prompt")
                out["tensors"].append(rec)
                continue
            got, info = assemble_input(events, inp["cseq"], L, final_ids)
            rec.update(info)
            if got is None:
                rec["status"] = "invalid"
                out["tensors"].append(rec)
                continue
            x, cand, per_rank = got
            for part, positions in (("prefill32", list(range(prompt_len - PREFILL_ROWS, prompt_len))),
                                    ("decode16", list(range(prompt_len, prompt_len + DECODE_ROWS)))):
                r2 = dict(rec, part=part, positions=[positions[0], positions[-1]])
                absent = [p for p in positions if p not in cand]
                if absent:
                    r2.update(status="invalid", problem=f"positions {absent[:4]} were not observed")
                    out["tensors"].append(r2)
                    continue
                c = torch.stack([cand[p] for p in positions]).cuda().float()
                xs = x[: max(positions) + 1].cuda()
                a = R.replay(weights[L], cfg, xs, positions, fp8_kv=True)
                a_exact = R.replay(weights[L], cfg, xs, positions, reading="exact")
                m = R.metric(c, a)
                r2.update(status="valid" if m.get("valid") else "invalid", **m,
                          rms_candidate=float(c.double().pow(2).mean().sqrt()),
                          ranks_per_row=sorted({len(per_rank[p]) for p in positions}))
                if m.get("valid"):
                    e_rep = relative_rms_error(a, a_exact)
                    r2.update(E_rep=e_rep, allowance=allowance(e_rep), ratio=m["E"] / max(e_rep, E_FLOOR),
                              within=within_policy(m["E"], m["max_abs"], e_rep))
                out["tensors"].append(r2)
    out["consumption"] = connection(events, attention_phase.get("consumption") or [], layers)
    out["finished"] = time.time()
    return out


def _tf_logits(events: list[dict], run: dict, prompt_len: int, fixed_tail: list[int]):
    """One teacher-forced run's logits, from the rank that served it: the eager path's rows for the last 32 prompt
    positions (the logits processor's input-log-probability logits) and the graph path's rows for the 16 positions
    after the prompt (decode or verify rows whose own input token is the fixed sequence's at that position)."""
    import torch
    from safetensors.torch import load_file
    mine = [e for e in events if e.get("cseq") == run["cseq"] and e["k"] in ("logits", "pruned_logits")]
    serving = sorted({e["pid"] for e in mine if e["k"] == "pruned_logits"})
    if not serving:
        return None, None, "no prefill logits were recorded for the run"
    pid = serving[0]
    pruned = [e for e in mine if e["pid"] == pid and e["k"] == "pruned_logits"]
    eager = torch.cat([load_file(e["file"])["logits"] for e in sorted(pruned, key=lambda e: e["call"])])
    if eager.shape[0] != PREFILL_ROWS:
        return None, None, f"{eager.shape[0]} prefill logits rows, {PREFILL_ROWS} expected"
    base = prompt_len + DECODE_ROWS + 1 - len(fixed_tail)
    want = {prompt_len + k: fixed_tail[prompt_len + k - base] for k in range(DECODE_ROWS)}
    rows: dict = {}
    for e in sorted((e for e in mine if e["pid"] == pid and e["k"] == "logits" and e.get("can_run_graph")),
                    key=lambda e: e["n"]):
        if e.get("mode") not in ("DECODE", "TARGET_VERIFY") or not e.get("positions") or not e.get("input_ids"):
            continue
        t = load_file(e["file"])["logits"]
        for j, p in enumerate(e["positions"]):
            if p in want and j < len(e["input_ids"]) and e["input_ids"][j] == want[p]:
                rows[p] = t[j]
    missing = [p for p in want if p not in rows]
    if missing:
        return eager, None, f"graph-path rows missing at positions {missing[:4]}"
    return eager, torch.stack([rows[p] for p in sorted(want)]), None


def _pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _compare(vals: list) -> dict:
    """The unmodified runs and the zeroed one (last): the fixed rule (10 x the largest repeat difference and 1e-6,
    maximum absolute logit difference), and the argmax agreement of the unmodified runs with the zeroed run's flips."""
    base, zeroed = vals[:-1], vals[-1]
    repeat = max(float((base[i] - base[j]).abs().max()) for i, j in _pairs(len(base)))
    zero = float((zeroed - base[0]).abs().max())
    thr = max(10 * repeat, 1e-6)
    am = [v.argmax(dim=-1) for v in vals]
    agree = am[0] == am[0]
    for a in am[1:-1]:
        agree = agree & (a == am[0])
    flips = int(((am[-1] != am[0]) & agree).sum())
    n_agree = int(agree.sum())
    return {"max_repeat_diff": repeat, "zeroed_diff": zero, "threshold": thr, "consumed": zero > thr,
            "positions": int(agree.numel()), "argmax_agreeing_positions": n_agree, "zeroed_argmax_flips": flips,
            "zeroed_flip_fraction": flips / n_agree if n_agree else None}


def _rms(t) -> float:
    return float(t.double().pow(2).mean().sqrt())


CONNECTION_FLOOR = 2.0 ** -8
CONNECTION_K = 3.0
REEXECUTIONS = 3


def _branch_rows(events: list[dict], cseq: int) -> dict:
    """(pid, layer, path) -> branch -> {row key: row}, from one armed run's `mlp_input` events. Rows are keyed by
    position when they aligned with the layer's positions, else by (call order, index) within the branch."""
    from safetensors.torch import load_file
    groups: dict = {}
    order: dict = {}
    for e in sorted((e for e in events if e.get("k") == "mlp_input" and e.get("cseq") == cseq and e.get("rows")),
                    key=lambda e: e["n"]):
        t = load_file(e["file"])
        key = (e["pid"], e["layer"], "graph" if e.get("where") == "graph" else "eager")
        rows = groups.setdefault(key, {}).setdefault(e["branch"], {})
        if e.get("aligned", True) and "positions" in t:
            for p, h in zip(t["positions"].tolist(), t["h"]):
                rows[("p", p)] = h
        else:
            k = order.setdefault((key, e["branch"]), [0])
            for h in t["h"]:
                rows[("r", k[0])] = h
                k[0] += 1
    return groups


def _connected(branches: dict) -> dict:
    """The within-forward rule on one (probe, layer, path, rank): R = RMS(m_U), floor = 2^-8 R, D = RMS(m_U - m_Z),
    N = max_k RMS(m_U - m_Uk); consumed when D > floor and D > 3 N."""
    import torch
    reex = sorted(b for b in branches if b.startswith("U") and b != "U")
    if "U" not in branches or "Z" not in branches or len(reex) != REEXECUTIONS:
        return {"status": "invalid", "problem": f"branches {sorted(branches)}: U, Z and {REEXECUTIONS} re-executions "
                                                f"expected"}
    keys = sorted(set(branches["U"]).intersection(branches["Z"], *[branches[b] for b in reex]))
    if not keys or any(len(branches[b]) != len(keys) for b in ("U", "Z", *reex)):
        return {"status": "invalid", "problem": "the branches did not record the same rows"}
    mu = torch.stack([branches["U"][k] for k in keys]).double()
    R_, D = _rms(mu), _rms(mu - torch.stack([branches["Z"][k] for k in keys]).double())
    Ns = [_rms(mu - torch.stack([branches[b][k] for k in keys]).double()) for b in reex]
    N, floor = max(Ns), CONNECTION_FLOOR * _rms(mu)
    return {"status": "valid", "rows": len(keys), "R": R_, "D": D, "N": N, "N_each": Ns, "floor": floor,
            "D_over_floor": D / floor if floor else None, "D_over_N": D / N if N else None,
            "consumed": D > floor and D > CONNECTION_K * N}


def connection(events: list[dict], probes: list[dict], layers: list[int] | None = None) -> dict:
    """The connection control: per probe, designated layer, path and rank, the within-forward rule on the MLP
    sublayer's input (`_connected`); at the dense probe also part (i), the logit rule on three unmodified runs and one
    with every designated layer's observed result zeroed (`_compare`)."""
    if not probes:
        return {"status": "invalid", "problem": "no consumption probe ran"}
    out: dict = {"status": "valid", "inputs": [], "floor_rel": CONNECTION_FLOOR, "k": CONNECTION_K,
                 "reexecutions": REEXECUTIONS}
    for probe in probes:
        rec: dict = {"id": probe["id"], "total": probe["total"], "observable": probe.get("observable")}
        if not probe.get("served_is_forced") or "cseq" not in probe:
            rec.update(status="invalid", problem="the armed run did not serve the fixed sequence")
            out["inputs"].append(rec)
            out["status"] = "invalid"
            continue
        groups = _branch_rows(events, probe["cseq"])
        rec["layers"] = {}
        problems = []
        for L in layers or []:
            per: dict = {}
            for (pid, layer, path), branches in sorted(groups.items()):
                if layer == L:
                    per.setdefault(str(pid), {})[path] = _connected(branches)
            if not per:
                problems.append(f"layer {L}: no rank recorded the MLP input in the armed run")
            for pid, paths in per.items():
                if "graph" not in paths:
                    problems.append(f"layer {L} rank {pid}: no graph rows")
                for path, q in sorted(paths.items()):
                    if q["status"] != "valid":
                        problems.append(f"layer {L} rank {pid} {path}: {q['problem']}")
            rec["layers"][str(L)] = per
        runs = probe.get("dense_runs")
        if runs is not None:
            if len(runs) < 4 or not runs[-1].get("zero") or not all(r.get("served_is_forced") for r in runs):
                problems.append("the dense probe's four runs did not all serve the fixed sequence")
            else:
                vals = [_tf_logits(events, r, probe["prompt_len"], probe["fixed_tail"]) for r in runs]
                bad = [v[2] for v in vals if v[2]]
                if bad:
                    problems.append(f"dense probe: {bad[0]}")
                else:
                    rec["dense"] = {"eager": _compare([v[0] for v in vals]), "graph": _compare([v[1] for v in vals])}
        rec["status"] = "invalid" if problems else "valid"
        if problems:
            rec["problem"] = "; ".join(problems[:4])
            out["status"] = "invalid"
        out["inputs"].append(rec)
    return out
