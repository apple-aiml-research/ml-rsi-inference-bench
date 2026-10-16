#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The serving-binding judgement (the sparse-indexer contract) over the observer's records.

Coverage: every consumer layer and path has at least one restrictive row (more than 16,384 visible compressed
positions) checked fully, and its attention use checked on all four TP ranks. The mathematics runs on rank 0; the
other ranks' pools and selections must equal rank 0's for the same step and row.

Outcome: a failed check is the candidate's (`candidate_fail`); a path that was not observed, an observer that did not
install, or a record that cannot be read is `blocked_by:verifier` (censored). An observation never turns into a pass
by its absence.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from infb_verify import _trusted

REF = _trusted.load("sparse_indexer/reference.py")
B = REF.CANDIDATE_BLOCK
NC = REF.NUM_CANDIDATES
K = REF.TOPK
RESTRICTIVE = 16384
RANKS = 4


def _torch():
    import torch  # noqa: PLC0415

    return torch


def load(audit_dir: Path) -> dict:
    """{rank: {"install": [records], "steps": [records]}} from the observer's directory."""
    torch = _torch()
    import json  # noqa: PLC0415

    out: dict = {}
    for d in sorted(Path(audit_dir).glob("rank*")):
        try:
            rank = int(d.name[4:])
        except ValueError:
            continue
        inst = [json.loads(p.read_text()) for p in sorted(d.glob("install-*.json"))]
        steps = [torch.load(p, weights_only=False) for p in sorted(d.glob("step*.pt"))]
        faults = [p.read_text()[:2000] for p in sorted(d.glob("after-step-fault-*.txt"))]
        out[rank] = {"install": inst, "steps": steps, "faults": faults}
    return out


def dequant_keys(rec: dict, visible: int):
    """The first `visible` index keys of one request, FP32 [visible, 128], from its raw pages."""
    torch = _torch()
    raw = rec["raw"]
    page_kv = int(rec["page_kv"])
    n = raw.shape[0]
    kv = torch.as_strided(raw.contiguous(), (n, page_kv, 1, REF.PAGE_RECORD_BYTES),
                          (page_kv * REF.PAGE_RECORD_BYTES, REF.PAGE_RECORD_BYTES, REF.PAGE_RECORD_BYTES, 1))
    values, words = REF.page_tokens(kv, torch.arange(n), visible)
    return REF.dequantize(values, words)


def row_scores(torch, q_u8, q_scale, weights, keys):
    q = REF.dequantize(q_u8.view(torch.int8).reshape(1, REF.NUM_HEADS, REF.PACKED_BYTES),
                       q_scale.reshape(1, REF.NUM_HEADS).view(torch.int32))
    return REF.scores(q, weights.reshape(1, REF.NUM_HEADS).float(), keys)[0]


def source_pool_problems(torch, pool, scores, visible: int) -> list[str]:
    """The source-pool rule for one row: legal and unique blocks, the newest reachable block included, all
    blocks when they fit, otherwise no excluded block's block-max more than 2 delta above the lowest kept non-forced
    block's."""
    problems = []
    nblocks = math.ceil(visible / B)
    p = pool.to(torch.int64)
    valid = p[p >= 0]
    if (p < -1).any():
        problems.append("a pool entry below -1")
    if valid.unique().numel() != valid.numel():
        problems.append("a block twice in the pool")
    if (valid >= nblocks).any():
        problems.append("a pool block beyond the visible positions")
    if valid.numel() != min(NC, nblocks):
        problems.append(f"{valid.numel()} pool blocks, the contract requires {min(NC, nblocks)}")
    if nblocks and not bool((valid == nblocks - 1).any()):
        problems.append("the pool lacks the newest reachable block")
    if problems or nblocks <= NC:
        if nblocks <= NC and not problems and valid.numel() != nblocks:
            problems.append("every visible block fits and the pool does not hold them all")
        return problems
    pad = nblocks * B - visible
    s = torch.cat([scores.float(), torch.full((pad,), -float("inf"))]) if pad else scores.float()
    bm = s.view(nblocks, B).max(1).values
    delta = REF.SOURCE_DELTA * float(scores.abs().max())
    kept = valid[valid != nblocks - 1]
    excluded = torch.ones(nblocks, dtype=torch.bool)
    excluded[valid] = False
    if kept.numel() and excluded.any():
        lowest = float(bm[kept].min())
        worst = float(bm[excluded].max())
        if worst > lowest + 2 * delta:
            problems.append(f"an excluded block's max score {worst:.6g} exceeds the lowest kept block's {lowest:.6g} "
                            f"by more than 2 delta ({2 * delta:.6g})")
    return problems


def selection_problems(torch, sel, pool, scores, visible: int) -> list[str]:
    """A consumer's selection for one row: legal within the pool, then within 2 delta of its own scores."""
    allowed = REF.candidate_positions(pool, visible)
    probs = REF.selection_problems(sel, allowed, visible)
    if probs:
        return probs
    delta = REF.SOURCE_DELTA * float(scores.abs().max()) if scores.numel() else 0.0
    if not REF.within_selection_slack(sel, allowed, scores, delta):
        return ["an excluded pool position scores more than 2 delta above the smallest selected one"]
    return []


def weights_problems(torch, consumed, proj, fused: dict | None) -> list[str]:
    """Weight rounding: consumed == bf16(fp32(projection) * fp32(softmax scale * head scale)), exactly, per head."""
    if not fused or not math.isfinite(fused.get("softmax_scale", float("nan"))):
        return ["unobserved: the supplied fused-query call was not seen for this layer"]
    scale = torch.tensor(float(fused["softmax_scale"]) * float(fused["head_scale"]), dtype=torch.float32)
    want = (proj.float() * scale).to(torch.bfloat16).float()
    got = consumed.float()
    if not torch.equal(got, want):
        bad = int((got != want).sum())
        return [f"the consumed weights are not the scaled projection rounded once to BF16 ({bad} of {got.numel()} "
                f"heads differ)"]
    return []


def decode_attention_problems(torch, idx, count: int, table, block_size: int, sel) -> list[str]:
    """The decode call's consumed compressed positions: the selection through the page table, as a set, with its
    count."""
    s = sel.to(torch.int64)
    s = s[s >= 0]
    want = table.to(torch.int64)[s // block_size] * block_size + s % block_size
    got = idx.to(torch.int64)[:count]
    problems = []
    if count != s.numel():
        problems.append(f"FlashMLA consumed {count} compressed positions, the selection has {s.numel()}")
    if not torch.equal(torch.sort(got).values, torch.sort(want).values):
        problems.append("the consumed physical slots are not the selection through the page table")
    if (got < 0).any():
        problems.append("a consumed compressed slot below 0")
    return problems


def prefill_attention_problems(torch, idx, count: int, region: int, wbase: int, pos: int, layout, sel) -> list[str]:
    """The prefill call's combined row: the selection's first min((pos + 1) // cr, top-k) entries at the row's region,
    then its sliding window's positions at the window base, and the count."""
    window, cr, topk = (int(x) for x in layout)
    if window <= 0 or cr <= 0:
        return ["unobserved: no combined layout recorded for the row"]
    topk_len = min((pos + 1) // cr, topk)
    swa_start = max(pos - (window - 1), 0)
    swa_len = pos - swa_start + 1
    got = idx.to(torch.int64)
    problems = []
    s = sel.to(torch.int64)
    if int((s[:topk_len] >= 0).sum()) != topk_len:
        problems.append(f"the selection's first {topk_len} entries are not all valid positions")
    if not torch.equal(got[:topk_len] - region, s[:topk_len]):
        problems.append("the consumed compressed positions are not the selection at the request's region")
    want_swa = wbase + torch.arange(swa_start, swa_start + swa_len)
    if not torch.equal(got[topk_len: topk_len + swa_len], want_swa):
        problems.append("the consumed window positions are not the row's sliding window")
    if count != topk_len + swa_len:
        problems.append(f"the consumed count {count} is not {topk_len} compressed + {swa_len} window")
    return problems


def _event(step: dict, what: str, layer: int) -> list[dict]:
    return [e for e in step.get("events", []) if e.get("what") == what and e.get("layer") == layer]


#: The per-layer facts the observer writes on the device (sparse_probe.Buffers: "facts"), so a replayed graph reports
#: them as an eager step does: the index-key cache, the call's rows, the attention output buffer and its bytes, the
#: decode and prefill FlashMLA outputs, the projection's input, the decode path's compressed page size.
K_CACHE, ROWS, OUT, OUT_BYTES, FLASH_DECODE, FLASH_PREFILL, O_PROJ, BLOCK_SIZE = range(8)


def _fact(layer_rec: dict, i: int) -> int:
    f = layer_rec.get("facts")
    return int(f[i]) if f is not None and len(f) > i else 0


def judge(data: dict, source_layer: int, consumer_layers: list[int]) -> dict:
    """The node's outcome from the loaded records: {"status", "why", "coverage", "checked", "problems", ...}."""
    torch = _torch()
    out: dict = {"problems": [], "unobserved": [], "checked": [], "coverage": {}, "ranks": sorted(data)}
    installed = {r: any(i.get("status") == "installed" for i in d["install"]) for r, d in data.items()}
    out["installed"] = installed
    if sorted(r for r, ok in installed.items() if ok) != list(range(RANKS)):
        out.update(status="blocked_by:verifier",
                   why=f"verifier_fault: the observer installed on ranks {sorted(r for r, ok in installed.items() if ok)}"
                       f", not all of 0-{RANKS - 1}: {[i.get('error') for d in data.values() for i in d['install']][:2]}")
        return out
    faults = [f for d in data.values() for f in d["faults"]]
    if faults:
        out["observer_faults"] = faults[:4]
    by_rank: dict = {}
    for r, d in data.items():
        for st in d["steps"]:
            for layer, lr in st["layers"].items():
                for i, s in enumerate(lr["slots"]):
                    by_rank.setdefault(r, {})[(st["step"], int(layer), int(lr["rows"][i]))] = (st, lr, i)
    covered: dict = {}
    for (step_no, layer, row), (st, lr, i) in sorted(by_rank.get(0, {}).items()):
        if layer not in consumer_layers:
            continue
        path = "decode" if bool(lr["decode"][i]) else "prefill"
        vis = int(lr["vis"][i])
        tag = f"step {step_no} layer {layer} {path} row {row} (visible {vis})"
        src = by_rank[0].get((step_no, source_layer, row))
        if src is None:
            out["unobserved"].append(f"{tag}: the source's row was not kept")
            continue
        s_st, s_lr, s_i = src
        key = f"{path}:{int(lr['key'][i])}"
        kr = next((stp["keys"][key] for stp in data[0]["steps"] if key in stp.get("keys", {})
                   and int(stp["keys"][key]["visible"]) >= vis), None)
        if kr is None:
            out["unobserved"].append(f"{tag}: the request's index keys were not read")
            continue
        problems = []
        keys = dequant_keys(kr, vis)
        src_scores = row_scores(torch, s_lr["q"][s_i], s_lr["q_scale"][s_i], s_lr["weights"][s_i], keys)
        problems += [f"source: {p}" for p in source_pool_problems(torch, s_lr["pool"][s_i], src_scores, vis)]
        sp, cp = s_lr["pool"][s_i].to(torch.int64), lr["pool"][i].to(torch.int64)
        if not torch.equal(torch.sort(sp[sp >= 0]).values, torch.sort(cp[cp >= 0]).values):
            problems.append("the consumer read another candidate pool than the source wrote for the row")
        s_ptr, c_ptr = _fact(s_lr, K_CACHE), _fact(lr, K_CACHE)
        if not (s_ptr and c_ptr):
            out["unobserved"].append(f"{tag}: the index-key caches the calls read were not recorded")
        elif s_ptr != c_ptr:
            problems.append("the consumer read another index-key cache than the source's")
        c_scores = row_scores(torch, lr["q"][i], lr["q_scale"][i], lr["weights"][i], keys)
        problems += [f"selection: {p}" for p in selection_problems(torch, lr["selection"][i], lr["pool"][i],
                                                                   c_scores, vis)]
        w = weights_problems(torch, lr["weights"][i], lr["proj"][i], lr.get("fused"))
        (out["unobserved"] if w and w[0].startswith("unobserved") else problems).extend(f"{tag}: weight rounding: {p}" for p in w)
        ranks_ok = []
        for r in range(RANKS):
            got = by_rank.get(r, {}).get((step_no, layer, row))
            if got is None:
                out["unobserved"].append(f"{tag}: rank {r} did not keep the row")
                continue
            r_st, r_lr, r_i = got
            if r:
                for name in ("pool", "selection"):
                    if not torch.equal(r_lr[name][r_i], lr[name][i]):
                        problems.append(f"rank {r}: its {name} differs from rank 0's")
            if path == "decode":
                if not bool(r_lr["att_dec_valid"][r_i]):
                    out["unobserved"].append(f"{tag}: rank {r}: the decode FlashMLA call was not observed")
                    continue
                bs = _fact(r_lr, BLOCK_SIZE)
                if not bs:
                    out["unobserved"].append(f"{tag}: rank {r}: no compressed block size")
                    continue
                pr = decode_attention_problems(torch, r_lr["att_dec_idx"][r_i], int(r_lr["att_dec_len"][r_i]),
                                               r_lr["att_dec_table"][r_i], bs, r_lr["selection"][r_i])
            else:
                if not bool(r_lr["att_pre_valid"][r_i]):
                    out["unobserved"].append(f"{tag}: rank {r}: the prefill FlashMLA call was not observed")
                    continue
                pr = prefill_attention_problems(torch, r_lr["att_pre_idx"][r_i], int(r_lr["att_pre_len"][r_i]),
                                                int(r_lr["att_pre_region"][r_i]), int(r_lr["att_pre_wbase"][r_i]),
                                                int(r_lr["att_pre_pos"][r_i]), r_lr["att_pre_layout"][r_i],
                                                r_lr["selection"][r_i])
            problems += [f"rank {r}: attention: {p}" for p in pr]
            o_ptr, o_bytes = _fact(r_lr, OUT), _fact(r_lr, OUT_BYTES)
            flash = _fact(r_lr, FLASH_DECODE if path == "decode" else FLASH_PREFILL)
            proj_in = _fact(r_lr, O_PROJ)
            if not (o_ptr and o_bytes and flash and proj_in):
                out["unobserved"].append(f"{tag}: rank {r}: no output-use record")
                continue
            if not o_ptr <= flash < o_ptr + o_bytes:
                problems.append(f"rank {r}: the FlashMLA output is not the attention layer's output buffer")
            if not o_ptr <= proj_in < o_ptr + o_bytes:
                problems.append(f"rank {r}: the attention projection never read the FlashMLA output buffer")
            if not pr:
                ranks_ok.append(r)
        problems = [p if p.startswith("step ") else f"{tag}: {p}" for p in problems]
        out["problems"] += problems
        full = not problems and ranks_ok == list(range(RANKS))
        out["checked"].append({"row": tag, "restrictive": vis > RESTRICTIVE, "ok": full, "ranks_ok": ranks_ok})
        if full and vis > RESTRICTIVE:
            covered[(layer, path)] = covered.get((layer, path), 0) + 1
    replay = {}
    for st in data.get(0, {}).get("steps", []):
        for layer, lr in st["layers"].items():
            if int(layer) in consumer_layers and bool(lr["decode"].any()):
                eager = bool(_event(st, "fused_query", int(layer)) or _event(st, "indexer", int(layer)))
                replay[int(layer)] = replay.get(int(layer), False) or not eager
    out["replayed_decode"] = replay
    for layer in consumer_layers:
        for path in ("prefill", "decode"):
            out["coverage"][f"{layer}:{path}"] = covered.get((layer, path), 0)
    missing = [k for k, v in out["coverage"].items() if v == 0]
    no_replay = [layer for layer in consumer_layers if not replay.get(layer)]
    if out["problems"]:
        out.update(status="candidate_fail", why=f"{len(out['problems'])} problem(s): " + "; ".join(out["problems"][:3]))
    elif missing or no_replay:
        why = []
        if missing:
            why.append(f"no fully checked restrictive row for {missing}")
        if no_replay:
            why.append(f"no decode observation from a replayed graph for layers {no_replay}")
        out.update(status="blocked_by:verifier", why="verifier_fault: unobserved: " + "; ".join(why)
                   + (f" ({out['unobserved'][:3]})" if out["unobserved"] else ""))
    else:
        out.update(status="pass", why="")
    return out


def summary(result: dict) -> dict:
    return {k: result.get(k) for k in ("status", "why", "coverage", "replayed_decode", "installed")} | {
        "rows_checked": len(result.get("checked", [])), "problems": result.get("problems", [])[:10],
        "unobserved": result.get("unobserved", [])[:10]}


if __name__ == "__main__":
    import json
    import sys

    res = judge(load(Path(sys.argv[1])), int(os.environ.get("SOURCE_LAYER", 20)), [24, 28, 32, 36])
    print(json.dumps(summary(res), indent=1, default=str))
