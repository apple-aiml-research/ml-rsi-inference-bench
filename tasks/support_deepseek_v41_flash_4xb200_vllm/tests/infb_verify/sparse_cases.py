#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The sparse indexer's private correctness corpus (the sparse-indexer contract): fixed case families, driven through the
public contract -- the four dependency operations and the two entry points -- and checked against the contract's
FP32 mathematics (the grading copy of sparse_indexer/reference.py).

Every case has its own generator, seeded from the private seed and its case id, and draws packed E2M1 bytes and
finite UE8M0 scale bytes directly (private scale exponents 120-128); BF16 weights are seeded unit normals; exact,
tie and zero cases are constructed. Each family returns one record per case: its identity, what was checked, the
continuous errors and the verdict (`pass`, `candidate_fail` with its reasons, or `verifier_fault`). Nothing here is
agent-facing.
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

if __name__ == "__main__":  # run as a script under `python -I` (test_sparse_indexer.py): the package's root on the path
    sys.path.insert(0, os.environ.get("TESTS_DIR", "/tests"))

import torch

from infb_verify import _trusted

C = _trusted.load("sparse_indexer/contract.py")
REF = _trusted.load("sparse_indexer/reference.py")

SEED = 41029
H, D, PB, B, NC, K = C.NUM_HEADS, C.HEAD_DIM, C.PACKED_BYTES, C.CANDIDATE_BLOCK, C.NUM_CANDIDATES, C.TOPK
NCONS = C.NUM_CONSUMERS


# ── generation ─────────────────────────────────────────────────────────────────────────────────────────────────


def generator(case_id: str) -> torch.Generator:
    """The case's own generator (CPU), from the private seed and the case id."""
    g = torch.Generator(device="cpu")
    h = 1469598103934665603
    for ch in case_id.encode():
        h = ((h ^ ch) * 1099511628211) % (2**64)
    g.manual_seed((SEED * 1_000_003 + h) % (2**63))
    return g


def mxfp4(g, shape, lo: int = 120, hi: int = 128):
    """Random packed E2M1 bytes and finite UE8M0 scale words: int8 [*shape, 64], int32 [*shape]."""
    values = torch.randint(-128, 128, (*shape, PB), generator=g, dtype=torch.int16).to(torch.int8)
    sb = torch.randint(lo, hi + 1, (*shape, 4), generator=g, dtype=torch.int16).to(torch.uint8)
    return values, sb.view(torch.int32).view(*shape)


def weights(g, *shape):
    return torch.randn(shape, generator=g).to(torch.bfloat16)


def pool_for(g, visible: int, forced: bool = True, exclude: set[int] | None = None) -> torch.Tensor:
    """A valid candidate pool for a row with `visible` positions: all visible blocks when they fit, else 2,048 unique
    blocks with the newest included (unless `forced` is off), in random order, -1 padded."""
    blocks = math.ceil(visible / B)
    row = torch.full((NC,), -1, dtype=torch.int32)
    if blocks == 0:
        return row
    if blocks <= NC:
        chosen = torch.arange(blocks)
    else:
        pool = [b for b in range(blocks - 1) if not exclude or b not in exclude]
        perm = torch.randperm(len(pool), generator=g)[: NC - 1 if forced else NC]
        chosen = torch.tensor([pool[i] for i in perm.tolist()] + ([blocks - 1] if forced else []))
    row[: chosen.numel()] = chosen[torch.randperm(chosen.numel(), generator=g)].to(torch.int32)
    return row


class CandidateError(RuntimeError):
    """An exception raised by the implementation under test (its entry points or its dependency operations, including
    an asynchronous CUDA error surfacing at the synchronization after its call)."""


def candidate(fn, *args, **kwargs):
    """Call the implementation under test and synchronize; its exceptions become CandidateError."""
    try:
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        return out
    except Exception as exc:  # noqa: BLE001 - attributed to the candidate by construction
        raise CandidateError(f"{type(exc).__name__}: {str(exc)[:600]}") from exc


@dataclass
class Record:
    case: str
    family: str
    checks: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    status: str = "pass"
    seconds: float = 0.0

    def fail(self, what: str) -> None:
        self.problems.append(what)
        self.status = "candidate_fail"

    def as_dict(self) -> dict:
        return {"case": self.case, "family": self.family, "status": self.status, "checks": self.checks,
                "problems": self.problems[:40], "n_problems": len(self.problems), "metrics": self.metrics,
                "seconds": round(self.seconds, 3)}


# ── the contract's semantics, per row ───────────────────────────────────────────────────────────────────────────


def check_row(rec: Record, tag: str, out_row, allowed, visible: int, row_scores, witness_excluded=None) -> None:
    """Structure, then the selection rule for unmaterialized scores, then any strict-gap witness."""
    probs = REF.selection_problems(out_row, allowed, visible)
    for p in probs:
        rec.fail(f"{tag}: {p}")
    if probs:
        return
    delta = REF.SOURCE_DELTA * float(row_scores.abs().max()) if row_scores.numel() else 0.0
    if not REF.within_selection_slack(out_row, allowed, row_scores, delta):
        rec.fail(f"{tag}: an excluded selectable position scores more than 2 delta above the smallest selected one")
    if witness_excluded is not None and witness_excluded.numel():
        sel = out_row[out_row >= 0].to(torch.int64)
        if torch.isin(witness_excluded, sel).any():
            rec.fail(f"{tag}: selected a position of a block outside the candidate pool (the excluded-block witness)")


def dequant_rows(q, qs):
    return REF.dequantize(q, qs)


def run_prefill(step, rec: Record, workspace=None):
    prefill, _ = candidate(C.load_entry_points)
    out = C.output_buffer(step.rows, step.q.device)
    ws = candidate(C.run_prefill, prefill, step, out, workspace)
    return out, ws


def run_decode(step, rec: Record, workspace=None):
    _, decode = candidate(C.load_entry_points)
    out = C.output_buffer(step.rows, step.q.device)
    ws = candidate(C.run_decode, decode, step, out, workspace)
    return out, ws


def check_packed_step(rec: Record, step, out, tag: str = "", witnesses: dict | None = None) -> None:
    """Every consumer row of a packed step against the FP32 mathematics."""
    kd = REF.dequantize(step.k_quant, step.k_scale)
    for c in range(NCONS):
        qd = REF.dequantize(step.q[c], step.q_scale[c])
        for r in range(step.rows):
            ks, ke = int(step.k_start[r]), int(step.k_end[r])
            visible = ke - ks
            s = REF.scores(qd[r : r + 1], step.weights[c, r : r + 1], kd[ks:ke])[0] if visible else kd.new_zeros(0)
            allowed = REF.candidate_positions(step.candidate_blocks[r], visible)
            w = (witnesses or {}).get(r)
            check_row(rec, f"{tag}c{c} r{r}", out[c, r], allowed, visible, s, w)


def check_paged_step(rec: Record, step, out, tag: str = "", witnesses: dict | None = None) -> None:
    for r in range(step.rows):
        visible = int(step.context_lens[r])
        values, words = REF.page_tokens(step.kv_cache, step.block_table[r], visible)
        kd = REF.dequantize(values, words)
        allowed = REF.candidate_positions(step.candidate_blocks[r], visible)
        for c in range(NCONS):
            qd = REF.dequantize(step.q[c, r : r + 1, 0], step.q_scale[c, r : r + 1, 0])
            s = REF.scores(qd, step.weights[c, r : r + 1], kd)[0] if visible else kd.new_zeros(0)
            w = (witnesses or {}).get(r)
            check_row(rec, f"{tag}c{c} r{r}", out[c, r], allowed, visible, s, w)


def guarded(t: torch.Tensor, pad: int = 64, value: int = 0x5A):
    """`t` placed inside a larger buffer whose neighbouring bytes carry a known pattern: (view, whole, pattern)."""
    flat = torch.full((t.numel() * t.element_size() + 2 * pad,), value, dtype=torch.uint8, device=t.device)
    view = flat[pad : pad + t.numel() * t.element_size()].view(t.dtype).view(t.shape)
    view.copy_(t)
    return view, flat, pad


def guards_intact(flat, pad, value: int = 0x5A) -> bool:
    return bool((flat[:pad] == value).all() and (flat[-pad:] == value).all())


# ── packed construction ─────────────────────────────────────────────────────────────────────────────────────────


def packed_step(case_id: str, visibles_per_request: list[list[int]], residues: list[int], device,
                candidate_stride: int = NC, pools: list | None = None):
    """A packed step: requests with the given start residues and per-row visible lengths, separated by guards; each
    request owns its keys; candidate rows optionally stored with a wider outer stride."""
    g = generator(case_id)
    starts, pos = [], 0
    for i, rows in enumerate(visibles_per_request):
        pos = math.ceil((pos + 24) / 8) * 8 + residues[i % len(residues)]
        starts.append(pos)
        pos += max(rows) if rows else 0
    n = max(pos + 24, 8)
    kq, ks = mxfp4(g, (n,))
    k_start, k_end, rows_pools = [], [], []
    for i, rows in enumerate(visibles_per_request):
        for v in rows:
            k_start.append(starts[i])
            k_end.append(starts[i] + v)
            rows_pools.append(pool_for(g, v) if pools is None else pools[len(rows_pools)])
    R = len(k_start)
    q, qs = mxfp4(g, (NCONS, R, H))
    w = weights(g, NCONS, R, H)
    store = torch.full((R, candidate_stride), -1, dtype=torch.int32)
    store[:, :NC] = torch.stack(rows_pools) if R else store[:, :NC]
    cand = store.to(device)[:, :NC]
    return C.PackedStep(q=q.to(device), q_scale=qs.to(device), k_quant=kq.to(device), k_scale=ks.to(device),
                        weights=w.to(device), k_start=torch.tensor(k_start, dtype=torch.int32, device=device),
                        k_end=torch.tensor(k_end, dtype=torch.int32, device=device), candidate_blocks=cand)


def paged_step(case_id: str, requests: list[list[int]], page_kv: int, device, holes: int = 3, extra_stride: int = 512,
               pools: list | None = None, keys=None):
    """A paged step: request i has rows with visible lengths `requests[i]` (its rows share its page table), its own
    keys in permuted physical pages with unused holes, page stride round_up(page_bytes, 512) + `extra_stride`."""
    g = generator(case_id)
    page_bytes = page_kv * C.PAGE_RECORD_BYTES
    stride = math.ceil(page_bytes / 512) * 512 + extra_stride
    pages_per = [math.ceil(max(rows) / page_kv) if rows and max(rows) else 1 for rows in requests]
    total = sum(pages_per) + holes
    storage = torch.randint(0, 256, (total, stride), generator=g, dtype=torch.int16).to(torch.uint8)
    physical = torch.randperm(total, generator=g)
    tables, cursor = [], 0
    for i, n_pages in enumerate(pages_per):
        tables.append(physical[cursor : cursor + n_pages].to(torch.int32))
        cursor += n_pages
    for i, n_pages in enumerate(pages_per):
        vals, words = mxfp4(g, (n_pages, page_kv)) if keys is None else keys[i]
        for j, p in enumerate(tables[i].tolist()):
            storage[p, : page_kv * PB] = vals[j].view(torch.uint8).flatten()
            storage[p, page_kv * PB : page_bytes] = words[j].view(torch.uint8).flatten()
    max_pages = max(pages_per)
    lens, table, owner, rows_pools = [], [], [], []
    for i, rows in enumerate(requests):
        padded = torch.full((max_pages,), int(physical[-1]), dtype=torch.int32)  # an unused page: never read
        padded[: tables[i].numel()] = tables[i]
        for v in rows:
            lens.append(v)
            table.append(padded)
            owner.append(i)
            rows_pools.append(pool_for(g, v) if pools is None else pools[len(rows_pools)])
    R = len(lens)
    q, qs = mxfp4(g, (NCONS, R, 1, H))
    w = weights(g, NCONS, R, H)
    st = storage.to(device)
    kv = st.as_strided((total, page_kv, 1, C.PAGE_RECORD_BYTES), (stride, C.PAGE_RECORD_BYTES, C.PAGE_RECORD_BYTES, 1))
    return C.PagedStep(q=q.to(device), q_scale=qs.to(device), kv_cache=kv, weights=w.to(device),
                       context_lens=torch.tensor(lens, dtype=torch.int32, device=device),
                       block_table=torch.stack(table).to(device).contiguous(),
                       row_indices=torch.tensor(owner, dtype=torch.int32, device=device),
                       candidate_blocks=torch.stack(rows_pools).to(device)), st


def assign_tables(step, perm, pages_by_request: list[int], rows_by_request: list[int], device) -> None:
    """New page tables for a paged step from a permutation of its physical pages: request i gets the next
    `pages_by_request[i]` pages (padded with the permutation's last, unused, page), repeated for its rows."""
    width = step.block_table.shape[1]
    assert sum(pages_by_request) < perm.numel() and max(pages_by_request) <= width
    cursor, row = 0, 0
    for n_pages, n_rows in zip(pages_by_request, rows_by_request):
        padded = torch.full((width,), int(perm[-1]), dtype=torch.int32)
        padded[:n_pages] = perm[cursor : cursor + n_pages]
        cursor += n_pages
        for _ in range(n_rows):
            step.block_table[row].copy_(padded.to(device))
            row += 1


# ── families ─────────────────────────────────────────────────────────────────────────────────────────────────

#: Set by main(): the family's record, rewritten after every case so that a crash leaves the cases before it.
_SINK: dict | None = None


def _run(family: str, case: str, fn: Callable[[Record], None]) -> dict:
    rec = Record(case=case, family=family)
    t0 = time.perf_counter()
    try:
        fn(rec)
    except CandidateError as e:
        rec.fail(str(e))
        rec.metrics["traceback"] = traceback.format_exc()[-2000:]
    except Exception as e:  # noqa: BLE001 - the verifier's own code: never a candidate failure
        rec.status = "verifier_fault"
        rec.problems.append(f"verifier: {type(e).__name__}: {str(e)[:500]}")
        rec.metrics["traceback"] = traceback.format_exc()[-2000:]
    rec.seconds = time.perf_counter() - t0
    d = rec.as_dict()
    if _SINK is not None:
        import json  # noqa: PLC0415

        _SINK["cases"].append(d)
        with open(_SINK["path"], "w") as f:
            json.dump({"family": family, "cases": _SINK["cases"], "complete": False}, f, indent=1)
    return d


def aligned(g, n: int, pattern, magnitude_codes=(7, 7), scale_byte: int = 130):
    """`n` packed rows whose E2M1 signs follow `pattern` (bool [128], True = negative) with magnitudes drawn from the
    code range `magnitude_codes`, and one scale byte for all four groups: int8 [n, 64], int32 [n]."""
    lo, hi = magnitude_codes
    mags = torch.randint(lo, hi + 1, (n, D), generator=g, dtype=torch.int16)
    codes = (mags | (pattern.to(torch.int16) << 3)).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
    words = torch.full((n, 4), scale_byte, dtype=torch.uint8).view(torch.int32).view(n)
    return packed.contiguous(), words


SOURCE_VISIBLE = [0, 1, 7, 8, 9, 511, 512, 513, 16383, 16384, 16385, 20480]


def two_level(device) -> list[dict]:
    """Source semantics through the supplied selection primitive, then the four consumers through both entry points.
    The newest block is made the source's lowest before forcing; beyond 16,384 positions one older block, ranked
    lowest of all by the source, is every consumer's strongest (a strict score gap): a consumer that ignores the pool
    selects it, a correct one never does. Consumers share the keys and have their own queries and weights."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import select_candidate_blocks  # noqa: PLC0415

    out = []
    for V in SOURCE_VISIBLE:
        def case(rec: Record, V=V):
            g = generator(f"two-level-{V}")
            nblocks = math.ceil(V / B)
            pattern = torch.rand(D, generator=g) < 0.5
            kq, ks = mxfp4(g, (max(V, 1),))
            victim = None
            if nblocks > NC:
                victim = int(torch.randint(0, nblocks - 1, (1,), generator=g))
                vpos = list(range(victim * B, min(victim * B + B, V)))
                kq[vpos], ks[vpos] = aligned(g, len(vpos), pattern)
            sq, sqs = mxfp4(g, (1, H))
            sw = weights(g, 1, H)
            kd = REF.dequantize(kq.to(device), ks.to(device))
            src = (REF.scores(REF.dequantize(sq.to(device), sqs.to(device)), sw.to(device), kd[:V])[0]
                   if V else torch.zeros(0, device=device))
            if V:
                lowest = float(src.min())
                src[(nblocks - 1) * B : V] = lowest - 1.0
                if victim is not None:
                    src[victim * B : victim * B + B] = lowest - 2.0
            pool = torch.full((1, NC), -1, dtype=torch.int32, device=device)
            select_candidate_blocks(src.view(1, -1).float().contiguous(), None,
                                    torch.tensor([V], dtype=torch.int32, device=device), NC, B, pool)
            valid = pool[0][pool[0] >= 0].tolist()
            rec.checks.append("the source pool: newest kept, lowest excluded, size")
            if nblocks and nblocks - 1 not in valid:
                rec.fail("the source pool lacks the newest visible block")
            if len(valid) != min(nblocks, NC) or len(set(valid)) != len(valid):
                rec.fail(f"the source pool holds {len(valid)} blocks ({len(set(valid))} distinct), not {min(nblocks, NC)}")
            if victim is not None and victim in valid:
                rec.fail("the source pool kept the lowest-ranked block")
            q = torch.empty((NCONS, 1, H, PB), dtype=torch.int8)
            qs = torch.empty((NCONS, 1, H), dtype=torch.int32)
            for c in range(NCONS):
                q[c, 0], qs[c, 0] = aligned(g, H, pattern, magnitude_codes=(4, 7), scale_byte=127)
            w = (torch.randn((NCONS, 1, H), generator=g).abs() + 0.05).to(torch.bfloat16)
            witness = ({0: torch.arange(victim * B, min(victim * B + B, V), device=device)}
                       if victim is not None else None)
            step = C.PackedStep(q=q.to(device), q_scale=qs.to(device), k_quant=kq.to(device), k_scale=ks.to(device),
                                weights=w.to(device), k_start=torch.zeros(1, dtype=torch.int32, device=device),
                                k_end=torch.tensor([V], dtype=torch.int32, device=device),
                                candidate_blocks=pool.contiguous())
            o, _ = run_prefill(step, rec)
            rec.checks.append("the consumers score within the pool (packed prefill)")
            check_packed_step(rec, step, o, "packed ", witness)
            page_kv = 64
            n_pages = max(1, math.ceil(V / page_kv))
            vals = torch.zeros((n_pages * page_kv, PB), dtype=torch.int8)
            words = torch.full((n_pages * page_kv,), 0x7F7F7F7F, dtype=torch.int32)
            vals[:V], words[:V] = kq[:V], ks[:V]
            pstep, _ = paged_step(f"two-level-{V}-paged", [[V]], page_kv, device, holes=2,
                                  pools=[pool[0].cpu()],
                                  keys=[(vals.view(n_pages, page_kv, PB), words.view(n_pages, page_kv))])
            pstep.q.copy_(q.to(device).view(NCONS, 1, 1, H, PB))
            pstep.q_scale.copy_(qs.to(device).view(NCONS, 1, 1, H))
            pstep.weights.copy_(w.to(device))
            po, _ = run_decode(pstep, rec)
            rec.checks.append("the consumers score within the pool (paged decode)")
            check_paged_step(rec, pstep, po, "paged ", witness)
            rec.metrics.update({"visible": V, "blocks": nblocks, "pool": len(valid), "witness_block": victim})
        out.append(_run("two_level", f"two-level-{V}", case))

    def ties(rec: Record):
        # equal consumer scores everywhere: any 512 of the 1,024 positions is valid
        g = generator("two-level-ties")
        kq, ks = mxfp4(g, (1,))
        kq, ks = kq.repeat(1024, 1), ks.repeat(1024)
        step = packed_step("two-level-ties", [[1024]], [0], device)
        step.k_quant.zero_()
        step.k_quant[int(step.k_start[0]) : int(step.k_start[0]) + 1024] = kq.to(device)
        step.k_scale[int(step.k_start[0]) : int(step.k_start[0]) + 1024] = ks.to(device)
        o, _ = run_prefill(step, rec)
        rec.checks.append("exact ties: any valid choice")
        check_packed_step(rec, step, o, "ties ")
    out.append(_run("two_level", "two-level-ties", ties))
    return out


# ── packed correctness ────────────────────────────────────────────────────────────────────────────────────────


PACKED_R = [1, 2, 3, 4, 5, 31, 32, 33, 127, 128, 129, 512]


def _logit_check(rec: Record, step, dep, c: int, tag: str) -> None:
    """The dependency operations directly for one consumer: sparse block ids per the ABI, metadata and compute in
    unaligned mode together, the valid columns against FP32 scores (per-row normalized error p99 < 0.01, max < 0.02)."""
    rows, lens = [], []
    for r in range(step.rows):
        ks, ke = int(step.k_start[r]), int(step.k_end[r])
        pool = step.candidate_blocks[r]
        valid = pool[(pool >= 0) & (pool < math.ceil((ke - ks) / B))].to(torch.int64)
        blocks = torch.sort(valid).values + ks // B
        row = torch.full((NC,), int(blocks[-1]) if blocks.numel() else ks // B, dtype=torch.int32, device=pool.device)
        row[: blocks.numel()] = blocks.to(torch.int32)
        rows.append(row)
        lens.append(blocks.numel())
    idx = torch.stack(rows)
    meta = candidate(dep["get_sparse_mqa_logits_metadata"], step.k_start, step.k_end, step.k_quant.shape[0], idx,
                     torch.int8, C.SPARSE_BLOCK_KV, use_unaligned_ks=True)
    logits = candidate(dep["fp8_fp4_sparse_mqa_logits"], (step.q[c], step.q_scale[c]), (step.k_quant, step.k_scale),
                       step.weights[c], meta, NC, C.SPARSE_BLOCK_KV, use_unaligned_ks=True)
    if tuple(logits.shape) != (step.rows, NC * B) or logits.dtype != torch.bfloat16:
        rec.fail(f"{tag}: logits {tuple(logits.shape)} {logits.dtype}, not bf16 [{step.rows}, {NC * B}]")
        return
    kd = REF.dequantize(step.k_quant, step.k_scale)
    qd = REF.dequantize(step.q[c], step.q_scale[c])
    worst = (0.0, 0.0)
    for r in range(step.rows):
        ks, ke = int(step.k_start[r]), int(step.k_end[r])
        n = lens[r]
        if n == 0:
            continue
        tok = (idx[r, :n].to(torch.int64)[:, None] * B + ks % B + torch.arange(B, device=idx.device)).flatten()
        valid = tok < ke
        want = REF.scores(qd[r : r + 1], step.weights[c, r : r + 1], kd[tok[valid]])[0]
        p99, mx = REF.row_error(logits[r, : tok.numel()][valid], want)
        worst = (max(worst[0], p99), max(worst[1], mx))
    rec.metrics[f"{tag}logit_p99"], rec.metrics[f"{tag}logit_max"] = worst
    if not (worst[0] < REF.LOGIT_P99 and worst[1] < REF.LOGIT_MAX):
        rec.fail(f"{tag}: logits p99 {worst[0]:.4g}, max {worst[1]:.4g} (allowance < 0.01, < 0.02)")


def packed(device) -> list[dict]:
    """Packed score mathematics, offsets and unaligned starts, candidate expansion, selection and remap, strided
    views, the compute schedule's row counts and guards, through the entry point and the four operations."""
    dep = candidate(C.load_dependency)
    out = []

    def coordinates(rec: Record, L_case: int):
        step = packed_step(f"packed-coord-{L_case}", [[0], [9], [513], [L_case]], [0, 1, 5, 7], device,
                           candidate_stride=NC + 256)
        o, _ = run_prefill(step, rec)
        rec.checks.append("residues 0,1,5,7; lengths 0,9,513,%d; strided candidate view" % L_case)
        check_packed_step(rec, step, o, "")
        for c in range(NCONS):
            _logit_check(rec, step, dep, c, f"c{c} ")
    out.append(_run("packed", "packed-coord-20480", lambda r: coordinates(r, 20480)))
    out.append(_run("packed", "packed-coord-32768", lambda r: coordinates(r, 32768)))

    for R in PACKED_R:
        L = 513 if R not in (33, 128, 512) else (20480 if R == 33 else 32768)
        def sched(rec: Record, R=R, L=L):
            per = [max(1, R // 4)] * 4
            per[-1] += R - sum(per)
            per = [x for x in per if x > 0]
            vis = [[L - (n - 1 - j) for j in range(n)] for n in per]
            step = packed_step(f"packed-sched-{R}-{L}", vis, [0, 1, 5, 7], device)
            o, _ = run_prefill(step, rec)
            rec.checks.append(f"R={R}, L={L}")
            sample = sorted(set([0, step.rows - 1] + list(range(0, step.rows, max(1, step.rows // 24)))))
            sub = C.PackedStep(q=step.q[:, sample].contiguous(), q_scale=step.q_scale[:, sample].contiguous(),
                               k_quant=step.k_quant, k_scale=step.k_scale, weights=step.weights[:, sample].contiguous(),
                               k_start=step.k_start[sample], k_end=step.k_end[sample],
                               candidate_blocks=step.candidate_blocks[sample])
            check_packed_step(rec, sub, o[:, sample], "")
            _logit_check(rec, sub, dep, 0, "c0 ")
        out.append(_run("packed", f"packed-sched-R{R}-L{L}", sched))


    def bounds(rec: Record):
        """Empty and padded rows, a partial newest block, output padding, guarded outputs and neighbouring storage."""
        step = packed_step("packed-bounds", [[0, 3, 8, 13], [1, 511, 519]], [3, 6], device)
        gout, flat, pad = guarded(C.output_buffer(step.rows, device))
        prefill, _ = candidate(C.load_entry_points)
        gk, kflat, kpad = guarded(step.k_quant)
        step.k_quant = gk
        candidate(C.run_prefill, prefill, step, gout)
        rec.checks.append("empty and partial rows; guards around the outputs and the keys")
        if not guards_intact(flat, pad):
            rec.fail("the entry point wrote outside its output buffer")
        if not guards_intact(kflat, kpad):
            rec.fail("the entry point wrote into its key workspace's neighbouring storage")
        check_packed_step(rec, step, gout, "")
        empty_batch = C.PackedStep(q=step.q[:, :0], q_scale=step.q_scale[:, :0], k_quant=step.k_quant,
                                   k_scale=step.k_scale, weights=step.weights[:, :0], k_start=step.k_start[:0],
                                   k_end=step.k_end[:0], candidate_blocks=step.candidate_blocks[:0])
        candidate(C.run_prefill, prefill, empty_batch, C.output_buffer(0, device))
        rec.checks.append("an empty batch")
    out.append(_run("packed", "packed-bounds", bounds))
    return out


# ── paged correctness ─────────────────────────────────────────────────────────────────────────────────────────


def paged(device) -> list[dict]:
    """The fused page layout, physical pages permuted with unused holes and a wider page stride, per-row visibility and
    logical positions, through the entry point and the paged operations."""
    dep = candidate(C.load_dependency)
    out = []
    for page_kv in (64, 128):
        def storage(rec: Record, page_kv=page_kv):
            reqs = [[700], [1100, 1101], [5000], [20480]]
            step, st = paged_step(f"paged-storage-{page_kv}", reqs, page_kv, device, holes=5, extra_stride=512)
            o, _ = run_decode(step, rec)
            rec.checks.append(f"page_kv {page_kv}; permuted pages with holes; stride round_up(page_bytes, 512) + 512")
            check_paged_step(rec, step, o, "")
            rows, lens = [], []
            for r in range(step.rows):
                pool = step.candidate_blocks[r]
                v = int(step.context_lens[r])
                valid = torch.sort(pool[(pool >= 0) & (pool < math.ceil(v / B))].to(torch.int64)).values
                row = torch.full((NC,), int(valid[-1]) if valid.numel() else 0, dtype=torch.int32, device=device)
                row[: valid.numel()] = valid.to(torch.int32)
                rows.append(row)
                lens.append(valid.numel())
            idx = torch.stack(rows)
            meta = candidate(dep["get_paged_sparse_mqa_logits_metadata"], step.context_lens, step.block_table,
                             step.row_indices, page_kv, idx, torch.int8, C.SPARSE_BLOCK_KV)
            logits = candidate(dep["fp8_fp4_paged_sparse_mqa_logits"], (step.q[0], step.q_scale[0]), step.kv_cache,
                               step.weights[0], meta, NC, C.SPARSE_BLOCK_KV)
            worst = (0.0, 0.0)
            qd = REF.dequantize(step.q[0, :, 0], step.q_scale[0, :, 0])
            for r in range(step.rows):
                v = int(step.context_lens[r])
                values, words = REF.page_tokens(step.kv_cache, step.block_table[r], v)
                kd = REF.dequantize(values, words)
                tok = (idx[r, : lens[r]].to(torch.int64)[:, None] * B + torch.arange(B, device=device)).flatten()
                ok = tok < v
                want = REF.scores(qd[r : r + 1], step.weights[0, r : r + 1], kd[tok[ok]])[0]
                p99, mx = REF.row_error(logits[r, : tok.numel()][ok], want)
                worst = (max(worst[0], p99), max(worst[1], mx))
            rec.metrics["logit_p99"], rec.metrics["logit_max"] = worst
            if not (worst[0] < REF.LOGIT_P99 and worst[1] < REF.LOGIT_MAX):
                rec.fail(f"paged logits p99 {worst[0]:.4g}, max {worst[1]:.4g}")
        out.append(_run("paged", f"paged-storage-{page_kv}", storage))
    return out


# ── replay, speculative rows and lifecycle ─────────────────────────────────────────────────────────────────────


def replay_and_spec(device) -> list[dict]:
    """Verification groups of one to six rows with per-row lengths L + i and their own pools, request order changed
    between steps, the accepted-history boundary with a poisoned rejected tail, a changed next step and a reused slot,
    and CUDA-graph replay of both entry points on changed buffers."""
    out = []
    for L in (511, 16383, 20475):
        def groups(rec: Record, L=L):
            reqs = [[L + i for i in range(n)] for n in range(1, 7)]
            step, _ = paged_step(f"spec-groups-{L}", reqs, 64, device)
            o, ws = run_decode(step, rec)
            rec.checks.append(f"groups 1..6 at L={L}")
            check_paged_step(rec, step, o, "step1 ")
            perm = torch.randperm(step.rows, generator=generator(f"spec-order-{L}")).to(device)
            nxt = C.PagedStep(q=step.q[:, perm].contiguous(), q_scale=step.q_scale[:, perm].contiguous(),
                              kv_cache=step.kv_cache, weights=step.weights[:, perm].contiguous(),
                              context_lens=step.context_lens[perm].contiguous(),
                              block_table=step.block_table[perm].contiguous(),
                              row_indices=step.row_indices[perm].contiguous(),
                              candidate_blocks=step.candidate_blocks[perm].contiguous())
            o2, _ = run_decode(nxt, rec, workspace=ws)
            rec.checks.append("next step: rows reordered and interleaved; row order restored in the output")
            check_paged_step(rec, nxt, o2, "step2 ")
        out.append(_run("replay_spec", f"spec-groups-L{L}", groups))

    for a in range(6):
        def accepted(rec: Record, a=a):
            # one request's six verification rows at step 1 (lengths L..L+5, all visible keys valid); `a` of its five
            # drafts accepted, so step 2's row sees L+a+1 positions -- and the rejected tail beyond them is rewritten
            # with keys every consumer would prefer. Only the accepted prefix may be selected.
            L = 4095
            g = generator(f"spec-accept-{a}")
            pattern = torch.rand(D, generator=g) < 0.5
            page_kv = 64
            n_pages = math.ceil((L + 6) / page_kv)
            vals, words = mxfp4(g, (n_pages * page_kv,))
            step1, st = paged_step(f"spec-accept-{a}-verify", [[L + i for i in range(6)]], page_kv, device,
                                   keys=[(vals.view(n_pages, page_kv, PB), words.view(n_pages, page_kv))])
            o1, ws = run_decode(step1, rec)
            rec.checks.append("step 1: six verification rows with lengths L..L+5")
            check_paged_step(rec, step1, o1, "step1 ")
            visible = L + a + 1
            tail = list(range(visible, L + 6))
            if tail:
                tv, tw = aligned(g, len(tail), pattern)
                page_stride = step1.kv_cache.stride(0)
                table = step1.block_table[0].tolist()
                for n, pos in enumerate(tail):
                    page, slot = table[pos // page_kv], pos % page_kv
                    st[page, slot * PB : (slot + 1) * PB] = tv[n].view(torch.uint8).to(device)
                    st[page, page_kv * PB + slot * 4 : page_kv * PB + slot * 4 + 4] = tw[n : n + 1].view(torch.uint8).to(device)
            step2 = C.PagedStep(q=step1.q[:, :1].clone(), q_scale=step1.q_scale[:, :1].clone(), kv_cache=step1.kv_cache,
                                weights=step1.weights[:, :1].clone(),
                                context_lens=torch.tensor([visible], dtype=torch.int32, device=device),
                                block_table=step1.block_table[:1].clone(), row_indices=step1.row_indices[:1].clone(),
                                candidate_blocks=step1.candidate_blocks[:1].clone())
            for c in range(NCONS):
                qa, qsa = aligned(g, H, pattern, magnitude_codes=(4, 7), scale_byte=127)
                step2.q[c, 0, 0], step2.q_scale[c, 0, 0] = qa.to(device), qsa.to(device)
            nb = math.ceil((L + 6) / B)
            pool = torch.full((NC,), -1, dtype=torch.int32)
            pool[:nb] = torch.randperm(nb, generator=g).to(torch.int32)  # a pool that includes the tail's blocks
            step2.candidate_blocks.copy_(pool.view(1, -1).to(device))
            o2, _ = run_decode(step2, rec, workspace=ws)
            rec.checks.append(f"step 2: {a} accepted; the poisoned rejected tail never selected")
            check_paged_step(rec, step2, o2, "step2 ")
            if tail:
                sel = o2[:, 0][o2[:, 0] >= 0].to(torch.int64)
                if torch.isin(sel, torch.tensor(tail, device=device)).any():
                    rec.fail("a rejected draft position was selected")
        out.append(_run("replay_spec", f"spec-accepted-{a}", accepted))

    def lifecycle(rec: Record):
        # one key cache shared by the four consumers across three steps: step 2 changes keys, pools, lengths and the
        # page table; step 3 gives request 0's slot to an unrelated request with other keys -- the workspace of each
        # step passed to the next
        step, st = paged_step("lifecycle-1", [[900], [1300, 1301]], 64, device, holes=8)
        o, ws = run_decode(step, rec)
        check_paged_step(rec, step, o, "step1 ")
        g = generator("lifecycle-2")
        n_pages = st.shape[0]
        vals, words = mxfp4(g, (n_pages, 64))
        st[:, : 64 * PB] = vals.view(torch.uint8).view(n_pages, -1).to(device)
        st[:, 64 * PB : 64 * C.PAGE_RECORD_BYTES] = words.view(torch.uint8).view(n_pages, -1).to(device)
        perm = torch.randperm(n_pages, generator=g).to(torch.int32)
        assign_tables(step, perm, [math.ceil(950 / 64), math.ceil(1341 / 64)], [1, 2], device)
        step.context_lens.copy_(torch.tensor([950, 1340, 1341], dtype=torch.int32, device=device))
        step.candidate_blocks.copy_(torch.stack([pool_for(g, v) for v in (950, 1340, 1341)]).to(device))
        o2, ws = run_decode(step, rec, workspace=ws)
        check_paged_step(rec, step, o2, "step2 ")
        g3 = generator("lifecycle-3")
        vals3, words3 = mxfp4(g3, (n_pages, 64))
        rows0 = step.block_table[0].tolist()
        for p_ in rows0:
            st[p_, : 64 * PB] = vals3[p_].view(torch.uint8).flatten().to(device)
            st[p_, 64 * PB : 64 * C.PAGE_RECORD_BYTES] = words3[p_].view(torch.uint8).flatten().to(device)
        step.context_lens[0] = 600
        step.candidate_blocks[0].copy_(pool_for(g3, 600).to(device))
        step.q.copy_(mxfp4(g3, tuple(step.q.shape[:-1]))[0].to(device))
        o3, _ = run_decode(step, rec, workspace=ws)
        check_paged_step(rec, step, o3, "step3 ")
        rec.checks.append("one shared cache over three steps: changed keys, pools, lengths, page tables and a reused slot")
    out.append(_run("replay_spec", "lifecycle", lifecycle))

    def graphs(rec: Record):
        # both entry points captured after warm-up on the capture stream; three replays on the same buffers with new
        # queries, weights, pools, lengths and page-table values. Whatever the implementation prepares per step must be
        # inside the captured work.
        prefill, decode = candidate(C.load_entry_points)
        pstep = packed_step("graph-packed", [[700, 701], [1900, 1901]], [1, 5], device)
        qstep, qst = paged_step("graph-paged", [[800], [1500, 1501]], 64, device, holes=6)
        pout, qout = C.output_buffer(pstep.rows, device), C.output_buffer(qstep.rows, device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            wsp = candidate(C.run_prefill, prefill, pstep, pout)
            wsq = candidate(C.run_decode, decode, qstep, qout)
            candidate(C.run_prefill, prefill, pstep, pout, wsp)
            candidate(C.run_decode, decode, qstep, qout, wsq)
        torch.cuda.current_stream().wait_stream(stream)
        gp, gq = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(gp, stream=stream):
                C.run_prefill(prefill, pstep, pout, workspace=wsp)
            with torch.cuda.graph(gq, stream=stream):
                C.run_decode(decode, qstep, qout, workspace=wsq)
        except Exception as exc:  # noqa: BLE001 - the implementation could not be captured
            raise CandidateError(f"CUDA-graph capture failed: {type(exc).__name__}: {str(exc)[:500]}") from exc
        for i in range(3):
            g = generator(f"graph-replay-{i}")
            n = pstep.k_quant.shape[0]
            kq, ks = mxfp4(g, (n,))
            pstep.k_quant.copy_(kq.to(device)); pstep.k_scale.copy_(ks.to(device))
            q, qs = mxfp4(g, tuple(pstep.q.shape[:-1]))
            pstep.q.copy_(q.to(device)); pstep.q_scale.copy_(qs.to(device))
            pstep.weights.copy_(weights(g, *pstep.weights.shape).to(device))
            lens = (pstep.k_end - pstep.k_start) - (3 * (i + 1))
            pstep.k_end.copy_(pstep.k_start + lens)
            pstep.candidate_blocks.copy_(torch.stack([pool_for(g, int(v)) for v in lens.tolist()]).to(device))
            npg = qst.shape[0]
            vals, words = mxfp4(g, (npg, 64))
            qst[:, : 64 * PB] = vals.view(torch.uint8).view(npg, -1).to(device)
            qst[:, 64 * PB : 64 * C.PAGE_RECORD_BYTES] = words.view(torch.uint8).view(npg, -1).to(device)
            qlens = torch.tensor([800 - 5 * (i + 1), 1500 - 2 * (i + 1), 1501 - 2 * (i + 1)], dtype=torch.int32)
            perm = torch.randperm(npg, generator=g).to(torch.int32)
            assign_tables(qstep, perm, [math.ceil(int(qlens[0]) / 64), math.ceil(int(qlens[2]) / 64)], [1, 2], device)
            qstep.context_lens.copy_(qlens.to(device))
            qstep.candidate_blocks.copy_(torch.stack([pool_for(g, int(v)) for v in qlens.tolist()]).to(device))
            q2, qs2 = mxfp4(g, tuple(qstep.q.shape[:-1]))
            qstep.q.copy_(q2.to(device)); qstep.q_scale.copy_(qs2.to(device))
            qstep.weights.copy_(weights(g, *qstep.weights.shape).to(device))
            pout.fill_(-7); qout.fill_(-7)
            candidate(gp.replay)
            candidate(gq.replay)
            check_packed_step(rec, pstep, pout, f"replay{i + 1} packed ")
            check_paged_step(rec, qstep, qout, f"replay{i + 1} paged ")
        rec.checks.append("graph capture; three replays with changed keys, queries, weights, pools, lengths and pages")
    out.append(_run("replay_spec", "graphs", graphs))
    return out


FAMILIES = {"two_level": two_level, "packed": packed, "paged": paged, "replay_spec": replay_and_spec}


def run_family(name: str, device: str = "cuda:0") -> list[dict]:
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    return FAMILIES[name](device)


def main(argv: list[str] | None = None) -> int:
    """`python -m infb_verify.sparse_cases <family> --out FILE [--device cuda:0]`: one family in its own process; the
    record is written case by case, so a crash of the implementation under test leaves the cases before it."""
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=sorted(FAMILIES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    global _SINK
    _SINK = {"path": a.out, "cases": []}
    with open(a.out, "w") as f:
        json.dump({"family": a.family, "cases": [], "complete": False}, f)
    t0 = time.perf_counter()
    records = run_family(a.family, a.device)
    with open(a.out, "w") as f:
        json.dump({"family": a.family, "cases": records, "complete": True,
                   "seconds": round(time.perf_counter() - t0, 3)}, f, indent=1)
    bad = [r for r in records if r["status"] != "pass"]
    print(f"[sparse_cases] {a.family}: {len(records)} cases, {len(bad)} not passing", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
