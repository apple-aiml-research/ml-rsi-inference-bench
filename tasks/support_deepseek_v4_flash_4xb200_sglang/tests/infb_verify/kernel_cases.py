#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The private component cases of the four
correctness nodes and the private timing seeds of the two efficiency nodes. Verifier-only: never agent material, never
imported by an implementation under test (the worker imports it before it loads an arm; the checker never loads one).

Seeds: SHA256("dsv4-kernel-state-20260929/<case-id>/<draw>"), the first eight bytes read little-endian,
modulo 2**63. Every input is generated on the CPU from its case's own torch.Generator; the fixture manifest
(tests/hidden/kernel_state_v49.json) freezes each case's input digest, and an input that differs from its recorded digest
is a verifier fault, never a candidate result.

Value distributions:
  attention         BF16 sparse prefill: q, kv = randn / 10 + (u - 0.5) / 10 in BF16, u one seeded uniform per tensor,
                   clamped to [-10, 10] (FlashMLA tests/lib.py 123-127 at 9241ae3). Packed decode: key values
                   randn / 10 clamped to [-1, 1] (lib.py 261-262), q randn clamped to [-1, 1] (lib.py 232-233). Softmax
                   scale 512 ** -0.5, the model's. The "distinct seven-group scales" are
                   made by multiplying each packed entry's seven NoPE groups by a per-entry permutation of 2 ** (-3..3).
  compression      normal clamped to [-3, 3]; the score channels get + 0.1 * request + 0.01 * (coordinate % 7).
  store            shuffled integers from [-m, m] (both extrema in every group) times 2 ** e, m cycling
                   3/5/7 and e cycling -3..3 by quantization group; a separate all-zero group (the scale floor).

The matrices are listed by `cases(node)`; `inputs(case)` builds a case's tensors.
    python kernel_cases.py --manifest <path>          # write the fixture manifest (on the task image, whose torch it records)
    python kernel_cases.py --check-manifest <path>    # exit 1 unless every generated digest matches
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

SEED_LABEL = "dsv4-kernel-state-20260929"
SCALE = 512 ** -0.5
H, D = 64, 512
D_NOPE, D_ROPE, GROUP, N_GROUPS = 448, 64, 64, 7
RING = {4: 16, 128: 256}
SWA_PAGE = 256
FAMILIES = {"c4-512": (4, 512, "flashmla", 64), "c4-128": (4, 128, "indexer", 64), "c128-512": (128, 512, "flashmla", 2)}

NODE_BF16 = "test_sparse_bf16_correctness"
NODE_PACKED = "test_sparse_paged_correctness"
NODE_STATE = "test_compression_state_correctness"
NODE_TRANSFORM = "test_cache_transform_correctness"
NODE_PREFILL = "test_kernel_prefill_efficiency"
NODE_DECODE = "test_kernel_decode_efficiency"
CORRECTNESS_NODES = (NODE_BF16, NODE_PACKED, NODE_STATE, NODE_TRANSFORM)
EFFICIENCY_NODES = {NODE_PREFILL: "prefill", NODE_DECODE: "decode"}

PACKED_CONFIGS = {"main": (256, 128, None, None), "c4-64": (256, 128, 64, 64), "c4-512": (256, 128, 64, 512),
                  "c128-64": (256, 128, 2, 64), "c128-512": (256, 128, 2, 512)}
PACKED_BSQ = ((1, 1), (7, 1), (1, 4), (3, 4), (129, 1))
MAIN_SLOTS, EXTRA_SLOTS = 4096, 2048
LENGTH_CYCLE = (0, 1, 63, 64, 65, None)
BF16_PATTERNS = ("shuffled", "sorted", "holes", "repeats", "tail", "short")


@dataclass(frozen=True)
class Case:
    id: str
    node: str
    kind: str
    p: dict = field(default_factory=dict, hash=False, compare=False)


def seed_of(case_id: str, draw: int = 0) -> int:
    h = hashlib.sha256(f"{SEED_LABEL}/{case_id}/{draw}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (1 << 63)


def generator(case_id: str, draw: int = 0) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed_of(case_id, draw))
    return g


def timing_seed(phase: str, family: str) -> int:
    """The private seed of one efficiency family's timing inputs (both arms receive the same inputs)."""
    return seed_of(f"timing/{phase}/{family}")


def _uniform(g) -> float:
    return float(torch.rand((), generator=g))


def _randint(lo: int, hi: int, g) -> int:
    """An integer in [lo, hi)."""
    return int(torch.randint(lo, hi, (), generator=g))




def cases(node: str) -> list[Case]:
    out: list[Case] = []
    if node == NODE_BF16:
        for q in (1, 17, 129):
            for k in (64, 256, 512):
                out.append(Case(f"bf16/q{q}-k{k}", node, "bf16", {"q": q, "k": k, "kv_len": 1024, "offset": len(out)}))
        out.append(Case("bf16/empty", node, "bf16", {"q": 17, "k": 64, "kv_len": 1024, "offset": 0, "empty": True}))
    elif node == NODE_PACKED:
        for b, sq in PACKED_BSQ:
            for cfg in PACKED_CONFIGS:
                out.append(Case(f"packed/b{b}-s{sq}/{cfg}", node, "packed", {"b": b, "sq": sq, "cfg": cfg,
                                                                              "offset": len(out)}))
        out.append(Case("/".join(("packed", "long-c128")), node, "packed", {"b": 7, "sq": 1, "cfg": "long-c128"}))
        out.append(Case("packed/empty", node, "packed", {"b": 3, "sq": 4, "cfg": "c4-64", "offset": 0, "empty": True}))
        out.append(Case("packed/sink", node, "packed", {"b": 7, "sq": 1, "cfg": "c4-512", "offset": 3, "sink": True}))
    elif node == NODE_STATE:
        for fam, (r, hd, _, _) in FAMILIES.items():
            prefixes = (r - 1, r, r + 1, 2 * r - 1)
            for i, pre in enumerate(prefixes):
                for ext in (1, 3, r + 1):
                    cid = f"state/{fam}/p{pre}-e{ext}"
                    three = (pre, prefixes[(i + 1) % 4], prefixes[(i + 2) % 4])
                    out.append(Case(cid, node, "state", {"family": fam, "prefixes": three, "extend": ext,
                                                         "decode_steps": r + 2}))
        for fam, (r, hd, _, _) in FAMILIES.items():
            for pre in (r - 1, 2 * r - 1):
                for acc in range(5):
                    cid = f"branch/{fam}/p{pre}-a{acc}"
                    out.append(Case(cid, node, "branch", {"family": fam, "prefix": pre, "accepted": acc,
                                                          "suffix": 4, "decode_steps": r + 2}))
    elif node == NODE_TRANSFORM:
        for fam in FAMILIES:
            for phase in ("prefill", "decode"):
                out.append(Case(f"normrope/{fam}/{phase}", node, "normrope", {"family": fam, "phase": phase}))
        for name, (typ, page) in {"flashmla-p64": ("flashmla", 64), "flashmla-p2": ("flashmla", 2),
                                  "indexer-p64": ("indexer", 64)}.items():
            out.append(Case(f"store/{name}", node, "store", {"type": typ, "page": page}))
        for fam in FAMILIES:
            out.append(Case(f"compose/{fam}", node, "compose", {"family": fam}))
    else:
        raise ValueError(f"no correctness cases for node {node!r}")
    return out


def all_cases() -> list[Case]:
    return [c for n in CORRECTNESS_NODES for c in cases(n)]


def case_by_id(case_id: str) -> Case:
    for c in all_cases():
        if c.id == case_id:
            return c
    raise KeyError(case_id)




def _bf16_c18(shape, g) -> torch.Tensor:
    """FlashMLA's sparse-prefill test values (lib.py 123-127): BF16 randn / 10 + (u - 0.5) / 10, clamped to [-10, 10]."""
    t = torch.randn(*shape, generator=g, dtype=torch.bfloat16) / 10 + (_uniform(g) - 0.5) / 10
    return t.clamp_(-10, 10)


def _bf16_row(pattern: str, k: int, kv_len: int, g) -> tuple[torch.Tensor, int]:
    """One query row's selected list (int32 [k]) and its valid length, by pattern."""
    perm = torch.randperm(kv_len, generator=g)[:k].to(torch.int32)
    length = k
    if pattern == "sorted":
        perm = perm.sort().values
    elif pattern == "holes":
        mask = torch.rand(k, generator=g) < 0.125
        mask[_randint(0, k, g)] = True
        perm[mask] = -1
    elif pattern == "repeats":
        pool = perm[: max(1, k // 4)]
        perm = pool[torch.randint(0, pool.numel(), (k,), generator=g)]
        if pool.numel() > 1 and k > 1:
            perm[1] = perm[0]
    elif pattern == "tail":
        perm[k - max(1, k // 4):] = -1
    elif pattern == "short":
        length = _randint(1, k, g) if k > 1 else 1
    return perm, length


def _bf16_inputs(case: Case) -> dict:
    p = case.p
    g = generator(case.id)
    q, k, kv_len = p["q"], p["k"], p["kv_len"]
    t = {"q": _bf16_c18((q, H, D), g), "kv": _bf16_c18((kv_len, 1, D), g)}
    idx = torch.empty(q, 1, k, dtype=torch.int32)
    lengths = torch.empty(q, dtype=torch.int32)
    for i in range(q):
        if p.get("empty"):
            row = torch.randperm(kv_len, generator=g)[:k].to(torch.int32)
            if i % 2 == 0:
                row[:] = -1
                length = k
            else:
                length = 0
        else:
            row, length = _bf16_row(BF16_PATTERNS[(i + p["offset"]) % len(BF16_PATTERNS)], k, kv_len, g)
        idx[i, 0] = row
        lengths[i] = length
    t["indices"] = idx
    t["topk_length"] = lengths
    t["attn_sink"] = torch.tensor([(-8.0, 0.0, 8.0)[h % 3] for h in range(H)], dtype=torch.float32)
    return t


def _pack_model1(values: torch.Tensor, raw: torch.Tensor, page: int, phys: torch.Tensor) -> None:
    """MODEL1 bytes of float values [n, 512] at physical token indices, FlashMLA's test-data convention (tests/quant.py):
    per group scale = 2 ** ceil(log2(max(max|x| / 448, 1e-4))), E4M3 values, rotary coordinates BF16. Builds legal
    attention inputs; it is not the engine's cache-store rule (F 4.2)."""
    v = values.to(torch.float32)
    n = v.shape[0]
    grp = v[:, :D_NOPE].reshape(n, N_GROUPS, GROUP)
    s = torch.pow(2.0, torch.ceil(torch.log2(torch.clamp_min(grp.abs().amax(-1) / 448.0, 1e-4))))
    q8 = (grp / s.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8).reshape(n, D_NOPE)
    rope = v[:, D_NOPE:].to(torch.bfloat16).contiguous().view(torch.uint8).reshape(n, 2 * D_ROPE)
    sbytes = (torch.log2(s).round().to(torch.int64) + 127).to(torch.uint8)
    pg, off = phys // page, phys % page
    cols = torch.arange(576)
    raw[pg.unsqueeze(1), (off * 576).unsqueeze(1) + cols.unsqueeze(0)] = torch.cat([q8, rope], dim=-1)
    scol = torch.arange(N_GROUPS)
    raw[pg.unsqueeze(1), (576 * page + off * 8).unsqueeze(1) + scol.unsqueeze(0)] = sbytes


def _packed_values(n: int, g) -> torch.Tensor:
    """Packed key values with distinct seven-group scales: randn / 10 clamped to [-1, 1], each entry's seven NoPE groups
    multiplied by its own permutation of 2 ** (-3..3)."""
    v = (torch.randn(n, D, generator=g) / 10).clamp_(-1, 1)
    exps = torch.stack([torch.randperm(N_GROUPS, generator=g) for _ in range(n)]).to(torch.float32) - 3.0
    v[:, :D_NOPE] = (v[:, :D_NOPE].view(n, N_GROUPS, GROUP) * torch.pow(2.0, exps).unsqueeze(-1)).view(n, D_NOPE)
    return v


def model1_page_bytes(page: int) -> int:
    return math.ceil(584 * page / 576) * 576


def _pool(name: str, page: int, slots: int, g, values: torch.Tensor | None = None) -> dict:
    """A packed pool: every physical slot holds a legal entry; logical slots 0..slots-1 at a random permutation of the
    non-reserved physical slots (physical 0 is the reserved padding slot)."""
    pages = math.ceil(slots / page) + 1
    nphys = pages * page
    raw = torch.zeros(pages, model1_page_bytes(page), dtype=torch.uint8)
    fill = _packed_values(nphys, g)
    phys = torch.randperm(nphys - 1, generator=g)[:slots] + 1
    if values is not None:
        fill[phys] = values
    _pack_model1(fill, raw, page, torch.arange(nphys))
    return {"name": name, "page": page, "raw": raw, "phys": phys}


def _lists(pool: dict, b: int, sq: int, cap: int, slots: int, lengths: list[int], g, holes: bool = True) -> torch.Tensor:
    """[b, sq, cap] physical indices: per query row distinct logical slots in random order; positions at or past the
    request's valid length point at the reserved padding slot 0; -1 holes inside the valid prefix (K 6.3: applied
    after packing)."""
    idx = torch.empty(b, sq, cap, dtype=torch.int32)
    for i in range(b):
        for s in range(sq):
            sel = pool["phys"][torch.randperm(slots, generator=g)[:cap]].to(torch.int32)
            ln = lengths[i]
            sel[ln:] = 0
            if holes and ln >= 2:
                mask = torch.rand(cap, generator=g) < 1.0 / 16
                mask[ln:] = False
                sel[mask] = -1
            idx[i, s] = sel
    return idx


def _packed_inputs(case: Case) -> dict:
    p = case.p
    g = generator(case.id)
    b, sq = p["b"], p["sq"]
    t: dict = {}
    if p["cfg"] == "long-c128":
        t["q"] = torch.zeros(b, sq, H, D, dtype=torch.bfloat16)
        t["q"][..., 0] = 1.0
        main = _pool("main", 256, 4096, g, values=torch.zeros(4096, D))
        extra_vals = torch.zeros(4096, D)
        order = torch.randperm(4096, generator=g)[:2048]
        extra_vals[order[512:], 1] = 1.0
        extra = _pool("extra", 2, 4096, g, values=extra_vals)
        t["main.raw"], t["extra.raw"] = main["raw"], extra["raw"]
        mains = torch.stack([main["phys"][torch.randperm(4096, generator=g)[:128]] for _ in range(b)]).to(torch.int32)
        t["main.indices"] = mains.view(b, sq, 128)
        t["main.lengths"] = torch.full((b,), 128, dtype=torch.int32)
        t["extra.indices"] = extra["phys"][order].to(torch.int32).view(1, 1, 2048).expand(b, sq, 2048).contiguous()
        t["extra.lengths"] = torch.tensor([511, 512, 513, 1023, 1024, 2047, 2048], dtype=torch.int32)
        t["attn_sink"] = torch.zeros(H, dtype=torch.float32)
        return t
    mpage, mcap, epage, ecap = PACKED_CONFIGS[p["cfg"]]
    t["q"] = torch.randn(b, sq, H, D, generator=g).clamp_(-1, 1).to(torch.bfloat16)
    if p.get("sink"):
        t["attn_sink"] = torch.tensor([(20.0, -20.0, 5.0, 0.0)[h % 4] for h in range(H)], dtype=torch.float32)
    else:
        t["attn_sink"] = torch.randn(H, generator=g).to(torch.float32)
    pools = [("main", mpage, mcap, MAIN_SLOTS)] + ([("extra", epage, ecap, EXTRA_SLOTS)] if epage else [])
    for j, (name, page, cap, slots) in enumerate(pools):
        pool = _pool(name, page, slots, g)
        cyc = [(cap if v is None else min(v, cap)) for v in LENGTH_CYCLE]
        lengths = [cyc[(i + p["offset"] + 2 * j) % len(cyc)] for i in range(b)]
        if p.get("empty"):
            lengths = [0, 0, cap][:b] + [cap] * max(0, b - 3)
        idx = _lists(pool, b, sq, cap, slots, lengths, g)
        if p.get("empty"):
            idx[2:] = -1
        t[f"{name}.raw"] = pool["raw"]
        t[f"{name}.indices"] = idx
        t[f"{name}.lengths"] = torch.tensor(lengths, dtype=torch.int32)
    return t


def packed_pages(case: Case) -> dict[str, int]:
    if case.p["cfg"] == "long-c128":
        return {"main": 256, "extra": 2}
    mpage, _, epage, _ = PACKED_CONFIGS[case.p["cfg"]]
    return {"main": mpage, **({"extra": epage} if epage else {})}




def _normal(shape, g) -> torch.Tensor:
    return torch.randn(*shape, generator=g, dtype=torch.float32).clamp_(-3, 3)


def kv_score_width(ratio: int, hd: int) -> int:
    return 2 * (2 if ratio == 4 else 1) * hd


def _kv_score(n: int, ratio: int, hd: int, request: int, g) -> torch.Tensor:
    """[n, width] projected rows; the score channels offset by 0.1 * request + 0.01 * (coordinate % 7) (K 6.3)."""
    x = _normal((n, kv_score_width(ratio, hd)), g)
    off = 0.1 * request + 0.01 * (torch.arange(hd) % 7).to(torch.float32)
    if ratio == 4:
        x[:, 2 * hd:3 * hd] += off
        x[:, 3 * hd:4 * hd] += off
    else:
        x[:, hd:2 * hd] += off
    return x


def _pages(n_req: int, per_req: int, total: int, g) -> list[list[int]]:
    """Distinct physical SWA pages 1..total-1 for each request, in a random order (page 0 stays unused)."""
    perm = (torch.randperm(total - 1, generator=g) + 1).tolist()
    return [perm[i * per_req:(i + 1) * per_req] for i in range(n_req)]


STATE_SWA_PAGES = 16
STATE_SENTINEL = (7.0, 3.0)


def _state_inputs(case: Case) -> dict:
    p = case.p
    g = generator(case.id)
    r, hd, _, _ = FAMILIES[p["family"]]
    t: dict = {"ape": _normal((r * (2 if r == 4 else 1), hd), g)}
    if case.kind == "state":
        totals = [pre + p["extend"] + p["decode_steps"] for pre in p["prefixes"]]
        for i, n in enumerate(totals):
            t[f"x{i}"] = _kv_score(n, r, hd, i, g)
        per = max(1, math.ceil(max(totals) / SWA_PAGE))
        t["pages"] = torch.tensor(_pages(3, per, STATE_SWA_PAGES, g), dtype=torch.int32)
        t["order"] = torch.randperm(3, generator=g).to(torch.int32)
    else:
        pre, acc, suf, steps = p["prefix"], p["accepted"], p["suffix"], p["decode_steps"]
        t["prefix"] = _kv_score(pre, r, hd, 0, g)
        t["suffix"] = _kv_score(suf, r, hd, 0, g)
        t["new"] = _kv_score(steps, r, hd, 0, g)
        per = max(1, math.ceil((pre + suf + steps) / SWA_PAGE))
        t["pages"] = torch.tensor(_pages(1, per, STATE_SWA_PAGES, g), dtype=torch.int32)
    return t


def state_windows(ratio: int) -> int:
    return STATE_SWA_PAGES * RING[ratio] // ratio


def state_loc(pages: list[int], pos: int, ratio: int) -> int:
    pos = max(pos, 0)
    ring = RING[ratio]
    return (pages[pos // SWA_PAGE] * ring + pos % ring) // ratio


def used_windows(pages_per_request: list[list[int]], lengths: list[int], ratio: int) -> set[int]:
    return {state_loc(pg, pos, ratio) for pg, n in zip(pages_per_request, lengths) for pos in range(n)}




ROPE_MAX = 4096 + 2 * 128 + 16
NORM_EPS = 1e-6


def rope_table(max_pos: int = ROPE_MAX, base: float = 10000.0) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, D_ROPE, 2, dtype=torch.float64) / D_ROPE))
    ang = torch.outer(torch.arange(max_pos, dtype=torch.float64), inv)
    return torch.polar(torch.ones_like(ang), ang).to(torch.complex64)


def norm_weight(hd: int) -> torch.Tensor:
    return (1.0 + 0.01 * (torch.arange(hd) % 17)).to(torch.float32)


def normrope_requests(ratio: int, phase: str) -> list[tuple[int, int]]:
    """(prefix, extend) per request for a prefill case, (seq_len, 1) per request for a decode case: positions before,
    at and after each of the first two boundaries, and a far position for the rotary table."""
    if phase == "prefill":
        return [(0, 2 * ratio + 1), (ratio - 1, ratio + 2), (4096 - 2, ratio + 3)]
    return [(n, 1) for n in (ratio - 1, ratio, ratio + 1, 2 * ratio - 1, 2 * ratio, 2 * ratio + 1, 4096 + ratio)]


def store_rows(n: int, width: int, groups: int, g) -> torch.Tensor:
    """K 6.3's exact-store probes [n, width] float32: per quantization group shuffled integers from [-m, m] with both
    extrema, times 2 ** e, m cycling 3/5/7 and e cycling -3..3 by group; row 0's first group all zero."""
    gw = width // groups
    rows = []
    for r in range(n):
        parts = []
        for gi in range(groups):
            m = (3, 5, 7)[(r + gi) % 3]
            e = (-3, -2, -1, 0, 1, 2, 3)[(r + gi) % 7]
            ints = torch.randint(-m, m + 1, (gw,), generator=g).to(torch.float32)
            ints[0], ints[1] = m, -m
            parts.append(ints[torch.randperm(gw, generator=g)] * (2.0 ** e))
        rows.append(torch.cat(parts))
    x = torch.stack(rows)
    x[0, :gw] = 0.0
    return x


STORE_SENTINEL = 0xA5
STORE_ROWS = 24
STORE_PADDING_ROWS = 3


def _store_dests(page: int, n_live: int, n_pad: int, pages: int, g) -> torch.Tensor:
    """Distinct live destinations spread over the non-reserved slots of every page but the last (a guard page),
    shuffled, then `n_pad` rows directed at the reserved padding slot 0."""
    live = torch.randperm((pages - 1) * page - 1, generator=g)[:n_live] + 1
    dest = torch.cat([live, torch.zeros(n_pad, dtype=torch.int64)])
    return dest[torch.randperm(dest.numel(), generator=g)].to(torch.int32)


def store_pool_pages(page: int) -> int:
    return max(4, math.ceil(STORE_ROWS / page) + 3)


def indexer_page_bytes(page: int) -> int:
    return (128 + 4) * page


def _transform_inputs(case: Case) -> dict:
    p = case.p
    g = generator(case.id)
    if case.kind == "normrope":
        r, hd, _, _ = FAMILIES[p["family"]]
        reqs = normrope_requests(r, p["phase"])
        n = sum(e for _, e in reqs)
        return {"x": _normal((n, hd), g)}
    if case.kind == "store":
        page = p["page"]
        pages = store_pool_pages(page)
        if p["type"] == "flashmla":
            x = store_rows(STORE_ROWS, D, N_GROUPS + 1, g)
            nbytes = model1_page_bytes(page)
        else:
            x = store_rows(STORE_ROWS, 128, 1, g)
            x[1] = 0.0
            nbytes = indexer_page_bytes(page)
        return {"x": x, "dest": _store_dests(page, STORE_ROWS - STORE_PADDING_ROWS, STORE_PADDING_ROWS, pages, g),
                "pool": torch.full((pages, nbytes), STORE_SENTINEL, dtype=torch.uint8)}
    r, hd, typ, page = FAMILIES[p["family"]]
    n0, steps = 2 * r + 3, r + 2
    total = n0 + steps
    emits = [pos for pos in range(total) if (pos + 1) % r == 0]
    pages = max(4, math.ceil(len(emits) / page) + 3)
    nbytes = model1_page_bytes(page) if typ == "flashmla" else indexer_page_bytes(page)
    live = torch.randperm((pages - 1) * page - 1, generator=g)[:len(emits)] + 1
    return {"x": _kv_score(total, r, hd, 0, g), "ape": _normal((r * (2 if r == 4 else 1), hd), g),
            "pages": torch.tensor(_pages(1, max(1, math.ceil(total / SWA_PAGE)), STATE_SWA_PAGES, g), dtype=torch.int32),
            "dest": live.to(torch.int32), "pool": torch.full((pages, nbytes), STORE_SENTINEL, dtype=torch.uint8)}


def compose_layout(family: str) -> tuple[int, int, list[int]]:
    r = FAMILIES[family][0]
    n0, steps = 2 * r + 3, r + 2
    return n0, steps, [pos for pos in range(n0 + steps) if (pos + 1) % r == 0]




def inputs(case: Case) -> dict:
    if case.kind == "bf16":
        return _bf16_inputs(case)
    if case.kind == "packed":
        return _packed_inputs(case)
    if case.kind in ("state", "branch"):
        return _state_inputs(case)
    return _transform_inputs(case)


def digest(tensors: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k].detach().contiguous().cpu()
        h.update(f"{k}|{tuple(t.shape)}|{t.dtype}|".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


FIDELITY_FIXTURES = Path(__file__).resolve().parent.parent / "hidden" / "fidelity"


def replay_schedule() -> dict:
    """The generated-history nodes' schedule (generated_replay.build_schedule over the private fidelity corpus)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import generated_replay  # noqa: PLC0415

    return generated_replay.build_schedule(FIDELITY_FIXTURES)


def manifest() -> dict:
    return {"schema": "infb-kernel-state/1", "seed_label": SEED_LABEL, "torch": torch.__version__,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": {c.id: {"node": c.node, "kind": c.kind, "digest": digest(inputs(c))} for c in all_cases()},
            "timing": {n: {fam: timing_seed(ph, fam) for fam in _timing_families(ph)} for n, ph in EFFICIENCY_NODES.items()},
            "replay": replay_schedule()}


def _timing_families(phase: str) -> list[str]:
    fams = ["bf16"] if phase == "prefill" else []
    return fams + ["packed-main", "packed-main+c4", "packed-main+c128", "c4-core", "c4-indexer", "c128"]


def check_digest(case: Case, tensors: dict, recorded: dict) -> str | None:
    """None when the case's generated inputs match the manifest, else the problem (a verifier fault)."""
    want = recorded.get("cases", {}).get(case.id, {}).get("digest")
    if want is None:
        return f"case {case.id} is not in the fixture manifest"
    got = digest(tensors)
    return None if got == want else f"case {case.id} generated inputs {got[:16]}, the manifest records {want[:16]}"


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--manifest":
        m = manifest()
        Path(argv[2]).write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
        print(f"{len(m['cases'])} cases written to {argv[2]}")
        return 0
    if len(argv) == 3 and argv[1] == "--check-manifest":
        rec = json.loads(Path(argv[2]).read_text())
        bad = [x for c in all_cases() if (x := check_digest(c, inputs(c), rec))]
        if rec.get("replay") != replay_schedule():
            bad.append("the replay schedule differs from the manifest's")
        print("\n".join(bad) if bad else f"{len(all_cases())} case inputs and the replay schedule match the manifest")
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
