#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The private MSA operator cases, regenerated at grading from fixed seeds and checked against the
input digests recorded in ``tests/hidden/msa_cases.json``. Private material: never published, never in the agent image.

Seeds: the first eight bytes of SHA256(``m3-msa-v1/<case-id>/<draw>``), little-endian, modulo 2**63. Values are
generated on the CPU and moved to the GPU. Page pools are 1.25x the pages the geometry needs, block tables a seeded
permutation of the pool, and every page no block table references holds sentinels (K/V +-96, index keys +-8 in every
dimension) so that an out-of-table read changes results.

Geometries: prefill PG1-PG16 (requests as (context, queries)), decode DG1-DG8 (seq_lens). Value families:

  S-rand     index_q, index_k ~ N(0,1) clamped to [-3, 3], BF16                        every PG and DG (selection)
  S-struct   index_q = s_t * e0, s_t in {0.5, 1, 1.5}; in block b the token at offset (37b + 11) mod 128 has
             k0 = A_b, a value of the step-0.25 grid on [-4, 4] shuffled independently per request (the grid is
             tiled when a request has more than its 33 values' worth of blocks), the block's other tokens
             k0 = A_b - 0.5 - 0.25 * ((7j) mod 8) (j the offset in the block); other dimensions N(0,1)
                                                                                   PG5 PG10 PG12 PG15 DG7
  S-tie      S-struct with, in every run of eight consecutive blocks, four sharing A_b exactly and two pairs
             differing by 1e-4 * |A_b| (stored in BF16, which rounds such a pair to a tie or one ulp apart)
                                                                                   PG10 DG7
  A-rand     q, k, v ~ N(0,1) clamped to [-3, 3], BF16 (FP8 cases cast k, v to float8_e4m3fn); selection = the
             FP64 reference selection of S-rand on the same geometry               every PG and DG, both KV dtypes
  A-randsel  A-rand values; selection = the local block plus n - 1 distinct uniformly drawn visible blocks, shuffled
                                                                                   PG10 PG15 DG7, both KV dtypes
  A-probe    q = 0; k ~ N(0,1); v[j, d] = (((5b + d) mod 9) - 4) / 2 + (((3i + d) mod 5) - 2) / 4 with b = j // 128,
             i = j mod 128 (a 1/4 grid on [-2.5, 2.5], exact in BF16 and E4M3); selection as A-rand
                                                                                   PG1 PG7 PG8 PG10 PG15 DG1 DG3 DG7
"""

from __future__ import annotations

import hashlib
import math

import torch

from minimax_m3.msa import check  # the trusted copy of the public protocol (/tests/eval/minimax_m3/msa)
from minimax_m3.msa import reference as ref

PREFILL = {
    "PG1": [(0, 1)], "PG2": [(0, 127)], "PG3": [(0, 128)], "PG4": [(0, 129)], "PG5": [(0, 2049)],
    "PG6": [(0, 4097)], "PG7": [(1, 127)], "PG8": [(2047, 2)], "PG9": [(2048, 1)], "PG10": [(2049, 300)],
    "PG11": [(16383, 257)], "PG12": [(65537, 1024)], "PG13": [(131071, 2048)], "PG14": [(260096, 2048)],
    "PG15": [(0, 129), (2050, 64), (8191, 513), (40000, 7)], "PG16": [(100000, 1000), (5, 3), (262000, 144)],
}
DECODE = {
    "DG1": [1], "DG2": [128], "DG3": [129], "DG4": [2048], "DG5": [2049], "DG6": [262144],
    "DG7": [1, 2, 127, 128, 129, 1000, 2047, 2048, 2049, 4096, 16385, 65536, 131073, 0, 0, 0],
    "DG8": [262144, 0, 262143, 5000],
}
S_STRUCT = {"PG5", "PG10", "PG12", "PG15", "DG7"}
S_TIE = {"PG10", "DG7"}
A_RANDSEL = {"PG10", "PG15", "DG7"}
A_PROBE = {"PG1", "PG7", "PG8", "PG10", "PG15", "DG1", "DG3", "DG7"}
KV = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
SENTINEL_KV, SENTINEL_INDEX = 96.0, 8.0


def seed_of(case_id: str, draw: int = 0) -> int:
    h = hashlib.sha256(f"m3-msa-v1/{case_id}/{draw}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 63)


def case_ids() -> dict[str, list[str]]:
    """Every private case id, grouped by the scored node that runs it."""
    sp = [f"{g}/S-rand" for g in PREFILL] + [f"{g}/S-struct" for g in PREFILL if g in S_STRUCT] + \
         [f"{g}/S-tie" for g in PREFILL if g in S_TIE]
    sd = [f"{g}/S-rand" for g in DECODE] + [f"{g}/S-struct" for g in DECODE if g in S_STRUCT] + \
         [f"{g}/S-tie" for g in DECODE if g in S_TIE]
    ap, ad = [], []
    for geo, out in ((PREFILL, ap), (DECODE, ad)):
        for g in geo:
            fams = ["A-rand"] + (["A-randsel"] if g in A_RANDSEL else []) + (["A-probe"] if g in A_PROBE else [])
            out += [f"{g}/{f}/{d}" for f in fams for d in KV]
    return {"test_select_prefill_correctness": sp, "test_select_decode_correctness": sd,
            "test_attend_prefill_correctness": ap, "test_attend_decode_correctness": ad}


def _layout(gen: torch.Generator, seq: list[int]):
    pages_per = [max(1, math.ceil(s / ref.BLOCK)) if s > 0 else 0 for s in seq]
    need = sum(pages_per)
    pool = max(1, math.ceil(need * 1.25))
    perm = torch.randperm(pool, generator=gen).to(torch.int32)
    table = torch.zeros((len(seq), max(1, max(pages_per))), dtype=torch.int32)
    used = 0
    referenced = torch.zeros(pool, dtype=torch.bool)
    for r, n in enumerate(pages_per):
        table[r, :n] = perm[used:used + n]
        referenced[perm[used:used + n].long()] = True
        used += n
    return pool, table, referenced


def _sentinel(gen, pool, referenced, value):
    signs = torch.where(torch.rand((pool, ref.BLOCK, ref.HEAD_DIM), generator=gen) < 0.5, -1.0, 1.0)
    return (signs * value).masked_fill(referenced[:, None, None], 0.0)


def _randn(gen, shape):
    return torch.randn(shape, generator=gen).clamp_(-3, 3)


def _struct_index_keys(gen, seq, table, pool, tie: bool):
    """Index keys of S-struct (and S-tie): dimension 0 structured per block, other dimensions N(0,1)."""
    keys = _randn(gen, (pool, ref.BLOCK, ref.HEAD_DIM))
    grid = torch.arange(-16, 17, dtype=torch.float32) * 0.25  # 33 values on [-4, 4]
    offs = torch.arange(ref.BLOCK)
    lower = 0.5 + 0.25 * ((7 * offs) % 8).float()
    for r, s in enumerate(seq):
        n_blocks = math.ceil(s / ref.BLOCK)
        if n_blocks == 0:
            continue
        reps = math.ceil(n_blocks / grid.numel())
        vals = grid.repeat(reps)[torch.randperm(grid.numel() * reps, generator=gen)][:n_blocks].clone()
        if tie:
            for g0 in range(0, n_blocks, 8):
                grp = vals[g0:g0 + 8]
                grp[1:4] = grp[0]  # four sharing A_b exactly
                for a in (4, 6):  # two pairs differing by 1e-4 * |A_b|
                    if a + 1 < grp.numel():
                        grp[a + 1] = grp[a] + 1e-4 * abs(float(grp[a]))
        for b in range(n_blocks):
            page = int(table[r, b])
            keys[page, :, 0] = vals[b] - lower
            keys[page, (37 * b + 11) % ref.BLOCK, 0] = vals[b]
    return keys


def _probe_values(seq, table, pool):
    v = torch.zeros((pool, ref.BLOCK, ref.HEAD_DIM))
    d = torch.arange(ref.HEAD_DIM)
    i = torch.arange(ref.BLOCK)
    for r, s in enumerate(seq):
        for b in range(math.ceil(s / ref.BLOCK)):
            page = int(table[r, b])
            v[page] = ((((5 * b + d) % 9) - 4) / 2.0)[None, :] + ((((3 * i[:, None] + d[None, :]) % 5) - 2) / 4.0)
    return v


def _geometry(geo: str):
    if geo in PREFILL:
        reqs = PREFILL[geo]
        seq = [c + n for c, n in reqs]
        qsl = [0]
        for _, n in reqs:
            qsl.append(qsl[-1] + n)
        return "prefill", seq, reqs, torch.tensor(qsl, dtype=torch.int32), \
            torch.tensor([c for c, _ in reqs], dtype=torch.int32), qsl[-1]
    seq = DECODE[geo]
    return "decode", seq, None, None, None, len(seq)


def make_case(case_id: str, draw: int = 0) -> check.Case:
    parts = case_id.split("/")
    geo, fam = parts[0], parts[1]
    dtype = KV[parts[2]] if len(parts) > 2 else torch.bfloat16
    phase, seq, reqs, qsl, ctx, rows = _geometry(geo)
    gen = torch.Generator().manual_seed(seed_of(case_id, draw))
    pool, table, referenced = _layout(gen, seq)
    common = dict(case_id=case_id, kv_dtype=dtype, num_pages=pool, block_table=table,
                  seq_lens=torch.tensor(seq, dtype=torch.int32), query_start_loc=qsl, context_lens=ctx)
    if fam.startswith("S-"):
        if fam == "S-rand":
            ik = _randn(gen, (pool, ref.BLOCK, ref.HEAD_DIM))
            iq = _randn(gen, (rows, ref.HEAD_DIM))
        else:
            ik = _struct_index_keys(gen, seq, table, pool, tie=(fam == "S-tie"))
            iq = torch.zeros((rows, ref.HEAD_DIM))
            iq[:, 0] = torch.tensor([0.5, 1.0, 1.5])[torch.randint(0, 3, (rows,), generator=gen)]
        ik = torch.where(referenced[:, None, None], ik, _sentinel(gen, pool, referenced, SENTINEL_INDEX))
        kv = _sentinel(gen, pool, referenced, SENTINEL_KV)
        return check.Case(op=f"select_{phase}", k_pages=kv.to(torch.bfloat16).to(dtype),
                          v_pages=kv.to(torch.bfloat16).to(dtype), index_k_pages=ik.to(torch.bfloat16),
                          index_q=iq.to(torch.bfloat16), **common)
    q = _randn(gen, (rows, ref.NUM_HEADS, ref.HEAD_DIM))
    k = _randn(gen, (pool, ref.BLOCK, ref.HEAD_DIM))
    v = _randn(gen, (pool, ref.BLOCK, ref.HEAD_DIM))
    if fam == "A-probe":
        q = torch.zeros_like(q)
        v = _probe_values(seq, table, pool)
    sk = _sentinel(gen, pool, referenced, SENTINEL_KV)
    k = torch.where(referenced[:, None, None], k, sk)
    v = torch.where(referenced[:, None, None], v, _sentinel(gen, pool, referenced, SENTINEL_KV))
    ik = torch.where(referenced[:, None, None], _randn(gen, (pool, ref.BLOCK, ref.HEAD_DIM)),
                     _sentinel(gen, pool, referenced, SENTINEL_INDEX))
    if fam == "A-randsel":
        selection = _random_selection(gen, phase, seq, reqs, rows)
    else:
        selection = reference_selection_of(f"{geo}/S-rand", draw)
    return check.Case(op=f"attend_{phase}", k_pages=k.to(torch.bfloat16).to(dtype),
                      v_pages=v.to(torch.bfloat16).to(dtype), index_k_pages=ik.to(torch.bfloat16),
                      q=q.to(torch.bfloat16), selection=selection, **common)


def _random_selection(gen, phase, seq, reqs, rows):
    sel = torch.full((rows, ref.TOPK), -1, dtype=torch.int32)
    positions = []
    if phase == "prefill":
        for c, n in reqs:
            positions += [c + i for i in range(n)]
    else:
        positions = [s - 1 if s > 0 else None for s in seq]
    for t, p in enumerate(positions):
        if p is None:
            continue
        vis = p // ref.BLOCK + 1
        n = min(ref.TOPK, vis)
        local = p // ref.BLOCK
        others = [b for b in torch.randperm(vis, generator=gen).tolist() if b != local][:n - 1]
        row = [local] + others
        order = torch.randperm(n, generator=gen).tolist()
        sel[t, :n] = torch.tensor([row[o] for o in order], dtype=torch.int32)
    return sel


_REF_SELECTION: dict[tuple[str, int], torch.Tensor] = {}


def reference_selection_of(select_case_id: str, draw: int = 0) -> torch.Tensor:
    key = (select_case_id, draw)
    if key not in _REF_SELECTION:
        case = make_case(select_case_id, draw)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _REF_SELECTION[key] = check.reference_selection(case.to(dev)).cpu()
    return _REF_SELECTION[key]
