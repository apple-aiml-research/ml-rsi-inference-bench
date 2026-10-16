#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The MSA timing protocol (README "Efficiency of the operators"), identical for every implementation measured.

Six families at fixed long-context workloads, each on one B200 with no model server resident:

  prefill (one request, 8,192 queries at positions 122,880-131,071, so 1,024 visible blocks at the last query):
      select_prefill, attend_prefill_bf16, attend_prefill_fp8
  decode (16 requests, one query each at position 65,535):
      select_decode, attend_decode_bf16, attend_decode_fp8

Inputs are N(0,1) values clamped to [-3, 3] (index queries and keys; queries, keys and values) from a seed, on a page
pool 1.25x the pages needed with a seeded-permutation block table; the attention families receive the FP64
reference selection of the same inputs. Protocol: 10 untimed calls, then 3 draws of 20 (prefill) or 100 (decode)
back-to-back calls; per-call GPU time from CUDA events around each draw's loop, per-call synchronized wall time from
the host clock around the same loop; the statistic is the median of the 3 draws, for each clock. Compilation,
warm-up and allocation count toward the operation's time limit, not toward the statistic.

  python /app/eval/minimax_m3/run.py msa-measure --family attend_decode_fp8 --output /tmp/t.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from . import check
from . import reference as ref

FAMILIES = {
    "select_prefill": ("prefill", "select", torch.bfloat16),
    "attend_prefill_bf16": ("prefill", "attend", torch.bfloat16),
    "attend_prefill_fp8": ("prefill", "attend", torch.float8_e4m3fn),
    "select_decode": ("decode", "select", torch.bfloat16),
    "attend_decode_bf16": ("decode", "attend", torch.bfloat16),
    "attend_decode_fp8": ("decode", "attend", torch.float8_e4m3fn),
}
WARMUP = 10
DRAWS = 3
CALLS = {"prefill": 20, "decode": 100}
PREFILL = {"context": 122_880, "queries": 8_192}
DECODE = {"requests": 16, "seq_len": 65_536}
PUBLIC_SEED = 20260629


def workload_case(family: str, seed: int, device="cuda", *, prefill=None, decode=None) -> check.Case:
    """The family's case: geometry, values from ``seed``, and for attention the FP64 reference selection."""
    phase, kind, kv_dtype = FAMILIES[family]
    prefill = prefill or PREFILL
    decode = decode or DECODE
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if phase == "prefill":
        seq = [prefill["context"] + prefill["queries"]]
        rows = prefill["queries"]
    else:
        seq = [decode["seq_len"]] * decode["requests"]
        rows = decode["requests"]
    pages_per = [(s + ref.BLOCK - 1) // ref.BLOCK for s in seq]
    pool = int(sum(pages_per) * 1.25) + 1
    perm = torch.randperm(pool, generator=gen).to(torch.int32)
    table = torch.zeros((len(seq), max(pages_per)), dtype=torch.int32)
    used = 0
    for r, n in enumerate(pages_per):
        table[r, :n] = perm[used:used + n]
        used += n

    def randn(shape):
        return torch.randn(shape, generator=gen).clamp_(-3, 3).to(torch.bfloat16)

    k = randn((pool, ref.BLOCK, ref.HEAD_DIM))
    v = randn((pool, ref.BLOCK, ref.HEAD_DIM))
    ik = randn((pool, ref.BLOCK, ref.HEAD_DIM))
    iq = randn((rows, ref.HEAD_DIM))
    q = randn((rows, ref.NUM_HEADS, ref.HEAD_DIM))
    common = dict(num_pages=pool, block_table=table, seq_lens=torch.tensor(seq, dtype=torch.int32),
                  k_pages=k.to(kv_dtype), v_pages=v.to(kv_dtype), index_k_pages=ik)
    if phase == "prefill":
        common.update(query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
                      context_lens=torch.tensor([prefill["context"]], dtype=torch.int32))
    op = f"{kind}_{phase}"
    sel_case = check.Case(case_id=f"{family}/select", op=f"select_{phase}", kv_dtype=kv_dtype, index_q=iq,
                          **common).to(device)
    if kind == "select":
        return sel_case
    selection = check.reference_selection(sel_case)
    return check.Case(case_id=family, op=op, kv_dtype=kv_dtype, q=q, selection=selection,
                      **common).to(device)


def time_calls(fn, calls: int) -> tuple[float, float]:
    """(per-call GPU ms, per-call synchronized wall ms) over ``calls`` back-to-back calls."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    for _ in range(calls):
        fn()
    end.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3 / calls
    return start.elapsed_time(end) / calls, wall


def measure_case(module, case: check.Case, calls: int, device="cuda") -> dict:
    t0 = time.monotonic()
    cache, problems = check.install(module, case, device)
    if problems:
        return {"status": "candidate_fail", "structure": problems}
    fn = lambda: check.call(module, cache, case)  # noqa: E731
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    setup_s = time.monotonic() - t0
    gpu, wall = [], []
    for _ in range(DRAWS):
        g, w = time_calls(fn, calls)
        gpu.append(g)
        wall.append(w)
    return {"status": "measured", "calls_per_draw": calls, "warmup_calls": WARMUP, "gpu_ms_per_call": gpu,
            "wall_ms_per_call": wall, "gpu_ms_median": statistics.median(gpu),
            "wall_ms_median": statistics.median(wall), "setup_s": round(setup_s, 3),
            "input_digest": check.case_digest(case)}


def measure_family(module, family: str, seed: int, device="cuda") -> dict:
    phase = FAMILIES[family][0]
    case = workload_case(family, seed, device)
    rec = measure_case(module, case, CALLS[phase], device)
    rec.update({"family": family, "seed_digest": hashlib.sha256(str(seed).encode()).hexdigest()[:16],
                "device": torch.cuda.get_device_name(0),
                "workload": PREFILL if phase == "prefill" else DECODE})
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=[*FAMILIES, "all"])
    ap.add_argument("--contract", type=Path, default=check.CONTRACT)
    ap.add_argument("--seed", type=int, default=PUBLIC_SEED)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    module = check.load_contract(args.contract)
    fams = list(FAMILIES) if args.family == "all" else [args.family]
    out = {"contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
           "families": [measure_family(module, f, args.seed) for f in fams],
           "note": "the candidate's own timings; grading compares them with its private comparator and bar"}
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    for r in out["families"]:
        print(f"{r['family']}: gpu {r.get('gpu_ms_median')} ms, wall {r.get('wall_ms_median')} ms per call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
