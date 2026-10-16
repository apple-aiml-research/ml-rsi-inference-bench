#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Time the relative-attention entry point by the grading protocol (README.md, 'Efficiency').

    python measure.py --family decode_full [--adapter /app/inkling_rel_attention.py] [--output /tmp/t.json]

Families, all in the paged form (the served form), V-rand values from a public seed and finite unreferenced pages:
  prefill_full, prefill_local  one request, q_len 8,192 at positions 122,880-131,071 (seqused_k 131,072)
  decode_full, decode_local    16 requests, one query each, seqused_k 65,536
(`full`: E = 1,024 and window (None, None); `local`: E = 512 and window (511, 0).)

Protocol, identical for grading: one GPU (cuda:0) with no server resident; 10 untimed calls, then 3 draws of 20
(prefill) or 100 (decode) back-to-back calls; per draw, the per-call GPU time from CUDA events around the draw's loop
and the per-call synchronized wall time (synchronize, loop, synchronize, divided by the call count); the statistic is
the median of the 3 draws. Input construction, JIT compilation, warm-up and allocation fall outside the statistic but
inside the invocation limit (`../limits.json`, `invocation`). Grading draws the values from a private seed and applies
the same protocol to both arms; this prints measurements, not a verdict.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402
from reference import Case  # noqa: E402

FAMILIES = {
    "prefill_full": ("F", [(131072, 8192)], 20),
    "prefill_local": ("L", [(131072, 8192)], 20),
    "decode_full": ("F", [(65536, 1)] * 16, 100),
    "decode_local": ("L", [(65536, 1)] * 16, 100),
}
WARMUP = 10
DRAWS = 3


def workload(family: str, seed_label: str, device: str = "cuda:0") -> Case:
    kind, reqs, _ = FAMILIES[family]
    E = 1024 if kind == "F" else 512
    window = None if kind == "F" else 511
    gen = torch.Generator().manual_seed(check.seed_of(seed_label))
    total_q = sum(q for _, q in reqs)
    q = (4 * torch.randn(total_q, check.HEADS, check.DIM, generator=gen).clamp(-3, 3)).to(torch.bfloat16)
    bias = torch.randn(total_q, check.HEADS, E, generator=gen).clamp(-3, 3).to(torch.bfloat16)
    k_tok = [torch.randn(kl, check.KV_HEADS, check.DIM, generator=gen).clamp(-3, 3).to(torch.bfloat16) for kl, _ in reqs]
    v_tok = [torch.randn(kl, check.KV_HEADS, check.DIM, generator=gen).clamp(-3, 3).to(torch.bfloat16) for kl, _ in reqs]
    k, v, table, seqused = check.build_paged(reqs, k_tok, v_tok, gen, window)
    cu_q = torch.tensor([0] + [sum(q for _, q in reqs[: b + 1]) for b in range(len(reqs))], dtype=torch.int32)
    case = Case(q=q, k=k, v=v, rel_bias=bias, cu_seqlens_q=cu_q, max_seqlen_q=max(q for _, q in reqs),
                window_size=(511, 0) if window else (None, None), softmax_scale=1.0 / 128,
                page_table=table, seqused_k=seqused, max_seqlen_k=max(k for k, _ in reqs),
                meta={"family": family, "seed": seed_label})
    return check.to_device(case, device)


def call_args(case: Case):
    kwargs = dict(page_table=case.page_table, seqused_k=case.seqused_k, max_seqlen_k=case.max_seqlen_k, causal=True,
                  window_size=case.window_size, softmax_scale=case.softmax_scale)
    return (case.q, case.k, case.v, case.rel_bias, case.cu_seqlens_q, case.max_seqlen_q), kwargs


def draw(fn, args, kwargs, n: int) -> tuple[float, float]:
    """One draw: per-call GPU seconds (CUDA events around the loop) and per-call synchronized wall seconds."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    for _ in range(n):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / n
    return start.elapsed_time(end) / 1e3 / n, wall


def measure(fn, family: str, seed_label: str) -> dict:
    t_setup = time.monotonic()
    case = workload(family, seed_label)
    args, kwargs = call_args(case)
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    setup_s = time.monotonic() - t_setup
    n = FAMILIES[family][2]
    draws = [draw(fn, args, kwargs, n) for _ in range(DRAWS)]
    return {"family": family, "seed": seed_label, "calls_per_draw": n, "warmup_calls": WARMUP,
            "gpu_s_per_call": [g for g, _ in draws], "wall_s_per_call": [w for _, w in draws],
            "gpu_s_median": statistics.median(g for g, _ in draws), "wall_s_median": statistics.median(w for _, w in draws),
            "setup_and_warmup_s": round(setup_s, 3), "device": torch.cuda.get_device_name(0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", action="append", choices=sorted(FAMILIES), help="repeatable; default all four")
    ap.add_argument("--adapter", default="/app/inkling_rel_attention.py")
    ap.add_argument("--seed", default="inkling-rel-v1-public-timing", help="public seed label prefix")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    fn = check.load_candidate(args.adapter)
    results = []
    for family in args.family or sorted(FAMILIES):
        rec = measure(fn, family, f"{args.seed}/{family}")
        results.append(rec)
        print(json.dumps(rec), flush=True)
    if args.output:
        args.output.write_text(json.dumps({"adapter": args.adapter, "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
