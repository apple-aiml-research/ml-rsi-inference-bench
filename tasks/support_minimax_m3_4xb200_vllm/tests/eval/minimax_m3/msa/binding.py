#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Kernel attribution for the MSA contract (README "What the attribution protocol observes").

1. Component side. One call of each entry point is profiled with the PyTorch profiler (CUDA activity). The entry
   point's identity set is the set of GPU kernel names launched inside that call, minus memory copies and fills
   (``Memcpy``/``Memset`` activity and PyTorch's copy and fill kernels). The two attention entry points are profiled
   once per KV-cache dtype.
2. Serving side. With the server's torch profiler armed, a fixed prompt set is served between ``/start_profile``
   and ``/stop_profile``; each tensor-parallel rank writes its own trace.
3. The observation, per rank: every entry point has at least one identity-set kernel in the trace (the attention
   sets of the deployment's KV dtype), how often the most frequent ``attend_decode`` kernel runs, and whether CUDA
   graphs are replayed (``cudaGraphLaunch``) during decode.

A static import, a registration or a kernel name printed in a log is not an observation. The grader applies its own
counts; this module reports what it saw.

  python /app/eval/minimax_m3/run.py binding --traces <dir with the ranks' traces> --output /tmp/b.json
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path

import torch

from . import check, measure

EXCLUDE = re.compile(r"(^|[^a-z])(memcpy|memset)|at::native::.*(copy|fill)|direct_copy|FillFunctor", re.IGNORECASE)
ENTRY_FAMILIES = {
    "select_prefill": ("select_prefill",),
    "select_decode": ("select_decode",),
    "attend_prefill": ("attend_prefill_bf16", "attend_prefill_fp8"),
    "attend_decode": ("attend_decode_bf16", "attend_decode_fp8"),
}
SMALL_PREFILL = {"context": 3_968, "queries": 256}
SMALL_DECODE = {"requests": 4, "seq_len": 4_224}


def kernels_of(prof) -> collections.Counter:
    names = collections.Counter()
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = ev.name
        if EXCLUDE.search(name):
            continue
        names[name] += 1
    return names


def component_identity_sets(module, seed: int, device="cuda", families=None) -> dict:
    """{family: sorted kernel names} for one profiled call of each family (after one warm call); ``families``
    restricts the families profiled (default all six)."""
    from torch.profiler import ProfilerActivity, profile

    out = {}
    for family in (f for fams in ENTRY_FAMILIES.values() for f in fams):
        if families is not None and family not in families:
            continue
        phase = measure.FAMILIES[family][0]
        kw = {"prefill": SMALL_PREFILL} if phase == "prefill" else {"decode": SMALL_DECODE}
        case = measure.workload_case(family, seed, device, **kw)
        cache, problems = check.install(module, case, device)
        if problems:
            out[family] = {"status": "candidate_fail", "structure": problems}
            continue
        check.call(module, cache, case)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            check.call(module, cache, case)
            torch.cuda.synchronize()
        out[family] = {"status": "observed", "kernels": sorted(kernels_of(prof))}
    return out


def load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def trace_counts(trace: dict) -> tuple[collections.Counter, int]:
    """(GPU kernel name counts, cudaGraphLaunch runtime calls) of one chrome trace."""
    kernels, launches = collections.Counter(), 0
    for ev in trace.get("traceEvents", []):
        cat, name = ev.get("cat", ""), ev.get("name", "")
        if cat == "kernel" and not EXCLUDE.search(name):
            kernels[name] += 1
        elif cat in ("cuda_runtime", "cuda_driver") and name.startswith(("cudaGraphLaunch", "cuGraphLaunch")):
            launches += 1
    return kernels, launches


def observe_rank(identity: dict, trace: dict, kv: str) -> dict:
    kernels, launches = trace_counts(trace)
    fams = {"select_prefill": "select_prefill", "select_decode": "select_decode",
            "attend_prefill": f"attend_prefill_{kv}", "attend_decode": f"attend_decode_{kv}"}
    present = {}
    for entry, fam in fams.items():
        ids = set(identity.get(fam, {}).get("kernels", []))
        present[entry] = sorted(k for k in ids if kernels.get(k))
    decode_ids = identity.get(fams["attend_decode"], {}).get("kernels", [])
    top = max(decode_ids, key=lambda k: kernels.get(k, 0), default=None)
    return {"entry_points_observed": {e: bool(v) for e, v in present.items()}, "observed_kernels": present,
            "attend_decode_top_kernel": top, "attend_decode_top_count": kernels.get(top, 0) if top else 0,
            "cuda_graph_launches": launches, "trace_kernels": len(kernels)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contract", type=Path, default=check.CONTRACT)
    ap.add_argument("--seed", type=int, default=measure.PUBLIC_SEED)
    ap.add_argument("--traces", type=Path, help="a directory of the ranks' torch-profiler traces")
    ap.add_argument("--kv", choices=("bf16", "fp8"), default="bf16")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    identity = component_identity_sets(check.load_contract(args.contract), args.seed)
    out: dict = {"identity_sets": identity}
    if args.traces:
        files = sorted(p for p in args.traces.rglob("*") if p.name.endswith((".json", ".json.gz")))
        out["ranks"] = {str(p): observe_rank(identity, load_trace(p), args.kv) for p in files}
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v.get("kernels", v.get("status")) for k, v in identity.items()}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
