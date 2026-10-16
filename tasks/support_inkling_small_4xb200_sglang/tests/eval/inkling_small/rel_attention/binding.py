#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Record which GPU kernels the relative-attention entry point launches (README.md, 'Attribution').

    python binding.py --output /tmp/ids.json [--adapter /app/inkling_rel_attention.py]
    python binding.py --serving-trace-dir <dir with the four ranks' traces> --identity /tmp/ids.json

Component side: one ragged prefill call, one paged prefill call and one paged decode call per layer kind run under the
PyTorch profiler (CUDA activities) after one untimed warm-up call each. Four identity sets result: `prefill_full` and
`prefill_local` (the kernel names launched inside that kind's two prefill calls, united), `decode_full` and
`decode_local`, each minus memcpy/memset and PyTorch copy/fill kernels.

Serving side (`--serving-trace-dir`): reads the torch-profiler traces a running server wrote between `/start_profile`
and `/stop_profile` (`serving/witness.py` drives that workload) and counts, per rank, the kernels of each identity set
and `cudaGraphLaunch`. Grading observes the served model this way on all four ranks and requires, on every rank: each
prefill identity set present; the most frequent `decode_full` kernel at least 7 x 31 times and the most frequent
`decode_local` kernel at least 35 x 31 times (42 x 31 if they are the same kernel); `cudaGraphLaunch` during decode.
This prints the observations, not a verdict.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402

EXCLUDED = re.compile(r"memcpy|memset|Memcpy|Memset|copy_kernel|CopyKernel|fill_kernel|FillFunctor|elementwise_kernel"
                      r".*(copy|fill)|direct_copy", re.I)
CALLS = {
    "prefill": [("ragged", [(2048, 2048), (700, 188)]), ("paged", [(4224, 129), (3000, 3000)])],
    "decode": [("paged", [(5000, 1), (1025, 1), (129, 1)])],
}


def spec(form: str, kind: str, reqs, label: str) -> dict:
    return {"id": f"binding-{label}", "kind": kind, "form": form, "requests": [list(r) for r in reqs]}


def kernels_of(fn, case) -> set[str]:
    kwargs = dict(causal=True, window_size=case.window_size, softmax_scale=case.softmax_scale)
    if case.paged:
        kwargs.update(page_table=case.page_table, seqused_k=case.seqused_k, max_seqlen_k=case.max_seqlen_k)
    else:
        kwargs.update(cu_seqlens_k=case.cu_seqlens_k, max_seqlen_k=case.max_seqlen_k)
    args = (case.q, case.k, case.v, case.rel_bias, case.cu_seqlens_q, case.max_seqlen_q)
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*args, **kwargs)
        torch.cuda.synchronize()
    names = set()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and not EXCLUDED.search(ev.name):
            names.add(ev.name)
    return names


def component(adapter: str) -> dict:
    fn = check.load_candidate(adapter)
    ids: dict[str, set] = {}
    for kind, name in (("F", "full"), ("L", "local")):
        for phase, calls in CALLS.items():
            key = f"{phase}_{name}"
            for i, (form, reqs) in enumerate(calls):
                case = check.materialize(spec(form, kind, reqs, f"{key}-{form}-{i}"))
                ids.setdefault(key, set()).update(kernels_of(fn, case))
    return {k: sorted(v) for k, v in sorted(ids.items())}


def read_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return json.load(fh).get("traceEvents", [])


def serving(trace_dir: Path, identity: dict) -> dict:
    out = {}
    for path in sorted(trace_dir.rglob("*.json*")):
        m = re.search(r"TP-(\d+)", path.name)
        rank = m.group(1) if m else path.name
        counts = collections.Counter()
        graph_launches = 0
        for ev in read_trace(path):
            cat, name = ev.get("cat", ""), ev.get("name", "")
            if cat == "kernel":
                counts[name] += 1
            elif name == "cudaGraphLaunch":
                graph_launches += 1
        rec = {"trace": str(path), "cudaGraphLaunch": graph_launches, "families": {}}
        for fam, names in identity.items():
            present = {n: counts[n] for n in names if counts[n]}
            top = max(names, key=lambda n: counts[n]) if names else None
            rec["families"][fam] = {"kernels_present": present, "most_frequent_component_kernel": top,
                                    "its_count": counts[top] if top else 0}
        out[f"rank {rank}"] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="/app/inkling_rel_attention.py")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--serving-trace-dir", type=Path)
    ap.add_argument("--identity", type=Path, help="identity sets written by the component side")
    args = ap.parse_args()
    if args.serving_trace_dir:
        if not args.identity:
            ap.error("--serving-trace-dir needs --identity")
        ids = json.loads(args.identity.read_text())
        rec = {"serving": serving(args.serving_trace_dir, ids.get("identity_sets", ids))}
    else:
        rec = {"adapter": args.adapter, "identity_sets": component(args.adapter)}
    text = json.dumps(rec, indent=1)
    print(text)
    if args.output:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
