#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The private relative-attention cases and their component invocations.

Expected verdicts, stated first: the oracle adapter passes every case on the
reference; a candidate passes a correctness node iff every one of its cases passes the numerical policy of
`rel_attention/policy.py` with its inputs unchanged.

    python rel_cases.py digests --out <hidden/rel_cases.json>                   # record every case's input digests
    python rel_cases.py run --node prefill|decode --kind F|L --family <fam> --adapter <py> --out <json>

The case matrix (about 250): the ragged geometries RG1-RG16 and the paged geometries PD1-PD11 and PE1-PE6, each in
both kinds (F: E = 1,024, window (None, None); L: E = 512, window (511, 0)), with the value families V-rand (draws 1
and 2, every geometry), V-probe, V-edge, V-window (L only), V-headmix and V-sentinel (every paged geometry), on the
geometries `families_for` names. Seeds are the first eight bytes of SHA-256(`inkling-rel-v1/<case-id>/<kind>/<family>/<draw>`),
little-endian, mod 2^63; values are drawn on the CPU and moved to the GPU. Page pools hold ceil(1.25 x pages needed) + 1
pages behind a seeded permutation; unreferenced slots and pages hold +-96 in every coordinate, except
V-sentinel, whose whole unreferenced pages hold NaN. V-window's values for rows that are not decode rows follow V-probe
(q = 0, bias 0 there) and its values v follow V-probe, so that a wrongly attended key moves the output.

`run` materializes one (node, kind, family) invocation's cases, checks each case's input digests against
`hidden/rel_cases.json`, calls the candidate through `rel_attention/check.py`'s `run_case` and writes each case's record
before the next case runs. The FP64 reference runs on the GPU in 128-query chunks (`rel_attention/reference.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
REL = APP_DIR / "eval" / "inkling_small" / "rel_attention"
sys.path.insert(0, str(REL))
import check  # noqa: E402
from reference import PAGE, Case  # noqa: E402

HEADS, KV_HEADS, DIM, SENTINEL = check.HEADS, check.KV_HEADS, check.DIM, check.SENTINEL

RAGGED = {
    "RG1": [(1, 1)], "RG2": [(127, 127)], "RG3": [(128, 128)], "RG4": [(129, 129)], "RG5": [(511, 511)],
    "RG6": [(512, 512)], "RG7": [(513, 513)], "RG8": [(1023, 1023)], "RG9": [(1024, 1024)], "RG10": [(1025, 1025)],
    "RG11": [(2049, 2049)], "RG12": [(600, 88)], "RG13": [(1152, 128)], "RG14": [(4224, 129)], "RG15": [(16384, 8192)],
    "RG16": [(1, 1), (9, 9), (513, 1), (1027, 3), (2049, 2049), (5000, 1), (4224, 129)],
}
PD11_ROWS = [1, 2, 127, 128, 129, 511, 512, 513, 700, 1023, 1024, 1025, 2047, 2048, 2049, 4096, 9999, 16385, 65536,
             131072, 131073, 200000, 262143, 262144]
PAGED = {
    "PD1": [(1, 1)], "PD2": [(128, 1)], "PD3": [(129, 1)], "PD4": [(511, 1)], "PD5": [(512, 1)], "PD6": [(513, 1)],
    "PD7": [(1023, 1)], "PD8": [(1024, 1)], "PD9": [(1025, 1)], "PD10": [(262144, 1)], "PD11": [(s, 1) for s in PD11_ROWS],
    "PE1": [(4224, 129)], "PE2": [(16511, 127)], "PE3": [(24576, 8192)], "PE4": [(131200, 1024)], "PE5": [(262144, 2048)],
    "PE6": [(1, 1), (129, 129), (600, 88), (5000, 1), (70000, 1000), (262144, 1)],
}
NODE_GEOMETRIES = {"prefill": [g for g in RAGGED] + [g for g in PAGED if g.startswith("PE")],
                   "decode": [g for g in PAGED if g.startswith("PD")]}
RANGE = lambda prefix, lo, hi: [f"{prefix}{i}" for i in range(lo, hi + 1)]  # noqa: E731


def families_for(geometry: str, kind: str) -> list[tuple[str, int]]:
    fams = [("V-rand", 1), ("V-rand", 2)]
    probe = RANGE("RG", 12, 16) + RANGE("PD", 2, 11) + RANGE("PE", 1, 6)
    if geometry in probe:
        fams.append(("V-probe", 1))
    edge = (RANGE("RG", 9, 11) + RANGE("RG", 13, 15) + RANGE("PD", 7, 11) + RANGE("PE", 1, 6)) if kind == "F" else \
        (RANGE("RG", 5, 7) + RANGE("PD", 4, 6) + ["PD11"])
    if geometry in edge:
        fams.append(("V-edge", 1))
    if kind == "L" and geometry in RANGE("PD", 6, 11) + ["RG16", "PE6"]:
        fams.append(("V-window", 1))
    if geometry in ("RG12", "RG16", "PD11", "PE6"):
        fams.append(("V-headmix", 1))
    if geometry in PAGED:
        fams.append(("V-sentinel", 1))
    return fams


def matrix() -> list[dict]:
    out = []
    for geometry in list(RAGGED) + list(PAGED):
        for kind in ("F", "L"):
            for fam, draw in families_for(geometry, kind):
                node = "decode" if geometry.startswith("PD") else "prefill"
                out.append({"case": f"{geometry}/{kind}/{fam}/{draw}", "geometry": geometry, "kind": kind,
                            "family": fam, "draw": draw, "node": node,
                            "form": "paged" if geometry in PAGED else "ragged",
                            "requests": [list(r) for r in (PAGED.get(geometry) or RAGGED[geometry])]})
    return out


def seed(spec: dict) -> int:
    label = f"inkling-rel-v1/{spec['geometry']}/{spec['kind']}/{spec['family']}/{spec['draw']}"
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little") % 2 ** 63


def _probe_v(k_len: int) -> torch.Tensor:
    j = torch.arange(k_len).view(-1, 1, 1)
    c = torch.arange(DIM).view(1, 1, -1)
    return ((((5 * j + c) % 9) - 4) / 2).expand(k_len, KV_HEADS, DIM).to(torch.bfloat16)


def values(spec: dict, gen: torch.Generator):
    """q, rel_bias (flattened over requests) and per-request token keys and values, for the spec's family."""
    kind, fam, reqs = spec["kind"], spec["family"], [tuple(r) for r in spec["requests"]]
    E = 1024 if kind == "F" else 512
    total_q = sum(q for _, q in reqs)
    if fam in ("V-rand", "V-headmix", "V-sentinel"):
        q = (4 * torch.randn(total_q, HEADS, DIM, generator=gen).clamp(-3, 3)).to(torch.bfloat16)
        if fam == "V-headmix":  # q shared across the four heads of a KV group; only the bias tells heads apart
            q = q.view(total_q, KV_HEADS, HEADS // KV_HEADS, DIM)[:, :, :1].expand(-1, -1, HEADS // KV_HEADS, -1)
            q = q.reshape(total_q, HEADS, DIM).contiguous()
        bias = torch.randn(total_q, HEADS, E, generator=gen).clamp(-3, 3).to(torch.bfloat16)
        k_tok = [torch.randn(kl, KV_HEADS, DIM, generator=gen).clamp(-3, 3).to(torch.bfloat16) for kl, _ in reqs]
        v_tok = [torch.randn(kl, KV_HEADS, DIM, generator=gen).clamp(-3, 3).to(torch.bfloat16) for kl, _ in reqs]
        return q, bias, k_tok, v_tok
    q = torch.zeros(total_q, HEADS, DIM, dtype=torch.bfloat16)
    bias = torch.zeros(total_q, HEADS, E, dtype=torch.bfloat16)
    k_tok = [torch.randn(kl, KV_HEADS, DIM, generator=gen).to(torch.bfloat16) for kl, _ in reqs]
    v_tok = [_probe_v(kl).clone() for kl, _ in reqs]
    t_base = 0
    for b, (kl, ql) in enumerate(reqs):
        for i in range(ql):
            t = t_base + i
            p = kl - ql + i
            if fam == "V-probe":
                for h in range(HEADS):
                    bias[t, h, (37 * t + 101 * h + 11) % E] = 8
            elif fam == "V-edge":
                bias[t, :, E - 2:] = 6
            elif fam == "V-window" and ql == 1 and p >= 512:
                q[t, :, 0] = 32
        if fam == "V-window" and ql == 1 and kl - 1 >= 512:
            p = kl - 1
            k_tok[b] = (0.1 * torch.randn(kl, KV_HEADS, DIM, generator=gen)).to(torch.bfloat16)
            j = p - 512
            k_tok[b][j] = 0
            k_tok[b][j, :, 0] = 32
        t_base += ql
    return q, bias, k_tok, v_tok


def build(spec: dict, device: str = "cuda") -> Case:
    gen = torch.Generator().manual_seed(seed(spec))
    reqs = [tuple(r) for r in spec["requests"]]
    q, bias, k_tok, v_tok = values(spec, gen)
    kind = spec["kind"]
    window = None if kind == "F" else 511
    cu_q = torch.tensor([0] + [sum(ql for _, ql in reqs[: b + 1]) for b in range(len(reqs))], dtype=torch.int32)
    kw = dict(q=q, rel_bias=bias, cu_seqlens_q=cu_q, max_seqlen_q=max(ql for _, ql in reqs),
              window_size=(511, 0) if window else (None, None), softmax_scale=1.0 / 128, meta=dict(spec))
    if spec["form"] == "ragged":
        cu_k = torch.tensor([0] + [sum(kl for kl, _ in reqs[: b + 1]) for b in range(len(reqs))], dtype=torch.int32)
        case = Case(k=torch.cat(k_tok), v=torch.cat(v_tok), cu_seqlens_k=cu_k, max_seqlen_k=max(kl for kl, _ in reqs), **kw)
    else:
        k, v, table, seqused = check.build_paged(reqs, k_tok, v_tok, gen, window,
                                                 nan_pages=spec["family"] == "V-sentinel")
        case = Case(k=k, v=v, page_table=table, seqused_k=seqused, max_seqlen_k=max(kl for kl, _ in reqs), **kw)
    return check.to_device(case, device)


NAMES = ("q", "k", "v", "rel_bias", "cu_seqlens_q", "cu_seqlens_k", "page_table", "seqused_k")


def digests(case: Case) -> dict:
    return {n: check._digest(getattr(case, n)) for n in NAMES}


def cmd_digests(args) -> int:
    out = []
    t0 = time.monotonic()
    for spec in matrix():
        case = build(spec, device="cpu")
        out.append({**spec, "seed": seed(spec), "digests": digests(case)})
    Path(args.out).write_text(json.dumps({"schema": "inkling-rel-cases/1", "cases": out}, indent=1))
    print(json.dumps({"cases": len(out), "seconds": round(time.monotonic() - t0, 1)}))
    return 0


def cmd_run(args) -> int:
    recorded = {c["case"]: c for c in json.loads((TESTS_DIR / "hidden" / "rel_cases.json").read_text())["cases"]}
    specs = [s for s in matrix() if s["node"] == args.node and s["kind"] == args.kind and s["family"] == args.family]
    fn = check.load_candidate(args.adapter)
    results = []
    for spec in specs:
        case = build(spec)
        rec = {"case": spec["case"]}
        got = digests(case)
        if got != recorded[spec["case"]]["digests"]:
            rec.update({"passed": False, "verifier_fault": True, "reason": "input digests differ from hidden/rel_cases.json"})
        else:
            rec.update(check.run_case(fn, case))
        results.append(rec)
        Path(args.out).write_text(json.dumps({"node": args.node, "kind": args.kind, "family": args.family,
                                              "results": results}, indent=1))
        print(json.dumps({k: rec.get(k) for k in ("case", "passed", "max_abs_err", "bound", "reason")}), flush=True)
        del case
        torch.cuda.empty_cache()
    return 0 if all(r.get("passed") for r in results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("digests")
    d.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--node", choices=("prefill", "decode"), required=True)
    r.add_argument("--kind", choices=("F", "L"), required=True)
    r.add_argument("--family", required=True)
    r.add_argument("--adapter", default=str(APP_DIR / "inkling_rel_attention.py"))
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    return cmd_digests(args) if args.cmd == "digests" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
