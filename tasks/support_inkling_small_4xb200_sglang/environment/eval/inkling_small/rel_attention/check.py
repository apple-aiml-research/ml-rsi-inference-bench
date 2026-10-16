#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Run the relative-attention operator contract on explicit cases and report every metric (README.md).

    python check.py --example paged-decode-local --output /tmp/rel.json [--adapter /app/inkling_rel_attention.py]

`load_candidate` imports the committed adapter by path; `run_case` calls `relative_attention` on one case,
validates the result's structure first (a tensor, bf16, [total_q, 8, 128], on the inputs' device), checks the
inputs are bitwise unchanged, then applies `policy.evaluate` against `reference.reference(float64)` and
`reference.reference_bf16`. `materialize` builds a public example (`examples/rel_cases.json`): V-rand values
from the example's public seed, paged caches over a seeded permutation of a page pool, and unreferenced slots
and pages holding +-96 in every coordinate (NaN in whole unreferenced pages when the example says so).
A passing example establishes only that example; acceptance also uses cases this package does not ship.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import policy  # noqa: E402
import reference  # noqa: E402
from reference import PAGE, Case  # noqa: E402

HEADS, KV_HEADS, DIM = 8, 2, 128
SENTINEL = 96.0


def seed_of(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little") % 2 ** 63


def load_candidate(path: str = "/app/inkling_rel_attention.py"):
    spec = importlib.util.spec_from_file_location("inkling_rel_attention_candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.relative_attention


def signed_sentinel(shape, gen: torch.Generator) -> torch.Tensor:
    sign = torch.randint(0, 2, shape, generator=gen) * 2 - 1
    return (sign * SENTINEL).to(torch.bfloat16)


def build_paged(requests: list[tuple[int, int]], k_tokens: list[torch.Tensor], v_tokens: list[torch.Tensor],
                gen: torch.Generator, window: int | None, nan_pages: bool = False):
    """Paged k/v [num_pages, 128, 2, 128] from per-request token rows; returns k, v, page_table, seqused_k."""
    need = [math.ceil(k / PAGE) for k, _ in requests]
    pool = math.ceil(1.25 * sum(need)) + 1
    perm = torch.randperm(pool, generator=gen)
    k = signed_sentinel((pool, PAGE, KV_HEADS, DIM), gen)
    v = signed_sentinel((pool, PAGE, KV_HEADS, DIM), gen)
    width = max(need)
    table = torch.empty(len(requests), width, dtype=torch.int32)
    used = set()
    cursor = 0
    referenced_pages = []
    for b, ((k_len, q_len), n) in enumerate(zip(requests, need)):
        pages = perm[cursor: cursor + n]
        cursor += n
        # entries past the request's pages point at arbitrary pool pages (unreferenced, item 2)
        spare = torch.randint(0, pool, (width - n,), generator=gen)
        table[b] = torch.cat([pages, spare]).to(torch.int32)
        rows = k_tokens[b].reshape(-1, KV_HEADS, DIM)
        vrows = v_tokens[b].reshape(-1, KV_HEADS, DIM)
        first_attended = 0 if window is None else max(0, k_len - q_len - window)
        for i, pg in enumerate(pages.tolist()):
            lo, hi = i * PAGE, min(k_len, (i + 1) * PAGE)
            if hi <= first_attended:
                continue  # every position below every query's window (item 3): the page keeps its sentinel
            k[pg, : hi - lo] = rows[lo:hi]
            v[pg, : hi - lo] = vrows[lo:hi]
            referenced_pages.append(pg)
        used.update(pages.tolist())
    if nan_pages:
        referenced = set(referenced_pages)
        for pg in range(pool):
            if pg not in referenced:
                k[pg] = float("nan")
                v[pg] = float("nan")
    seqused = torch.tensor([k_len for k_len, _ in requests], dtype=torch.int32)
    return k, v, table, seqused


def materialize(spec: dict, device: str = "cuda") -> Case:
    """A public example: V-rand values from the example's public seed."""
    kind, form, reqs = spec["kind"], spec["form"], [tuple(r) for r in spec["requests"]]
    E = 1024 if kind == "F" else 512
    window = None if kind == "F" else 511
    gen = torch.Generator().manual_seed(seed_of(f"inkling-rel-v1-public/{spec['id']}/{kind}"))
    total_q = sum(q for _, q in reqs)
    q = (4 * torch.randn(total_q, HEADS, DIM, generator=gen).clamp(-3, 3)).to(torch.bfloat16)
    bias = torch.randn(total_q, HEADS, E, generator=gen).clamp(-3, 3).to(torch.bfloat16)
    k_tok = [torch.randn(kl, KV_HEADS, DIM, generator=gen).clamp(-3, 3).to(torch.bfloat16) for kl, _ in reqs]
    v_tok = [torch.randn(kl, KV_HEADS, DIM, generator=gen).clamp(-3, 3).to(torch.bfloat16) for kl, _ in reqs]
    cu_q = torch.tensor([0] + list(__import__("itertools").accumulate(q for _, q in reqs)), dtype=torch.int32)
    kw = dict(q=q, rel_bias=bias, cu_seqlens_q=cu_q, max_seqlen_q=max(q for _, q in reqs),
              window_size=(511, 0) if window else (None, None), softmax_scale=1.0 / 128, meta=dict(spec))
    if form == "ragged":
        cu_k = torch.tensor([0] + list(__import__("itertools").accumulate(k for k, _ in reqs)), dtype=torch.int32)
        case = Case(k=torch.cat(k_tok), v=torch.cat(v_tok), cu_seqlens_k=cu_k, max_seqlen_k=max(k for k, _ in reqs), **kw)
    else:
        k, v, table, seqused = build_paged(reqs, k_tok, v_tok, gen, window, nan_pages=spec.get("unreferenced") == "nan")
        case = Case(k=k, v=v, page_table=table, seqused_k=seqused, max_seqlen_k=max(k for k, _ in reqs), **kw)
    return to_device(case, device)


def to_device(case: Case, device: str) -> Case:
    for name in ("q", "k", "v", "rel_bias", "cu_seqlens_q", "cu_seqlens_k", "page_table", "seqused_k"):
        t = getattr(case, name)
        if t is not None:
            setattr(case, name, t.to(device))
    return case


def _digest(t: torch.Tensor | None) -> str | None:
    if t is None:
        return None
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes() if t.dtype != torch.int32
                          else t.contiguous().cpu().numpy().tobytes()).hexdigest()


def run_case(fn, case: Case) -> dict:
    names = ("q", "k", "v", "rel_bias", "cu_seqlens_q", "cu_seqlens_k", "page_table", "seqused_k")
    before = {n: _digest(getattr(case, n)) for n in names}
    kwargs = dict(causal=True, window_size=case.window_size, softmax_scale=case.softmax_scale)
    if case.paged:
        kwargs.update(page_table=case.page_table, seqused_k=case.seqused_k, max_seqlen_k=case.max_seqlen_k)
    else:
        kwargs.update(cu_seqlens_k=case.cu_seqlens_k, max_seqlen_k=case.max_seqlen_k)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    try:
        out = fn(case.q, case.k, case.v, case.rel_bias, case.cu_seqlens_q, case.max_seqlen_q, **kwargs)
        torch.cuda.synchronize()
    except Exception as exc:
        return {"passed": False, "reason": f"relative_attention raised {type(exc).__name__}: {exc}"[:2000]}
    wall = time.monotonic() - t0
    rec = {"wall_s": round(wall, 4)}
    want = (case.q.shape[0], HEADS, case.v.shape[-1])
    problems = []
    if not isinstance(out, torch.Tensor):
        problems.append(f"result is {type(out).__name__}, not a tensor")
    else:
        if out.dtype != torch.bfloat16:
            problems.append(f"result dtype {out.dtype}, not bfloat16")
        if tuple(out.shape) != want:
            problems.append(f"result shape {tuple(out.shape)}, not {want}")
        if out.device != case.q.device:
            problems.append(f"result on {out.device}, not {case.q.device}")
    if problems:
        rec.update({"passed": False, "reason": "; ".join(problems)})
        return rec
    after = {n: _digest(getattr(case, n)) for n in names}
    unchanged = before == after
    r = reference.reference(case, torch.float64)
    o_bf16 = reference.reference_bf16(case)
    rec.update(policy.evaluate(out, r, o_bf16, reference.valid_rows(case), unchanged))
    if not unchanged:
        rec["modified_inputs"] = [n for n in names if before[n] != after[n]]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", action="append", help="example id from examples/rel_cases.json (repeatable; default all)")
    ap.add_argument("--examples", type=Path, default=HERE.parent / "examples" / "rel_cases.json")
    ap.add_argument("--adapter", default="/app/inkling_rel_attention.py")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    specs = json.loads(args.examples.read_text())["cases"]
    if args.example:
        specs = [s for s in specs if s["id"] in set(args.example)]
    fn = load_candidate(args.adapter)
    results = []
    for spec in specs:
        rec = {"id": spec["id"], **run_case(fn, materialize(spec))}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    if args.output:
        args.output.write_text(json.dumps({"adapter": args.adapter, "results": results}, indent=1))
    return 0 if all(r.get("passed") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
