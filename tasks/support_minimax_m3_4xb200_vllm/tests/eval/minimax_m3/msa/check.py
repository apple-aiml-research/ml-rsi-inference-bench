#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Run the MSA operator contract (``/app/minimax_m3_msa.py``) on explicit cases and apply the published policies.

A case names one operator, a KV-cache dtype, a page pool, a block table, the batch geometry and the tensors. The
check installs the cache contents through ``create_cache`` / ``cache_views``, calls the entry point, validates the
result's structure first (type, device, dtype, shape, finite valid rows, caches unmodified), then applies
``policy.py`` against the FP64 reference of ``reference.py`` and reports every metric. It says which locally decidable
checks passed; it does not declare the task passed.

  python /app/eval/minimax_m3/run.py msa-check --example prefill-bf16 --output /tmp/msa.json

Examples live in ``examples/msa_cases.json`` (N(0,1) inputs from public seeds, both KV dtypes): each prefill example
runs ``select_prefill`` and then ``attend_prefill`` on the reference selection of the same inputs, each decode example
``select_decode`` then ``attend_decode``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch

from . import policy
from . import reference as ref

CONTRACT = Path("/app/minimax_m3_msa.py")
EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "msa_cases.json"
KV_DTYPES = {"bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn}
OPS = ("select_prefill", "select_decode", "attend_prefill", "attend_decode")


@dataclasses.dataclass
class Case:
    """One operator invocation. Prefill cases carry ``query_start_loc`` and ``context_lens``; decode cases do not."""

    case_id: str
    op: str
    kv_dtype: torch.dtype
    num_pages: int
    block_table: torch.Tensor  # int32 [R, max_pages]
    seq_lens: torch.Tensor  # int32 [R]
    k_pages: torch.Tensor  # [P, 128, 128] kv_dtype
    v_pages: torch.Tensor  # [P, 128, 128] kv_dtype
    index_k_pages: torch.Tensor  # [P, 128, 128] bf16
    index_q: torch.Tensor | None = None  # bf16 [T, 128]
    q: torch.Tensor | None = None  # bf16 [T, 16, 128]
    selection: torch.Tensor | None = None  # int32 [T, 16]
    query_start_loc: torch.Tensor | None = None  # int32 [R + 1]
    context_lens: torch.Tensor | None = None  # int32 [R]

    @property
    def prefill(self) -> bool:
        return self.op.endswith("prefill")

    def to(self, device) -> "Case":
        moved = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            moved[f.name] = v.to(device) if isinstance(v, torch.Tensor) else v
        return Case(**moved)

    def rows(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(positions, valid, request) per output row."""
        if self.prefill:
            pos, req = [], []
            qsl, ctx = self.query_start_loc.tolist(), self.context_lens.tolist()
            for r in range(len(ctx)):
                n = qsl[r + 1] - qsl[r]
                pos += [ctx[r] + i for i in range(n)]
                req += [r] * n
            p = torch.tensor(pos, dtype=torch.int64)
            return p, torch.ones_like(p, dtype=torch.bool), torch.tensor(req, dtype=torch.int64)
        s = self.seq_lens.to("cpu", torch.int64)
        return (s - 1).clamp(min=0), s > 0, torch.arange(s.numel())


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def case_digest(case: Case) -> str:
    h = hashlib.sha256()
    for f in dataclasses.fields(case):
        v = getattr(case, f.name)
        if isinstance(v, torch.Tensor):
            h.update(f.name.encode() + digest(v).encode())
        else:
            h.update(f"{f.name}={v}".encode())
    return h.hexdigest()


def load_contract(path: Path = CONTRACT):
    spec = importlib.util.spec_from_file_location("minimax_m3_msa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(module, case: Case, device) -> tuple[object, list[str]]:
    """Create the cache and write the case's contents through the contract's views."""
    problems: list[str] = []
    cache = module.create_cache(case.num_pages, case.kv_dtype, device)
    views = module.cache_views(cache)
    if not isinstance(views, (tuple, list)) or len(views) != 3:
        return cache, [f"cache_views returned {type(views).__name__}, expected (k, v, index_k)"]
    want = [(case.kv_dtype, "k"), (case.kv_dtype, "v"), (torch.bfloat16, "index_k")]
    for view, (dtype, name), src in zip(views, want, (case.k_pages, case.v_pages, case.index_k_pages)):
        if not isinstance(view, torch.Tensor):
            problems.append(f"{name} view is {type(view).__name__}")
            continue
        if tuple(view.shape) != (case.num_pages, ref.BLOCK, ref.HEAD_DIM) or view.dtype != dtype:
            problems.append(f"{name} view {tuple(view.shape)} {view.dtype}, expected "
                            f"{(case.num_pages, ref.BLOCK, ref.HEAD_DIM)} {dtype}")
            continue
        if view.device.type != torch.device(device).type:
            problems.append(f"{name} view on {view.device}")
            continue
        if src.element_size() == 1:
            view.view(torch.uint8).copy_(src.view(torch.uint8))
        else:
            view.copy_(src)
    return cache, problems


def unchanged(module, cache, case: Case) -> list[str]:
    k, v, ik = module.cache_views(cache)
    out = []
    for view, src, name in ((k, case.k_pages, "k"), (v, case.v_pages, "v"), (ik, case.index_k_pages, "index_k")):
        if not torch.equal(view.contiguous().view(torch.uint8), src.contiguous().view(torch.uint8)):
            out.append(f"{name} cache modified by {case.op}")
    return out


def call(module, cache, case: Case):
    if case.op == "select_prefill":
        return module.select_prefill(cache, case.index_q, case.block_table, case.query_start_loc, case.seq_lens,
                                     case.context_lens)
    if case.op == "select_decode":
        return module.select_decode(cache, case.index_q, case.block_table, case.seq_lens)
    if case.op == "attend_prefill":
        return module.attend_prefill(cache, case.q, case.selection, case.block_table, case.query_start_loc,
                                     case.seq_lens, case.context_lens)
    if case.op == "attend_decode":
        return module.attend_decode(cache, case.q, case.selection, case.block_table, case.seq_lens)
    raise ValueError(case.op)


def reference_selection(case: Case, dtype=torch.float64) -> torch.Tensor:
    if case.prefill:
        return ref.select_prefill(case.index_k_pages, case.index_q, case.block_table, case.query_start_loc,
                                  case.seq_lens, case.context_lens, dtype=dtype)
    return ref.select_decode(case.index_k_pages, case.index_q, case.block_table, case.seq_lens, dtype=dtype)


def selection_policy(case: Case, selection: torch.Tensor) -> dict:
    positions, valid, request = case.rows()
    keys = {}

    def row_scores(i: int) -> torch.Tensor:
        r = int(request[i])
        if r not in keys:
            keys[r] = ref.token_rows(case.index_k_pages, case.block_table[r], int(case.seq_lens[r]))
        p = int(positions[i])
        return ref.block_scores(case.index_q[i:i + 1], keys[r], torch.tensor([p], device=case.index_q.device),
                                torch.float64)[0].cpu()

    return policy.check_selection(selection.cpu(), positions, valid, row_scores)


def attention_policy(case: Case, out: torch.Tensor) -> dict:
    positions, valid, request = case.rows()
    fp8 = case.kv_dtype == torch.float8_e4m3fn
    if case.prefill:
        args = (case.k_pages, case.v_pages, case.q, case.selection, case.block_table, case.query_start_loc,
                case.seq_lens, case.context_lens)
        r = ref.attend_prefill(*args, dtype=torch.float64)
        r_bf16 = None if fp8 else ref.attend_prefill(*args, bf16_operands=True)
    else:
        args = (case.k_pages, case.v_pages, case.q, case.selection, case.block_table, case.seq_lens)
        r = ref.attend_decode(*args, dtype=torch.float64)
        r_bf16 = None if fp8 else ref.attend_decode(*args, bf16_operands=True)
    equal = torch.zeros((positions.numel(), ref.NUM_HEADS), dtype=torch.bool)
    max_v = torch.zeros(positions.numel(), dtype=torch.float64)
    cache_k: dict[int, tuple] = {}
    for i in range(positions.numel()):
        if not bool(valid[i]):
            continue
        rq = int(request[i])
        if rq not in cache_k:
            s = int(case.seq_lens[rq])
            cache_k[rq] = (ref.token_rows(case.k_pages, case.block_table[rq], s).to(torch.float64),
                           ref.token_rows(case.v_pages, case.block_table[rq], s).to(torch.float64))
        k, v = cache_k[rq]
        kk, vv, mask = ref.gather_attended(k, v, case.selection[i:i + 1], positions[i:i + 1].to(k.device))
        logits = torch.einsum("hd,sd->hs", case.q[i].to(torch.float64) * ref.SCALE, kk[0])[:, mask[0]]
        if logits.numel():
            equal[i] = (logits.amax(dim=-1) == logits.amin(dim=-1)).cpu()
            max_v[i] = float(vv[0][mask[0]].abs().max())
    return policy.check_attention(out.detach().cpu(), r.cpu(), None if r_bf16 is None else r_bf16.cpu(), fp8,
                                  valid, equal, max_v)


def structure(case: Case, out) -> list[str]:
    positions, valid, _ = case.rows()
    rows = positions.numel()
    want_dtype, want_shape = ((torch.int32, (rows, ref.TOPK)) if case.op.startswith("select")
                              else (torch.bfloat16, (rows, ref.NUM_HEADS, ref.HEAD_DIM)))
    if not isinstance(out, torch.Tensor):
        return [f"{case.op} returned {type(out).__name__}"]
    problems = []
    if out.dtype != want_dtype:
        problems.append(f"dtype {out.dtype}, expected {want_dtype}")
    if tuple(out.shape) != want_shape:
        problems.append(f"shape {tuple(out.shape)}, expected {want_shape}")
    if out.device.type != "cuda":
        problems.append(f"device {out.device}")
    if not problems and case.op.startswith("attend") and not bool(torch.isfinite(out.float()[valid.to(out.device)]).all()):
        problems.append("non-finite values in valid rows")
    return problems


def run_case(module, case: Case, device="cuda") -> dict:
    """Install, call, validate structure, apply the policy. Returns the case record (``pass`` is its verdict)."""
    case = case.to(device)
    rec: dict = {"case": case.case_id, "op": case.op, "kv_dtype": str(case.kv_dtype).replace("torch.", ""),
                 "input_digest": case_digest(case), "status": None}
    t0 = time.monotonic()
    try:
        cache, problems = install(module, case, device)
        if problems:
            rec.update({"status": "candidate_fail", "structure": problems, "pass": False})
            return rec
        torch.cuda.synchronize()
        t1 = time.monotonic()
        out = call(module, cache, case)
        torch.cuda.synchronize()
        rec["call_s"] = round(time.monotonic() - t1, 4)
        problems = structure(case, out) + unchanged(module, cache, case)
    except Exception as exc:  # the candidate's own exception is its failure, recorded verbatim
        rec.update({"status": "candidate_fail", "exception": f"{type(exc).__name__}: {exc}"[:2000], "pass": False,
                    "elapsed_s": round(time.monotonic() - t0, 3)})
        return rec
    if problems:
        rec.update({"status": "candidate_fail", "structure": problems, "pass": False})
    else:
        verdict = selection_policy(case, out) if case.op.startswith("select") else attention_policy(case, out)
        rec.update({"structure": [], "policy": verdict, "pass": bool(verdict["pass"]),
                    "status": "pass" if verdict["pass"] else "candidate_fail"})
    rec["elapsed_s"] = round(time.monotonic() - t0, 3)
    return rec


# ---------------------------------------------------------------- public examples

def _randn(gen, shape, dtype):
    return torch.randn(shape, generator=gen, dtype=torch.float32).clamp_(-3, 3).to(dtype)


def public_cases(spec: dict) -> list[Case]:
    """The select and attend cases of one public example (N(0,1) values clamped to [-3, 3], from its seed)."""
    gen = torch.Generator().manual_seed(int(spec["seed"]))
    kv_dtype = KV_DTYPES[spec["kv_dtype"]]
    if spec["phase"] == "prefill":
        reqs = [tuple(x) for x in spec["requests"]]  # (context, q_len)
        seq = [c + n for c, n in reqs]
    else:
        seq = list(spec["seq_lens"])
    pages_per = [max(1, (s + ref.BLOCK - 1) // ref.BLOCK) for s in seq]
    pool = int(sum(pages_per) * 1.25) + 1
    perm = torch.randperm(pool, generator=gen).to(torch.int32)
    table = torch.zeros((len(seq), max(pages_per)), dtype=torch.int32)
    used = 0
    for r, n in enumerate(pages_per):
        table[r, :n] = perm[used:used + n]
        used += n
    k = _randn(gen, (pool, ref.BLOCK, ref.HEAD_DIM), torch.bfloat16).to(kv_dtype)
    v = _randn(gen, (pool, ref.BLOCK, ref.HEAD_DIM), torch.bfloat16).to(kv_dtype)
    ik = _randn(gen, (pool, ref.BLOCK, ref.HEAD_DIM), torch.bfloat16)
    common = dict(kv_dtype=kv_dtype, num_pages=pool, block_table=table, seq_lens=torch.tensor(seq, dtype=torch.int32),
                  k_pages=k, v_pages=v, index_k_pages=ik)
    if spec["phase"] == "prefill":
        total = sum(n for _, n in reqs)
        qsl = torch.tensor([0] + [sum(n for _, n in reqs[:i + 1]) for i in range(len(reqs))], dtype=torch.int32)
        ctx = torch.tensor([c for c, _ in reqs], dtype=torch.int32)
        common.update(query_start_loc=qsl, context_lens=ctx)
        rows = total
        ops = ("select_prefill", "attend_prefill")
    else:
        rows = len(seq)
        ops = ("select_decode", "attend_decode")
    iq = _randn(gen, (rows, ref.HEAD_DIM), torch.bfloat16)
    q = _randn(gen, (rows, ref.NUM_HEADS, ref.HEAD_DIM), torch.bfloat16)
    sel_case = Case(case_id=f"{spec['id']}/{ops[0]}", op=ops[0], index_q=iq, **common)
    selection = reference_selection(sel_case)
    att_case = Case(case_id=f"{spec['id']}/{ops[1]}", op=ops[1], q=q, selection=selection, **common)
    return [sel_case, att_case]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", required=True, help="an example id from examples/msa_cases.json, or 'all'")
    ap.add_argument("--contract", type=Path, default=CONTRACT)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    specs = json.loads(EXAMPLES.read_text())["examples"]
    chosen = specs if args.example == "all" else [s for s in specs if s["id"] == args.example]
    if not chosen:
        raise SystemExit(f"unknown example {args.example!r}; have {[s['id'] for s in specs]}")
    module = load_contract(args.contract)
    records = [run_case(module, c) for spec in chosen for c in public_cases(spec)]
    out = {"contract": str(args.contract), "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
           "cases": records,
           "note": "local checks on public examples only; grading uses its own inputs and acceptance policy"}
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    for r in records:
        print(f"{r['case']}: {r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
