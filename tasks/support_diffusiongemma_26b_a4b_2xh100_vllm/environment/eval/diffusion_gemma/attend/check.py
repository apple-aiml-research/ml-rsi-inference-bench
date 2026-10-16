#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Runs an `attend` implementation on explicit case objects and applies the contract's policies.

    python -m eval.diffusion_gemma.attend.check --cases examples/attend_cases.json [--impl /app/diffusion_gemma_attend.py]

A case object:
    {"id": str, "layer_kind": "sliding"|"full", "tp": 1|2, "values": "rand"|"equal"|"probe", "seed": int,
     "sequences": [{"kind": "encode", "start": s, "m": m, "spans": [[a, b], ...]} |
                   {"kind": "denoise", "c": c, "m": m}],
     "probes": [[sequence, row, key], ...]}          # probe cases; key is an absolute position
Inputs are generated on the CPU from the seed and moved to the GPU. The pool holds 1.25x the blocks
needed; the block table is a seeded permutation; unreferenced blocks hold sentinels (K and V = +-64
in every dimension), so an out-of-table read changes results. Structure is validated first; then
the any-case rule (FP64 reference), the equal-logits rule on cases whose values are "equal" and the
witness on probe cases. The entry points must not modify the caches.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from eval.diffusion_gemma.attend import policy, reference, witness  # type: ignore
else:
    from . import policy, reference, witness

SHAPES = {("sliding", 1): (16, 8, 256), ("sliding", 2): (8, 4, 256),
          ("full", 1): (16, 2, 512), ("full", 2): (8, 1, 512)}
SENTINEL = 64.0


def load_impl(path: str):
    spec = importlib.util.spec_from_file_location("diffusion_gemma_attend", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("create_cache", "cache_views", "attend"):
        if not callable(getattr(mod, name, None)):
            raise SystemExit(f"{path} does not define {name}()")
    return mod


def geometry(case: dict) -> list[dict]:
    out = []
    for s in case["sequences"]:
        if s["kind"] == "encode":
            out.append({"kind": "encode", "n": s["start"] + s["m"], "m": s["m"], "spans": [tuple(x) for x in s.get("spans", [])]})
        else:
            out.append({"kind": "denoise", "n": s["c"] + s["m"], "m": s["m"], "spans": []})
    return out


class Inputs:
    """One case's inputs, written into an implementation's cache through its views."""

    def __init__(self, case: dict, impl, device):
        self.case, self.impl, self.device = case, impl, device
        self.layer_kind, tp = case["layer_kind"], case["tp"]
        self.hq, self.hk, self.d = SHAPES[(self.layer_kind, tp)]
        self.seqs = geometry(case)
        g = torch.Generator().manual_seed(int(case["seed"]))
        total = sum(s["n"] for s in self.seqs)
        self.cache = impl.create_cache(self.layer_kind, tp, max(1, math.ceil(1.25 * total) + 64), device)
        k_view, v_view = impl.cache_views(self.cache)
        self._check_views(k_view, v_view)
        nb, bs = k_view.shape[0], k_view.shape[1]
        need = [math.ceil(s["n"] / bs) for s in self.seqs]
        if sum(need) > nb:
            raise SystemExit(f"{case['id']}: the cache holds {nb} blocks of {bs}, the case needs {sum(need)}")
        perm = torch.randperm(nb, generator=g)
        self.block_table = torch.zeros(len(self.seqs), max(max(need), 1), dtype=torch.int32)
        used, k0 = [], 0
        for r, n_b in enumerate(need):
            self.block_table[r, :n_b] = perm[k0:k0 + n_b].to(torch.int32)
            used += perm[k0:k0 + n_b].tolist()
            k0 += n_b
        k_host = torch.full((nb, bs, self.hk, self.d), SENTINEL, dtype=torch.bfloat16)
        v_host = torch.full((nb, bs, self.hk, self.d), -SENTINEL, dtype=torch.bfloat16)
        scale = self.d ** -0.25
        for r, s in enumerate(self.seqs):
            kk = (torch.randn((s["n"], self.hk, self.d), generator=g) * scale).to(torch.bfloat16)
            vv = torch.randn((s["n"], self.hk, self.d), generator=g).to(torch.bfloat16)
            self._write(k_host, v_host, r, kk, vv, bs)
        self.q = [(torch.randn((s["m"], self.hq, self.d), generator=g) * scale).to(torch.bfloat16) for s in self.seqs]
        if case.get("values") == "equal":
            self.q = [torch.zeros_like(x) for x in self.q]
        self.probes = []
        if case.get("values") == "probe":
            self._install_probes(k_host, v_host, bs, g)
        k_view.copy_(k_host.to(device))
        v_view.copy_(v_host.to(device))
        self.k_host, self.v_host = k_host, v_host
        self.q_all = torch.cat(self.q, 0).to(device) if sum(s["m"] for s in self.seqs) else torch.empty(0, self.hq, self.d, dtype=torch.bfloat16, device=device)
        self.seq_lens = torch.tensor([s["n"] for s in self.seqs], dtype=torch.int32, device=device)
        qsl = [0]
        for s in self.seqs:
            qsl.append(qsl[-1] + s["m"])
        self.query_start_loc = torch.tensor(qsl, dtype=torch.int32, device=device)
        self.encode = torch.tensor([1 if s["kind"] == "encode" else 0 for s in self.seqs], dtype=torch.int32, device=device)
        max_spans = max([len(s["spans"]) for s in self.seqs] + [1])
        ir = torch.zeros(len(self.seqs), max_spans, 2, dtype=torch.int32)
        for r, s in enumerate(self.seqs):
            for j, (a, b) in enumerate(s["spans"]):
                ir[r, j, 0], ir[r, j, 1] = a, b
        self.image_ranges = ir.to(device)
        self.block_table = self.block_table.to(device)
        self.digest_before = self._digest()

    def _check_views(self, k_view, v_view):
        for name, t in (("k", k_view), ("v", v_view)):
            if t.dtype != torch.bfloat16 or t.dim() != 4 or t.shape[2] != self.hk or t.shape[3] != self.d:
                raise SystemExit(f"cache_views {name}: expected BF16 [blocks, block_size, {self.hk}, {self.d}], "
                                 f"got {t.dtype} {tuple(t.shape)}")

    def _write(self, k_host, v_host, r, kk, vv, bs):
        n = kk.shape[0]
        for b in range(math.ceil(n / bs)):
            phys = int(self.block_table[r, b])
            seg = slice(b * bs, min((b + 1) * bs, n))
            k_host[phys, : seg.stop - seg.start] = kk[seg]
            v_host[phys, : seg.stop - seg.start] = vv[seg]

    def _install_probes(self, k_host, v_host, bs, g):
        group = self.hq // self.hk
        probes = [tuple(p) for p in self.case["probes"]]
        axes = witness.place(probes, self.hk)
        for h, a in set(axes):
            col = k_host[:, :, h, a]
            k_host[:, :, h, a] = torch.where(col.abs() == SENTINEL, col, torch.zeros_like(col))
        markers: dict[tuple[int, int, int], torch.Tensor] = {}
        for (r, i, p), (h, a) in zip(probes, axes):
            qh = h * group
            self.q[r][i, qh].zero_()
            self.q[r][i, qh, a] = witness.QUERY_SCALE
            blk, off = int(self.block_table[r, p // bs]), p % bs
            if (r, p, h) not in markers:  # the first probe of this key on this head
                k_host[blk, off, h].zero_()
                markers[(r, p, h)] = torch.where(torch.rand((self.d,), generator=g) < 0.5, -2.0, 2.0).to(torch.bfloat16)
                v_host[blk, off, h] = markers[(r, p, h)]
            k_host[blk, off, h, a] = witness.KEY_SCALE
            self.probes.append({"seq": r, "row": i, "key": p, "kv_head": h, "q_head": qh, "marker": markers[(r, p, h)]})

    def _digest(self) -> str:
        """A GPU checksum of both views (position-weighted, so a moved value changes it), taken in
        chunks so a 262,144-token cache needs no full-size int64 copy."""
        k_view, v_view = self.impl.cache_views(self.cache)
        parts, step = [], 1 << 26
        for t in (k_view, v_view):
            flat = t.contiguous().view(torch.int16).reshape(-1)
            total, weighted = 0, 0
            for a in range(0, flat.numel(), step):
                x = flat[a:a + step].to(torch.int64)
                w = torch.arange(a + 1, a + 1 + x.numel(), device=x.device, dtype=torch.int64) % 1000003
                total += int(x.sum())
                weighted += int((x * w).sum())
            parts.append(f"{total}:{weighted}")
        return "/".join(parts)

    def call(self):
        return self.impl.attend(self.cache, self.q_all, self.block_table, self.seq_lens, self.query_start_loc,
                                self.encode, self.image_ranges, self.layer_kind)


def run_case(case: dict, impl, device) -> dict:
    t0 = time.monotonic()
    x = Inputs(case, impl, device)
    o = x.call()
    torch.cuda.synchronize()
    rec = {"id": case["id"], "layer_kind": x.layer_kind, "tp": case["tp"], "values": case.get("values", "rand")}
    T = x.q_all.shape[0]
    shape_ok = isinstance(o, torch.Tensor) and o.dtype == torch.bfloat16 and tuple(o.shape) == (T, x.hq, x.d)
    rec["structure"] = {"pass": shape_ok, "got": [str(getattr(o, "dtype", None)), list(getattr(o, "shape", []))]}
    rec["caches_unmodified"] = x._digest() == x.digest_before
    if not shape_ok:
        rec["pass"] = False
        return rec
    k_view, v_view = impl.cache_views(x.cache)
    args = (k_view, v_view, x.q_all, x.block_table, x.seq_lens, x.query_start_loc, x.encode, x.image_ranges, x.layer_kind)
    readings = {}
    for rd in reference.READINGS:
        r64 = reference.attend_reference(*args, dtype=torch.float64, reading=rd)
        rbf = reference.attend_bf16(*args, reading=rd)
        readings[rd] = policy.accuracy(o, r64, rbf)
    rec["accuracy"] = readings
    acc_ok = any(readings[rd]["pass"] for rd in reference.READINGS)
    ok = acc_ok and rec["caches_unmodified"]
    if case.get("values") == "equal":
        eq = {rd: policy.equal_logits(o, *args[:-1], x.layer_kind, reading=rd) for rd in reference.READINGS}
        rec["equal_logits"] = eq
        ok &= any(eq[rd]["pass"] for rd in reference.READINGS)
    if x.probes:
        offs = x.query_start_loc.tolist()
        table = []
        for pr in x.probes:
            s = x.seqs[pr["seq"]]
            obs, rel = witness.observe(o[offs[pr["seq"]] + pr["row"], pr["q_head"]].float().cpu(), pr["marker"])
            exp = {rd: witness.expected(s["kind"], s["n"], s["m"], s["spans"], x.layer_kind, pr["row"], pr["key"], rd)
                   for rd in ("w1", "w")}
            table.append({"seq": pr["seq"], "row": pr["row"], "key": pr["key"], "observed": obs, "rel": rel, "expected": exp})
        rec["witness"] = witness.verdict(table)
        rec["witness_table"] = table
        ok &= rec["witness"]["pass"]
    rec["pass"] = bool(ok)
    rec["seconds"] = round(time.monotonic() - t0, 2)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--impl", default="/app/diffusion_gemma_attend.py")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    cases = json.loads(args.cases.read_text())
    cases = cases["cases"] if isinstance(cases, dict) else cases
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    impl = load_impl(args.impl)
    device = torch.device("cuda")
    records = []
    for case in cases:
        rec = run_case(case, impl, device)
        records.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("id", "pass", "seconds")}
                         | {"acc": {rd: round(v["max_err"], 6) for rd, v in rec.get("accuracy", {}).items()}}
                         | ({"witness": rec["witness"]["fits"]} if "witness" in rec else {})), flush=True)
        torch.cuda.empty_cache()
    summary = {"cases": len(records), "passed": sum(r["pass"] for r in records)}
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "records": records}, default=str))
    print(json.dumps(summary))
    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
