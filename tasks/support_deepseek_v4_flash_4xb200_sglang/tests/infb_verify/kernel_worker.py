#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""One arm's execution of one component node, in a process of its
own. The worker generates the node's private inputs from `kernel_cases` and checks them against the fixture manifest
BEFORE it imports anything from the arm's tree; then it loads the arm (`check.load_arm`: the tree's `python/` and its
vendored `third_party/flash_mla` ahead of everything installed), runs every case through the public interfaces and
writes the raw outputs. It judges nothing: the verdicts are computed by `kernel_verdict.py` in a process that never
imports an implementation under test.

    python -I kernel_worker.py --tree <tree> --node <node id> --out <dir> --records <jsonl> --manifest <json>

Outputs: `<dir>/<case>.pt` (a dict of CPU tensors, read back with `torch.load(weights_only=True)`) and one JSONL line
per case in `--records` (status `ran`, `unavailable` for NotImplementedError, `error`, with the elapsed time and the
peak device memory). An efficiency node writes one line per family: `check.measure_family`'s record, under the
family's own invocation limit. Exit 0 when the node ran (whatever its cases did), 3 on a verifier fault (inputs that
differ from the manifest, an arm whose modules load from outside its tree, the measurement functions replaced).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_cases as kc  # noqa: E402  (trusted; imported before any arm code)
import torch  # noqa: E402

CHECK_PATH = HERE.parent / "eval" / "deepseek_v4_flash" / "kernels" / "check.py"
VERIFIER_FAULT = 3


def load_check():
    spec = importlib.util.spec_from_file_location("infb_kernel_check", CHECK_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["infb_kernel_check"] = mod
    spec.loader.exec_module(mod)
    return mod


chk = load_check()
_TIMING = {"perf_counter": time.perf_counter, "monotonic": time.monotonic, "Event": torch.cuda.Event,
           "elapsed_time": torch.cuda.Event.elapsed_time, "synchronize": torch.cuda.synchronize,
           "CUDAGraph": torch.cuda.CUDAGraph, "Measure.run": chk.Measure.run}


def timing_intact() -> list[str]:
    now = {"perf_counter": time.perf_counter, "monotonic": time.monotonic, "Event": torch.cuda.Event,
           "elapsed_time": torch.cuda.Event.elapsed_time, "synchronize": torch.cuda.synchronize,
           "CUDAGraph": torch.cuda.CUDAGraph, "Measure.run": chk.Measure.run}
    return [k for k in _TIMING if now[k] is not _TIMING[k]]


def fname(case_id: str) -> str:
    return case_id.replace("/", "__") + ".pt"


def cpu(**tensors) -> dict:
    return {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}


# ── one case per kind ───────────────────────────────────────────────────────────────────────────


def run_bf16(arm, case, t) -> dict:
    out, mx, lse = arm.flash_mla.flash_mla_sparse_fwd(
        t["q"].cuda(), t["kv"].cuda(), t["indices"].cuda(), kc.SCALE, kc.D, t["attn_sink"].cuda(),
        t["topk_length"].cuda())
    torch.cuda.synchronize()
    return cpu(out=out, max_logits=mx, lse=lse)


def run_packed(arm, case, t) -> dict:
    pools = {}
    for name, page in kc.packed_pages(case).items():
        raw = t[f"{name}.raw"].cuda()
        pools[name] = {"raw": raw, "view": raw[:, : page * 584].view(raw.shape[0], page, 1, 584), "page": page,
                       "indices": t[f"{name}.indices"].cuda(), "lengths": t[f"{name}.lengths"].cuda()}
    inp = {"q": t["q"].cuda(), "pools": pools, "attn_sink": t["attn_sink"].cuda(), "sm_scale": kc.SCALE}
    out, lse, _ = chk.call_packed(arm, inp)
    torch.cuda.synchronize()
    return cpu(out=out, lse=lse)


def new_state(ratio: int, hd: int):
    """The state pool of a case with one guard window on each side, every window holding the stale sentinel: the
    engine's pool is uninitialized memory (CompressStatePool: torch.empty), so no window starts cleared."""
    last = (4 if ratio == 4 else 2) * hd
    big = torch.empty(kc.state_windows(ratio) + 2, ratio, last, dtype=torch.float32, device="cuda")
    big[..., : last // 2] = kc.STATE_SENTINEL[0]
    big[..., last // 2:] = kc.STATE_SENTINEL[1]
    return big


def untouched(big, ratio: int, used: set[int]) -> torch.Tensor:
    """[guard windows changed, idle windows changed]: elements that differ from the sentinel."""
    last = big.shape[-1]
    sent = torch.empty_like(big[0])
    sent[..., : last // 2] = kc.STATE_SENTINEL[0]
    sent[..., last // 2:] = kc.STATE_SENTINEL[1]
    guards = int((big[0] != sent).sum() + (big[-1] != sent).sum())
    idle = [w for w in range(kc.state_windows(ratio)) if w not in used]
    changed = int((big[1:-1][idle] != sent).sum()) if idle else 0
    return torch.tensor([guards, changed], dtype=torch.int64)


def _prefill(arm, state, reqs, batch, rows, extends, ratio, hd, ape):
    """One ragged prefill/extend call over `batch` (request indices in batch order); (out, [(request, position)])."""
    seq = torch.tensor([reqs[i].length + e for i, e in zip(batch, extends)], dtype=torch.int64)
    ext = torch.tensor(extends, dtype=torch.int64)
    plan = arm.dsv4.compress_plan(ratio, int(ext.sum()), seq, ext, "cuda")
    idx, extra = chk.prefill_metadata([reqs[i] for i in batch], extends, ratio)
    out = arm.dsv4.compress_forward(state, rows, ape, idx, plan, extra, head_dim=hd, compress_ratio=ratio)
    where = [(i, reqs[i].length + j) for i, e in zip(batch, extends) for j in range(e)]
    for i, e in zip(batch, extends):
        reqs[i].length += e
    return out, where, plan


def _decode(arm, state, reqs, batch, rows, ratio, hd, ape):
    for i in batch:
        reqs[i].length += 1
    seq = torch.tensor([reqs[i].length for i in batch], dtype=torch.int32, device="cuda")
    plan = arm.dsv4.compress_plan(ratio, len(batch), seq, None, "cuda")
    idx, extra = chk.decode_metadata([reqs[i] for i in batch], ratio)
    out = arm.dsv4.compress_forward(state, rows, ape, idx, plan, extra, head_dim=hd, compress_ratio=ratio)
    return out, [(i, reqs[i].length - 1) for i in batch], plan


def run_state(arm, case, t) -> dict:
    p = case.p
    ratio, hd, _, _ = kc.FAMILIES[p["family"]]
    ape = t["ape"].cuda()
    pages = t["pages"].tolist()
    order = t["order"].tolist()
    xs = [t[f"x{i}"].cuda() for i in range(3)]
    pre, ext, steps = list(p["prefixes"]), p["extend"], p["decode_steps"]
    totals = [a + ext + steps for a in pre]
    used = kc.used_windows(pages, totals, ratio)
    res = {}
    for mode in ("oneshot", "chunked"):
        big = new_state(ratio, hd)
        state = big[1:-1]
        reqs = [chk.Request(pages=pages[i], length=0) for i in range(3)]
        emitted, keys = [], []
        chunks = [[a + ext for a in pre]] if mode == "oneshot" else [pre, [ext] * 3]
        for lens in chunks:
            extends = [lens[i] for i in order]
            rows = torch.cat([xs[i][reqs[i].length: reqs[i].length + lens[i]] for i in order])
            out, where, _ = _prefill(arm, state, reqs, order, rows, extends, ratio, hd, ape)
            for j, (i, pos) in enumerate(where):
                if (pos + 1) % ratio == 0:
                    emitted.append(out[j])
                    keys.append((i, pos))
        for s in range(steps):
            rows = torch.stack([xs[i][reqs[i].length] for i in order])
            out, where, _ = _decode(arm, state, reqs, order, rows, ratio, hd, ape)
            for j, (i, pos) in enumerate(where):
                if (pos + 1) % ratio == 0:
                    emitted.append(out[j])
                    keys.append((i, pos))
        torch.cuda.synchronize()
        res[f"{mode}.rows"] = torch.stack(emitted) if emitted else torch.empty(0, hd, device="cuda")
        res[f"{mode}.keys"] = torch.tensor(keys, dtype=torch.int64).view(-1, 2)
        res[f"{mode}.untouched"] = untouched(big, ratio, used)
    return cpu(**res)


def run_branch(arm, case, t) -> dict:
    p = case.p
    ratio, hd, _, _ = kc.FAMILIES[p["family"]]
    ape = t["ape"].cuda()
    pages = t["pages"].tolist()
    pre, acc, steps = p["prefix"], p["accepted"], p["decode_steps"]
    big = new_state(ratio, hd)
    reqs = [chk.Request(pages=pages[0], length=0)]
    emitted, keys = [], []
    out, where, _ = _prefill(arm, big[1:-1], reqs, [0], t["prefix"].cuda(), [pre], ratio, hd, ape)
    for j, (_, pos) in enumerate(where):
        if (pos + 1) % ratio == 0:
            emitted.append(out[j])
            keys.append(pos)
    copy = big.clone()                                    # the copied accepted-prefix state
    state = copy[1:-1]
    out, where, _ = _prefill(arm, state, reqs, [0], t["suffix"].cuda(), [p["suffix"]], ratio, hd, ape)
    for j, (_, pos) in enumerate(where):
        if (pos + 1) % ratio == 0 and pos < pre + acc:     # an emission inside the accepted part of the suffix
            emitted.append(out[j])
            keys.append(pos)
    reqs[0].length = pre + acc                            # the driver restores the logical accepted history
    new = t["new"].cuda()
    for s in range(steps):
        out, where, _ = _decode(arm, state, reqs, [0], new[s: s + 1], ratio, hd, ape)
        pos = where[0][1]
        if (pos + 1) % ratio == 0:
            emitted.append(out[0])
            keys.append(pos)
    torch.cuda.synchronize()
    used = kc.used_windows(pages, [max(pre + p["suffix"], pre + acc + steps)], ratio)
    return cpu(rows=torch.stack(emitted) if emitted else torch.empty(0, hd, device="cuda"),
               keys=torch.tensor(keys, dtype=torch.int64), untouched=untouched(copy, ratio, used))


def run_normrope(arm, case, t) -> dict:
    p = case.p
    ratio, hd, _, _ = kc.FAMILIES[p["family"]]
    reqs = kc.normrope_requests(ratio, p["phase"])
    x = t["x"].cuda()
    if p["phase"] == "prefill":
        seq = torch.tensor([a + e for a, e in reqs], dtype=torch.int64)
        ext = torch.tensor([e for _, e in reqs], dtype=torch.int64)
        plan = arm.dsv4.compress_plan(ratio, x.shape[0], seq, ext, "cuda")
    else:
        seq = torch.tensor([n for n, _ in reqs], dtype=torch.int32, device="cuda")
        plan = arm.dsv4.compress_plan(ratio, x.shape[0], seq, None, "cuda")
    y = x.clone()
    arm.dsv4.compress_fused_norm_rope_inplace(y, kc.norm_weight(hd).cuda(), kc.NORM_EPS, kc.rope_table().cuda(), plan)
    torch.cuda.synchronize()
    return cpu(y=y)


def run_store(arm, case, t) -> dict:
    p = case.p
    pool = t["pool"].cuda()
    x = t["x"].cuda() if p["type"] == "flashmla" else t["x"].to(torch.bfloat16).cuda()
    arm.dsv4.fused_store_cache(x, pool, t["dest"].cuda(), page_size=p["page"], type=p["type"])
    torch.cuda.synchronize()
    return cpu(pool=pool)


def run_compose(arm, case, t) -> dict:
    ratio, hd, typ, page = kc.FAMILIES[case.p["family"]]
    n0, steps, emits = kc.compose_layout(case.p["family"])
    loc_of = {pos: int(d) for pos, d in zip(emits, t["dest"].tolist())}
    ape, x = t["ape"].cuda(), t["x"].cuda()
    w, freqs = kc.norm_weight(hd).cuda(), kc.rope_table().cuda()
    pool = t["pool"].cuda()
    big = new_state(ratio, hd)
    state = big[1:-1]
    req = chk.Request(pages=t["pages"].tolist()[0], length=0)
    post = []
    calls = [(0, n0)] + [(n0 + s, 1) for s in range(steps)]
    for start, n in calls:
        positions = list(range(start, start + n))
        loc = torch.tensor([loc_of.get(pos, 0) for pos in positions], dtype=torch.int32, device="cuda")
        if n > 1 or start == 0:
            plan = arm.dsv4.compress_plan(ratio, n, torch.tensor([start + n]), torch.tensor([n]), "cuda")
            idx, extra = chk.prefill_metadata([req], [n], ratio)
        else:
            req.length = start + 1
            plan = arm.dsv4.compress_plan(ratio, 1, torch.tensor([start + 1], dtype=torch.int32, device="cuda"),
                                          None, "cuda")
            idx, extra = chk.decode_metadata([req], ratio)
        out = chk.compress_step(arm, state=state, kv_score_input=x[start: start + n], ape=ape, indices=idx, plan=plan,
                                extra=extra, head_dim=hd, ratio=ratio, norm_weight=w, eps=kc.NORM_EPS, freqs=freqs,
                                cache_raw=pool, cache_page=page, store_type=typ, store_loc=loc)
        for j, pos in enumerate(positions):
            if pos in loc_of:
                post.append(out[j].clone())
        req.length = start + n
    torch.cuda.synchronize()
    return cpu(post=torch.stack(post), pool=pool)


RUNNERS = {"bf16": run_bf16, "packed": run_packed, "state": run_state, "branch": run_branch,
           "normrope": run_normrope, "store": run_store, "compose": run_compose}


# ── the node ───────────────────────────────────────────────────────────────────────────────────


def origins_ok(arm, tree: Path) -> list[str]:
    bad = []
    for key in ("flash_mla", "flash_mla_cuda", "deepseek_v4"):
        path = Path(arm.identity.get(key, "")).resolve()
        if tree not in path.parents:
            bad.append(f"{key} loaded from {path}, outside the arm's tree {tree}")
    return bad


def write(records: Path, rec: dict) -> None:
    with records.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def correctness(arm, node: str, out: Path, records: Path, prepared: dict) -> None:
    for case in kc.cases(node):
        t = prepared[case.id]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        rec = {"case": case.id, "node": node, "kind": case.kind}
        try:
            res = RUNNERS[case.kind](arm, case, t)
            torch.save(res, out / fname(case.id))
            rec["status"] = "ran"
        except NotImplementedError as exc:
            rec.update(status="unavailable", error=f"NotImplementedError: {exc}"[:2000])
        except Exception as exc:  # noqa: BLE001 - the candidate's failure, recorded
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000],
                       traceback=traceback.format_exc()[-4000:])
        rec.update(elapsed_s=round(time.monotonic() - t0, 3), peak_allocated_bytes=int(torch.cuda.max_memory_allocated()))
        write(records, rec)
        if rec["status"] == "error" and "CUDA" in rec.get("error", "") and "illegal" in rec.get("error", "").lower():
            break  # a corrupted context runs nothing further; the remaining cases are missing outputs


def efficiency(arm, node: str, records: Path) -> None:
    phase = kc.EFFICIENCY_NODES[node]
    limit = int(chk.LIMITS["invocation"]["limit_s"])
    for fam in chk.workload_families(phase):
        seed = kc.timing_seed(phase, fam["family"])
        write(records, {"family": fam["family"], "phase": phase, "status": "started", "seed": seed,
                        "t": time.time()})

        def expire(name=fam["family"]):
            write(records, {"family": name, "phase": phase, "status": "timeout",
                            "failure": f"the family measurement did not finish within its {limit} s invocation limit"})
            os._exit(0)

        timer = threading.Timer(limit, expire)
        timer.daemon = True
        timer.start()
        try:
            rec = chk.measure_family(arm, fam, phase, seed)
            rec["status"] = "measured" if "failure" not in rec else "over_limit"
        except NotImplementedError as exc:
            rec = {"family": fam["family"], "phase": phase, "status": "unavailable",
                   "failure": f"NotImplementedError: {exc}"[:2000]}
        except Exception as exc:  # noqa: BLE001
            rec = {"family": fam["family"], "phase": phase, "status": "error",
                   "failure": f"{type(exc).__name__}: {exc}"[:2000], "traceback": traceback.format_exc()[-4000:]}
        finally:
            timer.cancel()
        replaced = timing_intact()
        if replaced:
            rec.update(status="harness_modified", failure=f"measurement functions replaced in-process: {replaced}")
        rec["seed"] = seed
        write(records, rec)
        torch.cuda.empty_cache()
    # execution attribution (README "Execution attribution"): one untimed profiled invocation per family, after every
    # measurement, so the profiler never touches a timed interval
    for fam in chk.workload_families(phase):
        try:
            prof = chk.profile_family(arm, fam, phase, kc.timing_seed(phase, fam["family"]))
            write(records, {"profile": fam["family"], "phase": phase, "status": "profiled", "role": prof["role"],
                            "kernels": prof["kernels"]})
        except Exception as exc:  # noqa: BLE001 - recorded; attribution then has nothing from this family
            write(records, {"profile": fam["family"], "phase": phase, "status": "profile_error",
                            "failure": f"{type(exc).__name__}: {exc}"[:2000]})
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args(argv)
    tree = Path(args.tree).resolve()
    out, records = Path(args.out), Path(args.records)
    out.mkdir(parents=True, exist_ok=True)
    records.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    head = {"worker": "start", "node": args.node, "tree": str(tree), "t": time.time(), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0)}
    prepared = {}
    if args.node in kc.CORRECTNESS_NODES:
        for case in kc.cases(args.node):
            t = kc.inputs(case)
            problem = kc.check_digest(case, t, manifest)
            if problem:
                write(records, {**head, "status": "verifier_fault", "failure": problem})
                return VERIFIER_FAULT
            prepared[case.id] = t
    elif args.node in kc.EFFICIENCY_NODES:
        phase = kc.EFFICIENCY_NODES[args.node]
        want = manifest.get("timing", {}).get(args.node, {})
        got = {f["family"]: kc.timing_seed(phase, f["family"]) for f in chk.workload_families(phase)}
        if want != got:
            write(records, {**head, "status": "verifier_fault",
                            "failure": f"timing families/seeds {got} differ from the manifest's {want}"})
            return VERIFIER_FAULT
    else:
        write(records, {**head, "status": "verifier_fault", "failure": f"unknown node {args.node}"})
        return VERIFIER_FAULT
    t0 = time.monotonic()
    try:
        arm = chk.load_arm(tree)
    except NotImplementedError as exc:
        write(records, {**head, "status": "unavailable", "failure": f"NotImplementedError: {exc}"[:2000]})
        return 0
    except Exception as exc:  # noqa: BLE001 - an arm that does not import is the arm's failure
        write(records, {**head, "status": "error", "failure": f"{type(exc).__name__}: {exc}"[:2000],
                        "traceback": traceback.format_exc()[-4000:]})
        return 0
    bad = origins_ok(arm, tree)
    replaced = timing_intact()
    write(records, {**head, "status": "loaded", "identity": arm.identity, "load_s": round(time.monotonic() - t0, 3),
                    "origin_problems": bad, "timing_replaced": replaced})
    if bad:
        return VERIFIER_FAULT
    if args.node in kc.CORRECTNESS_NODES:
        correctness(arm, args.node, out, records, prepared)
    else:
        if replaced:
            write(records, {**head, "status": "harness_modified",
                            "failure": f"measurement functions replaced on import: {replaced}"})
            return 0
        efficiency(arm, args.node, records)
    write(records, {"worker": "end", "node": args.node, "t": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
