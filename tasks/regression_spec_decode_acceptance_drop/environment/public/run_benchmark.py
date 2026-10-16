#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Self-test: bench your edited tree, and optionally bench a pristine copy to get a real ratio.

NOT the sealed verifier — this is the cheap measurement you use to decide whether the edits in
`repo/` recovered the regression.

WHY IT CAN BENCH BOTH ARMS. The scored baseline is re-benched from the pinned tree at scoring time,
so there is no number committed here to compare against — a stale one would be worse than none.  # leak-ok: 'stale' means an out-of-date committed figure
`--with-baseline` reconstructs the pristine arm from the `inference_base` tag in `repo/`'s own git
history and benches it the same way, which is the only way to get a ratio that means anything
before you submit.

TWO CAUTIONS about reading a ratio off this rig:

  * A single unrotated sample lies. Benching A then B once can show a difference in the WRONG
    direction. `--rounds N` alternates the arms.
  * The first bench in a fresh process pays one-time initialisation. A warm-up is discarded by
    default (`--no-warmup` if you know why you want to skip it).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _workspace() -> Path:
    w = os.environ.get("INFB_WORKSPACE")
    return Path(w) if w and Path(w).is_dir() else Path(__file__).resolve().parents[1]


def _engine_python() -> str:
    v = os.environ.get("INFB_VLLM_PYTHON")
    if v and Path(v).is_file():
        return v
    venv = os.environ.get("INFB_VENV")
    if venv and (Path(venv) / "bin" / "python").is_file():
        return str(Path(venv) / "bin" / "python")
    return sys.executable


def _matrix(ws: Path) -> dict:
    """Every point of public/workloads.json: the anchor, the other scored points, the guard points."""
    doc = json.loads((ws / "public" / "workloads.json").read_text())
    points = {w["name"]: w for w in doc.get("workloads", []) if isinstance(w, dict) and "name" in w}
    anchor = doc.get("primary_workload") or next(iter(points))
    scored = [str(n) for n in doc.get("scored_workloads") or [anchor]]
    guards = [str(n) for n in doc.get("guard_workloads") or []]
    benched = [anchor] + [n for n in scored if n != anchor] + [n for n in guards
                                                              if not isinstance(points[n].get("from"), str)]
    return {"anchor": anchor, "scored": scored, "guards": guards, "points": points, "benched": benched}


def _workload(ws: Path) -> dict:
    m = _matrix(ws)
    return m["points"][m["anchor"]]


def _geomean(values):
    values = [v for v in values if isinstance(v, (int, float)) and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def _model(ws: Path) -> str:
    v = os.environ.get("INFB_MODEL_PATH")
    if v and Path(v).exists():
        return v
    link = ws / "model"
    if link.exists():
        return str(link.resolve())
    raise SystemExit("no model: set INFB_MODEL_PATH or stage <workspace>/model")


def _bench(tree: Path, ws: Path, out_json: Path, log: Path, point: str | None = None) -> dict | None:
    """One harness run (one server boot) for one point; returns the record, or None when it measured
    nothing comparable. The anchor is the default; each other point is its own boot because its
    concurrency is the server's max_num_seqs."""
    matrix = _matrix(ws)
    wl = matrix["points"][point or matrix["anchor"]]
    max_len = int(wl.get("max_model_len") or 0) or max(
        4096, int(wl["input_len"]) + int(wl["output_len"]) + 256
    )
    cmd = [
        _engine_python(), str(Path(__file__).resolve().parent / "bench_spec_decode.py"),
        "--workspace", str(ws),
        "--input-len", str(wl["input_len"]),
        "--output-len", str(wl["output_len"]),
        "--num-prompts", str(wl["num_prompts"]),
        "--max-num-seqs", str(wl.get("max_concurrency") or 32),
        "--max-model-len", str(max_len),
        "--tensor-parallel-size", "1",
        "--draft-tokens", "7",
        "--dataset-name", "custom",
        "--kv-cache-dtype", "fp8",
        "--gpu-memory-utilization", "0.85",
        "--max-num-batched-tokens", "8192",
        "--cudagraph-capture", "2048",
        "--stream-interval", "20",
        "--output-json", str(out_json),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}:{env.get('PYTHONPATH', '')}".rstrip(":")
    with log.open("w") as fh:
        subprocess.run(cmd, env=env, cwd="/tmp", stdout=fh, stderr=subprocess.STDOUT)
    if not out_json.is_file():
        return None
    doc = json.loads(out_json.read_text())
    if not doc.get("all_preconditions_held", True):
        missing = sorted(k for k, v in (doc.get("exercised") or {}).items() if not v)
        print(f"  preconditions did not hold ({missing}); this run measured a different "
              f"configuration and its number is not comparable.")
        return None
    return doc if doc.get("tokens_per_second") else None


def _point_value(doc: dict | None, point: dict) -> float | None:
    """The reading a point is compared on: total tokens/s unless the point names another key."""
    if not doc:
        return None
    key = str(point.get("metric_key") or "throughput.total_tokens_per_sec")
    value = doc.get("mean_ttft_ms") if key == "latency_ms.ttft.mean" else doc.get("tokens_per_second")
    return float(value) if isinstance(value, (int, float)) else None


def _stage_pristine(ws: Path, dest: Path) -> Path | None:
    """A pristine copy of the pinned tree, from repo/'s own `inference_base` tag."""
    repo = ws / "repo"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    proc = subprocess.run(["git", "-C", str(repo), "archive", "inference_base"], capture_output=True)
    if proc.returncode != 0:
        print(f"[run_benchmark] cannot stage pristine: {proc.stderr.decode()[:300]}", file=sys.stderr)
        return None
    if subprocess.run(["tar", "-x", "-C", str(dest)], input=proc.stdout).returncode != 0:
        return None
    for pat in ("*.so", "_version.py"):
        for src in (repo / "vllm").glob(pat):
            tgt = dest / "vllm" / src.name
            if not tgt.exists():
                tgt.symlink_to(src.resolve())
    fa_src, fa_dst = repo / "vllm" / "vllm_flash_attn", dest / "vllm" / "vllm_flash_attn"
    if fa_src.is_dir() and fa_dst.is_dir():
        for src in fa_src.glob("*.so"):
            tgt = fa_dst / src.name
            if not tgt.exists():
                tgt.symlink_to(src.resolve())
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--rounds", type=int, default=1, help="alternating A/B rounds")
    ap.add_argument("--with-baseline", action="store_true",
                    help="also bench a pristine copy staged from the inference_base tag")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--workloads", default="all",
                    help="`all` (every scored point of public/workloads.json, each its own boot), or a "
                         "comma-separated list; the anchor is always benched")
    args = ap.parse_args()

    ws = _workspace()
    iters = ws / "iterations"
    iters.mkdir(exist_ok=True)
    if args.output is None:
        used = [int(m.group(1)) for p in iters.glob("iter_*.json")
                if (m := re.match(r"iter_(\d+)\.json$", p.name))]
        args.output = iters / f"iter_{max(used, default=0) + 1}.json"

    raw = iters / ".raw"
    raw.mkdir(exist_ok=True)
    cand_tree = ws / "repo"
    base_tree = _stage_pristine(ws, ws / ".baseline_tree") if args.with_baseline else None
    if args.with_baseline and base_tree is None:
        print("[run_benchmark] baseline staging failed; benching the candidate only", file=sys.stderr)

    if not args.no_warmup:
        for name, tree in (("candidate", cand_tree), ("baseline", base_tree)):
            if tree is None:
                continue
            print(f"[run_benchmark] warm-up (discarded), {name}...", flush=True)
            _bench(tree, ws, raw / f"warmup_{name}.json", raw / f"warmup_{name}.log")

    matrix = _matrix(ws)
    anchor = matrix["anchor"]
    benched = [n for n in matrix["benched"]
               if args.workloads == "all" or n == anchor or n in args.workloads.split(",")]
    docs: dict[str, dict[str, list[dict]]] = {"candidate": {n: [] for n in benched},
                                              "baseline": {n: [] for n in benched}}
    for r in range(1, args.rounds + 1):
        order = [("candidate", cand_tree)]
        if base_tree is not None:
            order.append(("baseline", base_tree))
        if r % 2 == 0:
            order.reverse()
        for label, tree in order:
            for n in benched:
                t0 = time.time()
                doc = _bench(tree, ws, raw / f"r{r}_{label}_{n}.json", raw / f"r{r}_{label}_{n}.log", point=n)
                tps = doc.get("tokens_per_second") if doc else None
                print(
                    f"[run_benchmark] r{r} {label:9s} {n:20s} "
                    + (f"{tps:9.1f} tok/s" if tps else "FAILED (see iterations/.raw/)")
                    + f"   [{time.time() - t0:.0f}s]",
                    flush=True,
                )
                if doc:
                    docs[label][n].append(doc)

    def mean_value(label: str, name: str) -> float | None:
        pt = matrix["points"][name]
        source = pt["from"] if isinstance(pt.get("from"), str) else name
        values = [v for v in (_point_value(d, pt) for d in docs[label].get(source, [])) if v is not None]
        return statistics.mean(values) if values else None

    names_out = benched + [g for g in matrix["guards"] if g not in benched]
    result: dict[str, object] = {
        "workload": matrix["points"][anchor],
        "rounds": args.rounds,
        "candidate_tokens_per_second": mean_value("candidate", anchor),
        "baseline_tokens_per_second": mean_value("baseline", anchor),
        "points": {n: {"role": "scored" if n in matrix["scored"] else "guard",
                       "candidate": mean_value("candidate", n), "baseline": mean_value("baseline", n)}
                   for n in names_out},
    }
    per_point: dict[str, float] = {}
    for n in names_out:
        c, b = mean_value("candidate", n), mean_value("baseline", n)
        if c and b:
            lower = bool(matrix["points"][n].get("lower_is_better", False))
            per_point[n] = (b / c) if lower else (c / b)
    if per_point:
        result["speedup_per_point"] = per_point
        scored = [per_point[n] for n in matrix["scored"] if n in per_point]
        if len(scored) == len([n for n in matrix["scored"] if n in benched]):
            result["speedup"] = _geomean(scored)
        result["guards"] = {n: per_point.get(n) for n in matrix["guards"]}
    cand_anchor = [v for v in (_point_value(d, matrix["points"][anchor]) for d in docs["candidate"][anchor]) if v]
    base_anchor = [v for v in (_point_value(d, matrix["points"][anchor]) for d in docs["baseline"][anchor]) if v]
    if len(cand_anchor) > 1:
        result["candidate_sd"] = statistics.stdev(cand_anchor)
    if len(base_anchor) > 1:
        result["baseline_sd"] = statistics.stdev(base_anchor)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n[run_benchmark] wrote {args.output}")
    for n, s in (result.get("speedup_per_point") or {}).items():
        print(f"[run_benchmark] {n:20s} {s:.4f}x  ({'scored' if n in matrix['scored'] else 'guard'})")
    if result.get("speedup"):
        print(f"[run_benchmark] speedup = {result['speedup']:.4f}  (geometric mean over the scored points; "
              f"the score is the measured speedup itself, and a guard shape that reads slower than the pristine tree "
              f"scores zero)")
    elif not args.with_baseline:
        print("[run_benchmark] no ratio: pass --with-baseline to bench the pinned tree too")
    return 0 if any(docs["candidate"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
