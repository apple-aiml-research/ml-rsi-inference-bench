#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Throughput bench runner for regression_spec_decode_acceptance_drop.

Optimize-style per-task runner (mirrors tasks/optimize_*/tests/_candidate_runner.py).
test.sh invokes it as a subprocess for BOTH the pristine-baseline side and the
candidate side; the verifier then reads the two throughput JSONs and computes the
relative speedup.

Design points that fix the earlier regression-bench breakage:
  * Workload shape comes from the shipped public/workloads.json (the same file the
    agent's run_benchmark.py reads). task.toml is NOT available in the verifier pod,
    so we do not read it. Reading an agent-visible shape is safe because scoring is a
    ratio (candidate / pristine at the SAME shape) — shrinking it cannot be gamed.
  * We pass BOTH --input-len and --random-input-len (= the intended value). vLLM>=0.19
    `bench throughput` adds --random-input-len (default 1024) and PREFERS it over
    --input-len, which otherwise overruns --max-model-len -> AssertionError. The flag
    is only added when the installed vLLM exposes it (0.9.0 does not).
  * Per-task bench knobs are baked in CONFIG (sealed under tests/), the same way
    optimize keeps per-task runner variants.

Writes vllm_bench_throughput.json + vllm_bench_manifest.json into --raw-dir, the
contract verify_regression_real.py checks (throughput JSON present + manifest
returncode == 0 + command mentions "vllm"/"bench").
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

CONFIG = {
    "enforce_eager": False,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.85,
    "draft_tokens": 7,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 8192,
    "cudagraph_capture": 2048,
    "stream_interval": 20,
    "kv_cache_dtype": "fp8",
    "max_model_len_floor": 2048,
    "dataset_name": "custom",
    "env_required": {},
}

CORRECTNESS = {
    "num_prompts": 8,
    "output_len": 128,
    "max_num_seqs": 1,
}

BENCH_HARNESS = "bench_spec_decode.py"

def _assert_benched_tree(python: str, repo: Path, env: dict[str, str]) -> str:
    """Which tree did `import vllm` actually resolve to? Refuse to bench if it is not `repo`."""
    probe = subprocess.run(
        [python, "-c", "import vllm, sys; sys.stdout.write(vllm.__file__)"],
        env=env, cwd="/tmp", capture_output=True, text=True,
    )
    resolved = (probe.stdout or "").strip()
    if probe.returncode != 0 or not resolved:
        raise SystemExit(
            f"cannot import vllm with PYTHONPATH={env.get('PYTHONPATH')!r}: "
            f"{(probe.stderr or '')[-500:]}"
        )
    if not resolved.startswith(str(repo)):
        raise SystemExit(
            f"REFUSING to bench: `import vllm` resolved to {resolved}, not the tree under test "
            f"({repo}). Both arms would measure the same code and the ratio would be meaningless."
        )
    return resolved


def _workspace() -> Path:
    w = os.environ.get("INFB_WORKSPACE")
    return Path(w) if w and Path(w).is_dir() else Path("/app")


def _venv_python() -> str:
    v = os.environ.get("INFB_VLLM_PYTHON")
    if v and Path(v).is_file():
        return v
    venv = os.environ.get("INFB_VENV")
    if venv and (Path(venv) / "bin" / "python").is_file():
        return str(Path(venv) / "bin" / "python")
    return sys.executable


def _model(ws: Path) -> str:
    v = os.environ.get("INFB_MODEL_PATH")
    if v and Path(v).exists():
        return v
    link = ws / "model"
    if link.exists():
        return str(link)
    raise SystemExit("model not resolved (set INFB_MODEL_PATH or <workspace>/model)")


def _primary_workload(ws: Path) -> dict:
    doc = json.loads((Path(__file__).resolve().parent / "workloads.json").read_text())
    wls = doc["workloads"]
    name = doc.get("primary_workload")
    if name:
        for w in wls:
            if w.get("name") == name:
                return w
    return wls[0]


def _select_extra_points(selection: str, spec: dict) -> list[str]:
    """The non-anchor points this invocation benches, scored ones first, then guards."""
    if selection in ("", "anchor"):
        return []
    candidates = [n for n in spec["scored"] if n != spec["anchor"]] + list(spec["guards"])
    if selection != "all":
        wanted = [s.strip() for s in selection.split(",") if s.strip()]
        unknown = sorted(set(wanted) - set(spec["points"]))
        if unknown:
            raise SystemExit(f"--workloads names points the spec does not declare: {unknown}")
        candidates = [n for n in candidates if n in wanted]
    return [n for n in candidates if not isinstance(spec["points"][n].get("from"), str)]


def _supports_random_input(python: str) -> bool:
    """vLLM>=0.10 `bench throughput` exposes --random-input-len; 0.9.0 does not."""
    try:
        return subprocess.run(
            [python, "-c",
             "import sys,importlib.metadata as M;"
             "v=M.version('vllm').split('+')[0].split('.');"
             "sys.exit(0 if (int(v[0]),int(v[1]))>=(0,10) else 1)"],
            timeout=120,
        ).returncode == 0
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["throughput", "correctness"])
    ap.add_argument("--repo", required=True, help="engine source tree to overlay via PYTHONPATH")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument(
        "--no-speculation", action="store_true",
        help="serve without a drafter. Only meaningful in `correctness` mode, where this arm is "
             "the reference: what the target model answers on its own.",
    )
    ap.add_argument(
        "--workloads", default="anchor",
        help="`anchor` (the primary workload only, the warmup default), `all` (every scored point of "
             "tests/workloads.json, each on its own server boot because a point's concurrency is the "
             "server's max_num_seqs), or a comma-separated list. Ignored in `correctness` mode.",
    )
    args = ap.parse_args()

    ws = _workspace()
    repo = Path(args.repo).resolve()
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    out_json = raw / "vllm_bench_throughput.json"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _verifier_lib.bench_stats import load_matrix_spec, point_raw_dir  # noqa: E402
    spec = load_matrix_spec(Path(__file__).resolve().parent / "workloads.json")
    extra_points = [] if args.mode == "correctness" else _select_extra_points(args.workloads, spec)
    points_benched: list[dict] = []
    manifest = raw / "vllm_bench_manifest.json"

    python = _venv_python()
    model = _model(ws)
    wl = _primary_workload(ws)
    in_len, out_len = int(wl["input_len"]), int(wl["output_len"])
    num_prompts = int(wl["num_prompts"])
    max_num_seqs = int(CONFIG["max_num_seqs"])
    if args.mode == "correctness":
        out_len = int(CORRECTNESS["output_len"])
        num_prompts = int(CORRECTNESS["num_prompts"])
        max_num_seqs = int(CORRECTNESS["max_num_seqs"])
    max_len = int(wl.get("max_model_len") or 0) or max(
        CONFIG["max_model_len_floor"], in_len + out_len + 256
    )
    random_ok = _supports_random_input(python)

    harness = Path(__file__).resolve().parent / BENCH_HARNESS

    probe_json = raw / "async_path_probe.json"
    probe_rc = None
    probe_env = dict(os.environ)
    probe_env["PYTHONPATH"] = f"{repo}:{probe_env.get('PYTHONPATH', '')}".rstrip(":")
    with (raw / "async_path_probe.log").open("w") as pf:
        probe_rc = None if args.mode == "correctness" else subprocess.run(
            [python, str(harness), "--workspace", str(ws),
             "--probe",
             "--dataset-name", CONFIG["dataset_name"],
             "--max-model-len", str(max_len),
             "--tensor-parallel-size", str(CONFIG["tensor_parallel_size"]),
             "--draft-tokens", str(CONFIG["draft_tokens"]),
             "--kv-cache-dtype", CONFIG["kv_cache_dtype"],
             "--output-json", str(probe_json)],
            env=probe_env, cwd="/tmp", stdout=pf, stderr=subprocess.STDOUT,
        ).returncode

    cmd = [
        python, str(harness),
        "--workspace", str(ws),
        "--input-len", str(in_len),
        "--output-len", str(out_len),
        "--num-prompts", str(num_prompts),
        "--max-num-seqs", str(max_num_seqs),
        "--max-model-len", str(max_len),
        "--dataset-name", CONFIG["dataset_name"],
        "--tensor-parallel-size", str(CONFIG["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(CONFIG["gpu_memory_utilization"]),
        "--draft-tokens", str(CONFIG["draft_tokens"]),
        "--kv-cache-dtype", CONFIG["kv_cache_dtype"],
        "--max-num-batched-tokens", str(CONFIG["max_num_batched_tokens"]),
        "--cudagraph-capture", str(CONFIG["cudagraph_capture"]),
        "--stream-interval", str(CONFIG["stream_interval"]),
        "--output-json", str(out_json),
    ]
    if CONFIG["enforce_eager"]:
        cmd.append("--enforce-eager")
    if args.no_speculation:
        cmd.append("--no-speculation")

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}".rstrip(":")
    for k, v in CONFIG["env_required"].items():
        env[k] = v

    resolved = _assert_benched_tree(python, repo, env)

    t0 = time.time()
    with (raw / "vllm_bench_stderr.txt").open("w") as errf, \
         (raw / "vllm_bench_stdout.txt").open("w") as outf:
        rc = subprocess.run(cmd, env=env, stdout=outf, stderr=errf).returncode
    for name in extra_points:
        point = spec["points"][name]
        pdir = point_raw_dir(raw, name, spec["anchor"])
        pdir.mkdir(parents=True, exist_ok=True)
        p_out = pdir / "vllm_bench_throughput.json"
        p_out.unlink(missing_ok=True)
        pcmd = list(cmd)
        for flag, value in (("--output-len", int(point["output_len"])),
                            ("--num-prompts", int(point["num_prompts"])),
                            ("--max-num-seqs", int(point.get("max_concurrency") or max_num_seqs)),
                            ("--output-json", str(p_out))):
            i = pcmd.index(flag)
            pcmd[i + 1] = str(value)
        print(f"[_candidate_runner] point {name}: c={point.get('max_concurrency')} n={point['num_prompts']}",
              flush=True)
        pt0 = time.time()
        with (pdir / "vllm_bench_stderr.txt").open("w") as perrf, \
             (pdir / "vllm_bench_stdout.txt").open("w") as poutf:
            prc = subprocess.run(pcmd, env=env, stdout=poutf, stderr=perrf).returncode
        error = None
        if prc != 0:
            error = f"harness exited rc={prc}" + (" (preconditions did not hold)" if prc == 2 else "")
            if p_out.exists():
                p_out.rename(pdir / "vllm_bench_throughput.refused.json")
        elif not p_out.is_file():
            error = "harness wrote no record"
        else:
            print(f"[_candidate_runner] point {name}: done in {time.time() - pt0:.0f}s", flush=True)
        if error:
            print(f"[_candidate_runner] point {name}: FAILED -- {error}", flush=True)
        (pdir / "vllm_bench_manifest.json").write_text(json.dumps({
            "command": " ".join(shlex.quote(c) for c in pcmd), "returncode": prc, "error": error,
            "point": name, "mode": args.mode, "elapsed_sec": round(time.time() - pt0, 1),
            "workload": {"input_len": in_len, "output_len": int(point["output_len"]),
                         "num_prompts": int(point["num_prompts"]),
                         "max_concurrency": int(point.get("max_concurrency") or max_num_seqs),
                         "max_model_len": max_len},
        }, indent=2))
        points_benched.append({"name": name, "returncode": prc, "error": error})
    elapsed = time.time() - t0

    manifest.write_text(json.dumps({
        "command": " ".join(shlex.quote(c) for c in cmd),
        "returncode": rc,
        "benched_tree": resolved,
        "async_path_probe_rc": probe_rc,
        "mode": args.mode,
        "no_speculation": bool(args.no_speculation),
        "max_num_seqs": max_num_seqs,
        "elapsed_sec": elapsed,
        "random_input_len_used": random_ok,
        "workloads_benched": [wl["name"]] + [pt["name"] for pt in points_benched if not pt["error"]],
        "points": points_benched,
        "workload": {
            "input_len": in_len, "output_len": out_len,
            "num_prompts": num_prompts, "max_model_len": max_len,
        },
    }, indent=2))
    print(f"[_candidate_runner] repo={repo} rc={rc} elapsed={elapsed:.1f}s "
          f"in={in_len} out={out_len} n={num_prompts} maxlen={max_len} "
          f"random_flags={random_ok}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
