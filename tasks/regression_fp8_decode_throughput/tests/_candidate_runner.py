#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""SERVE bench runner for regression_fp8_decode_throughput.

Optimize-style per-task runner (mirrors tasks/optimize_*/tests/_candidate_runner.py).
tests/_bench_rounds.py invokes it once per arm per round -- a discarded warmup on the
pristine tree, then rotated scored rounds of both trees -- and the verifier scores the
median of the per-round paired mean-TPOT ratios.

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

Writes vllm_bench_serve.json + vllm_bench_manifest.json (and the server log and the
correctness recording) into --raw-dir, the contract verify_regression_real.py checks
(serve JSON present + manifest returncode == 0 + command mentions "vllm"/"bench").
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _verifier_lib.bench_stats import load_matrix_spec, point_raw_dir  # noqa: E402

CONFIG = {
    "enforce_eager": False,
    "load_format": "auto",
    "data_parallel_size": 8,
    "enable_expert_parallel": True,
    "gpu_memory_utilization": 0.90,
    "trust_remote_code": True,
    "env_required": {
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "VLLM_USE_FLASHINFER_MOE_FP4": "1",
    },
    "max_num_seqs": 2048,
    "max_cudagraph_capture_size": 2048,
    "stream_interval": 100,
    "max_model_len_floor": 1280,
}


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


def _point_bench_cmd(python: str, model: str, port: int, wl: dict, out_json: Path) -> list[str]:
    """The anchor's `vllm bench serve` line, with the point's own shape and result file."""
    cmd = [
        python, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm", "--model", model, "--tokenizer", model,
        "--port", str(port), "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(int(wl["input_len"])), "--random-output-len", str(int(wl["output_len"])),
        "--max-concurrency", str(int(wl.get("max_concurrency") or wl["num_prompts"])),
        "--num-prompts", str(int(wl["num_prompts"])),
        "--ignore-eos",
        "--save-result", "--result-filename", str(out_json),
        "--temperature", "1.0", "--top-p", "1.0", "--top-k", "0",
    ]
    if CONFIG["trust_remote_code"]:
        cmd.append("--trust-remote-code")
    return cmd


def _bench_point(python: str, env: dict, model: str, port: int, name: str, wl: dict,
                 pdir: Path, serve_cmd: list[str]) -> dict:
    """One extra point against the already-running server; its own record and manifest under `pdir`."""
    pdir.mkdir(parents=True, exist_ok=True)
    out_json = pdir / "vllm_bench_serve.json"
    out_json.unlink(missing_ok=True)
    cmd = _point_bench_cmd(python, model, port, wl, out_json)
    print(f"[_candidate_runner] point {name}: c={wl.get('max_concurrency')} n={wl['num_prompts']}", flush=True)
    t0 = time.time()
    with (pdir / "vllm_bench_stderr.txt").open("w") as errf, (pdir / "vllm_bench_stdout.txt").open("w") as outf:
        rc = subprocess.run(cmd, env=env, stdout=outf, stderr=errf).returncode
    error = None
    if rc != 0:
        error = f"bench exited rc={rc}"
    elif not out_json.is_file():
        error = "bench wrote no result file"
    else:
        try:
            doc = json.loads(out_json.read_text())
        except ValueError as exc:
            doc, error = {}, f"unreadable result: {exc}"
        if not error:
            if int(doc.get("completed", 0)) != int(wl["num_prompts"]):
                error = f"completed {doc.get('completed')} of {wl['num_prompts']} requests"
            elif not isinstance(doc.get("mean_tpot_ms"), (int, float)):
                error = "no mean_tpot_ms in the result"
            else:
                print(f"[_candidate_runner] point {name}: mean TPOT {doc['mean_tpot_ms']:.3f} ms  "
                      f"output {doc.get('output_throughput', float('nan')):.1f} tok/s", flush=True)
    if error:
        print(f"[_candidate_runner] point {name}: FAILED -- {error}", flush=True)
    (pdir / "vllm_bench_manifest.json").write_text(json.dumps({
        "command": " ".join(shlex.quote(c) for c in cmd),
        "serve_command": " ".join(shlex.quote(c) for c in serve_cmd),
        "returncode": rc if rc else (1 if error else 0), "error": error, "point": name,
        "elapsed_sec": round(time.time() - t0, 2),
        "workload": {k: wl.get(k) for k in ("input_len", "output_len", "num_prompts", "max_concurrency", "max_model_len")},
    }, indent=2))
    return {"name": name, "returncode": rc, "error": error}


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
    ap.add_argument("mode", choices=["serve"])
    ap.add_argument("--repo", required=True, help="engine source tree to overlay via PYTHONPATH")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument(
        "--workloads", default="anchor",
        help="`anchor` (the primary workload only, the warmup default), `all` (every scored and guard "
             "point of tests/workloads.json against the same server), or a comma-separated list",
    )
    args = ap.parse_args()

    ws = _workspace()
    repo = Path(args.repo).resolve()
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    out_json = raw / "vllm_bench_serve.json"
    manifest = raw / "vllm_bench_manifest.json"
    serve_log = raw / "vllm_serve.log"

    python = _venv_python()
    model = _model(ws)
    wl = _primary_workload(ws)
    spec = load_matrix_spec(Path(__file__).resolve().parent / "workloads.json")
    extra_points = _select_extra_points(args.workloads, spec)
    points_benched: list[dict] = []
    in_len, out_len = int(wl["input_len"]), int(wl["output_len"])
    num_prompts = int(wl["num_prompts"])
    concurrency = int(wl.get("max_concurrency") or num_prompts)
    max_len = int(wl.get("max_model_len") or 0) or max(
        CONFIG["max_model_len_floor"], in_len + out_len + 256
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}".rstrip(":")
    for k, v in CONFIG["env_required"].items():
        env[k] = v

    resolved = _assert_benched_tree(python, repo, env)

    port = int(os.environ.get("INFB_BENCH_PORT", "8791"))
    serve_cmd = [
        python, "-m", "vllm.entrypoints.cli.main", "serve", model,
        "--port", str(port),
        "--data-parallel-size", str(CONFIG["data_parallel_size"]),
        "--max-num-seqs", str(CONFIG["max_num_seqs"]),
        "--max-num-batched-tokens", "16384",
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", str(CONFIG["gpu_memory_utilization"]),
        "--max-cudagraph-capture-size", str(CONFIG["max_cudagraph_capture_size"]),
        "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        "--stream-interval", str(CONFIG["stream_interval"]),
        "--load-format", CONFIG["load_format"],
        "--disable-log-requests",
    ]
    if CONFIG["enable_expert_parallel"]:
        serve_cmd.append("--enable-expert-parallel")
    if CONFIG["trust_remote_code"]:
        serve_cmd.append("--trust-remote-code")

    bench_cmd = [
        python, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm", "--model", model, "--tokenizer", model,
        "--port", str(port), "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(in_len), "--random-output-len", str(out_len),
        "--max-concurrency", str(concurrency), "--num-prompts", str(num_prompts),
        "--ignore-eos",
        "--save-result", "--result-filename", str(out_json),
        "--temperature", "1.0", "--top-p", "1.0", "--top-k", "0",
    ]
    if CONFIG["trust_remote_code"]:
        bench_cmd.append("--trust-remote-code")

    t0 = time.time()
    rc = 2
    warmups = int(os.environ.get("INFB_WARMUP_BOOTS", "1"))
    with serve_log.open("w") as slf:
        proc = subprocess.Popen(serve_cmd, env=env, stdout=slf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            _wait_ready(port, proc, int(os.environ.get("INFB_BOOT_TIMEOUT_SEC", "3600")))
            for i in range(warmups):
                print(f"[_candidate_runner] warm-up sweep {i + 1}/{warmups} (full size)", flush=True)
                subprocess.run(bench_cmd, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with (raw / "vllm_bench_stderr.txt").open("w") as errf, \
                 (raw / "vllm_bench_stdout.txt").open("w") as outf:
                rc = subprocess.run(bench_cmd, env=env, stdout=outf, stderr=errf).returncode
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                import _correctness
                tests_dir = Path(__file__).resolve().parent
                prompts = _correctness.load_prompts(tests_dir)
                max_tok = int(os.environ.get("INFB_CORRECTNESS_MAX_TOKENS", "256"))
                role = ("baseline"
                        if raw.parent.name in ("baseline", "pristine")
                        or raw.parent.parent.name == "warmup"
                        else "candidate")
                records = _correctness.collect_generation(port, model, prompts, max_tok)
                (raw / "correctness_recording.json").write_text(
                    json.dumps({"role": role, "prompts": len(prompts),
                                "max_tokens": max_tok, "records": records}, indent=2)
                )
                print(f"[_candidate_runner] correctness: {role}, {len(records)} prompts",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 -- never let this kill a scored bench
                (raw / "correctness_recording.json").write_text(
                    json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)
                )
                print(f"[_candidate_runner] correctness FAILED to record: {exc}", flush=True)
            for name in extra_points:
                points_benched.append(_bench_point(
                    python, env, model, port, name, spec["points"][name],
                    point_raw_dir(raw, name, spec["anchor"]), serve_cmd))
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            time.sleep(30)
    elapsed = time.time() - t0

    manifest.write_text(json.dumps({
        "command": " ".join(shlex.quote(c) for c in bench_cmd),
        "serve_command": " ".join(shlex.quote(c) for c in serve_cmd),
        "returncode": rc,
        "benched_tree": resolved,
        "elapsed_sec": elapsed,
        "warmup_boots": warmups,
        "workloads_benched": [wl["name"]] + [pt["name"] for pt in points_benched if not pt["error"]],
        "points": points_benched,
        "workload": {
            "input_len": in_len, "output_len": out_len, "num_prompts": num_prompts,
            "max_concurrency": concurrency, "max_model_len": max_len,
        },
    }, indent=2))
    print(f"[_candidate_runner] repo={repo} rc={rc} elapsed={elapsed:.1f}s "
          f"in={in_len} out={out_len} n={num_prompts} c={concurrency} maxlen={max_len}", flush=True)
    return 0


def _wait_ready(port: int, proc: subprocess.Popen, timeout: int) -> None:
    """Block until /health answers, and DUMP THE SERVER LOG if the process dies first."""
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"vllm serve exited early rc={proc.returncode}; see vllm_serve.log")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(10)
    raise SystemExit("vllm serve never became ready")


if __name__ == "__main__":
    raise SystemExit(main())
