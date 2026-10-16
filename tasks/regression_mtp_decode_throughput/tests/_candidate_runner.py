#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Serve bench runner for regression_mtp_decode_throughput.

Optimize-style per-task runner (mirrors tasks/optimize_*/tests/_candidate_runner.py).
tests/_bench_rounds.py invokes it once per ARM per ROUND -- discarded warmups on the pristine
tree, then rotated scored rounds of both arms, every invocation over the whole matrix -- and the
verifier reads the aggregate.

ONE FRESH ENGINE SESSION PER SCORED POINT, THE GUARDS ON THE ANCHOR'S. This task's defect is a per-
server tax: the pristine kernel compiles once per batch shape a server has not seen, so the first
bench a fresh server runs pays for every shape it visits and every later bench on that server pays
nothing. This was measured on one server per arm-round: a high ratio at the
anchor and roughly parity at every point benched after it, whatever its shape. So each SCORED point boots a
`vllm serve` of its own, carrying every engine flag the offline command carried, and is driven with
`vllm bench serve`: the anchor first, at the raw root, then the other scored points under
`wl/<name>/`, each with its own boot; the guards run against the anchor's server after its bench,
warm, under `wl/<name>/` too -- a guard reads steady state, which is what a repair must not cost.
Concurrency is a CLIENT knob (`--max-concurrency`) against servers whose `--max-num-seqs` stays at
the anchor's 256 for every point: the matrix varies the request stream, never the deployment.

ONE COLD JIT CACHE PER SESSION, created before each boot under the host point's own raw directory
(`<raw>/bench_jit/` for the anchor, `<raw>/wl/<name>/bench_jit/` for the other scored points), which
the rounds driver makes unique per bench and per arm (`<artifacts>/rounds/<i>/<arm>/raw`). A cache
that survived from one session into the next would hand the second every specialisation the first
compiled and erase the thing being measured. tests/assert_gdn_kernel_recompiles.py counts the
anchor session's specialisations under `<raw>/bench_jit/`, exactly as before.

Design points kept from the offline runner:
  * Workload shape comes from the TRUSTED tests/workloads.json beside this file (the same body the
    agent's public/workloads.json carries). task.toml is NOT available in the verifier pod.
  * Per-task bench knobs are baked in CONFIG (sealed under tests/), the same way optimize keeps
    per-task runner variants.

Writes vllm_bench_serve.json + vllm_bench_manifest.json (+ vllm_serve.log, the client's stdout and
stderr) into --raw-dir, the contract verify_regression_real.py checks (serve JSON present +
manifest returncode == 0 + command mentions "vllm"/"bench"), and one record + manifest per extra
point under wl/<name>/. A point that fails keeps no record, so the rounds driver reads it as
unpairable rather than as a number.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import shutil
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
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.90,
    "trust_remote_code": True,
    "env_required": {},
    "max_num_seqs": 256,
    "max_num_batched_tokens": 16384,
    "max_model_len_floor": 4096,
    "speculative_config": '{"method":"qwen3_next_mtp","num_speculative_tokens":2}',
    "cudagraph_mode": "PIECEWISE",
    "bench_mode": "serve",
    "engine_seed": 12345,
    "port": int(os.environ.get("INFB_BENCH_PORT", "8791")),
    "timeout_sec": int(os.environ.get("INFB_BENCH_TIMEOUT_SEC", "3000")),
    "boot_timeout_sec": int(os.environ.get("INFB_BOOT_TIMEOUT_SEC", "1200")),
}


def _cuda_graph_size_ceiling() -> int:
    """The largest token count a decode step of this workload actually submits."""
    num_spec = int(json.loads(CONFIG["speculative_config"])["num_speculative_tokens"])
    return CONFIG["max_num_seqs"] * (1 + num_spec)


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


def _cold_jit_cache(raw: Path) -> Path:
    """A fresh, empty compile-cache root for THIS session: `<its point's raw dir>/bench_jit/`."""
    root = raw / "bench_jit"
    shutil.rmtree(root, ignore_errors=True)
    (root / "triton").mkdir(parents=True, exist_ok=True)
    (root / "vllm").mkdir(parents=True, exist_ok=True)
    return root


def _wait_for_free_gpu(timeout_s: float = 300.0, floor_mib: int = 4096) -> str:
    """Block until the previous engine has released the GPUs, or say why we gave up."""
    if not shutil.which("nvidia-smi"):
        return "nvidia-smi absent; not waiting"
    deadline = time.time() + timeout_s
    last = -1
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30).stdout
            last = max(int(v.strip()) for v in out.split() if v.strip().isdigit())
        except (OSError, ValueError, subprocess.SubprocessError):
            return "nvidia-smi unreadable; not waiting"
        if last <= floor_mib:
            return f"free after {timeout_s - (deadline - time.time()):.0f}s ({last} MiB used)"
        time.sleep(5)
    return f"STILL BUSY after {timeout_s:.0f}s ({last} MiB used) -- benching anyway"


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


def _run_seed() -> int:
    """The per-eval prompt seed: INFB_BENCH_SEED when pinned (the rounds driver exports one value for
    the whole verifier process), time-based otherwise -- optimize's convention."""
    raw = os.environ.get("INFB_BENCH_SEED")
    return int(raw) if raw and raw.strip() else (int(time.time()) % (2 ** 31))


def _serve_cmd(python: str, model: str, port: int, max_len: int) -> list[str]:
    """The ONE server every point of an invocation runs against."""
    cmd = [
        python, "-m", "vllm.entrypoints.cli.main", "serve", model,
        "--port", str(port),
        "--tensor-parallel-size", str(CONFIG["tensor_parallel_size"]),
        "--max-num-seqs", str(CONFIG["max_num_seqs"]),
        "--max-num-batched-tokens", str(CONFIG["max_num_batched_tokens"]),
        "--max-model-len", str(max_len),
        "--speculative-config", CONFIG["speculative_config"],
        "-O.cudagraph_mode=" + CONFIG["cudagraph_mode"],
        "--cuda-graph-sizes", str(_cuda_graph_size_ceiling()),
        "--gpu-memory-utilization", str(CONFIG["gpu_memory_utilization"]),
        "--load-format", CONFIG["load_format"],
        "--no-enable-chunked-prefill",
        "--seed", str(CONFIG["engine_seed"]),
        "--disable-log-stats",
    ]
    if CONFIG["trust_remote_code"]:
        cmd.append("--trust-remote-code")
    if CONFIG["enforce_eager"]:
        cmd.append("--enforce-eager")
    return cmd


def _point_bench_cmd(python: str, model: str, port: int, wl: dict, out_json: Path, seed: int) -> list[str]:
    """One point's `vllm bench serve` line: the point's own shape, seed and result file."""
    cmd = [
        python, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm", "--model", model, "--tokenizer", model,
        "--port", str(port), "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(int(wl["input_len"])), "--random-output-len", str(int(wl["output_len"])),
        "--random-range-ratio", "0.0",
        "--max-concurrency", str(int(wl.get("max_concurrency") or wl["num_prompts"])),
        "--num-prompts", str(int(wl["num_prompts"])),
    ]
    if wl.get("request_rate") is not None:
        cmd += ["--request-rate", str(float(wl["request_rate"])), "--burstiness", "1.0"]
    cmd += [
        "--ignore-eos",
        "--seed", str(seed),
        "--save-result", "--result-filename", str(out_json),
        "--temperature", "1.0", "--top-p", "1.0", "--top-k", "0",
    ]
    if CONFIG["trust_remote_code"]:
        cmd.append("--trust-remote-code")
    return cmd


def _kill_group(proc: subprocess.Popen, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(proc.pid), sig)


def _stop_server(proc: subprocess.Popen) -> None:
    """Stop the server's whole process group (the API server, the engine core, the TP workers)."""
    if proc.poll() is None:
        _kill_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            _kill_group(proc, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)


def _wait_ready(port: int, proc: subprocess.Popen, timeout: float) -> None:
    """Block until /health answers, and NAME THE SERVER LOG if the process dies first."""
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"vllm serve exited early rc={proc.returncode}; see vllm_serve.log")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001 -- not up yet
            pass
        time.sleep(5)
    raise SystemExit(f"vllm serve never became ready within {timeout:.0f}s; see vllm_serve.log")


def _bench_point(python: str, env: dict, model: str, port: int, name: str, wl: dict, pdir: Path,
                 serve_cmd: list[str], seed: int, deadline: float, server: subprocess.Popen,
                 *, write_manifest: bool = True) -> dict:
    """One point against the already-running server; its own record (and manifest) under `pdir`."""
    pdir.mkdir(parents=True, exist_ok=True)
    out_json = pdir / "vllm_bench_serve.json"
    out_json.unlink(missing_ok=True)
    cmd = _point_bench_cmd(python, model, port, wl, out_json, seed)
    want = int(wl["num_prompts"])
    print(f"[_candidate_runner] point {name}: c={wl.get('max_concurrency')} n={want}", flush=True)
    t0 = time.time()
    rc, error, timed_out = 1, None, False
    summary: dict[str, object] = {}
    budget = deadline - time.time()
    if server.poll() is not None:
        error = f"vllm serve exited rc={server.returncode} before this point; see vllm_serve.log"
    elif budget < 30:
        error = f"invocation deadline ({CONFIG['timeout_sec']}s) reached before this point"
    else:
        with (pdir / "vllm_bench_stderr.txt").open("w") as errf, \
             (pdir / "vllm_bench_stdout.txt").open("w") as outf:
            client = subprocess.Popen(cmd, env=env, stdout=outf, stderr=errf, start_new_session=True)
            try:
                rc = client.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                timed_out, rc = True, 124
                _kill_group(client, signal.SIGKILL)
                client.wait()
        if timed_out:
            error = f"bench client killed at the invocation deadline ({CONFIG['timeout_sec']}s)"
        elif rc != 0:
            error = f"bench exited rc={rc}"
        elif not out_json.is_file():
            error = "bench wrote no result file"
        else:
            try:
                doc = json.loads(out_json.read_text())
            except ValueError as exc:
                doc, error = {}, f"unreadable result: {exc}"
            if not error:
                completed = doc.get("completed")
                failed = doc.get("failed") or 0
                tps = doc.get("total_token_throughput")
                if int(completed or 0) != want:
                    error = f"completed {completed} of {want} requests"
                elif int(failed) != 0:
                    error = f"{failed} request(s) failed"
                elif not isinstance(tps, (int, float)) or tps <= 0:
                    error = "no total_token_throughput in the result"
                else:
                    summary = {
                        "total_token_throughput": float(tps),
                        "output_throughput": doc.get("output_throughput"),
                        "mean_ttft_ms": doc.get("mean_ttft_ms"),
                        "mean_tpot_ms": doc.get("mean_tpot_ms"),
                        "mean_itl_ms": doc.get("mean_itl_ms"),
                        "completed": int(completed),
                    }
                    print(f"[_candidate_runner] point {name}: {float(tps):.1f} total tok/s  "
                          f"TPOT {doc.get('mean_tpot_ms', float('nan')):.2f} ms  "
                          f"TTFT {doc.get('mean_ttft_ms', float('nan')):.0f} ms  "
                          f"({time.time() - t0:.0f}s)", flush=True)
    if error:
        print(f"[_candidate_runner] point {name}: FAILED -- {error}", flush=True)
        if out_json.exists():
            out_json.rename(pdir / "incomplete_result.json")
    entry = {"name": name, "returncode": 0 if not error else (rc or 1), "error": error,
             "timed_out": timed_out, "elapsed_sec": round(time.time() - t0, 1), **summary}
    if write_manifest:
        (pdir / "vllm_bench_manifest.json").write_text(json.dumps({
            "command": " ".join(shlex.quote(c) for c in cmd),
            "serve_command": " ".join(shlex.quote(c) for c in serve_cmd),
            "returncode": entry["returncode"], "error": error, "point": name, "seed": seed,
            "timed_out": timed_out, "elapsed_sec": entry["elapsed_sec"],
            "workload": {k: wl.get(k) for k in ("input_len", "output_len", "num_prompts",
                                                 "max_concurrency", "max_model_len")},
        }, indent=2))
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["serve"])
    ap.add_argument("--repo", required=True, help="engine source tree to overlay via PYTHONPATH")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument(
        "--workloads", default="anchor",
        help="`anchor` (the primary workload only, the warmup default), `all` (every scored and guard "
             "point of tests/workloads.json: each scored point on a server of its own, the guards after "
             "the anchor on its server), or a comma-separated list",
    )
    args = ap.parse_args()

    ws = _workspace()
    repo = Path(args.repo).resolve()
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    manifest = raw / "vllm_bench_manifest.json"

    python = _venv_python()
    model = _model(ws)
    wl = _primary_workload(ws)
    spec = load_matrix_spec(Path(__file__).resolve().parent / "workloads.json")
    extra_points = _select_extra_points(args.workloads, spec)
    scored_extra = [n for n in extra_points if n in spec["scored"]]
    guards = [n for n in extra_points if n not in spec["scored"]]
    in_len, out_len = int(wl["input_len"]), int(wl["output_len"])
    num_prompts = int(wl["num_prompts"])
    concurrency = int(wl.get("max_concurrency") or num_prompts)
    max_len = int(wl.get("max_model_len") or 0) or max(
        CONFIG["max_model_len_floor"], in_len + out_len + 256
    )
    seed = _run_seed()
    port = int(CONFIG["port"])

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}".rstrip(":")
    for k, v in CONFIG["env_required"].items():
        env[k] = v

    resolved = _assert_benched_tree(python, repo, env)
    serve_cmd = _serve_cmd(python, model, port, max_len)
    anchor_cmd = _point_bench_cmd(python, model, port, wl, raw / "vllm_bench_serve.json", seed)

    t0 = time.time()
    deadline = t0 + CONFIG["timeout_sec"]
    anchor: dict = {"name": wl["name"], "returncode": 1, "error": "not benched", "timed_out": False}
    points_benched: list[dict] = []
    sessions: list[dict] = []
    error: str | None = None

    def _session(host: str, host_wl: dict, host_dir: Path, riders: list[str]) -> None:
        """ONE FRESH SERVER: a cold compile cache made before the boot, the host point against it,
        then the riders (the guards, on the anchor's session only). Never raises: a session that
        fails records its host and every rider not yet benched as failed points -- no record, a
        manifest that says why -- and the invocation moves on to its next session."""
        nonlocal anchor, error
        host_dir.mkdir(parents=True, exist_ok=True)
        jit_root = _cold_jit_cache(host_dir)
        senv = dict(env)
        senv["TRITON_CACHE_DIR"] = str(jit_root / "triton")
        senv["VLLM_CACHE_ROOT"] = str(jit_root / "vllm")
        gpu_wait = _wait_for_free_gpu(timeout_s=min(300.0, max(1.0, deadline - time.time())))
        print(f"[_candidate_runner] gpu: {gpu_wait}", flush=True)
        record: dict = {"point": host, "riders": list(riders), "jit_root": str(jit_root),
                        "gpu_wait": gpu_wait, "boot_sec": None, "error": None}
        done: set[str] = set()
        if deadline - time.time() < 60:
            record["error"] = f"invocation deadline ({CONFIG['timeout_sec']}s) reached before this session"
        else:
            tb = time.time()
            with (host_dir / "vllm_serve.log").open("w") as slf:
                proc = subprocess.Popen(serve_cmd, env=senv, stdout=slf, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    _wait_ready(port, proc, min(float(CONFIG["boot_timeout_sec"]),
                                                max(1.0, deadline - time.time())))
                    record["boot_sec"] = round(time.time() - tb, 1)
                    print(f"[_candidate_runner] server for {host} ready after {record['boot_sec']:.0f}s; "
                          f"seed={seed} points={[host] + list(riders)}", flush=True)
                    entry = _bench_point(python, senv, model, port, host, host_wl, host_dir, serve_cmd,
                                         seed, deadline, proc, write_manifest=host != wl["name"])
                    entry["session"] = host
                    done.add(host)
                    if host == wl["name"]:
                        anchor = entry
                    else:
                        points_benched.append(entry)
                    for name in riders:
                        rider = _bench_point(python, senv, model, port, name, spec["points"][name],
                                             point_raw_dir(raw, name, spec["anchor"]), serve_cmd, seed,
                                             deadline, proc)
                        rider["session"] = host
                        done.add(name)
                        points_benched.append(rider)
                except SystemExit as exc:
                    record["error"] = str(exc)
                except Exception as exc:  # noqa: BLE001 -- the manifest must still be written
                    record["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    _stop_server(proc)
        if record["error"]:
            print(f"[_candidate_runner] session {host}: FAILED -- {record['error']}", flush=True)
            for name in [host] + list(riders):
                if name in done:
                    continue
                failed = {"name": name, "returncode": 1, "error": record["error"], "timed_out": False,
                          "elapsed_sec": 0.0, "session": host}
                if name == wl["name"]:
                    anchor = failed
                    error = record["error"]
                else:
                    points_benched.append(failed)
                    pdir = point_raw_dir(raw, name, spec["anchor"])
                    pdir.mkdir(parents=True, exist_ok=True)
                    (pdir / "vllm_bench_manifest.json").write_text(json.dumps({
                        "serve_command": " ".join(shlex.quote(c) for c in serve_cmd),
                        "returncode": 1, "error": record["error"], "point": name, "seed": seed,
                        "timed_out": False, "elapsed_sec": 0.0, "session": host,
                    }, indent=2))
        sessions.append(record)

    _session(wl["name"], wl, raw, guards)
    for name in scored_extra:
        _session(name, spec["points"][name], point_raw_dir(raw, name, spec["anchor"]), [])
    elapsed = time.time() - t0

    manifest.write_text(json.dumps({
        "command": " ".join(shlex.quote(c) for c in anchor_cmd),
        "serve_command": " ".join(shlex.quote(c) for c in serve_cmd),
        "returncode": anchor["returncode"] if not error else 1,
        "error": error or anchor.get("error"),
        "timed_out": bool(anchor.get("timed_out")) or any(p.get("timed_out") for p in points_benched),
        "timeout_sec": CONFIG["timeout_sec"],
        "benched_tree": resolved,
        "elapsed_sec": round(elapsed, 1),
        "boot_sec": sessions[0]["boot_sec"] if sessions else None,
        "jit_root": sessions[0]["jit_root"] if sessions else None,
        "gpu_wait": sessions[0]["gpu_wait"] if sessions else None,
        "sessions": sessions,
        "seed": seed,
        "port": port,
        "workloads_benched": ([wl["name"]] if not anchor.get("error") and not error else [])
                             + [pt["name"] for pt in points_benched if not pt["error"]],
        "points": [anchor] + points_benched,
        "workload": {
            "input_len": in_len, "output_len": out_len, "num_prompts": num_prompts,
            "max_concurrency": concurrency, "max_model_len": max_len,
        },
    }, indent=2))
    print(f"[_candidate_runner] repo={repo} rc={anchor['returncode']} elapsed={elapsed:.1f}s "
          f"in={in_len} out={out_len} n={num_prompts} c={concurrency} maxlen={max_len} "
          f"points_ok={len([p for p in points_benched if not p['error']])}/{len(points_benched)}",
          flush=True)
    print(f"[_candidate_runner] gpu after shutdown: {_wait_for_free_gpu(timeout_s=90.0)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
