#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Bench ONE engine tree: boot a server from it, drive it with aiperf, tear it down, and write the
metrics the verifier divides."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

CONFIG = {
    "metric_key": "output_throughput",
    "extra_server_args": [],
    "log_level": "warning",
    "server_ready_timeout_sec": int(os.environ.get("INFB_SERVER_READY_SEC", "1800")),
    "bench_timeout_sec": int(os.environ.get("INFB_BENCH_TIMEOUT_SEC", "1800")),
    "port": int(os.environ.get("INFB_BENCH_PORT", "31418")),
}


def _workspace() -> Path:
    w = os.environ.get("INFB_WORKSPACE")
    return Path(w) if w else Path("/app")


def _venv_python() -> str:
    for cand in (os.environ.get("INFB_VLLM_PYTHON"), "/opt/venv/bin/python3", sys.executable):
        if cand and Path(cand).exists():
            return cand
    return sys.executable


def _model(ws: Path) -> str:
    env = os.environ.get("INFB_MODEL_PATH")
    if env:
        return env
    link = ws / "model"
    if link.exists():
        return str(link.resolve())
    raise SystemExit("[_candidate_runner] no model: neither INFB_MODEL_PATH nor <workspace>/model")


def _primary_workload() -> dict:
    wl = json.loads((Path(__file__).resolve().parent / "workloads.json").read_text())
    entries = wl["workloads"]
    primary = wl.get("primary_workload")
    for spec in entries:
        if primary is None or spec.get("name") == primary:
            return dict(spec)
    raise SystemExit(f"[_candidate_runner] primary_workload {primary!r} not in workloads.json")


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


def _aiperf_argv(port: int, model: str, wl: dict, artifact_dir: Path) -> list[str]:
    """aiperf's scored line for one point: the anchor's flags with the point's own shape."""
    return [
        _aiperf_bin(), "profile",
        "--url", f"http://127.0.0.1:{port}",
        "--model", "scored-model",
        "--tokenizer", model,
        "--tokenizer-trust-remote-code",
        "--endpoint-type", "completions",
        "--streaming",
        "--ui", "simple",
        "--extra-inputs", f"min_tokens:{int(wl['output_len'])}",
        "--extra-inputs", f"max_tokens:{int(wl['output_len'])}",
        "--extra-inputs", "ignore_eos:true",
        "--concurrency", str(int(wl["concurrency"])),
        "--synthetic-input-tokens-mean", str(int(wl["input_len"])),
        "--synthetic-input-tokens-stddev", "0",
        "--output-tokens-mean", str(int(wl["output_len"])),
        "--output-tokens-stddev", "0",
        "--request-count", str(int(wl["num_prompts"])),
        "--warmup-request-count", "1",
        "--artifact-dir", str(artifact_dir),
    ]


def _with_latency_aliases(parsed: dict) -> dict:
    """aiperf's names under the ones the verifier's normaliser reads (`mean_ttft_ms`, `mean_itl_ms`)."""
    out = dict(parsed)
    if isinstance(out.get("time_to_first_token_ms"), (int, float)):
        out.setdefault("mean_ttft_ms", out["time_to_first_token_ms"])
    if isinstance(out.get("inter_token_latency_ms"), (int, float)):
        out.setdefault("mean_itl_ms", out["inter_token_latency_ms"])
    return out


def _aiperf_bin() -> str:
    """The scored harness binary, from the venv the image keeps it in."""
    for cand in (os.environ.get("INFB_AIPERF_BIN"), "/opt/aiperf-venv/bin/aiperf"):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit("[_candidate_runner] aiperf not found; the image must install it in /opt/aiperf-venv")


def _gpu_used_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60).stdout
        return sum(int(x) for x in out.split() if x.strip().isdigit())
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["throughput"])
    ap.add_argument("--repo", required=True, help="engine tree to overlay via PYTHONPATH")
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
    out_json = raw / "sglang_bench_throughput.json"
    manifest = raw / "sglang_bench_manifest.json"

    python = _venv_python()
    model = _model(ws)
    wl = _primary_workload()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _verifier_lib.bench_stats import load_matrix_spec, point_raw_dir  # noqa: E402
    spec = load_matrix_spec(Path(__file__).resolve().parent / "workloads.json")
    extra_points = _select_extra_points(args.workloads, spec)
    points_benched: list[dict] = []
    port = CONFIG["port"]

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["TOKENIZERS_PARALLELISM"] = "false"
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(key, None)

    server_cmd = [
        python, "-m", "sglang.launch_server",
        "--model-path", model,
        "--served-model-name", "scored-model",
        "--tp", str(int(wl.get("tp_size", 4))),
        "--host", "127.0.0.1", "--port", str(port),
        "--log-level", str(CONFIG["log_level"]),
        *CONFIG["extra_server_args"],
    ]
    aiperf_dir = raw / "aiperf"
    aiperf_cmd = [
        _aiperf_bin(), "profile",
        "--url", f"http://127.0.0.1:{port}",
        "--model", "scored-model",
        "--tokenizer", model,
        "--tokenizer-trust-remote-code",
        "--endpoint-type", "completions",
        "--streaming",
        "--ui", "simple",
        "--extra-inputs", f"min_tokens:{int(wl['output_len'])}",
        "--extra-inputs", f"max_tokens:{int(wl['output_len'])}",
        "--extra-inputs", "ignore_eos:true",
        "--concurrency", str(int(wl["concurrency"])),
        "--synthetic-input-tokens-mean", str(int(wl["input_len"])),
        "--synthetic-input-tokens-stddev", "0",
        "--output-tokens-mean", str(int(wl["output_len"])),
        "--output-tokens-stddev", "0",
        "--request-count", str(int(wl["num_prompts"])),
        "--warmup-request-count", "1",
        "--artifact-dir", str(aiperf_dir),
    ]
    crosscheck_json = raw / "crosscheck_stream_client.json"
    crosscheck_cmd = [
        python, str(Path(__file__).resolve().parent / "_stream_client.py"),
        "--base-url", f"http://127.0.0.1:{port}",
        "--tokenizer", model,
        "--input-len", str(int(wl["input_len"])),
        "--output-len", str(int(wl["output_len"])),
        "--num-prompts", str(int(wl["num_prompts"])),
        "--concurrency", str(int(wl["concurrency"])),
        "--seed", str(int(wl.get("seed", 1234))),
        "--out-json", str(crosscheck_json),
        "--floor-probe-out", str(raw / "min_new_tokens_floor.json"),
        "--floor-min-tokens", "32",
        "--floor-max-tokens", "256",
    ]

    engine_package = None
    try:
        probe = subprocess.run(
            [python, "-c", "import sglang, sys; sys.stdout.write(sglang.__file__)"],
            env=env, capture_output=True, text=True, timeout=180)
        engine_package = probe.stdout.strip() or None
    except Exception:
        pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _probe:
        _probe.settimeout(2.0)
        if _probe.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(
                f"port {port} is already serving before this arm started; refusing to bench against "
                f"a server this runner did not launch. A leftover public/run_benchmark.py server "
                f"would be benched as if it were the arm under test."
            )

    t0 = time.time()
    rc, ready, timed_out, ready_sec = 1, False, False, None
    srv_log = (raw / "sglang_server.log").open("w")
    server = subprocess.Popen(server_cmd, env=env, stdout=srv_log, stderr=subprocess.STDOUT,
                              start_new_session=True)
    try:
        import urllib.request
        deadline = time.time() + CONFIG["server_ready_timeout_sec"]
        while time.time() < deadline:
            if server.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health_generate", timeout=10):
                    ready = True
                    break
            except Exception:
                time.sleep(5)
        ready_sec = time.time() - t0
        if ready:
            with (raw / "sglang_bench_stderr.txt").open("w") as errf, \
                 (raw / "sglang_bench_stdout.txt").open("w") as outf:
                try:
                    rc = subprocess.run(aiperf_cmd, env=env, stdout=outf, stderr=errf,
                                        timeout=CONFIG["bench_timeout_sec"]).returncode
                except subprocess.TimeoutExpired:
                    rc, timed_out = 124, True
                try:
                    subprocess.run(crosscheck_cmd, env=env, stdout=outf, stderr=errf,
                                   timeout=CONFIG["bench_timeout_sec"])
                except Exception:
                    pass
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from _aiperf_metrics import read as _read_aiperf  # noqa: E402
            parsed = _read_aiperf(aiperf_dir)
            if parsed is None:
                rc = rc or 1
            else:
                out_json.write_text(json.dumps(_with_latency_aliases(parsed), indent=2, sort_keys=True) + "\n")
            for name in extra_points:
                point = spec["points"][name]
                pdir = point_raw_dir(raw, name, spec["anchor"])
                pdir.mkdir(parents=True, exist_ok=True)
                pcmd = _aiperf_argv(port, model, point, pdir / "aiperf")
                pt0, prc, perr = time.time(), 1, None
                with (pdir / "sglang_bench_stderr.txt").open("w") as perrf, \
                     (pdir / "sglang_bench_stdout.txt").open("w") as poutf:
                    try:
                        prc = subprocess.run(pcmd, env=env, stdout=poutf, stderr=perrf,
                                             timeout=CONFIG["bench_timeout_sec"]).returncode
                    except subprocess.TimeoutExpired:
                        prc, perr = 124, "aiperf timed out"
                pparsed = _read_aiperf(pdir / "aiperf") if prc == 0 else None
                if prc == 0 and pparsed is None:
                    prc, perr = 1, "aiperf exited 0 but left no usable export"
                if pparsed is not None and prc == 0:
                    (pdir / "sglang_bench_throughput.json").write_text(
                        json.dumps(_with_latency_aliases(pparsed), indent=2, sort_keys=True) + "\n")
                    print(f"[_candidate_runner] point {name}: {pparsed.get('output_throughput')} tok/s  "
                          f"ITL {pparsed.get('inter_token_latency_ms')} ms  "
                          f"TTFT {pparsed.get('time_to_first_token_ms')} ms", flush=True)
                else:
                    print(f"[_candidate_runner] point {name}: FAILED -- {perr or f'rc={prc}'}", flush=True)
                (pdir / "sglang_bench_manifest.json").write_text(json.dumps({
                    "scored_client_command": " ".join(shlex.quote(c) for c in pcmd),
                    "returncode": prc, "error": perr, "point": name, "workload": point,
                    "elapsed_sec": round(time.time() - pt0, 2), "metric_key": CONFIG["metric_key"],
                }, indent=2))
                points_benched.append({"name": name, "returncode": prc, "error": perr})
    finally:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            server.wait(timeout=90)
        except Exception:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except Exception:
                pass
        srv_log.close()
        for _ in range(60):
            if _gpu_used_mb() < 4000:
                break
            time.sleep(5)

    elapsed = time.time() - t0
    client_cost = crosscheck_tps = None
    try:
        cc = json.loads(crosscheck_json.read_text())
        client_cost = cc.get("client_overhead_us_per_token")
        crosscheck_tps = cc.get("output_tokens_per_second")
    except Exception:
        pass

    manifest.write_text(json.dumps({
        "server_command": " ".join(shlex.quote(c) for c in server_cmd),
        "scored_client_command": " ".join(shlex.quote(c) for c in aiperf_cmd),
        "crosscheck_client_command": " ".join(shlex.quote(c) for c in crosscheck_cmd),
        "returncode": rc,
        "server_ready": ready,
        "server_ready_sec": ready_sec,
        "timed_out": timed_out,
        "bench_timeout_sec": CONFIG["bench_timeout_sec"],
        "elapsed_sec": elapsed,
        "repo": str(repo),
        "metric_key": CONFIG["metric_key"],
        "engine_package": engine_package,
        "engine_kwargs": {
            "tp_size": int(wl.get("tp_size", 4)),
            "log_level": CONFIG["log_level"],
            "extra_server_args": list(CONFIG["extra_server_args"]),
        },
        "crosscheck_output_tokens_per_second": crosscheck_tps,
        "crosscheck_client_overhead_us_per_token": client_cost,
        "workload": wl,
        "workloads_benched": [wl["name"]] + [pt["name"] for pt in points_benched if pt["returncode"] == 0],
        "points": points_benched,
    }, indent=2))
    print(f"[_candidate_runner] repo={repo} ready={ready} rc={rc} elapsed={elapsed:.1f}s "
          f"crosscheck={crosscheck_tps} cost={client_cost}us/tok", flush=True)
    return 0 if (ready and rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
