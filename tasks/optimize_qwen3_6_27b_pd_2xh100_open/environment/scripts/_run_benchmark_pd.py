#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Agent-facing PD benchmark: bring up two pools, run the multi-turn bench, print the SCORED metric."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))



def _resolve_workspace() -> Path:
    return Path(os.environ.get("INFB_WORKSPACE", "/app"))


def _resolve_venv() -> Path:
    """The venv ROOT, not its python: pd_topology.start() wants `venv / "bin" / "vllm"`."""
    from _target_env import resolve_venv_python  # noqa: PLC0415
    return Path(resolve_venv_python()).absolute().parent.parent


def _resolve_model() -> str:
    if env := os.environ.get("INFB_MODEL_PATH"):
        if Path(env).is_dir():
            return env
    workspace = _resolve_workspace()
    if (workspace / "model").exists():
        return str((workspace / "model").resolve())
    raise SystemExit("FATAL: cannot resolve the model path (no INFB_MODEL_PATH, no /app/model)")


def _load_published() -> tuple[dict, dict]:
    """Return (workload, pd_table) from the task's own published shape."""
    path = _resolve_workspace() / "public" / "workloads.json"
    if not path.exists():
        raise SystemExit(f"FATAL: workloads file missing at {path}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    workloads = blob["workloads"]
    pd = blob.get("pd")
    if pd is None:
        raise SystemExit(
            f"FATAL: {path} has no `pd` block, so this task's pool topology is unpublished and this "
            f"script cannot boot anything. A PD task must publish it in workloads.json."
        )
    scored = [w for w in workloads if w.get("name") == "pd_multiturn"]
    if not scored:
        raise SystemExit(
            "FATAL: no `pd_multiturn` workload in workloads.json. This script measures the PD workload; "
            "for an aggregated one use _run_benchmark_serve.py."
        )
    return scored[0], dict(pd)


def _pd_shared(name: str):
    """Import a PD module from beside this file."""
    import importlib.util  # noqa: PLC0415
    path = _HERE / (name + ".py")
    if not path.is_file():
        raise SystemExit(
            f"FATAL: {name}.py is not beside this script ({_HERE}). The PD rig ships as a set; a missing "
            f"member means this task's scripts/ directory is incomplete, not that you are invoking it wrong."
        )
    spec = importlib.util.spec_from_file_location("_pd_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



def _resident_tokens(pd: dict) -> int:
    """Tokens the workload holds resident at its declared concurrency, at the END of the conversation."""
    user_turns = max(1, int(pd.get("conversation_turns", 8)) // 2)
    per_conversation = (int(pd.get("first_turn_prefix_tokens", 8000))
                        + user_turns * (int(pd["new_tokens_per_turn"]) + int(pd["reply_tokens"])))
    return per_conversation * int(pd.get("conversations", 16))


def _report_kv_headroom(pd: dict, log_dir: Path, *, concurrency: int) -> None:
    """Print the resident set against what the consumer pool actually reported as its KV capacity."""
    resident = _resident_tokens(pd)
    per_conv = resident // max(1, int(pd.get("conversations", 16)))
    capacity = None
    for log in sorted(log_dir.glob("*.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if "KV cache size" in line and "token" in line:
                digits = "".join(c for c in line.split("KV cache size")[1] if c.isdigit())
                if digits:
                    value = int(digits)
                    capacity = value if capacity is None else max(capacity, value)
    print(f"[pd] resident set at --concurrency {concurrency}: {concurrency} conversations x {per_conv:,} "
          f"tokens = {per_conv * concurrency:,} tokens", flush=True)
    if capacity is None:
        print("[pd] consumer pool KV capacity: not found in the pool logs (the engine prints it as "
              "`GPU KV cache size: N tokens`); cannot report oversubscription", flush=True)
        return
    ratio = (per_conv * concurrency) / float(capacity)
    verdict = ("fits" if ratio <= 1.0 else
               f"OVERSUBSCRIBED {ratio:.2f}x -- past 1.0 the pool preempts, and what you then measure is "
               f"preemption rather than disaggregation")
    print(f"[pd] consumer pool KV = {capacity:,} tokens; resident/capacity = {ratio:.2f}x ({verdict})",
          flush=True)



def _sglang_venv_site_packages(venv: Path) -> list[str]:
    """The artifact venv's site-packages, which SGLang needs on PYTHONPATH explicitly."""
    out: list[str] = []
    for root in (venv, Path("/opt/venv")):
        for cand in sorted(root.glob("lib/python*/site-packages")):
            if (cand / "IPython").is_dir() or (cand / "torch").is_dir():
                out.append(str(cand))
    return out


def _pool_env(repo: Path, venv: Path, *, engine: str, side: str) -> dict:
    """The pool environment. THE OVERLAY IS THE WHOLE POINT -- without it this bench cannot measure anything."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    pp_parts = [str(repo)]
    if engine == "sglang":
        pp_parts += _sglang_venv_site_packages(venv)
    overlay = _resolve_workspace() / ".flashinfer_overlay"
    if overlay.is_dir():
        if engine == "sglang":
            pp_parts.insert(0, str(overlay))
        else:
            pp_parts.append(str(overlay))
    if engine == "sglang" and side.lower().startswith("cand"):
        sglk = _resolve_workspace() / ".sgl_kernel_overlay"
        if sglk.is_dir():
            pp_parts.insert(0, str(sglk))
    env["PYTHONPATH"] = ":".join(pp_parts)
    env["PATH"] = f"{venv}/bin:" + env.get(
        "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def _engine_argv(*, model_path: str, venv: Path, pd: dict, width: int, served: str) -> list[str]:
    """The argv both pools share, built from the argv the VERIFIER uses."""
    user_turns = max(1, int(pd.get("conversation_turns", 8)) // 2)
    need = (int(pd.get("first_turn_prefix_tokens", 8000))
            + user_turns * (int(pd["new_tokens_per_turn"]) + int(pd["reply_tokens"])))
    ceiling = int(pd.get("context_window", need))
    max_len = min(ceiling, int(need * 1.25))
    slots = {
        "{python}": str(venv / "bin" / "python"),
        "{vllm}": str(venv / "bin" / "vllm"),
        "{model}": model_path,
        "{width}": str(width),
        "{served}": served,
        "{max_len}": str(max_len),
        "{batched_tokens}": str(min(8192, max_len)),
        "{moe_runner_backend}": str(pd.get("moe_runner_backend") or "auto"),
        "{chunked_prefill_size}": str(int(pd["chunked_prefill_size"])) if "chunked_prefill_size" in pd else None,
    }
    template = pd.get("engine_argv_template")
    if not template:
        raise SystemExit(
            "FATAL: this task publishes no `engine_argv_template`, so the pools cannot be started with the "
            "flags the scored run uses. The template is published with the task; guessing an argv would "
            "boot a differently-configured deployment and report a plausible number."
        )
    argv = []
    for item in template:
        if item.startswith("{") and item.endswith("}"):
            if slots.get(item) is None:
                raise SystemExit(
                    f"FATAL: the published engine argv template has the slot {item}, which this script "
                    f"cannot fill. The generator and this substitution must be edited together."
                )
            argv.append(slots[item])
        else:
            argv.append(item)
    block = pd.get("kv_block_size")
    if block is not None:
        argv += ["--block-size", str(int(block))]
    return argv


def _measure_side(*, side: str, repo: Path, pd: dict, workload: dict, model_path: str,
                  num_runs: int, warmup_runs: int, concurrency: list[int], boot_timeout: float,
                  cuda_devices: str, workdir: Path) -> dict:
    """Boot the pools ONCE, bench at every concurrency point, return {c: {"scores", "extracts"}}."""
    topo = _pd_shared("pd_topology")
    metrics = _pd_shared("pd_turn_metrics")
    venv = _resolve_venv()

    topology = topo.Topology.from_metadata(pd, cuda_devices, aggregated=False)
    width = topo.tensor_parallel_size(topology)
    served = "model"
    print(f"[pd:{side}] topology {topology.label()} on devices {cuda_devices} (TP={width} per pool)",
          flush=True)

    log_dir = workdir / ("pd_" + side)
    log_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.suppress(Exception):
        from _paired_helper import kill_stray_gpu_processes  # noqa: PLC0415
        kill_stray_gpu_processes(cuda_device=cuda_devices, grace_sec=2.0)

    engine_argv = _engine_argv(model_path=model_path, venv=venv, pd=pd, width=width, served=served)
    by_c: dict[int, dict] = {c: {"scores": [], "extracts": []} for c in concurrency}
    router = None
    try:
        router = topo.start(topology, engine_argv=engine_argv, venv=venv,
                            router_module=_HERE / "pd_router.py", log_dir=log_dir,
                            served_model_name=served,
                            env=_pool_env(repo, venv,
                                            engine=str(pd.get("engine") or "vllm"), side=side))
        topo.wait_ready(topology, served, timeout_s=float(boot_timeout))
        for c in concurrency:
            _report_kv_headroom(pd, log_dir, concurrency=c)
        for c in concurrency:
            for index in range(warmup_runs + num_runs):
                label = ("warmup" if index < warmup_runs else f"run{index - warmup_runs}") + f"_c{c}"
                stats = log_dir / f"stats_{label}.json"
                rc = _run_bench(pd=pd, model_path=model_path, venv=venv, repo=repo,
                                router_port=topology.router_port, stats_out=stats,
                                log_path=log_dir / f"bench_{label}.log", served_model_name=served,
                                concurrency=c)
                topo.assert_pd_happened(topology)
                if index < warmup_runs:
                    continue
                if rc != 0 or not stats.is_file():
                    print(f"[pd:{side}] {label} bench rc={rc}; no sample recorded", flush=True)
                    continue
                extract = metrics.extract(stats, **metrics.expected_counts(pd))
                by_c[c]["scores"].append(float(extract["pd_latency_itl_geomean_ms"]))
                by_c[c]["extracts"].append(extract)
                print(f"[pd:{side}] {label} pd_latency_itl_geomean_ms "
                      f"{extract['pd_latency_itl_geomean_ms']:.1f} ms "
                      f"(e2e p95 {extract['continuation_p95_latency_ms']:.0f} ms, "
                      f"pooled p99 ITL {extract['pooled_p99_itl_ms']:.2f} ms, "
                      f"continuation TTFT p95 {extract.get('continuation_p95_ttft_ms', float('nan')):.0f} ms, "
                      f"n={extract['continuation_count']})", flush=True)
                if extract.get("pd_itl_p99_on_cliff_edge"):
                    print(f"[pd:{side}] {label} WARNING: pooled p99 ITL is standing on a stall-cluster edge "
                          f"(gap {extract.get('pd_itl_cliff_gap')}x, mass "
                          f"{extract.get('pd_itl_cliff_mass_pct')}%) -- it may move discontinuously", flush=True)
        topo.assert_pd_happened(topology)
    finally:
        topo.stop(topology, router)
    return by_c


def _run_bench(*, pd: dict, model_path: str, venv: Path, repo: Path,
               router_port: int, stats_out: Path, log_path: Path, served_model_name: str,
               concurrency: int) -> int:
    """Run the patched multi-turn client against the router. Mirrors the verifier's own bench call."""
    client = repo / "benchmarks" / "multi_turn" / "benchmark_serving_multi_turn.py"
    if not client.is_file():
        vendored = _HERE / "multi_turn" / "benchmark_serving_multi_turn.py"
        if not vendored.is_file():
            raise SystemExit(
                f"FATAL: no multi-turn bench client at {client} and none vendored at {vendored}."
            )
        client = vendored
    from pd_client_patch import patch_multi_turn_client  # noqa: PLC0415
    print("[pd] multi-turn client edits: " + patch_multi_turn_client(client), flush=True)

    spec_mod = _pd_shared("pd_bench_spec")
    spec = spec_mod.write_spec(pd, stats_out.parent)
    argv = spec_mod.build_bench_argv(client=client, venv=venv, pd=pd, model_path=model_path,
                                     router_port=router_port, stats_out=stats_out,
                                     served_model_name=served_model_name, concurrency=concurrency,
                                     spec=spec)
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(argv, cwd=str(client.parent), env=spec_mod.bench_env(),
                              stdout=fh, stderr=subprocess.STDOUT, check=False)
    return proc.returncode



def _report(candidate: dict, pristine: dict | None, *, scored_points: list[int], workload: dict) -> int:
    """Print the per-point detail and then THE score: the geomean across the declared points, per round."""
    from _paired_helper import trimmed_mean  # noqa: PLC0415
    metrics = _pd_shared("pd_turn_metrics")

    points = sorted(candidate)
    if not any(candidate[c]["scores"] for c in points):
        print("\n*** NO MEASUREMENT -- the candidate side recorded 0 run(s) at every concurrency point. The "
              "[pd:candidate] lines above name where it stopped. ***", flush=True)
        return 1

    print("\n" + "=" * 78)
    header = f"{'c':>5}  {'candidate':>12}"
    if pristine is not None:
        header += f"  {'pristine':>12}  {'ratio':>9}"
    print(header + "        (ms, LOWER IS BETTER)")
    for c in points:
        cs = candidate[c]["scores"]
        mark = "  scored" if c in scored_points else "  (probe only, not scored)"
        if not cs:
            print(f"{c:>5}  {'no run':>12}{mark}")
            continue
        cand = trimmed_mean(cs)
        line = f"{c:>5}  {cand:>12.2f}"
        if pristine is not None:
            ps = pristine.get(c, {}).get("scores") or []
            line += f"  {trimmed_mean(ps):>12.2f}  {trimmed_mean(ps) / cand:>8.4f}x" if ps else \
                f"  {'no run':>12}  {'-':>9}"
        print(line + mark)

    missing = [c for c in scored_points if not candidate.get(c, {}).get("scores")]
    if missing:
        print(f"\n*** NO SCORE -- the declared point(s) {missing} produced no measurement. The score is a "
              f"geomean over ALL {len(scored_points)} declared points; over a subset it would be a different "
              f"axis from the declared one, so no number is printed. ***")
        print("=" * 78, flush=True)
        return 1

    profiles = metrics.profile_points(workload)
    def _agg(side: dict) -> list[float]:
        return metrics.aggregate_profiles(
            {p["name"]: side[int(p["concurrency"])]["scores"] for p in profiles}, profiles)

    cand_rounds = _agg(candidate)
    cand = trimmed_mean(cand_rounds)
    print(f"\nSCORED  geomean over {[p['name'] for p in profiles]}, per round: "
          f"{[round(v, 1) for v in cand_rounds]}")
    print(f"pd_latency_itl_geomean_ms  candidate  {cand:9.2f} ms   (n={len(cand_rounds)} round(s))")
    if pristine is None:
        print("mode SOLO -- one arm only. No speedup is printed: there is no live pristine arm to divide by, "
              "and a stored baseline is not available to this script by design. Re-run with --paired.")
        print("=" * 78, flush=True)
        return 0
    if any(not pristine.get(c, {}).get("scores") for c in scored_points):
        print("mode PAIRED -- but the pristine arm is missing a declared point, so no ratio.")
        print("=" * 78, flush=True)
        return 1
    pris = trimmed_mean(_agg(pristine))
    speedup = pris / cand if cand else float("nan")
    print(f"pd_latency_itl_geomean_ms  pristine   {pris:9.2f} ms")
    print(f"speedup (pristine/candidate)          {speedup:9.4f}x  ->  reward "
          f"{max(0.0, speedup - 1.0):.4f}")
    print("=" * 78, flush=True)
    return 0


def _sweep_points(spec: str | None, *, scored: int) -> list[int]:
    """Parse `--concurrency-sweep`, always including the scored point."""
    points = {int(scored)}
    if spec:
        for chunk in spec.replace(" ", "").split(","):
            if not chunk:
                continue
            value = int(chunk)
            if value < 1:
                raise SystemExit(f"FATAL: --concurrency-sweep got {value}; concurrency must be >= 1")
            points.add(value)
    return sorted(points)


JIT_STACK_FLOOR_BYTES = 64 * 1024 * 1024
JIT_HOST_COMPILERS = ("clang++", "clang++-18", "clang++-17")


def _raise_stack_limit_for_jit(pd: dict) -> str:
    """Lift RLIMIT_STACK so the engine's trial-time JIT can compile its heaviest translation unit."""
    import os as _os
    import resource
    import shutil as _shutil
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    inf = resource.RLIM_INFINITY
    outcome = None
    for want in ((inf, inf), (hard, hard)):
        if (soft, hard) == want:
            outcome = "already " + str(soft) + "/" + str(hard)
            break
        try:
            resource.setrlimit(resource.RLIMIT_STACK, want)
            outcome = str(soft) + "/" + str(hard) + " -> " + str(want[0]) + "/" + str(want[1])
            break
        except (ValueError, OSError):
            continue
    if outcome is None:
        outcome = "unchanged " + str(soft) + "/" + str(hard)
    now = resource.getrlimit(resource.RLIMIT_STACK)[0]
    if now != inf and now < JIT_STACK_FLOOR_BYTES:
        try:
            declared = str((pd or {}).get("jit_host_compiler") or "").strip()
        except Exception:
            declared = ""
        chosen = None
        if declared:
            for _cand in (declared, *(c for c in JIT_HOST_COMPILERS if c != declared)):
                chosen = _shutil.which(_cand)
                if chosen:
                    break
        if declared and chosen is None:
            outcome += ("; the task declares jit_host_compiler=" + repr(declared) + " and it is NOT on PATH,"
                        " so nvcc keeps its default and a deep-template unit may still exhaust the stack")
        elif chosen is not None:
            existing = _os.environ.get("NVCC_APPEND_FLAGS", "").strip()
            if "-ccbin" not in existing:
                _os.environ["NVCC_APPEND_FLAGS"] = (existing + " -ccbin " + chosen).strip()
            outcome += ("; stack below " + str(JIT_STACK_FLOOR_BYTES) + " and the task declares "
                        "jit_host_compiler=" + repr(declared) + ", so NVCC_APPEND_FLAGS='"
                        + _os.environ["NVCC_APPEND_FLAGS"] + "'. A task may only declare this when "
                        "DeepGEMM's JIT is off its critical path, because -ccbin clang breaks that JIT")
        else:
            outcome += ("; stack below " + str(JIT_STACK_FLOOR_BYTES) + " and left there -- the clang "
                        "-ccbin route is NOT taken by default (nvcc's frontend fails on glibc's _FloatN "
                        "in clang mode), so a deep-template JIT unit could still exhaust it")
    return outcome


def main() -> int:
    workload, pd = _load_published()
    p = argparse.ArgumentParser(
        description="Measure this PD task's scored metric. Defaults are the task's published topology, so "
                    "a bare run boots it and prints the number.")
    p.add_argument("--reply-tokens", type=int, default=int(pd["reply_tokens"]),
                   help="tokens each assistant turn emits")
    p.add_argument("--concurrency", type=int, default=int(workload.get("max_concurrency", 16)),
                   help="conversations in flight. The scored run uses the published value.")
    p.add_argument("--concurrency-sweep", default=None, metavar="C1,C2,...",
                   help="bench at several concurrency points inside ONE pool boot and print the curve. "
                        "Total work per point is IDENTICAL (conversations x turns/2 requests, from the spec), "
                        "so the points are comparable; only the published point is scored. PD's benefit is "
                        "load-dependent by construction -- there is no decode interference to remove at low "
                        "load -- so the shape of this curve is the thing a single point cannot show.")
    p.add_argument("--num-runs", type=int, default=1)
    p.add_argument("--num-warmup", type=int, default=0)
    p.add_argument("--cuda-devices", default=None,
                   help="devices for BOTH pools (default: CUDA_VISIBLE_DEVICES, else all of them)")
    p.add_argument("--boot-timeout", type=float, default=float(os.environ.get("INFB_SERVE_READY_TIMEOUT_SEC", "1800")))
    p.add_argument("--workdir", default="/logs/agent/pd_bench")
    p.add_argument("--paired", dest="paired", action="store_true", default=True,
                   help="measure a LIVE pristine arm too and print the ratio (default)")
    p.add_argument("--solo", dest="paired", action="store_false",
                   help="candidate only. Prints one number and no ratio -- there is no stored denominator "
                        "for a PD task and this script will not invent one.")
    args = p.parse_args()

    pd["reply_tokens"] = args.reply_tokens
    topo = _pd_shared("pd_topology")
    pd = topo.apply_workload_overrides(pd)
    print("[pd] JIT stack/host-compiler: " + _raise_stack_limit_for_jit(pd), flush=True)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    repo = _resolve_workspace()
    if (repo / "repo" / ".git").exists():
        repo = repo / "repo"
    model_path = _resolve_model()
    devices = args.cuda_devices or os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(
        str(i) for i in range(int(pd["prefill_gpus"]) + int(pd["decode_gpus"])))

    declared = [int(x["concurrency"]) for x in _pd_shared("pd_turn_metrics").profile_points(workload)]
    if args.concurrency_sweep or args.concurrency != int(workload.get("max_concurrency", 16)):
        points = _sweep_points(args.concurrency_sweep, scored=args.concurrency)
    else:
        points = sorted(set(declared))

    print(f"[pd] {pd['prefill_gpus']}P:{pd['decode_gpus']}D  connector={pd.get('connector')}  "
          f"reply_tokens={pd['reply_tokens']}  concurrency={','.join(str(c) for c in points)} "
          f"(scored: {args.concurrency})  "
          f"mode={'PAIRED' if args.paired else 'SOLO'}  runs={args.num_runs}"
          f"(+{args.num_warmup} warmup)", flush=True)

    started = time.time()
    candidate = _measure_side(side="candidate", repo=repo, pd=pd, workload=workload,
                              model_path=model_path, num_runs=args.num_runs,
                              warmup_runs=args.num_warmup, concurrency=points,
                              boot_timeout=args.boot_timeout, cuda_devices=devices, workdir=workdir)
    pristine = None
    if args.paired:
        from _paired_helper import disposable_pristine_worktree  # noqa: PLC0415
        with disposable_pristine_worktree(repo, workdir=workdir) as pris_repo:
            pristine = _measure_side(side="pristine", repo=pris_repo, pd=pd, workload=workload,
                                     model_path=model_path, num_runs=args.num_runs,
                                     warmup_runs=args.num_warmup, concurrency=points,
                                     boot_timeout=args.boot_timeout, cuda_devices=devices,
                                     workdir=workdir)
    rc = _report(candidate, pristine, scored_points=declared, workload=workload)
    print(f"[pd] total wall time {time.time() - started:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
