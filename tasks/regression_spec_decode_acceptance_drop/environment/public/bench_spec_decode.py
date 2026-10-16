#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Speculative-decoding throughput harness for this task."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent

COUNTERS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)

_PRECONDITIONS = {
    "async_scheduling": re.compile(r"async_scheduling'?\s*[:=]\s*True|using async scheduling", re.I),
    "drafter_loaded": re.compile(r"Detected EAGLE model", re.I),
}


def scrape(port: int) -> dict[str, float]:
    """Sum each counter family over all of its label sets."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=20) as fh:
        body = fh.read().decode()
    out = {c: 0.0 for c in COUNTERS}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\deE.+naN]+)$", line)
        if not m:
            continue
        for c in COUNTERS:
            if m.group(1) in (c, c + "_total"):
                try:
                    out[c] += float(m.group(3))
                except ValueError:
                    pass
    return out


def gpu_total_gib() -> float:
    """Capacity of the smallest visible card, so --min-free-gib can be derived rather than guessed."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        rows = [l for l in q.stdout.strip().splitlines() if l.strip()]
        if visible:
            want = {int(x) for x in visible.split(",") if x.strip().isdigit()}
            rows = [r for i, r in enumerate(rows) if i in want] or rows
        return min(int(r) for r in rows) / 1024.0
    except Exception:  # noqa: BLE001 -- no nvidia-smi means no wait
        return 0.0


def wait_healthy(port: int, proc: subprocess.Popen, deadline_s: float) -> float:
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup, rc={proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as fh:
                if fh.status == 200:
                    return time.time() - t0
        except Exception:  # noqa: BLE001 -- the server is simply not up yet
            time.sleep(5)
    raise RuntimeError(f"server did not become healthy within {deadline_s}s")


def warm_up(port: int, model: str, concurrency: int, *, max_tokens: int = 32,
            timeout_s: float = 300.0) -> dict[str, object]:
    """Two windows of `concurrency` distinct requests, answered in parallel and thrown away."""
    import concurrent.futures
    base = ("The committee met on a grey morning to review the harbour accounts, and the clerk "
            "read the ledger aloud while the members followed the columns with their pens. ")
    n = max(1, 2 * concurrency)
    prompts = [f"Note {i}. " + base * 6 for i in range(n)]

    def one(prompt: str) -> bool:
        body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                           "temperature": 0.0, "ignore_eos": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as fh:
                fh.read()
            return True
        except Exception:  # noqa: BLE001
            return False

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        oks = list(ex.map(one, prompts))
    return {"requests": n, "concurrency": concurrency, "max_tokens": max_tokens,
            "completed": sum(oks), "elapsed_s": round(time.time() - t0, 1)}


def resolve(ws: pathlib.Path, name: str, env_override: str) -> str:
    """`<workspace>/model` and `<workspace>/draft`, both planted by prepare_target_env.sh."""
    override = os.environ.get(env_override, "")
    if override and pathlib.Path(override).is_dir():
        return override
    p = ws / name
    if not p.exists():
        raise SystemExit(
            f"no {name} at {p}. prepare_target_env.sh plants both `model` and `draft`; the drafter "
            f"is staged from [metadata.inference_bench.speculative_config] model, NOT from "
            f"auxiliary_hf_model_ids."
        )
    return str(p.resolve())


def wait_for_gpus_free(min_free_gib: float = 60.0, deadline_s: float = 300.0) -> dict[str, object]:
    """Block until the GPUs this process can see are actually free again."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    t0 = time.time()
    last: list[float] = []
    while time.time() - t0 < deadline_s:
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30,
            )
            rows = [l for l in q.stdout.strip().splitlines() if l.strip()]
            if visible:
                want = {int(x) for x in visible.split(",") if x.strip().isdigit()}
                rows = [r for i, r in enumerate(rows) if i in want] or rows
            last = [(int(tot) - int(used)) / 1024.0
                    for tot, used in (r.split(",") for r in rows)]
        except Exception:  # noqa: BLE001 -- no nvidia-smi is not a reason to hang
            return {"waited_s": round(time.time() - t0, 1), "free_gib": None,
                    "note": "nvidia-smi unavailable; not waiting"}
        if last and min(last) >= min_free_gib:
            return {"waited_s": round(time.time() - t0, 1),
                    "free_gib": [round(f, 1) for f in last]}
        time.sleep(10)
    return {"waited_s": round(time.time() - t0, 1),
            "free_gib": [round(f, 1) for f in last],
            "note": f"still below {min_free_gib} GiB after {deadline_s}s; proceeding anyway so the "
                    f"failure is the engine's own message rather than a silent hang"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--input-len", type=int, default=0, help="ignored: the prompts are a fixed set")
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--draft-tokens", type=int, default=3)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--kv-cache-dtype", default="fp8")
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--cudagraph-capture", type=int, default=2048)
    ap.add_argument("--stream-interval", type=int, default=20)
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--dataset-name", default="random",
                    help="random | custom | hf. `random` emits random token ids and needs a long "
                         "--output-len to stay off the acceptance floor; `custom` reads the shipped "
                         "natural-text prompts.jsonl.")
    ap.add_argument("--dataset-path", default="")
    ap.add_argument("--no-speculation", action="store_true",
                    help="serve WITHOUT --speculative-config. Speculative decoding's defining "
                         "guarantee is that it returns what the target model would have returned "
                         "anyway, so this arm is the reference the speculative one must match.")
    ap.add_argument("--natural-eos", action="store_true",
                    help="let requests stop at EOS instead of running to --output-len. Changes the "
                         "regime: forced continuation past the natural stopping point is much "
                         "harder to draft for, so acceptance differs by a lot between the two.")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--boot-timeout", type=float, default=2400)
    ap.add_argument("--bench-timeout", type=float, default=5400)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--min-free-gib", type=float, default=0.0,
                    help="GPU memory that must be free before this round boots AND after it tears "
                         "down. 0 (the default) DERIVES it from --gpu-memory-utilization and the "
                         "card's own capacity, which is the only value that cannot be wrong.")
    a = ap.parse_args()

    if a.min_free_gib <= 0:
        a.min_free_gib = round(gpu_total_gib() * a.gpu_memory_utilization + 1.0, 1)

    ws = pathlib.Path(a.workspace).resolve()
    model = resolve(ws, "model", "INFB_MODEL_PATH")
    draft = resolve(ws, "draft", "INFB_DRAFT_MODEL_PATH")
    out_json = pathlib.Path(a.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    output_len = a.output_len
    num_prompts = a.num_prompts
    max_seqs = a.max_num_seqs

    spec = json.dumps({"method": "eagle3", "model": draft,
                       "num_speculative_tokens": a.draft_tokens}, separators=(",", ":"))
    serve = [a.python, "-m", "vllm.entrypoints.cli.main", "serve", model,
             "--port", str(a.port),
             "--tensor-parallel-size", str(a.tensor_parallel_size),
             "--max-model-len", str(a.max_model_len),
             "--max-num-seqs", str(max_seqs),
             "--gpu-memory-utilization", str(a.gpu_memory_utilization),
             "--async-scheduling",
             "--no-enable-prefix-caching",
             "--disable-custom-all-reduce",
             "--kv-cache-dtype", a.kv_cache_dtype,
             "--max-num-batched-tokens", str(a.max_num_batched_tokens),
             "--stream-interval", str(a.stream_interval),
             "--compilation-config",
             json.dumps({"max_cudagraph_capture_size": a.cudagraph_capture},
                        separators=(",", ":")),
]
    if not a.no_speculation:
        serve += ["--speculative-config", spec]
    if a.enforce_eager:
        serve.append("--enforce-eager")

    server_log = out_json.with_suffix(".server.log")
    rec: dict[str, object] = {
        "model": model, "draft": draft,
        "output_len": output_len, "num_prompts": num_prompts,
        "max_num_seqs": max_seqs, "max_model_len": a.max_model_len,
        "draft_tokens": a.draft_tokens, "enforce_eager": bool(a.enforce_eager),
        "kv_cache_dtype": a.kv_cache_dtype, "tensor_parallel_size": a.tensor_parallel_size,
        "max_num_batched_tokens": a.max_num_batched_tokens,
        "cudagraph_capture": a.cudagraph_capture, "stream_interval": a.stream_interval,
        "command": " ".join(serve),
    }

    rec["gpu_wait_before"] = wait_for_gpus_free(a.min_free_gib)

    with server_log.open("w") as lf:
        proc = subprocess.Popen(serve, env=dict(os.environ), stdout=lf,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rec["boot_s"] = round(wait_healthy(a.port, proc, a.boot_timeout), 1)
            rec["warmup"] = None if a.probe else warm_up(a.port, model, max_seqs)
            before = scrape(a.port)
            bench_json = out_json.with_suffix(".bench.json")
            bench = [a.python, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
                     "--model", model, "--host", "127.0.0.1", "--port", str(a.port)]
            if a.dataset_name == "random":
                bench += ["--dataset-name", "random",
                          "--random-input-len", str(a.input_len),
                          "--random-range-ratio", "0.0",
                          "--output-len", str(output_len)]
            elif a.dataset_name == "custom":
                bench += ["--dataset-name", "custom",
                          "--dataset-path", a.dataset_path or a.prompts,
                          "--custom-output-len", str(output_len)]
            else:
                bench += ["--dataset-name", a.dataset_name,
                          "--dataset-path", a.dataset_path,
                          "--output-len", str(output_len)]
            if not a.natural_eos:
                bench.append("--ignore-eos")
            bench += ["--save-detailed"]
            bench += ["--num-prompts", str(num_prompts),
                      "--max-concurrency", str(max_seqs),
                      "--seed", "12345",
                      "--save-result", "--result-filename", str(bench_json)]
            b = subprocess.run(bench, env=dict(os.environ), capture_output=True, text=True,
                               timeout=a.bench_timeout)
            rec["returncode"] = b.returncode
            out_json.with_suffix(".client.log").write_text(
                b.stdout + "\n===STDERR===\n" + b.stderr)
            after = scrape(a.port)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=120)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
            rec["gpu_wait_after"] = wait_for_gpus_free(a.min_free_gib)

    log_text = server_log.read_text(errors="replace")
    exercised = {name: bool(rx.search(log_text)) for name, rx in _PRECONDITIONS.items()}
    d_drafts = after[COUNTERS[0]] - before[COUNTERS[0]]
    d_draft_tok = after[COUNTERS[1]] - before[COUNTERS[1]]
    d_acc = after[COUNTERS[2]] - before[COUNTERS[2]]
    observed_k = (d_draft_tok / d_drafts) if d_drafts else 0.0
    rec["observed_draft_len"] = observed_k
    exercised["k_as_configured"] = observed_k >= (a.draft_tokens - 0.5)
    exercised["rejections_exist"] = bool(d_draft_tok) and d_acc < d_draft_tok
    _al = (1 + d_acc / d_drafts) if d_drafts else 0.0
    rec["acceptance_length_observed"] = _al
    if a.no_speculation:
        exercised = {"async_scheduling": exercised.get("async_scheduling", False)}

    try:
        _bench_doc = json.loads(bench_json.read_text())
        _texts = _bench_doc.get("generated_texts") or []
        rec["output_digests"] = [
            hashlib.sha256(t.encode("utf-8", "replace")).hexdigest()[:16] for t in _texts
        ]
        rec["output_digest"] = hashlib.sha256(
            "".join(rec["output_digests"]).encode()
        ).hexdigest()[:16]
    except (OSError, ValueError):
        rec["output_digests"] = []
        rec["output_digest"] = None
    rec["num_drafts"] = d_drafts
    rec["num_draft_tokens"] = d_draft_tok
    rec["num_accepted_tokens"] = d_acc
    rec["mean_acceptance_length"] = (1 + d_acc / d_drafts) if d_drafts else None
    rec["draft_acceptance_rate"] = (d_acc / d_draft_tok) if d_draft_tok else None
    rec["comparable"] = True
    rec["exercised"] = exercised
    rec["all_preconditions_held"] = all(exercised.values())

    if bench_json.is_file():
        try:
            j = json.loads(bench_json.read_text())
            for key in ("output_throughput", "total_token_throughput", "request_throughput",
                        "mean_ttft_ms", "median_ttft_ms", "mean_tpot_ms", "mean_itl_ms",
                        "duration", "completed"):
                if key in j:
                    rec[key] = j[key]
        except Exception as e:  # noqa: BLE001
            rec["bench_parse_error"] = str(e)
    rec.setdefault("tokens_per_second", rec.get("total_token_throughput"))

    out_json.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rec, indent=2, sort_keys=True))

    if not rec["all_preconditions_held"]:
        missing = sorted(k for k, v in exercised.items() if not v)
        print(f"FATAL: preconditions did not hold: {missing}. The measurement is void, not slow: "
              f"both arms would bench the same non-speculative path and an unmodified tree would "
              f"score a pass.", file=sys.stderr)
        return 2
    return 0 if rec.get("returncode") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
