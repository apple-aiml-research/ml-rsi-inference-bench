#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The served-recipe checks for DeepSeek-V4-Flash on SGLang, read from the running server rather than from the tree's
source; the checker the grader runs, shared byte for byte with it."""
from __future__ import annotations

import collections
import gzip
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import requests

TRTLLM_MXFP4 = {
    "bmm_*MxE2m1*": re.compile(r"bmm_\w*MxE2m1"),
    "moe::dev::routing": re.compile(r"moe::dev::routing"),
    "moe::dev::finalize": re.compile(r"moe::dev::finalize"),
}
TRITON_FUSED_MOE = re.compile(r"fused_moe_kernel")
BACKEND_KEYS = ("attention_backend", "decode_attention_backend", "prefill_attention_backend")
PROMPT = "Q: Explain gravity briefly.\nA:"
_LIMITS = json.loads((Path(__file__).resolve().parent.parent / "limits.json").read_text())["operations"]
ACTION_S = int(_LIMITS["action"]["limit_s"])
REQUEST_S = int(_LIMITS["http_request"]["limit_s"])


def base_name(kernel: str) -> str:
    """A kernel's name without template arguments, parameter list or leading `void` (`../kernels/check.py`'s rule)."""
    n = kernel.strip()
    if n.startswith("void "):
        n = n[5:]
    out, i = [], 0
    while i < len(n):
        if n.startswith("(anonymous namespace)", i):
            out.append("(anonymous namespace)")
            i += len("(anonymous namespace)")
            continue
        if n[i] in "<(":
            break
        out.append(n[i])
        i += 1
    return "".join(out).strip()


def server_info(base_url: str) -> dict:
    response = requests.get(f"{base_url}/get_server_info", timeout=ACTION_S)
    response.raise_for_status()
    return response.json()


def max_model_len(base_url: str) -> int | None:
    response = requests.get(f"{base_url}/v1/models", timeout=ACTION_S)
    response.raise_for_status()
    data = response.json().get("data") or []
    return data[0].get("max_model_len") if data else None


def profile_ranks(base_url: str, tp_size: int, steps: int = 6, wait_s: float = 90.0,
                  attention_kernels: set[str] | None = None) -> dict[str, dict]:
    """Kernel-family counts per tensor-parallel rank from a GPU trace of a short greedy decode."""
    out_dir = Path(tempfile.mkdtemp(prefix="infb-recipe-profile-"))
    try:
        r = requests.post(f"{base_url}/start_profile", json={"output_dir": str(out_dir), "num_steps": steps,
                                                             "activities": ["GPU"], "profile_by_stage": False},
                          timeout=ACTION_S)
        if r.status_code != 200:
            raise AssertionError(f"/start_profile answered HTTP {r.status_code}: {r.text[:200]}")
        g = requests.post(f"{base_url}/generate", json={"text": PROMPT, "sampling_params": {
            "temperature": 0, "max_new_tokens": 40, "ignore_eos": True}}, timeout=REQUEST_S)
        if g.status_code != 200:
            raise AssertionError(f"the profiled decode answered HTTP {g.status_code}: {g.text[:200]}")
        time.sleep(3)
        try:
            requests.post(f"{base_url}/stop_profile", timeout=ACTION_S)
        except requests.RequestException:
            pass
        ranks: dict[str, dict] = {}
        files: dict[str, list[str]] = {}
        deadline = time.time() + wait_s
        while time.time() < deadline:
            files = {}
            for f in os.listdir(out_dir):
                m = re.search(r"TP-(\d+)", f)
                if m and f.endswith(".trace.json.gz"):
                    files.setdefault(m.group(1), []).append(f)
            if len(files) >= tp_size:
                time.sleep(5)
                break
            time.sleep(3)
        for rank, names in sorted(files.items()):
            counts: collections.Counter = collections.Counter()
            for f in names:
                trace = json.load(gzip.open(out_dir / f))
                for e in trace.get("traceEvents", []):
                    if e.get("ph") == "X" and e.get("cat") == "kernel":
                        counts[e.get("name", "")] += 1
            ranks[rank] = {
                "kernel_events": sum(counts.values()),
                "trtllm_mxfp4": {k: sum(c for n, c in counts.items() if p.search(n)) for k, p in TRTLLM_MXFP4.items()},
                "triton_fused_moe": sum(c for n, c in counts.items() if TRITON_FUSED_MOE.search(n)),
            }
            if attention_kernels:
                ranks[rank]["attention_kernels"] = {base_name(n)[-120:]: c for n, c in counts.items()
                                                    if base_name(n) in attention_kernels}
        return ranks
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def check(base_url: str, declared_context: int, attention_kernels: list[str] | None = None) -> tuple[list[str], dict]:
    """(the failures, one line each; the facts read). (e) is checked when the tree's packed-attention kernels (base
    names, from its component map) are given."""
    info = server_info(base_url)
    tp = int(info.get("tp_size") or 4)
    facts = {"kv_cache_dtype": info.get("kv_cache_dtype"), "disable_radix_cache": info.get("disable_radix_cache"),
             "moe_runner_backend": info.get("moe_runner_backend"), "tp_size": tp,
             "max_model_len": max_model_len(base_url), "declared_context": declared_context}
    family = set(attention_kernels) if attention_kernels else None
    if family is not None:
        facts["attention_backend"] = info.get("attention_backend")
        facts["internal_states_backends"] = [{k: st.get(k) for k in BACKEND_KEYS}
                                             for st in (info.get("internal_states") or [])]
    ranks = profile_ranks(base_url, tp, attention_kernels=family)
    facts["ranks"] = ranks
    failures = []
    if len(ranks) < tp:
        failures.append(f"(a) the GPU trace covered {len(ranks)} of {tp} tensor-parallel ranks")
    for rank, r in sorted(ranks.items()):
        missing = [k for k, n in r["trtllm_mxfp4"].items() if n == 0]
        if missing:
            failures.append(f"(a) rank {rank} ran no {', '.join(missing)} (the routed experts are not on the "
                            f"TRT-LLM MXFP4 runner that --moe-runner-backend flashinfer_mxfp4 selects)")
        if r["triton_fused_moe"]:
            failures.append(f"(a) rank {rank} ran the Triton fused-MoE kernel {r['triton_fused_moe']} times")
    if facts["kv_cache_dtype"] != "fp8_e4m3":
        failures.append(f"(b) the KV cache is {facts['kv_cache_dtype']!r}, not 'fp8_e4m3'")
    if facts["disable_radix_cache"] is not False:
        failures.append(f"(c) prefix caching is off (disable_radix_cache={facts['disable_radix_cache']!r})")
    if facts["max_model_len"] != declared_context:
        failures.append(f"(d) the served context is {facts['max_model_len']}, not the checkpoint's declared {declared_context}")
    if family is not None:
        for rank, r in sorted(ranks.items()):
            if not r.get("attention_kernels"):
                failures.append(f"(e) rank {rank} ran none of the tree's own packed-attention kernels "
                                f"({', '.join(sorted(family))[:300]}): the served attention is not the computation the "
                                f"tree's public interface performs")
    return failures, facts


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="check the served recipe on a running server")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--recipe", default=str(Path(__file__).resolve().parent / "recipe.json"))
    ap.add_argument("--component-map", default=None, help="the tree's component map (../kernels/check.py --profile-map)")
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.recipe).read_text())
    declared = int(json.loads((Path(spec["model_dir"]) / "config.json").read_text())["max_position_embeddings"])
    kernels = None
    if args.component_map:
        roles = json.loads(Path(args.component_map).read_text())["roles"]
        kernels = (roles.get("packed_attention") or {}).get("base_names") or None
    failures, facts = check(args.base_url, declared, kernels)
    print(json.dumps(facts, sort_keys=True))
    for f in failures:
        print(f"FAIL {f}")
    print("served recipe observations: " + ("no mismatches in the inspected fields" if not failures else f"{len(failures)} check(s) failed"))
    print("This report covers the configuration and profiling observations above; it does not establish "
          "CUDA-graph replay, effective speculation, or full task acceptance.")
    sys.stdout.flush()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
