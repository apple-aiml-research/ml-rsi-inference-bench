#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The served configuration a running server reports (README, `run.py configuration`).

    python configuration.py --base-url http://127.0.0.1:30000 [--launch-log server.log] [--output /tmp/c.json]

Reads `/server_info` and prints, for each item graded by the served-configuration node, the observed value and
whether it matches the registered recipe (tests/test_inkling_small_nvfp4.py, production class): TP 4, page size 128,
`modelopt_fp4` with the `flashinfer_trtllm` FP4 GEMM runner and the `flashinfer_trtllm_routed` MoE runner, attention
backend `fa4`, the `extra_buffer` mamba radix-cache strategy, both memory ratios 0.1, `mem_fraction_static` 0.85, the
radix cache enabled, chunked prefill enabled (0 < chunked_prefill_size < 262,144), decode CUDA graphs enabled, and a
served input length of at least 262,144 tokens. The unified radix tree is the launch environment's
`SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`; with `--launch-log`, lines mentioning it are shown. The resolved
`chunked_prefill_size`, `cuda_graph_backend_prefill` and the pool sizes are printed as diagnostics, not conditions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from server import server_info  # noqa: E402

SCOPE = 262144
CHECKS = [
    ("tp_size", lambda v: v == 4, "4"),
    ("page_size", lambda v: v == 128, "128"),
    ("quantization", lambda v: v == "modelopt_fp4", "modelopt_fp4"),
    ("fp4_gemm_runner_backend", lambda v: v == "flashinfer_trtllm", "flashinfer_trtllm"),
    ("moe_runner_backend", lambda v: v == "flashinfer_trtllm_routed", "flashinfer_trtllm_routed"),
    ("attention_backend", lambda v: v == "fa4", "fa4"),
    ("mamba_radix_cache_strategy", lambda v: v == "extra_buffer", "extra_buffer"),
    ("swa_full_tokens_ratio", lambda v: v == 0.1, "0.1"),
    ("mamba_full_memory_ratio", lambda v: v == 0.1, "0.1"),
    ("mem_fraction_static", lambda v: v == 0.85, "0.85"),
    ("disable_radix_cache", lambda v: v is False, "false (radix cache enabled)"),
    ("chunked_prefill_size", lambda v: isinstance(v, int) and 0 < v < SCOPE, f"0 < size < {SCOPE}"),
    ("disable_cuda_graph", lambda v: v is False, "false (decode CUDA graphs enabled)"),
    ("max_req_input_len", lambda v: isinstance(v, int) and v >= SCOPE, f">= {SCOPE}"),
]
DIAGNOSTICS = ("chunked_prefill_size", "cuda_graph_backend_prefill", "cuda_graph_backend_decode", "max_total_num_tokens",
               "max_req_input_len", "context_length", "enable_deterministic_inference", "mamba_track_interval", "version")


def evaluate(info: dict) -> dict:
    items = []
    for key, ok, want in CHECKS:
        have = info.get(key)
        items.append({"key": key, "observed": have, "expected": want, "matches": bool(ok(have))})
    return {"checks": items, "all_match": all(i["matches"] for i in items),
            "diagnostics": {k: info.get(k) for k in DIAGNOSTICS}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--launch-log", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    rec = evaluate(server_info(args.base_url))
    if args.launch_log:
        lines = args.launch_log.read_text(errors="replace").splitlines()
        rec["launch_log_unified_radix_tree"] = [ln[:300] for ln in lines if "unified" in ln.lower()][:20]
    for i in rec["checks"]:
        print(f"{'ok  ' if i['matches'] else 'DIFF'} {i['key']}: {i['observed']!r} (recipe: {i['expected']})")
    print("diagnostics:", json.dumps(rec["diagnostics"]))
    if args.output:
        args.output.write_text(json.dumps(rec, indent=1))
    return 0 if rec["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
