#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The two public MiniMax-M3-MXFP8 serving deployments (BF16 KV cache and FP8 KV cache), text only.

Both: TP4, 128-token KV blocks, a 262,144-token configured context, language model only, the ``minimax_m3``
reasoning and tool-call parsers with automatic tool choice, GPU memory utilization 0.85 (below the engine default to
leave headroom for prompt log-probability requests at the configured chunk size), and engine defaults otherwise
(chunked prefill, prefix caching, compilation and CUDA graphs). ``fp8kv`` adds ``--kv-cache-dtype fp8``. The dense
layers' attention backend is named per deployment.

  python /app/eval/minimax_m3/serving/recipes.py bf16kv --port 8000     # exec's vllm serve
"""

import argparse
import os

MODEL = "/model"
SERVED_MODEL_NAME = "MiniMaxAI/MiniMax-M3-MXFP8"
MAX_MODEL_LEN = 262_144
#: Below the engine default (0.92) to leave headroom for prompt log-probability requests at the configured chunk size.
GPU_MEMORY_UTILIZATION = 0.85
COMMON_ARGS = [
    "--served-model-name", SERVED_MODEL_NAME,
    "--tensor-parallel-size", "4",
    "--block-size", "128",
    "--max-model-len", str(MAX_MODEL_LEN),
    "--language-model-only",
    "--reasoning-parser", "minimax_m3",
    "--tool-call-parser", "minimax_m3",
    "--enable-auto-tool-choice",
    "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
    "--disable-uvicorn-access-log",
]
DEPLOYMENTS = {
    "bf16kv": [],
    "fp8kv": ["--kv-cache-dtype", "fp8"],
}
#: The dense layers' (0-2) attention backend, per deployment (the sparse layers' is the model's own).
DENSE_ATTENTION_BACKEND = {"bf16kv": "FLASHINFER", "fp8kv": "FLASHINFER"}


def server_args(deployment: str) -> list[str]:
    if deployment not in DEPLOYMENTS:
        raise ValueError(f"unknown deployment {deployment!r}; have {sorted(DEPLOYMENTS)}")
    args = [*COMMON_ARGS, *DEPLOYMENTS[deployment]]
    backend = DENSE_ATTENTION_BACKEND[deployment]
    if backend:
        args += ["--attention-backend", backend]
    return args


def argv(deployment: str = "bf16kv", *, port=None) -> list[str]:
    result = ["vllm", "serve", MODEL, *server_args(deployment)]
    if port is not None:
        result.extend(("--port", str(port)))
    return [*result, "--seed", "0"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("deployment", choices=sorted(DEPLOYMENTS))
    parser.add_argument("--port", type=int, default=8000)
    a = parser.parse_args()
    command = argv(a.deployment, port=a.port)
    os.execvp(command[0], command)
