#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The serving recipe: the one launch configuration this task specifies, and a launcher for it.

    python recipe.py argv [--port 8000]                 # print the launch command
    python recipe.py serve [--port 8000] [--log FILE]   # launch it in the foreground, output teed to FILE

The served checks, the efficiency measurements and the evaluation all run against a server started with
exactly this argv: `vllm serve <checkpoint>` with the arguments below, then `--port <port> --seed 0`.
Only the port varies. This file is the one declarative statement of the deployment: its launch arguments
(`SERVER_ARGS`) and the resolved engine configuration they must produce (`RESOLVED`), which
`tests/served_configuration.py` reads a launch log against:

  * tensor parallelism 4; pipeline and data parallelism 1; expert parallelism off;
  * native sparse-MLA attention on every rank and automatic MoE backend selection;
  * CompilationMode.NONE with CUDA graphs, CUDAGraphMode.FULL_AND_PIECEWISE;
  * deepseek_v41 tokenizer mode, maximum context 32768 and seed 0;
  * DSpark with 5 speculative tokens, using the checkpoint's draft weights.

These settings include engine defaults resolved for this model, not only explicit CLI arguments. The
configuration a launch log states is a configuration fact; it does not establish model correctness,
sustained CUDA-graph replay, real draft work or efficiency.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

MODEL = "/model/deepseek-ai/DeepSeek-V4.1-Flash"
#: The recipe's arguments, in order. `--port` and `--seed 0` follow them.
SERVER_ARGS = [
    "--tensor-parallel-size", "4",
    "--tokenizer-mode", "deepseek_v41",
    "--max-model-len", "32768",
    "--speculative-config", '{"method": "dspark", "num_speculative_tokens": 5}',
    "--disable-uvicorn-access-log",
]
SEED = "0"
#: The resolved engine configuration the launch must produce, field by field as the engine's configuration line
#: ("Initializing a V1 LLM engine ... with config:") prints it.
RESOLVED = {
    "tensor_parallel_size": "4",
    "pipeline_parallel_size": "1",
    "data_parallel_size": "1",
    "moe_backend": "'auto'",
    "mode": "NONE",
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "tokenizer_mode": "deepseek_v41",
    "max_seq_len": "32768",
    "seed": SEED,
    "speculative_method": "dspark",
    "num_spec_tokens": "5",
}
#: Expert parallelism is off: no rank may serve as an expert-parallel worker.
EXPERT_PARALLEL = False
#: Every tensor-parallel rank serves attention on the engine's native sparse-MLA backend family: FLASHMLA_SPARSE, or a
#: backend registered under a name that extends it.
ATTENTION_BACKEND_FAMILY = "FLASHMLA_SPARSE"
#: How long a launch may take to become healthy.
STARTUP_MAX_WAIT_SECONDS = 7000
#: The engine's interpreter and its `vllm` entry point.
PYTHON = "/opt/vllm/bin/python"
VLLM = "/opt/vllm/bin/vllm"


def serve_argv(port: int | str) -> list[str]:
    """The launch command for one port: `vllm serve <checkpoint> <recipe arguments> --port <port> --seed 0`."""
    return ["vllm", "serve", MODEL, *SERVER_ARGS, "--port", str(port), "--seed", SEED]


def serve(port: int, log: str | None) -> int:
    """Run the server in the foreground; with `log`, every line it prints also goes to that file."""
    argv = serve_argv(port)
    env = dict(os.environ, PATH=f"{os.path.dirname(VLLM)}:{os.environ.get('PATH', '')}")
    exe = [VLLM, *argv[1:]] if os.path.exists(VLLM) else argv
    if not log:
        return subprocess.call(exe, env=env)
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, "wb") as sink:
        proc = subprocess.Popen(exe, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for chunk in iter(lambda: proc.stdout.read1(1 << 16), b""):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                sink.write(chunk)
                sink.flush()
        except KeyboardInterrupt:
            proc.terminate()
        return proc.wait()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("argv", help="print the launch command")
    a.add_argument("--port", default="8000")
    s = sub.add_parser("serve", help="launch the recipe in the foreground")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--log", help="also write the server's output here (tests/served_configuration.py reads it)")
    args = ap.parse_args(argv)
    if args.cmd == "argv":
        print(shlex.join(serve_argv(args.port)))
        return 0
    return serve(args.port, args.log)


if __name__ == "__main__":
    sys.exit(main())
