#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Launch one of the prescribed Kimi-Linear-48B-A3B-Instruct deployments.

    python /app/eval/kimi_linear/launch.py [--deployment tp4|dp2|apc|apc-pressure] [--port 8000] [--log FILE]

`tp4`, the default, is the serving recipe: TP4/BF16 at vLLM's defaults for this commit (chunked prefill,
compiled CUDA graphs in FULL_AND_PIECEWISE mode, FlashMLA for the MLA layers), prefix caching explicitly
off, one long-prefill threshold beyond them, and the KDA worker extension, whose RPCs serve the
loaded-layer inspection, sample and timing of `kda/client.py`. Each other deployment differs from it only
by the arguments in `DEPLOYMENTS`:

* `dp2`: two TP4 data-parallel replicas (`--data-parallel-size 2`), prefix caching off;
* `apc`: TP4 with prefix caching on and 512-token state checkpoints (`--mamba-block-size 512`); the engine
  sets the attention block size to match;
* `apc-pressure`: `apc` at a 16,384-token context with a 64-block cache (`--num-gpu-blocks-override 64`), at
  most 16 running requests and 8,192 batched tokens per step: the capacity-limited configuration for cache
  turnover and preemption.

Data parallelism and prefix caching are separate deployments, never combined. A TP4 deployment uses four
GPUs and `dp2` all eight: the launch sets `CUDA_VISIBLE_DEVICES` to the first four, or all eight, of the
devices visible to it (those of an inherited `CUDA_VISIBLE_DEVICES`, else every device of the host). Each
deployment must actually serve requests on all of its GPUs.

Evaluation launches its servers from this file's `server_args(deployment)` (`SERVER_ARGS` is the `tp4`
recipe) and `environment()`, through vLLM's own test server wrapper, which adds a free `--port` and
`--seed 0`. A server has the published startup limit (`operation-limits.json`, 600 seconds) to become
healthy. Use `--log` to keep the server output that `tests/acceptance.py --mode recipe` checks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
MAX_MODEL_LEN = 1048576
PRESSURE_MAX_MODEL_LEN = 16384
SERVER_ARGS = [
    "--tensor-parallel-size",
    "4",
    "--max-model-len",
    str(MAX_MODEL_LEN),
    "--trust-remote-code",
    # Caps how many tokens ONE long prompt prefills per step. Without it, prompt_logprobs for one
    # 8,192-token chunk over this model's 163,840-token vocabulary need about 12 GiB that the
    # engine's startup memory profile never reserves. Every prompt at or below 2,048 tokens is
    # scheduled exactly as without it.
    "--long-prefill-token-threshold",
    "2048",
    "--worker-extension-cls",
    "worker.KDAWorker",
    "--no-enable-prefix-caching",
]
DEPLOYMENTS = {
    "tp4": {"remove": [], "add": [], "gpus": 4},
    "dp2": {"remove": [], "add": ["--data-parallel-size", "2"], "gpus": 8},
    "apc": {"remove": ["--no-enable-prefix-caching"],
            "add": ["--enable-prefix-caching", "--mamba-block-size", "512"], "gpus": 4},
    "apc-pressure": {"remove": ["--no-enable-prefix-caching"],
                     "add": ["--enable-prefix-caching", "--mamba-block-size", "512",
                             "--num-gpu-blocks-override", "64", "--max-num-seqs", "16",
                             "--max-num-batched-tokens", "8192"],
                     "max_model_len": PRESSURE_MAX_MODEL_LEN, "gpus": 4},
}
SEED = 0
LAUNCH_BOUND_S = json.loads((HERE / "operation-limits.json").read_text())["limits_s"]["startup"]


def server_args(deployment: str = "tp4") -> list[str]:
    """The arguments of `deployment`: the recipe with that deployment's stated difference."""
    spec = DEPLOYMENTS[deployment]
    args = [a for a in SERVER_ARGS if a not in spec["remove"]]
    if "max_model_len" in spec:
        args[args.index("--max-model-len") + 1] = str(spec["max_model_len"])
    return args + spec["add"]


def max_model_len(deployment: str = "tp4") -> int:
    return DEPLOYMENTS[deployment].get("max_model_len", MAX_MODEL_LEN)


def visible_devices() -> list[str]:
    """The devices a launch may use: an inherited CUDA_VISIBLE_DEVICES, else every device of the host."""
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None and inherited.strip():
        return [d.strip() for d in inherited.split(",") if d.strip()]
    listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return [str(i) for i, line in enumerate(l for l in listed.splitlines() if l.startswith("GPU "))]


def devices(deployment: str = "tp4", visible: list[str] | None = None) -> list[str]:
    """The GPUs of `deployment`: the first four visible devices for TP4, all eight for `dp2`."""
    available = visible_devices() if visible is None else visible
    wanted = DEPLOYMENTS[deployment]["gpus"]
    if len(available) < wanted:
        raise RuntimeError(f"the {deployment} deployment needs {wanted} GPUs; {len(available)} are visible")
    return available[:wanted]


def environment(*, kda_dir=None, cache_root=None, pythonpath="", deployment=None, visible=None):
    """What a launch adds to its environment.

    `kda_dir` holds the worker extension (default: this tree's `kda/`). `cache_root` gives the
    server its own Triton, Inductor and vLLM caches. With `deployment`, the GPUs it uses
    (`CUDA_VISIBLE_DEVICES`, from `devices`).
    """
    kda = Path(kda_dir) if kda_dir else HERE / "kda"
    env = {
        "PYTHONPATH": str(kda) + os.pathsep + pythonpath,
        "PYTHONDONTWRITEBYTECODE": "1", "VLLM_SERVER_DEV_MODE": "1",
    }
    if deployment is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices(deployment, visible))
    if cache_root is not None:
        cache = Path(cache_root)
        env.update({"TRITON_CACHE_DIR": str(cache / "triton"),
                    "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
                    "VLLM_CACHE_ROOT": str(cache / "vllm")})
    return env


def argv(port, seed=SEED, deployment="tp4"):
    """The complete command line of a launch of `deployment` at `port`."""
    return ["vllm", "serve", MODEL, *server_args(deployment), "--port", str(port), "--seed", str(seed)]


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deployment", choices=sorted(DEPLOYMENTS), default="tp4")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log", type=Path, help="also append the server's output to this file")
    parser.add_argument("--print", action="store_true", help="print the command and environment, launch nothing")
    options = parser.parse_args(args)
    added = environment(pythonpath=os.environ.get("PYTHONPATH", ""), deployment=options.deployment)
    command = argv(options.port, deployment=options.deployment)
    if options.print:
        for key, value in added.items():
            print(f"{key}={value}")
        print(" ".join(command))
        return 0
    env = {**os.environ, **added}
    if options.log is None:
        os.execvpe(command[0], command, env)
    options.log.parent.mkdir(parents=True, exist_ok=True)
    with options.log.open("ab") as log:
        server = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for line in server.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            server.terminate()
        return server.wait()


if __name__ == "__main__":
    sys.exit(main())
