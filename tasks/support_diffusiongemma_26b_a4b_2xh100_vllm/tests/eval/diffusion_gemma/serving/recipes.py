#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The public DiffusionGemma serving recipes.

The base command is the day-0 recipe's "full-featured server" (vllm-project/recipes,
models/Google/diffusiongemma-26B-A4B-it.yaml): a configured context of 262,144 tokens, four
concurrent sequences, 85% of GPU memory, images at up to 1,120 soft tokens and seven per prompt, and
the `gemma4` reasoning and tool-call parsers. Everything else is at the engine's defaults: chunked
prefill, prefix caching, asynchronous scheduling, compilation and CUDA graphs; no attention backend
is named. Compilation runs with TorchInductor's deterministic mode and combo kernels off (the
`--compilation-config` argument): under it a read whose batch is fixed repeats between launches of
the same build, so the verifier judges a fixed-batch read once. There are two deployments -- one GPU, the published recipe, and tensor parallelism across
two GPUs -- and two fixed-work generation configurations, in which no canvas stops early:

    tp1         GPU 0, the checkpoint's generation configuration
    tp1_fixed   GPU 0, confidence_threshold 0.0, t_max 5.0, t_min 0.15
    tp2_fixed   GPUs 0 and 1, tensor parallel size 2, confidence_threshold 0.0
    ar_smoke    Qwen2.5-0.5B-Instruct on GPU 0, engine defaults

A fixed-work configuration is the checkpoint's generation_config.json with the keys above changed.
`python recipes.py gencfg --out DIR` writes them (one directory per configuration); the verifier
keeps the same files under /opt/infb/gencfg. `python recipes.py serve <recipe>` execs the server.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MODEL = "/model"
SERVED_NAME = "google/diffusiongemma-26B-A4B-it"
AR_MODEL = "/models/qwen2.5-0.5b-instruct"
BASE_ARGS = [
    "--served-model-name", SERVED_NAME,
    "--max-model-len", "262144",
    "--max-num-seqs", "4",
    "--gpu-memory-utilization", "0.85",
    "--mm-processor-kwargs", json.dumps({"max_soft_tokens": 1120}),
    "--limit-mm-per-prompt", json.dumps({"image": 7}),
    "--enable-auto-tool-choice",
    "--tool-call-parser", "gemma4",
    "--reasoning-parser", "gemma4",
    "--compilation-config", json.dumps({"inductor_compile_config": {"deterministic": True, "combo_kernels": False}}),
]
GENCFG = {
    "tp1_fixed": {"confidence_threshold": 0.0, "t_max": 5.0, "t_min": 0.15},
    "tp2_fixed": {"confidence_threshold": 0.0},
}
VERIFIER_GENCFG_ROOT = "/opt/infb/gencfg"
RECIPES = {
    "tp1": {"devices": "0", "model": MODEL, "args": BASE_ARGS, "gencfg": None},
    "tp1_fixed": {"devices": "0", "model": MODEL, "args": BASE_ARGS, "gencfg": "tp1_fixed"},
    "tp2_fixed": {"devices": "0,1", "model": MODEL, "args": [*BASE_ARGS, "--tensor-parallel-size", "2"],
                  "gencfg": "tp2_fixed"},
    "ar_smoke": {"devices": "0", "model": AR_MODEL, "args": [], "gencfg": None},
}


def write_gencfg(out: str | os.PathLike, model: str = MODEL) -> dict[str, str]:
    """Write each fixed-work generation configuration as DIR/<name>/generation_config.json."""
    base = json.loads((Path(model) / "generation_config.json").read_text())
    written = {}
    for name, changes in GENCFG.items():
        target = Path(out) / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "generation_config.json").write_text(json.dumps({**base, **changes}, indent=2) + "\n")
        written[name] = str(target)
    return written


def argv(recipe: str, *, gencfg_root: str | os.PathLike | None = None, port: int | None = None) -> list[str]:
    """The server command of a recipe; the fixed-work configurations are read from gencfg_root."""
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; choose from {sorted(RECIPES)}")
    spec = RECIPES[recipe]
    result = ["vllm", "serve", spec["model"], *spec["args"]]
    if spec["gencfg"]:
        root = Path(gencfg_root or VERIFIER_GENCFG_ROOT)
        result.extend(["--generation-config", str(root / spec["gencfg"])])
    if port is not None:
        result.extend(["--port", str(port)])
    return result


def devices(recipe: str) -> str:
    """The CUDA_VISIBLE_DEVICES a recipe serves on."""
    return RECIPES[recipe]["devices"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gencfg", help="write the fixed-work generation configurations")
    g.add_argument("--out", required=True)
    s = sub.add_parser("serve", help="exec the server of a recipe")
    s.add_argument("recipe", choices=sorted(RECIPES))
    s.add_argument("--gencfg-root", default=None)
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "gencfg":
        print(json.dumps(write_gencfg(args.out), indent=1))
        return
    command = argv(args.recipe, gencfg_root=args.gencfg_root, port=args.port)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", devices(args.recipe))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
