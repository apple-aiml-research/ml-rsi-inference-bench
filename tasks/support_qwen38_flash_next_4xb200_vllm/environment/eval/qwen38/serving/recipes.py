#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The two public Qwen38 serving recipes."""

import argparse
import json
import os

MODEL = "Qwen/Qwen3.8-Flash-Next-FP8"
SERVER_ARGS = ["--tensor-parallel-size", "4", "--trust-remote-code",
               "--disable-uvicorn-access-log", "--gdn-prefill-backend", "flashinfer"]
MTP_ARGS = ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3})]


def argv(recipe="ordinary", *, port=None):
    if recipe not in ("ordinary", "mtp3"):
        raise ValueError(f"unknown recipe {recipe!r}")
    result = ["vllm", "serve", MODEL, *SERVER_ARGS]
    if recipe == "mtp3":
        result.extend(MTP_ARGS)
    if port is not None:
        result.extend(("--port", str(port)))
    return [*result, "--seed", "0"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=("ordinary", "mtp3"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    command = argv(args.recipe, port=args.port)
    os.execvp(command[0], command)
