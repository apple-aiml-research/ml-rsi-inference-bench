#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The served-configuration check: read a server's launch log and report every departure from the recipe.

    python served_configuration.py LAUNCH_LOG

LAUNCH_LOG is everything the server printed while it started, every rank's lines included (for example the
file `recipe.py serve --log FILE` writes). The check passes when the log shows at least one engine
configuration line and no departure; it prints its reading as JSON either way.

The configuration it reads against is `../recipe.py`'s (`RESOLVED`, `EXPERT_PARALLEL`, `ATTENTION_BACKEND_FAMILY`): the
recipe is the one statement of the deployment, and this check only reads a log against it. What is read, from the
engine's own log lines:
  * every "Initializing a V1 LLM engine ... with config:" line: tensor-, pipeline- and data-parallel sizes, the MoE
    backend, the compilation mode, the CUDA-graph mode, the tokenizer mode, the context length, the seed, and the
    speculative method and token count;
  * expert parallelism: any "Expert parallelism is enabled" line or any `Worker_TP<i>_EP<j>` worker is a departure;
  * each rank's attention backend, from the line it logs when the backend sets its KV-cache block size: every rank
    must serve attention on the engine's sparse-MLA backend family, `FLASHMLA_SPARSE` or a backend registered
    under a name that extends it.
The log's capture announcements are reported as facts, not required: no particular log spelling is part of the
deployment. These are configuration facts; whether CUDA graphs are actually replayed, whether the draft does real
work and whether the model computes correctly are runtime claims a launch log cannot establish.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path


def _recipe():
    """`../recipe.py`, beside this file in either copy of the materials, loaded by path (it imports the standard library
    alone)."""
    path = Path(__file__).resolve().parent.parent / "recipe.py"
    spec = importlib.util.spec_from_file_location("infb_dsv41_recipe_for_served_configuration", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RECIPE = _recipe()
#: The configuration the recipe specifies, as the engine's configuration line prints it (`recipe.RESOLVED`).
STATED_CONFIGURATION = dict(_RECIPE.RESOLVED)
_ENGINE_FIELDS = {
    "tensor_parallel_size": re.compile(r"tensor_parallel_size=(\d+)"),
    "pipeline_parallel_size": re.compile(r"pipeline_parallel_size=(\d+)"),
    "data_parallel_size": re.compile(r"data_parallel_size=(\d+)"),
    "moe_backend": re.compile(r"moe_backend=('[^']*'|None)"),
    "mode": re.compile(r"'mode': <CompilationMode\.(\w+)"),
    "cudagraph_mode": re.compile(r"'cudagraph_mode': <CUDAGraphMode\.(\w+)"),
    "tokenizer_mode": re.compile(r"tokenizer_mode=(\w+)"),
    "max_seq_len": re.compile(r"max_seq_len=(\d+)"),
    "seed": re.compile(r"seed=(\d+)"),
    "speculative_method": re.compile(r"speculative_config=SpeculativeConfig\(method='(\w+)'"),
    "num_spec_tokens": re.compile(r"speculative_config=SpeculativeConfig\([^)]*num_spec_tokens=(\d+)"),
}
_EP_WORKER = re.compile(r"\((Worker_TP\d+_EP\d+) pid=")
#: The engine's sparse-MLA backend family (`recipe.ATTENTION_BACKEND_FAMILY`): every backend registered as its subclass
#: under a name that extends it. The line is the one each rank logs when the attention backend's preferred KV-cache block
#: size replaces the default.
ATTENTION_BACKEND_FAMILY = _RECIPE.ATTENTION_BACKEND_FAMILY
_ATTENTION_BACKEND = re.compile(r"\((Worker_TP\d+)(?:_EP\d+)? pid=\d+\).*Setting kv cache block size to (\d+) for (\w+) backend\.")
#: Capture announcements the engine may log; counted as a fact for the record, never required.
_BREAKABLE = ("Breakable CUDA graph enabled", "Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1")


def served_configuration(launch_log: str) -> dict:
    """Read a launch log against `STATED_CONFIGURATION`; list every departure, one line each."""
    engine_lines: list[dict] = []
    expert_parallel_lines = 0
    ep_workers: set[str] = set()
    breakable_lines = 0
    attention_backends: dict[str, str] = {}
    for line in launch_log.splitlines():
        if "Initializing a V1 LLM engine" in line and "with config:" in line:
            engine_lines.append(
                {k: (m.group(1) if (m := rx.search(line)) else None) for k, rx in _ENGINE_FIELDS.items()}
            )
        if "Expert parallelism is enabled" in line:
            expert_parallel_lines += 1
        if m := _EP_WORKER.search(line):
            ep_workers.add(m.group(1))
        if any(b in line for b in _BREAKABLE):
            breakable_lines += 1
        if m := _ATTENTION_BACKEND.search(line):
            attention_backends[m.group(1)] = m.group(3)
    departures: list[str] = []
    for cfg in engine_lines:
        for key, stated in STATED_CONFIGURATION.items():
            if cfg.get(key) != stated:
                departures.append(f"{key} is {cfg.get(key)}, where the stated configuration has {stated}")
    if not _RECIPE.EXPERT_PARALLEL and (expert_parallel_lines or ep_workers):
        departures.append(
            f"expert parallelism is enabled ({expert_parallel_lines} 'Expert parallelism is enabled' lines, "
            f"workers {sorted(ep_workers)})"
        )
    if engine_lines:
        ranks = [f"Worker_TP{r}" for r in range(int(STATED_CONFIGURATION["tensor_parallel_size"]))]
        for rank in ranks:
            backend = attention_backends.get(rank)
            if backend is None:
                departures.append(
                    f"{rank} logs no attention backend (no 'Setting kv cache block size ... backend.' line), so "
                    f"it cannot be read as the engine's sparse-MLA backend"
                )
            elif not backend.startswith(ATTENTION_BACKEND_FAMILY):
                departures.append(
                    f"{rank} serves attention on {backend}, outside the engine's sparse-MLA backend family "
                    f"({ATTENTION_BACKEND_FAMILY} or a backend registered as its subclass)"
                )
    return {
        "engine_config_lines": engine_lines,
        "expert_parallel_lines": expert_parallel_lines,
        "expert_parallel_workers": sorted(ep_workers),
        "breakable_lines": breakable_lines,
        "attention_backends": dict(sorted(attention_backends.items())),
        "departures": sorted(set(departures)),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    path = Path(args[0])
    text = path.read_text(errors="replace") if path.is_file() else ""
    report = served_configuration(text)
    print(json.dumps(report, indent=1, sort_keys=True))
    if not text:
        print(f"no launch log at {path}")
        return 1
    if not report["engine_config_lines"]:
        print("no engine configuration line ('Initializing a V1 LLM engine ... with config:') in the log")
        return 1
    if report["departures"]:
        print("departures from the recipe:\n  " + "\n  ".join(report["departures"]))
        return 1
    print("the served configuration is the recipe's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
