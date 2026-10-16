#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The native-support checks: registry resolution, configuration parsing, the supplied kernels, and the build.

    python native.py [--workspace /app] [--python /opt/vllm/bin/python]

Each check runs in its own interpreter process, from `/` so that only the installed (editable) engine is imported:
  * registry: vLLM's model registry resolves the checkpoint's declared architecture to a native class inside the
    engine (not a Transformers fallback);
  * configuration: the checkpoint's composite configuration parses through the engine, with the text model's fields;
  * supplied kernels: the kernel modules supplied in the starting tree still import from your tree;
  * build: the engine's compiled extensions are present in the tree and `vllm` imports from the workspace.

After a change to compiled sources or the build's configuration (`compiled_sources_touched` below lists what
counts), rebuild with `../build.sh` first: the image's compiled extensions are the build of the starting sources.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

MODEL = "/model/deepseek-ai/DeepSeek-V4.1-Flash"
#: The checkpoint's declared architecture and the text model's fields, as its configuration states them.
ARCHITECTURE = "DeepseekV41ForCausalLM"
MODEL_TYPE = "deepseek_v41"
NUM_HIDDEN_LAYERS = 40
ENGRAM_LAYER_IDS = [1, 14]
#: The kernel modules the starting tree supplies.
SUPPLIED_KERNEL_MODULES = [
    "vllm.model_executor.kernels.attention.dsa.candidate_blocks",
    "vllm.model_executor.kernels.linear.mxfp8.deep_gemm",
    "vllm.model_executor.kernels.linear.mxfp8.flashinfer",
    "vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel",
    "vllm.model_executor.kernels.linear",
    "vllm.model_executor.kernels.mhc.tilelang",
    "vllm.model_executor.kernels.mhc.tilelang_kernels",
    "vllm.model_executor.kernels.mhc.torch",
    "vllm.model_executor.kernels.mhc.triton",
    "vllm.model_executor.kernels.mhc.warmup",
    "vllm.model_executor.layers.fused_moe.router.fused_moe_router",
    "vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant",
    "vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl",
    "vllm.models.deepseek_v4.nvidia.ops.o_proj",
]
#: The engine's compiled extensions under `<workspace>/vllm`.
COMPILED_EXTENSIONS = ["_C_stable_libtorch", "_moe_C_stable_libtorch", "cumem_allocator"]

#: A change to any path matching this makes the image's compiled extensions stale, so the tree must be REBUILT.
#: Deliberately conservative: everything the engine's build reads, source or configuration, and any binary a patch
#: could add. Anything else -- Python, and the Triton and TileLang kernels, which compile at run time -- leaves the
#: image's own build valid, because the install is editable.
_COMPILED_SOURCE = re.compile(
    r"^(csrc|cmake|requirements|rust)/"
    r"|(^|/)CMakeLists\.txt$|^setup\.py$|^setup\.cfg$|^pyproject\.toml$|^MANIFEST\.in$|(^|/)Cargo\.(toml|lock)$"
    r"|\.(cu|cuh|cpp|cc|cxx|c|h|hpp|hxx|inl|pyx|pxd|so|a|o|rs)$"
)


def patch_paths(patch_text: str) -> tuple[set[str], bool]:
    """Every path a patch names, on either side: additions, deletions, both ends of a rename or copy, and
    binaries. Returns (paths, parsed). A `diff --git` header that cannot be parsed (a quoted path with
    spaces) makes `parsed` False, and the caller then rebuilds: the conservative failure."""
    paths, parsed = set(), True
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            m = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", line)
            if m:
                paths.update(m.groups())
            else:
                parsed = False
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            paths.add(line.split(" ", 2)[2])
    return paths, parsed


def compiled_sources_touched(paths) -> list[str]:
    """The paths among `paths` whose change makes the compiled extensions stale."""
    return sorted(p for p in paths if _COMPILED_SOURCE.search(p))


def run(python: str, code: str, timeout: int) -> subprocess.CompletedProcess:
    """`python -c code`, from `/`, so the workspace is not on the interpreter's path."""
    return subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=timeout, cwd="/")


def registry_code(model: str) -> str:
    """The program that resolves the checkpoint's architecture through vLLM's registry and prints one INFB_L0 line."""
    return f"""
import json
from vllm.config.model import ModelConfig
from vllm.model_executor.models.registry import ModelRegistry
cfg = ModelConfig(model={model!r}, max_model_len=4096)
cls, arch = ModelRegistry.resolve_model_cls(cfg.architectures, cfg)
print("INFB_L0 " + json.dumps({{
    "module": cls.__module__, "cls": cls.__name__, "arch": arch,
    "model_type": getattr(cfg.hf_config, "model_type", None),
    "architectures": list(getattr(cfg.hf_config, "architectures", []) or []),
}}))
"""


def config_code(model: str) -> str:
    """The program that parses the checkpoint's configuration through the engine and prints one INFB_L0 line."""
    return f"""
import json
from vllm.config.model import ModelConfig
cfg = ModelConfig(model={model!r}, max_model_len=4096)
text = cfg.hf_text_config
print("INFB_L0 " + json.dumps({{
    "model_type": getattr(cfg.hf_config, "model_type", None),
    "num_hidden_layers": getattr(text, "num_hidden_layers", None),
    "engram_layer_ids": list(getattr(text, "engram_layer_ids", []) or []),
}}))
"""


def records(stdout: str) -> list[str]:
    """The INFB_L0 lines a check program printed."""
    return [ln for ln in stdout.splitlines() if ln.startswith("INFB_L0 ")]


def record(line: str) -> dict:
    return json.loads(line[len("INFB_L0 "):])


def registry_problems(info: dict, architecture: str) -> list[str]:
    """What is wrong with a registry resolution: a different architecture, a class outside the engine, the Transformers
    fallback."""
    out = []
    if info["arch"] != architecture:
        out.append(f"resolved {info['arch']!r}, not {architecture!r}")
    if not info["module"].startswith("vllm."):
        out.append(f"resolved outside the engine: {info}")
    if "transformers" in info["module"].rsplit(".", 1)[-1]:
        out.append(f"resolved to the Transformers fallback: {info}")
    return out


def import_failures(python: str, modules: list[str], timeout: int = 600) -> list[str]:
    """Each module that does not import from the installed engine, with the last line of its error."""
    failures = []
    for mod in modules:
        r = run(python, f"import {mod}; print('ok', {mod!r})", timeout)
        if r.returncode != 0:
            failures.append(f"{mod}: " + (r.stderr.strip().splitlines() or ["no output"])[-1][:400])
    return failures


def missing_extensions(workspace: Path, extensions: list[str]) -> list[str]:
    """The compiled extensions not present under `<workspace>/vllm`."""
    return [rel for rel in extensions if not list((workspace / "vllm").glob(f"{rel}*.so"))]


def import_root_code() -> str:
    return "import os, vllm, vllm._C_stable_libtorch; print(os.path.realpath(vllm.__file__))"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workspace", default="/app")
    ap.add_argument("--python", default="/opt/vllm/bin/python")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args(argv)
    workspace = Path(a.workspace)
    failed = 0

    def report(name: str, problems: list[str]) -> None:
        nonlocal failed
        failed += bool(problems)
        print(f"[{'FAIL' if problems else 'ok'}] {name}" + "".join(f"\n    {p}" for p in problems), flush=True)

    missing = missing_extensions(workspace, COMPILED_EXTENSIONS)
    r = run(a.python, import_root_code(), 600)
    problems = [f"compiled extensions missing from {workspace / 'vllm'}: {missing}"] if missing else []
    if r.returncode != 0:
        problems.append("import vllm failed: " + r.stderr[-2000:])
    elif str(workspace.resolve()) not in r.stdout:
        problems.append(f"vllm resolved outside the workspace: {r.stdout.strip()}")
    report("build: compiled extensions present; vllm imports from the workspace", problems)

    report("supplied kernels import", import_failures(a.python, SUPPLIED_KERNEL_MODULES))

    r = run(a.python, registry_code(a.model), 1200)
    lines = records(r.stdout)
    if r.returncode != 0 or not lines:
        report("registry resolves the architecture natively", ["the architecture did not resolve: " + r.stderr[-3000:]])
    else:
        info = record(lines[-1])
        print(f"    {info}")
        report("registry resolves the architecture natively", registry_problems(info, ARCHITECTURE))

    r = run(a.python, config_code(a.model), 900)
    lines = records(r.stdout)
    if r.returncode != 0 or not lines:
        report("configuration parses", ["the configuration did not parse: " + r.stderr[-3000:]])
    else:
        info = record(lines[-1])
        print(f"    {info}")
        want = {"model_type": MODEL_TYPE, "num_hidden_layers": NUM_HIDDEN_LAYERS, "engram_layer_ids": ENGRAM_LAYER_IDS}
        report("configuration parses", [f"{k} is {info.get(k)!r}, not {v!r}" for k, v in want.items() if info.get(k) != v])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
