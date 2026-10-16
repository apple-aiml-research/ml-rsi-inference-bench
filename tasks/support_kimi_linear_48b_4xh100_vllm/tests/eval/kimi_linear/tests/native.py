#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Native support checks: the build, the engine's registry and the checkpoint's configuration.

    python native.py                 # the build, registry and configuration checks
    python native.py --upstream      # also vLLM's two registration tests for this model (one GPU)

These are the grader's own checks. The build check: `vllm` and its compiled core import from
/app, and the image's seven compiled extensions are present. A change to a compiled source or the
build's configuration (the paths `compiled_sources_touched` names) needs `../build.sh`, and the
grader runs that same script on such a submission; Python and Triton edits need no rebuild, because
the install is editable. The registry check: the engine resolves the checkpoint's declared
`KimiLinearForCausalLM` to a native vLLM class that reports itself hybrid. The configuration
check: the checkpoint's configuration parses through the engine with its declared structure,
read from `/model/config.json`. `--upstream` runs, from /app, vLLM's initialization test for this
model (one layer, dummy weights, tensor-parallel size 1 on one GPU) and its registry-import test.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
ARCHITECTURE = "KimiLinearForCausalLM"
BASE = os.environ.get("INFB_BASE_COMMIT")
PY = sys.executable
ACTION_LIMIT_S = json.loads((Path(__file__).resolve().parents[1] / "operation-limits.json").read_text())["limits_s"]["action"]
UPSTREAM_NODES = (
    f"tests/models/test_initialization.py::test_can_initialize_large_subset[{ARCHITECTURE}]",
    f"tests/models/test_registry.py::test_registry_imports[{ARCHITECTURE}]",
)

_COMPILED_SOURCE = re.compile(
    r"^(csrc|cmake|requirements)/"
    r"|(^|/)CMakeLists\.txt$|^setup\.py$|^setup\.cfg$|^pyproject\.toml$|^MANIFEST\.in$"
    r"|\.(cu|cuh|cpp|cc|cxx|c|h|hpp|hxx|inl|pyx|pxd|so|a|o)$"
)
EXTENSIONS = (("", "_C"), ("", "_moe_C"), ("", "_flashmla_C"), ("", "_flashmla_extension_C"),
              ("", "cumem_allocator"), ("vllm_flash_attn", "_vllm_fa2_C"), ("vllm_flash_attn", "_vllm_fa3_C"))


def run(code: str, timeout: int = ACTION_LIMIT_S) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=timeout, cwd="/")


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


def tree_paths(base: str, app: Path = APP_DIR) -> set[str]:
    """What the working tree changed from `base`: its diff (renames split into both ends) and untracked files."""
    paths = set()
    for cmd in (["git", "-C", str(app), "diff", "--name-only", "--no-renames", base],
                ["git", "-C", str(app), "ls-files", "--others", "--exclude-standard"]):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=ACTION_LIMIT_S)
        assert r.returncode == 0, f"{cmd}: {r.stderr[-2000:]}"
        paths |= set(r.stdout.split())
    return paths


def missing_extensions(app: Path = APP_DIR) -> list[str]:
    return [f"{d}/{m}" if d else m for d, m in EXTENSIONS if not list((app / "vllm" / d).glob(f"{m}*.so"))]


def import_check(app: Path = APP_DIR) -> None:
    """`vllm` and its compiled core import, from the tree at `app`."""
    r = run("import os, vllm, vllm._C; print(os.path.realpath(vllm.__file__))")
    assert r.returncode == 0, "import vllm failed:\n" + r.stdout[-2000:] + r.stderr[-6000:]
    assert str(app.resolve()) in r.stdout, f"vllm resolved outside the tree: {r.stdout.strip()}"


def _record(r: subprocess.CompletedProcess, what: str) -> dict:
    assert r.returncode == 0, (f"{what}:\n" + r.stdout[-3000:] + r.stderr[-10000:])
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("INFB_L0 ")]
    assert lines, "no record in output:\n" + r.stdout[-3000:]
    return json.loads(lines[-1][len("INFB_L0 "):])


def registry_resolution(model_id: str = MODEL) -> dict:
    """The engine's own registry resolves the checkpoint's declared architecture: module, class, hybrid."""
    code = f"""
import json
from vllm.config.model import ModelConfig
from vllm.model_executor.models.registry import ModelRegistry
cfg = ModelConfig(model={model_id!r}, trust_remote_code=True, max_model_len=4096)
cls, arch = ModelRegistry.resolve_model_cls(cfg.architectures, cfg)
model_info, _ = ModelRegistry.inspect_model_cls(cfg.architectures, cfg)
print("INFB_L0 " + json.dumps({{
    "module": cls.__module__, "cls": cls.__name__, "arch": arch,
    "is_hybrid": bool(getattr(model_info, "is_hybrid", False)),
    "model_type": getattr(cfg.hf_config, "model_type", None),
    "architectures": list(getattr(cfg.hf_config, "architectures", []) or []),
}}))
"""
    return _record(run(code), "the architecture did not resolve")


def check_registry(info: dict, architecture: str = ARCHITECTURE) -> None:
    assert info["arch"] == architecture, info
    assert info["module"].startswith("vllm."), f"resolved outside the engine: {info}"
    assert "transformers" not in info["module"].rsplit(".", 1)[-1], (
        f"resolved to the Transformers fallback: {info}"
    )
    assert info["is_hybrid"], f"the model must report as hybrid for the hybrid KV manager: {info}"


def config_structure(model_id: str = MODEL) -> dict:
    """The checkpoint's configuration, parsed through the engine."""
    code = f"""
import json
from vllm.config.model import ModelConfig
cfg = ModelConfig(model={model_id!r}, trust_remote_code=True, max_model_len=4096)
hf = cfg.hf_config
text = cfg.hf_text_config
lac = getattr(text, "linear_attn_config", None) or {{}}
print("INFB_L0 " + json.dumps({{
    "model_type": getattr(hf, "model_type", None),
    "num_hidden_layers": getattr(text, "num_hidden_layers", None),
    "full_attn_layers": list(lac.get("full_attn_layers", []) or []),
    "kda_layers": list(lac.get("kda_layers", []) or []),
    "mla_use_nope": bool(getattr(text, "mla_use_nope", False)),
    "num_experts": getattr(text, "num_experts", None),
    "num_experts_per_token": getattr(text, "num_experts_per_token", None),
    "num_shared_experts": getattr(text, "num_shared_experts", None),
    "num_nextn_predict_layers": getattr(text, "num_nextn_predict_layers", None),
}}))
"""
    return _record(run(code), "the checkpoint's configuration did not parse")


def declared_structure(config_path: Path = Path("/model/config.json")) -> dict:
    """The structure the checkpoint's own config.json declares."""
    cfg = json.loads(config_path.read_text())
    lac = cfg.get("linear_attn_config") or {}
    return {"model_type": cfg["model_type"], "num_hidden_layers": cfg["num_hidden_layers"],
            "full_attn_layers": list(lac.get("full_attn_layers") or []),
            "kda_layer_count": len(lac.get("kda_layers") or []), "num_experts": cfg["num_experts"],
            "num_experts_per_token": cfg["num_experts_per_token"], "num_shared_experts": cfg["num_shared_experts"]}


def check_config(info: dict, want: dict) -> None:
    assert info["model_type"] == want["model_type"], info
    assert info["num_hidden_layers"] == want["num_hidden_layers"], info
    assert info["full_attn_layers"] == want["full_attn_layers"], info
    assert len(info["kda_layers"]) == want["kda_layer_count"], info
    assert info["mla_use_nope"] is True, info
    assert info["num_experts"] == want["num_experts"], info
    assert info["num_experts_per_token"] == want["num_experts_per_token"], info
    assert info["num_shared_experts"] == want["num_shared_experts"], info
    assert info["num_nextn_predict_layers"] == 0, info


def run_upstream(nodes=UPSTREAM_NODES, app: Path = APP_DIR, junit: Path | None = None) -> int:
    """vLLM's own tests for a newly registered model, run in place from the tree."""
    command = [PY, "-m", "pytest", "-p", "no:cacheprovider", "--rootdir", str(app), "-q", *nodes]
    if junit is not None:
        command += ["--junitxml", str(junit)]
    return subprocess.run(command, cwd=str(app)).returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--upstream", action="store_true", help="also run the two upstream tests (one GPU)")
    args = parser.parse_args(argv)
    results, failed = {}, False
    base = BASE or subprocess.run(["git", "-C", str(APP_DIR), "rev-list", "--max-parents=0", "HEAD"],
                                  capture_output=True, text=True).stdout.split()[-1]
    checks = {
        "build": lambda: (import_check(), {"missing_extensions": missing_extensions(),
                                           "needs_build_sh": compiled_sources_touched(tree_paths(base))})[1],
        "registry": lambda: (check_registry(info := registry_resolution()), info)[1],
        "config": lambda: (check_config(info := config_structure(), declared_structure()), info)[1],
    }
    for name, check in checks.items():
        try:
            results[name] = {"passed": True, **check()}
            if name == "build" and results[name]["missing_extensions"]:
                results[name]["passed"] = False
        except AssertionError as exc:
            results[name] = {"passed": False, "why": str(exc)[-4000:]}
        failed |= not results[name]["passed"]
        print(f"[native] {name}: {'ok' if results[name]['passed'] else 'FAILED'}", flush=True)
    print(json.dumps(results, indent=2))
    if args.upstream:
        rc = run_upstream()
        print(f"[native] upstream tests exit {rc}", flush=True)
        failed |= rc != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
