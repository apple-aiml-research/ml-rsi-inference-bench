#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""L0, build and registry: the committed tree builds and the engine resolves the model natively.

The task facts (checkpoint id, expected architecture) come from the "infb" block of
/tests/config.json. Scored by pytest node id under junit, as every DeepSWE check is. Runs after
grader.py prepare applied the candidate's model.patch to the pristine prepared base at /app and
after test.sh restored the upstream test package to its base blobs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
sys.path.insert(0, str(TESTS_DIR))
from infb_verify import budgets  # noqa: E402

D = budgets.deadlines()["l0"]
PY = sys.executable


def _run(code: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=timeout, cwd="/")


def test_build_editable_install():
    """The tree builds and installs editable, offline, through the SAME script the agent is given
    (/app/eval/qwen38/build.sh, run from its trusted copy) and the image itself was built with; and the
    package resolves from /app rather than from any cached copy. CUDA, C++ and Rust sources are
    compiled here."""
    r = subprocess.run(
        ["/bin/bash", str(TESTS_DIR / "eval/qwen38/build.sh")],
        env=dict(os.environ, INFB_WORKSPACE=str(APP_DIR)),
        capture_output=True,
        text=True,
        timeout=D["build_editable_install"],
        cwd=str(APP_DIR),
    )
    assert r.returncode == 0, "offline rebuild failed:\n" + r.stdout[-6000:] + r.stderr[-6000:]
    r = _run("import os, vllm; print(os.path.realpath(vllm.__file__))", D["import_after_build"])
    assert r.returncode == 0, "import vllm failed:\n" + r.stdout[-2000:] + r.stderr[-6000:]
    assert str(APP_DIR.resolve()) in r.stdout, f"vllm resolved outside the tree: {r.stdout.strip()}"


def test_supplied_kernels_import():
    """The kernels the task SUPPLIES must still import from the candidate's built tree.

    The check sits here rather than in preflight, and the move is an attribution fix rather than a
    tidy-up. Preflight runs after `grader.py prepare` has applied `model.patch`, so it was importing
    CANDIDATE code: a submission that broke `vllm.models.qwen4_exp.nvidia.ops.hc` -- by editing the
    supplied module, or far more likely by breaking the package `__init__` chain that leads to it --
    made preflight fail, and a preflight failure exits `test.sh` without a reward so the frame's trap
    writes the -1 SENTINEL. That reads as "the environment was not the declared one" when what
    actually happened is that the candidate's tree does not import. -1 is reserved for faults that
    are not the candidate's; this is one of the candidate's, so it is a scored node and the trial
    scores 0.

    IT IS PASS-TO-PASS, measured rather than assumed: every one of these modules is present at the
    prepared base and imports only machinery that exists there (checked symbol by symbol at the
    upstream base when the preparation commit was authored), so the UNSOLVED base passes this node.
    A node that the base fails would be fail-to-pass and would be saying something else.

    It runs after the rebuild deliberately: the point is that the tree the verifier BUILT imports,
    and a candidate may have changed compiled sources the Triton modules sit on top of.
    """
    results = {}
    failures = []
    for mod in X["importable_candidate"]:
        r = _run(f"import {mod}; print('ok', {mod!r})", D["supplied_kernels_import"])
        results[mod] = r.returncode
        if r.returncode != 0:
            failures.append(f"{mod}: " + (r.stderr.strip().splitlines() or ["no output"])[-1][:400])
    assert not failures, (
        "supplied kernel modules that the committed tree cannot import:\n  " + "\n  ".join(failures)
        + "\n\nThese are SUPPLIED by the task (the preparation commit the instruction names), so they "
        "import at the base. A failure here means the submitted tree broke them or broke the package "
        "chain that reaches them."
    )


def test_registry_resolves_native_class():
    """The engine's own registry resolves the checkpoint's declared architecture to a native vLLM
    model class -- not to the Transformers fallback, which at this commit could not run it anyway
    (Transformers is pinned at 5.12.1 and does not know this model type). The unsolved base cannot
    get this far: it fails while building the model config."""
    code = f"""
import json
from vllm.config.model import ModelConfig
from vllm.model_executor.models.registry import ModelRegistry
cfg = ModelConfig(model={X["model_id"]!r}, trust_remote_code=True, max_model_len=4096)
cls, arch = ModelRegistry.resolve_model_cls(cfg.architectures, cfg)
model_info, _ = ModelRegistry.inspect_model_cls(cfg.architectures, cfg)
print("INFB_L0 " + json.dumps({{
    "module": cls.__module__, "cls": cls.__name__, "arch": arch,
    "is_hybrid": bool(getattr(model_info, "is_hybrid", False)),
    "model_type": getattr(cfg.hf_config, "model_type", None),
    "architectures": list(getattr(cfg.hf_config, "architectures", []) or []),
}}))
"""
    r = _run(code, D["registry_resolves"])
    assert r.returncode == 0, (
        "the architecture did not resolve:\n" + r.stdout[-3000:] + r.stderr[-10000:]
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("INFB_L0 ")]
    assert lines, "no resolution record in output:\n" + r.stdout[-3000:]
    info = json.loads(lines[-1][len("INFB_L0 ") :])
    assert info["arch"] == X["architectures"][0], info
    assert info["module"].startswith("vllm."), f"resolved outside the engine: {info}"
    assert "transformers" not in info["module"].rsplit(".", 1)[-1], (
        f"resolved to the Transformers fallback: {info}"
    )
    assert info["is_hybrid"], f"the model must report as hybrid for the hybrid KV manager: {info}"


def test_config_parses():
    """The checkpoint's configuration parses through the engine at the recipe's parallelism, with
    the structure the instruction states. No GPU work: this is the step the unsolved base fails,
    with a pydantic ValidationError on the unknown model type."""
    code = f"""
import json
from vllm.config.model import ModelConfig
cfg = ModelConfig(model={X["model_id"]!r}, trust_remote_code=True, max_model_len=4096)
text = cfg.hf_text_config
print("INFB_L0 " + json.dumps({{
    "model_type": getattr(cfg.hf_config, "model_type", None),
    "text_model_type": getattr(text, "model_type", None),
    "ple_layer_ids": list(getattr(text, "ple_layer_ids", []) or []),
    "num_hidden_layers": getattr(text, "num_hidden_layers", None),
}}))
"""
    r = _run(code, D["config_parses"])
    assert r.returncode == 0, (
        "the checkpoint's configuration did not parse:\n" + r.stdout[-3000:] + r.stderr[-10000:]
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("INFB_L0 ")]
    assert lines, "no configuration record in output:\n" + r.stdout[-3000:]
    info = json.loads(lines[-1][len("INFB_L0 ") :])
    assert info["model_type"] == X["model_type"], info
    assert info["ple_layer_ids"] == X["ple_layer_ids"], info
    assert info["num_hidden_layers"] == X["num_hidden_layers"], info
