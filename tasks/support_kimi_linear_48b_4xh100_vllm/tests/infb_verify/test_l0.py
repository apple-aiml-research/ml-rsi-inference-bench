#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""L0, build and registry: the committed tree builds and the engine resolves the model natively."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
EVAL = TESTS_DIR / "eval/kimi_linear"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NATIVE = _load("infb_kimi_native", EVAL / "tests/native.py")
REGRESSIONS = _load("infb_kimi_regressions", EVAL / "tests/regressions.py")
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/logs/artifacts"))
REBUILD_LIMIT_S = json.loads((EVAL / "operation-limits.json").read_text())["limits_s"]["rebuild"]
patch_paths = NATIVE.patch_paths
compiled_sources_touched = NATIVE.compiled_sources_touched
SIBLING_LAYOUT = REGRESSIONS.SIBLING_LAYOUT


def _submission_paths() -> tuple[set[str], bool]:
    """What the submission changed: its patch parsed on both sides (which sees git-ignored paths too),
    united with the working tree's diff from the base (renames split into both ends) and untracked files."""
    paths, parsed = set(), True
    model_patch = ARTIFACTS_DIR / "model.patch"
    if model_patch.is_file():
        p, parsed = patch_paths(model_patch.read_text(errors="replace"))
        paths |= p
    paths |= NATIVE.tree_paths(X["base_commit"], APP_DIR)
    return paths, parsed


def test_build_editable_install():
    """The tree builds and installs editable, offline, through the SAME script the agent is given and
    the image itself was built with; and the package resolves from /app rather than from any cached
    copy.

    CONDITIONAL. When the submission touches a compiled source or
    the build's configuration (`compiled_sources_touched`), the tree is rebuilt exactly as before, CUDA and C++ compiled. When it touches none, the image's own build products are already the
    build of those unchanged sources, so rebuilding would only recompile them; the node then REQUIRES
    those products (the seven extensions of this commit) and requires that `vllm` and its compiled core
    import from /app, and says so in one line. A pure-Python submission is served as the agent tested it.
    The script is the grading copy of the supplied `/app/eval/kimi_linear/build.sh`, byte for byte.

    This is a PASS-TO-PASS node: the unsolved base builds too, and must. It is scored so that a
    submission which breaks the build is a failure with a named cause rather than a cascade of
    unexplained ones.
    """
    paths, parsed = _submission_paths()
    touched = compiled_sources_touched(paths)
    if touched or not parsed:
        why = ", ".join(touched[:8]) if touched else "an unparseable patch header"
        print(f"[l0] rebuild: the diff touches compiled source or build configuration ({why})", flush=True)
        r = subprocess.run(
            ["/bin/bash", str(EVAL / "build.sh")],
            env=dict(os.environ, INFB_WORKSPACE=str(APP_DIR)),
            capture_output=True,
            text=True,
            timeout=REBUILD_LIMIT_S,
            cwd=str(APP_DIR),
        )
        assert r.returncode == 0, "offline rebuild failed:\n" + r.stdout[-6000:] + r.stderr[-6000:]
    else:
        print(f"[l0] rebuild skipped: diff touches no compiled source ({len(paths)} path(s) changed)", flush=True)
        missing = NATIVE.missing_extensions(APP_DIR)
        assert not missing, f"the image's compiled extensions are missing from /app/vllm: {missing}"
    NATIVE.import_check(APP_DIR)


def test_registry_resolves_native_class():
    """The engine's own registry resolves the checkpoint's declared architecture to a native vLLM
    model class -- not to the Transformers fallback, which at this commit could not run it anyway
    (Transformers is pinned at 4.57.1 and does not know this model type; it was additionally
    measured that `--model-impl transformers` is refused, because the checkpoint's `auto_map` has no
    `AutoModel` the Transformers backend can use). The unsolved base cannot get this far: it fails
    while building the model config.

    `is_hybrid` is asserted because this model MUST report as hybrid for the engine's hybrid KV-cache
    manager to lay out 20 recurrent-state layers beside 7 full-attention ones. A class that resolves
    but does not declare itself hybrid would serve the wrong cache.
    """
    NATIVE.check_registry(NATIVE.registry_resolution(X["model_id"]), X["architectures"][0])


def test_config_parses():
    """The checkpoint's configuration parses through the engine, with the structure the instruction
    states. No GPU work: this is the step the unsolved base fails, with a pydantic ValidationError on
    the unknown model type `kimi_linear`.

    The layer split is asserted because it is the shape of the whole task: 27 layers of which exactly
    7 are full attention and 20 are Kimi Delta Attention, and `mla_use_nope` true, which is why those
    7 layers have no rotary embedding. A configuration that parses with the wrong split would build a
    model that is not this one.
    """
    want = {k: X[k] for k in ("model_type", "num_hidden_layers", "full_attn_layers", "kda_layer_count",
                              "num_experts", "num_experts_per_token", "num_shared_experts")}
    NATIVE.check_config(NATIVE.config_structure(X["model_id"]), want)


def test_sibling_hybrid_config_unchanged():
    """The tree does not change how the engine configures another hybrid architecture (instruction item 6).
    Weight-free and GPU-free: a tiny Jamba configuration written to disk,
    no checkpoint. `JambaForSequenceClassification` is the one other hybrid architecture in `MODELS_CONFIG_MAP`
    at this commit (vllm/model_executor/models/config.py:496). At the base it gets exactly the hybrid KV-cache
    layout `JambaForCausalLM` gets, because `HybridAttentionMambaModelConfig` runs for both: the attention block
    is sized to cover the mamba page and the mamba page is padded to it. A tree that skips that step for the
    map's entries leaves the classifier unaligned; one agent's `vllm/config/vllm.py` did.
    Both forms are compared with each other and with the base's own values, measured on the reference
    implementation and on a second control tree.
    """
    REGRESSIONS.check_sibling(REGRESSIONS.sibling_layout(), SIBLING_LAYOUT)
