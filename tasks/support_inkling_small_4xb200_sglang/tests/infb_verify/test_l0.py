#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""L0: the committed tree installs, parses the checkpoint's configuration and resolves the model natively.

Expected verdicts, stated first: the prepared base passes `test_build_editable_install` and fails the other two (the
base has no Inkling model and Transformers 5.12.1 has no `inkling` model type); the reference passes all three.

  test_build_editable_install          P2P  `pip install --no-deps --no-build-isolation --no-index -e /app/python`
                                            with SGLANG_BUILD_RUST_EXTS=none, then `sglang` resolves from /app/python
  test_registry_resolves_native_class  F2P  the engine's registry resolves `InklingForConditionalGeneration` to a class
                                            whose source lies under /app/python (not the Transformers fallback)
  test_config_parses                   F2P  the engine's model-config loader parses the checkpoint and returns the text
                                            configuration's architecture facts

Each check is one existing-engine test-file operation (`limits.json` `unit_test_file`, 1,800 s).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
sys.path.insert(0, str(TESTS_DIR))
from infb_efficiency import operations  # noqa: E402

X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
MODEL = X.get("model_dir", "/model/thinkingmachines/Inkling-Small-NVFP4")
PY = sys.executable
LIMIT_S = operations.limit_s("unit_test_file")
EXPECT = {"num_hidden_layers": 42, "sliding_window_size": 512, "rel_extent": 1024, "d_rel": 16,
          "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "log_scaling_n_floor": 128000,
          "log_scaling_alpha": 0.1, "route_scale": 8.0, "num_experts_per_tok": 6, "n_routed_experts": 256,
          "n_shared_experts": 2}
FULL_LAYERS = (5, 11, 17, 23, 29, 35, 41)


def _run(code: str) -> tuple[subprocess.CompletedProcess, dict | None]:
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=LIMIT_S, cwd="/")
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("INFB_L0 ")]
    return r, (json.loads(lines[-1][len("INFB_L0 "):]) if lines else None)


def test_build_editable_install():
    env = dict(os.environ, SGLANG_BUILD_RUST_EXTS="none", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    with operations.operation("install", "the documented editable install"):
        r = subprocess.run([PY, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "--no-build-isolation",
                            "--no-index", "-e", str(APP_DIR / "python")], env=env, capture_output=True, text=True,
                           timeout=operations.limit_s("install"), cwd=str(APP_DIR))
    assert r.returncode == 0, "editable install failed:\n" + r.stdout[-4000:] + r.stderr[-4000:]
    r, info = _run('import json, os, sglang; print("INFB_L0 " + json.dumps(os.path.realpath(sglang.__file__)))')
    assert r.returncode == 0 and info, "import sglang failed:\n" + r.stdout[-2000:] + r.stderr[-4000:]
    assert str((APP_DIR / "python").resolve()) in info, f"sglang resolved outside the tree: {info}"


def test_registry_resolves_native_class():
    code = f"""
import inspect, json
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.utils import get_model_architecture
cfg = ModelConfig(model_path={MODEL!r}, trust_remote_code=False)
cls, arch = get_model_architecture(cfg)
print("INFB_L0 " + json.dumps({{"module": cls.__module__, "cls": cls.__name__, "arch": arch,
      "file": inspect.getsourcefile(cls)}}))
"""
    with operations.operation("unit_test_file", "the registry resolution"):
        r, info = _run(code)
    assert info, "the configuration did not parse or the architecture did not resolve:\n" + r.stdout[-3000:] + r.stderr[-8000:]
    assert info["arch"] == "InklingForConditionalGeneration", info
    assert info["module"].startswith("sglang.srt.models.") and not info["module"].endswith(".transformers"), info
    assert str((APP_DIR / "python").resolve()) in str(Path(info["file"]).resolve()), f"the class is not under /app/python: {info}"


def test_config_parses():
    code = f"""
import json
from sglang.srt.configs.model_config import ModelConfig
cfg = ModelConfig(model_path={MODEL!r}, trust_remote_code=False)
hf = cfg.hf_config
tc = getattr(hf, "text_config", hf)
keys = {sorted(EXPECT)!r} + ["local_layer_ids", "layer_types"]
print("INFB_L0 " + json.dumps({{k: getattr(tc, k, None) for k in keys}}, default=str))
"""
    with operations.operation("unit_test_file", "the configuration parse"):
        r, info = _run(code)
    assert info, "the configuration did not parse:\n" + r.stdout[-3000:] + r.stderr[-8000:]
    wrong = {k: (info.get(k), v) for k, v in EXPECT.items() if info.get(k) != v}
    local = info.get("local_layer_ids")
    types = info.get("layer_types")
    derived = sorted(i for i, t in enumerate(types or []) if t == "hybrid_sliding") if types else None
    got_local = sorted(local) if local else derived
    if got_local != [i for i in range(42) if i not in FULL_LAYERS]:
        wrong["local layers"] = (got_local, f"every layer but {FULL_LAYERS}")
    assert not wrong, f"text configuration facts differ (observed, expected): {wrong}"
