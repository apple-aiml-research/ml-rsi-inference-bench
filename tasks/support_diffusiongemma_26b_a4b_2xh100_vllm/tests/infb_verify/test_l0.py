#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""L0: the in-place rebuild, the registry and the configuration.

    test_build_editable_install (P2P)            build.sh succeeds on the submitted tree, and the
        compiled products (_C_stable_libtorch, _moe_C_stable_libtorch, _vllm_fa2_C, _vllm_fa3_C)
        and the FA4 CuTe-DSL interface import in a fresh interpreter, the interface through the
        vendored link.
    test_registry_resolves_native_class (F2P)   DiffusionGemmaForBlockDiffusion resolves to a class
        whose source file lies under /app/vllm.
    test_config_parses (F2P)                    get_config("/model", trust_remote_code=False)
        returns the text configuration's layer types, head dimensions and window, and the canvas
        length, as /model/config.json states them.

Expected: all three pass on the oracle; on the prepared base the build passes and the other two
fail (no DiffusionGemma architecture or configuration).
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
from pathlib import Path

from infb_verify import budgets, common

APP = Path(os.environ.get("INFB_WORKSPACE", "/app"))
PYTHON = "/opt/vllm/bin/python"
IMPORTS = ("import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; "
           "import vllm.vllm_flash_attn._vllm_fa2_C, vllm.vllm_flash_attn._vllm_fa3_C; "
           "import vllm.vllm_flash_attn.cute.interface as i; print(i.__file__)")


def test_build_editable_install():
    log = Path("/logs/verifier/build.log")
    try:
        with log.open("w") as fh:
            p = subprocess.run(["bash", str(APP / "eval/diffusion_gemma/build.sh")], cwd=APP, stdout=fh,
                               stderr=subprocess.STDOUT, timeout=budgets.seconds("offline_rebuild"))
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    link = os.path.realpath(APP / "vllm/vllm_flash_attn/cute")
    imp = subprocess.run([PYTHON, "-c", IMPORTS], cwd="/", capture_output=True, text=True, timeout=600)
    rec = {"build_rc": rc, "log": str(log), "cute_link": link,
           "link_ok": link == str((APP / "third_party/vllm-flash-attn/flash_attn/cute").resolve()),
           "imports_rc": imp.returncode, "imports_out": (imp.stdout + imp.stderr)[-2000:]}
    rec["pass"] = rc == 0 and rec["link_ok"] and imp.returncode == 0
    common.record("test_build_editable_install", rec)
    assert rec["pass"], rec


def test_registry_resolves_native_class():
    from vllm.model_executor.models.registry import ModelRegistry

    cls = ModelRegistry._try_load_model_cls("DiffusionGemmaForBlockDiffusion")
    src = os.path.realpath(inspect.getsourcefile(cls)) if cls is not None else None
    rec = {"class": None if cls is None else f"{cls.__module__}.{cls.__qualname__}", "source": src,
           "pass": src is not None and src.startswith(str(APP.resolve() / "vllm") + os.sep)}
    common.record("test_registry_resolves_native_class", rec)
    assert rec["pass"], rec


def test_config_parses():
    from vllm.transformers_utils.config import get_config

    want = json.loads(Path("/model/config.json").read_text())
    cfg = get_config("/model", trust_remote_code=False)
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", cfg)
    tw = want["text_config"]
    got = {"layer_types": list(getattr(text, "layer_types", None) or []), "head_dim": getattr(text, "head_dim", None),
           "global_head_dim": getattr(text, "global_head_dim", None), "sliding_window": getattr(text, "sliding_window", None),
           "canvas_length": getattr(cfg, "canvas_length", None)}
    exp = {"layer_types": tw["layer_types"], "head_dim": tw["head_dim"], "global_head_dim": tw["global_head_dim"],
           "sliding_window": tw["sliding_window"], "canvas_length": want["canvas_length"]}
    rec = {"type": type(cfg).__name__, "mismatches": {k: [got[k], exp[k]] for k in exp if got[k] != exp[k]}}
    rec["pass"] = not rec["mismatches"]
    common.record("test_config_parses", rec)
    assert rec["pass"], rec
