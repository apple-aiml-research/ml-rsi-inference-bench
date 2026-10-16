#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""L0: the committed tree builds, the supplied kernels still run, and the engine resolves and parses the model natively.

Runs after `grader.py prepare` applied the candidate's `model.patch` to the pristine prepared base at /app and after
test.sh restored the upstream test package to its base blobs. Task facts (checkpoint path, declared architectures,
text-configuration fields) come from the `infb` block of /tests/config.json; every bound from the deadline
composition (`infb_verify/deadline_dag.py`, derived from the public `limits.json`).

  P2P  test_build_editable_install     the documented rebuild succeeds and the compiled extensions import
  P2P  test_supplied_kernels_import    the three supplied kernels each run one small case
  F2P  test_registry_resolves_native_class   both declared architectures resolve to classes under /app/vllm
  F2P  test_config_parses              get_config("/model", trust_remote_code=False) returns the text fields
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from infb_verify import deadline_dag

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
D = deadline_dag.deadlines()["l0"]
PY = sys.executable


def _run(code: str, timeout: float, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=timeout, cwd="/",
                          env=dict(os.environ, **(env or {})))


def _record(r: subprocess.CompletedProcess, what: str) -> dict:
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("INFB_L0 ")]
    assert r.returncode == 0 and lines, f"{what}:\n" + r.stdout[-3000:] + r.stderr[-10000:]
    return json.loads(lines[-1][len("INFB_L0 "):])


def test_build_editable_install():
    """The tree builds and installs editable, offline, through the same script the agent is given and the image was
    built with (it imports the compiled extensions before it exits 0); the package then resolves from /app."""
    r = subprocess.run(["/bin/bash", "/public/build.sh"], env=dict(os.environ, INFB_WORKSPACE=str(APP_DIR)),
                       capture_output=True, text=True, timeout=D["build_editable_install"], cwd=str(APP_DIR))
    assert r.returncode == 0, "offline rebuild failed:\n" + r.stdout[-6000:] + r.stderr[-6000:]
    r = _run("import os, vllm; print(os.path.realpath(vllm.__file__))", D["import_after_build"])
    assert r.returncode == 0, "import vllm failed:\n" + r.stdout[-2000:] + r.stderr[-6000:]
    assert str(APP_DIR.resolve()) in r.stdout, f"vllm resolved outside the tree: {r.stdout.strip()}"


ROUTER_GEMM = """
import json, torch
from vllm._custom_ops import fp32_router_gemm
torch.manual_seed(0)
a = torch.randn(4, 6144, dtype=torch.bfloat16, device="cuda")
b = torch.randn(128, 6144, dtype=torch.float32, device="cuda")
out = fp32_router_gemm(a, b)
ref = torch.nn.functional.linear(a.float(), b)
assert out.shape == (4, 128) and out.dtype == torch.float32, (out.shape, out.dtype)
err = (out - ref).abs().max().item()
assert err <= 2e-2, err
print("INFB_L0 " + json.dumps({"kernel": "fp32_router_gemm", "max_abs_err": err}))
"""

CLAMPED_SWIGLU = """
import json, torch
from vllm.config import VllmConfig, set_current_vllm_config
with set_current_vllm_config(VllmConfig()):
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    op = SiluAndMulWithClamp(7.0, 1.702, 1.0)
    torch.manual_seed(0)
    x = (4 * torch.randn(8, 2 * 3072, device="cuda")).to(torch.bfloat16)
    out = op.forward_cuda(x)
    ref = op.forward_native(x)
assert out.shape == (8, 3072) and out.dtype == torch.bfloat16, (out.shape, out.dtype)
err = (out.float() - ref.float()).abs().max().item()
torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
print("INFB_L0 " + json.dumps({"kernel": "silu_and_mul_with_clamp", "max_abs_err": err}))
"""

FUSED_ALLREDUCE_NORM = """
import json, os, sys, torch
import torch.multiprocessing as mp


def worker(rank, world, port, out_path):
    torch.cuda.set_device(rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                                     init_distributed_environment)
        from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
        from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import fused_allreduce_gemma_rms_norm
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        init_distributed_environment(world, rank, f"tcp://127.0.0.1:{port}", rank, "nccl")
        ensure_model_parallel_initialized(world, 1)
        torch.manual_seed(42)
        norm = GemmaRMSNorm(2048, eps=1e-6).cuda().to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.normal_(mean=0.0, std=0.1)
        torch.manual_seed(49)
        residual = torch.randn(128, 2048, dtype=torch.bfloat16, device="cuda")
        torch.manual_seed(1042 + rank)
        partial = torch.randn(128, 2048, dtype=torch.bfloat16, device="cuda")
        ref_out, ref_res = norm(tensor_model_parallel_all_reduce(partial.clone()), residual.clone())
        out, res = fused_allreduce_gemma_rms_norm(partial.clone(), residual.clone(), norm)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(res, ref_res, atol=2e-2, rtol=2e-2)
        if rank == 0:
            with open(out_path, "w") as fh:
                json.dump({"kernel": "fused_allreduce_gemma_rms_norm", "world_size": world,
                           "max_abs_err": (out.float() - ref_out.float()).abs().max().item()}, fh)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import socket, tempfile
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out_path = tempfile.mktemp(suffix=".json")
    mp.spawn(worker, args=(2, port, out_path), nprocs=2, join=True)
    print("INFB_L0 " + open(out_path).read())
"""


def test_supplied_kernels_import():
    """The kernels the task supplies in the starting tree run one small case each from the candidate's built
    tree: the FP32 router GEMM at the checkpoint's shape, the scaled clamped SwiGLU against its native form, and the
    all-reduce + Gemma RMSNorm fusion at TP 2 against the unfused path. Pass-to-pass: all three are present at the
    prepared base, so the unsolved base passes."""
    results, failures = {}, []
    for name, code, argv in (("fp32_router_gemm", ROUTER_GEMM, None),
                             ("silu_and_mul_with_clamp", CLAMPED_SWIGLU, None),
                             ("fused_allreduce_gemma_rms_norm", FUSED_ALLREDUCE_NORM, "file")):
        try:
            if argv == "file":  # multiprocessing spawn needs an importable main module
                path = Path("/tmp") / f"infb_l0_{name}.py"
                path.write_text(code)
                r = subprocess.run([PY, str(path)], capture_output=True, text=True,
                                   timeout=D["supplied_kernels_import"], cwd="/")
            else:
                r = _run(code, D["supplied_kernels_import"])
            results[name] = _record(r, name)
        except (AssertionError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{name}: {str(exc)[-1500:]}")
    assert not failures, (
        "supplied kernels that the committed tree cannot run:\n  " + "\n  ".join(failures)
        + "\n\nThese are supplied by the task in the starting tree and run at the base; a "
        "failure here means the submitted tree broke them or the package chain that reaches them."
    )


def test_registry_resolves_native_class():
    """The engine's registry resolves both declared architectures -- the checkpoint's and its text model's -- to native
    model classes whose source files lie under /app/vllm, not to the Transformers fallback (Transformers is pinned at
    5.5.3, which does not know this model type). The unsolved base fails while building the model configuration."""
    code = f"""
import inspect, json
from vllm.config.model import ModelConfig
from vllm.model_executor.models.registry import ModelRegistry
cfg = ModelConfig(model={X["model_path"]!r}, trust_remote_code=False, max_model_len=4096)
out = {{}}
for arch in {X["declared_architectures"]!r}:
    cls, resolved = ModelRegistry.resolve_model_cls([arch], cfg)
    out[arch] = {{"resolved": resolved, "module": cls.__module__, "cls": cls.__name__,
                 "file": inspect.getsourcefile(cls)}}
print("INFB_L0 " + json.dumps(out))
"""
    info = _record(_run(code, D["registry_resolves"]), "the architectures did not resolve")
    app = str((APP_DIR / "vllm").resolve())
    for arch in X["declared_architectures"]:
        got = info[arch]
        assert got["resolved"] == arch, got
        assert got["file"] and os.path.realpath(got["file"]).startswith(app + os.sep), (
            f"{arch} resolved to a class outside /app/vllm: {got}")
        assert "transformers" not in got["module"].rsplit(".", 1)[-1], f"{arch} resolved to the fallback: {got}"


def test_config_parses():
    """`get_config("/model", trust_remote_code=False)` parses the checkpoint's configuration without executing
    checkpoint code and returns the text configuration's fields. No GPU work; the unsolved base fails here on the
    unknown model type."""
    code = f"""
import json
from vllm.transformers_utils.config import get_config
cfg = get_config({X["model_path"]!r}, trust_remote_code=False)
text = cfg.get_text_config()
sparse = getattr(text, "sparse_attention_config", None)
if sparse is not None and not isinstance(sparse, dict):
    sparse = {{k: getattr(sparse, k) for k in dir(sparse) if k.startswith("sparse_") or k == "use_sparse_attention"}}
fields = {{k: getattr(text, k, None) for k in {sorted(X["text_config"])!r}}}
print("INFB_L0 " + json.dumps({{"model_type": getattr(cfg, "model_type", None),
                                "architectures": list(getattr(cfg, "architectures", None) or []),
                                "fields": fields, "sparse": sparse}}, default=str))
"""
    info = _record(_run(code, D["config_parses"]), "the checkpoint's configuration did not parse")
    assert info["model_type"] == X["model_type"], info
    assert info["fields"] == X["text_config"], {"parsed": info["fields"], "expected": X["text_config"]}
    sparse = info["sparse"] or {}
    for k, v in X["sparse_attention_config"].items():
        assert sparse.get(k) == v, {"field": k, "parsed": sparse.get(k), "expected": v}
