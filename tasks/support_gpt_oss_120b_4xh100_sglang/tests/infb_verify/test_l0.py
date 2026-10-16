#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""L0, build and registry: the committed tree installs and the engine resolves the model natively.

The task facts (checkpoint path, expected architecture) come from the "infb" block of
/tests/config.json. Scored by pytest node id under junit, as every DeepSWE check is. Runs after
grader.py prepare applied the candidate's model.patch to the pristine base at /app and after
test.sh restored the upstream test packages.
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
PY = sys.executable


def test_build_editable_install():
    """The tree installs editable, offline, without build isolation (the backend is in the image),
    and the package resolves from /app/python rather than from any cached copy."""
    env = dict(os.environ, SGLANG_BUILD_RUST_EXTS="none", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    r = subprocess.run(
        [PY, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "--no-build-isolation", "--no-index",
         "-e", str(APP_DIR / "third_party/triton_kernels"), "-e", str(APP_DIR / "python")],
        env=env, capture_output=True, text=True, timeout=1800, cwd=str(APP_DIR),
    )
    assert r.returncode == 0, "editable install failed:\n" + r.stdout[-4000:] + r.stderr[-4000:]
    r = subprocess.run(
        [PY, "-c", "import json,os,sglang,triton_kernels; print('ORIGINS '+json.dumps({n:os.path.realpath(m.__file__) for n,m in [('sglang',sglang),('triton_kernels',triton_kernels)]}))"],
        capture_output=True, text=True, timeout=600, cwd="/",
    )
    assert r.returncode == 0, "import sglang failed:\n" + r.stdout[-2000:] + r.stderr[-4000:]
    rows = [line[len("ORIGINS "):] for line in r.stdout.splitlines() if line.startswith("ORIGINS ")]
    assert rows, f"no import origin record: {r.stdout}"
    origins = json.loads(rows[-1])
    for name, root in (("sglang", APP_DIR / "python"),
                       ("triton_kernels", APP_DIR / "third_party/triton_kernels")):
        assert Path(origins[name]).is_relative_to(root.resolve()), f"{name} resolved outside editable source: {origins}"


def test_registry_resolves_native_class():
    """The checkpoint's config parses under the pinned Transformers through the engine (the base
    cannot: Transformers 5.12.1 does not know the model type) and the engine's own registry resolves
    the declared architecture to a native SGLang model class, not the Transformers fallback."""
    code = f"""
import json
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.utils import get_model_architecture
cfg = ModelConfig(model_path={X["model_dir"]!r}, trust_remote_code=False)
cls, arch = get_model_architecture(cfg)
print("INFB_L0 " + json.dumps({{"module": cls.__module__, "cls": cls.__name__, "arch": arch,
      "model_type": getattr(cfg.hf_config, "model_type", None),
      "architectures": list(getattr(cfg.hf_config, "architectures", []) or [])}}))
"""
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=900, cwd="/")
    assert r.returncode == 0, "config did not parse or the architecture did not resolve:\n" + r.stdout[-3000:] + r.stderr[-8000:]
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("INFB_L0 ")]
    assert lines, "no resolution record in output:\n" + r.stdout[-3000:]
    info = json.loads(lines[-1][len("INFB_L0 "):])
    assert info["arch"] == X["architectures"][0], info
    assert info["module"].startswith("sglang.srt.models."), f"resolved outside the engine's model package: {info}"
    assert not info["module"].endswith(".transformers"), f"resolved to the Transformers fallback: {info}"
    assert info["model_type"] == X["model_type"], info


def test_supplied_kernel_topk_survives_preparation():
    """Generic top-k stays usable at the prepared base and solved tree.

    Compare semantic selections/weights, without prescribing output ordering;
    repeat with changed logits through the same captured graph.
    """
    import torch
    from triton_kernels.topk import topk_forward
    generator = torch.Generator(device='cuda').manual_seed(190927)
    logits = torch.randn(17, 128, generator=generator, device='cuda')
    def check(values, indices):
        assert torch.equal(indices.long().sort(-1).values,
                           logits.topk(4, dim=-1).indices.sort(-1).values)
        expected = logits.gather(1, indices.long()).softmax(-1)
        torch.testing.assert_close(values, expected, atol=2e-6, rtol=2e-5)
    values, indices, _ = topk_forward(logits, 4)
    check(values, indices)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            topk_forward(logits, 4)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, indices, _ = topk_forward(logits, 4)
    logits.copy_(logits.roll(37, dims=-1).neg())
    graph.replay()
    torch.cuda.synchronize()
    check(values, indices)
