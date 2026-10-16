#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the MXFP4 sparse MQA-logits runtime's `_C` pybind11 extension for <TARGET_PY>.

Derived from tools/build_deepgemm_C.py: the same three arguments and Torch ABI detection. The source is the
editable tree third_party/deepgemm_v41_sparse; the output directory is the package
vllm/third_party/deep_gemm_v41_sparse. The build compiles csrc/python_api_sparse.cpp, replaces the package's
JIT include tree with the source's own (DeepGEMM, DeepJIT, CUTLASS and CuTe), and writes build-manifest.json
(the source's file hashes, the initializer, this builder, the compiler, Torch and the flags; `source_id` is the
SHA256 of the sorted path/hash pairs of the source tree (bytecode caches aside), the initializer and this builder). The driver runs
against the build interpreter's torch; <TARGET_PY> is only consulted for INCLUDEPY and SOABI.

Usage: python build_deepgemm_v41_sparse.py <SRC> <OUT> <TARGET_PY>
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension

if len(sys.argv) != 4:
    sys.exit(f"usage: {sys.argv[0]} <SRC> <OUT> <TARGET_PY>")

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
target_py = sys.argv[3]
out.mkdir(parents=True, exist_ok=True)

info = json.loads(
    subprocess.check_output(
        [
            target_py,
            "-c",
            "import sysconfig, json; "
            "print(json.dumps({k: sysconfig.get_config_var(k) "
            "for k in ('EXT_SUFFIX', 'INCLUDEPY')}))",
        ]
    ).decode()
)

cuda_home = cpp_extension.CUDA_HOME
if cuda_home is None:
    sys.exit("CUDA_HOME not found; cannot build the sparse MQA-logits _C")
# CCCL lives outside the standard CUDAToolkit search (mirrors DeepGEMM's setup.py).
includes = [
    info["INCLUDEPY"],
    f"{cuda_home}/include",
    f"{cuda_home}/include/cccl",
    str(src / "csrc"),
    str(src / "deep_gemm/include"),
    str(src / "third-party/deep_jit/include"),
    str(src / "third-party/cutlass/include"),
    *cpp_extension.include_paths(device_type="cuda"),
]

flags = [
    "-shared",
    "-fPIC",
    "-std=c++20",
    "-O3",
    "-g0",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
    "-DTORCH_EXTENSION_NAME=_C",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
]
libs = [
    *(f"-L{p}" for p in cpp_extension.library_paths(device_type="cuda")),
    f"-L{cuda_home}/lib64",
    "-ltorch",
    "-ltorch_python",
    "-ltorch_cpu",
    "-ltorch_cuda",
    "-lc10",
    "-lc10_cuda",
    "-lcudart",
    "-lnvrtc",
]
cxx = os.environ.get("CXX", "g++")
so = out / f"_C{info['EXT_SUFFIX']}"
cmd = [
    cxx,
    *flags,
    *(f"-I{p}" for p in includes),
    str(src / "csrc/python_api_sparse.cpp"),
    *libs,
    "-o",
    str(so),
]
print("[build_deepgemm_v41_sparse] " + " ".join(cmd), flush=True)
subprocess.check_call(cmd)

# The JIT include tree, replaced whole: a header deleted or renamed in the source cannot survive from a
# previous build.
include = out / "include"
shutil.rmtree(include, ignore_errors=True)
for rel in ("deep_gemm/include", "third-party/deep_jit/include", "third-party/cutlass/include"):
    shutil.copytree(src / rel, include, dirs_exist_ok=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


files = {
    f"third_party/deepgemm_v41_sparse/{p.relative_to(src).as_posix()}": sha256(p)
    for p in sorted(src.rglob("*"))
    if p.is_file() and "__pycache__" not in p.parts
}
files["vllm/third_party/deep_gemm_v41_sparse/__init__.py"] = sha256(out / "__init__.py")
files["tools/build_deepgemm_v41_sparse.py"] = sha256(Path(__file__).resolve())
source_id = hashlib.sha256(
    "".join(f"{p}\0{h}\n" for p, h in sorted(files.items())).encode()
).hexdigest()
compiler = subprocess.check_output([cxx, "--version"]).decode().splitlines()[0]
manifest = {
    "schema": "deepgemm-v41-sparse-build/1",
    "source_id": source_id,
    "files": files,
    "compiler": compiler,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cxx11_abi": int(torch.compiled_with_cxx11_abi()),
    "flags": flags,
    "libraries": libs,
    "extension": so.name,
    "extension_sha256": sha256(so),
    "include_files": sum(1 for p in include.rglob("*") if p.is_file()),
}
(out / "build-manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
print(f"[build_deepgemm_v41_sparse] source_id {source_id}; {len(files)} source files; {so}", flush=True)
