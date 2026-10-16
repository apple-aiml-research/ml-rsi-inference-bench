#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Prepare the supplied MXFP4 sparse MQA-logits runtime in the engine tree (the task's sparse-indexer preparation).

    python3 prepare.py --app /app --work /tmp/infb-sparse-prep [--source-dir DIR]

Run by the image build, inside one step, so that no layer holds the unprepared dependency source. It

  1. fetches the pinned dependency source (sources.json: DeepGEMM at its pinned commit, with the pinned CUTLASS and
     DeepJIT commits its gitlinks name) and checks every commit identity and every file hash this script relies on;
  2. writes the editable source tree third_party/deepgemm_v41_sparse/: DeepGEMM's csrc/ and deep_gemm/include/, its
     generic Python quantization utilities, the DeepJIT and CUTLASS headers and the licenses -- no Git database,
     tests, wheels, build products, caches or API dumps -- less the four optimized target files the removal
     manifest lists, with the four target function bodies in csrc/apis/attention.hpp replaced by signature-compatible
     stubs that raise NotImplementedError, the JIT configuration prefix changed from DG to DSV41_SPARSE, and the
     narrow extension source csrc/python_api_sparse.cpp added;
  3. installs the package initializer, the builder and the two measurement entry points into the engine tree, and
     applies engine.patch (the DeepSelect top-k build and configuration, and the declarative sparse switch);
  4. checks every output against the hashes removal-manifest.json records, and deletes the fetched source.

It commits nothing: the image build commits the result, with the public materials, atop the materials commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch(url: str, commit: str, dest: Path) -> None:
    """A single commit of `url` into `dest`, verified; no history beyond that commit."""
    dest.mkdir(parents=True, exist_ok=True)
    git("init", "-q", cwd=dest)
    for attempt in range(3):
        try:
            git("fetch", "-q", "--depth", "1", url, commit, cwd=dest)
            break
        except subprocess.CalledProcessError as e:
            if attempt == 2:
                raise SystemExit(f"FATAL: cannot fetch {url} {commit}: {e.stderr}")
    git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD", cwd=dest)
    head = git("rev-parse", "HEAD", cwd=dest)
    if head != commit:
        raise SystemExit(f"FATAL: {url} checked out {head}, not {commit}")


def gitlink(repo: Path, path: str) -> str:
    """The commit a gitlink at `path` names in `repo`'s HEAD tree."""
    line = git("ls-tree", "HEAD", path, cwd=repo)
    mode, kind, commit = line.split()[:3]
    if mode != "160000" or kind != "commit":
        raise SystemExit(f"FATAL: {path} is not a gitlink in {repo}: {line}")
    return commit


def copy_tree(src: Path, dst: Path, exclude=()) -> None:
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src).as_posix()
        if p.is_dir() or any(rel == e or rel.startswith(e + "/") for e in exclude):
            continue
        if ".git" in p.relative_to(src).parts or "__pycache__" in p.relative_to(src).parts:
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out)


def replace_body(text: str, signature_start: str, stub: str) -> str:
    """Replace the body of the one function whose definition begins with `signature_start`: from the first `{`
    after its parameter list to the matching `}`. The signature, and the comment above it, are kept."""
    i = text.index(signature_start)
    assert text.count(signature_start) == 1, signature_start
    depth, j = 0, text.index("(", i)
    # skip the parameter list
    while True:
        c = text[j]
        depth += c == "("
        depth -= c == ")"
        j += 1
        if depth == 0:
            break
    k = text.index("{", j)
    depth, m = 0, k
    while True:
        c = text[m]
        depth += c == "{"
        depth -= c == "}"
        m += 1
        if depth == 0:
            break
    return text[:k] + "{\n" + stub + "}" + text[m:]


def remove_function(text: str, signature_start: str) -> str:
    """Remove one function definition, from its first line to its closing brace and the blank line after it."""
    i = text.index(signature_start)
    assert text.count(signature_start) == 1, signature_start
    k = text.index("{", text.index(")", i))
    depth, m = 0, k
    while True:
        c = text[m]
        depth += c == "{"
        depth -= c == "}"
        m += 1
        if depth == 0:
            break
    while text[m] == "\n":
        m += 1
    return text[:i] + text[m:]


NOT_IMPLEMENTED_HELPER = """// The four sparse MQA-logits operations of this runtime are not implemented here: each one raises Python's
// NotImplementedError with its name. Their declarations and registrations (names, argument names and defaults)
// are the interface.
[[noreturn]] static void sparse_mqa_logits_not_implemented(const char* name) {
    PyErr_SetString(PyExc_NotImplementedError, (std::string(name) + " is not implemented").c_str());
    throw pybind11::error_already_set();
}

"""

TARGETS = [
    ("static torch::Tensor get_sparse_mqa_logits_metadata(", "get_sparse_mqa_logits_metadata"),
    ("static torch::Tensor get_paged_sparse_mqa_logits_metadata(", "get_paged_sparse_mqa_logits_metadata"),
    ("static torch::Tensor fp8_fp4_sparse_mqa_logits(", "fp8_fp4_sparse_mqa_logits"),
    ("static torch::Tensor fp8_fp4_paged_sparse_mqa_logits(", "fp8_fp4_paged_sparse_mqa_logits"),
]


def stub_attention(text: str) -> str:
    """attention.hpp with the target implementation header's include, the workspace helper only the targets use,
    and the four target bodies removed; everything else byte for byte."""
    include = '#include "../jit_kernels/impls/sm100_sparse_mqa_logits.hpp"\n'
    assert text.count(include) == 1
    text = text.replace(include, "")
    text = remove_function(text, "static const torch::Tensor& get_sparse_mqa_logits_workspace(")
    first = text.index(TARGETS[0][0])
    # the helper goes before the comment block that precedes the first target
    comment = text.rfind("\n\n", 0, first) + 2
    text = text[:comment] + NOT_IMPLEMENTED_HELPER + text[comment:]
    for signature, name in TARGETS:
        text = replace_body(text, signature, f'    sparse_mqa_logits_not_implemented("{name}");\n')
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--app", type=Path, required=True, help="the engine tree (the vLLM repository)")
    ap.add_argument("--work", type=Path, required=True, help="scratch for the fetched source; deleted at the end")
    ap.add_argument("--source-dir", type=Path, help="an already-fetched source (development only; still verified)")
    ap.add_argument("--manifest-out", type=Path, help="write the computed manifest here (authoring only)")
    a = ap.parse_args()
    sources = json.loads((HERE / "sources.json").read_text())
    manifest = json.loads((HERE / "removal-manifest.json").read_text()) if (HERE / "removal-manifest.json").exists() else None
    dg = sources["deepgemm"]
    src = a.source_dir or (a.work / "DeepGEMM")
    if a.source_dir is None:
        if a.work.exists():
            shutil.rmtree(a.work)
        fetch(dg["url"], dg["commit"], src)
        for sub in ("cutlass", "deep_jit"):
            want = dg["gitlinks"][sub]
            if gitlink(src, f"third-party/{sub}") != want:
                raise SystemExit(f"FATAL: DeepGEMM's gitlink third-party/{sub} is not {want}")
            fetch(sources[sub]["url"], want, src / "third-party" / sub)
    else:
        if git("rev-parse", "HEAD", cwd=src) != dg["commit"]:
            raise SystemExit("FATAL: the source directory is not at the pinned commit")

    out = a.app / "third_party" / "deepgemm_v41_sparse"
    if out.exists():
        shutil.rmtree(out)
    removed = sources["removed"]
    copy_tree(src / "csrc", out / "csrc", exclude=[r[len("csrc/"):] for r in removed if r.startswith("csrc/")])
    copy_tree(src / "deep_gemm" / "include", out / "deep_gemm" / "include",
              exclude=[r[len("deep_gemm/include/"):] for r in removed if r.startswith("deep_gemm/include/")])
    for rel in sources["python_utilities"]:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, out / rel)
    shutil.copyfile(src / "LICENSE", out / "LICENSE")
    copy_tree(src / "third-party" / "deep_jit" / "include", out / "third-party" / "deep_jit" / "include")
    copy_tree(src / "third-party" / "cutlass" / "include", out / "third-party" / "cutlass" / "include")
    shutil.copyfile(src / "third-party" / "cutlass" / "LICENSE.txt", out / "third-party" / "cutlass" / "LICENSE.txt")
    attention = out / "csrc" / "apis" / "attention.hpp"
    original_attention = (src / "csrc" / "apis" / "attention.hpp").read_bytes()
    attention.write_text(stub_attention(original_attention.decode()))
    jit = out / "csrc" / "runtime" / "jit.hpp"
    original_jit = (src / "csrc" / "runtime" / "jit.hpp").read_bytes()
    jit_text, n = re.subn(r'^(\s*)"DG",$', r'\1"DSV41_SPARSE",', original_jit.decode(), flags=re.M)
    assert n == 1, "the JIT configuration prefix is not where it is expected"
    jit.write_text(jit_text)
    shutil.copyfile(HERE / "python_api_sparse.cpp", out / "csrc" / "python_api_sparse.cpp")

    for name, dest in sources["engine_files"].items():
        (a.app / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, a.app / dest)
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(HERE / "engine.patch")], cwd=a.app, check=True)

    record = {
        "inputs": {
            "csrc/apis/attention.hpp": sha256(original_attention),
            "csrc/runtime/jit.hpp": sha256(original_jit),
            **{r: sha256((src / r).read_bytes()) for r in removed},
        },
        "outputs": {
            p.relative_to(a.app).as_posix(): sha256(p.read_bytes())
            for p in sorted(out.rglob("*")) if p.is_file()
        },
    }
    for rel in removed:
        if (out / rel).exists():
            raise SystemExit(f"FATAL: {rel} survived the removal")
    if a.manifest_out:
        a.manifest_out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    if manifest is not None:
        bad = [k for k, v in manifest["inputs"].items() if record["inputs"].get(k) != v]
        bad += [k for k in sorted(set(manifest["outputs"]) | set(record["outputs"]))
                if manifest["outputs"].get(k) != record["outputs"].get(k)]
        if bad:
            raise SystemExit(f"FATAL: {len(bad)} file(s) differ from removal-manifest.json, e.g. {bad[:5]}")
    if a.source_dir is None:
        shutil.rmtree(a.work)
    print(f"[sparse-indexer prepare] {len(record['outputs'])} files in third_party/deepgemm_v41_sparse; "
          f"{len(removed)} targets withheld; engine patch applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
