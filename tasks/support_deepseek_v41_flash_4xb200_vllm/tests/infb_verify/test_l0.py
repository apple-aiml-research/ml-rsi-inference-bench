#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""L0, build, registry and construction: the committed tree builds, the engine resolves the model natively, and the
class it resolves constructs and loads at the recipe's layout."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from infb_verify import _trusted

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
CONFIG = json.loads((TESTS_DIR / "config.json").read_text())
X = CONFIG["infb"]
# The prepared base is config.json's top-level `base_commit` (the grader's key), not an infb key.
BASE_COMMIT = CONFIG["base_commit"]
PY = sys.executable
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/logs/artifacts"))
ACTION_S = int(X["operation_caps_s"]["action"])


class OperationCapExceeded(AssertionError):
    """A candidate operation that ran past its public limit on a healthy host: a resource failure, scored."""


def _run(code: str, timeout: int = ACTION_S) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=timeout, cwd="/")
    except subprocess.TimeoutExpired as exc:
        raise OperationCapExceeded(f"the operation exceeded its {timeout} s limit: {code[:160]!r}") from exc


_NATIVE = _trusted.load("tests/native.py")
patch_paths = _NATIVE.patch_paths
compiled_sources_touched = _NATIVE.compiled_sources_touched
BUILD_SH = str(_trusted.EVAL / "build.sh")


def _submission_paths() -> tuple[set[str], bool]:
    """What the submission changed: its patch parsed on both sides (which sees git-ignored paths too),
    united with the working tree's diff from the base (renames split into both ends) and untracked files."""
    paths, parsed = set(), True
    model_patch = ARTIFACTS_DIR / "model.patch"
    if model_patch.is_file():
        p, parsed = patch_paths(model_patch.read_text(errors="replace"))
        paths |= p
    for cmd in (["git", "-C", str(APP_DIR), "diff", "--name-only", "--no-renames", BASE_COMMIT],
                ["git", "-C", str(APP_DIR), "ls-files", "--others", "--exclude-standard"]):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=ACTION_S)
        assert r.returncode == 0, f"{cmd}: {r.stderr[-2000:]}"
        paths |= set(r.stdout.split())
    return paths, parsed


def test_build_editable_install():
    """The tree builds and installs editable, offline, through the SAME script the agent is given and the
    image itself was built with; and the package resolves from /app rather than from any cached copy.

    CONDITIONAL (the closure is Python plus Triton and TileLang sources that compile at run time).
    When the submission touches a compiled source or the build's configuration (`_COMPILED_SOURCE`), the
    tree is rebuilt with the task's build.sh; when it touches none, the image's own build products are
    already the build of those unchanged sources, so the node REQUIRES them (the extensions named in
    config.json `infb.compiled_extensions`) and requires that `vllm` imports from /app, and says so in one
    line. PASS-TO-PASS: the unsolved base builds too, and must.
    """
    paths, parsed = _submission_paths()
    touched = compiled_sources_touched(paths)
    if touched or not parsed:
        why = ", ".join(touched[:8]) if touched else "an unparseable patch header"
        print(f"[l0] rebuild: the diff touches compiled source or build configuration ({why})", flush=True)
        limit = int(os.environ.get("INFB_REBUILD_TIMEOUT_SEC", str(X["operation_caps_s"]["rebuild"])))
        try:
            r = subprocess.run(
                ["/bin/bash", BUILD_SH],
                env=dict(os.environ, INFB_WORKSPACE=str(APP_DIR)),
                capture_output=True,
                text=True,
                timeout=limit,
                cwd=str(APP_DIR),
            )
        except subprocess.TimeoutExpired as exc:
            raise OperationCapExceeded(f"the compiled-source rebuild exceeded its {limit} s limit") from exc
        assert r.returncode == 0, "offline rebuild failed:\n" + r.stdout[-6000:] + r.stderr[-6000:]
    else:
        print(f"[l0] rebuild skipped: diff touches no compiled source ({len(paths)} path(s) changed)", flush=True)
        missing = [rel for rel in X["compiled_extensions"] if not list((APP_DIR / "vllm").glob(f"{rel}*.so"))]
        assert not missing, f"the image's compiled extensions are missing from /app/vllm: {missing}"
    r = _run("import os, vllm, vllm._C_stable_libtorch; print(os.path.realpath(vllm.__file__))")
    assert r.returncode == 0, "import vllm failed:\n" + r.stdout[-2000:] + r.stderr[-6000:]
    assert str(APP_DIR.resolve()) in r.stdout, f"vllm resolved outside the tree: {r.stdout.strip()}"


def test_supplied_kernels_import():
    """The kernels the task SUPPLIES (the preparation commit) must still import from the candidate's
    built tree. PASS-TO-PASS, measured: each module imports at the prepared base. A failure means the
    submitted tree broke them or the package chain that reaches them -- the candidate's fault, so a scored
    node rather than the preflight sentinel."""
    failures = []
    for mod in X["importable_candidate"]:
        r = _run(f"import {mod}; print('ok', {mod!r})")
        if r.returncode != 0:
            failures.append(f"{mod}: " + (r.stderr.strip().splitlines() or ["no output"])[-1][:400])
    assert not failures, (
        "supplied kernel modules that the committed tree cannot import:\n  " + "\n  ".join(failures)
    )


def test_registry_resolves_native_class():
    """The engine's own registry resolves the checkpoint's declared architecture to a native vLLM model
    class, not to the Transformers fallback (no Transformers release knows this model). The unsolved base
    fails here: its registry has no entry for the architecture."""
    code = _NATIVE.registry_code(X["model_id"])
    r = _run(code)
    assert r.returncode == 0, "the architecture did not resolve:\n" + r.stdout[-3000:] + r.stderr[-10000:]
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("INFB_L0 ")]
    assert lines, "no resolution record in output:\n" + r.stdout[-3000:]
    info = json.loads(lines[-1][len("INFB_L0 "):])
    assert info["arch"] == X["architectures"][0], info
    assert info["module"].startswith("vllm."), f"resolved outside the engine: {info}"
    assert "transformers" not in info["module"].rsplit(".", 1)[-1], f"resolved to the Transformers fallback: {info}"


def test_config_parses():
    """The checkpoint's configuration parses through the engine with the structure the instruction states.
    PASS-TO-PASS: the base already carries this model's configuration class with its frontends."""
    code = _NATIVE.config_code(X["model_id"])
    r = _run(code)
    assert r.returncode == 0, "the checkpoint's configuration did not parse:\n" + r.stdout[-3000:] + r.stderr[-10000:]
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("INFB_L0 ")]
    assert lines, "no configuration record in output:\n" + r.stdout[-3000:]
    info = json.loads(lines[-1][len("INFB_L0 "):])
    assert info["model_type"] == X["model_type"], info
    assert info["num_hidden_layers"] == X["num_hidden_layers"], info
    assert info["engram_layer_ids"] == X["engram_layer_ids"], info


CONSTRUCT = TESTS_DIR / "infb_verify" / "construct.py"
_RECIPE = _trusted.load("recipe.py")
#: Markers the reward's disposition reads in a failure's message (disposition.py); text a candidate's code raised never
#: carries one into this node's message.
_MARKERS = re.compile(r"blocked_by:|(reference|verifier|environment)_fault:|DeadlineExceeded:|"
                      r"reached the verifier clock's end|the group was killed at its|before this node reported")


def _quoted(text: str, limit: int = 600) -> str:
    return _MARKERS.sub(lambda m: m.group(0).replace(":", ";").replace(" ", "_"), str(text))[:limit]


def test_registered_class_constructs():
    """The class the engine's registry resolves for the checkpoint's architecture constructs at the recipe's layout, and
    its own `load_weights` takes the checkpoint. On each of the recipe's four ranks, one process per GPU, `construct.py`: the engine's
    configuration from the served command line, the engine's worker distributed initialisation (NCCL, TP 4), the model
    built as the engine's loader builds it, and then the constructed class's own `load_weights`, called once over a lazy
    generator that walks the checkpoint shard by shard in the index's order and creates each tensor only when the loader
    draws it, with the exact name, dtype and shape from that shard's safetensors header and zero data. Every name the
    index holds passes exactly once. Every rank completes, or the node fails. Structural and deterministic: no
    tolerance, no list of expected names. Not run: post-load processing, the DSpark draft and serving, which the served
    nodes judge; a loader that silently drops a tensor it cannot place passes here and fails there.

    FAIL-TO-PASS: the unsolved base has no class for the architecture, and the engine refuses its
    configuration. One runtime observation, under this node's own bound, which is longer than one unit-test
    file process and so is not the common invocation cap. The GPUs must be free when it starts (otherwise
    `blocked_by:gpu_busy`, an environment fault), and no process of it outlives it."""
    from infb_verify import construct  # noqa: PLC0415  (the standard library only, at module level)

    c = X["l0"]["construct"]
    records = Path(c["records"])
    try:
        records.mkdir(parents=True, exist_ok=True)
        for old in [*records.glob("*.json"), records / "construct.log"]:
            old.unlink(missing_ok=True)
        request = records / "request.json"
        request.write_text(json.dumps({"argv": [_RECIPE.MODEL, *_RECIPE.SERVER_ARGS, "--seed", _RECIPE.SEED],
                                       "records": str(records), "grace_s": c["grace_s"]}, indent=1) + "\n")
    except OSError as exc:
        pytest.fail(f"verifier_fault: the construction node's records could not be prepared: {exc}", pytrace=False)
    busy = construct.gpus_busy()
    if busy:
        pytest.fail(f"blocked_by:gpu_busy: before the construction node: {busy}", pytrace=False)

    limit = int(c["cap_s"])
    with open(records / "construct.log", "wb") as log:
        p = subprocess.Popen([PY, "-I", str(CONSTRUCT), "launch", "--request", str(request)], stdout=log,
                             stderr=subprocess.STDOUT, cwd="/", start_new_session=True)
        try:
            rc = p.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            rc = None
        finally:
            try:  # the launcher's ranks die with it; anything they started dies with its process group
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
    left = construct.gpus_free(float(c["release_wait_s"]))
    if left:
        print(f"[l0] the GPUs are still held {c['release_wait_s']} s after the construction node: {left}", flush=True)
    try:
        rec = json.loads((records / "construct.json").read_text())
    except (OSError, ValueError) as exc:
        pytest.fail(f"verifier_fault: the construction program wrote no record (rc {rc}): {exc}", pytrace=False)
    print("[l0] construction: " + json.dumps({k: rec.get(k) for k in ("status", "phase", "seconds", "layout")}),
          flush=True)
    if rc is None:
        raise OperationCapExceeded(f"the construction exceeded its {limit} s limit, in phase {rec.get('phase')}")
    if rec.get("status") == "pass":
        assert rc == 0, f"the construction program exited {rc} after its record read pass"
        return
    ranks = rec.get("ranks") or []
    first = next((r for r in ranks if r.get("status") == "fail"), None)
    if first is None:
        err = rec.get("error") or {}
        why = f"{rec.get('phase')}: {err.get('type', f'ended (rc {rc}) without its result')}: {err.get('message', '')}"
    else:
        err = first.get("error") or {}
        why = f"rank {first.get('rank')}, {first.get('phase')}: {err.get('type')}: {err.get('message', '')}"
    raise AssertionError("the registered class did not construct and load at the recipe's layout: " + _quoted(why))
