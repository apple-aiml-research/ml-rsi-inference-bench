#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The hidden class: the two scored efficiency nodes on this task's one recipe.

**Why this class launches its own server instead of mixing into the upstream one.** The kit's usual
shape is `class TestHiddenX(EfficiencyMixin, upstream.TestUpstreamClass)`, which works because the
upstream class launches the candidate in `setUpClass` and the `test_0_` nodes then sort ahead of the
upstream node and reuse that server. This task's registered class does not: `BaseTestGptOss` has no
`setUpClass`, and `popen_launch_server` is called inside `run_test`, i.e. inside the test method. A
mixin over it would run the efficiency nodes first, with no server and no `base_url`. So the launch is
done here, with the recipe's own argv, and `config.json` declares that argv statically -- the kit's
documented fallback for "a harness with no class to run".

**The guard that keeps the static argv honest.** A declared argv can drift from the registered test it
is meant to mirror, and then the efficiency nodes would measure a server the acceptance test never
runs. `setUpClass` therefore reads the registered module with `ast` -- no import, no execution -- pulls
the literal `other_args` of `test_mxfp4_120b` and the model id its variant maps to, and refuses to
launch unless every one of them appears in the declared argv, in order. The registered file is present
because `test.sh` restores the kit to base and re-applies `test.patch` before any of this runs.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # /tests, for infb_efficiency

from infb_efficiency.nodes import EfficiencyMixin  # noqa: E402

TESTS_DIR = pathlib.Path(os.environ.get("TESTS_DIR", "/tests"))
REGISTERED = pathlib.Path(os.environ.get("INFB_REGISTERED_TEST", "/app/test/srt/test_gpt_oss_4gpu.py"))
RECIPE = "TestHiddenAcceptance"

# The (variant, quantization) pairs this reader accepts, and the checkpoint each serves.
#
# ONE ENTRY, BECAUSE ONE PAIR IS EVER READ. `BaseTestGptOss.run_test` upstream knows four pairs, and
# this reader was written as a copy of that table. But the only method it is ever asked about is
# `test_mxfp4_120b` (the single call site below passes the default), and that method names
# ("120b", "mxfp4"). The other three entries named checkpoints this task never loads, never declares
# and does not ship -- `openai/gpt-oss-20b`, `lmsys/gpt-oss-120b-bf16`, `lmsys/gpt-oss-20b-bf16`.
# A released bundle has no way to obtain a checkpoint nothing declares, so an entry for one is a
# staging gap waiting to happen, and there is no 20b method at all. The reader now names only the
# model it loads.
#
# The assert below is unchanged, so a registered method that names any other pair still fails loudly
# -- it fails here, at the verifier's own guard, rather than at a server launch for a checkpoint that
# is not in the image.
REGISTERED_MODELS = {("120b", "mxfp4"): "openai/gpt-oss-120b"}


def _config() -> dict:
    return json.loads((TESTS_DIR / "config.json").read_text(encoding="utf-8"))["infb"]


def registered_launch_words(path: pathlib.Path, method: str = "test_mxfp4_120b") -> list[str]:
    """The literal launch words the registered test builds, read from source without importing it.

    `run_test(model_variant=..., quantization=..., other_args=[...])` -> the model id the
    (variant, quantization) pair maps to in `REGISTERED_MODELS`, followed by `other_args`.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method:
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                kw = {k.arg: k.value for k in call.keywords if k.arg}
                if "other_args" not in kw:
                    continue
                args = [ast.literal_eval(x) for x in kw["other_args"].elts]
                variant = ast.literal_eval(kw["model_variant"]) if "model_variant" in kw else None
                quant = ast.literal_eval(kw["quantization"]) if "quantization" in kw else None
                model = REGISTERED_MODELS.get((variant, quant))
                assert model, f"{method} names an unknown variant/quantization: {variant!r}/{quant!r}"
                return [model] + [str(a) for a in args]
    raise AssertionError(f"{path} has no {method} calling run_test with other_args")


def assert_argv_matches_registered(argv: list[str]) -> None:
    """Every launch word the registered test builds appears in the declared argv, in order."""
    want = registered_launch_words(REGISTERED)
    i = 0
    missing = []
    for w in want:
        try:
            i = argv.index(w, i) + 1
        except ValueError:
            missing.append(w)
    assert not missing, (
        f"config.json's declared launch argv has drifted from {REGISTERED}: it does not contain "
        f"{missing} in order. Declared: {argv}. The registered test builds: {want}.")


def split_launch(argv: list[str]) -> tuple[str, list[str]]:
    """(model, server flags) from the declared full command line.

    `config.json`'s `launch.argv` is a complete command line, because that is what the kit's reference
    phase runs (it appends `--host`/`--port` and executes it). `popen_launch_server`, which the
    registered test uses and this class reuses, takes the model separately and adds host and port
    itself -- so the launcher words, the model flag and any host/port are dropped here and everything
    else is passed through unchanged.
    """
    model = None
    flags: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--model-path", "--model"):
            model = argv[i + 1]
            i += 2
        elif a in ("--host", "--port"):
            i += 2
        elif a in ("python", "python3", "-m", "sglang.launch_server"):
            i += 1
        else:
            flags.append(a)
            i += 1
    assert model, f"the declared launch argv names no --model-path: {argv}"
    return model, flags


class TestHiddenAcceptance(EfficiencyMixin, unittest.TestCase):
    """One scored recipe at C = 64: S1 and S_C against the reference measured earlier in this run.

    `unittest.TestCase` is a base and not decoration: `EfficiencyMixin` calls `self.assertTrue`, and
    the kit's usual shape gets that from the upstream class it mixes over. This task has no upstream
    class to mix over (see the module docstring), so the base is named here. `unittest.TestCase` rather
    than SGLang's `CustomTestCase` on purpose -- that one retries a failed node, and a bar is never
    retried to green.
    """

    base_url = "http://127.0.0.1:8000"

    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils import kill_process_tree
        from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, popen_launch_server

        eff = _config()["efficiency"]
        argv = list((eff["recipes"][RECIPE]["launch"])["argv"])
        assert_argv_matches_registered(argv)
        model, other = split_launch(argv)
        cls._kill = kill_process_tree
        cls.process = popen_launch_server(model, cls.base_url,
                                          timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, other_args=other)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            cls._kill(cls.process.pid)


def load_tests(loader, tests, pattern):          # noqa: D401 - unittest hook
    """Only the two efficiency nodes run here; the mixin supplies both."""
    suite = loader.loadTestsFromTestCase(TestHiddenAcceptance)
    return suite
