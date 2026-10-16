#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Run the registered test's classes verbatim on the tree at /app (README, `run.py registered`).

    python deterministic.py --launch deterministic|production [--output /tmp/det.json] [--port 21000]

The registered test and its helper (`../tests/test_inkling_small_nvfp4.py`, `../tests/kl_test_utils.py`, upstream's
files at the registering commit) are staged into a work directory outside /app; the helper is imported as
`sglang.test.kl_test_utils`, as the registered test imports it. Before the class runs:
  * the public LongBench v2 draw (`../examples/longbench_public.json`) is placed at the helper's own cache path
    (`<helper dir>/.longbench_cache/input_ids__model_thinkingmachines_Inkling-Small-NVFP4_3000_48.json`), so the
    deterministic helpers fetch nothing;
  * upstream's public GSM8K test file (`../fixtures/gsm8k_test.jsonl`) is placed at /tmp/test.jsonl, where upstream's
    evaluator looks before downloading;
  * `INKLING_SMALL_TEST_MODEL_PATH=/model/thinkingmachines/Inkling-Small-NVFP4`, the test's own override;
  * the class's launch timeout (`DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH`) is set to the public startup limit
    (`../limits.json`, `startup`) before the module is imported, as grading does.
`--launch deterministic` runs the three helpers of `TestInklingSmallNvfp4Deterministic`; `--launch production` runs
`TestInklingSmallNvfp4.test_gsm8k`. Each test's outcome, message and printed metrics are recorded. Grading runs the
classes on hidden data; this prints measurements, not the hidden verdict.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
MODEL = "/model/thinkingmachines/Inkling-Small-NVFP4"
DET_TESTS = ("test_input_output_logprobs_match", "test_input_output_logprobs_match_prefill_cache_hit",
             "test_input_output_logprobs_match_decode_cache_hit")


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)


def stage(work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    for name in ("test_inkling_small_nvfp4.py", "kl_test_utils.py"):
        shutil.copyfile(EVAL / "tests" / name, work / name)
    cache = work / ".longbench_cache"
    cache.mkdir(exist_ok=True)
    safe = MODEL.replace("/", "_")
    shutil.copyfile(EVAL / "examples" / "longbench_public.json", cache / f"input_ids_{safe}_3000_48.json")
    gsm8k = EVAL / "fixtures" / "gsm8k_test.jsonl"
    if gsm8k.is_file() and not Path("/tmp/test.jsonl").exists():
        shutil.copyfile(gsm8k, "/tmp/test.jsonl")
    return work


def load_registered(work: Path, startup_limit: float, port: int | None):
    os.environ.setdefault("INKLING_SMALL_TEST_MODEL_PATH", MODEL)
    import sglang.test.test_utils as tu
    tu.DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = startup_limit
    spec = importlib.util.spec_from_file_location("sglang.test.kl_test_utils", work / "kl_test_utils.py")
    helper = importlib.util.module_from_spec(spec)
    sys.modules["sglang.test.kl_test_utils"] = helper
    spec.loader.exec_module(helper)
    spec = importlib.util.spec_from_file_location("test_inkling_small_nvfp4", work / "test_inkling_small_nvfp4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if port is not None:
        mod.DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{port}"
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launch", choices=("deterministic", "production"), required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/inkling_small_registered"))
    ap.add_argument("--port", type=int, help="serve on this port instead of the class's default test port")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    limits = json.loads((EVAL / "limits.json").read_text())["operations"]
    work = stage(args.work_dir)
    mod = load_registered(work, float(limits["startup"]["limit_s"]), args.port)
    if args.launch == "deterministic":
        cls, names = mod.TestInklingSmallNvfp4Deterministic, DET_TESTS
    else:
        cls, names = mod.TestInklingSmallNvfp4, ("test_gsm8k",)
    log_path = work / f"{args.launch}.log"
    tee = Tee(sys.__stdout__, open(log_path, "w"))
    old, sys.stdout = sys.stdout, tee
    t0 = time.monotonic()
    try:
        result = unittest.TextTestRunner(stream=tee, verbosity=2).run(unittest.TestSuite([cls(n) for n in names]))
    finally:
        sys.stdout = old
    failed = {t.id().rsplit(".", 1)[-1]: m for t, m in result.failures}
    errored = {t.id().rsplit(".", 1)[-1]: m for t, m in result.errors}
    text = log_path.read_text(errors="replace")
    rec = {"launch": args.launch, "class": cls.__name__, "wall_s": round(time.monotonic() - t0, 1), "log": str(log_path),
           "tests": {n: {"status": "fail" if n in failed else "error" if n in errored else "pass",
                         "message": (failed.get(n) or errored.get(n) or "")[-2000:]} for n in names},
           "avg_kl_div_printed_in_order": re.findall(r"avg_kl_div=([0-9eE+\-.naninf]+)", text),
           "gsm8k_printed": re.findall(r"gsm8k: ([0-9.]+)", text),
           "setup_errors": [m[-2000:] for t, m in result.errors if "setUpClass" in t.id()]}
    print(json.dumps({k: v for k, v in rec.items() if k != "tests"}))
    if args.output:
        args.output.write_text(json.dumps(rec, indent=1))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
