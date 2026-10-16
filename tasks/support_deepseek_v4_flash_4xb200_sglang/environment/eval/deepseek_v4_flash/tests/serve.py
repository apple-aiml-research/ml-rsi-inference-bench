#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Start one server exactly as the acceptance test launches it, and keep it serving, so the checks in this directory
and in `../fidelity/` and `../serving/` can run one after another against the same server, as the grader runs them."""
from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMITS = json.loads((HERE.parent / "limits.json").read_text())["operations"]
LAUNCH_ALLOWANCE_S = int(LIMITS["startup"]["limit_s"])


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def acceptance_class():
    """The acceptance test's class, with the adapter applied (the supplied sanity kit, the GSM8K fixture)."""
    load("deepseek_v4_flash_eval_conftest", HERE / "conftest.py")
    module = load("deepseek_v4_flash_eval_acceptance", HERE / "test_deepseek_v4_flash_fp4_b200.py")
    return module.TestDSV4FlashFP4B200


class LaunchTimeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise LaunchTimeout(f"the server did not become healthy within the graded {LAUNCH_ALLOWANCE_S} s allowance")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="start the server at the release recipe and keep it serving")
    ap.add_argument("--detach", action="store_true", help="exit once healthy, leaving the server running")
    args = ap.parse_args(argv)
    cls = acceptance_class()
    t0 = time.monotonic()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(LAUNCH_ALLOWANCE_S)
    try:
        cls.setUpClass()
    except LaunchTimeout as exc:
        print(f"serve: {exc}", flush=True)
        if getattr(cls, "process", None) is not None:
            cls.tearDownClass()
        return 3
    finally:
        signal.alarm(0)
    print(f"serve: healthy in {time.monotonic() - t0:.1f} s at {cls.base_url} (server pid {cls.process.pid})",
          flush=True)
    if args.detach:
        return 0
    try:
        while cls.process.poll() is None:
            time.sleep(5)
        print(f"serve: the server exited with {cls.process.returncode}", flush=True)
        return 1
    except KeyboardInterrupt:
        cls.tearDownClass()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
