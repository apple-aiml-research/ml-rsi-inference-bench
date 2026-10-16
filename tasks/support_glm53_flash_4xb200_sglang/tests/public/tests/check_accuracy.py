#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Model quality under the maintainers' GSM8K evaluation, against a running server.

    python3 /app/eval/glm53_flash/tests/check_accuracy.py --class high_throughput --url http://127.0.0.1:30000

It places the public GSM8K test set where upstream's evaluator reads it (`/tmp/test.jsonl`, from
`tests/data/gsm8k/test.jsonl`; the episode is offline) and runs the acceptance file's own `test_gsm8k` for the class,
unchanged: upstream's `run_eval` with its settings (the completion API, 20 shots, 500 questions, greedy) and its own
threshold. Exit 0 when that test passes.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from acceptance import CLASSES, MODEL, acceptance_class, bound, run_methods  # noqa: E402

DATA = HERE / "data" / "gsm8k" / "test.jsonl"
READ_FROM = Path("/tmp/test.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--class", dest="cls", choices=sorted(CLASSES), required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    if not (READ_FROM.is_file() and filecmp.cmp(DATA, READ_FROM, shallow=False)):
        shutil.copyfile(DATA, READ_FROM)
    result = run_methods(bound(acceptance_class(a.cls), a.url, a.model), ["test_gsm8k"])
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
