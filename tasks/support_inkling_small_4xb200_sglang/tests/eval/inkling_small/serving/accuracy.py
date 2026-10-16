#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""GSM8K through the registered evaluation, against a running server (README, `run.py registered` runs the same
evaluation inside the registered production class).

    python accuracy.py --base-url http://127.0.0.1:30000 [--data-path <jsonl>] [--output /tmp/gsm8k.json]

Upstream's evaluator, `sglang.test.few_shot_gsm8k.run_eval`, with the registered class's arguments: 10 shots,
200 questions, 512 new tokens, parallel 128. The default data file is upstream's public GSM8K test file, shipped at
`../fixtures/gsm8k_test.jsonl` (upstream's evaluator would otherwise download it to /tmp/test.jsonl). Grading runs the
same evaluator unchanged on a hidden 200-row slice; its floor is not printed here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
PUBLIC_ROWS = HERE.parent / "fixtures" / "gsm8k_test.jsonl"
REGISTERED = {"num_shots": 10, "num_questions": 200, "max_new_tokens": 512, "parallel": 128}


def run(base_url: str, data_path: Path | None) -> dict:
    from sglang.test.few_shot_gsm8k import run_eval
    url = urlparse(base_url)
    t0 = time.monotonic()
    metrics = run_eval(SimpleNamespace(data_path=str(data_path) if data_path else None, host=f"http://{url.hostname}",
                                       port=int(url.port), **REGISTERED))
    return {"data_path": str(data_path) if data_path else None, "arguments": REGISTERED,
            "wall_s": round(time.monotonic() - t0, 1),
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--data-path", type=Path, default=PUBLIC_ROWS)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    rec = run(args.base_url, args.data_path)
    print(json.dumps(rec))
    if args.output:
        args.output.write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
