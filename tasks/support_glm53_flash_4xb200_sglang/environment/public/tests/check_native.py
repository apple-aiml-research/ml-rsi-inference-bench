#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The tree installs, and the engine resolves the checkpoint's declared architecture as a native SGLang model.

    python3 /app/eval/glm53_flash/tests/check_native.py [--skip-install]

It reinstalls `/app/python` editable and offline (no dependencies, no build isolation, no Rust rebuild), checks that
`import sglang` resolves inside `/app/python`, then parses the checkpoint's `config.json` through the engine's own
`ModelConfig` and resolves it with the engine's model registry. The resolved class must carry the checkpoint's own
`architectures[0]` and `model_type`, live in `sglang.srt.models`, and not be the Transformers fallback. Exit 0 when
every step holds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm53_protocols.native import (install_editable, native_problems, resolution_record,  # noqa: E402
                                    resolve_architecture, sglang_location)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--app", default="/app")
    ap.add_argument("--model", default="/model/zai-org/GLM-5.3-Flash")
    ap.add_argument("--skip-install", action="store_true", help="check the resolution only")
    a = ap.parse_args()
    py, failed = sys.executable, False
    if not a.skip_install:
        r = install_editable(Path(a.app), py)
        print(f"editable install: rc {r.returncode}")
        if r.returncode != 0:
            print(r.stdout[-4000:] + r.stderr[-4000:])
            failed = True
        r = sglang_location(py)
        where = r.stdout.strip()
        inside = r.returncode == 0 and str((Path(a.app) / "python").resolve()) in where
        print(f"import sglang: {where or r.stderr.strip()[-2000:]} ({'inside' if inside else 'NOT inside'} the tree)")
        failed |= not inside
    cfg = json.loads((Path(a.model) / "config.json").read_text())
    r = resolve_architecture(a.model, py)
    record = resolution_record(r.stdout) if r.returncode == 0 else None
    if record is None:
        print("the config did not parse or the architecture did not resolve:\n" + r.stdout[-3000:] + r.stderr[-8000:])
        return 1
    print("resolved: " + json.dumps(record))
    problems = native_problems(record, cfg["architectures"][0], cfg["model_type"])
    for p in problems:
        print("NOT NATIVE: " + p)
    return 1 if failed or problems else 0


if __name__ == "__main__":
    sys.exit(main())
