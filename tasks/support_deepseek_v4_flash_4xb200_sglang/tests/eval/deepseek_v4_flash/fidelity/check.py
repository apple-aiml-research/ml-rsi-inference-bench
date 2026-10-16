#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Compare a running server with recorded public examples."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _fidelity():
    name = "deepseek_v4_flash_eval_fidelity"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "fidelity.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="log-probability fidelity of a running server (measured, no pass mark)")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--records", default=str(HERE / "examples" / "short"), help="a directory of records")
    ap.add_argument("--leave-out", type=int, default=8)
    args = ap.parse_args(argv)
    fidelity = _fidelity()
    records = fidelity.load_reference(Path(args.records))
    by_length: dict[int, list[float]] = {}
    invalid = 0
    for record in records:
        try:
            a = fidelity.rescore(args.base_url, record, args.leave_out)
        except (fidelity.StructuralFailure, fidelity.ServedRequestFailed) as exc:
            invalid += 1
            print(json.dumps({"id": record["id"], "tokens": len(record["input_ids"]), "valid": False,
                              "error": str(exc)}, sort_keys=True), flush=True)
            continue
        by_length.setdefault(a["tokens"], []).append(a["k3"])
        print(json.dumps({"id": record["id"], "tokens": a["tokens"], "valid": True, "k3": a["k3"],
                          f"k3_leaving_out_{args.leave_out}": a["k3_leaving_out"],
                          "positions": a["positions"], "expected_positions": a["expected_positions"],
                          "elapsed_s": a["elapsed_s"], "compared_with": record.get("reference")},
                         sort_keys=True), flush=True)
    for n, values in sorted(by_length.items()):
        print(f"{n} tokens: median k3 {fidelity.median(values):.6f} over {len(values)} valid record(s)")
    print(f"{len(records) - invalid} of {len(records)} answers structurally valid")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
