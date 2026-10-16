#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The MSA group's component worker: one process on one GPU that imports one implementation of the operator contract
and runs units -- private correctness cases, timing families, sparsity pairs or identity-set collection -- writing a
JSON line when each unit starts and when it ends, so the parent can enforce each unit's 1,800-second cap, attribute a
crash to its unit and resume after it.

  python -I msa_worker.py --contract /app/minimax_m3_msa.py --mode cases --node test_select_prefill_correctness \
      --start 0 --out <jsonl>
  ... --mode measure --units select_prefill,attend_prefill_bf16 --seed-file /tests/hidden/policy.json
  ... --mode sparsity --units decode,prefill --seed-file ...
  ... --mode poison --units decode_16k,decode_64k_bf16,... --seed-file ...   (sparsity.POISON's geometries)
  ... --mode identity --units select_prefill --seed-file ...

The trusted protocol copy (/tests/eval) is first on the path; the implementation under test is loaded from its file.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
for _path in (str(TESTS_DIR), str(TESTS_DIR / "eval")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

from minimax_m3.msa import binding, check, measure, sparsity  # noqa: E402


def emit(fh, **obj):
    fh.write(json.dumps(obj, default=str) + "\n")
    fh.flush()


def seed_of(path: str | None) -> int:
    if not path:
        return measure.PUBLIC_SEED
    return int(json.loads(Path(path).read_text())["msa"]["seed"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--mode", choices=("cases", "measure", "sparsity", "poison", "identity"), required=True)
    ap.add_argument("--node")
    ap.add_argument("--units", default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed-file")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fh = open(a.out, "a", encoding="utf-8")
    # The verifier's own setup first, before any code under test runs: a failure here is the verifier's.
    if a.mode == "cases":
        from infb_verify import msa_cases  # noqa: PLC0415 - private, from /tests
        units = msa_cases.case_ids()[a.node]
    else:
        units = [u for u in a.units.split(",") if u]
    seed = seed_of(a.seed_file)
    emit(fh, event="ready", pid=os.getpid(), device=torch.cuda.get_device_name(0), units=len(units), t=time.time())
    try:
        module = check.load_contract(a.contract)
    except Exception as exc:  # the implementation does not import: every unit fails, the first says why
        emit(fh, event="import_error", error=f"{type(exc).__name__}: {exc}"[:4000],
             traceback=traceback.format_exc()[-6000:])
        return 3
    for i in range(a.start, len(units)):
        unit = units[i]
        emit(fh, event="start", index=i, unit=unit, t=time.time())
        t0 = time.monotonic()
        try:
            if a.mode == "cases":
                record = check.run_case(module, msa_cases.make_case(unit))
            elif a.mode == "measure":
                record = measure.measure_family(module, unit, seed)
            elif a.mode == "sparsity":
                record = sparsity.measure_pair(module, unit, seed)
            elif a.mode == "poison":
                record = sparsity.poisoned_case(module, unit, seed)
            else:
                record = binding.component_identity_sets(module, seed, families={unit})[unit]
        except Exception as exc:  # outside run_case's own handling (e.g. case construction): the verifier's
            record = {"status": "verifier_fault", "error": f"{type(exc).__name__}: {exc}"[:4000],
                      "traceback": traceback.format_exc()[-6000:]}
        emit(fh, event="done", index=i, unit=unit, seconds=round(time.monotonic() - t0, 3), record=record)
        torch.cuda.empty_cache()
    emit(fh, event="end", t=time.time())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
