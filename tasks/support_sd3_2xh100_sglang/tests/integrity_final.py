#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'Tail recheck of trusted source hashes and executable candidate source.'
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _common import GateResult, RESULTS_DIR, write_gate
import verify_anti_cheat as A
import verify_no_hf_evaluated_path as H


def main():
    for module in (A,H):
        path = RESULTS_DIR / (module.GATE + '.gate.json')
        try:
            prior = json.loads(path.read_text())
        except (OSError,ValueError):
            prior = None
        try:
            result = module.run()
        except Exception as exc:
            result = GateResult(gate=module.GATE,passed=False,
                                error=f'final integrity check incomplete: {type(exc).__name__}: {exc}'[:500],
                                details={'not_evaluated':True})
        result.details['final_integrity_checked'] = True
        if not isinstance(prior,dict) or prior.get('passed') is not True:
            result.passed, result.score = False, 0.0
            result.details['prior_check'] = prior
            result.error = 'earlier integrity check failed/missing: ' + str((prior or {}).get('error'))[:500]
        write_gate(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
