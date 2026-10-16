#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail-closed image clauses from the trusted independent-reference scorer.

The verifier supplies metrics from reference_contract.image_check, never a
candidate declaration. All four frozen per-case bars must hold on every role.
An identical pair of candidate images does not establish correctness.
"""
from __future__ import annotations

import math

METRICS = ('psnr', 'ssim', 'clip_img', 'clip_txt')


def independent_clauses(checks, roles, profile):
    clauses, reasons = {}, []
    for role in roles:
        name = 'independent_' + role
        row = checks.get(role) if isinstance(checks, dict) else None
        if not isinstance(row, dict) or row.get('profile') != profile:
            clauses[name] = None
            reasons.append(f'{name}: missing independent {profile} reference reading')
            continue
        if row.get('verdict') == 'fail':
            clauses[name] = False
            reasons.append(f'{name}: ' + str(row.get('reason', 'independent image check failed')))
            continue
        metrics, bars = row.get('metrics'), row.get('bars')
        valid = (row.get('verdict') == 'pass' and isinstance(metrics, dict) and isinstance(bars, dict)
                 and all(type(d.get(k)) in (int, float) and math.isfinite(d[k])
                         for d in (metrics, bars) for k in METRICS))
        if not valid:
            clauses[name] = None
            reasons.append(f'{name}: incomplete/nonfinite independent image metrics or bars')
            continue
        below = [k for k in METRICS if metrics[k] < bars[k]]
        clauses[name] = not below
        if below:
            reasons.append(f'{name}: below fixed reference bars: ' + ', '.join(below))
    return clauses, reasons
