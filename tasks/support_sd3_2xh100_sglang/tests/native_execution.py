#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Trusted per-request native row work; returned trajectories are diagnostics.

Records come from hooks on the pipeline-held DiT. Counts alone never prove
consumption: the verifier also runs an unchanged repeat and an armed request.
"""
from collections import Counter
import math


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def observed_request(summary, request, tolerance=0):
    summary = summary or {}
    witness = summary.get('native_execution') or {}
    recorder = summary.get('recorder') or {}
    workers = set(recorder.get('worker_pids') or [])
    t0, t1 = request.get('t0'), request.get('t1')
    missing, failures = [], []
    if not witness.get('installed') or not workers or not recorder.get('evidence_complete'):
        missing.append('native row recorder absent/incomplete in a served worker')
    if not finite(t0) or not finite(t1) or t1 < t0:
        missing.append('request time window missing or invalid')
        events = []
    else:
        events = [e for e in witness.get('forwards', []) if e.get('pid') in workers
                  and finite(e.get('t')) and t0 <= e['t'] <= t1]
        if any(e.get('pid') in workers and finite(e.get('t')) and t0 <= e['t'] <= t1
               for e in witness.get('errors', [])):
            missing.append('native row recorder failed inside this request')
    work = Counter()
    cfg, outputs, expected = request.get('cfg'), request.get('num_outputs', 1), request.get('steps')
    valid_request = finite(cfg) and type(outputs) is int and outputs > 0 and type(expected) is int and expected > 0
    if not valid_request:
        missing.append('requested CFG/output/step parameters absent or invalid')
    branches = 2 if finite(cfg) and cfg > 1 else 1
    for e in events:
        installed = any(i.get('pid') == e['pid'] and i.get('pre_hook') is True
                        and finite(i.get('t')) and i['t'] <= e['t']
                        for i in witness.get('installations', []))
        if not installed:
            missing.append('forward lacks same-worker row-hook installation receipt')
        rows, ts = e.get('rows'), e.get('timesteps') or []
        if (not valid_request or type(rows) is not int or rows not in {outputs, outputs * branches}
                or len(ts) != rows or not ts or any(not finite(t) for t in ts) or len(set(ts)) != 1):
            failures.append('invalid native timestep/CFG row work')
            continue
        if e.get('completed') is not True:
            failures.append('served DiT forward did not complete')
            continue


        if any(len(e.get(field) or []) != rows or any(not isinstance(v, dict) or not v.get('fp')
                   for v in e.get(field) or []) for field in ('x', 'y')):
            missing.append('native input/output row fingerprints incomplete')
            continue
        work[ts[0]] += rows
    if not events:
        missing.append('no trusted native forwards in request window')
    if valid_request and events and not missing and (not expected - tolerance <= len(work) <= expected + tolerance
            or any(rows != outputs * branches for rows in work.values())):
        failures.append('native timesteps/CFG row work differ from the requested denoising sequence')
    if not request.get('image'):
        failures.append('request produced no image')
    return {'ok': not missing and not failures, 'not_evaluated': bool(missing) and not failures,
            'reason': '; '.join(dict.fromkeys(failures + missing)), 'observed_steps': len(work),
            'forwards': len(events), 'branches': branches, 'expected_steps': expected,
            'trajectory_len_diagnostic': request.get('trajectory_len'),
            'perturbed_forwards': sum(e.get('native_perturbed') is True for e in events)}


def consumption_verdict(repeat, armed, repeat_equal, armed_equal):
    missing, failures = [], []
    for name, reading in (('repeat', repeat), ('armed', armed)):
        if not reading.get('ok'):
            (missing if reading.get('not_evaluated') else failures).append(name + ': ' + reading.get('reason', 'missing'))
    if repeat_equal is None or armed_equal is None:
        missing.append('paired image readback incomplete')
    elif repeat_equal is not True:
        missing.append('unchanged repeat not reproducible within the existing isolation bar; intervention cannot be attributed')
    elif armed_equal is True and repeat.get('ok') and armed.get('ok') and armed.get('perturbed_forwards'):
        failures.append('the served DiT output intervention did not reach the image')
    if not armed.get('perturbed_forwards'):
        missing.append('output intervention never fired on the served DiT')
    return {'ok': not missing and not failures, 'not_evaluated': bool(missing) and not failures,
            'reason': '; '.join(dict.fromkeys(failures + missing)), 'repeat': repeat, 'armed': armed,
            'repeat_equal': repeat_equal, 'armed_equal': armed_equal}
