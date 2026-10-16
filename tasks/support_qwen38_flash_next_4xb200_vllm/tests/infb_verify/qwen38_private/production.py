#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Separate production interventions on native enclosing QSA and PLE outputs (MTP3).

These requests never enter performance draws. The server starts with the private
DiagnosticWorker and a native mapping (the owners the component record bound) before
compilation. Raw observations and noise/sensitivity summaries are returned without an
acceptance policy; `expanded.join` joins useful execution to the component identities.
"""
from __future__ import annotations

import json
from pathlib import Path

from qwen38.measurement import observations as obs
from qwen38.serving import mtp

from .observer_client import observation
from .stateful import vocabulary


def validate_series(series, ids, tokens):
    return obs.validate_series(series, ids, tokens, vocabulary=vocabulary())


def _int_keys(mapping):
    return {int(k): v for k, v in mapping.items()}


def aligned_difference(left, right, ids, tokens):
    """Compare probabilities only while both continuations have equal history."""
    a = validate_series(left, ids, tokens)
    b = validate_series(right, ids, tokens)
    for field in ('evaluation_id', 'source_identity', 'tokenizer_identity', 'workload_identity', 'recipe'):
        if not left.get(field) or left[field] != right.get(field):
            raise ValueError(f'production replay identity mismatch: {field}')
    rows = []
    for cid in ids:
        x, y = a[cid], b[cid]
        divergence = next((i for i in range(tokens) if x['tokens'][i] != y['tokens'][i]), None)
        count = tokens if divergence is None else divergence + 1
        differences = []
        for i in range(count):
            left, right = _int_keys(x['top_logprobs'][i]), _int_keys(y['top_logprobs'][i])
            differences.extend(abs(left[key] - right[key]) for key in left.keys() & right.keys())
        near_tie = divergence is None or (
            obs.among_top(x['tokens'][divergence], _int_keys(y['top_logprobs'][divergence])) and
            obs.among_top(y['tokens'][divergence], _int_keys(x['top_logprobs'][divergence])))
        rows.append({'id': cid, 'aligned_positions': count, 'first_divergence': divergence,
                     'near_tie_holds': near_tie, 'common_scores': len(differences),
                     'max_aligned_score_difference': max(differences, default=0.)})
    return rows


def collect(base_url, cases, *, recipe, output, evaluation_id, tokens=32, amplitude=.125, owners=None):
    """Record unchanged/repeated/intervened/restored complete continuations."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ids = [cid for cid, _ in cases]
    report = {'schema': 'qwen38-production-observation/1', 'recipe': recipe,
              'amplitude': amplitude, 'phases': {}, 'status': 'running'}
    def save(): output.write_text(json.dumps(report, indent=1, allow_nan=False))
    initial = observation(base_url, recipe=recipe, reset=True)
    if owners is None:
        owners = {component: next(site['identity']['owner'] for site in initial['ranks'][0]['sites']
                                  if site['identity']['component'] == component)
                  for component in ('qsa','ple')}
    if set(owners) != {'qsa','ple'} or any(not isinstance(value,str) for value in owners.values()):
        raise ValueError('production intervention needs one actual owner for each component')
    identity = dict(evaluation_id=evaluation_id, source=initial['source'], tokenizer=mtp.tokenizer_identity())
    report.update(source=initial['source'], protocol=initial['protocol'], owners=owners)
    try:
        for phase, component, amount in (
            ('normal', None, 0.), ('repeat', None, 0.),
            ('qsa-intervened', 'qsa', amplitude), ('qsa-restored', None, 0.),
            ('ple-intervened', 'ple', amplitude), ('ple-restored', None, 0.)):
            owner = owners.get(component)
            observation(base_url, recipe=recipe, component=component, owner=owner, amplitude=amount, reset=True)
            # Structural and greedy validity of every observation (non-argmax outputs are structural).
            series = mtp.collect_series(base_url, tokens, cases, recipe=recipe, vocabulary=vocabulary(), **identity)
            validate_series(series, ids, tokens)
            ranks = observation(base_url, recipe=recipe, component=component, owner=owner, amplitude=amount)
            if ranks['source'] != report['source'] or ranks['protocol'] != report['protocol']:
                raise ValueError('production source or protocol changed during replay')
            for rank in ranks['ranks']:
                rank['sites'] = [site for site in rank['sites'] if site['device_calls'] > 0]
            report['phases'][phase] = {'series': series, 'observation': ranks}
            if phase != 'normal':
                report['phases'][phase]['comparison'] = aligned_difference(
                    report['phases']['normal']['series'], series, ids, tokens)
            save()
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        # Disarm even after a malformed draw, without hiding the original error.
        try: report['restoration'] = observation(base_url, recipe=recipe, reset=True)
        except Exception as exc: report.update(status='failed', restoration_error=repr(exc))
        save()
    return report
