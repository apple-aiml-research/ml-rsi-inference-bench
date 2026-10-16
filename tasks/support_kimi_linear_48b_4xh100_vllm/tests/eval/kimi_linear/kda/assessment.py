#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Interpretation of KDA measurement records under the published numerical policy. No engine imports.

Reads the records `client.py` writes for `--mode sample` and `--mode timing`. Every error statistic
is checked against `numerical-policy.json`. A report keeps three things apart:

* `evidence_complete`: the record is a complete, well-formed measurement (every rank, case, step,
  statistic and required check present);
* `sample_correctness` (sample records) or `numerical_checks` (timing records): `pass` or `fail`
  under the policy, or `unavailable` when the evidence is incomplete;
* `task_acceptance`: always `not_determined`. Correct samples do not establish model support, and
  no speed bar is supplied, so a timing record reports measurements, never an efficiency verdict.

The command exits 0 when the evidence is complete and every comparison meets the policy, 1 when a
comparison fails it, and 2 when the evidence is incomplete or unusable.
"""

import json
import math
from pathlib import Path
import statistics

POLICY_PATH = Path(__file__).with_name('numerical-policy.json')
#: The complete-layer timing families, in protocol order (protocol.json `timing.families`).
WORKLOADS = [('prefill', (n,)) for n in (128, 512, 2048)]
WORKLOADS += [('prefill_packed', (2048,) * 4)]
WORKLOADS += [('decode', (1,) * n) for n in (1, 3, 16, 64)]


def policy():
    return json.loads(POLICY_PATH.read_text())


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def kind_of(path):
    return 'state' if 'state' in path else ('attention' if '.attention' in path else 'output')


def numerical_problems(value, bars, path='result'):
    """Every error statistic in `value` checked against `bars`; required boolean evidence must be true.

    A statistic passes when its relative RMSE is within its kind's bar or its maximum absolute error is
    within the absolute escape; a missing or non-finite statistic never passes."""
    problems = []
    if isinstance(value, dict):
        if 'relative_rmse' in value or 'max_abs' in value:
            kind = kind_of(path)
            absolute, relative = value.get('max_abs'), value.get('relative_rmse')
            if not all(finite_number(x) and x >= 0 for x in (absolute, relative)):
                problems.append(f'{path}: missing/nonfinite error statistic')
            elif absolute > bars[kind]['absolute_escape'] and relative > bars[kind]['relative_rmse']:
                problems.append(f'{path}: relative RMSE {relative} exceeds {bars[kind]["relative_rmse"]}')
        for key, item in value.items():
            if key in ('untouched_exact', 'attention_observed', 'outputs_finite') and item is not True:
                problems.append(f'{path}.{key}: required evidence is false')
            problems.extend(numerical_problems(item, bars, f'{path}.{key}'))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            problems.extend(numerical_problems(item, bars, f'{path}[{i}]'))
    return problems


def ranks(records):
    if not isinstance(records, list) or sorted(r.get('rank', -1) for r in records) != [0, 1, 2, 3]:
        raise ValueError('evidence must contain every TP rank exactly once')
    return sorted(records, key=lambda r: r['rank'])


def error_statistics(value, path='result'):
    """Every error statistic in a record, by kind, with its location; no threshold applied."""
    rows = []
    if isinstance(value, dict):
        if 'relative_rmse' in value or 'max_abs' in value:
            rows.append({'path': path, 'kind': kind_of(path), 'relative_rmse': value.get('relative_rmse'),
                         'max_abs': value.get('max_abs')})
        for key, item in value.items():
            rows.extend(error_statistics(item, f'{path}.{key}'))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            rows.extend(error_statistics(item, f'{path}[{i}]'))
    return rows


def evidence_problems(value, path='result'):
    """Locally decidable failures: missing/non-finite statistics and required evidence that is false."""
    problems = []
    if isinstance(value, dict):
        if 'relative_rmse' in value or 'max_abs' in value:
            if not all(finite_number(value.get(x)) and value.get(x) >= 0 for x in ('max_abs', 'relative_rmse')):
                problems.append(f'{path}: missing/nonfinite error statistic')
        for key, item in value.items():
            if key in ('untouched_exact', 'attention_observed', 'outputs_finite') and item is not True:
                problems.append(f'{path}.{key}: required evidence is false')
            problems.extend(evidence_problems(item, f'{path}.{key}'))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            problems.extend(evidence_problems(item, f'{path}[{i}]'))
    return problems


def worst_errors(rows):
    """The largest relative RMSE of each kind, and where it was measured."""
    worst = {}
    for row in rows:
        value = row.get('relative_rmse')
        if finite_number(value) and (row['kind'] not in worst or value > worst[row['kind']]['relative_rmse']):
            worst[row['kind']] = row
    return worst


def expected_workloads(axis=None, family=None):
    if family is not None and (type(family) is not int or family not in range(8)):
        raise ValueError(f'unknown timing family: {family}')
    expected = set(WORKLOADS) if family is None else {WORKLOADS[family]}
    if axis is not None:
        if axis not in ('prefill', 'decode'):
            raise ValueError(f'unknown timing axis: {axis}')
        expected = {key for key in expected if key[0].startswith(axis)}
        if not expected:
            raise ValueError(f'timing family {family} is outside axis {axis}')
    return expected


def _timed(record, axis, family):
    return [f for f in record.get('families', [])
            if (axis is None or f['family'].startswith(axis))
            and (family is None or (f['family'], tuple(f['lengths'])) == WORKLOADS[family])]


def timing_table(records, axis=None, family=None, side='record'):
    """Per family: each draw's time on the slowest TP rank and the median over draws; incompleteness as problems.

    A rank may return from its RPC earlier than its peers; synchronized collective work still waits for the
    slowest rank, so each draw is that critical path before any median is taken."""
    expected = expected_workloads(axis, family)
    problems, by_rank = [], []
    for record in ranks(records):
        families = _timed(record, axis, family)
        keyed = {(f['family'], tuple(f['lengths'])): f for f in families}
        if set(keyed) != expected or len(families) != len(expected):
            problems.append(f'{side} rank{record["rank"]}: incomplete timing families')
        by_rank.append(keyed)
        for workload in families:
            for when in ('before_checks', 'after_checks'):
                checks = workload.get(when, [])
                if len(checks) != 2 or any('output' not in c or len(c.get('states', [])) != 4 for c in checks):
                    problems.append(f'{side} rank{record["rank"]} {workload["family"]}: missing {when}')
    keys = set(by_rank[0])
    if any(set(row) != keys for row in by_rank):
        raise ValueError(f'{side}: ranks timed different work')
    times = {}
    for key in keys:
        lengths = {len(row[key]['draws']) for row in by_rank}
        if len(lengths) != 1 or min(lengths) < 3:
            raise ValueError(f'{side} {key}: incomplete paired draws')
        draw_times = []
        for draw in range(next(iter(lengths))):
            values = [row[key]['draws'][draw]['wall_s'] for row in by_rank]
            if not all(finite_number(t) and t > 0 for t in values):
                raise ValueError(f'{side} {key}: invalid timing value')
            draw_times.append(max(values))
        times[key] = {'critical_path_draws_s': draw_times, 'median_s': statistics.median(draw_times)}
    return times, problems


def timing_checks(records, bars, axis=None, family=None, side='record'):
    """The before/after numerical checks of one record's timed families, under `bars`."""
    problems = []
    for record in ranks(records):
        problems.extend(numerical_problems(_timed(record, axis, family), bars, f'{side}.rank{record["rank"]}'))
    return problems


def sample_problems(records):
    """Completeness of a loaded-layer sample record: every rank, the protocol's cases, every step's comparisons."""
    protocol = json.loads(Path(__file__).with_name('protocol.json').read_text())['sample']['cases']
    problems = []
    for record in ranks(records):
        label = f'rank{record["rank"]}'
        cases = record.get('cases', [])
        if [c.get('name') for c in cases] != [c['name'] for c in protocol]:
            problems.append(f'{label}: the sample cases are {[c.get("name") for c in cases]}')
            continue
        if len(record.get('layers', [])) != 20:
            problems.append(f'{label}: incomplete loaded implementation identities')
        for case, spec in zip(cases, protocol):
            measurements = case.get('measurements', [])
            if len(measurements) != spec['steps']:
                problems.append(f'{label}.{spec["name"]}: {len(measurements)} steps, not {spec["steps"]}')
            for m in measurements:
                if 'output' not in m or len(m.get('states', [])) != 4 or 'attention' not in m:
                    problems.append(f'{label}.{spec["name"]}: missing output, attention or four states')
    return problems


def _report(kind, record_problems, numerical, extra):
    evidence = not record_problems
    verdict_key = 'sample_correctness' if kind == 'sample' else 'numerical_checks'
    report = {'mode': kind, 'evidence_complete': evidence, 'evidence_problems': record_problems,
              verdict_key: 'unavailable' if not evidence else ('fail' if numerical else 'pass'),
              'numerical_problems': numerical if evidence else [], 'task_acceptance': 'not_determined', **extra}
    if not evidence:
        report['verdict'] = 'evidence incomplete: no numerical verdict; full task acceptance is not determined'
    elif kind == 'sample':
        report['verdict'] = (('sample fails the numerical policy' if numerical else
                              'sample meets the numerical policy') + '; full task acceptance is not determined')
    else:
        report['verdict'] = (('speed measured; numerical checks fail the numerical policy' if numerical else
                              'speed measured; numerical checks meet the numerical policy')
                             + '; no efficiency verdict is determined here; full task acceptance is not determined')
    return report


def sample_report(records):
    """A loaded-layer sample record under the published policy."""
    try:
        problems = sample_problems(records) + evidence_problems(records)
    except (ValueError, KeyError, TypeError) as exc:
        problems = [f'unusable evidence: {exc}']
    numerical = [] if problems else numerical_problems(records, policy())
    return _report('sample', problems, numerical, {'worst': worst_errors(error_statistics(records))})


def timing_report(candidate, reference=None, axis=None, family=None):
    """Per-family slowest-rank medians and the record's numerical checks under the policy.

    With `reference`, another complete record of the same families on the same host, each family's
    ratio (its median / this record's median) is reported; no bar is applied to it."""
    extra = {}
    try:
        times, problems = timing_table(candidate, axis=axis, family=family)
        problems += [p for p in evidence_problems(candidate) if 'nonfinite' in p]
        rows = [{'family': key[0], 'lengths': list(key[1]), **value} for key, value in sorted(times.items())]
        if reference is not None:
            other, other_problems = timing_table(reference, axis=axis, family=family, side='comparison')
            problems += other_problems
            if set(other) != set(times):
                raise ValueError('the two records timed different work')
            for row in rows:
                row['comparison_median_s'] = other[(row['family'], tuple(row['lengths']))]['median_s']
                row['ratio'] = row['comparison_median_s'] / row['median_s']
        extra['families'] = rows
        numerical = [] if problems else timing_checks(candidate, policy(), axis=axis, family=family)
    except (ValueError, KeyError, TypeError) as exc:
        problems, numerical = [f'unusable evidence: {exc}'], []
    extra['worst'] = worst_errors(error_statistics(candidate))
    return _report('timing', problems, numerical, extra)


def exit_code(report):
    if not report['evidence_complete']:
        return 2
    return 1 if report['numerical_problems'] else 0


def main(argv=None):
    """Report on sample and timing records under the published policy, without importing the engine."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--mode', choices=['sample', 'timing'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path,
                        help='timing: another complete record of the same families on the same host to compare with')
    parser.add_argument('--axis', choices=['prefill', 'decode'])
    parser.add_argument('--family', type=int, choices=range(8), help='report one published family only')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        record = json.loads(args.input.read_text())
        if not record.get('completed'):
            raise ValueError('input measurement is incomplete')
        comparison = None
        if args.mode == 'timing' and args.reference is not None:
            comparison = json.loads(args.reference.read_text())
            if not comparison.get('completed'):
                raise ValueError('comparison measurement is incomplete')
            comparison = comparison['result']
        if args.mode == 'sample':
            report = sample_report(record['result'])
        else:
            report = timing_report(record['result'], comparison, axis=args.axis, family=args.family)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = _report(args.mode, [f'unusable evidence: {exc}'], [], {})
    body = json.dumps(report, indent=2) + '\n'
    print(body, end='')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
    return exit_code(report)


if __name__ == '__main__':
    raise SystemExit(main())
