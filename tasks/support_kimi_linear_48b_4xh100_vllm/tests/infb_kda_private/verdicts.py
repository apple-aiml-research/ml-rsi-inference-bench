#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""The verifier's KDA verdicts: completeness of the private schedule, paired timing, binding. No engine imports.

Moved from the evaluation tree's kda/assessment.py with unchanged semantics. The shared numerical interpretation, rank handling and per-family timing table are the
trusted copy of the supplied assessment.py; this module adds what only the grader decides: the complete
correctness schedule and GDN inventory, the paired ratio against the reference with its floor, source
identity across launches and the binding interventions' effect.
"""

import importlib.util
from pathlib import Path

TRUSTED = Path(__file__).resolve().parents[1] / 'eval' / 'kimi_linear' / 'kda'
_SPEC = importlib.util.spec_from_file_location('infb_kda_supplied_assessment', TRUSTED / 'assessment.py')
supplied = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(supplied)

finite_number = supplied.finite_number
numerical_problems = supplied.numerical_problems
ranks = supplied.ranks


def identity_problems(left, right):
    """Bind independently launched math/timing/evidence servers to the same loaded code.

    Allocation pool size may differ after profiling reserves memory. Per-slot state
    geometry, semantic entry and loaded source must agree on each TP rank.
    """
    def signature(layer):
        return (layer.get('layer'), layer.get('entry'), layer.get('source_sha256'),
                layer.get('module_source', {}).get('sha256'), layer.get('callable_module'),
                [shape[1:] for shape in layer.get('state_shapes', [])],
                layer.get('state_strides'))

    problems = []
    for first, second in zip(ranks(left), ranks(right)):
        a, b = first.get('layers', []), second.get('layers', [])
        if len(a) != 20 or len(b) != 20:
            problems.append(f'rank{first["rank"]}: incomplete loaded implementation identities')
        elif [signature(layer) for layer in a] != [signature(layer) for layer in b]:
            problems.append(f'rank{first["rank"]}: different code or state layout was measured and served')
    return problems


def _specification():
    import json
    return json.loads(Path(__file__).with_name('specification.json').read_text())


def correctness_problems(records, bars):
    """The retained mathematics schedule, by its trusted manifest: every rank ran exactly the manifest's cases
    (identity, order, multiplicity, rows, history, steps and layer) and the three splits, completely."""
    spec = _specification()['mathematics']
    manifest = spec['manifest']
    splits = ['split-' + '-'.join(map(str, segments)) for segments in spec['split_prefill']]
    problems = []
    for record in ranks(records):
        label = f'rank{record["rank"]}'
        cases = record.get('cases', [])
        if [c.get('id') for c in cases] != [m['id'] for m in manifest]:
            problems.append(f'{label}: the mathematical cases are not the manifest\'s '
                            f'({len(cases)} of {len(manifest)}: {[c.get("id") for c in cases]})')
        if [c.get('id') for c in record.get('split_cases', [])] != splits:
            problems.append(f'{label}: the split cases are not the specification\'s')
        for case, expected in zip(cases, manifest):
            name = f'{label}.{expected["id"]}'
            if any(case.get(key) != expected[key] for key in ('lengths', 'computed', 'valid', 'steps')) \
                    or case.get('layer_index') != expected['layer']:
                problems.append(f'{name}: rows, history, steps or layer differ from the manifest')
            measurements = case.get('measurements', [])
            if len(measurements) != expected['steps']:
                problems.append(f'{name}: {len(measurements)} of {expected["steps"]} steps')
            for measurement in measurements:
                if 'output' not in measurement or len(measurement.get('states', [])) != 4:
                    problems.append(f'{name}: missing output or four states')
        problems.extend(numerical_problems(record, bars, label))
    return problems


def _schedules():
    spec = importlib.util.spec_from_file_location('infb_kda_private_schedules', Path(__file__).with_name('schedules.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _without_attention(value):
    """The record without its core-attention readings: the native-metadata families judge the four states and
    the complete-layer outputs; their attention is reported, not judged."""
    if isinstance(value, dict):
        return {k: _without_attention(v) for k, v in value.items() if k not in ('attention', 'attention_observed')}
    if isinstance(value, list):
        return [_without_attention(v) for v in value]
    return value


def metadata_attention_diagnostics(records, bars):
    """The new families' core-attention readings against the retained node's bar: reported, never judged."""
    bar = bars['attention']
    rows = []
    for record in records:
        identity = record.get('identity', {})
        for case in record.get('cases', []):
            for j, call in enumerate(case.get('calls', [])):
                reading = call.get('attention')
                if reading:
                    rows.append({'dp': identity.get('dp_rank'), 'tp': identity.get('tp_rank'), 'id': case.get('id'),
                                 'call': j, 'lengths': call.get('lengths'), **reading})
    above = [r for r in rows if r['max_abs'] > bar['absolute_escape'] and r['relative_rmse'] > bar['relative_rmse']]
    return {'judged': False, 'ruling': 'reported, never judged', 'readings': len(rows),
            'above_retained_bar': above,
            'max_relative_rmse': max((r['relative_rmse'] for r in rows), default=None)}


def metadata_problems(records, bars, deployment):
    """The native-metadata schedules of one deployment, from the per-worker records.

    `tp4`: every family on TP ranks 0..3; `dp2`: the replay family on each replica, (dp, tp) ranks (0..1, 0..3).
    Every worker ran exactly the scheduled cases in order, completely, with the four states and the complete-layer
    outputs under the numerical policy (core attention is diagnostic here); every replay used its captured
    graph on unchanged builder storage, some graph extent exceeded the builder's own padded extent, and the
    zero-live step touched only the null block."""
    schedules = _schedules()
    expected_workers = ({(0, t) for t in range(4)} if deployment == 'tp4'
                        else {(d, t) for d in range(2) for t in range(4)})
    families = None if deployment == 'tp4' else {'replay'}
    cases = [c for c in schedules.schedule() if families is None or c[0] in families]
    problems = []
    seen = {}
    for record in records:
        identity = record.get('identity', {})
        key = (identity.get('dp_rank') if deployment != 'tp4' else 0, identity.get('tp_rank'))
        seen.setdefault(key, []).append(identity.get('device_uuid'))
    if set(seen) != expected_workers or any(len(v) != 1 for v in seen.values()):
        problems.append(f'{deployment}: worker records {sorted(seen)} are not exactly {sorted(expected_workers)}')
    uuids = [u for v in seen.values() for u in v]
    if None in uuids or len(set(uuids)) != len(uuids):
        problems.append(f'{deployment}: worker device identities are missing or repeated')
    for record in records:
        identity = record.get('identity', {})
        label = f'{deployment}.dp{identity.get("dp_rank")}.tp{identity.get("tp_rank")}'
        ordinary = [c for c in cases if c[0] != 'replay']
        got = [(c.get('family'), c.get('id'), c.get('layer_index')) for c in record.get('cases', [])]
        if got != [(f, name, layer) for f, name, _, layer, _ in ordinary]:
            problems.append(f'{label}: the component cases are not the schedule ({len(got)} of {len(ordinary)})')
        for case, (family, name, _, _, params) in zip(record.get('cases', []), ordinary):
            calls = schedules.calls_expected(family, params)
            if calls is not None:
                measured = case.get('calls', [])
                if len(measured) != calls or any('output' not in m or len(m.get('states', [])) != 4
                                                 for m in measured):
                    problems.append(f'{label}.{name}: incomplete calls')
            else:
                split = case.get('split', {})
                if (len(split.get('segment_calls', [])) != len(params['segments'])
                        or len(split.get('final_states', [])) != 4):
                    problems.append(f'{label}.{name}: incomplete checkpoint split')
        replays = [c for c in cases if c[0] == 'replay']
        replay = record.get('replay', {})
        populations = replay.get('populations', [])
        if [p.get('id') for p in populations] != [name for _, name, _, _, _ in replays]:
            problems.append(f'{label}: the replay populations are not the schedule')
        captured = {}
        for event in replay.get('captures', []):
            captured[(event.get('layer_index'), event.get('extent'))] = event.get('builder_storage')
        wider = False
        for population, (_, name, _, layer, params) in zip(populations, replays):
            extent = schedules.capture_extent(params['population'])
            storage = captured.get((layer, extent))
            if population.get('layer_index') != layer or population.get('graph_extent') != extent or storage is None:
                problems.append(f'{label}.{name}: not replayed on the captured {extent}-row graph of layer {layer}')
            elif population.get('builder_storage') != storage:
                problems.append(f'{label}.{name}: builder staging storage moved after capture')
            if params['population'] == 0:
                if population.get('dummy') is not True:
                    problems.append(f'{label}.{name}: the zero-live step did not take the dummy path')
            else:
                if 'output' not in population or len(population.get('states', [])) != 4:
                    problems.append(f'{label}.{name}: missing output or four states')
                if population.get('graph_extent', 0) > population.get('builder_extent', 0) > 0:
                    wider = True
        if replays and not wider:
            problems.append(f'{label}: no replay extent exceeded the builder\'s own padded extent')
        problems.extend(numerical_problems(_without_attention({'cases': record.get('cases', []), 'replay': populations}),
                                           bars, label))
    if deployment == 'tp4':
        layers = {c[3] for c in cases if c[0] in ('replay', 'checkpoint_split')}
        if layers != set(range(20)):
            problems.append('replay and split cases do not cover every layer')
    return problems


def gdn_problems(record, bars):
    cases = record.get('cases', [])
    problems = []
    if not record.get('completed') or len(cases) != 55:
        problems.append('incomplete GDN case set')
    if {(c.get('batch'), c.get('step')) for c in cases if c.get('family') == 'decode_continuation'} != {
            (batch, step) for batch in (1, 3, 16, 64) for step in range(8)}:
        problems.append('incomplete GDN continuation cases')
    if any('output' not in case or 'state' not in case for case in cases):
        problems.append('missing GDN output or state comparison')
    return problems + numerical_problems(cases, bars, 'gdn')


def timing_ratios(reference, candidate, floor, bars, axis=None, family=None):
    """Every workload family must pass independently; slowest TP rank decides each draw.

    The per-family table (critical path per draw, median over draws, completeness) and the numerical
    checks are the supplied assessment's; the paired ratio and its floor are the grader's."""
    for records in (reference, candidate):
        ranks(records)  # every TP rank exactly once on both sides, before anything is read
    problems, tables = [], []
    for side, records in (('reference', reference), ('candidate', candidate)):
        times, incomplete = supplied.timing_table(records, axis=axis, family=family, side=side)
        problems.extend(incomplete)
        problems.extend(supplied.timing_checks(records, bars, axis=axis, family=family, side=side))
        tables.append(times)
    if set(tables[0]) != set(tables[1]):
        raise ValueError('reference and candidate timed different work')
    ratios = []
    for key in sorted(tables[0]):
        ref, cand = (table[key] for table in tables)
        ratio = ref['median_s'] / cand['median_s']
        ratios.append({'family': key[0], 'lengths': list(key[1]), 'ratio': ratio,
                       'reference': ref, 'candidate': cand, 'pass': ratio >= floor})
        if ratio < floor:
            problems.append(f'{key}: paired ratio {ratio:.6f} below {floor}')
    return {'families': ratios, 'problems': problems}


def completion_distance(left, right, first_only=False):
    a, b = [v['choices'][0]['logprobs'] for v in (left, right)]
    ta, tb = a['tokens'], b['tokens']
    pa, pb = a['token_logprobs'], b['token_logprobs']
    if len(ta) != len(tb) or not ta:
        raise ValueError('incomplete matched completion')
    pairs = list(zip(ta, tb, pa, pb))[:1] if first_only else list(zip(ta, tb, pa, pb))
    if not all(finite_number(x) and finite_number(y) for _, _, x, y in pairs):
        raise ValueError('nonfinite serving logprob')
    return {'token_changes': sum(a != b for a, b, _, _ in pairs),
            'max_logprob_delta': max(abs(x - y) for _, _, x, y in pairs)}


def binding_problems(record, noise_multiplier, absolute_floor):
    problems = []
    for rank in ranks(record.get('execution')):
        label = f'rank{rank["rank"]}'
        stats, layers = rank.get('stats', []), rank.get('layers', [])
        if len(stats) != 20 or len(layers) != 20:
            problems.append(f'{label}: missing native layer evidence')
            continue
        for i, counts in enumerate(stats):
            if len(counts) != 3:
                problems.append(f'{label}.layer{i}: missing execution modes')
                continue
            for mode, row in enumerate(counts):
                if len(row) != 6 or not all(finite_number(x) for x in row) or row[0] <= 0 or row[5] <= 0:
                    problems.append(f'{label}.layer{i}.mode{mode}: no executed state update')
            eager = sum(e['layer'] == i and e['mode'] == 1 for e in rank.get('events', []))
            if counts[1][0] <= eager:
                problems.append(f'{label}.layer{i}: graph replay not observed')
            identity = layers[i]
            if not identity.get('source_sha256') and not identity.get('module_source', {}).get('sha256'):
                problems.append(f'{label}.layer{i}: loaded code identity unavailable')
    baseline = record.get('baseline', [])
    if len(baseline) != 2:
        raise ValueError('two baseline repetitions required')
    natural = completion_distance(*baseline)
    if natural['token_changes']:
        problems.append('binding baseline itself changed token sequence; intervention inconclusive')
    bar = max(absolute_floor, noise_multiplier * natural['max_logprob_delta'])
    interventions = record.get('interventions', [])
    wanted = {(i, kind) for i in range(20) for kind in ('output', 'state')}
    if {(v['layer'], v['kind']) for v in interventions} != wanted or len(interventions) != 40:
        problems.append('incomplete output/state interventions')
    for value in interventions:
        label = f'layer{value["layer"]}.{value["kind"]}'
        delta = completion_distance(baseline[0], value['completion'])
        if not delta['token_changes'] and delta['max_logprob_delta'] <= bar:
            if value['kind'] == 'output':
                # Name both causes: the message cannot distinguish "serving ignores this layer's output"
                # from "the declared output is a per-call buffer the consume entry consumed before returning".
                problems.append(f'{label}: scaling the declared consume output after the consume entry '
                                f"returned did not change serving; either serving does not consume this "
                                f'layer\'s output, or the declared output is a buffer the consume entry '
                                f'consumed before returning (README, Alternative native layouts)')
            else:
                problems.append(f'{label}: changing mapped data did not affect serving')
        if value['kind'] == 'state':
            first = completion_distance(baseline[0], value['completion'], first_only=True)
            if first['token_changes'] or first['max_logprob_delta'] > bar:
                problems.append(f'{label}: post-call state intervention altered the first output')
    return {'problems': problems, 'baseline_noise': natural, 'intervention_bar': bar}
