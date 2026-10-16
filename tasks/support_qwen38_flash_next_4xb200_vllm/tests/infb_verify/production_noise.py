#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private sensitivity policy over complete, unchanged public observations.

Compare a token score only with noise for that token at the same logical
history. Require the witness in both predeclared probes; a single surprising
replay is insufficient. The caller supplies inputs, amplitudes and bounds.
"""
from __future__ import annotations

from infb_verify.qwen38_private import production as native
from infb_verify.qwen38_private.production import validate_series

BASELINES = ('normal', 'repeat', 'qsa-restored', 'ple-restored')


def matched_effect(baselines, target, bars):
    anchor = baselines[0]
    ids = [row['id'] for row in anchor['cases']]
    count = anchor['max_tokens']
    # Validation includes greedy consistency: a non-argmax observation is structural.
    controls = [validate_series(series, ids, count) for series in baselines]
    candidate = validate_series(target, ids, count)
    noise_rows = [{row['id']: row for row in native.aligned_difference(anchor, series, ids, count)}
                  for series in baselines]
    effect_rows = [{row['id']: row for row in native.aligned_difference(series, target, ids, count)}
                   for series in baselines]
    if any(not row['near_tie_holds'] for rows in noise_rows for row in rows.values()):
        raise ValueError('unchanged production replays violate mutual top-five agreement')
    rows = []
    for cid in ids:
        # Include the first divergence: that score still has identical history.
        positions = min([effect_rows[0][cid]['aligned_positions']] +
                        [row[cid]['aligned_positions'] for row in noise_rows])
        coordinates = []
        for position in range(positions):
            maps = [{int(k): v for k, v in control[cid]['top_logprobs'][position].items()} for control in controls]
            scores = {int(k): v for k, v in candidate[cid]['top_logprobs'][position].items()}
            shared = set(scores)
            for mapping in maps:
                shared.intersection_update(mapping)
            for token in sorted(shared):
                values = [mapping[token] for mapping in maps]
                noise = max(values) - min(values)
                effect = min(abs(scores[token] - value) for value in values)
                threshold = max(bars['minimum_score_change'], bars['noise_multiple'] * noise)
                coordinates.append({'position': position, 'token': token,
                                    'noise': noise, 'effect': effect, 'threshold': threshold,
                                    'sensitive': effect > threshold})
        rows.append({'id': cid, 'aligned_positions': positions,
                     'outside_all_replays': all(not row[cid]['near_tie_holds'] for row in effect_rows),
                     'coordinates': coordinates})
    return rows


def paired_evidence(records, bars):
    """Assess two fixed probes, including repeated zero probes used as controls.

Eight uninfluenced series provide the noise envelope: normal, repeat, and both
restorations from each probe. Both interventions must affect the same score
coordinate, or both must break mutual top five for the same case against every
replay. Missing, malformed, stale or non-restoring observations fail closed.
"""
    if len(records) != 2 or any(record.get('status') != 'complete' for record in records):
        raise ValueError('production sensitivity requires two complete probes')
    for field in ('source', 'protocol', 'recipe', 'owners'):
        if not records[0].get(field) or records[0][field] != records[1].get(field):
            raise ValueError(f'production probe identity mismatch: {field}')
    baselines = [record['phases'][name]['series'] for record in records for name in BASELINES]
    result = {'schema': 'qwen38-paired-production-sensitivity/1',
              'noise_replays': len(baselines), 'bounds': bars, 'components': {}}
    for component in ('qsa', 'ple'):
        evidence = [matched_effect(baselines, record['phases'][component + '-intervened']['series'], bars)
                    for record in records]
        witnesses = []
        for left, right in zip(*evidence):
            assert left['id'] == right['id']
            common = {(row['position'], row['token']) for row in left['coordinates'] if row['sensitive']} & {
                (row['position'], row['token']) for row in right['coordinates'] if row['sensitive']}
            outside = left['outside_all_replays'] and right['outside_all_replays']
            if outside or common:
                witnesses.append({'id': left['id'], 'outside_all_replays': outside,
                                  'score_coordinates': sorted(common)})
        result['components'][component] = {'sensitive': bool(witnesses),
                                           'witnesses': witnesses, 'probes': evidence}
    return result
