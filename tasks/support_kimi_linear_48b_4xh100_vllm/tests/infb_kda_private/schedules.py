#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""The native-metadata case schedule (specification.json `metadata`), shared by the worker-side runner
(`metadata.py`) and the verdicts. Standard library only."""

import hashlib
import json
from pathlib import Path
import random

SPEC = json.loads(Path(__file__).with_name('specification.json').read_text())['metadata']


def case_seed(seed, family, case):
    """A case's own seed: the component seed plus the first four bytes of SHA256 of its name."""
    digest = hashlib.sha256(f'kimi-linear-v55/component/{family}/{case}'.encode()).digest()
    return seed + int.from_bytes(digest[:4], 'little') % (1 << 20)


def nonadjacent_slots(rng, count):
    """`count` distinct odd slots (never adjacent) in a seeded order."""
    return rng.sample(range(1, 4 * count + 2, 2), count)


def capture_extent(population):
    """The captured graph a population replays: the smallest captured extent holding it."""
    return next(e for e in sorted(SPEC['replay']['capture']) if e >= max(population, 1))


def schedule(layers_count=20):
    """Every case of the specification, in execution order: (family, case id, seed, layer, parameters)."""
    cases = []
    for seed in SPEC['seeds']:
        for layer in range(layers_count):
            for v, vector in enumerate(SPEC['mixed_validity']):
                name = f'mixed-{v}-layer{layer}-seed{seed}'
                rng = random.Random(case_seed(seed, 'mixed', name))
                cases.append(('mixed_validity', name, case_seed(seed, 'mixed', name), layer,
                              {**vector, 'slots': nonadjacent_slots(rng, len(vector['lengths']))}))
            name = f'turnover-layer{layer}-seed{seed}'
            cases.append(('turnover', name, case_seed(seed, 'turnover', name), layer, SPEC['turnover']))
            name = f'branch-layer{layer}-seed{seed}'
            cases.append(('branch', name, case_seed(seed, 'branch', name), layer, SPEC['branch']))
    counter = 0
    for seed in SPEC['seeds']:
        for k, population in enumerate(SPEC['replay']['populations']):
            name = f'replay-{k}-p{population}-seed{seed}'
            cases.append(('replay', name, case_seed(seed, 'replay', name), counter % layers_count,
                          {'population': population, 'position': k}))
            counter += 1
    for seed in SPEC['seeds']:
        for segments in SPEC['checkpoint_split']:
            name = f'split-{"-".join(map(str, segments))}-seed{seed}'
            cases.append(('checkpoint_split', name, case_seed(seed, 'split', name), counter % layers_count,
                          {'segments': segments}))
            counter += 1
    return cases


def calls_expected(family, params):
    """How many controlled calls a case of `family` makes."""
    return {'mixed_validity': SPEC['mixed_validity_steps'], 'turnover': 3, 'branch': 5}.get(family)
