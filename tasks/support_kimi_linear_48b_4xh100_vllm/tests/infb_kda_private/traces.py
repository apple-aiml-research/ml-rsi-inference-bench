#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Inspect complete release-era vLLM profiler traces, independently of engine imports.

These traces corroborate the intervention and device-counter evidence. A kernel name alone
does not prove that its result is consumed by serving.
"""

import gzip
import hashlib
import json
from pathlib import Path
import re


def inspect_traces(directory):
    records = []
    for path in sorted(Path(directory).glob('*.pt.trace.json.gz')):
        rank = re.search(r'-rank-(\d+)\.', path.name)
        if rank is None:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        # A truncated gzip stream or JSON document is an error, never partial evidence.
        with gzip.open(path, 'rt') as source:
            trace = json.load(source)
        events = trace['traceEvents']
        markers = set()
        kernels = graphs = 0
        for event in events:
            name = event.get('name', '')
            marker = re.fullmatch(r'infb\.kda\.native\.layer\.(\d+)\.mode\.(\d+)', name)
            if marker:
                markers.add(tuple(map(int, marker.groups())))
            kernels += event.get('cat') == 'kernel'
            graphs += name.startswith(('cudaGraphLaunch', 'cuGraphLaunch'))
        records.append({'rank': int(rank.group(1)), 'path': str(path),
                        'bytes': path.stat().st_size, 'sha256': digest.hexdigest(),
                        'events': len(events), 'kernels': kernels, 'graph_launches': graphs,
                        'native_markers': [list(x) for x in sorted(markers)]})
    return records


def trace_problems(records):
    problems = []
    if sorted(row['rank'] for row in records) != [0, 1, 2, 3]:
        return ['one complete profiler trace from every TP rank is required']
    for row in records:
        label = f'trace.rank{row["rank"]}'
        if row['kernels'] <= 0 or row['graph_launches'] <= 0:
            problems.append(f'{label}: missing GPU kernels or graph replay')
        have = set(map(tuple, row['native_markers']))
        if not {(layer, mode) for layer in range(20) for mode in (0, 2)}.issubset(have):
            problems.append(f'{label}: missing native prefill or mixed execution markers')
    return problems
