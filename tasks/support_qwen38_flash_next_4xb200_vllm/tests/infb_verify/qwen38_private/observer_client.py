#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Clients of the retained MTP3 production observers (interventions and runner lifecycle)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid

from qwen38.measurement.worker import protocol_identity


def observation(base_url, *, recipe, component=None, owner=None, amplitude=0., reset=False):
    request_id = uuid.uuid4().hex
    protocol = protocol_identity()
    options = dict(request_id=request_id, protocol=protocol, component=component, owner=owner,
                   amplitude=amplitude, reset=reset)
    payload = {'method': 'qwen38_observation', 'timeout': 120, 'args': [json.dumps(options)]}
    request = urllib.request.Request(base_url.rstrip('/') + '/collective_rpc',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=180) as response:
        data = json.load(response)
    rows = data['results']
    if len(rows) != 4 or {row['rank'] for row in rows} != {0, 1, 2, 3}:
        raise ValueError('missing or duplicate native observation TP rank')
    if any(row.get('request_id') != request_id or row.get('protocol') != protocol for row in rows):
        raise ValueError('stale or mismatched native observation RPC')
    if any(row.get('status') != 'complete' for row in rows):
        raise ValueError(f'native observer failed: {rows}')
    sources = {row.get('source') for row in rows}
    if len(sources) != 1 or None in sources or any(row['recipe'] != recipe for row in rows):
        raise ValueError('native observation source or recipe mismatch')
    return {'schema': 'qwen38-native-observation/1', 'source': sources.pop(),
            'recipe': recipe, 'protocol': protocol, 'ranks': rows}


def lifecycle(base_url, *, action='read'):
    request_id = uuid.uuid4().hex
    options = {'request_id': request_id, 'action': action}
    payload = {'method': 'qwen38_lifecycle', 'timeout': 120, 'args': [json.dumps(options)]}
    request = urllib.request.Request(base_url.rstrip('/') + '/collective_rpc',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=180) as response:
        data = json.load(response)
    rows = data['results']
    if len(rows) != 4 or {row['rank'] for row in rows} != {0, 1, 2, 3}:
        raise ValueError('missing or duplicate lifecycle TP rank')
    if any(row.get('request_id') != request_id for row in rows):
        raise ValueError('stale lifecycle RPC response')
    return {'schema': 'qwen38-lifecycle-response/1', 'action': action, 'ranks': rows,
            'status': 'failed' if any(row['status'] != 'complete' or
                row.get('value', {}).get('failure_count', 0) for row in rows) else 'complete'}
