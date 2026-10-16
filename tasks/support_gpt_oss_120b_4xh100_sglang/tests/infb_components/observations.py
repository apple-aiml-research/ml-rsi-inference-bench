#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Aggregate raw production observations with independent structural checks.

Numerical, storage and speed limits are supplied by private gates.
Recorded summaries and aggregate server acceptance counters are not trusted.
"""
from collections import Counter
import json
import math
from pathlib import Path



def finite(value):
    assert type(value) in (int,float) and math.isfinite(value)
    return value


def collect(root):
    ranks = []
    for rank in range(4):
        rows = [json.loads(line) for line in (Path(root)/f'rank-{rank}.jsonl').open()]
        assert rows and all(row['rank']==rank for row in rows)
        kinds=Counter(row['kind'] for row in rows)
        loaded=[r for r in rows if r['kind']=='loaded']
        assert len(loaded)==1
        loaded=loaded[0]
        assert {b['layer'] for b in loaded['bindings']}==set(range(36))
        assert len(loaded['bindings'])==36
        storage=loaded['expert_storage']['bytes']
        assert type(storage) is int and storage>=0
        model_storage=loaded['all_model_storage']['bytes']
        assert type(model_storage) is int and model_storage>=storage
        assert loaded['allocator_live_delta']>0 or loaded['device_used_delta']>0
        executions=[r for r in rows if r['kind']=='gpu_execution']
        assert executions and all(r['kernels'] and sum(r['kernels'].values())>0 for r in executions)
        invoked={v['layer'] for row in executions for v in row['invocations']}
        assert invoked==set(range(36)), 'not every bound layer executed in the target model'
        result={'rank':rank,'kinds':dict(kinds),'expert_storage_bytes':storage,
                'all_model_storage_bytes':model_storage,
                'allocator_live_delta':loaded['allocator_live_delta'],
                'allocator_reserved_delta':loaded['allocator_reserved_delta'],
                'device_used_delta':loaded['device_used_delta'],'bindings':loaded['bindings'],
                'residency_after_execution':[
                    {'expert_storage_bytes':r['expert_storage']['bytes'],
                     'all_model_storage_bytes':r['all_model_storage']['bytes'],
                     'allocator_live_delta':r['allocator_live_delta'],
                     'allocator_reserved_delta':r['allocator_reserved_delta'],
                     'device_used_delta':r['device_used_delta'],
                     'pool_storage_bytes':r['pool_storage_bytes'],
                     'mode':r['mode'],'graph':r['graph']}
                    for r in rows if r['kind']=='resident_after_execution'],
                'executions':[{k:v for k,v in r.items() if k in ('mode','tokens','batch_size','graph')}
                              for r in rows if r['kind']=='target_execution']}
        ranks.append(result)
    return ranks
