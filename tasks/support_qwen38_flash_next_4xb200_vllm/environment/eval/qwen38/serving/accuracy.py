#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The upstream EvalScope measurement with explicit offline dataset resolution.

Generation settings and datasets are public inputs. Optional acceptance policy
is a separate caller input; an uncalibrated local measurement reports scores.
"""
from __future__ import annotations

import math
from pathlib import Path


def run_evalscope(eval_config, base_url, work_dir, *, dataset_root):
    names = list(eval_config['datasets'])
    root = Path(dataset_root).resolve()
    if (not names or any(not isinstance(name, str) or Path(name).name != name for name in names)
            or len(names) != len(set(names)) or any(not (root/name).is_dir() for name in names)):
        raise ValueError('every declared dataset must exist locally')
    from evalscope.run import run_task

    api_url = base_url.rstrip('/')
    if not api_url.endswith('/v1'):
        api_url += '/v1'
    reports = run_task({
        'model':eval_config['model_name'],'api_url':api_url,
        'api_key':'EMPTY_TOKEN','datasets':names,
        'dataset_args':{name:{'dataset_id':str(root/name)} for name in names},
        'eval_batch_size':eval_config.get('eval_batch_size',32),
        'generation_config':eval_config['generation_config'],
        'work_dir':str(work_dir),'no_timestamp':True,
    })
    if not isinstance(reports,dict) or set(reports)!=set(names):
        raise ValueError('EvalScope returned an incomplete dataset inventory')
    scores = {name:float(reports[name].score) for name in names}
    if any(not math.isfinite(value) or not 0<=value<=1 for value in scores.values()):
        raise ValueError('EvalScope returned an invalid accuracy score')
    return scores


def assess(scores, policy):
    if set(scores)!=set(policy):
        raise ValueError('accuracy policy must cover exactly the measured datasets')
    if not scores or any(type(value) not in (int, float) or not math.isfinite(value)
                         or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError('invalid accuracy score')
    for minimum in policy.values():
        if type(minimum) not in (int,float) or not math.isfinite(minimum) or not 0<=minimum<=1:
            raise ValueError('invalid accuracy minimum')
    return {'scores':scores,'minimums':policy,
            'passed':all(scores[name]>=policy[name] for name in scores)}
