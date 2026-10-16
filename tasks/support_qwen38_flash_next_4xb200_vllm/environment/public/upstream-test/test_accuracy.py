#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream GSM8K/AIME25 workload through the common offline measurement."""
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, '/app/eval')

from qwen38.serving.accuracy import assess, run_evalscope
from qwen38.serving.recipes import SERVER_ARGS
from qwen38.serving.server import Server as RemoteOpenAIServer


def test_qwen4_exp_accuracy(config_filename: Path, tmp_path: Path):
    eval_config = yaml.safe_load(config_filename.read_text(encoding='utf-8'))
    dataset_root = os.environ.get('INFB_EVALSCOPE_DATASET_ROOT', '/public/datasets')
    with RemoteOpenAIServer(
        eval_config['model_name'], [*SERVER_ARGS, '--seed', '0'],
        env_dict=eval_config.get('env'),
        max_wait_seconds=eval_config.get('startup_max_wait_seconds', 1800),
    ) as remote_server:
        scores = run_evalscope(eval_config, remote_server.url_for('v1'), tmp_path,
                               dataset_root=dataset_root)
    result = {'scores': scores}
    if 'acceptance_policy' in eval_config:
        result = assess(scores, eval_config['acceptance_policy'])
    (tmp_path / 'accuracy.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False))
    if 'passed' in result:
        assert result['passed'], result
