#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""The verifier's worker extension: the supplied worker's RPCs plus the private correctness, native-metadata, binding
and lifecycle RPCs.

The private serve wrapper (`tests/infb_verify/serve.py`) and the reference launch (`config.json`
`infb.efficiency.reference.pythonpath`) put this directory first on the server's PYTHONPATH, through
the evaluation tree's own launch seam `launch.environment(kda_dir=...)`, and the trusted copy of the
supplied `kda/` directly after it. The recipe's own `--worker-extension-cls worker.KDAWorker` then
resolves here, while every supplied primitive (`layer_adapter`, `mathematics`, `recurrence`)
resolves from the trusted copy. The supplied worker is loaded from that copy by path, under its own
name, and extended; no candidate code or configuration takes part.
"""

import importlib.util
import json
import os
from pathlib import Path

import torch

TRUSTED = Path(__file__).resolve().parents[1] / 'eval' / 'kimi_linear' / 'kda'


def _supplied(name):
    spec = importlib.util.spec_from_file_location('infb_kda_supplied_' + name, TRUSTED / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPPLIED = _supplied('worker')

import correctness  # noqa: E402 - this directory: the private schedule on the supplied primitives
import metadata  # noqa: E402 - this directory: the native-metadata schedules on the supplied primitives


class KDAWorker(SUPPLIED.KDAWorker):
    @torch.inference_mode()
    def infb_kda_correctness(self, seed):
        return correctness.run(self, int(seed), SUPPLIED)

    @torch.inference_mode()
    def infb_kda_metadata(self, families='all', out=''):
        """The native-metadata schedules (specification.json `metadata`); `out` receives this worker's record."""
        return metadata.run(self, 0, SUPPLIED, families, out or None)

    def infb_kda_binding(self, command='report', control='{}'):
        binding = self.model_runner.infb_kda_binding
        if command == 'control':
            binding.set_control(**json.loads(control))
        elif command != 'report':
            raise ValueError(command)
        return {'rank': self.rank, **binding.report()}

    def infb_kda_lifecycle(self, command='report', payload='{}'):
        lifecycle = self.model_runner.infb_kda_lifecycle
        return {'rank': self.rank, **lifecycle.command(command, json.loads(payload))}

    def infb_kda_timing(self, *args, **kwargs):
        if os.environ.get('INFB_KDA_BINDING') == '1' or os.environ.get('INFB_KDA_LIFECYCLE') == '1':
            raise AssertionError('timing requires a separate uninstrumented server')
        return super().infb_kda_timing(*args, **kwargs)


if os.environ.get('INFB_KDA_BINDING') == '1':
    from binding import install_startup_hook
    install_startup_hook()

if os.environ.get('INFB_KDA_LIFECYCLE') == '1':
    import lifecycle
    lifecycle.install_startup_hook()
