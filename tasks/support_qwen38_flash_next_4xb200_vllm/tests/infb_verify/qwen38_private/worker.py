#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Grading worker extensions: the public measurement protocol with independent expected outputs."""
from __future__ import annotations

import os

from qwen38.measurement.worker import MeasurementWorker

from . import comparator, expected as expected_outputs


class GradingWorker(MeasurementWorker):
    """Public capture, replay and timing; expected outputs from the private comparator."""

    def expected(self, result, capture, case):
        ckpt = self.__dict__.get("_qwen38_checkpoint")
        if ckpt is None:
            ckpt = self.__dict__["_qwen38_checkpoint"] = comparator.Checkpoint(
                os.environ.get("QWEN38_MODEL_DIR", "/model"))
        return expected_outputs.expected(result, capture, case, ckpt)


from .instrument import NativeObservationWorker  # noqa: E402
from .lifecycle import LifecycleWorker  # noqa: E402


class DiagnosticWorker(GradingWorker, LifecycleWorker, NativeObservationWorker):
    """The retained MTP3 production/lifecycle observers on a separate diagnostic launch."""
