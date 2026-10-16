#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""test_semantic_majority: the hidden cases through the public protocol
(eval/diffusion_gemma/serving/semantic.py). Expected on the reference: pass (a case is kept only if
the reference reliably hits >= 0.9 on it).
"""
from __future__ import annotations

from eval.diffusion_gemma.serving import semantic as S

from infb_verify import common

draws = S.draws


def run(srv, deployment: str) -> dict:
    return S.semantic_majority(srv, common.hidden("semantic_cases.json")[deployment])
