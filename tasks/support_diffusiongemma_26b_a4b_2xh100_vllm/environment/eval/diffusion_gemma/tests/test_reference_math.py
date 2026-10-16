#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Self-tests of the public reference statement (attend/reference.py), CPU only.

    python -m pytest /app/eval/diffusion_gemma/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
torch = pytest.importorskip("torch")

from eval.diffusion_gemma.attend.reference import visible_mask  # noqa: E402

W = 1024


def test_encode_is_causal_within_the_window():
    m = visible_mask("encode", 3000, 1000, [], "sliding", "w1")
    q = 500  # row 500 sits at absolute position 2,500
    row = m[q]
    assert bool(row[2500]) and not bool(row[2501])
    assert bool(row[2500 - W + 1]) and not bool(row[2500 - W])


def test_denoise_canvas_is_bidirectional_and_shares_one_window():
    n, mm = 3000, 256
    c = n - mm
    for reading, left in (("w1", c - (W - 1)), ("w", c - W)):
        m = visible_mask("denoise", n, mm, [], "sliding", reading)
        for row in (0, 128, 255):
            assert bool(m[row, n - 1]) and bool(m[row, c])
            assert bool(m[row, left]) and not bool(m[row, left - 1])


def test_full_layers_see_all_committed_positions():
    m = visible_mask("denoise", 5000, 256, [], "full", "w1")
    assert bool(m[0, 0]) and bool(m[255, 4999])


def test_image_span_is_bidirectional_within_itself():
    n, mm = 1300, 1300
    m = visible_mask("encode", n, mm, [[40, 320]], "sliding", "w1")
    assert bool(m[40, 319]) and bool(m[319, 40])
    assert not bool(m[39, 40])
