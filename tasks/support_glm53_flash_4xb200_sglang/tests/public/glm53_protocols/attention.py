#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The attention sublayer's numerical policy: how a complete attention sublayer's output is compared with the model
author's computation of the same sublayer on the same input history (`attention-numerics.md`)."""
from __future__ import annotations

import math

EPS = 1e-8
ABS_ESCAPE = 1e-6
ALPHA = math.sqrt(2.0)
E_FLOOR = 2.0 ** -8


def _f64(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu", dtype=torch.float64).numpy()
    except ImportError:  # pragma: no cover - NumPy inputs need no PyTorch
        pass
    import numpy as np
    return np.asarray(x, dtype=np.float64)


def relative_rms_error(candidate, author) -> float:
    import numpy as np
    c, a = _f64(candidate), _f64(author)
    if c.shape != a.shape:
        raise ValueError(f"shape {c.shape} against {a.shape}")
    if not np.isfinite(c).all():
        raise ValueError("the candidate holds a value that is not finite")
    return float(np.sqrt(np.mean((c - a) ** 2)) / (np.sqrt(np.mean(a ** 2)) + EPS))


def max_abs_difference(candidate, author) -> float:
    import numpy as np
    c, a = _f64(candidate), _f64(author)
    if c.shape != a.shape:
        raise ValueError(f"shape {c.shape} against {a.shape}")
    return float(np.max(np.abs(c - a)))


def allowance(e_rep: float) -> float:
    e_rep = float(e_rep)
    if not (math.isfinite(e_rep) and e_rep >= 0.0):
        raise ValueError(f"the representation's effect must be a finite non-negative number, not {e_rep}")
    return ALPHA * max(e_rep, E_FLOOR)


def within_policy(e: float, max_abs: float, e_rep: float) -> bool:
    return e <= allowance(e_rep) or max_abs <= ABS_ESCAPE
