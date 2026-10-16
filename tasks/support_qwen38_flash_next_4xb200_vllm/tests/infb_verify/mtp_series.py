#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private inputs for the source-identical public MTP observation protocol."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
for _path in (str(TESTS_DIR / "eval"), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from qwen38.measurement import observations  # noqa: E402
from qwen38.serving import mtp  # noqa: E402
from infb_verify.qwen38_private.stateful import vocabulary  # noqa: E402

SERIES_FILE = Path(os.environ.get("INFB_MTP_SERIES", "/logs/verifier/mtp/base-series.json"))
ObservationError = observations.ObservationError


def prompts() -> list[tuple[str, str]]:
    out = []
    for name in ("prefix_reuse.json", "prefix_witness.json"):
        data = json.loads((TESTS_DIR / "hidden" / name).read_text())
        out.extend((case["id"], case["prefix"]) for case in data["cases"])
    return out


def ids() -> list[str]:
    return [cid for cid, _ in prompts()]


def identity() -> dict:
    return {"evaluation_id": os.environ["INFB_EVALUATION_ID"],
            "source": mtp.source_identity(), "tokenizer": mtp.tokenizer_identity()}


def series(base: str, n: int, *, recipe: str = "ordinary") -> dict:
    return mtp.collect_series(base, n, prompts(), recipe=recipe, vocabulary=vocabulary(), **identity())


def validate(series: dict) -> dict:
    """Every case complete, valid and greedy-consistent (a non-argmax observation is structural)."""
    return observations.validate_series(series, ids(), series["max_tokens"], vocabulary=vocabulary())


def compare(a: dict, b: dict) -> list[dict]:
    return observations.compare(a, b, ids(), a["max_tokens"], vocabulary=vocabulary())


def acceptance(base: str, n: int) -> dict:
    return mtp.acceptance(base, n, prompts(), vocabulary=vocabulary(), **identity())


def record(name: str, data: dict) -> None:
    path = SERIES_FILE.parent / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False))
