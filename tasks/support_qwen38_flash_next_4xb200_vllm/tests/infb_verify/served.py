#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private inputs for the served-API observations both recipes share: API probes and fidelity.

The executable protocols are the public `qwen38.serving.api` and `qwen38.serving.fidelity`; this
module supplies the private prompt and teacher data, the calibrated band policy and the operation
allowances, and records every observation before a verdict.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
for _path in (str(TESTS_DIR / "eval"), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from qwen38.measurement import budget  # noqa: E402
from qwen38.serving import api  # noqa: E402
from qwen38.serving.fidelity import evaluate  # noqa: E402
from infb_verify.qwen38_private.stateful import vocabulary  # noqa: E402

X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
LOGS = Path(os.environ.get("INFB_SERVED_DIR", "/logs/verifier"))
PROMPT = "Explain in one short paragraph why a hash table has amortised constant lookup."


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False))


def api_probes(server, recipe: str) -> dict:
    """Repeat and streamed greedy agreement (near-tie rule at the first divergence), the length
    cap's termination and first-token isolation of eight concurrent requests; every record is
    validated structurally before any comparison."""
    with budget.operation("api_observation_batch"):
        result = api.probes(server.url_root, X["model_id"], PROMPT, vocabulary=vocabulary())
    write(LOGS / "api" / f"{recipe}.json", result)
    print("INFB_API_PROBES " + json.dumps({k: result[k] for k in ("repeat", "stream", "length", "isolation",
                                                                  "holds")}))
    return result


def fidelity(server, recipe: str) -> dict:
    """The independent 40-sequence teacher data under the fixed band policy (bounded rereads)."""
    data = json.loads((TESTS_DIR / "hidden/fidelity.json").read_text())
    bands = X["bars"]["fidelity_bands"]
    assert bands, "fidelity calibration is missing"
    with budget.operation("fidelity_evaluation"):
        result = evaluate(server.url_root, X["model_id"], data["sequences"], vocabulary=vocabulary(), bands=bands)
    result["reference"] = data["provenance"]
    result["recipe"] = recipe
    path = LOGS / "fidelity" / recipe / "observations.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as output:
        json.dump(result, output, allow_nan=False)
    report = {key: value for key, value in result.items() if key not in ("reads", "sequences")}
    print("INFB_FIDELITY " + json.dumps(report, allow_nan=False))
    return result
