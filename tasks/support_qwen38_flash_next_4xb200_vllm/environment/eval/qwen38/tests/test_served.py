#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Served examples, against a running server: set QWEN38_BASE_URL (and QWEN38_RECIPE=mtp3 for the
MTP3 recipe). The components example needs the measurement worker extension on the server."""
import json
import os
import uuid

import pytest

from qwen38.measurement import observations as obs
from qwen38.serving import api, fidelity, recipes, reuse
from qwen38.serving.mtp import request

BASE = os.environ.get("QWEN38_BASE_URL")
RECIPE = os.environ.get("QWEN38_RECIPE", "ordinary")
EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
pytestmark = pytest.mark.skipif(not BASE, reason="QWEN38_BASE_URL names no running server")


def _vocabulary():
    return obs.vocabulary_size(os.environ.get("QWEN38_MODEL_DIR", "/model"))


def test_api_example():
    prompt = json.load(open(os.path.join(EXAMPLES, "prompts.json")))["cases"][0]["prompt"]
    result = api.probes(BASE, recipes.MODEL, prompt, vocabulary=_vocabulary())
    assert result["holds"], {k: result[k] for k in ("repeat", "stream", "length", "isolation")}


def test_fidelity_examples_are_complete():
    data = json.load(open(os.path.join(EXAMPLES, "fidelity/examples.json")))
    result = fidelity.evaluate(BASE, recipes.MODEL, data["sequences"], vocabulary=_vocabulary())
    assert len(result["sequences"]) == 5


@pytest.mark.skipif(RECIPE != "ordinary", reason="the prefix-reuse example is an ordinary-recipe protocol")
def test_prefix_reuse_example():
    data = json.load(open(os.path.join(EXAMPLES, "prefix_reuse.json")))
    ids = request(BASE, "/tokenize", {"model": recipes.MODEL, "prompt": data["prefix"]}, timeout=60)["tokens"]
    result = reuse.pair(BASE, recipes.MODEL, ids, tail=data["tail_tokens"], tokens=data["tokens"],
                        vocabulary=_vocabulary(), salt=f"qwen38-cold-{uuid.uuid4().hex}")
    assert result["warm_read_the_boundary"] and result["position_zero"]["terms"]
