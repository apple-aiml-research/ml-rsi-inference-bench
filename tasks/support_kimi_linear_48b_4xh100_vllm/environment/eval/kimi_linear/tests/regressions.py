#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Existing engine behavior: vLLM's own tests beside the files native support touches, and a sibling layout.

    python regressions.py            # the sibling hybrid-layout check, then the upstream files in place
    python regressions.py --sibling  # the sibling check only (no GPU, no checkpoint)

The upstream files are run whole, from /app, so their node ids are upstream's:
`tests/models/test_registry.py`, `tests/v1/core/test_kv_cache_utils.py`, `tests/test_config.py` and
`tests/v1/worker/test_gpu_model_runner.py`. Some of their tests fail at the starting tree as well,
for reasons unrelated to this model; what must hold is that your tree fails nothing the starting
tree passes. Run the same command on a clean checkout of the base commit to compare.

The sibling check configures a tiny `JambaForCausalLM` and `JambaForSequenceClassification` (weight-
and GPU-free) and requires the hybrid KV-cache layout the starting tree gives both: the attention
block sized to cover the mamba page and the mamba page padded to it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
PY = sys.executable
ACTION_LIMIT_S = json.loads((Path(__file__).resolve().parents[1] / "operation-limits.json").read_text())["limits_s"]["action"]
UPSTREAM_FILES = ("tests/models/test_registry.py", "tests/v1/core/test_kv_cache_utils.py", "tests/test_config.py",
                  "tests/v1/worker/test_gpu_model_runner.py")
#: What the starting tree gives a tiny Jamba configuration: [attention block size, padded mamba page bytes].
SIBLING_LAYOUT = [48, 12288]

SIBLING_CODE = """
import json, tempfile
from transformers import JambaConfig
from vllm.engine.arg_utils import EngineArgs
d = tempfile.mkdtemp(prefix="infb-jamba-tiny-")
out = {}
for arch in ("JambaForCausalLM", "JambaForSequenceClassification"):
    cfg = JambaConfig(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, attn_layer_period=2, attn_layer_offset=1,
                      expert_layer_period=2, expert_layer_offset=1, num_experts=2, num_experts_per_tok=1,
                      mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_dt_rank=8,
                      max_position_embeddings=2048, num_labels=1)
    cfg.architectures = [arch]
    cfg.save_pretrained(d)
    vc = EngineArgs(model=d, skip_tokenizer_init=True, max_model_len=2048, dtype="bfloat16").create_engine_config()
    out[arch] = [vc.cache_config.block_size, vc.cache_config.mamba_page_size_padded]
print("INFB_L0 " + json.dumps(out))
"""


def sibling_layout() -> dict:
    """The hybrid KV-cache layout the engine gives the two tiny Jamba forms."""
    r = subprocess.run([PY, "-c", SIBLING_CODE], capture_output=True, text=True, timeout=ACTION_LIMIT_S, cwd="/")
    assert r.returncode == 0, (
        "the engine could not configure a tiny Jamba model:\n" + r.stdout[-3000:] + r.stderr[-10000:]
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("INFB_L0 ")]
    assert lines, "no configuration record in output:\n" + r.stdout[-3000:]
    return json.loads(lines[-1][len("INFB_L0 ") :])


def check_sibling(got: dict, want: list = SIBLING_LAYOUT) -> None:
    assert got["JambaForCausalLM"] == want, (
        f"the tree changes the engine's hybrid layout for JambaForCausalLM: {got['JambaForCausalLM']}, "
        f"the base gives {want}"
    )
    assert got["JambaForSequenceClassification"] == want, (
        f"the tree changes the engine's hybrid layout for JambaForSequenceClassification: "
        f"{got['JambaForSequenceClassification']}, the base gives {want}"
    )


def run_upstream(files=UPSTREAM_FILES, app: Path = APP_DIR, junit: Path | None = None) -> int:
    command = [PY, "-m", "pytest", "-p", "no:cacheprovider", "--rootdir", str(app), "-q", *files]
    if junit is not None:
        command += ["--junitxml", str(junit)]
    return subprocess.run(command, cwd=str(app)).returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sibling", action="store_true", help="the sibling layout check only")
    parser.add_argument("--junit", type=Path, help="the upstream files' junit report")
    args = parser.parse_args(argv)
    try:
        got = sibling_layout()
        check_sibling(got)
        print(f"[regressions] sibling hybrid layout unchanged: {json.dumps(got)}", flush=True)
        rc = 0
    except AssertionError as exc:
        print(f"[regressions] sibling hybrid layout: FAILED\n{exc}", flush=True)
        rc = 1
    if not args.sibling:
        upstream = run_upstream(junit=args.junit)
        print(f"[regressions] upstream files exit {upstream}; compare the failures with the starting tree's", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
