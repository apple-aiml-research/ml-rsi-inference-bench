#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The serving side of attribution (README, `run.py binding --base-url`): profile a running server the way grading
observes it, then count the component identity sets in its traces with `../rel_attention/binding.py`.

    python witness.py --base-url http://127.0.0.1:30000 --identity /tmp/ids.json --trace-dir /tmp/witness

Last on a production launch, grading sends `/start_profile` (CPU and GPU activities, output to a fresh directory),
then one 6,144-token prose prompt generating 32 tokens, then three concurrent 3,000-token prompts generating 32 tokens
each, then `/stop_profile`, and reads the four ranks' traces. The prompts here are cut from `sample.jsonl.xz` with the
model's tokenizer (the hidden run uses hidden text). The profiler writes where the server can write; the directory
must be visible to this process (the same machine). A static import, a registration or a kernel name printed in a log
is not evidence; only kernels observed in the trace count.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import lzma
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rel_attention"))
from server import generate  # noqa: E402

TOKENIZER = "/model/thinkingmachines/Inkling-Small-NVFP4/tokenizer.json"
ACTION_S = 300.0


def prompts(tokenizer_path: str) -> tuple[list[int], list[list[int]]]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    docs = [json.loads(ln) for ln in lzma.open(HERE / "sample.jsonl.xz", "rt")]
    prose = [d for d in docs if d["family"] == "prose"]
    ids = [tok.encode(d["text"], add_special_tokens=False).ids for d in prose[:8]]
    long = next(i for i in ids if len(i) >= 6144)[:6144]
    short = [i[:3000] for i in ids if len(i) >= 3000 and i[:6144] != long][:3]
    return long, short


def run(base_url: str, trace_dir: Path, tokenizer_path: str) -> dict:
    long, short = prompts(tokenizer_path)
    trace_dir.mkdir(parents=True, exist_ok=True)
    body = {"output_dir": str(trace_dir), "activities": ["CPU", "GPU"], "with_stack": False, "record_shapes": False}
    r = requests.post(f"{base_url}/start_profile", json=body, timeout=ACTION_S)
    r.raise_for_status()
    t0 = time.monotonic()
    try:
        generate(base_url, long, max_new_tokens=32, timeout=900.0, return_logprob=False)
        with concurrent.futures.ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda ids: generate(base_url, ids, max_new_tokens=32, timeout=900.0, return_logprob=False), short))
    finally:
        requests.post(f"{base_url}/stop_profile", timeout=ACTION_S)
    return {"trace_dir": str(trace_dir), "workload_s": round(time.monotonic() - t0, 2),
            "prompts": {"single": len(long), "concurrent": [len(s) for s in short]}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--identity", type=Path, required=True, help="identity sets from rel_attention/binding.py")
    ap.add_argument("--trace-dir", type=Path, default=Path("/tmp/inkling_small_witness"))
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--wait-s", type=float, default=120.0, help="how long to wait for the four ranks' trace files")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    import binding
    rec = run(args.base_url, args.trace_dir, args.tokenizer)
    deadline = time.monotonic() + args.wait_s
    while time.monotonic() < deadline and len(list(args.trace_dir.rglob("*.json*"))) < 4:
        time.sleep(2)
    ids = json.loads(args.identity.read_text())
    rec["serving"] = binding.serving(args.trace_dir, ids.get("identity_sets", ids))
    text = json.dumps(rec, indent=1)
    print(text)
    if args.output:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
