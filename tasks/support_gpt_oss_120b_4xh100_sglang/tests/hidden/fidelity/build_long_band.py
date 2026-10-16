#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Compose the long band's three held-out fidelity sequences from the task's public efficiency sample.

Authoring step, run once INSIDE the task image (it needs the checkpoint's tokenizer); the reference
log-probabilities are filled in afterwards by `record_reference.py` under torchrun at world 8 with
`--q-chunk 1024`. Never run at grading.

    /opt/sglang/bin/python build_long_band.py --texts /public/efficiency/sample.jsonl.xz --out <dir>
    torchrun --nproc-per-node=8 record_reference.py --seqs <dir> --checkpoint <original/> \
        --reference-root /public --mlp-chunk 64 --q-chunk 1024

Why this text: the band exists to reach past the checkpoint's initial 4,096-token context, where
YaRN's scaling and the 128-token window meet context the other bands never reach, and the task's own
corpus (`corpus.py`) has no text that long. The public sample does. Its documents, each followed by a
blank line and tokenized with the checkpoint's tokenizer without special tokens, are concatenated in
file order, and three sequences are cut at fixed token offsets. So the band's text is public; what is
held out is the three sequences and the reference's values at their scored positions, index 4,096
onward. `build_public_examples.py` refuses any public example that reaches one of those positions.

The rule is fixed and needs no judgment: the offsets and lengths are in CUTS.
"""
from __future__ import annotations

import argparse
import json
import lzma
import pathlib

CONFIG = pathlib.Path(__file__).resolve().parents[2] / "config.json"
CUTS = [("long_01", 0, 8192), ("long_02", 20000, 8192), ("long_03", 40000, 16384)]
SCORED_FROM = 4096
SOURCE = ("the public efficiency sample's documents, concatenated in file order with a blank line between, "
          "tokenized with the checkpoint's tokenizer without special tokens")
RECORDING = ("the task recorder (record_reference.py) at world 8, --mlp-chunk 64, with torch sdpa evaluated in "
             "1,024-row query chunks: bit-identical to the unchunked sdpa on both 8k sequences, and the only "
             "way a 16,384-token forward fits")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="/public/efficiency/sample.jsonl.xz")
    ap.add_argument("--tokenizer", default="", help="default: config.json's infb.model_repo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer or json.loads(CONFIG.read_text())["infb"]["model_repo"])
    docs = [json.loads(line) for line in lzma.open(a.texts, "rt") if line.strip()]
    need = max(off + n for _, off, n in CUTS)
    ids: list[int] = []
    for d in docs:
        ids.extend(tok.encode(d["text"] + "\n\n", add_special_tokens=False))
        if len(ids) >= need:
            break
    assert len(ids) >= need, f"the sample gives {len(ids)} tokens, the cuts need {need}"

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, off, n in CUTS:
        rec = {"id": name, "kind": "hidden", "band": "long",
               "construction": {"source": SOURCE, "token_offset": off, "target_tokens": n,
                                "scored_from_index": SCORED_FROM, "recording": RECORDING},
               "n_tokens": n, "token_ids": ids[off:off + n], "reference": None}
        (out / f"{name}.json").write_text(json.dumps(rec) + "\n")
        print(f"[long] {name}: tokens {off}..{off + n - 1} of the concatenation, scored from index {SCORED_FROM}")
    print(f"[long] wrote {len(CUTS)} sequences to {out} from {len(ids)} concatenated tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
