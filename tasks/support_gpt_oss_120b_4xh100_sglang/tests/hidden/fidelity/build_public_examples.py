#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Compose the four PUBLIC fidelity examples from the task's public efficiency sample.

Authoring step, run once INSIDE the task image (it needs `openai_harmony`); the reference
log-probabilities are filled in afterwards by `record_reference.py` under torchrun, with the same
settings that recorded the held-out set. Never run at grading.

    /opt/sglang/bin/python build_public_examples.py --texts /public/efficiency/sample.jsonl.xz \
        --hidden tests/hidden/fidelity --out <dir>

Why these texts: `/public/efficiency/sample.jsonl.xz` is already public, it is public-domain text of
the same four families the grader's efficiency prompts come from, and it shares no source with
`corpus.py`, which is text written for this task. The held-out passages and the four `public_0N`
sequences built from them stay unshipped: every one of those passages also composes held-out
sequences, so they are held-out material.

The selection rule is fixed and needs no judgment: for the families prose, math, code and struct, in
that order, the FIRST document of the family in the sample's file order, encoded exactly as
`build_sequences.py` encodes its plain sequences (the Harmony encoding, `allowed_special=set()`, no
special tokens added) and cut to its first 512, 768, 1,024 and 1,536 tokens respectively -- the
lengths of the held-out plain set.

Disjointness is checked here, on token ids, not argued: no 16-token window of any example may occur
anywhere in the held-out sequences of the Harmony, plain and broad bands, nor reach a scored position
of the long band's three, and the script refuses to write if one does. The long band is cut from this
same public sample, so only its scored positions are held-out material: `example_03`'s text is the
unscored opening of `long_01`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import pathlib
import sys

FAMILIES = [("prose", 512), ("math", 768), ("code", 1024), ("struct", 1536)]
WINDOW = 16


def windows(ids: list[int], w: int = WINDOW) -> set[tuple[int, ...]]:
    return {tuple(ids[i:i + w]) for i in range(len(ids) - w + 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--hidden", required=True, help="the held-out fidelity directory (tests/hidden/fidelity)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    raw = lzma.open(a.texts, "rb").read()
    docs = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]

    held, long_band = [], []
    for p in sorted(pathlib.Path(a.hidden).glob("*.json")):
        if p.name == "index.json":
            continue
        d = json.loads(p.read_text())
        if d["kind"] == "hidden" and d.get("band") == "long":
            # every window that reaches a scored position
            first = max(0, d["construction"]["scored_from_index"] - WINDOW + 1)
            long_band.append(d["token_ids"][first:])
        elif d["kind"] == "hidden":
            held.append(d["token_ids"])
    assert len(held) == 32, f"expected the 32 held-out sequences, found {len(held)}"
    assert len(long_band) == 3, f"expected the long band's 3 sequences, found {len(long_band)}"
    held_windows: set[tuple[int, ...]] = set()
    for ids in held + long_band:
        held_windows |= windows(ids)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for n, (family, target) in enumerate(FAMILIES, start=1):
        doc = next(d for d in docs if d["family"] == family)
        ids = enc.encode(doc["text"], allowed_special=set())[:target]
        assert len(ids) == target, (family, len(ids), target)
        shared = windows(ids) & held_windows
        print(f"[public] example_{n:02d} {family:6} {target:5} tokens from {doc['id']} ({doc['work']}, offset "
              f"{doc['offset_chars']}): {len(shared)} of {len(windows(ids))} {WINDOW}-token windows shared with "
              f"the held-out set")
        if shared:
            print("[public] REFUSED: an example shares text with the held-out set", file=sys.stderr)
            return 1
        recs.append({
            "id": f"example_{n:02d}",
            "kind": "public_example",
            "family": family,
            "source": {"file": "/public/efficiency/sample.jsonl.xz", "document": doc["id"], "work": doc["work"],
                       "offset_chars": doc["offset_chars"],
                       "document_text_sha256": hashlib.sha256(doc["text"].encode()).hexdigest()},
            "construction": {"rule": f"the first {family} document in file order, its first {target} tokens",
                             "encoding": "openai_harmony HARMONY_GPT_OSS, encode(text, allowed_special=set())",
                             "target_tokens": target},
            "disjointness": {"window_tokens": WINDOW, "held_out_sequences": len(held), "shared_windows": 0},
            "n_tokens": len(ids),
            "text": enc.decode(ids),
            "token_ids": ids,
            "reference": None,
        })
    for rec in recs:
        (out / f"{rec['id']}.json").write_text(json.dumps(rec) + "\n")
    print(f"[public] wrote {len(recs)} examples to {out}; every one disjoint from the held-out set at "
          f"{WINDOW} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
