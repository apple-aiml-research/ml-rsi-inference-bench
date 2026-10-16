#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Materialise the scored long-context prompt bank from nvidia/SPEED-Bench, inside the image.

WHY THE BANK IS NOT COMMITTED. It used to be carried as a copy of the rows, and two things follow
from that -- both bad:

  * The row count was capped by the byte cost, not by anything about the measurement. 32 rows out of a
    454-row clean pool -- 7% -- and c32's 64 requests were served by using each prompt twice.
  * `nvidia/SPEED-Bench` is published under the NVIDIA Evaluation Dataset License, which is not open
    source. Committing it and baking it into public task images REDISTRIBUTES it. Fetching does not.

Large third-party artifacts are fetched rather than carried -- model weights come from public
huggingface.co by default. A prompt bank was the outlier, not the rule.

WHAT IS COMMITTED INSTEAD is a ~0.5 KB spec: the dataset, the pinned revision, the selection RULE, the
expected row count, and the sha256 of the bank the rule produces. That is strictly more information than
the bytes carried: it says where the rows came from and why those rows, which is what a reader actually
needs and what a committed blob of JSON could not tell them.

WHY BUILD TIME AND NOT RUN TIME. This runs from the task Dockerfile, which already reaches pypi.org,
github.com and download.pytorch.org during the engine stage -- one more public host on the same step.
Staging the bank at run time instead makes it conditional on a path that need not exist on a
`--env docker` run, and a bank missing at measurement time is the worst place to discover it.

WHY THE SHA IS CHECKED HERE AND AGAIN AT SCORING TIME. Here, so a changed or withdrawn upstream fails
the BUILD, loudly, before any GPU time. At scoring time, because this file lands inside the image where
the agent works: re-checking it is a tamper check that carrying the bytes instead would not give.

Self-contained on purpose -- no imports from the repo, because the repo is not in the image.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import sys
import urllib.request

HF_HOST = "https://huggingface.co"
MASK = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"


def longest_identical_line_run(text: str) -> int:
    """Longest run of the SAME non-blank line repeated back to back.

    This is the filler detector, and it is a CONTIGUITY measure rather than a repetition rate because
    the two populations overlap on the rate: legitimate rows reach 0.361 8-gram repetition (one code row
    repeats an idiom 34 times, scattered) while padded rows start at 0.108. On contiguity they do not
    overlap at all -- legitimate text tops out at 7, upstream padding starts at 66.
    """
    best = cur = 0
    prev = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        cur = cur + 1 if line == prev else 1
        prev = line
        best = max(best, cur)
    return best


def first_turn(row: dict) -> str:
    turns = row.get("turns") or []
    if hasattr(turns, "tolist"):
        turns = list(turns)
    return turns[0] if turns and isinstance(turns[0], str) else ""


def select(rows: list[dict], spec: dict, token_len) -> list[str]:
    """Apply the spec's rule. The ONE implementation -- `--print-sha` authors the spec through it too.

    No sampling, no ordering, no row cap: every row that passes the rule is in, which is why there is no
    seed here and no "why these N" to justify.
    """
    rule = spec["rule"]
    seen: dict[str, int] = {}
    for r in rows:
        t = first_turn(r)
        seen[t] = seen.get(t, 0) + 1

    out: list[str] = []
    for r in rows:
        t = first_turn(r)
        if not t or t == MASK or seen[t] != 1 or len(t) <= rule["min_chars"]:
            continue
        if r.get("category") != rule["category"]:
            continue
        if bool(r.get("multiturn")) != rule["multiturn"]:
            continue
        if longest_identical_line_run(t) > rule["max_identical_line_run"]:
            continue
        if token_len is not None and token_len(t) > rule["max_tokens"]:
            continue
        out.append(t)
    return sorted(out)


def render(texts: list[str]) -> str:
    return "\n".join(json.dumps({"prompt": t}, ensure_ascii=False) for t in texts) + "\n"


def _fetch_parquet(spec: dict) -> bytes:
    url = (f"{HF_HOST}/datasets/{spec['dataset']}/resolve/{spec['revision']}/"
           f"{spec['config']}/test-00000-of-00001.parquet")
    with urllib.request.urlopen(url, timeout=600) as fh:   # noqa: S310 -- fixed https host
        return fh.read()


def _rows_from_parquet(blob: bytes) -> list[dict]:
    import pyarrow.parquet as pq
    table = pq.read_table(io.BytesIO(blob))
    return table.to_pylist()


def _token_len_fn(spec: dict, tokenizer_dir: pathlib.Path | None):
    """Longest of the declared tokenizers, or None if the spec declares none.

    The ceiling exists so a prompt cannot crowd out `output_len`; the LONGEST tokenizer is the binding
    one, so a row that fits under it fits under all of them. Never a silent skip, and never estimated
    from a chars-per-token constant: that understates long rows badly enough to ship a bank 23% short of
    its declared length, which no CPU-side check catches.

    The tokenizers are FETCHED, from the full Hugging Face ids the spec names, for the same reason the
    bank is: they are 6-12 MB apiece and ungated, and hardcoding a name-to-id mapping per task is a
    mapping that goes stale silently. Cached under `tokenizer_dir` so a rebuild does not refetch.
    """
    ids = spec["rule"].get("ceiling_tokenizers") or []
    if not ids:
        return None
    from tokenizers import Tokenizer
    root = tokenizer_dir or pathlib.Path("/opt/infb/tokenizers")
    toks = []
    for hf_id in ids:
        f = root / hf_id.replace("/", "__") / "tokenizer.json"
        if not f.is_file():
            f.parent.mkdir(parents=True, exist_ok=True)
            url = f"{HF_HOST}/{hf_id}/resolve/main/tokenizer.json"
            try:
                with urllib.request.urlopen(url, timeout=300) as fh:   # noqa: S310
                    f.write_bytes(fh.read())
            except Exception as exc:  # noqa: BLE001
                sys.exit(f"FATAL: cannot fetch tokenizer {hf_id} from {url}: {exc}. The token ceiling "
                         f"cannot be applied and this build must not guess it.")
        toks.append(Tokenizer.from_file(str(f)))
    return lambda t: max(len(x.encode(t).ids) for x in toks)


def build(spec: dict, tokenizer_dir: pathlib.Path | None = None) -> tuple[str, dict]:
    rows = _rows_from_parquet(_fetch_parquet(spec))
    texts = select(rows, spec, _token_len_fn(spec, tokenizer_dir))
    body = render(texts)
    got = {"rows": len(texts), "sha256": hashlib.sha256(body.encode()).hexdigest(),
           "rows_published": len(rows)}
    return body, got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--tokenizer-dir", type=pathlib.Path, default=None)
    ap.add_argument("--print-sha", action="store_true",
                    help="write nothing; print the sha the rule produces (used to author the spec)")
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    body, got = build(spec, args.tokenizer_dir)

    if args.print_sha:
        print(json.dumps(got, indent=2))
        return 0

    problems = []
    if got["rows"] != spec["expected_rows"]:
        problems.append(f"rows {got['rows']} != expected {spec['expected_rows']}")
    if got["sha256"] != spec["bank_sha256"]:
        problems.append(f"sha256 {got['sha256'][:16]} != expected {spec['bank_sha256'][:16]}")
    if problems:
        sys.exit("FATAL: the fetched bank does not match its spec -- upstream changed, or the rule did.\n"
                 + "\n".join(f"  {p}" for p in problems)
                 + f"\n  dataset={spec['dataset']} config={spec['config']} revision={spec['revision']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(f"  wrote {args.out}  {got['rows']} rows  {len(body) / 1048576:.1f} MB  sha {got['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
