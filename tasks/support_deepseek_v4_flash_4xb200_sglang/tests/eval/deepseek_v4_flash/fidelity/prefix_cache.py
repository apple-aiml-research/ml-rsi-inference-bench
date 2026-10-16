#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Prefix caching, measured as a behaviour of the served tree: the protocol and comparison the grader uses, shared
byte for byte with it."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import requests

def _fidelity():
    """The sibling fidelity core, loaded by path under the grader's module name (one instance in a process)."""
    name = "deepseek_v4_flash_eval_fidelity"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "fidelity.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


fidelity = _fidelity()
ACTION_LIMIT_S = fidelity.ACTION_LIMIT_S


def validate_item(item, where: str = "") -> None:
    """Raise `fidelity.ReferenceFault` unless `item` is a prefix item: an id, token ids, and a prefix that leaves at
    least two continuation tokens (the continuation's first position has no value)."""
    label = where or (item.get("id") if isinstance(item, dict) else None) or "?"
    if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
        raise fidelity.ReferenceFault(f"prefix item {label}: no id")
    ids, prefix = item.get("input_ids"), item.get("prefix_tokens")
    if not isinstance(ids, list) or not all(fidelity._is_token_id(t) for t in ids):  # noqa: SLF001 - sibling module
        raise fidelity.ReferenceFault(f"prefix item {item['id']}: input_ids is not a list of token ids")
    if isinstance(prefix, bool) or not isinstance(prefix, int) or not (0 < prefix <= len(ids) - 2):
        raise fidelity.ReferenceFault(f"prefix item {item['id']}: prefix_tokens {prefix!r} for {len(ids)} tokens")
    if "continuation_tokens" in item and item["continuation_tokens"] != len(ids) - prefix:
        raise fidelity.ReferenceFault(f"prefix item {item['id']}: continuation_tokens {item['continuation_tokens']!r} "
                                      f"but {len(ids) - prefix} tokens follow the prefix")


def load_items(prefix_dir: Path) -> list[dict]:
    """Every item record of a directory, sorted by file name so the median is over a fixed set in a fixed order, each
    checked by `validate_item` and the ids checked unique, before anything is sent."""
    out, seen = [], set()
    for p in sorted(Path(prefix_dir).glob("*.json")):
        if p.name == "provenance.json":
            continue
        try:
            item = json.loads(p.read_text())
        except ValueError as exc:
            raise fidelity.ReferenceFault(f"prefix item {p.name}: not JSON ({exc})") from exc
        validate_item(item, p.name)
        if item["id"] in seen:
            raise fidelity.ReferenceFault(f"prefix item id {item['id']} appears twice")
        seen.add(item["id"])
        out.append(item)
    if not out:
        raise fidelity.ReferenceFault(f"no prefix items in {prefix_dir}")
    return out


def flush(base_url: str) -> None:
    """SGLang's own cache flush, with no other request in flight."""
    try:
        response = requests.post(f"{base_url}/flush_cache", timeout=ACTION_LIMIT_S)
    except requests.Timeout as exc:
        raise fidelity.ServedRequestFailed(f"/flush_cache: no answer within its {ACTION_LIMIT_S} s limit") from exc
    if response.status_code != 200:
        raise fidelity.ServedRequestFailed(f"/flush_cache answered HTTP {response.status_code}: {response.text[:200]}")
    time.sleep(2)


def teacher_forced(base_url: str, input_ids: list[int], start: int, item_id: str) -> dict:
    """The fidelity call with `logprob_start_len` at `start`, parsed and checked: the parse, with the answer's
    `cached_tokens` (`fidelity.StructuralFailure` if it is not a count)."""
    got = fidelity.request_logprobs(base_url, input_ids, start, item_id)
    body = got.pop("body")
    raw = (body.get("meta_info") or {}).get("input_token_logprobs") if isinstance(body, dict) \
        and isinstance(body.get("meta_info"), dict) else None
    try:
        parsed = fidelity.parse_input_logprobs(input_ids, start, body, item_id)
        cached = body["meta_info"].get("cached_tokens")
        if isinstance(cached, bool) or not isinstance(cached, int) or cached < 0:
            raise fidelity.StructuralFailure(f"item {item_id}: meta_info.cached_tokens is {cached!r}, not a count")
    except fidelity.StructuralFailure as exc:
        exc.observation = {**got, "raw": raw}
        raise
    return {**parsed, **got, "cached_tokens": cached, "raw": raw}


def prime(base_url: str, prefix_ids: list[int], item_id: str) -> None:
    """Serve the prefix alone (one generated token), so the cache holds it."""
    limit = fidelity.operation_limit_s(len(prefix_ids))
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={"input_ids": prefix_ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
            timeout=(limit, limit),
        )
    except requests.Timeout as exc:
        raise fidelity.ServedRequestFailed(f"item {item_id}: priming the prefix: no answer within its {limit} s "
                                           f"limit") from exc
    if response.status_code != 200:
        raise fidelity.ServedRequestFailed(f"item {item_id}: priming the prefix: HTTP {response.status_code}: "
                                           f"{response.text[:300]}")


def measure_item(base_url: str, item: dict, *, draw: int = 1, on_attempt=None) -> dict:
    """One item, cold then warm, each checked in full before the comparison; the attempt record. `on_attempt(attempt)`
    sees every attempt before this returns or raises."""
    validate_item(item)
    ids, start, iid = item["input_ids"], int(item["prefix_tokens"]), item["id"]
    attempt = {"check": "prefix", "id": iid, "draw": draw, "tokens": len(ids), "logprob_start_len": start,
               "input_digest": fidelity.ids_digest(ids), "expected_positions": len(ids) - start - 1,
               "expected_span": [start + 1, len(ids) - 1]}
    side = "cold"
    try:
        flush(base_url)
        cold = teacher_forced(base_url, ids, start, f"{iid} (cold)")
        attempt["cold"] = {k: v for k, v in cold.items() if k != "values"}
        side = "prime"
        flush(base_url)
        prime(base_url, ids[:start], iid)
        side = "warm"
        warm = teacher_forced(base_url, ids, start, f"{iid} (warm)")
        attempt["warm"] = {k: v for k, v in warm.items() if k != "values"}
    except (fidelity.StructuralFailure, fidelity.ServedRequestFailed) as exc:
        attempt.update({"status": exc.status, "structural": "invalid", "failed_side": side, "error": str(exc)[:1000],
                        side: getattr(exc, "observation", {})})
        if on_attempt:
            on_attempt(attempt)
        raise
    except requests.RequestException as exc:
        attempt.update({"status": "unreachable", "structural": "none", "failed_side": side, "error": repr(exc)[:1000]})
        if on_attempt:
            on_attempt(attempt)
        raise
    attempt.update({"structural": "valid", "status": "measured", "positions": cold["positions"],
                    "cached_tokens": warm["cached_tokens"], "k3": fidelity.k3(cold["values"], warm["values"])})
    if on_attempt:
        on_attempt(attempt)
    return attempt


def measure(base_url: str, prefix_dir: Path, *, draw: int = 1, on_attempt=None) -> dict:
    """Per item: cold (after a flush), then prime and warm; k3 of warm against cold, and warm's cached_tokens. Every
    item is checked before the first request."""
    items = load_items(prefix_dir)
    per_item: dict[str, dict] = {}
    for item in items:
        a = measure_item(base_url, item, draw=draw, on_attempt=on_attempt)
        per_item[item["id"]] = {"cached_tokens": a["cached_tokens"], "k3": a["k3"], "positions": a["positions"]}
    values = [v["k3"] for v in per_item.values()]
    return {"median_k3": fidelity.median(values), "per_item": per_item,
            "min_cached": min(v["cached_tokens"] for v in per_item.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="measure prefix reuse and cold/warm log-probability agreement")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--items", default=str(Path(__file__).resolve().parent / "examples" / "prefix"),
                    help="a directory of item records (default: the supplied examples)")
    args = ap.parse_args(argv)
    items = load_items(Path(args.items))
    measured, invalid = {}, 0
    for item in items:
        try:
            a = measure_item(args.base_url, item)
        except (fidelity.StructuralFailure, fidelity.ServedRequestFailed) as exc:
            invalid += 1
            print(f"{item['id']}: invalid answer: {exc}")
            continue
        measured[item["id"]] = a
        print(f"{item['id']}: cached_tokens {a['cached_tokens']}, warm-against-cold k3 {a['k3']:.6f} over "
              f"{a['positions']} validated positions {a['expected_span'][0]}-{a['expected_span'][1]}")
    summary = {"items": len(items), "valid": len(measured), "invalid": invalid}
    if measured:
        summary |= {"median_k3": fidelity.median([a["k3"] for a in measured.values()]),
                    "min_cached": min(a["cached_tokens"] for a in measured.values())}
    print(json.dumps(summary, sort_keys=True))
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
