#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Read aiperf's export and write the metrics shape the verifier divides."""
from __future__ import annotations

import json
import pathlib
import sys


def _find(o: object, key: str):
    """First `key` anywhere in aiperf's export, unwrapping its {avg,p50,...} records."""
    if isinstance(o, dict):
        v = o.get(key)
        if isinstance(v, dict) and "avg" in v:
            return v["avg"]
        if isinstance(v, (int, float)):
            return v
        for x in o.values():
            r = _find(x, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for x in o:
            r = _find(x, key)
            if r is not None:
                return r
    return None


def read(artifact_dir: pathlib.Path) -> dict | None:
    """Normalise one aiperf artifact directory, or None when it produced nothing usable."""
    hits = sorted(artifact_dir.rglob("profile_export_aiperf.json"))
    if not hits:
        return None
    try:
        doc = json.loads(hits[0].read_text())
    except Exception:
        return None
    tps = _find(doc, "output_token_throughput")
    itl = _find(doc, "inter_token_latency")
    osl = _find(doc, "output_sequence_length")
    isl = _find(doc, "input_sequence_length")
    duration = _find(doc, "benchmark_duration")
    requests = _find(doc, "request_count")
    if tps is None:
        return None
    elapsed = duration if isinstance(duration, (int, float)) and duration > 0 else None
    rps = (requests / elapsed) if (elapsed and isinstance(requests, (int, float))) else None
    total_tps = ((((isl or 0) + (osl or 0)) * requests / elapsed)
                 if (elapsed and isinstance(requests, (int, float))) else None)
    return {
        "elapsed_time": elapsed,
        "requests_per_second": rps,
        "tokens_per_second": total_tps,
        "output_throughput": tps,
        "output_tokens_per_second": tps,
        "inter_token_latency_ms": itl,
        "time_to_first_token_ms": _find(doc, "time_to_first_token"),
        "request_latency_ms": _find(doc, "request_latency"),
        "output_sequence_length": osl,
        "input_sequence_length": isl,
        "source": "aiperf",
        "export": str(hits[0]),
    }


def main() -> int:
    d = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    res = read(d)
    if res is None:
        print(f"[aiperf_metrics] no usable export under {d}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(f"[aiperf_metrics] {res['output_throughput']:.2f} tok/s  "
          f"ITL {res['inter_token_latency_ms']:.3f} ms  "
          f"OSL {res['output_sequence_length']}  -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
