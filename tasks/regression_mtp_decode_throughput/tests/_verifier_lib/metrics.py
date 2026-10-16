#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Vendored metric normalization for regression verifier tasks.

Copy of inference_bench.metrics.normalize_engine_metrics, made self-contained
so the verifier doesn't need the inference_bench package on PYTHONPATH.

Keep this in sync with inference_bench/metrics.py when the upstream schema
changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_engine_metrics(
    input_path: str | Path,
    output_path: str | Path,
    *,
    metric_source: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(input_path)
    records = load_records(source)
    standard_records = [standardize_record(record, metric_source=metric_source) for record in records]
    metrics = {
        "schema_version": 1,
        "metric_source": metric_source,
        "source_path": str(source),
        "provenance": _drop_empty(provenance or {}),
        "summary": standard_records[-1] if standard_records else {},
        "records": [
            {"standard": standard, "raw": raw}
            for standard, raw in zip(standard_records, records, strict=True)
        ],
    }
    _write_json(Path(output_path), metrics)
    return metrics


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(item)
        return records
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data
    raise ValueError(f"Expected JSON object, object array, or JSONL object records: {path}")


def standardize_record(record: dict[str, Any], *, metric_source: str = "generic") -> dict[str, Any]:
    standard: dict[str, Any] = {
        "metric_source": metric_source,
        "benchmark": {},
        "duration_sec": {},
        "requests": {},
        "tokens": {},
        "throughput": {},
        "latency_ms": {},
        "memory": {},
        "kv_cache": {},
        "reliability": {},
        "gpu": {},
    }
    _copy_numeric(record, standard["requests"], "completed_requests", "completed", "successful_requests", "num_requests")
    _copy_numeric(record, standard["requests"], "failed_requests", "failed", "failed_requests")
    _copy_numeric(record, standard["tokens"], "input", "total_input_tokens")
    _copy_numeric(record, standard["tokens"], "output", "total_output_tokens")
    _copy_numeric(record, standard["tokens"], "output_retokenized", "total_output_tokens_retokenized")
    _copy_numeric(record, standard["tokens"], "total", "total_num_tokens")
    _copy_numeric(record, standard["duration_sec"], "elapsed", "duration", "elapsed_time")
    _copy_numeric(record, standard["throughput"], "request_per_sec", "request_throughput", "requests_per_second")
    _copy_numeric(record, standard["throughput"], "request_goodput_per_sec", "request_goodput")
    _copy_numeric(record, standard["throughput"], "input_tokens_per_sec", "input_throughput")
    _copy_numeric(record, standard["throughput"], "output_tokens_per_sec", "output_throughput")
    # `total_throughput` IS SGLANG'S SPELLING. vLLM's benches write `total_token_throughput`;
    # `sglang.bench_serving` writes `total_throughput`. Without the alias the normalised summary
    # simply has no `total_tokens_per_sec`, the `serve` bucket's default `min_total_tokens_per_sec:
    # 1.0` bound reads None, and the run fails `failed_checks: ["total_tokens_per_sec"]` -- with a
    # correct ratio sitting next to it. A MISSING key and a key BELOW a floor are indistinguishable
    # in that message.
    _copy_numeric(
        record,
        standard["throughput"],
        "total_tokens_per_sec",
        "total_token_throughput",
        "total_throughput",
        "tokens_per_second",
    )
    _copy_numeric(record, standard["memory"], "peak_memory_mb", "peak_memory_mb", "peak_gpu_memory_mb", "gpu_peak_memory_mb", "max_gpu_memory_mb")
    _copy_numeric(record, standard["memory"], "allocated_memory_mb", "allocated_memory_mb", "gpu_allocated_memory_mb")
    _copy_numeric(record, standard["kv_cache"], "usage_pct", "kv_cache_usage_pct", "kv_cache_utilization_pct", "kv_cache_usage")
    _copy_numeric(record, standard["kv_cache"], "tokens", "kv_cache_tokens", "num_cached_tokens")
    _copy_numeric(record, standard["reliability"], "error_count", "error_count", "num_errors", "errors")
    _copy_numeric(record, standard["reliability"], "crash_count", "crash_count", "num_crashes")
    _copy_numeric(record, standard["reliability"], "oom_count", "oom_count", "num_ooms")
    _copy_numeric(record, standard["reliability"], "timeout_count", "timeout_count", "num_timeouts")
    _copy_numeric(record, standard["gpu"], "utilization_pct", "gpu_utilization_pct", "gpu_utilization")
    _copy_numeric(record, standard["gpu"], "sm_utilization_pct", "sm_utilization_pct", "gpu_sm_utilization_pct")
    _copy_numeric(record, standard["gpu"], "power_watts", "gpu_power_watts", "power_watts")
    for metric in ("ttft", "tpot", "itl"):
        group = _latency_group(record, [metric])
        if group:
            standard["latency_ms"][metric] = group
    e2e = _latency_group(record, ["e2el", "e2e_latency", "e2e"])
    if e2e:
        standard["latency_ms"]["e2e"] = e2e
    _copy_latency_seconds(record, standard["latency_ms"], "e2e", "avg_latency", out_stat="mean")
    _copy_latency_seconds(record, standard["latency_ms"], "e2e", "median_latency", out_stat="median")
    _copy_latency_percentiles_seconds(record, standard["latency_ms"], "e2e")
    if "latencies" in record and isinstance(record["latencies"], list):
        standard["benchmark"]["latency_sample_count"] = len(record["latencies"])
    _detect_benchmark_kind(record, standard["benchmark"])
    return _drop_empty(standard)


def _latency_group(record: dict[str, Any], aliases: list[str]) -> dict[str, float | int]:
    group: dict[str, float | int] = {}
    for stat in ("mean", "median", "std", "p50", "p90", "p95", "p99"):
        keys = [f"{stat}_{alias}_ms" for alias in aliases]
        keys.extend(f"{alias}_{stat}_ms" for alias in aliases)
        value = _first_numeric(record, keys)
        if value is not None:
            group[stat] = value
    return group


def _copy_numeric(record: dict[str, Any], target: dict[str, Any], out_key: str, *keys: str) -> None:
    value = _first_numeric(record, list(keys))
    if value is not None:
        target[out_key] = value


def _first_numeric(record: dict[str, Any], keys: list[str]) -> float | int | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return value
    return None


def _copy_latency_seconds(
    record: dict[str, Any],
    target: dict[str, Any],
    group_name: str,
    source_key: str,
    *,
    out_stat: str,
) -> None:
    value = record.get(source_key)
    if isinstance(value, (int, float)):
        target.setdefault(group_name, {})[out_stat] = value * 1000


def _copy_latency_percentiles_seconds(
    record: dict[str, Any],
    target: dict[str, Any],
    group_name: str,
) -> None:
    percentiles = record.get("percentiles")
    if not isinstance(percentiles, dict):
        return
    group = target.setdefault(group_name, {})
    for key, value in percentiles.items():
        if not isinstance(value, (int, float)):
            continue
        try:
            percentile = float(key)
        except (TypeError, ValueError):
            continue
        stat = f"p{int(percentile)}" if percentile.is_integer() else f"p{str(percentile).replace('.', '_')}"
        group[stat] = value * 1000


def _detect_benchmark_kind(record: dict[str, Any], target: dict[str, Any]) -> None:
    if {"requests_per_second", "tokens_per_second", "elapsed_time"} & set(record):
        target["name"] = "throughput"
    elif {"request_throughput", "mean_ttft_ms", "mean_e2el_ms"} & set(record):
        target["name"] = "serve"
    elif {"avg_latency", "latencies", "percentiles"} & set(record):
        target["name"] = "latency"


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            next_item = _drop_empty(item)
            if next_item not in ({}, [], None):
                cleaned[key] = next_item
        return cleaned
    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            next_item = _drop_empty(item)
            if next_item not in ({}, [], None):
                cleaned_items.append(next_item)
        return cleaned_items
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
