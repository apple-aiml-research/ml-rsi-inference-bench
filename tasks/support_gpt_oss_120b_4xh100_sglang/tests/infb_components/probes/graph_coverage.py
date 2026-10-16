#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private graph-coverage workloads and the structured coverage record.

On a server launched through `infb_components.probes.observe_launch`, each workload
opens an observation window, flushes the prefix cache, submits its prompts as one
batched `/generate` (256-token prompts and the public client's greedy continuation with ignore-EOS), re-scores
every request by fresh prefill with the trusted public decode-batch client, and closes
the window. The record joins the server's own per-rank observations -- captured
shapes, every forward's live and padded batch, graph replays, eager native-module
calls and a profiled warm decode -- with every request's numerical result. The gates
in `infb_components.gates` judge it; this module has no limits of its own.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

SCHEMA = "gpt-oss-graph-coverage/1"
SETTLE_S = 1.0


def _decode_batch_client(protocols: Path):
    """The trusted public decode-batch client, loaded by path without writing bytecode beside it."""
    path = Path(protocols) / "gpt_oss/decode_batch/probe.py"
    spec = importlib.util.spec_from_file_location("infb_private_decode_batch_client", path)
    module = importlib.util.module_from_spec(spec)
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    return module


def read_ranks(observe: Path, ranks: int = 4) -> list[list[dict]]:
    result = []
    for rank in range(ranks):
        path = Path(observe) / f"rank-{rank}.jsonl"
        result.append([json.loads(line) for line in path.open()] if path.is_file() else [])
    return result


def workload(base_url: str, observe: Path, protocols: Path, name: str, prompts: list[list[int]],
             new_tokens: int | None = None) -> dict:
    """One workload in its own observation window; every attempt is returned, never raised away.
    The continuation length is the public client's own protocol constant."""
    client = _decode_batch_client(protocols)
    new_tokens = client.NEW_TOKENS if new_tokens is None else new_tokens
    marker = Path(observe) / "window.json"
    raw: list = []
    record = {"name": name, "n": len(prompts), "new_tokens": new_tokens,
              "prompt_sha256": [hashlib.sha256(json.dumps(p).encode()).hexdigest()[:16] for p in prompts]}
    marker.write_text(json.dumps({"name": name}))
    start = time.monotonic()
    try:
        measured = client.probe(base_url, prompts, new_tokens=new_tokens, raw=raw)
        record.update(structurally_valid=True, medians=measured["medians"], worst=measured["worst"],
                      median=measured["median"], flush=measured["flush"])
    except Exception as exc:  # noqa: BLE001 - a malformed or missing response is terminal for this workload
        record.update(structurally_valid=False, error=f"{type(exc).__name__}: {exc}"[:500])
    finally:
        time.sleep(SETTLE_S)
        marker.unlink(missing_ok=True)
        record["elapsed_s"] = time.monotonic() - start
    record["requests"] = raw
    return record


def summarize(ranks: list[list[dict]], name: str) -> list[dict]:
    """Per rank: the window's forwards and its profiled warm decode."""
    summary = []
    for rank, rows in enumerate(ranks):
        forwards = [r for r in rows if r.get("kind") == "forward" and r.get("window") == name]
        profiled = [r for r in rows if r.get("kind") == "gpu_execution" and r.get("window") == name]
        decode = [r for r in forwards if r["mode"] == "DECODE"]
        graph = [r for r in decode if r["graph"]]
        summary.append({
            "rank": rank, "pids": sorted({r["pid"] for r in forwards}),
            "forwards": len(forwards), "decode_forwards": len(decode),
            "graph_decode_forwards": len(graph),
            "eager_decode_forwards": len(decode) - len(graph),
            "graph_decode_replays": sum(r["replays"] for r in graph),
            "graph_decode_native_calls": sum(r["native_calls"] for r in graph),
            "graph_decode_without_replay": sum(1 for r in graph if r["replays"] < 1),
            "live_to_padded": sorted({(r["live_bs"], r["padded_bs"]) for r in graph}),
            "largest_live_batch": max((r["live_bs"] for r in decode), default=0),
            "replay_interval_ns": ([min(r["t0_ns"] for r in graph), max(r["t1_ns"] for r in graph)]
                                   if graph else None),
            "profiled_decode": [{"graph": r["graph"], "batch_size": r.get("batch_size"),
                                 "attribution": r.get("attribution")} for r in profiled],
        })
    return summary


def draws(base_url: str, observe: Path, protocols: Path, name: str, prompts: list[list[int]],
          bar: float | None, limit: int = 2) -> list[dict]:
    """The protocol's bounded numerical retry: a structurally valid draw at or over the bar is
    drawn once more; a malformed draw is terminal and a passing draw needs no redraw."""
    result = []
    for index in range(limit):
        reading = workload(base_url, observe, protocols, f"{name}-{index + 1}", prompts)
        result.append(reading)
        if not reading["structurally_valid"] or bar is None or reading["worst"] < bar:
            break
    return result


def coverage(base_url: str, observe: Path, protocols: Path, prompt_pool: list[list[int]],
             *, workloads: dict[str, dict], recipe_max: int, bar: float | None, limit: int = 2) -> dict:
    """The complete coverage record: the fixed workloads (each `{"n": ..., "graph_replay": ...}`),
    then the largest-captured-batch workload derived from the observed captures, bounded by the
    recipe's maximum. Which workloads must replay on graphs is carried with each, not decided here."""
    ranks = read_ranks(observe)
    loaded = [next((r for r in rows if r.get("kind") == "loaded"), None) for rows in ranks]
    graphs = [next((r for r in rows if r.get("kind") == "graphs"), None) for rows in ranks]
    record = {"schema": SCHEMA, "recipe_max": recipe_max, "bar": bar,
              "ranks": [{"rank": rank, "pid": (l or {}).get("pid"),
                         "model_identity": (l or {}).get("identity"),
                         "binding_identities": sorted({json.dumps(b["identity"], sort_keys=True)
                                                       for b in (l or {}).get("bindings", [])}),
                         "captured": (g or {}).get("captured"), "capture_bs": (g or {}).get("capture_bs"),
                         "max_bs": (g or {}).get("max_bs")}
                        for rank, (l, g) in enumerate(zip(loaded, graphs))
                        if l is not None or g is not None],
              "workloads": {}}
    for name, spec in workloads.items():
        n = spec["n"]
        if n > len(prompt_pool):
            raise ValueError(f"workload {name} needs {n} prompts; the pool holds {len(prompt_pool)}")
        record["workloads"][name] = {"n": n, "graph_replay": bool(spec["graph_replay"]),
                                     "draws": draws(base_url, observe, protocols, name, prompt_pool[:n], bar, limit)}
    captured = [set(r["captured"] or []) for r in record["ranks"]]
    common = set.intersection(*captured) if len(captured) == 4 and all(captured) else set()
    n = min(max(common), recipe_max) if common else None
    record["largest_graph"] = {"n": n, "graph_replay": True,
                               "draws": (draws(base_url, observe, protocols, "largest", prompt_pool[:n], bar, limit)
                                         if n else [])}
    ranks = read_ranks(observe)
    for entry in [*record["workloads"].values(), record["largest_graph"]]:
        for reading in entry["draws"]:
            reading["observations"] = summarize(ranks, reading["name"])
    return record
