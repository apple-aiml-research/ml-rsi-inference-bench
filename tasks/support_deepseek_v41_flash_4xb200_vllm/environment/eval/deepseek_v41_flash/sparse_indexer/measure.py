#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The component measurement of the sparse indexer (README.md, "Measurement"): the work unit is one call of an entry
point -- one step of the four consumers -- timed by wall clock around device synchronization and by CUDA events.

    python measure.py --phase prefill|decode --cases examples.json --output OUT.json [--device cuda:0]

The input sets of a measurement record are generated here from its parameters (four sets per case, rotated between
work units; storage shapes fixed, values, pools, lengths and page tables varying). The script prints continuous
per-work-unit times; it declares no pass.

PyTorch and the engine are imported inside the functions, so the module imports with the standard library alone.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import contract  # noqa: E402

WARMUP_UNITS = 10
PILOT_MIN_MS = 200.0
PILOT_MAX_REPETITIONS = 2048
BLOCKS = 3
INPUT_SETS = 4


def _rng(seed: int, case_id: str, index: int):
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed((seed * 1_000_003 + sum(ord(c) * 131**i for i, c in enumerate(case_id)) * 97 + index) % (2**63))
    return g


def _mxfp4(g, shape, scale_lo: int = 120, scale_hi: int = 128):
    """Random packed E2M1 bytes and finite UE8M0 scale words, drawn directly."""
    import torch

    values = torch.randint(-128, 128, (*shape, contract.PACKED_BYTES), generator=g, dtype=torch.int16).to(torch.int8)
    scale_bytes = torch.randint(scale_lo, scale_hi + 1, (*shape, 4), generator=g, dtype=torch.int16).to(torch.uint8)
    return values, scale_bytes.view(torch.int32).view(*shape)


def _pool(g, visible: int):
    """A pool of unique visible blocks, newest included, not a consecutive run: all visible blocks when they fit."""
    import torch

    blocks = math.ceil(visible / contract.CANDIDATE_BLOCK)
    if blocks <= contract.NUM_CANDIDATES:
        chosen = torch.arange(blocks)
    else:
        others = torch.randperm(blocks - 1, generator=g)[: contract.NUM_CANDIDATES - 1]
        chosen = torch.cat([others, torch.tensor([blocks - 1])])
    row = torch.full((contract.NUM_CANDIDATES,), -1, dtype=torch.int32)
    row[: chosen.numel()] = chosen[torch.randperm(chosen.numel(), generator=g)].to(torch.int32)
    return row


def packed_sets(case: dict, seed: int, device: str) -> list:
    """Four packed input sets: `requests` requests with start residues `residues` modulo 8 and guards between them,
    `rows` query rows divided equally, each request's visible lengths rising to its final length across its rows."""
    import torch

    rows, final = int(case["rows"]), int(case["visible"])
    residues = case.get("residues", [0, 1, 5, 7])
    requests = int(case.get("requests", len(residues)))
    per = rows // requests
    sets = []
    for s in range(INPUT_SETS):
        g = _rng(seed, case["id"], s)
        L = final - 3 * s  # lengths vary between sets, never beyond the listed length
        starts, pos = [], 0
        for i in range(requests):
            pos = math.ceil((pos + 32) / 8) * 8 + residues[i % len(residues)]  # a guard, then the residue
            starts.append(pos)
            pos += L
        n = pos + 32
        k_quant, k_scale = _mxfp4(g, (n,))
        k_start, k_end, pools = [], [], []
        for i in range(requests):
            for j in range(per):
                visible = L - (per - 1 - j)
                k_start.append(starts[i])
                k_end.append(starts[i] + visible)
                pools.append(_pool(g, visible))
        q, q_scale = _mxfp4(g, (contract.NUM_CONSUMERS, rows, contract.NUM_HEADS))
        weights = torch.randn((contract.NUM_CONSUMERS, rows, contract.NUM_HEADS), generator=g).to(torch.bfloat16)
        cand = torch.stack(pools)
        sets.append(contract.PackedStep(
            q=q.to(device), q_scale=q_scale.to(device), k_quant=k_quant.to(device), k_scale=k_scale.to(device),
            weights=weights.to(device), k_start=torch.tensor(k_start, dtype=torch.int32, device=device),
            k_end=torch.tensor(k_end, dtype=torch.int32, device=device), candidate_blocks=cand.to(device)))
    return sets


def paged_sets(case: dict, seed: int, device: str) -> list:
    """Four paged input sets: `requests` requests of `rows_per_request` rows (row lengths rising by one to the final
    length), independent request caches in permuted physical pages of `page_kv` tokens."""
    import torch

    rows, final, page_kv = int(case["rows"]), int(case["visible"]), int(case["page_kv"])
    per = int(case.get("rows_per_request", 1))
    requests = rows // per
    page_bytes = page_kv * contract.PAGE_RECORD_BYTES
    stride = math.ceil(page_bytes / contract.PAGE_STRIDE_ALIGNMENT) * contract.PAGE_STRIDE_ALIGNMENT
    pages_per_request = math.ceil(final / page_kv)
    num_pages = requests * pages_per_request
    sets = []
    for s in range(INPUT_SETS):
        g = _rng(seed, case["id"], s)
        L = final - 3 * s
        storage = torch.zeros((num_pages, stride), dtype=torch.uint8)
        values, words = _mxfp4(g, (num_pages, page_kv))
        storage[:, : page_kv * contract.PACKED_BYTES] = values.view(torch.uint8).view(num_pages, -1)
        storage[:, page_kv * contract.PACKED_BYTES : page_bytes] = words.view(torch.uint8).view(num_pages, -1)
        physical = torch.randperm(num_pages, generator=g).to(torch.int32).view(requests, pages_per_request)
        lens, table, owner, pools = [], [], [], []
        for i in range(requests):
            for j in range(per):
                visible = L - (per - 1 - j)
                lens.append(visible)
                table.append(physical[i])
                owner.append(i)
                pools.append(_pool(g, visible))
        q, q_scale = _mxfp4(g, (contract.NUM_CONSUMERS, rows, 1, contract.NUM_HEADS))
        weights = torch.randn((contract.NUM_CONSUMERS, rows, contract.NUM_HEADS), generator=g).to(torch.bfloat16)
        storage = storage.to(device)
        kv = storage.as_strided((num_pages, page_kv, 1, contract.PAGE_RECORD_BYTES),
                                (stride, contract.PAGE_RECORD_BYTES, contract.PAGE_RECORD_BYTES, 1))
        sets.append(contract.PagedStep(
            q=q.to(device), q_scale=q_scale.to(device), kv_cache=kv, weights=weights.to(device),
            context_lens=torch.tensor(lens, dtype=torch.int32, device=device),
            block_table=torch.stack(table).to(device).contiguous(),
            row_indices=torch.tensor(owner, dtype=torch.int32, device=device),
            candidate_blocks=torch.stack(pools).to(device)))
    return sets


def time_units(unit, count: int, torch):
    """Wall and GPU seconds per work unit over `count` units."""
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    start.record()
    for i in range(count):
        unit(i)
    stop.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter_ns() - t0) / 1e9
    return wall / count, start.elapsed_time(stop) / 1e3 / count


def measure(unit, torch, repetitions: int | None = None) -> dict:
    """The protocol: warm-up, a repetition count fixed by a pilot (or given), three blocks."""
    t0 = time.perf_counter()
    for i in range(WARMUP_UNITS):
        unit(i)
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0
    pilots = []
    if repetitions is None:
        repetitions = 1
        while True:
            wall, _ = time_units(unit, repetitions, torch)
            pilots.append({"repetitions": repetitions, "wall_ms": wall * repetitions * 1e3})
            if wall * repetitions * 1e3 >= PILOT_MIN_MS or repetitions >= PILOT_MAX_REPETITIONS:
                break
            repetitions = min(repetitions * 2, PILOT_MAX_REPETITIONS)
    blocks = [time_units(unit, repetitions, torch) for _ in range(BLOCKS)]
    return {
        "warm_s": warm, "repetitions": repetitions, "pilots": pilots,
        "blocks": [{"wall_s": w, "gpu_s": g} for w, g in blocks],
        "median_wall_s": statistics.median(w for w, _ in blocks),
        "median_gpu_s": statistics.median(g for _, g in blocks),
    }


def unit_for(phase: str, sets: list, device: str):
    """The work unit over the rotated input sets, with the workspace carried between calls and one output buffer per
    set."""
    prefill, decode = contract.load_entry_points()
    outs = [contract.output_buffer(s.rows, device) for s in sets]
    state = {"workspace": None}

    def unit(i: int) -> None:
        s = sets[i % len(sets)]
        if phase == "prefill":
            state["workspace"] = contract.run_prefill(prefill, s, outs[i % len(sets)], state["workspace"])
        else:
            state["workspace"] = contract.run_decode(decode, s, outs[i % len(sets)], state["workspace"])

    unit.outputs = outs  # the output buffer of each input set, for checking what the last work unit on it wrote
    return unit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase", choices=("prefill", "decode"), required=True)
    ap.add_argument("--cases", type=Path, default=HERE / "examples.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    import torch

    torch.cuda.set_device(a.device)
    records = json.loads(a.cases.read_text())
    seed = int(records["seed"])
    kind = "packed" if a.phase == "prefill" else "paged"
    results = []
    for case in records["measurements"]:
        if case["kind"] != kind:
            continue
        sets = packed_sets(case, seed, a.device) if kind == "packed" else paged_sets(case, seed, a.device)
        problems = [p for s in sets for p in (contract.packed_problems(s) if kind == "packed"
                                              else contract.paged_problems(s))]
        if problems:
            raise SystemExit(f"{case['id']}: {sorted(set(problems))}")
        record = {"id": case["id"], **{k: case[k] for k in case if k not in ("id", "kind")},
                  **measure(unit_for(a.phase, sets, a.device), torch)}
        print(f"{case['id']}: {record['median_wall_s'] * 1e6:.1f} us wall, {record['median_gpu_s'] * 1e6:.1f} us GPU "
              f"per work unit ({record['repetitions']} units per block)", flush=True)
        results.append(record)
    a.output.write_text(json.dumps({"phase": a.phase, "device": torch.cuda.get_device_name(a.device),
                                    "results": results}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
