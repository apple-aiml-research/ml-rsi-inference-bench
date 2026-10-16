#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The served configuration of a running deployment, read from its launch log and its own endpoints."""
from __future__ import annotations

import ast
import re

from . import recipes
from .http import metrics, request

RANKS = {0, 1, 2, 3}
ENGINE = {key: re.compile(r"\b" + key + r"=([^,\s)]+)") for key in (
    "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "decode_context_parallel_size",
    "max_seq_len", "trust_remote_code", "quantization", "kv_cache_dtype", "enforce_eager", "enable_prefix_caching",
    "enable_chunked_prefill", "served_model_name", "dtype", "seed")}
COMPILE_MODE = re.compile(r"'mode': <CompilationMode\.(\w+)")
CUDAGRAPH_MODE = re.compile(r"'cudagraph_mode': <CUDAGraphMode\.(\w+)")
RANK = re.compile(r"\(Worker_TP(\d+)(?:_EP\d+)? pid=")
BACKEND = re.compile(r"Using AttentionBackendEnum\.(\w+) backend\.|Using (\w+) attention backend out of")
LAYOUT = re.compile(r"Using \w+ KV cache layout for (\w+) backend")
SPARSE = re.compile(r"MiniMax M3 sparse attention selected (\w+) \(kv_cache_dtype=(\w+)")
CHUNKED = re.compile(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)")
MEMORY = re.compile(r"The current --gpu-memory-utilization=([0-9.]+) is equivalent")
INFO_LABELS = re.compile(r'(\w+)="([^"]*)"')


def _literal(node):
    """``ast.literal_eval`` that also reads a dataclass repr ``Name(k=v, ...)`` as ``{"__class__": Name, k: v}`` (the
    argument record prints config objects, e.g. ``profiler_config``, that way)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.args:
        return {"__class__": node.func.id, **{k.arg: _literal(k.value) for k in node.keywords}}
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_literal(e) for e in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else items
    return ast.literal_eval(node)


def _cache_config_info(samples: dict[str, float]) -> dict:
    """The labels of the ``vllm:cache_config_info`` sample in ``http.metrics``' {sample name with labels: value}."""
    for name in samples:
        if name.startswith("vllm:cache_config_info{"):
            return dict(INFO_LABELS.findall(name[name.index("{"):name.rindex("}") + 1]))
    return {}


def served_configuration(log: str, deployment: str, *, base_url: str | None = None) -> dict:
    if deployment not in recipes.DEPLOYMENTS:
        raise ValueError(f"unknown deployment {deployment!r}")
    fp8 = "--kv-cache-dtype" in recipes.DEPLOYMENTS[deployment]
    backend_want = recipes.DENSE_ATTENTION_BACKEND[deployment]
    engines, args, departures = [], [], []
    ranks, rank_backends, rank_layouts, sparse, chunked, memory = set(), {}, {}, [], set(), set()
    compile_mode = cudagraph_mode = None
    for line in log.splitlines():
        if "Initializing a V1 LLM engine" in line and "with config:" in line:
            engines.append({k: (m[1] if (m := p.search(line)) else None) for k, p in ENGINE.items()})
            compile_mode = (m[1] if (m := COMPILE_MODE.search(line)) else None)
            cudagraph_mode = (m[1] if (m := CUDAGRAPH_MODE.search(line)) else None)
        if "non-default args:" in line and "APIServer" in line:
            try:
                args.append(_literal(ast.parse(line.split("non-default args:", 1)[1].strip(), mode="eval").body))
            except (ValueError, SyntaxError):
                departures.append("unreadable launch argument record")
        rank = RANK.search(line)
        if rank:
            r = int(rank[1])
            ranks.add(r)
            if m := BACKEND.search(line):
                rank_backends.setdefault(r, set()).add(m[1] or m[2])
            if m := LAYOUT.search(line):
                rank_layouts.setdefault(r, set()).add(m[1])
            if m := SPARSE.search(line):
                sparse.append({"rank": r, "impl": m[1], "kv_cache_dtype": m[2]})
        if m := CHUNKED.search(line):
            chunked.add(int(m[1]))
        if m := MEMORY.search(line):
            memory.add(float(m[1]))

    if len(engines) != 1:
        departures.append(f"{len(engines)} engine configuration records (expected one)")
    for row in engines:
        want = {"tensor_parallel_size": "4", "pipeline_parallel_size": "1", "data_parallel_size": "1",
                "decode_context_parallel_size": "1", "max_seq_len": str(recipes.MAX_MODEL_LEN),
                "trust_remote_code": "False", "quantization": "mxfp8", "enable_prefix_caching": "True",
                "enable_chunked_prefill": "True", "served_model_name": recipes.SERVED_MODEL_NAME,
                "kv_cache_dtype": "fp8" if fp8 else "auto"}
        for key, value in want.items():
            if row[key] != value:
                departures.append(f"{key}: served {row[key]!r}, the recipe's {value!r}")
    if not args:
        departures.append("missing the API server's launch argument record")
    for row in args:
        if row.get("block_size") != 128:
            departures.append(f"block size {row.get('block_size')!r}, the recipe's 128")
        if row.get("language_model_only") is not True:
            departures.append("not served language-model-only")
        if row.get("trust_remote_code"):
            departures.append("remote code was enabled")
        if row.get("gpu_memory_utilization") != recipes.GPU_MEMORY_UTILIZATION:
            departures.append(f"gpu_memory_utilization {row.get('gpu_memory_utilization')!r}, the recipe's "
                              f"{recipes.GPU_MEMORY_UTILIZATION}")
        if backend_want and row.get("attention_backend") != backend_want:
            departures.append(f"attention backend {row.get('attention_backend')!r}, the recipe's {backend_want!r}")
        for key in ("reasoning_parser", "tool_call_parser"):
            if row.get(key) != "minimax_m3":
                departures.append(f"{key} {row.get(key)!r}, the recipe's 'minimax_m3'")
        if row.get("enable_auto_tool_choice") is not True:
            departures.append("automatic tool choice not enabled")
        for key in ("enable_expert_parallel", "enforce_eager"):
            if row.get(key):
                departures.append(f"{key} was requested")
    if chunked != {8192}:
        departures.append(f"max_num_batched_tokens {sorted(chunked)}, expected [8192]")
    if memory != {recipes.GPU_MEMORY_UTILIZATION}:
        departures.append(f"workers report gpu_memory_utilization {sorted(memory)}, the recipe's "
                          f"{recipes.GPU_MEMORY_UTILIZATION}")
    if ranks != RANKS:
        departures.append(f"worker ranks {sorted(ranks)}, expected the four TP ranks")
    if backend_want:
        missing = sorted(r for r in RANKS if backend_want not in rank_backends.get(r, set()) | rank_layouts.get(r, set()))
        if missing:
            departures.append(f"ranks {missing} did not report the {backend_want} attention backend")
    cache_info, models = {}, None
    if base_url is not None:
        body = request(base_url, "/v1/models", timeout=300)
        models = [{"id": d.get("id"), "max_model_len": d.get("max_model_len")} for d in body.get("data", [])]
        if models != [{"id": recipes.SERVED_MODEL_NAME, "max_model_len": recipes.MAX_MODEL_LEN}]:
            departures.append(f"/v1/models serves {models}")
        cache_info = _cache_config_info(metrics(base_url))
        if cache_info:
            if cache_info.get("block_size") != "128":
                departures.append(f"cache_config_info block_size {cache_info.get('block_size')!r}")
            if cache_info.get("cache_dtype") != ("fp8" if fp8 else "auto"):
                departures.append(f"cache_config_info cache_dtype {cache_info.get('cache_dtype')!r}")
            if cache_info.get("enable_prefix_caching") != "True":
                departures.append("cache_config_info: prefix caching not enabled")
    return {"deployment": deployment, "engine_config": engines, "launch_arguments": args,
            "worker_ranks": sorted(ranks), "rank_attention_backends": {r: sorted(v) for r, v in rank_backends.items()},
            "rank_kv_layouts": {r: sorted(v) for r, v in rank_layouts.items()}, "sparse_selection": sparse,
            "max_num_batched_tokens": sorted(chunked), "gpu_memory_utilization": sorted(memory),
            "compilation_mode": compile_mode, "cudagraph_mode": cudagraph_mode, "models": models,
            "cache_config_info": cache_info, "departures": sorted(set(departures))}
