#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Default verifier adapter for the engine's actual GPT-OSS expert module."""

import hashlib
from contextlib import contextmanager
import inspect
import os
from pathlib import Path
from types import FunctionType, MethodType, ModuleType

import torch


def initialize(model_path="/model", custom_all_reduce=True):
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel, set_custom_all_reduce,
    )
    from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend
    from sglang.srt.managers.schedule_batch import GLOBAL_SERVER_ARGS_KEYS, global_server_args_dict
    from sglang.srt.server_args import ServerArgs

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise ValueError("expert qualification requires exactly four ranks")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(8)
    args = ServerArgs(model_path=model_path, tp_size=world,
                      disable_custom_all_reduce=not custom_all_reduce)
    global_server_args_dict.update({k: getattr(args, k) for k in GLOBAL_SERVER_ARGS_KEYS})
    global_server_args_dict.update(enable_triton_kernel_moe=True,
                                  moe_a2a_backend=MoeA2ABackend.STANDARD,
                                  deepep_mode=DeepEPMode.AUTO)
    set_custom_all_reduce(custom_all_reduce)
    init_distributed_environment(world, rank, local_rank=local_rank, timeout=1800)
    initialize_model_parallel(world)
    complete_collectives(custom_all_reduce)
    return rank


def complete_collectives(custom_all_reduce=True, tokens=(1, 4096), hidden=2880):
    """Finish distributed initialization before any baseline is read."""
    import gc
    import torch.distributed as dist
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    dist.barrier()
    dist.all_reduce(torch.zeros(1, device="cuda"), op=dist.ReduceOp.MAX)
    graphs = []
    for custom in sorted({custom_all_reduce, False}, reverse=True):
        with collective(custom):
            for count in tokens:
                x = torch.zeros(count, hidden, dtype=torch.bfloat16, device="cuda")
                tensor_model_parallel_all_reduce(x)
                graph = torch.cuda.CUDAGraph()
                with capture_graph(graph):
                    tensor_model_parallel_all_reduce(x)
                graph.replay()
                graphs.append(graph)
    torch.cuda.synchronize()
    dist.barrier()
    del graphs
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def capture_graph(graph):
    """Use the engine's graph registration and collective stream context."""
    from sglang.srt.distributed import get_tp_group
    with get_tp_group().graph_capture() as context:
        with torch.cuda.graph(graph, stream=context.stream):
            yield


@contextmanager
def collective(custom):
    """Match the serving recipe's collective, including inside graph capture."""
    from sglang.srt.distributed import get_tp_group
    communicator = get_tp_group().ca_comm
    previous = communicator.disabled if communicator is not None else None
    if communicator is not None and not custom:
        communicator.disabled = True
    try:
        yield
    finally:
        if communicator is not None:
            communicator.disabled = previous


def source_identity(obj):
    path = Path(inspect.getfile(type(obj))).resolve()
    return {"class": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def implementation_identity(module):
    """Observe the bound object; an adapter does not certify its own origin."""
    result = source_identity(module)
    if hasattr(module, 'quant_method'):
        result['quant_method'] = source_identity(module.quant_method)
    return result


def live_storages(root, excluding=None):
    """Inspect reachable tensor storage, including quant-method/layout objects."""
    seen, storages = set(), {}
    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr())
                storages[key] = max(storages.get(key, 0), storage.nbytes())
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list, set)):
            for child in value:
                visit(child)
        elif not isinstance(value, (type, ModuleType, FunctionType, MethodType)) and hasattr(value, "__dict__"):
            visit(vars(value))
    visit(root)
    if excluding is not None:
        included = storages.copy()
        seen.clear()
        storages.clear()
        visit(excluding)
        storages = {key: size for key, size in included.items() if key not in storages}
    by_device = {}
    for (device, start), size in storages.items():
        by_device.setdefault(device, []).append((start, start + size))
    total = ranges = 0
    for spans in by_device.values():
        end = None
        for start, stop in sorted(spans):
            if end is None or start >= end:
                total += stop - start
                ranges += 1
                end = stop
            elif stop > end:
                total += stop - end
                end = stop
    return {"bytes": total, "allocations": len(storages), "ranges": ranges}


class NativeExperts:
    @staticmethod
    def locate(model):
        """Identify actual loaded expert modules; no timing/correctness claims."""
        return [(index, layer.mlp.experts) for index, layer in enumerate(model.model.layers)]

    def __init__(self, weights):
        from sglang.srt.models.gpt_oss import GptOssConfig, GptOssForCausalLM, GptOssSparseMoeBlock
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config

        experts, twice_intermediate, half_hidden = weights.gate_up.shape
        config = GptOssConfig(hidden_size=2*half_hidden, intermediate_size=twice_intermediate//2,
                              num_local_experts=experts, num_experts_per_tok=4,
                              hidden_act="silu", hidden_act_alpha=1.702, swiglu_limit=7,
                              torch_dtype=torch.bfloat16, num_attention_heads=64)
        with torch.device("cuda"):
            block = GptOssSparseMoeBlock(0, config, Mxfp4Config())
        holder = torch.nn.Module()
        holder.config = config
        holder.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.mlp = block
        holder.model.layers = torch.nn.ModuleList([layer])
        prefix = "model.layers.0.mlp.experts."
        pairs = [("gate_up_proj_blocks", weights.gate_up),
                 ("gate_up_proj_scales", weights.gate_up_scale),
                 ("down_proj_blocks", weights.down),
                 ("down_proj_scales", weights.down_scale),
                 ("gate_up_proj_bias", weights.gate_up_bias),
                 ("down_proj_bias", weights.down_bias)]
        GptOssForCausalLM._load_mxfp4_experts_weights(holder, ((prefix+n, w) for n, w in pairs))
        block.experts.quant_method.process_weights_after_loading(block.experts)
        self.module = block.experts
        self.topk = block.topk
        self.experts = experts
        self.identity = implementation_identity(self.module)

    def __call__(self, x, ids, probabilities):
        from sglang.srt.distributed import tensor_model_parallel_all_reduce
        logits = torch.full((x.shape[0], self.experts), -torch.inf, device=x.device)
        logits.scatter_(1, ids.long(), probabilities.log())
        routing = self.topk(x, logits)
        result = self.module(hidden_states=x, topk_output=routing)
        return tensor_model_parallel_all_reduce(result)
