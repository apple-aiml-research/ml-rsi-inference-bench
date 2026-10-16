#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Read the stated ordinary/MTP3 configuration from actual engine launch logs."""
from __future__ import annotations

import ast
import re

EXPECTED = {
    'tensor_parallel_size': '4', 'pipeline_parallel_size': '1',
    'data_parallel_size': '1', 'decode_context_parallel_size': '1',
    'dtype': 'torch.bfloat16', 'max_seq_len': '262144', 'quantization': 'fp8',
    'enforce_eager': 'False', 'enable_prefix_caching': 'True',
    'enable_chunked_prefill': 'True', 'seed': '0', 'moe_backend': "'auto'",
    'mode': 'VLLM_COMPILE', 'cudagraph_mode': 'FULL_AND_PIECEWISE',
}
FIELDS = {key: re.compile(r'\b' + key + r'=([^,\s)]+)') for key in EXPECTED}
FIELDS.update({
    'mode': re.compile(r"'mode': <CompilationMode\.(\w+)"),
    'cudagraph_mode': re.compile(r"'cudagraph_mode': <CUDAGraphMode\.(\w+)"),
    'kv_cache_dtype': re.compile(r'\bkv_cache_dtype=([^,\s)]+)'),
})
COMPILE_EVIDENCE = ('Dynamo bytecode transform time', 'Compiling a graph for compile range',
                    'Directly load the compiled graph(s) for compile range',
                    'Directly load AOT compilation from path')
BREAKABLE = ('Breakable CUDA graph enabled', 'Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1')


def served_configuration(launch_log, recipe='ordinary'):
    if recipe not in ('ordinary', 'mtp3'):
        raise ValueError('unknown serving recipe')
    engines, args, gdn, ranks, departures = [], [], [], set(), []
    compile_lines = graph_lines = ep_lines = breakable_lines = aligned_lines = 0
    scheduler_limits, moe_kernels = set(), set()
    for line in launch_log.splitlines():
        if 'Initializing a V1 LLM engine' in line and 'with config:' in line:
            row = {key: (m[1] if (m := pattern.search(line)) else None) for key,pattern in FIELDS.items()}
            if recipe=='ordinary':
                row['speculative_recipe'] = bool(re.search(r'\bspeculative_config=None(?:,|\s)',line))
            else:
                spec = re.search(r'speculative_config=SpeculativeConfig\((.*?)\), tokenizer=',line)
                row['speculative_recipe'] = bool(spec and "method='mtp'" in spec[1] and
                    re.search(r'\bnum_spec_tokens=3(?:,|$)',spec[1]))
            engines.append(row)
        if 'non-default args:' in line and 'APIServer' in line:
            try: args.append(ast.literal_eval(line.split('non-default args:',1)[1].strip()))
            except (ValueError,SyntaxError): departures.append('unreadable launch argument record')
        rank = re.search(r'\(Worker_TP(\d+)(?:_EP\d+)? pid=',line)
        if rank: ranks.add(int(rank[1]))
        if rank and rank[1]=='0':
            compile_lines += int(any(text in line for text in COMPILE_EVIDENCE))
            if match := re.search(r'Using (\w+) GDN prefill kernel \(requested=(\w+)',line):
                gdn.append({'kernel':match[1],'requested':match[2]})
        graph_lines += int('Graph capturing finished' in line)
        ep_lines += int('Expert parallelism is enabled' in line or bool(re.search(r'Worker_TP\d+_EP\d+',line)))
        breakable_lines += int(any(text in line for text in BREAKABLE))
        aligned_lines += int("Mamba cache mode is set to 'align'" in line)
        if match := re.search(r'Chunked prefill is enabled with max_num_batched_tokens=(\d+)',line):
            scheduler_limits.add(int(match[1]))
        if match := re.search(r'Using (\w+) Fp8 MoE backend',line): moe_kernels.add(match[1])
    if not engines: departures.append('missing engine configuration')
    for row in engines:
        for key,want in EXPECTED.items():
            if row[key]!=want: departures.append(f'{key}: observed {row[key]!r}, required {want!r}')
        if row['kv_cache_dtype'] not in ('auto','bfloat16'):
            departures.append('the required BF16 KV cache was not configured')
        if not row['speculative_recipe']: departures.append(f'engine did not configure the {recipe} recipe')
    if not args: departures.append('missing API-server launch argument record')
    for row in args:
        if row.get('gdn_prefill_backend')!='flashinfer': departures.append('GDN prefill was not requested as FlashInfer')
        if row.get('enable_expert_parallel',False): departures.append('expert parallelism was requested')
        if row.get('prefill_context_parallel_size',1)!=1: departures.append('prefill context parallelism was requested')
        if row.get('mamba_cache_mode','align')!='align': departures.append('Mamba state cache is not aligned')
        if recipe=='mtp3' and row.get('speculative_config')!={'method':'mtp','num_speculative_tokens':3}:
            departures.append('the API-server MTP request is not the stated MTP3 configuration')
        if recipe=='ordinary' and row.get('speculative_config') is not None:
            departures.append('speculation was requested for ordinary serving')
    if not gdn or any(row!={'kernel':'FlashInfer','requested':'flashinfer'} for row in gdn):
        departures.append('missing or incorrect actual FlashInfer GDN prefill evidence')
    if ranks!={0,1,2,3}: departures.append(f'worker ranks are {sorted(ranks)}, expected all four TP ranks')
    if not compile_lines: departures.append('rank 0 recorded no actual torch.compile work or cache load')
    if not graph_lines: departures.append('no completed CUDA graph capture')
    if ep_lines: departures.append('expert parallelism was enabled')
    if breakable_lines: departures.append('breakable graphs were enabled')
    if not aligned_lines and not any(row.get('mamba_cache_mode')=='align' for row in args):
        departures.append('aligned Mamba state caching was not witnessed')
    return {'recipe':recipe,'engine_config_lines':engines,'named_backends_top_level':args,
            'gdn_prefill_rank0':gdn,'worker_ranks':sorted(ranks),
            'compile_evidence_lines':compile_lines,'graph_capture_lines':graph_lines,
            'expert_parallel_lines':ep_lines,'breakable_lines':breakable_lines,
            'scheduler_token_limits':sorted(scheduler_limits),'moe_kernels_logged':sorted(moe_kernels),
            'departures':sorted(set(departures))}
