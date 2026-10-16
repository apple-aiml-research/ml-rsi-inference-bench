#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Five alternating paired blocks, preserving every component draw.

Public code owns the complete operation, input rotation and CUDA timing; the
private driver `infb_components.probes.component_timing` invokes it with the
trusted public primitives. This runner supplies isolated source roots, host
brackets and the comparison. Only an independently invalid host bracket permits
one symmetric pair redraw. Missing, killed or malformed draws cannot retain a
successful earlier result, and a candidate failure keeps every acquired draw.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

from infb_efficiency import canary


COUNTS = (1, 8, 64, 192, 256, 1024, 4096)
CELLS = {(n, family, n in COUNTS[:4]) for n in COUNTS for family in ('balanced', 'skewed')}


class HostFault(RuntimeError):
    """The independent reference/host failed, before a candidate verdict."""


def distributed_argv(python, module):
    # torchrun normally starts workers with a fresh interpreter and drops the
    # parent's -S. Name the worker interpreter explicitly so candidate .pth or
    # sitecustomize hooks cannot enter reference workers before our source roots.
    return [python, '-S', '-m', 'torch.distributed.run', '--standalone',
            '--nproc-per-node=4', '--no-python', python, '-S', '-m', module]


def bindings(tree):
    path = Path(tree)/'gpt_oss_binding.json'
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    assert isinstance(data, dict) and set(data) <= {'expert', 'attention'}
    for value in data.values():
        assert isinstance(value, str) and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', value)
    return data


PRIVATE_ROOT = Path(__file__).resolve().parents[1]  # /tests: infb_components, never the agent image


def environment(engine, kernels, protocols, python, cache):
    env = {k:v for k,v in os.environ.items() if not k.startswith('PYTHON')}
    version = f'python{sys.version_info.major}.{sys.version_info.minor}'
    packages = Path(python).parent.parent/'lib'/version/'site-packages'
    assert packages.is_dir(), f'missing pinned packages: {packages}'
    # Trusted public primitives, then the private probes, then the engine under test.
    env['PYTHONPATH'] = os.pathsep.join(map(str, [protocols, PRIVATE_ROOT, engine/'python', kernels, packages]))
    env.update(TRITON_CACHE_DIR=str(cache/'triton'), TVM_FFI_CACHE_DIR=str(cache/'tvm-ffi'))
    return env


def execute(argv, env, log, timeout):
    with log.open('w') as output:
        process = subprocess.Popen(argv, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Teardown: TERM, then KILL after 10 s.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            raise TimeoutError(f'component draw exceeded {timeout}s; {log}')
        finally:
            # A child may exit while leaving workers in its process group.
            # Reap only the group created for this draw before another GPU job.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if code:
            raise RuntimeError(f'component draw exited {code}; {log}')


def draw_records(path):
    records = [json.loads((path/f'rank-{rank}.json').read_text()) for rank in range(4)]
    for rank, record in enumerate(records):
        assert record['rank'] == rank and record['schema'] == 'gpt-oss-expert-timing/2'
        rows = record['rows']
        assert len(rows) == len(CELLS)
        assert {(r['count'],r['family'],r['graph']) for r in rows} == CELLS
        for row in rows:
            assert math.isfinite(row['slowest_rank_ms']) and row['slowest_rank_ms'] > 0
            assert row['repeats'] >= 128 and row['rotating_slots'] == 8
            assert row['setup_peak_workspace_bytes'] >= 0 and row['steady_peak_workspace_bytes'] >= 0
    values = [{(r['count'],r['family'],r['graph']):r['slowest_rank_ms'] for r in x['rows']}
              for x in records]
    assert all(row == values[0] for row in values), 'rank timing is not the common completed boundary'
    return records


def partial_records(path):
    records = []
    for rank in range(4):
        try:
            records.append(json.loads((path/f'rank-{rank}.json').read_text()))
        except (OSError, ValueError):
            records.append(None)
    return records


def compare(records):
    kept = [r for r in records if r['valid']]
    assert len(kept) == 5 and {r['block'] for r in kept} == set(range(5))
    result = []
    for count, family, graph in sorted(CELLS):
        samples = {}
        for side in ('reference', 'candidate'):
            samples[side] = [next(r['slowest_rank_ms'] for r in pair['sides'][side]['ranks'][0]['rows']
                                  if (r['count'],r['family'],r['graph']) == (count,family,graph))
                             for pair in kept]
        ref, candidate = (statistics.median(samples[k]) for k in ('reference','candidate'))
        result.append({'count':count, 'family':family, 'graph':graph, 'samples_ms':samples,
                       'reference_median_ms':ref, 'candidate_median_ms':candidate,
                       'ratio':ref/candidate})
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    record = {'schema':'gpt-oss-component-pairs/1', 'status':'running', 'pairs':[], 'rows':[]}
    destination = args.output/'paired.json'
    start = time.monotonic()
    def save():
        record['seconds'] = time.monotonic()-start
        destination.write_text(json.dumps(record,indent=2)+'\n')
    tests = Path(__file__).resolve().parents[1]
    canary_argv = [args.python,'-I','-S',str(tests/'trusted_python.py'),
                   str(tests/'infb_efficiency/canary.py'),'read']
    clean = {k:v for k,v in os.environ.items() if not k.startswith('PYTHON')}
    first = None
    try:
        for block in range(5):
            for attempt in range(2):
                pair = {'block':block,'attempt':attempt,'valid':False,'sides':{}}
                record['pairs'].append(pair)
                order = ('reference','candidate') if block%2 == 0 else ('candidate','reference')
                for side in order:
                    record['active_side'] = 'host'
                    save()
                    before = canary.take(canary_argv,clean)
                    if not before.get('ok'):
                        raise HostFault(f'host canary unavailable: {before.get("error")}')
                    if first is None:
                        first = before
                        record['first_canary'] = first
                    output = args.output/f'block-{block}-attempt-{attempt}-{side}'
                    entry = pair['sides'][side] = {'output':str(output),'before':before}
                    save()
                    engine = getattr(args,side)
                    kernels = getattr(args,side+'_kernels') or engine/'third_party/triton_kernels'
                    env = environment(engine,kernels,args.protocols,args.python,args.output/'cache'/side)
                    argv = [*distributed_argv(args.python, 'infb_components.probes.component_timing'),
                            '--output',str(output),'--seed',str(args.seed+1009*block),'--repeats','128']
                    if block == 0:
                        argv += ['--check','--profile','--witness']
                    selected_binding = bindings(engine).get('expert')
                    if selected_binding:
                        argv += ['--binding', selected_binding]
                    record['active_side'] = side
                    save()
                    try:
                        execute(argv,env,output.with_suffix('.log'),args.draw_timeout)
                        entry['ranks'] = draw_records(output)
                    except Exception as exc:
                        entry['error'] = f'{type(exc).__name__}: {exc}'
                        # Whatever complete rank records the failed draw left are retained.
                        entry['partial_ranks'] = partial_records(output)
                        if side == 'reference':
                            raise HostFault(f'reference component draw failed: {exc}') from exc
                        raise
                    record['active_side'] = 'host'
                    save()
                    after = entry['after'] = canary.take(canary_argv,clean)
                    if not after.get('ok'):
                        raise HostFault(f'host canary unavailable: {after.get("error")}')
                    entry['host_valid'] = all(canary.compare(first,x,args.canary_tolerance)['valid']
                                              for x in (before,after))
                    save()
                pair['valid'] = all(x['host_valid'] for x in pair['sides'].values())
                save()
                if pair['valid']:
                    break
            else:
                raise HostFault('host bracket invalid after the one symmetric pair redraw')
        record['rows'] = compare(record['pairs'])
        record['status'] = 'complete'
        record['active_side'] = None
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',rows=[],
                      fault='host' if isinstance(exc, HostFault) else 'candidate')
        raise
    finally:
        save()
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--reference-kernels',type=Path)
    p.add_argument('--candidate-kernels',type=Path)
    p.add_argument('--protocols',type=Path,default=Path('/tests/public_eval'))
    p.add_argument('--python',default='/opt/sglang/bin/python')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--canary-tolerance',type=float,required=True)
    p.add_argument('--draw-timeout',type=int,default=1800)
    a=p.parse_args()
    assert 0 < a.canary_tolerance < 1
    run(a)


if __name__ == '__main__':
    main()
