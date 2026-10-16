#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fixed, bounded qualification of the declared ordinary GPT-OSS contract.

The parent imports only private checking code. Submitted engine code runs in
children with explicit source roots; raw results are retained alongside JUnit.
Reference failures, held-out data outside the admitted domain and independently
invalid host brackets exit 3, never zero. Every node records its status --
pass, candidate_fail, blocked_by:<cause> -- and its separate facts, and the
record is persisted before the verdict is raised.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import requests

from infb_components import gates, observations, paired

CLASS = 'infb_components.test_contract.TestContract'
NODES = ('test_expert_math', 'test_attention_math', 'test_complete_expert_performance',
         'test_harmony_protocol', 'test_harmony_roundtrip', 'test_native_production_binding')


class Blocked(Exception):
    """A dependent observation whose prerequisite is unavailable: blocked, not a candidate defect."""
    def __init__(self, by, message):
        super().__init__(message)
        self.by = by


class Failed(AssertionError):
    """A candidate failure carrying the facts that were established."""
    def __init__(self, message, facts):
        super().__init__(message)
        self.facts = facts


class Reports:
    def __init__(self, output):
        self.output = output
        self.rows = []

    def check(self, name, action):
        assert name in NODES and not any(r['name'] == name for r in self.rows)
        start = time.monotonic()
        row = {'name': name, 'ok': False}
        try:
            row['result'] = action()
            row['ok'] = True
            row['status'] = 'pass'
        except (paired.HostFault, gates.ReferenceFault):
            raise
        except Blocked as exc:
            row['status'] = f'blocked_by:{exc.by}'
            row['error'] = f'{row["status"]}: {exc}'
        except Failed as exc:
            row['status'] = 'candidate_fail'
            row['facts'] = exc.facts
            row['error'] = f'candidate_fail: {exc}\n' + traceback.format_exc()
        except Exception:
            row['status'] = 'candidate_fail'
            row['error'] = 'candidate_fail: ' + traceback.format_exc()
        row['seconds'] = time.monotonic() - start
        self.rows.append(row)
        self.save()
        print(f"[components] {name}: {row['status'].upper()} ({row['seconds']:.1f}s)", flush=True)
        return row

    def save(self):
        (self.output / 'checks.json').write_text(json.dumps(self.rows, indent=2, default=str) + '\n')
        suite = ET.Element('testsuite', name=CLASS, tests=str(len(self.rows)),
                           failures=str(sum(not r['ok'] for r in self.rows)))
        for row in self.rows:
            case = ET.SubElement(suite, 'testcase', classname=CLASS, name=row['name'],
                                 time=str(row['seconds']))
            properties = ET.SubElement(case, 'properties')
            ET.SubElement(properties, 'property', name='status', value=row['status'])
            if not row['ok']:
                ET.SubElement(case, 'failure', message=row['status']).text = row['error']
        ET.ElementTree(suite).write(self.output / 'junit-components.xml', encoding='utf-8',
                                   xml_declaration=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def require_rows(rows, count):
    assert len(rows) == count and all(r.get('ok') is True for r in rows), rows
    return {'cases': count}


def terminate(process, teardown):
    """TERM the group, KILL after 10 s, and reap within the teardown allowance."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=max(1, teardown - 10))


@contextmanager
def server(engine, protocols, python, output, bindings, bound, perturb=False):
    output.mkdir(parents=True, exist_ok=False)
    env = paired.environment(engine, engine / 'third_party/triton_kernels', protocols,
                             python, output / 'cache')
    env['GPT_OSS_OBSERVE'] = str(output / 'observations')
    if bindings.get('expert'):
        env['GPT_OSS_BINDING'] = bindings['expert']
    if perturb:
        env['GPT_OSS_PERTURB_LAYER'] = '0'
    recipes = read_json(protocols / 'gpt_oss/recipes.json')
    argv = [python, '-S', '-m', 'infb_components.probes.observe_launch', '--model-path', recipes['model'],
            *recipes['S0'], '--host', '127.0.0.1', '--port', '28290']
    (output / 'launch.json').write_text(json.dumps({'argv': argv, 'engine': str(engine)}, indent=2) + '\n')
    base = 'http://127.0.0.1:28290'
    with (output / 'server.log').open('w') as log:
        process = subprocess.Popen(argv, cwd='/', env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            start = time.monotonic()
            end = start + bound['launch']
            while time.monotonic() < end:
                if process.poll() is not None:
                    raise RuntimeError(f'candidate server exited {process.returncode}; {output}/server.log')
                try:
                    if requests.get(base + '/health_generate', timeout=3).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(f'candidate server health exceeded {bound["launch"]}s')
            (output / 'ready.json').write_text(json.dumps({'ready_after_s': time.monotonic() - start}) + '\n')
            yield base
        finally:
            terminate(process, bound['teardown'])


def production_requests(base, output, protocols, cfg):
    """The structured graph-coverage record on this observed production server, and the
    recipe observed on it. Both consumers -- the binding gate and the largest-graph
    predicate -- read this one record; no graph size is a literal here."""
    from infb_components.probes.graph_coverage import coverage
    from infb_verify.test_served import TestServed, coverage_workloads, largest_graph_prompts, recipe_maximum
    workloads, recipe_max = coverage_workloads(), recipe_maximum()
    bar = cfg['infb']['bars']['largest_graph_decode_worst_request_T']
    record = coverage(base, output / 'observations', protocols,
                      largest_graph_prompts(max(spec['n'] for spec in workloads.values())),
                      workloads=workloads, recipe_max=recipe_max, bar=bar)
    (output / 'coverage.json').write_text(json.dumps(record, indent=2) + '\n')
    TestServed.base_url = base
    TestServed.log_path = output / 'server.log'
    served = []
    with (output / 'server.log').open() as log:
        TestServed._log = log
        result = unittest.TestResult()
        TestServed('test_recipe_conformance').run(result)
        served.append({'id': 'test_recipe_conformance', 'ok': result.wasSuccessful(),
                       'failures': [message for _, message in result.errors + result.failures]})
    (output / 'served-checks.json').write_text(json.dumps(served, indent=2) + '\n')
    return record, served


def input_logprobs(base, tokens):
    from gpt_oss.fidelity import candidate_logprobs
    response = requests.post(base + '/flush_cache', timeout=300)
    response.raise_for_status()
    return candidate_logprobs(base, tokens, seq_id='native-binding-witness')


def causal_change(normal, repeat, suppressed):
    assert len(normal) == len(repeat) == len(suppressed)
    changes, noise = [], []
    for a, b, c in zip(normal, repeat, suppressed):
        if a is None and b is None and c is None:
            continue
        assert all(type(x) in (int, float) and math.isfinite(x) for x in (a, b, c))
        noise.append(abs(a-b))
        changes.append(abs(a-c))
    assert len(changes) >= 200, 'missing fixed-input log probabilities'
    floor = max(0.0001, 10 * statistics.median(noise))
    change = statistics.median(changes)
    assert change > floor, f'served output does not depend on native expert: change={change}, floor={floor}'
    return {'positions': len(changes), 'median_change': change,
            'median_repeat_noise': statistics.median(noise), 'floor': floor}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read_json(args.config)
    c = cfg['infb']['components']
    tests = args.config.parent
    protocols = tests / 'public_eval'
    limits = gates.policy(protocols)
    bindings = paired.bindings(args.candidate)
    env = paired.environment(args.candidate, args.candidate / 'third_party/triton_kernels',
                             protocols, args.python, args.output / 'candidate-cache')
    reports = Reports(args.output)
    bound = c['operation_limits_sec']
    expert_binding = ['--binding', bindings['expert']] if bindings.get('expert') else []
    ranges = c['workspace_ranges']

    def command(argv, name, timeout):
        paired.execute(argv, env, args.output / (name + '.log'), timeout)

    def expert_math():
        results = []
        failures = []
        for case in c['expert_cases']:
            destination = args.output / ('math-' + case['id'])
            argv = [*paired.distributed_argv(args.python, 'infb_components.probes.expert_cases'),
                    '--output', str(destination), '--seed', str(c['seed']),
                    '--counts', ','.join(map(str, case['counts'])),
                    '--families', ','.join(case['families']), '--repeats', '8', *expert_binding]
            if 'checkpoint_layer' in case:
                argv += ['--checkpoint-layer', str(case['checkpoint_layer'])]
            else:
                argv += ['--weight-family', case['weight_family']]
            try:
                try:
                    command(argv, 'math-' + case['id'], bound['component_invocation'])
                except RuntimeError:
                    records = gates.read_ranks(destination)
                    if any(r.get('author_data_fault') for r in records):
                        gates.expert(records, expected_cells=set(), limits=limits['expert'],
                                     storage_limit=c['block_storage_bytes'], root=args.candidate)
                    raise
                expected = {(n, f, graph) for n in case['counts'] for f in case['families']
                            for graph in (False, True)}
                source = gates.expert(gates.read_ranks(destination), expected_cells=expected,
                                      limits=limits['expert'], storage_limit=c['block_storage_bytes'],
                                      root=args.candidate)
                results.append({'case': case['id'], 'source': source, 'cells': len(expected)})
            except gates.ReferenceFault:
                raise
            except Exception:
                failures.append({'case': case['id'], 'error': traceback.format_exc()})
            (args.output / 'expert-cases.json').write_text(
                json.dumps({'passed': results, 'failed': failures}, indent=2) + '\n')
        assert not failures, failures
        assert results
        return results

    reports.check('test_expert_math', expert_math)

    def attention():
        path = args.output / 'attention.json'
        argv = [args.python, '-S', '-m', 'infb_components.probes.attention_cases', '--output', str(path),
                '--seed', str(c['seed'])]
        if bindings.get('attention'):
            argv += ['--binding', bindings['attention']]
        try:
            command(argv, 'attention', bound['component_invocation'])
        except RuntimeError:
            if path.is_file() and read_json(path).get('author_data_fault'):
                gates.attention(read_json(path), limits['attention'])
            raise
        gates.attention(read_json(path), limits['attention'])
        return {'cases': 24}

    reports.check('test_attention_math', attention)

    component = {}

    def component_pairs():
        options = SimpleNamespace(reference=args.reference, candidate=args.candidate,
                                  reference_kernels=None, candidate_kernels=None,
                                  protocols=protocols, python=args.python,
                                  output=args.output / 'paired', seed=c['seed'],
                                  canary_tolerance=c['canary_tolerance'],
                                  draw_timeout=bound['component_invocation'])
        failure = None
        try:
            raw = paired.run(options)
        except paired.HostFault:
            raise
        except Exception as exc:
            # A candidate draw failed: keep every acquired observation and report the facts.
            failure = exc
            raw = read_json(options.output / 'paired.json')
        common = dict(limits=limits['expert'], storage_limit=c['block_storage_bytes'], ranges=ranges)
        if failure is None:
            # Independently verify the reference's math, resources and execution. Failure of
            # this control invalidates the comparison, not the candidate.
            reference = gates.component_facts(raw, rho=0, root=args.reference, side='reference', **common)
            broken = [k for k in ('measurement_complete', 'identity', 'numerical', 'resources', 'execution')
                      if not reference[k]['ok']]
            if broken:
                raise paired.HostFault(f"reference component health: {broken}: "
                                       f"{[reference[k].get('error') for k in broken]}")
        facts = gates.component_facts(raw, rho=c['rho'], root=args.candidate, **common)
        if failure is not None:
            facts['measurement_complete'] = {'ok': False, 'error': f'{type(failure).__name__}: {failure}'}
            facts['ok'] = False
        component['facts'] = facts
        (args.output / 'component-facts.json').write_text(json.dumps(facts, indent=2, default=str) + '\n')
        if not facts['ok']:
            failed = [k for k in ('measurement_complete', 'identity', 'numerical', 'resources', 'execution',
                                  'speed') if not facts[k]['ok']]
            raise Failed('; '.join(f"{k}: {facts[k].get('error')}" for k in failed), facts)
        return {'identity': facts['identity']['value'], 'ratios': facts['speed']['value']['ratios']}

    reports.check('test_complete_expert_performance', component_pairs)

    def protocol():
        path = args.output / 'harmony-fixtures.json'
        fixtures = read_json(tests / 'hidden/harmony-fixtures.json')
        argv = [args.python, '-S', '-m', 'infb_components.probes.harmony_fixtures', '--model', '/model',
                '--recipes', str(protocols / 'gpt_oss/recipes.json'),
                '--fixtures', str(tests / 'hidden/harmony-fixtures.json'),
                '--delay', '0.03', '--output', str(path)]
        command(argv, 'harmony-fixtures', bound['action'])
        return require_rows(read_json(path), len(fixtures) * 4)

    reports.check('test_harmony_protocol', protocol)

    normal_dir = args.output / 'production-normal'
    normal, repeat = [], []
    tokens = read_json(tests / 'hidden/fidelity/broad_01.json')['token_ids'][:256]
    production = {}
    try:
        with server(args.candidate, protocols, args.python, normal_dir, bindings, bound) as base:
            def roundtrip():
                from gpt_oss.tool_calls.probe import probe
                examples = read_json(tests / 'hidden/harmony-live.json')
                rows = probe(base, 'openai/gpt-oss-120b', examples,
                             args.output / 'harmony-live', workers=4)
                (args.output / 'harmony-live/results.json').write_text(json.dumps(rows, indent=2) + '\n')
                return require_rows(rows, len(examples) * 2)
            reports.check('test_harmony_roundtrip', roundtrip)
            try:
                production['coverage'], production['served'] = production_requests(base, normal_dir, protocols, cfg)
            except Exception:
                production['coverage_error'] = traceback.format_exc()
            normal, repeat = input_logprobs(base, tokens), input_logprobs(base, tokens)
    except Exception:
        production['server_error'] = traceback.format_exc()
        if not any(r['name'] == 'test_harmony_roundtrip' for r in reports.rows):
            def unavailable():
                raise Blocked('production_server', production['server_error'])
            reports.check('test_harmony_roundtrip', unavailable)

    def identity_source():
        facts = component.get('facts') or {}
        if (facts.get('identity') or {}).get('ok'):
            return facts['identity']['value'], 'component timing records'
        # Timing evidence is incomplete: one bounded identity-only invocation of the same
        # prepared native operation. It supplies no timing or numerical evidence.
        destination = args.output / 'identity-only'
        argv = [*paired.distributed_argv(args.python, 'infb_components.probes.component_timing'),
                '--output', str(destination), '--seed', str(c['seed']), '--identity-only', *expert_binding]
        try:
            command(argv, 'identity-only', bound['component_invocation'])
            return gates.identity_only(gates.read_ranks(destination), args.candidate), 'identity-only invocation'
        except Exception as exc:
            raise Blocked('component_identity', f'no valid identity from timing or the identity-only '
                                                f'invocation: {type(exc).__name__}: {exc}')

    def binding():
        if production.get('server_error'):
            raise Blocked('production_server', production['server_error'])
        source, origin = identity_source()
        if production.get('coverage_error'):
            raise Blocked('graph_coverage', production['coverage_error'])
        ranks = observations.collect(normal_dir / 'observations')
        coverage = production['coverage']
        runtime = gates.production(ranks, source=source, root=args.candidate,
                                   storage_limit=36*c['block_storage_bytes'],
                                   ordinary_allowance=c['ordinary_allowance_bytes'],
                                   coverage_record=coverage,
                                   concurrency=cfg['infb']['efficiency']['recipes']['TestHiddenAcceptance']['C'],
                                   layers=36)
        try:
            with server(args.candidate, protocols, args.python, args.output / 'production-suppressed',
                        bindings, bound, perturb=True) as base:
                suppressed = input_logprobs(base, tokens)
        except Exception as exc:
            raise Blocked('suppressed_server', f'{type(exc).__name__}: {exc}')
        witness = {'tokens': tokens, 'normal': normal, 'repeat': repeat, 'suppressed': suppressed}
        (args.output / 'causal-witness.json').write_text(json.dumps(witness, indent=2) + '\n')
        numerical = gates._fact(lambda: gates.coverage_numerical(coverage, bar=coverage['bar']))
        return {'identity': source, 'identity_origin': origin, 'runtime': runtime,
                'causal': causal_change(normal, repeat, suppressed),
                'coverage_numerical_reported': numerical, 'ranks': ranks}

    reports.check('test_native_production_binding', binding)
    assert {r['name'] for r in reports.rows} == set(NODES)
    return all(r['ok'] for r in reports.rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('/tests/config.json'))
    p.add_argument('--candidate', type=Path, default=Path('/app'))
    p.add_argument('--reference', type=Path, default=Path('/opt/infb-reference/tree'))
    p.add_argument('--python', default='/opt/sglang/bin/python')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        return 0 if run(args) else 1
    except (paired.HostFault, gates.ReferenceFault):
        traceback.print_exc()
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
