#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""CPU-only publication tail, entered by the isolated clock after quiescence.

Interrupted GPU work never donates a passing gate. Only a recorded native
parallel fault/recovery stall is attributable to the candidate here; an
unknown recorder/environment/clock failure retains the -1 crash sentinel.
No engine, reference generation, tensor library or candidate module is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(value, root):
    path = Path(value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('receipt path escapes current artifact root')
    return path


def parallel_stalls(artifacts, run_id):
    """Conservative attribution from native observer + unfinished API receipt."""
    result = []
    summary = artifacts / 'sd3_day0_parallel.json'
    if not summary.is_file():
        return result
    res = read(summary)
    if not run_id or res.get('run_id') != run_id:
        return result
    work = local_path(res['work'], artifacts)
    for mode in ('tp2', 'sp2'):
        rp, ep = work / mode / 'run.json', work / mode / 'record.jsonl'
        if not rp.is_file() or not ep.is_file():
            continue
        run = read(rp)
        inflight = run.get('inflight') or {}
        if (run.get('verifier_run_id') != run_id or run.get('mode') != mode
                or run.get('built') is not True or run.get('producer_complete') is True
                or inflight.get('key') not in ('fault', 'after_fault')):
            continue
        t0 = inflight.get('t0')
        if type(t0) not in (int, float) or not math.isfinite(t0):
            continue
        installed, completed, faults = set(), set(), []
        size = 0
        with ep.open('rb') as f:
            for line in f:
                size += len(line)
                if size > 64 * 1024 * 1024 or len(line) > 2 * 1024 * 1024:
                    raise ValueError('native evidence exceeds bounded tail reader')
                e = json.loads(line)
                if not isinstance(e, dict):
                    raise ValueError('native evidence event must be an object')


                if e.get('event') not in ('batch_observer_installed', 'batch_dit', 'fault_fired'):
                    continue
                rank = e.get('rank')
                if not isinstance(rank, dict):
                    continue
                r, world = rank.get('dist_rank'), rank.get('dist_world')
                if type(r) is not int or r not in (0, 1) or world != 2:
                    continue
                if e.get('event') == 'batch_observer_installed':
                    installed.add(r)
                if e.get('event') == 'batch_dit' and e.get('completed') is True:
                    completed.add(r)
                if e.get('event') == 'fault_fired' and e.get('phase') == 'after_dit':
                    faults.append(e)
        if installed != {0, 1} or completed != {0, 1} or len(faults) != 1:
            continue
        fault = faults[0]
        ft = fault.get('t')
        if type(ft) not in (int, float) or not math.isfinite(ft):
            continue
        if inflight['key'] == 'fault' and ft < t0:
            continue
        if inflight['key'] == 'after_fault':
            prior = run.get('readings', {}).get('fault') or {}
            if prior.get('raised') is not True or not prior.get('t0', math.inf) <= ft <= prior.get('t1', -math.inf):
                continue
        result.append(dict(mode=mode, gate='lifecycle_under_parallel_' + mode,
                           inflight=inflight, fault=fault,
                           recorder_healthy_ranks=sorted(installed),
                           run_path=str(rp), run_sha256=sha(rp),
                           record_path=str(ep), record_sha256=sha(ep)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--body-result', type=Path, required=True)
    args = parser.parse_args()
    body = read(args.body_result)
    if body.get('quiescent') is not True:
        raise RuntimeError('publication forbidden before owned-worker quiescence')
    root = args.body_result.parent
    artifacts = Path(os.environ.get('INFB_ARTIFACTS_DIR', '/logs/artifacts'))
    verifier = Path(os.environ.get('INFB_VERIFIER_DIR', '/logs/verifier'))
    verifier.mkdir(parents=True, exist_ok=True)


    for name in ('reward.json', 'reward.txt'):
        path = verifier / name
        if path.exists():
            (root / ('body_' + name)).write_bytes(path.read_bytes())
            path.unlink()

    (verifier / 'sd3_qfe_report.json').unlink(missing_ok=True)
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines() if line]
    interrupted = [e for e in events if e.get('event') == 'end' and e.get('phase') == 'work'
                   and e.get('returncode') not in (None, 0)]
    ended = {e.get('segment') for e in events if e.get('event') == 'end' and e.get('phase') == 'work'}
    unfinished = [dict(e, status='missing_segment_end', returncode=75)
                  for e in events if e.get('event') == 'start' and e.get('phase') == 'work'
                  and e.get('segment') not in ended]
    interrupted.extend(unfinished)


    import verifier_clock as C
    for e in interrupted:
        C.invalidate_changed(root, e['segment'], verifier, {}, e['status'],
                             set(e.get('owned_gate_ids') or []))
    attribution_error = None
    try:
        stalls = parallel_stalls(artifacts, os.environ.get('SD3_VERIFIER_RUN_ID'))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        stalls = []
        attribution_error = f'{type(exc).__name__}: {exc}'[:500]
    failed_cleanup = [e for e in interrupted if e.get('status') == 'cleanup_unproven']
    known_stalls = {s['gate'] for s in stalls}


    attributable = (body.get('status') in ('failed', 'timed_out') and bool(stalls)
        and not body.get('error') and not body.get('journal_error')
        and all(set(e.get('owned_gate_ids') or []) & known_stalls for e in failed_cleanup)
        and not any(e.get('status') in ('instrument_error', 'cancelled', 'launch_error') for e in interrupted))
    unsafe = [e for e in interrupted if e.get('status') in
              ('instrument_error', 'cancelled', 'launch_error', 'cleanup_unproven', 'residual_processes',
               'missing_segment_end')]


    normal = (body.get('returncode') == 0 and body.get('term', 0) == 0 and body.get('kill', 0) == 0
              and not unsafe and attribution_error is None)
    if unfinished or any(e.get('status') == 'residual_processes' for e in interrupted):
        attributable = False
    outcome = dict(schema='sd3-verifier-outcome-v1', body=body, quiescent=True,
        publication='normal' if normal else ('candidate_parallel_recovery_timeout' if attributable else 'void'),
        candidate_stalls=stalls, attribution_error=attribution_error,
        interrupted_segments=interrupted, unresolved_execution_segments=unsafe,
        missing_gates_receive_credit=False)
    C.atomic_json(verifier / 'execution_outcome.json', outcome)
    if not normal and not attributable:
        print('[sd3-tail] unclassified verifier/environment failure; retain void result', flush=True)
        return 75
    from _common import GateResult, write_gate
    for s in stalls:
        write_gate(GateResult(gate=s['gate'], passed=False, score=0.0,
            error='native parallel fault/recovery request did not finish within its assigned segment',
            details=dict(execution_outcome=s, not_evaluated=False, incomplete_native_sequence=True)))
    env = dict(os.environ, TESTS_DIR=str(HERE), INFB_TESTS_DIR=str(HERE),
               VERIFIER_DIR=str(verifier), INFB_VERIFIER_DIR=str(verifier),
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)


    for script, argv in (('integrity_final.py', []), ('image_score_profiles.py', ['prepare']),
                         ('gates_to_ctrf.py', []), ('grader.py', ['grade']),
                         ('image_score_profiles.py', ['annotate'])):
        subprocess.run([sys.executable, '-I', '-S', str(HERE / script), *argv],
                       env=env, cwd=verifier, check=True)
    reward = read(verifier / 'reward.json')
    cfg = read(HERE / 'config.json')
    ctrf = read(verifier / 'ctrf.json')
    assert reward['f2p_total'] == len(cfg['f2p_node_ids'])
    assert reward['p2p_total'] == len(cfg['p2p_node_ids'])
    assert len(ctrf['results']['tests']) == len(cfg['f2p_node_ids']) + len(cfg['p2p_node_ids'])
    assert all(read(verifier / 'gates' / (g + '.gate.json')).get('details', {}).get('final_integrity_checked') is True
               for g in cfg['integrity_node_ids'])

    subprocess.run([sys.executable, '-I', '-S', str(HERE / 'layered_grader.py')],
                   env=env, cwd=verifier, check=False)


    subprocess.run([sys.executable, '-I', '-S', str(HERE / 'qfe_report.py'), '--verifier-dir', str(verifier),
                    '--tests-dir', str(HERE)], env=env, cwd=verifier, check=False)
    C.atomic_json(root / 'publication.json', dict(completed=True, quiescent=True,
        reward_sha256=sha(verifier / 'reward.json'), ctrf_sha256=sha(verifier / 'ctrf.json'),
        outcome_sha256=sha(verifier / 'execution_outcome.json'), cpu_only=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
