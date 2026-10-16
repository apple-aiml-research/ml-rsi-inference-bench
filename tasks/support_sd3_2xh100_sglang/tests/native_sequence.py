#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Bound native API waits and judge completed stages only after worker cleanup.

The producer never writes gates. A separate, isolated Linux subreaper owns its
workers; only the parent evaluator writes gates after a quiescence receipt.
Completed stages pin their request receipts, native events and image bytes.
An interrupted stage cannot be promoted to a complete stage.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
REQUEST_TIMEOUT_S = 120.0
SHUTDOWN_TIMEOUT_S = 30.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    os.replace(tmp, path)


def save_run(root, run):
    atomic(Path(root)/'run.json', {k:v for k,v in run.items() if k not in ('images','events')})


def expire(root, run, operation, t0, seconds):
    run['timeout'] = dict(operation=operation, t0=t0, t1=time.time(), limit_s=seconds)
    save_run(root, run)


    os._exit(124)


def bounded(root, run, operation, fn, seconds=REQUEST_TIMEOUT_S):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('finite positive native wait required')
    t0 = time.time()
    answer = []
    def invoke():
        try:
            answer.append((True, fn()))
        except BaseException as exc:
            answer.append((False, exc))
    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        expire(root, run, operation, t0, seconds)
    ok, value = answer[0]
    if not ok:
        raise value
    return value


def events(path):
    path = Path(path)
    if path.stat().st_size > 64*1024*1024:
        raise ValueError('native evidence exceeds bounded reader')
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('incomplete native event')
    out = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not all(isinstance(e, dict) for e in out):
        raise ValueError('native event is not an object')
    return out


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('evidence path escapes producer root')
    return path


def checkpoint(root, run, name, required):
    root = Path(root).resolve()
    dest = inside(root/'completed'/name, root)
    if not run.get('verifier_run_id') or dest.exists():
        raise ValueError('stage must have an invocation identity and may be sealed only once')
    required = tuple(required)
    for key in required:
        rd = run.get('readings', {}).get(key) or {}
        t0, t1 = rd.get('t0'), rd.get('t1')
        if not all(type(t) in (int,float) and math.isfinite(t) for t in (t0,t1)) or t1 < t0:
            raise ValueError('cannot seal an unfinished request: '+key)
    record = inside(run['record'], root)
    events(record)
    dest.mkdir(parents=True)
    shutil.copyfile(record, dest/'record.jsonl')
    snapshot = {k:v for k,v in run.items() if k not in ('images','events','inflight')}
    snapshot['record'] = str(dest/'record.jsonl')
    snapshot['stage_complete'] = name
    snapshot['required_readings'] = list(required)
    files = {str((dest/'record.jsonl').relative_to(root)): sha(dest/'record.jsonl')}
    for rd in snapshot.get('readings', {}).values():
        for value in ([rd['image']] if rd.get('image') else rd.get('image_files', [])):
            if value is not None:
                path = inside(value, root)
                files[str(path.relative_to(root))] = sha(path)
    if run.get('cap_dir'):
        cap = inside(run['cap_dir'], root)
        (dest/'clip').mkdir()
        for path in cap.glob('*.pt'):
            target = dest/'clip'/path.name
            shutil.copyfile(inside(path, root), target)
            files[str(target.relative_to(root))] = sha(target)
        snapshot['cap_dir'] = str(dest/'clip')
    atomic(dest/'run.json', snapshot)
    files[str((dest/'run.json').relative_to(root))] = sha(dest/'run.json')
    atomic(dest/'manifest.json', dict(schema='sd3-completed-native-stage-v1',
        run_id=run['verifier_run_id'], stage=name, required_readings=list(required), files=files))


def load_checkpoint(root, name, run_id):
    root = Path(root).resolve()
    dest = inside(root/'completed'/name, root)
    meta = json.loads((dest/'manifest.json').read_text())
    if (not run_id or meta.get('run_id') != run_id or meta.get('stage') != name
            or meta.get('schema') != 'sd3-completed-native-stage-v1'):
        raise ValueError('completed stage identity mismatch')
    files = meta.get('files') or {}
    for path in (dest/'run.json', dest/'record.jsonl'):
        if str(path.relative_to(root)) not in files:
            raise ValueError('completed stage manifest missing required file')
    for relative, digest in files.items():
        if sha(inside(root/relative, root)) != digest:
            raise ValueError('completed stage bytes changed: '+relative)
    run = json.loads((dest/'run.json').read_text())
    if run.get('verifier_run_id') != run_id or run.get('stage_complete') != name:
        raise ValueError('completed stage receipt mismatch')
    if run.get('required_readings') != meta.get('required_readings'):
        raise ValueError('completed stage request identity mismatch')
    for rd in run.get('readings',{}).values():
        for value in ([rd['image']] if rd.get('image') else rd.get('image_files',[])):
            if value is not None and str(inside(value,root).relative_to(root)) not in files:
                raise ValueError('completed stage omitted an image digest')
    if run.get('cap_dir'):
        for path in inside(run['cap_dir'],root).glob('*.pt'):
            if str(path.relative_to(root)) not in files:
                raise ValueError('completed stage omitted a CLIP capture digest')
    run['events'] = events(inside(run['record'], root))
    run['stage_manifest'] = str(dest/'manifest.json')
    run['stage_manifest_sha256'] = sha(dest/'manifest.json')
    return run


def supervise(script, root, argv, *, reserve_s=45., term_s=5., kill_s=5.):
    """Use only the enclosing segment's remaining budget, with evaluator reserve."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ['SD3_COMMAND_DEADLINE']) - reserve_s
    if not math.isfinite(deadline):
        raise ValueError('native producer requires the enclosing clock deadline')
    clock_python = os.environ.get('SD3_CLOCK_PYTHON')
    if not clock_python or not Path(clock_python).is_absolute():
        raise ValueError('fixed isolated clock Python missing')
    command = [clock_python, '-I', '-S', str(HERE/'native_sequence.py'), '--supervise',
               str(root), str(deadline), str(term_s), str(kill_s),
               sys.executable, str(script), *map(str, argv)]
    proc = subprocess.run(command, check=False)
    path = root/'supervisor/result.json'
    if not path.is_file():
        raise SystemExit(75)
    result = json.loads(path.read_text())
    if (result.get('quiescent') is not True or result.get('error') or result.get('journal_error')
            or result.get('status') in ('instrument_error','cancelled','launch_error','cleanup_unproven')
            or proc.returncode == 75):
        raise SystemExit(75)
    result['receipt_path'] = str(path)
    result['receipt_sha256'] = sha(path)
    return result


def read_supervised(root, execution):
    """Only a stopped producer is readable, including an interrupted receipt."""
    if execution.get('quiescent') is not True:
        raise ValueError('native evidence read before quiescence')
    root = Path(root)
    run = json.loads((root/'run.json').read_text()) if (root/'run.json').is_file() else {}
    if run.get('verifier_run_id') != os.environ.get('SD3_VERIFIER_RUN_ID'):
        raise ValueError('producer identity mismatch')
    run['execution'] = execution
    return run


def timeout_verdict(run, *, world):
    """Attribute only a bounded API wait witnessed by healthy native workers."""
    timeout = run.get('timeout') or {}
    out = dict(verdict='not_evaluated', reasons=['native sequence did not complete; attribution unavailable'],
               details={'timeout':timeout, 'execution':run.get('execution')})
    try:
        ev = events(run['record'])
        t0, t1 = timeout['t0'], timeout['t1']
        if not all(type(t) in (int,float) and math.isfinite(t) for t in (t0,t1)) or t1 < t0:
            return out
        ranks = set(range(world))
        installed, completed = set(), set()
        for e in ev:
            if e.get('event') in ('batch_observer_error','record_error','truncated'):
                return out
            r = e.get('rank')
            t = e.get('t')
            if not isinstance(r, dict) or type(t) not in (int,float) or not math.isfinite(t) or t > t1:
                continue
            rank = r.get('dist_rank')
            if type(rank) is not int or rank not in ranks or r.get('dist_world') != world:
                continue
            if e.get('event') == 'batch_observer_installed' and e.get('pre_hook') is True:
                installed.add(rank)
            if e.get('event') == 'batch_dit' and e.get('completed') is True:
                completed.add(rank)
        if installed != ranks or completed != ranks or run.get('built') is not True:
            return out
        op = timeout.get('operation','')
        if op in ('fault','failure'):
            fired = [e for e in ev if e.get('event') == 'fault_fired' and e.get('phase') == 'after_dit'
                     and t0 <= e.get('t', -1) <= t1]
            if len(fired) != 1 or (op == 'failure' and fired[0].get('rows',0) < 2):
                return out
        if op == 'after_fault' or op.startswith('failure_after/'):
            prior = run.get('readings',{}).get('fault') or {}
            lo = prior.get('t0') if op == 'after_fault' else run.get('t_failure')
            hi = prior.get('t1') if op == 'after_fault' else run.get('t_failure_end')
            if (not all(type(t) in (int,float) and math.isfinite(t) for t in (lo,hi))
                    or hi > t0 or lo > hi):
                return out
            fired = [e for e in ev if e.get('event') == 'fault_fired' and e.get('phase') == 'after_dit'
                     and lo <= e.get('t', -1) <= hi]
            if len(fired) != 1:
                return out
        if run.get('execution',{}).get('quiescent') is not True:
            return out
        out.update(verdict='fail', reasons=[f'native {op} exceeded its fixed {timeout["limit_s"]} s limit'])
        out['details']['recorder_healthy_ranks'] = sorted(installed)
        return out
    except (OSError, ValueError, KeyError, TypeError):
        return out


if __name__ == '__main__':
    if len(sys.argv) < 8 or sys.argv[1] != '--supervise':
        raise SystemExit('native sequence supervisor requires an explicit command')

    sys.path.insert(0, str(HERE))
    import verifier_clock as C
    root = Path(sys.argv[2])/'supervisor'
    root.mkdir(parents=True, exist_ok=True)
    deadline, term_s, kill_s = map(float, sys.argv[3:6])
    result = C.execute(root, sys.argv[6:], label='native API producer', phase='work',
                       cap=max(.001, deadline-time.monotonic()), deadline=deadline,
                       term_s=term_s, kill_s=kill_s)
    C.atomic_json(root/'result.json', result)
    raise SystemExit(result['returncode'])
