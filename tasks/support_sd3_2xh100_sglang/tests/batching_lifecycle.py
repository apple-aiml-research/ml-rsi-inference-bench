#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Persist batching wait attribution without changing native or trial budgets."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time
import uuid

import native_sequence as N


@contextmanager
def operation(root, run, name, *, kind='api', seconds=None):
    """Written by the producer coordinator, separately from request threads."""
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('finite positive operation limit required')
    if not run.get('verifier_run_id'):
        raise ValueError('batching operation requires an invocation identity')
    receipt = dict(schema='sd3-batching-operation-v1', run_id=run['verifier_run_id'],
                   id=uuid.uuid4().hex, operation=name, kind=kind, status='active',
                   t0=time.time(), monotonic_start=time.monotonic(), limit_s=seconds)
    deadline = os.environ.get('SD3_COMMAND_DEADLINE')
    receipt['producer_deadline'] = float(deadline) if deadline else None
    if deadline and not math.isfinite(receipt['producer_deadline']):
        raise ValueError('finite producer deadline required')
    receipt['operation_deadline'] = receipt['monotonic_start'] + seconds if seconds else None
    path = Path(root)/'operation.json'
    N.atomic(path, receipt)
    try:
        yield receipt
    except BaseException as exc:
        receipt.update(status='error', error=f'{type(exc).__name__}: {exc}'[:400])
        raise
    else:
        receipt['status'] = 'completed'
    finally:
        receipt.update(t1=time.time(), monotonic_end=time.monotonic())
        N.atomic(path, receipt)
        N.atomic(Path(root)/'operations'/(receipt['id']+'.json'), receipt)


def bounded(root, run, name, fn, *, seconds):
    with operation(root, run, name, seconds=seconds):
        return N.bounded(root, run, name, fn, seconds=seconds)


def wait_rows(root, run, name, observed, threads, *, seconds):
    """Once all requests returned, waiting cannot produce an in-flight group.

    Absence remains false, never a synthetic success. Read once more after
    observing completion so a final worker event is not missed in the race.
    """
    with operation(root, run, name, kind='observation', seconds=seconds) as receipt:
        while True:
            if observed():
                receipt['outcome'] = 'rows_observed'
                return True
            if not any(th.is_alive() for th in threads):
                found = bool(observed())
                receipt['outcome'] = 'rows_observed' if found else 'requests_finished_without_rows'
                return found
            left = receipt['operation_deadline'] - time.monotonic()
            if left <= 0:
                receipt['outcome'] = 'observation_limit_without_rows'
                return False
            time.sleep(min(.05, left))


def interruption_verdict(root, run, *, world):
    """Separate a proven API timeout from exhaustion of the producer budget.

    An outer deadline is not proof that the active API used its full limit.
    Its stage can nevertheless be identified. Candidate budget attribution
    requires quiescence and healthy native observations from every rank.
    """
    out = N.timeout_verdict(run, world=world)
    try:
        path = Path(root)/'operation.json'
        op = json.loads(path.read_text())
        if (op.get('schema') != 'sd3-batching-operation-v1'
                or op.get('run_id') != run.get('verifier_run_id')
                or not op.get('run_id') or not isinstance(op.get('operation'), str)
                or op.get('kind') not in ('api', 'observation', 'setup', 'processing')
                or op.get('status') not in ('active', 'completed', 'error')):
            raise ValueError('operation identity or schema mismatch')
        for key in ('t0', 'monotonic_start'):
            if type(op.get(key)) not in (int, float) or not math.isfinite(op[key]):
                raise ValueError('invalid operation clock')
        for key in ('limit_s', 'operation_deadline', 'producer_deadline'):
            if op.get(key) is not None and (type(op[key]) not in (int, float) or not math.isfinite(op[key])):
                raise ValueError('invalid operation deadline')
        out['details']['operation_receipt'] = op
        out['details']['operation_sha256'] = N.sha(path)
        execution = run.get('execution') or {}
        if run.get('timeout'):
            out['details']['interruption_kind'] = 'operation_limit'
            return out
        if execution.get('status') != 'timed_out' or execution.get('quiescent') is not True:
            return out
        out['details']['interruption_kind'] = 'producer_budget'
        active = op['status'] == 'active'
        where = f"during {op['operation']} ({op['kind']})" if active else f"after {op['operation']} ({op['status']})"
        out['reasons'] = [f'native sequence exhausted its producer budget {where}; '
                          'this does not establish a per-operation timeout']
        installed, completed = set(), set()
        ranks = set(range(world))
        for event in N.events(run['record']):
            if event.get('event') in ('batch_observer_error', 'record_error', 'truncated'):
                out['details']['attribution'] = 'observer_error'
                return out
            rank = event.get('rank') or {}
            r = rank.get('dist_rank') if isinstance(rank, dict) else None
            if type(r) is not int or r not in ranks or rank.get('dist_world') != world:
                continue
            if type(event.get('t')) not in (int, float) or not math.isfinite(event['t']):
                continue
            if event.get('event') == 'batch_observer_installed' and event.get('pre_hook') is True:
                installed.add(r)
            if event.get('event') == 'batch_dit' and event.get('completed') is True:
                completed.add(r)
        if active and run.get('built') is True and installed == completed == ranks and op['kind'] in ('api', 'observation'):
            out.update(verdict='fail')
            out['details'].update(attribution='candidate_budget', recorder_healthy_ranks=sorted(ranks))
        else:
            out['details']['attribution'] = 'unproven_native_health_or_non_candidate_stage'
        return out
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out['details']['operation_receipt_error'] = f'{type(exc).__name__}: {exc}'
        return out
