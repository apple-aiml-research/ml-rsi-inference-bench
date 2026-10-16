#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Protocol-neutral queued-abort coverage from the verifier's wire recorder."""
import math


def finite(t):
    return type(t) in (int, float) and math.isfinite(t)


def queued_public_outcome(long, queued, abort, *, control_ok):
    """Causal public-API control, independent of the engine's abort transport.

    This is only one clause: the caller also checks healthy native row work,
    zero queued compute, a surviving peer and post-abort recovery.
    """
    times = (long.get('t0'), long.get('t1'), queued.get('t0'), queued.get('t1'), abort.get('t'))
    def result(verdict, reason):
        return {'verdict': verdict, 'reason': reason}
    if not all(finite(t) for t in times):
        return result('not_evaluated', 'queued cancellation time windows incomplete')
    l0, l1, q0, q1, at = times
    if q1 < at:
        return result('fail', 'queued request returned before abort was invoked')
    if not l0 <= q0 <= at < l1 or not q0 <= q1:
        return result('not_evaluated', 'abort did not overlap an active peer and pending request')
    if control_ok is not True:
        return result('not_evaluated', 'matching un-aborted queued control did not return its reference image')
    text = (str(queued.get('type', '')) + ' ' + str(queued.get('error', ''))).lower()
    if queued.get('raised') is True:
        if not any(s in text for s in ('abort', 'cancel', '499')):
            return result('fail', 'queued abort returned an unrelated error')
    elif queued.get('none') is not True:
        return result('fail', 'queued abort did not stop without an image')
    return result('pass', None)


def observed_peer_work(events, t0, t1):
    """Include all native workers in the request window, not just AbortReq receivers."""
    relevant = ('batch_dit_start', 'batch_dit', 'batch_observer_error', 'truncated', 'record_error')
    selected = [e for e in events if e.get('event') in relevant and finite(e.get('t'))
                and finite(t0) and finite(t1) and t0 <= e['t'] <= t1]
    if any(type(e.get('pid')) is not int for e in selected):
        return {'measured': False, 'rows': None, 'reason': 'native worker identity missing'}
    pids = {e['pid'] for e in selected}
    return native_row_work(events, t0, t1, pids)


def queued_abort_receipt(events, request_id, queued_t0, abort_t, running_t1, queued_t1):
    if (not all(type(t) in (int, float) and math.isfinite(t) for t in (queued_t0, abort_t, running_t1, queued_t1))
            or not queued_t0 <= abort_t <= min(running_t1, queued_t1)):
        return {'ok': False, 'reason': 'request/abort time windows incomplete'}
    installed = {e.get('pid') for e in events if e.get('event') == 'request_wire_installed'
                 and type(e.get('pid')) is int and finite(e.get('t')) and e['t'] <= queued_t0}
    generation = [e for e in events if e.get('event') == 'request_wire_received'
                  and e.get('role') == 'generation' and e.get('request_id') == request_id
                  and finite(e.get('t')) and queued_t0 <= e['t'] <= abort_t and e.get('pid') in installed]
    aborted = [e for e in events if e.get('event') == 'request_wire_received'
               and e.get('role') == 'abort' and e.get('request_id') == request_id
               and finite(e.get('t')) and abort_t <= e['t'] <= min(running_t1, queued_t1) and e.get('pid') in installed]
    pids = {e['pid'] for e in generation} & {e['pid'] for e in aborted}
    errors = [e for e in events if e.get('event') == 'request_wire_error'
              and e.get('pid') in pids and finite(e.get('t')) and queued_t0 <= e['t'] <= queued_t1]
    ok = bool(pids) and not errors
    return {'ok': ok, 'reason': None if ok else 'no healthy same-worker generation + abort receipt before the peer finished',
            'worker_pids': sorted(pids), 'generation': generation, 'abort': aborted, 'errors': errors}


def native_row_work(events, t0, t1, worker_pids):
    """Pair starts/finishes owned by this peer's window, on observed workers.

    A preceding request can return a cancellation before its last forward
    finishes. Its completion timestamp does not assign that work to the peer.
    Conversely an unmatched peer start must not vanish from the denominator.
    """
    if not all(type(t) in (int, float) and math.isfinite(t) for t in (t0, t1)) or t1 < t0:
        return {'measured': False, 'reason': 'peer window missing', 'rows': None}
    pids = set(worker_pids)
    installed = {e.get('pid') for e in events if e.get('event') == 'batch_observer_installed'
                 and e.get('pre_hook') is True and finite(e.get('t')) and e['t'] <= t0}
    completed = [e for e in events if e.get('event') == 'batch_dit' and e.get('pid') in pids
               and finite(e.get('t')) and t0 <= e['t'] <= t1]
    def key(e):
        return (e.get('pid'), e.get('n'), e.get('t_start'))
    all_starts = [e for e in events if e.get('event') == 'batch_dit_start' and e.get('pid') in pids
                  and finite(e.get('t_start')) and finite(e.get('t')) and e['t_start'] <= e['t']]
    starts = [e for e in all_starts if t0 <= e['t_start'] <= t1]

    preexisting = {key(e) for e in all_starts if e['t_start'] < t0 and type(e.get('n')) is int}
    forward = [e for e in completed if not finite(e.get('t_start')) or not
               (e['t_start'] < t0 and type(e.get('n')) is int and key(e) in preexisting)]
    valid_keys = all(type(e.get('n')) is int and e['n'] > 0 and finite(e.get('t_start'))
                     for e in starts + forward)
    start_keys = [key(e) for e in starts] if valid_keys else []
    finish_keys = [key(e) for e in forward] if valid_keys else []
    paired = (valid_keys and len(set(start_keys)) == len(start_keys)
              and len(set(finish_keys)) == len(finish_keys) and set(start_keys) == set(finish_keys))
    start_rows = {key(e): e.get('rows') for e in starts} if valid_keys else {}
    errors = [e for e in events if e.get('event') in ('batch_observer_error', 'truncated', 'record_error')
              and e.get('pid') in pids and finite(e.get('t')) and t0 <= e['t'] <= t1]
    healthy = bool(forward) and paired and not errors and pids <= installed and all(e.get('completed') is True
        and e['t_start'] <= e['t'] and type(e.get('rows')) is int and e['rows'] > 0
        and type(start_rows.get(key(e))) is int and start_rows[key(e)] == e['rows'] for e in forward)
    return {'measured': healthy, 'rows': sum(e['rows'] for e in forward) if healthy else None,
            'reason': None if healthy else 'peer row recorder absent/incomplete or failed', 'forwards': len(forward),
            'starts': len(starts), 'paired': paired,
            'excluded_preexisting_forwards': len(completed) - len(forward)}
