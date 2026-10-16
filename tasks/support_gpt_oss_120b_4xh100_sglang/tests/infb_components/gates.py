#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private verdicts over complete public protocol records.

Each family and workload cell stands on its own. Missing/nonfinite data cannot
be converted to a pass or hidden by a median over unrelated cases. Numerical
limits come from the trusted public `numerical-policy.json`; resource budgets and
coverage parameters from the private configuration. Facts are kept separate:
a failed speed verdict does not erase a valid identity, and a valid identity
does not waive failed speed, correctness or resources.
"""
import json
import math
from pathlib import Path

from .paired import CELLS, compare


class ReferenceFault(RuntimeError):
    """The reference, the held-out author data or the host failed; not a candidate result."""


class InvalidAccounting(AssertionError):
    """An allocation record that cannot be interpreted: never zero usage or a clipped pass."""


def finite(value):
    assert type(value) in (int, float) and math.isfinite(value), f'nonfinite measurement: {value!r}'
    return value


def bounded(value, ceiling, label):
    assert 0 <= finite(value) <= ceiling, f'{label}: {value} exceeds {ceiling}'


def policy(protocols):
    """The published operation limits, read from the trusted restored public copy."""
    from gpt_oss.numerics import policy as read
    data = read(Path(protocols) / 'gpt_oss/numerical-policy.json')
    return {name: {'max': float(v['max']), 'rms': float(v['rms'])} for name, v in data['operations'].items()}


def numerical(row, limits, label):
    if row.get('structurally_valid') is False:
        raise AssertionError(f"{label}: structurally invalid result: {row.get('reason')}")
    assert type(row['count']) is int and row['count'] > 0, f'{label}: missing elements'
    for key in ('max', 'rms'):
        bounded(row[key], limits[key], f'{label}/{key}')


def workspace_budget(count, ranges):
    """The workspace ceiling for a complete invocation of `count` tokens: a total function on
    1-4,096 from contiguous [first, last, bytes] ranges."""
    assert type(count) is int, f'token count must be an integer, not {count!r}'
    expected = 1
    for first, last, budget in ranges:
        assert first == expected and last >= first and type(budget) is int and budget > 0, \
            f'workspace ranges are not contiguous from 1: {ranges}'
        if first <= count <= last:
            return budget
        expected = last + 1
    raise AssertionError(f'{count} tokens is outside the workspace policy 1-{expected - 1}')


def accounting(row):
    """setup peak, retained and steady scratch, and external device allocations of one cell.
    The accepted peak is max(setup, retained + steady) plus live allocations outside the
    allocator. A negative delta or an unexplained device discrepancy is invalid accounting."""
    values = {key: row.get(name) for key, name in (
        ('setup', 'setup_peak_workspace_bytes'), ('retained', 'retained_workspace_bytes'),
        ('steady', 'steady_peak_workspace_bytes'), ('external', 'external_retained_bytes'))}
    for key, value in values.items():
        if type(value) is not int:
            raise InvalidAccounting(f'workspace {key} is not an integer byte count: {value!r}')
        if value < 0:
            raise InvalidAccounting(f'workspace {key} is negative ({value}): the accounting is invalid')
    values['accepted'] = max(values['setup'], values['retained'] + values['steady']) + values['external']
    return values


def resident(record):
    """Unique tensor storage ranges plus persistent allocations outside their pool.

    Allocator reservation is not live weight storage. The device delta outside the
    allocator reservation is counted; an opaque binding is not zero bytes, and a
    device delta below the reservation is an unexplained discrepancy.
    """
    values = record['persistent']
    allocated = record.get('allocator_live_delta', values.get('allocator_live_delta'))
    reserved = record.get('allocator_reserved_delta', values.get('allocator_reserved_delta'))
    device = record.get('device_used_delta', values.get('device_used_delta'))
    assert all(v is not None for v in (allocated, reserved, device)), 'missing allocator observations'
    for value in (values['bytes'], allocated, reserved, device):
        finite(value)
    if values['bytes'] < 0 or allocated < 0 or reserved < 0:
        raise InvalidAccounting(f'negative resident delta: {values["bytes"]}, {allocated}, {reserved}')
    if device < reserved:
        raise InvalidAccounting(f'device use grew {device} B, less than the allocator reservation {reserved} B')
    assert max(allocated, device) > 0, 'no observed native weight allocation'
    return max(values['bytes'], allocated) + (device - reserved)


def identity(value, root):
    """Recompute source hashes instead of trusting a binding's identity claim."""
    import hashlib
    assert isinstance(value, dict) and isinstance(value.get('path'), str) and \
        isinstance(value.get('sha256'), str), f'malformed identity record: {value!r}'
    root = Path(root).resolve()
    path = Path(value['path']).resolve()
    assert path.is_relative_to(root), f'bound source outside reconstructed submission: {path}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256'], f'source hash differs: {path}'
    result = {'path': str(path.relative_to(root)), 'sha256': value['sha256']}
    if 'quant_method' in value:
        result['quant_method'] = identity(value['quant_method'], root)
    return result


def expert(records, *, expected_cells, limits, storage_limit, root):
    assert len(records) == 4, f'{len(records)} rank records, not 4'
    faults = [r.get('author_data_fault') for r in records if r.get('author_data_fault')]
    if faults:
        raise ReferenceFault(f'held-out expert case outside the admitted domain: {faults[0]}')
    identities = []
    for rank, record in enumerate(records):
        assert record['schema'] == 'gpt-oss-expert-measurement/2' and record['rank'] == rank
        identities.append(identity(record['identity'], root))
        bounded(resident(record), storage_limit, f'rank {rank} expert residency')
        observed = {(r['count'], r['family'], r['graph']) for r in record['rows']}
        assert observed == set(expected_cells) and len(record['rows']) == len(expected_cells)
        if rank == 0:
            for row in record['rows']:
                label = f"{row['family']}/{row['count']}/graph={row['graph']}"
                for key in ('error', 'changed_error'):
                    numerical(row[key], limits, label+'/'+key)
                    assert row[key]['count'] == row['count'] * 2880
    assert all(i == identities[0] for i in identities), 'TP ranks bind different native implementations'
    return identities[0]


def binding(record):
    assert record['ordinary_native_calls'] > 0, 'adapter does not invoke its named native module'
    assert record['suppressed_native_calls'] > 0, 'perturbed native module was not invoked'
    assert record['finite'] is True
    assert finite(record['max_output_change']) > 0 and record['changed_elements'] > 0, \
        'adapter result does not depend on its named native module'


def _fact(check):
    """Run one predicate; a fact is ok, failed (with its reason) or invalid accounting."""
    try:
        return {'ok': True, 'value': check()}
    except InvalidAccounting as exc:
        return {'ok': False, 'invalid_accounting': True, 'error': str(exc)}
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}


def complete_draws(record, side='candidate'):
    """Every draw of one side whose four rank records were collected, in draw order."""
    return [(pair['block'], pair['attempt'], pair['sides'][side]['ranks'])
            for pair in record.get('pairs', [])
            if len(pair.get('sides', {}).get(side, {}).get('ranks') or []) == 4]


def component_facts(record, *, rho, limits, storage_limit, ranges, root, side='candidate'):
    """Separate facts over the complete raw paired record: measurement completeness, the
    measured source identity, numerical validity, resources, execution and the speed verdict.
    Every ratio is reported. Identity comes from complete raw rank records, whatever the speed."""
    draws = complete_draws(record, side)
    facts = {'draws_collected': len(draws)}

    def measured():
        assert record.get('schema') == 'gpt-oss-component-pairs/1', 'unexpected paired schema'
        rows = compare(record['pairs'])
        assert record.get('status') == 'complete', f"paired status {record.get('status')!r}"
        assert len(rows) == len(CELLS)
        return rows
    facts['measurement_complete'] = _fact(measured)

    def source():
        assert draws, 'no complete draw: identity unavailable'
        found = None
        for block, attempt, ranks in draws:
            for rank, data in enumerate(ranks):
                assert data['rank'] == rank, f'rank record {rank} names rank {data.get("rank")}'
                current = identity(data['identity'], root)
                found = found or current
                assert current == found, 'measured expert implementation changed between draws/ranks'
        return found
    facts['identity'] = _fact(source)

    def numbers():
        first = [ranks for block, attempt, ranks in draws if block == 0]
        assert first, 'the first block, which carries the timed-cell checks, was not collected'
        rows = first[0][0]['rows']
        assert {(r['count'], r['family'], r['graph']) for r in rows} == CELLS
        checked = []
        for row in rows:
            slots = {c['slot']: c for c in row.get('checked_slots', [])}
            assert set(slots) == {0, 7}, f"timed cell {row['count']}/{row['family']}: checked slots {sorted(slots)}"
            for slot, check in sorted(slots.items()):
                error = check['error']
                numerical(error, limits, f"timed cell {row['count']}/{row['family']}/slot {slot}")
                assert error['count'] == row['count'] * 2880
                checked.append({'count': row['count'], 'family': row['family'], 'slot': slot,
                                'max': error['max'], 'rms': error['rms']})
        return checked
    facts['numerical'] = _fact(numbers)

    def resources():
        peaks = []
        for block, attempt, ranks in draws:
            for rank, data in enumerate(ranks):
                bounded(resident(data), storage_limit, f'rank {rank} persistent expert')
                for row in data['rows']:
                    account = accounting(row)
                    budget = workspace_budget(row['count'], ranges)
                    bounded(account['accepted'], budget, f"rank {rank} workspace/{row['count']}/graph={row['graph']}")
                    peaks.append({'block': block, 'rank': rank, 'count': row['count'], 'family': row['family'],
                                  'graph': row['graph'], 'budget': budget, **account})
        assert peaks, 'no resource observation'
        return {'cells': len(peaks), 'largest_fraction': max(p['accepted'] / p['budget'] for p in peaks)}
    facts['resources'] = _fact(resources)

    def execution():
        first = [ranks for block, attempt, ranks in draws if block == 0]
        assert first, 'the first block was not collected'
        for rank, data in enumerate(first[0]):
            binding(data['binding_witness'])
            for row in data['rows']:
                expected_mode = 'DECODE' if row['graph'] else 'EXTEND'
                assert row['serving_mode'] == expected_mode
                assert row['custom_all_reduce'] is True
                assert row['kernels'] and sum(row['kernels'].values()) > 0, \
                    f"rank {rank} cell {row['count']}/{row['family']}: no executed kernels"
        return {'witness_ranks': 4}
    facts['execution'] = _fact(execution)

    def speed():
        rows = facts['measurement_complete']
        assert rows['ok'], 'speed not evaluated: the measurement is incomplete'
        ratios = [{'count': r['count'], 'family': r['family'], 'graph': r['graph'], 'ratio': finite(r['ratio'])}
                  for r in rows['value']]
        slow = [r for r in ratios if not r['ratio'] >= rho]
        return {'ratios': ratios, 'slow': slow, 'rho': rho}
    facts['speed'] = _fact(speed)
    if facts['speed']['ok'] and facts['speed']['value']['slow']:
        facts['speed']['ok'] = False
        facts['speed']['error'] = 'slow complete expert cells: ' + ', '.join(
            f"{r['count']}/{r['family']}={r['ratio']:.4f}" for r in facts['speed']['value']['slow'])
    facts['ok'] = all(facts[k]['ok'] for k in ('measurement_complete', 'identity', 'numerical',
                                                'resources', 'execution', 'speed'))
    return facts


def identity_only(records, root):
    """The one bounded identity-only invocation: the same prepared native operation on four ranks."""
    assert len(records) == 4, f'{len(records)} identity records, not 4'
    found = None
    for rank, record in enumerate(records):
        assert record['schema'] == 'gpt-oss-expert-identity/1' and record['rank'] == rank
        current = identity(record['identity'], root)
        found = found or current
        assert current == found, 'TP ranks bind different native implementations'
    return found


def attention(record, limits):
    assert record['schema'] == 'gpt-oss-attention-measurement/3'
    if record.get('author_data_fault'):
        raise ReferenceFault(f"held-out attention case outside the admitted domain: {record['author_data_fault']}")
    expected = {(w, tuple(p), mode) for w in (-1, 127)
                for p in ([0], [127], [128,129,511], [4096,127])
                for mode in ('eager', 'graph', 'changed-replay')}
    observed = {(r['window'], tuple(r['prefixes']), r['mode']) for r in record['rows']}
    assert observed == expected and len(record['rows']) == len(expected)
    for row in record['rows']:
        numerical(row, limits, f"attention {row['window']}/{row['prefixes']}/{row['mode']}")
        assert row['count'] == sum(row['counts']) * 16 * 64


def production_storage(record, *, expert_limit, ordinary_allowance):
    """Check expert tensors and the whole persistent load before KV allocation.

    The second bound catches retained original/expanded expert copies reachable
    from other objects or held outside the model. Its separately published
    allowance covers ordinary embeddings, attention/router weights and loader
    support allocations. Free caching-allocator reservation is not residency.
    """
    device, reserved = finite(record['device_used_delta']), finite(record['allocator_reserved_delta'])
    if device < reserved:
        raise InvalidAccounting(f'device use grew {device} B, less than the allocator reservation {reserved} B')
    external = device - reserved
    expert_bytes = finite(record['expert_storage_bytes']) + external
    total_bytes = max(finite(record['all_model_storage_bytes']),
                      finite(record['allocator_live_delta'])) + external
    assert expert_bytes > 0 and total_bytes >= expert_bytes
    bounded(expert_bytes, expert_limit, 'production expert tensors')
    bounded(total_bytes, expert_limit+ordinary_allowance,
            'whole model residency, including retained expert copies')
    return {'expert_bytes': expert_bytes, 'whole_model_bytes': total_bytes}


def coverage_runtime(record, *, concurrency, layers, source=None):
    """Graph execution evidence from the structured coverage record: every rank
    observed, the same captures on every rank bounded by the recipe, a capture able to hold the concurrent
    serving load, useful model graph replay on every rank in each workload that requires it (C1, C64 and
    the largest captured batch) -- replays during graph decode, no eager native-expert calls inside it,
    padded to a captured shape, a profiled warm graph decode whose kernels ran inside graph launches --
    and complete rank coverage in the others (C200 serves correctness and capacity, not replay)."""
    assert record.get('schema') == 'gpt-oss-graph-coverage/1', 'no structured coverage record'
    ranks = record['ranks']
    assert [r['rank'] for r in ranks] == [0, 1, 2, 3], 'incomplete rank evidence'
    captured = [tuple(r['captured'] or ()) for r in ranks]
    assert all(captured) and all(c == captured[0] for c in captured), f'captures differ or are absent: {captured}'
    assert max(captured[0]) <= record['recipe_max'], 'captured beyond the recipe maximum'
    assert max(captured[0]) >= concurrency, \
        f'the largest captured graph {max(captured[0])} cannot hold the concurrent load {concurrency}'
    if source is not None:
        for r in ranks:
            assert r['binding_identities'], f"rank {r['rank']}: no bound expert identity"
    largest = record.get('largest_graph') or {}
    assert largest.get('n') and largest['n'] >= concurrency, \
        f"the largest captured batch {largest.get('n')} is below the concurrent load {concurrency}"
    checked = {}
    for name, entry in [*record['workloads'].items(), ('largest_graph', largest)]:
        assert entry.get('draws'), f'workload {name} did not run'
        check = graph_replay if entry.get('graph_replay') else rank_coverage
        checked[name] = [check(draw, captured[0], layers, draw['name']) for draw in entry['draws']]
    return {'captured': list(captured[0]), 'workloads': checked}


def rank_coverage(entry, captured, layers, name):
    """A workload served on all four ranks: every rank recorded the window's decode forwards."""
    observations = entry.get('observations') or []
    assert [o['rank'] for o in observations] == [0, 1, 2, 3], f'{name}: incomplete rank evidence'
    for o in observations:
        assert o['decode_forwards'] > 0, f"{name}: rank {o['rank']} recorded no decode step"
    return {'ranks': 4, 'decode_forwards': [o['decode_forwards'] for o in observations],
            'graph_decode_forwards': [o['graph_decode_forwards'] for o in observations]}


def graph_replay(entry, captured, layers, name):
    observations = entry.get('observations') or []
    assert [o['rank'] for o in observations] == [0, 1, 2, 3], f'{name}: incomplete rank evidence'
    for o in observations:
        rank = o['rank']
        assert o['graph_decode_forwards'] > 0, f'{name}: rank {rank} served no decode step from a CUDA graph'
        assert o['graph_decode_without_replay'] == 0, f'{name}: rank {rank} reported graph decode without a replay'
        assert o['graph_decode_native_calls'] == 0, \
            f'{name}: rank {rank} ran the native experts eagerly during graph decode (detached graph)'
        for live, padded in o['live_to_padded']:
            assert padded in captured and padded >= live, f'{name}: rank {rank} replayed {padded} for {live}'
        profiled = [p for p in o['profiled_decode'] if p['graph']]
        assert profiled, f'{name}: rank {rank} has no profiled warm graph decode'
        a = profiled[0]['attribution'] or {}
        assert a.get('graph_launches', 0) >= 1, f'{name}: rank {rank}: no cudaGraphLaunch in a graph decode'
        assert a.get('graph_kernels', 0) >= layers and a['graph_kernels'] > a.get('direct_kernels', 0), \
            f"{name}: rank {rank}: model kernels ran outside the graph launch ({a})"
    return {'ranks': 4, 'graph_decode_forwards': [o['graph_decode_forwards'] for o in observations]}


def coverage_numerical(record, *, bar):
    """Numerical correctness of every admitted request, the last included, in every fixed
    workload and the largest-graph diagnostic. A malformed reading is terminal; a structurally
    valid reading at or over the bar has the protocol's one redraw and passes only if it is below."""
    results = {}
    largest = record.get('largest_graph') or {}
    assert largest.get('n'), 'no captured graph to derive the largest-graph workload from'
    for name, entry in [*record['workloads'].items(), ('largest_graph', largest)]:
        readings = entry.get('draws') or []
        assert readings, f'{name}: the workload did not run'
        worst = [requests_below(reading, bar) for reading in readings]
        assert any(w < bar for w in worst), (
            f"{name}: at {entry['n']} concurrent requests the worst request's median |decode - prefill| is "
            f"{min(worst):.6g} in every draw, at or over {bar}")
        results[name] = {'n': entry['n'], 'draws': worst}
    return results


def requests_below(reading, bar):
    """The worst request's median of one structurally valid draw (every admitted request answered)."""
    name = reading.get('name')
    assert reading.get('structurally_valid'), f"{name}: malformed reading is terminal: {reading.get('error')}"
    medians = reading['medians']
    assert len(medians) == reading['n'] and len(reading.get('requests') or []) == reading['n'], \
        f"{name}: {len(medians)} results for {reading['n']} admitted requests"
    assert all(type(m) in (int, float) and not math.isnan(m) for m in medians), f'{name}: malformed medians'
    return max(medians)


def production(ranks, *, source, root, storage_limit, ordinary_allowance, coverage_record,
               concurrency, layers):
    assert len(ranks) == 4
    for rank, record in enumerate(ranks):
        assert record['rank'] == rank
        assert len(record['bindings']) == layers
        for binding_record in record['bindings']:
            assert identity(binding_record['identity'], root) == source, 'microbenchmark differs from actual serving module'
        production_storage(record, expert_limit=storage_limit,
                           ordinary_allowance=ordinary_allowance)
        residency = record['residency_after_execution']
        assert any(r['mode']=='EXTEND' for r in residency)
        assert any(r['mode']=='DECODE' and r['graph'] for r in residency)
        for observation in residency:
            assert finite(observation['pool_storage_bytes']) > 0
            production_storage(observation, expert_limit=storage_limit,
                               ordinary_allowance=ordinary_allowance)
        executions = record['executions']
        assert any(e['mode'] == 'EXTEND' and not e['graph'] for e in executions)
    return coverage_runtime(coverage_record, concurrency=concurrency, layers=layers, source=source)


def read_ranks(root):
    return [json.loads((Path(root)/f'rank-{r}.json').read_text()) for r in range(4)]
