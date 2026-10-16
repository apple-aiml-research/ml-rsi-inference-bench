#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Serving state-lifecycle observation. A separate evidence pass: never imported by a component
timing process, and efficiency stays uninstrumented.

Read-only. The observer records what the engine scheduled and executed and the canonical KDA states
at the boundaries it selects; it never chooses tokens, repairs metadata, forces a cache hit,
performs a restore or writes a state. Like `binding.py`, it is installed when the worker extension
is imported, before load_model and graph capture:

* `GPUModelRunner.execute_model`: each step's `SchedulerOutput` -- new, resumed
  (`scheduled_cached_reqs.resumed_req_ids`), running and finished requests with their computed,
  scheduled and output counts. After the step, a producer snapshot of every target request whose
  computed count reached a multiple of B, read from the state row its KDA operation used. For each
  multi-token step of a request with generated history (a recompute), its block ids by position in
  each state group (`state_blocks`, 0 for the null block).
* the runner module's `set_forward_context`, entered by every forward, real or dummy: the actual
  graph mode and extent, DP token counts and live rows. For a real step, after any native restore
  and before the model runs, a consumer snapshot of every target request that starts from a
  restored state (a prefix hit, or a resumption from a retained state), read from the state row the
  mapped KDA operation is about to consume.
* `GPUModelRunner._dummy_run`: marks DP dummy/idle forwards.
* the 20 mapped KDA state-consumption calls (`layer_adapter.discover`, `lifecycle.consume_entry`): one device
  counter each, captured with the graphs, so a replay is counted on the device rather than inferred from
  Python calls.

A layer's state-index entry for a row is a slot, or a list of slots (a block table, upstream's mamba prefix-caching
layout). From a list, a snapshot reads the column that holds the state after the token count it is taken at:
(count - 1) // b, with b the state group's block size -- upstream's `cdiv(count, b) - 1` for the last computed
(consumer) and the last scheduled (producer) token (`mamba2_attn.py` and `mamba_mixer2.py` at 1994de99). An entry
of neither shape, or a column outside its list, is recorded as a `mapping_shape` fault: the candidate's mapping. A
list with no MambaSpec state group to give b is the observer's own limit, recorded as `observer_incompatible`.

A consumer snapshot is compared with the producer snapshot of the same token prefix bitwise (SHA-256 of each
canonical view) and numerically (`mathematics.error`, per layer and view), from the producer's retained copy.

One deliberate exception to read-only, armed only by a case's `begin` payload (`intervene`), for the separate
restored-state correctness pass: after the consumer snapshot of the named request's first
restore is taken, and before the model runs, a constant is added to the named view(s) of its consumed state row
in the named layers -- the binding's state intervention, applied at the restore. Evidence runs never carry it.

Snapshots are device copies taken outside any capture and hashed after the step; only the counter
increment runs inside a CUDA graph. Each worker writes its own file,
`<root>/<deployment>/<case>/wave-<n>/rank-dp<d>-tp<t>-<device uuid>.jsonl` (schema
`infb-kimi-state-lifecycle/1`). The client's `begin`/`flush`/`end` RPCs bound a case; the engine
broadcasts them to every DP engine and waits for all, so their return is the completion barrier.
The RPC result itself carries only the first engine's ranks and is never coverage evidence.
"""

import contextlib
import functools
import hashlib
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import layer_adapter as adapter
from binding import source_root

SCHEMA = 'infb-kimi-state-lifecycle/1'
BLOCK = 512
VIEWS = ('conv_q', 'conv_k', 'conv_v', 'recurrent')
MAX_EVENTS = 200_000
# Per rank and case. Device copies live within one step and CPU copies are capped by MAX_RETAINED,
# so memory does not grow with this cap.
MAX_SNAPSHOTS = 4_096
MAX_RETAINED = 64
MAX_DUMPS = 4
MAX_SHAPE_FAULTS = 8
FLUSH_EVERY = 2_000

_ACTIVE = None  # this worker's observer: one model runner per worker process


def token_digest(tokens):
    return hashlib.sha256(np.asarray(list(tokens), dtype='<i4').tobytes()).hexdigest()


def state_slot(entry, tokens, block):
    """The slot holding one layer's state after `tokens` tokens, from that layer's state-index entry for a row: an int
    is the slot itself; a list is a block table, whose column (tokens - 1) // block holds it (upstream's
    `cdiv(tokens, block) - 1`). None when the entry is neither shape, or the column is outside the list."""
    if isinstance(entry, bool):
        return None
    if isinstance(entry, int):
        return entry
    if (isinstance(entry, list) and isinstance(tokens, int) and tokens > 0 and isinstance(block, int) and block > 0
            and all(isinstance(x, int) and not isinstance(x, bool) for x in entry)):
        column = (tokens - 1) // block
        return entry[column] if column < len(entry) else None
    return None


def _sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def classify_consume(entries, digests, step):
    """How a consumed state relates to every producer of its prefix.

    A prefix's content can be produced by more than one request in a run, legitimately and with genuinely
    different states: two requests may compute the same tokens with different chunking, and an implementation
    that may not reuse a state still being produced in the same step has to compute its own copy. Keeping one
    entry per prefix and overwriting it judged a restore against whichever request produced that content last,
    which need not be the checkpoint the request actually read.

      - `completed` are the producers whose step is strictly BEFORE the consume's step. A block still being
        produced in the same step is not a completed checkpoint.
      - `exact` is a completed producer whose state is bitwise equal: the checkpoint the request read.
      - `same_step_exact` is a producer at or after this step whose state is bitwise equal, when no completed
        one is. That is a same-step read **by identity**, which the contract forbids as an act, however close
        the state happens to be to an earlier checkpoint.
      - `candidates` carries every completed producer with the views that differ from it, so the published
        state policy can be applied to each: the restore passes if ANY completed producer is within policy.
        A candidate whose retained copy was evicted under MAX_RETAINED is unavailable, never a pass.

    Pure, so a repository guard can run it against recorded events with no engine.
    """
    completed = [p for p in entries if p.get('step', 0) < step]
    later = [p for p in entries if p.get('step', 0) >= step]
    exact = next((p for p in reversed(completed) if p['digests'] == digests), None)
    same_step_exact = None if exact is not None else next(
        (p for p in reversed(later) if p['digests'] == digests), None)
    candidates = [{'producer': p,
                   'mismatched': [[l, VIEWS[v]]
                                  for l, (mine, theirs) in enumerate(zip(digests, p['digests']))
                                  for v in range(len(VIEWS)) if mine[v] != theirs[v]]}
                  for p in completed]
    return {'completed': completed, 'exact': exact, 'same_step_exact': same_step_exact,
            'candidates': candidates}


def _error(actual, expected):
    # `mathematics.error` (the KDA policy's statistic) for finite tensors; a non-finite consumed state is recorded as
    # such -- the candidate's evidence, never an observer error.
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        return {'max_abs': None, 'relative_rmse': None, 'finite': False}
    from mathematics import error
    return {**error(actual, expected), 'finite': True}


class Lifecycle:
    def __init__(self, runner):
        self.runner = runner
        self.layers = adapter.discover(runner.model, source_root(runner.model))
        self.device_calls = torch.zeros(len(self.layers), dtype=torch.int64, device=runner.device)
        self.python_calls = [0] * len(self.layers)
        for i, layer in enumerate(self.layers):
            self._wrap(i, layer)
        self.root = Path(os.environ.get('INFB_KDA_LIFECYCLE_DIR', '/logs/state-lifecycle'))
        self.deployment = os.environ.get('INFB_KDA_LIFECYCLE_DEPLOYMENT', 'unnamed')
        self.identity = None
        self.case = None
        self.step = None
        self.dummy = None
        self.errors = 0

    def _wrap(self, i, layer):
        # The state-consumption call (`lifecycle.consume_entry`, by default the inner `_forward` the layer's
        # opaque custom op runs), never the compiled outer `forward`.
        original = layer.consume
        counter = self.device_calls[i:i + 1]
        calls = self.python_calls

        @functools.wraps(original)
        def observed(*args, **kwargs):
            counter.add_(1)
            if not torch.cuda.is_current_stream_capturing():
                calls[i] += 1
            return original(*args, **kwargs)

        setattr(layer.consume_owner, layer.consume_name, observed)

    # ── identity and records ────────────────────────────────────────────────────────────────────

    def _identify(self):
        from vllm.distributed.parallel_state import get_dp_group, get_tp_group
        parallel = self.runner.vllm_config.parallel_config
        try:
            dp_group_rank = get_dp_group().rank_in_group
        except Exception:  # noqa: BLE001 -- recorded as absent, never guessed
            dp_group_rank = None
        uuid = getattr(torch.cuda.get_device_properties(self.runner.device), 'uuid', None)
        return {'deployment': self.deployment, 'dp_rank': parallel.data_parallel_rank,
                'dp_group_rank': dp_group_rank, 'dp_size': parallel.data_parallel_size,
                'tp_rank': get_tp_group().rank_in_group, 'tp_size': get_tp_group().world_size,
                'device_uuid': f'GPU-{uuid}' if uuid is not None else None,
                'host': socket.gethostname(), 'pid': os.getpid()}

    def _engine(self):
        cache = self.runner.cache_config
        compilation = self.runner.vllm_config.compilation_config
        return {'enable_prefix_caching': cache.enable_prefix_caching, 'block_size': cache.block_size,
                'mamba_block_size': getattr(cache, 'mamba_block_size', None),
                'mamba_cache_mode': getattr(cache, 'mamba_cache_mode', None),
                'cudagraph_mode': str(compilation.cudagraph_mode),
                'cudagraph_capture_sizes': list(compilation.cudagraph_capture_sizes or []),
                'max_num_seqs': self.runner.scheduler_config.max_num_seqs}

    def _emit(self, event, **fields):
        if self.case is None:
            return
        if self.written + len(self.buffer) >= MAX_EVENTS:
            if not self.overflow:
                self.overflow = True
                self.buffer.append(self._record('fault', reason='observer_overflow', limit=MAX_EVENTS))
            return
        self.buffer.append(self._record(event, **fields))
        if len(self.buffer) >= FLUSH_EVERY:
            self._flush()

    def _record(self, event, **fields):
        record = {'schema': SCHEMA, 'event': event, 'case_id': self.case, 'wave': self.wave,
                  'deployment': self.deployment, 'dp_rank': self.identity['dp_rank'],
                  'tp_rank': self.identity['tp_rank'], 'device_uuid': self.identity['device_uuid'],
                  'step': self.step_index, 'time': time.time()}
        record.update(fields)
        return record

    def _flush(self):
        for record in self.buffer:
            self.file.write(json.dumps(record, separators=(',', ':')) + '\n')
        self.written += len(self.buffer)
        self.buffer = []
        self.file.flush()

    def guarded(self, method, *args):
        """Run one hook; an observer error becomes a recorded fault and never changes execution."""
        try:
            return method(*args)
        except Exception as error:  # noqa: BLE001
            self.errors += 1
            if self.errors <= 8:
                self._emit('fault', reason='observer_error', hook=method.__name__, error=repr(error),
                           traceback=traceback.format_exc()[-4000:])
            return None

    def _target(self, request_id):
        return self.targets is None or any(t in request_id for t in self.targets)

    def _counters(self):
        device = self.device_calls.tolist()
        python = list(self.python_calls)
        delta = ([a - b for a, b in zip(device, self.last_device)],
                 [a - b for a, b in zip(python, self.last_python)])
        self.last_device, self.last_python = device, python
        return {'layer_calls_device': [min(delta[0]), max(delta[0])],
                'layer_calls_python': [min(delta[1]), max(delta[1])]}

    # ── RPC ─────────────────────────────────────────────────────────────────────────────────────

    def command(self, command, payload):
        self.identity = self.identity or self._identify()
        if command == 'begin':
            if self.case is not None:
                self._end('superseded')
            return self._begin(payload)
        if command == 'flush':
            if self.case is not None:
                self._flush()
            return {'armed': self.case is not None, **self.identity}
        if command == 'end':
            return self._end('end')
        if command == 'report':
            return {'armed': self.case is not None, **self.identity}
        raise ValueError(command)

    def _begin(self, payload):
        self.case, self.wave = str(payload['case']), int(payload.get('wave', 1))
        self.targets = payload.get('targets')
        intervene = payload.get('intervene')
        if intervene is not None:
            views = intervene['views']
            if not views or any(v not in VIEWS for v in views) or not intervene.get('request'):
                raise ValueError(f'intervention {intervene!r}')
            intervene = {'request': str(intervene['request']), 'views': list(views),
                         'layers': [int(l) for l in intervene['layers']], 'add': float(intervene['add']),
                         'applied': None}
        self.intervene = intervene
        directory = self.root / self.deployment / self.case / f'wave-{self.wave}'
        directory.mkdir(parents=True, exist_ok=True)
        name = 'rank-dp{dp_rank}-tp{tp_rank}-{device_uuid}'.format(**self.identity)
        duplicate = False
        try:
            self.file = open(directory / f'{name}.jsonl', 'x')
        except FileExistsError:
            # Two workers with one identity, or a reused case/wave: both files stay visible.
            duplicate = True
            self.file = open(directory / f'{name}-dup-{os.getpid()}.jsonl', 'x')
        self.directory, self.path = directory, Path(self.file.name)
        self.buffer, self.written, self.overflow = [], 0, False
        self.snapshots, self.dumps, self.step_index, self.errors, self.shape_faults = 0, 0, 0, 0, 0
        self.producers, self.retained, self.prompts = {}, {}, {}
        self.seq = 0
        # Per KV-cache group: spec type and block size, for allocation tracking and prefix blocks.
        groups = self.runner.kv_cache_config.kv_cache_groups
        self.groups = [(type(group.kv_cache_spec).__name__, group.kv_cache_spec.block_size) for group in groups]
        # The state groups: those whose spec is a MambaSpec (a subclass included). Their block size is b for the
        # column a block-table row is read at, and their block ids are the recompute's `state_blocks`.
        try:
            from vllm.v1.kv_cache_interface import MambaSpec  # noqa: PLC0415
            self.state_groups = [i for i, group in enumerate(groups) if isinstance(group.kv_cache_spec, MambaSpec)]
        except ImportError:
            self.state_groups = [i for i, (kind, _) in enumerate(self.groups) if kind == 'MambaSpec']
        self.state_block = self.groups[self.state_groups[0]][1] if self.state_groups else None
        torch.cuda.synchronize()
        self.last_device, self.last_python = self.device_calls.tolist(), list(self.python_calls)
        self._emit('begin', engine=self._engine(), identity=self.identity, targets=self.targets,
                   duplicate=duplicate, groups=self.groups, state_groups=self.state_groups,
                   layers=[layer.identity() for layer in self.layers], intervene=self.intervene)
        return {'path': str(self.path), 'duplicate': duplicate, **self.identity}

    def _end(self, reason):
        if self.case is None:
            return {'path': None, **self.identity}
        summary = {'reason': reason, 'events': self.written + len(self.buffer) + 1,
                   'snapshots': self.snapshots, 'producers': len(self.producers),
                   'overflow': self.overflow, 'dumps': self.dumps, 'errors': self.errors,
                   'intervention_applied': (self.intervene or {}).get('applied')}
        self._emit('end', **summary)
        self._flush()
        self.file.close()
        path, self.case, self.step = self.path, None, None
        return {'path': str(path), **summary, **self.identity}

    # ── hooks ───────────────────────────────────────────────────────────────────────────────────

    def before_step(self, scheduler_output):
        self.step_index += 1
        cached = scheduler_output.scheduled_cached_reqs
        scheduled = scheduler_output.num_scheduled_tokens
        plan = {}
        # `history` is the accumulated token count (prompt + generated) when scheduled. The scheduler's
        # `all_token_ids` is cut at computed + scheduled (scheduler.py, `_make_cached_request_data`), so
        # its length is recorded separately as `sent_tokens`, never as the history.
        # `fresh_blocks`: per group, the block ids newly allocated to the request this step, i.e. not
        # a prefix hit; a fresh allocation of a block that held a cached prefix is its eviction.
        for request in scheduler_output.scheduled_new_reqs:
            computed = request.num_computed_tokens
            plan[request.req_id] = {'kind': 'new', 'computed_before': computed,
                                    'scheduled_tokens': scheduled.get(request.req_id, 0),
                                    'output_count': 0, 'history': len(request.prompt_token_ids or ()),
                                    'sent_tokens': None, 'fresh_blocks': self._fresh(request.block_ids, computed)}
        for k, rid in enumerate(cached.req_ids):
            resumed = rid in cached.resumed_req_ids
            state = self.runner.requests.get(rid)
            output_count = cached.num_output_tokens[k]
            sent = cached.all_token_ids.get(rid)
            new_ids = cached.new_block_ids[k]
            if new_ids is None:
                fresh = None
            elif resumed:
                # A resumed request is sent all of its blocks, prefix hits included.
                fresh = self._fresh(new_ids, cached.num_computed_tokens[k])
            else:
                fresh = [[block for block in ids if block] for ids in new_ids]
            plan[rid] = {'kind': 'resumed' if resumed else 'running',
                         'computed_before': cached.num_computed_tokens[k],
                         'scheduled_tokens': scheduled.get(rid, 0), 'output_count': output_count,
                         'history': (len(state.prompt_token_ids) + output_count) if state is not None else None,
                         'sent_tokens': len(sent) if sent is not None else None, 'fresh_blocks': fresh}
        self.step = {'plan': plan, 'finished': sorted(scheduler_output.finished_req_ids),
                     'total_tokens': scheduler_output.total_num_scheduled_tokens,
                     'forward': None, 'rows': {}, 'indices': None, 'consumers': {}, 'consumer_slots': {}}

    def _fresh(self, block_ids, computed):
        # Positions below computed // block size are prefix hits (or null placeholders); 0 is the null block.
        return [[block for position, block in enumerate(ids) if position >= computed // size and block]
                for ids, (_, size) in zip(block_ids, self.groups)]

    def _prefix_blocks(self, rid, length):
        # The blocks a hit of `length` tokens needs: every attention block below it, and the single state
        # block that ends at it.
        state = self.runner.requests[rid]
        blocks = []
        for ids, (kind, size) in zip(state.block_ids, self.groups):
            n = length // size
            if kind == 'MambaSpec':
                blocks.append([ids[n - 1]] if 0 < n <= len(ids) else [])
            else:
                blocks.append(list(ids[:n]))
        return blocks

    def on_forward(self, attn_metadata, kwargs):
        mode = kwargs.get('cudagraph_runtime_mode')
        descriptor = kwargs.get('batch_descriptor')
        across = kwargs.get('num_tokens_across_dp')
        tokens = kwargs.get('num_tokens')
        name = getattr(mode, 'name', None) or 'NONE'
        extent = descriptor.num_tokens if descriptor is not None else (tokens if name != 'NONE' else None)
        forward = {'graph_mode': name, 'graph_extent': extent, 'forward_tokens': tokens,
                   'dp_tokens': [int(x) for x in across.tolist()] if across is not None else None}
        if self.dummy is not None:
            self._emit('dummy', live_rows=0, requested_tokens=self.dummy[0], uniform_decode=self.dummy[1],
                       **forward)
            return
        if self.step is None:
            return
        self.step['forward'] = forward
        plan = self.step['plan']
        n = len(plan)
        if n == 0:
            return
        if not isinstance(attn_metadata, dict) or any(layer.prefix not in attn_metadata
                                                        for layer in self.layers):
            self._emit('fault', reason='observer_incompatible', detail='no KDA metadata in a real step')
            return
        req_ids = list(self.runner.input_batch.req_ids[:n])
        # The documented mapping paths of each layer's native metadata (`lifecycle.state_indices`,
        # `lifecycle.query_start_loc`).
        indices = torch.stack([layer.metadata_view(attn_metadata[layer.prefix], 'state_indices')[:n].long()
                               for layer in self.layers]).tolist()
        first = self.layers[0]
        starts = first.metadata_view(attn_metadata[first.prefix], 'query_start_loc')[:n + 1].tolist()
        lengths = [starts[r + 1] - starts[r] for r in range(n)]
        if (sorted(req_ids) != sorted(plan)
                or any(lengths[r] != plan[rid]['scheduled_tokens'] for r, rid in enumerate(req_ids))):
            self._emit('fault', reason='observer_incompatible', detail='rows do not map to scheduled requests',
                       req_ids=req_ids, query_lengths=lengths,
                       scheduled={rid: p['scheduled_tokens'] for rid, p in plan.items()})
            return
        self.step['rows'] = {rid: r for r, rid in enumerate(req_ids)}
        self.step['indices'] = indices
        for rid, entry in plan.items():
            if entry['kind'] in ('new', 'resumed') and entry['computed_before'] > 0 and self._target(rid):
                row = [indices[l][self.step['rows'][rid]] for l in range(len(self.layers))]
                # The restored state is the one after the computed tokens: a block-table row's last computed column.
                slots = self._slots(row, entry['computed_before'])
                copy = self._copy(slots, entry['computed_before']) if slots is not None else None
                if copy is not None:
                    self.step['consumers'][rid] = copy
                    self.step['consumer_slots'][rid] = slots
                    if (self.intervene is not None and self.intervene['applied'] is None
                            and self.intervene['request'] in rid):
                        self._intervene(rid, slots)

    def _intervene(self, rid, row):
        # The restored-state intervention: after the consumer snapshot, before the model runs.
        spec = self.intervene
        before, after = [], []
        for l in spec['layers']:
            states = self.layers[l].states
            for name in spec['views']:
                view = states[VIEWS.index(name)][row[l]]
                before.append(_sha(view))
                view.add_(spec['add'])
                after.append(_sha(view))
        spec['applied'] = {'request_id': rid, 'step': self.step_index, 'state_indices': row}
        self._emit('intervene', request_id=rid, layers=spec['layers'], views=spec['views'], add=spec['add'],
                   state_indices=row, sha256_before=before, sha256_after=after)

    def dummy_done(self):
        self._emit('dummy_done', **self._counters())

    def _slots(self, indices, tokens):
        """Each layer's slot after `tokens` tokens (`state_slot`). None, with a `mapping_shape` fault (at most
        MAX_SHAPE_FAULTS per case), when an entry is neither a slot nor a block-table row covering that column. A
        block-table row with no MambaSpec state group to give its block size is the observer's own limit
        (`observer_incompatible`), never the candidate's."""
        slots = [state_slot(entry, tokens, self.state_block) for entry in indices]
        bad = next((l for l, slot in enumerate(slots) if slot is None), None)
        if bad is None:
            return slots
        if self.shape_faults < MAX_SHAPE_FAULTS:
            self.shape_faults += 1
            unread = self.state_block is None and isinstance(indices[bad], list)
            self._emit('fault', reason='observer_incompatible' if unread else 'mapping_shape', layer=bad,
                       tokens=tokens, block=self.state_block,
                       detail=f'layer {bad}: the state index entry {indices[bad]!r:.200} names no slot after {tokens} '
                              f'tokens at block size {self.state_block}')
        return None

    def _copy(self, indices, tokens=None):
        slots = self._slots(indices, tokens)
        if slots is None:
            return None
        if min(slots) < 0:
            self._emit('fault', reason='observer_incompatible', detail='negative state index for a live row',
                       indices=slots)
            return None
        if self.snapshots >= MAX_SNAPSHOTS:
            if not self.overflow:
                self.overflow = True
                self._emit('fault', reason='observer_overflow', limit=MAX_SNAPSHOTS, what='snapshots')
            return None
        self.snapshots += 1
        return [[view[index].clone() for view in layer.states] for layer, index in zip(self.layers, slots)]

    def after_step(self):
        step, self.step = self.step, None
        if step is None or self.case is None:
            return
        counters = self._counters()  # synchronizes the step's stream
        forward = step['forward'] or {'graph_mode': None, 'graph_extent': None, 'forward_tokens': None,
                                      'dp_tokens': None}
        plan, rows = step['plan'], step['rows']
        common = {'live_rows': len(plan), **forward}
        self._emit('step', total_tokens=step['total_tokens'], finished=step['finished'], **common,
                   **counters)
        for rid, entry in plan.items():
            request = self.runner.requests.get(rid)
            if rid not in self.prompts and request is not None:
                self.prompts[rid] = token_digest(request.prompt_token_ids)
            fields = {'request_id': rid, 'prompt_digest': self.prompts.get(rid), **entry,
                      'row': rows.get(rid),
                      'state_indices': self._indices(step, rid) if rid in rows and step['indices'] else None,
                      **common}
            if entry['output_count'] and entry['scheduled_tokens'] > 1 and request is not None:
                # A recompute of generated history: where its checkpoints can live, by position, per state group.
                fields['state_blocks'] = [list(request.block_ids[g]) for g in self.state_groups]
            self._emit('schedule', **fields)
            before = entry['computed_before']
            after = before + entry['scheduled_tokens']
            produce = after > 0 and after % BLOCK == 0 and rid in rows and self._target(rid)
            if request is None or (rid not in step['consumers'] and not produce):
                continue
            tokens = list(request.prompt_token_ids) + list(request.output_token_ids)
            if rid in step['consumers']:
                self._consume(rid, fields, step, tokens[:before])
            if produce and len(tokens) >= after:
                self._produce(rid, fields, step, tokens[:after], after)
        for rid in step['finished']:
            self._emit('finish', request_id=rid, prompt_digest=self.prompts.pop(rid, None))

    def _indices(self, step, rid):
        return [step['indices'][l][step['rows'][rid]] for l in range(len(self.layers))]

    def _produce(self, rid, fields, step, prefix, length):
        # The produced state is the one after `length` tokens: a block-table row's last scheduled column.
        slots = self._slots(self._indices(step, rid), length)
        copy = self._copy(slots, length) if slots is not None else None
        if copy is None:
            return
        digests = [[_sha(view) for view in views] for views in copy]
        key = (length, token_digest(prefix))
        # Every producer of a prefix is kept, not only the last. Two requests can compute the same prefix
        # content in the same run -- legitimately, with different chunking, so with different states -- and a
        # consumer restores one particular checkpoint. Keying one entry per prefix and overwriting it compared a
        # restore against whichever request happened to produce that content last.
        self.seq += 1
        entry = {'request_id': rid, 'digests': digests, 'step': self.step_index, 'state_indices': slots,
                 'seq': self.seq}
        self.producers.setdefault(key, []).append(entry)
        self.retained[(key, self.seq)] = [[view.cpu() for view in views] for views in copy]
        while len(self.retained) > MAX_RETAINED:
            self.retained.pop(next(iter(self.retained)))
        # The schedule fields already carry the row's state indices; the producer's own reading (the slots it read)
        # replaces them.
        self._emit('produce', **{**fields, 'state_indices': slots}, prefix_length=length, prefix_digest=key[1],
                   prefix_blocks=self._prefix_blocks(rid, length), state_sha256=digests,
                   combined_sha256=hashlib.sha256(json.dumps(digests).encode()).hexdigest())

    def _consume(self, rid, fields, step, prefix):
        copy = step['consumers'][rid]
        digests = [[_sha(view) for view in views] for views in copy]
        key = (len(prefix), token_digest(prefix))
        c = classify_consume(self.producers.get(key, []), digests, self.step_index)

        # Every completed producer is compared, not only the closest: a producer with fewer differing views can
        # still exceed the policy where another is within it. The errors go out per candidate and the
        # published policy is applied by the verifier, which owns the bars.
        candidates, unavailable = [], 0
        for cand in c['candidates']:
            p = cand['producer']
            retained = self.retained.get((key, p['seq']))
            if retained is None and cand['mismatched']:
                unavailable += 1
            errors = None
            if cand['mismatched'] and retained is not None:
                errors = [{'layer': l, 'view': view, **_error(copy[l][VIEWS.index(view)].float().cpu(),
                                                              retained[l][VIEWS.index(view)].float())}
                          for l, view in cand['mismatched']]
            candidates.append({'request_id': p['request_id'], 'step': p['step'], 'seq': p['seq'],
                               'state_indices': p['state_indices'],
                               'mismatched': cand['mismatched'], 'errors': errors,
                               'copy_available': retained is not None})

        by_seq = {p['seq']: p for p in c['completed']}
        if c['exact'] is not None:
            producer, mismatched, errors = c['exact'], [], None
        elif candidates:
            # The one reported is the closest by its own worst differing view, among those that can be measured;
            # a candidate whose copy was evicted sorts last and is never reported as a pass.
            def worst(entry):
                vals = [e['relative_rmse'] for e in (entry['errors'] or []) if e.get('relative_rmse') is not None]
                return (0 if entry['errors'] else 1, max(vals) if vals else float('inf'), len(entry['mismatched']))
            best = min(candidates, key=worst)
            producer, mismatched, errors = by_seq[best['seq']], best['mismatched'], best['errors']
        else:
            producer, mismatched, errors = None, None, None

        dump = None
        if mismatched and self.dumps < MAX_DUMPS:
            self.dumps += 1
            dump = self.directory / ('dump-{}-{}-dp{}-tp{}.pt'.format(
                rid, len(prefix), self.identity['dp_rank'], self.identity['tp_rank']))
            torch.save({'consumer': [[view.cpu() for view in views] for views in copy],
                        'producer': self.retained.get((key, producer['seq'])) if producer else None}, dump)
        slots = step['consumer_slots'].get(rid)
        self._emit('consume', **{**fields, 'state_indices': slots}, prefix_length=len(prefix),
                   prefix_digest=key[1], state_sha256=digests,
                   combined_sha256=hashlib.sha256(json.dumps(digests).encode()).hexdigest(),
                   producer=(None if producer is None else
                             {'request_id': producer['request_id'], 'step': producer['step'],
                              'state_indices': producer['state_indices']}),
                   match=(None if producer is None else not mismatched), mismatched=mismatched,
                   state_errors=errors,
                   # How many producers of this prefix completed before this step, which of them this state is
                   # bitwise equal to, and -- when none is -- whether it is bitwise equal to one at or after
                   # this step, which is a same-step read by identity.
                   completed_producers=len(c['completed']),
                   exact_completed_producer=(c['exact']['request_id'] if c['exact'] is not None else None),
                   same_step_exact_producer=(c['same_step_exact']['request_id']
                                             if c['same_step_exact'] is not None else None),
                   candidate_producers=candidates,
                   completed_without_copy=unavailable,
                   numerical_available=(None if producer is None else (not mismatched or errors is not None)),
                   aliases_producer_rows=(None if producer is None else slots == producer['state_indices']),
                   dump=str(dump) if dump else None)


def install_startup_hook():
    """Called when the verifier worker extension is loaded, before load_model/capture."""
    import vllm.v1.worker.gpu_model_runner as runner_module
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    load = GPUModelRunner.load_model

    @functools.wraps(load)
    def load_and_observe(runner, *args, **kwargs):
        global _ACTIVE
        result = load(runner, *args, **kwargs)
        runner.infb_kda_lifecycle = _ACTIVE = Lifecycle(runner)
        return result

    execute = GPUModelRunner.execute_model

    @functools.wraps(execute)
    def execute_and_observe(runner, scheduler_output, *args, **kwargs):
        observer = getattr(runner, 'infb_kda_lifecycle', None)
        if observer is None or observer.case is None:
            return execute(runner, scheduler_output, *args, **kwargs)
        observer.guarded(observer.before_step, scheduler_output)
        try:
            return execute(runner, scheduler_output, *args, **kwargs)
        finally:
            with torch.inference_mode():
                observer.guarded(observer.after_step)

    dummy = GPUModelRunner._dummy_run

    @functools.wraps(dummy)
    def dummy_and_observe(runner, num_tokens, *args, **kwargs):
        observer = getattr(runner, 'infb_kda_lifecycle', None)
        if observer is None or observer.case is None:
            return dummy(runner, num_tokens, *args, **kwargs)
        observer.dummy = (num_tokens, bool(kwargs.get('uniform_decode', False)))
        try:
            return dummy(runner, num_tokens, *args, **kwargs)
        finally:
            observer.dummy = None
            observer.guarded(observer.dummy_done)

    forward_context = runner_module.set_forward_context

    @contextlib.contextmanager
    def observed_forward_context(attn_metadata, *args, **kwargs):
        with forward_context(attn_metadata, *args, **kwargs):
            observer = _ACTIVE
            if observer is not None and observer.case is not None:
                observer.guarded(observer.on_forward, attn_metadata, kwargs)
            yield

    GPUModelRunner.load_model = load_and_observe
    GPUModelRunner.execute_model = execute_and_observe
    GPUModelRunner._dummy_run = dummy_and_observe
    runner_module.set_forward_context = observed_forward_context
