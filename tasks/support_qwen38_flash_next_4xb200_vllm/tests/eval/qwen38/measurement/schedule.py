#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Drive a logical request schedule through the loaded engine runner and capture native regions.

The observer acts as the engine's scheduler for a few private requests while the server is idle:
the engine's own `KVCacheManager` allocates blocks from the scheduler view of the runner's KV cache
configuration, and each step is the `SchedulerOutput` the scheduler builds for the v2 runner. The
runner (`execute_model`, `sample_tokens`), its attention metadata builders and the model are the
candidate's; the observer constructs no metadata and no state.

A case names requests by their history length (tokens already computed before the measured work)
and active tokens per measured step. History is prepared natively by prefilling it in steps within
the scheduler's token and sequence budgets. Measured prefill is one step; decode and speculative
cases run `steps` measured steps; one more probe step follows so that state written by the last
measured step is read by a later valid continuation. In a joining case, a fresh request prefills in
the named measured steps alongside the speculative requests (a mixed speculative/prefill iteration)
and finishes after its step.

Runner steps execute without compiled graphs (the engine's own `skip_compiled` forward-context flag)
so that module methods run in Python and can be observed; timing replays happen later, in the
required regime (`measurement.timing`). Generated and drafted tokens follow a planned token stream,
and in planned-acceptance speculative cases each verification accepts a planned number of drafts:
an admissible probabilistic-sampling outcome, identical for every implementation paired on the case.
In native-acceptance cases the engine's own drafts and verification decide progress.
"""
from __future__ import annotations

import contextlib
import copy
import time
import uuid


class ScheduleError(RuntimeError):
    """The engine could not execute the planned schedule (for example, no KV blocks)."""


class _Proxy:
    """Delegate every attribute to the engine object; `__call__` goes through `hook`."""

    def __init__(self, inner, hook):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_hook", hook)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)

    def __call__(self, *args, **kwargs):
        return self._hook(self._inner, *args, **kwargs)


class _SpeculatorProxy(_Proxy):
    def propose(self, input_batch, *args, **kwargs):
        drafts = self._inner.propose(input_batch, *args, **kwargs)
        return self._hook(drafts, input_batch)


@contextlib.contextmanager
def eager_steps():
    """Runner steps execute module methods in Python instead of compiled graphs."""
    import vllm.v1.worker.gpu.model_runner as runner_module
    original = runner_module.set_forward_context

    def forced(*args, **kwargs):
        kwargs["skip_compiled"] = True
        return original(*args, **kwargs)

    runner_module.set_forward_context = forced
    try:
        yield
    finally:
        runner_module.set_forward_context = original


class Planner:
    """Planned generated tokens and speculative acceptance, by request."""

    def __init__(self, streams: dict, *, acceptance, vocabulary: int, draft_length: int):
        self.streams = streams            # req_id -> planned tokens after the prompt
        self.acceptance = acceptance      # "native" or {req_id: [accepted drafts per step]}
        self.vocabulary = vocabulary
        self.draft_length = draft_length
        self.emitted = {r: [] for r in streams}
        self.accepted = {r: [] for r in streams}
        self.spec_steps = dict.fromkeys(streams, 0)
        self.prefilling = set()           # requests prefilling in the current step (joining prefills)
        self.one_shot = set()             # joining requests: one prefill step, no drafts

    def next_accept(self, req_id):
        pattern = self.acceptance[req_id]
        return pattern[self.spec_steps[req_id] % len(pattern)]

    def sample(self, inner, logits, input_batch, *args, **kwargs):
        """Run the engine sampler, then emit the plan's tokens (and planned acceptance)."""
        import torch
        out = inner(logits, input_batch, *args, **kwargs)
        rows = list(input_batch.req_ids)
        if not any(r in self.streams for r in rows):
            return out
        speculative = bool(args) or "draft_logits" in kwargs
        planned = self.acceptance != "native"
        sampled = out.sampled_token_ids.clone()
        num_sampled = out.num_sampled.clone() if out.num_sampled is not None else None
        num_rejected = out.num_rejected.clone() if getattr(out, "num_rejected", None) is not None else None
        counts = num_sampled.tolist() if num_sampled is not None else [1] * len(rows)
        for row, req_id in enumerate(rows):
            if req_id not in self.streams or counts[row] <= 0:
                continue  # a partial prefill emits nothing
            done = len(self.emitted[req_id])
            spec_row = speculative and req_id not in self.prefilling
            if spec_row and planned:
                k = self.next_accept(req_id)
                tokens = self.streams[req_id][done:done + k + 1]
                sampled[row, :k + 1] = torch.tensor(tokens, dtype=sampled.dtype, device=sampled.device)
                sampled[row, k + 1:] = -1
                num_sampled[row] = k + 1
                if num_rejected is not None:
                    num_rejected[row] = sampled.shape[1] - 1 - k
            elif spec_row:
                # Native progress: record the engine's own verification outcome.
                k = counts[row] - 1
                tokens = sampled[row, :k + 1].tolist()
            else:
                k, tokens = None, [self.streams[req_id][done]]
                sampled[row, 0] = tokens[0]
            self.emitted[req_id].extend(tokens)
            if spec_row:
                self.accepted[req_id].append(k)
                self.spec_steps[req_id] += 1
        out.sampled_token_ids = sampled
        if num_sampled is not None:
            out.num_sampled = num_sampled
        if num_rejected is not None:
            out.num_rejected = num_rejected
        return out

    def drafts(self, drafts, input_batch):
        import torch
        rows = list(input_batch.req_ids)
        if self.acceptance == "native" or not any(r in self.streams for r in rows):
            return drafts
        drafts = drafts.clone()
        for row, req_id in enumerate(rows):
            if req_id not in self.streams or req_id in self.one_shot:
                continue
            done = len(self.emitted[req_id])
            k = self.next_accept(req_id)
            stream = self.streams[req_id]
            want = []
            for j in range(drafts.shape[1]):
                token = stream[min(done + j, len(stream) - 1)]
                # A draft past the planned acceptance differs from the planned token: a rejected
                # draft is never the token the verification then emits.
                want.append(token if j < k else (token + 1) % self.vocabulary)
            drafts[row] = torch.tensor(want, dtype=drafts.dtype, device=drafts.device)
        return drafts


class PrivateScheduler:
    """The engine scheduler's protocol for the observer's own requests."""

    def __init__(self, worker):
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.kv_cache_utils import (generate_scheduler_kv_cache_config,
                                                 resolve_kv_cache_block_sizes)
        self.worker = worker
        self.runner = worker.model_runner
        self.cfg = worker.vllm_config
        self.kv_config = generate_scheduler_kv_cache_config([copy.deepcopy(self.runner.kv_cache_config)])
        block, hash_block = resolve_kv_cache_block_sizes(self.kv_config, self.cfg)
        self.block_size = block
        self.kvm = KVCacheManager(kv_cache_config=self.kv_config, max_model_len=self.cfg.model_config.max_model_len,
                                  scheduler_block_size=block, hash_block_size=hash_block,
                                  enable_caching=False, num_prefill_lookahead=0)
        self.requests, self.new, self.finished = {}, set(), set()
        self.budget = self.cfg.scheduler_config.max_num_batched_tokens
        self.max_seqs = self.cfg.scheduler_config.max_num_seqs
        self.spec = self.cfg.num_speculative_tokens or 0

    def add(self, req_id: str, prompt: list[int], max_tokens: int):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        self.requests[req_id] = Request(request_id=req_id, prompt_token_ids=list(prompt),
                                        sampling_params=params, pooling_params=None)
        self.new.add(req_id)

    def group_blocks(self, layer_name, req_ids) -> list[int]:
        """Blocks the KV group holding `layer_name` assigned to the requests (all groups if unknown)."""
        groups = [g for g, group in enumerate(self.kv_config.kv_cache_groups) if layer_name in group.layer_names]
        out = set()
        for req_id in req_ids:
            if req_id in self.requests:
                ids = self.kvm.get_block_ids(req_id)
                for g in (groups or range(len(ids))):
                    out.update(ids[g])
        return sorted(out)

    def blocks(self, req_ids) -> list[int]:
        out = set()
        for req_id in req_ids:
            if req_id in self.requests:
                for group in self.kvm.get_block_ids(req_id):
                    out.update(group)
        return sorted(out)

    def output(self, schedule: dict, *, drafts: int = 0):
        """The SchedulerOutput for {req_id: tokens to compute}; `drafts` per decoding request."""
        from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
        if sum(schedule.values()) + drafts * sum(1 for r in schedule if r not in self.new) > self.budget \
                or len(schedule) > self.max_seqs:
            raise ScheduleError("step exceeds the scheduler's token or sequence budget")
        new_reqs, cached = [], CachedRequestData.make_empty()
        num_scheduled, spec_tokens = {}, {}
        for req_id, n in schedule.items():
            request = self.requests[req_id]
            extra = drafts if req_id not in self.new else 0
            blocks = self.kvm.allocate_slots(request, n + extra,
                                             num_lookahead_tokens=self.cfg.num_lookahead_tokens)
            if blocks is None:
                raise ScheduleError(f"{req_id}: no KV cache blocks for {n + extra} tokens")
            if req_id in self.new:
                new_reqs.append(NewRequestData.from_request(request, blocks.get_block_ids(),
                                                            request._all_token_ids))
                self.new.discard(req_id)
            else:
                cached.req_ids.append(req_id)
                cached.new_block_ids.append(blocks.get_block_ids(allow_none=True))
                cached.num_computed_tokens.append(request.num_computed_tokens)
                cached.num_output_tokens.append(request.num_output_tokens)
                if extra:
                    spec_tokens[req_id] = [-1] * extra
            num_scheduled[req_id] = n + extra
        out = SchedulerOutput(
            scheduled_new_reqs=new_reqs, scheduled_cached_reqs=cached,
            num_scheduled_tokens=num_scheduled, total_num_scheduled_tokens=sum(num_scheduled.values()),
            scheduled_spec_decode_tokens=spec_tokens, scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * len(self.kv_config.kv_cache_groups),
            finished_req_ids=set(self.finished), free_encoder_mm_hashes=[],
            new_block_ids_to_zero=self.kvm.take_new_block_ids() or None)
        self.finished.clear()
        return out

    def advance(self, req_id: str, computed: int, emitted: list[int]):
        request = self.requests[req_id]
        request.num_computed_tokens += computed
        for token in emitted:
            request.append_output_token_ids(token)

    def finish(self, req_ids):
        for req_id in list(req_ids):
            request = self.requests.pop(req_id, None)
            if request is not None:
                self.kvm.free(request)
                self.finished.add(req_id)
                self.new.discard(req_id)


def run(worker, case: dict, prompts: dict, streams: dict, regions: list[tuple[dict, dict]], *,
        vocabulary: int, intervene=None, record_model_output: bool = False, downstream=None) -> dict:
    """Execute one case and capture each region during it.

    `prompts[req_id]` are the prompt tokens (history plus prefill tokens) and `streams[req_id]` the
    planned tokens after the prompt. History steps record only each region's hidden-state input;
    measured and probe steps record complete calls. State rows are saved at the phase boundaries a
    replay needs: `pre` (every block the measured steps use, as it was before the first measured
    step, including blocks allocated later, saved at allocation), `post` (after the last measured
    step) and `probe_pre` (blocks first allocated for the probe step). `intervene`, when given,
    perturbs each region's output in the measured steps only."""
    import torch
    from . import native

    runner = worker.model_runner
    sched = PrivateScheduler(worker)
    mode, lengths, histories = case["mode"], case["lengths"], case["histories"]
    steps = case.get("steps", 1)
    spec = sched.spec if mode == "spec" else 0
    if mode == "spec" and not spec:
        raise ScheduleError("speculative cases need the MTP3 recipe")
    if mode in ("decode", "spec") and min(histories) < 1:
        raise ScheduleError("decode and speculative requests need a nonempty history")
    joining = case.get("joining") or []
    join_ids = [r for r in prompts if "-join-" in r]
    if len(join_ids) != len(joining):
        raise ScheduleError("one joining prompt per joining prefill")
    join_at = {step: req_id for (step, _), req_id in zip(joining, join_ids)}
    ids = [r for r in prompts if r not in join_ids]
    acceptance = case.get("acceptance", "native")
    planner = Planner(streams, acceptance=acceptance, vocabulary=vocabulary, draft_length=spec)
    planner.one_shot = set(join_ids)
    for req_id in ids:
        sched.add(req_id, prompts[req_id], max_tokens=len(streams[req_id]))
    captures = [native.Capture(entry, region) for entry, region in regions]
    layout, executed, outputs, downstream_inputs = [], [], [], []
    downstream_step = {"on": False}
    if downstream is not None:
        original_forward = downstream.forward

        def observed_forward(*args, **kwargs):
            if downstream_step["on"]:
                first = args[0] if args else next(iter(kwargs.values()))
                downstream_inputs.append(first.detach().clone())
            return original_forward(*args, **kwargs)
        downstream.forward = observed_forward
    snapshots = {"pre": [[] for _ in captures], "post": None, "probe_pre": [[] for _ in captures]}

    def step(schedule, *, drafts=0, phase):
        before_groups = {layer: set(sched.group_blocks(layer, ids))
                         for capture in captures for _, _, layer in capture.state}
        so = sched.output(schedule, drafts=drafts)
        if phase in ("measured", "probe"):
            target = snapshots["pre"] if phase == "measured" else snapshots["probe_pre"]
            first = phase == "measured" and not any(target)

            def rows(layer_name, first=first, before=before_groups):
                now = set(sched.group_blocks(layer_name, ids))
                return sorted(now if first else now - before.get(layer_name, set()))
            for i, capture in enumerate(captures):
                capture.mode = "full"
                capture.intervene = intervene if phase == "measured" else None
                # Rows as the region first sees them in this step: its groups' blocks already holding
                # history at the first measured step, and blocks first allocated for this step.
                capture.pending = (target[i], rows)
        else:
            for capture in captures:
                capture.mode, capture.intervene, capture.pending = "history", None, None
        before = {r: len(planner.emitted[r]) for r in schedule}
        started = time.perf_counter()
        downstream_step["on"] = phase == "measured"
        runner.execute_model(so, is_profile=True)
        downstream_step["on"] = False
        state = runner.execute_model_state
        batch = state.input_batch
        layout.append({"phase": phase, "req_ids": list(batch.req_ids),
                       "query_start_loc": [int(x) for x in batch.query_start_loc_np[:len(batch.req_ids) + 1]],
                       "num_computed_tokens": [int(x) for x in batch.num_computed_tokens_np[:len(batch.req_ids)]],
                       "input_ids": batch.input_ids[:batch.num_tokens].tolist(),
                       "calls": [len(c.calls) for c in captures]})
        if record_model_output and phase == "measured":
            outputs.append(state.hidden_states[:batch.num_tokens].detach().clone())
        runner.sample_tokens(None)
        torch.cuda.synchronize()
        record = {"phase": phase, "seconds": time.perf_counter() - started, "requests": {}}
        for req_id, n in schedule.items():
            emitted = planner.emitted[req_id][before[req_id]:]
            computed = len(emitted) if req_id in so.scheduled_spec_decode_tokens else n
            sched.advance(req_id, computed, emitted)
            record["requests"][req_id] = {"scheduled": so.num_scheduled_tokens[req_id], "emitted": emitted,
                                          "speculative": req_id in so.scheduled_spec_decode_tokens}
        executed.append(record)
        for capture in captures:
            capture.mode, capture.intervene, capture.pending = None, None, None

    samplers = {"sampler": runner.sampler}
    if getattr(runner, "rejection_sampler", None) is not None:
        samplers["rejection_sampler"] = runner.rejection_sampler
    speculator = getattr(runner, "speculator", None)
    try:
        runner.sampler = _Proxy(samplers["sampler"], planner.sample)
        if "rejection_sampler" in samplers:
            runner.rejection_sampler = _Proxy(samplers["rejection_sampler"], planner.sample)
        if speculator is not None and acceptance != "native":
            runner.speculator = _SpeculatorProxy(speculator, planner.drafts)
        with eager_steps(), contextlib.ExitStack() as stack:
            for capture in captures:
                stack.enter_context(capture)
            # Native history preparation, within the scheduler's token and sequence budgets.
            remaining = {r: h for r, h in zip(ids, histories)}
            while any(remaining.values()):
                schedule, room = {}, sched.budget
                for req_id in ids:
                    if remaining[req_id] and room and len(schedule) < sched.max_seqs:
                        take = min(remaining[req_id], room)
                        schedule[req_id] = take
                        remaining[req_id] -= take
                        room -= take
                step(schedule, phase="history")
            for index in range(steps):
                if mode == "prefill":
                    step({r: n for r, n in zip(ids, lengths)}, phase="measured")
                elif index in join_at:
                    # A mixed speculative/prefill iteration: a fresh request prefills alongside.
                    joiner = join_at[index]
                    sched.add(joiner, prompts[joiner], max_tokens=len(streams[joiner]))
                    planner.prefilling = {joiner}
                    step({**{r: 1 for r in ids}, joiner: len(prompts[joiner])}, drafts=spec, phase="measured")
                    planner.prefilling = set()
                    sched.finish([joiner])
                else:
                    step({r: 1 for r in ids}, drafts=spec, phase="measured")
            snapshots["post"] = [native.StateRows(c.state, lambda layer: sched.group_blocks(layer, ids))
                                 if c.state else None for c in captures]
            # The probe step reads the state written by the last measured step.
            step({r: 1 for r in ids}, drafts=spec, phase="probe")
            blocks = sched.blocks(ids)
            group_blocks = {layer: sched.group_blocks(layer, ids) for c in captures for _, _, layer in c.state}
    finally:
        runner.sampler = samplers["sampler"]
        if "rejection_sampler" in samplers:
            runner.rejection_sampler = samplers["rejection_sampler"]
        if speculator is not None:
            runner.speculator = speculator
        sched.finish(ids)
        runner.execute_model(sched.output({}))
        if downstream is not None:
            del downstream.forward
    return {"downstream_inputs": downstream_inputs, "mode": mode, "captures": captures, "layout": layout,
            "executed": executed, "blocks": blocks, "group_blocks": group_blocks,
            "snapshots": snapshots, "accepted": planner.accepted, "emitted": planner.emitted,
            "model_outputs": outputs}
