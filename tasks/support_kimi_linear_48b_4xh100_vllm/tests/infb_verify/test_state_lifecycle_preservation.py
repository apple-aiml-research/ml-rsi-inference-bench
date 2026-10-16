#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Generic preservation checks for the state-lifecycle revision: existing cache ownership and existing scheduler
continuation and release (pass-to-pass).

Both run on the CPU, on tiny in-memory requests and cache specifications, against the candidate's engine and the
restored upstream tests: no checkpoint, model download or serving deployment. They preserve pre-existing
behaviour at the generic interfaces that Kimi prefix-state support changes -- the KV cache manager's
ownership of shared prefix blocks and the scheduler's resumed requests -- and pass on the unsolved prepared base.
They require no prefix support for another model family. They assert semantics only: never physical block IDs,
free-list order, a preemption victim, a schedule-step count, a Kimi caching mode or a count of intermediate
checkpoints. Same-step readiness of a prefix is the Kimi fail-to-pass obligation, not this module's: every
prefix looked up here was completed first.

The request and configuration helpers are upstream's own (`tests/v1/core/test_prefix_caching.py` and
`tests/v1/core/utils.py`), loaded from the restored `/app/tests` package. The scheduler fixture's model
configuration is a tiny local OPT `config.json` written by this module (no tokenizer is initialized), since the
image is offline and carries no test checkpoint; nothing is downloaded.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

from infb_verify import records

APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
BLOCK = 16


def _upstream(name: str):
    """A module of upstream's test package, from the restored /app/tests (as serve.py loads tests.utils)."""
    if "tests" not in sys.modules:
        spec = importlib.machinery.PathFinder.find_spec("tests", [str(APP_DIR)])
        assert spec is not None, f"no upstream test package under {APP_DIR}"
        package = importlib.util.module_from_spec(spec)
        sys.modules["tests"] = package
        if spec.loader is not None:
            spec.loader.exec_module(package)
    loaded = list(getattr(sys.modules["tests"], "__path__", None) or [])
    assert loaded == [str(APP_DIR / "tests")], f"the upstream test package resolved to {loaded}"
    return importlib.import_module(name)


def _raised(exc: BaseException) -> str:
    """An exception that ended a subcase, with the frame that raised it: an engine-internal assertion is evidence
    too, but it is not the same reading as a semantic check of this module."""
    frame = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
    where = f" at {frame.filename}:{frame.lineno}" if frame else ""
    return f"{type(exc).__name__}: {exc}{where}"


# ── existing cache ownership ─────────────────────────────────────────────────────────────────────


def _spec(kind: str):
    import torch  # noqa: PLC0415
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec  # noqa: PLC0415

    if kind == "full_attention":
        return FullAttentionSpec(BLOCK, 1, 1, torch.float32)
    # The existing default MambaSpec (mamba_type "mamba2"), with no caching-mode argument.
    return MambaSpec(block_size=BLOCK, shapes=((1, 4),), dtypes=(torch.float32,))


def _ownership_case(helpers, kind: str, hash_fn, case: dict) -> None:
    """One spec and one hash function: a 65-token completed prefix, two borrowers, release and turnover.

    Fills `case["readings"]` and `case["problems"]` as it goes, so a later exception keeps every earlier finding."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: PLC0415
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec  # noqa: PLC0415

    helpers.init_none_hash(hash_fn)
    config = KVCacheConfig(num_blocks=16, kv_cache_tensors=[],
                           kv_cache_groups=[KVCacheGroupSpec(["layer"], _spec(kind))])
    manager = KVCacheManager(config, max_model_len=8192, enable_caching=True)
    pool = manager.block_pool
    usable = pool.get_num_free_blocks()
    problems: list[str] = case["problems"]
    readings: dict = case["readings"]
    readings["usable_blocks"] = usable
    live: dict = {}

    def check(condition, message):
        if not condition:
            problems.append(message)

    def blocks(request):
        return list(manager.get_blocks(request.request_id).blocks[0])

    def owners_hold():
        """Every real block a live request holds counts exactly its live owners; null entries are the pool's."""
        counts: dict[int, int] = {}
        objects = {}
        for name, request in live.items():
            for block in blocks(request):
                if block.is_null:
                    check(block is pool.null_block, f"{name}: a null entry that is not the pool's null block")
                    continue
                counts[id(block)] = counts.get(id(block), 0) + 1
                objects[id(block)] = block
        for key, block in objects.items():
            check(block.ref_cnt == counts[key],
                  f"a block held by {counts[key]} live request(s) has reference count {block.ref_cnt}")
        # A block a live request holds is never available for allocation; every other block is.
        check(pool.get_num_free_blocks() == usable - len(objects),
              f"{pool.get_num_free_blocks()} blocks are free while live requests hold {len(objects)} of {usable}")
        return {id(b) for b in objects.values()}

    def allocate(request, new_tokens, hit=0, hit_blocks=None):
        got = manager.allocate_slots(request, new_tokens, hit, hit_blocks)
        check(got is not None, f"{request.request_id}: allocation of {new_tokens} tokens failed")
        return got

    def complete(request):
        request.num_computed_tokens = request.num_tokens

    def lookup(name, tokens):
        request = helpers.make_request(name, tokens, BLOCK, hash_fn)
        hit_blocks, hit = manager.get_computed_blocks(request)
        return request, hit_blocks, hit

    prefix = [1000 + i for i in range(65)]
    producer, hit_blocks, hit = lookup("producer", prefix)
    check(hit == 0, f"the producer hit {hit} tokens of an empty cache")
    allocate(producer, len(prefix), hit, hit_blocks)
    complete(producer)  # its computation is done: the prefix is published complete before anyone looks it up
    live["producer"] = producer
    produced = blocks(producer)

    borrowers = {}
    for name, base in (("borrower_a", 2000), ("borrower_b", 3000)):
        request, hit_blocks, hit = lookup(name, prefix + [base + i for i in range(15)])
        readings[f"{name}_hit_tokens"] = hit
        check(hit == 64, f"{name}: reused {hit} tokens, not the completed prefix's 64 block-aligned tokens")
        allocate(request, request.num_tokens - hit, hit, hit_blocks)
        complete(request)
        live[name] = request
        borrowers[name] = request
        held = blocks(request)
        boundary = hit // BLOCK - 1
        if hit == 64 and len(held) > boundary:
            check(not held[boundary].is_null and held[boundary] is produced[boundary],
                  f"{name}: the reused state at the prefix boundary is not the producer's completed block")
            for index in range(boundary):
                check(held[index].is_null or held[index] is produced[index],
                      f"{name}: reused entry {index} is neither null nor the producer's block")
    owners_hold()

    _, _, other = lookup("different_prefix", [999] + prefix[1:] + [4000 + i for i in range(15)])
    readings["different_prefix_hit_tokens"] = other
    check(other == 0, f"a different token prefix hit {other} tokens")

    # Release one borrower, then the producer; the remaining borrower keeps the shared prefix alive.
    manager.free(borrowers["borrower_a"])
    del live["borrower_a"]
    owners_hold()
    manager.free(producer)
    del live["producer"]
    kept = owners_hold()
    remaining = borrowers["borrower_b"]
    for step in range(20):  # continue it across a block boundary
        remaining.append_output_token_ids(5000 + step)
        allocate(remaining, 1)
        remaining.num_computed_tokens += 1
        kept = owners_hold()

    # Finite-pool turnover while it is live: no allocation may hand out a block it holds, and its prefix stays.
    fillers = []
    for index in range(usable):
        filler, hit_blocks, hit = lookup(f"filler_{index}", [6000 + 16 * index + i for i in range(16)])
        got = manager.allocate_slots(filler, 16, hit, hit_blocks)
        if got is None:
            break
        complete(filler)
        fillers.append(filler)
        taken = {id(b) for b in got.blocks[0] if not b.is_null}
        check(not taken & kept, f"{filler.request_id}: a block held by the live borrower was handed out")
    readings["turnover_requests"] = len(fillers)
    check(fillers and pool.get_num_free_blocks() == 0, "the turnover did not cycle the finite pool")
    _, _, during = lookup("lookup_while_owned", prefix + [7000 + i for i in range(15)])
    readings["hit_while_owned_after_turnover"] = during
    check(during == 64, f"the live borrower's prefix hit {during} tokens after turnover, not 64")
    for filler in fillers:
        manager.free(filler)

    manager.free(remaining)
    del live["borrower_b"]
    check(pool.get_num_free_blocks() == usable,
          f"{pool.get_num_free_blocks()} of {usable} blocks are free after every request was released")
    _, _, after = lookup("lookup_after_release", prefix + [8000 + i for i in range(15)])
    readings["hit_after_release"] = after
    check(after == 64, f"the released prefix hit {after} tokens before any reallocation, not 64")

    # Turnover after release: one request takes every block, so every released prefix block is reallocated.
    sweep, hit_blocks, hit = lookup("sweep", [9000 + i for i in range(usable * BLOCK)])
    allocate(sweep, sweep.num_tokens, hit, hit_blocks)
    complete(sweep)
    manager.free(sweep)
    _, _, evicted = lookup("lookup_after_turnover", prefix + [10000 + i for i in range(15)])
    readings["hit_after_turnover"] = evicted
    check(evicted == 0, f"a prefix whose blocks were all reallocated still hit {evicted} tokens")

    # Rewarm, and a new borrower hits the completed prefix again.
    rewarm, hit_blocks, hit = lookup("rewarm", prefix)
    allocate(rewarm, rewarm.num_tokens - hit, hit, hit_blocks)
    complete(rewarm)
    again, hit_blocks, hit = lookup("borrower_after_rewarm", prefix + [11000 + i for i in range(15)])
    readings["hit_after_rewarm"] = hit
    check(hit == 64, f"the rewarmed prefix hit {hit} tokens, not 64")
    allocate(again, again.num_tokens - hit, hit, hit_blocks)
    manager.free(again)
    manager.free(rewarm)
    check(pool.get_num_free_blocks() == usable, "blocks were not all released at the end")


def test_existing_cache_ownership():
    """A completed prefix shared by two borrowers keeps its ownership and reuse semantics through release and
    turnover, for the existing full-attention spec and the existing default Mamba spec, under both hash functions.
    """
    helpers = _upstream("tests.v1.core.test_prefix_caching")

    def body(record: dict):
        record["cases"] = cases = []
        for kind in ("full_attention", "mamba2"):
            for hash_fn in (helpers.sha256, helpers.sha256_cbor):
                case = {"spec": kind, "hash": getattr(hash_fn, "__name__", str(hash_fn)), "readings": {},
                        "problems": []}
                cases.append(case)
                try:
                    _ownership_case(helpers, kind, hash_fn, case)
                except Exception as exc:  # noqa: BLE001 - every subcase is kept; the node fails below
                    case["problems"].append(_raised(exc))
        problems = list(dict.fromkeys(f"{c['spec']}/{c['hash']}: {p}" for c in cases for p in c["problems"]))
        record["problems"] = problems
        return not problems, problems

    problems = records.run("test_existing_cache_ownership", {"block_size": BLOCK, "num_blocks": 16}, body)
    assert not problems, "\n".join(problems)


# ── existing scheduler continuation and release ──────────────────────────────────────────────────

#: A minimal local OPT configuration: the scheduler needs a model configuration, never weights or a tokenizer.
TINY_OPT = {
    "architectures": ["OPTForCausalLM"], "model_type": "opt", "vocab_size": 50272, "hidden_size": 64,
    "num_hidden_layers": 1, "ffn_dim": 128, "num_attention_heads": 2, "max_position_embeddings": 2048,
    "word_embed_proj_dim": 64, "do_layer_norm_before": True, "torch_dtype": "float16",
    "bos_token_id": 2, "eos_token_id": 2, "pad_token_id": 1,
}


def _model_output(requests, tokens):
    from vllm.v1.outputs import ModelRunnerOutput  # noqa: PLC0415

    return ModelRunnerOutput(
        req_ids=[r.request_id for r in requests],
        req_id_to_index={r.request_id: i for i, r in enumerate(requests)},
        sampled_token_ids=tokens, logprobs=None, prompt_logprobs_dict={}, pooler_output=[],
    )


def _scheduler_case(utils, model_dir: Path, prefix_caching: bool, ending: str, case: dict) -> None:
    from vllm.v1.request import RequestStatus  # noqa: PLC0415

    num_blocks = 5
    scheduler = utils.create_scheduler(model=str(model_dir), max_num_seqs=2, max_num_batched_tokens=200,
                                       num_blocks=num_blocks, block_size=BLOCK, enable_prefix_caching=prefix_caching,
                                       skip_tokenizer_init=True)
    manager = scheduler.kv_cache_manager
    usable = manager.block_pool.get_num_free_blocks()
    problems: list[str] = case["problems"]
    readings: dict = case["readings"]

    def check(condition, message):
        if not condition:
            problems.append(message)

    # The competing 32-token request, scheduled and given one output through the ordinary path.
    competing = utils.create_requests(num_requests=2, num_tokens=32)[1]
    scheduler.add_request(competing)
    output = scheduler.schedule()
    scheduler.update_from_output(output, _model_output([competing], [[100]]))

    # The legal preempted state, built from the ordinary request, queue and cache-manager objects: a 30-token
    # prompt with two generated tokens in its history, its old allocation freed, its computed count reset and
    # status PREEMPTED in the waiting queue.
    resumed = utils.create_requests(num_requests=1, num_tokens=30)[0]
    scheduler.add_request(resumed)
    resumed.append_output_token_ids([101, 102])
    held = manager.allocate_slots(resumed, resumed.num_tokens)
    check(held is not None, "the preempted request's earlier allocation could not be made")
    manager.free(resumed)
    resumed.status = RequestStatus.PREEMPTED
    resumed.num_computed_tokens = 0
    resumed.num_preemptions += 1
    history = list(resumed.all_token_ids)
    readings["history_tokens"] = len(history)
    check(history[30:] == [101, 102] and len(history) == 32, "the constructed history is not 30 + 2 tokens")

    # The competing request finishes through the native API; the scheduler produces the resumed work.
    scheduler.finish_requests(competing.request_id, RequestStatus.FINISHED_STOPPED)
    rid = resumed.request_id
    steps, done, first_resume = 0, False, None
    while not done and steps < 8:
        output = scheduler.schedule()
        steps += 1
        cached = output.scheduled_cached_reqs
        if rid not in cached.req_ids:
            check(False, f"schedule {steps}: the preempted request was not scheduled as a cached request")
            break
        index = cached.req_ids.index(rid)
        scheduled = output.num_scheduled_tokens[rid]
        computed = cached.num_computed_tokens[index]
        if first_resume is None:
            sent = cached.all_token_ids.get(rid)
            first_resume = {"resumed": rid in cached.resumed_req_ids, "computed": computed, "scheduled": scheduled,
                            "sent_tokens": None if sent is None else len(sent),
                            "num_output_tokens": cached.num_output_tokens[index],
                            "new_block_ids": cached.new_block_ids[index]}
            check(first_resume["resumed"], "the preempted request was not identified as resumed")
            check(sent is not None and list(sent) == history[: computed + scheduled],
                  f"the resumed request's token history was not sent intact: "
                  f"{None if sent is None else len(sent)} of {computed + scheduled} tokens")
            check(cached.num_output_tokens[index] == 2,
                  f"the resumed request reports {cached.num_output_tokens[index]} output tokens, not 2")
            owned = manager.get_block_ids(rid)
            sent_blocks = cached.new_block_ids[index]
            check(sent_blocks is not None and [list(g) for g in sent_blocks] == [list(g) for g in owned],
                  "the resumed request was not sent its complete replacement block list")
            for group in owned:
                for block_id in group:
                    block = manager.block_pool.blocks[block_id]
                    check(not block.is_null and block.ref_cnt >= 1,
                          f"the resumed request holds block {block_id} without a live reference")
            check(sum(len(g) for g in owned) >= -(-(computed + scheduled) // BLOCK),
                  "the resumed request holds fewer blocks than its scheduled tokens need")
        done = computed + scheduled >= len(history)
        scheduler.update_from_output(output, _model_output([resumed], [[103] if done else []]))
    readings["first_resume"] = first_resume
    check(first_resume is not None, "the preempted request was never resumed")
    check(resumed.status == RequestStatus.RUNNING, f"the resumed request's status is {resumed.status}")
    check(list(resumed.output_token_ids) == [101, 102, 103],
          f"the next model output was not appended exactly once: {list(resumed.output_token_ids)}")
    check(list(resumed.all_token_ids) == history + [103], "the prompt and generated history did not survive")

    if ending == "finish":
        output = scheduler.schedule()
        scheduler.update_from_output(output, _model_output([resumed], [[utils.EOS_TOKEN_ID]]))
        check(resumed.status == RequestStatus.FINISHED_STOPPED,
              f"a sampled end-of-sequence token finished the request with {resumed.status}")
    else:
        scheduler.finish_requests(resumed.request_id, RequestStatus.FINISHED_ABORTED)
        check(resumed.status == RequestStatus.FINISHED_ABORTED, f"the abort left status {resumed.status}")
    output = scheduler.schedule()
    readings["finished_reported"] = resumed.request_id in output.finished_req_ids
    check(readings["finished_reported"], "the finished request was not reported to the workers")
    check(resumed.request_id not in scheduler.requests, "the finished request is still tracked")
    check(scheduler.get_num_unfinished_requests() == 0, "requests remain unfinished")
    check(manager.block_pool.get_num_free_blocks() == usable,
          f"{manager.block_pool.get_num_free_blocks()} of {usable} blocks are free after release")


def test_existing_scheduler_continuation_and_release(tmp_path):
    """A preempted request with generated history resumes with its complete token history, valid replacement
    blocks and exactly-once output appends, then releases everything on normal finish and on abort."""
    utils = _upstream("tests.v1.core.utils")
    model_dir = tmp_path / "tiny-opt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(TINY_OPT))

    def body(record: dict):
        record["cases"] = cases = []
        for prefix_caching in (False, True):
            for ending in ("finish", "abort"):
                case = {"prefix_caching": prefix_caching, "ending": ending, "readings": {}, "problems": []}
                cases.append(case)
                try:
                    _scheduler_case(utils, model_dir, prefix_caching, ending, case)
                except Exception as exc:  # noqa: BLE001 - every subcase is kept; the node fails below
                    case["problems"].append(_raised(exc))
        problems = list(dict.fromkeys(f"prefix_caching={c['prefix_caching']}/{c['ending']}: {p}"
                                      for c in cases for p in c["problems"]))
        record["problems"] = problems
        return not problems, problems

    problems = records.run("test_existing_scheduler_continuation_and_release", {"block_size": BLOCK}, body)
    assert not problems, "\n".join(problems)
