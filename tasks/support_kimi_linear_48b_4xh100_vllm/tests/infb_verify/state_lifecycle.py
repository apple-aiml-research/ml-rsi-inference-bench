#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The state-lifecycle sessions: bounded scenario producers and the late cold reference.

    cd /app && PYTHONPATH=/tests python -P -m infb_verify.state_lifecycle --session tp4|dp2|apc|pressure
    python -I /tests/infb_verify/state_lifecycle.py --session reference

Each candidate session launches one deployment of the evaluation tree's `launch.py` through `serve.launch` with the
private lifecycle observer installed (`infb_kda_private/lifecycle.py`, per-worker records under
`<root>/observer/<deployment>/<case>/wave-<n>/`), runs the fixed families of `/tests/hidden/state-lifecycle.json`
in order, records every request and response, and tears the server down. The `reference` session then serves the
independent reference tree at the cache-disabled `tp4` recipe through the efficiency kit's pristine-snapshot route
and teacher-forces every scored target's prompt plus its exact continuation. Every raw record is written before
any verdict; `test_state_lifecycle.py` reads them in one final process.

A session never decides a verdict. It records what happened and, when it cannot go on, why: `candidate` (the
deployment did not start, a request failed or exhausted its published limit, the recipe is not served),
`observation` (a trusted observation could not establish what it must: its own reason codes) or `reference`
(the independent reference could not launch or read). A session that a bound ends leaves its record `running`;
the final verdicts read that as blocked by the group, never as a pass.

The generation reader, the teacher-forced reader, the window enumeration and the rule are the evaluation tree's
own `state_lifecycle/client.py` (the trusted copy), unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

TESTS = Path(os.environ.get("TESTS_DIR", "/tests"))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

ROOT = Path(os.environ.get("INFB_STATE_LIFECYCLE_DIR", "/logs/state-lifecycle"))
OBSERVER = ROOT / "observer"
MANIFEST = json.loads((TESTS / "hidden" / "state-lifecycle.json").read_text())
CFG = json.loads((TESTS / "config.json").read_text())
X = CFG["infb"]
B = MANIFEST["block_tokens"]
WAVES = MANIFEST["coverage_waves"]
PROBES = list(MANIFEST["probe_token_ids"])
FAMILIES = MANIFEST["families"]
SESSIONS = MANIFEST["sessions"]
EVAL = TESTS / "eval/kimi_linear"
EXPECTED_RANKS = {"tp4": [(0, t) for t in range(4)], "apc": [(0, t) for t in range(4)],
                  "apc-pressure": [(0, t) for t in range(4)], "dp2": [(d, t) for d in range(2) for t in range(4)]}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LC = _load("infb_state_lifecycle_client", EVAL / "state_lifecycle/client.py")


def log(message: str) -> None:
    print(f"[state-lifecycle {time.strftime('%H:%M:%S', time.gmtime())}Z] {message}", flush=True)


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, default=str) + "\n")
    temporary.replace(path)
    return path


def read_json(path: Path):
    return json.loads(path.read_text())


class SessionFailure(Exception):
    """A session cannot go on: `kind` is candidate, observation or reference; `reason` is an observation-fault
    code."""

    def __init__(self, kind: str, message: str, reason: str | None = None, evidence=None):
        super().__init__(message)
        self.kind, self.reason, self.evidence = kind, reason, None if evidence is None else str(evidence)


# ── the hidden pool ──────────────────────────────────────────────────────────────────────────────


class Pool:
    """Contiguous spans of the hidden natural-text pool; each span's offset is a function of the seed and its key."""

    def __init__(self):
        spec = MANIFEST["pool"]
        path = TESTS / spec["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != spec["sha256"]:
            raise SessionFailure("observation", f"the hidden pool {path} is {digest}, not {spec['sha256']}",
                                 "observer_incompatible", path)
        tokenizer = LC.serving.load_tokenizer(spec["tokenizer"])
        self.tokens: list[int] = []
        for doc in LC.serving.load_texts(path):
            self.tokens += tokenizer.encode(doc["text"])
        self.seed = spec["seed"]
        self.tokenizer_sha256 = tokenizer.sha256

    def span(self, key: str, n: int) -> list[int]:
        h = int(hashlib.sha256(f"{self.seed}/{key}".encode()).hexdigest(), 16)
        start = h % (len(self.tokens) - n)
        return self.tokens[start:start + n]

    def suffix(self, key: str, n: int, avoid_first=()) -> list[int]:
        """A span whose first token differs from every token in `avoid_first`."""
        for attempt in range(64):
            out = self.span(f"{key}/{attempt}", n)
            if out[0] not in avoid_first:
                return out
        raise SessionFailure("observation", f"no suffix for {key} avoiding {sorted(avoid_first)}",
                             "observer_incompatible")


# ── observer records ─────────────────────────────────────────────────────────────────────────────


def case_dir(deployment: str, case: str, wave: int) -> Path:
    return OBSERVER / deployment / case / f"wave-{wave}"


def load_case(deployment: str, case: str, wave: int) -> dict[str, list[dict]]:
    out = {}
    for path in sorted(case_dir(deployment, case, wave).glob("rank-*.jsonl")):
        events = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        events.append({"event": "fault", "reason": "unreadable_line", "line": line[:200]})
        out[path.name] = events
    return out


class Tail:
    """Incremental reader of one case/wave's worker files (new complete lines since the last read)."""

    def __init__(self, deployment: str, case: str, wave: int):
        self.directory = case_dir(deployment, case, wave)
        self.offsets: dict[str, int] = {}

    def new(self, tp: int | None = 0) -> list[dict]:
        out = []
        for path in sorted(self.directory.glob("rank-*.jsonl")):
            if tp is not None and f"-tp{tp}-" not in path.name:
                continue
            with path.open("rb") as fh:
                fh.seek(self.offsets.get(path.name, 0))
                data = fh.read()
            end = data.rfind(b"\n") + 1
            self.offsets[path.name] = self.offsets.get(path.name, 0) + end
            for line in data[:end].splitlines():
                if line.strip():
                    out.append(json.loads(line))
        return out


def same_request(rid: str, engine_id) -> bool:
    """Whether an engine request id is the one the client sent as `request_id` (vLLM: `cmpl-<id>-<index>`)."""
    engine_id = str(engine_id)
    return engine_id == rid or engine_id.startswith(f"cmpl-{rid}-")


def rank_problems(files: dict[str, list[dict]], deployment: str) -> tuple[list[str], list[str]]:
    """(rank-collection problems, observer faults) of one case/wave: the complete expected (dp, tp) set with distinct
    device UUIDs, one begin and one end per worker, and no observer fault."""
    collection, faults = [], []
    seen = []
    for name, events in files.items():
        begin = next((e for e in events if e.get("event") == "begin"), None)
        if begin is None:
            collection.append(f"{name}: no begin record")
            continue
        ident = begin["identity"]
        seen.append((ident["dp_rank"], ident["tp_rank"], ident["device_uuid"]))
        if begin.get("duplicate") or "-dup-" in name:
            collection.append(f"{name}: duplicate worker identity")
        if ident.get("dp_group_rank") is not None and ident["dp_group_rank"] != ident["dp_rank"]:
            collection.append(f"{name}: dp_rank {ident['dp_rank']} is not its DP group rank {ident['dp_group_rank']}")
        if not any(e.get("event") == "end" for e in events):
            collection.append(f"{name}: no end record")
        for e in events:
            if e.get("event") == "fault":
                faults.append(f"{name}: {e.get('reason')}: {str(e.get('detail') or e.get('error') or '')[:300]}")
    ranks = sorted((d, t) for d, t, _ in seen)
    if ranks != sorted(EXPECTED_RANKS[deployment]):
        collection.append(f"worker files cover {ranks}, not {sorted(EXPECTED_RANKS[deployment])}")
    uuids = [u for *_, u in seen]
    if None in uuids or len(set(uuids)) != len(uuids):
        collection.append(f"device UUIDs {uuids} are not distinct and present")
    return collection, faults


def mapping_fault(files: dict[str, list[dict]]) -> list[str]:
    """Observer errors raised while resolving the documented mapping's paths, and state-index entries the observer
    could not read a slot from (`mapping_shape`): the candidate's mapping (a malformed required mapping remains a
    candidate failure)."""
    out = []
    for name, events in files.items():
        for e in events:
            if (e.get("event") == "fault" and e.get("reason") == "observer_error"
                    and "layer_adapter.py" in (e.get("traceback") or "") and "in resolve" in (e.get("traceback") or "")):
                out.append(f"{name}: the lifecycle mapping does not resolve: {e.get('error')}")
            elif e.get("event") == "fault" and e.get("reason") == "mapping_shape":
                out.append(f"{name}: the lifecycle mapping's state indices name no slot: {e.get('detail')}")
    return out


# ── candidate-shaped observer values ──────────────────────────────────────────────────────


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _count(value) -> bool:
    return _int(value) and value >= 0


def shape_problem(e: dict) -> str | None:
    """What is malformed in one `schedule` record, among the fields the analysis reads; None when nothing is. The
    counts come from the engine's scheduler and the state indices from the candidate's mapping, so both are
    candidate-shaped: they are validated before any analysis reads them, and a malformed value is a judged problem
    stated with its value, never an exception and never a void."""
    for key in ("computed_before", "scheduled_tokens", "output_count"):
        if not _count(e.get(key)):
            return f"{key} {e.get(key)!r:.80} is not a token count"
    if e.get("history") is not None and not _count(e["history"]):
        return f"history {e['history']!r:.80} is not a token count"
    indices = e.get("state_indices")
    if indices is not None:
        if not isinstance(indices, list):
            return f"the state indices {indices!r:.120} are not one entry per layer"
        for layer, entry in enumerate(indices):
            if not (_int(entry) or (isinstance(entry, list) and all(_int(x) for x in entry))):
                return f"layer {layer}: the state index entry {entry!r:.120} is neither a slot nor a list of slots"
    blocks = e.get("state_blocks")
    if blocks is not None and not (isinstance(blocks, list)
                                   and all(isinstance(ids, list) and all(_count(x) for x in ids) for ids in blocks)):
        return f"the state blocks {blocks!r:.120} are not block ids by position"
    return None


def state_block(events: list[dict]) -> int | None:
    """The state group's block size b, from a worker file's `begin` record (its `groups` and `state_groups`)."""
    for e in events:
        if e.get("event") != "begin":
            continue
        groups = e.get("groups") or []
        state = e.get("state_groups")
        if state is None:
            state = [i for i, g in enumerate(groups) if isinstance(g, list) and g and g[0] == "MambaSpec"]
        for i in state:
            if _int(i) and 0 <= i < len(groups) and _count(groups[i][1]) and groups[i][1] > 0:
                return groups[i][1]
    return None


def row_slots(entry, computed: int, scheduled: int, block: int | None) -> tuple[int | None, set[int]]:
    """The slots one live row reads and writes in one layer during a step, from its state-index entry.

    An int is one slot, read and written in place; below 0 it is a padded row with none. A list is a block table: the
    step reads column (computed - 1) // b and writes columns computed // b through (computed + scheduled - 1) // b,
    upstream's last computed, first scheduled and last scheduled columns (`mamba2_attn.py` 205-216 and
    `mamba_mixer2.py` 745-747, 781-784 at 1994de99). Entries at or below 0 are null placeholders or padding and count
    as neither. An entry of neither shape (validated separately) has no slots."""
    if _int(entry):
        return (entry, {entry}) if entry >= 0 else (None, set())
    if not (isinstance(entry, list) and all(_int(x) for x in entry) and _int(block) and block > 0):
        return None, set()
    read = None
    if computed > 0 and (computed - 1) // block < len(entry) and entry[(computed - 1) // block] > 0:
        read = entry[(computed - 1) // block]
    last = min((computed + max(scheduled, 1) - 1) // block, len(entry) - 1)
    return read, {entry[c] for c in range(computed // block, last + 1) if entry[c] > 0}


def isolation_conflict(rows: list[dict], block: int | None) -> str | None:
    """The first state-row collision between two live requests of one step, or None.

    An int entry keeps the simple rule: two live requests on one slot consume one state row. For block tables a shared
    read-only checkpoint is legitimate (two restores from one cached prefix read the same block), so a collision is
    two requests writing one block, or one request writing the block another reads."""
    for layer in range(len(rows[0]["state_indices"])):
        entries = [r["state_indices"][layer] for r in rows]
        if all(_int(x) for x in entries):
            live = [x for x in entries if x >= 0]
            if len(set(live)) != len(live):
                return f"two live requests consume one state row of layer {layer}"
            continue
        writers: dict[int, str] = {}
        reads: list[tuple[int, str]] = []
        for r, entry in zip(rows, entries):
            read, writes = row_slots(entry, r["computed_before"], r["scheduled_tokens"], block)
            for slot in writes:
                if slot in writers and writers[slot] != r["request_id"]:
                    return (f"layer {layer}: requests {writers[slot]} and {r['request_id']} write one state block "
                            f"({slot})")
                writers[slot] = r["request_id"]
            if read is not None:
                reads.append((read, r["request_id"]))
        for slot, rid in reads:
            if slot in writers and writers[slot] != rid:
                return f"layer {layer}: request {writers[slot]} writes the state block ({slot}) request {rid} reads"
    return None


# ── the session ──────────────────────────────────────────────────────────────────────────────────


class Session:
    def __init__(self, name: str, server, pool: Pool):
        self.name = name
        self.deployment = SESSIONS[name]["deployment"]
        self.server = server
        self.base = server.url_root
        self.pool = pool
        self.model = LC.model_name(self.base)
        self.dir = ROOT / name
        self.case = None

    def lifecycle(self, command: str, **payload):
        import requests  # noqa: PLC0415

        response = requests.post(self.base + "/collective_rpc", timeout=LC.LIMITS["limits_s"]["action"] + 60,
                                 json={"method": "infb_kda_lifecycle", "timeout": LC.LIMITS["limits_s"]["action"],
                                       "kwargs": {"command": command, "payload": json.dumps(payload)}})
        if response.status_code != 200:
            raise SessionFailure("candidate", f"the lifecycle RPC {command} answered HTTP {response.status_code}: "
                                              f"{response.text[:300]}")
        return response.json()["results"]

    def begin(self, case: str, wave: int = 1, targets=None, intervene=None):
        self.case = (case, wave)
        payload = {"case": case, "wave": wave, "targets": targets}
        if intervene is not None:
            payload["intervene"] = intervene
        return self.lifecycle("begin", **payload)

    def flush(self):
        return self.lifecycle("flush")

    def end(self):
        self.case = None
        return self.lifecycle("end")

    def item(self, name: str, prompt, max_tokens: int, salt: str | None = None, **meta) -> dict:
        rid = f"{self.name}-{name}"
        return {"name": name, "request_id": rid, "prompt": list(prompt), "max_tokens": int(max_tokens),
                "salt": salt or f"salt-{rid}", **meta}

    async def _one(self, it: dict, deadline: float, on_first=None, cancel=False) -> dict:
        record = await LC.generate(self.base, it["prompt"], it["max_tokens"], salt=it["salt"],
                                   request_id=it["request_id"], model=self.model, deadline=deadline,
                                   on_first=on_first, cancel_after_first=cancel)
        record.update({k: v for k, v in it.items() if k not in record})
        record["name"] = it["name"]
        return record

    def batch(self, items: list[dict], concurrency: int | None = None) -> list[dict]:
        """One explicitly submitted batch under one published limit."""
        deadline = time.perf_counter() + LC.http_limit_s([it["prompt"] for it in items])

        async def go():
            semaphore = asyncio.Semaphore(concurrency or len(items))

            async def bounded(it):
                async with semaphore:
                    return await self._one(it, deadline)
            return await asyncio.gather(*(bounded(it) for it in items))
        return LC.run(go())

    def one(self, item: dict) -> dict:
        return self.batch([item])[0]

    async def _as_one_request(self, items: list[dict], request_id: str, deadline: float) -> list[dict]:
        """`items` as ONE /v1/completions carrying a list of token-ID prompts, so the engine admits them at one
        step boundary instead of at whichever boundaries their separate arrivals happen to fall on.

        The agent-visible client has no multi-prompt path and does not gain one: this is a verifier-side
        scheduling device, so it posts directly, as `lifecycle` does, through the client's own SSE and stream
        helpers. Every body field is `generation_body`'s, with one shared `request_id` and `cache_salt`.

        vLLM names each prompt's engine request `cmpl-<request_id>-<index>`, so each returned record carries that
        exact id. It must not carry the shared one: `same_request` accepts the `cmpl-<rid>-` prefix, so a shared
        id would make every sibling match all eight engine ids and `_consumes` would charge each of them with all
        eight siblings' events.
        """
        salt = items[0]["salt"]
        if any(it["salt"] != salt for it in items):
            raise SessionFailure("observation", "one request carries one cache_salt; these items differ",
                                 "observer_incompatible")
        max_tokens = max(int(it["max_tokens"]) for it in items)
        body = {"prompt": [list(it["prompt"]) for it in items], "max_tokens": max_tokens, "temperature": 0,
                "ignore_eos": True, "skip_special_tokens": False, "logprobs": 1, "return_token_ids": True,
                "stream": True, "cache_salt": salt, "request_id": request_id}
        if self.model:
            body["model"] = self.model
        records = []
        for i, it in enumerate(items):
            rec = {"request_id": f"cmpl-{request_id}-{i}", "shared_request_id": request_id, "choice_index": i,
                   "salt": salt, "prompt": list(it["prompt"]), "prompt_digest": LC.token_digest(it["prompt"]),
                   "max_tokens": int(it["max_tokens"]), "ids": [], "logprobs": [], "finish_reason": None,
                   "status": None, "error": None, "cancelled": False, "chunks": 0, "malformed": [],
                   "t_send_unix": time.time(), "t_first_unix": None, "t_end_unix": None}
            rec.update({k: v for k, v in it.items() if k not in rec})
            rec["name"] = it["name"]
            records.append(rec)

        def on_event(event, arrival):
            if not isinstance(event, dict):
                records[0]["malformed"].append(f"event is {type(event).__name__}")
                return
            if event.get("error") or event.get("object") == "error":
                message = "server error event: " + json.dumps(event.get("error") or event)[:300]
                for rec in records:
                    rec["error"] = message
                return
            for choice in event.get("choices") or []:
                index = choice.get("index")
                if not isinstance(index, int) or not 0 <= index < len(records):
                    records[0]["malformed"].append(f"choice index {index!r} outside 0..{len(records) - 1}")
                    continue
                rec = records[index]
                ids = choice.get("token_ids")
                values = (choice.get("logprobs") or {}).get("token_logprobs") if isinstance(
                    choice.get("logprobs"), dict) else None
                if not ids and not values:
                    if choice.get("finish_reason"):
                        rec["finish_reason"] = str(choice["finish_reason"])
                    continue
                rec["chunks"] += 1
                if not isinstance(ids, list) or not isinstance(values, list) or len(ids) != len(values):
                    rec["malformed"].append(
                        f"chunk {rec['chunks']}: {len(ids) if isinstance(ids, list) else ids!r} token ids and "
                        f"{len(values) if isinstance(values, list) else values!r} log-probabilities")
                    continue
                rec["ids"].extend(ids)
                rec["logprobs"].extend(values)
                if choice.get("finish_reason"):
                    rec["finish_reason"] = str(choice["finish_reason"])
                if ids and rec["t_first_unix"] is None:
                    rec["t_first_unix"] = time.time()

        host, port = LC.serving.split_url(self.base)
        sse = LC.serving.SSE(on_event)
        try:
            status, _, error = await LC.serving.http_stream(host, port, "/v1/completions",
                                                            json.dumps(body).encode(), sse.feed, deadline=deadline)
            for rec in records:
                rec["status"] = status
                if status != 200:
                    rec["error"] = f"HTTP {status}: {error[:300].decode('utf-8', 'replace')}"
        except asyncio.TimeoutError:
            for rec in records:
                rec["error"] = "the request did not complete within its published operation limit"
                rec["operation_limit"] = True
        except (OSError, LC.serving.HttpError, ValueError, asyncio.IncompleteReadError) as exc:
            for rec in records:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        for rec in records:
            rec["t_end_unix"] = time.time()
        return records

    def overlapped(self, first: list[dict], then: list[dict], *, wait_all: bool, cancel=(),
                   then_as_one_request: str | None = None, cancel_first_after_then: bool = False
                   ) -> tuple[list, list]:
        """`first` streamed together; `then` submitted together once the first output of any (or of every, with
        `wait_all`) `first` request has arrived. Requests named in `cancel` are cancelled after their first output.

        With `then_as_one_request`, `then` goes as ONE request under that id (`_as_one_request`), so the engine
        admits the whole group at one step boundary. With `cancel_first_after_then`, the `first` requests are
        cancelled once `then` has completed rather than after their own first output — which is what keeps a
        blocker decoding across the whole of `then` instead of letting the engine fall idle again.
        """
        deadline = time.perf_counter() + LC.http_limit_s([it["prompt"] for it in first + then])

        async def go():
            started = asyncio.Event()
            pending = {it["request_id"] for it in first}

            def arrived(rid):
                pending.discard(rid)
                if not wait_all or not pending:
                    started.set()

            tasks = [asyncio.create_task(self._one(it, deadline, on_first=(lambda r=it["request_id"]: arrived(r)),
                                                   cancel=it["name"] in cancel)) for it in first]
            everything = asyncio.gather(*tasks)
            waiter = asyncio.create_task(started.wait())
            await asyncio.wait([waiter, everything], return_when=asyncio.FIRST_COMPLETED,
                               timeout=max(0.0, deadline - time.perf_counter()))
            waiter.cancel()
            if then_as_one_request is not None:
                done = await self._as_one_request(then, then_as_one_request, deadline)
            else:
                done = list(await asyncio.gather(*(asyncio.create_task(self._one(it, deadline)) for it in then)))
            if cancel_first_after_then:
                for task in tasks:
                    task.cancel()
                opened = await asyncio.gather(*tasks, return_exceptions=True)
                kept = []
                for it, got in zip(first, opened):
                    if isinstance(got, BaseException):
                        kept.append({"request_id": it["request_id"], "name": it["name"], "salt": it["salt"],
                                     "prompt": list(it["prompt"]), "max_tokens": int(it["max_tokens"]),
                                     "ids": [], "logprobs": [], "finish_reason": None, "status": 200,
                                     "error": None, "cancelled": True, "chunks": 0, "malformed": [],
                                     **{k: v for k, v in it.items()
                                        if k not in ("request_id", "name", "salt", "prompt", "max_tokens")}})
                    else:
                        kept.append(got)
                return kept, done
            return list(await everything), done
        return LC.run(go())

    def counters(self) -> dict:
        return LC.counters(self.base)

    def preemptions(self):
        """The engine's preemption counter (informational beside the native trace), or None when not exported."""
        got = LC.serving.scrape_counters(self.base, ("vllm:num_preemptions",),
                                         time.perf_counter() + LC.LIMITS["limits_s"]["action"])
        return got["counters"].get("vllm:num_preemptions") if got.get("available") else None


def scored_problems(records: list[dict]) -> list[str]:
    out = []
    for r in records:
        for p in LC.reading_problems(r, partial=r.get("cancelled", False) and r.get("cancel_expected", False)):
            out.append(f"{r['name']}: {p}")
        if r.get("cancelled") and not r.get("cancel_expected"):
            out.append(f"{r['name']}: the stream ended early")
    return out


def operation_limit(records: list[dict]) -> list[str]:
    return [r["name"] for r in records if r.get("operation_limit")]


# ── families ─────────────────────────────────────────────────────────────────────────────────────


def family_metadata(s: Session) -> dict:
    spec = FAMILIES["metadata"][s.deployment]
    out_dir = s.dir / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    import requests  # noqa: PLC0415

    limit = LC.LIMITS["limits_s"]["component_invocation"]
    started = time.time()
    response = requests.post(s.base + "/collective_rpc", timeout=limit + 60, json={
        "method": "infb_kda_metadata", "timeout": limit,
        "kwargs": {"families": spec["families"], "out": str(out_dir)}})
    rec = {"families": spec["families"], "elapsed_s": time.time() - started, "status": response.status_code,
           "directory": str(out_dir)}
    if response.status_code != 200:
        rec["error"] = response.text[:2000]
        rec["candidate_problems"] = [f"the metadata RPC answered HTTP {response.status_code}"]
    rec["files"] = sorted(p.name for p in out_dir.glob("rank-*.json"))
    return rec


def _reuse(events: list[dict], rid: str) -> dict:
    """Whether the request's state rows (its first schedule event) were allocated earlier to another request on the
    same replica: the ownership stream's fresh allocations (global block ids)."""
    mine = [e for e in events if e.get("event") == "schedule" and same_request(rid, e.get("request_id"))]
    if not mine:
        return {"observed": False, "reused": False}
    first = mine[0]
    entries = first.get("state_indices") or []
    if shape_problem(first) is not None:
        rows = []
    elif all(_int(x) for x in entries):
        rows = sorted(set(entries))
    else:
        block = state_block(events)
        rows = sorted({slot for entry in entries
                       for slot in row_slots(entry, first["computed_before"], first["scheduled_tokens"], block)[1]})
    dp, step = first["dp_rank"], first["step"]
    owners: dict[int, list] = {}
    for e in events:
        if (e.get("event") == "schedule" and e["dp_rank"] == dp and e.get("fresh_blocks")
                and not same_request(rid, e.get("request_id")) and e["step"] < step):
            for blocks in e["fresh_blocks"]:
                for b in blocks:
                    owners.setdefault(b, []).append(e["request_id"])
    prior = {str(r): owners.get(r, [])[-1:] for r in rows}
    return {"observed": True, "request_id": first["request_id"], "dp_rank": dp, "step": step,
            "state_rows": rows, "prior_owner": prior,
            "reused": bool(rows) and all(owners.get(r) for r in rows)}


def family_d99(s: Session) -> dict:
    spec = FAMILIES["d99"]
    t = spec["turnover"]
    cap = t["max_rounds"][s.deployment]
    case = "d99"
    s.begin(case, targets=[])
    tail = Tail(s.deployment, case, 1)
    events: list[dict] = []
    rec = {"rounds": [], "requests": [], "case": case, "cap": cap}

    def absorb():
        s.flush()
        events.extend(tail.new(tp=0))

    def turnover_round(k):
        items = [s.item(f"d99-turnover-{k}-{i}", s.pool.span(f"d99/turnover/{s.name}/{k}/{i}", t["input_tokens"]),
                        t["output_tokens"], role="turnover", scored=False) for i in range(t["round_requests"])]
        started = time.time()
        got = s.batch(items, concurrency=t["concurrency"])
        problems = scored_problems(got)
        rec["rounds"].append({"round": k, "seconds": time.time() - started, "problems": problems,
                              "operation_limit": operation_limit(got),
                              "requests": [{"name": g["name"], "n": len(g["ids"]), "finish": g["finish_reason"]}
                                           for g in got]})
        if problems:
            raise SessionFailure("candidate", f"turnover round {k}: {problems[:4]}")

    def replicas_reused():
        out = {}
        for dp in sorted({d for d, _ in EXPECTED_RANKS[s.deployment]}):
            owners: dict = {}
            for e in events:
                if e.get("event") == "schedule" and e["dp_rank"] == dp and e.get("fresh_blocks"):
                    for blocks in e["fresh_blocks"]:
                        for b in blocks:
                            owners.setdefault(b, set()).add(e["request_id"])
            out[dp] = sum(1 for v in owners.values() if len(v) > 1)
        return out

    rounds = 0
    while rounds < cap:
        turnover_round(rounds)
        rounds += 1
        absorb()
        reused = replicas_reused()
        if all(v > 0 for v in reused.values()):
            break
    rec["turnover"] = {"rounds": rounds, "reused_blocks_by_replica": replicas_reused()}
    queue = [(r, tok) for r in range(spec["repeats"]) for tok in PROBES]
    attempts: dict = {}
    while queue:
        r, tok = queue.pop(0)
        attempts[(r, tok)] = n = attempts.get((r, tok), 0) + 1
        item = s.item(f"d99-probe-{r}-{tok}-a{n}", [tok], spec["max_tokens"], role="probe", scored=True,
                      slot=[r, tok], attempt=n)
        got = s.one(item)
        absorb()
        got["reuse"] = _reuse(events, item["request_id"])
        rec["requests"].append(got)
        if not got["reuse"]["reused"] and n < WAVES:
            if rounds < cap:
                turnover_round(rounds)
                rounds += 1
                absorb()
            queue.insert(0, (r, tok))
    rec["turnover"]["rounds_total"] = rounds
    s.end()
    rec["slots"] = {f"{r}-{tok}": any(g["reuse"]["reused"] for g in rec["requests"] if g["slot"] == [r, tok])
                    for r in range(spec["repeats"]) for tok in PROBES}
    return rec


def family_continued(s: Session) -> dict:
    spec = FAMILIES["continued"]
    items = []
    for n in spec["lengths"]:
        for i in range(spec["per_length"]):
            items.append(s.item(f"continued-{n}-{i}", s.pool.span(f"continued/{s.name}/{n}/{i}", n),
                                spec["max_tokens"], role="continued", scored=True))
    s.begin("continued", targets=[])
    got = s.batch(items)
    s.end()
    return {"case": "continued", "requests": got}


def _overlap(events: list[dict], probe_ids: list[str]) -> list[dict]:
    """Steps that schedule a fresh one-token probe beside running requests."""
    steps: dict = {}
    for e in events:
        if e.get("event") == "schedule":
            steps.setdefault((e["dp_rank"], e["step"]), []).append(e)
    out = []
    for (dp, step), rows in steps.items():
        fresh = [e for e in rows if e["kind"] == "new" and e["computed_before"] == 0
                 and any(same_request(p, e["request_id"]) for p in probe_ids)]
        running = [e for e in rows if e["kind"] == "running" and e["computed_before"] > 0]
        if fresh and running:
            out.append({"dp_rank": dp, "step": step, "fresh": [e["request_id"] for e in fresh],
                        "running": len(running), "graph_mode": rows[0].get("graph_mode")})
    return out


def family_fresh_during_decode(s: Session) -> dict:
    spec = FAMILIES["fresh_during_decode"]
    rec = {"waves": [], "requests": []}
    for wave in range(1, WAVES + 1):
        case = "fresh_during_decode"
        decode = [s.item(f"fdd-w{wave}-decode-{i}", s.pool.span(f"fdd/{s.name}/{wave}/{i}", spec["prompt_tokens"]),
                         spec["max_tokens"], role="decode", scored=True, wave=wave) for i in range(spec["decode_requests"])]
        probes = [s.item(f"fdd-w{wave}-probe-{tok}", [tok], spec["probe_max_tokens"], role="probe", scored=True,
                         wave=wave) for tok in PROBES]
        started = time.time()
        s.begin(case, wave=wave, targets=[])
        a, b = s.overlapped(decode, probes, wait_all=True)
        s.end()
        files = load_case(s.deployment, case, wave)
        tp0 = [e for name, ev in files.items() if "-tp0-" in name for e in ev]
        overlap = _overlap(tp0, [p["request_id"] for p in probes])
        rec["requests"] += a + b
        rec["waves"].append({"wave": wave, "seconds": time.time() - started, "overlap_steps": overlap[:16],
                             "overlap_count": len(overlap)})
        if overlap:
            break
    return rec


def family_dp_idle(s: Session) -> dict:
    spec = FAMILIES["dp_idle"]
    s.begin("dp_idle", targets=[])
    got = [s.one(s.item(f"dp-idle-{i}", s.pool.span(f"dp_idle/{i}", n), spec["max_tokens"], role="idle",
                        scored=True)) for i, n in enumerate(spec["lengths"])]
    s.end()
    return {"case": "dp_idle", "requests": got}


def family_dp_load(s: Session) -> dict:
    spec = FAMILIES["dp_load"]
    cycle, k = spec["lengths"], 0
    rec = {"bursts": [], "requests": []}
    s.begin("dp_load", targets=[])
    for b, size in enumerate(spec["bursts"]):
        items = []
        for i in range(size):
            items.append(s.item(f"dp-load-b{b}-{i}", s.pool.span(f"dp_load/b{b}/{i}", cycle[k % len(cycle)]),
                                spec["max_tokens"], role="burst", scored=True))
            k += 1
        got = s.batch(items)
        rec["bursts"].append({"burst": b, "size": size, "lengths": [len(it["prompt"]) for it in items]})
        rec["requests"] += got
    first_n, then_n = spec["waves"]
    first = [s.item(f"dp-load-wa-{i}", s.pool.span(f"dp_load/wa/{i}", cycle[i % len(cycle)]), spec["max_tokens"],
                    role="wave", scored=True) for i in range(first_n)]
    then = [s.item(f"dp-load-wb-{i}", s.pool.span(f"dp_load/wb/{i}", cycle[i % len(cycle)]), spec["max_tokens"],
                   role="wave", scored=True) for i in range(then_n)]
    a, b2 = s.overlapped(first, then, wait_all=False)
    rec["requests"] += a + b2
    s.end()
    rec["case"] = "dp_load"
    return rec


def family_dp_quality(s: Session) -> dict:
    acceptance = _load("infb_lifecycle_acceptance", EVAL / "tests/acceptance.py")
    fidelity = _load("infb_lifecycle_fidelity", EVAL / "fidelity/check.py")
    rec: dict = {}
    port = int(s.base.rsplit(":", 1)[1])
    try:
        rec["gsm8k"] = acceptance.accuracy(
            port, dataset_dir=Path(os.environ.get("INFB_GSM8K_DATASET_DIR", X["dataset_root"])),
            evaluator=EVAL / "tests/upstream/gsm8k_eval.py", num_questions=X["gsm8k"]["num_questions"],
            num_shots=X["gsm8k"]["num_shots"], max_tokens=X["gsm8k"]["max_tokens"], seed=X["gsm8k"]["seed"])
    except Exception as exc:  # noqa: BLE001 - recorded for the verdict; the session goes on
        rec["gsm8k_error"] = f"{type(exc).__name__}: {exc}"[:2000]
    sequences = json.loads((TESTS / "hidden/fidelity-sequences.json").read_text())["hidden"]
    reference = {r["id"]: r for r in json.loads((TESTS / "hidden/fidelity-reference.json").read_text())["hidden"]}
    by_band: dict = {}
    for seq in sequences:
        by_band.setdefault(seq["band"], []).append(seq)
    url = s.base + "/v1/completions"
    try:
        rec["fidelity"] = fidelity.graded_bands(
            lambda seq, tag: fidelity.read_author(url, seq, reference[seq["id"]], tag, prefix="infb-lifecycle-dp2"),
            by_band, X["fidelity"]["thresholds"])
    except Exception as exc:  # noqa: BLE001
        rec["fidelity_error"] = f"{type(exc).__name__}: {exc}"[:2000]
        rec["fidelity_detail"] = getattr(exc, "bands", None)
    prompts = json.loads((TESTS / "hidden/concurrency-prompts.json").read_text())["prompts"]
    cfg = X["concurrency"]
    try:
        rec["concurrency"] = acceptance.concurrency(
            s.server.url_for, prompts, list(cfg["T_per_band"]), model=X["served_model_name"],
            max_tokens=cfg["max_tokens"], top_k=cfg["top_k"], bars=cfg["T_per_band"])
    except Exception as exc:  # noqa: BLE001
        rec["concurrency_error"] = f"{type(exc).__name__}: {exc}"[:2000]
    return rec


def _hit(events: list[dict], rid: str) -> dict | None:
    """The request's first consume (restore) event on this rank, or None."""
    return next((e for e in events if e.get("event") == "consume" and same_request(rid, e.get("request_id"))), None)


def _producers(events: list[dict]) -> dict:
    """(prefix length, prefix digest) -> the latest produce event."""
    out = {}
    for e in events:
        if e.get("event") == "produce":
            out[(e["prefix_length"], e["prefix_digest"])] = e
    return out


def _warm_pair(s: Session, tag: str, prefix, salt: str, n: int, suffix_keys=("s0", "s1")):
    suffixes, used = [], set()
    for k in suffix_keys:
        sfx = s.pool.suffix(f"{tag}/{k}", 1, avoid_first=used)
        used.add(sfx[0])
        suffixes.append(sfx)
    warm = s.one(s.item(f"{tag}-warm", prefix + suffixes[0], n, salt, role="warm", scored=True))
    before = s.counters()
    hit = s.one(s.item(f"{tag}-hit", prefix + suffixes[1], n, salt, role="hit", scored=True, expect_hit=len(prefix)))
    hit["hit_tokens_counter"] = LC.counter_delta(before, s.counters())
    return warm, hit


def family_a1(s: Session) -> dict:
    spec = FAMILIES["a1"]
    rec = {"case": "a1", "requests": []}
    s.begin("a1")
    for i, n in enumerate(spec["lengths"]):
        rec["requests"].append(s.one(s.item(f"a1-cold-{n}", s.pool.span(f"a1/cold/{i}", n), spec["max_tokens"],
                                            role="cold", scored=True, expect_hit=0)))
    salt = f"{s.name}-a1"
    for p in spec["hit_prefixes"]:
        prefix = s.pool.span(f"a1/prefix/{p}", p)
        rec["requests"] += _warm_pair(s, f"a1-{p}", prefix, salt, spec["max_tokens"])
    s.end()
    return rec


def family_a6(s: Session) -> dict:
    spec = FAMILIES["a6"]
    salt = f"{s.name}-a6"
    rec = {"case": "a6", "requests": []}
    s.begin("a6")
    prefixes, used = {}, {}
    for p in spec["warm_prefixes"]:
        prefixes[p] = s.pool.span(f"a6/prefix/{p}", p)
        sfx = s.pool.suffix(f"a6/warm/{p}", 1)
        used[p] = {sfx[0]}
        rec["requests"].append(s.one(s.item(f"a6-warm-{p}", prefixes[p] + sfx, spec["max_tokens"], salt,
                                            role="warm", scored=True)))
    items = []
    for i in range(spec["hits"]):
        p = spec["warm_prefixes"][i % len(spec["warm_prefixes"])]
        sfx = s.pool.suffix(f"a6/hit/{i}", 1, avoid_first=used[p])
        used[p].add(sfx[0])
        items.append(s.item(f"a6-hit-{i}", prefixes[p] + sfx, spec["max_tokens"], salt, role="hit", scored=True,
                            expect_hit=p))
    for i in range(spec["fresh"]):
        n = spec["fresh_lengths"][i % len(spec["fresh_lengths"])]
        items.append(s.item(f"a6-fresh-{i}", s.pool.span(f"a6/fresh/{i}", n), spec["max_tokens"], role="fresh",
                            scored=True, expect_hit=0))
    rec["requests"] += s.batch(items)
    s.end()
    return rec


def family_a2(s: Session) -> dict:
    spec = FAMILIES["a2"]
    salt = f"{s.name}-a2"
    prefix = s.pool.span("a2/prefix", spec["prefix"])
    rec = {"case": "a2", "requests": [], "order": spec["order"]}
    s.begin("a2")
    rec["requests"].append(s.one(s.item("a2-warm", prefix + s.pool.suffix("a2/warm", 1), spec["warm_max_tokens"],
                                        salt, role="warm", scored=True)))
    firsts = {rec["requests"][0]["prompt"][len(prefix)]}
    suffixes = []
    for i, n in enumerate(spec["suffixes"]):
        sfx = s.pool.suffix(f"a2/branch/{i}", n, avoid_first=firsts)
        firsts.add(sfx[0])
        suffixes.append(sfx)
    for k, b in enumerate(spec["order"]):
        rec["requests"].append(s.one(s.item(f"a2-branch-{b}-x{k}", prefix + suffixes[b], spec["max_tokens"], salt,
                                            role="branch", scored=True, branch=b, execution=k,
                                            expect_hit=spec["prefix"])))
    s.end()
    return rec


def family_a3(s: Session) -> dict:
    """The eight cold siblings share one 2,048-token prefix, so exactly one of them produces its blocks and the
    rest should hit them. The bug C5 models — a sibling restoring a block its producer has not finished in the
    same step — can only be read when a sibling is scheduled in the step that is still computing those blocks.

    Sending the eight as eight separate requests against an idle engine leaves it to arrival jitter whether
    any sibling lands in the producer's step; when none does, the candidate's readings match the
    reference's and the node has nothing to judge.

    This design makes the co-scheduling certain without touching the recipe. A blocker prompt is streamed first
    and the eight go only once its first token has arrived, so the engine is in steady decode with a step in
    flight; and they go as ONE request carrying eight prompts, so they are admitted at one step boundary
    together rather than at whichever boundaries eight separate arrivals fall on. Prompts, sizes, salt and the
    follow-up rule are unchanged.
    """
    spec = FAMILIES["a3"]
    salt = f"{s.name}-a3"
    prefix = s.pool.span("a3/prefix", spec["prefix"])
    firsts: set = set()
    siblings = []
    for i in range(spec["siblings"] + 1):
        sfx = s.pool.suffix(f"a3/sibling/{i}", spec["suffix"], avoid_first=firsts)
        firsts.add(sfx[0])
        siblings.append(prefix + sfx)
    blocker_prompt = s.pool.suffix("a3/blocker", spec["blocker"], avoid_first={prefix[0]})
    s.begin("a3")
    blocker = s.item("a3-blocker", blocker_prompt, spec["blocker_max_tokens"], f"{salt}-blocker",
                     role="blocker", scored=False, cancel_expected=True)
    held, cold = s.overlapped(
        [blocker],
        [s.item(f"a3-cold-{i}", siblings[i], spec["max_tokens"], salt, role="cold-sibling", scored=True)
         for i in range(spec["siblings"])],
        wait_all=False, then_as_one_request=f"{s.name}-a3-cold", cancel_first_after_then=True)
    follow = s.one(s.item("a3-follow", siblings[-1], spec["max_tokens"], salt, role="follow-up", scored=True,
                          expect_hit=spec["prefix"]))
    s.end()
    return {"case": "a3", "requests": cold + [follow], "unscored": held}


def family_a4(s: Session) -> dict:
    spec = FAMILIES["a4"]
    salt = f"{s.name}-a4"
    rec = {"case": "a4", "requests": []}
    s.begin("a4")
    for n in spec["prompts"]:
        prompt = s.pool.span(f"a4/prompt/{n}", n)
        rec["requests"].append(s.one(s.item(f"a4-long-{n}", prompt, spec["max_tokens"], salt, role="long",
                                            scored=True)))
        for k in spec["siblings"][str(n)]:
            sfx = s.pool.suffix(f"a4/sibling/{n}/{k}", spec["sibling_suffix"], avoid_first={prompt[k]})
            rec["requests"].append(s.one(s.item(f"a4-sibling-{n}-{k}", prompt[:k] + sfx, spec["sibling_max_tokens"],
                                                salt, role="sibling", scored=True, shared_prefix=k)))
    s.end()
    return rec


def family_a5(s: Session) -> dict:
    spec = FAMILIES["a5"]
    salt = f"{s.name}-a5"
    prompt = s.pool.span("a5/prompt", spec["prompt"])
    s.begin("a5")
    base = s.one(s.item("a5-base", prompt, spec["max_tokens"], salt, role="history", scored=True))
    out = [base]
    history = prompt + base["ids"]
    for k in spec["history_prefixes"]:
        if len(history) < k:
            break
        sfx = s.pool.suffix(f"a5/suffix/{k}", spec["suffix"], avoid_first={history[k]} if len(history) > k else ())
        out.append(s.one(s.item(f"a5-follow-{k}", history[:k] + sfx, spec["follow_max_tokens"], salt,
                                role="history-branch", scored=True, shared_prefix=k)))
    s.end()
    return {"case": "a5", "requests": out}


def family_binding(s: Session) -> dict:
    spec = FAMILIES["binding"]
    prefix = s.pool.span("binding/prefix", spec["prefix"])
    sfx = [s.pool.suffix("binding/s0", 1)]
    sfx.append(s.pool.suffix("binding/s1", 1, avoid_first={sfx[0][0]}))
    sfx.append(s.pool.suffix("binding/s2", 1, avoid_first={sfx[0][0], sfx[1][0]}))
    rec = {"passes": []}
    for layer in spec["layers"]:
        for views in spec["views"]:
            tag = f"binding-l{layer}-{'-'.join(views)}"
            control_case, target_case = f"{tag}-control", f"{tag}-target"
            s.begin(control_case)
            warm_c = s.one(s.item(f"{tag}-c-warm", prefix + sfx[0], spec["max_tokens"], f"{s.name}-{tag}-c",
                                  role="warm", scored=True))
            control = s.one(s.item(f"{tag}-c-control", prefix + sfx[1], spec["max_tokens"], f"{s.name}-{tag}-c",
                                   role="control", scored=True, expect_hit=spec["prefix"]))
            s.end()
            target = s.item(f"{tag}-t-perturbed", prefix + sfx[1], spec["max_tokens"], f"{s.name}-{tag}-t",
                            role="perturbed", scored=False, expect_hit=spec["prefix"])
            s.begin(target_case, intervene={"request": f"cmpl-{target['request_id']}-", "views": views,
                                            "layers": [layer], "add": spec["add"]})
            warm_t = s.one(s.item(f"{tag}-t-warm", prefix + sfx[0], spec["max_tokens"], f"{s.name}-{tag}-t",
                                  role="warm", scored=True))
            target = s.one(target)
            sibling = s.one(s.item(f"{tag}-t-sibling", prefix + sfx[2], spec["max_tokens"], f"{s.name}-{tag}-t",
                                   role="sibling", scored=True, expect_hit=spec["prefix"]))
            s.end()
            rec["passes"].append({"layer": layer, "views": views, "control_case": control_case,
                                  "target_case": target_case, "requests": [warm_c, control, warm_t, target, sibling]})
    rec["requests"] = [r for p in rec["passes"] for r in p["requests"]]
    return rec


def _evicted(events: list[dict], producer: dict) -> dict | None:
    """The first fresh allocation, after the producer step, of a block of the producer's prefix (global block ids):
    the prefix is no longer retained."""
    needed = {b for ids in (producer.get("prefix_blocks") or []) for b in ids}
    for e in events:
        if (e.get("event") == "schedule" and e.get("fresh_blocks") and e["step"] > producer["step"]
                and needed & {b for ids in e["fresh_blocks"] for b in ids}):
            return {"request_id": e["request_id"], "step": e["step"]}
    return None


def family_p1(s: Session) -> dict:
    spec = FAMILIES["p1"]
    salt = f"{s.name}-p1"
    prefix = s.pool.span("p1/prefix", spec["warm_prefix"])
    sfx = [s.pool.suffix("p1/s0", 1)]
    sfx.append(s.pool.suffix("p1/s1", 1, avoid_first={sfx[0][0]}))
    sfx.append(s.pool.suffix("p1/s2", 1, avoid_first={sfx[0][0], sfx[1][0]}))
    rec = {"case": "p1", "requests": [], "fills": []}
    s.begin("p1")
    rec["requests"].append(s.one(s.item("p1-warm", prefix + sfx[0], spec["warm_max_tokens"], salt, role="warm",
                                        scored=True)))
    rec["requests"].append(s.one(s.item("p1-hit", prefix + sfx[1], spec["warm_max_tokens"], salt, role="hit",
                                        scored=True, expect_hit=spec["warm_prefix"])))
    tail = Tail(s.deployment, "p1", 1)
    events: list[dict] = []
    warm_id = rec["requests"][0]["request_id"]
    for wave in range(1, WAVES + 1):
        started = time.time()
        fills = [s.item(f"p1-fill-w{wave}-{i}", s.pool.span(f"p1/fill/{wave}/{i}", spec["fill_prompt_tokens"]),
                        spec["fill_max_tokens"], role="fill", scored=True, expect_hit=0)
                 for i in range(spec["fill_count"])]
        rec["requests"] += s.batch(fills)
        s.flush()
        events += tail.new(tp=0)
        producer = next((e for e in events if e.get("event") == "produce" and same_request(warm_id, e["request_id"])
                         and e["prefix_length"] == spec["warm_prefix"]), None)
        evicted = _evicted(events, producer) if producer else None
        rec["fills"].append({"wave": wave, "seconds": time.time() - started, "producer_seen": producer is not None,
                             "evicted_by": evicted})
        if evicted:
            break
    rec["requests"].append(s.one(s.item("p1-original", prefix + sfx[0], spec["warm_max_tokens"], salt,
                                        role="original-branch", scored=True)))
    rec["requests"].append(s.one(s.item("p1-rewarm-hit", prefix + sfx[2], spec["warm_max_tokens"], salt,
                                        role="hit", scored=True, expect_hit=spec["warm_prefix"])))
    s.end()
    return rec


def family_p2(s: Session) -> dict:
    """The two pressure cases: a resumed request's recompute, and a branch from its retained boundary (`_select_resumed` picks it)."""
    spec = FAMILIES["p2"]
    rec = {"waves": [], "requests": [], "selection_rule": FAMILIES["p2"]["rule"]}
    for wave in range(1, WAVES + 1):
        salt = f"{s.name}-p2-w{wave}"
        case = "p2"
        started = time.time()
        s.begin(case, wave=wave)
        before = s.preemptions()
        items = [s.item(f"p2-w{wave}-{i:02d}", s.pool.span(f"p2/{wave}/{i}", spec["prompt_tokens"]),
                        spec["max_tokens"], salt, role="pressure", scored=True, wave=wave) for i in range(spec["requests"])]
        got = s.batch(items)
        rec["requests"] += got
        s.flush()
        files = load_case(s.deployment, case, wave)
        events = [e for name, ev in files.items() if "-tp0-" in name for e in ev]
        resumed = [e for e in events if e.get("event") == "schedule" and e["kind"] == "resumed"]
        qualifying = [e for e in resumed if e["output_count"] > 0 and e.get("history") is not None
                      and e["history"] % B != 0]
        w = {"wave": wave, "resumed_events": len(resumed), "qualifying_events": len(qualifying),
             "preemptions": None if before is None or (after := s.preemptions()) is None else after - before}
        rec["waves"].append(w)
        selected = _select_resumed(events, qualifying, got)
        w["selection"] = selected
        if selected and selected.get("boundary"):
            response = next(g for g in got if same_request(g["request_id"], selected["request_id"]))
            history = response["prompt"] + response["ids"]
            L = selected["boundary"]
            sfx = s.pool.suffix(f"p2/branch/{wave}", spec["branch_suffix"],
                                avoid_first={history[L]} if len(history) > L else ())
            branch = s.one(s.item(f"p2-w{wave}-branch", history[:L] + sfx, spec["branch_max_tokens"], salt,
                                  role="history-branch", scored=True, wave=wave, expect_hit=L,
                                  includes_generated=L > len(response["prompt"])))
            rec["requests"].append(branch)
            w["branch"] = branch["request_id"]
            s.end()
            w["seconds"] = time.time() - started
            break
        s.end()
        w["seconds"] = time.time() - started
    return rec


def _select_resumed(events: list[dict], qualifying: list[dict], responses: list[dict]) -> dict | None:
    """Among qualifying resumed requests with a completed boundary retained at wave end, the earliest by (step,
    batch row); its largest retained completed boundary strictly below its final computed count. Reads the
    allocator trace only, never the restore comparison."""
    order = sorted(qualifying, key=lambda e: (e["step"], e["row"] if e.get("row") is not None else 1 << 30))
    seen: dict = {}
    for e in order:
        rid = e["request_id"]
        if rid in seen:
            continue
        mine = [x for x in events if x.get("request_id") == rid]
        final = max(x["computed_before"] + x["scheduled_tokens"] for x in mine if x.get("event") == "schedule")
        latest = {}
        for x in mine:
            if x.get("event") == "produce" and x["prefix_length"] < final:
                latest[x["prefix_length"]] = x
        retained = [k for k, x in sorted(latest.items()) if _evicted(events, x) is None]
        seen[rid] = {"first_qualifying_resume": {"step": e["step"], "row": e.get("row")}, "final_computed": final,
                     "retained": retained, "history": e["history"], "output_count": e["output_count"]}
    eligible = [rid for rid, v in seen.items() if v["retained"]]
    if not seen:
        return None
    out = {"qualifying_requests": seen, "unfiltered_pick": next(iter(seen))}
    if eligible:
        rid = eligible[0]
        out.update(request_id=rid, boundary=seen[rid]["retained"][-1])
    return out


def family_p4(s: Session) -> dict:
    spec = FAMILIES["p4"]
    salt = f"{s.name}-p4"
    names = [f"p4-{i:02d}" for i in range(spec["requests"])]
    cancel = {names[i] for i in spec["cancel"]}
    items = [s.item(names[i], s.pool.span(f"p4/{i}", spec["prompt_tokens"]), spec["max_tokens"], salt, role="wave",
                    scored=True, cancel_expected=names[i] in cancel) for i in range(spec["requests"])]
    s.begin("p4")
    wave, _ = s.overlapped(items, [], wait_all=False, cancel=cancel)
    s.flush()
    files = load_case(s.deployment, "p4", 1)
    events = [e for name, ev in files.items() if "-tp0-" in name for e in ev]
    unaffected = [g for g in wave if g["name"] not in cancel]
    finished = sorted(unaffected, key=lambda g: (g.get("t_end_unix") or 0))
    chosen = None
    for g in finished:
        producer = next((e for e in events if e.get("event") == "produce"
                         and same_request(g["request_id"], e["request_id"])
                         and e["prefix_length"] == spec["branch_prefix"]), None)
        if producer is not None and _evicted(events, producer) is None:
            chosen = g
            break
    fresh = [s.item(f"p4-fresh-{tok}", [tok], spec["fresh_max_tokens"], role="fresh", scored=True)
             for tok in PROBES[:4]]
    rec = {"case": "p4", "requests": list(wave), "cancelled": sorted(cancel),
           "branch_source": chosen["request_id"] if chosen else None}
    rec["requests"] += s.batch(fresh)
    if chosen is not None:
        firsts = {chosen["prompt"][spec["branch_prefix"]]}
        for j in range(spec["branches"]):
            sfx = s.pool.suffix(f"p4/branch/{j}", spec["branch_suffix"], avoid_first=firsts)
            firsts.add(sfx[0])
            rec["requests"].append(s.one(s.item(f"p4-branch-{j}", chosen["prompt"][:spec["branch_prefix"]] + sfx,
                                                spec["branch_max_tokens"], salt, role="branch", scored=True,
                                                expect_hit=spec["branch_prefix"])))
    s.end()
    return rec


RUNNERS = {"metadata": family_metadata, "d99": family_d99, "continued": family_continued,
           "fresh_during_decode": family_fresh_during_decode, "dp_idle": family_dp_idle, "dp_load": family_dp_load,
           "dp_quality": family_dp_quality, "a1": family_a1, "a6": family_a6, "a2": family_a2, "a3": family_a3,
           "a4": family_a4, "a5": family_a5, "binding": family_binding, "p1": family_p1, "p2": family_p2,
           "p4": family_p4}


# ── one candidate session ────────────────────────────────────────────────────────────────────────


def run_session(name: str) -> int:
    from infb_verify import serve  # noqa: PLC0415 - the candidate session's launch route

    deployment = SESSIONS[name]["deployment"]
    directory = ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    record = {"session": name, "deployment": deployment, "state": "running", "started_unix": time.time(),
              "manifest_sha256": hashlib.sha256((TESTS / "hidden/state-lifecycle.json").read_bytes()).hexdigest(),
              "families": {}, "failures": []}
    path = directory / "session.json"
    write_json(path, record)
    try:
        pool = Pool()
    except Exception as exc:  # noqa: BLE001 - the hidden pool is the verifier's material
        record["failures"].append({"kind": "verifier", "family": None,
                                   "message": f"the hidden pool could not be read: {type(exc).__name__}: {exc}"[:2000]})
        record.update(state="failed", ended_unix=time.time())
        write_json(path, record)
        return 0
    record["pool_tokens"], record["tokenizer_sha256"] = len(pool.tokens), pool.tokenizer_sha256
    launcher = serve.launch(deployment=deployment, lifecycle=str(OBSERVER))
    started = time.time()
    try:
        server = next(launcher)
    except Exception as exc:  # noqa: BLE001 - the candidate's deployment did not start
        record["launch"] = {"healthy": False, "seconds": time.time() - started,
                            "error": f"{type(exc).__name__}: {exc}"[:4000]}
        record["failures"].append({"kind": "candidate", "family": None, "message": f"the {deployment} deployment did "
                                                                                     f"not start: {exc}"[:2000]})
        record.update(state="failed", ended_unix=time.time())
        write_json(path, record)
        return 0
    record["launch"] = {"healthy": True, "seconds": time.time() - started, "url": server.url_root,
                        "log": str(serve.SERVER_LOG.with_name(f"{deployment}-lifecycle-server.log"))}
    write_json(path, record)
    session = Session(name, server, pool)
    try:
        for family in SESSIONS[name]["families"]:
            entry = record["families"][family] = {"state": "running", "started_unix": time.time()}
            write_json(path, record)
            log(f"{name}: {family}")
            try:
                result = RUNNERS[family](session)
                write_json(directory / f"{family}.json", {"family": family, "session": name,
                                                          "deployment": deployment, **result})
                entry["state"] = "complete"
            except SessionFailure as exc:
                entry.update(state="failed", kind=exc.kind, reason=exc.reason, error=str(exc)[:4000])
                record["failures"].append({"kind": exc.kind, "reason": exc.reason, "family": family,
                                           "message": str(exc)[:2000], "evidence": exc.evidence})
            except Exception as exc:  # noqa: BLE001 - a request/RPC failure at the candidate's surface
                entry.update(state="failed", kind="candidate", error=f"{type(exc).__name__}: {exc}"[:4000],
                             traceback=traceback.format_exc()[-4000:])
                record["failures"].append({"kind": "candidate", "family": family,
                                           "message": f"{type(exc).__name__}: {exc}"[:2000]})
            entry["ended_unix"] = time.time()
            if session.case is not None:
                try:
                    session.end()
                except Exception:  # noqa: BLE001
                    pass
            write_json(path, record)
        record["self_consistency"] = self_teacher_force(session)
        write_json(path, record)
        acceptance = _load("infb_lifecycle_acceptance_recipe", EVAL / "tests/acceptance.py")
        launch = _load("infb_lifecycle_launch", EVAL / "launch.py")
        log_path = Path(record["launch"]["log"])
        problems, summary = acceptance.recipe_problems(log_path.read_text(errors="replace") if log_path.is_file()
                                                       else "", max_model_len=launch.max_model_len(deployment),
                                                       deployment=deployment)
        record["recipe"] = {"problems": problems, "summary": summary}
    finally:
        teardown_started = time.time()
        try:
            launcher.close()
            record["teardown"] = {"seconds": time.time() - teardown_started, "ok": True}
        except Exception as exc:  # noqa: BLE001
            record["teardown"] = {"seconds": time.time() - teardown_started, "ok": False,
                                  "error": f"{type(exc).__name__}: {exc}"[:2000]}
        record.update(state="complete", ended_unix=time.time())
        write_json(path, record)
    return 0


# ── the late cold reference ──────────────────────────────────────────────────────────────────────


def scored_targets() -> list[dict]:
    """Every scored target of every candidate session's family records, with a reading to compare."""
    out = []
    for name in SESSIONS:
        for path in sorted((ROOT / name).glob("*.json")):
            if path.name == "session.json":
                continue
            data = read_json(path)
            for r in data.get("requests", []) or []:
                if r.get("scored") and r.get("ids"):
                    out.append({"key": f"{name}/{data['family']}/{r['name']}", "session": name,
                                "family": data["family"], "request_id": r["request_id"], "prompt": r["prompt"],
                                "ids": r["ids"]})
    return out


def self_teacher_force(s: Session) -> dict:
    """Supplemental: the session's scored targets teacher-forced on the candidate's own server,
    one request at a time, `prompt_logprobs` 0 (no prefix reuse), so each warm reading stands beside the same
    deployment's cold computation of the same tokens. Recorded, never judged and never a substitute for the
    independent reference (the same incorrect computation could agree with itself); a failure here fails nothing."""
    path = s.dir / "self-teacher-forced.json"
    targets = [t for t in scored_targets() if t["session"] == s.name]
    record: dict = {"state": "running", "started_unix": time.time(), "n_targets": len(targets), "targets": {}}
    try:
        for t in targets:
            started = time.time()
            values = LC.teacher_force(s.base, t["prompt"], t["ids"], f"{s.name}-self-{t['key']}")
            record["targets"][t["key"]] = {"family": t["family"], "request_id": t["request_id"],
                                           "prompt_digest": LC.token_digest(t["prompt"]),
                                           "ids_digest": LC.token_digest(t["ids"]), "self": values,
                                           "seconds": time.time() - started}
        record["state"] = "complete"
    except Exception as exc:  # noqa: BLE001 - supplemental evidence; recorded, never a failure
        record.update(state="failed", error=f"{type(exc).__name__}: {exc}"[:2000])
    record["ended_unix"] = time.time()
    write_json(path, record)
    return {"state": record["state"], "n_targets": len(targets), "read": len(record["targets"]),
            "seconds": record["ended_unix"] - record["started_unix"]}


def teacher_force_all(base: str, record: dict, path: Path, concurrency: int = 1) -> None:
    """Every scored target's prompt plus exact continuation, teacher-forced on the server at `base` (`prompt_logprobs`
    0, its own salt), one request at a time as the public `compare` does: a reference value is then a function of
    its sequence alone, not of what else the cold server happened to batch with it. The values go into
    `record["targets"]`."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    targets = scored_targets()
    record["n_targets"], record["concurrency"] = len(targets), concurrency
    write_json(path, record)

    def read(t):
        started = time.time()
        values = LC.teacher_force(base, t["prompt"], t["ids"], f"lifecycle-reference-{t['key']}")
        return t["key"], {"session": t["session"], "family": t["family"], "request_id": t["request_id"],
                          "prompt_digest": LC.token_digest(t["prompt"]), "ids_digest": LC.token_digest(t["ids"]),
                          "reference": values, "seconds": time.time() - started}

    started = time.time()
    with ThreadPoolExecutor(concurrency) as pool:
        for key, value in pool.map(read, targets):
            record["targets"][key] = value
    record["teacher_forcing_s"] = time.time() - started


def run_reference() -> int:
    """The independent reference, re-identified from its pristine snapshot, served once at the
    cache-disabled tp4 recipe, and every scored continuation teacher-forced on it."""
    sys.path.insert(0, str(TESTS / "infb_efficiency"))
    import reference as kit  # noqa: PLC0415 - the efficiency kit's reference route, reused

    directory = ROOT / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "teacher-forced.json"
    record: dict = {"state": "running", "started_unix": time.time(), "targets": {}}
    write_json(path, record)
    eff = X["efficiency"]
    ref = eff["reference"]
    proc = None
    try:
        record["isolation_dropped_paths"] = kit.check_isolated(Path(ref["app"]))
        record["snapshot"] = {k: v for k, v in kit.check_snapshot(ref).items() if k != "products"}
        recipe = eff["recipes"]["test_hidden"]
        py = eff["python"]
        cap = kit.capture_argv(ref, recipe["launch"], py)
        port = int(ref.get("port", 31000)) + 7
        argv = kit.with_port(cap["argv"], port)
        tree = Path(ref["root"]) / "tree"
        env = kit.server_env(tree, ref)
        env.update(ref.get("launch_env", {}))
        env.update(recipe.get("launch_env", {}))
        record["launch"] = {"argv": argv, "engine_file": cap["engine_file"],
                            "jit_cache": kit.private_jit_caches(env, Path(ref["root"]) / "jit-cache")}
        base = f"http://127.0.0.1:{port}"
        proc, info = kit.launch_reference(argv, env, Path(ref["root"]), base + ref.get("health_path", "/health"),
                                          float(eff["deadlines"]["launch_s"]), directory / "reference-server.log")
        record["launch"].update(info)
        if proc is None:
            raise SessionFailure("reference", f"the reference did not become healthy: {info['attempts']}")
        teacher_force_all(base, record, path)
        record["state"] = "complete"
    except SessionFailure as exc:
        record.update(state="failed", error=str(exc)[:4000])
    except Exception as exc:  # noqa: BLE001 - any failure here is the reference route's
        record.update(state="failed", error=f"{type(exc).__name__}: {exc}"[:4000],
                      traceback=traceback.format_exc()[-4000:])
    finally:
        record["teardown"] = kit.teardown(proc, eff, eff["deadlines"])
        record["ended_unix"] = time.time()
        write_json(path, record)
    return 0


# ── verdicts: read by test_state_lifecycle.py, one final process ─────────────────────────────────

NODES = {
    "test_native_state_metadata_and_replay": {"tp4": ["metadata"], "dp2": ["metadata"]},
    "test_fresh_request_state_lifecycle": {s: ["d99", "continued", "fresh_during_decode"] for s in ("tp4", "dp2", "apc")},
    "test_dp2_correctness_under_load": {"dp2": ["d99", "continued", "fresh_during_decode", "dp_idle", "dp_load",
                                                "dp_quality"]},
    "test_prefix_reuse_and_continuations": {"apc": ["a1", "a3", "a4", "a5", "a6"]},
    "test_prefix_branch_state_and_binding": {"apc": ["a2", "binding"]},
    "test_prefix_turnover_and_resume": {"pressure": ["p1", "p2", "p4"]},
}
CONTINUATIONS = {
    "test_fresh_request_state_lifecycle": {s: ["d99", "continued", "fresh_during_decode"] for s in ("tp4", "dp2", "apc")},
    "test_dp2_correctness_under_load": {"dp2": ["dp_idle", "dp_load"]},
    "test_prefix_reuse_and_continuations": {"apc": ["a1", "a3", "a4", "a5", "a6"]},
    "test_prefix_branch_state_and_binding": {"apc": ["a2", "binding"]},
    "test_prefix_turnover_and_resume": {"pressure": ["p1", "p2", "p4"]},
}
FAULT_REASONS = {"observer_overflow": "observer_overflow"}


class Verdict:
    """One node's evidence: candidate failures, observation/reference/verifier faults and blocks, kept apart."""

    def __init__(self, node: str):
        self.node = node
        self.candidate: list[tuple[str, tuple[str, ...]]] = []
        self.unobserved: set[str] = set()
        self.observation: list[dict] = []
        self.reference: list[str] = []
        self.verifier: list[str] = []
        self.blocked: list[dict] = []
        self.summary: dict = {}
        self.unjudged = 0
        self.exception: dict | None = None
        self._bounded = None

    def fail(self, message: str, cases=()):
        """A candidate problem. `cases` names the observed case/waves whose worker records it is read from: it is
        proved only when each of them was completely observed. Missing worker files alone do not prove
        which cause occurred, so a problem read from an incompletely observed case is kept as unproven beside that
        case's observation fault; problems read from responses or the reference are proved as they stand."""
        self.candidate.append((message, tuple(cases)))

    def observe(self, reason: str, message: str, evidence=None):
        self.observation.append({"reason": reason, "message": message,
                                 "evidence": None if evidence is None else str(evidence)})

    def bounded(self) -> dict:
        if self._bounded is None:
            attribution = _load("infb_lifecycle_attribution", TESTS / "attribution.py")
            self._bounded = attribution.bounded_steps(Path(os.environ.get("RUN_LOG", "/logs/verifier/run.log")))
        return self._bounded

    def incomplete(self, step: str, message: str):
        """A producer that left no complete record: blocked when its step ended at its bound, else the verifier's."""
        row = self.bounded().get(step)
        if row is not None and row.get("reason") == "bound":
            self.blocked.append({"cause": step, "message": message})
        else:
            self.verifier.append(message)

    def checker(self, exc: BaseException):
        """An exception that escaped this node's own analysis (`records.checker_exception`: type, message,
        file:line, check). The node is unmeasured, the verifier's, unless a problem it had already proved stands."""
        from infb_verify import records  # noqa: PLC0415

        self.exception = records.checker_exception(self.node, exc)
        self.verifier.append(f"the {self.node} analysis raised {type(exc).__name__}: {exc}"[:2000])

    def conclude(self, record: dict):
        from infb_verify import records  # noqa: PLC0415

        proved = [m for m, cases in self.candidate if not self.unobserved.intersection(cases)]
        unproven = [{"problem": m, "unobserved": sorted(self.unobserved.intersection(cases))}
                    for m, cases in self.candidate if self.unobserved.intersection(cases)]
        record.update(candidate_problems=proved, unproven_candidate_problems=unproven,
                      observation_faults=self.observation, reference_faults=self.reference,
                      verifier_faults=self.verifier, blocked=self.blocked, summary=self.summary)
        if self.exception is not None:
            record["checker_exception"] = self.exception
        if proved:
            raise AssertionError(f"{len(proved)} problem(s): " + "; ".join(proved[:12]))
        if self.exception is not None:
            fault = records.VerifierFault("; ".join(self.verifier[:6]))
            fault.checker_exception = self.exception
            raise fault
        if self.reference:
            raise records.ReferenceFault("; ".join(self.reference[:6]))
        if self.observation:
            first = self.observation[0]
            raise records.ObservationFault(first["reason"], f"{len(self.observation)} observation fault(s); first: "
                                                            f"{first['message']}", first["evidence"])
        if self.unjudged:
            self.verifier.append(f"T_lifecycle is pending in policy.json: {self.unjudged} continuation(s) recorded, "
                                 "none judged")
        if self.verifier:
            raise records.VerifierFault("; ".join(self.verifier[:6]))
        if self.blocked:
            raise records.Blocked(self.blocked[0]["cause"], self.blocked[0]["message"])


def session_record(v: Verdict, name: str, families: list[str]) -> dict | None:
    path = ROOT / name / "session.json"
    if not path.is_file():
        v.incomplete(f"lifecycle-{name}", f"the {name} session left no record")
        return None
    rec = read_json(path)
    if rec.get("state") != "complete" and rec.get("state") != "failed":
        v.incomplete(f"lifecycle-{name}", f"the {name} session did not complete its record (state {rec.get('state')})")
    for f in rec.get("failures", []):
        if f.get("family") is not None and f["family"] not in families:
            continue
        label = f"{name}/{f.get('family') or 'session'}: {f.get('message')}"
        if f["kind"] == "candidate":
            v.fail(label)
        elif f["kind"] == "observation":
            v.observe(f.get("reason") or "coverage_unreached", label, f.get("evidence"))
        elif f["kind"] == "reference":
            v.reference.append(label)
        else:
            v.verifier.append(label)
    for family in families:
        entry = rec.get("families", {}).get(family)
        if rec.get("launch", {}).get("healthy") and entry is None and rec.get("state") == "complete":
            v.verifier.append(f"{name}/{family}: never ran")
    recipe = rec.get("recipe")
    if recipe and recipe.get("problems"):
        v.fail(f"{name}: the {rec['deployment']} deployment is not served as launch.py states it: "
               + "; ".join(recipe["problems"][:6]))
    return rec


def family_record(rec: dict | None, name: str, family: str) -> dict | None:
    if rec is None or rec.get("families", {}).get(family, {}).get("state") != "complete":
        return None
    path = ROOT / name / f"{family}.json"
    return read_json(path) if path.is_file() else None


class Files(dict):
    """One case/wave's worker records by file name; `case` is its label in `Verdict.fail`'s `cases`."""

    case = ""


def case_checks(v: Verdict, deployment: str, case: str, wave: int = 1) -> Files:
    """Rank coverage, observer faults, the candidate's mapping, the shape of every schedule record and live-row state
    isolation of one case/wave. A case with a rank-collection problem or an observer fault is incompletely observed
    (`Verdict.unobserved`)."""
    label = f"{deployment}/{case}/wave-{wave}"
    files = Files(load_case(deployment, case, wave))
    files.case = label
    collection, faults = rank_problems(files, deployment)
    for problem in collection:
        v.observe("rank_collection", f"{label}: {problem}", case_dir(deployment, case, wave))
    mapping = mapping_fault(files)
    for problem in mapping:
        v.fail(f"{label}: {problem}")
    if not mapping:
        for fault in faults:
            reason = next((r for k, r in FAULT_REASONS.items() if k in fault), "observer_incompatible")
            v.observe(reason, f"{label}: {fault}", case_dir(deployment, case, wave))
    if collection or (faults and not mapping):
        v.unobserved.add(label)
    for name, events in files.items():
        block = state_block(events)
        steps: dict = {}
        malformed = []
        for e in events:
            if e.get("event") != "schedule":
                continue
            problem = shape_problem(e)
            if problem is not None:
                malformed.append(f"step {e.get('step')}: request {e.get('request_id')}: {problem}")
            elif e.get("state_indices"):
                steps.setdefault(e["step"], []).append(e)
        if malformed:
            v.fail(f"{label}/{name}: {len(malformed)} malformed schedule record(s); first: {malformed[0]}", [label])
        incomparable = []
        for step, rows in steps.items():
            if len({len(r["state_indices"]) for r in rows}) != 1:
                incomparable.append(f"step {step}: the live rows carry unequal layer counts of state indices")
                continue
            if block is None and any(isinstance(x, list) for r in rows for x in r["state_indices"]):
                incomparable.append(f"step {step}: block-table state indices with no state group block size")
                continue
            conflict = isolation_conflict(rows, block)
            if conflict is not None:
                v.fail(f"{label}/{name}: step {step}: {conflict}", [label])
        if incomparable:
            v.observe("observer_incompatible", f"{label}/{name}: {len(incomparable)} step(s) whose state isolation "
                                               f"cannot be read; first: {incomparable[0]}",
                      case_dir(deployment, case, wave))
            v.unobserved.add(label)
    return files


def reference_targets(v: Verdict) -> dict | None:
    path = ROOT / "reference" / "teacher-forced.json"
    if not path.is_file():
        v.incomplete("lifecycle-reference", "the independent cold reference left no record")
        return None
    rec = read_json(path)
    if rec.get("state") != "complete":
        if rec.get("state") == "running":
            v.incomplete("lifecycle-reference", "the independent cold reference did not complete")
        else:
            v.reference.append(f"the independent cold reference failed: {rec.get('error')}")
        return None
    return rec["targets"]


_SELF: dict[str, dict] = {}


def self_targets(name: str) -> dict:
    """The session's supplemental cold self-readings, empty when it recorded none."""
    if name not in _SELF:
        path = ROOT / name / "self-teacher-forced.json"
        _SELF[name] = read_json(path).get("targets", {}) if path.is_file() else {}
    return _SELF[name]


def continuation_rows(v: Verdict, name: str, family: str, requests: list[dict], targets: dict | None) -> list[dict]:
    """For every scored target: the strict reading, then k3 over the whole continuation and every window
    of 4 and 16 against the independent reference's teacher-forced values, judged at T_lifecycle. Beside it, never
    judged, the same statistic against the candidate's own cold teacher-forcing (supplemental evidence)."""
    T = LC.tolerance()
    own = self_targets(name)
    rows = []
    for r in requests:
        if not r.get("scored"):
            continue
        key = f"{name}/{family}/{r['name']}"
        partial = bool(r.get("cancelled") and r.get("cancel_expected"))
        problems = LC.reading_problems(r, partial=partial)
        if r.get("cancelled") and not r.get("cancel_expected"):
            problems.append("the stream ended early")
        if problems:
            v.fail(f"{key}: {problems[:3]}")
            continue
        if targets is None:
            continue
        t = targets.get(key)
        if t is None or t.get("ids_digest") != LC.token_digest(r["ids"]) \
                or t.get("prompt_digest") != LC.token_digest(r["prompt"]):
            v.reference.append(f"{key}: no matching teacher-forced reference reading")
            continue
        readings = LC.window_readings(t["reference"], r["logprobs"])
        row = {"key": key, "n": readings["n"], "full": readings["full"],
               "max": {w: x["max"] for w, x in readings["windows"].items()},
               "start": {w: x["start"] for w, x in readings["windows"].items()}}
        mine = own.get(key)
        if mine is not None and mine.get("ids_digest") == t["ids_digest"] and len(mine.get("self", [])) == readings["n"]:
            cold = LC.window_readings(mine["self"], r["logprobs"])
            row["self"] = {"full": cold["full"], "max": {w: x["max"] for w, x in cold["windows"].items()}}
        if T is None:
            v.unjudged += 1
        else:
            judged = LC.judge(readings, T)
            row["passed"] = judged["passed"]
            if not judged["passed"]:
                row["violations"] = judged["violations"][:32]
                v.fail(f"{key}: {len(judged['violations'])} value(s) above T_lifecycle {T}: {judged['violations'][:3]}")
        rows.append(row)
    return rows


def _consumes(files: dict, rid: str) -> list[dict]:
    return [e for events in files.values() for e in events
            if e.get("event") == "consume" and same_request(rid, e.get("request_id"))]


def _foreign(events: list[dict]) -> list[dict]:
    """Restores of a state another request produced (or whose producer was not observed). A preempted request that
    resumes from its own retained checkpoint continues its own history, which every family allows;
    whether that restore is exact is `_state_problems`' question, not this one."""
    return [e for e in events
            if e.get("producer") is None or e["producer"].get("request_id") != e.get("request_id")]


def _within_policy(errors: list | None, bars: dict) -> bool:
    """Whether one candidate producer's differing views all sit inside the published KDA state policy.

    A candidate whose retained copy was evicted has no errors to read and is **unavailable**, never a pass.
    """
    if not errors:
        return False
    for err in errors:
        if not err.get("finite", True):
            return False
        if (err["relative_rmse"] > bars["state"]["relative_rmse"]
                and err["max_abs"] > bars["state"]["absolute_escape"]):
            return False
    return True


def _state_problems(events: list[dict], bars: dict) -> list[str]:
    """A restore compared with the producer it read: bitwise equal, or every differing view within the KDA state
    policy against **some** completed producer of that prefix. A prefix's content can be produced by several
    requests in one run, with genuinely different states; comparing only against the last of them judged a
    restore against a checkpoint the request never read."""
    out = []
    for e in events:
        if e.get("same_step_exact_producer"):
            out.append(f"tp{e['tp_rank']}: restored a state still being produced in the same step "
                       f"({e['same_step_exact_producer']}), equal to no completed checkpoint")
            continue
        if e.get("match") is False:
            candidates = e.get("candidate_producers")
            if candidates is not None and any(_within_policy(c.get("errors"), bars) for c in candidates):
                continue
            for err in e.get("state_errors") or []:
                if not err.get("finite", True):
                    out.append(f"tp{e['tp_rank']} layer {err['layer']} {err['view']}: non-finite restored state")
                elif (err["relative_rmse"] > bars["state"]["relative_rmse"]
                      and err["max_abs"] > bars["state"]["absolute_escape"]):
                    out.append(f"tp{e['tp_rank']} layer {err['layer']} {err['view']}: restored state relative RMSE "
                               f"{err['relative_rmse']:.3g} against its checkpoint")
            if e.get("completed_without_copy"):
                out.append(f"tp{e['tp_rank']}: {e['completed_without_copy']} completed producer(s) of this "
                           f"prefix could not be compared, their retained copy having been evicted")
            if e.get("numerical_available") is False:
                out.append(f"tp{e['tp_rank']}: restored state differs bitwise from its checkpoint at "
                           f"{len(e.get('mismatched') or [])} view(s), and the checkpoint copy was not retained")
    return out


def _produced(files: dict, rid: str) -> dict[int, dict]:
    """The request's produce events (tp0) by prefix length: the boundaries its execution completed."""
    out = {}
    for name, events in files.items():
        if "-tp0-" not in name:
            continue
        for e in events:
            if e.get("event") == "produce" and same_request(rid, e.get("request_id")):
                out[e["prefix_length"]] = e
    return out


def _tp0(files: dict) -> list[dict]:
    return [e for name, events in files.items() if "-tp0-" in name for e in events]


def _expect_restore(v: Verdict, files: dict, label: str, r: dict, length: int, producer: dict | None, bars: dict,
                    required_comparison: bool = False) -> bool:
    """A deliberate hit: restored at exactly `length` on every rank, equal to its checkpoint. Returns whether the
    restore was observed. Without an observed completed and retained producer the hit is not required."""
    consumed = _consumes(files, r["request_id"])
    ranks = {e["tp_rank"] for e in consumed if e.get("prefix_length") == length}
    expected = {t for d, t in EXPECTED_RANKS["apc"]}
    if not consumed:
        if producer is not None and _evicted(_tp0(files), producer) is None:
            v.fail(f"{label}: the completed, retained {length}-token prefix was not restored (no real reuse)",
                   [files.case])
        return False
    wrong = sorted({e.get("prefix_length") for e in consumed} - {length})
    if wrong:
        v.fail(f"{label}: restored a {wrong}-token prefix where the largest completed retained boundary is {length}",
               [files.case])
    if ranks != expected:
        v.observe("rank_collection", f"{label}: the restore is recorded on TP ranks {sorted(ranks)}")
        v.unobserved.add(files.case)
    for problem in _state_problems(consumed, bars):
        v.fail(f"{label}: {problem}", [files.case])
    if required_comparison and any(e.get("match") is None for e in consumed):
        v.observe("observer_incompatible", f"{label}: no producer snapshot of the restored {length}-token prefix")
    return True


def _bars() -> dict:
    return json.loads((EVAL / "kda/numerical-policy.json").read_text())


def collect_observer_events(observer, into):
    """Copy the observer's **produce and consume** events into the verifier's collected records.

    Output only: nothing here decides anything, no bar and no graded logic. It exists because the attribution a
    verdict rests on cannot be audited after the fact from the verdicts alone: an earlier revision kept the
    verdicts and none of the events, so a question about which producer a restore was judged against had no
    answer. The tensor dumps are deliberately left behind: on a full run the whole observer tree is about
    272 MB, and these two event kinds are 6 to 8 MB of it.

    Returns {'files': n, 'events': n, 'bytes': n} and writes `index.json` beside the copies.
    """
    out = {'files': 0, 'events': 0, 'bytes': 0, 'sources': {}}
    observer, into = Path(observer), Path(into)
    if not observer.is_dir():
        return out
    for src in sorted(observer.rglob('*.jsonl')):
        kept = []
        for line in src.open(errors='replace'):
            if '"produce"' not in line and '"consume"' not in line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if isinstance(event, dict) and event.get('event') in ('produce', 'consume'):
                kept.append(line if line.endswith('\n') else line + '\n')
        if not kept:
            continue
        dest = into / src.relative_to(observer)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(''.join(kept))
        size = sum(len(x) for x in kept)
        out['files'] += 1
        out['events'] += len(kept)
        out['bytes'] += size
        out['sources'][str(src.relative_to(observer))] = {'events': len(kept), 'bytes': size}
    into.mkdir(parents=True, exist_ok=True)
    (into / 'index.json').write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


def _graph_replays(events: list[dict]) -> int:
    return sum(1 for e in events if e.get("event") == "step" and e.get("graph_mode") == "FULL"
               and (e.get("layer_calls_device") or [0])[0] >= 1 and (e.get("layer_calls_python") or [0, 1])[1] == 0)


def check_metadata(v: Verdict) -> None:
    verdicts = _load("infb_lifecycle_verdicts", TESTS / "infb_kda_private/verdicts.py")
    bars = _bars()
    for name, deployment in (("tp4", "tp4"), ("dp2", "dp2")):
        rec = session_record(v, name, ["metadata"])
        fam = family_record(rec, name, "metadata")
        if fam is None:
            continue
        files = sorted((ROOT / name / "metadata").glob("rank-*.json"))
        records = [read_json(f) for f in files]
        expected = len(EXPECTED_RANKS[deployment])
        if fam.get("candidate_problems"):
            for p in fam["candidate_problems"]:
                v.fail(f"{name}/metadata: {p}")
            continue
        scope = f"{name}/metadata"
        if len(records) != expected:
            v.observe("rank_collection", f"{name}/metadata: {len(records)} worker records, not {expected}",
                      ROOT / name / "metadata")
            v.unobserved.add(scope)
        problems = verdicts.metadata_problems(records, bars, deployment)
        for problem in problems:
            if problem.startswith(f"{deployment}: worker"):
                v.observe("rank_collection", problem, ROOT / name / "metadata")
                v.unobserved.add(scope)
            else:
                v.fail(problem, [scope])
        v.summary[name] = {"workers": len(records), "problems": len(problems)}


def check_fresh(v: Verdict) -> None:
    targets = reference_targets(v)
    rows = []
    for name in ("tp4", "dp2", "apc"):
        families = NODES[v.node][name]
        rec = session_record(v, name, families)
        deployment = SESSIONS[name]["deployment"]
        d99 = family_record(rec, name, "d99")
        if d99 is not None:
            files = case_checks(v, deployment, "d99")
            missing = [k for k, ok in d99["slots"].items() if not ok]
            if missing:
                v.observe("coverage_unreached", f"{name}/d99: {len(missing)} probe slot(s) never landed on reused state "
                                                f"storage within {d99['turnover']['rounds_total']} turnover rounds: "
                                                f"{missing[:8]}", case_dir(deployment, "d99", 1))
            for dp in sorted({d for d, _ in EXPECTED_RANKS[deployment]}):
                replays = _graph_replays([e for n, ev in files.items() if f"-dp{dp}-tp0-" in n for e in ev])
                if not replays:
                    v.fail(f"{name}/d99: replica {dp}: ordinary decode never replayed a full CUDA graph", [files.case])
            rows += continuation_rows(v, name, "d99", d99["requests"], targets)
            v.summary[f"{name}/d99"] = {"turnover_rounds": d99["turnover"]["rounds_total"],
                                        "probes": len(d99["requests"]),
                                        "reused": sum(1 for r in d99["requests"] if r["reuse"]["reused"])}
        if d99 is not None:
            _no_restores(v, f"{name}/d99", files)
        cont = family_record(rec, name, "continued")
        if cont is not None:
            _no_restores(v, f"{name}/continued", case_checks(v, deployment, "continued"))
            rows += continuation_rows(v, name, "continued", cont["requests"], targets)
        fdd = family_record(rec, name, "fresh_during_decode")
        if fdd is not None:
            for w in fdd["waves"]:
                _no_restores(v, f"{name}/fresh_during_decode/wave-{w['wave']}",
                             case_checks(v, deployment, "fresh_during_decode", w["wave"]))
            if not any(w["overlap_count"] for w in fdd["waves"]):
                v.observe("coverage_unreached", f"{name}/fresh_during_decode: no step scheduled a fresh probe beside "
                                                f"running requests in {len(fdd['waves'])} wave(s)")
            rows += continuation_rows(v, name, "fresh_during_decode", fdd["requests"], targets)
    v.summary["continuations"] = _summarize(rows)
    v.summary["rows"] = rows


def _no_restores(v: Verdict, label: str, files: dict) -> None:
    """Families whose every request has its own salt (or no prefix cache): no request may start from a restored
    prefix state."""
    consumed = [e for events in files.values() for e in events if e.get("event") == "consume"]
    restored = sorted({e["request_id"] for e in _foreign(consumed)})
    if restored:
        v.fail(f"{label}: {len(restored)} request(s) with no shared prefix started from another request's restored "
               f"state: {restored[:4]}", [files.case])


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    worst = max(rows, key=lambda r: max([r["full"], *r["max"].values()]))
    out = {"n": len(rows), "max_full": max(r["full"] for r in rows),
           "max_window": {w: max(r["max"][w] for r in rows) for w in rows[0]["max"]},
           "worst": worst["key"], "failed": sum(1 for r in rows if r.get("passed") is False)}
    own = [r["self"] for r in rows if "self" in r]
    out["self_consistency"] = {"n": len(own), "max_full": max((x["full"] for x in own), default=None),
                               "max_window": {w: max(x["max"][w] for x in own) for w in own[0]["max"]} if own else {}}
    return out


def check_dp2(v: Verdict) -> None:
    targets = reference_targets(v)
    name, deployment = "dp2", "dp2"
    rec = session_record(v, name, NODES[v.node][name])
    if rec is None:
        return
    events_by_dp: dict = {0: [], 1: []}
    observed: list[str] = []
    for family, cases in (("d99", [("d99", 1)]), ("continued", [("continued", 1)]), ("dp_idle", [("dp_idle", 1)]),
                          ("dp_load", [("dp_load", 1)]), ("fresh_during_decode", None)):
        fam = family_record(rec, name, family)
        if fam is None:
            continue
        waves = cases or [("fresh_during_decode", w["wave"]) for w in fam["waves"]]
        for case, wave in waves:
            files = case_checks(v, deployment, case, wave)
            observed.append(files.case)
            for n, events in files.items():
                for dp in (0, 1):
                    if f"-dp{dp}-tp0-" in n:
                        events_by_dp[dp] += events
    real = {dp: sum(1 for e in ev if e.get("event") == "step" and e.get("live_rows", 0) > 0)
            for dp, ev in events_by_dp.items()}
    dummies = {dp: sum(1 for e in ev if e.get("event") == "dummy") for dp, ev in events_by_dp.items()}
    replays = {dp: _graph_replays(ev) for dp, ev in events_by_dp.items()}
    beyond = {}
    for dp, ev in events_by_dp.items():
        begin = next((e for e in ev if e.get("event") == "begin"), None)
        sizes = sorted((begin or {}).get("engine", {}).get("cudagraph_capture_sizes") or [])

        def own_pad(tokens):
            return min((x for x in sizes if x >= tokens), default=tokens)
        beyond[dp] = sum(1 for e in ev if e.get("event") in ("step", "dummy") and e.get("graph_mode") in
                         ("FULL", "PIECEWISE") and e.get("graph_extent") is not None
                         and e["graph_extent"] > own_pad(e.get("total_tokens") if e["event"] == "step"
                                                         else e.get("requested_tokens") or 1))
    v.summary["replicas"] = {"real_steps": real, "dummy_forwards": dummies, "full_graph_replays": replays,
                             "extent_beyond_own_padded": beyond}
    if any(real.values()):
        for dp in (0, 1):
            if not real[dp]:
                v.fail(f"dp2: replica {dp} never scheduled a real request (two TP4 replicas are required)", observed)
            if not replays[dp]:
                v.fail(f"dp2: replica {dp}: ordinary decode never replayed a full CUDA graph", observed)
        if not any(dummies.values()):
            v.observe("coverage_unreached", "dp2: no dummy/idle forward observed on either replica")
        if not any(beyond.values()):
            v.observe("coverage_unreached", "dp2: no replayed graph extent above a replica's own padded live count")
    rows = []
    for family in ("dp_idle", "dp_load"):
        fam = family_record(rec, name, family)
        if fam is not None:
            rows += continuation_rows(v, name, family, fam["requests"], targets)
    v.summary["continuations"] = _summarize(rows)
    v.summary["rows"] = rows
    quality = family_record(rec, name, "dp_quality")
    if quality is not None:
        g = quality.get("gsm8k")
        if g is None:
            v.fail(f"dp2/gsm8k: {quality.get('gsm8k_error')}")
        else:
            if g["accuracy"] < X["gsm8k"]["floor"]:
                v.fail(f"dp2/gsm8k: accuracy {g['accuracy']:.4f} below the floor {X['gsm8k']['floor']}")
            if g["invalid_rate"] > X["gsm8k"]["max_invalid_rate"]:
                v.fail(f"dp2/gsm8k: invalid rate {g['invalid_rate']:.4f} above {X['gsm8k']['max_invalid_rate']}")
            v.summary["gsm8k"] = {k: g.get(k) for k in ("accuracy", "invalid_rate", "num_questions")}
        f = quality.get("fidelity")
        if f is None:
            error = quality.get("fidelity_error") or "no fidelity record"
            (v.reference.append if error.startswith("ReferenceFault") else v.fail)(f"dp2/fidelity: {error}")
        else:
            for band, b in f.items():
                if not b["passed"]:
                    v.fail(f"dp2/fidelity: band {band} median k3 {[d['median'] for d in b['draws']]} not below "
                           f"{b['T_b']}")
            v.summary["fidelity"] = {band: [d["median"] for d in b["draws"]] for band, b in f.items()}
        c = quality.get("concurrency")
        if c is None:
            v.fail(f"dp2/concurrency: {quality.get('concurrency_error')}")
        else:
            for band, ok in c["passed"].items():
                if not ok:
                    v.fail(f"dp2/concurrency: band {band} medians {[d['medians'][band] for d in c['draws']]} not below "
                           f"{X['concurrency']['T_per_band'][band]}")
            v.summary["concurrency"] = {"passed": c["passed"], "medians": [d["medians"] for d in c["draws"]]}


def check_prefix_reuse(v: Verdict) -> None:
    targets = reference_targets(v)
    name, deployment = "apc", "apc"
    rec = session_record(v, name, NODES[v.node][name])
    bars = _bars()
    rows, hits, observed = [], 0, []
    for family in ("a1", "a3", "a4", "a5", "a6"):
        fam = family_record(rec, name, family)
        if fam is None:
            continue
        files = case_checks(v, deployment, family)
        observed.append(files.case)
        tp0 = _tp0(files)
        producers: dict = {}
        for e in tp0:
            if e.get("event") == "produce":
                producers[(e["prefix_length"], e["prefix_digest"])] = e
        for r in fam["requests"]:
            label = f"{name}/{family}/{r['name']}"
            expect = r.get("expect_hit")
            if expect == 0:
                if _foreign(_consumes(files, r["request_id"])):
                    v.fail(f"{label}: a fresh prompt with its own salt restored another request's prefix state",
                           [files.case])
                for problem in _state_problems(_consumes(files, r["request_id"]), bars):
                    v.fail(f"{label}: {problem}", [files.case])
                continue
            if family == "a3" and r.get("role") == "cold-sibling":
                for e in _consumes(files, r["request_id"]):
                    producer = e.get("producer")
                    if (producer is None or e.get("completed_producers", 1) == 0
                            or producer["step"] >= e["step"]):
                        v.fail(f"{label}: restored a {e['prefix_length']}-token state not completed in an earlier step",
                               [files.case])
                for problem in _state_problems(_consumes(files, r["request_id"]), bars):
                    v.fail(f"{label}: {problem}", [files.case])
                continue
            shared = r.get("shared_prefix") if r.get("shared_prefix") is not None else expect
            if shared is None:
                continue
            prompt = r["prompt"]
            candidates = [(k, p) for (k, digest), p in producers.items()
                          if k <= shared and digest == LC.token_digest(prompt[:k]) and p["step"] < _first_step(tp0, r)
                          and _evicted(tp0, p) is None]
            if not candidates:
                if _consumes(files, r["request_id"]):
                    restored = _consumes(files, r["request_id"])
                    for problem in _state_problems(restored, bars):
                        v.fail(f"{label}: {problem}", [files.case])
                    hits += 1
                continue
            length, producer = max(candidates, key=lambda c: c[0])
            if _expect_restore(v, files, label, r, length, producer, bars):
                hits += 1
        rows += continuation_rows(v, name, family, fam["requests"], targets)
    if rec is not None and not hits and not v.blocked:
        v.fail("apc: no deliberate warm-prefix request restored a completed prefix (prefix reuse never observed)",
               observed)
    v.summary["restores"] = hits
    v.summary["continuations"] = _summarize(rows)
    v.summary["rows"] = rows


def _first_step(events: list[dict], r: dict) -> int:
    steps = [e["step"] for e in events if e.get("event") == "schedule" and same_request(r["request_id"],
                                                                                         e.get("request_id"))]
    return min(steps) if steps else 1 << 30


def check_branch_binding(v: Verdict) -> None:
    targets = reference_targets(v)
    verdicts = _load("infb_lifecycle_verdicts_b", TESTS / "infb_kda_private/verdicts.py")
    name, deployment = "apc", "apc"
    rec = session_record(v, name, NODES[v.node][name])
    bars = _bars()
    rows = []
    a2 = family_record(rec, name, "a2")
    if a2 is not None:
        files = case_checks(v, deployment, "a2")
        warm = a2["requests"][0]
        produced = _produced(files, warm["request_id"])
        prefix = FAMILIES["a2"]["prefix"]
        producer = produced.get(prefix)
        for r in a2["requests"][1:]:
            _expect_restore(v, files, f"apc/a2/{r['name']}", r, prefix, producer, bars, required_comparison=True)
        if producer is None:
            v.observe("coverage_unreached", f"apc/a2: no completed {prefix}-token checkpoint observed for the warm prefix")
        rows += continuation_rows(v, name, "a2", a2["requests"], targets)
    binding = family_record(rec, name, "binding")
    if binding is not None:
        prefix = FAMILIES["binding"]["prefix"]
        v.summary["binding"] = []
        for p in binding["passes"]:
            warm_c, control, warm_t, target, sibling = p["requests"]
            label = f"apc/binding/l{p['layer']}-{'-'.join(p['views'])}"
            files_c = case_checks(v, deployment, p["control_case"])
            files_t = case_checks(v, deployment, p["target_case"])
            _expect_restore(v, files_c, f"{label}/control", control, prefix,
                            _produced(files_c, warm_c["request_id"]).get(prefix), bars, required_comparison=True)
            restored = _expect_restore(v, files_t, f"{label}/target", target, prefix,
                                       _produced(files_t, warm_t["request_id"]).get(prefix), bars,
                                       required_comparison=True)
            _expect_restore(v, files_t, f"{label}/sibling", sibling, prefix,
                            _produced(files_t, warm_t["request_id"]).get(prefix), bars, required_comparison=True)
            applied = [e for events in files_t.values() for e in events if e.get("event") == "intervene"]
            if restored and len({e["tp_rank"] for e in applied}) != len(EXPECTED_RANKS[deployment]):
                v.observe("observer_incompatible", f"{label}: the intervention was applied on "
                                                   f"{sorted({e['tp_rank'] for e in applied})} of the TP ranks")
            for problem in LC.reading_problems(target):
                v.fail(f"{label}/target: {problem}")
            try:
                wrap = lambda g: {"choices": [{"logprobs": {"tokens": g["ids"], "token_logprobs": g["logprobs"]}}]}  # noqa: E731
                natural = verdicts.completion_distance(wrap(warm_c), wrap(warm_t))
                effect = verdicts.completion_distance(wrap(control), wrap(target))
            except ValueError as exc:
                v.fail(f"{label}: {exc}")
                continue
            criteria = X["kda"]["acceptance"]["binding"]
            bar = max(criteria["absolute_floor"], criteria["noise_multiplier"] * natural["max_logprob_delta"])
            if natural["token_changes"]:
                v.observe("observer_interference", f"{label}: the two matched warm requests differ in tokens; the "
                                                   "intervention is inconclusive")
            elif restored and applied and not effect["token_changes"] and effect["max_logprob_delta"] <= bar:
                v.fail(f"{label}: perturbing the restored {'/'.join(p['views'])} state of layer {p['layer']} did not "
                       f"affect the continuation (max delta {effect['max_logprob_delta']:.3g}, bar {bar:.3g})",
                       [files_c.case, files_t.case])
            v.summary["binding"].append({"pass": label, "natural": natural, "effect": effect, "bar": bar,
                                         "applied_ranks": sorted({e["tp_rank"] for e in applied})})
        rows += continuation_rows(v, name, "binding", binding["requests"], targets)
    v.summary["continuations"] = _summarize(rows)
    v.summary["rows"] = rows


def _resumed_checkpoints(v: Verdict, files: Files) -> None:
    """Invariant 5: generated tokens do not turn a resumed recompute into ordinary decode.

    A prefill of a resumed request's complete history H stores the checkpoint at B* = floor(H / b) * b, its last cache
    position (the reference's `_mamba_block_aligned_split`, which #33706 applies to resumed requests). So every resume
    of a request with generated history (tp0) whose computed count C is below B* must store B*: the recompute step
    that first reaches B* ends on it, or crosses it with a real block at position B*/b - 1 in every state group (an
    in-step write, as an all-mode layout makes). A crossing step with the null block there is the candidate's failure.
    A recompute that does not reach B* within the case (finished, cancelled or preempted again) is not judged.
    Limit: block ids cannot tell an in-step write from an earlier chunk that ended inside B*'s block."""
    block = next((state_block(events) for name, events in files.items() if "-tp0-" in name), None)
    schedules = [e for e in _tp0(files) if e.get("event") == "schedule" and shape_problem(e) is None]
    counts = {"in_scope": 0, "ended_at_boundary": 0, "stored_in_step": 0, "violations": 0, "not_reached": 0}
    gaps = []
    for k, e in enumerate(schedules):
        if e.get("kind") != "resumed" or not e["output_count"]:
            continue
        if block is None or e.get("history") is None:
            gaps.append(f"request {e.get('request_id')} resumed at step {e.get('step')}: no "
                        f"{'state group block size' if block is None else 'history'} recorded")
            continue
        computed, history = e["computed_before"], e["history"]
        boundary = history // block * block
        if computed >= boundary:
            continue
        counts["in_scope"] += 1
        outcome = "not_reached"
        for x in schedules[k:]:
            if x.get("request_id") != e.get("request_id"):
                continue
            if x is not e and x.get("kind") == "resumed":
                break
            start, end = x["computed_before"], x["computed_before"] + x["scheduled_tokens"]
            if end < boundary:
                continue
            if start >= boundary:
                break
            if end == boundary:
                outcome = "ended_at_boundary"
                break
            column, blocks = boundary // block - 1, x.get("state_blocks")
            if not blocks:
                gaps.append(f"request {e.get('request_id')}: no state blocks recorded on its crossing step "
                            f"{x.get('step')}")
                outcome = "gap"
            elif any(len(ids) <= column or ids[column] == 0 for ids in blocks):
                outcome = "violations"
                entries = [ids[column] if len(ids) > column else None for ids in blocks]
                v.fail(f"{files.case}: request {e.get('request_id')} resumed at {computed} of {history} tokens of "
                       f"history; its recompute crossed its last cache position {boundary} at step {x.get('step')} "
                       f"({start} -> {end}) with no block at position {column} (state blocks {entries}): the "
                       "checkpoint a prefill of that history stores was not stored (invariant 5)", [files.case])
            else:
                outcome = "stored_in_step"
            break
        if outcome != "gap":
            counts[outcome] += 1
    if gaps:
        v.observe("observer_incompatible", f"{files.case}: {len(gaps)} resumed recompute(s) whose invariant-5 "
                                           f"checkpoint cannot be read; first: {gaps[0]}")
        v.unobserved.add(files.case)
    v.summary.setdefault("resumed_checkpoints", {})[files.case] = counts


def check_turnover_resume(v: Verdict) -> None:
    targets = reference_targets(v)
    name, deployment = "pressure", "apc-pressure"
    rec = session_record(v, name, NODES[v.node][name])
    bars = _bars()
    rows = []
    p1 = family_record(rec, name, "p1")
    if p1 is not None:
        files = case_checks(v, deployment, "p1")
        _resumed_checkpoints(v, files)
        by = {r["name"]: r for r in p1["requests"]}
        prefix = FAMILIES["p1"]["warm_prefix"]
        _expect_restore(v, files, "pressure/p1/hit", by["p1-hit"], prefix,
                        _produced(files, by["p1-warm"]["request_id"]).get(prefix), bars)
        if not any(f["evicted_by"] for f in p1["fills"]):
            v.observe("coverage_unreached", f"pressure/p1: the warm prefix was never evicted in {len(p1['fills'])} fill "
                                            "wave(s)")
        rewarm = by["p1-rewarm-hit"]
        sources = [_produced(files, by[n]["request_id"]).get(prefix) for n in ("p1-original", "p1-warm")]
        source = next((p for p in sources if p is not None and _evicted(_tp0(files), p) is None), None)
        _expect_restore(v, files, "pressure/p1/rewarm-hit", rewarm, prefix, source, bars)
        for r in p1["requests"]:
            if r.get("expect_hit") == 0 and _foreign(_consumes(files, r["request_id"])):
                v.fail(f"pressure/p1/{r['name']}: a prompt with its own salt restored another request's prefix state",
                       [files.case])
            for problem in _state_problems([e for e in _consumes(files, r["request_id"]) if not _foreign([e])], bars):
                v.fail(f"pressure/p1/{r['name']}: resumed from its own checkpoint: {problem}", [files.case])
        rows += continuation_rows(v, name, "p1", p1["requests"], targets)
    p2 = family_record(rec, name, "p2")
    if p2 is not None:
        reached2 = reached3 = False
        for w in p2["waves"]:
            files = case_checks(v, deployment, "p2", w["wave"])
            _resumed_checkpoints(v, files)
            if w["qualifying_events"]:
                reached2 = True
            selection = w.get("selection") or {}
            if w.get("branch"):
                reached3 = True
                branch = next(r for r in p2["requests"] if r["request_id"] == w["branch"])
                source = _produced(files, selection["request_id"].removeprefix("cmpl-").rsplit("-", 1)[0]) \
                    if selection.get("request_id") else {}
                _expect_restore(v, files, f"pressure/p2/wave-{w['wave']}/branch", branch, selection["boundary"],
                                source.get(selection["boundary"]), bars, required_comparison=True)
        if not reached2:
            v.observe("coverage_unreached", "pressure/p2: no resumed request with generated history in "
                                            f"{len(p2['waves'])} wave(s)")
        elif not reached3:
            v.observe("coverage_unreached", "pressure/p2: no qualifying resumed request retained a completed boundary")
        rows += continuation_rows(v, name, "p2", p2["requests"], targets)
    p4 = family_record(rec, name, "p4")
    if p4 is not None:
        files = case_checks(v, deployment, "p4")
        _resumed_checkpoints(v, files)
        tp0 = _tp0(files)
        for r in p4["requests"]:
            if not r.get("cancel_expected"):
                continue
            label = f"pressure/p4/{r['name']}"
            if not r.get("cancelled"):
                v.observe("coverage_unreached", f"{label}: completed before it could be cancelled")
                continue
            finish = [e for e in tp0 if e.get("event") == "finish" and same_request(r["request_id"], e.get("request_id"))]
            if not finish:
                v.fail(f"{label}: the engine never finished the cancelled request", [files.case])
                continue
            later = [e for e in tp0 if e.get("event") == "schedule" and same_request(r["request_id"], e.get("request_id"))
                     and e["step"] > finish[0]["step"]]
            if later:
                v.fail(f"{label}: scheduled again after it was cancelled and finished", [files.case])
        if p4.get("branch_source"):
            source = next(r for r in p4["requests"] if r["request_id"] == p4["branch_source"])
            produced = _produced(files, source["request_id"])
            for r in p4["requests"]:
                if r.get("role") == "branch":
                    _expect_restore(v, files, f"pressure/p4/{r['name']}", r, FAMILIES["p4"]["branch_prefix"],
                                    produced.get(FAMILIES["p4"]["branch_prefix"]), bars)
        else:
            v.observe("coverage_unreached", "pressure/p4: no unaffected request retained its 1,024-token boundary")
        rows += continuation_rows(v, name, "p4", p4["requests"], targets)
    v.summary["continuations"] = _summarize(rows)
    v.summary["rows"] = rows


CHECKS = {"test_native_state_metadata_and_replay": check_metadata,
          "test_fresh_request_state_lifecycle": check_fresh,
          "test_dp2_correctness_under_load": check_dp2,
          "test_prefix_reuse_and_continuations": check_prefix_reuse,
          "test_prefix_branch_state_and_binding": check_branch_binding,
          "test_prefix_turnover_and_resume": check_turnover_resume}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--session", required=True, choices=[*SESSIONS, "reference"])
    args = parser.parse_args(argv)
    return run_reference() if args.session == "reference" else run_session(args.session)


if __name__ == "__main__":
    sys.exit(main())
