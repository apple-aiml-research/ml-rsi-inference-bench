#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""L0's construction node: the registered class constructs at the recipe's layout, and its own `load_weights` takes every tensor of
the checkpoint exactly once, with zero data, shard by shard.

    python -I /tests/infb_verify/construct.py launch --request <request.json>

Verifier-only and trusted: test_l0.py runs it by path from /tests, never from /app, in the engine's interpreter, from
`/`, so the only engine it imports is the candidate's installed tree. The launcher builds the engine's configuration
once from the served command line (the recipe, `vllm serve <model> <args> --seed 0` as upstream's server wrapper
launches it), with the engine's own parser and `create_engine_config`, as `vllm serve` does. It then starts one process
per rank of that configuration (TP 4), each handed the configuration as the engine hands it to a worker, and each:

  1. takes its device and runs the engine's own worker distributed initialisation (`init_worker_distributed_environment`,
     NCCL at TP 4), as a served worker does;
  2. builds the model the way the engine's model loader does (`initialize_model` under the model's default dtype, on the
     rank's device), so the class is the one the registry resolves for the checkpoint's architecture;
  3. calls the constructed class's own `load_weights` once, over a lazy generator that walks the checkpoint shard by
     shard in the index's order and yields each tensor only when the loader draws it: the exact name, dtype and shape
     come from that shard's safetensors header, and the data is zeros. No tensor data is read and no shard is
     memory-mapped. Every name the index holds passes exactly once, which the generator checks as it goes and again
     at the end.

     Each tensor is backed by **its own untouched private anonymous mapping** of exactly its nbytes, wrapped with
     `torch.frombuffer` and reinterpreted to the header's dtype and shape. The node must not change what the class
     under test does, and what it does may be to hold every tensor at once: the reference's own `load_weights` sorts
     the whole iterable before it copies anything, and a candidate's may too. A zero-filled tensor costs physical
     memory, so holding the whole checkpoint would exhaust the host; a tensor on untouched anonymous pages reads as
     zeros from the kernel's shared zero page and costs address space instead. The mapping is contiguous and
     page-aligned, so every dtype reinterpretation or view a loader may do still works, and it is never written to.

Not run: the loader's `process_weights_after_loading`, the DSpark draft model and serving; the served nodes judge them.
A rank passes when steps 1-3 complete without an exception and the name check holds; the node passes when every rank
passes. The check is structural and deterministic: no tolerance, no threshold, no list of expected names.

Each rank's wall per phase, its peak host RSS (`VmHWM`) and its virtual size (`VmSize`) are recorded. The pair is the
evidence that the pass holds address space rather than memory. The peak is evidence, never graded; the node's bound is
`l0.construct.cap_s`.

Records: `<records>/rank<r>.json` for each rank and `<records>/construct.json` for the node, rewritten at every phase,
so a rank that dies leaves the phase it died in; the launcher's and every rank's output go to `<records>/construct.log`.
Module level imports only the standard library: test_l0.py imports `gpus_busy` and `gpus_free` from here in the pytest
process, which never imports the engine.
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

NODE = "infb_verify.test_l0.test_registered_class_constructs"
DTYPES = {"BOOL": "bool", "U8": "uint8", "I8": "int8", "F8_E5M2": "float8_e5m2", "F8_E4M3": "float8_e4m3fn",
          "F8_E8M0": "float8_e8m0fnu", "I16": "int16", "U16": "uint16", "F16": "float16", "BF16": "bfloat16",
          "I32": "int32", "U32": "uint32", "F32": "float32", "F64": "float64", "I64": "int64", "U64": "uint64"}


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write(path: Path, rec: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def _error(exc: BaseException) -> dict:
    return {"type": f"{type(exc).__module__}.{type(exc).__qualname__}", "message": str(exc)[:2000],
            "traceback": traceback.format_exc()[-6000:]}


def _rss() -> dict:
    """This process's host memory from /proc/self/status, in bytes: the current RSS, the peak (`VmHWM`) and the
    virtual size (`VmSize`). The pair is the evidence that the weights pass holds address space and not memory: a
    tensor backed by an untouched anonymous mapping grows VmSize and leaves VmHWM where it was. Evidence for the
    bound (the rule measures the wall); never graded, and a kernel without these fields reads null."""
    fields = {"VmRSS:": "rss_bytes", "VmHWM:": "peak_rss_bytes", "VmSize:": "virtual_bytes"}
    out: dict = dict.fromkeys(fields.values())
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key = line.split(maxsplit=1)[0] if line else ""
            if key in fields:
                out[fields[key]] = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return out


def gpus_busy() -> str | None:
    """None when no compute process holds any GPU; otherwise what holds them (the component gate's reading)."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"nvidia-smi could not be read: {exc}"
    if p.returncode != 0:
        return f"nvidia-smi exited {p.returncode}: {p.stderr.strip()[:300]}"
    held = [line.strip() for line in p.stdout.splitlines() if line.strip()]
    return ("GPU compute processes: " + "; ".join(held)[:600]) if held else None


def gpus_free(wait_s: float) -> str | None:
    """Wait up to `wait_s` for every compute process to leave the GPUs; None once they have, else what still holds them."""
    deadline = time.monotonic() + wait_s
    while True:
        busy = gpus_busy()
        if busy is None or time.monotonic() >= deadline:
            return busy
        time.sleep(2)


# ── the rank ─────────────────────────────────────────────────────────────────────────────────────────────────────


def _die_with_parent(parent: int) -> None:
    """SIGKILL this process when the launcher ends, however it ends (a timeout's kill included), so no rank outlives
    the node and holds a GPU into the next group."""
    import ctypes  # noqa: PLC0415

    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() != parent:
        os._exit(1)


class ZeroWeights:
    """The checkpoint's tensors with the data replaced (step 3), shard by shard in the index's order.

    The index (`model.safetensors.index.json`) is the authority for which names exist, for their order, and for the
    shard each lives in; a checkpoint with no index is the single `model.safetensors`. Each shard's safetensors header
    is the authority for a name's dtype and shape: the header is the file's own first bytes, read on its own, so no
    shard is opened for its data or memory-mapped.

    `generator()` is lazy. It creates a tensor only when the loader draws the next item, keeps no reference to one it
    has yielded, and holds only the current shard's header. What the class under test then does with the tensors is
    its own business -- it may hold them all -- so each one is backed by an untouched private anonymous mapping and
    costs address space rather than memory (see `zeros`).
    Every name of the index passes exactly once: a name seen twice, a name the shard's header does not hold, and a
    name left over at the end are each an error, with no tolerance.
    """

    MAP_FLAGS = mmap.MAP_PRIVATE | getattr(mmap, "MAP_ANONYMOUS", getattr(mmap, "MAP_ANON", 0)) \
        | getattr(mmap, "MAP_NORESERVE", 0)

    def __init__(self, torch, model_path: str):
        self.torch = torch
        self.root = Path(model_path)
        self.shards: list[str] = []
        self.by_shard: dict[str, list[str]] = {}
        self.names: list[str] = []
        self.calls, self.tensors, self.bytes, self.by_dtype = 0, 0, 0, {}
        self.mapped, self.mappings = 0, 0
        self._index()

    def _index(self) -> None:
        if not self.root.is_dir():
            raise NotADirectoryError(f"the configuration's model is {str(self.root)!r}, which is not a local "
                                     "checkpoint directory; the recipe serves the image's own checkpoint")
        idx = self.root / "model.safetensors.index.json"
        if idx.is_file():
            weight_map = json.loads(idx.read_text())["weight_map"]
            for name, shard in weight_map.items():            # the index's own order
                if name in self.by_shard.get(shard, ()):
                    raise ValueError(f"the checkpoint index names {name!r} twice")
                self.by_shard.setdefault(shard, []).append(name)
                self.names.append(name)
            self.shards = list(dict.fromkeys(weight_map.values()))
        else:
            shard = "model.safetensors"
            header = self._header(shard)
            self.shards, self.by_shard, self.names = [shard], {shard: list(header)}, list(header)
        if len(set(self.names)) != len(self.names):
            raise ValueError("the checkpoint index names a tensor more than once")
        self._check_headers()

    def _check_headers(self) -> None:
        """Read every shard's header once, before any tensor is made, and fail now rather than an hour in: the index's
        names for a shard must all be in that shard's header, and every dtype must be one this reader knows. Headers
        are the files' first bytes, so this costs one small read per shard."""
        for shard in self.shards:
            header = self._header(shard)
            for name in self.by_shard[shard]:
                entry = header.get(name)
                if entry is None:
                    raise KeyError(f"the index puts {name!r} in {shard}, whose header does not hold it")
                if entry["dtype"] not in DTYPES:
                    raise KeyError(f"{shard} gives {name!r} the dtype {entry['dtype']!r}, which this reader does "
                                   f"not know; known: {sorted(DTYPES)}")

    def _header(self, shard: str) -> dict:
        """A safetensors file's header alone: its first 8 bytes are the header's length, then that much JSON."""
        with open(self.root / shard, "rb") as f:
            size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(size))
        header.pop("__metadata__", None)
        return header

    def zeros(self, dtype_name: str, shape: list[int]):
        """A zero tensor of this dtype and shape that costs address space, not memory.

        The class under test may hold every tensor it is handed at once -- the reference's own `load_weights` sorts
        the whole iterable before it copies anything, and a candidate's may too. The node must not change what the
        class does, so it makes the tensors cheap to hold instead. A *zero-filled* tensor costs physical memory; a
        tensor backed by its own untouched private anonymous mapping does not, because reads of untouched anonymous
        pages come from the kernel's shared zero page. Holding a million of them costs virtual address space.

        The mapping is exactly `nbytes` long, is never written to, and is wrapped with `torch.frombuffer` as uint8,
        then reinterpreted and reshaped. The result is contiguous and page-aligned, so every dtype reinterpretation
        or view a loader may do still works. `torch.frombuffer` keeps the mapping alive for the tensor's lifetime.
        """
        dtype = getattr(self.torch, DTYPES[dtype_name])
        nbytes = math.prod(shape) * dtype.itemsize
        if nbytes == 0:                                     # an empty tensor maps nothing
            return self.torch.empty(shape, dtype=dtype)
        buf = mmap.mmap(-1, nbytes, flags=self.MAP_FLAGS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self.mapped += nbytes
        self.mappings += 1
        return self.torch.frombuffer(buf, dtype=self.torch.uint8).view(dtype).reshape(shape)

    def generator(self):
        self.calls += 1
        drawn = set()
        for shard in self.shards:
            header = self._header(shard)
            for name in self.by_shard[shard]:
                entry = header.get(name)
                if entry is None:
                    raise KeyError(f"the index puts {name!r} in {shard}, whose header does not hold it")
                if name in drawn:
                    raise ValueError(f"{name!r} was drawn twice")
                drawn.add(name)
                tensor = self.zeros(entry["dtype"], list(entry["shape"]))
                self.tensors += 1
                self.bytes += tensor.numel() * tensor.element_size()
                self.by_dtype[entry["dtype"]] = self.by_dtype.get(entry["dtype"], 0) + 1
                yield name, tensor
                del tensor
            del header              # only the current shard's header is live
        missing = [n for n in self.names if n not in drawn]
        if missing:
            raise ValueError(f"{len(missing)} checkpoint tensor(s) were never drawn, the first {missing[:3]}")

    def summary(self) -> dict:
        return {"generator_calls": self.calls, "shards": len(self.shards), "shard_names": self.shards,
                "index_names": len(self.names), "tensors": self.tensors, "bytes": self.bytes,
                "mappings": self.mappings, "mapped_bytes": self.mapped,
                "every_name_once": self.tensors == len(self.names), "by_dtype": self.by_dtype}


def rank_main(vllm_config, rank: int, world: int, init_method: str, records: str, parent: int) -> None:
    _die_with_parent(parent)
    path = Path(records) / f"rank{rank}.json"
    rec: dict = {"rank": rank, "world_size": world, "pid": os.getpid(), "started_at": _utc(), "status": "running",
                 "phase": "import", "phases": {}}
    t_phase = time.monotonic()

    def phase(name: str) -> None:
        nonlocal t_phase
        now = time.monotonic()
        rec["phases"][rec["phase"]] = round(now - t_phase, 3)
        rec["phase"], t_phase = name, now
        rec["host_rss"] = _rss()
        _write(path, rec)

    _write(path, rec)
    code = 1
    try:
        import torch  # noqa: PLC0415

        from vllm.config import set_current_vllm_config  # noqa: PLC0415
        from vllm.model_executor.model_loader import get_model_loader  # noqa: PLC0415
        from vllm.model_executor.model_loader.utils import initialize_model  # noqa: PLC0415
        from vllm.platforms import current_platform  # noqa: PLC0415
        from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed  # noqa: PLC0415
        from vllm.v1.worker.gpu_worker import init_worker_distributed_environment  # noqa: PLC0415

        model_config = vllm_config.model_config
        with set_current_vllm_config(vllm_config):
            phase("distributed")
            device = torch.device(f"cuda:{rank}")  # one node: the local rank is the rank
            torch.accelerator.set_device_index(device)
            rec["device"] = str(device)
            check = getattr(current_platform, "check_if_supports_dtype", None)
            if check is not None:
                check(model_config.dtype)
            init_worker_distributed_environment(vllm_config, rank, init_method, rank, current_platform.dist_backend)
            set_random_seed(model_config.seed)

            phase("construction")
            loader = get_model_loader(vllm_config.load_config)
            load_config = vllm_config.load_config
            load_device = vllm_config.device_config.device if load_config.device is None else load_config.device
            with set_default_torch_dtype(model_config.dtype):
                with torch.device(load_device):
                    model = initialize_model(vllm_config=vllm_config, model_config=model_config)
                cls = type(model)
                params = list(model.parameters())
                rec["model"] = {"class": f"{cls.__module__}.{cls.__qualname__}", "loader": type(loader).__name__,
                                "parameters": len(params),
                                "parameter_bytes": sum(p.numel() * p.element_size() for p in params),
                                "allocated_bytes": torch.cuda.memory_allocated(device)}
                del params

                phase("weights")
                zero = ZeroWeights(torch, model_config.model)
                try:
                    loaded = model.load_weights(zero.generator())
                finally:
                    rec["weights"] = zero.summary()
                    rec["host_rss"] = _rss()
                if isinstance(loaded, (set, frozenset, list, tuple, dict)):
                    rec["weights"]["loaded_names"] = len(loaded)
                if not rec["weights"]["every_name_once"]:
                    raise ValueError(f"{zero.tensors} tensor(s) passed, the index holds {len(zero.names)}")
        phase("done")
        rec["status"] = "pass"
        code = 0
    except BaseException as exc:  # noqa: BLE001  (every way a rank ends is recorded)
        rec.update(status="fail", error=_error(exc))
    finally:
        rec["finished_at"] = _utc()
        _write(path, rec)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)  # no interpreter teardown: NCCL's threads can hang it


# ── the launcher ─────────────────────────────────────────────────────────────────────────────────────────────────


def _engine_config(argv: list[str]):
    """The engine's configuration from the served command line, as `vllm serve` builds it."""
    import vllm  # noqa: PLC0415
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args  # noqa: PLC0415
    from vllm.usage.usage_lib import UsageContext  # noqa: PLC0415
    from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: PLC0415

    args = make_arg_parser(FlexibleArgumentParser()).parse_args(argv)
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag
    validate_parsed_serve_args(args)
    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
    return engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(request: Path) -> int:
    _die_with_parent(os.getppid())  # the node's pytest process: a kill of the L0 group ends the launcher, then its ranks
    req = json.loads(request.read_text())
    out = Path(req["records"])
    node = out / "construct.json"
    rec: dict = {"node": NODE, "argv": req["argv"], "started_at": _utc(), "status": "running",
                 "phase": "configuration", "phases": {}}
    _write(node, rec)
    t0 = time.monotonic()
    try:
        vllm_config = _engine_config(req["argv"])
        pc, mc = vllm_config.parallel_config, vllm_config.model_config
        rec["layout"] = {"world_size": pc.world_size, "tensor_parallel_size": pc.tensor_parallel_size,
                         "pipeline_parallel_size": pc.pipeline_parallel_size, "model": mc.model,
                         "architectures": list(mc.architectures or []), "dtype": str(mc.dtype),
                         "quantization": mc.quantization}
    except BaseException as exc:  # noqa: BLE001
        rec.update(status="fail", error=_error(exc), finished_at=_utc())
        rec["phases"]["configuration"] = round(time.monotonic() - t0, 3)
        _write(node, rec)
        print("INFB_L0 " + json.dumps({"status": "fail", "phase": "configuration", "error": rec["error"]["type"]}),
              flush=True)
        return 1
    rec["phases"]["configuration"] = round(time.monotonic() - t0, 3)
    world, grace = int(vllm_config.parallel_config.world_size), float(req.get("grace_s", 30))
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    rec.update(phase="ranks", init_method=init_method)
    _write(node, rec)

    import multiprocessing as mp  # noqa: PLC0415
    from multiprocessing.connection import wait  # noqa: PLC0415

    ctx = mp.get_context("spawn")
    t1 = time.monotonic()
    procs = [ctx.Process(target=rank_main, name=f"infb-l0-rank{r}",
                         args=(vllm_config, r, world, init_method, str(out), os.getpid())) for r in range(world)]
    try:
        for p in procs:
            p.start()  # the configuration is pickled to each rank, as the engine hands it to its workers
    except BaseException as exc:  # noqa: BLE001
        for p in procs:
            if p.pid is not None:
                p.kill()
        rec.update(status="fail", error=_error(exc), finished_at=_utc())
        _write(node, rec)
        print("INFB_L0 " + json.dumps({"status": "fail", "phase": "ranks", "error": rec["error"]["type"]}), flush=True)
        return 1
    alive, failed_at, stopped = set(range(world)), None, []
    while alive:
        by_sentinel = {procs[r].sentinel: r for r in alive}
        for s in wait(list(by_sentinel), timeout=5):
            r = by_sentinel[s]
            procs[r].join()
            alive.discard(r)
            if procs[r].exitcode != 0 and failed_at is None:
                failed_at = time.monotonic()
        if alive and failed_at is not None and time.monotonic() - failed_at > grace:
            # a rank failed and the others have not ended within the grace: they wait on it in a collective
            stopped = sorted(alive)
            for r in stopped:
                procs[r].kill()
            for r in stopped:
                procs[r].join(30)
            alive.clear()
    rec["phases"]["ranks"] = round(time.monotonic() - t1, 3)

    ranks = []
    for r, p in enumerate(procs):
        try:
            rr = json.loads((out / f"rank{r}.json").read_text())
        except (OSError, ValueError):
            rr = {"rank": r, "status": "fail", "phase": "start", "error": {"type": "no record", "message": ""}}
        if r in stopped:
            rr.update(status="stopped", why=f"killed {grace:.0f} s after another rank failed")
        elif p.exitcode != 0 and rr.get("status") != "fail":
            rr.update(status="fail", error={"type": "exit", "message": f"exit code {p.exitcode} in phase {rr.get('phase')}"})
        rr["exitcode"] = p.exitcode
        ranks.append({k: rr.get(k) for k in ("rank", "status", "phase", "phases", "device", "model", "weights",
                                             "host_rss", "error", "why", "exitcode")})
    ok = all(r["status"] == "pass" for r in ranks)
    first = next((r for r in ranks if r["status"] == "fail"), None)
    peaks = [(r.get("host_rss") or {}).get("peak_rss_bytes") for r in ranks]
    peaks = [p for p in peaks if p is not None]
    rec.update(status="pass" if ok else "fail", phase="done", ranks=ranks, finished_at=_utc(),
               peak_rss_bytes_max=max(peaks) if peaks else None,
               seconds=round(time.monotonic() - t0, 3))
    _write(node, rec)
    line = {"status": rec["status"], "ranks": [r["status"] for r in ranks], "seconds": rec["seconds"],
            "peak_rss_bytes_max": rec["peak_rss_bytes_max"]}
    if first is not None:
        line.update(first_failure={"rank": first["rank"], "phase": first["phase"],
                                   "error": (first.get("error") or {}).get("type")})
    print("INFB_L0 " + json.dumps(line), flush=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("launch")
    sp.add_argument("--request", required=True, type=Path)
    a = ap.parse_args(argv)
    return launch(a.request)


if __name__ == "__main__":
    sys.exit(main())
