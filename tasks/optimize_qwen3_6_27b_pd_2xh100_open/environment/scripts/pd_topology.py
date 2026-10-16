#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Bring up a prefill/decode topology, or an aggregated control, and tear the whole thing down."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PREFILL_HTTP_BASE = 20001
DECODE_HTTP_BASE = 20011
PREFILL_KV_BASE = 21001
DECODE_KV_BASE = 22001

PREFILL_NIXL_SIDE_CHANNEL_BASE = 25001
DECODE_NIXL_SIDE_CHANNEL_BASE = 25101

PRODUCER_BUFFER = "1e1"
CONSUMER_BUFFER = "3e10"

PRODUCER_GPU_UTIL = "0.90"
CONSUMER_GPU_UTIL = "0.70"
CONSUMER_NEEDS_NO_HEADROOM = frozenset({"nixl"})


def consumer_gpu_util(protocol: str = "p2pnccl", declared: str | None = None) -> str:
    """How much of the card the CONSUMER pool's engine may take, keyed by the transport that measured it."""
    default = PRODUCER_GPU_UTIL if protocol in CONSUMER_NEEDS_NO_HEADROOM else CONSUMER_GPU_UTIL
    if declared is not None:
        value = float(declared)
        if not 0.0 < value <= 0.98:
            raise SystemExit(
                f"FATAL: the task declares consumer_gpu_util={declared!r}, outside (0, 0.98]. This is the "
                f"engine's whole static budget on the consumer pool; 1.0 leaves nothing for activations."
            )
        print(f"[pd-topology] consumer gpu utilization {default} -> {value} (declared by the task)",
              flush=True)
        default = str(value)
    override = os.environ.get("INFB_PD_CONSUMER_GPU_UTIL", "").strip()
    if not override:
        return default
    try:
        value = float(override)
    except ValueError:
        raise SystemExit(
            f"FATAL: INFB_PD_CONSUMER_GPU_UTIL={override!r} is not a number. It is a GPU memory fraction "
            f"passed straight to the engine, and a non-numeric value would reach the argv."
        ) from None
    if not 0.0 < value <= 0.98:
        raise SystemExit(
            f"FATAL: INFB_PD_CONSUMER_GPU_UTIL={value} is outside (0, 0.98]. The consumer pool needs SOME "
            f"headroom for the connector's receive buffers; a CUDA OOM there is a silent recompute, not an "
            f"error, so an over-large value buys a plausible number for a different experiment."
        )
    print(f"[pd-topology] consumer gpu utilization overridden {default} -> {value} "
          f"(INFB_PD_CONSUMER_GPU_UTIL). The producer stays at {PRODUCER_GPU_UTIL}.", flush=True)
    return str(value)


REPLY_TOKENS_BOUNDS = (64, 16384)


def apply_reply_tokens_override(table: dict) -> dict:
    """Return the PD table with `reply_tokens` overridden by INFB_PD_REPLY_LEN, or unchanged."""
    raw = os.environ.get("INFB_PD_REPLY_LEN", "").strip()
    if not raw:
        return table
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(
            f"FATAL: INFB_PD_REPLY_LEN={raw!r} is not an integer. It becomes the bench client's "
            f"`max_tokens` and the input to the derived max_model_len, so a non-integer would reach "
            f"both the request payload and the engine argv."
        ) from None
    low, high = REPLY_TOKENS_BOUNDS
    if not low <= value <= high:
        raise SystemExit(
            f"FATAL: INFB_PD_REPLY_LEN={value} is outside [{low}, {high}]. Below {low} the pooled "
            f"P99 ITL has too few gaps per request to be a tail statistic; above {high} the derived "
            f"max_model_len leaves the KV budget these pools were measured at."
        )
    declared = table.get("reply_tokens")
    if value == declared:
        return table
    print(f"[pd-topology] reply_tokens overridden {declared} -> {value} (INFB_PD_REPLY_LEN). "
          f"This changes the DECODE share of every turn and therefore the derived max_model_len; it is "
          f"an experiment control and no task declares it.", flush=True)
    out = dict(table)
    out["reply_tokens"] = value
    return out


TURN_LEN_BOUNDS = (256, 24000)


def apply_turn_len_override(table: dict) -> dict:
    """Return the PD table with `new_tokens_per_turn` overridden by INFB_PD_TURN_LEN, or unchanged."""
    raw = os.environ.get("INFB_PD_TURN_LEN", "").strip()
    if not raw:
        return table
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(
            f"FATAL: INFB_PD_TURN_LEN={raw!r} is not an integer. It becomes the per-turn prefill length and "
            f"feeds the derived max_model_len, so a non-integer would reach both the corpus and the engine "
            f"argv."
        ) from None
    low, high = TURN_LEN_BOUNDS
    if not low <= value <= high:
        raise SystemExit(
            f"FATAL: INFB_PD_TURN_LEN={value} is outside [{low}, {high}]. Below {low} a turn adds less "
            f"prefill than the first-turn prefix and the multi-turn shape stops being one; above {high} the "
            f"derived max_model_len leaves the KV budget these pools were measured at."
        )
    declared = table.get("new_tokens_per_turn")
    if value == declared:
        return table
    print(f"[pd-topology] new_tokens_per_turn overridden {declared} -> {value} (INFB_PD_TURN_LEN). "
          f"This changes the PREFILL share of every turn and therefore the derived max_model_len; it is "
          f"an experiment control and no task declares it.", flush=True)
    out = dict(table)
    out["new_tokens_per_turn"] = value
    return out


def _apply_int_override(table: dict, key: str, env: str, bounds: tuple[int, int]) -> dict:
    """A generic integer experiment override on one pd-table key. Same contract as its two elder siblings."""
    raw = os.environ.get(env, "").strip()
    if not raw:
        return table
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(
            f"FATAL: {env}={raw!r} is not an integer. It replaces the pd table's {key!r}, which feeds the "
            f"derived max_model_len and the generated corpus, so a non-integer would reach the engine argv."
        ) from None
    low, high = bounds
    if not low <= value <= high:
        raise SystemExit(f"FATAL: {env}={value} is outside [{low}, {high}] for pd table key {key!r}.")
    if value == table.get(key):
        return table
    print(f"[pd-topology] {key} overridden {table.get(key)} -> {value} ({env}). This is an experiment "
          f"control and no task declares it.", flush=True)
    out = dict(table)
    out[key] = value
    return out


def conversation_tokens(table: dict) -> int:
    """The longest prompt a conversation reaches, in tokens."""
    turns = max(1, int(table.get("conversation_turns", 8)) // 2)
    return (int(table.get("first_turn_prefix_tokens", 8000))
            + turns * (int(table["new_tokens_per_turn"]) + int(table["reply_tokens"])))


TEMPLATE_RESERVE_TOKENS = 256


def assert_fits_context(table: dict) -> dict:
    """Refuse a workload the served context cannot hold. Returns the table so it can wrap a chain."""
    ceiling = table.get("context_window")
    if not ceiling:
        return table
    need = conversation_tokens(table)
    if need + TEMPLATE_RESERVE_TOKENS <= int(ceiling):
        return table
    turns = max(1, int(table.get("conversation_turns", 8)) // 2)
    prefix = int(table.get("first_turn_prefix_tokens", 8000))
    reply_cap = (int(ceiling) - TEMPLATE_RESERVE_TOKENS - prefix) // turns - int(table["new_tokens_per_turn"])
    raise SystemExit(
        f"FATAL: this PD workload needs {need:,} tokens per conversation but context_window is "
        f"{int(ceiling):,}.\n"
        f"  need = first_turn_prefix_tokens {prefix:,} + {turns} user turns x "
        f"(new_tokens_per_turn {int(table['new_tokens_per_turn']):,} + "
        f"reply_tokens {int(table['reply_tokens']):,})\n"
        f"  At this new_tokens_per_turn the largest servable reply_tokens is {reply_cap:,}.\n"
        f"  Lower INFB_PD_TURN_LEN to buy room for a larger INFB_PD_REPLY_LEN -- they move the same budget "
        f"and only their SUM is bounded. Left unrefused, the last turn of the longest conversations 400s, "
        f"those turns are silently dropped, and the aggregated control arm is not refused for it, so the "
        f"pair stops being comparable while both halves still report a number."
    )


def apply_workload_overrides(table: dict) -> dict:
    """THE production entry point: apply every experiment override, then refuse an unservable result."""
    out = apply_turn_len_override(apply_reply_tokens_override(table))
    out = _apply_int_override(out, "conversation_turns", "INFB_PD_TURNS", (2, 32))
    out = _apply_int_override(out, "context_window", "INFB_PD_CTX", (8192, 262144))
    out = _apply_int_override(out, "conversations", "INFB_PD_CONVERSATIONS", (2, 256))
    return assert_fits_context(out)


@dataclass
class Instance:
    role: str  # "kv_producer" | "kv_consumer" | "plain"
    gpu: str
    http_port: int
    kv_port: int
    log: Path
    proc: subprocess.Popen | None = None


@dataclass
class Topology:
    """A parsed `[metadata.inference_bench.pd]` table plus the devices actually visible to this run."""

    prefill_gpus: list[str]
    decode_gpus: list[str]
    router_port: int = 10001
    discovery_port: int = 30001
    connector: str = "P2pNcclConnector"
    engine: str = "vllm"
    transfer_backend: str = "mooncake"
    page_size: int | None = None
    attention_backend: str = "triton"
    aggregated: bool = False
    consumer_gpu_util: str | None = None
    consumer_max_mamba_cache_size: int | None = None
    instances: list[Instance] = field(default_factory=list)

    @classmethod
    def from_metadata(cls, meta: dict, visible_devices: str, *, aggregated: bool) -> "Topology":
        """Split the visible devices into pools by the COUNTS the task declared."""
        devices = [d.strip() for d in visible_devices.split(",") if d.strip()]
        want_p = int(meta.get("prefill_gpus", 0))
        want_d = int(meta.get("decode_gpus", 0))
        want_total = want_p + want_d
        if want_total and want_total > len(devices):
            raise SystemExit(
                f"FATAL: the task declares prefill_gpus={want_p} + decode_gpus={want_d} = {want_total} "
                f"but CUDA_VISIBLE_DEVICES has only {len(devices)} device(s): {visible_devices!r}."
            )
        if want_total and want_total < len(devices):
            if not bool(meta.get("narrower_than_pod", False)):
                raise SystemExit(
                    f"FATAL: the task declares prefill_gpus={want_p} + decode_gpus={want_d} = {want_total} "
                    f"but {len(devices)} device(s) are visible: {visible_devices!r}. Leaving GPUs idle is "
                    f"allowed, but it has to be deliberate -- declare `narrower_than_pod = true` in the pd "
                    f"table with the reason. Without it this is indistinguishable from a half-finished edit "
                    f"to one of the two pool counts."
                )
            print(f"[pd-topology] the task declares {want_p}P:{want_d}D = {want_total} device(s) and "
                  f"{len(devices)} are visible; using {devices[:want_total]} and leaving "
                  f"{devices[want_total:]} idle on BOTH arms", flush=True)
        if aggregated:
            return cls(prefill_gpus=[], decode_gpus=devices[:want_total] if want_total else devices,
                       router_port=int(meta.get("router_port", 10001)),
                       discovery_port=int(meta.get("discovery_port", 30001)),
                       page_size=(int(meta["page_size"]) if meta.get("page_size") is not None else None),
                       attention_backend=str(meta.get("attention_backend", "triton")),
                       connector=str(meta.get("connector", "P2pNcclConnector")),
                       engine=str(meta.get("engine", "vllm")),
                       transfer_backend=str(meta.get("transfer_backend", "mooncake")),
                       aggregated=True)
        if want_p < 1 or want_d < 1:
            raise SystemExit(f"FATAL: both pools must be non-empty; got {want_p}P:{want_d}D")
        return cls(prefill_gpus=devices[:want_p], decode_gpus=devices[want_p:want_total],
                   router_port=int(meta.get("router_port", 10001)),
                   discovery_port=int(meta.get("discovery_port", 30001)),
                       page_size=(int(meta["page_size"]) if meta.get("page_size") is not None else None),
                       attention_backend=str(meta.get("attention_backend", "triton")),
                   connector=str(meta.get("connector", "P2pNcclConnector")),
                   engine=str(meta.get("engine", "vllm")),
                   transfer_backend=str(meta.get("transfer_backend", "mooncake")),
                   consumer_gpu_util=(str(meta["consumer_gpu_util"])
                                      if meta.get("consumer_gpu_util") is not None else None),
                   consumer_max_mamba_cache_size=(int(meta["consumer_max_mamba_cache_size"])
                                                  if meta.get("consumer_max_mamba_cache_size") is not None
                                                  else None),
                   aggregated=False)

    def label(self) -> str:
        return ("aggregated:%dx" % len(self.decode_gpus)) if self.aggregated else \
            "%dP:%dD" % (len(self.prefill_gpus), len(self.decode_gpus))


def _listening(port: int, timeout: float = 2.0) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def _get(port: int, path: str, timeout: float = 20.0) -> str:
    """HTTP GET with NO proxy handler."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}{path}", timeout=timeout) as fh:
            return fh.read().decode("utf-8", "ignore")
    except Exception as exc:  # noqa: BLE001
        return f"__ERR__ {exc}"


SELF_REGISTERING_PROTOCOLS = frozenset({"p2pnccl"})

_PD_PROTOCOLS = {
    "P2pNcclConnector": "p2pnccl",
    "NixlConnector": "nixl",
    "NixlPushConnector": "nixl",
}


SGLANG_CACHE_MODES = ("off", "decode_radix", "hicache")


# Model families SGLang refuses --disaggregation-decode-enable-radix-cache on, matched against the
# resolved --model-path. Absence from this list means UNMEASURED, not known-good.
SGLANG_DECODE_RADIX_INCOMPATIBLE = ("Qwen3.6",)


def _sglang_cache_flags(mode: str, model_path: str = "") -> list[str]:
    """Extra SGLang PD cache flags for one role, selected by INFB_PD_SGLANG_CACHE. Empty unless set."""
    raw = os.environ.get("INFB_PD_SGLANG_CACHE", "").strip()
    if not raw or raw == "off":
        return []
    if raw not in SGLANG_CACHE_MODES:
        raise SystemExit(
            f"FATAL: INFB_PD_SGLANG_CACHE={raw!r} is not one of {SGLANG_CACHE_MODES}. It selects which of "
            f"SGLang's PD multi-turn cache features to enable; an unrecognised value would silently measure "
            f"the shipped configuration under an experiment's label."
        )
    blocked = [m for m in SGLANG_DECODE_RADIX_INCOMPATIBLE if m.lower() in (model_path or "").lower()]
    if blocked:
        raise SystemExit(
            f"FATAL: INFB_PD_SGLANG_CACHE={raw!r} needs "
            f"--disaggregation-decode-enable-radix-cache, and SGLang refuses that flag for {blocked[0]} — "
            f"it is a Mamba/SSM hybrid and the engine answers `ValueError: "
            f"--disaggregation-decode-enable-radix-cache is incompatible with Mamba/SSM models`. The "
            f"decode pool then dies at boot, after the whole topology has been brought up and the "
            f"weights loaded. Refusing here costs nothing."
        )
    if raw == "decode_radix":
        return ["--disaggregation-decode-enable-radix-cache"] if mode == "decode" else []
    if mode == "prefill":
        return ["--enable-hierarchical-cache", "--hicache-storage-backend", "file"]
    return ["--disaggregation-decode-enable-radix-cache",
            "--disaggregation-decode-enable-offload-kvcache",
            "--hicache-storage-backend", "file"]


def _model_path_of(engine_argv: list[str]) -> str:
    """The resolved `--model-path` from the argv `start()` was handed, or "" if absent."""
    argv = list(engine_argv or [])
    for flag in ("--model-path", "--model"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return str(argv[i + 1])
    return ""


def pd_protocol(topology: Topology) -> str:
    """The pairing protocol this deployment speaks."""
    if topology.engine == "sglang":
        return "sglang"
    protocol = _PD_PROTOCOLS.get(topology.connector)
    if protocol is None:
        raise SystemExit(
            f"FATAL: no PD pairing protocol is registered for connector {topology.connector!r}. Add it to "
            f"_PD_PROTOCOLS, read out of that connector's own scheduler, or the router will pair nothing "
            f"and the run will look healthy."
        )
    return protocol


def _kv_config(topology: Topology, role: str, inst: Instance) -> str:
    return json.dumps({
        "kv_connector": topology.connector,
        "kv_role": role,
        "kv_buffer_size": PRODUCER_BUFFER if role == "kv_producer" else CONSUMER_BUFFER,
        "kv_port": str(inst.kv_port),
        "kv_connector_extra_config": {
            "proxy_ip": "127.0.0.1",
            "proxy_port": str(topology.discovery_port),
            "http_port": str(inst.http_port),
            "send_type": "PUT_ASYNC",
            "nccl_num_channels": "16",
        },
    }, separators=(",", ":"))


def _engine_of(engine_argv: list[str]) -> str:
    joined = " ".join(engine_argv)
    if "sglang" in joined:
        return "sglang"
    return "vllm"


def _role_flags(topology: Topology, role: str, inst: Instance,
                engine_argv: list[str] | None = None) -> list[str]:
    """The per-role flags that make an instance a producer or a consumer, per engine."""
    if topology.engine == "sglang":
        mode = "prefill" if role == "kv_producer" else "decode"
        flags = [
            "--disaggregation-mode", mode,
            "--disaggregation-transfer-backend", topology.transfer_backend,
            "--disaggregation-bootstrap-port", str(inst.kv_port),
            "--attention-backend", topology.attention_backend,
        ]
        if topology.page_size is not None:
            flags += ["--page-size", str(topology.page_size)]
        flags += _sglang_cache_flags(mode, _model_path_of(engine_argv or []))
        return flags
    return ["--kv-transfer-config", _kv_config(topology, role, inst)]


def plan(topology: Topology, log_dir: Path) -> list[Instance]:
    """ONE instance per pool, tensor-parallel across that pool's devices."""
    out: list[Instance] = []
    if topology.aggregated:
        out.append(Instance("plain", ",".join(topology.decode_gpus), PREFILL_HTTP_BASE, 0,
                            log_dir / "plain_all.log"))
    else:
        out.append(Instance("kv_producer", ",".join(topology.prefill_gpus),
                            PREFILL_HTTP_BASE, PREFILL_KV_BASE, log_dir / "prefill_pool.log"))
        out.append(Instance("kv_consumer", ",".join(topology.decode_gpus),
                            DECODE_HTTP_BASE, DECODE_KV_BASE, log_dir / "decode_pool.log"))
    topology.instances = out
    return out


def tensor_parallel_size(topology: Topology) -> int:
    """The width each instance is launched at: the pool's device count."""
    if topology.aggregated:
        return len(topology.decode_gpus)
    widths = {len(topology.prefill_gpus), len(topology.decode_gpus)}
    if len(widths) != 1:
        raise SystemExit(
            f"FATAL: the two pools have different widths {sorted(widths)}. A single declared "
            f"tensor_parallel_size cannot describe both, so either the split is wrong or this family "
            f"needs a per-pool width in the task declaration."
        )
    return widths.pop()


def start(topology: Topology, *, engine_argv: list[str], venv: Path, router_module: Path,
          log_dir: Path, served_model_name: str, env: dict | None = None) -> subprocess.Popen:
    """Start the router and every engine. Returns the router process; instances hang off the topology."""
    log_dir.mkdir(parents=True, exist_ok=True)
    base_env = dict(os.environ if env is None else env)
    for key in ("no_proxy", "NO_PROXY"):
        base_env[key] = ",".join(filter(None, [base_env.get(key, ""), "127.0.0.1", "localhost", "0.0.0.0"]))

    router_argv = [str(venv / "bin" / "python"), str(router_module),
                   "--mode", "aggregated" if topology.aggregated else "pd",
                   "--protocol", pd_protocol(topology),
                   "--port", str(topology.router_port),
                   "--discovery-port", str(topology.discovery_port)]
    if topology.aggregated:
        for inst in plan(topology, log_dir):
            router_argv += ["--instance", f"127.0.0.1:{inst.http_port}"]
    else:
        instances = plan(topology, log_dir)
        if pd_protocol(topology) not in SELF_REGISTERING_PROTOCOLS:
            for inst in instances:
                if inst.role == "kv_producer":
                    router_argv += ["--pd-instance", f"P:127.0.0.1:{inst.http_port}:{inst.kv_port}"]
                elif inst.role == "kv_consumer":
                    router_argv += ["--pd-instance", f"D:127.0.0.1:{inst.http_port}"]

    router_log = (log_dir / "router.log").open("w")
    router = subprocess.Popen(router_argv, stdout=router_log, stderr=subprocess.STDOUT,
                              env=base_env, start_new_session=True)
    for _ in range(30):
        if _listening(topology.router_port):
            break
        time.sleep(1)
    if not _listening(topology.router_port):
        raise SystemExit(f"FATAL: the router never listened on {topology.router_port}; see {log_dir}/router.log")

    for inst in topology.instances:
        argv = list(engine_argv) + ["--port", str(inst.http_port)]
        if inst.role != "plain":
            argv += _role_flags(topology, inst.role, inst, engine_argv)
        mem_flag = "--mem-fraction-static" if topology.engine == "sglang" else "--gpu-memory-utilization"
        if mem_flag not in argv:
            argv += [mem_flag,
                     consumer_gpu_util(pd_protocol(topology), topology.consumer_gpu_util)
                     if inst.role == "kv_consumer" else PRODUCER_GPU_UTIL]
        if (inst.role == "kv_consumer" and topology.engine == "sglang"
                and topology.consumer_max_mamba_cache_size is not None
                and "--max-mamba-cache-size" not in argv):
            argv += ["--max-mamba-cache-size", str(topology.consumer_max_mamba_cache_size)]
        inst_env = dict(base_env)
        inst_env["CUDA_VISIBLE_DEVICES"] = inst.gpu
        if topology.engine == "vllm" and inst.role != "plain":
            inst_env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(
                PREFILL_NIXL_SIDE_CHANNEL_BASE if inst.role == "kv_producer"
                else DECODE_NIXL_SIDE_CHANNEL_BASE
            )
        if topology.engine == "sglang" and inst.role != "plain":
            inst_env.setdefault("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "INTRA_NODE_NVLINK")
        handle = inst.log.open("w")
        inst.proc = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT,
                                     env=inst_env, start_new_session=True)
    return router


def wait_ready(topology: Topology, served_model_name: str, *, timeout_s: float = 1800.0) -> None:
    """Every instance must SERVE THE MODEL, not merely accept a connection."""
    deadline = time.time() + timeout_s
    for inst in topology.instances:
        while True:
            if inst.proc is not None and inst.proc.poll() is not None:
                raise SystemExit(
                    f"FATAL: the {inst.role} instance on GPU {inst.gpu} (port {inst.http_port}) exited "
                    f"with rc={inst.proc.returncode} before serving. See {inst.log}."
                )
            if _listening(inst.http_port) and served_model_name in _get(inst.http_port, "/v1/models"):
                break
            if time.time() > deadline:
                raise SystemExit(
                    f"FATAL: the {inst.role} instance on GPU {inst.gpu} (port {inst.http_port}) never "
                    f"served {served_model_name!r} within {timeout_s:.0f}s. See {inst.log}."
                )
            time.sleep(5)
    _wait_router_ready(topology, deadline=deadline, timeout_s=timeout_s)


def _router_health(topology: Topology) -> tuple[bool, str]:
    """(is the router ready, what it actually said)."""
    raw = _get(topology.router_port, "/health")
    if raw.startswith("__ERR__"):
        return False, raw[:200]
    try:
        parsed = json.loads(raw)
    except ValueError:
        return True, "non-JSON 200 body"
    if isinstance(parsed, dict) and "ready" in parsed:
        return bool(parsed["ready"]), str(parsed)[:200]
    return True, str(parsed)[:200]


def _wait_router_ready(topology: Topology, *, deadline: float, timeout_s: float) -> None:
    """Poll the router to the SAME deadline the instances got."""
    while True:
        ready, detail = _router_health(topology)
        if ready:
            return
        if time.time() > deadline:
            raise SystemExit(
                f"FATAL: every engine serves but the router on port {topology.router_port} never became "
                f"ready within {timeout_s:.0f}s. Last answer: {detail}"
            )
        time.sleep(5)


def router_stats(topology: Topology) -> dict:
    raw = _get(topology.router_port, "/stats")
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return {"__unparsed__": raw[:200]}


_VLLM_TRANSPORT_MARKERS: dict[str, tuple[str, ...]] = {
    "P2pNcclConnector": ("ncclCommInitRank Success",),
    "NixlConnector": ("NIXL compatibility check passed",),
    "NixlPushConnector": ("NIXL compatibility check passed",),
}


def _vllm_transport_opened(topology: Topology) -> bool:
    """Did either pool log its connector's channel-open line?"""
    markers = _VLLM_TRANSPORT_MARKERS.get(topology.connector)
    if not markers:
        raise SystemExit(
            f"FATAL: no channel-open marker is registered for connector {topology.connector!r}, so "
            f"whether any KV moved cannot be established. Add one to _VLLM_TRANSPORT_MARKERS, read out "
            f"of that connector's own logger.info calls."
        )
    for inst in topology.instances:
        if not inst.log.is_file():
            continue
        text = inst.log.read_text(encoding="utf-8", errors="ignore")
        if any(marker in text for marker in markers):
            return True
    return False



# declared transport -> (markers that prove it ran, markers that prove it degraded)
_TRANSPORT_EVIDENCE: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "nixl": (("Backend UCX", "NIXL INFO"), ("Using TCP transport",)),
    "mooncake": (("Transfer Engine", "Mooncake"), ("Using TCP transport",)),
}


def _assert_transport_is_the_declared_one(topology: "Topology", inst: "Instance", text: str) -> None:
    """A run may only be scored on the wire its task declared."""
    if topology.engine != "sglang" or inst.role == "plain":
        return
    declared = (topology.transfer_backend or "").strip().lower()
    if declared not in _TRANSPORT_EVIDENCE:
        raise SystemExit(
            f"FATAL: the task declares transfer_backend={declared!r}, which has no entry in "
            f"_TRANSPORT_EVIDENCE. Add one with the markers that prove it ran and the markers that prove "
            f"it degraded -- an unverified transport must not score, because a silent fallback to a slower "
            f"wire is exactly what this check exists to catch."
        )
    ran, degraded = _TRANSPORT_EVIDENCE[declared]
    for marker in degraded:
        if marker in text:
            raise SystemExit(
                f"FATAL: the task declares transfer_backend={declared!r} but the instance on GPU "
                f"{inst.gpu} logged {marker!r}, so the transport DEGRADED. The run would still produce a "
                f"score, and that score would measure the fallback wire rather than the engine, which is "
                f"easy to miss without this check. See {inst.log}."
            )
    if not any(marker in text for marker in ran):
        raise SystemExit(
            f"FATAL: the task declares transfer_backend={declared!r} but the instance on GPU {inst.gpu} "
            f"shows none of {ran!r} in its log, so there is no evidence the declared transport is the one "
            f"that ran. See {inst.log}."
        )

def assert_pd_happened(topology: Topology) -> None:
    """Refuse a run that looks like PD and was not."""
    if topology.aggregated:
        return
    stats = router_stats(topology)
    if not stats.get("requests"):
        raise SystemExit(f"FATAL: the router served 0 requests, so nothing was disaggregated: {stats}")
    if stats.get("prefill_errors"):
        raise SystemExit(f"FATAL: {stats['prefill_errors']} prefill hop(s) failed: {stats}")
    for inst in topology.instances:
        text = inst.log.read_text(encoding="utf-8", errors="ignore") if inst.log.is_file() else ""
        if topology.engine == "sglang":
            if inst.role == "kv_producer" and "disaggregation_mode='prefill'" not in text:
                raise SystemExit(
                    f"FATAL: the producer on GPU {inst.gpu} was not running in prefill mode -- its "
                    f"server_args echo does not say disaggregation_mode='prefill', so the role never "
                    f"applied and this is an aggregated run wearing a PD label. See {inst.log}."
                )
            if "Mooncake Transfer Engine initialization failed." in text:
                raise SystemExit(
                    f"FATAL: the instance on GPU {inst.gpu} logged `Mooncake Transfer Engine "
                    f"initialization failed.`, so no KV could move over it. See {inst.log}."
                )
        elif inst.role == "kv_producer" and not _vllm_transport_opened(topology):
            raise SystemExit(
                f"FATAL: no {topology.connector} channel between the pools ever opened. The run served "
                f"requests without moving any KV, which is an aggregated run wearing a PD label. Looked "
                f"for {_VLLM_TRANSPORT_MARKERS.get(topology.connector, ())!r} in the pool logs. "
                f"See {inst.log}."
            )
        if inst.role == "kv_consumer" and "kv_cache is None" in text:
            raise SystemExit(
                f"FATAL: the consumer on GPU {inst.gpu} logged `kv_cache is None`, i.e. it recomputed "
                f"instead of receiving. Any latency measured here is not a PD latency. See {inst.log}."
            )
        _assert_transport_is_the_declared_one(topology, inst, text)


def stop(topology: Topology, router: subprocess.Popen | None) -> None:
    """Kill every process GROUP. Never a name pattern."""
    procs = [inst.proc for inst in topology.instances] + [router]
    for proc in procs:
        if proc is None or proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.time() + 30
    while time.time() < deadline and any(p is not None and p.poll() is None for p in procs):
        time.sleep(1)
    for proc in procs:
        if proc is None or proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
