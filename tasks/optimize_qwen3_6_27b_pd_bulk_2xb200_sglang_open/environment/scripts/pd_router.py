#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""A PD router that both arms of the comparison traverse, so the only difference is the P/D split.

WHY NOT UPSTREAM'S PROXY. Four reasons, all read out of its 190 lines rather than guessed:

  1. it routes round-robin on a global counter, with NO conversation affinity. At 4P:4D, turn 2 of a
     conversation lands on a different decode instance than turn 1, so the prefix cache misses every
     turn -- which deletes the premise that makes continuation-turn p95 the right metric at all;
  2. only the PD arm would have a proxy hop. Comparing "aggregated straight to vllm serve" against
     "PD through a single-process Quart dev server" charges PD for the instrument;
  3. it needs `quart`, which these PD environments do not pin. This one uses
     fastapi/uvicorn/aiohttp/zmq/msgspec, all already pinned wherever a PD task runs, so the router
     costs zero further dependencies.

     THE DECODER IS `msgspec`, NOT `msgpack`, and that is load-bearing rather than taste. `msgpack` is
     not a guaranteed dependency of either engine: some PD environments pull it in only transitively
     (through `mooncake-transfer-engine`), and a NIXL-based one does not need it at all -- so a router
     that imports it dies at import with `ModuleNotFoundError: No module named 'msgpack'` on exactly
     those. Adding it to the environment would be the wrong repair: it would change the venv under
     measurement, and the only intended difference between a PD environment and its parent is the
     connector. `msgspec` is a hard vLLM dependency, so it is already present wherever a PD task runs,
     and `msgspec.msgpack.decode` reads the same wire bytes the engines send;
  4. its `forward_request` yields only on HTTP 200 and its `except` falls off the end returning None, so
     a failed prefill can surface as an empty 500 rather than a counted error.

WHAT IT TAKES VERBATIM FROM UPSTREAM, because it is protocol and not policy:
  * the ZMQ ROUTER service-discovery message `{"type": "P"|"D", "http_address": .., "zmq_address": ..}`,
    msgpack-encoded, which every engine instance pings at `proxy_ip:proxy_port`;
  * the request-id convention the connector parses to decide who sends and who receives:
    `___prefill_addr_<prefill_zmq>___decode_addr_<decode_zmq>_<uuid>`;
  * the sequence: the prompt goes to a prefill instance with max_tokens=1, then the same prompt goes to a
    decode instance, which pulls the KV rather than recomputing it.

AFFINITY KEY. The bench client speaks plain OpenAI chat-completions and sends no conversation id, so the
router derives one: the conversation's unique long first user message IS its identity, and it is present
in every turn's message list. Hashing a fixed slice of `messages[0]["content"]` gives a key that is stable
across turns of one conversation and distinct between conversations, with no client change. In aggregated
mode the same key picks the instance, so both arms have identical locality.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import socket
import threading
import time
import uuid
from typing import Any

import aiohttp
import msgspec.msgpack
import uvicorn
import zmq
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("pd_router")

# One process, so plain dicts under a lock are enough. Values are (zmq_address, expires_at).
PREFILL: dict[str, tuple[str, float]] = {}
DECODE: dict[str, tuple[str, float]] = {}
_LOCK = threading.Lock()
PING_TTL = 30.0  # upstream uses 5s; a 30s TTL survives a GC pause without evicting a live instance

STATS: dict[str, int] = {"requests": 0, "prefill_errors": 0, "decode_errors": 0, "router_errors": 0}


def _discovery(host: str, port: int) -> threading.Thread:
    """Listen for the engines' registration pings. Same wire format upstream defines."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(f"tcp://{host}:{port}")
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    def loop() -> None:
        while True:
            if sock not in dict(poller.poll()):
                continue
            _addr, message = sock.recv_multipart()
            data = msgspec.msgpack.decode(message)
            table = PREFILL if data.get("type") == "P" else DECODE if data.get("type") == "D" else None
            if table is None:
                log.warning("unexpected registration: %s", data)
                continue
            with _LOCK:
                first = data["http_address"] not in table
                table[data["http_address"]] = (data["zmq_address"], time.time() + PING_TTL)
            if first:
                log.info("registered %s %s (zmq %s)", data["type"], data["http_address"],
                         data["zmq_address"])

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def _live(table: dict[str, tuple[str, float]]) -> list[tuple[str, str]]:
    now = time.time()
    with _LOCK:
        return sorted((http, zmq_addr) for http, (zmq_addr, exp) in table.items() if exp > now)


# The first 4 KB of the first user message. Long enough to be unique per conversation (each carries its
# own multi-thousand-token prefix) and cheap to hash on every turn.
_KEY_BYTES = 4096


def _conversation_key(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    if messages and isinstance(messages[0], dict):
        head = str(messages[0].get("content", ""))[:_KEY_BYTES]
    else:
        head = str(payload.get("prompt", ""))[:_KEY_BYTES]
    return hashlib.sha256(head.encode("utf-8", "ignore")).hexdigest()


def _pick(instances: list[tuple[str, str]], key: str) -> tuple[str, str]:
    """Sticky by conversation. Same key -> same instance, for as long as the instance list is stable."""
    return instances[int(key[:8], 16) % len(instances)]


def build_app(mode: str, timeout_s: float, protocol: str = "p2pnccl") -> FastAPI:
    app = FastAPI()
    client_timeout = aiohttp.ClientTimeout(total=timeout_s)

    @app.get("/health")
    async def health() -> JSONResponse:
        p, d = _live(PREFILL), _live(DECODE)
        ready = bool(d) if mode == "aggregated" else bool(p and d)
        return JSONResponse({"mode": mode, "ready": ready, "prefill": [a for a, _ in p],
                             "decode": [a for a, _ in d], "stats": STATS},
                            status_code=200 if ready else 503)

    @app.get("/stats")
    async def stats() -> JSONResponse:
        return JSONResponse(dict(STATS))

    async def _pump(session: aiohttp.ClientSession, url: str, payload: dict, rid: str):
        async with session.post(url, json=payload, headers={"X-Request-Id": rid}) as resp:
            if resp.status != 200:
                body = (await resp.read())[:400].decode("utf-8", "ignore")
                STATS["decode_errors"] += 1
                log.error("decode %s returned HTTP %s: %s", url, resp.status, body)
                yield f"data: {json.dumps({'error': f'decode HTTP {resp.status}', 'body': body})}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return
            buf = b""
            events = 0
            last = b""
            # Forward WHOLE SSE events, never fixed-size chunks. iter_chunked(8192) re-cuts the
            # stream at byte boundaries, splitting `data: {...}\n\n` events in half and making the
            # client under-count output tokens -- a router artifact masquerading as a property of
            # disaggregation.
            async for chunk in resp.content.iter_any():
                buf += chunk
                while True:
                    idx = buf.find(b"\n\n")
                    if idx < 0:
                        break
                    event = buf[: idx + 2]
                    events += 1
                    last = event
                    yield event
                    buf = buf[idx + 2:]
            if buf:
                events += 1
                last = buf
                yield buf
            reason = "?"
            for token in (b'"finish_reason":"', b'"finish_reason": "'):
                i = last.rfind(token)
                if i >= 0:
                    reason = last[i + len(token):].split(b'"')[0].decode("utf-8", "ignore")
                    break
            log.info("stream end events=%d finish_reason=%s url=%s", events, reason, url)

    @app.post("/v1/completions")
    @app.post("/v1/chat/completions")
    async def handle(request: Request):
        path = request.url.path
        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001
            STATS["router_errors"] += 1
            return JSONResponse({"error": f"bad request json: {exc}"}, status_code=400)

        key = _conversation_key(payload)
        decode = _live(DECODE)
        if not decode:
            STATS["router_errors"] += 1
            return JSONResponse({"error": "no decode instance registered"}, status_code=503)
        decode_http, decode_zmq = _pick(decode, key)
        STATS["requests"] += 1

        session = aiohttp.ClientSession(timeout=client_timeout)

        if mode == "aggregated":
            rid = uuid.uuid4().hex

            async def stream_agg():
                try:
                    async for chunk in _pump(session, f"http://{decode_http}{path}", payload, rid):
                        yield chunk
                finally:
                    await session.close()

            return StreamingResponse(stream_agg(), media_type="text/event-stream")

        prefill = _live(PREFILL)
        if not prefill:
            await session.close()
            STATS["router_errors"] += 1
            return JSONResponse({"error": "no prefill instance registered"}, status_code=503)
        prefill_http, prefill_zmq = _pick(prefill, key)

        if protocol == "sglang":
            bootstrap_room = random.getrandbits(63)
            body = dict(payload)
            body["bootstrap_host"] = prefill_http.rsplit(":", 1)[0]
            body["bootstrap_port"] = int(prefill_zmq) if str(prefill_zmq).isdigit() else prefill_zmq
            body["bootstrap_room"] = bootstrap_room
            rid = f"pd-{bootstrap_room:016x}"

            async def _prefill_leg():
                try:
                    async with session.post(f"http://{prefill_http}{path}", json=body,
                                            headers={"X-Request-Id": rid}) as resp:
                        text = (await resp.read())[:400].decode("utf-8", "ignore")
                        if resp.status != 200:
                            STATS["prefill_errors"] += 1
                            log.error("prefill %s returned HTTP %s: %s", prefill_http, resp.status, text)
                        return resp.status
                except Exception as exc:  # noqa: BLE001
                    STATS["prefill_errors"] += 1
                    log.error("prefill %s failed: %s", prefill_http, exc)
                    return 0

            async def stream_sglang():
                leg = asyncio.ensure_future(_prefill_leg())
                try:
                    async for chunk in _pump(session, f"http://{decode_http}{path}", body, rid):
                        yield chunk
                    try:
                        status = await leg
                    except Exception:  # noqa: BLE001
                        status = 0
                    if status != 200:
                        STATS["router_errors"] += 1
                        log.error("prefill leg for %s ended %s; the decode pool served this request "
                                  "without a paired prefill", rid, status)
                        yield (f"data: {json.dumps({'error': 'prefill leg failed', 'status': status})}"
                               f"\n\n").encode()
                        yield b"data: [DONE]\n\n"
                finally:
                    if not leg.done():
                        leg.cancel()
                    await session.close()

            return StreamingResponse(stream_sglang(), media_type="text/event-stream")

        if protocol == "nixl":
            pre = dict(payload)
            pre["max_tokens"] = 1
            if "max_completion_tokens" in pre:
                pre["max_completion_tokens"] = 1
            pre.pop("min_tokens", None)
            pre.pop("stream_options", None)
            pre["stream"] = False
            pre["kv_transfer_params"] = {"do_remote_decode": True}
            rid = uuid.uuid4().hex
            try:
                async with session.post(f"http://{prefill_http}{path}", json=pre,
                                        headers={"X-Request-Id": rid}) as resp:
                    raw = await resp.read()
                    if resp.status != 200:
                        STATS["prefill_errors"] += 1
                        await session.close()
                        return JSONResponse(
                            {"error": f"prefill HTTP {resp.status}",
                             "body": raw[:400].decode("utf-8", "ignore")}, status_code=502)
            except Exception as exc:  # noqa: BLE001
                STATS["prefill_errors"] += 1
                await session.close()
                return JSONResponse({"error": f"prefill failed: {exc}"}, status_code=502)

            try:
                xfer = (json.loads(raw) or {}).get("kv_transfer_params")
            except Exception as exc:  # noqa: BLE001
                xfer = None
                log.error("prefill response was not JSON: %s", exc)
            if not xfer:
                STATS["prefill_errors"] += 1
                await session.close()
                return JSONResponse(
                    {"error": "prefill returned no kv_transfer_params; the decode side would silently "
                              "recompute instead of pulling KV",
                     "body": raw[:400].decode("utf-8", "ignore")}, status_code=502)

            dec = dict(payload)
            dec["kv_transfer_params"] = xfer

            async def stream_nixl():
                try:
                    async for chunk in _pump(session, f"http://{decode_http}{path}", dec, rid):
                        yield chunk
                finally:
                    await session.close()

            return StreamingResponse(stream_nixl(), media_type="text/event-stream")

        rid = f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_{uuid.uuid4().hex}"

        pre = dict(payload)
        pre["max_tokens"] = 1
        if "max_completion_tokens" in pre:
            pre["max_completion_tokens"] = 1
        # Three fields must be POPPED or the prefill server answers 400: `min_tokens`, which the
        # client sets to the target reply length and which is a Bad Request when it exceeds
        # max_tokens (set to 1 just above); `stream_options`, illegal without stream=true; and
        # `stream` itself.
        pre.pop("min_tokens", None)
        pre.pop("stream_options", None)
        pre["stream"] = False
        try:
            async with session.post(f"http://{prefill_http}{path}", json=pre,
                                    headers={"X-Request-Id": rid}) as resp:
                body = await resp.read()
                if resp.status != 200:
                    STATS["prefill_errors"] += 1
                    await session.close()
                    return JSONResponse(
                        {"error": f"prefill HTTP {resp.status}",
                         "body": body[:400].decode("utf-8", "ignore")}, status_code=502)
        except Exception as exc:  # noqa: BLE001
            STATS["prefill_errors"] += 1
            await session.close()
            return JSONResponse({"error": f"prefill failed: {exc}"}, status_code=502)

        async def stream_pd():
            try:
                async for chunk in _pump(session, f"http://{decode_http}{path}", payload, rid):
                    yield chunk
            finally:
                await session.close()

        return StreamingResponse(stream_pd(), media_type="text/event-stream")

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("pd", "aggregated"), required=True)
    ap.add_argument("--instance", action="append", default=[], metavar="HOST:PORT",
                    help="aggregated mode: an ordinary replica, given statically. The control arm must "
                         "carry NO kv-transfer-config: giving its replicas kv_role=kv_consumer "
                         "purely so they register sends every request into start_load_kv, where "
                         "parse_request_id raises on a plain uuid and the engine core dies with "
                         "EngineDeadError. A control that runs the connector is not a control.")
    ap.add_argument("--port", type=int, default=10001, help="the OpenAI-compatible port the client uses")
    ap.add_argument("--discovery-port", type=int, default=30001,
                    help="ZMQ ROUTER port the engines ping (their kv_connector_extra_config.proxy_port)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--request-timeout", type=float, default=6 * 3600)
    ap.add_argument("--protocol", choices=("p2pnccl", "nixl", "sglang"), default="p2pnccl",
                    help="how the two hops are paired, which is a property of the CONNECTOR and not of the "
                         "engine. p2pnccl encodes the peer addresses in the request id; nixl round-trips "
                         "`kv_transfer_params` from the prefill response into the decode request; sglang "
                         "sends a shared bootstrap room to both servers at once. Two of the three are vLLM "
                         "connectors, which is why keying on the engine was wrong.")
    ap.add_argument("--pd-instance", action="append", default=[], metavar="ROLE:HOST:PORT[:BOOTSTRAP]",
                    help="PD mode: an instance given STATICALLY, for an engine that does not self-register. "
                         "vLLM's P2pNcclConnector pings the discovery port because its "
                         "kv_connector_extra_config carries proxy_ip/proxy_port; SGLang has no equivalent "
                         "-- its flags are --disaggregation-{mode,transfer-backend,bootstrap-port} and none "
                         "of them names a proxy -- so nothing ever registered and the router stayed 503 "
                         "until its readiness deadline. ROLE is P or D; BOOTSTRAP is the prefill "
                         "instance's --disaggregation-bootstrap-port.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[router] %(message)s")
    if args.mode == "aggregated" and args.instance:
        for addr in args.instance:
            DECODE[addr] = ("", float("inf"))
            log.info("static instance %s", addr)
    elif args.pd_instance:
        for spec in args.pd_instance:
            role, _, rest = spec.partition(":")
            host, _, tail = rest.partition(":")
            port, _, bootstrap = tail.partition(":")
            addr = f"{host}:{port}"
            table = PREFILL if role.upper().startswith("P") else DECODE
            table[addr] = (bootstrap, float("inf"))
            log.info("static %s instance %s (bootstrap %s)", role.upper(), addr, bootstrap or "-")
        _discovery(args.host, args.discovery_port)
    else:
        _discovery(args.host, args.discovery_port)
    log.info("mode=%s http=%s:%s discovery=%s:%s hostname=%s",
             args.mode, args.host, args.port, args.host, args.discovery_port, socket.gethostname())
    uvicorn.run(build_app(args.mode, args.request_timeout, args.protocol), host=args.host, port=args.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
