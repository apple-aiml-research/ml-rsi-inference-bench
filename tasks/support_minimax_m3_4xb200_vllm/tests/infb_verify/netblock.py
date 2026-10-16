#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The verifier blocks the network itself, on every route: no test passes or fails by reaching the outside.

  unshare -r -n python3 -I /tests/infb_verify/netblock.py -- <command> [args...]
      inside a fresh user and network namespace: bring loopback up (the user namespace grants CAP_NET_ADMIN over
      its own network namespace), prove the outside unreachable, then exec the command in place of this process
  python3 -I /tests/infb_verify/netblock.py --probe
      where the runtime refuses the namespace: exit 0 only if the outside is already unreachable (the backend's own
      block, e.g. Docker's network_mode none), 20 otherwise

"Unreachable" means no TCP connection opens to a public address literal, to any address the probe hosts resolve to,
or to a proxy the environment names. Each invocation appends one line to /logs/verifier/network_block.jsonl.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

LOG = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier")) / "network_block.jsonl"
PROBE_HOSTS = ("huggingface.co", "openaipublic.blob.core.windows.net", "pypi.org", "github.com")
PROBE_ADDRESSES = (("1.1.1.1", 443), ("8.8.8.8", 53), ("2606:4700:4700::1111", 443))
PROXY_VARIABLES = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
SIOCGIFFLAGS, SIOCSIFFLAGS, IFF_UP = 0x8913, 0x8914, 0x1
UNREACHABLE_EXIT = 20


def loopback_up() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        flags = struct.unpack("16sH", fcntl.ioctl(s, SIOCGIFFLAGS, struct.pack("16sH", b"lo", 0))[:18])[1]
        fcntl.ioctl(s, SIOCSIFFLAGS, struct.pack("16sH", b"lo", flags | IFF_UP))


def connects(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def reachable() -> list[str]:
    """Every outside endpoint that accepted a connection."""
    targets = list(PROBE_ADDRESSES)
    for host in PROBE_HOSTS:
        try:
            targets += sorted({(info[4][0], 443) for info in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
        except OSError:
            pass
    for name in PROXY_VARIABLES:
        value = os.environ.get(name, "")
        url = urlsplit(value if "://" in value else "//" + value)
        if url.hostname:
            targets.append((url.hostname, url.port or 80))
    return [f"{host}:{port}" for host, port in dict.fromkeys(targets) if connects(host, port)]


def record(mode: str, open_to: list[str], command: list[str]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps({"t": time.time(), "mode": mode, "reachable": open_to, "command": command[:4]}) + "\n")


def main(argv: list[str]) -> int:
    if argv == ["--probe"]:
        open_to = reachable()
        record("backend", open_to, [])
        return UNREACHABLE_EXIT if open_to else 0
    if not argv or argv[0] != "--" or len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    loopback_up()
    open_to = reachable()
    record("namespace", open_to, argv[1:])
    if open_to:
        print(f"netblock: the outside is reachable from the namespace: {open_to}", file=sys.stderr)
        return UNREACHABLE_EXIT
    os.execvp(argv[1], argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
