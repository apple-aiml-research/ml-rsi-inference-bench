#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The L3 step's offline condition.

The task declares `allow_internet = false`; the verifier imposes it on the pass-to-pass step itself, on every route:
test.sh runs the step's processes with the proxy variables removed and this directory first on PYTHONPATH, so every
Python process of the step -- the pytest processes and the servers they launch, which inherit PYTHONPATH -- loads this
module at start-up. It lets an internet-family socket connect to a loopback address only; any other connection
fails as it would with the network blocked. Native connections made below Python (NCCL, gloo, TCPStore) are not
affected. The system's own sitecustomize, which this module shadows, is run after it.
"""
import ipaddress
import os
import socket

_REFUSED = "the verifier's pass-to-pass step runs with the network blocked (allow_internet = false)"
_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex


def _loopback(address) -> bool:
    host = address[0] if isinstance(address, tuple) and address else address
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _guarded(sock) -> bool:
    return sock.family in (socket.AF_INET, socket.AF_INET6)


def connect(self, address):
    if _guarded(self) and not _loopback(address):
        raise OSError(101, f"Network is unreachable: {_REFUSED}")
    return _connect(self, address)


def connect_ex(self, address):
    if _guarded(self) and not _loopback(address):
        return 101
    return _connect_ex(self, address)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
os.environ["INFB_OFFLINE_STEP"] = "1"

_system = "/usr/lib/python3.12/sitecustomize.py"
if os.path.isfile(_system):
    try:
        exec(compile(open(_system, encoding="utf-8").read(), _system, "exec"), {"__name__": "sitecustomize"})
    except Exception:  # the system hook is a crash reporter; its absence changes no test
        pass
