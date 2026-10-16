#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Run a command with no network but the loopback: the verifier's own offline condition.

    python offline_exec.py --record <json> -- <command...>

The process enters a new unprivileged user and network namespace (what `unshare -r -n` does), brings
its loopback up, records whether an outside host is reachable (the assertion that the block holds),
and execs the command. The L3 group runs under it, so its membership holds on every backend,
whatever the host's network. Where
the kernel or the runtime refuses the namespace, the command runs as it is and the record says so
(the harness's allow_internet = false is then the only block); nothing here ever refuses to run.
Expected on the oracle: the namespace is entered and the outside is unreachable.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import struct
import sys
import time
import urllib.request

SIOCGIFFLAGS, SIOCSIFFLAGS, IFF_UP = 0x8913, 0x8914, 0x1
PROBE = "https://huggingface.co"


def _enter() -> dict:
    uid, gid = os.getuid(), os.getgid()
    try:
        os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)
    except OSError as exc:
        return {"namespace": False, "error": f"unshare refused: {exc}", "loopback_up": None}
    try:
        for path, text in (("/proc/self/setgroups", "deny"), ("/proc/self/uid_map", f"0 {uid} 1"),
                           ("/proc/self/gid_map", f"0 {gid} 1")):
            with open(path, "w") as f:
                f.write(text)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ifr = struct.pack("16sh", b"lo", 0)
        flags = struct.unpack("16sh", fcntl.ioctl(s, SIOCGIFFLAGS, ifr))[1]
        fcntl.ioctl(s, SIOCSIFFLAGS, struct.pack("16sh", b"lo", flags | IFF_UP))
        s.close()
    except OSError as exc:  # the namespace holds (no route out); only the loopback may be missing
        return {"namespace": True, "error": f"loopback: {exc}", "loopback_up": False}
    return {"namespace": True, "error": None, "loopback_up": True}


def _outside_reachable() -> bool:
    try:
        urllib.request.urlopen(PROBE, timeout=10)
        return True
    except Exception:  # noqa: BLE001 -- any failure to reach it is the block holding
        return False


def main(argv: list[str]) -> None:
    record = None
    if argv[:1] == ["--record"]:
        record, argv = argv[1], argv[2:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        raise SystemExit("usage: offline_exec.py [--record JSON] -- COMMAND...")
    rec = {**_enter(), "probe": PROBE, "outside_reachable": _outside_reachable(),
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if record:
        os.makedirs(os.path.dirname(record) or ".", exist_ok=True)
        with open(record, "w") as f:
            json.dump(rec, f, indent=1)
    print(f"offline_exec: {json.dumps(rec)}", file=sys.stderr, flush=True)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main(sys.argv[1:])
