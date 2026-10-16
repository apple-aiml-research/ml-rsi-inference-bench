#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Preflight for the verifier: the environment is what the task declares, or no reward at all.

Runs first in tests/test.sh, inside the separate verifier container, before any scored check. On
any failure it prints every reason and exits 2 WITHOUT writing a reward, so DeepSWE's frame trap
writes reward.txt = -1: an environment fault is never scored as a candidate defect. Nothing else
produces -1.

Every task fact is read from the "infb" block of /tests/config.json (ignored by grader.py) and from
the image's own record of the checkpoint (/model/.infb/manifest.json, written when the recipe
downloaded and hashed the files). The policy: devices by name and a memory floor, never a byte
count; full shard hashes at image build; per run, every file's size and the hashes of the small
files. The measurements are written to /logs/verifier/preflight.json for the record.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
VERIFIER_DIR = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
SMALL_FILE_BYTES = 64 << 20  # files at or below this size are re-hashed on every run
GIB = 1 << 30


def load_expectations() -> dict:
    cfg = json.loads((TESTS_DIR / "config.json").read_text())
    return {"base_commit": cfg["base_commit"], **cfg["infb"]}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_gpus(x: dict, out: dict) -> list[str]:
    fails: list[str] = []
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return [f"torch import failed: {exc!r}"]
    out["torch"] = torch.__version__
    if not torch.cuda.is_available():
        return ["torch.cuda.is_available() is False"]
    n = torch.cuda.device_count()
    out["gpu_count"] = n
    if n != int(x["gpus"]):
        fails.append(f"expected {x['gpus']} visible GPUs, found {n}")
    devices = []
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)
        devices.append({"index": i, "name": name, "total_gib": round(total / GIB, 1), "free_gib": round(free / GIB, 1)})
        if x["gpu_name"] not in name:
            fails.append(f"GPU {i} is {name!r}, expected a {x['gpu_name']}")
        if total < float(x["gpu_min_total_gib"]) * GIB:
            fails.append(f"GPU {i} has {total / GIB:.1f} GiB, floor {x['gpu_min_total_gib']} GiB")
        if free < float(x["gpu_min_free_fraction"]) * total:
            fails.append(f"GPU {i} has only {free / GIB:.1f} of {total / GIB:.1f} GiB free: a stale process holds it")
    out["devices"] = devices
    out["cuda_driver"] = getattr(torch.version, "cuda", None)
    return fails


def check_weights(x: dict, out: dict) -> list[str]:
    fails: list[str] = []
    model_dir = Path(x["model_dir"])
    record_path = Path(x["model_record"])
    if not record_path.is_file():
        return [f"missing checkpoint record {record_path}"]
    record = json.loads(record_path.read_text())
    if not record.get("complete"):
        fails.append(f"{record_path} does not record a complete checkpoint")
    if record.get("repo_id") != x["model_repo"] or record.get("revision") != x["model_revision"]:
        fails.append(f"checkpoint record is {record.get('repo_id')}@{record.get('revision')}, expected {x['model_repo']}@{x['model_revision']}")
    files = record.get("files", {})
    if len(files) != int(x["model_files"]):
        fails.append(f"checkpoint record lists {len(files)} files, expected {x['model_files']}")
    rehashed = 0
    for rel, entry in sorted(files.items()):
        p = model_dir / rel
        if not p.is_file():
            fails.append(f"missing checkpoint file {p}")
            continue
        size = p.stat().st_size
        if size != int(entry["size"]):
            fails.append(f"{p}: size {size} != recorded {entry['size']}")
            continue
        if size <= SMALL_FILE_BYTES:
            digest = sha256_of(p)
            rehashed += 1
            if digest != entry["sha256"]:
                fails.append(f"{p}: sha256 {digest[:12]} != recorded {entry['sha256'][:12]}")
    out["checkpoint"] = {"files": len(files), "rehashed_small_files": rehashed, "total_bytes": record.get("total_bytes")}
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
        out["checkpoint"]["architectures"] = cfg.get("architectures")
        out["checkpoint"]["model_type"] = cfg.get("model_type")
        if cfg.get("architectures") != x["architectures"]:
            fails.append(f"config.json architectures {cfg.get('architectures')} != {x['architectures']}")
    except Exception as exc:  # noqa: BLE001
        fails.append(f"config.json unreadable: {exc!r}")
    return fails


def check_tree(x: dict, out: dict) -> list[str]:
    fails: list[str] = []
    r = subprocess.run(["git", "-C", str(APP_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
    head = r.stdout.strip()
    out["head"] = head
    if r.returncode != 0 or head != x["base_commit"]:
        fails.append(f"/app HEAD is {head or r.stderr.strip()!r}, expected the base {x['base_commit']}")
    if not (APP_DIR / "python" / "sglang" / "__init__.py").is_file():
        fails.append("/app/python/sglang is not a package")
    return fails


def check_python(x: dict, out: dict) -> list[str]:
    fails: list[str] = []
    r = subprocess.run([sys.executable, "-S", "-c", "import transformers; print(transformers.__version__)"], capture_output=True, text=True, timeout=300, cwd="/")
    tv = r.stdout.strip()
    out["transformers"] = tv
    if r.returncode != 0 or tv != x["transformers_version"]:
        why = tv or (r.stderr.strip().splitlines() or ["no output"])[-1]
        fails.append(f"transformers is {why!r}, expected {x['transformers_version']}")
    imports = {}
    for mod in x["importable"]:
        t0 = time.time()
        # Preflight imports supplied dependencies only. Editable source behavior
        # belongs to candidate L0, not to host/reference health classification.
        if mod.split(".")[0] in ("sglang", "triton_kernels"):
            imports[mod] = {"deferred_to": "candidate L0 and isolated reference"}
            continue
        r = subprocess.run([sys.executable, "-S", "-c", f"import {mod}"], capture_output=True, text=True, timeout=600, cwd="/")
        imports[mod] = {"ok": r.returncode == 0, "seconds": round(time.time() - t0, 1)}
        if r.returncode != 0:
            fails.append(f"import {mod} failed: {(r.stderr.strip().splitlines() or ['no output'])[-1][:300]}")
    out["imports"] = imports
    for lib in x["shared_libraries"]:
        try:
            ctypes.CDLL(lib)
        except OSError as exc:
            fails.append(f"{lib} not loadable: {exc}")
    return fails


def check_no_server(x: dict, out: dict) -> list[str]:
    fails: list[str] = []
    stale = []
    for pid in Path("/proc").iterdir():
        if not pid.name.isdigit() or int(pid.name) == os.getpid():
            continue
        try:
            cmd = (pid / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "sglang" in cmd and ("serve" in cmd or "launch_server" in cmd or "scheduler" in cmd):
            stale.append(f"{pid.name}: {cmd[:120]}")
    out["stale_servers"] = stale
    if stale:
        fails.append("an SGLang server is already running in this container: " + "; ".join(stale))
    busy = []
    for port in x["ports_free"]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", int(port)))
            except OSError:
                busy.append(int(port))
    out["busy_ports"] = busy
    if busy:
        fails.append(f"ports in use that the test launches on: {busy}")
    return fails


def check_shm(x: dict, out: dict) -> list[str]:
    st = os.statvfs("/dev/shm")
    total = st.f_frsize * st.f_blocks
    out["dev_shm_gib"] = round(total / GIB, 1)
    if total < float(x["shm_min_gib"]) * GIB:
        return [f"/dev/shm is {total / GIB:.1f} GiB, floor {x['shm_min_gib']} GiB"]
    return []


def record_host_diagnostics(x: dict, out: dict) -> None:
    """Diagnostics only, never a failure.

    Recorded before the reference is served, so a slow load can be explained after the fact:
    - the GPU and NIC topology;
    - the NUMA layout, from numactl if present and from /sys regardless;
    - the device under the checkpoint;
    - one timed read of the smallest checkpoint shard with O_DIRECT, which bypasses the page cache and so
      cannot warm the reference's first load.
    """
    def run(cmd: list[str], timeout: float = 60) -> dict:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return {"rc": r.returncode, "out": (r.stdout or r.stderr)[-6000:]}
        except FileNotFoundError:
            return {"rc": None, "out": "absent"}
        except Exception as exc:  # noqa: BLE001 - a diagnostic never raises
            return {"rc": None, "out": repr(exc)[:300]}

    diag: dict = {"nvidia_smi_topo": run(["nvidia-smi", "topo", "-m"]),
                  "numactl_hardware": run(["numactl", "--hardware"])}
    nodes = {}
    for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        try:
            nodes[node.name] = {"cpulist": (node / "cpulist").read_text().strip(),
                                "meminfo": [l for l in (node / "meminfo").read_text().splitlines()
                                            if "MemTotal" in l or "MemFree" in l]}
        except OSError as exc:
            nodes[node.name] = repr(exc)[:200]
    diag["sys_numa_nodes"] = nodes
    model_dir = Path(x["model_dir"]).resolve()
    diag["checkpoint_mount"] = run(["findmnt", "-T", str(model_dir), "-o", "SOURCE,FSTYPE,TARGET,OPTIONS"])
    diag["checkpoint_df"] = run(["df", "-hT", str(model_dir)])
    shards = sorted(model_dir.glob("*.safetensors"), key=lambda q: q.stat().st_size)
    if shards:
        t0 = time.time()
        dd = run(["dd", f"if={shards[0]}", "of=/dev/null", "bs=16M", "iflag=direct"], timeout=600)
        diag["timed_shard_read"] = {"file": shards[0].name, "bytes": shards[0].stat().st_size,
                                    "seconds": round(time.time() - t0, 3), "o_direct": True, "dd": dd}
    out["host_diagnostics"] = diag


def main() -> int:
    x = load_expectations()
    out: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "task": x.get("task")}
    failures: list[str] = []
    for name, fn in (("gpus", check_gpus), ("weights", check_weights), ("tree", check_tree),
                     ("python", check_python), ("no_server", check_no_server), ("shm", check_shm)):
        try:
            fails = fn(x, out)
        except Exception as exc:  # noqa: BLE001
            fails = [f"{name}: {exc!r}"]
        for f in fails:
            print(f"[preflight] FAIL {name}: {f}", flush=True)
        failures.extend(fails)
    out["failures"] = failures
    try:
        record_host_diagnostics(x, out)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never change the preflight's verdict
        out["host_diagnostics"] = {"error": repr(exc)[:300]}
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_DIR / "preflight.json").write_text(json.dumps(out, indent=1))
    if failures:
        print(f"[preflight] {len(failures)} failure(s); the environment is not the declared one, no reward is written", flush=True)
        return 2
    print(f"[preflight] ok: {out.get('gpu_count')} GPUs, checkpoint {out['checkpoint']['files']} files, tree at {out['head'][:10]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
