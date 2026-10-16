#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Preflight for the verifier: the environment is what the task declares, or no reward at all."""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import mmap
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
    r = subprocess.run([sys.executable, "-c", "import transformers; print(transformers.__version__)"], capture_output=True, text=True, timeout=300)
    tv = r.stdout.strip()
    out["transformers"] = tv
    if r.returncode != 0 or tv != x["transformers_version"]:
        why = tv or (r.stderr.strip().splitlines() or ["no output"])[-1]
        fails.append(f"transformers is {why!r}, expected {x['transformers_version']}")
    imports = {}
    for mod in x["importable"]:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True, text=True, timeout=600, cwd="/")
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


def record_tests_digest(x: dict, out: dict) -> list[str]:
    """Record what the verifier is running, as a measurement: a digest over every file under /tests."""
    if not x.get("record_tests_digest"):
        return []
    h = hashlib.sha256()
    files = 0
    for p in sorted(TESTS_DIR.rglob("*")):
        if not p.is_file():
            continue
        h.update(str(p.relative_to(TESTS_DIR)).encode())
        h.update(b"\0")
        h.update(sha256_of(p).encode())
        files += 1
    out["tests_digest"] = {"files": files, "sha256": h.hexdigest()}
    return []


def _timed_read(path: Path, budget: int, limit_s: float) -> dict:
    """Read up to `budget` bytes of `path` for at most `limit_s` seconds and time it, bypassing the page cache where
    the filesystem allows (O_DIRECT); elsewhere buffered, with the file's cached pages dropped afterwards."""
    chunk = 16 << 20
    buf = mmap.mmap(-1, chunk)  # anonymous memory is page-aligned, as O_DIRECT requires
    direct = getattr(os, "O_DIRECT", 0)
    mode = "o_direct" if direct else "buffered (no O_DIRECT on this platform)"
    try:
        fd = os.open(path, os.O_RDONLY | direct)
    except OSError as exc:
        fd, mode = os.open(path, os.O_RDONLY), f"buffered (O_DIRECT open refused: {exc.strerror})"
    done, t0 = 0, time.monotonic()
    try:
        while done < budget and time.monotonic() - t0 < limit_s:
            try:
                n = os.preadv(fd, [buf], done)
            except OSError as exc:
                if mode != "o_direct" or exc.errno != errno.EINVAL:
                    raise
                os.close(fd)  # the filesystem took the flag but refuses the read: start over, buffered
                fd, mode = os.open(path, os.O_RDONLY), "buffered (O_DIRECT read refused: EINVAL)"
                done, t0 = 0, time.monotonic()
                continue
            if n <= 0:
                break
            done += n
            if n < chunk:
                break
        secs = time.monotonic() - t0
        if mode != "o_direct":
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass
    finally:
        os.close(fd)
        buf.close()
    return {"bytes": done, "seconds": round(secs, 3),
            "mib_per_s": round(done / (1 << 20) / secs, 1) if secs > 0 else None, "mode": mode}


def record_host_topology(x: dict, out: dict) -> list[str]:
    """Diagnostics for a slow reference load: recorded, never failed."""
    rec: dict = {}
    for key, argv in (("nvidia_smi_topo", ["nvidia-smi", "topo", "-m"]), ("numactl_hardware", ["numactl", "--hardware"])):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
            rec[key] = (done.stdout or done.stderr)[-8000:]
        except (OSError, subprocess.SubprocessError) as exc:
            rec[key] = f"unavailable: {type(exc).__name__}: {exc}"[:300]
    model_dir = Path(x["model_dir"])
    try:
        best = ("", "", "")
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            mount_point, rest = fields[4], line.split(" - ", 1)[1].split()
            if str(model_dir).startswith(mount_point) and len(mount_point) > len(best[0]):
                best = (mount_point, rest[0], rest[1])
        rec["model_dir_mount"] = {"mount_point": best[0], "fstype": best[1], "source": best[2]}
    except (OSError, IndexError) as exc:
        rec["model_dir_mount"] = f"unreadable: {exc!r}"[:300]
    try:
        weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        shards: dict[str, set] = {}
        for name, shard in weight_map.items():
            shards.setdefault(shard, set()).add(name.split(".", 1)[0])
        draft_only = sorted(s for s, heads in shards.items() if heads == {"mtp"})
        if draft_only:
            rec["timed_shard_read"] = {"shard": draft_only[0], **_timed_read(model_dir / draft_only[0], 1 << 30, 60.0),
                                       "why_this_shard": "mtp.0 only: the EAGLE draft, loaded after the target"}
        else:
            rec["timed_shard_read"] = "skipped: no shard holds the draft alone"
    except Exception as exc:  # noqa: BLE001 - a diagnostic never fails the preflight
        rec["timed_shard_read"] = f"failed: {type(exc).__name__}: {exc}"[:300]
    out["host_topology"] = rec
    return []


def main() -> int:
    x = load_expectations()
    out: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "task": x.get("task")}
    failures: list[str] = []
    for name, fn in (("gpus", check_gpus), ("weights", check_weights), ("tree", check_tree),
                     ("python", check_python), ("no_server", check_no_server), ("shm", check_shm),
                     ("tests_digest", record_tests_digest), ("host_topology", record_host_topology)):
        try:
            fails = fn(x, out)
        except Exception as exc:  # noqa: BLE001
            fails = [f"{name}: {exc!r}"]
        for f in fails:
            print(f"[preflight] FAIL {name}: {f}", flush=True)
        failures.extend(fails)
    out["failures"] = failures
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_DIR / "preflight.json").write_text(json.dumps(out, indent=1))
    if failures:
        print(f"[preflight] {len(failures)} failure(s); the environment is not the declared one, no reward is written", flush=True)
        return 2
    print(f"[preflight] ok: {out.get('gpu_count')} GPUs, checkpoint {out['checkpoint']['files']} files, tree at {out['head'][:10]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
