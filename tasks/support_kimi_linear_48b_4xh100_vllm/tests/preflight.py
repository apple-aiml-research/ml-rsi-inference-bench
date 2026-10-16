#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Preflight for the verifier: the environment is what the task declares, or no reward at all.

Runs first in tests/test.sh, inside the separate verifier container, before any scored check. On
any failure it prints every reason and exits 2 WITHOUT writing a reward, so DeepSWE's frame trap
writes reward.txt = -1: an environment fault is never scored as a candidate defect. Nothing else produces -1.

Every task fact is read from the "infb" block of /tests/config.json
(ignored by grader.py) and from the image's own record of the checkpoint
($HF_HUB_CACHE/.infb/manifest.json, written when the recipe downloaded and hashed the files): devices by name and a memory floor, never a byte count; full shard hashes at
image build; per run, every file's size and the hashes of the small files. The measurements are
written to /logs/verifier/preflight.json for the record.
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
    model_dir = Path(x["model_dir"])  # the snapshot the repository id resolves to
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
    # Inspect immutable base blobs. A candidate may rename/delete/reimplement these paths;
    # that is scored by its numerical/serving checks, never an environment-invalid result.
    base_paths = ["vllm/__init__.py", *x["supplied_kernel_paths"]]
    missing = [rel for rel in base_paths if subprocess.run(
        ["git", "-C", str(APP_DIR), "cat-file", "-e", f"{x['base_commit']}:{rel}"],
        capture_output=True, timeout=30).returncode]
    if missing:
        fails.append(f"the supplied kernels are missing from the tree: {missing}")
    r = subprocess.run(["git", "-C", str(APP_DIR), "diff", "--name-only", x["base_commit"], "--",
                        *x["supplied_kernel_paths"]], capture_output=True, text=True)
    out["supplied_kernels_changed"] = [ln for ln in r.stdout.split() if ln]
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


def check_datasets(x: dict, out: dict) -> list[str]:
    """The offline datasets the vendored evaluator reads, by digest: a wrong file here would be
    discovered minutes into the accuracy run otherwise, and a truncated test split would silently
    change what "all 1,319 questions" means."""
    fails: list[str] = []
    seen = {}
    for rel, want in sorted(x["dataset_digests"].items()):
        p = Path(x["dataset_root"]) / rel
        if not p.is_file():
            fails.append(f"missing dataset file {p}")
            continue
        got = sha256_of(p)
        seen[rel] = got
        if got != want:
            fails.append(f"{p}: sha256 {got[:12]} != recorded {want[:12]}")
    out["datasets"] = seen
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
        if "vllm" in cmd and ("serve" in cmd or "EngineCore" in cmd or "VLLM::" in cmd):
            stale.append(f"{pid.name}: {cmd[:120]}")
    out["stale_servers"] = stale
    if stale:
        fails.append("a vLLM server is already running in this container: " + "; ".join(stale))
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


def record_host_io(x: dict, out: dict) -> list[str]:
    """NON-GATING host I/O diagnostics: on some hosts the reference's weights have loaded much more
    slowly on the non-zero ranks, by a mechanism this check cannot see.
    Recorded for attribution only; this function never returns a failure. The shard read is O_DIRECT, so it
    bypasses the page cache and does not warm the reference's first load."""
    rec: dict = {}
    def run(cmd, timeout=60):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return (r.stdout or r.stderr).strip()[-6000:]
        except Exception as exc:  # noqa: BLE001
            return f"unavailable: {exc!r}"
    rec["nvidia_smi_topo"] = run(["nvidia-smi", "topo", "-m"])
    numactl = run(["numactl", "--hardware"])
    if numactl.startswith("unavailable"):
        nodes = {}
        for d in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
            try:
                nodes[d.name] = (d / "cpulist").read_text().strip()
            except OSError:
                pass
        rec["numa"] = {"numactl": numactl, "sysfs_nodes": nodes}
    else:
        rec["numa"] = {"numactl": numactl}
    try:
        model_dir = Path(os.path.realpath(x.get("model_dir", "/model")))
        shards = sorted(model_dir.glob("*.safetensors"))
        target = Path(os.path.realpath(shards[-1])) if shards else None
        best = ("", "")
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            parts = line.split(" - ")
            mnt = parts[0].split()[4]; src = parts[1].split() if len(parts) > 1 else ["?", "?"]
            if target and str(target).startswith(mnt.rstrip("/") + "/") and len(mnt) > len(best[0]):
                best = (mnt, f"{src[1]} {src[0]}")
        rec["checkpoint_mount"] = {"mount": best[0], "source": best[1], "shard": str(target) if target else None}
        if target:
            t0 = time.time()
            r = subprocess.run(["dd", f"if={target}", "of=/dev/null", "bs=4M", "count=256", "iflag=direct"],
                               capture_output=True, text=True, timeout=300)
            dt = time.time() - t0
            rec["shard_direct_read"] = {"bytes": 256 * 4 * 1024 * 1024, "seconds": round(dt, 2),
                                        "mib_per_s": round(1024 / dt, 1) if r.returncode == 0 and dt > 0 else None,
                                        "rc": r.returncode, "dd": (r.stderr or "").strip()[-300:]}
    except Exception as exc:  # noqa: BLE001
        rec["error"] = repr(exc)
    out["host_io"] = rec
    return []


def main() -> int:
    x = load_expectations()
    out: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "task": x.get("task")}
    failures: list[str] = []
    for name, fn in (("gpus", check_gpus), ("weights", check_weights), ("tree", check_tree),
                     ("python", check_python), ("datasets", check_datasets),
                     ("no_server", check_no_server), ("shm", check_shm), ("host_io", record_host_io)):
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
    print(
        f"[preflight] ok: {out.get('gpu_count')} GPUs, checkpoint {out['checkpoint']['files']} files, "
        f"tree at {out['head'][:10]}, transformers {out.get('transformers')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
