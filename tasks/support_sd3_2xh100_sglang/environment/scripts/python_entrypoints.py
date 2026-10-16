#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Install and check the task's Python launchers without changing its interpreter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile


def write_atomic(path: Path, data: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".sd3-python-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def interpreter() -> Path:
    # Keep the venv path: resolving its symlink would discard pyvenv.cfg.
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix) / "bin" / "python3"
    return Path(sys.executable).resolve()


def install(workspace: Path, bindir: Path, site: Path) -> dict:
    target = interpreter().absolute()
    aliases = [bindir / "python", bindir / "python3"]
    if not target.is_file() or not os.access(target, os.X_OK):
        raise ValueError(f"Python interpreter is not executable: {target}")
    target_entry = target.parent.resolve() / target.name
    if target_entry in [p.parent.resolve() / p.name for p in aliases]:
        raise ValueError("Refusing to overwrite the interpreter with its own launcher")
    if not site.is_dir():
        raise ValueError(f"Dependency directory is missing: {site}")
    source = workspace / "repo" / "python"
    body = "\n".join([
        "#!/bin/sh",
        "# Use the task interpreter; preserve explicitly supplied import precedence.",
        "if [ \"${PYTHONPATH+x}\" != x ]; then",
        "    PYTHONPATH=" + shlex.quote(str(source)),
        "fi",
        "_sd3_site=" + shlex.quote(str(site)),
        'case ":${PYTHONPATH}:" in',
        '    *":${_sd3_site}:"*) ;;',
        '    *) PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${_sd3_site}" ;;',
        "esac",
        "export PYTHONPATH",
        "unset _sd3_site",
        "exec " + shlex.quote(str(target)) + ' "$@"',
        "",
    ])
    for alias in aliases:
        write_atomic(alias, body, 0o755)
    receipt = {
        "schema": "sd3-python-entrypoints-v1",
        "python_executable": str(target),
        "python_prefix": sys.prefix,
        "source": str(source),
        "site_packages": str(site),
        "bin_dir": str(bindir),
        "entrypoints": [str(p) for p in aliases],
    }
    write_atomic(workspace / "python-entrypoints.json", json.dumps(receipt, indent=2) + "\n")
    return receipt


CHECK = r'''
import json, pathlib, sys
import torch, diffusers, sglang
from sglang.multimodal_gen import DiffGenerator
print(json.dumps({
    "python_executable": sys.executable, "python_prefix": sys.prefix,
    "torch": torch.__version__, "diffusers": diffusers.__version__,
    "sglang_file": str(pathlib.Path(sglang.__file__).resolve()),
    "api_callable": callable(DiffGenerator.from_pretrained),
}))
'''


def check(workspace: Path) -> dict:
    receipt = json.loads((workspace / "python-entrypoints.json").read_text())
    write_atomic(workspace / "python-entrypoint-check.json", json.dumps({"passed": False, "status": "checking"}) + "\n")
    readings = []
    for name in ("python", "python3"):
        # A fresh shell must work without preparation's transient exports.
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        command = "exec " + name + " -c " + shlex.quote(CHECK)
        proc = subprocess.run(["/bin/sh", "-c", command], env=env, cwd=workspace,
                              text=True, capture_output=True, timeout=120)
        if proc.returncode:
            raise RuntimeError(f"{name} startup failed (exit {proc.returncode}): {proc.stderr[-2000:]}")
        reading = json.loads(proc.stdout.strip().splitlines()[-1])
        if reading["python_prefix"] != receipt["python_prefix"]:
            raise RuntimeError(f"{name} resolved to a different Python environment")
        if not Path(reading["sglang_file"]).is_relative_to(Path(receipt["source"]).resolve()):
            raise RuntimeError(f"{name} imports SGLang outside the candidate tree")
        if not reading["api_callable"]:
            raise RuntimeError(f"{name} cannot access DiffGenerator.from_pretrained")
        readings.append({"entrypoint": name, **reading})
    result = {"schema": "sd3-python-entrypoint-check-v1", "passed": True, "readings": readings}
    write_atomic(workspace / "python-entrypoint-check.json", json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("install", "check"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--site-packages", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.absolute()
    if args.operation == "install":
        if args.bin_dir is None:
            parser.error("install requires --bin-dir")
        site = args.site_packages or Path(sysconfig.get_paths()["purelib"])
        result = install(workspace, args.bin_dir.absolute(), site.absolute())
        # Preparation keeps this raw interpreter for isolated verifier references.
        print(result["python_executable"])
    else:
        print(json.dumps(check(workspace)))


if __name__ == "__main__":
    main()
