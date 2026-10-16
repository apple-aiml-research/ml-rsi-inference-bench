#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Native support, checked as the grader checks it: the committed tree installs from the working tree, and SGLang's own
configuration and model registries resolve the checkpoint natively. The checking core the grader's L0 nodes run,
shared byte for byte with them; the model facts are in `native.json` beside this file."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

LIMIT_S = int(json.loads((Path(__file__).resolve().parent.parent / "limits.json").read_text())
              ["operations"]["unit_test_file"]["limit_s"])


def editable_install(app_dir: Path, py: str = sys.executable) -> list[str]:
    """[] when the tree installs editable, offline, and `sglang` resolves from `<app_dir>/python`."""
    env = dict(os.environ, SGLANG_BUILD_RUST_EXTS="none", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    r = subprocess.run(
        [py, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "--no-build-isolation", "--no-index", "-e", str(app_dir / "python")],
        env=env, capture_output=True, text=True, timeout=LIMIT_S, cwd=str(app_dir),
    )
    if r.returncode != 0:
        return ["editable install failed:\n" + r.stdout[-4000:] + r.stderr[-4000:]]
    r = subprocess.run(
        [py, "-c", "import os, sglang; print(os.path.realpath(sglang.__file__))"],
        capture_output=True, text=True, timeout=LIMIT_S, cwd="/",
    )
    if r.returncode != 0:
        return ["import sglang failed:\n" + r.stdout[-2000:] + r.stderr[-4000:]]
    if str((app_dir / "python").resolve()) not in r.stdout:
        return [f"sglang resolved outside the tree: {r.stdout.strip()}"]
    return []


def native_resolution(model_dir: str, architecture: str, model_type: str,
                      py: str = sys.executable) -> tuple[list[str], dict | None]:
    """([] when the declared architecture resolves natively with the checkpoint's model type, else the failures;
    the resolution record, or None when there is none)."""
    code = f"""
import json
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.utils import get_model_architecture
cfg = ModelConfig(model_path={model_dir!r}, trust_remote_code=False)
cls, arch = get_model_architecture(cfg)
print("INFB_L0 " + json.dumps({{"module": cls.__module__, "cls": cls.__name__, "arch": arch,
      "model_type": getattr(cfg.hf_config, "model_type", None),
      "architectures": list(getattr(cfg.hf_config, "architectures", []) or [])}}))
"""
    r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=LIMIT_S, cwd="/")
    if r.returncode != 0:
        return ["config did not parse or the architecture did not resolve:\n" + r.stdout[-3000:] + r.stderr[-8000:]], None
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("INFB_L0 ")]
    if not lines:
        return ["no resolution record in output:\n" + r.stdout[-3000:]], None
    info = json.loads(lines[-1][len("INFB_L0 "):])
    if info["arch"] != architecture:
        return [str(info)], info
    if not info["module"].startswith("sglang.srt.models."):
        return [f"resolved outside the engine's model package: {info}"], info
    if info["module"].endswith(".transformers"):
        return [f"resolved to the Transformers fallback: {info}"], info
    if info["model_type"] != model_type:
        return [str(info)], info
    return [], info


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="check that the tree installs and resolves the model natively")
    ap.add_argument("--app", default="/app")
    ap.add_argument("--facts", default=str(Path(__file__).resolve().parent / "native.json"))
    ap.add_argument("--skip-install", action="store_true", help="only the registry resolution")
    args = ap.parse_args(argv)
    facts = json.loads(Path(args.facts).read_text())
    failures = [] if args.skip_install else editable_install(Path(args.app))
    print("editable install: " + ("skipped" if args.skip_install else "holds" if not failures else failures[0]))
    res, info = native_resolution(facts["model_dir"], facts["architecture"], facts["model_type"])
    print(f"native resolution: {'holds' if not res else res[0]}" + (f" ({json.dumps(info)})" if info else ""))
    return 1 if failures or res else 0


if __name__ == "__main__":
    raise SystemExit(main())
