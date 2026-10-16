#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Install and verify the SD3 reference implementation."""
from pathlib import Path
import hashlib, json, os, subprocess
root = Path(__file__).resolve().parent
lock = json.loads((root / "oracle.lock").read_text())
workspace = Path(os.environ.get("INFB_WORKSPACE", "/app"))
receipt = workspace / "effective_target_env.json"
repo = Path(json.loads(receipt.read_text())["sglang_src"]) if receipt.exists() else workspace / "repo"
def git(*args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.STDOUT)
patch = root / "solution.patch"
if hashlib.sha256(patch.read_bytes()).hexdigest() != lock["patch_sha256"]:
    raise SystemExit("Oracle patch checksum mismatch")
if git("status", "--porcelain", "--untracked-files=all").strip():
    raise SystemExit("Oracle installation requires a clean candidate repository")
git("apply", "--check", str(patch))
git("apply", str(patch))
for name, expected in lock["changed_files_sha256"].items():
    if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
        raise SystemExit("Oracle file checksum mismatch: " + name)
print("Installed SD3 reference implementation")
