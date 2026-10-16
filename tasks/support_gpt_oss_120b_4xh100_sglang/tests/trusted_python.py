#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Start private host/reference code without editable .pth or sitecustomize.

Invoke as /opt/sglang/bin/python -I -S /tests/trusted_python.py script.py ... .
Only pinned dependencies and private verifier code enter this process. Reference
children receive their own explicit engine/kernel roots and also skip site.
"""
import os
from pathlib import Path
import runpy
import sys

assert sys.flags.isolated and sys.flags.no_site
root = Path(__file__).resolve().parent
dependencies = Path(sys.executable).parent.parent / "lib" / (
    f"python{sys.version_info.major}.{sys.version_info.minor}") / "site-packages"
assert dependencies.is_dir(), f"missing pinned dependency directory: {dependencies}"
target = Path(sys.argv[1]).resolve()
assert target.is_relative_to(root), "trusted entrypoint must be in the private verifier"
sys.path.extend([str(root), str(root / "public_eval"), str(dependencies)])
os.environ["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "public_eval"), str(dependencies)])
sys.argv = sys.argv[1:]
runpy.run_path(str(target), run_name="__main__")
