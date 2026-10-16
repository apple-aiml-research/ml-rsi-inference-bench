#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Launch one configuration's server the way the acceptance file does, for development, with its log kept.

    python3 /app/eval/glm53_flash/tests/serve.py --class low_latency --log /tmp/low_latency.log

It runs `sglang serve --model-path /model/zai-org/GLM-5.3-Flash <the class's flags> --host 127.0.0.1 --port 30000`
in the foreground, writing the server's own output both to the terminal and to `--log`; stop it with Ctrl-C. Run the
checks against that one server: `check_recipe.py` first, before any request, then the workloads, then
`check_adaptive.py --log` on the same log.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acceptance import CLASSES, MODEL, server_args  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--class", dest="cls", choices=sorted(CLASSES), required=True)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--log", required=True, help="the file the server's own output is written to")
    a = ap.parse_args()
    argv = [shutil.which("sglang") or "sglang", "serve", "--model-path", a.model, *server_args(a.cls),
            "--host", "127.0.0.1", "--port", str(a.port)]
    print("+ " + " ".join(argv), flush=True)
    with open(a.log, "w") as log:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            proc.terminate()
        return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
