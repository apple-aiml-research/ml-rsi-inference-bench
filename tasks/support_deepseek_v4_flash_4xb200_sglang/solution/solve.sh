#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# The oracle, in DeepSWE's form: apply the closure diff recorded in oracle.lock onto the base
# commit and commit it on a branch; only the committed tree is graded (pre_artifacts.sh captures
# base'..HEAD). Refuses when the tree is not at base' or the patch digest does not match the lock.
# Absolute paths throughout: the harness's exec carries no image PATH.
set -euo pipefail
cd "${INFB_WORKSPACE:-/app}"
HERE="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$HERE/oracle.lock"
BASE="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"])' "$LOCK")"
PATCH_SHA="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["solution_patch"]["sha256"])' "$LOCK")"

HEAD_SHA="$(/usr/bin/git rev-parse HEAD)"
if [ "$HEAD_SHA" != "$BASE" ]; then
  echo "solve.sh: HEAD is $HEAD_SHA, not the base $BASE; refusing" >&2
  exit 2
fi
echo "$PATCH_SHA  $HERE/solution.patch" | /usr/bin/sha256sum -c - >/dev/null \
  || { echo "solve.sh: solution.patch does not match the digest in oracle.lock; refusing" >&2; exit 3; }

/usr/bin/git apply --check --whitespace=nowarn "$HERE/solution.patch"
/usr/bin/git apply --whitespace=nowarn "$HERE/solution.patch"
for f in python/sglang/srt/models/deepseek_v4.py python/sglang/srt/configs/deepseek_v4.py \
         python/sglang/srt/layers/attention/deepseek_v4_backend.py python/sglang/srt/layers/mhc.py \
         python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py; do
  [ -f "$f" ] || { echo "solve.sh: $f missing after apply" >&2; exit 4; }
done

# Commit the solution like a normal submission (only committed work is graded).
/usr/bin/git checkout -q -b feature/solution 2>/dev/null || true
/usr/bin/git add -A
/usr/bin/git -c user.name="oracle" -c user.email="oracle@local" commit -q --no-verify \
  -m "Apply reference solution: DeepSeek-V4-Flash landing (sglang #23882)"
echo "solve.sh: committed $(/usr/bin/git rev-parse --short HEAD) on $(/usr/bin/git rev-parse --abbrev-ref HEAD)"
