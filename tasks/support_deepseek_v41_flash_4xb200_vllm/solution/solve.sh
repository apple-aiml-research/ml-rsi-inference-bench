#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# The oracle, in DeepSWE's form: apply the closure diff recorded in oracle.lock onto the prepared
# base and commit it on a branch; only the committed tree is graded (pre_artifacts.sh captures
# base..HEAD). Refuses when the tree is not at the prepared base or the patch digest does not match
# the lock. Absolute paths throughout: Pier's exec carries no image PATH.
#
# The rebuild is NOT done here. The engine is installed editable and the closure is Python plus
# Triton and TileLang sources that compile at run time; the verifier's L0 node rebuilds only when a
# submission touches compiled sources, exactly as for a candidate.
set -euo pipefail
cd "${INFB_WORKSPACE:-/app}"
HERE="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$HERE/oracle.lock"
BASE="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"])' "$LOCK")"
PATCH_SHA="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["solution_patch"]["sha256"])' "$LOCK")"

HEAD_SHA="$(/usr/bin/git rev-parse HEAD)"
if [ "$HEAD_SHA" != "$BASE" ]; then
  echo "solve.sh: HEAD is $HEAD_SHA, not the prepared base $BASE; refusing" >&2
  exit 2
fi
echo "$PATCH_SHA  $HERE/solution.patch" | /usr/bin/sha256sum -c - >/dev/null \
  || { echo "solve.sh: solution.patch does not match the digest in oracle.lock; refusing" >&2; exit 3; }

/usr/bin/git apply --check --whitespace=nowarn "$HERE/solution.patch"
/usr/bin/git apply --whitespace=nowarn "$HERE/solution.patch"
for f in vllm/models/deepseek_v4_1/__init__.py vllm/models/deepseek_v4_1/nvidia/model.py \
         vllm/models/deepseek_v4_1/nvidia/vl_model.py vllm/models/deepseek_v4_1/common/engram.py \
         vllm/models/deepseek_v4_1/nvidia/dspark.py; do
  [ -f "$f" ] || { echo "solve.sh: $f missing after apply" >&2; exit 4; }
done

# Commit the solution like a normal submission (only committed work is graded).
/usr/bin/git checkout -q -b feature/solution 2>/dev/null || true
/usr/bin/git add -A
/usr/bin/git -c user.name="oracle" -c user.email="oracle@local" commit -q --no-verify \
  -m "Apply reference solution: DeepSeek-V4.1-Flash landing"
echo "solve.sh: committed $(/usr/bin/git rev-parse --short HEAD) on $(/usr/bin/git rev-parse --abbrev-ref HEAD)"
