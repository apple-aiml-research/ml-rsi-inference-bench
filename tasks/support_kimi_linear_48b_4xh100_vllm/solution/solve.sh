#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail
cd "${INFB_WORKSPACE:-/app}"
HERE="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$HERE/oracle.lock"
BASE="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"])' "$LOCK")"
PATCH_SHA="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["solution_patch"]["sha256"])' "$LOCK")"
WANT_TREE="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["solution_patch"]["expected_tree"])' "$LOCK")"

HEAD_SHA="$(/usr/bin/git rev-parse HEAD)"
if [ "$HEAD_SHA" != "$BASE" ]; then
  echo "solve.sh: HEAD is $HEAD_SHA, not the base $BASE; refusing" >&2
  exit 2
fi
echo "$PATCH_SHA  $HERE/solution.patch" | /usr/bin/sha256sum -c - >/dev/null \
  || { echo "solve.sh: solution.patch does not match the digest in oracle.lock; refusing" >&2; exit 3; }

/usr/bin/git apply --check --whitespace=nowarn "$HERE/solution.patch"
/usr/bin/git apply --whitespace=nowarn "$HERE/solution.patch"
for f in vllm/model_executor/models/kimi_linear.py \
         vllm/model_executor/layers/kda.py \
         vllm/transformers_utils/configs/kimi_linear.py; do
  [ -f "$f" ] || { echo "solve.sh: $f missing after apply" >&2; exit 4; }
done

/usr/bin/git checkout -q -b reference-solution
mapfile -t CHANGED < <(/usr/bin/git apply --numstat "$HERE/solution.patch" | /usr/bin/cut -f3)
/usr/bin/git add -- "${CHANGED[@]}"
/usr/bin/git -c user.name="oracle" -c user.email="oracle@local" commit -q --no-verify -o \
  -m "Apply reference solution: KDA kernels and Kimi-Linear support" -- "${CHANGED[@]}"

GOT_TREE="$(/usr/bin/git rev-parse 'HEAD^{tree}')"
if [ "$GOT_TREE" != "$WANT_TREE" ]; then
  echo "solve.sh: committed tree $GOT_TREE, expected $WANT_TREE from oracle.lock" >&2
  exit 5
fi
echo "solve.sh: committed $(/usr/bin/git rev-parse --short HEAD) on $(/usr/bin/git rev-parse --abbrev-ref HEAD), tree $GOT_TREE"
