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
# The rebuild is NOT done here. The engine is installed editable and the verifier runs
# /public/build.sh on the submitted tree exactly as a candidate would, so the oracle and every
# candidate take the same build path; a different one for the oracle would be a defect.
set -euo pipefail
cd "${INFB_WORKSPACE:-/app}"
HERE="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$HERE/oracle.lock"
BASE="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"])' "$LOCK")"
PATCH_SHA="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["solution_patch"]["sha256"])' "$LOCK")"
TARGET_TREE="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["closure_oracle_tree"])' "$LOCK")"

HEAD_SHA="$(/usr/bin/git rev-parse HEAD)"
if [ "$HEAD_SHA" != "$BASE" ]; then
  echo "solve.sh: HEAD is $HEAD_SHA, not the prepared base $BASE; refusing" >&2
  exit 2
fi
[ -z "$(/usr/bin/git status --porcelain)" ] \
  || { echo "solve.sh: prepared tree is not clean; refusing" >&2; exit 2; }
echo "$PATCH_SHA  $HERE/solution.patch" | /usr/bin/sha256sum -c - >/dev/null \
  || { echo "solve.sh: solution.patch does not match the digest in oracle.lock; refusing" >&2; exit 3; }

/usr/bin/git switch -q -c reference-solution
/usr/bin/git apply --check --whitespace=nowarn "$HERE/solution.patch"
/usr/bin/git apply --index --whitespace=nowarn "$HERE/solution.patch"
[ "$(/usr/bin/git write-tree)" = "$TARGET_TREE" ] \
  || { echo "solve.sh: applied tree differs from the complete reference identity" >&2; exit 4; }
for f in vllm/models/qwen4_exp/nvidia/model.py vllm/models/qwen4_exp/nvidia/ple_layer.py \
         vllm/models/qwen4_exp/common/qsa_cache.py vllm/models/qwen4_exp/config.py \
         vllm/transformers_utils/configs/qwen4_exp.py; do
  [ -f "$f" ] || { echo "solve.sh: $f missing after apply" >&2; exit 4; }
done

# Commit only the paths in the verified patch. Public evaluation is unchanged.
PATCH_PATHS="$(mktemp)"
trap 'rm -f "$PATCH_PATHS"' EXIT
/usr/bin/git diff --cached --name-only -z > "$PATCH_PATHS"
xargs -0 /usr/bin/git -c user.name="oracle" -c user.email="oracle@local" commit -o -q --no-verify \
  -m "Apply the complete Qwen38 native reference and admitted repairs" -- < "$PATCH_PATHS"
[ "$(/usr/bin/git rev-parse 'HEAD^{tree}')" = "$TARGET_TREE" ]
echo "solve.sh: committed $(/usr/bin/git rev-parse --short HEAD) on $(/usr/bin/git rev-parse --abbrev-ref HEAD)"
