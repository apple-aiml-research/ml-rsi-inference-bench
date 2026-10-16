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
# /app/eval/diffusion_gemma/build.sh on the submitted tree exactly as a candidate would, so the
# oracle and every candidate take the same build path; a different one for the oracle would be a
# defect. The closure changes no native source (the snapshot's native-source rule).
set -euo pipefail
cd "${INFB_WORKSPACE:-/app}"
HERE="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$HERE/oracle.lock"
BASE="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["prepared_base"]["commit"])' "$LOCK")"
PATCH_SHA="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["solution_patch"]["sha256"])' "$LOCK")"
TARGET_TREE="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["closure_oracle_tree"])' "$LOCK")"

# An index the container runtime left unreadable is rebuilt from HEAD; a valid index is left
# untouched. A pulled image's index carries stale stats; refresh it before anything reads it.
/usr/bin/git status >/dev/null 2>&1 || { rm -f .git/index && /usr/bin/git read-tree HEAD; }
/usr/bin/git update-index -q --refresh >/dev/null 2>&1 || true
# Integrity read, after the repair and before any change: the working tree against HEAD, the
# zero-length files under /app (outside .git), and the shared objects whose header is not ELF.
echo "solve.sh: integrity: status-porcelain-lines=$(/usr/bin/git status --porcelain | wc -l)"
/usr/bin/git status --porcelain | head -20
find /app -xdev -type f -size 0 -not -path '/app/.git/*' 2>/dev/null | LC_ALL=C sort > /tmp/infb-zero-length.txt
echo "solve.sh: integrity: zero-length-files=$(wc -l < /tmp/infb-zero-length.txt) sha256=$(sha256sum /tmp/infb-zero-length.txt | cut -c1-64)"
/usr/bin/python3 - <<'PYEOF'
import os
bad, n = [], 0
for root, _, files in os.walk("/app"):
    if "/.git" in root:
        continue
    for f in files:
        if f.endswith(".so"):
            p = os.path.join(root, f)
            if os.path.islink(p):
                continue
            n += 1
            with open(p, "rb") as fh:
                if fh.read(4) != b"\x7fELF":
                    bad.append(p)
print(f"solve.sh: integrity: shared-objects={n} non-ELF={len(bad)} {bad[:10]}")
PYEOF
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
/usr/bin/git apply --check --binary --whitespace=nowarn "$HERE/solution.patch"
/usr/bin/git apply --index --binary --whitespace=nowarn "$HERE/solution.patch"
[ "$(/usr/bin/git write-tree)" = "$TARGET_TREE" ] \
  || { echo "solve.sh: applied tree differs from the complete reference identity" >&2; exit 4; }
for f in vllm/model_executor/models/diffusion_gemma.py diffusion_gemma_attend.py \
         vllm/v1/attention/ops/diffusion_gemma_mask.py vllm/utils/diffusion.py; do
  [ -f "$f" ] || { echo "solve.sh: $f missing after apply" >&2; exit 4; }
done

# Commit only the paths in the verified patch. Public evaluation is unchanged.
PATCH_PATHS="$(mktemp)"
trap 'rm -f "$PATCH_PATHS"' EXIT
/usr/bin/git diff --cached --name-only -z > "$PATCH_PATHS"
xargs -0 /usr/bin/git -c user.name="oracle" -c user.email="oracle@local" commit -o -q --no-verify \
  -m "Apply the complete DiffusionGemma reference and admitted repairs" -- < "$PATCH_PATHS"
[ "$(/usr/bin/git rev-parse 'HEAD^{tree}')" = "$TARGET_TREE" ]
echo "solve.sh: committed $(/usr/bin/git rev-parse --short HEAD) on $(/usr/bin/git rev-parse --abbrev-ref HEAD)"
