#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# The oracle, in DeepSWE's form: apply the closure diff recorded in oracle.lock onto the base commit
# and commit it on a branch; only the committed tree is graded (pre_artifacts.sh captures
# base..HEAD). Refuses when the tree is not at the base, when the patch digest does not match the
# lock, or when the applied tree is not the tree the lock names.
# Absolute paths throughout: the executing shell carries no image PATH.
set -euo pipefail
cd "${INFB_WORKSPACE:-/app}"
HERE="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$HERE/oracle.lock"
J() { /usr/bin/python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));exec("print("+sys.argv[2]+")")' "$LOCK" "$1"; }
BASE="$(J 'd["base"]')"
PATCH_SHA="$(J 'd["solution_patch"]["sha256"]')"
WANT_TREE="$(J 'd.get("closure_oracle_tree", "")')"

HEAD_SHA="$(/usr/bin/git rev-parse HEAD)"
if [ "$HEAD_SHA" != "$BASE" ]; then
  echo "solve.sh: HEAD is $HEAD_SHA, not the base $BASE; refusing" >&2
  exit 2
fi
echo "$PATCH_SHA  $HERE/solution.patch" | /usr/bin/sha256sum -c - >/dev/null \
  || { echo "solve.sh: solution.patch does not match the digest in oracle.lock; refusing" >&2; exit 3; }

/usr/bin/git apply --check --whitespace=nowarn "$HERE/solution.patch"
/usr/bin/git apply --whitespace=nowarn "$HERE/solution.patch"
# The tree, not the patch bytes, is what the reference IS: check it before committing. Staging first
# is deliberate -- `git write-tree` reads the INDEX, so the tree asserted below is exactly the tree
# the commit will carry; stage and commit only the exact patch paths.
#
# Do not replace the apply + add pair with `git apply --index`: with --index git stat-checks every
# pre-existing path against the index's cached stat data and refuses on a stat mismatch alone, which
# is the normal state of a tree whose image layer was distributed through a registry -- the pull
# re-extracts every file with a new inode and ctime.
SPEC="$(/usr/bin/mktemp /tmp/gpt-oss-reference-paths.XXXXXX)"
trap '/usr/bin/rm -f "$SPEC"' EXIT
/usr/bin/python3 -I -S - "$HERE/solution.patch" "$SPEC" <<'PY'
import pathlib, subprocess, sys
rows = subprocess.check_output(['/usr/bin/git', 'apply', '--numstat', '-z', sys.argv[1]])
paths = [row.split(b'\t', 2)[2] for row in rows.split(b'\0') if row]
assert paths and all(path and not path.startswith(b'/') for path in paths)
pathlib.Path(sys.argv[2]).write_bytes(b'\0'.join(paths) + b'\0')
PY
/usr/bin/git --literal-pathspecs add --force --pathspec-from-file="$SPEC" --pathspec-file-nul
GOT_TREE="$(/usr/bin/git write-tree)"
# closure_oracle_tree is optional: a variant that is by construction not the closure has no tree to
# match, while the digest check above still pins exactly which patch ran.
if [ -n "$WANT_TREE" ] && [ "$GOT_TREE" != "$WANT_TREE" ]; then
  echo "solve.sh: applied tree $GOT_TREE != oracle.lock closure_oracle_tree $WANT_TREE; refusing" >&2
  exit 4
fi
for f in python/sglang/srt/models/gpt_oss.py python/sglang/srt/layers/quantization/mxfp4.py \
         python/sglang/srt/layers/quantization/mxfp4_tensor.py; do
  [ -f "$f" ] || { echo "solve.sh: $f missing after apply" >&2; exit 5; }
done

# Commit the solution like a normal submission (only committed work is graded). Everything is already
# staged by the exact path list above, which is what the tree assertion measured.
/usr/bin/git switch -q -c reference-solution
/usr/bin/git --literal-pathspecs -c user.name="oracle" -c user.email="oracle@local" commit -q --no-verify -o \
  --pathspec-from-file="$SPEC" --pathspec-file-nul \
  -m "Apply reference solution: native MXFP4 gpt-oss-120b support for SGLang"
echo "solve.sh: committed $(/usr/bin/git rev-parse --short HEAD) on $(/usr/bin/git rev-parse --abbrev-ref HEAD)"
