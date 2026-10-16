#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail
ws="${INFB_WORKSPACE:-/app}"
repo="${ws}/repo"
patch_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/min_new_tokens_no_host_sync.patch"

[ -d "$repo" ] || { echo "[solve] FATAL: no candidate repo at $repo" >&2; exit 2; }
[ -f "$patch_file" ] || { echo "[solve] FATAL: no patch at $patch_file" >&2; exit 2; }

cd "$repo"
# -p2 strips `a/python/`: the staged tree is the FLATTENED package root, so repo/sglang is where
# python/sglang lives upstream.
if patch -p2 --batch --forward < "$patch_file"; then
  echo "[solve] patch applied"
else
  echo "[solve] patch(1) refused; retrying with git apply" >&2
  git apply -p2 --verbose "$patch_file"
  echo "[solve] git apply succeeded"
fi

pen=sglang/srt/sampling/penaltylib/min_new_tokens.py
orc=sglang/srt/sampling/penaltylib/orchestrator.py
sb=sglang/srt/managers/schedule_batch.py
check () { grep -q "$2" "$1" || { echo "[solve] FATAL: $1 lacks '$2' after patching" >&2; exit 3; }; }
check "$pen" "masked_fill_"
check "$pen" "uses_output_ids = False"
check "$orc" "def needs_output_ids"
check "$orc" "uses_output_ids: bool = True"
check "$sb"  "needs_output_ids()"
# And that the syncing forms are GONE, not merely joined by the new ones.
if grep -qE "logits\[mask\]" "$pen"; then
  echo "[solve] FATAL: $pen still contains the boolean-mask indexing" >&2; exit 3
fi
if grep -q "TODO: this can be slow" "$sb"; then
  echo "[solve] FATAL: $sb still builds the delayed tensor unconditionally" >&2; exit 3
fi
"${INFB_VLLM_PYTHON:-python3}" -c "
import ast, pathlib
for f in ('$pen', '$orc', '$sb'):
    ast.parse(pathlib.Path(f).read_text())
print('[solve] all three patched files parse')
"
