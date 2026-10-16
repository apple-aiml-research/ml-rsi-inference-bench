#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#












set -uo pipefail

REPO="${INFB_WORKSPACE:-/app}/repo"
OUT_DIR="${INFB_ARTIFACTS_DIR:-/logs/artifacts}"
OUT="${OUT_DIR}/model.patch"
mkdir -p "$OUT_DIR"












CAPTURE_STATUS="${OUT_DIR}/model.patch.capture.json"
CAPTURE_STDERR="${OUT_DIR}/model.patch.capture.stderr"
capture_failures=""
capture_base=""
capture_done=0
capture_fail() {
  echo "[pre_artifacts] CAPTURE FAILED: $1" >&2
  capture_failures="${capture_failures:+${capture_failures}; }$1"
}
_capture_json() { printf '%s' "$1" | tr '\n\t\r' '   ' | tr -d '\000-\037' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
_capture_write_status() {
  printf '{"schema": "infb.capture_status.v1", "status": "%s", "failures": "%s", "base": "%s"}\n' \
    "$1" "$(_capture_json "$2")" "$(_capture_json "$capture_base")" > "${CAPTURE_STATUS}.tmp" \
    && mv -f -- "${CAPTURE_STATUS}.tmp" "$CAPTURE_STATUS"
}
_capture_finish() {
  local rc="${1:-$?}"
  [ "$capture_done" = 1 ] || [ -n "$capture_failures" ] \
    || capture_fail "the capture did not reach its end (pre_artifacts.sh was interrupted, exit ${rc})"
  if [ -n "$capture_failures" ]; then
    if [ -f "$OUT" ]; then mv -f -- "$OUT" "${OUT}.failed"; fi
    tail -n 20 "$CAPTURE_STDERR" >&2 2>/dev/null
    _capture_write_status failed "$capture_failures"
    exit 3
  fi
  [ -s "$CAPTURE_STDERR" ] || rm -f -- "$CAPTURE_STDERR"
  _capture_write_status ok ""
  exit "$rc"
}
rm -f -- "${OUT}.failed"
: > "$CAPTURE_STDERR"
_capture_write_status failed "the capture did not finish (pre_artifacts.sh started and never reached its exit trap)"
trap _capture_finish EXIT


if [ ! -d "$REPO/.git" ]; then
  capture_fail "no git repository at $REPO, so there is nothing to capture"
  exit 0
fi


SCOPE=(':(exclude).infb_autoresearch[/**]' ':(exclude)iterations[/**]')
if ! git -C "$REPO" add -A "${SCOPE[@]}" 2>>"$CAPTURE_STDERR"; then
  SCOPE=(.)
  git -C "$REPO" add -A 2>>"$CAPTURE_STDERR" || capture_fail "git add -A failed in $REPO"
fi


if   git -C "$REPO" rev-parse --verify -q inference_base >/dev/null 2>&1; then
  capture_base="inference_base"
  git -C "$REPO" diff --binary --no-ext-diff --cached inference_base > "$OUT" 2>>"$CAPTURE_STDERR" \
    || capture_fail "git diff --cached inference_base failed"
elif git -C "$REPO" diff --binary --no-ext-diff HEAD > "$OUT" 2>>"$CAPTURE_STDERR"; then
  capture_base="HEAD (FALLBACK: the inference_base tag is missing)"
else
  capture_base="worktree (LAST RESORT: neither inference_base nor HEAD was usable)"
  git -C "$REPO" diff --binary --no-ext-diff > "$OUT" 2>>"$CAPTURE_STDERR" || capture_fail "git diff failed"
fi



left="$( { git -C "$REPO" diff --name-only --ignore-submodules=dirty -- "${SCOPE[@]}" \
           && git -C "$REPO" ls-files --others --exclude-standard -- "${SCOPE[@]}"; } 2>>"$CAPTURE_STDERR" )" \
  || capture_fail "could not compare the worktree with the captured index"
[ -z "$left" ] || capture_fail "the worktree no longer matches the captured index ($(printf '%s\n' "$left" | wc -l | tr -d ' ') path(s): $(printf '%s\n' "$left" | head -n 5 | tr '\n' ' '))"

capture_done=1
echo "[pre_artifacts] wrote $(wc -c < "$OUT" 2>/dev/null || echo 0) bytes -> $OUT"
