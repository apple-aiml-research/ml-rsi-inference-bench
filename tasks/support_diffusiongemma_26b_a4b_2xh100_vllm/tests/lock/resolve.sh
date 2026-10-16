#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Resolve the hash-locked wheel sets environment/Dockerfile installs, inside the pinned CUDA base
# image (the target platform and interpreter), from public indexes only. Run once at authoring:
#
#   docker run --rm --network host -v <this dir>:/out -v <this file>:/resolve.sh \
#     docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301d... bash /resolve.sh
#
# The rule: requirements.lock pins every package the base
# commit's requirements/test/cuda.txt lists, at that file's pin -- the engine's own CI environment
# at the base -- and resolves the rest of requirements/cuda.txt (which includes common.txt) and
# requirements/build/cuda.txt, the latter for the no-build-isolation editable install, with the
# resolver's clock set to the base commit's date, so no package newer than the engine's starting
# point enters. transformers is pinned explicitly at 5.5.3, the base's test pin: 5.11.0 and later
# carry a complete implementation of the model this task asks for, which must not be importable.
#
# Also written: requirements-tools.lock (huggingface_hub for the checkpoint download, at the test
# lock's pin, plus hf_transfer), requirements-flashinfer-jit-cache.lock (FlashInfer's pre-compiled
# JIT modules, the same release as flashinfer-python, installed --no-deps in its own layer) and
# RESOLVED.md, which records every input and both checks of the rule.
set -euo pipefail
BASE_SHA=1ce3cdc5c14f656f81f91e264d557e0b6e6fea54
BASE_DATE=2026-06-12T05:16:14Z
UV_VERSION=0.9.28
OUT=${OUT:-/out}

apt-get update -qq && apt-get install -y -qq --no-install-recommends python3.12-venv curl ca-certificates >/dev/null
python3.12 -m venv /tmp/uvenv
/tmp/uvenv/bin/pip install -q "uv==${UV_VERSION}"
UV=/tmp/uvenv/bin/uv
PY=/tmp/uvenv/bin/python

mkdir -p /tmp/src/requirements/build /tmp/src/requirements/test "$OUT" && cd /tmp/src
for f in requirements/cuda.txt requirements/common.txt requirements/build/cuda.txt requirements/test/cuda.txt; do
  curl -fsSL --retry 5 -o "$f" "https://raw.githubusercontent.com/vllm-project/vllm/${BASE_SHA}/${f}"
done
sha() { sha256sum "$1" | cut -d' ' -f1; }

# Every name==version line of the test lock, as requirements (not only constraints), so the whole
# CI environment is installed at its pins.
grep -E '^[A-Za-z0-9_.-]+(\[[^]]*\])?==[^ ;]+' requirements/test/cuda.txt | sed 's/[[:space:]].*$//' > test-pins.in
cat > extra.in <<'EOF'
# The base's own test pin, restated so a change to the test lock cannot move it silently.
transformers==5.5.3
EOF

# --exclude-newer applies the base's date to every candidate that carries an upload time. The
# PyTorch index serves some files without one (uv warns and cannot exclude them), so the rule is
# also checked after resolution: every locked package the test lock does not pin must be a PyPI
# release uploaded before the base's date (check 2 below, recorded in RESOLVED.md).
"$UV" pip compile requirements/cuda.txt requirements/build/cuda.txt test-pins.in extra.in \
  --generate-hashes \
  --python "$PY" \
  --exclude-newer "$BASE_DATE" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl \
  --index-strategy unsafe-best-match \
  -o "$OUT/requirements.lock" 2> resolve-stderr.txt || { cat resolve-stderr.txt >&2; exit 3; }

FIV=$(grep -oE '^flashinfer-python==[0-9A-Za-z.+]+' "$OUT/requirements.lock" | head -1 | cut -d= -f3)
printf 'flashinfer-jit-cache==%s+cu130\n' "$FIV" > jit-cache.in
"$UV" pip compile jit-cache.in --no-deps --generate-hashes --python "$PY" \
  --index-url https://flashinfer.ai/whl/cu130 \
  -o "$OUT/requirements-flashinfer-jit-cache.lock"

HFV=$(grep -oE '^huggingface[-_]hub==[0-9A-Za-z.+]+' "$OUT/requirements.lock" | head -1)
printf '%s\nhf_transfer\n' "$HFV" > tools.in
"$UV" pip compile tools.in --generate-hashes --python "$PY" --exclude-newer "$BASE_DATE" \
  --index-url https://pypi.org/simple -o "$OUT/requirements-tools.lock"

cat > checks.py <<'EOF'
"""The rule's two checks. 1: every test pin is locked at exactly its version. 2: every locked
package the test lock does not pin is a PyPI release first uploaded before the base's date."""
import json, re, sys, urllib.request
lock, test, cutoff, which = sys.argv[1:5]
key = lambda n: n.lower().replace("_", "-")
pat = re.compile(r'^([A-Za-z0-9_.-]+)(?:\[[^]]*\])?==([^\s\\;]+)')
pins = {key(m.group(1)): m.group(2) for m in map(pat.match, open(lock)) if m}
tpins = {key(m.group(1)): m.group(2) for m in map(pat.match, open(test)) if m}
if which == "1":
    bad = [(n, v, pins.get(n)) for n, v in tpins.items() if pins.get(n) != v]
    print(f"{len(tpins)} test pins; {len(bad)} not honoured: {bad}")
    sys.exit(1 if bad else 0)
late = []
for name in sorted(set(pins) - set(tpins)):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{pins[name]}/json", timeout=60) as r:
        times = sorted(u["upload_time_iso_8601"] for u in json.load(r).get("urls", []))
    first = times[0] if times else "none"
    ok = first != "none" and first <= cutoff
    late += [] if ok else [name]
    print(f"{name}=={pins[name]} first upload {first} {'ok' if ok else 'AFTER THE BASE DATE'}")
print(f"{len(set(pins) - set(tpins))} packages not pinned by the test lock; {len(late)} first uploaded after the base date: {late}")
sys.exit(1 if late else 0)
EOF
python3.12 checks.py "$OUT/requirements.lock" test-pins.in "$BASE_DATE" 1 > check1.txt
python3.12 checks.py "$OUT/requirements.lock" test-pins.in "$BASE_DATE" 2 > check2.txt
cat check1.txt; tail -1 check2.txt

{
  echo "# Lock provenance"
  echo
  echo "- resolved: $(date -u +%Y-%m-%dT%H:%M:%SZ) inside nvidia/cuda:13.0.3-devel-ubuntu24.04 (x86_64, $($PY --version))"
  echo "- resolver: uv ${UV_VERSION} (pip compile, --generate-hashes, --exclude-newer ${BASE_DATE})"
  echo "- inputs at vllm ${BASE_SHA}:"
  echo "  - requirements/cuda.txt        sha256 $(sha requirements/cuda.txt)"
  echo "  - requirements/common.txt      sha256 $(sha requirements/common.txt)   (pulled in by cuda.txt's -r)"
  echo "  - requirements/build/cuda.txt  sha256 $(sha requirements/build/cuda.txt)"
  echo "  - requirements/test/cuda.txt   sha256 $(sha requirements/test/cuda.txt)   ($(wc -l < test-pins.in) pins, each required at its version)"
  echo "- additions (extra.in): transformers==5.5.3"
  echo "- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130,"
  echo "           https://flashinfer.ai/whl/cu130, https://flashinfer.ai/whl (unsafe-best-match)"
  echo "- result: $(grep -cE '^[A-Za-z0-9_.-]+==' "$OUT/requirements.lock") packages in requirements.lock"
  echo "- FlashInfer JIT cache: flashinfer-jit-cache==${FIV}+cu130 from https://flashinfer.ai/whl/cu130 (--no-deps)"
  echo "- tools: ${HFV}, hf_transfer"
  echo "- check 1: $(cat check1.txt)"
  echo "- check 2: $(tail -1 check2.txt)"
  echo "- resolver warnings: $(grep -c 'missing an upload date' resolve-stderr.txt || true) index files without an upload date, for" \
       "$(grep -oE 'warning: [A-Za-z0-9_.]+-[0-9]' resolve-stderr.txt | sed -E 's/^warning: //; s/-[0-9]$//' | sort -u | tr '\n' ' ')"
  echo "  (every such package is covered by check 1 or check 2)"
  echo
  echo "## Check 2, per package"
  echo
  sed 's/^/    /' check2.txt
} > "$OUT/RESOLVED.md"
echo "resolved: $(grep -cE '^[A-Za-z0-9_.-]+==' "$OUT/requirements.lock") packages in requirements.lock"
