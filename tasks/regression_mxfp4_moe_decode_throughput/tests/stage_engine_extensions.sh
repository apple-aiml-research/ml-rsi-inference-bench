#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

log() { printf '[engine-ext] %s\n' "$*"; }
die() { printf '[engine-ext] FATAL: %s\n' "$*" >&2; exit 1; }

ext_default="${INFB_VLLM_ENGINE_EXT:-/opt/inference-bench/engine-ext}"

_site_packages() {
  local venv="$1"
  local sp
  sp="$("${venv}/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])' 2>/dev/null)" \
    || die "cannot run ${venv}/bin/python"
  [[ -n "${sp}" ]] || die "could not resolve site-packages for ${venv}"
  printf '%s\n' "${sp}"
}

harvest() {
  local venv="$1" ext="${2:-${ext_default}}"
  [[ -x "${venv}/bin/python" ]] || die "no interpreter at ${venv}/bin/python"

  local site pkg
  site="$(_site_packages "${venv}")"
  pkg="${site}/sglang"
  [[ -d "${pkg}" ]] || die "no engine package at ${pkg} -- nothing to harvest"

  rm -rf "${ext}"
  mkdir -p "${ext}/sglang"

  local n=0 rel
  while IFS= read -r rel; do
    mkdir -p "${ext}/sglang/$(dirname "${rel}")"
    cp -a "${pkg}/${rel}" "${ext}/sglang/${rel}"
    n=$((n + 1))
  done < <(
    cd "${pkg}"
    { find . -name '*.so' -type f -printf '%P\n'
      [[ -f _version.py ]] && printf '_version.py\n'
    } | sort -u
  )
  log "preserved ${n} build-time artefact(s) into ${ext}"

  if ! PYTHONPATH= "${venv}/bin/python" -c \
       "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('sgl_kernel') else 1)" \
       >/dev/null 2>&1; then
    die "sgl_kernel is not installed in this venv. On this engine version the compiled half does NOT live inside the engine distribution, so nothing the harvest does can supply it -- check the install step, not this function."
  fi
  local n_so
  n_so="$(find "${ext}/sglang" -name '*.so' -type f 2>/dev/null | wc -l)"
  log "compiled half: sgl_kernel imports; ${n_so} .so inside the engine package itself"

  PYTHONPATH= "${venv}/bin/python" - <<'PY' || die "triton_kernels is not importable from this venv"
import importlib.util, sys
spec = importlib.util.find_spec("triton_kernels")
if spec is None:
    sys.exit(1)
print(f"[engine-ext] triton_kernels resolves to {spec.origin}")
PY

  local ver
  ver="$("${venv}/bin/python" -c 'import importlib.metadata as m; print(m.version("sglang"))' 2>/dev/null || true)"

  "${venv}/bin/pip" uninstall -y sglang >/dev/null
  rm -rf "${pkg}"

  if [[ -n "${ver}" ]]; then
    local di="${site}/sglang-${ver}.dist-info"
    mkdir -p "${di}"
    printf 'Metadata-Version: 2.1\nName: sglang\nVersion: %s\n' "${ver}" > "${di}/METADATA"
    log "kept a METADATA-ONLY dist-info at ${di} (version ${ver}) so platform detection can still"
    log "  read a version; it contains no Python and no RECORD"
  else
    log "WARNING: could not read the engine version before eviction; platform detection may fail"
  fi

  [[ ! -e "${pkg}" ]] || die "${pkg} still exists after eviction"
  local strays
  strays="$(find "${site}" -maxdepth 1 \( -name 'sglang' -o -name 'sglang-*' \) \
                 -not -name 'sglang-*.dist-info' -print 2>/dev/null || true)"
  [[ -z "${strays}" ]] || die "engine artefacts remain at the top of ${site}: ${strays}"
  local di_py
  di_py="$(find "${site}" -maxdepth 2 -path '*/sglang-*.dist-info/*' \( -name '*.py' -o -name '*.so' \) -print 2>/dev/null || true)"
  [[ -z "${di_py}" ]] || die "the preserved dist-info carries code, not just metadata: ${di_py}"
  if PYTHONPATH= "${venv}/bin/python" -c \
       "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('sglang') else 1)" \
       >/dev/null 2>&1; then
    die "the venv can still resolve an 'sglang' package on its own -- a second copy of the engine survives"
  fi
  log "evicted the engine distribution; this venv can no longer import an engine of its own"
}

overlay() {
  local repo="$1" ext="${2:-${ext_default}}"
  [[ -d "${repo}/python/sglang" ]] || die "no engine package at ${repo}/python/sglang"
  if [[ ! -d "${ext}/sglang" ]]; then
    log "WARNING: no harvested artefacts at ${ext}; ${repo} will report a fallback version string"
    return 0
  fi

  local n=0 rel
  while IFS= read -r rel; do
    mkdir -p "${repo}/python/sglang/$(dirname "${rel}")"
    ln -sfn "${ext}/sglang/${rel}" "${repo}/python/sglang/${rel}"
    n=$((n + 1))
  done < <(cd "${ext}/sglang" && find . -type f -printf '%P\n' | sort)
  log "linked ${n} compiled artefact(s) from ${ext} into ${repo}"
}

case "${1:-}" in
  harvest) shift; [[ $# -ge 1 ]] || die "usage: $0 harvest <venv> [<ext-dir>]"; harvest "$@" ;;
  overlay) shift; [[ $# -ge 1 ]] || die "usage: $0 overlay <repo> [<ext-dir>]"; overlay "$@" ;;
  *) die "usage: $0 {harvest <venv>|overlay <repo>} [<ext-dir>]" ;;
esac
