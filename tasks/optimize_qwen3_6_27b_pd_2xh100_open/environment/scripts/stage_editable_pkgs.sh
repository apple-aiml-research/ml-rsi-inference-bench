#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

infb_stage_editable_pkgs() {
    local target_dir="$1" venv_dir="$2"
    local pkgs=",${INFB_EDIT_POLICY_PKGS:-},"
    [[ "${pkgs}" == ",," && "${INFB_EDIT_POLICY_FA_REBUILD:-}" != "1" \
       && "${INFB_EDIT_POLICY_CSRC_DEPS:-}" != "1" ]] && return 0

    local venv_vllm; venv_vllm="$(echo "${venv_dir}"/lib/python*/site-packages/vllm 2>/dev/null | head -1)"
    local venv_site; venv_site="$(echo "${venv_dir}"/lib/python*/site-packages 2>/dev/null | head -1)"
    local changed=0

    _infb_open_intree() {   # $1=surface  $2=repo-rel path (under repo)  $3=venv abs src
        local s="$1" rel="$2" src="$3" dst="${target_dir}/$2"
        [[ "${pkgs}" != *",${s},"* ]] && return 0
        if [[ ! -d "${src}" ]]; then
            echo "[stage_editable_pkgs] FATAL: ${s} is listed in INFB_EDIT_POLICY_PKGS but its venv source is missing: ${src}. The task declares repo/${rel} editable+scored; refusing to grade a surface that was never opened." >&2; return 93
        fi
        sed -i -E "\\#^/${rel}/?\$#d" "${target_dir}/.gitignore" 2>/dev/null || true
        [[ -e "${dst}" || -L "${dst}" ]] && rm -rf "${dst}"
        mkdir -p "$(dirname "${dst}")"
        cp -aL "${src}" "${dst}"
        find "${dst}" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
        find "${dst}" -type f -name '*.pyc' -delete 2>/dev/null || true
        printf '/%s/**/__pycache__/\n/%s/**/*.pyc\n' "${rel}" "${rel}" >> "${target_dir}/.gitignore"
        git -C "${target_dir}" add -f "${rel}" .gitignore   # -f: bypass any residual gitignore
        changed=1
        echo "[stage_editable_pkgs] OPENED ${s} → editable real copy at ${dst}"
    }

    _infb_open_toplevel() {  # $1=surface  $2=pkg dir name (top-level in repo)  $3=venv abs src
        local s="$1" name="$2" src="$3" dst="${target_dir}/$2"
        [[ "${pkgs}" != *",${s},"* ]] && return 0
        if [[ ! -d "${src}" ]]; then
            echo "[stage_editable_pkgs] FATAL: ${s} is listed in INFB_EDIT_POLICY_PKGS but its venv source is missing: ${src}. See the note above _infb_open_intree." >&2; return 94
        fi
        [[ -e "${dst}" ]] && return 0
        cp -aL "${src}" "${dst}"
        find "${dst}" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
        find "${dst}" -type f -name '*.pyc' -delete 2>/dev/null || true
        printf '/%s/**/__pycache__/\n/%s/**/*.pyc\n' "${name}" "${name}" >> "${target_dir}/.gitignore"
        git -C "${target_dir}" add "${name}" .gitignore
        changed=1
        echo "[stage_editable_pkgs] OPENED ${s} → editable real copy at ${dst}"
    }

    _infb_open_intree  cute       "vllm/vllm_flash_attn/cute"   "${venv_vllm}/vllm_flash_attn/cute" \
        || return $?
    _infb_open_intree  deep_gemm  "vllm/third_party/deep_gemm"  "${venv_vllm}/third_party/deep_gemm" \
        || return $?
    _infb_open_toplevel flashinfer "flashinfer"                 "${venv_site}/flashinfer" \
        || return $?

    _infb_open_fa() {
        [[ "${INFB_EDIT_POLICY_FA_REBUILD:-}" == "1" ]] || return 0
        local dst="${target_dir}/.fa_src"
        local _fa_img="${INFB_FA_SRC_IMAGE_DIR:-/opt/csrc-deps/flash-attention}"
        if [[ -d "${dst}" && -n "$(ls -A "${dst}" 2>/dev/null)" ]]; then
            :  # already staged (idempotent)
        elif [[ -n "${INFB_FA_SRC_LOCAL:-}" && -f "${INFB_FA_SRC_LOCAL}" ]]; then
            echo "[stage_editable_pkgs] extracting FA submodule source from mounted ${INFB_FA_SRC_LOCAL} → ${dst}"
            mkdir -p "${dst}"
            if ! tar xzf "${INFB_FA_SRC_LOCAL}" -C "${dst}" --strip-components=1 2>/dev/null; then
                rm -rf "${dst}"
                echo "[stage_editable_pkgs] FATAL: INFB_EDIT_POLICY_FA_REBUILD=1 but the local FA tarball failed to extract (${INFB_FA_SRC_LOCAL}). Refusing to continue: the task declares repo/.fa_src editable+scored, so a run without it would bench baseline-vs-baseline and publish ~1.0 as an FA-open result." >&2
                return 90
            fi
        elif [[ -d "${_fa_img}" && -n "$(ls -A "${_fa_img}" 2>/dev/null)" ]]; then
            echo "[stage_editable_pkgs] copying FA submodule source from the image: ${_fa_img} → ${dst} (contract pin INFB_FA_SRC_COMMIT=${INFB_FA_SRC_COMMIT:-<unset>})"
            mkdir -p "${dst}"
            if ! cp -a "${_fa_img}/." "${dst}/" 2>/dev/null; then
                rm -rf "${dst}"
                echo "[stage_editable_pkgs] FATAL: INFB_EDIT_POLICY_FA_REBUILD=1 but copying the image's FA source failed (${_fa_img} → ${dst}). Refusing to continue — see the note above _infb_open_fa." >&2
                return 91
            fi
            rm -f "${dst}/.staged"   # a build-time marker, not part of the source tree
        else
            echo "[stage_editable_pkgs] FATAL: INFB_EDIT_POLICY_FA_REBUILD=1 but repo/.fa_src does not exist, INFB_FA_SRC_LOCAL is unset, and the image has no FA source at ${_fa_img}. This task declares the FA surface open but has no source to open — fix the wiring rather than grading an unopened surface. The source is normally BAKED INTO THE IMAGE at build time: a checkout of https://github.com/vllm-project/flash-attention at the commit this vLLM's CMake GIT_TAG pins, which the contract also carries as INFB_FA_SRC_COMMIT=${INFB_FA_SRC_COMMIT:-<unset>}. If you build your own image, stage that commit at ${_fa_img} (set INFB_FA_SRC_IMAGE_DIR to move it), or mount a tarball of it and point INFB_FA_SRC_LOCAL at the file." >&2
            return 92
        fi
        sed -i -E "\\#^/\\.fa_src/?\$#d" "${target_dir}/.gitignore" 2>/dev/null || true
        find "${dst}" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
        find "${dst}" -type f -name '*.pyc' -delete 2>/dev/null || true
        printf '/.fa_src/**/__pycache__/\n/.fa_src/**/*.pyc\n' >> "${target_dir}/.gitignore"
        rm -rf "${dst}/.git" 2>/dev/null || true
        git -C "${target_dir}" add -f .fa_src .gitignore
        changed=1
        echo "[stage_editable_pkgs] OPENED fa → editable FA source at ${dst} (INFB_FA_SRC_DIR defaults to repo/.fa_src)"
    }
    _infb_open_fa
    local _fa_rc=$?
    if [[ "${_fa_rc}" -ne 0 ]]; then
        echo "[stage_editable_pkgs] aborting: FA surface declared open but not staged (rc=${_fa_rc})" >&2
        return "${_fa_rc}"
    fi

    _infb_open_csrc_deps() {
        [[ "${INFB_EDIT_POLICY_CSRC_DEPS:-}" == "1" ]] || return 0
        local root="${INFB_CSRC_DEPS_DIR:-/opt/csrc-deps}"
        local manifest="${root}/knobs.tsv"
        if [[ ! -s "${manifest}" ]]; then
            echo "[stage_editable_pkgs] WARN: INFB_EDIT_POLICY_CSRC_DEPS=1 but this image has no ${manifest}. repo/.csrc_src was NOT staged — the CMake knobs will fall back to whatever the image has, which is NOT editable and NOT in the diff. The image is older than the configuration that set this flag." >&2
            return 0
        fi
        local dst_root="${target_dir}/.csrc_src"
        local name knob subdir declared src dst staged=0 skipped=0
        while IFS=$'\t' read -r name knob subdir declared; do
            [[ -n "${name}" && -n "${knob}" ]] || continue
            [[ "${name}" == "flash-attention" ]] && continue
            src="${root}/${name}"
            dst="${dst_root}/${name}"
            if [[ ! -d "${src}" ]]; then
                skipped=$((skipped + 1)); continue
            fi
            if [[ -d "${dst}" && -n "$(ls -A "${dst}" 2>/dev/null)" ]]; then
                staged=$((staged + 1)); continue   # idempotent
            fi
            mkdir -p "${dst}"
            if ! cp -a "${src}/." "${dst}/" 2>/dev/null; then
                rm -rf "${dst}"
                echo "[stage_editable_pkgs] FATAL: INFB_EDIT_POLICY_CSRC_DEPS=1 and ${src} exists, but copying it to ${dst} failed. Refusing to continue: the instruction names repo/.csrc_src/${name} editable+scored, and a run that skipped it would grade an unopened surface." >&2
                return 95
            fi
            rm -rf "${dst}/.git" 2>/dev/null || true
            staged=$((staged + 1))
        done < "${manifest}"
        [[ "${staged}" -gt 0 ]] || { echo "[stage_editable_pkgs] csrc deps: manifest listed no tree present in this image — .csrc_src not staged" >&2; return 0; }
        find "${dst_root}" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
        find "${dst_root}" -type f -name '*.pyc' -delete 2>/dev/null || true
        sed -i -E "\\#^/\\.csrc_src/?\$#d" "${target_dir}/.gitignore" 2>/dev/null || true
        printf '/.csrc_src/**/__pycache__/\n/.csrc_src/**/*.pyc\n' >> "${target_dir}/.gitignore"
        git -C "${target_dir}" add -f .csrc_src .gitignore
        changed=1
        echo "[stage_editable_pkgs] OPENED csrc deps → ${staged} editable tree(s) at ${dst_root} (${skipped} listed but absent from the image)"
    }
    _infb_open_csrc_deps
    local _deps_rc=$?
    if [[ "${_deps_rc}" -ne 0 ]]; then
        echo "[stage_editable_pkgs] aborting: csrc dep trees declared open but not staged (rc=${_deps_rc})" >&2
        return "${_deps_rc}"
    fi

    if [[ "${changed}" == 1 ]]; then
        git -C "${target_dir}" -c gc.auto=0 -c user.name=infb -c user.email=infb@local commit -q --amend --no-edit
        git -C "${target_dir}" tag -f inference_base >/dev/null
        echo "[stage_editable_pkgs] folded opened surface(s) into inference_base"
    fi
}
