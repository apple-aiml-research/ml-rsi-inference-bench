#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Vendor the engine's FlashAttention fork at its pinned revision into the prepared base.

At the base commit, cmake/external_projects/vllm_flash_attn.cmake fetches the fork from GitHub at
`GIT_TAG dd62dac7` during the build, and vllm/vllm_flash_attn/* is ignored by Git, so the fork's
sources are neither editable in the tree nor available offline. This script makes them both, from
public objects only, and changes nothing else:

  --export DEST [--cmake FILE]
        fetch the fork at FORK_SHA and CUTLASS at CUTLASS_SHA; write the fork's `git archive` into
        DEST (submodule contents are not part of an archive, so no composable_kernel or aiter code)
        and CUTLASS's include/ and LICENSE.txt into DEST/csrc/cutlass/, the only CUTLASS paths the
        fork's FA2 and FA3 targets include; check every exported file against the materials
        manifest (path, mode, git blob id), and the file count; with --cmake, insert the three-line
        default into the engine's CMake file after checking its preimage blob, and check its
        postimage blob. Any mismatch fails.
  --write-materials MANIFEST [--cmake FILE]
        authoring only: the same export into a temporary directory, then write MANIFEST from what
        was exported, with the CMake hunk's preimage and postimage blobs.
  --identifiers --vllm-repo R --fork-repo F --pristine DIR... --out JSON
        authoring only, outside any image: the withheld-identifier list the image's build check
        asserts absent. Every def/class name and every string constant of 12 or more characters on
        a line the landing, the fork's dd62dac7..1c726f3c or an amendment adds, kept iff it occurs
        nowhere in the pristine material (the base tree, site-packages, the checkpoint, the author
        implementation supplied for reading, and the task's own public material, which is how the
        task's supplied names -- the author's class names, the read interface's fields -- drop out).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FORK_URL = "https://github.com/vllm-project/flash-attention.git"
FORK_SHA = "dd62dac706b1cf7895bd99b18c6cb7e7e117ee25"
CUTLASS_URL = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_SHA = "62750a2b75c802660e4894434dc55e839f322277"
CUTLASS_SUBSET = ("include", "LICENSE.txt")
DEST_REL = "third_party/vllm-flash-attn"
CMAKE_REL = "cmake/external_projects/vllm_flash_attn.cmake"
CMAKE_ANCHOR = (
    "if (DEFINED ENV{VLLM_FLASH_ATTN_SRC_DIR})\n"
    "  set(VLLM_FLASH_ATTN_SRC_DIR $ENV{VLLM_FLASH_ATTN_SRC_DIR})\n"
    "endif()\n"
)
CMAKE_ADDED = (
    "if(NOT VLLM_FLASH_ATTN_SRC_DIR AND EXISTS ${CMAKE_SOURCE_DIR}/" + DEST_REL + "/CMakeLists.txt)\n"
    "  set(VLLM_FLASH_ATTN_SRC_DIR ${CMAKE_SOURCE_DIR}/" + DEST_REL + ")\n"
    "endif()\n"
)
SCHEMA = "diffusion-gemma-fa-materials/1"

# Identifier generation (--identifiers): the diffs whose added lines are mined, by repository.
VLLM_BASE = "1ce3cdc5c14f656f81f91e264d557e0b6e6fea54"
VLLM_DIFFS = {
    "landing": (VLLM_BASE, "eb28452b10a1376d143b2847a78b31726db346dd"),
    "amendment-57414": ("2c88fb131c7ae0be01907cd8c276911db5e7aad4^", "2c88fb131c7ae0be01907cd8c276911db5e7aad4"),
    "amendment-57417": ("63d9ad0a3a435cdf3a44495028b10f390a38f960^", "63d9ad0a3a435cdf3a44495028b10f390a38f960"),
    "amendment-57250": ("1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8^", "1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8"),
    "amendment-51994": ("merge-base", "283637a6535d7cfe4b49f32cfe368d9e9fd5602a"),
    "window-reading-49202": ("merge-base", "945f22d0a294f87dde449cddbaa4ecdbc3caadd3"),
}
FORK_DIFFS = {"fork": (FORK_SHA, "1c726f3c48b155aa399e7e47f044cce0beb9fcbb")}
MUST_INCLUDE = ("dynamic_causal", "DiffusionGemmaModelState", "DiffusionGemmaRequestStates",
                "DiffusionSampler", "_compiled_sample_step", "is_encoder_phase")
MIN_STRING = 12


def git(*args: str, cwd: Path | None = None, binary: bool = False):
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
    return out.stdout if binary else out.stdout.decode()


def blob_id(path: Path) -> str:
    data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def text_blob_id(text: str) -> str:
    data = text.encode()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def mode_of(path: Path) -> str:
    if path.is_symlink():
        return "120000"
    return "100755" if os.access(path, os.X_OK) else "100644"


def fetch(url: str, sha: str, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    git("init", "-q", str(into))
    git("-C", str(into), "fetch", "-q", "--depth", "1", url, sha)
    got = git("-C", str(into), "rev-parse", "FETCH_HEAD^{commit}").strip()
    if got != sha:
        raise SystemExit(f"[prepare_fa_vendor] {url} resolved {got}, expected {sha}")
    return into


def extract(repo: Path, sha: str, dest: Path, paths: tuple[str, ...] = ()) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", sha, *paths],
                             check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)


def export(dest: Path, work: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"[prepare_fa_vendor] {dest} is not empty")
    fork = fetch(FORK_URL, FORK_SHA, work / "fork")
    cutlass = fetch(CUTLASS_URL, CUTLASS_SHA, work / "cutlass")
    extract(fork, FORK_SHA, dest)
    extract(cutlass, CUTLASS_SHA, dest / "csrc" / "cutlass", CUTLASS_SUBSET)
    # Submodule gitlinks leave empty directories in an archive; Git does not track them.
    for d in sorted((p for p in dest.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        if not any(d.iterdir()):
            d.rmdir()


def inventory(dest: Path) -> list[dict]:
    rows = []
    for p in sorted(dest.rglob("*")):
        if p.is_dir() and not p.is_symlink():
            continue
        rel = p.relative_to(dest).as_posix()
        source = "cutlass" if rel.startswith("csrc/cutlass/") else "fork"
        rows.append({"path": rel, "source": source, "mode": mode_of(p), "blob": blob_id(p),
                     "size": len(os.readlink(p)) if p.is_symlink() else p.stat().st_size})
    return rows


def patch_cmake(cmake: Path, manifest: dict | None) -> dict:
    text = cmake.read_text()
    pre = text_blob_id(text)
    if manifest and pre != manifest["cmake_hunk"]["pre_blob"]:
        raise SystemExit(f"[prepare_fa_vendor] {cmake}: preimage blob {pre}, expected "
                         f"{manifest['cmake_hunk']['pre_blob']}")
    if text.count(CMAKE_ANCHOR) != 1:
        raise SystemExit(f"[prepare_fa_vendor] {cmake}: the environment-variable block is not unique")
    new = text.replace(CMAKE_ANCHOR, CMAKE_ANCHOR + CMAKE_ADDED)
    post = text_blob_id(new)
    if manifest and post != manifest["cmake_hunk"]["post_blob"]:
        raise SystemExit(f"[prepare_fa_vendor] {cmake}: postimage blob {post}, expected "
                         f"{manifest['cmake_hunk']['post_blob']}")
    cmake.write_text(new)
    return {"path": CMAKE_REL, "pre_blob": pre, "post_blob": post, "after": CMAKE_ANCHOR,
            "added": CMAKE_ADDED, "added_lines": CMAKE_ADDED.count("\n")}


def check_against(rows: list[dict], manifest: dict) -> None:
    want = {r["path"]: r for r in manifest["files"]}
    got = {r["path"]: r for r in rows}
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    wrong = sorted(p for p in set(want) & set(got)
                   if (want[p]["blob"], want[p]["mode"]) != (got[p]["blob"], got[p]["mode"]))
    if missing or extra or wrong:
        raise SystemExit(f"[prepare_fa_vendor] materials mismatch: missing {missing[:10]} "
                         f"({len(missing)}), extra {extra[:10]} ({len(extra)}), wrong {wrong[:10]} ({len(wrong)})")
    counts = {s: sum(r["source"] == s for r in rows) for s in ("fork", "cutlass")}
    if counts != {"fork": manifest["fork"]["files"], "cutlass": manifest["cutlass"]["files"]}:
        raise SystemExit(f"[prepare_fa_vendor] file counts {counts} differ from the manifest")


# ------------------------------------------------------------------ withheld identifiers (authoring)
DEF_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
STR_RE = re.compile(r"""(?<![A-Za-z0-9_])[rbuRBU]{0,2}(?:"([^"\\\n]*(?:\\.[^"\\\n]*)*)"|'([^'\\\n]*(?:\\.[^'\\\n]*)*)')""")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def added_lines(repo: Path, a: str, b: str) -> list[tuple[str, str]]:
    if a == "merge-base":
        base = git("-C", str(repo), "rev-parse", "--verify", "-q", "origin/main").strip() or "HEAD"
        a = git("-C", str(repo), "merge-base", base, b).strip()
    diff = git("-C", str(repo), "diff", "--no-ext-diff", "-U0", a, b, "--", "*.py")
    rows, path = [], None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("+") and path:
            rows.append((path, line[1:]))
    return rows


def candidates(rows: list[tuple[str, str]]) -> dict[str, set[str]]:
    names, strings = set(), set()
    for _, line in rows:
        m = DEF_RE.match(line)
        if m:
            names.add(m.group(1))
        for sm in STR_RE.finditer(line):
            s = sm.group(1) if sm.group(1) is not None else sm.group(2)
            if s is not None and len(s) >= MIN_STRING and "{" not in s:
                strings.add(s)
    return {"names": names, "strings": strings}


def scan_text(path: Path, limit: int = 64 << 20) -> str | None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def pristine_filter(names: set[str], strings: set[str], roots: list[Path]) -> tuple[set[str], set[str], int]:
    words: set[str] = set()
    alive = set(strings)
    files = 0
    for root in roots:
        for p in (root.rglob("*") if root.is_dir() else [root]):
            text = scan_text(p)
            if text is None:
                continue
            files += 1
            words.update(WORD_RE.findall(text))
            if alive:
                alive = {s for s in alive if s not in text}
    return names - words, alive, files


# The objects whose exact bytes are withheld: every file content a withheld change produces (its
# post-image blobs), minus any blob the prepared image legitimately holds -- every blob reachable from
# the base (its whole history, which the image keeps) and every blob of the vendored fork -- so neither
# a coincidence (an empty __init__.py) nor an older file version (a PR head built on an older main
# re-creates them) can fail the audit.
VLLM_BLOB_SOURCES = dict(VLLM_DIFFS, **{
    "repair-45468": ("521b88c29ef29b37efefa64efda7b59ec99d210c^", "521b88c29ef29b37efefa64efda7b59ec99d210c"),
    "repair-45417": ("6c5872efc5fca7a53f5f3aa589e8cb7896d98618^", "6c5872efc5fca7a53f5f3aa589e8cb7896d98618"),
    "repair-46177": ("701a23d99f405668158d1395e11c30107dd65b75^", "701a23d99f405668158d1395e11c30107dd65b75"),
    "repair-45965": ("b136cc2c2c590593c9dfff2bd732edbb24bcceeb^", "b136cc2c2c590593c9dfff2bd732edbb24bcceeb"),
    "repair-47464": ("c5b66233b26824f463fcca8af0779ff9e9d7930c^", "c5b66233b26824f463fcca8af0779ff9e9d7930c"),
    "repair-46659": ("8fc1b2d046f4a991b5c24bb5470bf45efcfe9d01^", "8fc1b2d046f4a991b5c24bb5470bf45efcfe9d01"),
    "landing-pr-head": (VLLM_BASE, "d42ae12f36d5a9136815832fc108d0f6873320e8"),
})
FORK_BLOB_SOURCES = {"fork": (FORK_SHA, "1c726f3c48b155aa399e7e47f044cce0beb9fcbb"),
                     "fork-landing-pin": (FORK_SHA, "803020a8fa15407871341d41eba4919ade2ee1ee")}


def post_image_blobs(repo: Path, a: str, b: str) -> dict[str, str]:
    if a == "merge-base":
        base = git("-C", str(repo), "rev-parse", "--verify", "-q", "origin/main").strip() or "HEAD"
        a = git("-C", str(repo), "merge-base", base, b).strip()
    out = {}
    tokens = git("-C", str(repo), "diff", "--no-ext-diff", "--raw", "--no-abbrev", "--no-renames", "-z", a, b).split("\0")
    i = 0
    while i + 1 < len(tokens) and tokens[i].startswith(":"):
        _, _, _, new_blob, status = tokens[i][1:].split()
        if status != "D" and new_blob != "0" * 40:
            out[new_blob] = tokens[i + 1]
        i += 2
    return out


def reachable_blobs(repo: Path, rev: str) -> set[str]:
    out = git("-C", str(repo), "rev-list", "--objects", "--no-object-names", "--filter=object:type=blob", rev)
    return {line.strip() for line in out.splitlines() if line.strip()}


def withheld_blobs(vllm: Path, fork: Path) -> tuple[dict[str, dict], int]:
    present = reachable_blobs(vllm, VLLM_BASE)
    present |= {line.split()[2] for line in git("-C", str(fork), "ls-tree", "-r", FORK_SHA).splitlines()}
    blobs: dict[str, dict] = {}
    for repo, sources in ((vllm, VLLM_BLOB_SOURCES), (fork, FORK_BLOB_SOURCES)):
        for label, (a, b) in sources.items():
            for blob, path in post_image_blobs(repo, a, b).items():
                if blob not in present:
                    blobs.setdefault(blob, {"path": path, "sources": []})["sources"].append(label)
    return blobs, len(present)


def identifiers(args) -> int:
    rows_by_source = {}
    for label, (a, b) in VLLM_DIFFS.items():
        rows_by_source[label] = added_lines(args.vllm_repo, a, b)
    for label, (a, b) in FORK_DIFFS.items():
        rows_by_source[label] = added_lines(args.fork_repo, a, b)
    names, strings = set(), set()
    per_source = {}
    for label, rows in rows_by_source.items():
        c = candidates(rows)
        per_source[label] = {"added_python_lines": len(rows), "names": len(c["names"]), "strings": len(c["strings"])}
        names |= c["names"]
        strings |= c["strings"]
    keep_names, keep_strings, files = pristine_filter(names, strings, [Path(r) for r in args.pristine])
    # The named identifiers join the rule's output (most are parameter or attribute names, which the
    # def/class rule does not see); one that occurs in the pristine material is a contradiction.
    forced, _, _ = pristine_filter(set(MUST_INCLUDE), set(), [Path(r) for r in args.pristine])
    missing = [m for m in MUST_INCLUDE if m not in forced]
    keep_names |= forced
    blobs, present = withheld_blobs(args.vllm_repo, args.fork_repo)
    record = {
        "schema": "diffusion-gemma-withheld-identifiers/1",
        "rule": "def/class names and string constants of >= 12 characters on lines added by the landing, "
                "the fork's dd62dac7..1c726f3c and the amendments; kept iff absent from every pristine root",
        "diffs": {**{k: list(v) for k, v in VLLM_DIFFS.items()}, **{k: list(v) for k, v in FORK_DIFFS.items()}},
        "per_source": per_source,
        "pristine_roots": [str(r) for r in args.pristine],
        "pristine_text_files_scanned": files,
        "candidates": {"names": len(names), "strings": len(strings)},
        "must_include": list(MUST_INCLUDE),
        "must_include_missing": missing,
        "names": sorted(keep_names),
        "strings": sorted(keep_strings),
        "blob_rule": "post-image blobs of the landing, its PR head, the repairs, the amendments, the window "
                     "reading and the fork's dd62dac7..803020a8 and dd62dac7..1c726f3c, minus every blob reachable "
                     "from the base (its whole history) and every blob of the vendored fork",
        "blob_sources": {**{k: list(v) for k, v in VLLM_BLOB_SOURCES.items()},
                         **{k: list(v) for k, v in FORK_BLOB_SOURCES.items()}},
        "blobs_excluded_as_present": present,
        "blobs": {k: blobs[k] for k in sorted(blobs)},
    }
    Path(args.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"[prepare_fa_vendor] identifiers: {len(keep_names)} names, {len(keep_strings)} strings "
          f"from {len(names)}/{len(strings)} candidates; {files} pristine files; missing {missing}; "
          f"{len(blobs)} withheld blobs")
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", type=Path, metavar="DEST")
    mode.add_argument("--write-materials", type=Path, metavar="MANIFEST")
    mode.add_argument("--identifiers", action="store_true")
    ap.add_argument("--materials", type=Path, default=Path(__file__).with_name("fa-materials.json"))
    ap.add_argument("--cmake", type=Path)
    ap.add_argument("--vllm-repo", type=Path)
    ap.add_argument("--fork-repo", type=Path)
    ap.add_argument("--pristine", nargs="*", default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.identifiers:
        if not (args.vllm_repo and args.fork_repo and args.pristine and args.out):
            ap.error("--identifiers needs --vllm-repo, --fork-repo, --pristine and --out")
        return identifiers(args)

    with tempfile.TemporaryDirectory(prefix="fa-vendor-") as tmp:
        work = Path(tmp)
        if args.write_materials:
            dest = work / "export"
            export(dest, work / "src")
            rows = inventory(dest)
            hunk = None
            if args.cmake:
                probe = work / "cmake-probe"
                shutil.copyfile(args.cmake, probe)
                hunk = patch_cmake(probe, None)
            manifest = {
                "schema": SCHEMA,
                "destination": DEST_REL,
                "fork": {"repository": FORK_URL, "revision": FORK_SHA, "operation": "git archive (no submodule contents)",
                         "files": sum(r["source"] == "fork" for r in rows),
                         "bytes": sum(r["size"] for r in rows if r["source"] == "fork")},
                "cutlass": {"repository": CUTLASS_URL, "revision": CUTLASS_SHA, "operation": "header subset",
                            "subset": list(CUTLASS_SUBSET), "destination": DEST_REL + "/csrc/cutlass",
                            "files": sum(r["source"] == "cutlass" for r in rows),
                            "bytes": sum(r["size"] for r in rows if r["source"] == "cutlass")},
                "cmake_hunk": hunk,
                "files": rows,
            }
            args.write_materials.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
            print(f"[prepare_fa_vendor] wrote {args.write_materials}: fork {manifest['fork']['files']} files / "
                  f"{manifest['fork']['bytes']} B, cutlass {manifest['cutlass']['files']} files / "
                  f"{manifest['cutlass']['bytes']} B")
            return 0

        manifest = json.loads(args.materials.read_text())
        if manifest.get("schema") != SCHEMA:
            raise SystemExit(f"[prepare_fa_vendor] {args.materials}: unknown schema {manifest.get('schema')}")
        export(args.export, work / "src")
        check_against(inventory(args.export), manifest)
        # The fork at the engine's pin predates its per-sequence phases; the check is the manifest's
        # blob identity above, and this names the property it guarantees.
        phased = [str(p) for p in args.export.rglob("*") if p.is_file() and not p.is_symlink()
                  and b"dynamic_causal" in p.read_bytes()]
        if phased:
            raise SystemExit(f"[prepare_fa_vendor] the vendored tree carries per-sequence phases: {phased[:5]}")
        print(f"[prepare_fa_vendor] exported {args.export}: fork {manifest['fork']['files']} files, "
              f"cutlass {manifest['cutlass']['files']} files, every blob as the manifest records")
        if args.cmake:
            patch_cmake(args.cmake, manifest)
            print(f"[prepare_fa_vendor] {args.cmake}: three-line default added after the environment block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
