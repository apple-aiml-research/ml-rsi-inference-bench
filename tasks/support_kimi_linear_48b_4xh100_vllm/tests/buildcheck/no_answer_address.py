#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if the answer's address is on the agent's surface.

This check makes it a build assertion rather than a claim:

  * no task-authored file the image ships names the landing squash, the landing PR by number, or its
    author -- `/opt/infb` (minus its pip venv of public packages), the evaluation tree `/app/eval`, and the
    checkpoint's own text files, which are the model author's and would be a finding to report, not
    something to edit;
  * the same for every post-merge PR whose code the reference may carry, and their authors;
  * HEAD's commit message and identity name none of them. HEAD is a constructed clean root commit with a neutral message and identity;
  * the landing squash's OBJECT is not in /app's store, so it cannot be recovered from history either
    (the source stage exports the adapted tree into a fresh object database).

Third-party venvs are not scanned for PR numbers: those occur by chance in wheel data.
They are scanned for the withheld implementation's definitions, bytecode and compiled symbols.

This file is BIND-MOUNTED for the one build step that runs it and is never copied into the image: it
has to carry the very strings it forbids, so a copy of it in /opt would be the leak it guards.
"""

from __future__ import annotations

import re
import hashlib
import json
import marshal
import subprocess
import sys
import types
import zipfile
from pathlib import Path

LANDING = "4e68cc9b6aa2b9cfe8d799c2b1cd156a01bca438"
#: The landing, and every post-merge PR whose code the closure may carry
#: (solution/oracle.lock: criticality_candidates). Authors read from the GitHub API.
PR_NUMBERS = (27654, 27809, 27834, 27871, 27905, 27811, 27322, 28445, 28579, 51565, 30877, 29387, 33706, 34418)
PR_AUTHORS = ("zhiyuan1i", "ZJY0516", "jeejeelee", "toncao", "LucasWilkinson", "taking-lying-flat", "cherry77-cloud",
              "peakcrosser7", "heheda12345", "haosdent")
CLOSURE_COMMITS = ("e5ef4dfc", "3857eb87", "3a5de7d2", "fa183e92",
                   "56539cdd", "79468c20", "5206e5e2", "97fa8f65", "bf37812c", "4137c5df")
PATTERNS: dict[str, re.Pattern[str]] = {
    "landing squash": re.compile(r"4e68cc9b"),
    "kernel squash": re.compile(r"e88bdd60"),
    "closure commits": re.compile("|".join(CLOSURE_COMMITS)),
    **{
        f"PR #{n} by number": re.compile(rf"(?:#|\bPR\s*#?|pull/|pulls/){n}\b", re.IGNORECASE)
        for n in PR_NUMBERS
    },
    "a PR author": re.compile("|".join(PR_AUTHORS), re.IGNORECASE),
}
ROOTS = [Path("/opt/infb"), Path("/app/eval"), Path("/model")]
SKIP = [Path("/opt/infb/tools"),  # the fetch/lock tooling venv: public packages only
        # the engine stage's compiler cache holds binary compiler outputs, not task-authored text; read as
        # text (invalid bytes dropped), a 2.7 MB entry matched a PR number across bytes that are not text at all.
        Path("/opt/infb/ccache")]
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".py", ".sh", ".toml", ".yaml", ".yml", ".lock",
                 ".patch", ".cfg", ".ini", ".jinja", ".model", ""}
MAX_BYTES = 64 * 1024 * 1024  # the checkpoint's shards are skipped by suffix; this bounds the rest
TARGET_NAMES = rb'(?:chunk_kda\w*|fused_recurrent_kda\w*|(?:fused_)?kda_gate\w*|FusedRMSNormGated|layer_norm_gated_fwd_kernel1?)'
TARGET_DEFINITION = re.compile(rb'\b(?:def|class)\s+' + TARGET_NAMES + rb'\b')
TARGET_BINARY = re.compile(TARGET_NAMES)
#: The verifier's private KDA package -- its correctness schedule, binding interventions, GDN
#: case inventory and family timing loop -- must not be on the agent's surface, in a shipped task-authored file or in
#: any object of /app's store, so the public/private split cannot be undone from the image. The object store holds the
#: whole engine too, so it is scanned for the benchmark-unique names only; the benchmark-authored trees for all.
PRIVATE_UNIQUE = re.compile(r"infb_kda_correctness|infb_kda_binding|infb_kda_private|case_to_layer|measure_families|"
                            r"decode_continuation|"
                            # the private state-lifecycle observer, its hidden manifest and its producers
                            r"infb_kda_lifecycle|infb_kda_metadata|state-lifecycle\.json|_select_resumed|"
                            r"teacher_force_all|turnover_round")
PRIVATE_AUTHORED = re.compile(PRIVATE_UNIQUE.pattern + r"|split_prefill|repeated_decode|state_add|install_startup_hook|"
                              r"interventions")
AUTHORED_ROOTS = [Path("/opt/infb"), Path("/app/eval")]
ARTIFACT_ROOTS = (Path('/app'), Path('/opt/vllm'), Path('/opt/infb'),
                  Path('/root/.cache'), Path('/root/.triton'), Path('/tmp'), Path('/hf/modules'))
BINARY_SUFFIXES = {'.so', '.a', '.o', '.cubin', '.ptx', '.ttir', '.ttgir', '.llir', '.bc'}


def bytecode_has_target(body: bytes) -> bool:
    """Inspect definitions, not imports of names in the declared model-author source."""
    root = marshal.loads(body[16:])
    pending = [root]
    while pending:
        code = pending.pop()
        if isinstance(code, types.CodeType):
            if TARGET_BINARY.fullmatch(code.co_name.encode()):
                return True
            pending.extend(code.co_consts)
    return False


def scan_artifact(path: Path) -> list[str]:
    """Known source, bytecode, GPU artifacts and source/wheel archives in the final image.

    Ordinary C++ ccache entries are retained; the exact prepared source and unchanged
    extension inputs establish their provenance. Runtime KDA is Python/Triton, so its
    bytecode and GPU cache formats are checked explicitly, including renamed files.
    """
    suffix = path.suffix
    try:
        if suffix in {'.py', '.pyi', '.patch', '.diff'}:
            return [f'{path}: target KDA definition'] if TARGET_DEFINITION.search(path.read_bytes()) else []
        if suffix == '.pyc':
            return [f'{path}: target KDA bytecode'] if bytecode_has_target(path.read_bytes()) else []
        if suffix in {'.zip', '.whl'}:
            hits = []
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if Path(member.filename).suffix not in {'.py', '.pyi', '.pyc'}:
                        continue
                    if member.file_size > MAX_BYTES:
                        hits.append(f'{path}!{member.filename}: source member exceeds audit bound')
                        continue
                    body = archive.read(member)
                    found = (bytecode_has_target(body) if member.filename.endswith('.pyc')
                             else bool(TARGET_DEFINITION.search(body)))
                    if found:
                        hits.append(f'{path}!{member.filename}: target KDA source/bytecode')
            return hits
        if suffix in BINARY_SUFFIXES or '.so.' in path.name:
            with path.open('rb') as stream:
                tail = b''
                for block in iter(lambda: stream.read(4 << 20), b''):
                    if TARGET_BINARY.search(tail + block):
                        return [f'{path}: target KDA symbol in compiled artifact']
                    tail = block[-256:]
    except (OSError, EOFError, ValueError, TypeError, zipfile.BadZipFile) as exc:
        return [f'{path}: could not audit {type(exc).__name__}: {exc}']
    return []


def check_declared_math(root: Path) -> list[str]:
    """The only declared FLA extraction is pinned source, not a directory exemption."""
    manifest = json.loads((root / 'source-manifest.json').read_text())
    hits = []
    actual = {p.name for p in root.iterdir() if p.is_file()} - {'source-manifest.json'}
    if actual != set(manifest):
        hits.append(f'{root}: mathematical-source manifest does not match its files')
    for name, digest in manifest.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            hits.append(f'{path}: mathematical-source digest mismatch')
    return hits


def scan_file(path: Path) -> list[str]:
    try:
        if path.stat().st_size > MAX_BYTES:
            return []
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    hits = [f"{path}: {name}" for name, rx in PATTERNS.items() if rx.search(text)]
    private = PRIVATE_AUTHORED if any(path.is_relative_to(r) for r in AUTHORED_ROOTS) else PRIVATE_UNIQUE
    if private.search(text):
        hits.append(f"{path}: private verifier material ({private.search(text).group(0)})")
    return hits


def main() -> int:
    hits: list[str] = []
    scanned = 0
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if any(path.is_relative_to(s) for s in SKIP) or not path.is_file():
                continue
            if path.suffix not in TEXT_SUFFIXES:
                continue
            scanned += 1
            hits += scan_file(path)

    head = subprocess.run(
        ["git", "-C", "/app", "log", "-1", "--format=%an%n%ae%n%cn%n%ce%n%B"],
        capture_output=True, text=True, check=True,
    ).stdout
    hits += [f"HEAD commit: {n}" for n, rx in PATTERNS.items() if rx.search(head)]

    present = subprocess.run(
        ["git", "-C", "/app", "cat-file", "-e", f"{LANDING}^{{commit}}"], capture_output=True
    ).returncode == 0
    if present:
        hits.append(f"/app/.git: the landing squash {LANDING[:12]} is still in the object store")

    # There is one freshly constructed root commit. Check all objects, including
    # unreachable residue, rather than trusting the visible branch alone.
    count = subprocess.check_output(['git', '-C', '/app', 'rev-list', '--all', '--count'], text=True).strip()
    if count != '1':
        hits.append(f'/app/.git: expected one clean root commit, found {count}')
    residue = subprocess.run(['git', '-C', '/app', 'fsck', '--full', '--no-reflogs', '--unreachable'],
                             capture_output=True, text=True, check=True).stdout.strip()
    if residue:
        hits.append(f'/app/.git: unexpected object residue: {residue[:500]}')
    objects = subprocess.check_output(['git', '-C', '/app', 'cat-file', '--batch-all-objects',
                                      '--batch-check=%(objectname) %(objecttype)'], text=True)
    for line in objects.splitlines():
        oid, kind = line.split()
        if kind == 'blob':
            body = subprocess.check_output(['git', '-C', '/app', 'cat-file', 'blob', oid])
            if TARGET_DEFINITION.search(body):
                hits.append(f'/app/.git: optimized KDA definition in object {oid}')
            private = PRIVATE_UNIQUE.search(body.decode('utf-8', errors='ignore'))
            if private:
                hits.append(f'/app/.git: private verifier material ({private.group(0)}) in object {oid}')
    artifact_count = 0
    for root in ARTIFACT_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if path.is_file() and '.git' not in path.parts:
                artifact_count += 1
                hits += scan_artifact(path)
    hits += check_declared_math(Path('/app/eval/kimi_linear/kda'))

    if hits:
        print("FATAL: the answer's address is on the agent's surface:", file=sys.stderr)
        for h in hits:
            print("  " + h, file=sys.stderr)
        return 1
    print(f"no answer address: {scanned} task-authored files, {artifact_count} artifact paths, "
          f"the declared mathematics, HEAD's commit, and the object store "
          f"all clean of the answer's address and of private verifier material ({len(PR_NUMBERS)} PR numbers, "
          f"{len(PR_AUTHORS)} authors, {2 + len(CLOSURE_COMMITS)} commits scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
