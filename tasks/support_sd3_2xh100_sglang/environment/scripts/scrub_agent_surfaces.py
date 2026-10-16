#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Remove every agent AUTO-INJECTION surface from a source tree, and record exactly what was removed.

WHAT A SURFACE IS. A file or directory that some coding-agent harness loads into the model's context, runs, or
uses to configure itself, WITHOUT being asked: Claude Code's CLAUDE.md and .claude/ (skills, commands, agents,
settings with hooks), the cross-vendor AGENTS.md, Gemini CLI's GEMINI.md, Cursor's rules, Copilot's
.github/copilot-instructions.md, project MCP server configs, and the rest of the table below. Ordinary source,
docs and tests are NOT surfaces, however useful: a harness never loads them on its own, and every agent can read
them alike. Matching is by path component only, never by content.

WHY. A surface that one vendor's harness loads automatically makes the starting state differ between agents, so a
workspace meant to be identical for every agent removes all of them, the same way, and keeps a record.

TWO SUBCOMMANDS.

  scrub   deletes the surfaces under a tree and merges a deterministic receipt: tree label, and for every removed
          file its path relative to the tree, sha256, size and the rule that matched. Idempotent: a second run
          merges into the existing receipt, so a retried setup cannot erase the first run's record.
  verify  compares an on-disk receipt with a committed expectation and with this file's own sha256, and
          censuses a git ref for surfaces independently. A receipt stored in a writable place is treated as a
          claim to check, not as the record itself.

STDLIB ONLY. The environment and verifier copies must remain byte-identical. The committed expectation
records this file's checksum as well as the removed surfaces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

SCHEMA = "infb-agent-surface-scrub/1"
PATTERN_SET_VERSION = "2026-09-29.1"


SURFACE_DIRS: dict[str, str] = {
    ".claude": "Claude Code: CLAUDE.md, skills/, commands/, agents/, settings*.json (hooks, permissions)",
    ".codex": "OpenAI Codex CLI: project config and prompts",
    ".gemini": "Gemini CLI: settings, commands, context files",
    ".cursor": "Cursor: rules/, mcp.json",
    ".windsurf": "Windsurf: rules/",
    ".clinerules": "Cline: rules directory form",
    ".roo": "Roo Code: rules and modes",
    ".kiro": "Kiro: steering and specs",
    ".junie": "JetBrains Junie: guidelines",
    ".amazonq": "Amazon Q Developer: rules",
    ".continue": "Continue: rules and prompts",
    ".augment": "Augment Code: rules",
    ".trae": "Trae: rules",
    ".openhands": "OpenHands: microagents",
    ".qwen": "Qwen Code: settings and context",
    ".opencode": "opencode: agents and commands",
}
SURFACE_FILES: dict[str, str] = {
    "CLAUDE.md": "Claude Code project memory (loaded from the working directory, its parents and read subtrees)",
    "CLAUDE.local.md": "Claude Code local project memory",
    "AGENTS.md": "AGENTS.md convention: Codex, Copilot coding agent, Amp, Zed, Cursor and others",
    "AGENTS.override.md": "OpenAI Codex CLI override instructions",
    "AGENT.md": "Amp agent instructions",
    "GEMINI.md": "Gemini CLI context file",
    "QWEN.md": "Qwen Code context file",
    "CRUSH.md": "Crush context file",
    ".cursorrules": "Cursor legacy rules file",
    ".windsurfrules": "Windsurf rules file",
    ".clinerules": "Cline rules, file form",
    ".roorules": "Roo Code rules file",
    ".roomodes": "Roo Code custom modes",
    ".goosehints": "Goose hints file",
    ".rules": "Zed agent rules file",
    ".augment-guidelines": "Augment Code guidelines file",
    ".openhands_instructions": "OpenHands repository instructions",
    ".aider.conf.yml": "Aider config (can auto-load read-only convention files)",
    ".mcp.json": "project MCP server config (Claude Code and others)",
}
SURFACE_NESTED: dict[tuple[str, str], str] = {
    (".github", "copilot-instructions.md"): "GitHub Copilot repository instructions",
    (".github", "instructions"): "GitHub Copilot path-specific instructions (directory)",
    (".github", "prompts"): "GitHub Copilot prompt files (directory)",
    (".github", "chatmodes"): "GitHub Copilot chat modes (directory)",
    (".vscode", "mcp.json"): "VS Code MCP server config",
}

_SKIP_DIRS = frozenset({".git"})


def pattern_table() -> dict:
    """The rule set as data. Its sha256 is in every receipt, so a changed table is visible as a changed receipt."""
    return {
        "version": PATTERN_SET_VERSION,
        "dirs": dict(sorted(SURFACE_DIRS.items())),
        "files": dict(sorted(SURFACE_FILES.items())),
        "nested": {f"{p}/{n}": why for (p, n), why in sorted(SURFACE_NESTED.items())},
    }


def _canonical(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def pattern_sha256() -> str:
    return hashlib.sha256(_canonical(pattern_table())).hexdigest()


def own_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def classify(rel_parts: tuple[str, ...]) -> str | None:
    """The rule id a relative path's LAST component matches as a surface root, or None.

    Only the last component is tested: a caller walking a tree stops at the first matching ancestor, so a file
    inside `.claude/` is reached as part of that directory, never classified on its own name."""
    if not rel_parts:
        return None
    name = rel_parts[-1]
    parent = rel_parts[-2] if len(rel_parts) >= 2 else ""
    if (parent, name) in SURFACE_NESTED:
        return f"nested:{parent}/{name}"
    return None if name not in SURFACE_FILES and name not in SURFACE_DIRS else (
        f"file:{name}" if name in SURFACE_FILES else f"dir:{name}")


def _hash_entry(path: Path) -> tuple[str, int, str]:
    """(sha256, bytes, kind) of one file or symlink; a symlink is hashed by its TARGET STRING, never followed."""
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        target = os.readlink(path).encode("utf-8", "surrogateescape")
        return hashlib.sha256(target).hexdigest(), len(target), "symlink"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), st.st_size, "file"


def find_surfaces(root: Path) -> list[tuple[Path, str]]:
    """Every surface ROOT under `root` (a matched directory is not descended), as (absolute path, rule), sorted."""
    found: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        here = Path(dirpath)
        rel_here = here.relative_to(root).parts
        keep = []
        for d in sorted(dirnames):
            if d in _SKIP_DIRS:
                continue
            rule = classify(rel_here + (d,))
            if rule is not None:
                found.append((here / d, rule))
            else:
                keep.append(d)
        dirnames[:] = keep
        for fn in sorted(filenames):
            rule = classify(rel_here + (fn,))
            if rule is not None:
                found.append((here / fn, rule))
    return sorted(found, key=lambda t: t[0].as_posix())


def expand(root: Path, surface: Path, rule: str) -> list[dict]:
    """The receipt entries for one surface root: every file or symlink under it (or itself), hashed."""
    out: list[dict] = []
    if surface.is_dir() and not surface.is_symlink():
        for dirpath, dirnames, filenames in os.walk(surface, topdown=True, followlinks=False):
            dirnames.sort()
            for fn in sorted(filenames) + sorted(d for d in dirnames if Path(dirpath, d).is_symlink()):
                p = Path(dirpath) / fn
                sha, size, kind = _hash_entry(p)
                out.append({"path": p.relative_to(root).as_posix(), "sha256": sha, "bytes": size, "kind": kind,
                            "rule": rule})
    else:
        sha, size, kind = _hash_entry(surface)
        out.append({"path": surface.relative_to(root).as_posix(), "sha256": sha, "bytes": size, "kind": kind,
                    "rule": rule})
    return out


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        for dirpath, dirnames, filenames in os.walk(path, topdown=False, followlinks=False):
            for fn in filenames:
                os.unlink(os.path.join(dirpath, fn))
            for d in dirnames:
                full = os.path.join(dirpath, d)
                if os.path.islink(full):
                    os.unlink(full)
                else:
                    os.rmdir(full)
        os.rmdir(path)
    else:
        os.unlink(path)


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _empty_receipt() -> dict:
    return {"schema": SCHEMA, "pattern_set_version": PATTERN_SET_VERSION, "pattern_set_sha256": pattern_sha256(),
            "scrubber_sha256": own_sha256(), "trees": {}}


def deterministic_view(receipt: dict) -> dict:
    """The part of a receipt two correct runs must agree on byte for byte: no roots, no routes, no times."""
    trees = {}
    for label, t in sorted((receipt.get("trees") or {}).items()):
        trees[label] = {
            "action": t.get("action"),
            "removed": sorted(({k: e.get(k) for k in ("path", "sha256", "bytes", "kind", "rule")}
                               for e in (t.get("removed") or [])), key=lambda e: str(e["path"])),
            "surviving": t.get("surviving"),
        }
    return {"schema": receipt.get("schema"), "pattern_set_version": receipt.get("pattern_set_version"),
            "pattern_set_sha256": receipt.get("pattern_set_sha256"),
            "scrubber_sha256": receipt.get("scrubber_sha256"), "trees": trees}


def cmd_scrub(args: argparse.Namespace) -> int:
    root = Path(args.tree).resolve()
    if not root.is_dir():
        print(f"[scrub-agent-surfaces] FATAL: {root} is not a directory", file=sys.stderr)
        return 2
    receipt_path = Path(args.receipt)
    receipt = _load(receipt_path) if receipt_path.exists() else None
    if receipt is None or receipt.get("schema") != SCHEMA:
        receipt = _empty_receipt()
    if receipt.get("pattern_set_sha256") != pattern_sha256() or receipt.get("scrubber_sha256") != own_sha256():
        print("[scrub-agent-surfaces] FATAL: an existing receipt was written by a different scrubber or rule set; "
              "refusing to merge two definitions of a surface into one record", file=sys.stderr)
        return 4
    surfaces = find_surfaces(root)
    entries = [e for s, rule in surfaces for e in expand(root, s, rule)]
    writable = os.access(root, os.W_OK)
    action = "inventory_only" if args.inventory_only else ("deleted" if writable else "inventory_only_not_writable")
    if action == "deleted":
        for s, _rule in surfaces:
            try:
                _remove(s)
            except OSError as exc:
                print(f"[scrub-agent-surfaces] FATAL: could not remove {s}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                return 3
    tree = receipt["trees"].setdefault(args.label, {"action": action, "removed": [], "surviving": None})
    known = {e["path"]: e for e in tree.get("removed") or []}
    for e in entries:
        known.setdefault(e["path"], e)
    tree["removed"] = sorted(known.values(), key=lambda e: e["path"])
    if tree.get("action") != action and entries:
        tree["action"] = action
    survivors = find_surfaces(root)
    tree["surviving"] = len(survivors)
    run = receipt.setdefault("run", {})
    run.setdefault("tree_roots", {})[args.label] = str(root)
    if args.route:
        run["route"] = args.route
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(_canonical(receipt))
    print(f"[scrub-agent-surfaces] {args.label}: {len(surfaces)} surface root(s), {len(entries)} file(s) "
          f"{'recorded' if action != 'deleted' else 'removed'} under {root}; surviving={len(survivors)}; "
          f"receipt {receipt_path} sha256={hashlib.sha256(receipt_path.read_bytes()).hexdigest()}")
    for s, rule in surfaces:
        print(f"[scrub-agent-surfaces]   {rule:32s} {s.relative_to(root).as_posix()}")
    if action == "deleted" and survivors:
        print(f"[scrub-agent-surfaces] FATAL: {len(survivors)} surface(s) survived deletion under {root}",
              file=sys.stderr)
        return 3
    return 0


def census_git_ref(repo: Path, ref: str) -> tuple[list[str] | None, str | None]:
    """Surface paths committed at `ref` in `repo`, from git's own object store (the tree a workspace started from)."""
    try:
        out = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo), "ls-tree", "-r", "--name-only",
                              ref], capture_output=True, text=True, check=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    hits = []
    for line in out.splitlines():
        parts = tuple(line.split("/"))
        if any(classify(parts[: i + 1]) for i in range(len(parts)) if parts[i] not in _SKIP_DIRS):
            hits.append(line)
    return sorted(hits), None


def verify_receipt(receipt: dict | None, expected: dict) -> list[str]:
    """Every way the on-disk receipt differs from the committed expectation; [] means it matches."""
    if receipt is None:
        return ["no readable receipt: prepare did not record a scrub, or the file was removed or corrupted"]
    problems = []
    got, want = deterministic_view(receipt), deterministic_view(expected)
    if got["schema"] != SCHEMA:
        problems.append(f"schema {got['schema']!r} is not {SCHEMA!r}")
    if got["pattern_set_sha256"] != pattern_sha256():
        problems.append("the receipt's rule-set sha256 is not this copy's rule set")
    if got["scrubber_sha256"] != own_sha256():
        problems.append("the receipt was written by a scrubber whose bytes differ from this copy")
    for label, w in want["trees"].items():
        g = got["trees"].get(label)
        if g is None:
            problems.append(f"tree {label!r} is absent from the receipt")
            continue
        if g["action"] != w["action"]:
            problems.append(f"tree {label!r}: action {g['action']!r}, expected {w['action']!r}")
        if g["surviving"] != 0:
            problems.append(f"tree {label!r}: {g['surviving']} surface(s) survived the scrub")
        gp = {e["path"]: e for e in g["removed"]}
        wp = {e["path"]: e for e in w["removed"]}
        missing, extra = sorted(set(wp) - set(gp)), sorted(set(gp) - set(wp))
        changed = sorted(p for p in set(gp) & set(wp) if gp[p] != wp[p])
        if missing:
            problems.append(f"tree {label!r}: {len(missing)} expected removal(s) not recorded, e.g. {missing[:3]}")
        if extra:
            problems.append(f"tree {label!r}: {len(extra)} unexpected removal(s) recorded, e.g. {extra[:3]}")
        if changed:
            problems.append(f"tree {label!r}: {len(changed)} removal(s) with a different hash/size, e.g. {changed[:3]}")
    for label in sorted(set(got["trees"]) - set(want["trees"])):
        problems.append(f"tree {label!r} is in the receipt but not in the expectation")
    return problems


def cmd_verify(args: argparse.Namespace) -> int:
    expected = _load(Path(args.expected))
    receipt_path = Path(args.receipt)
    receipt = _load(receipt_path) if receipt_path.exists() else None
    details: dict = {"capability_id": "b0_agent_surface_scrub", "layer": "integrity", "scored": False,
                     "receipt_path": str(receipt_path), "expected_path": args.expected,
                     "checker_scrubber_sha256": own_sha256(), "pattern_set_sha256": pattern_sha256(),
                     "pattern_set_version": PATTERN_SET_VERSION}
    if expected is None:
        problems = [f"the committed expectation {args.expected} is unreadable"]
    else:
        problems = verify_receipt(receipt, expected)
        details["expected_removed"] = {k: len(v.get("removed") or []) for k, v in (expected.get("trees") or {}).items()}
    if receipt is not None:
        details["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        details["receipt_removed"] = {k: len(v.get("removed") or []) for k, v in (receipt.get("trees") or {}).items()}
        details["receipt_run"] = receipt.get("run")
        if args.copy_receipt_to:
            try:
                Path(args.copy_receipt_to).parent.mkdir(parents=True, exist_ok=True)
                Path(args.copy_receipt_to).write_bytes(receipt_path.read_bytes())
            except OSError as exc:
                details["copy_error"] = f"{type(exc).__name__}: {exc}"
    if args.baseline_repo:
        hits, err = census_git_ref(Path(args.baseline_repo), args.baseline_ref)
        details["baseline_census"] = {"repo": args.baseline_repo, "ref": args.baseline_ref,
                                      "surfaces": hits, "error": err}
        if err is not None:
            problems.append(f"the baseline census could not run: {err}")
        elif hits:
            problems.append(f"{len(hits)} surface path(s) are committed in {args.baseline_ref}, e.g. {hits[:3]}")


        wt = Path(args.baseline_repo)
        if wt.is_dir():
            details["worktree_surfaces_now"] = [s.relative_to(wt).as_posix() for s, _r in find_surfaces(wt)]
    passed = not problems
    gate = {"gate": args.gate, "passed": passed, "score": 1.0 if passed else 0.0, "is_hard_gate": False,
            "level": "diagnostic", "error": None if passed else "; ".join(problems)[:2000],
            "details": {**details, "problems": problems}}
    if args.gate_out:
        Path(args.gate_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.gate_out).write_text(json.dumps(gate, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[scrub-agent-surfaces] verify: {'MATCH' if passed else 'MISMATCH'}"
          + ("" if passed else " -- " + "; ".join(problems)))
    return 0 if passed else 1


def cmd_expect(args: argparse.Namespace) -> int:
    """Write the expected manifest from a pristine tree WITHOUT deleting anything (maintainer tool).

    `--label L=ACTION` expects tree L to record every surface of `--tree` under that action; `--empty-label
    L=ACTION` expects tree L to record none (e.g. a working copy rsynced from an already-scrubbed source)."""
    root = Path(args.tree).resolve()
    receipt = _empty_receipt()
    entries = [e for s, rule in find_surfaces(root) for e in expand(root, s, rule)]
    for spec, recorded in [(x, entries) for x in args.label] + [(x, []) for x in args.empty_label]:
        label, _, action = spec.partition("=")
        if not label or not action:
            print(f"[scrub-agent-surfaces] FATAL: label spec {spec!r} is not LABEL=ACTION", file=sys.stderr)
            return 2
        receipt["trees"][label] = {"action": action, "removed": recorded, "surviving": 0}
    out = deterministic_view(receipt)
    out["generated_from"] = args.provenance
    Path(args.out).write_bytes(_canonical(out))
    print(f"[scrub-agent-surfaces] expectation written to {args.out}: "
          + ", ".join(f"{k}={len(v['removed'])}" for k, v in out["trees"].items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scrub", help="remove the surfaces under a tree and merge the receipt")
    s.add_argument("--tree", required=True)
    s.add_argument("--label", required=True, help="the receipt key for this tree, e.g. engine_source")
    s.add_argument("--receipt", required=True)
    s.add_argument("--route", default="", help="recorded in the receipt's non-deterministic run block")
    s.add_argument("--inventory-only", action="store_true", help="record, do not delete")
    s.set_defaults(fn=cmd_scrub)
    v = sub.add_parser("verify", help="check an on-disk receipt against a committed expectation")
    v.add_argument("--receipt", required=True)
    v.add_argument("--expected", required=True)
    v.add_argument("--gate", required=True, help="the id written into the gate file's `gate` field")
    v.add_argument("--gate-out", default="")
    v.add_argument("--baseline-repo", default="")
    v.add_argument("--baseline-ref", default="inference_base")
    v.add_argument("--copy-receipt-to", default="")
    v.set_defaults(fn=cmd_verify)
    e = sub.add_parser("expect", help="write an expected manifest from a pristine tree (does not delete)")
    e.add_argument("--tree", required=True)
    e.add_argument("--label", action="append", default=[],
                   help="LABEL=ACTION whose removals are every surface of --tree, e.g. engine_source=deleted")
    e.add_argument("--empty-label", action="append", default=[],
                   help="LABEL=ACTION expected to remove nothing, e.g. candidate_repo=deleted")
    e.add_argument("--out", required=True)
    e.add_argument("--provenance", default="")
    e.set_defaults(fn=cmd_expect)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
