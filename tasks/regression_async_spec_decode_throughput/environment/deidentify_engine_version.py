#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Remove the build commit from an installed engine distribution's version stamp.

Packaging tools derive a version from the source checkout and write the commit into it in several
places at once: the generated `_version.py` (`__version__`, `__version_tuple__` and `__commit_id__`
all carry it), the distribution METADATA, the `.dist-info` DIRECTORY NAME, and `direct_url.json`,
which records the URL the wheel was installed from. Removing one of those and leaving the rest is the
common mistake, and it is worse than doing nothing: the step is visible, so the next reader stops
looking.

This truncates the version to its PEP 440 public segment -- `1.2.3rc1.dev60+g<commit>` becomes
`1.2.3rc1` -- rather than inventing a string, so `importlib.metadata.version()` keeps resolving and
the release/minor comparisons the engine makes on its own version keep their meaning.

Two modes:

    --package <name>              rewrite, then re-scan and exit non-zero if anything survived
    --package <name> --scan       assert only, for an install that is expected to be clean already

`--version-file PATH` names an additional copy to rewrite (repeatable): a build that copies the
generated file out of site-packages has two copies, and fixing only the original fixes the one
nothing reads.

Why it fails closed: a scrub that silently missed a carrier leaves the build green and the value in
place, which is the one outcome worse than no scrub at all.
"""
import argparse
import json
import re
import shutil
import sys
import sysconfig
from pathlib import Path

LEAK = re.compile(r"(\+g[0-9a-f]{7,40}|\bg[0-9a-f]{9,40}\b|\b[0-9a-f]{40}\b)")


def public_version(raw: str) -> str:
    """`1.2.3rc1.dev60+g<commit>` -> `1.2.3rc1`. Everything from `.dev` or `+` is the local part."""
    return re.split(r"\.dev|\+", raw, maxsplit=1)[0]


def version_tuple(pub: str):
    """`0.14.0rc1` -> `(0, 14, 0, 'rc1')`, the shape setuptools_scm itself writes."""
    m = re.match(r"^(\d+(?:\.\d+)*)((?:a|b|c|rc|alpha|beta|post|dev)\d*)?$", pub)
    if not m:
        return tuple(int(c) if c.isdigit() else c for c in pub.split("."))
    parts: list = [int(c) for c in m.group(1).split(".")]
    if m.group(2):
        parts.append(m.group(2))
    return tuple(parts)



def _move_dir(src: Path, dst: Path) -> None:
    """Rename a directory, and fall back when the filesystem will not rename it."""
    try:
        src.rename(dst)
        return
    except OSError:
        pass
    try:
        shutil.move(str(src), str(dst))
        return
    except OSError:
        pass
    shutil.copytree(src, dst)
    shutil.rmtree(src)


def scan_only(roots) -> int:
    """Assert that no commit-shaped token remains under the paths given."""
    residue = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for path in ([base] if base.is_file() else list(base.rglob("*"))):
            if LEAK.search(path.name):
                residue.append(f"{path} (name)")
            if path.is_file() and path.suffix in (".py", ".json", ".txt", "") or path.name in (
                    "METADATA", "RECORD", "direct_url.json"):
                try:
                    if path.is_file() and LEAK.search(path.read_text(errors="replace")[:200000]):
                        residue.append(str(path))
                except OSError:
                    continue
    if residue:
        print("[de-identify] FATAL: a commit-shaped token survived:\n  "
              + "\n  ".join(sorted(set(residue))[:12]), file=sys.stderr)
        return 9
    print(f"[de-identify] scan clean: {', '.join(str(r) for r in roots)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None, help="site-packages (default: this interpreter's purelib)")
    ap.add_argument("--package", default="vllm")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--scan", nargs="*", default=None,
                    help="assert-only mode: fail if any path below still carries a commit stamp")
    ap.add_argument("--version-file", action="append", default=[],
                    help="an additional generated _version.py to rewrite (repeatable)")
    args = ap.parse_args()
    site = Path(args.site) if args.site else Path(sysconfig.get_paths()["purelib"])
    pkg = args.package

    if args.scan is not None:
        roots = args.scan or (
            [site / pkg / "_version.py"] + sorted(site.glob(f"{pkg}-*.dist-info")))
        return scan_only(list(roots) + [Path(x) for x in args.version_file])

    vfile = site / pkg / "_version.py"
    raw = ""
    if vfile.is_file():
        m = re.search(r"""__version__\s*=\s*version\s*=\s*['"]([^'"]+)['"]""", vfile.read_text())
        raw = m.group(1) if m else ""
    dists = sorted(site.glob(f"{pkg}-*.dist-info"))
    if not raw and dists:
        m = re.search(r"^Version:\s*(.+)$", (dists[0] / "METADATA").read_text(errors="replace"), re.M)
        raw = m.group(1).strip() if m else ""
    if not raw:
        print(f"[de-identify] FATAL: no version found under {site} for {pkg}", file=sys.stderr)
        return 8
    pub = public_version(raw)
    print(f"[de-identify] {pkg} {raw} -> {pub}")
    if args.check_only:
        return 0

    for extra in [vfile] + [Path(x) for x in args.version_file]:
        if not extra.is_file():
            continue
        extra.write_text(
            "# file generated by setuptools-scm\n"
            '__all__ = ["__version__", "__version_tuple__", "version", "version_tuple",\n'
            '           "__commit_id__", "commit_id"]\n\n'
            f"__version__ = version = {pub!r}\n"
            f"__version_tuple__ = version_tuple = {version_tuple(pub)!r}\n"
            "__commit_id__ = commit_id = None\n",
            encoding="utf-8")
        print(f"[de-identify] rewrote {extra}")

    for dist in dists:
        meta = dist / "METADATA"
        if meta.is_file():
            body = meta.read_text(errors="replace")
            body = re.sub(r"^Version:.*$", f"Version: {pub}", body, count=1, flags=re.M)
            body = LEAK.sub("", body)
            meta.write_text(body, encoding="utf-8")
        (dist / "direct_url.json").unlink(missing_ok=True)
        record = dist / "RECORD"
        if record.is_file():
            record.write_text(record.read_text(errors="replace").replace(dist.name,
                                                                         f"{pkg}-{pub}.dist-info"),
                              encoding="utf-8")
        target = dist.parent / f"{pkg}-{pub}.dist-info"
        if target != dist:
            if target.exists():
                shutil.rmtree(target)
            _move_dir(dist, target)

    residue = []
    for path in (list(site.glob(f"{pkg}-*.dist-info/*")) + [site / pkg / "_version.py"]
                 + [Path(x) for x in args.version_file]):
        if not path.is_file():
            continue
        if LEAK.search(path.name) or LEAK.search(path.read_text(errors="replace")):
            residue.append(str(path))
    for path in site.glob(f"{pkg}-*"):
        if LEAK.search(path.name):
            residue.append(f"{path} (directory name)")
    if residue:
        print("[de-identify] FATAL: a commit-shaped token survived:\n  " + "\n  ".join(residue[:10]),
              file=sys.stderr)
        return 9
    print(f"[de-identify] clean: no commit-shaped token under {site}/{pkg} or its dist-info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
