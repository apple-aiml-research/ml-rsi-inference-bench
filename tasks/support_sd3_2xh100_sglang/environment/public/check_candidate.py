#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Check diffusion-engine Python syntax without importing candidate code.

This is a development aid, not an SD3 implementation or acceptance check.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parents[1] / "repo"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                        help="Engine checkout (default: /app/repo)")
    args = parser.parse_args(argv)
    source = args.repo / "python" / "sglang" / "multimodal_gen"
    if not source.is_dir():
        print(f"missing diffusion-engine source directory: {source}")
        return 1

    paths = sorted(source.rglob("*.py"))
    if not paths:
        print(f"no Python source files found under {source}")
        return 1

    errors = []
    for path in paths:
        try:


            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        except (SyntaxError, ValueError, OSError) as exc:
            errors.append(f"{path.relative_to(args.repo)}: {exc}")
    if errors:
        for error in errors:
            print(f"syntax/read error: {error}")
        return 1

    print(f"Python syntax OK: {len(paths)} files under {source}.")
    print("Syntax only: registration, model loading, numerical correctness and "
          "serving behavior have not been checked. Exercise the public API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
