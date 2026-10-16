#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/publish.sh [options]

Build and publish datacurve-pier to PyPI, then tag and create a GitHub release.

Options:
  --bump patch|minor|major  Bump the project version before building.
  --dry-run                Build artifacts and run uv publish --dry-run only.
  --no-push                Do not push main or the tag.
  --no-github-release      Do not create a GitHub release.
  -h, --help               Show this help.

Authentication:
  Set UV_PUBLISH_TOKEN for PyPI token auth, or rely on uv's configured
  credentials/trusted publishing.
EOF
}

BUMP=""
DRY_RUN=0
PUSH=1
GITHUB_RELEASE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bump)
      if [[ $# -lt 2 ]]; then
        echo "error: --bump requires patch, minor, or major" >&2
        exit 2
      fi
      BUMP="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-push)
      PUSH=0
      shift
      ;;
    --no-github-release)
      GITHUB_RELEASE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$BUMP" && "$BUMP" != "patch" && "$BUMP" != "minor" && "$BUMP" != "major" ]]; then
  echo "error: --bump must be patch, minor, or major" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree has uncommitted changes; commit or stash them before publishing" >&2
  exit 1
fi

if [[ -n "$BUMP" ]]; then
  uv version --bump "$BUMP"
fi

echo "Building viewer assets..."
(
  cd apps/viewer
  bun install
  bun run build
)

rm -rf src/pier/viewer/static
mkdir -p src/pier/viewer/static
cp -R apps/viewer/build/client/. src/pier/viewer/static/

rm -rf dist build
uv build

publish_args=()
if [[ -n "${UV_PUBLISH_TOKEN:-}" ]]; then
  publish_args+=(--token "$UV_PUBLISH_TOKEN")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  publish_args+=(--dry-run)
fi

uv publish "${publish_args[@]}"

VERSION="$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete for v${VERSION}."
  exit 0
fi

if [[ -n "$BUMP" ]]; then
  git add pyproject.toml uv.lock
  git commit -m "v${VERSION}"
fi

if git rev-parse "v${VERSION}" >/dev/null 2>&1; then
  echo "Tag v${VERSION} already exists."
else
  git tag -a "v${VERSION}" -m "v${VERSION}"
fi

if [[ "$PUSH" == "1" ]]; then
  git push origin main "v${VERSION}"
fi

if [[ "$GITHUB_RELEASE" == "1" ]]; then
  gh release create "v${VERSION}" --title "v${VERSION}" --generate-notes
fi
