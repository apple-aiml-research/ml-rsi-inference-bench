#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Reference subprocesses may import engine code only from their full snapshot.

Loaded by Python at startup from the trusted verifier directory, before the
reference server or its workers import vLLM. This also rejects editable-install
fallbacks to the candidate checkout. It has no effect in candidate processes.
"""

import os
import sys

_tree = os.path.realpath(os.environ["INFB_REFERENCE_TREE"])
_candidate = os.path.realpath(os.environ.get("INFB_CANDIDATE_TREE", "/app"))


def _inside(path, root):
    path = os.path.realpath(path)
    return path == root or path.startswith(root + os.sep)


sys.path[:] = [entry for entry in sys.path
               if not _inside(entry or os.getcwd(), _candidate)]


class ReferenceImports:
    def find_spec(self, fullname, path=None, target=None):
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None:
                continue
            locations = list(spec.submodule_search_locations or ())
            if spec.origin not in (None, "built-in", "frozen"):
                locations.append(spec.origin)
            if any(_inside(location, _candidate) for location in locations):
                raise ImportError(f"reference import {fullname!r} attempted candidate code: {locations}")
            if fullname.split(".", 1)[0] in ("vllm", "tests"):
                if not locations or any(not _inside(location, _tree) for location in locations):
                    raise ImportError(f"reference import {fullname!r} escaped its snapshot: {locations}")
            return spec
        return None


sys.meta_path.insert(0, ReferenceImports())
