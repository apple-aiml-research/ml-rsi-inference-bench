#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'Required SD3 native implementation integrity check.'
from __future__ import annotations

import ast
import io
import re
import sys
import subprocess
import tokenize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, GateResult, WORK_DIR, added_line_hit, candidate_diff, safe_run  # noqa: E402

GATE = "no_hf_evaluated_path"


FORBIDDEN: list[tuple[str, str]] = [
    (r"from\s+diffusers(?:\.[\w.]+)?\s+import\s+[^\n]*Pipeline", "imports an end-to-end diffusers Pipeline class"),
    (r"from\s+diffusers\.pipelines", "imports from diffusers.pipelines (end-to-end)"),
    (r"\bdiffusers\.[\w.]*Pipeline\s*\(", "instantiates a diffusers.*Pipeline directly"),
    (r"\b\w*Pipeline\s*\.\s*from_pretrained\s*\(", "*Pipeline.from_pretrained (end-to-end diffusers load)"),


    (r"from\s+diffusers\.models\.transformers\b", "imports a whole model from diffusers.models.transformers"),
    (r"from\s+diffusers\.models\.autoencoders\b", "imports a whole model from diffusers.models.autoencoders"),
    (r"from\s+diffusers\.models\.unets\b", "imports a whole model from diffusers.models.unets"),


    (r"(?:import_module|__import__)\s*\(\s*[\"']diffusers\.models\.(?:transformers|autoencoders|unets)",
     "dynamically imports a whole diffusers model module (import_module/__import__)"),
    (r"(?:import_module|__import__)\s*\(\s*[\"']diffusers\.pipelines",
     "dynamically imports diffusers.pipelines (end-to-end)"),
]


DIFFUSERS_IMPORT = re.compile(
    r"from\s+diffusers(?:\.[\w.]+)?\s+import\s+([^\n(]+)|import\s+diffusers\s*$", re.M)


def diffusers_names(txt: str) -> set[str]:
    """Names bound in this file by an import from diffusers."""
    names: set[str] = set()
    for m in DIFFUSERS_IMPORT.finditer(txt):
        imported = m.group(1)
        if not imported:
            continue
        for part in imported.replace("(", " ").replace(")", " ").split(","):
            part = part.strip()
            if not part:
                continue
            names.add(part.split()[-1])
    return names


def scan_hits(txt: str, added, rel: str, baseline=None) -> list:
    """Every FORBIDDEN pattern and `<diffusers name>.from_pretrained(` in one file, kept only where the match touches a
    line the candidate added (`added` None = a new file, every line). Comments and docstrings are blanked first; the
    diffusers names are read from the WHOLE file, since the import may be a baseline line and the call an added one."""
    from integrity_ast import scan
    return scan(txt, added, rel, "hf", baseline=baseline)


def _prose_blanked(txt: str) -> str:
    """Blank COMMENTS and DOCSTRINGS, preserving line/column layout, and nothing else.

    Documentation may describe a forbidden import without executing it. Matching that prose
    would reject an implementation based on comment wording rather than executable behavior.

    WHY NOT blank every string: a string literal is a real evasion surface —
    `importlib.import_module("diffusers.models.transformers")` and `__import__(...)` route around
    every import-shaped pattern. Docstrings are prose by position (an `Expr` holding a bare
    Constant str at module/class/function head); every other literal stays visible to the scan.

    Layout is preserved character-for-character so reported line numbers keep pointing at the
    original file.
    """
    lines = txt.splitlines(keepends=True)

    def blank_span(lo_line: int, lo_col: int, hi_line: int, hi_col: int) -> None:

        for ln in range(lo_line, hi_line + 1):
            i = ln - 1
            if i >= len(lines):
                break
            s = lines[i]
            nl = "\n" if s.endswith("\n") else ""
            body = s[:-1] if nl else s
            a = lo_col if ln == lo_line else 0
            b = hi_col if ln == hi_line else len(body)
            a, b = max(0, min(a, len(body))), max(0, min(b, len(body)))
            lines[i] = body[:a] + (" " * (b - a)) + body[b:] + nl

    try:
        tree = ast.parse(txt)
    except SyntaxError:


        return txt

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None) or []
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            c = first.value
            blank_span(c.lineno, c.col_offset, c.end_lineno, c.end_col_offset)

    for tok in tokenize.generate_tokens(io.StringIO(txt).readline):
        if tok.type == tokenize.COMMENT:
            blank_span(tok.start[0], tok.start[1], tok.end[0], tok.end[1])

    return "".join(lines)


def pinned_baseline_source(rel):
    """A missing path is a new file; Git/read failures are instrument gaps."""
    from _common import BASELINE_TAG
    git=['git', '-c', 'safe.directory=*', '-C', str(REPO)]
    previous=subprocess.run(git+['show', f'{BASELINE_TAG}:{rel}'],capture_output=True,text=True,timeout=30)
    if previous.returncode==0:
        return previous.stdout
    tree=subprocess.run(git+['ls-tree','-z',BASELINE_TAG,'--',':(literal)'+rel],
                        capture_output=True,text=True,timeout=30)
    if tree.returncode!=0 or tree.stdout:
        raise RuntimeError('could not read pinned baseline: '+rel)
    return None


def run() -> GateResult:
    scope = candidate_diff(REPO)
    if "error" in scope:

        return GateResult(gate=GATE, passed=False, score=0.0,
                          error=f"not evaluated: the candidate's added lines could not be read ({scope['error']})",
                          details={"not_evaluated": True, "scope": "added lines since inference_base"})
    files = scope["files"]
    details = {"scope": "added lines since inference_base, every .py under repo/python/",
               "scanned": sorted(files), "new_files": sorted(r for r, a in files.items() if a is None),
               "added_lines": sum(len(a) for a in files.values() if a is not None)}
    if not files:
        return GateResult(gate=GATE, passed=True, score=1.0,
                          details={**details, "note": "the candidate added no .py lines; vacuously clean"})
    hits = []
    for rel, added in sorted(files.items()):
        raw = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        baseline = None
        if added is not None:


            try:
                baseline = pinned_baseline_source(rel)
            except Exception as exc:
                return GateResult(gate=GATE,passed=False,score=0.0,
                    error=f'not evaluated: {exc}',details={**details,'not_evaluated':True})
        found = scan_hits(raw, added, str((REPO / rel).relative_to(WORK_DIR)), baseline=baseline)
        if found:
            from hf_test_scope import exempt_test_lines
            deferred = exempt_test_lines(REPO, rel, changed_files=files)
            ignored = [h for h in found if h['line'] in deferred]
            if ignored:
                details.setdefault('isolated_unittest_references', []).extend(ignored)
            found = [h for h in found if h['line'] not in deferred]
        hits += found
    details["hits"] = hits[:20]
    details["hit_count"] = len(hits)
    if hits:
        return GateResult(gate=GATE, passed=False, score=0.0,
                          error=f"{len(hits)} prohibited whole-model or loader source reference(s)", details=details)
    return GateResult(gate=GATE, passed=True, score=1.0, details=details)


if __name__ == "__main__":
    sys.exit(safe_run(run, GATE, fail_exit_code=0))
