#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Narrow source-scan exception for isolated, deferred unittest references.

This never excludes a directory. Module-level imports, helpers, decorators,
and test modules referenced by candidate production code stay in the scan.
New unresolvable dynamic imports/evaluation conservatively disable the exception.
"""
import ast
from pathlib import Path


def _main_guard(node):
    return (isinstance(node, ast.If)
            and ast.dump(node.test) == ast.dump(ast.parse('__name__ == "__main__"', mode='eval').body)
            and not node.orelse and len(node.body) == 1
            and ast.dump(node.body[0]) == ast.dump(ast.parse('unittest.main()').body[0]))


def exempt_test_lines(repo, rel, changed_files=None):
    path = Path(rel)
    if (path.parent.as_posix() != 'python/sglang/multimodal_gen/test/unit'
            or not path.name.startswith('test_') or path.suffix != '.py'):
        return set()
    try:
        tree = ast.parse((repo/path).read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
                   and not n.decorator_list and not n.keywords and len(n.bases) == 1
                   and ast.unparse(n.bases[0]) == 'unittest.TestCase']
        methods = [f for c in classes for f in c.body if isinstance(f, ast.FunctionDef)
                   and f.name.startswith('test_') and not f.decorator_list]
        if any(isinstance(n, ast.ClassDef) and n not in classes for n in tree.body):
            return set()
        definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        definitions += [f for c in classes for f in c.body if isinstance(f, ast.FunctionDef)]
        for f in definitions:


            headers = [*f.args.defaults, *f.args.kw_defaults, f.returns]
            headers += [a.annotation for a in (*f.args.posonlyargs, *f.args.args,
                        *f.args.kwonlyargs, f.args.vararg, f.args.kwarg) if a is not None]
            unsafe_decorators = [d for d in f.decorator_list
                                 if not isinstance(d, ast.Name) or d.id not in ('staticmethod', 'classmethod')]
            if unsafe_decorators or any(isinstance(n, ast.Call) for value in headers
                                      if value is not None for n in ast.walk(value)):
                return set()


        allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
        if any(not isinstance(n, allowed) and not _main_guard(n)
               and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                        and isinstance(n.value.value, str)) for n in tree.body):
            return set()
        if any(not isinstance(n, (ast.FunctionDef, ast.Pass))
               and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
               for c in classes for n in c.body):
            return set()
        if any(n.decorator_list for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))):
            return set()
        symbols = {path.stem, *(c.name for c in classes), *(f.name for f in methods),
                   *(n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))}
        module = '.'.join(path.with_suffix('').parts[1:])
        others = sorted(changed_files) if changed_files is not None else (
            str(p.relative_to(repo)) for p in (repo/'python').rglob('*.py'))
        for other in others:
            if other == rel:
                continue
            other_tree = ast.parse((repo/other).read_text())
            added = changed_files.get(other) if changed_files is not None else None
            prose = {id(n.body[0].value) for n in ast.walk(other_tree)
                     if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.body and isinstance(n.body[0], ast.Expr)
                     and isinstance(n.body[0].value, ast.Constant)
                     and isinstance(n.body[0].value.value, str)}
            for n in ast.walk(other_tree):
                if isinstance(n, ast.Name) and n.id in symbols:
                    return set()
                if isinstance(n, ast.Attribute) and n.attr in symbols:
                    return set()
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in n.names] + [getattr(n, 'module', '') or '']
                    if any(s in name.split('.') or (name.startswith('sglang.multimodal_gen.test')
                           and module.startswith(name + '.'))
                           for name in names for s in symbols):
                        return set()
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in prose:
                    if any(s in n.value for s in symbols):
                        return set()
                if isinstance(n, ast.Call):


                    if added is not None and not any(
                            line in added for line in range(n.lineno, n.end_lineno + 1)):
                        continue
                    name = ast.unparse(n.func).rsplit('.', 1)[-1]
                    if name in ('exec', 'eval') and (isinstance(n.func, ast.Name)
                            or ast.unparse(n.func) in ('builtins.exec', 'builtins.eval')):
                        return set()
                    if name in ('run_path', 'run_module', 'spec_from_file_location'):
                        return set()
                    if name in ('import_module', '__import__') and (not n.args
                            or not isinstance(n.args[0], ast.Constant)):
                        return set()
        return {line for f in methods for node in f.body
                for line in range(node.lineno, node.end_lineno + 1)}
    except (OSError, SyntaxError, ValueError):
        return set()
