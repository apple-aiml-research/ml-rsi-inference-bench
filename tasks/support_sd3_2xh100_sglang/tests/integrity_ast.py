#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Conservative executable-source checks; prose and generic .npy IO are legal.

These are bounded static checks, not a sandbox or an exhaustive proof against
obfuscated Python. Native runtime witnesses remain required independently.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import posixpath

PRIVATE_ENV = {'ANTI_CHEAT_PRE_HASHES_JSON', 'SD3_GOLDEN_DIR', 'SD3_DAY0_GOLDEN_SPEC',
               'INFB_TESTS_DIR', 'TESTS_DIR', 'INFB_VERIFIER_DIR'}
IO = {'open', 'load', 'loadtxt', 'read_bytes', 'read_text', 'write_bytes', 'write_text',
      'iterdir', 'glob', 'rglob', 'listdir', 'scandir', 'walk', 'copy', 'copy2', 'copyfile', 'rmtree'}
LAYER_BLOCKS = {'JointTransformerBlock', 'BasicTransformerBlock', 'PatchEmbed', 'Attention',
                'AdaLayerNormContinuous', 'AdaLayerNormZero', 'AdaLayerNormZeroSingle',
                'Downsample2D', 'Upsample2D', 'ResnetBlock2D', 'SpatialNorm',
                'Timesteps', 'TimestepEmbedding'}


def private_path(value):
    if not isinstance(value, str):
        return False
    p = posixpath.normpath(value)
    return any(p == root or p.startswith(root + '/') for root in ('/tests', '/hidden', '/logs/verifier'))


def whole_hf(path):


    if path in {'diffusers.models.autoencoders.vae.DecoderOutput',
                'diffusers.models.autoencoders.vae.DiagonalGaussianDistribution'}:
        return False
    if path.startswith('diffusers.'):


        if path.rsplit('.',1)[-1] in LAYER_BLOCKS:
            return False
        if path.startswith(('diffusers.pipelines', 'diffusers.models.transformers',
                            'diffusers.models.autoencoders', 'diffusers.models.unets')):
            return True
        name = path.rsplit('.', 1)[-1]
        return name.endswith('Pipeline') or (name.endswith('Model') and name != 'ModelMixin') or name == 'AutoencoderKL'
    if path.startswith('transformers.'):
        name = path.rsplit('.', 1)[-1]
        return ((name.endswith('Model') and name not in ('PreTrainedModel', 'PretrainedModel'))
                or name.endswith('ModelWithProjection') or name.startswith('AutoModel'))
    return False


class Scanner:
    def __init__(self, added, rel, mode):
        self.added, self.rel, self.mode, self.hits = added, rel, mode, []
        self.context = []

    def hit(self, node, label):
        lo, hi = node.lineno, getattr(node, 'end_lineno', node.lineno)
        if self.added is None or any(lo <= ln <= hi for ln in self.added):
            identity = repr((self.context, label, ast.dump(node, include_attributes=False)))
            row = {'file':self.rel, 'pattern':label, 'line':lo,
                   'semantic_sha256': hashlib.sha256(identity.encode()).hexdigest()}
            if row not in self.hits:
                self.hits.append(row)

    def expr(self, node, names):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return names.get(node.id, ('symbol', node.id))
        if isinstance(node, ast.Attribute):
            base = self.expr(node.value, names)
            if isinstance(base, tuple) and base[0] == 'symbol':
                return ('symbol', base[1] + '.' + node.attr)
        if isinstance(node, ast.BinOp):
            a, b = self.expr(node.left,names), self.expr(node.right,names)
            if isinstance(a,str) and isinstance(b,str):
                if isinstance(node.op, ast.Add): return a + b
                if isinstance(node.op, ast.Div): return posixpath.join(a,b)
        if isinstance(node, ast.Call):
            f = self.expr(node.func,names)
            fn = f[1] if isinstance(f,tuple) and f[0] == 'symbol' else ''
            args = [self.expr(a,names) for a in node.args]
            if fn in ('importlib.import_module', '__import__') and args and isinstance(args[0],str):
                return ('symbol', args[0])
            if fn == 'getattr' and len(args)>=2 and isinstance(args[0],tuple) and isinstance(args[1],str):
                return ('symbol', args[0][1]+'.'+args[1])
            if fn in ('pathlib.Path','pathlib.PurePath','pathlib.PosixPath','Path') and args:
                if all(isinstance(a,str) for a in args): return posixpath.join(*args)
            if fn in ('os.path.join','posixpath.join') and args and all(isinstance(a,str) for a in args):
                return posixpath.join(*args)
        return None

    def visit(self, node, names):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                path = ((node.module + '.' if node.module else '') + alias.name
                        if isinstance(node, ast.ImportFrom) else alias.name)
                bound = alias.asname or (alias.name if isinstance(node,ast.ImportFrom) else alias.name.split('.')[0])
                names[bound] = ('symbol', path if alias.asname or isinstance(node,ast.ImportFrom) else bound)
                if self.mode == 'hf' and whole_hf(path):
                    self.hit(node, 'imports whole HF model/pipeline: '+path)
            return
        if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            for expr in getattr(node,'decorator_list',[]): self.visit(expr,names)
            for expr in getattr(node,'bases',[]):
                base = self.expr(expr,names)
                if self.mode == 'hf' and isinstance(base,tuple) and whole_hf(base[1]):
                    self.hit(expr,'subclasses whole HF model/pipeline: '+base[1])
                self.visit(expr,names)
            inner = dict(names)
            args = getattr(node,'args',None)
            if args:
                for a in (*args.posonlyargs,*args.args,*args.kwonlyargs): inner[a.arg] = None
                for a in (args.vararg,args.kwarg):
                    if a: inner[a.arg] = None
                for expr in (*args.defaults,*args.kw_defaults):
                    if expr: self.visit(expr,names)
            self.context.append(node.name)
            self.block(node.body,inner)
            self.context.pop()
            names[node.name] = None
            return
        if isinstance(node,(ast.Assign,ast.AnnAssign,ast.NamedExpr)):
            value = node.value
            if value: self.visit(value,names)
            val = self.expr(value,names) if value else None
            targets = node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target,ast.Name): names[target.id] = val
            return
        if isinstance(node,ast.Call):
            f = self.expr(node.func,names)
            fn = f[1] if isinstance(f,tuple) and f[0]=='symbol' else ''
            imported = self.expr(node,names)
            if self.mode == 'hf':
                if whole_hf(fn) or (fn.startswith('diffusers.') and fn.endswith('.from_pretrained')):
                    self.hit(node,'calls whole HF model or diffusers weight loader: '+fn)
                if fn in ('importlib.import_module','__import__') and isinstance(imported,tuple) and whole_hf(imported[1]):
                    self.hit(node,'dynamically imports whole HF model/pipeline: '+imported[1])
            else:
                args = [self.expr(a,names) for a in node.args]
                args += [self.expr(k.value,names) for k in node.keywords]
                if fn in ('os.getenv','os.environ.get','os.environ.__getitem__') and args and args[0] in PRIVATE_ENV:
                    self.hit(node,'reads private verifier environment: '+args[0])
                method = node.func.attr if isinstance(node.func,ast.Attribute) else fn
                receiver = self.expr(node.func.value,names) if isinstance(node.func,ast.Attribute) else None
                if method in IO and any(private_path(v) for v in args+[receiver]):
                    self.hit(node,'accesses private verifier path through '+method)
        if self.mode == 'private' and isinstance(node,ast.Subscript):
            root = self.expr(node.value,names)
            key = self.expr(node.slice,names)
            if root == ('symbol','os.environ') and key in PRIVATE_ENV:
                self.hit(node,'reads private verifier environment: '+key)
        for child in ast.iter_child_nodes(node): self.visit(child,names)

    def block(self, body, names):
        for node in body: self.visit(node,names)


def scan(txt, added, rel, mode, baseline=None):
    tree = ast.parse(txt)
    scanner = Scanner(added,rel,mode)
    scanner.block(tree.body,{})
    if mode != 'hf' or baseline is None or added is None:
        return scanner.hits


    previous = scan(baseline, None, rel, mode)
    allowance = Counter(h['semantic_sha256'] for h in previous)
    current = scan(txt, None, rel, mode)
    added_hits = {(h['line'], h['pattern'], h['semantic_sha256']) for h in scanner.hits}
    for h in current:
        identity = (h['line'], h['pattern'], h['semantic_sha256'])
        if identity not in added_hits and allowance[h['semantic_sha256']] > 0:
            allowance[h['semantic_sha256']] -= 1
    retained = []
    for h in scanner.hits:
        key = h['semantic_sha256']
        if allowance[key] > 0:
            allowance[key] -= 1
        else:
            retained.append(h)
    return retained
