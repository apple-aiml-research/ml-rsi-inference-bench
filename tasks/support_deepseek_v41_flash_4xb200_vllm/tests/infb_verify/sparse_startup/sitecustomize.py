#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Diagnostic-child-only startup of the trusted sparse-indexer observer (the sparse-indexer contract).

`runtime_evidence.instrumented_server` puts this directory first on the instrumented launch's PYTHONPATH and sets
INFB_SPARSE_AUDIT_DIR; no other launch of the verifier has either, so the timed session and the reference never load
it. In every process of that launch Python imports this module at startup. It imports neither torch nor vLLM, creates
no CUDA context, process group or thread: it registers an import hook that, when the engine's model-runner module is
loaded, wraps the runner's `load_model` so that once the model is loaded -- before graph capture -- the trusted observer
(`/tests/infb_verify/sparse_probe.py`, loaded by path) is installed on it. A failure to install is recorded by the
observer and never raised into the engine. Any other sitecustomize on the path is then run, as Python would have.
"""
import os
import sys

_AUDIT = os.environ.get("INFB_SPARSE_AUDIT_DIR")
#: The engine's model-runner modules (V2 and V1) and their runner class.
_RUNNERS = {"vllm.v1.worker.gpu.model_runner": "GPUModelRunner", "vllm.v1.worker.gpu_model_runner": "GPUModelRunner"}


def _probe():
    import importlib.util

    name = "infb_sparse_probe"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.environ.get("TESTS_DIR", "/tests"), "infb_verify", "sparse_probe.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _patch(module, cls_name):
    cls = getattr(module, cls_name, None)
    if cls is None or getattr(cls, "_infb_sparse_patched", False) or not hasattr(cls, "load_model"):
        return
    orig = cls.load_model

    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            _probe().install(self)
        except Exception as exc:  # noqa: BLE001  (recorded by the observer where it can; never raised into the engine)
            try:
                os.makedirs(_AUDIT, exist_ok=True)
                with open(os.path.join(_AUDIT, f"startup-fault-{os.getpid()}.txt"), "w") as fh:
                    fh.write(f"{type(exc).__name__}: {exc}\n")
            except OSError:
                pass
        return out

    cls.load_model = load_model
    cls._infb_sparse_patched = True


if _AUDIT:
    import importlib.abc
    import importlib.machinery

    class _RunnerFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name not in _RUNNERS:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path, target)
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return None
            loader, cls_name = spec.loader, _RUNNERS[name]

            class _Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return loader.create_module(spec)

                def exec_module(self, module):
                    loader.exec_module(module)
                    _patch(module, cls_name)

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _RunnerFinder())

# Run any other sitecustomize on the path, as Python would have without this directory first.
_here = os.path.dirname(os.path.abspath(__file__))
for _entry in list(sys.path):
    _cand = os.path.join(_entry, "sitecustomize.py")
    if _entry and os.path.abspath(_entry) != _here and os.path.isfile(_cand):
        with open(_cand) as _fh:
            exec(compile(_fh.read(), _cand, "exec"), {"__name__": "sitecustomize", "__file__": _cand})
        break
