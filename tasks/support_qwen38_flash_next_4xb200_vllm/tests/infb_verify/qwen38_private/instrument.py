#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Finite native-output interventions, including captured CUDA graph replay (MTP3 production).

This diagnostic is never enabled during performance measurements. Its mapping names the loaded
native owners the component record bound (class, method, output), not alternate implementations.
It is installed before model construction, so the device counters, observations and intervention
scalar are included in the engine's ordinary captured graphs. Worker RPC only changes those
scalars; it does not replace graph execution.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

_SITES = {}
_INSTALLED = []
_TARGET = (None, 0.)
_TARGET_IDENTIFIERS = set()
_OWNERS = {}
_LIBRARY = None


def install(mapping, *, engine_root="/app", model_dir=None):
    """Install declarative observers before compilation/model construction.

    Each row has ``component`` (qsa/ple), ``class`` (qualified native module
    class), ``method`` and ``output_argument`` (zero-based excluding self, or
    null for a returned tensor). A fused boundary can use its enclosing native
    result; both implementations must use the same timing boundary separately.

    With ``model_dir`` (the production launch), the checkpoint's declared
    architecture is first loaded through the engine's own model registry, the
    import the engine performs before constructing the model. An owner's module
    is then never imported ahead of the model module that imports it, whatever
    import order the implementation's modules have among themselves.
    """
    import torch

    if not isinstance(mapping, list) or not mapping:
        raise ValueError("native mapping must be a nonempty list")
    components = {row.get("component") for row in mapping}
    if components != {"qsa", "ple"}:
        raise ValueError("mapping must identify useful QSA and PLE results")
    if model_dir is not None:
        _load_declared_architecture(model_dir)
    for row in mapping:
        if set(row) != {"component", "class", "method", "output_argument"}:
            raise ValueError("unknown or missing native mapping fields")
        module_name, class_name = row["class"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        if not issubclass(cls, torch.nn.Module):
            raise ValueError("native owner must be a loaded model module")
        method = getattr(cls, row["method"])
        source = inspect.getsourcefile(inspect.unwrap(method))
        if not source or not Path(source).resolve().is_relative_to(Path(engine_root).resolve()):
            raise ValueError("native observer must identify engine source")
        if any(old[0] is cls and old[1] == row["method"] for old in _INSTALLED):
            raise ValueError("duplicate native boundary")
        argument = row["output_argument"]
        if argument is not None and not (isinstance(argument, str) or (type(argument) is int and argument >= 0)):
            raise ValueError("output_argument must be an argument name, a nonnegative position or null")
        provenance = dict(row, source=str(Path(source).resolve()),
                          source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())

        _register_operator()
        attribute = '_qwen38_observer_' + row['method']
        original_init = cls.__init__

        def constructor(native, attribute, specification, provenance):
            @functools.wraps(native)
            def initialize(owner, *args, **kwargs):
                native(owner, *args, **kwargs)
                identifier = len(_OWNERS) + 1
                _OWNERS[identifier] = (owner, specification, provenance)
                setattr(owner, attribute, identifier)
            return initialize

        def wrap(native, specification, attribute):
            signature = inspect.signature(native)

            @functools.wraps(native)
            def observed(owner, *args, **kwargs):
                result = native(owner, *args, **kwargs)
                where = specification['output_argument']
                if where is None:
                    output = result
                elif isinstance(where, str):
                    output = signature.bind(owner, *args, **kwargs).arguments[where]
                else:
                    output = args[where]
                # The opaque mutation remains inside the compiled/captured
                # graph. Python bookkeeping and capture-state queries stay
                # outside Dynamo tracing, without disabling fullgraph mode.
                torch.ops.qwen38_diagnostic.observe(output, getattr(owner, attribute))
                return result
            return observed

        setattr(cls, '__init__', constructor(original_init, attribute, dict(row), provenance))
        setattr(cls, row['method'], wrap(method, dict(row), attribute))
        _INSTALLED.append((cls, row['method'], method))
        _INSTALLED.append((cls, '__init__', original_init))


def _load_declared_architecture(model_dir):
    from vllm.model_executor.models.registry import ModelRegistry

    architectures = json.loads((Path(model_dir) / "config.json").read_text()).get("architectures") or []
    # As the engine resolves a model: the first declared architecture that loads.
    if not any(ModelRegistry._try_load_model_cls(name) is not None for name in architectures):
        raise ValueError(f"no declared architecture of the checkpoint loads through the engine registry: {architectures}")


def _register_operator():
    import torch
    global _LIBRARY
    if _LIBRARY is not None: return
    library = torch.library.Library('qwen38_diagnostic', 'DEF')
    library.define('observe(Tensor(a!) output, int identifier) -> ()')
    library.impl('observe', _observe, 'CUDA')
    library.impl('observe', lambda output, identifier: None, 'Meta')
    _LIBRARY = library


def _observe(output, identifier):
    import torch
    owner, specification, provenance = _OWNERS[identifier]
    if not output.is_floating_point() or not output.numel() or output.device.type != 'cuda':
        raise ValueError('native output must be a nonempty floating CUDA tensor')
    key = (identifier, tuple(output.shape), tuple(output.stride()))
    site = _SITES.get(key)
    if site is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('native observer shape was not warmed before capture')
        amount = _TARGET[1] if identifier in _TARGET_IDENTIFIERS else 0.
        site = {'owner': owner, 'provenance': provenance, 'shape': list(output.shape),
                'stride': list(output.stride()), 'dtype': str(output.dtype),
                'amplitude': torch.full((), amount, device=output.device, dtype=torch.float32),
                'count': torch.zeros((), device=output.device, dtype=torch.int64),
                'sample': torch.zeros(32, device=output.device, dtype=output.dtype),
                'python_calls': 0, 'capture_entries': 0}
        _SITES[key] = site
    site['python_calls'] += 1
    site['capture_entries'] += int(torch.cuda.is_current_stream_capturing())
    site['count'].add_(1)
    width = output.shape[-1]
    pattern = torch.arange(width, device=output.device, dtype=torch.float32).remainder_(7).sub_(3)
    output.add_((pattern * site['amplitude']).to(output.dtype))
    flat_sample = output.reshape(-1)[:32]
    site['sample'].zero_()
    site['sample'][:flat_sample.numel()].copy_(flat_sample)


def control(model, *, component=None, owner=None, amplitude=0., reset=False):
    """Control already captured observers on objects in the loaded model."""
    import torch
    with torch.inference_mode():
        return _control(model, component=component, owner=owner, amplitude=amplitude, reset=reset)


def _control(model, *, component, owner, amplitude, reset):
    import math
    import torch

    global _TARGET, _TARGET_IDENTIFIERS

    if component not in (None, "qsa", "ple") or not math.isfinite(amplitude) or abs(amplitude) > 1:
        raise ValueError("invalid diagnostic intervention")
    modules = {id(module): name for name, module in model.named_modules()}
    if owner is not None and (not isinstance(owner, str) or owner not in modules.values()):
        raise ValueError('intervention owner is not in the loaded target model')
    torch.cuda.synchronize()
    _TARGET = (component, amplitude)
    _TARGET_IDENTIFIERS = {identifier for identifier, (module, spec, _) in _OWNERS.items()
        if id(module) in modules and spec['component'] == component
        and (owner is None or modules[id(module)] == owner)}
    result = []
    for site in _SITES.values():
        if id(site["owner"]) not in modules:
            # Profiling/speculator objects outside the inspected model cannot
            # provide its evidence. They remain unperturbed.
            site["amplitude"].zero_()
            continue
        selected = component == site["provenance"]["component"] and (
            owner is None or modules[id(site['owner'])] == owner)
        site["amplitude"].fill_(amplitude if selected else 0.)
        result.append({"identity": dict(site["provenance"], owner=modules[id(site["owner"])]),
                       "shape": site["shape"], "stride": site["stride"], "dtype": site["dtype"],
                       "device_calls": int(site["count"].item()), "python_calls": site["python_calls"],
                       "capture_entries": site["capture_entries"],
                       "sample": [] if reset else site["sample"].float().cpu().tolist()})
        if reset:
            site["count"].zero_(); site["sample"].zero_()
            site["python_calls"] = site["capture_entries"] = 0
    torch.cuda.synchronize()
    return result


class NativeObservationWorker:
    """Optional vLLM diagnostic worker extension; ordinary performance omits it."""

    def qwen38_observation(self, request):
        import torch.distributed as dist
        from qwen38.measurement.worker import protocol_identity
        from qwen38.serving.mtp import source_identity
        identity = {'rank': dist.get_rank(), 'protocol': protocol_identity()}
        try:
            options = json.loads(request)
            identity['request_id'] = options.pop('request_id')
            if options.pop('protocol') != identity['protocol']:
                raise ValueError('client and observer protocol differ')
            if set(options) - {"component", "owner", "amplitude", "reset"}:
                raise ValueError("unknown observation request fields")
            identity['source'] = source_identity(os.environ.get('QWEN38_ENGINE_ROOT', '/app'))
            identity['recipe'] = 'mtp3' if self.vllm_config.num_speculative_tokens == 3 else 'ordinary'
            rows = control(self.model_runner.get_model(), **options)
            json.dumps(rows, allow_nan=False)
            return {**identity, 'status': 'complete', 'sites': rows}
        except Exception as exc:
            return {**identity, 'status': 'failed', 'error': repr(exc)}


if os.environ.get("QWEN38_NATIVE_MAPPING"):
    install(json.loads(Path(os.environ["QWEN38_NATIVE_MAPPING"]).read_text()),
            engine_root=os.environ.get('QWEN38_ENGINE_ROOT', '/app'),
            model_dir=os.environ.get('QWEN38_MODEL_DIR', '/model'))
