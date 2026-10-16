#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Minimal, dependency-free safetensors reader/writer for numpy arrays, plus exact bfloat16 conversion.

WHY NOT THE safetensors PACKAGE. The synthetic LoRA generator and the dense-delta reference must run where only
numpy exists, such as a preparation environment before any engine is installed, and the
reference must not share code with the engine it judges. The format is small enough to implement exactly:

    8 bytes little-endian u64 N | N bytes of JSON header | raw tensor bytes

The header maps each tensor name to {"dtype", "shape", "data_offsets": [begin, end]} (offsets relative to the end
of the header) and may carry "__metadata__": {str: str}. This writer is DETERMINISTIC: tensors in sorted name
order, header JSON with sorted keys and compact separators, padded with spaces to a multiple of 8 bytes (what the
official library does, so its reader accepts the file). The output format is compatible with
safetensors.torch.load_file.

bfloat16: numpy has no bf16 dtype, so BF16 tensors travel as uint16 bit patterns. `to_bf16_bits` rounds to nearest
even (the same rounding torch's .to(torch.bfloat16) applies); `from_bf16_bits` is exact.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_NP_OF = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32,
          "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}
_ST_OF = {np.dtype(v): k for k, v in _NP_OF.items()}


def to_bf16_bits(x) -> np.ndarray:
    """float32 -> uint16 bfloat16 bit patterns, round to nearest even. NaN is refused (never produced here)."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if np.isnan(a).any():
        raise ValueError("to_bf16_bits: NaN")
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    r = (u + 0x7FFF + lsb) >> 16
    return r.astype(np.uint16)


def from_bf16_bits(b) -> np.ndarray:
    u = np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def is_bf16_exact(x) -> bool:
    a = np.asarray(x, dtype=np.float32)
    return bool(np.array_equal(from_bf16_bits(to_bf16_bits(a)), a))


class Tensor:
    """A named tensor to write: `data` is a numpy array; `dtype` is the safetensors dtype string."""

    __slots__ = ("data", "dtype")

    def __init__(self, data: np.ndarray, dtype: str):
        self.data = np.ascontiguousarray(data)
        self.dtype = dtype


def bf16(x) -> Tensor:
    return Tensor(to_bf16_bits(x), "BF16")


def dumps(tensors: dict, metadata: dict | None = None) -> bytes:
    """Serialize {name: Tensor | np.ndarray} deterministically."""
    header: dict = {}
    blobs = []
    offset = 0
    for name in sorted(tensors):
        t = tensors[name]
        if not isinstance(t, Tensor):
            arr = np.ascontiguousarray(t)
            if arr.dtype not in _ST_OF:
                raise TypeError(f"{name}: unsupported numpy dtype {arr.dtype}")
            t = Tensor(arr, _ST_OF[arr.dtype])
        raw = t.data.tobytes()
        header[name] = {"dtype": t.dtype, "shape": list(t.data.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    if metadata:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise TypeError("safetensors metadata must be {str: str}")
        header["__metadata__"] = dict(sorted(metadata.items()))
    h = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h += b" " * ((8 - len(h) % 8) % 8)
    return struct.pack("<Q", len(h)) + h + b"".join(blobs)


def save(path, tensors: dict, metadata: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(dumps(tensors, metadata))


def load(path) -> tuple[dict, dict]:
    """-> ({name: np.ndarray}, metadata). BF16 tensors are returned as exact float32 values."""
    with open(path, "rb") as f:
        buf = f.read()
    (n,) = struct.unpack("<Q", buf[:8])
    header = json.loads(buf[8:8 + n])
    meta = header.pop("__metadata__", {}) or {}
    base = 8 + n
    out = {}
    for name, info in header.items():
        b, e = info["data_offsets"]
        raw = buf[base + b: base + e]
        shape = tuple(info["shape"])
        if info["dtype"] == "BF16":
            out[name] = from_bf16_bits(np.frombuffer(raw, dtype=np.uint16)).reshape(shape)
        elif info["dtype"] in _NP_OF:
            out[name] = np.frombuffer(raw, dtype=_NP_OF[info["dtype"]]).reshape(shape).copy()
        else:
            raise TypeError(f"{name}: unsupported safetensors dtype {info['dtype']}")
    return out, meta


def header(path) -> dict:
    """The raw header (with __metadata__), without reading tensor bytes."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))
