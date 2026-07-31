"""Lossless, fast depth-frame codec for diagnostic/production capture.

Aligned depth was previously stored as raw ``.npy`` (1080x1920 uint16 =
~4.05 MiB/frame -> ~124 MB/s at 30 fps).  Depth is very smooth, so a plain
zlib pass is lossless yet ~14x smaller at level 3 while encoding in ~12 ms/frame
(comfortably inside the 30 fps budget).  This changes only the *encoding* of each
depth file, not its shape/dtype or the alignment: decoding returns the byte-exact
same array, so every downstream consumer (offline-RL conversion, authenticity
finalizer, re-alignment, viz) sees identical depth.

Files keep the ``.npy`` name; ``load_depth`` auto-detects the payload, so legacy
raw-``.npy`` sessions still load unchanged.
"""

from __future__ import annotations

import io
import zlib

import numpy as np
import numpy.lib.format as _npy_format

_NPY_MAGIC = b"\x93NUMPY"          # a bare numpy .npy stream (legacy, uncompressed)
_ZLIB_LEVEL = 3                    # 14x, ~12 ms/frame -- lossless, fits 30 fps


def encode_depth(array: np.ndarray) -> bytes:
    """Losslessly encode one depth frame (zlib-wrapped .npy, level 3)."""
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return zlib.compress(buffer.getvalue(), _ZLIB_LEVEL)


def decode_depth(data: bytes) -> np.ndarray:
    """Decode a depth frame written by ``encode_depth`` or a legacy raw ``.npy``."""
    if data[:6] == _NPY_MAGIC:
        return np.load(io.BytesIO(data), allow_pickle=False)   # legacy uncompressed
    try:
        raw = zlib.decompress(data)
    except zlib.error as exc:
        raise ValueError("depth payload is neither a raw .npy nor zlib-wrapped") from exc
    return np.load(io.BytesIO(raw), allow_pickle=False)


def load_depth(path) -> np.ndarray:
    """Read a depth frame file (new compressed or legacy raw), returning uint16."""
    with open(path, "rb") as handle:
        return decode_depth(handle.read())


def peek_depth_meta(path) -> "tuple[tuple, np.dtype]":
    """Return (shape, dtype) without decoding pixels -- O(1) per frame.

    Reads only the small ``.npy`` header (raw, or the leading bytes of the zlib
    stream), so per-frame End-of-session validation stays cheap for long sessions.
    """
    with open(path, "rb") as handle:
        prefix = handle.read(2048)
    if prefix[:6] == _NPY_MAGIC:
        stream = io.BytesIO(prefix)
    else:
        try:
            stream = io.BytesIO(zlib.decompressobj().decompress(prefix, 1024))
        except zlib.error as exc:
            raise ValueError("depth payload is neither a raw .npy nor zlib-wrapped") from exc
    version = _npy_format.read_magic(stream)
    shape, _fortran, dtype = _npy_format._read_array_header(stream, version)
    return tuple(shape), dtype


__all__ = ["decode_depth", "encode_depth", "load_depth", "peek_depth_meta"]
