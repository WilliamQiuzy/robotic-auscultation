"""Lossless, self-identifying, strictly-verified depth-frame codec.

Aligned depth was stored as raw ``.npy`` (~4.05 MiB/frame -> ~124 MB/s at 30 fps).
Depth is smooth, so a lossless zlib pass is byte-exact yet ~14x smaller on real
frames.  This codec fixes the earlier mistakes that made compression unsafe to
ship:

* **Self-identifying, not masquerading.**  A compressed frame is framed with a
  distinct magic + the exact uncompressed length, and new files use the
  ``.npy.zlib`` suffix and the ``formal_per_frame_zlib_npy_v1`` storage id -- it is
  never presented as a bare ``.npy`` that ``np.load`` might read as garbage.
* **Strict decode.**  ``decode_depth`` requires a clean zlib EOF, rejects any
  trailing / concatenated bytes, enforces the declared length, caps the
  decompressed size (no decompression bomb), and requires the inner ``.npy`` to be
  fully consumed.  A frame truncated by even one byte fails to decode.
* **Full verification before seal.**  ``verify_depth_file`` fully decodes and
  checks dtype/shape, so an undecodable frame can never be sealed
  ``diagnostic_complete``.

Legacy raw ``.npy`` sessions still decode unchanged.
"""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np

_NPY_MAGIC = b"\x93NUMPY"                       # a bare numpy .npy stream (legacy)
_MAGIC = b"WAMZDPT1"                            # WAM zlib depth container, v1
_HEADER = struct.Struct("<Q")                  # uncompressed length (little-endian u64)
_ZLIB_LEVEL = 3                                # lossless; ~14x on real depth
_MAX_DEPTH_RAW_BYTES = 64 * 1024 * 1024        # hard cap (8K*8K*u16 << this); anti-bomb

DEPTH_SUFFIX = ".npy.zlib"                      # new, self-identifying
LEGACY_DEPTH_SUFFIX = ".npy"
DEPTH_STORAGE_ID = "formal_per_frame_zlib_npy_v1"
LEGACY_DEPTH_STORAGE_ID = "formal_per_frame_npy_v1"


class DepthCodecError(ValueError):
    """A depth payload is malformed, truncated, oversized, or not fully consumed."""


def encode_depth(array: np.ndarray) -> bytes:
    """Losslessly encode one depth frame into the self-identifying container."""
    if not isinstance(array, np.ndarray) or array.ndim != 2 or array.dtype != np.uint16:
        raise DepthCodecError("depth frame must be a 2-D uint16 ndarray")
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    raw = buffer.getvalue()
    if len(raw) > _MAX_DEPTH_RAW_BYTES:
        raise DepthCodecError("depth frame exceeds the maximum encodable size")
    return _MAGIC + _HEADER.pack(len(raw)) + zlib.compress(raw, _ZLIB_LEVEL)


def _load_npy_fully_consumed(raw: bytes) -> np.ndarray:
    stream = io.BytesIO(raw)
    array = np.load(stream, allow_pickle=False)
    if stream.read(1):
        raise DepthCodecError("depth .npy stream has trailing bytes")
    if array.dtype != np.uint16:
        raise DepthCodecError(f"depth frame array must be uint16, got {array.dtype}")
    if array.ndim != 2 or not array.size:
        raise DepthCodecError("depth frame array must be a non-empty 2-D array")
    return np.ascontiguousarray(array)


def decode_depth(data: bytes) -> np.ndarray:
    """Strictly decode a depth frame (new container or legacy raw ``.npy``)."""
    if data[:6] == _NPY_MAGIC:
        return _load_npy_fully_consumed(data)                       # legacy uncompressed
    if data[: len(_MAGIC)] != _MAGIC:
        raise DepthCodecError("unrecognised depth payload (no WAM depth magic)")
    body = data[len(_MAGIC):]
    if len(body) < _HEADER.size:
        raise DepthCodecError("depth payload is truncated in the header")
    (declared_len,) = _HEADER.unpack(body[: _HEADER.size])
    if declared_len == 0 or declared_len > _MAX_DEPTH_RAW_BYTES:
        raise DepthCodecError("depth declared length is zero or exceeds the cap")
    stream = body[_HEADER.size:]
    decompressor = zlib.decompressobj()
    try:
        # Bound the OUTPUT at the cap.  If any compressed input remains after
        # producing cap bytes, the decompressed size exceeds the cap -> reject
        # BEFORE flush(), so an oversized/bomb payload is never fully expanded.
        raw = decompressor.decompress(stream, _MAX_DEPTH_RAW_BYTES)
        if decompressor.unconsumed_tail:
            raise DepthCodecError("depth decompressed output exceeds the maximum size")
        raw += decompressor.flush()
    except zlib.error as exc:
        raise DepthCodecError("depth zlib payload is corrupt") from exc
    if not decompressor.eof:
        raise DepthCodecError("depth zlib stream is truncated or exceeds the cap")
    if decompressor.unused_data:
        raise DepthCodecError("depth payload has trailing bytes after the zlib stream")
    if len(raw) != declared_len:
        raise DepthCodecError("depth decompressed length does not match the header")
    return _load_npy_fully_consumed(raw)


def load_depth(path) -> np.ndarray:
    with open(path, "rb") as handle:
        return decode_depth(handle.read())


def verify_depth_file(path, *, expected_shape=None) -> "tuple[tuple, np.dtype]":
    """Fully decode + validate one depth frame; raise on any problem.

    Used by the End-of-session seal: an undecodable / corrupt / truncated frame
    can never be sealed ``diagnostic_complete``.  Returns (shape, dtype).
    """
    array = load_depth(path)
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        raise DepthCodecError(f"depth shape {array.shape} != expected {tuple(expected_shape)}")
    return tuple(array.shape), array.dtype


def is_compressed_depth_name(name: str) -> bool:
    return str(name).endswith(DEPTH_SUFFIX)


__all__ = [
    "DEPTH_STORAGE_ID",
    "DEPTH_SUFFIX",
    "DepthCodecError",
    "LEGACY_DEPTH_STORAGE_ID",
    "LEGACY_DEPTH_SUFFIX",
    "decode_depth",
    "encode_depth",
    "is_compressed_depth_name",
    "load_depth",
    "verify_depth_file",
]
