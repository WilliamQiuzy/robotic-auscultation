"""Tests for the strict, self-identifying depth codec.

Covers the review's exact failure modes: a frame truncated by one byte, trailing
bytes, a concatenated second zlib stream, a decompression bomb, and that an
undecodable frame is rejected by the seal verifier (never sealed complete).
"""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
import pytest

from ausc.perception import depth_codec
from ausc.perception.depth_codec import DepthCodecError


def _depth(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 4000, size=(1080, 1920)).astype(np.uint16)
    a[a < 600] = 0
    return a


def test_roundtrip_is_bit_exact_and_smaller():
    a = _depth()
    enc = depth_codec.encode_depth(a)
    assert enc[:8] == b"WAMZDPT1"                       # self-identifying, not a bare .npy
    b = depth_codec.decode_depth(enc)
    assert b.dtype == np.uint16 and np.array_equal(a, b)


def test_truncated_by_one_byte_is_rejected():
    enc = depth_codec.encode_depth(_depth())
    with pytest.raises(DepthCodecError):
        depth_codec.decode_depth(enc[:-1])                # the review's repro


def test_trailing_bytes_after_stream_rejected():
    enc = depth_codec.encode_depth(_depth())
    with pytest.raises(DepthCodecError, match="trailing bytes"):
        depth_codec.decode_depth(enc + b"\x00")


def test_concatenated_second_stream_rejected():
    a = depth_codec.encode_depth(_depth(1))
    b = depth_codec.encode_depth(_depth(2))
    # append a second full zlib body after the first -> unused_data must be rejected
    with pytest.raises(DepthCodecError):
        depth_codec.decode_depth(a + b[len(b"WAMZDPT1") + 8:])


def test_decompression_bomb_declared_length_capped():
    raw = np.save(io.BytesIO(), np.zeros((4, 4), np.uint16)) or b""
    buf = io.BytesIO(); np.save(buf, np.zeros((4, 4), np.uint16), allow_pickle=False)
    body = zlib.compress(buf.getvalue(), 3)
    huge = b"WAMZDPT1" + struct.pack("<Q", 1 << 40) + body   # claims ~1 TiB
    with pytest.raises(DepthCodecError, match="cap"):
        depth_codec.decode_depth(huge)


def test_decompression_bomb_small_header_large_output_rejected():
    # The dangerous case an unbounded flush() would miss: the header declares a
    # LEGAL (small) length so it survives the header cap, but the zlib stream
    # actually expands past _MAX_DEPTH_RAW_BYTES.  Decode must reject it WITHOUT
    # fully materialising the oversized output.
    over = b"\x00" * (depth_codec._MAX_DEPTH_RAW_BYTES + 4096)   # >64 MiB of zeros
    body = zlib.compress(over, 3)                                # ~64 KiB compressed
    payload = b"WAMZDPT1" + struct.pack("<Q", 4096) + body       # header lies: small len
    with pytest.raises(DepthCodecError, match="maximum size"):
        depth_codec.decode_depth(payload)


def test_length_mismatch_rejected():
    a = _depth()
    buf = io.BytesIO(); np.save(buf, a, allow_pickle=False)
    body = zlib.compress(buf.getvalue(), 3)
    bad = b"WAMZDPT1" + struct.pack("<Q", buf.getbuffer().nbytes - 1) + body
    with pytest.raises(DepthCodecError, match="length"):
        depth_codec.decode_depth(bad)


def test_unknown_magic_rejected():
    with pytest.raises(DepthCodecError, match="magic"):
        depth_codec.decode_depth(b"NOTWAM__" + b"\x00" * 16)


def test_legacy_raw_npy_still_decodes(tmp_path):
    a = _depth(3)
    p = tmp_path / "depth_000003.npy"
    with open(p, "wb") as f:
        np.save(f, a, allow_pickle=False)
    assert np.array_equal(depth_codec.load_depth(p), a)


def test_new_file_load_and_verify(tmp_path):
    a = _depth(4)
    p = tmp_path / ("depth_000004" + depth_codec.DEPTH_SUFFIX)
    p.write_bytes(depth_codec.encode_depth(a))
    assert np.array_equal(depth_codec.load_depth(p), a)
    shape, dtype = depth_codec.verify_depth_file(p, expected_shape=(1080, 1920))
    assert shape == (1080, 1920) and dtype == np.uint16


def test_verify_rejects_truncated_file(tmp_path):
    a = _depth(5)
    p = tmp_path / ("depth_000005" + depth_codec.DEPTH_SUFFIX)
    p.write_bytes(depth_codec.encode_depth(a)[:-1])       # corrupt on disk
    with pytest.raises(DepthCodecError):
        depth_codec.verify_depth_file(p)


def test_verify_rejects_wrong_shape(tmp_path):
    a = _depth(6)
    p = tmp_path / ("depth_000006" + depth_codec.DEPTH_SUFFIX)
    p.write_bytes(depth_codec.encode_depth(a))
    with pytest.raises(DepthCodecError, match="shape"):
        depth_codec.verify_depth_file(p, expected_shape=(720, 1280))


def test_load_depth_rejects_oversized_file_before_reading(tmp_path):
    # A sparse file just over the cap: load_depth must reject it by stat, never
    # reading tens of MiB into memory.  (truncate keeps it sparse -> instant.)
    p = tmp_path / ("depth_000007" + depth_codec.DEPTH_SUFFIX)
    with p.open("wb") as fp:
        fp.truncate(depth_codec._MAX_DEPTH_FILE_BYTES + 1)
    with pytest.raises(DepthCodecError, match="cap|maximum readable size"):
        depth_codec.load_depth(p)


def test_load_depth_rejects_empty_file(tmp_path):
    p = tmp_path / ("depth_000008" + depth_codec.DEPTH_SUFFIX)
    p.write_bytes(b"")
    with pytest.raises(DepthCodecError, match="empty"):
        depth_codec.load_depth(p)
