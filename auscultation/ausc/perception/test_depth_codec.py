"""Tests for the lossless depth codec (bit-exact + legacy .npy backward compat)."""

from __future__ import annotations

import numpy as np
import pytest

from ausc.perception import depth_codec


def _depth(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 4000, size=(1080, 1920)).astype(np.uint16)
    a[a < 600] = 0  # invalid regions, like real aligned depth
    return a


def test_roundtrip_is_bit_exact():
    a = _depth()
    b = depth_codec.decode_depth(depth_codec.encode_depth(a))
    assert b.dtype == np.uint16 and b.shape == a.shape
    assert np.array_equal(a, b)


def test_encoding_is_smaller():
    a = _depth()
    assert len(depth_codec.encode_depth(a)) < a.nbytes


def test_legacy_raw_npy_still_decodes(tmp_path):
    a = _depth(1)
    p = tmp_path / "depth_000001.npy"
    with open(p, "wb") as f:
        np.save(f, a, allow_pickle=False)  # legacy uncompressed
    assert np.array_equal(depth_codec.load_depth(p), a)


def test_new_compressed_file_decodes(tmp_path):
    a = _depth(2)
    p = tmp_path / "depth_000002.npy"
    with open(p, "wb") as f:
        f.write(depth_codec.encode_depth(a))
    assert np.array_equal(depth_codec.load_depth(p), a)


def test_peek_meta_matches_for_both_formats(tmp_path):
    a = _depth(3)
    comp = tmp_path / "c.npy"
    comp.write_bytes(depth_codec.encode_depth(a))
    raw = tmp_path / "r.npy"
    with open(raw, "wb") as f:
        np.save(f, a, allow_pickle=False)
    for p in (comp, raw):
        shape, dtype = depth_codec.peek_depth_meta(p)
        assert shape == (1080, 1920) and dtype == np.uint16


def test_bad_payload_raises():
    with pytest.raises(ValueError):
        depth_codec.decode_depth(b"this is neither npy nor a zlib stream")
