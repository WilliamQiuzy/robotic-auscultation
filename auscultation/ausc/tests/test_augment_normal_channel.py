"""Depth-normal-channel augmentation (2026-08-07 audit fix).

The production landmark model trains on depth-derived normals and its only
"sensor noise" augmentation was ``random.gauss(0, 4)`` — a SINGLE SCALAR added
to the whole image (a DC offset), so the model never saw the per-pixel noise
that real Orbbec depth produces. These tests pin the fix: genuine per-pixel
noise, a spatially-correlated component, neutral holes, and preserved mean.
"""
import numpy as np

from ausc.augment import augment_normal_channel


def _base(res=128):
    return np.full((res, res, 3), 127.0, dtype=np.float32)


def test_noise_is_per_pixel_not_a_dc_offset():
    # The bug: rng.gauss(0,4) added ONE scalar to every pixel -> residual has
    # zero spatial variance. The fix must produce real per-pixel variation.
    rng = np.random.RandomState(0)
    base = _base()
    out = augment_normal_channel(base.copy(), rng, hole_prob=0.0)
    resid = out - base
    # per-pixel std within a single channel must be clearly non-zero
    assert float(resid[..., 0].std()) > 2.0
    # and it must NOT be a constant offset: neighbouring pixels differ
    assert float(np.abs(np.diff(resid[..., 0], axis=1)).mean()) > 1.0


def test_mean_is_approximately_preserved():
    rng = np.random.RandomState(1)
    base = _base()
    out = augment_normal_channel(base.copy(), rng, hole_prob=0.0)
    assert abs(float(out.mean()) - 127.0) < 3.0    # zero-mean noise, no systematic shift


def test_holes_are_neutral_127_when_drawn():
    rng = np.random.RandomState(2)
    base = np.full((128, 128, 3), 200.0, dtype=np.float32)
    # force holes every call
    out = augment_normal_channel(base.copy(), rng, sigma=0.0, corr_sigma=0.0,
                                 hole_prob=1.0, max_holes=3, hole_frac=0.1)
    neutral = np.all(np.isclose(out, 127.0), axis=2)
    assert bool(neutral.any())                      # at least one hole rendered
    assert float(neutral.mean()) < 0.6              # but not the whole image


def test_returns_float32_and_shape_preserved():
    rng = np.random.RandomState(3)
    out = augment_normal_channel(_base(96), rng)
    assert out.dtype == np.float32 and out.shape == (96, 96, 3)


def test_deterministic_under_seed():
    a = augment_normal_channel(_base(), np.random.RandomState(7))
    b = augment_normal_channel(_base(), np.random.RandomState(7))
    assert np.array_equal(a, b)
