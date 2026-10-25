"""Property / invariant / adversarial tests for the TORCH-FREE geometry helpers in detector.py.

Covers ``normals_from_depth_m``, ``_fill_depth_holes`` and ``NormalsDetector._back_plausible``
(called unbound with self=None since it ignores self). No model checkpoint is ever loaded.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from ausc.perception.tracking.detector import (  # noqa: E402
    normals_from_depth_m,
    _fill_depth_holes,
    NormalsDetector,
)

# ``_back_plausible`` ignores ``self`` (only reads its dc_m arg), so we can call it unbound.
_back_plausible = lambda dc: NormalsDetector._back_plausible(None, dc)  # noqa: E731


# --------------------------------------------------------------------------------------
# _fill_depth_holes
# --------------------------------------------------------------------------------------
def test_fill_no_holes_is_identity():
    """A depth map with no holes (<=0) must be returned unchanged (and equal)."""
    D = np.full((16, 16), 1.5, np.float32)
    out = _fill_depth_holes(D)
    assert np.array_equal(out, D)


def test_fill_all_holes_returns_finite_zeros():
    """Degenerate guard: an all-hole map cannot be filled — must return FINITE output (never crash,
    never propagate NaN), all-zero for an all-zero input."""
    Z = np.zeros((16, 16), np.float32)
    out = _fill_depth_holes(Z)
    assert out.shape == Z.shape and np.isfinite(out).all() and not (out > 0).any()
    # the bug this guards: an all-NaN map must NOT pass NaN through (NaN <= 0 is False)
    N = np.full((16, 16), np.nan, np.float32)
    assert np.isfinite(_fill_depth_holes(N)).all()


def test_fill_single_valid_pixel_fills_whole_map():
    """One valid pixel surrounded by holes → nearest-valid fill makes the entire map that value;
    no pixel may remain <=0."""
    S = np.zeros((12, 12), np.float32)
    S[7, 3] = 2.0
    out = _fill_depth_holes(S)
    assert np.allclose(out, 2.0)
    assert (out > 0).all()


def test_fill_checkerboard_leaves_no_holes_and_keeps_valid_values():
    """A checkerboard of holes must be fully filled (no <=0 anywhere) and each filled pixel takes a
    value that actually existed among the valid pixels (nearest-valid, never invented)."""
    rng = np.random.default_rng(7)
    cb = np.zeros((16, 16), np.float32)
    valid_vals = np.array([0.8, 1.1, 1.4, 2.0], np.float32)
    # place distinct valid depths on the even-even cells, holes elsewhere
    for (i, j), v in zip([(i, j) for i in range(0, 16, 2) for j in range(0, 16, 2)],
                         rng.choice(valid_vals, size=64)):
        cb[i, j] = v
    out = _fill_depth_holes(cb)
    assert (out > 0).all()                         # no hole left
    assert np.isin(out, valid_vals).all()          # only ever copies an existing valid value


def test_fill_preserves_already_valid_pixels():
    """Nearest-valid fill must not overwrite pixels that were already valid."""
    rng = np.random.default_rng(1)
    D = (1.0 + 0.5 * rng.random((20, 20))).astype(np.float32)
    holes = rng.random((20, 20)) < 0.3
    D[holes] = 0.0
    valid_before = ~holes
    out = _fill_depth_holes(D)
    assert np.allclose(out[valid_before], D[valid_before])
    assert (out > 0).all()


def test_fill_no_nonpositive_output_when_any_valid():
    """Output invariant: if the input had ANY valid pixel, the filled map has no <=0 entries."""
    rng = np.random.default_rng(3)
    D = (0.7 + rng.random((24, 24))).astype(np.float32)
    D[rng.random((24, 24)) < 0.55] = 0.0           # 55% holes — still recoverable
    assert (D > 0).any()
    out = _fill_depth_holes(D)
    assert (out > 0).all()


def test_fill_does_not_leak_nan_from_sensor_noise():
    """ADVERSARIAL: a NaN pixel from the sensor (nan<=0 is False, so it is *not* counted as a hole)
    must not be smeared into neighbouring real holes. The genuine <=0 hole must still be filled
    with a finite value and the result must stay finite."""
    D = np.full((10, 10), 1.0, np.float32)
    D[2, 2] = np.nan                                # sensor NaN
    D[5, 5] = 0.0                                   # a genuine hole
    out = _fill_depth_holes(D)
    assert out[5, 5] > 0                            # the real hole got a positive fill
    assert np.isfinite(out).all()                  # no NaN may propagate into the output


# --------------------------------------------------------------------------------------
# normals_from_depth_m
# --------------------------------------------------------------------------------------
def test_normals_flat_plane_faces_camera():
    """A fronto-parallel plane → unit normals ≈ (0,0,-1) (pointing toward the camera, -z)."""
    n = normals_from_depth_m(np.full((48, 48), 1.0, np.float32))[2:-2, 2:-2]
    assert np.allclose(n.mean(0).mean(0), [0.0, 0.0, -1.0], atol=1e-3)
    # every interior normal is a unit vector
    norms = np.linalg.norm(n, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_normals_tilted_plane_sign_correct_and_camera_facing():
    """Depth increasing with +x tilts the surface away in +x: normal gains +x, keeps -z (facing cam)."""
    xs = np.arange(64)[None, :].astype(np.float32)
    Dt = (1.0 + 0.003 * xs).repeat(64, 0)
    nt = normals_from_depth_m(Dt)[2:-2, 2:-2]
    assert nt[..., 0].mean() > 1e-3                 # tips toward +x
    assert nt[..., 2].mean() < 0.0                  # still camera-facing
    # opposite tilt → opposite x sign
    Dt2 = (1.0 - 0.003 * xs + 0.003 * 64).repeat(64, 0)
    nt2 = normals_from_depth_m(Dt2)[2:-2, 2:-2]
    assert nt2[..., 0].mean() < -1e-3


def test_normals_holes_are_zero():
    """Pixels where depth==0 must have an exactly-zero normal (no leakage from neighbours)."""
    D = np.full((20, 20), 1.0, np.float32)
    D[5:9, 6:10] = 0.0
    n = normals_from_depth_m(D)
    assert np.allclose(n[5:9, 6:10], 0.0)


def test_normals_all_zero_input_is_all_zero_output():
    """Degenerate: an all-hole depth map → all-zero normals, no crash, finite."""
    n = normals_from_depth_m(np.zeros((16, 16), np.float32))
    assert np.allclose(n, 0.0)
    assert np.isfinite(n).all()


def test_normals_tiny_arrays_do_not_crash():
    """1x1 and 2xN depth maps have no interior; must return correctly-shaped all-zero normals."""
    for shape in [(1, 1), (2, 5), (5, 2), (3, 3)]:
        n = normals_from_depth_m(np.full(shape, 1.0, np.float32))
        assert n.shape == (shape[0], shape[1], 3)
        # the border is always zero; for (3,3) only the single center pixel is interior
        assert np.isfinite(n).all()


def test_normals_finite_under_nan_depth():
    """ADVERSARIAL: a NaN depth pixel must not poison the normal map with NaN/inf. The detector
    feeds this into a uint8 image and a model; non-finite normals are a real failure mode."""
    D = np.full((16, 16), 1.0, np.float32)
    D[8, 8] = np.nan
    n = normals_from_depth_m(D)
    assert np.isfinite(n).all()


def test_normals_scale_invariant_in_metres_units():
    """SCALE-INVARIANCE claim: normals are unit vectors built from cross-products of points that all
    scale linearly with z, so multiplying the WHOLE depth map by k leaves every unit normal
    unchanged (k in {0.5, 2}). Holes (==0) stay holes under scaling."""
    rng = np.random.default_rng(11)
    D = (1.0 + 0.4 * rng.random((40, 40))).astype(np.float32)
    D[rng.random((40, 40)) < 0.1] = 0.0            # include some holes
    n1 = normals_from_depth_m(D)
    for k in (0.5, 2.0):
        nk = normals_from_depth_m((D * k).astype(np.float32))
        # interior, finite normals must match to tight tolerance
        m = np.isfinite(n1).all(-1) & np.isfinite(nk).all(-1)
        assert np.allclose(n1[m], nk[m], atol=1e-5)


def test_normals_independent_of_intrinsics_offset_for_flat_plane():
    """A flat plane is (0,0,-1) regardless of principal point — guards against a cx/cy sign bug."""
    D = np.full((32, 32), 1.2, np.float32)
    n = normals_from_depth_m(D, cx=5.0, cy=27.0)[2:-2, 2:-2]
    assert np.allclose(n.mean(0).mean(0), [0.0, 0.0, -1.0], atol=2e-3)


# --------------------------------------------------------------------------------------
# NormalsDetector._back_plausible  (METRES — the OOD gate)
# --------------------------------------------------------------------------------------
def test_back_plausible_real_back_true():
    """A synthetic real patient back in METRES: median ~1.1 m, depth-extent >80 mm → plausible."""
    rng = np.random.default_rng(5)
    dc = (1.1 + 0.05 * rng.standard_normal((64, 64))).astype(np.float32)
    v = dc[(dc > 0.2) & (dc < 3.0)]
    rng_mm = (np.percentile(v, 90) - np.percentile(v, 10)) * 1000.0
    assert rng_mm > 80.0                            # sanity: this fixture really has extent
    assert _back_plausible(dc) is True


def test_back_plausible_flat_wall_false():
    """A constant-depth flat wall has ~0 depth-extent → NOT a body-back."""
    flat = np.full((64, 64), 1.1, np.float32)
    assert _back_plausible(flat) is False


def test_back_plausible_empty_and_allzero_false():
    """Empty / all-zero crops (nothing in [0.2,3.0] m) → not plausible, and must not crash."""
    assert _back_plausible(np.zeros((64, 64), np.float32)) is False
    assert _back_plausible(np.full((64, 64), 0.0, np.float32)) is False


def test_back_plausible_far_wall_false():
    """A far wall (median 2.5 m) is out of the 0.4–1.6 m body band → rejected even if it had extent."""
    rng = np.random.default_rng(9)
    far = (2.5 + 0.05 * rng.standard_normal((64, 64))).astype(np.float32)
    assert _back_plausible(far) is False


def test_back_plausible_requires_metres_not_millimetres():
    """CRITICAL units check: the gate is defined in METRES. The *same* real back expressed in
    millimetres (median ~1100) lands entirely outside the [0.2,3.0] window, so the gate must
    reject it. If this returned True the depth_scale conversion upstream would be wrong."""
    rng = np.random.default_rng(13)
    back_m = (1.1 + 0.05 * rng.standard_normal((64, 64))).astype(np.float32)
    assert _back_plausible(back_m) is True          # metres: accepted
    back_mm = (back_m * 1000.0).astype(np.float32)  # same scene, wrong (mm) units
    assert _back_plausible(back_mm) is False         # mm: rejected → forces correct metre conversion


def test_back_plausible_too_few_valid_pixels_false():
    """If <12% of pixels fall in the body band the crop is rejected (mostly-hole / mostly-OOD)."""
    rng = np.random.default_rng(17)
    dc = np.zeros((64, 64), np.float32)             # mostly holes
    n = dc.size
    idx = rng.choice(n, size=int(0.08 * n), replace=False)
    flat = dc.reshape(-1)
    flat[idx] = 1.1
    dc = flat.reshape(64, 64)
    assert _back_plausible(dc) is False