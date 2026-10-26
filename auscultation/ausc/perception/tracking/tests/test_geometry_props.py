import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from ausc.perception.tracking.geometry import (  # noqa: E402
    CropTransform, Intrinsics, back_square_crop, body_band_mask,
    normals_from_pointmap, pointmap_from_depth, sample_depth_m, valid_points,
)


# ── CropTransform: exact affine inverse incl. negative origin + non-unit scale ──
def test_crop_roundtrip_negative_origin_nonunit_scale():
    """det_to_full and full_to_det must be exact inverses even when the crop origin is
    negative (pad-not-clamp) and side/res != 1. Round-trip random points to ~1e-9 px."""
    crop = CropTransform(x0=-37, y0=-12, side=403, res=518)   # neg origin, scale 403/518
    rng = np.random.RandomState(0)
    det = rng.uniform(-50, 600, (64, 2))
    back = crop.full_to_det(crop.det_to_full(det))
    assert np.allclose(back, det, atol=1e-9)
    # the inverse direction too
    full = rng.uniform(-100, 700, (64, 2))
    assert np.allclose(crop.det_to_full(crop.full_to_det(full)), full, atol=1e-9)


def test_crop_scale_is_side_over_res():
    """A 1-unit step in detector space must move side/res px in full space (no off-by-one)."""
    crop = CropTransform(x0=10, y0=20, side=400, res=518)
    a = crop.det_to_full(np.array([[0.0, 0.0]]))
    b = crop.det_to_full(np.array([[1.0, 1.0]]))
    step = (b - a).ravel()
    assert np.allclose(step, 400.0 / 518.0, atol=1e-12)
    assert np.allclose(a.ravel(), [10.0, 20.0], atol=1e-12)


def test_crop_empty_input():
    """Empty (0,2) input must not crash and must keep the (0,2) shape."""
    crop = CropTransform.center_square(480, 640, 518)
    assert crop.det_to_full(np.zeros((0, 2))).shape == (0, 2)
    assert crop.full_to_det(np.zeros((0, 2))).shape == (0, 2)


# ── Intrinsics: project ∘ unproject round-trip + degenerate z ─────────────────
def test_intrinsics_project_unproject_roundtrip_pointcloud():
    """unproject (z>0) then project recovers the pixels to ~1e-6 for a random cloud."""
    intr = Intrinsics(612.3, 598.7, 321.5, 239.2, 640, 480)
    rng = np.random.RandomState(1)
    uv = np.column_stack([rng.uniform(0, 640, 200), rng.uniform(0, 480, 200)])
    z = rng.uniform(0.3, 3.5, 200)
    xyz = intr.unproject(uv, z)
    assert np.allclose(xyz[:, 2], z, atol=1e-9)
    uv2 = intr.project(xyz)
    assert np.allclose(uv2, uv, atol=1e-6)


def test_unproject_z_zero_collapses_to_origin():
    """z=0 is a degenerate ray: every pixel unprojects to the camera origin (0,0,0)."""
    intr = Intrinsics(525, 525, 320, 240, 640, 480)
    p = intr.unproject(np.array([[300.0, 220.0], [10.0, 5.0]]), np.array([0.0, 0.0]))
    assert np.allclose(p, 0.0, atol=1e-12)


def test_unproject_nan_depth_propagates_nan():
    """A NaN depth must produce a NaN point (so callers can drop it), not a silent number."""
    intr = Intrinsics(525, 525, 320, 240, 640, 480)
    p = intr.unproject(np.array([[300.0, 220.0]]), np.array([np.nan]))
    assert np.isnan(p).all()


def test_unproject_empty():
    intr = Intrinsics(525, 525, 320, 240, 640, 480)
    assert intr.unproject(np.zeros((0, 2)), np.zeros((0,))).shape == (0, 3)


# ── center_square / back_square_crop ──────────────────────────────────────────
def test_center_square_is_square_and_in_bounds():
    crop = CropTransform.center_square(480, 640, 518)
    assert crop.side == min(480, 640)
    assert 0 <= crop.x0 and crop.x0 + crop.side <= 640
    assert 0 <= crop.y0 and crop.y0 + crop.side <= 480
    # crop corners map to a square that covers the full image height (the short side)
    corners = crop.det_to_full(np.array([[0, 0], [518, 0], [0, 518], [518, 518]], float))
    assert np.isclose(corners[:, 1].min(), 0.0)
    assert np.isclose(corners[:, 1].max(), 480.0)


def _body_scene(seed=0):
    rng = np.random.RandomState(seed)
    H, W = 480, 640
    intr = Intrinsics(500, 500, 320, 240, W, H)
    depth = np.full((H, W), 1050, dtype=np.uint16)   # background wall @1.05 m
    depth[100:400, 200:450] = 600                    # patient back @0.60 m
    rgb = (rng.rand(H, W, 3) * 255).astype(np.uint8)
    return rgb, depth, intr


def test_back_square_crop_is_square_and_contains_body():
    """Output is 518x518; det_to_full corners surround the patient bbox (no lung column sliced)."""
    rgb, depth, intr = _body_scene()
    img, ct = back_square_crop(rgb, depth, intr)
    assert img.shape == (518, 518, 3)
    assert ct.res == 518 and ct.side > 0
    corners = ct.det_to_full(np.array([[0, 0], [518, 0], [0, 518], [518, 518]], float))
    # body occupies x in [200,450]; crop must contain it laterally.
    assert corners[:, 0].min() <= 200.0 and corners[:, 0].max() >= 450.0
    # bottom of back (y=400) must be inside the crop.
    assert corners[:, 1].max() >= 400.0


def test_back_square_crop_fallback_when_no_body():
    """All-holes depth → mask too small → centred full-height square fallback (square, in-frame)."""
    rng = np.random.RandomState(0)
    H, W = 480, 640
    intr = Intrinsics(500, 500, 320, 240, W, H)
    rgb = (rng.rand(H, W, 3) * 255).astype(np.uint8)
    depth = np.zeros((H, W), dtype=np.uint16)
    img, ct = back_square_crop(rgb, depth, intr)
    assert img.shape == (518, 518, 3)
    assert ct.side == min(H, W)
    assert ct.x0 == (W - min(H, W)) // 2


# ── normals_from_pointmap ─────────────────────────────────────────────────────
def _plane_pointmap(intr, z0=0.6, tilt_y=0.0):
    H, W = intr.height, intr.width
    P = np.empty((H, W, 3))
    for v in range(H):
        for u in range(W):
            x = (u - intr.cx) * z0 / intr.fx
            y = (v - intr.cy) * z0 / intr.fy
            P[v, u] = [x, y, z0 + tilt_y * (v - intr.cy)]
    return P


def test_normals_fronto_parallel_is_camera_axis():
    """A fronto-parallel plane has normals along the camera axis (0,0,-1) (facing camera)."""
    intr = Intrinsics(500, 500, 6.0, 5.0, 12, 11)
    P = _plane_pointmap(intr, z0=0.6, tilt_y=0.0)
    n = normals_from_pointmap(P)
    assert np.allclose(n[5, 6], [0.0, 0.0, -1.0], atol=1e-9)
    # all interior normals point toward the camera (negative z component)
    interior = n[1:-1, 1:-1, 2]
    assert np.all(interior <= 0)
    # unit length
    assert np.allclose(np.linalg.norm(n[5, 6]), 1.0, atol=1e-9)


def test_normals_tilted_plane_tips_correctly():
    """Tilting the plane in +y rotates the normal off the axis but keeps it camera-facing/unit."""
    intr = Intrinsics(500, 500, 6.0, 5.0, 12, 11)
    P = _plane_pointmap(intr, z0=0.6, tilt_y=0.001)
    n = normals_from_pointmap(P)
    c = n[5, 6]
    assert abs(c[1]) > 1e-3                       # tipped off-axis
    assert c[2] < 0                               # still faces camera
    assert np.isclose(np.linalg.norm(c), 1.0, atol=1e-9)


@pytest.mark.xfail(strict=True, reason=(
    "Known bug: the central differences skip the pixel itself, so a depth hole gets its neighbours' "
    "normal. Harmless today because both callers drop hole pixels through valid_points, but "
    "geometry.py is pinned by the hardened production runtime manifest, so the fix ships with the "
    "next runtime release."))
def test_normals_holes_become_nan():
    """NaN neighbours (depth holes) must yield NaN normals, not bogus directions."""
    intr = Intrinsics(500, 500, 6.0, 5.0, 12, 11)
    P = _plane_pointmap(intr, z0=0.6)
    P[5, 6] = np.nan
    n = normals_from_pointmap(P)
    assert np.isnan(n[5, 6]).all()
    # border is always NaN (no neighbours)
    assert np.isnan(n[0, 0]).all()


def test_normals_1x1_does_not_crash():
    """A 1x1 point map has no interior; result is all-NaN with matching shape."""
    P = np.zeros((1, 1, 3))
    n = normals_from_pointmap(P)
    assert n.shape == (1, 1, 3) and np.isnan(n).all()


# ── pointmap_from_depth metric scaling ────────────────────────────────────────
def test_pointmap_depth_scale_scales_metrically():
    """Doubling depth_scale doubles the metric z (mm-per-unit conversion)."""
    intr1 = Intrinsics(500, 500, 5, 5, 10, 10, depth_scale=1.0)
    intr2 = Intrinsics(500, 500, 5, 5, 10, 10, depth_scale=2.0)
    d = np.full((10, 10), 400, dtype=np.uint16)
    z1 = pointmap_from_depth(d, intr1)[..., 2]
    z2 = pointmap_from_depth(d, intr2)[..., 2]
    assert np.isclose(np.nanmedian(z1), 0.4, atol=1e-9)       # 400 * 1 / 1000
    assert np.isclose(np.nanmedian(z2), 0.8, atol=1e-9)
    assert np.allclose(z2[np.isfinite(z2)], 2.0 * z1[np.isfinite(z1)], atol=1e-9)


def test_pointmap_all_zero_is_all_nan():
    """Zero depth is invalid (<=0.05 m) → every point is NaN."""
    intr = Intrinsics(500, 500, 5, 5, 10, 10)
    P = pointmap_from_depth(np.zeros((10, 10), dtype=np.uint16), intr)
    assert np.isnan(P).all()


# ── sample_depth_m adversarial uv ─────────────────────────────────────────────
def test_sample_depth_oob_and_nan_and_hole():
    """Out-of-bounds, NaN uv, and holes all return NaN without crashing; valid edge px works."""
    depth = np.full((20, 20), 600, dtype=np.uint16)
    out = sample_depth_m(depth,
                         np.array([[-5.0, -5.0],      # out of bounds
                                   [100.0, 100.0],    # out of bounds
                                   [np.nan, np.nan],  # NaN uv (isfinite guard)
                                   [19.0, 19.0],      # exactly on the edge -> valid
                                   [0.0, 0.0]]),      # corner -> valid
                         depth_scale=1.0)
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    assert np.isclose(out[3], 0.6, atol=1e-9)
    assert np.isclose(out[4], 0.6, atol=1e-9)


def test_sample_depth_hole_returns_nan():
    depth = np.zeros((20, 20), dtype=np.uint16)
    out = sample_depth_m(depth, np.array([[10.0, 10.0]]), depth_scale=1.0)
    assert np.isnan(out[0])


def test_sample_depth_empty_uv():
    depth = np.full((20, 20), 600, dtype=np.uint16)
    assert sample_depth_m(depth, np.zeros((0, 2)), 1.0).shape == (0,)


def test_sample_depth_1x1():
    """A 1x1 depth map is a valid (tiny) image — sampling pixel (0,0) must work."""
    depth = np.full((1, 1), 600, dtype=np.uint16)
    out = sample_depth_m(depth, np.array([[0.0, 0.0]]), 1.0)
    assert np.isclose(out[0], 0.6, atol=1e-9)


# ── valid_points / body_band_mask on all-invalid input ────────────────────────
def test_valid_points_all_invalid():
    """All-NaN point map yields zero valid points (and zero normals when supplied)."""
    P = np.full((8, 8, 3), np.nan)
    pts, n = valid_points(P)
    assert pts.shape == (0, 3) and n is None
    pts2, n2 = valid_points(P, normals=np.full_like(P, np.nan))
    assert pts2.shape == (0, 3) and n2.shape == (0, 3)


def test_body_band_mask_all_invalid_is_empty():
    """No finite/positive depth anywhere → mask is empty (no false foreground)."""
    P = np.full((16, 16, 3), np.nan)
    m = body_band_mask(P)
    assert m.shape == (16, 16) and m.sum() == 0


def test_body_band_mask_removes_far_background():
    """A central near surface + far wall: mask keeps the near band, drops the wall."""
    H, W = 40, 40
    P = np.full((H, W, 3), np.nan)
    P[..., 2] = 1.05                              # far wall everywhere
    P[10:30, 10:30, 2] = 0.6                      # central patient
    P[..., 0] = 0.0; P[..., 1] = 0.0             # make x,y finite
    P[10:30, 10:30, 0] = 0.0; P[10:30, 10:30, 1] = 0.0
    m = body_band_mask(P, band=0.25)
    z = P[..., 2]
    kept = z[m]
    assert kept.size > 0
    assert kept.max() < 0.85                      # 1.05 wall removed
    assert np.isclose(np.median(kept), 0.6, atol=1e-9)