"""Property / invariant / fuzz tests for the TORCH-FREE tracking primitives.

Targets (numpy + cv2 + scipy only, no torch / GPU / hardware / network):
  - canonical.voxel_downsample, CanonicalModel, PoseEstimate
  - flow.LKFlow              (cv2 optical flow on synthetic images)
  - rgbd_odometry.FeatureField

Goal: empirically surface bugs. Where an input *should* work but the code crashes
or returns a wrong value, the test asserts the CORRECT behaviour so it FAILS.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

import cv2  # noqa: E402

from ausc.perception.tracking.canonical import (  # noqa: E402
    voxel_downsample, CanonicalModel, PoseEstimate)
from ausc.perception.tracking.flow import LKFlow, FlowResult  # noqa: E402
from ausc.perception.tracking.rgbd_odometry import FeatureField  # noqa: E402
from ausc.perception.tracking.geometry import Intrinsics  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────
def _textured_gray(seed=42, h=200, w=200):
    rng = np.random.RandomState(seed)
    img = (rng.rand(h, w) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _trackable_corners(img, n=20, lo=25, hi=170):
    c = cv2.goodFeaturesToTrack(img, n, 0.01, 10)
    pts = c.reshape(-1, 2)
    keep = (pts[:, 0] > lo) & (pts[:, 0] < hi) & (pts[:, 1] > lo) & (pts[:, 1] < hi)
    return pts[keep]


def _ones_n(n):
    return np.tile([0.0, 0.0, 1.0], (n, 1))


# ══════════════════════════════════════════════════════════════════════════════
# voxel_downsample
# ══════════════════════════════════════════════════════════════════════════════
def test_voxel_output_count_le_input():
    rng = np.random.RandomState(0)
    pts = rng.uniform(0, 0.1, (300, 3))
    sp, sn = voxel_downsample(pts, _ones_n(300), 0.01)
    assert len(sp) <= len(pts)
    assert len(sp) == len(sn)


def test_voxel_identical_points_collapse_to_one():
    P = np.tile([0.5, 0.5, 0.5], (50, 1))
    sp, sn = voxel_downsample(P, _ones_n(50), 0.01)
    assert len(sp) == 1
    assert np.allclose(sp[0], [0.5, 0.5, 0.5])


def test_voxel_preserves_rough_geometry_centroid():
    rng = np.random.RandomState(1)
    voxel = 0.02
    pts = rng.uniform(0, 0.1, (200, 3))
    sp, _ = voxel_downsample(pts, _ones_n(200), voxel)
    # the downsampled centroid should stay within one voxel of the original centroid
    assert np.linalg.norm(sp.mean(0) - pts.mean(0)) < voxel


def test_voxel_empty_input_returns_empty():
    sp, sn = voxel_downsample(np.zeros((0, 3)), np.zeros((0, 3)), 0.01)
    assert len(sp) == 0 and len(sn) == 0


def test_voxel_single_point_returns_itself():
    sp, sn = voxel_downsample(np.array([[1.0, 2.0, 3.0]]), np.array([[0.0, 0.0, 1.0]]), 0.005)
    assert len(sp) == 1
    assert np.allclose(sp[0], [1.0, 2.0, 3.0])
    assert np.isfinite(sn).all()


def test_voxel_nan_row_does_not_corrupt_output():
    """A NaN input row must be dropped, not silently written into the surfel cloud.
    floor(nan).astype(int64) yields a garbage bucket and a NaN surfel survives -> a corrupt
    canonical surface that poisons every downstream ICP/KD-tree query. Correct: output finite."""
    pts = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    sp, sn = voxel_downsample(pts, _ones_n(3), 0.005)
    assert np.isfinite(sp).all(), "voxel_downsample leaked a NaN point into the output"
    assert np.isfinite(sn).all()


def test_voxel_normals_are_unit_or_zero():
    rng = np.random.RandomState(2)
    pts = rng.uniform(0, 0.1, (120, 3))
    nrm = rng.randn(120, 3)
    _, sn = voxel_downsample(pts, nrm, 0.02)
    norms = np.linalg.norm(sn, axis=1)
    assert np.all(norms <= 1.0 + 1e-6)  # averaged + renormalised, never blows up


# ══════════════════════════════════════════════════════════════════════════════
# CanonicalModel (uses voxel_downsample + estimate_pose fallbacks)
# ══════════════════════════════════════════════════════════════════════════════
def test_canonical_initialize_bounds_surfels():
    rng = np.random.RandomState(3)
    m = CanonicalModel(voxel=0.005, max_surfels=50)
    surf = rng.uniform(0, 1.0, (400, 3))
    m.initialize(rng.randn(10, 3), surf, _ones_n(400), [(1, 0.5, 0.5)] * 10)
    assert m.initialized
    assert m.surfels is not None and len(m.surfels) <= m.max_surfels
    assert m.points.shape == (10, 3)


def test_canonical_estimate_pose_coast_when_no_surfels_no_corr():
    m = CanonicalModel()
    est = m.estimate_pose(np.zeros((0, 3)), np.zeros((0, 3)), T_init=None)
    assert isinstance(est, PoseEstimate)
    assert est.method == "coast"
    assert est.T_cam_canon.shape == (4, 4)
    assert np.allclose(est.T_cam_canon, np.eye(4))


def test_canonical_estimate_pose_kabsch_from_correspondences():
    rng = np.random.RandomState(4)
    m = CanonicalModel()
    src = rng.randn(8, 3)
    R = cv2.Rodrigues(np.array([0.05, -0.03, 0.02]))[0]
    t = np.array([0.1, -0.05, 0.2])
    dst = (R @ src.T).T + t
    est = m.estimate_pose(np.zeros((0, 3)), np.zeros((0, 3)), T_init=None,
                          corr_canon=src, corr_cam=dst)
    assert est.method == "kabsch"
    assert np.allclose(est.T_cam_canon[:3, 3], t, atol=1e-6)
    assert np.allclose(est.T_cam_canon[:3, :3], R, atol=1e-6)


def test_canonical_snap_no_tree_is_identity():
    m = CanonicalModel()
    p = np.array([0.0, 0.0, 0.6])
    out, disp = m.snap_to_surface(p, None, np.zeros((0, 3)), np.zeros((0, 3)))
    assert np.allclose(out, p) and disp == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LKFlow
# ══════════════════════════════════════════════════════════════════════════════
def test_lkflow_recovers_known_integer_shift():
    img = _textured_gray(seed=42)
    shift = np.array([5.0, 3.0])  # (dx, dy)
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    img2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    pts = _trackable_corners(img)[:10]
    assert len(pts) >= 4
    res = LKFlow().track(img, img2, pts)
    assert isinstance(res, FlowResult)
    assert res.valid.any(), "no points tracked across a clean integer shift"
    disp = (res.pts - pts)[res.valid]
    err = np.linalg.norm(disp - shift, axis=1)
    assert np.median(err) < 1.0  # within ~1px


def test_lkflow_lost_points_are_marked_nan_and_invalid():
    img = _textured_gray(seed=7)
    # big shift with a BLACK border: the far-corner region becomes uniform black,
    # so the point there has nothing to lock onto and must be reported lost.
    img2 = cv2.warpAffine(img, np.float32([[1, 0, 60], [0, 1, 60]]),
                          (img.shape[1], img.shape[0]), borderValue=0)
    edge = np.array([[195.0, 195.0]])
    res = LKFlow().track(img, img2, edge)
    assert res.valid[0] == False  # noqa: E712
    assert not np.isfinite(res.pts[0]).all()  # NaN-filled where lost
    assert res.fb_err[0] == np.inf


def test_lkflow_out_of_bounds_seed_is_invalid():
    img = _textured_gray(seed=7)
    res = LKFlow().track(img, img, np.array([[500.0, 500.0]]))  # seed outside the image
    assert res.valid[0] == False  # noqa: E712
    assert not np.isfinite(res.pts[0]).all()


def test_lkflow_blank_region_is_low_confidence():
    blank = np.full((200, 200), 120, np.uint8)
    res = LKFlow().track(blank, blank, np.array([[100.0, 100.0]]))
    # untextured region: nothing to lock onto -> must not be reported as a confident track
    assert res.valid[0] == False  # noqa: E712


def test_lkflow_nan_input_point_reported_invalid():
    img = _textured_gray(seed=11)
    img2 = cv2.warpAffine(img, np.float32([[1, 0, 4], [0, 1, 2]]), (img.shape[1], img.shape[0]))
    pts = np.array([[np.nan, np.nan], [50.0, 50.0]])
    res = LKFlow().track(img, img2, pts)
    assert res.valid[0] == False  # noqa: E712  (NaN seed can never be a valid track)


def test_lkflow_empty_points_handled():
    """N=0 tracked points should yield N=0 results, not crash cv2's assertion."""
    img = _textured_gray(seed=1)
    img2 = _textured_gray(seed=2)
    res = LKFlow().track(img, img2, np.zeros((0, 2)))
    assert len(res.pts) == 0
    assert len(res.valid) == 0
    assert len(res.fb_err) == 0


def test_lkflow_to_gray_shapes():
    rgb = (np.random.RandomState(0).rand(40, 50, 3) * 255).astype(np.uint8)
    g = LKFlow.to_gray(rgb)
    assert g.shape == (40, 50)
    g2 = LKFlow.to_gray(g)  # already gray -> unchanged
    assert g2.shape == (40, 50)


# ══════════════════════════════════════════════════════════════════════════════
# FeatureField
# ══════════════════════════════════════════════════════════════════════════════
def _ff_intr():
    return Intrinsics(525, 525, 160, 120, 320, 240, depth_scale=1.0)


def _ff_scene(seed=7):
    rng = np.random.RandomState(seed)
    gray = cv2.GaussianBlur((rng.rand(240, 320) * 255).astype(np.uint8), (3, 3), 0)
    depth = np.full((240, 320), 600, np.uint16)  # flat plane at 0.6 m
    return gray, depth


def test_featurefield_reset_seeds_features():
    field = FeatureField(_ff_intr())
    gray, depth = _ff_scene()
    field.reset(gray, depth, np.eye(4))
    assert len(field.pix) > 0
    assert field.pix.shape[0] == field.canon.shape[0]
    assert np.isfinite(field.canon).all()


def test_featurefield_identity_frame_recovers_identity():
    field = FeatureField(_ff_intr())
    gray, depth = _ff_scene()
    field.reset(gray, depth, np.eye(4))
    T, fit, rmse = field.estimate(gray, depth, np.eye(4))
    assert fit > 0.9
    assert np.allclose(T, np.eye(4), atol=1e-4)
    assert rmse < 1e-3


def test_featurefield_recovers_known_tangential_shift():
    field = FeatureField(_ff_intr())
    gray, depth = _ff_scene()
    field.reset(gray, depth, np.eye(4))
    dpix = 4
    g2 = cv2.warpAffine(gray, np.float32([[1, 0, dpix], [0, 1, 0]]), (320, 240))
    T, fit, rmse = field.estimate(g2, depth, np.eye(4))
    assert fit > 0.8
    expected_mm = dpix * 0.6 / 525 * 1000.0
    assert abs(abs(T[0, 3]) * 1000.0 - expected_mm) < 1.5  # within 1.5 mm of geometry
    assert abs(T[1, 3]) < 2e-3 and abs(T[2, 3]) < 2e-3


def test_featurefield_all_hole_depth_does_not_crash():
    field = FeatureField(_ff_intr())
    gray, _ = _ff_scene()
    depth0 = np.zeros((240, 320), np.uint16)
    field.reset(gray, depth0, np.eye(4))
    assert len(field.pix) == 0  # nothing has valid depth -> no features
    T, fit, rmse = field.estimate(gray, depth0, np.eye(4))
    assert np.allclose(T, np.eye(4))  # falls back to T_init
    assert fit == 0.0 and rmse == np.inf


def test_featurefield_gray_none_returns_init():
    field = FeatureField(_ff_intr())
    _, depth = _ff_scene()
    T_init = np.eye(4); T_init[0, 3] = 0.123
    T, fit, rmse = field.estimate(None, depth, T_init)
    assert np.allclose(T, T_init)
    assert fit == 0.0 and rmse == np.inf


def test_featurefield_estimate_before_reset_falls_back_then_seeds():
    field = FeatureField(_ff_intr())
    gray, depth = _ff_scene()
    T, fit, rmse = field.estimate(gray, depth, np.eye(4))  # no reset first
    assert fit == 0.0  # first call cannot estimate
    assert len(field.pix) > 0  # but it self-seeds for next time


def test_featurefield_tiny_image_does_not_crash():
    field = FeatureField(_ff_intr())
    tiny = np.zeros((4, 4), np.uint8)
    dtiny = np.full((4, 4), 600, np.uint16)
    field.reset(tiny, dtiny, np.eye(4))  # must not raise
    assert len(field.pix) == 0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
