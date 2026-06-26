"""Tests for the streaming auscultation tracker (numpy/scipy/cv2 only — no torch/GPU)."""

import os
import sys

import numpy as np
import pytest

# import the package as top-level `tracking` (avoids importing the heavy ausc package)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tracking.canonical import CanonicalModel  # noqa: E402
from tracking.ekf import EKFConfig, PointEKF  # noqa: E402
from tracking.geometry import (CropTransform, Intrinsics, axial_noise_var,  # noqa: E402
                               body_band_mask, make_T, point_to_plane_icp, pointmap_from_depth,
                               ransac_rigid, rigid_align, transform_points)
from tracking.replay import (SyntheticDetector, make_synthetic_sequence,  # noqa: E402
                             run_over_frames)
from tracking.rgbd_odometry import FeatureField  # noqa: E402
from tracking.tracker import StreamingAuscultationTracker, TrackerConfig  # noqa: E402
from tracking.detector import LUNG_ORDER, iuv_locate  # noqa: E402
from tracking.flow import LKFlow  # noqa: E402
from tracking.replay import _rot, _surface  # noqa: E402

ADDRS = [(1, 0.5, 0.5)] * len(LUNG_ORDER)


# ── geometry ────────────────────────────────────────────────────────────────
def test_intrinsics_roundtrip():
    intr = Intrinsics(525, 525, 320, 240, 640, 480)
    uv = np.array([[300.0, 220.0], [400.0, 260.0]])
    z = np.array([0.6, 0.8])
    xyz = intr.unproject(uv, z)
    uv2 = intr.project(xyz)
    assert np.allclose(uv, uv2, atol=1e-6)


def test_crop_transform_inverse():
    crop = CropTransform.center_square(480, 640, 518, up_bias=0.12)
    det = np.array([[100.0, 200.0], [300.0, 400.0]])
    full = crop.det_to_full(det)
    assert np.allclose(crop.full_to_det(full), det, atol=1e-6)


def test_kabsch_recovers_rigid():
    rng = np.random.RandomState(0)
    src = rng.randn(20, 3)
    R = _rot(0.1, -0.2, 0.15)
    t = np.array([0.3, -0.1, 0.05])
    dst = (R @ src.T).T + t
    Rr, tr, s = rigid_align(src, dst)
    assert np.allclose(Rr, R, atol=1e-6)
    assert np.allclose(tr, t, atol=1e-6)


def test_icp_recovers_pose():
    obj, nrm, _ = _surface(n=80)
    src = obj + np.array([0, 0, 0.6])               # place in front of camera
    R = _rot(np.radians(3), np.radians(-2), 0.0)
    t = np.array([0.01, -0.008, 0.006])
    T_true = make_T(R, t)
    dst = transform_points(T_true, src)
    dst_n = (R @ nrm.T).T
    T_est, fit, rmse = point_to_plane_icp(src, dst, dst_n, max_iters=30, max_corr_dist=0.05)
    err_t = np.linalg.norm(T_est[:3, 3] - t)
    assert fit > 0.8
    assert err_t < 3e-3                              # < 3 mm translation error


# ── EKF ──────────────────────────────────────────────────────────────────────
def test_ekf_converges_and_gates_outliers():
    truth = np.array([0.1, -0.05, 0.6])
    ekf = PointEKF(truth + 0.03, EKFConfig())
    rng = np.random.RandomState(0)
    for _ in range(60):
        ekf.predict(1 / 30)
        z = truth + rng.normal(0, 0.003, 3)
        ekf.update(z, var=(0.003) ** 2)
    assert np.linalg.norm(ekf.position - truth) < 3e-3
    # a gross outlier must be rejected
    accepted = ekf.update(truth + np.array([0.3, 0.0, 0.0]), var=(0.003) ** 2)
    assert accepted is False
    assert np.linalg.norm(ekf.position - truth) < 3e-3


# ── IUV semantic locate ───────────────────────────────────────────────────────
def test_iuv_locate():
    part = np.ones((40, 40), dtype=int)
    uv = np.zeros((2, 40, 40))
    uv[0] = np.linspace(0, 1, 40)[None, :]          # U varies along x
    uv[1] = np.linspace(0, 1, 40)[:, None]          # V varies along y
    loc = iuv_locate(part, uv, (1, 0.5, 0.25))
    assert loc is not None
    x, y = loc
    assert abs(x - 19.5) < 2 and abs(y / 39 - 0.25) < 0.05
    assert iuv_locate(np.zeros((40, 40), dtype=int), uv, (1, 0.5, 0.5)) is None


# ── end-to-end synthetic ──────────────────────────────────────────────────────
def test_e2e_static_high_accuracy():
    frames, intr = make_synthetic_sequence(n_frames=60, motion=False, breathing=False,
                                           depth_noise_mm=1.0, seed=2)
    det = SyntheticDetector(intr, noise_px=0.5)
    cfg = TrackerConfig(detect_interval_s=0.5)
    tr = StreamingAuscultationTracker(det, intr, cfg, addrs=ADDRS)
    m = run_over_frames(tr, frames, det)
    assert tr.canonical.initialized
    assert m["median_err_mm"] < 6.0
    assert m["frac_tracking"] > 0.8


def test_e2e_dynamic_tracks_through_motion_and_breathing():
    frames, intr = make_synthetic_sequence(n_frames=150, motion=True, breathing=True,
                                           depth_noise_mm=2.0, seed=3)
    det = SyntheticDetector(intr, noise_px=1.0)
    cfg = TrackerConfig(detect_interval_s=0.5)             # detector throttled to 2 Hz
    tr = StreamingAuscultationTracker(det, intr, cfg, addrs=ADDRS)
    m = run_over_frames(tr, frames, det)
    assert tr.canonical.initialized
    assert m["median_err_mm"] < 12.0
    assert m["p90_err_mm"] < 25.0
    assert m["frac_tracking"] > 0.6


def test_memory_bounded_over_long_run():
    frames, intr = make_synthetic_sequence(n_frames=200, motion=True, breathing=True, seed=4)
    det = SyntheticDetector(intr)
    tr = StreamingAuscultationTracker(det, intr, TrackerConfig(), addrs=ADDRS)
    run_over_frames(tr, frames, det)
    assert tr.canonical.surfels is not None
    assert len(tr.canonical.surfels) <= tr.canonical.max_surfels


def test_identity_preserved_points_distinct():
    """Each tracked point stays nearest to its own GT (no identity swaps)."""
    frames, intr = make_synthetic_sequence(n_frames=120, motion=True, breathing=True, seed=5)
    det = SyntheticDetector(intr, noise_px=1.0)
    tr = StreamingAuscultationTracker(det, intr, TrackerConfig(detect_interval_s=0.5), addrs=ADDRS)
    swaps = 0
    last = frames[-1]
    out = None
    for fr in frames:
        det.set_truth(fr.gt_pixels)
        out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
    for k, tp in enumerate(out.points):
        if not np.isfinite(tp.xyz_camera).all():
            continue
        d = np.linalg.norm(last.gt_xyz_cam - tp.xyz_camera, axis=1)
        if int(np.argmin(d)) != k:
            swaps += 1
    assert swaps == 0


# ── RGB-D specific (aligned depth camera) ─────────────────────────────────────
def test_ransac_rigid_rejects_outliers():
    rng = np.random.RandomState(0)
    src = rng.randn(40, 3)
    R = _rot(0.05, 0.1, -0.05); t = np.array([0.02, -0.01, 0.03])
    dst = (R @ src.T).T + t
    dst[:12] += rng.randn(12, 3) * 0.5            # 30% gross outliers
    Rr, tr, fit = ransac_rigid(src, dst, thresh=0.01, iters=200, rng=rng)
    assert np.linalg.norm(tr - t) < 5e-3
    assert fit > 0.6


def _true_pose(t):
    ax = np.radians(4.0) * np.sin(2 * np.pi * 0.10 * t)
    ay = np.radians(5.0) * np.sin(2 * np.pi * 0.07 * t + 1.0)
    tx = 0.02 * np.sin(2 * np.pi * 0.05 * t); ty = 0.015 * np.sin(2 * np.pi * 0.06 * t + 0.5)
    return _rot(ax, ay, 0.0), np.array([tx, ty, 0.6])


def test_rgbd_odometry_recovers_motion_on_smooth_surface():
    """Feature VO must recover camera/patient motion on a geometrically smooth (but RGB-textured)
    back — the case where geometry-only ICP slides tangentially."""
    frames, intr = make_synthetic_sequence(n_frames=25, motion=True, breathing=False,
                                           depth_noise_mm=1.0, background=True, seed=11)
    field = FeatureField(intr)
    g0 = LKFlow.to_gray(frames[0].rgb)
    field.reset(g0, frames[0].depth, np.eye(4))
    R0, tr0 = _true_pose(frames[0].t)
    T = np.eye(4)
    for fr in frames[1:]:
        T, fit, _ = field.estimate(LKFlow.to_gray(fr.rgb), fr.depth, T)
    Ri, tri = _true_pose(frames[-1].t)
    R_true = Ri @ R0.T; t_true = tri - R_true @ tr0
    # VO alone (no absolute correction) over 24 frames; in the tracker it is re-grounded by the
    # detector every ~0.5 s, so drift never accumulates this far. Tangential lock works.
    assert np.linalg.norm(T[:3, 3] - t_true) < 12e-3
    assert fit > 0.5


def test_body_mask_removes_background():
    frames, intr = make_synthetic_sequence(n_frames=1, motion=False, breathing=False,
                                           depth_noise_mm=0.0, background=True, seed=1)
    P = pointmap_from_depth(frames[0].depth, intr, stride=4)
    mask = body_band_mask(P, band=0.25)
    z = P[..., 2]
    kept = z[mask & np.isfinite(z)]
    assert kept.size > 100
    assert kept.max() < 0.85                # the 1.05 m background wall is removed
    assert abs(np.median(kept) - 0.6) < 0.05


def test_axial_noise_grows_with_depth():
    # σ ≈ max(floor, k·z²): grows quadratically in the far range, floor-limited near range.
    assert axial_noise_var(2.0) > axial_noise_var(1.0) > axial_noise_var(0.6)
    assert axial_noise_var(0.1) == pytest.approx((1.5e-3) ** 2)   # near-range floor


def test_respiratory_signal_detected():
    fr_b, intr = make_synthetic_sequence(n_frames=120, motion=False, breathing=True, seed=20)
    det = SyntheticDetector(intr, noise_px=0.5)
    tr = StreamingAuscultationTracker(det, intr, TrackerConfig(detect_interval_s=0.5), addrs=ADDRS)
    out = None
    for fr in fr_b:
        det.set_truth(fr.gt_pixels)
        out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
    assert out.respiratory_amplitude_m > 0.001          # ~6 mm breathing is picked up


def test_surface_gate():
    obj, _, _ = _surface(n=20)
    patch = obj[:30] + np.array([0, 0, 0.6])            # coherent surface patch
    assert StreamingAuscultationTracker._on_surface(patch, 40.0) is True
    scattered = np.random.RandomState(0).randn(10, 3) * 0.3 + np.array([0, 0, 0.6])
    assert StreamingAuscultationTracker._on_surface(scattered, 40.0) is False


# ── track-primary: the detector is the weak link, don't let it break the track ────────────────
def test_bad_pose_detections_do_not_break_track():
    """An unstable detector that returns garbage (large displacement) for a window of frames must
    NOT corrupt the track: the agreement gate rejects it; tracking carries on."""
    frames, intr = make_synthetic_sequence(n_frames=160, motion=True, breathing=True, seed=33)
    det = SyntheticDetector(intr, noise_px=1.0)
    cfg = TrackerConfig(drift_audit_s=0.3)        # detector fires often → gate must reject bad ones
    tr = StreamingAuscultationTracker(det, intr, cfg, addrs=ADDRS)
    errs, out = [], None
    for i, fr in enumerate(frames):
        det.set_truth(fr.gt_pixels + (180.0 if 70 <= i < 110 else 0.0))   # garbage during 70..110
        out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
        if out.initialized and 80 <= i < 110:     # measure during the corrupted window
            for k, tp in enumerate(out.points):
                if np.isfinite(tp.xyz_camera).all():
                    errs.append(np.linalg.norm(tp.xyz_camera - fr.gt_xyz_cam[k]))
    assert np.median(errs) < 10e-3                 # track held despite garbage detections
    # no identity swap at the end
    last = frames[-1]
    for k, tp in enumerate(out.points):
        if np.isfinite(tp.xyz_camera).all():
            assert int(np.argmin(np.linalg.norm(last.gt_xyz_cam - tp.xyz_camera, axis=1))) == k


def test_init_needs_confirmation():
    """A single (possibly bad) detection must not anchor the canonical model."""
    frames, intr = make_synthetic_sequence(n_frames=5, motion=False, breathing=False, seed=1)
    det = SyntheticDetector(intr, noise_px=0.5)
    tr = StreamingAuscultationTracker(det, intr, TrackerConfig(init_confirm_frames=3), addrs=ADDRS)
    det.set_truth(frames[0].gt_pixels)
    out = tr.process(frames[0].rgb, frames[0].depth, t=frames[0].t, depth_scale=1.0)
    assert not out.initialized                      # 1 detection is not enough to anchor
    for fr in frames[1:3]:
        det.set_truth(fr.gt_pixels)
        out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=1.0)
    assert out.initialized                           # confirmed after 3 consistent detections


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
