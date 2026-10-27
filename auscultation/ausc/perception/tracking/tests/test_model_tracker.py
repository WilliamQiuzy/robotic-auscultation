"""Tests for the model-based deformable tracker (torch-free: synthetic detector + scene)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
)))

from ausc.perception.tracking.replay import make_synthetic_sequence, SyntheticDetector  # noqa: E402
from ausc.perception.tracking.model_tracker import ModelBasedTracker, ModelConfig  # noqa: E402
from ausc.perception.tracking.production_geometry_gate import (  # noqa: E402
    PRODUCTION_DECISION_SURFACE,
    ProductionGeometryGate,
    ProductionGeometryProfile,
    production_geometry_profile_semantic_sha256,
)
from ausc.perception.tracking.detector import DetectorResult  # noqa: E402
from ausc.perception.tracking.geometry import CropTransform, Intrinsics  # noqa: E402
from ausc.perception.tracking.detector import normals_from_depth_m  # noqa: E402


def _production_geometry_gate():
    values = {
        "camera_profile_sha256": "a" * 64,
        "camera_projection_sha256": "b" * 64,
        "depth_noise_model_sha256": "c" * 64,
        "independent_model_sha256": "d" * 64,
        "lateral_noise_std_m": 1.0e-3,
        "axial_noise_std_m": 2.0e-3,
        "robust_delta_m": 3.0e-3,
        "max_scale_deviation_relative": 0.02,
        "depth_sample_window_px": 2,
        "min_depth_m": 0.05,
        "max_depth_m": 3.0,
        "min_depth_patch_valid_pixels": 9,
        "min_depth_patch_valid_fraction": 0.60,
        "max_depth_patch_mad_m": 0.004,
        "max_depth_patch_p90_p10_m": 0.020,
        "min_back_mask_support_fraction": 0.60,
        "max_independent_model_innovation_m": 0.010,
        "max_independent_model_innovation_sigma": 3.0,
        "decision_surface": PRODUCTION_DECISION_SURFACE,
    }
    values["semantic_sha256"] = production_geometry_profile_semantic_sha256(
        **values,
    )
    return ProductionGeometryGate(
        ProductionGeometryProfile(**values)
    )


def test_normals_from_depth_frontal_plane_faces_camera():
    """Data-engine geometry primitive: a fronto-parallel plane → normals ≈ (0,0,-1) (toward camera);
    a plane tilted in +x → normal gains a -x component. (Was untested.)"""
    D = np.full((64, 64), 1.0, np.float32)              # flat plane at 1 m
    n = normals_from_depth_m(D)[2:-2, 2:-2]             # interior
    assert np.allclose(n.mean(0).mean(0), [0, 0, -1], atol=1e-3)
    xs = np.arange(64)[None, :].astype(np.float32)
    Dt = (1.0 + 0.002 * xs).repeat(64, 0)              # depth increases with x → surface tilts away
    nt = normals_from_depth_m(Dt)[2:-2, 2:-2]
    assert nt[..., 0].mean() > 1e-3 and nt[..., 2].mean() < 0   # normal tips in +x, still camera-facing


def _run(noise_px, cfg=None, n=120):
    frames, intr = make_synthetic_sequence(n_frames=n)
    det = SyntheticDetector(intr, noise_px=noise_px)
    tr = ModelBasedTracker(det, intr, cfg or ModelConfig(init_frames=8))
    outs = []
    for fr in frames:
        det.set_truth(fr.gt_pixels)             # drive the synthetic detector with GT (+noise)
        outs.append((tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale), fr))
    return outs


def test_initializes_and_tracks_gt():
    outs = _run(noise_px=1.0)
    init = [o for o, _ in outs if o.initialized]
    assert len(init) > 90                        # initializes well within the clip
    errs = []
    for o, fr in outs:
        if not o.initialized:
            continue
        for k, p in enumerate(o.points):
            if np.isfinite(p.xyz_camera).all():
                errs.append(float(np.linalg.norm(p.xyz_camera - fr.gt_xyz_cam[k])))
    assert np.median(errs) < 0.015               # < 15 mm vs ground-truth 3D


def test_output_is_smooth_under_detector_noise():
    """High per-frame detector noise must NOT pass through to the output: the shared-pose model
    averages it out (this is the whole point vs per-point re-detection)."""
    outs = _run(noise_px=4.0)
    P = np.array([[p.xyz_camera for p in o.points] for o, _ in outs if o.initialized])
    dP = np.diff(P, axis=0)
    cm = np.median(dP, axis=1, keepdims=True)    # remove gross common-mode motion
    jit = np.linalg.norm(dP - cm, axis=2)
    assert np.median(jit) < 0.003                # < 3 mm output jitter despite 4 px detector noise


def _depth_jitter(P):
    """per-point frame-to-frame depth(z) jitter, common-mode (global) motion removed [m]."""
    dz = np.diff(P[:, :, 2], axis=0)
    cm = np.median(dz, axis=1, keepdims=True)
    return float(np.median(np.abs(dz - cm)))


def _run_depth(noise_px, depth_noise_mm, curv, cfg):
    frames, intr = make_synthetic_sequence(n_frames=120, depth_noise_mm=depth_noise_mm, curv=curv)
    det = SyntheticDetector(intr, noise_px=noise_px)
    tr = ModelBasedTracker(det, intr, cfg)
    P = []
    for fr in frames:
        det.set_truth(fr.gt_pixels)
        o = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
        if o.initialized:
            P.append([p.xyz_camera for p in o.points])
    return np.array(P)


def test_surface_snap_preserves_accuracy_and_stability():
    """Regression guard for the Step-1 surface-snap path. The real depth-jitter defect comes from
    clothed-back geometry + holes the synthetic harness does NOT model (synthetic depth jitter is
    already ~0.3 mm), so the *fix* is validated on the real clip, not here. Here we only require
    that turning surface_snap ON does not degrade GT accuracy or inject jitter, and exercises the
    snap code path (cloud build + ray∩plane) without error."""
    kw = dict(noise_px=3.0, depth_noise_mm=4.0, curv=1.6)
    P = _run_depth(cfg=ModelConfig(init_frames=8, surface_snap=True), **kw)
    assert len(P) > 90
    assert _depth_jitter(P) * 1000 < 3.0                  # stays stable with the snap path active
    Poff = _run_depth(cfg=ModelConfig(init_frames=8, surface_snap=False), **kw)
    assert _depth_jitter(P) <= _depth_jitter(Poff) + 0.001  # snap never worsens depth stability


class _ArmOccluder:
    """Wraps a detector and DROPS (NaN) chosen points to simulate the robot arm occluding part of
    the back from the detector. The test toggles ``self.drop`` per frame."""

    def __init__(self, base):
        self.base = base
        self.drop = set()

    def set_truth(self, g):
        self.base.set_truth(g)

    def detect(self, rgb, depth, intr):
        import dataclasses
        r = self.base.detect(rgb, depth, intr)
        if r is None or not self.drop:
            return r
        xy = r.xy_det.astype(float).copy()
        for k in self.drop:
            xy[k] = np.nan                       # detector cannot see this point (arm in the way)
        return dataclasses.replace(r, xy_det=xy)


def test_clean_run_all_points_measured():
    frames, intr = make_synthetic_sequence(n_frames=100)
    det = SyntheticDetector(intr, noise_px=1.0)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=8))
    outs = []
    for fr in frames:
        det.set_truth(fr.gt_pixels)
        outs.append(tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale))
    late = [o for o in outs[-30:] if o.initialized]
    assert late and np.median([o.n_measured for o in late]) >= 9   # ~all observed on clean data


def test_tracker_fails_closed_without_bound_frame_geometry_evidence():
    frames, intr = make_synthetic_sequence(
        n_frames=12, motion=True, breathing=False, depth_noise_mm=0.0,
    )
    det = SyntheticDetector(intr, noise_px=0.0)
    tracker = ModelBasedTracker(
        det,
        intr,
        ModelConfig(init_frames=3, coast_max=10),
        geometry_gate=_production_geometry_gate(),
    )
    output = None
    for frame in frames[:5]:
        det.set_truth(frame.gt_pixels)
        output = tracker.process(
            frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
        )
    assert output is not None and output.initialized
    assert output.pose_measured_this_frame is True
    assert output.pose_support_mask.shape == (10,)
    assert output.pose_support_mask.dtype == np.bool_
    # A detector point and its z coordinate cannot self-authorize. The tracker
    # has not received independently bound RGB-D/back-mask/model evidence.
    assert output.pose_geometry_accepted is False
    assert output.sim3_rotation_camera_from_patient.shape == (3, 3)
    assert output.sim3_scale > 0.0
    assert output.sim3_translation_camera_from_patient_m.shape == (3,)
    assert output.points[0].xyz_patient.shape == (3,)
    assert output.points[0].measurement_residual_m >= 0.0
    assert output.points[0].robot_occluded is False
    assert output.points[0].direct_geometry_accepted is False

    target = 0
    masked_frame = frames[5]
    det.set_truth(masked_frame.gt_pixels)
    mask = np.zeros(masked_frame.depth.shape, dtype=np.bool_)
    u, v = np.rint(masked_frame.gt_pixels[target]).astype(int)
    mask[v, u] = True
    masked = tracker.process(
        masked_frame.rgb,
        masked_frame.depth,
        t=masked_frame.t,
        depth_scale=masked_frame.depth_scale,
        robot_mask=mask,
    )
    assert masked.pose_measured_this_frame is True
    assert masked.points[target].robot_occluded is True
    assert masked.points[target].direct_geometry_accepted is False
    assert masked.points[target].measured is False

    held = tracker.process(
        None,
        None,
        t=masked_frame.t + 0.05,
        depth_scale=masked_frame.depth_scale,
    )
    assert held.pose_measured_this_frame is False
    assert not held.pose_support_mask.any()
    assert held.pose_geometry_accepted is False


def test_tracker_cannot_authorize_even_broad_points_without_bound_evidence():
    _frames, intr = make_synthetic_sequence(n_frames=1)
    detector = SyntheticDetector(intr, noise_px=0.0)
    tracker = ModelBasedTracker(
        detector,
        intr,
        ModelConfig(init_frames=1),
        geometry_gate=_production_geometry_gate(),
    )
    broad = np.array(
        [
            [-0.12, -0.15, 1.0],
            [0.12, -0.15, 1.0],
            [-0.12, 0.15, 1.0],
            [0.12, 0.15, 1.0],
        ],
        dtype=np.float64,
    )
    canonical = np.full((10, 3), np.nan, dtype=np.float64)
    canonical[:4] = broad
    tracker.canonical = canonical

    broad_fit = tracker._fit_pose(canonical.copy())

    assert broad_fit is not None
    assert broad_fit[-1].accepted is False
    assert broad_fit[-1].failure_reasons == (
        "production_bound_geometry_evidence_required",
    )

    collinear = np.array(
        [
            [-0.025, -0.0010, 1.0],
            [-0.015, 0.0007, 1.0],
            [-0.005, -0.0004, 1.0],
            [0.005, 0.0005, 1.0],
            [0.015, -0.0008, 1.0],
            [0.025, 0.0010, 1.0],
        ],
        dtype=np.float64,
    )
    canonical[:] = np.nan
    canonical[:6] = collinear
    tracker.canonical = canonical
    collinear_fit = tracker._fit_pose(canonical.copy())

    assert collinear_fit is not None
    assert collinear_fit[-1].accepted is False


def test_arm_occlusion_flags_unmeasured_never_silent():
    """SAFETY: occluded points must come back measured=False / COASTING (with higher position_std),
    NEVER silently reported as a confident TRACKING placement — and must recover afterwards."""
    frames, intr = make_synthetic_sequence(n_frames=120)
    occ = _ArmOccluder(SyntheticDetector(intr, noise_px=1.0))
    tr = ModelBasedTracker(occ, intr, ModelConfig(init_frames=8))
    DROP = {4, 5}                                  # R3, L3 — an arm covering one back level
    outs = []
    for i, fr in enumerate(frames):
        occ.set_truth(fr.gt_pixels)
        occ.drop = DROP if 50 <= i < 85 else set()
        outs.append(tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale))
    # invariant across the whole run: measured <=> TRACKING; unmeasured => COASTING/LOST (never TRACKING)
    for o in outs:
        if not o.initialized:
            continue
        for p in o.points:
            assert (p.status == "TRACKING") == p.measured
            if not p.measured:
                assert p.status in ("COASTING", "LOST")
    mid = outs[70]
    assert mid.initialized
    for k in DROP:                                 # the occluded points are flagged + still positioned (extrapolated)
        assert not mid.points[k].measured
        assert np.isfinite(mid.points[k].xyz_camera).all()
        assert mid.points[k].position_std > mid.points[(k + 2) % 10].position_std  # coasting std grew
    assert mid.n_measured <= 8                     # the two occluded points dropped out
    post = [outs[i] for i in range(105, 119) if outs[i].initialized]
    assert post and np.median([o.n_measured for o in post]) >= 9   # full recovery, no identity loss


class _SpikeDetector:
    """Wraps a detector and adds a TRANSIENT random-direction pixel spike to chosen points each frame
    — simulates per-frame detector glitches/outliers (IRLS must reject them every frame, and they must
    not be absorbed by the slow per-point residual)."""

    def __init__(self, base, spike_idx=(), spike_px=80.0, seed=0):
        self.base = base; self.spike_idx = set(spike_idx); self.spike_px = spike_px
        self.rng = np.random.RandomState(seed)

    def set_truth(self, g):
        self.base.set_truth(g)

    def detect(self, rgb, depth, intr):
        import dataclasses
        r = self.base.detect(rgb, depth, intr)
        if r is None or not self.spike_idx:
            return r
        xy = r.xy_det.astype(float).copy()
        for k in self.spike_idx:
            ang = self.rng.uniform(0, 2 * np.pi)             # random direction each frame
            xy[k] = xy[k] + self.spike_px * np.array([np.cos(ang), np.sin(ang)])
        return dataclasses.replace(r, xy_det=xy)


def test_outlier_spikes_rejected_by_irls():
    """Detector glitches on 2 points must NOT corrupt the others: IRLS rejects the outliers (they come
    back measured=False) while the remaining points stay accurate vs ground truth."""
    frames, intr = make_synthetic_sequence(n_frames=120)
    det = _SpikeDetector(SyntheticDetector(intr, noise_px=1.0), spike_idx={2, 3}, spike_px=90.0)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=8))
    out = None
    for fr in frames:
        det.set_truth(fr.gt_pixels)
        out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
    assert out.initialized
    assert not out.points[2].measured and not out.points[3].measured     # spiked → rejected
    clean = [k for k in range(10) if k not in (2, 3)]
    err = [np.linalg.norm(out.points[k].xyz_camera - frames[-1].gt_xyz_cam[k]) for k in clean]
    assert np.median(err) < 0.02 and all(out.points[k].measured for k in clean)  # others unharmed


def test_frame_dropout_coasts_and_recovers():
    """Intermittent missing frames (sensor dropout) → tracker coasts, never crashes, recovers."""
    frames, intr = make_synthetic_sequence(n_frames=120)
    det = SyntheticDetector(intr, noise_px=1.0)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=8))
    saw_coast = False; last = None
    for i, fr in enumerate(frames):
        det.set_truth(fr.gt_pixels)
        if 40 <= i < 50:                                       # 10-frame dropout
            last = tr.process(None, None, t=fr.t)
            if last.initialized and last.n_measured == 0:
                saw_coast = True
        else:
            last = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
    assert saw_coast and last.initialized and last.n_measured >= 9   # coasted during gap, recovered after


def test_extrinsics_produce_world_points():
    frames, intr = make_synthetic_sequence(n_frames=40)
    ext = np.eye(4); ext[:3, 3] = [0.1, -0.2, 0.3]      # arbitrary camera→base
    det = SyntheticDetector(intr, noise_px=1.0)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=8), extrinsics=ext)
    out = None
    for fr in frames:
        det.set_truth(fr.gt_pixels)
        out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
    assert out.initialized
    p = out.points[0]
    assert p.xyz_world is not None
    assert np.allclose(p.xyz_world, ext[:3, :3] @ p.xyz_camera + ext[:3, 3])


def test_fast_motion_stays_bounded():
    """3x-faster patient/camera motion (subsampled sequence) must not blow up or lose track."""
    frames, intr = make_synthetic_sequence(n_frames=240)
    frames = frames[::3]                                # larger inter-frame motion
    det = SyntheticDetector(intr, noise_px=1.0)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=8))
    errs = []
    for fr in frames:
        det.set_truth(fr.gt_pixels)
        o = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
        if o.initialized:
            for k, p in enumerate(o.points):
                if np.isfinite(p.xyz_camera).all():
                    errs.append(np.linalg.norm(p.xyz_camera - fr.gt_xyz_cam[k]))
    assert np.median(errs) < 0.03                       # bounded under fast motion (some EMA lag ok)


def test_long_duration_no_drift():
    """Over a long run the error must not grow (canonical-anchored, no accumulation)."""
    frames, intr = make_synthetic_sequence(n_frames=600)
    det = SyntheticDetector(intr, noise_px=1.0)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=8))
    err_t = []
    for fr in frames:
        det.set_truth(fr.gt_pixels)
        o = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
        e = [np.linalg.norm(o.points[k].xyz_camera - fr.gt_xyz_cam[k]) for k in range(10)
             if o.initialized and np.isfinite(o.points[k].xyz_camera).all()]
        err_t.append(np.median(e) if e else np.nan)
    first = np.nanmedian(err_t[20:120]); last = np.nanmedian(err_t[-100:])
    assert last < first * 1.5 + 0.002                   # no drift: late error ≈ early error


def test_coasts_when_detection_missing():
    frames, intr = make_synthetic_sequence(n_frames=60)
    det = SyntheticDetector(intr, noise_px=1.0)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=8))
    last = None
    for i, fr in enumerate(frames):
        det.set_truth(fr.gt_pixels)
        if i >= 30:                               # drop detections after frame 30
            det.set_truth(None) if hasattr(det, "set_truth") else None
            det._gt_full = None
        out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
        if out.initialized:
            last = out
    assert last is not None and last.initialized  # still emits (coasts) without detections
    assert all(p.status in ("COASTING", "LOST", "TRACKING") for p in last.points)


@pytest.mark.parametrize("bad_conf", [np.nan, np.inf])
def test_nan_confidence_rejected_fail_closed(bad_conf):
    """A detection whose confidence could not be computed (NaN/inf) must be REJECTED through the
    public process() path, not admitted as if it passed the threshold (nan < min_conf and
    inf < min_conf are both False in Python, so the gate must fail closed)."""
    import dataclasses
    frames, intr = make_synthetic_sequence(n_frames=20)
    base = SyntheticDetector(intr, noise_px=0.0)

    class _NanConfDetector:
        def detect(self, rgb, depth=None, intr=None):
            r = base.detect(rgb, depth, intr)
            return None if r is None else dataclasses.replace(r, confidence=bad_conf)

    tr = ModelBasedTracker(_NanConfDetector(), intr, ModelConfig(init_frames=8))
    out = None
    for fr in frames:
        base.set_truth(fr.gt_pixels)
        out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
    assert out is not None and not out.initialized   # every non-finite-conf detection rejected
    assert out.detector_confidence == 0.0 and not out.back_present


class _FixedFullFrameDetector:
    """Detector with exact, caller-supplied full-frame pixels (no model/noise dependency)."""

    def __init__(self, full_pixels):
        self.full_pixels = np.asarray(full_pixels, dtype=np.float64)

    def detect(self, rgb, depth, intr):
        height, width = rgb.shape[:2]
        crop = CropTransform.center_square(height, width, res=64, up_bias=0.0)
        return DetectorResult(
            xy_det=crop.full_to_det(self.full_pixels),
            crop=crop,
            confidence=1.0,
            back_present=True,
        )


def _raw_depth_surface_switch_case(*, dilation_px=0):
    """One target's 5x5 raw-depth footprint has 12 patient + 13 robot samples."""
    intr = Intrinsics(
        fx=80.0, fy=80.0, cx=31.5, cy=31.5,
        width=64, height=64, depth_scale=1.0,
    )
    pixels = np.array([
        [20.0, 20.0], [44.0, 20.0],
        [20.0, 26.0], [44.0, 26.0],
        [20.0, 32.0], [44.0, 32.0],
        [20.0, 38.0], [44.0, 38.0],
        [20.0, 44.0], [44.0, 44.0],
    ])
    depth = np.full((64, 64), 1000, dtype=np.uint16)
    footprint = [(y, x) for y in range(18, 23) for x in range(18, 23)]
    robot_samples = footprint[:13]
    for y, x in robot_samples:
        depth[y, x] = 500
    robot_mask = np.zeros_like(depth, dtype=bool)
    for y, x in robot_samples:
        robot_mask[y, x] = True
    tracker = ModelBasedTracker(
        _FixedFullFrameDetector(pixels), intr,
        ModelConfig(init_frames=1, robot_mask_dilation_px=dilation_px),
    )
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    return tracker, rgb, depth, robot_mask


def test_robot_mask_blocks_raw_depth_median_surface_switch_before_5x5_lift():
    """A 12/25 patient to 13/25 robot transition flips the raw median to the robot surface;
    the aligned robot mask must reject that association instead of lifting the wrong surface."""
    tracker, rgb, depth, robot_mask = _raw_depth_surface_switch_case()

    unmasked = tracker._detect3d(rgb, depth)
    masked = tracker._detect3d(rgb, depth, robot_mask=robot_mask)

    assert unmasked[0, 2] == pytest.approx(0.5)  # 13/25 robot samples switch the median
    assert np.isnan(masked[0]).all()             # rejected before raw-depth unprojection
    assert np.isfinite(masked[1:]).all()         # no fixed-count/global point rejection


def test_declared_robot_mask_dilation_is_applied_to_depth_association_footprint():
    tracker0, rgb, depth, _ = _raw_depth_surface_switch_case(dilation_px=0)
    tracker1, _, _, _ = _raw_depth_surface_switch_case(dilation_px=1)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[17, 20] = True  # one pixel outside the target's y=[18,22] raw-depth footprint

    no_dilation = tracker0._detect3d(rgb, depth, robot_mask=mask)
    with_dilation = tracker1._detect3d(rgb, depth, robot_mask=mask)

    assert np.isfinite(no_dilation[0]).all()
    assert np.isnan(with_dilation[0]).all()


def test_robot_mask_dilation_large_radius_uses_bounded_full_frame_semantics():
    tracker, _, _, _ = _raw_depth_surface_switch_case(dilation_px=639)
    mask = np.zeros((480, 640), dtype=bool)
    mask[0, 0] = True

    fully_covered = tracker._expanded_robot_mask(mask)
    assert fully_covered.shape == mask.shape
    assert fully_covered.all()

    tracker.cfg.robot_mask_dilation_px = 638
    almost = tracker._expanded_robot_mask(mask)
    assert almost[479, 638]
    assert not almost[0, 639]


def test_robot_mask_is_removed_from_surface_snap_cloud_outside_raw_5x5_footprint():
    """A nearby robot plane must not pull a surface snap through the KD-tree.

    The target's raw 5x5 footprint ends at x=66 while the robot begins at x=67,
    so the measurement itself remains eligible.  The explicit mask must still
    remove robot samples from the wider 3 cm surface-snap neighbourhood.
    """
    intr = Intrinsics(
        fx=800.0, fy=800.0, cx=63.5, cy=63.5,
        width=128, height=128, depth_scale=1.0,
    )
    pixels = np.asarray(
        [
            [64.0, 64.0], [20.0, 20.0], [30.0, 20.0], [40.0, 20.0],
            [50.0, 20.0], [20.0, 40.0], [30.0, 40.0], [40.0, 40.0],
            [50.0, 40.0], [20.0, 60.0],
        ],
        dtype=np.float64,
    )
    depth = np.full((128, 128), 1000, dtype=np.uint16)
    depth[:, 67:] = 990
    robot_mask = np.zeros(depth.shape, dtype=bool)
    robot_mask[:, 67:] = True
    tracker = ModelBasedTracker(
        _FixedFullFrameDetector(pixels), intr,
        ModelConfig(
            init_frames=1,
            surface_snap=True,
            icp_stride=1,
            snap_radius_m=0.03,
            snap_sanity_m=1.0,
            robot_mask_dilation_px=0,
        ),
    )
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)

    output = tracker.process(rgb, depth, robot_mask=robot_mask)

    assert output.initialized
    assert output.n_snapped == len(pixels)
    assert output.points[0].measured
    assert output.points[0].xyz_camera[2] == pytest.approx(1.0, abs=1e-6)


def test_single_robot_masked_point_follows_shared_constellation_pose_while_observable():
    frames, intr = make_synthetic_sequence(
        n_frames=60, motion=True, breathing=False, depth_noise_mm=0.0,
    )
    detector = SyntheticDetector(intr, noise_px=0.0)
    tracker = ModelBasedTracker(
        detector, intr,
        ModelConfig(init_frames=8, coast_max=30, robot_mask_dilation_px=0),
    )
    target = 0
    coast_camera = []
    coast_truth = []
    coast_std = []

    for index, frame in enumerate(frames):
        detector.set_truth(frame.gt_pixels)
        mask = None
        if 25 <= index < 40:
            mask = np.zeros(frame.depth.shape, dtype=bool)
            u, v = np.rint(frame.gt_pixels[target]).astype(int)
            mask[v, u] = True
        output = tracker.process(
            frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
            robot_mask=mask,
        )
        if mask is not None:
            point = output.points[target]
            assert output.initialized
            assert not point.measured
            assert point.status == "COASTING"
            coast_camera.append(point.xyz_camera.copy())
            coast_truth.append(frame.gt_xyz_cam[target].copy())
            coast_std.append(point.position_std)

    assert np.linalg.norm(coast_camera[-1] - coast_camera[0]) > 1e-4
    # This is a real comparison with the synthetic scene truth, not the algebraic
    # identity obtained by applying the tracker's own inverse pose to its output.
    errors = np.linalg.norm(np.asarray(coast_camera) - np.asarray(coast_truth), axis=1)
    assert float(np.percentile(errors, 95)) < 0.020
    assert coast_std[-1] > coast_std[0] > tracker.cfg.base_std_m


def test_insufficient_unmasked_points_hold_camera_pose_and_require_external_bounded_wrapper():
    """With fewer than four correspondences the raw Sim(3) fit is undefined.

    This deliberately documents the boundary used by the offline diagnostic: a
    patient-relative safety coast needs a separately validated patient-pose signal;
    it must not be inferred from this raw camera-frame hold.
    """
    frames, intr = make_synthetic_sequence(
        n_frames=35, motion=True, breathing=False, depth_noise_mm=0.0,
    )
    detector = SyntheticDetector(intr, noise_px=0.0)
    tracker = ModelBasedTracker(
        detector, intr, ModelConfig(init_frames=8, coast_max=30),
    )
    previous = None
    masked_outputs = []
    masked_truth = []
    for index, frame in enumerate(frames):
        detector.set_truth(frame.gt_pixels)
        mask = None
        if index >= 20:
            mask = np.zeros(frame.depth.shape, dtype=bool)
            for target in range(7):
                u, v = np.rint(frame.gt_pixels[target]).astype(int)
                mask[v, u] = True
        output = tracker.process(
            frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
            robot_mask=mask,
        )
        if index == 19:
            previous = np.asarray([point.xyz_camera for point in output.points])
        if mask is not None:
            assert output.initialized
            assert output.n_measured == 0
            masked_outputs.append(
                np.asarray([point.xyz_camera for point in output.points])
            )
            masked_truth.append(frame.gt_xyz_cam.copy())

    assert previous is not None
    np.testing.assert_allclose(
        masked_outputs,
        np.repeat(previous[None], len(masked_outputs), axis=0),
        atol=0.0,
        rtol=0.0,
    )
    assert np.linalg.norm(masked_truth[-1][0] - masked_truth[0][0]) > 1e-4


def test_point_with_only_one_bad_init_sample_is_lost_then_stably_bootstraps():
    """One late 300 mm depth outlier must not become an anatomical canonical point."""
    frames, intr = make_synthetic_sequence(
        n_frames=20, motion=False, breathing=False, depth_noise_mm=0.0,
    )
    detector = SyntheticDetector(intr, noise_px=0.0)
    tracker = ModelBasedTracker(
        detector, intr, ModelConfig(init_frames=3, coast_max=10),
    )
    target = 0
    outputs = []
    for index, frame in enumerate(frames):
        detector.set_truth(frame.gt_pixels)
        mask = None
        depth = frame.depth
        if index < 2 or index >= 8:
            mask = np.zeros(frame.depth.shape, dtype=bool)
            u, v = np.rint(frame.gt_pixels[target]).astype(int)
            mask[v, u] = True
        elif index == 2:
            depth = frame.depth.copy()
            u, v = np.rint(frame.gt_pixels[target]).astype(int)
            depth[v - 2 : v + 3, u - 2 : u + 3] = 900
        outputs.append(
            tracker.process(
                frame.rgb, depth, t=frame.t,
                depth_scale=frame.depth_scale, robot_mask=mask,
            )
        )

    initialized = outputs[2]
    assert initialized.initialized
    assert initialized.points[target].status == "LOST"
    assert initialized.points[target].measured is False
    assert not np.isfinite(initialized.points[target].xyz_camera).any()

    # Late canonical seeding must itself use a full stable streak, rather than
    # trusting the first newly visible depth sample.
    for output in outputs[3:5]:
        assert output.points[target].status == "LOST"
        assert output.points[target].measured is False
    recovered = outputs[5]
    assert recovered.points[target].status == "TRACKING"
    assert recovered.points[target].measured is True
    assert np.isfinite(recovered.points[target].xyz_camera).all()
    assert np.linalg.norm(
        recovered.points[target].xyz_camera - frames[5].gt_xyz_cam[target]
    ) < 0.020
    assert outputs[8].points[target].status == "COASTING"
    assert outputs[8].points[target].measured is False
    assert np.isfinite(outputs[8].points[target].xyz_camera).all()
    assert outputs[18].points[target].status == "LOST"
    assert outputs[18].points[target].measured is False


def test_late_bootstrap_streak_resets_when_shared_pose_is_unobservable():
    """A frame below the four-point mathematical floor breaks continuity."""
    frames, intr = make_synthetic_sequence(
        n_frames=12, motion=False, breathing=False, depth_noise_mm=0.0,
    )
    detector = SyntheticDetector(intr, noise_px=0.0)
    tracker = ModelBasedTracker(
        detector, intr, ModelConfig(init_frames=3, coast_max=10),
    )
    target = 0
    outputs = []
    for index, frame in enumerate(frames):
        detector.set_truth(frame.gt_pixels)
        mask = None
        masked_targets = set()
        if index < 3:
            masked_targets = {target}
        elif index == 5:
            masked_targets = set(range(7))  # only three canonical points remain
        if masked_targets:
            mask = np.zeros(frame.depth.shape, dtype=bool)
            for masked_target in masked_targets:
                u, v = np.rint(frame.gt_pixels[masked_target]).astype(int)
                mask[v, u] = True
        outputs.append(
            tracker.process(
                frame.rgb, frame.depth, t=frame.t,
                depth_scale=frame.depth_scale, robot_mask=mask,
            )
        )

    assert outputs[2].initialized
    assert outputs[2].points[target].status == "LOST"
    assert outputs[3].points[target].status == "LOST"  # streak 1/3
    assert outputs[4].points[target].status == "LOST"  # streak 2/3
    assert outputs[5].n_measured == 0                   # pose fit unavailable; reset streak
    assert outputs[6].points[target].status == "LOST"  # fresh streak 1/3, not recovered
    assert outputs[7].points[target].status == "LOST"  # fresh streak 2/3
    assert outputs[8].points[target].status == "TRACKING"
    assert outputs[8].points[target].measured is True


def test_unstable_point_present_during_global_init_is_deferred_to_late_bootstrap():
    frames, intr = make_synthetic_sequence(
        n_frames=8, motion=False, breathing=False, depth_noise_mm=0.0,
    )
    detector = SyntheticDetector(intr, noise_px=0.0)
    tracker = ModelBasedTracker(
        detector, intr, ModelConfig(init_frames=3, coast_max=10),
    )
    target = 0
    outputs = []
    for index, frame in enumerate(frames):
        detector.set_truth(frame.gt_pixels)
        depth = frame.depth
        if index == 2:
            depth = frame.depth.copy()
            u, v = np.rint(frame.gt_pixels[target]).astype(int)
            depth[v - 2 : v + 3, u - 2 : u + 3] = 900
        outputs.append(
            tracker.process(
                frame.rgb, depth, t=frame.t, depth_scale=frame.depth_scale,
            )
        )

    assert not outputs[2].initialized
    assert not outputs[3].initialized
    assert not outputs[4].initialized
    assert outputs[5].initialized
    assert outputs[5].points[target].status == "TRACKING"
    assert outputs[5].points[target].measured is True


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.zeros((64, 64), dtype=np.uint8),
        np.zeros((63, 64), dtype=bool),
        [[False] * 64 for _ in range(64)],
    ],
)
def test_robot_mask_type_and_shape_contract_fail_closed(bad_mask):
    tracker, rgb, depth, _ = _raw_depth_surface_switch_case()

    with pytest.raises(ValueError, match="robot_mask"):
        tracker.process(rgb, depth, robot_mask=bad_mask)


@pytest.mark.parametrize("bad_dilation", [-1, 1.5, True])
def test_robot_mask_dilation_contract_fails_closed(bad_dilation):
    tracker, rgb, depth, robot_mask = _raw_depth_surface_switch_case()
    tracker.cfg.robot_mask_dilation_px = bad_dilation

    with pytest.raises(ValueError, match="robot_mask_dilation_px"):
        tracker.process(rgb, depth, robot_mask=robot_mask)


def test_default_process_path_matches_explicit_no_robot_mask():
    frames, intr = make_synthetic_sequence(n_frames=35, depth_noise_mm=0.0)
    detector_default = SyntheticDetector(intr, noise_px=0.0)
    detector_explicit = SyntheticDetector(intr, noise_px=0.0)
    tracker_default = ModelBasedTracker(detector_default, intr, ModelConfig(init_frames=8))
    tracker_explicit = ModelBasedTracker(detector_explicit, intr, ModelConfig(init_frames=8))

    for frame in frames:
        detector_default.set_truth(frame.gt_pixels)
        detector_explicit.set_truth(frame.gt_pixels)
        default = tracker_default.process(
            frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
        )
        explicit = tracker_explicit.process(
            frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
            robot_mask=None,
        )
        assert default.initialized == explicit.initialized
        assert default.n_measured == explicit.n_measured
        np.testing.assert_array_equal(
            np.asarray([point.xyz_camera for point in default.points]),
            np.asarray([point.xyz_camera for point in explicit.points]),
        )


def _truth_mask(frame, point_indices):
    mask = np.zeros(frame.depth.shape, dtype=bool)
    for point_index in point_indices:
        u, v = np.rint(frame.gt_pixels[point_index]).astype(int)
        mask[v, u] = True
    return mask


def _output_fingerprint(output):
    return (
        output.initialized,
        output.pose_fit_mm,
        output.n_tracking,
        output.n_snapped,
        output.n_measured,
        output.back_present,
        output.detector_confidence,
        output.reason,
        tuple(
            (
                point.status,
                point.measured,
                point.position_std,
                point.xyz_camera.tobytes(),
                point.pixel.tobytes(),
            )
            for point in output.points
        ),
    )


def test_dormant_reacquisition_is_default_off_and_explicit_false_is_bit_exact():
    assert ModelConfig().dormant_reacquisition is False
    frames, intr = make_synthetic_sequence(
        n_frames=12, motion=False, breathing=False, depth_noise_mm=0.0,
    )
    default_detector = SyntheticDetector(intr, noise_px=0.0)
    false_detector = SyntheticDetector(intr, noise_px=0.0)
    default_tracker = ModelBasedTracker(
        default_detector, intr, ModelConfig(init_frames=3, coast_max=2),
    )
    false_tracker = ModelBasedTracker(
        false_detector, intr,
        ModelConfig(init_frames=3, coast_max=2, dormant_reacquisition=False),
    )

    for index, frame in enumerate(frames):
        mask = _truth_mask(frame, range(7)) if index >= 4 else None
        default_detector.set_truth(frame.gt_pixels)
        false_detector.set_truth(frame.gt_pixels)
        default = default_tracker.process(
            frame.rgb, frame.depth, t=frame.t,
            depth_scale=frame.depth_scale, robot_mask=mask,
        )
        explicit_false = false_tracker.process(
            frame.rgb, frame.depth, t=frame.t,
            depth_scale=frame.depth_scale, robot_mask=mask,
        )
        assert _output_fingerprint(default) == _output_fingerprint(explicit_false)


def _enter_dormant(*, motion=False):
    frames, intr = make_synthetic_sequence(
        n_frames=30, motion=motion, breathing=False, depth_noise_mm=0.0,
    )
    detector = SyntheticDetector(intr, noise_px=0.0)
    tracker = ModelBasedTracker(
        detector, intr,
        ModelConfig(init_frames=3, coast_max=2, dormant_reacquisition=True),
    )
    outputs = []
    for index in range(7):
        frame = frames[index]
        detector.set_truth(frame.gt_pixels)
        mask = _truth_mask(frame, range(7)) if index >= 4 else None
        outputs.append(
            tracker.process(
                frame.rgb, frame.depth, t=frame.t,
                depth_scale=frame.depth_scale, robot_mask=mask,
            )
        )
    return frames, detector, tracker, outputs


def _assert_fail_closed_dormant(output, state):
    assert not output.initialized
    assert output.tracking_state == state
    assert output.pose_valid is False
    assert output.canonical_retained is True
    assert output.n_measured == 0
    assert all(point.status == "LOST" for point in output.points)
    assert all(point.measured is False for point in output.points)
    assert all(not np.isfinite(point.xyz_camera).any() for point in output.points)
    assert all(not np.isfinite(point.pixel).any() for point in output.points)


def test_long_seven_point_occlusion_retains_body_canonical_without_camera_leakage():
    frames, detector, tracker, outputs = _enter_dormant()
    canonical = tracker.canonical.copy()
    residual = tracker.residual.copy()

    _assert_fail_closed_dormant(outputs[6], "DORMANT")
    np.testing.assert_array_equal(tracker.canonical, canonical)
    np.testing.assert_array_equal(tracker.residual, residual)

    frame = frames[7]
    detector.set_truth(frame.gt_pixels)
    still_hidden = tracker.process(
        frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
        robot_mask=_truth_mask(frame, range(7)),
    )
    _assert_fail_closed_dormant(still_hidden, "DORMANT")
    np.testing.assert_array_equal(tracker.canonical, canonical)
    np.testing.assert_array_equal(tracker.residual, residual)


def test_reacquisition_stays_lost_for_four_frames_and_recovers_on_fifth():
    frames, detector, tracker, _ = _enter_dormant()
    canonical = tracker.canonical.copy()
    residual = tracker.residual.copy()
    provisional = []
    for index in range(7, 12):
        frame = frames[index]
        detector.set_truth(frame.gt_pixels)
        provisional.append(
            tracker.process(
                frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
            )
        )

    for output in provisional[:4]:
        _assert_fail_closed_dormant(output, "REACQUIRING")
    recovered = provisional[4]
    assert recovered.initialized
    assert recovered.tracking_state == "ACTIVE"
    assert recovered.pose_valid is True
    assert recovered.canonical_retained is True
    assert recovered.n_measured >= 6
    np.testing.assert_array_equal(tracker.canonical, canonical)
    np.testing.assert_array_equal(tracker.residual, residual)


def test_reacquisition_streak_is_cleared_by_an_unobservable_frame():
    frames, detector, tracker, _ = _enter_dormant()
    for index in range(7, 10):
        frame = frames[index]
        detector.set_truth(frame.gt_pixels)
        output = tracker.process(
            frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
        )
        _assert_fail_closed_dormant(output, "REACQUIRING")

    interrupted_frame = frames[10]
    detector.set_truth(interrupted_frame.gt_pixels)
    interrupted = tracker.process(
        interrupted_frame.rgb, interrupted_frame.depth, t=interrupted_frame.t,
        depth_scale=interrupted_frame.depth_scale,
        robot_mask=_truth_mask(interrupted_frame, range(7)),
    )
    _assert_fail_closed_dormant(interrupted, "DORMANT")

    for index in range(11, 15):
        frame = frames[index]
        detector.set_truth(frame.gt_pixels)
        output = tracker.process(
            frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
        )
        _assert_fail_closed_dormant(output, "REACQUIRING")
    frame = frames[15]
    detector.set_truth(frame.gt_pixels)
    recovered = tracker.process(
        frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
    )
    assert recovered.initialized and recovered.tracking_state == "ACTIVE"


class _PresenceToggleDetector:
    def __init__(self, base):
        self.base = base
        self.back_present = True

    def set_truth(self, truth):
        self.base.set_truth(truth)

    def detect(self, rgb, depth, intr):
        import dataclasses
        result = self.base.detect(rgb, depth, intr)
        if result is None:
            return None
        return dataclasses.replace(result, back_present=self.back_present)


def test_no_back_destroys_dormant_canonical():
    frames, intr = make_synthetic_sequence(
        n_frames=12, motion=False, breathing=False, depth_noise_mm=0.0,
    )
    detector = _PresenceToggleDetector(SyntheticDetector(intr, noise_px=0.0))
    tracker = ModelBasedTracker(
        detector, intr,
        ModelConfig(init_frames=3, coast_max=2, dormant_reacquisition=True),
    )
    output = None
    for index in range(7):
        frame = frames[index]
        detector.set_truth(frame.gt_pixels)
        mask = _truth_mask(frame, range(7)) if index >= 4 else None
        output = tracker.process(
            frame.rgb, frame.depth, t=frame.t,
            depth_scale=frame.depth_scale, robot_mask=mask,
        )
    _assert_fail_closed_dormant(output, "DORMANT")

    detector.back_present = False
    # Debounced (2026-08-07 audit): one flickered no-back frame must NOT wipe the
    # canonical; the reset fires only after no_back_reset_frames consecutive frames.
    no_back = None
    for index in range(7, 7 + tracker.cfg.no_back_reset_frames):
        frame = frames[index]
        detector.set_truth(frame.gt_pixels)
        no_back = tracker.process(
            frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
        )
    assert not no_back.initialized
    assert no_back.tracking_state == "COLD_INIT"
    assert no_back.pose_valid is False
    assert no_back.canonical_retained is False
    assert tracker.canonical is None


def test_patient_or_camera_motion_during_dormancy_never_leaks_old_camera_coordinates():
    frames, detector, tracker, outputs = _enter_dormant(motion=True)
    _assert_fail_closed_dormant(outputs[6], "DORMANT")
    truth_at_entry = frames[6].gt_xyz_cam.copy()
    for index in range(7, 12):
        frame = frames[index]
        detector.set_truth(frame.gt_pixels)
        output = tracker.process(
            frame.rgb, frame.depth, t=frame.t, depth_scale=frame.depth_scale,
            robot_mask=_truth_mask(frame, range(7)),
        )
        _assert_fail_closed_dormant(output, "DORMANT")
    assert np.linalg.norm(frames[11].gt_xyz_cam - truth_at_entry) > 1e-4


def _depth_shift_at_points(frame, point_indices, shift_mm):
    depth = frame.depth.copy()
    for point_index in point_indices:
        u, v = np.rint(frame.gt_pixels[point_index]).astype(int)
        patch = depth[v - 2 : v + 3, u - 2 : u + 3]
        patch[...] = np.maximum(
            1, patch.astype(np.int64) + int(shift_mm),
        ).astype(depth.dtype)
    return depth


def test_stable_111mm_wrong_point_is_rejected_without_blocking_clean_pose_recovery():
    frames, detector, tracker, _ = _enter_dormant()
    canonical = tracker.canonical.copy()
    residual = tracker.residual.copy()
    outputs = []
    for index in range(7, 12):
        frame = frames[index]
        detector.set_truth(frame.gt_pixels)
        outputs.append(
            tracker.process(
                frame.rgb, _depth_shift_at_points(frame, [0], -111),
                t=frame.t, depth_scale=frame.depth_scale,
            )
        )

    assert outputs[-1].initialized and outputs[-1].tracking_state == "ACTIVE"
    assert outputs[-1].points[0].measured is False
    assert outputs[-1].points[0].status != "TRACKING"
    assert outputs[-1].n_measured >= 6
    np.testing.assert_array_equal(tracker.canonical, canonical)
    np.testing.assert_array_equal(tracker.residual, residual)


def test_coherent_stable_wrong_surface_cannot_reacquire():
    frames, detector, tracker, _ = _enter_dormant()
    canonical = tracker.canonical.copy()
    residual = tracker.residual.copy()
    for index in range(7, 14):
        frame = frames[index]
        detector.set_truth(frame.gt_pixels)
        output = tracker.process(
            frame.rgb, _depth_shift_at_points(frame, range(10), -111),
            t=frame.t, depth_scale=frame.depth_scale,
        )
        _assert_fail_closed_dormant(output, "DORMANT")

    np.testing.assert_array_equal(tracker.canonical, canonical)
    np.testing.assert_array_equal(tracker.residual, residual)


def _controlled_surface_frame(intr, xyz, visible=range(10)):
    xyz = np.asarray(xyz, dtype=np.float64)
    pixels = intr.project(xyz)
    declared_pixels = np.full((10, 2), np.nan, dtype=np.float64)
    depth = np.zeros((intr.height, intr.width), dtype=np.uint16)
    for point_index in visible:
        declared_pixels[point_index] = pixels[point_index]
        u, v = np.rint(pixels[point_index]).astype(int)
        assert 2 <= u < intr.width - 2 and 2 <= v < intr.height - 2
        depth[v - 2 : v + 3, u - 2 : u + 3] = int(
            round(xyz[point_index, 2] * 1000.0 / intr.depth_scale)
        )
    rgb = np.zeros((intr.height, intr.width, 3), dtype=np.uint8)
    return rgb, depth, declared_pixels


def _controlled_tracker_in_dormant():
    intr = Intrinsics(
        fx=80.0, fy=80.0, cx=31.5, cy=31.5,
        width=64, height=64, depth_scale=1.0,
    )
    base_pixels = np.array([
        [20.0, 20.0], [44.0, 20.0],
        [20.0, 26.0], [44.0, 26.0],
        [20.0, 32.0], [44.0, 32.0],
        [20.0, 38.0], [44.0, 38.0],
        [20.0, 44.0], [44.0, 44.0],
    ])
    base_xyz = intr.unproject(base_pixels, np.full(10, 0.6))
    rgb, depth, pixels = _controlled_surface_frame(intr, base_xyz)
    detector = _FixedFullFrameDetector(pixels)
    tracker = ModelBasedTracker(
        detector, intr,
        ModelConfig(init_frames=3, coast_max=2, dormant_reacquisition=True),
    )
    timestamp = 0.0
    for _ in range(3):
        detector.full_pixels = pixels
        output = tracker.process(rgb, depth, t=timestamp, depth_scale=intr.depth_scale)
        timestamp += 0.05
    assert output.initialized and output.tracking_state == "ACTIVE"

    for _ in range(3):
        mask = _truth_mask(
            type("Frame", (), {"depth": depth, "gt_pixels": pixels})(), range(7),
        )
        output = tracker.process(
            rgb, depth, t=timestamp, depth_scale=intr.depth_scale, robot_mask=mask,
        )
        timestamp += 0.05
    _assert_fail_closed_dormant(output, "DORMANT")
    return intr, detector, tracker, base_xyz, timestamp


def _drive_controlled_reacquisition(
    intr, detector, tracker, xyz, timestamp, *, visible=range(10), frame_count=5,
    timestamps=None,
):
    outputs = []
    sequence = (
        list(timestamps)
        if timestamps is not None
        else [timestamp + 0.05 * index for index in range(frame_count)]
    )
    assert len(sequence) == frame_count
    for frame_timestamp in sequence:
        rgb, depth, pixels = _controlled_surface_frame(intr, xyz, visible=visible)
        detector.full_pixels = pixels
        outputs.append(
            tracker.process(
                rgb, depth, t=frame_timestamp, depth_scale=intr.depth_scale,
            )
        )
    return outputs


@pytest.mark.parametrize(
    "visible",
    [range(10), (0, 1, 2, 3, 4, 5)],
    ids=["ten_points", "bilateral_three_level_six_points"],
)
def test_public_process_rejects_stable_isomorphic_surface_900mm_away(visible):
    intr, detector, tracker, base_xyz, timestamp = _controlled_tracker_in_dormant()
    fake_surface = base_xyz + np.array([0.0, 0.0, 0.9])

    outputs = _drive_controlled_reacquisition(
        intr, detector, tracker, fake_surface, timestamp, visible=visible,
    )

    for output in outputs:
        _assert_fail_closed_dormant(output, "DORMANT")


def test_public_process_rejects_first_proposal_with_1_6m_and_90deg_jump():
    intr, detector, tracker, base_xyz, timestamp = _controlled_tracker_in_dormant()
    rotation = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    fake_surface = (rotation @ base_xyz.T).T + np.array([0.0, 0.0, 1.6])

    outputs = _drive_controlled_reacquisition(
        intr, detector, tracker, fake_surface, timestamp,
    )

    for output in outputs:
        _assert_fail_closed_dormant(output, "DORMANT")


def test_public_process_rejects_8deg_rotation_with_84mm_template_point_jump():
    intr, detector, tracker, base_xyz, timestamp = _controlled_tracker_in_dormant()
    angle = np.deg2rad(8.0)
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    fake_surface = (rotation @ base_xyz.T).T
    point_displacement = np.linalg.norm(fake_surface - base_xyz, axis=1)
    assert float(np.max(point_displacement)) > 0.08
    assert 8.0 < tracker.cfg.reacquire_absolute_rotation_deg

    outputs = _drive_controlled_reacquisition(
        intr, detector, tracker, fake_surface, timestamp,
    )

    for output in outputs:
        _assert_fail_closed_dormant(output, "DORMANT")


def test_public_process_rejects_stable_isomorphic_111mm_translation():
    intr, detector, tracker, base_xyz, timestamp = _controlled_tracker_in_dormant()
    fake_surface = base_xyz + np.array([0.0, 0.0, 0.111])

    outputs = _drive_controlled_reacquisition(
        intr, detector, tracker, fake_surface, timestamp,
    )

    for output in outputs:
        _assert_fail_closed_dormant(output, "DORMANT")


def test_public_process_allows_gated_6mm_translation_after_dormancy():
    intr, detector, tracker, base_xyz, timestamp = _controlled_tracker_in_dormant()
    moved_patient = base_xyz + np.array([0.006, 0.0, 0.0])

    outputs = _drive_controlled_reacquisition(
        intr, detector, tracker, moved_patient, timestamp,
    )

    for output in outputs[:4]:
        _assert_fail_closed_dormant(output, "REACQUIRING")
    assert outputs[4].initialized
    assert outputs[4].tracking_state == "ACTIVE"
    assert outputs[4].pose_valid is True


def test_reacquire_frames_cannot_be_configured_below_safe_minimum():
    for unsafe_count in (0, 1, 4):
        with pytest.raises(ValueError, match="reacquire_frames"):
            ModelConfig(reacquire_frames=unsafe_count)


@pytest.mark.parametrize("bad_gap", [0.0, -0.05, 0.25], ids=["duplicate", "backward", "expired"])
def test_public_process_timestamp_break_clears_reacquisition_streak(bad_gap):
    intr, detector, tracker, base_xyz, timestamp = _controlled_tracker_in_dormant()
    first = timestamp
    second = first + 0.05
    bad = second + bad_gap
    initial = _drive_controlled_reacquisition(
        intr, detector, tracker, base_xyz, timestamp,
        frame_count=3, timestamps=[first, second, bad],
    )
    _assert_fail_closed_dormant(initial[0], "REACQUIRING")
    _assert_fail_closed_dormant(initial[1], "REACQUIRING")
    _assert_fail_closed_dormant(initial[2], "DORMANT")

    restart = max(first, second, bad) + 0.05
    recovered = _drive_controlled_reacquisition(
        intr, detector, tracker, base_xyz, restart,
        frame_count=5,
    )
    for output in recovered[:4]:
        _assert_fail_closed_dormant(output, "REACQUIRING")
    assert recovered[4].initialized and recovered[4].tracking_state == "ACTIVE"


def test_public_process_same_timestamp_replay_never_reacquires():
    intr, detector, tracker, base_xyz, timestamp = _controlled_tracker_in_dormant()
    outputs = _drive_controlled_reacquisition(
        intr, detector, tracker, base_xyz, timestamp,
        frame_count=8, timestamps=[timestamp] * 8,
    )

    assert not any(output.initialized for output in outputs)
    assert all(output.pose_valid is False for output in outputs)
    assert all(output.tracking_state in ("DORMANT", "REACQUIRING") for output in outputs)


def test_reacquisition_gap_config_must_be_positive_and_finite():
    for invalid in (0.0, -0.1, np.inf, np.nan, True):
        with pytest.raises(ValueError, match="reacquire_max_gap_s"):
            ModelConfig(reacquire_max_gap_s=invalid)


@pytest.mark.parametrize(
    "field",
    [
        "reacquire_absolute_translation_m",
        "reacquire_absolute_rotation_deg",
        "reacquire_max_point_displacement_m",
    ],
)
def test_reacquisition_absolute_gate_configs_require_real_positive_finite_values(field):
    for invalid in (0.0, -0.1, np.inf, np.nan, True):
        with pytest.raises(ValueError, match="reacquisition limits"):
            ModelConfig(**{field: invalid})


def test_reacquisition_gates_the_exact_retained_output_template():
    intr = Intrinsics(
        fx=80.0, fy=80.0, cx=31.5, cy=31.5,
        width=64, height=64, depth_scale=1.0,
    )
    base_pixels = np.array([
        [20.0, 20.0], [44.0, 20.0],
        [20.0, 26.0], [44.0, 26.0],
        [20.0, 32.0], [44.0, 32.0],
        [20.0, 38.0], [44.0, 38.0],
        [20.0, 44.0], [44.0, 44.0],
    ])
    base_xyz = intr.unproject(base_pixels, np.full(10, 0.6))
    rgb, depth, pixels = _controlled_surface_frame(intr, base_xyz)
    detector = _FixedFullFrameDetector(pixels)
    tracker = ModelBasedTracker(
        detector, intr,
        ModelConfig(
            init_frames=3, coast_max=2, resid_ema=1.0,
            dormant_reacquisition=True,
        ),
    )
    timestamp = 0.0
    for _ in range(3):
        detector.full_pixels = pixels
        output = tracker.process(rgb, depth, t=timestamp, depth_scale=intr.depth_scale)
        timestamp += 0.05
    assert output.initialized

    displaced = base_xyz.copy()
    displaced[0, 0] += 0.01739
    shifted_rgb, shifted_depth, shifted_pixels = _controlled_surface_frame(intr, displaced)
    detector.full_pixels = shifted_pixels
    shifted = tracker.process(
        shifted_rgb, shifted_depth, t=timestamp, depth_scale=intr.depth_scale,
    )
    timestamp += 0.05
    assert shifted.points[0].measured
    assert np.linalg.norm(tracker.residual[0]) > 0.015
    retained_residual = tracker.residual.copy()

    detector.full_pixels = pixels
    for _ in range(3):
        output = tracker.process(
            rgb, depth, t=timestamp, depth_scale=intr.depth_scale,
            robot_mask=_truth_mask(
                type("Frame", (), {"depth": depth, "gt_pixels": pixels})(), range(7),
            ),
        )
        timestamp += 0.05
    _assert_fail_closed_dormant(output, "DORMANT")

    outputs = _drive_controlled_reacquisition(
        intr, detector, tracker, base_xyz, timestamp,
    )
    assert outputs[-1].initialized
    measured_indices = [
        index for index, point in enumerate(outputs[-1].points) if point.measured
    ]
    assert len(measured_indices) >= 6
    measured_error = [
        np.linalg.norm(outputs[-1].points[index].xyz_camera - base_xyz[index])
        for index in measured_indices
    ]
    assert max(measured_error) <= tracker.cfg.reacquire_max_resid_m
    np.testing.assert_array_equal(tracker.residual, retained_residual)


def test_individually_lost_point_readmits_via_canonical_gate():
    # 2026-08-07 audit (critical): a single point that exceeded coast_max stayed
    # LOST for many seconds (R3 absent 97% of a static session) even when its
    # observation returned. The re-admission path re-admits it after a streak of
    # observations consistent with the pose-predicted anatomical position.
    frames, intr = make_synthetic_sequence(
        n_frames=80, motion=False, breathing=False, depth_noise_mm=0.0,
    )
    det = SyntheticDetector(intr, noise_px=0.0)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=4, coast_max=3))
    victim = 4
    for i, fr in enumerate(frames):
        truth = fr.gt_pixels.copy()
        if 10 <= i < 30:
            truth[victim] = np.nan                  # only this point disappears
        det.set_truth(truth)
        out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
        if i == 29:
            assert tr._miss_pp[victim] > tr.cfg.coast_max  # individually LOST
    # after the observation returns, the point must be re-admitted and measured
    assert tr._miss_pp[victim] == 0
    assert out.points[victim].status == "TRACKING"
    # and it must have landed back on the true anatomy (within a few mm)
    gt3d = frames[-1].gt_xyz_cam[victim]
    err = float(np.linalg.norm(np.asarray(out.points[victim].xyz_camera) - gt3d))
    assert err < 0.008, err


def test_pose_snap_after_gap_prevents_confident_crawl_back():
    # 2026-08-07 audit: after a gap the pose EMA crawled from the stale pose to
    # the new fit over several frames, publishing confidently-wrong points. With
    # the snap, the FIRST post-gap fit is adopted outright when the median
    # displacement exceeds pose_snap_m.
    frames, intr = make_synthetic_sequence(n_frames=60, motion=True)
    det = SyntheticDetector(intr, noise_px=0.5)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=6, coast_max=30))
    out = None
    for i, fr in enumerate(frames):
        det.set_truth(fr.gt_pixels)
        if 25 <= i < 40:
            out = tr.process(None, None, t=fr.t)     # 15-frame dropout while moving
        else:
            out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
        if i == 40:
            # FIRST frame after the gap: published points already at the new truth
            errs = [
                float(np.linalg.norm(
                    np.asarray(pt.xyz_camera) - fr.gt_xyz_cam[j]
                ))
                for j, pt in enumerate(out.points)
                if pt.status == "TRACKING" and np.isfinite(pt.xyz_camera).all()
            ]
            assert errs and float(np.median(errs)) < 0.010, errs


def test_oscillation_recovery_returns_to_same_anatomy():
    # The operator's core question: after a large oscillation + detector dropout,
    # do the 10 points reconverge to the SAME anatomical locations?
    frames, intr = make_synthetic_sequence(n_frames=100, motion=True, breathing=True)
    det = SyntheticDetector(intr, noise_px=1.0)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=6, coast_max=10))
    out = None
    for i, fr in enumerate(frames):
        det.set_truth(fr.gt_pixels)
        if 45 <= i < 55:
            out = tr.process(None, None, t=fr.t)     # oscillation blur: no detections
        else:
            out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
    assert out.initialized
    errs = [
        float(np.linalg.norm(np.asarray(pt.xyz_camera) - frames[-1].gt_xyz_cam[j]))
        for j, pt in enumerate(out.points)
        if np.isfinite(pt.xyz_camera).all()
    ]
    assert len(errs) >= 9
    assert float(np.median(errs)) < 0.006, errs      # reconverged to anatomy
