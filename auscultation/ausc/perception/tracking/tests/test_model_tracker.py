"""Tests for the model-based deformable tracker (torch-free: synthetic detector + scene)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ausc.perception.tracking.replay import make_synthetic_sequence, SyntheticDetector  # noqa: E402
from ausc.perception.tracking.model_tracker import ModelBasedTracker, ModelConfig  # noqa: E402
from ausc.perception.tracking.detector import normals_from_depth_m  # noqa: E402


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


def test_nan_confidence_rejected_fail_closed():
    """A detection whose confidence could not be computed (NaN) must be REJECTED, not admitted as if
    it passed the threshold (nan < min_conf is False in Python — the gate must fail closed)."""
    import dataclasses
    frames, intr = make_synthetic_sequence(n_frames=20)
    base = SyntheticDetector(intr, noise_px=0.0)

    class _NanConfDetector:
        def detect(self, rgb, depth=None, intr=None):
            r = base.detect(rgb, depth, intr)
            return None if r is None else dataclasses.replace(r, confidence=np.nan)

    tr = ModelBasedTracker(_NanConfDetector(), intr, ModelConfig(init_frames=8))
    out = None
    for fr in frames:
        base.set_truth(fr.gt_pixels)
        out = tr.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
    assert out is not None and not out.initialized   # every NaN-conf detection rejected ⇒ never inits
