"""Tests for the model-based deformable tracker (torch-free: synthetic detector + scene)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ausc.perception.tracking.replay import make_synthetic_sequence, SyntheticDetector  # noqa: E402
from ausc.perception.tracking.model_tracker import ModelBasedTracker, ModelConfig  # noqa: E402


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
