"""Fail-closed regression tests for the live 10-point detector/tracker path."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
)))

from ausc.perception.tracking import detector as detector_mod  # noqa: E402
from ausc.perception.tracking.detector import (  # noqa: E402
    DenseIUVDetector,
    LUNG_INDEX,
    LUNG_ORDER,
    NormalsDetector,
    _fill_depth_holes,
    normals_from_depth_m,
)
from ausc.perception.tracking.geometry import Intrinsics  # noqa: E402
from ausc.perception.tracking.model_tracker import (  # noqa: E402
    ModelBasedTracker,
    ModelConfig,
)
from ausc.perception.tracking.replay import (  # noqa: E402
    SyntheticDetector,
    make_synthetic_sequence,
)


def _detector_shell(cls, *, res=64):
    detector = object.__new__(cls)
    detector.res = res
    detector.review_scale = 24.0
    return detector


def _flip_counterparts(xy, res):
    """Build the horizontally-flipped, R/L-swapped equivalent of ``xy``."""
    out = np.empty_like(xy)
    for k, name in enumerate(LUNG_ORDER):
        other = "L" + name[1:] if name[0] == "R" else "R" + name[1:]
        out[LUNG_INDEX[other]] = [res - 1 - xy[k, 0], xy[k, 1]]
    return out


def test_dense_flip_confidence_structures_the_flipped_prediction():
    detector = _detector_shell(DenseIUVDetector)
    xy = np.array([[5.0 + k, 10.0 + 2 * k] for k in range(10)])
    structured = _flip_counterparts(xy, detector.res)
    calls = []
    detector._points_from_im = lambda _im: (np.zeros_like(xy), None, None)
    detector._structure = lambda raw: calls.append(raw.copy()) or structured

    confidence = detector._flip_conf(np.zeros((8, 8, 3), np.uint8), xy)

    assert len(calls) == 1
    assert confidence == 1.0


def test_flip_confidence_is_finite_when_model_points_are_not():
    for detector_cls in (DenseIUVDetector, NormalsDetector):
        detector = _detector_shell(detector_cls)
        xy = np.zeros((10, 2), dtype=float)
        detector._points_from_im = lambda _im: (np.full_like(xy, np.nan), None, None)
        detector._structure = lambda raw: raw

        confidence = detector._flip_conf(np.zeros((8, 8, 3), np.uint8), xy)

        assert np.isfinite(confidence)
        assert confidence == 0.0


def test_normals_treat_nonfinite_depth_as_holes():
    depth = np.ones((16, 16), dtype=np.float32)
    depth[6, 6] = np.nan
    depth[9, 9] = np.inf

    normals = normals_from_depth_m(depth, fx=20.0, fy=20.0, cx=8.0, cy=8.0)

    assert np.isfinite(normals).all()
    assert np.array_equal(normals[6, 6], np.zeros(3))
    assert np.array_equal(normals[9, 9], np.zeros(3))


def test_depth_hole_fill_repairs_nonfinite_values_and_zeroes_all_bad_input():
    partly_valid = np.array([[np.nan, 2.0], [np.inf, 0.0]], dtype=np.float32)
    filled = _fill_depth_holes(partly_valid)
    assert np.isfinite(filled).all()
    assert np.allclose(filled, 2.0)

    all_bad = np.array([[np.nan, np.inf], [0.0, -1.0]], dtype=np.float32)
    assert np.array_equal(_fill_depth_holes(all_bad), np.zeros_like(all_bad))


def test_normal_image_uses_intrinsics_depth_scale(monkeypatch):
    detector = _detector_shell(NormalsDetector, res=8)
    detector.fill_holes = False
    detector.plausibility_gate = False
    captured = {}

    def fake_normals(depth_m, *_args):
        captured["depth_m"] = depth_m.copy()
        return np.zeros((*depth_m.shape, 3), dtype=np.float32)

    monkeypatch.setattr(detector_mod, "normals_from_depth_m", fake_normals)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth_raw = np.full((8, 8), 2000, dtype=np.uint16)
    intr = Intrinsics(10.0, 10.0, 4.0, 4.0, 8, 8, depth_scale=0.5)

    detector._normal_image(rgb, depth_raw, intr)

    assert np.allclose(captured["depth_m"], 1.0)


def test_normals_detection_reports_explicit_back_presence():
    detector = _detector_shell(NormalsDetector)
    detector.compute_confidence = False
    detector._normal_image = lambda _rgb, _depth, _intr: (
        np.zeros((8, 8, 3), np.uint8),
        SimpleNamespace(det_to_full=lambda xy: xy),
        False,
    )
    detector._points_from_im = lambda _im: (np.zeros((10, 2)), None, None)
    detector._structure = lambda xy: xy

    result = detector.detect(np.zeros((8, 8, 3), np.uint8), np.ones((8, 8)), object())

    assert result.back_present is False
    assert result.confidence == 0.0


class _PresenceDetector:
    def __init__(self, base):
        self.base = base
        self.back_present = True

    def set_truth(self, truth):
        self.base.set_truth(truth)

    def detect(self, rgb, depth, intr):
        result = self.base.detect(rgb, depth, intr)
        if result is None:
            return None
        fields = dict(vars(result))
        fields["back_present"] = self.back_present
        if not self.back_present:
            fields["confidence"] = 0.0
        return SimpleNamespace(**fields)


def _single_frame_tracker(*, init_frames=3, coast_max=2):
    frames, intr = make_synthetic_sequence(
        n_frames=1, motion=False, breathing=False, depth_noise_mm=0.0,
    )
    detector = _PresenceDetector(SyntheticDetector(intr, noise_px=0.0))
    detector.set_truth(frames[0].gt_pixels)
    tracker = ModelBasedTracker(
        detector, intr, ModelConfig(init_frames=init_frames, coast_max=coast_max),
    )
    return tracker, detector, frames[0]


def _initialize(tracker, frame, count):
    output = None
    for k in range(count):
        output = tracker.process(
            frame.rgb, frame.depth, t=float(k), depth_scale=frame.depth_scale,
        )
    return output


def test_model_tracker_rejects_nonfinite_detector_confidence():
    tracker, detector, frame = _single_frame_tracker(init_frames=2)
    original_detect = detector.detect

    def nan_confidence(*args):
        result = original_detect(*args)
        fields = dict(vars(result)); fields["confidence"] = np.nan
        return SimpleNamespace(**fields)

    detector.detect = nan_confidence

    assert tracker._detect3d(frame.rgb, frame.depth) is None


def test_lts_pose_fit_rejects_a_coherent_three_point_outlier_cluster():
    tracker, _, _ = _single_frame_tracker()
    rows = np.linspace(-0.1, 0.1, 5)
    canonical = np.array([
        [sign * 0.05, y, 0.02 * (y / 0.1) ** 2]
        for y in rows for sign in (1, -1)
    ])
    angle = 0.15
    expected_R = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    expected_s = 1.03
    expected_t = np.array([0.03, -0.02, 0.8])
    observations = np.array([
        [0.09657131366358433, -0.11370802610850475, 0.8223382125910963],
        [-0.00690950804403813, -0.1276203860361846, 0.820566794100714],
        [0.07920899649717172, -0.23841711757715392, 0.8076505847232768],
        [-0.021639617062835088, -0.2504222815727752, 0.806752844867373],
        [0.080776336416907, -0.011092347820290444, 0.800100003532995],
        [-0.02558005575565354, -0.20635895970609353, 0.8069272445316678],
        [0.07330782398881831, 0.03921521102672579, 0.8045420060660387],
        [-0.030501338035097053, 0.022898882591817726, 0.8048319458220365],
        [0.0650358760753341, 0.08941684789508074, 0.8204693050594536],
        [-0.037425095945875064, 0.07409663774410537, 0.8208920629315865],
    ])
    tracker.canonical = canonical

    R, scale, translation, _measured, _fit = tracker._fit_pose(observations)
    predicted = scale * (R @ canonical.T).T + translation
    expected = expected_s * (expected_R @ canonical.T).T + expected_t

    assert np.median(np.linalg.norm(predicted - expected, axis=1)) < 0.004


def test_initialization_requires_consecutive_valid_frames():
    tracker, detector, frame = _single_frame_tracker(init_frames=3)
    tracker.process(frame.rgb, frame.depth, t=0.0)
    detector.base._gt_full = None
    tracker.process(frame.rgb, frame.depth, t=1.0)
    detector.set_truth(frame.gt_pixels)

    assert not tracker.process(frame.rgb, frame.depth, t=2.0).initialized
    assert not tracker.process(frame.rgb, frame.depth, t=3.0).initialized
    assert tracker.process(frame.rgb, frame.depth, t=4.0).initialized


def test_reset_never_exports_stale_coasting_points():
    tracker, _, frame = _single_frame_tracker(init_frames=2)
    assert _initialize(tracker, frame, 2).initialized

    tracker.reset()
    output = tracker.process(None, None, t=3.0)

    assert not output.initialized
    assert output.reason == "NO_BACK"
    assert all(not np.isfinite(point.pixel).any() for point in output.points)


def test_explicit_no_back_immediately_resets_an_initialized_track():
    tracker, detector, frame = _single_frame_tracker(init_frames=2, coast_max=20)
    assert _initialize(tracker, frame, 2).initialized
    detector.back_present = False

    output = tracker.process(frame.rgb, frame.depth, t=3.0)

    assert not output.initialized
    assert output.back_present is False
    assert output.reason == "NO_BACK"
    assert all(not np.isfinite(point.pixel).any() for point in output.points)


def test_all_lost_track_resets_and_requires_fresh_consecutive_initialization():
    tracker, detector, frame = _single_frame_tracker(init_frames=2, coast_max=2)
    assert _initialize(tracker, frame, 2).initialized
    detector.base._gt_full = None

    output = None
    for k in range(3):
        output = tracker.process(frame.rgb, frame.depth, t=3.0 + k)
    assert not output.initialized
    assert output.reason == "LOST"

    detector.set_truth(frame.gt_pixels)
    assert not tracker.process(frame.rgb, frame.depth, t=7.0).initialized
    assert tracker.process(frame.rgb, frame.depth, t=8.0).initialized


def test_runtime_frame_hides_stale_points_unless_tracking_a_present_back():
    from ausc.perception.tracking.runtime import runtime_frame

    stale = SimpleNamespace(
        initialized=True,
        back_present=False,
        detector_confidence=0.0,
        reason="NO_BACK",
        points=[SimpleNamespace(
            name="R1", pixel=np.array([123.0, 234.0]), status="COASTING",
            measured=False, position_std=0.1,
        )],
    )

    output = runtime_frame(stale)

    assert output.back_present is False
    assert output.reason == "NO_BACK"
    assert output.points == ()


def test_public_runtime_frame_exports_only_finite_nonlost_tracking_points():
    from ausc.perception.tracking import runtime_frame

    output = SimpleNamespace(
        initialized=True,
        back_present=True,
        detector_confidence=0.91,
        reason="TRACKING",
        pose_measured_this_frame=True,
        pose_support_mask=np.array(
            [True, True, False, False, False, False, False, False, False, False],
            dtype=np.bool_,
        ),
        pose_geometry_accepted=True,
        pose_uncertainty_bound_m=0.006,
        sim3_rotation_camera_from_patient=np.eye(3),
        sim3_scale=1.0,
        sim3_translation_camera_from_patient_m=np.array([0.01, 0.0, 0.0]),
        points=[
            SimpleNamespace(
                name="R1", pixel=np.array([123.0, 234.0]), status="TRACKING",
                measured=True, position_std=0.003,
                xyz_camera=np.array([0.1, -0.2, 0.8]),
                xyz_patient=np.array([0.09, -0.2, 0.8]),
                measurement_residual_m=0.001,
                robot_occluded=False,
                direct_geometry_accepted=True,
            ),
            SimpleNamespace(
                name="L1", pixel=np.array([130.0, 240.0]), status="COASTING",
                measured=False, position_std=0.02,
                xyz_camera=np.array([np.nan, np.nan, np.nan]),
            ),
            SimpleNamespace(
                name="R2", pixel=np.array([140.0, 250.0]), status="LOST",
                measured=False, position_std=0.2,
                xyz_camera=np.array([0.2, -0.1, 0.9]),
            ),
        ],
    )

    frame = runtime_frame(output)

    assert frame.back_present is True
    assert frame.detector_confidence == 0.91
    assert [(point.name, point.pixel, point.xyz_camera) for point in frame.points] == [
        ("R1", (123.0, 234.0), (0.1, -0.2, 0.8)),
        ("L1", (130.0, 240.0), None),
    ]
    assert frame.pose_measured_this_frame is True
    assert frame.pose_geometry_accepted is True
    assert frame.pose_uncertainty_bound_m == pytest.approx(0.006)
    assert frame.pose_support_mask[:2] == (True, True)
    assert frame.sim3_scale == pytest.approx(1.0)
    assert frame.points[0].xyz_patient == (0.09, -0.2, 0.8)
    assert frame.points[0].measurement_residual_m == pytest.approx(0.001)
    assert frame.points[0].robot_occluded is False
    assert frame.points[0].direct_geometry_accepted is True
