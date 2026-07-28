from __future__ import annotations

import numpy as np

from ausc.perception.tracking.occlusion_strategy import (
    OcclusionConfig,
    OcclusionTargetTracker,
)


def _points() -> np.ndarray:
    return np.array(
        [
            [-0.10, 0.18, 1.00], [0.10, 0.18, 1.00],
            [-0.08, 0.09, 1.01], [0.08, 0.09, 1.01],
            [-0.07, 0.00, 1.02], [0.07, 0.00, 1.02],
            [-0.09, -0.09, 1.01], [0.09, -0.09, 1.01],
            [-0.12, -0.18, 1.00], [0.12, -0.18, 1.00],
        ],
        dtype=float,
    )


def test_patient_frame_lock_follows_rigid_patient_motion() -> None:
    reference = _points()
    tracker = OcclusionTargetTracker(reference, target_index=4)
    tracker.update(reference, np.ones(10, dtype=bool))

    angle = np.deg2rad(7.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
    )
    translation = np.array([0.035, -0.012, 0.018])
    moved = (rotation @ reference.T).T + translation
    visible = np.ones(10, dtype=bool)
    visible[4] = False

    estimate = tracker.update(moved, visible)

    assert estimate.source == "patient_frame_locked"
    assert np.linalg.norm(estimate.position - moved[4]) < 1e-9
    assert estimate.visible_support == 9


def test_camera_lock_does_not_follow_patient_translation() -> None:
    reference = _points()
    tracker = OcclusionTargetTracker(reference, target_index=4)
    tracker.update(reference, np.ones(10, dtype=bool))
    locked = tracker.locked_camera_position.copy()

    moved = reference + np.array([0.040, 0.0, 0.0])
    visible = np.ones(10, dtype=bool)
    visible[4] = False
    tracker.update(moved, visible)

    assert np.isclose(np.linalg.norm(locked - moved[4]), 0.040)


def test_insufficient_support_freezes_only_for_bounded_window() -> None:
    reference = _points()
    tracker = OcclusionTargetTracker(
        reference,
        target_index=4,
        config=OcclusionConfig(min_visible=6, max_freeze_frames=2, coast_growth_m=0.004),
    )
    tracker.update(reference, np.ones(10, dtype=bool))
    visible = np.zeros(10, dtype=bool)
    visible[:4] = True

    first = tracker.update(reference, visible)
    second = tracker.update(reference, visible)
    lost = tracker.update(reference, visible)

    assert first.source == "patient_frame_frozen"
    assert second.source == "patient_frame_frozen"
    assert second.position_std_m > first.position_std_m
    assert lost.source == "lost"
    assert np.isnan(lost.position).all()


def test_locked_std_scales_with_extrapolation_distance() -> None:
    # An occluded point's honest uncertainty grows with how far it is
    # extrapolated from the visible support (real-data LOO: corr 0.44). A corner
    # target reconstructed from a far, one-sided support must report a larger std
    # than the same target reconstructed from the full surrounding constellation,
    # and both must exceed the legacy flat 5mm for a real extrapolation.
    reference = _points()

    near = OcclusionTargetTracker(reference, target_index=8)
    near.update(reference, np.ones(10, dtype=bool))
    near_visible = np.ones(10, dtype=bool)
    near_visible[8] = False
    est_near = near.update(reference, near_visible)

    far = OcclusionTargetTracker(reference, target_index=8)
    far.update(reference, np.ones(10, dtype=bool))
    far_visible = np.zeros(10, dtype=bool)
    far_visible[:6] = True  # only the top/mid rows; target 8 is the far bottom
    est_far = far.update(reference, far_visible)

    assert est_near.source == "patient_frame_locked"
    assert est_far.source == "patient_frame_locked"
    assert est_far.position_std_m > est_near.position_std_m
    assert est_near.position_std_m > 0.005


def test_freeze_coasts_from_adaptive_locked_std() -> None:
    # A freeze must coast from the real extrapolation-scaled std it last held,
    # not from the legacy constant, so the coast starts honestly wide.
    reference = _points()
    tracker = OcclusionTargetTracker(
        reference,
        target_index=8,
        config=OcclusionConfig(min_visible=6, max_freeze_frames=5, coast_growth_m=0.004),
    )
    tracker.update(reference, np.ones(10, dtype=bool))
    locked_visible = np.ones(10, dtype=bool)
    locked_visible[8] = False
    est_locked = tracker.update(reference, locked_visible)

    freeze_visible = np.zeros(10, dtype=bool)
    freeze_visible[:3] = True  # support < min_visible -> freeze
    est_freeze = tracker.update(reference, freeze_visible)

    assert est_locked.source == "patient_frame_locked"
    assert est_freeze.source == "patient_frame_frozen"
    assert np.isclose(est_freeze.position_std_m, est_locked.position_std_m + 0.004)


def test_contact_handoff_uses_calibrated_robot_tip() -> None:
    reference = _points()
    tracker = OcclusionTargetTracker(reference, target_index=4)
    tracker.update(reference, np.ones(10, dtype=bool))
    visible = np.ones(10, dtype=bool)
    visible[4] = False
    tip = reference[4] + np.array([0.001, -0.001, 0.002])

    estimate = tracker.update(reference, visible, contact=True, tool_tip_position=tip)

    assert estimate.source == "robot_kinematic_contact"
    assert np.array_equal(estimate.position, tip)
    assert estimate.position_std_m == tracker.config.contact_std_m
