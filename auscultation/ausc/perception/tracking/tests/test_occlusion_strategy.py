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
