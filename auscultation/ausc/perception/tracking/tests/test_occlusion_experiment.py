from __future__ import annotations

import numpy as np

from ausc.perception.tracking.occlusion_experiment import (
    Scenario,
    simulate_trial,
    summarize_trials,
)
from scripts.perception.isaac_occlusion_experiment import _rgb_has_scene_content


def test_simulation_is_deterministic_for_one_seed() -> None:
    scenario = Scenario(name="deterministic", duration_s=5.0, fps=20)

    first = simulate_trial(scenario, seed=17)
    second = simulate_trial(scenario, seed=17)

    assert np.array_equal(first["visible"], second["visible"])
    assert np.allclose(first["truth"], second["truth"])
    assert np.allclose(first["patient_lock"], second["patient_lock"], equal_nan=True)


def test_patient_frame_lock_beats_absolute_camera_lock_during_motion() -> None:
    scenario = Scenario(
        name="moving_patient",
        duration_s=9.0,
        fps=30,
        patient_sway_m=0.045,
        occluder_radius_m=0.055,
    )

    summary = summarize_trials([simulate_trial(scenario, seed=s) for s in range(5)])

    assert summary["occluded_samples"] > 100
    assert summary["patient_lock_median_mm"] < 0.6 * summary["camera_lock_median_mm"]


def test_contact_handoff_reduces_error_and_uses_robot_source() -> None:
    scenario = Scenario(name="contact", duration_s=9.0, fps=30)
    trial = simulate_trial(scenario, seed=23)
    contact = trial["contact"]

    assert contact.any()
    assert set(trial["hybrid_source"][contact]) == {"robot_kinematic_contact"}
    patient_error = np.linalg.norm(trial["patient_lock"][contact] - trial["truth"][contact], axis=1)
    hybrid_error = np.linalg.norm(trial["hybrid"][contact] - trial["truth"][contact], axis=1)
    assert np.nanmedian(hybrid_error) < np.nanmedian(patient_error)


def test_severe_occlusion_exposes_bounded_freeze_state() -> None:
    scenario = Scenario(
        name="severe",
        duration_s=9.0,
        fps=30,
        occluder_radius_m=0.14,
        max_freeze_frames=8,
    )
    trial = simulate_trial(scenario, seed=3)

    assert "patient_frame_frozen" in set(trial["patient_source"])
    assert "lost" in set(trial["patient_source"])


def test_summary_uses_none_when_a_strategy_has_no_valid_samples() -> None:
    scenario = Scenario(
        name="severe",
        duration_s=9.0,
        fps=30,
        occluder_radius_m=0.14,
        max_freeze_frames=8,
    )

    summary = summarize_trials([simulate_trial(scenario, seed=3)])

    assert summary["contact_patient_median_mm"] is None


def test_render_gate_rejects_black_frame_and_accepts_scene_content() -> None:
    black = np.zeros((120, 160, 3), dtype=np.uint8)
    scene = black.copy()
    scene[20:100, 30:130] = np.array([30, 90, 180], dtype=np.uint8)

    assert not _rgb_has_scene_content(black)
    assert _rgb_has_scene_content(scene)
