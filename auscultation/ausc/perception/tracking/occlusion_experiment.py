"""Deterministic kinematic occlusion experiment shared by Isaac rendering and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .occlusion_strategy import OcclusionConfig, OcclusionTargetTracker


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_s: float = 9.0
    fps: int = 30
    target_index: int = 4
    patient_sway_m: float = 0.035
    breathing_amplitude_m: float = 0.005
    measurement_noise_m: float = 0.0025
    kinematic_noise_m: float = 0.0015
    occluder_radius_m: float = 0.055
    approach_start_s: float = 1.0
    contact_start_s: float = 4.0
    contact_end_s: float = 6.2
    depart_end_s: float = 8.2
    min_visible: int = 6
    max_freeze_frames: int = 20


def canonical_back_points() -> np.ndarray:
    return np.array(
        [
            [-0.105, 0.180, 1.000], [0.105, 0.180, 1.000],
            [-0.082, 0.090, 1.012], [0.082, 0.090, 1.012],
            [-0.070, 0.000, 1.022], [0.070, 0.000, 1.022],
            [-0.092, -0.090, 1.012], [0.092, -0.090, 1.012],
            [-0.125, -0.180, 1.000], [0.125, -0.180, 1.000],
        ],
        dtype=float,
    )


def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _rotation_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return z @ y @ x


def _patient_points(reference: np.ndarray, t: float, scenario: Scenario) -> np.ndarray:
    breathing_weights = np.array([0.45, 0.45, 0.8, 0.8, 1.0, 1.0, 0.8, 0.8, 0.4, 0.4])
    deformed = reference.copy()
    deformed[:, 2] += (
        scenario.breathing_amplitude_m
        * breathing_weights
        * np.sin(2.0 * np.pi * 0.25 * t)
    )
    rotation = _rotation_xyz(
        np.deg2rad(1.5) * np.sin(2.0 * np.pi * 0.09 * t),
        np.deg2rad(3.0) * np.sin(2.0 * np.pi * 0.11 * t + 0.4),
        np.deg2rad(5.0) * np.sin(2.0 * np.pi * 0.13 * t + 0.2),
    )
    translation = np.array(
        [
            scenario.patient_sway_m * np.sin(2.0 * np.pi * 0.12 * t),
            0.012 * np.sin(2.0 * np.pi * 0.17 * t + 0.7),
            0.012 * np.sin(2.0 * np.pi * 0.08 * t + 0.5),
        ]
    )
    center = np.array([0.0, 0.0, 1.0])
    return (rotation @ (deformed - center).T).T + center + translation


def _arm_keypoints(t: float, target: np.ndarray, scenario: Scenario) -> tuple[np.ndarray, bool]:
    start_tip = np.array([0.38, 0.24, 0.55])
    contact_tip = target.copy()
    if t < scenario.contact_start_s:
        progress = _smoothstep(
            (t - scenario.approach_start_s)
            / max(scenario.contact_start_s - scenario.approach_start_s, 1e-6)
        )
        tip = (1.0 - progress) * start_tip + progress * contact_tip
    elif t <= scenario.contact_end_s:
        progress = 1.0
        tip = contact_tip
    else:
        depart = _smoothstep(
            (t - scenario.contact_end_s)
            / max(scenario.depart_end_s - scenario.contact_end_s, 1e-6)
        )
        progress = 1.0 - depart
        tip = progress * contact_tip + (1.0 - progress) * start_tip

    shoulder = np.array([0.48, -0.28, 0.40])
    elbow_contact = np.array([0.31, -0.12, 0.62])
    elbow_home = np.array([0.44, -0.18, 0.48])
    elbow = progress * elbow_contact + (1.0 - progress) * elbow_home
    wrist_offset = np.array([0.12, 0.055, -0.12])
    wrist = tip + wrist_offset
    contact = scenario.contact_start_s <= t <= scenario.contact_end_s
    return np.stack([shoulder, elbow, wrist, tip]), contact


def _point_segment_distance_xy(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    direction = second[:2] - first[:2]
    denominator = float(direction @ direction)
    fraction = 0.0 if denominator <= 1e-12 else float(
        np.clip(((point[:2] - first[:2]) @ direction) / denominator, 0.0, 1.0)
    )
    closest = first + fraction * (second - first)
    return float(np.linalg.norm(point[:2] - closest[:2])), float(closest[2])


def _visibility(points: np.ndarray, arm: np.ndarray, radius: float) -> np.ndarray:
    visible = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        for first, second in zip(arm[:-1], arm[1:]):
            distance, arm_depth = _point_segment_distance_xy(point, first, second)
            # A contacting tool tip is at the skin depth, while every upstream link is closer to
            # the camera. Both block line of sight; requiring a large strict depth gap would
            # incorrectly mark the exact contact target as visible.
            if distance <= radius and arm_depth <= point[2] + 0.003:
                visible[index] = False
                break
    return visible


def simulate_trial(scenario: Scenario, seed: int) -> dict[str, np.ndarray | int | dict]:
    rng = np.random.default_rng(seed)
    reference = canonical_back_points()
    count = int(round(scenario.duration_s * scenario.fps)) + 1
    times = np.arange(count, dtype=float) / scenario.fps
    truth_points = np.stack([_patient_points(reference, t, scenario) for t in times])
    first_reference = truth_points[0].copy()
    config = OcclusionConfig(
        min_visible=scenario.min_visible,
        max_freeze_frames=scenario.max_freeze_frames,
        coast_growth_m=0.003,
        contact_std_m=scenario.kinematic_noise_m,
    )
    patient_tracker = OcclusionTargetTracker(first_reference, scenario.target_index, config)
    hybrid_tracker = OcclusionTargetTracker(first_reference, scenario.target_index, config)

    visible_rows = []
    measured_rows = []
    arm_rows = []
    contact_rows = []
    camera_rows = []
    patient_rows = []
    hybrid_rows = []
    patient_sources = []
    hybrid_sources = []
    patient_std = []
    support = []
    camera_lock = first_reference[scenario.target_index].copy()

    for frame, t in enumerate(times):
        points = truth_points[frame]
        arm, contact = _arm_keypoints(float(t), points[scenario.target_index], scenario)
        visible = _visibility(points, arm, scenario.occluder_radius_m)
        measured = points + rng.normal(0.0, scenario.measurement_noise_m, points.shape)
        measured[~visible] = np.nan
        if visible[scenario.target_index]:
            camera_lock = measured[scenario.target_index].copy()

        patient = patient_tracker.update(measured, visible)
        if contact:
            tool_tip = points[scenario.target_index] + rng.normal(
                0.0, scenario.kinematic_noise_m, 3,
            )
            hybrid = hybrid_tracker.update(
                measured,
                visible,
                contact=True,
                tool_tip_position=tool_tip,
            )
        else:
            hybrid = hybrid_tracker.update(measured, visible)

        visible_rows.append(visible)
        measured_rows.append(measured)
        arm_rows.append(arm)
        contact_rows.append(contact)
        camera_rows.append(camera_lock.copy())
        patient_rows.append(patient.position)
        hybrid_rows.append(hybrid.position)
        patient_sources.append(patient.source)
        hybrid_sources.append(hybrid.source)
        patient_std.append(patient.position_std_m)
        support.append(patient.visible_support)

    return {
        "scenario": asdict(scenario),
        "seed": int(seed),
        "time_s": times,
        "truth": truth_points[:, scenario.target_index],
        "truth_points": truth_points,
        "measured_points": np.asarray(measured_rows),
        "visible": np.asarray(visible_rows),
        "arm_keypoints": np.asarray(arm_rows),
        "contact": np.asarray(contact_rows, dtype=bool),
        "camera_lock": np.asarray(camera_rows),
        "patient_lock": np.asarray(patient_rows),
        "hybrid": np.asarray(hybrid_rows),
        "patient_source": np.asarray(patient_sources, dtype="U32"),
        "hybrid_source": np.asarray(hybrid_sources, dtype="U32"),
        "patient_std_m": np.asarray(patient_std),
        "visible_support": np.asarray(support, dtype=int),
    }


def _errors(trial: dict, key: str, mask: np.ndarray) -> np.ndarray:
    estimate = np.asarray(trial[key])[mask]
    truth = np.asarray(trial["truth"])[mask]
    error = np.linalg.norm(estimate - truth, axis=1) * 1000.0
    return error[np.isfinite(error)]


def summarize_trials(trials: Iterable[dict]) -> dict[str, float | int | None]:
    camera_errors = []
    patient_errors = []
    hybrid_errors = []
    contact_patient = []
    contact_hybrid = []
    occluded_samples = lost_samples = frozen_samples = 0
    for trial in trials:
        target = int(trial["scenario"]["target_index"])
        occluded = ~np.asarray(trial["visible"])[:, target]
        contact = np.asarray(trial["contact"], dtype=bool)
        camera_errors.extend(_errors(trial, "camera_lock", occluded))
        patient_errors.extend(_errors(trial, "patient_lock", occluded))
        hybrid_errors.extend(_errors(trial, "hybrid", occluded))
        contact_patient.extend(_errors(trial, "patient_lock", contact))
        contact_hybrid.extend(_errors(trial, "hybrid", contact))
        source = np.asarray(trial["patient_source"])
        occluded_samples += int(occluded.sum())
        lost_samples += int(np.sum(source == "lost"))
        frozen_samples += int(np.sum(source == "patient_frame_frozen"))

    def percentile(values: list[float], q: float) -> float | None:
        return float(np.percentile(values, q)) if values else None

    return {
        "occluded_samples": occluded_samples,
        "camera_lock_median_mm": percentile(camera_errors, 50),
        "camera_lock_p90_mm": percentile(camera_errors, 90),
        "patient_lock_median_mm": percentile(patient_errors, 50),
        "patient_lock_p90_mm": percentile(patient_errors, 90),
        "hybrid_median_mm": percentile(hybrid_errors, 50),
        "hybrid_p90_mm": percentile(hybrid_errors, 90),
        "contact_patient_median_mm": percentile(contact_patient, 50),
        "contact_hybrid_median_mm": percentile(contact_hybrid, 50),
        "lost_samples": lost_samples,
        "frozen_samples": frozen_samples,
    }
