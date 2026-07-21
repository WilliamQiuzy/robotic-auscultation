"""Patient-relative target locking for robot-occluded auscultation points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class OcclusionConfig:
    min_visible: int = 6
    max_freeze_frames: int = 20
    vision_std_m: float = 0.003
    locked_std_m: float = 0.005
    coast_growth_m: float = 0.003
    contact_std_m: float = 0.003


@dataclass(frozen=True)
class TargetEstimate:
    position: np.ndarray
    source: str
    position_std_m: float
    visible_support: int


def _similarity(source: np.ndarray, destination: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_zero = source - source_mean
    destination_zero = destination - destination_mean
    covariance = destination_zero.T @ source_zero / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    reflection = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        reflection[2, 2] = -1
    rotation = u @ reflection @ vt
    source_variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    scale = float((singular * np.diag(reflection)).sum() / max(source_variance, 1e-12))
    translation = destination_mean - scale * rotation @ source_mean
    return rotation, scale, translation


def _robust_similarity(
    source: np.ndarray,
    destination: np.ndarray,
    minimum: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    keep = np.arange(len(source))
    retained = max(minimum, int(round(0.75 * len(source))))
    for _ in range(4):
        rotation, scale, translation = _similarity(source[keep], destination[keep])
        fitted = scale * (rotation @ source.T).T + translation
        residual = np.linalg.norm(destination - fitted, axis=1)
        keep = np.argsort(residual, kind="stable")[:retained]
    return _similarity(source[keep], destination[keep])


class OcclusionTargetTracker:
    """Latch one target and keep it patient-relative until robot contact or visual reacquisition."""

    def __init__(
        self,
        reference_points: np.ndarray,
        target_index: int,
        config: Optional[OcclusionConfig] = None,
    ) -> None:
        reference = np.asarray(reference_points, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError("reference_points must have shape (N, 3)")
        if not 0 <= target_index < len(reference):
            raise ValueError("target_index is out of range")
        self.reference_points = reference.copy()
        self.target_index = int(target_index)
        self.config = config or OcclusionConfig()
        self.rotation = np.eye(3)
        self.scale = 1.0
        self.translation = np.zeros(3)
        self.locked_camera_position = reference[self.target_index].copy()
        self._freeze_frames = 0

    def _patient_position(self) -> np.ndarray:
        point = self.reference_points[self.target_index]
        return self.scale * (self.rotation @ point) + self.translation

    def update(
        self,
        measured_points: np.ndarray,
        visible: np.ndarray,
        *,
        contact: bool = False,
        tool_tip_position: Optional[np.ndarray] = None,
    ) -> TargetEstimate:
        measured = np.asarray(measured_points, dtype=float)
        visibility = np.asarray(visible, dtype=bool)
        if measured.shape != self.reference_points.shape or visibility.shape != (len(measured),):
            raise ValueError("measurement shapes do not match the reference constellation")
        finite_visible = visibility & np.isfinite(measured).all(axis=1)
        support = int(finite_visible.sum())

        if contact:
            tip = np.asarray(tool_tip_position, dtype=float) if tool_tip_position is not None else None
            if tip is None or tip.shape != (3,) or not np.isfinite(tip).all():
                raise ValueError("contact requires a finite calibrated tool_tip_position")
            return TargetEstimate(tip.copy(), "robot_kinematic_contact", self.config.contact_std_m, support)

        if finite_visible[self.target_index]:
            if support >= self.config.min_visible:
                self.rotation, self.scale, self.translation = _robust_similarity(
                    self.reference_points[finite_visible],
                    measured[finite_visible],
                    self.config.min_visible,
                )
            position = measured[self.target_index].copy()
            self.locked_camera_position = position.copy()
            self._freeze_frames = 0
            return TargetEstimate(position, "vision_measured", self.config.vision_std_m, support)

        if support >= self.config.min_visible:
            self.rotation, self.scale, self.translation = _robust_similarity(
                self.reference_points[finite_visible],
                measured[finite_visible],
                self.config.min_visible,
            )
            self._freeze_frames = 0
            return TargetEstimate(
                self._patient_position(),
                "patient_frame_locked",
                self.config.locked_std_m,
                support,
            )

        self._freeze_frames += 1
        if self._freeze_frames <= self.config.max_freeze_frames:
            return TargetEstimate(
                self._patient_position(),
                "patient_frame_frozen",
                self.config.locked_std_m + self._freeze_frames * self.config.coast_growth_m,
                support,
            )
        return TargetEstimate(np.full(3, np.nan), "lost", float("inf"), support)

