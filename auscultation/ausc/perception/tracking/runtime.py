"""Fail-closed adapter from tracker output to live UI/runtime point payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .model_tracker import REASON_TRACKING, STATUS_LOST


@dataclass(frozen=True)
class RuntimePoint:
    name: str
    pixel: Tuple[float, float]
    xyz_camera: Optional[Tuple[float, float, float]]
    status: str
    measured: bool
    position_std: float
    xyz_patient: Optional[Tuple[float, float, float]]
    measurement_residual_m: float
    robot_occluded: bool
    direct_geometry_accepted: bool


@dataclass(frozen=True)
class RuntimeFrame:
    back_present: bool
    reason: str
    initialized: bool
    detector_confidence: float
    points: Tuple[RuntimePoint, ...]
    pose_measured_this_frame: bool
    pose_support_mask: Tuple[bool, ...]
    pose_geometry_accepted: bool
    pose_uncertainty_bound_m: float
    sim3_rotation_camera_from_patient: Optional[
        Tuple[Tuple[float, float, float], ...]
    ]
    sim3_scale: float
    sim3_translation_camera_from_patient_m: Optional[
        Tuple[float, float, float]
    ]
    # DISPLAY-ONLY passthrough (2026-08-29): the detector's structure-fitted
    # full-frame pixels for all landmarks, carried verbatim from
    # MTOutput.detector_pixels so the operator console can render withheld
    # points in an explicit unverified style. Never consumed downstream.
    detector_pixels: Optional[Tuple[Tuple[float, float], ...]] = None
    # Canonical landmark order (all 10 names, matching detector_pixels rows),
    # independent of which points are currently visible.
    point_order: Tuple[str, ...] = ()


def runtime_frame(output) -> RuntimeFrame:
    """Expose points only for an initialized track on an explicitly present back."""
    visible = bool(output.back_present and output.initialized
                   and output.reason == REASON_TRACKING)
    points = []
    if visible:
        for point in output.points:
            pixel = np.asarray(point.pixel, dtype=float).reshape(-1)
            if point.status == STATUS_LOST or pixel.size != 2 or not np.isfinite(pixel).all():
                continue
            xyz = np.asarray(getattr(point, "xyz_camera", None), dtype=float).reshape(-1)
            xyz_camera = ((float(xyz[0]), float(xyz[1]), float(xyz[2]))
                          if xyz.size == 3 and np.isfinite(xyz).all() else None)
            patient = np.asarray(
                getattr(point, "xyz_patient", None), dtype=float,
            ).reshape(-1)
            xyz_patient = (
                (float(patient[0]), float(patient[1]), float(patient[2]))
                if patient.size == 3 and np.isfinite(patient).all()
                else None
            )
            residual = float(
                getattr(point, "measurement_residual_m", float("inf"))
            )
            robot_occluded = bool(getattr(point, "robot_occluded", False))
            direct_geometry_accepted = bool(
                getattr(point, "direct_geometry_accepted", False)
                and xyz_camera is not None
                and xyz_patient is not None
                and np.isfinite(residual)
                and not robot_occluded
            )
            points.append(RuntimePoint(
                name=str(point.name),
                pixel=(float(pixel[0]), float(pixel[1])),
                xyz_camera=xyz_camera,
                status=str(point.status),
                measured=bool(point.measured),
                position_std=float(point.position_std),
                xyz_patient=xyz_patient,
                measurement_residual_m=residual,
                robot_occluded=robot_occluded,
                direct_geometry_accepted=direct_geometry_accepted,
            ))
    support_raw = np.asarray(
        getattr(output, "pose_support_mask", ()), dtype=np.bool_,
    ).reshape(-1)
    support = tuple(bool(value) for value in support_raw)
    rotation_raw = np.asarray(
        getattr(output, "sim3_rotation_camera_from_patient", None),
        dtype=float,
    )
    translation_raw = np.asarray(
        getattr(output, "sim3_translation_camera_from_patient_m", None),
        dtype=float,
    ).reshape(-1)
    rotation = (
        tuple(tuple(float(value) for value in row) for row in rotation_raw)
        if rotation_raw.shape == (3, 3) and np.isfinite(rotation_raw).all()
        else None
    )
    translation = (
        tuple(float(value) for value in translation_raw)
        if translation_raw.shape == (3,) and np.isfinite(translation_raw).all()
        else None
    )
    scale = float(getattr(output, "sim3_scale", float("nan")))
    uncertainty = float(
        getattr(output, "pose_uncertainty_bound_m", float("inf"))
    )
    pose_geometry_accepted = bool(
        getattr(output, "pose_geometry_accepted", False)
        and rotation is not None
        and translation is not None
        and np.isfinite(scale)
        and scale > 0.0
        and np.isfinite(uncertainty)
    )
    detector_pixels = None
    raw_detector_pixels = getattr(output, "detector_pixels", None)
    if raw_detector_pixels is not None:
        rows = []
        for row in np.asarray(raw_detector_pixels, dtype=float).reshape(-1, 2):
            rows.append((float(row[0]), float(row[1])))
        detector_pixels = tuple(rows)
    return RuntimeFrame(
        back_present=bool(output.back_present),
        reason=str(output.reason),
        initialized=bool(output.initialized),
        detector_confidence=float(output.detector_confidence),
        points=tuple(points),
        pose_measured_this_frame=bool(
            getattr(output, "pose_measured_this_frame", False)
        ),
        pose_support_mask=support,
        pose_geometry_accepted=pose_geometry_accepted,
        pose_uncertainty_bound_m=uncertainty,
        sim3_rotation_camera_from_patient=rotation,
        sim3_scale=scale,
        sim3_translation_camera_from_patient_m=translation,
        detector_pixels=detector_pixels,
        point_order=tuple(
            str(getattr(point, "name", "")) for point in output.points
        ),
    )
