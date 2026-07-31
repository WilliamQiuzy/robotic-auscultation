"""Adapters that turn the two live model outputs into geometry-channel arrays.

The dual-channel geometry packet needs, per landmark, a metric position
``points_m`` ``(N,3)`` and a full ``covariances_m2`` ``(N,3,3)`` for each of the
perception and independent channels.  Neither producer emits that pair directly:

* the perception tracker gives a metric ``xyz_camera`` per landmark plus a
  **scalar** 1-sigma ``position_std`` — so its covariance is isotropic
  ``std^2 * I3`` (with a tiny floor so it is never singular); and
* the independent metric patient-pose gives an SE(3) pose + a 6x6
  ``covariance_se3`` — so each landmark is the canonical patient-frame template
  point pushed through the pose (``R p + t``) and its covariance is the pose
  covariance propagated by ``J = [-R [p]_x | I3]`` (``J Sigma J^T``).

The propagation reuses the *exact* Jacobian and convention the pose packet
already publishes (``rotation_patient_rad_then_translation_camera_m``, right
perturbation, patient->camera; see ``PatientPosePacket.propagated_position_bound_m``),
so this module introduces no new geometry convention — only the per-landmark 3x3
form the packet contract requires.  Both functions are pure and fully unit-tested.
"""

from __future__ import annotations

import numpy as np

#: Floor on the isotropic 1-sigma so a perfectly-confident perception point still
#: yields a symmetric, positive-definite (nonzero) covariance the packet accepts.
DEFAULT_MIN_POSITION_STD_M = 1.0e-6


def _skew(point: np.ndarray) -> np.ndarray:
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )


def perception_channel_geometry(
    points_xyz_camera_m: np.ndarray,
    position_std_m: np.ndarray,
    *,
    min_position_std_m: float = DEFAULT_MIN_POSITION_STD_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the perception channel's ``(points_m, covariances_m2)``.

    ``position_std_m`` is the per-landmark scalar 1-sigma; the covariance is the
    isotropic ``max(std, floor)^2 * I3``.
    """

    points = np.asarray(points_xyz_camera_m, dtype=np.float64)
    std = np.asarray(position_std_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("perception points must be (N, 3)")
    if std.shape != (points.shape[0],):
        raise ValueError("perception position std must be one scalar per point")
    if not np.isfinite(points).all() or not np.isfinite(std).all():
        raise ValueError("perception geometry has non-finite values")
    # A negative std is a producer bug, not a tiny uncertainty: reject it rather
    # than silently flooring it to a small variance.
    if np.any(std < 0.0):
        raise ValueError("perception position std must be non-negative")
    if float(min_position_std_m) <= 0.0:
        raise ValueError("minimum position std must be positive")
    floored = np.maximum(std, float(min_position_std_m))
    variance = floored * floored
    covariances = variance[:, None, None] * np.eye(3, dtype=np.float64)[None, :, :]
    return points, covariances


def independent_channel_geometry(
    pose_packet: object,
    template_points_patient_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the canonical patient-frame template through the pose + propagate cov.

    ``pose_packet`` is a ``PatientPosePacket``-like object exposing
    ``rotation_camera_from_patient`` (3x3), ``covariance_se3`` (6x6), and
    ``camera_from_patient(point)`` (``R p + t``).  Returns the per-landmark
    camera-frame ``(points_m, covariances_m2)``.
    """

    template = np.asarray(template_points_patient_m, dtype=np.float64)
    if template.ndim != 2 or template.shape[1] != 3:
        raise ValueError("template points must be (N, 3)")
    if not np.isfinite(template).all():
        raise ValueError("template points have non-finite values")
    rotation = np.asarray(pose_packet.rotation_camera_from_patient, dtype=np.float64)
    covariance_se3 = np.asarray(pose_packet.covariance_se3, dtype=np.float64)
    if rotation.shape != (3, 3) or covariance_se3.shape != (6, 6):
        raise ValueError("pose packet rotation/covariance shapes are invalid")

    count = template.shape[0]
    points = np.empty((count, 3), dtype=np.float64)
    covariances = np.empty((count, 3, 3), dtype=np.float64)
    for index in range(count):
        point_patient = template[index]
        points[index] = np.asarray(
            pose_packet.camera_from_patient(point_patient), dtype=np.float64
        )
        jacobian = np.hstack(
            (-rotation @ _skew(point_patient), np.eye(3, dtype=np.float64))
        )
        propagated = jacobian @ covariance_se3 @ jacobian.T
        # The packet requires exact symmetry (atol 1e-15); symmetrize explicitly.
        covariances[index] = 0.5 * (propagated + propagated.T)
    return points, covariances


__all__ = [
    "DEFAULT_MIN_POSITION_STD_M",
    "independent_channel_geometry",
    "perception_channel_geometry",
]
