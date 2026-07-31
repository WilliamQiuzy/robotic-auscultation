"""Contracts for the perception/independent geometry-channel adapters."""

from __future__ import annotations

import numpy as np
import pytest

from ausc.perception.tracking.geometry_channel_adapters import (
    DEFAULT_MIN_POSITION_STD_M,
    independent_channel_geometry,
    perception_channel_geometry,
)


def _is_packet_valid_covariance(cov: np.ndarray) -> bool:
    # Mirrors GeometryChannelPacket's covariance rule: symmetric, PSD, nonzero.
    symmetric = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    if not np.allclose(cov, symmetric, rtol=0.0, atol=1.0e-15):
        return False
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return bool(
        np.isfinite(eigenvalues).all()
        and np.all(eigenvalues >= -1.0e-15)
        and np.all(eigenvalues[:, -1] > 0.0)
    )


def test_perception_isotropic_covariance_from_scalar_std():
    points = np.array([[0.1, 0.2, 0.5], [0.3, 0.1, 0.6]], dtype=np.float64)
    std = np.array([0.01, 0.02], dtype=np.float64)
    out_points, cov = perception_channel_geometry(points, std)
    np.testing.assert_array_equal(out_points, points)
    np.testing.assert_allclose(cov[0], np.eye(3) * 0.01**2)
    np.testing.assert_allclose(cov[1], np.eye(3) * 0.02**2)
    assert _is_packet_valid_covariance(cov)


def test_perception_floors_a_zero_std_to_stay_nonsingular():
    points = np.array([[0.0, 0.0, 0.5]], dtype=np.float64)
    _out, cov = perception_channel_geometry(points, np.array([0.0]))
    assert cov[0, 0, 0] == pytest.approx(DEFAULT_MIN_POSITION_STD_M**2)
    assert _is_packet_valid_covariance(cov)


def test_perception_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="one scalar per point"):
        perception_channel_geometry(
            np.zeros((3, 3)), np.zeros((2,)),
        )


def test_perception_rejects_a_negative_std():
    # A negative std is a producer bug; it must be rejected, not floored.
    with pytest.raises(ValueError, match="non-negative"):
        perception_channel_geometry(
            np.zeros((2, 3)), np.array([0.01, -0.01]),
        )


class _PoseStub:
    def __init__(self, rotation, translation, covariance_se3):
        self.rotation_camera_from_patient = np.asarray(rotation, dtype=np.float64)
        self.translation_camera_from_patient_m = np.asarray(translation, dtype=np.float64)
        self.covariance_se3 = np.asarray(covariance_se3, dtype=np.float64)

    def camera_from_patient(self, point):
        return (
            self.rotation_camera_from_patient @ np.asarray(point, dtype=np.float64)
            + self.translation_camera_from_patient_m
        )


def test_independent_identity_pose_propagates_closed_form():
    # R = I, t = 0, Sigma = diag([a,a,a, b,b,b]).  For point p the propagated
    # covariance is a*(||p||^2 I - p p^T) + b*I.
    a, b = 4.0e-4, 1.0e-4
    sigma = np.diag([a, a, a, b, b, b]).astype(np.float64)
    pose = _PoseStub(np.eye(3), np.zeros(3), sigma)
    template = np.array([[0.2, -0.1, 0.05], [0.0, 0.3, -0.2]], dtype=np.float64)

    points, cov = independent_channel_geometry(pose, template)

    np.testing.assert_allclose(points, template)  # identity pose -> template
    for i, p in enumerate(template):
        expected = a * (float(p @ p) * np.eye(3) - np.outer(p, p)) + b * np.eye(3)
        np.testing.assert_allclose(cov[i], expected, atol=1e-12)
    assert _is_packet_valid_covariance(cov)


def test_independent_general_pose_projects_and_stays_packet_valid():
    # A proper rotation (90 deg about z) + translation.
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([0.1, -0.2, 0.5])
    sigma = np.diag([3e-4, 3e-4, 3e-4, 1e-4, 1e-4, 1e-4]).astype(np.float64)
    pose = _PoseStub(rotation, translation, sigma)
    template = np.array([[0.2, 0.1, 0.0], [-0.1, 0.2, 0.05], [0.0, 0.0, 0.3]])

    points, cov = independent_channel_geometry(pose, template)

    for i, p in enumerate(template):
        np.testing.assert_allclose(points[i], rotation @ p + translation)
    assert cov.shape == (3, 3, 3)
    assert _is_packet_valid_covariance(cov)


def test_independent_rejects_bad_shapes():
    pose = _PoseStub(np.eye(3), np.zeros(3), np.eye(6))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        independent_channel_geometry(pose, np.zeros((3, 2)))
