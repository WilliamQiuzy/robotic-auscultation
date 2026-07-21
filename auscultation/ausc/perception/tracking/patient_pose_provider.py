"""Independent metric patient-pose candidate for fixed-camera RGB-D capture.

This provider intentionally does not consume the learned ten-point predictions.
It defines a patient coordinate frame from the first independently accepted
back-support RGB-D frame, keeps a bounded set of metric reference surfels and
RGB features, and estimates subsequent camera-from-patient transforms from:

1. forward/backward Lucas-Kanade feature tracks outside the interval-swept
   robot mask;
2. aligned metric depth at those tracked pixels;
3. robust 3-D RANSAC followed by a bounded paired point-to-point ICP refinement.

Every authorization gate is fail closed.  The class is a candidate provider,
not evidence that any real camera/profile has been commissioned.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import secrets
from typing import Optional, Tuple

import cv2
import numpy as np

from .geometry import Intrinsics
from .depth_noise_profile import ActivatedDepthNoiseProfile
from .occlusion_runtime import PatientPosePacket
from .patient_frame_evidence import (
    PatientFrameEvidence,
    PatientFrameEvidenceError,
)


_SOURCE = "rgbd_feature_ransac_plus_icp"
_CHI2_3D_95 = 7.814727903251179
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[a-z][a-z0-9_]*-[0-9a-f]{20}$")


def _skew(point: np.ndarray) -> np.ndarray:
    x, y, z = (float(value) for value in point)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64,)


def _rigid_align(
    source: np.ndarray, target: np.ndarray, weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted Kabsch alignment mapping ``source`` to ``target``."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(source), dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(weight))
    if (
        source.ndim != 2
        or target.shape != source.shape
        or source.shape[1:] != (3,)
        or len(source) < 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or not np.isfinite(weight).all()
        or total <= 0.0
    ):
        raise ValueError("rigid alignment input is invalid")
    weight = weight / total
    source_mean = np.sum(source * weight[:, None], axis=0)
    target_mean = np.sum(target * weight[:, None], axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (weight[:, None] * source_centered).T @ target_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if float(np.linalg.det(rotation)) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def _similarity_scale(source: np.ndarray, target: np.ndarray) -> float:
    """Umeyama scale audit without authorizing a non-rigid transform."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_centered = source - np.mean(source, axis=0)
    target_centered = target - np.mean(target, axis=0)
    denominator = float(np.sum(source_centered * source_centered))
    if denominator <= 1.0e-12:
        return float("nan")
    singular_values = np.linalg.svd(
        source_centered.T @ target_centered, compute_uv=False
    )
    return float(np.sum(singular_values) / denominator)


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


@dataclass(frozen=True)
class PatientPoseProviderConfig:
    """Bounded runtime and fail-closed acceptance envelope."""

    max_features: int = 160
    max_surfels: int = 320
    max_reference_bytes: int = 64 * 1024
    min_math_support: int = 4
    min_back_pixels: int = 256
    max_robot_back_fraction: float = 0.65
    robot_mask_dilation_px: int = 4
    feature_quality: float = 0.005
    feature_min_distance_px: int = 5
    lk_window_px: int = 21
    lk_max_level: int = 3
    max_forward_backward_error_px: float = 1.5
    min_depth_m: float = 0.20
    max_depth_m: float = 3.0
    min_second_spatial_eigenvalue_m2: float = 2.0e-4
    ransac_iterations: int = 96
    ransac_threshold_m: float = 0.008
    min_ransac_inlier_fraction: float = 0.40
    icp_iterations: int = 4
    icp_huber_delta_m: float = 0.004
    max_fit_rmse_m: float = 0.004
    scale_tolerance: float = 0.01
    max_translation_step_m: float = 0.050
    max_rotation_step_rad: float = math.radians(15.0)
    max_covariance_condition: float = 1.0e9
    max_target_position_bound_m: float = 0.010
    target_radius_m: float = 0.45
    max_frame_gap_s: float = 0.25

    def __post_init__(self) -> None:
        integer_values = (
            self.max_features,
            self.max_surfels,
            self.max_reference_bytes,
            self.min_math_support,
            self.min_back_pixels,
            self.robot_mask_dilation_px,
            self.feature_min_distance_px,
            self.lk_window_px,
            self.lk_max_level,
            self.ransac_iterations,
            self.icp_iterations,
        )
        if any(type(value) is not int or value < 0 for value in integer_values):
            raise ValueError("patient pose integer limits are invalid")
        if (
            self.min_math_support < 4
            or self.max_features < self.min_math_support
            or self.max_surfels < self.min_math_support
            or self.lk_window_px < 3
            or self.ransac_iterations < 1
            or self.icp_iterations < 1
        ):
            raise ValueError("patient pose mathematical/runtime bounds are invalid")
        finite_positive = (
            self.feature_quality,
            self.max_forward_backward_error_px,
            self.min_depth_m,
            self.max_depth_m,
            self.min_second_spatial_eigenvalue_m2,
            self.ransac_threshold_m,
            self.icp_huber_delta_m,
            self.max_fit_rmse_m,
            self.scale_tolerance,
            self.max_translation_step_m,
            self.max_rotation_step_rad,
            self.max_covariance_condition,
            self.max_target_position_bound_m,
            self.target_radius_m,
            self.max_frame_gap_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("patient pose numeric limits are invalid")
        if (
            not 0.0 < self.max_robot_back_fraction < 1.0
            or not 0.0 < self.min_ransac_inlier_fraction <= 1.0
            or self.min_depth_m >= self.max_depth_m
            or self.scale_tolerance > 0.01
            or self.max_target_position_bound_m > 0.010
        ):
            raise ValueError("patient pose safety envelope is invalid")
        if (
            self.min_back_pixels < 256
            or self.max_robot_back_fraction > 0.65
            or self.robot_mask_dilation_px < 4
            or self.feature_quality < 0.005
            or self.feature_min_distance_px < 5
            or self.lk_max_level > 3
            or self.max_forward_backward_error_px > 1.5
            or self.min_second_spatial_eigenvalue_m2 < 2.0e-4
            or self.ransac_iterations < 96
            or self.ransac_threshold_m > 0.008
            or self.min_ransac_inlier_fraction < 0.40
            or self.icp_iterations < 4
            or self.icp_huber_delta_m > 0.004
            or self.max_fit_rmse_m > 0.004
            or self.max_translation_step_m > 0.050
            or self.max_rotation_step_rad > math.radians(15.0)
            or self.max_covariance_condition > 1.0e9
            or self.target_radius_m < 0.45
            or self.max_frame_gap_s > 0.25
        ):
            raise ValueError(
                "patient pose configuration exceeds the production safety envelope"
            )


class MetricPatientPoseProvider:
    """Fixed-memory, non-learning RGB-D patient-frame pose provider."""

    def __init__(
        self,
        intrinsics: Intrinsics,
        *,
        camera_id: str,
        camera_profile_sha256: str,
        alignment_sha256: str,
        patient_pose_release_id: str,
        patient_pose_sha256: str,
        depth_noise_profile: ActivatedDepthNoiseProfile,
        back_mask_release_id: str,
        back_mask_sha256: str,
        config: Optional[PatientPoseProviderConfig] = None,
        allow_unbound_test_input: bool = False,
    ) -> None:
        if not isinstance(intrinsics, Intrinsics):
            raise TypeError("intrinsics must be an Intrinsics instance")
        if (
            intrinsics.width <= 0
            or intrinsics.height <= 0
            or intrinsics.fx <= 0.0
            or intrinsics.fy <= 0.0
            or intrinsics.depth_scale <= 0.0
        ):
            raise ValueError("camera intrinsics are invalid")
        if not isinstance(camera_id, str) or not camera_id:
            raise ValueError("patient pose camera identity is invalid")
        for release_id, digest, label in (
            (patient_pose_release_id, patient_pose_sha256, "patient pose"),
            (back_mask_release_id, back_mask_sha256, "back mask"),
        ):
            if (
                not isinstance(release_id, str)
                or _RELEASE_ID.fullmatch(release_id) is None
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
            ):
                raise ValueError(f"patient pose {label} identity is invalid")
        for digest, label in (
            (camera_profile_sha256, "camera profile"),
            (alignment_sha256, "alignment"),
        ):
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(f"patient pose {label} identity is invalid")
        self.intrinsics = intrinsics
        self.camera_id = camera_id
        self.camera_profile_sha256 = camera_profile_sha256
        self.alignment_sha256 = alignment_sha256
        self.patient_pose_release_id = patient_pose_release_id
        self.patient_pose_sha256 = patient_pose_sha256
        if not isinstance(depth_noise_profile, ActivatedDepthNoiseProfile):
            raise TypeError("an activated depth-noise profile is required")
        depth_noise_profile.validate_for_camera(
            camera_id=camera_id,
            camera_profile_sha256=camera_profile_sha256,
            alignment_sha256=alignment_sha256,
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            depth_scale_mm=intrinsics.depth_scale,
        )
        self.depth_noise_profile = depth_noise_profile
        self.depth_noise_model_sha256 = depth_noise_profile.semantic_sha256
        self.back_mask_release_id = back_mask_release_id
        self.back_mask_sha256 = back_mask_sha256
        self.config = config or PatientPoseProviderConfig()
        if type(allow_unbound_test_input) is not bool:
            raise TypeError("allow_unbound_test_input must be a bool")
        self.allow_unbound_test_input = allow_unbound_test_input
        self._rng = np.random.RandomState(20260717)
        self.reset()

    def reset(self) -> None:
        """Explicitly release the old patient frame before a new episode."""

        self._initialized = False
        self._reset_required = False
        # A caller-visible patient/session label is not a coordinate-frame
        # identity.  Generate a fresh nonce on every reset so an occlusion
        # runtime can never reuse anchors from an earlier patient frame merely
        # because the same external episode label was supplied again.
        self._pose_epoch_id = secrets.token_hex(16)
        self._external_episode_id: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._last_seen_frame_index: Optional[int] = None
        self._last_seen_t_s: Optional[float] = None
        self._last_valid_t_s: Optional[float] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._pixels = np.empty((0, 2), dtype=np.float64)
        self._canonical_features = np.empty((0, 3), dtype=np.float64)
        self._canonical_feature_covariances_m2 = np.empty((0, 3, 3), dtype=np.float64)
        self._reference_surfels = np.empty((0, 3), dtype=np.float64)
        self._rotation = np.eye(3, dtype=np.float64)
        self._translation = np.zeros(3, dtype=np.float64)
        self.last_inlier_count = 0
        self.last_robot_rejected_count = 0

    @property
    def reference_feature_count(self) -> int:
        return int(len(self._canonical_features))

    @property
    def reference_surfel_count(self) -> int:
        return int(len(self._reference_surfels))

    @property
    def reference_memory_bytes(self) -> int:
        return int(
            self._pixels.nbytes
            + self._canonical_features.nbytes
            + self._canonical_feature_covariances_m2.nbytes
            + self._reference_surfels.nbytes
        )

    def _invalid(
        self, *, frame_index: int, camera_host_t_s: float, reason: str,
    ) -> PatientPosePacket:
        return PatientPosePacket(
            frame_index=int(frame_index),
            camera_host_t_s=float(camera_host_t_s),
            camera_id=self.camera_id,
            camera_profile_sha256=self.camera_profile_sha256,
            alignment_sha256=self.alignment_sha256,
            patient_pose_release_id=self.patient_pose_release_id,
            patient_pose_sha256=self.patient_pose_sha256,
            depth_noise_model_sha256=self.depth_noise_model_sha256,
            back_mask_release_id=self.back_mask_release_id,
            back_mask_sha256=self.back_mask_sha256,
            rotation_camera_from_patient=np.eye(3, dtype=np.float64),
            translation_camera_from_patient_m=np.zeros(3, dtype=np.float64),
            covariance_se3=np.eye(6, dtype=np.float64),
            support_mask=np.zeros(10, dtype=np.bool_),
            geometry_accepted=False,
            fit_rmse_m=float("inf"),
            episode_id=self._episode_id or "unbound",
            source=_SOURCE,
            scale_audit=1.0,
            scale_tolerance=self.config.scale_tolerance,
            valid=False,
            reason=str(reason),
            covariance_tangent_order=("rotation_patient_rad_then_translation_camera_m"),
            covariance_perturbation="right",
            covariance_transform="patient_to_camera",
        )

    def _valid_packet(
        self,
        *,
        frame_index: int,
        camera_host_t_s: float,
        rotation: np.ndarray,
        translation: np.ndarray,
        covariance: np.ndarray,
        support_mask: np.ndarray,
        fit_rmse_m: float,
        scale_audit: float,
    ) -> PatientPosePacket:
        return PatientPosePacket(
            frame_index=int(frame_index),
            camera_host_t_s=float(camera_host_t_s),
            camera_id=self.camera_id,
            camera_profile_sha256=self.camera_profile_sha256,
            alignment_sha256=self.alignment_sha256,
            patient_pose_release_id=self.patient_pose_release_id,
            patient_pose_sha256=self.patient_pose_sha256,
            depth_noise_model_sha256=self.depth_noise_model_sha256,
            back_mask_release_id=self.back_mask_release_id,
            back_mask_sha256=self.back_mask_sha256,
            rotation_camera_from_patient=rotation,
            translation_camera_from_patient_m=translation,
            covariance_se3=covariance,
            support_mask=np.asarray(support_mask, dtype=np.bool_),
            geometry_accepted=True,
            fit_rmse_m=float(fit_rmse_m),
            episode_id=str(self._episode_id),
            source=_SOURCE,
            scale_audit=float(scale_audit),
            scale_tolerance=self.config.scale_tolerance,
            valid=True,
            reason="OK",
            covariance_tangent_order=("rotation_patient_rad_then_translation_camera_m"),
            covariance_perturbation="right",
            covariance_transform="patient_to_camera",
        )

    def _frame_arrays(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        back_mask: np.ndarray,
        robot_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        height, width = self.intrinsics.height, self.intrinsics.width
        rgb_array = np.asarray(rgb)
        depth_array = np.asarray(depth)
        back_array = np.asarray(back_mask)
        robot_array = np.asarray(robot_mask)
        if (
            rgb_array.shape not in ((height, width), (height, width, 3))
            or rgb_array.dtype != np.uint8
            or depth_array.shape != (height, width)
            or depth_array.dtype not in (np.uint16, np.float32, np.float64)
            or back_array.shape != (height, width)
            or back_array.dtype != np.bool_
            or robot_array.shape != (height, width)
            or robot_array.dtype != np.bool_
        ):
            raise ValueError(
                "aligned RGB-D/mask arrays do not match the camera profile"
            )
        if not np.isfinite(depth_array).all():
            raise ValueError("aligned depth contains non-finite values")
        if rgb_array.ndim == 3:
            gray = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.array(rgb_array, copy=True)
        depth_m = (
            depth_array.astype(np.float64) * float(self.intrinsics.depth_scale) / 1000.0
        )
        return gray, depth_m, back_array, robot_array

    def _safe_mask(
        self, depth_m: np.ndarray, back_mask: np.ndarray, robot_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        back_count = int(np.count_nonzero(back_mask))
        robot_fraction = (
            float(np.count_nonzero(back_mask & robot_mask)) / float(back_count)
            if back_count
            else 0.0
        )
        if self.config.robot_mask_dilation_px > 0:
            size = 2 * self.config.robot_mask_dilation_px + 1
            kernel = np.ones((size, size), dtype=np.uint8)
            expanded_robot = cv2.dilate(
                robot_mask.astype(np.uint8), kernel, iterations=1
            ).astype(np.bool_)
        else:
            expanded_robot = robot_mask
        valid_depth = (
            np.isfinite(depth_m)
            & (depth_m >= self.config.min_depth_m)
            & (depth_m <= self.config.max_depth_m)
            & (depth_m >= self.depth_noise_profile.calibrated_min_depth_m)
            & (depth_m <= self.depth_noise_profile.calibrated_max_depth_m)
        )
        safe = back_mask & ~expanded_robot & valid_depth
        return safe, robot_fraction

    def _unproject(
        self, pixels: np.ndarray, depth_m: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        rounded = np.rint(pixels).astype(np.int64)
        inside = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < self.intrinsics.width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < self.intrinsics.height)
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] <= self.intrinsics.width - 1)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] <= self.intrinsics.height - 1)
        )
        depth_values = np.full(len(pixels), np.nan, dtype=np.float64)
        depth_values[inside] = depth_m[rounded[inside, 1], rounded[inside, 0]]
        valid = (
            inside
            & np.isfinite(depth_values)
            & (depth_values >= self.config.min_depth_m)
            & (depth_values <= self.config.max_depth_m)
            & (depth_values >= self.depth_noise_profile.calibrated_min_depth_m)
            & (depth_values <= self.depth_noise_profile.calibrated_max_depth_m)
        )
        points = np.full((len(pixels), 3), np.nan, dtype=np.float64)
        if np.any(valid):
            points[valid] = self.intrinsics.unproject(
                pixels[valid], depth_values[valid]
            )
        return points, valid, depth_values

    def _detect_features(
        self,
        gray: np.ndarray,
        depth_m: np.ndarray,
        safe_mask: np.ndarray,
        *,
        maximum: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        maximum = self.config.max_features if maximum is None else int(maximum)
        if maximum <= 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        feature_mask = safe_mask.astype(np.uint8) * 255
        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=maximum,
            qualityLevel=self.config.feature_quality,
            minDistance=float(self.config.feature_min_distance_px),
            mask=feature_mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if features is None:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        pixels = features.reshape(-1, 2).astype(np.float64)
        points, depth_valid, depth_values = self._unproject(pixels, depth_m)
        rounded = np.rint(pixels).astype(np.int64)
        mask_valid = safe_mask[rounded[:, 1], rounded[:, 0]]
        keep = depth_valid & mask_valid
        kept_pixels = pixels[keep]
        kept_depth = depth_values[keep]
        if len(kept_pixels) == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        covariances = self.depth_noise_profile.measurement_covariances_m2(
            kept_pixels, kept_depth
        )
        return kept_pixels, points[keep], covariances, kept_depth

    def _sample_surfels(self, depth_m: np.ndarray, safe_mask: np.ndarray) -> np.ndarray:
        rows, columns = np.nonzero(safe_mask)
        if len(rows) == 0:
            return np.empty((0, 3), dtype=np.float64)
        if len(rows) > self.config.max_surfels:
            indices = np.linspace(
                0, len(rows) - 1, self.config.max_surfels, dtype=np.int64
            )
            rows, columns = rows[indices], columns[indices]
        pixels = np.stack([columns.astype(np.float64), rows.astype(np.float64)], axis=1)
        points = self.intrinsics.unproject(pixels, depth_m[rows, columns])
        return np.asarray(points, dtype=np.float64)

    def _geometry_accepted(self, points: np.ndarray) -> bool:
        points = np.asarray(points, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) < self.config.min_math_support
            or not np.isfinite(points).all()
        ):
            return False
        centered = points - np.mean(points, axis=0)
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(points))
        return bool(
            np.isfinite(eigenvalues).all()
            and float(eigenvalues[-2]) >= self.config.min_second_spatial_eigenvalue_m2
        )

    def _support_mask(self, pixels: np.ndarray) -> np.ndarray:
        """Encode 2x5 image-region coverage, never learned anatomical support.

        ``PatientPosePacket`` has a fixed ten-bit support field.  Here those
        bits are deliberately spatial evidence bins, not the ten auscultation
        point predictions, preserving the provider's independence.
        """

        support = np.zeros(10, dtype=np.bool_)
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        for x, y in pixels:
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            column = min(4, max(0, int(5.0 * x / self.intrinsics.width)))
            row = min(1, max(0, int(2.0 * y / self.intrinsics.height)))
            support[row * 5 + column] = True
        return support

    def _ransac(
        self, source: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = len(source)
        best_mask = np.zeros(count, dtype=np.bool_)
        best_rotation = np.eye(3, dtype=np.float64)
        best_translation = np.zeros(3, dtype=np.float64)
        for _ in range(self.config.ransac_iterations):
            sample = self._rng.choice(count, 3, replace=False)
            try:
                rotation, translation = _rigid_align(source[sample], target[sample])
            except (ValueError, np.linalg.LinAlgError):
                continue
            residual = np.linalg.norm(
                (rotation @ source.T).T + translation - target, axis=1
            )
            inliers = residual <= self.config.ransac_threshold_m
            if int(np.count_nonzero(inliers)) > int(np.count_nonzero(best_mask)):
                best_mask = inliers
                best_rotation = rotation
                best_translation = translation
        if int(np.count_nonzero(best_mask)) >= self.config.min_math_support:
            best_rotation, best_translation = _rigid_align(
                source[best_mask], target[best_mask]
            )
        return best_rotation, best_translation, best_mask

    def _paired_icp_refine(
        self,
        source: np.ndarray,
        target: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bounded robust point-to-point ICP on feature-established pairs."""

        rotation = np.asarray(rotation, dtype=np.float64).copy()
        translation = np.asarray(translation, dtype=np.float64).copy()
        residual = np.full(len(source), np.inf, dtype=np.float64)
        for _ in range(self.config.icp_iterations):
            residual = np.linalg.norm(
                (rotation @ source.T).T + translation - target, axis=1
            )
            weights = np.ones(len(source), dtype=np.float64)
            large = residual > self.config.icp_huber_delta_m
            weights[large] = self.config.icp_huber_delta_m / np.maximum(
                residual[large], 1.0e-12
            )
            rotation, translation = _rigid_align(source, target, weights)
        residual = np.linalg.norm(
            (rotation @ source.T).T + translation - target, axis=1
        )
        return rotation, translation, residual

    def _covariance(
        self,
        source: np.ndarray,
        rotation: np.ndarray,
        fit_rmse_m: float,
        *,
        reference_covariances_m2: np.ndarray,
        current_covariances_m2: np.ndarray,
        current_depth_m: np.ndarray,
    ) -> Optional[np.ndarray]:
        source = np.asarray(source, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        reference_covariances = np.asarray(reference_covariances_m2, dtype=np.float64)
        current_covariances = np.asarray(current_covariances_m2, dtype=np.float64)
        current_depth = np.asarray(current_depth_m, dtype=np.float64).reshape(-1)
        count = len(source)
        if (
            source.ndim != 2
            or source.shape[1:] != (3,)
            or count < self.config.min_math_support
            or rotation.shape != (3, 3)
            or reference_covariances.shape != (count, 3, 3)
            or current_covariances.shape != (count, 3, 3)
            or current_depth.shape != (count,)
            or not np.isfinite(source).all()
            or np.any(source[:, 2] < self.depth_noise_profile.calibrated_min_depth_m)
            or np.any(source[:, 2] > self.depth_noise_profile.calibrated_max_depth_m)
            or not np.isfinite(rotation).all()
            or not np.isfinite(reference_covariances).all()
            or not np.isfinite(current_covariances).all()
            or not np.isfinite(current_depth).all()
            or np.any(current_depth < self.depth_noise_profile.calibrated_min_depth_m)
            or np.any(current_depth > self.depth_noise_profile.calibrated_max_depth_m)
            or not math.isfinite(float(fit_rmse_m))
            or float(fit_rmse_m) < 0.0
        ):
            return None
        hessian = np.zeros((6, 6), dtype=np.float64)
        fit_sigma_m = (
            float(fit_rmse_m)
            * self.depth_noise_profile.fit_residual_scale
            * self.depth_noise_profile.uncertainty_expansion_factor
        )
        fit_measurement_covariance = np.eye(3, dtype=np.float64) * (
            fit_sigma_m * fit_sigma_m
        )
        for point, reference_covariance, current_covariance in zip(
            source, reference_covariances, current_covariances
        ):
            for measurement_covariance in (
                reference_covariance,
                current_covariance,
            ):
                symmetric = 0.5 * (measurement_covariance + measurement_covariance.T)
                measurement_eigenvalues = np.linalg.eigvalsh(symmetric)
                if (
                    not np.isfinite(measurement_eigenvalues).all()
                    or float(measurement_eigenvalues[0]) <= 0.0
                ):
                    return None
            pair_covariance = (
                rotation @ reference_covariance @ rotation.T
                + current_covariance
                + fit_measurement_covariance
            )
            pair_covariance = 0.5 * (pair_covariance + pair_covariance.T)
            pair_eigenvalues = np.linalg.eigvalsh(pair_covariance)
            if (
                not np.isfinite(pair_eigenvalues).all()
                or float(pair_eigenvalues[0]) <= 0.0
            ):
                return None
            jacobian = np.hstack(
                (-rotation @ _skew(point), np.eye(3, dtype=np.float64))
            )
            try:
                hessian += jacobian.T @ np.linalg.solve(pair_covariance, jacobian)
            except np.linalg.LinAlgError:
                return None
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
        if (
            not np.isfinite(eigenvalues).all()
            or float(eigenvalues[0]) <= 1.0e-12
            or float(eigenvalues[-1] / eigenvalues[0])
            > self.config.max_covariance_condition
        ):
            return None
        try:
            covariance = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            return None
        maximum_depth = float(max(np.max(source[:, 2]), np.max(current_depth)))
        try:
            covariance += self.depth_noise_profile.common_alignment_pose_covariance_se3(
                maximum_depth_m=maximum_depth,
                target_radius_m=self.config.target_radius_m,
            )
            covariance += self.depth_noise_profile.fit_residual_pose_covariance_se3(
                fit_rmse_m=float(fit_rmse_m),
                target_radius_m=self.config.target_radius_m,
            )
        except ValueError:
            return None
        covariance = 0.5 * (covariance + covariance.T)
        covariance_eigenvalues = np.linalg.eigvalsh(covariance)
        if (
            not np.isfinite(covariance).all()
            or not np.isfinite(covariance_eigenvalues).all()
            or float(covariance_eigenvalues[0]) <= 0.0
        ):
            return None
        return covariance

    def _maximum_target_bound(
        self,
        covariance: np.ndarray,
        rotation: np.ndarray,
        reference_surfels: Optional[np.ndarray] = None,
    ) -> float:
        reference = (
            self._reference_surfels
            if reference_surfels is None
            else np.asarray(reference_surfels, dtype=np.float64)
        )
        points = reference
        if len(points) == 0:
            return float("inf")
        if len(points) > 64:
            indices = np.linspace(0, len(points) - 1, 64, dtype=np.int64)
            points = points[indices]
        directions = np.array(
            [
                [self.config.target_radius_m, 0.0, 0.0],
                [-self.config.target_radius_m, 0.0, 0.0],
                [0.0, self.config.target_radius_m, 0.0],
                [0.0, -self.config.target_radius_m, 0.0],
            ],
            dtype=np.float64,
        )
        centroid = np.mean(reference, axis=0)
        audit_points = np.vstack((points, centroid + directions))
        maximum = 0.0
        for point in audit_points:
            jacobian = np.hstack(
                (-rotation @ _skew(point), np.eye(3, dtype=np.float64))
            )
            propagated = jacobian @ covariance @ jacobian.T
            eigenvalues = np.linalg.eigvalsh(0.5 * (propagated + propagated.T))
            if not np.isfinite(eigenvalues).all() or eigenvalues[-1] < 0.0:
                return float("inf")
            maximum = max(
                maximum, math.sqrt(_CHI2_3D_95 * max(float(eigenvalues[-1]), 0.0)),
            )
        return float(maximum)

    def _initialize(
        self,
        *,
        frame_index: int,
        camera_host_t_s: float,
        gray: np.ndarray,
        depth_m: np.ndarray,
        safe_mask: np.ndarray,
    ) -> PatientPosePacket:
        surfels = self._sample_surfels(depth_m, safe_mask)
        if not self._geometry_accepted(surfels):
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="DEGENERATE_GEOMETRY",
            )
        (pixels, points, point_covariances, point_depths,) = self._detect_features(
            gray, depth_m, safe_mask
        )
        if len(points) < self.config.min_math_support:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="INSUFFICIENT_FEATURE_SUPPORT",
            )
        if not self._geometry_accepted(points):
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="DEGENERATE_GEOMETRY",
            )
        candidate_surfels = surfels[: self.config.max_surfels].copy()
        candidate_pixels = pixels[: self.config.max_features].copy()
        candidate_features = points[: self.config.max_features].copy()
        candidate_covariances = point_covariances[: self.config.max_features].copy()
        candidate_depths = point_depths[: self.config.max_features].copy()
        candidate_memory_bytes = int(
            candidate_surfels.nbytes
            + candidate_pixels.nbytes
            + candidate_features.nbytes
            + candidate_covariances.nbytes
        )
        if candidate_memory_bytes > self.config.max_reference_bytes:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="REFERENCE_MEMORY_BOUND_EXCEEDED",
            )
        covariance = self._covariance(
            candidate_features,
            np.eye(3),
            0.0,
            reference_covariances_m2=candidate_covariances,
            current_covariances_m2=candidate_covariances,
            current_depth_m=candidate_depths,
        )
        if covariance is None:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="DEGENERATE_COVARIANCE",
            )
        bound = self._maximum_target_bound(covariance, np.eye(3), candidate_surfels)
        if not math.isfinite(bound) or bound > self.config.max_target_position_bound_m:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason=f"POSE_BOUND_EXCEEDED_{bound:.6f}",
            )
        support = self._support_mask(candidate_pixels)
        if int(np.count_nonzero(support)) < self.config.min_math_support:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="INSUFFICIENT_SPATIAL_SUPPORT",
            )
        self._reference_surfels = candidate_surfels
        self._pixels = candidate_pixels
        self._canonical_features = candidate_features
        self._canonical_feature_covariances_m2 = candidate_covariances
        self._prev_gray = np.array(gray, copy=True)
        self._rotation = np.eye(3, dtype=np.float64)
        self._translation = np.zeros(3, dtype=np.float64)
        self._last_valid_t_s = float(camera_host_t_s)
        self._initialized = True
        self.last_inlier_count = len(self._canonical_features)
        return self._valid_packet(
            frame_index=frame_index,
            camera_host_t_s=camera_host_t_s,
            rotation=self._rotation,
            translation=self._translation,
            covariance=covariance,
            support_mask=support,
            fit_rmse_m=0.0,
            scale_audit=1.0,
        )

    def _track(
        self,
        *,
        frame_index: int,
        camera_host_t_s: float,
        gray: np.ndarray,
        depth_m: np.ndarray,
        safe_mask: np.ndarray,
        robot_mask: np.ndarray,
    ) -> PatientPosePacket:
        if (
            self._prev_gray is None
            or len(self._pixels) < self.config.min_math_support
            or len(self._canonical_features) != len(self._pixels)
            or len(self._canonical_feature_covariances_m2) != len(self._pixels)
            or self._last_valid_t_s is None
        ):
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="NO_REFERENCE",
            )
        if camera_host_t_s - self._last_valid_t_s > self.config.max_frame_gap_s:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="FRAME_GAP_EXCEEDED",
            )
        previous = self._pixels.astype(np.float32).reshape(-1, 1, 2)
        lk = {
            "winSize": (self.config.lk_window_px, self.config.lk_window_px),
            "maxLevel": self.config.lk_max_level,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01,),
        }
        current, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, previous, None, **lk
        )
        if current is None or forward_status is None:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="INSUFFICIENT_FEATURE_SUPPORT",
            )
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, current, None, **lk
        )
        if backward is None or backward_status is None:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="INSUFFICIENT_FEATURE_SUPPORT",
            )
        current_pixels = current.reshape(-1, 2).astype(np.float64)
        forward_backward = np.linalg.norm(
            previous.reshape(-1, 2) - backward.reshape(-1, 2), axis=1
        )
        rounded = np.rint(current_pixels).astype(np.int64)
        inside = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < self.intrinsics.width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < self.intrinsics.height)
            & (current_pixels[:, 0] >= 0.0)
            & (current_pixels[:, 0] <= self.intrinsics.width - 1)
            & (current_pixels[:, 1] >= 0.0)
            & (current_pixels[:, 1] <= self.intrinsics.height - 1)
        )
        optical_ok = (
            (forward_status.reshape(-1) == 1)
            & (backward_status.reshape(-1) == 1)
            & np.isfinite(current_pixels).all(axis=1)
            & np.isfinite(forward_backward)
            & (forward_backward <= self.config.max_forward_backward_error_px)
            & inside
        )
        safe_at_pixel = np.zeros(len(current_pixels), dtype=np.bool_)
        robot_at_pixel = np.zeros(len(current_pixels), dtype=np.bool_)
        safe_at_pixel[inside] = safe_mask[rounded[inside, 1], rounded[inside, 0]]
        robot_at_pixel[inside] = robot_mask[rounded[inside, 1], rounded[inside, 0]]
        previous_rounded = np.rint(previous.reshape(-1, 2)).astype(np.int64)
        previous_inside = (
            (previous_rounded[:, 0] >= 0)
            & (previous_rounded[:, 0] < self.intrinsics.width)
            & (previous_rounded[:, 1] >= 0)
            & (previous_rounded[:, 1] < self.intrinsics.height)
        )
        robot_at_previous = np.zeros(len(current_pixels), dtype=np.bool_)
        robot_at_previous[previous_inside] = robot_mask[
            previous_rounded[previous_inside, 1], previous_rounded[previous_inside, 0],
        ]
        self.last_robot_rejected_count = int(
            np.count_nonzero(robot_at_pixel | robot_at_previous)
        )
        candidate = optical_ok & safe_at_pixel & ~robot_at_pixel & ~robot_at_previous
        observed, depth_valid, observed_depth = self._unproject(current_pixels, depth_m)
        candidate &= depth_valid
        if int(np.count_nonzero(candidate)) < self.config.min_math_support:
            self.last_inlier_count = 0
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="INSUFFICIENT_FEATURE_SUPPORT",
            )
        source = self._canonical_features[candidate]
        reference_covariances = self._canonical_feature_covariances_m2[candidate]
        target = observed[candidate]
        target_depth = observed_depth[candidate]
        accepted_pixels = current_pixels[candidate]
        try:
            current_covariances = self.depth_noise_profile.measurement_covariances_m2(
                accepted_pixels, target_depth
            )
        except ValueError:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="UNCALIBRATED_DEPTH_UNCERTAINTY",
            )
        if not self._geometry_accepted(source) or not self._geometry_accepted(target):
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="DEGENERATE_GEOMETRY",
            )
        scale = _similarity_scale(source, target)
        if not math.isfinite(scale) or abs(scale - 1.0) > self.config.scale_tolerance:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason=f"SCALE_DRIFT_{scale:.6f}",
            )
        try:
            rotation, translation, ransac_inliers = self._ransac(source, target)
            inlier_count = int(np.count_nonzero(ransac_inliers))
            self.last_inlier_count = inlier_count
            if (
                inlier_count < self.config.min_math_support
                or inlier_count / len(source) < self.config.min_ransac_inlier_fraction
            ):
                return self._invalid(
                    frame_index=frame_index,
                    camera_host_t_s=camera_host_t_s,
                    reason="RANSAC_SUPPORT_REJECTED",
                )
            rotation, translation, residual = self._paired_icp_refine(
                source, target, rotation, translation
            )
        except (ValueError, np.linalg.LinAlgError):
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="REGISTRATION_FAILED",
            )
        capped_residual = np.minimum(residual, 3.0 * self.config.max_fit_rmse_m)
        fit_rmse = float(math.sqrt(np.mean(capped_residual ** 2)))
        if not math.isfinite(fit_rmse) or fit_rmse > self.config.max_fit_rmse_m:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason=f"FIT_RESIDUAL_{fit_rmse:.6f}",
            )
        relative_rotation = rotation @ self._rotation.T
        translation_step = float(np.linalg.norm(translation - self._translation))
        rotation_step = _rotation_angle(relative_rotation)
        if (
            not math.isfinite(translation_step)
            or not math.isfinite(rotation_step)
            or translation_step > self.config.max_translation_step_m
            or rotation_step > self.config.max_rotation_step_rad
        ):
            self._reset_required = True
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason=(
                    f"MOTION_JUMP_t{translation_step:.6f}" f"_r{rotation_step:.6f}"
                ),
            )
        covariance_source = source[ransac_inliers]
        if not self._geometry_accepted(covariance_source):
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="DEGENERATE_INLIER_GEOMETRY",
            )
        covariance = self._covariance(
            covariance_source,
            rotation,
            fit_rmse,
            reference_covariances_m2=reference_covariances[ransac_inliers],
            current_covariances_m2=current_covariances[ransac_inliers],
            current_depth_m=target_depth[ransac_inliers],
        )
        if covariance is None:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="DEGENERATE_COVARIANCE",
            )
        bound = self._maximum_target_bound(covariance, rotation)
        if not math.isfinite(bound) or bound > self.config.max_target_position_bound_m:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason=f"POSE_BOUND_EXCEEDED_{bound:.6f}",
            )
        support = self._support_mask(accepted_pixels[ransac_inliers])
        if int(np.count_nonzero(support)) < self.config.min_math_support:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                reason="INSUFFICIENT_SPATIAL_SUPPORT",
            )

        # Commit state only after every safety gate passes.  Thus a rejected
        # frame cannot move the patient coordinate system.
        self._pixels = accepted_pixels[: self.config.max_features].copy()
        self._canonical_features = source[: self.config.max_features].copy()
        self._canonical_feature_covariances_m2 = reference_covariances[
            : self.config.max_features
        ].copy()
        self._prev_gray = np.array(gray, copy=True)
        self._rotation = rotation.copy()
        self._translation = translation.copy()
        self._last_valid_t_s = float(camera_host_t_s)
        return self._valid_packet(
            frame_index=frame_index,
            camera_host_t_s=camera_host_t_s,
            rotation=rotation,
            translation=translation,
            covariance=covariance,
            support_mask=support,
            fit_rmse_m=fit_rmse,
            scale_audit=scale,
        )

    def _process_arrays(
        self,
        *,
        frame_index: int,
        camera_host_t_s: float,
        rgb: np.ndarray,
        aligned_depth: np.ndarray,
        back_mask: np.ndarray,
        robot_mask: np.ndarray,
        episode_id: str,
    ) -> PatientPosePacket:
        """Process arrays after their evidence boundary has been authorized."""

        if (
            type(frame_index) is not int
            or frame_index < 0
            or isinstance(camera_host_t_s, (bool, np.bool_))
            or not isinstance(camera_host_t_s, (int, float, np.integer, np.floating))
            or not math.isfinite(float(camera_host_t_s))
        ):
            raise ValueError("patient pose frame identity is invalid")
        timestamp = float(camera_host_t_s)
        if self._last_seen_frame_index is not None and (
            frame_index <= self._last_seen_frame_index
            or timestamp <= float(self._last_seen_t_s)
        ):
            self._reset_required = True
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=timestamp,
                reason="NON_MONOTONIC_FRAME_OR_TIME",
            )
        self._last_seen_frame_index = frame_index
        self._last_seen_t_s = timestamp
        if self._reset_required:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=timestamp,
                reason="RESET_REQUIRED",
            )
        if not isinstance(episode_id, str) or not episode_id:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=timestamp,
                reason="INVALID_EPISODE",
            )
        if self._external_episode_id is None:
            self._external_episode_id = episode_id
            self._episode_id = f"{episode_id}/pose-epoch-{self._pose_epoch_id}"
        elif episode_id != self._external_episode_id:
            self._reset_required = True
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=timestamp,
                reason="EPISODE_CHANGED_REQUIRES_RESET",
            )
        try:
            gray, depth_m, back, robot = self._frame_arrays(
                rgb, aligned_depth, back_mask, robot_mask
            )
        except (TypeError, ValueError):
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=timestamp,
                reason="INVALID_ALIGNED_FRAME",
            )
        back_count = int(np.count_nonzero(back))
        if back_count < self.config.min_back_pixels:
            if self._initialized:
                self._reset_required = True
            return self._invalid(
                frame_index=frame_index, camera_host_t_s=timestamp, reason="NO_BACK",
            )
        safe, robot_fraction = self._safe_mask(depth_m, back, robot)
        if robot_fraction > self.config.max_robot_back_fraction:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=timestamp,
                reason=f"ROBOT_MASK_POLLUTION_{robot_fraction:.6f}",
            )
        if int(np.count_nonzero(safe)) < self.config.min_back_pixels:
            return self._invalid(
                frame_index=frame_index,
                camera_host_t_s=timestamp,
                reason="INSUFFICIENT_UNOCCLUDED_BACK",
            )
        if not self._initialized:
            return self._initialize(
                frame_index=frame_index,
                camera_host_t_s=timestamp,
                gray=gray,
                depth_m=depth_m,
                safe_mask=safe,
            )
        return self._track(
            frame_index=frame_index,
            camera_host_t_s=timestamp,
            gray=gray,
            depth_m=depth_m,
            safe_mask=safe,
            robot_mask=robot,
        )

    def process_evidence(
        self, evidence: PatientFrameEvidence, episode_id: str,
    ) -> PatientPosePacket:
        """Process the only default production input: one bound frame evidence."""

        if not isinstance(evidence, PatientFrameEvidence):
            raise PatientFrameEvidenceError("patient frame evidence is required")
        # Dataclasses can be forged with ``object.__new__`` or mutated through
        # ``object.__setattr__`` despite their frozen API.  Re-run the complete
        # bundle and back-packet verification at the consumption boundary.
        evidence.__post_init__()
        self.depth_noise_profile.__post_init__()
        camera = evidence.frame_bundle.camera_frame
        back = evidence.back_mask_packet
        camera.__post_init__()
        back.camera_frame.__post_init__()
        expected = (
            (camera.camera_id, self.camera_id, "camera identity"),
            (
                camera.camera_profile_sha256,
                self.camera_profile_sha256,
                "camera profile",
            ),
            (camera.alignment_sha256, self.alignment_sha256, "alignment"),
            (
                back.back_mask_release_id,
                self.back_mask_release_id,
                "back-mask release",
            ),
            (back.back_mask_model_sha256, self.back_mask_sha256, "back-mask model",),
            (
                self.depth_noise_profile.semantic_sha256,
                self.depth_noise_model_sha256,
                "depth-noise profile",
            ),
        )
        for actual, activated, label in expected:
            if actual != activated:
                raise ValueError(f"patient evidence {label} does not match activation")
        if float(camera.depth_scale).hex() != float(self.intrinsics.depth_scale).hex():
            raise ValueError("patient evidence depth scale does not match activation")
        self.depth_noise_profile.validate_for_camera(
            camera_id=camera.camera_id,
            camera_profile_sha256=camera.camera_profile_sha256,
            alignment_sha256=camera.alignment_sha256,
            width=camera.frame.shape[1],
            height=camera.frame.shape[0],
            fx=self.intrinsics.fx,
            fy=self.intrinsics.fy,
            cx=self.intrinsics.cx,
            cy=self.intrinsics.cy,
            depth_scale_mm=camera.depth_scale,
        )
        return self._process_arrays(
            frame_index=evidence.index,
            camera_host_t_s=evidence.t,
            rgb=evidence.rgb,
            aligned_depth=evidence.aligned_depth,
            back_mask=evidence.back_mask,
            robot_mask=evidence.robot_mask,
            episode_id=episode_id,
        )

    def process(
        self,
        *,
        frame_index: int,
        camera_host_t_s: float,
        rgb: np.ndarray,
        aligned_depth: np.ndarray,
        back_mask: np.ndarray,
        robot_mask: np.ndarray,
        episode_id: str,
    ) -> PatientPosePacket:
        """Explicit test/offline-only raw-array entrypoint."""

        if not self.allow_unbound_test_input:
            raise PermissionError("unbound raw patient-pose input is disabled")
        return self._process_arrays(
            frame_index=frame_index,
            camera_host_t_s=camera_host_t_s,
            rgb=rgb,
            aligned_depth=aligned_depth,
            back_mask=back_mask,
            robot_mask=robot_mask,
            episode_id=episode_id,
        )


__all__ = [
    "MetricPatientPoseProvider",
    "PatientPoseProviderConfig",
]
