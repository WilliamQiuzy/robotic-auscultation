"""Fail-closed runtime contract for robot-occluded auscultation targets.

This module is deliberately pure: it does not read a camera, robot, model, or
calibration file.  Providers must first bind those inputs into immutable packets.
The runtime then decides only whether each 3-D point is safe to publish.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import re
from typing import Optional, Sequence, Tuple

import numpy as np


DIRECT = "DIRECT"
PATIENT_FRAME_LOCKED = "PATIENT_FRAME_LOCKED"
BOUNDED_FREEZE = "BOUNDED_FREEZE"
REACQUIRING = "REACQUIRING"
LOST = "LOST"

LEGACY_HANDEYE_CAPABILITY_SCHEMA = "wam_openarm_handeye_v1"
PROVISIONAL_HANDEYE_CAPABILITY_SCHEMA = (
    "wam_provisional_robot_mask_handeye_capability_v1"
)
LEGACY_HANDEYE_CALIBRATION_TIER = "QUALITY_GATE_COMPLETE"
PROVISIONAL_HANDEYE_CALIBRATION_TIER = (
    "USER_ACCEPTED_BELOW_GATE_5MM"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[a-z][a-z0-9_]*-[0-9a-f]{20}$")
_CHI2_3D_95 = 7.814727903251179
_NS_PER_SECOND = 1_000_000_000
_COVARIANCE_TANGENT_ORDER = (
    "rotation_patient_rad_then_translation_camera_m"
)
_COVARIANCE_PERTURBATION = "right"
_COVARIANCE_TRANSFORM = "patient_to_camera"
_PATIENT_POSE_SOURCE = "rgbd_feature_ransac_plus_icp"
_MAX_PATIENT_POSE_RMSE_M = 0.004
_MIN_PATIENT_POSE_SUPPORT = 4


def _readonly_array(value, *, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_scalar(value) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    ).hexdigest()


def _skew(point: np.ndarray) -> np.ndarray:
    x, y, z = (float(value) for value in point)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _elapsed_seconds(current: float, previous: float) -> float:
    """Subtract authored decimal timestamps without binary boundary drift."""

    return float(Decimal(str(float(current))) - Decimal(str(float(previous))))


def _projection_payload(
    *,
    camera_id: str,
    camera_profile_sha256: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_projection_depth_m: float,
) -> dict:
    return {
        "camera_id": camera_id,
        "camera_profile_sha256": camera_profile_sha256,
        "cx": cx,
        "cy": cy,
        "fx": fx,
        "fy": fy,
        "height": height,
        "min_projection_depth_m": min_projection_depth_m,
        "projection": "rectified_rgb_aligned_depth_pinhole",
        "schema": "wam_camera_projection_v1",
        "width": width,
    }


def _semantic_sha256(value: object) -> str:
    try:
        canonical = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "camera projection cannot be encoded as canonical JSON"
        ) from exc
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class ActivatedCameraProjection:
    """Semantic-hashed pinhole model for the exact aligned RGB-D profile."""

    camera_id: str
    camera_profile_sha256: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    min_projection_depth_m: float
    camera_projection_sha256: str

    @classmethod
    def create(
        cls,
        *,
        camera_id: str,
        camera_profile_sha256: str,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        min_projection_depth_m: float,
    ) -> "ActivatedCameraProjection":
        payload = _projection_payload(
            camera_id=camera_id,
            camera_profile_sha256=camera_profile_sha256,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            min_projection_depth_m=min_projection_depth_m,
        )
        return cls(
            camera_id=camera_id,
            camera_profile_sha256=camera_profile_sha256,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            min_projection_depth_m=min_projection_depth_m,
            camera_projection_sha256=_semantic_sha256(payload),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera projection identity is missing")
        if _SHA256.fullmatch(str(self.camera_profile_sha256)) is None:
            raise ValueError("camera projection profile SHA-256 is invalid")
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or not 16 <= self.width <= 8192
            or not 16 <= self.height <= 8192
        ):
            raise ValueError("camera projection dimensions are invalid")
        intrinsics = (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.min_projection_depth_m,
        )
        if (
            not all(_finite_scalar(value) for value in intrinsics)
            or float(self.fx) <= 0.0
            or float(self.fy) <= 0.0
            or not 0.0 <= float(self.cx) < self.width
            or not 0.0 <= float(self.cy) < self.height
            or not 0.005 <= float(self.min_projection_depth_m) <= 0.5
        ):
            raise ValueError("camera projection intrinsics are invalid")
        digest = _semantic_sha256(
            _projection_payload(
                camera_id=self.camera_id,
                camera_profile_sha256=self.camera_profile_sha256,
                width=self.width,
                height=self.height,
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
                min_projection_depth_m=self.min_projection_depth_m,
            )
        )
        if self.camera_projection_sha256 != digest:
            raise ValueError(
                "camera projection SHA-256 does not match semantic content"
            )

    def validate_for_frame(
        self,
        *,
        camera_id: str,
        camera_profile_sha256: str,
        aligned_depth_shape: Tuple[int, int],
    ) -> None:
        if (
            str(camera_id) != self.camera_id
            or str(camera_profile_sha256) != self.camera_profile_sha256
            or tuple(aligned_depth_shape) != (self.height, self.width)
        ):
            raise ValueError(
                "camera projection does not match the aligned camera frame"
            )

    def project_camera_point(
        self, point_camera_m: np.ndarray
    ) -> Optional[np.ndarray]:
        try:
            point = np.asarray(point_camera_m, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if point.shape != (3,) or not np.isfinite(point).all():
            return None
        x, y, z = (float(value) for value in point)
        if z < float(self.min_projection_depth_m):
            return None
        u = float(self.fx) * x / z + float(self.cx)
        v = float(self.fy) * y / z + float(self.cy)
        if (
            not math.isfinite(u)
            or not math.isfinite(v)
            or u < 0.0
            or v < 0.0
            or u >= self.width
            or v >= self.height
        ):
            return None
        return _readonly_array((u, v), dtype=np.float64)

    def uncertainty_region_fully_visible(
        self,
        point_camera_m: np.ndarray,
        uncertainty_bound_m: float,
    ) -> bool:
        """Conservatively require the complete metric uncertainty ball in view."""

        try:
            point = np.asarray(point_camera_m, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return False
        if (
            point.shape != (3,)
            or not np.isfinite(point).all()
            or not _finite_scalar(uncertainty_bound_m)
            or float(uncertainty_bound_m) < 0.0
        ):
            return False
        radius = float(uncertainty_bound_m)
        # The axis-aligned cube contains the 3-D confidence ball.  Requiring
        # all eight cube corners to project into the image is conservative and
        # avoids displaying a centre whose credible region leaves the camera.
        for dx in (-radius, radius):
            for dy in (-radius, radius):
                for dz in (-radius, radius):
                    if self.project_camera_point(
                        point + np.array((dx, dy, dz), dtype=np.float64)
                    ) is None:
                        return False
        return True


@dataclass(frozen=True)
class ActivatedOcclusionIdentity:
    """Exact immutable artifacts authorized for one occlusion runtime."""

    camera_id: str
    camera_profile_sha256: str
    geometry_release_id: str
    geometry_sha256: str
    tool_release_id: str
    tool_sha256: str
    handeye_release_id: str
    handeye_sha256: str
    handeye_capability_schema: str
    handeye_calibration_tier: str
    handeye_policy_sha256: str
    camera_latency_sha256: str
    camera_projection_sha256: str
    occlusion_profile_sha256: str
    occlusion_runtime_profile_sha256: str
    alignment_sha256: str
    joint_mapper_sha256: str
    pair_contract_sha256: str
    controller_binary_sha256: str
    patient_pose_release_id: str
    patient_pose_sha256: str
    depth_noise_model_sha256: str
    back_mask_release_id: str
    back_mask_sha256: str
    max_camera_to_exposure_age_ns: int
    max_exposure_interval_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("activated camera identity is invalid")
        for release_id, digest, label in (
            (self.geometry_release_id, self.geometry_sha256, "geometry"),
            (self.tool_release_id, self.tool_sha256, "tool"),
            (self.handeye_release_id, self.handeye_sha256, "hand-eye"),
            (
                self.patient_pose_release_id,
                self.patient_pose_sha256,
                "patient pose",
            ),
            (
                self.back_mask_release_id,
                self.back_mask_sha256,
                "back mask",
            ),
        ):
            if (
                _RELEASE_ID.fullmatch(str(release_id)) is None
                or _SHA256.fullmatch(str(digest)) is None
            ):
                raise ValueError(f"activated {label} identity is invalid")
        for digest, label in (
            (self.camera_profile_sha256, "camera profile"),
            (self.camera_latency_sha256, "camera latency"),
            (self.camera_projection_sha256, "camera projection"),
            (self.occlusion_profile_sha256, "occlusion profile"),
            (
                self.occlusion_runtime_profile_sha256,
                "occlusion runtime profile",
            ),
            (self.alignment_sha256, "alignment"),
            (self.joint_mapper_sha256, "joint mapper"),
            (self.pair_contract_sha256, "pair contract"),
            (self.controller_binary_sha256, "controller binary"),
            (self.depth_noise_model_sha256, "depth noise model"),
            (self.handeye_policy_sha256, "hand-eye policy"),
        ):
            if _SHA256.fullmatch(str(digest)) is None:
                raise ValueError(f"activated {label} identity is invalid")
        if self.handeye_capability_schema == PROVISIONAL_HANDEYE_CAPABILITY_SCHEMA:
            valid_handeye_policy = (
                self.handeye_calibration_tier
                == PROVISIONAL_HANDEYE_CALIBRATION_TIER
                and self.handeye_release_id
                == "handeye_activation-" + self.handeye_sha256[:20]
            )
        elif self.handeye_capability_schema == LEGACY_HANDEYE_CAPABILITY_SCHEMA:
            valid_handeye_policy = (
                self.handeye_calibration_tier
                == LEGACY_HANDEYE_CALIBRATION_TIER
                and self.handeye_release_id
                == "handeye-" + self.handeye_sha256[:20]
            )
        else:
            valid_handeye_policy = False
        if not valid_handeye_policy:
            raise ValueError(
                "activated hand-eye schema/tier/policy identity is invalid"
            )
        if (
            type(self.max_camera_to_exposure_age_ns) is not int
            or type(self.max_exposure_interval_ns) is not int
            or not 0 < self.max_camera_to_exposure_age_ns <= 100_000_000
            or not 0 < self.max_exposure_interval_ns <= 40_000_000
        ):
            raise ValueError(
                "activated timing values exceed the production safety envelope"
            )


@dataclass(frozen=True)
class RobotMaskPacket:
    """One interval-swept robot/tool mask bound to one exact camera frame."""

    frame_index: int
    camera_host_t_s: float
    camera_host_monotonic_ns: int
    camera_device_t_ms: Optional[float]
    camera_id: str
    camera_profile_sha256: str
    exposure_lower_host_t_s: float
    exposure_nominal_host_t_s: float
    exposure_upper_host_t_s: float
    exposure_lower_monotonic_ns: int
    exposure_nominal_monotonic_ns: int
    exposure_upper_monotonic_ns: int
    mask: np.ndarray
    mask_sha256: str
    masked_pixel_count: int
    joint_sequence_first: int
    joint_sequence_last: int
    joint_time_first_s: float
    joint_time_last_s: float
    joint_time_first_monotonic_ns: int
    joint_time_last_monotonic_ns: int
    joint_max_gap_s: float
    joint_max_gap_ns: int
    geometry_release_id: str
    geometry_sha256: str
    tool_release_id: str
    tool_sha256: str
    handeye_release_id: str
    handeye_sha256: str
    handeye_capability_schema: str
    handeye_calibration_tier: str
    handeye_policy_sha256: str
    camera_latency_sha256: str
    camera_projection_sha256: str
    occlusion_profile_sha256: str
    alignment_sha256: str
    joint_publisher_identity: str
    status: str
    reason: str
    handeye_calibration_status: str = "COMPLETE"
    handeye_provisional: bool = False
    handeye_quality_gate_passed: Optional[bool] = None
    handeye_workspace_1sigma_max_m: Optional[float] = None
    handeye_covariance_bound_transform: str = (
        "transform_camera_from_follower_base"
    )
    handeye_covariance_scope: str = "legacy_production"
    handeye_systematic_bias_status: str = "LEGACY_NOT_APPLICABLE"
    handeye_systematic_bias_bound_m: Optional[float] = None
    handeye_autonomous_targeting_authorized: bool = True
    handeye_provisional_policy_allowance_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask", _readonly_array(self.mask))

    def validate_for_frame(
        self,
        *,
        frame_index: int,
        camera_host_t_s: float,
        camera_id: str,
        camera_profile_sha256: str,
        depth_shape: Tuple[int, int],
        activated_identity: ActivatedOcclusionIdentity,
    ) -> None:
        if not isinstance(activated_identity, ActivatedOcclusionIdentity):
            raise ValueError("activated occlusion identity is missing")
        if self.status != "OK":
            raise ValueError(f"robot mask status is not OK: {self.reason}")
        if type(self.frame_index) is not int or self.frame_index != int(frame_index):
            raise ValueError("robot mask frame index does not match the camera frame")
        if (
            not _finite_scalar(self.camera_host_t_s)
            or not _finite_scalar(camera_host_t_s)
            or float(self.camera_host_t_s) != float(camera_host_t_s)
        ):
            raise ValueError("robot mask host timestamp does not match the camera frame")
        if not _finite_scalar(self.camera_device_t_ms):
            raise ValueError("robot mask camera device timestamp is invalid")
        if (
            type(self.camera_host_monotonic_ns) is not int
            or self.camera_host_monotonic_ns <= 0
            or self.camera_host_monotonic_ns
            != int(round(float(self.camera_host_t_s) * _NS_PER_SECOND))
        ):
            raise ValueError("robot mask camera monotonic timestamp is invalid")
        if (
            not isinstance(self.camera_id, str)
            or not self.camera_id
            or self.camera_id != str(camera_id)
        ):
            raise ValueError("robot mask camera identity does not match the live camera")
        if (
            _SHA256.fullmatch(str(self.camera_profile_sha256)) is None
            or _SHA256.fullmatch(str(camera_profile_sha256)) is None
            or self.camera_profile_sha256 != str(camera_profile_sha256)
        ):
            raise ValueError("robot mask camera profile does not match the live frame")
        if (
            not isinstance(self.mask, np.ndarray)
            or self.mask.dtype != np.bool_
            or self.mask.ndim != 2
            or tuple(self.mask.shape) != tuple(depth_shape)
        ):
            raise ValueError("robot mask shape/dtype does not match aligned depth")
        if (
            _SHA256.fullmatch(str(self.mask_sha256)) is None
            or self.mask_sha256 != _mask_digest(self.mask)
        ):
            raise ValueError("robot mask SHA-256 does not match mask bytes")
        if (
            type(self.masked_pixel_count) is not int
            or self.masked_pixel_count != int(np.count_nonzero(self.mask))
        ):
            raise ValueError("robot mask masked pixel count does not match mask bytes")

        exposure = (
            self.exposure_lower_host_t_s,
            self.exposure_nominal_host_t_s,
            self.exposure_upper_host_t_s,
        )
        if (
            not all(_finite_scalar(value) for value in exposure)
            or not exposure[0] < exposure[1] < exposure[2]
            or exposure[2] > float(self.camera_host_t_s)
        ):
            raise ValueError("robot mask exposure interval is invalid")
        exposure_ns = (
            self.exposure_lower_monotonic_ns,
            self.exposure_nominal_monotonic_ns,
            self.exposure_upper_monotonic_ns,
        )
        if (
            not all(type(value) is int and value > 0 for value in exposure_ns)
            or not exposure_ns[0] < exposure_ns[1] < exposure_ns[2]
            or exposure_ns[2] > self.camera_host_monotonic_ns
            or any(
                value_ns != int(round(float(value_s) * _NS_PER_SECOND))
                for value_ns, value_s in zip(exposure_ns, exposure)
            )
            or exposure_ns[2] - exposure_ns[0]
            > activated_identity.max_exposure_interval_ns
            or self.camera_host_monotonic_ns - exposure_ns[2]
            > activated_identity.max_camera_to_exposure_age_ns
        ):
            raise ValueError("robot mask exposure monotonic interval is invalid")
        joint_values = (
            self.joint_time_first_s,
            self.joint_time_last_s,
            self.joint_max_gap_s,
        )
        if (
            type(self.joint_sequence_first) is not int
            or type(self.joint_sequence_last) is not int
            or self.joint_sequence_first < 0
            or self.joint_sequence_last < self.joint_sequence_first
            or not all(_finite_scalar(value) for value in joint_values)
            or self.joint_time_first_s > exposure[0]
            or self.joint_time_last_s < exposure[2]
            or not 0.0 < self.joint_max_gap_s <= 0.020
        ):
            raise ValueError("robot telemetry does not bracket the exposure interval")
        if (
            type(self.joint_time_first_monotonic_ns) is not int
            or type(self.joint_time_last_monotonic_ns) is not int
            or type(self.joint_max_gap_ns) is not int
            or self.joint_time_first_monotonic_ns
            != int(round(float(self.joint_time_first_s) * _NS_PER_SECOND))
            or self.joint_time_last_monotonic_ns
            != int(round(float(self.joint_time_last_s) * _NS_PER_SECOND))
            or self.joint_max_gap_ns
            != int(round(float(self.joint_max_gap_s) * _NS_PER_SECOND))
            or self.joint_time_first_monotonic_ns > exposure_ns[0]
            or self.joint_time_last_monotonic_ns < exposure_ns[2]
            or not 0 < self.joint_max_gap_ns <= 20_000_000
        ):
            raise ValueError(
                "robot telemetry monotonic interval does not bracket exposure"
            )

        releases = (
            (self.geometry_release_id, self.geometry_sha256, "geometry"),
            (self.tool_release_id, self.tool_sha256, "tool"),
            (self.handeye_release_id, self.handeye_sha256, "hand-eye"),
        )
        for release_id, digest, label in releases:
            if (
                _RELEASE_ID.fullmatch(str(release_id)) is None
                or _SHA256.fullmatch(str(digest)) is None
            ):
                raise ValueError(f"robot mask {label} provenance is invalid")
        for digest, label in (
            (self.camera_latency_sha256, "camera latency"),
            (self.camera_projection_sha256, "camera projection"),
            (self.occlusion_profile_sha256, "occlusion profile"),
            (self.alignment_sha256, "alignment"),
            (self.handeye_policy_sha256, "hand-eye policy"),
        ):
            if _SHA256.fullmatch(str(digest)) is None:
                raise ValueError(f"robot mask {label} provenance is invalid")
        if not isinstance(self.joint_publisher_identity, str) or not self.joint_publisher_identity:
            raise ValueError("robot mask joint publisher identity is invalid")
        expected_values = (
            (self.camera_id, activated_identity.camera_id),
            (
                self.camera_profile_sha256,
                activated_identity.camera_profile_sha256,
            ),
            (self.geometry_release_id, activated_identity.geometry_release_id),
            (self.geometry_sha256, activated_identity.geometry_sha256),
            (self.tool_release_id, activated_identity.tool_release_id),
            (self.tool_sha256, activated_identity.tool_sha256),
            (self.handeye_release_id, activated_identity.handeye_release_id),
            (self.handeye_sha256, activated_identity.handeye_sha256),
            (
                self.handeye_capability_schema,
                activated_identity.handeye_capability_schema,
            ),
            (
                self.handeye_calibration_tier,
                activated_identity.handeye_calibration_tier,
            ),
            (
                self.handeye_policy_sha256,
                activated_identity.handeye_policy_sha256,
            ),
            (
                self.camera_latency_sha256,
                activated_identity.camera_latency_sha256,
            ),
            (
                self.camera_projection_sha256,
                activated_identity.camera_projection_sha256,
            ),
            (
                self.occlusion_profile_sha256,
                activated_identity.occlusion_profile_sha256,
            ),
            (self.alignment_sha256, activated_identity.alignment_sha256),
        )
        if any(actual != expected for actual, expected in expected_values):
            raise ValueError(
                "robot mask provenance does not match activated occlusion identity"
            )
        identity_is_provisional = (
            activated_identity.handeye_capability_schema
            == PROVISIONAL_HANDEYE_CAPABILITY_SCHEMA
        )
        if self.handeye_provisional is not identity_is_provisional:
            raise ValueError(
                "robot mask hand-eye provisional policy does not match "
                "the activated identity"
            )
        if identity_is_provisional:
            if (
                self.handeye_calibration_status
                != "USER_ACCEPTED_BELOW_GATE"
                or self.handeye_quality_gate_passed is not False
                or not _finite_scalar(self.handeye_workspace_1sigma_max_m)
                or not 0.0 < float(self.handeye_workspace_1sigma_max_m) <= 0.0065
                or self.handeye_covariance_bound_transform
                != "transform_follower_base_from_camera"
                or self.handeye_covariance_scope != "random_only"
                or self.handeye_systematic_bias_status != "NOT_CHARACTERIZED"
                or self.handeye_systematic_bias_bound_m is not None
                or self.handeye_autonomous_targeting_authorized is not False
                or not _finite_scalar(
                    self.handeye_provisional_policy_allowance_m
                )
                or float(self.handeye_provisional_policy_allowance_m) != 0.0065
            ):
                raise ValueError(
                    "robot mask provisional hand-eye evidence is invalid"
                )
        elif (
            self.handeye_calibration_status != "COMPLETE"
            or self.handeye_quality_gate_passed is not None
            or self.handeye_workspace_1sigma_max_m is not None
            or self.handeye_covariance_bound_transform
            != "transform_camera_from_follower_base"
            or self.handeye_covariance_scope != "legacy_production"
            or self.handeye_systematic_bias_status != "LEGACY_NOT_APPLICABLE"
            or self.handeye_systematic_bias_bound_m is not None
            or self.handeye_autonomous_targeting_authorized is not True
            or self.handeye_provisional_policy_allowance_m != 0.0
        ):
            raise ValueError("robot mask legacy hand-eye evidence is invalid")

    def covers_pixel(self, pixel: np.ndarray) -> bool:
        values = np.asarray(pixel, dtype=np.float64).reshape(-1)
        if values.shape != (2,) or not np.isfinite(values).all():
            return False
        x, y = (int(round(float(value))) for value in values)
        return bool(
            0 <= y < self.mask.shape[0]
            and 0 <= x < self.mask.shape[1]
            and self.mask[y, x]
        )


@dataclass(frozen=True)
class PatientPosePacket:
    """Independent metric patient pose with an auditable covariance."""

    frame_index: int
    camera_host_t_s: float
    camera_id: str
    camera_profile_sha256: str
    alignment_sha256: str
    patient_pose_release_id: str
    patient_pose_sha256: str
    depth_noise_model_sha256: str
    back_mask_release_id: str
    back_mask_sha256: str
    rotation_camera_from_patient: np.ndarray
    translation_camera_from_patient_m: np.ndarray
    covariance_se3: np.ndarray
    support_mask: np.ndarray
    geometry_accepted: bool
    fit_rmse_m: float
    episode_id: str
    source: str
    scale_audit: float
    scale_tolerance: float
    valid: bool
    reason: str
    covariance_tangent_order: str
    covariance_perturbation: str
    covariance_transform: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rotation_camera_from_patient",
            _readonly_array(self.rotation_camera_from_patient, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "translation_camera_from_patient_m",
            _readonly_array(self.translation_camera_from_patient_m, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "covariance_se3",
            _readonly_array(self.covariance_se3, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "support_mask",
            _readonly_array(self.support_mask),
        )

    def validate_for_frame(
        self,
        *,
        frame_index: int,
        camera_host_t_s: float,
        camera_id: str,
        camera_profile_sha256: str,
        activated_identity: ActivatedOcclusionIdentity,
    ) -> bool:
        if not isinstance(activated_identity, ActivatedOcclusionIdentity):
            raise ValueError("activated patient-pose identity is missing")
        if type(self.frame_index) is not int or self.frame_index != int(frame_index):
            raise ValueError("patient pose frame index does not match the camera frame")
        if (
            not _finite_scalar(self.camera_host_t_s)
            or not _finite_scalar(camera_host_t_s)
            or float(self.camera_host_t_s) != float(camera_host_t_s)
        ):
            raise ValueError("patient pose host timestamp does not match the camera frame")
        if (
            not isinstance(self.camera_id, str)
            or self.camera_id != str(camera_id)
            or self.camera_id != activated_identity.camera_id
            or self.camera_profile_sha256 != str(camera_profile_sha256)
            or self.camera_profile_sha256
            != activated_identity.camera_profile_sha256
            or self.alignment_sha256 != activated_identity.alignment_sha256
            or self.patient_pose_release_id
            != activated_identity.patient_pose_release_id
            or self.patient_pose_sha256
            != activated_identity.patient_pose_sha256
            or self.depth_noise_model_sha256
            != activated_identity.depth_noise_model_sha256
            or self.back_mask_release_id
            != activated_identity.back_mask_release_id
            or self.back_mask_sha256 != activated_identity.back_mask_sha256
        ):
            raise ValueError(
                "patient pose provenance does not match the activated camera/provider"
            )
        for release_id, digest, label in (
            (
                self.patient_pose_release_id,
                self.patient_pose_sha256,
                "patient pose",
            ),
            (
                self.back_mask_release_id,
                self.back_mask_sha256,
                "back mask",
            ),
        ):
            if (
                _RELEASE_ID.fullmatch(str(release_id)) is None
                or _SHA256.fullmatch(str(digest)) is None
            ):
                raise ValueError(f"patient pose {label} identity is invalid")
        for digest, label in (
            (self.camera_profile_sha256, "camera profile"),
            (self.alignment_sha256, "alignment"),
            (self.depth_noise_model_sha256, "depth noise model"),
        ):
            if _SHA256.fullmatch(str(digest)) is None:
                raise ValueError(f"patient pose {label} identity is invalid")
        rotation = self.rotation_camera_from_patient
        translation = self.translation_camera_from_patient_m
        covariance = self.covariance_se3
        support = self.support_mask
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8)
            or not math.isclose(
                float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-8
            )
        ):
            raise ValueError("patient pose rotation is not a rigid SO(3) transform")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("patient pose translation is invalid")
        if (
            covariance.shape != (6, 6)
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, atol=1.0e-12)
            or float(np.min(np.linalg.eigvalsh(covariance))) < -1.0e-12
            or float(np.max(np.linalg.eigvalsh(covariance))) <= 0.0
        ):
            raise ValueError("patient pose covariance is invalid")
        if (
            support.shape != (10,)
            or support.dtype != np.bool_
        ):
            raise ValueError("patient pose support mask is invalid")
        if (
            not _finite_scalar(self.scale_audit)
            or not _finite_scalar(self.scale_tolerance)
            or self.scale_tolerance <= 0.0
            or abs(float(self.scale_audit) - 1.0) > float(self.scale_tolerance)
        ):
            raise ValueError("patient pose scale audit is outside tolerance")
        if type(self.valid) is not bool or type(self.geometry_accepted) is not bool:
            raise ValueError("patient pose validity flags are invalid")
        if (
            self.covariance_tangent_order != _COVARIANCE_TANGENT_ORDER
            or self.covariance_perturbation != _COVARIANCE_PERTURBATION
            or self.covariance_transform != _COVARIANCE_TRANSFORM
        ):
            raise ValueError("patient pose covariance convention is invalid")
        if not self.valid:
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("invalid patient pose needs a reason")
            return False
        if (
            not self.geometry_accepted
            or not _finite_scalar(self.fit_rmse_m)
            or self.fit_rmse_m < 0.0
            or self.fit_rmse_m > _MAX_PATIENT_POSE_RMSE_M
            or int(np.count_nonzero(support)) < _MIN_PATIENT_POSE_SUPPORT
            or not isinstance(self.episode_id, str)
            or not self.episode_id
            or self.source != _PATIENT_POSE_SOURCE
        ):
            raise ValueError("valid patient pose lacks accepted metric evidence")
        return True

    def camera_from_patient(self, point_patient_m: np.ndarray) -> np.ndarray:
        point = np.asarray(point_patient_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("patient-frame point is invalid")
        return (
            self.rotation_camera_from_patient @ point
            + self.translation_camera_from_patient_m
        )

    def patient_from_camera(self, point_camera_m: np.ndarray) -> np.ndarray:
        point = np.asarray(point_camera_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("camera-frame point is invalid")
        return self.rotation_camera_from_patient.T @ (
            point - self.translation_camera_from_patient_m
        )

    def propagated_position_bound_m(self, point_patient_m: np.ndarray) -> float:
        point = np.asarray(point_patient_m, dtype=np.float64)
        jacobian = np.hstack(
            (
                -self.rotation_camera_from_patient @ _skew(point),
                np.eye(3, dtype=np.float64),
            )
        )
        propagated = jacobian @ self.covariance_se3 @ jacobian.T
        eigenvalues = np.linalg.eigvalsh(0.5 * (propagated + propagated.T))
        if not np.isfinite(eigenvalues).all() or float(eigenvalues[-1]) < 0.0:
            return float("inf")
        return float(math.sqrt(_CHI2_3D_95 * max(float(eigenvalues[-1]), 0.0)))


@dataclass(frozen=True)
class OcclusionPointObservation:
    name: str
    xyz_camera_m: np.ndarray
    xyz_patient_m: np.ndarray
    pixel: np.ndarray
    measured: bool
    direct_geometry_accepted: bool
    robot_occluded: bool
    uncertainty_bound_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "xyz_camera_m", _readonly_array(self.xyz_camera_m, dtype=np.float64)
        )
        object.__setattr__(
            self, "xyz_patient_m", _readonly_array(self.xyz_patient_m, dtype=np.float64)
        )
        object.__setattr__(
            self, "pixel", _readonly_array(self.pixel, dtype=np.float64)
        )


@dataclass(frozen=True)
class OcclusionFrameObservation:
    frame_index: int
    camera_host_t_s: float
    camera_id: str
    camera_profile_sha256: str
    aligned_depth_shape: Tuple[int, int]
    back_present: bool
    initialized: bool
    points: Tuple[OcclusionPointObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))


@dataclass(frozen=True)
class OcclusionRuntimeConfig:
    max_freeze_age_s: float = 0.40
    max_uncertainty_bound_m: float = 0.010
    patient_motion_bound_m_s: float = 0.013
    handeye_bound_m: float = 0.003
    timing_sweep_bound_m: float = 0.002
    projection_bound_m: float = 0.001
    tool_geometry_bound_m: float = 0.001
    reacquire_frames: int = 5
    reacquire_max_displacement_m: float = 0.05
    reacquire_max_gap_s: float = 0.10

    def __post_init__(self) -> None:
        positive = (
            self.max_freeze_age_s,
            self.max_uncertainty_bound_m,
            self.patient_motion_bound_m_s,
            self.reacquire_max_displacement_m,
            self.reacquire_max_gap_s,
        )
        fixed_positive = (
            self.handeye_bound_m,
            self.timing_sweep_bound_m,
            self.projection_bound_m,
            self.tool_geometry_bound_m,
        )
        if not all(_finite_scalar(value) and float(value) > 0.0 for value in positive):
            raise ValueError("occlusion runtime positive bounds are invalid")
        if not all(
            _finite_scalar(value) and float(value) > 0.0
            for value in fixed_positive
        ):
            raise ValueError("occlusion runtime fixed bounds are invalid")
        if float(self.patient_motion_bound_m_s) < 0.013:
            raise ValueError(
                "occlusion runtime patient motion bound is below the "
                "commissioned production floor"
            )
        if (
            float(self.max_freeze_age_s) > 0.40
            or float(self.max_uncertainty_bound_m) > 0.010
            or type(self.reacquire_frames) is not int
            or self.reacquire_frames < 5
            or float(self.reacquire_max_displacement_m) > 0.05
            or float(self.reacquire_max_gap_s) > 0.10
        ):
            raise ValueError(
                "occlusion runtime values exceed the production safety envelope"
            )


def _runtime_profile_payload(
    *,
    max_freeze_age_s: float,
    max_uncertainty_bound_m: float,
    patient_motion_bound_m_s: float,
    handeye_bound_m: float,
    timing_sweep_bound_m: float,
    projection_bound_m: float,
    tool_geometry_bound_m: float,
    reacquire_frames: int,
    reacquire_max_displacement_m: float,
    reacquire_max_gap_s: float,
    patient_motion_artifact_sha256: str,
    handeye_artifact_sha256: str,
    camera_latency_artifact_sha256: str,
    camera_projection_artifact_sha256: str,
    tool_geometry_artifact_sha256: str,
) -> dict:
    return {
        "artifact_derived_components": {
            "camera_latency_artifact_sha256": (
                camera_latency_artifact_sha256
            ),
            "camera_projection_artifact_sha256": (
                camera_projection_artifact_sha256
            ),
            "handeye_artifact_sha256": handeye_artifact_sha256,
            "patient_motion_artifact_sha256": (
                patient_motion_artifact_sha256
            ),
            "tool_geometry_artifact_sha256": (
                tool_geometry_artifact_sha256
            ),
        },
        "runtime_bounds": {
            "handeye_bound_m": float(handeye_bound_m),
            "max_freeze_age_s": float(max_freeze_age_s),
            "max_uncertainty_bound_m": float(max_uncertainty_bound_m),
            "patient_motion_bound_m_s": float(patient_motion_bound_m_s),
            "projection_bound_m": float(projection_bound_m),
            "reacquire_frames": int(reacquire_frames),
            "reacquire_max_displacement_m": float(
                reacquire_max_displacement_m
            ),
            "reacquire_max_gap_s": float(reacquire_max_gap_s),
            "timing_sweep_bound_m": float(timing_sweep_bound_m),
            "tool_geometry_bound_m": float(tool_geometry_bound_m),
        },
        "schema": "wam_occlusion_runtime_profile_v1",
    }


@dataclass(frozen=True)
class ActivatedOcclusionRuntimeProfile:
    """Semantic-hashed runtime bounds and their exact source artifacts."""

    max_freeze_age_s: float
    max_uncertainty_bound_m: float
    patient_motion_bound_m_s: float
    handeye_bound_m: float
    timing_sweep_bound_m: float
    projection_bound_m: float
    tool_geometry_bound_m: float
    reacquire_frames: int
    reacquire_max_displacement_m: float
    reacquire_max_gap_s: float
    patient_motion_artifact_sha256: str
    handeye_artifact_sha256: str
    camera_latency_artifact_sha256: str
    camera_projection_artifact_sha256: str
    tool_geometry_artifact_sha256: str
    occlusion_runtime_profile_sha256: str

    @classmethod
    def create(
        cls,
        *,
        config: OcclusionRuntimeConfig,
        patient_motion_artifact_sha256: str,
        handeye_artifact_sha256: str,
        camera_latency_artifact_sha256: str,
        camera_projection_artifact_sha256: str,
        tool_geometry_artifact_sha256: str,
    ) -> "ActivatedOcclusionRuntimeProfile":
        if not isinstance(config, OcclusionRuntimeConfig):
            raise ValueError("occlusion runtime config is required")
        payload = _runtime_profile_payload(
            max_freeze_age_s=config.max_freeze_age_s,
            max_uncertainty_bound_m=config.max_uncertainty_bound_m,
            patient_motion_bound_m_s=config.patient_motion_bound_m_s,
            handeye_bound_m=config.handeye_bound_m,
            timing_sweep_bound_m=config.timing_sweep_bound_m,
            projection_bound_m=config.projection_bound_m,
            tool_geometry_bound_m=config.tool_geometry_bound_m,
            reacquire_frames=config.reacquire_frames,
            reacquire_max_displacement_m=(
                config.reacquire_max_displacement_m
            ),
            reacquire_max_gap_s=config.reacquire_max_gap_s,
            patient_motion_artifact_sha256=(
                patient_motion_artifact_sha256
            ),
            handeye_artifact_sha256=handeye_artifact_sha256,
            camera_latency_artifact_sha256=(
                camera_latency_artifact_sha256
            ),
            camera_projection_artifact_sha256=(
                camera_projection_artifact_sha256
            ),
            tool_geometry_artifact_sha256=(
                tool_geometry_artifact_sha256
            ),
        )
        return cls(
            max_freeze_age_s=float(config.max_freeze_age_s),
            max_uncertainty_bound_m=float(
                config.max_uncertainty_bound_m
            ),
            patient_motion_bound_m_s=float(
                config.patient_motion_bound_m_s
            ),
            handeye_bound_m=float(config.handeye_bound_m),
            timing_sweep_bound_m=float(config.timing_sweep_bound_m),
            projection_bound_m=float(config.projection_bound_m),
            tool_geometry_bound_m=float(config.tool_geometry_bound_m),
            reacquire_frames=config.reacquire_frames,
            reacquire_max_displacement_m=float(
                config.reacquire_max_displacement_m
            ),
            reacquire_max_gap_s=float(config.reacquire_max_gap_s),
            patient_motion_artifact_sha256=(
                patient_motion_artifact_sha256
            ),
            handeye_artifact_sha256=handeye_artifact_sha256,
            camera_latency_artifact_sha256=(
                camera_latency_artifact_sha256
            ),
            camera_projection_artifact_sha256=(
                camera_projection_artifact_sha256
            ),
            tool_geometry_artifact_sha256=(
                tool_geometry_artifact_sha256
            ),
            occlusion_runtime_profile_sha256=_semantic_sha256(payload),
        )

    def __post_init__(self) -> None:
        # Re-run the same safety envelope for direct construction and
        # dataclasses.replace(), then bind every value to one semantic digest.
        OcclusionRuntimeConfig(
            max_freeze_age_s=self.max_freeze_age_s,
            max_uncertainty_bound_m=self.max_uncertainty_bound_m,
            patient_motion_bound_m_s=self.patient_motion_bound_m_s,
            handeye_bound_m=self.handeye_bound_m,
            timing_sweep_bound_m=self.timing_sweep_bound_m,
            projection_bound_m=self.projection_bound_m,
            tool_geometry_bound_m=self.tool_geometry_bound_m,
            reacquire_frames=self.reacquire_frames,
            reacquire_max_displacement_m=(
                self.reacquire_max_displacement_m
            ),
            reacquire_max_gap_s=self.reacquire_max_gap_s,
        )
        for digest, label in (
            (
                self.patient_motion_artifact_sha256,
                "patient motion artifact",
            ),
            (self.handeye_artifact_sha256, "hand-eye artifact"),
            (
                self.camera_latency_artifact_sha256,
                "camera latency artifact",
            ),
            (
                self.camera_projection_artifact_sha256,
                "camera projection artifact",
            ),
            (
                self.tool_geometry_artifact_sha256,
                "tool geometry artifact",
            ),
        ):
            if _SHA256.fullmatch(str(digest)) is None:
                raise ValueError(f"occlusion runtime {label} SHA-256 is invalid")
        payload = _runtime_profile_payload(
            max_freeze_age_s=self.max_freeze_age_s,
            max_uncertainty_bound_m=self.max_uncertainty_bound_m,
            patient_motion_bound_m_s=self.patient_motion_bound_m_s,
            handeye_bound_m=self.handeye_bound_m,
            timing_sweep_bound_m=self.timing_sweep_bound_m,
            projection_bound_m=self.projection_bound_m,
            tool_geometry_bound_m=self.tool_geometry_bound_m,
            reacquire_frames=self.reacquire_frames,
            reacquire_max_displacement_m=(
                self.reacquire_max_displacement_m
            ),
            reacquire_max_gap_s=self.reacquire_max_gap_s,
            patient_motion_artifact_sha256=(
                self.patient_motion_artifact_sha256
            ),
            handeye_artifact_sha256=self.handeye_artifact_sha256,
            camera_latency_artifact_sha256=(
                self.camera_latency_artifact_sha256
            ),
            camera_projection_artifact_sha256=(
                self.camera_projection_artifact_sha256
            ),
            tool_geometry_artifact_sha256=(
                self.tool_geometry_artifact_sha256
            ),
        )
        if self.occlusion_runtime_profile_sha256 != _semantic_sha256(payload):
            raise ValueError(
                "occlusion runtime profile SHA-256 does not match semantic content"
            )

    def validate_for_activation(
        self,
        *,
        config: OcclusionRuntimeConfig,
        activated_identity: ActivatedOcclusionIdentity,
    ) -> None:
        if not isinstance(config, OcclusionRuntimeConfig):
            raise ValueError("occlusion runtime config is missing")
        if not isinstance(activated_identity, ActivatedOcclusionIdentity):
            raise ValueError("activated occlusion identity is missing")
        configured_bounds = (
            float(config.max_freeze_age_s),
            float(config.max_uncertainty_bound_m),
            float(config.patient_motion_bound_m_s),
            float(config.handeye_bound_m),
            float(config.timing_sweep_bound_m),
            float(config.projection_bound_m),
            float(config.tool_geometry_bound_m),
            config.reacquire_frames,
            float(config.reacquire_max_displacement_m),
            float(config.reacquire_max_gap_s),
        )
        profiled_bounds = (
            float(self.max_freeze_age_s),
            float(self.max_uncertainty_bound_m),
            float(self.patient_motion_bound_m_s),
            float(self.handeye_bound_m),
            float(self.timing_sweep_bound_m),
            float(self.projection_bound_m),
            float(self.tool_geometry_bound_m),
            self.reacquire_frames,
            float(self.reacquire_max_displacement_m),
            float(self.reacquire_max_gap_s),
        )
        if configured_bounds != profiled_bounds:
            raise ValueError(
                "occlusion runtime profile does not match runtime config"
            )
        artifact_sources = (
            (
                self.handeye_artifact_sha256,
                activated_identity.handeye_sha256,
            ),
            (
                self.camera_latency_artifact_sha256,
                activated_identity.camera_latency_sha256,
            ),
            (
                self.camera_projection_artifact_sha256,
                activated_identity.camera_projection_sha256,
            ),
            (
                self.tool_geometry_artifact_sha256,
                activated_identity.tool_sha256,
            ),
        )
        if any(actual != expected for actual, expected in artifact_sources):
            raise ValueError(
                "occlusion runtime profile artifact sources do not match "
                "the activated identity"
            )
        if (
            self.occlusion_runtime_profile_sha256
            != activated_identity.occlusion_runtime_profile_sha256
        ):
            raise ValueError(
                "occlusion runtime profile does not match activated identity"
            )


@dataclass(frozen=True)
class SafePoint:
    name: str
    occlusion_state: str
    valid: bool
    measured: bool
    xyz_camera_m: Optional[np.ndarray]
    pixel: Optional[np.ndarray]
    uncertainty_bound_m: float
    robot_occluded: bool
    source: str
    reason: str

    def __post_init__(self) -> None:
        if self.xyz_camera_m is not None:
            object.__setattr__(
                self,
                "xyz_camera_m",
                _readonly_array(self.xyz_camera_m, dtype=np.float64),
            )
        if self.pixel is not None:
            object.__setattr__(
                self,
                "pixel",
                _readonly_array(self.pixel, dtype=np.float64),
            )


@dataclass(frozen=True)
class SafeTrackingFrame:
    frame_index: int
    camera_host_t_s: float
    back_present: bool
    initialized: bool
    points: Tuple[SafePoint, ...]
    reason: str


@dataclass
class _PointHistory:
    patient_anchor_m: Optional[np.ndarray] = None
    patient_episode_id: Optional[str] = None
    anchor_base_bound_m: Optional[float] = None
    last_camera_m: Optional[np.ndarray] = None
    last_pixel: Optional[np.ndarray] = None
    last_valid_t_s: Optional[float] = None
    last_bound_m: Optional[float] = None
    ever_valid: bool = False
    lost: bool = False
    requires_reacquisition: bool = False
    reacquire_count: int = 0
    reacquire_last_candidate_m: Optional[np.ndarray] = None
    reacquire_last_frame_index: Optional[int] = None
    reacquire_last_t_s: Optional[float] = None


class OcclusionRuntime:
    """Publish only points backed by current geometry or bounded history."""

    def __init__(
        self,
        point_names: Sequence[str],
        config: Optional[OcclusionRuntimeConfig] = None,
        *,
        activated_identity: ActivatedOcclusionIdentity,
        camera_projection: ActivatedCameraProjection,
        runtime_profile: ActivatedOcclusionRuntimeProfile,
    ) -> None:
        names = tuple(str(name) for name in point_names)
        if not names or len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("occlusion runtime point names must be unique and nonempty")
        self.point_names = names
        self.config = config or OcclusionRuntimeConfig()
        if not isinstance(activated_identity, ActivatedOcclusionIdentity):
            raise ValueError("activated occlusion identity is required")
        if not isinstance(camera_projection, ActivatedCameraProjection):
            raise ValueError("activated camera projection is required")
        if not isinstance(
            runtime_profile,
            ActivatedOcclusionRuntimeProfile,
        ):
            raise ValueError("activated occlusion runtime profile is required")
        if (
            camera_projection.camera_id != activated_identity.camera_id
            or camera_projection.camera_profile_sha256
            != activated_identity.camera_profile_sha256
            or camera_projection.camera_projection_sha256
            != activated_identity.camera_projection_sha256
        ):
            raise ValueError(
                "activated camera projection does not match occlusion identity"
            )
        runtime_profile.validate_for_activation(
            config=self.config,
            activated_identity=activated_identity,
        )
        self.activated_identity = activated_identity
        self.camera_projection = camera_projection
        self.runtime_profile = runtime_profile
        self._history = {name: _PointHistory() for name in names}
        self._last_frame_t_s: Optional[float] = None
        self._last_frame_index: Optional[int] = None

    def reset(self) -> None:
        self._clear_tracking_history()
        self._last_frame_t_s = None
        self._last_frame_index = None

    def _clear_tracking_history(self) -> None:
        self._history = {name: _PointHistory() for name in self.point_names}

    @staticmethod
    def _enter_lost(history: _PointHistory) -> None:
        if history.ever_valid or history.last_camera_m is not None:
            history.lost = True
            history.requires_reacquisition = True
        history.reacquire_count = 0
        history.reacquire_last_candidate_m = None
        history.reacquire_last_frame_index = None
        history.reacquire_last_t_s = None

    def _enter_all_lost(self) -> None:
        for history in self._history.values():
            self._enter_lost(history)

    @staticmethod
    def _lost_point(name: str, reason: str, *, robot_occluded=False) -> SafePoint:
        return SafePoint(
            name=name,
            occlusion_state=LOST,
            valid=False,
            measured=False,
            xyz_camera_m=None,
            pixel=None,
            uncertainty_bound_m=float("inf"),
            robot_occluded=bool(robot_occluded),
            source="none_fail_closed",
            reason=str(reason),
        )

    def _failed_frame(
        self, observation: OcclusionFrameObservation, reason: str
    ) -> SafeTrackingFrame:
        self._enter_all_lost()
        return SafeTrackingFrame(
            frame_index=int(observation.frame_index),
            camera_host_t_s=float(observation.camera_host_t_s),
            back_present=False,
            initialized=False,
            points=tuple(self._lost_point(name, reason) for name in self.point_names),
            reason=str(reason),
        )

    def _fixed_bound_m(self) -> float:
        return float(
            self.config.handeye_bound_m
            + self.config.timing_sweep_bound_m
            + self.config.projection_bound_m
            + self.config.tool_geometry_bound_m
        )

    def _validate_observation(
        self, observation: OcclusionFrameObservation
    ) -> dict[str, OcclusionPointObservation]:
        if (
            type(observation.frame_index) is not int
            or observation.frame_index < 0
            or not _finite_scalar(observation.camera_host_t_s)
            or not isinstance(observation.camera_id, str)
            or not observation.camera_id
            or _SHA256.fullmatch(str(observation.camera_profile_sha256)) is None
            or not isinstance(observation.aligned_depth_shape, tuple)
            or len(observation.aligned_depth_shape) != 2
            or any(type(value) is not int or value <= 0
                   for value in observation.aligned_depth_shape)
            or type(observation.back_present) is not bool
            or type(observation.initialized) is not bool
        ):
            raise ValueError("occlusion frame observation is invalid")
        by_name = {}
        for point in observation.points:
            if not isinstance(point, OcclusionPointObservation):
                raise ValueError("occlusion point observation type is invalid")
            if point.name in by_name or point.name not in self._history:
                raise ValueError("occlusion point names are invalid")
            if (
                point.xyz_camera_m.shape != (3,)
                or point.xyz_patient_m.shape != (3,)
                or point.pixel.shape != (2,)
                or type(point.measured) is not bool
                or type(point.direct_geometry_accepted) is not bool
                or type(point.robot_occluded) is not bool
                or not _finite_scalar(point.uncertainty_bound_m)
                or point.uncertainty_bound_m < 0.0
            ):
                raise ValueError("occlusion point fields are invalid")
            by_name[point.name] = point
        if set(by_name) != set(self.point_names):
            raise ValueError("occlusion frame does not contain every point exactly once")
        return by_name

    def advance(
        self,
        observation: OcclusionFrameObservation,
        *,
        robot_mask_packet: RobotMaskPacket,
        patient_pose_packet: PatientPosePacket,
    ) -> SafeTrackingFrame:
        try:
            by_name = self._validate_observation(observation)
            self.camera_projection.validate_for_frame(
                camera_id=observation.camera_id,
                camera_profile_sha256=observation.camera_profile_sha256,
                aligned_depth_shape=observation.aligned_depth_shape,
            )
            if not isinstance(robot_mask_packet, RobotMaskPacket):
                raise ValueError("robot mask packet is missing or invalid")
            if not isinstance(patient_pose_packet, PatientPosePacket):
                raise ValueError("patient pose packet is missing or invalid")
            robot_mask_packet.validate_for_frame(
                frame_index=observation.frame_index,
                camera_host_t_s=observation.camera_host_t_s,
                camera_id=observation.camera_id,
                camera_profile_sha256=observation.camera_profile_sha256,
                depth_shape=observation.aligned_depth_shape,
                activated_identity=self.activated_identity,
            )
            pose_valid = patient_pose_packet.validate_for_frame(
                frame_index=observation.frame_index,
                camera_host_t_s=observation.camera_host_t_s,
                camera_id=observation.camera_id,
                camera_profile_sha256=observation.camera_profile_sha256,
                activated_identity=self.activated_identity,
            )
        except (TypeError, ValueError):
            return self._failed_frame(observation, "INVALID_OCCLUSION_EVIDENCE")

        timestamp = float(observation.camera_host_t_s)
        if (
            (self._last_frame_t_s is not None and timestamp <= self._last_frame_t_s)
            or (
                self._last_frame_index is not None
                and observation.frame_index <= self._last_frame_index
            )
        ):
            return self._failed_frame(observation, "NON_MONOTONIC_TIME")
        self._last_frame_t_s = timestamp
        self._last_frame_index = observation.frame_index

        pose_resets_patient_frame = bool(
            not pose_valid
            and patient_pose_packet.reason
            in {
                "NO_BACK",
                "INVALID_EPISODE",
                "EPISODE_CHANGED_REQUIRES_RESET",
                "RESET_REQUIRED",
                "NON_MONOTONIC_FRAME_OR_TIME",
            }
        )
        if pose_resets_patient_frame:
            self._clear_tracking_history()
            return self._failed_frame(observation, "NO_INITIALIZED_BACK")

        if not observation.back_present or not observation.initialized:
            self._clear_tracking_history()
            return self._failed_frame(observation, "NO_INITIALIZED_BACK")

        safe_points = []
        for name in self.point_names:
            point = by_name[name]
            history = self._history[name]
            mask_occluded = robot_mask_packet.covers_pixel(point.pixel)
            previous_pixel_occluded = bool(
                history.last_pixel is not None
                and robot_mask_packet.covers_pixel(history.last_pixel)
            )
            if (
                history.ever_valid
                and previous_pixel_occluded
                and not mask_occluded
            ):
                # A learned point can jump from the true arm-covered location
                # to an unmasked pixel.  That transition is not new direct
                # evidence; it must pass the full consecutive-frame gate.
                history.requires_reacquisition = True
            if mask_occluded != point.robot_occluded:
                self._enter_lost(history)
                safe_points.append(
                    self._lost_point(name, "ROBOT_MASK_POINT_EVIDENCE_MISMATCH")
                )
                continue
            finite_camera = bool(
                point.xyz_camera_m.shape == (3,)
                and np.isfinite(point.xyz_camera_m).all()
            )
            finite_patient = bool(
                point.xyz_patient_m.shape == (3,)
                and np.isfinite(point.xyz_patient_m).all()
            )
            finite_pixel = bool(
                point.pixel.shape == (2,) and np.isfinite(point.pixel).all()
            )
            direct = bool(
                point.measured
                and point.direct_geometry_accepted
                and not mask_occluded
                and finite_camera
                and finite_patient
                and finite_pixel
                and point.uncertainty_bound_m
                <= self.config.max_uncertainty_bound_m
            )
            if direct:
                if history.lost or history.requires_reacquisition:
                    reference = history.last_camera_m
                    previous_candidate = history.reacquire_last_candidate_m
                    reference_jump = (
                        float("inf")
                        if reference is None
                        else float(np.linalg.norm(point.xyz_camera_m - reference))
                    )
                    consecutive_jump = (
                        0.0
                        if previous_candidate is None
                        else float(
                            np.linalg.norm(
                                point.xyz_camera_m - previous_candidate
                            )
                        )
                    )
                    if (
                        not math.isfinite(reference_jump)
                        or reference_jump
                        > self.config.reacquire_max_displacement_m
                        or not math.isfinite(consecutive_jump)
                        or consecutive_jump
                        > self.config.reacquire_max_displacement_m
                    ):
                        history.reacquire_count = 0
                        history.reacquire_last_candidate_m = None
                        history.reacquire_last_frame_index = None
                        history.reacquire_last_t_s = None
                        safe_points.append(
                            self._lost_point(
                                name, "REACQUISITION_DISCONTINUITY"
                            )
                        )
                        continue
                    consecutive = bool(
                        history.reacquire_count == 0
                        or (
                            history.reacquire_last_frame_index is not None
                            and observation.frame_index
                            == history.reacquire_last_frame_index + 1
                            and history.reacquire_last_t_s is not None
                            and 0.0
                            < _elapsed_seconds(
                                timestamp, history.reacquire_last_t_s
                            )
                            <= self.config.reacquire_max_gap_s
                        )
                    )
                    if not consecutive:
                        history.reacquire_count = 0
                        history.reacquire_last_candidate_m = None
                    history.reacquire_count += 1
                    history.reacquire_last_candidate_m = (
                        point.xyz_camera_m.copy()
                    )
                    history.reacquire_last_frame_index = observation.frame_index
                    history.reacquire_last_t_s = timestamp
                    if history.reacquire_count < self.config.reacquire_frames:
                        safe_points.append(
                            SafePoint(
                                name=name,
                                occlusion_state=REACQUIRING,
                                valid=False,
                                measured=False,
                                xyz_camera_m=None,
                                pixel=None,
                                uncertainty_bound_m=float("inf"),
                                robot_occluded=False,
                                source="hidden_reacquisition_streak",
                                reason="reacquisition_requires_consecutive_frames",
                            )
                        )
                        continue
                if pose_valid:
                    anchor = patient_pose_packet.patient_from_camera(
                        point.xyz_camera_m
                    )
                    anchor_pose_bound = (
                        patient_pose_packet.propagated_position_bound_m(anchor)
                    )
                    anchor_base_bound = (
                        float(point.uncertainty_bound_m)
                        + float(anchor_pose_bound)
                    )
                    if (
                        math.isfinite(anchor_base_bound)
                        and anchor_base_bound
                        <= self.config.max_uncertainty_bound_m
                    ):
                        history.patient_anchor_m = np.asarray(
                            anchor, dtype=np.float64
                        ).copy()
                        history.patient_episode_id = (
                            patient_pose_packet.episode_id
                        )
                        history.anchor_base_bound_m = anchor_base_bound
                    else:
                        history.patient_anchor_m = None
                        history.patient_episode_id = None
                        history.anchor_base_bound_m = None
                else:
                    history.patient_anchor_m = None
                    history.patient_episode_id = None
                    history.anchor_base_bound_m = None
                history.last_camera_m = point.xyz_camera_m.copy()
                history.last_pixel = point.pixel.copy()
                history.last_valid_t_s = timestamp
                history.last_bound_m = float(point.uncertainty_bound_m)
                history.ever_valid = True
                history.lost = False
                history.requires_reacquisition = False
                history.reacquire_count = 0
                history.reacquire_last_candidate_m = None
                history.reacquire_last_frame_index = None
                history.reacquire_last_t_s = None
                safe_points.append(
                    SafePoint(
                        name=name,
                        occlusion_state=DIRECT,
                        valid=True,
                        measured=True,
                        xyz_camera_m=point.xyz_camera_m,
                        pixel=point.pixel,
                        uncertainty_bound_m=float(point.uncertainty_bound_m),
                        robot_occluded=False,
                        source="current_direct_geometry",
                        reason="current_point_observed_and_geometry_accepted",
                    )
                )
                continue

            if history.ever_valid:
                history.requires_reacquisition = True
            if (
                mask_occluded
                and pose_valid
                and history.patient_anchor_m is not None
                and history.anchor_base_bound_m is not None
            ):
                if history.patient_episode_id != patient_pose_packet.episode_id:
                    history.patient_anchor_m = None
                    history.patient_episode_id = None
                    history.anchor_base_bound_m = None
                    history.lost = True
                    history.requires_reacquisition = True
                    history.reacquire_count = 0
                    history.reacquire_last_candidate_m = None
                    history.reacquire_last_frame_index = None
                    history.reacquire_last_t_s = None
                    safe_points.append(
                        self._lost_point(
                            name,
                            "PATIENT_POSE_EPISODE_CHANGED",
                            robot_occluded=True,
                        )
                    )
                    continue
                camera_point = patient_pose_packet.camera_from_patient(
                    history.patient_anchor_m
                )
                projected_pixel = self.camera_projection.project_camera_point(
                    camera_point
                )
                bound = (
                    float(history.anchor_base_bound_m)
                    + patient_pose_packet.propagated_position_bound_m(
                        history.patient_anchor_m
                    )
                    + self._fixed_bound_m()
                )
                uncertainty_visible = bool(
                    math.isfinite(bound)
                    and bound <= self.config.max_uncertainty_bound_m
                    and self.camera_projection.uncertainty_region_fully_visible(
                        camera_point,
                        bound,
                    )
                )
                projected_robot_occluded = bool(
                    projected_pixel is not None
                    and robot_mask_packet.covers_pixel(projected_pixel)
                )
                if (
                    np.isfinite(camera_point).all()
                    and projected_pixel is not None
                    and math.isfinite(bound)
                    and bound <= self.config.max_uncertainty_bound_m
                    and uncertainty_visible
                    and projected_robot_occluded
                ):
                    history.last_camera_m = camera_point.copy()
                    history.last_pixel = projected_pixel.copy()
                    history.last_valid_t_s = timestamp
                    history.last_bound_m = float(bound)
                    history.ever_valid = True
                    history.lost = False
                    safe_points.append(
                        SafePoint(
                            name=name,
                            occlusion_state=PATIENT_FRAME_LOCKED,
                            valid=True,
                            measured=False,
                            xyz_camera_m=camera_point,
                            pixel=projected_pixel,
                            uncertainty_bound_m=float(bound),
                            robot_occluded=True,
                            source="independent_patient_pose",
                            reason="robot_occluded_with_current_metric_patient_pose",
                        )
                    )
                    continue
                if projected_pixel is None:
                    # A current accepted patient pose places this target behind
                    # the camera or outside the activated image.  Falling back
                    # to the previous on-screen pixel would contradict newer
                    # geometry and could direct the operator to the wrong
                    # location.
                    self._enter_lost(history)
                    safe_points.append(
                        self._lost_point(
                            name,
                            "LOCKED_TARGET_OUTSIDE_CAMERA_PROJECTION",
                            robot_occluded=True,
                        )
                    )
                    continue
                if (
                    math.isfinite(bound)
                    and bound <= self.config.max_uncertainty_bound_m
                    and not uncertainty_visible
                ):
                    self._enter_lost(history)
                    safe_points.append(
                        self._lost_point(
                            name,
                            "LOCKED_UNCERTAINTY_OUTSIDE_CAMERA",
                            robot_occluded=True,
                        )
                    )
                    continue
                if (
                    math.isfinite(bound)
                    and bound <= self.config.max_uncertainty_bound_m
                    and not projected_robot_occluded
                ):
                    self._enter_lost(history)
                    safe_points.append(
                        self._lost_point(
                            name,
                            "PROJECTED_TARGET_NOT_ROBOT_OCCLUDED",
                            robot_occluded=False,
                        )
                    )
                    continue

            if (
                history.ever_valid
                and history.last_camera_m is not None
                and history.last_valid_t_s is not None
                and history.last_bound_m is not None
            ):
                age = _elapsed_seconds(timestamp, history.last_valid_t_s)
                bound = float(
                    Decimal(str(history.last_bound_m))
                    + Decimal(str(self.config.patient_motion_bound_m_s))
                    * Decimal(str(age))
                )
                if (
                    age >= 0.0
                    and age <= self.config.max_freeze_age_s
                    and bound <= self.config.max_uncertainty_bound_m
                ):
                    safe_points.append(
                        SafePoint(
                            name=name,
                            occlusion_state=BOUNDED_FREEZE,
                            valid=True,
                            measured=False,
                            xyz_camera_m=history.last_camera_m,
                            pixel=history.last_pixel,
                            uncertainty_bound_m=bound,
                            robot_occluded=mask_occluded,
                            source="last_valid_bounded_freeze",
                            reason="patient_pose_temporarily_unavailable",
                        )
                    )
                    continue

            self._enter_lost(history)
            safe_points.append(
                self._lost_point(
                    name,
                    "NO_CURRENT_OR_BOUNDED_3D_ESTIMATE",
                    robot_occluded=mask_occluded,
                )
            )

        return SafeTrackingFrame(
            frame_index=observation.frame_index,
            camera_host_t_s=timestamp,
            back_present=True,
            initialized=all(point.valid for point in safe_points),
            points=tuple(safe_points),
            reason=(
                "TRACKING"
                if all(point.valid for point in safe_points)
                else "INCOMPLETE_OCCLUSION_SAFE_CONSTELLATION"
            ),
        )


__all__ = [
    "ActivatedCameraProjection",
    "ActivatedOcclusionIdentity",
    "ActivatedOcclusionRuntimeProfile",
    "BOUNDED_FREEZE",
    "DIRECT",
    "LOST",
    "OcclusionFrameObservation",
    "OcclusionPointObservation",
    "OcclusionRuntime",
    "OcclusionRuntimeConfig",
    "PATIENT_FRAME_LOCKED",
    "PatientPosePacket",
    "REACQUIRING",
    "RobotMaskPacket",
    "SafePoint",
    "SafeTrackingFrame",
]
