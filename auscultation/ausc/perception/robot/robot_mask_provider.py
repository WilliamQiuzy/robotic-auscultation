"""Fail-closed interval-swept OpenArm robot/tool mask generation.

The production runtime in this module is intentionally limited to NumPy and
OpenCV.  Collision meshes are converted offline into bounded link/tool vertex
sets; no URDF or mesh parser is imported while the collector is running.

The only accepted hardware mapping is the left unilateral OpenArm pair:
leader ``can1`` drives follower ``can3``.  The follower joint state is used for
camera occlusion geometry while the leader state remains part of the exact
telemetry/process provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .camera_exposure_evidence import CameraExposureEvidence
from .robot_state_stream import (
    EXACT_CONTROL_RECORD_SCHEMA,
    EXACT_CONTROL_RECORD_SCHEMA_V3,
    EXACT_CONTROL_RECORD_SCHEMA_V4,
    BracketedExactControlTrajectory,
    ExactControlRecord,
    ExactFollowerAction,
    InterpolatedFollowerObservation,
)
from ..tracking.occlusion_runtime import (
    ActivatedOcclusionIdentity,
    LEGACY_HANDEYE_CALIBRATION_TIER,
    LEGACY_HANDEYE_CAPABILITY_SCHEMA,
    PROVISIONAL_HANDEYE_CALIBRATION_TIER,
    PROVISIONAL_HANDEYE_CAPABILITY_SCHEMA,
    RobotMaskPacket,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_RE = re.compile(r"^[a-z][a-z0-9_]*-[0-9a-f]{20}$")
_NS_PER_SECOND = 1_000_000_000
_LEFT_JOINT_NAMES = tuple(
    f"openarm_left_joint{index}" for index in range(1, 8)
)
_MAX_LINKS = 64
_MAX_VERTICES_PER_COMPONENT = 8_192
_MAX_TOTAL_VERTICES = 65_536
_MAX_DILATION_KERNEL_RADIUS_PX = 256
# 2026-08-14: the 96 px full-mask cost hatch (previous revision) fired on
# EVERY frame of the real seated pose (610/610 in the sealed diagnostic
# session) and blinded the whole perception stack: a full robot mask makes
# the patient's back invisible downstream (NO_BACK, zero published points).
# Large radii now use a CONSERVATIVE PYRAMID dilation instead: max-pool the
# fill by factor f, dilate at coarse scale with radius ceil(r/f)+1, upsample
# nearest. Provably a superset of the exact dilation (disc radius
# (ceil(r/f)+1)*f >= r + f covers both the scaled radius and the <=f-1 px
# quantization) at ~1/f^2 of the cost, and it stays LOCAL. The factor is
# ADAPTIVE (2026-08-14 second sealed forensics: the real seated pose drives
# radii past 256 px, so a fixed hatch at the kernel cap still produced the
# full-frame mask on every sampled frame): f = max(4, ceil(r/64)) keeps the
# coarse kernel radius <= 65 regardless of r, so any radius is cheap. The
# full-mask hatch survives only as a sentinel where full frame is honest:
# nominal-outside, or a radius so large the disc spans the frame anyway.
_PYRAMID_DILATION_MIN_RADIUS_PX = 64
_PYRAMID_DILATION_FACTOR = 4
_PYRAMID_COARSE_RADIUS_TARGET = 64
_EXTENDED_CANVAS_AREA_FACTOR = 16
_FULL_MASK_COST_HATCH_RADIUS_PX = 2048
_PROVISIONAL_HAND_EYE_RUN_RE = re.compile(
    r"^handeye_left_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$"
)
_PROVISIONAL_HAND_EYE_COVARIANCE_ORDER = (
    "rx_rad", "ry_rad", "rz_rad", "tx_m", "ty_m", "tz_m",
)
_PROVISIONAL_HAND_EYE_COVARIANCE_DEFINITION = (
    "empirical bootstrap cov of delta=[Rodrigues(R_ref^T R_boot), "
    "t_boot - t_ref]; random precision only, excludes systematic bias"
)
_PROVISIONAL_HAND_EYE_SCOPES = (
    "bootstrap_teleop_logging",
    "conservative_robot_mask",
)
_PROVISIONAL_HAND_EYE_POLICY_ALLOWANCE_M = 0.0065
_PROVISIONAL_HAND_EYE_KEYS = {
    "acceptance_note",
    "authorized_scopes",
    "autonomous_targeting_authorized",
    "bootstrap_reproducibility",
    "calibration_status",
    "camera_id",
    "code_release",
    "covariance_scope",
    "covariance_se3",
    "failed_quality_gate_m",
    "provisional",
    "quality",
    "quality_gate_passed",
    "schema",
    "side",
    "source_evidence",
    "source_frame",
    "source_run_id",
    "systematic_bias_bound_m",
    "systematic_bias_status",
    "target_frame",
    "tip_offset_frame",
    "tip_offset_tcp_m",
    "transform_camera_from_follower_base",
    "transform_follower_base_from_camera",
    "workspace_1sigma_m",
}


class RobotMaskProviderError(ValueError):
    """Inputs cannot authorize an OpenArm occlusion mask."""


def _readonly_array(value, *, dtype=np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RobotMaskProviderError(
            "artifact cannot be encoded as canonical JSON"
        ) from exc


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RobotMaskProviderError(f"{label} SHA-256 is invalid")
    return value


def _require_release(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _RELEASE_RE.fullmatch(value) is None:
        raise RobotMaskProviderError(f"{label} release ID is invalid")
    return value


def _finite_scalar(value: object, *, label: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise RobotMaskProviderError(f"{label} must be finite")
    return float(value)


def _validate_transform(value: object, *, label: str) -> np.ndarray:
    result = _readonly_array(value)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise RobotMaskProviderError(f"{label} must be a finite 4x4 SE(3)")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-10):
        raise RobotMaskProviderError(f"{label} homogeneous row is invalid")
    rotation = result[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7
        )
    ):
        raise RobotMaskProviderError(f"{label} rotation is not in SO(3)")
    if float(np.linalg.norm(result[:3, 3])) > 25.0:
        raise RobotMaskProviderError(f"{label} translation exceeds 25 m")
    return result


def _validate_vertices(value: object, *, label: str) -> np.ndarray:
    result = _readonly_array(value)
    if (
        result.ndim != 2
        or result.shape[1:] != (3,)
        or not 4 <= result.shape[0] <= _MAX_VERTICES_PER_COMPONENT
        or not np.isfinite(result).all()
    ):
        raise RobotMaskProviderError(
            f"{label} must contain 4..{_MAX_VERTICES_PER_COMPONENT} finite xyz vertices"
        )
    if np.max(np.linalg.norm(result, axis=1)) > 4.0:
        raise RobotMaskProviderError(f"{label} vertices exceed the 4 m bound")
    if np.linalg.matrix_rank(result - np.mean(result, axis=0)) < 2:
        raise RobotMaskProviderError(f"{label} vertices are degenerate")
    return result


def _array_payload(value: np.ndarray) -> list:
    return np.asarray(value, dtype=np.float64).tolist()


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    if not math.isfinite(float(angle)):
        raise RobotMaskProviderError("forward kinematics received a non-finite joint")
    axis = np.asarray(axis, dtype=np.float64)
    skew = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=np.float64,
    )
    sine = math.sin(float(angle))
    cosine = math.cos(float(angle))
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = (
        np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)
    )
    if not np.isfinite(value).all():
        raise RobotMaskProviderError("forward kinematics produced non-finite rotation")
    return value


@dataclass(frozen=True)
class KinematicCollisionLink:
    """One preprocessed collision component in a parent-first chain."""

    name: str
    parent_index: int
    joint_index: Optional[int]
    transform_parent_from_joint_zero: np.ndarray
    joint_axis: np.ndarray
    vertices_link_m: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise RobotMaskProviderError("collision link name is missing")
        if type(self.parent_index) is not int or self.parent_index < -1:
            raise RobotMaskProviderError("collision link parent index is invalid")
        if self.joint_index is not None and (
            type(self.joint_index) is not int
            or not 0 <= self.joint_index < 7
        ):
            raise RobotMaskProviderError("collision link joint index is invalid")
        transform = _validate_transform(
            self.transform_parent_from_joint_zero,
            label=f"{self.name} zero transform",
        )
        axis = _readonly_array(self.joint_axis)
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise RobotMaskProviderError(f"{self.name} joint axis is invalid")
        axis_norm = float(np.linalg.norm(axis))
        if self.joint_index is not None:
            if not math.isclose(axis_norm, 1.0, abs_tol=1e-7):
                raise RobotMaskProviderError(
                    f"{self.name} joint axis must be unit length"
                )
        elif axis_norm > 1e-10:
            raise RobotMaskProviderError(
                f"{self.name} fixed component must have a zero joint axis"
            )
        vertices = _validate_vertices(
            self.vertices_link_m,
            label=f"{self.name} collision vertices",
        )
        object.__setattr__(
            self, "transform_parent_from_joint_zero", transform
        )
        object.__setattr__(self, "joint_axis", axis)
        object.__setattr__(self, "vertices_link_m", vertices)

    def payload(self) -> dict:
        return {
            "joint_axis": _array_payload(self.joint_axis),
            "joint_index": self.joint_index,
            "name": self.name,
            "parent_index": self.parent_index,
            "transform_parent_from_joint_zero": _array_payload(
                self.transform_parent_from_joint_zero
            ),
            "vertices_link_m": _array_payload(self.vertices_link_m),
        }


@dataclass(frozen=True)
class OpenArmCollisionGeometry:
    """Bounded collision vertices derived offline from official OpenArm assets."""

    schema: str
    source: str
    side: str
    joint_names: Tuple[str, ...]
    urdf_sha256: str
    links: Tuple[KinematicCollisionLink, ...]
    geometry_release_id: str
    geometry_sha256: str

    @classmethod
    def create(
        cls,
        *,
        source: str,
        side: str,
        joint_names: Sequence[str],
        urdf_sha256: str,
        links: Sequence[KinematicCollisionLink],
    ) -> "OpenArmCollisionGeometry":
        joint_names = tuple(joint_names)
        links = tuple(links)
        payload = {
            "joint_names": list(joint_names),
            "links": [link.payload() for link in links],
            "schema": "wam_openarm_collision_geometry_v1",
            "side": side,
            "source": source,
            "urdf_sha256": urdf_sha256,
        }
        digest = _semantic_sha256(payload)
        return cls(
            schema="wam_openarm_collision_geometry_v1",
            source=source,
            side=side,
            joint_names=joint_names,
            urdf_sha256=urdf_sha256,
            links=links,
            geometry_release_id=f"openarm_geometry-{digest[:20]}",
            geometry_sha256=digest,
        )

    def __post_init__(self) -> None:
        joint_names = tuple(self.joint_names)
        links = tuple(self.links)
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(self, "links", links)
        if self.schema != "wam_openarm_collision_geometry_v1":
            raise RobotMaskProviderError("collision geometry schema is unsupported")
        if self.source != "official_openarm_v10_urdf_collision_meshes":
            raise RobotMaskProviderError(
                "collision geometry is not derived from the official OpenArm asset"
            )
        if self.side != "left" or joint_names != _LEFT_JOINT_NAMES:
            raise RobotMaskProviderError(
                "collision geometry must be the canonical left OpenArm chain"
            )
        _require_sha256(self.urdf_sha256, label="collision geometry URDF")
        if not 7 <= len(links) <= _MAX_LINKS:
            raise RobotMaskProviderError("collision chain link count is invalid")
        joint_indices = []
        total_vertices = 0
        for index, link in enumerate(links):
            if not isinstance(link, KinematicCollisionLink):
                raise RobotMaskProviderError(
                    "collision chain contains a non-link value"
                )
            if index == 0:
                if link.parent_index != -1:
                    raise RobotMaskProviderError(
                        "collision chain root must have parent -1"
                    )
            elif not 0 <= link.parent_index < index:
                raise RobotMaskProviderError(
                    "collision chain must be stored in parent-first order"
                )
            if link.joint_index is not None:
                joint_indices.append(link.joint_index)
            total_vertices += int(link.vertices_link_m.shape[0])
        if tuple(sorted(joint_indices)) != tuple(range(7)):
            raise RobotMaskProviderError(
                "collision chain must contain each of the seven joints exactly once"
            )
        if total_vertices > _MAX_TOTAL_VERTICES:
            raise RobotMaskProviderError(
                "collision geometry exceeds the bounded vertex budget"
            )
        payload = {
            "joint_names": list(joint_names),
            "links": [link.payload() for link in links],
            "schema": self.schema,
            "side": self.side,
            "source": self.source,
            "urdf_sha256": self.urdf_sha256,
        }
        digest = _semantic_sha256(payload)
        if self.geometry_sha256 != digest:
            raise RobotMaskProviderError(
                "collision geometry SHA-256 does not match preprocessed bytes"
            )
        if self.geometry_release_id != f"openarm_geometry-{digest[:20]}":
            raise RobotMaskProviderError(
                "collision geometry release ID does not match its content"
            )


@dataclass(frozen=True)
class OpenArmToolGeometry:
    """Commissioned bounded tool geometry attached to an OpenArm link."""

    schema: str
    source: str
    parent_link_index: int
    transform_parent_from_tool: np.ndarray
    vertices_tool_m: np.ndarray
    tool_release_id: str
    tool_sha256: str

    @classmethod
    def create(
        cls,
        *,
        source: str,
        parent_link_index: int,
        transform_parent_from_tool: np.ndarray,
        vertices_tool_m: np.ndarray,
    ) -> "OpenArmToolGeometry":
        transform = _validate_transform(
            transform_parent_from_tool, label="tool attachment transform"
        )
        vertices = _validate_vertices(
            vertices_tool_m, label="tool collision vertices"
        )
        payload = {
            "parent_link_index": parent_link_index,
            "schema": "wam_openarm_tool_geometry_v1",
            "source": source,
            "transform_parent_from_tool": _array_payload(transform),
            "vertices_tool_m": _array_payload(vertices),
        }
        digest = _semantic_sha256(payload)
        return cls(
            schema="wam_openarm_tool_geometry_v1",
            source=source,
            parent_link_index=parent_link_index,
            transform_parent_from_tool=transform,
            vertices_tool_m=vertices,
            tool_release_id=f"openarm_tool-{digest[:20]}",
            tool_sha256=digest,
        )

    def __post_init__(self) -> None:
        if self.schema != "wam_openarm_tool_geometry_v1":
            raise RobotMaskProviderError("tool geometry schema is unsupported")
        if self.source != "commissioned_openarm_stethoscope_tool":
            raise RobotMaskProviderError(
                "tool geometry is not the commissioned stethoscope tool"
            )
        if type(self.parent_link_index) is not int or self.parent_link_index < 0:
            raise RobotMaskProviderError("tool parent link index is invalid")
        transform = _validate_transform(
            self.transform_parent_from_tool, label="tool attachment transform"
        )
        vertices = _validate_vertices(
            self.vertices_tool_m, label="tool collision vertices"
        )
        object.__setattr__(self, "transform_parent_from_tool", transform)
        object.__setattr__(self, "vertices_tool_m", vertices)
        payload = {
            "parent_link_index": self.parent_link_index,
            "schema": self.schema,
            "source": self.source,
            "transform_parent_from_tool": _array_payload(transform),
            "vertices_tool_m": _array_payload(vertices),
        }
        digest = _semantic_sha256(payload)
        if self.tool_sha256 != digest:
            raise RobotMaskProviderError(
                "tool geometry SHA-256 does not match preprocessed bytes"
            )
        if self.tool_release_id != f"openarm_tool-{digest[:20]}":
            raise RobotMaskProviderError(
                "tool geometry release ID does not match its content"
            )


@dataclass(frozen=True)
class ActivatedHandEye:
    """Completed camera-from-follower-base calibration and covariance."""

    camera_id: str
    side: str
    transform_camera_from_follower_base: np.ndarray
    covariance_se3: np.ndarray
    status: str
    handeye_release_id: str
    handeye_sha256: str

    @property
    def capability_schema(self) -> str:
        return LEGACY_HANDEYE_CAPABILITY_SCHEMA

    @property
    def calibration_tier(self) -> str:
        return LEGACY_HANDEYE_CALIBRATION_TIER

    @property
    def policy_sha256(self) -> str:
        return _semantic_sha256({
            "autonomous_targeting_authorized": True,
            "calibration_artifact_sha256": self.handeye_sha256,
            "calibration_status": self.status,
            "calibration_tier": self.calibration_tier,
            "capability_schema": self.capability_schema,
            "provisional": False,
            "schema": "wam_legacy_handeye_capability_policy_v1",
        })

    @classmethod
    def create(
        cls,
        *,
        camera_id: str,
        side: str,
        transform_camera_from_follower_base: np.ndarray,
        covariance_se3: np.ndarray,
        status: str,
    ) -> "ActivatedHandEye":
        transform = _validate_transform(
            transform_camera_from_follower_base,
            label="hand-eye camera-from-follower-base transform",
        )
        covariance = _readonly_array(covariance_se3)
        payload = {
            "camera_id": camera_id,
            "covariance_order": "rotation_rad_then_translation_m",
            "covariance_se3": _array_payload(covariance),
            "schema": "wam_openarm_handeye_v1",
            "side": side,
            "status": status,
            "transform_camera_from_follower_base": _array_payload(transform),
        }
        digest = _semantic_sha256(payload)
        return cls(
            camera_id=camera_id,
            side=side,
            transform_camera_from_follower_base=transform,
            covariance_se3=covariance,
            status=status,
            handeye_release_id=f"handeye-{digest[:20]}",
            handeye_sha256=digest,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise RobotMaskProviderError("hand-eye camera identity is missing")
        if self.side != "left":
            raise RobotMaskProviderError("hand-eye must target the left follower")
        transform = _validate_transform(
            self.transform_camera_from_follower_base,
            label="hand-eye camera-from-follower-base transform",
        )
        covariance = _readonly_array(self.covariance_se3)
        if (
            covariance.shape != (6, 6)
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, atol=1e-12)
        ):
            raise RobotMaskProviderError(
                "hand-eye covariance must be finite symmetric 6x6"
            )
        eigenvalues = np.linalg.eigvalsh(covariance)
        if float(np.min(eigenvalues)) < -1e-12:
            raise RobotMaskProviderError("hand-eye covariance is not PSD")
        if float(np.max(eigenvalues)) > 0.25:
            raise RobotMaskProviderError(
                "hand-eye covariance exceeds the commissioning envelope"
            )
        if self.status not in ("STAGING", "COMPLETE"):
            raise RobotMaskProviderError("hand-eye activation status is invalid")
        object.__setattr__(
            self, "transform_camera_from_follower_base", transform
        )
        object.__setattr__(self, "covariance_se3", covariance)
        payload = {
            "camera_id": self.camera_id,
            "covariance_order": "rotation_rad_then_translation_m",
            "covariance_se3": _array_payload(covariance),
            "schema": "wam_openarm_handeye_v1",
            "side": self.side,
            "status": self.status,
            "transform_camera_from_follower_base": _array_payload(transform),
        }
        digest = _semantic_sha256(payload)
        if self.handeye_sha256 != digest:
            raise RobotMaskProviderError(
                "hand-eye SHA-256 does not match calibration content"
            )
        if self.handeye_release_id != f"handeye-{digest[:20]}":
            raise RobotMaskProviderError(
                "hand-eye release ID does not match calibration content"
            )


def _provisional_handeye_policy_payload(
    *,
    camera_id: str,
    side: str,
    source_frame: str,
    target_frame: str,
    transform_camera_from_follower_base: np.ndarray,
    transform_follower_base_from_camera: np.ndarray,
    covariance_se3: np.ndarray,
    covariance_order: Tuple[str, ...],
    covariance_definition: str,
    covariance_bound_transform: str,
    covariance_scope: str,
    lifecycle_status: str,
    calibration_status: str,
    provisional: bool,
    quality_gate_passed: bool,
    failed_quality_gate_m: float,
    workspace_1sigma_median: float,
    workspace_1sigma_max: float,
    systematic_bias_status: str,
    systematic_bias_bound_m: Optional[float],
    provisional_policy_allowance_m: float,
    authorized_scopes: Tuple[str, ...],
    autonomous_targeting_authorized: bool,
    tip_offset_frame: str,
    tip_offset_tcp_m: np.ndarray,
    source_run_id: str,
    code_release: str,
    acceptance_note: str,
    handeye_sha256: str,
) -> dict:
    return {
        "acceptance_note": acceptance_note,
        "authorized_scopes": list(authorized_scopes),
        "autonomous_targeting_authorized": autonomous_targeting_authorized,
        "calibration_status": calibration_status,
        "camera_id": camera_id,
        "code_release": code_release,
        "covariance_bound_transform": covariance_bound_transform,
        "covariance_definition": covariance_definition,
        "covariance_order": list(covariance_order),
        "covariance_scope": covariance_scope,
        "covariance_se3": _array_payload(covariance_se3),
        "failed_quality_gate_m": failed_quality_gate_m,
        "handeye_artifact_sha256": handeye_sha256,
        "lifecycle_status": lifecycle_status,
        "provisional": provisional,
        "provisional_policy_allowance_m": provisional_policy_allowance_m,
        "quality_gate_passed": quality_gate_passed,
        "schema": "wam_provisional_robot_mask_handeye_capability_v1",
        "side": side,
        "source_frame": source_frame,
        "source_run_id": source_run_id,
        "systematic_bias_bound_m": systematic_bias_bound_m,
        "systematic_bias_status": systematic_bias_status,
        "target_frame": target_frame,
        "tip_offset_frame": tip_offset_frame,
        "tip_offset_tcp_m": _array_payload(tip_offset_tcp_m),
        "transform_camera_from_follower_base": _array_payload(
            transform_camera_from_follower_base
        ),
        "transform_follower_base_from_camera": _array_payload(
            transform_follower_base_from_camera
        ),
        "workspace_1sigma_m": {
            "max": workspace_1sigma_max,
            "median": workspace_1sigma_median,
        },
    }


@dataclass(frozen=True)
class ProvisionalRobotMaskHandEye:
    """User-accepted below-gate hand-eye capability for robot masking only.

    ``lifecycle_status=COMPLETE`` only proves immutable publication.  It does
    not mean that the calibration passed the 3 mm quality gate.  The source
    covariance remains bound to ``base<-camera`` exactly as issued.
    """

    camera_id: str
    side: str
    source_frame: str
    target_frame: str
    transform_camera_from_follower_base: np.ndarray
    transform_follower_base_from_camera: np.ndarray
    covariance_se3: np.ndarray
    covariance_order: Tuple[str, ...]
    covariance_definition: str
    covariance_bound_transform: str
    covariance_scope: str
    lifecycle_status: str
    calibration_status: str
    provisional: bool
    quality_gate_passed: bool
    failed_quality_gate_m: float
    workspace_1sigma_median: float
    workspace_1sigma_max: float
    systematic_bias_status: str
    systematic_bias_bound_m: Optional[float]
    provisional_policy_allowance_m: float
    authorized_scopes: Tuple[str, ...]
    autonomous_targeting_authorized: bool
    tip_offset_frame: str
    tip_offset_tcp_m: np.ndarray
    source_run_id: str
    code_release: str
    acceptance_note: str
    handeye_release_id: str
    handeye_sha256: str
    policy_sha256: str

    @property
    def capability_schema(self) -> str:
        return PROVISIONAL_HANDEYE_CAPABILITY_SCHEMA

    @property
    def calibration_tier(self) -> str:
        return PROVISIONAL_HANDEYE_CALIBRATION_TIER

    @classmethod
    def from_activation_v2(
        cls,
        payload: Mapping[str, object],
        *,
        artifact_sha256: str,
        publication_status: str,
    ) -> "ProvisionalRobotMaskHandEye":
        if type(payload) is not dict or set(payload) != _PROVISIONAL_HAND_EYE_KEYS:
            raise RobotMaskProviderError(
                "provisional hand-eye artifact schema is invalid"
            )
        artifact_sha256 = _require_sha256(
            artifact_sha256, label="provisional hand-eye artifact",
        )
        canonical_artifact_sha256 = hashlib.sha256(
            _canonical_bytes(dict(payload)) + b"\n"
        ).hexdigest()
        if canonical_artifact_sha256 != artifact_sha256:
            raise RobotMaskProviderError(
                "provisional hand-eye artifact SHA-256 is not canonical"
            )
        if payload.get("schema") != "wam_openarm_handeye_activation_v2":
            raise RobotMaskProviderError(
                "provisional hand-eye artifact schema is unsupported"
            )
        if publication_status != "COMPLETE":
            raise RobotMaskProviderError(
                "provisional hand-eye publication lifecycle is not COMPLETE"
            )
        if (
            payload.get("calibration_status") != "USER_ACCEPTED_BELOW_GATE"
            or payload.get("provisional") is not True
            or payload.get("quality_gate_passed") is not False
            or payload.get("failed_quality_gate_m") != 0.003
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye quality decision is invalid"
            )
        scopes = payload.get("authorized_scopes")
        if type(scopes) is not list or tuple(scopes) != _PROVISIONAL_HAND_EYE_SCOPES:
            raise RobotMaskProviderError(
                "provisional hand-eye authorized scope is invalid"
            )
        if payload.get("autonomous_targeting_authorized") is not False:
            raise RobotMaskProviderError(
                "provisional hand-eye autonomous targeting must remain disabled"
            )
        if (
            payload.get("source_frame") != "orbbec_color_optical_frame"
            or payload.get("target_frame") != "openarm_left_link0"
            or payload.get("side") != "left"
            or payload.get("tip_offset_frame") != "openarm_left_hand_tcp"
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye frame identity is invalid"
            )
        camera_id = payload.get("camera_id")
        if type(camera_id) is not str or not camera_id:
            raise RobotMaskProviderError(
                "provisional hand-eye camera identity is invalid"
            )
        source_run_id = payload.get("source_run_id")
        if (
            type(source_run_id) is not str
            or _PROVISIONAL_HAND_EYE_RUN_RE.fullmatch(source_run_id) is None
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye source run is invalid"
            )
        code_release = _require_release(
            payload.get("code_release"), label="provisional hand-eye code",
        )
        acceptance_note = payload.get("acceptance_note")
        if type(acceptance_note) is not str or not acceptance_note.strip():
            raise RobotMaskProviderError(
                "provisional hand-eye acceptance note is missing"
            )
        if payload.get("bootstrap_reproducibility") != {
            "missing": ["bootstrap_seed", "bootstrap_sample_count", "resample_policy"],
            "status": "INCOMPLETE",
        }:
            raise RobotMaskProviderError(
                "provisional hand-eye bootstrap provenance is incomplete or invalid"
            )
        source_evidence = payload.get("source_evidence")
        if type(source_evidence) is not dict or source_evidence.get("sample_count") != 23:
            raise RobotMaskProviderError(
                "provisional hand-eye source evidence is invalid"
            )
        if (
            payload.get("systematic_bias_status") != "NOT_CHARACTERIZED"
            or payload.get("systematic_bias_bound_m") is not None
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye systematic bias state is invalid"
            )

        follower_base_from_camera = _validate_transform(
            payload.get("transform_follower_base_from_camera"),
            label="provisional hand-eye follower-base-from-camera transform",
        )
        camera_from_follower_base = _validate_transform(
            payload.get("transform_camera_from_follower_base"),
            label="provisional hand-eye camera-from-follower-base transform",
        )
        if not np.allclose(
            camera_from_follower_base @ follower_base_from_camera,
            np.eye(4), atol=1e-9,
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye transforms are not inverse"
            )
        covariance_payload = payload.get("covariance_se3")
        if (
            type(covariance_payload) is not dict
            or set(covariance_payload) != {"bound_transform", "definition", "matrix", "order"}
            or covariance_payload.get("bound_transform")
            != "transform_follower_base_from_camera"
            or covariance_payload.get("definition")
            != _PROVISIONAL_HAND_EYE_COVARIANCE_DEFINITION
            or tuple(covariance_payload.get("order", ()))
            != _PROVISIONAL_HAND_EYE_COVARIANCE_ORDER
            or payload.get("covariance_scope") != "random_only"
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye covariance semantics are invalid"
            )
        covariance = _readonly_array(covariance_payload.get("matrix"))

        workspace = payload.get("workspace_1sigma_m")
        if type(workspace) is not dict or set(workspace) != {"max", "median"}:
            raise RobotMaskProviderError(
                "provisional hand-eye workspace precision is invalid"
            )
        workspace_median = _finite_scalar(
            workspace.get("median"), label="provisional workspace median",
        )
        workspace_max = _finite_scalar(
            workspace.get("max"), label="provisional workspace maximum",
        )
        if (
            not 0.0 < workspace_median <= workspace_max
            or workspace_median > 0.0055
            or workspace_max > _PROVISIONAL_HAND_EYE_POLICY_ALLOWANCE_M
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye workspace precision exceeds the 5 mm tier"
            )
        quality = payload.get("quality")
        if type(quality) is not dict:
            raise RobotMaskProviderError(
                "provisional hand-eye quality evidence is invalid"
            )
        try:
            claimed_median = float(quality["result_precision_mm_1sigma_median"]) / 1000.0
            claimed_max = float(quality["result_precision_mm_1sigma_max"]) / 1000.0
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotMaskProviderError(
                "provisional hand-eye quality evidence is invalid"
            ) from exc
        if (
            not math.isclose(claimed_median, workspace_median, abs_tol=5e-5)
            or not math.isclose(claimed_max, workspace_max, abs_tol=5e-5)
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye quality and workspace precision disagree"
            )
        tip_offset = _readonly_array(payload.get("tip_offset_tcp_m"))
        if (
            tip_offset.shape != (3,)
            or not np.isfinite(tip_offset).all()
            or not 0.10 < float(np.linalg.norm(tip_offset)) < 0.30
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye tip offset is invalid"
            )
        policy = _provisional_handeye_policy_payload(
            camera_id=camera_id,
            side="left",
            source_frame="orbbec_color_optical_frame",
            target_frame="openarm_left_link0",
            transform_camera_from_follower_base=camera_from_follower_base,
            transform_follower_base_from_camera=follower_base_from_camera,
            covariance_se3=covariance,
            covariance_order=_PROVISIONAL_HAND_EYE_COVARIANCE_ORDER,
            covariance_definition=_PROVISIONAL_HAND_EYE_COVARIANCE_DEFINITION,
            covariance_bound_transform="transform_follower_base_from_camera",
            covariance_scope="random_only",
            lifecycle_status="COMPLETE",
            calibration_status="USER_ACCEPTED_BELOW_GATE",
            provisional=True,
            quality_gate_passed=False,
            failed_quality_gate_m=0.003,
            workspace_1sigma_median=workspace_median,
            workspace_1sigma_max=workspace_max,
            systematic_bias_status="NOT_CHARACTERIZED",
            systematic_bias_bound_m=None,
            provisional_policy_allowance_m=(
                _PROVISIONAL_HAND_EYE_POLICY_ALLOWANCE_M
            ),
            authorized_scopes=_PROVISIONAL_HAND_EYE_SCOPES,
            autonomous_targeting_authorized=False,
            tip_offset_frame="openarm_left_hand_tcp",
            tip_offset_tcp_m=tip_offset,
            source_run_id=source_run_id,
            code_release=code_release,
            acceptance_note=acceptance_note,
            handeye_sha256=artifact_sha256,
        )
        return cls(
            camera_id=camera_id,
            side="left",
            source_frame="orbbec_color_optical_frame",
            target_frame="openarm_left_link0",
            transform_camera_from_follower_base=camera_from_follower_base,
            transform_follower_base_from_camera=follower_base_from_camera,
            covariance_se3=covariance,
            covariance_order=_PROVISIONAL_HAND_EYE_COVARIANCE_ORDER,
            covariance_definition=_PROVISIONAL_HAND_EYE_COVARIANCE_DEFINITION,
            covariance_bound_transform="transform_follower_base_from_camera",
            covariance_scope="random_only",
            lifecycle_status="COMPLETE",
            calibration_status="USER_ACCEPTED_BELOW_GATE",
            provisional=True,
            quality_gate_passed=False,
            failed_quality_gate_m=0.003,
            workspace_1sigma_median=workspace_median,
            workspace_1sigma_max=workspace_max,
            systematic_bias_status="NOT_CHARACTERIZED",
            systematic_bias_bound_m=None,
            provisional_policy_allowance_m=(
                _PROVISIONAL_HAND_EYE_POLICY_ALLOWANCE_M
            ),
            authorized_scopes=_PROVISIONAL_HAND_EYE_SCOPES,
            autonomous_targeting_authorized=False,
            tip_offset_frame="openarm_left_hand_tcp",
            tip_offset_tcp_m=tip_offset,
            source_run_id=source_run_id,
            code_release=code_release,
            acceptance_note=acceptance_note,
            handeye_release_id="handeye_activation-" + artifact_sha256[:20],
            handeye_sha256=artifact_sha256,
            policy_sha256=_semantic_sha256(policy),
        )

    def __post_init__(self) -> None:
        transform_camera = _validate_transform(
            self.transform_camera_from_follower_base,
            label="provisional hand-eye camera-from-follower-base transform",
        )
        transform_base = _validate_transform(
            self.transform_follower_base_from_camera,
            label="provisional hand-eye follower-base-from-camera transform",
        )
        covariance = _readonly_array(self.covariance_se3)
        tip_offset = _readonly_array(self.tip_offset_tcp_m)
        object.__setattr__(
            self, "transform_camera_from_follower_base", transform_camera,
        )
        object.__setattr__(
            self, "transform_follower_base_from_camera", transform_base,
        )
        object.__setattr__(self, "covariance_se3", covariance)
        object.__setattr__(self, "tip_offset_tcp_m", tip_offset)
        object.__setattr__(self, "covariance_order", tuple(self.covariance_order))
        object.__setattr__(self, "authorized_scopes", tuple(self.authorized_scopes))
        if not np.allclose(transform_camera @ transform_base, np.eye(4), atol=1e-9):
            raise RobotMaskProviderError(
                "provisional hand-eye transforms are not inverse"
            )
        if (
            covariance.shape != (6, 6)
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, atol=1e-12)
            or float(np.min(np.linalg.eigvalsh(covariance))) < -1e-12
            or float(np.max(np.linalg.eigvalsh(covariance))) <= 0.0
            or float(np.max(np.linalg.eigvalsh(covariance))) > 0.25
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye covariance is invalid"
            )
        if (
            self.lifecycle_status != "COMPLETE"
            or self.calibration_status != "USER_ACCEPTED_BELOW_GATE"
            or self.provisional is not True
            or self.quality_gate_passed is not False
            or self.failed_quality_gate_m != 0.003
            or self.authorized_scopes != _PROVISIONAL_HAND_EYE_SCOPES
            or self.autonomous_targeting_authorized is not False
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye capability policy is invalid"
            )
        if (
            self.covariance_order != _PROVISIONAL_HAND_EYE_COVARIANCE_ORDER
            or self.covariance_definition
            != _PROVISIONAL_HAND_EYE_COVARIANCE_DEFINITION
            or self.covariance_bound_transform
            != "transform_follower_base_from_camera"
            or self.covariance_scope != "random_only"
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye covariance semantics are invalid"
            )
        if (
            self.side != "left"
            or self.source_frame != "orbbec_color_optical_frame"
            or self.target_frame != "openarm_left_link0"
            or self.tip_offset_frame != "openarm_left_hand_tcp"
            or self.systematic_bias_status != "NOT_CHARACTERIZED"
            or self.systematic_bias_bound_m is not None
            or self.provisional_policy_allowance_m
            != _PROVISIONAL_HAND_EYE_POLICY_ALLOWANCE_M
            or not 0.0 < self.workspace_1sigma_median
            <= self.workspace_1sigma_max
            <= _PROVISIONAL_HAND_EYE_POLICY_ALLOWANCE_M
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye frame/uncertainty policy is invalid"
            )
        _require_sha256(self.handeye_sha256, label="provisional hand-eye")
        if self.handeye_release_id != "handeye_activation-" + self.handeye_sha256[:20]:
            raise RobotMaskProviderError(
                "provisional hand-eye release ID does not match artifact content"
            )
        policy = _provisional_handeye_policy_payload(
            camera_id=self.camera_id,
            side=self.side,
            source_frame=self.source_frame,
            target_frame=self.target_frame,
            transform_camera_from_follower_base=transform_camera,
            transform_follower_base_from_camera=transform_base,
            covariance_se3=covariance,
            covariance_order=self.covariance_order,
            covariance_definition=self.covariance_definition,
            covariance_bound_transform=self.covariance_bound_transform,
            covariance_scope=self.covariance_scope,
            lifecycle_status=self.lifecycle_status,
            calibration_status=self.calibration_status,
            provisional=self.provisional,
            quality_gate_passed=self.quality_gate_passed,
            failed_quality_gate_m=self.failed_quality_gate_m,
            workspace_1sigma_median=self.workspace_1sigma_median,
            workspace_1sigma_max=self.workspace_1sigma_max,
            systematic_bias_status=self.systematic_bias_status,
            systematic_bias_bound_m=self.systematic_bias_bound_m,
            provisional_policy_allowance_m=self.provisional_policy_allowance_m,
            authorized_scopes=self.authorized_scopes,
            autonomous_targeting_authorized=(
                self.autonomous_targeting_authorized
            ),
            tip_offset_frame=self.tip_offset_frame,
            tip_offset_tcp_m=tip_offset,
            source_run_id=self.source_run_id,
            code_release=self.code_release,
            acceptance_note=self.acceptance_note,
            handeye_sha256=self.handeye_sha256,
        )
        if self.policy_sha256 != _semantic_sha256(policy):
            raise RobotMaskProviderError(
                "provisional hand-eye policy SHA-256 does not match content"
            )

    def require_scope(self, scope: str) -> None:
        if scope not in self.authorized_scopes:
            raise RobotMaskProviderError(
                "provisional hand-eye does not authorize scope: " + str(scope)
            )


RobotMaskHandEye = Union[ActivatedHandEye, ProvisionalRobotMaskHandEye]


@dataclass(frozen=True)
class CameraProjectionProfile:
    """Activated pinhole projection for RGB-aligned depth."""

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
    ) -> "CameraProjectionProfile":
        payload = {
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
        digest = _semantic_sha256(payload)
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
            camera_projection_sha256=digest,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise RobotMaskProviderError("projection camera identity is missing")
        _require_sha256(
            self.camera_profile_sha256, label="projection camera profile"
        )
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or not 16 <= self.width <= 8192
            or not 16 <= self.height <= 8192
        ):
            raise RobotMaskProviderError("projection image dimensions are invalid")
        fx = _finite_scalar(self.fx, label="projection fx")
        fy = _finite_scalar(self.fy, label="projection fy")
        cx = _finite_scalar(self.cx, label="projection cx")
        cy = _finite_scalar(self.cy, label="projection cy")
        minimum = _finite_scalar(
            self.min_projection_depth_m, label="minimum projection depth"
        )
        if (
            fx <= 0.0
            or fy <= 0.0
            or not 0.0 <= cx < self.width
            or not 0.0 <= cy < self.height
            or not 0.005 <= minimum <= 0.5
        ):
            raise RobotMaskProviderError("projection intrinsics are unsafe")
        payload = {
            "camera_id": self.camera_id,
            "camera_profile_sha256": self.camera_profile_sha256,
            "cx": self.cx,
            "cy": self.cy,
            "fx": self.fx,
            "fy": self.fy,
            "height": self.height,
            "min_projection_depth_m": self.min_projection_depth_m,
            "projection": "rectified_rgb_aligned_depth_pinhole",
            "schema": "wam_camera_projection_v1",
            "width": self.width,
        }
        digest = _semantic_sha256(payload)
        if self.camera_projection_sha256 != digest:
            raise RobotMaskProviderError(
                "camera projection SHA-256 does not match profile content"
            )


@dataclass(frozen=True)
class RobotMaskUncertaintyProfile:
    """Immutable conservative dilation bounds for one occlusion release."""

    sigma: float
    base_dilation_px: float
    link_geometry_error_m: float
    tool_geometry_error_m: float
    timing_uncertainty_s: float
    projection_error_px: float
    occlusion_profile_sha256: str

    @classmethod
    def create(
        cls,
        *,
        sigma: float,
        base_dilation_px: float,
        link_geometry_error_m: float,
        tool_geometry_error_m: float,
        timing_uncertainty_s: float,
        projection_error_px: float,
    ) -> "RobotMaskUncertaintyProfile":
        payload = {
            "base_dilation_px": base_dilation_px,
            "link_geometry_error_m": link_geometry_error_m,
            "projection_error_px": projection_error_px,
            "schema": "wam_robot_mask_uncertainty_v1",
            "sigma": sigma,
            "timing_uncertainty_s": timing_uncertainty_s,
            "tool_geometry_error_m": tool_geometry_error_m,
        }
        digest = _semantic_sha256(payload)
        return cls(
            sigma=sigma,
            base_dilation_px=base_dilation_px,
            link_geometry_error_m=link_geometry_error_m,
            tool_geometry_error_m=tool_geometry_error_m,
            timing_uncertainty_s=timing_uncertainty_s,
            projection_error_px=projection_error_px,
            occlusion_profile_sha256=digest,
        )

    def __post_init__(self) -> None:
        sigma = _finite_scalar(self.sigma, label="uncertainty sigma")
        base = _finite_scalar(
            self.base_dilation_px, label="base mask dilation"
        )
        link_error = _finite_scalar(
            self.link_geometry_error_m, label="link geometry error"
        )
        tool_error = _finite_scalar(
            self.tool_geometry_error_m, label="tool geometry error"
        )
        timing = _finite_scalar(
            self.timing_uncertainty_s, label="timing uncertainty"
        )
        projection = _finite_scalar(
            self.projection_error_px, label="projection error"
        )
        if (
            not 1.0 <= sigma <= 6.0
            or not 0.0 <= base <= 100.0
            or not 0.0 <= link_error <= 0.1
            or not 0.0 <= tool_error <= 0.1
            or not 0.0 <= timing <= 0.02
            or not 0.0 <= projection <= 20.0
        ):
            raise RobotMaskProviderError(
                "robot mask uncertainty profile exceeds safety bounds"
            )
        payload = {
            "base_dilation_px": self.base_dilation_px,
            "link_geometry_error_m": self.link_geometry_error_m,
            "projection_error_px": self.projection_error_px,
            "schema": "wam_robot_mask_uncertainty_v1",
            "sigma": self.sigma,
            "timing_uncertainty_s": self.timing_uncertainty_s,
            "tool_geometry_error_m": self.tool_geometry_error_m,
        }
        digest = _semantic_sha256(payload)
        if self.occlusion_profile_sha256 != digest:
            raise RobotMaskProviderError(
                "occlusion profile SHA-256 does not match bound content"
            )


class ProductionRobotMaskProvider:
    """Create one provenance-bound mask for a complete camera exposure."""

    def __init__(
        self,
        *,
        activated_identity: ActivatedOcclusionIdentity,
        geometry: OpenArmCollisionGeometry,
        tool: OpenArmToolGeometry,
        handeye: RobotMaskHandEye,
        projection: CameraProjectionProfile,
        uncertainty_profile: RobotMaskUncertaintyProfile,
    ):
        try:
            if not isinstance(activated_identity, ActivatedOcclusionIdentity):
                raise RobotMaskProviderError(
                    "activated occlusion identity is missing"
                )
            if not isinstance(geometry, OpenArmCollisionGeometry):
                raise RobotMaskProviderError(
                    "official OpenArm collision geometry artifact is missing"
                )
            if not isinstance(tool, OpenArmToolGeometry):
                raise RobotMaskProviderError(
                    "commissioned OpenArm tool geometry artifact is missing"
                )
            if not isinstance(
                handeye, (ActivatedHandEye, ProvisionalRobotMaskHandEye)
            ):
                raise RobotMaskProviderError(
                    "activated hand-eye artifact is missing"
                )
            if not isinstance(projection, CameraProjectionProfile):
                raise RobotMaskProviderError(
                    "activated camera projection profile is missing"
                )
            if not isinstance(
                uncertainty_profile, RobotMaskUncertaintyProfile
            ):
                raise RobotMaskProviderError(
                    "activated robot mask uncertainty profile is missing"
                )
            if isinstance(handeye, ActivatedHandEye):
                if handeye.status != "COMPLETE":
                    raise RobotMaskProviderError(
                        "hand-eye calibration is not COMPLETE"
                    )
            else:
                if handeye.lifecycle_status != "COMPLETE":
                    raise RobotMaskProviderError(
                        "provisional hand-eye lifecycle is not COMPLETE"
                    )
                handeye.require_scope("conservative_robot_mask")
                if handeye.autonomous_targeting_authorized is not False:
                    raise RobotMaskProviderError(
                        "provisional hand-eye cannot authorize autonomous targeting"
                    )
            if not 0 <= tool.parent_link_index < len(geometry.links):
                raise RobotMaskProviderError(
                    "tool attachment is outside the collision chain"
                )
            expected = (
                (projection.camera_id, activated_identity.camera_id, "camera"),
                (
                    projection.camera_profile_sha256,
                    activated_identity.camera_profile_sha256,
                    "camera profile",
                ),
                (
                    geometry.geometry_release_id,
                    activated_identity.geometry_release_id,
                    "geometry release",
                ),
                (
                    geometry.geometry_sha256,
                    activated_identity.geometry_sha256,
                    "geometry",
                ),
                (
                    tool.tool_release_id,
                    activated_identity.tool_release_id,
                    "tool release",
                ),
                (
                    tool.tool_sha256,
                    activated_identity.tool_sha256,
                    "tool",
                ),
                (
                    handeye.handeye_release_id,
                    activated_identity.handeye_release_id,
                    "hand-eye release",
                ),
                (
                    handeye.handeye_sha256,
                    activated_identity.handeye_sha256,
                    "hand-eye",
                ),
                (
                    handeye.capability_schema,
                    activated_identity.handeye_capability_schema,
                    "hand-eye capability schema",
                ),
                (
                    handeye.calibration_tier,
                    activated_identity.handeye_calibration_tier,
                    "hand-eye calibration tier",
                ),
                (
                    handeye.policy_sha256,
                    activated_identity.handeye_policy_sha256,
                    "hand-eye policy",
                ),
                (
                    projection.camera_projection_sha256,
                    activated_identity.camera_projection_sha256,
                    "camera projection",
                ),
                (
                    uncertainty_profile.occlusion_profile_sha256,
                    activated_identity.occlusion_profile_sha256,
                    "occlusion profile",
                ),
            )
            mismatch = next(
                (label for actual, pinned, label in expected if actual != pinned),
                None,
            )
            if mismatch is not None:
                raise RobotMaskProviderError(
                    f"{mismatch} does not match the activated identity"
                )
            if (
                handeye.camera_id != projection.camera_id
                or handeye.side != geometry.side
            ):
                raise RobotMaskProviderError(
                    "hand-eye frame does not match camera/OpenArm geometry"
                )
        except RobotMaskProviderError:
            raise
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            raise RobotMaskProviderError(
                "robot mask provider activation is invalid"
            ) from exc

        self.activated_identity = activated_identity
        self.geometry = geometry
        self.tool = tool
        self.handeye = handeye
        self.projection = projection
        self.uncertainty_profile = uncertainty_profile

        covariance = np.asarray(handeye.covariance_se3, dtype=np.float64)
        self._dilation_kernels: dict = {}
        self._provisional_handeye = isinstance(
            handeye, ProvisionalRobotMaskHandEye
        )
        self._rotation_sigma_rad = (
            uncertainty_profile.sigma
            * math.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(covariance[:3, :3])))))
        )
        self._translation_sigma_m = (
            uncertainty_profile.sigma
            * math.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(covariance[3:, 3:])))))
        )
        self._ancestor_joint_indices = tuple(
            self._link_ancestor_joints(index)
            for index in range(len(geometry.links))
        )

    def _link_ancestor_joints(self, link_index: int) -> Tuple[int, ...]:
        result = []
        current = link_index
        while current >= 0:
            link = self.geometry.links[current]
            if link.joint_index is not None:
                result.append(link.joint_index)
            current = link.parent_index
        return tuple(reversed(result))

    def _validate_evidence(
        self,
        exposure: CameraExposureEvidence,
        trajectory: BracketedExactControlTrajectory,
    ) -> None:
        if not isinstance(exposure, CameraExposureEvidence):
            raise RobotMaskProviderError("camera exposure evidence is missing")
        if not isinstance(trajectory, BracketedExactControlTrajectory):
            raise RobotMaskProviderError(
                "exact control trajectory v2 is required"
            )
        if (
            type(exposure.frame_index) is not int
            or exposure.frame_index < 0
            or exposure.camera_id != self.projection.camera_id
            or exposure.camera_profile_sha256
            != self.projection.camera_profile_sha256
            or exposure.camera_profile_sha256
            != self.activated_identity.camera_profile_sha256
            or exposure.camera_latency_sha256
            != self.activated_identity.camera_latency_sha256
            or exposure.alignment_sha256
            != self.activated_identity.alignment_sha256
        ):
            raise RobotMaskProviderError(
                "camera exposure provenance does not match activation"
            )
        for seconds, nanoseconds, label in (
            (
                exposure.camera_host_t_s,
                exposure.camera_host_monotonic_ns,
                "camera host",
            ),
            (
                exposure.exposure_lower_host_t_s,
                exposure.exposure_lower_monotonic_ns,
                "exposure lower",
            ),
            (
                exposure.exposure_nominal_host_t_s,
                exposure.exposure_nominal_monotonic_ns,
                "exposure nominal",
            ),
            (
                exposure.exposure_upper_host_t_s,
                exposure.exposure_upper_monotonic_ns,
                "exposure upper",
            ),
        ):
            value = _finite_scalar(seconds, label=f"{label} time")
            if (
                type(nanoseconds) is not int
                or nanoseconds <= 0
                or int(round(value * _NS_PER_SECOND)) != nanoseconds
            ):
                raise RobotMaskProviderError(
                    f"{label} seconds/nanoseconds disagree"
                )
        _finite_scalar(
            exposure.camera_device_t_ms, label="camera device timestamp"
        )
        if not (
            exposure.exposure_lower_monotonic_ns
            < exposure.exposure_nominal_monotonic_ns
            < exposure.exposure_upper_monotonic_ns
            < exposure.camera_host_monotonic_ns
        ):
            raise RobotMaskProviderError(
                "camera exposure interval is not strictly ordered"
            )
        if (
            exposure.exposure_upper_monotonic_ns
            - exposure.exposure_lower_monotonic_ns
            > self.activated_identity.max_exposure_interval_ns
            or exposure.camera_host_monotonic_ns
            - exposure.exposure_upper_monotonic_ns
            > self.activated_identity.max_camera_to_exposure_age_ns
        ):
            raise RobotMaskProviderError(
                "camera exposure interval exceeds activated timing bounds"
            )
        if (
            trajectory.exposure_lower_ns
            != exposure.exposure_lower_monotonic_ns
            or trajectory.exposure_nominal_ns
            != exposure.exposure_nominal_monotonic_ns
            or trajectory.exposure_upper_ns
            != exposure.exposure_upper_monotonic_ns
            or len(trajectory.records) < 2
            or trajectory.before.monotonic_ns
            > exposure.exposure_lower_monotonic_ns
            or trajectory.after.monotonic_ns
            < exposure.exposure_upper_monotonic_ns
            or not 0 < trajectory.max_gap_ns <= 2_000_000
        ):
            raise RobotMaskProviderError(
                "robot trajectory does not bracket the exact exposure interval"
                " (exposure lower/nominal/upper="
                f"{exposure.exposure_lower_monotonic_ns}/"
                f"{exposure.exposure_nominal_monotonic_ns}/"
                f"{exposure.exposure_upper_monotonic_ns}"
                f" trajectory lnu={trajectory.exposure_lower_ns}/"
                f"{trajectory.exposure_nominal_ns}/{trajectory.exposure_upper_ns}"
                f" records={len(trajectory.records)}"
                f" before={trajectory.before.monotonic_ns}"
                f" after={trajectory.after.monotonic_ns}"
                f" max_gap_ns={trajectory.max_gap_ns})"
            )
        previous = None
        publisher_identity = trajectory.publisher_identity
        source_closure = None
        for packet in trajectory.records:
            if (
                not isinstance(packet, ExactControlRecord)
                or packet.schema not in (
                    EXACT_CONTROL_RECORD_SCHEMA,
                    EXACT_CONTROL_RECORD_SCHEMA_V3,
                    EXACT_CONTROL_RECORD_SCHEMA_V4,
                )
                or type(packet.control_tick_seq) is not int
                or not 0 <= packet.control_tick_seq < 2**64
                or type(packet.observation_monotonic_ns) is not int
                or not 0 < packet.observation_monotonic_ns < 2**63
                or type(packet.reference_generation) is not int
                or type(packet.envelope_seq) is not int
                or not 0 <= packet.envelope_seq < 2**64
                or any(
                    type(timestamp_ns) is not int
                    for timestamp_ns in (
                        packet.reference_source_monotonic_ns,
                        packet.control_call_begin_monotonic_ns,
                        packet.arm_call_return_monotonic_ns,
                        packet.gripper_call_begin_monotonic_ns,
                        packet.gripper_call_return_monotonic_ns,
                        packet.control_call_end_monotonic_ns,
                        packet.publisher_monotonic_ns,
                    )
                )
                or type(packet.reference_snapshot_valid) is not bool
                # Reference-provenance constraints apply ONLY when the tick
                # claims a valid snapshot (2026-08-11): the mask is pure
                # forward kinematics over OBSERVED joint state; a tick whose
                # reference read collided (seqlock, common under shared-box
                # load) still has exact observations and must not void the
                # bracket — requiring validity here deadlocked model start
                # against the advisory reference-run tolerance (every bracket
                # rejected, model never came up, start never authorized).
                or (
                    packet.reference_snapshot_valid
                    and not (
                        0 < packet.reference_generation < 2**64
                        and 0 < packet.reference_source_monotonic_ns
                        <= packet.control_call_begin_monotonic_ns
                    )
                )
                or not (
                    packet.observation_monotonic_ns
                    <= packet.control_call_begin_monotonic_ns
                    <= packet.arm_call_return_monotonic_ns
                    <= packet.gripper_call_begin_monotonic_ns
                    <= packet.gripper_call_return_monotonic_ns
                    <= packet.control_call_end_monotonic_ns
                    <= packet.publisher_monotonic_ns
                )
            ):
                raise RobotMaskProviderError(
                    "exact control record schema/timing is invalid"
                )
            follower_q = np.asarray(packet.follower_q_rad, dtype=np.float64)
            follower_qd = np.asarray(
                packet.follower_qd_rad_s, dtype=np.float64
            )
            action = packet.action
            if not isinstance(action, ExactFollowerAction):
                raise RobotMaskProviderError(
                    "exact control v2 action is missing"
                )
            action_vectors = tuple(
                np.asarray(values, dtype=np.float64)
                for values in (
                    action.arm_kp,
                    action.arm_kd,
                    action.arm_q,
                    action.arm_dq,
                    action.arm_tau,
                )
            )
            action_scalars = (
                action.gripper_kp,
                action.gripper_kd,
                action.gripper_q,
                action.gripper_dq,
                action.gripper_tau,
            )
            if (
                packet.side != "left"
                or packet.controller_mode != "unilateral"
                or packet.leader_can_interface != "can1"
                or packet.follower_can_interface != "can3"
                or packet.joint_names != _LEFT_JOINT_NAMES
                or packet.urdf_sha256 != self.geometry.urdf_sha256
                or follower_q.shape != (7,)
                or follower_qd.shape != (7,)
                or not np.isfinite(follower_q).all()
                or not np.isfinite(follower_qd).all()
                or any(values.shape != (7,) for values in action_vectors)
                or any(not np.isfinite(values).all() for values in action_vectors)
                or any(not math.isfinite(float(value)) for value in action_scalars)
            ):
                raise RobotMaskProviderError(
                    "robot telemetry is not the activated left CAN1-to-CAN3 OpenArm pair"
                )
            for actual, pinned, label in (
                (
                    packet.joint_mapper_sha256,
                    self.activated_identity.joint_mapper_sha256,
                    "joint mapper",
                ),
                (
                    packet.pair_contract_sha256,
                    self.activated_identity.pair_contract_sha256,
                    "pair contract",
                ),
                (
                    packet.controller_binary_sha256,
                    self.activated_identity.controller_binary_sha256,
                    "controller binary",
                ),
            ):
                if actual != pinned:
                    raise RobotMaskProviderError(
                        f"{label} SHA-256 does not match the activated occlusion identity"
                    )
            current_source_closure = (
                _require_sha256(
                    packet.controller_source_manifest_sha256,
                    label="controller source manifest",
                ),
                _require_sha256(
                    packet.follower_config_sha256,
                    label="follower config",
                ),
                _require_sha256(
                    packet.openarm_sdk_sha256,
                    label="OpenArm SDK",
                ),
            )
            if source_closure is None:
                source_closure = current_source_closure
            elif current_source_closure != source_closure:
                labels = (
                    "controller source manifest",
                    "follower config",
                    "OpenArm SDK",
                )
                changed_label = next(
                    label
                    for label, previous_value, current_value in zip(
                        labels, source_closure, current_source_closure
                    )
                    if previous_value != current_value
                )
                raise RobotMaskProviderError(
                    f"{changed_label} SHA-256 changed inside exact control trajectory"
                )
            if packet.publisher_identity != publisher_identity:
                raise RobotMaskProviderError(
                    "exact control publisher provenance changed inside trajectory"
                )
            if previous is not None and (
                packet.seq != previous.seq + 1
                or packet.monotonic_ns <= previous.monotonic_ns
            ):
                raise RobotMaskProviderError(
                    "robot trajectory sequence/clock continuity is invalid"
                )
            previous = packet

    def _forward_kinematics(
        self, state: InterpolatedFollowerObservation
    ) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        if not isinstance(state, InterpolatedFollowerObservation):
            raise RobotMaskProviderError(
                "robot geometry requires an interpolated follower observation"
            )
        q = np.asarray(state.follower_q_rad, dtype=np.float64)
        qd = np.asarray(state.follower_qd_rad_s, dtype=np.float64)
        if (
            q.shape != (7,)
            or qd.shape != (7,)
            or not np.isfinite(q).all()
            or not np.isfinite(qd).all()
        ):
            raise RobotMaskProviderError(
                "OpenArm follower state is not finite seven-joint telemetry"
            )
        transforms = []
        joint_origins = [None] * 7
        joint_axes = [None] * 7
        for link in self.geometry.links:
            parent = (
                np.eye(4, dtype=np.float64)
                if link.parent_index == -1
                else transforms[link.parent_index]
            )
            joint_frame = parent @ link.transform_parent_from_joint_zero
            transform = joint_frame
            if link.joint_index is not None:
                joint_index = link.joint_index
                joint_origins[joint_index] = np.asarray(
                    joint_frame[:3, 3], dtype=np.float64
                )
                joint_axes[joint_index] = np.asarray(
                    joint_frame[:3, :3] @ link.joint_axis,
                    dtype=np.float64,
                )
                transform = joint_frame @ _rotation(
                    link.joint_axis, q[joint_index]
                )
            if not np.isfinite(transform).all():
                raise RobotMaskProviderError(
                    "OpenArm forward kinematics produced non-finite SE(3)"
                )
            transforms.append(transform)
        if any(value is None for value in joint_origins + joint_axes):
            raise RobotMaskProviderError(
                "OpenArm forward kinematics omitted a joint"
            )
        return (
            tuple(transforms),
            tuple(joint_origins),
            tuple(joint_axes),
        )

    @staticmethod
    def _transform_vertices(
        transform: np.ndarray, vertices: np.ndarray
    ) -> np.ndarray:
        result = (
            transform[:3, :3] @ np.asarray(vertices, dtype=np.float64).T
        ).T + transform[:3, 3]
        if not np.isfinite(result).all():
            raise RobotMaskProviderError(
                "OpenArm forward kinematics produced non-finite vertices"
            )
        return result

    def _dilation_kernel(self, radius_px: int) -> np.ndarray:
        cached = self._dilation_kernels.get(int(radius_px))
        if cached is None:
            cached = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * int(radius_px) + 1, 2 * int(radius_px) + 1),
            )
            self._dilation_kernels[int(radius_px)] = cached
        return cached

    @staticmethod
    def _point_sigma_reference(
        vertices_camera: np.ndarray,
        rotation: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """Scalar per-vertex oracle — the original loop, kept verbatim as the
        golden reference for the batched implementation below."""

        point_sigma = 0.0
        for point_camera in vertices_camera:
            skew = np.asarray(
                (
                    (0.0, -point_camera[2], point_camera[1]),
                    (point_camera[2], 0.0, -point_camera[0]),
                    (-point_camera[1], point_camera[0], 0.0),
                ),
                dtype=np.float64,
            )
            jacobian = np.hstack((skew, -rotation.T))
            point_covariance = jacobian @ covariance @ jacobian.T
            eigenvalue = float(np.max(np.linalg.eigvalsh(point_covariance)))
            if not math.isfinite(eigenvalue) or eigenvalue < -1e-12:
                raise RobotMaskProviderError(
                    "provisional hand-eye point covariance is invalid"
                )
            point_sigma = max(point_sigma, math.sqrt(max(0.0, eigenvalue)))
        return point_sigma

    @staticmethod
    def _point_sigma_batched(
        vertices_camera: np.ndarray,
        rotation: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """Batched point-sigma: one gufunc eigvalsh over (N,3,3) stacks.

        Bit-identical to :meth:`_point_sigma_reference` on the production
        BLAS (empirically verified per-eigenvalue and per-result on linbox
        numpy 2.4.4/AVX512; the in-tree property test re-proves it on every
        run): the same LAPACK routine runs per 3x3 matrix, sqrt is monotone
        and correctly rounded so max-then-sqrt equals the running max of
        sqrts, and the validity checks mirror the scalar messages exactly.
        21.6 ms -> 1.6 ms per 3216-vertex state on the production machine.
        """

        count = int(vertices_camera.shape[0])
        if count == 0:
            return 0.0
        x = vertices_camera[:, 0]
        y = vertices_camera[:, 1]
        z = vertices_camera[:, 2]
        zeros = np.zeros(count, dtype=np.float64)
        skew = np.empty((count, 3, 3), dtype=np.float64)
        skew[:, 0, 0] = zeros; skew[:, 0, 1] = -z;    skew[:, 0, 2] = y
        skew[:, 1, 0] = z;     skew[:, 1, 1] = zeros; skew[:, 1, 2] = -x
        skew[:, 2, 0] = -y;    skew[:, 2, 1] = x;     skew[:, 2, 2] = zeros
        jacobian = np.concatenate(
            (
                skew,
                np.broadcast_to(
                    -rotation.T, (count, 3, 3),
                ),
            ),
            axis=2,
        )
        point_covariance = jacobian @ covariance @ jacobian.transpose(0, 2, 1)
        eigenvalues = np.linalg.eigvalsh(point_covariance)[:, -1]
        if not bool(np.isfinite(eigenvalues).all()) or bool(
            np.any(eigenvalues < -1e-12)
        ):
            raise RobotMaskProviderError(
                "provisional hand-eye point covariance is invalid"
            )
        return math.sqrt(max(0.0, float(np.max(eigenvalues))))

    def _metric_bound(
        self,
        *,
        vertices_base: np.ndarray,
        ancestor_joints: Tuple[int, ...],
        joint_origins: Tuple[np.ndarray, ...],
        joint_axes: Tuple[np.ndarray, ...],
        joint_velocity: np.ndarray,
        component_error_m: float,
        vertices_camera: Optional[np.ndarray] = None,
    ) -> float:
        max_base_radius = float(np.max(np.linalg.norm(vertices_base, axis=1)))
        timing_bound = 0.0
        for joint_index in ancestor_joints:
            axis = joint_axes[joint_index]
            origin = joint_origins[joint_index]
            offsets = vertices_base - origin
            radii = np.linalg.norm(np.cross(axis, offsets), axis=1)
            timing_bound += (
                abs(float(joint_velocity[joint_index]))
                * self.uncertainty_profile.timing_uncertainty_s
                * float(np.max(radii))
            )
        if self._provisional_handeye:
            handeye = self.handeye
            if not isinstance(handeye, ProvisionalRobotMaskHandEye):
                raise RobotMaskProviderError(
                    "provisional hand-eye capability type changed"
                )
            if vertices_camera is None:
                vertices_camera = self._transform_vertices(
                    handeye.transform_camera_from_follower_base,
                    vertices_base,
                )
            rotation = handeye.transform_follower_base_from_camera[:3, :3]
            covariance = np.asarray(handeye.covariance_se3, dtype=np.float64)
            point_sigma = self._point_sigma_batched(
                vertices_camera, rotation, covariance,
            )
            random_bound = self.uncertainty_profile.sigma * max(
                point_sigma,
                handeye.workspace_1sigma_max,
            )
            handeye_bound = (
                random_bound + handeye.provisional_policy_allowance_m
            )
        else:
            handeye_bound = (
                self._translation_sigma_m
                + self._rotation_sigma_rad * max_base_radius
            )
        result = handeye_bound + timing_bound + component_error_m
        if not math.isfinite(result) or result < 0.0:
            raise RobotMaskProviderError(
                "robot mask uncertainty bound is invalid"
            )
        return result

    def _rasterize_component(
        self,
        *,
        mask: np.ndarray,
        vertices_base: np.ndarray,
        ancestor_joints: Tuple[int, ...],
        joint_origins: Tuple[np.ndarray, ...],
        joint_axes: Tuple[np.ndarray, ...],
        joint_velocity: np.ndarray,
        component_error_m: float,
        deferred: Optional[dict] = None,
    ) -> bool:
        vertices_camera = self._transform_vertices(
            self.handeye.transform_camera_from_follower_base,
            vertices_base,
        )
        z = vertices_camera[:, 2]
        minimum_depth = self.projection.min_projection_depth_m
        metric_bound = self._metric_bound(
            vertices_base=vertices_base,
            ancestor_joints=ancestor_joints,
            joint_origins=joint_origins,
            joint_axes=joint_axes,
            joint_velocity=joint_velocity,
            component_error_m=component_error_m,
            vertices_camera=vertices_camera,
        )
        uncertainty_depth_upper = float(np.max(z)) + metric_bound
        if uncertainty_depth_upper < minimum_depth:
            return False          # entirely behind/too close: cannot occlude the image
        # Near-plane clipping (2026-08-11): vertices at/behind the plane cannot
        # occlude the image. A component STRADDLING the plane (proximal links
        # physically mounted beside the camera, live on this rig) used to
        # reject EVERY bracket — the tracker could never start. Project only
        # the in-front vertices; a link close to the lens then projects to a
        # huge hull and falls into the conservative full-mask hatch below,
        # which is the correct occlusion semantics.
        in_front = z > minimum_depth
        if not bool(np.any(in_front)):
            return False
        vertices_camera = vertices_camera[in_front]
        z = z[in_front]
        uncertainty_depth_lower = max(
            float(np.min(z)) - metric_bound, minimum_depth,
        )

        u = self.projection.fx * vertices_camera[:, 0] / z + self.projection.cx
        v = self.projection.fy * vertices_camera[:, 1] / z + self.projection.cy
        projected = np.column_stack((u, v))
        if not np.isfinite(projected).all() or np.max(np.abs(projected)) > 1e7:
            raise RobotMaskProviderError(
                "robot/tool projection is non-finite or numerically unsafe"
            )
        raw_radius_px = (
            self.uncertainty_profile.base_dilation_px
            + self.uncertainty_profile.projection_error_px
            + max(self.projection.fx, self.projection.fy)
            * metric_bound
            / uncertainty_depth_lower
        )
        if not math.isfinite(raw_radius_px) or raw_radius_px < 0.0:
            raise RobotMaskProviderError(
                "robot/tool projected uncertainty radius is invalid"
            )
        if (
            float(np.max(u)) + raw_radius_px < 0.0
            or float(np.min(u)) - raw_radius_px >= mask.shape[1]
            or float(np.max(v)) + raw_radius_px < 0.0
            or float(np.min(v)) - raw_radius_px >= mask.shape[0]
        ):
            return False

        # 2026-08-14 (third sealed forensics): the former nominal-outside
        # full-mask branch fired on EVERY frame — the arm's lower links are
        # always outside the camera view while their uncertainty discs touch
        # it, so the mask was the full frame regardless of both hatch
        # revisions and perception stayed blind (constant all-True mask sha
        # across sessions). The painter now rasterizes in extended hull
        # coordinates (off-frame fill exists before dilation, only the
        # frame-intersecting window is written back), which is exactly
        # conservative for off-frame geometry, so no full-mask fallback is
        # needed for it. Only a radius so large the dilated disc spans the
        # frame anyway still short-circuits to the honest full mask.
        if raw_radius_px > _FULL_MASK_COST_HATCH_RADIUS_PX:
            mask[:] = True
            return True

        radius_px = int(math.ceil(raw_radius_px))
        hull = cv2.convexHull(
            np.rint(projected).astype(np.int32).reshape((-1, 1, 2))
        ).reshape((-1, 2))
        if len(hull) == 0:
            raise RobotMaskProviderError(
                "robot/tool projected convex hull is empty"
            )
        return self._paint_dilated_hull(mask, hull, radius_px, deferred)

    def _paint_dilated_hull(
        self,
        mask: np.ndarray,
        hull: np.ndarray,
        radius_px: int,
        deferred: Optional[dict] = None,
    ) -> bool:
        # ROI-cropped rasterization + dilation (2026-08-12): dilating a
        # full 1920x1080 frame with elliptical kernels of radius 25-55 px
        # cost 12-85 ms PER COMPONENT and dominated the per-frame mask
        # budget. Bitwise-identical reformulation: painted pixels are a
        # subset of the hull bounding box, and the frame-sized buffer
        # clipped out-of-frame pixels before dilation (they never existed,
        # so they never dilated back in). Therefore: paint into the
        # hull-bbox-INTERSECT-frame viewport, zero-pad by the kernel
        # radius (zeros contribute nothing to dilation), dilate the small
        # buffer, and OR the frame-clipped window back into the mask.
        height, width = mask.shape
        pad = int(radius_px) if radius_px > 0 else 0
        if pad > _PYRAMID_DILATION_MIN_RADIUS_PX:
            # Large radii never rasterize at fine resolution: the pyramid
            # paints from the hull directly on the coarse grid (the fine
            # fill below cost 242-471 ms across the real pose's off-frame
            # links and aborted armed sessions on the staleness bound).
            # With a deferral dict, same-scale components accumulate on one
            # shared coarse canvas and pay for the distance transform,
            # upsample, and frame OR exactly once per (factor, radius)
            # bucket — dilation distributes over union (byte-identity test).
            if deferred is None:
                return self._paint_pyramid_dilated(mask, hull, pad)
            f = self._pyramid_factor(pad)
            rc = -(-pad // f) + 2
            key = ("pyramid", f, rc)
            bucket = deferred.get(key)
            if bucket is None:
                ch = -(-mask.shape[0] // f)
                cw = -(-mask.shape[1] // f)
                bucket = {
                    "canvas": np.zeros(
                        (ch + 2 * rc, cw + 2 * rc), dtype=np.uint8,
                    ),
                }
                deferred[key] = bucket
            self._rasterize_coarse_hull(bucket["canvas"], hull, f, rc)
            return True
        # EXTENDED viewport (2026-08-14): rasterize in hull coordinates so
        # off-frame fill EXISTS before dilation and its in-frame reach is
        # painted — the exact conservative semantics for links outside the
        # camera view (whose crude full-mask fallback blinded perception
        # across three sealed sessions).
        #
        # Two tiers. Normal hulls rasterize on their FULL bounding box: the
        # polygon never touches the canvas edge, so the fill is byte-exact
        # versus the infinite-plane reference (OpenCV geometrically clips
        # polygons against the canvas with integer rounding, which perturbs
        # rasterization along the whole cut edge — cutting is NOT crop-safe).
        # Only pathological hulls (near-plane perspective blow-ups, bbox area
        # beyond the canvas budget) are cut to the pad-reach band, then
        # safety-inflated by 1 px so rounding artifacts can only ADD pixels:
        # strictly conservative, never blinding.
        band = pad + 2
        bx0 = int(hull[:, 0].min())
        by0 = int(hull[:, 1].min())
        bx1 = int(hull[:, 0].max()) + 1
        by1 = int(hull[:, 1].max()) + 1
        x0 = max(bx0, -band)
        y0 = max(by0, -band)
        x1 = min(bx1, width + band)
        y1 = min(by1, height + band)
        if x0 >= x1 or y0 >= y1:
            # Hull entirely beyond dilation reach of the frame.
            return False
        clipped = (x0, y0, x1, y1) != (bx0, by0, bx1, by1)
        if clipped and (
            (bx1 - bx0) * (by1 - by0)
            <= _EXTENDED_CANVAS_AREA_FACTOR * height * width
        ):
            x0, y0, x1, y1 = bx0, by0, bx1, by1
            clipped = False
        roi = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        shift = np.asarray((-x0, -y0), dtype=np.int64)
        shifted = hull + shift
        if len(hull) >= 3:
            cv2.fillConvexPoly(roi, shifted.astype(np.int32), 1)
        elif len(hull) == 2:
            cv2.line(
                roi,
                tuple(int(value) for value in shifted[0]),
                tuple(int(value) for value in shifted[1]),
                1,
                1,
            )
        else:
            cv2.circle(
                roi,
                tuple(int(value) for value in shifted[0]),
                0,
                1,
                -1,
            )
        if clipped and pad > 0:
            # Cut-polygon rasterization can lose up to 1 px along the cut
            # edge; a 3x3 inflate turns any loss into (bounded) overshoot.
            roi = cv2.dilate(
                roi, np.ones((3, 3), dtype=np.uint8), iterations=1,
            )
        component_any = bool(np.any(roi))
        canvas_interior = (
            x0 >= 0 and y0 >= 0 and x1 <= width and y1 <= height
        )
        if pad > 0 and deferred is not None and canvas_interior:
            # Radius-bucketed deferral (2026-08-12, owner-authorized): binary
            # dilation distributes over union for the same structuring
            # element, so every fill sharing an integer radius is accumulated
            # and dilated ONCE at flush time — byte-identical to dilating each
            # component separately (multi-op property test) at a fraction of
            # the dilate cost. The accumulator is frame-sized, so only
            # interior canvases may enter; extended (edge/off-frame) canvases
            # take the direct path below, which handles their coordinates.
            if not component_any:
                return False
            bucket = deferred.get(pad)
            if bucket is None:
                bucket = {
                    "acc": np.zeros((height, width), dtype=np.uint8),
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                }
                deferred[pad] = bucket
            else:
                bucket["x0"] = min(bucket["x0"], x0)
                bucket["y0"] = min(bucket["y0"], y0)
                bucket["x1"] = max(bucket["x1"], x1)
                bucket["y1"] = max(bucket["y1"], y1)
            acc_window = bucket["acc"][y0:y1, x0:x1]
            np.bitwise_or(acc_window, roi, out=acc_window)
            return True
        if pad > 0:
            roi = cv2.copyMakeBorder(
                roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0,
            )
            roi = self._euclidean_dilate(roi, radius_px)
        # Destination window in frame coordinates: viewport expanded by the
        # dilation radius, clipped to the frame.
        dx0 = max(x0 - pad, 0)
        dy0 = max(y0 - pad, 0)
        dx1 = min(x1 + pad, width)
        dy1 = min(y1 + pad, height)
        if dx0 >= dx1 or dy0 >= dy1:
            return False
        src = roi[
            dy0 - (y0 - pad):dy1 - (y0 - pad),
            dx0 - (x0 - pad):dx1 - (x0 - pad),
        ]
        window = mask[dy0:dy1, dx0:dx1]
        component_any = bool(np.any(src))
        np.logical_or(window, src, out=window)
        return component_any

    @staticmethod
    def _pyramid_factor(pad: int) -> int:
        """Adaptive pyramid factor: coarse kernel radius stays <= target+1."""
        return max(
            _PYRAMID_DILATION_FACTOR,
            -(-int(pad) // _PYRAMID_COARSE_RADIUS_TARGET),
        )

    @staticmethod
    def _rasterize_coarse_hull(canvas, hull, f, rc) -> None:
        """Fill one hull on the shared coarse canvas (floor-divided)."""
        coarse_hull = np.floor_divide(
            np.asarray(hull, dtype=np.int64), f,
        ) + rc
        if len(coarse_hull) >= 3:
            cv2.fillConvexPoly(canvas, coarse_hull.astype(np.int32), 1)
        elif len(coarse_hull) == 2:
            cv2.line(
                canvas,
                tuple(int(value) for value in coarse_hull[0]),
                tuple(int(value) for value in coarse_hull[1]),
                1,
                1,
            )
        else:
            cv2.circle(
                canvas,
                tuple(int(value) for value in coarse_hull[0]),
                0,
                1,
                -1,
            )

    @staticmethod
    def _euclidean_dilate(filled: np.ndarray, radius_px: int) -> np.ndarray:
        """Exact-Euclidean binary dilation via distance transform.

        O(N) regardless of radius, replacing O(N*k^2) elliptical-kernel
        cv2.dilate (the 129^2 kernels of the <=64 px buckets cost 30-80 ms
        per flush on frame-sized accumulators and consumed the recording
        drain margin). The closed Euclidean disc contains the rasterized
        MORPH_ELLIPSE kernel of the same radius, so coverage only grows —
        the same conservative direction as the pyramid path.
        """
        if radius_px <= 0:
            return filled
        inverted = (filled == 0).astype(np.uint8)
        distance = cv2.distanceTransform(
            inverted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
        )
        return (distance <= float(radius_px)).astype(np.uint8)

    @staticmethod
    def _finish_coarse_canvas(mask, canvas, f, rc) -> bool:
        """Inflate, exact-Euclidean dilate, upsample, OR into the frame.

        cv2.dilate with a large elliptical kernel is O(N*k^2) and cost tens
        of ms per component; distanceTransform(DIST_MASK_PRECISE) gives the
        exact Euclidean disc in O(N), and the closed disc of radius rc
        contains the rasterized MORPH_ELLIPSE kernel of the same radius, so
        coverage can only grow (superset property tests).
        """
        height, width = mask.shape
        canvas = cv2.dilate(
            canvas, np.ones((3, 3), dtype=np.uint8), iterations=1,
        )
        if not np.any(canvas):
            return False
        inverted = (canvas == 0).astype(np.uint8)
        distance = cv2.distanceTransform(
            inverted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
        )
        coverage = (distance <= float(rc)).astype(np.uint8)
        fine = cv2.resize(
            coverage,
            (coverage.shape[1] * f, coverage.shape[0] * f),
            interpolation=cv2.INTER_NEAREST,
        )
        fx0 = -rc * f
        fy0 = -rc * f
        dx0 = max(fx0, 0)
        dy0 = max(fy0, 0)
        dx1 = min(fx0 + fine.shape[1], width)
        dy1 = min(fy0 + fine.shape[0], height)
        if dx0 >= dx1 or dy0 >= dy1:
            return False
        src = fine[dy0 - fy0:dy1 - fy0, dx0 - fx0:dx1 - fx0]
        window = mask[dy0:dy1, dx0:dx1]
        painted = bool(np.any(src))
        np.logical_or(window, src.astype(np.bool_), out=window)
        return painted

    def _paint_pyramid_dilated(self, mask, hull, pad) -> bool:
        """Conservative-superset dilation for one large-radius component.

        Direct coarse rasterization + exact-Euclidean coarse dilation +
        nearest upsample (see the coarse helpers for the full cost and
        conservatism rationale; property-proved superset in tests).
        """

        f = self._pyramid_factor(pad)
        rc = -(-pad // f) + 2
        ch = -(-mask.shape[0] // f)
        cw = -(-mask.shape[1] // f)
        canvas = np.zeros((ch + 2 * rc, cw + 2 * rc), dtype=np.uint8)
        self._rasterize_coarse_hull(canvas, hull, f, rc)
        return self._finish_coarse_canvas(mask, canvas, f, rc)

    def _flush_deferred_dilations(
        self, mask: np.ndarray, deferred: dict,
    ) -> None:
        height, width = mask.shape
        for key, bucket in sorted(
            deferred.items(), key=lambda item: str(item[0]),
        ):
            if isinstance(key, tuple) and key[0] == "pyramid":
                _tag, f, rc = key
                self._finish_coarse_canvas(mask, bucket["canvas"], f, rc)
                continue
            radius_px = key
            x0, y0 = bucket["x0"], bucket["y0"]
            x1, y1 = bucket["x1"], bucket["y1"]
            pad = int(radius_px)
            roi = cv2.copyMakeBorder(
                bucket["acc"][y0:y1, x0:x1],
                pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0,
            )
            roi = self._euclidean_dilate(roi, radius_px)
            dx0 = max(x0 - pad, 0)
            dy0 = max(y0 - pad, 0)
            dx1 = min(x1 + pad, width)
            dy1 = min(y1 + pad, height)
            src = roi[
                dy0 - (y0 - pad):dy1 - (y0 - pad),
                dx0 - (x0 - pad):dx1 - (x0 - pad),
            ]
            window = mask[dy0:dy1, dx0:dx1]
            np.logical_or(window, src, out=window)

    def _build(
        self,
        *,
        exposure: CameraExposureEvidence,
        trajectory: BracketedExactControlTrajectory,
        learned_mask: Optional[np.ndarray],
    ) -> RobotMaskPacket:
        self._validate_evidence(exposure, trajectory)
        mask = np.zeros(
            (self.projection.height, self.projection.width), dtype=np.bool_
        )
        any_geometric_pixels = False
        deferred_dilations: dict = {}
        sample_times = {
            trajectory.exposure_lower_ns,
            trajectory.exposure_nominal_ns,
            trajectory.exposure_upper_ns,
        }
        sample_times.update(
            record.observation_monotonic_ns
            for record in trajectory.records
            if trajectory.exposure_lower_ns
            < record.observation_monotonic_ns
            < trajectory.exposure_upper_ns
        )
        states = tuple(
            trajectory.sample_observation_at(timestamp_ns)
            for timestamp_ns in sorted(sample_times)
        )
        if len(states) < 3:
            raise RobotMaskProviderError(
                "exposure trajectory omitted lower/nominal/upper samples"
            )
        for state in states:
            transforms, joint_origins, joint_axes = self._forward_kinematics(
                state
            )
            joint_velocity = np.asarray(
                state.follower_qd_rad_s, dtype=np.float64
            )
            for link_index, (link, transform) in enumerate(
                zip(self.geometry.links, transforms)
            ):
                vertices_base = self._transform_vertices(
                    transform, link.vertices_link_m
                )
                any_geometric_pixels |= self._rasterize_component(
                    mask=mask,
                    vertices_base=vertices_base,
                    ancestor_joints=self._ancestor_joint_indices[link_index],
                    joint_origins=joint_origins,
                    joint_axes=joint_axes,
                    joint_velocity=joint_velocity,
                    component_error_m=(
                        self.uncertainty_profile.link_geometry_error_m
                    ),
                    deferred=deferred_dilations,
                )
            tool_transform = (
                transforms[self.tool.parent_link_index]
                @ self.tool.transform_parent_from_tool
            )
            tool_vertices_base = self._transform_vertices(
                tool_transform, self.tool.vertices_tool_m
            )
            any_geometric_pixels |= self._rasterize_component(
                mask=mask,
                vertices_base=tool_vertices_base,
                ancestor_joints=self._ancestor_joint_indices[
                    self.tool.parent_link_index
                ],
                joint_origins=joint_origins,
                joint_axes=joint_axes,
                joint_velocity=joint_velocity,
                component_error_m=(
                    self.uncertainty_profile.tool_geometry_error_m
                ),
                deferred=deferred_dilations,
            )
        self._flush_deferred_dilations(mask, deferred_dilations)

        learned_pixels = False
        if learned_mask is not None:
            learned = np.asarray(learned_mask)
            if (
                learned.dtype != np.bool_
                or learned.shape != mask.shape
                or learned.ndim != 2
            ):
                raise RobotMaskProviderError(
                    "learned mask must be bool and match the activated camera shape"
                )
            learned_pixels = bool(np.any(learned))
            mask |= learned

        reason = (
            "interval_swept_openarm_robot_tool_mask"
            if any_geometric_pixels or learned_pixels
            else "robot_and_tool_proven_outside_camera_fov"
        )
        mask_sha256 = hashlib.sha256(
            np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
        ).hexdigest()
        return RobotMaskPacket(
            frame_index=exposure.frame_index,
            camera_host_t_s=exposure.camera_host_t_s,
            camera_host_monotonic_ns=exposure.camera_host_monotonic_ns,
            camera_device_t_ms=exposure.camera_device_t_ms,
            camera_id=exposure.camera_id,
            camera_profile_sha256=exposure.camera_profile_sha256,
            exposure_lower_host_t_s=exposure.exposure_lower_host_t_s,
            exposure_nominal_host_t_s=exposure.exposure_nominal_host_t_s,
            exposure_upper_host_t_s=exposure.exposure_upper_host_t_s,
            exposure_lower_monotonic_ns=exposure.exposure_lower_monotonic_ns,
            exposure_nominal_monotonic_ns=exposure.exposure_nominal_monotonic_ns,
            exposure_upper_monotonic_ns=exposure.exposure_upper_monotonic_ns,
            mask=mask,
            mask_sha256=mask_sha256,
            masked_pixel_count=int(np.count_nonzero(mask)),
            joint_sequence_first=trajectory.sequence_first,
            joint_sequence_last=trajectory.sequence_last,
            joint_time_first_s=trajectory.before.monotonic_ns / _NS_PER_SECOND,
            joint_time_last_s=trajectory.after.monotonic_ns / _NS_PER_SECOND,
            joint_time_first_monotonic_ns=trajectory.before.monotonic_ns,
            joint_time_last_monotonic_ns=trajectory.after.monotonic_ns,
            joint_max_gap_s=trajectory.max_gap_ns / _NS_PER_SECOND,
            joint_max_gap_ns=trajectory.max_gap_ns,
            geometry_release_id=self.geometry.geometry_release_id,
            geometry_sha256=self.geometry.geometry_sha256,
            tool_release_id=self.tool.tool_release_id,
            tool_sha256=self.tool.tool_sha256,
            handeye_release_id=self.handeye.handeye_release_id,
            handeye_sha256=self.handeye.handeye_sha256,
            handeye_capability_schema=self.handeye.capability_schema,
            handeye_calibration_tier=self.handeye.calibration_tier,
            handeye_policy_sha256=self.handeye.policy_sha256,
            camera_latency_sha256=exposure.camera_latency_sha256,
            camera_projection_sha256=(
                self.projection.camera_projection_sha256
            ),
            occlusion_profile_sha256=(
                self.uncertainty_profile.occlusion_profile_sha256
            ),
            alignment_sha256=exposure.alignment_sha256,
            joint_publisher_identity=trajectory.publisher_identity,
            status="OK",
            reason=reason,
            handeye_calibration_status=(
                self.handeye.calibration_status
                if isinstance(self.handeye, ProvisionalRobotMaskHandEye)
                else self.handeye.status
            ),
            handeye_provisional=isinstance(
                self.handeye, ProvisionalRobotMaskHandEye
            ),
            handeye_quality_gate_passed=(
                self.handeye.quality_gate_passed
                if isinstance(self.handeye, ProvisionalRobotMaskHandEye)
                else None
            ),
            handeye_workspace_1sigma_max_m=(
                self.handeye.workspace_1sigma_max
                if isinstance(self.handeye, ProvisionalRobotMaskHandEye)
                else None
            ),
            handeye_covariance_bound_transform=(
                self.handeye.covariance_bound_transform
                if isinstance(self.handeye, ProvisionalRobotMaskHandEye)
                else "transform_camera_from_follower_base"
            ),
            handeye_covariance_scope=(
                self.handeye.covariance_scope
                if isinstance(self.handeye, ProvisionalRobotMaskHandEye)
                else "legacy_production"
            ),
            handeye_systematic_bias_status=(
                self.handeye.systematic_bias_status
                if isinstance(self.handeye, ProvisionalRobotMaskHandEye)
                else "LEGACY_NOT_APPLICABLE"
            ),
            handeye_systematic_bias_bound_m=(
                self.handeye.systematic_bias_bound_m
                if isinstance(self.handeye, ProvisionalRobotMaskHandEye)
                else None
            ),
            handeye_autonomous_targeting_authorized=(
                self.handeye.autonomous_targeting_authorized
                if isinstance(self.handeye, ProvisionalRobotMaskHandEye)
                else True
            ),
            handeye_provisional_policy_allowance_m=(
                self.handeye.provisional_policy_allowance_m
                if isinstance(self.handeye, ProvisionalRobotMaskHandEye)
                else 0.0
            ),
        )

    def build(
        self,
        *,
        exposure: CameraExposureEvidence,
        trajectory: BracketedExactControlTrajectory,
        learned_mask: Optional[np.ndarray] = None,
    ) -> RobotMaskPacket:
        """Build a mask or raise; no degraded/empty fallback is permitted."""

        try:
            return self._build(
                exposure=exposure,
                trajectory=trajectory,
                learned_mask=learned_mask,
            )
        except RobotMaskProviderError:
            raise
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
            cv2.error,
            np.linalg.LinAlgError,
        ) as exc:
            raise RobotMaskProviderError(
                "robot mask generation failed closed"
            ) from exc


__all__ = [
    "ActivatedHandEye",
    "CameraProjectionProfile",
    "KinematicCollisionLink",
    "OpenArmCollisionGeometry",
    "OpenArmToolGeometry",
    "ProductionRobotMaskProvider",
    "RobotMaskProviderError",
    "RobotMaskUncertaintyProfile",
]
