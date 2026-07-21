"""Mac-local official-mesh rendering for oracle landmark-noise simulation.

This is a method upper bound, not trained-model validation.  It has no hardware,
Isaac, detector, or activation path.  It verifies the pinned converted asset
release, evaluates the official OpenArm tree, and renders RGB, metric depth,
and instance segmentation in a flat MuJoCo scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import html
import io
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import stat
from typing import Mapping, Sequence
from types import MappingProxyType
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .offline_official_mesh_assets import (
    DEFAULT_RAW_MANIFEST,
    EXPECTED_ASSET_MANIFEST_SHA256,
    EXPECTED_CONVERSION_MANIFEST_SHA256,
    MeshConversionError,
    WORKSPACE_ROOT,
    parse_fixed_obj_bytes,
    verify_conversion_release,
)
from .official_openarm_experiment import (
    LEFT_FOLLOWER_JOINTS,
    ExperimentContractError,
    POINT_NAMES,
    RenderedVisibility,
    decide_occlusion_policy,
    inspect_openarm_urdf,
    left_joint_limits,
    measure_rendered_visibility,
)


WIDTH = 640
HEIGHT = 360


def _read_only_float64_array(value: object) -> np.ndarray:
    """Return an array backed by immutable bytes, not merely a cleared flag."""

    source = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(source.tobytes(), dtype=np.float64).reshape(source.shape)


EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "7e3307a163e8b9e6c3c32cb038f442171d0d23794182cffb1aa1c1cb983ef346"
)
RUNTIME_MANIFEST_PATH = (
    WORKSPACE_ROOT
    / "docs/runs/2026-07-15-offline-official-mesh-occlusion/raw/runtime_manifest.json"
)
ORACLE_SIMULATION_STATUS = "oracle_landmark_noise_simulation"
FRAME_EVIDENCE_SCHEMA = "wam_oracle_landmark_noise_simulation_frame_evidence_v1"
FRAME_COMPLETION_SCHEMA = "wam_oracle_landmark_noise_simulation_frame_completion_v1"
COMPILED_VERTEX_TOLERANCE_M = 5.0e-7
COMPILED_NORMAL_TOLERANCE = 5.0e-7
SOURCE_INTRINSICS = MappingProxyType({
    "width": 1920,
    "height": 1080,
    "fx": 1123.173095703125,
    "fy": 1122.9962158203125,
    "cx": 955.7200927734375,
    "cy": 533.1757202148438,
})
INTRINSICS = MappingProxyType({
    "width": WIDTH,
    "height": HEIGHT,
    "fx": SOURCE_INTRINSICS["fx"] * WIDTH / SOURCE_INTRINSICS["width"],
    "fy": SOURCE_INTRINSICS["fy"] * HEIGHT / SOURCE_INTRINSICS["height"],
    "cx": SOURCE_INTRINSICS["cx"] * WIDTH / SOURCE_INTRINSICS["width"],
    "cy": SOURCE_INTRINSICS["cy"] * HEIGHT / SOURCE_INTRINSICS["height"],
})

CAMERA_POSITION_M = _read_only_float64_array([0.0, 2.45, 1.08])
CAMERA_LOOK_AT_M = _read_only_float64_array([0.0, 0.05, 0.90])
CAMERA_POSE_NAMES = ("primary", "oblique")
# A second physical camera placement for view-diversity checks.  It keeps the
# same patient-back target and intrinsics at a roughly 29-degree lateral view;
# no image-space warp or mesh scaling is involved.
OBLIQUE_CAMERA_POSITION_M = _read_only_float64_array([1.0, 1.85, 1.20])
OBLIQUE_CAMERA_LOOK_AT_M = _read_only_float64_array([0.0, 0.05, 0.90])

# Official OpenArm root placement used by the frozen reachability study:
# translation (-0.10, 0.45, 0.35) m and yaw 210 degrees.
ROBOT_WORLD_FROM_ROOT = _read_only_float64_array(
    [
        [-0.8660254037844386, 0.5, 0.0, -0.10],
        [-0.5, -0.8660254037844386, 0.0, 0.45],
        [0.0, 0.0, 1.0, 0.35],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

EXPECTED_SCENE_CONTRACT_SHA256 = (
    "ec571eb51deb21738145c63bbf5db9a6a89974bca8ca53a02131b23bcddc499c"
)
_CANONICAL_SCENE_BINDINGS = MappingProxyType(
    {
        "SOURCE_INTRINSICS": SOURCE_INTRINSICS,
        "INTRINSICS": INTRINSICS,
        "CAMERA_POSITION_M": CAMERA_POSITION_M,
        "CAMERA_LOOK_AT_M": CAMERA_LOOK_AT_M,
        "OBLIQUE_CAMERA_POSITION_M": OBLIQUE_CAMERA_POSITION_M,
        "OBLIQUE_CAMERA_LOOK_AT_M": OBLIQUE_CAMERA_LOOK_AT_M,
        "ROBOT_WORLD_FROM_ROOT": ROBOT_WORLD_FROM_ROOT,
    }
)


class OfflineSimulationError(RuntimeError):
    """The offline method scene violates a frozen simulation contract."""


def scene_contract_sha256() -> str:
    """Hash current public scene constants against an independent literal pin."""

    try:
        payload = {
            "schema": "wam_offline_scene_contract_v1",
            "width": WIDTH,
            "height": HEIGHT,
            "source_intrinsics": dict(SOURCE_INTRINSICS),
            "intrinsics": dict(INTRINSICS),
            "camera_pose_names": list(CAMERA_POSE_NAMES),
            "primary_position_m": CAMERA_POSITION_M.tolist(),
            "primary_look_at_m": CAMERA_LOOK_AT_M.tolist(),
            "oblique_position_m": OBLIQUE_CAMERA_POSITION_M.tolist(),
            "oblique_look_at_m": OBLIQUE_CAMERA_LOOK_AT_M.tolist(),
            "robot_world_from_root": ROBOT_WORLD_FROM_ROOT.tolist(),
        }
        raw = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OfflineSimulationError("canonical scene contract drifted") from exc
    return hashlib.sha256(raw).hexdigest()


def _assert_canonical_scene_contract() -> None:
    for name, expected in _CANONICAL_SCENE_BINDINGS.items():
        if globals().get(name) is not expected:
            raise OfflineSimulationError(
                f"canonical scene contract drifted: {name} was rebound"
            )
    if scene_contract_sha256() != EXPECTED_SCENE_CONTRACT_SHA256:
        raise OfflineSimulationError("canonical scene contract drifted")


@dataclass(frozen=True)
class RenderProducts:
    """One immutable set of sequentially rendered camera products."""

    rgb: np.ndarray
    depth_m: np.ndarray
    segmentation: np.ndarray
    robot_mask: np.ndarray
    human_mask: np.ndarray

    @property
    def depth_for_visibility_m(self) -> np.ndarray:
        """Return metric depth with non-model background marked invalid."""

        result = np.asarray(self.depth_m, dtype=np.float64).copy()
        result[~(self.robot_mask | self.human_mask)] = np.nan
        return result


@dataclass(frozen=True)
class AnchorProjection:
    """Immutable ten-anchor projection for the scene's physical camera pose."""

    camera_pose_name: str
    world_from_cv_camera: np.ndarray
    anchor_world_position_m: np.ndarray
    anchor_camera_position_m: np.ndarray
    projected_pixels: np.ndarray


@dataclass(frozen=True)
class OfflineFrameEvidence:
    """One oracle-simulation frame captured by :class:`OfflineMeshScene`."""

    timestamp_s: float
    conversion_release_id: str
    conversion_manifest_sha256: str
    asset_release_id: str
    asset_manifest_sha256: str
    runtime_manifest_sha256: str
    geom_provenance: Mapping[int, Mapping[str, object]]
    left_joint_positions: np.ndarray
    world_from_patient: np.ndarray
    world_from_cv_camera: np.ndarray
    anchor_names: tuple[str, ...]
    anchor_local_position_m: np.ndarray
    anchor_world_position_m: np.ndarray
    anchor_world_normal: np.ndarray
    anchor_camera_position_m: np.ndarray
    projected_pixels: np.ndarray
    visibility: tuple[RenderedVisibility, ...]
    products: RenderProducts
    support_radius_px: int
    robot_mask_dilation_px: int
    front_depth_margin_m: float
    _scene_capture_guard_hmac: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


POLICY_NAMES = (
    "direct_geometry_only",
    "indefinite_world_lock_baseline",
    "bounded_freeze_then_lost",
)
DEFAULT_MAX_FREEZE_AGE_S = 0.40
DEFAULT_MAX_FREEZE_ERROR_P95_M = 0.010


def _validate_policy_identity(
    scenario_id: object,
    target_name: object,
    wrist_index: object,
    frame_index: object,
    timestamp_s: object,
) -> tuple[str, str, int, int, float]:
    if (
        not isinstance(scenario_id, str)
        or not scenario_id
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in scenario_id)
    ):
        raise OfflineSimulationError("scenario_id is invalid")
    if target_name not in POINT_NAMES:
        raise OfflineSimulationError("target_name must be one of the ten frozen points")
    if type(wrist_index) is not int or wrist_index < 0:
        raise OfflineSimulationError("wrist_index must be a nonnegative integer")
    if type(frame_index) is not int or frame_index < 0:
        raise OfflineSimulationError("frame_index must be a nonnegative integer")
    timestamp = float(timestamp_s)
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise OfflineSimulationError("timestamp_s must be finite and nonnegative")
    return scenario_id, str(target_name), wrist_index, frame_index, timestamp


@dataclass(frozen=True)
class ObservationFrame:
    """One predeclared geometry observation; current ground truth is absent by design."""

    scenario_id: str
    target_name: str
    wrist_index: int
    frame_index: int
    timestamp_s: float
    geometry_accepted: bool
    accepted_estimate_world_m: np.ndarray | None
    accepted_error_p95_m: float | None
    patient_motion_bound_m_s: float
    visible_point_count_report_only: int
    observation_stream_sha256: str

    def __post_init__(self) -> None:
        scenario, target, wrist, frame, timestamp = _validate_policy_identity(
            self.scenario_id,
            self.target_name,
            self.wrist_index,
            self.frame_index,
            self.timestamp_s,
        )
        if type(self.geometry_accepted) is not bool:
            raise OfflineSimulationError("geometry_accepted must be a bool")
        motion_bound = float(self.patient_motion_bound_m_s)
        if not math.isfinite(motion_bound) or motion_bound < 0.0:
            raise OfflineSimulationError(
                "patient_motion_bound_m_s must be finite and nonnegative"
            )
        if (
            type(self.visible_point_count_report_only) is not int
            or not 0 <= self.visible_point_count_report_only <= len(POINT_NAMES)
        ):
            raise OfflineSimulationError(
                "visible_point_count_report_only must be an integer from zero to ten"
            )
        digest = self.observation_stream_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise OfflineSimulationError("observation_stream_sha256 is invalid")

        estimate: np.ndarray | None
        error_p95: float | None
        if self.geometry_accepted:
            if self.accepted_estimate_world_m is None or self.accepted_error_p95_m is None:
                raise OfflineSimulationError(
                    "accepted geometry requires an estimate and p95 error"
                )
            estimate = _immutable_array(
                _finite_vector(
                    self.accepted_estimate_world_m,
                    3,
                    "accepted_estimate_world_m",
                )
            )
            error_p95 = float(self.accepted_error_p95_m)
            if not math.isfinite(error_p95) or error_p95 < 0.0:
                raise OfflineSimulationError(
                    "accepted_error_p95_m must be finite and nonnegative"
                )
        else:
            if (
                self.accepted_estimate_world_m is not None
                or self.accepted_error_p95_m is not None
            ):
                raise OfflineSimulationError(
                    "rejected geometry cannot carry an accepted estimate"
                )
            estimate = None
            error_p95 = None

        object.__setattr__(self, "scenario_id", scenario)
        object.__setattr__(self, "target_name", target)
        object.__setattr__(self, "wrist_index", wrist)
        object.__setattr__(self, "frame_index", frame)
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "patient_motion_bound_m_s", motion_bound)
        object.__setattr__(self, "accepted_estimate_world_m", estimate)
        object.__setattr__(self, "accepted_error_p95_m", error_p95)


@dataclass(frozen=True)
class GroundTruthFrame:
    """Scoring-only target truth; this type is never accepted by the policy runner."""

    scenario_id: str
    target_name: str
    wrist_index: int
    frame_index: int
    timestamp_s: float
    observation_stream_sha256: str
    target_world_m: np.ndarray

    def __post_init__(self) -> None:
        scenario, target, wrist, frame, timestamp = _validate_policy_identity(
            self.scenario_id,
            self.target_name,
            self.wrist_index,
            self.frame_index,
            self.timestamp_s,
        )
        target_world = _immutable_array(
            _finite_vector(self.target_world_m, 3, "target_world_m")
        )
        digest = self.observation_stream_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise OfflineSimulationError(
                "ground-truth observation_stream_sha256 is invalid"
            )
        object.__setattr__(self, "scenario_id", scenario)
        object.__setattr__(self, "target_name", target)
        object.__setattr__(self, "wrist_index", wrist)
        object.__setattr__(self, "frame_index", frame)
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "target_world_m", target_world)


_STRATUM_STATUSES = {
    "COMPLETE",
    "IK_UNREACHABLE",
    "INSUFFICIENT_WRIST_ATTITUDES",
    "TRAJECTORY_INVALID",
}


@dataclass(frozen=True)
class RequiredStratum:
    """One predeclared matrix stratum, including failed-execution rows."""

    scenario_id: str
    target_name: str
    wrist_index: int
    observation_stream_sha256: str
    observation_seed: int
    patient_motion_seed: int
    execution_status: str
    declared_frame_count: int
    metric_eligible: bool | None = None
    metric_ineligibility_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_policy_identity(
            self.scenario_id,
            self.target_name,
            self.wrist_index,
            0,
            0.0,
        )
        digest = self.observation_stream_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise OfflineSimulationError("required-stratum stream SHA-256 is invalid")
        if type(self.observation_seed) is not int or self.observation_seed < 0:
            raise OfflineSimulationError("observation_seed must be nonnegative")
        if type(self.patient_motion_seed) is not int or self.patient_motion_seed < 0:
            raise OfflineSimulationError("patient_motion_seed must be nonnegative")
        if self.execution_status not in _STRATUM_STATUSES:
            raise OfflineSimulationError("required-stratum execution status is invalid")
        if type(self.declared_frame_count) is not int or self.declared_frame_count < 0:
            raise OfflineSimulationError("declared_frame_count must be nonnegative")
        if (self.execution_status == "COMPLETE") != (self.declared_frame_count > 0):
            raise OfflineSimulationError(
                "only a COMPLETE stratum may declare eligible frames"
            )
        metric_eligible = (
            self.execution_status == "COMPLETE"
            if self.metric_eligible is None
            else self.metric_eligible
        )
        if type(metric_eligible) is not bool:
            raise OfflineSimulationError("metric_eligible must be a bool")
        reason = self.metric_ineligibility_reason
        if metric_eligible:
            if self.execution_status != "COMPLETE" or reason is not None:
                raise OfflineSimulationError(
                    "metric-eligible stratum has inconsistent status/reason"
                )
        else:
            if reason is None:
                reason = f"execution_status_{self.execution_status.lower()}"
            if not isinstance(reason, str) or not reason:
                raise OfflineSimulationError(
                    "metric-ineligible stratum requires a reason"
                )
        object.__setattr__(self, "metric_eligible", metric_eligible)
        object.__setattr__(self, "metric_ineligibility_reason", reason)


@dataclass(frozen=True)
class PolicyFrameOutput:
    """One immutable policy output without truth or post-hoc target error."""

    policy_name: str
    scenario_id: str
    target_name: str
    wrist_index: int
    frame_index: int
    timestamp_s: float
    geometry_accepted: bool
    patient_motion_bound_m_s: float
    state: str
    estimate_world_m: np.ndarray | None
    valid: bool
    episode_id: str | None
    last_valid_age_s: float | None
    predicted_error_p95_m: float | None
    reacquired: bool
    deliberately_unsafe: bool
    visible_point_count_report_only: int
    observation_stream_sha256: str

    def __post_init__(self) -> None:
        if self.policy_name not in POLICY_NAMES:
            raise OfflineSimulationError("policy output name is invalid")
        _validate_policy_identity(
            self.scenario_id,
            self.target_name,
            self.wrist_index,
            self.frame_index,
            self.timestamp_s,
        )
        if type(self.valid) is not bool or type(self.geometry_accepted) is not bool:
            raise OfflineSimulationError("policy output boolean contract is invalid")
        if (
            not math.isfinite(float(self.patient_motion_bound_m_s))
            or self.patient_motion_bound_m_s < 0.0
        ):
            raise OfflineSimulationError("policy patient motion bound is invalid")
        if type(self.reacquired) is not bool or type(self.deliberately_unsafe) is not bool:
            raise OfflineSimulationError("policy output flag contract is invalid")
        estimate = self.estimate_world_m
        if self.valid:
            if estimate is None:
                raise OfflineSimulationError("valid policy output requires an estimate")
            immutable = _immutable_array(
                _finite_vector(estimate, 3, "policy estimate_world_m")
            )
        else:
            if estimate is not None:
                raise OfflineSimulationError("invalid policy output must be null")
            immutable = None
        if self.episode_id is not None and (
            not isinstance(self.episode_id, str)
            or len(self.episode_id) != 5
            or not self.episode_id.startswith("E")
            or not self.episode_id[1:].isdigit()
        ):
            raise OfflineSimulationError("policy episode_id is invalid")
        for label, value in (
            ("last_valid_age_s", self.last_valid_age_s),
            ("predicted_error_p95_m", self.predicted_error_p95_m),
        ):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise OfflineSimulationError(f"policy {label} is invalid")
        object.__setattr__(self, "estimate_world_m", immutable)


def _finite_vector(value: object, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise OfflineSimulationError(f"{label} must be a finite {length}-vector")
    return result


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _verified_runtime_manifest_sha256() -> str:
    """Bind this process to the exact persisted locked-runtime identity."""

    from configs.simulation.offline_openarm_mac import runtime_contract

    try:
        RUNTIME_MANIFEST_PATH.lstat()
        raw = RUNTIME_MANIFEST_PATH.read_bytes()
    except OSError as exc:
        raise OfflineSimulationError("locked runtime manifest is unavailable") from exc
    if RUNTIME_MANIFEST_PATH.is_symlink() or not os.path.isfile(RUNTIME_MANIFEST_PATH):
        raise OfflineSimulationError("locked runtime manifest must be a regular file")
    digest = _sha256_bytes(raw)
    if digest != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise OfflineSimulationError("locked runtime manifest identity changed")
    try:
        persisted = json.loads(raw.decode("utf-8"))
        current = runtime_contract.build_runtime_manifest()
    except (UnicodeDecodeError, json.JSONDecodeError, runtime_contract.RuntimeContractError) as exc:
        raise OfflineSimulationError("locked runtime verification failed") from exc
    if runtime_contract.canonical_json_bytes(persisted) != raw:
        raise OfflineSimulationError("locked runtime manifest is not canonical")
    if runtime_contract.canonical_json_bytes(current) != raw:
        raise OfflineSimulationError("current process differs from the locked runtime manifest")
    return digest


def _vector_attribute(
    element: ET.Element | None, name: str, default: str, label: str
) -> np.ndarray:
    raw = default if element is None else element.get(name, default)
    try:
        value = np.asarray([float(item) for item in str(raw).split()], dtype=np.float64)
    except ValueError as exc:
        raise OfflineSimulationError(f"URDF {label} is invalid") from exc
    if value.shape != (3,) or not np.isfinite(value).all():
        raise OfflineSimulationError(f"URDF {label} is invalid")
    return value


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    direction = _finite_vector(axis, 3, "joint axis")
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise OfflineSimulationError("URDF moving joint axis must be nonzero")
    x, y, z = direction / norm
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    complement = 1.0 - cosine
    return np.asarray(
        [
            [cosine + x * x * complement, x * y * complement - z * sine, x * z * complement + y * sine],
            [y * x * complement + z * sine, cosine + y * y * complement, y * z * complement - x * sine],
            [z * x * complement - y * sine, z * y * complement + x * sine, cosine + z * z * complement],
        ],
        dtype=np.float64,
    )


def _joint_origin(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_from_rpy(
        _vector_attribute(origin, "rpy", "0 0 0", "origin rpy")
    )
    transform[:3, 3] = _vector_attribute(origin, "xyz", "0 0 0", "origin xyz")
    return transform


def full_openarm_link_fk(raw: bytes, left_joint_positions: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate every official URDF link with a moving left and frozen right arm."""

    try:
        inventory = inspect_openarm_urdf(raw)
        limits = left_joint_limits(raw)
    except ExperimentContractError as exc:
        raise OfflineSimulationError(f"official OpenArm URDF is invalid: {exc}") from exc
    positions = _finite_vector(left_joint_positions, 7, "left joint positions")
    if np.any(positions < limits[:, 0]) or np.any(positions > limits[:, 1]):
        raise OfflineSimulationError("left joint position exceeds an official limit")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:  # pragma: no cover - guarded by inventory validation
        raise OfflineSimulationError("official OpenArm URDF is malformed") from exc

    link_names = [str(link.get("name")) for link in root.findall("link")]
    if len(link_names) != inventory.link_count or len(set(link_names)) != len(link_names):
        raise OfflineSimulationError("official OpenArm link inventory is invalid")
    joints = root.findall("joint")
    if len(joints) != inventory.joint_count:
        raise OfflineSimulationError("official OpenArm joint inventory is invalid")

    command_by_name = dict(zip(LEFT_FOLLOWER_JOINTS, positions, strict=True))
    children_by_parent: dict[str, list[ET.Element]] = {name: [] for name in link_names}
    child_names: set[str] = set()
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = None if parent is None else parent.get("link")
        child_name = None if child is None else child.get("link")
        if parent_name not in children_by_parent or child_name not in children_by_parent:
            raise OfflineSimulationError("URDF joint references an unknown link")
        if child_name in child_names:
            raise OfflineSimulationError("URDF link has more than one parent")
        child_names.add(str(child_name))
        children_by_parent[str(parent_name)].append(joint)

    roots = [name for name in link_names if name not in child_names]
    if roots != ["world"]:
        raise OfflineSimulationError("official OpenArm must have the single world root")
    transforms: dict[str, np.ndarray] = {"world": np.eye(4, dtype=np.float64)}
    pending = ["world"]
    while pending:
        parent_name = pending.pop(0)
        for joint in children_by_parent[parent_name]:
            child_name = str(joint.find("child").get("link"))
            joint_type = joint.get("type")
            transform = transforms[parent_name] @ _joint_origin(joint)
            if joint_type == "revolute":
                name = str(joint.get("name"))
                value = float(command_by_name.get(name, 0.0))
                axis = _vector_attribute(joint.find("axis"), "xyz", "0 0 0", "axis")
                motion = np.eye(4, dtype=np.float64)
                motion[:3, :3] = _axis_rotation(axis, value)
                transform = transform @ motion
            elif joint_type == "prismatic":
                # Fingers remain at the authored zero displacement for this
                # visibility-only experiment.
                axis = _vector_attribute(joint.find("axis"), "xyz", "0 0 0", "axis")
                if float(np.linalg.norm(axis)) <= 1.0e-12:
                    raise OfflineSimulationError("URDF prismatic axis must be nonzero")
            elif joint_type != "fixed":
                raise OfflineSimulationError(f"unsupported official joint type: {joint_type}")
            if not np.isfinite(transform).all():
                raise OfflineSimulationError("URDF FK produced a nonfinite transform")
            transforms[child_name] = transform
            pending.append(child_name)

    if set(transforms) != set(link_names) or len(transforms) != 24:
        raise OfflineSimulationError("official OpenArm tree is disconnected")
    # Preserve the exact official link inventory order even though traversal is
    # breadth-first across the left/right branches.
    return {name: transforms[name] for name in link_names}


def world_from_cv_camera(
    position_m: np.ndarray, look_at_m: np.ndarray, world_up: np.ndarray
) -> np.ndarray:
    """Build a proper CV camera frame with columns right, down, forward."""

    position = _finite_vector(position_m, 3, "camera position")
    target = _finite_vector(look_at_m, 3, "camera look-at")
    up_hint = _finite_vector(world_up, 3, "camera world-up")
    forward = target - position
    forward_norm = float(np.linalg.norm(forward))
    up_norm = float(np.linalg.norm(up_hint))
    if forward_norm <= 1.0e-12 or up_norm <= 1.0e-12:
        raise OfflineSimulationError("camera direction and world-up must be nonzero")
    forward /= forward_norm
    up_hint /= up_norm
    # This ordering is intentional: with a camera looking roughly along -Y,
    # screen-right is world -X.  Reversing it produces a left-handed CV frame.
    right = np.cross(forward, up_hint)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-12:
        raise OfflineSimulationError("camera direction cannot be parallel to world-up")
    right /= right_norm
    up = np.cross(right, forward)
    up /= float(np.linalg.norm(up))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.column_stack((right, -up, forward))
    result[:3, 3] = position
    if float(np.linalg.det(result[:3, :3])) <= 0.999999999:
        raise OfflineSimulationError("camera basis is not right-handed")
    return result


def _validate_camera_pose_name(camera_pose_name: object) -> str:
    if type(camera_pose_name) is not str or camera_pose_name not in CAMERA_POSE_NAMES:
        raise OfflineSimulationError(
            "camera pose name must be exactly 'primary' or 'oblique'"
        )
    return camera_pose_name


def camera_pose_transform(camera_pose_name: str) -> np.ndarray:
    """Return one declared physical camera pose as world-from-CV-camera."""

    _assert_canonical_scene_contract()
    name = _validate_camera_pose_name(camera_pose_name)
    if name == "primary":
        position = CAMERA_POSITION_M
        look_at = CAMERA_LOOK_AT_M
    else:
        position = OBLIQUE_CAMERA_POSITION_M
        look_at = OBLIQUE_CAMERA_LOOK_AT_M
    return world_from_cv_camera(
        position,
        look_at,
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )


def project_cv_points(points_cv_m: np.ndarray) -> np.ndarray:
    """Project finite positive-depth CV points using the frozen scaled K."""

    _assert_canonical_scene_contract()
    points = np.asarray(points_cv_m, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or np.any(points[:, 2] <= 0.0)
    ):
        raise OfflineSimulationError("CV projection points must be finite Nx3 with z > 0")
    return np.column_stack(
        (
            INTRINSICS["fx"] * points[:, 0] / points[:, 2] + INTRINSICS["cx"],
            INTRINSICS["fy"] * points[:, 1] / points[:, 2] + INTRINSICS["cy"],
        )
    )


def _camera_xml(camera_pose_name: str = "primary") -> str:
    world_from_cv = camera_pose_transform(camera_pose_name)
    right = world_from_cv[:3, 0]
    up = -world_from_cv[:3, 1]
    position = world_from_cv[:3, 3]
    focal = (INTRINSICS["fx"], INTRINSICS["fy"])
    principal = (WIDTH / 2.0 - INTRINSICS["cx"], HEIGHT / 2.0 - INTRINSICS["cy"])

    def values(items: Sequence[float]) -> str:
        return " ".join(format(float(value), ".17g") for value in items)

    return (
        '<camera name="offline_camera" '
        f'pos="{values(position)}" xyaxes="{values((*right, *up))}" '
        f'focalpixel="{values(focal)}" principalpixel="{values(principal)}" '
        f'sensorsize="{WIDTH} {HEIGHT}" resolution="{WIDTH} {HEIGHT}"/>'
    )


def camera_projection_probe() -> list[dict[str, object]]:
    """Cross-check analytic K against real MuJoCo segmentation centroids."""

    names = ("center", "x_negative", "x_positive", "y_negative", "y_positive")
    points_cv = np.asarray(
        [
            [0.0, 0.0, 2.0],
            [-0.24, 0.0, 2.0],
            [0.24, 0.0, 2.0],
            [0.0, -0.18, 2.0],
            [0.0, 0.18, 2.0],
        ],
        dtype=np.float64,
    )
    world_from_cv = world_from_cv_camera(
        CAMERA_POSITION_M, CAMERA_LOOK_AT_M, np.asarray([0.0, 0.0, 1.0])
    )
    points_world = (
        world_from_cv[:3, :3] @ points_cv.T + world_from_cv[:3, 3, None]
    ).T

    geoms = []
    for index, point in enumerate(points_world):
        xyz = " ".join(format(float(value), ".17g") for value in point)
        geoms.append(
            f'<geom name="probe_{index}" type="sphere" pos="{xyz}" '
            'size="0.015" rgba="1 1 1 1" contype="0" conaffinity="0"/>'
        )
    xml = (
        '<mujoco model="camera_projection_probe">'
        '<visual><global offwidth="640" offheight="360"/>'
        '<map znear="0.01" zfar="10"/></visual>'
        '<option gravity="0 0 0"/>'
        '<worldbody>'
        + "".join(geoms)
        + _camera_xml()
        + '</worldbody></mujoco>'
    )
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    except Exception as exc:
        raise OfflineSimulationError("MuJoCo camera probe could not be created") from exc
    try:
        _rgb, depth, segmentation = render_sequential_camera_products(
            renderer, data, camera_name="offline_camera"
        )
    finally:
        renderer.close()
    analytic = project_cv_points(points_cv)
    rows: list[dict[str, object]] = []
    for index, (name, expected) in enumerate(zip(names, analytic, strict=True)):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"probe_{index}")
        mask = (segmentation[..., 0] == geom_id) & (
            segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        )
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise OfflineSimulationError(f"MuJoCo camera probe {name} was not rendered")
        rendered = np.asarray([xs.mean() + 0.5, ys.mean() + 0.5], dtype=np.float64)
        nearest = int(
            np.argmin((xs + 0.5 - expected[0]) ** 2 + (ys + 0.5 - expected[1]) ** 2)
        )
        rendered_depth = float(depth[ys[nearest], xs[nearest]])
        expected_front_depth = float(points_cv[index, 2] - 0.015)
        rows.append(
            {
                "name": name,
                "point_cv_m": [float(value) for value in points_cv[index]],
                "analytic_pixel": [float(value) for value in expected],
                "rendered_pixel": [float(value) for value in rendered],
                "pixel_error": float(np.linalg.norm(rendered - expected)),
                "expected_front_depth_m": expected_front_depth,
                "rendered_front_depth_m": rendered_depth,
                "depth_error_m": abs(rendered_depth - expected_front_depth),
            }
        )
    return rows


def _validate_rigid_transform(value: object, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise OfflineSimulationError(f"{label} must be a finite 4x4 transform")
    if not np.array_equal(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
        raise OfflineSimulationError(f"{label} homogeneous row is invalid")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1.0e-10
    ):
        raise OfflineSimulationError(f"{label} rotation must be proper and orthonormal")
    return transform.copy()


def _matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation, dtype=np.float64).ravel())
    if not np.isfinite(quaternion).all() or float(np.linalg.norm(quaternion)) <= 0.0:
        raise OfflineSimulationError("pose rotation could not be converted to a quaternion")
    quaternion /= float(np.linalg.norm(quaternion))
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def decode_instance_masks(
    segmentation: np.ndarray,
    robot_geom_ids: Sequence[int],
    human_geom_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode only MuJoCo geom rows; body IDs and unknown IDs stay excluded."""

    product = np.asarray(segmentation)
    if product.ndim != 3 or product.shape[2] != 2 or not np.issubdtype(
        product.dtype, np.integer
    ):
        raise OfflineSimulationError("segmentation must be an integer HxWx2 product")
    robot_ids = frozenset(int(value) for value in robot_geom_ids)
    human_ids = frozenset(int(value) for value in human_geom_ids)
    if (
        not robot_ids
        or not human_ids
        or min((*robot_ids, *human_ids)) < 0
        or not robot_ids.isdisjoint(human_ids)
    ):
        raise OfflineSimulationError("robot/human geom IDs must be nonempty and disjoint")
    is_geom = product[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    object_ids = product[..., 0]
    robot = is_geom & np.isin(object_ids, tuple(sorted(robot_ids)))
    human = is_geom & np.isin(object_ids, tuple(sorted(human_ids)))
    return np.asarray(robot, dtype=np.bool_), np.asarray(human, dtype=np.bool_)


def render_sequential_camera_products(
    renderer: object,
    data: mujoco.MjData,
    *,
    camera_name: str = "offline_camera",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render RGB, depth, then segmentation with explicit mutually exclusive modes."""

    try:
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera=camera_name)
        rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()

        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_name)
        depth = np.asarray(renderer.render()).copy()
        renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera_name)
        segmentation = np.asarray(renderer.render(), dtype=np.int32).copy()
        return rgb, depth, segmentation
    finally:
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OfflineSimulationError("frame evidence is not canonical-JSON encodable") from exc


_SCENE_CAPTURE_GUARD_KEY = secrets.token_bytes(32)
_EXPECTED_ORACLE_ANCHOR_LOCAL_M: np.ndarray | None = None
_EXPECTED_ORACLE_ANCHOR_NORMAL: np.ndarray | None = None
_EXPECTED_ORACLE_PROVENANCE_ROWS: tuple[Mapping[str, object], ...] | None = None


def _hash_evidence_array(digest: object, label: str, value: object) -> None:
    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))


def _canonical_geom_provenance(
    provenance: Mapping[int, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    try:
        return {
            str(int(key)): dict(value)
            for key, value in sorted(provenance.items(), key=lambda item: int(item[0]))
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise OfflineSimulationError("frame geom provenance is invalid") from exc


def raw_render_sha256(evidence: OfflineFrameEvidence) -> str:
    """Hash deterministic render inputs/products, excluding event time and noise."""

    if not isinstance(evidence, OfflineFrameEvidence):
        raise OfflineSimulationError("evidence must be an OfflineFrameEvidence")
    metadata = {
        "schema": "wam_oracle_raw_render_binding_v1",
        "status": ORACLE_SIMULATION_STATUS,
        "conversion_release_id": evidence.conversion_release_id,
        "conversion_manifest_sha256": evidence.conversion_manifest_sha256,
        "asset_release_id": evidence.asset_release_id,
        "asset_manifest_sha256": evidence.asset_manifest_sha256,
        "runtime_manifest_sha256": evidence.runtime_manifest_sha256,
        "scene_contract_sha256": scene_contract_sha256(),
        "camera_intrinsics": dict(INTRINSICS),
        "geom_provenance": _canonical_geom_provenance(evidence.geom_provenance),
    }
    digest = hashlib.sha256(_canonical_json_bytes(metadata))
    for label, value in (
        ("left_joint_positions", evidence.left_joint_positions),
        ("world_from_patient", evidence.world_from_patient),
        ("world_from_cv_camera", evidence.world_from_cv_camera),
        ("rgb", evidence.products.rgb),
        ("depth_m", evidence.products.depth_m),
        ("segmentation", evidence.products.segmentation),
        ("robot_mask", evidence.products.robot_mask),
        ("human_mask", evidence.products.human_mask),
    ):
        _hash_evidence_array(digest, label, value)
    return digest.hexdigest()


def oracle_frame_evidence_sha256(evidence: OfflineFrameEvidence) -> str:
    """Hash one event binding separately from its deterministic raw render."""

    if not isinstance(evidence, OfflineFrameEvidence):
        raise OfflineSimulationError("evidence must be an OfflineFrameEvidence")
    visibility = []
    for row in evidence.visibility:
        edge = float(row.mask_edge_distance_px)
        visibility.append(
            {
                "name": row.name,
                "visible_fraction": float(row.visible_fraction),
                "robot_front_fraction": float(row.robot_front_fraction),
                "valid_depth_fraction": float(row.valid_depth_fraction),
                "mask_edge_distance_px": edge if math.isfinite(edge) else None,
                "visibility_source": row.visibility_source,
            }
        )
    metadata = {
        "schema": "wam_oracle_frame_evidence_binding_v1",
        "status": ORACLE_SIMULATION_STATUS,
        "raw_render_sha256": raw_render_sha256(evidence),
        "timestamp_s": float(evidence.timestamp_s),
        "anchor_names": tuple(evidence.anchor_names),
        "support_radius_px_report_only": int(evidence.support_radius_px),
        "robot_mask_dilation_px_report_only": int(
            evidence.robot_mask_dilation_px
        ),
        "front_depth_margin_m_report_only": float(evidence.front_depth_margin_m),
        "visibility": visibility,
    }
    digest = hashlib.sha256(_canonical_json_bytes(metadata))
    for label, value in (
        ("anchor_local_position_m", evidence.anchor_local_position_m),
        ("anchor_world_position_m", evidence.anchor_world_position_m),
        ("anchor_world_normal", evidence.anchor_world_normal),
        ("anchor_camera_position_m", evidence.anchor_camera_position_m),
        ("projected_pixels_report_only", evidence.projected_pixels),
    ):
        _hash_evidence_array(digest, label, value)
    return digest.hexdigest()


def _scene_capture_guard_hmac(evidence: OfflineFrameEvidence) -> str:
    """Return a non-security guard against accidental same-process misuse.

    Python private names and this module-local key are not a security boundary;
    an adversarial caller in this process can bypass this guard.
    """

    return hmac.new(
        _SCENE_CAPTURE_GUARD_KEY,
        oracle_frame_evidence_sha256(evidence).encode("ascii"),
        hashlib.sha256,
    ).hexdigest()


def _attach_scene_capture_guard(
    evidence: OfflineFrameEvidence,
) -> OfflineFrameEvidence:
    object.__setattr__(
        evidence,
        "_scene_capture_guard_hmac",
        _scene_capture_guard_hmac(evidence),
    )
    return evidence


def _expected_oracle_contract_arrays() -> tuple[np.ndarray, np.ndarray]:
    global _EXPECTED_ORACLE_ANCHOR_LOCAL_M, _EXPECTED_ORACLE_ANCHOR_NORMAL
    if (
        _EXPECTED_ORACLE_ANCHOR_LOCAL_M is None
        or _EXPECTED_ORACLE_ANCHOR_NORMAL is None
    ):
        manifest = _pinned_conversion_manifest()
        try:
            rows = manifest["human"]["labcoat"]["anchors"]
            local = np.asarray([row["position_m"] for row in rows], dtype=np.float64)
            normal = np.asarray([row["normal"] for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise OfflineSimulationError(
                "canonical oracle anchor contract is invalid"
            ) from exc
        if local.shape != (len(POINT_NAMES), 3) or normal.shape != (
            len(POINT_NAMES),
            3,
        ):
            raise OfflineSimulationError("canonical oracle anchor count changed")
        _EXPECTED_ORACLE_ANCHOR_LOCAL_M = _immutable_array(local)
        _EXPECTED_ORACLE_ANCHOR_NORMAL = _immutable_array(normal)
    return _EXPECTED_ORACLE_ANCHOR_LOCAL_M, _EXPECTED_ORACLE_ANCHOR_NORMAL


def _expected_oracle_provenance_rows() -> tuple[Mapping[str, object], ...]:
    global _EXPECTED_ORACLE_PROVENANCE_ROWS
    if _EXPECTED_ORACLE_PROVENANCE_ROWS is None:
        _EXPECTED_ORACLE_PROVENANCE_ROWS = tuple(
            MappingProxyType(dict(row)) for row in _pinned_geom_provenance_rows()
        )
    return _EXPECTED_ORACLE_PROVENANCE_ROWS


def validate_oracle_frame_evidence(
    evidence: OfflineFrameEvidence,
    *,
    allow_test_only_synthetic: bool = False,
) -> None:
    """Fail closed unless evidence is an exact primary-camera oracle capture.

    The scene-owned HMAC is only a non-security, same-process guard against
    accidental public construction/tampering.  Python private names are not a
    security boundary.  The guard is intentionally not persisted and cannot
    authenticate a frame in a later process.  Persisted evidence must be
    verified through manifest/file hashes and independent semantic rerender.
    """

    if not isinstance(evidence, OfflineFrameEvidence):
        raise OfflineSimulationError("evidence must be an OfflineFrameEvidence")
    if type(allow_test_only_synthetic) is not bool:
        raise OfflineSimulationError("test-only synthetic flag must be a bool")
    if not allow_test_only_synthetic:
        guard_hmac = evidence._scene_capture_guard_hmac
        if not isinstance(guard_hmac, str):
            raise OfflineSimulationError(
                "oracle evidence is not an intact scene-owned capture"
            )
        try:
            expected_guard_hmac = _scene_capture_guard_hmac(evidence)
        except (AttributeError, TypeError, ValueError, OfflineSimulationError) as exc:
            raise OfflineSimulationError(
                "oracle evidence is not an intact scene-owned capture"
            ) from exc
        if not hmac.compare_digest(guard_hmac, expected_guard_hmac):
            raise OfflineSimulationError(
                "oracle evidence is not an intact scene-owned capture"
            )

    _assert_canonical_scene_contract()
    expected_identities = {
        "conversion_release_id": (
            f"official_meshes-{EXPECTED_CONVERSION_MANIFEST_SHA256[:20]}"
        ),
        "conversion_manifest_sha256": EXPECTED_CONVERSION_MANIFEST_SHA256,
        "asset_release_id": f"official_assets-{EXPECTED_ASSET_MANIFEST_SHA256[:20]}",
        "asset_manifest_sha256": EXPECTED_ASSET_MANIFEST_SHA256,
        "runtime_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
    }
    for name, expected in expected_identities.items():
        if getattr(evidence, name) != expected:
            raise OfflineSimulationError(f"oracle evidence {name} changed")
    if not math.isfinite(float(evidence.timestamp_s)) or evidence.timestamp_s < 0.0:
        raise OfflineSimulationError("oracle evidence timestamp is invalid")

    left_q = np.asarray(evidence.left_joint_positions)
    patient = _validate_rigid_transform(
        evidence.world_from_patient, "oracle world_from_patient"
    )
    camera = _validate_rigid_transform(
        evidence.world_from_cv_camera, "oracle world_from_cv_camera"
    )
    if left_q.shape != (7,) or not np.isfinite(left_q).all():
        raise OfflineSimulationError("oracle evidence joint pose is invalid")
    primary_camera = camera_pose_transform("primary")
    if not np.allclose(camera, primary_camera, atol=1.0e-12, rtol=0.0):
        raise OfflineSimulationError("oracle evidence primary camera transform changed")
    if tuple(evidence.anchor_names) != POINT_NAMES:
        raise OfflineSimulationError("oracle evidence anchor identity changed")

    local = np.asarray(evidence.anchor_local_position_m)
    world = np.asarray(evidence.anchor_world_position_m)
    world_normal = np.asarray(evidence.anchor_world_normal)
    camera_points = np.asarray(evidence.anchor_camera_position_m)
    pixels = np.asarray(evidence.projected_pixels)
    if any(
        value.shape != (len(POINT_NAMES), 3) or not np.isfinite(value).all()
        for value in (local, world, world_normal, camera_points)
    ) or pixels.shape != (len(POINT_NAMES), 2) or not np.isfinite(pixels).all():
        raise OfflineSimulationError("oracle evidence anchor array shape changed")
    expected_local, expected_normal = _expected_oracle_contract_arrays()
    if not np.allclose(local, expected_local, atol=1.0e-9, rtol=0.0):
        raise OfflineSimulationError("oracle evidence anchor local contract changed")
    expected_world = (patient[:3, :3] @ local.T + patient[:3, 3, None]).T
    if not np.allclose(world, expected_world, atol=1.0e-12, rtol=0.0):
        raise OfflineSimulationError("oracle evidence anchor world relation changed")
    expected_world_normal = (patient[:3, :3] @ expected_normal.T).T
    if not np.allclose(
        world_normal,
        expected_world_normal,
        atol=COMPILED_NORMAL_TOLERANCE,
        rtol=0.0,
    ):
        raise OfflineSimulationError("oracle evidence anchor normal relation changed")
    expected_camera = (
        camera[:3, :3].T @ (world - camera[:3, 3]).T
    ).T
    if not np.allclose(
        camera_points, expected_camera, atol=1.0e-12, rtol=0.0
    ):
        raise OfflineSimulationError("oracle evidence anchor camera relation changed")
    expected_pixels = project_cv_points(camera_points)
    if not np.allclose(pixels, expected_pixels, atol=1.0e-12, rtol=0.0):
        raise OfflineSimulationError("oracle evidence anchor reprojection changed")

    products = evidence.products
    if not isinstance(products, RenderProducts):
        raise OfflineSimulationError("oracle evidence render products type changed")
    rgb = np.asarray(products.rgb)
    depth = np.asarray(products.depth_m)
    segmentation = np.asarray(products.segmentation)
    robot_mask = np.asarray(products.robot_mask)
    human_mask = np.asarray(products.human_mask)
    if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8:
        raise OfflineSimulationError("oracle evidence RGB shape or dtype changed")
    if (
        depth.shape != (HEIGHT, WIDTH)
        or not np.issubdtype(depth.dtype, np.floating)
        or not np.isfinite(depth).all()
        or np.any(depth <= 0.0)
    ):
        raise OfflineSimulationError("oracle evidence depth shape or values changed")
    if segmentation.shape != (HEIGHT, WIDTH, 2) or segmentation.dtype != np.int32:
        raise OfflineSimulationError("oracle evidence segmentation shape or dtype changed")
    if (
        robot_mask.shape != (HEIGHT, WIDTH)
        or human_mask.shape != (HEIGHT, WIDTH)
        or robot_mask.dtype != np.bool_
        or human_mask.dtype != np.bool_
    ):
        raise OfflineSimulationError("oracle evidence mask shape or dtype changed")
    if any(
        np.asarray(value).flags.writeable
        for value in (
            evidence.left_joint_positions,
            evidence.world_from_patient,
            evidence.world_from_cv_camera,
            evidence.anchor_local_position_m,
            evidence.anchor_world_position_m,
            evidence.anchor_world_normal,
            evidence.anchor_camera_position_m,
            evidence.projected_pixels,
            products.rgb,
            products.depth_m,
            products.segmentation,
            products.robot_mask,
            products.human_mask,
        )
    ):
        raise OfflineSimulationError("oracle evidence arrays must be immutable")

    observed_provenance = _canonical_geom_provenance(evidence.geom_provenance)
    expected_rows = _expected_oracle_provenance_rows()
    if tuple(int(key) for key in observed_provenance) != tuple(range(28)):
        raise OfflineSimulationError("oracle evidence geom provenance IDs changed")
    compiled_keys = {
        "kind",
        "identity",
        "output_path",
        "output_sha256",
        "vertex_count",
        "triangle_count",
        "compiled_vertex_max_error_m",
        "compiled_aabb_max_error_m",
        "compiled_face_indices_exact",
    }
    for geom_id, expected_row in enumerate(expected_rows):
        observed = observed_provenance[str(geom_id)]
        if set(observed) != compiled_keys or any(
            observed.get(key) != value
            for key, value in expected_row.items()
            if key != "geom_id"
        ):
            raise OfflineSimulationError("oracle evidence geom provenance changed")
        vertex_error = observed["compiled_vertex_max_error_m"]
        aabb_error = observed["compiled_aabb_max_error_m"]
        if (
            type(vertex_error) not in {int, float}
            or type(aabb_error) not in {int, float}
            or not math.isfinite(float(vertex_error))
            or not math.isfinite(float(aabb_error))
            or not 0.0 <= float(vertex_error) <= COMPILED_VERTEX_TOLERANCE_M
            or not 0.0 <= float(aabb_error) <= COMPILED_VERTEX_TOLERANCE_M
            or observed["compiled_face_indices_exact"] is not True
        ):
            raise OfflineSimulationError(
                "oracle evidence compiled geom provenance is invalid"
            )
    robot_ids = {
        int(key)
        for key, value in evidence.geom_provenance.items()
        if value.get("kind") == "robot_visual"
    }
    human_ids = {
        int(key)
        for key, value in evidence.geom_provenance.items()
        if value.get("kind") == "human"
    }
    decoded_robot, decoded_human = decode_instance_masks(
        segmentation, robot_ids, human_ids
    )
    if not np.array_equal(robot_mask, decoded_robot) or not np.array_equal(
        human_mask, decoded_human
    ):
        raise OfflineSimulationError(
            "oracle evidence segmentation and instance masks disagree"
        )
    if np.any(robot_mask & human_mask):
        raise OfflineSimulationError("oracle evidence robot/human masks overlap")
    geom_pixels = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    if np.any(
        geom_pixels
        & ~np.isin(segmentation[..., 0], np.asarray(sorted(robot_ids | human_ids)))
    ):
        raise OfflineSimulationError(
            "oracle evidence segmentation geom provenance is incomplete"
        )

    if type(evidence.support_radius_px) is not int or evidence.support_radius_px <= 0:
        raise OfflineSimulationError("oracle evidence support radius is invalid")
    if (
        type(evidence.robot_mask_dilation_px) is not int
        or evidence.robot_mask_dilation_px < 0
    ):
        raise OfflineSimulationError("oracle evidence mask dilation is invalid")
    if (
        not math.isfinite(float(evidence.front_depth_margin_m))
        or evidence.front_depth_margin_m < 0.0
    ):
        raise OfflineSimulationError("oracle evidence depth margin is invalid")
    if len(evidence.visibility) != len(POINT_NAMES):
        raise OfflineSimulationError("oracle evidence visibility count changed")
    try:
        expected_visibility = measure_rendered_visibility(
            point_names=POINT_NAMES,
            projected_pixels=pixels,
            point_depth_m=camera_points[:, 2],
            robot_mask=robot_mask,
            human_mask=human_mask,
            depth_m=products.depth_for_visibility_m,
            support_radius_px=evidence.support_radius_px,
            robot_mask_dilation_px=evidence.robot_mask_dilation_px,
            front_depth_margin_m=float(evidence.front_depth_margin_m),
        )
    except ExperimentContractError as exc:
        raise OfflineSimulationError(
            "oracle evidence visibility replay failed"
        ) from exc
    if tuple(evidence.visibility) != expected_visibility:
        raise OfflineSimulationError("oracle evidence visibility replay changed")


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _write_evidence_file(path: Path, raw: bytes) -> None:
    """Durably write one immutable staging file (fault-injection seam for tests)."""

    from scripts.perception import prepare_offline_official_assets as release_io

    release_io._write_exclusive(Path(path), raw)


def _frame_array_contract(evidence: OfflineFrameEvidence) -> Mapping[str, np.ndarray]:
    return {
        "rgb.npy": evidence.products.rgb,
        "depth_m.npy": evidence.products.depth_m,
        "segmentation.npy": evidence.products.segmentation,
        "robot_mask.npy": evidence.products.robot_mask,
        "human_mask.npy": evidence.products.human_mask,
        "anchor_local_position_m.npy": evidence.anchor_local_position_m,
        "anchor_world_position_m.npy": evidence.anchor_world_position_m,
        "anchor_world_normal.npy": evidence.anchor_world_normal,
        "anchor_camera_position_m.npy": evidence.anchor_camera_position_m,
        "projected_pixels.npy": evidence.projected_pixels,
    }


def _frame_manifest(
    evidence: OfflineFrameEvidence, files: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    visibility = []
    for row in evidence.visibility:
        edge = float(row.mask_edge_distance_px)
        visibility.append(
            {
                "name": row.name,
                "visible_fraction": float(row.visible_fraction),
                "robot_front_fraction": float(row.robot_front_fraction),
                "valid_depth_fraction": float(row.valid_depth_fraction),
                "mask_edge_distance_px": edge if math.isfinite(edge) else None,
                "visibility_source": row.visibility_source,
            }
        )
    geom_rows = []
    for geom_id, row in sorted(evidence.geom_provenance.items()):
        geom_rows.append({"geom_id": int(geom_id), **dict(row)})
    return {
        "schema": FRAME_EVIDENCE_SCHEMA,
        "status": ORACLE_SIMULATION_STATUS,
        "scope": ORACLE_SIMULATION_STATUS,
        "production_activation": False,
        "model_validation": False,
        "isaac_equivalence": False,
        "conversion_release_id": evidence.conversion_release_id,
        "conversion_manifest_sha256": evidence.conversion_manifest_sha256,
        "asset_release_id": evidence.asset_release_id,
        "asset_manifest_sha256": evidence.asset_manifest_sha256,
        "runtime_manifest_sha256": evidence.runtime_manifest_sha256,
        "geom_provenance": geom_rows,
        "timestamp_s": float(evidence.timestamp_s),
        "left_joint_positions": evidence.left_joint_positions.tolist(),
        "world_from_patient": evidence.world_from_patient.tolist(),
        "world_from_cv_camera": evidence.world_from_cv_camera.tolist(),
        "camera_intrinsics": dict(INTRINSICS),
        "anchor_names": list(evidence.anchor_names),
        "support_radius_px": evidence.support_radius_px,
        "robot_mask_dilation_px": evidence.robot_mask_dilation_px,
        "front_depth_margin_m": evidence.front_depth_margin_m,
        "visibility": visibility,
        "files": [dict(row) for row in files],
    }


def _decode_canonical_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfflineSimulationError(f"{label} is not valid JSON") from exc
    if type(value) is not dict or _canonical_json_bytes(value) != raw:
        raise OfflineSimulationError(f"{label} is not canonical JSON")
    return value


def _pinned_conversion_manifest() -> dict[str, object]:
    try:
        mode = DEFAULT_RAW_MANIFEST.lstat().st_mode
        raw = DEFAULT_RAW_MANIFEST.read_bytes()
    except OSError as exc:
        raise OfflineSimulationError("canonical conversion manifest is unavailable") from exc
    if (
        stat.S_ISLNK(mode)
        or not stat.S_ISREG(mode)
        or _sha256_bytes(raw) != EXPECTED_CONVERSION_MANIFEST_SHA256
    ):
        raise OfflineSimulationError("canonical conversion manifest identity changed")
    return _decode_canonical_json(raw, "canonical conversion manifest")


def _pinned_geom_provenance_rows() -> list[dict[str, object]]:
    manifest = _pinned_conversion_manifest()
    robot_rows = [
        row for row in manifest["robot"]["meshes"] if row["role"] == "visual"
    ]
    human_rows = list(manifest["human"]["meshes"])
    result = []
    for geom_id, row in enumerate(robot_rows):
        result.append(
            {
                "geom_id": geom_id,
                "kind": "robot_visual",
                "identity": row["link"],
                "output_path": row["output_path"],
                "output_sha256": row["output_sha256"],
                "vertex_count": row["vertex_count"],
                "triangle_count": row["triangle_count"],
            }
        )
    for offset, row in enumerate(human_rows, start=len(robot_rows)):
        result.append(
            {
                "geom_id": offset,
                "kind": "human",
                "identity": row["name"],
                "output_path": row["output_path"],
                "output_sha256": row["output_sha256"],
                "vertex_count": row["vertex_count"],
                "triangle_count": row["triangle_count"],
            }
        )
    if len(result) != 28:
        raise OfflineSimulationError("canonical geom provenance inventory changed")
    return result


def verify_frame_evidence_release(
    destination: Path, *, semantic_rerender: bool = True
) -> dict[str, object]:
    """Verify persisted hashes/contracts and optionally semantic-rerender.

    This verifier never treats the non-security process-local guard as
    cross-process evidence; the guard is absent from disk by design.  Durable
    trust comes from file/manifest hashes and the independent semantic rerender.
    """

    _assert_canonical_scene_contract()
    from scripts.perception import prepare_offline_official_assets as release_io

    root = Path(destination)
    try:
        mode = root.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise OfflineSimulationError("frame evidence release must be a real directory")
        release_io._assert_tree_no_symlinks(root)
    except (OSError, release_io.PreparationError) as exc:
        raise OfflineSimulationError("frame evidence release tree is invalid") from exc
    manifest_path = root / "frame_manifest.json"
    marker_path = root / "FRAME_COMPLETE.json"
    try:
        manifest_raw = manifest_path.read_bytes()
        marker_raw = marker_path.read_bytes()
    except OSError as exc:
        raise OfflineSimulationError("frame evidence completion files are missing") from exc
    manifest = _decode_canonical_json(manifest_raw, "frame manifest")
    marker = _decode_canonical_json(marker_raw, "frame completion marker")
    expected_top = {
        "schema",
        "status",
        "scope",
        "production_activation",
        "model_validation",
        "isaac_equivalence",
        "conversion_release_id",
        "conversion_manifest_sha256",
        "asset_release_id",
        "asset_manifest_sha256",
        "runtime_manifest_sha256",
        "geom_provenance",
        "timestamp_s",
        "left_joint_positions",
        "world_from_patient",
        "world_from_cv_camera",
        "camera_intrinsics",
        "anchor_names",
        "support_radius_px",
        "robot_mask_dilation_px",
        "front_depth_margin_m",
        "visibility",
        "files",
    }
    if set(manifest) != expected_top or manifest.get("schema") != FRAME_EVIDENCE_SCHEMA:
        raise OfflineSimulationError("frame manifest schema is invalid")
    if (
        manifest.get("status") != ORACLE_SIMULATION_STATUS
        or manifest.get("scope") != ORACLE_SIMULATION_STATUS
        or manifest.get("production_activation") is not False
        or manifest.get("model_validation") is not False
        or manifest.get("isaac_equivalence") is not False
        or manifest.get("conversion_release_id")
        != f"official_meshes-{EXPECTED_CONVERSION_MANIFEST_SHA256[:20]}"
        or manifest.get("conversion_manifest_sha256")
        != EXPECTED_CONVERSION_MANIFEST_SHA256
        or manifest.get("asset_release_id")
        != f"official_assets-{EXPECTED_ASSET_MANIFEST_SHA256[:20]}"
        or manifest.get("asset_manifest_sha256") != EXPECTED_ASSET_MANIFEST_SHA256
        or manifest.get("runtime_manifest_sha256")
        != EXPECTED_RUNTIME_MANIFEST_SHA256
    ):
        raise OfflineSimulationError("frame manifest official/runtime provenance changed")
    timestamp = manifest.get("timestamp_s")
    if type(timestamp) not in {int, float} or not math.isfinite(float(timestamp)) or timestamp < 0:
        raise OfflineSimulationError("frame timestamp is invalid")
    _finite_vector(manifest.get("left_joint_positions"), 7, "persisted left joints")
    _validate_rigid_transform(manifest.get("world_from_patient"), "persisted patient pose")
    _validate_rigid_transform(manifest.get("world_from_cv_camera"), "persisted camera pose")
    if manifest.get("camera_intrinsics") != INTRINSICS:
        raise OfflineSimulationError("persisted camera intrinsics changed")
    if tuple(manifest.get("anchor_names", ())) != POINT_NAMES:
        raise OfflineSimulationError("persisted anchor identity changed")
    geom_rows = manifest.get("geom_provenance")
    if type(geom_rows) is not list or len(geom_rows) != 28:
        raise OfflineSimulationError("persisted geom provenance is incomplete")
    if [row.get("geom_id") for row in geom_rows if type(row) is dict] != list(range(28)):
        raise OfflineSimulationError("persisted geom IDs changed")
    if sum(row.get("kind") == "robot_visual" for row in geom_rows) != 21 or sum(
        row.get("kind") == "human" for row in geom_rows
    ) != 7:
        raise OfflineSimulationError("persisted geom classes changed")
    compiled_keys = {
        "geom_id",
        "kind",
        "identity",
        "output_path",
        "output_sha256",
        "vertex_count",
        "triangle_count",
        "compiled_vertex_max_error_m",
        "compiled_aabb_max_error_m",
        "compiled_face_indices_exact",
    }
    for observed, expected in zip(
        geom_rows, _pinned_geom_provenance_rows(), strict=True
    ):
        if type(observed) is not dict or set(observed) != compiled_keys:
            raise OfflineSimulationError("persisted geom provenance row is invalid")
        if any(observed.get(key) != value for key, value in expected.items()):
            raise OfflineSimulationError("persisted geom provenance is not canonical")
        vertex_error = observed.get("compiled_vertex_max_error_m")
        aabb_error = observed.get("compiled_aabb_max_error_m")
        if (
            type(vertex_error) not in {int, float}
            or type(aabb_error) not in {int, float}
            or not math.isfinite(float(vertex_error))
            or not math.isfinite(float(aabb_error))
            or not 0.0 <= float(vertex_error) <= COMPILED_VERTEX_TOLERANCE_M
            or not 0.0 <= float(aabb_error) <= COMPILED_VERTEX_TOLERANCE_M
            or observed.get("compiled_face_indices_exact") is not True
        ):
            raise OfflineSimulationError("persisted compiled mesh evidence is invalid")
    visibility = manifest.get("visibility")
    if (
        type(visibility) is not list
        or tuple(row.get("name") for row in visibility if type(row) is dict) != POINT_NAMES
        or any(
            row.get("visibility_source") != "rendered_instance_mask_and_depth"
            for row in visibility
        )
    ):
        raise OfflineSimulationError("persisted rendered visibility is invalid")

    expected_arrays = {
        "rgb.npy": ((HEIGHT, WIDTH, 3), "uint8"),
        "depth_m.npy": ((HEIGHT, WIDTH), None),
        "segmentation.npy": ((HEIGHT, WIDTH, 2), "int32"),
        "robot_mask.npy": ((HEIGHT, WIDTH), "bool"),
        "human_mask.npy": ((HEIGHT, WIDTH), "bool"),
        "anchor_local_position_m.npy": ((10, 3), None),
        "anchor_world_position_m.npy": ((10, 3), None),
        "anchor_world_normal.npy": ((10, 3), None),
        "anchor_camera_position_m.npy": ((10, 3), None),
        "projected_pixels.npy": ((10, 2), None),
    }
    file_rows = manifest.get("files")
    if type(file_rows) is not list or [row.get("path") for row in file_rows if type(row) is dict] != list(
        expected_arrays
    ):
        raise OfflineSimulationError("frame file inventory is invalid")
    loaded: dict[str, np.ndarray] = {}
    for row in file_rows:
        if type(row) is not dict or set(row) != {"path", "sha256", "size", "shape", "dtype"}:
            raise OfflineSimulationError("frame file row schema is invalid")
        name = str(row["path"])
        path = root / name
        try:
            file_mode = path.lstat().st_mode
            raw_sha = _sha256_file(path)
            size = path.stat().st_size
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise OfflineSimulationError(f"frame array could not be verified: {name}") from exc
        if stat.S_ISLNK(file_mode) or not stat.S_ISREG(file_mode):
            raise OfflineSimulationError("frame array must be a regular file")
        expected_shape, expected_dtype = expected_arrays[name]
        if (
            row.get("sha256") != raw_sha
            or row.get("size") != size
            or row.get("shape") != list(array.shape)
            or row.get("dtype") != str(array.dtype)
            or array.shape != expected_shape
            or (expected_dtype is not None and str(array.dtype) != expected_dtype)
        ):
            raise OfflineSimulationError(f"frame array contract changed: {name}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise OfflineSimulationError(f"persisted raw array is nonfinite: {name}")
        loaded[name] = array
    robot_ids = {int(row["geom_id"]) for row in geom_rows if row["kind"] == "robot_visual"}
    human_ids = {int(row["geom_id"]) for row in geom_rows if row["kind"] == "human"}
    robot, human = decode_instance_masks(loaded["segmentation.npy"], robot_ids, human_ids)
    if not np.array_equal(robot, loaded["robot_mask.npy"]) or not np.array_equal(
        human, loaded["human_mask.npy"]
    ):
        raise OfflineSimulationError("persisted masks do not match raw segmentation")
    if np.any(loaded["depth_m.npy"] <= 0.0):
        raise OfflineSimulationError("persisted metric depth must be positive")

    conversion = _pinned_conversion_manifest()
    anchor_rows = conversion["human"]["labcoat"]["anchors"]
    pinned_local = np.asarray([row["position_m"] for row in anchor_rows], dtype=np.float64)
    pinned_normals = np.asarray([row["normal"] for row in anchor_rows], dtype=np.float64)
    local = np.asarray(loaded["anchor_local_position_m.npy"], dtype=np.float64)
    if not np.allclose(local, pinned_local, atol=1.0e-9, rtol=0.0):
        raise OfflineSimulationError("persisted local anchors differ from the pinned mesh")
    patient = _validate_rigid_transform(
        manifest["world_from_patient"], "persisted patient pose"
    )
    expected_world = (patient[:3, :3] @ local.T + patient[:3, 3, None]).T
    if not np.allclose(
        loaded["anchor_world_position_m.npy"], expected_world, atol=1.0e-12, rtol=0.0
    ):
        raise OfflineSimulationError("persisted patient-to-world anchor transfer changed")
    expected_world_normals = (patient[:3, :3] @ pinned_normals.T).T
    if not np.allclose(
        loaded["anchor_world_normal.npy"],
        expected_world_normals,
        atol=5.0e-8,
        rtol=0.0,
    ):
        raise OfflineSimulationError("persisted patient-to-world normals changed")
    frozen_world_from_cv = world_from_cv_camera(
        CAMERA_POSITION_M, CAMERA_LOOK_AT_M, np.asarray([0.0, 0.0, 1.0])
    )
    if not np.allclose(
        np.asarray(manifest["world_from_cv_camera"], dtype=np.float64),
        frozen_world_from_cv,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise OfflineSimulationError("persisted camera transform changed")
    expected_camera = (
        frozen_world_from_cv[:3, :3].T
        @ (expected_world - frozen_world_from_cv[:3, 3]).T
    ).T
    if not np.allclose(
        loaded["anchor_camera_position_m.npy"],
        expected_camera,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise OfflineSimulationError("persisted world-to-camera anchors changed")
    expected_pixels = project_cv_points(expected_camera)
    if not np.allclose(
        loaded["projected_pixels.npy"], expected_pixels, atol=1.0e-12, rtol=0.0
    ):
        raise OfflineSimulationError("persisted anchor projections changed")

    left_q = _finite_vector(
        manifest["left_joint_positions"], 7, "persisted left joints"
    )
    asset_release = (
        WORKSPACE_ROOT
        / ".wam-build/offline-official-openarm-assets/releases"
        / str(conversion["asset_release_id"])
    )
    urdf_path = asset_release / str(conversion["robot"]["urdf_path"])
    try:
        urdf_raw = urdf_path.read_bytes()
    except OSError as exc:
        raise OfflineSimulationError("pinned URDF is unavailable for frame verification") from exc
    if _sha256_bytes(urdf_raw) != conversion["robot"]["urdf_sha256"]:
        raise OfflineSimulationError("pinned URDF identity changed during frame verification")
    full_openarm_link_fk(urdf_raw, left_q)

    support_radius = manifest.get("support_radius_px")
    mask_dilation = manifest.get("robot_mask_dilation_px")
    depth_margin = manifest.get("front_depth_margin_m")
    if (
        type(support_radius) is not int
        or support_radius <= 0
        or type(mask_dilation) is not int
        or mask_dilation < 0
        or type(depth_margin) not in {int, float}
        or not math.isfinite(float(depth_margin))
        or depth_margin < 0.0
    ):
        raise OfflineSimulationError("persisted visibility tolerances are invalid")
    visibility_depth = np.asarray(loaded["depth_m.npy"], dtype=np.float64).copy()
    visibility_depth[~(robot | human)] = np.nan
    try:
        recomputed_visibility = measure_rendered_visibility(
            point_names=POINT_NAMES,
            projected_pixels=expected_pixels,
            point_depth_m=expected_camera[:, 2],
            robot_mask=robot,
            human_mask=human,
            depth_m=visibility_depth,
            support_radius_px=support_radius,
            robot_mask_dilation_px=mask_dilation,
            front_depth_margin_m=float(depth_margin),
        )
    except (ExperimentContractError, TypeError, ValueError) as exc:
        raise OfflineSimulationError("persisted visibility could not be recomputed") from exc
    for observed, recomputed in zip(
        manifest["visibility"], recomputed_visibility, strict=True
    ):
        expected_edge = float(recomputed.mask_edge_distance_px)
        observed_edge = observed.get("mask_edge_distance_px")
        if (
            observed.get("name") != recomputed.name
            or observed.get("visibility_source") != recomputed.visibility_source
            or not math.isclose(
                float(observed.get("visible_fraction")),
                float(recomputed.visible_fraction),
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                float(observed.get("robot_front_fraction")),
                float(recomputed.robot_front_fraction),
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                float(observed.get("valid_depth_fraction")),
                float(recomputed.valid_depth_fraction),
                abs_tol=1.0e-15,
            )
            or (
                (observed_edge is not None or math.isfinite(expected_edge))
                and not (
                    observed_edge is not None
                    and math.isclose(
                        float(observed_edge), expected_edge, abs_tol=1.0e-12
                    )
                )
            )
        ):
            raise OfflineSimulationError("persisted rendered visibility changed")
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    expected_files = {*expected_arrays, "frame_manifest.json", "FRAME_COMPLETE.json"}
    if actual_files != expected_files or any(path.is_dir() for path in root.iterdir()):
        raise OfflineSimulationError("frame evidence has missing or unexpected paths")
    expected_marker = {
        "schema": FRAME_COMPLETION_SCHEMA,
        "status": "complete",
        "frame_manifest_sha256": _sha256_bytes(manifest_raw),
    }
    if marker != expected_marker or marker_raw != _canonical_json_bytes(expected_marker):
        raise OfflineSimulationError("frame completion marker is invalid")
    if type(semantic_rerender) is not bool:
        raise OfflineSimulationError("semantic_rerender must be a bool")
    if semantic_rerender:
        conversion_release = (
            WORKSPACE_ROOT
            / ".wam-build/offline-official-openarm-meshes/releases"
            / f"official_meshes-{EXPECTED_CONVERSION_MANIFEST_SHA256[:20]}"
        )
        with OfflineMeshScene.from_verified_conversion(conversion_release) as scene:
            scene.set_frame(left_q, patient)
            replay = scene.capture_frame_evidence(
                float(manifest["timestamp_s"]),
                support_radius_px=support_radius,
                robot_mask_dilation_px=mask_dilation,
                front_depth_margin_m=float(depth_margin),
            )
        observed_provenance = {
            int(row["geom_id"]): {
                key: value for key, value in row.items() if key != "geom_id"
            }
            for row in geom_rows
        }
        if observed_provenance != replay.geom_provenance:
            raise OfflineSimulationError("persisted compiled geom diagnostics changed")
        replay_arrays = _frame_array_contract(replay)
        if any(
            not np.array_equal(loaded[name], replay_arrays[name])
            for name in expected_arrays
        ):
            raise OfflineSimulationError("persisted products do not match semantic rerender")
    return manifest


def persist_frame_evidence(
    evidence: OfflineFrameEvidence, destination: Path
) -> dict[str, object]:
    """Atomically publish one durable, strictly verifiable evidence frame."""

    if not isinstance(evidence, OfflineFrameEvidence):
        raise OfflineSimulationError("evidence must be an OfflineFrameEvidence")
    validate_oracle_frame_evidence(evidence)
    from scripts.perception import prepare_offline_official_assets as release_io

    output = Path(destination).absolute()
    try:
        release_io._mkdir_real(output.parent)
    except release_io.PreparationError as exc:
        raise OfflineSimulationError("frame evidence parent is invalid") from exc
    if os.path.lexists(os.fspath(output)):
        raise OfflineSimulationError("frame evidence destination must be new")
    staging = output.parent / f".{output.name}.staging-{secrets.token_hex(12)}"
    try:
        os.mkdir(staging, 0o700)
    except OSError as exc:
        raise OfflineSimulationError("frame evidence staging could not be created") from exc
    staging_identity = staging.stat()
    published = False
    files: list[dict[str, object]] = []
    try:
        for name, value in _frame_array_contract(evidence).items():
            array = np.asarray(value)
            raw = _npy_bytes(array)
            path = staging / name
            _write_evidence_file(path, raw)
            files.append(
                {
                    "path": name,
                    "sha256": _sha256_bytes(raw),
                    "size": len(raw),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            )
        manifest = _frame_manifest(evidence, files)
        manifest_raw = _canonical_json_bytes(manifest)
        _write_evidence_file(staging / "frame_manifest.json", manifest_raw)
        marker_raw = _canonical_json_bytes(
            {
                "schema": FRAME_COMPLETION_SCHEMA,
                "status": "complete",
                "frame_manifest_sha256": _sha256_bytes(manifest_raw),
            }
        )
        _write_evidence_file(staging / "FRAME_COMPLETE.json", marker_raw)
        release_io._fsync_tree(staging)
        verify_frame_evidence_release(staging, semantic_rerender=False)
        release_io._rename_no_replace(staging, output)
        published = True
        verified = verify_frame_evidence_release(output, semantic_rerender=False)
        return verified
    except Exception as exc:
        cleanup = output if published else staging
        try:
            if os.path.lexists(os.fspath(cleanup)):
                current = cleanup.lstat()
                if (current.st_dev, current.st_ino) == (
                    staging_identity.st_dev,
                    staging_identity.st_ino,
                ):
                    shutil.rmtree(cleanup)
        except OSError:
            pass
        if isinstance(exc, OfflineSimulationError) and not published:
            raise OfflineSimulationError("frame evidence persistence failed") from exc
        raise OfflineSimulationError("frame evidence persistence failed") from exc


def _quoted(value: object) -> str:
    return f'"{html.escape(str(value), quote=True)}"'


def _owned_verified_output_bytes(
    release: Path, row: Mapping[str, object]
) -> bytes:
    """Read one verified OBJ once and bind the owned bytes to its pinned row."""

    path = Path(release) / str(row.get("output_path"))
    try:
        mode = path.lstat().st_mode
        raw = path.read_bytes()
    except OSError as exc:
        raise OfflineSimulationError("verified conversion output is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise OfflineSimulationError("verified conversion output must be a regular file")
    if len(raw) != row.get("output_size") or _sha256_bytes(raw) != row.get(
        "output_sha256"
    ):
        raise OfflineSimulationError("conversion output changed after release verification")
    return raw


def _compiled_mesh_provenance(
    model: mujoco.MjModel,
    *,
    geom_name: str,
    kind: str,
    identity: str,
    row: Mapping[str, object],
    owned_raw: bytes,
) -> tuple[int, dict[str, object]]:
    geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
    if geom_id < 0:
        raise OfflineSimulationError("official scene geom name did not bind")
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise OfflineSimulationError("official scene geom did not bind a mesh")
    try:
        source_vertices, source_faces = parse_fixed_obj_bytes(owned_raw)
    except MeshConversionError as exc:
        raise OfflineSimulationError("owned official OBJ could not be decoded") from exc
    vertex_address = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_address = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    if vertex_count != len(source_vertices) or face_count != len(source_faces):
        raise OfflineSimulationError("compiled official mesh topology count changed")
    compiled_vertices = np.asarray(
        model.mesh_vert[vertex_address : vertex_address + vertex_count],
        dtype=np.float64,
    )
    compiled_faces = np.asarray(
        model.mesh_face[face_address : face_address + face_count], dtype=np.int64
    )
    mesh_rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mesh_rotation, model.geom_quat[geom_id])
    compiled_in_authored_frame = (
        compiled_vertices @ mesh_rotation.reshape(3, 3).T + model.geom_pos[geom_id]
    )
    vertex_error = float(
        np.max(np.abs(compiled_in_authored_frame - source_vertices))
    )
    face_indices_exact = bool(np.array_equal(compiled_faces, source_faces))
    compiled_aabb = np.asarray(
        [compiled_in_authored_frame.min(axis=0), compiled_in_authored_frame.max(axis=0)]
    )
    aabb_error = float(
        np.max(np.abs(compiled_aabb - np.asarray(row["output_aabb"], dtype=np.float64)))
    )
    if (
        vertex_error > COMPILED_VERTEX_TOLERANCE_M
        or aabb_error > COMPILED_VERTEX_TOLERANCE_M
        or not face_indices_exact
    ):
        raise OfflineSimulationError("compiled official mesh geometry changed")
    return geom_id, {
        "kind": kind,
        "identity": identity,
        "output_path": str(row["output_path"]),
        "output_sha256": str(row["output_sha256"]),
        "vertex_count": int(row["vertex_count"]),
        "triangle_count": int(row["triangle_count"]),
        "compiled_vertex_max_error_m": vertex_error,
        "compiled_aabb_max_error_m": aabb_error,
        "compiled_face_indices_exact": face_indices_exact,
    }


_MODEL_RENDER_ARRAYS = (
    "body_parentid",
    "body_jntadr",
    "body_jntnum",
    "jnt_type",
    "jnt_qposadr",
    "geom_type",
    "geom_dataid",
    "geom_bodyid",
    "geom_contype",
    "geom_conaffinity",
    "geom_pos",
    "geom_quat",
    "geom_rgba",
    "mesh_vertadr",
    "mesh_vertnum",
    "mesh_faceadr",
    "mesh_facenum",
    "mesh_pos",
    "mesh_quat",
    "mesh_vert",
    "mesh_face",
    "cam_bodyid",
    "cam_pos",
    "cam_quat",
    "cam_mat0",
    "cam_intrinsic",
    "cam_resolution",
    "cam_sensorsize",
    "cam_projection",
)


def _model_nested_groups(model: mujoco.MjModel) -> Mapping[str, object]:
    return {
        "vis.map": model.vis.map,
        "vis.global": model.vis.global_,
        "vis.headlight": model.vis.headlight,
        "vis.rgba": model.vis.rgba,
        "vis.scale": model.vis.scale,
        "vis.quality": model.vis.quality,
        "stat": model.stat,
        "opt": model.opt,
    }


def _nested_public_values(group: object) -> list[tuple[str, object]]:
    result = []
    for name in sorted(item for item in dir(group) if item and item[0].islower()):
        try:
            value = getattr(group, name)
        except Exception:
            continue
        if not callable(value) and (
            isinstance(value, np.ndarray)
            or isinstance(value, (bool, int, float, np.integer, np.floating))
        ):
            result.append((name, value))
    return result


def _freeze_model_arrays(model: mujoco.MjModel) -> None:
    """Make every exposed MjModel ndarray immutable to Python callers."""

    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue
        if isinstance(value, np.ndarray) and value.flags.writeable:
            value.flags.writeable = False
    for group in _model_nested_groups(model).values():
        for _name, value in _nested_public_values(group):
            if isinstance(value, np.ndarray) and value.flags.writeable:
                value.flags.writeable = False
    for name in _MODEL_RENDER_ARRAYS:
        value = getattr(model, name)
        if value.flags.writeable:
            raise OfflineSimulationError(f"MjModel render array remained writable: {name}")


def _model_render_digest(model: mujoco.MjModel) -> str:
    """Cheap per-frame guard for all mutable small render/camera state."""

    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {
                "nbody": int(model.nbody),
                "njnt": int(model.njnt),
                "ngeom": int(model.ngeom),
                "nmesh": int(model.nmesh),
                "ncam": int(model.ncam),
            }
        )
    )
    # Mesh vertices/faces are made permanently read-only above.  Their exact
    # hashes and compiled comparisons live in geom_provenance; hashing ~75 MB
    # again at every 20 Hz frame would change the experiment timing.
    small_arrays = tuple(
        name for name in _MODEL_RENDER_ARRAYS if name not in {"mesh_vert", "mesh_face"}
    )
    for name in small_arrays:
        value = np.ascontiguousarray(getattr(model, name))
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    for group_name, group in _model_nested_groups(model).items():
        for name, raw_value in _nested_public_values(group):
            value = np.asarray(raw_value)
            digest.update(f"{group_name}.{name}".encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.tobytes())
    return digest.hexdigest()


def _nested_model_arrays_are_read_only(model: mujoco.MjModel) -> bool:
    return all(
        not value.flags.writeable
        for group in _model_nested_groups(model).values()
        for _name, value in _nested_public_values(group)
        if isinstance(value, np.ndarray)
    )


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    if result.flags.writeable:
        raise OfflineSimulationError("evidence array did not become immutable")
    return result


def _immutable_render_products(products: RenderProducts) -> RenderProducts:
    return RenderProducts(
        rgb=_immutable_array(products.rgb),
        depth_m=_immutable_array(products.depth_m),
        segmentation=_immutable_array(products.segmentation),
        robot_mask=_immutable_array(products.robot_mask),
        human_mask=_immutable_array(products.human_mask),
    )


class OfflineMeshScene:
    """Flat kinematic MuJoCo scene bound to one verified conversion release."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        renderer: mujoco.Renderer,
        urdf_raw: bytes,
        robot_links: Sequence[str],
        human_meshes: Sequence[str],
        robot_geom_ids: set[int],
        human_geom_ids: set[int],
        geom_provenance: Mapping[int, Mapping[str, object]],
        anchor_names: Sequence[str],
        anchor_local_position_m: np.ndarray,
        anchor_local_normal: np.ndarray,
        conversion_release_id: str,
        conversion_manifest_sha256: str,
        asset_release_id: str,
        asset_manifest_sha256: str,
        runtime_manifest_sha256: str,
        camera_pose_name: str,
        world_from_cv_camera_transform: np.ndarray,
    ) -> None:
        self.model = model
        self.data = data
        self._renderer = renderer
        self._urdf_raw = urdf_raw
        self._robot_links = tuple(robot_links)
        self._human_meshes = tuple(human_meshes)
        self.robot_geom_ids = frozenset(robot_geom_ids)
        self.human_geom_ids = frozenset(human_geom_ids)
        self._geom_provenance = MappingProxyType(
            {
                int(key): MappingProxyType(dict(value))
                for key, value in geom_provenance.items()
            }
        )
        self.anchor_names = tuple(str(value) for value in anchor_names)
        self._anchor_local_position_m = np.asarray(
            anchor_local_position_m, dtype=np.float64
        ).copy()
        self._anchor_local_normal = np.asarray(anchor_local_normal, dtype=np.float64).copy()
        self._anchor_local_position_m.flags.writeable = False
        self._anchor_local_normal.flags.writeable = False
        self.conversion_release_id = str(conversion_release_id)
        self.conversion_manifest_sha256 = str(conversion_manifest_sha256)
        self.asset_release_id = str(asset_release_id)
        self.asset_manifest_sha256 = str(asset_manifest_sha256)
        self.runtime_manifest_sha256 = str(runtime_manifest_sha256)
        self._camera_pose_name = _validate_camera_pose_name(camera_pose_name)
        self._world_from_cv_camera = _immutable_array(
            _validate_rigid_transform(
                world_from_cv_camera_transform,
                "world_from_cv_camera_transform",
            )
        )
        self.compiled_from_owned_asset_bytes = True
        self._expected_model_render_digest = _model_render_digest(self.model)
        self._current_left_q: np.ndarray | None = None
        self._current_world_from_patient: np.ndarray | None = None
        self._expected_qpos: np.ndarray | None = None
        self._closed = False

    @property
    def geom_provenance(self) -> dict[int, dict[str, object]]:
        return {
            int(key): dict(value) for key, value in self._geom_provenance.items()
        }

    @property
    def anchor_local_position_m(self) -> np.ndarray:
        return self._anchor_local_position_m.copy()

    @property
    def anchor_local_normal(self) -> np.ndarray:
        return self._anchor_local_normal.copy()

    @property
    def camera_pose_name(self) -> str:
        return self._camera_pose_name

    @property
    def world_from_cv_camera(self) -> np.ndarray:
        return self._world_from_cv_camera.copy()

    @classmethod
    def from_verified_conversion(
        cls,
        release: Path,
        *,
        camera_pose_name: str = "primary",
    ) -> "OfflineMeshScene":
        """Verify all source/output hashes before compiling any render geometry."""

        camera_name = _validate_camera_pose_name(camera_pose_name)
        world_from_camera = camera_pose_transform(camera_name)
        release_path = Path(release)
        try:
            manifest = verify_conversion_release(release_path)
        except MeshConversionError as exc:
            raise OfflineSimulationError(f"official conversion release is invalid: {exc}") from exc
        release_path = release_path.resolve(strict=True)
        runtime_manifest_sha256 = _verified_runtime_manifest_sha256()
        robot = manifest["robot"]
        human = manifest["human"]
        robot_links = tuple(str(name) for name in robot["links"])
        robot_rows = [row for row in robot["meshes"] if row["role"] == "visual"]
        human_rows = list(human["meshes"])
        if len(robot_links) != 24 or len(robot_rows) != 21 or len(human_rows) != 7:
            raise OfflineSimulationError("verified conversion inventory is incomplete")

        labcoat = human["labcoat"]
        labcoat_rows = [
            row for row in human_rows if row["output_path"] == labcoat["output_path"]
        ]
        if len(labcoat_rows) != 1:
            raise OfflineSimulationError("verified labcoat output identity is ambiguous")
        try:
            labcoat_vertices, labcoat_faces = parse_fixed_obj_bytes(
                _owned_verified_output_bytes(release_path, labcoat_rows[0])
            )
        except (OfflineSimulationError, MeshConversionError) as exc:
            raise OfflineSimulationError("verified labcoat anchor mesh could not be decoded") from exc
        anchor_names: list[str] = []
        anchor_positions: list[np.ndarray] = []
        anchor_normals: list[np.ndarray] = []
        for anchor in labcoat["anchors"]:
            triangle_id = int(anchor["triangle_id"])
            face = labcoat_faces[triangle_id]
            declared_face = np.asarray(anchor["source_vertex_indices"], dtype=np.int64)
            barycentric = np.asarray(anchor["barycentric"], dtype=np.float64)
            if not np.array_equal(face, declared_face):
                raise OfflineSimulationError("labcoat anchor topology changed")
            position = barycentric @ labcoat_vertices[face]
            edge_a = labcoat_vertices[face[1]] - labcoat_vertices[face[0]]
            edge_b = labcoat_vertices[face[2]] - labcoat_vertices[face[0]]
            normal = np.cross(edge_a, edge_b)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1.0e-12:
                raise OfflineSimulationError("labcoat anchor triangle is degenerate")
            normal /= normal_norm
            if not np.allclose(position, anchor["position_m"], atol=1.0e-9, rtol=0.0):
                raise OfflineSimulationError("labcoat anchor barycentric position changed")
            declared_normal = np.asarray(anchor["normal"], dtype=np.float64)
            if (
                not np.isclose(np.linalg.norm(declared_normal), 1.0, atol=1.0e-12)
                or float(np.dot(normal, declared_normal)) < 0.999999999999
            ):
                raise OfflineSimulationError("labcoat anchor normal changed")
            anchor_names.append(str(anchor["name"]))
            anchor_positions.append(position)
            anchor_normals.append(normal)
        if tuple(anchor_names) != POINT_NAMES:
            raise OfflineSimulationError("labcoat ten-point identity changed")

        asset_release = (
            WORKSPACE_ROOT
            / ".wam-build/offline-official-openarm-assets/releases"
            / str(manifest["asset_release_id"])
        )
        urdf_path = asset_release / str(robot["urdf_path"])
        try:
            urdf_mode = urdf_path.lstat().st_mode
            urdf_raw = urdf_path.read_bytes()
        except OSError as exc:
            raise OfflineSimulationError("official URDF is unavailable") from exc
        if (
            stat.S_ISLNK(urdf_mode)
            or not stat.S_ISREG(urdf_mode)
            or _sha256_bytes(urdf_raw) != robot["urdf_sha256"]
        ):
            raise OfflineSimulationError("official URDF changed after release verification")
        try:
            inspect_openarm_urdf(urdf_raw)
        except ExperimentContractError as exc:
            raise OfflineSimulationError(f"official URDF verification failed: {exc}") from exc

        asset_xml: list[str] = []
        owned_mesh_assets: dict[str, bytes] = {}
        owned_raw_by_geom_name: dict[str, tuple[Mapping[str, object], bytes]] = {}
        robot_geoms_by_link: dict[str, list[str]] = {name: [] for name in robot_links}
        expected_robot_geom_names: list[str] = []
        for index, row in enumerate(robot_rows):
            mesh_name = f"robot_mesh_{index:02d}"
            geom_name = f"robot_geom_{index:02d}"
            asset_name = f"{mesh_name}.obj"
            owned_raw = _owned_verified_output_bytes(release_path, row)
            owned_mesh_assets[asset_name] = owned_raw
            owned_raw_by_geom_name[geom_name] = (row, owned_raw)
            asset_xml.append(f'<mesh name={_quoted(mesh_name)} file={_quoted(asset_name)}/>')
            robot_geoms_by_link[str(row["link"])].append(
                f'<geom name={_quoted(geom_name)} type="mesh" mesh={_quoted(mesh_name)} '
                'rgba="0.18 0.42 0.80 1" contype="0" conaffinity="0"/>'
            )
            expected_robot_geom_names.append(geom_name)

        human_body_rows: list[tuple[str, str]] = []
        expected_human_geom_names: list[str] = []
        human_colours = (
            "0.92 0.92 0.95 1",
            "0.20 0.48 0.66 1",
            "0.20 0.48 0.66 1",
            "0.16 0.12 0.10 1",
            "0.22 0.22 0.24 1",
            "0.90 0.82 0.15 1",
            "0.71 0.47 0.35 1",
        )
        for index, (row, colour) in enumerate(zip(human_rows, human_colours, strict=True)):
            mesh_name = f"human_mesh_{index:02d}"
            geom_name = f"human_geom_{index:02d}"
            body_name = f"human__{row['name']}"
            asset_name = f"{mesh_name}.obj"
            owned_raw = _owned_verified_output_bytes(release_path, row)
            owned_mesh_assets[asset_name] = owned_raw
            owned_raw_by_geom_name[geom_name] = (row, owned_raw)
            asset_xml.append(f'<mesh name={_quoted(mesh_name)} file={_quoted(asset_name)}/>')
            body = (
                f'<body name={_quoted(body_name)}>'
                '<freejoint/>'
                '<inertial pos="0 0 0" mass="0.000001" diaginertia="1e-9 1e-9 1e-9"/>'
                f'<geom name={_quoted(geom_name)} type="mesh" mesh={_quoted(mesh_name)} '
                f'rgba="{colour}" contype="0" conaffinity="0"/>'
                '</body>'
            )
            human_body_rows.append((str(row["name"]), body))
            expected_human_geom_names.append(geom_name)

        robot_body_xml: list[str] = []
        for link_name in robot_links:
            robot_body_xml.append(
                f'<body name={_quoted(f"robot__{link_name}")}>'
                '<freejoint/>'
                '<inertial pos="0 0 0" mass="0.000001" diaginertia="1e-9 1e-9 1e-9"/>'
                + "".join(robot_geoms_by_link[link_name])
                + '</body>'
            )

        xml = (
            '<mujoco model="offline_official_openarm_occlusion">'
            '<compiler angle="radian" inertiafromgeom="false"/>'
            '<option gravity="0 0 0" timestep="0.01"><flag contact="disable"/></option>'
            '<visual><global offwidth="640" offheight="360"/>'
            '<map znear="0.01" zfar="10"/></visual>'
            '<asset>'
            + "".join(asset_xml)
            + '</asset><worldbody>'
            + "".join(robot_body_xml)
            + "".join(body for _name, body in human_body_rows)
            + _camera_xml(camera_name)
            + '</worldbody></mujoco>'
        )
        try:
            model = mujoco.MjModel.from_xml_string(xml, assets=owned_mesh_assets)
            data = mujoco.MjData(model)
            # Every body is a direct world child.  Initialize all freejoint
            # quaternions before the first forward pass.
            data.qpos[:] = 0.0
            for joint_id in range(model.njnt):
                address = int(model.jnt_qposadr[joint_id])
                data.qpos[address + 3] = 1.0
            mujoco.mj_forward(model, data)
            renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        except Exception as exc:
            raise OfflineSimulationError("official MuJoCo scene could not be compiled") from exc
        try:
            if (
                model.nmesh != 28
                or model.ngeom != 28
                or np.any(model.geom_type != mujoco.mjtGeom.mjGEOM_MESH)
                or np.any(model.geom_contype != 0)
                or np.any(model.geom_conaffinity != 0)
            ):
                raise OfflineSimulationError(
                    "compiled scene changed mesh/contact inventory"
                )
            provenance: dict[int, Mapping[str, object]] = {}
            robot_geom_ids: set[int] = set()
            human_geom_ids: set[int] = set()
            for geom_name, row in zip(
                expected_robot_geom_names, robot_rows, strict=True
            ):
                geom_id, compiled = _compiled_mesh_provenance(
                    model,
                    geom_name=geom_name,
                    kind="robot_visual",
                    identity=str(row["link"]),
                    row=row,
                    owned_raw=owned_raw_by_geom_name[geom_name][1],
                )
                robot_geom_ids.add(geom_id)
                provenance[geom_id] = compiled
            for geom_name, row in zip(
                expected_human_geom_names, human_rows, strict=True
            ):
                geom_id, compiled = _compiled_mesh_provenance(
                    model,
                    geom_name=geom_name,
                    kind="human",
                    identity=str(row["name"]),
                    row=row,
                    owned_raw=owned_raw_by_geom_name[geom_name][1],
                )
                human_geom_ids.add(geom_id)
                provenance[geom_id] = compiled
            if (
                len(robot_geom_ids) != 21
                or len(human_geom_ids) != 7
                or not robot_geom_ids.isdisjoint(human_geom_ids)
                or set(provenance) != set(range(28))
            ):
                raise OfflineSimulationError("compiled geom identity inventory changed")
            _freeze_model_arrays(model)
            return cls(
                model=model,
                data=data,
                renderer=renderer,
                urdf_raw=urdf_raw,
                robot_links=robot_links,
                human_meshes=[name for name, _body in human_body_rows],
                robot_geom_ids=robot_geom_ids,
                human_geom_ids=human_geom_ids,
                geom_provenance=provenance,
                anchor_names=anchor_names,
                anchor_local_position_m=np.asarray(anchor_positions, dtype=np.float64),
                anchor_local_normal=np.asarray(anchor_normals, dtype=np.float64),
                conversion_release_id=release_path.name,
                conversion_manifest_sha256=EXPECTED_CONVERSION_MANIFEST_SHA256,
                asset_release_id=str(manifest["asset_release_id"]),
                asset_manifest_sha256=EXPECTED_ASSET_MANIFEST_SHA256,
                runtime_manifest_sha256=runtime_manifest_sha256,
                camera_pose_name=camera_name,
                world_from_cv_camera_transform=world_from_camera,
            )
        except BaseException:
            renderer.close()
            raise

    def _set_free_body_pose(self, body_name: str, world_from_body: np.ndarray) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0 or int(self.model.body_jntnum[body_id]) != 1:
            raise OfflineSimulationError(f"flat scene body is missing its freejoint: {body_name}")
        joint_id = int(self.model.body_jntadr[body_id])
        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise OfflineSimulationError(f"flat scene body joint is not free: {body_name}")
        address = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[address : address + 3] = world_from_body[:3, 3]
        self.data.qpos[address + 3 : address + 7] = _matrix_to_wxyz(
            world_from_body[:3, :3]
        )

    def set_frame(
        self, left_joint_positions: np.ndarray, world_from_patient: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Inject absolute robot and rigid-patient poses, then run only mj_forward."""

        if self._closed:
            raise OfflineSimulationError("offline scene is closed")
        patient = _validate_rigid_transform(world_from_patient, "world_from_patient")
        frames = full_openarm_link_fk(self._urdf_raw, left_joint_positions)
        for link_name in self._robot_links:
            self._set_free_body_pose(
                f"robot__{link_name}", ROBOT_WORLD_FROM_ROOT @ frames[link_name]
            )
        for mesh_name in self._human_meshes:
            self._set_free_body_pose(f"human__{mesh_name}", patient)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if int(self.data.ncon) != 0:
            raise OfflineSimulationError("contact unexpectedly entered the kinematic scene")
        self._current_left_q = np.asarray(left_joint_positions, dtype=np.float64).copy()
        self._current_world_from_patient = patient.copy()
        self._expected_qpos = self.data.qpos.copy()
        return {name: transform.copy() for name, transform in frames.items()}

    def _assert_pose_integrity(self) -> None:
        if self._expected_qpos is None:
            raise OfflineSimulationError("set_frame must run before rendering")
        if not np.array_equal(
            self._world_from_cv_camera,
            camera_pose_transform(self._camera_pose_name),
        ):
            raise OfflineSimulationError("bound camera pose identity drifted")
        if any(
            getattr(self.model, name).flags.writeable for name in _MODEL_RENDER_ARRAYS
        ) or not _nested_model_arrays_are_read_only(self.model):
            raise OfflineSimulationError("MjModel render arrays became writable")
        if _model_render_digest(self.model) != self._expected_model_render_digest:
            raise OfflineSimulationError("MjModel render/camera contract drifted")
        if not np.array_equal(self.data.qpos, self._expected_qpos) or not np.array_equal(
            self.data.qvel, np.zeros(self.model.nv, dtype=np.float64)
        ):
            raise OfflineSimulationError("scene pose drifted from declared frame metadata")
        # Recompute all derived xpos/xmat values so callers cannot mutate a
        # public MjData view and cause pixels to disagree with qpos metadata.
        mujoco.mj_forward(self.model, self.data)
        if int(self.data.ncon) != 0:
            raise OfflineSimulationError("contact unexpectedly entered the kinematic scene")

    def project_current_anchors(self) -> AnchorProjection:
        """Project the rigid official anchors through the bound physical camera."""

        if self._closed:
            raise OfflineSimulationError("offline scene is closed")
        if self._current_world_from_patient is None:
            raise OfflineSimulationError("set_frame must run before anchor projection")
        patient = self._current_world_from_patient
        world_positions = (
            patient[:3, :3] @ self._anchor_local_position_m.T
            + patient[:3, 3, None]
        ).T
        world_from_cv = self._world_from_cv_camera
        camera_positions = (
            world_from_cv[:3, :3].T
            @ (world_positions - world_from_cv[:3, 3]).T
        ).T
        projected = project_cv_points(camera_positions)
        return AnchorProjection(
            camera_pose_name=self.camera_pose_name,
            world_from_cv_camera=_immutable_array(world_from_cv),
            anchor_world_position_m=_immutable_array(world_positions),
            anchor_camera_position_m=_immutable_array(camera_positions),
            projected_pixels=_immutable_array(projected),
        )

    def render(self, *, camera_pose_name: str | None = None) -> RenderProducts:
        """Render independent RGB, depth, and segmentation products in sequence."""

        if self._closed:
            raise OfflineSimulationError("offline scene is closed")
        requested_camera = (
            self.camera_pose_name
            if camera_pose_name is None
            else _validate_camera_pose_name(camera_pose_name)
        )
        if requested_camera != self.camera_pose_name:
            raise OfflineSimulationError(
                "requested camera pose does not match the scene's bound camera pose"
            )
        self._assert_pose_integrity()
        rgb, depth, segmentation = render_sequential_camera_products(
            self._renderer, self.data
        )
        if (
            rgb.shape != (HEIGHT, WIDTH, 3)
            or rgb.dtype != np.uint8
            or depth.shape != (HEIGHT, WIDTH)
            or not np.issubdtype(depth.dtype, np.floating)
            or not np.isfinite(depth).all()
            or np.any(depth <= 0.0)
            or segmentation.shape != (HEIGHT, WIDTH, 2)
            or segmentation.dtype != np.int32
        ):
            raise OfflineSimulationError("MuJoCo camera product contract changed")
        robot_mask, human_mask = decode_instance_masks(
            segmentation, self.robot_geom_ids, self.human_geom_ids
        )
        return RenderProducts(
            rgb=rgb,
            depth_m=depth,
            segmentation=segmentation,
            robot_mask=np.asarray(robot_mask, dtype=np.bool_),
            human_mask=np.asarray(human_mask, dtype=np.bool_),
        )

    def capture_frame_evidence(
        self,
        timestamp_s: float,
        *,
        support_radius_px: int = 6,
        robot_mask_dilation_px: int = 1,
        front_depth_margin_m: float = 0.003,
    ) -> OfflineFrameEvidence:
        """Render and evaluate all ten anchors from only masks and metric depth."""

        if self.camera_pose_name != "primary":
            raise OfflineSimulationError(
                "frame evidence is restricted to the primary camera pose"
            )
        if self._current_left_q is None or self._current_world_from_patient is None:
            raise OfflineSimulationError("set_frame must run before evidence capture")
        if not math.isfinite(float(timestamp_s)) or float(timestamp_s) < 0.0:
            raise OfflineSimulationError("frame timestamp must be finite and nonnegative")
        if type(support_radius_px) is not int or support_radius_px <= 0:
            raise OfflineSimulationError("support radius must be a positive integer")
        if type(robot_mask_dilation_px) is not int or robot_mask_dilation_px < 0:
            raise OfflineSimulationError("robot mask dilation must be nonnegative")
        if not math.isfinite(float(front_depth_margin_m)) or front_depth_margin_m < 0.0:
            raise OfflineSimulationError("front depth margin must be finite and nonnegative")

        patient = self._current_world_from_patient
        rotation = patient[:3, :3]
        translation = patient[:3, 3]
        world_positions = (
            rotation @ self._anchor_local_position_m.T + translation[:, None]
        ).T
        world_normals = (rotation @ self._anchor_local_normal.T).T
        world_from_cv = world_from_cv_camera(
            CAMERA_POSITION_M, CAMERA_LOOK_AT_M, np.asarray([0.0, 0.0, 1.0])
        )
        camera_positions = (
            world_from_cv[:3, :3].T
            @ (world_positions - world_from_cv[:3, 3]).T
        ).T
        projected = project_cv_points(camera_positions)
        products = _immutable_render_products(self.render())
        try:
            visibility = measure_rendered_visibility(
                point_names=self.anchor_names,
                projected_pixels=projected,
                point_depth_m=camera_positions[:, 2],
                robot_mask=products.robot_mask,
                human_mask=products.human_mask,
                depth_m=products.depth_for_visibility_m,
                support_radius_px=support_radius_px,
                robot_mask_dilation_px=robot_mask_dilation_px,
                front_depth_margin_m=float(front_depth_margin_m),
            )
        except ExperimentContractError as exc:
            raise OfflineSimulationError(f"rendered visibility measurement failed: {exc}") from exc
        evidence = OfflineFrameEvidence(
            timestamp_s=float(timestamp_s),
            conversion_release_id=self.conversion_release_id,
            conversion_manifest_sha256=self.conversion_manifest_sha256,
            asset_release_id=self.asset_release_id,
            asset_manifest_sha256=self.asset_manifest_sha256,
            runtime_manifest_sha256=self.runtime_manifest_sha256,
            geom_provenance=self._geom_provenance,
            left_joint_positions=_immutable_array(self._current_left_q),
            world_from_patient=_immutable_array(patient),
            world_from_cv_camera=_immutable_array(world_from_cv),
            anchor_names=self.anchor_names,
            anchor_local_position_m=_immutable_array(self._anchor_local_position_m),
            anchor_world_position_m=_immutable_array(world_positions),
            anchor_world_normal=_immutable_array(world_normals),
            anchor_camera_position_m=_immutable_array(camera_positions),
            projected_pixels=_immutable_array(projected),
            visibility=visibility,
            products=products,
            support_radius_px=support_radius_px,
            robot_mask_dilation_px=robot_mask_dilation_px,
            front_depth_margin_m=float(front_depth_margin_m),
        )
        _attach_scene_capture_guard(evidence)
        validate_oracle_frame_evidence(evidence)
        return evidence

    def close(self) -> None:
        if not self._closed:
            self._renderer.close()
            self._closed = True

    def __enter__(self) -> "OfflineMeshScene":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _observation_group_key(frame: ObservationFrame) -> tuple[str, str, int, str]:
    return (
        frame.scenario_id,
        frame.target_name,
        frame.wrist_index,
        frame.observation_stream_sha256,
    )


def _scoring_key(
    frame: GroundTruthFrame | PolicyFrameOutput,
) -> tuple[str, str, int, int, float, str]:
    return (
        frame.scenario_id,
        frame.target_name,
        frame.wrist_index,
        frame.frame_index,
        frame.timestamp_s,
        frame.observation_stream_sha256,
    )


def _policy_output(
    observation: ObservationFrame,
    *,
    policy_name: str,
    state: str,
    estimate_world_m: np.ndarray | None,
    valid: bool,
    episode_id: str | None,
    last_valid_age_s: float | None,
    predicted_error_p95_m: float | None,
    reacquired: bool,
    deliberately_unsafe: bool = False,
) -> PolicyFrameOutput:
    return PolicyFrameOutput(
        policy_name=policy_name,
        scenario_id=observation.scenario_id,
        target_name=observation.target_name,
        wrist_index=observation.wrist_index,
        frame_index=observation.frame_index,
        timestamp_s=observation.timestamp_s,
        geometry_accepted=observation.geometry_accepted,
        patient_motion_bound_m_s=observation.patient_motion_bound_m_s,
        state=state,
        estimate_world_m=estimate_world_m,
        valid=valid,
        episode_id=episode_id,
        last_valid_age_s=last_valid_age_s,
        predicted_error_p95_m=predicted_error_p95_m,
        reacquired=reacquired,
        deliberately_unsafe=deliberately_unsafe,
        visible_point_count_report_only=observation.visible_point_count_report_only,
        observation_stream_sha256=observation.observation_stream_sha256,
    )


def evaluate_occlusion_policies(
    observations: Sequence[ObservationFrame],
    *,
    max_freeze_age_s: float = DEFAULT_MAX_FREEZE_AGE_S,
    max_freeze_error_p95_m: float = DEFAULT_MAX_FREEZE_ERROR_P95_M,
) -> dict[str, tuple[PolicyFrameOutput, ...]]:
    """Evaluate the three declared policies without access to scoring truth.

    Every policy consumes the exact same predeclared observation.  The visible
    point count is copied to evidence rows but is never read by a decision.
    """

    max_age = float(max_freeze_age_s)
    max_error = float(max_freeze_error_p95_m)
    if not math.isfinite(max_age) or max_age <= 0.0:
        raise OfflineSimulationError("max_freeze_age_s must be finite and positive")
    if not math.isfinite(max_error) or max_error <= 0.0:
        raise OfflineSimulationError(
            "max_freeze_error_p95_m must be finite and positive"
        )
    if not isinstance(observations, Sequence) or isinstance(
        observations, (str, bytes, bytearray)
    ):
        raise OfflineSimulationError("observations must be a finite sequence")

    outputs: dict[str, list[PolicyFrameOutput]] = {
        policy_name: [] for policy_name in POLICY_NAMES
    }
    stream_state: dict[tuple[str, str, int, str], dict[str, object]] = {}
    seen_frames: set[tuple[str, str, int, int, float, str]] = set()
    for observation in observations:
        if not isinstance(observation, ObservationFrame):
            raise OfflineSimulationError("every observation must be an ObservationFrame")
        group_key = _observation_group_key(observation)
        frame_key = (
            observation.scenario_id,
            observation.target_name,
            observation.wrist_index,
            observation.frame_index,
            observation.timestamp_s,
            observation.observation_stream_sha256,
        )
        if frame_key in seen_frames:
            raise OfflineSimulationError("observation frame key is duplicate")
        seen_frames.add(frame_key)
        state = stream_state.setdefault(
            group_key,
            {
                "next_frame_index": 0,
                "last_timestamp_s": None,
                "motion_bound_m_s": observation.patient_motion_bound_m_s,
                "last_estimate": None,
                "last_error_p95_m": None,
                "last_valid_timestamp_s": None,
                "episode_counter": 0,
                "episode_id": None,
                "had_accepted": False,
                "rejection_since_accepted": False,
            },
        )
        if observation.frame_index != state["next_frame_index"]:
            raise OfflineSimulationError(
                "observation frame_index must be contiguous within each stream"
            )
        previous_timestamp = state["last_timestamp_s"]
        if previous_timestamp is not None and observation.timestamp_s <= previous_timestamp:
            raise OfflineSimulationError(
                "observation timestamps must increase strictly within each stream"
            )
        if observation.patient_motion_bound_m_s != state["motion_bound_m_s"]:
            raise OfflineSimulationError(
                "patient motion bound must be predeclared and constant within a stream"
            )
        state["next_frame_index"] = observation.frame_index + 1
        state["last_timestamp_s"] = observation.timestamp_s

        if observation.geometry_accepted:
            assert observation.accepted_estimate_world_m is not None
            assert observation.accepted_error_p95_m is not None
            reacquired = bool(
                state["had_accepted"] and state["rejection_since_accepted"]
            )
            if not state["had_accepted"] or reacquired:
                state["episode_counter"] = int(state["episode_counter"]) + 1
                state["episode_id"] = f"E{int(state['episode_counter']):04d}"
            state["had_accepted"] = True
            state["rejection_since_accepted"] = False
            state["last_estimate"] = observation.accepted_estimate_world_m
            state["last_error_p95_m"] = observation.accepted_error_p95_m
            state["last_valid_timestamp_s"] = observation.timestamp_s
            direct_decision = decide_occlusion_policy(
                geometry_accepted=True,
                last_valid_age_s=0.0,
                last_target_error_p95_m=observation.accepted_error_p95_m,
                patient_motion_bound_m_s=observation.patient_motion_bound_m_s,
                max_freeze_age_s=max_age,
                max_freeze_error_p95_m=max_error,
            )
            if direct_decision.state != "DIRECT_GEOMETRY_UPDATE":
                raise OfflineSimulationError("bounded policy direct-update contract changed")
            for policy_name in POLICY_NAMES:
                outputs[policy_name].append(
                    _policy_output(
                        observation,
                        policy_name=policy_name,
                        state="DIRECT_GEOMETRY_UPDATE",
                        estimate_world_m=observation.accepted_estimate_world_m,
                        valid=True,
                        episode_id=str(state["episode_id"]),
                        last_valid_age_s=0.0,
                        predicted_error_p95_m=observation.accepted_error_p95_m,
                        reacquired=reacquired,
                    )
                )
            continue

        state["rejection_since_accepted"] = True
        last_estimate = state["last_estimate"]
        if last_estimate is None:
            for policy_name in POLICY_NAMES:
                outputs[policy_name].append(
                    _policy_output(
                        observation,
                        policy_name=policy_name,
                        state="LOST",
                        estimate_world_m=None,
                        valid=False,
                        episode_id=None,
                        last_valid_age_s=None,
                        predicted_error_p95_m=None,
                        reacquired=False,
                    )
                )
            continue

        last_timestamp = float(state["last_valid_timestamp_s"])
        last_error = float(state["last_error_p95_m"])
        # Frame schedules are declared in decimal seconds.  Quantize only the
        # subtraction artefact so an authored 0.30 s / 10 mm boundary is not
        # misclassified as 0.30000000000000004 s / 10.000000000000002 mm.
        age = round(observation.timestamp_s - last_timestamp, 12)
        if age < 0.0:
            raise OfflineSimulationError("last-valid age became negative")
        predicted_error = last_error + observation.patient_motion_bound_m_s * age
        episode_id = str(state["episode_id"])

        outputs["direct_geometry_only"].append(
            _policy_output(
                observation,
                policy_name="direct_geometry_only",
                state="LOST",
                estimate_world_m=None,
                valid=False,
                episode_id=None,
                last_valid_age_s=age,
                predicted_error_p95_m=None,
                reacquired=False,
            )
        )
        outputs["indefinite_world_lock_baseline"].append(
            _policy_output(
                observation,
                policy_name="indefinite_world_lock_baseline",
                state="INDEFINITE_WORLD_LOCK_UNSAFE",
                estimate_world_m=np.asarray(last_estimate),
                valid=True,
                episode_id=episode_id,
                last_valid_age_s=age,
                predicted_error_p95_m=predicted_error,
                reacquired=False,
                deliberately_unsafe=True,
            )
        )
        bounded_decision = decide_occlusion_policy(
            geometry_accepted=False,
            last_valid_age_s=age,
            last_target_error_p95_m=last_error,
            patient_motion_bound_m_s=observation.patient_motion_bound_m_s,
            max_freeze_age_s=max_age,
            max_freeze_error_p95_m=max_error,
        )
        bounded_valid = bounded_decision.state == "BOUNDED_PATIENT_FRAME_FREEZE"
        if bounded_decision.state not in {"BOUNDED_PATIENT_FRAME_FREEZE", "LOST"}:
            raise OfflineSimulationError("bounded policy rejection state is invalid")
        outputs["bounded_freeze_then_lost"].append(
            _policy_output(
                observation,
                policy_name="bounded_freeze_then_lost",
                state=bounded_decision.state,
                estimate_world_m=np.asarray(last_estimate) if bounded_valid else None,
                valid=bounded_valid,
                episode_id=episode_id if bounded_valid else None,
                last_valid_age_s=age,
                predicted_error_p95_m=bounded_decision.predicted_error_p95_m,
                reacquired=False,
            )
        )

    return {name: tuple(rows) for name, rows in outputs.items()}


def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    samples = np.asarray(tuple(values), dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise OfflineSimulationError("metric samples must be finite scalars")
    if len(samples) == 0:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "p99": None}
    quantiles = np.quantile(samples, [0.50, 0.90, 0.95, 0.99], method="linear")
    return {
        "count": int(len(samples)),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
    }


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise OfflineSimulationError("metric rate counts are invalid")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": None if denominator == 0 else float(numerator / denominator),
    }


def _strict_metric_threshold_exceeded(value: float, threshold: float) -> bool:
    """Apply the predeclared mathematical strict-greater-than comparison."""

    return value > threshold


def _validate_cross_policy_trace(
    outputs_by_policy: Mapping[str, Sequence[PolicyFrameOutput]],
) -> dict[str, tuple[PolicyFrameOutput, ...]]:
    if tuple(outputs_by_policy) != POLICY_NAMES:
        raise OfflineSimulationError("policy output inventory is incomplete or reordered")
    rows_by_policy = {
        name: tuple(outputs_by_policy[name]) for name in POLICY_NAMES
    }
    lengths = {len(rows) for rows in rows_by_policy.values()}
    if len(lengths) != 1:
        raise OfflineSimulationError("policies do not share one observation trace length")
    replay_state: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for index, reference in enumerate(rows_by_policy[POLICY_NAMES[0]]):
        if not isinstance(reference, PolicyFrameOutput):
            raise OfflineSimulationError("policy output row identity is invalid")
        shared_reference = (
            reference.scenario_id,
            reference.target_name,
            reference.wrist_index,
            reference.frame_index,
            reference.timestamp_s,
            reference.observation_stream_sha256,
            reference.geometry_accepted,
            reference.patient_motion_bound_m_s,
            reference.visible_point_count_report_only,
            reference.reacquired,
        )
        aligned = []
        for policy_name in POLICY_NAMES:
            row = rows_by_policy[policy_name][index]
            if not isinstance(row, PolicyFrameOutput) or row.policy_name != policy_name:
                raise OfflineSimulationError("policy output row identity is invalid")
            shared = (
                row.scenario_id,
                row.target_name,
                row.wrist_index,
                row.frame_index,
                row.timestamp_s,
                row.observation_stream_sha256,
                row.geometry_accepted,
                row.patient_motion_bound_m_s,
                row.visible_point_count_report_only,
                row.reacquired,
            )
            if shared != shared_reference:
                raise OfflineSimulationError(
                    "policies do not share the exact observation trace/stream"
                )
            aligned.append(row)

        direct, indefinite, bounded = aligned
        if reference.geometry_accepted:
            if any(
                row.state != "DIRECT_GEOMETRY_UPDATE"
                or not row.valid
                or row.estimate_world_m is None
                or row.episode_id is None
                or row.last_valid_age_s != 0.0
                or row.predicted_error_p95_m is None
                or row.deliberately_unsafe
                for row in aligned
            ):
                raise OfflineSimulationError(
                    "accepted observation has an invalid policy state contract"
                )
            if not all(
                np.array_equal(direct.estimate_world_m, row.estimate_world_m)
                and direct.predicted_error_p95_m == row.predicted_error_p95_m
                and direct.episode_id == row.episode_id
                for row in aligned[1:]
            ):
                raise OfflineSimulationError(
                    "accepted estimate differs across the shared observation trace"
                )
        else:
            if (
                direct.state != "LOST"
                or direct.valid
                or direct.estimate_world_m is not None
                or direct.episode_id is not None
                or direct.deliberately_unsafe
            ):
                raise OfflineSimulationError(
                    "direct policy contradicts the rejected geometry trace"
                )
            indefinite_contract = (
                indefinite.state == "LOST"
                and not indefinite.valid
                and indefinite.estimate_world_m is None
                and indefinite.episode_id is None
                and not indefinite.deliberately_unsafe
            ) or (
                indefinite.state == "INDEFINITE_WORLD_LOCK_UNSAFE"
                and indefinite.valid
                and indefinite.estimate_world_m is not None
                and indefinite.episode_id is not None
                and indefinite.deliberately_unsafe
            )
            bounded_contract = (
                bounded.state == "LOST"
                and not bounded.valid
                and bounded.estimate_world_m is None
                and bounded.episode_id is None
                and not bounded.deliberately_unsafe
            ) or (
                bounded.state == "BOUNDED_PATIENT_FRAME_FREEZE"
                and bounded.valid
                and bounded.estimate_world_m is not None
                and bounded.episode_id is not None
                and not bounded.deliberately_unsafe
                and bounded.last_valid_age_s is not None
                and bounded.last_valid_age_s <= DEFAULT_MAX_FREEZE_AGE_S
                and bounded.predicted_error_p95_m is not None
                and bounded.predicted_error_p95_m
                <= DEFAULT_MAX_FREEZE_ERROR_P95_M
            )
            if not indefinite_contract or not bounded_contract:
                raise OfflineSimulationError(
                    "held policy contradicts the rejected geometry trace"
                )

        group_key = (
            reference.scenario_id,
            reference.target_name,
            reference.wrist_index,
            reference.observation_stream_sha256,
        )
        replay = replay_state.setdefault(
            group_key,
            {
                "next_frame_index": 0,
                "last_timestamp_s": None,
                "motion_bound_m_s": reference.patient_motion_bound_m_s,
                "last_estimate": None,
                "last_error_p95_m": None,
                "last_valid_timestamp_s": None,
                "episode_counter": 0,
                "episode_id": None,
                "had_accepted": False,
                "rejection_since_accepted": False,
            },
        )
        if reference.frame_index != replay["next_frame_index"]:
            raise OfflineSimulationError(
                "policy observation trace frame indices are not contiguous"
            )
        prior_timestamp = replay["last_timestamp_s"]
        if prior_timestamp is not None and reference.timestamp_s <= prior_timestamp:
            raise OfflineSimulationError(
                "policy observation trace timestamps are not strictly increasing"
            )
        if reference.patient_motion_bound_m_s != replay["motion_bound_m_s"]:
            raise OfflineSimulationError(
                "policy observation trace changed its patient motion bound"
            )
        replay["next_frame_index"] = reference.frame_index + 1
        replay["last_timestamp_s"] = reference.timestamp_s

        if reference.geometry_accepted:
            expected_reacquired = bool(
                replay["had_accepted"] and replay["rejection_since_accepted"]
            )
            if reference.reacquired != expected_reacquired:
                raise OfflineSimulationError(
                    "reacquisition flag contradicts the last accepted episode"
                )
            if not replay["had_accepted"] or expected_reacquired:
                replay["episode_counter"] = int(replay["episode_counter"]) + 1
                replay["episode_id"] = f"E{int(replay['episode_counter']):04d}"
            if any(row.episode_id != replay["episode_id"] for row in aligned):
                raise OfflineSimulationError(
                    "accepted episode does not follow the last accepted observation"
                )
            replay["had_accepted"] = True
            replay["rejection_since_accepted"] = False
            replay["last_estimate"] = direct.estimate_world_m
            replay["last_error_p95_m"] = direct.predicted_error_p95_m
            replay["last_valid_timestamp_s"] = reference.timestamp_s
            continue

        if reference.reacquired:
            raise OfflineSimulationError(
                "rejected observation cannot be marked reacquired"
            )
        replay["rejection_since_accepted"] = True
        last_estimate = replay["last_estimate"]
        if last_estimate is None:
            if any(
                row.state != "LOST"
                or row.valid
                or row.estimate_world_m is not None
                or row.episode_id is not None
                or row.last_valid_age_s is not None
                or row.predicted_error_p95_m is not None
                for row in aligned
            ):
                raise OfflineSimulationError(
                    "policy hold exists without a last accepted observation"
                )
            continue

        expected_age = round(
            reference.timestamp_s - float(replay["last_valid_timestamp_s"]), 12
        )
        expected_error = float(replay["last_error_p95_m"]) + (
            reference.patient_motion_bound_m_s * expected_age
        )
        if direct.last_valid_age_s != expected_age or direct.predicted_error_p95_m is not None:
            raise OfflineSimulationError(
                "direct policy age contradicts the last accepted observation"
            )
        for row in (indefinite, bounded):
            if (
                row.last_valid_age_s != expected_age
                or row.predicted_error_p95_m != expected_error
            ):
                raise OfflineSimulationError(
                    "held policy age/p95 contradicts the last accepted observation"
                )
        expected_bounded = decide_occlusion_policy(
            geometry_accepted=False,
            last_valid_age_s=expected_age,
            last_target_error_p95_m=float(replay["last_error_p95_m"]),
            patient_motion_bound_m_s=reference.patient_motion_bound_m_s,
            max_freeze_age_s=DEFAULT_MAX_FREEZE_AGE_S,
            max_freeze_error_p95_m=DEFAULT_MAX_FREEZE_ERROR_P95_M,
        )
        expected_bounded_valid = (
            expected_bounded.state == "BOUNDED_PATIENT_FRAME_FREEZE"
        )
        if (
            bounded.state != expected_bounded.state
            or bounded.valid != expected_bounded_valid
            or bounded.predicted_error_p95_m
            != expected_bounded.predicted_error_p95_m
            or (
                expected_bounded_valid
                and (
                    bounded.estimate_world_m is None
                    or bounded.episode_id != replay["episode_id"]
                )
            )
            or (
                not expected_bounded_valid
                and (
                    bounded.estimate_world_m is not None
                    or bounded.episode_id is not None
                )
            )
        ):
            raise OfflineSimulationError(
                "bounded decision does not match the replayed FREEZE/LOST contract"
            )
        if (
            not np.array_equal(indefinite.estimate_world_m, last_estimate)
            or indefinite.episode_id != replay["episode_id"]
        ):
            raise OfflineSimulationError(
                "indefinite held estimate differs from the last accepted observation"
            )
        if bounded.valid and (
            not np.array_equal(bounded.estimate_world_m, last_estimate)
            or bounded.episode_id != replay["episode_id"]
        ):
            raise OfflineSimulationError(
                "bounded held estimate differs from the last accepted observation"
            )
    return rows_by_policy


def _score_policy_rows(
    rows: Sequence[PolicyFrameOutput],
    truth_by_key: Mapping[
        tuple[str, str, int, int, float, str], GroundTruthFrame
    ],
) -> dict[str, object]:
    errors: list[float] = []
    residual_by_key: dict[
        tuple[str, str, int, int, float, str], np.ndarray
    ] = {}
    state_counts: dict[str, int] = {}
    for row in rows:
        state_counts[row.state] = state_counts.get(row.state, 0) + 1
        if not row.valid:
            continue
        assert row.estimate_world_m is not None
        key = _scoring_key(row)
        residual = row.estimate_world_m - truth_by_key[key].target_world_m
        error = float(np.linalg.norm(residual))
        if not math.isfinite(error):
            raise OfflineSimulationError("scored target error is non-finite")
        errors.append(error)
        residual_by_key[key] = residual

    jitter: list[float] = []
    groups: dict[tuple[str, str, int, str], list[PolicyFrameOutput]] = {}
    for row in rows:
        groups.setdefault(
            (
                row.scenario_id,
                row.target_name,
                row.wrist_index,
                row.observation_stream_sha256,
            ),
            [],
        ).append(row)
    time_to_lost: list[float] = []
    reacquisition_time: list[float] = []
    rejection_run_duration: list[float] = []
    freeze_duration: list[float] = []
    censored_rejection_runs = 0
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda item: item.frame_index)
        for previous, current in zip(ordered[:-1], ordered[1:], strict=True):
            if (
                previous.valid
                and current.valid
                and previous.episode_id is not None
                and previous.episode_id == current.episode_id
                and current.frame_index == previous.frame_index + 1
            ):
                previous_residual = residual_by_key[_scoring_key(previous)]
                current_residual = residual_by_key[_scoring_key(current)]
                jitter.append(float(np.linalg.norm(current_residual - previous_residual)))

        index = 0
        acquired = False
        while index < len(ordered):
            if ordered[index].geometry_accepted:
                acquired = True
                index += 1
                continue
            run_start = index
            acquired_before_run = acquired
            while index < len(ordered) and not ordered[index].geometry_accepted:
                index += 1
            run = ordered[run_start:index]
            if acquired_before_run:
                lost = next((row for row in run if row.state == "LOST"), None)
                if lost is None:
                    censored_rejection_runs += 1
                else:
                    time_to_lost.append(lost.timestamp_s - run[0].timestamp_s)
                freeze_ages = [
                    float(row.last_valid_age_s)
                    for row in run
                    if row.state == "BOUNDED_PATIENT_FRAME_FREEZE"
                    and row.last_valid_age_s is not None
                ]
                if freeze_ages:
                    freeze_duration.append(max(freeze_ages))
                if index < len(ordered) and ordered[index].reacquired:
                    rejection_run_duration.append(
                        ordered[index].timestamp_s - run[0].timestamp_s
                    )
                    # In this method-only study, geometry acceptance and policy
                    # update occur on the same predeclared frame.  Detector
                    # reacquisition latency is explicitly outside scope.
                    reacquisition_time.append(0.0)

    valid_count = len(errors)
    return {
        "target_error_m": _distribution(errors),
        "jitter_m": _distribution(jitter),
        "coverage": _rate(valid_count, len(rows)),
        "false_valid_above_5mm": _rate(
            sum(_strict_metric_threshold_exceeded(error, 0.005) for error in errors),
            valid_count,
        ),
        "false_valid_above_10mm": _rate(
            sum(_strict_metric_threshold_exceeded(error, 0.010) for error in errors),
            valid_count,
        ),
        "time_to_lost_s": _distribution(time_to_lost),
        "reacquisition_time_s": _distribution(reacquisition_time),
        "rejection_run_duration_s": _distribution(rejection_run_duration),
        "freeze_duration_s": _distribution(freeze_duration),
        "censored_rejection_runs": censored_rejection_runs,
        "state_counts": state_counts,
    }


def _equal_weight_summary(
    completed_metrics: Sequence[Mapping[str, object]],
    *,
    required_count: int,
    completed_count: int,
) -> dict[str, object]:
    def rate_summary(name: str) -> dict[str, int | float | None]:
        values = [
            float(metric[name]["rate"])
            for metric in completed_metrics
            if metric[name]["rate"] is not None
        ]
        return {
            "contributing_strata_count": len(values),
            "mean": None if not values else float(np.mean(values)),
        }

    def distribution_summary(name: str) -> dict[str, int | float | None]:
        result: dict[str, int | float | None] = {}
        contributors = [metric[name] for metric in completed_metrics if metric[name]["count"]]
        result["contributing_strata_count"] = len(contributors)
        for quantile in ("p50", "p90", "p95", "p99"):
            result[quantile] = (
                None
                if not contributors
                else float(np.mean([float(row[quantile]) for row in contributors]))
            )
        return result

    states = sorted(
        {
            state
            for metric in completed_metrics
            for state in metric["state_counts"]
        }
    )
    state_fraction = {
        state: float(
            np.mean(
                [
                    metric["state_counts"].get(state, 0)
                    / metric["coverage"]["denominator"]
                    for metric in completed_metrics
                ]
            )
        )
        for state in states
    }
    return {
        "required_strata_count": required_count,
        "completed_strata_count": completed_count,
        "contributing_strata_count": len(completed_metrics),
        "coverage": rate_summary("coverage"),
        "target_error_m": distribution_summary("target_error_m"),
        "jitter_m": distribution_summary("jitter_m"),
        "false_valid_above_5mm": rate_summary("false_valid_above_5mm"),
        "false_valid_above_10mm": rate_summary("false_valid_above_10mm"),
        "time_to_lost_s": distribution_summary("time_to_lost_s"),
        "reacquisition_time_s": distribution_summary("reacquisition_time_s"),
        "rejection_run_duration_s": distribution_summary(
            "rejection_run_duration_s"
        ),
        "freeze_duration_s": distribution_summary("freeze_duration_s"),
        "state_fraction": state_fraction,
    }


def score_occlusion_policy_metrics(
    outputs_by_policy: Mapping[str, Sequence[PolicyFrameOutput]],
    ground_truth_frames: Sequence[GroundTruthFrame],
    *,
    required_strata: Sequence[RequiredStratum] | None = None,
) -> dict[str, dict[str, object]]:
    """Score one shared trace, retaining every predeclared required stratum."""

    rows_by_policy = _validate_cross_policy_trace(outputs_by_policy)
    truth_by_key: dict[
        tuple[str, str, int, int, float, str], GroundTruthFrame
    ] = {}
    for frame in ground_truth_frames:
        if not isinstance(frame, GroundTruthFrame):
            raise OfflineSimulationError("every scoring row must be a GroundTruthFrame")
        key = _scoring_key(frame)
        if key in truth_by_key:
            raise OfflineSimulationError("ground-truth one-to-one key is duplicate")
        truth_by_key[key] = frame

    expected_keys = set(truth_by_key)
    for policy_name in POLICY_NAMES:
        output_by_key: dict[
            tuple[str, str, int, int, float, str], PolicyFrameOutput
        ] = {}
        for row in rows_by_policy[policy_name]:
            key = _scoring_key(row)
            if key in output_by_key:
                raise OfflineSimulationError("policy output one-to-one key is duplicate")
            output_by_key[key] = row
        if set(output_by_key) != expected_keys:
            raise OfflineSimulationError(
                "policy output is missing a ground-truth one-to-one pair"
            )

    reference_groups: dict[
        tuple[str, str, int, str], list[PolicyFrameOutput]
    ] = {}
    for row in rows_by_policy[POLICY_NAMES[0]]:
        reference_groups.setdefault(
            (
                row.scenario_id,
                row.target_name,
                row.wrist_index,
                row.observation_stream_sha256,
            ),
            [],
        ).append(row)

    if required_strata is None:
        required = tuple(
            RequiredStratum(
                scenario_id=key[0],
                target_name=key[1],
                wrist_index=key[2],
                observation_stream_sha256=key[3],
                observation_seed=0,
                patient_motion_seed=0,
                execution_status="COMPLETE",
                declared_frame_count=len(rows),
            )
            for key, rows in reference_groups.items()
        )
    else:
        required = tuple(required_strata)
        if not all(isinstance(row, RequiredStratum) for row in required):
            raise OfflineSimulationError(
                "required_strata must contain RequiredStratum rows"
            )
    required_by_key: dict[tuple[str, str, int, str], RequiredStratum] = {}
    for row in required:
        key = (
            row.scenario_id,
            row.target_name,
            row.wrist_index,
            row.observation_stream_sha256,
        )
        if key in required_by_key:
            raise OfflineSimulationError("required stratum key is duplicate")
        required_by_key[key] = row
    completed_keys = {
        key for key, row in required_by_key.items() if row.execution_status == "COMPLETE"
    }
    metric_eligible_keys = {
        key for key, row in required_by_key.items() if row.metric_eligible
    }
    if set(reference_groups) != completed_keys:
        raise OfflineSimulationError(
            "required strata and completed observation streams do not match"
        )
    for key, rows in reference_groups.items():
        declared = required_by_key[key].declared_frame_count
        truth_count = sum(
            (
                frame.scenario_id,
                frame.target_name,
                frame.wrist_index,
                frame.observation_stream_sha256,
            )
            == key
            for frame in truth_by_key.values()
        )
        if len(rows) != declared or truth_count != declared:
            raise OfflineSimulationError(
                "completed stratum frame count differs from its predeclared denominator"
            )

    completed_count = len(completed_keys)
    required_count = len(required)
    metric_eligible_frame_count = sum(
        1
        for frame in truth_by_key.values()
        if (
            frame.scenario_id,
            frame.target_name,
            frame.wrist_index,
            frame.observation_stream_sha256,
        )
        in metric_eligible_keys
    )
    denominators = {
        "required_strata_count": required_count,
        "completed_strata_count": completed_count,
        "failed_strata_count": required_count - completed_count,
        "metric_eligible_strata_count": len(metric_eligible_keys),
        "metric_ineligible_strata_count": required_count
        - len(metric_eligible_keys),
        "contributing_strata_count": len(metric_eligible_keys),
        "eligible_paired_frames": metric_eligible_frame_count,
        "completion_rate": None
        if required_count == 0
        else float(completed_count / required_count),
    }

    result: dict[str, dict[str, object]] = {}
    for policy_name in POLICY_NAMES:
        rows = rows_by_policy[policy_name]
        metric_rows = tuple(
            row
            for row in rows
            if (
                row.scenario_id,
                row.target_name,
                row.wrist_index,
                row.observation_stream_sha256,
            )
            in metric_eligible_keys
        )
        metric_truth = {
            key: frame
            for key, frame in truth_by_key.items()
            if (
                frame.scenario_id,
                frame.target_name,
                frame.wrist_index,
                frame.observation_stream_sha256,
            )
            in metric_eligible_keys
        }
        pooled = _score_policy_rows(metric_rows, metric_truth)
        per_stratum: list[dict[str, object]] = []
        completed_metrics: list[Mapping[str, object]] = []
        for required_row in required:
            common = {
                "scenario_id": required_row.scenario_id,
                "target_name": required_row.target_name,
                "wrist_index": required_row.wrist_index,
                "observation_stream_sha256": required_row.observation_stream_sha256,
                "observation_seed": required_row.observation_seed,
                "patient_motion_seed": required_row.patient_motion_seed,
                "execution_status": required_row.execution_status,
                "declared_frame_count": required_row.declared_frame_count,
                "metric_eligible": required_row.metric_eligible,
                "metric_ineligibility_reason": (
                    required_row.metric_ineligibility_reason
                ),
            }
            if not required_row.metric_eligible:
                per_stratum.append(
                    {**common, "metrics_available": False, "metrics": None}
                )
                continue
            key = (
                required_row.scenario_id,
                required_row.target_name,
                required_row.wrist_index,
                required_row.observation_stream_sha256,
            )
            stratum_rows = tuple(
                row
                for row in rows
                if (
                    row.scenario_id,
                    row.target_name,
                    row.wrist_index,
                    row.observation_stream_sha256,
                )
                == key
            )
            stratum_truth = {
                scoring_key: frame
                for scoring_key, frame in truth_by_key.items()
                if (
                    frame.scenario_id,
                    frame.target_name,
                    frame.wrist_index,
                    frame.observation_stream_sha256,
                )
                == key
            }
            metrics = _score_policy_rows(stratum_rows, stratum_truth)
            completed_metrics.append(metrics)
            per_stratum.append(
                {**common, "metrics_available": True, "metrics": metrics}
            )
        entry = dict(pooled)
        entry.update(
            {
                "pooled_frame": pooled,
                "equal_weight_per_stratum": _equal_weight_summary(
                    completed_metrics,
                    required_count=required_count,
                    completed_count=completed_count,
                ),
                "denominators": denominators,
                "per_stratum": per_stratum,
            }
        )
        result[policy_name] = entry
    return result


__all__ = [
    "AnchorProjection",
    "CAMERA_LOOK_AT_M",
    "CAMERA_POSE_NAMES",
    "CAMERA_POSITION_M",
    "DEFAULT_MAX_FREEZE_AGE_S",
    "DEFAULT_MAX_FREEZE_ERROR_P95_M",
    "EXPECTED_SCENE_CONTRACT_SHA256",
    "GroundTruthFrame",
    "HEIGHT",
    "INTRINSICS",
    "ObservationFrame",
    "ORACLE_SIMULATION_STATUS",
    "OBLIQUE_CAMERA_LOOK_AT_M",
    "OBLIQUE_CAMERA_POSITION_M",
    "OfflineFrameEvidence",
    "OfflineMeshScene",
    "OfflineSimulationError",
    "POLICY_NAMES",
    "PolicyFrameOutput",
    "RequiredStratum",
    "ROBOT_WORLD_FROM_ROOT",
    "RenderProducts",
    "SOURCE_INTRINSICS",
    "WIDTH",
    "camera_projection_probe",
    "camera_pose_transform",
    "decode_instance_masks",
    "evaluate_occlusion_policies",
    "full_openarm_link_fk",
    "persist_frame_evidence",
    "project_cv_points",
    "raw_render_sha256",
    "render_sequential_camera_products",
    "score_occlusion_policy_metrics",
    "scene_contract_sha256",
    "oracle_frame_evidence_sha256",
    "validate_oracle_frame_evidence",
    "verify_frame_evidence_release",
    "world_from_cv_camera",
]
