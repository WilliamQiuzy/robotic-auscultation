"""Immutable, fail-closed binding of one RGB-D frame to robot evidence.

The bundle is constructed before the frame is fanned out to recording,
perception, or preview consumers.  It snapshots camera and mask bytes, then
requires the camera exposure, OpenArm telemetry bracket, and interval-swept
robot mask to describe the exact same physical frame.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
import re
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from ausc.perception.camera_stream import CameraFrame
from ausc.perception.robot.camera_exposure_evidence import CameraExposureEvidence
from ausc.perception.robot.robot_state_stream import (
    EXACT_CONTROL_RECORD_SCHEMA,
    BracketedExactControlTrajectory,
    ExactControlRecord,
    ExactFollowerAction,
)
from ausc.perception.tracking.occlusion_runtime import (
    ActivatedOcclusionIdentity,
    LEGACY_HANDEYE_CALIBRATION_TIER,
    LEGACY_HANDEYE_CAPABILITY_SCHEMA,
    PROVISIONAL_HANDEYE_CALIBRATION_TIER,
    PROVISIONAL_HANDEYE_CAPABILITY_SCHEMA,
    RobotMaskPacket,
)


BUNDLE_SCHEMA = "wam_frame_evidence_bundle_v3"
BUNDLE_ID_PREFIX = "frame_evidence-"

_NS_PER_SECOND = 1_000_000_000
_MAX_CAMERA_TO_EXPOSURE_NS = 100_000_000
_MAX_EXPOSURE_INTERVAL_NS = 40_000_000
_MAX_ROBOT_GAP_NS = 2_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[a-z][a-z0-9_]*-[0-9a-f]{20}$")


class FrameEvidenceBundleError(ValueError):
    """The inputs cannot safely be treated as evidence for one frame."""


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrameEvidenceBundleError(message)


def _require_sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{label} SHA-256 is invalid",
    )
    return str(value)


def _hard_readonly_array(value: np.ndarray) -> np.ndarray:
    """Return a defensive C-order snapshot backed by immutable ``bytes``."""

    contiguous = np.ascontiguousarray(value)
    result = np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
    ).reshape(contiguous.shape)
    # The bytes-backed view cannot later be made writeable with setflags.
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _mask_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.uint8).tobytes(order="C")
    ).hexdigest()


def _canonical_value(value: Any) -> Any:
    """Represent metadata without non-finite or platform-dependent JSON floats."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return int(value)
    if isinstance(value, (float, np.floating)):
        _require(math.isfinite(float(value)), "canonical metadata contains non-finite data")
        return {"float64_hex": float(value).hex()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise FrameEvidenceBundleError(
        f"canonical metadata contains unsupported type: {type(value).__name__}"
    )


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _dataclass_metadata(value: object, *, exclude: Tuple[str, ...] = ()) -> Dict[str, Any]:
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.name not in exclude
    }


@dataclass(frozen=True)
class ImmutableCameraFrame:
    """A detached RGB-D camera frame whose arrays cannot be made writeable."""

    frame: np.ndarray
    t: float
    index: int
    status: str
    depth: np.ndarray
    depth_scale: float
    age_s: float
    device_t_ms: float
    time_source: str
    camera_id: str
    camera_profile_sha256: str
    alignment_sha256: str
    camera_host_monotonic_ns: int
    rgb_sha256: str
    depth_sha256: str

    def __post_init__(self) -> None:
        _require(self.status == "OK", "camera frame status is not OK")
        _require(
            type(self.index) is int and self.index >= 0,
            "camera frame index must be a nonnegative integer",
        )
        _require(
            _finite_number(self.t) and float(self.t) > 0.0,
            "camera host timestamp is invalid",
        )
        _require(
            type(self.camera_host_monotonic_ns) is int
            and self.camera_host_monotonic_ns
            == int(round(float(self.t) * _NS_PER_SECOND))
            and self.camera_host_monotonic_ns > 0,
            "camera host monotonic timestamp is invalid",
        )
        _require(
            _finite_number(self.device_t_ms),
            "camera device timestamp is required",
        )
        _require(
            _finite_number(self.depth_scale) and float(self.depth_scale) > 0.0,
            "camera depth scale is invalid",
        )
        _require(
            _finite_number(self.age_s) and float(self.age_s) >= 0.0,
            "camera frame age is invalid",
        )
        _require(
            isinstance(self.time_source, str) and bool(self.time_source),
            "camera time source is missing",
        )
        _require(
            isinstance(self.camera_id, str) and bool(self.camera_id),
            "camera identity is invalid",
        )
        _require_sha256(self.camera_profile_sha256, "camera profile")
        _require_sha256(self.alignment_sha256, "alignment")

        _require(
            isinstance(self.frame, np.ndarray)
            and self.frame.dtype == np.uint8
            and self.frame.ndim == 3
            and self.frame.shape[2] == 3
            and self.frame.shape[0] > 0
            and self.frame.shape[1] > 0,
            "camera RGB frame must be a nonempty HxWx3 uint8 array",
        )
        _require(
            isinstance(self.depth, np.ndarray)
            and self.depth.dtype == np.uint16
            and self.depth.ndim == 2
            and tuple(self.depth.shape) == tuple(self.frame.shape[:2]),
            "camera depth must be aligned HxW uint16 data",
        )
        frozen_rgb = _hard_readonly_array(self.frame)
        frozen_depth = _hard_readonly_array(self.depth)
        object.__setattr__(self, "frame", frozen_rgb)
        object.__setattr__(self, "depth", frozen_depth)
        _require(
            self.rgb_sha256 == _array_sha256(frozen_rgb),
            "camera RGB SHA-256 does not match frame bytes",
        )
        _require(
            self.depth_sha256 == _array_sha256(frozen_depth),
            "camera depth SHA-256 does not match depth bytes",
        )

    @classmethod
    def from_camera_frame(
        cls,
        frame: CameraFrame,
        *,
        camera_id: str,
        camera_profile_sha256: str,
        alignment_sha256: str,
    ) -> "ImmutableCameraFrame":
        if not isinstance(frame, CameraFrame):
            raise FrameEvidenceBundleError("camera frame type is invalid")
        if frame.status != "OK":
            raise FrameEvidenceBundleError("camera frame status is not OK")
        if frame.frame is None:
            raise FrameEvidenceBundleError("camera RGB frame is missing")
        if frame.depth is None:
            raise FrameEvidenceBundleError("camera depth frame is missing")
        if frame.device_t_ms is None:
            raise FrameEvidenceBundleError("camera device timestamp is required")
        if not isinstance(frame.frame, np.ndarray):
            raise FrameEvidenceBundleError("camera RGB frame type is invalid")
        if not isinstance(frame.depth, np.ndarray):
            raise FrameEvidenceBundleError("camera depth frame type is invalid")
        return cls(
            frame=frame.frame,
            t=frame.t,
            index=frame.index,
            status=frame.status,
            depth=frame.depth,
            depth_scale=frame.depth_scale,
            age_s=frame.age_s,
            device_t_ms=frame.device_t_ms,
            time_source=frame.time_source,
            camera_id=camera_id,
            camera_profile_sha256=camera_profile_sha256,
            alignment_sha256=alignment_sha256,
            camera_host_monotonic_ns=int(round(float(frame.t) * _NS_PER_SECOND))
            if _finite_number(frame.t)
            else 0,
            rgb_sha256=_array_sha256(frame.frame),
            depth_sha256=_array_sha256(frame.depth),
        )


def _validate_exposure(
    camera_frame: ImmutableCameraFrame,
    exposure: CameraExposureEvidence,
    *,
    timing_artifact_sha256: str,
) -> None:
    _require(
        isinstance(exposure, CameraExposureEvidence),
        "camera exposure evidence type is invalid",
    )
    _require(
        type(exposure.frame_index) is int
        and exposure.frame_index == camera_frame.index,
        "camera exposure frame index does not match RGB-D frame",
    )
    _require(
        _finite_number(exposure.camera_host_t_s)
        and float(exposure.camera_host_t_s) == float(camera_frame.t),
        "camera exposure host timestamp does not match RGB-D frame",
    )
    _require(
        type(exposure.camera_host_monotonic_ns) is int
        and exposure.camera_host_monotonic_ns
        == camera_frame.camera_host_monotonic_ns,
        "camera exposure host monotonic timestamp does not match RGB-D frame",
    )
    _require(
        _finite_number(exposure.camera_device_t_ms)
        and float(exposure.camera_device_t_ms)
        == float(camera_frame.device_t_ms),
        "camera exposure device timestamp does not match RGB-D frame",
    )
    _require(
        exposure.camera_id == camera_frame.camera_id,
        "camera exposure camera identity does not match RGB-D frame",
    )
    _require(
        exposure.camera_profile_sha256 == camera_frame.camera_profile_sha256,
        "camera exposure camera profile does not match RGB-D frame",
    )
    _require(
        exposure.alignment_sha256 == camera_frame.alignment_sha256,
        "camera exposure alignment does not match RGB-D frame",
    )
    _require_sha256(timing_artifact_sha256, "timing artifact")
    _require_sha256(exposure.camera_latency_sha256, "camera timing artifact")
    _require(
        exposure.camera_latency_sha256 == timing_artifact_sha256,
        "camera exposure timing artifact does not match activated timing",
    )

    seconds = (
        exposure.exposure_lower_host_t_s,
        exposure.exposure_nominal_host_t_s,
        exposure.exposure_upper_host_t_s,
    )
    nanoseconds = (
        exposure.exposure_lower_monotonic_ns,
        exposure.exposure_nominal_monotonic_ns,
        exposure.exposure_upper_monotonic_ns,
    )
    _require(
        all(_finite_number(value) for value in seconds)
        and seconds[0] < seconds[1] < seconds[2] < camera_frame.t,
        "camera exposure interval is invalid",
    )
    _require(
        all(type(value) is int and value > 0 for value in nanoseconds)
        and nanoseconds[0] < nanoseconds[1] < nanoseconds[2]
        < camera_frame.camera_host_monotonic_ns
        and all(
            timestamp_ns == int(round(float(timestamp_s) * _NS_PER_SECOND))
            for timestamp_ns, timestamp_s in zip(nanoseconds, seconds)
        )
        and nanoseconds[2] - nanoseconds[0] <= _MAX_EXPOSURE_INTERVAL_NS
        and camera_frame.camera_host_monotonic_ns - nanoseconds[0]
        <= _MAX_CAMERA_TO_EXPOSURE_NS,
        "camera exposure interval does not match monotonic timing bounds",
    )


def _validate_robot_packet(packet: ExactControlRecord) -> None:
    _require(
        isinstance(packet, ExactControlRecord),
        "robot trajectory contains an invalid exact control v2 record",
    )
    _require(
        packet.schema == EXACT_CONTROL_RECORD_SCHEMA,
        "robot trajectory exact control v2 record schema is invalid",
    )
    _require(
        type(packet.seq) is int and 0 <= packet.seq < 2**64,
        "robot trajectory sequence is invalid",
    )
    _require(
        type(packet.monotonic_ns) is int and 0 < packet.monotonic_ns < 2**63,
        "robot trajectory timestamp is invalid",
    )
    _require(
        packet.side == "left",
        "robot trajectory must use the left OpenArm",
    )
    _require(
        packet.controller_mode == "unilateral",
        "robot trajectory controller mode must be unilateral",
    )
    _require(
        packet.leader_can_interface == "can1",
        "left OpenArm leader must use can1",
    )
    _require(
        packet.follower_can_interface == "can3",
        "left OpenArm follower must use can3",
    )
    _require(
        packet.joint_names
        == tuple(f"openarm_left_joint{index}" for index in range(1, 8)),
        "robot trajectory joint order is not canonical left OpenArm order",
    )
    for value, label in (
        (packet.follower_q_rad, "follower positions"),
        (packet.follower_qd_rad_s, "follower velocities"),
    ):
        _require(
            isinstance(value, tuple)
            and len(value) == 7
            and all(_finite_number(item) for item in value),
            f"robot trajectory {label} are invalid",
        )
    for value, label in (
        (packet.follower_gripper_q_rad, "follower gripper position"),
        (packet.follower_gripper_qd_rad_s, "follower gripper velocity"),
        (packet.follower_gripper_motor_q_rad, "follower gripper motor position"),
        (
            packet.follower_gripper_motor_qd_rad_s,
            "follower gripper motor velocity",
        ),
    ):
        _require(_finite_number(value), f"robot trajectory {label} is invalid")
    _require(
        isinstance(packet.action, ExactFollowerAction),
        "robot trajectory exact follower action is missing",
    )
    for value, label in (
        (packet.action.arm_kp, "action arm Kp"),
        (packet.action.arm_kd, "action arm Kd"),
        (packet.action.arm_q, "action arm q"),
        (packet.action.arm_dq, "action arm dq"),
        (packet.action.arm_tau, "action arm tau"),
    ):
        _require(
            isinstance(value, tuple)
            and len(value) == 7
            and all(_finite_number(item) for item in value),
            f"robot trajectory {label} is invalid",
        )
    for value, label in (
        (packet.action.gripper_kp, "action gripper Kp"),
        (packet.action.gripper_kd, "action gripper Kd"),
        (packet.action.gripper_q, "action gripper q"),
        (packet.action.gripper_dq, "action gripper dq"),
        (packet.action.gripper_tau, "action gripper tau"),
    ):
        _require(_finite_number(value), f"robot trajectory {label} is invalid")
    _require(
        type(packet.reference_generation) is int
        and 0 < packet.reference_generation < 2**64
        and packet.reference_snapshot_valid is True,
        "robot trajectory reference snapshot is invalid",
    )
    _require(
        all(
            type(value) is int and 0 < value < 2**63
            for value in (
                packet.reference_source_monotonic_ns,
                packet.control_call_begin_monotonic_ns,
                packet.arm_call_return_monotonic_ns,
                packet.gripper_call_begin_monotonic_ns,
                packet.gripper_call_return_monotonic_ns,
                packet.control_call_end_monotonic_ns,
                packet.publisher_monotonic_ns,
            )
        )
        and packet.reference_source_monotonic_ns
        <= packet.control_call_begin_monotonic_ns
        and packet.observation_monotonic_ns
        <= packet.control_call_begin_monotonic_ns
        <= packet.arm_call_return_monotonic_ns
        <= packet.gripper_call_begin_monotonic_ns
        <= packet.gripper_call_return_monotonic_ns
        <= packet.control_call_end_monotonic_ns
        <= packet.publisher_monotonic_ns,
        "robot trajectory exact control timing is invalid",
    )
    _require(
        type(packet.envelope_seq) is int and 0 <= packet.envelope_seq < 2**64,
        "robot trajectory envelope sequence is invalid",
    )
    _require(
        type(packet.controller_pid) is int and packet.controller_pid > 0,
        "robot trajectory controller process is invalid",
    )
    _require(
        isinstance(packet.controller_start_token, str)
        and bool(packet.controller_start_token),
        "robot trajectory controller start token is invalid",
    )
    for digest, label in (
        (packet.urdf_sha256, "URDF"),
        (packet.joint_mapper_sha256, "joint mapper"),
        (packet.pair_contract_sha256, "pair contract"),
        (packet.controller_binary_sha256, "controller binary"),
        (packet.controller_source_manifest_sha256, "controller source manifest"),
        (packet.follower_config_sha256, "follower config"),
        (packet.openarm_sdk_sha256, "OpenArm SDK"),
    ):
        _require_sha256(digest, f"robot trajectory {label}")
    _require(
        type(packet.send_status_observable) is bool
        and isinstance(packet.send_status_reason, str)
        and bool(packet.send_status_reason),
        "robot trajectory send-status provenance is invalid",
    )


def _packet_provenance(packet: ExactControlRecord) -> Tuple[object, ...]:
    return (
        packet.controller_pid,
        packet.controller_start_token,
        packet.side,
        packet.controller_mode,
        packet.leader_can_interface,
        packet.follower_can_interface,
        packet.joint_names,
        packet.urdf_sha256,
        packet.joint_mapper_sha256,
        packet.pair_contract_sha256,
        packet.controller_binary_sha256,
        packet.controller_source_manifest_sha256,
        packet.follower_config_sha256,
        packet.openarm_sdk_sha256,
        packet.send_status_observable,
        packet.send_status_reason,
    )


def _validate_trajectory(
    exposure: CameraExposureEvidence,
    trajectory: BracketedExactControlTrajectory,
) -> None:
    _require(
        isinstance(trajectory, BracketedExactControlTrajectory),
        "exact control robot trajectory v2 is required",
    )
    expected_interval = (
        exposure.exposure_lower_monotonic_ns,
        exposure.exposure_nominal_monotonic_ns,
        exposure.exposure_upper_monotonic_ns,
    )
    actual_interval = (
        trajectory.exposure_lower_ns,
        trajectory.exposure_nominal_ns,
        trajectory.exposure_upper_ns,
    )
    _require(
        actual_interval == expected_interval,
        "robot trajectory exposure interval does not match camera exposure",
    )
    _require(
        actual_interval[0] < actual_interval[1] < actual_interval[2],
        "robot trajectory exposure interval is invalid",
    )
    _require(
        isinstance(trajectory.records, tuple) and len(trajectory.records) >= 2,
        "robot trajectory must contain a complete exact control v2 bracket",
    )
    for packet in trajectory.records:
        _validate_robot_packet(packet)
    _require(
        trajectory.before.monotonic_ns <= actual_interval[0]
        and trajectory.after.monotonic_ns >= actual_interval[2],
        "robot trajectory does not bracket the complete camera exposure",
    )
    provenance = _packet_provenance(trajectory.before)
    for left, right in zip(trajectory.records, trajectory.records[1:]):
        _require(
            right.seq == left.seq + 1,
            "robot trajectory sequence is not contiguous",
        )
        gap_ns = right.monotonic_ns - left.monotonic_ns
        _require(
            0 < gap_ns <= _MAX_ROBOT_GAP_NS,
            "robot trajectory gap exceeds 2 ms",
        )
        _require(
            _packet_provenance(right) == provenance,
            "robot trajectory publisher provenance changed",
        )
    # Exercise observation interpolation at all exposure anchors. Actions are
    # deliberately never sampled or interpolated.
    try:
        for timestamp_ns in actual_interval:
            trajectory.sample_observation_at(timestamp_ns)
    except (TypeError, ValueError) as exc:
        raise FrameEvidenceBundleError(
            f"robot trajectory cannot sample camera exposure: {exc}"
        ) from exc


def _validate_mask(
    camera_frame: ImmutableCameraFrame,
    exposure: CameraExposureEvidence,
    trajectory: BracketedExactControlTrajectory,
    packet: RobotMaskPacket,
    *,
    timing_artifact_sha256: str,
    activated_identity: ActivatedOcclusionIdentity,
) -> None:
    _require(
        isinstance(packet, RobotMaskPacket),
        "robot mask packet type is invalid",
    )
    _require(
        isinstance(activated_identity, ActivatedOcclusionIdentity),
        "activated occlusion identity is missing",
    )
    _require(
        packet.status == "OK",
        f"robot mask status is not OK: {packet.reason}",
    )
    _require(
        isinstance(packet.reason, str) and bool(packet.reason),
        "robot mask reason is missing",
    )
    _require(
        type(packet.frame_index) is int
        and packet.frame_index == camera_frame.index == exposure.frame_index,
        "robot mask frame index does not match RGB-D frame",
    )
    _require(
        _finite_number(packet.camera_host_t_s)
        and float(packet.camera_host_t_s) == float(camera_frame.t),
        "robot mask host timestamp does not match RGB-D frame",
    )
    _require(
        type(packet.camera_host_monotonic_ns) is int
        and packet.camera_host_monotonic_ns
        == camera_frame.camera_host_monotonic_ns,
        "robot mask host monotonic timestamp does not match RGB-D frame",
    )
    _require(
        _finite_number(packet.camera_device_t_ms)
        and float(packet.camera_device_t_ms)
        == float(camera_frame.device_t_ms),
        "robot mask device timestamp does not match RGB-D frame",
    )
    _require(
        packet.camera_id == camera_frame.camera_id,
        "robot mask camera identity does not match RGB-D frame",
    )
    _require(
        packet.camera_profile_sha256 == camera_frame.camera_profile_sha256,
        "robot mask camera profile does not match RGB-D frame",
    )
    _require(
        packet.alignment_sha256 == camera_frame.alignment_sha256,
        "robot mask alignment does not match RGB-D frame",
    )
    _require(
        packet.camera_latency_sha256 == timing_artifact_sha256,
        "robot mask timing artifact does not match camera exposure",
    )

    mask_exposure_seconds = (
        packet.exposure_lower_host_t_s,
        packet.exposure_nominal_host_t_s,
        packet.exposure_upper_host_t_s,
    )
    exposure_seconds = (
        exposure.exposure_lower_host_t_s,
        exposure.exposure_nominal_host_t_s,
        exposure.exposure_upper_host_t_s,
    )
    mask_exposure_ns = (
        packet.exposure_lower_monotonic_ns,
        packet.exposure_nominal_monotonic_ns,
        packet.exposure_upper_monotonic_ns,
    )
    exposure_ns = (
        exposure.exposure_lower_monotonic_ns,
        exposure.exposure_nominal_monotonic_ns,
        exposure.exposure_upper_monotonic_ns,
    )
    _require(
        mask_exposure_seconds == exposure_seconds
        and mask_exposure_ns == exposure_ns,
        "robot mask exposure interval does not match camera exposure",
    )

    _require(
        isinstance(packet.mask, np.ndarray)
        and packet.mask.dtype == np.bool_
        and packet.mask.ndim == 2
        and tuple(packet.mask.shape) == tuple(camera_frame.depth.shape),
        "robot mask shape/dtype does not match aligned depth",
    )
    _require_sha256(packet.mask_sha256, "robot mask")
    _require(
        packet.mask_sha256 == _mask_sha256(packet.mask),
        "robot mask SHA-256 does not match mask bytes",
    )
    _require(
        type(packet.masked_pixel_count) is int
        and packet.masked_pixel_count == int(np.count_nonzero(packet.mask)),
        "robot mask masked pixel count does not match mask bytes",
    )

    expected_robot_values = (
        trajectory.sequence_first,
        trajectory.sequence_last,
        trajectory.before.monotonic_ns / _NS_PER_SECOND,
        trajectory.after.monotonic_ns / _NS_PER_SECOND,
        trajectory.before.monotonic_ns,
        trajectory.after.monotonic_ns,
        trajectory.max_gap_ns / _NS_PER_SECOND,
        trajectory.max_gap_ns,
    )
    actual_robot_values = (
        packet.joint_sequence_first,
        packet.joint_sequence_last,
        packet.joint_time_first_s,
        packet.joint_time_last_s,
        packet.joint_time_first_monotonic_ns,
        packet.joint_time_last_monotonic_ns,
        packet.joint_max_gap_s,
        packet.joint_max_gap_ns,
    )
    _require(
        actual_robot_values == expected_robot_values,
        "robot mask robot trajectory bracket does not match exact packets",
    )
    _require(
        packet.joint_publisher_identity == trajectory.publisher_identity,
        "robot mask publisher identity does not match robot trajectory",
    )

    for release_id, digest, label in (
        (packet.geometry_release_id, packet.geometry_sha256, "geometry"),
        (packet.tool_release_id, packet.tool_sha256, "tool"),
        (packet.handeye_release_id, packet.handeye_sha256, "hand-eye"),
    ):
        _require(
            isinstance(release_id, str)
            and _RELEASE_ID.fullmatch(release_id) is not None,
            f"robot mask {label} release identity is invalid",
        )
        _require_sha256(digest, f"robot mask {label}")
    for digest, label in (
        (packet.camera_projection_sha256, "camera projection"),
        (packet.occlusion_profile_sha256, "occlusion profile"),
        (packet.handeye_policy_sha256, "hand-eye policy"),
    ):
        _require_sha256(digest, f"robot mask {label}")
    if packet.handeye_capability_schema == PROVISIONAL_HANDEYE_CAPABILITY_SCHEMA:
        _require(
            packet.handeye_calibration_tier
            == PROVISIONAL_HANDEYE_CALIBRATION_TIER
            and packet.handeye_release_id
            == "handeye_activation-" + packet.handeye_sha256[:20],
            "robot mask hand-eye schema/tier/policy binding is invalid",
        )
        release_policy_is_provisional = True
    elif packet.handeye_capability_schema == LEGACY_HANDEYE_CAPABILITY_SCHEMA:
        release_policy_is_provisional = False
        _require(
            packet.handeye_calibration_tier
            == LEGACY_HANDEYE_CALIBRATION_TIER
            and packet.handeye_release_id
            == "handeye-" + packet.handeye_sha256[:20],
            "robot mask hand-eye schema/tier/policy binding is invalid",
        )
    else:
        raise FrameEvidenceBundleError(
            "robot mask hand-eye schema/tier/policy binding is invalid"
        )
    _require(
        packet.handeye_provisional is release_policy_is_provisional,
        "robot mask hand-eye provisional policy does not match its release",
    )
    if release_policy_is_provisional:
        _require(
            packet.handeye_release_id
            == "handeye_activation-" + packet.handeye_sha256[:20],
            "robot mask hand-eye schema/tier/policy binding is invalid",
        )
        _require(
            packet.handeye_calibration_status
            == "USER_ACCEPTED_BELOW_GATE"
            and packet.handeye_quality_gate_passed is False
            and _finite_number(packet.handeye_workspace_1sigma_max_m)
            and 0.0 < float(packet.handeye_workspace_1sigma_max_m) <= 0.0065
            and packet.handeye_covariance_bound_transform
            == "transform_follower_base_from_camera"
            and packet.handeye_covariance_scope == "random_only"
            and packet.handeye_systematic_bias_status == "NOT_CHARACTERIZED"
            and packet.handeye_systematic_bias_bound_m is None
            and packet.handeye_autonomous_targeting_authorized is False
            and _finite_number(
                packet.handeye_provisional_policy_allowance_m
            )
            and float(packet.handeye_provisional_policy_allowance_m) == 0.0065,
            "robot mask provisional hand-eye evidence is invalid",
        )
    else:
        _require(
            packet.handeye_calibration_status == "COMPLETE"
            and packet.handeye_quality_gate_passed is None
            and packet.handeye_workspace_1sigma_max_m is None
            and packet.handeye_covariance_bound_transform
            == "transform_camera_from_follower_base"
            and packet.handeye_covariance_scope == "legacy_production"
            and packet.handeye_systematic_bias_status
            == "LEGACY_NOT_APPLICABLE"
            and packet.handeye_systematic_bias_bound_m is None
            and packet.handeye_autonomous_targeting_authorized is True
            and packet.handeye_provisional_policy_allowance_m == 0.0,
            "robot mask legacy hand-eye evidence is invalid",
        )
    try:
        packet.validate_for_frame(
            frame_index=camera_frame.index,
            camera_host_t_s=camera_frame.t,
            camera_id=camera_frame.camera_id,
            camera_profile_sha256=camera_frame.camera_profile_sha256,
            depth_shape=tuple(camera_frame.depth.shape),
            activated_identity=activated_identity,
        )
    except (TypeError, ValueError) as exc:
        raise FrameEvidenceBundleError(
            "robot mask does not match the activated occlusion identity: "
            f"{exc}"
        ) from exc


def _clone_trajectory(
    trajectory: BracketedExactControlTrajectory,
) -> BracketedExactControlTrajectory:
    return BracketedExactControlTrajectory(
        exposure_lower_ns=trajectory.exposure_lower_ns,
        exposure_nominal_ns=trajectory.exposure_nominal_ns,
        exposure_upper_ns=trajectory.exposure_upper_ns,
        records=tuple(
            replace(record, action=replace(record.action))
            for record in trajectory.records
        ),
    )


def _clone_mask_packet(packet: RobotMaskPacket) -> RobotMaskPacket:
    frozen_mask = _hard_readonly_array(packet.mask)
    cloned = replace(packet, mask=frozen_mask)
    # RobotMaskPacket makes a conventional read-only copy in __post_init__.
    # Restore the bytes-backed snapshot so setflags(write=True) also fails.
    object.__setattr__(cloned, "mask", frozen_mask)
    return cloned


def _trajectory_metadata(
    trajectory: BracketedExactControlTrajectory,
) -> Dict[str, object]:
    return {
        "exposure_lower_ns": trajectory.exposure_lower_ns,
        "exposure_nominal_ns": trajectory.exposure_nominal_ns,
        "exposure_upper_ns": trajectory.exposure_upper_ns,
        "exact_control_records": [
            {
                **_dataclass_metadata(record, exclude=("action",)),
                "action": _dataclass_metadata(record.action),
            }
            for record in trajectory.records
        ],
    }


def _trajectory_sha256(trajectory: BracketedExactControlTrajectory) -> str:
    return hashlib.sha256(_canonical_json(_trajectory_metadata(trajectory))).hexdigest()


def _exact_action_metadata(
    trajectory: BracketedExactControlTrajectory,
) -> Dict[str, object]:
    return {
        "schema": EXACT_CONTROL_RECORD_SCHEMA,
        "records": [
            {
                "control_tick_seq": record.control_tick_seq,
                "action": _dataclass_metadata(record.action),
            }
            for record in trajectory.records
        ],
    }


def _exact_action_sha256(
    trajectory: BracketedExactControlTrajectory,
) -> str:
    return hashlib.sha256(
        _canonical_json(_exact_action_metadata(trajectory))
    ).hexdigest()


def _source_provenance(
    trajectory: BracketedExactControlTrajectory,
) -> Dict[str, object]:
    record = trajectory.before
    return {
        "controller_pid": record.controller_pid,
        "controller_start_token": record.controller_start_token,
        "side": record.side,
        "controller_mode": record.controller_mode,
        "leader_can_interface": record.leader_can_interface,
        "follower_can_interface": record.follower_can_interface,
        "joint_names": record.joint_names,
        "urdf_sha256": record.urdf_sha256,
        "joint_mapper_sha256": record.joint_mapper_sha256,
        "pair_contract_sha256": record.pair_contract_sha256,
        "controller_binary_sha256": record.controller_binary_sha256,
        "controller_source_manifest_sha256": (
            record.controller_source_manifest_sha256
        ),
        "follower_config_sha256": record.follower_config_sha256,
        "openarm_sdk_sha256": record.openarm_sdk_sha256,
        "send_status_observable": record.send_status_observable,
        "send_status_reason": record.send_status_reason,
        "publisher_identity": record.publisher_identity,
    }


def _bundle_metadata(
    camera_frame: ImmutableCameraFrame,
    exposure: CameraExposureEvidence,
    trajectory: BracketedExactControlTrajectory,
    mask_packet: RobotMaskPacket,
    *,
    timing_artifact_sha256: str,
    trajectory_sha256: str,
    exact_action_sha256: str,
) -> Dict[str, object]:
    camera_metadata = _dataclass_metadata(
        camera_frame,
        exclude=("frame", "depth"),
    )
    camera_metadata["rgb"] = {
        "shape": tuple(int(value) for value in camera_frame.frame.shape),
        "dtype": camera_frame.frame.dtype.str,
        "sha256": camera_frame.rgb_sha256,
    }
    camera_metadata["depth"] = {
        "shape": tuple(int(value) for value in camera_frame.depth.shape),
        "dtype": camera_frame.depth.dtype.str,
        "sha256": camera_frame.depth_sha256,
    }
    mask_metadata = _dataclass_metadata(mask_packet, exclude=("mask",))
    mask_metadata["mask"] = {
        "shape": tuple(int(value) for value in mask_packet.mask.shape),
        "dtype": mask_packet.mask.dtype.str,
        "sha256": mask_packet.mask_sha256,
    }
    return {
        "schema": BUNDLE_SCHEMA,
        "camera_frame": camera_metadata,
        "camera_exposure": _dataclass_metadata(exposure),
        "robot_trajectory_sha256": trajectory_sha256,
        "robot_exact_actions": {
            "sha256": exact_action_sha256,
            "control_tick_first": trajectory.sequence_first,
            "control_tick_last": trajectory.sequence_last,
            "count": len(trajectory.records),
        },
        "robot_source_provenance": _source_provenance(trajectory),
        "robot_mask": mask_metadata,
        "timing_artifact_sha256": timing_artifact_sha256,
    }


@dataclass(frozen=True)
class FrameEvidenceBundle:
    """One exact RGB-D frame with exposure, robot, and robot-mask evidence."""

    camera_frame: ImmutableCameraFrame
    exposure: CameraExposureEvidence
    robot_trajectory: BracketedExactControlTrajectory
    robot_mask_packet: RobotMaskPacket
    activated_identity: ActivatedOcclusionIdentity
    timing_artifact_sha256: str
    robot_trajectory_sha256: str
    robot_exact_action_sha256: str
    bundle_id: str

    @property
    def index(self) -> int:
        """Read-only frame index for worker fanout compatibility."""

        return self.camera_frame.index

    @property
    def t(self) -> float:
        """Read-only frame timestamp for worker fanout compatibility."""

        return self.camera_frame.t

    @property
    def robot_exact_action_tick_first(self) -> int:
        return self.robot_trajectory.sequence_first

    @property
    def robot_exact_action_tick_last(self) -> int:
        return self.robot_trajectory.sequence_last

    @property
    def robot_exact_action_count(self) -> int:
        return len(self.robot_trajectory.records)

    def __post_init__(self) -> None:
        _require(
            isinstance(self.camera_frame, ImmutableCameraFrame),
            "immutable camera frame is missing",
        )
        _require_sha256(self.timing_artifact_sha256, "timing artifact")
        _validate_exposure(
            self.camera_frame,
            self.exposure,
            timing_artifact_sha256=self.timing_artifact_sha256,
        )
        _validate_trajectory(self.exposure, self.robot_trajectory)
        _validate_mask(
            self.camera_frame,
            self.exposure,
            self.robot_trajectory,
            self.robot_mask_packet,
            timing_artifact_sha256=self.timing_artifact_sha256,
            activated_identity=self.activated_identity,
        )
        expected_trajectory_sha256 = _trajectory_sha256(self.robot_trajectory)
        _require(
            self.robot_trajectory_sha256 == expected_trajectory_sha256,
            "robot trajectory SHA-256 does not match packet evidence",
        )
        expected_action_sha256 = _exact_action_sha256(self.robot_trajectory)
        _require(
            self.robot_exact_action_sha256 == expected_action_sha256,
            "robot exact action SHA-256 does not match tick evidence",
        )
        expected_id = BUNDLE_ID_PREFIX + hashlib.sha256(
            _canonical_json(
                _bundle_metadata(
                    self.camera_frame,
                    self.exposure,
                    self.robot_trajectory,
                    self.robot_mask_packet,
                    timing_artifact_sha256=self.timing_artifact_sha256,
                    trajectory_sha256=expected_trajectory_sha256,
                    exact_action_sha256=expected_action_sha256,
                )
            )
        ).hexdigest()
        _require(
            self.bundle_id == expected_id,
            "frame evidence bundle ID does not match canonical metadata",
        )

    @classmethod
    def build(
        cls,
        *,
        camera_frame: CameraFrame,
        camera_id: str,
        camera_profile_sha256: str,
        alignment_sha256: str,
        timing_artifact_sha256: str,
        exposure: CameraExposureEvidence,
        robot_trajectory: BracketedExactControlTrajectory,
        robot_mask_packet: RobotMaskPacket,
        activated_identity: ActivatedOcclusionIdentity,
    ) -> "FrameEvidenceBundle":
        """Snapshot and bind evidence before any downstream fanout."""

        snapshot = ImmutableCameraFrame.from_camera_frame(
            camera_frame,
            camera_id=camera_id,
            camera_profile_sha256=camera_profile_sha256,
            alignment_sha256=alignment_sha256,
        )
        _require(
            isinstance(exposure, CameraExposureEvidence),
            "camera exposure evidence type is invalid",
        )
        _require(
            isinstance(robot_trajectory, BracketedExactControlTrajectory),
            "exact control robot trajectory v2 is required",
        )
        _require(
            isinstance(robot_mask_packet, RobotMaskPacket),
            "robot mask packet type is invalid",
        )
        _require(
            isinstance(activated_identity, ActivatedOcclusionIdentity),
            "activated occlusion identity is missing",
        )
        exposure_snapshot = replace(exposure)
        trajectory_snapshot = _clone_trajectory(robot_trajectory)
        mask_snapshot = _clone_mask_packet(robot_mask_packet)

        _validate_exposure(
            snapshot,
            exposure_snapshot,
            timing_artifact_sha256=timing_artifact_sha256,
        )
        _validate_trajectory(exposure_snapshot, trajectory_snapshot)
        _validate_mask(
            snapshot,
            exposure_snapshot,
            trajectory_snapshot,
            mask_snapshot,
            timing_artifact_sha256=timing_artifact_sha256,
            activated_identity=activated_identity,
        )
        trajectory_digest = _trajectory_sha256(trajectory_snapshot)
        action_digest = _exact_action_sha256(trajectory_snapshot)
        bundle_id = BUNDLE_ID_PREFIX + hashlib.sha256(
            _canonical_json(
                _bundle_metadata(
                    snapshot,
                    exposure_snapshot,
                    trajectory_snapshot,
                    mask_snapshot,
                    timing_artifact_sha256=timing_artifact_sha256,
                    trajectory_sha256=trajectory_digest,
                    exact_action_sha256=action_digest,
                )
            )
        ).hexdigest()
        return cls(
            camera_frame=snapshot,
            exposure=exposure_snapshot,
            robot_trajectory=trajectory_snapshot,
            robot_mask_packet=mask_snapshot,
            activated_identity=activated_identity,
            timing_artifact_sha256=timing_artifact_sha256,
            robot_trajectory_sha256=trajectory_digest,
            robot_exact_action_sha256=action_digest,
            bundle_id=bundle_id,
        )

    def canonical_metadata(self) -> Dict[str, object]:
        """Return a fresh serializable copy of the metadata hashed by bundle_id."""

        return _bundle_metadata(
            self.camera_frame,
            self.exposure,
            self.robot_trajectory,
            self.robot_mask_packet,
            timing_artifact_sha256=self.timing_artifact_sha256,
            trajectory_sha256=self.robot_trajectory_sha256,
            exact_action_sha256=self.robot_exact_action_sha256,
        )

    def canonical_metadata_bytes(self) -> bytes:
        """Return the exact canonical bytes whose digest forms ``bundle_id``."""

        return _canonical_json(self.canonical_metadata())


__all__ = [
    "BUNDLE_ID_PREFIX",
    "BUNDLE_SCHEMA",
    "FrameEvidenceBundle",
    "FrameEvidenceBundleError",
    "ImmutableCameraFrame",
]
