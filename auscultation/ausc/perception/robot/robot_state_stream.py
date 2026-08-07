"""Strict, fail-closed OpenArm telemetry buffering and exposure lookup.

Schema v1 contains a 100 Hz state/reference snapshot and is retained only for
diagnostic compatibility.  It is *not* action ground truth.

Schema v2 carries batches of exact 1 kHz follower-control records.  Each v2
record contains the ``MITParam`` values consumed by ``mit_control_all`` in one
follower control tick.  Observation interpolation remains available for
camera/geometry work, but actions are exposed only as exact tick records and
are never interpolated.

Schema v3 additionally carries the full-frame return status for every arm and
gripper SocketCAN write in that tick.  This proves only successful enqueue to
the Linux SocketCAN transmit path.  It is deliberately not represented as a
motor receipt or execution acknowledgement.

Schema v4 adds per-motor STATE RX counters, generations, CLOCK_MONOTONIC
timestamps, and cumulative maximum gaps.  They advance only after a matching
motor ID decodes successfully and never depend on joint motion.

This module only parses datagrams and retains a small time-bounded history; it
never opens CAN, changes commands, or owns a control resource.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROBOT_STATE_SCHEMA = "wam_openarm_robot_state_v1"
EXACT_CONTROL_RECORD_SCHEMA = "wam_openarm_exact_control_record_v2"
CONTROL_ENVELOPE_SCHEMA = "wam_openarm_control_envelope_v2"
EXACT_CONTROL_RECORD_SCHEMA_V3 = "wam_openarm_exact_control_record_v3"
CONTROL_ENVELOPE_SCHEMA_V3 = "wam_openarm_control_envelope_v3"
EXACT_CONTROL_RECORD_SCHEMA_V4 = "wam_openarm_exact_control_record_v4"
CONTROL_ENVELOPE_SCHEMA_V4 = "wam_openarm_control_envelope_v4"
CLOCK_ID = "CLOCK_MONOTONIC"
MAX_DATAGRAM_BYTES = 16_384
MAX_CONTROL_ENVELOPE_BYTES = 131_072
# Hard bound on the gap between consecutive 1 kHz control-tick observations.
# Nominal is 1 ms; the bound was 2 ms under the RT-scheduled-unshared-box
# assumption. Without rtprio on the shared 32-core box, 2-6 ms CFS preemptions
# of the control thread are routine and LOSE NOTHING (control_tick_seq stays
# contiguous; a lost tick still faults via the sequence checks) -- the gap only
# measures tick LATENESS. One constant shared by receiver, coordinator, stream
# and session writer so bring-up READY and capture acceptance can never disagree.
MAX_CONTROL_TICK_GAP_NS = 10_000_000

# Max missed 100 Hz envelopes absorbed as ONE bounded, exactly tick-aligned
# publisher hole (3 s) before the stream latches a terminal fault. The shared
# box's low-priority publisher thread gets starved for hundreds of ms by
# bring-up-adjacent CPU bursts while the 1 kHz tick counter keeps running
# (observed live 2026-08-07: 170-790 ms tick-aligned holes at READY+1.1 s).
# This absorption applies ONLY to opt-in live preflight/monitoring streams —
# session-capture streams stay strict, so RECORDED telemetry keeps its gapless
# guarantee and a hole during recording still fails fast. Consistent with the
# operator's doctrine: telemetry is advisory pre-recording; the arm is never
# power-cut by software. A real loss (bigger, tick-misaligned, duplicated, or
# backwards) still faults immediately. Single source of truth — the socket
# receiver imports this.
# (History 2026-08-07: 5 -> 100 -> 300 envelopes, then exact tick-alignment
# corroboration proved WRONG against the real C++ publisher — across a stall the
# ring/publisher tick-seq relationship shifts ('expected 552 got 567' with
# tick_delta 141 vs the aligned 151). Shape-based corroboration is abandoned:
# an advisory-stream hole is legal iff BOTH clocks moved forward by <= this
# bound and ticks progressed. Recording streams never absorb anything.)
_MAX_HOLE_NS = 3_500_000_000
_ENVELOPE_HOLE_TOLERANCE = 300  # retained for import compatibility/reporting
VOID_SEND_STATUS_REASON = "upstream_mit_control_all_returns_void"
SOCKETCAN_TX_STATUS_REASON = "openarm_can_1_2_2_full_frame_write_results"
SOCKETCAN_TX_ENQUEUE_SEMANTICS = (
    "linux_socketcan_full_frame_write_tx_enqueue_not_motor_ack"
)
MOTOR_ACK_UNOBSERVABLE_REASON = (
    "socketcan_tx_enqueue_does_not_acknowledge_motor_receipt_or_execution"
)
FEEDBACK_LIVENESS_REASON = (
    "openarm_can_1_2_2_matching_motor_id_valid_state_decode"
)
FEEDBACK_LIVENESS_SEMANTICS = (
    "counter_generation_increment_only_after_matching_motor_id_valid_state_decode"
)
FEEDBACK_MAX_AGE_NS = 20_000_000
FEEDBACK_MAX_GAP_NS = 20_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_START_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
_CAN_INTERFACES = frozenset({"can0", "can1", "can2", "can3"})
_PACKET_FIELDS = frozenset(
    {
        "schema",
        "seq",
        "clock_id",
        "monotonic_ns",
        "controller_pid",
        "controller_start_token",
        "side",
        "controller_mode",
        "leader_can_interface",
        "follower_can_interface",
        "joint_names",
        "follower_q_rad",
        "follower_qd_rad_s",
        "follower_gripper_motor_position_rad",
        "leader_q_rad",
        "leader_qd_rad_s",
        "leader_gripper_motor_position_rad",
        "follower_command_q_rad",
        "follower_command_gripper_motor_position_rad",
        "urdf_sha256",
        "joint_mapper_sha256",
        "pair_contract_sha256",
        "controller_binary_sha256",
    }
)


class RobotTelemetryError(ValueError):
    """Telemetry cannot safely be used for a camera exposure."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RobotTelemetryError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RobotTelemetryError(f"{label} must be finite")
    return result


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RobotTelemetryError(f"{label} SHA-256 is invalid")
    return value


def _numeric_vector(value: Any, *, label: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 7:
        raise RobotTelemetryError(f"{label} must contain exactly 7 values")
    return tuple(
        _finite_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _strict_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RobotTelemetryError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise RobotTelemetryError(f"non-finite JSON constant is forbidden: {value}")


def default_openarm_can_pair(side: str) -> Tuple[str, str]:
    """Return the upstream wiring default without overriding an activated pair."""

    if side == "right":
        return "can0", "can2"
    if side == "left":
        return "can1", "can3"
    raise ValueError("robot side must be left or right")


@dataclass(frozen=True)
class RobotStateContract:
    """Activated identities and bounds for one unilateral leader/follower pair."""

    side: str
    controller_mode: str
    joint_names: Tuple[str, ...]
    joint_position_limits_rad: Tuple[Tuple[float, float], ...]
    leader_can_interface: str
    follower_can_interface: str
    urdf_sha256: str
    joint_mapper_sha256: str
    camera_profile_sha256: str
    timing_artifact_sha256: str
    pair_contract_sha256: str
    controller_binary_sha256: str
    max_abs_joint_velocity_rad_s: float
    gripper_motor_position_limits_rad: Tuple[float, float]
    buffer_duration_ns: int = 2_000_000_000
    max_packets: int = 256
    max_gap_ns: int = 20_000_000
    max_staleness_ns: int = 50_000_000
    max_exposure_interval_ns: int = 40_000_000
    controller_source_manifest_sha256: str = ""
    follower_config_sha256: str = ""
    openarm_sdk_sha256: str = ""
    # P0-5: the activated follower STATE RX motor-ID mapping. When pinned (non-empty),
    # the runtime decode must match these EXACTLY in joint order, so a permuted or
    # mis-mapped controller cannot pass as valid. Empty = unpinned (legacy 7-unique
    # fallback, used by lower-level tests that do not carry an activation).
    expected_arm_rx_motor_ids: Tuple[int, ...] = ()
    expected_gripper_rx_motor_id: int = 0

    def __post_init__(self) -> None:
        joint_names = tuple(self.joint_names)
        limits = tuple(tuple(pair) for pair in self.joint_position_limits_rad)
        gripper_limits = tuple(self.gripper_motor_position_limits_rad)
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(self, "joint_position_limits_rad", limits)
        object.__setattr__(
            self, "gripper_motor_position_limits_rad", gripper_limits
        )

        if self.side not in ("left", "right"):
            raise ValueError("robot side must be left or right")
        if self.controller_mode != "unilateral":
            raise ValueError("controller mode must be unilateral for complete IL telemetry")
        for interface in (
            self.leader_can_interface,
            self.follower_can_interface,
        ):
            if interface not in _CAN_INTERFACES:
                raise ValueError("CAN interface must be one of can0, can1, can2, or can3")
        if self.leader_can_interface == self.follower_can_interface:
            raise ValueError("leader and follower CAN interfaces must be different")
        canonical = tuple(
            f"openarm_{self.side}_joint{index}" for index in range(1, 8)
        )
        if joint_names != canonical:
            raise ValueError("joint names must match the canonical seven-joint order")
        if len(limits) != 7:
            raise ValueError("joint position limits must contain seven pairs")
        for pair in limits:
            if len(pair) != 2:
                raise ValueError("joint position limits must contain lower/upper pairs")
            lower, upper = pair
            if (
                isinstance(lower, bool)
                or isinstance(upper, bool)
                or not isinstance(lower, (int, float))
                or not isinstance(upper, (int, float))
                or not math.isfinite(float(lower))
                or not math.isfinite(float(upper))
                or float(lower) >= float(upper)
            ):
                raise ValueError("joint position limits must be finite and ordered")

        for digest, label in (
            (self.urdf_sha256, "URDF"),
            (self.joint_mapper_sha256, "joint mapper"),
            (self.camera_profile_sha256, "camera profile"),
            (self.timing_artifact_sha256, "timing artifact"),
            (self.pair_contract_sha256, "pair contract"),
            (self.controller_binary_sha256, "controller binary"),
        ):
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{label} SHA-256 is invalid")

        velocity = self.max_abs_joint_velocity_rad_s
        if (
            isinstance(velocity, bool)
            or not isinstance(velocity, (int, float))
            or not math.isfinite(float(velocity))
            or float(velocity) <= 0.0
        ):
            raise ValueError("maximum joint velocity must be finite and positive")
        if len(gripper_limits) != 2:
            raise ValueError("gripper motor limits must contain lower and upper")
        gripper_lower, gripper_upper = gripper_limits
        if (
            isinstance(gripper_lower, bool)
            or isinstance(gripper_upper, bool)
            or not isinstance(gripper_lower, (int, float))
            or not isinstance(gripper_upper, (int, float))
            or not math.isfinite(float(gripper_lower))
            or not math.isfinite(float(gripper_upper))
            or float(gripper_lower) > float(gripper_upper)
        ):
            raise ValueError("gripper motor limits must be finite and ordered")

        for value, label in (
            (self.buffer_duration_ns, "buffer duration"),
            (self.max_packets, "maximum packet count"),
            (self.max_gap_ns, "maximum telemetry gap"),
            (self.max_staleness_ns, "maximum staleness"),
            (self.max_exposure_interval_ns, "maximum exposure interval"),
        ):
            if not _is_int(value) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.buffer_duration_ns != 2_000_000_000:
            raise ValueError("robot telemetry buffer duration must be exactly two seconds")
        if self.max_packets < 201:
            raise ValueError(
                "maximum packet count must retain at least 201 samples at 100 Hz"
            )
        if self.max_gap_ns > 20_000_000:
            raise ValueError("maximum telemetry gap cannot exceed 20 ms")
        for digest, label in (
            (
                self.controller_source_manifest_sha256,
                "controller source manifest",
            ),
            (self.follower_config_sha256, "follower controller config"),
            (self.openarm_sdk_sha256, "OpenArm SDK"),
        ):
            if digest and _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{label} SHA-256 is invalid")

        # P0-5: validate the pinned follower STATE RX motor-ID mapping when present.
        arm_rx_ids = tuple(int(motor_id) for motor_id in self.expected_arm_rx_motor_ids)
        object.__setattr__(self, "expected_arm_rx_motor_ids", arm_rx_ids)
        if arm_rx_ids:
            if (
                len(arm_rx_ids) != 7
                or len(set(arm_rx_ids)) != 7
                or any(not (0 < motor_id < 2**32) for motor_id in arm_rx_ids)
            ):
                raise ValueError(
                    "expected arm RX motor IDs must be seven unique positive uint32 values"
                )
            gripper_id = int(self.expected_gripper_rx_motor_id)
            if not (0 < gripper_id < 2**32) or gripper_id in arm_rx_ids:
                raise ValueError(
                    "expected gripper RX motor ID must be a positive uint32 distinct "
                    "from the arm IDs"
                )


@dataclass(frozen=True)
class RobotStatePacket:
    """One validated publisher datagram."""

    schema: str
    seq: int
    clock_id: str
    monotonic_ns: int
    controller_pid: int
    controller_start_token: str
    side: str
    controller_mode: str
    leader_can_interface: str
    follower_can_interface: str
    joint_names: Tuple[str, ...]
    follower_q_rad: Tuple[float, ...]
    follower_qd_rad_s: Tuple[float, ...]
    follower_gripper_motor_position_rad: float
    leader_q_rad: Tuple[float, ...]
    leader_qd_rad_s: Tuple[float, ...]
    leader_gripper_motor_position_rad: float
    follower_command_q_rad: Tuple[float, ...]
    follower_command_gripper_motor_position_rad: float
    urdf_sha256: str
    joint_mapper_sha256: str
    pair_contract_sha256: str
    controller_binary_sha256: str

    @property
    def process_identity(self) -> Tuple[int, str]:
        return self.controller_pid, self.controller_start_token

    @property
    def publisher_identity(self) -> str:
        return (
            f"pid={self.controller_pid};start={self.controller_start_token};"
            f"mode={self.controller_mode};leader_can={self.leader_can_interface};"
            f"follower_can={self.follower_can_interface};"
            f"urdf={self.urdf_sha256};mapper={self.joint_mapper_sha256};"
            f"pair={self.pair_contract_sha256};"
            f"binary={self.controller_binary_sha256}"
        )

    @property
    def action_ground_truth(self) -> bool:
        """v1 copied references are never authorized as consumed actions."""

        return False


@dataclass(frozen=True)
class InterpolatedRobotState:
    """Interpolated observation at one monotonic instant.

    ``follower_command_*`` is populated only when ``monotonic_ns`` is exactly
    one raw v1 packet time.  It is ``None`` between packets so callers cannot
    silently treat a linearly blended reference as an action.
    """

    monotonic_ns: int
    side: str
    controller_mode: str
    leader_can_interface: str
    follower_can_interface: str
    joint_names: Tuple[str, ...]
    follower_q_rad: Tuple[float, ...]
    follower_qd_rad_s: Tuple[float, ...]
    follower_gripper_motor_position_rad: float
    leader_q_rad: Tuple[float, ...]
    leader_qd_rad_s: Tuple[float, ...]
    leader_gripper_motor_position_rad: float
    follower_command_q_rad: Tuple[float, ...] | None
    follower_command_gripper_motor_position_rad: float | None
    action_is_exact: bool
    controller_pid: int
    controller_start_token: str
    urdf_sha256: str
    joint_mapper_sha256: str
    pair_contract_sha256: str
    controller_binary_sha256: str
    source_sequence_before: int
    source_sequence_after: int

    @property
    def process_identity(self) -> Tuple[int, str]:
        return self.controller_pid, self.controller_start_token


def parse_robot_state_datagram(
    payload: bytes,
    contract: RobotStateContract,
) -> RobotStatePacket:
    """Parse one exact JSON datagram and bind it to the activated contract."""

    if not isinstance(contract, RobotStateContract):
        raise TypeError("contract must be RobotStateContract")
    if not isinstance(payload, bytes):
        raise RobotTelemetryError("robot-state datagram must be bytes")
    if len(payload) > MAX_DATAGRAM_BYTES:
        raise RobotTelemetryError("robot-state datagram is too large")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RobotTelemetryError("robot-state datagram is not UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except RobotTelemetryError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RobotTelemetryError("invalid robot-state JSON") from exc
    if not isinstance(value, dict):
        raise RobotTelemetryError("robot-state JSON root must be an object")

    fields = frozenset(value)
    if fields != _PACKET_FIELDS:
        missing = sorted(_PACKET_FIELDS - fields)
        extra = sorted(fields - _PACKET_FIELDS)
        raise RobotTelemetryError(
            f"robot-state fields do not match contract; missing={missing}, extra={extra}"
        )
    if value["schema"] != ROBOT_STATE_SCHEMA:
        raise RobotTelemetryError("robot-state schema is unsupported")
    if value["clock_id"] != CLOCK_ID:
        raise RobotTelemetryError("robot-state clock must be CLOCK_MONOTONIC")

    seq = value["seq"]
    if not _is_int(seq) or not 0 <= seq < 2**64:
        raise RobotTelemetryError("seq must be an unsigned 64-bit integer")
    timestamp = value["monotonic_ns"]
    if not _is_int(timestamp) or not 0 < timestamp < 2**63:
        raise RobotTelemetryError("monotonic_ns must be a positive signed 64-bit integer")
    pid = value["controller_pid"]
    if not _is_int(pid) or not 0 < pid < 2**31:
        raise RobotTelemetryError("controller_pid must be a positive process ID")
    start_token = value["controller_start_token"]
    if (
        not isinstance(start_token, str)
        or _START_TOKEN_RE.fullmatch(start_token) is None
    ):
        raise RobotTelemetryError("controller start token is invalid")
    if value["side"] != contract.side:
        raise RobotTelemetryError("robot side does not match the activated contract")
    if value["controller_mode"] != "unilateral":
        raise RobotTelemetryError(
            "controller mode must be unilateral; incomplete action telemetry is forbidden"
        )
    leader_can = value["leader_can_interface"]
    follower_can = value["follower_can_interface"]
    if leader_can not in _CAN_INTERFACES:
        raise RobotTelemetryError("leader CAN interface must be one of can0 through can3")
    if follower_can not in _CAN_INTERFACES:
        raise RobotTelemetryError("follower CAN interface must be one of can0 through can3")
    if leader_can == follower_can:
        raise RobotTelemetryError("leader and follower CAN interfaces must be different")
    if leader_can != contract.leader_can_interface:
        raise RobotTelemetryError(
            "leader CAN interface does not match the activated contract"
        )
    if follower_can != contract.follower_can_interface:
        raise RobotTelemetryError(
            "follower CAN interface does not match the activated contract"
        )
    joint_names = value["joint_names"]
    if not isinstance(joint_names, list) or tuple(joint_names) != contract.joint_names:
        raise RobotTelemetryError("robot joint names/order do not match the activated contract")

    follower_q = _numeric_vector(value["follower_q_rad"], label="follower_q_rad")
    follower_qd = _numeric_vector(
        value["follower_qd_rad_s"], label="follower_qd_rad_s"
    )
    leader_q = _numeric_vector(value["leader_q_rad"], label="leader_q_rad")
    leader_qd = _numeric_vector(
        value["leader_qd_rad_s"], label="leader_qd_rad_s"
    )
    follower_command = _numeric_vector(
        value["follower_command_q_rad"], label="follower_command_q_rad"
    )
    follower_gripper = _finite_number(
        value["follower_gripper_motor_position_rad"],
        label="follower gripper motor position",
    )
    leader_gripper = _finite_number(
        value["leader_gripper_motor_position_rad"],
        label="leader gripper motor position",
    )
    follower_command_gripper = _finite_number(
        value["follower_command_gripper_motor_position_rad"],
        label="follower command gripper motor position",
    )

    for label, vector in (
        ("follower", follower_q),
        ("leader", leader_q),
        ("follower command", follower_command),
    ):
        for index, (position, limits) in enumerate(
            zip(vector, contract.joint_position_limits_rad)
        ):
            lower, upper = limits
            if not float(lower) <= position <= float(upper):
                raise RobotTelemetryError(
                    f"{label} joint {index + 1} is outside activated joint limits"
                )
    velocity_limit = float(contract.max_abs_joint_velocity_rad_s)
    if any(
        abs(value_) > velocity_limit
        for value_ in follower_qd + leader_qd
    ):
        raise RobotTelemetryError(
            "leader/follower joint velocity exceeds the activated velocity limit"
        )
    gripper_lower, gripper_upper = (
        float(value_) for value_ in contract.gripper_motor_position_limits_rad
    )
    for label, gripper in (
        ("follower", follower_gripper),
        ("leader", leader_gripper),
        ("follower command", follower_command_gripper),
    ):
        if not gripper_lower <= gripper <= gripper_upper:
            raise RobotTelemetryError(
                f"{label} gripper motor position is outside the activated limits"
            )

    urdf_sha = _sha256(value["urdf_sha256"], label="URDF")
    if urdf_sha != contract.urdf_sha256:
        raise RobotTelemetryError("URDF SHA-256 does not match the activated contract")
    mapper_sha = _sha256(value["joint_mapper_sha256"], label="joint mapper")
    if mapper_sha != contract.joint_mapper_sha256:
        raise RobotTelemetryError(
            "joint mapper SHA-256 does not match the activated contract"
        )
    pair_sha = _sha256(value["pair_contract_sha256"], label="pair contract")
    if pair_sha != contract.pair_contract_sha256:
        raise RobotTelemetryError(
            "pair contract SHA-256 does not match the activated contract"
        )
    binary_sha = _sha256(
        value["controller_binary_sha256"], label="controller binary"
    )
    if binary_sha != contract.controller_binary_sha256:
        raise RobotTelemetryError(
            "controller binary SHA-256 does not match the activated contract"
        )

    return RobotStatePacket(
        schema=ROBOT_STATE_SCHEMA,
        seq=seq,
        clock_id=CLOCK_ID,
        monotonic_ns=timestamp,
        controller_pid=pid,
        controller_start_token=start_token,
        side=contract.side,
        controller_mode="unilateral",
        leader_can_interface=leader_can,
        follower_can_interface=follower_can,
        joint_names=contract.joint_names,
        follower_q_rad=follower_q,
        follower_qd_rad_s=follower_qd,
        follower_gripper_motor_position_rad=follower_gripper,
        leader_q_rad=leader_q,
        leader_qd_rad_s=leader_qd,
        leader_gripper_motor_position_rad=leader_gripper,
        follower_command_q_rad=follower_command,
        follower_command_gripper_motor_position_rad=follower_command_gripper,
        urdf_sha256=urdf_sha,
        joint_mapper_sha256=mapper_sha,
        pair_contract_sha256=pair_sha,
        controller_binary_sha256=binary_sha,
    )


def _interpolate(
    before: RobotStatePacket,
    after: RobotStatePacket,
    timestamp_ns: int,
) -> InterpolatedRobotState:
    if not before.monotonic_ns <= timestamp_ns <= after.monotonic_ns:
        raise RobotTelemetryError("trajectory extrapolation is forbidden")
    if before.process_identity != after.process_identity:
        raise RobotTelemetryError("cannot interpolate across a process identity change")
    if (
        before.urdf_sha256 != after.urdf_sha256
        or before.joint_mapper_sha256 != after.joint_mapper_sha256
        or before.side != after.side
        or before.controller_mode != after.controller_mode
        or before.leader_can_interface != after.leader_can_interface
        or before.follower_can_interface != after.follower_can_interface
        or before.joint_names != after.joint_names
        or before.pair_contract_sha256 != after.pair_contract_sha256
        or before.controller_binary_sha256 != after.controller_binary_sha256
    ):
        raise RobotTelemetryError("cannot interpolate across a provenance change")

    if before.monotonic_ns == after.monotonic_ns:
        alpha = 0.0
    else:
        alpha = (timestamp_ns - before.monotonic_ns) / (
            after.monotonic_ns - before.monotonic_ns
        )

    def blend(left: Sequence[float], right: Sequence[float]) -> Tuple[float, ...]:
        return tuple(
            float(left_value) + alpha * (float(right_value) - float(left_value))
            for left_value, right_value in zip(left, right)
        )

    action_is_exact = (
        before.seq == after.seq and timestamp_ns == before.monotonic_ns
    )
    return InterpolatedRobotState(
        monotonic_ns=timestamp_ns,
        side=before.side,
        controller_mode=before.controller_mode,
        leader_can_interface=before.leader_can_interface,
        follower_can_interface=before.follower_can_interface,
        joint_names=before.joint_names,
        follower_q_rad=blend(before.follower_q_rad, after.follower_q_rad),
        follower_qd_rad_s=blend(
            before.follower_qd_rad_s, after.follower_qd_rad_s
        ),
        follower_gripper_motor_position_rad=(
            before.follower_gripper_motor_position_rad
            + alpha
            * (
                after.follower_gripper_motor_position_rad
                - before.follower_gripper_motor_position_rad
            )
        ),
        leader_q_rad=blend(before.leader_q_rad, after.leader_q_rad),
        leader_qd_rad_s=blend(before.leader_qd_rad_s, after.leader_qd_rad_s),
        leader_gripper_motor_position_rad=(
            before.leader_gripper_motor_position_rad
            + alpha
            * (
                after.leader_gripper_motor_position_rad
                - before.leader_gripper_motor_position_rad
            )
        ),
        follower_command_q_rad=(
            before.follower_command_q_rad if action_is_exact else None
        ),
        follower_command_gripper_motor_position_rad=(
            before.follower_command_gripper_motor_position_rad
            if action_is_exact
            else None
        ),
        action_is_exact=action_is_exact,
        controller_pid=before.controller_pid,
        controller_start_token=before.controller_start_token,
        urdf_sha256=before.urdf_sha256,
        joint_mapper_sha256=before.joint_mapper_sha256,
        pair_contract_sha256=before.pair_contract_sha256,
        controller_binary_sha256=before.controller_binary_sha256,
        source_sequence_before=before.seq,
        source_sequence_after=after.seq,
    )


@dataclass(frozen=True)
class BracketedRobotTrajectory:
    """Raw brackets plus interpolation across one complete exposure interval."""

    exposure_lower_ns: int
    exposure_nominal_ns: int
    exposure_upper_ns: int
    packets: Tuple[RobotStatePacket, ...]

    @property
    def before(self) -> RobotStatePacket:
        return self.packets[0]

    @property
    def after(self) -> RobotStatePacket:
        return self.packets[-1]

    @property
    def sequence_first(self) -> int:
        return self.before.seq

    @property
    def sequence_last(self) -> int:
        return self.after.seq

    @property
    def max_gap_ns(self) -> int:
        return max(
            right.monotonic_ns - left.monotonic_ns
            for left, right in zip(self.packets, self.packets[1:])
        ) if len(self.packets) > 1 else 0

    @property
    def publisher_identity(self) -> str:
        return self.before.publisher_identity

    @property
    def at_lower(self) -> InterpolatedRobotState:
        return self.sample_at(self.exposure_lower_ns)

    @property
    def at_nominal(self) -> InterpolatedRobotState:
        return self.sample_at(self.exposure_nominal_ns)

    @property
    def at_upper(self) -> InterpolatedRobotState:
        return self.sample_at(self.exposure_upper_ns)

    def sample_at(self, timestamp_ns: int) -> InterpolatedRobotState:
        if not _is_int(timestamp_ns):
            raise RobotTelemetryError("trajectory sample time must be integer nanoseconds")
        if not self.before.monotonic_ns <= timestamp_ns <= self.after.monotonic_ns:
            raise RobotTelemetryError("trajectory extrapolation is forbidden")
        timestamps = tuple(packet.monotonic_ns for packet in self.packets)
        right_index = bisect_left(timestamps, timestamp_ns)
        if (
            right_index < len(self.packets)
            and self.packets[right_index].monotonic_ns == timestamp_ns
        ):
            packet = self.packets[right_index]
            return _interpolate(packet, packet, timestamp_ns)
        if right_index == 0 or right_index >= len(self.packets):
            raise RobotTelemetryError("trajectory extrapolation is forbidden")
        return _interpolate(
            self.packets[right_index - 1],
            self.packets[right_index],
            timestamp_ns,
        )

    def sample_many(
        self, timestamps_ns: Iterable[int]
    ) -> Tuple[InterpolatedRobotState, ...]:
        return tuple(self.sample_at(timestamp) for timestamp in timestamps_ns)

    def sample_exposure_trajectory(self) -> Tuple[InterpolatedRobotState, ...]:
        """Sample boundaries, nominal time, and every raw state inside exposure."""

        timestamps = {
            self.exposure_lower_ns,
            self.exposure_nominal_ns,
            self.exposure_upper_ns,
        }
        timestamps.update(
            packet.monotonic_ns
            for packet in self.packets
            if self.exposure_lower_ns
            < packet.monotonic_ns
            < self.exposure_upper_ns
        )
        return self.sample_many(sorted(timestamps))


class RobotStateStream:
    """Two-second receiver buffer that permanently latches any stream fault."""

    def __init__(self, contract: RobotStateContract):
        if not isinstance(contract, RobotStateContract):
            raise TypeError("contract must be RobotStateContract")
        self.contract = contract
        self._packets: List[RobotStatePacket] = []
        self._last_sequence = None
        self._last_monotonic_ns = None
        self._process_identity = None
        self._fault_reason = ""

    @property
    def faulted(self) -> bool:
        return bool(self._fault_reason)

    @property
    def fault_reason(self) -> str:
        return self._fault_reason

    @property
    def packet_count(self) -> int:
        return len(self._packets)

    @property
    def oldest_monotonic_ns(self) -> int:
        if not self._packets:
            raise RobotTelemetryError("robot telemetry buffer is empty")
        return self._packets[0].monotonic_ns

    @property
    def latest_monotonic_ns(self) -> int:
        if not self._packets:
            raise RobotTelemetryError("robot telemetry buffer is empty")
        return self._packets[-1].monotonic_ns

    def _raise_if_faulted(self) -> None:
        if self.faulted:
            raise RobotTelemetryError(
                f"robot telemetry stream is faulted: {self._fault_reason}"
            )

    def _latch_fault(self, reason: str) -> None:
        if not self._fault_reason:
            self._fault_reason = reason
        raise RobotTelemetryError(reason)

    def ingest_datagram(self, payload: bytes) -> RobotStatePacket:
        self._raise_if_faulted()
        try:
            packet = parse_robot_state_datagram(payload, self.contract)
        except RobotTelemetryError as exc:
            self._latch_fault(str(exc))

        if self._last_sequence is not None:
            if packet.process_identity != self._process_identity:
                self._latch_fault("robot telemetry process identity changed")
            if packet.seq != self._last_sequence + 1:
                self._latch_fault(
                    "robot telemetry sequence loss or duplication: "
                    f"expected {self._last_sequence + 1}, received {packet.seq}"
                )
            if packet.monotonic_ns <= self._last_monotonic_ns:
                self._latch_fault("robot telemetry CLOCK_MONOTONIC clock rollback")
        else:
            self._process_identity = packet.process_identity

        self._packets.append(packet)
        self._last_sequence = packet.seq
        self._last_monotonic_ns = packet.monotonic_ns
        cutoff_ns = packet.monotonic_ns - self.contract.buffer_duration_ns
        while self._packets and self._packets[0].monotonic_ns < cutoff_ns:
            self._packets.pop(0)
        while len(self._packets) > self.contract.max_packets:
            self._packets.pop(0)
        return packet

    def query_exposure_interval(
        self,
        *,
        exposure_lower_ns: int,
        exposure_nominal_ns: int,
        exposure_upper_ns: int,
        now_monotonic_ns: int,
        camera_profile_sha256: str,
        timing_artifact_sha256: str,
    ) -> BracketedRobotTrajectory:
        self._raise_if_faulted()
        for value, label in (
            (exposure_lower_ns, "exposure lower"),
            (exposure_nominal_ns, "exposure nominal"),
            (exposure_upper_ns, "exposure upper"),
            (now_monotonic_ns, "query clock"),
        ):
            if not _is_int(value) or value <= 0:
                raise RobotTelemetryError(f"{label} time must be positive integer ns")
        if not exposure_lower_ns <= exposure_nominal_ns <= exposure_upper_ns:
            raise RobotTelemetryError("exposure interval ordering is invalid")
        if (
            exposure_upper_ns - exposure_lower_ns
            > self.contract.max_exposure_interval_ns
        ):
            raise RobotTelemetryError(
                "exposure interval width exceeds the activated timing artifact"
            )
        if camera_profile_sha256 != self.contract.camera_profile_sha256:
            raise RobotTelemetryError("camera profile does not match the activated contract")
        if timing_artifact_sha256 != self.contract.timing_artifact_sha256:
            raise RobotTelemetryError("timing artifact does not match the activated contract")
        if not self._packets:
            raise RobotTelemetryError("robot telemetry cannot bracket an empty buffer")

        latest_ns = self._packets[-1].monotonic_ns
        if now_monotonic_ns < latest_ns:
            raise RobotTelemetryError("query clock precedes the latest robot telemetry")
        if now_monotonic_ns - latest_ns > self.contract.max_staleness_ns:
            raise RobotTelemetryError("robot telemetry buffer is stale")

        timestamps = tuple(packet.monotonic_ns for packet in self._packets)
        before_index = bisect_right(timestamps, exposure_lower_ns) - 1
        after_index = bisect_left(timestamps, exposure_upper_ns)
        if before_index < 0 or after_index >= len(self._packets):
            raise RobotTelemetryError(
                "robot telemetry does not bracket the complete exposure interval"
            )
        packets = tuple(self._packets[before_index:after_index + 1])
        if len(packets) < 2 and exposure_lower_ns != exposure_upper_ns:
            raise RobotTelemetryError(
                "robot telemetry does not bracket the complete exposure interval"
            )
        gaps = tuple(
            right.monotonic_ns - left.monotonic_ns
            for left, right in zip(packets, packets[1:])
        )
        if any(gap > self.contract.max_gap_ns for gap in gaps):
            raise RobotTelemetryError(
                "robot telemetry gap exceeds 20 ms inside the exposure bracket"
            )
        if any(gap <= 0 for gap in gaps):
            raise RobotTelemetryError("robot telemetry clock rollback in exposure bracket")

        return BracketedRobotTrajectory(
            exposure_lower_ns=exposure_lower_ns,
            exposure_nominal_ns=exposure_nominal_ns,
            exposure_upper_ns=exposure_upper_ns,
            packets=packets,
        )


_CONTROL_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "envelope_seq",
        "clock_id",
        "publisher_monotonic_ns",
        "controller_pid",
        "controller_start_token",
        "side",
        "controller_mode",
        "leader_can_interface",
        "follower_can_interface",
        "joint_names",
        "urdf_sha256",
        "joint_mapper_sha256",
        "pair_contract_sha256",
        "controller_binary_sha256",
        "controller_source_manifest_sha256",
        "follower_config_sha256",
        "openarm_sdk_sha256",
        "publisher_nominal_hz",
        "control_nominal_hz",
        "send_status_observable",
        "send_status_reason",
        "ring_overflow_count",
        "publisher_rejected_record_count",
        "socket_send_failure_count",
        "records",
    }
)
_CONTROL_ENVELOPE_FIELDS_V3 = _CONTROL_ENVELOPE_FIELDS | frozenset(
    {
        "send_status_semantics",
        "motor_ack_observable",
        "motor_ack_reason",
    }
)
_CONTROL_ENVELOPE_FIELDS_V4 = _CONTROL_ENVELOPE_FIELDS_V3 | frozenset(
    {
        "feedback_liveness_observable",
        "feedback_liveness_reason",
        "feedback_liveness_semantics",
        "feedback_max_age_ns",
        "feedback_max_gap_ns",
    }
)
_EXACT_CONTROL_RECORD_FIELDS = frozenset(
    {
        "schema",
        "control_tick_seq",
        "observation_monotonic_ns",
        "reference_generation",
        "reference_source_monotonic_ns",
        "reference_snapshot_valid",
        "control_call_begin_monotonic_ns",
        "arm_call_return_monotonic_ns",
        "gripper_call_begin_monotonic_ns",
        "gripper_call_return_monotonic_ns",
        "control_call_end_monotonic_ns",
        "follower_q_rad",
        "follower_qd_rad_s",
        "follower_gripper_q_rad",
        "follower_gripper_qd_rad_s",
        "follower_gripper_motor_q_rad",
        "follower_gripper_motor_qd_rad_s",
        "follower_arm_mit_kp",
        "follower_arm_mit_kd",
        "follower_arm_mit_q",
        "follower_arm_mit_dq",
        "follower_arm_mit_tau",
        "follower_gripper_mit_kp",
        "follower_gripper_mit_kd",
        "follower_gripper_mit_q",
        "follower_gripper_mit_dq",
        "follower_gripper_mit_tau",
    }
)
_EXACT_CONTROL_RECORD_FIELDS_V3 = _EXACT_CONTROL_RECORD_FIELDS | frozenset(
    {
        "arm_tx_expected_frame_count",
        "arm_tx_full_frame_write_count",
        "arm_tx_failed_frame_write_count",
        "gripper_tx_expected_frame_count",
        "gripper_tx_full_frame_write_count",
        "gripper_tx_failed_frame_write_count",
        "socketcan_tx_enqueue_success",
    }
)
_EXACT_CONTROL_RECORD_FIELDS_V4 = _EXACT_CONTROL_RECORD_FIELDS_V3 | frozenset(
    {
        "feedback_snapshot_monotonic_ns",
        "follower_arm_state_rx_motor_ids",
        "follower_arm_state_rx_counters",
        "follower_arm_state_rx_generations",
        "follower_arm_state_rx_monotonic_ns",
        "follower_arm_state_rx_last_gap_ns",
        "follower_arm_state_rx_gap_violation_counts",
        "follower_gripper_state_rx_motor_id",
        "follower_gripper_state_rx_counter",
        "follower_gripper_state_rx_generation",
        "follower_gripper_state_rx_monotonic_ns",
        "follower_gripper_state_rx_last_gap_ns",
        "follower_gripper_state_rx_gap_violation_count",
    }
)


@dataclass(frozen=True)
class ExactFollowerAction:
    """The exact motor-domain ``MITParam`` consumed in one follower tick."""

    arm_kp: Tuple[float, ...]
    arm_kd: Tuple[float, ...]
    arm_q: Tuple[float, ...]
    arm_dq: Tuple[float, ...]
    arm_tau: Tuple[float, ...]
    gripper_kp: float
    gripper_kd: float
    gripper_q: float
    gripper_dq: float
    gripper_tau: float


@dataclass(frozen=True)
class ExactControlRecord:
    """One exact v2/v3/v4 follower-control record flattened from an envelope."""

    schema: str
    control_tick_seq: int
    observation_monotonic_ns: int
    reference_generation: int
    reference_source_monotonic_ns: int
    reference_snapshot_valid: bool
    control_call_begin_monotonic_ns: int
    arm_call_return_monotonic_ns: int
    gripper_call_begin_monotonic_ns: int
    gripper_call_return_monotonic_ns: int
    control_call_end_monotonic_ns: int
    follower_q_rad: Tuple[float, ...]
    follower_qd_rad_s: Tuple[float, ...]
    follower_gripper_q_rad: float
    follower_gripper_qd_rad_s: float
    follower_gripper_motor_q_rad: float
    follower_gripper_motor_qd_rad_s: float
    action: ExactFollowerAction
    envelope_seq: int
    publisher_monotonic_ns: int
    controller_pid: int
    controller_start_token: str
    side: str
    controller_mode: str
    leader_can_interface: str
    follower_can_interface: str
    joint_names: Tuple[str, ...]
    urdf_sha256: str
    joint_mapper_sha256: str
    pair_contract_sha256: str
    controller_binary_sha256: str
    controller_source_manifest_sha256: str
    follower_config_sha256: str
    openarm_sdk_sha256: str
    send_status_observable: bool
    send_status_reason: str
    send_status_semantics: str = "unobservable_void_sdk_api"
    motor_ack_observable: bool = False
    arm_tx_expected_frame_count: int = 0
    arm_tx_full_frame_write_count: int = 0
    arm_tx_failed_frame_write_count: int = 0
    gripper_tx_expected_frame_count: int = 0
    gripper_tx_full_frame_write_count: int = 0
    gripper_tx_failed_frame_write_count: int = 0
    socketcan_tx_enqueue_success: bool = False
    feedback_liveness_observable: bool = False
    feedback_snapshot_monotonic_ns: int = 0
    follower_arm_state_rx_motor_ids: Tuple[int, ...] = ()
    follower_arm_state_rx_counters: Tuple[int, ...] = ()
    follower_arm_state_rx_generations: Tuple[int, ...] = ()
    follower_arm_state_rx_monotonic_ns: Tuple[int, ...] = ()
    follower_arm_state_rx_last_gap_ns: Tuple[int, ...] = ()
    follower_arm_state_rx_gap_violation_counts: Tuple[int, ...] = ()
    follower_gripper_state_rx_motor_id: int = 0
    follower_gripper_state_rx_counter: int = 0
    follower_gripper_state_rx_generation: int = 0
    follower_gripper_state_rx_monotonic_ns: int = 0
    follower_gripper_state_rx_last_gap_ns: int = 0
    follower_gripper_state_rx_gap_violation_count: int = 0

    @property
    def seq(self) -> int:
        return self.control_tick_seq

    @property
    def monotonic_ns(self) -> int:
        """Observation time used only for continuous state interpolation."""

        return self.observation_monotonic_ns

    @property
    def process_identity(self) -> Tuple[int, str]:
        return self.controller_pid, self.controller_start_token

    @property
    def publisher_identity(self) -> str:
        return (
            f"pid={self.controller_pid};start={self.controller_start_token};"
            f"mode={self.controller_mode};leader_can={self.leader_can_interface};"
            f"follower_can={self.follower_can_interface};"
            f"urdf={self.urdf_sha256};mapper={self.joint_mapper_sha256};"
            f"pair={self.pair_contract_sha256};"
            f"binary={self.controller_binary_sha256};"
            f"source={self.controller_source_manifest_sha256};"
            f"config={self.follower_config_sha256};sdk={self.openarm_sdk_sha256}"
        )

    @property
    def action_parameters_exact(self) -> bool:
        return True

    @property
    def action_send_status_verified(self) -> bool:
        return (
            self.send_status_observable
            and self.send_status_semantics == SOCKETCAN_TX_ENQUEUE_SEMANTICS
            and self.arm_tx_expected_frame_count == 7
            and self.arm_tx_full_frame_write_count == 7
            and self.arm_tx_failed_frame_write_count == 0
            and self.gripper_tx_expected_frame_count == 1
            and self.gripper_tx_full_frame_write_count == 1
            and self.gripper_tx_failed_frame_write_count == 0
            and self.socketcan_tx_enqueue_success
        )

    def require_verified_send_status(self) -> None:
        """Fail closed when a training policy requires hardware send outcome."""

        if not self.send_status_observable:
            raise RobotTelemetryError(
                "follower MITParam is exact, but hardware send status is "
                f"unobservable: {self.send_status_reason}"
            )
        if not self.action_send_status_verified:
            raise RobotTelemetryError(
                "follower MITParam is exact, but at least one expected "
                "SocketCAN frame did not receive a full-frame TX enqueue result"
            )

    @property
    def feedback_rx_all_fresh(self) -> bool:
        if not self.feedback_liveness_observable:
            return False
        timestamps = self.follower_arm_state_rx_monotonic_ns + (
            self.follower_gripper_state_rx_monotonic_ns,
        )
        gaps = self.follower_arm_state_rx_last_gap_ns + (
            self.follower_gripper_state_rx_last_gap_ns,
        )
        return (
            all(
                0 < timestamp <= self.feedback_snapshot_monotonic_ns
                for timestamp in timestamps
            )
            and all(
                self.feedback_snapshot_monotonic_ns - timestamp
                <= FEEDBACK_MAX_AGE_NS
                for timestamp in timestamps
            )
            and all(0 <= gap <= FEEDBACK_MAX_GAP_NS for gap in gaps)
        )


@dataclass(frozen=True)
class RobotControlEnvelope:
    """One validated 100 Hz transport envelope containing exact 1 kHz ticks."""

    schema: str
    envelope_seq: int
    publisher_monotonic_ns: int
    controller_pid: int
    controller_start_token: str
    records: Tuple[ExactControlRecord, ...]
    # Cumulative publisher sendto-failure count carried for TRANSITION checks: a
    # constant baseline (pre-receiver drops) is tolerated; any INCREASE across
    # received envelopes is a mid-stream loss and must fault.
    socket_send_failure_count: int = 0

    @property
    def process_identity(self) -> Tuple[int, str]:
        return self.controller_pid, self.controller_start_token


@dataclass(frozen=True)
class InterpolatedFollowerObservation:
    """Continuous follower observation with deliberately no action fields."""

    monotonic_ns: int
    follower_q_rad: Tuple[float, ...]
    follower_qd_rad_s: Tuple[float, ...]
    follower_gripper_q_rad: float
    follower_gripper_qd_rad_s: float
    follower_gripper_motor_q_rad: float
    follower_gripper_motor_qd_rad_s: float
    source_sequence_before: int
    source_sequence_after: int


@dataclass(frozen=True)
class BracketedExactControlTrajectory:
    """Exact control ticks plus observation-only interpolation."""

    exposure_lower_ns: int
    exposure_nominal_ns: int
    exposure_upper_ns: int
    records: Tuple[ExactControlRecord, ...]

    @property
    def before(self) -> ExactControlRecord:
        return self.records[0]

    @property
    def after(self) -> ExactControlRecord:
        return self.records[-1]

    @property
    def sequence_first(self) -> int:
        return self.before.control_tick_seq

    @property
    def sequence_last(self) -> int:
        return self.after.control_tick_seq

    @property
    def max_gap_ns(self) -> int:
        if len(self.records) < 2:
            return 0
        return max(
            right.observation_monotonic_ns - left.observation_monotonic_ns
            for left, right in zip(self.records, self.records[1:])
        )

    @property
    def publisher_identity(self) -> str:
        return self.before.publisher_identity

    @property
    def exact_actions(self) -> Tuple[Tuple[int, ExactFollowerAction], ...]:
        """Return actions keyed only by their exact control tick sequence."""

        return tuple(
            (record.control_tick_seq, record.action) for record in self.records
        )

    def exact_action_for_tick(self, control_tick_seq: int) -> ExactFollowerAction:
        if not _is_int(control_tick_seq):
            raise RobotTelemetryError("control tick sequence must be an integer")
        for record in self.records:
            if record.control_tick_seq == control_tick_seq:
                return record.action
        raise RobotTelemetryError("no exact action exists for that control tick")

    def sample_observation_at(
        self, timestamp_ns: int
    ) -> InterpolatedFollowerObservation:
        """Interpolate observations only; action interpolation is unavailable."""

        if not _is_int(timestamp_ns):
            raise RobotTelemetryError("trajectory sample time must be integer nanoseconds")
        timestamps = tuple(
            record.observation_monotonic_ns for record in self.records
        )
        right_index = bisect_left(timestamps, timestamp_ns)
        if (
            right_index < len(self.records)
            and self.records[right_index].observation_monotonic_ns == timestamp_ns
        ):
            left = right = self.records[right_index]
        else:
            if right_index == 0 or right_index >= len(self.records):
                raise RobotTelemetryError("trajectory extrapolation is forbidden")
            left = self.records[right_index - 1]
            right = self.records[right_index]
        if left.process_identity != right.process_identity:
            raise RobotTelemetryError(
                "cannot interpolate observation across a process identity change"
            )
        if left.monotonic_ns == right.monotonic_ns:
            alpha = 0.0
        else:
            alpha = (timestamp_ns - left.monotonic_ns) / (
                right.monotonic_ns - left.monotonic_ns
            )

        def blend(
            left_values: Sequence[float], right_values: Sequence[float]
        ) -> Tuple[float, ...]:
            return tuple(
                float(a) + alpha * (float(b) - float(a))
                for a, b in zip(left_values, right_values)
            )

        return InterpolatedFollowerObservation(
            monotonic_ns=timestamp_ns,
            follower_q_rad=blend(left.follower_q_rad, right.follower_q_rad),
            follower_qd_rad_s=blend(
                left.follower_qd_rad_s, right.follower_qd_rad_s
            ),
            follower_gripper_q_rad=(
                left.follower_gripper_q_rad
                + alpha
                * (
                    right.follower_gripper_q_rad
                    - left.follower_gripper_q_rad
                )
            ),
            follower_gripper_qd_rad_s=(
                left.follower_gripper_qd_rad_s
                + alpha
                * (
                    right.follower_gripper_qd_rad_s
                    - left.follower_gripper_qd_rad_s
                )
            ),
            follower_gripper_motor_q_rad=(
                left.follower_gripper_motor_q_rad
                + alpha
                * (
                    right.follower_gripper_motor_q_rad
                    - left.follower_gripper_motor_q_rad
                )
            ),
            follower_gripper_motor_qd_rad_s=(
                left.follower_gripper_motor_qd_rad_s
                + alpha
                * (
                    right.follower_gripper_motor_qd_rad_s
                    - left.follower_gripper_motor_qd_rad_s
                )
            ),
            source_sequence_before=left.control_tick_seq,
            source_sequence_after=right.control_tick_seq,
        )


def _require_exact_fields(
    value: Any, expected: frozenset[str], *, label: str
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RobotTelemetryError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != expected:
        raise RobotTelemetryError(
            f"{label} fields do not match contract; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _u64(value: Any, *, label: str) -> int:
    if not _is_int(value) or not 0 <= value < 2**64:
        raise RobotTelemetryError(f"{label} must be an unsigned 64-bit integer")
    return value


def _positive_i64(value: Any, *, label: str) -> int:
    if not _is_int(value) or not 0 < value < 2**63:
        raise RobotTelemetryError(f"{label} must be a positive signed 64-bit integer")
    return value


def _u64_vector(value: Any, *, label: str, positive: bool) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != 7:
        raise RobotTelemetryError(f"{label} must contain exactly 7 integers")
    result = tuple(_u64(item, label=f"{label}[{index}]") for index, item in enumerate(value))
    if positive and any(item == 0 for item in result):
        raise RobotTelemetryError(f"{label} values must be positive")
    return result


def _nonnegative_i64_vector(value: Any, *, label: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != 7:
        raise RobotTelemetryError(f"{label} must contain exactly 7 integers")
    result = []
    for index, item in enumerate(value):
        if not _is_int(item) or not 0 <= item < 2**63:
            raise RobotTelemetryError(
                f"{label}[{index}] must be a non-negative signed 64-bit integer"
            )
        result.append(item)
    return tuple(result)


def _parse_exact_control_record(
    value: Any,
    *,
    contract: RobotStateContract,
    envelope: Dict[str, Any],
) -> ExactControlRecord:
    if not isinstance(value, dict):
        raise RobotTelemetryError("exact control record must be an object")
    record_schema = value.get("schema")
    if record_schema == EXACT_CONTROL_RECORD_SCHEMA:
        expected_fields = _EXACT_CONTROL_RECORD_FIELDS
    elif record_schema == EXACT_CONTROL_RECORD_SCHEMA_V3:
        expected_fields = _EXACT_CONTROL_RECORD_FIELDS_V3
    elif record_schema == EXACT_CONTROL_RECORD_SCHEMA_V4:
        expected_fields = _EXACT_CONTROL_RECORD_FIELDS_V4
    else:
        raise RobotTelemetryError("exact control record schema is unsupported")
    record = _require_exact_fields(
        value, expected_fields, label="exact control record"
    )
    envelope_schema = envelope["schema"]
    expected_record_schema = {
        CONTROL_ENVELOPE_SCHEMA: EXACT_CONTROL_RECORD_SCHEMA,
        CONTROL_ENVELOPE_SCHEMA_V3: EXACT_CONTROL_RECORD_SCHEMA_V3,
        CONTROL_ENVELOPE_SCHEMA_V4: EXACT_CONTROL_RECORD_SCHEMA_V4,
    }[envelope_schema]
    if record_schema != expected_record_schema:
        raise RobotTelemetryError(
            "exact control record schema does not match its envelope"
        )
    control_tick_seq = _u64(
        record["control_tick_seq"], label="control_tick_seq"
    )
    observation_ns = _positive_i64(
        record["observation_monotonic_ns"],
        label="observation_monotonic_ns",
    )
    reference_generation = _u64(
        record["reference_generation"], label="reference_generation"
    )
    reference_source_ns = _positive_i64(
        record["reference_source_monotonic_ns"],
        label="reference_source_monotonic_ns",
    )
    # reference_snapshot_valid=false marks a controller-side seqlock read collision:
    # the leader thread updated the reference epoch during this tick's read, so THIS
    # tick's reference provenance is uncertain (observations remain exact; the
    # consumed command was still sent). The controller does not retry the read, so
    # isolated collisions are expected at 1 kHz. Carry the flag through instead of
    # rejecting the record here — the receiver faults on SUSTAINED runs (>=3
    # consecutive invalid ticks), which is what an actually torn stream looks like.
    if not isinstance(record["reference_snapshot_valid"], bool):
        raise RobotTelemetryError(
            "reference_snapshot_valid must be a boolean"
        )
    arm_begin_ns = _positive_i64(
        record["control_call_begin_monotonic_ns"],
        label="control_call_begin_monotonic_ns",
    )
    arm_return_ns = _positive_i64(
        record["arm_call_return_monotonic_ns"],
        label="arm_call_return_monotonic_ns",
    )
    gripper_begin_ns = _positive_i64(
        record["gripper_call_begin_monotonic_ns"],
        label="gripper_call_begin_monotonic_ns",
    )
    gripper_return_ns = _positive_i64(
        record["gripper_call_return_monotonic_ns"],
        label="gripper_call_return_monotonic_ns",
    )
    control_end_ns = _positive_i64(
        record["control_call_end_monotonic_ns"],
        label="control_call_end_monotonic_ns",
    )
    if not (
        reference_source_ns <= arm_begin_ns
        and observation_ns <= arm_begin_ns
        <= arm_return_ns
        <= gripper_begin_ns
        <= gripper_return_ns
        <= control_end_ns
        <= envelope["publisher_monotonic_ns"]
    ):
        raise RobotTelemetryError("exact control record timestamp ordering is invalid")
    if reference_generation <= 0:
        raise RobotTelemetryError("reference generation must be positive")

    if record_schema in {
        EXACT_CONTROL_RECORD_SCHEMA_V3,
        EXACT_CONTROL_RECORD_SCHEMA_V4,
    }:
        tx_counts = {}
        for field in (
            "arm_tx_expected_frame_count",
            "arm_tx_full_frame_write_count",
            "arm_tx_failed_frame_write_count",
            "gripper_tx_expected_frame_count",
            "gripper_tx_full_frame_write_count",
            "gripper_tx_failed_frame_write_count",
        ):
            count = record[field]
            if not _is_int(count) or count < 0:
                raise RobotTelemetryError(
                    f"{field} must be a non-negative integer"
                )
            tx_counts[field] = count
        if tx_counts["arm_tx_expected_frame_count"] != 7:
            raise RobotTelemetryError(
                "v3/v4 requires exactly seven arm SocketCAN frames per tick"
            )
        if tx_counts["gripper_tx_expected_frame_count"] != 1:
            raise RobotTelemetryError(
                "v3/v4 requires exactly one gripper SocketCAN frame per tick"
            )
        for group in ("arm", "gripper"):
            expected = tx_counts[f"{group}_tx_expected_frame_count"]
            full = tx_counts[f"{group}_tx_full_frame_write_count"]
            failed = tx_counts[f"{group}_tx_failed_frame_write_count"]
            if full + failed != expected:
                raise RobotTelemetryError(
                    f"{group} SocketCAN write counts do not equal expected frames"
                )
        tx_success = record["socketcan_tx_enqueue_success"]
        if not isinstance(tx_success, bool):
            raise RobotTelemetryError(
                "SocketCAN TX enqueue success flag must be boolean"
            )
        computed_success = (
            tx_counts["arm_tx_full_frame_write_count"] == 7
            and tx_counts["arm_tx_failed_frame_write_count"] == 0
            and tx_counts["gripper_tx_full_frame_write_count"] == 1
            and tx_counts["gripper_tx_failed_frame_write_count"] == 0
        )
        if tx_success is not computed_success:
            raise RobotTelemetryError(
                "SocketCAN TX enqueue success flag disagrees with write counts"
            )
        send_status_semantics = SOCKETCAN_TX_ENQUEUE_SEMANTICS
        motor_ack_observable = False
    else:
        tx_counts = {
            "arm_tx_expected_frame_count": 0,
            "arm_tx_full_frame_write_count": 0,
            "arm_tx_failed_frame_write_count": 0,
            "gripper_tx_expected_frame_count": 0,
            "gripper_tx_full_frame_write_count": 0,
            "gripper_tx_failed_frame_write_count": 0,
        }
        tx_success = False
        send_status_semantics = "unobservable_void_sdk_api"
        motor_ack_observable = False

    if record_schema == EXACT_CONTROL_RECORD_SCHEMA_V4:
        feedback_snapshot_ns = _positive_i64(
            record["feedback_snapshot_monotonic_ns"],
            label="feedback_snapshot_monotonic_ns",
        )
        if feedback_snapshot_ns != observation_ns:
            raise RobotTelemetryError(
                "v4 feedback snapshot must be atomically bound to observation time"
            )
        arm_rx_motor_ids = _u64_vector(
            record["follower_arm_state_rx_motor_ids"],
            label="follower_arm_state_rx_motor_ids",
            positive=True,
        )
        expected_arm_rx = contract.expected_arm_rx_motor_ids
        if expected_arm_rx:
            # P0-5: exact activated mapping in joint order -- rejects permutation/mis-map.
            if tuple(arm_rx_motor_ids) != tuple(expected_arm_rx):
                raise RobotTelemetryError(
                    "v4 arm STATE RX motor IDs must exactly match the activated "
                    f"mapping {tuple(expected_arm_rx)} (got {tuple(arm_rx_motor_ids)})"
                )
        elif len(set(arm_rx_motor_ids)) != 7 or any(
            motor_id >= 2**32 for motor_id in arm_rx_motor_ids
        ):
            raise RobotTelemetryError(
                "v4 arm STATE RX motor IDs must be seven unique uint32 values"
            )
        arm_rx_counters = _u64_vector(
            record["follower_arm_state_rx_counters"],
            label="follower_arm_state_rx_counters",
            positive=True,
        )
        arm_rx_generations = _u64_vector(
            record["follower_arm_state_rx_generations"],
            label="follower_arm_state_rx_generations",
            positive=True,
        )
        if arm_rx_generations != arm_rx_counters:
            raise RobotTelemetryError(
                "v4 STATE RX generation must equal its decode counter"
            )
        arm_rx_timestamps = tuple(
            _positive_i64(item, label=f"follower_arm_state_rx_monotonic_ns[{index}]")
            for index, item in enumerate(
                record["follower_arm_state_rx_monotonic_ns"]
                if isinstance(record["follower_arm_state_rx_monotonic_ns"], list)
                and len(record["follower_arm_state_rx_monotonic_ns"]) == 7
                else ()
            )
        )
        if len(arm_rx_timestamps) != 7:
            raise RobotTelemetryError(
                "follower_arm_state_rx_monotonic_ns must contain exactly 7 timestamps"
            )
        arm_rx_last_gaps = _nonnegative_i64_vector(
            record["follower_arm_state_rx_last_gap_ns"],
            label="follower_arm_state_rx_last_gap_ns",
        )
        arm_rx_gap_violations = _u64_vector(
            record["follower_arm_state_rx_gap_violation_counts"],
            label="follower_arm_state_rx_gap_violation_counts",
            positive=False,
        )
        gripper_rx_motor_id = _u64(
            record["follower_gripper_state_rx_motor_id"],
            label="follower_gripper_state_rx_motor_id",
        )
        gripper_rx_counter = _u64(
            record["follower_gripper_state_rx_counter"],
            label="follower_gripper_state_rx_counter",
        )
        gripper_rx_generation = _u64(
            record["follower_gripper_state_rx_generation"],
            label="follower_gripper_state_rx_generation",
        )
        gripper_rx_timestamp = _positive_i64(
            record["follower_gripper_state_rx_monotonic_ns"],
            label="follower_gripper_state_rx_monotonic_ns",
        )
        gripper_rx_last_gap = record["follower_gripper_state_rx_last_gap_ns"]
        gripper_rx_gap_violation_count = _u64(
            record["follower_gripper_state_rx_gap_violation_count"],
            label="follower_gripper_state_rx_gap_violation_count",
        )
        if (
            gripper_rx_motor_id == 0
            or gripper_rx_motor_id >= 2**32
            or gripper_rx_motor_id in arm_rx_motor_ids
            or gripper_rx_counter == 0
            or gripper_rx_generation != gripper_rx_counter
            or not _is_int(gripper_rx_last_gap)
            or not 0 <= gripper_rx_last_gap < 2**63
        ):
            raise RobotTelemetryError("v4 gripper STATE RX identity is invalid")
        if (
            contract.expected_gripper_rx_motor_id
            and gripper_rx_motor_id != contract.expected_gripper_rx_motor_id
        ):
            # P0-5: the gripper STATE RX ID must match the activated mapping exactly.
            raise RobotTelemetryError(
                "v4 gripper STATE RX motor ID must match the activated gripper mapping "
                f"{contract.expected_gripper_rx_motor_id} (got {gripper_rx_motor_id})"
            )
        all_timestamps = arm_rx_timestamps + (gripper_rx_timestamp,)
        if any(
            timestamp > feedback_snapshot_ns
            for timestamp in all_timestamps
        ):
            raise RobotTelemetryError("v4 STATE RX feedback is from the future")
        feedback_liveness_observable = True
    else:
        feedback_snapshot_ns = 0
        arm_rx_motor_ids = ()
        arm_rx_counters = ()
        arm_rx_generations = ()
        arm_rx_timestamps = ()
        arm_rx_last_gaps = ()
        arm_rx_gap_violations = ()
        gripper_rx_motor_id = 0
        gripper_rx_counter = 0
        gripper_rx_generation = 0
        gripper_rx_timestamp = 0
        gripper_rx_last_gap = 0
        gripper_rx_gap_violation_count = 0
        feedback_liveness_observable = False

    follower_q = _numeric_vector(
        record["follower_q_rad"], label="follower_q_rad"
    )
    follower_qd = _numeric_vector(
        record["follower_qd_rad_s"], label="follower_qd_rad_s"
    )
    gripper_q = _finite_number(
        record["follower_gripper_q_rad"],
        label="follower gripper joint position",
    )
    gripper_qd = _finite_number(
        record["follower_gripper_qd_rad_s"],
        label="follower gripper joint velocity",
    )
    gripper_motor_q = _finite_number(
        record["follower_gripper_motor_q_rad"],
        label="follower gripper motor position",
    )
    gripper_motor_qd = _finite_number(
        record["follower_gripper_motor_qd_rad_s"],
        label="follower gripper motor velocity",
    )
    action = ExactFollowerAction(
        arm_kp=_numeric_vector(
            record["follower_arm_mit_kp"], label="follower_arm_mit_kp"
        ),
        arm_kd=_numeric_vector(
            record["follower_arm_mit_kd"], label="follower_arm_mit_kd"
        ),
        arm_q=_numeric_vector(
            record["follower_arm_mit_q"], label="follower_arm_mit_q"
        ),
        arm_dq=_numeric_vector(
            record["follower_arm_mit_dq"], label="follower_arm_mit_dq"
        ),
        arm_tau=_numeric_vector(
            record["follower_arm_mit_tau"], label="follower_arm_mit_tau"
        ),
        gripper_kp=_finite_number(
            record["follower_gripper_mit_kp"],
            label="follower_gripper_mit_kp",
        ),
        gripper_kd=_finite_number(
            record["follower_gripper_mit_kd"],
            label="follower_gripper_mit_kd",
        ),
        gripper_q=_finite_number(
            record["follower_gripper_mit_q"], label="follower_gripper_mit_q"
        ),
        gripper_dq=_finite_number(
            record["follower_gripper_mit_dq"], label="follower_gripper_mit_dq"
        ),
        gripper_tau=_finite_number(
            record["follower_gripper_mit_tau"],
            label="follower_gripper_mit_tau",
        ),
    )

    for index, (position, limits) in enumerate(
        zip(follower_q, contract.joint_position_limits_rad)
    ):
        lower, upper = limits
        if not float(lower) <= position <= float(upper):
            raise RobotTelemetryError(
                f"follower joint {index + 1} is outside activated joint limits"
            )
    for index, (position, limits) in enumerate(
        zip(action.arm_q, contract.joint_position_limits_rad)
    ):
        lower, upper = limits
        if not float(lower) <= position <= float(upper):
            raise RobotTelemetryError(
                f"MIT action joint {index + 1} is outside activated joint limits"
            )
    if any(
        abs(value) > float(contract.max_abs_joint_velocity_rad_s)
        for value in follower_qd
    ):
        raise RobotTelemetryError(
            "follower joint velocity exceeds the activated velocity limit"
        )
    if any(
        abs(value) > float(contract.max_abs_joint_velocity_rad_s)
        for value in action.arm_dq
    ):
        raise RobotTelemetryError(
            "MIT action joint velocity exceeds the activated velocity limit"
        )
    if any(value < 0.0 for value in action.arm_kp):
        raise RobotTelemetryError("MIT action stiffness must be non-negative")
    if any(value < 0.0 for value in action.arm_kd):
        raise RobotTelemetryError("MIT action damping must be non-negative")
    if action.gripper_kp < 0.0:
        raise RobotTelemetryError(
            "gripper MIT stiffness must be non-negative"
        )
    if action.gripper_kd < 0.0:
        raise RobotTelemetryError(
            "gripper MIT damping must be non-negative"
        )
    gripper_lower, gripper_upper = (
        float(value) for value in contract.gripper_motor_position_limits_rad
    )
    for label, value in (
        ("follower", gripper_motor_q),
        ("follower MIT", action.gripper_q),
    ):
        if not gripper_lower <= value <= gripper_upper:
            raise RobotTelemetryError(
                f"{label} gripper motor position is outside activated limits: "
                f"{value:.4f} rad not in [{gripper_lower:.4f}, {gripper_upper:.4f}]"
            )

    return ExactControlRecord(
        schema=record_schema,
        control_tick_seq=control_tick_seq,
        observation_monotonic_ns=observation_ns,
        reference_generation=reference_generation,
        reference_source_monotonic_ns=reference_source_ns,
        reference_snapshot_valid=bool(record["reference_snapshot_valid"]),
        control_call_begin_monotonic_ns=arm_begin_ns,
        arm_call_return_monotonic_ns=arm_return_ns,
        gripper_call_begin_monotonic_ns=gripper_begin_ns,
        gripper_call_return_monotonic_ns=gripper_return_ns,
        control_call_end_monotonic_ns=control_end_ns,
        follower_q_rad=follower_q,
        follower_qd_rad_s=follower_qd,
        follower_gripper_q_rad=gripper_q,
        follower_gripper_qd_rad_s=gripper_qd,
        follower_gripper_motor_q_rad=gripper_motor_q,
        follower_gripper_motor_qd_rad_s=gripper_motor_qd,
        action=action,
        envelope_seq=envelope["envelope_seq"],
        publisher_monotonic_ns=envelope["publisher_monotonic_ns"],
        controller_pid=envelope["controller_pid"],
        controller_start_token=envelope["controller_start_token"],
        side=contract.side,
        controller_mode="unilateral",
        leader_can_interface=contract.leader_can_interface,
        follower_can_interface=contract.follower_can_interface,
        joint_names=contract.joint_names,
        urdf_sha256=contract.urdf_sha256,
        joint_mapper_sha256=contract.joint_mapper_sha256,
        pair_contract_sha256=contract.pair_contract_sha256,
        controller_binary_sha256=contract.controller_binary_sha256,
        controller_source_manifest_sha256=(
            contract.controller_source_manifest_sha256
        ),
        follower_config_sha256=contract.follower_config_sha256,
        openarm_sdk_sha256=contract.openarm_sdk_sha256,
        send_status_observable=(
            envelope_schema
            in {CONTROL_ENVELOPE_SCHEMA_V3, CONTROL_ENVELOPE_SCHEMA_V4}
        ),
        send_status_reason=(
            SOCKETCAN_TX_STATUS_REASON
            if envelope_schema
            in {CONTROL_ENVELOPE_SCHEMA_V3, CONTROL_ENVELOPE_SCHEMA_V4}
            else VOID_SEND_STATUS_REASON
        ),
        send_status_semantics=send_status_semantics,
        motor_ack_observable=motor_ack_observable,
        arm_tx_expected_frame_count=tx_counts[
            "arm_tx_expected_frame_count"
        ],
        arm_tx_full_frame_write_count=tx_counts[
            "arm_tx_full_frame_write_count"
        ],
        arm_tx_failed_frame_write_count=tx_counts[
            "arm_tx_failed_frame_write_count"
        ],
        gripper_tx_expected_frame_count=tx_counts[
            "gripper_tx_expected_frame_count"
        ],
        gripper_tx_full_frame_write_count=tx_counts[
            "gripper_tx_full_frame_write_count"
        ],
        gripper_tx_failed_frame_write_count=tx_counts[
            "gripper_tx_failed_frame_write_count"
        ],
        socketcan_tx_enqueue_success=tx_success,
        feedback_liveness_observable=feedback_liveness_observable,
        feedback_snapshot_monotonic_ns=feedback_snapshot_ns,
        follower_arm_state_rx_motor_ids=arm_rx_motor_ids,
        follower_arm_state_rx_counters=arm_rx_counters,
        follower_arm_state_rx_generations=arm_rx_generations,
        follower_arm_state_rx_monotonic_ns=arm_rx_timestamps,
        follower_arm_state_rx_last_gap_ns=arm_rx_last_gaps,
        follower_arm_state_rx_gap_violation_counts=(
            arm_rx_gap_violations
        ),
        follower_gripper_state_rx_motor_id=gripper_rx_motor_id,
        follower_gripper_state_rx_counter=gripper_rx_counter,
        follower_gripper_state_rx_generation=gripper_rx_generation,
        follower_gripper_state_rx_monotonic_ns=gripper_rx_timestamp,
        follower_gripper_state_rx_last_gap_ns=gripper_rx_last_gap,
        follower_gripper_state_rx_gap_violation_count=(
            gripper_rx_gap_violation_count
        ),
    )


def _validate_v4_feedback_transition(
    previous: ExactControlRecord,
    current: ExactControlRecord,
) -> None:
    if (
        previous.schema != EXACT_CONTROL_RECORD_SCHEMA_V4
        or current.schema != EXACT_CONTROL_RECORD_SCHEMA_V4
        or current.follower_arm_state_rx_motor_ids
        != previous.follower_arm_state_rx_motor_ids
        or current.follower_gripper_state_rx_motor_id
        != previous.follower_gripper_state_rx_motor_id
    ):
        raise RobotTelemetryError("v4 STATE RX motor identity changed")
    for label, counters, prior_counters, timestamps, prior_timestamps, violations, prior_violations in (
        (
            "arm",
            current.follower_arm_state_rx_counters,
            previous.follower_arm_state_rx_counters,
            current.follower_arm_state_rx_monotonic_ns,
            previous.follower_arm_state_rx_monotonic_ns,
            current.follower_arm_state_rx_gap_violation_counts,
            previous.follower_arm_state_rx_gap_violation_counts,
        ),
        (
            "gripper",
            (current.follower_gripper_state_rx_counter,),
            (previous.follower_gripper_state_rx_counter,),
            (current.follower_gripper_state_rx_monotonic_ns,),
            (previous.follower_gripper_state_rx_monotonic_ns,),
            (current.follower_gripper_state_rx_gap_violation_count,),
            (previous.follower_gripper_state_rx_gap_violation_count,),
        ),
    ):
        for counter, prior_counter, timestamp, prior_timestamp, violation, prior_violation in zip(
            counters,
            prior_counters,
            timestamps,
            prior_timestamps,
            violations,
            prior_violations,
        ):
            if counter < prior_counter or timestamp < prior_timestamp:
                raise RobotTelemetryError(
                    f"v4 {label} STATE RX counter or timestamp rolled back"
                )
            if (counter == prior_counter) != (timestamp == prior_timestamp):
                raise RobotTelemetryError(
                    f"v4 {label} STATE RX counter/timestamp progression disagrees"
                )
            if violation < prior_violation:
                raise RobotTelemetryError(
                    f"v4 {label} STATE RX gap violation counter rolled back"
                )


def parse_control_envelope_datagram(
    payload: bytes, contract: RobotStateContract
) -> RobotControlEnvelope:
    """Parse one v2/v3/v4 envelope and return every exact 1 kHz control record."""

    if not isinstance(contract, RobotStateContract):
        raise TypeError("contract must be RobotStateContract")
    for digest, label in (
        (
            contract.controller_source_manifest_sha256,
            "controller source manifest",
        ),
        (contract.follower_config_sha256, "follower controller config"),
        (contract.openarm_sdk_sha256, "OpenArm SDK"),
    ):
        if _SHA256_RE.fullmatch(digest) is None:
            raise RobotTelemetryError(
                f"exact-control contract requires activated {label} SHA-256"
            )
    if not isinstance(payload, bytes):
        raise RobotTelemetryError("control envelope datagram must be bytes")
    if not payload or len(payload) > MAX_CONTROL_ENVELOPE_BYTES:
        raise RobotTelemetryError("control envelope datagram has invalid size")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except RobotTelemetryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RobotTelemetryError("invalid control envelope JSON") from exc
    if not isinstance(value, dict):
        raise RobotTelemetryError("control envelope must be an object")
    envelope_schema = value.get("schema")
    if envelope_schema == CONTROL_ENVELOPE_SCHEMA:
        expected_envelope_fields = _CONTROL_ENVELOPE_FIELDS
    elif envelope_schema == CONTROL_ENVELOPE_SCHEMA_V3:
        expected_envelope_fields = _CONTROL_ENVELOPE_FIELDS_V3
    elif envelope_schema == CONTROL_ENVELOPE_SCHEMA_V4:
        expected_envelope_fields = _CONTROL_ENVELOPE_FIELDS_V4
    else:
        raise RobotTelemetryError("control envelope schema is unsupported")
    envelope = _require_exact_fields(
        value, expected_envelope_fields, label="control envelope"
    )
    if envelope["clock_id"] != CLOCK_ID:
        raise RobotTelemetryError("control envelope clock must be CLOCK_MONOTONIC")
    envelope_seq = _u64(envelope["envelope_seq"], label="envelope_seq")
    publisher_ns = _positive_i64(
        envelope["publisher_monotonic_ns"], label="publisher_monotonic_ns"
    )
    pid = envelope["controller_pid"]
    if not _is_int(pid) or not 0 < pid < 2**31:
        raise RobotTelemetryError("controller_pid must be a positive process ID")
    start_token = envelope["controller_start_token"]
    if (
        not isinstance(start_token, str)
        or _START_TOKEN_RE.fullmatch(start_token) is None
    ):
        raise RobotTelemetryError("controller start token is invalid")
    for field, expected, label in (
        ("side", contract.side, "robot side"),
        ("controller_mode", "unilateral", "controller mode"),
        (
            "leader_can_interface",
            contract.leader_can_interface,
            "leader CAN interface",
        ),
        (
            "follower_can_interface",
            contract.follower_can_interface,
            "follower CAN interface",
        ),
    ):
        if envelope[field] != expected:
            raise RobotTelemetryError(f"{label} does not match activated contract")
    if (
        not isinstance(envelope["joint_names"], list)
        or tuple(envelope["joint_names"]) != contract.joint_names
    ):
        raise RobotTelemetryError("joint names do not match activated contract")
    for field, expected, label in (
        ("urdf_sha256", contract.urdf_sha256, "URDF"),
        ("joint_mapper_sha256", contract.joint_mapper_sha256, "joint mapper"),
        ("pair_contract_sha256", contract.pair_contract_sha256, "pair contract"),
        (
            "controller_binary_sha256",
            contract.controller_binary_sha256,
            "controller binary",
        ),
        (
            "controller_source_manifest_sha256",
            contract.controller_source_manifest_sha256,
            "controller source manifest",
        ),
        (
            "follower_config_sha256",
            contract.follower_config_sha256,
            "follower controller config",
        ),
        ("openarm_sdk_sha256", contract.openarm_sdk_sha256, "OpenArm SDK"),
    ):
        digest = _sha256(envelope[field], label=label)
        if digest != expected:
            raise RobotTelemetryError(
                f"{label} SHA-256 does not match activated contract"
            )
    if envelope["publisher_nominal_hz"] != 100:
        raise RobotTelemetryError("publisher nominal frequency must be 100 Hz")
    if envelope["control_nominal_hz"] != 1000:
        raise RobotTelemetryError("control nominal frequency must be 1000 Hz")
    if envelope_schema == CONTROL_ENVELOPE_SCHEMA:
        if envelope["send_status_observable"] is not False:
            raise RobotTelemetryError(
                "send status must not be represented as observable for the void SDK API"
            )
        if envelope["send_status_reason"] != VOID_SEND_STATUS_REASON:
            raise RobotTelemetryError(
                "send status reason does not match audited SDK API"
            )
    else:
        if envelope["send_status_observable"] is not True:
            raise RobotTelemetryError(
                "v3 SocketCAN write status must be observable"
            )
        if envelope["send_status_reason"] != SOCKETCAN_TX_STATUS_REASON:
            raise RobotTelemetryError(
                "v3 send status reason does not match patched SDK API"
            )
        if envelope["send_status_semantics"] != SOCKETCAN_TX_ENQUEUE_SEMANTICS:
            raise RobotTelemetryError(
                "v3 send status semantics must describe SocketCAN TX enqueue only"
            )
        if envelope["motor_ack_observable"] is not False:
            raise RobotTelemetryError(
                "motor acknowledgement must remain explicitly unobservable"
            )
        if envelope["motor_ack_reason"] != MOTOR_ACK_UNOBSERVABLE_REASON:
            raise RobotTelemetryError(
                "motor acknowledgement reason does not match v3 semantics"
            )
        if envelope_schema == CONTROL_ENVELOPE_SCHEMA_V4:
            if envelope["feedback_liveness_observable"] is not True:
                raise RobotTelemetryError(
                    "v4 per-motor feedback liveness must be observable"
                )
            if envelope["feedback_liveness_reason"] != FEEDBACK_LIVENESS_REASON:
                raise RobotTelemetryError(
                    "v4 feedback liveness reason does not match the SDK hook"
                )
            if (
                envelope["feedback_liveness_semantics"]
                != FEEDBACK_LIVENESS_SEMANTICS
            ):
                raise RobotTelemetryError(
                    "v4 feedback liveness semantics do not match valid decode accounting"
                )
            if envelope["feedback_max_age_ns"] != FEEDBACK_MAX_AGE_NS:
                raise RobotTelemetryError("v4 feedback maximum age must be 20 ms")
            if envelope["feedback_max_gap_ns"] != FEEDBACK_MAX_GAP_NS:
                raise RobotTelemetryError("v4 feedback maximum gap must be 20 ms")
    for field in (
        "ring_overflow_count",
        "publisher_rejected_record_count",
    ):
        if not _is_int(envelope[field]) or envelope[field] != 0:
            raise RobotTelemetryError(f"{field} must be zero")
    # socket_send_failure_count: a nonblocking sendto that fails drops THAT envelope,
    # which the receiver-side envelope-sequence contiguity check already latches on —
    # so this counter adds no integrity the seq check doesn't. Requiring it to be
    # zero forever only criminalizes harmless pre-receiver-start drops (on a shared
    # box the receiver thread can lag controller start by seconds and the socket
    # queue overflows before draining begins). Accept any non-negative count; real
    # mid-stream loss still faults via "control envelope sequence loss".
    if not _is_int(envelope["socket_send_failure_count"]) or (
        envelope["socket_send_failure_count"] < 0
    ):
        raise RobotTelemetryError(
            "socket_send_failure_count must be a non-negative integer"
        )
    records_value = envelope["records"]
    if (
        not isinstance(records_value, list)
        or not 1 <= len(records_value) <= 16
    ):
        raise RobotTelemetryError(
            "control envelope must contain between 1 and 16 records"
        )
    parsed_context = {
        "schema": envelope_schema,
        "envelope_seq": envelope_seq,
        "publisher_monotonic_ns": publisher_ns,
        "controller_pid": pid,
        "controller_start_token": start_token,
    }
    records = tuple(
        _parse_exact_control_record(
            record, contract=contract, envelope=parsed_context
        )
        for record in records_value
    )
    for left, right in zip(records, records[1:]):
        if right.control_tick_seq != left.control_tick_seq + 1:
            raise RobotTelemetryError(
                "exact control records are not sequence-contiguous"
            )
        if right.observation_monotonic_ns <= left.observation_monotonic_ns:
            raise RobotTelemetryError(
                "exact control observation clock is not strictly monotonic"
            )
        if envelope_schema == CONTROL_ENVELOPE_SCHEMA_V4:
            _validate_v4_feedback_transition(left, right)
    return RobotControlEnvelope(
        schema=envelope_schema,
        envelope_seq=envelope_seq,
        publisher_monotonic_ns=publisher_ns,
        controller_pid=pid,
        controller_start_token=start_token,
        records=records,
        socket_send_failure_count=int(envelope["socket_send_failure_count"]),
    )


def validate_send_failure_transition(previous_count, current_count):
    """Fault on any mid-stream CHANGE of the cumulative sendto-failure counter.

    A constant nonzero baseline is legal (envelopes dropped before the receiver
    started draining never reached us and are invisible to sequence checks); an
    INCREASE across received envelopes means an envelope was lost mid-stream, and
    a DECREASE means the publisher state is inconsistent. Both fault.
    """
    if current_count > previous_count:
        raise RobotTelemetryError(
            "socket_send_failure_count increased mid-stream "
            f"({previous_count} -> {current_count}): an envelope was lost"
        )
    if current_count < previous_count:
        raise RobotTelemetryError(
            "socket_send_failure_count rolled back "
            f"({previous_count} -> {current_count})"
        )


class ExactControlStateStream:
    """Two-second v2/v3/v4 stream retaining exact actions and observations.

    ``telemetry_schema`` intentionally remains v2 so the bare stream is an
    explicit commissioning-compatible sink.  Production capture is exposed by
    ``RobotCaptureCoordinator``, whose preferred schema is v3.
    """

    # Structural marker used by transport/fanout sinks.  Production receivers
    # must never silently accept the diagnostic v1 RobotStateStream.
    telemetry_schema = CONTROL_ENVELOPE_SCHEMA

    def __init__(self, contract: RobotStateContract, *,
                 absorb_bounded_holes: bool = False):
        if not isinstance(contract, RobotStateContract):
            raise TypeError("contract must be RobotStateContract")
        self.contract = contract
        # Bounded publisher-hole absorption is OPT-IN and belongs ONLY to the
        # live preflight/monitoring stream (arm stays usable through a missed
        # 10 ms publisher slot). Session-capture streams keep the default
        # STRICT contract: a hole during an active recording faults the
        # session immediately (fail fast) so finalized raw telemetry remains
        # gapless — the finalize validators depend on that invariant.
        self._absorb_bounded_holes = bool(absorb_bounded_holes)
        self._last_record_count = 0
        self._last_publisher_monotonic_ns = 0
        self._records: List[ExactControlRecord] = []
        self._last_envelope_seq: int | None = None
        self._last_send_failure_count = 0
        self._last_control_tick_seq: int | None = None
        self._last_monotonic_ns: int | None = None
        self._process_identity: Tuple[int, str] | None = None
        self._envelope_schema: str | None = None
        self._fault_reason = ""

    @property
    def faulted(self) -> bool:
        return bool(self._fault_reason)

    @property
    def fault_reason(self) -> str:
        return self._fault_reason

    @property
    def record_count(self) -> int:
        return len(self._records)

    def _raise_if_faulted(self) -> None:
        if self.faulted:
            raise RobotTelemetryError(
                f"exact control stream is faulted: {self._fault_reason}"
            )

    def _latch_fault(self, reason: str) -> None:
        if not self._fault_reason:
            self._fault_reason = reason
        raise RobotTelemetryError(reason)

    def ingest_datagram(self, payload: bytes) -> RobotControlEnvelope:
        self._raise_if_faulted()
        try:
            envelope = parse_control_envelope_datagram(payload, self.contract)
        except RobotTelemetryError as exc:
            self._latch_fault(str(exc))
        if self._last_envelope_seq is not None:
            if envelope.schema != self._envelope_schema:
                self._latch_fault("exact-control envelope schema changed")
            if envelope.process_identity != self._process_identity:
                self._latch_fault("control publisher process identity changed")
            seq_jump = envelope.envelope_seq - self._last_envelope_seq
            tick_delta = (
                envelope.records[0].control_tick_seq - self._last_control_tick_seq
            )
            records_per_envelope = self._last_record_count
            missed = seq_jump - 1
            # Bounded, exactly tick-aligned publisher hole (see
            # _ENVELOPE_HOLE_TOLERANCE): the shared box's low-priority publisher
            # can miss an occasional 10 ms slot while the 1 kHz tick counter
            # keeps running. Absorb-eligible ONLY when this stream opted in
            # (live preflight; never session capture) and the schema is a
            # production one — the SEALED controller publishes v4 (verified in
            # the deployed ELF 2026-08-07; a v2-only gate made this dead code
            # live). Tolerated only when the tick counter corroborates the hole
            # exactly and the observation clock gap stays within the aligned
            # bound; everything else still latches.
            observation_delta_ns = (
                envelope.records[0].observation_monotonic_ns
                - self._last_monotonic_ns
            )
            publisher_delta_ns = (
                envelope.publisher_monotonic_ns
                - self._last_publisher_monotonic_ns
            )
            hole_ok = (
                self._absorb_bounded_holes
                and envelope.schema in (
                    CONTROL_ENVELOPE_SCHEMA, CONTROL_ENVELOPE_SCHEMA_V4,
                )
                and missed >= 1
                and tick_delta >= 1
                and 0 < observation_delta_ns <= _MAX_HOLE_NS
                and 0 < publisher_delta_ns <= _MAX_HOLE_NS
            )
            if seq_jump != 1 and not hole_ok:
                self._latch_fault(
                    "control envelope sequence loss or duplication"
                )
            if hole_ok:
                # Inside an absorbed hole the publisher-side counters (failed
                # sendto calls, ring overwrites) are the SAME already-accounted
                # event seen from the other side — no transition validation;
                # the time bounds above already cap the trust gap.
                pass
            else:
                try:
                    validate_send_failure_transition(
                        self._last_send_failure_count,
                        envelope.socket_send_failure_count,
                    )
                except RobotTelemetryError as exc:
                    self._latch_fault(str(exc))
            if seq_jump == 1 and tick_delta != 1:
                self._latch_fault("exact control tick sequence loss or duplication")
            if (
                envelope.records[0].observation_monotonic_ns
                <= self._last_monotonic_ns
            ):
                self._latch_fault("exact control CLOCK_MONOTONIC clock rollback")
            if envelope.schema == CONTROL_ENVELOPE_SCHEMA_V4 and seq_jump == 1:
                # the v4 feedback transition compares CONSECUTIVE records and is
                # undefined across an absorbed hole
                try:
                    _validate_v4_feedback_transition(
                        self._records[-1], envelope.records[0]
                    )
                except RobotTelemetryError as exc:
                    self._latch_fault(str(exc))
        else:
            self._process_identity = envelope.process_identity
            self._envelope_schema = envelope.schema

        self._records.extend(envelope.records)
        self._last_record_count = len(envelope.records)
        self._last_publisher_monotonic_ns = envelope.publisher_monotonic_ns
        self._last_envelope_seq = envelope.envelope_seq
        self._last_send_failure_count = envelope.socket_send_failure_count
        self._last_control_tick_seq = envelope.records[-1].control_tick_seq
        self._last_monotonic_ns = envelope.records[-1].observation_monotonic_ns
        cutoff_ns = self._last_monotonic_ns - self.contract.buffer_duration_ns
        while (
            self._records
            and self._records[0].observation_monotonic_ns < cutoff_ns
        ):
            self._records.pop(0)
        while len(self._records) > 4096:
            self._records.pop(0)
        return envelope

    def query_exposure_interval(
        self,
        *,
        exposure_lower_ns: int,
        exposure_nominal_ns: int,
        exposure_upper_ns: int,
        now_monotonic_ns: int,
        camera_profile_sha256: str,
        timing_artifact_sha256: str,
    ) -> BracketedExactControlTrajectory:
        self._raise_if_faulted()
        for value, label in (
            (exposure_lower_ns, "exposure lower"),
            (exposure_nominal_ns, "exposure nominal"),
            (exposure_upper_ns, "exposure upper"),
            (now_monotonic_ns, "query clock"),
        ):
            if not _is_int(value) or value <= 0:
                raise RobotTelemetryError(f"{label} time must be positive integer ns")
        if not exposure_lower_ns <= exposure_nominal_ns <= exposure_upper_ns:
            raise RobotTelemetryError("exposure interval ordering is invalid")
        if (
            exposure_upper_ns - exposure_lower_ns
            > self.contract.max_exposure_interval_ns
        ):
            raise RobotTelemetryError("exposure interval is too wide")
        if camera_profile_sha256 != self.contract.camera_profile_sha256:
            raise RobotTelemetryError("camera profile does not match activated contract")
        if timing_artifact_sha256 != self.contract.timing_artifact_sha256:
            raise RobotTelemetryError("timing artifact does not match activated contract")
        if not self._records:
            raise RobotTelemetryError("exact control buffer is empty")
        latest_ns = self._records[-1].observation_monotonic_ns
        if now_monotonic_ns < latest_ns:
            raise RobotTelemetryError("query clock precedes exact control telemetry")
        if now_monotonic_ns - latest_ns > self.contract.max_staleness_ns:
            raise RobotTelemetryError("exact control telemetry is stale")
        timestamps = tuple(
            record.observation_monotonic_ns for record in self._records
        )
        before_index = bisect_right(timestamps, exposure_lower_ns) - 1
        after_index = bisect_left(timestamps, exposure_upper_ns)
        if before_index < 0 or after_index >= len(self._records):
            raise RobotTelemetryError(
                "exact control telemetry does not bracket complete exposure"
            )
        records = tuple(self._records[before_index : after_index + 1])
        gaps = tuple(
            right.observation_monotonic_ns - left.observation_monotonic_ns
            for left, right in zip(records, records[1:])
        )
        if any(gap <= 0 or gap > MAX_CONTROL_TICK_GAP_NS for gap in gaps):
            raise RobotTelemetryError(
                "exact control gap exceeds the hard tick-gap bound"
            )
        return BracketedExactControlTrajectory(
            exposure_lower_ns=exposure_lower_ns,
            exposure_nominal_ns=exposure_nominal_ns,
            exposure_upper_ns=exposure_upper_ns,
            records=records,
        )


__all__ = [
    "CLOCK_ID",
    "CONTROL_ENVELOPE_SCHEMA",
    "CONTROL_ENVELOPE_SCHEMA_V3",
    "CONTROL_ENVELOPE_SCHEMA_V4",
    "EXACT_CONTROL_RECORD_SCHEMA",
    "EXACT_CONTROL_RECORD_SCHEMA_V3",
    "EXACT_CONTROL_RECORD_SCHEMA_V4",
    "FEEDBACK_LIVENESS_REASON",
    "FEEDBACK_LIVENESS_SEMANTICS",
    "FEEDBACK_MAX_AGE_NS",
    "FEEDBACK_MAX_GAP_NS",
    "MAX_CONTROL_ENVELOPE_BYTES",
    "MAX_DATAGRAM_BYTES",
    "ROBOT_STATE_SCHEMA",
    "MOTOR_ACK_UNOBSERVABLE_REASON",
    "SOCKETCAN_TX_ENQUEUE_SEMANTICS",
    "SOCKETCAN_TX_STATUS_REASON",
    "VOID_SEND_STATUS_REASON",
    "BracketedRobotTrajectory",
    "BracketedExactControlTrajectory",
    "ExactControlRecord",
    "ExactControlStateStream",
    "ExactFollowerAction",
    "InterpolatedFollowerObservation",
    "InterpolatedRobotState",
    "RobotControlEnvelope",
    "RobotStateContract",
    "RobotStatePacket",
    "RobotStateStream",
    "RobotTelemetryError",
    "default_openarm_can_pair",
    "parse_control_envelope_datagram",
    "parse_robot_state_datagram",
]
