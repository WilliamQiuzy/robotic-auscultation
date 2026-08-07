"""Durable, fail-closed OpenArm robot capture for one sensor session.

The sole controller process remains the CAN owner.  This component consumes its
strict Unix datagrams, writes the exact bytes once, derives compatibility/audit
CSVs from the same validated packet, and binds each camera exposure to the
complete robot-state bracket used by perception.

It deliberately does not decide whether a session is training eligible.  That
decision belongs to the combined session finalizer after every hardware stream
has independently passed its own coverage contract.
"""

from __future__ import annotations

import csv
from collections import deque
from contextlib import ExitStack
import hashlib
import json
import math
import os
import re
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, IO, List, Mapping, Optional, Sequence, Tuple

from .robot_state_stream import (
    MAX_CONTROL_TICK_GAP_NS,
    BracketedExactControlTrajectory,
    BracketedRobotTrajectory,
    ExactControlRecord,
    ExactControlStateStream,
    InterpolatedFollowerObservation,
    RobotControlEnvelope,
    RobotStateContract,
    RobotStatePacket,
    RobotStateStream,
    RobotTelemetryError,
    CONTROL_ENVELOPE_SCHEMA_V3,
    CONTROL_ENVELOPE_SCHEMA_V4,
    EXACT_CONTROL_RECORD_SCHEMA_V3,
    EXACT_CONTROL_RECORD_SCHEMA_V4,
    FEEDBACK_LIVENESS_REASON,
    FEEDBACK_LIVENESS_SEMANTICS,
    FEEDBACK_MAX_AGE_NS,
    FEEDBACK_MAX_GAP_NS,
    MOTOR_ACK_UNOBSERVABLE_REASON,
    SOCKETCAN_TX_ENQUEUE_SEMANTICS,
    SOCKETCAN_TX_STATUS_REASON,
    VOID_SEND_STATUS_REASON,
    parse_robot_state_datagram,
)


DIAGNOSTIC_V1_CAPTURE_SUMMARY_SCHEMA = "wam_openarm_robot_capture_summary_v1"
DIAGNOSTIC_V1_FRAME_INDEX_SCHEMA = "wam_openarm_robot_frame_evidence_v1"

RAW_PARTIAL_NAME = "robot_telemetry.jsonl.partial"
COMPATIBLE_PARTIAL_NAME = "robot.csv.partial"
AUDIT_PARTIAL_NAME = "robot_leader_audit.csv.partial"
FRAME_INDEX_PARTIAL_NAME = "robot_frame_index.jsonl.partial"
SUMMARY_PARTIAL_NAME = "robot_capture_summary.json.partial"

_PARTIAL_TO_FINAL = (
    (RAW_PARTIAL_NAME, "robot_telemetry.jsonl"),
    (COMPATIBLE_PARTIAL_NAME, "robot.csv"),
    (AUDIT_PARTIAL_NAME, "robot_leader_audit.csv"),
    (FRAME_INDEX_PARTIAL_NAME, "robot_frame_index.jsonl"),
    (SUMMARY_PARTIAL_NAME, "robot_capture_summary.json"),
)

_CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
_FRAME_INDEX_FIELDS = frozenset(
    {
        "schema",
        "camera_id",
        "frame_index",
        "exposure_lower_ns",
        "exposure_nominal_ns",
        "exposure_upper_ns",
        "query_now_monotonic_ns",
        "robot_seq_first",
        "robot_seq_last",
        "robot_timestamp_first_ns",
        "robot_timestamp_last_ns",
        "robot_max_gap_ns",
        "controller_pid",
        "controller_start_token",
        "publisher_identity",
        "side",
        "controller_mode",
        "leader_can_interface",
        "follower_can_interface",
        "urdf_sha256",
        "joint_mapper_sha256",
        "camera_profile_sha256",
        "timing_artifact_sha256",
        "pair_contract_sha256",
        "controller_binary_sha256",
    }
)


class RobotCaptureError(RuntimeError):
    """Robot data cannot be admitted into a production capture."""


@dataclass(frozen=True)
class _VerifiedPartials:
    packets: Tuple[RobotStatePacket, ...]
    frame_count: int
    max_gap_ns: int
    hash_inventory: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class DiagnosticV1CapturedRobotFrameEvidence:
    """The exact trajectory used by perception plus its persisted index row."""

    trajectory: BracketedRobotTrajectory
    index: Mapping[str, object]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RobotCaptureError(f"duplicate frame-index JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise RobotCaptureError(f"non-finite JSON constant is forbidden: {value}")


def _seconds_text(monotonic_ns: int) -> str:
    return f"{monotonic_ns // 1_000_000_000}.{monotonic_ns % 1_000_000_000:09d}"


def _float_text(value: float) -> str:
    result = float(value)
    if not math.isfinite(result):
        raise RobotCaptureError("robot CSV source contains a non-finite value")
    return format(result, ".17g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _compatible_header() -> List[str]:
    return (
        ["host_t_s"]
        + [f"q_joint_{index}" for index in range(1, 8)]
        + [f"a_joint_{index}" for index in range(1, 8)]
    )


def _compatible_row(packet: RobotStatePacket) -> List[str]:
    return (
        [_seconds_text(packet.monotonic_ns)]
        + [_float_text(value) for value in packet.follower_q_rad]
        + [_float_text(value) for value in packet.follower_command_q_rad]
    )


def _audit_header(joint_names: Sequence[str]) -> List[str]:
    return (
        ["monotonic_ns", "seq"]
        + [
            f"observation.follower.{name}.motor_rad"
            for name in joint_names
        ]
        + [
            f"observation.follower.{name}.motor_rad_s"
            for name in joint_names
        ]
        + ["observation.follower.gripper.motor_rad"]
        + [
            f"audit.leader_measured.{name}.motor_rad"
            for name in joint_names
        ]
        + [
            f"audit.leader_measured.{name}.motor_rad_s"
            for name in joint_names
        ]
        + ["audit.leader_measured.gripper.motor_rad"]
        + [
            f"action.follower_command.{name}.motor_rad"
            for name in joint_names
        ]
        + ["action.follower_command.gripper.motor_rad"]
    )


def _audit_row(packet: RobotStatePacket) -> List[str]:
    return (
        [str(packet.monotonic_ns), str(packet.seq)]
        + [_float_text(value) for value in packet.follower_q_rad]
        + [_float_text(value) for value in packet.follower_qd_rad_s]
        + [_float_text(packet.follower_gripper_motor_position_rad)]
        + [_float_text(value) for value in packet.leader_q_rad]
        + [_float_text(value) for value in packet.leader_qd_rad_s]
        + [_float_text(packet.leader_gripper_motor_position_rad)]
        + [_float_text(value) for value in packet.follower_command_q_rad]
        + [_float_text(packet.follower_command_gripper_motor_position_rad)]
    )


class DiagnosticV1RobotSessionCapture:
    """Capture the left OpenArm leader/follower pair into one session directory."""

    def __init__(
        self,
        session_dir: object,
        contract: RobotStateContract,
        *,
        preflight_min_packets: int = 11,
        preflight_duration_ns: int = 100_000_000,
    ) -> None:
        if not isinstance(contract, RobotStateContract):
            raise TypeError("contract must be RobotStateContract")
        if (
            contract.side != "left"
            or contract.leader_can_interface != "can1"
            or contract.follower_can_interface != "can3"
        ):
            raise RobotCaptureError(
                "production capture requires the left OpenArm pair can1 leader -> can3 follower"
            )
        if (
            not _is_int(preflight_min_packets)
            or preflight_min_packets < 2
            or not _is_int(preflight_duration_ns)
            or preflight_duration_ns <= 0
        ):
            raise ValueError("robot preflight bounds must be positive")
        if preflight_duration_ns > contract.buffer_duration_ns:
            raise ValueError("robot preflight duration exceeds the telemetry buffer")

        path = Path(session_dir)
        if not path.is_dir():
            raise RobotCaptureError("robot session directory must already exist")

        self.session_dir = path
        self.contract = contract
        self.stream = RobotStateStream(contract)
        self.preflight_min_packets = preflight_min_packets
        self.preflight_duration_ns = preflight_duration_ns

        self._lock = threading.RLock()
        self._recent_packets: List[RobotStatePacket] = []
        self._last_packet: Optional[RobotStatePacket] = None
        self._capture_first_packet: Optional[RobotStatePacket] = None
        self._capture_last_packet: Optional[RobotStatePacket] = None
        self._captured_packet_count = 0
        self._captured_max_gap_ns = 0
        self._raw_digest = hashlib.sha256()
        self._raw_bytes_written = 0
        self._frame_count = 0
        self._frame_index_digest = hashlib.sha256()
        self._frame_index_bytes_written = 0
        self._last_frame_row: Optional[Dict[str, object]] = None
        self._last_frame_by_camera: Dict[str, Tuple[int, int]] = {}

        self._raw: Optional[IO[bytes]] = None
        self._compatible_handle: Optional[IO[str]] = None
        self._audit_handle: Optional[IO[str]] = None
        self._frame_index: Optional[IO[str]] = None
        self._compatible_writer: Optional[Any] = None
        self._audit_writer: Optional[Any] = None

        self._recording = False
        self._finalized = False
        self._fault_reason = ""

    @property
    def faulted(self) -> bool:
        return bool(self._fault_reason)

    @property
    def fault_reason(self) -> str:
        return self._fault_reason

    @property
    def recording(self) -> bool:
        return self._recording

    def _raise_if_unavailable(self) -> None:
        if self.faulted:
            raise RobotCaptureError(
                f"robot session capture is faulted: {self._fault_reason}"
            )
        if self._finalized:
            raise RobotCaptureError("robot session capture is already finalized")

    def _close_writers(self) -> Optional[Exception]:
        first_error: Optional[Exception] = None
        handles = (
            self._raw,
            self._compatible_handle,
            self._audit_handle,
            self._frame_index,
        )
        for handle in handles:
            if handle is None or handle.closed:
                continue
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except Exception as exc:  # noqa: BLE001 - preserve every partial
                if first_error is None:
                    first_error = exc
            try:
                handle.close()
            except Exception as exc:  # noqa: BLE001 - preserve every partial
                if first_error is None:
                    first_error = exc
        self._recording = False
        return first_error

    def _abort(self, reason: str) -> RobotCaptureError:
        if not self._fault_reason:
            self._fault_reason = str(reason)
        close_error = self._close_writers()
        try:
            _fsync_directory(self.session_dir)
        except OSError as exc:
            if close_error is None:
                close_error = exc
        if close_error is not None:
            self._fault_reason = (
                f"{self._fault_reason}; partial flush/close failed: {close_error}"
            )
        return RobotCaptureError(self._fault_reason)

    def abort(self, reason: str) -> None:
        with self._lock:
            if self._finalized:
                raise RobotCaptureError("cannot abort a finalized robot capture")
            raise self._abort(reason or "robot capture aborted")

    def ingest_datagram(self, payload: bytes) -> RobotStatePacket:
        """Validate one publisher datagram and durably append it while recording."""

        with self._lock:
            self._raise_if_unavailable()
            if not isinstance(payload, bytes):
                raise self._abort("robot-state datagram must be bytes")
            if b"\n" in payload or b"\r" in payload:
                raise self._abort(
                    "robot-state datagram must be one JSONL-safe line"
                )
            try:
                packet = self.stream.ingest_datagram(payload)
            except RobotTelemetryError as exc:
                raise self._abort(str(exc)) from exc

            if self._last_packet is not None:
                gap_ns = packet.monotonic_ns - self._last_packet.monotonic_ns
                if gap_ns > self.contract.max_gap_ns:
                    raise self._abort(
                        "robot telemetry gap exceeds 20 ms: "
                        f"{gap_ns} ns"
                    )
                if gap_ns <= 0:
                    raise self._abort(
                        "robot telemetry CLOCK_MONOTONIC clock rollback"
                    )
            else:
                gap_ns = 0

            self._last_packet = packet
            self._recent_packets.append(packet)
            cutoff = packet.monotonic_ns - self.contract.buffer_duration_ns
            while self._recent_packets and (
                self._recent_packets[0].monotonic_ns < cutoff
                or len(self._recent_packets) > self.contract.max_packets
            ):
                self._recent_packets.pop(0)

            if self._recording:
                try:
                    assert self._raw is not None
                    assert self._compatible_writer is not None
                    assert self._audit_writer is not None
                    raw_record = payload + b"\n"
                    self._raw.write(raw_record)
                    self._compatible_writer.writerow(_compatible_row(packet))
                    self._audit_writer.writerow(_audit_row(packet))
                    self._raw.flush()
                    assert self._compatible_handle is not None
                    assert self._audit_handle is not None
                    self._compatible_handle.flush()
                    self._audit_handle.flush()
                except Exception as exc:  # noqa: BLE001 - any sink error is fatal
                    raise self._abort(
                        f"robot telemetry partial write failed: {exc}"
                    ) from exc
                self._raw_digest.update(raw_record)
                self._raw_bytes_written += len(raw_record)

                if self._capture_first_packet is None:
                    self._capture_first_packet = packet
                elif self._capture_last_packet is not None:
                    self._captured_max_gap_ns = max(
                        self._captured_max_gap_ns,
                        packet.monotonic_ns
                        - self._capture_last_packet.monotonic_ns,
                    )
                self._capture_last_packet = packet
                self._captured_packet_count += 1
            return packet

    def start(self, *, now_monotonic_ns: int) -> None:
        """Open partials only after a fresh continuous can1 -> can3 preflight."""

        with self._lock:
            self._raise_if_unavailable()
            if self._recording:
                raise RobotCaptureError("robot session capture is already recording")
            if not _is_int(now_monotonic_ns) or now_monotonic_ns <= 0:
                raise RobotCaptureError(
                    "robot preflight query time must be positive integer ns"
                )
            if len(self._recent_packets) < self.preflight_min_packets:
                raise RobotCaptureError(
                    "robot preflight has too few continuous packets"
                )
            latest = self._recent_packets[-1]
            if now_monotonic_ns < latest.monotonic_ns:
                raise RobotCaptureError(
                    "robot preflight query clock precedes latest telemetry"
                )
            if (
                now_monotonic_ns - latest.monotonic_ns
                > self.contract.max_staleness_ns
            ):
                raise RobotCaptureError("robot preflight telemetry is stale")
            if (
                latest.monotonic_ns - self._recent_packets[0].monotonic_ns
                < self.preflight_duration_ns
            ):
                raise RobotCaptureError(
                    "robot preflight continuous duration is too short"
                )
            gaps = [
                right.monotonic_ns - left.monotonic_ns
                for left, right in zip(
                    self._recent_packets, self._recent_packets[1:]
                )
            ]
            if not gaps or any(
                gap <= 0 or gap > self.contract.max_gap_ns for gap in gaps
            ):
                raise self._abort(
                    "robot preflight is not continuous within the 20 ms gap bound"
                )
            if any(
                packet.side != "left"
                or packet.leader_can_interface != "can1"
                or packet.follower_can_interface != "can3"
                for packet in self._recent_packets
            ):
                raise self._abort(
                    "robot preflight is not the left can1 leader -> can3 follower pair"
                )

            for partial_name, final_name in _PARTIAL_TO_FINAL:
                if (self.session_dir / partial_name).exists():
                    raise self._abort(
                        f"robot capture partial already exists: {partial_name}"
                    )
                if (self.session_dir / final_name).exists():
                    raise self._abort(
                        f"robot capture artifact already exists: {final_name}"
                    )

            try:
                self._raw = (self.session_dir / RAW_PARTIAL_NAME).open("xb")
                self._compatible_handle = (
                    self.session_dir / COMPATIBLE_PARTIAL_NAME
                ).open("x", newline="", encoding="utf-8")
                self._audit_handle = (
                    self.session_dir / AUDIT_PARTIAL_NAME
                ).open("x", newline="", encoding="utf-8")
                self._frame_index = (
                    self.session_dir / FRAME_INDEX_PARTIAL_NAME
                ).open("x", newline="\n", encoding="utf-8")
                self._compatible_writer = csv.writer(
                    self._compatible_handle, lineterminator="\n"
                )
                self._audit_writer = csv.writer(
                    self._audit_handle, lineterminator="\n"
                )
                self._compatible_writer.writerow(_compatible_header())
                self._audit_writer.writerow(_audit_header(self.contract.joint_names))
                self._compatible_handle.flush()
                self._audit_handle.flush()
                self._frame_index.flush()
                self._raw.flush()
                _fsync_directory(self.session_dir)
            except Exception as exc:  # noqa: BLE001 - preserve any created partial
                raise self._abort(
                    f"robot capture partial initialization failed: {exc}"
                ) from exc
            self._recording = True

    def record_frame_evidence(
        self,
        *,
        camera_id: str,
        frame_index: int,
        exposure_lower_ns: int,
        exposure_nominal_ns: int,
        exposure_upper_ns: int,
        now_monotonic_ns: int,
        camera_profile_sha256: str,
        timing_artifact_sha256: str,
    ) -> DiagnosticV1CapturedRobotFrameEvidence:
        """Bind one camera frame to a complete, non-extrapolated robot bracket."""

        with self._lock:
            self._raise_if_unavailable()
            if not self._recording:
                raise RobotCaptureError("robot session capture is not recording")
            if (
                not isinstance(camera_id, str)
                or _CAMERA_ID_RE.fullmatch(camera_id) is None
            ):
                raise self._abort("camera ID is invalid")
            if not _is_int(frame_index) or frame_index < 0:
                raise self._abort("camera frame index must be a nonnegative integer")
            previous = self._last_frame_by_camera.get(camera_id)
            if previous is not None and (
                frame_index <= previous[0]
                or exposure_nominal_ns <= previous[1]
            ):
                raise self._abort(
                    "camera frame indices and exposure times must increase"
                )
            try:
                trajectory = self.stream.query_exposure_interval(
                    exposure_lower_ns=exposure_lower_ns,
                    exposure_nominal_ns=exposure_nominal_ns,
                    exposure_upper_ns=exposure_upper_ns,
                    now_monotonic_ns=now_monotonic_ns,
                    camera_profile_sha256=camera_profile_sha256,
                    timing_artifact_sha256=timing_artifact_sha256,
                )
            except RobotTelemetryError as exc:
                raise self._abort(str(exc)) from exc

            first = self._capture_first_packet
            last = self._capture_last_packet
            if (
                first is None
                or last is None
                or trajectory.sequence_first < first.seq
                or trajectory.sequence_last > last.seq
            ):
                raise self._abort(
                    "camera exposure robot bracket is outside the recorded session"
                )
            row: Dict[str, object] = {
                "schema": DIAGNOSTIC_V1_FRAME_INDEX_SCHEMA,
                "camera_id": camera_id,
                "frame_index": frame_index,
                "exposure_lower_ns": exposure_lower_ns,
                "exposure_nominal_ns": exposure_nominal_ns,
                "exposure_upper_ns": exposure_upper_ns,
                "query_now_monotonic_ns": now_monotonic_ns,
                "robot_seq_first": trajectory.sequence_first,
                "robot_seq_last": trajectory.sequence_last,
                "robot_timestamp_first_ns": trajectory.before.monotonic_ns,
                "robot_timestamp_last_ns": trajectory.after.monotonic_ns,
                "robot_max_gap_ns": trajectory.max_gap_ns,
                "controller_pid": trajectory.before.controller_pid,
                "controller_start_token": (
                    trajectory.before.controller_start_token
                ),
                "publisher_identity": trajectory.publisher_identity,
                "side": trajectory.before.side,
                "controller_mode": trajectory.before.controller_mode,
                "leader_can_interface": (
                    trajectory.before.leader_can_interface
                ),
                "follower_can_interface": (
                    trajectory.before.follower_can_interface
                ),
                "urdf_sha256": trajectory.before.urdf_sha256,
                "joint_mapper_sha256": (
                    trajectory.before.joint_mapper_sha256
                ),
                "camera_profile_sha256": camera_profile_sha256,
                "timing_artifact_sha256": timing_artifact_sha256,
                "pair_contract_sha256": (
                    trajectory.before.pair_contract_sha256
                ),
                "controller_binary_sha256": (
                    trajectory.before.controller_binary_sha256
                ),
            }
            try:
                assert self._frame_index is not None
                frame_record = (
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                self._frame_index.write(frame_record)
                self._frame_index.flush()
            except Exception as exc:  # noqa: BLE001 - missing frame evidence is fatal
                raise self._abort(
                    f"robot frame-index partial write failed: {exc}"
                ) from exc

            frame_record_bytes = frame_record.encode("utf-8")
            self._frame_index_digest.update(frame_record_bytes)
            self._frame_index_bytes_written += len(frame_record_bytes)
            self._frame_count += 1
            self._last_frame_row = row
            self._last_frame_by_camera[camera_id] = (
                frame_index,
                exposure_nominal_ns,
            )
            return DiagnosticV1CapturedRobotFrameEvidence(
                trajectory=trajectory,
                index=MappingProxyType(dict(row)),
            )

    def _validate_csv(
        self,
        *,
        path: Path,
        expected_header: Sequence[str],
        expected_rows: Sequence[Sequence[str]],
        label: str,
    ) -> None:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise RobotCaptureError(f"{label} CSV cannot be reread") from exc
        if not rows or rows[0] != list(expected_header):
            raise RobotCaptureError(f"{label} CSV header does not match contract")
        if len(rows) - 1 != len(expected_rows):
            raise RobotCaptureError(f"{label} CSV packet_count mismatch")
        width = len(expected_header)
        for index, (actual, expected) in enumerate(
            zip(rows[1:], expected_rows)
        ):
            if len(actual) != width:
                raise RobotCaptureError(
                    f"{label} CSV field width mismatch at row {index}"
                )
            if actual != list(expected):
                raise RobotCaptureError(
                    f"{label} CSV row {index} is not derived from the raw packet"
                )

    def _read_raw_packets(self) -> Tuple[RobotStatePacket, ...]:
        path = self.session_dir / RAW_PARTIAL_NAME
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RobotCaptureError("raw robot telemetry cannot be reread") from exc
        if (
            len(raw) != self._raw_bytes_written
            or hashlib.sha256(raw).digest() != self._raw_digest.digest()
        ):
            raise RobotCaptureError(
                "raw robot telemetry byte digest does not match the live writer"
            )
        if not raw or not raw.endswith(b"\n"):
            raise RobotCaptureError(
                "raw robot telemetry must contain newline-terminated packets"
            )
        records = raw.splitlines(keepends=True)
        packets: List[RobotStatePacket] = []
        for index, record in enumerate(records):
            if not record.endswith(b"\n") or b"\r" in record:
                raise RobotCaptureError(
                    f"raw robot telemetry JSONL framing is invalid at row {index}"
                )
            payload = record[:-1]
            if b"\n" in payload:
                raise RobotCaptureError(
                    f"raw robot telemetry datagram is not lossless at row {index}"
                )
            try:
                packets.append(parse_robot_state_datagram(payload, self.contract))
            except RobotTelemetryError as exc:
                raise RobotCaptureError(
                    f"raw robot telemetry row {index} violates the exact packet schema"
                ) from exc
        if not packets:
            raise RobotCaptureError("raw robot telemetry has no packets")
        return tuple(packets)

    def _read_frame_rows(
        self, packets_by_seq: Mapping[int, RobotStatePacket]
    ) -> List[Dict[str, object]]:
        path = self.session_dir / FRAME_INDEX_PARTIAL_NAME
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RobotCaptureError("robot frame index cannot be reread") from exc
        if (
            len(raw) != self._frame_index_bytes_written
            or hashlib.sha256(raw).digest()
            != self._frame_index_digest.digest()
        ):
            raise RobotCaptureError(
                "robot frame-index byte digest does not match the live writer"
            )
        if not raw or not raw.endswith(b"\n"):
            raise RobotCaptureError(
                "robot frame index must contain newline-terminated rows"
            )
        rows: List[Dict[str, object]] = []
        per_camera: Dict[str, Tuple[int, int]] = {}
        for line_index, line in enumerate(raw.splitlines(keepends=True)):
            if not line.endswith(b"\n") or b"\r" in line:
                raise RobotCaptureError(
                    f"robot frame index framing is invalid at row {line_index}"
                )
            try:
                value = json.loads(
                    line[:-1].decode("utf-8"),
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
            except RobotCaptureError:
                raise
            except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise RobotCaptureError(
                    f"robot frame index row {line_index} is invalid JSON"
                ) from exc
            if not isinstance(value, dict) or frozenset(value) != _FRAME_INDEX_FIELDS:
                raise RobotCaptureError(
                    f"robot frame index field width mismatch at row {line_index}"
                )
            if value["schema"] != DIAGNOSTIC_V1_FRAME_INDEX_SCHEMA:
                raise RobotCaptureError("robot frame index schema is unsupported")
            camera_id = value["camera_id"]
            frame_index = value["frame_index"]
            if (
                not isinstance(camera_id, str)
                or _CAMERA_ID_RE.fullmatch(camera_id) is None
                or not _is_int(frame_index)
                or frame_index < 0
            ):
                raise RobotCaptureError("robot frame index camera identity is invalid")
            integer_fields = (
                "exposure_lower_ns",
                "exposure_nominal_ns",
                "exposure_upper_ns",
                "query_now_monotonic_ns",
                "robot_seq_first",
                "robot_seq_last",
                "robot_timestamp_first_ns",
                "robot_timestamp_last_ns",
                "robot_max_gap_ns",
                "controller_pid",
            )
            if any(
                not _is_int(value[field]) or value[field] < 0
                for field in integer_fields
            ):
                raise RobotCaptureError("robot frame index integer field is invalid")
            lower = value["exposure_lower_ns"]
            nominal = value["exposure_nominal_ns"]
            upper = value["exposure_upper_ns"]
            query_now = value["query_now_monotonic_ns"]
            if not 0 < lower <= nominal <= upper <= query_now:
                raise RobotCaptureError(
                    "robot frame index exposure ordering is invalid"
                )
            previous = per_camera.get(camera_id)
            if previous is not None and (
                frame_index <= previous[0] or nominal <= previous[1]
            ):
                raise RobotCaptureError(
                    "robot frame index is not increasing per camera"
                )
            per_camera[camera_id] = (frame_index, nominal)

            sequence_first = value["robot_seq_first"]
            sequence_last = value["robot_seq_last"]
            first = packets_by_seq.get(sequence_first)
            last = packets_by_seq.get(sequence_last)
            if first is None or last is None or sequence_first > sequence_last:
                raise RobotCaptureError(
                    "robot frame index references an unavailable packet bracket"
                )
            bracket: List[RobotStatePacket] = []
            for sequence in range(sequence_first, sequence_last + 1):
                packet = packets_by_seq.get(sequence)
                if packet is None:
                    raise RobotCaptureError(
                        "robot frame index packet bracket is not continuous"
                    )
                bracket.append(packet)
            gaps = [
                right.monotonic_ns - left.monotonic_ns
                for left, right in zip(bracket, bracket[1:])
            ]
            max_gap = max(gaps) if gaps else 0
            if (
                first.monotonic_ns > lower
                or last.monotonic_ns < upper
                or any(
                    gap <= 0 or gap > self.contract.max_gap_ns
                    for gap in gaps
                )
                or value["robot_timestamp_first_ns"] != first.monotonic_ns
                or value["robot_timestamp_last_ns"] != last.monotonic_ns
                or value["robot_max_gap_ns"] != max_gap
            ):
                raise RobotCaptureError(
                    "robot frame index does not bracket the complete exposure"
                )
            expected_identity = {
                "controller_pid": first.controller_pid,
                "controller_start_token": first.controller_start_token,
                "publisher_identity": first.publisher_identity,
                "side": "left",
                "controller_mode": "unilateral",
                "leader_can_interface": "can1",
                "follower_can_interface": "can3",
                "urdf_sha256": self.contract.urdf_sha256,
                "joint_mapper_sha256": self.contract.joint_mapper_sha256,
                "camera_profile_sha256": self.contract.camera_profile_sha256,
                "timing_artifact_sha256": self.contract.timing_artifact_sha256,
                "pair_contract_sha256": self.contract.pair_contract_sha256,
                "controller_binary_sha256": (
                    self.contract.controller_binary_sha256
                ),
            }
            if any(value[field] != expected for field, expected in expected_identity.items()):
                raise RobotCaptureError(
                    "robot frame index provenance identity mismatch"
                )
            rows.append(value)
        return rows

    def _verify_partials(self) -> _VerifiedPartials:
        packets = self._read_raw_packets()
        if len(packets) != self._captured_packet_count:
            raise RobotCaptureError(
                "raw robot telemetry packet_count does not match the live writer"
            )
        first_sequence = packets[0].seq
        last_sequence = packets[-1].seq
        first_process_identity = packets[0].process_identity
        first_publisher_identity = packets[0].publisher_identity
        if any(
            packet.process_identity != first_process_identity
            or packet.publisher_identity != first_publisher_identity
            for packet in packets[1:]
        ):
            raise RobotCaptureError(
                "raw robot telemetry publisher identity changed"
            )
        if len(packets) != last_sequence - first_sequence + 1:
            raise RobotCaptureError(
                "raw robot telemetry packet_count != sequence_last - sequence_first + 1"
            )
        gaps: List[int] = []
        for left, right in zip(packets, packets[1:]):
            if right.seq != left.seq + 1:
                raise RobotCaptureError(
                    "raw robot telemetry sequence is not contiguous"
                )
            gap = right.monotonic_ns - left.monotonic_ns
            if gap <= 0 or gap > self.contract.max_gap_ns:
                raise RobotCaptureError(
                    "raw robot telemetry max gap exceeds 20 ms"
                )
            gaps.append(gap)
        if (
            self._capture_first_packet is None
            or self._capture_last_packet is None
            or packets[0].seq != self._capture_first_packet.seq
            or packets[-1].seq != self._capture_last_packet.seq
        ):
            raise RobotCaptureError(
                "raw robot telemetry sequence bounds changed after capture"
            )
        max_gap_ns = max(gaps) if gaps else 0
        if max_gap_ns != self._captured_max_gap_ns:
            raise RobotCaptureError(
                "raw robot telemetry maximum gap changed after capture"
            )

        self._validate_csv(
            path=self.session_dir / COMPATIBLE_PARTIAL_NAME,
            expected_header=_compatible_header(),
            expected_rows=[_compatible_row(packet) for packet in packets],
            label="compatible robot",
        )
        self._validate_csv(
            path=self.session_dir / AUDIT_PARTIAL_NAME,
            expected_header=_audit_header(self.contract.joint_names),
            expected_rows=[_audit_row(packet) for packet in packets],
            label="robot leader audit",
        )
        frames = self._read_frame_rows(
            {packet.seq: packet for packet in packets}
        )
        if len(frames) != self._frame_count:
            raise RobotCaptureError(
                "robot frame-index frame_count does not match the live writer"
            )
        if not frames:
            raise RobotCaptureError("robot capture contains no camera frame evidence")

        inventory: Dict[str, Mapping[str, object]] = {}
        for partial_name, final_name in _PARTIAL_TO_FINAL[:-1]:
            path = self.session_dir / partial_name
            inventory[final_name] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        return _VerifiedPartials(
            packets=packets,
            frame_count=len(frames),
            max_gap_ns=max_gap_ns,
            hash_inventory=inventory,
        )

    def _write_summary_partial(
        self, verified: _VerifiedPartials
    ) -> Dict[str, object]:
        first = verified.packets[0]
        last = verified.packets[-1]
        summary: Dict[str, object] = {
            "schema": DIAGNOSTIC_V1_CAPTURE_SUMMARY_SCHEMA,
            "status": "robot_capture_validated",
            "side": "left",
            "controller_mode": "unilateral",
            "leader_can_interface": "can1",
            "follower_can_interface": "can3",
            "joint_names": list(self.contract.joint_names),
            "packet_count": len(verified.packets),
            "sequence_first": first.seq,
            "sequence_last": last.seq,
            "monotonic_ns_first": first.monotonic_ns,
            "monotonic_ns_last": last.monotonic_ns,
            "max_gap_ns": verified.max_gap_ns,
            "frame_count": verified.frame_count,
            "controller_pid": first.controller_pid,
            "controller_start_token": first.controller_start_token,
            "publisher_identity": first.publisher_identity,
            "urdf_sha256": self.contract.urdf_sha256,
            "joint_mapper_sha256": self.contract.joint_mapper_sha256,
            "camera_profile_sha256": self.contract.camera_profile_sha256,
            "timing_artifact_sha256": self.contract.timing_artifact_sha256,
            "pair_contract_sha256": self.contract.pair_contract_sha256,
            "controller_binary_sha256": (
                self.contract.controller_binary_sha256
            ),
            "column_semantics": {
                "observation": "follower_measured",
                "leader": "measured_audit_only",
                "action": "follower_command",
                "position_unit": "motor_rad",
                "velocity_unit": "motor_rad_s",
            },
            "compatibility_csv": {
                "path": "robot.csv",
                "state_columns": "q_joint_1..q_joint_7",
                "action_columns": "a_joint_1..a_joint_7",
                "gripper_and_leader_audit_path": "robot_leader_audit.csv",
            },
            "verified_field_widths": {
                "joint_count": 7,
                "compatible_csv_columns": len(_compatible_header()),
                "audit_csv_columns": len(
                    _audit_header(self.contract.joint_names)
                ),
            },
            "hash_inventory": dict(verified.hash_inventory),
            "eligibility_decision": "deferred_to_combined_session_finalizer",
        }
        path = self.session_dir / SUMMARY_PARTIAL_NAME
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:  # noqa: BLE001 - retain all existing partials
            raise RobotCaptureError(
                f"robot capture summary partial write failed: {exc}"
            ) from exc
        return summary

    def _verify_summary_partial(
        self, expected: Mapping[str, object]
    ) -> None:
        path = self.session_dir / SUMMARY_PARTIAL_NAME
        try:
            raw = path.read_bytes()
            observed = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
        except RobotCaptureError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RobotCaptureError(
                "robot capture summary partial cannot be reread"
            ) from exc
        if observed != dict(expected):
            raise RobotCaptureError(
                "robot capture summary partial changed after write"
            )
        if "training_eligible" in observed:
            raise RobotCaptureError(
                "robot capture summary must not decide training eligibility"
            )
        inventory = observed.get("hash_inventory")
        if not isinstance(inventory, dict):
            raise RobotCaptureError(
                "robot capture summary hash inventory is invalid"
            )
        for partial_name, final_name in _PARTIAL_TO_FINAL[:-1]:
            expected_item = inventory.get(final_name)
            path = self.session_dir / partial_name
            actual_item = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            if expected_item != actual_item:
                raise RobotCaptureError(
                    f"robot capture hash inventory mismatch for {final_name}"
                )

    def _promote_partials(
        self, expected_inventory: Mapping[str, object]
    ) -> None:
        for _, final_name in _PARTIAL_TO_FINAL:
            if (self.session_dir / final_name).exists():
                raise RobotCaptureError(
                    f"robot final artifact already exists: {final_name}"
                )
        data_pairs = _PARTIAL_TO_FINAL[:-1]
        summary_partial_name, summary_final_name = _PARTIAL_TO_FINAL[-1]
        linked: List[Tuple[Path, Path]] = []
        summary_promoted = False
        try:
            for partial_name, final_name in data_pairs:
                partial = self.session_dir / partial_name
                final = self.session_dir / final_name
                os.link(partial, final)
                linked.append((partial, final))
                expected_item = expected_inventory.get(final_name)
                actual_item = {
                    "sha256": _sha256(final),
                    "size_bytes": final.stat().st_size,
                }
                if expected_item != actual_item:
                    raise RobotCaptureError(
                        f"robot capture pre-commit hash mismatch for {final_name}"
                    )
            # All immutable data names are durable before the summary exists.
            _fsync_directory(self.session_dir)
            summary_partial = self.session_dir / summary_partial_name
            summary_final = self.session_dir / summary_final_name
            os.replace(summary_partial, summary_final)
            summary_promoted = True
            # The summary is the single atomic commit marker.
            _fsync_directory(self.session_dir)
            for partial, _ in linked:
                partial.unlink()
            _fsync_directory(self.session_dir)
        except Exception as exc:  # noqa: BLE001 - roll back to fail-closed partials
            rollback_errors: List[str] = []
            summary_partial = self.session_dir / summary_partial_name
            summary_final = self.session_dir / summary_final_name
            if summary_promoted and summary_final.exists():
                try:
                    os.replace(summary_final, summary_partial)
                except OSError as rollback_exc:
                    rollback_errors.append(str(rollback_exc))
            for partial, final in reversed(linked):
                try:
                    if not partial.exists() and final.exists():
                        os.link(final, partial)
                    if final.exists():
                        final.unlink()
                except OSError as rollback_exc:
                    rollback_errors.append(str(rollback_exc))
            try:
                _fsync_directory(self.session_dir)
            except OSError as sync_exc:
                rollback_errors.append(str(sync_exc))
            suffix = (
                f"; rollback errors={rollback_errors}"
                if rollback_errors
                else ""
            )
            raise RobotCaptureError(
                f"robot capture atomic promotion failed: {exc}{suffix}"
            ) from exc

    def finish(self, *, now_monotonic_ns: int) -> Mapping[str, object]:
        """Recheck the final exposure, reread all partials, and atomically promote."""

        with self._lock:
            self._raise_if_unavailable()
            if not self._recording:
                raise RobotCaptureError("robot session capture is not recording")
            if self._last_frame_row is None:
                raise self._abort(
                    "robot capture cannot finish without camera frame evidence"
                )
            if (
                not _is_int(now_monotonic_ns)
                or now_monotonic_ns
                < int(self._last_frame_row["exposure_upper_ns"])
            ):
                raise self._abort(
                    "End must not precede the last camera exposure upper bound"
                )

            try:
                final_bracket = self.stream.query_exposure_interval(
                    exposure_lower_ns=int(
                        self._last_frame_row["exposure_lower_ns"]
                    ),
                    exposure_nominal_ns=int(
                        self._last_frame_row["exposure_nominal_ns"]
                    ),
                    exposure_upper_ns=int(
                        self._last_frame_row["exposure_upper_ns"]
                    ),
                    now_monotonic_ns=now_monotonic_ns,
                    camera_profile_sha256=str(
                        self._last_frame_row["camera_profile_sha256"]
                    ),
                    timing_artifact_sha256=str(
                        self._last_frame_row["timing_artifact_sha256"]
                    ),
                )
            except RobotTelemetryError as exc:
                raise self._abort(
                    f"End has no final exposure upper bracket: {exc}"
                ) from exc
            if (
                final_bracket.sequence_first
                != self._last_frame_row["robot_seq_first"]
                or final_bracket.sequence_last
                != self._last_frame_row["robot_seq_last"]
                or final_bracket.after.monotonic_ns
                < int(self._last_frame_row["exposure_upper_ns"])
            ):
                raise self._abort(
                    "End final exposure upper bracket identity changed"
                )

            close_error = self._close_writers()
            if close_error is not None:
                raise self._abort(
                    f"robot capture partial fsync/close failed: {close_error}"
                )
            try:
                _fsync_directory(self.session_dir)
                verified = self._verify_partials()
                summary = self._write_summary_partial(verified)
                self._verify_summary_partial(summary)
                _fsync_directory(self.session_dir)
                self._promote_partials(summary["hash_inventory"])
            except RobotCaptureError as exc:
                raise self._abort(str(exc)) from exc
            except (OSError, ValueError, TypeError) as exc:
                raise self._abort(
                    f"robot capture final verification failed: {exc}"
                ) from exc
            self._finalized = True
            return summary


EXACT_CAPTURE_SUMMARY_SCHEMA_V2_COMMISSIONING = (
    "wam_openarm_exact_control_capture_summary_v2"
)
EXACT_FRAME_INDEX_SCHEMA_V2_COMMISSIONING = (
    "wam_openarm_exact_control_frame_evidence_v2"
)
EXACT_CAPTURE_SUMMARY_SCHEMA_V3_PRODUCTION = (
    "wam_openarm_exact_control_capture_summary_v3"
)
EXACT_FRAME_INDEX_SCHEMA_V3_PRODUCTION = (
    "wam_openarm_exact_control_frame_evidence_v3"
)
EXACT_CAPTURE_SUMMARY_SCHEMA_V4_PRODUCTION = (
    "wam_openarm_exact_control_capture_summary_v4"
)
EXACT_FRAME_INDEX_SCHEMA_V4_PRODUCTION = (
    "wam_openarm_exact_control_frame_evidence_v4"
)
FEEDBACK_START_BASELINE_SCHEMA = "wam_openarm_feedback_start_baseline_v1"
TRAJECTORY_VALIDATION_SCHEMA = "wam_openarm_trajectory_validation_v1"
TRAJECTORY_MIN_WHOLE_7D_RANGE_RAD = 1.0e-4
COMMAND_SIGNIFICANT_RANGE_RAD = 1.0e-2
COMMAND_RESPONSE_FROZEN_RANGE_RAD = 1.0e-4
COMMAND_RESPONSE_MIN_TRACKING_GAIN = 0.10
COMMAND_RESPONSE_MAX_TRACKING_GAIN = 5.0
COMMAND_RESPONSE_MIN_CORRELATION = 0.50
COMMAND_RESPONSE_DIRECTION_MIN_NET_FRACTION = 0.50
FEEDBACK_LIVENESS_UNOBSERVABLE_REASON = (
    "v3_has_no_per_motor_feedback_rx_counters_or_timestamps"
)
# Deprecated compatibility aliases. Production code below uses only the
# explicit commissioning/production names.
EXACT_CAPTURE_SUMMARY_SCHEMA = EXACT_CAPTURE_SUMMARY_SCHEMA_V2_COMMISSIONING
EXACT_FRAME_INDEX_SCHEMA = EXACT_FRAME_INDEX_SCHEMA_V2_COMMISSIONING
EXACT_CAPTURE_SUMMARY_SCHEMA_V3 = EXACT_CAPTURE_SUMMARY_SCHEMA_V3_PRODUCTION
EXACT_FRAME_INDEX_SCHEMA_V3 = EXACT_FRAME_INDEX_SCHEMA_V3_PRODUCTION
CONTROL_ENVELOPES_PARTIAL_NAME = "robot_control_envelopes.jsonl.partial"
EXACT_CONTROL_PARTIAL_NAME = "robot_exact_control.csv.partial"
EXACT_FRAME_INDEX_PARTIAL_NAME = "robot_frame_index.jsonl.partial"
EXACT_SUMMARY_PARTIAL_NAME = "robot_capture_summary.json.partial"
CONTROL_ENVELOPES_NAME = "robot_control_envelopes.jsonl"
EXACT_CONTROL_NAME = "robot_exact_control.csv"
EXACT_FRAME_INDEX_NAME = "robot_frame_index.jsonl"
EXACT_SUMMARY_NAME = "robot_capture_summary.json"
EXACT_ROBOT_ARTIFACT_NAMES = (
    CONTROL_ENVELOPES_NAME,
    EXACT_CONTROL_NAME,
    EXACT_FRAME_INDEX_NAME,
    EXACT_SUMMARY_NAME,
)
_EXACT_PARTIAL_TO_FINAL = (
    (CONTROL_ENVELOPES_PARTIAL_NAME, CONTROL_ENVELOPES_NAME),
    (EXACT_CONTROL_PARTIAL_NAME, EXACT_CONTROL_NAME),
    (EXACT_FRAME_INDEX_PARTIAL_NAME, EXACT_FRAME_INDEX_NAME),
    (EXACT_SUMMARY_PARTIAL_NAME, EXACT_SUMMARY_NAME),
)


def _exact_control_header(
    joint_names: Sequence[str], *, include_tx_status: bool = False,
    include_feedback_liveness: bool = False,
) -> List[str]:
    fields = [
        "envelope_seq",
        "control_tick_seq",
        "observation_monotonic_ns",
        "reference_generation",
        "reference_source_monotonic_ns",
        "control_call_begin_monotonic_ns",
        "arm_call_return_monotonic_ns",
        "gripper_call_begin_monotonic_ns",
        "gripper_call_return_monotonic_ns",
        "control_call_end_monotonic_ns",
    ]
    if include_tx_status:
        fields += [
            "arm_tx_expected_frame_count",
            "arm_tx_full_frame_write_count",
            "arm_tx_failed_frame_write_count",
            "gripper_tx_expected_frame_count",
            "gripper_tx_full_frame_write_count",
            "gripper_tx_failed_frame_write_count",
            "socketcan_tx_enqueue_success",
        ]
    if include_feedback_liveness:
        fields.append("feedback_snapshot_monotonic_ns")
        for prefix in (
            "state_rx_motor_id",
            "state_rx_counter",
            "state_rx_generation",
            "state_rx_monotonic_ns",
            "state_rx_last_gap_ns",
            "state_rx_gap_violation_count",
        ):
            fields += [
                f"feedback.follower.{name}.{prefix}" for name in joint_names
            ]
        fields += [
            "feedback.follower.gripper.state_rx_motor_id",
            "feedback.follower.gripper.state_rx_counter",
            "feedback.follower.gripper.state_rx_generation",
            "feedback.follower.gripper.state_rx_monotonic_ns",
            "feedback.follower.gripper.state_rx_last_gap_ns",
            "feedback.follower.gripper.state_rx_gap_violation_count",
        ]
    fields += [
        f"observation.follower.{name}.motor_rad" for name in joint_names
    ]
    fields += [
        f"observation.follower.{name}.motor_rad_s" for name in joint_names
    ]
    fields += [
        "observation.follower.gripper.joint_rad",
        "observation.follower.gripper.joint_rad_s",
        "observation.follower.gripper.motor_rad",
        "observation.follower.gripper.motor_rad_s",
    ]
    for parameter, unit in (
        ("kp", "gain"),
        ("kd", "gain"),
        ("q", "motor_rad"),
        ("dq", "motor_rad_s"),
        ("tau", "motor_torque"),
    ):
        fields += [
            f"action.arm_mit_{parameter}.{name}.{unit}"
            for name in joint_names
        ]
    fields += [
        "action.gripper_mit_kp.gain",
        "action.gripper_mit_kd.gain",
        "action.gripper_mit_q.motor_rad",
        "action.gripper_mit_dq.motor_rad_s",
        "action.gripper_mit_tau.motor_torque",
    ]
    return fields


def _exact_control_row(record: ExactControlRecord) -> List[str]:
    action = record.action
    integers = (
        record.envelope_seq,
        record.control_tick_seq,
        record.observation_monotonic_ns,
        record.reference_generation,
        record.reference_source_monotonic_ns,
        record.control_call_begin_monotonic_ns,
        record.arm_call_return_monotonic_ns,
        record.gripper_call_begin_monotonic_ns,
        record.gripper_call_return_monotonic_ns,
        record.control_call_end_monotonic_ns,
    )
    fields = [str(value) for value in integers]
    if record.schema in {
        EXACT_CONTROL_RECORD_SCHEMA_V3,
        EXACT_CONTROL_RECORD_SCHEMA_V4,
    }:
        fields += [
            str(record.arm_tx_expected_frame_count),
            str(record.arm_tx_full_frame_write_count),
            str(record.arm_tx_failed_frame_write_count),
            str(record.gripper_tx_expected_frame_count),
            str(record.gripper_tx_full_frame_write_count),
            str(record.gripper_tx_failed_frame_write_count),
            "true" if record.socketcan_tx_enqueue_success else "false",
        ]
    if record.schema == EXACT_CONTROL_RECORD_SCHEMA_V4:
        fields += [str(record.feedback_snapshot_monotonic_ns)]
        for values in (
            record.follower_arm_state_rx_motor_ids,
            record.follower_arm_state_rx_counters,
            record.follower_arm_state_rx_generations,
            record.follower_arm_state_rx_monotonic_ns,
            record.follower_arm_state_rx_last_gap_ns,
            record.follower_arm_state_rx_gap_violation_counts,
        ):
            fields += [str(value) for value in values]
        fields += [
            str(record.follower_gripper_state_rx_motor_id),
            str(record.follower_gripper_state_rx_counter),
            str(record.follower_gripper_state_rx_generation),
            str(record.follower_gripper_state_rx_monotonic_ns),
            str(record.follower_gripper_state_rx_last_gap_ns),
            str(record.follower_gripper_state_rx_gap_violation_count),
        ]
    return (
        fields
        + [_float_text(value) for value in record.follower_q_rad]
        + [_float_text(value) for value in record.follower_qd_rad_s]
        + [
            _float_text(record.follower_gripper_q_rad),
            _float_text(record.follower_gripper_qd_rad_s),
            _float_text(record.follower_gripper_motor_q_rad),
            _float_text(record.follower_gripper_motor_qd_rad_s),
        ]
        + [_float_text(value) for value in action.arm_kp]
        + [_float_text(value) for value in action.arm_kd]
        + [_float_text(value) for value in action.arm_q]
        + [_float_text(value) for value in action.arm_dq]
        + [_float_text(value) for value in action.arm_tau]
        + [
            _float_text(action.gripper_kp),
            _float_text(action.gripper_kd),
            _float_text(action.gripper_q),
            _float_text(action.gripper_dq),
            _float_text(action.gripper_tau),
        ]
    )


def _action_payload(record: ExactControlRecord) -> Dict[str, object]:
    action = record.action
    return {
        "control_tick_seq": record.control_tick_seq,
        "follower_arm_mit_kp": list(action.arm_kp),
        "follower_arm_mit_kd": list(action.arm_kd),
        "follower_arm_mit_q": list(action.arm_q),
        "follower_arm_mit_dq": list(action.arm_dq),
        "follower_arm_mit_tau": list(action.arm_tau),
        "follower_gripper_mit_kp": action.gripper_kp,
        "follower_gripper_mit_kd": action.gripper_kd,
        "follower_gripper_mit_q": action.gripper_q,
        "follower_gripper_mit_dq": action.gripper_dq,
        "follower_gripper_mit_tau": action.gripper_tau,
    }


def _exact_action_sha256(records: Sequence[ExactControlRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            (
                json.dumps(
                    _action_payload(record),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _exact_trajectory_sha256(records: Sequence[ExactControlRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        payload = {
            "envelope_seq": record.envelope_seq,
            "control_tick_seq": record.control_tick_seq,
            "observation_monotonic_ns": record.observation_monotonic_ns,
            "reference_generation": record.reference_generation,
            "reference_source_monotonic_ns": record.reference_source_monotonic_ns,
            "control_call_begin_monotonic_ns": record.control_call_begin_monotonic_ns,
            "arm_call_return_monotonic_ns": record.arm_call_return_monotonic_ns,
            "gripper_call_begin_monotonic_ns": record.gripper_call_begin_monotonic_ns,
            "gripper_call_return_monotonic_ns": record.gripper_call_return_monotonic_ns,
            "control_call_end_monotonic_ns": record.control_call_end_monotonic_ns,
            "follower_q_rad": list(record.follower_q_rad),
            "follower_qd_rad_s": list(record.follower_qd_rad_s),
            "follower_gripper_q_rad": record.follower_gripper_q_rad,
            "follower_gripper_qd_rad_s": record.follower_gripper_qd_rad_s,
            "follower_gripper_motor_q_rad": record.follower_gripper_motor_q_rad,
            "follower_gripper_motor_qd_rad_s": record.follower_gripper_motor_qd_rad_s,
            "action": _action_payload(record),
        }
        if record.schema in {
            EXACT_CONTROL_RECORD_SCHEMA_V3,
            EXACT_CONTROL_RECORD_SCHEMA_V4,
        }:
            payload["socketcan_tx_enqueue"] = {
                "arm_expected_frame_count": record.arm_tx_expected_frame_count,
                "arm_full_frame_write_count": record.arm_tx_full_frame_write_count,
                "arm_failed_frame_write_count": record.arm_tx_failed_frame_write_count,
                "gripper_expected_frame_count": (
                    record.gripper_tx_expected_frame_count
                ),
                "gripper_full_frame_write_count": (
                    record.gripper_tx_full_frame_write_count
                ),
                "gripper_failed_frame_write_count": (
                    record.gripper_tx_failed_frame_write_count
                ),
                "success": record.socketcan_tx_enqueue_success,
            }
        if record.schema == EXACT_CONTROL_RECORD_SCHEMA_V4:
            payload["feedback_liveness"] = _feedback_snapshot_payload(record)
        digest.update(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _feedback_snapshot_payload(record: ExactControlRecord) -> Dict[str, object]:
    if record.schema != EXACT_CONTROL_RECORD_SCHEMA_V4:
        raise RobotCaptureError("feedback snapshot requires a v4 control record")
    return {
        "control_tick_seq": record.control_tick_seq,
        "observation_monotonic_ns": record.observation_monotonic_ns,
        "feedback_snapshot_monotonic_ns": record.feedback_snapshot_monotonic_ns,
        "motor_ids": list(record.follower_arm_state_rx_motor_ids)
        + [record.follower_gripper_state_rx_motor_id],
        "counters": list(record.follower_arm_state_rx_counters)
        + [record.follower_gripper_state_rx_counter],
        "generations": list(record.follower_arm_state_rx_generations)
        + [record.follower_gripper_state_rx_generation],
        "monotonic_ns": list(record.follower_arm_state_rx_monotonic_ns)
        + [record.follower_gripper_state_rx_monotonic_ns],
        "last_gap_ns": list(record.follower_arm_state_rx_last_gap_ns)
        + [record.follower_gripper_state_rx_last_gap_ns],
        "gap_violation_counts": list(
            record.follower_arm_state_rx_gap_violation_counts
        )
        + [record.follower_gripper_state_rx_gap_violation_count],
    }


def _feedback_start_baseline_payload(
    record: ExactControlRecord, *, start_monotonic_ns: int
) -> Dict[str, object]:
    snapshot = _feedback_snapshot_payload(record)
    encoded = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "schema": FEEDBACK_START_BASELINE_SCHEMA,
        "start_monotonic_ns": start_monotonic_ns,
        "publisher_identity": record.publisher_identity,
        **snapshot,
        "snapshot_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _validate_feedback_start_baseline(
    value: object, first_record: ExactControlRecord
) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise RobotCaptureError("v4 feedback Start baseline receipt is missing")
    expected_fields = {
        "schema",
        "start_monotonic_ns",
        "publisher_identity",
        "control_tick_seq",
        "observation_monotonic_ns",
        "feedback_snapshot_monotonic_ns",
        "motor_ids",
        "counters",
        "generations",
        "monotonic_ns",
        "last_gap_ns",
        "gap_violation_counts",
        "snapshot_sha256",
    }
    if set(value) != expected_fields:
        raise RobotCaptureError("v4 feedback Start baseline field set mismatch")
    snapshot = {
        key: value[key]
        for key in (
            "control_tick_seq",
            "observation_monotonic_ns",
            "feedback_snapshot_monotonic_ns",
            "motor_ids",
            "counters",
            "generations",
            "monotonic_ns",
            "last_gap_ns",
            "gap_violation_counts",
        )
    }
    encoded = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if (
        value["schema"] != FEEDBACK_START_BASELINE_SCHEMA
        or value["publisher_identity"] != first_record.publisher_identity
        or value["snapshot_sha256"] != hashlib.sha256(encoded).hexdigest()
        or not _is_int(value["start_monotonic_ns"])
        or not _is_int(value["control_tick_seq"])
        or not _is_int(value["observation_monotonic_ns"])
        or not _is_int(value["feedback_snapshot_monotonic_ns"])
        or value["feedback_snapshot_monotonic_ns"]
        != value["observation_monotonic_ns"]
        or not value["observation_monotonic_ns"]
        <= value["start_monotonic_ns"]
        <= first_record.observation_monotonic_ns
    ):
        raise RobotCaptureError("v4 feedback Start baseline identity is invalid")
    expected_motor_ids = list(first_record.follower_arm_state_rx_motor_ids) + [
        first_record.follower_gripper_state_rx_motor_id
    ]
    sequences = [
        value["motor_ids"],
        value["counters"],
        value["generations"],
        value["monotonic_ns"],
        value["last_gap_ns"],
        value["gap_violation_counts"],
    ]
    if (
        any(not isinstance(sequence, list) or len(sequence) != 8 for sequence in sequences)
        or value["motor_ids"] != expected_motor_ids
        or any(
            not _is_int(item) or item < 0
            for sequence in sequences
            for item in sequence
        )
        or value["generations"] != value["counters"]
    ):
        raise RobotCaptureError("v4 feedback Start baseline vectors are invalid")
    first_snapshot = _feedback_snapshot_payload(first_record)
    for field in ("counters", "monotonic_ns", "gap_violation_counts"):
        if any(
            baseline > observed
            for baseline, observed in zip(value[field], first_snapshot[field])
        ):
            raise RobotCaptureError(
                f"v4 feedback Start baseline {field} exceeds first captured record"
            )
    return value


def _evaluate_v4_feedback(
    baseline_value: object,
    replay: "ExactCaptureReplayStats",
    *,
    end_monotonic_ns: int,
) -> Dict[str, object]:
    baseline = _validate_feedback_start_baseline(
        baseline_value, replay.first_record
    )
    last_snapshot = _feedback_snapshot_payload(replay.last_record)
    baseline_counters = list(baseline["counters"])
    baseline_violations = list(baseline["gap_violation_counts"])
    violation_delta = [
        end - start
        for start, end in zip(
            baseline_violations, last_snapshot["gap_violation_counts"]
        )
    ]
    counter_progress = all(
        end > start
        for start, end in zip(baseline_counters, last_snapshot["counters"])
    )
    if (
        not _is_int(end_monotonic_ns)
        or end_monotonic_ns < replay.last_record.feedback_snapshot_monotonic_ns
    ):
        raise RobotCaptureError("v4 feedback End boundary is invalid")
    end_ages = [
        end_monotonic_ns - timestamp
        for timestamp in last_snapshot["monotonic_ns"]
    ]
    if any(age < 0 for age in end_ages):
        raise RobotCaptureError("v4 feedback End precedes a motor RX timestamp")
    max_age_ns = max(replay.feedback_max_age_ns, *end_ages)
    all_ticks_fresh = (
        replay.feedback_all_ticks_fresh and max_age_ns <= FEEDBACK_MAX_AGE_NS
    )
    gap_clean = (
        all(delta == 0 for delta in violation_delta)
        and replay.feedback_max_last_gap_ns <= FEEDBACK_MAX_GAP_NS
    )
    return {
        "feedback_start_baseline": dict(baseline),
        "capture_end_monotonic_ns": end_monotonic_ns,
        "feedback_counter_progress_all_motors": counter_progress,
        "feedback_gap_violation_count_delta": violation_delta,
        "feedback_all_ticks_fresh": all_ticks_fresh,
        "feedback_max_observed_age_ns": max_age_ns,
        "feedback_max_observed_gap_ns": replay.feedback_max_last_gap_ns,
        "feedback_clean": counter_progress and all_ticks_fresh and gap_clean,
        "feedback_gap_clean": gap_clean,
    }


def _evaluate_training_trajectory(
    replay: "ExactCaptureReplayStats", joint_names: Sequence[str]
) -> Dict[str, object]:
    observation_joint_ranges = [
        upper - lower
        for lower, upper in zip(
            replay.follower_q_min_rad, replay.follower_q_max_rad
        )
    ]
    action_joint_ranges = [
        upper - lower
        for lower, upper in zip(replay.action_q_min_rad, replay.action_q_max_rad)
    ]
    actuator_names = [*joint_names, "gripper"]
    observation_ranges = observation_joint_ranges + [
        replay.follower_gripper_motor_q_max_rad
        - replay.follower_gripper_motor_q_min_rad
    ]
    action_ranges = action_joint_ranges + [
        replay.action_gripper_q_max_rad - replay.action_gripper_q_min_rad
    ]
    first_observations = list(replay.first_record.follower_q_rad) + [
        replay.first_record.follower_gripper_motor_q_rad
    ]
    last_observations = list(replay.last_record.follower_q_rad) + [
        replay.last_record.follower_gripper_motor_q_rad
    ]
    first_actions = list(replay.first_record.action.arm_q) + [
        replay.first_record.action.gripper_q
    ]
    last_actions = list(replay.last_record.action.arm_q) + [
        replay.last_record.action.gripper_q
    ]
    action_net_displacements = [
        last - first for first, last in zip(first_actions, last_actions)
    ]
    observation_net_displacements = [
        last - first
        for first, last in zip(first_observations, last_observations)
    ]
    materially_moving = [
        value >= COMMAND_SIGNIFICANT_RANGE_RAD for value in action_ranges
    ]
    correlations: List[Optional[float]] = []
    tracking_gains: List[Optional[float]] = []
    direction_consistent: List[Optional[bool]] = []
    violation_reasons: Dict[str, List[str]] = {}
    for (
        name,
        commanded,
        action_range,
        observation_range,
        action_m2,
        observation_m2,
        covariance,
        action_net,
        observation_net,
    ) in zip(
        actuator_names,
        materially_moving,
        action_ranges,
        observation_ranges,
        replay.command_response_action_m2,
        replay.command_response_observation_m2,
        replay.command_response_covariance,
        action_net_displacements,
        observation_net_displacements,
    ):
        if action_m2 > 0.0 and observation_m2 > 0.0:
            correlation = covariance / math.sqrt(action_m2 * observation_m2)
            correlation = max(-1.0, min(1.0, correlation))
        else:
            correlation = None
        correlations.append(correlation)
        if not commanded:
            tracking_gains.append(None)
            direction_consistent.append(None)
            continue

        tracking_gain = observation_range / action_range
        tracking_gains.append(tracking_gain)
        direction_is_observable = (
            abs(action_net)
            >= COMMAND_RESPONSE_DIRECTION_MIN_NET_FRACTION * action_range
        )
        if observation_range < COMMAND_RESPONSE_FROZEN_RANGE_RAD:
            direction_ok = False
        elif direction_is_observable:
            direction_ok = action_net * observation_net > 0.0
        else:
            direction_ok = correlation is not None and correlation > 0.0
        direction_consistent.append(direction_ok)

        reasons: List[str] = []
        if observation_range < COMMAND_RESPONSE_FROZEN_RANGE_RAD:
            reasons.append("frozen_or_near_zero_response")
        if not (
            COMMAND_RESPONSE_MIN_TRACKING_GAIN
            <= tracking_gain
            <= COMMAND_RESPONSE_MAX_TRACKING_GAIN
        ):
            reasons.append("tracking_gain_out_of_bounds")
        if correlation is None or correlation < COMMAND_RESPONSE_MIN_CORRELATION:
            reasons.append("correlation_below_minimum")
        if not direction_ok:
            reasons.append("direction_inconsistent")
        if reasons:
            violation_reasons[name] = reasons

    all_zero = all(
        lower == 0.0 and upper == 0.0
        for lower, upper in zip(
            replay.follower_q_min_rad, replay.follower_q_max_rad
        )
    )
    bit_identical = all(value == 0.0 for value in observation_ranges)
    maximum_range = max(observation_ranges)
    command_response_violations = [
        name for name in actuator_names if name in violation_reasons
    ]
    # A non-degenerate observation trajectory is not enough: a stationary arm with
    # ~1e-4 rad sensor noise satisfies the observation-range floor while no real
    # action was commanded. Training requires that at least one action channel was
    # actually driven by a materially-moving command (>= COMMAND_SIGNIFICANT_RANGE_RAD),
    # so an all-zero / noise-only action stream is rejected. (One held joint is
    # still allowed as long as some other channel materially moves.)
    any_command_materially_moving = any(materially_moving)
    trajectory_nondegenerate = (
        not all_zero
        and not bit_identical
        and maximum_range >= TRAJECTORY_MIN_WHOLE_7D_RANGE_RAD
        and any_command_materially_moving
        and not command_response_violations
    )
    return {
        "trajectory_validation_schema": TRAJECTORY_VALIDATION_SCHEMA,
        "trajectory_min_whole_7d_range_rad": (
            TRAJECTORY_MIN_WHOLE_7D_RANGE_RAD
        ),
        "command_significant_range_rad": COMMAND_SIGNIFICANT_RANGE_RAD,
        "command_response_frozen_range_rad": (
            COMMAND_RESPONSE_FROZEN_RANGE_RAD
        ),
        "command_response_min_tracking_gain": (
            COMMAND_RESPONSE_MIN_TRACKING_GAIN
        ),
        "command_response_max_tracking_gain": (
            COMMAND_RESPONSE_MAX_TRACKING_GAIN
        ),
        "command_response_min_correlation": (
            COMMAND_RESPONSE_MIN_CORRELATION
        ),
        "command_response_direction_min_net_fraction": (
            COMMAND_RESPONSE_DIRECTION_MIN_NET_FRACTION
        ),
        "actuator_names": actuator_names,
        "observation_joint_range_rad": observation_joint_ranges,
        "action_joint_range_rad": action_joint_ranges,
        "observation_actuator_range_rad": observation_ranges,
        "action_actuator_range_rad": action_ranges,
        "command_materially_moving": materially_moving,
        "any_command_materially_moving": any_command_materially_moving,
        "command_response_tracking_gain": tracking_gains,
        "command_response_correlation": correlations,
        "command_net_displacement_rad": action_net_displacements,
        "observation_net_displacement_rad": observation_net_displacements,
        "command_response_direction_consistent": direction_consistent,
        "observation_trajectory_all_zero": all_zero,
        "observation_trajectory_bit_identical": bit_identical,
        "observation_trajectory_max_joint_range_rad": maximum_range,
        "trajectory_nondegenerate": trajectory_nondegenerate,
        "command_response_violation_joint_names": command_response_violations,
        "command_response_violation_actuator_names": (
            command_response_violations
        ),
        "command_response_violation_reasons": violation_reasons,
        "command_response_consistent": not command_response_violations,
    }


def _observation_payload(
    observation: InterpolatedFollowerObservation,
) -> Dict[str, object]:
    return {
        "monotonic_ns": observation.monotonic_ns,
        "source_control_tick_before": observation.source_sequence_before,
        "source_control_tick_after": observation.source_sequence_after,
        "follower_q_rad": list(observation.follower_q_rad),
        "follower_qd_rad_s": list(observation.follower_qd_rad_s),
        "follower_gripper_q_rad": observation.follower_gripper_q_rad,
        "follower_gripper_qd_rad_s": observation.follower_gripper_qd_rad_s,
        "follower_gripper_motor_q_rad": (
            observation.follower_gripper_motor_q_rad
        ),
        "follower_gripper_motor_qd_rad_s": (
            observation.follower_gripper_motor_qd_rad_s
        ),
    }


@dataclass(frozen=True)
class _VerifiedExactPartials:
    replay: "ExactCaptureReplayStats"
    hash_inventory: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class ExactCaptureReplayStats:
    envelope_schema: str
    envelope_count: int
    envelope_sequence_first: int
    envelope_sequence_last: int
    control_tick_count: int
    control_tick_first: int
    control_tick_last: int
    observation_monotonic_ns_first: int
    observation_monotonic_ns_last: int
    max_observation_gap_ns: int
    frame_count: int
    peak_buffered_control_ticks: int
    arm_tx_expected_frame_count: int
    arm_tx_full_frame_write_count: int
    arm_tx_failed_frame_write_count: int
    gripper_tx_expected_frame_count: int
    gripper_tx_full_frame_write_count: int
    gripper_tx_failed_frame_write_count: int
    socketcan_tx_enqueue_failure_tick_count: int
    feedback_all_ticks_fresh: bool
    feedback_max_age_ns: int
    feedback_max_last_gap_ns: int
    follower_q_min_rad: Tuple[float, ...]
    follower_q_max_rad: Tuple[float, ...]
    action_q_min_rad: Tuple[float, ...]
    action_q_max_rad: Tuple[float, ...]
    follower_gripper_motor_q_min_rad: float
    follower_gripper_motor_q_max_rad: float
    action_gripper_q_min_rad: float
    action_gripper_q_max_rad: float
    command_response_action_m2: Tuple[float, ...]
    command_response_observation_m2: Tuple[float, ...]
    command_response_covariance: Tuple[float, ...]
    first_record: ExactControlRecord
    last_record: ExactControlRecord
    hash_inventory: Mapping[str, Mapping[str, object]]
    file_identities: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class CapturedExactControlFrameEvidence:
    trajectory: BracketedExactControlTrajectory
    observation_at_nominal: InterpolatedFollowerObservation
    index: Mapping[str, object]


class RobotSessionCapture:
    """Exact left-OpenArm capture: v2 commissioning, v3/v4 production.

    Observations may be interpolated inside an exact tick bracket.  Actions are
    persisted and exposed only at their original 1 kHz control tick.
    """

    def __init__(
        self,
        session_dir: object,
        contract: RobotStateContract,
        *,
        preflight_min_envelopes: int = 11,
        preflight_duration_ns: int = 100_000_000,
    ) -> None:
        if not isinstance(contract, RobotStateContract):
            raise TypeError("contract must be RobotStateContract")
        if (
            contract.side != "left"
            or contract.leader_can_interface != "can1"
            or contract.follower_can_interface != "can3"
        ):
            raise RobotCaptureError(
                "production capture requires the left OpenArm pair can1 leader -> can3 follower"
            )
        if any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (
                contract.controller_source_manifest_sha256,
                contract.follower_config_sha256,
                contract.openarm_sdk_sha256,
            )
        ):
            raise RobotCaptureError(
                "exact capture requires complete source, config, and SDK SHA-256 closure"
            )
        if (
            not _is_int(preflight_min_envelopes)
            or preflight_min_envelopes < 2
            or not _is_int(preflight_duration_ns)
            or preflight_duration_ns <= 0
            or preflight_duration_ns > contract.buffer_duration_ns
        ):
            raise ValueError("v2/v3 preflight bounds are invalid")
        path = Path(session_dir)
        if not path.is_dir():
            raise RobotCaptureError("robot session directory must already exist")

        self.session_dir = path
        self.contract = contract
        self.stream = ExactControlStateStream(contract)
        self.preflight_min_envelopes = preflight_min_envelopes
        self.preflight_duration_ns = preflight_duration_ns
        self._lock = threading.RLock()
        self._recent_envelopes: List[RobotControlEnvelope] = []
        self._last_envelope: Optional[RobotControlEnvelope] = None
        self._capture_first_record: Optional[ExactControlRecord] = None
        self._capture_last_record: Optional[ExactControlRecord] = None
        self._capture_first_envelope_seq: Optional[int] = None
        self._capture_last_envelope_seq: Optional[int] = None
        self._captured_envelope_count = 0
        self._captured_control_tick_count = 0
        self._captured_max_observation_gap_ns = 0
        self._raw_digest = hashlib.sha256()
        self._raw_bytes_written = 0
        self._frame_index_digest = hashlib.sha256()
        self._frame_index_bytes_written = 0
        self._frame_count = 0
        self._last_frame_row: Optional[Dict[str, object]] = None
        self._last_frame_by_camera: Dict[str, Tuple[int, int]] = {}
        self._raw: Optional[IO[bytes]] = None
        self._exact_handle: Optional[IO[str]] = None
        self._exact_writer: Optional[Any] = None
        self._frame_index: Optional[IO[str]] = None
        self._capture_envelope_schema: Optional[str] = None
        self._feedback_start_baseline: Optional[Dict[str, object]] = None
        self._capture_end_monotonic_ns: Optional[int] = None
        self._recording = False
        self._finalized = False
        self._fault_reason = ""

    @property
    def faulted(self) -> bool:
        return bool(self._fault_reason)

    @property
    def fault_reason(self) -> str:
        return self._fault_reason

    @property
    def recording(self) -> bool:
        return self._recording

    def _raise_if_unavailable(self) -> None:
        if self.faulted:
            raise RobotCaptureError(
                f"robot session capture is faulted: {self._fault_reason}"
            )
        if self._finalized:
            raise RobotCaptureError("robot session capture is already finalized")

    def _close_writers(self) -> Optional[Exception]:
        first_error: Optional[Exception] = None
        for handle in (self._raw, self._exact_handle, self._frame_index):
            if handle is None or handle.closed:
                continue
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = exc
            try:
                handle.close()
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = exc
        self._recording = False
        return first_error

    def _abort(self, reason: str) -> RobotCaptureError:
        if not self._fault_reason:
            self._fault_reason = str(reason)
        close_error = self._close_writers()
        try:
            _fsync_directory(self.session_dir)
        except OSError as exc:
            if close_error is None:
                close_error = exc
        if close_error is not None:
            self._fault_reason += f"; partial flush/close failed: {close_error}"
        return RobotCaptureError(self._fault_reason)

    def abort(self, reason: str) -> None:
        with self._lock:
            if self._finalized:
                raise RobotCaptureError("cannot abort a finalized robot capture")
            raise self._abort(reason or "robot capture aborted")

    def ingest_datagram(self, payload: bytes) -> RobotControlEnvelope:
        with self._lock:
            self._raise_if_unavailable()
            if not isinstance(payload, bytes):
                raise self._abort("control envelope datagram must be bytes")
            if b"\n" in payload or b"\r" in payload:
                raise self._abort("control envelope must be one JSONL-safe line")
            try:
                envelope = self.stream.ingest_datagram(payload)
            except RobotTelemetryError as exc:
                raise self._abort(str(exc)) from exc
            self._last_envelope = envelope
            self._recent_envelopes.append(envelope)
            cutoff = (
                envelope.records[-1].observation_monotonic_ns
                - self.contract.buffer_duration_ns
            )
            while (
                self._recent_envelopes
                and self._recent_envelopes[0].records[0].observation_monotonic_ns
                < cutoff
            ):
                self._recent_envelopes.pop(0)

            if self._recording:
                if envelope.schema != self._capture_envelope_schema:
                    raise self._abort(
                        "robot exact-control schema changed during capture"
                    )
                try:
                    assert self._raw is not None
                    assert self._exact_writer is not None
                    assert self._exact_handle is not None
                    raw_record = payload + b"\n"
                    self._raw.write(raw_record)
                    for record in envelope.records:
                        self._exact_writer.writerow(_exact_control_row(record))
                    self._raw.flush()
                    self._exact_handle.flush()
                except Exception as exc:  # noqa: BLE001
                    raise self._abort(
                        f"exact control partial write failed: {exc}"
                    ) from exc
                self._raw_digest.update(raw_record)
                self._raw_bytes_written += len(raw_record)
                if self._capture_first_record is None:
                    self._capture_first_record = envelope.records[0]
                    self._capture_first_envelope_seq = envelope.envelope_seq
                if self._capture_last_record is not None:
                    gap = (
                        envelope.records[0].observation_monotonic_ns
                        - self._capture_last_record.observation_monotonic_ns
                    )
                    self._captured_max_observation_gap_ns = max(
                        self._captured_max_observation_gap_ns, gap
                    )
                for left, right in zip(envelope.records, envelope.records[1:]):
                    self._captured_max_observation_gap_ns = max(
                        self._captured_max_observation_gap_ns,
                        right.observation_monotonic_ns
                        - left.observation_monotonic_ns,
                    )
                self._capture_last_record = envelope.records[-1]
                self._capture_last_envelope_seq = envelope.envelope_seq
                self._captured_envelope_count += 1
                self._captured_control_tick_count += len(envelope.records)
            return envelope

    def start(self, *, now_monotonic_ns: int) -> None:
        with self._lock:
            self._raise_if_unavailable()
            if self._recording:
                raise RobotCaptureError("robot session capture is already recording")
            if not _is_int(now_monotonic_ns) or now_monotonic_ns <= 0:
                raise RobotCaptureError("robot preflight query time is invalid")
            if len(self._recent_envelopes) < self.preflight_min_envelopes:
                raise RobotCaptureError(
                    "exact-control preflight has too few envelopes"
                )
            records = tuple(
                record
                for envelope in self._recent_envelopes
                for record in envelope.records
            )
            latest = records[-1]
            if now_monotonic_ns < latest.observation_monotonic_ns:
                raise RobotCaptureError("robot preflight query clock precedes telemetry")
            if (
                now_monotonic_ns - latest.observation_monotonic_ns
                > self.contract.max_staleness_ns
            ):
                raise RobotCaptureError("exact-control preflight telemetry is stale")
            if (
                latest.observation_monotonic_ns
                - records[0].observation_monotonic_ns
                < self.preflight_duration_ns
            ):
                raise RobotCaptureError(
                    "exact-control preflight duration is too short"
                )
            gaps = [
                right.observation_monotonic_ns - left.observation_monotonic_ns
                for left, right in zip(records, records[1:])
            ]
            if any(gap <= 0 or gap > MAX_CONTROL_TICK_GAP_NS for gap in gaps):
                raise self._abort(
                    "exact-control preflight ticks are not continuous"
                )
            envelope_schemas = {
                envelope.schema for envelope in self._recent_envelopes
            }
            if len(envelope_schemas) != 1:
                raise self._abort(
                    "robot exact-control schema changed during preflight"
                )
            self._capture_envelope_schema = next(iter(envelope_schemas))
            if self._capture_envelope_schema == CONTROL_ENVELOPE_SCHEMA_V4:
                if not latest.feedback_rx_all_fresh:
                    raise RobotCaptureError(
                        "v4 feedback preflight is stale or exceeds the 20 ms gap bound"
                    )
                self._feedback_start_baseline = _feedback_start_baseline_payload(
                    latest, start_monotonic_ns=now_monotonic_ns
                )
            for partial_name, final_name in _EXACT_PARTIAL_TO_FINAL:
                if (self.session_dir / partial_name).exists() or (
                    self.session_dir / final_name
                ).exists():
                    raise self._abort(
                        f"robot capture artifact already exists: {partial_name} or {final_name}"
                    )
            try:
                self._raw = (
                    self.session_dir / CONTROL_ENVELOPES_PARTIAL_NAME
                ).open("xb")
                self._exact_handle = (
                    self.session_dir / EXACT_CONTROL_PARTIAL_NAME
                ).open("x", newline="", encoding="utf-8")
                self._frame_index = (
                    self.session_dir / EXACT_FRAME_INDEX_PARTIAL_NAME
                ).open("x", newline="\n", encoding="utf-8")
                self._exact_writer = csv.writer(
                    self._exact_handle, lineterminator="\n"
                )
                self._exact_writer.writerow(
                    _exact_control_header(
                        self.contract.joint_names,
                        include_tx_status=(
                            self._capture_envelope_schema
                            in {
                                CONTROL_ENVELOPE_SCHEMA_V3,
                                CONTROL_ENVELOPE_SCHEMA_V4,
                            }
                        ),
                        include_feedback_liveness=(
                            self._capture_envelope_schema
                            == CONTROL_ENVELOPE_SCHEMA_V4
                        ),
                    )
                )
                self._raw.flush()
                self._exact_handle.flush()
                self._frame_index.flush()
                _fsync_directory(self.session_dir)
            except Exception as exc:  # noqa: BLE001
                raise self._abort(
                    f"robot capture partial initialization failed: {exc}"
                ) from exc
            self._recording = True

    def record_frame_evidence(
        self,
        *,
        camera_id: str,
        frame_index: int,
        exposure_lower_ns: int,
        exposure_nominal_ns: int,
        exposure_upper_ns: int,
        now_monotonic_ns: int,
        camera_profile_sha256: str,
        timing_artifact_sha256: str,
    ) -> CapturedExactControlFrameEvidence:
        with self._lock:
            self._raise_if_unavailable()
            if not self._recording:
                raise RobotCaptureError("robot session capture is not recording")
            if not isinstance(camera_id, str) or _CAMERA_ID_RE.fullmatch(camera_id) is None:
                raise self._abort("camera ID is invalid")
            if not _is_int(frame_index) or frame_index < 0:
                raise self._abort("camera frame index must be nonnegative")
            previous = self._last_frame_by_camera.get(camera_id)
            if previous is not None and (
                frame_index <= previous[0] or exposure_nominal_ns <= previous[1]
            ):
                raise self._abort("camera frame index and exposure must increase")
            try:
                trajectory = self.stream.query_exposure_interval(
                    exposure_lower_ns=exposure_lower_ns,
                    exposure_nominal_ns=exposure_nominal_ns,
                    exposure_upper_ns=exposure_upper_ns,
                    now_monotonic_ns=now_monotonic_ns,
                    camera_profile_sha256=camera_profile_sha256,
                    timing_artifact_sha256=timing_artifact_sha256,
                )
                nominal = trajectory.sample_observation_at(exposure_nominal_ns)
            except RobotTelemetryError as exc:
                raise self._abort(str(exc)) from exc
            first = self._capture_first_record
            last = self._capture_last_record
            if (
                first is None
                or last is None
                or trajectory.sequence_first < first.control_tick_seq
                or trajectory.sequence_last > last.control_tick_seq
            ):
                raise self._abort(
                    "camera exposure exact control bracket is outside the recorded session"
                )
            source = trajectory.before
            is_v4 = source.schema == EXACT_CONTROL_RECORD_SCHEMA_V4
            has_tx_status = source.schema in {
                EXACT_CONTROL_RECORD_SCHEMA_V3,
                EXACT_CONTROL_RECORD_SCHEMA_V4,
            }
            row: Dict[str, object] = {
                "schema": (
                    EXACT_FRAME_INDEX_SCHEMA_V4_PRODUCTION
                    if is_v4
                    else (
                        EXACT_FRAME_INDEX_SCHEMA_V3_PRODUCTION
                        if has_tx_status
                        else EXACT_FRAME_INDEX_SCHEMA_V2_COMMISSIONING
                    )
                ),
                "camera_id": camera_id,
                "frame_index": frame_index,
                "exposure_lower_ns": exposure_lower_ns,
                "exposure_nominal_ns": exposure_nominal_ns,
                "exposure_upper_ns": exposure_upper_ns,
                "query_now_monotonic_ns": now_monotonic_ns,
                "observation_tick_first": trajectory.sequence_first,
                "observation_tick_last": trajectory.sequence_last,
                "observation_timestamp_first_ns": source.observation_monotonic_ns,
                "observation_timestamp_last_ns": trajectory.after.observation_monotonic_ns,
                "observation_max_gap_ns": trajectory.max_gap_ns,
                "observation_at_nominal": _observation_payload(nominal),
                "nominal_alignment_rule": "interpolate_observation_only_at_exposure_nominal_ns",
                "exact_action_tick_first": trajectory.sequence_first,
                "exact_action_tick_last": trajectory.sequence_last,
                "exact_action_tick_count": len(trajectory.records),
                "exact_action_sha256": _exact_action_sha256(trajectory.records),
                "exact_control_trajectory_sha256": _exact_trajectory_sha256(
                    trajectory.records
                ),
                "action_interpolation": "forbidden",
                "controller_pid": source.controller_pid,
                "controller_start_token": source.controller_start_token,
                "publisher_identity": source.publisher_identity,
                "side": "left",
                "controller_mode": "unilateral",
                "leader_can_interface": "can1",
                "follower_can_interface": "can3",
                "urdf_sha256": source.urdf_sha256,
                "joint_mapper_sha256": source.joint_mapper_sha256,
                "camera_profile_sha256": camera_profile_sha256,
                "timing_artifact_sha256": timing_artifact_sha256,
                "pair_contract_sha256": source.pair_contract_sha256,
                "controller_binary_sha256": source.controller_binary_sha256,
                "controller_source_manifest_sha256": source.controller_source_manifest_sha256,
                "follower_config_sha256": source.follower_config_sha256,
                "openarm_sdk_sha256": source.openarm_sdk_sha256,
                "send_status_observable": source.send_status_observable,
                "send_status_reason": source.send_status_reason,
            }
            if has_tx_status:
                failed_ticks = sum(
                    not record.socketcan_tx_enqueue_success
                    for record in trajectory.records
                )
                row.update(
                    send_status_semantics=SOCKETCAN_TX_ENQUEUE_SEMANTICS,
                    motor_ack_observable=False,
                    motor_ack_reason=MOTOR_ACK_UNOBSERVABLE_REASON,
                    socketcan_tx_enqueue_failure_tick_count=failed_ticks,
                    socketcan_tx_enqueue_success_all_ticks=(
                        failed_ticks == 0
                    ),
                )
            if is_v4:
                row.update(
                    feedback_liveness_observable=True,
                    feedback_liveness_reason=FEEDBACK_LIVENESS_REASON,
                    feedback_liveness_semantics=FEEDBACK_LIVENESS_SEMANTICS,
                    feedback_max_age_ns=FEEDBACK_MAX_AGE_NS,
                    feedback_max_gap_ns=FEEDBACK_MAX_GAP_NS,
                    feedback_all_ticks_fresh=all(
                        record.feedback_rx_all_fresh
                        for record in trajectory.records
                    ),
                    feedback_gap_violation_counts_first=(
                        _feedback_snapshot_payload(trajectory.records[0])[
                            "gap_violation_counts"
                        ]
                    ),
                    feedback_gap_violation_counts_last=(
                        _feedback_snapshot_payload(trajectory.records[-1])[
                            "gap_violation_counts"
                        ]
                    ),
                )
            try:
                assert self._frame_index is not None
                frame_record = json.dumps(
                    row, sort_keys=True, separators=(",", ":"), allow_nan=False
                ) + "\n"
                self._frame_index.write(frame_record)
                self._frame_index.flush()
            except Exception as exc:  # noqa: BLE001
                raise self._abort(f"robot frame-index write failed: {exc}") from exc
            frame_bytes = frame_record.encode("utf-8")
            self._frame_index_digest.update(frame_bytes)
            self._frame_index_bytes_written += len(frame_bytes)
            self._frame_count += 1
            self._last_frame_row = row
            self._last_frame_by_camera[camera_id] = (
                frame_index,
                exposure_nominal_ns,
            )
            return CapturedExactControlFrameEvidence(
                trajectory=trajectory,
                observation_at_nominal=nominal,
                index=MappingProxyType(dict(row)),
            )

    def _verify_partials(self) -> _VerifiedExactPartials:
        raw_path = self.session_dir / CONTROL_ENVELOPES_PARTIAL_NAME
        frame_path = self.session_dir / EXACT_FRAME_INDEX_PARTIAL_NAME
        replay = _stream_validate_capture_data(
            raw_path,
            self.session_dir / EXACT_CONTROL_PARTIAL_NAME,
            frame_path,
            self.contract,
        )
        if replay.hash_inventory[CONTROL_ENVELOPES_NAME] != {
            "sha256": self._raw_digest.hexdigest(),
            "size_bytes": self._raw_bytes_written,
        }:
            raise RobotCaptureError(
                "raw control envelope byte digest does not match the live writer"
            )
        if replay.hash_inventory[EXACT_FRAME_INDEX_NAME] != {
            "sha256": self._frame_index_digest.hexdigest(),
            "size_bytes": self._frame_index_bytes_written,
        }:
            raise RobotCaptureError(
                "robot frame-index byte digest does not match the live writer"
            )
        if replay.envelope_count != self._captured_envelope_count:
            raise RobotCaptureError("raw envelope_count changed after capture")
        if replay.control_tick_count != self._captured_control_tick_count:
            raise RobotCaptureError("raw control_tick_count changed after capture")
        if (
            self._capture_first_record is None
            or self._capture_last_record is None
            or replay.control_tick_first
            != self._capture_first_record.control_tick_seq
            or replay.control_tick_last
            != self._capture_last_record.control_tick_seq
        ):
            raise RobotCaptureError("raw exact control tick bounds changed")
        if (
            replay.max_observation_gap_ns
            != self._captured_max_observation_gap_ns
        ):
            raise RobotCaptureError("raw exact control maximum gap changed")
        if replay.frame_count != self._frame_count:
            raise RobotCaptureError("robot frame-index frame_count mismatch")
        return _VerifiedExactPartials(
            replay=replay,
            hash_inventory=replay.hash_inventory,
        )

    def _write_summary_partial(
        self, verified: _VerifiedExactPartials
    ) -> Dict[str, object]:
        replay = verified.replay
        first = replay.first_record
        is_v3 = replay.envelope_schema == CONTROL_ENVELOPE_SCHEMA_V3
        is_v4 = replay.envelope_schema == CONTROL_ENVELOPE_SCHEMA_V4
        has_tx_status = is_v3 or is_v4
        tx_clean = has_tx_status and (
            replay.arm_tx_failed_frame_write_count == 0
            and replay.gripper_tx_failed_frame_write_count == 0
            and replay.socketcan_tx_enqueue_failure_tick_count == 0
            and replay.arm_tx_expected_frame_count
            == replay.control_tick_count * 7
            and replay.arm_tx_full_frame_write_count
            == replay.control_tick_count * 7
            and replay.gripper_tx_expected_frame_count
            == replay.control_tick_count
            and replay.gripper_tx_full_frame_write_count
            == replay.control_tick_count
        )
        feedback_evaluation: Dict[str, object] = {}
        if is_v4:
            if (
                self._feedback_start_baseline is None
                or self._capture_end_monotonic_ns is None
            ):
                raise RobotCaptureError(
                    "v4 feedback capture boundary receipt is missing"
                )
            feedback_evaluation = _evaluate_v4_feedback(
                self._feedback_start_baseline,
                replay,
                end_monotonic_ns=self._capture_end_monotonic_ns,
            )
            feedback_clean = bool(feedback_evaluation["feedback_clean"])
        else:
            feedback_clean = False
        trajectory_evaluation: Dict[str, object] = (
            _evaluate_training_trajectory(replay, self.contract.joint_names)
            if is_v4
            else {}
        )
        trajectory_clean = (
            bool(trajectory_evaluation["trajectory_nondegenerate"])
            and bool(trajectory_evaluation["command_response_consistent"])
            if is_v4
            else False
        )
        if has_tx_status:
            eligibility_blockers = []
            if not tx_clean:
                eligibility_blockers.append(
                    "robot_socketcan_tx_enqueue_failure"
                )
            if is_v3:
                eligibility_blockers.append(
                    "robot_feedback_liveness_unobservable"
                )
            else:
                if not feedback_evaluation[
                    "feedback_counter_progress_all_motors"
                ]:
                    eligibility_blockers.append(
                        "robot_feedback_counter_no_progress"
                    )
                if not feedback_evaluation["feedback_all_ticks_fresh"]:
                    eligibility_blockers.append("robot_feedback_stale")
                if not feedback_evaluation["feedback_gap_clean"]:
                    eligibility_blockers.append(
                        "robot_feedback_gap_exceeded"
                    )
                if not trajectory_evaluation["trajectory_nondegenerate"]:
                    eligibility_blockers.append(
                        "robot_observation_trajectory_degenerate"
                    )
                if not trajectory_evaluation["command_response_consistent"]:
                    eligibility_blockers.append(
                        "robot_command_response_inconsistent"
                    )
        else:
            eligibility_blockers = [
                "robot_action_send_status_unobservable"
            ]
        summary: Dict[str, object] = {
            "schema": (
                EXACT_CAPTURE_SUMMARY_SCHEMA_V4_PRODUCTION
                if is_v4
                else (
                    EXACT_CAPTURE_SUMMARY_SCHEMA_V3_PRODUCTION
                    if is_v3
                    else EXACT_CAPTURE_SUMMARY_SCHEMA_V2_COMMISSIONING
                )
            ),
            "status": (
                (
                    "exact_control_capture_validated_feedback_liveness"
                    if tx_clean and feedback_clean
                    else "exact_control_capture_validated_feedback_liveness_failure"
                )
                if is_v4
                else (
                    "exact_control_capture_validated_socketcan_tx_enqueue"
                    if tx_clean
                    else (
                        "exact_control_capture_validated_socketcan_tx_enqueue_failure"
                        if is_v3
                        else "exact_control_capture_validated_send_status_unobservable"
                    )
                )
            ),
            "side": "left",
            "controller_mode": "unilateral",
            "leader_can_interface": "can1",
            "follower_can_interface": "can3",
            "joint_names": list(self.contract.joint_names),
            "envelope_count": replay.envelope_count,
            "envelope_sequence_first": replay.envelope_sequence_first,
            "envelope_sequence_last": replay.envelope_sequence_last,
            "control_tick_count": replay.control_tick_count,
            "control_tick_first": replay.control_tick_first,
            "control_tick_last": replay.control_tick_last,
            "observation_monotonic_ns_first": replay.observation_monotonic_ns_first,
            "observation_monotonic_ns_last": replay.observation_monotonic_ns_last,
            "max_observation_gap_ns": replay.max_observation_gap_ns,
            "frame_count": replay.frame_count,
            "controller_pid": first.controller_pid,
            "controller_start_token": first.controller_start_token,
            "publisher_identity": first.publisher_identity,
            "urdf_sha256": self.contract.urdf_sha256,
            "joint_mapper_sha256": self.contract.joint_mapper_sha256,
            "camera_profile_sha256": self.contract.camera_profile_sha256,
            "timing_artifact_sha256": self.contract.timing_artifact_sha256,
            "pair_contract_sha256": self.contract.pair_contract_sha256,
            "controller_binary_sha256": self.contract.controller_binary_sha256,
            "controller_source_manifest_sha256": (
                self.contract.controller_source_manifest_sha256
            ),
            "follower_config_sha256": self.contract.follower_config_sha256,
            "openarm_sdk_sha256": self.contract.openarm_sdk_sha256,
            "joint_position_limits_rad": [
                [float(lower), float(upper)]
                for lower, upper in self.contract.joint_position_limits_rad
            ],
            "max_abs_joint_velocity_rad_s": float(
                self.contract.max_abs_joint_velocity_rad_s
            ),
            "gripper_motor_position_limits_rad": [
                float(value)
                for value in self.contract.gripper_motor_position_limits_rad
            ],
            "buffer_duration_ns": self.contract.buffer_duration_ns,
            "max_packets": self.contract.max_packets,
            "max_gap_ns": self.contract.max_gap_ns,
            "max_staleness_ns": self.contract.max_staleness_ns,
            "max_exposure_interval_ns": (
                self.contract.max_exposure_interval_ns
            ),
            "publisher_nominal_hz": 100,
            "control_nominal_hz": 1000,
            "ring_overflow_count": 0,
            "publisher_rejected_record_count": 0,
            "socket_send_failure_count": 0,
            "send_status_observable": first.send_status_observable,
            "send_status_reason": first.send_status_reason,
            "action_parameters_exact": True,
            "action_semantics": "exact_mitparam_consumed_at_control_tick",
            "action_interpolation": "forbidden",
            "observation_interpolation": "linear_within_exact_tick_bracket",
            "nominal_alignment_rule": (
                "interpolate_observation_only_at_exposure_nominal_ns"
            ),
            "training_eligible": tx_clean and feedback_clean and trajectory_clean,
            "eligibility_blockers": eligibility_blockers,
            "hash_inventory": dict(verified.hash_inventory),
        }
        if has_tx_status:
            summary.update(
                send_status_semantics=SOCKETCAN_TX_ENQUEUE_SEMANTICS,
                motor_ack_observable=False,
                motor_ack_reason=MOTOR_ACK_UNOBSERVABLE_REASON,
                arm_tx_expected_frame_count=(
                    replay.arm_tx_expected_frame_count
                ),
                arm_tx_full_frame_write_count=(
                    replay.arm_tx_full_frame_write_count
                ),
                arm_tx_failed_frame_write_count=(
                    replay.arm_tx_failed_frame_write_count
                ),
                gripper_tx_expected_frame_count=(
                    replay.gripper_tx_expected_frame_count
                ),
                gripper_tx_full_frame_write_count=(
                    replay.gripper_tx_full_frame_write_count
                ),
                gripper_tx_failed_frame_write_count=(
                    replay.gripper_tx_failed_frame_write_count
                ),
                socketcan_tx_enqueue_failure_tick_count=(
                    replay.socketcan_tx_enqueue_failure_tick_count
                ),
            )
        if is_v3:
            summary.update(
                feedback_liveness_observable=False,
                feedback_liveness_reason=(
                    FEEDBACK_LIVENESS_UNOBSERVABLE_REASON
                ),
            )
        if is_v4:
            summary.update(
                feedback_liveness_observable=True,
                feedback_liveness_reason=FEEDBACK_LIVENESS_REASON,
                feedback_liveness_semantics=FEEDBACK_LIVENESS_SEMANTICS,
                feedback_max_age_ns=FEEDBACK_MAX_AGE_NS,
                feedback_max_gap_ns=FEEDBACK_MAX_GAP_NS,
                **feedback_evaluation,
                **trajectory_evaluation,
            )
        path = self.session_dir / EXACT_SUMMARY_PARTIAL_NAME
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        summary, indent=2, sort_keys=True, allow_nan=False
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:  # noqa: BLE001
            raise RobotCaptureError(
                f"robot capture summary write failed: {exc}"
            ) from exc
        observed = _read_strict_json_object(path, "robot capture summary")
        if observed != summary:
            raise RobotCaptureError("robot capture summary changed after write")
        return summary

    def _promote_partials(
        self, expected_inventory: Mapping[str, object]
    ) -> None:
        for _, final_name in _EXACT_PARTIAL_TO_FINAL:
            if (self.session_dir / final_name).exists():
                raise RobotCaptureError(
                    f"robot final artifact already exists: {final_name}"
                )
        linked: List[Tuple[Path, Path]] = []
        summary_promoted = False
        try:
            for partial_name, final_name in _EXACT_PARTIAL_TO_FINAL[:-1]:
                partial = self.session_dir / partial_name
                final = self.session_dir / final_name
                os.link(partial, final)
                linked.append((partial, final))
                actual = {
                    "sha256": _sha256(final),
                    "size_bytes": final.stat().st_size,
                }
                if expected_inventory.get(final_name) != actual:
                    raise RobotCaptureError(
                        f"robot capture pre-commit hash mismatch for {final_name}"
                    )
            _fsync_directory(self.session_dir)
            summary_partial = self.session_dir / EXACT_SUMMARY_PARTIAL_NAME
            summary_final = self.session_dir / EXACT_SUMMARY_NAME
            os.replace(summary_partial, summary_final)
            summary_promoted = True
            _fsync_directory(self.session_dir)
            for partial, _ in linked:
                partial.unlink()
            _fsync_directory(self.session_dir)
        except Exception as exc:  # noqa: BLE001
            rollback_errors: List[str] = []
            summary_partial = self.session_dir / EXACT_SUMMARY_PARTIAL_NAME
            summary_final = self.session_dir / EXACT_SUMMARY_NAME
            if summary_promoted and summary_final.exists():
                try:
                    os.replace(summary_final, summary_partial)
                except OSError as rollback_exc:
                    rollback_errors.append(str(rollback_exc))
            for partial, final in reversed(linked):
                try:
                    if not partial.exists() and final.exists():
                        os.link(final, partial)
                    if final.exists():
                        final.unlink()
                except OSError as rollback_exc:
                    rollback_errors.append(str(rollback_exc))
            try:
                _fsync_directory(self.session_dir)
            except OSError as sync_exc:
                rollback_errors.append(str(sync_exc))
            suffix = (
                f"; rollback errors={rollback_errors}"
                if rollback_errors
                else ""
            )
            raise RobotCaptureError(
                f"robot capture atomic promotion failed: {exc}{suffix}"
            ) from exc

    def finish(self, *, now_monotonic_ns: int) -> Mapping[str, object]:
        with self._lock:
            self._raise_if_unavailable()
            if not self._recording:
                raise RobotCaptureError("robot session capture is not recording")
            if self._last_frame_row is None:
                raise self._abort(
                    "robot capture cannot finish without camera frame evidence"
                )
            if (
                not _is_int(now_monotonic_ns)
                or now_monotonic_ns
                < int(self._last_frame_row["exposure_upper_ns"])
            ):
                raise self._abort(
                    "End must not precede the last camera exposure upper bound"
                )
            try:
                final_bracket = self.stream.query_exposure_interval(
                    exposure_lower_ns=int(
                        self._last_frame_row["exposure_lower_ns"]
                    ),
                    exposure_nominal_ns=int(
                        self._last_frame_row["exposure_nominal_ns"]
                    ),
                    exposure_upper_ns=int(
                        self._last_frame_row["exposure_upper_ns"]
                    ),
                    now_monotonic_ns=now_monotonic_ns,
                    camera_profile_sha256=str(
                        self._last_frame_row["camera_profile_sha256"]
                    ),
                    timing_artifact_sha256=str(
                        self._last_frame_row["timing_artifact_sha256"]
                    ),
                )
            except RobotTelemetryError as exc:
                raise self._abort(
                    f"End has no final exact control bracket: {exc}"
                ) from exc
            if (
                final_bracket.sequence_first
                != self._last_frame_row["exact_action_tick_first"]
                or final_bracket.sequence_last
                != self._last_frame_row["exact_action_tick_last"]
                or _exact_action_sha256(final_bracket.records)
                != self._last_frame_row["exact_action_sha256"]
            ):
                raise self._abort("End final exact control bracket changed")
            self._capture_end_monotonic_ns = now_monotonic_ns
            close_error = self._close_writers()
            if close_error is not None:
                raise self._abort(
                    f"robot capture partial fsync/close failed: {close_error}"
                )
            try:
                _fsync_directory(self.session_dir)
                verified = self._verify_partials()
                summary = self._write_summary_partial(verified)
                self._promote_partials(summary["hash_inventory"])
            except RobotCaptureError as exc:
                raise self._abort(str(exc)) from exc
            except (OSError, ValueError, TypeError) as exc:
                raise self._abort(
                    f"robot capture final verification failed: {exc}"
                ) from exc
            self._finalized = True
            return summary


_MAX_STREAMED_FRAME_ACTION_TICKS = 128
_MAX_STREAMED_CAPTURE_CAMERAS = 16
_MAX_FRAME_INDEX_LINE_BYTES = 65_536


class _BoundedExactControlWindow:
    """Small tick-indexed window shared by production replay and stress tests."""

    def __init__(self, *, max_ticks: int) -> None:
        if not _is_int(max_ticks) or max_ticks <= 0:
            raise ValueError("maximum buffered control ticks must be positive")
        self.max_ticks = max_ticks
        self._items: deque[Any] = deque()
        self.peak_buffered_ticks = 0

    @staticmethod
    def _tick(item: Any) -> int:
        if _is_int(item):
            return item
        tick = getattr(item, "control_tick_seq", None)
        if not _is_int(tick):
            raise RobotCaptureError("buffered control record has no exact tick")
        return tick

    def append(self, item: Any) -> None:
        tick = self._tick(item)
        if self._items and tick <= self._tick(self._items[-1]):
            raise RobotCaptureError("buffered exact control ticks are not increasing")
        self._items.append(item)
        self.peak_buffered_ticks = max(
            self.peak_buffered_ticks, len(self._items)
        )

    def discard_before(self, tick: int) -> None:
        if not _is_int(tick):
            raise RobotCaptureError("buffer trim tick must be an integer")
        while self._items and self._tick(self._items[0]) < tick:
            self._items.popleft()

    def bracket(self, first_tick: int, last_tick: int) -> Tuple[Any, ...]:
        return tuple(
            item
            for item in self._items
            if first_tick <= self._tick(item) <= last_tick
        )

    def clear(self) -> None:
        self._items.clear()

    def require_within_bound(self) -> None:
        if len(self._items) > self.max_ticks:
            raise RobotCaptureError(
                "streamed exact control frame buffer exceeded bound"
            )

    def __len__(self) -> int:
        return len(self._items)


def _validate_streamed_frame_shape(
    row: object,
    *,
    contract: RobotStateContract,
    per_camera: Dict[str, Tuple[int, int]],
    previous_first_tick: Optional[int],
) -> Dict[str, object]:
    if not isinstance(row, dict):
        raise RobotCaptureError("exact robot frame index must be an object")
    frame_schema = row.get("schema")
    if frame_schema == EXACT_FRAME_INDEX_SCHEMA_V2_COMMISSIONING:
        expected_fields = _EXACT_FRAME_FIELDS
    elif frame_schema == EXACT_FRAME_INDEX_SCHEMA_V3_PRODUCTION:
        expected_fields = _EXACT_FRAME_FIELDS_V3
    elif frame_schema == EXACT_FRAME_INDEX_SCHEMA_V4_PRODUCTION:
        expected_fields = _EXACT_FRAME_FIELDS_V4
    else:
        raise RobotCaptureError("exact robot frame index schema mismatch")
    if frozenset(row) != expected_fields:
        raise RobotCaptureError("exact robot frame index field set mismatch")
    camera_id = row["camera_id"]
    frame_index = row["frame_index"]
    if (
        not isinstance(camera_id, str)
        or _CAMERA_ID_RE.fullmatch(camera_id) is None
        or not _is_int(frame_index)
        or frame_index < 0
    ):
        raise RobotCaptureError("exact robot frame camera identity is invalid")
    lower = row["exposure_lower_ns"]
    nominal = row["exposure_nominal_ns"]
    upper = row["exposure_upper_ns"]
    query_now = row["query_now_monotonic_ns"]
    if any(
        not _is_int(value) for value in (lower, nominal, upper, query_now)
    ) or not 0 < lower <= nominal <= upper <= query_now:
        raise RobotCaptureError("exact robot frame exposure ordering is invalid")
    if upper - lower > contract.max_exposure_interval_ns:
        raise RobotCaptureError("exact robot frame exposure interval is too wide")
    first_tick = row["exact_action_tick_first"]
    last_tick = row["exact_action_tick_last"]
    tick_count = row["exact_action_tick_count"]
    if (
        not _is_int(first_tick)
        or not _is_int(last_tick)
        or not _is_int(tick_count)
        or first_tick > last_tick
        or tick_count != last_tick - first_tick + 1
        or tick_count > _MAX_STREAMED_FRAME_ACTION_TICKS
    ):
        raise RobotCaptureError("exact action tick range is invalid or unbounded")
    if previous_first_tick is not None and first_tick < previous_first_tick:
        raise RobotCaptureError(
            "exact robot frame rows must be globally time ordered"
        )
    if frame_schema in {
        EXACT_FRAME_INDEX_SCHEMA_V3_PRODUCTION,
        EXACT_FRAME_INDEX_SCHEMA_V4_PRODUCTION,
    }:
        failure_count = row["socketcan_tx_enqueue_failure_tick_count"]
        success = row["socketcan_tx_enqueue_success_all_ticks"]
        if (
            row["send_status_observable"] is not True
            or row["send_status_reason"] != SOCKETCAN_TX_STATUS_REASON
            or row["send_status_semantics"] != SOCKETCAN_TX_ENQUEUE_SEMANTICS
            or row["motor_ack_observable"] is not False
            or row["motor_ack_reason"] != MOTOR_ACK_UNOBSERVABLE_REASON
        ):
            raise RobotCaptureError(
                "exact robot frame v3 send-status provenance mismatch"
            )
        if (
            not _is_int(failure_count)
            or not 0 <= failure_count <= tick_count
            or not isinstance(success, bool)
            or success is not (failure_count == 0)
        ):
            raise RobotCaptureError(
                "exact robot frame v3 failure count is inconsistent"
            )
    if frame_schema == EXACT_FRAME_INDEX_SCHEMA_V4_PRODUCTION:
        first_counts = row["feedback_gap_violation_counts_first"]
        last_counts = row["feedback_gap_violation_counts_last"]
        if (
            row["feedback_liveness_observable"] is not True
            or row["feedback_liveness_reason"] != FEEDBACK_LIVENESS_REASON
            or row["feedback_liveness_semantics"]
            != FEEDBACK_LIVENESS_SEMANTICS
            or row["feedback_max_age_ns"] != FEEDBACK_MAX_AGE_NS
            or row["feedback_max_gap_ns"] != FEEDBACK_MAX_GAP_NS
            or not isinstance(row["feedback_all_ticks_fresh"], bool)
            or not isinstance(first_counts, list)
            or not isinstance(last_counts, list)
            or len(first_counts) != 8
            or len(last_counts) != 8
            or any(
                not _is_int(value) or value < 0
                for value in first_counts + last_counts
            )
        ):
            raise RobotCaptureError(
                "exact robot frame v4 feedback provenance is invalid"
            )
    previous = per_camera.get(camera_id)
    if previous is not None and (
        frame_index <= previous[0] or nominal <= previous[1]
    ):
        raise RobotCaptureError("exact robot frame index is not increasing")
    if camera_id not in per_camera and len(per_camera) >= _MAX_STREAMED_CAPTURE_CAMERAS:
        raise RobotCaptureError("exact robot frame index has too many cameras")
    per_camera[camera_id] = (frame_index, nominal)
    return row


def _validate_streamed_frame_bracket(
    row: Mapping[str, object],
    bracket: Sequence[ExactControlRecord],
    contract: RobotStateContract,
) -> None:
    first_tick = int(row["exact_action_tick_first"])
    last_tick = int(row["exact_action_tick_last"])
    if (
        len(bracket) != last_tick - first_tick + 1
        or bracket[0].control_tick_seq != first_tick
        or bracket[-1].control_tick_seq != last_tick
        or any(
            right.control_tick_seq != left.control_tick_seq + 1
            for left, right in zip(bracket, bracket[1:])
        )
    ):
        raise RobotCaptureError("exact action tick range is incomplete")
    lower = int(row["exposure_lower_ns"])
    nominal = int(row["exposure_nominal_ns"])
    upper = int(row["exposure_upper_ns"])
    trajectory = BracketedExactControlTrajectory(
        exposure_lower_ns=lower,
        exposure_nominal_ns=nominal,
        exposure_upper_ns=upper,
        records=tuple(bracket),
    )
    if (
        bracket[0].observation_monotonic_ns > lower
        or bracket[-1].observation_monotonic_ns < upper
        or row["observation_tick_first"] != first_tick
        or row["observation_tick_last"] != last_tick
        or row["observation_timestamp_first_ns"]
        != bracket[0].observation_monotonic_ns
        or row["observation_timestamp_last_ns"]
        != bracket[-1].observation_monotonic_ns
        or row["observation_max_gap_ns"] != trajectory.max_gap_ns
    ):
        raise RobotCaptureError(
            "exact robot trajectory does not bracket the exposure"
        )
    if row["exact_action_sha256"] != _exact_action_sha256(bracket):
        raise RobotCaptureError("exact action range hash mismatch")
    if row["exact_control_trajectory_sha256"] != _exact_trajectory_sha256(
        bracket
    ):
        raise RobotCaptureError("exact control trajectory hash mismatch")
    try:
        observation = _observation_payload(
            trajectory.sample_observation_at(nominal)
        )
    except RobotTelemetryError as exc:
        raise RobotCaptureError(
            "nominal observation cannot be interpolated inside bracket"
        ) from exc
    if row["observation_at_nominal"] != observation:
        raise RobotCaptureError("nominal observation interpolation mismatch")
    first = bracket[0]
    identity = {
        "nominal_alignment_rule": "interpolate_observation_only_at_exposure_nominal_ns",
        "action_interpolation": "forbidden",
        "controller_pid": first.controller_pid,
        "controller_start_token": first.controller_start_token,
        "publisher_identity": first.publisher_identity,
        "side": "left",
        "controller_mode": "unilateral",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "urdf_sha256": contract.urdf_sha256,
        "joint_mapper_sha256": contract.joint_mapper_sha256,
        "camera_profile_sha256": contract.camera_profile_sha256,
        "timing_artifact_sha256": contract.timing_artifact_sha256,
        "pair_contract_sha256": contract.pair_contract_sha256,
        "controller_binary_sha256": contract.controller_binary_sha256,
        "controller_source_manifest_sha256": contract.controller_source_manifest_sha256,
        "follower_config_sha256": contract.follower_config_sha256,
        "openarm_sdk_sha256": contract.openarm_sdk_sha256,
        "send_status_observable": first.send_status_observable,
        "send_status_reason": first.send_status_reason,
    }
    if first.schema in {
        EXACT_CONTROL_RECORD_SCHEMA_V3,
        EXACT_CONTROL_RECORD_SCHEMA_V4,
    }:
        failed_ticks = sum(
            not record.socketcan_tx_enqueue_success for record in bracket
        )
        identity.update(
            send_status_semantics=SOCKETCAN_TX_ENQUEUE_SEMANTICS,
            motor_ack_observable=False,
            motor_ack_reason=MOTOR_ACK_UNOBSERVABLE_REASON,
            socketcan_tx_enqueue_failure_tick_count=failed_ticks,
            socketcan_tx_enqueue_success_all_ticks=(failed_ticks == 0),
        )
    if first.schema == EXACT_CONTROL_RECORD_SCHEMA_V4:
        identity.update(
            feedback_liveness_observable=True,
            feedback_liveness_reason=FEEDBACK_LIVENESS_REASON,
            feedback_liveness_semantics=FEEDBACK_LIVENESS_SEMANTICS,
            feedback_max_age_ns=FEEDBACK_MAX_AGE_NS,
            feedback_max_gap_ns=FEEDBACK_MAX_GAP_NS,
            feedback_all_ticks_fresh=all(
                record.feedback_rx_all_fresh for record in bracket
            ),
            feedback_gap_violation_counts_first=(
                _feedback_snapshot_payload(bracket[0])[
                    "gap_violation_counts"
                ]
            ),
            feedback_gap_violation_counts_last=(
                _feedback_snapshot_payload(bracket[-1])[
                    "gap_violation_counts"
                ]
            ),
        )
    expected_frame_schema = (
        EXACT_FRAME_INDEX_SCHEMA_V4_PRODUCTION
        if first.schema == EXACT_CONTROL_RECORD_SCHEMA_V4
        else (
            EXACT_FRAME_INDEX_SCHEMA_V3_PRODUCTION
            if first.schema == EXACT_CONTROL_RECORD_SCHEMA_V3
            else EXACT_FRAME_INDEX_SCHEMA_V2_COMMISSIONING
        )
    )
    if row["schema"] != expected_frame_schema:
        raise RobotCaptureError(
            "exact robot frame schema does not match control records"
        )
    if any(row[field] != value for field, value in identity.items()):
        raise RobotCaptureError("exact robot frame provenance mismatch")


def _open_regular_nofollow(path: Path, label: str) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RobotCaptureError(
            f"{label} no-follow open failed; symlink artifacts are forbidden"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RobotCaptureError(f"{label} must be a regular file")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _own_regular_descriptor(source: object, label: str) -> int:
    """Return a validator-owned descriptor without reopening a pinned source."""

    if _is_int(source):
        if source < 0:
            raise RobotCaptureError(f"{label} descriptor is invalid")
        try:
            descriptor = os.dup(source)
        except OSError as exc:
            raise RobotCaptureError(f"{label} descriptor duplication failed") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RobotCaptureError(f"{label} must be a regular file")
        except Exception:
            os.close(descriptor)
            raise
        return descriptor
    return _open_regular_nofollow(Path(source), label)


def _hash_stable_descriptor(
    descriptor: int, label: str
) -> Tuple[Mapping[str, object], Mapping[str, int]]:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
        size += len(block)
    after = os.fstat(descriptor)
    identity = {
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "mode": int(before.st_mode),
        "size_bytes": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
        "ctime_ns": int(before.st_ctime_ns),
    }
    after_identity = {
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "mode": int(after.st_mode),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
    }
    if identity != after_identity or size != before.st_size:
        raise RobotCaptureError(f"{label} changed while hashing")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return (
        {"sha256": digest.hexdigest(), "size_bytes": size},
        identity,
    )


def _stream_validate_capture_data(
    raw_path: object,
    csv_path: object,
    frame_path: object,
    contract: RobotStateContract,
) -> ExactCaptureReplayStats:
    """Lockstep replay with memory bounded by stream and one exposure."""

    stream = ExactControlStateStream(contract)
    window = _BoundedExactControlWindow(
        max_ticks=_MAX_STREAMED_FRAME_ACTION_TICKS
    )
    per_camera: Dict[str, Tuple[int, int]] = {}
    previous_frame_first_tick: Optional[int] = None
    frame_line_index = 0
    frame_count = 0
    peak_buffered = 0
    envelope_count = 0
    envelope_schema: Optional[str] = None
    envelope_first: Optional[int] = None
    envelope_last: Optional[int] = None
    control_count = 0
    first_record: Optional[ExactControlRecord] = None
    last_record: Optional[ExactControlRecord] = None
    max_gap = 0
    arm_tx_expected = 0
    arm_tx_full = 0
    arm_tx_failed = 0
    gripper_tx_expected = 0
    gripper_tx_full = 0
    gripper_tx_failed = 0
    tx_failure_ticks = 0
    feedback_all_ticks_fresh = True
    feedback_max_age_ns = 0
    feedback_max_last_gap_ns = 0
    follower_q_min = [math.inf] * 7
    follower_q_max = [-math.inf] * 7
    action_q_min = [math.inf] * 7
    action_q_max = [-math.inf] * 7
    follower_gripper_motor_q_min = math.inf
    follower_gripper_motor_q_max = -math.inf
    action_gripper_q_min = math.inf
    action_gripper_q_max = -math.inf
    response_action_mean = [0.0] * 8
    response_observation_mean = [0.0] * 8
    response_action_m2 = [0.0] * 8
    response_observation_m2 = [0.0] * 8
    response_covariance = [0.0] * 8

    with ExitStack() as stack:
        raw_descriptor = _own_regular_descriptor(
            raw_path, "raw control envelopes"
        )
        stack.callback(os.close, raw_descriptor)
        csv_descriptor = _own_regular_descriptor(csv_path, "exact control CSV")
        stack.callback(os.close, csv_descriptor)
        frame_descriptor = _own_regular_descriptor(
            frame_path, "exact robot frame index"
        )
        stack.callback(os.close, frame_descriptor)
        raw_receipt, raw_identity = _hash_stable_descriptor(
            raw_descriptor, "raw control envelopes"
        )
        csv_receipt, csv_identity = _hash_stable_descriptor(
            csv_descriptor, "exact control CSV"
        )
        frame_receipt, frame_identity = _hash_stable_descriptor(
            frame_descriptor, "exact robot frame index"
        )
        raw_handle = stack.enter_context(
            os.fdopen(os.dup(raw_descriptor), "rb")
        )
        csv_handle = stack.enter_context(
            os.fdopen(
                os.dup(csv_descriptor),
                "r",
                newline="",
                encoding="utf-8",
            )
        )
        frame_handle = stack.enter_context(
            os.fdopen(os.dup(frame_descriptor), "rb")
        )
        csv_reader = csv.reader(csv_handle)
        try:
            header = next(csv_reader)
        except (StopIteration, csv.Error) as exc:
            raise RobotCaptureError("exact control CSV is empty") from exc
        valid_headers = (
            _exact_control_header(contract.joint_names),
            _exact_control_header(
                contract.joint_names, include_tx_status=True
            ),
            _exact_control_header(
                contract.joint_names,
                include_tx_status=True,
                include_feedback_liveness=True,
            ),
        )
        if header not in valid_headers:
            raise RobotCaptureError(
                "exact control CSV header violates v2/v3/v4 contract"
            )
        if any("follower_command" in field for field in header):
            raise RobotCaptureError(
                "v1 copied follower_command is forbidden as action"
            )

        def read_frame() -> Optional[Dict[str, object]]:
            nonlocal frame_line_index, previous_frame_first_tick
            line = frame_handle.readline(_MAX_FRAME_INDEX_LINE_BYTES + 1)
            if not line:
                return None
            if len(line) > _MAX_FRAME_INDEX_LINE_BYTES or not line.endswith(b"\n"):
                raise RobotCaptureError("exact robot frame index line is unbounded")
            if b"\r" in line:
                raise RobotCaptureError("exact robot frame index framing is invalid")
            try:
                value = json.loads(
                    line[:-1].decode("utf-8"),
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
            except RobotCaptureError:
                raise
            except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise RobotCaptureError(
                    f"exact robot frame index row {frame_line_index} is invalid"
                ) from exc
            row = _validate_streamed_frame_shape(
                value,
                contract=contract,
                per_camera=per_camera,
                previous_first_tick=previous_frame_first_tick,
            )
            previous_frame_first_tick = int(row["exact_action_tick_first"])
            frame_line_index += 1
            return row

        next_frame = read_frame()
        while True:
            line = raw_handle.readline(131_074)
            if not line:
                break
            if len(line) > 131_073 or not line.endswith(b"\n") or b"\r" in line:
                raise RobotCaptureError("raw control envelope framing is invalid")
            payload = line[:-1]
            if not payload:
                raise RobotCaptureError("raw control envelope row is empty")
            try:
                envelope = stream.ingest_datagram(payload)
            except RobotTelemetryError as exc:
                raise RobotCaptureError(
                    f"raw control envelope violates v2/v3 schema: {exc}"
                ) from exc
            envelope_count += 1
            if envelope_schema is None:
                envelope_schema = envelope.schema
            elif envelope.schema != envelope_schema:
                raise RobotCaptureError(
                    "raw exact-control envelope schema changed"
                )
            if envelope_first is None:
                envelope_first = envelope.envelope_seq
            envelope_last = envelope.envelope_seq
            for record in envelope.records:
                try:
                    csv_row = next(csv_reader)
                except (StopIteration, csv.Error) as exc:
                    raise RobotCaptureError(
                        "exact control CSV ended before raw exact ticks"
                    ) from exc
                if csv_row != _exact_control_row(record):
                    raise RobotCaptureError(
                        "exact control CSV must contain every exact action tick once"
                    )
                if first_record is None:
                    first_record = record
                if last_record is not None:
                    gap = (
                        record.observation_monotonic_ns
                        - last_record.observation_monotonic_ns
                    )
                    if gap <= 0 or gap > MAX_CONTROL_TICK_GAP_NS:
                        raise RobotCaptureError(
                            "raw exact control gap exceeds the hard tick-gap bound"
                        )
                    max_gap = max(max_gap, gap)
                last_record = record
                control_count += 1
                arm_tx_expected += record.arm_tx_expected_frame_count
                arm_tx_full += record.arm_tx_full_frame_write_count
                arm_tx_failed += record.arm_tx_failed_frame_write_count
                gripper_tx_expected += record.gripper_tx_expected_frame_count
                gripper_tx_full += record.gripper_tx_full_frame_write_count
                gripper_tx_failed += record.gripper_tx_failed_frame_write_count
                tx_failure_ticks += int(
                    record.send_status_observable
                    and not record.socketcan_tx_enqueue_success
                )
                for index, value in enumerate(record.follower_q_rad):
                    follower_q_min[index] = min(follower_q_min[index], value)
                    follower_q_max[index] = max(follower_q_max[index], value)
                for index, value in enumerate(record.action.arm_q):
                    action_q_min[index] = min(action_q_min[index], value)
                    action_q_max[index] = max(action_q_max[index], value)
                follower_gripper_motor_q_min = min(
                    follower_gripper_motor_q_min,
                    record.follower_gripper_motor_q_rad,
                )
                follower_gripper_motor_q_max = max(
                    follower_gripper_motor_q_max,
                    record.follower_gripper_motor_q_rad,
                )
                action_gripper_q_min = min(
                    action_gripper_q_min, record.action.gripper_q
                )
                action_gripper_q_max = max(
                    action_gripper_q_max, record.action.gripper_q
                )
                response_actions = tuple(record.action.arm_q) + (
                    record.action.gripper_q,
                )
                response_observations = tuple(record.follower_q_rad) + (
                    record.follower_gripper_motor_q_rad,
                )
                for index, (action_value, observation_value) in enumerate(
                    zip(response_actions, response_observations)
                ):
                    action_delta = (
                        action_value - response_action_mean[index]
                    )
                    response_action_mean[index] += action_delta / control_count
                    observation_delta = (
                        observation_value - response_observation_mean[index]
                    )
                    response_observation_mean[index] += (
                        observation_delta / control_count
                    )
                    response_action_m2[index] += action_delta * (
                        action_value - response_action_mean[index]
                    )
                    response_observation_m2[index] += observation_delta * (
                        observation_value - response_observation_mean[index]
                    )
                    response_covariance[index] += action_delta * (
                        observation_value - response_observation_mean[index]
                    )
                if record.schema == EXACT_CONTROL_RECORD_SCHEMA_V4:
                    feedback_all_ticks_fresh = (
                        feedback_all_ticks_fresh
                        and record.feedback_rx_all_fresh
                    )
                    feedback_timestamps = (
                        record.follower_arm_state_rx_monotonic_ns
                        + (record.follower_gripper_state_rx_monotonic_ns,)
                    )
                    feedback_last_gaps = (
                        record.follower_arm_state_rx_last_gap_ns
                        + (record.follower_gripper_state_rx_last_gap_ns,)
                    )
                    feedback_max_age_ns = max(
                        feedback_max_age_ns,
                        *(record.feedback_snapshot_monotonic_ns - timestamp
                          for timestamp in feedback_timestamps),
                    )
                    feedback_max_last_gap_ns = max(
                        feedback_max_last_gap_ns, *feedback_last_gaps
                    )
                if (
                    next_frame is not None
                    and record.control_tick_seq
                    >= int(next_frame["exact_action_tick_first"])
                ):
                    window.append(record)

                while (
                    next_frame is not None
                    and record.control_tick_seq
                    >= int(next_frame["exact_action_tick_last"])
                ):
                    first_tick = int(next_frame["exact_action_tick_first"])
                    last_tick = int(next_frame["exact_action_tick_last"])
                    bracket = window.bracket(first_tick, last_tick)
                    _validate_streamed_frame_bracket(
                        next_frame, bracket, contract
                    )
                    frame_count += 1
                    next_frame = read_frame()
                if next_frame is None:
                    window.clear()
                else:
                    needed_tick = int(next_frame["exact_action_tick_first"])
                    window.discard_before(needed_tick)
                    window.require_within_bound()
                    peak_buffered = max(
                        peak_buffered, window.peak_buffered_ticks
                    )

        if first_record is None or last_record is None or envelope_first is None or envelope_last is None or envelope_schema is None:
            raise RobotCaptureError("raw control envelope capture is empty")
        if next_frame is not None:
            raise RobotCaptureError("exact frame index exceeds captured control ticks")
        if frame_count == 0:
            raise RobotCaptureError("robot capture contains no camera frame evidence")
        try:
            extra_row = next(csv_reader)
        except StopIteration:
            extra_row = None
        except csv.Error as exc:
            raise RobotCaptureError("exact control CSV is invalid") from exc
        if extra_row is not None:
            raise RobotCaptureError("exact control CSV contains extra action ticks")
        for descriptor, label, expected_receipt, expected_identity in (
            (
                raw_descriptor,
                "raw control envelopes",
                raw_receipt,
                raw_identity,
            ),
            (
                csv_descriptor,
                "exact control CSV",
                csv_receipt,
                csv_identity,
            ),
            (
                frame_descriptor,
                "exact robot frame index",
                frame_receipt,
                frame_identity,
            ),
        ):
            observed_receipt, observed_identity = _hash_stable_descriptor(
                descriptor, label
            )
            if (
                observed_receipt != expected_receipt
                or observed_identity != expected_identity
            ):
                raise RobotCaptureError(f"{label} changed during validation")

    return ExactCaptureReplayStats(
        envelope_schema=envelope_schema,
        envelope_count=envelope_count,
        envelope_sequence_first=envelope_first,
        envelope_sequence_last=envelope_last,
        control_tick_count=control_count,
        control_tick_first=first_record.control_tick_seq,
        control_tick_last=last_record.control_tick_seq,
        observation_monotonic_ns_first=first_record.observation_monotonic_ns,
        observation_monotonic_ns_last=last_record.observation_monotonic_ns,
        max_observation_gap_ns=max_gap,
        frame_count=frame_count,
        peak_buffered_control_ticks=peak_buffered,
        arm_tx_expected_frame_count=arm_tx_expected,
        arm_tx_full_frame_write_count=arm_tx_full,
        arm_tx_failed_frame_write_count=arm_tx_failed,
        gripper_tx_expected_frame_count=gripper_tx_expected,
        gripper_tx_full_frame_write_count=gripper_tx_full,
        gripper_tx_failed_frame_write_count=gripper_tx_failed,
        socketcan_tx_enqueue_failure_tick_count=tx_failure_ticks,
        feedback_all_ticks_fresh=feedback_all_ticks_fresh,
        feedback_max_age_ns=feedback_max_age_ns,
        feedback_max_last_gap_ns=feedback_max_last_gap_ns,
        follower_q_min_rad=tuple(follower_q_min),
        follower_q_max_rad=tuple(follower_q_max),
        action_q_min_rad=tuple(action_q_min),
        action_q_max_rad=tuple(action_q_max),
        follower_gripper_motor_q_min_rad=follower_gripper_motor_q_min,
        follower_gripper_motor_q_max_rad=follower_gripper_motor_q_max,
        action_gripper_q_min_rad=action_gripper_q_min,
        action_gripper_q_max_rad=action_gripper_q_max,
        command_response_action_m2=tuple(response_action_m2),
        command_response_observation_m2=tuple(response_observation_m2),
        command_response_covariance=tuple(response_covariance),
        first_record=first_record,
        last_record=last_record,
        hash_inventory={
            CONTROL_ENVELOPES_NAME: raw_receipt,
            EXACT_CONTROL_NAME: csv_receipt,
            EXACT_FRAME_INDEX_NAME: frame_receipt,
        },
        file_identities={
            CONTROL_ENVELOPES_NAME: raw_identity,
            EXACT_CONTROL_NAME: csv_identity,
            EXACT_FRAME_INDEX_NAME: frame_identity,
        },
    )


def _read_strict_json_object(source: object, label: str) -> Dict[str, object]:
    descriptor = _own_regular_descriptor(source, label)
    try:
        receipt, identity = _hash_stable_descriptor(descriptor, label)
        if int(receipt["size_bytes"]) > 1_048_576:
            raise RobotCaptureError(f"{label} exceeds the bounded JSON size")
        raw = bytearray()
        while len(raw) < int(receipt["size_bytes"]):
            block = os.read(
                descriptor,
                int(receipt["size_bytes"]) - len(raw),
            )
            if not block:
                break
            raw.extend(block)
        observed_receipt, observed_identity = _hash_stable_descriptor(
            descriptor, label
        )
        if (
            observed_receipt != receipt
            or observed_identity != identity
            or len(raw) != int(receipt["size_bytes"])
        ):
            raise RobotCaptureError(f"{label} changed while reading")
        value = json.loads(
            bytes(raw).decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except RobotCaptureError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RobotCaptureError(f"{label} is not strict JSON") from exc
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise RobotCaptureError(f"{label} must be a JSON object")
    return value


_EXACT_FRAME_FIELDS = frozenset(
    {
        "schema",
        "camera_id",
        "frame_index",
        "exposure_lower_ns",
        "exposure_nominal_ns",
        "exposure_upper_ns",
        "query_now_monotonic_ns",
        "observation_tick_first",
        "observation_tick_last",
        "observation_timestamp_first_ns",
        "observation_timestamp_last_ns",
        "observation_max_gap_ns",
        "observation_at_nominal",
        "nominal_alignment_rule",
        "exact_action_tick_first",
        "exact_action_tick_last",
        "exact_action_tick_count",
        "exact_action_sha256",
        "exact_control_trajectory_sha256",
        "action_interpolation",
        "controller_pid",
        "controller_start_token",
        "publisher_identity",
        "side",
        "controller_mode",
        "leader_can_interface",
        "follower_can_interface",
        "urdf_sha256",
        "joint_mapper_sha256",
        "camera_profile_sha256",
        "timing_artifact_sha256",
        "pair_contract_sha256",
        "controller_binary_sha256",
        "controller_source_manifest_sha256",
        "follower_config_sha256",
        "openarm_sdk_sha256",
        "send_status_observable",
        "send_status_reason",
    }
)
_EXACT_FRAME_FIELDS_V3 = _EXACT_FRAME_FIELDS | frozenset(
    {
        "send_status_semantics",
        "motor_ack_observable",
        "motor_ack_reason",
        "socketcan_tx_enqueue_failure_tick_count",
        "socketcan_tx_enqueue_success_all_ticks",
    }
)
_EXACT_FRAME_FIELDS_V4 = _EXACT_FRAME_FIELDS_V3 | frozenset(
    {
        "feedback_liveness_observable",
        "feedback_liveness_reason",
        "feedback_liveness_semantics",
        "feedback_max_age_ns",
        "feedback_max_gap_ns",
        "feedback_all_ticks_fresh",
        "feedback_gap_violation_counts_first",
        "feedback_gap_violation_counts_last",
    }
)


def validate_robot_capture_artifacts(
    session_dir: object,
    contract: RobotStateContract,
    *,
    artifact_fds: Optional[Mapping[str, int]] = None,
) -> Mapping[str, object]:
    """Replay one v2/v3/v4 capture from paths or a caller-pinned fd bundle."""

    path = Path(session_dir)
    if (
        path.is_symlink()
        or not path.is_dir()
        or not isinstance(contract, RobotStateContract)
    ):
        raise RobotCaptureError("committed robot capture path or contract is invalid")
    if any((path / name).exists() for name, _ in _EXACT_PARTIAL_TO_FINAL):
        raise RobotCaptureError("committed robot capture still contains partials")
    if artifact_fds is None:
        sources: Mapping[str, object] = {
            CONTROL_ENVELOPES_NAME: path / CONTROL_ENVELOPES_NAME,
            EXACT_CONTROL_NAME: path / EXACT_CONTROL_NAME,
            EXACT_FRAME_INDEX_NAME: path / EXACT_FRAME_INDEX_NAME,
            EXACT_SUMMARY_NAME: path / EXACT_SUMMARY_NAME,
        }
    else:
        if (
            not isinstance(artifact_fds, Mapping)
            or frozenset(artifact_fds) != frozenset(EXACT_ROBOT_ARTIFACT_NAMES)
            or any(
                not _is_int(descriptor) or descriptor < 0
                for descriptor in artifact_fds.values()
            )
        ):
            raise RobotCaptureError("pinned robot artifact descriptor set is invalid")
        sources = artifact_fds
    summary = _read_strict_json_object(
        sources[EXACT_SUMMARY_NAME], "robot capture summary"
    )
    if summary.get("schema") not in {
        EXACT_CAPTURE_SUMMARY_SCHEMA_V2_COMMISSIONING,
        EXACT_CAPTURE_SUMMARY_SCHEMA_V3_PRODUCTION,
        EXACT_CAPTURE_SUMMARY_SCHEMA_V4_PRODUCTION,
    }:
        raise RobotCaptureError("robot capture summary schema mismatch")
    inventory = summary.get("hash_inventory")
    if not isinstance(inventory, dict) or frozenset(inventory) != frozenset(
        final_name for _, final_name in _EXACT_PARTIAL_TO_FINAL[:-1]
    ):
        raise RobotCaptureError("robot capture hash inventory field set mismatch")
    replay = _stream_validate_capture_data(
        sources[CONTROL_ENVELOPES_NAME],
        sources[EXACT_CONTROL_NAME],
        sources[EXACT_FRAME_INDEX_NAME],
        contract,
    )
    if inventory != dict(replay.hash_inventory):
        raise RobotCaptureError(
            "robot capture hash inventory does not match stable replay descriptors"
        )
    first = replay.first_record
    is_v3 = replay.envelope_schema == CONTROL_ENVELOPE_SCHEMA_V3
    is_v4 = replay.envelope_schema == CONTROL_ENVELOPE_SCHEMA_V4
    has_tx_status = is_v3 or is_v4
    expected_summary_schema = (
        EXACT_CAPTURE_SUMMARY_SCHEMA_V4_PRODUCTION
        if is_v4
        else (
            EXACT_CAPTURE_SUMMARY_SCHEMA_V3_PRODUCTION
            if is_v3
            else EXACT_CAPTURE_SUMMARY_SCHEMA_V2_COMMISSIONING
        )
    )
    tx_clean = has_tx_status and (
        replay.arm_tx_failed_frame_write_count == 0
        and replay.gripper_tx_failed_frame_write_count == 0
        and replay.socketcan_tx_enqueue_failure_tick_count == 0
        and replay.arm_tx_expected_frame_count == replay.control_tick_count * 7
        and replay.arm_tx_full_frame_write_count == replay.control_tick_count * 7
        and replay.gripper_tx_expected_frame_count == replay.control_tick_count
        and replay.gripper_tx_full_frame_write_count == replay.control_tick_count
    )
    feedback_evaluation: Dict[str, object] = {}
    if is_v4:
        feedback_evaluation = _evaluate_v4_feedback(
            summary.get("feedback_start_baseline"),
            replay,
            end_monotonic_ns=summary.get("capture_end_monotonic_ns"),
        )
        feedback_clean = bool(feedback_evaluation["feedback_clean"])
    else:
        feedback_clean = False
    trajectory_evaluation: Dict[str, object] = (
        _evaluate_training_trajectory(replay, contract.joint_names)
        if is_v4
        else {}
    )
    trajectory_clean = (
        bool(trajectory_evaluation["trajectory_nondegenerate"])
        and bool(trajectory_evaluation["command_response_consistent"])
        if is_v4
        else False
    )
    if has_tx_status:
        eligibility_blockers = []
        if not tx_clean:
            eligibility_blockers.append("robot_socketcan_tx_enqueue_failure")
        if is_v3:
            eligibility_blockers.append(
                "robot_feedback_liveness_unobservable"
            )
        else:
            if not feedback_evaluation[
                "feedback_counter_progress_all_motors"
            ]:
                eligibility_blockers.append(
                    "robot_feedback_counter_no_progress"
                )
            if not feedback_evaluation["feedback_all_ticks_fresh"]:
                eligibility_blockers.append("robot_feedback_stale")
            if not feedback_evaluation["feedback_gap_clean"]:
                eligibility_blockers.append("robot_feedback_gap_exceeded")
            if not trajectory_evaluation["trajectory_nondegenerate"]:
                eligibility_blockers.append(
                    "robot_observation_trajectory_degenerate"
                )
            if not trajectory_evaluation["command_response_consistent"]:
                eligibility_blockers.append(
                    "robot_command_response_inconsistent"
                )
    else:
        eligibility_blockers = ["robot_action_send_status_unobservable"]
    expected = {
        "schema": expected_summary_schema,
        "status": (
            (
                "exact_control_capture_validated_feedback_liveness"
                if tx_clean and feedback_clean
                else "exact_control_capture_validated_feedback_liveness_failure"
            )
            if is_v4
            else (
                "exact_control_capture_validated_socketcan_tx_enqueue"
                if tx_clean
                else (
                    "exact_control_capture_validated_socketcan_tx_enqueue_failure"
                    if is_v3
                    else "exact_control_capture_validated_send_status_unobservable"
                )
            )
        ),
        "side": "left",
        "controller_mode": "unilateral",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "joint_names": list(contract.joint_names),
        "envelope_count": replay.envelope_count,
        "envelope_sequence_first": replay.envelope_sequence_first,
        "envelope_sequence_last": replay.envelope_sequence_last,
        "control_tick_count": replay.control_tick_count,
        "control_tick_first": replay.control_tick_first,
        "control_tick_last": replay.control_tick_last,
        "observation_monotonic_ns_first": replay.observation_monotonic_ns_first,
        "observation_monotonic_ns_last": replay.observation_monotonic_ns_last,
        "max_observation_gap_ns": replay.max_observation_gap_ns,
        "frame_count": replay.frame_count,
        "controller_pid": first.controller_pid,
        "controller_start_token": first.controller_start_token,
        "publisher_identity": first.publisher_identity,
        "urdf_sha256": contract.urdf_sha256,
        "joint_mapper_sha256": contract.joint_mapper_sha256,
        "camera_profile_sha256": contract.camera_profile_sha256,
        "timing_artifact_sha256": contract.timing_artifact_sha256,
        "pair_contract_sha256": contract.pair_contract_sha256,
        "controller_binary_sha256": contract.controller_binary_sha256,
        "controller_source_manifest_sha256": contract.controller_source_manifest_sha256,
        "follower_config_sha256": contract.follower_config_sha256,
        "openarm_sdk_sha256": contract.openarm_sdk_sha256,
        "joint_position_limits_rad": [
            [float(lower), float(upper)]
            for lower, upper in contract.joint_position_limits_rad
        ],
        "max_abs_joint_velocity_rad_s": float(
            contract.max_abs_joint_velocity_rad_s
        ),
        "gripper_motor_position_limits_rad": [
            float(value)
            for value in contract.gripper_motor_position_limits_rad
        ],
        "buffer_duration_ns": contract.buffer_duration_ns,
        "max_packets": contract.max_packets,
        "max_gap_ns": contract.max_gap_ns,
        "max_staleness_ns": contract.max_staleness_ns,
        "max_exposure_interval_ns": contract.max_exposure_interval_ns,
        "ring_overflow_count": 0,
        "publisher_rejected_record_count": 0,
        "socket_send_failure_count": 0,
        "send_status_observable": first.send_status_observable,
        "send_status_reason": first.send_status_reason,
        "training_eligible": tx_clean and feedback_clean and trajectory_clean,
        "eligibility_blockers": eligibility_blockers,
    }
    if has_tx_status:
        expected.update(
            send_status_semantics=SOCKETCAN_TX_ENQUEUE_SEMANTICS,
            motor_ack_observable=False,
            motor_ack_reason=MOTOR_ACK_UNOBSERVABLE_REASON,
            feedback_liveness_observable=False,
            feedback_liveness_reason=FEEDBACK_LIVENESS_UNOBSERVABLE_REASON,
            arm_tx_expected_frame_count=replay.arm_tx_expected_frame_count,
            arm_tx_full_frame_write_count=(
                replay.arm_tx_full_frame_write_count
            ),
            arm_tx_failed_frame_write_count=(
                replay.arm_tx_failed_frame_write_count
            ),
            gripper_tx_expected_frame_count=(
                replay.gripper_tx_expected_frame_count
            ),
            gripper_tx_full_frame_write_count=(
                replay.gripper_tx_full_frame_write_count
            ),
            gripper_tx_failed_frame_write_count=(
                replay.gripper_tx_failed_frame_write_count
            ),
            socketcan_tx_enqueue_failure_tick_count=(
                replay.socketcan_tx_enqueue_failure_tick_count
            ),
        )
    if is_v3:
        expected.update(
            feedback_liveness_observable=False,
            feedback_liveness_reason=FEEDBACK_LIVENESS_UNOBSERVABLE_REASON,
        )
    if is_v4:
        expected.update(
            feedback_liveness_observable=True,
            feedback_liveness_reason=FEEDBACK_LIVENESS_REASON,
            feedback_liveness_semantics=FEEDBACK_LIVENESS_SEMANTICS,
            feedback_max_age_ns=FEEDBACK_MAX_AGE_NS,
            feedback_max_gap_ns=FEEDBACK_MAX_GAP_NS,
            **feedback_evaluation,
            **trajectory_evaluation,
        )
    if any(summary.get(field) != value for field, value in expected.items()):
        raise RobotCaptureError("robot capture summary replay mismatch")
    return summary


__all__ = [
    "CONTROL_ENVELOPES_NAME",
    "CapturedExactControlFrameEvidence",
    "DIAGNOSTIC_V1_CAPTURE_SUMMARY_SCHEMA",
    "DIAGNOSTIC_V1_FRAME_INDEX_SCHEMA",
    "DiagnosticV1CapturedRobotFrameEvidence",
    "DiagnosticV1RobotSessionCapture",
    "EXACT_CAPTURE_SUMMARY_SCHEMA_V2_COMMISSIONING",
    "EXACT_CAPTURE_SUMMARY_SCHEMA_V3_PRODUCTION",
    "EXACT_CAPTURE_SUMMARY_SCHEMA_V4_PRODUCTION",
    "EXACT_CONTROL_NAME",
    "EXACT_FRAME_INDEX_SCHEMA_V2_COMMISSIONING",
    "EXACT_FRAME_INDEX_SCHEMA_V3_PRODUCTION",
    "EXACT_FRAME_INDEX_SCHEMA_V4_PRODUCTION",
    "FEEDBACK_LIVENESS_UNOBSERVABLE_REASON",
    "EXACT_FRAME_INDEX_NAME",
    "EXACT_ROBOT_ARTIFACT_NAMES",
    "EXACT_SUMMARY_NAME",
    "RobotCaptureError",
    "RobotSessionCapture",
    "validate_robot_capture_artifacts",
]
