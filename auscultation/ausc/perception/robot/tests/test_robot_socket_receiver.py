from __future__ import annotations

from collections import deque
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import struct
import time

import pytest

import ausc.perception.robot.robot_socket_receiver as receiver_module
from ausc.perception.robot.robot_socket_receiver import (
    ControllerProcessAttestation,
    ControllerProcessSnapshot,
    LinuxControllerProcessInspector,
    RobotSocketPreflightError,
    RobotSocketReceiver,
    RobotSocketReceiverError,
    linux_peer_credentials_available,
)
from ausc.perception.robot.robot_state_stream import (
    CLOCK_ID,
    CONTROL_ENVELOPE_SCHEMA,
    CONTROL_ENVELOPE_SCHEMA_V3,
    CONTROL_ENVELOPE_SCHEMA_V4,
    EXACT_CONTROL_RECORD_SCHEMA,
    EXACT_CONTROL_RECORD_SCHEMA_V3,
    EXACT_CONTROL_RECORD_SCHEMA_V4,
    FEEDBACK_LIVENESS_REASON,
    FEEDBACK_LIVENESS_SEMANTICS,
    MAX_CONTROL_ENVELOPE_BYTES,
    VOID_SEND_STATUS_REASON,
    MOTOR_ACK_UNOBSERVABLE_REASON,
    SOCKETCAN_TX_ENQUEUE_SEMANTICS,
    SOCKETCAN_TX_STATUS_REASON,
    ExactControlStateStream,
    RobotControlEnvelope,
    RobotStateContract,
    RobotStateStream,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_1 = "1" * 64
SHA_2 = "2" * 64
SHA_3 = "3" * 64
JOINT_NAMES = tuple(f"openarm_left_joint{i}" for i in range(1, 8))
JOINT_LIMITS = tuple((-4.0, 4.0) for _ in range(7))
LINUX_CREDENTIALS = linux_peer_credentials_available()


class _FakeProcessInspector:
    """Deterministic pidfd/procfs seam; never reads this Mac's process table."""

    def __init__(self, snapshot: ControllerProcessSnapshot) -> None:
        self.snapshot = snapshot
        self.exited = False
        self.opened_pids: list[int] = []
        self.closed_guards: list[object] = []

    def open_pidfd(self, pid: int) -> object:
        self.opened_pids.append(pid)
        return ("pidfd", pid, len(self.opened_pids))

    def pidfd_exited(self, _guard: object) -> bool:
        return self.exited

    def inspect(self, _pid: int) -> ControllerProcessSnapshot:
        return self.snapshot

    def close_pidfd(self, guard: object) -> None:
        self.closed_guards.append(guard)


def _process_snapshot(**changes) -> ControllerProcessSnapshot:
    values = {
        "pid": os.getpid(),
        "uid": os.geteuid(),
        "proc_start_time_ticks": 987_654_321,
        "executable_device": 41,
        "executable_inode": 73,
        "executable_sha256": SHA_F,
        "cgroup_sha256": SHA_D,
    }
    values.update(changes)
    return ControllerProcessSnapshot(**values)


def _process_authority() -> tuple[
    ControllerProcessAttestation, _FakeProcessInspector
]:
    inspector = _FakeProcessInspector(_process_snapshot())
    attestation = ControllerProcessAttestation.capture(
        pid=os.getpid(),
        expected_uid=os.geteuid(),
        inspector=inspector,
    )
    return attestation, inspector


def _contract(**changes) -> RobotStateContract:
    values = {
        "side": "left",
        "controller_mode": "unilateral",
        "joint_names": JOINT_NAMES,
        "joint_position_limits_rad": JOINT_LIMITS,
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "urdf_sha256": SHA_A,
        "joint_mapper_sha256": SHA_B,
        "camera_profile_sha256": SHA_C,
        "timing_artifact_sha256": SHA_D,
        "pair_contract_sha256": SHA_E,
        "controller_binary_sha256": SHA_F,
        "controller_source_manifest_sha256": SHA_1,
        "follower_config_sha256": SHA_2,
        "openarm_sdk_sha256": SHA_3,
        "max_abs_joint_velocity_rad_s": 8.0,
        "gripper_motor_position_limits_rad": (-0.20, 0.20),
        "buffer_duration_ns": 2_000_000_000,
        "max_packets": 256,
        "max_gap_ns": 20_000_000,
        "max_staleness_ns": 50_000_000,
        "max_exposure_interval_ns": 40_000_000,
    }
    values.update(changes)
    return RobotStateContract(**values)


def _record(control_tick_seq: int, observation_ns: int) -> dict:
    q = [0.01 * index for index in range(7)]
    return {
        "schema": EXACT_CONTROL_RECORD_SCHEMA,
        "control_tick_seq": control_tick_seq,
        "observation_monotonic_ns": observation_ns,
        "reference_generation": control_tick_seq + 1,
        "reference_source_monotonic_ns": observation_ns - 100_000,
        "reference_snapshot_valid": True,
        "control_call_begin_monotonic_ns": observation_ns + 50_000,
        "arm_call_return_monotonic_ns": observation_ns + 100_000,
        "gripper_call_begin_monotonic_ns": observation_ns + 110_000,
        "gripper_call_return_monotonic_ns": observation_ns + 160_000,
        "control_call_end_monotonic_ns": observation_ns + 170_000,
        "follower_q_rad": q,
        "follower_qd_rad_s": [0.5] * 7,
        "follower_gripper_q_rad": -0.070,
        "follower_gripper_qd_rad_s": 0.010,
        "follower_gripper_motor_q_rad": -0.075,
        "follower_gripper_motor_qd_rad_s": 0.015,
        "follower_arm_mit_kp": [240.0] * 7,
        "follower_arm_mit_kd": [3.0] * 7,
        "follower_arm_mit_q": q,
        "follower_arm_mit_dq": [0.4] * 7,
        "follower_arm_mit_tau": [0.0] * 7,
        "follower_gripper_mit_kp": 25.0,
        "follower_gripper_mit_kd": 0.3,
        "follower_gripper_mit_q": -0.080,
        "follower_gripper_mit_dq": 0.0,
        "follower_gripper_mit_tau": 0.0,
    }


def _envelope(
    *,
    envelope_seq: int,
    first_tick: int,
    first_observation_ns: int,
    count: int = 10,
    controller_pid: int | None = None,
    **changes,
) -> dict:
    records = [
        _record(first_tick + index, first_observation_ns + index * 1_000_000)
        for index in range(count)
    ]
    value = {
        "schema": CONTROL_ENVELOPE_SCHEMA,
        "envelope_seq": envelope_seq,
        "clock_id": CLOCK_ID,
        "publisher_monotonic_ns": records[-1]["control_call_end_monotonic_ns"] + 500_000,
        "controller_pid": os.getpid() if controller_pid is None else controller_pid,
        "controller_start_token": "pytest-controller-start",
        "side": "left",
        "controller_mode": "unilateral",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "joint_names": list(JOINT_NAMES),
        "urdf_sha256": SHA_A,
        "joint_mapper_sha256": SHA_B,
        "pair_contract_sha256": SHA_E,
        "controller_binary_sha256": SHA_F,
        "controller_source_manifest_sha256": SHA_1,
        "follower_config_sha256": SHA_2,
        "openarm_sdk_sha256": SHA_3,
        "publisher_nominal_hz": 100,
        "control_nominal_hz": 1000,
        "send_status_observable": False,
        "send_status_reason": VOID_SEND_STATUS_REASON,
        "ring_overflow_count": 0,
        "publisher_rejected_record_count": 0,
        "socket_send_failure_count": 0,
        "records": records,
    }
    value.update(changes)
    return value


def _wire(value: dict) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _v3_envelope(**kwargs) -> dict:
    value = _envelope(**kwargs)
    value.update(
        schema=CONTROL_ENVELOPE_SCHEMA_V3,
        send_status_observable=True,
        send_status_reason=SOCKETCAN_TX_STATUS_REASON,
        send_status_semantics=SOCKETCAN_TX_ENQUEUE_SEMANTICS,
        motor_ack_observable=False,
        motor_ack_reason=MOTOR_ACK_UNOBSERVABLE_REASON,
    )
    for record in value["records"]:
        record.update(
            schema=EXACT_CONTROL_RECORD_SCHEMA_V3,
            arm_tx_expected_frame_count=7,
            arm_tx_full_frame_write_count=7,
            arm_tx_failed_frame_write_count=0,
            gripper_tx_expected_frame_count=1,
            gripper_tx_full_frame_write_count=1,
            gripper_tx_failed_frame_write_count=0,
            socketcan_tx_enqueue_success=True,
        )
    return value


def _v4_envelope(**kwargs) -> dict:
    value = _v3_envelope(**kwargs)
    value.update(
        schema=CONTROL_ENVELOPE_SCHEMA_V4,
        feedback_liveness_observable=True,
        feedback_liveness_reason=FEEDBACK_LIVENESS_REASON,
        feedback_liveness_semantics=FEEDBACK_LIVENESS_SEMANTICS,
        feedback_max_age_ns=20_000_000,
        feedback_max_gap_ns=20_000_000,
    )
    for record in value["records"]:
        tick = record["control_tick_seq"]
        observation_ns = record["observation_monotonic_ns"]
        arm_counters = [tick * 8 + index + 1 for index in range(7)]
        record.update(
            schema=EXACT_CONTROL_RECORD_SCHEMA_V4,
            feedback_snapshot_monotonic_ns=observation_ns,
            follower_arm_state_rx_motor_ids=list(range(1, 8)),
            follower_arm_state_rx_counters=arm_counters,
            follower_arm_state_rx_generations=arm_counters,
            follower_arm_state_rx_monotonic_ns=[
                observation_ns - 500_000 + index for index in range(7)
            ],
            follower_arm_state_rx_last_gap_ns=[1_000_000] * 7,
            follower_arm_state_rx_gap_violation_counts=[0] * 7,
            follower_gripper_state_rx_motor_id=8,
            follower_gripper_state_rx_counter=tick * 8 + 8,
            follower_gripper_state_rx_generation=tick * 8 + 8,
            follower_gripper_state_rx_monotonic_ns=observation_ns - 400_000,
            follower_gripper_state_rx_last_gap_ns=1_000_000,
            follower_gripper_state_rx_gap_violation_count=0,
        )
    return value


def _fresh_v4() -> dict:
    return _v4_envelope(
        envelope_seq=1,
        first_tick=100,
        first_observation_ns=time.monotonic_ns() - 20_000_000,
    )


def _wire_v4(env: dict) -> bytes:
    return json.dumps(env, separators=(",", ":")).encode("utf-8")


def test_pinned_rx_ids_reject_permuted_and_wrong_gripper_accept_matching():
    # P0-5: when the activated pair pins the follower STATE RX motor-ID mapping, the
    # runtime decode must match it EXACTLY -- a permuted or mis-mapped controller is
    # rejected (defense in depth atop the binary attestation).
    from ausc.perception.robot.robot_state_stream import RobotTelemetryError

    contract = _contract(
        expected_arm_rx_motor_ids=(1, 2, 3, 4, 5, 6, 7),
        expected_gripper_rx_motor_id=8,
    )
    # exact match -> accepted
    ExactControlStateStream(contract).ingest_datagram(_wire_v4(_fresh_v4()))

    # permuted arm IDs -> rejected
    permuted = _fresh_v4()
    permuted["records"][0]["follower_arm_state_rx_motor_ids"] = [2, 1, 3, 4, 5, 6, 7]
    with pytest.raises(RobotTelemetryError, match="exactly match the activated"):
        ExactControlStateStream(contract).ingest_datagram(_wire_v4(permuted))

    # wrong gripper ID -> rejected
    bad_gripper = _fresh_v4()
    bad_gripper["records"][0]["follower_gripper_state_rx_motor_id"] = 9
    bad_gripper["records"][0]["follower_gripper_state_rx_counter"] = 99
    bad_gripper["records"][0]["follower_gripper_state_rx_generation"] = 99
    with pytest.raises(RobotTelemetryError, match="gripper STATE RX motor ID"):
        ExactControlStateStream(contract).ingest_datagram(_wire_v4(bad_gripper))


def test_unpinned_contract_keeps_seven_unique_fallback():
    # P0-5 backward-compat: a contract with no pinned mapping still accepts any
    # seven-unique arm RX set (the legacy path used by lower-level fixtures).
    ExactControlStateStream(_contract()).ingest_datagram(_wire_v4(_fresh_v4()))


class _RecordingExactSink:
    telemetry_schema = CONTROL_ENVELOPE_SCHEMA

    def __init__(self, contract: RobotStateContract) -> None:
        self.contract = contract
        # mirrors production wiring: the coordinator's live preflight stream
        # opts into bounded-hole absorption (session capture stays strict)
        self.stream = ExactControlStateStream(contract, absorb_bounded_holes=True)
        self.raw_payloads: list[bytes] = []

    def ingest_datagram(self, payload: bytes) -> RobotControlEnvelope:
        self.raw_payloads.append(payload)
        return self.stream.ingest_datagram(payload)


class _RecordingV3Sink(_RecordingExactSink):
    telemetry_schema = CONTROL_ENVELOPE_SCHEMA_V3
    telemetry_schemas = frozenset(
        {CONTROL_ENVELOPE_SCHEMA, CONTROL_ENVELOPE_SCHEMA_V3}
    )


class _RecordingV4Sink(_RecordingExactSink):
    telemetry_schema = CONTROL_ENVELOPE_SCHEMA_V4
    telemetry_schemas = frozenset({CONTROL_ENVELOPE_SCHEMA_V4})


class _FakeRecvmsgSocket:
    def __init__(
        self,
        payloads: list[bytes],
        *,
        pid: int | None = None,
        flags: int = 0,
    ) -> None:
        self.payloads = deque(payloads)
        self.pid = os.getpid() if pid is None else pid
        self.flags = flags
        self.recv_sizes: list[int] = []

    def recvmsg(self, payload_size: int, _ancillary_size: int):
        self.recv_sizes.append(payload_size)
        if not self.payloads:
            raise BlockingIOError
        payload = self.payloads.popleft()
        kind = receiver_module._SCM_CREDENTIALS
        assert kind is not None
        ancillary = [
            (
                socket.SOL_SOCKET,
                int(kind),
                struct.pack("=3i", self.pid, os.geteuid(), os.getegid()),
            )
        ]
        return payload, ancillary, self.flags, None


def _receiver(
    tmp_path: Path,
    *,
    sink: object | None = None,
    aborts: list[str] | None = None,
    **changes,
) -> RobotSocketReceiver:
    runtime_dir = Path("/tmp") / (
        "wam-rx-" + hashlib.sha256(os.fsencode(tmp_path)).hexdigest()[:16]
    )
    attestation, inspector = _process_authority()
    values = {
        "socket_path": runtime_dir / "openarm-left-control-v2.sock",
        "sink": _RecordingExactSink(_contract()) if sink is None else sink,
        "runtime_dir": runtime_dir,
        "controller_process_attestation": attestation,
        "process_inspector": inspector,
        "abort_callback": None if aborts is None else aborts.append,
        "poll_timeout_s": 0.01,
        "preflight_min_envelopes": 3,
        "preflight_duration_ns": 20_000_000,
    }
    values.update(changes)
    return RobotSocketReceiver(**values)


def _unit_ingest(receiver: RobotSocketReceiver, payload: bytes, *, pid: int | None = None) -> None:
    fake = _FakeRecvmsgSocket([payload], pid=pid)
    assert receiver._receive_one(fake) is True


def _wait_for(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true before timeout")


def test_attestation_capture_reads_fake_procfs_and_hashes_the_open_executable_fd(
    tmp_path: Path,
) -> None:
    pid = 4242
    proc_root = tmp_path / "proc"
    process_root = proc_root / str(pid)
    process_root.mkdir(parents=True)
    executable = tmp_path / "openarm-controller"
    executable_bytes = b"trusted-left-openarm-controller-v2\x00"
    executable.write_bytes(executable_bytes)
    (process_root / "exe").symlink_to(executable)
    # Field 22 is process starttime.  The comm field deliberately contains a
    # space so the parser cannot use a naive whole-line split.
    tail_fields = ["S"] + ["0"] * 18 + ["987654321"] + ["0"] * 4
    (process_root / "stat").write_text(
        f"{pid} (openarm control) {' '.join(tail_fields)}\n",
        encoding="utf-8",
    )
    (process_root / "status").write_text(
        f"Name:\topenarm\nUid:\t{os.geteuid()}\t{os.geteuid()}\t"
        f"{os.geteuid()}\t{os.geteuid()}\n",
        encoding="utf-8",
    )
    cgroup_bytes = b"0::/wam/openarm-left.service\n"
    (process_root / "cgroup").write_bytes(cgroup_bytes)
    opened: list[int] = []
    closed: list[object] = []
    inspector = LinuxControllerProcessInspector(
        proc_root=proc_root,
        pidfd_open=lambda requested_pid: opened.append(requested_pid) or 91,
        pidfd_exited=lambda guard: False,
        pidfd_close=closed.append,
    )

    attestation = ControllerProcessAttestation.capture(
        pid=pid,
        expected_uid=os.geteuid(),
        inspector=inspector,
    )

    executable_stat = executable.stat()
    assert attestation.pid == pid
    assert attestation.uid == os.geteuid()
    assert attestation.proc_start_time_ticks == 987_654_321
    assert attestation.executable_device == executable_stat.st_dev
    assert attestation.executable_inode == executable_stat.st_ino
    assert attestation.executable_sha256 == hashlib.sha256(executable_bytes).hexdigest()
    assert attestation.cgroup_sha256 == hashlib.sha256(cgroup_bytes).hexdigest()
    assert opened == [pid]
    assert closed == [91]
    with pytest.raises(FrozenInstanceError):
        attestation.pid = pid + 1  # type: ignore[misc]


def test_constructor_requires_independent_controller_process_attestation(
    tmp_path: Path,
) -> None:
    runtime_dir = Path("/tmp") / (
        "wam-rx-" + hashlib.sha256(os.fsencode(tmp_path)).hexdigest()[:16]
    )
    with pytest.raises(RobotSocketReceiverError, match="process attestation"):
        RobotSocketReceiver(
            socket_path=runtime_dir / "openarm-left-control-v2.sock",
            sink=_RecordingExactSink(_contract()),
            runtime_dir=runtime_dir,
        )


def test_deferred_controller_attestation_is_an_explicit_two_phase_mode(
    tmp_path: Path,
) -> None:
    runtime_dir = Path("/tmp") / (
        "wam-rx-" + hashlib.sha256(os.fsencode(tmp_path)).hexdigest()[:16]
    )
    _attestation, inspector = _process_authority()
    receiver = RobotSocketReceiver(
        socket_path=runtime_dir / "openarm-left-control-v2.sock",
        sink=_RecordingExactSink(_contract()),
        runtime_dir=runtime_dir,
        deferred_controller_attestation=True,
        process_inspector=inspector,
    )

    assert receiver.health().status == "STOPPED"
    with pytest.raises(
        RobotSocketReceiverError, match="attach.*after.*prepared"
    ):
        receiver.attach_controller_process(_process_authority()[0])
    with pytest.raises(
        RobotSocketReceiverError, match="attested controller process"
    ):
        receiver.start()


def test_deferred_mode_cannot_hide_an_already_available_attestation(
    tmp_path: Path,
) -> None:
    runtime_dir = Path("/tmp") / (
        "wam-rx-" + hashlib.sha256(os.fsencode(tmp_path)).hexdigest()[:16]
    )
    attestation, inspector = _process_authority()
    with pytest.raises(
        RobotSocketReceiverError, match="deferred.*without.*attestation"
    ):
        RobotSocketReceiver(
            socket_path=runtime_dir / "openarm-left-control-v2.sock",
            sink=_RecordingExactSink(_contract()),
            runtime_dir=runtime_dir,
            controller_process_attestation=attestation,
            deferred_controller_attestation=True,
            process_inspector=inspector,
        )


def test_constructor_rejects_old_diagnostic_state_stream(tmp_path: Path) -> None:
    runtime_dir = Path("/tmp") / (
        "wam-rx-" + hashlib.sha256(os.fsencode(tmp_path)).hexdigest()[:16]
    )
    with pytest.raises(
        RobotSocketReceiverError, match="exact control envelope schema"
    ):
        RobotSocketReceiver(
            socket_path=runtime_dir / "openarm-left-control-v2.sock",
            sink=RobotStateStream(_contract()),
            runtime_dir=runtime_dir,
        )


def test_constructor_rejects_attested_executable_not_pinned_by_contract(
    tmp_path: Path,
) -> None:
    inspector = _FakeProcessInspector(
        _process_snapshot(executable_sha256="0" * 64)
    )
    attestation = ControllerProcessAttestation.capture(
        pid=os.getpid(),
        expected_uid=os.geteuid(),
        inspector=inspector,
    )
    with pytest.raises(RobotSocketReceiverError, match="executable SHA-256"):
        _receiver(
            tmp_path,
            controller_process_attestation=attestation,
            process_inspector=inspector,
        )


def test_constructor_rejects_any_pair_other_than_left_can1_to_can3(tmp_path: Path) -> None:
    right_contract = replace(
        _contract(),
        side="right",
        joint_names=tuple(f"openarm_right_joint{i}" for i in range(1, 8)),
        leader_can_interface="can0",
        follower_can_interface="can2",
    )
    with pytest.raises(RobotSocketReceiverError, match="left.*can1.*can3"):
        _receiver(tmp_path, sink=_RecordingExactSink(right_contract))


@pytest.mark.parametrize(
    "missing_hash",
    (
        "controller_source_manifest_sha256",
        "follower_config_sha256",
        "openarm_sdk_sha256",
    ),
)
def test_constructor_requires_complete_v2_source_identity(tmp_path: Path, missing_hash: str) -> None:
    contract = _contract(**{missing_hash: ""})
    with pytest.raises(RobotSocketReceiverError, match="activated.*SHA-256"):
        _receiver(tmp_path, sink=_RecordingExactSink(contract))


def test_receive_uses_full_v2_envelope_capacity_and_preserves_exact_wire_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    sink = _RecordingExactSink(_contract())
    receiver = _receiver(tmp_path, sink=sink)
    payload = _wire(
        _envelope(
            envelope_seq=1,
            first_tick=100,
            first_observation_ns=time.monotonic_ns() - 20_000_000,
        )
    )
    fake = _FakeRecvmsgSocket([payload])

    assert receiver._receive_one(fake) is True
    assert fake.recv_sizes == [MAX_CONTROL_ENVELOPE_BYTES]
    assert sink.raw_payloads == [payload]
    assert receiver.health().last_envelope_seq == 1
    assert receiver.health().last_control_tick_seq == 109


def test_v3_sink_rejects_v2_commissioning_packet_before_contamination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    sink = _RecordingV3Sink(_contract())
    receiver = _receiver(tmp_path, sink=sink)
    payload = _wire(
        _envelope(
            envelope_seq=1,
            first_tick=100,
            first_observation_ns=time.monotonic_ns() - 20_000_000,
        )
    )

    with pytest.raises(RobotSocketReceiverError, match="requires.*v3"):
        _unit_ingest(receiver, payload)
    assert sink.raw_payloads == []


def test_v3_sink_accepts_observable_enqueue_packet_with_motor_ack_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    sink = _RecordingV3Sink(_contract())
    receiver = _receiver(tmp_path, sink=sink)
    payload = _wire(
        _v3_envelope(
            envelope_seq=1,
            first_tick=100,
            first_observation_ns=time.monotonic_ns() - 20_000_000,
        )
    )

    _unit_ingest(receiver, payload)
    assert sink.raw_payloads == [payload]
    assert receiver.required_control_envelope_schema == CONTROL_ENVELOPE_SCHEMA_V3


def test_v4_production_sink_rejects_v3_before_ingest_and_accepts_fresh_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    sink = _RecordingV4Sink(_contract())
    receiver = _receiver(tmp_path, sink=sink)
    common = {
        "envelope_seq": 1,
        "first_tick": 100,
        "first_observation_ns": time.monotonic_ns() - 20_000_000,
    }

    with pytest.raises(RobotSocketReceiverError, match="requires.*v4"):
        _unit_ingest(receiver, _wire(_v3_envelope(**common)))
    assert sink.raw_payloads == []

    payload = _wire(_v4_envelope(**common))
    _unit_ingest(receiver, payload)
    assert sink.raw_payloads == [payload]
    assert receiver.required_control_envelope_schema == CONTROL_ENVELOPE_SCHEMA_V4


def test_credential_pid_must_match_v2_envelope_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    receiver = _receiver(tmp_path)
    payload = _wire(
        _envelope(
            envelope_seq=1,
            first_tick=100,
            first_observation_ns=time.monotonic_ns() - 20_000_000,
            controller_pid=os.getpid() + 1,
        )
    )
    with pytest.raises(RobotSocketReceiverError, match="credential PID"):
        _unit_ingest(receiver, payload, pid=os.getpid())


def test_same_uid_sender_with_wrong_scm_pid_cannot_impersonate_attested_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    receiver = _receiver(tmp_path)
    payload = _wire(
        _envelope(
            envelope_seq=1,
            first_tick=100,
            first_observation_ns=time.monotonic_ns() - 20_000_000,
        )
    )
    with pytest.raises(RobotSocketReceiverError, match="attested controller PID"):
        _unit_ingest(receiver, payload, pid=os.getpid() + 1)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda snapshot: replace(
                snapshot,
                proc_start_time_ticks=snapshot.proc_start_time_ticks + 1,
            ),
            "starttime",
        ),
        (
            lambda snapshot: replace(
                snapshot,
                executable_inode=snapshot.executable_inode + 1,
            ),
            "executable inode",
        ),
        (
            lambda snapshot: replace(snapshot, executable_sha256="0" * 64),
            "executable SHA-256",
        ),
        (
            lambda snapshot: replace(snapshot, cgroup_sha256="0" * 64),
            "cgroup",
        ),
        (
            lambda snapshot: replace(snapshot, uid=snapshot.uid + 1),
            "UID",
        ),
    ),
)
def test_proc_identity_change_faults_before_correctly_self_reported_packet_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    attestation, inspector = _process_authority()
    receiver = _receiver(
        tmp_path,
        controller_process_attestation=attestation,
        process_inspector=inspector,
    )
    receiver._started = True
    # All envelope identities, including the binary hash and arbitrary
    # controller_start_token, remain syntactically correct.  Authority comes
    # from procfs + pidfd, not from these self-reported fields.
    value = _envelope(
        envelope_seq=1,
        first_tick=100,
        first_observation_ns=time.monotonic_ns() - 20_000_000,
        controller_start_token="attacker-copied-start-token",
    )
    inspector.snapshot = mutation(inspector.snapshot)

    with pytest.raises(RobotSocketReceiverError, match=message):
        _unit_ingest(receiver, _wire(value))
    assert receiver.health().status == "FAULT"
    assert receiver.health().preflight_ready is False


def test_pidfd_exit_revokes_ready_state_without_waiting_for_another_datagram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    attestation, inspector = _process_authority()
    receiver = _receiver(
        tmp_path,
        controller_process_attestation=attestation,
        process_inspector=inspector,
    )
    receiver._started = True
    base = time.monotonic_ns() - 30_000_000
    for index in range(3):
        _unit_ingest(
            receiver,
            _wire(
                _envelope(
                    envelope_seq=10 + index,
                    first_tick=100 + index * 10,
                    first_observation_ns=base + index * 10_000_000,
                )
            ),
        )
    assert receiver.require_start_preflight(
        now_monotonic_ns=base + 30_000_000
    ).status == "READY"

    inspector.exited = True

    assert receiver.health(now_monotonic_ns=base + 31_000_000).status == "FAULT"
    with pytest.raises(RobotSocketPreflightError, match="pidfd.*exited"):
        receiver.require_start_preflight(
            now_monotonic_ns=base + 31_000_000
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(ring_overflow_count=1), "ring_overflow"),
        (lambda value: value.update(leader_can_interface="can0"), "leader CAN"),
        (lambda value: value.update(follower_can_interface="can2"), "follower CAN"),
        (lambda value: value.update(controller_source_manifest_sha256="0" * 64), "source manifest"),
        (lambda value: value.update(follower_config_sha256="0" * 64), "config"),
        (lambda value: value.update(openarm_sdk_sha256="0" * 64), "SDK"),
    ),
)
def test_receive_rejects_loss_wrong_pair_or_wrong_v2_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    receiver = _receiver(tmp_path)
    value = _envelope(
        envelope_seq=1,
        first_tick=100,
        first_observation_ns=time.monotonic_ns() - 20_000_000,
    )
    mutation(value)
    with pytest.raises(Exception, match=message):
        _unit_ingest(receiver, _wire(value))


@pytest.mark.parametrize("loss", ("envelope", "tick"))
def test_receive_rejects_envelope_or_exact_tick_loss_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loss: str,
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    receiver = _receiver(tmp_path)
    base = time.monotonic_ns() - 30_000_000
    _unit_ingest(
        receiver,
        _wire(_envelope(envelope_seq=1, first_tick=100, first_observation_ns=base)),
    )
    envelope_seq = 3 if loss == "envelope" else 2
    first_tick = 110 if loss == "envelope" else 111
    with pytest.raises(Exception, match="envelope sequence|control tick"):
        _unit_ingest(
            receiver,
            _wire(
                _envelope(
                    envelope_seq=envelope_seq,
                    first_tick=first_tick,
                    first_observation_ns=base + 10_000_000,
                )
            ),
        )


def test_preflight_requires_continuous_1khz_ticks_and_100hz_envelopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    receiver = _receiver(tmp_path)
    receiver._started = True
    base = time.monotonic_ns() - 30_000_000
    for index in range(3):
        _unit_ingest(
            receiver,
            _wire(
                _envelope(
                    envelope_seq=10 + index,
                    first_tick=100 + index * 10,
                    first_observation_ns=base + index * 10_000_000,
                )
            ),
        )

    health = receiver.require_start_preflight(now_monotonic_ns=base + 30_000_000)
    assert health.status == "READY"
    assert health.pair == "left:can1->can3"
    assert health.envelope_count == 3
    assert health.control_tick_count == 30
    assert health.last_envelope_seq == 12
    assert health.last_control_tick_seq == 129


@pytest.mark.parametrize("failure", ("tx", "feedback"))
def test_v4_preflight_and_health_require_verified_tx_and_fresh_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    receiver = _receiver(
        tmp_path,
        sink=_RecordingV4Sink(_contract()),
    )
    receiver._started = True
    base = time.monotonic_ns() - 30_000_000
    for index in range(3):
        envelope = _v4_envelope(
            envelope_seq=10 + index,
            first_tick=100 + index * 10,
            first_observation_ns=base + index * 10_000_000,
        )
        if failure == "tx":
            record = envelope["records"][5]
            record["arm_tx_full_frame_write_count"] = 6
            record["arm_tx_failed_frame_write_count"] = 1
            record["socketcan_tx_enqueue_success"] = False
        else:
            for record in envelope["records"]:
                record["follower_arm_state_rx_monotonic_ns"][3] = (
                    record["feedback_snapshot_monotonic_ns"] - 20_000_001
                )
        _unit_ingest(receiver, _wire(envelope))

    health = receiver.health(now_monotonic_ns=base + 30_000_000)
    assert health.status == "WARMING"
    assert health.preflight_ready is False
    expected = "TX enqueue failure" if failure == "tx" else "feedback.*stale"
    with pytest.raises(RobotSocketPreflightError, match=expected):
        receiver.require_start_preflight(
            now_monotonic_ns=base + 30_000_000
        )


def test_v4_preflight_allows_stationary_commands_with_fresh_motor_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    receiver = _receiver(
        tmp_path,
        sink=_RecordingV4Sink(_contract()),
    )
    receiver._started = True
    base = time.monotonic_ns() - 30_000_000
    for index in range(3):
        envelope = _v4_envelope(
            envelope_seq=10 + index,
            first_tick=100 + index * 10,
            first_observation_ns=base + index * 10_000_000,
        )
        # The helper's q/action values are intentionally stationary. Fresh
        # per-motor RX evidence, not commanded motion, gates Start.
        _unit_ingest(receiver, _wire(envelope))

    health = receiver.require_start_preflight(
        now_monotonic_ns=base + 30_000_000
    )
    assert health.status == "READY"
    assert health.preflight_ready is True


def test_preflight_rejects_slow_control_ticks_even_when_sequences_are_contiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    receiver = _receiver(tmp_path, preflight_min_envelopes=2, preflight_duration_ns=10_000_000)
    receiver._started = True
    base = time.monotonic_ns() - 30_000_000
    first = _envelope(envelope_seq=1, first_tick=1, first_observation_ns=base)
    second = _envelope(
        envelope_seq=2,
        first_tick=11,
        first_observation_ns=base + 25_000_000,
    )
    _unit_ingest(receiver, _wire(first))
    with pytest.raises(Exception, match="2 ms|1 kHz"):
        _unit_ingest(receiver, _wire(second))


@pytest.mark.skipif(
    not LINUX_CREDENTIALS,
    reason="Linux SCM_CREDENTIALS integration is verified on Lingbox",
)
def test_real_unix_datagram_checks_permissions_v2_credentials_and_preflight(
    tmp_path: Path,
) -> None:
    receiver = _receiver(tmp_path)
    receiver.start()
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        assert stat.S_IMODE(receiver.runtime_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(receiver.socket_path.stat().st_mode) == 0o600
        assert stat.S_ISSOCK(receiver.socket_path.stat().st_mode)
        assert receiver.socket_flags["nonblocking"]
        assert receiver.socket_flags["close_on_exec"]
        assert receiver.socket_flags["pass_credentials"]

        with pytest.raises(RobotSocketPreflightError, match="no telemetry"):
            receiver.require_start_preflight()

        base = time.monotonic_ns() - 30_000_000
        for index in range(3):
            sender.sendto(
                _wire(
                    _envelope(
                        envelope_seq=10 + index,
                        first_tick=100 + index * 10,
                        first_observation_ns=base + index * 10_000_000,
                    )
                ),
                os.fspath(receiver.socket_path),
            )
        _wait_for(lambda: receiver.health().last_envelope_seq == 12)
        health = receiver.require_start_preflight()
        assert health.status == "READY"
        assert health.controller_pid == os.getpid()
        assert health.pair == "left:can1->can3"
    finally:
        sender.close()
        receiver.stop()
    assert not receiver.socket_path.exists()


@pytest.mark.skipif(
    not LINUX_CREDENTIALS,
    reason="Linux two-phase socket/attestation startup is verified on Lingbox",
)
def test_two_phase_socket_prepare_precedes_controller_attestation_and_receive(
    tmp_path: Path,
) -> None:
    runtime_dir = Path("/tmp") / (
        "wam-rx-" + hashlib.sha256(os.fsencode(tmp_path)).hexdigest()[:16]
    )
    attestation, inspector = _process_authority()
    receiver = RobotSocketReceiver(
        socket_path=runtime_dir / "openarm-left-control-v2.sock",
        sink=_RecordingExactSink(_contract()),
        runtime_dir=runtime_dir,
        deferred_controller_attestation=True,
        process_inspector=inspector,
        poll_timeout_s=0.01,
        preflight_min_envelopes=3,
        preflight_duration_ns=20_000_000,
    )

    receiver.prepare_socket()
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        assert receiver.socket_path.exists()
        assert receiver.health().status == "WAITING_CONTROLLER"

        # The real launcher starts OpenArm only after this socket exists, then
        # captures the final exec identity and attaches it before reception.
        receiver.attach_controller_process(attestation)
        receiver.start()

        base = time.monotonic_ns() - 30_000_000
        for index in range(3):
            sender.sendto(
                _wire(
                    _envelope(
                        envelope_seq=10 + index,
                        first_tick=100 + index * 10,
                        first_observation_ns=base + index * 10_000_000,
                    )
                ),
                os.fspath(receiver.socket_path),
            )
        _wait_for(lambda: receiver.health().last_envelope_seq == 12)
        assert receiver.require_start_preflight().status == "READY"
    finally:
        sender.close()
        receiver.stop()
    assert not receiver.socket_path.exists()


@pytest.mark.skipif(
    not LINUX_CREDENTIALS,
    reason="Linux SCM_CREDENTIALS integration is verified on Lingbox",
)
def test_fault_latches_once_and_stop_does_not_unlink_replaced_path(tmp_path: Path) -> None:
    aborts: list[str] = []
    receiver = _receiver(tmp_path, aborts=aborts)
    receiver.start()
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        value = _envelope(
            envelope_seq=1,
            first_tick=1,
            first_observation_ns=time.monotonic_ns() - 10_000_000,
            ring_overflow_count=1,
        )
        sender.sendto(_wire(value), os.fspath(receiver.socket_path))
        _wait_for(lambda: receiver.health().status == "FAULT")
        assert aborts == [receiver.health().error]

        socket_path = receiver.socket_path
        target = tmp_path / "must-survive"
        target.write_text("keep", encoding="utf-8")
        socket_path.unlink()
        socket_path.symlink_to(target)
    finally:
        sender.close()
        receiver.stop()

    assert socket_path.is_symlink()
    assert target.read_text(encoding="utf-8") == "keep"


def test_open_pidfd_uses_syscall_fallback_when_cpython_lacks_wrapper(monkeypatch):
    # A CPython build without os.pidfd_open must still attest via the raw pidfd
    # syscall (a genuine pidfd), never degrade to a non-pidfd path or hard-fail.
    real_fd = os.open(os.devnull, os.O_RDONLY)
    seen: dict = {}

    def fake_syscall(pid):
        seen["pid"] = pid
        return real_fd

    monkeypatch.delattr(os, "pidfd_open", raising=False)
    monkeypatch.setattr(receiver_module, "_SYSCALL_PIDFD_OPEN", fake_syscall)
    inspector = LinuxControllerProcessInspector()
    guard = inspector.open_pidfd(4321)
    assert guard == real_fd and seen["pid"] == 4321
    inspector.close_pidfd(guard)


def test_open_pidfd_fails_closed_when_no_pidfd_available(monkeypatch):
    # No CPython wrapper AND no usable syscall fallback -> hard error, never a
    # silent skip of controller attestation.
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    monkeypatch.setattr(receiver_module, "_SYSCALL_PIDFD_OPEN", None)
    with pytest.raises(receiver_module.RobotSocketReceiverError):
        LinuxControllerProcessInspector().open_pidfd(4321)


def test_receive_absorbs_bounded_tick_aligned_publisher_hole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One missed 10 ms publisher slot (the 2026-08-07 'expected 109 got 110'
    live fault) must be absorbed and counted, not latched as a terminal fault —
    provided the tick counter corroborates the hole exactly."""
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    receiver = _receiver(tmp_path)
    base = time.monotonic_ns() - 60_000_000
    _unit_ingest(
        receiver,
        _wire(_envelope(envelope_seq=1, first_tick=100, first_observation_ns=base)),
    )
    # seq jumps 1 -> 3 (one missed envelope); ticks jump 109 -> 120 = 1 + 1*10,
    # observations resume 20 ms later: a genuine, exactly-aligned bounded hole.
    _unit_ingest(
        receiver,
        _wire(
            _envelope(
                envelope_seq=3,
                first_tick=120,
                first_observation_ns=base + 20_000_000,
            )
        ),
    )
    health = receiver.health()
    assert health.status != "FAULT"
    assert health.publisher_hole_count == 1
    assert health.missed_envelope_total == 1
    # stream continues strictly afterwards
    _unit_ingest(
        receiver,
        _wire(
            _envelope(
                envelope_seq=4,
                first_tick=130,
                first_observation_ns=base + 30_000_000,
            )
        ),
    )
    assert receiver.health().status != "FAULT"


def test_receive_rejects_hole_larger_than_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large telemetry hole (e.g. the 1.33 s cold-load stall) must still fault:
    data-void policy for real losses is unchanged."""
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    receiver = _receiver(tmp_path)
    base = time.monotonic_ns() - 200_000_000
    _unit_ingest(
        receiver,
        _wire(_envelope(envelope_seq=1, first_tick=100, first_observation_ns=base)),
    )
    with pytest.raises(RobotSocketReceiverError, match="sequence loss or duplication"):
        _unit_ingest(
            receiver,
            _wire(
                _envelope(
                    envelope_seq=9,  # 7 missed > tolerance of 5
                    first_tick=180,
                    first_observation_ns=base + 80_000_000,
                )
            ),
        )


def test_receive_rejects_seq_jump_without_matching_tick_hole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seq jump whose ticks are NOT aligned to the hole (ticks contiguous =>
    renumbering/duplication, not a real publisher hole) must still fault."""
    monkeypatch.setattr(receiver_module, "_SCM_CREDENTIALS", 2)
    receiver = _receiver(tmp_path)
    base = time.monotonic_ns() - 60_000_000
    _unit_ingest(
        receiver,
        _wire(_envelope(envelope_seq=1, first_tick=100, first_observation_ns=base)),
    )
    with pytest.raises(RobotSocketReceiverError, match="sequence loss or duplication"):
        _unit_ingest(
            receiver,
            _wire(
                _envelope(
                    envelope_seq=3,
                    first_tick=110,  # contiguous ticks but seq skipped => inconsistent
                    first_observation_ns=base + 10_000_000,
                )
            ),
        )


def test_preflight_stream_absorbs_bounded_v4_hole_capture_stream_stays_strict():
    """The SEALED production controller publishes the v4 schema (verified in the
    deployed ELF 2026-08-07); a v2-only hole gate made absorption dead code live.
    Policy: the live preflight stream (opt-in) absorbs a bounded tick-aligned
    hole so bring-up survives a missed 10 ms publisher slot; the session-capture
    stream keeps the DEFAULT strict contract so recorded telemetry stays gapless
    (a hole during recording fails fast instead of dying at finalize)."""
    from ausc.perception.robot.robot_state_stream import (
        ExactControlStateStream,
        RobotTelemetryError,
    )

    base = time.monotonic_ns() - 60_000_000

    def wires():
        return (
            _wire_v4(_v4_envelope(
                envelope_seq=1, first_tick=100, first_observation_ns=base)),
            _wire_v4(_v4_envelope(
                envelope_seq=3, first_tick=120,
                first_observation_ns=base + 20_000_000)),
        )

    absorbing = ExactControlStateStream(_contract(), absorb_bounded_holes=True)
    first, second = wires()
    absorbing.ingest_datagram(first)
    absorbing.ingest_datagram(second)      # one missed slot -> absorbed, no raise
    assert not absorbing.faulted

    strict = ExactControlStateStream(_contract())
    first, second = wires()
    strict.ingest_datagram(first)
    with pytest.raises(RobotTelemetryError, match="sequence loss"):
        strict.ingest_datagram(second)
    assert strict.faulted


def test_absorbing_stream_still_rejects_oversized_and_misaligned_holes():
    from ausc.perception.robot.robot_state_stream import (
        ExactControlStateStream,
        RobotTelemetryError,
    )

    base = time.monotonic_ns() - 200_000_000
    # oversized: 7 missed envelopes > tolerance of 5
    stream = ExactControlStateStream(_contract(), absorb_bounded_holes=True)
    stream.ingest_datagram(_wire_v4(_v4_envelope(
        envelope_seq=1, first_tick=100, first_observation_ns=base)))
    with pytest.raises(RobotTelemetryError, match="sequence loss"):
        stream.ingest_datagram(_wire_v4(_v4_envelope(
            envelope_seq=9, first_tick=180,
            first_observation_ns=base + 80_000_000)))
    # misaligned: seq skips but ticks contiguous (renumbering, not a hole)
    stream = ExactControlStateStream(_contract(), absorb_bounded_holes=True)
    stream.ingest_datagram(_wire_v4(_v4_envelope(
        envelope_seq=1, first_tick=100, first_observation_ns=base)))
    with pytest.raises(RobotTelemetryError, match="sequence loss"):
        stream.ingest_datagram(_wire_v4(_v4_envelope(
            envelope_seq=3, first_tick=110,
            first_observation_ns=base + 10_000_000)))
