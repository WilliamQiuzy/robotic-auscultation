from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ausc.perception.robot.camera_exposure_evidence import CameraExposureContract
from ausc.perception.robot.robot_capture_coordinator import (
    ProductionRobotCaptureCoordinator,
    RobotCaptureCoordinatorError,
    RobotCaptureNotReady,
)
from ausc.perception.robot.robot_state_stream import (
    CLOCK_ID,
    CONTROL_ENVELOPE_SCHEMA_V4,
    EXACT_CONTROL_RECORD_SCHEMA_V4,
    FEEDBACK_LIVENESS_REASON,
    FEEDBACK_LIVENESS_SEMANTICS,
    MOTOR_ACK_UNOBSERVABLE_REASON,
    SOCKETCAN_TX_ENQUEUE_SEMANTICS,
    SOCKETCAN_TX_STATUS_REASON,
    RobotStateContract,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_G = "1" * 64
SHA_H = "2" * 64
SHA_I = "3" * 64
SHA_J = "4" * 64
JOINT_NAMES = tuple(f"openarm_left_joint{i}" for i in range(1, 8))
JOINT_LIMITS = tuple((-4.0, 4.0) for _ in range(7))


class _Clock:
    def __init__(self, now_ns: int = 1_005_000_000) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


class _FakeReceiver:
    def __init__(
        self,
        *,
        sink,
        socket_path,
        runtime_dir,
        abort_callback,
        preflight_min_envelopes,
        preflight_duration_ns,
        **_kwargs,
    ) -> None:
        self.sink = sink
        self.contract = sink.contract
        self.socket_path = Path(socket_path)
        self.runtime_dir = runtime_dir
        self.abort_callback = abort_callback
        self.preflight_min_envelopes = preflight_min_envelopes
        self.preflight_duration_ns = preflight_duration_ns
        self.deferred = bool(
            _kwargs.get("deferred_controller_attestation", False)
        )
        self.prepared = False
        self.attestation = _kwargs.get("controller_process_attestation")
        self.started = False
        self.stopped = False
        self.faulted = False
        self.fault_reason = ""
        self.preflight_calls = 0

    def prepare_socket(self) -> None:
        self.prepared = True

    def attach_controller_process(self, attestation) -> None:
        if not self.prepared:
            raise RuntimeError("socket must be prepared first")
        self.attestation = attestation

    def start(self) -> None:
        if self.deferred and (
            not self.prepared or self.attestation is None
        ):
            raise RuntimeError("deferred controller is not attached")
        self.started = True

    def stop(self) -> None:
        self.started = False
        self.stopped = True

    def require_start_preflight(self, *, now_monotonic_ns: int):
        self.preflight_calls += 1
        if not self.started:
            raise RuntimeError("fake receiver is not running")
        return SimpleNamespace(
            status="READY",
            preflight_ready=True,
            pair="left:can1->can3",
            last_age_ns=0,
            packet_count=999,
        )

    def health(self, *, now_monotonic_ns: int):
        return SimpleNamespace(
            status="FAULT" if self.faulted else ("READY" if self.started else "STOPPED"),
            preflight_ready=self.started and not self.faulted,
            pair="left:can1->can3",
            last_age_ns=0,
            packet_count=999,
            error=self.fault_reason,
        )

    def push(self, payload: bytes):
        if not self.started:
            raise RuntimeError("fake receiver is not running")
        return self.sink.ingest_datagram(payload)

    def fail(self, reason: str) -> None:
        if self.faulted:
            return
        self.faulted = True
        self.fault_reason = reason
        self.abort_callback(reason)


class _ReceiverFactory:
    def __init__(self) -> None:
        self.instance = None

    def __call__(self, **kwargs):
        self.instance = _FakeReceiver(**kwargs)
        return self.instance


def _robot_contract(**changes) -> RobotStateContract:
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
        "controller_source_manifest_sha256": SHA_H,
        "follower_config_sha256": SHA_I,
        "openarm_sdk_sha256": SHA_J,
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


def _camera_contract(**changes) -> CameraExposureContract:
    values = {
        "camera_id": "CL8326300A0",
        "camera_latency_sha256": SHA_D,
        "camera_profile_sha256": SHA_C,
        "alignment_sha256": SHA_G,
        "latency_ns": 18_000_000,
        "latency_uncertainty_ns": 6_000_000,
        "capture_mode_json": "{}",
        "measurement_method": "hardware_led_flash",
        "n_samples": 40,
    }
    values.update(changes)
    return CameraExposureContract(**values)


def _record(*, control_tick_seq: int, observation_ns: int) -> dict:
    q_base = (control_tick_seq % 10) * 0.01
    follower = [q_base + index * 0.01 for index in range(7)]
    return {
        "schema": EXACT_CONTROL_RECORD_SCHEMA_V4,
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
        "follower_q_rad": follower,
        "follower_qd_rad_s": [0.5] * 7,
        "follower_gripper_q_rad": -0.070,
        "follower_gripper_qd_rad_s": 0.010,
        "follower_gripper_motor_q_rad": -0.075,
        "follower_gripper_motor_qd_rad_s": 0.015,
        "follower_arm_mit_kp": [240.0] * 7,
        "follower_arm_mit_kd": [3.0] * 7,
        "follower_arm_mit_q": follower,
        "follower_arm_mit_dq": [0.4] * 7,
        "follower_arm_mit_tau": [0.0] * 7,
        "follower_gripper_mit_kp": 25.0,
        "follower_gripper_mit_kd": 0.3,
        "follower_gripper_mit_q": -0.080,
        "follower_gripper_mit_dq": 0.0,
        "follower_gripper_mit_tau": 0.0,
        "arm_tx_expected_frame_count": 7,
        "arm_tx_full_frame_write_count": 7,
        "arm_tx_failed_frame_write_count": 0,
        "gripper_tx_expected_frame_count": 1,
        "gripper_tx_full_frame_write_count": 1,
        "gripper_tx_failed_frame_write_count": 0,
        "socketcan_tx_enqueue_success": True,
        "feedback_snapshot_monotonic_ns": observation_ns,
        "follower_arm_state_rx_motor_ids": list(range(1, 8)),
        "follower_arm_state_rx_counters": [
            control_tick_seq * 8 + index + 1 for index in range(7)
        ],
        "follower_arm_state_rx_generations": [
            control_tick_seq * 8 + index + 1 for index in range(7)
        ],
        "follower_arm_state_rx_monotonic_ns": [
            observation_ns - 500_000 + index for index in range(7)
        ],
        "follower_arm_state_rx_last_gap_ns": [1_000_000] * 7,
        "follower_arm_state_rx_gap_violation_counts": [0] * 7,
        "follower_gripper_state_rx_motor_id": 8,
        "follower_gripper_state_rx_counter": control_tick_seq * 8 + 8,
        "follower_gripper_state_rx_generation": control_tick_seq * 8 + 8,
        "follower_gripper_state_rx_monotonic_ns": observation_ns - 400_000,
        "follower_gripper_state_rx_last_gap_ns": 1_000_000,
        "follower_gripper_state_rx_gap_violation_count": 0,
    }


def _envelope(
    *,
    envelope_seq: int,
    first_tick: int,
    first_observation_ns: int,
    count: int = 10,
) -> dict:
    records = [
        _record(
            control_tick_seq=first_tick + index,
            observation_ns=first_observation_ns + index * 1_000_000,
        )
        for index in range(count)
    ]
    return {
        "schema": CONTROL_ENVELOPE_SCHEMA_V4,
        "envelope_seq": envelope_seq,
        "clock_id": CLOCK_ID,
        "publisher_monotonic_ns": (
            records[-1]["control_call_end_monotonic_ns"] + 500_000
        ),
        "controller_pid": 4242,
        "controller_start_token": "proc-start-991",
        "side": "left",
        "controller_mode": "unilateral",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "joint_names": list(JOINT_NAMES),
        "urdf_sha256": SHA_A,
        "joint_mapper_sha256": SHA_B,
        "pair_contract_sha256": SHA_E,
        "controller_binary_sha256": SHA_F,
        "controller_source_manifest_sha256": SHA_H,
        "follower_config_sha256": SHA_I,
        "openarm_sdk_sha256": SHA_J,
        "publisher_nominal_hz": 100,
        "control_nominal_hz": 1000,
        "send_status_observable": True,
        "send_status_reason": SOCKETCAN_TX_STATUS_REASON,
        "send_status_semantics": SOCKETCAN_TX_ENQUEUE_SEMANTICS,
        "motor_ack_observable": False,
        "motor_ack_reason": MOTOR_ACK_UNOBSERVABLE_REASON,
        "feedback_liveness_observable": True,
        "feedback_liveness_reason": FEEDBACK_LIVENESS_REASON,
        "feedback_liveness_semantics": FEEDBACK_LIVENESS_SEMANTICS,
        "feedback_max_age_ns": 20_000_000,
        "feedback_max_gap_ns": 20_000_000,
        "ring_overflow_count": 0,
        "publisher_rejected_record_count": 0,
        "socket_send_failure_count": 0,
        "records": records,
    }


def _wire(
    *,
    envelope_seq: int,
    first_tick: int,
    first_observation_ns: int,
) -> bytes:
    return json.dumps(
        _envelope(
            envelope_seq=envelope_seq,
            first_tick=first_tick,
            first_observation_ns=first_observation_ns,
        ),
        separators=(",", ":"),
    ).encode("utf-8")


def _new_coordinator(
    tmp_path: Path,
    *,
    callback=None,
    clock: _Clock | None = None,
    deferred: bool = False,
):
    receiver_factory = _ReceiverFactory()
    clock = _Clock() if clock is None else clock
    coordinator = ProductionRobotCaptureCoordinator(
        contract=_robot_contract(),
        camera_exposure_contract=_camera_contract(),
        socket_path=tmp_path / "runtime" / "openarm-left.sock",
        runtime_dir=tmp_path / "runtime",
        receiver_factory=receiver_factory,
        deferred_controller_attestation=deferred,
        abort_callback=callback,
        clock_ns=clock,
        preflight_min_envelopes=3,
        preflight_duration_ns=20_000_000,
        warmup_min_new_envelopes=2,
    )
    assert receiver_factory.instance is not None
    return coordinator, receiver_factory.instance, clock


def _preflight(receiver: _FakeReceiver) -> None:
    for envelope_seq, first_tick, timestamp_ns in (
        (10, 100, 970_000_000),
        (11, 110, 980_000_000),
        (12, 120, 990_000_000),
    ):
        receiver.push(
            _wire(
                envelope_seq=envelope_seq,
                first_tick=first_tick,
                first_observation_ns=timestamp_ns,
            )
        )


def _start(tmp_path: Path, **kwargs):
    coordinator, receiver, clock = _new_coordinator(tmp_path, **kwargs)
    coordinator.start_receiver()
    _preflight(receiver)
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    coordinator.start_session(session_dir)
    return coordinator, receiver, clock, session_dir


def _warm_and_arm(
    coordinator: ProductionRobotCaptureCoordinator,
    receiver: _FakeReceiver,
    clock: _Clock,
):
    for envelope_seq, first_tick, timestamp_ns in (
        (13, 130, 1_010_000_000),
        (14, 140, 1_020_000_000),
        (15, 150, 1_030_000_000),
        (16, 160, 1_040_000_000),
    ):
        receiver.push(
            _wire(
                envelope_seq=envelope_seq,
                first_tick=first_tick,
                first_observation_ns=timestamp_ns,
            )
        )
    clock.now_ns = 1_050_000_000
    return coordinator.authorize_camera_arm()


def test_constructor_accepts_only_left_can1_to_can3_and_matching_camera_timing(
    tmp_path: Path,
) -> None:
    assert (
        ProductionRobotCaptureCoordinator.telemetry_schema
        == CONTROL_ENVELOPE_SCHEMA_V4
    )
    factory = _ReceiverFactory()
    with pytest.raises(ValueError, match="left OpenArm"):
        ProductionRobotCaptureCoordinator(
            contract=replace(
                _robot_contract(),
                side="right",
                joint_names=tuple(f"openarm_right_joint{i}" for i in range(1, 8)),
                leader_can_interface="can0",
                follower_can_interface="can2",
            ),
            camera_exposure_contract=_camera_contract(),
            socket_path=tmp_path / "runtime" / "openarm-left.sock",
            runtime_dir=tmp_path / "runtime",
            receiver_factory=factory,
        )
    with pytest.raises(ValueError, match="camera profile"):
        ProductionRobotCaptureCoordinator(
            contract=_robot_contract(),
            camera_exposure_contract=_camera_contract(camera_profile_sha256="0" * 64),
            socket_path=tmp_path / "runtime" / "openarm-left.sock",
            runtime_dir=tmp_path / "runtime",
            receiver_factory=factory,
        )
    with pytest.raises(ValueError, match="timing artifact"):
        ProductionRobotCaptureCoordinator(
            contract=_robot_contract(),
            camera_exposure_contract=_camera_contract(camera_latency_sha256="0" * 64),
            socket_path=tmp_path / "runtime" / "openarm-left.sock",
            runtime_dir=tmp_path / "runtime",
            receiver_factory=factory,
        )


def test_two_phase_receiver_socket_is_prepared_before_controller_attachment(
    tmp_path: Path,
) -> None:
    coordinator, receiver, _clock = _new_coordinator(
        tmp_path, deferred=True
    )
    with pytest.raises(
        RobotCaptureCoordinatorError, match="attachment failed"
    ):
        coordinator.attach_controller_process(object())
    assert coordinator.faulted

    clean = tmp_path / "clean"
    clean.mkdir()
    coordinator, receiver, _clock = _new_coordinator(
        clean, deferred=True
    )
    coordinator.prepare_receiver_socket()
    authority = object()
    coordinator.attach_controller_process(authority)
    coordinator.start_receiver()

    assert receiver.prepared is True
    assert receiver.attestation is authority
    assert receiver.started is True


def test_start_requires_receiver_and_fresh_continuous_preflight(
    tmp_path: Path,
) -> None:
    coordinator, receiver, _clock = _new_coordinator(tmp_path)
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    with pytest.raises(RobotCaptureNotReady, match="receiver"):
        coordinator.start_session(session_dir)

    coordinator.start_receiver()
    receiver.push(
        _wire(
            envelope_seq=10,
            first_tick=100,
            first_observation_ns=980_000_000,
        )
    )
    receiver.push(
        _wire(
            envelope_seq=11,
            first_tick=110,
            first_observation_ns=990_000_000,
        )
    )
    with pytest.raises(RobotCaptureNotReady, match="preflight"):
        coordinator.start_session(session_dir)
    assert not coordinator.faulted
    assert not tuple(session_dir.iterdir())


def test_live_preview_can_query_exact_robot_bracket_before_recording(
    tmp_path: Path,
) -> None:
    coordinator, receiver, clock = _new_coordinator(tmp_path)
    coordinator.start_receiver()
    _preflight(receiver)
    clock.now_ns = 1_010_000_000

    inputs = coordinator.build_live_frame_evidence_inputs(
        frame_index=5,
        camera_host_t_s=1.010,
        camera_device_t_ms=1200.0,
    )

    assert inputs.camera_exposure.exposure_lower_monotonic_ns == 986_000_000
    assert inputs.camera_exposure.exposure_upper_monotonic_ns == 998_000_000
    assert inputs.robot_trajectory.sequence_first == 116
    assert inputs.robot_trajectory.sequence_last == 128
    assert coordinator.recording is False


def test_live_preview_waits_for_upper_tick_without_fault_or_persistence(
    tmp_path: Path,
) -> None:
    coordinator, receiver, clock = _new_coordinator(tmp_path)
    coordinator.start_receiver()
    _preflight(receiver)
    clock.now_ns = 1_020_000_000

    with pytest.raises(RobotCaptureNotReady, match="upper bracket"):
        coordinator.build_live_frame_evidence_inputs(
            frame_index=6,
            camera_host_t_s=1.020,
            camera_device_t_ms=1201.0,
        )
    assert coordinator.faulted is False

    receiver.push(
        _wire(
            envelope_seq=13,
            first_tick=130,
            first_observation_ns=1_000_000_000,
        )
    )
    inputs = coordinator.build_live_frame_evidence_inputs(
        frame_index=6,
        camera_host_t_s=1.020,
        camera_device_t_ms=1201.0,
    )
    assert inputs.robot_trajectory.sequence_first == 126
    assert inputs.robot_trajectory.sequence_last == 138


def test_start_replays_preflight_then_opens_partials_before_camera_arm(
    tmp_path: Path,
) -> None:
    coordinator, receiver, _clock, session_dir = _start(tmp_path)

    assert coordinator.recording
    assert not coordinator.camera_arm_authorized
    assert receiver.preflight_calls == 1
    for name in (
        "robot_control_envelopes.jsonl.partial",
        "robot_exact_control.csv.partial",
        "robot_frame_index.jsonl.partial",
    ):
        assert (session_dir / name).is_file()
    assert (
        session_dir / "robot_control_envelopes.jsonl.partial"
    ).read_bytes() == b""
    assert not (session_dir / "robot_exact_control.csv").exists()


def test_camera_arm_requires_new_recorded_packets_and_full_camera_preroll(
    tmp_path: Path,
) -> None:
    coordinator, receiver, clock, _session_dir = _start(tmp_path)
    with pytest.raises(RobotCaptureNotReady, match="post-start"):
        coordinator.authorize_camera_arm()

    receiver.push(
        _wire(
            envelope_seq=13,
            first_tick=130,
            first_observation_ns=1_010_000_000,
        )
    )
    receiver.push(
        _wire(
            envelope_seq=14,
            first_tick=140,
            first_observation_ns=1_020_000_000,
        )
    )
    clock.now_ns = 1_030_000_000
    with pytest.raises(RobotCaptureNotReady, match="pre-roll"):
        coordinator.authorize_camera_arm()

    receiver.push(
        _wire(
            envelope_seq=15,
            first_tick=150,
            first_observation_ns=1_030_000_000,
        )
    )
    receiver.push(
        _wire(
            envelope_seq=16,
            first_tick=160,
            first_observation_ns=1_040_000_000,
        )
    )
    clock.now_ns = 1_050_000_000
    permit = coordinator.authorize_camera_arm()

    assert permit.pair == "left:can1->can3"
    assert permit.first_recorded_envelope_seq == 13
    assert permit.last_recorded_envelope_seq == 16
    assert permit.first_recorded_control_tick_seq == 130
    assert permit.last_recorded_control_tick_seq == 169
    assert permit.minimum_preroll_ns == 24_000_000
    assert permit.authorized_monotonic_ns == 1_050_000_000
    assert coordinator.camera_arm_authorized
    assert not coordinator.faulted


def test_frame_build_derives_exposure_and_persists_the_exact_robot_bracket(
    tmp_path: Path,
) -> None:
    coordinator, receiver, clock, session_dir = _start(tmp_path)
    _warm_and_arm(coordinator, receiver, clock)
    receiver.push(
        _wire(
            envelope_seq=17,
            first_tick=170,
            first_observation_ns=1_050_000_000,
        )
    )
    clock.now_ns = 1_060_000_000

    inputs = coordinator.build_frame_evidence_inputs(
        frame_index=7,
        camera_host_t_s=1.060,
        camera_device_t_ms=1234.5,
    )

    assert inputs.camera_exposure.exposure_lower_monotonic_ns == 1_036_000_000
    assert inputs.camera_exposure.exposure_nominal_monotonic_ns == 1_042_000_000
    assert inputs.camera_exposure.exposure_upper_monotonic_ns == 1_048_000_000
    assert inputs.robot_trajectory.sequence_first == 156
    assert inputs.robot_trajectory.sequence_last == 168
    assert inputs.persisted_index["exact_action_tick_first"] == 156
    assert inputs.persisted_index["exact_action_tick_last"] == 168
    assert inputs.persisted_index["exact_action_tick_count"] == 13
    assert len(inputs.robot_trajectory.exact_actions) == 13
    assert (
        inputs.persisted_index["controller_source_manifest_sha256"]
        == SHA_H
    )
    assert inputs.persisted_index["follower_config_sha256"] == SHA_I
    assert inputs.persisted_index["openarm_sdk_sha256"] == SHA_J
    persisted = json.loads(
        (session_dir / "robot_frame_index.jsonl.partial").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == dict(inputs.persisted_index)


def test_frame_waits_retryably_for_upper_bracket_without_faulting_capture(
    tmp_path: Path,
) -> None:
    aborts = []
    coordinator, receiver, clock, session_dir = _start(
        tmp_path, callback=aborts.append
    )
    _warm_and_arm(coordinator, receiver, clock)
    clock.now_ns = 1_070_000_000

    with pytest.raises(RobotCaptureNotReady, match="upper bracket"):
        coordinator.build_frame_evidence_inputs(
            frame_index=8,
            camera_host_t_s=1.070,
            camera_device_t_ms=1235.5,
        )
    assert not coordinator.faulted
    assert coordinator.recording
    assert aborts == []
    assert (
        session_dir / "robot_frame_index.jsonl.partial"
    ).read_bytes() == b""
    with pytest.raises(RobotCaptureNotReady, match="pending camera frame"):
        coordinator.finish_session()
    assert not coordinator.faulted

    receiver.push(
        _wire(
            envelope_seq=17,
            first_tick=170,
            first_observation_ns=1_050_000_000,
        )
    )
    receiver.push(
        _wire(
            envelope_seq=18,
            first_tick=180,
            first_observation_ns=1_060_000_000,
        )
    )
    inputs = coordinator.build_frame_evidence_inputs(
        frame_index=8,
        camera_host_t_s=1.070,
        camera_device_t_ms=1235.5,
    )
    assert inputs.robot_trajectory.sequence_first == 166
    assert inputs.robot_trajectory.sequence_last == 178


def test_end_waits_for_a_frame_and_last_exposure_then_atomically_validates(
    tmp_path: Path,
) -> None:
    coordinator, receiver, clock, session_dir = _start(tmp_path)
    _warm_and_arm(coordinator, receiver, clock)
    with pytest.raises(RobotCaptureNotReady, match="camera frame"):
        coordinator.finish_session()
    assert coordinator.recording

    receiver.push(
        _wire(
            envelope_seq=17,
            first_tick=170,
            first_observation_ns=1_050_000_000,
        )
    )
    clock.now_ns = 1_060_000_000
    inputs = coordinator.build_frame_evidence_inputs(
        frame_index=7,
        camera_host_t_s=1.060,
        camera_device_t_ms=1234.5,
    )
    clock.now_ns = inputs.camera_exposure.exposure_upper_monotonic_ns - 1
    with pytest.raises(RobotCaptureNotReady, match="last camera exposure"):
        coordinator.finish_session()
    assert coordinator.recording

    clock.now_ns = 1_060_000_000
    summary = coordinator.finish_session()
    assert summary["frame_count"] == 1
    assert summary["side"] == "left"
    assert summary["leader_can_interface"] == "can1"
    assert summary["follower_can_interface"] == "can3"
    assert not coordinator.recording
    for name in (
        "robot_control_envelopes.jsonl",
        "robot_exact_control.csv",
        "robot_frame_index.jsonl",
        "robot_capture_summary.json",
    ):
        assert (session_dir / name).is_file()
        assert not (session_dir / f"{name}.partial").exists()


def test_operator_abort_closes_only_the_active_session_and_keeps_receiver_reusable(
    tmp_path: Path,
) -> None:
    coordinator, receiver, _clock, session_dir = _start(tmp_path)

    coordinator.abort_session("camera capture aborted")

    assert coordinator.recording is False
    assert coordinator.faulted is False
    assert receiver.started is True
    assert coordinator.last_summary is None
    for name in (
        "robot_control_envelopes.jsonl",
        "robot_exact_control.csv",
        "robot_frame_index.jsonl",
        "robot_capture_summary.json",
    ):
        assert not (session_dir / name).exists()

    replacement = tmp_path / "replacement-session"
    replacement.mkdir()
    coordinator.start_session(replacement)
    assert coordinator.recording is True


def test_frame_before_camera_arm_is_fatal_and_invokes_abort_once(
    tmp_path: Path,
) -> None:
    aborts = []
    coordinator, receiver, clock, _session_dir = _start(
        tmp_path, callback=aborts.append
    )
    for envelope_seq, first_tick, timestamp_ns in (
        (13, 130, 1_010_000_000),
        (14, 140, 1_020_000_000),
        (15, 150, 1_030_000_000),
    ):
        receiver.push(
            _wire(
                envelope_seq=envelope_seq,
                first_tick=first_tick,
                first_observation_ns=timestamp_ns,
            )
        )
    clock.now_ns = 1_040_000_000

    with pytest.raises(RobotCaptureCoordinatorError, match="camera arm"):
        coordinator.build_frame_evidence_inputs(
            frame_index=0,
            camera_host_t_s=1.040,
            camera_device_t_ms=1.0,
        )
    assert coordinator.faulted
    assert len(aborts) == 1
    with pytest.raises(RobotCaptureCoordinatorError, match="faulted"):
        coordinator.build_frame_evidence_inputs(
            frame_index=1,
            camera_host_t_s=1.040,
            camera_device_t_ms=2.0,
        )
    assert len(aborts) == 1


def test_receiver_or_capture_fault_latches_and_aborts_platform_once(
    tmp_path: Path,
) -> None:
    aborts = []
    coordinator, receiver, _clock, session_dir = _start(
        tmp_path, callback=aborts.append
    )
    receiver.fail("credential PID changed")
    receiver.fail("second fault")

    assert coordinator.faulted
    assert "credential PID changed" in coordinator.fault_reason
    assert len(aborts) == 1
    assert not coordinator.recording
    assert (
        session_dir / "robot_control_envelopes.jsonl.partial"
    ).exists()
    with pytest.raises(RobotCaptureCoordinatorError, match="faulted"):
        receiver.push(
            _wire(
                envelope_seq=13,
                first_tick=130,
                first_observation_ns=1_010_000_000,
            )
        )
    assert len(aborts) == 1


def test_jsonl_unsafe_wire_record_faults_before_it_can_enter_preflight(
    tmp_path: Path,
) -> None:
    aborts = []
    coordinator, receiver, _clock = _new_coordinator(
        tmp_path, callback=aborts.append
    )
    coordinator.start_receiver()

    with pytest.raises(RobotCaptureCoordinatorError, match="JSONL-safe"):
        receiver.push(
            _wire(
                envelope_seq=1,
                first_tick=1,
                first_observation_ns=1_000_000_000,
            )
            + b"\n"
        )
    assert coordinator.faulted
    assert len(aborts) == 1


def test_invalid_camera_frame_faults_capture_and_never_emits_final_summary(
    tmp_path: Path,
) -> None:
    aborts = []
    coordinator, receiver, clock, session_dir = _start(
        tmp_path, callback=aborts.append
    )
    permit = _warm_and_arm(coordinator, receiver, clock)
    receiver.push(
        _wire(
            envelope_seq=17,
            first_tick=170,
            first_observation_ns=1_050_000_000,
        )
    )
    clock.now_ns = 1_060_000_000

    with pytest.raises(RobotCaptureCoordinatorError, match="before camera arm"):
        coordinator.build_frame_evidence_inputs(
            frame_index=0,
            camera_host_t_s=(permit.authorized_monotonic_ns - 1) / 1e9,
            camera_device_t_ms=1.0,
        )
    assert len(aborts) == 1
    assert not (session_dir / "robot_capture_summary.json").exists()


def test_stop_during_capture_is_a_fault_but_idle_stop_is_clean(
    tmp_path: Path,
) -> None:
    aborts = []
    coordinator, receiver, _clock = _new_coordinator(
        tmp_path, callback=aborts.append
    )
    coordinator.start_receiver()
    coordinator.stop_receiver()
    assert receiver.stopped
    assert not coordinator.faulted
    assert aborts == []

    other = tmp_path / "other"
    other.mkdir()
    coordinator2, receiver2, _clock2, _session_dir = _start(
        other, callback=aborts.append
    )
    coordinator2.stop_receiver()
    assert receiver2.stopped
    assert coordinator2.faulted
    assert len(aborts) == 1


def test_finished_summary_hash_inventory_matches_promoted_bytes(
    tmp_path: Path,
) -> None:
    coordinator, receiver, clock, session_dir = _start(tmp_path)
    _warm_and_arm(coordinator, receiver, clock)
    receiver.push(
        _wire(
            envelope_seq=17,
            first_tick=170,
            first_observation_ns=1_050_000_000,
        )
    )
    clock.now_ns = 1_060_000_000
    coordinator.build_frame_evidence_inputs(
        frame_index=0,
        camera_host_t_s=1.060,
        camera_device_t_ms=1.0,
    )
    summary = coordinator.finish_session()

    for name, item in summary["hash_inventory"].items():
        data = (session_dir / name).read_bytes()
        assert item == {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
