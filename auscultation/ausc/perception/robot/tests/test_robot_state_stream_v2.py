from __future__ import annotations

import json

import pytest

from ausc.perception.robot.robot_state_stream import (
    CLOCK_ID,
    CONTROL_ENVELOPE_SCHEMA,
    CONTROL_ENVELOPE_SCHEMA_V3,
    EXACT_CONTROL_RECORD_SCHEMA,
    EXACT_CONTROL_RECORD_SCHEMA_V3,
    MOTOR_ACK_UNOBSERVABLE_REASON,
    SOCKETCAN_TX_ENQUEUE_SEMANTICS,
    SOCKETCAN_TX_STATUS_REASON,
    VOID_SEND_STATUS_REASON,
    ExactControlStateStream,
    RobotStateContract,
    RobotTelemetryError,
    parse_control_envelope_datagram,
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


def _contract(**changes) -> RobotStateContract:
    values = {
        "side": "left",
        "controller_mode": "unilateral",
        "joint_names": JOINT_NAMES,
        "joint_position_limits_rad": tuple((-2.0, 2.0) for _ in range(7)),
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "urdf_sha256": SHA_A,
        "joint_mapper_sha256": SHA_B,
        "camera_profile_sha256": SHA_C,
        "timing_artifact_sha256": SHA_D,
        "pair_contract_sha256": SHA_E,
        "controller_binary_sha256": SHA_F,
        "max_abs_joint_velocity_rad_s": 8.0,
        "gripper_motor_position_limits_rad": (-0.20, 0.20),
        "max_staleness_ns": 50_000_000,
        "max_exposure_interval_ns": 40_000_000,
        "controller_source_manifest_sha256": SHA_1,
        "follower_config_sha256": SHA_2,
        "openarm_sdk_sha256": SHA_3,
    }
    values.update(changes)
    return RobotStateContract(**values)


def _record(control_tick_seq: int, observation_ns: int) -> dict:
    base = float(control_tick_seq) / 1000.0
    q = [base + index * 0.01 for index in range(7)]
    return {
        "schema": EXACT_CONTROL_RECORD_SCHEMA,
        "control_tick_seq": control_tick_seq,
        "observation_monotonic_ns": observation_ns,
        "reference_generation": 17,
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
        "follower_arm_mit_kp": [240.0, 240.0, 240.0, 240.0, 40.0, 48.0, 40.0],
        "follower_arm_mit_kd": [3.0, 3.0, 3.0, 3.0, 0.3, 0.3, 0.3],
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
    envelope_seq: int = 3,
    first_tick: int = 100,
    first_observation_ns: int = 1_000_000_000,
    count: int = 3,
) -> dict:
    records = [
        _record(first_tick + index, first_observation_ns + index * 1_000_000)
        for index in range(count)
    ]
    return {
        "schema": CONTROL_ENVELOPE_SCHEMA,
        "envelope_seq": envelope_seq,
        "clock_id": CLOCK_ID,
        "publisher_monotonic_ns": (
            records[-1]["control_call_end_monotonic_ns"] + 1_000_000
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


def _wire(value: dict) -> bytes:
    return json.dumps(
        value, allow_nan=True, separators=(",", ":")
    ).encode("utf-8")


def _v3_envelope(**changes) -> dict:
    value = _envelope()
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
    value.update(changes)
    return value


def test_v2_parser_preserves_exact_consumed_mit_params_and_void_return_limit() -> None:
    envelope = parse_control_envelope_datagram(_wire(_envelope()), _contract())

    assert envelope.schema == CONTROL_ENVELOPE_SCHEMA
    assert envelope.envelope_seq == 3
    assert [record.control_tick_seq for record in envelope.records] == [100, 101, 102]
    record = envelope.records[0]
    assert record.action_parameters_exact is True
    assert record.action.arm_kp == pytest.approx(
        (240.0, 240.0, 240.0, 240.0, 40.0, 48.0, 40.0)
    )
    assert record.action.arm_kd == pytest.approx(
        (3.0, 3.0, 3.0, 3.0, 0.3, 0.3, 0.3)
    )
    assert record.action.arm_q == pytest.approx(record.follower_q_rad)
    assert record.action.arm_dq == pytest.approx((0.4,) * 7)
    assert record.action.arm_tau == pytest.approx((0.0,) * 7)
    assert record.action.gripper_kp == pytest.approx(25.0)
    assert record.action.gripper_q == pytest.approx(-0.080)
    assert record.action_send_status_verified is False
    with pytest.raises(RobotTelemetryError, match="unobservable"):
        record.require_verified_send_status()


def test_v3_parser_proves_only_full_socketcan_tx_enqueue_for_every_expected_frame() -> None:
    envelope = parse_control_envelope_datagram(
        _wire(_v3_envelope()), _contract()
    )

    assert envelope.schema == CONTROL_ENVELOPE_SCHEMA_V3
    record = envelope.records[0]
    assert record.schema == EXACT_CONTROL_RECORD_SCHEMA_V3
    assert record.arm_tx_expected_frame_count == 7
    assert record.arm_tx_full_frame_write_count == 7
    assert record.arm_tx_failed_frame_write_count == 0
    assert record.gripper_tx_expected_frame_count == 1
    assert record.gripper_tx_full_frame_write_count == 1
    assert record.gripper_tx_failed_frame_write_count == 0
    assert record.socketcan_tx_enqueue_success is True
    assert record.action_send_status_verified is True
    assert record.send_status_semantics == SOCKETCAN_TX_ENQUEUE_SEMANTICS
    assert record.motor_ack_observable is False
    record.require_verified_send_status()


def test_v3_parser_retains_failed_tick_but_never_marks_it_send_verified() -> None:
    value = _v3_envelope()
    value["records"][1].update(
        arm_tx_full_frame_write_count=6,
        arm_tx_failed_frame_write_count=1,
        socketcan_tx_enqueue_success=False,
    )

    envelope = parse_control_envelope_datagram(_wire(value), _contract())
    failed = envelope.records[1]
    assert failed.action_send_status_verified is False
    with pytest.raises(RobotTelemetryError, match="SocketCAN.*enqueue"):
        failed.require_verified_send_status()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["records"][0].update(
                arm_tx_expected_frame_count=6
            ),
            "seven arm",
        ),
        (
            lambda value: value["records"][0].update(
                arm_tx_full_frame_write_count=6
            ),
            "write counts",
        ),
        (
            lambda value: value["records"][0].update(
                socketcan_tx_enqueue_success=False
            ),
            "success flag",
        ),
        (
            lambda value: value.update(motor_ack_observable=True),
            "motor acknowledgement",
        ),
        (
            lambda value: value.update(
                send_status_semantics="motor_acknowledged"
            ),
            "semantics",
        ),
    ),
)
def test_v3_parser_rejects_inconsistent_or_overclaimed_tx_status(
    mutation, message: str
) -> None:
    value = _v3_envelope()
    mutation(value)
    with pytest.raises(RobotTelemetryError, match=message):
        parse_control_envelope_datagram(_wire(value), _contract())


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(send_status_observable=True), "send status"),
        (lambda value: value.update(ring_overflow_count=1), "ring_overflow"),
        (
            lambda value: value.update(publisher_rejected_record_count=1),
            "publisher_rejected",
        ),
        (
            lambda value: value.update(socket_send_failure_count=1),
            "socket_send_failure",
        ),
        (
            lambda value: value["records"][0].update(
                reference_snapshot_valid=False
            ),
            "reference snapshot",
        ),
        (
            lambda value: value["records"][0].update(
                arm_call_return_monotonic_ns=999_000_000
            ),
            "timestamp ordering",
        ),
        (
            lambda value: value["records"][1].update(control_tick_seq=104),
            "sequence-contiguous",
        ),
    ),
)
def test_v2_parser_fails_closed_on_loss_provenance_or_timing_drift(
    mutation, message: str
) -> None:
    value = _envelope()
    mutation(value)
    with pytest.raises(RobotTelemetryError, match=message):
        parse_control_envelope_datagram(_wire(value), _contract())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "follower_arm_mit_q",
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "MIT action joint 1.*outside activated joint limits",
        ),
        (
            "follower_arm_mit_dq",
            [9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "MIT action joint velocity",
        ),
        (
            "follower_arm_mit_kp",
            [-1.0, 240.0, 240.0, 240.0, 40.0, 48.0, 40.0],
            "MIT action stiffness",
        ),
        (
            "follower_arm_mit_kd",
            [-0.1, 3.0, 3.0, 3.0, 0.3, 0.3, 0.3],
            "MIT action damping",
        ),
        ("follower_gripper_mit_kp", -1.0, "gripper MIT stiffness"),
        ("follower_gripper_mit_kd", -0.1, "gripper MIT damping"),
    ),
)
def test_v2_parser_rejects_physically_invalid_exact_mit_action(
    field: str, value: object, message: str
) -> None:
    envelope = _envelope()
    envelope["records"][0][field] = value

    with pytest.raises(RobotTelemetryError, match=message):
        parse_control_envelope_datagram(_wire(envelope), _contract())


def test_v2_requires_full_source_config_and_sdk_hash_closure() -> None:
    for field in (
        "controller_source_manifest_sha256",
        "follower_config_sha256",
        "openarm_sdk_sha256",
    ):
        with pytest.raises(RobotTelemetryError, match="requires activated"):
            parse_control_envelope_datagram(
                _wire(_envelope()), _contract(**{field: ""})
            )


def test_v2_stream_brackets_observation_but_never_interpolates_action() -> None:
    stream = ExactControlStateStream(_contract())
    stream.ingest_datagram(
        _wire(
            _envelope(
                envelope_seq=10,
                first_tick=100,
                first_observation_ns=1_000_000_000,
                count=3,
            )
        )
    )
    stream.ingest_datagram(
        _wire(
            _envelope(
                envelope_seq=11,
                first_tick=103,
                first_observation_ns=1_003_000_000,
                count=3,
            )
        )
    )

    trajectory = stream.query_exposure_interval(
        exposure_lower_ns=1_001_500_000,
        exposure_nominal_ns=1_002_500_000,
        exposure_upper_ns=1_004_500_000,
        now_monotonic_ns=1_006_000_000,
        camera_profile_sha256=SHA_C,
        timing_artifact_sha256=SHA_D,
    )

    assert trajectory.sequence_first == 101
    assert trajectory.sequence_last == 105
    observation = trajectory.sample_observation_at(1_002_500_000)
    assert observation.follower_q_rad[0] == pytest.approx(0.1025)
    assert not hasattr(observation, "action")
    assert [tick for tick, _ in trajectory.exact_actions] == [101, 102, 103, 104, 105]
    assert trajectory.exact_action_for_tick(103).arm_q[0] == pytest.approx(0.103)
    with pytest.raises(RobotTelemetryError, match="no exact action"):
        trajectory.exact_action_for_tick(999)


def test_v2_stream_latches_envelope_or_control_tick_loss() -> None:
    stream = ExactControlStateStream(_contract())
    stream.ingest_datagram(_wire(_envelope(envelope_seq=1, first_tick=10)))
    with pytest.raises(RobotTelemetryError, match="envelope sequence"):
        stream.ingest_datagram(
            _wire(_envelope(envelope_seq=3, first_tick=13))
        )
    assert stream.faulted is True


def test_exact_control_stream_rejects_v2_to_v3_schema_switch() -> None:
    stream = ExactControlStateStream(_contract())
    stream.ingest_datagram(
        _wire(_envelope(envelope_seq=1, first_tick=10, count=3))
    )
    switched = _v3_envelope()
    switched["envelope_seq"] = 2
    switched["records"] = [
        {
            **record,
            "control_tick_seq": 13 + index,
            "observation_monotonic_ns": 1_003_000_000 + index * 1_000_000,
            "reference_source_monotonic_ns": 1_002_900_000
            + index * 1_000_000,
            "control_call_begin_monotonic_ns": 1_003_050_000
            + index * 1_000_000,
            "arm_call_return_monotonic_ns": 1_003_100_000
            + index * 1_000_000,
            "gripper_call_begin_monotonic_ns": 1_003_110_000
            + index * 1_000_000,
            "gripper_call_return_monotonic_ns": 1_003_160_000
            + index * 1_000_000,
            "control_call_end_monotonic_ns": 1_003_170_000
            + index * 1_000_000,
        }
        for index, record in enumerate(switched["records"])
    ]
    switched["publisher_monotonic_ns"] = 1_006_500_000
    with pytest.raises(RobotTelemetryError, match="schema changed"):
        stream.ingest_datagram(_wire(switched))
