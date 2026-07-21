from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ausc.perception.robot.robot_state_stream import (
    CLOCK_ID,
    ROBOT_STATE_SCHEMA,
    RobotStateContract,
    RobotStateStream,
    RobotTelemetryError,
    default_openarm_can_pair,
    parse_robot_state_datagram,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
JOINT_NAMES = tuple(f"openarm_left_joint{i}" for i in range(1, 8))
JOINT_LIMITS = tuple((-2.0, 2.0) for _ in range(7))


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


def _packet(
    *,
    seq: int = 1,
    monotonic_ns: int = 1_000_000_000,
    controller_pid: int = 4242,
    controller_start_token: str = "proc-start-991",
    q_base: float = 0.0,
) -> dict:
    q = [q_base + i * 0.01 for i in range(7)]
    return {
        "schema": ROBOT_STATE_SCHEMA,
        "seq": seq,
        "clock_id": CLOCK_ID,
        "monotonic_ns": monotonic_ns,
        "controller_pid": controller_pid,
        "controller_start_token": controller_start_token,
        "side": "left",
        "controller_mode": "unilateral",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "joint_names": list(JOINT_NAMES),
        "follower_q_rad": q,
        "follower_qd_rad_s": [0.5] * 7,
        "follower_gripper_motor_position_rad": -0.070,
        "leader_q_rad": [value + 0.10 for value in q],
        "leader_qd_rad_s": [0.4] * 7,
        "leader_gripper_motor_position_rad": -0.090,
        "follower_command_q_rad": [value + 0.02 for value in q],
        "follower_command_gripper_motor_position_rad": -0.080,
        "urdf_sha256": SHA_A,
        "joint_mapper_sha256": SHA_B,
        "pair_contract_sha256": SHA_E,
        "controller_binary_sha256": SHA_F,
    }


def _wire(**changes) -> bytes:
    packet = _packet()
    packet.update(changes)
    return json.dumps(packet, allow_nan=True, separators=(",", ":")).encode("utf-8")


def _ingest_sample(
    stream: RobotStateStream,
    *,
    seq: int,
    monotonic_ns: int,
    q_base: float | None = None,
    **changes,
) -> None:
    packet = _packet(
        seq=seq,
        monotonic_ns=monotonic_ns,
        q_base=float(monotonic_ns) / 1_000_000_000 if q_base is None else q_base,
    )
    packet.update(changes)
    stream.ingest_datagram(
        json.dumps(packet, allow_nan=True, separators=(",", ":")).encode("utf-8")
    )


def test_strict_packet_parser_accepts_complete_bound_packet() -> None:
    packet = parse_robot_state_datagram(_wire(), _contract())

    assert packet.schema == ROBOT_STATE_SCHEMA
    assert packet.seq == 1
    assert packet.clock_id == "CLOCK_MONOTONIC"
    assert packet.monotonic_ns == 1_000_000_000
    assert packet.process_identity == (4242, "proc-start-991")
    assert packet.side == "left"
    assert packet.controller_mode == "unilateral"
    assert packet.leader_can_interface == "can1"
    assert packet.follower_can_interface == "can3"
    assert packet.joint_names == JOINT_NAMES
    assert packet.follower_q_rad == pytest.approx(tuple(i * 0.01 for i in range(7)))
    assert packet.follower_qd_rad_s == pytest.approx((0.5,) * 7)
    assert packet.follower_gripper_motor_position_rad == pytest.approx(-0.070)
    assert packet.leader_q_rad == pytest.approx(
        tuple(i * 0.01 + 0.10 for i in range(7))
    )
    assert packet.leader_qd_rad_s == pytest.approx((0.4,) * 7)
    assert packet.leader_gripper_motor_position_rad == pytest.approx(-0.090)
    assert packet.follower_command_q_rad == pytest.approx(
        tuple(i * 0.01 + 0.02 for i in range(7))
    )
    assert packet.follower_command_gripper_motor_position_rad == pytest.approx(-0.080)
    assert packet.urdf_sha256 == SHA_A
    assert packet.joint_mapper_sha256 == SHA_B
    assert packet.pair_contract_sha256 == SHA_E
    assert packet.controller_binary_sha256 == SHA_F


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"schema": "wam_robot_state_v0"}, "schema"),
        ({"clock_id": "CLOCK_REALTIME"}, "CLOCK_MONOTONIC"),
        ({"seq": True}, "seq"),
        ({"seq": -1}, "seq"),
        ({"monotonic_ns": 1.0}, "monotonic_ns"),
        ({"controller_pid": 0}, "controller_pid"),
        ({"controller_start_token": ""}, "start token"),
        ({"controller_start_token": "bad token"}, "start token"),
        ({"side": "right"}, "side"),
        ({"controller_mode": "gravity_comp"}, "unilateral"),
        ({"leader_can_interface": "can0"}, "leader CAN"),
        ({"follower_can_interface": "can2"}, "follower CAN"),
        ({"leader_can_interface": "can3"}, "different"),
        ({"joint_names": list(reversed(JOINT_NAMES))}, "joint names"),
        ({"follower_q_rad": [0.0] * 6}, "follower_q_rad"),
        ({"follower_qd_rad_s": [0.0] * 8}, "follower_qd_rad_s"),
        ({"leader_q_rad": [0.0] * 6}, "leader_q_rad"),
        ({"leader_qd_rad_s": [0.0] * 8}, "leader_qd_rad_s"),
        ({"follower_command_q_rad": [0.0] * 6}, "follower_command_q_rad"),
        ({"urdf_sha256": "0" * 64}, "URDF"),
        ({"joint_mapper_sha256": "0" * 64}, "joint mapper"),
        ({"pair_contract_sha256": "0" * 64}, "pair contract"),
        ({"controller_binary_sha256": "0" * 64}, "controller binary"),
    ),
)
def test_strict_packet_parser_rejects_protocol_or_provenance_mismatch(
    changes: dict, message: str,
) -> None:
    with pytest.raises(RobotTelemetryError, match=message):
        parse_robot_state_datagram(_wire(**changes), _contract())


def test_strict_packet_parser_rejects_missing_extra_duplicate_and_oversize_fields() -> None:
    missing = _packet()
    missing.pop("follower_command_gripper_motor_position_rad")
    with pytest.raises(RobotTelemetryError, match="fields"):
        parse_robot_state_datagram(json.dumps(missing).encode(), _contract())

    with pytest.raises(RobotTelemetryError, match="fields"):
        parse_robot_state_datagram(_wire(untrusted=True), _contract())

    duplicate = _wire().decode("utf-8").replace(
        '"seq":1,', '"seq":1,"seq":2,', 1
    )
    with pytest.raises(RobotTelemetryError, match="duplicate"):
        parse_robot_state_datagram(duplicate.encode(), _contract())

    with pytest.raises(RobotTelemetryError, match="too large"):
        parse_robot_state_datagram(b" " * 16_385, _contract())


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"follower_q_rad": [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0, 0.0]}, "finite"),
        ({"follower_q_rad": [0.0, 0.0, 2.01, 0.0, 0.0, 0.0, 0.0]}, "joint limits"),
        ({"follower_command_q_rad": [0.0, 0.0, -2.01, 0.0, 0.0, 0.0, 0.0]}, "joint limits"),
        ({"leader_q_rad": [0.0, 0.0, 2.01, 0.0, 0.0, 0.0, 0.0]}, "joint limits"),
        ({"follower_qd_rad_s": [0.0, 0.0, 8.01, 0.0, 0.0, 0.0, 0.0]}, "velocity"),
        ({"leader_qd_rad_s": [0.0, 0.0, -8.01, 0.0, 0.0, 0.0, 0.0]}, "velocity"),
        ({"follower_gripper_motor_position_rad": float("inf")}, "finite"),
        ({"follower_gripper_motor_position_rad": 0.201}, "gripper"),
        ({"leader_gripper_motor_position_rad": -0.201}, "gripper"),
        ({"follower_command_gripper_motor_position_rad": 0.201}, "gripper"),
    ),
)
def test_parser_fails_closed_on_nonfinite_or_out_of_limit_state(
    changes: dict, message: str,
) -> None:
    with pytest.raises(RobotTelemetryError, match=message):
        parse_robot_state_datagram(_wire(**changes), _contract())


def test_interval_lookup_brackets_entire_exposure_and_interpolates_trajectory() -> None:
    stream = RobotStateStream(_contract())
    for seq, timestamp_ns in enumerate(
        (980_000_000, 990_000_000, 1_000_000_000, 1_010_000_000, 1_020_000_000),
        start=10,
    ):
        _ingest_sample(stream, seq=seq, monotonic_ns=timestamp_ns)

    trajectory = stream.query_exposure_interval(
        exposure_lower_ns=995_000_000,
        exposure_nominal_ns=1_003_000_000,
        exposure_upper_ns=1_015_000_000,
        now_monotonic_ns=1_020_000_000,
        camera_profile_sha256=SHA_C,
        timing_artifact_sha256=SHA_D,
    )

    assert trajectory.before.monotonic_ns == 990_000_000
    assert trajectory.after.monotonic_ns == 1_020_000_000
    assert trajectory.sequence_first == 11
    assert trajectory.sequence_last == 14
    assert trajectory.max_gap_ns == 10_000_000
    assert trajectory.at_lower.follower_q_rad[0] == pytest.approx(0.995)
    assert trajectory.at_nominal.follower_q_rad[0] == pytest.approx(1.003)
    assert trajectory.at_upper.follower_q_rad[0] == pytest.approx(1.015)

    exposure_samples = trajectory.sample_exposure_trajectory()
    assert tuple(sample.monotonic_ns for sample in exposure_samples) == (
        995_000_000,
        1_000_000_000,
        1_003_000_000,
        1_010_000_000,
        1_015_000_000,
    )
    # v1 references are diagnostic-only and are never linearly interpolated as
    # if they were consumed actions.
    assert exposure_samples[0].follower_command_q_rad is None
    assert exposure_samples[0].action_is_exact is False
    assert exposure_samples[1].follower_command_q_rad[0] == pytest.approx(1.02)
    assert exposure_samples[1].action_is_exact is True
    assert exposure_samples[0].leader_q_rad[0] == pytest.approx(1.095)
    assert exposure_samples[-1].follower_gripper_motor_position_rad == pytest.approx(
        -0.070
    )
    assert exposure_samples[-1].leader_gripper_motor_position_rad == pytest.approx(
        -0.090
    )
    assert exposure_samples[-1].follower_command_gripper_motor_position_rad is None

    arbitrary = trajectory.sample_at(1_007_500_000)
    assert arbitrary.follower_q_rad[0] == pytest.approx(1.0075)
    with pytest.raises(RobotTelemetryError, match="extrapolation"):
        trajectory.sample_at(989_999_999)
    with pytest.raises(RobotTelemetryError, match="extrapolation"):
        trajectory.sample_at(1_020_000_001)


@pytest.mark.parametrize(
    ("lower", "nominal", "upper"),
    (
        (970_000_000, 980_000_000, 990_000_000),
        (1_010_000_000, 1_020_000_000, 1_030_000_000),
    ),
)
def test_interval_lookup_never_extrapolates(
    lower: int, nominal: int, upper: int,
) -> None:
    stream = RobotStateStream(_contract())
    for seq, timestamp_ns in enumerate((980_000_000, 1_000_000_000, 1_020_000_000), start=1):
        _ingest_sample(stream, seq=seq, monotonic_ns=timestamp_ns)

    with pytest.raises(RobotTelemetryError, match="bracket"):
        stream.query_exposure_interval(
            exposure_lower_ns=lower,
            exposure_nominal_ns=nominal,
            exposure_upper_ns=upper,
            now_monotonic_ns=1_020_000_000,
            camera_profile_sha256=SHA_C,
            timing_artifact_sha256=SHA_D,
        )


def test_interval_lookup_rejects_any_gap_intersecting_exposure() -> None:
    stream = RobotStateStream(_contract())
    for seq, timestamp_ns in enumerate(
        (970_000_000, 990_000_000, 1_011_000_000, 1_020_000_000), start=1
    ):
        _ingest_sample(stream, seq=seq, monotonic_ns=timestamp_ns)

    with pytest.raises(RobotTelemetryError, match="gap"):
        stream.query_exposure_interval(
            exposure_lower_ns=995_000_000,
            exposure_nominal_ns=1_000_000_000,
            exposure_upper_ns=1_015_000_000,
            now_monotonic_ns=1_020_000_000,
            camera_profile_sha256=SHA_C,
            timing_artifact_sha256=SHA_D,
        )


def test_exact_20_ms_gap_is_allowed_but_next_nanosecond_is_rejected() -> None:
    exact = RobotStateStream(_contract())
    for seq, timestamp_ns in enumerate(
        (980_000_000, 1_000_000_000, 1_020_000_000), start=1
    ):
        _ingest_sample(exact, seq=seq, monotonic_ns=timestamp_ns)
    trajectory = exact.query_exposure_interval(
        exposure_lower_ns=995_000_000,
        exposure_nominal_ns=1_000_000_000,
        exposure_upper_ns=1_005_000_000,
        now_monotonic_ns=1_020_000_000,
        camera_profile_sha256=SHA_C,
        timing_artifact_sha256=SHA_D,
    )
    assert trajectory.max_gap_ns == 20_000_000

    too_wide = RobotStateStream(_contract())
    for seq, timestamp_ns in enumerate(
        (980_000_000, 1_000_000_001, 1_020_000_001), start=1
    ):
        _ingest_sample(too_wide, seq=seq, monotonic_ns=timestamp_ns)
    with pytest.raises(RobotTelemetryError, match="gap"):
        too_wide.query_exposure_interval(
            exposure_lower_ns=995_000_000,
            exposure_nominal_ns=1_000_000_000,
            exposure_upper_ns=1_005_000_000,
            now_monotonic_ns=1_020_000_001,
            camera_profile_sha256=SHA_C,
            timing_artifact_sha256=SHA_D,
        )


def test_exact_instant_lookup_is_bracketed_without_an_empty_gap_reduction() -> None:
    stream = RobotStateStream(_contract())
    _ingest_sample(stream, seq=1, monotonic_ns=1_000_000_000)

    trajectory = stream.query_exposure_interval(
        exposure_lower_ns=1_000_000_000,
        exposure_nominal_ns=1_000_000_000,
        exposure_upper_ns=1_000_000_000,
        now_monotonic_ns=1_000_000_000,
        camera_profile_sha256=SHA_C,
        timing_artifact_sha256=SHA_D,
    )

    assert trajectory.before is trajectory.after
    assert trajectory.max_gap_ns == 0
    assert trajectory.at_nominal.follower_q_rad[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("second_changes", "message"),
    (
        ({"seq": 12}, "sequence"),
        ({"controller_pid": 4243}, "process identity"),
        ({"controller_start_token": "proc-start-992"}, "process identity"),
        ({"monotonic_ns": 999_999_999}, "clock rollback"),
    ),
)
def test_stream_latches_fault_on_loss_process_change_or_clock_rollback(
    second_changes: dict, message: str,
) -> None:
    stream = RobotStateStream(_contract())
    _ingest_sample(stream, seq=10, monotonic_ns=1_000_000_000)
    packet = _packet(seq=11, monotonic_ns=1_010_000_000, q_base=1.01)
    packet.update(second_changes)

    with pytest.raises(RobotTelemetryError, match=message):
        stream.ingest_datagram(json.dumps(packet).encode())

    assert stream.faulted is True
    assert message in stream.fault_reason
    with pytest.raises(RobotTelemetryError, match="faulted"):
        stream.query_exposure_interval(
            exposure_lower_ns=1_000_000_000,
            exposure_nominal_ns=1_000_000_000,
            exposure_upper_ns=1_000_000_000,
            now_monotonic_ns=1_010_000_000,
            camera_profile_sha256=SHA_C,
            timing_artifact_sha256=SHA_D,
        )


def test_malformed_packet_latches_the_stream_fail_closed() -> None:
    stream = RobotStateStream(_contract())

    with pytest.raises(RobotTelemetryError, match="JSON"):
        stream.ingest_datagram(b"{not json")

    assert stream.faulted
    with pytest.raises(RobotTelemetryError, match="faulted"):
        stream.ingest_datagram(_wire())


def test_two_second_buffer_is_time_and_capacity_bounded() -> None:
    contract = _contract(max_packets=205)
    stream = RobotStateStream(contract)
    start_ns = 5_000_000_000
    for index in range(401):
        _ingest_sample(
            stream,
            seq=index,
            monotonic_ns=start_ns + index * 10_000_000,
            q_base=0.0,
        )

    assert stream.packet_count == 201
    assert stream.oldest_monotonic_ns == 7_000_000_000
    assert stream.latest_monotonic_ns == 9_000_000_000
    assert stream.latest_monotonic_ns - stream.oldest_monotonic_ns == 2_000_000_000


def test_stale_buffer_profile_or_timing_mismatch_and_wide_interval_fail_closed() -> None:
    stream = RobotStateStream(_contract())
    for seq, timestamp_ns in enumerate((990_000_000, 1_000_000_000, 1_010_000_000), start=1):
        _ingest_sample(stream, seq=seq, monotonic_ns=timestamp_ns)

    base = {
        "exposure_lower_ns": 995_000_000,
        "exposure_nominal_ns": 1_000_000_000,
        "exposure_upper_ns": 1_005_000_000,
        "now_monotonic_ns": 1_010_000_000,
        "camera_profile_sha256": SHA_C,
        "timing_artifact_sha256": SHA_D,
    }
    with pytest.raises(RobotTelemetryError, match="stale"):
        stream.query_exposure_interval(**{**base, "now_monotonic_ns": 1_060_000_001})
    with pytest.raises(RobotTelemetryError, match="camera profile"):
        stream.query_exposure_interval(**{**base, "camera_profile_sha256": "e" * 64})
    with pytest.raises(RobotTelemetryError, match="timing artifact"):
        stream.query_exposure_interval(**{**base, "timing_artifact_sha256": "e" * 64})
    with pytest.raises(RobotTelemetryError, match="width"):
        stream.query_exposure_interval(
            **{
                **base,
                "exposure_lower_ns": 960_000_000,
                "exposure_nominal_ns": 980_000_000,
                "exposure_upper_ns": 1_005_000_001,
            }
        )


def test_contract_rejects_noncanonical_joint_order_and_invalid_limits() -> None:
    with pytest.raises(ValueError, match="canonical"):
        replace(_contract(), joint_names=tuple(reversed(JOINT_NAMES)))
    with pytest.raises(ValueError, match="limits"):
        replace(
            _contract(),
            joint_position_limits_rad=JOINT_LIMITS[:-1] + ((2.0, -2.0),),
        )
    with pytest.raises(ValueError, match="201"):
        replace(_contract(), max_packets=200)


def test_contract_binds_left_pair_and_exposes_both_upstream_defaults() -> None:
    assert default_openarm_can_pair("right") == ("can0", "can2")
    assert default_openarm_can_pair("left") == ("can1", "can3")
    with pytest.raises(ValueError, match="side"):
        default_openarm_can_pair("front")
    with pytest.raises(ValueError, match="different"):
        replace(_contract(), follower_can_interface="can1")
    with pytest.raises(ValueError, match="can0"):
        replace(_contract(), leader_can_interface="vcan0")

    right_joint_names = tuple(f"openarm_right_joint{i}" for i in range(1, 8))
    right_contract = replace(
        _contract(),
        side="right",
        joint_names=right_joint_names,
        leader_can_interface="can0",
        follower_can_interface="can2",
    )
    packet = _packet()
    packet["side"] = "right"
    packet["joint_names"] = list(right_joint_names)
    packet["leader_can_interface"] = "can0"
    packet["follower_can_interface"] = "can2"
    parsed = parse_robot_state_datagram(
        json.dumps(packet, separators=(",", ":")).encode(),
        right_contract,
    )
    assert parsed.leader_can_interface == "can0"
    assert parsed.follower_can_interface == "can2"


def test_100_hz_synthetic_soak_keeps_gap_free_interpolation_available() -> None:
    stream = RobotStateStream(_contract())
    base_ns = 10_000_000_000
    for index in range(1_001):
        _ingest_sample(
            stream,
            seq=20_000 + index,
            monotonic_ns=base_ns + index * 10_000_000,
            q_base=float(index % 100) / 100.0,
        )
        if index >= 4:
            latest = base_ns + index * 10_000_000
            trajectory = stream.query_exposure_interval(
                exposure_lower_ns=latest - 35_000_000,
                exposure_nominal_ns=latest - 25_000_000,
                exposure_upper_ns=latest - 15_000_000,
                now_monotonic_ns=latest,
                camera_profile_sha256=SHA_C,
                timing_artifact_sha256=SHA_D,
            )
            assert trajectory.max_gap_ns == 10_000_000
            assert len(trajectory.sample_exposure_trajectory()) >= 4

    assert stream.packet_count <= 201
    assert stream.latest_monotonic_ns - stream.oldest_monotonic_ns <= 2_000_000_000
