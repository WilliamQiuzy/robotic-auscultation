from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from ausc.perception.robot import robot_session_capture as capture_module
from ausc.perception.robot.robot_session_capture import (
    EXACT_CAPTURE_SUMMARY_SCHEMA_V3_PRODUCTION,
    EXACT_CAPTURE_SUMMARY_SCHEMA_V4_PRODUCTION,
    EXACT_FRAME_INDEX_SCHEMA_V3_PRODUCTION,
    EXACT_FRAME_INDEX_SCHEMA_V4_PRODUCTION,
    RobotCaptureError,
    RobotSessionCapture,
    validate_robot_capture_artifacts,
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
    MOTOR_ACK_UNOBSERVABLE_REASON,
    SOCKETCAN_TX_ENQUEUE_SEMANTICS,
    SOCKETCAN_TX_STATUS_REASON,
    VOID_SEND_STATUS_REASON,
    RobotStateContract,
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
        "joint_position_limits_rad": tuple((-4.0, 4.0) for _ in range(7)),
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
    envelope_seq: int,
    first_tick: int,
    first_observation_ns: int,
    count: int = 10,
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
    value.update(changes)
    return value


def _wire(value: dict, *, pretty: bool = False) -> bytes:
    if pretty:
        return json.dumps(value, separators=(", ", ": "), sort_keys=False).encode("utf-8")
    return json.dumps(value, separators=(",", ":"), sort_keys=False).encode("utf-8")


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


def _v4_envelope(
    *,
    frozen_q: bool = False,
    all_zero_q: bool = False,
    near_zero_q: bool = False,
    noisy_obs_static_command: bool = False,
    hold_joint_index: int | None = None,
    arm_response_fault: tuple[int, str] | None = None,
    gripper_response: str = "stationary",
    frozen_feedback: tuple[list[int], list[int], int, int] | None = None,
    gap_violation_count: int = 5,
    stale_by_ns: int = 0,
    **kwargs,
) -> dict:
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
        # `_record` intentionally uses equal values, but response-fault tests
        # must not alias the observation list to the exact action list.
        record["follower_q_rad"] = list(record["follower_q_rad"])
        if all_zero_q:
            record["follower_q_rad"] = [0.0] * 7
            record["follower_qd_rad_s"] = [0.0] * 7
            record["follower_arm_mit_q"] = [0.0] * 7
            record["follower_arm_mit_dq"] = [0.0] * 7
        elif near_zero_q:
            values = [
                0.25 + index * 0.01 + tick * 1.0e-7
                for index in range(7)
            ]
            record["follower_q_rad"] = values
            record["follower_qd_rad_s"] = [0.0] * 7
            record["follower_arm_mit_q"] = list(values)
            record["follower_arm_mit_dq"] = [0.0] * 7
        elif noisy_obs_static_command:
            # Observation drifts well above the 1e-4 range floor (passive motion /
            # sensor noise) but the commanded action is held constant, so no action
            # channel materially moves -> the session must be rejected.
            record["follower_q_rad"] = [
                0.25 + index * 0.01 + tick * 1.0e-4 for index in range(7)
            ]
            record["follower_qd_rad_s"] = [0.0] * 7
            record["follower_arm_mit_q"] = [
                0.25 + index * 0.01 for index in range(7)
            ]
            record["follower_arm_mit_dq"] = [0.0] * 7
        elif frozen_q:
            record["follower_q_rad"] = [0.25 + index * 0.01 for index in range(7)]
            record["follower_qd_rad_s"] = [0.0] * 7
        if hold_joint_index is not None:
            record["follower_q_rad"][hold_joint_index] = 0.25
            record["follower_qd_rad_s"][hold_joint_index] = 0.0
            record["follower_arm_mit_q"][hold_joint_index] = 0.25
            record["follower_arm_mit_dq"][hold_joint_index] = 0.0
        if arm_response_fault is not None:
            joint_index, fault = arm_response_fault
            if fault == "frozen":
                observed = 0.25
            elif fault == "opposite":
                observed = 0.50 - tick * 1.0e-3
            elif fault == "weak":
                observed = 0.25 + tick * 1.0e-5
            elif fault == "uncorrelated":
                observed = 0.25 + (0.02 if tick % 4 in (0, 1) else -0.02)
            else:
                raise AssertionError(f"unknown arm response fault: {fault}")
            record["follower_q_rad"][joint_index] = observed
            record["follower_qd_rad_s"][joint_index] = 0.0
        if gripper_response != "stationary":
            command = -0.16 + tick * 1.0e-3
            record["follower_gripper_mit_q"] = command
            if gripper_response == "tracking":
                observed_gripper = command - 0.002
            elif gripper_response == "frozen":
                observed_gripper = -0.075
            elif gripper_response == "opposite":
                observed_gripper = 0.08 - tick * 1.0e-3
            else:
                raise AssertionError(
                    f"unknown gripper response: {gripper_response}"
                )
            record["follower_gripper_motor_q_rad"] = observed_gripper
            record["follower_gripper_motor_qd_rad_s"] = 0.0
        if frozen_feedback is None:
            arm_counters = [tick * 8 + index + 1 for index in range(7)]
            arm_timestamps = [
                observation_ns - 500_000 + index for index in range(7)
            ]
            gripper_counter = tick * 8 + 8
            gripper_timestamp = observation_ns - 400_000
        else:
            arm_counters, arm_timestamps, gripper_counter, gripper_timestamp = (
                frozen_feedback
            )
        record.update(
            schema=EXACT_CONTROL_RECORD_SCHEMA_V4,
            feedback_snapshot_monotonic_ns=observation_ns,
            follower_arm_state_rx_motor_ids=list(range(1, 8)),
            follower_arm_state_rx_counters=list(arm_counters),
            follower_arm_state_rx_generations=list(arm_counters),
            follower_arm_state_rx_monotonic_ns=[
                timestamp - stale_by_ns for timestamp in arm_timestamps
            ],
            follower_arm_state_rx_last_gap_ns=[1_000_000] * 7,
            follower_arm_state_rx_gap_violation_counts=[
                gap_violation_count
            ] * 7,
            follower_gripper_state_rx_motor_id=8,
            follower_gripper_state_rx_counter=gripper_counter,
            follower_gripper_state_rx_generation=gripper_counter,
            follower_gripper_state_rx_monotonic_ns=(
                gripper_timestamp - stale_by_ns
            ),
            follower_gripper_state_rx_last_gap_ns=1_000_000,
            follower_gripper_state_rx_gap_violation_count=(
                gap_violation_count
            ),
        )
    return value


def _start_and_record_v4(
    capture: RobotSessionCapture,
    *,
    frozen_q: bool = False,
    all_zero_q: bool = False,
    near_zero_q: bool = False,
    noisy_obs_static_command: bool = False,
    hold_joint_index: int | None = None,
    arm_response_fault: tuple[int, str] | None = None,
    gripper_response: str = "stationary",
    freeze_feedback_after_start: bool = False,
    increment_gap_violation_after_start: bool = False,
    stale_by_ns: int = 0,
) -> None:
    preflight = (
        _v4_envelope(
            envelope_seq=10,
            first_tick=100,
            first_observation_ns=980_000_000,
            frozen_q=frozen_q,
            all_zero_q=all_zero_q,
            near_zero_q=near_zero_q, noisy_obs_static_command=noisy_obs_static_command,
            hold_joint_index=hold_joint_index,
            arm_response_fault=arm_response_fault,
            gripper_response=gripper_response,
        ),
        _v4_envelope(
            envelope_seq=11,
            first_tick=110,
            first_observation_ns=990_000_000,
            frozen_q=frozen_q,
            all_zero_q=all_zero_q,
            near_zero_q=near_zero_q, noisy_obs_static_command=noisy_obs_static_command,
            hold_joint_index=hold_joint_index,
            arm_response_fault=arm_response_fault,
            gripper_response=gripper_response,
        ),
    )
    for envelope in preflight:
        capture.ingest_datagram(_wire(envelope))
    baseline = preflight[-1]["records"][-1]
    capture.start(now_monotonic_ns=1_000_000_000)
    frozen_feedback = None
    if freeze_feedback_after_start:
        frozen_feedback = (
            baseline["follower_arm_state_rx_counters"],
            baseline["follower_arm_state_rx_monotonic_ns"],
            baseline["follower_gripper_state_rx_counter"],
            baseline["follower_gripper_state_rx_monotonic_ns"],
        )
    for sequence, first_tick, first_ns in (
        (12, 120, 1_000_000_000),
        (13, 130, 1_010_000_000),
        (14, 140, 1_020_000_000),
    ):
        capture.ingest_datagram(
            _wire(
                _v4_envelope(
                    envelope_seq=sequence,
                    first_tick=first_tick,
                    first_observation_ns=first_ns,
                    frozen_q=frozen_q,
                    all_zero_q=all_zero_q,
                    near_zero_q=near_zero_q, noisy_obs_static_command=noisy_obs_static_command,
                    hold_joint_index=hold_joint_index,
                    arm_response_fault=arm_response_fault,
                    gripper_response=gripper_response,
                    frozen_feedback=frozen_feedback,
                    gap_violation_count=(
                        6 if increment_gap_violation_after_start else 5
                    ),
                    stale_by_ns=stale_by_ns,
                )
            )
        )


def _start_and_record_v3(
    capture: RobotSessionCapture, *, fail_tick: int | None = None
) -> None:
    for envelope in (
        _v3_envelope(
            envelope_seq=10,
            first_tick=100,
            first_observation_ns=980_000_000,
        ),
        _v3_envelope(
            envelope_seq=11,
            first_tick=110,
            first_observation_ns=990_000_000,
        ),
    ):
        capture.ingest_datagram(_wire(envelope))
    capture.start(now_monotonic_ns=1_000_000_000)
    for sequence, first_tick, first_ns in (
        (12, 120, 1_000_000_000),
        (13, 130, 1_010_000_000),
        (14, 140, 1_020_000_000),
    ):
        envelope = _v3_envelope(
            envelope_seq=sequence,
            first_tick=first_tick,
            first_observation_ns=first_ns,
        )
        if fail_tick is not None and first_tick <= fail_tick < first_tick + 10:
            record = envelope["records"][fail_tick - first_tick]
            record.update(
                arm_tx_full_frame_write_count=6,
                arm_tx_failed_frame_write_count=1,
                socketcan_tx_enqueue_success=False,
            )
        capture.ingest_datagram(_wire(envelope))


def _new_capture(path: Path) -> RobotSessionCapture:
    return RobotSessionCapture(
        path,
        _contract(),
        preflight_min_envelopes=2,
        preflight_duration_ns=19_000_000,
    )


def _preflight(capture: RobotSessionCapture) -> None:
    capture.ingest_datagram(
        _wire(
            _envelope(
                envelope_seq=10,
                first_tick=100,
                first_observation_ns=980_000_000,
            )
        )
    )
    capture.ingest_datagram(
        _wire(
            _envelope(
                envelope_seq=11,
                first_tick=110,
                first_observation_ns=990_000_000,
            )
        )
    )


def _start_and_record(capture: RobotSessionCapture) -> tuple[bytes, ...]:
    _preflight(capture)
    capture.start(now_monotonic_ns=1_000_000_000)
    payloads = (
        _wire(
            _envelope(
                envelope_seq=12,
                first_tick=120,
                first_observation_ns=1_000_000_000,
            ),
            pretty=True,
        ),
        _wire(
            _envelope(
                envelope_seq=13,
                first_tick=130,
                first_observation_ns=1_010_000_000,
            )
        ),
        _wire(
            _envelope(
                envelope_seq=14,
                first_tick=140,
                first_observation_ns=1_020_000_000,
            )
        ),
    )
    for payload in payloads:
        capture.ingest_datagram(payload)
    return payloads


def _record_frame(capture: RobotSessionCapture):
    return capture.record_frame_evidence(
        camera_id="orbbec-left-rgbd",
        frame_index=7,
        exposure_lower_ns=1_011_500_000,
        exposure_nominal_ns=1_017_500_000,
        exposure_upper_ns=1_025_500_000,
        now_monotonic_ns=1_029_500_000,
        camera_profile_sha256=SHA_C,
        timing_artifact_sha256=SHA_D,
    )


def _expected_action_hash(first_tick: int, last_tick: int) -> str:
    digest = hashlib.sha256()
    for tick in range(first_tick, last_tick + 1):
        record = _record(tick, 1_000_000_000 + (tick - 120) * 1_000_000)
        action = {
            "control_tick_seq": tick,
            "follower_arm_mit_kp": record["follower_arm_mit_kp"],
            "follower_arm_mit_kd": record["follower_arm_mit_kd"],
            "follower_arm_mit_q": record["follower_arm_mit_q"],
            "follower_arm_mit_dq": record["follower_arm_mit_dq"],
            "follower_arm_mit_tau": record["follower_arm_mit_tau"],
            "follower_gripper_mit_kp": record["follower_gripper_mit_kp"],
            "follower_gripper_mit_kd": record["follower_gripper_mit_kd"],
            "follower_gripper_mit_q": record["follower_gripper_mit_q"],
            "follower_gripper_mit_dq": record["follower_gripper_mit_dq"],
            "follower_gripper_mit_tau": record["follower_gripper_mit_tau"],
        }
        digest.update(
            (json.dumps(action, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def test_v2_capture_persists_raw_envelopes_and_each_exact_tick_once(tmp_path: Path) -> None:
    capture = _new_capture(tmp_path)
    payloads = _start_and_record(capture)
    frame = _record_frame(capture)

    nominal = frame.trajectory.sample_observation_at(1_017_500_000)
    assert nominal.follower_q_rad[0] == pytest.approx(0.1375)
    assert not hasattr(nominal, "action")
    assert [tick for tick, _ in frame.trajectory.exact_actions] == list(range(131, 147))

    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    raw = (tmp_path / "robot_control_envelopes.jsonl").read_bytes()
    assert raw == b"".join(payload + b"\n" for payload in payloads)

    with (tmp_path / "robot_exact_control.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    ticks = [int(row["control_tick_seq"]) for row in rows]
    assert ticks == list(range(120, 150))
    assert len(ticks) == len(set(ticks))
    assert not any("follower_command" in field for field in rows[0])
    assert float(rows[17]["action.arm_mit_q.openarm_left_joint1.motor_rad"]) == pytest.approx(
        0.137
    )
    assert float(rows[17]["observation.follower.openarm_left_joint1.motor_rad"]) == pytest.approx(
        0.137
    )

    frame_row = json.loads(
        (tmp_path / "robot_frame_index.jsonl").read_text(encoding="utf-8")
    )
    assert frame_row["exact_action_tick_first"] == 131
    assert frame_row["exact_action_tick_last"] == 146
    assert frame_row["exact_action_tick_count"] == 16
    assert frame_row["exact_action_sha256"] == _expected_action_hash(131, 146)
    observation = frame_row["observation_at_nominal"]
    assert observation["source_control_tick_before"] == 137
    assert observation["source_control_tick_after"] == 138
    assert observation["follower_q_rad"][0] == pytest.approx(0.1375)
    assert not any("action" in key for key in observation)

    assert summary["envelope_count"] == 3
    assert summary["control_tick_count"] == 30
    assert summary["control_tick_first"] == 120
    assert summary["control_tick_last"] == 149


def test_summary_binds_full_source_closure_and_unobservable_send_blocker(tmp_path: Path) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record(capture)
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["side"] == "left"
    assert summary["leader_can_interface"] == "can1"
    assert summary["follower_can_interface"] == "can3"
    assert summary["controller_source_manifest_sha256"] == SHA_1
    assert summary["follower_config_sha256"] == SHA_2
    assert summary["openarm_sdk_sha256"] == SHA_3
    assert summary["joint_position_limits_rad"] == [
        [-4.0, 4.0] for _ in range(7)
    ]
    assert summary["max_abs_joint_velocity_rad_s"] == 8.0
    assert summary["gripper_motor_position_limits_rad"] == [-0.2, 0.2]
    assert summary["ring_overflow_count"] == 0
    assert summary["publisher_rejected_record_count"] == 0
    assert summary["socket_send_failure_count"] == 0
    assert summary["send_status_observable"] is False
    assert summary["send_status_reason"] == VOID_SEND_STATUS_REASON
    assert summary["training_eligible"] is False
    assert "robot_action_send_status_unobservable" in summary["eligibility_blockers"]
    assert summary["action_semantics"] == "exact_mitparam_consumed_at_control_tick"
    assert summary["action_interpolation"] == "forbidden"
    for name, item in summary["hash_inventory"].items():
        data = (tmp_path / name).read_bytes()
        assert item == {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }


def test_v3_clean_socketcan_tx_is_diagnostic_without_feedback_liveness(
    tmp_path: Path,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v3(capture)
    frame = _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["schema"] == EXACT_CAPTURE_SUMMARY_SCHEMA_V3_PRODUCTION
    assert frame.index["schema"] == EXACT_FRAME_INDEX_SCHEMA_V3_PRODUCTION
    assert summary["send_status_observable"] is True
    assert summary["send_status_semantics"] == SOCKETCAN_TX_ENQUEUE_SEMANTICS
    assert summary["motor_ack_observable"] is False
    assert summary["feedback_liveness_observable"] is False
    assert summary["feedback_liveness_reason"] == (
        "v3_has_no_per_motor_feedback_rx_counters_or_timestamps"
    )
    assert summary["arm_tx_expected_frame_count"] == 210
    assert summary["arm_tx_full_frame_write_count"] == 210
    assert summary["arm_tx_failed_frame_write_count"] == 0
    assert summary["gripper_tx_expected_frame_count"] == 30
    assert summary["gripper_tx_full_frame_write_count"] == 30
    assert summary["gripper_tx_failed_frame_write_count"] == 0
    assert summary["socketcan_tx_enqueue_failure_tick_count"] == 0
    assert summary["training_eligible"] is False
    assert summary["eligibility_blockers"] == [
        "robot_feedback_liveness_unobservable"
    ]
    assert validate_robot_capture_artifacts(tmp_path, _contract()) == summary


def test_v4_frozen_joint_values_keep_feedback_clean_but_block_training(
    tmp_path: Path,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v4(capture, frozen_q=True)
    frame = _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["schema"] == EXACT_CAPTURE_SUMMARY_SCHEMA_V4_PRODUCTION
    assert frame.index["schema"] == EXACT_FRAME_INDEX_SCHEMA_V4_PRODUCTION
    assert summary["feedback_liveness_observable"] is True
    assert summary["feedback_start_baseline"]["gap_violation_counts"] == [5] * 8
    assert summary["feedback_gap_violation_count_delta"] == [0] * 8
    assert summary["feedback_counter_progress_all_motors"] is True
    assert summary["feedback_all_ticks_fresh"] is True
    assert summary["feedback_clean"] is True
    assert summary["trajectory_nondegenerate"] is False
    assert summary["command_response_consistent"] is False
    assert summary["training_eligible"] is False
    assert summary["eligibility_blockers"] == [
        "robot_observation_trajectory_degenerate",
        "robot_command_response_inconsistent",
    ]
    assert validate_robot_capture_artifacts(tmp_path, _contract()) == summary


def test_v4_moving_trajectory_and_feedback_receipts_are_training_eligible(
    tmp_path: Path,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v4(capture)
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["feedback_clean"] is True
    assert summary["trajectory_nondegenerate"] is True
    assert summary["command_response_consistent"] is True
    assert summary["training_eligible"] is True
    assert summary["eligibility_blockers"] == []
    assert summary["buffer_duration_ns"] == 2_000_000_000
    assert summary["max_packets"] == 256
    assert summary["max_gap_ns"] == 20_000_000
    assert summary["max_staleness_ns"] == 50_000_000
    assert summary["max_exposure_interval_ns"] == 40_000_000

    with pytest.raises(RobotCaptureError, match="summary replay mismatch"):
        validate_robot_capture_artifacts(
            tmp_path,
            replace(_contract(), max_staleness_ns=49_000_000),
        )


@pytest.mark.parametrize("mode", ["all_zero", "near_zero"])
def test_v4_whole_7d_zero_or_near_zero_trajectory_is_blocked(
    tmp_path: Path, mode: str
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v4(
        capture,
        all_zero_q=mode == "all_zero",
        near_zero_q=mode == "near_zero",
    )
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["feedback_clean"] is True
    assert summary["trajectory_nondegenerate"] is False
    assert summary["training_eligible"] is False
    assert "robot_observation_trajectory_degenerate" in summary[
        "eligibility_blockers"
    ]


def test_v4_one_held_joint_is_allowed_when_rest_of_7d_trajectory_moves(
    tmp_path: Path,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v4(capture, hold_joint_index=3)
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["observation_joint_range_rad"][3] == 0.0
    assert summary["trajectory_nondegenerate"] is True
    assert summary["command_response_consistent"] is True
    assert summary["training_eligible"] is True


def test_v4_moving_observation_with_static_action_is_blocked(
    tmp_path: Path,
) -> None:
    # Reviewer P0: a stationary arm whose observation drifts above the 1e-4 range
    # floor (noise / passive motion) but whose commanded action never materially
    # moves must NOT pass the trajectory authenticity gate. A non-degenerate
    # observation alone is not proof that a real action was executed.
    capture = _new_capture(tmp_path)
    _start_and_record_v4(capture, noisy_obs_static_command=True)
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["observation_trajectory_max_joint_range_rad"] >= 1.0e-4
    assert summary["any_command_materially_moving"] is False
    assert summary["command_materially_moving"] == [False] * 8
    assert summary["trajectory_nondegenerate"] is False
    assert summary["training_eligible"] is False


@pytest.mark.parametrize(
    ("fault", "expected_reason"),
    (
        ("frozen", "frozen_or_near_zero_response"),
        ("weak", "tracking_gain_out_of_bounds"),
        ("opposite", "direction_inconsistent"),
        ("uncorrelated", "correlation_below_minimum"),
    ),
)
def test_v4_each_materially_commanded_arm_joint_requires_correlated_tracking(
    tmp_path: Path,
    fault: str,
    expected_reason: str,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v4(capture, arm_response_fault=(3, fault))
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    joint_name = JOINT_NAMES[3]
    assert summary["command_materially_moving"][3] is True
    assert summary["command_response_consistent"] is False
    assert summary["command_response_violation_joint_names"] == [joint_name]
    assert expected_reason in summary["command_response_violation_reasons"][
        joint_name
    ]
    assert summary["trajectory_nondegenerate"] is False
    assert summary["training_eligible"] is False


def test_v4_materially_commanded_gripper_requires_live_tracking_response(
    tmp_path: Path,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v4(capture, gripper_response="frozen")
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["actuator_names"][-1] == "gripper"
    assert summary["command_materially_moving"][-1] is True
    assert summary["observation_actuator_range_rad"][-1] == 0.0
    assert summary["command_response_violation_joint_names"] == ["gripper"]
    assert summary["command_response_violation_reasons"]["gripper"] == [
        "frozen_or_near_zero_response",
        "tracking_gain_out_of_bounds",
        "correlation_below_minimum",
        "direction_inconsistent",
    ]
    assert summary["training_eligible"] is False


def test_v4_tracking_arm_and_gripper_persist_per_actuator_response_evidence(
    tmp_path: Path,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v4(capture, gripper_response="tracking")
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["actuator_names"] == [*JOINT_NAMES, "gripper"]
    assert summary["command_materially_moving"] == [True] * 8
    assert all(
        correlation is not None and correlation > 0.99
        for correlation in summary["command_response_correlation"]
    )
    assert summary["command_response_direction_consistent"] == [True] * 8
    assert all(
        gain is not None and 0.9 <= gain <= 1.1
        for gain in summary["command_response_tracking_gain"]
    )
    assert summary["command_response_violation_joint_names"] == []
    assert summary["training_eligible"] is True


def test_v4_changing_joint_values_cannot_replace_decode_counter_progress(
    tmp_path: Path,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v4(capture, freeze_feedback_after_start=True)
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["feedback_counter_progress_all_motors"] is False
    assert summary["training_eligible"] is False
    assert "robot_feedback_counter_no_progress" in summary["eligibility_blockers"]


def test_v4_pre_start_gap_history_is_baselined_but_new_violation_blocks(
    tmp_path: Path,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v4(capture, increment_gap_violation_after_start=True)
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["feedback_start_baseline"]["gap_violation_counts"] == [5] * 8
    assert summary["feedback_gap_violation_count_delta"] == [1] * 8
    assert summary["training_eligible"] is False
    assert "robot_feedback_gap_exceeded" in summary["eligibility_blockers"]


def test_v4_stale_feedback_blocks_without_inferring_from_joint_motion(
    tmp_path: Path,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v4(capture, freeze_feedback_after_start=True)
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["feedback_all_ticks_fresh"] is False
    assert summary["training_eligible"] is False
    assert "robot_feedback_stale" in summary["eligibility_blockers"]


def test_exact_schema_exports_name_commissioning_and_production_explicitly():
    from ausc.perception.robot import robot_session_capture as capture_module

    assert (
        capture_module.EXACT_CAPTURE_SUMMARY_SCHEMA_V2_COMMISSIONING
        == "wam_openarm_exact_control_capture_summary_v2"
    )
    assert (
        capture_module.EXACT_FRAME_INDEX_SCHEMA_V2_COMMISSIONING
        == "wam_openarm_exact_control_frame_evidence_v2"
    )
    assert (
        capture_module.EXACT_CAPTURE_SUMMARY_SCHEMA_V3_PRODUCTION
        == "wam_openarm_exact_control_capture_summary_v3"
    )
    assert (
        capture_module.EXACT_FRAME_INDEX_SCHEMA_V3_PRODUCTION
        == "wam_openarm_exact_control_frame_evidence_v3"
    )
    assert (
        capture_module.EXACT_CAPTURE_SUMMARY_SCHEMA_V4_PRODUCTION
        == "wam_openarm_exact_control_capture_summary_v4"
    )
    assert (
        capture_module.EXACT_FRAME_INDEX_SCHEMA_V4_PRODUCTION
        == "wam_openarm_exact_control_frame_evidence_v4"
    )
    assert "EXACT_CAPTURE_SUMMARY_SCHEMA" not in capture_module.__all__
    assert "EXACT_FRAME_INDEX_SCHEMA" not in capture_module.__all__


def test_v3_failed_full_frame_write_is_durable_but_training_ineligible(
    tmp_path: Path,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record_v3(capture, fail_tick=137)
    _record_frame(capture)
    summary = capture.finish(now_monotonic_ns=1_030_000_000)

    assert summary["arm_tx_full_frame_write_count"] == 209
    assert summary["arm_tx_failed_frame_write_count"] == 1
    assert summary["socketcan_tx_enqueue_failure_tick_count"] == 1
    assert summary["feedback_liveness_observable"] is False
    assert summary["training_eligible"] is False
    assert summary["eligibility_blockers"] == [
        "robot_socketcan_tx_enqueue_failure",
        "robot_feedback_liveness_unobservable",
    ]
    assert validate_robot_capture_artifacts(tmp_path, _contract()) == summary


def test_exact_capture_requires_source_closure_and_left_can1_to_can3(
    tmp_path: Path,
) -> None:
    right = replace(
        _contract(),
        side="right",
        joint_names=tuple(f"openarm_right_joint{i}" for i in range(1, 8)),
        leader_can_interface="can0",
        follower_can_interface="can2",
    )
    with pytest.raises(RobotCaptureError, match="left.*can1.*can3"):
        RobotSessionCapture(tmp_path, right)

    for field in (
        "controller_source_manifest_sha256",
        "follower_config_sha256",
        "openarm_sdk_sha256",
    ):
        with pytest.raises(RobotCaptureError, match="source.*config.*SDK"):
            RobotSessionCapture(tmp_path, _contract(**{field: ""}))


def test_start_requires_fresh_contiguous_v2_preflight(tmp_path: Path) -> None:
    capture = _new_capture(tmp_path)
    capture.ingest_datagram(
        _wire(
            _envelope(
                envelope_seq=10,
                first_tick=100,
                first_observation_ns=980_000_000,
            )
        )
    )
    with pytest.raises(RobotCaptureError, match="preflight"):
        capture.start(now_monotonic_ns=1_000_000_000)
    assert not capture.faulted
    assert not (tmp_path / "robot_control_envelopes.jsonl.partial").exists()

    capture.ingest_datagram(
        _wire(
            _envelope(
                envelope_seq=11,
                first_tick=110,
                first_observation_ns=990_000_000,
            )
        )
    )
    capture.start(now_monotonic_ns=1_000_000_000)
    assert capture.recording


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("ring_overflow_count", "ring_overflow"),
        ("publisher_rejected_record_count", "publisher_rejected"),
        ("socket_send_failure_count", "socket_send_failure"),
    ),
)
def test_any_runtime_loss_counter_faults_and_preserves_raw_partial(
    tmp_path: Path, field: str, message: str
) -> None:
    capture = _new_capture(tmp_path)
    _preflight(capture)
    capture.start(now_monotonic_ns=1_000_000_000)
    bad = _envelope(
        envelope_seq=12,
        first_tick=120,
        first_observation_ns=1_000_000_000,
    )
    bad[field] = 1
    with pytest.raises(RobotCaptureError, match=message):
        capture.ingest_datagram(_wire(bad))
    assert capture.faulted
    assert (tmp_path / "robot_control_envelopes.jsonl.partial").exists()
    assert not (tmp_path / "robot_capture_summary.json").exists()


def test_frame_must_be_fully_bracketed_by_ticks_recorded_after_start(tmp_path: Path) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record(capture)
    with pytest.raises(RobotCaptureError, match="bracket"):
        capture.record_frame_evidence(
            camera_id="orbbec-left-rgbd",
            frame_index=1,
            exposure_lower_ns=1_025_000_000,
            exposure_nominal_ns=1_029_000_000,
            exposure_upper_ns=1_035_000_000,
            now_monotonic_ns=1_040_000_000,
            camera_profile_sha256=SHA_C,
            timing_artifact_sha256=SHA_D,
        )
    assert capture.faulted
    assert not (tmp_path / "robot_capture_summary.json").exists()


def test_finish_detects_raw_or_frame_action_hash_tampering(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    raw_capture = _new_capture(raw_path)
    _start_and_record(raw_capture)
    _record_frame(raw_capture)
    with (raw_path / "robot_control_envelopes.jsonl.partial").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(RobotCaptureError, match="digest|framing"):
        raw_capture.finish(now_monotonic_ns=1_030_000_000)
    assert not (raw_path / "robot_capture_summary.json").exists()

    frame_path = tmp_path / "frame"
    frame_path.mkdir()
    frame_capture = _new_capture(frame_path)
    _start_and_record(frame_capture)
    _record_frame(frame_capture)
    index_path = frame_path / "robot_frame_index.jsonl.partial"
    row = json.loads(index_path.read_text(encoding="utf-8"))
    row["exact_action_sha256"] = "0" * 64
    index_path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    # Match the live byte digest so semantic verification, not only the writer
    # digest, proves the action-range hash is independently checked.
    data = index_path.read_bytes()
    frame_capture._frame_index_digest = hashlib.sha256(data)  # noqa: SLF001
    frame_capture._frame_index_bytes_written = len(data)  # noqa: SLF001
    with pytest.raises(RobotCaptureError, match="action.*hash"):
        frame_capture.finish(now_monotonic_ns=1_030_000_000)
    assert not (frame_path / "robot_capture_summary.json").exists()


def test_atomic_promotion_failure_leaves_no_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record(capture)
    _record_frame(capture)
    real_link = capture_module.os.link
    calls = 0

    def fail_second_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected promotion failure")
        return real_link(source, destination)

    monkeypatch.setattr(capture_module.os, "link", fail_second_link)
    with pytest.raises(RobotCaptureError, match="promotion"):
        capture.finish(now_monotonic_ns=1_030_000_000)
    assert not (tmp_path / "robot_capture_summary.json").exists()
    assert not (tmp_path / "robot_control_envelopes.jsonl").exists()


def test_public_validator_streams_large_artifacts_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record(capture)
    _record_frame(capture)
    expected = capture.finish(now_monotonic_ns=1_030_000_000)
    real_read_bytes = Path.read_bytes

    def reject_unbounded_data_read(path: Path):
        if path.name in {
            "robot_control_envelopes.jsonl",
            "robot_exact_control.csv",
            "robot_frame_index.jsonl",
        }:
            raise AssertionError(f"unbounded read_bytes forbidden for {path.name}")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_data_read)
    assert validate_robot_capture_artifacts(tmp_path, _contract()) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("joint_position_limits_rad", [[-5.0, 5.0] for _ in range(7)]),
        ("max_abs_joint_velocity_rad_s", 9.0),
        ("gripper_motor_position_limits_rad", [-0.3, 0.3]),
    ],
)
def test_public_validator_rejects_tampered_safety_bounds(
    tmp_path: Path, field: str, value: object,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record(capture)
    _record_frame(capture)
    capture.finish(now_monotonic_ns=1_030_000_000)
    summary_path = tmp_path / "robot_capture_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RobotCaptureError, match="summary replay mismatch"):
        validate_robot_capture_artifacts(tmp_path, _contract())


def test_60_minute_1khz_tick_window_has_duration_independent_memory() -> None:
    total_ticks = 60 * 60 * 1000
    window = capture_module._BoundedExactControlWindow(max_ticks=128)  # noqa: SLF001
    seen = 0
    for tick in range(total_ticks):
        # A 40 ms camera exposure plus one bracketing tick on each side.
        window.discard_before(max(0, tick - 41))
        window.append(tick)
        window.require_within_bound()
        seen += 1

    assert seen == 3_600_000
    assert len(window) == 42
    assert window.peak_buffered_ticks == 42


def test_public_validator_rejects_symlinked_artifact_even_when_bytes_match(
    tmp_path: Path,
) -> None:
    capture = _new_capture(tmp_path)
    _start_and_record(capture)
    _record_frame(capture)
    capture.finish(now_monotonic_ns=1_030_000_000)
    raw_path = tmp_path / "robot_control_envelopes.jsonl"
    copied_path = tmp_path / "same-bytes-control-envelopes.jsonl"
    shutil.copyfile(raw_path, copied_path)
    raw_path.unlink()
    raw_path.symlink_to(copied_path.name)

    with pytest.raises(RobotCaptureError, match="symlink|regular file|no-follow"):
        validate_robot_capture_artifacts(tmp_path, _contract())
