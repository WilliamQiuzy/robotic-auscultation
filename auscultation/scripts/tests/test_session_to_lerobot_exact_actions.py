"""Exact OpenArm v3 action conversion tests.

These tests deliberately build the sidecar through the production capture
writer.  The converter must therefore consume the same bytes that the live
collector commits, not a hand-written approximation of the schema.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
PACKAGE = SCRIPTS.parent
for entry in (str(SCRIPTS), str(PACKAGE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import session_to_lerobot_v2 as conv  # noqa: E402
from ausc.perception.robot.robot_session_capture import RobotSessionCapture  # noqa: E402
from ausc.perception.robot.robot_state_stream import RobotStateContract  # noqa: E402


def _record(
    tick: int, observation_ns: int, *, v3: bool = True, v4: bool = False
) -> dict:
    follower_q = [tick / 1000.0 + index * 0.01 for index in range(7)]
    action_q = [1.0 + tick / 1000.0 + index * 0.02 for index in range(7)]
    record = {
        "schema": (
            "wam_openarm_exact_control_record_v4"
            if v4
            else (
                "wam_openarm_exact_control_record_v3"
                if v3
                else "wam_openarm_exact_control_record_v2"
            )
        ),
        "control_tick_seq": tick,
        "observation_monotonic_ns": observation_ns,
        "reference_generation": 17,
        "reference_source_monotonic_ns": observation_ns - 100_000,
        "reference_snapshot_valid": True,
        "control_call_begin_monotonic_ns": observation_ns + 50_000,
        "arm_call_return_monotonic_ns": observation_ns + 100_000,
        "gripper_call_begin_monotonic_ns": observation_ns + 110_000,
        "gripper_call_return_monotonic_ns": observation_ns + 160_000,
        "control_call_end_monotonic_ns": observation_ns + 170_000,
        "follower_q_rad": follower_q,
        "follower_qd_rad_s": [0.5] * 7,
        "follower_gripper_q_rad": -0.07,
        "follower_gripper_qd_rad_s": 0.01,
        "follower_gripper_motor_q_rad": -0.075,
        "follower_gripper_motor_qd_rad_s": 0.015,
        "follower_arm_mit_kp": [240, 240, 240, 240, 40, 48, 40],
        "follower_arm_mit_kd": [3.0, 3.0, 3.0, 3.0, 0.3, 0.3, 0.3],
        "follower_arm_mit_q": action_q,
        "follower_arm_mit_dq": [0.4] * 7,
        "follower_arm_mit_tau": [0.01 * tick] * 7,
        "follower_gripper_mit_kp": 25.0,
        "follower_gripper_mit_kd": 0.3,
        "follower_gripper_mit_q": -0.08 - tick / 100_000.0,
        "follower_gripper_mit_dq": 0.0,
        "follower_gripper_mit_tau": 0.02 * tick,
    }
    if v3 or v4:
        record.update(
            arm_tx_expected_frame_count=7,
            arm_tx_full_frame_write_count=7,
            arm_tx_failed_frame_write_count=0,
            gripper_tx_expected_frame_count=1,
            gripper_tx_full_frame_write_count=1,
            gripper_tx_failed_frame_write_count=0,
            socketcan_tx_enqueue_success=True,
        )
    if v4:
        record.update(
            feedback_snapshot_monotonic_ns=observation_ns,
            follower_arm_state_rx_motor_ids=list(range(1, 8)),
            follower_arm_state_rx_counters=[tick * 8 + index for index in range(1, 8)],
            follower_arm_state_rx_generations=[tick * 8 + index for index in range(1, 8)],
            follower_arm_state_rx_monotonic_ns=[
                observation_ns - 500_000 + index for index in range(7)
            ],
            follower_arm_state_rx_last_gap_ns=[1_000_000] * 7,
            follower_arm_state_rx_gap_violation_counts=[5] * 7,
            follower_gripper_state_rx_motor_id=8,
            follower_gripper_state_rx_counter=tick * 8 + 8,
            follower_gripper_state_rx_generation=tick * 8 + 8,
            follower_gripper_state_rx_monotonic_ns=observation_ns - 400_000,
            follower_gripper_state_rx_last_gap_ns=1_000_000,
            follower_gripper_state_rx_gap_violation_count=5,
        )
    return record


def _write_sidecar(
    session: Path,
    *,
    v3: bool = True,
    v4: bool = False,
    frame_index: int = 7,
    recorded_first_observation_offset_ns: int = -10_400_000,
) -> tuple[dict, int]:
    joint_names = tuple(f"openarm_left_joint{index}" for index in range(1, 8))
    contract = RobotStateContract(
        side="left",
        controller_mode="unilateral",
        joint_names=joint_names,
        joint_position_limits_rad=tuple((-4.0, 4.0) for _ in range(7)),
        leader_can_interface="can1",
        follower_can_interface="can3",
        urdf_sha256="a" * 64,
        joint_mapper_sha256="b" * 64,
        camera_profile_sha256="c" * 64,
        timing_artifact_sha256="d" * 64,
        pair_contract_sha256="e" * 64,
        controller_binary_sha256="f" * 64,
        controller_source_manifest_sha256="1" * 64,
        follower_config_sha256="2" * 64,
        openarm_sdk_sha256="3" * 64,
        max_abs_joint_velocity_rad_s=8.0,
        gripper_motor_position_limits_rad=(-0.2, 0.2),
    )
    nominal_ns = int(round((100.0 + frame_index / 30.0) * 1_000_000_000))

    def envelope(sequence: int, first_tick: int, first_ns: int, count: int) -> bytes:
        records = [
            _record(
                first_tick + offset,
                first_ns + offset * 1_000_000,
                v3=v3,
                v4=v4,
            )
            for offset in range(count)
        ]
        payload = {
            "schema": (
                "wam_openarm_control_envelope_v4"
                if v4
                else (
                    "wam_openarm_control_envelope_v3"
                    if v3
                    else "wam_openarm_control_envelope_v2"
                )
            ),
            "envelope_seq": sequence,
            "clock_id": "CLOCK_MONOTONIC",
            "publisher_monotonic_ns": records[-1]["control_call_end_monotonic_ns"] + 500_000,
            "controller_pid": 4242,
            "controller_start_token": "proc-start-991",
            "side": "left",
            "controller_mode": "unilateral",
            "leader_can_interface": "can1",
            "follower_can_interface": "can3",
            "joint_names": list(joint_names),
            "urdf_sha256": contract.urdf_sha256,
            "joint_mapper_sha256": contract.joint_mapper_sha256,
            "pair_contract_sha256": contract.pair_contract_sha256,
            "controller_binary_sha256": contract.controller_binary_sha256,
            "controller_source_manifest_sha256": contract.controller_source_manifest_sha256,
            "follower_config_sha256": contract.follower_config_sha256,
            "openarm_sdk_sha256": contract.openarm_sdk_sha256,
            "publisher_nominal_hz": 100,
            "control_nominal_hz": 1000,
            "send_status_observable": v3 or v4,
            "send_status_reason": (
                "openarm_can_1_2_2_full_frame_write_results"
                if v3 or v4 else "upstream_mit_control_all_returns_void"
            ),
            "ring_overflow_count": 0,
            "publisher_rejected_record_count": 0,
            "socket_send_failure_count": 0,
            "records": records,
        }
        if v3 or v4:
            payload.update(
                send_status_semantics=(
                    "linux_socketcan_full_frame_write_tx_enqueue_not_motor_ack"
                ),
                motor_ack_observable=False,
                motor_ack_reason=(
                    "socketcan_tx_enqueue_does_not_acknowledge_motor_receipt_or_execution"
                ),
            )
        if v4:
            payload.update(
                feedback_liveness_observable=True,
                feedback_liveness_reason=(
                    "openarm_can_1_2_2_matching_motor_id_valid_state_decode"
                ),
                feedback_liveness_semantics=(
                    "counter_generation_increment_only_after_matching_motor_id_valid_state_decode"
                ),
                feedback_max_age_ns=20_000_000,
                feedback_max_gap_ns=20_000_000,
            )
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    capture = RobotSessionCapture(
        session,
        contract,
        preflight_min_envelopes=2,
        preflight_duration_ns=10_000_000,
    )
    capture.ingest_datagram(envelope(48, 70, nominal_ns - 31_000_000, 10))
    capture.ingest_datagram(envelope(49, 80, nominal_ns - 21_000_000, 10))
    capture.start(now_monotonic_ns=nominal_ns - 11_000_000)
    capture.ingest_datagram(envelope(
        50, 90, nominal_ns + recorded_first_observation_offset_ns, 16,
    ))
    capture.record_frame_evidence(
        camera_id="orbbec-left-rgbd",
        frame_index=frame_index,
        exposure_lower_ns=nominal_ns - 500_000,
        exposure_nominal_ns=nominal_ns,
        exposure_upper_ns=nominal_ns + 500_000,
        now_monotonic_ns=nominal_ns + 6_000_000,
        camera_profile_sha256=contract.camera_profile_sha256,
        timing_artifact_sha256=contract.timing_artifact_sha256,
    )
    return dict(capture.finish(now_monotonic_ns=nominal_ns + 6_000_000)), nominal_ns


def _camera(frame_index: int, nominal_ns: int) -> conv.CameraTimeline:
    return conv.CameraTimeline(
        frame_index=np.asarray([frame_index], dtype=np.int64),
        t_exposure_s=np.asarray([nominal_ns / 1_000_000_000.0]),
        host_arrival_t_s=np.asarray([nominal_ns / 1_000_000_000.0 + 0.012]),
        device_t_ms=np.asarray([1234.0]),
        clock_fit={"accepted": True},
        latency_ms=12.0,
    )


def _manifest(summary: dict) -> dict:
    return {
        "robot": summary,
        "arm": {
            "side": "left",
            "joint_names": summary["joint_names"],
            "state_unit": "rad",
            "action_type": "exact_mitparam_consumed_at_control_tick",
        },
        "camera": {"camera_id": "orbbec-left-rgbd"},
    }


def test_exact_v3_reader_selects_one_real_tick_without_action_interpolation(tmp_path):
    summary, nominal_ns = _write_sidecar(tmp_path)

    stream = conv.read_exact_robot_frames(
        tmp_path, _camera(7, nominal_ns), _manifest(summary),
    )

    # Tick observations land at nominal-0.4 ms and nominal+0.6 ms. Tick 100's
    # complete arm+gripper call ends at nominal-0.23 ms, so it is the latest
    # fully consumed real action and must never be blended with tick 101.
    assert stream.action_tick_seq.tolist() == [100]
    assert stream.action_tick_monotonic_ns.tolist() == [nominal_ns - 230_000]
    assert stream.action.shape == (1, 40)
    assert stream.action_names[:7] == tuple(
        f"arm_mit_kp.{name}" for name in summary["joint_names"]
    )
    assert stream.action[0, :7].tolist() == pytest.approx([240, 240, 240, 240, 40, 48, 40])
    action_q_offset = 14
    assert stream.action[0, action_q_offset] == pytest.approx(1.1)
    assert stream.state[0, 0] != pytest.approx(stream.action[0, action_q_offset])
    selection = stream.selection_records[0]
    assert selection["selection_rule"] == conv.EXACT_ACTION_SELECTION_RULE
    assert selection["selected_control_tick_seq"] == 100
    assert selection["action_interpolation"] == "forbidden"


def test_action_selection_never_uses_a_tick_that_completed_after_exposure(tmp_path):
    summary, nominal_ns = _write_sidecar(
        tmp_path, recorded_first_observation_offset_ns=-10_100_000,
    )

    stream = conv.read_exact_robot_frames(
        tmp_path, _camera(7, nominal_ns), _manifest(summary),
    )

    # Tick 100 observed state at nominal-0.1 ms, but its complete arm+gripper
    # control call ends at nominal+0.07 ms.  It is a future action for this
    # frame, so the converter must select completed tick 99.
    assert stream.action_tick_seq.tolist() == [99]
    selection = stream.selection_records[0]
    assert selection["selected_control_call_end_monotonic_ns"] <= nominal_ns
    assert selection["selected_control_tick_seq"] == 99


def test_exact_v3_reader_canonical_replay_rejects_tampered_tick_csv(tmp_path):
    summary, nominal_ns = _write_sidecar(tmp_path)
    path = tmp_path / "robot_exact_control.csv"
    path.write_bytes(path.read_bytes().replace(b",240,", b",241,", 1))

    with pytest.raises(ValueError, match="canonical replay"):
        conv.read_exact_robot_frames(
            tmp_path, _camera(7, nominal_ns), _manifest(summary),
        )


def test_exact_v3_reader_requires_one_robot_row_for_each_rgbd_frame(tmp_path):
    summary, nominal_ns = _write_sidecar(tmp_path, frame_index=7)

    with pytest.raises(ValueError, match="frame index.*camera timeline"):
        conv.read_exact_robot_frames(
            tmp_path, _camera(8, nominal_ns), _manifest(summary),
        )


def test_current_void_send_status_sidecar_is_never_training_eligible(tmp_path):
    summary, nominal_ns = _write_sidecar(tmp_path, v3=False)
    stream = conv.read_exact_robot_frames(
        tmp_path, _camera(7, nominal_ns), _manifest(summary),
    )

    with pytest.raises(ValueError, match="send-status.*blocker"):
        conv.require_exact_robot_training_eligibility(stream.summary)


def test_clean_v3_socketcan_tx_sidecar_is_rejected_without_feedback_liveness(
    tmp_path,
):
    summary, nominal_ns = _write_sidecar(tmp_path)
    stream = conv.read_exact_robot_frames(
        tmp_path, _camera(7, nominal_ns), _manifest(summary),
    )

    assert stream.summary["feedback_liveness_observable"] is False
    assert stream.summary["training_eligible"] is False
    with pytest.raises(ValueError, match="feedback liveness.*blocker"):
        conv.require_exact_robot_training_eligibility(stream.summary)


def test_v3_schema_cannot_self_assert_feedback_liveness(tmp_path):
    summary, nominal_ns = _write_sidecar(tmp_path)
    stream = conv.read_exact_robot_frames(
        tmp_path, _camera(7, nominal_ns), _manifest(summary),
    )
    forged = dict(stream.summary)
    forged.update(
        feedback_liveness_observable=True,
        feedback_liveness_reason="self_asserted_without_motor_rx_evidence",
        training_eligible=True,
        eligibility_blockers=[],
    )

    with pytest.raises(ValueError, match="feedback liveness.*v3"):
        conv.require_exact_robot_training_eligibility(forged)


def test_clean_v4_decode_receipts_are_canonical_training_eligible(tmp_path):
    summary, nominal_ns = _write_sidecar(tmp_path, v4=True)
    stream = conv.read_exact_robot_frames(
        tmp_path, _camera(7, nominal_ns), _manifest(summary),
    )

    assert stream.summary["schema"] == (
        "wam_openarm_exact_control_capture_summary_v4"
    )
    assert stream.summary["feedback_counter_progress_all_motors"] is True
    assert stream.summary["feedback_gap_violation_count_delta"] == [0] * 8
    assert stream.summary["training_eligible"] is True
    conv.require_exact_robot_training_eligibility(stream.summary)


def test_v4_summary_with_forged_feedback_receipt_is_rejected(tmp_path):
    summary, nominal_ns = _write_sidecar(tmp_path, v4=True)
    forged = dict(summary)
    forged["feedback_gap_violation_count_delta"] = [0] * 7 + [1]

    with pytest.raises(ValueError, match="feedback.*v4"):
        conv.require_exact_robot_training_eligibility(forged)


def test_exact_reader_pins_one_sidecar_bundle_across_path_swap_and_revert(
    tmp_path, monkeypatch,
):
    summary, nominal_ns = _write_sidecar(tmp_path)
    frame_path = tmp_path / "robot_frame_index.jsonl"
    original_frame = json.loads(frame_path.read_text())
    expected_state = original_frame["observation_at_nominal"]["follower_q_rad"][0]

    backup = tmp_path / "backup"
    attacker = tmp_path / "attacker"
    backup.mkdir()
    attacker.mkdir()
    for name in conv.EXACT_ROBOT_FILES:
        (attacker / name).write_bytes((tmp_path / name).read_bytes())
    malicious_frame = json.loads((attacker / "robot_frame_index.jsonl").read_text())
    malicious_frame["observation_at_nominal"]["follower_q_rad"][0] = 3.9
    (attacker / "robot_frame_index.jsonl").write_text(
        json.dumps(malicious_frame, sort_keys=True, separators=(",", ":")) + "\n"
    )

    from ausc.perception.robot import robot_session_capture as robot_capture

    original_validate = robot_capture.validate_robot_capture_artifacts
    swap_count = 0
    replay_completed = False

    def swap_during_replay(session_dir, contract, **kwargs):  # noqa: ANN001
        nonlocal replay_completed, swap_count
        swap_count += 1
        for name in conv.EXACT_ROBOT_FILES:
            os.replace(tmp_path / name, backup / name)
            os.replace(attacker / name, tmp_path / name)
        try:
            result = original_validate(session_dir, contract, **kwargs)
            replay_completed = True
            return result
        finally:
            for name in conv.EXACT_ROBOT_FILES:
                os.replace(tmp_path / name, attacker / name)
                os.replace(backup / name, tmp_path / name)

    monkeypatch.setattr(
        robot_capture, "validate_robot_capture_artifacts", swap_during_replay,
    )

    with pytest.raises(ValueError, match="fd identity changed"):
        conv.read_exact_robot_frames(
            tmp_path, _camera(7, nominal_ns), _manifest(summary),
        )

    assert swap_count == 1
    assert replay_completed is True
    assert expected_state != pytest.approx(3.9)


def test_exact_reader_never_reopens_sidecar_paths_after_fd_pin(
    tmp_path, monkeypatch,
):
    summary, nominal_ns = _write_sidecar(tmp_path)
    original_path_open = Path.open

    def reject_sidecar_path_open(path, *args, **kwargs):  # noqa: ANN001
        if path.name in conv.EXACT_ROBOT_FILES:
            raise AssertionError(f"sidecar path reopened: {path.name}")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_sidecar_path_open)

    stream = conv.read_exact_robot_frames(
        tmp_path, _camera(7, nominal_ns), _manifest(summary),
    )

    assert stream.action_tick_seq.tolist() == [100]


def test_exact_causal_records_keep_selected_tick_and_full_mit_action(tmp_path):
    summary, nominal_ns = _write_sidecar(tmp_path)
    camera = _camera(7, nominal_ns)
    robot = conv.read_exact_robot_frames(tmp_path, camera, _manifest(summary))
    query_t = float(camera.t_exposure_s[0])
    force = conv.ForceStream(
        t_host_s=np.asarray([query_t - 0.01]),
        force_n=np.asarray([4.1], dtype=np.float32),
        status=("OK",),
        device_t_ms=None,
    )
    targets = conv.TargetStream(
        t_s=np.asarray([query_t - 0.02]),
        names=("R1",),
        uv=np.asarray([[0.2, 0.3]], dtype=np.float32),
        xyz=((0.1, 0.2, 0.3),),
        xyz_frames=("robot_base",),
    )

    records = conv.prepare_exact_causal_records(
        camera,
        robot,
        force,
        targets,
        [conv.RewardSpan(query_t - 0.1, query_t + 0.1, 0.75)],
        force_max_age_s=0.05,
        target_max_age_s=0.25,
    )

    frame = records[0].frame
    assert frame["action"].shape == (40,)
    assert frame["observation.state"][0] != frame["action"][14]
    # Tick identity is audit metadata, not a policy observation: exposing an
    # absolute sequence/time lets a policy learn session position leakage.
    assert "observation.robot_action_tick_seq" not in frame
    assert "observation.robot_action_tick_monotonic_ns" not in frame
    assert robot.selection_records[0]["selected_control_tick_seq"] == 100
    assert robot.selection_records[0]["selected_observation_monotonic_ns"] == (
        nominal_ns - 400_000
    )


def test_selection_index_is_canonical_and_hashed_for_world_action_replay(tmp_path):
    summary, nominal_ns = _write_sidecar(tmp_path)
    stream = conv.read_exact_robot_frames(
        tmp_path, _camera(7, nominal_ns), _manifest(summary),
    )
    destination = tmp_path / "robot_frame_action_selection.jsonl"

    metadata = conv.write_exact_action_selection(destination, stream)

    assert metadata["schema_version"] == conv.EXACT_ACTION_SELECTION_SCHEMA
    assert metadata["selection_rule"] == conv.EXACT_ACTION_SELECTION_RULE
    assert metadata["frame_count"] == 1
    assert metadata["sha256"] == conv._sha256(destination)
    row = json.loads(destination.read_text())
    assert row["selected_control_tick_seq"] == 100
    assert metadata["world_action_exact_control_sidecars"] == {
        name: {"path": name, "sha256": stream.sidecar_hashes[name]}
        for name in conv.EXACT_ROBOT_FILES
    }


def test_legacy_position_target_manifest_cannot_pass_canonical_conversion():
    manifest = {
        "schema_version": "wam_session_v2",
        "promotion_contract_version": conv.PROMOTION_CONTRACT_VERSION,
        "status": "complete",
        "training_eligible": True,
        "capture_kind": "production_teleop_v4",
        "rehearsal_overrides": [],
        "streams": ["force", "audio", "cam", "robot"],
        "arm": {
            "side": "left",
            "joint_names": [f"joint_{index}" for index in range(1, 8)],
            "state_unit": "rad",
            "action_type": "position_target",
        },
        "force": {"calibrated": True},
        "artifact_validation": {},
        "artifact_hashes": {},
        "camera": {
            "backend": "orbbec",
            "camera_id": "orbbec-left-rgbd",
            "camera_id_source": "verified_orbbec_serial",
            "has_depth": True,
        },
    }

    with pytest.raises(ValueError, match="exact MIT action"):
        conv.validate_manifest_for_conversion(manifest)


def test_diagnostic_capture_kind_cannot_be_converted_even_if_eligible():
    # A forged manifest may claim training_eligible=true, but a diagnostic
    # capture_kind must still be refused before any conversion work.
    manifest = {
        "schema_version": "wam_session_v2",
        "promotion_contract_version": conv.PROMOTION_CONTRACT_VERSION,
        "status": "complete",
        "training_eligible": True,
        "capture_kind": "diagnostic_perception",
        "rehearsal_overrides": [],
    }

    with pytest.raises(ValueError, match="production capture_kind"):
        conv.validate_manifest_for_conversion(manifest)


def test_tracker_jsonl_is_a_required_raw_artifact():
    assert "tracker.jsonl" in conv.REQUIRED_RAW_ARTIFACTS
