from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib

import numpy as np
import pytest

from ausc.perception.camera_stream import CameraFrame
from ausc.perception.robot.camera_exposure_evidence import CameraExposureEvidence
from ausc.perception.robot.frame_evidence_bundle import (
    FrameEvidenceBundle,
    FrameEvidenceBundleError,
    ImmutableCameraFrame,
)
from ausc.perception.robot.robot_state_stream import (
    EXACT_CONTROL_RECORD_SCHEMA,
    VOID_SEND_STATUS_REASON,
    BracketedExactControlTrajectory,
    BracketedRobotTrajectory,
    ExactControlRecord,
    ExactFollowerAction,
)
from ausc.perception.tracking.occlusion_runtime import (
    ActivatedOcclusionIdentity,
    RobotMaskPacket,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
JOINT_NAMES = tuple(f"openarm_left_joint{index}" for index in range(1, 8))


def _camera_frame(
    *,
    status: str = "OK",
    index: int = 12,
    t: float = 1.05,
    device_t_ms: float | None = 400.0,
    rgb: np.ndarray | None = None,
    depth: np.ndarray | None = None,
) -> CameraFrame:
    return CameraFrame(
        frame=(
            np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
            if rgb is None
            else rgb
        ),
        depth=(
            np.arange(4 * 5, dtype=np.uint16).reshape(4, 5) + 900
            if depth is None
            else depth
        ),
        t=t,
        index=index,
        status=status,
        depth_scale=1.0,
        age_s=0.0,
        device_t_ms=device_t_ms,
        time_source="orbbec_device_and_host_arrival",
    )


def _exposure(**changes) -> CameraExposureEvidence:
    values = {
        "frame_index": 12,
        "camera_id": "orbbec-main",
        "camera_host_t_s": 1.05,
        "camera_host_monotonic_ns": 1_050_000_000,
        "camera_device_t_ms": 400.0,
        "exposure_lower_host_t_s": 1.01,
        "exposure_nominal_host_t_s": 1.015,
        "exposure_upper_host_t_s": 1.02,
        "exposure_lower_monotonic_ns": 1_010_000_000,
        "exposure_nominal_monotonic_ns": 1_015_000_000,
        "exposure_upper_monotonic_ns": 1_020_000_000,
        "camera_latency_sha256": SHA_C,
        "camera_profile_sha256": SHA_A,
        "alignment_sha256": SHA_B,
    }
    values.update(changes)
    return CameraExposureEvidence(**values)


def _record(
    seq: int,
    monotonic_ns: int,
    *,
    side: str = "left",
    leader_can: str = "can1",
    follower_can: str = "can3",
    q_offset: float = 0.0,
) -> ExactControlRecord:
    q = tuple(q_offset + index * 0.01 for index in range(7))
    action_q = tuple(value + 0.02 for value in q)
    return ExactControlRecord(
        schema=EXACT_CONTROL_RECORD_SCHEMA,
        control_tick_seq=seq,
        observation_monotonic_ns=monotonic_ns,
        reference_generation=seq,
        reference_source_monotonic_ns=monotonic_ns - 100_000,
        reference_snapshot_valid=True,
        control_call_begin_monotonic_ns=monotonic_ns + 50_000,
        arm_call_return_monotonic_ns=monotonic_ns + 100_000,
        gripper_call_begin_monotonic_ns=monotonic_ns + 110_000,
        gripper_call_return_monotonic_ns=monotonic_ns + 160_000,
        control_call_end_monotonic_ns=monotonic_ns + 170_000,
        follower_q_rad=q,
        follower_qd_rad_s=(0.1,) * 7,
        follower_gripper_q_rad=-0.07,
        follower_gripper_qd_rad_s=0.01,
        follower_gripper_motor_q_rad=-0.075,
        follower_gripper_motor_qd_rad_s=0.015,
        action=ExactFollowerAction(
            arm_kp=(240.0, 240.0, 240.0, 240.0, 40.0, 48.0, 40.0),
            arm_kd=(3.0, 3.0, 3.0, 3.0, 0.3, 0.3, 0.3),
            arm_q=action_q,
            arm_dq=(0.2,) * 7,
            arm_tau=(0.0,) * 7,
            gripper_kp=25.0,
            gripper_kd=0.3,
            gripper_q=-0.08,
            gripper_dq=0.0,
            gripper_tau=0.0,
        ),
        envelope_seq=10 + (seq - 100) // 10,
        publisher_monotonic_ns=monotonic_ns + 500_000,
        controller_pid=456,
        controller_start_token="openarm-left-active",
        side=side,
        controller_mode="unilateral",
        leader_can_interface=leader_can,
        follower_can_interface=follower_can,
        joint_names=JOINT_NAMES,
        urdf_sha256=SHA_D,
        joint_mapper_sha256=SHA_E,
        pair_contract_sha256=SHA_F,
        controller_binary_sha256=SHA_D,
        controller_source_manifest_sha256=SHA_A,
        follower_config_sha256=SHA_B,
        openarm_sdk_sha256=SHA_C,
        send_status_observable=False,
        send_status_reason=VOID_SEND_STATUS_REASON,
    )


def _trajectory(
    *,
    side: str = "left",
    leader_can: str = "can1",
    follower_can: str = "can3",
    times: tuple[int, ...] = tuple(
        1_009_000_000 + offset * 1_000_000 for offset in range(13)
    ),
    sequences: tuple[int, ...] = tuple(range(100, 113)),
    lower_ns: int = 1_010_000_000,
    nominal_ns: int = 1_015_000_000,
    upper_ns: int = 1_020_000_000,
) -> BracketedExactControlTrajectory:
    return BracketedExactControlTrajectory(
        exposure_lower_ns=lower_ns,
        exposure_nominal_ns=nominal_ns,
        exposure_upper_ns=upper_ns,
        records=tuple(
            _record(
                sequence,
                timestamp,
                side=side,
                leader_can=leader_can,
                follower_can=follower_can,
                q_offset=packet_index * 0.001,
            )
            for packet_index, (sequence, timestamp) in enumerate(
                zip(sequences, times)
            )
        ),
    )


def _mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    ).hexdigest()


def _mask_packet(
    *,
    exposure: CameraExposureEvidence | None = None,
    trajectory: BracketedExactControlTrajectory | None = None,
    mask: np.ndarray | None = None,
    **changes,
) -> RobotMaskPacket:
    exposure = exposure or _exposure()
    trajectory = trajectory or _trajectory()
    mask = (
        np.zeros((4, 5), dtype=np.bool_)
        if mask is None
        else np.asarray(mask)
    )
    mask[1:3, 2:4] = True
    values = {
        "frame_index": exposure.frame_index,
        "camera_host_t_s": exposure.camera_host_t_s,
        "camera_host_monotonic_ns": exposure.camera_host_monotonic_ns,
        "camera_device_t_ms": exposure.camera_device_t_ms,
        "camera_id": exposure.camera_id,
        "camera_profile_sha256": exposure.camera_profile_sha256,
        "exposure_lower_host_t_s": exposure.exposure_lower_host_t_s,
        "exposure_nominal_host_t_s": exposure.exposure_nominal_host_t_s,
        "exposure_upper_host_t_s": exposure.exposure_upper_host_t_s,
        "exposure_lower_monotonic_ns": exposure.exposure_lower_monotonic_ns,
        "exposure_nominal_monotonic_ns": exposure.exposure_nominal_monotonic_ns,
        "exposure_upper_monotonic_ns": exposure.exposure_upper_monotonic_ns,
        "mask": mask,
        "mask_sha256": _mask_digest(mask),
        "masked_pixel_count": int(np.count_nonzero(mask)),
        "joint_sequence_first": trajectory.sequence_first,
        "joint_sequence_last": trajectory.sequence_last,
        "joint_time_first_s": trajectory.before.monotonic_ns / 1_000_000_000,
        "joint_time_last_s": trajectory.after.monotonic_ns / 1_000_000_000,
        "joint_time_first_monotonic_ns": trajectory.before.monotonic_ns,
        "joint_time_last_monotonic_ns": trajectory.after.monotonic_ns,
        "joint_max_gap_s": trajectory.max_gap_ns / 1_000_000_000,
        "joint_max_gap_ns": trajectory.max_gap_ns,
        "geometry_release_id": "openarm_geometry-" + "1" * 20,
        "geometry_sha256": SHA_D,
        "tool_release_id": "stethoscope_tool-" + "2" * 20,
        "tool_sha256": SHA_E,
        "handeye_release_id": "handeye-" + SHA_F[:20],
        "handeye_sha256": SHA_F,
        "handeye_capability_schema": "wam_openarm_handeye_v1",
        "handeye_calibration_tier": "QUALITY_GATE_COMPLETE",
        "handeye_policy_sha256": SHA_D,
        "camera_latency_sha256": exposure.camera_latency_sha256,
        "camera_projection_sha256": SHA_D,
        "occlusion_profile_sha256": SHA_E,
        "alignment_sha256": exposure.alignment_sha256,
        "joint_publisher_identity": trajectory.publisher_identity,
        "status": "OK",
        "reason": "interval_swept_openarm_robot_tool_mask",
    }
    values.update(changes)
    return RobotMaskPacket(**values)


def _activated_identity(
    packet: RobotMaskPacket | None = None,
) -> ActivatedOcclusionIdentity:
    packet = packet or _mask_packet()
    return ActivatedOcclusionIdentity(
        camera_id=packet.camera_id,
        camera_profile_sha256=packet.camera_profile_sha256,
        geometry_release_id=packet.geometry_release_id,
        geometry_sha256=packet.geometry_sha256,
        tool_release_id=packet.tool_release_id,
        tool_sha256=packet.tool_sha256,
        handeye_release_id=packet.handeye_release_id,
        handeye_sha256=packet.handeye_sha256,
        handeye_capability_schema=packet.handeye_capability_schema,
        handeye_calibration_tier=packet.handeye_calibration_tier,
        handeye_policy_sha256=packet.handeye_policy_sha256,
        camera_latency_sha256=packet.camera_latency_sha256,
        camera_projection_sha256=packet.camera_projection_sha256,
        occlusion_profile_sha256=packet.occlusion_profile_sha256,
        occlusion_runtime_profile_sha256=SHA_F,
        alignment_sha256=packet.alignment_sha256,
        joint_mapper_sha256=SHA_A,
        pair_contract_sha256=SHA_B,
        controller_binary_sha256=SHA_C,
        patient_pose_release_id="patient_pose-" + "4" * 20,
        patient_pose_sha256=SHA_D,
        depth_noise_model_sha256=SHA_E,
        back_mask_release_id="back_mask-" + "5" * 20,
        back_mask_sha256=SHA_F,
        max_camera_to_exposure_age_ns=100_000_000,
        max_exposure_interval_ns=40_000_000,
    )


def _bundle(
    *,
    camera_frame: CameraFrame | None = None,
    exposure: CameraExposureEvidence | None = None,
    trajectory: BracketedExactControlTrajectory | None = None,
    robot_mask_packet: RobotMaskPacket | None = None,
    camera_id: str = "orbbec-main",
    camera_profile_sha256: str = SHA_A,
    alignment_sha256: str = SHA_B,
    timing_artifact_sha256: str = SHA_C,
    activated_identity: ActivatedOcclusionIdentity | None = None,
) -> FrameEvidenceBundle:
    exposure = exposure or _exposure()
    trajectory = trajectory or _trajectory()
    packet = (
        robot_mask_packet
        or _mask_packet(exposure=exposure, trajectory=trajectory)
    )
    return FrameEvidenceBundle.build(
        camera_frame=camera_frame or _camera_frame(),
        camera_id=camera_id,
        camera_profile_sha256=camera_profile_sha256,
        alignment_sha256=alignment_sha256,
        timing_artifact_sha256=timing_artifact_sha256,
        exposure=exposure,
        robot_trajectory=trajectory,
        robot_mask_packet=packet,
        activated_identity=(
            activated_identity or _activated_identity(packet)
        ),
    )


def test_provisional_handeye_policy_is_hashed_into_frame_evidence() -> None:
    packet = _mask_packet(
        handeye_release_id="handeye_activation-" + "3" * 20,
        handeye_sha256="3" * 64,
        handeye_capability_schema=(
            "wam_provisional_robot_mask_handeye_capability_v1"
        ),
        handeye_calibration_tier="USER_ACCEPTED_BELOW_GATE_5MM",
        handeye_calibration_status="USER_ACCEPTED_BELOW_GATE",
        handeye_provisional=True,
        handeye_quality_gate_passed=False,
        handeye_workspace_1sigma_max_m=0.00605,
        handeye_covariance_bound_transform=(
            "transform_follower_base_from_camera"
        ),
        handeye_covariance_scope="random_only",
        handeye_systematic_bias_status="NOT_CHARACTERIZED",
        handeye_systematic_bias_bound_m=None,
        handeye_autonomous_targeting_authorized=False,
        handeye_provisional_policy_allowance_m=0.0065,
    )

    bundle = _bundle(robot_mask_packet=packet)
    metadata = bundle.canonical_metadata()["robot_mask"]

    assert metadata["handeye_calibration_status"] == "USER_ACCEPTED_BELOW_GATE"
    assert metadata["handeye_provisional"] is True
    assert metadata["handeye_workspace_1sigma_max_m"] == 0.00605
    assert metadata["handeye_autonomous_targeting_authorized"] is False

    tampered = replace(packet, handeye_autonomous_targeting_authorized=True)
    with pytest.raises(FrameEvidenceBundleError, match="provisional hand-eye"):
        _bundle(robot_mask_packet=tampered)


def test_frame_bundle_rejects_v2_handeye_relabeled_as_legacy() -> None:
    v2_release_id = "handeye_activation-" + SHA_F[:20]
    relabeled = _mask_packet(
        handeye_release_id=v2_release_id,
        handeye_capability_schema=(
            "wam_provisional_robot_mask_handeye_capability_v1"
        ),
        handeye_calibration_tier="USER_ACCEPTED_BELOW_GATE_5MM",
        handeye_calibration_status="COMPLETE",
        handeye_provisional=False,
        handeye_quality_gate_passed=None,
        handeye_workspace_1sigma_max_m=None,
        handeye_covariance_bound_transform=(
            "transform_camera_from_follower_base"
        ),
        handeye_covariance_scope="legacy_production",
        handeye_systematic_bias_status="LEGACY_NOT_APPLICABLE",
        handeye_systematic_bias_bound_m=None,
        handeye_autonomous_targeting_authorized=True,
        handeye_provisional_policy_allowance_m=0.0,
    )

    with pytest.raises(FrameEvidenceBundleError, match="hand-eye.*policy"):
        _bundle(robot_mask_packet=relabeled)


def test_frame_bundle_rejects_complete_v2_to_legacy_identity_relabel() -> None:
    provisional = _mask_packet(
        handeye_release_id="handeye_activation-" + "3" * 20,
        handeye_sha256="3" * 64,
        handeye_capability_schema=(
            "wam_provisional_robot_mask_handeye_capability_v1"
        ),
        handeye_calibration_tier="USER_ACCEPTED_BELOW_GATE_5MM",
        handeye_policy_sha256="4" * 64,
        handeye_calibration_status="USER_ACCEPTED_BELOW_GATE",
        handeye_provisional=True,
        handeye_quality_gate_passed=False,
        handeye_workspace_1sigma_max_m=0.00605,
        handeye_covariance_bound_transform=(
            "transform_follower_base_from_camera"
        ),
        handeye_covariance_scope="random_only",
        handeye_systematic_bias_status="NOT_CHARACTERIZED",
        handeye_systematic_bias_bound_m=None,
        handeye_autonomous_targeting_authorized=False,
        handeye_provisional_policy_allowance_m=0.0065,
    )
    activated_identity = _activated_identity(provisional)
    forged_legacy = replace(
        provisional,
        handeye_release_id="handeye-" + provisional.handeye_sha256[:20],
        handeye_capability_schema="wam_openarm_handeye_v1",
        handeye_calibration_tier="QUALITY_GATE_COMPLETE",
        handeye_policy_sha256=SHA_A,
        handeye_calibration_status="COMPLETE",
        handeye_provisional=False,
        handeye_quality_gate_passed=None,
        handeye_workspace_1sigma_max_m=None,
        handeye_covariance_bound_transform=(
            "transform_camera_from_follower_base"
        ),
        handeye_covariance_scope="legacy_production",
        handeye_systematic_bias_status="LEGACY_NOT_APPLICABLE",
        handeye_systematic_bias_bound_m=None,
        handeye_autonomous_targeting_authorized=True,
        handeye_provisional_policy_allowance_m=0.0,
    )

    with pytest.raises(FrameEvidenceBundleError, match="activated.*identity"):
        _bundle(
            robot_mask_packet=forged_legacy,
            activated_identity=activated_identity,
        )


def test_build_copies_one_read_only_rgbd_snapshot_and_binds_left_pair() -> None:
    camera = _camera_frame()
    packet = _mask_packet()
    bundle = _bundle(camera_frame=camera, robot_mask_packet=packet)

    assert isinstance(bundle.camera_frame, ImmutableCameraFrame)
    assert bundle.camera_frame.index == camera.index
    assert bundle.camera_frame.t == camera.t
    assert bundle.camera_frame.device_t_ms == camera.device_t_ms
    assert bundle.camera_frame.camera_id == "orbbec-main"
    assert bundle.camera_frame.camera_profile_sha256 == SHA_A
    assert bundle.camera_frame.alignment_sha256 == SHA_B
    assert bundle.index == camera.index

    canonical = bundle.canonical_metadata_bytes()
    assert bundle.bundle_id == (
        "frame_evidence-" + hashlib.sha256(canonical).hexdigest()
    )
    assert canonical == bundle.canonical_metadata_bytes()
    assert bundle.t == camera.t
    assert bundle.robot_trajectory.before.side == "left"
    assert bundle.robot_trajectory.before.leader_can_interface == "can1"
    assert bundle.robot_trajectory.before.follower_can_interface == "can3"
    assert bundle.robot_mask_packet.status == "OK"
    assert bundle.bundle_id.startswith("frame_evidence-")
    assert len(bundle.bundle_id) == len("frame_evidence-") + 64

    camera.frame[:] = 255
    camera.depth[:] = 0
    packet.mask.setflags(write=True)
    packet.mask[:] = False
    assert not np.all(bundle.camera_frame.frame == 255)
    assert np.all(bundle.camera_frame.depth > 0)
    assert bundle.robot_mask_packet.masked_pixel_count == 4
    assert np.count_nonzero(bundle.robot_mask_packet.mask) == 4

    for array in (
        bundle.camera_frame.frame,
        bundle.camera_frame.depth,
        bundle.robot_mask_packet.mask,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        bundle.camera_frame.index = 99


def test_bundle_id_is_deterministic_and_binds_camera_and_robot_bytes() -> None:
    first = _bundle()
    second = _bundle()
    changed_rgb = _camera_frame()
    changed_rgb.frame[0, 0, 0] += 1
    changed_robot = _trajectory()
    changed_records = list(changed_robot.records)
    changed_records[1] = replace(
        changed_records[1],
        action=replace(changed_records[1].action, arm_q=(0.5,) * 7),
    )
    changed_robot = replace(changed_robot, records=tuple(changed_records))

    assert first.bundle_id == second.bundle_id
    assert _bundle(camera_frame=changed_rgb).bundle_id != first.bundle_id
    assert (
        _bundle(
            trajectory=changed_robot,
            robot_mask_packet=_mask_packet(trajectory=changed_robot),
        ).bundle_id
        != first.bundle_id
    )


@pytest.mark.parametrize(
    ("camera", "message"),
    (
        (_camera_frame(status="STALE"), "camera frame status"),
        (_camera_frame(rgb=np.zeros((4, 5, 3), dtype=np.float32)), "RGB"),
        (_camera_frame(depth=np.zeros((4, 5), dtype=np.float32)), "depth"),
        (_camera_frame(device_t_ms=None), "device timestamp"),
    ),
)
def test_missing_stale_or_malformed_rgbd_frame_fails_closed(
    camera: CameraFrame,
    message: str,
) -> None:
    with pytest.raises(FrameEvidenceBundleError, match=message):
        _bundle(camera_frame=camera)


@pytest.mark.parametrize(
    ("exposure_changes", "build_changes", "message"),
    (
        ({"frame_index": 13}, {}, "frame index"),
        ({"camera_host_t_s": 1.050000001}, {}, "host timestamp"),
        ({"camera_device_t_ms": 401.0}, {}, "device timestamp"),
        ({"camera_id": "other-camera"}, {}, "camera identity"),
        ({"camera_profile_sha256": SHA_D}, {}, "camera profile"),
        ({"alignment_sha256": SHA_D}, {}, "alignment"),
        ({"camera_latency_sha256": SHA_D}, {}, "timing artifact"),
        (
            {"exposure_lower_host_t_s": 1.011},
            {},
            "exposure interval",
        ),
        ({}, {"camera_id": "other-camera"}, "camera identity"),
        ({}, {"camera_profile_sha256": SHA_D}, "camera profile"),
        ({}, {"alignment_sha256": SHA_D}, "alignment"),
        ({}, {"timing_artifact_sha256": SHA_D}, "timing artifact"),
    ),
)
def test_camera_exposure_identity_or_clock_mismatch_fails_closed(
    exposure_changes: dict,
    build_changes: dict,
    message: str,
) -> None:
    exposure = _exposure(**exposure_changes)
    with pytest.raises(FrameEvidenceBundleError, match=message):
        _bundle(
            exposure=exposure,
            robot_mask_packet=_mask_packet(exposure=exposure),
            **build_changes,
        )


@pytest.mark.parametrize(
    ("trajectory", "message"),
    (
        (_trajectory(side="right"), "left"),
        (_trajectory(leader_can="can0"), "can1"),
        (_trajectory(follower_can="can2"), "can3"),
        (_trajectory(lower_ns=1_011_000_000), "exposure interval"),
        (
            _trajectory(
                times=(1_011_000_000, 1_015_000_000, 1_020_000_000),
                sequences=(100, 101, 102),
            ),
            "bracket",
        ),
        (
            _trajectory(
                times=(
                    1_000_000_000,
                    1_010_000_000,
                    1_031_000_000,
                    1_040_000_000,
                ),
            ),
            "2 ms",
        ),
        (
            _trajectory(
                sequences=(100, 102) + tuple(range(103, 114)),
            ),
            "sequence",
        ),
    ),
)
def test_robot_trajectory_must_be_exact_contiguous_left_can1_to_can3_bracket(
    trajectory: BracketedExactControlTrajectory,
    message: str,
) -> None:
    with pytest.raises(FrameEvidenceBundleError, match=message):
        _bundle(
            trajectory=trajectory,
            robot_mask_packet=_mask_packet(trajectory=trajectory),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"status": "NO_DATA"}, "mask status"),
        ({"frame_index": 13}, "frame index"),
        ({"camera_host_t_s": 1.051}, "host timestamp"),
        ({"camera_device_t_ms": 401.0}, "device timestamp"),
        ({"camera_id": "other-camera"}, "camera identity"),
        ({"camera_profile_sha256": SHA_D}, "camera profile"),
        ({"alignment_sha256": SHA_D}, "alignment"),
        ({"camera_latency_sha256": SHA_D}, "timing artifact"),
        ({"exposure_nominal_monotonic_ns": 1_016_000_000}, "exposure interval"),
        ({"joint_sequence_first": 99}, "robot trajectory"),
        ({"joint_time_last_monotonic_ns": 1_040_000_000}, "robot trajectory"),
        ({"joint_publisher_identity": "wrong"}, "publisher identity"),
        ({"mask_sha256": "0" * 64}, "mask SHA-256"),
        ({"masked_pixel_count": 5}, "masked pixel count"),
    ),
)
def test_robot_mask_must_be_the_exact_same_frame_and_trajectory(
    changes: dict,
    message: str,
) -> None:
    with pytest.raises(FrameEvidenceBundleError, match=message):
        _bundle(robot_mask_packet=_mask_packet(**changes))


def test_mask_shape_must_match_aligned_depth_shape() -> None:
    mask = np.zeros((3, 5), dtype=np.bool_)
    with pytest.raises(FrameEvidenceBundleError, match="mask shape"):
        _bundle(robot_mask_packet=_mask_packet(mask=mask))


def test_bundle_binds_exact_action_digest_tick_range_and_full_source_closure() -> None:
    original = _bundle()
    trajectory = _trajectory()
    changed_action = replace(
        trajectory.records[5],
        action=replace(trajectory.records[5].action, arm_tau=(0.25,) * 7),
    )
    action_trajectory = replace(
        trajectory,
        records=trajectory.records[:5]
        + (changed_action,)
        + trajectory.records[6:],
    )
    action_bundle = _bundle(
        trajectory=action_trajectory,
        robot_mask_packet=_mask_packet(trajectory=action_trajectory),
    )
    source_trajectory = replace(
        trajectory,
        records=tuple(
            replace(record, openarm_sdk_sha256="1" * 64)
            for record in trajectory.records
        ),
    )
    source_bundle = _bundle(
        trajectory=source_trajectory,
        robot_mask_packet=_mask_packet(trajectory=source_trajectory),
    )

    assert original.robot_exact_action_tick_first == 100
    assert original.robot_exact_action_tick_last == 112
    assert original.robot_exact_action_count == 13
    assert len(original.robot_exact_action_sha256) == 64
    assert action_bundle.robot_exact_action_sha256 != original.robot_exact_action_sha256
    assert source_bundle.robot_exact_action_sha256 == original.robot_exact_action_sha256
    assert source_bundle.bundle_id != original.bundle_id
    assert (
        source_bundle.canonical_metadata()["robot_source_provenance"]
        ["openarm_sdk_sha256"]
        == "1" * 64
    )


def test_bundle_exposes_read_only_index_and_time_for_worker_fanout() -> None:
    bundle = _bundle()

    assert bundle.index == 12
    assert bundle.t == 1.05
    with pytest.raises(FrozenInstanceError):
        bundle.index = 13
    with pytest.raises(FrozenInstanceError):
        bundle.t = 2.0


def test_production_bundle_rejects_legacy_v1_robot_trajectory() -> None:
    legacy = BracketedRobotTrajectory(
        exposure_lower_ns=1_010_000_000,
        exposure_nominal_ns=1_015_000_000,
        exposure_upper_ns=1_020_000_000,
        packets=(),
    )

    with pytest.raises(FrameEvidenceBundleError, match="exact control.*v2"):
        FrameEvidenceBundle.build(
            camera_frame=_camera_frame(),
            camera_id="orbbec-main",
            camera_profile_sha256=SHA_A,
            alignment_sha256=SHA_B,
            timing_artifact_sha256=SHA_C,
            exposure=_exposure(),
            robot_trajectory=legacy,
            robot_mask_packet=_mask_packet(),
            activated_identity=_activated_identity(),
        )


def test_factory_path_validates_once_external_path_still_validates(monkeypatch):
    """Owner-authorized de-dup (2026-08-12): build() performs the validation
    suite exactly once; any non-factory construction still re-verifies."""

    import ausc.perception.robot.frame_evidence_bundle as fev

    calls = {"mask": 0}
    original = fev._validate_mask

    def counting_validate_mask(*args, **kwargs):
        calls["mask"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fev, "_validate_mask", counting_validate_mask)
    bundle = _bundle()
    assert calls["mask"] == 1  # was 2 before the de-dup

    # External construction (no witness) re-runs the full suite.
    calls["mask"] = 0
    clone = fev.FrameEvidenceBundle(
        camera_frame=bundle.camera_frame,
        exposure=bundle.exposure,
        robot_trajectory=bundle.robot_trajectory,
        robot_mask_packet=bundle.robot_mask_packet,
        activated_identity=bundle.activated_identity,
        timing_artifact_sha256=bundle.timing_artifact_sha256,
        robot_trajectory_sha256=bundle.robot_trajectory_sha256,
        robot_exact_action_sha256=bundle.robot_exact_action_sha256,
        bundle_id=bundle.bundle_id,
    )
    assert calls["mask"] == 1
    assert clone.bundle_id == bundle.bundle_id

    # And a tampered digest on the external path still fails closed.
    with pytest.raises(fev.FrameEvidenceBundleError):
        fev.FrameEvidenceBundle(
            camera_frame=bundle.camera_frame,
            exposure=bundle.exposure,
            robot_trajectory=bundle.robot_trajectory,
            robot_mask_packet=bundle.robot_mask_packet,
            activated_identity=bundle.activated_identity,
            timing_artifact_sha256=bundle.timing_artifact_sha256,
            robot_trajectory_sha256="0" * 64,
            robot_exact_action_sha256=bundle.robot_exact_action_sha256,
            bundle_id=bundle.bundle_id,
        )
