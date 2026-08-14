from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from ausc.perception.robot.camera_exposure_evidence import CameraExposureEvidence
from ausc.perception.robot.robot_mask_provider import (
    ActivatedHandEye,
    CameraProjectionProfile,
    KinematicCollisionLink,
    OpenArmCollisionGeometry,
    OpenArmToolGeometry,
    ProvisionalRobotMaskHandEye,
    ProductionRobotMaskProvider,
    RobotMaskProviderError,
    RobotMaskUncertaintyProfile,
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
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
JOINT_NAMES = tuple(f"openarm_left_joint{index}" for index in range(1, 8))


def _box(center, half=(0.025, 0.022, 0.018)):
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    return np.asarray(
        [
            center + np.asarray((sx, sy, sz)) * half
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )


def _transform(x=0.0, y=0.0, z=0.0):
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = (x, y, z)
    return value


def _geometry():
    links = []
    for index in range(7):
        links.append(
            KinematicCollisionLink(
                name=f"openarm_left_link{index + 1}",
                parent_index=index - 1,
                joint_index=index,
                transform_parent_from_joint_zero=(
                    _transform(z=1.0) if index == 0 else _transform(x=0.07)
                ),
                joint_axis=np.asarray((0.0, 0.0, 1.0)),
                vertices_link_m=_box((0.035, 0.0, 0.0)),
            )
        )
    return OpenArmCollisionGeometry.create(
        source="official_openarm_v10_urdf_collision_meshes",
        side="left",
        joint_names=JOINT_NAMES,
        urdf_sha256=SHA_A,
        links=tuple(links),
    )


def _tool(*, outside=False):
    return OpenArmToolGeometry.create(
        source="commissioned_openarm_stethoscope_tool",
        parent_link_index=6,
        transform_parent_from_tool=_transform(x=10.0 if outside else 0.07),
        vertices_tool_m=_box((0.05, 0.0, 0.0), (0.022, 0.028, 0.02)),
    )


def _handeye(*, x=0.0, z=0.0, covariance_scale=1e-8, status="COMPLETE"):
    return ActivatedHandEye.create(
        camera_id="orbbec-main",
        side="left",
        transform_camera_from_follower_base=_transform(x=x, z=z),
        covariance_se3=np.eye(6, dtype=np.float64) * covariance_scale,
        status=status,
    )


def _provisional_handeye_payload():
    follower_base_from_camera = _transform(x=0.30, y=-0.20, z=0.70)
    camera_from_follower_base = np.linalg.inv(follower_base_from_camera)
    covariance = np.diag((1.0e-6, 1.1e-6, 1.2e-6, 4.0e-6, 5.0e-6, 6.0e-6))
    return {
        "acceptance_note": "William accepted approximately 5 mm for bootstrap use.",
        "authorized_scopes": [
            "bootstrap_teleop_logging",
            "conservative_robot_mask",
        ],
        "autonomous_targeting_authorized": False,
        "bootstrap_reproducibility": {
            "missing": ["bootstrap_seed", "bootstrap_sample_count", "resample_policy"],
            "status": "INCOMPLETE",
        },
        "calibration_status": "USER_ACCEPTED_BELOW_GATE",
        "camera_id": "orbbec-main",
        "code_release": "handeye_calibration_code-" + "a" * 20,
        "covariance_scope": "random_only",
        "covariance_se3": {
            "bound_transform": "transform_follower_base_from_camera",
            "definition": (
                "empirical bootstrap cov of delta=[Rodrigues(R_ref^T R_boot), "
                "t_boot - t_ref]; random precision only, excludes systematic bias"
            ),
            "matrix": covariance.tolist(),
            "order": ["rx_rad", "ry_rad", "rz_rad", "tx_m", "ty_m", "tz_m"],
        },
        "failed_quality_gate_m": 0.003,
        "provisional": True,
        "quality": {"result_precision_mm_1sigma_max": 6.05,
                    "result_precision_mm_1sigma_median": 4.96},
        "quality_gate_passed": False,
        "schema": "wam_openarm_handeye_activation_v2",
        "side": "left",
        "source_evidence": {"sample_count": 23},
        "source_frame": "orbbec_color_optical_frame",
        "source_run_id": "handeye_left_20260718T195334Z_70cf253b",
        "systematic_bias_bound_m": None,
        "systematic_bias_status": "NOT_CHARACTERIZED",
        "target_frame": "openarm_left_link0",
        "tip_offset_frame": "openarm_left_hand_tcp",
        "tip_offset_tcp_m": [0.002, 0.028, 0.180],
        "transform_camera_from_follower_base": camera_from_follower_base.tolist(),
        "transform_follower_base_from_camera": follower_base_from_camera.tolist(),
        "workspace_1sigma_m": {"max": 0.00605, "median": 0.00496},
    }


def _provisional_handeye():
    payload = _provisional_handeye_payload()
    body = (
        json.dumps(payload, allow_nan=False, ensure_ascii=False,
                   separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    return ProvisionalRobotMaskHandEye.from_activation_v2(
        payload,
        artifact_sha256=hashlib.sha256(body).hexdigest(),
        publication_status="COMPLETE",
    )


def _projection(
    *,
    width=160,
    height=120,
    fx=100.0,
    fy=100.0,
    cx=80.0,
    cy=60.0,
):
    return CameraProjectionProfile.create(
        camera_id="orbbec-main",
        camera_profile_sha256=SHA_B,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        min_projection_depth_m=0.05,
    )


def _profile(
    *,
    metric_error=0.0005,
    timing_s=0.0002,
    projection_px=0.25,
    sigma=3.0,
):
    return RobotMaskUncertaintyProfile.create(
        sigma=sigma,
        base_dilation_px=1.0,
        link_geometry_error_m=metric_error,
        tool_geometry_error_m=metric_error,
        timing_uncertainty_s=timing_s,
        projection_error_px=projection_px,
    )


def _record(seq, t_ns, q, qd=None, *, leader_can="can1", follower_can="can3"):
    q = tuple(float(value) for value in q)
    qd = tuple(float(value) for value in (qd or (0.0,) * 7))
    return ExactControlRecord(
        schema=EXACT_CONTROL_RECORD_SCHEMA,
        control_tick_seq=seq,
        observation_monotonic_ns=t_ns,
        reference_generation=seq,
        reference_source_monotonic_ns=t_ns - 100_000,
        reference_snapshot_valid=True,
        control_call_begin_monotonic_ns=t_ns + 50_000,
        arm_call_return_monotonic_ns=t_ns + 100_000,
        gripper_call_begin_monotonic_ns=t_ns + 110_000,
        gripper_call_return_monotonic_ns=t_ns + 160_000,
        control_call_end_monotonic_ns=t_ns + 170_000,
        follower_q_rad=q,
        follower_qd_rad_s=qd,
        follower_gripper_q_rad=0.0,
        follower_gripper_qd_rad_s=0.0,
        follower_gripper_motor_q_rad=0.0,
        follower_gripper_motor_qd_rad_s=0.0,
        action=ExactFollowerAction(
            arm_kp=(240.0, 240.0, 240.0, 240.0, 40.0, 48.0, 40.0),
            arm_kd=(3.0, 3.0, 3.0, 3.0, 0.3, 0.3, 0.3),
            arm_q=q,
            arm_dq=qd,
            arm_tau=(0.0,) * 7,
            gripper_kp=25.0,
            gripper_kd=0.3,
            gripper_q=0.0,
            gripper_dq=0.0,
            gripper_tau=0.0,
        ),
        envelope_seq=10 + (seq - 100) // 10,
        publisher_monotonic_ns=t_ns + 500_000,
        controller_pid=456,
        controller_start_token="openarm-left-commissioning",
        side="left",
        controller_mode="unilateral",
        leader_can_interface=leader_can,
        follower_can_interface=follower_can,
        joint_names=JOINT_NAMES,
        urdf_sha256=SHA_A,
        joint_mapper_sha256=SHA_C,
        pair_contract_sha256=SHA_D,
        controller_binary_sha256=SHA_E,
        controller_source_manifest_sha256=SHA_A,
        follower_config_sha256=SHA_B,
        openarm_sdk_sha256=SHA_C,
        send_status_observable=False,
        send_status_reason=VOID_SEND_STATUS_REASON,
    )


def _trajectory(*, start_q=0.0, end_q=0.0, leader_can="can1", follower_can="can3"):
    records = []
    samples = tuple(range(13))
    for offset in samples:
        seq = 100 + offset
        t_ns = 1_009_000_000 + offset * 1_000_000
        alpha = offset / (len(samples) - 1)
        q0 = start_q + alpha * (end_q - start_q)
        records.append(
            _record(
                seq,
                t_ns,
                (q0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                ((end_q - start_q) / 0.012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                leader_can=leader_can,
                follower_can=follower_can,
            )
        )
    return BracketedExactControlTrajectory(
        exposure_lower_ns=1_010_000_000,
        exposure_nominal_ns=1_015_000_000,
        exposure_upper_ns=1_020_000_000,
        records=tuple(records),
    )


def _exposure():
    return CameraExposureEvidence(
        frame_index=12,
        camera_id="orbbec-main",
        camera_host_t_s=1.05,
        camera_host_monotonic_ns=1_050_000_000,
        camera_device_t_ms=400.0,
        exposure_lower_host_t_s=1.01,
        exposure_nominal_host_t_s=1.015,
        exposure_upper_host_t_s=1.02,
        exposure_lower_monotonic_ns=1_010_000_000,
        exposure_nominal_monotonic_ns=1_015_000_000,
        exposure_upper_monotonic_ns=1_020_000_000,
        camera_latency_sha256=SHA_C,
        camera_profile_sha256=SHA_B,
        alignment_sha256=SHA_D,
    )


def _identity(geometry, tool, handeye, projection, profile):
    return ActivatedOcclusionIdentity(
        camera_id=projection.camera_id,
        camera_profile_sha256=projection.camera_profile_sha256,
        geometry_release_id=geometry.geometry_release_id,
        geometry_sha256=geometry.geometry_sha256,
        tool_release_id=tool.tool_release_id,
        tool_sha256=tool.tool_sha256,
        handeye_release_id=handeye.handeye_release_id,
        handeye_sha256=handeye.handeye_sha256,
        handeye_capability_schema=handeye.capability_schema,
        handeye_calibration_tier=handeye.calibration_tier,
        handeye_policy_sha256=handeye.policy_sha256,
        camera_latency_sha256=SHA_C,
        camera_projection_sha256=projection.camera_projection_sha256,
        occlusion_profile_sha256=profile.occlusion_profile_sha256,
        occlusion_runtime_profile_sha256=SHA_A,
        alignment_sha256=SHA_D,
        joint_mapper_sha256=SHA_C,
        pair_contract_sha256=SHA_D,
        controller_binary_sha256=SHA_E,
        patient_pose_release_id="patient_pose-" + "4" * 20,
        patient_pose_sha256=SHA_A,
        depth_noise_model_sha256=SHA_B,
        back_mask_release_id="back_mask-" + "5" * 20,
        back_mask_sha256=SHA_C,
        max_camera_to_exposure_age_ns=100_000_000,
        max_exposure_interval_ns=40_000_000,
    )


def _provider(
    *,
    geometry=None,
    tool=None,
    handeye=None,
    projection=None,
    profile=None,
    identity=None,
):
    geometry = geometry or _geometry()
    tool = tool or _tool()
    handeye = handeye or _handeye()
    projection = projection or _projection()
    profile = profile or _profile()
    identity = identity or _identity(
        geometry, tool, handeye, projection, profile
    )
    return ProductionRobotMaskProvider(
        activated_identity=identity,
        geometry=geometry,
        tool=tool,
        handeye=handeye,
        projection=projection,
        uncertainty_profile=profile,
    )


def test_static_left_openarm_builds_exact_runtime_packet():
    provider = _provider()
    exposure = _exposure()
    packet = provider.build(
        exposure=exposure,
        trajectory=_trajectory(),
    )

    assert packet.status == "OK"
    assert packet.mask.dtype == np.bool_
    assert packet.mask.shape == (120, 160)
    assert packet.masked_pixel_count > 0
    assert packet.joint_sequence_first == 100
    assert packet.joint_sequence_last == 112
    assert "leader_can=can1" in packet.joint_publisher_identity
    assert "follower_can=can3" in packet.joint_publisher_identity
    packet.validate_for_frame(
        frame_index=exposure.frame_index,
        camera_host_t_s=exposure.camera_host_t_s,
        camera_id=exposure.camera_id,
        camera_profile_sha256=exposure.camera_profile_sha256,
        depth_shape=(120, 160),
        activated_identity=provider.activated_identity,
    )


def test_interval_sweep_is_larger_than_nominal_static_mask():
    provider = _provider()
    moving = provider.build(
        exposure=_exposure(),
        trajectory=_trajectory(start_q=-0.45, end_q=0.45),
    )
    nominal_only = provider.build(
        exposure=_exposure(),
        trajectory=_trajectory(start_q=0.0, end_q=0.0),
    )

    assert moving.masked_pixel_count > nominal_only.masked_pixel_count
    assert np.any(moving.mask & ~nominal_only.mask)


def test_tool_geometry_is_included_in_mask_union():
    geometry = _geometry()
    handeye = _handeye()
    projection = _projection()
    profile = _profile()
    visible_tool = _tool()
    outside_tool = _tool(outside=True)
    with_tool = _provider(
        geometry=geometry,
        tool=visible_tool,
        handeye=handeye,
        projection=projection,
        profile=profile,
    ).build(exposure=_exposure(), trajectory=_trajectory())
    without_visible_tool = _provider(
        geometry=geometry,
        tool=outside_tool,
        handeye=handeye,
        projection=projection,
        profile=profile,
    ).build(exposure=_exposure(), trajectory=_trajectory())

    assert with_tool.masked_pixel_count > without_visible_tool.masked_pixel_count
    assert np.any(with_tool.mask & ~without_visible_tool.mask)


@pytest.mark.parametrize(
    "failure",
    (
        "wrong_can_pair",
        "wrong_geometry_pin",
        "incomplete_handeye",
        "camera_profile_mismatch",
        "camera_projection_pin",
        "nonbracketed_trajectory",
        "nonfinite_fk",
    ),
)
def test_mismatched_or_uncommissioned_inputs_fail_closed(failure):
    geometry = _geometry()
    tool = _tool()
    handeye = _handeye()
    projection = _projection()
    profile = _profile()
    identity = _identity(geometry, tool, handeye, projection, profile)
    trajectory = _trajectory()
    exposure = _exposure()

    if failure == "wrong_can_pair":
        trajectory = _trajectory(leader_can="can0", follower_can="can2")
    elif failure == "wrong_geometry_pin":
        identity = replace(identity, geometry_sha256="f" * 64)
    elif failure == "incomplete_handeye":
        handeye = _handeye(status="STAGING")
    elif failure == "camera_profile_mismatch":
        exposure = replace(exposure, camera_profile_sha256="f" * 64)
    elif failure == "camera_projection_pin":
        identity = replace(identity, camera_projection_sha256="f" * 64)
    elif failure == "nonbracketed_trajectory":
        trajectory = replace(
            trajectory,
            exposure_lower_ns=1_011_000_000,
        )
    elif failure == "nonfinite_fk":
        first = replace(
            trajectory.records[0],
            follower_q_rad=(float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        trajectory = replace(
            trajectory,
            records=(first,) + trajectory.records[1:],
        )

    with pytest.raises(RobotMaskProviderError):
        ProductionRobotMaskProvider(
            activated_identity=identity,
            geometry=geometry,
            tool=tool,
            handeye=handeye,
            projection=projection,
            uncertainty_profile=profile,
        ).build(exposure=exposure, trajectory=trajectory)


def test_missing_collision_artifact_cannot_return_empty_ok_mask():
    geometry = _geometry()
    tool = _tool()
    handeye = _handeye()
    projection = _projection()
    profile = _profile()
    identity = _identity(geometry, tool, handeye, projection, profile)

    with pytest.raises(RobotMaskProviderError, match="geometry"):
        ProductionRobotMaskProvider(
            activated_identity=identity,
            geometry=None,
            tool=tool,
            handeye=handeye,
            projection=projection,
            uncertainty_profile=profile,
        )


def test_handeye_completion_status_is_bound_into_its_content_hash():
    staging = _handeye(status="STAGING")

    with pytest.raises(RobotMaskProviderError, match="SHA-256"):
        replace(staging, status="COMPLETE")


def test_provisional_v2_handeye_preserves_frames_covariance_and_policy():
    handeye = _provisional_handeye()

    assert handeye.lifecycle_status == "COMPLETE"
    assert handeye.calibration_status == "USER_ACCEPTED_BELOW_GATE"
    assert handeye.provisional is True
    assert handeye.quality_gate_passed is False
    assert handeye.failed_quality_gate_m == pytest.approx(0.003)
    assert handeye.workspace_1sigma_median == pytest.approx(0.00496)
    assert handeye.workspace_1sigma_max == pytest.approx(0.00605)
    assert handeye.covariance_bound_transform == (
        "transform_follower_base_from_camera"
    )
    assert handeye.covariance_scope == "random_only"
    assert handeye.systematic_bias_status == "NOT_CHARACTERIZED"
    assert handeye.systematic_bias_bound_m is None
    assert handeye.provisional_policy_allowance_m == pytest.approx(0.0065)
    assert handeye.autonomous_targeting_authorized is False
    assert handeye.authorized_scopes == (
        "bootstrap_teleop_logging",
        "conservative_robot_mask",
    )
    assert np.allclose(
        handeye.transform_camera_from_follower_base
        @ handeye.transform_follower_base_from_camera,
        np.eye(4),
    )
    assert handeye.handeye_release_id == (
        "handeye_activation-" + handeye.handeye_sha256[:20]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update({"autonomous_targeting_authorized": True}),
            "autonomous",
        ),
        (
            lambda payload: payload.update({"authorized_scopes": ["bootstrap_teleop_logging"]}),
            "scope",
        ),
        (
            lambda payload: payload.update({"quality_gate_passed": True}),
            "quality",
        ),
        (
            lambda payload: payload["covariance_se3"].update(
                {"bound_transform": "transform_camera_from_follower_base"}
            ),
            "covariance",
        ),
    ],
)
def test_provisional_v2_handeye_policy_cannot_be_relaxed(mutation, message):
    payload = _provisional_handeye_payload()
    mutation(payload)
    body = (
        json.dumps(payload, allow_nan=False, ensure_ascii=False,
                   separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")

    with pytest.raises(RobotMaskProviderError, match=message):
        ProvisionalRobotMaskHandEye.from_activation_v2(
            payload,
            artifact_sha256=hashlib.sha256(body).hexdigest(),
            publication_status="COMPLETE",
        )


def test_provisional_v2_handeye_is_accepted_only_for_conservative_robot_mask():
    handeye = _provisional_handeye()
    provider = _provider(handeye=handeye)

    packet = provider.build(exposure=_exposure(), trajectory=_trajectory())

    assert packet.status == "OK"
    assert packet.handeye_sha256 == handeye.handeye_sha256
    assert packet.handeye_calibration_status == "USER_ACCEPTED_BELOW_GATE"
    assert packet.handeye_provisional is True
    assert packet.handeye_quality_gate_passed is False
    assert packet.handeye_workspace_1sigma_max_m == pytest.approx(0.00605)
    assert packet.handeye_covariance_bound_transform == (
        "transform_follower_base_from_camera"
    )
    assert packet.handeye_covariance_scope == "random_only"
    assert packet.handeye_systematic_bias_status == "NOT_CHARACTERIZED"
    assert packet.handeye_systematic_bias_bound_m is None
    assert packet.handeye_autonomous_targeting_authorized is False
    assert packet.handeye_provisional_policy_allowance_m == pytest.approx(0.0065)
    assert provider.handeye.calibration_status == "USER_ACCEPTED_BELOW_GATE"


def test_trusted_empty_ok_mask_requires_geometry_proven_outside_fov():
    geometry = _geometry()
    tool = _tool()
    handeye = _handeye(x=20.0)
    projection = _projection()
    profile = _profile()
    provider = _provider(
        geometry=geometry,
        tool=tool,
        handeye=handeye,
        projection=projection,
        profile=profile,
    )

    packet = provider.build(
        exposure=_exposure(),
        trajectory=_trajectory(),
    )

    assert packet.status == "OK"
    assert packet.masked_pixel_count == 0
    assert not packet.mask.any()
    assert packet.reason == "robot_and_tool_proven_outside_camera_fov"


def test_geometry_behind_or_straddling_the_near_plane_is_clipped_not_rejected():
    """2026-08-11 contract change: vertices at/behind the near plane cannot
    occlude the image. Fully-behind components are skipped; a component
    STRADDLING the plane (live on this rig: proximal links mounted beside the
    camera) is clipped to its in-front vertices instead of rejecting every
    bracket — the old reject deadlocked tracker start permanently."""
    geometry = _geometry()
    tool = _tool()
    projection = _projection()
    profile = _profile(metric_error=0.1)

    # fully behind: not an occluder — build succeeds, no raise
    provider = _provider(
        geometry=geometry,
        tool=tool,
        handeye=_handeye(z=-1.05),
        projection=projection,
        profile=profile,
    )
    packet = provider.build(exposure=_exposure(), trajectory=_trajectory())
    assert packet is not None

    # straddling: mesh spans the near plane — clipped projection, no raise
    provider = _provider(
        geometry=geometry,
        tool=tool,
        handeye=_handeye(z=-0.01),
        projection=projection,
        profile=profile,
    )
    packet = provider.build(exposure=_exposure(), trajectory=_trajectory())
    assert packet is not None


def test_offscreen_geometry_paints_local_reach_not_full_mask():
    """2026-08-14: the nominal-outside full-mask fallback is GONE. Off-frame
    geometry whose uncertainty disc reaches the frame paints its honest local
    reach (extended-coordinate rasterization); it must produce a non-empty,
    non-full mask instead of blinding the whole frame — the crude fallback
    fired on every real frame (arm's lower links are always off-camera) and
    zeroed points across three sealed sessions."""
    geometry = _geometry()
    tool = _tool()
    handeye = _handeye(x=0.5, covariance_scale=1e-5)
    projection = _projection(
        width=32,
        height=32,
        cx=16.0,
        cy=16.0,
    )
    profile = _profile(
        metric_error=0.1,
        timing_s=0.02,
        projection_px=20.0,
        sigma=6.0,
    )
    provider = _provider(
        geometry=geometry,
        tool=tool,
        handeye=handeye,
        projection=projection,
        profile=profile,
    )

    packet = provider.build(
        exposure=_exposure(),
        trajectory=_trajectory(),
    )

    assert packet.reason == "interval_swept_openarm_robot_tool_mask"
    assert 0 < packet.masked_pixel_count
    assert packet.masked_pixel_count == int(np.count_nonzero(packet.mask))


@pytest.mark.parametrize(
    ("field", "wrong_sha", "label"),
    (
        ("joint_mapper_sha256", SHA_A, "joint mapper"),
        ("pair_contract_sha256", SHA_B, "pair contract"),
        ("controller_binary_sha256", SHA_B, "controller binary"),
    ),
)
def test_every_trajectory_packet_hash_must_match_activated_pin(
    field,
    wrong_sha,
    label,
):
    trajectory = _trajectory()
    trajectory = replace(
        trajectory,
        records=tuple(
            replace(packet, **{field: wrong_sha})
            for packet in trajectory.records
        ),
    )

    with pytest.raises(RobotMaskProviderError, match=label):
        _provider().build(
            exposure=_exposure(),
            trajectory=trajectory,
        )


@pytest.mark.parametrize(
    ("field", "label"),
    (
        ("controller_source_manifest_sha256", "source manifest"),
        ("follower_config_sha256", "follower config"),
        ("openarm_sdk_sha256", "OpenArm SDK"),
    ),
)
def test_every_v2_source_closure_hash_is_required_and_stable(field, label):
    trajectory = _trajectory()
    changed = replace(trajectory.records[4], **{field: "f" * 64})
    trajectory = replace(
        trajectory,
        records=trajectory.records[:4] + (changed,) + trajectory.records[5:],
    )

    with pytest.raises(RobotMaskProviderError, match=label):
        _provider().build(exposure=_exposure(), trajectory=trajectory)


def test_production_mask_rejects_legacy_v1_trajectory_type():
    legacy = BracketedRobotTrajectory(
        exposure_lower_ns=1_010_000_000,
        exposure_nominal_ns=1_015_000_000,
        exposure_upper_ns=1_020_000_000,
        packets=(),
    )

    with pytest.raises(RobotMaskProviderError, match="exact control.*v2"):
        _provider().build(exposure=_exposure(), trajectory=legacy)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reference_generation", True),
        ("reference_source_monotonic_ns", 1_008_900_000.0),
        ("control_call_begin_monotonic_ns", 1_009_050_000.0),
        ("publisher_monotonic_ns", 1_009_500_000.0),
        ("envelope_seq", -1),
    ),
)
def test_mask_revalidates_exact_v2_integer_timing_when_dataclass_is_bypassed(
    field,
    value,
):
    trajectory = _trajectory()
    changed = replace(trajectory.records[0], **{field: value})
    trajectory = replace(
        trajectory,
        records=(changed,) + trajectory.records[1:],
    )

    with pytest.raises(RobotMaskProviderError, match="exact control record"):
        _provider().build(exposure=_exposure(), trajectory=trajectory)


def test_larger_calibrated_uncertainty_expands_mask():
    geometry = _geometry()
    tool = _tool()
    handeye = _handeye()
    projection = _projection()
    low = _profile(metric_error=0.0001, timing_s=0.00001, projection_px=0.05)
    high = _profile(metric_error=0.012, timing_s=0.003, projection_px=2.0)
    trajectory = _trajectory(start_q=-0.05, end_q=0.05)

    low_mask = _provider(
        geometry=geometry,
        tool=tool,
        handeye=handeye,
        projection=projection,
        profile=low,
    ).build(exposure=_exposure(), trajectory=trajectory)
    high_mask = _provider(
        geometry=geometry,
        tool=tool,
        handeye=handeye,
        projection=projection,
        profile=high,
    ).build(exposure=_exposure(), trajectory=trajectory)

    assert high_mask.masked_pixel_count > low_mask.masked_pixel_count
    assert np.all(high_mask.mask[low_mask.mask])


def test_learned_mask_can_only_add_pixels_to_geometric_mask():
    provider = _provider()
    geometric = provider.build(
        exposure=_exposure(),
        trajectory=_trajectory(),
    )
    learned = np.zeros((120, 160), dtype=np.bool_)
    learned[5:9, 5:9] = True

    combined = provider.build(
        exposure=_exposure(),
        trajectory=_trajectory(),
        learned_mask=learned,
    )

    assert np.all(combined.mask[geometric.mask])
    assert np.all(combined.mask[5:9, 5:9])
    assert combined.masked_pixel_count >= geometric.masked_pixel_count + 16


def test_mask_accepts_v4_records_from_the_sealed_controller():
    """The SEALED production controller emits wam_openarm_exact_control_record_v4
    (verified via strings(1) on the deployed ELF, 2026-08-11) while this
    provider required the v2 schema — every live bracket was rejected and the
    tracker never started. v2/v3/v4 records must all be accepted."""
    from ausc.perception.robot.robot_state_stream import (
        EXACT_CONTROL_RECORD_SCHEMA_V4,
    )

    trajectory = _trajectory()
    upgraded = tuple(
        replace(record, schema=EXACT_CONTROL_RECORD_SCHEMA_V4)
        for record in trajectory.records
    )
    trajectory = replace(trajectory, records=upgraded)
    packet = _provider().build(exposure=_exposure(), trajectory=trajectory)
    assert packet is not None


def test_mask_accepts_uncertain_reference_ticks():
    """Ticks whose reference seqlock read collided (reference_snapshot_valid
    False, common under shared-box load) still carry exact observations; the
    mask is pure FK over observed joints and must not reject the bracket."""
    trajectory = _trajectory()
    softened = tuple(
        replace(record, reference_snapshot_valid=False)
        for record in trajectory.records
    )
    trajectory = replace(trajectory, records=softened)
    packet = _provider().build(exposure=_exposure(), trajectory=trajectory)
    assert packet is not None


def test_point_sigma_batched_is_bit_identical_to_reference():
    """Permanent oracle: the batched gufunc path must equal the verbatim
    scalar loop bit-for-bit (float.hex) across randomized geometry."""

    from ausc.perception.robot.robot_mask_provider import (
        ProductionRobotMaskProvider,
    )

    rng = np.random.default_rng(20260812)
    for case in range(1000):
        count = int(rng.integers(1, 40))
        vertices = np.column_stack((
            rng.uniform(-1.5, 1.5, count),
            rng.uniform(-1.5, 1.5, count),
            rng.uniform(0.05, 3.0, count),
        )).astype(np.float64)
        # random rotation via QR; random PSD 6x6 covariance
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        a = rng.normal(scale=rng.uniform(1e-6, 1e-2), size=(6, 6))
        covariance = a @ a.T
        ref = ProductionRobotMaskProvider._point_sigma_reference(
            vertices, q, covariance,
        )
        fast = ProductionRobotMaskProvider._point_sigma_batched(
            vertices, q, covariance,
        )
        assert float(fast).hex() == float(ref).hex(), (
            f"case {case}: {float(fast).hex()} != {float(ref).hex()}"
        )


def test_point_sigma_batched_empty_matches_reference():
    from ausc.perception.robot.robot_mask_provider import (
        ProductionRobotMaskProvider,
    )

    empty = np.zeros((0, 3), dtype=np.float64)
    eye = np.eye(3)
    cov = np.eye(6) * 1e-6
    assert ProductionRobotMaskProvider._point_sigma_reference(
        empty, eye, cov,
    ) == ProductionRobotMaskProvider._point_sigma_batched(empty, eye, cov)


def test_point_sigma_batched_error_parity_with_reference():
    from ausc.perception.robot.robot_mask_provider import (
        ProductionRobotMaskProvider,
        RobotMaskProviderError,
    )

    eye = np.eye(3)
    nan_vertices = np.array([[0.1, 0.2, float("nan")]], dtype=np.float64)
    cov = np.eye(6) * 1e-6
    outcomes = []
    for fn in (
        ProductionRobotMaskProvider._point_sigma_reference,
        ProductionRobotMaskProvider._point_sigma_batched,
    ):
        try:
            fn(nan_vertices, eye, cov)
            outcomes.append(("ok", None))
        except Exception as exc:  # noqa: BLE001 - parity check
            outcomes.append((type(exc).__name__, str(exc)[:60]))
    # Both paths must fail the same way (build() converts either identically).
    assert outcomes[0] == outcomes[1], outcomes
    assert outcomes[0][0] != "ok"
    del RobotMaskProviderError  # imported for symmetry; parity is type-based


def test_roi_dilated_paint_is_byte_identical_to_full_frame_reference():
    """The banded paint+dilate must reproduce the EXTENDED reference
    byte-for-byte: fill the hull in unclipped coordinates, dilate, then crop
    to the frame. Off-frame fill within dilation reach therefore contributes
    (2026-08-14 semantics — the crude nominal-outside full-mask fallback this
    replaces blinded perception), including hulls crossing or fully outside
    the frame and degenerate 1-2 point hulls."""

    import cv2

    from ausc.perception.robot.robot_mask_provider import (
        ProductionRobotMaskProvider,
    )

    class _Holder:
        _dilation_kernels: dict = {}
        _dilation_kernel = ProductionRobotMaskProvider._dilation_kernel
        _paint_dilated_hull = ProductionRobotMaskProvider._paint_dilated_hull
        _paint_pyramid_dilated = (
            ProductionRobotMaskProvider._paint_pyramid_dilated
        )
        _pyramid_factor = staticmethod(
            ProductionRobotMaskProvider._pyramid_factor
        )

    def reference(mask, hull, radius_px):
        height, width = mask.shape
        margin = max(int(radius_px), 0) + 1
        # Enlarge the reference canvas to hold the WHOLE hull so the
        # reference itself never clips fill: this is the true
        # infinite-plane fill-dilate-crop semantics.
        margin_x = margin + max(
            0, -int(hull[:, 0].min()), int(hull[:, 0].max()) + 1 - width,
        )
        margin_y = margin + max(
            0, -int(hull[:, 1].min()), int(hull[:, 1].max()) + 1 - height,
        )
        canvas = np.zeros(
            (height + 2 * margin_y, width + 2 * margin_x), dtype=np.uint8
        )
        shifted = hull + [margin_x, margin_y]
        if len(hull) >= 3:
            cv2.fillConvexPoly(canvas, shifted.astype(np.int32), 1)
        elif len(hull) == 2:
            cv2.line(
                canvas,
                tuple(int(v) for v in shifted[0]),
                tuple(int(v) for v in shifted[1]),
                1,
                1,
            )
        else:
            cv2.circle(
                canvas, tuple(int(v) for v in shifted[0]), 0, 1, -1,
            )
        if radius_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * radius_px + 1, 2 * radius_px + 1),
            )
            canvas = cv2.dilate(canvas, kernel, iterations=1)
        component = canvas[
            margin_y:margin_y + height, margin_x:margin_x + width,
        ]
        mask |= component.astype(np.bool_)
        return bool(np.any(component))

    rng = np.random.default_rng(20260812)
    height, width = 108, 192
    holder = _Holder()
    for case in range(400):
        count = int(rng.integers(1, 12))
        # Spread hull points inside, near-edge, and far outside the frame.
        scale = rng.choice([0.5, 1.0, 2.5])
        offset = rng.uniform(-80, 80, size=2)
        points = (
            rng.uniform(0, 1, size=(count, 2)) * [width, height] * scale
            + offset
        )
        raw_hull = cv2.convexHull(
            np.rint(points).astype(np.int32).reshape((-1, 1, 2))
        ).reshape((-1, 2))
        radius = int(rng.choice([0, 1, 3, 10, 25, 53]))
        mask_ref = np.zeros((height, width), dtype=np.bool_)
        mask_roi = np.zeros((height, width), dtype=np.bool_)
        any_ref = reference(mask_ref, raw_hull, radius)
        any_roi = holder._paint_dilated_hull(mask_roi, raw_hull, radius)
        assert np.array_equal(mask_ref, mask_roi), f"case {case} mask differs"
        assert any_ref == any_roi, f"case {case} any differs"


def test_bucketed_deferred_dilation_is_byte_identical_to_sequential():
    """Dilation distributes over union: accumulating same-radius fills and
    dilating once must equal per-component dilation byte-for-byte, across
    random multi-hull scenes with mixed radii and edge-crossing hulls."""

    import cv2

    from ausc.perception.robot.robot_mask_provider import (
        ProductionRobotMaskProvider,
    )

    class _Holder:
        _dilation_kernels: dict = {}
        _dilation_kernel = ProductionRobotMaskProvider._dilation_kernel
        _paint_dilated_hull = ProductionRobotMaskProvider._paint_dilated_hull
        _flush_deferred_dilations = (
            ProductionRobotMaskProvider._flush_deferred_dilations
        )

    rng = np.random.default_rng(20260813)
    height, width = 108, 192
    holder = _Holder()
    for case in range(120):
        ops = []
        for _ in range(int(rng.integers(2, 10))):
            count = int(rng.integers(1, 10))
            scale = rng.choice([0.4, 1.0, 2.0])
            offset = rng.uniform(-60, 60, size=2)
            points = (
                rng.uniform(0, 1, size=(count, 2)) * [width, height] * scale
                + offset
            )
            hull = cv2.convexHull(
                np.rint(points).astype(np.int32).reshape((-1, 1, 2))
            ).reshape((-1, 2))
            radius = int(rng.choice([0, 2, 2, 7, 7, 15]))
            ops.append((hull, radius))

        mask_seq = np.zeros((height, width), dtype=np.bool_)
        any_seq = [
            holder._paint_dilated_hull(mask_seq, hull, radius)
            for hull, radius in ops
        ]

        mask_bkt = np.zeros((height, width), dtype=np.bool_)
        deferred: dict = {}
        any_bkt = [
            holder._paint_dilated_hull(mask_bkt, hull, radius, deferred)
            for hull, radius in ops
        ]
        holder._flush_deferred_dilations(mask_bkt, deferred)

        assert np.array_equal(mask_seq, mask_bkt), f"case {case} mask differs"
        assert any_seq == any_bkt, f"case {case} any-flags differ"


def test_cost_hatch_threshold_is_owner_authorized_value():
    import ausc.perception.robot.robot_mask_provider as mask_module

    # 2026-08-14: the 96px full-mask hatch blinded perception (610/610 full
    # masks in the real seated pose), and the second sealed forensics showed
    # the real pose drives radii past the 256px kernel cap too, so a hatch at
    # the cap STILL produced full-frame masks. The adaptive pyramid handles
    # any radius locally; the hatch survives only as a far sentinel where a
    # full frame is honest (the dilated disc spans the frame anyway).
    assert mask_module._FULL_MASK_COST_HATCH_RADIUS_PX == 2048
    assert mask_module._PYRAMID_DILATION_MIN_RADIUS_PX == 64
    assert mask_module._PYRAMID_DILATION_FACTOR == 4
    assert mask_module._PYRAMID_COARSE_RADIUS_TARGET == 64


def test_radius_above_cost_hatch_takes_full_mask(monkeypatch):
    """Owner-authorized 2026-08-12: radii above the cost hatch previously
    dilated with huge kernels (300+ ms live, killed armed sessions); they now
    take the conservative full-mask hatch. The fixture's profile safety
    bounds cap reachable radii ~50 px, so the threshold is patched down to
    prove the branch wiring; the production value is pinned by the test
    above."""

    import ausc.perception.robot.robot_mask_provider as mask_module

    profile = _profile(
        metric_error=0.04, timing_s=0.002, projection_px=5.0, sigma=6.0,
    )
    provider = _provider(profile=profile)
    baseline = provider.build(exposure=_exposure(), trajectory=_trajectory())
    assert not baseline.mask.all()  # partial dilated mask below the hatch

    monkeypatch.setattr(mask_module, "_FULL_MASK_COST_HATCH_RADIUS_PX", 5)
    hatched = provider.build(exposure=_exposure(), trajectory=_trajectory())
    assert hatched.mask.all()
    assert hatched.masked_pixel_count > baseline.masked_pixel_count


def test_pyramid_dilation_is_conservative_superset_of_exact():
    """Large-radius dilation via the coarse pyramid must contain every pixel
    the exact dilation would set (superset), across random hulls including
    frame-edge crossings, while staying local (never the trivial full mask
    unless the exact dilation is near-full too)."""

    import cv2

    from ausc.perception.robot.robot_mask_provider import (
        ProductionRobotMaskProvider,
    )

    class _Holder:
        _dilation_kernels: dict = {}
        _dilation_kernel = ProductionRobotMaskProvider._dilation_kernel
        _paint_dilated_hull = ProductionRobotMaskProvider._paint_dilated_hull
        _paint_pyramid_dilated = (
            ProductionRobotMaskProvider._paint_pyramid_dilated
        )
        _pyramid_factor = staticmethod(
            ProductionRobotMaskProvider._pyramid_factor
        )

    def exact_reference(mask, hull, radius_px):
        component = np.zeros_like(mask, dtype=np.uint8)
        if len(hull) >= 3:
            cv2.fillConvexPoly(component, hull.astype(np.int32), 1)
        elif len(hull) == 2:
            cv2.line(
                component,
                tuple(int(v) for v in hull[0]),
                tuple(int(v) for v in hull[1]),
                1, 1,
            )
        else:
            cv2.circle(component, tuple(int(v) for v in hull[0]), 0, 1, -1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1),
        )
        component = cv2.dilate(component, kernel, iterations=1)
        mask |= component.astype(np.bool_)

    rng = np.random.default_rng(20260814)
    height, width = 270, 480
    holder = _Holder()
    for case in range(60):
        count = int(rng.integers(1, 10))
        scale = rng.choice([0.3, 0.8, 1.5])
        offset = rng.uniform(-120, 120, size=2)
        points = (
            rng.uniform(0, 1, size=(count, 2)) * [width, height] * scale
            + offset
        )
        hull = cv2.convexHull(
            np.rint(points).astype(np.int32).reshape((-1, 1, 2))
        ).reshape((-1, 2))
        radius = int(rng.choice([65, 96, 150, 255, 300, 420]))

        mask_exact = np.zeros((height, width), dtype=np.bool_)
        exact_reference(mask_exact, hull, radius)

        mask_pyr = np.zeros((height, width), dtype=np.bool_)
        holder._paint_dilated_hull(mask_pyr, hull, radius)

        missing = mask_exact & ~mask_pyr
        assert not missing.any(), (
            f"case {case}: pyramid mask misses {int(missing.sum())} exact px"
        )
        # Local, not trivially full: overshoot bounded by pyramid granularity.
        extra = int((mask_pyr & ~mask_exact).sum())
        exact_count = int(mask_exact.sum())
        if exact_count and exact_count < mask_exact.size * 0.7:
            assert extra <= mask_exact.size * 0.25, (
                f"case {case}: pyramid overshoot {extra}px is not local"
            )


def test_pyramid_factor_bounds_coarse_kernel_radius():
    """The adaptive factor must keep the coarse kernel radius <= target+1 for
    every radius up to the full-mask sentinel, and stay at the base factor for
    the tested small-radius regime."""

    from ausc.perception.robot.robot_mask_provider import (
        _FULL_MASK_COST_HATCH_RADIUS_PX,
        _PYRAMID_COARSE_RADIUS_TARGET,
        _PYRAMID_DILATION_FACTOR,
        _PYRAMID_DILATION_MIN_RADIUS_PX,
        ProductionRobotMaskProvider,
    )

    factor = ProductionRobotMaskProvider._pyramid_factor
    for pad in range(
        _PYRAMID_DILATION_MIN_RADIUS_PX + 1,
        _FULL_MASK_COST_HATCH_RADIUS_PX + 1,
    ):
        f = factor(pad)
        assert f >= _PYRAMID_DILATION_FACTOR
        coarse_radius = -(-pad // f) + 1
        assert coarse_radius <= _PYRAMID_COARSE_RADIUS_TARGET + 1, (
            f"pad={pad}: coarse radius {coarse_radius} exceeds the bound"
        )
    assert factor(65) == _PYRAMID_DILATION_FACTOR
    assert factor(256) == _PYRAMID_DILATION_FACTOR
    assert factor(2048) == 32


def test_pathological_giant_hull_cut_path_is_conservative_superset():
    """Hulls whose bbox exceeds the extended-canvas budget (near-plane
    perspective blow-ups) take the cut-and-inflate path: the painted frame
    window must contain every pixel of the true infinite-plane reference
    (loss would blind perception; bounded overshoot is the accepted cost)."""

    import cv2

    from ausc.perception.robot.robot_mask_provider import (
        _EXTENDED_CANVAS_AREA_FACTOR,
        ProductionRobotMaskProvider,
    )

    class _Holder:
        _dilation_kernels: dict = {}
        _dilation_kernel = ProductionRobotMaskProvider._dilation_kernel
        _paint_dilated_hull = ProductionRobotMaskProvider._paint_dilated_hull
        _paint_pyramid_dilated = (
            ProductionRobotMaskProvider._paint_pyramid_dilated
        )
        _pyramid_factor = staticmethod(
            ProductionRobotMaskProvider._pyramid_factor
        )

    def reference(mask, hull, radius_px):
        height, width = mask.shape
        margin = max(int(radius_px), 0) + 1
        margin_x = margin + max(
            0, -int(hull[:, 0].min()), int(hull[:, 0].max()) + 1 - width,
        )
        margin_y = margin + max(
            0, -int(hull[:, 1].min()), int(hull[:, 1].max()) + 1 - height,
        )
        canvas = np.zeros(
            (height + 2 * margin_y, width + 2 * margin_x), dtype=np.uint8
        )
        cv2.fillConvexPoly(
            canvas, (hull + [margin_x, margin_y]).astype(np.int32), 1,
        )
        if radius_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * radius_px + 1, 2 * radius_px + 1),
            )
            canvas = cv2.dilate(canvas, kernel, iterations=1)
        component = canvas[
            margin_y:margin_y + height, margin_x:margin_x + width,
        ]
        mask |= component.astype(np.bool_)

    rng = np.random.default_rng(20260815)
    height, width = 108, 192
    budget = _EXTENDED_CANVAS_AREA_FACTOR * height * width
    holder = _Holder()
    tested = 0
    for case in range(40):
        # Giant triangle: two far vertices force a bbox beyond the budget,
        # one near vertex keeps fill within reach of the frame.
        far = rng.uniform(2000, 3000, size=(2, 2)) * rng.choice(
            [-1.0, 1.0], size=(2, 2)
        )
        near = rng.uniform(-40, 40, size=(1, 2)) + [width / 2, height / 2]
        pts = np.vstack([far, near])
        hull = cv2.convexHull(
            np.rint(pts).astype(np.int32).reshape((-1, 1, 2))
        ).reshape((-1, 2))
        bbox_area = (
            (int(hull[:, 0].max()) + 1 - int(hull[:, 0].min()))
            * (int(hull[:, 1].max()) + 1 - int(hull[:, 1].min()))
        )
        if bbox_area <= budget:
            continue
        tested += 1
        radius = int(rng.choice([5, 30, 90]))
        mask_ref = np.zeros((height, width), dtype=np.bool_)
        mask_cut = np.zeros((height, width), dtype=np.bool_)
        reference(mask_ref, hull, radius)
        holder._paint_dilated_hull(mask_cut, hull, radius)
        missing = mask_ref & ~mask_cut
        assert not missing.any(), (
            f"case {case}: cut path misses {int(missing.sum())} px"
        )
    assert tested >= 20, f"only {tested} giant-bbox cases exercised"
