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


def test_nominal_geometry_behind_camera_with_uncertainty_crossing_near_plane_fails():
    geometry = _geometry()
    tool = _tool()
    handeye = _handeye(z=-1.05)
    projection = _projection()
    profile = _profile(metric_error=0.1)
    provider = _provider(
        geometry=geometry,
        tool=tool,
        handeye=handeye,
        projection=projection,
        profile=profile,
    )

    with pytest.raises(RobotMaskProviderError, match="near plane"):
        provider.build(
            exposure=_exposure(),
            trajectory=_trajectory(),
        )


def test_raw_uncertainty_intersection_uses_full_mask_without_giant_kernel():
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

    assert packet.masked_pixel_count == projection.width * projection.height
    assert packet.mask.all()
    assert packet.reason == "interval_swept_openarm_robot_tool_mask"


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
