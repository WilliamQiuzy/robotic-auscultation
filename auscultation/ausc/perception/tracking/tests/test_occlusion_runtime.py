from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import numpy as np
import pytest

from ausc.perception.tracking.occlusion_runtime import (
    ActivatedCameraProjection,
    ActivatedOcclusionIdentity,
    ActivatedOcclusionRuntimeProfile,
    OcclusionFrameObservation,
    OcclusionPointObservation,
    OcclusionRuntime,
    OcclusionRuntimeConfig,
    PatientPosePacket,
    RobotMaskPacket,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_0 = "0" * 64


def _mask_sha(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    ).hexdigest()


def _mask_packet(
    *,
    frame_index: int = 7,
    camera_host_t_s: float = 10.0,
    shape: tuple[int, int] = (16, 20),
    masked_pixel: tuple[int, int] | None = None,
) -> RobotMaskPacket:
    mask = np.zeros(shape, dtype=np.bool_)
    if masked_pixel is not None:
        mask[masked_pixel] = True
    return RobotMaskPacket(
        frame_index=frame_index,
        camera_host_t_s=camera_host_t_s,
        camera_host_monotonic_ns=int(round(camera_host_t_s * 1_000_000_000)),
        camera_device_t_ms=1234.5,
        camera_id="CL8326300A0",
        camera_profile_sha256=SHA_A,
        exposure_lower_host_t_s=camera_host_t_s - 0.025,
        exposure_nominal_host_t_s=camera_host_t_s - 0.020,
        exposure_upper_host_t_s=camera_host_t_s - 0.015,
        exposure_lower_monotonic_ns=int(
            round((camera_host_t_s - 0.025) * 1_000_000_000)
        ),
        exposure_nominal_monotonic_ns=int(
            round((camera_host_t_s - 0.020) * 1_000_000_000)
        ),
        exposure_upper_monotonic_ns=int(
            round((camera_host_t_s - 0.015) * 1_000_000_000)
        ),
        mask=mask,
        mask_sha256=_mask_sha(mask),
        masked_pixel_count=int(np.count_nonzero(mask)),
        joint_sequence_first=100,
        joint_sequence_last=104,
        joint_time_first_s=camera_host_t_s - 0.030,
        joint_time_last_s=camera_host_t_s - 0.010,
        joint_time_first_monotonic_ns=int(
            round((camera_host_t_s - 0.030) * 1_000_000_000)
        ),
        joint_time_last_monotonic_ns=int(
            round((camera_host_t_s - 0.010) * 1_000_000_000)
        ),
        joint_max_gap_s=0.005,
        joint_max_gap_ns=5_000_000,
        geometry_release_id="robot_geometry-" + "1" * 20,
        geometry_sha256=SHA_A,
        tool_release_id="tool_geometry-" + "2" * 20,
        tool_sha256=SHA_B,
        handeye_release_id="handeye-" + SHA_C[:20],
        handeye_sha256=SHA_C,
        handeye_capability_schema="wam_openarm_handeye_v1",
        handeye_calibration_tier="QUALITY_GATE_COMPLETE",
        handeye_policy_sha256=SHA_E,
        camera_latency_sha256=SHA_D,
        camera_projection_sha256=_projection().camera_projection_sha256,
        occlusion_profile_sha256=SHA_F,
        alignment_sha256=SHA_0,
        joint_publisher_identity="pid=42;start=123;build=" + SHA_A,
        status="OK",
        reason="OK",
    )


def _pose_packet(
    *,
    frame_index: int = 7,
    camera_host_t_s: float = 10.0,
    translation=(0.0, 0.0, 0.0),
    covariance_scale: float = 1.0e-8,
    episode_id: str = "patient-0001",
    valid: bool = True,
) -> PatientPosePacket:
    return PatientPosePacket(
        frame_index=frame_index,
        camera_host_t_s=camera_host_t_s,
        camera_id="CL8326300A0",
        camera_profile_sha256=SHA_A,
        alignment_sha256=SHA_0,
        patient_pose_release_id="patient_pose-" + "4" * 20,
        patient_pose_sha256=SHA_D,
        depth_noise_model_sha256=SHA_E,
        back_mask_release_id="back_mask-" + "5" * 20,
        back_mask_sha256=SHA_F,
        rotation_camera_from_patient=np.eye(3),
        translation_camera_from_patient_m=np.asarray(translation, dtype=np.float64),
        covariance_se3=np.eye(6, dtype=np.float64) * covariance_scale,
        support_mask=np.ones(10, dtype=np.bool_),
        geometry_accepted=valid,
        fit_rmse_m=0.001 if valid else float("inf"),
        episode_id=episode_id,
        source="rgbd_feature_ransac_plus_icp",
        scale_audit=1.0,
        scale_tolerance=0.01,
        valid=valid,
        reason="OK" if valid else "POSE_UNAVAILABLE",
        covariance_tangent_order="rotation_patient_rad_then_translation_camera_m",
        covariance_perturbation="right",
        covariance_transform="patient_to_camera",
    )


def _point(
    *,
    name: str = "R1",
    xyz=(0.10, 0.20, 0.90),
    patient_xyz=(0.10, 0.20, 0.90),
    pixel=(4.0, 3.0),
    measured: bool = True,
    direct_geometry_accepted: bool = True,
    robot_occluded: bool = False,
    uncertainty_bound_m: float = 0.003,
) -> OcclusionPointObservation:
    return OcclusionPointObservation(
        name=name,
        xyz_camera_m=np.asarray(xyz, dtype=np.float64),
        xyz_patient_m=np.asarray(patient_xyz, dtype=np.float64),
        pixel=np.asarray(pixel, dtype=np.float64),
        measured=measured,
        direct_geometry_accepted=direct_geometry_accepted,
        robot_occluded=robot_occluded,
        uncertainty_bound_m=uncertainty_bound_m,
    )


def _frame(
    point: OcclusionPointObservation,
    *,
    frame_index: int = 7,
    timestamp_s: float = 10.0,
) -> OcclusionFrameObservation:
    return OcclusionFrameObservation(
        frame_index=frame_index,
        camera_host_t_s=timestamp_s,
        camera_id="CL8326300A0",
        camera_profile_sha256=SHA_A,
        aligned_depth_shape=(16, 20),
        back_present=True,
        initialized=True,
        points=(point,),
    )


def _config(**changes) -> OcclusionRuntimeConfig:
    values = {
        "max_freeze_age_s": 0.40,
        "max_uncertainty_bound_m": 0.010,
        "patient_motion_bound_m_s": 0.013,
        "handeye_bound_m": 0.0001,
        "timing_sweep_bound_m": 0.0001,
        "projection_bound_m": 0.0001,
        "tool_geometry_bound_m": 0.0001,
        "reacquire_frames": 5,
        "reacquire_max_displacement_m": 0.05,
    }
    values.update(changes)
    return OcclusionRuntimeConfig(**values)


@pytest.mark.parametrize(
    "field",
    (
        "handeye_bound_m",
        "timing_sweep_bound_m",
        "projection_bound_m",
        "tool_geometry_bound_m",
    ),
)
def test_runtime_config_rejects_zero_artifact_derived_fixed_bound(
    field: str,
) -> None:
    values = {
        "max_freeze_age_s": 0.40,
        "max_uncertainty_bound_m": 0.010,
        "patient_motion_bound_m_s": 0.013,
        "handeye_bound_m": 0.003,
        "timing_sweep_bound_m": 0.002,
        "projection_bound_m": 0.001,
        "tool_geometry_bound_m": 0.001,
        "reacquire_frames": 5,
        "reacquire_max_displacement_m": 0.05,
    }
    values[field] = 0.0

    with pytest.raises(ValueError, match="fixed bounds"):
        OcclusionRuntimeConfig(**values)


def test_runtime_config_rejects_motion_bound_below_commissioned_floor() -> None:
    with pytest.raises(ValueError, match="patient motion"):
        _config(patient_motion_bound_m_s=math.nextafter(0.013, 0.0))


def _projection(**changes) -> ActivatedCameraProjection:
    values = {
        "camera_id": "CL8326300A0",
        "camera_profile_sha256": SHA_A,
        "width": 20,
        "height": 16,
        "fx": 9.0,
        "fy": 9.0,
        "cx": 3.0,
        "cy": 1.0,
        "min_projection_depth_m": 0.05,
    }
    values.update(changes)
    return ActivatedCameraProjection.create(**values)


def _runtime_profile(
    config: OcclusionRuntimeConfig | None = None,
    **source_changes,
) -> ActivatedOcclusionRuntimeProfile:
    values = {
        "patient_motion_artifact_sha256": SHA_0,
        "handeye_artifact_sha256": SHA_C,
        "camera_latency_artifact_sha256": SHA_D,
        "camera_projection_artifact_sha256": (
            _projection().camera_projection_sha256
        ),
        "tool_geometry_artifact_sha256": SHA_B,
    }
    values.update(source_changes)
    return ActivatedOcclusionRuntimeProfile.create(
        config=config or _config(),
        **values,
    )


def test_runtime_profile_hashes_every_numeric_runtime_bound() -> None:
    profile = _runtime_profile()
    changes = {
        "max_freeze_age_s": 0.39,
        "max_uncertainty_bound_m": 0.009,
        "patient_motion_bound_m_s": 0.014,
        "handeye_bound_m": 0.0002,
        "timing_sweep_bound_m": 0.0002,
        "projection_bound_m": 0.0002,
        "tool_geometry_bound_m": 0.0002,
        "reacquire_frames": 6,
        "reacquire_max_displacement_m": 0.049,
        "reacquire_max_gap_s": 0.09,
    }

    for field, value in changes.items():
        with pytest.raises(ValueError, match="semantic content"):
            replace(profile, **{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "patient_motion_artifact_sha256",
        "handeye_artifact_sha256",
        "camera_latency_artifact_sha256",
        "camera_projection_artifact_sha256",
        "tool_geometry_artifact_sha256",
    ),
)
def test_runtime_profile_hashes_every_artifact_derived_component(
    field: str,
) -> None:
    profile = _runtime_profile()

    with pytest.raises(ValueError, match="semantic content"):
        replace(profile, **{field: SHA_E})


def _identity(**changes) -> ActivatedOcclusionIdentity:
    values = {
        "camera_id": "CL8326300A0",
        "camera_profile_sha256": SHA_A,
        "geometry_release_id": "robot_geometry-" + "1" * 20,
        "geometry_sha256": SHA_A,
        "tool_release_id": "tool_geometry-" + "2" * 20,
        "tool_sha256": SHA_B,
        "handeye_release_id": "handeye-" + SHA_C[:20],
        "handeye_sha256": SHA_C,
        "handeye_capability_schema": "wam_openarm_handeye_v1",
        "handeye_calibration_tier": "QUALITY_GATE_COMPLETE",
        "handeye_policy_sha256": SHA_E,
        "camera_latency_sha256": SHA_D,
        "camera_projection_sha256": _projection().camera_projection_sha256,
        "occlusion_profile_sha256": SHA_F,
        "occlusion_runtime_profile_sha256": (
            _runtime_profile().occlusion_runtime_profile_sha256
        ),
        "alignment_sha256": SHA_0,
        "joint_mapper_sha256": SHA_A,
        "pair_contract_sha256": SHA_B,
        "controller_binary_sha256": SHA_C,
        "patient_pose_release_id": "patient_pose-" + "4" * 20,
        "patient_pose_sha256": SHA_D,
        "depth_noise_model_sha256": SHA_E,
        "back_mask_release_id": "back_mask-" + "5" * 20,
        "back_mask_sha256": SHA_F,
        "max_camera_to_exposure_age_ns": 100_000_000,
        "max_exposure_interval_ns": 40_000_000,
    }
    values.update(changes)
    return ActivatedOcclusionIdentity(**values)


def _identity_with_runtime_profile(
    profile: ActivatedOcclusionRuntimeProfile,
    **changes,
) -> ActivatedOcclusionIdentity:
    changes.setdefault(
        "occlusion_runtime_profile_sha256",
        profile.occlusion_runtime_profile_sha256,
    )
    return _identity(**changes)


def test_v2_handeye_release_cannot_be_relabeled_as_legacy_policy() -> None:
    release_id = "handeye_activation-" + SHA_C[:20]
    identity = _identity(
        handeye_release_id=release_id,
        handeye_capability_schema=(
            "wam_provisional_robot_mask_handeye_capability_v1"
        ),
        handeye_calibration_tier="USER_ACCEPTED_BELOW_GATE_5MM",
    )
    provisional = replace(
        _mask_packet(),
        handeye_release_id=release_id,
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
    validation = {
        "frame_index": provisional.frame_index,
        "camera_host_t_s": provisional.camera_host_t_s,
        "camera_id": provisional.camera_id,
        "camera_profile_sha256": provisional.camera_profile_sha256,
        "depth_shape": provisional.mask.shape,
        "activated_identity": identity,
    }
    provisional.validate_for_frame(**validation)

    relabeled = replace(
        provisional,
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
    with pytest.raises(ValueError, match="hand-eye.*policy"):
        relabeled.validate_for_frame(**validation)


@pytest.mark.parametrize(
    "changes",
    (
        {
            "handeye_release_id": "handeye_activation-" + SHA_C[:20],
        },
        {
            "handeye_capability_schema": (
                "wam_provisional_robot_mask_handeye_capability_v1"
            ),
        },
        {
            "handeye_calibration_tier": "USER_ACCEPTED_BELOW_GATE_5MM",
        },
        {
            "handeye_policy_sha256": "not-a-sha256",
        },
    ),
)
def test_activated_identity_rejects_incoherent_handeye_policy_binding(
    changes,
) -> None:
    with pytest.raises(ValueError, match="hand-eye"):
        _identity(**changes)


def test_packet_policy_hash_must_match_activated_identity() -> None:
    packet = replace(_mask_packet(), handeye_policy_sha256=SHA_F)

    with pytest.raises(ValueError, match="activated occlusion identity"):
        packet.validate_for_frame(
            frame_index=packet.frame_index,
            camera_host_t_s=packet.camera_host_t_s,
            camera_id=packet.camera_id,
            camera_profile_sha256=packet.camera_profile_sha256,
            depth_shape=packet.mask.shape,
            activated_identity=_identity(),
        )


def test_runtime_requires_independent_identity_pinned_runtime_profile() -> None:
    config = _config()
    profile = _runtime_profile(config)
    identity = _identity_with_runtime_profile(profile)

    with pytest.raises(TypeError):
        OcclusionRuntime(
            ("R1",),
            config,
            activated_identity=identity,
            camera_projection=_projection(),
        )

    runtime = OcclusionRuntime(
        ("R1",),
        config,
        activated_identity=identity,
        camera_projection=_projection(),
        runtime_profile=profile,
    )

    assert runtime.runtime_profile is profile
    assert (
        identity.occlusion_profile_sha256
        != identity.occlusion_runtime_profile_sha256
    )


def test_runtime_rejects_runtime_profile_not_pinned_by_identity() -> None:
    config = _config()
    profile = _runtime_profile(config)
    identity = _identity_with_runtime_profile(
        profile,
        occlusion_runtime_profile_sha256=SHA_E,
    )

    with pytest.raises(ValueError, match="runtime profile.*identity"):
        OcclusionRuntime(
            ("R1",),
            config,
            activated_identity=identity,
            camera_projection=_projection(),
            runtime_profile=profile,
        )


def test_runtime_rejects_runtime_profile_numeric_config_mismatch() -> None:
    profile = _runtime_profile(_config())
    identity = _identity_with_runtime_profile(profile)

    with pytest.raises(ValueError, match="runtime config"):
        OcclusionRuntime(
            ("R1",),
            _config(max_freeze_age_s=0.39),
            activated_identity=identity,
            camera_projection=_projection(),
            runtime_profile=profile,
        )


@pytest.mark.parametrize(
    "source_change",
    (
        {"handeye_artifact_sha256": SHA_E},
        {"camera_latency_artifact_sha256": SHA_E},
        {"camera_projection_artifact_sha256": SHA_E},
        {"tool_geometry_artifact_sha256": SHA_E},
    ),
)
def test_runtime_rejects_profile_component_not_matching_activated_artifact(
    source_change: dict,
) -> None:
    config = _config()
    profile = _runtime_profile(config, **source_change)
    identity = _identity_with_runtime_profile(profile)

    with pytest.raises(ValueError, match="artifact sources"):
        OcclusionRuntime(
            ("R1",),
            config,
            activated_identity=identity,
            camera_projection=_projection(),
            runtime_profile=profile,
        )


def _runtime(
    config: OcclusionRuntimeConfig | None = None,
    *,
    identity: ActivatedOcclusionIdentity | None = None,
    projection: ActivatedCameraProjection | None = None,
    runtime_profile: ActivatedOcclusionRuntimeProfile | None = None,
) -> OcclusionRuntime:
    resolved_config = config or _config()
    runtime_profile = runtime_profile or _runtime_profile(resolved_config)
    return OcclusionRuntime(
        ("R1",),
        resolved_config,
        activated_identity=(
            identity
            or _identity(
                occlusion_runtime_profile_sha256=(
                    runtime_profile.occlusion_runtime_profile_sha256
                )
            )
        ),
        camera_projection=projection or _projection(),
        runtime_profile=runtime_profile,
    )


def _validate_pose(packet: PatientPosePacket) -> bool:
    return packet.validate_for_frame(
        frame_index=packet.frame_index,
        camera_host_t_s=packet.camera_host_t_s,
        camera_id="CL8326300A0",
        camera_profile_sha256=SHA_A,
        activated_identity=_identity(),
    )


def test_robot_mask_packet_binds_exact_frame_profile_interval_and_bytes() -> None:
    packet = _mask_packet()

    packet.validate_for_frame(
        frame_index=7,
        camera_host_t_s=10.0,
        camera_id="CL8326300A0",
        camera_profile_sha256=SHA_A,
        depth_shape=(16, 20),
        activated_identity=_identity(),
    )

    assert packet.mask.dtype == np.bool_
    assert packet.mask.flags.writeable is False
    assert packet.exposure_lower_host_t_s < packet.exposure_nominal_host_t_s
    assert packet.exposure_nominal_host_t_s < packet.exposure_upper_host_t_s


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"frame_index": 8}, "frame index"),
        ({"camera_host_t_s": 10.1}, "host timestamp"),
        ({"mask_sha256": "0" * 64}, "mask SHA-256"),
        ({"masked_pixel_count": 9}, "masked pixel count"),
        ({"camera_projection_sha256": ""}, "projection"),
        ({"status": "STALE", "reason": "JOINT_GAP"}, "status"),
    ),
)
def test_robot_mask_packet_rejects_mismatched_or_uncommissioned_evidence(
    change: dict, message: str,
) -> None:
    packet = replace(_mask_packet(), **change)

    with pytest.raises(ValueError, match=message):
        packet.validate_for_frame(
            frame_index=7,
            camera_host_t_s=10.0,
            camera_id="CL8326300A0",
            camera_profile_sha256=SHA_A,
            depth_shape=(16, 20),
            activated_identity=_identity(),
        )


def test_robot_mask_packet_rejects_wrong_shape_and_unbracketed_exposure() -> None:
    packet = _mask_packet()
    with pytest.raises(ValueError, match="shape"):
        packet.validate_for_frame(
            frame_index=7,
            camera_host_t_s=10.0,
            camera_id="CL8326300A0",
            camera_profile_sha256=SHA_A,
            depth_shape=(20, 16),
            activated_identity=_identity(),
        )

    unbracketed = replace(packet, joint_time_first_s=9.99)
    with pytest.raises(ValueError, match="bracket"):
        unbracketed.validate_for_frame(
            frame_index=7,
            camera_host_t_s=10.0,
            camera_id="CL8326300A0",
            camera_profile_sha256=SHA_A,
            depth_shape=(16, 20),
            activated_identity=_identity(),
        )


def test_patient_pose_packet_rejects_nonrigid_or_uncertain_scale() -> None:
    nonrigid = _pose_packet()
    rotation = np.eye(3)
    rotation[0, 0] = 1.1
    nonrigid = replace(nonrigid, rotation_camera_from_patient=rotation)
    with pytest.raises(ValueError, match="rotation"):
        _validate_pose(nonrigid)

    wrong_scale = replace(_pose_packet(), scale_audit=1.02)
    with pytest.raises(ValueError, match="scale"):
        _validate_pose(wrong_scale)


def test_valid_patient_pose_cannot_claim_zero_covariance() -> None:
    zero_covariance = replace(
        _pose_packet(),
        covariance_se3=np.zeros((6, 6), dtype=np.float64),
    )

    with pytest.raises(ValueError, match="covariance"):
        _validate_pose(zero_covariance)


def test_patient_pose_packet_rejects_untrusted_source_quality_and_support() -> None:
    untrusted = replace(
        _pose_packet(),
        source="untrusted",
        support_mask=np.zeros(10, dtype=np.bool_),
        fit_rmse_m=999.0,
    )

    with pytest.raises(ValueError, match="metric evidence"):
        _validate_pose(untrusted)


@pytest.mark.parametrize(
    "changes",
    (
        {"camera_id": "other-camera"},
        {"camera_profile_sha256": SHA_B},
        {"alignment_sha256": SHA_A},
        {"patient_pose_release_id": "patient_pose-" + "9" * 20},
        {"patient_pose_sha256": SHA_A},
        {"depth_noise_model_sha256": SHA_A},
        {"back_mask_release_id": "back_mask-" + "9" * 20},
        {"back_mask_sha256": SHA_A},
    ),
)
def test_patient_pose_must_match_every_activated_provenance_pin(
    changes: dict,
) -> None:
    result = _runtime().advance(
        _frame(_point()),
        robot_mask_packet=_mask_packet(),
        patient_pose_packet=replace(_pose_packet(), **changes),
    )

    assert result.reason == "INVALID_OCCLUSION_EVIDENCE"
    assert result.points[0].valid is False


def test_direct_measurement_updates_patient_anchor() -> None:
    runtime = _runtime()

    result = runtime.advance(
        _frame(_point()),
        robot_mask_packet=_mask_packet(),
        patient_pose_packet=_pose_packet(),
    )

    assert result.points[0].occlusion_state == "DIRECT"
    assert result.points[0].measured is True
    assert result.points[0].valid is True
    assert result.points[0].uncertainty_bound_m == pytest.approx(0.003)


def test_masked_target_follows_independent_patient_pose() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point()),
        robot_mask_packet=_mask_packet(),
        patient_pose_packet=_pose_packet(),
    )
    masked = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )

    result = runtime.advance(
        _frame(masked, frame_index=8, timestamp_s=10.1),
        robot_mask_packet=_mask_packet(
            frame_index=8, camera_host_t_s=10.1, masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8, camera_host_t_s=10.1, translation=(0.02, 0.0, 0.0),
        ),
    )

    point = result.points[0]
    assert point.occlusion_state == "PATIENT_FRAME_LOCKED"
    assert point.valid is True
    assert point.measured is False
    np.testing.assert_allclose(point.xyz_camera_m, [0.12, 0.20, 0.90])
    np.testing.assert_allclose(point.pixel, [4.2, 3.0])
    assert point.uncertainty_bound_m < 0.010


def test_locked_pixel_ignores_jittering_learned_pixel_and_reprojects_3d() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(pixel=(4.0, 3.0))),
        robot_mask_packet=_mask_packet(),
        patient_pose_packet=_pose_packet(),
    )
    jittering_learned_point = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
        pixel=(15.0, 12.0),
    )
    swept_mask = np.zeros((16, 20), dtype=np.bool_)
    swept_mask[12, 15] = True
    swept_mask[3, 4] = True
    mask_packet = replace(
        _mask_packet(
            frame_index=8,
            camera_host_t_s=10.1,
            masked_pixel=None,
        ),
        mask=swept_mask,
        mask_sha256=_mask_sha(swept_mask),
        masked_pixel_count=2,
    )

    result = runtime.advance(
        _frame(jittering_learned_point, frame_index=8, timestamp_s=10.1),
        robot_mask_packet=mask_packet,
        patient_pose_packet=_pose_packet(
            frame_index=8,
            camera_host_t_s=10.1,
            translation=(0.02, 0.0, 0.0),
        ),
    )

    point = result.points[0]
    assert point.occlusion_state == "PATIENT_FRAME_LOCKED"
    np.testing.assert_allclose(point.pixel, [4.2, 3.0])
    assert not np.array_equal(point.pixel, jittering_learned_point.pixel)


@pytest.mark.parametrize(
    "translation",
    (
        (2.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    ),
)
def test_invalid_or_out_of_fov_projection_is_not_published_as_locked(
    translation,
) -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point()),
        robot_mask_packet=_mask_packet(),
        patient_pose_packet=_pose_packet(),
    )
    masked = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )

    result = runtime.advance(
        _frame(masked, frame_index=8, timestamp_s=10.1),
        robot_mask_packet=_mask_packet(
            frame_index=8,
            camera_host_t_s=10.1,
            masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8,
            camera_host_t_s=10.1,
            translation=translation,
        ),
    )

    # A current, valid patient pose is stronger evidence than stale history.
    # If it says the locked 3-D target is behind the camera or outside the
    # image, retaining the old on-screen pixel would be actively misleading.
    assert result.points[0].occlusion_state == "LOST"
    assert result.points[0].pixel is None


def test_locked_projection_must_still_be_inside_the_robot_mask() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(pixel=(4.0, 3.0))),
        robot_mask_packet=_mask_packet(),
        patient_pose_packet=_pose_packet(),
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
        pixel=(4.0, 3.0),
    )

    result = runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.1),
        robot_mask_packet=_mask_packet(
            frame_index=8,
            camera_host_t_s=10.1,
            masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8,
            camera_host_t_s=10.1,
            translation=(0.11, 0.0, 0.0),
        ),
    )

    assert result.points[0].occlusion_state == "LOST"
    assert result.points[0].reason == "PROJECTED_TARGET_NOT_ROBOT_OCCLUDED"
    assert result.points[0].pixel is None


def test_locked_projection_requires_full_uncertainty_region_inside_image() -> None:
    projection = _projection(fx=90.0)
    runtime_profile = _runtime_profile(
        camera_projection_artifact_sha256=(
            projection.camera_projection_sha256
        )
    )
    identity = _identity(
        camera_projection_sha256=projection.camera_projection_sha256,
        occlusion_runtime_profile_sha256=(
            runtime_profile.occlusion_runtime_profile_sha256
        ),
    )
    runtime = _runtime(
        identity=identity,
        projection=projection,
        runtime_profile=runtime_profile,
    )
    edge_point = _point(
        xyz=(0.164, 0.20, 0.90),
        patient_xyz=(0.164, 0.20, 0.90),
        pixel=(19.4, 3.0),
        uncertainty_bound_m=0.005,
    )
    runtime.advance(
        _frame(edge_point),
        robot_mask_packet=replace(
            _mask_packet(masked_pixel=None),
            camera_projection_sha256=projection.camera_projection_sha256,
        ),
        patient_pose_packet=_pose_packet(),
    )
    hidden = _point(
        xyz=(0.164, 0.20, 0.90),
        patient_xyz=(0.164, 0.20, 0.90),
        pixel=(19.0, 3.0),
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
        uncertainty_bound_m=0.005,
    )

    result = runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.1),
        robot_mask_packet=replace(
            _mask_packet(
                frame_index=8,
                camera_host_t_s=10.1,
                masked_pixel=(3, 19),
            ),
            camera_projection_sha256=projection.camera_projection_sha256,
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8,
            camera_host_t_s=10.1,
        ),
    )

    assert result.points[0].occlusion_state == "LOST"
    assert result.points[0].reason == "LOCKED_UNCERTAINTY_OUTSIDE_CAMERA"
    assert result.points[0].pixel is None


def test_projection_contract_hash_is_semantic_and_must_match_activation() -> None:
    projection = _projection()
    with pytest.raises(ValueError, match="SHA-256"):
        replace(projection, fx=projection.fx + 1.0)

    wrong_identity = _identity(camera_projection_sha256=SHA_F)
    with pytest.raises(ValueError, match="activated camera projection"):
        _runtime(identity=wrong_identity, projection=projection)


def test_model_jump_outside_mask_cannot_bypass_occlusion_reacquisition() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(pixel=(4.0, 3.0)), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )

    jumped = runtime.advance(
        _frame(
            _point(
                xyz=(0.13, 0.20, 0.90),
                patient_xyz=(0.13, 0.20, 0.90),
                pixel=(8.0, 3.0),
                robot_occluded=False,
            ),
            frame_index=8,
            timestamp_s=10.05,
        ),
        robot_mask_packet=_mask_packet(
            frame_index=8,
            camera_host_t_s=10.05,
            masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8,
            camera_host_t_s=10.05,
        ),
    )

    assert jumped.points[0].occlusion_state == "REACQUIRING"
    assert jumped.points[0].valid is False
    assert jumped.points[0].xyz_camera_m is None


def test_direct_return_after_patient_frame_lock_requires_five_frames() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )
    locked = runtime.advance(
        _frame(
            _point(
                measured=False,
                direct_geometry_accepted=False,
                robot_occluded=True,
            ),
            frame_index=8,
            timestamp_s=10.05,
        ),
        robot_mask_packet=_mask_packet(
            frame_index=8,
            camera_host_t_s=10.05,
            masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8,
            camera_host_t_s=10.05,
        ),
    )
    assert locked.points[0].occlusion_state == "PATIENT_FRAME_LOCKED"

    states = []
    for offset in range(5):
        frame_index = 9 + offset
        timestamp = 10.10 + 0.04 * offset
        result = runtime.advance(
            _frame(
                _point(xyz=(0.101, 0.20, 0.90)),
                frame_index=frame_index,
                timestamp_s=timestamp,
            ),
            robot_mask_packet=_mask_packet(
                frame_index=frame_index,
                camera_host_t_s=timestamp,
            ),
            patient_pose_packet=_pose_packet(
                frame_index=frame_index,
                camera_host_t_s=timestamp,
            ),
        )
        states.append(result.points[0].occlusion_state)

    assert states[:4] == ["REACQUIRING"] * 4
    assert states[4] == "DIRECT"


def test_anchor_pose_uncertainty_is_counted_before_patient_frame_lock() -> None:
    runtime = _runtime()
    covariance = np.zeros((6, 6), dtype=np.float64)
    covariance[3:, 3:] = np.eye(3) * 7.45e-6
    anchor_pose = replace(_pose_packet(camera_host_t_s=10.0), covariance_se3=covariance)
    runtime.advance(
        _frame(_point(uncertainty_bound_m=0.003), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=anchor_pose,
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )

    result = runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.05),
        robot_mask_packet=_mask_packet(
            frame_index=8,
            camera_host_t_s=10.05,
            masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8,
            camera_host_t_s=10.05,
            covariance_scale=1.0e-12,
        ),
    )

    assert result.points[0].occlusion_state != "PATIENT_FRAME_LOCKED"
    assert result.points[0].uncertainty_bound_m >= 0.003


def test_pose_loss_freezes_at_exact_age_and_uncertainty_bound_then_loses() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(uncertainty_bound_m=0.0048), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )
    invalid_pose_at_boundary = _pose_packet(
        frame_index=8,
        camera_host_t_s=10.4,
        valid=False,
    )

    boundary = runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.4),
        robot_mask_packet=_mask_packet(
            frame_index=8, camera_host_t_s=10.4, masked_pixel=(3, 4),
        ),
        patient_pose_packet=invalid_pose_at_boundary,
    )

    assert boundary.points[0].occlusion_state == "BOUNDED_FREEZE"
    assert boundary.points[0].uncertainty_bound_m == pytest.approx(0.010)
    after = math.nextafter(10.4, math.inf)
    lost = runtime.advance(
        _frame(hidden, frame_index=9, timestamp_s=after),
        robot_mask_packet=_mask_packet(
            frame_index=9, camera_host_t_s=after, masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=9, camera_host_t_s=after, valid=False,
        ),
    )

    assert lost.points[0].occlusion_state == "LOST"
    assert lost.points[0].valid is False
    assert lost.points[0].xyz_camera_m is None
    assert math.isinf(lost.points[0].uncertainty_bound_m)


def test_nonmonotonic_time_fails_closed_without_destroying_runtime() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )

    failed = runtime.advance(
        _frame(_point(), frame_index=8, timestamp_s=10.0),
        robot_mask_packet=_mask_packet(frame_index=8, camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(frame_index=8, camera_host_t_s=10.0),
    )

    assert failed.points[0].occlusion_state == "LOST"
    assert failed.reason == "NON_MONOTONIC_TIME"


def test_direct_reacquisition_after_lost_stays_hidden_until_fifth_frame() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )
    lost = runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.5),
        robot_mask_packet=_mask_packet(
            frame_index=8, camera_host_t_s=10.5, masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8, camera_host_t_s=10.5, valid=False,
        ),
    )
    assert lost.points[0].occlusion_state == "LOST"

    states = []
    for offset in range(5):
        timestamp = 10.6 + 0.05 * offset
        frame_index = 9 + offset
        result = runtime.advance(
            _frame(_point(xyz=(0.102, 0.20, 0.90)), frame_index=frame_index,
                   timestamp_s=timestamp),
            robot_mask_packet=_mask_packet(
                frame_index=frame_index, camera_host_t_s=timestamp,
            ),
            patient_pose_packet=_pose_packet(
                frame_index=frame_index, camera_host_t_s=timestamp,
            ),
        )
        states.append((result.points[0].occlusion_state, result.points[0].valid))

    assert states[:4] == [("REACQUIRING", False)] * 4
    assert states[4] == ("DIRECT", True)


def test_reacquisition_jump_clears_streak_and_never_publishes_jump() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )
    runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.5),
        robot_mask_packet=_mask_packet(
            frame_index=8, camera_host_t_s=10.5, masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8, camera_host_t_s=10.5, valid=False,
        ),
    )
    first = runtime.advance(
        _frame(_point(xyz=(0.102, 0.20, 0.90)), frame_index=9, timestamp_s=10.6),
        robot_mask_packet=_mask_packet(frame_index=9, camera_host_t_s=10.6),
        patient_pose_packet=_pose_packet(frame_index=9, camera_host_t_s=10.6),
    )
    jump = runtime.advance(
        _frame(_point(xyz=(0.40, 0.20, 0.90)), frame_index=10, timestamp_s=10.65),
        robot_mask_packet=_mask_packet(frame_index=10, camera_host_t_s=10.65),
        patient_pose_packet=_pose_packet(frame_index=10, camera_host_t_s=10.65),
    )

    assert first.points[0].occlusion_state == "REACQUIRING"
    assert jump.points[0].occlusion_state == "LOST"
    assert jump.points[0].valid is False
    assert jump.points[0].reason == "REACQUISITION_DISCONTINUITY"


def test_composite_pose_uncertainty_cannot_authorize_locked_point() -> None:
    runtime = _runtime(
        _config(
            handeye_bound_m=0.003,
            timing_sweep_bound_m=0.002,
            projection_bound_m=0.001,
            tool_geometry_bound_m=0.001,
        ),
    )
    runtime.advance(
        _frame(_point(uncertainty_bound_m=0.003), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )

    result = runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.1),
        robot_mask_packet=_mask_packet(
            frame_index=8, camera_host_t_s=10.1, masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8,
            camera_host_t_s=10.1,
            covariance_scale=1.0e-6,
        ),
    )

    assert result.points[0].occlusion_state == "BOUNDED_FREEZE"
    assert result.points[0].uncertainty_bound_m == pytest.approx(0.0043)
    assert result.points[0].source == "last_valid_bounded_freeze"


def test_hidden_point_without_direct_history_is_lost() -> None:
    runtime = _runtime()
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )

    result = runtime.advance(
        _frame(hidden),
        robot_mask_packet=_mask_packet(masked_pixel=(3, 4)),
        patient_pose_packet=_pose_packet(),
    )

    assert result.points[0].occlusion_state == "LOST"
    assert result.points[0].reason == "NO_CURRENT_OR_BOUNDED_3D_ESTIMATE"


def test_frame_and_mask_packet_bind_live_camera_profile_and_depth_shape() -> None:
    point = _point()
    observation = OcclusionFrameObservation(
        frame_index=7,
        camera_host_t_s=10.0,
        camera_id="CL8326300A0",
        camera_profile_sha256=SHA_A,
        aligned_depth_shape=(16, 20),
        back_present=True,
        initialized=True,
        points=(point,),
    )
    packet = replace(
        _mask_packet(),
        camera_id="CL8326300A0",
        camera_profile_sha256=SHA_A,
    )
    runtime = _runtime()

    accepted = runtime.advance(
        observation,
        robot_mask_packet=packet,
        patient_pose_packet=_pose_packet(),
    )
    assert accepted.points[0].occlusion_state == "DIRECT"

    wrong_profile = replace(packet, camera_profile_sha256=SHA_B)
    rejected = runtime.advance(
        replace(observation, frame_index=8, camera_host_t_s=10.1),
        robot_mask_packet=replace(
            wrong_profile, frame_index=8, camera_host_t_s=10.1,
            exposure_lower_host_t_s=10.075,
            exposure_nominal_host_t_s=10.080,
            exposure_upper_host_t_s=10.085,
            joint_time_first_s=10.070,
            joint_time_last_s=10.090,
        ),
        patient_pose_packet=_pose_packet(frame_index=8, camera_host_t_s=10.1),
    )
    assert rejected.reason == "INVALID_OCCLUSION_EVIDENCE"


def test_repeated_locked_updates_do_not_accumulate_the_same_pose_bound() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(uncertainty_bound_m=0.003), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )

    bounds = []
    for offset in range(1, 21):
        timestamp = 10.0 + 0.01 * offset
        result = runtime.advance(
            _frame(hidden, frame_index=7 + offset, timestamp_s=timestamp),
            robot_mask_packet=_mask_packet(
                frame_index=7 + offset,
                camera_host_t_s=timestamp,
                masked_pixel=(3, 4),
            ),
            patient_pose_packet=_pose_packet(
                frame_index=7 + offset,
                camera_host_t_s=timestamp,
                translation=(0.001 * offset, 0.0, 0.0),
            ),
        )
        assert result.points[0].occlusion_state == "PATIENT_FRAME_LOCKED"
        bounds.append(result.points[0].uncertainty_bound_m)

    assert bounds == pytest.approx([bounds[0]] * len(bounds))


def test_patient_pose_episode_change_cannot_move_an_old_anchor() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )

    result = runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.1),
        robot_mask_packet=_mask_packet(
            frame_index=8, camera_host_t_s=10.1, masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8,
            camera_host_t_s=10.1,
            translation=(0.5, 0.0, 0.0),
            episode_id="patient-0002",
        ),
    )

    assert result.points[0].occlusion_state == "LOST"
    assert result.points[0].valid is False
    assert result.points[0].xyz_camera_m is None
    assert result.points[0].reason == "PATIENT_POSE_EPISODE_CHANGED"


def test_invalid_pose_cannot_seed_a_patient_frame_anchor() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0, valid=False),
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )

    result = runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.1),
        robot_mask_packet=_mask_packet(
            frame_index=8, camera_host_t_s=10.1, masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8,
            camera_host_t_s=10.1,
            translation=(0.9, 0.0, 0.0),
        ),
    )

    assert result.points[0].occlusion_state == "BOUNDED_FREEZE"
    np.testing.assert_allclose(result.points[0].xyz_camera_m, [0.10, 0.20, 0.90])


def test_pose_provider_no_back_clears_history_even_if_model_claims_back() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )
    no_back = replace(
        _pose_packet(
            frame_index=8,
            camera_host_t_s=10.1,
            valid=False,
        ),
        reason="NO_BACK",
    )

    result = runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.1),
        robot_mask_packet=_mask_packet(
            frame_index=8,
            camera_host_t_s=10.1,
            masked_pixel=(3, 4),
        ),
        patient_pose_packet=no_back,
    )

    assert result.reason == "NO_INITIALIZED_BACK"
    assert result.points[0].occlusion_state == "LOST"
    assert result.points[0].pixel is None


@pytest.mark.parametrize(
    "changes",
    (
        {"max_freeze_age_s": 0.401},
        {"max_uncertainty_bound_m": 0.0101},
        {"reacquire_frames": 4},
        {"reacquire_max_displacement_m": 0.051},
    ),
)
def test_config_rejects_values_looser_than_the_production_safety_envelope(
    changes: dict,
) -> None:
    with pytest.raises(ValueError, match="safety envelope"):
        _config(**changes)


def test_no_back_reset_does_not_allow_camera_time_to_move_backwards() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )
    no_back = replace(
        _frame(_point(), frame_index=8, timestamp_s=11.0),
        back_present=False,
        initialized=False,
    )
    reset_result = runtime.advance(
        no_back,
        robot_mask_packet=_mask_packet(frame_index=8, camera_host_t_s=11.0),
        patient_pose_packet=_pose_packet(frame_index=8, camera_host_t_s=11.0),
    )
    assert reset_result.reason == "NO_INITIALIZED_BACK"

    backwards = runtime.advance(
        _frame(_point(), frame_index=9, timestamp_s=9.0),
        robot_mask_packet=_mask_packet(frame_index=9, camera_host_t_s=9.0),
        patient_pose_packet=_pose_packet(frame_index=9, camera_host_t_s=9.0),
    )
    assert backwards.reason == "NON_MONOTONIC_TIME"
    assert backwards.points[0].valid is False


@pytest.mark.parametrize(
    ("robot_packet", "pose_packet"),
    (
        (None, _pose_packet()),
        (_mask_packet(), None),
    ),
)
def test_missing_runtime_packet_fails_closed_without_raising(
    robot_packet, pose_packet,
) -> None:
    runtime = _runtime()

    result = runtime.advance(
        _frame(_point()),
        robot_mask_packet=robot_packet,
        patient_pose_packet=pose_packet,
    )

    assert result.reason == "INVALID_OCCLUSION_EVIDENCE"
    assert result.points[0].occlusion_state == "LOST"


def test_reacquisition_streak_is_cleared_by_an_invalid_evidence_frame() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )
    runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.5),
        robot_mask_packet=_mask_packet(
            frame_index=8, camera_host_t_s=10.5, masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8, camera_host_t_s=10.5, valid=False,
        ),
    )
    first = runtime.advance(
        _frame(_point(), frame_index=9, timestamp_s=10.6),
        robot_mask_packet=_mask_packet(frame_index=9, camera_host_t_s=10.6),
        patient_pose_packet=_pose_packet(frame_index=9, camera_host_t_s=10.6),
    )
    assert first.points[0].occlusion_state == "REACQUIRING"

    invalid = runtime.advance(
        _frame(_point(), frame_index=10, timestamp_s=10.65),
        robot_mask_packet=replace(
            _mask_packet(frame_index=10, camera_host_t_s=10.65),
            mask_sha256="0" * 64,
        ),
        patient_pose_packet=_pose_packet(frame_index=10, camera_host_t_s=10.65),
    )
    assert invalid.reason == "INVALID_OCCLUSION_EVIDENCE"

    states = []
    for offset in range(4):
        frame_index = 11 + offset
        timestamp = 10.70 + 0.05 * offset
        result = runtime.advance(
            _frame(_point(), frame_index=frame_index, timestamp_s=timestamp),
            robot_mask_packet=_mask_packet(
                frame_index=frame_index, camera_host_t_s=timestamp,
            ),
            patient_pose_packet=_pose_packet(
                frame_index=frame_index, camera_host_t_s=timestamp,
            ),
        )
        states.append(result.points[0].occlusion_state)

    assert states == ["REACQUIRING"] * 4


def test_reacquisition_requires_adjacent_frames_and_a_bounded_gap() -> None:
    runtime = _runtime(_config(reacquire_max_gap_s=0.10))
    runtime.advance(
        _frame(_point(), timestamp_s=10.0),
        robot_mask_packet=_mask_packet(camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(camera_host_t_s=10.0),
    )
    hidden = _point(
        measured=False,
        direct_geometry_accepted=False,
        robot_occluded=True,
    )
    runtime.advance(
        _frame(hidden, frame_index=8, timestamp_s=10.5),
        robot_mask_packet=_mask_packet(
            frame_index=8, camera_host_t_s=10.5, masked_pixel=(3, 4),
        ),
        patient_pose_packet=_pose_packet(
            frame_index=8, camera_host_t_s=10.5, valid=False,
        ),
    )
    runtime.advance(
        _frame(_point(), frame_index=9, timestamp_s=10.6),
        robot_mask_packet=_mask_packet(frame_index=9, camera_host_t_s=10.6),
        patient_pose_packet=_pose_packet(frame_index=9, camera_host_t_s=10.6),
    )

    after_gap = runtime.advance(
        _frame(_point(), frame_index=10, timestamp_s=10.8),
        robot_mask_packet=_mask_packet(frame_index=10, camera_host_t_s=10.8),
        patient_pose_packet=_pose_packet(frame_index=10, camera_host_t_s=10.8),
    )
    assert after_gap.points[0].occlusion_state == "REACQUIRING"

    states = []
    for offset in range(1, 5):
        frame_index = 10 + offset
        timestamp = 10.8 + 0.05 * offset
        result = runtime.advance(
            _frame(_point(), frame_index=frame_index, timestamp_s=timestamp),
            robot_mask_packet=_mask_packet(
                frame_index=frame_index, camera_host_t_s=timestamp,
            ),
            patient_pose_packet=_pose_packet(
                frame_index=frame_index, camera_host_t_s=timestamp,
            ),
        )
        states.append(result.points[0].occlusion_state)

    assert states[:3] == ["REACQUIRING"] * 3
    assert states[3] == "DIRECT"


def test_frame_index_must_increase_even_when_timestamp_increases() -> None:
    runtime = _runtime()
    runtime.advance(
        _frame(_point(), frame_index=7, timestamp_s=10.0),
        robot_mask_packet=_mask_packet(frame_index=7, camera_host_t_s=10.0),
        patient_pose_packet=_pose_packet(frame_index=7, camera_host_t_s=10.0),
    )

    result = runtime.advance(
        _frame(_point(), frame_index=7, timestamp_s=10.1),
        robot_mask_packet=_mask_packet(frame_index=7, camera_host_t_s=10.1),
        patient_pose_packet=_pose_packet(frame_index=7, camera_host_t_s=10.1),
    )

    assert result.reason == "NON_MONOTONIC_TIME"
    assert result.points[0].valid is False


@pytest.mark.parametrize(
    "change",
    (
        {"geometry_sha256": SHA_B},
        {"tool_release_id": "tool_geometry-" + "9" * 20},
        {"handeye_sha256": SHA_D},
        {"camera_latency_sha256": SHA_E},
        {"camera_projection_sha256": SHA_F},
        {"occlusion_profile_sha256": SHA_0},
        {"alignment_sha256": SHA_A},
    ),
)
def test_packet_must_match_every_activated_occlusion_identity(
    change: dict,
) -> None:
    runtime = _runtime()
    packet = replace(_mask_packet(), **change)

    result = runtime.advance(
        _frame(_point()),
        robot_mask_packet=packet,
        patient_pose_packet=_pose_packet(),
    )

    assert result.reason == "INVALID_OCCLUSION_EVIDENCE"
    assert result.points[0].valid is False


def test_stale_or_float_ns_inconsistent_exposure_fails_closed() -> None:
    runtime = _runtime()
    stale = replace(
        _mask_packet(),
        camera_host_t_s=10.2,
        camera_host_monotonic_ns=10_200_000_000,
    )

    result = runtime.advance(
        _frame(_point(), timestamp_s=10.2),
        robot_mask_packet=stale,
        patient_pose_packet=_pose_packet(camera_host_t_s=10.2),
    )
    assert result.reason == "INVALID_OCCLUSION_EVIDENCE"

    inconsistent = replace(
        _mask_packet(),
        exposure_nominal_monotonic_ns=9_000_000_000,
    )
    second = _runtime().advance(
        _frame(_point()),
        robot_mask_packet=inconsistent,
        patient_pose_packet=_pose_packet(),
    )
    assert second.reason == "INVALID_OCCLUSION_EVIDENCE"


def test_camera_device_time_and_covariance_convention_are_mandatory() -> None:
    no_device_time = replace(_mask_packet(), camera_device_t_ms=None)
    result = _runtime().advance(
        _frame(_point()),
        robot_mask_packet=no_device_time,
        patient_pose_packet=_pose_packet(),
    )
    assert result.reason == "INVALID_OCCLUSION_EVIDENCE"

    wrong_convention = replace(
        _pose_packet(),
        covariance_tangent_order="translation_then_rotation",
    )
    second = _runtime().advance(
        _frame(_point()),
        robot_mask_packet=_mask_packet(),
        patient_pose_packet=wrong_convention,
    )
    assert second.reason == "INVALID_OCCLUSION_EVIDENCE"


def test_anisotropic_covariance_uses_declared_rotation_then_translation_order() -> None:
    covariance = np.zeros((6, 6), dtype=np.float64)
    covariance[2, 2] = 4.0e-6
    covariance[3, 3] = 1.0e-6
    packet = replace(_pose_packet(), covariance_se3=covariance)

    bound = packet.propagated_position_bound_m(
        np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )

    assert bound == pytest.approx(math.sqrt(7.814727903251179 * 5.0e-6))


@pytest.mark.parametrize(
    "change",
    (
        {"max_camera_to_exposure_age_ns": 100_000_001},
        {"max_exposure_interval_ns": 40_000_001},
    ),
)
def test_activated_identity_cannot_loosen_timing_safety_envelope(
    change: dict,
) -> None:
    with pytest.raises(ValueError, match="safety envelope"):
        _identity(**change)


@pytest.mark.parametrize(
    "field",
    (
        "joint_mapper_sha256",
        "pair_contract_sha256",
        "controller_binary_sha256",
    ),
)
def test_activated_identity_requires_explicit_robot_provenance_pins(
    field: str,
) -> None:
    with pytest.raises(ValueError, match="identity"):
        _identity(**{field: "not-a-sha256"})
