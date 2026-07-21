from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from ausc.perception.tracking.geometry import Intrinsics
from ausc.perception.tracking.back_mask_packet import BackMaskPacket
from ausc.perception.tracking.depth_noise_profile import (
    ActivatedDepthNoiseProfile,
    depth_noise_profile_semantic_sha256,
)
from ausc.perception.tracking.occlusion_runtime import (
    ActivatedOcclusionIdentity,
    PatientPosePacket,
)
from ausc.perception.tracking.patient_pose_provider import (
    MetricPatientPoseProvider,
    PatientPoseProviderConfig,
)
from ausc.perception.tracking.patient_frame_evidence import (
    PatientFrameEvidence,
    PatientFrameEvidenceError,
)


_ROBOT_TEST_PATH = (
    Path(__file__).parents[2] / "robot" / "tests" / "test_frame_evidence_bundle.py"
)
_ROBOT_SPEC = importlib.util.spec_from_file_location(
    "_wam_patient_pose_frame_bundle_helpers", _ROBOT_TEST_PATH
)
assert _ROBOT_SPEC is not None and _ROBOT_SPEC.loader is not None
_ROBOT_HELPERS = importlib.util.module_from_spec(_ROBOT_SPEC)
_ROBOT_SPEC.loader.exec_module(_ROBOT_HELPERS)


HEIGHT = 96
WIDTH = 128
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
INTRINSICS = Intrinsics(
    fx=400.0,
    fy=400.0,
    cx=(WIDTH - 1) / 2.0,
    cy=(HEIGHT - 1) / 2.0,
    width=WIDTH,
    height=HEIGHT,
    depth_scale=1.0,
)


def _depth_profile(**changes: object) -> ActivatedDepthNoiseProfile:
    values: dict[str, object] = {
        "camera_id": "orbbec-main",
        "camera_profile_sha256": SHA_A,
        "alignment_sha256": SHA_B,
        "calibration_artifact_sha256": SHA_D,
        "calibration_complete_sha256": SHA_F,
        "width": WIDTH,
        "height": HEIGHT,
        "fx": INTRINSICS.fx,
        "fy": INTRINSICS.fy,
        "cx": INTRINSICS.cx,
        "cy": INTRINSICS.cy,
        "depth_scale_mm": INTRINSICS.depth_scale,
        "calibrated_min_depth_m": 0.20,
        "calibrated_max_depth_m": 3.0,
        "axial_sigma_floor_m": 0.0005,
        "axial_sigma_quadratic_m_inv": 0.0005,
        "pixel_quantization_sigma_px": 0.30,
        "alignment_sigma_u_px": 0.25,
        "alignment_sigma_v_px": 0.20,
        "alignment_sigma_depth_m": 0.0004,
        "fit_residual_scale": 1.0,
        "uncertainty_expansion_factor": 1.0,
        "calibration_quantile": 0.95,
        "calibration_frame_count": 200,
        "calibration_sample_count": 20_000,
        "calibration_depth_bin_count": 10,
    }
    values.update(changes)
    semantic_sha256 = depth_noise_profile_semantic_sha256(**values)
    return ActivatedDepthNoiseProfile(**values, semantic_sha256=semantic_sha256,)


def _textured_rgb() -> np.ndarray:
    rng = np.random.RandomState(20260717)
    gray = rng.randint(0, 256, (HEIGHT, WIDTH), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0)
    for y in range(8, HEIGHT - 8, 12):
        for x in range(8, WIDTH - 8, 12):
            cv2.circle(gray, (x, y), 2, 255 if (x + y) % 24 else 0, -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _depth(value_mm: int = 1000) -> np.ndarray:
    return np.full((HEIGHT, WIDTH), value_mm, dtype=np.uint16)


def _masks() -> tuple[np.ndarray, np.ndarray]:
    back = np.ones((HEIGHT, WIDTH), dtype=np.bool_)
    robot = np.zeros_like(back)
    return back, robot


def _provider(
    config: PatientPoseProviderConfig | None = None,
    depth_noise_profile: ActivatedDepthNoiseProfile | None = None,
    *,
    allow_unbound_test_input: bool | None = True,
) -> MetricPatientPoseProvider:
    kwargs = (
        {}
        if allow_unbound_test_input is None
        else {"allow_unbound_test_input": allow_unbound_test_input}
    )
    return MetricPatientPoseProvider(
        INTRINSICS,
        camera_id="orbbec-main",
        camera_profile_sha256=SHA_A,
        alignment_sha256=SHA_B,
        patient_pose_release_id="patient_pose-" + "4" * 20,
        patient_pose_sha256=SHA_C,
        depth_noise_profile=depth_noise_profile or _depth_profile(),
        back_mask_release_id="back_mask-" + "5" * 20,
        back_mask_sha256=SHA_E,
        config=config or PatientPoseProviderConfig(),
        **kwargs,
    )


def _activated_identity() -> ActivatedOcclusionIdentity:
    return ActivatedOcclusionIdentity(
        camera_id="orbbec-main",
        camera_profile_sha256=SHA_A,
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
        camera_projection_sha256=SHA_E,
        occlusion_profile_sha256=SHA_F,
        occlusion_runtime_profile_sha256=SHA_A,
        alignment_sha256=SHA_B,
        joint_mapper_sha256=SHA_A,
        pair_contract_sha256=SHA_B,
        controller_binary_sha256=SHA_C,
        patient_pose_release_id="patient_pose-" + "4" * 20,
        patient_pose_sha256=SHA_C,
        depth_noise_model_sha256=_depth_profile().semantic_sha256,
        back_mask_release_id="back_mask-" + "5" * 20,
        back_mask_sha256=SHA_E,
        max_camera_to_exposure_age_ns=100_000_000,
        max_exposure_interval_ns=40_000_000,
    )


def _validate_packet(
    packet: PatientPosePacket, *, frame_index: int, camera_host_t_s: float,
) -> bool:
    return packet.validate_for_frame(
        frame_index=frame_index,
        camera_host_t_s=camera_host_t_s,
        camera_id="orbbec-main",
        camera_profile_sha256=SHA_A,
        activated_identity=_activated_identity(),
    )


def _process(
    provider: MetricPatientPoseProvider,
    *,
    frame_index: int,
    timestamp_s: float,
    rgb: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    back_mask: np.ndarray | None = None,
    robot_mask: np.ndarray | None = None,
    episode_id: str = "patient-0001",
) -> PatientPosePacket:
    default_back, default_robot = _masks()
    return provider.process(
        frame_index=frame_index,
        camera_host_t_s=timestamp_s,
        rgb=_textured_rgb() if rgb is None else rgb,
        aligned_depth=_depth() if depth is None else depth,
        back_mask=default_back if back_mask is None else back_mask,
        robot_mask=default_robot if robot_mask is None else robot_mask,
        episode_id=episode_id,
    )


def _evidence(
    *,
    rgb: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    back_mask: np.ndarray | None = None,
    back_mask_release_id: str = "back_mask-" + "5" * 20,
    back_mask_model_sha256: str = SHA_E,
    camera_id: str = "orbbec-main",
    camera_profile_sha256: str = SHA_A,
    alignment_sha256: str = SHA_B,
    depth_scale_mm: float = 1.0,
) -> PatientFrameEvidence:
    frame = _ROBOT_HELPERS._camera_frame(
        rgb=_textured_rgb() if rgb is None else rgb,
        depth=_depth() if depth is None else depth,
    )
    frame = replace(frame, depth_scale=depth_scale_mm)
    exposure = _ROBOT_HELPERS._exposure(
        camera_id=camera_id,
        camera_profile_sha256=camera_profile_sha256,
        alignment_sha256=alignment_sha256,
    )
    trajectory = _ROBOT_HELPERS._trajectory()
    robot_mask_packet = _ROBOT_HELPERS._mask_packet(
        exposure=exposure,
        trajectory=trajectory,
        mask=np.zeros((HEIGHT, WIDTH), dtype=np.bool_),
    )
    frame_bundle = _ROBOT_HELPERS._bundle(
        camera_frame=frame,
        exposure=exposure,
        trajectory=trajectory,
        robot_mask_packet=robot_mask_packet,
        camera_id=camera_id,
        camera_profile_sha256=camera_profile_sha256,
        alignment_sha256=alignment_sha256,
    )
    packet = BackMaskPacket.build(
        camera_frame=frame_bundle.camera_frame,
        mask=(
            np.ones((HEIGHT, WIDTH), dtype=np.bool_) if back_mask is None else back_mask
        ),
        back_mask_release_id=back_mask_release_id,
        back_mask_model_sha256=back_mask_model_sha256,
        operator_gate_decision_id="operator_back_gate-" + "6" * 64,
    )
    return PatientFrameEvidence.build(
        frame_bundle=frame_bundle, back_mask_packet=packet,
    )


def test_raw_array_process_is_disabled_by_default() -> None:
    provider = _provider(allow_unbound_test_input=None)

    with pytest.raises(PermissionError, match="unbound raw.*disabled"):
        _process(provider, frame_index=1, timestamp_s=10.0)


def test_process_evidence_is_the_default_production_entrypoint() -> None:
    provider = _provider(allow_unbound_test_input=False)
    evidence = _evidence()

    packet = provider.process_evidence(evidence, "patient-0001")

    assert packet.valid, packet.reason
    assert packet.frame_index == evidence.index
    assert packet.camera_host_t_s == evidence.t
    assert packet.back_mask_release_id == evidence.back_mask_packet.back_mask_release_id
    assert packet.back_mask_sha256 == evidence.back_mask_packet.back_mask_model_sha256


def test_process_evidence_revalidates_the_bound_evidence_id() -> None:
    provider = _provider(allow_unbound_test_input=False)
    evidence = _evidence()
    forged = object.__new__(PatientFrameEvidence)
    for name in evidence.__dataclass_fields__:
        object.__setattr__(
            forged,
            name,
            (
                "patient_frame_evidence-" + "0" * 64
                if name == "evidence_id"
                else getattr(evidence, name)
            ),
        )

    with pytest.raises(PatientFrameEvidenceError, match="evidence ID"):
        provider.process_evidence(forged, episode_id="patient-0001")


def test_process_evidence_rehashes_immutable_camera_bytes() -> None:
    provider = _provider(allow_unbound_test_input=False)
    evidence = _evidence()
    forged_rgb = np.array(evidence.rgb, copy=True)
    forged_rgb[0, 0, 0] ^= np.uint8(1)
    object.__setattr__(
        evidence.frame_bundle.camera_frame, "frame", forged_rgb,
    )

    with pytest.raises(ValueError, match="RGB SHA-256"):
        provider.process_evidence(evidence, episode_id="patient-0001")


@pytest.mark.parametrize(
    ("evidence_changes", "message"),
    (
        ({"camera_id": "other-camera"}, "camera identity"),
        ({"camera_profile_sha256": "9" * 64}, "camera profile"),
        ({"alignment_sha256": "9" * 64}, "alignment"),
        ({"depth_scale_mm": 0.5}, "depth scale"),
        ({"back_mask_release_id": "back_mask-" + "6" * 20}, "back-mask release",),
        ({"back_mask_model_sha256": "7" * 64}, "back-mask model"),
    ),
)
def test_process_evidence_rejects_nonactivated_provenance(
    evidence_changes: dict[str, object], message: str
) -> None:
    provider = _provider(allow_unbound_test_input=False)
    evidence = _evidence(**evidence_changes)

    with pytest.raises(ValueError, match=message):
        provider.process_evidence(evidence, episode_id="patient-0001")


def test_process_evidence_revalidates_depth_profile_semantics() -> None:
    profile = _depth_profile()
    provider = _provider(depth_noise_profile=profile, allow_unbound_test_input=False,)
    evidence = _evidence()
    object.__setattr__(profile, "axial_sigma_floor_m", 0.001)

    with pytest.raises(ValueError, match="semantic SHA-256"):
        provider.process_evidence(evidence, episode_id="patient-0001")


def test_process_evidence_preserves_explicit_no_back_status() -> None:
    provider = _provider(allow_unbound_test_input=False)
    evidence = _evidence(back_mask=np.zeros((HEIGHT, WIDTH), dtype=np.bool_))

    packet = provider.process_evidence(evidence, episode_id="patient-0001")

    assert packet.valid is False
    assert packet.reason == "NO_BACK"


def test_provider_requires_a_matching_activated_depth_noise_profile() -> None:
    profile = _depth_profile()
    provider = _provider(depth_noise_profile=profile)

    assert provider.depth_noise_model_sha256 == profile.semantic_sha256
    mismatch = _depth_profile(camera_profile_sha256="9" * 64)
    with pytest.raises(ValueError, match="camera profile"):
        _provider(depth_noise_profile=mismatch)


def test_depth_outside_the_activated_calibration_range_fails_closed() -> None:
    provider = _provider(
        depth_noise_profile=_depth_profile(
            calibrated_min_depth_m=0.80, calibrated_max_depth_m=1.20,
        )
    )

    packet = _process(provider, frame_index=1, timestamp_s=10.0, depth=_depth(700),)

    assert packet.valid is False
    assert packet.reason == "INSUFFICIENT_UNOCCLUDED_BACK"
    assert provider.reference_feature_count == 0


def test_pose_covariance_combines_reference_current_and_fit_uncertainty() -> None:
    provider = _provider()
    source = np.array(
        [
            [-0.12, -0.10, 1.0],
            [0.12, -0.10, 1.0],
            [-0.12, 0.10, 1.0],
            [0.12, 0.10, 1.0],
            [-0.08, 0.00, 1.04],
            [0.08, 0.00, 0.96],
        ],
        dtype=np.float64,
    )
    low = np.repeat((np.eye(3) * 1.0e-7)[None, :, :], len(source), axis=0)
    high = low * 9.0
    depth = np.ones(len(source), dtype=np.float64)

    baseline = provider._covariance(
        source,
        np.eye(3),
        0.0,
        reference_covariances_m2=low,
        current_covariances_m2=low,
        current_depth_m=depth,
    )
    reference_high = provider._covariance(
        source,
        np.eye(3),
        0.0,
        reference_covariances_m2=high,
        current_covariances_m2=low,
        current_depth_m=depth,
    )
    current_high = provider._covariance(
        source,
        np.eye(3),
        0.0,
        reference_covariances_m2=low,
        current_covariances_m2=high,
        current_depth_m=depth,
    )
    residual_high = provider._covariance(
        source,
        np.eye(3),
        0.002,
        reference_covariances_m2=low,
        current_covariances_m2=low,
        current_depth_m=depth,
    )

    assert baseline is not None
    assert reference_high is not None
    assert current_high is not None
    assert residual_high is not None
    assert np.trace(reference_high) > np.trace(baseline)
    assert np.trace(current_high) > np.trace(baseline)
    assert np.trace(residual_high) > np.trace(baseline)
    alignment_floor = provider.depth_noise_profile.common_alignment_pose_covariance_se3(
        maximum_depth_m=1.0, target_radius_m=provider.config.target_radius_m,
    )
    assert np.all(np.diag(baseline) >= np.diag(alignment_floor))

    out_of_calibration = depth.copy()
    out_of_calibration[0] = 0.10
    assert (
        provider._covariance(
            source,
            np.eye(3),
            0.0,
            reference_covariances_m2=low,
            current_covariances_m2=low,
            current_depth_m=out_of_calibration,
        )
        is None
    )

    out_of_calibration_reference = source.copy()
    out_of_calibration_reference[0, 2] = 3.10
    assert (
        provider._covariance(
            out_of_calibration_reference,
            np.eye(3),
            0.0,
            reference_covariances_m2=low,
            current_covariances_m2=low,
            current_depth_m=depth,
        )
        is None
    )


def test_initial_static_frame_builds_bounded_reference_and_valid_identity() -> None:
    provider = _provider()

    packet = _process(provider, frame_index=1, timestamp_s=10.0)

    assert _validate_packet(packet, frame_index=1, camera_host_t_s=10.0)
    assert packet.valid is True
    assert packet.source == "rgbd_feature_ransac_plus_icp"
    np.testing.assert_allclose(
        packet.rotation_camera_from_patient, np.eye(3), atol=1e-9
    )
    np.testing.assert_allclose(
        packet.translation_camera_from_patient_m, np.zeros(3), atol=1e-9
    )
    assert packet.scale_audit == pytest.approx(1.0)
    assert packet.support_mask.shape == (10,)
    assert packet.support_mask.dtype == np.bool_
    assert np.count_nonzero(packet.support_mask) >= 4
    assert 4 <= provider.reference_feature_count <= provider.config.max_features
    assert 4 <= provider.reference_surfel_count <= provider.config.max_surfels
    assert provider.reference_memory_bytes <= provider.config.max_reference_bytes


def test_known_five_millimetre_translation_is_recovered_metricly() -> None:
    provider = _provider()
    rgb0 = _textured_rgb()
    first = _process(provider, frame_index=1, timestamp_s=10.0, rgb=rgb0,)
    assert first.valid
    rgb1 = cv2.warpAffine(
        rgb0,
        np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    packet = _process(provider, frame_index=2, timestamp_s=10.05, rgb=rgb1,)

    assert packet.valid, packet.reason
    assert packet.fit_rmse_m <= 0.004
    assert packet.scale_audit == pytest.approx(1.0, abs=0.01)
    assert packet.translation_camera_from_patient_m[0] == pytest.approx(
        0.005, abs=0.0015
    )
    assert abs(packet.translation_camera_from_patient_m[1]) < 0.0015
    assert abs(packet.translation_camera_from_patient_m[2]) < 0.0015


def test_static_sequence_keeps_reference_memory_and_point_counts_bounded() -> None:
    provider = _provider()
    rgb = _textured_rgb()
    first = _process(provider, frame_index=1, timestamp_s=10.0, rgb=rgb,)
    assert first.valid
    initial_bytes = provider.reference_memory_bytes

    for frame_index in range(2, 18):
        packet = _process(
            provider,
            frame_index=frame_index,
            timestamp_s=10.0 + 0.02 * (frame_index - 1),
            rgb=rgb,
        )
        assert packet.valid, packet.reason
        assert provider.reference_feature_count <= provider.config.max_features
        assert provider.reference_surfel_count <= provider.config.max_surfels
        assert provider.reference_memory_bytes <= provider.config.max_reference_bytes

    assert provider.reference_memory_bytes <= initial_bytes


def test_robot_occlusion_drops_polluted_features_but_tracks_remaining_patient() -> None:
    provider = _provider()
    rgb0 = _textured_rgb()
    assert _process(provider, frame_index=1, timestamp_s=10.0, rgb=rgb0).valid
    rgb1 = cv2.warpAffine(
        rgb0,
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (WIDTH, HEIGHT),
        borderMode=cv2.BORDER_REFLECT,
    )
    back, robot = _masks()
    robot[24:72, 48:80] = True
    rgb1[robot] = (15, 15, 15)

    packet = _process(
        provider,
        frame_index=2,
        timestamp_s=10.05,
        rgb=rgb1,
        back_mask=back,
        robot_mask=robot,
    )

    assert packet.valid, packet.reason
    assert packet.geometry_accepted is True
    assert packet.fit_rmse_m <= 0.004
    assert provider.last_inlier_count >= provider.config.min_math_support
    assert provider.last_robot_rejected_count > 0


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    (
        ("no_back", "NO_BACK"),
        ("robot_pollution", "ROBOT_MASK_POLLUTION"),
        ("insufficient_features", "INSUFFICIENT_FEATURE_SUPPORT"),
        ("degenerate", "DEGENERATE_GEOMETRY"),
    ),
)
def test_initialization_fails_closed_without_independent_metric_support(
    case: str, expected_reason: str
) -> None:
    provider = _provider()
    rgb = _textured_rgb()
    back, robot = _masks()
    if case == "no_back":
        back[:] = False
    elif case == "robot_pollution":
        robot[:, :100] = True
    elif case == "insufficient_features":
        rgb[:] = 127
    elif case == "degenerate":
        back[:] = False
        back[46:50, :] = True

    packet = _process(
        provider,
        frame_index=1,
        timestamp_s=10.0,
        rgb=rgb,
        back_mask=back,
        robot_mask=robot,
    )

    assert packet.valid is False
    assert packet.geometry_accepted is False
    assert packet.reason.startswith(expected_reason)
    assert _validate_packet(packet, frame_index=1, camera_host_t_s=10.0) is False


def test_scale_drift_above_one_percent_is_rejected() -> None:
    provider = _provider()
    rgb = _textured_rgb()
    assert _process(provider, frame_index=1, timestamp_s=10.0, rgb=rgb).valid

    packet = _process(
        provider, frame_index=2, timestamp_s=10.05, rgb=rgb, depth=_depth(1020),
    )

    assert packet.valid is False
    assert packet.reason.startswith("SCALE_DRIFT")
    assert _validate_packet(packet, frame_index=2, camera_host_t_s=10.05) is False

    # A rejected frame must not move or redefine the accepted patient frame.
    recovered = _process(
        provider, frame_index=3, timestamp_s=10.10, rgb=rgb, depth=_depth(),
    )
    assert recovered.valid, recovered.reason
    np.testing.assert_allclose(
        recovered.translation_camera_from_patient_m, np.zeros(3), atol=0.001
    )


def test_inconsistent_depth_over_four_millimetres_is_rejected() -> None:
    provider = _provider()
    rgb = _textured_rgb()
    assert _process(provider, frame_index=1, timestamp_s=10.0, rgb=rgb).valid
    bad_depth = _depth()
    bad_depth[:, ::2] = 990
    bad_depth[:, 1::2] = 1010

    packet = _process(
        provider, frame_index=2, timestamp_s=10.05, rgb=rgb, depth=bad_depth,
    )

    assert packet.valid is False
    assert packet.reason.startswith("FIT_RESIDUAL")


def test_large_motion_step_is_rejected_even_when_features_track() -> None:
    config = replace(PatientPoseProviderConfig(), max_translation_step_m=0.003)
    provider = _provider(config)
    rgb0 = _textured_rgb()
    assert _process(provider, frame_index=1, timestamp_s=10.0, rgb=rgb0).valid
    rgb1 = cv2.warpAffine(
        rgb0,
        np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (WIDTH, HEIGHT),
        borderMode=cv2.BORDER_REFLECT,
    )

    packet = _process(provider, frame_index=2, timestamp_s=10.05, rgb=rgb1,)

    assert packet.valid is False
    assert packet.reason.startswith("MOTION_JUMP")


def test_covariance_target_bound_above_ten_millimetres_is_rejected() -> None:
    provider = _provider(depth_noise_profile=_depth_profile(axial_sigma_floor_m=0.020))

    packet = _process(provider, frame_index=1, timestamp_s=10.0)

    assert packet.valid is False
    assert packet.reason.startswith("POSE_BOUND_EXCEEDED")
    assert provider.reference_feature_count == 0
    assert provider.reference_surfel_count == 0
    assert provider.reference_memory_bytes == 0


@pytest.mark.parametrize(
    "changes",
    (
        {"min_back_pixels": 255},
        {"max_robot_back_fraction": 0.651},
        {"robot_mask_dilation_px": 3},
        {"feature_quality": 0.0049},
        {"feature_min_distance_px": 4},
        {"lk_max_level": 4},
        {"max_forward_backward_error_px": 1.501},
        {"min_second_spatial_eigenvalue_m2": 1.99e-4},
        {"ransac_iterations": 95},
        {"ransac_threshold_m": 0.0081},
        {"min_ransac_inlier_fraction": 0.399},
        {"icp_iterations": 3},
        {"icp_huber_delta_m": 0.0041},
        {"max_fit_rmse_m": 0.0041},
        {"max_translation_step_m": 0.0501},
        {"max_rotation_step_rad": np.deg2rad(15.01)},
        {"max_covariance_condition": 1.01e9},
        {"target_radius_m": 0.449},
        {"max_frame_gap_s": 0.251},
    ),
)
def test_patient_pose_config_cannot_loosen_production_authorization(
    changes: dict,
) -> None:
    with pytest.raises(ValueError, match="production safety envelope"):
        replace(PatientPoseProviderConfig(), **changes)


def test_episode_change_and_non_monotonic_frames_require_explicit_reset() -> None:
    provider = _provider()
    assert _process(provider, frame_index=4, timestamp_s=10.0).valid

    changed = _process(
        provider, frame_index=5, timestamp_s=10.05, episode_id="patient-0002",
    )
    assert changed.valid is False
    assert changed.reason == "EPISODE_CHANGED_REQUIRES_RESET"

    still_latched = _process(
        provider, frame_index=6, timestamp_s=10.10, episode_id="patient-0001",
    )
    assert still_latched.valid is False
    assert still_latched.reason == "RESET_REQUIRED"

    provider.reset()
    assert _process(
        provider, frame_index=7, timestamp_s=10.15, episode_id="patient-0002",
    ).valid

    stale = _process(
        provider, frame_index=7, timestamp_s=10.16, episode_id="patient-0002",
    )
    assert stale.valid is False
    assert stale.reason == "NON_MONOTONIC_FRAME_OR_TIME"


def test_reset_creates_a_new_pose_epoch_for_the_same_patient_episode() -> None:
    provider = _provider()
    first = _process(
        provider, frame_index=1, timestamp_s=10.0, episode_id="patient-0001",
    )
    assert first.valid

    provider.reset()
    second = _process(
        provider, frame_index=2, timestamp_s=10.1, episode_id="patient-0001",
    )

    assert second.valid
    assert second.episode_id != first.episode_id
    assert first.episode_id.startswith("patient-0001/")
    assert second.episode_id.startswith("patient-0001/")
