from __future__ import annotations

from dataclasses import fields, replace
import importlib

import numpy as np
import pytest


MODULE = importlib.import_module("ausc.perception.tracking.depth_noise_profile")
ActivatedDepthNoiseProfile = getattr(MODULE, "ActivatedDepthNoiseProfile", None)
depth_noise_profile_semantic_sha256 = getattr(
    MODULE, "depth_noise_profile_semantic_sha256", None
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
WIDTH = 128
HEIGHT = 96


def _profile_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "camera_id": "orbbec-main",
        "camera_profile_sha256": SHA_A,
        "alignment_sha256": SHA_B,
        "calibration_artifact_sha256": SHA_C,
        "calibration_complete_sha256": SHA_D,
        "width": WIDTH,
        "height": HEIGHT,
        "fx": 400.0,
        "fy": 401.0,
        "cx": (WIDTH - 1) / 2.0,
        "cy": (HEIGHT - 1) / 2.0,
        "depth_scale_mm": 1.0,
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
    return values


def _profile(**changes: object):
    values = _profile_values(**changes)
    semantic = depth_noise_profile_semantic_sha256(**values)
    return ActivatedDepthNoiseProfile(**values, semantic_sha256=semantic)


def test_activated_depth_noise_profile_module_exists() -> None:
    assert ActivatedDepthNoiseProfile is not None
    assert depth_noise_profile_semantic_sha256 is not None


@pytest.mark.skipif(
    ActivatedDepthNoiseProfile is None,
    reason="activated depth-noise profile is not implemented",
)
def test_semantic_digest_binds_every_identity_and_numeric_parameter() -> None:
    profile = _profile()
    valid_changes: dict[str, object] = {
        "camera_id": "orbbec-secondary",
        "camera_profile_sha256": "e" * 64,
        "alignment_sha256": "f" * 64,
        "calibration_artifact_sha256": "1" * 64,
        "calibration_complete_sha256": "2" * 64,
        "width": WIDTH + 1,
        "height": HEIGHT + 1,
        "fx": 401.0,
        "fy": 402.0,
        "cx": profile.cx + 0.1,
        "cy": profile.cy + 0.1,
        "depth_scale_mm": 0.999,
        "calibrated_min_depth_m": 0.21,
        "calibrated_max_depth_m": 2.99,
        "axial_sigma_floor_m": 0.0006,
        "axial_sigma_quadratic_m_inv": 0.0006,
        "pixel_quantization_sigma_px": 0.31,
        "alignment_sigma_u_px": 0.26,
        "alignment_sigma_v_px": 0.21,
        "alignment_sigma_depth_m": 0.0005,
        "fit_residual_scale": 1.1,
        "uncertainty_expansion_factor": 1.1,
        "calibration_quantile": 0.96,
        "calibration_frame_count": 201,
        "calibration_sample_count": 20_001,
        "calibration_depth_bin_count": 11,
    }
    covered = {
        field.name for field in fields(profile) if field.name != "semantic_sha256"
    }
    assert set(valid_changes) == covered

    for name, changed in valid_changes.items():
        with pytest.raises(ValueError, match="semantic SHA-256"):
            replace(profile, **{name: changed})


@pytest.mark.skipif(
    ActivatedDepthNoiseProfile is None,
    reason="activated depth-noise profile is not implemented",
)
@pytest.mark.parametrize(
    "changes",
    (
        {"camera_id": ""},
        {"camera_profile_sha256": "bad"},
        {"alignment_sha256": "bad"},
        {"calibration_artifact_sha256": "bad"},
        {"calibration_complete_sha256": "bad"},
        {"width": 0},
        {"height": True},
        {"fx": 0.0},
        {"fy": float("nan")},
        {"cx": -0.1},
        {"cy": float("inf")},
        {"depth_scale_mm": 0.0},
        {"calibrated_min_depth_m": 0.049},
        {"calibrated_max_depth_m": 3.001},
        {"axial_sigma_floor_m": 0.00049},
        {"axial_sigma_quadratic_m_inv": 0.0},
        {"pixel_quantization_sigma_px": 0.288},
        {"alignment_sigma_u_px": 0.0},
        {"alignment_sigma_v_px": -0.1},
        {"alignment_sigma_depth_m": 0.0},
        {"fit_residual_scale": 0.999},
        {"uncertainty_expansion_factor": 0.999},
        {"calibration_quantile": 0.949},
        {"calibration_frame_count": 49},
        {"calibration_sample_count": 999},
        {"calibration_depth_bin_count": 4},
    ),
)
def test_profile_rejects_uncommissioned_or_nonconservative_values(
    changes: dict[str, object],
) -> None:
    values = _profile_values(**changes)
    with pytest.raises(ValueError):
        semantic = depth_noise_profile_semantic_sha256(**values)
        ActivatedDepthNoiseProfile(**values, semantic_sha256=semantic)


@pytest.mark.skipif(
    ActivatedDepthNoiseProfile is None,
    reason="activated depth-noise profile is not implemented",
)
def test_measurement_covariance_covers_pixel_alignment_and_axial_noise() -> None:
    profile = _profile()
    pixel = np.array([[profile.cx, profile.cy]], dtype=np.float64)

    covariance = profile.measurement_covariances_m2(
        pixel, np.array([1.0], dtype=np.float64)
    )[0]

    expected_u_sigma_px = np.hypot(
        profile.pixel_quantization_sigma_px, profile.alignment_sigma_u_px
    )
    expected_v_sigma_px = np.hypot(
        profile.pixel_quantization_sigma_px, profile.alignment_sigma_v_px
    )
    expected_z_sigma_m = np.hypot(
        np.hypot(profile.axial_sigma_floor_m, profile.axial_sigma_quadratic_m_inv,),
        profile.alignment_sigma_depth_m,
    )
    assert covariance[0, 0] == pytest.approx((expected_u_sigma_px / profile.fx) ** 2)
    assert covariance[1, 1] == pytest.approx((expected_v_sigma_px / profile.fy) ** 2)
    assert covariance[2, 2] == pytest.approx(expected_z_sigma_m ** 2)
    np.testing.assert_allclose(covariance, covariance.T, atol=0.0)
    assert np.linalg.eigvalsh(covariance).min() > 0.0


@pytest.mark.skipif(
    ActivatedDepthNoiseProfile is None,
    reason="activated depth-noise profile is not implemented",
)
def test_measurement_covariance_grows_with_range_and_alignment_uncertainty() -> None:
    profile = _profile()
    pixel = np.array([[profile.cx + 20.0, profile.cy - 10.0]])
    near = profile.measurement_covariances_m2(pixel, np.array([0.5]))[0]
    far = profile.measurement_covariances_m2(pixel, np.array([2.0]))[0]
    larger_alignment = _profile(
        alignment_sigma_u_px=0.50,
        alignment_sigma_v_px=0.40,
        alignment_sigma_depth_m=0.0008,
    ).measurement_covariances_m2(pixel, np.array([2.0]))[0]

    assert np.trace(far) > np.trace(near)
    assert np.trace(larger_alignment) > np.trace(far)


@pytest.mark.skipif(
    ActivatedDepthNoiseProfile is None,
    reason="activated depth-noise profile is not implemented",
)
def test_uncalibrated_depth_and_mismatched_camera_contract_fail_closed() -> None:
    profile = _profile()
    pixel = np.array([[profile.cx, profile.cy]])
    with pytest.raises(ValueError, match="calibrated range"):
        profile.measurement_covariances_m2(pixel, np.array([0.19]))
    with pytest.raises(ValueError, match="camera profile"):
        profile.validate_for_camera(
            camera_id=profile.camera_id,
            camera_profile_sha256="e" * 64,
            alignment_sha256=profile.alignment_sha256,
            width=profile.width,
            height=profile.height,
            fx=profile.fx,
            fy=profile.fy,
            cx=profile.cx,
            cy=profile.cy,
            depth_scale_mm=profile.depth_scale_mm,
        )


@pytest.mark.skipif(
    ActivatedDepthNoiseProfile is None,
    reason="activated depth-noise profile is not implemented",
)
def test_common_alignment_and_fit_terms_do_not_average_away() -> None:
    profile = _profile()
    alignment = profile.common_alignment_pose_covariance_se3(
        maximum_depth_m=1.5, target_radius_m=0.45
    )
    fit = profile.fit_residual_pose_covariance_se3(
        fit_rmse_m=0.002, target_radius_m=0.45
    )

    assert alignment.shape == (6, 6)
    assert fit.shape == (6, 6)
    assert np.all(np.diag(alignment) > 0.0)
    assert np.all(np.diag(fit) > 0.0)
    assert np.linalg.eigvalsh(alignment).min() > 0.0
    assert np.linalg.eigvalsh(fit).min() > 0.0
