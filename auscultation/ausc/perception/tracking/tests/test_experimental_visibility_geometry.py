from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import ausc.perception.tracking.experimental_visibility_geometry as geometry_module
from ausc.perception.tracking.experimental_visibility_geometry import (
    DecisionSurface,
    evaluate_visibility_geometry,
)


SURFACE = DecisionSurface(
    min_relative_geometry_weight=0.05,
    max_depth_consistency_sigma=3.0,
    min_second_spatial_eigenvalue_m2=2.0e-4,
    min_hull_area_m2=1.0e-3,
    min_spread_m=8.0e-2,
    max_condition_number=40.0,
    max_residual_m=4.0e-3,
    max_target_leverage=4.0,
    max_target_error_p95_m=7.0e-3,
)


def _evaluate(
    reference: np.ndarray,
    *,
    observed: np.ndarray | None = None,
    target: np.ndarray | None = None,
    mask_confidence: np.ndarray | None = None,
    lateral_noise_std_m: np.ndarray | None = None,
    axial_noise_std_m: np.ndarray | None = None,
    depth_m: np.ndarray | None = None,
    surface: DecisionSurface = SURFACE,
):
    n = len(reference)
    observed = reference.copy() if observed is None else observed
    target = np.array([0.0, 0.0, 1.0]) if target is None else target
    return evaluate_visibility_geometry(
        reference,
        observed,
        target,
        depth_m=observed[:, 2].copy() if depth_m is None else depth_m,
        mask_confidence=np.full(n, 0.95) if mask_confidence is None else mask_confidence,
        lateral_noise_std_m=(
            np.full(n, 1.0e-3) if lateral_noise_std_m is None else lateral_noise_std_m
        ),
        axial_noise_std_m=(
            np.full(n, 2.0e-3) if axial_noise_std_m is None else axial_noise_std_m
        ),
        robust_delta_m=3.0e-3,
        decision_surface=surface,
    )


def _broad_four() -> np.ndarray:
    return np.array(
        [
            [-0.12, -0.15, 1.00],
            [0.12, -0.15, 1.00],
            [-0.12, 0.15, 1.00],
            [0.12, 0.15, 1.00],
        ],
        dtype=np.float64,
    )


def test_four_widely_distributed_points_can_pass_declared_sensitivity_surface() -> None:
    result = _evaluate(_broad_four())

    assert result.accepted
    assert result.decision == "pass_on_declared_sensitivity_surface"
    assert result.scope == "parameter_sensitivity_only"
    assert result.sim3_scale_enabled is False
    assert result.effective_support < 6.0
    assert result.spatial_eigenvalues.shape == (3,)
    assert result.hull_area_m2 >= SURFACE.min_hull_area_m2
    assert np.isclose(result.hull_area_m2, 0.072, atol=1e-12)
    assert result.spread_m >= SURFACE.min_spread_m
    assert np.isfinite(result.condition_number)
    assert result.residual_m <= SURFACE.max_residual_m
    assert result.target_leverage <= SURFACE.max_target_leverage
    assert result.propagated_covariance.shape == (3, 3)
    assert np.isfinite(result.propagated_covariance).all()
    assert result.target_error_p95_m <= SURFACE.max_target_error_p95_m
    assert result.decision_surface == SURFACE


def test_six_clustered_near_collinear_points_fail_while_four_broad_points_pass() -> None:
    clustered_six = np.array(
        [
            [-0.025, -0.0010, 1.0],
            [-0.015, 0.0007, 1.0],
            [-0.005, -0.0004, 1.0],
            [0.005, 0.0005, 1.0],
            [0.015, -0.0008, 1.0],
            [0.025, 0.0010, 1.0],
        ],
        dtype=np.float64,
    )
    target = np.array([0.0, 0.14, 1.0])

    clustered = _evaluate(clustered_six, target=target)
    broad = _evaluate(_broad_four(), target=target)

    assert not clustered.accepted
    assert broad.accepted
    assert clustered.effective_support > broad.effective_support
    assert clustered.target_error_p95_m > broad.target_error_p95_m
    assert clustered.target_error_p95_m > SURFACE.max_target_error_p95_m
    assert "target_uncertainty_above_surface" in clustered.failure_reasons


def test_three_collinear_correspondences_fail_closed() -> None:
    points = np.array(
        [[-0.1, 0.0, 1.0], [0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
        dtype=np.float64,
    )

    result = _evaluate(points)

    assert not result.accepted
    assert result.decision == "fail_closed"
    assert "degenerate_spatial_rank" in result.failure_reasons


def test_rank_deficient_weighted_information_fails_closed() -> None:
    mask_confidence = np.array([0.9, 0.0, 0.0, 0.0])

    result = _evaluate(_broad_four(), mask_confidence=mask_confidence)

    assert not result.accepted
    assert any("rank" in reason for reason in result.failure_reasons)
    assert result.effective_support == 1.0


def test_excessive_target_leverage_fails_even_with_good_constellation() -> None:
    surface = replace(
        SURFACE,
        max_target_leverage=2.0,
        max_target_error_p95_m=1.0,
    )

    result = _evaluate(
        _broad_four(),
        target=np.array([1.5, 1.5, 1.0]),
        surface=surface,
    )

    assert not result.accepted
    assert "excessive_target_leverage" in result.failure_reasons


def test_high_rigid_fit_residual_fails_closed() -> None:
    reference = _broad_four()
    observed = reference.copy()
    observed[3] += np.array([0.050, -0.035, 0.025])

    result = _evaluate(reference, observed=observed)

    assert not result.accepted
    assert result.residual_m > SURFACE.max_residual_m
    assert "residual_above_surface" in result.failure_reasons


def test_non_finite_measurement_covariance_fails_closed() -> None:
    axial_noise = np.full(4, 2.0e-3)
    axial_noise[2] = np.nan

    result = _evaluate(_broad_four(), axial_noise_std_m=axial_noise)

    assert not result.accepted
    assert result.decision == "fail_closed"
    assert "non_finite_covariance" in result.failure_reasons
    assert result.target_error_p95_m == float("inf")


def test_underflowed_positive_noise_fails_closed_instead_of_reaching_svd() -> None:
    tiny = np.full(4, 1.0e-300)

    result = _evaluate(
        _broad_four(),
        lateral_noise_std_m=tiny,
        axial_noise_std_m=tiny,
    )

    assert not result.accepted
    assert result.decision == "fail_closed"
    assert result.failure_reasons == ("non_finite_covariance",)
    assert result.target_error_p95_m == float("inf")


def test_linear_algebra_failure_is_caught_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_svd(*_args, **_kwargs):
        raise np.linalg.LinAlgError("synthetic decomposition failure")

    monkeypatch.setattr(geometry_module.np.linalg, "svd", fail_svd)

    result = _evaluate(_broad_four())

    assert not result.accepted
    assert result.decision == "fail_closed"
    assert result.failure_reasons == ("non_finite_covariance",)


def test_positive_but_wrong_depth_fails_explicit_observation_consistency_gate() -> None:
    reference = _broad_four()
    wrong_depth = reference[:, 2] + 0.020

    result = _evaluate(reference, depth_m=wrong_depth)

    assert not result.accepted
    assert "depth_observation_mismatch" in result.failure_reasons
    assert np.allclose(result.depth_residuals_m, 0.020)
    assert np.isclose(result.depth_consistency_max_sigma, 10.0)
    assert result.depth_provenance == "explicit_depth_vs_observed_camera_z"


def test_consistent_near_and_far_depth_change_propagated_covariance_evidence() -> None:
    near = _broad_four()
    near[:, 2] = 0.65
    far = _broad_four()
    far[:, 2] = 1.60

    near_result = _evaluate(near, target=np.array([0.0, 0.0, 0.65]))
    far_result = _evaluate(far, target=np.array([0.0, 0.0, 1.60]))

    assert near_result.depth_consistency_max_sigma == 0.0
    assert far_result.depth_consistency_max_sigma == 0.0
    assert not np.array_equal(
        near_result.propagated_covariance,
        far_result.propagated_covariance,
    )
    assert near_result.target_error_p95_m != far_result.target_error_p95_m


def test_five_and_ten_mm_contours_are_reference_only_p95_comparisons() -> None:
    reference = _broad_four()
    result = _evaluate(
        reference,
        lateral_noise_std_m=np.full(4, 2.0e-3),
        axial_noise_std_m=np.full(4, 4.0e-3),
        surface=replace(SURFACE, max_target_error_p95_m=2.0e-2),
    )

    assert 0.005 < result.target_error_p95_m < 0.010
    assert set(result.reference_contours_m_to_status) == {0.005, 0.010}
    five_mm = result.reference_contours_m_to_status[0.005]
    ten_mm = result.reference_contours_m_to_status[0.010]
    assert five_mm.within_reference is False
    assert ten_mm.within_reference is True
    for contour_m, status in result.reference_contours_m_to_status.items():
        assert status.label == "reference_only"
        assert status.scope == "parameter_sensitivity_only"
        assert status.within_reference == (result.target_error_p95_m <= contour_m)
        assert np.isclose(status.margin_m, contour_m - result.target_error_p95_m)
        assert "pass" not in status.label.lower()


def test_low_confidence_outlier_is_robustly_downweighted_in_se3_fit() -> None:
    reference = np.vstack([_broad_four(), np.array([[0.0, 0.20, 1.0]])])
    observed = reference + np.array([0.015, -0.010, 0.020])
    observed[-1] += np.array([0.10, -0.08, 0.05])
    masks = np.array([0.95, 0.95, 0.95, 0.95, 0.01])
    surface = replace(SURFACE, max_residual_m=2.0e-2)

    result = _evaluate(
        reference,
        observed=observed,
        target=np.array([0.0, 0.0, 1.0]),
        mask_confidence=masks,
        surface=surface,
    )

    assert result.transform.shape == (4, 4)
    assert np.linalg.norm(result.transform[:3, 3] - np.array([0.015, -0.010, 0.020])) < 2e-3
    assert result.effective_support < 4.2


def test_anisotropic_gls_p95_is_calibrated_by_deterministic_monte_carlo() -> None:
    """Regression: scalar Kabsch reported ~2.2x less than its empirical p95 here."""

    reference = np.array(
        [
            [-0.12, -0.18, 1.0],
            [0.12, -0.18, 1.0],
            [-0.12, 0.18, 1.0],
            [0.12, 0.18, 1.0],
            [-0.12, 0.00, 1.0],
            [0.12, 0.00, 1.0],
        ]
    )
    angle = 0.08
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.03, -0.02, 0.04])
    noiseless_observed = (rotation @ reference.T).T + translation
    target = np.array([0.25, 0.05, 1.0])
    true_target = rotation @ target + translation
    lateral_std = 1.0e-3
    axial_std = 2.0e-2
    covariances = []
    for point in noiseless_observed:
        ray = point / np.linalg.norm(point)
        axial_projector = np.outer(ray, ray)
        covariances.append(
            lateral_std**2 * (np.eye(3) - axial_projector)
            + axial_std**2 * axial_projector
        )

    loose_surface = replace(
        SURFACE,
        min_relative_geometry_weight=1.0e-3,
        min_second_spatial_eigenvalue_m2=1.0e-8,
        min_hull_area_m2=1.0e-8,
        min_spread_m=1.0e-2,
        max_condition_number=1.0e8,
        max_residual_m=1.0,
        max_target_leverage=1.0e6,
        max_target_error_p95_m=1.0,
    )
    rng = np.random.RandomState(13)
    errors = []
    predicted_p95 = []
    for _ in range(160):
        observed = np.array(
            [
                point + rng.multivariate_normal(np.zeros(3), covariance)
                for point, covariance in zip(noiseless_observed, covariances)
            ]
        )
        result = evaluate_visibility_geometry(
            reference,
            observed,
            target,
            depth_m=observed[:, 2],
            mask_confidence=np.ones(len(reference)),
            lateral_noise_std_m=np.full(len(reference), lateral_std),
            axial_noise_std_m=np.full(len(reference), axial_std),
            robust_delta_m=0.2,
            decision_surface=loose_surface,
        )
        assert np.isfinite(result.target_error_p95_m)
        estimated_target = result.transform[:3, :3] @ target + result.transform[:3, 3]
        errors.append(np.linalg.norm(estimated_target - true_target))
        predicted_p95.append(result.target_error_p95_m)

    empirical_p95 = float(np.quantile(errors, 0.95))
    reported_p95 = float(np.median(predicted_p95))
    assert empirical_p95 <= 1.35 * reported_p95
    assert empirical_p95 >= 0.45 * reported_p95
