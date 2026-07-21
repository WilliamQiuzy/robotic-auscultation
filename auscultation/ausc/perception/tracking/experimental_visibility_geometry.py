"""Experimental geometry gate for occluded patient-relative targets.

This module deliberately lives outside the production tracker.  It evaluates a
declared *parameter sensitivity* surface; it is not a commissioned accuracy
claim.  Correspondence count is reported through effective support but is never
used as a pass criterion.  The fit is rigid SE(3); Sim(3) scale estimation is
not implemented and is always reported disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Tuple

import numpy as np


_CHI2_3D_95 = 7.814727903251179


@dataclass(frozen=True)
class DecisionSurface:
    """Explicit sensitivity thresholds, with metric units encoded in names."""

    min_relative_geometry_weight: float
    max_depth_consistency_sigma: float
    min_second_spatial_eigenvalue_m2: float
    min_hull_area_m2: float
    min_spread_m: float
    max_condition_number: float
    max_residual_m: float
    max_target_leverage: float
    max_target_error_p95_m: float


@dataclass(frozen=True)
class ReferenceContourStatus:
    """Non-commissioning comparison against a reporting reference contour."""

    within_reference: bool
    margin_m: float
    label: str = "reference_only"
    scope: str = "parameter_sensitivity_only"


@dataclass(frozen=True)
class VisibilityGeometryResult:
    """Quality evidence for one weighted robust SE(3) fit."""

    accepted: bool
    decision: str
    scope: str
    failure_reasons: Tuple[str, ...]
    effective_support: float
    spatial_eigenvalues: np.ndarray
    hull_area_m2: float
    spread_m: float
    condition_number: float
    residual_m: float
    target_leverage: float
    propagated_covariance: np.ndarray
    target_error_p95_m: float
    reference_contours_m_to_status: Mapping[float, ReferenceContourStatus]
    depth_residuals_m: np.ndarray
    depth_consistency_max_sigma: float
    depth_provenance: str
    transform: np.ndarray
    decision_surface: DecisionSurface
    sim3_scale_enabled: bool = False


def _skew(point: np.ndarray) -> np.ndarray:
    x, y, z = point
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _effective_support(weights: np.ndarray) -> float:
    scale = float(np.max(weights)) if len(weights) else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return 0.0
    scaled = weights / scale
    if not np.isfinite(scaled).all() or np.any(scaled < 0.0):
        return 0.0
    total = float(np.sum(scaled))
    squared = float(np.dot(scaled, scaled))
    if total <= 0.0 or squared <= 0.0:
        return 0.0
    return total * total / squared


def _weighted_rigid(
    source: np.ndarray,
    destination: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted Kabsch fit with fixed unit scale."""

    scale = float(np.max(weights)) if len(weights) else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return np.eye(3), np.zeros(3)
    scaled = weights / scale
    total = float(scaled.sum())
    normalized = scaled / total
    source_mean = np.sum(normalized[:, None] * source, axis=0)
    destination_mean = np.sum(normalized[:, None] * destination, axis=0)
    source_zero = source - source_mean
    destination_zero = destination - destination_mean
    cross_covariance = (normalized[:, None] * source_zero).T @ destination_zero
    if not np.isfinite(cross_covariance).all():
        raise np.linalg.LinAlgError("non-finite weighted cross covariance")
    left, _, right_t = np.linalg.svd(cross_covariance)
    correction = np.eye(3)
    if np.linalg.det(right_t.T @ left.T) < 0.0:
        correction[-1, -1] = -1.0
    rotation = right_t.T @ correction @ left.T
    translation = destination_mean - rotation @ source_mean
    return rotation, translation


def _so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Rodrigues exponential for a right-multiplicative rotation update."""

    angle = float(np.linalg.norm(rotation_vector))
    generator = _skew(rotation_vector)
    if angle < 1.0e-10:
        return np.eye(3) + generator + 0.5 * generator @ generator
    unit_generator = generator / angle
    return (
        np.eye(3)
        + np.sin(angle) * unit_generator
        + (1.0 - np.cos(angle)) * unit_generator @ unit_generator
    )


def _gls_system(
    source: np.ndarray,
    destination: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    masks: np.ndarray,
    inverse_covariances: np.ndarray,
    robust_delta_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normal equations and matching score covariance for one robust GLS step."""

    fitted = (rotation @ source.T).T + translation
    residual_vectors = destination - fitted
    residuals = np.linalg.norm(residual_vectors, axis=1)
    if not np.isfinite(residuals).all():
        raise np.linalg.LinAlgError("non-finite rigid residual")
    robust = np.ones_like(residuals)
    outside = residuals > robust_delta_m
    robust[outside] = robust_delta_m / residuals[outside]
    if not np.isfinite(robust).all() or np.any(robust <= 0.0):
        raise np.linalg.LinAlgError("non-finite robust weights")

    normal = np.zeros((6, 6), dtype=np.float64)
    score_covariance = np.zeros((6, 6), dtype=np.float64)
    right_hand_side = np.zeros(6, dtype=np.float64)
    for point, residual_vector, mask, robust_weight, inverse_covariance in zip(
        source,
        residual_vectors,
        masks,
        robust,
        inverse_covariances,
    ):
        if mask <= 0.0:
            continue
        jacobian = np.hstack((-rotation @ _skew(point), np.eye(3)))
        information = jacobian.T @ inverse_covariance @ jacobian
        precision_weight = mask * robust_weight
        normal += precision_weight * information
        # Treat mask confidence as a precision multiplier: C_effective = C / mask.
        score_covariance += mask * robust_weight**2 * information
        right_hand_side += (
            precision_weight * jacobian.T @ inverse_covariance @ residual_vector
        )
    if not (
        np.isfinite(normal).all()
        and np.isfinite(score_covariance).all()
        and np.isfinite(right_hand_side).all()
    ):
        raise np.linalg.LinAlgError("non-finite robust GLS system")
    return normal, score_covariance, right_hand_side, robust


def _robust_gls_fit(
    source: np.ndarray,
    destination: np.ndarray,
    masks: np.ndarray,
    inverse_covariances: np.ndarray,
    robust_delta_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Anisotropic iteratively reweighted SE(3) generalized least squares."""

    rotation, translation = _weighted_rigid(source, destination, masks)
    for _ in range(20):
        normal, _, right_hand_side, _ = _gls_system(
            source,
            destination,
            rotation,
            translation,
            masks,
            inverse_covariances,
            robust_delta_m,
        )
        eigenvalues = np.linalg.eigvalsh(0.5 * (normal + normal.T))
        if not np.isfinite(eigenvalues).all():
            raise np.linalg.LinAlgError("non-finite GLS information spectrum")
        rank_tolerance = max(float(eigenvalues[-1]), 1.0) * 1.0e-12
        if np.count_nonzero(eigenvalues > rank_tolerance) < 6:
            break
        increment = np.linalg.solve(normal, right_hand_side)
        if not np.isfinite(increment).all():
            raise np.linalg.LinAlgError("non-finite GLS increment")
        rotation = rotation @ _so3_exp(increment[:3])
        translation = translation + increment[3:]
        if float(np.linalg.norm(increment)) < 1.0e-10:
            break

    normal, score_covariance, _, robust = _gls_system(
        source,
        destination,
        rotation,
        translation,
        masks,
        inverse_covariances,
        robust_delta_m,
    )
    fitted = (rotation @ source.T).T + translation
    residuals = np.linalg.norm(destination - fitted, axis=1)
    return rotation, translation, robust, residuals, normal, score_covariance


def _spatial_metrics(
    points: np.ndarray,
    weights: np.ndarray,
    min_relative_geometry_weight: float,
) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
    total = float(weights.sum())
    if total <= 0.0:
        return np.zeros(3), 0.0, 0.0, float("inf"), np.zeros(3)
    normalized = weights / total
    center = np.sum(normalized[:, None] * points, axis=0)
    centered = points - center
    covariance = (normalized[:, None] * centered).T @ centered
    if not np.isfinite(covariance).all():
        raise np.linalg.LinAlgError("non-finite spatial covariance")
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    second = float(eigenvalues[-2])
    condition = float(eigenvalues[-1] / second) if second > 0.0 else float("inf")

    supported = points[weights >= min_relative_geometry_weight * float(weights.max())]
    spread = 0.0
    for first, second_point in combinations(supported, 2):
        spread = max(spread, float(np.linalg.norm(first - second_point)))
    hull_area = _planar_hull_area(supported)
    return eigenvalues, hull_area, spread, condition, center


def _planar_hull_area(points: np.ndarray) -> float:
    """Area of the support's 2-D convex hull in its best-fit spatial plane."""

    if len(points) < 3:
        return 0.0
    centered = points - np.mean(points, axis=0)
    if not np.isfinite(centered).all():
        raise np.linalg.LinAlgError("non-finite hull coordinates")
    _, _, right_t = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ right_t[:2].T
    ordered = sorted({(float(point[0]), float(point[1])) for point in projected})
    if len(ordered) < 3:
        return 0.0

    def turn(origin, first, second) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return 0.5 * abs(
        sum(
            point[0] * hull[(index + 1) % len(hull)][1]
            - hull[(index + 1) % len(hull)][0] * point[1]
            for index, point in enumerate(hull)
        )
    )


def _failed_result(
    surface: DecisionSurface,
    reasons: Tuple[str, ...],
    *,
    effective_support: float = 0.0,
    depth_residuals_m: np.ndarray | None = None,
    depth_consistency_max_sigma: float = float("inf"),
) -> VisibilityGeometryResult:
    target_p95 = float("inf")
    return VisibilityGeometryResult(
        accepted=False,
        decision="fail_closed",
        scope="parameter_sensitivity_only",
        failure_reasons=reasons,
        effective_support=effective_support,
        spatial_eigenvalues=np.full(3, np.nan),
        hull_area_m2=0.0,
        spread_m=0.0,
        condition_number=float("inf"),
        residual_m=float("inf"),
        target_leverage=float("inf"),
        propagated_covariance=np.full((3, 3), np.inf),
        target_error_p95_m=target_p95,
        reference_contours_m_to_status=_reference_contours(target_p95),
        depth_residuals_m=(
            np.asarray(depth_residuals_m, dtype=np.float64)
            if depth_residuals_m is not None
            else np.array([], dtype=np.float64)
        ),
        depth_consistency_max_sigma=depth_consistency_max_sigma,
        depth_provenance="explicit_depth_vs_observed_camera_z",
        transform=np.eye(4),
        decision_surface=surface,
    )


def _reference_contours(target_error_p95_m: float) -> Mapping[float, ReferenceContourStatus]:
    """Compare p95 to reporting contours without making a physical PASS claim."""

    return {
        contour_m: ReferenceContourStatus(
            within_reference=target_error_p95_m <= contour_m,
            margin_m=contour_m - target_error_p95_m,
        )
        for contour_m in (0.005, 0.010)
    }


def evaluate_visibility_geometry(
    reference_points_m: np.ndarray,
    observed_points_m: np.ndarray,
    target_reference_point_m: np.ndarray,
    *,
    depth_m: np.ndarray,
    mask_confidence: np.ndarray,
    lateral_noise_std_m: np.ndarray,
    axial_noise_std_m: np.ndarray,
    robust_delta_m: float,
    decision_surface: DecisionSurface,
) -> VisibilityGeometryResult:
    """Fit SE(3), propagate target uncertainty, and evaluate one decision surface.

    Depth, mask confidence, lateral noise, and axial noise are mandatory inputs;
    this experiment has no hidden camera-noise defaults.  A returned ``accepted``
    value means only that the caller's declared sensitivity thresholds were met.
    It must not be presented as a physical 5 mm validation or commissioning PASS.
    """

    reference = np.asarray(reference_points_m, dtype=np.float64)
    observed = np.asarray(observed_points_m, dtype=np.float64)
    target = np.asarray(target_reference_point_m, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    masks = np.asarray(mask_confidence, dtype=np.float64)
    lateral_std = np.asarray(lateral_noise_std_m, dtype=np.float64)
    axial_std = np.asarray(axial_noise_std_m, dtype=np.float64)

    if reference.ndim != 2 or reference.shape[1:] != (3,):
        raise ValueError("reference_points_m must have shape (N, 3)")
    count = len(reference)
    if observed.shape != reference.shape:
        raise ValueError("observed_points_m must match reference_points_m")
    if target.shape != (3,):
        raise ValueError("target_reference_point_m must have shape (3,)")
    for name, values in (
        ("depth_m", depth),
        ("mask_confidence", masks),
        ("lateral_noise_std_m", lateral_std),
        ("axial_noise_std_m", axial_std),
    ):
        if values.shape != (count,):
            raise ValueError(f"{name} must have shape (N,)")
    if not np.isfinite(robust_delta_m) or robust_delta_m <= 0.0:
        raise ValueError("robust_delta_m must be finite and positive")

    threshold_values = np.array(
        [
            decision_surface.max_depth_consistency_sigma,
            decision_surface.min_second_spatial_eigenvalue_m2,
            decision_surface.min_relative_geometry_weight,
            decision_surface.min_hull_area_m2,
            decision_surface.min_spread_m,
            decision_surface.max_condition_number,
            decision_surface.max_residual_m,
            decision_surface.max_target_leverage,
            decision_surface.max_target_error_p95_m,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(threshold_values).all() or np.any(threshold_values <= 0.0):
        raise ValueError("decision_surface values must be finite and positive")
    if decision_surface.min_relative_geometry_weight > 1.0:
        raise ValueError("min_relative_geometry_weight must not exceed one")

    if not (
        np.isfinite(reference).all()
        and np.isfinite(observed).all()
        and np.isfinite(target).all()
        and np.isfinite(depth).all()
    ):
        return _failed_result(decision_surface, ("non_finite_input",))
    if not (
        np.isfinite(masks).all()
        and np.isfinite(lateral_std).all()
        and np.isfinite(axial_std).all()
        and np.all(lateral_std > 0.0)
        and np.all(axial_std > 0.0)
    ):
        return _failed_result(decision_surface, ("non_finite_covariance",))
    if np.any(depth <= 0.0) or np.any(masks < 0.0) or np.any(masks > 1.0):
        return _failed_result(decision_surface, ("invalid_depth_or_mask",))

    depth_residuals = depth - observed[:, 2]
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        lateral_variance = np.square(lateral_std)
        axial_variance = np.square(axial_std)
        scalar_variance = (2.0 * lateral_variance + axial_variance) / 3.0
    if not (
        np.isfinite(lateral_variance).all()
        and np.isfinite(axial_variance).all()
        and np.isfinite(scalar_variance).all()
        and np.all(lateral_variance > 0.0)
        and np.all(axial_variance > 0.0)
        and np.all(scalar_variance > 0.0)
    ):
        return _failed_result(
            decision_surface,
            ("non_finite_covariance",),
            depth_residuals_m=depth_residuals,
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        depth_consistency_sigma = np.abs(depth_residuals) / axial_std
    if not np.isfinite(depth_consistency_sigma).all():
        return _failed_result(
            decision_surface,
            ("non_finite_covariance",),
            depth_residuals_m=depth_residuals,
        )
    depth_consistency_max_sigma = (
        float(np.max(depth_consistency_sigma)) if count else 0.0
    )
    if depth_consistency_max_sigma > decision_surface.max_depth_consistency_sigma:
        return _failed_result(
            decision_surface,
            ("depth_observation_mismatch",),
            depth_residuals_m=depth_residuals,
            depth_consistency_max_sigma=depth_consistency_max_sigma,
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_geometry_precision = masks / scalar_variance
    positive_mask = masks > 0.0
    if not (
        np.isfinite(raw_geometry_precision).all()
        and np.all(raw_geometry_precision >= 0.0)
        and np.all(raw_geometry_precision[positive_mask] > 0.0)
    ):
        return _failed_result(
            decision_surface,
            ("non_finite_covariance",),
            depth_residuals_m=depth_residuals,
            depth_consistency_max_sigma=depth_consistency_max_sigma,
        )
    precision_scale = float(np.max(raw_geometry_precision)) if count else 0.0
    if not np.isfinite(precision_scale) or precision_scale <= 0.0:
        return _failed_result(
            decision_surface,
            ("non_finite_covariance",),
            depth_residuals_m=depth_residuals,
            depth_consistency_max_sigma=depth_consistency_max_sigma,
        )
    geometry_precision = raw_geometry_precision / precision_scale
    if not np.isfinite(geometry_precision).all():
        return _failed_result(
            decision_surface,
            ("non_finite_covariance",),
            depth_residuals_m=depth_residuals,
            depth_consistency_max_sigma=depth_consistency_max_sigma,
        )
    if count < 3 or np.count_nonzero(geometry_precision) < 3:
        support = _effective_support(geometry_precision)
        return _failed_result(
            decision_surface,
            ("degenerate_spatial_rank", "rank_deficient_information"),
            effective_support=support,
            depth_residuals_m=depth_residuals,
            depth_consistency_max_sigma=depth_consistency_max_sigma,
        )

    try:
        inverse_covariances = []
        for observation, depth_value, lateral_var, axial_var in zip(
            observed,
            depth,
            lateral_variance,
            axial_variance,
        ):
            ray = np.array([observation[0], observation[1], depth_value])
            ray_norm = float(np.linalg.norm(ray))
            if not np.isfinite(ray_norm) or ray_norm <= 0.0:
                raise np.linalg.LinAlgError("invalid depth ray")
            ray = ray / ray_norm
            axial_projector = np.outer(ray, ray)
            measurement_covariance = (
                lateral_var * (np.eye(3) - axial_projector)
                + axial_var * axial_projector
            )
            if not np.isfinite(measurement_covariance).all():
                raise np.linalg.LinAlgError("non-finite measurement covariance")
            covariance_eigenvalues = np.linalg.eigvalsh(measurement_covariance)
            if not np.isfinite(covariance_eigenvalues).all() or np.any(
                covariance_eigenvalues <= 0.0
            ):
                raise np.linalg.LinAlgError("non-positive measurement covariance")
            inverse_measurement_covariance = np.linalg.inv(measurement_covariance)
            if not np.isfinite(inverse_measurement_covariance).all():
                raise np.linalg.LinAlgError("non-finite measurement precision")
            inverse_covariances.append(inverse_measurement_covariance)
        inverse_covariances_array = np.asarray(inverse_covariances)

        (
            rotation,
            translation,
            robust_weights,
            point_residuals,
            information,
            score_covariance,
        ) = _robust_gls_fit(
            reference,
            observed,
            masks,
            inverse_covariances_array,
            robust_delta_m,
        )
        final_geometry_precision = geometry_precision * robust_weights
        if not (
            np.isfinite(rotation).all()
            and np.isfinite(translation).all()
            and np.isfinite(final_geometry_precision).all()
            and np.isfinite(point_residuals).all()
            and np.all(final_geometry_precision >= 0.0)
            and np.all(final_geometry_precision[positive_mask] > 0.0)
        ):
            raise np.linalg.LinAlgError("invalid robust SE3 result")
        support = _effective_support(final_geometry_precision)
        eigenvalues, hull_area, spread, condition, center = _spatial_metrics(
            reference,
            final_geometry_precision,
            decision_surface.min_relative_geometry_weight,
        )
        residual = float(
            np.sqrt(
                np.sum(geometry_precision * point_residuals**2)
                / np.sum(geometry_precision)
            )
        )
        if not np.isfinite(residual):
            raise np.linalg.LinAlgError("non-finite residual")

        normalized = final_geometry_precision / np.sum(final_geometry_precision)
        centered = reference - center
        covariance_geometry = (normalized[:, None] * centered).T @ centered
        if not np.isfinite(covariance_geometry).all():
            raise np.linalg.LinAlgError("non-finite geometry covariance")
        geometry_pseudoinverse = np.linalg.pinv(covariance_geometry, rcond=1.0e-12)
        if not np.isfinite(geometry_pseudoinverse).all():
            raise np.linalg.LinAlgError("non-finite geometry pseudoinverse")
        target_offset = target - center
        leverage_squared = float(
            1.0 / max(support, 1.0e-12)
            + target_offset @ geometry_pseudoinverse @ target_offset
            / max(support, 1.0e-12)
        )
        if not np.isfinite(leverage_squared):
            raise np.linalg.LinAlgError("non-finite target leverage")
        leverage = float(np.sqrt(max(leverage_squared, 0.0)))

        information = 0.5 * (information + information.T)
        score_covariance = 0.5 * (score_covariance + score_covariance.T)
        if not (
            np.isfinite(information).all()
            and np.isfinite(score_covariance).all()
        ):
            raise np.linalg.LinAlgError("non-finite GLS covariance system")
        information_eigenvalues = np.linalg.eigvalsh(information)
        if not np.isfinite(information_eigenvalues).all() or np.any(
            information_eigenvalues < -1.0e-8
        ):
            raise np.linalg.LinAlgError("invalid information spectrum")
        rank_tolerance = max(float(information_eigenvalues[-1]), 1.0) * 1e-12
        info_rank = int(np.count_nonzero(information_eigenvalues > rank_tolerance))
        covariance_ok = info_rank == 6
        if covariance_ok:
            inverse_information = np.linalg.inv(information)
            pose_covariance = (
                inverse_information @ score_covariance @ inverse_information
            )
            pose_covariance = 0.5 * (pose_covariance + pose_covariance.T)
            if not np.isfinite(pose_covariance).all():
                raise np.linalg.LinAlgError("non-finite pose covariance")
            pose_covariance_eigenvalues = np.linalg.eigvalsh(pose_covariance)
            if not np.isfinite(pose_covariance_eigenvalues).all() or np.any(
                pose_covariance_eigenvalues < -1.0e-12
            ):
                raise np.linalg.LinAlgError("invalid sandwich covariance spectrum")
            target_jacobian = np.hstack((-rotation @ _skew(target), np.eye(3)))
            propagated = target_jacobian @ pose_covariance @ target_jacobian.T
            propagated = 0.5 * (propagated + propagated.T)
            if not np.isfinite(propagated).all():
                raise np.linalg.LinAlgError("non-finite propagated covariance")
        else:
            propagated = np.full((3, 3), np.inf)

        if covariance_ok:
            covariance_eigenvalues = np.linalg.eigvalsh(propagated)
            covariance_ok = bool(
                np.isfinite(covariance_eigenvalues).all()
                and np.all(covariance_eigenvalues >= -1.0e-12)
            )
            if covariance_ok:
                target_p95 = float(
                    np.sqrt(_CHI2_3D_95 * max(float(covariance_eigenvalues[-1]), 0.0))
                )
            else:
                target_p95 = float("inf")
        else:
            target_p95 = float("inf")
    except np.linalg.LinAlgError:
        return _failed_result(
            decision_surface,
            ("non_finite_covariance",),
            depth_residuals_m=depth_residuals,
            depth_consistency_max_sigma=depth_consistency_max_sigma,
        )

    reasons = []
    if eigenvalues[-2] < decision_surface.min_second_spatial_eigenvalue_m2:
        reasons.append("degenerate_spatial_rank")
    if hull_area < decision_surface.min_hull_area_m2:
        reasons.append("hull_area_below_surface")
    if spread < decision_surface.min_spread_m:
        reasons.append("spread_below_surface")
    if condition > decision_surface.max_condition_number:
        reasons.append("ill_conditioned_geometry")
    if residual > decision_surface.max_residual_m:
        reasons.append("residual_above_surface")
    if leverage > decision_surface.max_target_leverage:
        reasons.append("excessive_target_leverage")
    if info_rank < 6:
        reasons.append("rank_deficient_information")
    if not covariance_ok:
        reasons.append("non_finite_covariance")
    if target_p95 > decision_surface.max_target_error_p95_m:
        reasons.append("target_uncertainty_above_surface")

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    accepted = not reasons
    return VisibilityGeometryResult(
        accepted=accepted,
        decision=("pass_on_declared_sensitivity_surface" if accepted else "fail_closed"),
        scope="parameter_sensitivity_only",
        failure_reasons=tuple(reasons),
        effective_support=support,
        spatial_eigenvalues=eigenvalues,
        hull_area_m2=hull_area,
        spread_m=spread,
        condition_number=condition,
        residual_m=residual,
        target_leverage=leverage,
        propagated_covariance=propagated,
        target_error_p95_m=target_p95,
        reference_contours_m_to_status=_reference_contours(target_p95),
        depth_residuals_m=depth_residuals,
        depth_consistency_max_sigma=depth_consistency_max_sigma,
        depth_provenance="explicit_depth_vs_observed_camera_z",
        transform=transform,
        decision_surface=decision_surface,
    )
