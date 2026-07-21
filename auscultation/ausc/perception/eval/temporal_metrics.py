"""Pure temporal metrics for tracked 3D point sequences."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


@dataclass(frozen=True)
class JitterMetrics:
    """Common-motion-removed frame-to-frame displacement statistics."""

    median_mm: float
    p90_mm: float
    sample_count: int


@dataclass(frozen=True)
class LagMetrics:
    """Bounded centroid lag, where positive values mean candidate delay."""

    axis_lags_frames: tuple[int, int, int]
    median_frames: int
    sample_count: int
    max_lag_frames: int


def _point_sequence(points_m: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"{name} must have shape (T, N, 3)")
    if points.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two frames")
    return points


def common_removed_jitter_mm(points_m: np.ndarray) -> JitterMetrics:
    """Measure per-point frame motion after removing median common translation."""

    points = _point_sequence(points_m, name="points_m")
    delta = np.diff(points, axis=0)
    complete = np.all(np.isfinite(delta), axis=2)
    if not np.any(complete):
        raise ValueError("points_m does not produce any finite jitter samples")
    complete_delta = np.where(complete[:, :, None], delta, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        common = np.nanmedian(complete_delta, axis=1, keepdims=True)
    samples_mm = np.linalg.norm(complete_delta - common, axis=2) * 1000.0
    finite = samples_mm[complete & np.isfinite(samples_mm)]
    return JitterMetrics(
        median_mm=float(np.median(finite)),
        p90_mm=float(np.percentile(finite, 90)),
        sample_count=int(finite.size),
    )


def _finite_centroids(points: np.ndarray) -> np.ndarray:
    complete_points = np.all(np.isfinite(points), axis=2)
    point_count = np.count_nonzero(complete_points, axis=1)
    centroids = np.full((points.shape[0], 3), np.nan, dtype=np.float64)
    finite_rows = point_count > 0
    if np.any(finite_rows):
        summed = np.sum(
            np.where(complete_points[:, :, None], points, 0.0), axis=1,
        )
        centroids[finite_rows] = (
            summed[finite_rows] / point_count[finite_rows, None]
        )
    return centroids


def _aligned_for_lag(
    reference: np.ndarray, candidate: np.ndarray, lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return reference[:-lag], candidate[lag:]
    if lag < 0:
        return reference[-lag:], candidate[:lag]
    return reference, candidate


def _axis_lag(
    reference: np.ndarray,
    candidate: np.ndarray,
    max_lag_frames: int,
    minimum_overlap: int,
) -> tuple[int, int]:
    scored: list[tuple[float, int, int]] = []
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        aligned_reference, aligned_candidate = _aligned_for_lag(
            reference, candidate, lag,
        )
        finite = np.isfinite(aligned_reference) & np.isfinite(aligned_candidate)
        sample_count = int(np.count_nonzero(finite))
        if sample_count < minimum_overlap:
            continue
        ref = aligned_reference[finite]
        cand = aligned_candidate[finite]
        ref = ref - np.mean(ref)
        cand = cand - np.mean(cand)
        denominator = float(np.linalg.norm(ref) * np.linalg.norm(cand))
        correlation = (
            0.0 if denominator == 0.0
            else float(np.dot(ref, cand) / denominator)
        )
        scored.append((correlation, lag, sample_count))
    if not scored:
        raise ValueError("inputs do not provide finite centroid overlap")
    _, lag, sample_count = max(
        scored,
        key=lambda item: (item[0], -abs(item[1]), -item[1]),
    )
    return lag, sample_count


def centroid_lag_frames(
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
    *,
    max_lag_frames: int,
) -> LagMetrics:
    """Estimate per-axis lag in a bounded, deterministic cross-correlation search.

    Positive lag means ``candidate_points`` follows ``reference_points``.
    """

    if type(max_lag_frames) is not int or max_lag_frames < 0:
        raise ValueError("max_lag_frames must be a non-negative integer")
    reference = _point_sequence(reference_points, name="reference_points")
    candidate = _point_sequence(candidate_points, name="candidate_points")
    if reference.shape != candidate.shape:
        raise ValueError("reference_points and candidate_points must have the same shape")
    if max_lag_frames > reference.shape[0] - 2:
        raise ValueError("max_lag_frames must be at most T - 2")

    reference_centroid = _finite_centroids(reference)
    candidate_centroid = _finite_centroids(candidate)
    minimum_overlap = max(2, reference.shape[0] - max_lag_frames)
    axis_results = tuple(
        _axis_lag(
            reference_centroid[:, axis],
            candidate_centroid[:, axis],
            max_lag_frames,
            minimum_overlap,
        )
        for axis in range(3)
    )
    axis_lags = tuple(result[0] for result in axis_results)
    return LagMetrics(
        axis_lags_frames=axis_lags,
        median_frames=int(np.median(axis_lags)),
        sample_count=min(result[1] for result in axis_results),
        max_lag_frames=max_lag_frames,
    )
