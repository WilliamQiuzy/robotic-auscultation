"""Tests for import-safe temporal point-cloud metrics."""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from ausc.perception.eval.temporal_metrics import (  # noqa: E402
    centroid_lag_frames as _centroid_lag_frames,
    common_removed_jitter_mm as _common_removed_jitter_mm,
)


def test_common_removed_jitter_differences_consecutive_frames() -> None:
    points = np.zeros((3, 2, 3), dtype=np.float64)
    points[:, 0, 0] = (0.0, 0.001, 0.003)

    result = _common_removed_jitter_mm(points)

    assert result.sample_count == 4
    assert result.median_mm == pytest.approx(0.75)
    assert result.p90_mm == pytest.approx(1.0)


def test_common_removed_jitter_removes_pure_common_translation() -> None:
    formation = np.array(
        [[-0.1, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.2, 1.0]],
        dtype=np.float64,
    )
    translations = np.array(
        [[0.0, 0.0, 0.0], [0.003, -0.002, 0.001], [0.008, 0.001, -0.004]],
        dtype=np.float64,
    )
    points = formation[None, :, :] + translations[:, None, :]

    result = _common_removed_jitter_mm(points)

    assert result.sample_count == 6
    assert result.median_mm == pytest.approx(0.0, abs=1e-12)
    assert result.p90_mm == pytest.approx(0.0, abs=1e-12)


def test_common_removed_jitter_uses_only_finite_samples() -> None:
    points = np.zeros((3, 3, 3), dtype=np.float64)
    points[:, 0, 0] = (0.0, 0.001, 0.003)
    points[:, 1, 0] = (0.0, 0.001, 0.002)
    points[:, 2, :] = np.nan

    result = _common_removed_jitter_mm(points)

    assert result.sample_count == 4
    assert result.median_mm == pytest.approx(0.25)


def test_common_removed_jitter_excludes_partial_nan_displacements() -> None:
    points = np.zeros((2, 4, 3), dtype=np.float64)
    points[1, :, 0] = (0.0, 0.002, 0.004, 0.100)
    points[1, 3, 1] = np.nan

    result = _common_removed_jitter_mm(points)

    assert result.sample_count == 3
    assert result.median_mm == pytest.approx(2.0)
    assert result.p90_mm == pytest.approx(2.0)


@pytest.mark.parametrize("nonfinite", [np.inf, -np.inf])
def test_common_removed_jitter_excludes_infinite_displacements(
    nonfinite: float,
) -> None:
    points = np.zeros((2, 4, 3), dtype=np.float64)
    points[1, :, 0] = (0.0, 0.002, 0.004, nonfinite)

    result = _common_removed_jitter_mm(points)

    assert result.sample_count == 3
    assert result.median_mm == pytest.approx(2.0)
    assert result.p90_mm == pytest.approx(2.0)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((3, 3), dtype=np.float64),
        np.zeros((3, 2, 2), dtype=np.float64),
    ],
)
def test_common_removed_jitter_rejects_wrong_shape(points: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"\(T, N, 3\)"):
        _common_removed_jitter_mm(points)


def test_common_removed_jitter_rejects_too_few_frames() -> None:
    with pytest.raises(ValueError, match="at least two frames"):
        _common_removed_jitter_mm(np.zeros((1, 2, 3), dtype=np.float64))


def test_common_removed_jitter_rejects_all_nonfinite_samples() -> None:
    with pytest.raises(ValueError, match="finite jitter"):
        _common_removed_jitter_mm(np.full((2, 2, 3), np.nan, dtype=np.float64))


def test_jitter_metrics_are_immutable() -> None:
    result = _common_removed_jitter_mm(np.zeros((2, 2, 3), dtype=np.float64))

    with pytest.raises(FrozenInstanceError):
        result.sample_count = 0


def test_centroid_lag_reports_positive_when_candidate_is_delayed() -> None:
    motion = np.array(
        [
            [0.0, 3.0, -1.0],
            [2.0, -2.0, 4.0],
            [-1.0, 5.0, 2.0],
            [4.0, 1.0, -3.0],
            [1.0, -4.0, 5.0],
            [-3.0, 2.0, 1.0],
            [5.0, 4.0, -2.0],
            [0.5, -1.0, 3.0],
            [-2.0, 0.5, 0.0],
            [3.0, -3.0, 2.5],
        ],
        dtype=np.float64,
    )
    formation = np.array([[-0.1, 0.0, 1.0], [0.1, 0.0, 1.0]])
    reference = formation[None, :, :] + motion[:, None, :] * 0.001
    candidate = np.full_like(reference, np.nan)
    candidate[2:] = reference[:-2] + np.array([0.2, -0.1, 0.05])
    candidate[:, 1, :] = np.nan  # finite centroid rows still come from point zero

    result = _centroid_lag_frames(
        reference, candidate, max_lag_frames=3,
    )

    assert result.axis_lags_frames == (2, 2, 2)
    assert result.median_frames == 2
    assert result.sample_count == 8
    assert result.max_lag_frames == 3


def test_centroid_lag_recovers_smooth_quadratic_delay_without_overlap_bias() -> None:
    time = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    motion = np.stack(
        (time**2, 2.0 * time**2 + 0.25, -0.5 * time**2 + 1.0), axis=1,
    )
    formation = np.array([[-0.1, 0.0, 1.0], [0.1, 0.0, 1.0]])
    reference = formation[None, :, :] + motion[:, None, :]
    candidate = np.empty_like(reference)
    candidate[:10] = reference[0]
    candidate[10:] = reference[:-10]

    result = _centroid_lag_frames(
        reference, candidate, max_lag_frames=10,
    )

    assert result.axis_lags_frames == (10, 10, 10)
    assert result.median_frames == 10
    assert result.sample_count == 90


def test_centroid_lag_ties_prefer_smallest_absolute_lag() -> None:
    points = np.ones((5, 2, 3), dtype=np.float64)

    result = _centroid_lag_frames(points, points, max_lag_frames=3)

    assert result.axis_lags_frames == (0, 0, 0)
    assert result.median_frames == 0


def test_centroid_lag_equal_absolute_tie_prefers_smaller_lag() -> None:
    reference_motion = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    candidate_motion = 1.0 - reference_motion
    reference = np.repeat(reference_motion[:, None, None], 3, axis=2)
    candidate = np.repeat(candidate_motion[:, None, None], 3, axis=2)

    result = _centroid_lag_frames(
        reference, candidate, max_lag_frames=1,
    )

    assert result.axis_lags_frames == (-1, -1, -1)


def test_centroid_lag_rejects_invalid_inputs() -> None:
    points = np.zeros((3, 2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="same shape"):
        _centroid_lag_frames(points, points[:, :1], max_lag_frames=1)
    with pytest.raises(ValueError, match="at least two frames"):
        _centroid_lag_frames(points[:1], points[:1], max_lag_frames=1)
    with pytest.raises(ValueError, match="non-negative integer"):
        _centroid_lag_frames(points, points, max_lag_frames=-1)
    with pytest.raises(ValueError, match="at most T - 2"):
        _centroid_lag_frames(points, points, max_lag_frames=2)
    with pytest.raises(ValueError, match="finite centroid overlap"):
        _centroid_lag_frames(
            np.full_like(points, np.nan), points, max_lag_frames=1,
        )


def test_lag_metrics_are_immutable() -> None:
    points = np.zeros((3, 2, 3), dtype=np.float64)
    result = _centroid_lag_frames(points, points, max_lag_frames=1)

    with pytest.raises(FrozenInstanceError):
        result.median_frames = 2


def test_model_effect_script_uses_shared_temporal_metrics() -> None:
    script = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "perception"
        / "eval_model_effect.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "ausc.perception.eval.temporal_metrics"
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {"common_removed_jitter_mm", "centroid_lag_frames"} <= imported
    assert {"common_removed_jitter_mm", "centroid_lag_frames"} <= called
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "jit"
        for node in ast.walk(tree)
    )

    fallback_handlers = [
        handler
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if isinstance(handler.type, ast.Name)
        and handler.type.id == "ModuleNotFoundError"
    ]
    assert len(fallback_handlers) == 1
    handler = fallback_handlers[0]
    assert handler.name == "exc"
    guards = [node for node in handler.body if isinstance(node, ast.If)]
    assert len(guards) == 1
    assert ast.unparse(guards[0].test) == "exc.name != 'ausc'"
    assert any(isinstance(node, ast.Raise) for node in guards[0].body)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "diff"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == 0
        for node in ast.walk(tree)
    )
