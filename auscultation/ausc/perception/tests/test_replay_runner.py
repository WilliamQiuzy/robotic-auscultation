"""Contracts for the framework-free model-replay runner core."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ausc.perception.replay_attestation import (
    COMPARISON_BIT_EXACT,
    COMPARISON_FROZEN_EPSILON,
    ReplayAttestationError,
    verify_replay_attestation,
)
from ausc.perception.replay_runner import (
    ReplayItem,
    ReplayRunnerError,
    compare_outputs,
    compute_bound_artifacts,
    run_replay,
)

_MODEL_SHA = "a" * 64
_MODEL_RELEASE = "perception-" + _MODEL_SHA[:20]
_PRODUCER = {
    "code_release_id": "code_release-" + "b" * 20,
    "module_sha256": "c" * 64,
    "activation_sha256": "d" * 64,
}


def test_compare_bit_exact_match_has_zero_residual():
    rec = {"points_m": np.ones((3, 3), dtype=np.float32)}
    matched, residuals = compare_outputs(
        rec, {"points_m": np.ones((3, 3), dtype=np.float32)},
        mode=COMPARISON_BIT_EXACT, epsilon={},
    )
    assert matched is True
    assert residuals == {"points_m": 0.0}


def test_compare_bit_exact_mismatch_reports_residual():
    rec = {"points_m": np.zeros((2,), dtype=np.float32)}
    rep = {"points_m": np.array([0.0, 0.25], dtype=np.float32)}
    matched, residuals = compare_outputs(rec, rep, mode=COMPARISON_BIT_EXACT, epsilon={})
    assert matched is False
    assert residuals["points_m"] == pytest.approx(0.25)


def test_compare_epsilon_within_tolerance_matches():
    rec = {"q": np.array([0.5], dtype=np.float64)}
    rep = {"q": np.array([0.5 + 1e-6], dtype=np.float64)}
    matched, residuals = compare_outputs(
        rec, rep, mode=COMPARISON_FROZEN_EPSILON, epsilon={"q": 1e-5},
    )
    assert matched is True
    assert residuals["q"] == pytest.approx(1e-6)


def test_compare_epsilon_exceeded_does_not_match():
    rec = {"q": np.array([0.5])}
    rep = {"q": np.array([0.5 + 1e-3])}
    matched, _ = compare_outputs(
        rec, rep, mode=COMPARISON_FROZEN_EPSILON, epsilon={"q": 1e-5},
    )
    assert matched is False


def test_compare_shape_mismatch_is_a_hard_error():
    with pytest.raises(ReplayRunnerError, match="shape"):
        compare_outputs(
            {"p": np.zeros((3,))}, {"p": np.zeros((4,))},
            mode=COMPARISON_BIT_EXACT, epsilon={},
        )


def test_compare_output_name_mismatch_is_rejected():
    with pytest.raises(ReplayRunnerError, match="do not match"):
        compare_outputs(
            {"a": np.zeros((1,))}, {"b": np.zeros((1,))},
            mode=COMPARISON_BIT_EXACT, epsilon={},
        )


def _geometry_session(tmp_path: Path) -> Path:
    session = tmp_path / "session-geo-001"
    session.mkdir()
    (session / "geometry_frame_packets.jsonl").write_bytes(b'{"frame":1}\n{"frame":2}\n')
    ledger = session / "geometry" / "model_input_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(b'{"sequence":1}\n{"sequence":2}\n')
    return session


def test_compute_bound_artifacts_hashes_the_required_set(tmp_path):
    session = _geometry_session(tmp_path)
    bound = compute_bound_artifacts(session, "geometry")
    assert set(bound) == {
        "geometry_frame_packets.jsonl",
        "geometry/model_input_ledger.jsonl",
    }
    assert all(len(sha) == 64 for sha in bound.values())


def test_compute_bound_artifacts_fails_when_required_file_missing(tmp_path):
    session = tmp_path / "session-geo-002"
    session.mkdir()
    (session / "geometry_frame_packets.jsonl").write_bytes(b"x\n")
    with pytest.raises(ReplayRunnerError, match="missing"):
        compute_bound_artifacts(session, "geometry")


def _geometry_items(n: int) -> list[ReplayItem]:
    items = []
    for i in range(n):
        recorded = {
            "points_m": np.full((5, 3), float(i), dtype=np.float32),
            "covariances_m2": np.eye(3, dtype=np.float32) * (i + 1),
        }
        items.append(ReplayItem(model_input={"frame": i}, recorded_outputs=recorded))
    return items


def test_run_replay_all_matched_round_trips_through_the_verifier(tmp_path):
    session = _geometry_session(tmp_path)
    items = _geometry_items(2)

    # A perfect model reproduces the recorded outputs exactly.
    def model_run(model_input):
        i = model_input["frame"]
        return {
            "points_m": np.full((5, 3), float(i), dtype=np.float32),
            "covariances_m2": np.eye(3, dtype=np.float32) * (i + 1),
        }

    manifest = run_replay(
        session, items, model_run,
        component="geometry",
        model_release_id=_MODEL_RELEASE, model_sha256=_MODEL_SHA,
        producer=_PRODUCER,
    )
    assert manifest["item_count"] == 2
    assert manifest["matched_count"] == 2
    assert manifest["comparison_mode"] == COMPARISON_BIT_EXACT

    # The finalizer verifier accepts exactly what the runner produced.
    evidence = verify_replay_attestation(
        session, manifest, component="geometry",
        recorded_identity={"release_id": _MODEL_RELEASE, "model_sha256": _MODEL_SHA},
        recorded_item_count=2,
    )
    assert evidence["production_authorized"] is True


def test_run_replay_unmatched_item_fails_closed_at_the_verifier(tmp_path):
    session = _geometry_session(tmp_path)
    items = _geometry_items(2)

    def broken_model(model_input):
        i = model_input["frame"]
        points = np.full((5, 3), float(i), dtype=np.float32)
        if i == 1:  # perturb the second frame so it cannot match bit-exactly
            points[0, 0] += 0.5
        return {
            "points_m": points,
            "covariances_m2": np.eye(3, dtype=np.float32) * (i + 1),
        }

    manifest = run_replay(
        session, items, broken_model,
        component="geometry",
        model_release_id=_MODEL_RELEASE, model_sha256=_MODEL_SHA,
        producer=_PRODUCER,
    )
    assert manifest["matched_count"] == 1  # honest: one frame did not reproduce
    assert manifest["item_count"] == 2
    # The finalizer refuses partial coverage.
    with pytest.raises(ReplayAttestationError, match="did not match"):
        verify_replay_attestation(
            session, manifest, component="geometry",
            recorded_identity={"release_id": _MODEL_RELEASE, "model_sha256": _MODEL_SHA},
            recorded_item_count=2,
        )


def test_run_replay_bit_exact_rejects_supplied_epsilon(tmp_path):
    session = _geometry_session(tmp_path)
    with pytest.raises(ReplayRunnerError, match="must not supply an epsilon"):
        run_replay(
            session, _geometry_items(1), lambda _i: {},
            component="geometry",
            model_release_id=_MODEL_RELEASE, model_sha256=_MODEL_SHA,
            producer=_PRODUCER,
            comparison_mode=COMPARISON_BIT_EXACT, epsilon={"points_m": 1e-6},
        )


def test_run_replay_empty_items_is_refused(tmp_path):
    session = _geometry_session(tmp_path)
    with pytest.raises(ReplayRunnerError, match="no replay items"):
        run_replay(
            session, [], lambda _i: {},
            component="geometry",
            model_release_id=_MODEL_RELEASE, model_sha256=_MODEL_SHA,
            producer=_PRODUCER,
        )


def test_run_replay_frozen_epsilon_within_a_calibrated_ceiling(tmp_path):
    session = _geometry_session(tmp_path)
    items = _geometry_items(2)

    def jittered_model(model_input):
        i = model_input["frame"]
        return {
            "points_m": np.full((5, 3), float(i), dtype=np.float32) + 1e-6,
            "covariances_m2": np.eye(3, dtype=np.float32) * (i + 1),
        }

    manifest = run_replay(
        session, items, jittered_model,
        component="geometry",
        model_release_id=_MODEL_RELEASE, model_sha256=_MODEL_SHA,
        producer=_PRODUCER,
        comparison_mode=COMPARISON_FROZEN_EPSILON,
        epsilon={"points_m": 1e-4, "covariances_m2": 1e-4},
    )
    assert manifest["matched_count"] == 2
    assert manifest["max_abs_residual"]["points_m"] == pytest.approx(1e-6, abs=1e-7)

    ceilings = {"geometry": {"points_m": 1e-3, "covariances_m2": 1e-3}}
    evidence = verify_replay_attestation(
        session, manifest, component="geometry",
        recorded_identity={"release_id": _MODEL_RELEASE, "model_sha256": _MODEL_SHA},
        recorded_item_count=2, epsilon_ceilings=ceilings,
    )
    assert evidence["production_authorized"] is True
