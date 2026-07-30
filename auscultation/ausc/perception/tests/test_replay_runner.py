"""Contracts for the framework-free model-replay runner core."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ausc.perception.replay_attestation import (
    COMPARISON_BIT_EXACT,
    COMPARISON_FROZEN_EPSILON,
    ReplayAttestationError,
    verify_replay_attestation,
)
from ausc.perception import replay_runner
from ausc.perception.replay_runner import (
    ReplayItem,
    ReplayRunnerError,
    compare_outputs,
    compute_bound_artifacts,
    reconstruct_geometry_items,
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


class _FakeRecord:
    """Stand-in for a ledger PersistedGeometryModelInput (reload is verified)."""

    def __init__(self, i: int) -> None:
        self.sequence = i + 1
        self.frame_index = i
        self.rgb_sha256 = f"{i:064x}"
        self.depth_sha256 = f"{i + 100:064x}"
        self.depth_scale = 0.001
        self.camera_t = 1.0 + i

    def reload_for_inference(self):
        return SimpleNamespace(
            rgb=np.full((4, 4, 3), self.frame_index, dtype=np.uint8),
            depth=np.full((4, 4), self.frame_index, dtype=np.uint16),
            back_mask=None,
            robot_mask=None,
        )


class _FakePacket:
    def __init__(self, i: int) -> None:
        self.sequence = i + 1
        self.frame_index = i
        self.input_rgb_sha256 = f"{i:064x}"
        self.input_depth_sha256 = f"{i + 100:064x}"
        self.point_names = ("R1", "R2")
        self.points_m = np.full((2, 3), float(i), dtype=np.float32)
        self.covariances_m2 = np.eye(3, dtype=np.float32) * (i + 1)


def _patch_geometry_sources(monkeypatch, n: int, *, release, model_sha, packet_mutator=None):
    from types import SimpleNamespace as NS
    records = tuple(_FakeRecord(i) for i in range(n))
    packets = []
    for i in range(n):
        packet = _FakePacket(i)
        if packet_mutator is not None:
            packet_mutator(packet, i)
        packets.append(NS(perception_observation_packet=packet))
    receipt = NS(perception_release_id=release, perception_model_sha256=model_sha)
    monkeypatch.setattr(
        "ausc.perception.tracking.production_evidence_capture.read_geometry_model_input_ledger",
        lambda _s: records,
    )
    monkeypatch.setattr(
        "ausc.perception.tracking.geometry_channel_packet.verify_geometry_packet_log_files",
        lambda _s: (tuple(packets), receipt),
    )


def test_reconstruct_geometry_items_builds_aligned_replay_items(tmp_path, monkeypatch):
    _patch_geometry_sources(monkeypatch, 3, release=_MODEL_RELEASE, model_sha=_MODEL_SHA)
    items, identity = reconstruct_geometry_items(tmp_path)
    assert len(items) == 3
    assert identity == {"release_id": _MODEL_RELEASE, "model_sha256": _MODEL_SHA}
    assert set(items[0].recorded_outputs) == {"points_m", "covariances_m2"}
    assert items[1].model_input.frame_index == 1
    assert items[1].model_input.rgb.shape == (4, 4, 3)


def test_reconstruct_geometry_rejects_input_sha_mismatch(tmp_path, monkeypatch):
    def break_input(packet, i):
        packet.input_rgb_sha256 = "f" * 64

    _patch_geometry_sources(
        monkeypatch, 2, release=_MODEL_RELEASE, model_sha=_MODEL_SHA,
        packet_mutator=break_input,
    )
    with pytest.raises(ReplayRunnerError, match="content addresses"):
        reconstruct_geometry_items(tmp_path)


def test_reconstruct_geometry_rejects_frame_misalignment(tmp_path, monkeypatch):
    def shift_frame(packet, i):
        packet.frame_index = i + 50

    _patch_geometry_sources(
        monkeypatch, 2, release=_MODEL_RELEASE, model_sha=_MODEL_SHA,
        packet_mutator=shift_frame,
    )
    with pytest.raises(ReplayRunnerError, match="not aligned"):
        reconstruct_geometry_items(tmp_path)


def test_reconstruct_then_run_replay_round_trips(tmp_path, monkeypatch):
    session = _geometry_session(tmp_path)
    _patch_geometry_sources(monkeypatch, 2, release=_MODEL_RELEASE, model_sha=_MODEL_SHA)
    items, identity = reconstruct_geometry_items(session)

    # A perfect activated model reproduces the recorded outputs from the input.
    def model_run(model_input):
        i = model_input.frame_index
        return {
            "points_m": np.full((2, 3), float(i), dtype=np.float32),
            "covariances_m2": np.eye(3, dtype=np.float32) * (i + 1),
        }

    manifest = run_replay(
        session, items, model_run,
        component="geometry",
        model_release_id=identity["release_id"], model_sha256=identity["model_sha256"],
        producer=_PRODUCER,
    )
    evidence = verify_replay_attestation(
        session, manifest, component="geometry",
        recorded_identity=identity, recorded_item_count=len(items),
    )
    assert evidence["production_authorized"] is True


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
