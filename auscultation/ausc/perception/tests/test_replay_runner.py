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
    verify_replay_attestation as _verify_impl,
)


def verify_replay_attestation(session_dir, attestation, *, trust_verifier=lambda _r: True, **kwargs):
    return _verify_impl(session_dir, attestation, trust_verifier=trust_verifier, **kwargs)
from ausc.perception import replay_runner
from ausc.perception.replay_runner import (
    ReplayItem,
    ReplayRunnerError,
    compare_outputs,
    compute_bound_artifacts,
    reconstruct_geometry_items,
    reconstruct_quality_items,
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


def test_compare_bit_exact_dtype_mismatch_is_rejected():
    # np.array_equal ignores dtype; bit-exact must not.
    with pytest.raises(ReplayRunnerError, match="dtype"):
        compare_outputs(
            {"p": np.zeros(2, dtype=np.float32)},
            {"p": np.zeros(2, dtype=np.float64)},
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


_INDEP_SHA = "b" * 64
_INDEP_RELEASE = "patient_pose-" + _INDEP_SHA[:20]


def _independent_packet(i: int):
    # Same frame/input identity, distinct model outputs (the independent channel).
    packet = _FakePacket(i)
    packet.points_m = np.full((2, 3), float(i) + 0.5, dtype=np.float32)
    packet.covariances_m2 = np.eye(3, dtype=np.float32) * (i + 2)
    return packet


def _patch_geometry_sources(monkeypatch, n: int, *, release, model_sha, packet_mutator=None):
    from types import SimpleNamespace as NS
    records = tuple(_FakeRecord(i) for i in range(n))
    entries = []
    for i in range(n):
        packet = _FakePacket(i)
        if packet_mutator is not None:
            packet_mutator(packet, i)
        entries.append(NS(
            perception_observation_packet=packet,
            independent_model_packet=_independent_packet(i),
        ))
    receipt = NS(
        perception_release_id=release, perception_model_sha256=model_sha,
        independent_release_id=_INDEP_RELEASE, independent_model_sha256=_INDEP_SHA,
    )
    monkeypatch.setattr(
        "ausc.perception.tracking.production_evidence_capture.read_geometry_model_input_ledger",
        lambda _s: records,
    )
    monkeypatch.setattr(
        "ausc.perception.tracking.geometry_channel_packet.verify_geometry_packet_log_files",
        lambda _s: (tuple(entries), receipt),
    )


def test_reconstruct_geometry_items_builds_aligned_replay_items(tmp_path, monkeypatch):
    _patch_geometry_sources(monkeypatch, 3, release=_MODEL_RELEASE, model_sha=_MODEL_SHA)
    items, identity = reconstruct_geometry_items(tmp_path)
    assert len(items) == 3
    assert identity == {"release_id": _MODEL_RELEASE, "model_sha256": _MODEL_SHA}
    assert set(items[0].recorded_outputs) == {"points_m", "covariances_m2"}
    assert items[1].model_input.frame_index == 1
    assert items[1].model_input.rgb.shape == (4, 4, 3)


def test_reconstruct_geometry_independent_channel_targets_the_independent_model(tmp_path, monkeypatch):
    # The independent channel must be independently replayable: its recorded
    # outputs are the independent model's, and its identity is the independent
    # model (distinct from perception).
    _patch_geometry_sources(monkeypatch, 3, release=_MODEL_RELEASE, model_sha=_MODEL_SHA)
    items, identity = reconstruct_geometry_items(tmp_path, channel="independent")
    assert len(items) == 3
    assert identity == {"release_id": _INDEP_RELEASE, "model_sha256": _INDEP_SHA}
    # distinct from the perception channel's recorded outputs
    assert float(items[1].recorded_outputs["points_m"][0, 0]) == pytest.approx(1.5)


def test_reconstruct_geometry_rejects_unknown_channel(tmp_path, monkeypatch):
    _patch_geometry_sources(monkeypatch, 2, release=_MODEL_RELEASE, model_sha=_MODEL_SHA)
    with pytest.raises(ReplayRunnerError, match="channel must be"):
        reconstruct_geometry_items(tmp_path, channel="bogus")


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


def _quality_session_with_windows(tmp_path, n, *, scores):
    import hashlib as _hashlib

    session = tmp_path / "session-q-001"
    (session / "quality_model_inputs").mkdir(parents=True)
    session.joinpath("quality_inference_packets.jsonl").write_bytes(b'{"p":1}\n')
    session.joinpath("audio.wav").write_bytes(b"real-audio-pcm")
    packets = []
    for i in range(n):
        window = np.linspace(0.1 + i, 0.2 + i, 32, dtype=np.float32)
        raw = window.tobytes()
        rel = f"quality_model_inputs/input_{i:020d}.f32le"
        (session / rel).write_bytes(raw)
        packets.append({
            "model": {"release_id": _MODEL_RELEASE, "model_sha256": _MODEL_SHA},
            "inference_sequence": i,
            "audio_window": {
                "sample_rate": 16_000,
                "model_input": {
                    "path": rel,
                    "sha256": _hashlib.sha256(raw).hexdigest(),
                    "shape": [32],
                },
            },
            "result": {"q_raw": scores[i]},
        })
    return session, packets


def _patch_quality_sources(monkeypatch, packets):
    receipt = {
        "quality_authenticity": {
            "model": {"release_id": _MODEL_RELEASE, "model_sha256": _MODEL_SHA},
        },
    }
    monkeypatch.setattr("ausc.realtime_q.verify_quality_packet_log", lambda _s: receipt)
    monkeypatch.setattr("ausc.realtime_q.read_quality_packet_log", lambda _p: list(packets))


def test_reconstruct_quality_items_loads_verified_windows(tmp_path, monkeypatch):
    session, packets = _quality_session_with_windows(tmp_path, 3, scores=[0.4, 0.55, 0.6])
    _patch_quality_sources(monkeypatch, packets)
    items, identity = reconstruct_quality_items(session)
    assert len(items) == 3
    assert identity == {"release_id": _MODEL_RELEASE, "model_sha256": _MODEL_SHA}
    assert items[1].model_input.window.shape == (32,)
    assert items[1].recorded_outputs["q_raw"][0] == pytest.approx(0.55)


def test_reconstruct_quality_rejects_tampered_window(tmp_path, monkeypatch):
    session, packets = _quality_session_with_windows(tmp_path, 1, scores=[0.4])
    # Corrupt the persisted window after its SHA was claimed.
    (session / packets[0]["audio_window"]["model_input"]["path"]).write_bytes(
        np.zeros(32, dtype=np.float32).tobytes()
    )
    _patch_quality_sources(monkeypatch, packets)
    with pytest.raises(ReplayRunnerError, match="SHA-256 does not match"):
        reconstruct_quality_items(session)


def test_reconstruct_quality_then_run_replay_round_trips(tmp_path, monkeypatch):
    scores = [0.4, 0.55, 0.6]
    session, packets = _quality_session_with_windows(tmp_path, 3, scores=scores)
    _patch_quality_sources(monkeypatch, packets)
    items, identity = reconstruct_quality_items(session)

    # A perfect scorer reproduces the recorded q_raw exactly for each window.
    def model_run(model_input):
        return {"q_raw": np.array([scores[model_input.inference_sequence]], dtype=np.float64)}

    manifest = run_replay(
        session, items, model_run,
        component="quality",
        model_release_id=identity["release_id"], model_sha256=identity["model_sha256"],
        producer=_PRODUCER,
    )
    evidence = verify_replay_attestation(
        session, manifest, component="quality",
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
