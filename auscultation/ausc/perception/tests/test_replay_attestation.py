"""Fail-closed contracts for the model-replay promotion attestation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ausc.perception.replay_attestation import (
    COMPARISON_BIT_EXACT,
    COMPARISON_FROZEN_EPSILON,
    REPLAY_ATTESTATION_SCHEMA,
    ReplayAttestation,
    ReplayAttestationError,
    verify_replay_attestation,
)

_MODEL_SHA = "a" * 64
_MODEL_RELEASE = "perception-" + _MODEL_SHA[:20]
_RECORDED_IDENTITY = {"release_id": _MODEL_RELEASE, "model_sha256": _MODEL_SHA}


def _write(session: Path, relative: str, body: bytes) -> str:
    path = session / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _geometry_session(tmp_path: Path) -> tuple[Path, dict]:
    session = tmp_path / "session-geo-001"
    session.mkdir()
    packets_sha = _write(session, "geometry_frame_packets.jsonl", b'{"frame":1}\n{"frame":2}\n')
    ledger_sha = _write(session, "geometry/model_input_ledger.jsonl", b'{"sequence":1}\n{"sequence":2}\n')
    attestation = {
        "schema_version": REPLAY_ATTESTATION_SCHEMA,
        "component": "geometry",
        "session_id": session.name,
        "bound_artifacts": {
            "geometry_frame_packets.jsonl": packets_sha,
            "geometry/model_input_ledger.jsonl": ledger_sha,
        },
        "model_release_id": _MODEL_RELEASE,
        "model_sha256": _MODEL_SHA,
        "comparison_mode": COMPARISON_BIT_EXACT,
        "item_count": 2,
        "matched_count": 2,
        "max_abs_residual": {"landmark_uv": 0.0, "depth_m": 0.0},
        "epsilon": {},
        "producer": {
            "code_release_id": "code_release-" + "b" * 20,
            "module_sha256": "c" * 64,
            "activation_sha256": "d" * 64,
        },
    }
    return session, attestation


def _quality_session(tmp_path: Path) -> tuple[Path, dict]:
    session = tmp_path / "session-q-001"
    session.mkdir()
    packets_sha = _write(session, "quality_inference_packets.jsonl", b'{"q":0.4}\n{"q":0.6}\n{"q":0.5}\n')
    audio_sha = _write(session, "audio.wav", b"real-audio-pcm-bytes")
    attestation = {
        "schema_version": REPLAY_ATTESTATION_SCHEMA,
        "component": "quality",
        "session_id": session.name,
        "bound_artifacts": {
            "quality_inference_packets.jsonl": packets_sha,
            "audio.wav": audio_sha,
        },
        "model_release_id": _MODEL_RELEASE,
        "model_sha256": _MODEL_SHA,
        "comparison_mode": COMPARISON_BIT_EXACT,
        "item_count": 3,
        "matched_count": 3,
        "max_abs_residual": {"quality": 0.0},
        "epsilon": {},
        "producer": {
            "code_release_id": "code_release-" + "b" * 20,
            "module_sha256": "c" * 64,
            "activation_sha256": "d" * 64,
        },
    }
    return session, attestation


def test_bit_exact_geometry_attestation_authorizes_promotion(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    evidence = verify_replay_attestation(
        session, attestation, component="geometry",
        recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
    )
    assert evidence["production_authorized"] is True
    assert evidence["comparison_mode"] == COMPARISON_BIT_EXACT
    assert evidence["item_count"] == 2


def test_bit_exact_quality_attestation_authorizes_promotion(tmp_path):
    session, attestation = _quality_session(tmp_path)
    evidence = verify_replay_attestation(
        session, attestation, component="quality",
        recorded_identity=_RECORDED_IDENTITY, recorded_item_count=3,
    )
    assert evidence["production_authorized"] is True


def test_attestation_accepts_canonical_bytes(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    raw = json.dumps(attestation).encode("utf-8")
    evidence = verify_replay_attestation(
        session, raw, component="geometry",
        recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
    )
    assert evidence["production_authorized"] is True


def test_tampered_bound_artifact_is_rejected(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    # Mutate the packet log on disk after the attestation bound its SHA.
    (session / "geometry_frame_packets.jsonl").write_bytes(b'{"frame":1}\n{"frame":99}\n')
    with pytest.raises(ReplayAttestationError, match="no longer matches"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
        )


def test_missing_required_bound_artifact_is_rejected(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    del attestation["bound_artifacts"]["geometry/model_input_ledger.jsonl"]
    with pytest.raises(ReplayAttestationError, match="required artifacts"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
        )


def test_replay_of_a_different_model_is_rejected(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    other_sha = "e" * 64
    attestation["model_sha256"] = other_sha
    attestation["model_release_id"] = "perception-" + other_sha[:20]
    with pytest.raises(ReplayAttestationError, match="different model"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
        )


def test_release_id_must_content_address_the_model_sha(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    attestation["model_release_id"] = "perception-" + ("f" * 20)
    with pytest.raises(ReplayAttestationError, match="content-address"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity={"release_id": attestation["model_release_id"],
                               "model_sha256": _MODEL_SHA},
            recorded_item_count=2,
        )


def test_item_count_must_equal_recorded_packet_count(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    with pytest.raises(ReplayAttestationError, match="item count"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=3,
        )


def test_partial_match_coverage_is_rejected(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    attestation["matched_count"] = 1
    with pytest.raises(ReplayAttestationError, match="did not match"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
        )


def test_bit_exact_with_nonzero_residual_is_rejected(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    attestation["max_abs_residual"] = {"landmark_uv": 0.0, "depth_m": 1e-6}
    with pytest.raises(ReplayAttestationError, match="non-zero residual"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
        )


def test_bit_exact_with_declared_epsilon_is_rejected(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    attestation["epsilon"] = {"depth_m": 1e-6}
    with pytest.raises(ReplayAttestationError, match="must not declare"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
        )


def test_frozen_epsilon_is_refused_until_a_ceiling_is_calibrated(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    attestation["comparison_mode"] = COMPARISON_FROZEN_EPSILON
    attestation["epsilon"] = {"landmark_uv": 1e-4, "depth_m": 1e-4}
    attestation["max_abs_residual"] = {"landmark_uv": 5e-5, "depth_m": 5e-5}
    # Default policy has no frozen ceiling -> refuse.
    with pytest.raises(ReplayAttestationError, match="refused until a tolerance"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
        )


def test_frozen_epsilon_within_a_frozen_ceiling_authorizes(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    attestation["comparison_mode"] = COMPARISON_FROZEN_EPSILON
    attestation["epsilon"] = {"landmark_uv": 1e-4, "depth_m": 1e-4}
    attestation["max_abs_residual"] = {"landmark_uv": 5e-5, "depth_m": 5e-5}
    ceilings = {"geometry": {"landmark_uv": 2e-4, "depth_m": 2e-4}}
    evidence = verify_replay_attestation(
        session, attestation, component="geometry",
        recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
        epsilon_ceilings=ceilings,
    )
    assert evidence["production_authorized"] is True
    assert evidence["comparison_mode"] == COMPARISON_FROZEN_EPSILON


def test_frozen_epsilon_above_the_ceiling_is_rejected(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    attestation["comparison_mode"] = COMPARISON_FROZEN_EPSILON
    attestation["epsilon"] = {"landmark_uv": 3e-4, "depth_m": 1e-4}
    attestation["max_abs_residual"] = {"landmark_uv": 1e-4, "depth_m": 5e-5}
    ceilings = {"geometry": {"landmark_uv": 2e-4, "depth_m": 2e-4}}
    with pytest.raises(ReplayAttestationError, match="exceeds the frozen ceiling"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
            epsilon_ceilings=ceilings,
        )


def test_frozen_epsilon_residual_above_declared_epsilon_is_rejected(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    attestation["comparison_mode"] = COMPARISON_FROZEN_EPSILON
    attestation["epsilon"] = {"landmark_uv": 1e-4, "depth_m": 1e-4}
    attestation["max_abs_residual"] = {"landmark_uv": 2e-4, "depth_m": 5e-5}
    ceilings = {"geometry": {"landmark_uv": 5e-4, "depth_m": 5e-4}}
    with pytest.raises(ReplayAttestationError, match="exceeds its declared epsilon"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
            epsilon_ceilings=ceilings,
        )


def test_frozen_epsilon_output_set_must_match_the_policy(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    attestation["comparison_mode"] = COMPARISON_FROZEN_EPSILON
    attestation["epsilon"] = {"landmark_uv": 1e-4}  # missing depth_m
    attestation["max_abs_residual"] = {"landmark_uv": 5e-5}
    ceilings = {"geometry": {"landmark_uv": 2e-4, "depth_m": 2e-4}}
    with pytest.raises(ReplayAttestationError, match="do not match the frozen policy"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
            epsilon_ceilings=ceilings,
        )


def test_component_mismatch_between_request_and_attestation_is_rejected(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    with pytest.raises(ReplayAttestationError, match="component does not match"):
        verify_replay_attestation(
            session, attestation, component="quality",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
        )


def test_symlinked_bound_artifact_is_rejected(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b'{"frame":1}\n{"frame":2}\n')
    target = session / "geometry_frame_packets.jsonl"
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ReplayAttestationError, match="symlink"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
        )


def test_manifest_field_set_must_be_exact(tmp_path):
    _, attestation = _geometry_session(tmp_path)
    attestation["unexpected"] = True
    with pytest.raises(ReplayAttestationError, match="field set is not exact"):
        ReplayAttestation.from_manifest(attestation)


def test_producer_identity_must_be_exact_and_valid(tmp_path):
    session, attestation = _geometry_session(tmp_path)
    attestation["producer"]["module_sha256"] = "not-a-sha"
    with pytest.raises(ReplayAttestationError, match="producer module"):
        verify_replay_attestation(
            session, attestation, component="geometry",
            recorded_identity=_RECORDED_IDENTITY, recorded_item_count=2,
        )
