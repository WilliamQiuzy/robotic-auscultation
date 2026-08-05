"""Strict canonical-v2 conversion tests; no LeRobot installation required."""

from __future__ import annotations

import json
import hashlib
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import session_to_lerobot_v2 as conv  # noqa: E402
from deployment import artifact_contract as promotion_contract  # noqa: E402


def _canonical_json(value: object) -> bytes:
    return promotion_contract.canonical_json_bytes(value)


def _seal_synthetic_outer_v2(
    session: Path, *, manifest_fields: dict | None = None,
) -> dict:
    """Create only the outer-chain fixture; its raw payloads are deliberately fake.

    Tests using this helper must stop at a real converter semantic/raw gate.  The
    helper is not a shortcut for manufacturing a training-eligible conversion.
    """
    raw_bodies = {
        "audio.wav": b"not-a-real-wav",
        "audio_capture.json": b'{"status":"sensors_finalized"}\n',
        "audio_input_times.jsonl": b'{"sequence":0}\n',
        "audio_times.jsonl": b'{"sequence":0}\n',
        "force.csv": b"host_t_s,raw,force_n\n1,100,0.1\n",
        "force_capture.json": b'{"status":"sensors_finalized"}\n',
        "FORCE_CAPTURE_COMPLETE.json": b'{"status":"complete"}\n',
        "force_calibration.json": b'{"calibrated":true}\n',
        "rgb.mp4": b"not-a-real-video",
        "rgb_times.csv": b"frame_index,t\n1,1.0\n",
        "obs.csv": b"t,force_n\n1,0.1\n",
        "robot_control_envelopes.jsonl": b'{"tick":1}\n',
        "robot_exact_control.csv": b"not-real-robot-state",
        "robot_frame_index.jsonl": b'{"frame":1}\n',
        "robot_capture_summary.json": b'{"schema":"v4"}\n',
        "targets.jsonl": b'{"point":1}\n',
        "tracker.jsonl": b'{"frame":1}\n',
        "depth/depth_times.csv": b"frame_index,t\n1,1.0\n",
        "depth/depth_000001.npy": b"not-real-depth",
        "quality_inference_packets.jsonl": b'{"q":0.4}\n{"q":0.6}\n',
        "quality_inference_authenticity.json": b'{"status":"PASS"}\n',
        "geometry_frame_packets.jsonl": b'{"frame":1}\n{"frame":2}\n',
        "geometry_frame_authenticity.json": b'{"status":"PASS"}\n',
        "geometry/patient_depth_evidence.jsonl": b'{"frame":1}\n',
        "geometry/patient_depth_authenticity.json": b'{"status":"PASS"}\n',
        "geometry/model_inputs/" + "a" * 64 + ".npy": b"model-input",
        "geometry/evidence_arrays/back_" + "b" * 64 + ".npy": b"back-mask",
        "geometry/evidence_arrays/robot_" + "c" * 64 + ".npy": b"robot-mask",
        "quality_model_inputs/input_00000000000000000000.f32le": b"quality-input",
    }
    raw_files = {}
    for relative, body in raw_bodies.items():
        path = session / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        raw_files[relative] = {
            "sha256": hashlib.sha256(body).hexdigest(),
            "size_bytes": len(body),
        }
    inventory = {
        "algorithm": "sha256",
        "files": raw_files,
        "schema_version": "wam_session_artifact_hashes_v1",
    }
    component_raw = {
        "audio": [
            "audio.wav", "audio_capture.json", "audio_input_times.jsonl",
            "audio_times.jsonl",
        ],
        "force": [
            "force.csv", "force_capture.json", "FORCE_CAPTURE_COMPLETE.json",
            "force_calibration.json",
        ],
        "geometry": [
            "rgb.mp4", "rgb_times.csv", "obs.csv", "targets.jsonl",
            "tracker.jsonl",
            "depth/depth_times.csv", "depth/depth_000001.npy",
            "geometry_frame_packets.jsonl", "geometry_frame_authenticity.json",
            "geometry/patient_depth_evidence.jsonl",
            "geometry/patient_depth_authenticity.json",
            "geometry/model_inputs/" + "a" * 64 + ".npy",
            "geometry/evidence_arrays/back_" + "b" * 64 + ".npy",
            "geometry/evidence_arrays/robot_" + "c" * 64 + ".npy",
        ],
        "quality": [
            "audio.wav", "force.csv", "quality_inference_packets.jsonl",
            "quality_inference_authenticity.json", "audio_times.jsonl",
            "quality_model_inputs/input_00000000000000000000.f32le",
        ],
        "robot": [
            "robot_control_envelopes.jsonl", "robot_exact_control.csv",
            "robot_frame_index.jsonl", "robot_capture_summary.json",
        ],
    }
    component_refs = {}
    for component, relatives in component_raw.items():
        receipt = {
            "component": component,
            "evidence": {"synthetic_outer_chain_only": True},
            "policy_version": f"{component}_authenticity_v1",
            "raw_artifacts": {
                relative: raw_files[relative]["sha256"] for relative in relatives
            },
            "schema_version": "wam_training_component_authenticity_v2",
            "status": "PASS",
            "training_signal_eligible": True,
            "verifier": {
                "code_release_id": "code-" + "4" * 20,
                "module_path": f"auscultation/verifiers/{component}.py",
                "module_sha256": "2" * 64,
                "policy_sha256": "3" * 64,
            },
        }
        relative = f"authenticity/{component}.json"
        raw = _canonical_json(receipt)
        (session / relative).parent.mkdir(parents=True, exist_ok=True)
        (session / relative).write_bytes(raw)
        component_refs[component] = {
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    authenticity = {
        "components": component_refs,
        "finalizer": {
            "code_manifest_sha256": "4" * 64,
            "code_release_id": "code-" + "4" * 20,
            "environment_id": "runtime-" + "6" * 20,
            "environment_lock_sha256": "6" * 64,
            "module_sha256": "7" * 64,
        },
        "promotion_contract_version": "wam_promotion_v2",
        "raw_artifact_inventory_sha256": hashlib.sha256(
            _canonical_json(inventory)
        ).hexdigest(),
        "schema_version": "wam_training_authenticity_v2",
        "session_id": session.name,
        "status": "PASS",
        "training_eligible": True,
    }
    authenticity_raw = _canonical_json(authenticity)
    (session / "TRAINING_AUTHENTICITY.json").write_bytes(authenticity_raw)
    manifest = {
        "artifact_hashes": inventory,
        "capture_kind": "production_teleop_v4",
        "promotion_contract_version": "wam_promotion_v2",
        "rehearsal_overrides": [],
        "schema_version": "wam_session_v2",
        "session_id": session.name,
        "status": "complete",
        "training_authenticity": {
            "path": "TRAINING_AUTHENTICITY.json",
            "schema_version": "wam_training_authenticity_v2",
            "sha256": hashlib.sha256(authenticity_raw).hexdigest(),
        },
        "training_eligible": True,
    }
    manifest.update(manifest_fields or {})
    manifest_raw = _canonical_json(manifest)
    (session / "manifest.json").write_bytes(manifest_raw)
    (session / "FINALIZATION.json").write_bytes(_canonical_json({
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "promotion_contract_version": "wam_promotion_v2",
        "schema_version": "wam_session_finalization_v2",
        "training_authenticity_sha256": hashlib.sha256(
            authenticity_raw
        ).hexdigest(),
    }))
    return manifest


def _trusted_finalization_record(session: Path) -> dict:
    authenticity_raw = (session / "TRAINING_AUTHENTICITY.json").read_bytes()
    authenticity = json.loads(authenticity_raw)
    manifest_raw = (session / "manifest.json").read_bytes()
    finalization_raw = (session / "FINALIZATION.json").read_bytes()
    return {
        "artifact_sha256": {
            "finalization": hashlib.sha256(finalization_raw).hexdigest(),
            "manifest": hashlib.sha256(manifest_raw).hexdigest(),
            "training_authenticity": hashlib.sha256(authenticity_raw).hexdigest(),
        },
        "component_verifiers": {
            component: json.loads(
                (session / reference["path"]).read_text()
            )["verifier"]
            for component, reference in authenticity["components"].items()
        },
        "finalizer": authenticity["finalizer"],
        "promotion_contract_version": "wam_promotion_v2",
        "schema_version": "wam_trusted_training_finalization_v1",
        "session_id": session.name,
    }


def _trusted_verifier_config(session: Path, root: Path) -> dict:
    registry = root / (session.name + "-trusted-finalizations.json")
    registry.write_bytes(_canonical_json({
        "records": [_trusted_finalization_record(session)],
        "schema_version": "wam_trusted_training_finalization_registry_v1",
    }))
    return {
        "trusted_finalization_registry": registry,
        "trusted_snapshot_root": root / (session.name + "-trusted-snapshots"),
    }


def _manifest(nj=2):
    return {
        "arm": {
            "side": "left",
            "joint_names": [f"joint_{i + 1}" for i in range(nj)],
            "state_unit": "rad",
            "action_type": "position_target",
        }
    }


def _artifact_contract() -> tuple[dict, dict]:
    names = [
        "audio.wav", "rgb.mp4", "force.csv", "rgb_times.csv",
        "obs.csv", "targets.jsonl", "tracker.jsonl", "depth/depth_times.csv",
        "depth/depth_000000.npy", *conv.EXACT_ROBOT_FILES,
    ]
    return (
        {
            "audio": {"sample_rate": 16_000, "n_frames": 16_000, "duration_s": 1.0},
            "video": {"n_frames": 1, "shape": [32, 32, 3], "encoded_fps": 30.0},
            "depth": {"n_frames": 1, "shape": [32, 32]},
            "observations": {"n_rows": 1},
        },
        {
            "schema_version": "wam_session_artifact_hashes_v1",
            "algorithm": "sha256",
            "files": {
                name: {"sha256": "a" * 64, "size_bytes": 1} for name in names
            },
        },
    )


def _diagnostic_v3_robot_summary() -> dict:
    return {
        "schema": conv.EXACT_ROBOT_SUMMARY_SCHEMA,
        "status": "exact_control_capture_validated_socketcan_tx_enqueue",
        "side": "left",
        "controller_mode": "unilateral",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "joint_names": [f"openarm_left_joint{index}" for index in range(1, 8)],
        "action_semantics": "exact_mitparam_consumed_at_control_tick",
        "action_interpolation": "forbidden",
        "send_status_observable": True,
        "send_status_reason": "openarm_can_1_2_2_full_frame_write_results",
        "send_status_semantics": (
            "linux_socketcan_full_frame_write_tx_enqueue_not_motor_ack"
        ),
        "motor_ack_observable": False,
        "motor_ack_reason": (
            "socketcan_tx_enqueue_does_not_acknowledge_motor_receipt_or_execution"
        ),
        "feedback_liveness_observable": False,
        "feedback_liveness_reason": (
            "v3_has_no_per_motor_feedback_rx_counters_or_timestamps"
        ),
        "training_eligible": False,
        "eligibility_blockers": ["robot_feedback_liveness_unobservable"],
        "control_tick_count": 3,
        "arm_tx_expected_frame_count": 21,
        "arm_tx_full_frame_write_count": 21,
        "arm_tx_failed_frame_write_count": 0,
        "gripper_tx_expected_frame_count": 3,
        "gripper_tx_full_frame_write_count": 3,
        "gripper_tx_failed_frame_write_count": 0,
        "socketcan_tx_enqueue_failure_tick_count": 0,
    }


def _synthetic_v4_robot_summary_for_gate_isolation() -> dict:
    """Name the v4 contract without bypassing it in production code.

    A few transaction/manifest tests monkeypatch the dedicated robot
    eligibility validator so they can exercise a later, independent gate.  The
    manifest-level schema check must still see the current v4 contract in those
    deliberately isolated tests.
    """
    summary = _diagnostic_v3_robot_summary()
    summary["schema"] = conv.EXACT_ROBOT_SUMMARY_SCHEMA_V4
    return summary


def test_training_gate_rejects_diagnostic_v3_without_feedback_liveness():
    summary = _diagnostic_v3_robot_summary()

    with pytest.raises(ValueError, match="feedback liveness.*v3"):
        conv.require_exact_robot_training_eligibility(summary)


def _exact_arm_metadata() -> dict:
    return {
        "side": "left",
        "joint_names": [f"openarm_left_joint{index}" for index in range(1, 8)],
        "state_unit": "rad",
        "action_type": "exact_mitparam_consumed_at_control_tick",
    }


def _write_snapshot_root_files(session: Path) -> None:
    for name in (
        "manifest.json", "audio.wav", "rgb.mp4", "force.csv", "rgb_times.csv",
        "obs.csv", "targets.jsonl", *conv.EXACT_ROBOT_FILES,
    ):
        (session / name).write_bytes(f"source:{name}".encode())


def test_canonical_converter_requires_promotion_v2_and_rejects_v1():
    assert conv.PROMOTION_CONTRACT_VERSION == "wam_promotion_v2"
    assert conv.FINALIZATION_SCHEMA_VERSION == "wam_session_finalization_v2"

    with pytest.raises(ValueError, match="wam_promotion_v2"):
        conv.validate_manifest_for_conversion({
            "schema_version": "wam_session_v2",
            "promotion_contract_version": "wam_promotion_v1",
        })


def test_v2_finalization_seal_binds_manifest_and_training_authenticity(tmp_path):
    manifest_raw = b'{"promotion_contract_version":"wam_promotion_v2"}\n'
    authenticity_raw = b'{"schema_version":"wam_training_authenticity_v2"}\n'
    (tmp_path / "manifest.json").write_bytes(manifest_raw)
    (tmp_path / "TRAINING_AUTHENTICITY.json").write_bytes(authenticity_raw)
    (tmp_path / "FINALIZATION.json").write_bytes(_canonical_json({
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "promotion_contract_version": "wam_promotion_v2",
        "schema_version": "wam_session_finalization_v2",
        "training_authenticity_sha256": hashlib.sha256(
            authenticity_raw
        ).hexdigest(),
    }))

    verified = conv._verify_finalization_seal(tmp_path)
    assert verified.sha256 == hashlib.sha256(
        (tmp_path / "FINALIZATION.json").read_bytes()
    ).hexdigest()

    (tmp_path / "TRAINING_AUTHENTICITY.json").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="training authenticity hash mismatch"):
        conv._verify_finalization_seal(tmp_path)


def test_verified_v2_snapshot_copies_all_outer_and_raw_files_and_reverifies(
    tmp_path, monkeypatch,
):
    session = tmp_path / "session-v2"
    session.mkdir()
    manifest = _seal_synthetic_outer_v2(session)
    trusted = _trusted_verifier_config(session, tmp_path)

    actual_verify = conv._verify_training_session_contract
    verified_paths = []

    def verify_and_record(path, **kwargs):
        verified_paths.append(Path(path).resolve())
        return actual_verify(path, **kwargs)

    monkeypatch.setattr(conv, "_verify_training_session_contract", verify_and_record)
    verified = conv._verify_training_session_contract(session, **trusted)
    trusted_source = Path(verified["verified_snapshot_path"])
    expected = conv._training_contract_file_hashes(
        trusted_source, manifest, verified,
    )
    root = tmp_path / "snapshot-root"
    provenance = conv.snapshot_session_sources(
        trusted_source,
        root,
        expected_contract_hashes=expected,
        verify_training_contract=True,
        verified_training_contract=verified,
    )
    snapshot = root / "source_sessions" / session.name

    assert verified_paths == [session.resolve()]
    for relative in expected:
        assert (snapshot / relative).read_bytes() == (trusted_source / relative).read_bytes()
    assert (snapshot / "TRAINING_AUTHENTICITY.json").is_file()
    assert sorted(path.name for path in (snapshot / "authenticity").glob("*.json")) == [
        "audio.json", "force.json", "geometry.json", "quality.json", "robot.json",
    ]
    assert (snapshot / "quality_inference_packets.jsonl").is_file()
    assert (snapshot / "geometry_frame_packets.jsonl").is_file()
    assert (snapshot / "force_capture.json").is_file()
    assert provenance["schema_version"] == "wam_conversion_provenance_v2"


def test_converter_trust_boundary_fails_closed_without_registry_or_snapshot_root(
    tmp_path, monkeypatch,
):
    session = tmp_path / "session-v2"
    session.mkdir()
    _seal_synthetic_outer_v2(session)
    monkeypatch.delenv(conv.TRUSTED_FINALIZATION_REGISTRY_ENV, raising=False)
    monkeypatch.delenv(conv.TRUSTED_SNAPSHOT_ROOT_ENV, raising=False)

    with pytest.raises(ValueError, match="trusted finalization registry.*required"):
        conv._verify_training_session_contract(session)

    trusted = _trusted_verifier_config(session, tmp_path)
    with pytest.raises(ValueError, match="trusted training snapshot root.*required"):
        conv._verify_training_session_contract(
            session,
            trusted_finalization_registry=trusted[
                "trusted_finalization_registry"
            ],
        )


def test_converter_reads_only_trusted_snapshot_after_outer_verification(tmp_path):
    session = tmp_path / "session-v2"
    session.mkdir()
    manifest = _seal_synthetic_outer_v2(session)
    verified = conv._verify_training_session_contract(
        session, **_trusted_verifier_config(session, tmp_path),
    )
    trusted_source = Path(verified["verified_snapshot_path"])
    original_audio = (trusted_source / "audio.wav").read_bytes()
    (session / "audio.wav").write_bytes(b"producer-mutated-after-verification")
    expected = conv._training_contract_file_hashes(
        trusted_source, manifest, verified,
    )

    provenance = conv.snapshot_session_sources(
        trusted_source,
        tmp_path / "working-copy",
        expected_contract_hashes=expected,
        verify_training_contract=True,
        verified_training_contract=verified,
    )
    copied = (
        tmp_path / "working-copy" / "source_sessions" / session.name / "audio.wav"
    )
    assert copied.read_bytes() == original_audio
    assert provenance["frozen_training_contract"]["verified_snapshot_path"] == str(
        trusted_source
    )


def test_v2_snapshot_rejects_raw_names_reserved_for_converter_derivatives(tmp_path):
    session = tmp_path / "session-v2"
    session.mkdir()
    raw = session / "provenance.json"
    raw.write_bytes(b"must-not-be-overwritten\n")

    with pytest.raises(ValueError, match="prior trusted training contract"):
        conv.snapshot_session_sources(
            session,
            tmp_path / "snapshot-root",
            expected_contract_hashes={
                "provenance.json": hashlib.sha256(raw.read_bytes()).hexdigest(),
            },
            verify_training_contract=True,
        )


def test_outer_v2_receipts_do_not_bypass_converter_robot_semantics(tmp_path):
    session = tmp_path / "session-v2"
    session.mkdir()
    _seal_synthetic_outer_v2(session, manifest_fields={
        "streams": ["force", "audio", "cam", "robot"],
        "arm": _exact_arm_metadata(),
        "robot": {"schema": "v4"},
        "force": {"calibrated": True},
        "artifact_validation": _artifact_contract()[0],
        "camera": {
            "backend": "orbbec", "camera_id": "fixture-camera",
            "camera_id_source": "verified_orbbec_serial", "has_depth": True,
        },
    })

    trusted = _trusted_verifier_config(session, tmp_path)
    with pytest.raises(ValueError, match="exact-control v4 robot summary"):
        conv.build(
            session, "ausc/test", tmp_path / "dataset", dataset_cls=object,
            **trusted,
        )
    assert not (tmp_path / "dataset").exists()


def test_build_verifies_outer_v2_chain_before_any_converter_semantic_gate(tmp_path):
    session = tmp_path / "session-v2"
    session.mkdir()
    _seal_synthetic_outer_v2(session, manifest_fields={
        "streams": ["force", "audio", "cam", "robot"],
        "arm": _exact_arm_metadata(),
        "robot": {"schema": "v4"},
    })
    trusted = _trusted_verifier_config(session, tmp_path)
    (session / "authenticity" / "audio.json").write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="training promotion v2 verification failed"):
        conv.build(
            session, "ausc/test", tmp_path / "dataset", dataset_cls=object,
            **trusted,
        )
    assert not (tmp_path / "dataset").exists()


def test_converter_requires_directory_basename_to_equal_verified_session_id(tmp_path):
    original = tmp_path / "session-logical-id"
    original.mkdir()
    _seal_synthetic_outer_v2(original)
    renamed = tmp_path / "renamed-directory"
    original.rename(renamed)

    trusted = _trusted_verifier_config(renamed, tmp_path)
    with pytest.raises(ValueError, match="training promotion v2 verification failed"):
        conv._verify_training_session_contract(renamed, **trusted)


def _install_passing_raw_component_verifiers(monkeypatch):
    from mechanical_sensor.force_feedback import force_capture
    from ausc import realtime_q
    from ausc.perception.tracking import geometry_channel_packet

    monkeypatch.setattr(
        conv,
        "_verify_audio_capture_artifacts",
        lambda _session: {"schema_version": "wam_audio_signal_authenticity_v1",
                          "status": "PASS"},
    )
    monkeypatch.setattr(
        force_capture,
        "verify_force_capture",
        lambda _session: SimpleNamespace(
            status="OK",
            signal_authenticity={
                "status": "PASS", "promotion_eligible": True,
                "promotion_blockers": [],
            },
        ),
    )
    model_sha = "8" * 64
    policy_sha = "9" * 64
    monkeypatch.setattr(
        geometry_channel_packet,
        "verify_geometry_packet_log_files",
        lambda _session: ((), SimpleNamespace(
            status="PASS",
            production_authorized=False,
            authorization_blockers=(
                geometry_channel_packet.GEOMETRY_RECEIPT_AUTHORIZATION_BLOCKER,
            ),
            packet_count=2,
            packet_log_sha256="a" * 64,
            receipt_sha256="b" * 64,
            perception_release_id="perception-" + "c" * 20,
            perception_model_sha256="c" * 64,
            independent_release_id="independent-" + "d" * 20,
            independent_model_sha256="d" * 64,
        )),
    )
    monkeypatch.setattr(
        realtime_q,
        "verify_quality_packet_log",
        lambda _session: {
            "schema_version": "wam_quality_packet_log_authenticity_v2",
            "status": "PASS",
            "promotion_eligible": False,
            "promotion_blockers": [realtime_q.QUALITY_PROMOTION_BLOCKER],
            "packet_count": 2,
            "sha256": "e" * 64,
            "quality_authenticity": {
                "model": {
                    "release_id": "quality-" + model_sha[:20],
                    "model_sha256": model_sha,
                },
                "policy_sha256": policy_sha,
            },
            "chain_verification": {
                "status": "PASS",
                "chain_integrity": "PASS",
                "training_signal_eligible": True,
                "output_nondegeneracy": {"status": "PASS"},
            },
        },
    )
    return realtime_q


def test_audio_raw_gate_decodes_wav_and_recomputes_authenticity(tmp_path):
    from ausc.audio.audio_stream import AudioSignalAuthenticityAccumulator

    sample_rate = 16_000
    pcm = np.random.default_rng(7).integers(
        -40, 41, size=sample_rate, dtype=np.int16,
    )
    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.astype("<i2").tobytes())
    analyzer = AudioSignalAuthenticityAccumulator(sample_rate)
    analyzer.update(pcm)
    receipt = analyzer.finish()
    (tmp_path / "audio_capture.json").write_bytes(_canonical_json({
        "schema_version": "wam_audio_stream_capture_v1",
        "status": "sensors_finalized",
        "training_eligible": False,
        "sample_rate": sample_rate,
        "n_samples": int(pcm.size),
        "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
        "signal_authenticity": receipt,
    }))

    assert conv._verify_audio_capture_artifacts(tmp_path) == receipt

    metadata = json.loads((tmp_path / "audio_capture.json").read_text())
    metadata["signal_authenticity"]["status"] = "FAIL"
    (tmp_path / "audio_capture.json").write_bytes(_canonical_json(metadata))
    with pytest.raises(ValueError, match="audio authenticity receipt"):
        conv._verify_audio_capture_artifacts(tmp_path)

    metadata["signal_authenticity"] = receipt
    metadata["sample_rate"] = str(sample_rate)
    (tmp_path / "audio_capture.json").write_bytes(_canonical_json(metadata))
    with pytest.raises(ValueError, match="sample identity"):
        conv._verify_audio_capture_artifacts(tmp_path)


def test_raw_geometry_and_q_receipts_alone_cannot_authorize_conversion(
    tmp_path, monkeypatch,
):
    _install_passing_raw_component_verifiers(monkeypatch)

    with pytest.raises(ValueError, match="outer promotion-v2 authorization"):
        conv._reverify_component_raw_gates(tmp_path, outer_contract=None)


def test_full_outer_v2_can_authorize_exact_receipt_only_blockers_after_raw_recheck(
    tmp_path, monkeypatch,
):
    session = tmp_path / "session-v2"
    session.mkdir()
    _seal_synthetic_outer_v2(session)
    outer = conv._verify_training_session_contract(
        session, **_trusted_verifier_config(session, tmp_path),
    )
    _install_passing_raw_component_verifiers(monkeypatch)

    verified = conv._reverify_component_raw_gates(session, outer_contract=outer)

    assert verified["status"] == "PASS"
    assert set(verified) == {
        "schema_version", "status", "audio", "force", "geometry", "quality",
    }
    assert verified["geometry"]["outer_authorized"] is True
    assert verified["quality"]["outer_authorized"] is True


def test_outer_v2_authorization_cannot_be_replayed_across_sessions(
    tmp_path, monkeypatch,
):
    source = tmp_path / "session-source"
    source.mkdir()
    _seal_synthetic_outer_v2(source)
    outer = conv._verify_training_session_contract(
        source, **_trusted_verifier_config(source, tmp_path),
    )
    target = tmp_path / "session-target"
    target.mkdir()
    _install_passing_raw_component_verifiers(monkeypatch)

    with pytest.raises(ValueError, match="outer promotion-v2 authorization"):
        conv._reverify_component_raw_gates(target, outer_contract=outer)


def test_outer_v2_still_rejects_quality_without_content_addressed_model_identity(
    tmp_path, monkeypatch,
):
    session = tmp_path / "session-v2"
    session.mkdir()
    _seal_synthetic_outer_v2(session)
    outer = conv._verify_training_session_contract(
        session, **_trusted_verifier_config(session, tmp_path),
    )
    realtime_q = _install_passing_raw_component_verifiers(monkeypatch)
    receipt = realtime_q.verify_quality_packet_log(session)
    receipt["quality_authenticity"]["model"] = {}
    monkeypatch.setattr(realtime_q, "verify_quality_packet_log", lambda _session: receipt)

    with pytest.raises(ValueError, match="quality model identity"):
        conv._reverify_component_raw_gates(session, outer_contract=outer)


def test_outer_v2_cannot_override_a_failed_raw_geometry_status(
    tmp_path, monkeypatch,
):
    from ausc.perception.tracking import geometry_channel_packet

    session = tmp_path / "session-v2"
    session.mkdir()
    _seal_synthetic_outer_v2(session)
    outer = conv._verify_training_session_contract(
        session, **_trusted_verifier_config(session, tmp_path),
    )
    _install_passing_raw_component_verifiers(monkeypatch)
    _entries, receipt = geometry_channel_packet.verify_geometry_packet_log_files(session)
    fields = dict(vars(receipt))
    fields.update(status="FAIL", production_authorized=True, authorization_blockers=())
    monkeypatch.setattr(
        geometry_channel_packet,
        "verify_geometry_packet_log_files",
        lambda _session: ((), SimpleNamespace(**fields)),
    )

    with pytest.raises(ValueError, match="geometry raw receipt"):
        conv._reverify_component_raw_gates(session, outer_contract=outer)


def test_asof_alignment_never_selects_a_future_sample():
    idx, age = conv.asof_within([1.0, 1.1], 1.08, max_age_s=0.10, stream="robot")
    assert idx == 0
    assert age == pytest.approx(0.08)
    with pytest.raises(ValueError, match="robot alignment gap"):
        conv.asof_within([1.0], 1.2, max_age_s=0.05, stream="robot")
    with pytest.raises(ValueError, match="no past robot"):
        conv.asof_within([1.1], 1.0, max_age_s=0.05, stream="robot")


def test_read_force_never_fabricates_a_missing_stream(tmp_path):
    with pytest.raises(FileNotFoundError, match="force.csv"):
        conv.read_force(tmp_path)


def test_read_force_resolves_columns_by_name_and_keeps_device_time(tmp_path):
    (tmp_path / "force.csv").write_text(
        "host_t_s,raw,hall_v,displacement_mm,force_n,status,device_t_ms\n"
        "1.0,10,1.0,0.0,3.9,OK,100.0\n"
        "1.1,11,1.1,0.1,4.0,OK,200.0\n"
    )
    stream = conv.read_force(tmp_path)
    assert stream.t_host_s.tolist() == [1.0, 1.1]
    assert stream.force_n.tolist() == pytest.approx([3.9, 4.0])
    assert stream.device_t_ms.tolist() == pytest.approx([100.0, 200.0])


def test_read_robot_requires_matching_metadata_backed_groups(tmp_path):
    q_columns = [f"q{i}" for i in range(1, 8)]
    a_columns = [f"a{i}" for i in range(1, 8)]
    row = ",".join(str(value / 10) for value in range(1, 15))
    (tmp_path / "robot.csv").write_text(
        f"t,{','.join(q_columns + a_columns)}\n"
        f"1.0,{row}\n"
        f"1.1,{row}\n"
    )
    stream = conv.read_robot(tmp_path, _manifest(7))
    assert stream.joint_names == tuple(f"joint_{i}" for i in range(1, 8))
    assert stream.state.shape == (2, 7)
    assert stream.action.shape == (2, 7)

    (tmp_path / "robot.csv").write_text("t,q1,q2,a1\n1.0,0.1,0.2,0.3\n")
    with pytest.raises(ValueError, match="robot.csv header"):
        conv.read_robot(tmp_path, _manifest())


def test_read_robot_compares_header_joint_identity_and_order_to_manifest(tmp_path):
    (tmp_path / "robot.csv").write_text(
        "t,q2,q1,a2,a1\n"
        "1.0,0.2,0.1,0.4,0.3\n"
        "1.1,0.3,0.2,0.5,0.4\n"
    )
    with pytest.raises(ValueError, match="joint identities/order"):
        conv.read_robot(tmp_path, _manifest())

    (tmp_path / "robot.csv").write_text(
        "t,q1,q2,a1,a2\n"
        "1.0,0.1,0.2,0.3,0.4\n"
        "1.1,0.2,0.3,0.4,0.5\n"
    )
    stream = conv.read_robot(tmp_path, _manifest())
    assert stream.joint_names == ("joint_1", "joint_2")


def test_read_targets_does_not_turn_missing_xyz_into_origin(tmp_path):
    (tmp_path / "targets.jsonl").write_text(
        json.dumps({"t": 1.0, "name": "R1", "uv": [0.2, 0.3], "xyz": None,
                    "xyz_frame": None}) + "\n"
    )
    stream = conv.read_targets(tmp_path)
    assert stream.xyz[0] is None


def test_read_targets_uses_bound_camera_exposure_time(tmp_path):
    (tmp_path / "targets.jsonl").write_text(json.dumps({
        "t": 10.25, "camera_frame_index": 7, "name": "R1", "uv": [0.2, 0.3],
        "xyz": [0.1, 0.2, 0.3], "xyz_frame": "robot_base",
    }) + "\n")
    camera = conv.CameraTimeline(
        frame_index=np.array([7]), t_exposure_s=np.array([9.9]),
        host_arrival_t_s=np.array([10.25]), device_t_ms=np.array([1234.5]),
        clock_fit={"accepted": True}, latency_ms=12.0,
    )
    stream = conv.read_targets(tmp_path, camera)
    assert stream.t_s.tolist() == pytest.approx([9.9])


def test_causal_frame_uses_only_past_samples_and_no_future_labels_as_observations():
    robot = conv.RobotStream(
        t_s=np.array([1.0, 1.1]),
        state=np.array([[1.0, 2.0], [9.0, 9.0]], np.float32),
        action=np.array([[3.0, 4.0], [8.0, 8.0]], np.float32),
        joint_names=("joint_1", "joint_2"),
    )
    force = conv.ForceStream(
        t_host_s=np.array([1.0, 1.07]), force_n=np.array([3.8, 4.1], np.float32),
        status=("OK", "OK"), device_t_ms=None,
    )
    targets = conv.TargetStream(
        t_s=np.array([0.9, 1.2]), names=("R1", "R2"),
        uv=np.array([[0.2, 0.3], [0.8, 0.8]], np.float32),
        xyz=((0.1, 0.2, 0.3), (0.9, 0.9, 0.9)), xyz_frames=("robot_base", "robot_base"),
    )

    built = conv.build_causal_frame(
        1.08, robot, force, targets, reward=0.75, done=False,
        robot_max_age_s=0.1, force_max_age_s=0.05, target_max_age_s=0.25,
    )
    frame = built.frame
    assert frame["observation.state"].tolist() == [1.0, 2.0]
    assert frame["action"].tolist() == [3.0, 4.0]
    assert frame["observation.contact_force_n"].item() == pytest.approx(4.1)
    assert frame["observation.robot_state_age_s"].item() == pytest.approx(0.08)
    assert frame["observation.contact_force_age_s"].item() == pytest.approx(0.01)
    assert built.target_name == "R1"
    # The VLA "L" is anatomy-grounded, not the opaque code (phrasing augmentation is applied
    # later at the call site, seeded per session/site).
    assert built.task == "auscultate the right upper posterior lung field"
    assert "R1" not in built.task
    assert "observation.q_usable" not in frame
    assert "observation.contact_gate" not in frame
    assert frame["next.reward"].item() == pytest.approx(0.75)


def _sync_capture_mode(
    force_sha="f" * 64, *, audio_index=3, hall_port="/dev/fixture-hall",
    device_name="Fixture USB Audio",
):
    return {
        "audio_channel": 1,
        "force_calibration_sha256": force_sha,
        "audio": {
            "requested_device": "fixture-audio", "portaudio_index": audio_index,
            "hostapi_name": "ALSA", "device_name": device_name,
            "max_input_channels": 2, "usb_serial": "AUDIO-SERIAL-001",
            "usb_vid": 0x1235, "usb_pid": 0x8211,
            "usb_identity_source": "linux_alsa_sysfs",
        },
        "hall": {
            "requested_port": hall_port, "device": hall_port,
            "usb_serial": "HALL-SERIAL-001", "usb_vid": 0x2E8A, "usb_pid": 0x000A,
            "manufacturer": "Raspberry Pi", "product": "Pico", "location": "1-2",
        },
    }


def _sync_identity_match(calibration, runtime):
    return {
        "hall_usb": "matched", "audio_usb": "matched",
        "audio_portaudio_index": (
            "matched" if calibration["audio"]["portaudio_index"]
            == runtime["audio"]["portaudio_index"] else "remapped"
        ),
    }


def _camera_capture_mode(
    backend="orbbec", camera_id="fixture-camera", width=640, height=480, fps=30.0,
):
    mode = {
        "backend": backend,
        "camera_id": camera_id,
        "width": width,
        "height": height,
        "fps": fps,
        "time_source": (
            "orbbec_device_and_host_arrival" if backend == "orbbec"
            else "opencv_host_arrival"
        ),
    }
    if backend == "orbbec":
        mode.update({
            "color_format": "RGB",
            "depth_mode": {"width": 640, "height": 576, "fps": 30.0, "format": "Y16"},
            "align_mode": "software_depth_to_color",
            "frame_aggregate_mode": "full_frame_require",
        })
    else:
        mode.update({
            "color_format": "opencv_decoded_bgr",
            "depth_mode": None,
            "align_mode": "disabled",
            "frame_aggregate_mode": "not_applicable",
        })
    return mode


def _camera_mode_validation(backend="orbbec", mode=None):
    mode = mode or _camera_capture_mode(
        backend=backend, camera_id="fixture-camera" if backend == "orbbec" else "avf-0",
    )
    validation = {
        "width": "matched_frame_shape",
        "height": "matched_frame_shape",
        "fps": "matched_negotiated_mode",
        "time_source": "matched_frame_metadata",
        "device_timestamp": (
            "present" if backend == "orbbec" else "not_provided_by_backend"
        ),
    }
    if backend == "orbbec":
        validation["camera_id"] = "matched_device_serial"
        validation["depth"] = "matched_pipeline_shape"
    validation["pipeline"] = "matched_negotiated_pipeline"
    validation["negotiated_mode"] = {
        field: mode[field]
        for field in (
            "width", "height", "fps", "color_format", "depth_mode", "align_mode",
            "frame_aggregate_mode",
        )
    }
    return validation


def _sync_payload(offset_ms=12.5, *, force_sha="f" * 64):
    return {
        "schema_version": "wam_sync_v1",
        "offset_definition": "force_t_minus_audio_t",
        "audio_to_force_offset_ms": offset_ms,
        "offset_std_ms": 1.0,
        "n_matched": 4,
        "unique_matches": True,
        "measurement_method": "physical_diaphragm_taps",
        "capture_mode": _sync_capture_mode(force_sha),
    }


def test_sync_hash_sign_and_exactly_once_application(tmp_path):
    sync_path = tmp_path / "sync.json"
    sync_path.write_text(json.dumps(_sync_payload()))
    digest = hashlib.sha256(sync_path.read_bytes()).hexdigest()
    calibration_mode = _sync_capture_mode()
    runtime_mode = _sync_capture_mode(
        audio_index=7, hall_port="/dev/ttyACM0",
        device_name="Fixture USB Audio (hw:7,0)",
    )
    manifest = {"sync": {
        "calibration_path": "sync.json",
        "calibration_sha256": digest,
        "offset_definition": "force_t_minus_audio_t",
        "offset_units": "ms",
        "audio_to_force_offset_ms": 12.5,
        "offset_std_ms": 1.0,
        "n_matched": 4,
        "unique_matches": True,
        "measurement_method": "physical_diaphragm_taps",
        "capture_mode": calibration_mode,
        "runtime_capture_mode": runtime_mode,
        "identity_match": _sync_identity_match(calibration_mode, runtime_mode),
        "applied": False,
    }, "audio": {
        "device": "fixture-audio", "channel": 1,
        "hardware_descriptor": runtime_mode["audio"],
    }, "force": {
        "port": "/dev/ttyACM0", "calibration_sha256": "f" * 64,
        "hardware_descriptor": runtime_mode["hall"],
    }}
    meta = conv.verify_sync_metadata(tmp_path, manifest)
    assert conv.apply_audio_sync(np.array([1.0, 2.0]), meta).tolist() == pytest.approx([1.0125, 2.0125])

    manifest["sync"]["applied"] = True
    with pytest.raises(ValueError, match="already applied"):
        conv.verify_sync_metadata(tmp_path, manifest)
    manifest["sync"]["applied"] = False
    manifest["sync"]["calibration_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        conv.verify_sync_metadata(tmp_path, manifest)
    manifest["sync"]["calibration_sha256"] = digest
    manifest["sync"]["offset_std_ms"] = 8.0
    with pytest.raises(ValueError, match="quality"):
        conv.verify_sync_metadata(tmp_path, manifest)


def test_sync_capture_mode_must_match_file_and_session_devices(tmp_path):
    sync_path = tmp_path / "sync.json"
    sync_path.write_text(json.dumps(_sync_payload()))
    calibration_mode = _sync_capture_mode()
    runtime_mode = _sync_capture_mode()
    runtime_mode["audio"]["usb_serial"] = "DIFFERENT-AUDIO"
    manifest = {
        "sync": {
            **_sync_payload(),
            "calibration_path": "sync.json",
            "calibration_sha256": hashlib.sha256(sync_path.read_bytes()).hexdigest(),
            "offset_units": "ms",
            "runtime_capture_mode": runtime_mode,
            "identity_match": _sync_identity_match(calibration_mode, runtime_mode),
            "applied": False,
        },
        "audio": {
            "device": "fixture-audio", "channel": 1,
            "hardware_descriptor": runtime_mode["audio"],
        },
        "force": {
            "port": "/dev/fixture-hall", "calibration_sha256": "f" * 64,
            "hardware_descriptor": runtime_mode["hall"],
        },
    }
    with pytest.raises(ValueError, match="sync capture_mode audio usb_serial"):
        conv.verify_sync_metadata(tmp_path, manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hostapi_name", "Core Audio"),
        ("max_input_channels", 4),
    ],
)
def test_sync_usb_remap_still_requires_audio_hostapi_and_input_capacity(field, value):
    calibration = _sync_capture_mode()
    runtime = _sync_capture_mode(
        audio_index=7, device_name="Fixture USB Audio (hw:7,0)",
    )
    runtime["audio"][field] = value
    with pytest.raises(ValueError, match=f"sync capture_mode audio {field}"):
        conv._match_sync_capture_modes(calibration, runtime)


def test_sync_usb_remap_still_requires_selected_audio_channel():
    calibration = _sync_capture_mode()
    runtime = _sync_capture_mode(audio_index=7)
    runtime["audio_channel"] = 0
    with pytest.raises(ValueError, match="audio_channel"):
        conv._match_sync_capture_modes(calibration, runtime)


def test_force_calibration_hash_validation_resolves_relative_and_absolute_paths(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    relative_cal = session / "calibrations" / "force.json"
    relative_cal.parent.mkdir()
    relative_cal.write_text(json.dumps({"calibrated": True, "hard_limit_force_n": 5.5}))
    relative_hash = hashlib.sha256(relative_cal.read_bytes()).hexdigest()
    manifest = {"force": {
        "calibrated": True,
        "calibration_path": "calibrations/force.json",
        "calibration_sha256": relative_hash,
    }}

    relative_meta = conv.verify_force_calibration_metadata(session, manifest)
    assert Path(relative_meta["calibration_path_resolved"]) == relative_cal.resolve()
    assert relative_meta["calibration_sha256"] == relative_hash

    absolute_cal = tmp_path / "machine_force_calibration.json"
    absolute_cal.write_text(json.dumps({"calibrated": True, "hard_limit_force_n": 5.5}))
    absolute_hash = hashlib.sha256(absolute_cal.read_bytes()).hexdigest()
    manifest["force"].update({
        "calibration_path": str(absolute_cal),
        "calibration_sha256": absolute_hash,
    })
    absolute_meta = conv.verify_force_calibration_metadata(session, manifest)
    assert Path(absolute_meta["calibration_path_resolved"]) == absolute_cal.resolve()


def test_force_calibration_validation_rejects_escape_tamper_and_post_verify_change(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    outside = tmp_path / "outside_force.json"
    outside.write_text(json.dumps({"calibrated": True}))
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest = {"force": {
        "calibrated": True,
        "calibration_path": "../outside_force.json",
        "calibration_sha256": digest,
    }}
    with pytest.raises(ValueError, match="escapes session directory"):
        conv.verify_force_calibration_metadata(session, manifest)

    inside = session / "force.json"
    inside.write_text(json.dumps({"calibrated": True, "revision": 1}))
    manifest["force"].update({
        "calibration_path": "force.json",
        "calibration_sha256": "0" * 64,
    })
    with pytest.raises(ValueError, match="force calibration hash mismatch"):
        conv.verify_force_calibration_metadata(session, manifest)
    manifest["force"]["calibration_sha256"] = hashlib.sha256(inside.read_bytes()).hexdigest()
    verified = conv.verify_force_calibration_metadata(session, manifest)
    inside.write_text(json.dumps({"calibrated": True, "revision": 2}))
    _write_snapshot_root_files(session)
    segments = tmp_path / "segments.jsonl"
    segments.write_text('{"segment_id":"s0"}\n')
    with pytest.raises(ValueError, match="verified provenance source hash mismatch"):
        conv.snapshot_session_sources(
            session,
            tmp_path / "dataset",
            generated_segments=segments,
            extra_sources={
                "calibrations/force.json": conv.VerifiedSource(
                    Path(verified["calibration_path_resolved"]),
                    verified["calibration_sha256"],
                ),
            },
        )


def test_device_clock_fit_has_numeric_acceptance_thresholds():
    device_ms = np.arange(20, dtype=np.float64) * 110.0
    host_s = 10.0 + device_ms * 0.001001
    fit = conv.fit_device_clock_checked(host_s, device_ms, stream="force")
    assert fit["accepted"] is True
    assert fit["drift_ppm"] == pytest.approx(1_000.0)

    noisy = host_s.copy(); noisy[10] += 0.050
    with pytest.raises(ValueError, match="force device clock fit rejected"):
        conv.fit_device_clock_checked(noisy, device_ms, stream="force")


def test_manifest_validation_rejects_rehearsal_data(monkeypatch):
    # Phase 1 intentionally has no training-eligible robot schema. This test
    # isolates the independent rehearsal and manifest-structure checks.
    monkeypatch.setattr(
        conv, "require_exact_robot_training_eligibility", lambda _summary: None,
    )
    artifact_validation, artifact_hashes = _artifact_contract()
    manifest = {
        "schema_version": "wam_session_v2",
        "promotion_contract_version": "wam_promotion_v2",
        "status": "complete", "training_eligible": True,
        "capture_kind": "production_teleop_v4",
        "rehearsal_overrides": [], "streams": ["force", "audio", "cam", "robot"],
        "arm": _exact_arm_metadata(),
        "robot": _synthetic_v4_robot_summary_for_gate_isolation(),
        "force": {"calibrated": True},
        "artifact_validation": artifact_validation,
        "artifact_hashes": artifact_hashes,
        "camera": {
            "backend": "orbbec", "camera_id": "fixture-camera",
            "camera_id_source": "verified_orbbec_serial", "has_depth": True,
        },
    }
    conv.validate_manifest_for_conversion(manifest)
    del manifest["promotion_contract_version"]
    with pytest.raises(ValueError, match="promotion_contract_version"):
        conv.validate_manifest_for_conversion(manifest)
    manifest["promotion_contract_version"] = "wam_promotion_v2"
    manifest["rehearsal_overrides"] = ["uncalibrated_force"]
    with pytest.raises(ValueError, match="rehearsal"):
        conv.validate_manifest_for_conversion(manifest)
    manifest["rehearsal_overrides"] = []
    manifest["camera"]["camera_id_source"] = "device_index"
    with pytest.raises(ValueError, match="serial-verifiable"):
        conv.validate_manifest_for_conversion(manifest)


def test_manifest_validation_requires_finalizer_artifact_evidence(monkeypatch):
    # Exercise the artifact gate independently of Phase 1's robot stop gate.
    monkeypatch.setattr(
        conv, "require_exact_robot_training_eligibility", lambda _summary: None,
    )
    artifact_validation, artifact_hashes = _artifact_contract()
    manifest = {
        "schema_version": "wam_session_v2",
        "promotion_contract_version": "wam_promotion_v2", "status": "complete",
        "training_eligible": True, "rehearsal_overrides": [],
        "capture_kind": "production_teleop_v4",
        "streams": ["force", "audio", "cam", "robot"],
        "arm": _exact_arm_metadata(),
        "robot": _synthetic_v4_robot_summary_for_gate_isolation(),
        "force": {"calibrated": True},
        "camera": {
            "backend": "orbbec", "camera_id": "fixture-camera",
            "camera_id_source": "verified_orbbec_serial", "has_depth": True,
        },
        "artifact_validation": artifact_validation,
        "artifact_hashes": artifact_hashes,
    }
    del manifest["artifact_hashes"]
    with pytest.raises(ValueError, match="artifact hash"):
        conv.validate_manifest_for_conversion(manifest)

    manifest["artifact_hashes"] = artifact_hashes
    del manifest["artifact_validation"]["depth"]
    with pytest.raises(ValueError, match="artifact validation"):
        conv.validate_manifest_for_conversion(manifest)

    manifest["artifact_validation"] = _artifact_contract()[0]
    manifest["artifact_validation"]["depth"]["n_frames"] = 0
    with pytest.raises(ValueError, match="artifact validation"):
        conv.validate_manifest_for_conversion(manifest)


def test_reward_spans_are_split_at_every_target_boundary():
    assert conv.split_spans_at_boundaries([(0.0, 10.0)], [4.0, 7.0]) == [
        (0.0, 4.0), (4.0, 7.0), (7.0, 10.0),
    ]


def test_reward_interval_normalization_only_closes_rounding_scale_edges():
    spans = [
        conv.RewardSpan(1.00001, 1.40, 0.2),
        conv.RewardSpan(1.40, 1.99999, 0.8),
    ]
    covered = conv._cover_reward_interval(spans, 1.0, 2.0)
    assert [(span.t_start, span.t_end) for span in covered] == [
        (1.0, 1.4), (1.4, 2.0),
    ]
    assert [span.reward for span in covered] == [0.2, 0.8]


def test_reward_interval_never_extends_contact_reward_across_noncontact_gap():
    spans = [
        conv.RewardSpan(1.0, 2.0, 0.9),
        conv.RewardSpan(2.4, 3.0, 0.0),
    ]

    with pytest.raises(ValueError, match="gap"):
        conv._cover_reward_interval(spans, 1.0, 3.0)


def test_canonical_conversion_rejects_the_unpinned_proxy_q_path(tmp_path):
    force = conv.ForceStream(
        t_host_s=np.asarray([1.0], dtype=np.float64),
        force_n=np.asarray([4.0], dtype=np.float64),
        status=("OK",),
        device_t_ms=np.asarray([1.0], dtype=np.float64),
    )
    targets = conv.TargetStream(
        t_s=np.asarray([1.0], dtype=np.float64),
        names=("R1",),
        uv=np.asarray([[0.5, 0.5]], dtype=np.float32),
        xyz=((0.1, 0.2, 0.3),),
        xyz_frames=("robot_base",),
    )

    with pytest.raises(ValueError, match="activated quality model.*proxy"):
        conv._build_target_split_segments(
            tmp_path,
            force,
            targets,
            {"audio_to_force_offset_ms": 0.0},
        )


def test_lerobot_033_writer_passes_task_separately_and_honors_done_boundaries():
    class Dataset:
        def __init__(self):
            self.calls = []; self.episodes = 0

        def add_frame(self, frame, *, task):
            assert "task" not in frame
            self.calls.append((dict(frame), task))

        def save_episode(self):
            self.episodes += 1

    dataset = Dataset()
    records = [
        conv.CausalFrame({"next.done": np.array([False])}, "auscultate the R1", "R1"),
        conv.CausalFrame({"next.done": np.array([True])}, "auscultate the R1", "R1"),
        conv.CausalFrame({"next.done": np.array([True])}, "auscultate the L1", "L1"),
    ]
    conv.write_lerobot_033(dataset, records)
    assert [task for _, task in dataset.calls] == [
        "auscultate the R1", "auscultate the R1", "auscultate the L1",
    ]
    assert dataset.episodes == 2


def test_camera_timeline_applies_hashed_transport_latency_after_drift_fit(tmp_path):
    device = np.arange(20, dtype=float) * 110.0
    host = 10.0 + device * 0.001001
    lines = ["frame_index,host_arrival_t_s,device_t_ms,time_source"]
    lines += [f"{i},{h:.9f},{d:.3f},orbbec_device_and_host_arrival"
              for i, (h, d) in enumerate(zip(host, device))]
    (tmp_path / "rgb_times.csv").write_text("\n".join(lines) + "\n")
    latency_path = tmp_path / "camera_latency.json"
    latency_path.write_text(json.dumps({
        "schema_version": "wam_camera_latency_v1",
        "latency_definition": "host_arrival_minus_exposure",
        "capture_mode": _camera_capture_mode(),
        "latency_ms": 12.0,
        "latency_std_ms": 1.0,
        "n_samples": 20,
        "measurement_method": "hardware_led_flash",
    }))
    digest = hashlib.sha256(latency_path.read_bytes()).hexdigest()
    manifest = {"camera": {"backend": "orbbec", "camera_id": "fixture-camera", "timing": {
        "calibration_path": "camera_latency.json", "calibration_sha256": digest,
        "latency_definition": "host_arrival_minus_exposure", "latency_ms": 12.0,
        "latency_std_ms": 1.0, "n_samples": 20,
        "measurement_method": "hardware_led_flash",
        "capture_mode": _camera_capture_mode(),
        "mode_validation": _camera_mode_validation(),
        "applied": False,
    }}}
    timeline = conv.read_camera_timeline(tmp_path, manifest)
    assert timeline.t_exposure_s[0] == pytest.approx(9.988, abs=1e-6)
    assert timeline.clock_fit["accepted"] is True
    manifest["camera"]["timing"]["capture_mode"]["color_format"] = "BGR"
    with pytest.raises(ValueError, match="capture_mode.*does not match"):
        conv.read_camera_timeline(tmp_path, manifest)
    manifest["camera"]["timing"]["capture_mode"]["color_format"] = "RGB"
    manifest["camera"]["timing"]["mode_validation"]["negotiated_mode"][
        "frame_aggregate_mode"
    ] = "disabled"
    with pytest.raises(ValueError, match="negotiated pipeline"):
        conv.read_camera_timeline(tmp_path, manifest)
    manifest["camera"]["timing"]["mode_validation"]["negotiated_mode"][
        "frame_aggregate_mode"
    ] = "full_frame_require"
    manifest["camera"]["timing"]["latency_std_ms"] = 8.0
    with pytest.raises(ValueError, match="quality"):
        conv.read_camera_timeline(tmp_path, manifest)


def test_camera_timeline_uses_host_arrival_for_avfoundation_without_inventing_device_clock(
    tmp_path,
):
    host = 10.0 + np.arange(20, dtype=float) * 0.11
    lines = ["frame_index,host_arrival_t_s,device_t_ms,time_source"]
    lines += [f"{i},{t:.9f},,opencv_host_arrival" for i, t in enumerate(host)]
    (tmp_path / "rgb_times.csv").write_text("\n".join(lines) + "\n")
    latency_path = tmp_path / "camera_latency.json"
    latency_path.write_text(json.dumps({
        "schema_version": "wam_camera_latency_v1",
        "latency_definition": "host_arrival_minus_exposure",
        "capture_mode": _camera_capture_mode(
            backend="avfoundation", camera_id="avf-0",
        ),
        "latency_ms": 18.0,
        "latency_std_ms": 1.0,
        "n_samples": 20,
        "measurement_method": "hardware_led_flash",
    }))
    digest = hashlib.sha256(latency_path.read_bytes()).hexdigest()
    manifest = {"camera": {
        "backend": "avfoundation", "camera_id": "avf-0", "timing": {
        "calibration_path": "camera_latency.json", "calibration_sha256": digest,
        "latency_definition": "host_arrival_minus_exposure", "latency_ms": 18.0,
        "latency_std_ms": 1.0, "n_samples": 20,
        "measurement_method": "hardware_led_flash", "applied": False,
            "capture_mode": _camera_capture_mode(
                backend="avfoundation", camera_id="avf-0",
            ),
            "mode_validation": _camera_mode_validation(
                "avfoundation", _camera_capture_mode(
                    backend="avfoundation", camera_id="avf-0",
                ),
            ),
        }}}

    timeline = conv.read_camera_timeline(tmp_path, manifest)

    assert timeline.t_exposure_s.tolist() == pytest.approx((host - 0.018).tolist())
    assert timeline.device_t_ms is None
    assert timeline.clock_fit is None

    payload = json.loads(latency_path.read_text())
    payload["capture_mode"]["backend"] = "orbbec"
    latency_path.write_text(json.dumps(payload))
    manifest["camera"]["timing"]["calibration_sha256"] = hashlib.sha256(
        latency_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="capture_mode"):
        conv.read_camera_timeline(tmp_path, manifest)

    payload["capture_mode"]["backend"] = "avfoundation"
    payload["capture_mode"]["time_source"] = "orbbec_device_and_host_arrival"
    latency_path.write_text(json.dumps(payload))
    manifest["camera"]["timing"]["calibration_sha256"] = hashlib.sha256(
        latency_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="capture_mode"):
        conv.read_camera_timeline(tmp_path, manifest)


def test_source_snapshot_copies_video_depth_generated_segments_and_detects_conflicts(tmp_path):
    session = tmp_path / "session_demo"; session.mkdir()
    _write_snapshot_root_files(session)
    (session / "depth").mkdir()
    (session / "depth" / "depth_000001.npy").write_bytes(b"depth")
    segments = tmp_path / "segments.jsonl"; segments.write_text('{"segment_id":"s0"}\n')
    root = tmp_path / "dataset"

    provenance = conv.snapshot_session_sources(session, root, generated_segments=segments)
    snap = root / "source_sessions" / "session_demo"
    assert (snap / "rgb.mp4").read_bytes() == (session / "rgb.mp4").read_bytes()
    assert (snap / "depth" / "depth_000001.npy").read_bytes() == b"depth"
    assert (snap / "segments.jsonl").read_bytes() == segments.read_bytes()
    assert provenance["files"]["rgb.mp4"]["sha256"] == hashlib.sha256(
        (session / "rgb.mp4").read_bytes()).hexdigest()
    assert (snap / "rgb.mp4").stat().st_ino != (session / "rgb.mp4").stat().st_ino

    (snap / "rgb.mp4").write_bytes(b"conflict")
    with pytest.raises(FileExistsError, match="different content"):
        conv.snapshot_session_sources(session, root, generated_segments=segments)


def test_source_snapshot_rejects_raw_artifact_changed_after_promotion(tmp_path):
    session = tmp_path / "session_demo"
    session.mkdir()
    for name in (
        "manifest.json", "audio.wav", "rgb.mp4", "force.csv",
        "rgb_times.csv", "obs.csv", "targets.jsonl", *conv.EXACT_ROBOT_FILES,
    ):
        (session / name).write_bytes(f"promoted:{name}".encode())
    depth = session / "depth"
    depth.mkdir()
    (depth / "depth_times.csv").write_bytes(b"depth-timeline")
    (depth / "depth_000000.npy").write_bytes(b"depth-frame")
    expected = {
        path.relative_to(session).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in session.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }

    (session / "robot_exact_control.csv").write_bytes(b"modified after promotion")

    with pytest.raises(ValueError, match="verified provenance source hash mismatch"):
        conv.snapshot_session_sources(
            session, tmp_path / "dataset", expected_source_hashes=expected,
        )


def test_source_snapshot_binds_the_exact_discovery_manifest_bytes(tmp_path):
    session = tmp_path / "session_demo"
    session.mkdir()
    for name in (
        "manifest.json", "audio.wav", "rgb.mp4", "force.csv",
        "rgb_times.csv", "obs.csv", "targets.jsonl", *conv.EXACT_ROBOT_FILES,
    ):
        (session / name).write_bytes(f"discovery:{name}".encode())
    discovery_manifest_hash = hashlib.sha256(
        (session / "manifest.json").read_bytes()
    ).hexdigest()
    (session / "manifest.json").write_bytes(b"changed between discovery and snapshot")

    with pytest.raises(ValueError, match="verified provenance source hash mismatch"):
        conv.snapshot_session_sources(
            session, tmp_path / "dataset",
            expected_manifest_sha256=discovery_manifest_hash,
        )


def test_finalization_seal_rejects_manifest_changed_after_promotion(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(_canonical_json({
        "schema_version": "wam_session_v2",
        "promotion_contract_version": "wam_promotion_v2",
    }))
    authenticity = tmp_path / "TRAINING_AUTHENTICITY.json"
    authenticity.write_bytes(_canonical_json({
        "schema_version": "wam_training_authenticity_v2",
    }))
    original_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "FINALIZATION.json").write_bytes(_canonical_json({
        "schema_version": "wam_session_finalization_v2",
        "promotion_contract_version": "wam_promotion_v2",
        "manifest_sha256": original_hash,
        "training_authenticity_sha256": hashlib.sha256(
            authenticity.read_bytes()
        ).hexdigest(),
    }))
    manifest.write_text('{"modified": true}\n')

    with pytest.raises(ValueError, match="finalization seal.*manifest hash"):
        conv._verify_finalization_seal(tmp_path)


def test_snapshot_persists_replayable_manifest_and_preserves_original(tmp_path):
    session = tmp_path / "session_demo"
    session.mkdir()
    calibrations = {}
    for name in ("force", "sync", "camera_latency"):
        path = session / f"{name}.json"
        path.write_text(json.dumps({"name": name}))
        calibrations[name] = path
    original_manifest = {
        "force": {
            "calibration_path": str(calibrations["force"]),
            "calibration_sha256": hashlib.sha256(calibrations["force"].read_bytes()).hexdigest(),
        },
        "sync": {
            "calibration_path": "sync.json",
            "calibration_sha256": hashlib.sha256(calibrations["sync"].read_bytes()).hexdigest(),
        },
        "camera": {"timing": {
            "calibration_path": "camera_latency.json",
            "calibration_sha256": hashlib.sha256(
                calibrations["camera_latency"].read_bytes()
            ).hexdigest(),
        }},
    }
    original_bytes = json.dumps(original_manifest).encode()
    for name in ("audio.wav", "rgb.mp4", "force.csv",
                 "rgb_times.csv", "obs.csv", "targets.jsonl", *conv.EXACT_ROBOT_FILES):
        (session / name).write_bytes(f"source:{name}".encode())
    (session / "manifest.json").write_bytes(original_bytes)
    authenticity = session / "TRAINING_AUTHENTICITY.json"
    authenticity.write_bytes(_canonical_json({
        "schema_version": "wam_training_authenticity_v2",
    }))
    finalization = session / "FINALIZATION.json"
    finalization.write_bytes(_canonical_json({
        "schema_version": "wam_session_finalization_v2",
        "promotion_contract_version": "wam_promotion_v2",
        "manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "training_authenticity_sha256": hashlib.sha256(
            authenticity.read_bytes()
        ).hexdigest(),
    }))
    root = tmp_path / "snapshot_root"
    sources = {
        "calibrations/force.json": conv.VerifiedSource(
            calibrations["force"], original_manifest["force"]["calibration_sha256"],
        ),
        "calibrations/sync.json": conv.VerifiedSource(
            calibrations["sync"], original_manifest["sync"]["calibration_sha256"],
        ),
        "calibrations/camera_latency.json": conv.VerifiedSource(
            calibrations["camera_latency"],
            original_manifest["camera"]["timing"]["calibration_sha256"],
        ),
        "FINALIZATION.original.json": conv.VerifiedSource(
            finalization, hashlib.sha256(finalization.read_bytes()).hexdigest(),
        ),
        "TRAINING_AUTHENTICITY.json": conv.VerifiedSource(
            authenticity, hashlib.sha256(authenticity.read_bytes()).hexdigest(),
        ),
    }
    conv.snapshot_session_sources(session, root, extra_sources=sources)
    snapshot = root / "source_sessions" / session.name

    replay_manifest = conv._manifest_bound_to_snapshot(original_manifest)
    conv._persist_replayable_snapshot_manifest(snapshot, replay_manifest)

    persisted = json.loads((snapshot / "manifest.json").read_text())
    assert persisted["force"]["calibration_path"] == "calibrations/force.json"
    assert persisted["sync"]["calibration_path"] == "calibrations/sync.json"
    assert persisted["camera"]["timing"]["calibration_path"] == (
        "calibrations/camera_latency.json"
    )
    assert (snapshot / "manifest.original.json").read_bytes() == original_bytes
    conv._verify_finalization_seal(snapshot)
    provenance = json.loads((snapshot / "provenance.json").read_text())
    for relative in (
        "manifest.json", "manifest.original.json", "FINALIZATION.json",
        "FINALIZATION.original.json",
    ):
        assert provenance["files"][relative]["sha256"] == hashlib.sha256(
            (snapshot / relative).read_bytes()
        ).hexdigest()


def test_source_snapshot_rejects_an_unverified_extra_source(tmp_path):
    session = tmp_path / "session_demo"
    session.mkdir()
    _write_snapshot_root_files(session)
    segments = tmp_path / "segments.jsonl"
    segments.write_text('{"segment_id":"s0"}\n')
    force_cal = session / "force_calibration.json"
    force_cal.write_text(json.dumps({"calibrated": True}))

    with pytest.raises(ValueError, match="extra provenance sources must be verified"):
        conv.snapshot_session_sources(
            session,
            tmp_path / "dataset",
            generated_segments=segments,
            extra_sources={"calibrations/force.json": force_cal},
        )


def test_source_snapshot_rejects_destination_overlapping_source_session(tmp_path):
    session = tmp_path / "session_demo"
    session.mkdir()
    _write_snapshot_root_files(session)
    segments = tmp_path / "segments.jsonl"
    segments.write_text('{"segment_id":"s0"}\n')

    with pytest.raises(ValueError, match="overlap"):
        conv.snapshot_session_sources(session, session, generated_segments=segments)


def test_prepare_records_marks_target_change_as_done_without_cross_episode_transition():
    robot = conv.RobotStream(
        t_s=np.array([1.0, 1.1, 1.2]), state=np.zeros((3, 2), np.float32),
        action=np.ones((3, 2), np.float32), joint_names=("j1", "j2"),
    )
    force = conv.ForceStream(
        t_host_s=np.array([1.0, 1.1, 1.2]), force_n=np.array([4.0, 4.0, 4.0], np.float32),
        status=("OK", "OK", "OK"), device_t_ms=None,
    )
    targets = conv.TargetStream(
        t_s=np.array([0.9, 1.15]), names=("R1", "L1"),
        uv=np.array([[0.2, 0.3], [0.7, 0.3]], np.float32),
        xyz=((0.1, 0.2, 0.3), (0.2, 0.2, 0.3)),
        xyz_frames=("robot_base", "robot_base"),
    )
    segments = [conv.RewardSpan(0.9, 1.15, 0.5), conv.RewardSpan(1.15, 1.3, 0.8)]
    records = conv.prepare_causal_records(
        np.array([1.0, 1.1, 1.2]), robot, force, targets, segments,
        robot_max_age_s=0.11, force_max_age_s=0.11, target_max_age_s=0.3,
    )
    assert [r.target_name for r in records] == ["R1", "R1", "L1"]
    assert [bool(r.frame["next.done"].item()) for r in records] == [False, True, True]
    assert [float(r.frame["next.reward"].item()) for r in records] == pytest.approx([0.5, 0.5, 0.8])


def test_canonical_build_rejects_legacy_promotion_v1_before_snapshot(
    tmp_path, monkeypatch,
):
    session = tmp_path / "legacy-session"
    session.mkdir()
    (session / "manifest.json").write_bytes(_canonical_json({
        "promotion_contract_version": "wam_promotion_v1",
        "rehearsal_overrides": [],
        "schema_version": "wam_session_v2",
        "session_id": session.name,
        "status": "complete",
        "training_eligible": True,
    }))
    snapshot_called = False

    def forbidden_snapshot(*_args, **_kwargs):
        nonlocal snapshot_called
        snapshot_called = True
        raise AssertionError("legacy session reached snapshot")

    monkeypatch.setattr(conv, "snapshot_session_sources", forbidden_snapshot)
    root = tmp_path / "dataset"
    registry = tmp_path / "empty-trusted-finalizations.json"
    registry.write_bytes(_canonical_json({
        "records": [],
        "schema_version": "wam_trusted_training_finalization_registry_v1",
    }))
    with pytest.raises(ValueError, match="training promotion v2 verification failed"):
        conv.build(
            session, "ausc/legacy", root, dataset_cls=object,
            trusted_finalization_registry=registry,
            trusted_snapshot_root=tmp_path / "trusted-snapshots",
        )
    assert not snapshot_called
    assert not root.exists()
