from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager
from typing import Optional

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import artifact_contract as contract  # noqa: E402


_DEPTH_FRAME = "depth/depth_000001.npy"
_GEOMETRY_MODEL_INPUT = "geometry/model_inputs/%s.npy" % ("a" * 64)
_BACK_EVIDENCE = "geometry/evidence_arrays/back_%s.npy" % ("b" * 64)
_ROBOT_EVIDENCE = "geometry/evidence_arrays/robot_%s.npy" % ("c" * 64)
_QUALITY_MODEL_INPUT = (
    "quality_model_inputs/input_00000000000000000001.f32le"
)


def _training_session(
    root: Path, session_id: str = "session-001", *,
    replace_audio_metadata: bool = False,
    swap_audio_force_bindings: bool = False,
) -> Path:
    session = root / session_id
    audio_metadata_name = (
        "audio_meta.json" if replace_audio_metadata else "audio_capture.json"
    )
    raw_bodies = {
        "audio.wav": b"real-audio-pcm",
        audio_metadata_name: b"{\"status\":\"sensors_finalized\"}\n",
        "audio_input_times.jsonl": b"{\"input_start\":0,\"input_end\":16}\n",
        "audio_times.jsonl": b"{\"sample_start\":0,\"sample_end\":16}\n",
        "force.csv": b"t,raw,force_n\n1,100,0.1\n",
        "force_capture.json": b"{\"status\":\"sensors_finalized\"}\n",
        "FORCE_CAPTURE_COMPLETE.json": b"{\"status\":\"complete\"}\n",
        "force_calibration.json": b"{\"calibrated\":true}\n",
        "rgb.mp4": b"real-video",
        "rgb_times.csv": b"frame_index,t\n1,1.0\n",
        "obs.csv": b"t,force_n\n1,0.1\n",
        "robot_control_envelopes.jsonl": b"{\"tick\":1}\n",
        "robot_exact_control.csv": b"real-robot-state",
        "robot_frame_index.jsonl": b"{\"frame\":1}\n",
        "robot_capture_summary.json": b"{\"schema\":\"v4\"}\n",
        "targets.jsonl": b"{\"point\":1}\n",
        "tracker.jsonl": b"{\"frame\":1}\n",
        "depth/depth_times.csv": b"frame_index,t\n1,1.0\n",
        _DEPTH_FRAME: b"real-depth",
        "quality_inference_packets.jsonl": b"{\"q\":0.4}\n{\"q\":0.6}\n",
        "quality_inference_authenticity.json": b"{\"status\":\"PASS\"}\n",
        _QUALITY_MODEL_INPUT: b"lossless-q-model-input",
        "geometry_frame_packets.jsonl": b"{\"frame\":1}\n{\"frame\":2}\n",
        "geometry_frame_authenticity.json": b"{\"status\":\"PASS\"}\n",
        "geometry/patient_depth_evidence.jsonl": b"{\"frame\":1}\n",
        "geometry/patient_depth_authenticity.json": b"{\"status\":\"PASS\"}\n",
        _GEOMETRY_MODEL_INPUT: b"lossless-geometry-model-input",
        _BACK_EVIDENCE: b"lossless-back-mask",
        _ROBOT_EVIDENCE: b"lossless-robot-mask",
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
        "robot": [
            "robot_control_envelopes.jsonl", "robot_exact_control.csv",
            "robot_frame_index.jsonl", "robot_capture_summary.json",
        ],
        "force": [
            "force.csv", "force_capture.json", "FORCE_CAPTURE_COMPLETE.json",
            "force_calibration.json",
        ],
        "audio": [
            "audio.wav", audio_metadata_name, "audio_input_times.jsonl",
            "audio_times.jsonl",
        ],
        "geometry": [
            "rgb.mp4", "rgb_times.csv", "obs.csv", "targets.jsonl",
            "tracker.jsonl",
            "depth/depth_times.csv", _DEPTH_FRAME,
            "geometry_frame_packets.jsonl", "geometry_frame_authenticity.json",
            "geometry/patient_depth_evidence.jsonl",
            "geometry/patient_depth_authenticity.json", _GEOMETRY_MODEL_INPUT,
            _BACK_EVIDENCE, _ROBOT_EVIDENCE,
        ],
        "quality": [
            "audio.wav", "audio_times.jsonl", "force.csv",
            "quality_inference_packets.jsonl",
            "quality_inference_authenticity.json", _QUALITY_MODEL_INPUT,
        ],
    }
    if swap_audio_force_bindings:
        component_raw["audio"], component_raw["force"] = (
            component_raw["force"], component_raw["audio"]
        )
    component_refs = {}
    for component, relatives in component_raw.items():
        receipt = {
            "component": component,
            "evidence": {"test_fixture": True},
            "policy_version": component + "_authenticity_v1",
            "raw_artifacts": {
                relative: raw_files[relative]["sha256"] for relative in relatives
            },
            "schema_version": "wam_training_component_authenticity_v2",
            "status": "PASS",
            "training_signal_eligible": True,
            "verifier": {
                "code_release_id": "code-" + "1" * 20,
                "module_path": "auscultation/verifiers/" + component + ".py",
                "module_sha256": "2" * 64,
                "policy_sha256": "3" * 64,
            },
        }
        receipt_path = session / "authenticity" / (component + ".json")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_raw = contract.canonical_json_bytes(receipt)
        receipt_path.write_bytes(receipt_raw)
        component_refs[component] = {
            "path": receipt_path.relative_to(session).as_posix(),
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
        }
    authenticity = {
        "components": component_refs,
        "finalizer": {
            "code_manifest_sha256": "1" * 64,
            "code_release_id": "code-" + "1" * 20,
            "environment_id": "runtime-" + "5" * 20,
            "environment_lock_sha256": "5" * 64,
            "module_sha256": "7" * 64,
        },
        "promotion_contract_version": "wam_promotion_v2",
        "raw_artifact_inventory_sha256": hashlib.sha256(
            contract.canonical_json_bytes(inventory)
        ).hexdigest(),
        "schema_version": "wam_training_authenticity_v2",
        "session_id": session.name,
        "status": "PASS",
        "training_eligible": True,
    }
    authenticity_raw = contract.canonical_json_bytes(authenticity)
    (session / "TRAINING_AUTHENTICITY.json").write_bytes(authenticity_raw)
    authenticity_sha = hashlib.sha256(authenticity_raw).hexdigest()
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
            "sha256": authenticity_sha,
        },
        "training_eligible": True,
    }
    manifest_raw = contract.canonical_json_bytes(manifest)
    (session / "manifest.json").write_bytes(manifest_raw)
    (session / "FINALIZATION.json").write_bytes(contract.canonical_json_bytes({
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "promotion_contract_version": "wam_promotion_v2",
        "schema_version": "wam_session_finalization_v2",
        "training_authenticity_sha256": authenticity_sha,
    }))
    return session


def _trusted_finalization_record(session: Path) -> dict:
    """Model an out-of-session record loaded from a deployment-owned registry."""
    authenticity_raw = (session / contract.TRAINING_AUTHENTICITY_NAME).read_bytes()
    authenticity = json.loads(authenticity_raw)
    manifest_raw = (session / contract.TRAINING_SESSION_MANIFEST_NAME).read_bytes()
    finalization_raw = (session / contract.TRAINING_SESSION_FINALIZATION_NAME).read_bytes()
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
        "promotion_contract_version": contract.TRAINING_PROMOTION_CONTRACT,
        "schema_version": "wam_trusted_training_finalization_v1",
        "session_id": session.name,
    }


def _verify_training_session(
    session: Path, *, trusted: Optional[dict] = None,
) -> dict:
    return contract.verify_training_session_contract(
        session,
        trusted_finalization_record=(
            _trusted_finalization_record(session) if trusted is None else trusted
        ),
        trusted_snapshot_root=session.parent / "trusted-training-snapshots",
    )


def _reseal_training_metadata(session: Path) -> None:
    """Rebind a deliberately mutated fixture as if a buggy finalizer signed it."""
    manifest_path = session / "manifest.json"
    authenticity_path = session / "TRAINING_AUTHENTICITY.json"
    manifest = json.loads(manifest_path.read_text())
    authenticity = json.loads(authenticity_path.read_text())
    for component, reference in authenticity["components"].items():
        receipt_raw = (session / reference["path"]).read_bytes()
        reference["sha256"] = hashlib.sha256(receipt_raw).hexdigest()
    authenticity["raw_artifact_inventory_sha256"] = hashlib.sha256(
        contract.canonical_json_bytes(manifest["artifact_hashes"])
    ).hexdigest()
    authenticity_raw = contract.canonical_json_bytes(authenticity)
    authenticity_path.write_bytes(authenticity_raw)
    authenticity_sha = hashlib.sha256(authenticity_raw).hexdigest()
    manifest["training_authenticity"]["sha256"] = authenticity_sha
    manifest_raw = contract.canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_raw)
    (session / "FINALIZATION.json").write_bytes(contract.canonical_json_bytes({
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "promotion_contract_version": "wam_promotion_v2",
        "schema_version": "wam_session_finalization_v2",
        "training_authenticity_sha256": authenticity_sha,
    }))


def _remove_raw_binding(session: Path, relative: str) -> None:
    (session / relative).unlink()
    manifest = json.loads((session / "manifest.json").read_text())
    manifest["artifact_hashes"]["files"].pop(relative)
    (session / "manifest.json").write_bytes(contract.canonical_json_bytes(manifest))
    for path in (session / "authenticity").glob("*.json"):
        receipt = json.loads(path.read_text())
        receipt["raw_artifacts"].pop(relative, None)
        path.write_bytes(contract.canonical_json_bytes(receipt))
    _reseal_training_metadata(session)


def _rename_raw_binding(session: Path, old: str, new: str) -> None:
    destination = session / new
    destination.parent.mkdir(parents=True, exist_ok=True)
    (session / old).rename(destination)
    manifest = json.loads((session / "manifest.json").read_text())
    files = manifest["artifact_hashes"]["files"]
    files[new] = files.pop(old)
    (session / "manifest.json").write_bytes(contract.canonical_json_bytes(manifest))
    for path in (session / "authenticity").glob("*.json"):
        receipt = json.loads(path.read_text())
        if old in receipt["raw_artifacts"]:
            receipt["raw_artifacts"][new] = receipt["raw_artifacts"].pop(old)
            path.write_bytes(contract.canonical_json_bytes(receipt))
    _reseal_training_metadata(session)


def _add_raw_binding(
    session: Path, relative: str, body: bytes, *, components: tuple[str, ...],
) -> None:
    path = session / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    manifest = json.loads((session / "manifest.json").read_text())
    manifest["artifact_hashes"]["files"][relative] = {
        "sha256": digest,
        "size_bytes": len(body),
    }
    (session / "manifest.json").write_bytes(contract.canonical_json_bytes(manifest))
    for component in components:
        receipt_path = session / "authenticity" / (component + ".json")
        receipt = json.loads(receipt_path.read_text())
        receipt["raw_artifacts"][relative] = digest
        receipt_path.write_bytes(contract.canonical_json_bytes(receipt))
    _reseal_training_metadata(session)


def _move_component_binding(
    session: Path, relative: str, *, source: str, destination: str,
) -> None:
    source_path = session / "authenticity" / (source + ".json")
    destination_path = session / "authenticity" / (destination + ".json")
    source_receipt = json.loads(source_path.read_text())
    destination_receipt = json.loads(destination_path.read_text())
    digest = source_receipt["raw_artifacts"].pop(relative)
    destination_receipt["raw_artifacts"][relative] = digest
    source_path.write_bytes(contract.canonical_json_bytes(source_receipt))
    destination_path.write_bytes(contract.canonical_json_bytes(destination_receipt))
    _reseal_training_metadata(session)


def _source(root: Path, body: bytes = b"hello\n") -> Path:
    (root / "auscultation" / "ausc").mkdir(parents=True)
    path = root / "auscultation" / "ausc" / "worker.py"
    path.write_bytes(body)
    path.chmod(0o640)
    return path


def _release(tmp_path: Path, *, body: bytes = b"hello\n") -> tuple[Path, dict]:
    source = tmp_path / "source"
    _source(source, body)
    manifest = contract.build_manifest(
        source,
        ["auscultation/ausc/worker.py"],
        artifact_kind="code",
        source_role="mac",
    )
    release = tmp_path / "release"
    contract.write_release(source, manifest, release, target_role="test")
    return release, manifest


def test_manifest_is_canonical_and_location_independent(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _source(left)
    _source(right)

    one = contract.build_manifest(
        left,
        ["auscultation/ausc/worker.py"],
        artifact_kind="code",
        source_role="mac",
    )
    two = contract.build_manifest(
        right,
        ["auscultation/ausc/worker.py"],
        artifact_kind="code",
        source_role="mac",
    )

    assert one == two
    assert one["release_id"] == f"code-{one['payload_sha256'][:20]}"
    canonical = contract.canonical_json_bytes(one["payload"])
    assert canonical.endswith(b"\n")
    assert b" " not in canonical
    assert hashlib.sha256(canonical).hexdigest() == one["payload_sha256"]
    assert one["payload"]["files"] == [{
        "mode": "0640",
        "path": "auscultation/ausc/worker.py",
        "sha256": hashlib.sha256(b"hello\n").hexdigest(),
        "size": 6,
    }]


@pytest.mark.parametrize("bad", ["../escape", "a/../../escape", "/absolute", "", "."])
def test_manifest_rejects_unsafe_paths(tmp_path: Path, bad: str) -> None:
    _source(tmp_path)
    with pytest.raises(contract.ArtifactError):
        contract.build_manifest(tmp_path, [bad], artifact_kind="code", source_role="mac")


@pytest.mark.parametrize("bad", ["a//b", "a/./b", "a\\b"])
def test_manifest_rejects_noncanonical_paths(tmp_path: Path, bad: str) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").write_text("present")
    with pytest.raises(contract.ArtifactError):
        contract.build_manifest(tmp_path, [bad], artifact_kind="code", source_role="mac")


@pytest.mark.parametrize("field,value", [
    ("artifact_kind", "../code"),
    ("artifact_kind", "code;touch-pwned"),
    ("source_role", "mac\nother"),
])
def test_manifest_rejects_unsafe_identifiers(tmp_path: Path, field: str, value: str) -> None:
    _source(tmp_path)
    kwargs = {"artifact_kind": "code", "source_role": "mac"}
    kwargs[field] = value
    with pytest.raises(contract.ArtifactError):
        contract.build_manifest(
            tmp_path,
            ["auscultation/ausc/worker.py"],
            **kwargs,
        )


def test_manifest_rejects_symlinked_files_and_parents(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = _source(source)
    (source / "linked.py").symlink_to(target)
    with pytest.raises(contract.ArtifactError, match="symlink"):
        contract.build_manifest(source, ["linked.py"], artifact_kind="code", source_role="mac")

    other = tmp_path / "other"
    _source(other)
    (source / "linked-dir").symlink_to(other / "auscultation")
    with pytest.raises(contract.ArtifactError, match="symlink"):
        contract.build_manifest(
            source,
            ["linked-dir/ausc/worker.py"],
            artifact_kind="code",
            source_role="mac",
        )


def test_release_marker_binds_verified_manifest(tmp_path: Path) -> None:
    release, manifest = _release(tmp_path)

    result = contract.verify_release(release)
    marker_bytes = (release / contract.COMPLETE_NAME).read_bytes()
    marker = json.loads(marker_bytes)
    manifest_bytes = (release / contract.MANIFEST_NAME).read_bytes()

    assert result["release_id"] == manifest["release_id"]
    assert marker["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert marker["payload_sha256"] == manifest["payload_sha256"]
    assert marker["verified_files"] == 1
    assert marker["target_role"] == "test"


def test_copy_hashes_and_copies_the_same_no_follow_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    payload = _source(source)
    secret = tmp_path / "secret"
    secret.write_bytes(b"must-not-be-copied")
    manifest = contract.build_manifest(
        source,
        ["auscultation/ausc/worker.py"],
        artifact_kind="code",
        source_role="mac",
    )
    original_open = contract._open_regular_fd

    @contextmanager
    def swap_after_open(root: Path, relative: str):
        with original_open(root, relative) as descriptor:
            payload.unlink()
            payload.symlink_to(secret)
            yield descriptor

    monkeypatch.setattr(contract, "_open_regular_fd", swap_after_open)
    release = tmp_path / "release"
    contract.write_release(source, manifest, release, target_role="test")
    assert (release / "auscultation" / "ausc" / "worker.py").read_bytes() == b"hello\n"


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed", "mode", "symlink"])
def test_exact_verification_rejects_mutations(tmp_path: Path, mutation: str) -> None:
    release, _ = _release(tmp_path)
    payload = release / "auscultation" / "ausc" / "worker.py"
    if mutation == "missing":
        payload.unlink()
    elif mutation == "extra":
        (release / "extra.txt").write_text("extra")
    elif mutation == "changed":
        payload.write_bytes(b"changed")
    elif mutation == "mode":
        payload.chmod(0o600)
    elif mutation == "symlink":
        payload.unlink()
        payload.symlink_to(release / contract.MANIFEST_NAME)

    with pytest.raises(contract.VerificationError):
        contract.verify_release(release)


def test_incomplete_release_is_rejected(tmp_path: Path) -> None:
    release, _ = _release(tmp_path)
    (release / contract.COMPLETE_NAME).unlink()
    with pytest.raises(contract.VerificationError, match="completion"):
        contract.verify_release(release)


def test_install_is_idempotent_and_activation_is_atomic(tmp_path: Path) -> None:
    source_release, manifest = _release(tmp_path)
    releases = tmp_path / "installed" / "releases"
    current = tmp_path / "installed" / "current"

    first = contract.install_release(source_release, releases, current, target_role="h200")
    second = contract.install_release(source_release, releases, current, target_role="h200")

    assert first == second == releases / manifest["release_id"]
    assert current.is_symlink()
    assert current.resolve() == first.resolve()
    assert not list(releases.glob(".staging-*"))


def test_conflicting_existing_release_is_rejected(tmp_path: Path) -> None:
    source_release, manifest = _release(tmp_path)
    releases = tmp_path / "installed" / "releases"
    current = tmp_path / "installed" / "current"
    conflict = releases / manifest["release_id"]
    conflict.mkdir(parents=True)
    (conflict / "unexpected").write_text("do not overwrite")

    with pytest.raises(contract.ConflictError):
        contract.install_release(source_release, releases, current, target_role="h200")
    assert (conflict / "unexpected").read_text() == "do not overwrite"


def test_training_promotion_v2_verifies_the_complete_authenticity_chain(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)

    verified = _verify_training_session(session)

    assert verified["promotion_contract_version"] == "wam_promotion_v2"
    assert verified["session_id"] == "session-001"
    assert verified["verified_components"] == [
        "audio", "force", "geometry", "quality", "robot",
    ]
    assert verified["verified_raw_files"] == 29
    assert len(verified["raw_payload_sha256"]) == 64
    assert Path(verified["verified_snapshot_path"]).parent == (
        tmp_path / "trusted-training-snapshots"
    )


def test_cached_training_snapshot_never_substitutes_for_the_current_source(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)
    trusted = _trusted_finalization_record(session)
    first = _verify_training_session(session, trusted=trusted)
    assert Path(first["verified_snapshot_path"]).is_dir()

    unrelated = tmp_path / "session-garbage"
    unrelated.mkdir()
    (unrelated / "garbage.bin").write_bytes(b"not the trusted session")

    with pytest.raises(contract.VerificationError, match="training"):
        contract.verify_training_session_contract(
            unrelated,
            trusted_finalization_record=trusted,
            trusted_snapshot_root=tmp_path / "trusted-training-snapshots",
        )


def test_training_promotion_v2_rejects_self_consistent_pass_without_trusted_record(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)

    with pytest.raises(
        contract.VerificationError,
        match="trusted finalization record",
    ):
        contract.verify_training_session_contract(session)


def test_trusted_finalization_registry_loads_one_exact_external_record(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)
    trusted = _trusted_finalization_record(session)
    registry = tmp_path / "TRAINING_FINALIZATIONS.json"
    registry.write_bytes(contract.canonical_json_bytes({
        "records": [trusted],
        "schema_version": "wam_trusted_training_finalization_registry_v1",
    }))

    by_session = contract.load_trusted_training_finalization_record(
        registry, session_id=session.name,
    )
    by_finalization = contract.load_trusted_training_finalization_record(
        registry,
        finalization_sha256=trusted["artifact_sha256"]["finalization"],
    )

    assert by_session == trusted
    assert by_finalization == trusted


@pytest.mark.parametrize("mutation", ["unexpected_field", "noncanonical_json", "duplicate"])
def test_trusted_finalization_registry_rejects_ambiguous_or_inexact_metadata(
    tmp_path: Path, mutation: str,
) -> None:
    session = _training_session(tmp_path)
    trusted = _trusted_finalization_record(session)
    registry_value = {
        "records": [trusted],
        "schema_version": "wam_trusted_training_finalization_registry_v1",
    }
    if mutation == "unexpected_field":
        registry_value["producer_supplied"] = True
        raw = contract.canonical_json_bytes(registry_value)
    elif mutation == "noncanonical_json":
        raw = json.dumps(registry_value, indent=2).encode("utf-8")
    else:
        registry_value["records"].append(trusted)
        raw = contract.canonical_json_bytes(registry_value)
    registry = tmp_path / "TRAINING_FINALIZATIONS.json"
    registry.write_bytes(raw)

    with pytest.raises(contract.VerificationError, match="registry"):
        contract.load_trusted_training_finalization_record(
            registry, session_id=session.name,
        )


def test_training_promotion_v2_rejects_resealed_forgery_against_prior_trusted_record(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)
    trusted = _trusted_finalization_record(session)
    forged_audio = b"attacker-controlled-but-internally-consistent-audio"
    (session / "audio.wav").write_bytes(forged_audio)
    manifest = json.loads((session / "manifest.json").read_text())
    manifest["artifact_hashes"]["files"]["audio.wav"] = {
        "sha256": hashlib.sha256(forged_audio).hexdigest(),
        "size_bytes": len(forged_audio),
    }
    (session / "manifest.json").write_bytes(contract.canonical_json_bytes(manifest))
    for component in ("audio", "quality"):
        receipt_path = session / "authenticity" / (component + ".json")
        receipt = json.loads(receipt_path.read_text())
        receipt["raw_artifacts"]["audio.wav"] = hashlib.sha256(
            forged_audio
        ).hexdigest()
        receipt_path.write_bytes(contract.canonical_json_bytes(receipt))
    _reseal_training_metadata(session)

    with pytest.raises(
        contract.VerificationError,
        match="trusted.*(manifest|authenticity|finalization)",
    ):
        _verify_training_session(session, trusted=trusted)


@pytest.mark.parametrize(
    ("identity", "field", "forged_value"),
    [
        ("finalizer", "code_manifest_sha256", "4" * 64),
        ("finalizer", "environment_lock_sha256", "6" * 64),
        ("audio", "code_release_id", "code-" + "9" * 20),
    ],
)
def test_trusted_record_requires_content_addressed_deployment_identities(
    tmp_path: Path, identity: str, field: str, forged_value: str,
) -> None:
    session = _training_session(tmp_path)
    trusted = _trusted_finalization_record(session)
    target = (
        trusted["finalizer"]
        if identity == "finalizer"
        else trusted["component_verifiers"][identity]
    )
    target[field] = forged_value

    with pytest.raises(
        contract.VerificationError,
        match="trusted.*(content-addressed|code release)",
    ):
        contract._verify_trusted_training_finalization_record(trusted)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("manifest", "bind manifest"),
        ("training_authenticity", "bind authenticity"),
        ("finalization", "bind finalization"),
    ],
)
def test_trusted_record_binds_each_session_artifact_digest(
    tmp_path: Path, field: str, message: str,
) -> None:
    session = _training_session(tmp_path)
    trusted = _trusted_finalization_record(session)
    trusted["artifact_sha256"][field] = "8" * 64

    with pytest.raises(contract.VerificationError, match=message):
        _verify_training_session(session, trusted=trusted)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("session_id", "session identity"),
        ("finalizer_module", "finalizer identity"),
        ("component_policy", "component verifier identity"),
    ],
)
def test_trusted_record_binds_session_and_static_verifier_identities(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    session = _training_session(tmp_path)
    trusted = _trusted_finalization_record(session)
    if mutation == "session_id":
        trusted["session_id"] = "session-elsewhere"
    elif mutation == "finalizer_module":
        trusted["finalizer"]["module_sha256"] = "8" * 64
    else:
        trusted["component_verifiers"]["audio"]["policy_sha256"] = "8" * 64

    with pytest.raises(contract.VerificationError, match=message):
        _verify_training_session(session, trusted=trusted)


@pytest.mark.parametrize("mutation", ["rewrite_raw", "add_late_extra"])
def test_training_promotion_success_is_bound_to_a_stable_controlled_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    session = _training_session(tmp_path)
    trusted = _trusted_finalization_record(session)
    original_audio = (session / "audio.wav").read_bytes()
    original_verify_raw = contract._verify_training_raw_files
    snapshot_verification_passes = 0

    def mutate_source_after_second_snapshot_raw_pass(
        root: Path, files: object,
    ) -> dict[str, str]:
        nonlocal snapshot_verification_passes
        result = original_verify_raw(root, files)
        if Path(root).parent == tmp_path / "trusted-training-snapshots":
            snapshot_verification_passes += 1
            if snapshot_verification_passes == 2:
                if mutation == "rewrite_raw":
                    (session / "audio.wav").write_bytes(b"late-source-rewrite")
                else:
                    (session / "late-extra.bin").write_bytes(b"late-source-extra")
        return result

    monkeypatch.setattr(
        contract,
        "_verify_training_raw_files",
        mutate_source_after_second_snapshot_raw_pass,
    )

    verified = _verify_training_session(session, trusted=trusted)

    snapshot = Path(verified["verified_snapshot_path"])
    assert snapshot_verification_passes == 2
    assert snapshot != session
    assert snapshot.name == (
        "training_session-" + verified["trusted_finalization_record_sha256"][:20]
    )
    assert (snapshot / "audio.wav").read_bytes() == original_audio
    assert not (snapshot / "late-extra.bin").exists()


def test_training_promotion_v2_requires_the_concrete_raw_stream_set(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path, replace_audio_metadata=True)

    with pytest.raises(contract.VerificationError, match="required raw streams"):
        _verify_training_session(session)


@pytest.mark.parametrize("relative", [
    "audio_input_times.jsonl",
    "audio_times.jsonl",
    "geometry/patient_depth_evidence.jsonl",
    "geometry/patient_depth_authenticity.json",
    _DEPTH_FRAME,
    _GEOMETRY_MODEL_INPUT,
    _BACK_EVIDENCE,
    _ROBOT_EVIDENCE,
    _QUALITY_MODEL_INPUT,
])
def test_training_promotion_v2_requires_each_lossless_raw_category(
    tmp_path: Path, relative: str,
) -> None:
    session = _training_session(tmp_path)
    _remove_raw_binding(session, relative)

    with pytest.raises(contract.VerificationError, match="required raw streams"):
        _verify_training_session(session)


@pytest.mark.parametrize(("old", "new"), [
    (_DEPTH_FRAME, "depth/depth_1.npy"),
    (_GEOMETRY_MODEL_INPUT, "geometry/model_inputs/not-a-sha.npy"),
    (_BACK_EVIDENCE, "geometry/evidence_arrays/back_%s.npy" % ("b" * 63)),
    (_ROBOT_EVIDENCE, "geometry/evidence_arrays/robot_%s.npy" % ("C" * 64)),
    (_QUALITY_MODEL_INPUT, "quality_model_inputs/input_0001.f32le"),
])
def test_training_promotion_v2_rejects_noncanonical_dynamic_raw_names(
    tmp_path: Path, old: str, new: str,
) -> None:
    session = _training_session(tmp_path)
    _rename_raw_binding(session, old, new)

    with pytest.raises(contract.VerificationError, match="canonical dynamic raw"):
        _verify_training_session(session)


def test_training_promotion_v2_rejects_malformed_extra_in_dynamic_directory(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)
    _add_raw_binding(
        session,
        "geometry/model_inputs/%s.bin" % ("d" * 64),
        b"wrong-extension",
        components=("geometry",),
    )

    with pytest.raises(contract.VerificationError, match="canonical dynamic raw"):
        _verify_training_session(session)


def test_training_promotion_v2_accepts_multiple_canonical_dynamic_raw_files(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)
    _add_raw_binding(
        session,
        "geometry/model_inputs/%s.npy" % ("d" * 64),
        b"second-geometry-input",
        components=("geometry",),
    )
    _add_raw_binding(
        session,
        "quality_model_inputs/input_00000000000000000002.f32le",
        b"second-quality-input",
        components=("quality",),
    )

    verified = _verify_training_session(session)

    assert verified["verified_raw_files"] == 31


@pytest.mark.parametrize(("relative", "source", "destination"), [
    ("audio_input_times.jsonl", "audio", "geometry"),
    ("audio_times.jsonl", "audio", "quality"),
    ("geometry/patient_depth_evidence.jsonl", "geometry", "audio"),
    ("geometry/patient_depth_authenticity.json", "geometry", "audio"),
])
def test_training_promotion_v2_requires_component_specific_static_bindings(
    tmp_path: Path, relative: str, source: str, destination: str,
) -> None:
    session = _training_session(tmp_path)
    _move_component_binding(
        session, relative, source=source, destination=destination,
    )

    with pytest.raises(contract.VerificationError, match="component raw requirements"):
        _verify_training_session(session)


def test_geometry_receipt_must_bind_every_dynamic_geometry_raw_file(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)
    second = "geometry/model_inputs/%s.npy" % ("d" * 64)
    _add_raw_binding(
        session, second, b"second-geometry-input", components=("audio",),
    )

    with pytest.raises(contract.VerificationError, match="component raw requirements"):
        _verify_training_session(session)


def test_quality_receipt_must_bind_every_lossless_q_model_input(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)
    second = "quality_model_inputs/input_00000000000000000002.f32le"
    _add_raw_binding(
        session, second, b"second-quality-input", components=("robot",),
    )

    with pytest.raises(contract.VerificationError, match="component raw requirements"):
        _verify_training_session(session)


def test_training_promotion_v2_rejects_dynamic_raw_path_traversal_even_if_resealed(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest["artifact_hashes"]["files"]
    files["geometry/model_inputs/../escaped.npy"] = files.pop(
        _GEOMETRY_MODEL_INPUT
    )
    manifest_path.write_bytes(contract.canonical_json_bytes(manifest))
    _reseal_training_metadata(session)

    with pytest.raises(contract.VerificationError, match="path is unsafe"):
        _verify_training_session(session)


def test_training_promotion_v2_rejects_symlinked_dynamic_raw(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)
    path = session / _GEOMETRY_MODEL_INPUT
    body = path.read_bytes()
    path.unlink()
    outside = tmp_path / "outside-model-input.npy"
    outside.write_bytes(body)
    path.symlink_to(outside)

    with pytest.raises(contract.VerificationError, match="unsafe"):
        _verify_training_session(session)


def test_training_promotion_v2_rejects_uninventoried_dynamic_extra_file(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path)
    extra = session / "quality_model_inputs/input_00000000000000000002.f32le"
    extra.write_bytes(b"unbound-extra")

    with pytest.raises(contract.VerificationError, match="exact file set"):
        _verify_training_session(session)


def test_training_promotion_v2_rejects_cross_component_raw_misbinding(
    tmp_path: Path,
) -> None:
    session = _training_session(tmp_path, swap_audio_force_bindings=True)

    with pytest.raises(contract.VerificationError, match="component raw requirements"):
        _verify_training_session(session)


def test_training_promotion_v2_rejects_cross_phase_raw_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _training_session(tmp_path)
    original = contract._read_regular_bytes
    changed = False

    def mutate_after_raw_verification(root: Path, relative: str) -> bytes:
        nonlocal changed
        raw = original(root, relative)
        if relative == contract.TRAINING_AUTHENTICITY_NAME and not changed:
            changed = True
            (Path(root) / "audio.wav").write_bytes(b"changed-after-first-hash")
        return raw

    monkeypatch.setattr(contract, "_read_regular_bytes", mutate_after_raw_verification)

    with pytest.raises(contract.VerificationError, match="changed during verification"):
        _verify_training_session(session)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("old_contract", "promotion"),
        ("component_failed", "component"),
        ("component_receipt_changed", "component receipt"),
        ("raw_changed", "raw artifact"),
        ("authenticity_changed", "authenticity"),
        ("missing_component", "component"),
        ("extra_file", "exact file set"),
    ],
)
def test_training_promotion_v2_fails_closed_on_any_broken_binding(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    session = _training_session(tmp_path)
    if mutation == "old_contract":
        manifest = json.loads((session / "manifest.json").read_text())
        manifest["promotion_contract_version"] = "wam_promotion_v1"
        (session / "manifest.json").write_bytes(contract.canonical_json_bytes(manifest))
    elif mutation == "component_failed":
        path = session / "authenticity" / "force.json"
        receipt = json.loads(path.read_text())
        receipt["status"] = "FAIL"
        path.write_bytes(contract.canonical_json_bytes(receipt))
    elif mutation == "component_receipt_changed":
        path = session / "authenticity" / "audio.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "raw_changed":
        (session / "audio.wav").write_bytes(b"silently-corrupted")
    elif mutation == "authenticity_changed":
        path = session / "TRAINING_AUTHENTICITY.json"
        authenticity = json.loads(path.read_text())
        authenticity["training_eligible"] = False
        path.write_bytes(contract.canonical_json_bytes(authenticity))
    elif mutation == "missing_component":
        path = session / "TRAINING_AUTHENTICITY.json"
        authenticity = json.loads(path.read_text())
        authenticity["components"].pop("quality")
        raw = contract.canonical_json_bytes(authenticity)
        path.write_bytes(raw)
        manifest = json.loads((session / "manifest.json").read_text())
        manifest["training_authenticity"]["sha256"] = hashlib.sha256(raw).hexdigest()
        manifest_raw = contract.canonical_json_bytes(manifest)
        (session / "manifest.json").write_bytes(manifest_raw)
        seal = json.loads((session / "FINALIZATION.json").read_text())
        seal["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
        seal["training_authenticity_sha256"] = hashlib.sha256(raw).hexdigest()
        (session / "FINALIZATION.json").write_bytes(
            contract.canonical_json_bytes(seal)
        )
    elif mutation == "extra_file":
        (session / "unbound.txt").write_text("must not be ignored")

    with pytest.raises(contract.VerificationError, match=message):
        _verify_training_session(session)


def test_write_release_cannot_overwrite_destination_created_during_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _source(source)
    manifest = contract.build_manifest(
        source,
        ["auscultation/ausc/worker.py"],
        artifact_kind="code",
        source_role="mac",
    )
    destination = tmp_path / "release"
    real_rename = contract._rename_noreplace

    def race(source_path: Path, destination_path: Path) -> None:
        destination_path.mkdir()
        (destination_path / "interloper").write_text("preserve me")
        real_rename(source_path, destination_path)

    monkeypatch.setattr(contract, "_rename_noreplace", race)
    with pytest.raises(contract.ConflictError):
        contract.write_release(source, manifest, destination, target_role="test")
    assert (destination / "interloper").read_text() == "preserve me"


def test_descriptor_relative_release_write_survives_parent_path_swap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _source(source)
    manifest = contract.build_manifest(
        source,
        ["auscultation/ausc/worker.py"],
        artifact_kind="simulation_code",
        source_role="mac",
    )
    destination_root = tmp_path / "releases"
    destination_root.mkdir()
    root_descriptor = os.open(
        str(destination_root), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    parked = tmp_path / "releases-parked"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination_root.rename(parked)
    destination_root.symlink_to(outside, target_is_directory=True)
    try:
        contract.write_release_at(
            source,
            manifest,
            root_descriptor,
            manifest["release_id"],
            target_role="linbox",
        )
        result = contract.verify_release_at(
            root_descriptor, manifest["release_id"],
        )
    finally:
        os.close(root_descriptor)

    assert result == manifest
    assert (parked / manifest["release_id"]).is_dir()
    assert list(outside.iterdir()) == []


def test_idempotent_install_rejects_wrong_target_role_marker(tmp_path: Path) -> None:
    source_release, manifest = _release(tmp_path)
    releases = tmp_path / "installed" / "releases"
    current = tmp_path / "installed" / "current"
    final = releases / manifest["release_id"]
    contract.write_release(source_release, manifest, final, target_role="linbox")

    with pytest.raises(contract.ConflictError, match="target role"):
        contract.install_release(source_release, releases, current, target_role="h200")


def test_failed_post_activation_restores_previous_pointer(tmp_path: Path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    current = tmp_path / "managed" / "current"
    current.parent.mkdir()
    current.symlink_to(old)
    source_release, _ = _release(tmp_path / "new")

    def fail(_path: Path) -> None:
        raise RuntimeError("post check failed")

    with pytest.raises(RuntimeError, match="post check"):
        contract.install_release(
            source_release,
            tmp_path / "managed" / "releases",
            current,
            target_role="h200",
            post_activate=fail,
        )

    assert current.resolve() == old.resolve()


def test_compare_and_swap_rejects_changed_pointer(tmp_path: Path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    current = tmp_path / "managed" / "current"
    current.parent.mkdir()
    current.symlink_to(old)
    source_release, _ = _release(tmp_path / "new")

    with pytest.raises(contract.ConflictError, match="compare-and-swap"):
        contract.install_release(
            source_release,
            tmp_path / "managed" / "releases",
            current,
            target_role="h200",
            expected_current="some-other-target",
        )
    assert current.resolve() == old.resolve()


def test_non_symlink_current_is_never_overwritten(tmp_path: Path) -> None:
    source_release, _ = _release(tmp_path / "new")
    current = tmp_path / "managed" / "current"
    current.mkdir(parents=True)

    with pytest.raises(contract.ConflictError, match="non-symlink"):
        contract.install_release(
            source_release,
            tmp_path / "managed" / "releases",
            current,
            target_role="h200",
        )


def test_deployment_lock_is_exclusive(tmp_path: Path) -> None:
    lock = tmp_path / "deploy.lock"
    with contract.deployment_lock(lock):
        with pytest.raises(contract.ConflictError, match="lock"):
            with contract.deployment_lock(lock):
                pass
    assert not lock.exists()
