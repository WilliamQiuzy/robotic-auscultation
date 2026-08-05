from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import artifact_contract as contract  # noqa: E402
import publish_perception_runtime as runtime  # noqa: E402


AUSC_ROOT = Path(__file__).resolve().parents[2]
LINBOX_LOCK = AUSC_ROOT / "configs" / "perception_runtime" / "linbox-ausc-live.lock.json"
V3_HARDENED_CODE_LOCK = (
    AUSC_ROOT / "configs" / "perception_runtime" / "hardened-code-sha256.json"
)
V4_HARDENED_CODE_LOCK = (
    AUSC_ROOT
    / "configs"
    / "perception_runtime"
    / "hardened-code-sha256-v4.json"
)
V5_HARDENED_CODE_LOCK = (
    AUSC_ROOT
    / "configs"
    / "perception_runtime"
    / "hardened-code-sha256-v5.json"
)
RELEASE_PROFILE = Path(__file__).resolve().parent / "release_profile.json"
POSTERIOR_GATE_PROFILE = (
    AUSC_ROOT / "configs" / "perception_runtime" / "posterior-gate-profile.json"
)
ENVIRONMENT_LOCK_V2_SCHEMA = (
    AUSC_ROOT
    / "configs"
    / "perception_runtime"
    / "perception-environment-lock-v2.schema.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_canonical(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contract.canonical_json_bytes(value))


def _code_release(
    root: Path, *, complete: bool = True, reviewed_semantics: bool = True,
    tampered_relative: str = "",
) -> tuple[Path, dict]:
    source = root / "code-source"
    relatives = ["auscultation/ausc/perception/tracking/runtime.py"]
    if complete:
        relatives = sorted(runtime.REQUIRED_HARDENED_CODE_FILES)
    for relative in relatives:
        worker = source / relative
        worker.parent.mkdir(parents=True, exist_ok=True)
        if relative == tampered_relative:
            worker.write_text("SELF_CONSISTENT_TAMPER = True\n", encoding="utf-8")
        elif reviewed_semantics:
            worker.write_bytes((AUSC_ROOT.parent / relative).read_bytes())
        else:
            worker.write_text("ENTRYPOINT = 'hardened'\n", encoding="utf-8")
    manifest = contract.build_manifest(
        source,
        relatives,
        artifact_kind="code_release",
        source_role="mac",
    )
    release = root / manifest["release_id"]
    contract.write_release(source, manifest, release, target_role="mac")
    return release, manifest


def _environment_lock(path: Path) -> dict:
    packages = [
        ("numpy", "1.26.4"),
        ("opencv-python", "4.11.0.86"),
        ("polars", "1.41.2"),
        ("polars-runtime-32", "1.41.2"),
        ("pyorbbecsdk2", "2.1.1"),
        ("scipy", "1.17.1"),
        ("torch", "2.11.0+cu128"),
        ("torchvision", "0.26.0+cu128"),
        ("ultralytics", "8.4.64"),
        ("ultralytics-thop", "2.0.20"),
        ("yourdfpy", "0.0.60"),
    ]
    value = {
        "schema": runtime.ENVIRONMENT_LOCK_SCHEMA,
        "environment_id": "linbox-perception-" + "f" * 20,
        "python": {
            "implementation": "CPython",
            "version": "3.11.15",
            "abi": "cp311",
            "platform": "linux-x86_64",
            "executable_sha256": "2" * 64,
        },
        "distribution_inventory": {
            "algorithm": runtime.DISTRIBUTION_INVENTORY_ALGORITHM,
            "count": 146,
            "sha256": "3" * 64,
        },
        "packages": [
            {
                "name": name,
                "version": version,
                "record_sha256": format(index + 4, "x") * 64,
            }
            for index, (name, version) in enumerate(packages)
        ],
        "modules": {
            "cv2": {"algorithm": runtime.MODULE_TREE_ALGORITHM, "count": 2, "sha256": "c" * 64},
            "pyorbbecsdk": {"algorithm": runtime.MODULE_TREE_ALGORITHM, "count": 1, "sha256": "d" * 64},
            "torch": {"algorithm": runtime.MODULE_TREE_ALGORITHM, "count": 3, "sha256": "e" * 64},
            "ultralytics": {"algorithm": runtime.MODULE_TREE_ALGORITHM, "count": 4, "sha256": "b" * 64},
        },
        "cuda": {
            "build": "12.8",
            "architectures": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
        },
        "provisioning": {
            "mode": "verified_clone",
            "source_environment_fingerprint_algorithm": (
                runtime.SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
            ),
            "source_environment_fingerprint_sha256": "f" * 64,
            "wheelhouse_complete": False,
        },
    }
    value["environment_id"] = (
        "linbox-perception-" + runtime.environment_identity_sha256(value)[:20]
    )
    _write_canonical(path, value)
    return value


def _hardened_hashes(code_manifest: dict) -> dict[str, str]:
    rows = code_manifest["payload"]["files"]
    by_path = {row["path"]: row["sha256"] for row in rows}
    return {
        relative: by_path[relative]
        for relative in sorted(runtime.REQUIRED_HARDENED_CODE_FILES)
    }


def test_v5_is_the_build_default_without_reinterpreting_frozen_v4() -> None:
    expected_v4_files_beyond_v3 = {
        "auscultation/ausc/realtime_q.py",
        "auscultation/ausc/perception/depth_codec.py",
        "auscultation/ausc/perception/production_runtime_activation.py",
        "auscultation/ausc/perception/robot/camera_exposure_evidence.py",
        "auscultation/ausc/perception/robot/controller_launch.py",
        "auscultation/ausc/perception/robot/frame_evidence_bundle.py",
        "auscultation/ausc/perception/robot/openarm_pair_activation.py",
        "auscultation/ausc/perception/robot/robot_capture_coordinator.py",
        "auscultation/ausc/perception/robot/robot_mask_provider.py",
        "auscultation/ausc/perception/robot/robot_session_capture.py",
        "auscultation/ausc/perception/robot/robot_socket_receiver.py",
        "auscultation/ausc/perception/robot/robot_state_stream.py",
        "auscultation/ausc/perception/robot/trusted_controller_companion.py",
        "auscultation/ausc/perception/tracking/back_mask_packet.py",
        "auscultation/ausc/perception/tracking/depth_noise_profile.py",
        "auscultation/ausc/perception/tracking/experimental_visibility_geometry.py",
        "auscultation/ausc/perception/tracking/geometry.py",
        "auscultation/ausc/perception/tracking/geometry_channel_packet.py",
        "auscultation/ausc/perception/tracking/occlusion_runtime.py",
        "auscultation/ausc/perception/tracking/patient_frame_evidence.py",
        "auscultation/ausc/perception/tracking/patient_pose_provider.py",
        "auscultation/ausc/perception/tracking/posterior_back_mask_provider.py",
        "auscultation/ausc/perception/tracking/production_evidence_capture.py",
        "auscultation/ausc/perception/tracking/production_geometry_gate.py",
        "auscultation/mechanical_sensor/force_feedback/force_capture.py",
        "auscultation/mechanical_sensor/force_feedback/force_sensor.py",
        "auscultation/scripts/perception/activate_handeye.py",
    }

    assert runtime.RUNTIME_MANIFEST_SCHEMA == "wam_perception_runtime_v5"
    assert runtime.V4_RUNTIME_MANIFEST_SCHEMA == "wam_perception_runtime_v4"
    assert runtime.V3_RUNTIME_MANIFEST_SCHEMA == "wam_perception_runtime_v3"
    assert runtime.V3_REQUIRED_HARDENED_CODE_FILES < (
        runtime.V4_REQUIRED_HARDENED_CODE_FILES
    )
    assert (
        runtime.V4_REQUIRED_HARDENED_CODE_FILES
        - runtime.V3_REQUIRED_HARDENED_CODE_FILES
    ) == expected_v4_files_beyond_v3
    assert (
        runtime.REQUIRED_HARDENED_CODE_FILES
        - runtime.V4_REQUIRED_HARDENED_CODE_FILES
    ) == {
        "auscultation/ausc/perception/robot/controller_launch_activation.py",
        "auscultation/scripts/deployment/artifact_contract.py",
        "auscultation/scripts/deployment/collection_service_unit.py",
        "auscultation/scripts/deployment/perception_environment_manifest.py",
        "auscultation/scripts/deployment/verify_perception_environment.py",
    }
    assert len(runtime.REQUIRED_HARDENED_CODE_FILES) == 46
    assert (
        "auscultation/scripts/deployment/publish_perception_runtime.py"
        not in runtime.REQUIRED_HARDENED_CODE_FILES
    )
    assert set(runtime.V4_REQUIRED_PRODUCTION_CHAIN_SEMANTIC_CODE_SHA256) == (
        runtime.V4_REQUIRED_HARDENED_CODE_FILES
    )
    assert set(runtime.REQUIRED_RAW_COLLECTION_RUNTIME_SEMANTIC_CODE_SHA256) == (
        runtime.REQUIRED_HARDENED_CODE_FILES
    )


def test_v4_hardened_contract_remains_frozen_and_schema_specific() -> None:
    expected = dict(runtime.V4_REQUIRED_PRODUCTION_CHAIN_SEMANTIC_CODE_SHA256)
    rows = [
        {"path": relative, "sha256": digest}
        for relative, digest in sorted(expected.items())
    ]

    assert runtime._verify_hardened_code_manifest(
        {"payload": {"files": rows}},
        expected,
        runtime_schema=runtime.V4_RUNTIME_MANIFEST_SCHEMA,
    ) == expected


def test_v3_hardened_contract_remains_frozen_and_schema_specific() -> None:
    assert runtime.V3_REQUIRED_HARDENED_CODE_FILES == frozenset({
        "auscultation/ausc/audio/audio_stream.py",
        "auscultation/ausc/models/iuv_model.py",
        "auscultation/ausc/perception/camera_stream.py",
        "auscultation/ausc/perception/tracking/__init__.py",
        "auscultation/ausc/perception/tracking/detector.py",
        "auscultation/ausc/perception/tracking/model_tracker.py",
        "auscultation/ausc/perception/tracking/runtime.py",
        "auscultation/ausc/record_session.py",
        "auscultation/scripts/collect_server.py",
        "auscultation/scripts/collection_platform.py",
        "auscultation/scripts/collection_supervisor.py",
        "auscultation/scripts/perception_capture.py",
        "auscultation/scripts/perception_worker.py",
        "auscultation/scripts/posterior_gate.py",
    })
    rows = []
    expected = {}
    for index, relative in enumerate(
        sorted(runtime.V3_REQUIRED_HARDENED_CODE_FILES), start=1,
    ):
        digest = runtime.V3_REQUIRED_BACK_GATE_SEMANTIC_CODE_SHA256.get(
            relative, format(index, "064x"),
        )
        expected[relative] = digest
        rows.append({"path": relative, "sha256": digest})

    assert runtime._verify_hardened_code_manifest(
        {"payload": {"files": rows}},
        expected,
        runtime_schema=runtime.V3_RUNTIME_MANIFEST_SCHEMA,
    ) == expected


def _inputs(tmp_path: Path) -> dict:
    checkpoint = tmp_path / "inputs" / "perc_lmh_v9_occ" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"exact-v9-checkpoint")
    posterior_gate_checkpoint = tmp_path / "inputs" / "yolo11x-pose.pt"
    posterior_gate_checkpoint.write_bytes(b"exact-yolo11x-pose-posterior-gate")
    dino_checkpoint = tmp_path / "inputs" / "dinov2_vitb14_pretrain.pth"
    dino_checkpoint.write_bytes(b"exact-dino-base")
    dino_repo = tmp_path / "inputs" / "facebookresearch_dinov2_main"
    (dino_repo / "dinov2").mkdir(parents=True)
    (dino_repo / "hubconf.py").write_text("def dinov2_vitb14(): pass\n", encoding="utf-8")
    (dino_repo / "dinov2" / "__init__.py").write_text("VERSION = 'fixture'\n", encoding="utf-8")
    code_release, code_manifest = _code_release(tmp_path / "inputs")
    environment_lock = tmp_path / "inputs" / "linbox-perception.lock.json"
    _environment_lock(environment_lock)
    return {
        "checkpoint": checkpoint,
        "posterior_gate_checkpoint": posterior_gate_checkpoint,
        "dino_checkpoint": dino_checkpoint,
        "dino_repo": dino_repo,
        "code_release": code_release,
        "code_manifest": code_manifest,
        "environment_lock": environment_lock,
    }


def _posterior_gate_kwargs(inputs: dict) -> dict:
    checkpoint = inputs["posterior_gate_checkpoint"]
    return {
        "posterior_gate_checkpoint": checkpoint,
        "expected_posterior_gate_checkpoint_sha256": _sha256(checkpoint),
        "expected_posterior_gate_checkpoint_size": checkpoint.stat().st_size,
    }


def _build_fixture_runtime(inputs: dict, releases_root: Path) -> Path:
    return runtime.build_perception_runtime_release(
        checkpoint=inputs["checkpoint"],
        expected_checkpoint_sha256=_sha256(inputs["checkpoint"]),
        expected_checkpoint_size=inputs["checkpoint"].stat().st_size,
        **_posterior_gate_kwargs(inputs),
        dino_checkpoint=inputs["dino_checkpoint"],
        expected_dino_checkpoint_sha256=_sha256(inputs["dino_checkpoint"]),
        expected_dino_checkpoint_size=inputs["dino_checkpoint"].stat().st_size,
        dino_repo=inputs["dino_repo"],
        expected_dino_repo_sha256=runtime.hash_dino_source_tree(
            inputs["dino_repo"],
        )["sha256"],
        code_release=inputs["code_release"],
        expected_code_release_id=inputs["code_manifest"]["release_id"],
        expected_code_payload_sha256=inputs["code_manifest"]["payload_sha256"],
        expected_hardened_code_sha256=_hardened_hashes(inputs["code_manifest"]),
        environment_lock=inputs["environment_lock"],
        expected_environment_lock_sha256=_sha256(inputs["environment_lock"]),
        releases_root=releases_root,
    )


def test_runtime_binds_box_only_person_detector_and_operator_confirmation(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)

    release = runtime.build_perception_runtime_release(
        checkpoint=inputs["checkpoint"],
        expected_checkpoint_sha256=_sha256(inputs["checkpoint"]),
        expected_checkpoint_size=inputs["checkpoint"].stat().st_size,
        **_posterior_gate_kwargs(inputs),
        dino_checkpoint=inputs["dino_checkpoint"],
        expected_dino_checkpoint_sha256=_sha256(inputs["dino_checkpoint"]),
        expected_dino_checkpoint_size=inputs["dino_checkpoint"].stat().st_size,
        dino_repo=inputs["dino_repo"],
        expected_dino_repo_sha256=runtime.hash_dino_source_tree(inputs["dino_repo"])["sha256"],
        code_release=inputs["code_release"],
        expected_code_release_id=inputs["code_manifest"]["release_id"],
        expected_code_payload_sha256=inputs["code_manifest"]["payload_sha256"],
        expected_hardened_code_sha256=_hardened_hashes(inputs["code_manifest"]),
        environment_lock=inputs["environment_lock"],
        expected_environment_lock_sha256=_sha256(inputs["environment_lock"]),
        releases_root=tmp_path / "runtime-releases",
    )

    semantic = json.loads((release / runtime.RUNTIME_MANIFEST_NAME).read_text())
    assert semantic["model"]["role"] == "sole_auscultation_point_source"
    assert semantic["posterior_gate"] == {
        "architecture": "YOLO11x-pose",
        "back_confirmation": "operator_required",
        "checkpoint_path": "model/posterior_gate/yolo11x-pose.pt",
        "checkpoint_sha256": _sha256(inputs["posterior_gate_checkpoint"]),
        "checkpoint_size": inputs["posterior_gate_checkpoint"].stat().st_size,
        "framework": "ultralytics",
        "module": "ultralytics",
        "orientation_authority": False,
        "person_presence_signal": "box_only",
        "produces_auscultation_points": False,
        "role": "single_person_presence_detector",
    }
    assert (
        release / "model" / "posterior_gate" / "yolo11x-pose.pt"
    ).read_bytes() == b"exact-yolo11x-pose-posterior-gate"
    assert runtime.offline_runtime_environment(release)[
        "AUSC_POSTERIOR_GATE_CHECKPOINT"
    ] == str(release / "model" / "posterior_gate" / "yolo11x-pose.pt")


def test_only_an_exactly_allowlisted_v2_runtime_can_be_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    v4_release = _build_fixture_runtime(inputs, tmp_path / "v4-releases")
    source = tmp_path / "legacy-source"
    shutil.copytree(v4_release, source)
    (source / contract.MANIFEST_NAME).unlink()
    (source / contract.COMPLETE_NAME).unlink()

    semantic_path = source / runtime.RUNTIME_MANIFEST_NAME
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["schema"] = runtime.LEGACY_RUNTIME_MANIFEST_SCHEMA
    semantic.pop("capabilities")
    semantic["code"]["hardened_file_sha256"] = {
        relative: semantic["code"]["hardened_file_sha256"][relative]
        for relative in sorted(runtime.V3_REQUIRED_HARDENED_CODE_FILES)
    }
    gate = semantic["posterior_gate"]
    semantic["posterior_gate"] = {
        "architecture": "YOLO11x-pose",
        "checkpoint_path": gate["checkpoint_path"],
        "checkpoint_sha256": gate["checkpoint_sha256"],
        "checkpoint_size": gate["checkpoint_size"],
        "framework": "ultralytics",
        "module": "ultralytics",
        "produces_auscultation_points": False,
        "role": "posterior_human_pose_gate",
    }
    _write_canonical(semantic_path, semantic)
    manifest = contract.build_manifest(
        source,
        runtime._regular_files(source),
        artifact_kind=runtime.RUNTIME_ARTIFACT_KIND,
        source_role="mac",
    )
    legacy_release = tmp_path / "legacy-releases" / manifest["release_id"]
    legacy_release.parent.mkdir()
    contract.write_release(
        source, manifest, legacy_release, target_role="mac",
    )

    with pytest.raises(
        runtime.PerceptionRuntimeError,
        match="not an approved rollback",
    ):
        runtime.verify_perception_runtime_release(legacy_release)

    monkeypatch.setitem(
        runtime.APPROVED_LEGACY_V2_ROLLBACKS,
        manifest["release_id"],
        {
            "payload_sha256": manifest["payload_sha256"],
            "code_release_id": semantic["code"]["release_id"],
            "code_payload_sha256": semantic["code"]["payload_sha256"],
        },
    )
    verified = runtime.verify_perception_runtime_release(legacy_release)
    assert verified["release_id"] == manifest["release_id"]


def test_builds_complete_content_addressed_perception_runtime(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    dino_tree_sha256 = runtime.hash_dino_source_tree(inputs["dino_repo"])["sha256"]

    release = runtime.build_perception_runtime_release(
        checkpoint=inputs["checkpoint"],
        expected_checkpoint_sha256=_sha256(inputs["checkpoint"]),
        expected_checkpoint_size=inputs["checkpoint"].stat().st_size,
        **_posterior_gate_kwargs(inputs),
        dino_checkpoint=inputs["dino_checkpoint"],
        expected_dino_checkpoint_sha256=_sha256(inputs["dino_checkpoint"]),
        expected_dino_checkpoint_size=inputs["dino_checkpoint"].stat().st_size,
        dino_repo=inputs["dino_repo"],
        expected_dino_repo_sha256=dino_tree_sha256,
        code_release=inputs["code_release"],
        expected_code_release_id=inputs["code_manifest"]["release_id"],
        expected_code_payload_sha256=inputs["code_manifest"]["payload_sha256"],
        expected_hardened_code_sha256=_hardened_hashes(inputs["code_manifest"]),
        environment_lock=inputs["environment_lock"],
        expected_environment_lock_sha256=_sha256(inputs["environment_lock"]),
        releases_root=tmp_path / "runtime-releases",
    )

    envelope = runtime.verify_perception_runtime_release(release)
    semantic = json.loads((release / runtime.RUNTIME_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert envelope["release_id"].startswith("perception_runtime-")
    assert release.name == envelope["release_id"]
    assert semantic["schema"] == runtime.RUNTIME_MANIFEST_SCHEMA
    assert semantic["capabilities"] == {
        "arm_control_authority": False,
        "controller_launch_state": "DORMANT_HIL_PENDING",
        "training_promotion": "NOT_ACTIVATED",
    }
    assert semantic["model"] == {
        "architecture": "NormalsDetector",
        "checkpoint_path": "model/perc_lmh_v9_occ/best.pt",
        "checkpoint_sha256": _sha256(inputs["checkpoint"]),
        "checkpoint_size": inputs["checkpoint"].stat().st_size,
        "encoder": "dinov2_vitb14",
        "half": True,
        "landmark_order": ["R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5"],
        "role": "sole_auscultation_point_source",
    }
    assert semantic["dino"]["repository_tree_sha256"] == dino_tree_sha256
    assert semantic["code"]["release_id"] == inputs["code_manifest"]["release_id"]
    assert semantic["environment"]["lock_path"] == "environment/linbox-perception.lock.json"
    assert semantic["runtime"]["torch_home"] == "torch_home"
    assert (
        release / "torch_home" / "hub" / "facebookresearch_dinov2_main" / "hubconf.py"
    ).is_file()
    assert (
        release / "torch_home" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth"
    ).is_file()
    assert runtime.offline_runtime_environment(release) == {
        "AUSC_DINOV2_CHECKPOINT": str(
            release / "torch_home" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth"
        ),
        "AUSC_DINOV2_REPO": str(
            release / "torch_home" / "hub" / "facebookresearch_dinov2_main"
        ),
        "AUSC_OFFLINE_RUNTIME": "1",
        "AUSC_POSTERIOR_GATE_CHECKPOINT": str(
            release / "model" / "posterior_gate" / "yolo11x-pose.pt"
        ),
        "HF_HUB_OFFLINE": "1",
        "TORCH_HOME": str(release / "torch_home"),
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert (release / "code" / contract.MANIFEST_NAME).is_file()
    assert contract.verify_release(release)["release_id"] == envelope["release_id"]


def test_v5_refuses_a_self_consistent_arm_authority_capability_tamper(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    release = _build_fixture_runtime(inputs, tmp_path / "runtime-releases")
    source = tmp_path / "tampered-source"
    shutil.copytree(release, source)
    (source / contract.MANIFEST_NAME).unlink()
    (source / contract.COMPLETE_NAME).unlink()

    semantic_path = source / runtime.RUNTIME_MANIFEST_NAME
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["capabilities"]["arm_control_authority"] = True
    _write_canonical(semantic_path, semantic)
    manifest = contract.build_manifest(
        source,
        runtime._regular_files(source),
        artifact_kind=runtime.RUNTIME_ARTIFACT_KIND,
        source_role="mac",
    )
    tampered_release = tmp_path / "tampered-releases" / manifest["release_id"]
    tampered_release.parent.mkdir()
    contract.write_release(
        source, manifest, tampered_release, target_role="mac",
    )

    with pytest.raises(runtime.PerceptionRuntimeError, match="capability boundary"):
        runtime.verify_perception_runtime_release(tampered_release)


def test_checked_in_environment_lock_matches_the_read_only_linbox_probe() -> None:
    raw = LINBOX_LOCK.read_bytes()
    value = json.loads(raw.decode("utf-8"))

    assert raw == contract.canonical_json_bytes(value)
    assert runtime.validate_environment_lock(value) == value
    assert runtime.is_strong_environment_lock(value) is False
    assert value["schema"] == runtime.ENVIRONMENT_LOCK_SCHEMA
    assert value["environment_id"] == "linbox-perception-b3aeceefe70ecfcb0655"
    assert value["python"]["executable_sha256"] == (
        "a95e8200921c9ad434b1a189f9bdfa7155a477035a8cb87979627dbed6532380"
    )
    assert value["distribution_inventory"] == {
        "algorithm": runtime.LEGACY_DISTRIBUTION_INVENTORY_ALGORITHM,
        "count": 149,
        "sha256": "b5e436a4917d559de591461ec188de9a7324c3dd27a1edd6e322dea0bf1023a7",
    }
    assert value["provisioning"]["source_environment_fingerprint_algorithm"] == (
        runtime.LEGACY_SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
    )
    assert value["provisioning"]["source_environment_fingerprint_sha256"] == (
        "e61e0276c598f129b24c72a5566bbd22f5a5effa6a39add347f613bb3ff782d6"
    )
    assert value["provisioning"]["wheelhouse_complete"] is False


def test_environment_lock_requires_exact_yolo_gate_runtime_profile(tmp_path: Path) -> None:
    lock_path = tmp_path / "environment.json"
    value = _environment_lock(lock_path)

    assert runtime.validate_environment_lock(value) == value
    assert {
        row["name"]: row["version"] for row in value["packages"]
    } | {} == runtime.SUPPORTED_PACKAGES
    assert set(value["modules"]) == {
        "cv2", "pyorbbecsdk", "torch", "ultralytics",
    }

    value["packages"] = [
        {
            **row,
            "version": "8.4.63" if row["name"] == "ultralytics" else row["version"],
        }
        for row in value["packages"]
    ]
    with pytest.raises(runtime.PerceptionRuntimeError, match="critical package profile"):
        runtime.validate_environment_lock(value)


def test_superseded_unshared_byte_algorithm_is_read_only(tmp_path: Path) -> None:
    lock_path = tmp_path / "environment.json"
    value = _environment_lock(lock_path)
    value["distribution_inventory"]["algorithm"] = (
        runtime.LEGACY_BYTE_DISTRIBUTION_INVENTORY_ALGORITHM
    )
    value["provisioning"]["source_environment_fingerprint_algorithm"] = (
        runtime.LEGACY_BYTE_SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
    )
    value["environment_id"] = (
        "linbox-perception-" + runtime.environment_identity_sha256(value)[:20]
    )

    assert runtime.validate_environment_lock(value) == value
    assert runtime.is_strong_environment_lock(value) is False
    with pytest.raises(
        runtime.PerceptionRuntimeError, match="strong installed-file byte lock",
    ):
        runtime.validate_environment_lock(
            value, require_posterior_gate=True,
        )


def test_v2_environment_id_binds_target_identity_not_only_clone_source(tmp_path: Path) -> None:
    lock_path = tmp_path / "environment.json"
    value = _environment_lock(lock_path)
    original_id = value["environment_id"]

    value["modules"]["ultralytics"]["sha256"] = "0" * 64

    with pytest.raises(runtime.PerceptionRuntimeError, match="complete environment identity"):
        runtime.validate_environment_lock(value)
    value["environment_id"] = (
        "linbox-perception-" + runtime.environment_identity_sha256(value)[:20]
    )
    assert value["environment_id"] != original_id
    assert runtime.validate_environment_lock(value) == value


def test_checked_in_gate_profile_and_lock_schema_pin_the_production_asset() -> None:
    profile_raw = POSTERIOR_GATE_PROFILE.read_bytes()
    profile = json.loads(profile_raw)
    schema_raw = ENVIRONMENT_LOCK_V2_SCHEMA.read_bytes()
    schema = json.loads(schema_raw)

    assert profile_raw == contract.canonical_json_bytes(profile)
    assert schema_raw == contract.canonical_json_bytes(schema)
    assert profile == {
        "asset": {
            "architecture": "YOLO11x-pose",
            "filename": "yolo11x-pose.pt",
            "sha256": "013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8",
            "size": 118481010,
        },
        "critical_distributions": [
            {"name": name, "version": version}
            for name, version in sorted(runtime.SUPPORTED_PACKAGES.items())
        ],
        "critical_modules": sorted(runtime.SUPPORTED_MODULES),
        "distribution_inventory_algorithm": (
            runtime.DISTRIBUTION_INVENTORY_ALGORITHM
        ),
        "environment_lock_schema": runtime.ENVIRONMENT_LOCK_SCHEMA,
        "module_tree_algorithm": runtime.MODULE_TREE_ALGORITHM,
        "point_source_policy": {
            "auscultation_model": "perc_lmh_v9_occ",
            "posterior_gate_produces_auscultation_points": False,
        },
        "schema": "wam_perception_posterior_gate_profile_v1",
        "source_environment_fingerprint_algorithm": (
            runtime.SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
        ),
    }
    assert schema["properties"]["schema"] == {
        "const": runtime.ENVIRONMENT_LOCK_SCHEMA,
    }
    assert schema["properties"]["distribution_inventory"]["properties"][
        "algorithm"
    ] == {"const": runtime.DISTRIBUTION_INVENTORY_ALGORITHM}
    for module in sorted(runtime.SUPPORTED_MODULES):
        identity = schema["properties"]["modules"]["properties"][module]
        assert identity["required"] == ["algorithm", "count", "sha256"]
        assert identity["properties"]["algorithm"] == {
            "const": runtime.MODULE_TREE_ALGORITHM,
        }
    assert schema["properties"]["provisioning"]["properties"][
        "source_environment_fingerprint_algorithm"
    ] == {"const": runtime.SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM}
    assert schema["$id"] == "wam://schemas/perception-environment-lock-v2"


def test_checked_in_v3_hardened_code_lock_is_frozen_for_rollback() -> None:
    raw = V3_HARDENED_CODE_LOCK.read_bytes()
    value = json.loads(raw.decode("utf-8"))

    assert _sha256(V3_HARDENED_CODE_LOCK) == (
        "7fbaaef4b3c1761bf3c2a199d08f991238f09760c99c2d78691c7e81a1172d93"
    )
    assert raw == contract.canonical_json_bytes(value)
    assert set(value) == runtime.V3_REQUIRED_HARDENED_CODE_FILES
    assert {
        relative: value[relative]
        for relative in runtime.V3_REQUIRED_BACK_GATE_SEMANTIC_CODE_SHA256
    } == runtime.V3_REQUIRED_BACK_GATE_SEMANTIC_CODE_SHA256


def test_checked_in_v4_hardened_code_lock_is_frozen_for_rollback() -> None:
    raw = V4_HARDENED_CODE_LOCK.read_bytes()
    value = json.loads(raw.decode("utf-8"))

    assert _sha256(V4_HARDENED_CODE_LOCK) == (
        "af9009a98db57fd8419929d8a67b3168551208fccb06e8d4355f5efd900d108c"
    )
    assert raw == contract.canonical_json_bytes(value)
    assert set(value) == runtime.V4_REQUIRED_HARDENED_CODE_FILES
    assert value == runtime.V4_REQUIRED_PRODUCTION_CHAIN_SEMANTIC_CODE_SHA256


def test_checked_in_v5_hardened_code_lock_matches_the_reviewed_files() -> None:
    raw = V5_HARDENED_CODE_LOCK.read_bytes()
    value = json.loads(raw.decode("utf-8"))

    assert raw == contract.canonical_json_bytes(value)
    assert set(value) == runtime.REQUIRED_HARDENED_CODE_FILES
    assert value == runtime.REQUIRED_RAW_COLLECTION_RUNTIME_SEMANTIC_CODE_SHA256
    for relative, expected_sha256 in value.items():
        assert _sha256(AUSC_ROOT.parent / relative) == expected_sha256


def test_environment_lock_rejects_an_unprobed_cuda_architecture(tmp_path: Path) -> None:
    lock_path = tmp_path / "environment.json"
    value = _environment_lock(lock_path)
    value["cuda"]["architectures"].append("sm_999")

    with pytest.raises(runtime.PerceptionRuntimeError, match="CUDA build/architectures"):
        runtime.validate_environment_lock(value)


def test_activation_uses_expected_current_compare_and_swap(tmp_path: Path) -> None:
    first_inputs = _inputs(tmp_path / "first")
    first = runtime.build_perception_runtime_release(
        checkpoint=first_inputs["checkpoint"],
        expected_checkpoint_sha256=_sha256(first_inputs["checkpoint"]),
        expected_checkpoint_size=first_inputs["checkpoint"].stat().st_size,
        **_posterior_gate_kwargs(first_inputs),
        dino_checkpoint=first_inputs["dino_checkpoint"],
        expected_dino_checkpoint_sha256=_sha256(first_inputs["dino_checkpoint"]),
        expected_dino_checkpoint_size=first_inputs["dino_checkpoint"].stat().st_size,
        dino_repo=first_inputs["dino_repo"],
        expected_dino_repo_sha256=runtime.hash_dino_source_tree(first_inputs["dino_repo"])["sha256"],
        code_release=first_inputs["code_release"],
        expected_code_release_id=first_inputs["code_manifest"]["release_id"],
        expected_code_payload_sha256=first_inputs["code_manifest"]["payload_sha256"],
        expected_hardened_code_sha256=_hardened_hashes(first_inputs["code_manifest"]),
        environment_lock=first_inputs["environment_lock"],
        expected_environment_lock_sha256=_sha256(first_inputs["environment_lock"]),
        releases_root=tmp_path / "built",
    )
    second_inputs = _inputs(tmp_path / "second")
    second_inputs["checkpoint"].write_bytes(b"different-exact-v9-checkpoint")
    second = runtime.build_perception_runtime_release(
        checkpoint=second_inputs["checkpoint"],
        expected_checkpoint_sha256=_sha256(second_inputs["checkpoint"]),
        expected_checkpoint_size=second_inputs["checkpoint"].stat().st_size,
        **_posterior_gate_kwargs(second_inputs),
        dino_checkpoint=second_inputs["dino_checkpoint"],
        expected_dino_checkpoint_sha256=_sha256(second_inputs["dino_checkpoint"]),
        expected_dino_checkpoint_size=second_inputs["dino_checkpoint"].stat().st_size,
        dino_repo=second_inputs["dino_repo"],
        expected_dino_repo_sha256=runtime.hash_dino_source_tree(second_inputs["dino_repo"])["sha256"],
        code_release=second_inputs["code_release"],
        expected_code_release_id=second_inputs["code_manifest"]["release_id"],
        expected_code_payload_sha256=second_inputs["code_manifest"]["payload_sha256"],
        expected_hardened_code_sha256=_hardened_hashes(second_inputs["code_manifest"]),
        environment_lock=second_inputs["environment_lock"],
        expected_environment_lock_sha256=_sha256(second_inputs["environment_lock"]),
        releases_root=tmp_path / "built",
    )
    releases = tmp_path / "installed" / "releases"
    current = tmp_path / "installed" / "current"

    first_installed = runtime.activate_perception_runtime_release(
        first,
        releases_root=releases,
        current_link=current,
        expected_current=None,
    )
    first_target = os.readlink(current)
    with pytest.raises(contract.ConflictError, match="compare-and-swap"):
        runtime.activate_perception_runtime_release(
            second,
            releases_root=releases,
            current_link=current,
            expected_current="releases/stale-release",
        )
    assert current.resolve() == first_installed.resolve()
    assert os.readlink(current) == first_target

    second_installed = runtime.activate_perception_runtime_release(
        second,
        releases_root=releases,
        current_link=current,
        expected_current=first_target,
    )
    assert current.resolve() == second_installed.resolve()
    assert runtime.verify_perception_runtime_release(current.resolve())["release_id"] == second.name


def test_release_profile_uses_a_dedicated_linbox_runtime_pointer() -> None:
    profile = json.loads(RELEASE_PROFILE.read_text(encoding="utf-8"))

    assert profile["hosts"]["linbox"]["perception_runtime"] == {
        "current_link": "/home/robot/wam_data/perception_runtime/current",
        "incoming_root": "/home/robot/wam_data/perception_runtime/.staging",
        "releases_root": "/home/robot/wam_data/perception_runtime/releases",
    }


def test_builder_rejects_a_code_release_without_hardened_runtime_files(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    incomplete_release, incomplete_manifest = _code_release(
        tmp_path / "incomplete", complete=False,
    )

    with pytest.raises(runtime.PerceptionRuntimeError, match="required hardened files"):
        runtime.build_perception_runtime_release(
            checkpoint=inputs["checkpoint"],
            expected_checkpoint_sha256=_sha256(inputs["checkpoint"]),
            expected_checkpoint_size=inputs["checkpoint"].stat().st_size,
            **_posterior_gate_kwargs(inputs),
            dino_checkpoint=inputs["dino_checkpoint"],
            expected_dino_checkpoint_sha256=_sha256(inputs["dino_checkpoint"]),
            expected_dino_checkpoint_size=inputs["dino_checkpoint"].stat().st_size,
            dino_repo=inputs["dino_repo"],
            expected_dino_repo_sha256=runtime.hash_dino_source_tree(inputs["dino_repo"])["sha256"],
            code_release=incomplete_release,
            expected_code_release_id=incomplete_manifest["release_id"],
            expected_code_payload_sha256=incomplete_manifest["payload_sha256"],
            expected_hardened_code_sha256={
                relative: "a" * 64
                for relative in sorted(runtime.REQUIRED_HARDENED_CODE_FILES)
            },
            environment_lock=inputs["environment_lock"],
            expected_environment_lock_sha256=_sha256(inputs["environment_lock"]),
            releases_root=tmp_path / "runtime-releases",
        )


def test_builder_rejects_unconfirmed_hardened_code_file_hash(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    hardened_hashes = _hardened_hashes(inputs["code_manifest"])
    hardened_hashes["auscultation/ausc/models/iuv_model.py"] = "0" * 64

    with pytest.raises(runtime.PerceptionRuntimeError, match="hardened code file"):
        runtime.build_perception_runtime_release(
            checkpoint=inputs["checkpoint"],
            expected_checkpoint_sha256=_sha256(inputs["checkpoint"]),
            expected_checkpoint_size=inputs["checkpoint"].stat().st_size,
            **_posterior_gate_kwargs(inputs),
            dino_checkpoint=inputs["dino_checkpoint"],
            expected_dino_checkpoint_sha256=_sha256(inputs["dino_checkpoint"]),
            expected_dino_checkpoint_size=inputs["dino_checkpoint"].stat().st_size,
            dino_repo=inputs["dino_repo"],
            expected_dino_repo_sha256=runtime.hash_dino_source_tree(inputs["dino_repo"])["sha256"],
            code_release=inputs["code_release"],
            expected_code_release_id=inputs["code_manifest"]["release_id"],
            expected_code_payload_sha256=inputs["code_manifest"]["payload_sha256"],
            expected_hardened_code_sha256=hardened_hashes,
            environment_lock=inputs["environment_lock"],
            expected_environment_lock_sha256=_sha256(inputs["environment_lock"]),
            releases_root=tmp_path / "runtime-releases",
        )


def test_v5_rejects_unreviewed_collection_code_even_when_hashes_are_self_consistent(
    tmp_path: Path,
) -> None:
    release, manifest = _code_release(
        tmp_path / "unreviewed-production", reviewed_semantics=False,
    )
    envelope = contract.verify_release(release)
    self_consistent_hashes = _hardened_hashes(manifest)

    with pytest.raises(
        runtime.PerceptionRuntimeError,
        match="raw collection runtime semantic code",
    ):
        runtime._verify_hardened_code_manifest(
            envelope,
            self_consistent_hashes,
            runtime_schema=runtime.RUNTIME_MANIFEST_SCHEMA,
        )


def test_v5_rejects_a_manifest_missing_the_dormant_controller_activation() -> None:
    expected = dict(runtime.REQUIRED_RAW_COLLECTION_RUNTIME_SEMANTIC_CODE_SHA256)
    missing = "auscultation/ausc/perception/robot/controller_launch_activation.py"
    expected.pop(missing)
    rows = [
        {"path": relative, "sha256": digest}
        for relative, digest in sorted(expected.items())
    ]

    with pytest.raises(runtime.PerceptionRuntimeError, match="required hardened files"):
        runtime._verify_hardened_code_manifest(
            {"payload": {"files": rows}},
            expected,
            runtime_schema=runtime.RUNTIME_MANIFEST_SCHEMA,
        )


@pytest.mark.parametrize(
    "tampered_relative",
    [
        "auscultation/scripts/deployment/artifact_contract.py",
        "auscultation/scripts/deployment/perception_environment_manifest.py",
        "auscultation/scripts/deployment/verify_perception_environment.py",
    ],
)
def test_v5_rejects_self_consistent_runtime_trust_closure_tamper(
    tmp_path: Path, tampered_relative: str,
) -> None:
    assert tampered_relative in runtime.REQUIRED_HARDENED_CODE_FILES
    release, manifest = _code_release(
        tmp_path / tampered_relative.rsplit("/", 1)[-1],
        tampered_relative=tampered_relative,
    )
    envelope = contract.verify_release(release)
    actual = _hardened_hashes(manifest)
    reviewed = runtime.REQUIRED_RAW_COLLECTION_RUNTIME_SEMANTIC_CODE_SHA256
    assert actual[tampered_relative] != reviewed[tampered_relative]
    assert {
        relative: digest
        for relative, digest in actual.items()
        if relative != tampered_relative
    } == {
        relative: digest
        for relative, digest in reviewed.items()
        if relative != tampered_relative
    }

    with pytest.raises(
        runtime.PerceptionRuntimeError,
        match="raw collection runtime semantic code",
    ):
        runtime._verify_hardened_code_manifest(
            envelope,
            actual,
            runtime_schema=runtime.RUNTIME_MANIFEST_SCHEMA,
        )


def test_builder_filters_dino_bytecode_from_a_live_torch_cache(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    transient = inputs["dino_repo"] / "dinov2" / "__pycache__" / "__init__.cpython-311.pyc"
    transient.parent.mkdir()
    transient.write_bytes(b"host-specific bytecode")

    release = runtime.build_perception_runtime_release(
        checkpoint=inputs["checkpoint"],
        expected_checkpoint_sha256=_sha256(inputs["checkpoint"]),
        expected_checkpoint_size=inputs["checkpoint"].stat().st_size,
        **_posterior_gate_kwargs(inputs),
        dino_checkpoint=inputs["dino_checkpoint"],
        expected_dino_checkpoint_sha256=_sha256(inputs["dino_checkpoint"]),
        expected_dino_checkpoint_size=inputs["dino_checkpoint"].stat().st_size,
        dino_repo=inputs["dino_repo"],
        expected_dino_repo_sha256=runtime.hash_dino_source_tree(inputs["dino_repo"])["sha256"],
        code_release=inputs["code_release"],
        expected_code_release_id=inputs["code_manifest"]["release_id"],
        expected_code_payload_sha256=inputs["code_manifest"]["payload_sha256"],
        expected_hardened_code_sha256=_hardened_hashes(inputs["code_manifest"]),
        environment_lock=inputs["environment_lock"],
        expected_environment_lock_sha256=_sha256(inputs["environment_lock"]),
        releases_root=tmp_path / "runtime-releases",
    )

    assert not any("__pycache__" in path.parts for path in release.rglob("*"))
    runtime.verify_perception_runtime_release(release)


def test_cli_build_verify_and_activate_require_explicit_cas(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _inputs(tmp_path)
    hardened = tmp_path / "hardened-code-sha256.json"
    _write_canonical(hardened, _hardened_hashes(inputs["code_manifest"]))
    built_root = tmp_path / "built"

    assert runtime.main([
        "build",
        "--checkpoint", str(inputs["checkpoint"]),
        "--checkpoint-sha256", _sha256(inputs["checkpoint"]),
        "--checkpoint-size", str(inputs["checkpoint"].stat().st_size),
        "--posterior-gate-checkpoint", str(inputs["posterior_gate_checkpoint"]),
        "--posterior-gate-checkpoint-sha256", _sha256(
            inputs["posterior_gate_checkpoint"]
        ),
        "--posterior-gate-checkpoint-size", str(
            inputs["posterior_gate_checkpoint"].stat().st_size
        ),
        "--dino-checkpoint", str(inputs["dino_checkpoint"]),
        "--dino-checkpoint-sha256", _sha256(inputs["dino_checkpoint"]),
        "--dino-checkpoint-size", str(inputs["dino_checkpoint"].stat().st_size),
        "--dino-repo", str(inputs["dino_repo"]),
        "--dino-repo-sha256", runtime.hash_dino_source_tree(inputs["dino_repo"])["sha256"],
        "--code-release", str(inputs["code_release"]),
        "--code-release-id", inputs["code_manifest"]["release_id"],
        "--code-payload-sha256", inputs["code_manifest"]["payload_sha256"],
        "--hardened-code-sha256", str(hardened),
        "--environment-lock", str(inputs["environment_lock"]),
        "--environment-lock-sha256", _sha256(inputs["environment_lock"]),
        "--releases-root", str(built_root),
    ]) == 0
    build_raw = capsys.readouterr().out
    build_output = json.loads(build_raw)
    assert build_raw.encode("utf-8") == contract.canonical_json_bytes(build_output)
    release = Path(build_output["release_dir"])
    assert build_output == {
        "command": "build",
        "release_dir": str(release),
        "release_id": release.name,
    }

    assert runtime.main(["verify", str(release)]) == 0
    verify_raw = capsys.readouterr().out
    verify_output = json.loads(verify_raw)
    assert verify_raw.encode("utf-8") == contract.canonical_json_bytes(verify_output)
    assert verify_output == {"command": "verify", "release_id": release.name}

    installed_root = tmp_path / "installed" / "releases"
    current = tmp_path / "installed" / "current"
    assert runtime.main([
        "activate", str(release),
        "--releases-root", str(installed_root),
        "--current-link", str(current),
        "--expect-no-current",
    ]) == 0
    activate_raw = capsys.readouterr().out
    activate_output = json.loads(activate_raw)
    assert activate_raw.encode("utf-8") == contract.canonical_json_bytes(activate_output)
    assert activate_output == {
        "command": "activate",
        "current": os.readlink(current),
        "release_dir": str(current.resolve()),
        "release_id": release.name,
    }

    with pytest.raises(SystemExit) as exc_info:
        runtime.main([
            "activate", str(release),
            "--releases-root", str(installed_root),
            "--current-link", str(current),
        ])
    assert exc_info.value.code == 2
