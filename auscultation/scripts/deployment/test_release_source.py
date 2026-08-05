from __future__ import annotations

from pathlib import Path
import sys

import pytest


DEPLOYMENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEPLOYMENT_DIR.parents[2]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

import release_source  # noqa: E402
import artifact_contract  # noqa: E402
import deploy_remote  # noqa: E402
import publish_perception_runtime  # noqa: E402


def _write(workspace: Path, relative: str, data: str = "fixture\n") -> None:
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def test_current_workspace_release_source_is_complete_and_deterministic() -> None:
    first = release_source.select_code_release_sources(WORKSPACE_ROOT)
    second = release_source.select_code_release_sources(WORKSPACE_ROOT)

    assert first == second == tuple(sorted(set(first)))
    assert set(release_source.REQUIRED_RELEASE_PATHS) <= set(first)
    assert all(path.startswith("auscultation/") for path in first)


def test_release_anchors_cover_every_hardened_perception_runtime_file() -> None:
    assert len(release_source.REQUIRED_RELEASE_PATHS) == len(
        set(release_source.REQUIRED_RELEASE_PATHS)
    )
    assert publish_perception_runtime.REQUIRED_HARDENED_CODE_FILES <= set(
        release_source.REQUIRED_RELEASE_PATHS
    )
    assert (
        "auscultation/scripts/deployment/collection_service_unit.py"
        in release_source.REQUIRED_RELEASE_PATHS
    )


def test_release_anchors_cover_left_robot_and_occlusion_runtime() -> None:
    assert {
        "auscultation/ausc/perception/robot/camera_exposure_evidence.py",
        "auscultation/ausc/perception/robot/controller_launch_activation.py",
        "auscultation/ausc/perception/robot/frame_evidence_bundle.py",
        "auscultation/ausc/perception/robot/openarm_pair_activation.py",
        "auscultation/ausc/perception/robot/robot_capture_coordinator.py",
        "auscultation/ausc/perception/robot/robot_mask_provider.py",
        "auscultation/ausc/perception/robot/robot_session_capture.py",
        "auscultation/ausc/perception/robot/robot_socket_receiver.py",
        "auscultation/ausc/perception/robot/robot_state_stream.py",
        "auscultation/ausc/perception/production_runtime_activation.py",
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
        "auscultation/configs/perception_runtime/hardened-code-sha256-v4.json",
        "auscultation/configs/perception_runtime/hardened-code-sha256-v5.json",
        "auscultation/mechanical_sensor/force_feedback/force_capture.py",
        "auscultation/mechanical_sensor/force_feedback/force_sensor.py",
    } <= set(release_source.REQUIRED_RELEASE_PATHS)


def test_release_anchors_cover_provisional_handeye_issuer_and_loader() -> None:
    assert {
        "auscultation/scripts/perception/activate_handeye.py",
        "auscultation/scripts/perception/test_activate_handeye.py",
        "auscultation/ausc/perception/production_runtime_activation.py",
    } <= set(release_source.REQUIRED_RELEASE_PATHS)


def test_release_anchors_cover_training_promotion_truth_boundary() -> None:
    assert {
        "auscultation/ausc/realtime_q.py",
        "auscultation/ausc/perception/tracking/geometry_channel_packet.py",
        "auscultation/ausc/perception/tracking/production_evidence_capture.py",
        "auscultation/scripts/training_authenticity_finalizer.py",
        "auscultation/scripts/deployment/artifact_contract.py",
        "auscultation/scripts/deployment/remote_agent.py",
        "auscultation/scripts/session_to_lerobot_v2.py",
    } <= set(release_source.REQUIRED_RELEASE_PATHS)


def test_release_anchors_cover_observable_openarm_candidate_and_verification() -> None:
    assert {
        "auscultation/openarm_bridge/README.md",
        "auscultation/openarm_bridge/apply_openarm_can_tx_status_patch.py",
        "auscultation/openarm_bridge/apply_openarm_telemetry_patch.py",
        "auscultation/openarm_bridge/lingbox_openarm_baseline.json",
        "auscultation/openarm_bridge/openarm_can_1.2.2.tx_status.patch",
        "auscultation/openarm_bridge/openarm_can_tx_enqueue_status.hpp",
        "auscultation/openarm_bridge/openarm_telemetry_publisher.cpp",
        "auscultation/openarm_bridge/openarm_telemetry_publisher.hpp",
        "auscultation/openarm_bridge/openarm_unilateral_control.telemetry.patch",
        "auscultation/openarm_bridge/tests/openarm_can_tx_status_harness.cpp",
        "auscultation/openarm_bridge/tests/publisher_harness.cpp",
        "auscultation/openarm_bridge/tests/test_openarm_telemetry_publisher.py",
    } <= set(release_source.REQUIRED_RELEASE_PATHS)


def test_real_code_release_manifest_contains_immutable_collector_generator() -> None:
    selected = release_source.select_code_release_sources(WORKSPACE_ROOT)
    manifest = artifact_contract.build_manifest(
        WORKSPACE_ROOT,
        selected,
        artifact_kind="code_release",
        source_role="mac",
    )
    rows = {
        row["path"]: row
        for row in manifest["payload"]["files"]
    }

    relative = "auscultation/scripts/deployment/collection_service_unit.py"
    assert relative in selected
    assert relative in rows
    assert rows[relative]["size"] == (WORKSPACE_ROOT / relative).stat().st_size


def test_selector_includes_code_config_docs_requirements_and_hardware_runtime(
    tmp_path: Path,
) -> None:
    included = {
        "auscultation/README.md",
        "auscultation/ausc/runtime.py",
        "auscultation/ausc/perception/robot/openarm_v10.urdf",
        "auscultation/scripts/posterior_gate.py",
        "auscultation/scripts/deployment/environment_builder.py",
        "auscultation/tests/test_training.py",
        "auscultation/configs/perception_runtime/lock.schema.json",
        "auscultation/docs/runtime.md",
        "auscultation/mechanical_sensor/force_feedback/force_sensor.py",
        "auscultation/mechanical_sensor/force_feedback/firmware/main.py",
        "auscultation/mechanical_sensor/force_feedback/v8b_hall_reader/reader.ino",
        "auscultation/mechanical_sensor/tests/test_force_sensor.py",
        "auscultation/openarm_bridge/openarm_telemetry_publisher.cpp",
        "auscultation/openarm_bridge/openarm_telemetry_publisher.hpp",
        "auscultation/openarm_bridge/openarm_unilateral_control.telemetry.patch",
        "auscultation/requirements-h200.lock.txt",
    }
    for relative in included:
        _write(tmp_path, relative)

    selected = release_source.select_code_release_sources(
        tmp_path, required_paths=()
    )

    assert set(selected) == included


def test_real_release_contains_left_openarm_telemetry_bridge() -> None:
    selected = set(release_source.select_code_release_sources(WORKSPACE_ROOT))

    assert {
        "auscultation/openarm_bridge/README.md",
        "auscultation/openarm_bridge/apply_openarm_can_tx_status_patch.py",
        "auscultation/openarm_bridge/apply_openarm_telemetry_patch.py",
        "auscultation/openarm_bridge/lingbox_openarm_baseline.json",
        "auscultation/openarm_bridge/openarm_can_1.2.2.tx_status.patch",
        "auscultation/openarm_bridge/openarm_can_tx_enqueue_status.hpp",
        "auscultation/openarm_bridge/openarm_telemetry_publisher.cpp",
        "auscultation/openarm_bridge/openarm_telemetry_publisher.hpp",
        "auscultation/openarm_bridge/openarm_unilateral_control.telemetry.patch",
        "auscultation/openarm_bridge/tests/openarm_can_tx_status_harness.cpp",
        "auscultation/openarm_bridge/tests/publisher_harness.cpp",
        "auscultation/openarm_bridge/tests/test_openarm_telemetry_publisher.py",
    } <= selected


def test_selector_excludes_secrets_data_evidence_caches_weights_media_and_results(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "auscultation/ausc/runtime.py")
    excluded = {
        "auscultation/.env",
        "auscultation/.env.example",
        "auscultation/ausc/.env.local",
        "auscultation/ausc/.ENV.json",
        "auscultation/ausc/__pycache__/runtime.py",
        "auscultation/ausc/.cache/state.json",
        "auscultation/ausc/.eggs/package/metadata.json",
        "auscultation/ausc/.mypy_cache/state.json",
        "auscultation/ausc/.ruff_cache/state.json",
        "auscultation/ausc/.venv/pyvenv.json",
        "auscultation/ausc/cache/state.json",
        "auscultation/ausc/dist/package.json",
        "auscultation/ausc/node_modules/package/config.json",
        "auscultation/ausc/wam.egg-info/requires.txt",
        "auscultation/ausc/checkpoints/model.json",
        "auscultation/ausc/model.pt",
        "auscultation/ausc/frame.png",
        "auscultation/ausc/results.jsonl",
        "auscultation/ausc/user/Results.jsonl",
        "auscultation/data/session/manifest.json",
        "auscultation/Log/OrbbecSDK.log.txt",
        "auscultation/logs/service.log.txt",
        "auscultation/docs/runs/evidence.json",
        "auscultation/outputs/run/metrics.json",
        "auscultation/scripts/audio/sprsound_loop/quality_gbm.joblib",
        "auscultation/scripts/my_experiment_results.jsonl",
        "auscultation/scripts/my_experiment_results.md",
        "auscultation/scripts/results/summary.json",
        "auscultation/scripts/metrics/latest.json",
        "auscultation/scripts/run_metrics.json",
        "auscultation/scripts/run_metrics.txt",
        "auscultation/scripts/subject_results.yaml",
        "auscultation/mechanical_sensor/tests/test_mount_outputs.py",
        "auscultation/mechanical_sensor/thinklab_one/notes.md",
    }
    for relative in excluded:
        _write(tmp_path, relative)

    selected = release_source.select_code_release_sources(
        tmp_path, required_paths=()
    )

    assert selected == ("auscultation/ausc/runtime.py",)


def test_selector_fails_closed_on_symlinks_inside_an_included_tree(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "auscultation/ausc/runtime.py")
    linked = tmp_path / "auscultation/ausc/linked.py"
    linked.symlink_to("runtime.py")

    with pytest.raises(release_source.ReleaseSourceError, match="symlink"):
        release_source.select_code_release_sources(tmp_path, required_paths=())


def test_selector_fails_closed_on_an_unreadable_included_subtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "auscultation/ausc").mkdir(parents=True)

    def denied_walk(root, *, followlinks, onerror):
        assert followlinks is False and onerror is not None
        onerror(PermissionError(13, "permission denied", str(root) + "/private"))
        return ()

    monkeypatch.setattr(release_source.os, "walk", denied_walk)
    with pytest.raises(release_source.ReleaseSourceError, match="traverse"):
        release_source.select_code_release_sources(tmp_path, required_paths=())


def test_selector_rejects_a_broken_symlink_used_as_an_included_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "auscultation").mkdir()
    (tmp_path / "auscultation/docs").symlink_to(
        "missing-docs", target_is_directory=True,
    )

    with pytest.raises(release_source.ReleaseSourceError, match="symlink"):
        release_source.select_code_release_sources(tmp_path, required_paths=())


def test_selector_rejects_a_broken_symlink_used_as_an_explicit_file(
    tmp_path: Path,
) -> None:
    target = (
        tmp_path
        / "auscultation/mechanical_sensor/tests/test_force_sensor.py"
    )
    target.parent.mkdir(parents=True)
    target.symlink_to("missing-force-sensor.py")

    with pytest.raises(release_source.ReleaseSourceError, match="symlink"):
        release_source.select_code_release_sources(tmp_path, required_paths=())


def test_selector_rejects_a_symlinked_ancestor_of_runtime_sources(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external-mechanical"
    _write(tmp_path, "external-mechanical/force_feedback/force_sensor.py")
    _write(tmp_path, "external-mechanical/tests/test_force_sensor.py")
    auscultation = tmp_path / "auscultation"
    auscultation.mkdir()
    (auscultation / "mechanical_sensor").symlink_to(external)

    with pytest.raises(release_source.ReleaseSourceError, match="symlink"):
        release_source.select_code_release_sources(tmp_path, required_paths=())


def test_deploy_boundary_translates_a_disappearing_file_to_policy_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "auscultation/ausc").mkdir(parents=True)

    def disappearing_walk(root, *, followlinks, onerror):
        assert followlinks is False and onerror is not None
        return iter(((str(root), [], ["vanished.py"]),))

    monkeypatch.setattr(release_source.os, "walk", disappearing_walk)
    with pytest.raises(deploy_remote.PolicyError, match="inspect"):
        deploy_remote.build_source_allowlist(tmp_path, required_paths=())


def test_selector_fails_when_a_required_release_anchor_is_missing(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "auscultation/ausc/runtime.py")

    with pytest.raises(release_source.ReleaseSourceError, match="missing required"):
        release_source.select_code_release_sources(
            tmp_path,
            required_paths=("auscultation/scripts/posterior_gate.py",),
        )


def test_remote_deployment_uses_the_same_release_source_contract(
    tmp_path: Path,
) -> None:
    for relative in (
        "auscultation/README.md",
        "auscultation/ausc/runtime.py",
        "auscultation/ausc/perception/robot/openarm_v10.urdf",
        "auscultation/mechanical_sensor/force_feedback/force_feedback_monitor.py",
        "auscultation/mechanical_sensor/tests/test_force_sensor.py",
        "auscultation/requirements-h200.lock.txt",
    ):
        _write(tmp_path, relative)

    assert deploy_remote.build_source_allowlist(
        tmp_path, required_paths=()
    ) == release_source.select_code_release_sources(tmp_path, required_paths=())
