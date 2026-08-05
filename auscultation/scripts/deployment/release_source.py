"""Deterministic, fail-closed source selection for WAM code releases.

The code-release manifest is the authority for what reaches Linbox and H200.
This module deliberately knows nothing about SSH or activation so the source
boundary can be tested without constructing a deployment plan.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Sequence, Tuple, Union


ALLOWED_SUFFIXES = frozenset({
    ".py", ".sh", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini",
    ".md", ".txt", ".ino", ".lock", ".urdf", ".cpp", ".hpp", ".patch",
})

INCLUDED_TREE_ROOTS = (
    "auscultation/ausc",
    "auscultation/scripts",
    "auscultation/tests",
    "auscultation/config",
    "auscultation/configs",
    "auscultation/docs",
    "auscultation/mechanical_sensor/force_feedback",
    "auscultation/openarm_bridge",
)

EXPLICIT_RELEASE_FILES = (
    "auscultation/mechanical_sensor/tests/test_force_sensor.py",
)

EXCLUDED_DIRECTORY_NAMES = frozenset({
    ".cache", ".eggs", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".venv", "__pycache__", "__pypackages__", "backups", "build",
    "cache", "caches", "checkpoints", "ckpt", "data", "dist", "env", "envs",
    "log", "logs", "metrics", "node_modules", "outputs", "results", "runs",
    "server_backups", "site-packages", "venv",
})

DENIED_BASENAMES = frozenset({
    "opera_results.jsonl", "perception_metrics.json", "quality_meta.json",
    "results.jsonl", "xdata_results.jsonl",
})
RESULT_ARTIFACT_SUFFIXES = frozenset({
    ".ini", ".json", ".jsonl", ".lock", ".md", ".txt", ".yaml", ".yml",
})
DENIED_ARTIFACT_STEM_SUFFIXES = ("_metrics", "_results")

# These are availability anchors, not a hand-maintained exhaustive file list.
# Entire audited trees are selected above so newly added source is included
# automatically; the anchors prevent a critical entrypoint from disappearing
# silently during a refactor.
REQUIRED_RELEASE_PATHS = (
    "auscultation/README.md",
    "auscultation/ausc/audio/audio_stream.py",
    "auscultation/ausc/health.py",
    "auscultation/ausc/models/iuv_model.py",
    "auscultation/ausc/perception/camera_stream.py",
    "auscultation/ausc/perception/depth_codec.py",
    "auscultation/ausc/perception/production_runtime_activation.py",
    "auscultation/ausc/perception/robot/camera_exposure_evidence.py",
    "auscultation/ausc/perception/robot/controller_launch.py",
    "auscultation/ausc/perception/robot/controller_launch_activation.py",
    "auscultation/ausc/perception/robot/frame_evidence_bundle.py",
    "auscultation/ausc/perception/robot/openarm_pair_activation.py",
    "auscultation/ausc/perception/robot/openarm_v10.urdf",
    "auscultation/ausc/perception/robot/robot_capture_coordinator.py",
    "auscultation/ausc/perception/robot/robot_mask_provider.py",
    "auscultation/ausc/perception/robot/robot_session_capture.py",
    "auscultation/ausc/perception/robot/robot_socket_receiver.py",
    "auscultation/ausc/perception/robot/robot_state_stream.py",
    "auscultation/ausc/perception/robot/trusted_controller_companion.py",
    "auscultation/ausc/perception/tracking/__init__.py",
    "auscultation/ausc/perception/tracking/back_mask_packet.py",
    "auscultation/ausc/perception/tracking/depth_noise_profile.py",
    "auscultation/ausc/perception/tracking/detector.py",
    "auscultation/ausc/perception/tracking/model_tracker.py",
    "auscultation/ausc/perception/tracking/experimental_visibility_geometry.py",
    "auscultation/ausc/perception/tracking/geometry.py",
    "auscultation/ausc/perception/tracking/geometry_channel_packet.py",
    "auscultation/ausc/perception/tracking/occlusion_runtime.py",
    "auscultation/ausc/perception/tracking/patient_frame_evidence.py",
    "auscultation/ausc/perception/tracking/patient_pose_provider.py",
    "auscultation/ausc/perception/tracking/posterior_back_mask_provider.py",
    "auscultation/ausc/perception/tracking/production_evidence_capture.py",
    "auscultation/ausc/perception/tracking/production_geometry_gate.py",
    "auscultation/ausc/perception/tracking/runtime.py",
    "auscultation/ausc/realtime_q.py",
    "auscultation/ausc/record_session.py",
    "auscultation/ausc/tests/test_recording_durability.py",
    "auscultation/configs/perception_runtime/hardened-code-sha256.json",
    "auscultation/configs/perception_runtime/hardened-code-sha256-v4.json",
    "auscultation/configs/perception_runtime/hardened-code-sha256-v5.json",
    "auscultation/configs/perception_runtime/perception-environment-lock-v3.schema.json",
    "auscultation/configs/perception_runtime/posterior-gate-profile.json",
    "auscultation/mechanical_sensor/force_feedback/firmware/main.py",
    "auscultation/mechanical_sensor/force_feedback/force_capture.py",
    "auscultation/mechanical_sensor/force_feedback/force_sensor.py",
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
    "auscultation/requirements-h200.in",
    "auscultation/requirements-h200.lock.txt",
    "auscultation/requirements-linbox-lerobot.in",
    "auscultation/requirements-linbox-lerobot.lock.txt",
    "auscultation/requirements-perception.txt",
    "auscultation/scripts/collect_server.py",
    "auscultation/scripts/collection_platform.py",
    "auscultation/scripts/collection_supervisor.py",
    "auscultation/scripts/convert_session_on_linbox.sh",
    "auscultation/scripts/deployment/activate_perception_environment.py",
    "auscultation/scripts/deployment/artifact_contract.py",
    "auscultation/scripts/deployment/collection_service_unit.py",
    "auscultation/scripts/deployment/deploy_remote.py",
    "auscultation/scripts/deployment/h200_env.py",
    "auscultation/scripts/deployment/h200_smoke.py",
    "auscultation/scripts/deployment/linbox_converter_env.py",
    "auscultation/scripts/deployment/perception_environment_manifest.py",
    "auscultation/scripts/deployment/prepare_perception_environment_candidate.py",
    "auscultation/scripts/deployment/publish_perception_runtime.py",
    "auscultation/scripts/deployment/release_profile.json",
    "auscultation/scripts/deployment/release_source.py",
    "auscultation/scripts/deployment/remote_agent.py",
    "auscultation/scripts/deployment/sync_openpi_source.py",
    "auscultation/scripts/deployment/test_activate_perception_environment.py",
    "auscultation/scripts/deployment/test_perception_environment_lock_v3.py",
    "auscultation/scripts/deployment/test_prepare_perception_environment_candidate.py",
    "auscultation/scripts/deployment/test_release_source.py",
    "auscultation/scripts/deployment/verify_perception_environment.py",
    "auscultation/scripts/perception/activate_handeye.py",
    "auscultation/scripts/perception/runtime_paths.py",
    "auscultation/scripts/perception/test_activate_handeye.py",
    "auscultation/scripts/perception_capture.py",
    "auscultation/scripts/perception_worker.py",
    "auscultation/scripts/posterior_gate.py",
    "auscultation/scripts/session_to_lerobot_v2.py",
    "auscultation/scripts/training_authenticity_finalizer.py",
    "auscultation/scripts/test_il_pipeline.py",
    "auscultation/scripts/tests/test_collect_server.py",
    "auscultation/scripts/tests/test_collection_supervisor.py",
    "auscultation/scripts/tests/test_perception_capture.py",
    "auscultation/scripts/tests/test_perception_worker.py",
    "auscultation/scripts/tests/test_posterior_gate.py",
    "auscultation/scripts/train_bc_smoke.py",
)


class ReleaseSourceError(RuntimeError):
    """The workspace does not satisfy the audited code-release boundary."""


def _reject_symlink_components(workspace: Path, relative: str) -> None:
    candidate = workspace
    for part in ("",) + Path(relative).parts:
        if part:
            candidate = candidate / part
        if candidate.is_symlink():
            raise ReleaseSourceError(
                "included source path has a symlink component: %s" % candidate
            )
        if not os.path.lexists(str(candidate)):
            return


def _walk_included_tree(workspace: Path, relative_root: str, selected: set) -> None:
    _reject_symlink_components(workspace, relative_root)
    root = workspace / relative_root
    if root.is_symlink():
        raise ReleaseSourceError(
            "included source root is a symlink: %s" % relative_root
        )
    if not root.exists():
        return
    if not root.is_dir():
        raise ReleaseSourceError(
            "included source root is a non-directory: %s" % relative_root
        )

    def traversal_error(error: OSError) -> None:
        raise ReleaseSourceError(
            "unable to traverse included source: %s"
            % (error.filename or relative_root)
        ) from error

    for directory, dirnames, filenames in os.walk(
        str(root), followlinks=False, onerror=traversal_error,
    ):
        base = Path(directory)
        kept = []
        for name in dirnames:
            candidate = base / name
            if candidate.is_symlink():
                raise ReleaseSourceError(
                    "included source contains a symlink: %s" % candidate
                )
            lower_name = name.lower()
            if (
                lower_name not in EXCLUDED_DIRECTORY_NAMES
                and not lower_name.endswith(".egg-info")
            ):
                kept.append(name)
        dirnames[:] = sorted(kept)
        for name in sorted(filenames):
            candidate = base / name
            lower_name = name.lower()
            if candidate.is_symlink():
                raise ReleaseSourceError(
                    "included source contains a symlink: %s" % candidate
                )
            try:
                info = candidate.lstat()
            except OSError as exc:
                raise ReleaseSourceError(
                    "unable to inspect included source: %s" % candidate
                ) from exc
            if not stat.S_ISREG(info.st_mode):
                raise ReleaseSourceError(
                    "included source contains a non-regular file: %s" % candidate
                )
            suffix = candidate.suffix.lower()
            stem = candidate.stem.lower()
            if (
                lower_name.startswith(".env")
                or lower_name in DENIED_BASENAMES
                or (
                    suffix in RESULT_ARTIFACT_SUFFIXES
                    and (
                        stem in {"metrics", "results"}
                        or stem.endswith(DENIED_ARTIFACT_STEM_SUFFIXES)
                    )
                )
                or suffix not in ALLOWED_SUFFIXES
            ):
                continue
            selected.add(candidate.relative_to(workspace).as_posix())


def _add_top_level_files(workspace: Path, selected: set) -> None:
    auscultation = workspace / "auscultation"
    for candidate in sorted(auscultation.iterdir()):
        name = candidate.name
        include = name == "README.md" or name == "pyproject.toml" or (
            name.startswith("requirements")
            and candidate.suffix.lower() in {".in", ".lock", ".txt"}
        )
        if not include:
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise ReleaseSourceError(
                "included top-level source is a symlink or non-file: %s" % candidate
            )
        selected.add(candidate.relative_to(workspace).as_posix())


def _add_explicit_files(workspace: Path, selected: set) -> None:
    for relative in EXPLICIT_RELEASE_FILES:
        _reject_symlink_components(workspace, relative)
        candidate = workspace / relative
        if candidate.is_symlink():
            raise ReleaseSourceError(
                "explicit source is a symlink: %s" % candidate
            )
        if not candidate.exists():
            continue
        if not candidate.is_file():
            raise ReleaseSourceError(
                "explicit source is a non-file: %s" % candidate
            )
        selected.add(relative)


def select_code_release_sources(
    workspace_root: Union[str, Path],
    *,
    required_paths: Sequence[str] = REQUIRED_RELEASE_PATHS,
) -> Tuple[str, ...]:
    """Return the canonical workspace-relative file set for a code release."""
    workspace = Path(workspace_root)
    auscultation = workspace / "auscultation"
    if not auscultation.is_dir() or auscultation.is_symlink():
        raise ReleaseSourceError("workspace lacks a real auscultation directory")

    selected = set()
    for relative_root in INCLUDED_TREE_ROOTS:
        _walk_included_tree(workspace, relative_root, selected)
    _add_top_level_files(workspace, selected)
    _add_explicit_files(workspace, selected)

    missing = sorted(set(required_paths).difference(selected))
    if missing:
        raise ReleaseSourceError(
            "code release is missing required source paths: %r" % missing
        )
    return tuple(sorted(selected))
