"""Verify and atomically publish the offline official-mesh evidence release.

This module intentionally does not run Isaac Sim, activate production, or access
Linbox/H200.  It is the fail-closed release boundary for the Mac-only method
simulation artifact.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from ausc.perception.tracking import offline_official_mesh_experiment as core_experiment
from ausc.perception.tracking.offline_official_mesh_assets import (
    canonical_json_bytes as core_canonical_json_bytes,
)
from ausc.perception.tracking.offline_mesh_simulation import (
    DEFAULT_MAX_FREEZE_AGE_S,
    DEFAULT_MAX_FREEZE_ERROR_P95_M,
    GroundTruthFrame,
    OfflineSimulationError,
    PolicyFrameOutput,
    RequiredStratum,
    score_occlusion_policy_metrics,
)


class OfflineReleaseError(ValueError):
    """Raised when an offline release violates its frozen output contract."""


STATUS = "OFFLINE_METHOD_SIMULATION_ONLY"
SCOPE = "offline_method_simulation_only"

RUNTIME_MANIFEST_SHA256 = (
    "7e3307a163e8b9e6c3c32cb038f442171d0d23794182cffb1aa1c1cb983ef346"
)
ASSET_MANIFEST_SHA256 = (
    "05e66c94cd8b36ff79bef7bfcaa73d176e621120ac6398e808ad5df938a0cf35"
)
CONVERSION_MANIFEST_SHA256 = (
    "b5e826af2539bceafea1d638503362a2e8811038ad238e29bb9c263a652c3202"
)
OPENARM_COMMIT = "846a82d8baf64efdbb7f55c7f64b9b004939061b"
OPENARM_URDF_SHA256 = "63caf1116a844c7eca25e91f0acf025c2798cc4795cf727b23433eacde80ca12"
NVIDIA_HUMAN_USD_SHA256 = (
    "2c65e15587a311c5d14cdbd6a1408e0c3d8f3bec69f483150f04ca4c8f4e3ccc"
)
ROBOT_REPOSITORY_URL = "https://github.com/enactic/openarm_description.git"
HUMAN_ASSET_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/People/Characters/"
    "original_male_adult_medical_01/male_adult_medical_01.usd"
)
ASSET_RELEASE_ID = "official_assets-05e66c94cd8b36ff79be"
CONVERSION_RELEASE_ID = "official_meshes-b5e826af2539bceafea1"

POINT_NAMES = ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5")
WRIST_INDICES = (0, 1)
OCCLUSION_DURATIONS_S = (0.10, 0.25, 0.50, 1.00)
POLICY_NAMES = (
    "direct_geometry_only",
    "indefinite_world_lock_baseline",
    "bounded_freeze_then_lost",
)
OFFICIAL_GEOM_OBJECT_TYPE = 5
OFFICIAL_ROBOT_GEOM_IDS = tuple(range(21))
OFFICIAL_HUMAN_GEOM_IDS = tuple(range(21, 28))
MAX_VIDEO_BYTES = 256 * 1024 * 1024
MAX_REPORT_FRAME_BYTES = 16 * 1024 * 1024
MAX_VIDEO_FRAME_COUNT = 10_000
MAX_NPY_HEADER_BYTES = 64 * 1024

FALSE_COMPLETION_GATES = {
    "isaac_simulation_complete": False,
    "physical_collision_contact_validated": False,
    "detector_in_loop_replay_complete": False,
    "hand_eye_calibration_complete": False,
    "force_safety_complete": False,
    "hardware_timing_sync_complete": False,
    "breathing_deformation_validated": False,
    "real_patient_motion_validated": False,
    "production_ready": False,
}

CSV_HEADER = (
    "scenario_id",
    "target_name",
    "wrist_index",
    "occlusion_duration_s",
    "frame_index",
    "timestamp_s",
    "event_name",
    "observation_seed",
    "noise_contract_sha256",
    "observation_stream_sha256",
    "geometry_accepted",
    "visible_point_count_report_only",
    "accepted_estimate_x_m",
    "accepted_estimate_y_m",
    "accepted_estimate_z_m",
    "accepted_error_p95_m",
    "patient_motion_bound_m_s",
    "policy_name",
    "policy_state",
    "episode_id",
    "last_valid_age_s",
    "predicted_error_p95_m",
    "reacquired",
    "deliberately_unsafe",
    "estimate_x_m",
    "estimate_y_m",
    "estimate_z_m",
    "valid",
    "ground_truth_x_m",
    "ground_truth_y_m",
    "ground_truth_z_m",
    "error_m",
    "raw_depth_path",
    "raw_segmentation_path",
)

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SUM_RE = re.compile(r"^([0-9a-f]{64})  ([^\x00\r\n]+)$")
_MATRIX_STATUSES = {
    "COMPLETE",
    "IK_UNREACHABLE",
    "INSUFFICIENT_WRIST_ATTITUDES",
    "TRAJECTORY_INVALID",
}
_EXPECTED_IDENTITY = {
    "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
    "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
    "conversion_manifest_sha256": CONVERSION_MANIFEST_SHA256,
    "asset_release_id": ASSET_RELEASE_ID,
    "conversion_release_id": CONVERSION_RELEASE_ID,
    "robot_repository_url": ROBOT_REPOSITORY_URL,
    "robot_commit": OPENARM_COMMIT,
    "robot_urdf_sha256": OPENARM_URDF_SHA256,
    "human_asset_url": HUMAN_ASSET_URL,
    "human_usd_sha256": NVIDIA_HUMAN_USD_SHA256,
}


def _regular_bytes(path: Path, *, maximum: int | None = None) -> bytes:
    """Read one non-symlink regular file without following a final symlink."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise OfflineReleaseError(f"required file missing: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OfflineReleaseError(f"release entry is not a regular file: {path}")
    if maximum is not None and before.st_size > maximum:
        raise OfflineReleaseError(f"release file is unexpectedly large: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OfflineReleaseError(
            f"could not safely open regular file: {path}"
        ) from exc
    try:
        after = os.fstat(descriptor)
        if not stat.S_ISREG(after.st_mode) or (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ) != (after.st_dev, after.st_ino, after.st_size):
            raise OfflineReleaseError(f"release file changed while opening: {path}")
        chunks: list[bytes] = []
        remaining = after.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise OfflineReleaseError(
                    f"release file truncated while reading: {path}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise OfflineReleaseError(f"release file grew while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_regular_bytes(path)).hexdigest()


def _safe_relative(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise OfflineReleaseError(f"{context} is not a safe relative POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise OfflineReleaseError(f"{context} contains path traversal")
    if pure.as_posix() != value:
        raise OfflineReleaseError(f"{context} is not a canonical POSIX path")
    return value


def _audit_tree(root: Path) -> set[str]:
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise OfflineReleaseError(f"release directory missing: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise OfflineReleaseError(
            "release root must be a real directory, not a symlink"
        )
    files: set[str] = set()
    for directory, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        for name in list(dirnames):
            child = directory_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise OfflineReleaseError(
                    f"release contains a symlink/special directory: {child}"
                )
        for name in filenames:
            child = directory_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise OfflineReleaseError(
                    f"release contains a symlink/special file: {child}"
                )
            relative = child.relative_to(root).as_posix()
            _safe_relative(relative, context="release entry")
            if relative in files:
                raise OfflineReleaseError(f"duplicate release entry: {relative}")
            files.add(relative)
    return files


def _reject_constant(value: str) -> None:
    raise OfflineReleaseError(f"JSON contains non-finite constant {value}")


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OfflineReleaseError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            _regular_bytes(path, maximum=64 * 1024 * 1024).decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_object_no_duplicates,
        )
    except OfflineReleaseError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfflineReleaseError(f"invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise OfflineReleaseError(f"JSON document must be an object: {path}")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OfflineReleaseError("JSON value is not canonically serializable") from exc


def _keys(value: Any, expected: Iterable[str], *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise OfflineReleaseError(f"{context} schema requires an object")
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        extra = sorted(actual - expected_set)
        missing = sorted(expected_set - actual)
        raise OfflineReleaseError(
            f"{context} schema keys mismatch; unexpected={extra}, missing={missing}"
        )
    return value


def _integer(value: Any, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OfflineReleaseError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, context: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfflineReleaseError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise OfflineReleaseError(f"{context} must be finite and >= {minimum}")
    return result


def _exact_false_gates(value: Any, *, context: str) -> dict[str, bool]:
    gates = _keys(value, FALSE_COMPLETION_GATES, context=context)
    if dict(gates) != FALSE_COMPLETION_GATES:
        raise OfflineReleaseError(
            f"{context} completion gate would assert production or physical completion"
        )
    return dict(gates)


def _verify_sums(root: Path, tree_files: set[str]) -> None:
    sums_path = root / "SHA256SUMS.txt"
    raw = _regular_bytes(sums_path, maximum=64 * 1024 * 1024)
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise OfflineReleaseError(
            "SHA256SUMS.txt must use canonical LF-terminated lines"
        )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise OfflineReleaseError("SHA256SUMS.txt is not UTF-8") from exc
    entries: list[tuple[str, str]] = []
    for line in lines:
        match = _SUM_RE.fullmatch(line)
        if match is None:
            raise OfflineReleaseError("SHA256SUMS.txt has an invalid checksum line")
        digest, relative = match.groups()
        _safe_relative(relative, context="SHA256SUMS path")
        if relative == "SHA256SUMS.txt":
            raise OfflineReleaseError("SHA256SUMS.txt must not hash itself")
        entries.append((digest, relative))
    paths = [relative for _, relative in entries]
    expected = tree_files - {"SHA256SUMS.txt"}
    if len(paths) != len(set(paths)) or set(paths) != expected:
        raise OfflineReleaseError(
            "SHA256SUMS.txt coverage must include every file exactly once except itself"
        )
    if paths != sorted(paths):
        raise OfflineReleaseError("SHA256SUMS.txt paths must be sorted")
    for expected_digest, relative in entries:
        if _sha256_file(root / relative) != expected_digest:
            raise OfflineReleaseError(f"SHA256SUMS mismatch for {relative}")


def _verify_frozen_manifests(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        "raw/runtime_manifest.json": RUNTIME_MANIFEST_SHA256,
        "raw/asset_manifest.json": ASSET_MANIFEST_SHA256,
        "raw/conversion_manifest.json": CONVERSION_MANIFEST_SHA256,
    }
    for relative, digest in expected.items():
        if _sha256_file(root / relative) != digest:
            raise OfflineReleaseError(
                f"frozen identity manifest SHA-256 mismatch: {relative}"
            )
    asset = _load_json(root / "raw/asset_manifest.json")
    conversion = _load_json(root / "raw/conversion_manifest.json")
    try:
        asset_robot = asset["robot_source"]
        asset_human = asset["human_source"]
        if (
            asset_robot["repository_url"] != ROBOT_REPOSITORY_URL
            or asset_robot["commit"] != OPENARM_COMMIT
            or asset_robot["urdf_sha256"] != OPENARM_URDF_SHA256
            or asset_human["asset_url"] != HUMAN_ASSET_URL
            or asset_human["usd_sha256"] != NVIDIA_HUMAN_USD_SHA256
            or conversion["asset_manifest_sha256"] != ASSET_MANIFEST_SHA256
            or conversion["asset_release_id"] != ASSET_RELEASE_ID
            or conversion["robot"]["urdf_sha256"] != OPENARM_URDF_SHA256
            or conversion["human"]["usd_sha256"] != NVIDIA_HUMAN_USD_SHA256
        ):
            raise OfflineReleaseError("frozen robot/human identity cross-link mismatch")
    except (KeyError, TypeError) as exc:
        raise OfflineReleaseError("frozen manifest identity schema is invalid") from exc
    return asset, conversion


def _validate_manifest(root: Path) -> dict[str, Any]:
    manifest = _load_json(root / "raw/experiment_manifest.json")
    _keys(
        manifest,
        (
            "schema",
            "status",
            "scope",
            "production_activation",
            "isaac_equivalence",
            "identities",
            "render",
            "matrix",
            "policies",
            "artifacts",
            "determinism",
            "completion_gates",
        ),
        context="experiment manifest",
    )
    if (
        manifest["schema"] != "wam_offline_official_mesh_occlusion_experiment_v2"
        or manifest["status"] != STATUS
        or manifest["scope"] != SCOPE
        or manifest["production_activation"] is not False
        or manifest["isaac_equivalence"] is not False
    ):
        raise OfflineReleaseError("experiment manifest status/scope schema is invalid")

    identities = _keys(
        manifest["identities"], _EXPECTED_IDENTITY, context="experiment identities"
    )
    if dict(identities) != _EXPECTED_IDENTITY:
        raise OfflineReleaseError("robot/human/runtime identity or commit mismatch")

    render = _keys(
        manifest["render"],
        (
            "codec",
            "pixel_format",
            "width",
            "height",
            "fps",
            "frame_count",
            "undistorted_pinhole_only",
            "raw_depth_dtype",
            "raw_segmentation_dtype",
            "raw_segmentation_channels",
            "segmentation_object_type",
            "robot_geom_ids",
            "human_geom_ids",
        ),
        context="render",
    )
    if (
        render["codec"] != "h264"
        or render["pixel_format"] != "yuv420p"
        or render["width"] != 640
        or render["height"] != 360
        or render["fps"] != 12
        or render["undistorted_pinhole_only"] is not True
        or render["raw_depth_dtype"] != "float32"
        or render["raw_segmentation_dtype"] != "int32"
        or render["raw_segmentation_channels"] != 2
    ):
        raise OfflineReleaseError(
            "render schema requires H.264 yuv420p 640x360 at 12 FPS"
        )
    _integer(render["frame_count"], context="render.frame_count", minimum=1)
    geom_type = _integer(
        render["segmentation_object_type"], context="render segmentation object type"
    )
    if (
        geom_type != OFFICIAL_GEOM_OBJECT_TYPE
        or render["robot_geom_ids"] != list(OFFICIAL_ROBOT_GEOM_IDS)
        or render["human_geom_ids"] != list(OFFICIAL_HUMAN_GEOM_IDS)
    ):
        raise OfflineReleaseError(
            "render provenance does not bind the exact compiled official geom inventory"
        )

    matrix = _keys(
        manifest["matrix"],
        (
            "point_names",
            "wrist_indices",
            "occlusion_durations_s",
            "required_strata_count",
            "records",
        ),
        context="matrix",
    )
    if (
        matrix["point_names"] != list(POINT_NAMES)
        or matrix["wrist_indices"] != list(WRIST_INDICES)
        or matrix["occlusion_durations_s"] != list(OCCLUSION_DURATIONS_S)
        or matrix["required_strata_count"] != 80
        or not isinstance(matrix["records"], list)
        or len(matrix["records"]) != 80
    ):
        raise OfflineReleaseError("experiment matrix schema/required strata mismatch")
    expected_combinations = {
        (point, wrist, duration)
        for point in POINT_NAMES
        for wrist in WRIST_INDICES
        for duration in OCCLUSION_DURATIONS_S
    }
    actual_combinations: set[tuple[str, int, float]] = set()
    scenario_ids: set[str] = set()
    for index, record_value in enumerate(matrix["records"]):
        record = _keys(
            record_value,
            (
                "scenario_id",
                "target_name",
                "wrist_index",
                "occlusion_duration_s",
                "status",
                "frame_count",
                "observation_seed",
                "patient_motion_seed",
                "observation_stream_sha256",
            ),
            context=f"matrix.records[{index}]",
        )
        if record["target_name"] not in POINT_NAMES:
            raise OfflineReleaseError(
                "matrix record target is outside the declared points"
            )
        wrist_index = _integer(
            record["wrist_index"], context=f"matrix.records[{index}].wrist_index",
        )
        duration_s = _number(
            record["occlusion_duration_s"],
            context=f"matrix.records[{index}].occlusion_duration_s",
            minimum=0.0,
        )
        if not isinstance(record["scenario_id"], str) or not isinstance(
            record["status"], str
        ):
            raise OfflineReleaseError(
                "matrix record scenario ID/status must be strings"
            )
        combination = (
            record["target_name"],
            wrist_index,
            duration_s,
        )
        milliseconds = int(round(combination[2] * 1000.0))
        expected_id = f"{combination[0]}__w{combination[1]}__occ{milliseconds:04d}ms"
        if (
            combination not in expected_combinations
            or record["scenario_id"] != expected_id
            or record["scenario_id"] in scenario_ids
            or record["status"] not in _MATRIX_STATUSES
        ):
            raise OfflineReleaseError(
                "matrix record scenario/target/wrist/status mismatch"
            )
        frame_count = _integer(
            record["frame_count"], context=f"matrix.records[{index}].frame_count"
        )
        _integer(
            record["observation_seed"],
            context=f"matrix.records[{index}].observation_seed",
        )
        _integer(
            record["patient_motion_seed"],
            context=f"matrix.records[{index}].patient_motion_seed",
        )
        if not isinstance(
            record["observation_stream_sha256"], str
        ) or not _SHA_RE.fullmatch(record["observation_stream_sha256"]):
            raise OfflineReleaseError("matrix observation stream SHA-256 is invalid")
        if (record["status"] == "COMPLETE") != (frame_count > 0):
            raise OfflineReleaseError("matrix COMPLETE status and frame count disagree")
        actual_combinations.add(combination)
        scenario_ids.add(record["scenario_id"])
    if actual_combinations != expected_combinations:
        raise OfflineReleaseError(
            "matrix does not cover each declared stratum exactly once"
        )

    if manifest["policies"] != list(POLICY_NAMES):
        raise OfflineReleaseError(
            "policy list/order is not the frozen three-policy contract"
        )
    artifacts = _keys(
        manifest["artifacts"],
        (
            "video_path",
            "report_frame_path",
            "metrics_path",
            "per_frame_csv_path",
            "determinism_rerun_csv_path",
            "input_contract_path",
            "video_timeline_path",
            "raw_frame_count",
        ),
        context="artifacts",
    )
    expected_paths = {
        "video_path": "offline_official_mesh_occlusion_h264.mp4",
        "report_frame_path": "report_frame.png",
        "metrics_path": "raw/metrics.json",
        "per_frame_csv_path": "raw/per_frame.csv",
        "determinism_rerun_csv_path": "raw/determinism_rerun/per_frame.csv",
        "input_contract_path": "raw/input_contract.json",
        "video_timeline_path": "raw/video_timeline.json",
    }
    for key, expected_path in expected_paths.items():
        _safe_relative(artifacts[key], context=f"artifacts.{key}")
        if artifacts[key] != expected_path:
            raise OfflineReleaseError(f"artifacts.{key} is not the frozen path")
    _integer(
        artifacts["raw_frame_count"], context="artifacts.raw_frame_count", minimum=1
    )

    determinism = _keys(
        manifest["determinism"],
        (
            "input_contract_sha256",
            "core_input_contract_sha256",
            "first_run_per_frame_sha256",
            "same_input_rerun_per_frame_sha256",
            "same_input_match",
        ),
        context="determinism",
    )
    for key in (
        "input_contract_sha256",
        "core_input_contract_sha256",
        "first_run_per_frame_sha256",
        "same_input_rerun_per_frame_sha256",
    ):
        if not isinstance(determinism[key], str) or not _SHA_RE.fullmatch(
            determinism[key]
        ):
            raise OfflineReleaseError(f"determinism.{key} is not SHA-256")
    if determinism["same_input_match"] is not True:
        raise OfflineReleaseError(
            "same-input determinism rerun is not declared matching"
        )
    _exact_false_gates(manifest["completion_gates"], context="experiment")
    return manifest


def _validate_input_contract(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], str, str]:
    path = root / manifest["artifacts"]["input_contract_path"]
    raw = _regular_bytes(path, maximum=16 * 1024 * 1024)
    contract = _load_json(path)
    if raw != _canonical_json_bytes(contract):
        raise OfflineReleaseError("input contract JSON is not canonical")
    _keys(
        contract,
        (
            "schema",
            "status",
            "scope",
            "production_activation",
            "core_contract_sha256",
            "core_contract",
            "identities",
            "policy_contract",
            "matrix",
            "render_provenance",
            "strata",
        ),
        context="input contract",
    )
    if (
        contract["schema"] != "wam_offline_official_mesh_occlusion_input_contract_v2"
        or contract["status"] != STATUS
        or contract["scope"] != SCOPE
        or contract["production_activation"] is not False
    ):
        raise OfflineReleaseError("input contract status/scope schema is invalid")

    (
        immutable_core,
        expected_core_sha256,
    ) = core_experiment.build_experiment_input_contract()
    expected_core = dict(immutable_core)
    expected_core["decision_surface"] = dict(immutable_core["decision_surface"])
    try:
        expected_core = json.loads(core_canonical_json_bytes(expected_core))
    except (TypeError, ValueError) as exc:  # pragma: no cover - frozen core invariant
        raise OfflineReleaseError(
            "frozen core input contract is not serializable"
        ) from exc
    if (
        hashlib.sha256(core_canonical_json_bytes(expected_core)).hexdigest()
        != expected_core_sha256
    ):
        raise OfflineReleaseError(
            "frozen core input contract hash is internally invalid"
        )
    if (
        contract["core_contract_sha256"] != expected_core_sha256
        or core_canonical_json_bytes(contract["core_contract"])
        != core_canonical_json_bytes(expected_core)
        or manifest["determinism"]["core_input_contract_sha256"] != expected_core_sha256
    ):
        raise OfflineReleaseError(
            "input wrapper does not embed the exact immutable sparse-core contract"
        )
    expected_identity = {
        key: _EXPECTED_IDENTITY[key]
        for key in (
            "runtime_manifest_sha256",
            "asset_manifest_sha256",
            "conversion_manifest_sha256",
        )
    }
    identities = _keys(
        contract["identities"], expected_identity, context="input contract identities"
    )
    if dict(identities) != expected_identity:
        raise OfflineReleaseError("input contract identity provenance mismatch")
    policy = _keys(
        contract["policy_contract"],
        (
            "policy_names",
            "max_freeze_age_s",
            "max_freeze_error_p95_m",
            "patient_motion_bound_m_s",
        ),
        context="input policy contract",
    )
    declared_motion_bound = _number(
        policy["patient_motion_bound_m_s"],
        context="input patient motion bound",
        minimum=0.0,
    )
    if (
        policy["policy_names"] != list(POLICY_NAMES)
        or _number(
            policy["max_freeze_age_s"], context="input max_freeze_age_s", minimum=0.0,
        )
        != DEFAULT_MAX_FREEZE_AGE_S
        or _number(
            policy["max_freeze_error_p95_m"],
            context="input max_freeze_error_p95_m",
            minimum=0.0,
        )
        != DEFAULT_MAX_FREEZE_ERROR_P95_M
        or declared_motion_bound != float(expected_core["patient_motion_bound_m_s"])
    ):
        raise OfflineReleaseError(
            "input policy thresholds/motion bound differ from the frozen core"
        )
    matrix = _keys(
        contract["matrix"],
        (
            "point_names",
            "wrist_indices",
            "occlusion_durations_s",
            "required_strata_count",
        ),
        context="input matrix",
    )
    manifest_matrix = manifest["matrix"]
    if dict(matrix) != {
        key: manifest_matrix[key]
        for key in (
            "point_names",
            "wrist_indices",
            "occlusion_durations_s",
            "required_strata_count",
        )
    }:
        raise OfflineReleaseError("input matrix does not match the experiment matrix")
    if dict(matrix) != {
        "point_names": expected_core["point_names"],
        "wrist_indices": expected_core["wrist_indices"],
        "occlusion_durations_s": expected_core["occlusion_durations_s"],
        "required_strata_count": expected_core["required_strata_count"],
    }:
        raise OfflineReleaseError("input matrix differs from the frozen sparse core")
    provenance = _keys(
        contract["render_provenance"],
        (
            "width",
            "height",
            "raw_depth_dtype",
            "raw_segmentation_dtype",
            "raw_segmentation_channels",
            "segmentation_object_type",
            "robot_geom_ids",
            "human_geom_ids",
        ),
        context="input render provenance",
    )
    render = manifest["render"]
    expected_provenance = {
        key: render[key]
        for key in (
            "width",
            "height",
            "raw_depth_dtype",
            "raw_segmentation_dtype",
            "raw_segmentation_channels",
            "segmentation_object_type",
            "robot_geom_ids",
            "human_geom_ids",
        )
    }
    if (
        dict(provenance) != expected_provenance
        or provenance["segmentation_object_type"] != OFFICIAL_GEOM_OBJECT_TYPE
        or provenance["robot_geom_ids"] != list(OFFICIAL_ROBOT_GEOM_IDS)
        or provenance["human_geom_ids"] != list(OFFICIAL_HUMAN_GEOM_IDS)
    ):
        raise OfflineReleaseError(
            "input contract does not bind exact official geom provenance"
        )

    strata = contract["strata"]
    records = manifest_matrix["records"]
    if not isinstance(strata, list) or len(strata) != len(records):
        raise OfflineReleaseError("input contract strata inventory is incomplete")
    for index, (stratum_value, record) in enumerate(zip(strata, records, strict=True)):
        stratum = _keys(
            stratum_value,
            (
                "scenario_id",
                "target_name",
                "wrist_index",
                "occlusion_duration_s",
                "status",
                "frame_count",
                "observation_seed",
                "patient_motion_seed",
                "patient_motion_direction_world",
                "events",
            ),
            context=f"input strata[{index}]",
        )
        expected_record = {
            key: record[key]
            for key in (
                "scenario_id",
                "target_name",
                "wrist_index",
                "occlusion_duration_s",
                "status",
                "frame_count",
                "observation_seed",
                "patient_motion_seed",
            )
        }
        if {key: stratum[key] for key in expected_record} != expected_record:
            raise OfflineReleaseError("input stratum seed/schedule identity mismatch")

        target_index = POINT_NAMES.index(record["target_name"])
        wrist_index = int(record["wrist_index"])
        duration_index = OCCLUSION_DURATIONS_S.index(
            float(record["occlusion_duration_s"])
        )
        sparse_index = target_index * 8 + wrist_index * 4 + duration_index
        expected_observation_seed = int(expected_core["observation_seed_root"]) + (
            10 * sparse_index
        )
        expected_patient_seed = int(expected_core["patient_motion_seed_root"]) + (
            sparse_index
        )
        if (
            record["observation_seed"] != expected_observation_seed
            or record["patient_motion_seed"] != expected_patient_seed
        ):
            raise OfflineReleaseError("input stratum seeds differ from the sparse core")
        direction_value = stratum["patient_motion_direction_world"]
        if not isinstance(direction_value, list) or len(direction_value) != 3:
            raise OfflineReleaseError("input patient motion direction is invalid")
        direction = np.asarray(
            [
                _number(value, context="input patient motion direction")
                for value in direction_value
            ],
            dtype=np.float64,
        )
        expected_direction = core_experiment._patient_direction(expected_patient_seed)
        if not np.array_equal(direction, expected_direction):
            raise OfflineReleaseError(
                "input patient motion direction differs from its frozen seed"
            )

        events_value = stratum["events"]
        if not isinstance(events_value, list):
            raise OfflineReleaseError("input stratum events must be an array")
        if record["status"] == "COMPLETE":
            expected_names = ("pre", "contact", "post")
            duration = float(record["occlusion_duration_s"])
            expected_times = (
                0.0,
                duration,
                duration + core_experiment.POST_EVENT_DELTA_S,
            )
            if len(events_value) != len(expected_names) or record["frame_count"] != len(
                expected_names
            ):
                raise OfflineReleaseError(
                    "input completed stratum does not declare the sparse-core events"
                )
            for frame_index, (event_value, event_name, timestamp) in enumerate(
                zip(events_value, expected_names, expected_times, strict=True)
            ):
                event = _keys(
                    event_value,
                    (
                        "frame_index",
                        "event_name",
                        "timestamp_s",
                        "observation_seed",
                        "noise_contract_sha256",
                    ),
                    context=f"input strata[{index}].events[{frame_index}]",
                )
                event_seed = _integer(
                    event["observation_seed"], context="input event observation seed",
                )
                noise_sha = event["noise_contract_sha256"]
                if (
                    _integer(event["frame_index"], context="input event frame index")
                    != frame_index
                    or event["event_name"] != event_name
                    or not math.isclose(
                        _number(
                            event["timestamp_s"],
                            context="input event timestamp",
                            minimum=0.0,
                        ),
                        timestamp,
                        abs_tol=1.0e-12,
                    )
                    or event_seed != expected_observation_seed + frame_index
                    or not isinstance(noise_sha, str)
                    or not _SHA_RE.fullmatch(noise_sha)
                    or noise_sha
                    != core_experiment.make_predeclared_observation_noise(
                        event_seed
                    ).contract_sha256
                ):
                    raise OfflineReleaseError(
                        "input event schedule/noise differs from the sparse core"
                    )
        elif record["frame_count"] != 0 or events_value:
            raise OfflineReleaseError(
                "failed input stratum cannot declare scoring events"
            )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["determinism"]["input_contract_sha256"]:
        raise OfflineReleaseError(
            "input contract SHA-256 does not match determinism identity"
        )
    return contract, digest, expected_core_sha256


def _csv_bool(value: str, *, context: str) -> bool:
    if value not in ("true", "false"):
        raise OfflineReleaseError(f"CSV {context} boolean must be lowercase true/false")
    return value == "true"


def _csv_int(value: str, *, context: str, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise OfflineReleaseError(f"CSV {context} is not an integer") from exc
    if value != str(parsed) or parsed < minimum:
        raise OfflineReleaseError(f"CSV {context} is not a canonical integer")
    return parsed


def _csv_float(value: str, *, context: str, minimum: float | None = None) -> float:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise OfflineReleaseError(f"CSV {context} is empty/noncanonical")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise OfflineReleaseError(f"CSV {context} is not numeric") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise OfflineReleaseError(f"CSV {context} is outside its finite range")
    return parsed


def _csv_vector(
    row: Mapping[str, str], names: tuple[str, str, str], *, context: str
) -> np.ndarray:
    return np.asarray(
        [_csv_float(row[name], context=f"{context}.{name}") for name in names],
        dtype=np.float64,
    )


def _load_csv(path: Path) -> list[dict[str, str]]:
    raw = _regular_bytes(path, maximum=512 * 1024 * 1024)
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise OfflineReleaseError("per-frame CSV must be LF terminated")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OfflineReleaseError("per-frame CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != CSV_HEADER:
        raise OfflineReleaseError("CSV header/order does not match the frozen schema")
    rows: list[dict[str, str]] = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise OfflineReleaseError("CSV row has a column-count/schema mismatch")
        rows.append(dict(row))
    if not rows:
        raise OfflineReleaseError("per-frame CSV has no eligible rows")
    return rows


def _optional_csv_float(value: str, *, context: str) -> float | None:
    return None if value == "" else _csv_float(value, context=context, minimum=0.0)


def _load_exact_npy(
    path: Path, *, expected_shape: tuple[int, ...], expected_dtype: np.dtype[Any]
) -> np.ndarray:
    data_bytes = math.prod(expected_shape) * expected_dtype.itemsize
    raw = _regular_bytes(path, maximum=data_bytes + MAX_NPY_HEADER_BYTES)
    if raw.startswith(b"PK\x03\x04"):
        raise OfflineReleaseError(f"raw NumPy product is an NPZ, not NPY: {path}")
    if not raw.startswith(b"\x93NUMPY"):
        raise OfflineReleaseError(f"raw NumPy product lacks NPY magic: {path}")
    stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise OfflineReleaseError("raw NPY version must be the bounded v1.0 format")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
            stream, max_header_size=MAX_NPY_HEADER_BYTES
        )
    except OfflineReleaseError:
        raise
    except Exception as exc:
        raise OfflineReleaseError(f"raw NPY header is invalid: {path}") from exc
    if (
        tuple(shape) != expected_shape
        or np.dtype(dtype) != expected_dtype
        or fortran_order
        or stream.tell() > MAX_NPY_HEADER_BYTES
        or len(raw) != stream.tell() + data_bytes
    ):
        raise OfflineReleaseError(
            f"raw NPY dtype/shape/header/byte length mismatch: {path}"
        )
    try:
        return np.frombuffer(raw, dtype=expected_dtype, offset=stream.tell()).reshape(
            expected_shape
        )
    except (TypeError, ValueError) as exc:
        raise OfflineReleaseError(f"raw NPY payload is invalid: {path}") from exc


def _observation_stream_payload(
    *,
    core_input_contract_sha256: str,
    record: Mapping[str, Any],
    stratum: Mapping[str, Any],
    frame_groups: Mapping[tuple[str, int], list[dict[str, str]]],
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for (scenario_id, _frame_index), group in sorted(frame_groups.items()):
        if scenario_id != record["scenario_id"]:
            continue
        row = group[0]
        accepted = row["geometry_accepted"] == "true"
        events.append(
            {
                "frame_index": int(row["frame_index"]),
                "event_name": row["event_name"],
                "timestamp_s": float(row["timestamp_s"]),
                "observation_seed": int(row["observation_seed"]),
                "noise_contract_sha256": row["noise_contract_sha256"],
                "geometry_accepted": accepted,
                "accepted_estimate_world_m": (
                    [
                        float(row["accepted_estimate_x_m"]),
                        float(row["accepted_estimate_y_m"]),
                        float(row["accepted_estimate_z_m"]),
                    ]
                    if accepted
                    else None
                ),
                "accepted_error_p95_m": (
                    float(row["accepted_error_p95_m"]) if accepted else None
                ),
                "visible_point_count_report_only": int(
                    row["visible_point_count_report_only"]
                ),
                "target_ground_truth_world_m": [
                    float(row["ground_truth_x_m"]),
                    float(row["ground_truth_y_m"]),
                    float(row["ground_truth_z_m"]),
                ],
            }
        )
    return {
        "schema": "wam_offline_sparse_observation_stream_v1",
        "input_contract_sha256": core_input_contract_sha256,
        "scenario_id": record["scenario_id"],
        "target_name": record["target_name"],
        "wrist_index": record["wrist_index"],
        "occlusion_duration_s": record["occlusion_duration_s"],
        "execution_status": record["status"],
        "observation_seed": record["observation_seed"],
        "patient_motion_seed": record["patient_motion_seed"],
        "patient_motion_direction_world": stratum["patient_motion_direction_world"],
        "events": events,
    }


def _validate_csv_and_raw(
    root: Path,
    manifest: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    core_input_contract_sha256: str,
) -> tuple[
    list[dict[str, str]],
    dict[str, int],
    set[str],
    dict[str, tuple[PolicyFrameOutput, ...]],
    tuple[GroundTruthFrame, ...],
    tuple[RequiredStratum, ...],
    dict[tuple[str, int], dict[str, str]],
]:
    artifacts = manifest["artifacts"]
    csv_path = root / artifacts["per_frame_csv_path"]
    rerun_path = root / artifacts["determinism_rerun_csv_path"]
    csv_bytes = _regular_bytes(csv_path, maximum=512 * 1024 * 1024)
    rerun_bytes = _regular_bytes(rerun_path, maximum=512 * 1024 * 1024)
    first_sha = hashlib.sha256(csv_bytes).hexdigest()
    rerun_sha = hashlib.sha256(rerun_bytes).hexdigest()
    determinism = manifest["determinism"]
    if (
        csv_bytes != rerun_bytes
        or first_sha != rerun_sha
        or first_sha != determinism["first_run_per_frame_sha256"]
        or rerun_sha != determinism["same_input_rerun_per_frame_sha256"]
    ):
        raise OfflineReleaseError("same-input determinism rerun is not byte exact")
    rows = _load_csv(csv_path)
    records = {
        record["scenario_id"]: record for record in manifest["matrix"]["records"]
    }
    input_strata = {
        stratum["scenario_id"]: stratum for stratum in input_contract["strata"]
    }
    record_order = {
        record["scenario_id"]: index
        for index, record in enumerate(manifest["matrix"]["records"])
    }
    frame_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    policy_outputs: dict[str, list[PolicyFrameOutput]] = {
        policy_name: [] for policy_name in POLICY_NAMES
    }
    raw_paths: set[str] = set()
    parsed_truth: dict[tuple[str, int], GroundTruthFrame] = {}
    shared_fields = (
        "target_name",
        "wrist_index",
        "occlusion_duration_s",
        "timestamp_s",
        "event_name",
        "observation_seed",
        "noise_contract_sha256",
        "observation_stream_sha256",
        "geometry_accepted",
        "visible_point_count_report_only",
        "accepted_estimate_x_m",
        "accepted_estimate_y_m",
        "accepted_estimate_z_m",
        "accepted_error_p95_m",
        "patient_motion_bound_m_s",
        "reacquired",
        "ground_truth_x_m",
        "ground_truth_y_m",
        "ground_truth_z_m",
        "raw_depth_path",
        "raw_segmentation_path",
    )
    for row_number, row in enumerate(rows, start=2):
        context = f"row {row_number}"
        scenario = row["scenario_id"]
        if scenario not in records or records[scenario]["status"] != "COMPLETE":
            raise OfflineReleaseError(
                f"CSV {context} references a non-COMPLETE scenario"
            )
        record = records[scenario]
        wrist = _csv_int(row["wrist_index"], context=f"{context}.wrist_index")
        duration = _csv_float(
            row["occlusion_duration_s"], context=f"{context}.occlusion_duration_s"
        )
        frame_index = _csv_int(row["frame_index"], context=f"{context}.frame_index")
        timestamp = _csv_float(
            row["timestamp_s"], context=f"{context}.timestamp_s", minimum=0.0
        )
        if not isinstance(row["event_name"], str) or not row["event_name"]:
            raise OfflineReleaseError("CSV event_name must be nonempty")
        _csv_int(row["observation_seed"], context=f"{context}.observation_seed")
        if not _SHA_RE.fullmatch(row["noise_contract_sha256"]):
            raise OfflineReleaseError("CSV noise contract SHA-256 is invalid")
        if (
            row["target_name"] != record["target_name"]
            or wrist != record["wrist_index"]
            or not math.isclose(duration, record["occlusion_duration_s"], abs_tol=1e-12)
            or row["observation_stream_sha256"] != record["observation_stream_sha256"]
        ):
            raise OfflineReleaseError(
                "CSV scenario/target/wrist/stream cross-link mismatch"
            )
        accepted = _csv_bool(row["geometry_accepted"], context="geometry_accepted")
        visible_count = _csv_int(
            row["visible_point_count_report_only"],
            context="visible_point_count_report_only",
        )
        if visible_count > len(POINT_NAMES):
            raise OfflineReleaseError("CSV report-only point count exceeds ten")
        motion_bound = _csv_float(
            row["patient_motion_bound_m_s"],
            context="patient_motion_bound_m_s",
            minimum=0.0,
        )
        accepted_names = (
            "accepted_estimate_x_m",
            "accepted_estimate_y_m",
            "accepted_estimate_z_m",
        )
        if accepted:
            accepted_estimate = _csv_vector(row, accepted_names, context="accepted")
            accepted_error = _csv_float(
                row["accepted_error_p95_m"],
                context="accepted_error_p95_m",
                minimum=0.0,
            )
        else:
            if any(
                row[name] != "" for name in (*accepted_names, "accepted_error_p95_m")
            ):
                raise OfflineReleaseError(
                    "CSV rejected geometry must have null accepted estimate/p95"
                )
            accepted_estimate = None
            accepted_error = None
        policy_name = row["policy_name"]
        if policy_name not in POLICY_NAMES:
            raise OfflineReleaseError("CSV policy name is outside the frozen contract")
        valid = _csv_bool(row["valid"], context="valid")
        estimate_names = ("estimate_x_m", "estimate_y_m", "estimate_z_m")
        ground_truth = _csv_vector(
            row,
            ("ground_truth_x_m", "ground_truth_y_m", "ground_truth_z_m"),
            context="ground_truth",
        )
        if valid:
            estimate = _csv_vector(row, estimate_names, context="estimate")
            error = _csv_float(row["error_m"], context="error_m", minimum=0.0)
            if not math.isclose(
                error, float(np.linalg.norm(estimate - ground_truth)), abs_tol=2e-12
            ):
                raise OfflineReleaseError(
                    "CSV error_m does not match estimate/ground truth"
                )
        else:
            estimate = None
            if any(row[name] != "" for name in (*estimate_names, "error_m")):
                raise OfflineReleaseError("CSV invalid output must null estimate/error")
        episode_id = None if row["episode_id"] == "" else row["episode_id"]
        last_age = _optional_csv_float(
            row["last_valid_age_s"], context="last_valid_age_s"
        )
        predicted_error = _optional_csv_float(
            row["predicted_error_p95_m"], context="predicted_error_p95_m"
        )
        reacquired = _csv_bool(row["reacquired"], context="reacquired")
        deliberately_unsafe = _csv_bool(
            row["deliberately_unsafe"], context="deliberately_unsafe"
        )
        try:
            output = PolicyFrameOutput(
                policy_name=policy_name,
                scenario_id=scenario,
                target_name=row["target_name"],
                wrist_index=wrist,
                frame_index=frame_index,
                timestamp_s=timestamp,
                geometry_accepted=accepted,
                patient_motion_bound_m_s=motion_bound,
                state=row["policy_state"],
                estimate_world_m=estimate,
                valid=valid,
                episode_id=episode_id,
                last_valid_age_s=last_age,
                predicted_error_p95_m=predicted_error,
                reacquired=reacquired,
                deliberately_unsafe=deliberately_unsafe,
                visible_point_count_report_only=visible_count,
                observation_stream_sha256=row["observation_stream_sha256"],
            )
        except (OfflineSimulationError, TypeError, ValueError) as exc:
            raise OfflineReleaseError(
                f"CSV PolicyFrameOutput contract invalid: {exc}"
            ) from exc
        if accepted and (
            output.estimate_world_m is None
            or accepted_estimate is None
            or not np.array_equal(output.estimate_world_m, accepted_estimate)
            or output.predicted_error_p95_m != accepted_error
        ):
            raise OfflineReleaseError(
                "CSV accepted PolicyFrameOutput differs from accepted observation"
            )
        policy_outputs[policy_name].append(output)
        truth_key = (scenario, frame_index)
        try:
            truth = GroundTruthFrame(
                scenario_id=scenario,
                target_name=row["target_name"],
                wrist_index=wrist,
                frame_index=frame_index,
                timestamp_s=timestamp,
                observation_stream_sha256=row["observation_stream_sha256"],
                target_world_m=ground_truth,
            )
        except (OfflineSimulationError, TypeError, ValueError) as exc:
            raise OfflineReleaseError(
                f"CSV ground-truth contract invalid: {exc}"
            ) from exc
        prior_truth = parsed_truth.setdefault(truth_key, truth)
        if (
            prior_truth.target_name != truth.target_name
            or prior_truth.wrist_index != truth.wrist_index
            or prior_truth.timestamp_s != truth.timestamp_s
            or prior_truth.observation_stream_sha256 != truth.observation_stream_sha256
            or not np.array_equal(prior_truth.target_world_m, truth.target_world_m)
        ):
            raise OfflineReleaseError(
                "CSV ground truth differs across aligned policies"
            )
        for name in ("raw_depth_path", "raw_segmentation_path"):
            relative = _safe_relative(row[name], context=f"CSV {name}")
            if not relative.startswith("raw/") or not relative.endswith(".npy"):
                raise OfflineReleaseError(f"CSV {name} must reference a raw .npy file")
            raw_paths.add(relative)
        frame_groups[truth_key].append(row)

    if len(rows) != len(frame_groups) * len(POLICY_NAMES):
        raise OfflineReleaseError(
            "CSV must contain exactly one row per frame and policy"
        )
    for group in frame_groups.values():
        if (
            len(group) != len(POLICY_NAMES)
            or tuple(row["policy_name"] for row in group) != POLICY_NAMES
        ):
            raise OfflineReleaseError(
                "CSV frame policies must occur exactly once in frozen policy order"
            )
        reference = group[0]
        for row in group[1:]:
            if any(row[name] != reference[name] for name in shared_fields):
                raise OfflineReleaseError(
                    "CSV policies did not receive the exact same observation"
                )
    scenario_counts = Counter(scenario for scenario, _ in frame_groups)
    for scenario, record in records.items():
        if scenario_counts.get(scenario, 0) != record["frame_count"]:
            raise OfflineReleaseError(
                "CSV frame count does not match matrix denominator"
            )
        payload = _observation_stream_payload(
            core_input_contract_sha256=core_input_contract_sha256,
            record=record,
            stratum=input_strata[scenario],
            frame_groups=frame_groups,
        )
        recomputed = hashlib.sha256(core_canonical_json_bytes(payload)).hexdigest()
        if recomputed != record["observation_stream_sha256"]:
            raise OfflineReleaseError(
                f"observation stream SHA-256 does not match recomputed payload: {scenario}"
            )
    schedule_by_scenario = {
        row["scenario_id"]: row for row in manifest["matrix"]["records"]
    }
    for scenario, record in schedule_by_scenario.items():
        if record["status"] != "COMPLETE":
            continue
        ordered = [
            frame_groups[(scenario, frame_index)][0]
            for frame_index in range(record["frame_count"])
            if (scenario, frame_index) in frame_groups
        ]
        if len(ordered) != record["frame_count"]:
            raise OfflineReleaseError("CSV frame indices are not contiguous from zero")
        timestamps = [float(row["timestamp_s"]) for row in ordered]
        if timestamps[0] != 0.0:
            raise OfflineReleaseError(
                "CSV timestamps do not start at the declared origin"
            )

    sort_key = lambda row: (record_order[row.scenario_id], row.frame_index,)
    outputs_tuple = {
        name: tuple(sorted(policy_outputs[name], key=sort_key)) for name in POLICY_NAMES
    }
    truths = tuple(
        truth
        for _, truth in sorted(
            parsed_truth.items(),
            key=lambda item: (record_order[item[0][0]], item[0][1]),
        )
    )
    try:
        required = tuple(
            RequiredStratum(
                scenario_id=record["scenario_id"],
                target_name=record["target_name"],
                wrist_index=record["wrist_index"],
                observation_stream_sha256=record["observation_stream_sha256"],
                observation_seed=record["observation_seed"],
                patient_motion_seed=record["patient_motion_seed"],
                execution_status=record["status"],
                declared_frame_count=record["frame_count"],
            )
            for record in manifest["matrix"]["records"]
        )
        # This call performs the exact cross-policy replay, including the
        # 0.40 s / 0.010 m decide_occlusion_policy boundary.
        score_occlusion_policy_metrics(outputs_tuple, truths, required_strata=required)
    except (OfflineSimulationError, TypeError, ValueError) as exc:
        raise OfflineReleaseError(
            f"CSV policy replay contract mismatch: {exc}"
        ) from exc

    raw_frame_count = len(frame_groups)
    if raw_frame_count != manifest["artifacts"]["raw_frame_count"]:
        raise OfflineReleaseError(
            "CSV/raw frame count does not match artifact manifest"
        )
    if len(raw_paths) != raw_frame_count * 2:
        raise OfflineReleaseError(
            "CSV raw references are not one depth/seg pair per frame"
        )
    render = manifest["render"]
    expected_shape = (render["height"], render["width"])
    robot_ids = set(OFFICIAL_ROBOT_GEOM_IDS)
    human_ids = set(OFFICIAL_HUMAN_GEOM_IDS)
    allowed_ids = robot_ids | human_ids
    saw_robot = False
    saw_human = False
    depth_count = 0
    segmentation_count = 0
    for relative in sorted(raw_paths):
        path = root / relative
        if relative.endswith("_depth_m.npy"):
            depth_count += 1
            array = _load_exact_npy(
                path, expected_shape=expected_shape, expected_dtype=np.dtype("float32")
            )
            if not np.isfinite(array).all() or np.any(array <= 0.0):
                raise OfflineReleaseError(
                    "raw depth contains non-finite/nonpositive values"
                )
        elif relative.endswith("_segmentation.npy"):
            segmentation_count += 1
            array = _load_exact_npy(
                path,
                expected_shape=(*expected_shape, 2),
                expected_dtype=np.dtype("int32"),
            )
            geom_ids = array[..., 0]
            object_types = array[..., 1]
            background = geom_ids == -1
            if not np.array_equal(background, object_types == -1):
                raise OfflineReleaseError(
                    "raw segmentation background channels disagree"
                )
            foreground = ~background
            if np.any(object_types[foreground] != OFFICIAL_GEOM_OBJECT_TYPE):
                raise OfflineReleaseError(
                    "raw segmentation object type is not mjOBJ_GEOM"
                )
            present = set(int(item) for item in np.unique(geom_ids[foreground]))
            if not present <= allowed_ids:
                raise OfflineReleaseError(
                    "raw segmentation contains non-official geom IDs"
                )
            saw_robot |= bool(present & robot_ids)
            saw_human |= bool(present & human_ids)
        else:
            raise OfflineReleaseError("CSV raw file suffix is not depth/segmentation")
    if depth_count != raw_frame_count or segmentation_count != raw_frame_count:
        raise OfflineReleaseError("raw depth/segmentation frame counts are incomplete")
    if not saw_robot or not saw_human:
        raise OfflineReleaseError("raw segmentation lacks official robot/human pixels")
    timeline_rows = {
        (scenario, frame_index): dict(group[0])
        for (scenario, frame_index), group in frame_groups.items()
    }
    return (
        rows,
        {
            "depth_frame_count": depth_count,
            "segmentation_frame_count": segmentation_count,
            "csv_render_frame_count": raw_frame_count,
        },
        raw_paths,
        outputs_tuple,
        truths,
        required,
        timeline_rows,
    )


def _validate_csv_schedule_against_input(
    contract: Mapping[str, Any],
    csv_frames: Mapping[tuple[str, int], Mapping[str, str]],
) -> None:
    declared_motion_bound = float(
        contract["policy_contract"]["patient_motion_bound_m_s"]
    )
    for stratum in contract["strata"]:
        scenario = stratum["scenario_id"]
        if stratum["status"] != "COMPLETE":
            if any(key[0] == scenario for key in csv_frames):
                raise OfflineReleaseError(
                    "failed input stratum unexpectedly has CSV frames"
                )
            continue
        event_rows: list[Mapping[str, str]] = []
        for event in stratum["events"]:
            frame_index = int(event["frame_index"])
            row = csv_frames.get((scenario, frame_index))
            if (
                row is None
                or row["event_name"] != event["event_name"]
                or int(row["observation_seed"]) != event["observation_seed"]
                or row["noise_contract_sha256"] != event["noise_contract_sha256"]
                or not math.isclose(
                    float(row["timestamp_s"]),
                    float(event["timestamp_s"]),
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["patient_motion_bound_m_s"]),
                    declared_motion_bound,
                    abs_tol=0.0,
                    rel_tol=0.0,
                )
            ):
                raise OfflineReleaseError(
                    "CSV event/noise/timestamp/constant motion bound differs from input"
                )
            event_rows.append(row)
        if (
            not event_rows
            or event_rows[0]["geometry_accepted"] != "true"
            or event_rows[-1]["geometry_accepted"] != "true"
        ):
            raise OfflineReleaseError(
                "COMPLETE sparse-core stratum requires accepted pre/post geometry"
            )
        base_truth = np.asarray(
            [
                float(event_rows[0]["ground_truth_x_m"]),
                float(event_rows[0]["ground_truth_y_m"]),
                float(event_rows[0]["ground_truth_z_m"]),
            ],
            dtype=np.float64,
        )
        direction = np.asarray(
            stratum["patient_motion_direction_world"], dtype=np.float64
        )
        for event, row in zip(stratum["events"], event_rows, strict=True):
            actual_truth = np.asarray(
                [
                    float(row["ground_truth_x_m"]),
                    float(row["ground_truth_y_m"]),
                    float(row["ground_truth_z_m"]),
                ],
                dtype=np.float64,
            )
            expected_truth = (
                base_truth
                + direction
                * core_experiment.PATIENT_TRANSLATION_SPEED_M_S
                * float(event["timestamp_s"])
            )
            if not np.allclose(actual_truth, expected_truth, atol=1e-12, rtol=0.0):
                raise OfflineReleaseError(
                    "CSV ground truth differs from frozen rigid patient motion"
                )


def _validate_metrics(
    root: Path,
    manifest: Mapping[str, Any],
    outputs_by_policy: Mapping[str, tuple[PolicyFrameOutput, ...]],
    truths: tuple[GroundTruthFrame, ...],
    required: tuple[RequiredStratum, ...],
) -> dict[str, Any]:
    path = root / manifest["artifacts"]["metrics_path"]
    raw = _regular_bytes(path, maximum=64 * 1024 * 1024)
    metrics = _load_json(path)
    if raw != _canonical_json_bytes(metrics):
        raise OfflineReleaseError("metrics JSON is not canonical")
    _keys(
        metrics,
        (
            "schema",
            "status",
            "scope",
            "production_activation",
            "isaac_equivalence",
            "identities",
            "denominators",
            "policies",
            "failed_strata",
            "completion_gates",
        ),
        context="metrics",
    )
    if (
        metrics["schema"] != "wam_offline_official_mesh_occlusion_metrics_v2"
        or metrics["status"] != STATUS
        or metrics["scope"] != SCOPE
        or metrics["production_activation"] is not False
        or metrics["isaac_equivalence"] is not False
    ):
        raise OfflineReleaseError("metrics status/scope schema is invalid")
    expected_identities = {
        key: _EXPECTED_IDENTITY[key]
        for key in (
            "runtime_manifest_sha256",
            "asset_manifest_sha256",
            "conversion_manifest_sha256",
        )
    }
    identities = _keys(
        metrics["identities"], expected_identities, context="metrics identities"
    )
    if dict(identities) != expected_identities:
        raise OfflineReleaseError("metrics identity cross-link mismatch")
    _keys(metrics["policies"], POLICY_NAMES, context="metrics policies")
    try:
        scored = score_occlusion_policy_metrics(
            outputs_by_policy, truths, required_strata=required,
        )
    except (OfflineSimulationError, TypeError, ValueError) as exc:
        raise OfflineReleaseError(
            f"metrics policy replay could not be scored: {exc}"
        ) from exc
    expected_denominators = scored[POLICY_NAMES[0]]["denominators"]
    if any(
        scored[name]["denominators"] != expected_denominators for name in POLICY_NAMES
    ):
        raise OfflineReleaseError("recomputed policy denominators disagree")
    expected_policies = {
        policy_name: {
            key: scored[policy_name][key]
            for key in ("pooled_frame", "equal_weight_per_stratum", "per_stratum",)
        }
        for policy_name in POLICY_NAMES
    }
    expected_failed = [
        record["scenario_id"]
        for record in manifest["matrix"]["records"]
        if record["status"] != "COMPLETE"
    ]
    if (
        metrics["denominators"] != expected_denominators
        or metrics["policies"] != expected_policies
        or metrics["failed_strata"] != expected_failed
    ):
        raise OfflineReleaseError(
            "metrics/denominators/per-stratum/equal-weight values do not exactly match recomputation"
        )
    _exact_false_gates(metrics["completion_gates"], context="metrics")
    return metrics


def _validate_video_timeline(
    root: Path,
    manifest: Mapping[str, Any],
    csv_frames: Mapping[tuple[str, int], Mapping[str, str]],
) -> dict[str, Any]:
    path = root / manifest["artifacts"]["video_timeline_path"]
    raw = _regular_bytes(path, maximum=64 * 1024 * 1024)
    timeline = _load_json(path)
    if raw != _canonical_json_bytes(timeline):
        raise OfflineReleaseError("video timeline JSON is not canonical")
    _keys(
        timeline,
        (
            "schema",
            "status",
            "video_path",
            "fps",
            "width",
            "height",
            "frame_count",
            "report_frame_video_frame_index",
            "overlay_contract",
            "frames",
        ),
        context="video timeline",
    )
    render = manifest["render"]
    timeline_frame_count = _integer(
        timeline["frame_count"], context="video timeline frame_count", minimum=1
    )
    if (
        timeline["schema"] != "wam_offline_official_mesh_occlusion_video_timeline_v1"
        or timeline["status"] != STATUS
        or timeline["video_path"] != manifest["artifacts"]["video_path"]
        or timeline["fps"] != render["fps"]
        or timeline["width"] != render["width"]
        or timeline["height"] != render["height"]
        or timeline_frame_count != render["frame_count"]
        or timeline_frame_count != len(csv_frames)
    ):
        raise OfflineReleaseError(
            "video timeline frame count/media identity is not bound to CSV/raw frames"
        )
    report_index = _integer(
        timeline["report_frame_video_frame_index"],
        context="video timeline report frame index",
    )
    if report_index >= timeline_frame_count:
        raise OfflineReleaseError("video timeline report frame index is out of range")
    overlay = _keys(
        timeline["overlay_contract"],
        ("region_xywh", "banner_text", "robot_commit", "policy_names"),
        context="video overlay contract",
    )
    if (
        overlay["banner_text"] != "OFFLINE METHOD SIM — NOT ISAAC / NOT PRODUCTION"
        or overlay["robot_commit"] != OPENARM_COMMIT
        or overlay["policy_names"] != list(POLICY_NAMES)
        or overlay["region_xywh"] != [0, 0, render["width"], 72]
        or any(type(value) is not int for value in overlay["region_xywh"])
    ):
        raise OfflineReleaseError(
            "video overlay banner/commit/policy identity mismatch"
        )
    frames = timeline["frames"]
    if not isinstance(frames, list) or len(frames) != len(csv_frames):
        raise OfflineReleaseError("video timeline is not a frame-level CSV bijection")
    seen_csv_keys: set[tuple[str, int]] = set()
    for expected_index, frame_value in enumerate(frames):
        frame = _keys(
            frame_value,
            (
                "video_frame_index",
                "scenario_id",
                "target_name",
                "wrist_index",
                "frame_index",
                "timestamp_s",
                "event_name",
                "observation_stream_sha256",
                "raw_depth_path",
                "raw_segmentation_path",
            ),
            context=f"video timeline frames[{expected_index}]",
        )
        video_frame_index = _integer(
            frame["video_frame_index"],
            context=f"video timeline frames[{expected_index}].video_frame_index",
        )
        frame_index = _integer(
            frame["frame_index"],
            context=f"video timeline frames[{expected_index}].frame_index",
        )
        if video_frame_index != expected_index:
            raise OfflineReleaseError("video timeline indices are not contiguous")
        if not isinstance(frame["scenario_id"], str):
            raise OfflineReleaseError("video timeline scenario ID must be a string")
        key = (frame["scenario_id"], frame_index)
        if key in seen_csv_keys or key not in csv_frames:
            raise OfflineReleaseError(
                "video timeline is not a bijection to CSV/raw keys"
            )
        row = csv_frames[key]
        if (
            frame["target_name"] != row["target_name"]
            or frame["wrist_index"] != int(row["wrist_index"])
            or frame["event_name"] != row["event_name"]
            or not math.isclose(
                _number(frame["timestamp_s"], context="video timeline timestamp"),
                float(row["timestamp_s"]),
                abs_tol=1e-12,
            )
            or frame["observation_stream_sha256"] != row["observation_stream_sha256"]
            or frame["raw_depth_path"] != row["raw_depth_path"]
            or frame["raw_segmentation_path"] != row["raw_segmentation_path"]
        ):
            raise OfflineReleaseError("video timeline CSV/raw cross-link mismatch")
        seen_csv_keys.add(key)
    if seen_csv_keys != set(csv_frames):
        raise OfflineReleaseError("video timeline omits a CSV/raw frame key")
    return timeline


def _probe_and_decode_video(
    path: Path,
    render: Mapping[str, Any],
    *,
    report_frame_index: int,
    overlay_region_xywh: tuple[int, int, int, int],
) -> tuple[dict[str, Any], np.ndarray]:
    try:
        video_size = path.lstat().st_size
    except OSError as exc:
        raise OfflineReleaseError("evidence video is missing") from exc
    if video_size <= 0 or video_size > MAX_VIDEO_BYTES:
        raise OfflineReleaseError("evidence video exceeds the strict media size cap")
    if render["frame_count"] > MAX_VIDEO_FRAME_COUNT:
        raise OfflineReleaseError("evidence video exceeds the strict frame-count cap")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise OfflineReleaseError("ffprobe is required to verify the evidence video")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v",
        "-count_frames",
        "-show_entries",
        "stream=codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, timeout=30)
        probe = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise OfflineReleaseError("ffprobe could not parse the evidence video") from exc
    streams = probe.get("streams") if isinstance(probe, dict) else None
    if not isinstance(streams, list) or len(streams) != 1:
        raise OfflineReleaseError("evidence video must have exactly one video stream")
    stream = streams[0]
    if stream.get("codec_name") != "h264":
        raise OfflineReleaseError("evidence video codec must be H.264")
    if stream.get("pix_fmt") != "yuv420p":
        raise OfflineReleaseError("evidence video pixel format must be yuv420p")
    if (
        stream.get("width") != render["width"]
        or stream.get("height") != render["height"]
    ):
        raise OfflineReleaseError(
            "evidence video resolution does not match the manifest"
        )
    try:
        fps = Fraction(stream["avg_frame_rate"])
        probed_count = int(stream["nb_read_frames"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise OfflineReleaseError(
            "evidence video FPS/frame count is unavailable"
        ) from exc
    if fps != Fraction(render["fps"], 1):
        raise OfflineReleaseError("evidence video FPS does not match the manifest")
    if probed_count != render["frame_count"]:
        raise OfflineReleaseError(
            "evidence video frame count does not match the manifest"
        )

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OfflineReleaseError("evidence video cannot be decoded")
    previous: np.ndarray | None = None
    selected: np.ndarray | None = None
    decoded_count = 0
    nonblack = False
    changing = False
    overlay_present = True
    x, y, width, height = overlay_region_xywh
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (render["height"], render["width"], 3):
                raise OfflineReleaseError(
                    "decoded evidence video frame has wrong shape"
                )
            if decoded_count == report_frame_index:
                selected = frame.copy()
            nonblack |= float(np.mean(frame)) > 1.0 and int(np.max(frame)) > 8
            region = frame[y : y + height, x : x + width]
            overlay_present &= (
                region.shape == (height, width, 3)
                and float(np.mean(region)) > 1.0
                and float(np.std(region)) > 8.0
            )
            if previous is not None:
                changing |= (
                    float(
                        np.mean(
                            np.abs(frame.astype(np.int16) - previous.astype(np.int16))
                        )
                    )
                    > 1.0
                )
            previous = frame
            decoded_count += 1
    finally:
        capture.release()
    if decoded_count != render["frame_count"]:
        raise OfflineReleaseError("decoded evidence video frame count is incomplete")
    if not nonblack:
        raise OfflineReleaseError("evidence video is black")
    if not changing:
        raise OfflineReleaseError("evidence video is static/not changing")
    if not overlay_present:
        raise OfflineReleaseError("video overlay region is absent/blank")
    if selected is None:
        raise OfflineReleaseError("declared report video frame was not decoded")
    return (
        {
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
            "width": render["width"],
            "height": render["height"],
            "fps": render["fps"],
            "frame_count": decoded_count,
            "nonblack": nonblack,
            "changing": changing,
            "timeline_bound": True,
            "report_frame_bound": True,
            "overlay_region_present": True,
        },
        selected,
    )


def _validate_completion(root: Path) -> None:
    completion = _load_json(root / "ARTIFACT_COMPLETE.json")
    _keys(
        completion,
        (
            "schema",
            "status",
            "artifact_write_complete",
            "production_activation",
            "linbox_deployment_complete",
            "physical_validation_complete",
        ),
        context="artifact completion marker",
    )
    if completion != {
        "schema": "wam_offline_official_mesh_occlusion_artifact_completion_v2",
        "status": STATUS,
        "artifact_write_complete": True,
        "production_activation": False,
        "linbox_deployment_complete": False,
        "physical_validation_complete": False,
    }:
        raise OfflineReleaseError("artifact completion marker is incomplete/unsafe")


def _payload_digest(root: Path, tree_files: set[str] | None = None) -> str:
    files = tree_files if tree_files is not None else _audit_tree(root)
    digest = hashlib.sha256()
    digest.update(b"wam-offline-official-mesh-release-payload-v1\x00")
    for relative in sorted(files):
        path_bytes = relative.encode("utf-8")
        file_digest = bytes.fromhex(_sha256_file(root / relative))
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(file_digest)
    return digest.hexdigest()


def verify_offline_release(release_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Strictly verify one completed offline-only evidence release."""

    root = Path(release_root)
    tree_files = _audit_tree(root)
    _verify_sums(root, tree_files)
    _verify_frozen_manifests(root)
    manifest = _validate_manifest(root)
    (
        input_contract,
        _input_contract_sha256,
        core_input_contract_sha256,
    ) = _validate_input_contract(root, manifest)
    (
        rows,
        raw_report,
        raw_paths,
        outputs_by_policy,
        truths,
        required,
        csv_frames,
    ) = _validate_csv_and_raw(
        root, manifest, input_contract, core_input_contract_sha256,
    )
    _validate_csv_schedule_against_input(input_contract, csv_frames)
    _validate_metrics(
        root, manifest, outputs_by_policy, truths, required,
    )
    _validate_completion(root)

    artifacts = manifest["artifacts"]
    timeline = _validate_video_timeline(root, manifest, csv_frames)
    video, decoded_report_frame = _probe_and_decode_video(
        root / artifacts["video_path"],
        manifest["render"],
        report_frame_index=timeline["report_frame_video_frame_index"],
        overlay_region_xywh=tuple(timeline["overlay_contract"]["region_xywh"]),
    )
    report_frame = cv2.imdecode(
        np.frombuffer(
            _regular_bytes(
                root / artifacts["report_frame_path"], maximum=MAX_REPORT_FRAME_BYTES,
            ),
            dtype=np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )
    if (
        report_frame is None
        or report_frame.shape
        != (manifest["render"]["height"], manifest["render"]["width"], 3)
        or float(np.mean(report_frame)) <= 1.0
    ):
        raise OfflineReleaseError(
            "report frame is missing, black, or has the wrong shape"
        )
    report_difference = np.abs(
        report_frame.astype(np.int16) - decoded_report_frame.astype(np.int16)
    )
    if (
        float(np.mean(report_difference)) > 5.0
        or float(np.quantile(report_difference, 0.95)) > 12.0
    ):
        raise OfflineReleaseError(
            "report frame does not match the declared decoded video frame"
        )
    readme = _regular_bytes(root / "README.md", maximum=4 * 1024 * 1024)
    if (
        b"OFFLINE METHOD SIM" not in readme
        or b"NOT ISAAC" not in readme
        or b"NOT PRODUCTION" not in readme
    ):
        raise OfflineReleaseError(
            "README lacks the permanent offline/not-production warning"
        )

    allowed_files = {
        "SHA256SUMS.txt",
        "README.md",
        "ARTIFACT_COMPLETE.json",
        "raw/runtime_manifest.json",
        "raw/asset_manifest.json",
        "raw/conversion_manifest.json",
        "raw/experiment_manifest.json",
        artifacts["input_contract_path"],
        artifacts["video_timeline_path"],
        artifacts["metrics_path"],
        artifacts["per_frame_csv_path"],
        artifacts["determinism_rerun_csv_path"],
        artifacts["video_path"],
        artifacts["report_frame_path"],
        *raw_paths,
    }
    if tree_files != allowed_files:
        raise OfflineReleaseError(
            "release schema has unexpected or missing artifact keys/files: "
            f"unexpected={sorted(tree_files - allowed_files)}, "
            f"missing={sorted(allowed_files - tree_files)}"
        )
    # Re-audit after media decoders ran, then bind the final exact tree.
    final_files = _audit_tree(root)
    if final_files != tree_files:
        raise OfflineReleaseError("release tree changed during verification")
    payload_sha = _payload_digest(root, final_files)
    return {
        "status": STATUS,
        "production_activation": False,
        "isaac_equivalence": False,
        "identities": {
            "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
            "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
            "conversion_manifest_sha256": CONVERSION_MANIFEST_SHA256,
            "robot_commit": OPENARM_COMMIT,
            "robot_urdf_sha256": OPENARM_URDF_SHA256,
            "human_usd_sha256": NVIDIA_HUMAN_USD_SHA256,
        },
        "video": video,
        "raw_products": raw_report,
        "completion_gates": dict(FALSE_COMPLETION_GATES),
        "sha256_coverage_complete": True,
        "same_input_rerun_match": True,
        "policy_replay_match": True,
        "metrics_recomputed_match": True,
        "input_contract_bound": True,
        "release_payload_sha256": payload_sha,
    }


def release_payload_sha256(release_root: str | os.PathLike[str]) -> str:
    """Return the verified, path-bound digest of a completed release."""

    return verify_offline_release(release_root)["release_payload_sha256"]


def _fsync_tree(root: Path) -> None:
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
        directory_flags |= os.O_NOFOLLOW
    for directory, _, filenames in os.walk(root, topdown=False, followlinks=False):
        directory_path = Path(directory)
        for filename in filenames:
            descriptor = os.open(directory_path / filename, file_flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(directory_path, directory_flags)
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, getattr(errno, "ENOTSUP", errno.EINVAL)):
                raise
        finally:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, getattr(errno, "ENOTSUP", errno.EINVAL)):
                raise
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"destination exists; refusing to replace: {destination}")
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        # RENAME_EXCL from <stdio.h>: atomically fail if destination exists.
        result = function(source_bytes, destination_bytes, ctypes.c_uint(0x00000004))
    elif hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, source_bytes, -100, destination_bytes, ctypes.c_uint(1))
    else:
        raise OfflineReleaseError("atomic no-replace directory rename is unavailable")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                f"destination exists; refusing to replace: {destination}"
            )
        raise OSError(error_number, os.strerror(error_number), str(destination))


def publish_staged_release(
    staging_root: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> dict[str, Any]:
    """Verify and atomically rename a staged release without replacing anything."""

    staging = Path(staging_root)
    target = Path(destination)
    if staging.parent != target.parent:
        raise OfflineReleaseError(
            "staging and destination must share a parent for atomic publication"
        )
    if not target.name or target.name in (".", ".."):
        raise OfflineReleaseError("publication destination is invalid")
    initial_report = verify_offline_release(staging)
    snapshot_digest = initial_report["release_payload_sha256"]
    if os.path.lexists(target):
        raise FileExistsError(f"destination exists; refusing to replace: {target}")
    try:
        staging_identity = staging.lstat()
        staging_device = staging_identity.st_dev
        parent_device = staging.parent.lstat().st_dev
    except OSError as exc:
        raise OfflineReleaseError(
            "staging/parent disappeared before publication"
        ) from exc
    if staging_device != parent_device:
        raise OfflineReleaseError("staging is not on the destination filesystem")
    _fsync_tree(staging)
    # fsync traverses attacker-controlled pathnames.  Reverify the complete
    # release and require the exact original payload immediately before rename.
    pre_rename_report = verify_offline_release(staging)
    current_identity = staging.lstat()
    if pre_rename_report["release_payload_sha256"] != snapshot_digest or (
        current_identity.st_dev,
        current_identity.st_ino,
    ) != (staging_identity.st_dev, staging_identity.st_ino):
        raise OfflineReleaseError(
            "staged release changed after fsync; immutable snapshot publication refused"
        )
    _rename_no_replace(staging, target)
    try:
        _fsync_directory(target.parent)
        published_identity = target.lstat()
        if (published_identity.st_dev, published_identity.st_ino) != (
            staging_identity.st_dev,
            staging_identity.st_ino,
        ):
            raise OfflineReleaseError(
                "published directory identity changed during rename"
            )
        published_report = verify_offline_release(target)
        if published_report["release_payload_sha256"] != snapshot_digest:
            raise OfflineReleaseError(
                "published release changed after rename; immutable snapshot mismatch"
            )
        return published_report
    except Exception as publish_error:
        rollback_error: Exception | None = None
        try:
            if os.path.lexists(target) and not os.path.lexists(staging):
                current = target.lstat()
                if (current.st_dev, current.st_ino) != (
                    staging_identity.st_dev,
                    staging_identity.st_ino,
                ):
                    raise OfflineReleaseError(
                        "refusing rollback of an unexpected destination identity"
                    )
                _rename_no_replace(target, staging)
                _fsync_directory(staging.parent)
        except Exception as exc:
            rollback_error = exc
        if rollback_error is not None:
            raise OfflineReleaseError(
                "post-rename snapshot verification failed and rollback failed"
            ) from rollback_error
        if isinstance(publish_error, OfflineReleaseError):
            raise publish_error
        raise OfflineReleaseError(
            "post-rename snapshot verification failed; publication rolled back"
        ) from publish_error


__all__ = (
    "OfflineReleaseError",
    "publish_staged_release",
    "release_payload_sha256",
    "verify_offline_release",
)
