#!/usr/bin/env python3
"""CAS-guarded OpenArm telemetry patch application.

This script is intentionally limited to a private candidate tree, never the live
Lingbox source. It requires an independently pinned, canonical full-tree
predecessor state, applies the versioned patch in an isolated temporary tree,
then revalidates every input before and after candidate writes. A failed write is
rolled back and an incomplete rollback emits an out-of-tree recovery artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Dict


HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "lingbox_openarm_baseline.json"
PATCH_PATH = HERE / "openarm_unilateral_control.telemetry.patch"

EXPECTED_BASELINE_SCHEMA = "wam_openarm_telemetry_patch_baseline_v5"
TREE_MANIFEST_SCHEMA = "wam_openarm_teleop_tree_state_v1"
TREE_MANIFEST_PURPOSE = "private_candidate_staging"
TREE_MANIFEST_COVERAGE = "full_tracked_tree"
RECOVERY_SCHEMA = "wam_openarm_telemetry_recovery_v1"
PATCH_INPUTS = {
    "control/openarm_unilateral_control.cpp": "unilateral_source_sha256",
    "src/controller/control.cpp": "control_source_sha256",
    "src/controller/control.hpp": "control_header_sha256",
}
SUPPORT_FILES = {
    "CMakeLists.txt": "cmake_sha256",
    "src/robot_state.hpp": "robot_state_header_sha256",
    "src/joint_state_converter.hpp": "joint_converter_header_sha256",
    "config/follower.yaml": "follower_config_sha256",
    "src/wam_openarm_telemetry/openarm_telemetry_publisher.cpp": (
        "telemetry_publisher_source_sha256"
    ),
    "src/wam_openarm_telemetry/openarm_telemetry_publisher.hpp": (
        "telemetry_publisher_header_sha256"
    ),
}
OUTPUT_HASH_FIELDS = {
    "control/openarm_unilateral_control.cpp": "unilateral_output_sha256",
    "src/controller/control.cpp": "control_source_output_sha256",
    "src/controller/control.hpp": "control_header_output_sha256",
}
EXTERNAL_FILES = {
    "openarm_mit_header_path": "openarm_mit_header_sha256",
    "openarm_collection_header_path": "openarm_collection_header_sha256",
    "openarm_static_library_path": "openarm_static_library_sha256",
}


class PatchApplicationError(RuntimeError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_regular_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PatchApplicationError(f"required regular file is missing: {path}")
    return path.read_bytes()


def _require_hash(path: Path, expected: str) -> bytes:
    payload = _read_regular_file(path)
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise PatchApplicationError(
            f"SHA-256 mismatch for {path}: expected {expected}, received {actual}"
        )
    return payload


def _load_baseline(raw: bytes | None = None) -> dict:
    try:
        if raw is None:
            raw = _read_regular_file(BASELINE_PATH)
        baseline = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PatchApplicationError("cannot read the versioned baseline") from exc
    if baseline.get("schema") != EXPECTED_BASELINE_SCHEMA:
        raise PatchApplicationError("unsupported OpenArm telemetry baseline schema")
    if (
        baseline.get("audited_side") != "left"
        or baseline.get("leader_can_interface") != "can1"
        or baseline.get("follower_can_interface") != "can3"
        or baseline.get("deployment_scope") != "stethoscope_left_only"
        or baseline.get("gripper_decouple_env") != "GRIPPER_DECOUPLE"
        or baseline.get("gripper_hold_position_env") != "GRIPPER_HOLD_POS"
        or baseline.get("gripper_hold_position_rad") != 0.0
        or baseline.get("gripper_hold_position_status") != "HIL_PENDING"
        or baseline.get("gripper_decouple_opt_in_value") != "1"
        or baseline.get("gripper_lead_env") != "GRIP_LEAD"
        or baseline.get("gripper_lead_default_rad") != 0.08
        or baseline.get("gripper_lead_min_rad") != 0.02
        or baseline.get("gripper_lead_max_rad") != 0.2
        or baseline.get("gripper_ramp_rate_rad_s") != 0.5
        or baseline.get("gripper_homing_step_s") != 0.01
        or baseline.get("predecessor_manifest_schema") != TREE_MANIFEST_SCHEMA
        or baseline.get("predecessor_manifest_state") != "init_reset_applied"
        or baseline.get("result_manifest_state") != "gripper_patch_applied"
        or baseline.get("predecessor_full_tree_manifest_status")
        not in {"COMPLETE", "REQUIRED_NOT_PINNED"}
    ):
        raise PatchApplicationError(
            "baseline is not the HIL-pending stethoscope left can1-to-can3 pair"
        )
    predecessor_status = baseline["predecessor_full_tree_manifest_status"]
    predecessor_sha256 = baseline.get("predecessor_full_tree_manifest_sha256")
    if (
        predecessor_status == "COMPLETE" and not _is_sha256(predecessor_sha256)
    ) or (
        predecessor_status == "REQUIRED_NOT_PINNED"
        and predecessor_sha256 is not None
    ):
        raise PatchApplicationError("invalid predecessor full-tree manifest pin")
    for hash_field in (*PATCH_INPUTS.values(), *SUPPORT_FILES.values(),
                       *OUTPUT_HASH_FIELDS.values(), *EXTERNAL_FILES.values(),
                       "patch_sha256"):
        if not _is_sha256(baseline.get(hash_field)):
            raise PatchApplicationError(f"invalid SHA-256 baseline field: {hash_field}")
    return baseline


def _manifest_relative_path(raw: object) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise PatchApplicationError("invalid full-tree manifest path")
    value = PurePosixPath(raw)
    if (
        value.is_absolute()
        or raw != value.as_posix()
        or any(part in {"", ".", ".."} for part in value.parts)
        or value.parts[0] in {".git", "build"}
    ):
        raise PatchApplicationError(f"invalid full-tree manifest path: {raw}")
    return raw


def _validate_tree_manifest_shape(
    value: object,
    *,
    expected_state: str,
    expected_build_authorized: bool,
) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "allowed_untracked_prefixes",
        "build_authorized",
        "coverage",
        "files",
        "purpose",
        "schema",
        "sealed",
        "state",
        "transition",
    }:
        raise PatchApplicationError("invalid full-tree manifest envelope")
    if (
        value.get("schema") != TREE_MANIFEST_SCHEMA
        or value.get("state") != expected_state
        or value.get("coverage") != TREE_MANIFEST_COVERAGE
        or value.get("purpose") != TREE_MANIFEST_PURPOSE
        or value.get("sealed") is not True
        or value.get("build_authorized") is not expected_build_authorized
        or value.get("allowed_untracked_prefixes") != []
        or not isinstance(value.get("transition"), dict)
    ):
        raise PatchApplicationError("unsealed or wrong-state full-tree manifest")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise PatchApplicationError("full-tree manifest has no files")
    prior_path = ""
    normalized_rows = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "kind",
            "mode",
            "path",
            "sha256",
        }:
            raise PatchApplicationError("invalid full-tree manifest file row")
        relative = _manifest_relative_path(row.get("path"))
        kind = row.get("kind")
        mode = row.get("mode")
        if relative <= prior_path:
            raise PatchApplicationError(
                "full-tree manifest paths must be unique and sorted"
            )
        if kind not in {"regular", "symlink"}:
            raise PatchApplicationError(f"invalid manifest kind for {relative}")
        if not isinstance(mode, str) or (
            kind == "regular"
            and (len(mode) != 4 or any(character not in "01234567" for character in mode))
        ) or (kind == "symlink" and mode != "120000"):
            raise PatchApplicationError(f"invalid manifest mode for {relative}")
        if not _is_sha256(row.get("sha256")):
            raise PatchApplicationError(f"invalid manifest SHA-256 for {relative}")
        normalized_rows.append(dict(row))
        prior_path = relative
    normalized = dict(value)
    normalized["files"] = normalized_rows
    return normalized


def _tree_payload(path: Path, kind: str) -> tuple[bytes, str]:
    if kind == "regular":
        payload = _read_regular_file(path)
        mode = f"{path.stat().st_mode & 0o777:04o}"
        return payload, mode
    if not path.is_symlink():
        raise PatchApplicationError(f"required symlink is missing: {path}")
    return os.readlink(path).encode("utf-8"), "120000"


def _candidate_tree_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for current_raw, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_raw)
        if current == root and ".git" in directory_names:
            directory_names.remove(".git")
        for name in list(directory_names):
            candidate = current / name
            if candidate.is_symlink():
                paths.add(candidate.relative_to(root).as_posix())
                directory_names.remove(name)
        for name in file_names:
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            if relative != ".git":
                paths.add(relative)
    return paths


def _verify_tree_against_manifest(root: Path, manifest: dict) -> None:
    rows = {row["path"]: row for row in manifest["files"]}
    actual_paths = _candidate_tree_paths(root)
    if actual_paths != set(rows):
        missing = sorted(set(rows) - actual_paths)
        extra = sorted(actual_paths - set(rows))
        raise PatchApplicationError(
            "full-tree manifest path set mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )
    for relative, row in rows.items():
        payload, mode = _tree_payload(root / relative, row["kind"])
        if _sha256_bytes(payload) != row["sha256"] or mode != row["mode"]:
            raise PatchApplicationError(
                f"full-tree manifest content mismatch for {relative}"
            )


def _load_predecessor_manifest(
    root: Path,
    path: Path | None,
    baseline: dict,
) -> tuple[dict, str]:
    if (
        baseline["predecessor_full_tree_manifest_status"] != "COMPLETE"
        or not _is_sha256(baseline.get("predecessor_full_tree_manifest_sha256"))
    ):
        raise PatchApplicationError(
            "independent predecessor full-tree manifest is not pinned"
        )
    if path is None:
        raise PatchApplicationError("predecessor full-tree manifest is required")
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise PatchApplicationError("predecessor full-tree manifest must be regular")
    manifest_path = expanded.resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise PatchApplicationError(
            "predecessor full-tree manifest must be outside candidate root"
        )
    raw = _require_hash(
        manifest_path,
        baseline["predecessor_full_tree_manifest_sha256"],
    )
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PatchApplicationError("cannot decode predecessor full-tree manifest") from exc
    if raw != _canonical_json_bytes(decoded):
        raise PatchApplicationError("predecessor full-tree manifest is not canonical")
    manifest = _validate_tree_manifest_shape(
        decoded,
        expected_state=baseline["predecessor_manifest_state"],
        expected_build_authorized=False,
    )
    _verify_tree_against_manifest(root, manifest)
    return manifest, _sha256_bytes(raw)


def _successor_manifest(
    predecessor: dict,
    predecessor_sha256: str,
    staged: Dict[str, bytes],
    baseline: dict,
) -> dict:
    rows = {row["path"]: dict(row) for row in predecessor["files"]}
    if not set(staged).issubset(rows):
        raise PatchApplicationError(
            "patched paths are absent from predecessor full-tree manifest"
        )
    for relative, payload in staged.items():
        rows[relative]["sha256"] = _sha256_bytes(payload)
    return {
        "allowed_untracked_prefixes": [],
        "build_authorized": True,
        "coverage": TREE_MANIFEST_COVERAGE,
        "files": [rows[path] for path in sorted(rows)],
        "purpose": TREE_MANIFEST_PURPOSE,
        "schema": TREE_MANIFEST_SCHEMA,
        "sealed": True,
        "state": baseline["result_manifest_state"],
        "transition": {
            "kind": "openarm_gripper_patch_v1",
            "patch_sha256": baseline["patch_sha256"],
            "predecessor_manifest_sha256": predecessor_sha256,
        },
    }


def _atomic_replace(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.wam-telemetry-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _recovery_artifact_path(root: Path, requested: Path | None) -> Path:
    path = (
        requested.expanduser()
        if requested is not None
        else root.parent / f"{root.name}.wam-telemetry-recovery.json"
    )
    if path.is_symlink():
        raise PatchApplicationError("recovery artifact path must not be a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved
    raise PatchApplicationError("recovery artifact must be outside candidate root")


def _write_recovery_artifact(
    path: Path,
    root: Path,
    originals: Dict[Path, tuple[bool, bytes, int]],
    errors: list[str],
) -> None:
    rows = []
    for target in sorted(originals, key=lambda value: value.as_posix()):
        existed, payload, mode = originals[target]
        current_sha256 = None
        current_mode = None
        if target.is_file() and not target.is_symlink():
            current_sha256 = _sha256_bytes(target.read_bytes())
            current_mode = f"{target.stat().st_mode & 0o777:04o}"
        rows.append(
            {
                "current_mode": current_mode,
                "current_sha256": current_sha256,
                "expected_existed": existed,
                "expected_mode": f"{mode:04o}" if existed else None,
                "expected_sha256": _sha256_bytes(payload) if existed else None,
                "path": target.relative_to(root).as_posix(),
            }
        )
    value = {
        "errors": errors,
        "schema": RECOVERY_SCHEMA,
        "status": "ROLLBACK_INCOMPLETE",
        "targets": rows,
        "teleop_root": str(root),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.wam-recovery-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rollback_outputs(
    root: Path,
    originals: Dict[Path, tuple[bool, bytes, int]],
    restore_targets: list[Path],
    recovery_artifact: Path,
) -> None:
    errors: list[str] = []
    for target in restore_targets:
        existed, payload, mode = originals[target]
        try:
            if existed:
                _atomic_replace(target, payload, mode)
            else:
                target.unlink(missing_ok=True)
        except Exception as exc:  # Recovery must inspect every target.
            errors.append(f"restore {target}: {exc}")
    for target, (existed, payload, mode) in originals.items():
        try:
            if existed:
                current = _read_regular_file(target)
                if current != payload or (target.stat().st_mode & 0o777) != mode:
                    errors.append(f"verify {target}: restored bytes or mode differ")
            elif target.exists() or target.is_symlink():
                errors.append(f"verify {target}: unexpected output remains")
        except Exception as exc:
            errors.append(f"verify {target}: {exc}")
    if errors:
        _write_recovery_artifact(recovery_artifact, root, originals, errors)
        raise PatchApplicationError(
            f"rollback incomplete; recovery artifact: {recovery_artifact}"
        )


def _apply_in_staging(
    base_payloads: Dict[str, bytes],
    baseline: dict | None = None,
) -> Dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="wam-openarm-telemetry-stage-") as raw:
        stage = Path(raw)
        for relative, payload in base_payloads.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)

        result = subprocess.run(
            ["patch", "--batch", "--forward", "-F0", "-p1", "-i", str(PATCH_PATH)],
            cwd=stage,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise PatchApplicationError(
                "versioned OpenArm patch did not apply exactly: "
                + (result.stderr.strip() or result.stdout.strip())
            )

        staged_paths = {
            path.relative_to(stage).as_posix()
            for path in stage.rglob("*") if path.is_file()
        }
        if staged_paths != set(PATCH_INPUTS):
            raise PatchApplicationError(
                "versioned patch changed files outside the gripper allowlist"
            )
        patched_source = (
            stage / "control/openarm_unilateral_control.cpp"
        ).read_bytes()
        patched_control_source = (
            stage / "src/controller/control.cpp"
        ).read_bytes()
        patched_control_header = (
            stage / "src/controller/control.hpp"
        ).read_bytes()
        source_text = patched_source.decode("utf-8")
        control_text = patched_control_source.decode("utf-8")
        control_header_text = patched_control_header.decode("utf-8")
        if (
            "follower_command_q_rad" in source_text
            or "leader_target_q_rad" in source_text
            or 'arm_side == "left_arm"' not in source_text
            or 'leader_can_interface == "can1"' not in source_text
            or 'follower_can_interface == "can3"' not in source_text
            or "FollowerReferenceEpoch" not in source_text
            or "follower_arm_mit" not in control_text
            or control_text.count("get_feedback_liveness_snapshot") != 2
            or "follower_arm_state_rx_counters" not in control_text
            or "follower_gripper_state_rx_counter" not in control_text
            or "try_enqueue" not in control_text
            or "mit_control_all(arm_cmds)" not in control_text
            or 'std::getenv("GRIPPER_DECOUPLE")' not in source_text
            or 'parse_bounded_env("GRIPPER_HOLD_POS"' not in source_text
            or 'parse_bounded_env("GRIP_LEAD"' not in source_text
            or "stethoscope_left_pair &&" not in source_text
            or "ConfigureFollowerGripperHold" not in source_text
            or source_text.index("ConfigureFollowerGripperHold")
            > source_text.index("&Control::AdjustPosition")
            or source_text.index('std::getenv("GRIPPER_DECOUPLE")')
            > source_text.index(
                "openarm_init::OpenArmInitializer::initialize_openarm"
            )
            or control_text.count("follower_gripper_hold_enabled_") < 4
            or "GRIP_RAMP_RATE * Ts_" not in control_text
            or "GRIP_RAMP_RATE * 0.010" not in control_text
            or "home_pose[NMOTORS - 1]" not in control_text
            or "ConfigureFollowerGripperHold" not in control_header_text
            or "follower_gripper_max_lead_rad_" not in control_header_text
            or "dynamics_f_->GetGravity" not in control_text
            or "GetJointLimits" not in control_text
            or "REF_GOVERNOR" not in control_text
        ):
            raise PatchApplicationError(
                "patched source failed the left-pair exact-action audit"
            )

        outputs = {
            "control/openarm_unilateral_control.cpp": patched_source,
            "src/controller/control.cpp": patched_control_source,
            "src/controller/control.hpp": patched_control_header,
        }
        if baseline is None:
            baseline = _load_baseline()
        for relative, hash_field in OUTPUT_HASH_FIELDS.items():
            if _sha256_bytes(outputs[relative]) != baseline[hash_field]:
                raise PatchApplicationError(
                    f"patched output SHA-256 mismatch for {relative}"
                )
        return outputs


def _verify_static_inputs(
    root: Path,
    baseline: dict,
    baseline_sha256: str,
    predecessor_manifest: Path,
    predecessor_sha256: str,
) -> None:
    _require_hash(BASELINE_PATH, baseline_sha256)
    _require_hash(PATCH_PATH, baseline["patch_sha256"])
    _require_hash(predecessor_manifest, predecessor_sha256)
    for relative, hash_field in SUPPORT_FILES.items():
        _require_hash(root / relative, baseline[hash_field])
    for path_field, hash_field in EXTERNAL_FILES.items():
        external_path = Path(baseline[path_field])
        if not external_path.is_absolute():
            raise PatchApplicationError(
                f"external baseline path must be absolute: {external_path}"
            )
        _require_hash(external_path, baseline[hash_field])


def _verify_post_write(
    root: Path,
    baseline: dict,
    baseline_sha256: str,
    predecessor_manifest: Path,
    predecessor_sha256: str,
    successor: dict,
) -> None:
    try:
        _verify_static_inputs(
            root,
            baseline,
            baseline_sha256,
            predecessor_manifest,
            predecessor_sha256,
        )
        for relative, hash_field in OUTPUT_HASH_FIELDS.items():
            _require_hash(root / relative, baseline[hash_field])
        _verify_tree_against_manifest(root, successor)
    except (OSError, PatchApplicationError) as exc:
        raise PatchApplicationError(
            f"post-write verification failed: {exc}"
        ) from exc


def apply_patch(
    teleop_root: Path,
    *,
    check_only: bool,
    predecessor_manifest: Path | None = None,
    recovery_artifact: Path | None = None,
) -> dict:
    expanded_root = teleop_root.expanduser()
    if expanded_root.is_symlink():
        raise PatchApplicationError("teleop root must not be a symlink")
    root = expanded_root.resolve()
    if not root.is_dir():
        raise PatchApplicationError(f"teleop root is not a directory: {root}")
    baseline_raw = _read_regular_file(BASELINE_PATH)
    baseline_sha256 = _sha256_bytes(baseline_raw)
    baseline = _load_baseline(baseline_raw)
    predecessor, predecessor_sha256 = _load_predecessor_manifest(
        root,
        predecessor_manifest,
        baseline,
    )
    if predecessor_manifest is None:  # Narrow the type after fail-closed loading.
        raise PatchApplicationError("predecessor full-tree manifest is required")
    predecessor_path = predecessor_manifest.expanduser().resolve()
    _verify_static_inputs(
        root,
        baseline,
        baseline_sha256,
        predecessor_path,
        predecessor_sha256,
    )

    base_payloads = {
        relative: _require_hash(root / relative, baseline[hash_field])
        for relative, hash_field in PATCH_INPUTS.items()
    }
    staged = _apply_in_staging(base_payloads, baseline)
    successor = _successor_manifest(
        predecessor,
        predecessor_sha256,
        staged,
        baseline,
    )
    successor = _validate_tree_manifest_shape(
        successor,
        expected_state=baseline["result_manifest_state"],
        expected_build_authorized=True,
    )
    successor_sha256 = _sha256_bytes(_canonical_json_bytes(successor))
    result = {
        "schema": "wam_openarm_telemetry_patch_result_v5",
        "mode": "check" if check_only else "applied",
        "teleop_root": str(root),
        "baseline_sha256": baseline_sha256,
        "patch_sha256": baseline["patch_sha256"],
        "predecessor_full_tree_manifest_sha256": predecessor_sha256,
        "result_full_tree_manifest": successor,
        "result_full_tree_manifest_sha256": successor_sha256,
        "side": "left",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "base_inputs": {
            relative: baseline[hash_field]
            for relative, hash_field in sorted(PATCH_INPUTS.items())
        },
        "support_inputs": {
            relative: baseline[hash_field]
            for relative, hash_field in sorted(SUPPORT_FILES.items())
        },
        "external_inputs": {
            baseline[path_field]: baseline[hash_field]
            for path_field, hash_field in sorted(EXTERNAL_FILES.items())
        },
        "outputs": {
            relative: _sha256_bytes(payload)
            for relative, payload in sorted(staged.items())
        },
    }
    if check_only:
        _verify_static_inputs(
            root,
            baseline,
            baseline_sha256,
            predecessor_path,
            predecessor_sha256,
        )
        _verify_tree_against_manifest(root, predecessor)
        return result

    recovery_path = _recovery_artifact_path(root, recovery_artifact)
    protected_paths = {
        BASELINE_PATH.resolve(),
        PATCH_PATH.resolve(),
        predecessor_path,
    }
    protected_paths.update(
        Path(baseline[path_field]).resolve() for path_field in EXTERNAL_FILES
    )
    if recovery_path in protected_paths:
        raise PatchApplicationError(
            "recovery artifact collides with protected input"
        )

    # Final predecessor-state comparison before the first candidate write.
    _verify_static_inputs(
        root,
        baseline,
        baseline_sha256,
        predecessor_path,
        predecessor_sha256,
    )
    _verify_tree_against_manifest(root, predecessor)

    originals: Dict[Path, tuple[bool, bytes, int]] = {}
    for relative in staged:
        target = root / relative
        payload = _require_hash(target, baseline[PATCH_INPUTS[relative]])
        originals[target] = (
            True,
            payload,
            target.stat().st_mode & 0o777,
        )

    committed_targets: list[Path] = []
    try:
        for relative, payload in staged.items():
            target = root / relative
            _require_hash(target, baseline[PATCH_INPUTS[relative]])
            _atomic_replace(target, payload, originals[target][2])
            committed_targets.append(target)
        _verify_post_write(
            root,
            baseline,
            baseline_sha256,
            predecessor_path,
            predecessor_sha256,
            successor,
        )
    except Exception:
        _rollback_outputs(root, originals, committed_targets, recovery_path)
        raise

    return result


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teleop-root", type=Path, required=True)
    parser.add_argument(
        "--predecessor-manifest",
        type=Path,
        required=True,
        help="independently pinned canonical full-tree predecessor manifest",
    )
    parser.add_argument(
        "--recovery-artifact",
        type=Path,
        help="out-of-tree path for an incomplete-rollback recovery record",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate and stage the exact patch without writing the target tree",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = apply_patch(
            args.teleop_root,
            check_only=args.check,
            predecessor_manifest=args.predecessor_manifest,
            recovery_artifact=args.recovery_artifact,
        )
    except (OSError, PatchApplicationError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
