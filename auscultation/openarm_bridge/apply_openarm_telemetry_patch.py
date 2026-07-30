#!/usr/bin/env python3
"""CAS-guarded OpenArm telemetry patch application.

This script is intentionally not run against the live Lingbox tree by the Mac
build.  It first proves every audited base hash, applies the versioned patch in
an isolated temporary tree, then performs a compare-and-swap write.  A mismatch
causes zero target writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict


HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "lingbox_openarm_baseline.json"
PATCH_PATH = HERE / "openarm_unilateral_control.telemetry.patch"
PUBLISHER_HEADER = HERE / "openarm_telemetry_publisher.hpp"
PUBLISHER_SOURCE = HERE / "openarm_telemetry_publisher.cpp"

EXPECTED_BASELINE_SCHEMA = "wam_openarm_telemetry_patch_baseline_v2"
PATCH_INPUTS = {
    "control/openarm_unilateral_control.cpp": "unilateral_source_sha256",
    "CMakeLists.txt": "cmake_sha256",
    "src/controller/control.cpp": "control_source_sha256",
    "src/controller/control.hpp": "control_header_sha256",
}
SUPPORT_FILES = {
    "src/robot_state.hpp": "robot_state_header_sha256",
    "src/joint_state_converter.hpp": "joint_converter_header_sha256",
    "config/follower.yaml": "follower_config_sha256",
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


def _load_baseline() -> dict:
    try:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatchApplicationError("cannot read the versioned baseline") from exc
    if baseline.get("schema") != EXPECTED_BASELINE_SCHEMA:
        raise PatchApplicationError("unsupported OpenArm telemetry baseline schema")
    if (
        baseline.get("audited_side") != "left"
        or baseline.get("leader_can_interface") != "can1"
        or baseline.get("follower_can_interface") != "can3"
    ):
        raise PatchApplicationError("baseline is not the audited left can1-to-can3 pair")
    return baseline


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


def _apply_in_staging(
    base_payloads: Dict[str, bytes],
) -> Dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="wam-openarm-telemetry-stage-") as raw:
        stage = Path(raw)
        control = stage / "control"
        controller = stage / "src" / "controller"
        bridge = stage / "src" / "wam_openarm_telemetry"
        control.mkdir(parents=True)
        controller.mkdir(parents=True)
        bridge.mkdir(parents=True)
        for relative, payload in base_payloads.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        shutil.copyfile(PUBLISHER_HEADER, bridge / PUBLISHER_HEADER.name)
        shutil.copyfile(PUBLISHER_SOURCE, bridge / PUBLISHER_SOURCE.name)

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

        patched_source = (control / "openarm_unilateral_control.cpp").read_bytes()
        patched_cmake = (stage / "CMakeLists.txt").read_bytes()
        patched_control_source = (controller / "control.cpp").read_bytes()
        patched_control_header = (controller / "control.hpp").read_bytes()
        source_text = patched_source.decode("utf-8")
        control_text = patched_control_source.decode("utf-8")
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
        ):
            raise PatchApplicationError(
                "patched source failed the left-pair exact-action audit"
            )

        return {
            "control/openarm_unilateral_control.cpp": patched_source,
            "CMakeLists.txt": patched_cmake,
            "src/controller/control.cpp": patched_control_source,
            "src/controller/control.hpp": patched_control_header,
            "src/wam_openarm_telemetry/openarm_telemetry_publisher.hpp": (
                bridge / PUBLISHER_HEADER.name
            ).read_bytes(),
            "src/wam_openarm_telemetry/openarm_telemetry_publisher.cpp": (
                bridge / PUBLISHER_SOURCE.name
            ).read_bytes(),
        }


def apply_patch(teleop_root: Path, *, check_only: bool) -> dict:
    root = teleop_root.expanduser().resolve()
    if not root.is_dir():
        raise PatchApplicationError(f"teleop root is not a directory: {root}")
    baseline = _load_baseline()

    base_payloads = {
        relative: _require_hash(root / relative, baseline[hash_field])
        for relative, hash_field in PATCH_INPUTS.items()
    }
    for relative, hash_field in SUPPORT_FILES.items():
        _require_hash(root / relative, baseline[hash_field])
    for path_field, hash_field in EXTERNAL_FILES.items():
        external_path = Path(baseline[path_field])
        if not external_path.is_absolute():
            raise PatchApplicationError(
                f"external baseline path must be absolute: {external_path}"
            )
        _require_hash(external_path, baseline[hash_field])

    staged = _apply_in_staging(base_payloads)
    result = {
        "schema": "wam_openarm_telemetry_patch_result_v2",
        "mode": "check" if check_only else "applied",
        "teleop_root": str(root),
        "side": "left",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "base_inputs": {
            relative: baseline[hash_field]
            for relative, hash_field in sorted(PATCH_INPUTS.items())
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
        return result

    # Compare again immediately before mutation.  This prevents a change between
    # staging and commit from being overwritten.
    for relative, hash_field in PATCH_INPUTS.items():
        _require_hash(root / relative, baseline[hash_field])

    originals: Dict[Path, tuple[bool, bytes, int]] = {}
    for relative in staged:
        target = root / relative
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise PatchApplicationError(
                    f"refusing non-regular output target: {target}"
                )
            originals[target] = (
                True,
                target.read_bytes(),
                target.stat().st_mode & 0o777,
            )
        else:
            originals[target] = (False, b"", 0o644)

    try:
        for relative, payload in staged.items():
            target = root / relative
            prior_mode = originals[target][2]
            _atomic_replace(target, payload, prior_mode)
    except Exception:
        for target, (existed, payload, mode) in originals.items():
            if existed:
                _atomic_replace(target, payload, mode)
            else:
                target.unlink(missing_ok=True)
        raise

    return result


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teleop-root", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate and stage the exact patch without writing the target tree",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = apply_patch(args.teleop_root, check_only=args.check)
    except (OSError, PatchApplicationError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
