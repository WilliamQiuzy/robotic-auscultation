#!/usr/bin/env python3
"""CAS-guarded OpenArm CAN 1.2.2 feedback-liveness patch applier."""

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


HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "lingbox_openarm_baseline.json"
PATCH_PATH = HERE / "openarm_can_1.2.2.feedback_liveness.patch"
TRACKER_HEADER = HERE / "openarm_can_feedback_liveness.hpp"
PATCH_INPUTS = {
    "CMakeLists.txt": "openarm_can_cmake_source_sha256",
    "include/openarm/damiao_motor/dm_motor.hpp": (
        "openarm_can_motor_header_source_sha256"
    ),
    "src/openarm/damiao_motor/dm_motor.cpp": (
        "openarm_can_motor_source_sha256"
    ),
    "src/openarm/damiao_motor/dm_motor_device.cpp": (
        "openarm_can_motor_device_source_sha256"
    ),
}
TRACKER_TARGET = (
    "include/openarm/damiao_motor/openarm_can_feedback_liveness.hpp"
)
TX_PREREQUISITES = {
    "include/openarm/damiao_motor/dm_motor_device_collection.hpp": (
        "openarm_can_tx_collection_header_output_sha256"
    ),
    "src/openarm/damiao_motor/dm_motor_device_collection.cpp": (
        "openarm_can_tx_collection_source_output_sha256"
    ),
    "include/openarm/damiao_motor/openarm_can_tx_enqueue_status.hpp": (
        "openarm_can_tx_status_header_output_sha256"
    ),
}


class PatchApplicationError(RuntimeError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_hash(path: Path, expected: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PatchApplicationError(f"required regular file is missing: {path}")
    payload = path.read_bytes()
    actual = _sha256(payload)
    if actual != expected:
        raise PatchApplicationError(
            f"SHA-256 mismatch for {path}: expected {expected}, received {actual}"
        )
    return payload


def _load_baseline() -> dict:
    try:
        value = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatchApplicationError("cannot read OpenArm CAN baseline") from exc
    if (
        value.get("feedback_candidate_schema")
        != "wam_openarm_feedback_liveness_candidate_v4"
        or value.get("openarm_can_version") != "1.2.2"
        or value.get("openarm_can_git_commit")
        != "3363f9879f8947f79db179e4e91f6dbf78f44a6f"
    ):
        raise PatchApplicationError("unsupported OpenArm CAN feedback baseline")
    return value


def _atomic_replace(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.wam-feedback-", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stage(base_payloads: dict[str, bytes]) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="wam-openarm-feedback-stage-") as raw:
        stage = Path(raw)
        for relative, payload in base_payloads.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        target = stage / TRACKER_TARGET
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TRACKER_HEADER, target)
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
                "OpenArm CAN feedback patch did not apply exactly: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        outputs = {
            relative: (stage / relative).read_bytes()
            for relative in (*PATCH_INPUTS, TRACKER_TARGET)
        }
        motor_header = outputs[
            "include/openarm/damiao_motor/dm_motor.hpp"
        ].decode("utf-8")
        device_source = outputs[
            "src/openarm/damiao_motor/dm_motor_device.cpp"
        ].decode("utf-8")
        cmake_source = outputs["CMakeLists.txt"].decode("utf-8")
        if (
            "get_feedback_liveness_snapshot" not in motor_header
            or "CLOCK_MONOTONIC" not in device_source
            or "openarm_can_tx_enqueue_status.hpp" not in cmake_source
            or "openarm_can_feedback_liveness.hpp" not in cmake_source
            or device_source.count(
                "observe_successfully_decoded_matching_state"
            ) != 2
            or device_source.count("motor_.update_state") != 2
        ):
            raise PatchApplicationError(
                "staged SDK feedback hook failed source audit"
            )
        return outputs


def apply_patch(root: Path, *, check_only: bool) -> dict:
    sdk_root = root.expanduser().resolve()
    if not sdk_root.is_dir():
        raise PatchApplicationError(f"OpenArm CAN root is not a directory: {sdk_root}")
    baseline = _load_baseline()
    base_payloads = {
        relative: _require_hash(sdk_root / relative, baseline[hash_field])
        for relative, hash_field in PATCH_INPUTS.items()
    }
    for relative, hash_field in TX_PREREQUISITES.items():
        _require_hash(sdk_root / relative, baseline[hash_field])
    outputs = _stage(base_payloads)
    result = {
        "schema": "wam_openarm_can_feedback_patch_result_v1",
        "mode": "check" if check_only else "applied",
        "openarm_can_root": str(sdk_root),
        "openarm_can_version": "1.2.2",
        "base_inputs": {
            relative: baseline[field]
            for relative, field in sorted(PATCH_INPUTS.items())
        },
        "tx_prerequisites": {
            relative: baseline[field]
            for relative, field in sorted(TX_PREREQUISITES.items())
        },
        "outputs": {
            relative: _sha256(payload)
            for relative, payload in sorted(outputs.items())
        },
    }
    if check_only:
        return result

    for relative, hash_field in TX_PREREQUISITES.items():
        _require_hash(sdk_root / relative, baseline[hash_field])
    for relative, hash_field in PATCH_INPUTS.items():
        _require_hash(sdk_root / relative, baseline[hash_field])
    originals: dict[Path, tuple[bool, bytes, int]] = {}
    for relative in outputs:
        path = sdk_root / relative
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise PatchApplicationError(f"refusing non-regular output: {path}")
            originals[path] = (True, path.read_bytes(), path.stat().st_mode & 0o777)
        else:
            originals[path] = (False, b"", 0o644)
    try:
        for relative, payload in outputs.items():
            path = sdk_root / relative
            _atomic_replace(path, payload, originals[path][2])
    except Exception:
        for path, (existed, payload, mode) in originals.items():
            if existed:
                _atomic_replace(path, payload, mode)
            else:
                path.unlink(missing_ok=True)
        raise
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openarm-can-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = apply_patch(args.openarm_can_root, check_only=args.check)
    except (OSError, PatchApplicationError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
