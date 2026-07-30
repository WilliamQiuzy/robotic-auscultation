#!/usr/bin/env python3
"""CAS-guarded candidate patch for OpenArm CAN 1.2.2 TX enqueue status.

This tool patches source only.  It never opens SocketCAN and never starts a
controller.  A hash mismatch causes zero target writes.
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


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "lingbox_openarm_baseline.json"
PATCH = HERE / "openarm_can_1.2.2.tx_status.patch"
STATUS_HEADER = HERE / "openarm_can_tx_enqueue_status.hpp"
INPUTS = {
    "include/openarm/damiao_motor/dm_motor_device_collection.hpp": (
        "openarm_can_collection_header_source_sha256"
    ),
    "src/openarm/damiao_motor/dm_motor_device_collection.cpp": (
        "openarm_can_collection_source_sha256"
    ),
    "include/openarm/canbus/can_socket.hpp": (
        "openarm_can_socket_header_source_sha256"
    ),
    "src/openarm/canbus/can_socket.cpp": "openarm_can_socket_source_sha256",
}
STATUS_RELATIVE = (
    "include/openarm/damiao_motor/openarm_can_tx_enqueue_status.hpp"
)


class PatchError(RuntimeError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require(path: Path, expected: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PatchError(f"required regular file is missing: {path}")
    payload = path.read_bytes()
    actual = _sha256(payload)
    if actual != expected:
        raise PatchError(
            f"SHA-256 mismatch for {path}: expected {expected}, received {actual}"
        )
    return payload


def _atomic_replace(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.wam-tx-", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_baseline() -> dict:
    value = json.loads(BASELINE.read_text(encoding="utf-8"))
    if (
        value.get("candidate_schema") != "wam_openarm_tx_status_candidate_v3"
        or value.get("openarm_can_version") != "1.2.2"
        or value.get("openarm_can_git_commit")
        != "3363f9879f8947f79db179e4e91f6dbf78f44a6f"
    ):
        raise PatchError("unsupported OpenArm CAN candidate baseline")
    return value


def _stage(payloads: dict[str, bytes]) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="wam-openarm-can-tx-stage-") as raw:
        root = Path(raw)
        for relative, payload in payloads.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        status = root / STATUS_RELATIVE
        status.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(STATUS_HEADER, status)
        result = subprocess.run(
            ["patch", "--batch", "--forward", "-F0", "-p1", "-i", str(PATCH)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise PatchError(
                "OpenArm CAN 1.2.2 patch did not apply exactly: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        source = (
            root / "src/openarm/damiao_motor/dm_motor_device_collection.cpp"
        ).read_text(encoding="utf-8")
        if (
            "MITControlSendStatus DMDeviceCollection::mit_control_all" not in source
            or "return can_socket_.write_can_frame(frame);" not in source
            or "return can_socket_.write_canfd_frame(frame);" not in source
        ):
            raise PatchError("patched SDK failed TX enqueue status audit")
        outputs = {
            relative: (root / relative).read_bytes()
            for relative in payloads
            if "dm_motor_device_collection" in relative
        }
        outputs[STATUS_RELATIVE] = status.read_bytes()
        return outputs


def apply(root: Path, *, check_only: bool) -> dict:
    source_root = root.expanduser().resolve()
    baseline = _load_baseline()
    payloads = {
        relative: _require(source_root / relative, baseline[field])
        for relative, field in INPUTS.items()
    }
    outputs = _stage(payloads)
    result = {
        "schema": "wam_openarm_can_tx_status_patch_result_v3",
        "mode": "check" if check_only else "applied",
        "openarm_can_version": "1.2.2",
        "openarm_can_git_commit": baseline["openarm_can_git_commit"],
        "status_semantics": (
            "linux_socketcan_full_frame_write_tx_enqueue_not_motor_ack"
        ),
        "outputs": {
            relative: _sha256(payload)
            for relative, payload in sorted(outputs.items())
        },
    }
    if check_only:
        return result
    for relative, field in INPUTS.items():
        _require(source_root / relative, baseline[field])
    originals = {}
    for relative in outputs:
        target = source_root / relative
        originals[target] = (
            target.exists(),
            target.read_bytes() if target.exists() else b"",
            (target.stat().st_mode & 0o777) if target.exists() else 0o644,
        )
    try:
        for relative, payload in outputs.items():
            target = source_root / relative
            _atomic_replace(target, payload, originals[target][2])
    except Exception:
        for target, (existed, payload, mode) in originals.items():
            if existed:
                _atomic_replace(target, payload, mode)
            else:
                target.unlink(missing_ok=True)
        raise
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openarm-can-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = apply(args.openarm_can_root, check_only=args.check)
    except (OSError, ValueError, json.JSONDecodeError, PatchError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
