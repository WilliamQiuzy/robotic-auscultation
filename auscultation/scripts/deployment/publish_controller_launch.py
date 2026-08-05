"""Build a content-addressed dormant OpenArm controller-launch sidecar.

This publisher copies only externally hash-confirmed bytes.  It never executes
the controller, preflight tool, stop tool, or any CAN operation.  The emitted
release remains ``HIL_PENDING`` and is suitable only for display/provenance
until a separate supervised HIL capability authorizes execution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Mapping, Union

from ausc.perception.robot import controller_launch_activation as activation
from ausc.perception.robot.openarm_pair_activation import (
    OpenArmPairActivation,
    load_activated_openarm_pair,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 1 << 31
_SOURCE_PATHS = frozenset({
    activation.CONTROLLER_PATH,
    activation.LEADER_CONFIG_PATH,
    activation.FOLLOWER_CONFIG_PATH,
    activation.LEADER_URDF_PATH,
    activation.FOLLOWER_URDF_PATH,
    activation.PREFLIGHT_PATH,
})
_ROLES = {
    activation.CONTROLLER_PATH: "controller_elf",
    activation.LEADER_CONFIG_PATH: "leader_config",
    activation.FOLLOWER_CONFIG_PATH: "follower_config",
    activation.LEADER_URDF_PATH: "leader_urdf",
    activation.FOLLOWER_URDF_PATH: "follower_urdf",
    activation.PREFLIGHT_PATH: "preflight_elf",
    activation.STOP_TOOL_DESCRIPTOR_PATH: "stop_tool_descriptor",
}
_EXECUTABLES = frozenset({
    activation.CONTROLLER_PATH,
    activation.PREFLIGHT_PATH,
})


class ControllerLaunchPublishError(RuntimeError):
    """The requested dormant sidecar cannot be published safely."""


@dataclass(frozen=True)
class PublishedControllerLaunch:
    trusted_root: Path
    release_directory: Path
    release_id: str
    contract_sha256: str


def _confirmed_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ControllerLaunchPublishError(
            "%s must be a lowercase SHA-256" % label
        )
    return value


def _stable_confirmed_file(
    source: object,
    *,
    expected_sha256: str,
    label: str,
) -> bytes:
    path = Path(os.fspath(source))
    if not path.is_absolute() or path != Path(os.path.abspath(os.fspath(path))):
        raise ControllerLaunchPublishError("%s must be canonical and absolute" % label)
    try:
        info = path.lstat()
    except OSError as exc:
        raise ControllerLaunchPublishError("%s is missing" % label) from exc
    if stat.S_ISLNK(info.st_mode):
        raise ControllerLaunchPublishError("%s must not be a symlink" % label)
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= _MAX_ARTIFACT_BYTES:
        raise ControllerLaunchPublishError("%s must be a bounded regular file" % label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise ControllerLaunchPublishError("%s cannot be opened" % label) from exc
    try:
        before = os.fstat(descriptor)
        chunks = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise ControllerLaunchPublishError(
                    "%s was truncated while reading" % label
                )
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ControllerLaunchPublishError("%s changed while reading" % label)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ControllerLaunchPublishError("%s SHA-256 mismatch" % label)
    return raw


def _write_payload(path: Path, raw: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _lock_directories(root: Path) -> None:
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        directory.chmod(0o555)


def _loaded_result(
    trusted_root: Path,
    release: Path,
    *,
    pair: OpenArmPairActivation,
    contract_sha256: str,
) -> PublishedControllerLaunch:
    loaded = activation.load_controller_launch_activation(
        release,
        trusted_root=trusted_root,
        expected_contract_sha256=contract_sha256,
        openarm_pair=pair,
    )
    return PublishedControllerLaunch(
        trusted_root=trusted_root,
        release_directory=release,
        release_id=loaded.release_id,
        contract_sha256=loaded.contract_sha256,
    )


def build_controller_launch_release(
    *,
    openarm_pair: OpenArmPairActivation,
    artifact_sources: Mapping[str, Union[str, Path]],
    expected_artifact_sha256: Mapping[str, str],
    releases_root: Union[str, Path],
) -> PublishedControllerLaunch:
    """Build and verify one dormant release without executing any payload."""

    if set(artifact_sources) != _SOURCE_PATHS:
        raise ControllerLaunchPublishError(
            "artifact source set must match the exact dormant launch allowlist"
        )
    if set(expected_artifact_sha256) != _SOURCE_PATHS:
        raise ControllerLaunchPublishError(
            "artifact SHA-256 confirmations must match the exact allowlist"
        )
    blobs = {}
    for relative in sorted(_SOURCE_PATHS):
        expected = _confirmed_sha256(
            expected_artifact_sha256[relative],
            label="%s confirmation" % relative,
        )
        blobs[relative] = _stable_confirmed_file(
            artifact_sources[relative],
            expected_sha256=expected,
            label=relative,
        )

    stop_descriptor = activation.canonical_json_bytes({
        "schema": activation.STOP_TOOL_DESCRIPTOR_SCHEMA,
        "status": "HIL_PENDING",
    })
    blobs[activation.STOP_TOOL_DESCRIPTOR_PATH] = stop_descriptor
    artifacts = []
    for relative, raw in sorted(blobs.items()):
        artifacts.append({
            "mode": "0555" if relative in _EXECUTABLES else "0444",
            "path": relative,
            "role": _ROLES[relative],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        })
    profile = {
        "argv": [
            activation.CONTROLLER_PATH,
            activation.LEADER_URDF_PATH,
            activation.FOLLOWER_URDF_PATH,
            "left_arm",
            "can1",
            "can3",
        ],
        "artifacts": artifacts,
        "cwd": ".",
        "environment": activation.build_controller_environment(openarm_pair),
        "open_target_status": "HIL_PENDING",
        "openarm_pair": {
            "contract_sha256": openarm_pair.contract_sha256,
            "release_id": openarm_pair.release_id,
            "schema": openarm_pair.schema,
        },
        "schema": activation.PROFILE_SCHEMA,
        "state": "DORMANT_HIL_PENDING",
    }
    envelope = activation.build_controller_launch_envelope(
        profile,
        openarm_pair=openarm_pair,
    )
    contract_raw = activation.canonical_json_bytes(envelope)
    complete_raw = activation.canonical_json_bytes(
        activation.build_completion_marker(envelope, contract_raw)
    )
    digest = str(envelope["contract_sha256"])
    release_id = str(envelope["release_id"])
    root_name = "controller_launch_root_v1-" + digest

    releases = Path(releases_root)
    if not releases.is_absolute() or releases != Path(os.path.abspath(os.fspath(releases))):
        raise ControllerLaunchPublishError(
            "controller launch releases root must be canonical and absolute"
        )
    if releases.is_symlink() or (releases.exists() and not releases.is_dir()):
        raise ControllerLaunchPublishError(
            "controller launch releases root must be a real directory"
        )
    destination_root = releases / root_name
    destination_release = destination_root / release_id
    if destination_root.exists():
        try:
            return _loaded_result(
                destination_root,
                destination_release,
                pair=openarm_pair,
                contract_sha256=digest,
            )
        except activation.ControllerLaunchActivationError as exc:
            raise ControllerLaunchPublishError(
                "existing content-addressed controller launch release is invalid"
            ) from exc

    releases.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(
        prefix=".controller-launch-staging-",
        dir=os.fspath(releases.parent),
    ))
    staged_root = workspace / root_name
    staged_release = staged_root / release_id
    staged_release.mkdir(parents=True)
    try:
        _write_payload(
            staged_release / activation.STAGING_MARKER,
            b"HIL_PENDING\n",
            mode=0o444,
        )
        for relative, raw in sorted(blobs.items()):
            _write_payload(
                staged_release / relative,
                raw,
                mode=0o555 if relative in _EXECUTABLES else 0o444,
            )
        _write_payload(
            staged_release / activation.CONTRACT_FILE,
            contract_raw,
            mode=0o444,
        )
        _write_payload(
            staged_release / activation.COMPLETE_MARKER,
            complete_raw,
            mode=0o444,
        )
        (staged_release / activation.STAGING_MARKER).unlink()
        # macOS refuses a cross-parent directory rename after the moved root
        # itself is made non-writable.  Lock the complete release subtree first,
        # keep only its empty wrapper writable for the atomic rename, and lock
        # that wrapper immediately after publication.
        _lock_directories(staged_release)
        staged_root.chmod(0o755)
        try:
            os.rename(os.fspath(staged_root), os.fspath(destination_root))
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            try:
                return _loaded_result(
                    destination_root,
                    destination_release,
                    pair=openarm_pair,
                    contract_sha256=digest,
                )
            except activation.ControllerLaunchActivationError as verify_exc:
                raise ControllerLaunchPublishError(
                    "concurrent controller launch publish did not match"
                ) from verify_exc
        destination_root.chmod(0o555)
        return _loaded_result(
            destination_root,
            destination_release,
            pair=openarm_pair,
            contract_sha256=digest,
        )
    finally:
        shutil.rmtree(os.fspath(workspace), ignore_errors=True)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a dormant, hash-confirmed controller launch sidecar",
    )
    parser.add_argument("--pair-release", required=True)
    parser.add_argument("--pair-contract-sha256", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--leader-config", required=True)
    parser.add_argument("--leader-config-sha256", required=True)
    parser.add_argument("--follower-config", required=True)
    parser.add_argument("--follower-config-sha256", required=True)
    parser.add_argument("--leader-urdf", required=True)
    parser.add_argument("--leader-urdf-sha256", required=True)
    parser.add_argument("--follower-urdf", required=True)
    parser.add_argument("--follower-urdf-sha256", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--releases-root", required=True)
    return parser


def main(argv: object = None) -> int:
    args = _argument_parser().parse_args(argv)
    pair = load_activated_openarm_pair(
        Path(args.pair_release),
        expected_contract_sha256=args.pair_contract_sha256,
    )
    sources = {
        activation.CONTROLLER_PATH: Path(args.controller),
        activation.LEADER_CONFIG_PATH: Path(args.leader_config),
        activation.FOLLOWER_CONFIG_PATH: Path(args.follower_config),
        activation.LEADER_URDF_PATH: Path(args.leader_urdf),
        activation.FOLLOWER_URDF_PATH: Path(args.follower_urdf),
        activation.PREFLIGHT_PATH: Path(args.preflight),
    }
    expected = {
        activation.CONTROLLER_PATH: args.controller_sha256,
        activation.LEADER_CONFIG_PATH: args.leader_config_sha256,
        activation.FOLLOWER_CONFIG_PATH: args.follower_config_sha256,
        activation.LEADER_URDF_PATH: args.leader_urdf_sha256,
        activation.FOLLOWER_URDF_PATH: args.follower_urdf_sha256,
        activation.PREFLIGHT_PATH: args.preflight_sha256,
    }
    result = build_controller_launch_release(
        openarm_pair=pair,
        artifact_sources=sources,
        expected_artifact_sha256=expected,
        releases_root=Path(args.releases_root),
    )
    sys.stdout.write(activation.canonical_json_bytes({
        "contract_sha256": result.contract_sha256,
        "release_directory": str(result.release_directory),
        "release_id": result.release_id,
        "status": "COMPLETE_HIL_PENDING",
        "trusted_root": str(result.trusted_root),
    }).decode("utf-8"))
    return 0


__all__ = [
    "ControllerLaunchPublishError",
    "PublishedControllerLaunch",
    "build_controller_launch_release",
]


if __name__ == "__main__":
    raise SystemExit(main())
