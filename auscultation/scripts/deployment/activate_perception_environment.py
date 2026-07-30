"""CAS-activate one verified Linbox perception environment.

The production entry point is deliberately fixed to
``/home/robot/wam_data/perception_envs``.  Relocation-aware v3 candidates are
built and completed in ``releases/<environment_id>``.  Activation verifies
that permanent, recursively immutable path and then performs only the
``current`` pointer compare-and-swap.  It never renames, rewrites, or freezes a
v3 candidate.  The older staged move/rebind/freeze workflow remains available
only for legacy v2 locks.

Post-activation rollback is not a generic pointer rewrite.  It accepts only
the canonical ACTIVE evidence created by this tool, verifies the recorded
active symlink inode is still current, fully verifies the recorded previous
release with an exact strong v2/v3 lock, and then performs another pointer
CAS.  Legacy v1 locks are deliberately ineligible.

All fallible environment work happens before the pointer CAS.  A failed
publication therefore leaves the production pointer untouched and preserves
the staged or published candidate for forensic inspection.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import uuid

import artifact_contract as contract
import publish_perception_runtime as publisher
import verify_perception_environment as verifier


LINBOX_PERCEPTION_ROOT = Path("/home/robot/wam_data/perception_envs")
LINBOX_STAGING_ROOT = LINBOX_PERCEPTION_ROOT / ".staging"
LINBOX_RELEASES_ROOT = LINBOX_PERCEPTION_ROOT / "releases"
LINBOX_CURRENT_LINK = LINBOX_PERCEPTION_ROOT / "current"
LINBOX_ACTIVATION_EVIDENCE_ROOT = LINBOX_STAGING_ROOT / "activation_evidence"
ACTIVATION_EVIDENCE_SCHEMA = "wam_linbox_perception_environment_activation_v1"
ROLLBACK_EVIDENCE_SCHEMA = "wam_linbox_perception_environment_rollback_v1"
DEFAULT_LOCK = verifier.DEFAULT_LOCK
DEFAULT_VERIFIER = Path(__file__).resolve().with_name(
    "verify_perception_environment.py"
)

_ENVIRONMENT_ID = re.compile(r"^linbox-perception-[0-9a-f]{20}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STAGING_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ActivationError(RuntimeError):
    """The perception environment could not be activated safely."""


class ActivationPolicyError(ActivationError):
    """A request violates the fixed Linbox production policy."""


class ActivationConflictError(ActivationError):
    """Managed state changed or already exists."""


class ActivationVerificationError(ActivationError):
    """The candidate does not satisfy its exact strong environment lock."""


VerifierRunner = Callable[[Path, Path], Mapping[str, object]]
PostSwitchCallback = Callable[[], None]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(contract.canonical_json_bytes(value)).hexdigest()


def _normalized_absolute(path: Union[str, Path], label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ActivationPolicyError("%s must be absolute" % label)
    normalized = Path(os.path.abspath(str(candidate)))
    if candidate != normalized:
        raise ActivationPolicyError("%s must be a normalized path" % label)
    return candidate


def _require_real_directory(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ActivationPolicyError("%s must be a real directory" % label) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ActivationPolicyError("%s must be a real directory" % label)
    return info


def _reject_symlink_ancestors(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise ActivationPolicyError("%s contains a symlink ancestor" % label)


def _validate_topology(
    candidate: Union[str, Path],
    releases_root: Union[str, Path],
    current_link: Union[str, Path],
) -> Tuple[Path, Path, Path, Path]:
    staged = _normalized_absolute(candidate, "staged environment")
    releases = _normalized_absolute(releases_root, "releases root")
    current = _normalized_absolute(current_link, "current link")
    root = releases.parent
    staging = root / ".staging"
    if (
        releases.name != "releases"
        or current != root / "current"
        or staged.parent != staging
        or not _STAGING_NAME.fullmatch(staged.name)
    ):
        raise ActivationPolicyError("managed paths do not use the canonical topology")
    for path, label in (
        (root, "perception environment root"),
        (staging, "perception staging root"),
        (releases, "perception releases root"),
    ):
        _require_real_directory(path, label)
        _reject_symlink_ancestors(path, label)
    if os.path.lexists(str(staged)):
        _require_real_directory(staged, "staged environment")
        _reject_symlink_ancestors(staged, "staged environment")
    if os.path.lexists(str(current)) and not current.is_symlink():
        raise ActivationPolicyError("current must be absent or a symlink")
    return root, staging, releases, current


def _validate_final_path_topology(
    candidate: Union[str, Path],
    releases_root: Union[str, Path],
    current_link: Union[str, Path],
    environment_id: str,
) -> Tuple[Path, Path, Path, Path]:
    final = _normalized_absolute(candidate, "final-path environment")
    releases = _normalized_absolute(releases_root, "releases root")
    current = _normalized_absolute(current_link, "current link")
    root = releases.parent
    staging = root / ".staging"
    if (
        releases.name != "releases"
        or current != root / "current"
        or final != releases / environment_id
    ):
        raise ActivationPolicyError(
            "v3 candidate must already be at its permanent final release path"
        )
    for path, label in (
        (root, "perception environment root"),
        (staging, "perception staging root"),
        (releases, "perception releases root"),
        (final, "final-path environment"),
    ):
        _require_real_directory(path, label)
        _reject_symlink_ancestors(path, label)
    if os.path.lexists(str(current)) and not current.is_symlink():
        raise ActivationPolicyError("current must be absent or a symlink")
    return root, staging, releases, current


def _load_strong_lock(
    lock_path: Union[str, Path],
) -> Tuple[dict, str, Path]:
    path = _normalized_absolute(lock_path, "environment lock")
    _reject_symlink_ancestors(path, "environment lock")
    try:
        lock, lock_sha256 = verifier.load_environment_lock(path)
    except verifier.EnvironmentVerificationError as exc:
        raise ActivationVerificationError(str(exc)) from exc
    if lock.get("schema") not in {
        publisher.ENVIRONMENT_LOCK_SCHEMA,
        publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA,
    }:
        raise ActivationVerificationError(
            "activation requires an exact strong v2 or v3 lock"
        )
    if not publisher.is_strong_environment_lock(lock):
        raise ActivationVerificationError(
            "activation requires the strong installed-file byte lock"
        )
    environment_id = lock.get("environment_id")
    if (
        not isinstance(environment_id, str)
        or not _ENVIRONMENT_ID.fullmatch(environment_id)
    ):
        raise ActivationVerificationError(
            "strong environment lock has an invalid environment_id"
        )
    return lock, lock_sha256, path


def _load_exact_lock(
    lock_path: Union[str, Path], environment_id: str,
) -> Tuple[dict, str, Path]:
    if not isinstance(environment_id, str) or not _ENVIRONMENT_ID.fullmatch(
        environment_id
    ):
        raise ActivationPolicyError("environment_id is invalid")
    lock, lock_sha256, path = _load_strong_lock(lock_path)
    if lock.get("environment_id") != environment_id:
        raise ActivationVerificationError("environment_id does not match the lock")
    return lock, lock_sha256, path


def _validate_confirmations(
    *, execute: bool, environment_id: str, lock_sha256: str,
    confirm_environment_id: Optional[str], confirm_lock_sha256: Optional[str],
) -> None:
    if not execute:
        return
    if (
        confirm_environment_id != environment_id
        or confirm_lock_sha256 != lock_sha256
    ):
        raise ActivationPolicyError(
            "execute double confirmation does not match environment id and lock"
        )


def _pointer_identity(path: Path) -> Tuple[int, int, int, int]:
    info = path.lstat()
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_size,
    )


def _pointer_identity_evidence(
    identity: Optional[Tuple[int, int, int, int]],
) -> Optional[dict]:
    if identity is None:
        return None
    return {
        "device": identity[0],
        "inode": identity[1],
        "mode": identity[2],
        "size": identity[3],
    }


def _pointer_state_evidence(
    environment_id: Optional[str], target: Optional[str],
    identity: Optional[Tuple[int, int, int, int]],
) -> Optional[dict]:
    if environment_id is None and target is None and identity is None:
        return None
    if (
        not isinstance(environment_id, str)
        or not _ENVIRONMENT_ID.fullmatch(environment_id)
        or target != "releases/%s" % environment_id
        or identity is None
    ):
        raise ActivationVerificationError("pointer state evidence is incomplete")
    return {
        "environment_id": environment_id,
        "identity": _pointer_identity_evidence(identity),
        "target": target,
    }


def _parse_pointer_state_evidence(
    value: object, *, label: str,
) -> Tuple[str, str, Tuple[int, int, int, int]]:
    if not isinstance(value, Mapping) or set(value) != {
        "environment_id", "identity", "target",
    }:
        raise ActivationVerificationError("%s is invalid" % label)
    environment_id = value["environment_id"]
    target = value["target"]
    identity = value["identity"]
    if (
        not isinstance(environment_id, str)
        or not _ENVIRONMENT_ID.fullmatch(environment_id)
        or target != "releases/%s" % environment_id
        or not isinstance(identity, Mapping)
        or set(identity) != {"device", "inode", "mode", "size"}
        or any(type(identity[key]) is not int or identity[key] < 0 for key in identity)
    ):
        raise ActivationVerificationError("%s is invalid" % label)
    return (
        environment_id,
        target,
        (
            identity["device"], identity["inode"],
            identity["mode"], identity["size"],
        ),
    )


def _directory_identity(path: Path, label: str) -> Tuple[int, int, int]:
    info = _require_real_directory(path, label)
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))


def _open_real_directory(path: Path, label: str) -> int:
    before = _directory_identity(path, label)
    try:
        descriptor = os.open(
            str(path),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ActivationConflictError("%s changed while opening" % label) from exc
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino, stat.S_IFMT(opened.st_mode)) != before:
        os.close(descriptor)
        raise ActivationConflictError("%s inode changed while opening" % label)
    return descriptor


def _read_current(
    current: Path, releases: Path,
) -> Tuple[Optional[str], Optional[str], Optional[Tuple[int, int, int, int]]]:
    if not os.path.lexists(str(current)):
        return None, None, None
    try:
        info = current.lstat()
    except FileNotFoundError:
        return None, None, None
    if not stat.S_ISLNK(info.st_mode):
        raise ActivationPolicyError("current must be absent or a symlink")
    target = os.readlink(str(current))
    prefix = "releases/"
    if not target.startswith(prefix) or target.count("/") != 1:
        raise ActivationPolicyError("current must use a canonical relative target")
    environment_id = target[len(prefix):]
    if not _ENVIRONMENT_ID.fullmatch(environment_id):
        raise ActivationPolicyError("current environment id is invalid")
    release = releases / environment_id
    _reject_symlink_ancestors(release, "current environment")
    _require_real_directory(release, "current environment")
    return environment_id, target, _pointer_identity(current)


def _validate_marker(
    marker: Mapping[str, object], *, candidate: Path, environment_id: str,
    lock: Mapping[str, object], lock_sha256: str,
) -> None:
    if marker.get("environment_root") != str(candidate):
        raise ActivationVerificationError("clone marker environment root mismatch")
    if marker.get("environment_id") != environment_id:
        raise ActivationVerificationError("clone marker environment_id mismatch")
    if marker.get("lock_sha256") != lock_sha256:
        raise ActivationVerificationError("clone marker lock SHA-256 mismatch")
    if marker.get("source_environment_fingerprint_sha256") != lock[
        "provisioning"
    ]["source_environment_fingerprint_sha256"]:
        raise ActivationVerificationError("clone marker source fingerprint mismatch")
    relative = marker.get("python_executable_relative")
    if not isinstance(relative, str):
        raise ActivationVerificationError("clone marker Python path is invalid")
    python_relative = Path(relative)
    if python_relative.is_absolute() or ".." in python_relative.parts:
        raise ActivationVerificationError("clone marker Python path is unsafe")
    _require_environment_file(
        candidate, python_relative, "candidate Python"
    )


def _require_environment_file(root: Path, relative: Path, label: str) -> Path:
    path = root / relative
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ActivationVerificationError("%s escapes the environment" % label) from exc
    except OSError as exc:
        raise ActivationVerificationError("%s is missing or invalid" % label) from exc
    if not resolved.is_file():
        raise ActivationVerificationError("%s is not a regular file" % label)
    return path


def _validate_tree_symlinks(root: Path) -> int:
    """Accept only relative symlink chains resolving inside this exact root."""
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ActivationVerificationError(
            "environment root cannot be resolved for symlink validation"
        ) from exc
    count = 0
    for raw_directory, dirnames, filenames in os.walk(
        str(root), topdown=True, onerror=_raise_walk_error, followlinks=False
    ):
        directory = Path(raw_directory)
        for name in list(dirnames) + list(filenames):
            path = directory / name
            if not path.is_symlink():
                continue
            if name in dirnames:
                dirnames.remove(name)
            try:
                raw_target = os.readlink(str(path))
            except OSError as exc:
                raise ActivationVerificationError(
                    "environment symlink cannot be read: %s" % path
                ) from exc
            if Path(raw_target).is_absolute():
                raise ActivationVerificationError(
                    "environment symlink uses an absolute target: %s" % path
                )
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(resolved_root)
            except FileNotFoundError as exc:
                raise ActivationVerificationError(
                    "environment symlink is dangling: %s" % path
                ) from exc
            except RuntimeError as exc:
                raise ActivationVerificationError(
                    "environment symlink chain is invalid: %s" % path
                ) from exc
            except ValueError as exc:
                raise ActivationVerificationError(
                    "environment symlink escapes its root: %s" % path
                ) from exc
            except OSError as exc:
                raise ActivationVerificationError(
                    "environment symlink cannot be resolved: %s" % path
                ) from exc
            count += 1
    return count


def _validate_tree_hardlinks(root: Path) -> dict:
    regular_entries = 0
    inodes = set()
    for raw_directory, dirnames, filenames in os.walk(
        str(root), topdown=True, onerror=_raise_walk_error, followlinks=False
    ):
        directory = Path(raw_directory)
        for name in list(dirnames):
            if (directory / name).is_symlink():
                dirnames.remove(name)
        for name in filenames:
            path = directory / name
            if path.is_symlink():
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ActivationVerificationError(
                    "hardlink audit found a non-regular file: %s" % path
                )
            if info.st_nlink != 1:
                raise ActivationVerificationError(
                    "environment regular file has an external hardlink or multiple links: %s"
                    % path
                )
            regular_entries += 1
            inodes.add((info.st_dev, info.st_ino))
    if len(inodes) != regular_entries:
        raise ActivationVerificationError(
            "environment hardlink inode accounting is inconsistent"
        )
    return {
        "regular_entries": regular_entries,
        "status": "VERIFIED_NO_EXTERNAL_HARDLINKS",
        "unique_inodes": len(inodes),
    }


def _raise_walk_error(error: OSError) -> None:
    raise ActivationVerificationError(
        "environment tree traversal failed: %s" % type(error).__name__
    ) from error


def _run_verifier_subprocess(
    environment_root: Path,
    lock_path: Path,
    *,
    verifier_script: Path = DEFAULT_VERIFIER,
    timeout_seconds: int = 600,
) -> Mapping[str, object]:
    script = _normalized_absolute(verifier_script, "environment verifier")
    if script.is_symlink() or not script.is_file():
        raise ActivationPolicyError("environment verifier must be a regular file")
    marker, _marker_sha256 = verifier.load_clone_marker(
        environment_root / verifier.CLONE_MARKER_NAME
    )
    relative = Path(str(marker["python_executable_relative"]))
    python = _require_environment_file(
        environment_root, relative, "candidate Python"
    )
    process_environment = dict(os.environ)
    process_environment.pop("PYTHONHOME", None)
    process_environment.pop("PYTHONPATH", None)
    process_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process_environment["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [
                str(python),
                str(script),
                "--lock",
                str(lock_path),
                "--environment-root",
                str(environment_root),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
            env=process_environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ActivationVerificationError("candidate verifier could not complete") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[:2048]
        raise ActivationVerificationError(
            "candidate verifier failed: %s" % detail
        )
    try:
        report = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActivationVerificationError(
            "candidate verifier did not emit canonical JSON"
        ) from exc
    if (
        not isinstance(report, dict)
        or completed.stdout != contract.canonical_json_bytes(report)
    ):
        raise ActivationVerificationError(
            "candidate verifier did not emit a canonical JSON object"
        )
    return report


def _validate_verification_report(
    report: Mapping[str, object], *, environment_root: Path,
    environment_id: str, lock_sha256: str,
) -> Dict[str, object]:
    if not isinstance(report, Mapping):
        raise ActivationVerificationError("verifier report must be an object")
    if report.get("schema") != verifier.VERIFICATION_SCHEMA:
        raise ActivationVerificationError("verifier report schema mismatch")
    if report.get("status") != "VERIFIED":
        raise ActivationVerificationError("verifier status must be VERIFIED")
    if report.get("schema_gaps") != []:
        raise ActivationVerificationError("verifier schema_gaps must be empty")
    if report.get("environment_id") != environment_id:
        raise ActivationVerificationError("verified environment_id mismatch")
    if report.get("environment_root") != str(environment_root):
        raise ActivationVerificationError("verified environment root mismatch")
    if report.get("lock_sha256") != lock_sha256:
        raise ActivationVerificationError("verified lock SHA-256 mismatch")
    return dict(report)


def _rewrite_marker_root(
    final: Path, *, marker: Mapping[str, object], environment_id: str,
    lock: Mapping[str, object], lock_sha256: str,
) -> Tuple[dict, str]:
    rebound = dict(marker)
    rebound["environment_root"] = str(final)
    marker_path = final / verifier.CLONE_MARKER_NAME
    contract._atomic_write(marker_path, contract.canonical_json_bytes(rebound))
    try:
        loaded, marker_sha256 = verifier.load_clone_marker(marker_path)
    except verifier.EnvironmentVerificationError as exc:
        raise ActivationVerificationError("rewritten clone marker is invalid") from exc
    _validate_marker(
        loaded,
        candidate=final,
        environment_id=environment_id,
        lock=lock,
        lock_sha256=lock_sha256,
    )
    return loaded, marker_sha256


def _make_tree_immutable(root: Path) -> None:
    directories = []
    for raw_directory, dirnames, filenames in os.walk(
        str(root), topdown=True, onerror=_raise_walk_error, followlinks=False
    ):
        directory = Path(raw_directory)
        directories.append(directory)
        for name in list(dirnames):
            path = directory / name
            if path.is_symlink():
                dirnames.remove(name)
        for name in filenames:
            path = directory / name
            if path.is_symlink():
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ActivationPolicyError(
                    "environment contains a non-regular file: %s" % path
                )
            path.chmod(stat.S_IMODE(info.st_mode) & ~0o222)
    for directory in reversed(directories):
        info = directory.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ActivationPolicyError("environment directory changed during freeze")
        directory.chmod(stat.S_IMODE(info.st_mode) & ~0o222)


def _verify_tree_immutable(root: Path) -> int:
    count = 0
    for raw_directory, dirnames, filenames in os.walk(
        str(root), topdown=True, onerror=_raise_walk_error, followlinks=False
    ):
        directory = Path(raw_directory)
        info = directory.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ActivationVerificationError(
                "immutable environment directory is invalid"
            )
        if stat.S_IMODE(info.st_mode) & 0o222:
            raise ActivationVerificationError(
                "environment directory remains writable: %s" % directory
            )
        count += 1
        for name in list(dirnames):
            path = directory / name
            if path.is_symlink():
                dirnames.remove(name)
        for name in filenames:
            path = directory / name
            if path.is_symlink():
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ActivationVerificationError(
                    "immutable environment contains a non-regular file"
                )
            if stat.S_IMODE(info.st_mode) & 0o222:
                raise ActivationVerificationError(
                    "environment file remains writable: %s" % path
                )
            count += 1
    return count


def _rename_noreplace_between(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise ActivationPolicyError("renameat2 is required for no-replace")
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            source_parent_fd,
            os.fsencode(source_name),
            destination_parent_fd,
            os.fsencode(destination_name),
            1,
        )
    elif sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        if operation is None:
            raise ActivationPolicyError("renameatx_np is required for no-replace")
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            source_parent_fd,
            os.fsencode(source_name),
            destination_parent_fd,
            os.fsencode(destination_name),
            0x00000004,
        )
    else:
        raise ActivationPolicyError("atomic no-replace is unsupported")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise ActivationConflictError("current appeared during compare-and-swap")
    raise OSError(error, os.strerror(error), destination_name)


def _exchange_between(
    first_parent_fd: int,
    first_name: str,
    second_parent_fd: int,
    second_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        flag = 2
    elif sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        flag = 0x00000002
    else:
        operation = None
        flag = 0
    if operation is None:
        raise ActivationPolicyError("atomic pointer exchange is unsupported")
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    result = operation(
        first_parent_fd,
        os.fsencode(first_name),
        second_parent_fd,
        os.fsencode(second_name),
        flag,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), second_name)


def _fsync_fd(descriptor: int) -> None:
    os.fsync(descriptor)


def _activation_evidence_path(staging: Path, environment_id: str) -> Path:
    return staging / "activation_evidence" / (environment_id + ".json")


def _write_activation_evidence(path: Path, value: dict) -> None:
    expected_parent = path.parent.parent / "activation_evidence"
    if (
        path.parent != expected_parent
        or not _ENVIRONMENT_ID.fullmatch(path.stem)
        or path.suffix != ".json"
    ):
        raise ActivationPolicyError("activation evidence path is not canonical")
    staging = path.parent.parent
    if not path.parent.exists():
        path.parent.mkdir(mode=0o755)
        contract._fsync_directory(staging)
    _require_real_directory(path.parent, "activation evidence root")
    _reject_symlink_ancestors(path.parent, "activation evidence root")
    if os.path.lexists(str(path)):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ActivationPolicyError("activation evidence must be a regular file")
    contract._atomic_write(path, contract.canonical_json_bytes(value))


def _load_evidence(path: Path) -> Tuple[dict, str]:
    if not os.path.lexists(str(path)):
        raise ActivationVerificationError("activation evidence is missing")
    if path.is_symlink() or not path.is_file():
        raise ActivationPolicyError("activation evidence must be a regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActivationVerificationError("activation evidence is invalid") from exc
    if not isinstance(value, dict) or raw != contract.canonical_json_bytes(value):
        raise ActivationVerificationError("activation evidence is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _load_active_evidence(path: Path) -> Optional[dict]:
    if not os.path.lexists(str(path)):
        return None
    value, _digest = _load_evidence(path)
    if value.get("status") != "ACTIVE":
        return None
    return value


def _same_pointer_state(
    current: Path,
    releases: Path,
    expected_id: Optional[str],
    expected_target: Optional[str],
    expected_identity: Optional[Tuple[int, int, int, int]],
) -> bool:
    try:
        actual_id, actual_target, actual_identity = _read_current(current, releases)
    except (ActivationError, FileNotFoundError):
        return False
    return (
        actual_id == expected_id
        and actual_target == expected_target
        and actual_identity == expected_identity
    )


def _cas_pointer(
    *, staging: Path, releases: Path, current: Path, environment_id: str,
    previous_id: Optional[str], previous_target: Optional[str],
    previous_identity: Optional[Tuple[int, int, int, int]],
    post_switch: PostSwitchCallback,
) -> str:
    target = "releases/%s" % environment_id
    temporary = staging / (".current.activate-%s" % uuid.uuid4().hex)
    temporary.symlink_to(target)
    base_fd = _open_real_directory(current.parent, "perception environment root")
    staging_fd = _open_real_directory(staging, "perception staging root")
    switched = False
    try:
        _fsync_fd(staging_fd)
        if not _same_pointer_state(
            current,
            releases,
            previous_id,
            previous_target,
            previous_identity,
        ):
            raise ActivationConflictError(
                "production current changed before compare-and-swap"
            )
        if previous_id is None:
            _rename_noreplace_between(
                staging_fd, temporary.name, base_fd, current.name
            )
        else:
            _exchange_between(staging_fd, temporary.name, base_fd, current.name)
            if (
                not temporary.is_symlink()
                or os.readlink(str(temporary)) != previous_target
                or _pointer_identity(temporary) != previous_identity
            ):
                _exchange_between(staging_fd, temporary.name, base_fd, current.name)
                raise ActivationConflictError(
                    "production current changed during compare-and-swap"
                )
        switched = True
        try:
            _fsync_fd(base_fd)
            _fsync_fd(staging_fd)
        except BaseException:
            if previous_id is None:
                os.replace(str(current), str(temporary))
            else:
                _exchange_between(staging_fd, temporary.name, base_fd, current.name)
            try:
                _fsync_fd(base_fd)
                _fsync_fd(staging_fd)
            except OSError:
                pass
            switched = False
            raise
        try:
            post_switch()
        except BaseException:
            if previous_id is None:
                os.replace(str(current), str(temporary))
            else:
                _exchange_between(staging_fd, temporary.name, base_fd, current.name)
            try:
                _fsync_fd(base_fd)
                _fsync_fd(staging_fd)
            except OSError:
                pass
            switched = False
            raise
        if previous_id is not None:
            try:
                temporary.unlink()
                _fsync_fd(staging_fd)
            except OSError:
                # The active pointer is already durable.  A retained old symlink in
                # .staging is forensic evidence, not an activation failure.
                pass
        return target
    finally:
        os.close(staging_fd)
        os.close(base_fd)
        if not switched:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _verify_immutable_final(
    final: Path,
    *,
    lock: Mapping[str, object],
    lock_sha256: str,
    environment_id: str,
    lock_path: Path,
    run_verifier: VerifierRunner,
) -> Tuple[str, Dict[str, object], int, int, dict]:
    final_identity = _directory_identity(final, "published environment")
    try:
        marker, marker_sha256 = verifier.load_clone_marker(
            final / verifier.CLONE_MARKER_NAME
        )
    except verifier.EnvironmentVerificationError as exc:
        raise ActivationVerificationError(
            "published clone marker is invalid"
        ) from exc
    _validate_marker(
        marker,
        candidate=final,
        environment_id=environment_id,
        lock=lock,
        lock_sha256=lock_sha256,
    )
    _validate_tree_symlinks(final)
    _validate_tree_hardlinks(final)
    report = _validate_verification_report(
        run_verifier(final, lock_path),
        environment_root=final,
        environment_id=environment_id,
        lock_sha256=lock_sha256,
    )
    symlinks = _validate_tree_symlinks(final)
    hardlinks = _validate_tree_hardlinks(final)
    immutable_entries = _verify_tree_immutable(final)
    if _directory_identity(final, "published environment") != final_identity:
        raise ActivationConflictError(
            "published environment inode changed during verification"
        )
    return marker_sha256, report, symlinks, immutable_entries, hardlinks


def _verify_rollback_target(
    *,
    releases: Path,
    lock_path: Union[str, Path],
    activated_environment_id: str,
    execute: bool,
    confirm_environment_id: Optional[str],
    confirm_lock_sha256: Optional[str],
    run_verifier: VerifierRunner,
) -> dict:
    lock, lock_sha256, exact_lock_path = _load_strong_lock(lock_path)
    environment_id = lock["environment_id"]
    if environment_id == activated_environment_id:
        raise ActivationPolicyError(
            "rollback target must differ from the environment being activated"
        )
    if (
        execute
        and (
            confirm_environment_id != environment_id
            or confirm_lock_sha256 != lock_sha256
        )
    ):
        raise ActivationPolicyError(
            "rollback-target double confirmation does not match"
        )
    final = releases / environment_id
    final_identity = _directory_identity(final, "rollback target environment")
    marker_sha256, report, symlinks, immutable_entries, hardlinks = (
        _verify_immutable_final(
            final,
            lock=lock,
            lock_sha256=lock_sha256,
            environment_id=environment_id,
            lock_path=exact_lock_path,
            run_verifier=run_verifier,
        )
    )
    if _directory_identity(
        final, "rollback target environment"
    ) != final_identity:
        raise ActivationConflictError(
            "rollback target inode changed during verification"
        )
    return {
        "environment_id": environment_id,
        "environment_root": str(final),
        "hardlink_policy": hardlinks,
        "immutability": {
            "non_symlink_entries": immutable_entries,
            "status": "VERIFIED_A_W_REMOVED",
        },
        "lock_path": str(exact_lock_path),
        "lock_sha256": lock_sha256,
        "marker_sha256": marker_sha256,
        "symlink_policy": {
            "internal_relative_symlinks": symlinks,
            "status": "VERIFIED_NO_DANGLING_OR_ESCAPE",
        },
        "verification_sha256": _canonical_sha256(report),
    }


def _active_evidence(
    *,
    evidence_path: Path,
    environment_id: str,
    final: Path,
    lock_sha256: str,
    marker_sha256: str,
    mode: str,
    previous_id: Optional[str],
    symlinks: int,
    immutable_entries: int,
    hardlinks: Mapping[str, object],
    verification: Mapping[str, str],
    previous_target: Optional[str] = None,
    previous_identity: Optional[Tuple[int, int, int, int]] = None,
    active_pointer: Optional[Mapping[str, object]] = None,
) -> dict:
    return {
        "active_pointer": None if active_pointer is None else dict(active_pointer),
        "activation_evidence_path": str(evidence_path),
        "environment_id": environment_id,
        "environment_root": str(final),
        "hardlink_policy": dict(hardlinks),
        "immutability": {
            "non_symlink_entries": immutable_entries,
            "status": "VERIFIED_A_W_REMOVED",
        },
        "lock_sha256": lock_sha256,
        "marker_sha256": marker_sha256,
        "mode": mode,
        "pointer_target": "releases/%s" % environment_id,
        "previous_environment_id": previous_id,
        "previous_pointer": (
            _pointer_state_evidence(
                previous_id, previous_target, previous_identity
            )
            if previous_target is not None and previous_identity is not None
            else None
        ),
        "schema": ACTIVATION_EVIDENCE_SCHEMA,
        "status": "ACTIVE",
        "symlink_policy": {
            "internal_relative_symlinks": symlinks,
            "status": "VERIFIED_NO_DANGLING_OR_ESCAPE",
        },
        "verification": dict(verification),
    }


def _prepared_evidence(active: Mapping[str, object]) -> dict:
    return {
        "active_pointer": None,
        "activation_evidence_path": active["activation_evidence_path"],
        "environment_id": active["environment_id"],
        "environment_root": active["environment_root"],
        "lock_sha256": active["lock_sha256"],
        "mode": active["mode"],
        "pointer_target": active["pointer_target"],
        "previous_environment_id": active["previous_environment_id"],
        "previous_pointer": active["previous_pointer"],
        "rollback_target": active.get("rollback_target"),
        "schema": ACTIVATION_EVIDENCE_SCHEMA,
        "status": "PREPARED",
    }


def _persist_with_pointer_cas(
    *,
    active_evidence: dict,
    evidence_path: Path,
    staging: Path,
    releases: Path,
    current: Path,
    environment_id: str,
    previous_id: Optional[str],
    previous_target: Optional[str],
    previous_identity: Optional[Tuple[int, int, int, int]],
) -> None:
    active_evidence["previous_pointer"] = _pointer_state_evidence(
        previous_id, previous_target, previous_identity
    )
    prepared = _prepared_evidence(active_evidence)
    _write_activation_evidence(evidence_path, prepared)

    def persist_active() -> None:
        active_id, active_target, active_identity = _read_current(
            current, releases
        )
        if active_id != environment_id:
            raise ActivationConflictError(
                "activated pointer changed before ACTIVE evidence"
            )
        active_evidence["active_pointer"] = _pointer_state_evidence(
            active_id, active_target, active_identity
        )
        _write_activation_evidence(evidence_path, active_evidence)

    try:
        _cas_pointer(
            staging=staging,
            releases=releases,
            current=current,
            environment_id=environment_id,
            previous_id=previous_id,
            previous_target=previous_target,
            previous_identity=previous_identity,
            post_switch=persist_active,
        )
    except BaseException as exc:
        rolled_back = dict(prepared)
        rolled_back["failure_type"] = type(exc).__name__
        rolled_back["status"] = "ROLLED_BACK"
        try:
            _write_activation_evidence(evidence_path, rolled_back)
        except BaseException:
            pass
        raise


def _validate_existing_active_evidence(
    value: Mapping[str, object], *, evidence_path: Path, final: Path,
    environment_id: str, lock_sha256: str,
    hardlinks: Mapping[str, object],
) -> dict:
    if (
        value.get("schema") != ACTIVATION_EVIDENCE_SCHEMA
        or value.get("status") != "ACTIVE"
        or value.get("activation_evidence_path") != str(evidence_path)
        or value.get("environment_id") != environment_id
        or value.get("environment_root") != str(final)
        or value.get("lock_sha256") != lock_sha256
        or value.get("pointer_target") != "releases/%s" % environment_id
    ):
        raise ActivationVerificationError(
            "existing ACTIVE evidence does not match the verified release"
        )
    if value.get("hardlink_policy") != dict(hardlinks):
        raise ActivationVerificationError(
            "existing ACTIVE evidence lacks the current hardlink audit"
        )
    return dict(value)


def _activate_legacy_at_paths(
    candidate_path: Union[str, Path],
    *,
    releases_root: Union[str, Path],
    current_link: Union[str, Path],
    lock_path: Union[str, Path],
    environment_id: str,
    expected_current: Optional[str],
    execute: bool,
    confirm_environment_id: Optional[str],
    confirm_lock_sha256: Optional[str],
    run_verifier: Optional[VerifierRunner] = None,
    reconcile: bool = False,
) -> Dict[str, object]:
    _root, staging, releases, current = _validate_topology(
        candidate_path, releases_root, current_link
    )
    candidate = Path(candidate_path)
    lock, lock_sha256, exact_lock_path = _load_exact_lock(
        lock_path, environment_id
    )
    if expected_current is not None and not _ENVIRONMENT_ID.fullmatch(
        expected_current
    ):
        raise ActivationPolicyError("expected_current is invalid")
    _validate_confirmations(
        execute=execute,
        environment_id=environment_id,
        lock_sha256=lock_sha256,
        confirm_environment_id=confirm_environment_id,
        confirm_lock_sha256=confirm_lock_sha256,
    )
    verify = _run_verifier_subprocess if run_verifier is None else run_verifier
    final = releases / environment_id
    evidence_path = _activation_evidence_path(staging, environment_id)
    releases_fd = _open_real_directory(releases, "perception releases root")
    try:
        staging_fd = _open_real_directory(staging, "perception staging root")
    except BaseException:
        os.close(releases_fd)
        raise
    try:
        fcntl.flock(releases_fd, fcntl.LOCK_EX)
        previous_id, previous_target, previous_identity = _read_current(
            current, releases
        )
        final_exists = os.path.lexists(str(final))
        if final_exists:
            _require_real_directory(final, "published environment")
            (
                marker_sha256, final_report, final_symlinks,
                immutable_entries, final_hardlinks,
            ) = (
                _verify_immutable_final(
                    final,
                    lock=lock,
                    lock_sha256=lock_sha256,
                    environment_id=environment_id,
                    lock_path=exact_lock_path,
                    run_verifier=verify,
                )
            )
            if previous_id == environment_id:
                if not _same_pointer_state(
                    current,
                    releases,
                    previous_id,
                    previous_target,
                    previous_identity,
                ):
                    raise ActivationConflictError(
                        "active pointer changed during idempotent verification"
                    )
                if not execute:
                    return {
                        "activation_evidence_path": str(evidence_path),
                        "environment_id": environment_id,
                        "environment_root": str(final),
                        "hardlink_policy": final_hardlinks,
                        "lock_sha256": lock_sha256,
                        "marker_sha256": marker_sha256,
                        "mode": "dry-run",
                        "pointer_target": "releases/%s" % environment_id,
                        "previous_environment_id": previous_id,
                        "schema": ACTIVATION_EVIDENCE_SCHEMA,
                        "status": "VERIFIED_DRY_RUN",
                        "symlink_policy": {
                            "internal_relative_symlinks": final_symlinks,
                            "status": "VERIFIED_NO_DANGLING_OR_ESCAPE",
                        },
                        "verification": {
                            "active_sha256": _canonical_sha256(final_report),
                        },
                    }
                existing = _load_active_evidence(evidence_path)
                if existing is not None:
                    return _validate_existing_active_evidence(
                        existing,
                        evidence_path=evidence_path,
                        final=final,
                        environment_id=environment_id,
                        lock_sha256=lock_sha256,
                        hardlinks=final_hardlinks,
                    )
                active = _active_evidence(
                    evidence_path=evidence_path,
                    environment_id=environment_id,
                    final=final,
                    lock_sha256=lock_sha256,
                    marker_sha256=marker_sha256,
                    mode="idempotent",
                    previous_id=(
                        expected_current
                        if expected_current != environment_id else None
                    ),
                    symlinks=final_symlinks,
                    immutable_entries=immutable_entries,
                    hardlinks=final_hardlinks,
                    verification={
                        "active_sha256": _canonical_sha256(final_report),
                    },
                )
                _write_activation_evidence(evidence_path, active)
                return active
            if not reconcile:
                raise ActivationConflictError(
                    "verified final exists while current differs; explicit reconcile is required"
                )
            if not execute:
                raise ActivationPolicyError("reconcile requires execute mode")
            if previous_id != expected_current:
                raise ActivationConflictError(
                    "production compare-and-swap mismatch: expected=%r actual=%r"
                    % (expected_current, previous_id)
                )
            if _load_active_evidence(evidence_path) is not None:
                raise ActivationConflictError(
                    "ACTIVE evidence conflicts with the non-active pointer"
                )
            active = _active_evidence(
                evidence_path=evidence_path,
                environment_id=environment_id,
                final=final,
                lock_sha256=lock_sha256,
                marker_sha256=marker_sha256,
                mode="reconcile",
                previous_id=previous_id,
                symlinks=final_symlinks,
                immutable_entries=immutable_entries,
                hardlinks=final_hardlinks,
                verification={
                    "post_immutable_sha256": _canonical_sha256(final_report),
                },
            )
            _persist_with_pointer_cas(
                active_evidence=active,
                evidence_path=evidence_path,
                staging=staging,
                releases=releases,
                current=current,
                environment_id=environment_id,
                previous_id=previous_id,
                previous_target=previous_target,
                previous_identity=previous_identity,
            )
            return active

        if reconcile:
            raise ActivationConflictError(
                "reconcile requires an existing verified final release"
            )
        if previous_id != expected_current:
            raise ActivationConflictError(
                "production compare-and-swap mismatch: expected=%r actual=%r"
                % (expected_current, previous_id)
            )
        _require_real_directory(candidate, "staged environment")
        try:
            marker, staged_marker_sha256 = verifier.load_clone_marker(
                candidate / verifier.CLONE_MARKER_NAME
            )
        except verifier.EnvironmentVerificationError as exc:
            raise ActivationVerificationError("staged clone marker is invalid") from exc
        _validate_marker(
            marker,
            candidate=candidate,
            environment_id=environment_id,
            lock=lock,
            lock_sha256=lock_sha256,
        )
        _validate_tree_symlinks(candidate)
        _validate_tree_hardlinks(candidate)
        candidate_identity = _directory_identity(
            candidate, "staged environment"
        )
        staged_report = _validate_verification_report(
            verify(candidate, exact_lock_path),
            environment_root=candidate,
            environment_id=environment_id,
            lock_sha256=lock_sha256,
        )
        if _directory_identity(candidate, "staged environment") != candidate_identity:
            raise ActivationConflictError(
                "staged environment inode changed after verification"
            )
        staged_symlinks = _validate_tree_symlinks(candidate)
        staged_hardlinks = _validate_tree_hardlinks(candidate)
        pointer_target = "releases/%s" % environment_id
        if not execute:
            return {
                "activation_evidence_path": str(evidence_path),
                "environment_id": environment_id,
                "environment_root": str(final),
                "hardlink_policy": staged_hardlinks,
                "lock_sha256": lock_sha256,
                "marker_sha256": staged_marker_sha256,
                "mode": "dry-run",
                "pointer_target": pointer_target,
                "previous_environment_id": previous_id,
                "schema": ACTIVATION_EVIDENCE_SCHEMA,
                "status": "VERIFIED_DRY_RUN",
                "symlink_policy": {
                    "internal_relative_symlinks": staged_symlinks,
                    "status": "VERIFIED_NO_DANGLING_OR_ESCAPE",
                },
                "verification": {
                    "staged_sha256": _canonical_sha256(staged_report),
                },
            }

        _rename_noreplace_between(
            staging_fd, candidate.name, releases_fd, environment_id
        )
        _fsync_fd(staging_fd)
        _fsync_fd(releases_fd)
        if _directory_identity(final, "published environment") != candidate_identity:
            raise ActivationConflictError(
                "published environment inode changed during no-replace move"
            )
        _rebound_marker, rebound_marker_sha256 = _rewrite_marker_root(
            final,
            marker=marker,
            environment_id=environment_id,
            lock=lock,
            lock_sha256=lock_sha256,
        )
        _validate_tree_symlinks(final)
        _validate_tree_hardlinks(final)
        post_move_report = _validate_verification_report(
            verify(final, exact_lock_path),
            environment_root=final,
            environment_id=environment_id,
            lock_sha256=lock_sha256,
        )
        _validate_tree_symlinks(final)
        _validate_tree_hardlinks(final)
        _make_tree_immutable(final)
        contract._fsync_directory(final)
        contract._fsync_directory(releases)
        post_immutable_report = _validate_verification_report(
            verify(final, exact_lock_path),
            environment_root=final,
            environment_id=environment_id,
            lock_sha256=lock_sha256,
        )
        final_symlinks = _validate_tree_symlinks(final)
        final_hardlinks = _validate_tree_hardlinks(final)
        immutable_entries = _verify_tree_immutable(final)
        if _directory_identity(final, "published environment") != candidate_identity:
            raise ActivationConflictError(
                "published environment inode changed before activation"
            )
        if not _same_pointer_state(
            current,
            releases,
            previous_id,
            previous_target,
            previous_identity,
        ):
            raise ActivationConflictError(
                "production current changed before compare-and-swap"
            )
        evidence = _active_evidence(
            evidence_path=evidence_path,
            environment_id=environment_id,
            final=final,
            lock_sha256=lock_sha256,
            marker_sha256=rebound_marker_sha256,
            mode="execute",
            previous_id=previous_id,
            symlinks=final_symlinks,
            immutable_entries=immutable_entries,
            hardlinks=final_hardlinks,
            verification={
                "post_immutable_sha256": _canonical_sha256(
                    post_immutable_report
                ),
                "post_move_sha256": _canonical_sha256(post_move_report),
                "staged_sha256": _canonical_sha256(staged_report),
            },
        )
        _persist_with_pointer_cas(
            active_evidence=evidence,
            evidence_path=evidence_path,
            staging=staging,
            releases=releases,
            current=current,
            environment_id=environment_id,
            previous_id=previous_id,
            previous_target=previous_target,
            previous_identity=previous_identity,
        )
        return evidence
    finally:
        try:
            fcntl.flock(releases_fd, fcntl.LOCK_UN)
        finally:
            os.close(staging_fd)
            os.close(releases_fd)


def _v3_dry_run_evidence(
    *, evidence_path: Path, environment_id: str, final: Path,
    lock_sha256: str, marker_sha256: str, previous_id: Optional[str],
    symlinks: int, hardlinks: Mapping[str, object], report: Mapping[str, object],
) -> dict:
    return {
        "activation_evidence_path": str(evidence_path),
        "environment_id": environment_id,
        "environment_root": str(final),
        "hardlink_policy": dict(hardlinks),
        "lock_sha256": lock_sha256,
        "marker_sha256": marker_sha256,
        "mode": "dry-run-v3-final-path",
        "pointer_target": "releases/%s" % environment_id,
        "previous_environment_id": previous_id,
        "schema": ACTIVATION_EVIDENCE_SCHEMA,
        "status": "VERIFIED_DRY_RUN",
        "symlink_policy": {
            "internal_relative_symlinks": symlinks,
            "status": "VERIFIED_NO_DANGLING_OR_ESCAPE",
        },
        "verification": {"final_sha256": _canonical_sha256(report)},
    }


def _activate_final_path_at_paths(
    candidate_path: Union[str, Path],
    *,
    releases_root: Union[str, Path],
    current_link: Union[str, Path],
    lock_path: Union[str, Path],
    environment_id: str,
    expected_current: Optional[str],
    execute: bool,
    confirm_environment_id: Optional[str],
    confirm_lock_sha256: Optional[str],
    rollback_lock_path: Optional[Union[str, Path]] = None,
    confirm_rollback_environment_id: Optional[str] = None,
    confirm_rollback_lock_sha256: Optional[str] = None,
    run_verifier: Optional[VerifierRunner] = None,
    reconcile: bool = False,
) -> Dict[str, object]:
    """Verify an already-complete v3 release, then change only ``current``."""

    _root, staging, releases, current = _validate_final_path_topology(
        candidate_path, releases_root, current_link, environment_id
    )
    final = Path(candidate_path)
    lock, lock_sha256, exact_lock_path = _load_exact_lock(
        lock_path, environment_id
    )
    if not publisher.is_relocation_aware_environment_lock(lock):
        raise ActivationVerificationError(
            "final-path activation requires the exact v3 lock"
        )
    if expected_current is not None and not _ENVIRONMENT_ID.fullmatch(
        expected_current
    ):
        raise ActivationPolicyError("expected_current is invalid")
    _validate_confirmations(
        execute=execute,
        environment_id=environment_id,
        lock_sha256=lock_sha256,
        confirm_environment_id=confirm_environment_id,
        confirm_lock_sha256=confirm_lock_sha256,
    )
    marker_path = final / verifier.CLONE_MARKER_NAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ActivationVerificationError(
            "v3 completion marker is missing from the final release"
        )
    try:
        marker, _marker_sha256 = verifier.load_clone_marker(marker_path)
    except verifier.EnvironmentVerificationError as exc:
        raise ActivationVerificationError(
            "v3 completion marker is invalid"
        ) from exc
    if marker.get("schema") != verifier.RELOCATION_AWARE_CLONE_MARKER_SCHEMA:
        raise ActivationVerificationError(
            "v3 completion marker schema is required"
        )

    verify = _run_verifier_subprocess if run_verifier is None else run_verifier
    evidence_path = _activation_evidence_path(staging, environment_id)
    releases_fd = _open_real_directory(releases, "perception releases root")
    try:
        fcntl.flock(releases_fd, fcntl.LOCK_EX)
        previous_id, previous_target, previous_identity = _read_current(
            current, releases
        )
        final_identity = _directory_identity(final, "final-path environment")
        (
            marker_sha256, report, symlinks, immutable_entries, hardlinks,
        ) = _verify_immutable_final(
            final,
            lock=lock,
            lock_sha256=lock_sha256,
            environment_id=environment_id,
            lock_path=exact_lock_path,
            run_verifier=verify,
        )
        if _directory_identity(final, "final-path environment") != final_identity:
            raise ActivationConflictError(
                "final-path environment inode changed during verification"
            )
        if not _same_pointer_state(
            current,
            releases,
            previous_id,
            previous_target,
            previous_identity,
        ):
            raise ActivationConflictError(
                "production current changed during final-path verification"
            )

        if previous_id == environment_id:
            if not execute:
                return _v3_dry_run_evidence(
                    evidence_path=evidence_path,
                    environment_id=environment_id,
                    final=final,
                    lock_sha256=lock_sha256,
                    marker_sha256=marker_sha256,
                    previous_id=previous_id,
                    symlinks=symlinks,
                    hardlinks=hardlinks,
                    report=report,
                )
            existing = _load_active_evidence(evidence_path)
            if existing is not None:
                return _validate_existing_active_evidence(
                    existing,
                    evidence_path=evidence_path,
                    final=final,
                    environment_id=environment_id,
                    lock_sha256=lock_sha256,
                    hardlinks=hardlinks,
                )
            active = _active_evidence(
                evidence_path=evidence_path,
                environment_id=environment_id,
                final=final,
                lock_sha256=lock_sha256,
                marker_sha256=marker_sha256,
                mode="idempotent-v3-final-path",
                previous_id=None,
                symlinks=symlinks,
                immutable_entries=immutable_entries,
                hardlinks=hardlinks,
                verification={"final_sha256": _canonical_sha256(report)},
            )
            _write_activation_evidence(evidence_path, active)
            return active

        if previous_id != expected_current:
            raise ActivationConflictError(
                "production compare-and-swap mismatch: expected=%r actual=%r"
                % (expected_current, previous_id)
            )
        rollback_target = None
        if rollback_lock_path is not None:
            rollback_target = _verify_rollback_target(
                releases=releases,
                lock_path=rollback_lock_path,
                activated_environment_id=environment_id,
                execute=execute,
                confirm_environment_id=confirm_rollback_environment_id,
                confirm_lock_sha256=confirm_rollback_lock_sha256,
                run_verifier=verify,
            )
            if not _same_pointer_state(
                current,
                releases,
                previous_id,
                previous_target,
                previous_identity,
            ):
                raise ActivationConflictError(
                    "production current changed during rollback-target verification"
                )
        elif (
            confirm_rollback_environment_id is not None
            or confirm_rollback_lock_sha256 is not None
        ):
            raise ActivationPolicyError(
                "rollback-target confirmations require a rollback lock"
            )
        if not execute:
            dry_run = _v3_dry_run_evidence(
                evidence_path=evidence_path,
                environment_id=environment_id,
                final=final,
                lock_sha256=lock_sha256,
                marker_sha256=marker_sha256,
                previous_id=previous_id,
                symlinks=symlinks,
                hardlinks=hardlinks,
                report=report,
            )
            dry_run["rollback_target"] = rollback_target
            return dry_run
        if _load_active_evidence(evidence_path) is not None:
            raise ActivationConflictError(
                "ACTIVE evidence exists while the v3 release is not active"
            )
        active = _active_evidence(
            evidence_path=evidence_path,
            environment_id=environment_id,
            final=final,
            lock_sha256=lock_sha256,
            marker_sha256=marker_sha256,
            mode=(
                "reconcile-v3-final-path"
                if reconcile else "execute-v3-final-path"
            ),
            previous_id=previous_id,
            symlinks=symlinks,
            immutable_entries=immutable_entries,
            hardlinks=hardlinks,
            verification={"final_sha256": _canonical_sha256(report)},
        )
        active["rollback_target"] = rollback_target
        _persist_with_pointer_cas(
            active_evidence=active,
            evidence_path=evidence_path,
            staging=staging,
            releases=releases,
            current=current,
            environment_id=environment_id,
            previous_id=previous_id,
            previous_target=previous_target,
            previous_identity=previous_identity,
        )
        return active
    finally:
        try:
            fcntl.flock(releases_fd, fcntl.LOCK_UN)
        finally:
            os.close(releases_fd)


def _rollback_at_paths(
    *,
    releases_root: Union[str, Path],
    current_link: Union[str, Path],
    activated_environment_id: str,
    previous_lock_path: Union[str, Path],
    execute: bool,
    confirm_activation_evidence_sha256: Optional[str],
    confirm_previous_environment_id: Optional[str],
    confirm_previous_lock_sha256: Optional[str],
    run_verifier: Optional[VerifierRunner] = None,
) -> Dict[str, object]:
    """CAS-rollback only the exact pointer transition recorded by ACTIVE evidence."""

    if (
        not isinstance(activated_environment_id, str)
        or not _ENVIRONMENT_ID.fullmatch(activated_environment_id)
    ):
        raise ActivationPolicyError("activated environment_id is invalid")
    releases = _normalized_absolute(releases_root, "releases root")
    current = _normalized_absolute(current_link, "current link")
    activated = releases / activated_environment_id
    _root, staging, releases, current = _validate_final_path_topology(
        activated, releases, current, activated_environment_id
    )
    evidence_path = _activation_evidence_path(
        staging, activated_environment_id
    )
    active, active_sha256 = _load_evidence(evidence_path)
    if (
        active.get("schema") != ACTIVATION_EVIDENCE_SCHEMA
        or active.get("status") != "ACTIVE"
        or active.get("mode") not in {
            "execute-v3-final-path", "reconcile-v3-final-path",
        }
        or active.get("activation_evidence_path") != str(evidence_path)
        or active.get("environment_id") != activated_environment_id
        or active.get("environment_root") != str(activated)
        or active.get("pointer_target")
        != "releases/%s" % activated_environment_id
    ):
        raise ActivationVerificationError(
            "rollback requires the exact just-created v3 ACTIVE evidence"
        )
    active_id, active_target, active_identity = _parse_pointer_state_evidence(
        active.get("active_pointer"), label="ACTIVE pointer evidence"
    )
    previous_id, previous_target, _previous_identity = (
        _parse_pointer_state_evidence(
            active.get("previous_pointer"), label="previous pointer evidence"
        )
    )
    rollback_target = active.get("rollback_target")
    if rollback_target is None:
        rollback_id = previous_id
    else:
        if (
            not isinstance(rollback_target, Mapping)
            or not isinstance(rollback_target.get("environment_id"), str)
            or not _ENVIRONMENT_ID.fullmatch(
                rollback_target["environment_id"]
            )
            or rollback_target.get("environment_root")
            != str(releases / rollback_target["environment_id"])
            or not isinstance(rollback_target.get("lock_path"), str)
            or _normalized_absolute(
                previous_lock_path, "previous environment lock"
            )
            != Path(rollback_target["lock_path"])
        ):
            raise ActivationVerificationError(
                "recorded rollback target is invalid"
            )
        rollback_id = rollback_target["environment_id"]
    if (
        active_id != activated_environment_id
        or active.get("previous_environment_id") != previous_id
    ):
        raise ActivationVerificationError(
            "activation pointer evidence does not match its transition"
        )
    if (
        execute
        and (
            confirm_activation_evidence_sha256 != active_sha256
            or confirm_previous_environment_id != rollback_id
        )
    ):
        raise ActivationPolicyError(
            "rollback activation-evidence/environment confirmations do not match"
        )

    verify = _run_verifier_subprocess if run_verifier is None else run_verifier
    releases_fd = _open_real_directory(releases, "perception releases root")
    try:
        fcntl.flock(releases_fd, fcntl.LOCK_EX)
        if not _same_pointer_state(
            current, releases, active_id, active_target, active_identity
        ):
            raise ActivationConflictError(
                "production current no longer has the recorded active pointer identity"
            )
        current_evidence, current_evidence_sha256 = _load_evidence(
            evidence_path
        )
        if (
            current_evidence_sha256 != active_sha256
            or current_evidence != active
        ):
            raise ActivationConflictError(
                "ACTIVE evidence changed before rollback verification"
            )

        previous_lock, previous_lock_sha256, exact_previous_lock_path = (
            _load_exact_lock(previous_lock_path, rollback_id)
        )
        if (
            rollback_target is not None
            and (
                rollback_target.get("lock_sha256") != previous_lock_sha256
                or rollback_target.get("lock_path")
                != str(exact_previous_lock_path)
            )
        ):
            raise ActivationVerificationError(
                "rollback lock no longer matches ACTIVE evidence"
            )
        if (
            execute
            and confirm_previous_lock_sha256 != previous_lock_sha256
        ):
            raise ActivationPolicyError(
                "rollback previous-lock confirmation does not match"
            )
        previous = releases / rollback_id
        previous_identity = _directory_identity(
            previous, "previous published environment"
        )
        (
            previous_marker_sha256,
            previous_report,
            previous_symlinks,
            previous_immutable_entries,
            previous_hardlinks,
        ) = _verify_immutable_final(
            previous,
            lock=previous_lock,
            lock_sha256=previous_lock_sha256,
            environment_id=rollback_id,
            lock_path=exact_previous_lock_path,
            run_verifier=verify,
        )
        if _directory_identity(
            previous, "previous published environment"
        ) != previous_identity:
            raise ActivationConflictError(
                "previous environment inode changed during rollback verification"
            )
        if not _same_pointer_state(
            current, releases, active_id, active_target, active_identity
        ):
            raise ActivationConflictError(
                "production current changed during rollback verification"
            )
        final_evidence, final_evidence_sha256 = _load_evidence(evidence_path)
        if final_evidence_sha256 != active_sha256 or final_evidence != active:
            raise ActivationConflictError(
                "ACTIVE evidence changed during rollback verification"
            )

        rollback_base = {
            "activation": dict(active),
            "activation_evidence_path": str(evidence_path),
            "activation_evidence_sha256": active_sha256,
            "activated_environment_id": activated_environment_id,
            "previous_environment_id": rollback_id,
            "previous_environment_root": str(previous),
            "previous_hardlink_policy": previous_hardlinks,
            "previous_immutability": {
                "non_symlink_entries": previous_immutable_entries,
                "status": "VERIFIED_A_W_REMOVED",
            },
            "previous_lock_sha256": previous_lock_sha256,
            "previous_marker_sha256": previous_marker_sha256,
            "previous_pointer": dict(active["previous_pointer"]),
            "previous_symlink_policy": {
                "internal_relative_symlinks": previous_symlinks,
                "status": "VERIFIED_NO_DANGLING_OR_ESCAPE",
            },
            "schema": ROLLBACK_EVIDENCE_SCHEMA,
            "verification": {
                "previous_sha256": _canonical_sha256(previous_report)
            },
        }
        if not execute:
            return {
                **rollback_base,
                "restored_pointer": None,
                "status": "VERIFIED_ROLLBACK_DRY_RUN",
            }

        result: Dict[str, object] = {}

        def persist_rollback() -> None:
            restored_id, restored_target, restored_identity = _read_current(
                current, releases
            )
            if restored_id != rollback_id:
                raise ActivationConflictError(
                    "rollback target changed before evidence persistence"
                )
            rolled_back = {
                **rollback_base,
                "restored_pointer": _pointer_state_evidence(
                    restored_id, restored_target, restored_identity
                ),
                "status": "ROLLED_BACK",
            }
            _write_activation_evidence(evidence_path, rolled_back)
            result.update(rolled_back)

        _cas_pointer(
            staging=staging,
            releases=releases,
            current=current,
            environment_id=rollback_id,
            previous_id=active_id,
            previous_target=active_target,
            previous_identity=active_identity,
            post_switch=persist_rollback,
        )
        if not result:
            raise ActivationVerificationError(
                "rollback evidence was not persisted"
            )
        return result
    finally:
        try:
            fcntl.flock(releases_fd, fcntl.LOCK_UN)
        finally:
            os.close(releases_fd)


def _activate_at_paths(
    candidate_path: Union[str, Path],
    *,
    releases_root: Union[str, Path],
    current_link: Union[str, Path],
    lock_path: Union[str, Path],
    environment_id: str,
    expected_current: Optional[str],
    execute: bool,
    confirm_environment_id: Optional[str],
    confirm_lock_sha256: Optional[str],
    rollback_lock_path: Optional[Union[str, Path]] = None,
    confirm_rollback_environment_id: Optional[str] = None,
    confirm_rollback_lock_sha256: Optional[str] = None,
    run_verifier: Optional[VerifierRunner] = None,
    reconcile: bool = False,
) -> Dict[str, object]:
    """Dispatch v3 final-path activation without weakening legacy v2."""

    lock, _lock_sha256, _exact_lock_path = _load_exact_lock(
        lock_path, environment_id
    )
    implementation = (
        _activate_final_path_at_paths
        if lock.get("schema") == publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA
        else _activate_legacy_at_paths
    )
    arguments = dict(
        releases_root=releases_root,
        current_link=current_link,
        lock_path=lock_path,
        environment_id=environment_id,
        expected_current=expected_current,
        execute=execute,
        confirm_environment_id=confirm_environment_id,
        confirm_lock_sha256=confirm_lock_sha256,
        run_verifier=run_verifier,
        reconcile=reconcile,
    )
    if implementation is _activate_final_path_at_paths:
        arguments.update(
            rollback_lock_path=rollback_lock_path,
            confirm_rollback_environment_id=confirm_rollback_environment_id,
            confirm_rollback_lock_sha256=confirm_rollback_lock_sha256,
        )
    elif (
        rollback_lock_path is not None
        or confirm_rollback_environment_id is not None
        or confirm_rollback_lock_sha256 is not None
    ):
        raise ActivationPolicyError(
            "rollback-target binding is available only for v3 activation"
        )
    return implementation(
        candidate_path,
        **arguments,
    )


def activate_perception_environment(
    staged_name: Optional[str],
    *,
    lock_path: Union[str, Path] = DEFAULT_LOCK,
    environment_id: str,
    expected_current: Optional[str],
    execute: bool,
    confirm_environment_id: Optional[str] = None,
    confirm_lock_sha256: Optional[str] = None,
    reconcile: bool = False,
    rollback_lock_path: Optional[Union[str, Path]] = None,
    confirm_rollback_environment_id: Optional[str] = None,
    confirm_rollback_lock_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Operate only on the fixed Linbox perception environment namespace."""
    lock, _lock_sha256, _exact_lock_path = _load_exact_lock(
        lock_path, environment_id
    )
    if lock.get("schema") == publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA:
        if staged_name is not None:
            raise ActivationPolicyError(
                "v3 activation does not accept a staged environment"
            )
        if execute and expected_current is not None and rollback_lock_path is None:
            raise ActivationPolicyError(
                "v3 production activation requires a preverified rollback lock"
            )
        candidate = LINBOX_RELEASES_ROOT / environment_id
    else:
        if (
            not isinstance(staged_name, str)
            or not _STAGING_NAME.fullmatch(staged_name)
        ):
            raise ActivationPolicyError("staged_name is required for legacy v2")
        candidate = LINBOX_STAGING_ROOT / staged_name
    return _activate_at_paths(
        candidate,
        releases_root=LINBOX_RELEASES_ROOT,
        current_link=LINBOX_CURRENT_LINK,
        lock_path=lock_path,
        environment_id=environment_id,
        expected_current=expected_current,
        execute=execute,
        confirm_environment_id=confirm_environment_id,
        confirm_lock_sha256=confirm_lock_sha256,
        reconcile=reconcile,
        rollback_lock_path=rollback_lock_path,
        confirm_rollback_environment_id=confirm_rollback_environment_id,
        confirm_rollback_lock_sha256=confirm_rollback_lock_sha256,
    )


def rollback_perception_environment(
    *,
    activated_environment_id: str,
    previous_lock_path: Union[str, Path],
    execute: bool,
    confirm_activation_evidence_sha256: Optional[str] = None,
    confirm_previous_environment_id: Optional[str] = None,
    confirm_previous_lock_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Rollback one fixed Linbox activation transition, never an arbitrary ID."""

    return _rollback_at_paths(
        releases_root=LINBOX_RELEASES_ROOT,
        current_link=LINBOX_CURRENT_LINK,
        activated_environment_id=activated_environment_id,
        previous_lock_path=previous_lock_path,
        execute=execute,
        confirm_activation_evidence_sha256=(
            confirm_activation_evidence_sha256
        ),
        confirm_previous_environment_id=confirm_previous_environment_id,
        confirm_previous_lock_sha256=confirm_previous_lock_sha256,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and CAS-activate a Linbox perception environment"
        )
    )
    parser.add_argument(
        "mode",
        choices=(
            "dry-run", "execute", "reconcile",
            "rollback-dry-run", "rollback-execute",
        ),
    )
    parser.add_argument(
        "--staged-name",
        help="legacy v2 staging name; forbidden for permanent-path v3 locks",
    )
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument(
        "--expected-current",
        help="expected current environment id; omit only when current must be absent",
    )
    parser.add_argument("--confirm-environment-id")
    parser.add_argument("--confirm-lock-sha256")
    parser.add_argument(
        "--rollback-lock",
        help="preverified strong rollback-target lock for v3 activation",
    )
    parser.add_argument("--confirm-rollback-environment-id")
    parser.add_argument("--confirm-rollback-lock-sha256")
    parser.add_argument(
        "--previous-lock",
        help="exact strong previous-release lock; required for rollback",
    )
    parser.add_argument("--confirm-activation-evidence-sha256")
    parser.add_argument("--confirm-previous-environment-id")
    parser.add_argument("--confirm-previous-lock-sha256")
    return parser


def _emit(stream, value: object) -> None:
    stream.buffer.write(contract.canonical_json_bytes(value))
    stream.flush()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        if args.mode.startswith("rollback-"):
            if args.staged_name is not None:
                raise ActivationPolicyError(
                    "rollback does not accept a staged environment"
                )
            if args.previous_lock is None:
                raise ActivationPolicyError("rollback requires --previous-lock")
            evidence = rollback_perception_environment(
                activated_environment_id=args.environment_id,
                previous_lock_path=args.previous_lock,
                execute=args.mode == "rollback-execute",
                confirm_activation_evidence_sha256=(
                    args.confirm_activation_evidence_sha256
                ),
                confirm_previous_environment_id=(
                    args.confirm_previous_environment_id
                ),
                confirm_previous_lock_sha256=args.confirm_previous_lock_sha256,
            )
        else:
            evidence = activate_perception_environment(
                args.staged_name,
                lock_path=args.lock,
                environment_id=args.environment_id,
                expected_current=args.expected_current,
                execute=args.mode in ("execute", "reconcile"),
                confirm_environment_id=args.confirm_environment_id,
                confirm_lock_sha256=args.confirm_lock_sha256,
                reconcile=args.mode == "reconcile",
                rollback_lock_path=args.rollback_lock,
                confirm_rollback_environment_id=(
                    args.confirm_rollback_environment_id
                ),
                confirm_rollback_lock_sha256=(
                    args.confirm_rollback_lock_sha256
                ),
            )
    except ActivationError as exc:
        _emit(
            sys.stderr,
            {
                "error": str(exc),
                "schema": ACTIVATION_EVIDENCE_SCHEMA,
                "status": "FAILED",
            },
        )
        return 1
    except Exception as exc:  # pragma: no cover - canonical fail-closed boundary
        _emit(
            sys.stderr,
            {
                "error": "unexpected activation failure: %s" % type(exc).__name__,
                "schema": ACTIVATION_EVIDENCE_SCHEMA,
                "status": "FAILED",
            },
        )
        return 1
    _emit(sys.stdout, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
