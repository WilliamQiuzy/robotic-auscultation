"""Fail-closed publication for isolated Linbox simulation evidence.

The generic :mod:`artifact_contract` owns content identity, exact-set
verification, completion-marker binding, atomic no-replace publication, CAS,
and rollback.  This module narrows those primitives to the three audited
simulation namespaces; it never accepts or mutates a production pointer.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Callable, Dict, Iterable, Mapping, Optional, Union
import uuid

import artifact_contract as contract


PROFILE_PATH = Path(__file__).resolve().with_name("release_profile.json")
AUDITED_SIMULATION_POLICY = {
    "assets": {
        "artifact_kind": "simulation_asset",
        "current_link": "/home/robot/wam_data/simulation/assets/current",
        "releases_root": "/home/robot/wam_data/simulation/assets/releases",
        "staging_root": "/home/robot/wam_data/simulation/assets/.staging",
    },
    "code": {
        "artifact_kind": "simulation_code",
        "current_link": "/home/robot/wam_data/simulation/code/current",
        "releases_root": "/home/robot/wam_data/simulation/code/releases",
        "staging_root": "/home/robot/wam_data/simulation/code/.staging",
    },
    "results": {
        "artifact_kind": "simulation_result",
        "current_link": "/home/robot/wam_data/simulation/results/current",
        "releases_root": "/home/robot/wam_data/simulation/results/releases",
        "staging_root": "/home/robot/wam_data/simulation/results/.staging",
    },
}
_ARTIFACT_KINDS = {
    "assets": "simulation_asset",
    "code": "simulation_code",
    "results": "simulation_result",
}
_SOURCE_ROLES = {"assets": "linbox", "code": "mac", "results": "linbox"}
_NONCE = re.compile(r"^[0-9a-f]{32}$")
_STAGING_NAME = re.compile(
    r"^incoming-(simulation_(?:asset|code|result)-[0-9a-f]{20})-([0-9a-f]{32})$"
)


class SimulationPolicyError(contract.ArtifactError):
    """A request escapes the audited Linbox simulation policy."""


def _canonical_absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise SimulationPolicyError("%s is not a canonical absolute path" % label)
    candidate = PurePosixPath(value)
    if (
        not candidate.is_absolute()
        or any(part in ("", ".", "..") for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise SimulationPolicyError("%s is not a canonical absolute path" % label)
    return Path(value)


def _validate_policy_shape(policy: object) -> Dict[str, Dict[str, str]]:
    if not isinstance(policy, dict) or set(policy) != set(_ARTIFACT_KINDS):
        raise SimulationPolicyError("simulation policy namespaces are invalid")
    checked: Dict[str, Dict[str, str]] = {}
    for namespace in ("assets", "code", "results"):
        value = policy.get(namespace)
        if not isinstance(value, dict) or set(value) != {
            "artifact_kind", "current_link", "releases_root", "staging_root",
        }:
            raise SimulationPolicyError(
                "simulation %s policy fields are invalid" % namespace
            )
        if value.get("artifact_kind") != _ARTIFACT_KINDS[namespace]:
            raise SimulationPolicyError(
                "simulation %s artifact kind is invalid" % namespace
            )
        current = _canonical_absolute_path(
            value.get("current_link"), "%s current link" % namespace,
        )
        releases = _canonical_absolute_path(
            value.get("releases_root"), "%s releases root" % namespace,
        )
        staging = _canonical_absolute_path(
            value.get("staging_root"), "%s staging root" % namespace,
        )
        if (
            current.name != "current"
            or releases.name != "releases"
            or staging.name != ".staging"
            or current.parent != releases.parent
            or current.parent != staging.parent
            or current.parent.name != namespace
        ):
            raise SimulationPolicyError(
                "audited simulation %s roots are not an isolated namespace" % namespace
            )
        checked[namespace] = dict(value)
    return checked


def load_simulation_policy(
    profile_path: Union[str, Path] = PROFILE_PATH,
) -> Dict[str, Dict[str, str]]:
    """Load only the exact checked-in Linbox simulation root policy."""
    try:
        value = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        policy = value["hosts"]["linbox"]["simulation"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SimulationPolicyError("release profile has no simulation policy") from exc
    checked = _validate_policy_shape(policy)
    audited = _validate_policy_shape(AUDITED_SIMULATION_POLICY)
    if checked != audited:
        raise SimulationPolicyError("simulation roots differ from the audited policy")
    return checked


def _open_real_directory(path: Path, *, create: bool) -> int:
    """Open an absolute directory through a no-follow descriptor walk."""
    _canonical_absolute_path(path.as_posix(), "directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise SimulationPolicyError("audited directory is missing")
                try:
                    os.mkdir(part, mode=0o750, dir_fd=descriptor)
                    os.fsync(descriptor)
                    child = os.open(part, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise SimulationPolicyError(
                        "audited directory could not be created safely"
                    ) from exc
            except OSError as exc:
                if exc.errno in (
                    errno.ELOOP, errno.EMLINK, errno.ENOTDIR,
                ):
                    raise SimulationPolicyError(
                        "audited directory has a symlink or non-directory ancestor"
                    ) from exc
                raise
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                raise SimulationPolicyError(
                    "audited directory has a symlink or non-directory ancestor"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _ensure_real_directory(path: Path) -> None:
    descriptor = _open_real_directory(path, create=True)
    os.close(descriptor)


def _open_child_directory(
    parent_descriptor: int, name: str, *, create: bool,
) -> int:
    try:
        if create:
            try:
                os.mkdir(name, mode=0o750, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except FileExistsError:
                pass
        return contract._open_directory_at(parent_descriptor, name)
    except (contract.ArtifactError, OSError) as exc:
        raise SimulationPolicyError(
            "audited directory has a symlink or non-directory child"
        ) from exc


def _directory_identity(descriptor: int) -> tuple:
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        raise SimulationPolicyError("pinned simulation descriptor is not a directory")
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))


def _canonical_identity_matches(path: Path, descriptor: int) -> bool:
    try:
        observed = _open_real_directory(path, create=False)
    except (SimulationPolicyError, OSError):
        return False
    try:
        return _directory_identity(observed) == _directory_identity(descriptor)
    finally:
        os.close(observed)


def _identities_intact(entries: Iterable[tuple]) -> bool:
    return all(
        _canonical_identity_matches(path, descriptor)
        for path, descriptor in entries
    )


def _require_identities(entries: Iterable[tuple]) -> None:
    if not _identities_intact(entries):
        raise SimulationPolicyError(
            "audited simulation directory identity changed during transaction"
        )


@contextmanager
def _simulation_lock(namespace_descriptor: int):
    name = ".simulation-deploy.lock"
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=namespace_descriptor,
        )
    except FileExistsError as exc:
        raise contract.ConflictError("simulation deployment lock is already held") from exc
    try:
        data = ("%s\n" % os.getpid()).encode("ascii")
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        os.fsync(namespace_descriptor)
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(name, dir_fd=namespace_descriptor)
            os.fsync(namespace_descriptor)
        except FileNotFoundError:
            pass


def _namespace_policy(
    namespace: str, profile_path: Union[str, Path],
) -> Dict[str, str]:
    policy = load_simulation_policy(profile_path)
    if namespace not in policy:
        raise SimulationPolicyError("unknown simulation namespace")
    return policy[namespace]


def _completion_target_role_fd(release_descriptor: int) -> object:
    try:
        raw = contract._read_regular_bytes_at(
            release_descriptor, contract.COMPLETE_NAME,
        )
        marker = json.loads(raw.decode("utf-8"))
    except (
        contract.ArtifactError, OSError, UnicodeDecodeError, json.JSONDecodeError,
    ) as exc:
        raise contract.VerificationError(
            "release completion marker is missing or invalid"
        ) from exc
    if not isinstance(marker, dict) or raw != contract.canonical_json_bytes(marker):
        raise contract.VerificationError(
            "release completion marker is not canonical JSON"
        )
    return marker.get("target_role")


def _verify_simulation_release_fd(
    release_descriptor: int, *, namespace: str, expected_kind: str,
) -> Dict[str, object]:
    envelope = contract.verify_release_fd(release_descriptor)
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping) or payload.get("artifact_kind") != expected_kind:
        raise contract.VerificationError(
            "release artifact kind does not match the simulation namespace"
        )
    expected_source_role = _SOURCE_ROLES[namespace]
    if payload.get("source_role") != expected_source_role:
        raise contract.VerificationError(
            "release source role does not match the simulation namespace"
        )
    if _completion_target_role_fd(release_descriptor) != "linbox":
        raise contract.VerificationError(
            "simulation completion marker has the wrong target role"
        )
    return envelope


def stage_simulation_release(
    source_root: Union[str, Path],
    relative_paths: Iterable[Union[str, Path]],
    *,
    namespace: str,
    nonce: str,
    profile_path: Union[str, Path] = PROFILE_PATH,
) -> Path:
    """Hash, exact-copy, verify, and complete one isolated staging release."""
    paths = _namespace_policy(namespace, profile_path)
    if not isinstance(nonce, str) or not _NONCE.fullmatch(nonce):
        raise SimulationPolicyError("staging nonce must be 32 lowercase hex characters")
    staging_root = Path(paths["staging_root"])
    namespace_root = Path(paths["current_link"]).parent
    namespace_descriptor = _open_real_directory(namespace_root, create=True)
    staging_descriptor: Optional[int] = None
    source_role = _SOURCE_ROLES[namespace]
    try:
        staging_descriptor = _open_child_directory(
            namespace_descriptor, ".staging", create=True,
        )
        identities = (
            (namespace_root, namespace_descriptor),
            (staging_root, staging_descriptor),
        )
        _require_identities(identities)
        manifest = contract.build_manifest(
            source_root,
            relative_paths,
            artifact_kind=paths["artifact_kind"],
            source_role=source_role,
        )
        destination_name = "incoming-%s-%s" % (
            manifest["release_id"], nonce,
        )
        destination = staging_root / destination_name
        if destination.parent != staging_root:
            raise SimulationPolicyError("staging destination escapes its audited root")
        created = False
        with _simulation_lock(namespace_descriptor):
            _require_identities(identities)
            try:
                contract.write_release_at(
                    source_root,
                    manifest,
                    staging_descriptor,
                    destination_name,
                    target_role="linbox",
                )
                created = True
                release_descriptor = contract._open_directory_at(
                    staging_descriptor, destination_name,
                )
                try:
                    _verify_simulation_release_fd(
                        release_descriptor,
                        namespace=namespace,
                        expected_kind=paths["artifact_kind"],
                    )
                finally:
                    os.close(release_descriptor)
                _require_identities(identities)
            except BaseException:
                if created:
                    contract.remove_release_at(
                        staging_descriptor, destination_name,
                    )
                raise
        return destination
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        os.close(namespace_descriptor)


def _require_staged_direct_child(
    staged: Path, staging_root: Path, release_id: object = None,
) -> None:
    if not staged.is_absolute() or staged.parent != staging_root:
        raise SimulationPolicyError(
            "staged release must be a direct child of the audited staging root"
        )
    if staged.is_symlink():
        raise SimulationPolicyError("staged release must not be a symlink")
    match = _STAGING_NAME.fullmatch(staged.name)
    if match is None or (
        release_id is not None and match.group(1) != release_id
    ):
        raise SimulationPolicyError("staged release name does not bind its release id")


def _safe_expected_target(value: object) -> object:
    if value is None:
        return None
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise SimulationPolicyError("expected current target is invalid")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or len(candidate.parts) != 2
        or candidate.parts[0] != "releases"
        or candidate.parts[1] in ("", ".", "..")
    ):
        raise SimulationPolicyError(
            "expected current target must be a release-relative direct child"
        )
    return value


def _validate_current_target(target: str, expected_kind: str) -> str:
    candidate = PurePosixPath(target)
    release_id = re.compile(
        r"^%s-[0-9a-f]{20}$" % re.escape(expected_kind)
    )
    if (
        candidate.is_absolute()
        or candidate.as_posix() != target
        or len(candidate.parts) != 2
        or candidate.parts[0] != "releases"
        or release_id.fullmatch(candidate.parts[1]) is None
    ):
        raise SimulationPolicyError(
            "simulation current pointer escapes its audited releases root"
        )
    return target


def _read_current_at(
    namespace_descriptor: int, expected_kind: str,
) -> Optional[str]:
    try:
        info = os.stat(
            "current", dir_fd=namespace_descriptor, follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    if not stat.S_ISLNK(info.st_mode):
        raise SimulationPolicyError("simulation current path must be a symlink")
    return _validate_current_target(
        os.readlink("current", dir_fd=namespace_descriptor), expected_kind,
    )


def _read_verified_current_at(
    namespace_descriptor: int,
    releases_descriptor: int,
    *,
    namespace: str,
    expected_kind: str,
) -> tuple:
    """Return a verified prior target and keep its directory FD pinned."""
    release_descriptor: Optional[int] = None
    try:
        target = _read_current_at(namespace_descriptor, expected_kind)
        if target is None:
            return None, None
        release_id = PurePosixPath(target).parts[1]
        release_descriptor = contract._open_directory_at(
            releases_descriptor, release_id,
        )
        envelope = _verify_simulation_release_fd(
            release_descriptor,
            namespace=namespace,
            expected_kind=expected_kind,
        )
        if envelope.get("release_id") != release_id:
            raise contract.VerificationError(
                "previous current release id does not bind its directory"
            )
        _verify_current_matches_descriptor(
            namespace_descriptor,
            target,
            release_descriptor,
        )
        pinned = release_descriptor
        release_descriptor = None
        return target, pinned
    except (
        SimulationPolicyError, contract.ArtifactError, OSError,
    ) as exc:
        if release_descriptor is not None:
            os.close(release_descriptor)
        try:
            _remove_current_at(namespace_descriptor)
        except SimulationPolicyError:
            pass
        raise SimulationPolicyError(
            "previous current is not a verified in-root simulation release"
        ) from exc


def _replace_current_at(namespace_descriptor: int, target: str) -> None:
    temporary = ".current.activate-%s" % uuid.uuid4().hex
    os.symlink(target, temporary, dir_fd=namespace_descriptor)
    try:
        os.replace(
            temporary,
            "current",
            src_dir_fd=namespace_descriptor,
            dst_dir_fd=namespace_descriptor,
        )
        os.fsync(namespace_descriptor)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=namespace_descriptor)
        except FileNotFoundError:
            pass
        raise


def _remove_current_at(namespace_descriptor: int) -> None:
    try:
        info = os.stat(
            "current", dir_fd=namespace_descriptor, follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not stat.S_ISLNK(info.st_mode):
        raise SimulationPolicyError(
            "refusing to remove a non-symlink simulation current path"
        )
    os.unlink("current", dir_fd=namespace_descriptor)
    os.fsync(namespace_descriptor)


def _verify_current_matches_descriptor(
    namespace_descriptor: int,
    expected_target: str,
    expected_release_descriptor: int,
) -> None:
    target = os.readlink("current", dir_fd=namespace_descriptor)
    if target != expected_target:
        raise contract.VerificationError(
            "active simulation pointer changed during verification"
        )
    active_descriptor: Optional[int] = None
    try:
        active_descriptor = os.open(
            "current",
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            dir_fd=namespace_descriptor,
        )
        if _directory_identity(active_descriptor) != _directory_identity(
            expected_release_descriptor
        ):
            raise contract.VerificationError(
                "active simulation pointer resolves outside the pinned release"
            )
    finally:
        if active_descriptor is not None:
            os.close(active_descriptor)


def _verify_active_identity(
    namespace_descriptor: int,
    releases_descriptor: int,
    release_id: str,
    expected_target: str,
) -> None:
    final_descriptor = contract._open_directory_at(
        releases_descriptor, release_id,
    )
    try:
        _verify_current_matches_descriptor(
            namespace_descriptor,
            expected_target,
            final_descriptor,
        )
    finally:
        os.close(final_descriptor)


def _restore_verified_previous_at(
    namespace_descriptor: int,
    releases_descriptor: int,
    previous: Optional[str],
    previous_release_descriptor: Optional[int],
    *,
    namespace: str,
    expected_kind: str,
) -> bool:
    """Restore only while the prior direct child is still the pinned release."""
    if previous is None:
        _remove_current_at(namespace_descriptor)
        return True
    if previous_release_descriptor is None:
        _remove_current_at(namespace_descriptor)
        return False
    release_id = PurePosixPath(previous).parts[1]
    observed: Optional[int] = None
    try:
        observed = contract._open_directory_at(releases_descriptor, release_id)
        if _directory_identity(observed) != _directory_identity(
            previous_release_descriptor
        ):
            raise contract.VerificationError(
                "previous release child identity changed before rollback"
            )
        os.close(observed)
        observed = None
        envelope = _verify_simulation_release_fd(
            previous_release_descriptor,
            namespace=namespace,
            expected_kind=expected_kind,
        )
        if envelope.get("release_id") != release_id:
            raise contract.VerificationError(
                "previous release content no longer binds its rollback target"
            )
        _replace_current_at(namespace_descriptor, previous)
        _verify_current_matches_descriptor(
            namespace_descriptor,
            previous,
            previous_release_descriptor,
        )
        observed = contract._open_directory_at(releases_descriptor, release_id)
        if _directory_identity(observed) != _directory_identity(
            previous_release_descriptor
        ):
            raise contract.VerificationError(
                "previous release child identity changed during rollback"
            )
        _verify_current_matches_descriptor(
            namespace_descriptor,
            previous,
            previous_release_descriptor,
        )
        return True
    except (
        SimulationPolicyError, contract.ArtifactError, OSError,
    ):
        try:
            _remove_current_at(namespace_descriptor)
        except SimulationPolicyError:
            pass
        return False
    finally:
        if observed is not None:
            os.close(observed)


def promote_simulation_release(
    staged_release: Union[str, Path],
    *,
    namespace: str,
    expected_current: object,
    profile_path: Union[str, Path] = PROFILE_PATH,
    post_verify: Optional[Callable[[Path], None]] = None,
) -> Path:
    """CAS-promote staging evidence and rollback its isolated pointer on failure."""
    paths = _namespace_policy(namespace, profile_path)
    staging_root = Path(paths["staging_root"])
    releases_root = Path(paths["releases_root"])
    current_link = Path(paths["current_link"])
    staged = Path(staged_release)
    _require_staged_direct_child(staged, staging_root)
    expectation = _safe_expected_target(expected_current)
    namespace_root = current_link.parent
    namespace_descriptor = _open_real_directory(namespace_root, create=True)
    staging_descriptor: Optional[int] = None
    releases_descriptor: Optional[int] = None
    staged_descriptor: Optional[int] = None
    previous_release_descriptor: Optional[int] = None
    try:
        staging_descriptor = _open_child_directory(
            namespace_descriptor, ".staging", create=True,
        )
        releases_descriptor = _open_child_directory(
            namespace_descriptor, "releases", create=True,
        )
        identities = (
            (namespace_root, namespace_descriptor),
            (staging_root, staging_descriptor),
            (releases_root, releases_descriptor),
        )
        _require_identities(identities)
        try:
            staged_descriptor = contract._open_directory_at(
                staging_descriptor, staged.name,
            )
        except contract.ArtifactError as exc:
            raise SimulationPolicyError(
                "staged release is missing, unsafe, or no longer pinned"
            ) from exc
        envelope = _verify_simulation_release_fd(
            staged_descriptor,
            namespace=namespace,
            expected_kind=paths["artifact_kind"],
        )
        release_id = str(envelope["release_id"])
        _require_staged_direct_child(staged, staging_root, release_id)
        final = releases_root / release_id
        created_final = False
        activation_started = False
        previous: Optional[str] = None
        with _simulation_lock(namespace_descriptor):
            _require_identities(identities)
            previous, previous_release_descriptor = _read_verified_current_at(
                namespace_descriptor,
                releases_descriptor,
                namespace=namespace,
                expected_kind=paths["artifact_kind"],
            )
            if previous != expectation:
                raise contract.ConflictError(
                    "compare-and-swap current mismatch: expected=%r actual=%r"
                    % (expectation, previous)
                )
            try:
                os.stat(
                    release_id,
                    dir_fd=releases_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                contract.write_release_at(
                    staged_descriptor,
                    envelope,
                    releases_descriptor,
                    release_id,
                    target_role="linbox",
                )
                created_final = True
            else:
                installed = contract.verify_release_at(
                    releases_descriptor, release_id,
                )
                if installed != envelope:
                    raise contract.ConflictError(
                        "conflicting existing simulation release has different content"
                    )
                installed_descriptor = contract._open_directory_at(
                    releases_descriptor, release_id,
                )
                try:
                    _verify_simulation_release_fd(
                        installed_descriptor,
                        namespace=namespace,
                        expected_kind=paths["artifact_kind"],
                    )
                finally:
                    os.close(installed_descriptor)

            link_target = "releases/%s" % release_id
            try:
                _require_identities(identities)
                activation_started = True
                _replace_current_at(namespace_descriptor, link_target)
                _require_identities(identities)
                _verify_active_identity(
                    namespace_descriptor,
                    releases_descriptor,
                    release_id,
                    link_target,
                )
                installed_descriptor = contract._open_directory_at(
                    releases_descriptor, release_id,
                )
                try:
                    _verify_simulation_release_fd(
                        installed_descriptor,
                        namespace=namespace,
                        expected_kind=paths["artifact_kind"],
                    )
                finally:
                    os.close(installed_descriptor)
                if post_verify is not None:
                    post_verify(final)
                _require_identities(identities)
                _verify_active_identity(
                    namespace_descriptor,
                    releases_descriptor,
                    release_id,
                    link_target,
                )
            except BaseException:
                intact = _identities_intact(identities)
                if activation_started and intact:
                    _restore_verified_previous_at(
                        namespace_descriptor,
                        releases_descriptor,
                        previous,
                        previous_release_descriptor,
                        namespace=namespace,
                        expected_kind=paths["artifact_kind"],
                    )
                elif not intact:
                    _remove_current_at(namespace_descriptor)
                if created_final and not intact:
                    contract.remove_release_at(releases_descriptor, release_id)
                raise
        return final
    finally:
        if previous_release_descriptor is not None:
            os.close(previous_release_descriptor)
        if staged_descriptor is not None:
            os.close(staged_descriptor)
        if releases_descriptor is not None:
            os.close(releases_descriptor)
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        os.close(namespace_descriptor)
