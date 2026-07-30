"""Fixed-path remote half of the WAM deployment protocol."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import sys
from typing import Callable, Dict, Iterator, Optional, Union
import uuid

_FIXED_CONTRACT = Path(__file__).resolve().with_name(".wam_artifact_contract.py")
if _FIXED_CONTRACT.is_file():
    _spec = importlib.util.spec_from_file_location("artifact_contract", str(_FIXED_CONTRACT))
    if _spec is None or _spec.loader is None:
        raise ImportError("cannot load fixed artifact contract")
    contract = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(contract)
else:
    import artifact_contract as contract


REQUEST_SCHEMA = "wam_remote_install_request_v1"
RESPONSE_SCHEMA = "wam_remote_install_response_v1"
_INCOMING_NAME = re.compile(
    r"^incoming-[a-z][a-z0-9_-]{0,63}-[0-9a-f]{20}-[0-9a-f]{32}$"
)
_REQUEST_FIELDS = frozenset({
    "adopt_existing", "agent_sha256", "contract_sha256", "current_link",
    "expected_manifest_sha256", "expected_payload_sha256", "expected_release_id",
    "incoming_path", "operation", "reservation_id", "releases_root", "schema",
    "scope", "target_role",
})
_CONTROL_FIELDS = frozenset({
    "agent_sha256", "contract_sha256", "operation", "schema", "scope", "target_role",
})
_ROLLBACK_FIELDS = _CONTROL_FIELDS | {"expected_current_target", "previous_target"}
_PREPARE_FIELDS = _CONTROL_FIELDS | {"incoming_path", "reservation_id"}
_CLEANUP_FIELDS = _CONTROL_FIELDS | {
    "created_release_id", "incoming_path", "pre_current_target", "pre_release_ids",
    "reservation_id",
}
_PREPARE_DATA_FIELDS = frozenset({
    "agent_sha256", "contract_sha256", "data_operation", "incoming_path",
    "operation", "reservation_id", "schema", "target_role",
})
_RAW_DATA_FIELDS = frozenset({
    "agent_sha256", "contract_sha256", "expected_finalization_sha256",
    "expected_manifest_sha256", "incoming_path", "operation", "releases_root",
    "reservation_id", "schema", "target_role",
})
_CONVERTED_DATA_FIELDS = frozenset({
    "agent_sha256", "contract_sha256", "expected_manifest_sha256",
    "expected_payload_sha256", "expected_release_id", "incoming_path", "operation",
    "releases_root", "reservation_id", "schema", "target_role",
})
_RAW_RECONCILE_FIELDS = _RAW_DATA_FIELDS | {"data_operation"}
_CONVERTED_RECONCILE_FIELDS = _CONVERTED_DATA_FIELDS | {"data_operation"}
_PERCEPTION_PREPARE_FIELDS = frozenset({
    "agent_sha256", "contract_sha256", "data_kind", "incoming_path",
    "operation", "reservation_id", "schema", "source_role", "target_role",
})
_PERCEPTION_REQUEST_FIELDS = frozenset({
    "agent_sha256", "contract_sha256", "data_kind",
    "expected_manifest_sha256", "expected_payload_sha256",
    "expected_release_id", "incoming_path", "operation", "releases_root",
    "reservation_id", "schema", "source_role", "target_role",
})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}-[0-9a-f]{20}$")
_RESERVATION_SCHEMA = "wam_remote_reservation_v1"
_PRODUCTION_CAPABILITY = "task8_frozen_release_activation_v1"
_PRODUCTION_COMMON_FIELDS = frozenset({
    "agent_sha256", "capability", "contract_sha256", "frozen_payload_sha256",
    "frozen_release_id", "incoming_path", "operation", "reservation_id",
    "schema", "target_role",
})
_PRODUCTION_ACTIVATE_FIELDS = _PRODUCTION_COMMON_FIELDS | {
    "expected_manifest_sha256",
}

HOST_POLICIES = {
    "h200": {
        "home": "/home/ssh-ziyue",
        "helper_path": "/home/ssh-ziyue/.wam_remote_agent.py",
        "contract_path": "/home/ssh-ziyue/.wam_artifact_contract.py",
        "canary": {
            "current_link": "/home/ssh-ziyue/wam_ausc/.deploy-canary/current",
            "incoming_root": "/home/ssh-ziyue/wam_ausc/.deploy-canary/incoming",
            "releases_root": "/home/ssh-ziyue/wam_ausc/.deploy-canary/releases",
        },
        "production": {
            "current_link": "/home/ssh-ziyue/wam_ausc/current",
            "incoming_root": "/home/ssh-ziyue/wam_ausc/.incoming",
            "releases_root": "/home/ssh-ziyue/wam_ausc/.releases",
        },
        "data": {
            "training_trust": {
                "finalization_registry": (
                    "/home/ssh-ziyue/wam_ausc/trust/"
                    "TRAINING_FINALIZATIONS.json"
                ),
                "snapshot_root": (
                    "/home/ssh-ziyue/wam_ausc/data/trusted_training_snapshots"
                ),
            },
            "raw_session": {
                "incoming_root": "/home/ssh-ziyue/wam_ausc/data/raw_sessions/.staging",
                "releases_root": "/home/ssh-ziyue/wam_ausc/data/raw_sessions/releases",
            },
            "converted_dataset": {
                "incoming_root": "/home/ssh-ziyue/wam_ausc/data/lerobot/.staging",
                "releases_root": "/home/ssh-ziyue/wam_ausc/data/lerobot/releases",
            },
            "perception": {
                "perception_canary_evaluation": {
                    "artifact_kind": "perception_canary_evaluation",
                    "releases_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_canary_evaluation/releases"
                    ),
                    "remote_upload": False,
                    "source_role": "h200",
                    "staging_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_canary_evaluation/.staging"
                    ),
                },
                "perception_geometry_source": {
                    "artifact_kind": "perception_geometry_source",
                    "releases_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_geometry_source/releases"
                    ),
                    "remote_upload": True,
                    "source_role": "mac",
                    "staging_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_geometry_source/.staging"
                    ),
                },
                "perception_labels_holdout": {
                    "artifact_kind": "perception_labels_holdout",
                    "releases_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_labels_holdout/releases"
                    ),
                    "remote_upload": True,
                    "source_role": "mac",
                    "staging_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_labels_holdout/.staging"
                    ),
                },
                "perception_labels_train": {
                    "artifact_kind": "perception_labels_train",
                    "releases_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_labels_train/releases"
                    ),
                    "remote_upload": True,
                    "source_role": "mac",
                    "staging_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_labels_train/.staging"
                    ),
                },
                "perception_session2_train": {
                    "artifact_kind": "perception_session2_train",
                    "releases_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_session2_train/releases"
                    ),
                    "remote_upload": False,
                    "source_role": "h200",
                    "staging_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_session2_train/.staging"
                    ),
                },
                "perception_session3_holdout": {
                    "artifact_kind": "perception_session3_holdout",
                    "releases_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_session3_holdout/releases"
                    ),
                    "remote_upload": False,
                    "source_role": "h200",
                    "staging_root": (
                        "/home/ssh-ziyue/wam_ausc/data/"
                        "perception_session3_holdout/.staging"
                    ),
                },
                "perception_stage0": {
                    "artifact_kind": "perception_stage0",
                    "releases_root": (
                        "/home/ssh-ziyue/wam_ausc/data/perception_stage0/releases"
                    ),
                    "remote_upload": False,
                    "source_role": "h200",
                    "staging_root": (
                        "/home/ssh-ziyue/wam_ausc/data/perception_stage0/.staging"
                    ),
                },
            },
            "perception_holdout_consumption_root": (
                "/home/ssh-ziyue/wam_ausc/data/perception_holdout_consumption"
            ),
        },
    },
    "linbox": {
        "home": "/home/robot",
        "helper_path": "/home/robot/.wam_remote_agent.py",
        "contract_path": "/home/robot/.wam_artifact_contract.py",
        "canary": {
            "current_link": "/home/robot/Desktop/stethoscope/.deploy-canary/current",
            "incoming_root": "/home/robot/Desktop/stethoscope/.deploy-canary/incoming",
            "releases_root": "/home/robot/Desktop/stethoscope/.deploy-canary/releases",
        },
        "production": {
            "current_link": "/home/robot/Desktop/stethoscope/.wam-current",
            "incoming_root": "/home/robot/Desktop/stethoscope/.wam-incoming",
            "releases_root": "/home/robot/Desktop/stethoscope/.wam-releases",
        },
    },
}


class AgentPolicyError(RuntimeError):
    """The stdin request is not authorized for this host."""


@dataclass(frozen=True)
class ValidatedRequest:
    current_link: Path
    incoming_path: Path
    releases_root: Path
    scope: str
    target_role: str
    expected_manifest_sha256: str
    expected_payload_sha256: str
    expected_release_id: str
    reservation_id: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_real_directory(path: Path, *, create: bool) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise AgentPolicyError("canonical directory path is invalid")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise AgentPolicyError("canonical directory is missing")
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as exc:
                raise AgentPolicyError(
                    "canonical directory has a symlink or non-directory ancestor"
                ) from exc
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                raise AgentPolicyError(
                    "canonical directory has a symlink or non-directory ancestor"
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


def _require_real_directory(path: Path) -> None:
    descriptor = _open_real_directory(path, create=False)
    os.close(descriptor)


@contextmanager
def _data_deployment_lock(releases_root: Union[str, Path]) -> Iterator[None]:
    """Serialize one H200 data namespace without stale lock ownership."""
    namespace = Path(releases_root).parent
    _ensure_real_directory(namespace)
    parent_descriptor = _open_real_directory(namespace, create=False)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            ".deploy.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise AgentPolicyError("data deployment lock is not a regular file")
        os.fsync(parent_descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(parent_descriptor)


@contextmanager
def _production_deployment_lock(local_role: str) -> Iterator[None]:
    """Serialize production mutation with a persistent no-follow flock."""
    if local_role not in HOST_POLICIES:
        raise AgentPolicyError("unknown production host role")
    base = Path(HOST_POLICIES[local_role]["production"]["current_link"]).parent
    _ensure_real_directory(base)
    parent_descriptor = _open_real_directory(base, create=False)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            ".wam-production.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise AgentPolicyError("production deployment lock is not a regular file")
        os.fsync(parent_descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(parent_descriptor)


def _reservation_marker_path(incoming: Path) -> Path:
    return incoming.parent / (".reservation-%s.json" % incoming.name)


def _reservation_record(
    incoming: Path, *, reservation_id: str, target_role: str, operation: str
) -> Dict[str, str]:
    return {
        "incoming_path": incoming.as_posix(),
        "operation": operation,
        "reservation_id": reservation_id,
        "schema": _RESERVATION_SCHEMA,
        "target_role": target_role,
    }


def _create_owned_reservation(
    incoming: Path, *, reservation_id: str, target_role: str, operation: str
) -> None:
    if not isinstance(reservation_id, str) or not _HEX64.fullmatch(reservation_id):
        raise AgentPolicyError("reservation ownership id is invalid")
    parent_descriptor = _open_real_directory(incoming.parent, create=False)
    try:
        os.mkdir(incoming.name, mode=0o700, dir_fd=parent_descriptor)
    except FileExistsError as exc:
        os.close(parent_descriptor)
        raise AgentPolicyError("incoming path is already reserved") from exc
    except BaseException:
        os.close(parent_descriptor)
        raise
    marker = _reservation_marker_path(incoming)
    descriptor: Optional[int] = None
    marker_created = False
    try:
        descriptor = os.open(
            marker.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600, dir_fd=parent_descriptor,
        )
        marker_created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(contract.canonical_json_bytes(_reservation_record(
                incoming, reservation_id=reservation_id,
                target_role=target_role, operation=operation,
            )))
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent_descriptor)
    except BaseException as exc:
        if descriptor is not None:
            os.close(descriptor)
        if marker_created:
            try:
                os.unlink(marker.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(incoming.name, dir_fd=parent_descriptor)
        except OSError:
            pass
        if isinstance(exc, AgentPolicyError):
            raise
        raise AgentPolicyError("incoming reservation marker already exists or is unsafe") from exc
    finally:
        os.close(parent_descriptor)


def _require_owned_reservation(
    incoming: Path, *, reservation_id: str, target_role: str, operation: str
) -> Path:
    if not isinstance(reservation_id, str) or not _HEX64.fullmatch(reservation_id):
        raise AgentPolicyError("reservation ownership id is invalid")
    marker = _reservation_marker_path(incoming)
    parent_descriptor: Optional[int] = None
    descriptor: Optional[int] = None
    try:
        parent_descriptor = _open_real_directory(incoming.parent, create=False)
        descriptor = os.open(
            marker.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        info = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            raw = handle.read(65537)
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentPolicyError("reservation ownership marker is missing or invalid") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    expected = _reservation_record(
        incoming, reservation_id=reservation_id,
        target_role=target_role, operation=operation,
    )
    if (
        not stat.S_ISREG(info.st_mode)
        or not isinstance(value, dict)
        or value != expected
        or raw != contract.canonical_json_bytes(value)
    ):
        raise AgentPolicyError("reservation ownership marker does not match this request")
    return marker


def _remove_owned_reservation_marker(
    incoming: Path, *, reservation_id: str, target_role: str, operation: str
) -> None:
    marker = _require_owned_reservation(
        incoming, reservation_id=reservation_id,
        target_role=target_role, operation=operation,
    )
    parent_descriptor = _open_real_directory(marker.parent, create=False)
    try:
        os.unlink(marker.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _bundle_contract_path(agent_path: Path) -> Path:
    fixed = agent_path.resolve().with_name(".wam_artifact_contract.py")
    if fixed.is_file():
        return fixed
    return Path(contract.__file__).resolve()


def _validate_bundle_hashes(request: Dict[str, object], agent_path: Path) -> None:
    if request.get("agent_sha256") != _sha256_file(agent_path):
        raise AgentPolicyError("remote agent hash does not match the caller")
    if request.get("contract_sha256") != _sha256_file(_bundle_contract_path(agent_path)):
        raise AgentPolicyError("artifact contract hash does not match the caller")


def validate_request(
    request: object,
    *,
    local_role: str,
    agent_path: Union[str, Path],
) -> ValidatedRequest:
    if not isinstance(request, dict) or set(request) != _REQUEST_FIELDS:
        raise AgentPolicyError("remote request fields are invalid")
    if request.get("schema") != REQUEST_SCHEMA or request.get("operation") != "install_release":
        raise AgentPolicyError("remote request schema or operation is invalid")
    if local_role not in HOST_POLICIES or request.get("target_role") != local_role:
        raise AgentPolicyError("remote request role does not match this host")
    if request.get("scope") != "canary":
        raise AgentPolicyError("production code installation requires a separate Task 8 capability")
    if request.get("adopt_existing") is not False:
        raise AgentPolicyError("existing real paths require the separate audited migration gate")
    _validate_bundle_hashes(request, Path(agent_path))
    for field in ("expected_manifest_sha256", "expected_payload_sha256"):
        if not isinstance(request.get(field), str) or not _HEX64.fullmatch(request[field]):
            raise AgentPolicyError("remote expected identity is invalid")
    if not isinstance(request.get("reservation_id"), str) or not _HEX64.fullmatch(
        request["reservation_id"]
    ):
        raise AgentPolicyError("remote reservation ownership id is invalid")
    if not isinstance(request.get("expected_release_id"), str) or not _RELEASE_ID.fullmatch(
        request["expected_release_id"]
    ):
        raise AgentPolicyError("remote expected release id is invalid")

    policy = HOST_POLICIES[local_role][request["scope"]]
    if request.get("releases_root") != policy["releases_root"]:
        raise AgentPolicyError("remote releases root is outside policy")
    if request.get("current_link") != policy["current_link"]:
        raise AgentPolicyError("remote current link is outside policy")
    raw_incoming = request.get("incoming_path")
    if not isinstance(raw_incoming, str):
        raise AgentPolicyError("remote incoming path is invalid")
    incoming = Path(raw_incoming)
    if (
        not incoming.is_absolute()
        or incoming.as_posix() != raw_incoming
        or incoming.parent.as_posix() != policy["incoming_root"]
        or not _INCOMING_NAME.fullmatch(incoming.name)
    ):
        raise AgentPolicyError("remote incoming path is outside policy")
    return ValidatedRequest(
        current_link=Path(policy["current_link"]),
        incoming_path=incoming,
        releases_root=Path(policy["releases_root"]),
        scope=request["scope"],
        target_role=local_role,
        expected_manifest_sha256=request["expected_manifest_sha256"],
        expected_payload_sha256=request["expected_payload_sha256"],
        expected_release_id=request["expected_release_id"],
        reservation_id=request["reservation_id"],
    )


def _validate_production_request(
    request: object, *, local_role: str, agent_path: Union[str, Path]
) -> Dict[str, object]:
    if not isinstance(request, dict):
        raise AgentPolicyError("production request must be an object")
    operation = request.get("operation")
    expected_schema = {
        "prepare_production_incoming": "wam_remote_production_prepare_request_v1",
        "activate_production_release": "wam_remote_production_activate_request_v1",
        "reconcile_production_release": "wam_remote_production_reconcile_request_v1",
    }.get(operation)
    expected_fields = (
        _PRODUCTION_COMMON_FIELDS
        if operation == "prepare_production_incoming"
        else _PRODUCTION_ACTIVATE_FIELDS
    )
    if expected_schema is None or request.get("schema") != expected_schema:
        raise AgentPolicyError("production request schema or operation is invalid")
    if set(request) != expected_fields:
        raise AgentPolicyError("production request fields are invalid")
    if request.get("capability") != _PRODUCTION_CAPABILITY:
        raise AgentPolicyError("production capability is missing or invalid")
    if local_role not in HOST_POLICIES or request.get("target_role") != local_role:
        raise AgentPolicyError("production request role does not match this host")
    _validate_bundle_hashes(request, Path(agent_path))
    for field in ("frozen_payload_sha256",):
        if not isinstance(request.get(field), str) or not _HEX64.fullmatch(request[field]):
            raise AgentPolicyError("production frozen payload identity is invalid")
    if operation != "prepare_production_incoming" and (
        not isinstance(request.get("expected_manifest_sha256"), str)
        or not _HEX64.fullmatch(request["expected_manifest_sha256"])
    ):
        raise AgentPolicyError("production manifest identity is invalid")
    if (
        not isinstance(request.get("frozen_release_id"), str)
        or not _RELEASE_ID.fullmatch(request["frozen_release_id"])
        or not isinstance(request.get("reservation_id"), str)
        or not _HEX64.fullmatch(request["reservation_id"])
    ):
        raise AgentPolicyError("production release or reservation identity is invalid")
    paths = HOST_POLICIES[local_role]["production"]
    raw_incoming = request.get("incoming_path")
    if not isinstance(raw_incoming, str):
        raise AgentPolicyError("production incoming path is invalid")
    incoming = Path(raw_incoming)
    if (
        not incoming.is_absolute()
        or incoming.as_posix() != raw_incoming
        or incoming.parent.as_posix() != paths["incoming_root"]
        or not _INCOMING_NAME.fullmatch(incoming.name)
        or not incoming.name.startswith("incoming-%s-" % request["frozen_release_id"])
    ):
        raise AgentPolicyError("production incoming path is outside policy")
    return dict(request)


def prepare_production_incoming(
    local_role: str,
    *,
    incoming_path: str,
    reservation_id: str,
    frozen_release_id: str,
    frozen_payload_sha256: str,
) -> Dict[str, object]:
    """Reserve one production incoming path without using canary control state."""
    if (
        local_role not in HOST_POLICIES
        or not _RELEASE_ID.fullmatch(frozen_release_id)
        or not _HEX64.fullmatch(frozen_payload_sha256)
    ):
        raise AgentPolicyError("production frozen identity is invalid")
    paths = HOST_POLICIES[local_role]["production"]
    incoming = Path(incoming_path)
    if (
        incoming.parent.as_posix() != paths["incoming_root"]
        or not _INCOMING_NAME.fullmatch(incoming.name)
        or not incoming.name.startswith("incoming-%s-" % frozen_release_id)
    ):
        raise AgentPolicyError("production incoming reservation is outside policy")
    with _production_deployment_lock(local_role):
        _ensure_real_directory(Path(paths["incoming_root"]))
        _ensure_real_directory(Path(paths["releases_root"]))
        _assert_no_other_production_transaction(Path(paths["current_link"]).parent)
        _create_owned_reservation(
            incoming,
            reservation_id=reservation_id,
            target_role=local_role,
            operation="activate_production_release",
        )
    return {
        "frozen_payload_sha256": frozen_payload_sha256,
        "frozen_release_id": frozen_release_id,
        "incoming_path": incoming.as_posix(),
        "reservation_id": reservation_id,
        "schema": "wam_remote_production_prepare_response_v1",
        "target_role": local_role,
    }


def detect_local_role(home: Optional[Union[str, Path]] = None) -> str:
    actual = Path(home) if home is not None else Path.home()
    matches = [role for role, policy in HOST_POLICIES.items() if policy["home"] == actual.as_posix()]
    if len(matches) != 1:
        raise AgentPolicyError("remote agent cannot identify an audited host role")
    return matches[0]


def _canary_paths(local_role: str) -> Dict[str, str]:
    if local_role not in HOST_POLICIES:
        raise AgentPolicyError("unknown host role")
    return HOST_POLICIES[local_role]["canary"]


def bootstrap_canary(local_role: str) -> Dict[str, object]:
    paths = _canary_paths(local_role)
    for key in ("incoming_root", "releases_root"):
        _ensure_real_directory(Path(paths[key]))
    _ensure_real_directory(Path(paths["current_link"]).parent)
    return {"schema": "wam_remote_bootstrap_response_v1", "target_role": local_role}


def prepare_incoming(
    local_role: str, incoming_path: str, reservation_id: str
) -> Dict[str, object]:
    paths = _canary_paths(local_role)
    incoming = Path(incoming_path)
    incoming_root = Path(paths["incoming_root"])
    if (
        not incoming.is_absolute()
        or incoming.as_posix() != incoming_path
        or incoming.parent.as_posix() != paths["incoming_root"]
        or not _INCOMING_NAME.fullmatch(incoming.name)
    ):
        raise AgentPolicyError("incoming reservation is outside canary policy")
    _ensure_real_directory(incoming_root)
    _create_owned_reservation(
        incoming, reservation_id=reservation_id,
        target_role=local_role, operation="install_release",
    )
    return {
        "incoming_path": incoming_path,
        "reservation_id": reservation_id,
        "schema": "wam_remote_prepare_response_v1",
        "target_role": local_role,
    }


def prepare_data_incoming(
    local_role: str, *, data_operation: str, incoming_path: str, reservation_id: str
) -> Dict[str, object]:
    policy = operation_policy(local_role, data_operation)
    incoming = Path(incoming_path)
    incoming_root = Path(policy["incoming_root"])
    valid_name = (
        re.fullmatch(r"transfer-raw-[0-9a-f]{32}", incoming.name) is not None
        if data_operation == "archive_raw_session"
        else _RELEASE_ID.fullmatch(incoming.name) is not None
    )
    if (
        not incoming.is_absolute()
        or incoming.as_posix() != incoming_path
        or incoming.parent.as_posix() != policy["incoming_root"]
        or not valid_name
    ):
        raise AgentPolicyError("data incoming reservation is outside policy")
    with _data_deployment_lock(policy["releases_root"]):
        _ensure_real_directory(incoming_root)
        _create_owned_reservation(
            incoming, reservation_id=reservation_id,
            target_role=local_role, operation=data_operation,
        )
    return {
        "incoming_path": incoming_path,
        "reservation_id": reservation_id,
        "schema": "wam_remote_data_prepare_response_v1",
        "target_role": local_role,
    }


def prepare_perception_incoming(
    local_role: str,
    *,
    data_kind: str,
    source_role: str,
    incoming_path: str,
    reservation_id: str,
) -> Dict[str, object]:
    policy = perception_artifact_policy(local_role, data_kind)
    if policy["remote_upload"] is not True or policy["source_role"] != source_role:
        raise AgentPolicyError("perception artifact does not allow this remote upload")
    incoming = Path(incoming_path)
    if (
        not incoming.is_absolute()
        or incoming.as_posix() != incoming_path
        or incoming.parent.as_posix() != policy["staging_root"]
        or re.fullmatch(
            r"incoming-%s-[0-9a-f]{20}-[0-9a-f]{32}" % re.escape(data_kind),
            incoming.name,
        )
        is None
    ):
        raise AgentPolicyError("perception incoming reservation is outside policy")
    with _data_deployment_lock(policy["releases_root"]):
        _ensure_real_directory(Path(policy["staging_root"]))
        _ensure_real_directory(Path(policy["releases_root"]))
        _create_owned_reservation(
            incoming,
            reservation_id=reservation_id,
            target_role=local_role,
            operation="install_perception_artifact",
        )
    return {
        "data_kind": data_kind,
        "incoming_path": incoming_path,
        "reservation_id": reservation_id,
        "schema": "wam_remote_perception_prepare_response_v1",
        "source_role": source_role,
        "target_role": local_role,
    }


def inspect_canary(local_role: str) -> Dict[str, object]:
    paths = _canary_paths(local_role)
    current = Path(paths["current_link"])
    if os.path.lexists(str(current)) and not current.is_symlink():
        raise AgentPolicyError("canary current is not a symlink")
    releases = Path(paths["releases_root"])
    _require_real_directory(current.parent)
    _require_real_directory(releases)
    release_ids = []
    if releases.is_dir():
        for candidate in sorted(releases.iterdir()):
            if candidate.name.startswith("."):
                continue
            if candidate.is_symlink() or not candidate.is_dir() or not _RELEASE_ID.fullmatch(
                candidate.name
            ):
                raise AgentPolicyError("canary release root contains an unsafe entry")
            contract.verify_release(candidate)
            release_ids.append(candidate.name)
    current_target = os.readlink(str(current)) if current.is_symlink() else None
    if current_target is not None:
        relative = PurePosixPath(current_target)
        resolved = (current.parent / Path(*relative.parts)).resolve()
        if (
            relative.is_absolute()
            or relative.as_posix() != current_target
            or ".." in relative.parts
            or resolved.parent != releases.resolve()
            or resolved.name not in release_ids
        ):
            raise AgentPolicyError("canary current target is outside the release root")
        contract.verify_release(resolved)
    return {
        "current_target": current_target,
        "release_ids": release_ids,
        "schema": "wam_remote_canary_inspection_v1",
        "target_role": local_role,
    }


def _resolve_canary_target(
    current: Path, releases: Path, target: Optional[str], label: str
) -> Optional[Path]:
    if target is None:
        return None
    relative = PurePosixPath(target)
    if (
        relative.is_absolute()
        or relative.as_posix() != target
        or ".." in relative.parts
    ):
        raise AgentPolicyError("%s is not a safe relative path" % label)
    resolved = (current.parent / Path(*relative.parts)).resolve()
    if resolved.parent != releases.resolve():
        raise AgentPolicyError("%s is outside canary releases" % label)
    contract.verify_release(resolved)
    return resolved


def force_canary_rollback(
    local_role: str,
    previous_target: Optional[str],
    *,
    expected_current_target: Optional[str],
) -> Dict[str, object]:
    paths = _canary_paths(local_role)
    current = Path(paths["current_link"])
    releases = Path(paths["releases_root"])
    _require_real_directory(current.parent)
    _require_real_directory(releases)
    _resolve_canary_target(
        current, releases, expected_current_target, "expected current target"
    )
    previous_release = _resolve_canary_target(
        current, releases, previous_target, "rollback target"
    )
    with contract.deployment_lock(releases.parent / ".deploy.lock"):
        if os.path.lexists(str(current)) and not current.is_symlink():
            raise AgentPolicyError("canary current is not a symlink")
        actual_current = os.readlink(str(current)) if current.is_symlink() else None
        if actual_current != expected_current_target:
            raise AgentPolicyError("canary rollback compare-and-swap current mismatch")
        if previous_target is None:
            current.unlink(missing_ok=True)
        else:
            contract.verify_release(previous_release)
            temporary = current.parent / (".current.rollback-%s" % os.getpid())
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(previous_target)
            os.replace(str(temporary), str(current))
            contract._fsync_directory(current.parent)
    return {
        "rolled_back_to": previous_target,
        "schema": "wam_remote_canary_rollback_v1",
        "target_role": local_role,
    }


def cleanup_canary(
    local_role: str,
    *,
    incoming_path: str,
    reservation_id: str,
    created_release_id: Optional[str],
    pre_current_target: Optional[str],
    pre_release_ids: tuple,
) -> Dict[str, object]:
    paths = _canary_paths(local_role)
    current = Path(paths["current_link"])
    releases = Path(paths["releases_root"])
    incoming = Path(incoming_path)
    if (
        not incoming.is_absolute()
        or incoming.as_posix() != incoming_path
        or incoming.parent.as_posix() != paths["incoming_root"]
        or not isinstance(pre_release_ids, tuple)
        or any(not isinstance(item, str) or not _RELEASE_ID.fullmatch(item) for item in pre_release_ids)
        or tuple(sorted(set(pre_release_ids))) != pre_release_ids
    ):
        raise AgentPolicyError("canary cleanup request is outside policy")
    _require_real_directory(current.parent)
    _require_real_directory(releases)
    with contract.deployment_lock(releases.parent / ".deploy.lock"):
        _require_owned_reservation(
            incoming, reservation_id=reservation_id,
            target_role=local_role, operation="install_release",
        )
        if os.path.lexists(str(current)) and not current.is_symlink():
            raise AgentPolicyError("canary current diverged before cleanup")
        actual_current = os.readlink(str(current)) if current.is_symlink() else None
        if actual_current != pre_current_target:
            raise AgentPolicyError("canary current diverged before cleanup")
        actual_releases = set()
        if releases.is_dir():
            for item in releases.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_symlink() or not item.is_dir() or not _RELEASE_ID.fullmatch(item.name):
                    raise AgentPolicyError("canary releases diverged before cleanup")
                actual_releases.add(item.name)
        expected = set(pre_release_ids)
        if created_release_id is not None:
            if not _RELEASE_ID.fullmatch(created_release_id) or created_release_id in expected:
                raise AgentPolicyError("created release cleanup identity is invalid")
            expected.add(created_release_id)
        if actual_releases != expected:
            raise AgentPolicyError("canary releases diverged before cleanup")
        if incoming.exists():
            if incoming.is_symlink() or not incoming.is_dir():
                raise AgentPolicyError("refusing unsafe incoming cleanup")
            shutil.rmtree(str(incoming))
        removed_release = None
        if created_release_id is not None:
            candidate = releases / created_release_id
            contract.verify_release(candidate)
            shutil.rmtree(str(candidate))
            removed_release = created_release_id
        remaining = {
            item.name for item in releases.iterdir()
            if item.is_dir() and not item.is_symlink() and not item.name.startswith(".")
        } if releases.is_dir() else set()
        if remaining != set(pre_release_ids):
            raise AgentPolicyError("canary cleanup did not restore the pre-run release set")
        _remove_owned_reservation_marker(
            incoming, reservation_id=reservation_id,
            target_role=local_role, operation="install_release",
        )
        final_current = os.readlink(str(current)) if current.is_symlink() else None
        final_releases = {
            item.name for item in releases.iterdir()
            if item.is_dir() and not item.is_symlink() and not item.name.startswith(".")
        } if releases.is_dir() else set()
        if (
            final_current != pre_current_target
            or final_releases != set(pre_release_ids)
            or os.path.lexists(str(incoming))
            or os.path.lexists(str(_reservation_marker_path(incoming)))
        ):
            raise AgentPolicyError("canary state diverged during final cleanup verification")
        return {
            "removed_incoming": incoming_path,
            "removed_release_id": removed_release,
            "schema": "wam_remote_canary_cleanup_v1",
            "target_role": local_role,
        }


def _validate_control_request(
    request: object, *, local_role: str, agent_path: Union[str, Path]
) -> Dict[str, object]:
    if not isinstance(request, dict):
        raise AgentPolicyError("remote control request fields are invalid")
    fields = {
        "rollback_canary": _ROLLBACK_FIELDS,
        "prepare_incoming": _PREPARE_FIELDS,
        "cleanup_canary": _CLEANUP_FIELDS,
    }.get(request.get("operation"), _CONTROL_FIELDS)
    if set(request) != fields:
        raise AgentPolicyError("remote control request fields are invalid")
    if request.get("schema") != "wam_remote_control_request_v1":
        raise AgentPolicyError("remote control request schema is invalid")
    if request.get("target_role") != local_role or request.get("scope") != "canary":
        raise AgentPolicyError("remote control request is not for this canary")
    if request.get("operation") not in {
        "bootstrap_canary", "prepare_incoming", "inspect_canary", "cleanup_canary",
        "rollback_canary"
    }:
        raise AgentPolicyError("remote control operation is invalid")
    _validate_bundle_hashes(request, Path(agent_path))
    if request["operation"] == "prepare_incoming" and (
        not isinstance(request.get("incoming_path"), str)
        or not isinstance(request.get("reservation_id"), str)
        or not _HEX64.fullmatch(request["reservation_id"])
    ):
        raise AgentPolicyError("incoming reservation evidence is invalid")
    if request["operation"] == "rollback_canary" and any(
        request.get(field) is not None and not isinstance(request.get(field), str)
        for field in ("previous_target", "expected_current_target")
    ):
        raise AgentPolicyError("rollback target evidence is invalid")
    if request["operation"] == "cleanup_canary":
        release_ids = request.get("pre_release_ids")
        if (
            not isinstance(request.get("incoming_path"), str)
            or not isinstance(request.get("reservation_id"), str)
            or not _HEX64.fullmatch(request["reservation_id"])
            or request.get("created_release_id") is not None
            and not isinstance(request.get("created_release_id"), str)
            or request.get("pre_current_target") is not None
            and not isinstance(request.get("pre_current_target"), str)
            or not isinstance(release_ids, list)
            or any(not isinstance(item, str) for item in release_ids)
        ):
            raise AgentPolicyError("cleanup pre-release evidence is invalid")
    return request


def _validate_data_prepare_request(
    request: object, *, local_role: str, agent_path: Union[str, Path]
) -> Dict[str, object]:
    if not isinstance(request, dict) or set(request) != _PREPARE_DATA_FIELDS:
        raise AgentPolicyError("remote data prepare request fields are invalid")
    if (
        request.get("schema") != "wam_remote_data_prepare_request_v1"
        or request.get("operation") != "prepare_data_incoming"
        or request.get("data_operation") not in {
            "archive_raw_session", "install_converted_dataset"
        }
        or local_role != "h200"
        or request.get("target_role") != "h200"
    ):
        raise AgentPolicyError("remote data prepare request is outside policy")
    _validate_bundle_hashes(request, Path(agent_path))
    if not isinstance(request.get("reservation_id"), str) or not _HEX64.fullmatch(
        request["reservation_id"]
    ):
        raise AgentPolicyError("remote data reservation ownership id is invalid")
    raw = request.get("incoming_path")
    if not isinstance(raw, str):
        raise AgentPolicyError("remote data prepare path is invalid")
    policy = operation_policy(local_role, request["data_operation"])
    incoming = Path(raw)
    valid_name = (
        re.fullmatch(r"transfer-raw-[0-9a-f]{32}", incoming.name) is not None
        if request["data_operation"] == "archive_raw_session"
        else _RELEASE_ID.fullmatch(incoming.name) is not None
    )
    if (
        not incoming.is_absolute()
        or incoming.as_posix() != raw
        or incoming.parent.as_posix() != policy["incoming_root"]
        or not valid_name
    ):
        raise AgentPolicyError("remote data prepare path is outside policy")
    return request


def _validate_data_request(
    request: object, *, local_role: str, agent_path: Union[str, Path]
) -> Dict[str, object]:
    if not isinstance(request, dict) or request.get("schema") != "wam_remote_data_request_v1":
        raise AgentPolicyError("remote data request schema is invalid")
    operation = request.get("operation")
    fields = _RAW_DATA_FIELDS if operation == "archive_raw_session" else _CONVERTED_DATA_FIELDS
    if operation not in ("archive_raw_session", "install_converted_dataset") or set(request) != fields:
        raise AgentPolicyError("remote data request fields are invalid")
    if local_role != "h200" or request.get("target_role") != "h200":
        raise AgentPolicyError("remote data operation is H200-only")
    _validate_bundle_hashes(request, Path(agent_path))
    if not isinstance(request.get("reservation_id"), str) or not _HEX64.fullmatch(
        request["reservation_id"]
    ):
        raise AgentPolicyError("remote data reservation ownership id is invalid")
    policy = operation_policy(local_role, operation)
    if request.get("releases_root") != policy["releases_root"]:
        raise AgentPolicyError("remote data release root is outside policy")
    raw = request.get("incoming_path")
    if not isinstance(raw, str) or not Path(raw).is_absolute() or Path(raw).as_posix() != raw:
        raise AgentPolicyError("remote data incoming path is invalid")
    if Path(raw).parent.as_posix() != policy["incoming_root"]:
        raise AgentPolicyError("remote data incoming path is outside policy")
    for key in set(request).intersection({
        "expected_finalization_sha256", "expected_manifest_sha256", "expected_payload_sha256"
    }):
        if not isinstance(request[key], str) or not _HEX64.fullmatch(request[key]):
            raise AgentPolicyError("remote data identity hash is invalid")
    if "expected_release_id" in request and (
        not isinstance(request["expected_release_id"], str)
        or not _RELEASE_ID.fullmatch(request["expected_release_id"])
    ):
        raise AgentPolicyError("remote data release id is invalid")
    incoming = Path(raw)
    if operation == "archive_raw_session":
        if re.fullmatch(r"transfer-raw-[0-9a-f]{32}", incoming.name) is None:
            raise AgentPolicyError("remote raw staging identity is invalid")
    elif incoming.name != request["expected_release_id"]:
        raise AgentPolicyError("remote converted staging identity is invalid")
    return request


def _validate_data_reconcile_request(
    request: object, *, local_role: str, agent_path: Union[str, Path]
) -> Dict[str, object]:
    if not isinstance(request, dict):
        raise AgentPolicyError("remote data reconciliation request is invalid")
    data_operation = request.get("data_operation")
    fields = (
        _RAW_RECONCILE_FIELDS
        if data_operation == "archive_raw_session"
        else _CONVERTED_RECONCILE_FIELDS
    )
    if (
        request.get("schema") != "wam_remote_data_reconcile_request_v1"
        or request.get("operation") != "reconcile_data_transfer"
        or data_operation not in {"archive_raw_session", "install_converted_dataset"}
        or set(request) != fields
    ):
        raise AgentPolicyError("remote data reconciliation fields are invalid")
    promotion = dict(request)
    promotion.pop("data_operation")
    promotion["operation"] = data_operation
    promotion["schema"] = "wam_remote_data_request_v1"
    _validate_data_request(promotion, local_role=local_role, agent_path=agent_path)
    return request


def _validate_perception_prepare_request(
    request: object, *, local_role: str, agent_path: Union[str, Path]
) -> Dict[str, object]:
    if not isinstance(request, dict) or set(request) != _PERCEPTION_PREPARE_FIELDS:
        raise AgentPolicyError("remote perception prepare request fields are invalid")
    if (
        request.get("schema") != "wam_remote_perception_prepare_request_v1"
        or request.get("operation") != "prepare_perception_incoming"
        or local_role != "h200"
        or request.get("target_role") != "h200"
        or request.get("source_role") != "mac"
    ):
        raise AgentPolicyError("remote perception prepare request is outside policy")
    _validate_bundle_hashes(request, Path(agent_path))
    if not isinstance(request.get("reservation_id"), str) or not _HEX64.fullmatch(
        request["reservation_id"]
    ):
        raise AgentPolicyError("remote perception reservation ownership id is invalid")
    data_kind = request.get("data_kind")
    if not isinstance(data_kind, str):
        raise AgentPolicyError("remote perception data kind is invalid")
    policy = perception_artifact_policy(local_role, data_kind)
    if policy["remote_upload"] is not True or policy["source_role"] != "mac":
        raise AgentPolicyError("remote perception data kind forbids remote upload")
    raw = request.get("incoming_path")
    if not isinstance(raw, str):
        raise AgentPolicyError("remote perception prepare path is invalid")
    incoming = Path(raw)
    if (
        not incoming.is_absolute()
        or incoming.as_posix() != raw
        or incoming.parent.as_posix() != policy["staging_root"]
        or re.fullmatch(
            r"incoming-%s-[0-9a-f]{20}-[0-9a-f]{32}" % re.escape(data_kind),
            incoming.name,
        )
        is None
    ):
        raise AgentPolicyError("remote perception prepare path is outside policy")
    return request


def _validate_perception_request(
    request: object, *, local_role: str, agent_path: Union[str, Path]
) -> Dict[str, object]:
    if not isinstance(request, dict) or set(request) != _PERCEPTION_REQUEST_FIELDS:
        raise AgentPolicyError("remote perception request fields are invalid")
    if (
        request.get("schema") != "wam_remote_perception_request_v1"
        or request.get("operation") != "install_perception_artifact"
        or local_role != "h200"
        or request.get("target_role") != "h200"
        or request.get("source_role") != "mac"
    ):
        raise AgentPolicyError("remote perception request is outside policy")
    _validate_bundle_hashes(request, Path(agent_path))
    if not isinstance(request.get("reservation_id"), str) or not _HEX64.fullmatch(
        request["reservation_id"]
    ):
        raise AgentPolicyError("remote perception reservation ownership id is invalid")
    data_kind = request.get("data_kind")
    if not isinstance(data_kind, str):
        raise AgentPolicyError("remote perception data kind is invalid")
    policy = perception_artifact_policy(local_role, data_kind)
    if policy["remote_upload"] is not True or policy["source_role"] != "mac":
        raise AgentPolicyError("remote perception data kind forbids remote upload")
    if request.get("releases_root") != policy["releases_root"]:
        raise AgentPolicyError("remote perception releases root is outside policy")
    for field in ("expected_manifest_sha256", "expected_payload_sha256"):
        if not isinstance(request.get(field), str) or not _HEX64.fullmatch(request[field]):
            raise AgentPolicyError("remote perception expected identity is invalid")
    release_id = request.get("expected_release_id")
    if (
        not isinstance(release_id, str)
        or not _RELEASE_ID.fullmatch(release_id)
        or release_id
        != "%s-%s" % (data_kind, request["expected_payload_sha256"][:20])
    ):
        raise AgentPolicyError("remote perception release identity is invalid")
    raw = request.get("incoming_path")
    if not isinstance(raw, str):
        raise AgentPolicyError("remote perception incoming path is invalid")
    incoming = Path(raw)
    if (
        not incoming.is_absolute()
        or incoming.as_posix() != raw
        or incoming.parent.as_posix() != policy["staging_root"]
        or re.fullmatch(
            r"incoming-%s-[0-9a-f]{32}" % re.escape(release_id), incoming.name
        )
        is None
    ):
        raise AgentPolicyError("remote perception incoming path is outside policy")
    return request


def _validate_perception_reconcile_request(
    request: object, *, local_role: str, agent_path: Union[str, Path]
) -> Dict[str, object]:
    if (
        not isinstance(request, dict)
        or set(request) != _PERCEPTION_REQUEST_FIELDS
        or request.get("schema")
        != "wam_remote_perception_reconcile_request_v1"
        or request.get("operation") != "reconcile_perception_transfer"
    ):
        raise AgentPolicyError("remote perception reconciliation fields are invalid")
    promotion = dict(request)
    promotion["operation"] = "install_perception_artifact"
    promotion["schema"] = "wam_remote_perception_request_v1"
    _validate_perception_request(
        promotion, local_role=local_role, agent_path=agent_path
    )
    return request


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise AgentPolicyError("artifact inventory path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise AgentPolicyError("artifact inventory path is unsafe")
    return value


def _read_json_file(path: Path) -> tuple:
    if path.is_symlink() or not path.is_file():
        raise AgentPolicyError("required JSON file is missing or unsafe: %s" % path.name)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentPolicyError("required JSON file is invalid: %s" % path.name) from exc
    if not isinstance(value, dict):
        raise AgentPolicyError("required JSON object is invalid: %s" % path.name)
    return value, raw


def _direct_child(path: Path, root: Path, label: str) -> None:
    _require_real_directory(root)
    if path.is_symlink() or path.parent.as_posix() != root.as_posix():
        raise AgentPolicyError("%s is outside its allowed root" % label)


def validate_raw_session(
    session_dir: Union[str, Path], *, allowed_sessions_root: Union[str, Path],
    trusted_finalization_registry: Union[str, Path],
    trusted_snapshot_root: Union[str, Path],
    expected_finalization_sha256: str,
) -> Dict[str, object]:
    session = Path(session_dir)
    root = Path(allowed_sessions_root)
    _direct_child(session, root, "raw session")
    if not session.is_dir():
        raise AgentPolicyError("raw session is missing")
    try:
        trusted_record = contract.load_trusted_training_finalization_record(
            trusted_finalization_registry,
            finalization_sha256=expected_finalization_sha256,
        )
        verified = contract.verify_training_session_contract(
            session,
            trusted_finalization_record=trusted_record,
            trusted_snapshot_root=trusted_snapshot_root,
        )
    except (contract.ArtifactError, OSError, ValueError) as exc:
        raise AgentPolicyError(
            "raw training promotion contract verification failed: %s" % exc
        ) from exc
    snapshot = Path(str(verified.get("verified_snapshot_path", "")))
    if not snapshot.is_absolute() or not snapshot.is_dir() or snapshot.is_symlink():
        raise AgentPolicyError("raw training verifier returned an unsafe snapshot")
    if verified.get("finalization_sha256") != expected_finalization_sha256:
        raise AgentPolicyError(
            "raw training verifier returned the wrong trusted finalization"
        )
    return dict(verified)


def validate_converted_dataset(
    release_dir: Union[str, Path], *, allowed_releases_root: Union[str, Path],
    raw_releases_root: Optional[Union[str, Path]] = None,
    trusted_finalization_registry: Optional[Union[str, Path]] = None,
    trusted_snapshot_root: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    release = Path(release_dir)
    root = Path(allowed_releases_root)
    _direct_child(release, root, "converted dataset")
    envelope = contract.verify_release(release)
    payload = envelope["payload"]
    if payload.get("artifact_kind") != "lerobot_dataset" or payload.get("source_role") != "linbox":
        raise AgentPolicyError("converted dataset artifact role or kind is invalid")
    if release.name != envelope["release_id"]:
        raise AgentPolicyError("converted dataset release path does not match its identity")
    payload_paths = {row.get("path") for row in payload.get("files", []) if isinstance(row, dict)}
    required_meta = {"DATASET_MANIFEST.json", "DATASET_PROVENANCE.json"}
    if not required_meta <= payload_paths:
        raise AgentPolicyError("converted dataset metadata is missing from the payload")
    dataset_manifest, dataset_manifest_raw = _read_json_file(release / "DATASET_MANIFEST.json")
    provenance, provenance_raw = _read_json_file(release / "DATASET_PROVENANCE.json")
    if dataset_manifest_raw != contract.canonical_json_bytes(dataset_manifest):
        raise AgentPolicyError("converted dataset manifest is not canonical")
    expected_provenance_fields = {
        "converter_code_release_id", "converter_environment_id", "converter_lock_sha256",
        "dataset_manifest_sha256", "dataset_schema_version", "lerobot_version",
        "promotion_contract_version", "raw_finalization_sha256", "raw_manifest_sha256",
        "raw_payload_sha256", "raw_release_id", "raw_session_id",
        "raw_training_authenticity_sha256", "schema",
    }
    if set(provenance) != expected_provenance_fields or provenance_raw != contract.canonical_json_bytes(provenance):
        raise AgentPolicyError("converted dataset provenance is invalid")
    if (
        provenance.get("schema") != "wam_converted_dataset_provenance_v2"
        or provenance.get("promotion_contract_version") != "wam_promotion_v2"
    ):
        raise AgentPolicyError("converted dataset provenance schema is invalid")
    if provenance.get("lerobot_version") != "0.3.3":
        raise AgentPolicyError("converted dataset must use exact LeRobot 0.3.3")
    if provenance.get("dataset_schema_version") != "lerobot_v2":
        raise AgentPolicyError("converted dataset schema version is unsupported")
    if provenance.get("dataset_manifest_sha256") != hashlib.sha256(dataset_manifest_raw).hexdigest():
        raise AgentPolicyError("converted dataset provenance does not bind its manifest")
    for field in (
        "converter_lock_sha256", "raw_finalization_sha256", "raw_manifest_sha256",
        "raw_payload_sha256", "raw_training_authenticity_sha256",
    ):
        if not isinstance(provenance.get(field), str) or not _HEX64.fullmatch(provenance[field]):
            raise AgentPolicyError("converted dataset provenance hash is invalid")
    for field in ("converter_code_release_id", "converter_environment_id", "raw_release_id"):
        if not isinstance(provenance.get(field), str) or not _RELEASE_ID.fullmatch(provenance[field]):
            raise AgentPolicyError("converted dataset provenance release identity is invalid")
    expected_raw_release = "raw_session-" + provenance["raw_manifest_sha256"][:20]
    if provenance["raw_release_id"] != expected_raw_release:
        raise AgentPolicyError("converted dataset raw release identity is inconsistent")
    if (
        not isinstance(provenance.get("raw_session_id"), str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", provenance["raw_session_id"]
        )
        is None
    ):
        raise AgentPolicyError("converted dataset raw session identity is invalid")
    if raw_releases_root is not None:
        raw_root = Path(raw_releases_root)
        raw_release = raw_root / provenance["raw_release_id"]
        try:
            raw_identity = validate_raw_session(
                raw_release,
                allowed_sessions_root=raw_root,
                trusted_finalization_registry=trusted_finalization_registry,
                trusted_snapshot_root=trusted_snapshot_root,
                expected_finalization_sha256=provenance[
                    "raw_finalization_sha256"
                ],
            )
        except AgentPolicyError as exc:
            raise AgentPolicyError(
                "converted dataset bound raw archive is missing or invalid"
            ) from exc
        expected_raw_identity = {
            "finalization_sha256": provenance["raw_finalization_sha256"],
            "manifest_sha256": provenance["raw_manifest_sha256"],
            "promotion_contract_version": provenance["promotion_contract_version"],
            "raw_payload_sha256": provenance["raw_payload_sha256"],
            "session_id": provenance["raw_session_id"],
            "training_authenticity_sha256": provenance[
                "raw_training_authenticity_sha256"
            ],
        }
        if any(
            raw_identity.get(field) != expected
            for field, expected in expected_raw_identity.items()
        ):
            raise AgentPolicyError(
                "converted dataset provenance does not match its raw archive"
            )
    if dataset_manifest.get("schema") != "wam_dataset_payload_manifest_v1" or not isinstance(
        dataset_manifest.get("files"), list
    ):
        raise AgentPolicyError("converted dataset payload manifest is invalid")
    dataset_names = set()
    for row in dataset_manifest["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
            raise AgentPolicyError("converted dataset file row is invalid")
        relative = _safe_relative(row["path"])
        if not relative.startswith("dataset/"):
            raise AgentPolicyError("converted payload path is outside dataset root")
        path = release / Path(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise AgentPolicyError("converted dataset file is missing or unsafe")
        if path.stat().st_size != row["size"] or _sha256_file(path) != row["sha256"]:
            raise AgentPolicyError("converted dataset payload hash or size mismatch")
        dataset_names.add(relative)
    actual_dataset = {
        path.relative_to(release).as_posix()
        for path in (release / "dataset").rglob("*") if path.is_file()
    }
    if actual_dataset != dataset_names:
        raise AgentPolicyError("converted dataset exact payload set mismatch")
    result = dict(provenance)
    result.update(release_id=envelope["release_id"], payload_sha256=envelope["payload_sha256"])
    return result


def operation_policy(local_role: str, operation: str) -> Dict[str, str]:
    mapping = {"archive_raw_session": "raw_session", "install_converted_dataset": "converted_dataset"}
    if local_role != "h200" or operation not in mapping:
        raise AgentPolicyError("data operation is allowed only on H200")
    return dict(HOST_POLICIES[local_role]["data"][mapping[operation]])


def perception_artifact_policy(
    local_role: str, data_kind: str
) -> Dict[str, object]:
    """Return one exact pointer-free H200 perception namespace policy."""
    if local_role != "h200":
        raise AgentPolicyError("perception artifact policy is available only on H200")
    values = HOST_POLICIES[local_role]["data"].get("perception")
    expected_kinds = {
        "perception_canary_evaluation",
        "perception_geometry_source",
        "perception_labels_holdout",
        "perception_labels_train",
        "perception_session2_train",
        "perception_session3_holdout",
        "perception_stage0",
    }
    if not isinstance(values, dict) or set(values) != expected_kinds:
        raise AgentPolicyError("H200 perception artifact policy is missing or invalid")
    value = values.get(data_kind)
    if not isinstance(value, dict) or set(value) != {
        "artifact_kind", "releases_root", "remote_upload", "source_role",
        "staging_root",
    }:
        raise AgentPolicyError("H200 perception data kind is not authorized")
    if (
        value.get("artifact_kind") != data_kind
        or type(value.get("remote_upload")) is not bool
        or value.get("source_role")
        != ("mac" if value.get("remote_upload") else "h200")
    ):
        raise AgentPolicyError("H200 perception artifact policy is inconsistent")
    for field in ("releases_root", "staging_root"):
        path = value.get(field)
        normalized = Path(path) if isinstance(path, str) else None
        if (
            not isinstance(path, str)
            or normalized is None
            or not normalized.is_absolute()
            or normalized.as_posix() != path
            or ".." in normalized.parts
            or "current" in normalized.parts
        ):
            raise AgentPolicyError("H200 perception artifact root is invalid")
    return dict(value)


def trusted_training_policy(local_role: str) -> Dict[str, str]:
    if local_role != "h200":
        raise AgentPolicyError("training trust policy is available only on H200")
    value = HOST_POLICIES[local_role]["data"].get("training_trust")
    if not isinstance(value, dict) or set(value) != {
        "finalization_registry", "snapshot_root",
    }:
        raise AgentPolicyError("H200 training trust policy is missing or invalid")
    result = {}
    for field in ("finalization_registry", "snapshot_root"):
        path = value.get(field)
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or Path(path).as_posix() != path
        ):
            raise AgentPolicyError("H200 training trust path is invalid")
        result[field] = path
    return result


def archive_raw_session(
    incoming: Union[str, Path],
    *,
    releases_root: Union[str, Path],
    expected_manifest_sha256: str,
    expected_finalization_sha256: str,
    trusted_finalization_registry: Union[str, Path],
    trusted_snapshot_root: Union[str, Path],
) -> Dict[str, object]:
    source = Path(incoming)
    identity = validate_raw_session(
        source,
        allowed_sessions_root=source.parent,
        trusted_finalization_registry=trusted_finalization_registry,
        trusted_snapshot_root=trusted_snapshot_root,
        expected_finalization_sha256=expected_finalization_sha256,
    )
    if (
        identity["manifest_sha256"] != expected_manifest_sha256
        or identity["finalization_sha256"] != expected_finalization_sha256
    ):
        raise AgentPolicyError("raw session identity does not match the transfer request")
    releases = Path(releases_root)
    _ensure_real_directory(releases)
    release_id = "raw_session-" + expected_manifest_sha256[:20]
    final = releases / release_id
    if os.path.lexists(str(final)):
        installed = validate_raw_session(
            final,
            allowed_sessions_root=releases,
            trusted_finalization_registry=trusted_finalization_registry,
            trusted_snapshot_root=trusted_snapshot_root,
            expected_finalization_sha256=expected_finalization_sha256,
        )
        if installed != identity:
            raise AgentPolicyError("conflicting raw session release already exists")
        shutil.rmtree(str(source), ignore_errors=True)
    else:
        verified_source = Path(identity["verified_snapshot_path"])
        staging = releases / (".staging-%s-%s" % (release_id, uuid.uuid4().hex))
        try:
            contract._materialize_training_snapshot(verified_source, staging)
            contract._rename_noreplace(staging, final)
            contract._make_training_snapshot_read_only(final)
            contract._fsync_tree_directories(final)
            contract._fsync_directory(releases)
            installed = validate_raw_session(
                final,
                allowed_sessions_root=releases,
                trusted_finalization_registry=trusted_finalization_registry,
                trusted_snapshot_root=trusted_snapshot_root,
                expected_finalization_sha256=expected_finalization_sha256,
            )
            if installed != identity:
                raise AgentPolicyError("archived raw session identity changed")
        except BaseException:
            shutil.rmtree(str(staging), ignore_errors=True)
            raise
        shutil.rmtree(str(source), ignore_errors=True)
    return {
        "final_release": final.as_posix(),
        "release_id": release_id,
        "schema": "wam_raw_session_archive_response_v1",
        "target_role": "h200",
    }


def install_converted_dataset(
    incoming: Union[str, Path],
    *,
    releases_root: Union[str, Path],
    expected_release_id: str,
    expected_payload_sha256: str,
    expected_manifest_sha256: str,
    raw_releases_root: Union[str, Path],
    trusted_finalization_registry: Union[str, Path],
    trusted_snapshot_root: Union[str, Path],
) -> Dict[str, object]:
    source = Path(incoming)
    identity = validate_converted_dataset(
        source,
        allowed_releases_root=source.parent,
        raw_releases_root=raw_releases_root,
        trusted_finalization_registry=trusted_finalization_registry,
        trusted_snapshot_root=trusted_snapshot_root,
    )
    if (
        identity["release_id"] != expected_release_id
        or identity["payload_sha256"] != expected_payload_sha256
        or _sha256_file(source / contract.MANIFEST_NAME) != expected_manifest_sha256
    ):
        raise AgentPolicyError("converted dataset identity does not match the transfer request")
    releases = Path(releases_root)
    _ensure_real_directory(releases)
    final = releases / expected_release_id
    if os.path.lexists(str(final)):
        installed = validate_converted_dataset(
            final,
            allowed_releases_root=releases,
            raw_releases_root=raw_releases_root,
            trusted_finalization_registry=trusted_finalization_registry,
            trusted_snapshot_root=trusted_snapshot_root,
        )
        if installed != identity:
            raise AgentPolicyError("conflicting converted dataset release already exists")
        shutil.rmtree(str(source), ignore_errors=True)
    else:
        contract._rename_noreplace(source, final)
        contract._fsync_directory(releases)
        validate_converted_dataset(
            final,
            allowed_releases_root=releases,
            raw_releases_root=raw_releases_root,
            trusted_finalization_registry=trusted_finalization_registry,
            trusted_snapshot_root=trusted_snapshot_root,
        )
    return {
        "final_release": final.as_posix(),
        "release_id": expected_release_id,
        "schema": "wam_converted_dataset_install_response_v1",
        "target_role": "h200",
    }


def reconcile_data_transfer(
    request: Dict[str, object], *, local_role: str
) -> Dict[str, object]:
    data_operation = request["data_operation"]
    incoming = Path(request["incoming_path"])
    releases = Path(request["releases_root"])
    trust = trusted_training_policy(local_role)
    with _data_deployment_lock(releases):
        _ensure_real_directory(releases)
        if data_operation == "archive_raw_session":
            release_id = "raw_session-" + request["expected_manifest_sha256"][:20]
        else:
            release_id = request["expected_release_id"]
        final = releases / release_id
        promoted = False
        if os.path.lexists(str(final)):
            if data_operation == "archive_raw_session":
                identity = validate_raw_session(
                    final,
                    allowed_sessions_root=releases,
                    trusted_finalization_registry=trust["finalization_registry"],
                    trusted_snapshot_root=trust["snapshot_root"],
                    expected_finalization_sha256=request[
                        "expected_finalization_sha256"
                    ],
                )
                promoted = (
                    identity["manifest_sha256"] == request["expected_manifest_sha256"]
                    and identity["finalization_sha256"]
                    == request["expected_finalization_sha256"]
                )
            else:
                identity = validate_converted_dataset(
                    final,
                    allowed_releases_root=releases,
                    raw_releases_root=operation_policy(
                        local_role, "archive_raw_session"
                    )["releases_root"],
                    trusted_finalization_registry=trust["finalization_registry"],
                    trusted_snapshot_root=trust["snapshot_root"],
                )
                promoted = (
                    identity["release_id"] == request["expected_release_id"]
                    and identity["payload_sha256"] == request["expected_payload_sha256"]
                    and _sha256_file(final / contract.MANIFEST_NAME)
                    == request["expected_manifest_sha256"]
                )
            if not promoted:
                raise AgentPolicyError("reconciled data final release has conflicting identity")
        marker = _reservation_marker_path(incoming)
        marker_present = os.path.lexists(str(marker))
        incoming_present = os.path.lexists(str(incoming))
        if marker_present:
            _require_owned_reservation(
                incoming, reservation_id=request["reservation_id"],
                target_role=local_role, operation=data_operation,
            )
            if incoming_present:
                if incoming.is_symlink() or not incoming.is_dir():
                    raise AgentPolicyError("owned data incoming path is unsafe")
                shutil.rmtree(str(incoming))
            _remove_owned_reservation_marker(
                incoming, reservation_id=request["reservation_id"],
                target_role=local_role, operation=data_operation,
            )
        elif incoming_present:
            raise AgentPolicyError("data incoming ownership marker is missing")
        return {
            "final_release": final.as_posix() if promoted else None,
            "incoming_present": os.path.lexists(str(incoming)),
            "marker_present": os.path.lexists(str(marker)),
            "schema": "wam_remote_data_reconcile_response_v1",
            "status": "promoted" if promoted else "cleaned",
            "target_role": local_role,
        }


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_regular_at(parent_descriptor: int, name: str) -> str:
    descriptor: Optional[int] = None
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise AgentPolicyError("perception manifest is not a stable regular file")
        digest = _sha256_fd(descriptor)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(opened) != identity(after):
            raise AgentPolicyError("perception manifest changed while hashing")
        return digest
    except OSError as exc:
        raise AgentPolicyError("perception manifest is missing or unsafe") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verify_perception_release_fd(
    release_descriptor: int,
    request: Dict[str, object],
    *,
    expected_target_role: Optional[str],
) -> tuple:
    try:
        envelope = contract.verify_release_fd(release_descriptor)
        manifest_sha256 = _sha256_regular_at(
            release_descriptor, contract.MANIFEST_NAME
        )
        marker = contract._load_completion_marker_fd(release_descriptor)
        if contract.verify_release_fd(release_descriptor) != envelope:
            raise AgentPolicyError("perception release changed during verification")
    except AgentPolicyError:
        raise
    except (contract.ArtifactError, OSError, ValueError) as exc:
        raise AgentPolicyError(
            "perception release verification failed: %s" % exc
        ) from exc
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_kind") != request["data_kind"]
        or payload.get("source_role") != request["source_role"]
        or envelope.get("release_id") != request["expected_release_id"]
        or envelope.get("payload_sha256") != request["expected_payload_sha256"]
        or manifest_sha256 != request["expected_manifest_sha256"]
        or not isinstance(marker, dict)
        or marker.get("source_role") != request["source_role"]
        or (
            expected_target_role is not None
            and marker.get("target_role") != expected_target_role
        )
    ):
        raise AgentPolicyError("perception release identity does not match the request")
    return envelope, manifest_sha256


def _open_release_child(parent_descriptor: int, name: str) -> int:
    descriptor: Optional[int] = None
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(before.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise AgentPolicyError("perception release directory is unsafe")
        return descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _cleanup_owned_perception_incoming(
    incoming: Path, *, reservation_id: str, local_role: str
) -> None:
    marker = _reservation_marker_path(incoming)
    marker_present = os.path.lexists(str(marker))
    incoming_present = os.path.lexists(str(incoming))
    if not marker_present:
        if incoming_present:
            raise AgentPolicyError("perception incoming ownership marker is missing")
        return
    _require_owned_reservation(
        incoming,
        reservation_id=reservation_id,
        target_role=local_role,
        operation="install_perception_artifact",
    )
    if incoming_present:
        if incoming.is_symlink() or not incoming.is_dir():
            raise AgentPolicyError("owned perception incoming path is unsafe")
        _remove_real_tree_path(incoming)
    _remove_owned_reservation_marker(
        incoming,
        reservation_id=reservation_id,
        target_role=local_role,
        operation="install_perception_artifact",
    )


def install_perception_artifact(
    request: Dict[str, object], *, local_role: str
) -> Dict[str, object]:
    policy = perception_artifact_policy(local_role, str(request["data_kind"]))
    incoming = Path(str(request["incoming_path"]))
    releases = Path(str(request["releases_root"]))
    final = releases / str(request["expected_release_id"])
    created_release = False
    with _data_deployment_lock(releases):
        _ensure_real_directory(Path(str(policy["staging_root"])))
        _ensure_real_directory(releases)
        _require_owned_reservation(
            incoming,
            reservation_id=str(request["reservation_id"]),
            target_role=local_role,
            operation="install_perception_artifact",
        )
        incoming_descriptor = _open_real_directory(incoming, create=False)
        releases_descriptor: Optional[int] = None
        try:
            envelope, _manifest_sha256 = _verify_perception_release_fd(
                incoming_descriptor, request, expected_target_role=None
            )
            releases_descriptor = _open_real_directory(releases, create=False)
            try:
                os.stat(
                    final.name,
                    dir_fd=releases_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                try:
                    contract.write_release_at(
                        incoming_descriptor,
                        envelope,
                        releases_descriptor,
                        final.name,
                        target_role=local_role,
                    )
                    created_release = True
                except contract.ConflictError as exc:
                    raise AgentPolicyError(
                        "conflicting perception release appeared during publication"
                    ) from exc
            final_descriptor: Optional[int] = None
            try:
                final_descriptor = _open_release_child(
                    releases_descriptor, final.name
                )
                _verify_perception_release_fd(
                    final_descriptor, request, expected_target_role=local_role
                )
            except BaseException as exc:
                if created_release:
                    raise
                raise AgentPolicyError(
                    "conflicting perception final release is invalid"
                ) from exc
            finally:
                if final_descriptor is not None:
                    os.close(final_descriptor)
        finally:
            if releases_descriptor is not None:
                os.close(releases_descriptor)
            os.close(incoming_descriptor)
        _cleanup_owned_perception_incoming(
            incoming,
            reservation_id=str(request["reservation_id"]),
            local_role=local_role,
        )
    return {
        "artifact_kind": request["data_kind"],
        "created_release": created_release,
        "final_release": final.as_posix(),
        "manifest_sha256": request["expected_manifest_sha256"],
        "payload_sha256": request["expected_payload_sha256"],
        "release_id": request["expected_release_id"],
        "schema": "wam_perception_artifact_install_response_v1",
        "source_role": request["source_role"],
        "target_role": local_role,
    }


def reconcile_perception_transfer(
    request: Dict[str, object], *, local_role: str
) -> Dict[str, object]:
    incoming = Path(str(request["incoming_path"]))
    releases = Path(str(request["releases_root"]))
    final = releases / str(request["expected_release_id"])
    promoted = False
    with _data_deployment_lock(releases):
        _ensure_real_directory(releases)
        releases_descriptor = _open_real_directory(releases, create=False)
        final_descriptor: Optional[int] = None
        try:
            try:
                os.stat(
                    final.name,
                    dir_fd=releases_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                try:
                    final_descriptor = _open_release_child(
                        releases_descriptor, final.name
                    )
                    _verify_perception_release_fd(
                        final_descriptor, request, expected_target_role=local_role
                    )
                    promoted = True
                except BaseException as exc:
                    raise AgentPolicyError(
                        "reconciled perception final release has conflicting identity"
                    ) from exc
        finally:
            if final_descriptor is not None:
                os.close(final_descriptor)
            os.close(releases_descriptor)
        _cleanup_owned_perception_incoming(
            incoming,
            reservation_id=str(request["reservation_id"]),
            local_role=local_role,
        )
        marker = _reservation_marker_path(incoming)
        return {
            "artifact_kind": request["data_kind"],
            "final_release": final.as_posix() if promoted else None,
            "incoming_present": os.path.lexists(str(incoming)),
            "marker_present": os.path.lexists(str(marker)),
            "schema": "wam_remote_perception_reconcile_response_v1",
            "status": "promoted" if promoted else "cleaned",
            "target_role": local_role,
        }


def _inventory_tree_descriptor(root_descriptor: int) -> str:
    rows = []

    def walk(descriptor: int, prefix: str) -> None:
        directory_before = os.fstat(descriptor)
        for name in sorted(os.listdir(descriptor)):
            relative = "%s/%s" % (prefix, name) if prefix else name
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            mode = format(stat.S_IMODE(info.st_mode), "04o")
            if stat.S_ISDIR(info.st_mode):
                rows.append({"mode": mode, "path": relative, "type": "directory"})
                child = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                child = os.open(
                    name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    opened = os.fstat(child)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
                    ):
                        raise AgentPolicyError("legacy inventory entry changed during open")
                    digest = _sha256_fd(child)
                    closed = os.fstat(child)
                    identity = lambda value: (
                        value.st_dev, value.st_ino, value.st_size,
                        value.st_mtime_ns, value.st_ctime_ns,
                    )
                    if identity(opened) != identity(closed):
                        raise AgentPolicyError("legacy inventory file changed during hashing")
                    rows.append({
                        "mode": mode,
                        "path": relative,
                        "sha256": digest,
                        "size": opened.st_size,
                        "type": "file",
                    })
                finally:
                    os.close(child)
            else:
                raise AgentPolicyError("legacy auscultation contains a symlink or non-regular entry")
        directory_after = os.fstat(descriptor)
        if (
            directory_before.st_dev, directory_before.st_ino,
            directory_before.st_mtime_ns, directory_before.st_ctime_ns,
        ) != (
            directory_after.st_dev, directory_after.st_ino,
            directory_after.st_mtime_ns, directory_after.st_ctime_ns,
        ):
            raise AgentPolicyError("legacy inventory directory changed during hashing")

    walk(root_descriptor, "")
    projection = {"rows": rows, "schema": "wam_legacy_tree_inventory_v1"}
    return hashlib.sha256(contract.canonical_json_bytes(projection)).hexdigest()


def _inventory_real_tree(root: Path) -> str:
    """Hash an exact real-directory inventory using no-follow descriptors."""
    root_descriptor = _open_real_directory(root, create=False)
    try:
        return _inventory_tree_descriptor(root_descriptor)
    finally:
        os.close(root_descriptor)


@dataclass
class _LegacyInventoryEvidence:
    digest: str
    identity: tuple
    parent_descriptor: int
    root_descriptor: int
    root_name: str

    def close(self) -> None:
        if self.root_descriptor >= 0:
            os.close(self.root_descriptor)
            self.root_descriptor = -1
        if self.parent_descriptor >= 0:
            os.close(self.parent_descriptor)
            self.parent_descriptor = -1


def _open_legacy_inventory(root: Path) -> _LegacyInventoryEvidence:
    parent = _open_real_directory(root.parent, create=False)
    descriptor: Optional[int] = None
    try:
        identity = _path_identity_at(parent, root.name)
        if identity[2] != stat.S_IFDIR:
            raise AgentPolicyError("legacy auscultation is not a real directory")
        descriptor = os.open(
            root.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != identity[:2]:
            raise AgentPolicyError("legacy root inode changed during open")
        digest = _inventory_tree_descriptor(descriptor)
        if (
            _path_identity_at(parent, root.name) != identity
            or (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
            != identity[:2]
        ):
            raise AgentPolicyError("legacy root changed after inventory")
        return _LegacyInventoryEvidence(
            digest=digest,
            identity=identity,
            parent_descriptor=parent,
            root_descriptor=descriptor,
            root_name=root.name,
        )
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)
        raise


def _rename_verified_legacy(
    evidence: _LegacyInventoryEvidence, destination_name: str,
) -> None:
    if (
        _path_identity_at(evidence.parent_descriptor, evidence.root_name)
        != evidence.identity
        or _inventory_tree_descriptor(evidence.root_descriptor) != evidence.digest
    ):
        raise AgentPolicyError("legacy root changed between inventory and rename")
    _rename_noreplace_at(
        evidence.parent_descriptor,
        evidence.root_name,
        destination_name,
        expected_identity=evidence.identity,
    )
    if (
        _path_identity_at(evidence.parent_descriptor, destination_name)[:3]
        != evidence.identity[:3]
        or (os.fstat(evidence.root_descriptor).st_dev, os.fstat(evidence.root_descriptor).st_ino)
        != evidence.identity[:2]
        or _inventory_tree_descriptor(evidence.root_descriptor) != evidence.digest
    ):
        raise AgentPolicyError("renamed legacy backup failed inode or content verification")
    os.fsync(evidence.parent_descriptor)


def _read_current_target(current: Path) -> Optional[str]:
    if not os.path.lexists(str(current)):
        return None
    if not current.is_symlink():
        raise AgentPolicyError("production current path is not a symlink")
    return os.readlink(str(current))


def _path_identity_at(parent_descriptor: int, name: str) -> tuple:
    info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    return (
        info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode),
        info.st_mtime_ns, info.st_ctime_ns,
    )


def _pointer_identity_document(identity: tuple) -> Dict[str, int]:
    if len(identity) < 3 or identity[2] != stat.S_IFLNK:
        raise AgentPolicyError("production pointer identity is not a symlink")
    return {
        "device": int(identity[0]),
        "inode": int(identity[1]),
    }


def _pointer_identity_tuple(value: object, label: str) -> tuple:
    if not isinstance(value, dict) or set(value) != {"device", "inode"}:
        raise AgentPolicyError("%s fields are invalid" % label)
    if any(type(value[key]) is not int or value[key] < 0 for key in value):
        raise AgentPolicyError("%s values are invalid" % label)
    return value["device"], value["inode"], stat.S_IFLNK


def _read_pointer_at(parent_descriptor: int, name: str) -> tuple:
    try:
        identity = _path_identity_at(parent_descriptor, name)
    except FileNotFoundError:
        return None, None
    if identity[2] != stat.S_IFLNK:
        raise AgentPolicyError("production current path is not a symlink")
    return os.readlink(name, dir_fd=parent_descriptor), identity


def _rename_noreplace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    *,
    expected_identity: tuple,
) -> None:
    """Descriptor-relative atomic no-replace with a final source identity check."""
    _rename_noreplace_between_at(
        parent_descriptor,
        source_name,
        parent_descriptor,
        destination_name,
        expected_identity=expected_identity,
    )


def _rename_noreplace_between_at(
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
    *,
    expected_identity: tuple,
) -> None:
    """Atomically move one verified inode between held directories, no-replace."""
    if _path_identity_at(source_parent_descriptor, source_name) != expected_identity:
        raise AgentPolicyError("rename source inode changed after verification")
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is None:
            raise AgentPolicyError("renameat2 is required for production no-replace")
        function.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            source_parent_descriptor, os.fsencode(source_name),
            destination_parent_descriptor, os.fsencode(destination_name), 1,
        )
    elif sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        if function is None:
            raise AgentPolicyError("renameatx_np is required for production no-replace")
        function.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            source_parent_descriptor, os.fsencode(source_name),
            destination_parent_descriptor, os.fsencode(destination_name), 0x00000004,
        )
    else:
        raise AgentPolicyError("atomic production no-replace is unsupported")
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise contract.ConflictError("production destination already exists")
        raise OSError(error, os.strerror(error), destination_name)
    if (
        _path_identity_at(destination_parent_descriptor, destination_name)[:3]
        != expected_identity[:3]
    ):
        raise AgentPolicyError("renamed production inode identity changed")


def _exchange_at(
    parent_descriptor: int,
    first: str,
    second: str,
    *,
    expected_first_identity: tuple,
    expected_second_identity: tuple,
) -> None:
    if (
        _path_identity_at(parent_descriptor, first) != expected_first_identity
        or _path_identity_at(parent_descriptor, second) != expected_second_identity
    ):
        raise AgentPolicyError(
            "production pointer compare-and-swap identity changed before exchange"
        )
    _exchange_names_at(parent_descriptor, first, second)


def _exchange_names_at(parent_descriptor: int, first: str, second: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is None:
            raise AgentPolicyError("renameat2 is required for production pointer CAS")
        function.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor, os.fsencode(first),
            parent_descriptor, os.fsencode(second), 2,
        )
    elif sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        if function is None:
            raise AgentPolicyError("renameatx_np is required for production pointer CAS")
        function.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor, os.fsencode(first),
            parent_descriptor, os.fsencode(second), 0x00000002,
        )
    else:
        raise AgentPolicyError("atomic production pointer exchange is unsupported")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), second)


def _validate_previous_current(
    current: Path, releases: Path, target: Optional[str], *, local_role: str
) -> None:
    if target is None:
        return
    parsed = PurePosixPath(target)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise AgentPolicyError("previous production current target is outside managed releases")
    try:
        resolved = current.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AgentPolicyError("previous production current target is broken") from exc
    if resolved.parent.as_posix() != releases.resolve().as_posix():
        raise AgentPolicyError("previous production current target is outside managed releases")
    # Existing current releases predate directory sealing.  Exact verification
    # must succeed before this repair mutates directory modes; unexpected files
    # therefore remain visible and block activation.
    _seal_verified_production_release(resolved)
    marker, _raw = _read_json_file(resolved / contract.COMPLETE_NAME)
    if marker.get("target_role") != local_role:
        raise AgentPolicyError("previous production current has the wrong host role")


def _prepare_production_pointer(
    parent_descriptor: int,
    current_name: str,
    target: str,
    reservation_id: str,
) -> tuple[str, tuple]:
    if not _HEX64.fullmatch(reservation_id):
        raise AgentPolicyError("production pointer reservation is invalid")
    staging_name = ".%s.activate-staging-%s" % (current_name, uuid.uuid4().hex)
    try:
        os.symlink(target, staging_name, dir_fd=parent_descriptor)
    except FileExistsError as exc:
        raise AgentPolicyError("prepared production pointer already exists") from exc
    identity = _path_identity_at(parent_descriptor, staging_name)
    temporary_name = ".%s.activate-%s-%x-%x" % (
        current_name, reservation_id, identity[0], identity[1]
    )
    _rename_noreplace_at(
        parent_descriptor,
        staging_name,
        temporary_name,
        expected_identity=identity,
    )
    identity = _path_identity_at(parent_descriptor, temporary_name)
    observed_target, observed_identity = _read_pointer_at(
        parent_descriptor, temporary_name
    )
    if observed_target != target or observed_identity != identity:
        raise AgentPolicyError("prepared production pointer did not persist")
    os.fsync(parent_descriptor)
    return temporary_name, identity


def _set_pointer(
    current: Path,
    target: str,
    *,
    expected_previous: Optional[str],
    expected_identity: Optional[tuple],
    parent_descriptor: Optional[int] = None,
    prepared_name: Optional[str] = None,
    prepared_identity: Optional[tuple] = None,
) -> tuple:
    owned_parent = parent_descriptor is None
    parent = (
        _open_real_directory(current.parent, create=False)
        if parent_descriptor is None else parent_descriptor
    )
    temporary_name = prepared_name or (
        ".%s.activate-%s" % (current.name, uuid.uuid4().hex)
    )
    remove_temporary = True
    try:
        if prepared_name is None:
            if prepared_identity is not None:
                raise AgentPolicyError("unprepared pointer has a prepared identity")
            os.symlink(target, temporary_name, dir_fd=parent)
            temporary_identity = _path_identity_at(parent, temporary_name)
        else:
            if prepared_identity is None:
                raise AgentPolicyError("prepared pointer identity is missing")
            temporary_identity = _path_identity_at(parent, temporary_name)
            prepared_target, _observed = _read_pointer_at(parent, temporary_name)
            if (
                prepared_target != target
                or temporary_identity[:3] != prepared_identity[:3]
            ):
                remove_temporary = False
                raise AgentPolicyError("prepared production pointer identity changed")
        if expected_previous is None:
            if expected_identity is not None:
                raise AgentPolicyError("absent pointer CAS has an unexpected identity")
            _rename_noreplace_at(
                parent, temporary_name, current.name,
                expected_identity=temporary_identity,
            )
        else:
            if expected_identity is None:
                raise AgentPolicyError("existing pointer CAS lacks inode identity")
            _exchange_at(
                parent,
                temporary_name,
                current.name,
                expected_first_identity=temporary_identity,
                expected_second_identity=expected_identity,
            )
            observed_target, observed_identity = _read_pointer_at(parent, temporary_name)
            if (
                observed_target != expected_previous
                or observed_identity is None
                or observed_identity[:3] != expected_identity[:3]
            ):
                # Put the pointer observed at the exchange instant back.  The
                # third-party value is never overwritten by our new target.
                current_target, current_identity = _read_pointer_at(parent, current.name)
                if (
                    current_target != target
                    or current_identity is None
                    or current_identity[:3] != temporary_identity[:3]
                ):
                    remove_temporary = False
                    raise AgentPolicyError(
                        "production pointer diverged during failed CAS restoration"
                    )
                _exchange_at(
                    parent,
                    temporary_name,
                    current.name,
                    expected_first_identity=observed_identity,
                    expected_second_identity=current_identity,
                )
                os.fsync(parent)
                raise AgentPolicyError("production pointer compare-and-swap mismatch")
            os.unlink(temporary_name, dir_fd=parent)
        os.fsync(parent)
        installed_target, installed_identity = _read_pointer_at(parent, current.name)
        if installed_target != target or installed_identity is None:
            raise AgentPolicyError("production pointer activation did not persist")
        return installed_identity
    finally:
        if remove_temporary:
            try:
                os.unlink(temporary_name, dir_fd=parent)
            except FileNotFoundError:
                pass
        if owned_parent:
            os.close(parent)


def _restore_pointer_cas(
    current: Path, *, expected_current: str, previous: Optional[str],
    expected_identity: Optional[tuple] = None,
) -> None:
    parent = _open_real_directory(current.parent, create=False)
    try:
        actual, actual_identity = _read_pointer_at(parent, current.name)
        if actual != expected_current:
            raise AgentPolicyError("production rollback compare-and-swap mismatch")
        if (
            expected_identity is not None
            and actual_identity[:3] != expected_identity[:3]
        ):
            raise AgentPolicyError("production rollback pointer inode changed")
        if previous is None:
            os.unlink(current.name, dir_fd=parent)
            os.fsync(parent)
        else:
            _set_pointer(
                current,
                previous,
                expected_previous=expected_current,
                expected_identity=actual_identity,
                parent_descriptor=parent,
            )
    finally:
        os.close(parent)


def _release_directory_fd(parent_fd: int, name: str, label: str) -> int:
    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise AgentPolicyError("%s is missing, swapped, or symlinked" % label) from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(before.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        os.close(descriptor)
        raise AgentPolicyError("%s identity changed while opening" % label)
    return descriptor


def _seal_release_directory_descriptor(descriptor: int) -> None:
    before = os.fstat(descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise AgentPolicyError("production release directory is unsafe")
    names = sorted(os.listdir(descriptor))
    for name in names:
        if not name or "/" in name or name in (".", ".."):
            raise AgentPolicyError("production release directory entry is unsafe")
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child = _release_directory_fd(
                descriptor, name, "production release directory"
            )
            try:
                child_identity = os.fstat(child)
                _seal_release_directory_descriptor(child)
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(observed.st_mode)
                    or (observed.st_dev, observed.st_ino)
                    != (child_identity.st_dev, child_identity.st_ino)
                ):
                    raise AgentPolicyError(
                        "production release directory changed during sealing"
                    )
            finally:
                os.close(child)
        elif not stat.S_ISREG(info.st_mode):
            raise AgentPolicyError(
                "production release contains a symlink or special entry"
            )
    after_walk = os.fstat(descriptor)
    if (
        (before.st_dev, before.st_ino) != (after_walk.st_dev, after_walk.st_ino)
        or names != sorted(os.listdir(descriptor))
    ):
        raise AgentPolicyError("production release directory changed during sealing")
    os.fchmod(descriptor, 0o555)
    os.fsync(descriptor)
    sealed = os.fstat(descriptor)
    if (
        (sealed.st_dev, sealed.st_ino) != (before.st_dev, before.st_ino)
        or stat.S_IMODE(sealed.st_mode) != 0o555
    ):
        raise AgentPolicyError("production release directory seal did not persist")


def _manifest_release_directories(envelope: Dict[str, object]) -> set:
    expected = {""}
    payload = envelope.get("payload")
    rows = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise AgentPolicyError("production release manifest file rows are invalid")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise AgentPolicyError("production release manifest file path is invalid")
        parent = PurePosixPath(str(row["path"])).parent
        while parent.as_posix() != ".":
            expected.add(parent.as_posix())
            parent = parent.parent
    return expected


def _release_directory_inventory_descriptor(
    descriptor: int, relative: str = ""
) -> set:
    before = os.fstat(descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise AgentPolicyError("production release directory is unsafe")
    names = sorted(os.listdir(descriptor))
    found = {relative}
    for name in names:
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child = _release_directory_fd(
                descriptor, name, "production release directory"
            )
            try:
                child_identity = os.fstat(child)
                child_relative = name if not relative else relative + "/" + name
                found.update(
                    _release_directory_inventory_descriptor(child, child_relative)
                )
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(observed.st_mode)
                    or (observed.st_dev, observed.st_ino)
                    != (child_identity.st_dev, child_identity.st_ino)
                ):
                    raise AgentPolicyError(
                        "production release directory changed during inventory"
                    )
            finally:
                os.close(child)
        elif not stat.S_ISREG(info.st_mode):
            raise AgentPolicyError(
                "production release contains a symlink or special entry"
            )
    after = os.fstat(descriptor)
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or names != sorted(os.listdir(descriptor))
    ):
        raise AgentPolicyError("production release directory changed during inventory")
    return found


def _seal_exact_release_directories_descriptor(
    descriptor: int, expected: set
) -> None:
    actual = _release_directory_inventory_descriptor(descriptor)
    if actual != expected:
        raise AgentPolicyError(
            "production release directory exact-set mismatch; missing=%r extra=%r"
            % (sorted(expected - actual), sorted(actual - expected))
        )
    _seal_release_directory_descriptor(descriptor)
    after = _release_directory_inventory_descriptor(descriptor)
    if after != expected:
        raise AgentPolicyError(
            "production release directory exact-set changed during sealing"
        )


def _verify_release_directory_seal_descriptor(descriptor: int) -> None:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o555
    ):
        raise AgentPolicyError("production release directory seal is missing")
    names = sorted(os.listdir(descriptor))
    for name in names:
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child = _release_directory_fd(
                descriptor, name, "sealed production release directory"
            )
            try:
                child_identity = os.fstat(child)
                _verify_release_directory_seal_descriptor(child)
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(observed.st_mode)
                    or (observed.st_dev, observed.st_ino)
                    != (child_identity.st_dev, child_identity.st_ino)
                ):
                    raise AgentPolicyError(
                        "sealed production release directory changed"
                    )
            finally:
                os.close(child)
        elif not stat.S_ISREG(info.st_mode):
            raise AgentPolicyError(
                "sealed production release contains an unsafe entry"
            )
    after = os.fstat(descriptor)
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or names != sorted(os.listdir(descriptor))
    ):
        raise AgentPolicyError("sealed production release directory changed")


def _with_production_release_descriptor(
    release: Path, operation: Callable[[int], None]
) -> None:
    releases = release.parent
    if (
        not release.is_absolute()
        or not _RELEASE_ID.fullmatch(release.name)
        or release.is_symlink()
    ):
        raise AgentPolicyError("production release path is unsafe")
    releases_fd = _open_real_directory(releases, create=False)
    release_fd = -1
    try:
        release_fd = _release_directory_fd(
            releases_fd, release.name, "production release"
        )
        opened = os.fstat(release_fd)
        operation(release_fd)
        observed = os.stat(
            release.name, dir_fd=releases_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(observed.st_mode)
            or (observed.st_dev, observed.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise AgentPolicyError("production release path changed during operation")
        os.fsync(releases_fd)
    finally:
        if release_fd >= 0:
            os.close(release_fd)
        os.close(releases_fd)


def _seal_verified_production_release(release: Path) -> Dict[str, object]:
    envelope = contract.verify_release(release)
    expected_directories = _manifest_release_directories(envelope)
    _with_production_release_descriptor(
        release,
        lambda descriptor: _seal_exact_release_directories_descriptor(
            descriptor, expected_directories
        ),
    )
    verified = contract.verify_release(release)
    if verified != envelope:
        raise AgentPolicyError("sealed production release identity changed")
    _with_production_release_descriptor(
        release, _verify_release_directory_seal_descriptor
    )
    return verified


def _verify_production_release_seal(release: Path) -> None:
    _with_production_release_descriptor(
        release, _verify_release_directory_seal_descriptor
    )


def _install_managed_production_release(
    incoming: Path, releases: Path, *, local_role: str, reservation_id: str
) -> tuple[Path, bool]:
    envelope = contract.verify_release(incoming)
    final = releases / str(envelope["release_id"])
    if final.parent.as_posix() != releases.as_posix():
        raise AgentPolicyError("production release is not a direct managed child")
    if os.path.lexists(str(final)):
        if final.is_symlink() or not final.is_dir():
            raise AgentPolicyError("managed production release path conflicts")
        installed = contract.verify_release(final)
        marker, _raw = _read_json_file(final / contract.COMPLETE_NAME)
        if installed != envelope or marker.get("target_role") != local_role:
            raise AgentPolicyError("managed production release identity conflicts")
        _seal_verified_production_release(final)
        return final, False
    staging = releases / (
        ".staging-%s-%s" % (envelope["release_id"], reservation_id)
    )
    if os.path.lexists(str(staging)):
        raise AgentPolicyError("production staging path already exists")
    try:
        contract.write_release(incoming, envelope, staging, target_role=local_role)
        contract._rename_noreplace(staging, final)
        contract._fsync_directory(releases)
        _seal_verified_production_release(final)
    except BaseException:
        if os.path.lexists(str(staging)):
            _remove_real_tree_path(staging)
        raise
    return final, True


def _postverify_production(
    *,
    final: Path,
    current: Path,
    local_role: str,
    frozen_release_id: str,
    frozen_payload_sha256: str,
    expected_manifest_sha256: str,
    compatibility: Optional[Path],
    legacy: Optional[Path],
    legacy_inventory_sha256: Optional[str] = None,
    legacy_identity: Optional[tuple] = None,
) -> None:
    envelope = contract.verify_release(final)
    _verify_production_release_seal(final)
    marker, _raw = _read_json_file(final / contract.COMPLETE_NAME)
    expected_target = os.path.relpath(str(final), str(current.parent))
    if (
        envelope.get("release_id") != frozen_release_id
        or envelope.get("payload_sha256") != frozen_payload_sha256
        or _sha256_file(final / contract.MANIFEST_NAME) != expected_manifest_sha256
        or marker.get("target_role") != local_role
        or _read_current_target(current) != expected_target
        or current.resolve() != final.resolve()
    ):
        raise AgentPolicyError("production post-activation identity verification failed")
    if local_role == "linbox":
        if compatibility is None or not compatibility.is_symlink():
            raise AgentPolicyError("linbox compatibility link is missing")
        if os.readlink(str(compatibility)) != ".wam-current/auscultation":
            raise AgentPolicyError("linbox compatibility link target is invalid")
        expected_code = final / "auscultation"
        if not expected_code.is_dir() or expected_code.is_symlink():
            raise AgentPolicyError("linbox managed release lacks real auscultation code")
        if compatibility.resolve() != expected_code.resolve():
            raise AgentPolicyError("linbox compatibility link resolves incorrectly")
        if legacy is not None:
            if legacy.is_symlink() or not legacy.is_dir():
                raise AgentPolicyError("linbox legacy directory was not preserved")
            parent = _open_real_directory(legacy.parent, create=False)
            try:
                observed = _path_identity_at(parent, legacy.name)
            finally:
                os.close(parent)
            if (
                legacy_identity is None
                or observed[:2] != legacy_identity[:2]
                or legacy_inventory_sha256 is None
                or _inventory_real_tree(legacy) != legacy_inventory_sha256
            ):
                raise AgentPolicyError("linbox legacy backup inode or content drifted")


_PRODUCTION_TRANSACTION_FIELDS = {
    "activated_pointer_identity", "activated_target", "compatibility_preexisting",
    "frozen_payload_sha256", "frozen_release_id", "incoming_path",
    "expected_manifest_sha256", "legacy_device", "legacy_inode",
    "legacy_inventory_sha256", "legacy_path", "migrate_real_directory",
    "prepared_pointer_name", "previous_current_target", "previous_pointer_identity",
    "request_identity_sha256", "reservation_id", "schema", "target_role",
}
_PRODUCTION_TRANSACTION_NAME = re.compile(
    r"^\.wam-production-transaction-([0-9a-f]{64})\.json$"
)
_PRODUCTION_TRANSACTION_SEAL_NAME = re.compile(
    r"^\.wam-production-transaction-([0-9a-f]{64})\.seal\.json$"
)
_PRODUCTION_PREPARED_POINTER_NAME = re.compile(
    r"^\.([A-Za-z0-9._-]+)\.activate-([0-9a-f]{64})-"
    r"([0-9a-f]+)-([0-9a-f]+)$"
)


def _production_transaction_path(base: Path, reservation_id: str) -> Path:
    if not _HEX64.fullmatch(reservation_id):
        raise AgentPolicyError("production transaction reservation is invalid")
    return base / (".wam-production-transaction-%s.json" % reservation_id)


def _assert_no_other_production_transaction(
    base: Path, *, allowed: Optional[Path] = None
) -> None:
    descriptor = _open_real_directory(base, create=False)
    try:
        found = []
        allowed_reservation = None
        if allowed is not None:
            match = _PRODUCTION_TRANSACTION_NAME.fullmatch(allowed.name)
            allowed_reservation = match.group(1) if match else None
        for name in os.listdir(descriptor):
            match = (
                _PRODUCTION_TRANSACTION_NAME.fullmatch(name)
                or _PRODUCTION_TRANSACTION_SEAL_NAME.fullmatch(name)
            )
            if match:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    raise AgentPolicyError("production transaction path is unsafe")
                if allowed_reservation is None or match.group(1) != allowed_reservation:
                    found.append(name)
        if found:
            raise AgentPolicyError(
                "unresolved production transaction requires reconciliation"
            )
    finally:
        os.close(descriptor)


def _validate_production_transaction(value: object) -> Dict[str, object]:
    if not isinstance(value, dict) or set(value) != _PRODUCTION_TRANSACTION_FIELDS:
        raise AgentPolicyError("production transaction fields are invalid")
    if (
        value.get("schema") != "wam_production_transaction_v2"
        or value.get("target_role") not in HOST_POLICIES
        or not isinstance(value.get("reservation_id"), str)
        or not _HEX64.fullmatch(value["reservation_id"])
        or not isinstance(value.get("frozen_release_id"), str)
        or not _RELEASE_ID.fullmatch(value["frozen_release_id"])
        or not isinstance(value.get("frozen_payload_sha256"), str)
        or not _HEX64.fullmatch(value["frozen_payload_sha256"])
        or not isinstance(value.get("expected_manifest_sha256"), str)
        or not _HEX64.fullmatch(value["expected_manifest_sha256"])
        or not isinstance(value.get("request_identity_sha256"), str)
        or not _HEX64.fullmatch(value["request_identity_sha256"])
        or not isinstance(value.get("incoming_path"), str)
        or not isinstance(value.get("activated_target"), str)
        or type(value.get("compatibility_preexisting")) is not bool
        or type(value.get("migrate_real_directory")) is not bool
    ):
        raise AgentPolicyError("production transaction identity is invalid")
    for key in (
        "legacy_inventory_sha256", "legacy_path", "previous_current_target",
    ):
        if value[key] is not None and not isinstance(value[key], str):
            raise AgentPolicyError("production transaction optional field is invalid")
    if value["legacy_inventory_sha256"] is not None and not _HEX64.fullmatch(
        value["legacy_inventory_sha256"]
    ):
        raise AgentPolicyError("production transaction legacy hash is invalid")
    for key in ("legacy_device", "legacy_inode"):
        if value[key] is not None and (
            type(value[key]) is not int or value[key] < 0
        ):
            raise AgentPolicyError("production transaction legacy inode is invalid")
    _pointer_identity_tuple(
        value.get("activated_pointer_identity"),
        "production activated pointer identity",
    )
    previous_identity = value.get("previous_pointer_identity")
    if previous_identity is not None:
        _pointer_identity_tuple(
            previous_identity, "production previous pointer identity"
        )
    if (
        not isinstance(value.get("prepared_pointer_name"), str)
        or "/" in value["prepared_pointer_name"]
        or "\x00" in value["prepared_pointer_name"]
    ):
        raise AgentPolicyError("production prepared pointer name is invalid")
    return dict(value)


def _production_request_identity(request: Dict[str, object]) -> str:
    projection = {
        "capability": _PRODUCTION_CAPABILITY,
        "expected_manifest_sha256": request["expected_manifest_sha256"],
        "frozen_payload_sha256": request["frozen_payload_sha256"],
        "frozen_release_id": request["frozen_release_id"],
        "incoming_path": request["incoming_path"],
        "reservation_id": request["reservation_id"],
        "schema": "wam_production_transaction_request_identity_v1",
        "target_role": request["target_role"],
    }
    return hashlib.sha256(contract.canonical_json_bytes(projection)).hexdigest()


def _validate_managed_previous_target(
    target: str, *, base: Path, releases: Path, local_role: str,
) -> Path:
    parsed = PurePosixPath(target)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != target:
        raise AgentPolicyError("journal previous current target is outside managed releases")
    candidate = Path(os.path.normpath(str(base / Path(*parsed.parts))))
    if (
        candidate.parent.as_posix() != releases.as_posix()
        or not _RELEASE_ID.fullmatch(candidate.name)
        or os.path.relpath(str(candidate), str(base)) != target
        or candidate.is_symlink()
        or not candidate.is_dir()
    ):
        raise AgentPolicyError("journal previous current target is outside managed releases")
    contract.verify_release(candidate)
    marker, _raw = _read_json_file(candidate / contract.COMPLETE_NAME)
    if marker.get("target_role") != local_role:
        raise AgentPolicyError("journal previous current release has the wrong role")
    return candidate


def _validate_production_transaction_for_request(
    value: Dict[str, object], request: Dict[str, object], *, local_role: str,
) -> Dict[str, object]:
    transaction = _validate_production_transaction(value)
    paths = HOST_POLICIES[local_role]["production"]
    current = Path(paths["current_link"])
    base = current.parent
    releases = Path(paths["releases_root"])
    incoming = Path(paths["incoming_root"]) / Path(str(request["incoming_path"])).name
    final = releases / str(request["frozen_release_id"])
    expected_activated = os.path.relpath(str(final), str(base))
    prepared_match = _PRODUCTION_PREPARED_POINTER_NAME.fullmatch(
        str(transaction["prepared_pointer_name"])
    )
    activated_pointer_identity = _pointer_identity_tuple(
        transaction["activated_pointer_identity"],
        "production activated pointer identity",
    )
    if (
        transaction["target_role"] != local_role
        or transaction["reservation_id"] != request["reservation_id"]
        or transaction["frozen_release_id"] != request["frozen_release_id"]
        or transaction["frozen_payload_sha256"] != request["frozen_payload_sha256"]
        or transaction["expected_manifest_sha256"] != request["expected_manifest_sha256"]
        or transaction["incoming_path"] != request["incoming_path"]
        or transaction["incoming_path"] != incoming.as_posix()
        or transaction["activated_target"] != expected_activated
        or prepared_match is None
        or prepared_match.group(1) != current.name
        or prepared_match.group(2) != request["reservation_id"]
        or int(prepared_match.group(3), 16) != activated_pointer_identity[0]
        or int(prepared_match.group(4), 16) != activated_pointer_identity[1]
        or transaction["request_identity_sha256"] != _production_request_identity(request)
    ):
        raise AgentPolicyError("production transaction is not bound to the request")
    previous = transaction["previous_current_target"]
    previous_identity = transaction["previous_pointer_identity"]
    if (previous is None) != (previous_identity is None):
        raise AgentPolicyError(
            "production transaction previous pointer target and identity are inconsistent"
        )
    if previous is not None:
        _validate_managed_previous_target(
            str(previous), base=base, releases=releases, local_role=local_role,
        )
    migrate = bool(transaction["migrate_real_directory"])
    legacy_values = (
        transaction["legacy_path"], transaction["legacy_inventory_sha256"],
        transaction["legacy_device"], transaction["legacy_inode"],
    )
    if local_role == "h200":
        if (
            migrate or transaction["compatibility_preexisting"] is not False
            or any(item is not None for item in legacy_values)
        ):
            raise AgentPolicyError("H200 transaction contains linbox migration state")
    elif migrate:
        legacy_path, legacy_hash, legacy_device, legacy_inode = legacy_values
        if (
            transaction["compatibility_preexisting"] is not False
            or not isinstance(legacy_path, str)
            or not isinstance(legacy_hash, str)
            or type(legacy_device) is not int
            or type(legacy_inode) is not int
        ):
            raise AgentPolicyError("linbox transaction legacy fields are incomplete")
        legacy = Path(legacy_path)
        if (
            legacy.parent.as_posix() != base.as_posix()
            or re.fullmatch(
                r"auscultation\.legacy-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{16}",
                legacy.name,
            ) is None
            or not legacy.name.endswith("-" + legacy_hash[:16])
        ):
            raise AgentPolicyError("linbox transaction legacy path is noncanonical")
    elif any(item is not None for item in legacy_values):
        raise AgentPolicyError("non-migration transaction contains legacy state")
    return transaction


def _create_production_transaction(
    path: Path,
    value: Dict[str, object],
    *,
    owned_state_descriptor: int,
) -> None:
    transaction = _validate_production_transaction(value)
    raw = contract.canonical_json_bytes(transaction)
    parent = _open_real_directory(path.parent, create=False)
    state_parent = owned_state_descriptor
    descriptor: Optional[int] = None
    staging_name = path.name + ".pending-staging-" + uuid.uuid4().hex
    temporary_name = staging_name
    temporary_identity: Optional[tuple] = None
    try:
        descriptor = os.open(
            staging_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=state_parent,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_identity = _path_identity_at(state_parent, temporary_name)
        _rename_noreplace_between_at(
            state_parent,
            temporary_name,
            parent,
            path.name,
            expected_identity=temporary_identity,
        )
        os.fsync(state_parent)
        os.fsync(parent)
    except (FileExistsError, contract.ConflictError) as exc:
        raise AgentPolicyError(
            "production transaction already exists; reconcile before retry"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            observed = _path_identity_at(state_parent, temporary_name)
        except FileNotFoundError:
            observed = None
        if (
            temporary_identity is not None
            and observed is not None
            and observed[:3] == temporary_identity[:3]
        ):
            try:
                os.unlink(temporary_name, dir_fd=state_parent)
                os.fsync(state_parent)
            except FileNotFoundError:
                pass
        os.close(parent)


def _read_production_transaction(path: Path) -> Dict[str, object]:
    value, raw = _read_json_file(path)
    transaction = _validate_production_transaction(value)
    if raw != contract.canonical_json_bytes(transaction):
        raise AgentPolicyError("production transaction is not canonical")
    return transaction


def _remove_production_transaction(path: Path) -> None:
    parent = _open_real_directory(path.parent, create=False)
    try:
        info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise AgentPolicyError("production transaction path is unsafe")
        os.unlink(path.name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)


def _cleanup_prepared_production_pointer(
    transaction: Dict[str, object], current: Path,
) -> None:
    parent = _open_real_directory(current.parent, create=False)
    name = str(transaction["prepared_pointer_name"])
    try:
        try:
            target, identity = _read_pointer_at(parent, name)
        except FileNotFoundError:
            return
        if identity is None:
            return
        activated_identity = _pointer_identity_tuple(
            transaction["activated_pointer_identity"],
            "production activated pointer identity",
        )
        previous_identity_raw = transaction["previous_pointer_identity"]
        previous_identity = (
            None
            if previous_identity_raw is None
            else _pointer_identity_tuple(
                previous_identity_raw, "production previous pointer identity"
            )
        )
        owned = (
            target == transaction["activated_target"]
            and identity[:3] == activated_identity[:3]
        ) or (
            previous_identity is not None
            and target == transaction["previous_current_target"]
            and identity[:3] == previous_identity[:3]
        )
        if not owned:
            raise AgentPolicyError(
                "prepared production pointer identity diverged during recovery"
            )
        os.unlink(name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)


def _rollback_production_transaction(
    value: Dict[str, object], request: Dict[str, object]
) -> None:
    local_role = str(request["target_role"])
    transaction = _validate_production_transaction_for_request(
        value, request, local_role=local_role,
    )
    paths = HOST_POLICIES[local_role]["production"]
    current = Path(paths["current_link"])
    base = current.parent
    parent = _open_real_directory(base, create=False)
    try:
        actual, actual_identity = _read_pointer_at(parent, current.name)
    finally:
        os.close(parent)
    activated = str(transaction["activated_target"])
    activated_identity = _pointer_identity_tuple(
        transaction["activated_pointer_identity"],
        "production activated pointer identity",
    )
    previous = transaction["previous_current_target"]
    if actual == activated:
        _restore_pointer_cas(
            current, expected_current=activated,
            previous=previous if isinstance(previous, str) else None,
            expected_identity=activated_identity,
        )
    elif actual != previous:
        raise AgentPolicyError("production current diverged before transaction rollback")
    elif previous is not None:
        expected_previous_identity = _pointer_identity_tuple(
            transaction["previous_pointer_identity"],
            "production previous pointer identity",
        )
        if actual_identity[:3] != expected_previous_identity[:3]:
            raise AgentPolicyError(
                "production previous pointer inode diverged before rollback"
            )
    _cleanup_prepared_production_pointer(transaction, current)
    if local_role != "linbox":
        return
    compatibility = base / "auscultation"
    preexisting = bool(transaction["compatibility_preexisting"])
    migrate = bool(transaction["migrate_real_directory"])
    legacy_raw = transaction["legacy_path"]
    legacy = Path(legacy_raw) if isinstance(legacy_raw, str) else None
    if not preexisting and compatibility.is_symlink():
        if os.readlink(str(compatibility)) != ".wam-current/auscultation":
            raise AgentPolicyError("linbox compatibility diverged before rollback")
        compatibility.unlink()
        contract._fsync_directory(base)
    if migrate:
        if legacy is None or legacy.parent.as_posix() != base.as_posix():
            raise AgentPolicyError("linbox legacy rollback path is invalid")
        legacy_exists = os.path.lexists(str(legacy))
        real_exists = os.path.lexists(str(compatibility))
        expected_pair = (
            int(transaction["legacy_device"]), int(transaction["legacy_inode"]),
        )
        expected_hash = str(transaction["legacy_inventory_sha256"])
        if legacy_exists and not real_exists:
            if legacy.is_symlink() or not legacy.is_dir():
                raise AgentPolicyError("linbox legacy rollback source is unsafe")
            parent = _open_real_directory(base, create=False)
            try:
                identity = _path_identity_at(parent, legacy.name)
                if identity[:2] != expected_pair or _inventory_real_tree(legacy) != expected_hash:
                    raise AgentPolicyError("linbox legacy rollback source identity drifted")
                _rename_noreplace_at(
                    parent, legacy.name, compatibility.name,
                    expected_identity=identity,
                )
                os.fsync(parent)
            finally:
                os.close(parent)
        elif real_exists:
            if compatibility.is_symlink() or not compatibility.is_dir():
                raise AgentPolicyError("linbox restored real directory is unsafe")
            evidence = _open_legacy_inventory(compatibility)
            try:
                if (
                    evidence.identity[:2] != expected_pair
                    or evidence.digest != expected_hash
                ):
                    raise AgentPolicyError("linbox real directory changed before rollback")
            finally:
                evidence.close()
        else:
            raise AgentPolicyError("linbox legacy transaction state is ambiguous")


def _remove_tree_at(parent_descriptor: int, name: str) -> None:
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise AgentPolicyError("owned production tree path is unsafe") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(before.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        os.close(descriptor)
        raise AgentPolicyError("owned production tree identity changed while opening")
    try:
        # Sealed releases are intentionally non-writable.  Only the directory
        # reached through this held nofollow descriptor is temporarily opened
        # for owner cleanup; payload file modes remain untouched.
        os.fchmod(descriptor, 0o700)
        os.fsync(descriptor)
        for child in os.listdir(descriptor):
            info = os.stat(child, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                _remove_tree_at(descriptor, child)
            else:
                os.unlink(child, dir_fd=descriptor)
        os.fsync(descriptor)
        observed = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or (observed.st_dev, observed.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise AgentPolicyError("owned production tree changed during cleanup")
    finally:
        os.close(descriptor)
    os.rmdir(name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


def _remove_real_tree_path(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise AgentPolicyError("owned production tree path is unsafe")
    parent = _open_real_directory(path.parent, create=False)
    try:
        _remove_tree_at(parent, path.name)
        os.fsync(parent)
    finally:
        os.close(parent)


def _cleanup_owned_production_incoming(
    incoming: Path, *, reservation_id: str, local_role: str
) -> None:
    marker = _reservation_marker_path(incoming)
    if not os.path.lexists(str(marker)):
        if os.path.lexists(str(incoming)):
            raise AgentPolicyError("production incoming exists without its ownership marker")
        return
    _require_owned_reservation(
        incoming,
        reservation_id=reservation_id,
        target_role=local_role,
        operation="activate_production_release",
    )
    if os.path.lexists(str(incoming)):
        if incoming.is_symlink() or not incoming.is_dir():
            raise AgentPolicyError("production incoming path is unsafe")
        _remove_real_tree_path(incoming)
    _remove_owned_reservation_marker(
        incoming,
        reservation_id=reservation_id,
        target_role=local_role,
        operation="activate_production_release",
    )


def activate_production_release(
    request: Dict[str, object], *, local_role: str
) -> Dict[str, object]:
    paths = HOST_POLICIES[local_role]["production"]
    incoming = Path(str(request["incoming_path"]))
    releases = Path(paths["releases_root"])
    current = Path(paths["current_link"])
    base = current.parent
    compatibility = base / "auscultation" if local_role == "linbox" else None
    previous_current: Optional[str] = None
    activated_target: Optional[str] = None
    legacy: Optional[Path] = None
    legacy_hash: Optional[str] = None
    legacy_identity: Optional[tuple] = None
    legacy_evidence: Optional[_LegacyInventoryEvidence] = None
    migrated_real_directory = False
    final: Optional[Path] = None
    pointer_parent_descriptor: Optional[int] = None
    owned_state_descriptor: Optional[int] = None
    previous_pointer_identity: Optional[tuple] = None
    activated_pointer_identity: Optional[tuple] = None
    prepared_pointer_name: Optional[str] = None
    transaction_path = _production_transaction_path(
        base, str(request["reservation_id"])
    )
    transaction: Optional[Dict[str, object]] = None
    transaction_published = False
    with _production_deployment_lock(local_role):
        _ensure_real_directory(Path(paths["incoming_root"]))
        _ensure_real_directory(releases)
        _assert_no_other_production_transaction(base)
        _require_owned_reservation(
            incoming,
            reservation_id=str(request["reservation_id"]),
            target_role=local_role,
            operation="activate_production_release",
        )
        try:
            envelope = contract.verify_release(incoming)
            if (
                envelope.get("release_id") != request["frozen_release_id"]
                or envelope.get("payload_sha256") != request["frozen_payload_sha256"]
                or _sha256_file(incoming / contract.MANIFEST_NAME)
                != request["expected_manifest_sha256"]
            ):
                raise AgentPolicyError("production incoming does not match frozen identities")
            final, created_release = _install_managed_production_release(
                incoming, releases, local_role=local_role,
                reservation_id=str(request["reservation_id"]),
            )
            owned_state_descriptor = _open_real_directory(incoming, create=False)
            pointer_parent_descriptor = _open_real_directory(base, create=False)
            previous_current, previous_pointer_identity = _read_pointer_at(
                pointer_parent_descriptor, current.name
            )
            _validate_previous_current(
                current, releases, previous_current, local_role=local_role
            )
            compatibility_preexisting = False
            if local_role == "linbox":
                assert compatibility is not None
                if os.path.lexists(str(compatibility)):
                    if compatibility.is_symlink():
                        if os.readlink(str(compatibility)) != ".wam-current/auscultation":
                            raise AgentPolicyError("linbox compatibility symlink is outside policy")
                        compatibility_preexisting = True
                    elif compatibility.is_dir():
                        legacy_evidence = _open_legacy_inventory(compatibility)
                        legacy_hash = legacy_evidence.digest
                        legacy_identity = legacy_evidence.identity
                        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        legacy = base / (
                            "auscultation.legacy-%s-%s" % (timestamp, legacy_hash[:16])
                        )
                        migrated_real_directory = True
                    else:
                        raise AgentPolicyError("linbox auscultation path is unsafe")
            activated_target = os.path.relpath(str(final), str(current.parent))
            prepared_pointer_name, activated_pointer_identity = (
                _prepare_production_pointer(
                    owned_state_descriptor,
                    current.name,
                    activated_target,
                    str(request["reservation_id"]),
                )
            )
            transaction = {
                "activated_pointer_identity": _pointer_identity_document(
                    activated_pointer_identity
                ),
                "activated_target": activated_target,
                "compatibility_preexisting": compatibility_preexisting,
                "frozen_payload_sha256": request["frozen_payload_sha256"],
                "frozen_release_id": request["frozen_release_id"],
                "incoming_path": incoming.as_posix(),
                "expected_manifest_sha256": request["expected_manifest_sha256"],
                "legacy_device": (
                    legacy_identity[0] if legacy_identity is not None else None
                ),
                "legacy_inode": (
                    legacy_identity[1] if legacy_identity is not None else None
                ),
                "legacy_inventory_sha256": legacy_hash,
                "legacy_path": legacy.as_posix() if legacy is not None else None,
                "migrate_real_directory": migrated_real_directory,
                "prepared_pointer_name": prepared_pointer_name,
                "previous_current_target": previous_current,
                "previous_pointer_identity": (
                    None
                    if previous_pointer_identity is None
                    else _pointer_identity_document(previous_pointer_identity)
                ),
                "request_identity_sha256": _production_request_identity(request),
                "reservation_id": request["reservation_id"],
                "schema": "wam_production_transaction_v2",
                "target_role": local_role,
            }
            transaction = _validate_production_transaction_for_request(
                transaction, request, local_role=local_role,
            )
            _create_production_transaction(
                transaction_path,
                transaction,
                owned_state_descriptor=owned_state_descriptor,
            )
            transaction_published = True
            _rename_noreplace_between_at(
                owned_state_descriptor,
                prepared_pointer_name,
                pointer_parent_descriptor,
                prepared_pointer_name,
                expected_identity=activated_pointer_identity,
            )
            os.fsync(owned_state_descriptor)
            os.fsync(pointer_parent_descriptor)
            if migrated_real_directory:
                assert legacy is not None and legacy_evidence is not None
                _rename_verified_legacy(legacy_evidence, legacy.name)
                legacy_evidence.close()
                legacy_evidence = None
            installed_pointer_identity = _set_pointer(
                current,
                activated_target,
                expected_previous=previous_current,
                expected_identity=previous_pointer_identity,
                parent_descriptor=pointer_parent_descriptor,
                prepared_name=prepared_pointer_name,
                prepared_identity=activated_pointer_identity,
            )
            if installed_pointer_identity[:3] != activated_pointer_identity[:3]:
                raise AgentPolicyError(
                    "activated production pointer identity did not match journal"
                )
            os.close(pointer_parent_descriptor)
            pointer_parent_descriptor = None
            os.close(owned_state_descriptor)
            owned_state_descriptor = None
            if local_role == "linbox":
                assert compatibility is not None
                if not os.path.lexists(str(compatibility)):
                    temporary = base / (".auscultation.compat-%s" % uuid.uuid4().hex)
                    temporary.symlink_to(".wam-current/auscultation")
                    try:
                        contract._rename_noreplace(temporary, compatibility)
                        contract._fsync_directory(base)
                    finally:
                        try:
                            temporary.unlink()
                        except FileNotFoundError:
                            pass
            _postverify_production(
                final=final,
                current=current,
                local_role=local_role,
                frozen_release_id=str(request["frozen_release_id"]),
                frozen_payload_sha256=str(request["frozen_payload_sha256"]),
                expected_manifest_sha256=str(request["expected_manifest_sha256"]),
                compatibility=compatibility,
                legacy=legacy,
                legacy_inventory_sha256=legacy_hash,
                legacy_identity=legacy_identity,
            )
            _remove_production_transaction(transaction_path)
            transaction = None
            transaction_published = False
            _cleanup_owned_production_incoming(
                incoming,
                reservation_id=str(request["reservation_id"]),
                local_role=local_role,
            )
            return {
                "compatibility_link": (
                    compatibility.as_posix() if compatibility is not None else None
                ),
                "created_release": created_release,
                "current_link": current.as_posix(),
                "final_release": final.as_posix(),
                "legacy_inventory_sha256": legacy_hash,
                "legacy_path": legacy.as_posix() if legacy is not None else None,
                "payload_sha256": request["frozen_payload_sha256"],
                "previous_current_target": previous_current,
                "release_id": request["frozen_release_id"],
                "schema": "wam_remote_production_activate_response_v1",
                "target_role": local_role,
            }
        except BaseException as original:
            rollback_error: Optional[BaseException] = None
            try:
                if legacy_evidence is not None:
                    legacy_evidence.close()
                    legacy_evidence = None
                if pointer_parent_descriptor is not None:
                    os.close(pointer_parent_descriptor)
                    pointer_parent_descriptor = None
                if owned_state_descriptor is not None:
                    os.close(owned_state_descriptor)
                    owned_state_descriptor = None
                if transaction is not None:
                    _rollback_production_transaction(transaction, request)
                    if transaction_published:
                        _remove_production_transaction(transaction_path)
                    transaction = None
                    transaction_published = False
                _cleanup_owned_production_incoming(
                    incoming,
                    reservation_id=str(request["reservation_id"]),
                    local_role=local_role,
                )
            except BaseException as exc:
                rollback_error = exc
            if rollback_error is not None:
                raise AgentPolicyError(
                    "production activation failed and rollback did not converge: %s"
                    % rollback_error
                ) from original
            raise


def reconcile_production_release(
    request: Dict[str, object], *, local_role: str
) -> Dict[str, object]:
    paths = HOST_POLICIES[local_role]["production"]
    incoming = Path(str(request["incoming_path"]))
    releases = Path(paths["releases_root"])
    current = Path(paths["current_link"])
    final = releases / str(request["frozen_release_id"])
    compatibility = current.parent / "auscultation" if local_role == "linbox" else None
    transaction_path = _production_transaction_path(
        current.parent, str(request["reservation_id"])
    )
    with _production_deployment_lock(local_role):
        _assert_no_other_production_transaction(
            current.parent,
            allowed=transaction_path if os.path.lexists(str(transaction_path)) else None,
        )
        transaction: Optional[Dict[str, object]] = None
        if os.path.lexists(str(transaction_path)):
            transaction = _validate_production_transaction_for_request(
                _read_production_transaction(transaction_path),
                request,
                local_role=local_role,
            )
        promoted = False
        reconciled_legacy: Optional[Path] = None
        reconciled_legacy_identity: Optional[tuple] = None
        reconciled_legacy_hash: Optional[str] = None
        if transaction is not None and isinstance(transaction.get("legacy_path"), str):
            reconciled_legacy = Path(str(transaction["legacy_path"]))
            reconciled_legacy_identity = (
                int(transaction["legacy_device"]), int(transaction["legacy_inode"]),
            )
            reconciled_legacy_hash = str(transaction["legacy_inventory_sha256"])
        target = _read_current_target(current)
        expected_target = os.path.relpath(str(final), str(current.parent))
        if final.is_dir() and not final.is_symlink() and target == expected_target:
            try:
                if transaction is not None:
                    parent = _open_real_directory(current.parent, create=False)
                    try:
                        observed_target, observed_identity = _read_pointer_at(
                            parent, current.name
                        )
                    finally:
                        os.close(parent)
                    expected_identity = _pointer_identity_tuple(
                        transaction["activated_pointer_identity"],
                        "production activated pointer identity",
                    )
                    if (
                        observed_target != expected_target
                        or observed_identity[:3] != expected_identity[:3]
                    ):
                        raise AgentPolicyError(
                            "production activated pointer inode diverged"
                        )
                _postverify_production(
                    final=final,
                    current=current,
                    local_role=local_role,
                    frozen_release_id=str(request["frozen_release_id"]),
                    frozen_payload_sha256=str(request["frozen_payload_sha256"]),
                    expected_manifest_sha256=str(request["expected_manifest_sha256"]),
                    compatibility=compatibility,
                    legacy=reconciled_legacy,
                    legacy_inventory_sha256=reconciled_legacy_hash,
                    legacy_identity=reconciled_legacy_identity,
                )
                promoted = True
            except AgentPolicyError:
                promoted = False
        if promoted and transaction is not None:
            _cleanup_prepared_production_pointer(transaction, current)
            _remove_production_transaction(transaction_path)
            transaction = None
        elif not promoted and transaction is not None:
            _rollback_production_transaction(transaction, request)
            _remove_production_transaction(transaction_path)
            transaction = None
            target = _read_current_target(current)
        staging = releases / (
            ".staging-%s-%s" % (
                request["frozen_release_id"], request["reservation_id"],
            )
        )
        if os.path.lexists(str(staging)):
            _remove_real_tree_path(staging)
        if not promoted and os.path.lexists(str(_reservation_marker_path(incoming))):
            _cleanup_owned_production_incoming(
                incoming,
                reservation_id=str(request["reservation_id"]),
                local_role=local_role,
            )
        elif promoted and os.path.lexists(str(_reservation_marker_path(incoming))):
            _cleanup_owned_production_incoming(
                incoming,
                reservation_id=str(request["reservation_id"]),
                local_role=local_role,
            )
        return {
            "compatibility_link": (
                compatibility.as_posix() if compatibility is not None else None
            ),
            "current_target": target,
            "final_release": final.as_posix() if final.exists() else None,
            "legacy_path": (
                reconciled_legacy.as_posix() if reconciled_legacy is not None else None
            ),
            "schema": "wam_remote_production_reconcile_response_v1",
            "status": "promoted" if promoted else "cleaned",
            "target_role": local_role,
        }


def execute_request(
    request: object,
    *,
    local_role: str,
    agent_path: Union[str, Path],
) -> Dict[str, object]:
    validated = validate_request(request, local_role=local_role, agent_path=agent_path)
    _ensure_real_directory(validated.releases_root)
    _ensure_real_directory(validated.current_link.parent)
    _require_owned_reservation(
        validated.incoming_path,
        reservation_id=validated.reservation_id,
        target_role=validated.target_role,
        operation="install_release",
    )
    try:
        incoming_envelope = contract.verify_release(validated.incoming_path)
        manifest_sha256 = _sha256_file(validated.incoming_path / contract.MANIFEST_NAME)
        if (
            incoming_envelope.get("release_id") != validated.expected_release_id
            or incoming_envelope.get("payload_sha256") != validated.expected_payload_sha256
            or manifest_sha256 != validated.expected_manifest_sha256
        ):
            raise AgentPolicyError("incoming release identity does not match the request")
        final_candidate = validated.releases_root / validated.expected_release_id
        created_release = not os.path.lexists(str(final_candidate))
        responses = []

        def post_activate(final: Path) -> None:
            envelope = contract.verify_release(final)
            marker = json.loads((final / contract.COMPLETE_NAME).read_text(encoding="utf-8"))
            expected_target = os.path.relpath(str(final), str(validated.current_link.parent))
            if (
                final.as_posix() != final_candidate.as_posix()
                or final.parent.resolve() != validated.releases_root.resolve()
                or envelope.get("release_id") != validated.expected_release_id
                or envelope.get("payload_sha256") != validated.expected_payload_sha256
                or _sha256_file(final / contract.MANIFEST_NAME)
                != validated.expected_manifest_sha256
                or marker.get("target_role") != validated.target_role
                or not validated.current_link.is_symlink()
                or os.readlink(str(validated.current_link)) != expected_target
                or validated.current_link.resolve() != final.resolve()
            ):
                raise contract.VerificationError(
                    "installed release failed exact post-activation verification"
                )
            responses.append({
                "created_release": created_release,
                "current_link": validated.current_link.as_posix(),
                "final_release": final.as_posix(),
                "release_id": envelope["release_id"],
                "schema": RESPONSE_SCHEMA,
                "target_role": validated.target_role,
            })

        contract.install_release(
            validated.incoming_path,
            validated.releases_root,
            validated.current_link,
            target_role=validated.target_role,
            post_activate=post_activate,
        )
        return responses[0]
    finally:
        shutil.rmtree(str(validated.incoming_path), ignore_errors=True)


def dispatch_request(
    request: object, *, local_role: str, agent_path: Union[str, Path]
) -> Dict[str, object]:
    operation = request.get("operation") if isinstance(request, dict) else None
    if operation in {
        "prepare_production_incoming",
        "activate_production_release",
        "reconcile_production_release",
    }:
        production = _validate_production_request(
            request, local_role=local_role, agent_path=agent_path,
        )
        if operation == "prepare_production_incoming":
            return prepare_production_incoming(
                local_role,
                incoming_path=str(production["incoming_path"]),
                reservation_id=str(production["reservation_id"]),
                frozen_release_id=str(production["frozen_release_id"]),
                frozen_payload_sha256=str(production["frozen_payload_sha256"]),
            )
        if operation == "activate_production_release":
            return activate_production_release(production, local_role=local_role)
        return reconcile_production_release(production, local_role=local_role)
    if operation == "install_release":
        return execute_request(request, local_role=local_role, agent_path=agent_path)
    if operation == "prepare_perception_incoming":
        perception_prepare = _validate_perception_prepare_request(
            request, local_role=local_role, agent_path=agent_path
        )
        return prepare_perception_incoming(
            local_role,
            data_kind=str(perception_prepare["data_kind"]),
            source_role=str(perception_prepare["source_role"]),
            incoming_path=str(perception_prepare["incoming_path"]),
            reservation_id=str(perception_prepare["reservation_id"]),
        )
    if operation == "install_perception_artifact":
        perception = _validate_perception_request(
            request, local_role=local_role, agent_path=agent_path
        )
        return install_perception_artifact(perception, local_role=local_role)
    if operation == "reconcile_perception_transfer":
        perception_reconcile = _validate_perception_reconcile_request(
            request, local_role=local_role, agent_path=agent_path
        )
        return reconcile_perception_transfer(
            perception_reconcile, local_role=local_role
        )
    if operation == "prepare_data_incoming":
        data_prepare = _validate_data_prepare_request(
            request, local_role=local_role, agent_path=agent_path
        )
        return prepare_data_incoming(
            local_role,
            data_operation=data_prepare["data_operation"],
            incoming_path=data_prepare["incoming_path"],
            reservation_id=data_prepare["reservation_id"],
        )
    if operation == "reconcile_data_transfer":
        reconcile = _validate_data_reconcile_request(
            request, local_role=local_role, agent_path=agent_path
        )
        return reconcile_data_transfer(reconcile, local_role=local_role)
    if operation in ("archive_raw_session", "install_converted_dataset"):
        data = _validate_data_request(request, local_role=local_role, agent_path=agent_path)
        incoming = Path(data["incoming_path"])
        trust = trusted_training_policy(local_role)
        with _data_deployment_lock(data["releases_root"]):
            _require_owned_reservation(
                incoming, reservation_id=data["reservation_id"],
                target_role=local_role, operation=operation,
            )
            if operation == "archive_raw_session":
                result = archive_raw_session(
                    data["incoming_path"],
                    releases_root=data["releases_root"],
                    expected_manifest_sha256=data["expected_manifest_sha256"],
                    expected_finalization_sha256=data["expected_finalization_sha256"],
                    trusted_finalization_registry=trust[
                        "finalization_registry"
                    ],
                    trusted_snapshot_root=trust["snapshot_root"],
                )
            else:
                result = install_converted_dataset(
                    data["incoming_path"],
                    releases_root=data["releases_root"],
                    expected_release_id=data["expected_release_id"],
                    expected_payload_sha256=data["expected_payload_sha256"],
                    expected_manifest_sha256=data["expected_manifest_sha256"],
                    raw_releases_root=operation_policy(
                        local_role, "archive_raw_session"
                    )["releases_root"],
                    trusted_finalization_registry=trust[
                        "finalization_registry"
                    ],
                    trusted_snapshot_root=trust["snapshot_root"],
                )
            _remove_owned_reservation_marker(
                incoming, reservation_id=data["reservation_id"],
                target_role=local_role, operation=operation,
            )
            return result
    control = _validate_control_request(request, local_role=local_role, agent_path=agent_path)
    if control["operation"] == "bootstrap_canary":
        return bootstrap_canary(local_role)
    if control["operation"] == "prepare_incoming":
        return prepare_incoming(
            local_role, control["incoming_path"], control["reservation_id"]
        )
    if control["operation"] == "inspect_canary":
        return inspect_canary(local_role)
    if control["operation"] == "cleanup_canary":
        return cleanup_canary(
            local_role,
            incoming_path=control["incoming_path"],
            reservation_id=control["reservation_id"],
            created_release_id=control["created_release_id"],
            pre_current_target=control["pre_current_target"],
            pre_release_ids=tuple(control["pre_release_ids"]),
        )
    if control["operation"] == "rollback_canary":
        return force_canary_rollback(
            local_role, control["previous_target"],
            expected_current_target=control["expected_current_target"],
        )
    raise AgentPolicyError("remote operation is unsupported")


def main() -> int:
    raw = sys.stdin.buffer.read(65537)
    if not raw or len(raw) > 65536:
        raise AgentPolicyError("remote request is empty or too large")
    try:
        request = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentPolicyError("remote request is not UTF-8 JSON") from exc
    if raw != contract.canonical_json_bytes(request):
        raise AgentPolicyError("remote request is not canonical JSON")
    role = detect_local_role()
    expected_helper = HOST_POLICIES[role].get("helper_path")
    if expected_helper and Path(__file__).resolve().as_posix() != expected_helper:
        raise AgentPolicyError("remote agent was not invoked from its fixed policy path")
    response = dispatch_request(
        request,
        local_role=role,
        agent_path=Path(__file__).resolve(),
    )
    sys.stdout.buffer.write(contract.canonical_json_bytes(response))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.buffer.write(contract.canonical_json_bytes({
            "error": str(exc),
            "error_type": type(exc).__name__,
            "schema": "wam_remote_error_v1",
        }))
        raise SystemExit(1)
