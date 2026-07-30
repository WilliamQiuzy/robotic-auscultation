"""Fail-closed local and SSH deployment planning for WAM releases.

All remote operation data is serialized as JSON on stdin.  The only remote shell
command this module will construct is the profile-pinned Python interpreter plus
the profile-pinned ``remote_agent.py`` path.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
import uuid

import artifact_contract as contract
import release_source


PROFILE_SCHEMA = "wam_release_profile_v1"
REMOTE_REQUEST_SCHEMA = "wam_remote_install_request_v1"
PRODUCTION_CAPABILITY = "task8_frozen_release_activation_v1"
PROFILE_PATH = Path(__file__).resolve().with_name("release_profile.json")
REMOTE_AGENT_PATH = Path(__file__).resolve().with_name("remote_agent.py")
_MAC_PERCEPTION_RELEASES_ROOT = Path(
    "/Users/williamqiu/.wam/perception-inputs/releases"
)
_NONCE = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}-[0-9a-f]{20}$")
_SOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_REQUIRED_PATHS = release_source.REQUIRED_RELEASE_PATHS


class PolicyError(RuntimeError):
    """A deployment request is outside the audited release policy."""


class DeploymentExecutionError(RuntimeError):
    """A planned transport command failed or returned an invalid response."""

    def __init__(self, message: str, *, evidence: Optional[Dict[str, object]] = None):
        super().__init__(message)
        self.evidence = evidence or {}


_CANONICAL_HOSTS = {
    "h200": {
        "canary": {
            "current_link": "/home/ssh-ziyue/wam_ausc/.deploy-canary/current",
            "incoming_root": "/home/ssh-ziyue/wam_ausc/.deploy-canary/incoming",
            "releases_root": "/home/ssh-ziyue/wam_ausc/.deploy-canary/releases",
        },
        "contract_path": "/home/ssh-ziyue/.wam_artifact_contract.py",
        "data": {
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
            "raw_session": {
                "incoming_root": "/home/ssh-ziyue/wam_ausc/data/raw_sessions/.staging",
                "releases_root": "/home/ssh-ziyue/wam_ausc/data/raw_sessions/releases",
            },
        },
        "helper_path": "/home/ssh-ziyue/.wam_remote_agent.py",
        "home": "/home/ssh-ziyue",
        "production": {
            "current_link": "/home/ssh-ziyue/wam_ausc/current",
            "incoming_root": "/home/ssh-ziyue/wam_ausc/.incoming",
            "releases_root": "/home/ssh-ziyue/wam_ausc/.releases",
        },
        "python": "/usr/bin/python3",
        "role": "h200",
    },
    "linbox": {
        "canary": {
            "current_link": "/home/robot/Desktop/stethoscope/.deploy-canary/current",
            "incoming_root": "/home/robot/Desktop/stethoscope/.deploy-canary/incoming",
            "releases_root": "/home/robot/Desktop/stethoscope/.deploy-canary/releases",
        },
        "contract_path": "/home/robot/.wam_artifact_contract.py",
        "data_sources": {
            "converted_dataset_root": "/home/robot/wam_data/lerobot/releases",
            "raw_session_root": "/home/robot/wam_data/sessions",
        },
        "helper_path": "/home/robot/.wam_remote_agent.py",
        "home": "/home/robot",
        "perception_runtime": {
            "current_link": "/home/robot/wam_data/perception_runtime/current",
            "incoming_root": "/home/robot/wam_data/perception_runtime/.staging",
            "releases_root": "/home/robot/wam_data/perception_runtime/releases",
        },
        "production": {
            "current_link": "/home/robot/Desktop/stethoscope/.wam-current",
            "incoming_root": "/home/robot/Desktop/stethoscope/.wam-incoming",
            "releases_root": "/home/robot/Desktop/stethoscope/.wam-releases",
        },
        "python": "/home/robot/envs/ausc/bin/python",
        "role": "linbox",
        "simulation": {
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
        },
    },
}

_CANONICAL_CONNECTIONS = [
    {"alias": "nebius-h200", "source_role": "mac", "target_role": "h200"},
    {"alias": "linbox", "source_role": "mac", "target_role": "linbox"},
    {"alias": "h200", "source_role": "linbox", "target_role": "h200"},
]


def load_profile(path: Union[str, Path]) -> Dict[str, object]:
    """Load the one audited three-host profile; configuration is not extensible."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyError("release profile is missing or invalid") from exc
    if not isinstance(value, dict) or set(value) != {
        "connections", "hosts", "schema", "workspace_root"
    }:
        raise PolicyError("release profile fields are invalid")
    if value.get("schema") != PROFILE_SCHEMA:
        raise PolicyError("release profile schema is invalid")
    if value.get("workspace_root") != "/Users/williamqiu/Desktop/Harvard/WAM":
        raise PolicyError("release profile workspace is invalid")
    if value.get("connections") != _CANONICAL_CONNECTIONS:
        raise PolicyError("release profile connection policy is invalid")
    if value.get("hosts") != _CANONICAL_HOSTS:
        raise PolicyError("release profile host policy is invalid")
    return value


@dataclass(frozen=True)
class CommandStep:
    name: str
    argv: Tuple[str, ...]
    stdin: bytes = b""


@dataclass(frozen=True)
class SSHPlan:
    agent_sha256: str
    contract_sha256: str
    expected_release_id: str
    current_link: str
    host_alias: str
    incoming_path: str
    phases: Tuple[str, ...]
    remote_command: str
    reservation_id: str
    releases_root: str
    steps: Tuple[CommandStep, ...]
    target_role: str


@dataclass(frozen=True)
class DataSSHPlan:
    agent_sha256: str
    contract_sha256: str
    expected_final_release: str
    expected_release_id: str
    host_alias: str
    incoming_path: str
    operation: str
    reservation_id: str
    releases_root: str
    remote_command: str
    source_path: str
    steps: Tuple[CommandStep, ...]
    target_role: str
    data_kind: Optional[str] = None
    expected_manifest_sha256: Optional[str] = None
    expected_payload_sha256: Optional[str] = None


@dataclass(frozen=True)
class ProductionSSHPlan:
    agent_sha256: str
    contract_sha256: str
    current_link: str
    expected_manifest_sha256: str
    expected_payload_sha256: str
    expected_release_id: str
    host_alias: str
    incoming_path: str
    reconcile_step: CommandStep
    releases_root: str
    remote_command: str
    reservation_id: str
    steps: Tuple[CommandStep, ...]
    target_role: str


@dataclass(frozen=True)
class LocalDeploymentResult:
    final_release: Path
    incoming_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connection(
    profile: Dict[str, object], source_role: str, target_role: str, host_alias: str
) -> Dict[str, str]:
    wanted = {
        "alias": host_alias,
        "source_role": source_role,
        "target_role": target_role,
    }
    if wanted not in profile["connections"]:
        raise PolicyError("requested connection is not in the audited profile")
    return wanted


def build_ssh_plan(
    source_release: Union[str, Path],
    *,
    source_role: str,
    target_role: str,
    host_alias: str,
    scope: str = "canary",
    nonce: Optional[str] = None,
    reservation_id: Optional[str] = None,
    profile_path: Union[str, Path] = PROFILE_PATH,
    remote_agent_path: Union[str, Path] = REMOTE_AGENT_PATH,
) -> SSHPlan:
    """Build, but never execute, the audited rsync + fixed-agent SSH protocol."""
    source = Path(source_release).resolve()
    manifest = contract.verify_release(source)
    profile = load_profile(profile_path)
    _connection(profile, source_role, target_role, host_alias)
    if scope != "canary":
        raise PolicyError("Task 2 SSH execution is canary-only; production activation is disabled")
    token = nonce or uuid.uuid4().hex
    if not isinstance(token, str) or not _NONCE.fullmatch(token):
        raise PolicyError("deployment nonce must be 32 lowercase hexadecimal characters")
    owner = reservation_id or (uuid.uuid4().hex + uuid.uuid4().hex)
    if not isinstance(owner, str) or not _HEX64.fullmatch(owner):
        raise PolicyError("reservation id must be 64 lowercase hexadecimal characters")
    host = profile["hosts"][target_role]
    paths = host[scope]
    incoming_name = "incoming-%s-%s" % (manifest["release_id"], token)
    incoming_path = paths["incoming_root"] + "/" + incoming_name
    agent_path = Path(remote_agent_path).resolve()
    if agent_path != REMOTE_AGENT_PATH or not agent_path.is_file() or agent_path.is_symlink():
        raise PolicyError("remote agent must be the audited fixed local file")
    agent_sha256 = _sha256_file(agent_path)
    contract_path = Path(contract.__file__).resolve()
    if contract_path != Path(__file__).resolve().with_name("artifact_contract.py"):
        raise PolicyError("artifact contract must be the audited fixed local file")
    contract_sha256 = _sha256_file(contract_path)
    remote_command = "%s %s" % (host["python"], host["helper_path"])
    manifest_sha256 = _sha256_file(source / contract.MANIFEST_NAME)
    request = {
        "adopt_existing": False,
        "agent_sha256": agent_sha256,
        "contract_sha256": contract_sha256,
        "current_link": paths["current_link"],
        "expected_manifest_sha256": manifest_sha256,
        "expected_payload_sha256": manifest["payload_sha256"],
        "expected_release_id": manifest["release_id"],
        "incoming_path": incoming_path,
        "operation": "install_release",
        "reservation_id": owner,
        "releases_root": paths["releases_root"],
        "schema": REMOTE_REQUEST_SCHEMA,
        "scope": scope,
        "target_role": target_role,
    }
    control_common = {
        "agent_sha256": agent_sha256,
        "contract_sha256": contract_sha256,
        "schema": "wam_remote_control_request_v1",
        "scope": scope,
        "target_role": target_role,
    }
    bootstrap_request = dict(control_common, operation="bootstrap_canary")
    prepare_request = dict(
        control_common, operation="prepare_incoming", incoming_path=incoming_path,
        reservation_id=owner,
    )
    inspect_request = dict(control_common, operation="inspect_canary")
    # macOS openrsync 2.6.9 supports these flags; unique destinations make
    # --inplace/--partial unnecessary and unsafe.
    rsync_options = (
        "rsync", "-a", "-c", "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=10", "--",
    )
    ssh_command = (
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--",
        host_alias, remote_command,
    )
    steps = (
        CommandStep(
            "upload_contract",
            rsync_options + (str(contract_path), "%s:%s" % (host_alias, host["contract_path"])),
        ),
        CommandStep(
            "upload_agent",
            rsync_options + (str(agent_path), "%s:%s" % (host_alias, host["helper_path"])),
        ),
        CommandStep(
            "bootstrap",
            ssh_command,
            contract.canonical_json_bytes(bootstrap_request),
        ),
        CommandStep(
            "pre_inspect", ssh_command, contract.canonical_json_bytes(inspect_request)
        ),
        CommandStep(
            "prepare_incoming", ssh_command, contract.canonical_json_bytes(prepare_request)
        ),
        CommandStep(
            "stage_release",
            rsync_options + (str(source) + "/", "%s:%s/" % (host_alias, incoming_path)),
        ),
        CommandStep(
            "install",
            ssh_command,
            contract.canonical_json_bytes(request),
        ),
        CommandStep(
            "post_inspect", ssh_command, contract.canonical_json_bytes(inspect_request)
        ),
    )
    return SSHPlan(
        agent_sha256=agent_sha256,
        contract_sha256=contract_sha256,
        expected_release_id=str(manifest["release_id"]),
        current_link=paths["current_link"],
        host_alias=host_alias,
        incoming_path=incoming_path,
        phases=(
            "verify_completed_source",
            "allocate_unique_incoming",
            "upload_content_hashed_agent",
            "rsync_no_inplace",
            "remote_exact_verify",
            "atomic_promotion",
            "capture_previous_pointer",
            "atomic_activation",
            "post_activation_verify_or_rollback",
        ),
        remote_command=remote_command,
        reservation_id=owner,
        releases_root=paths["releases_root"],
        steps=steps,
        target_role=target_role,
    )


def build_production_ssh_plan(
    source_release: Union[str, Path],
    *,
    target_role: str,
    host_alias: str,
    confirm_release: Optional[str],
    confirm_payload_sha256: Optional[str],
    nonce: Optional[str] = None,
    reservation_id: Optional[str] = None,
    profile_path: Union[str, Path] = PROFILE_PATH,
    remote_agent_path: Union[str, Path] = REMOTE_AGENT_PATH,
) -> ProductionSSHPlan:
    """Build the separate, explicitly confirmed Task 8 production capability."""
    source = Path(source_release).resolve()
    envelope = contract.verify_release(source)
    release_id = str(envelope["release_id"])
    payload_sha256 = str(envelope["payload_sha256"])
    if confirm_release != release_id:
        raise PolicyError("--confirm-release must exactly match the frozen release ID")
    if confirm_payload_sha256 != payload_sha256:
        raise PolicyError(
            "--confirm-payload-sha256 must exactly match the frozen payload hash"
        )
    profile = load_profile(profile_path)
    _connection(profile, "mac", target_role, host_alias)
    token = nonce or uuid.uuid4().hex
    if not isinstance(token, str) or not _NONCE.fullmatch(token):
        raise PolicyError("deployment nonce must be 32 lowercase hexadecimal characters")
    owner = reservation_id or (uuid.uuid4().hex + uuid.uuid4().hex)
    if not isinstance(owner, str) or not _HEX64.fullmatch(owner):
        raise PolicyError("reservation id must be 64 lowercase hexadecimal characters")
    host = profile["hosts"][target_role]
    paths = host["production"]
    incoming_path = "%s/incoming-%s-%s" % (
        paths["incoming_root"], release_id, token,
    )
    agent_path = Path(remote_agent_path).resolve()
    if agent_path != REMOTE_AGENT_PATH or not agent_path.is_file() or agent_path.is_symlink():
        raise PolicyError("remote agent must be the audited fixed local file")
    contract_path = Path(contract.__file__).resolve()
    if contract_path != Path(__file__).resolve().with_name("artifact_contract.py"):
        raise PolicyError("artifact contract must be the audited fixed local file")
    agent_sha256 = _sha256_file(agent_path)
    contract_sha256 = _sha256_file(contract_path)
    manifest_sha256 = _sha256_file(source / contract.MANIFEST_NAME)
    remote_command = "%s %s" % (host["python"], host["helper_path"])
    common = {
        "agent_sha256": agent_sha256,
        "capability": PRODUCTION_CAPABILITY,
        "contract_sha256": contract_sha256,
        "frozen_payload_sha256": payload_sha256,
        "frozen_release_id": release_id,
        "incoming_path": incoming_path,
        "reservation_id": owner,
        "target_role": target_role,
    }
    prepare = dict(
        common,
        operation="prepare_production_incoming",
        schema="wam_remote_production_prepare_request_v1",
    )
    activate = dict(
        common,
        expected_manifest_sha256=manifest_sha256,
        operation="activate_production_release",
        schema="wam_remote_production_activate_request_v1",
    )
    reconcile = dict(
        common,
        expected_manifest_sha256=manifest_sha256,
        operation="reconcile_production_release",
        schema="wam_remote_production_reconcile_request_v1",
    )
    rsync_options = (
        "rsync", "-a", "-c", "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=10", "--",
    )
    ssh_command = (
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--",
        host_alias, remote_command,
    )
    steps = (
        CommandStep(
            "upload_contract", rsync_options + (
                str(contract_path), "%s:%s" % (host_alias, host["contract_path"]),
            ),
        ),
        CommandStep(
            "upload_agent", rsync_options + (
                str(agent_path), "%s:%s" % (host_alias, host["helper_path"]),
            ),
        ),
        CommandStep(
            "prepare_production", ssh_command, contract.canonical_json_bytes(prepare),
        ),
        CommandStep(
            "stage_release", rsync_options + (
                str(source) + "/", "%s:%s/" % (host_alias, incoming_path),
            ),
        ),
        CommandStep(
            "activate_production", ssh_command, contract.canonical_json_bytes(activate),
        ),
    )
    return ProductionSSHPlan(
        agent_sha256=agent_sha256,
        contract_sha256=contract_sha256,
        current_link=paths["current_link"],
        expected_manifest_sha256=manifest_sha256,
        expected_payload_sha256=payload_sha256,
        expected_release_id=release_id,
        host_alias=host_alias,
        incoming_path=incoming_path,
        reconcile_step=CommandStep(
            "reconcile_production", ssh_command, contract.canonical_json_bytes(reconcile),
        ),
        releases_root=paths["releases_root"],
        remote_command=remote_command,
        reservation_id=owner,
        steps=steps,
        target_role=target_role,
    )


def _data_identity(value: str, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise PolicyError("%s must be a lowercase SHA-256 identity" % label)
    return value


def _linbox_source(source_path: Union[str, Path], expected_root: str) -> str:
    raw = source_path.as_posix() if isinstance(source_path, Path) else source_path
    if not isinstance(raw, str):
        raise PolicyError("data source path is invalid")
    source = PurePosixPath(raw)
    if (
        not source.is_absolute()
        or source.as_posix() != raw
        or ".." in source.parts
        or source.parent.as_posix() != expected_root
        or not _SOURCE_NAME.fullmatch(source.name)
    ):
        raise PolicyError("data source path is outside the audited linbox source root")
    return raw


def _data_plan(
    source_path: str,
    *,
    operation: str,
    expected_release_id: str,
    incoming_path: str,
    reservation_id: str,
    request_identity: Dict[str, str],
    profile_path: Union[str, Path],
    remote_agent_path: Union[str, Path],
) -> DataSSHPlan:
    profile = load_profile(profile_path)
    _connection(profile, "linbox", "h200", "h200")
    host = profile["hosts"]["h200"]
    data_kind = (
        "raw_session" if operation == "archive_raw_session" else "converted_dataset"
    )
    paths = host["data"][data_kind]
    agent_path = Path(remote_agent_path).resolve()
    if agent_path != REMOTE_AGENT_PATH or not agent_path.is_file() or agent_path.is_symlink():
        raise PolicyError("remote agent must be the audited fixed local file")
    contract_path = Path(contract.__file__).resolve()
    if contract_path != Path(__file__).resolve().with_name("artifact_contract.py"):
        raise PolicyError("artifact contract must be the audited fixed local file")
    agent_sha256 = _sha256_file(agent_path)
    contract_sha256 = _sha256_file(contract_path)
    common = {
        "agent_sha256": agent_sha256,
        "contract_sha256": contract_sha256,
        "reservation_id": reservation_id,
        "target_role": "h200",
    }
    prepare_request = dict(
        common,
        data_operation=operation,
        incoming_path=incoming_path,
        operation="prepare_data_incoming",
        schema="wam_remote_data_prepare_request_v1",
    )
    install_request = dict(
        common,
        incoming_path=incoming_path,
        operation=operation,
        reservation_id=reservation_id,
        releases_root=paths["releases_root"],
        schema="wam_remote_data_request_v1",
        **request_identity,
    )
    rsync_options = (
        "rsync", "-a", "-c", "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=10", "--",
    )
    remote_command = "%s %s" % (host["python"], host["helper_path"])
    ssh_command = (
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--",
        "h200", remote_command,
    )
    steps = (
        CommandStep(
            "upload_contract",
            rsync_options + (str(contract_path), "h200:%s" % host["contract_path"]),
        ),
        CommandStep(
            "upload_agent",
            rsync_options + (str(agent_path), "h200:%s" % host["helper_path"]),
        ),
        CommandStep(
            "prepare_data_incoming",
            ssh_command,
            contract.canonical_json_bytes(prepare_request),
        ),
        CommandStep(
            "stage_data",
            rsync_options + (source_path + "/", "h200:%s/" % incoming_path),
        ),
        CommandStep(
            operation,
            ssh_command,
            contract.canonical_json_bytes(install_request),
        ),
    )
    return DataSSHPlan(
        agent_sha256=agent_sha256,
        contract_sha256=contract_sha256,
        expected_final_release=paths["releases_root"] + "/" + expected_release_id,
        expected_release_id=expected_release_id,
        host_alias="h200",
        incoming_path=incoming_path,
        operation=operation,
        reservation_id=reservation_id,
        releases_root=paths["releases_root"],
        remote_command=remote_command,
        source_path=source_path,
        steps=steps,
        target_role="h200",
    )


def build_raw_archive_plan(
    source_session: Union[str, Path],
    *,
    manifest_sha256: str,
    finalization_sha256: str,
    nonce: Optional[str] = None,
    reservation_id: Optional[str] = None,
    profile_path: Union[str, Path] = PROFILE_PATH,
    remote_agent_path: Union[str, Path] = REMOTE_AGENT_PATH,
) -> DataSSHPlan:
    """Build the strict linbox-to-H200 raw-session archive plan."""
    profile = load_profile(profile_path)
    source_root = profile["hosts"]["linbox"]["data_sources"]["raw_session_root"]
    source = _linbox_source(source_session, source_root)
    manifest = _data_identity(manifest_sha256, "manifest_sha256")
    finalization = _data_identity(finalization_sha256, "finalization_sha256")
    token = nonce or uuid.uuid4().hex
    if not isinstance(token, str) or not _NONCE.fullmatch(token):
        raise PolicyError("data transfer nonce must be 32 lowercase hexadecimal characters")
    owner = reservation_id or (uuid.uuid4().hex + uuid.uuid4().hex)
    if not isinstance(owner, str) or not _HEX64.fullmatch(owner):
        raise PolicyError("reservation id must be 64 lowercase hexadecimal characters")
    incoming_root = profile["hosts"]["h200"]["data"]["raw_session"]["incoming_root"]
    expected_release_id = "raw_session-" + manifest[:20]
    return _data_plan(
        source,
        operation="archive_raw_session",
        expected_release_id=expected_release_id,
        incoming_path=incoming_root + "/transfer-raw-" + token,
        reservation_id=owner,
        request_identity={
            "expected_finalization_sha256": finalization,
            "expected_manifest_sha256": manifest,
        },
        profile_path=profile_path,
        remote_agent_path=remote_agent_path,
    )


def build_converted_dataset_plan(
    source_release: Union[str, Path],
    *,
    release_id: str,
    payload_sha256: str,
    manifest_sha256: str,
    reservation_id: Optional[str] = None,
    profile_path: Union[str, Path] = PROFILE_PATH,
    remote_agent_path: Union[str, Path] = REMOTE_AGENT_PATH,
) -> DataSSHPlan:
    """Build the strict linbox-to-H200 converted-dataset install plan."""
    profile = load_profile(profile_path)
    source_root = profile["hosts"]["linbox"]["data_sources"]["converted_dataset_root"]
    source = _linbox_source(source_release, source_root)
    if not isinstance(release_id, str) or not _RELEASE_ID.fullmatch(release_id):
        raise PolicyError("converted dataset release id is invalid")
    if PurePosixPath(source).name != release_id:
        raise PolicyError("converted dataset source path does not bind the release identity")
    payload = _data_identity(payload_sha256, "payload_sha256")
    manifest = _data_identity(manifest_sha256, "manifest_sha256")
    owner = reservation_id or (uuid.uuid4().hex + uuid.uuid4().hex)
    if not isinstance(owner, str) or not _HEX64.fullmatch(owner):
        raise PolicyError("reservation id must be 64 lowercase hexadecimal characters")
    incoming_root = profile["hosts"]["h200"]["data"]["converted_dataset"]["incoming_root"]
    return _data_plan(
        source,
        operation="install_converted_dataset",
        expected_release_id=release_id,
        incoming_path=incoming_root + "/" + release_id,
        reservation_id=owner,
        request_identity={
            "expected_manifest_sha256": manifest,
            "expected_payload_sha256": payload,
            "expected_release_id": release_id,
        },
        profile_path=profile_path,
        remote_agent_path=remote_agent_path,
    )


def _sha256_regular_nofollow(path: Path) -> str:
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PolicyError("perception release manifest is not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise PolicyError("perception release manifest changed while hashing")
        return digest.hexdigest()
    except OSError as exc:
        raise PolicyError("perception release manifest is missing or unsafe") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def build_perception_artifact_plan(
    source_release: Union[str, Path],
    *,
    data_kind: str,
    source_role: str,
    target_role: str,
    host_alias: str,
    nonce: Optional[str] = None,
    reservation_id: Optional[str] = None,
    profile_path: Union[str, Path] = PROFILE_PATH,
    remote_agent_path: Union[str, Path] = REMOTE_AGENT_PATH,
) -> DataSSHPlan:
    """Build a pointer-free Mac-to-H200 perception artifact transfer plan."""
    profile = load_profile(profile_path)
    if (source_role, target_role, host_alias) != ("mac", "h200", "nebius-h200"):
        raise PolicyError("perception transport allows only Mac-to-H200 remote upload")
    _connection(profile, source_role, target_role, host_alias)
    host = profile["hosts"]["h200"]
    policies = host["data"]["perception"]
    policy = policies.get(data_kind) if isinstance(data_kind, str) else None
    if (
        not isinstance(policy, dict)
        or policy.get("remote_upload") is not True
        or policy.get("source_role") != "mac"
        or policy.get("artifact_kind") != data_kind
    ):
        raise PolicyError("perception data kind is not authorized for remote upload")

    source = Path(source_release)
    expected_root = Path(_MAC_PERCEPTION_RELEASES_ROOT)
    if not source.is_absolute() or source.parent != expected_root:
        raise PolicyError("perception source release is outside the audited source root")
    try:
        safe_root = expected_root.resolve(strict=True)
        safe_source = source.resolve(strict=True)
    except OSError as exc:
        raise PolicyError("perception source release is missing or unsafe") from exc
    if (
        safe_root != expected_root
        or safe_source != source
        or source.is_symlink()
    ):
        raise PolicyError("perception source release is outside the audited source root")
    try:
        envelope = contract.verify_release(source)
    except (contract.ArtifactError, OSError) as exc:
        raise PolicyError("perception source release verification failed") from exc
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or payload.get("artifact_kind") != data_kind:
        raise PolicyError("perception source artifact kind does not match data kind")
    if payload.get("source_role") != source_role:
        raise PolicyError("perception source role does not match transport policy")
    release_id = envelope.get("release_id")
    payload_sha256 = envelope.get("payload_sha256")
    if (
        not isinstance(release_id, str)
        or not _RELEASE_ID.fullmatch(release_id)
        or source.name != release_id
        or not release_id.startswith(data_kind + "-")
        or not isinstance(payload_sha256, str)
        or not _HEX64.fullmatch(payload_sha256)
    ):
        raise PolicyError("perception source release identity is invalid")
    manifest_sha256 = _sha256_regular_nofollow(source / contract.MANIFEST_NAME)
    try:
        if contract.verify_release(source) != envelope:
            raise PolicyError("perception source release changed during planning")
    except (contract.ArtifactError, OSError) as exc:
        raise PolicyError("perception source release changed during planning") from exc

    token = nonce or uuid.uuid4().hex
    if not isinstance(token, str) or not _NONCE.fullmatch(token):
        raise PolicyError("perception transfer nonce must be 32 lowercase hexadecimal characters")
    owner = reservation_id or (uuid.uuid4().hex + uuid.uuid4().hex)
    if not isinstance(owner, str) or not _HEX64.fullmatch(owner):
        raise PolicyError("reservation id must be 64 lowercase hexadecimal characters")

    agent_path = Path(remote_agent_path).resolve()
    if agent_path != REMOTE_AGENT_PATH or not agent_path.is_file() or agent_path.is_symlink():
        raise PolicyError("remote agent must be the audited fixed local file")
    contract_path = Path(contract.__file__).resolve()
    if contract_path != Path(__file__).resolve().with_name("artifact_contract.py"):
        raise PolicyError("artifact contract must be the audited fixed local file")
    agent_sha256 = _sha256_file(agent_path)
    contract_sha256 = _sha256_file(contract_path)
    incoming_path = "%s/incoming-%s-%s" % (
        policy["staging_root"], release_id, token,
    )
    common = {
        "agent_sha256": agent_sha256,
        "contract_sha256": contract_sha256,
        "data_kind": data_kind,
        "incoming_path": incoming_path,
        "reservation_id": owner,
        "source_role": source_role,
        "target_role": target_role,
    }
    prepare_request = dict(
        common,
        operation="prepare_perception_incoming",
        schema="wam_remote_perception_prepare_request_v1",
    )
    install_request = dict(
        common,
        expected_manifest_sha256=manifest_sha256,
        expected_payload_sha256=payload_sha256,
        expected_release_id=release_id,
        operation="install_perception_artifact",
        releases_root=policy["releases_root"],
        schema="wam_remote_perception_request_v1",
    )
    upload_options = (
        "rsync", "-a", "-c", "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=10", "--",
    )
    stage_options = (
        "rsync", "-a", "-c", "--delete", "--delay-updates", "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=10", "--",
    )
    remote_command = "%s %s" % (host["python"], host["helper_path"])
    ssh_command = (
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--",
        host_alias, remote_command,
    )
    steps = (
        CommandStep(
            "upload_contract",
            upload_options + (str(contract_path), "%s:%s" % (host_alias, host["contract_path"])),
        ),
        CommandStep(
            "upload_agent",
            upload_options + (str(agent_path), "%s:%s" % (host_alias, host["helper_path"])),
        ),
        CommandStep(
            "prepare_perception_incoming",
            ssh_command,
            contract.canonical_json_bytes(prepare_request),
        ),
        CommandStep(
            "stage_perception_artifact",
            stage_options + (source.as_posix() + "/", "%s:%s/" % (host_alias, incoming_path)),
        ),
        CommandStep(
            "install_perception_artifact",
            ssh_command,
            contract.canonical_json_bytes(install_request),
        ),
    )
    return DataSSHPlan(
        agent_sha256=agent_sha256,
        contract_sha256=contract_sha256,
        data_kind=data_kind,
        expected_final_release=policy["releases_root"] + "/" + release_id,
        expected_manifest_sha256=manifest_sha256,
        expected_payload_sha256=payload_sha256,
        expected_release_id=release_id,
        host_alias=host_alias,
        incoming_path=incoming_path,
        operation="install_perception_artifact",
        reservation_id=owner,
        releases_root=policy["releases_root"],
        remote_command=remote_command,
        source_path=source.as_posix(),
        steps=steps,
        target_role=target_role,
    )


def _parse_response(step: CommandStep, completed: object, plan: SSHPlan) -> Dict[str, object]:
    raw = bytes(completed.stdout or b"")
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentExecutionError("%s returned invalid JSON" % step.name) from exc
    if raw != contract.canonical_json_bytes(response) or not isinstance(response, dict):
        raise DeploymentExecutionError("%s returned non-canonical JSON" % step.name)
    operation = json.loads(step.stdin)["operation"]
    expected_fields = {
        "bootstrap_canary": {"schema", "target_role"},
        "prepare_incoming": {
            "incoming_path", "reservation_id", "schema", "target_role",
        },
        "inspect_canary": {"current_target", "release_ids", "schema", "target_role"},
        "install_release": {
            "created_release", "current_link", "final_release", "release_id", "schema",
            "target_role",
        },
        "rollback_canary": {"rolled_back_to", "schema", "target_role"},
        "cleanup_canary": {
            "removed_incoming", "removed_release_id", "schema", "target_role",
        },
    }[operation]
    if set(response) != expected_fields or response.get("target_role") != plan.target_role:
        raise DeploymentExecutionError("%s returned fields or role outside policy" % step.name)
    schema = {
        "bootstrap_canary": "wam_remote_bootstrap_response_v1",
        "prepare_incoming": "wam_remote_prepare_response_v1",
        "inspect_canary": "wam_remote_canary_inspection_v1",
        "install_release": "wam_remote_install_response_v1",
        "rollback_canary": "wam_remote_canary_rollback_v1",
        "cleanup_canary": "wam_remote_canary_cleanup_v1",
    }[operation]
    if response.get("schema") != schema:
        raise DeploymentExecutionError("%s returned the wrong schema" % step.name)
    if operation == "prepare_incoming" and (
        response["incoming_path"] != plan.incoming_path
        or response["reservation_id"] != plan.reservation_id
    ):
        raise DeploymentExecutionError("prepare response ownership mismatch")
    if operation == "inspect_canary":
        if response["current_target"] is not None and not isinstance(response["current_target"], str):
            raise DeploymentExecutionError("inspection current target is invalid")
        if not isinstance(response["release_ids"], list) or any(
            not isinstance(item, str) or not _RELEASE_ID.fullmatch(item)
            for item in response["release_ids"]
        ):
            raise DeploymentExecutionError("inspection release list is invalid")
        if response["release_ids"] != sorted(set(response["release_ids"])):
            raise DeploymentExecutionError("inspection release list is not canonical")
        if response["current_target"] is not None:
            target = PurePosixPath(response["current_target"])
            if (
                target.parts[:1] != ("releases",)
                or len(target.parts) != 2
                or target.parts[1] not in response["release_ids"]
            ):
                raise DeploymentExecutionError("inspection current target is invalid")
    if operation == "install_release":
        expected_final = plan.releases_root + "/" + plan.expected_release_id
        if (
            response["release_id"] != plan.expected_release_id
            or response["current_link"] != plan.current_link
            or response["final_release"] != expected_final
            or not isinstance(response["created_release"], bool)
        ):
            raise DeploymentExecutionError("install response identity or path mismatch")
    if operation == "rollback_canary":
        if response["rolled_back_to"] != json.loads(step.stdin)["previous_target"]:
            raise DeploymentExecutionError("rollback response target mismatch")
    if operation == "cleanup_canary":
        request = json.loads(step.stdin)
        if (
            response["removed_incoming"] != request["incoming_path"]
            or response["removed_release_id"] != request["created_release_id"]
        ):
            raise DeploymentExecutionError("cleanup response identity mismatch")
    return response


def _run_step(step: CommandStep, runner: Callable[..., object], plan: SSHPlan) -> Dict[str, object]:
    completed = runner(
        list(step.argv), input=step.stdin if step.stdin else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        stderr = bytes(completed.stderr or b"").decode("utf-8", "replace").strip()
        raise DeploymentExecutionError(
            "%s failed with exit %s: %s" % (step.name, completed.returncode, stderr)
        )
    return _parse_response(step, completed, plan) if step.stdin else {}


def _control_step(plan: SSHPlan, name: str, operation: str, **extra: object) -> CommandStep:
    request = {
        "agent_sha256": plan.agent_sha256,
        "contract_sha256": plan.contract_sha256,
        "operation": operation,
        "schema": "wam_remote_control_request_v1",
        "scope": "canary",
        "target_role": plan.target_role,
    }
    request.update(extra)
    ssh_argv = plan.steps[2].argv
    return CommandStep(name, ssh_argv, contract.canonical_json_bytes(request))


def execute_ssh_plan(
    plan: SSHPlan,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> Dict[str, object]:
    """Run an isolated canary, force rollback, and clean only state created by this run."""
    evidence: Dict[str, object] = {"schema": "wam_ssh_canary_evidence_v1"}
    primary: Optional[BaseException] = None
    pre: Optional[Dict[str, object]] = None
    install: Optional[Dict[str, object]] = None
    install_attempted = False
    reservation_owned = False
    rollback_state_known = False
    rollback_expected_target: Optional[str] = None
    try:
        for step in plan.steps:
            if step.name == "install":
                install_attempted = True
            response = _run_step(step, runner, plan)
            if step.name == "pre_inspect":
                pre = response
                evidence["pre_inspect"] = response
            elif step.name == "prepare_incoming":
                reservation_owned = True
                evidence["prepare_incoming"] = response
            elif step.name == "install":
                install = response
                evidence["install"] = response
                rollback_state_known = True
                rollback_expected_target = "releases/" + plan.expected_release_id
                expected_created = plan.expected_release_id not in pre["release_ids"]
                if response["created_release"] is not expected_created:
                    raise DeploymentExecutionError("install creation evidence mismatch")
            elif step.name == "post_inspect":
                evidence["post_inspect"] = response
                expected_target = "releases/" + plan.expected_release_id
                expected_releases = sorted(set(pre["release_ids"]) | {plan.expected_release_id})
                if (
                    response["current_target"] != expected_target
                    or response["release_ids"] != expected_releases
                ):
                    raise DeploymentExecutionError("post-inspection canary state mismatch")
    except BaseException as exc:
        primary = exc
    finally:
        if pre is not None:
            created_release_id = (
                plan.expected_release_id
                if install is not None and install["created_release"]
                else None
            )
            rollback_is_safe = True
            if install_attempted and install is None:
                post_inspect = next(
                    step for step in plan.steps if step.name == "post_inspect"
                )
                try:
                    ambiguous = _run_step(post_inspect, runner, plan)
                    evidence["ambiguous_install_inspect"] = ambiguous
                    pre_releases = set(pre["release_ids"])
                    observed_releases = set(ambiguous["release_ids"])
                    expected_releases = pre_releases | {plan.expected_release_id}
                    expected_target = "releases/" + plan.expected_release_id
                    expected_is_new = (
                        plan.expected_release_id not in pre_releases
                        and plan.expected_release_id in observed_releases
                    )
                    rollback_is_safe = (
                        observed_releases in (pre_releases, expected_releases)
                        and ambiguous["current_target"]
                        in (pre["current_target"], expected_target)
                    )
                    evidence["ambiguous_install_state"] = {
                        "current_changed": (
                            ambiguous["current_target"] != pre["current_target"]
                        ),
                        "expected_release_newly_exists": expected_is_new,
                        "state_matches_run_boundary": rollback_is_safe,
                    }
                    if rollback_is_safe and expected_is_new:
                        created_release_id = plan.expected_release_id
                    if rollback_is_safe:
                        rollback_state_known = True
                        rollback_expected_target = ambiguous["current_target"]
                except BaseException as exc:
                    rollback_is_safe = False
                    evidence["ambiguous_install_inspect_error"] = str(exc)
            rollback = _control_step(
                plan, "forced_rollback", "rollback_canary",
                expected_current_target=rollback_expected_target,
                previous_target=pre["current_target"],
            )
            cleanup = _control_step(
                plan, "cleanup", "cleanup_canary",
                incoming_path=plan.incoming_path,
                reservation_id=plan.reservation_id,
                created_release_id=created_release_id,
                pre_current_target=pre["current_target"],
                pre_release_ids=pre["release_ids"],
            )
            control_steps = []
            if install_attempted and rollback_is_safe and rollback_state_known:
                control_steps.append(("rollback", rollback))
            else:
                evidence["rollback_skipped"] = (
                    "install not attempted" if not install_attempted
                    else "ambiguous remote state"
                )
            if reservation_owned:
                control_steps.append(("cleanup", cleanup))
            else:
                evidence["cleanup_skipped"] = "incoming reservation ownership not established"
            for key, step in control_steps:
                try:
                    evidence[key] = _run_step(step, runner, plan)
                except BaseException as exc:
                    evidence[key + "_error"] = str(exc)
                    if primary is None:
                        primary = exc
    if primary is not None:
        if isinstance(primary, DeploymentExecutionError):
            primary.evidence.update(evidence)
            raise primary
        raise DeploymentExecutionError(str(primary), evidence=evidence) from primary
    return evidence


def _parse_production_response(
    step: CommandStep, completed: object, plan: ProductionSSHPlan,
) -> Dict[str, object]:
    try:
        raw = bytes(completed.stdout or b"")
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentExecutionError(
            "%s returned invalid JSON" % step.name
        ) from exc
    if not isinstance(response, dict) or raw != contract.canonical_json_bytes(response):
        raise DeploymentExecutionError("%s returned a non-canonical response" % step.name)
    operation = json.loads(step.stdin)["operation"]
    current_parent = PurePosixPath(plan.current_link).parent
    expected_compatibility = (
        (current_parent / "auscultation").as_posix()
        if plan.target_role == "linbox" else None
    )
    expected_current_target = PurePosixPath(
        os.path.relpath(
            plan.releases_root + "/" + plan.expected_release_id,
            current_parent.as_posix(),
        )
    ).as_posix()
    if operation == "prepare_production_incoming":
        fields = {
            "frozen_payload_sha256", "frozen_release_id", "incoming_path",
            "reservation_id", "schema", "target_role",
        }
        if set(response) != fields or (
            response.get("schema") != "wam_remote_production_prepare_response_v1"
            or response.get("target_role") != plan.target_role
            or response.get("incoming_path") != plan.incoming_path
            or response.get("reservation_id") != plan.reservation_id
            or response.get("frozen_release_id") != plan.expected_release_id
            or response.get("frozen_payload_sha256") != plan.expected_payload_sha256
        ):
            raise DeploymentExecutionError("production prepare response identity mismatch")
    elif operation == "activate_production_release":
        fields = {
            "compatibility_link", "created_release", "current_link", "final_release",
            "legacy_inventory_sha256", "legacy_path", "payload_sha256",
            "previous_current_target", "release_id", "schema", "target_role",
        }
        expected_final = plan.releases_root + "/" + plan.expected_release_id
        if set(response) != fields or (
            response.get("schema") != "wam_remote_production_activate_response_v1"
            or response.get("target_role") != plan.target_role
            or response.get("release_id") != plan.expected_release_id
            or response.get("payload_sha256") != plan.expected_payload_sha256
            or response.get("final_release") != expected_final
            or response.get("current_link") != plan.current_link
            or not isinstance(response.get("created_release"), bool)
        ):
            raise DeploymentExecutionError("production activation response identity mismatch")
        for key in (
            "compatibility_link", "legacy_inventory_sha256", "legacy_path",
            "previous_current_target",
        ):
            if response[key] is not None and not isinstance(response[key], str):
                raise DeploymentExecutionError("production activation response type mismatch")
        if response["compatibility_link"] != expected_compatibility:
            raise DeploymentExecutionError("production compatibility response path mismatch")
        legacy_path = response["legacy_path"]
        legacy_hash = response["legacy_inventory_sha256"]
        if plan.target_role == "h200":
            if legacy_path is not None or legacy_hash is not None:
                raise DeploymentExecutionError("H200 response must not contain linbox legacy state")
        elif legacy_path is None:
            if legacy_hash is not None:
                raise DeploymentExecutionError("linbox legacy response fields are inconsistent")
        else:
            legacy = PurePosixPath(legacy_path)
            if (
                legacy.parent != current_parent
                or re.fullmatch(
                    r"auscultation\.legacy-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{16}",
                    legacy.name,
                ) is None
                or not isinstance(legacy_hash, str)
                or not _HEX64.fullmatch(legacy_hash)
                or not legacy.name.endswith("-" + legacy_hash[:16])
            ):
                raise DeploymentExecutionError("linbox legacy response identity mismatch")
        previous = response["previous_current_target"]
        if isinstance(previous, str):
            parsed_previous = PurePosixPath(previous)
            if parsed_previous.is_absolute() or ".." in parsed_previous.parts:
                raise DeploymentExecutionError("previous production target is unsafe")
    elif operation == "reconcile_production_release":
        fields = {
            "compatibility_link", "current_target", "final_release", "legacy_path",
            "schema", "status", "target_role",
        }
        if set(response) != fields or (
            response.get("schema") != "wam_remote_production_reconcile_response_v1"
            or response.get("target_role") != plan.target_role
            or response.get("status") not in {"promoted", "cleaned"}
        ):
            raise DeploymentExecutionError("production reconcile response identity mismatch")
        for key in ("compatibility_link", "current_target", "final_release", "legacy_path"):
            if response[key] is not None and not isinstance(response[key], str):
                raise DeploymentExecutionError("production reconcile response type mismatch")
        if response["status"] == "promoted" and response["final_release"] != (
            plan.releases_root + "/" + plan.expected_release_id
        ):
            raise DeploymentExecutionError("reconciled production release path mismatch")
        if response["compatibility_link"] != expected_compatibility:
            raise DeploymentExecutionError("reconciled compatibility path mismatch")
        if (
            response["status"] == "promoted"
            and response["current_target"] != expected_current_target
        ):
            raise DeploymentExecutionError("reconciled production pointer mismatch")
        legacy_path = response["legacy_path"]
        if legacy_path is not None:
            legacy = PurePosixPath(legacy_path)
            if (
                plan.target_role != "linbox"
                or legacy.parent != current_parent
                or re.fullmatch(
                    r"auscultation\.legacy-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{16}",
                    legacy.name,
                ) is None
            ):
                raise DeploymentExecutionError("reconciled legacy path is outside policy")
    else:
        raise DeploymentExecutionError("unexpected production response operation")
    return response


def _run_production_step(
    step: CommandStep,
    runner: Callable[..., object],
    plan: ProductionSSHPlan,
) -> Dict[str, object]:
    completed = runner(
        list(step.argv), input=step.stdin if step.stdin else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        stderr = bytes(completed.stderr or b"").decode("utf-8", "replace").strip()
        raise DeploymentExecutionError(
            "%s failed with exit %s: %s" % (step.name, completed.returncode, stderr)
        )
    return _parse_production_response(step, completed, plan) if step.stdin else {}


def execute_production_ssh_plan(
    plan: ProductionSSHPlan,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> Dict[str, object]:
    """Execute only an already double-confirmed production plan, reconciling ambiguity."""
    evidence: Dict[str, object] = {
        "schema": "wam_ssh_production_evidence_v1",
        "status": "started",
        "target_role": plan.target_role,
    }
    prepare_attempted = False
    try:
        for step in plan.steps:
            if step.name == "prepare_production":
                prepare_attempted = True
            response = _run_production_step(step, runner, plan)
            if response:
                evidence[step.name] = response
        evidence["status"] = "activated"
        return evidence
    except BaseException as primary:
        if prepare_attempted:
            try:
                reconciled = _run_production_step(plan.reconcile_step, runner, plan)
                evidence["reconcile"] = reconciled
                if reconciled["status"] == "promoted":
                    evidence["status"] = "reconciled_promoted"
                    return evidence
            except BaseException as reconcile_error:
                evidence["reconcile_error"] = str(reconcile_error)
        evidence["status"] = "failed"
        if isinstance(primary, DeploymentExecutionError):
            primary.evidence.update(evidence)
            raise primary
        raise DeploymentExecutionError(str(primary), evidence=evidence) from primary


def _parse_data_response(
    step: CommandStep, completed: object, plan: DataSSHPlan
) -> Dict[str, object]:
    raw = bytes(completed.stdout or b"")
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentExecutionError("%s returned invalid JSON" % step.name) from exc
    if not isinstance(response, dict) or raw != contract.canonical_json_bytes(response):
        raise DeploymentExecutionError("%s returned non-canonical JSON" % step.name)
    operation = json.loads(step.stdin)["operation"]
    if operation == "prepare_data_incoming":
        if (
            set(response) != {"incoming_path", "reservation_id", "schema", "target_role"}
            or response.get("schema") != "wam_remote_data_prepare_response_v1"
            or response.get("target_role") != plan.target_role
            or response.get("incoming_path") != plan.incoming_path
            or response.get("reservation_id") != plan.reservation_id
        ):
            raise DeploymentExecutionError(
                "prepare_data_incoming returned fields, role, or path outside policy"
            )
        return response
    if operation == "prepare_perception_incoming":
        if (
            set(response) != {
                "data_kind", "incoming_path", "reservation_id", "schema",
                "source_role", "target_role",
            }
            or response.get("schema")
            != "wam_remote_perception_prepare_response_v1"
            or response.get("data_kind") != plan.data_kind
            or response.get("source_role") != "mac"
            or response.get("target_role") != plan.target_role
            or response.get("incoming_path") != plan.incoming_path
            or response.get("reservation_id") != plan.reservation_id
        ):
            raise DeploymentExecutionError(
                "prepare_perception_incoming returned fields or identity outside policy"
            )
        return response
    if operation == "reconcile_data_transfer":
        if (
            set(response) != {
                "final_release", "incoming_present", "marker_present", "schema",
                "status", "target_role",
            }
            or response.get("schema") != "wam_remote_data_reconcile_response_v1"
            or response.get("target_role") != plan.target_role
            or response.get("status") not in {"cleaned", "promoted"}
            or response.get("incoming_present") is not False
            or response.get("marker_present") is not False
            or response.get("final_release")
            != (plan.expected_final_release if response.get("status") == "promoted" else None)
        ):
            raise DeploymentExecutionError(
                "data reconciliation returned state outside policy"
            )
        return response
    if operation == "reconcile_perception_transfer":
        if (
            set(response) != {
                "artifact_kind", "final_release", "incoming_present",
                "marker_present", "schema", "status", "target_role",
            }
            or response.get("schema")
            != "wam_remote_perception_reconcile_response_v1"
            or response.get("artifact_kind") != plan.data_kind
            or response.get("target_role") != plan.target_role
            or response.get("status") not in {"cleaned", "promoted"}
            or response.get("incoming_present") is not False
            or response.get("marker_present") is not False
            or response.get("final_release")
            != (plan.expected_final_release if response.get("status") == "promoted" else None)
        ):
            raise DeploymentExecutionError(
                "perception reconciliation returned state outside policy"
            )
        return response
    if operation == "install_perception_artifact":
        if (
            operation != plan.operation
            or set(response) != {
                "artifact_kind", "created_release", "final_release",
                "manifest_sha256", "payload_sha256", "release_id", "schema",
                "source_role", "target_role",
            }
            or response.get("schema")
            != "wam_perception_artifact_install_response_v1"
            or response.get("artifact_kind") != plan.data_kind
            or type(response.get("created_release")) is not bool
            or response.get("final_release") != plan.expected_final_release
            or response.get("manifest_sha256") != plan.expected_manifest_sha256
            or response.get("payload_sha256") != plan.expected_payload_sha256
            or response.get("release_id") != plan.expected_release_id
            or response.get("source_role") != "mac"
            or response.get("target_role") != plan.target_role
        ):
            raise DeploymentExecutionError(
                "install_perception_artifact returned identity outside policy"
            )
        return response
    schema = {
        "archive_raw_session": "wam_raw_session_archive_response_v1",
        "install_converted_dataset": "wam_converted_dataset_install_response_v1",
    }.get(operation)
    if (
        operation != plan.operation
        or set(response) != {"final_release", "release_id", "schema", "target_role"}
        or response.get("schema") != schema
        or response.get("target_role") != plan.target_role
        or response.get("release_id") != plan.expected_release_id
        or response.get("final_release") != plan.expected_final_release
    ):
        raise DeploymentExecutionError(
            "%s returned fields, role, or final identity outside policy" % step.name
        )
    return response


def _data_reconcile_step(plan: DataSSHPlan) -> CommandStep:
    request = json.loads(plan.steps[-1].stdin)
    if plan.operation == "install_perception_artifact":
        request["operation"] = "reconcile_perception_transfer"
        request["schema"] = "wam_remote_perception_reconcile_request_v1"
        step_name = "reconcile_perception_transfer"
    else:
        request["data_operation"] = request["operation"]
        request["operation"] = "reconcile_data_transfer"
        request["schema"] = "wam_remote_data_reconcile_request_v1"
        step_name = "reconcile_data_transfer"
    return CommandStep(
        step_name,
        plan.steps[2].argv,
        contract.canonical_json_bytes(request),
    )


def _run_data_step(
    step: CommandStep, runner: Callable[..., object], plan: DataSSHPlan
) -> Dict[str, object]:
    completed = runner(
        list(step.argv), input=step.stdin if step.stdin else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        stderr = bytes(completed.stderr or b"").decode("utf-8", "replace").strip()
        raise DeploymentExecutionError(
            "%s failed with exit %s: %s" % (step.name, completed.returncode, stderr)
        )
    return _parse_data_response(step, completed, plan) if step.stdin else {}


def execute_data_ssh_plan(
    plan: DataSSHPlan,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> Dict[str, object]:
    """Execute one identity-bound linbox-to-H200 data transfer plan."""
    evidence: Dict[str, object] = {
        "operation": plan.operation,
        "schema": "wam_ssh_data_evidence_v1",
        "target_role": plan.target_role,
    }
    primary: Optional[BaseException] = None
    prepare_attempted = False
    try:
        for step in plan.steps:
            if step.name in {"prepare_data_incoming", "prepare_perception_incoming"}:
                prepare_attempted = True
            response = _run_data_step(step, runner, plan)
            if step.stdin:
                if step.name in {"prepare_data_incoming", "prepare_perception_incoming"}:
                    evidence["prepare"] = response
                else:
                    evidence["result"] = response
    except BaseException as exc:
        primary = exc
    finally:
        if primary is not None and prepare_attempted:
            try:
                evidence["reconcile"] = _run_data_step(
                    _data_reconcile_step(plan), runner, plan
                )
            except BaseException as exc:
                evidence["reconcile_error"] = str(exc)
    if primary is not None:
        if isinstance(primary, DeploymentExecutionError):
            primary.evidence.update(evidence)
            raise primary
        raise DeploymentExecutionError(str(primary), evidence=evidence) from primary
    if set(evidence) != {"operation", "prepare", "result", "schema", "target_role"}:
        raise DeploymentExecutionError("data plan did not produce complete evidence", evidence=evidence)
    return evidence


def build_source_allowlist(
    workspace_root: Union[str, Path],
    *,
    required_paths: Sequence[str] = DEFAULT_REQUIRED_PATHS,
) -> Tuple[str, ...]:
    """Return the audited code-release source set used by every transport."""
    try:
        return release_source.select_code_release_sources(
            workspace_root, required_paths=required_paths,
        )
    except release_source.ReleaseSourceError as exc:
        raise PolicyError(str(exc)) from exc


def deploy_local(
    source_release: Union[str, Path],
    target_root: Union[str, Path],
    *,
    target_role: str,
    staged_callback: Optional[Callable[[Path], None]] = None,
    post_activate: Optional[Callable[[Path], None]] = None,
) -> LocalDeploymentResult:
    """Exercise the full transfer/install protocol without SSH or remote mutation."""
    source = Path(source_release)
    contract.verify_release(source)
    root = Path(target_root)
    incoming_root = root / "incoming"
    incoming_root.mkdir(parents=True, exist_ok=True)
    incoming = incoming_root / ("incoming-%s" % uuid.uuid4().hex)
    final: Optional[Path] = None
    try:
        shutil.copytree(str(source), str(incoming), symlinks=True)
        if staged_callback is not None:
            staged_callback(incoming)
        contract.verify_release(incoming)
        final = contract.install_release(
            incoming,
            root / "releases",
            root / "current",
            target_role=target_role,
            post_activate=post_activate,
        )
        return LocalDeploymentResult(final_release=final, incoming_path=incoming)
    finally:
        shutil.rmtree(str(incoming), ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or execute an audited WAM SSH deployment")
    parser.add_argument("--source-release", required=True)
    parser.add_argument("--source-role", required=True, choices=("mac", "linbox"))
    parser.add_argument("--target-role", required=True, choices=("linbox", "h200"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--data-kind")
    parser.add_argument("--confirm-release")
    parser.add_argument("--confirm-payload-sha256")
    args = parser.parse_args(argv)
    if args.data_kind is not None:
        if args.production:
            raise PolicyError("perception data transport cannot use production code mode")
        perception_plan = build_perception_artifact_plan(
            args.source_release,
            data_kind=args.data_kind,
            source_role=args.source_role,
            target_role=args.target_role,
            host_alias=args.host,
        )
        if args.execute:
            if args.confirm_release != perception_plan.expected_release_id:
                raise PolicyError(
                    "--confirm-release must exactly match the perception release id"
                )
            if (
                args.confirm_payload_sha256
                != perception_plan.expected_payload_sha256
            ):
                raise PolicyError(
                    "--confirm-payload-sha256 must exactly match the perception payload"
                )
            response = execute_data_ssh_plan(perception_plan)
            print(contract.canonical_json_bytes(response).decode("utf-8"), end="")
        else:
            print(json.dumps({
                "data_kind": perception_plan.data_kind,
                "final_release": perception_plan.expected_final_release,
                "incoming_path": perception_plan.incoming_path,
                "manifest_sha256": perception_plan.expected_manifest_sha256,
                "payload_sha256": perception_plan.expected_payload_sha256,
                "release_id": perception_plan.expected_release_id,
                "remote_command": perception_plan.remote_command,
                "steps": [step.name for step in perception_plan.steps],
            }, indent=2, sort_keys=True))
        return 0
    if args.production:
        if args.source_role != "mac":
            raise PolicyError("production code deployment must be orchestrated from Mac")
        production_plan = build_production_ssh_plan(
            args.source_release,
            target_role=args.target_role,
            host_alias=args.host,
            confirm_release=args.confirm_release,
            confirm_payload_sha256=args.confirm_payload_sha256,
        )
        if args.execute:
            response = execute_production_ssh_plan(production_plan)
            print(contract.canonical_json_bytes(response).decode("utf-8"), end="")
        else:
            print(json.dumps({
                "frozen_payload_sha256": production_plan.expected_payload_sha256,
                "frozen_release_id": production_plan.expected_release_id,
                "incoming_path": production_plan.incoming_path,
                "production": True,
                "remote_command": production_plan.remote_command,
                "steps": [step.name for step in production_plan.steps],
            }, indent=2, sort_keys=True))
        return 0
    plan = build_ssh_plan(
        args.source_release,
        source_role=args.source_role,
        target_role=args.target_role,
        host_alias=args.host,
        scope="canary",
    )
    if args.execute:
        response = execute_ssh_plan(plan)
        print(contract.canonical_json_bytes(response).decode("utf-8"), end="")
    else:
        print(json.dumps({
            "incoming_path": plan.incoming_path,
            "phases": plan.phases,
            "remote_command": plan.remote_command,
            "steps": [list(step.argv) for step in plan.steps],
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
