"""Build a relocation-aware Linbox perception environment in its final path.

The build recipe is resolved before any environment bytes are created.  Its
SHA-256 selects ``releases/linbox-perception-<recipe-prefix>``; that directory
is atomically reserved and is never renamed.  Progress lives in an external
journal, while absence of ``PERCEPTION_ENVIRONMENT_CLONE_COMPLETE.json`` keeps
every interrupted build unpublishable.  The completion marker is written only
after offline installs, strict bytecode omission, smokes, immutability, and the
complete v3 RECORD/tree lock have succeeded.

This command is intended to run *on Linbox*.  It does not contain credentials,
does not contact H200, and only downloads the two explicitly pinned official
PyPI wheels when ``--allow-official-repair-download`` is supplied.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
import fcntl
import hashlib
import importlib.metadata
import json
import marshal
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import types
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import urllib.request
import zipfile

import activate_perception_environment as activation
import artifact_contract as contract
import perception_environment_manifest as environment_manifest
import publish_perception_runtime as publisher
import verify_perception_environment as verifier


LINBOX_PERCEPTION_ROOT = Path("/home/robot/wam_data/perception_envs")
WHEELHOUSE_MANIFEST_SCHEMA = "wam_perception_wheelhouse_manifest_v1"
MANAGED_WHEELHOUSE_MANIFEST_NAME = "WHEELHOUSE_MANIFEST.json"
MANAGED_WHEELHOUSE_COMPLETE_NAME = "WHEELHOUSE_COMPLETE.json"
MANAGED_WHEELHOUSE_MANIFEST_SCHEMA = "wam_python_wheelhouse_manifest_v1"
LEGACY_MANAGED_WHEELHOUSE_MANIFEST_SCHEMA = "wam_python_wheelhouse_v1"
MANAGED_WHEELHOUSE_PAYLOAD_SCHEMA = "wam_python_wheelhouse_payload_v1"
MANAGED_WHEELHOUSE_COMPLETE_SCHEMA = "wam_python_wheelhouse_complete_v1"
MANAGED_WHEELHOUSE_PAYLOAD_ALGORITHM = (
    "sha256-canonical-json-wheel-files-v1"
)
BUILD_JOURNAL_SCHEMA = "wam_perception_environment_build_journal_v1"
PREPARATION_RESULT_SCHEMA = "wam_perception_environment_preparation_v1"
SOURCE_SNAPSHOT_SCHEMA = "wam_perception_legacy_source_snapshot_v1"
STRONG_LOCK_NAME = "PERCEPTION_ENVIRONMENT_STRONG_LOCK.json"
INHERITED_VERIFICATION_NAME = "PERCEPTION_ENVIRONMENT_VERIFICATION.json"
INHERITED_ENVIRONMENT_METADATA = (
    verifier.CLONE_MARKER_NAME,
    STRONG_LOCK_NAME,
    INHERITED_VERIFICATION_NAME,
)
BUILD_LOCK_NAME = ".perception-environment-build.lock"
NUMPY_GENERATED_PYC_SOURCE = (
    "lib/python3.11/site-packages/numpy/distutils/conv_template.py"
)
# CPython 3.11 magic 3495, encoded as importlib.util.MAGIC_NUMBER.
CPYTHON_311_MAGIC = bytes.fromhex("a70d0d0a")
V9_CHECKPOINT_SHA256 = (
    "68037e6932fd7315af33722a69060f0d9838afbc1a60d6f0349da909917f6a14"
)
POSTERIOR_GATE_CHECKPOINT_SHA256 = (
    "013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8"
)
OPENCV_REPAIR_WHEEL_SHA256 = (
    "6b02611523803495003bd87362db3e1d2a0454a6a63025dc6658a9830570aa0d"
)

_ENVIRONMENT_ID = re.compile(r"^linbox-perception-[0-9a-f]{20}$")
_MANAGED_WHEELHOUSE_RELEASE_ID = re.compile(
    r"^[a-z0-9][a-z0-9-]*-[0-9a-f]{20}$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PURPOSES = frozenset(("gate-additions", "opencv-repair", "runtime-repairs"))
_NORMALIZE_DISTRIBUTION = re.compile(r"[-_.]+")


class PreparationError(RuntimeError):
    """A final-path environment candidate could not be prepared safely."""


class PreparationPolicyError(PreparationError):
    """A request violates the fixed reproducible-build policy."""


class PreparationConflictError(PreparationError):
    """An existing path or journal conflicts with the selected recipe."""


class PreparationVerificationError(PreparationError):
    """Observed bytes or runtime behavior do not satisfy the recipe."""


@dataclass(frozen=True)
class CandidatePlan:
    root: Path
    releases: Path
    journal_root: Path
    environment_id: str
    candidate: Path
    journal_path: Path
    recipe: dict


PhaseAction = Callable[[], Mapping[str, object]]
PhaseVerifier = Callable[[dict], None]


def _read_regular_bytes(path: Union[str, Path]) -> Tuple[bytes, os.stat_result]:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except FileNotFoundError as exc:
        raise PreparationVerificationError("required file is missing: %s" % candidate) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PreparationVerificationError("required file is not regular: %s" % candidate)
    try:
        with candidate.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise PreparationVerificationError("file changed while opening: %s" % candidate)
            chunks = []
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                chunks.append(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise PreparationVerificationError("file cannot be read: %s" % candidate) from exc
    identity_before = (
        before.st_dev, before.st_ino, before.st_size,
        getattr(before, "st_mtime_ns", int(before.st_mtime * 1_000_000_000)),
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_size,
        getattr(after, "st_mtime_ns", int(after.st_mtime * 1_000_000_000)),
    )
    if identity_before != identity_after:
        raise PreparationVerificationError("file changed while hashing: %s" % candidate)
    return b"".join(chunks), after


def _sha256_file(path: Union[str, Path]) -> Tuple[str, int]:
    data, info = _read_regular_bytes(path)
    return hashlib.sha256(data).hexdigest(), info.st_size


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PreparationPolicyError("%s must be a lowercase SHA-256" % label)
    return value


def normalized_absolute_path(value: Union[str, Path], label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        raise PreparationPolicyError("%s must be absolute" % label)
    normalized = Path(os.path.abspath(str(candidate)))
    if candidate != normalized:
        raise PreparationPolicyError("%s must be normalized" % label)
    return candidate


def _canonical_object(path: Union[str, Path], label: str) -> Tuple[dict, str]:
    candidate = Path(path)
    raw, _info = _read_regular_bytes(candidate)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationVerificationError("%s is not canonical JSON" % label) from exc
    if not isinstance(value, dict) or raw != contract.canonical_json_bytes(value):
        raise PreparationVerificationError("%s is not a canonical JSON object" % label)
    return value, digest


def _managed_canonical_object(
    path: Union[str, Path], label: str,
) -> Tuple[dict, str]:
    """Accept both canonical encodings emitted by the managed v1 producers."""

    candidate = Path(path)
    raw, _info = _read_regular_bytes(candidate)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationVerificationError("%s is not canonical JSON" % label) from exc
    if not isinstance(value, dict):
        raise PreparationVerificationError("%s is not a canonical JSON object" % label)
    canonical = contract.canonical_json_bytes(value)
    if raw not in (canonical, canonical[:-1]):
        raise PreparationVerificationError("%s is not a canonical JSON object" % label)
    return value, digest


def wheelhouse_manifest_sha256(value: Mapping[str, object]) -> str:
    """Hash a wheelhouse manifest without its circular display digest."""

    if not isinstance(value, Mapping):
        raise PreparationPolicyError("wheelhouse manifest must be an object")
    projection = {key: value[key] for key in value if key != "sha256"}
    return hashlib.sha256(contract.canonical_json_bytes(projection)).hexdigest()


def _validate_wheel_rows(rows: object) -> list:
    if not isinstance(rows, list):
        raise PreparationPolicyError("wheelhouse wheels must be a list")
    validated = []
    names = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"filename", "sha256", "size"}:
            raise PreparationPolicyError("wheelhouse wheel row is invalid")
        filename = row["filename"]
        if (
            not isinstance(filename, str)
            or not filename.endswith(".whl")
            or Path(filename).name != filename
            or filename in names
        ):
            raise PreparationPolicyError("wheelhouse wheel filename is invalid")
        names.add(filename)
        digest = _require_sha256(row["sha256"], "wheel SHA-256")
        size = row["size"]
        if type(size) is not int or size < 0:
            raise PreparationPolicyError("wheel size is invalid")
        validated.append({"filename": filename, "sha256": digest, "size": size})
    if [row["filename"] for row in validated] != sorted(names):
        raise PreparationPolicyError("wheelhouse wheels must be filename-sorted")
    return validated


def validate_wheelhouse_manifest(value: object, *, purpose: str) -> dict:
    if purpose not in _PURPOSES:
        raise PreparationPolicyError("wheelhouse purpose is unsupported")
    if not isinstance(value, Mapping) or set(value) != {
        "purpose", "schema", "sha256", "wheels",
    }:
        raise PreparationPolicyError("wheelhouse manifest shape is invalid")
    if value["schema"] != WHEELHOUSE_MANIFEST_SCHEMA or value["purpose"] != purpose:
        raise PreparationPolicyError("wheelhouse manifest identity is invalid")
    rows = _validate_wheel_rows(value["wheels"])
    normalized = {
        "purpose": purpose,
        "schema": WHEELHOUSE_MANIFEST_SCHEMA,
        "sha256": value["sha256"],
        "wheels": rows,
    }
    if normalized["sha256"] != wheelhouse_manifest_sha256(normalized):
        raise PreparationPolicyError("wheelhouse manifest SHA-256 mismatch")
    return normalized


def _managed_wheelhouse_payload_rows(rows: object) -> list:
    if not isinstance(rows, list):
        raise PreparationVerificationError(
            "managed wheelhouse payload wheels must be a list"
        )
    validated = []
    filenames = set()
    expected_keys = {
        "filename", "name", "requires_dist", "requires_python",
        "sha256", "size", "version",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise PreparationVerificationError(
                "managed wheelhouse payload row shape is invalid"
            )
        filename = row["filename"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".whl")
            or filename in filenames
        ):
            raise PreparationVerificationError(
                "managed wheelhouse payload filename is invalid"
            )
        filenames.add(filename)
        if (
            not isinstance(row["name"], str)
            or not row["name"]
            or not isinstance(row["version"], str)
            or not row["version"]
            or not isinstance(row["requires_python"], str)
            or not isinstance(row["requires_dist"], list)
            or not all(isinstance(item, str) for item in row["requires_dist"])
            or not isinstance(row["sha256"], str)
            or not _SHA256.fullmatch(row["sha256"])
            or type(row["size"]) is not int
            or row["size"] < 0
        ):
            raise PreparationVerificationError(
                "managed wheelhouse payload metadata is invalid"
            )
        validated.append(dict(row))
    if [row["filename"] for row in validated] != sorted(filenames):
        raise PreparationVerificationError(
            "managed wheelhouse payload wheels are not filename-sorted"
        )
    return validated


def _verify_managed_wheelhouse_metadata(root: Path, observed_rows: list) -> None:
    manifest_path = root / MANAGED_WHEELHOUSE_MANIFEST_NAME
    complete_path = root / MANAGED_WHEELHOUSE_COMPLETE_NAME
    manifest_present = manifest_path.exists() or manifest_path.is_symlink()
    complete_present = complete_path.exists() or complete_path.is_symlink()
    if not manifest_present and not complete_present:
        return
    if not manifest_present or not complete_present:
        raise PreparationVerificationError(
            "managed wheelhouse metadata pair is incomplete"
        )

    manifest, manifest_sha256 = _managed_canonical_object(
        manifest_path, "managed wheelhouse manifest"
    )
    if set(manifest) != {"payload", "payload_sha256", "release_id", "schema"}:
        raise PreparationVerificationError(
            "managed wheelhouse manifest shape is invalid"
        )
    payload = manifest["payload"]
    if not isinstance(payload, dict) or set(payload) not in (
        {"algorithm", "wheels"},
        {"algorithm", "schema", "wheels"},
    ):
        raise PreparationVerificationError(
            "managed wheelhouse payload shape is invalid"
        )
    payload_sha256 = manifest["payload_sha256"]
    release_id = manifest["release_id"]
    manifest_schema = manifest["schema"]
    legacy_schema = manifest_schema == LEGACY_MANAGED_WHEELHOUSE_MANIFEST_SCHEMA
    if (
        manifest_schema not in {
            MANAGED_WHEELHOUSE_MANIFEST_SCHEMA,
            LEGACY_MANAGED_WHEELHOUSE_MANIFEST_SCHEMA,
        }
        or (legacy_schema and "schema" in payload)
        or payload["algorithm"] != MANAGED_WHEELHOUSE_PAYLOAD_ALGORITHM
        or (
            "schema" in payload
            and payload["schema"] != MANAGED_WHEELHOUSE_PAYLOAD_SCHEMA
        )
        or not isinstance(payload_sha256, str)
        or not _SHA256.fullmatch(payload_sha256)
        or not isinstance(release_id, str)
        or not _MANAGED_WHEELHOUSE_RELEASE_ID.fullmatch(release_id)
        or not release_id.endswith("-" + payload_sha256[:20])
        or root.name != release_id
    ):
        raise PreparationVerificationError(
            "managed wheelhouse manifest identity is invalid"
        )

    # The first managed producer hashed the schema-free payload directly.  A
    # later OpenCV-only producer added the payload schema after selecting the
    # release digest.  Preserve that historical artifact, but never trust its
    # digest as the build input: the projection below independently re-hashes
    # every wheel and is what the v3 recipe binds.
    if "schema" not in payload:
        payload_raw = contract.canonical_json_bytes(payload)
        observed_payload_sha256 = {
            hashlib.sha256(payload_raw).hexdigest(),
            hashlib.sha256(payload_raw[:-1]).hexdigest(),
        }
        if payload_sha256 not in observed_payload_sha256:
            raise PreparationVerificationError(
                "managed wheelhouse payload SHA-256 mismatch"
            )

    managed_rows = _managed_wheelhouse_payload_rows(payload["wheels"])
    managed_projection = [
        {
            "filename": row["filename"],
            "sha256": row["sha256"],
            "size": row["size"],
        }
        for row in managed_rows
    ]
    if managed_projection != observed_rows:
        raise PreparationVerificationError(
            "managed wheelhouse payload does not bind observed wheel bytes"
        )

    complete, _complete_sha256 = _managed_canonical_object(
        complete_path, "managed wheelhouse completion"
    )
    expected_complete = {
        "manifest_sha256": manifest_sha256,
        "payload_sha256": payload_sha256,
        "release_id": release_id,
        "schema": MANAGED_WHEELHOUSE_COMPLETE_SCHEMA,
        "verified_wheels": len(observed_rows),
    }
    if complete != expected_complete:
        raise PreparationVerificationError(
            "managed wheelhouse completion does not bind the verified release"
        )


def inventory_wheelhouse(path: Union[str, Path], *, purpose: str) -> dict:
    root = Path(path)
    try:
        info = root.lstat()
    except FileNotFoundError as exc:
        raise PreparationPolicyError("wheelhouse must be a real directory") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PreparationPolicyError("wheelhouse must be a real directory")
    rows = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if entry.name in {
            MANAGED_WHEELHOUSE_MANIFEST_NAME,
            MANAGED_WHEELHOUSE_COMPLETE_NAME,
        }:
            continue
        if not entry.name.endswith(".whl"):
            raise PreparationPolicyError("wheelhouse contains a non-wheel entry")
        digest, size = _sha256_file(entry)
        rows.append({"filename": entry.name, "sha256": digest, "size": size})
    _verify_managed_wheelhouse_metadata(root, rows)
    value = {
        "purpose": purpose,
        "schema": WHEELHOUSE_MANIFEST_SCHEMA,
        "wheels": rows,
    }
    value["sha256"] = wheelhouse_manifest_sha256(value)
    return validate_wheelhouse_manifest(value, purpose=purpose)


def exact_repair_wheelhouse_manifest() -> dict:
    value = {
        "algorithm": publisher.REPAIR_WHEELHOUSE_ALGORITHM,
        "wheels": [dict(row) for row in publisher.EXACT_REPAIR_WHEELS],
    }
    value["sha256"] = wheelhouse_manifest_sha256(value)
    return value


def _validate_repair_manifest(value: object) -> dict:
    expected = exact_repair_wheelhouse_manifest()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise PreparationPolicyError("runtime repair wheelhouse manifest is not exact")
    return expected


def build_recipe(
    *, source_environment: Mapping[str, object],
    gate_wheelhouse_manifest: Mapping[str, object],
    opencv_wheelhouse_manifest: Mapping[str, object],
    repair_wheelhouse_manifest: Mapping[str, object],
) -> dict:
    gate = validate_wheelhouse_manifest(
        gate_wheelhouse_manifest, purpose="gate-additions"
    )
    opencv = validate_wheelhouse_manifest(
        opencv_wheelhouse_manifest, purpose="opencv-repair"
    )
    repair = _validate_repair_manifest(repair_wheelhouse_manifest)
    source = dict(source_environment)
    if set(source) != {
        "environment_id", "lock_sha256", "marker_sha256", "path",
        "snapshot_sha256",
    }:
        raise PreparationPolicyError("source environment identity is incomplete")
    _require_sha256(source["snapshot_sha256"], "source snapshot SHA-256")
    recipe = {
        "algorithm": publisher.FINAL_PATH_BUILD_RECIPE_ALGORITHM,
        "operations": list(publisher.FINAL_PATH_BUILD_OPERATIONS),
        "source_environment": source,
        "target": {
            "abi": "cp311",
            "cuda_build": "12.8",
            "platform": "linux-x86_64",
            "python_version": "3.11.15",
            "releases_root": publisher.LINBOX_PERCEPTION_RELEASES_ROOT,
            "strategy": "reserve-and-build-in-final-path-no-rename",
        },
        "wheelhouse_manifests": [
            {"purpose": "gate-additions", "sha256": gate["sha256"]},
            {"purpose": "opencv-repair", "sha256": opencv["sha256"]},
            {"purpose": "runtime-repairs", "sha256": repair["sha256"]},
        ],
    }
    recipe["sha256"] = publisher.build_recipe_sha256(recipe)
    try:
        publisher.validate_final_path_build_recipe(recipe)
    except publisher.PerceptionRuntimeError as exc:
        raise PreparationPolicyError(str(exc)) from exc
    return recipe


def environment_id_for_recipe(recipe: Mapping[str, object]) -> str:
    try:
        value = publisher.validate_final_path_build_recipe(recipe)
    except publisher.PerceptionRuntimeError as exc:
        raise PreparationPolicyError(str(exc)) from exc
    return "linbox-perception-" + str(value["sha256"])[:20]


def candidate_plan(*, root: Union[str, Path], recipe: Mapping[str, object]) -> CandidatePlan:
    base = Path(os.path.abspath(str(root)))
    if not base.is_absolute():  # pragma: no cover - abspath is defensive
        raise PreparationPolicyError("perception root must be absolute")
    normalized_recipe = dict(recipe)
    environment_id = environment_id_for_recipe(normalized_recipe)
    releases = base / "releases"
    journal_root = base / ".staging/build_journal" / environment_id
    return CandidatePlan(
        root=base,
        releases=releases,
        journal_root=journal_root,
        environment_id=environment_id,
        candidate=releases / environment_id,
        journal_path=journal_root / "BUILD_JOURNAL.json",
        recipe=normalized_recipe,
    )


def _require_real_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise PreparationPolicyError("%s must be a real directory" % label) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PreparationPolicyError("%s must be a real directory" % label)


class _RootBuildLock:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.fd = -1

    def __enter__(self) -> "_RootBuildLock":
        journal_parent = self.root / ".staging/build_journal"
        _require_real_directory(self.root, "perception environment root")
        staging = self.root / ".staging"
        _require_real_directory(staging, "perception staging root")
        if not os.path.lexists(str(journal_parent)):
            try:
                os.mkdir(str(journal_parent), 0o700)
                contract._fsync_directory(staging)
            except FileExistsError:
                pass
        _require_real_directory(journal_parent, "build journal root")
        _require_real_directory(self.root / "releases", "perception releases root")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(str(journal_parent / BUILD_LOCK_NAME), flags, 0o600)
        info = os.fstat(self.fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            os.close(self.fd)
            self.fd = -1
            raise PreparationPolicyError("build lock is not a private regular file")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = -1


def _load_journal(plan: CandidatePlan) -> dict:
    value, _digest = _canonical_object(plan.journal_path, "build journal")
    required = {
        "candidate", "environment_id", "journal_path", "phases", "recipe",
        "schema", "status",
    }
    if (
        set(value) != required
        or value.get("schema") != BUILD_JOURNAL_SCHEMA
        or value.get("candidate") != str(plan.candidate)
        or value.get("environment_id") != plan.environment_id
        or value.get("journal_path") != str(plan.journal_path)
        or value.get("recipe") != plan.recipe
        or value.get("status") not in {"INCOMPLETE", "COMPLETE"}
        or not isinstance(value.get("phases"), dict)
    ):
        raise PreparationConflictError("build journal does not match the selected recipe")
    return value


def _write_journal(plan: CandidatePlan, value: Mapping[str, object]) -> None:
    contract._atomic_write(plan.journal_path, contract.canonical_json_bytes(value))
    plan.journal_path.chmod(0o600)
    contract._fsync_directory(plan.journal_path.parent)


def _inherited_marker_evidence(plan: CandidatePlan) -> dict:
    marker_path = plan.candidate / verifier.CLONE_MARKER_NAME
    try:
        marker, marker_sha256 = verifier.load_clone_marker(marker_path)
    except verifier.EnvironmentVerificationError as exc:
        raise PreparationVerificationError(
            "candidate inherited marker is invalid"
        ) from exc
    source = plan.recipe["source_environment"]
    normalized = dict(marker)
    normalized["environment_root"] = source["path"]
    normalized_sha256 = hashlib.sha256(
        contract.canonical_json_bytes(normalized)
    ).hexdigest()
    if (
        marker.get("environment_id") != source["environment_id"]
        or marker.get("environment_root") != str(plan.candidate)
        or marker.get("lock_sha256") != source["lock_sha256"]
        or normalized_sha256 != source["marker_sha256"]
    ):
        raise PreparationVerificationError(
            "candidate marker is not the exact inherited source marker"
        )
    _digest, size = _sha256_file(marker_path)
    return {
        "path": verifier.CLONE_MARKER_NAME,
        "sha256": marker_sha256,
        "size": size,
    }


def _can_resume_with_inherited_marker(
    plan: CandidatePlan, journal: Mapping[str, object],
) -> bool:
    phases = journal.get("phases")
    if not isinstance(phases, Mapping):
        return False
    clone = phases.get("clone")
    sanitized = phases.get("sanitize-inherited-metadata")
    if (
        not isinstance(clone, Mapping)
        or clone.get("status") != "COMPLETE"
        or (
            isinstance(sanitized, Mapping)
            and sanitized.get("status") == "COMPLETE"
        )
    ):
        return False
    try:
        _inherited_marker_evidence(plan)
    except PreparationError:
        return False
    return True


def _inherited_strong_lock_evidence(plan: CandidatePlan) -> dict:
    """Bind a cloned lock to the source bytes plus Conda's exact path rewrite."""

    source = plan.recipe["source_environment"]
    source_lock_path = Path(source["path"]) / STRONG_LOCK_NAME
    try:
        source_lock, source_lock_sha256 = verifier.load_environment_lock(
            source_lock_path
        )
    except verifier.EnvironmentVerificationError as exc:
        raise PreparationVerificationError(
            "selected source strong lock is invalid"
        ) from exc
    if (
        source_lock_sha256 != source["lock_sha256"]
        or source_lock.get("environment_id") != source["environment_id"]
    ):
        raise PreparationVerificationError(
            "selected source strong lock does not match the build recipe"
        )
    source_raw, _source_info = _read_regular_bytes(source_lock_path)
    if hashlib.sha256(source_raw).hexdigest() != source_lock_sha256:
        raise PreparationVerificationError(
            "selected source strong lock changed while validating"
        )

    candidate_lock_path = plan.candidate / STRONG_LOCK_NAME
    candidate_raw, candidate_info = _read_regular_bytes(candidate_lock_path)
    expected_raw = source_raw.replace(
        str(source["path"]).encode("utf-8"),
        str(plan.candidate).encode("utf-8"),
    )
    if candidate_raw != expected_raw:
        raise PreparationVerificationError(
            "candidate strong lock is not the exact Conda path rewrite of the source"
        )
    try:
        candidate_value = json.loads(candidate_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationVerificationError(
            "candidate inherited strong lock is not canonical JSON"
        ) from exc
    if (
        not isinstance(candidate_value, dict)
        or candidate_raw != contract.canonical_json_bytes(candidate_value)
    ):
        raise PreparationVerificationError(
            "candidate inherited strong lock is not canonical JSON"
        )
    return {
        "path": STRONG_LOCK_NAME,
        "sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "size": candidate_info.st_size,
    }


def sanitize_inherited_environment_metadata(plan: CandidatePlan) -> dict:
    """Remove only source-bound publication metadata copied by Conda clone."""

    _require_real_directory(plan.candidate, "final-path candidate")
    source = plan.recipe["source_environment"]
    rows = []
    marker_path = plan.candidate / verifier.CLONE_MARKER_NAME
    if os.path.lexists(str(marker_path)):
        rows.append(_inherited_marker_evidence(plan))

    lock_path = plan.candidate / STRONG_LOCK_NAME
    if os.path.lexists(str(lock_path)):
        rows.append(_inherited_strong_lock_evidence(plan))

    verification_path = plan.candidate / INHERITED_VERIFICATION_NAME
    if os.path.lexists(str(verification_path)):
        digest, size = _sha256_file(verification_path)
        rows.append({
            "path": INHERITED_VERIFICATION_NAME,
            "sha256": digest,
            "size": size,
        })

    rows.sort(key=lambda row: row["path"])
    for row in rows:
        path = plan.candidate / row["path"]
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PreparationVerificationError(
                "inherited environment metadata is not a single-link regular file"
            )
    for row in rows:
        (plan.candidate / row["path"]).unlink()
    contract._fsync_directory(plan.candidate)
    for name in INHERITED_ENVIRONMENT_METADATA:
        if os.path.lexists(str(plan.candidate / name)):
            raise PreparationVerificationError(
                "inherited environment metadata removal did not complete"
            )
    return {"removed": rows}


def verify_sanitized_inherited_environment_metadata(
    plan: CandidatePlan, evidence: Mapping[str, object],
) -> None:
    if not isinstance(evidence, Mapping) or set(evidence) != {"removed"}:
        raise PreparationVerificationError(
            "inherited metadata sanitization evidence is invalid"
        )
    rows = evidence["removed"]
    if not isinstance(rows, list):
        raise PreparationVerificationError(
            "inherited metadata sanitization rows are invalid"
        )
    names = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256", "size"}
            or row["path"] not in INHERITED_ENVIRONMENT_METADATA
            or not isinstance(row["size"], int)
            or row["size"] < 0
        ):
            raise PreparationVerificationError(
                "inherited metadata sanitization row is invalid"
            )
        _require_sha256(row["sha256"], "inherited metadata")
        names.append(row["path"])
    if names != sorted(set(names)):
        raise PreparationVerificationError(
            "inherited metadata sanitization rows are not canonical"
        )
    for name in INHERITED_ENVIRONMENT_METADATA:
        if os.path.lexists(str(plan.candidate / name)):
            raise PreparationVerificationError(
                "inherited environment metadata remains in the candidate"
            )


def reserve_candidate(plan: CandidatePlan) -> dict:
    """Atomically reserve the recipe-selected final path or resume its journal."""

    with _RootBuildLock(plan.root):
        if plan.journal_root.is_symlink():
            raise PreparationPolicyError("build journal path must not be a symlink")
        plan.journal_root.mkdir(mode=0o700, parents=False, exist_ok=True)
        marker_path = plan.candidate / verifier.CLONE_MARKER_NAME
        journal_exists = plan.journal_path.exists()
        candidate_exists = os.path.lexists(str(plan.candidate))
        if candidate_exists:
            _require_real_directory(plan.candidate, "final-path candidate")
            if not journal_exists:
                raise PreparationConflictError(
                    "unowned incomplete final-path candidate already exists"
                )
            journal = _load_journal(plan)
            if marker_path.exists() and journal["status"] != "COMPLETE":
                finalizing = journal["phases"].get("finalize", {})
                if (
                    finalizing.get("status") not in {"RUNNING", "FAILED"}
                    and not _can_resume_with_inherited_marker(plan, journal)
                ):
                    raise PreparationConflictError(
                        "completion marker exists without a complete matching journal"
                    )
            if journal["status"] == "COMPLETE" and not marker_path.is_file():
                raise PreparationConflictError(
                    "complete journal is missing its completion marker"
                )
            return journal
        if journal_exists:
            raise PreparationConflictError(
                "build journal exists but its reserved candidate is missing"
            )
        try:
            os.mkdir(str(plan.candidate), 0o755)
        except FileExistsError as exc:
            raise PreparationConflictError(
                "final-path reservation lost a no-replace race"
            ) from exc
        contract._fsync_directory(plan.releases)
        journal = {
            "candidate": str(plan.candidate),
            "environment_id": plan.environment_id,
            "journal_path": str(plan.journal_path),
            "phases": {},
            "recipe": plan.recipe,
            "schema": BUILD_JOURNAL_SCHEMA,
            "status": "INCOMPLETE",
        }
        _write_journal(plan, journal)
        return journal


def run_resumable_phase(
    plan: CandidatePlan, name: str, *, action: PhaseAction, verify: PhaseVerifier,
) -> dict:
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
        raise PreparationPolicyError("build phase name is invalid")
    with _RootBuildLock(plan.root):
        journal = _load_journal(plan)
        if journal["status"] == "COMPLETE":
            raise PreparationConflictError("complete candidate cannot run another phase")
        existing = journal["phases"].get(name)
        if isinstance(existing, Mapping) and existing.get("status") == "COMPLETE":
            evidence = existing.get("evidence")
            if not isinstance(evidence, dict):
                raise PreparationConflictError("completed phase evidence is invalid")
            verify(dict(evidence))
            return dict(evidence)
        journal["phases"][name] = {"status": "RUNNING"}
        _write_journal(plan, journal)
        try:
            raw_evidence = action()
            if not isinstance(raw_evidence, Mapping):
                raise PreparationVerificationError("phase action did not return evidence")
            evidence = dict(raw_evidence)
            verify(evidence)
        except BaseException as exc:
            failed = _load_journal(plan)
            failed["phases"][name] = {
                "failure_type": type(exc).__name__,
                "status": "FAILED",
            }
            try:
                _write_journal(plan, failed)
            except BaseException:
                pass
            raise
        journal = _load_journal(plan)
        journal["phases"][name] = {
            "evidence": evidence,
            "evidence_sha256": hashlib.sha256(
                contract.canonical_json_bytes(evidence)
            ).hexdigest(),
            "status": "COMPLETE",
        }
        _write_journal(plan, journal)
        return evidence


def inspect_numpy_generated_pyc(
    environment_root: Union[str, Path],
) -> dict:
    root = Path(environment_root)
    source = root / NUMPY_GENERATED_PYC_SOURCE
    pyc = root / environment_manifest.GENERATED_PYC_OMISSION
    source_digest, source_size = _sha256_file(source)
    pyc_digest, _pyc_size = _sha256_file(pyc)
    raw = pyc.read_bytes()
    if len(raw) < 17 or raw[:4] != CPYTHON_311_MAGIC:
        raise PreparationVerificationError(
            "generated pyc does not use the exact CPython 3.11 magic"
        )
    flags, source_mtime, recorded_source_size = struct.unpack("<III", raw[4:16])
    if flags != 0:
        raise PreparationVerificationError(
            "generated pyc is not timestamp-based"
        )
    if recorded_source_size != source_size:
        raise PreparationVerificationError(
            "generated pyc recorded source size does not match conv_template.py"
        )
    marshalled = raw[16:]
    try:
        code = marshal.loads(marshalled)
    except Exception as exc:
        raise PreparationVerificationError(
            "generated pyc does not contain a readable module code object"
        ) from exc
    if not isinstance(code, types.CodeType):
        raise PreparationVerificationError(
            "generated pyc does not contain a module code object"
        )
    filename = str(code.co_filename).replace("\\", "/")
    if not filename.endswith(NUMPY_GENERATED_PYC_SOURCE):
        raise PreparationVerificationError(
            "generated pyc co_filename does not end in the exact source-relative path"
        )
    inspection = {
        "code_filename": filename,
        "code_sha256": hashlib.sha256(marshalled).hexdigest(),
        "flags": flags,
        "header_sha256": hashlib.sha256(raw[:16]).hexdigest(),
        "magic_hex": raw[:4].hex(),
        "pyc_sha256": pyc_digest,
        "recorded_source_mtime": source_mtime,
        "recorded_source_size": recorded_source_size,
        "source_path": NUMPY_GENERATED_PYC_SOURCE,
        "source_sha256": source_digest,
    }
    evidence = environment_manifest.generated_pyc_omission_evidence(
        path=environment_manifest.GENERATED_PYC_OMISSION,
        source_path=NUMPY_GENERATED_PYC_SOURCE,
        source_sha256=source_digest,
        inspection_sha256=hashlib.sha256(
            contract.canonical_json_bytes(inspection)
        ).hexdigest(),
    )
    return {"inspection": inspection, "manifest_evidence": evidence}


def inspect_and_remove_numpy_generated_pyc(
    environment_root: Union[str, Path],
) -> dict:
    root = Path(environment_root)
    result = inspect_numpy_generated_pyc(root)
    pyc = root / environment_manifest.GENERATED_PYC_OMISSION
    pyc.unlink()
    if os.path.lexists(str(pyc)):
        raise PreparationVerificationError("generated pyc omission did not persist")
    contract._fsync_directory(pyc.parent)
    return result


def _validate_persisted_pyc_omission(
    root: Path, value: object,
) -> dict:
    if not isinstance(value, Mapping) or set(value) != {
        "inspection", "manifest_evidence",
    }:
        raise PreparationVerificationError("persisted pyc omission evidence is invalid")
    inspection = value["inspection"]
    evidence = value["manifest_evidence"]
    if not isinstance(inspection, Mapping) or not isinstance(evidence, Mapping):
        raise PreparationVerificationError("persisted pyc omission evidence is invalid")
    if evidence.get("path") != environment_manifest.GENERATED_PYC_OMISSION:
        raise PreparationVerificationError("persisted pyc omission path is invalid")
    if evidence.get("source_path") != NUMPY_GENERATED_PYC_SOURCE:
        raise PreparationVerificationError("persisted pyc source path is invalid")
    source_digest, _source_size = _sha256_file(root / NUMPY_GENERATED_PYC_SOURCE)
    if evidence.get("source_sha256") != source_digest:
        raise PreparationVerificationError("generated pyc source SHA-256 drift")
    inspection_sha256 = hashlib.sha256(
        contract.canonical_json_bytes(dict(inspection))
    ).hexdigest()
    if evidence.get("inspection_sha256") != inspection_sha256:
        raise PreparationVerificationError("generated pyc inspection SHA-256 drift")
    try:
        expected_evidence = environment_manifest.generated_pyc_omission_evidence(
            path=evidence.get("path"),
            source_path=evidence.get("source_path"),
            source_sha256=evidence.get("source_sha256"),
            inspection_sha256=evidence.get("inspection_sha256"),
        )
    except environment_manifest.ManifestError as exc:
        raise PreparationVerificationError("persisted pyc omission evidence is invalid") from exc
    if dict(evidence) != expected_evidence:
        raise PreparationVerificationError("persisted pyc omission evidence is noncanonical")
    return {"inspection": dict(inspection), "manifest_evidence": expected_evidence}


def persist_and_remove_numpy_generated_pyc(
    environment_root: Union[str, Path], *, evidence_path: Union[str, Path],
    inherited_evidence: Optional[Mapping[str, object]] = None,
) -> dict:
    """Persist the strict inspection before deletion and resume either crash edge."""

    root = Path(environment_root)
    path = Path(evidence_path)
    pyc = root / environment_manifest.GENERATED_PYC_OMISSION
    if path.exists():
        persisted, _digest = _canonical_object(path, "pyc omission evidence")
        result = _validate_persisted_pyc_omission(root, persisted)
        if pyc.exists():
            pyc_digest, _size = _sha256_file(pyc)
            if pyc_digest != result["inspection"].get("pyc_sha256"):
                raise PreparationVerificationError(
                    "generated pyc changed after its persisted inspection"
                )
            pyc.unlink()
            contract._fsync_directory(pyc.parent)
        if os.path.lexists(str(pyc)):
            raise PreparationVerificationError("generated pyc omission did not persist")
        return result

    if not os.path.lexists(str(pyc)) and inherited_evidence is not None:
        result = _validate_persisted_pyc_omission(root, inherited_evidence)
        contract._atomic_write(path, contract.canonical_json_bytes(result))
        path.chmod(0o600)
        contract._fsync_directory(path.parent)
        persisted, _digest = _canonical_object(path, "pyc omission evidence")
        return _validate_persisted_pyc_omission(root, persisted)

    result = inspect_numpy_generated_pyc(root)
    contract._atomic_write(path, contract.canonical_json_bytes(result))
    path.chmod(0o600)
    contract._fsync_directory(path.parent)
    # The evidence above is durable before this sole authorized cache deletion.
    pyc.unlink()
    contract._fsync_directory(pyc.parent)
    if os.path.lexists(str(pyc)):
        raise PreparationVerificationError("generated pyc omission did not persist")
    persisted, _digest = _canonical_object(path, "pyc omission evidence")
    return _validate_persisted_pyc_omission(root, persisted)


def load_source_pyc_omission_evidence(
    plan: CandidatePlan, source_lock: Mapping[str, object],
) -> dict:
    """Load a source omission proof only when its strong lock binds it."""

    source = plan.recipe["source_environment"]
    if (
        not isinstance(source_lock, Mapping)
        or source_lock.get("environment_id") != source["environment_id"]
    ):
        raise PreparationVerificationError(
            "source pyc omission evidence has the wrong environment identity"
        )
    provisioning = source_lock.get("provisioning")
    locked_evidence = (
        provisioning.get("generated_pyc_omission")
        if isinstance(provisioning, Mapping)
        else None
    )
    if not isinstance(locked_evidence, Mapping):
        raise PreparationVerificationError(
            "source strong lock lacks generated pyc omission evidence"
        )
    evidence_path = (
        plan.root / ".staging/build_journal" / source["environment_id"]
        / "PYC_OMISSION_EVIDENCE.json"
    )
    persisted, _digest = _canonical_object(
        evidence_path, "source pyc omission evidence"
    )
    source_root = Path(source["path"])
    result = _validate_persisted_pyc_omission(source_root, persisted)
    if result["manifest_evidence"] != dict(locked_evidence):
        raise PreparationVerificationError(
            "source pyc omission evidence does not match the source strong lock"
        )
    source_pyc = source_root / environment_manifest.GENERATED_PYC_OMISSION
    if os.path.lexists(str(source_pyc)):
        raise PreparationVerificationError(
            "source strong lock claims an omission but the generated pyc exists"
        )
    return result


def _verify_wheel(path: Path, row: Mapping[str, object]) -> None:
    digest, size = _sha256_file(path)
    if size != row["size"]:
        raise PreparationVerificationError(
            "wheel size mismatch: %s" % row["filename"]
        )
    if digest != row["sha256"]:
        raise PreparationVerificationError(
            "wheel %s SHA-256 mismatch" % row["filename"]
        )


def _resolve_candidate_python(path: Union[str, Path]) -> Path:
    python_path = Path(path)
    if not python_path.is_absolute() or python_path.parent.name != "bin":
        raise PreparationVerificationError("candidate Python path is not canonical")
    environment_root = python_path.parent.parent
    try:
        info = python_path.lstat()
    except FileNotFoundError as exc:
        raise PreparationVerificationError("candidate Python is missing") from exc
    if stat.S_ISLNK(info.st_mode):
        try:
            target = os.readlink(str(python_path))
        except OSError as exc:
            raise PreparationVerificationError("candidate Python symlink cannot be read") from exc
        if Path(target).is_absolute():
            raise PreparationVerificationError("candidate Python symlink escapes its environment")
    try:
        resolved = python_path.resolve(strict=True)
        resolved.relative_to(environment_root.resolve(strict=True))
    except ValueError as exc:
        raise PreparationVerificationError("candidate Python escapes its environment") from exc
    except OSError as exc:
        raise PreparationVerificationError("candidate Python cannot be resolved") from exc
    resolved_info = resolved.lstat()
    if not stat.S_ISREG(resolved_info.st_mode):
        raise PreparationVerificationError("candidate Python is not a regular file")
    return resolved


def offline_install_command(
    *, python: Union[str, Path], wheelhouse: Union[str, Path],
    manifest: Mapping[str, object], bootstrap_pip_wheel: Optional[Union[str, Path]] = None,
) -> list:
    python_path = Path(python)
    _resolve_candidate_python(python_path)
    root = Path(wheelhouse)
    purpose = manifest.get("purpose") if isinstance(manifest, Mapping) else None
    value = validate_wheelhouse_manifest(manifest, purpose=str(purpose))
    wheels = []
    for row in value["wheels"]:
        wheel = root / row["filename"]
        _verify_wheel(wheel, row)
        wheels.append(str(wheel))
    command = [
        str(python_path), "-m", "pip", "install", "--no-index", "--no-deps",
        "--force-reinstall", "--no-compile", *wheels,
    ]
    if bootstrap_pip_wheel is not None:
        bootstrap = Path(bootstrap_pip_wheel)
        if bootstrap.parent != root:
            raise PreparationPolicyError("bootstrap pip wheel must be in the exact wheelhouse")
    return command


def wheel_distribution_profiles(
    *, wheelhouse: Union[str, Path], manifest: Mapping[str, object],
) -> list:
    purpose = manifest.get("purpose") if isinstance(manifest, Mapping) else None
    value = validate_wheelhouse_manifest(manifest, purpose=str(purpose))
    root = Path(wheelhouse)
    profiles = []
    names = set()
    for row in value["wheels"]:
        wheel = root / row["filename"]
        _verify_wheel(wheel, row)
        try:
            with zipfile.ZipFile(wheel, "r") as archive:
                metadata_names = [
                    name for name in archive.namelist()
                    if name.endswith(".dist-info/METADATA")
                    and name.count("/") == 1
                ]
                if len(metadata_names) != 1:
                    raise PreparationVerificationError(
                        "wheel must contain exactly one dist-info METADATA: %s"
                        % row["filename"]
                    )
                metadata = BytesParser(policy=email_policy).parsebytes(
                    archive.read(metadata_names[0])
                )
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            raise PreparationVerificationError(
                "wheel metadata cannot be inspected: %s" % row["filename"]
            ) from exc
        raw_name = metadata.get("Name")
        version = metadata.get("Version")
        if not isinstance(raw_name, str) or not raw_name or not isinstance(version, str) or not version:
            raise PreparationVerificationError(
                "wheel Name/Version metadata is incomplete: %s" % row["filename"]
            )
        name = _NORMALIZE_DISTRIBUTION.sub("-", raw_name).lower()
        if name in names:
            raise PreparationVerificationError(
                "wheelhouse repeats a normalized distribution: %s" % name
            )
        names.add(name)
        profiles.append({"name": name, "version": version})
    return sorted(profiles, key=lambda row: row["name"])


def _probe_installed_profiles(python: Path, profiles: Sequence[Mapping[str, str]]) -> list:
    requested = [str(row["name"]) for row in profiles]
    program = (
        "import importlib.metadata,json,sys;"
        "names=json.loads(sys.argv[1]);rows=[];"
        "[(rows.append({'name':n,'version':importlib.metadata.version(n)})) "
        "for n in names];"
        "sys.stdout.write(json.dumps(rows,sort_keys=True,separators=(',',':')))"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", program, json.dumps(requested, separators=(",", ":"))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=120, env=_safe_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    try:
        value = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def install_wheelhouse_resumable(
    *, python: Union[str, Path], wheelhouse: Union[str, Path],
    manifest: Mapping[str, object],
    bootstrap_pip_wheel: Optional[Union[str, Path]] = None,
    skip_if_satisfied: bool = True,
    profile_probe: Callable[[Path, Sequence[Mapping[str, str]]], list] = _probe_installed_profiles,
    runner: Optional[Callable[..., Mapping[str, object]]] = None,
) -> dict:
    """Skip a completed/crash-resumed install only after a live version probe."""

    python_path = Path(python)
    profiles = wheel_distribution_profiles(
        wheelhouse=wheelhouse, manifest=manifest
    )
    observed = profile_probe(python_path, profiles)
    if skip_if_satisfied and observed == profiles:
        return {"profiles": profiles, "status": "ALREADY_SATISFIED"}
    command = offline_install_command(
        python=python_path,
        wheelhouse=wheelhouse,
        manifest=manifest,
        bootstrap_pip_wheel=bootstrap_pip_wheel,
    )
    environment = (
        _safe_environment(pythonpath=str(bootstrap_pip_wheel))
        if bootstrap_pip_wheel is not None else _safe_environment()
    )
    execute = _run_command if runner is None else runner
    command_evidence = execute(command, env=environment)
    if not isinstance(command_evidence, Mapping):
        raise PreparationVerificationError("offline install runner returned invalid evidence")
    if profile_probe(python_path, profiles) != profiles:
        raise PreparationVerificationError(
            "offline install did not establish the exact wheel versions"
        )
    return {
        "command": dict(command_evidence),
        "profiles": profiles,
        "status": "INSTALLED",
    }


def verify_installed_profiles_evidence(
    python: Union[str, Path], evidence: Mapping[str, object],
    *, profile_probe: Callable[[Path, Sequence[Mapping[str, str]]], list] = _probe_installed_profiles,
) -> None:
    profiles = evidence.get("profiles") if isinstance(evidence, Mapping) else None
    if not isinstance(profiles, list) or evidence.get("status") not in {
        "ALREADY_SATISFIED", "INSTALLED",
    }:
        raise PreparationVerificationError("offline install evidence is invalid")
    if profile_probe(Path(python), profiles) != profiles:
        raise PreparationVerificationError("installed wheel versions drifted")


def _probe_distribution_absent(python: Path, distribution: str) -> bool:
    program = (
        "import importlib.metadata,sys;"
        "name=sys.argv[1];"
        "\ntry:\n importlib.metadata.version(name)\nexcept importlib.metadata.PackageNotFoundError:\n sys.exit(0)\nelse:\n sys.exit(3)"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", program, distribution],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=120, env=_safe_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def ensure_distribution_absent_resumable(
    *, python: Union[str, Path], distribution: str,
    absence_probe: Callable[[Path, str], bool] = _probe_distribution_absent,
    runner: Optional[Callable[..., Mapping[str, object]]] = None,
) -> dict:
    python_path = Path(python)
    normalized = _NORMALIZE_DISTRIBUTION.sub("-", distribution).lower()
    if normalized != distribution or not distribution:
        raise PreparationPolicyError("distribution removal name is not normalized")
    if absence_probe(python_path, distribution):
        return {"distribution": distribution, "status": "ABSENT"}
    execute = _run_command if runner is None else runner
    command_evidence = execute([
        str(python_path), "-m", "pip", "uninstall", "--yes", distribution,
    ], env=_safe_environment())
    if not isinstance(command_evidence, Mapping):
        raise PreparationVerificationError("distribution removal returned invalid evidence")
    if not absence_probe(python_path, distribution):
        raise PreparationVerificationError("distribution remains installed after removal")
    return {
        "command": dict(command_evidence),
        "distribution": distribution,
        "status": "ABSENT",
    }


def verify_distribution_absent_evidence(
    python: Union[str, Path], evidence: Mapping[str, object], *,
    absence_probe: Callable[[Path, str], bool] = _probe_distribution_absent,
) -> None:
    distribution = evidence.get("distribution") if isinstance(evidence, Mapping) else None
    if (
        not isinstance(distribution, str)
        or evidence.get("status") != "ABSENT"
        or not absence_probe(Path(python), distribution)
    ):
        raise PreparationVerificationError("removed distribution is present or evidence drifted")


def download_verified_wheel(
    row: Mapping[str, object], destination: Union[str, Path], *,
    allow_download: bool,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> Path:
    required = {"filename", "name", "sha256", "size", "url", "version"}
    if not isinstance(row, Mapping) or set(row) != required:
        raise PreparationPolicyError("official repair wheel row is invalid")
    target_root = Path(destination)
    _require_real_directory(target_root, "repair wheelhouse")
    target = target_root / str(row["filename"])
    if target.exists():
        _verify_wheel(target, row)
        return target
    if not allow_download:
        raise PreparationPolicyError(
            "missing official repair wheel; pass --allow-official-repair-download"
        )
    url = str(row["url"])
    if not url.startswith("https://files.pythonhosted.org/"):
        raise PreparationPolicyError("repair wheel URL is not the pinned official host")
    temporary = target_root / (target.name + ".partial")
    if os.path.lexists(str(temporary)):
        temporary.unlink()
    try:
        response = opener(url, timeout=120)
        try:
            with temporary.open("xb") as output:
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > row["size"]:
                        raise PreparationVerificationError(
                            "official repair wheel exceeded pinned size"
                        )
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
        _verify_wheel(temporary, row)
        os.replace(str(temporary), str(target))
        target.chmod(0o444)
        contract._fsync_directory(target_root)
        _verify_wheel(target, row)
        return target
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _safe_environment(*, pythonpath: Optional[str] = None) -> dict:
    value = dict(os.environ)
    value.pop("PYTHONHOME", None)
    value.pop("PYTHONPATH", None)
    value["PYTHONDONTWRITEBYTECODE"] = "1"
    value["PYTHONNOUSERSITE"] = "1"
    if pythonpath is not None:
        value["PYTHONPATH"] = pythonpath
    return value


def _run_command(
    command: Sequence[str], *, timeout: int = 1800,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> dict:
    try:
        completed = subprocess.run(
            list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=timeout,
            env=dict(_safe_environment() if env is None else env),
            cwd=None if cwd is None else str(cwd),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreparationVerificationError("build command could not complete") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-2048:]
        raise PreparationVerificationError(
            "build command failed with exit %d: %s" % (completed.returncode, detail)
        )
    return {
        "argv_sha256": hashlib.sha256(
            contract.canonical_json_bytes(list(command))
        ).hexdigest(),
        "returncode": completed.returncode,
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }


def _make_tree_immutable(root: Path) -> dict:
    activation._validate_tree_symlinks(root)
    hardlinks = activation._validate_tree_hardlinks(root)
    activation._make_tree_immutable(root)
    contract._fsync_directory(root.parent)
    count = activation._verify_tree_immutable(root)
    return {"hardlinks": hardlinks, "immutable_entries": count}


def _temporarily_writable_root(root: Path):
    class _WritableRoot:
        def __enter__(self):
            info = root.lstat()
            self.mode = stat.S_IMODE(info.st_mode)
            root.chmod(self.mode | stat.S_IWUSR | stat.S_IXUSR)
            return root

        def __exit__(self, _type, _value, _traceback):
            root.chmod(self.mode)
            contract._fsync_directory(root.parent)

    return _WritableRoot()


def write_lock_and_completion_marker(
    *, candidate: Union[str, Path], lock: Mapping[str, object],
    marker: Mapping[str, object],
) -> dict:
    root = Path(candidate)
    try:
        validated_lock = publisher.validate_environment_lock(dict(lock))
    except publisher.PerceptionRuntimeError as exc:
        raise PreparationVerificationError("v3 lock is invalid: %s" % exc) from exc
    if validated_lock["schema"] != publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA:
        raise PreparationVerificationError("completion requires the exact v3 lock")
    try:
        verifier._validate_marker(dict(marker))
    except verifier.EnvironmentVerificationError as exc:
        raise PreparationVerificationError("completion marker is invalid") from exc
    if marker.get("schema") != verifier.RELOCATION_AWARE_CLONE_MARKER_SCHEMA:
        raise PreparationVerificationError("completion requires the v3 marker")
    lock_raw = contract.canonical_json_bytes(validated_lock)
    lock_sha256 = hashlib.sha256(lock_raw).hexdigest()
    if marker.get("lock_sha256") != lock_sha256:
        raise PreparationVerificationError("completion marker lock SHA-256 mismatch")
    lock_path = root / STRONG_LOCK_NAME
    marker_path = root / verifier.CLONE_MARKER_NAME
    if marker_path.exists():
        existing, existing_sha = verifier.load_clone_marker(marker_path)
        if existing != dict(marker):
            raise PreparationConflictError("different completion marker already exists")
        return {"lock_sha256": lock_sha256, "marker_sha256": existing_sha}
    if lock_path.exists():
        existing_lock, existing_sha = verifier.load_environment_lock(lock_path)
        if existing_lock != validated_lock or existing_sha != lock_sha256:
            raise PreparationConflictError("different strong lock already exists")
    else:
        contract._atomic_write(lock_path, lock_raw)
        lock_path.chmod(0o444)
        contract._fsync_directory(root)
    # This is deliberately the final write inside the environment namespace.
    contract._atomic_write(marker_path, contract.canonical_json_bytes(dict(marker)))
    marker_path.chmod(0o444)
    contract._fsync_directory(root)
    loaded, marker_sha256 = verifier.load_clone_marker(marker_path)
    if loaded != dict(marker):
        raise PreparationVerificationError("completion marker changed after write")
    return {"lock_sha256": lock_sha256, "marker_sha256": marker_sha256}


def build_lock_from_observation(
    *, recipe: Mapping[str, object], observation: Mapping[str, object],
    generated_pyc_omission: Mapping[str, object],
    source_environment_fingerprint_sha256: str,
) -> dict:
    environment_id = environment_id_for_recipe(recipe)
    modules = {
        name: {
            key: observation["modules"][name][key]
            for key in ("algorithm", "count", "sha256")
        }
        for name in sorted(observation["modules"])
    }
    ffmpeg = {
        key: observation["ffmpeg"][key]
        for key in (
            "distribution", "mode", "path", "resolver", "sha256", "size",
            "version",
        )
    }
    repair = exact_repair_wheelhouse_manifest()
    lock = {
        "build_recipe": dict(recipe),
        "cuda": {
            "architectures": observation["cuda"]["architectures"],
            "build": observation["cuda"]["build"],
        },
        "environment_id": environment_id,
        "environment_tree": observation["environment_tree"],
        "ffmpeg": ffmpeg,
        "modules": modules,
        "packages": observation["packages"],
        "provisioning": {
            "generated_pyc_omission": dict(generated_pyc_omission),
            "mode": "verified_clone_with_exact_repairs",
            "repair_wheelhouse": {
                "algorithm": repair["algorithm"],
                "manifest_sha256": repair["sha256"],
                "wheels": repair["wheels"],
            },
            "required_service_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
            "source_environment_fingerprint_algorithm": (
                publisher.SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
            ),
            "source_environment_fingerprint_sha256": _require_sha256(
                source_environment_fingerprint_sha256,
                "source environment fingerprint",
            ),
            "wheelhouse_complete": False,
        },
        "python": {
            key: observation["python"][key]
            for key in (
                "abi", "executable_sha256", "implementation", "platform", "version",
            )
        },
        "record_observation": observation["record_observation"],
        "schema": publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA,
    }
    try:
        return publisher.validate_environment_lock(lock)
    except publisher.PerceptionRuntimeError as exc:
        raise PreparationVerificationError("generated v3 lock is invalid: %s" % exc) from exc


def source_fingerprint_from_verification_report(
    report: Mapping[str, object],
) -> str:
    if report.get("status") not in {"VERIFIED", "VERIFIED_WITH_SCHEMA_GAPS"}:
        raise PreparationVerificationError("source verification status is not verified")
    inventory = report.get("distribution_inventory")
    if not isinstance(inventory, Mapping):
        raise PreparationVerificationError(
            "source verification lacks a distribution inventory"
        )
    observed_inventory = inventory.get("observed", inventory)
    observation = {
        "cuda": report.get("cuda"),
        "distribution_inventory": observed_inventory,
        "modules": report.get("modules"),
        "packages": report.get("packages"),
        "python": report.get("python"),
    }
    try:
        return publisher.source_environment_fingerprint_sha256(observation)
    except publisher.PerceptionRuntimeError as exc:
        raise PreparationVerificationError(
            "source verification observation cannot be fingerprinted"
        ) from exc


def _source_tree_summary(value: Mapping[str, object]) -> dict:
    return {
        key: value[key]
        for key in (
            "algorithm", "directories", "entries_count", "regular_files",
            "sha256", "symlinks",
        )
    }


def _legacy_source_repair_debt(record: Mapping[str, object]) -> dict:
    missing = []
    for distribution in record.get("distributions", []):
        if not isinstance(distribution, Mapping):
            raise PreparationVerificationError(
                "source RECORD distribution row is invalid"
            )
        name = distribution.get("name")
        for claim in distribution.get("claims", []):
            if not isinstance(claim, Mapping):
                raise PreparationVerificationError(
                    "source RECORD claim row is invalid"
                )
            if claim.get("status") != "missing":
                continue
            installed = claim.get("installed_path")
            if (
                name != "pip"
                or not isinstance(installed, str)
                or (
                    installed != "bin/pip3.13"
                    and re.fullmatch(
                        r"lib/python3\.11/site-packages/pip/.*/__pycache__/"
                        r"[^/]+\.cpython-313\.pyc",
                        installed,
                    ) is None
                    and re.fullmatch(
                        r"lib/python3\.11/site-packages/pip/__pycache__/"
                        r"[^/]+\.cpython-313\.pyc",
                        installed,
                    ) is None
                )
            ):
                raise PreparationVerificationError(
                    "legacy source has unsupported missing RECORD debt"
                )
            missing.append({
                "distribution": name,
                "installed_path": installed,
                "record_path": claim.get("record_path"),
            })
    missing.sort(key=lambda row: (row["distribution"], row["installed_path"]))
    return {
        "allowed_profile": "conda-pip-26.1.2-cpython-313-cache-debt-v1",
        "missing_claim_count": len(missing),
        "missing_claims_sha256": hashlib.sha256(
            contract.canonical_json_bytes(missing)
        ).hexdigest(),
        "status_counts": dict(record["status_counts"]),
    }


def _validate_legacy_source_runtime(
    *, lock: Mapping[str, object], observation: Mapping[str, object],
) -> None:
    python = observation.get("python")
    if not isinstance(python, Mapping) or {
        key: python.get(key) for key in lock["python"]
    } != dict(lock["python"]):
        raise PreparationVerificationError("legacy source Python identity drifted")
    inventory = observation.get("distribution_inventory")
    if (
        not isinstance(inventory, Mapping)
        or inventory.get("count") != lock["distribution_inventory"]["count"]
    ):
        raise PreparationVerificationError(
            "legacy source distribution count drifted"
        )
    expected_versions = {
        row["name"]: row["version"] for row in lock["packages"]
    }
    observed_packages = observation.get("packages")
    if (
        not isinstance(observed_packages, list)
        or {row.get("name"): row.get("version") for row in observed_packages}
        != expected_versions
    ):
        raise PreparationVerificationError(
            "legacy source critical package versions drifted"
        )
    modules = observation.get("modules")
    if not isinstance(modules, Mapping) or set(modules) != set(lock["modules"]):
        raise PreparationVerificationError("legacy source module set drifted")
    for name, expected in lock["modules"].items():
        observed = modules[name]
        actual = (
            {key: observed.get(key) for key in ("algorithm", "count", "sha256")}
            if set(expected) == {"algorithm", "count", "sha256"}
            else {"sha256": observed.get("entrypoint_sha256")}
        )
        if actual != dict(expected):
            raise PreparationVerificationError(
                "legacy source module identity drifted: %s" % name
            )
    cuda = observation.get("cuda")
    if (
        not isinstance(cuda, Mapping)
        or {key: cuda.get(key) for key in ("build", "architectures")}
        != dict(lock["cuda"])
        or cuda.get("runtime_available") is not True
        or "sm_120" not in cuda.get("device_capabilities", [])
    ):
        raise PreparationVerificationError("legacy source CUDA identity drifted")


def collect_source_snapshot(
    *, lock_path: Union[str, Path], environment_root: Union[str, Path],
) -> dict:
    """Bind every current legacy source byte before an immutable clone build."""
    lock, lock_sha256 = verifier.load_environment_lock(lock_path)
    root = Path(environment_root).resolve(strict=True)
    if lock.get("schema") == publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA:
        report = verifier.verify_current_environment(
            lock_path=lock_path, environment_root=root,
        )
        if report.get("status") != "VERIFIED":
            raise PreparationVerificationError(
                "strong source environment is not verified"
            )
        return {
            "environment_id": lock["environment_id"],
            "environment_tree": report["environment_tree"],
            "lock_sha256": lock_sha256,
            "record_observation": report["record_observation"],
            "repair_debt": None,
            "schema": SOURCE_SNAPSHOT_SCHEMA,
            "source_fingerprint_sha256": lock["provisioning"][
                "source_environment_fingerprint_sha256"
            ],
        }

    observation = verifier.collect_current_environment(
        lock=lock, environment_root=root,
    )
    _validate_legacy_source_runtime(lock=lock, observation=observation)
    record = environment_manifest.collect_record_observation(
        importlib.metadata.distributions(), environment_root=root,
    )
    record = environment_manifest.validate_record_observation(record)
    tree = environment_manifest.collect_environment_tree(
        root, allow_generated_pyc=True, allow_hardlinks=True,
    )
    return {
        "environment_id": lock["environment_id"],
        "environment_tree": _source_tree_summary(tree),
        "lock_sha256": lock_sha256,
        "record_observation": verifier.record_observation_summary(record),
        "repair_debt": _legacy_source_repair_debt(record),
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "source_fingerprint_sha256": (
            publisher.source_environment_fingerprint_sha256(observation)
        ),
    }


def source_snapshot_sha256(value: Mapping[str, object]) -> str:
    if not isinstance(value, Mapping) or value.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
        raise PreparationVerificationError("source snapshot schema is invalid")
    return hashlib.sha256(contract.canonical_json_bytes(dict(value))).hexdigest()


def completed_inactive_result(
    *, plan: CandidatePlan, final_evidence: Mapping[str, object],
    source_evidence: Mapping[str, object],
) -> dict:
    return {
        "candidate": str(plan.candidate),
        "environment_id": plan.environment_id,
        "journal_path": str(plan.journal_path),
        "lock_sha256": _require_sha256(
            final_evidence.get("lock_sha256"), "final lock",
        ),
        "recipe_sha256": plan.recipe["sha256"],
        "schema": PREPARATION_RESULT_SCHEMA,
        "source_snapshot_sha256": _require_sha256(
            source_evidence.get("snapshot_sha256"), "source snapshot",
        ),
        "status": "COMPLETE_INACTIVE",
    }


def _completion_marker(root: Path, lock: Mapping[str, object]) -> dict:
    lock_sha256 = hashlib.sha256(contract.canonical_json_bytes(lock)).hexdigest()
    return {
        "complete": True,
        "environment_id": lock["environment_id"],
        "environment_root": str(root),
        "lock_sha256": lock_sha256,
        "observed_environment_tree": verifier.environment_tree_summary(
            lock["environment_tree"]
        ),
        "observed_record_observation": verifier.record_observation_summary(
            lock["record_observation"]
        ),
        "python_executable_relative": "bin/python",
        "required_service_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
        "schema": verifier.RELOCATION_AWARE_CLONE_MARKER_SCHEMA,
        "source_environment_fingerprint_sha256": lock["provisioning"][
            "source_environment_fingerprint_sha256"
        ],
    }


def _source_identity(
    *, root: Path, environment_id: str, lock_path: Path,
) -> Tuple[dict, dict, dict, dict]:
    if not _ENVIRONMENT_ID.fullmatch(environment_id):
        raise PreparationPolicyError("source environment_id is invalid")
    source = root / "releases" / environment_id
    expected = Path(publisher.LINBOX_PERCEPTION_RELEASES_ROOT) / environment_id
    if root == LINBOX_PERCEPTION_ROOT and source != expected:
        raise PreparationPolicyError("source release path is not canonical")
    _require_real_directory(source, "source environment")
    lock, lock_sha256 = verifier.load_environment_lock(lock_path)
    if lock.get("environment_id") != environment_id:
        raise PreparationVerificationError("source lock is not the selected release")
    marker, marker_sha256 = verifier.load_clone_marker(
        source / verifier.CLONE_MARKER_NAME
    )
    if (
        marker.get("environment_id") != environment_id
        or marker.get("environment_root") != str(source)
        or marker.get("lock_sha256") != lock_sha256
    ):
        raise PreparationVerificationError("source marker does not bind the source release")
    identity = {
        "environment_id": environment_id,
        "lock_sha256": lock_sha256,
        "marker_sha256": marker_sha256,
        "path": str(expected),
    }
    snapshot = collect_source_snapshot(
        lock_path=lock_path, environment_root=source,
    )
    if snapshot.get("environment_id") != environment_id:
        raise PreparationVerificationError(
            "source snapshot environment_id drifted"
        )
    identity["snapshot_sha256"] = source_snapshot_sha256(snapshot)
    return identity, lock, marker, snapshot


def _snapshot_wheelhouse(
    plan: CandidatePlan, *, purpose: str, source: Path, manifest: Mapping[str, object],
) -> Tuple[Path, dict]:
    value = validate_wheelhouse_manifest(manifest, purpose=purpose)
    destination = plan.journal_root / "inputs" / purpose
    if destination.exists():
        observed = inventory_wheelhouse(destination, purpose=purpose)
        if observed != value:
            raise PreparationConflictError("snapshotted wheelhouse drift: %s" % purpose)
        return destination, observed
    destination.mkdir(mode=0o700, parents=True)
    for row in value["wheels"]:
        source_wheel = source / row["filename"]
        _verify_wheel(source_wheel, row)
        target = destination / row["filename"]
        shutil.copyfile(str(source_wheel), str(target), follow_symlinks=False)
        target.chmod(0o400)
        _verify_wheel(target, row)
    destination.chmod(0o500)
    contract._fsync_directory(destination)
    observed = inventory_wheelhouse(destination, purpose=purpose)
    if observed != value:
        raise PreparationVerificationError("wheelhouse snapshot identity mismatch")
    return destination, observed


def _repair_install_manifest() -> dict:
    rows = [
        {"filename": row["filename"], "sha256": row["sha256"], "size": row["size"]}
        for row in publisher.EXACT_REPAIR_WHEELS
    ]
    value = {
        "purpose": "runtime-repairs",
        "schema": WHEELHOUSE_MANIFEST_SCHEMA,
        "wheels": sorted(rows, key=lambda row: row["filename"]),
    }
    value["sha256"] = wheelhouse_manifest_sha256(value)
    return value


def _internal_json(
    python: Path, script: Path, arguments: Sequence[str], *, timeout: int = 1800,
) -> dict:
    completed = subprocess.run(
        [str(python), str(script), *arguments],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        timeout=timeout, env=_safe_environment(),
    )
    if completed.returncode != 0:
        raise PreparationVerificationError(
            "candidate internal probe failed: %s"
            % completed.stderr.decode("utf-8", errors="replace")[-2048:]
        )
    try:
        value = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationVerificationError("candidate probe emitted invalid JSON") from exc
    if not isinstance(value, dict) or completed.stdout != contract.canonical_json_bytes(value):
        raise PreparationVerificationError("candidate probe emitted noncanonical JSON")
    return value


def _internal_collect(arguments: argparse.Namespace) -> int:
    spec, _digest = _canonical_object(arguments.spec, "collection specification")
    observation = verifier.collect_current_environment(
        lock=spec, environment_root=arguments.environment_root
    )
    sys.stdout.buffer.write(contract.canonical_json_bytes(observation))
    return 0


def _internal_model_smoke(arguments: argparse.Namespace) -> int:
    # Imports deliberately occur only inside the candidate interpreter.
    evidence_root = Path(arguments.evidence_root)
    _require_real_directory(evidence_root, "model smoke evidence root")
    os.environ["YOLO_CONFIG_DIR"] = str(evidence_root / "ultralytics-config")
    os.environ["TORCH_HOME"] = str(evidence_root / "torch-home")
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from ultralytics import YOLO

        code_root = Path(arguments.code_root)
        sys.path.insert(0, str(code_root / "auscultation"))
        from ausc.perception.tracking.detector import NormalsDetector
        from ausc.perception.tracking.geometry import Intrinsics

        for path, expected, label in (
            (Path(arguments.v9_checkpoint), V9_CHECKPOINT_SHA256, "v9 checkpoint"),
            (
                Path(arguments.posterior_gate_checkpoint),
                POSTERIOR_GATE_CHECKPOINT_SHA256,
                "posterior gate checkpoint",
            ),
        ):
            digest, _size = _sha256_file(path)
            if digest != expected:
                raise PreparationVerificationError("%s SHA-256 mismatch" % label)
        if not torch.cuda.is_available():
            raise PreparationVerificationError("CUDA is unavailable")
        capability = "sm_%d%d" % torch.cuda.get_device_capability()
        if capability != "sm_120" or "sm_120" not in torch.cuda.get_arch_list():
            raise PreparationVerificationError("candidate does not serve sm_120")
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        pose = YOLO(arguments.posterior_gate_checkpoint)
        pose.predict(blank, device=0, verbose=False)

        height, width = 576, 640
        yy, xx = np.mgrid[0:height, 0:width]
        radial = np.exp(
            -(((xx - width / 2.0) / (width * 0.28)) ** 2
              + ((yy - height / 2.0) / (height * 0.34)) ** 2)
        )
        depth = np.rint(
            (1.08 - 0.18 * radial + 0.04 * (yy / float(height) - 0.5))
            * 1000.0
        ).astype(np.uint16)
        intrinsics = Intrinsics(
            fx=560.0, fy=560.0, cx=(width - 1) / 2.0,
            cy=(height - 1) / 2.0, width=width, height=height,
            depth_scale=1.0,
        )
        detector = NormalsDetector(
            ckpt=arguments.v9_checkpoint, device="cuda",
            compute_confidence=False, half=True,
        )
        positive = detector.detect(
            np.zeros((height, width, 3), dtype=np.uint8),
            depth=depth, intr=intrinsics,
        )
        negative = detector.detect(
            np.zeros((height, width, 3), dtype=np.uint8),
            depth=np.full((height, width), 1000, dtype=np.uint16),
            intr=intrinsics,
        )
        if (
            positive is None or positive.back_present is not True
            or positive.xy_det.shape != (10, 2)
            or not np.isfinite(positive.xy_det).all()
        ):
            raise PreparationVerificationError("v9 positive inference failed")
        if negative is not None and negative.back_present is not False:
            raise PreparationVerificationError("v9 negative-depth inference failed")
        value = {
            "cuda_capability": capability,
            "landmark_count": 10,
            "posterior_gate_inference": "PASS",
            "status": "PASS",
            "v9_negative_depth": "PASS",
        }
    sys.stdout.buffer.write(contract.canonical_json_bytes(value))
    return 0


def _ffmpeg_concat_smoke(ffmpeg: Path, evidence_root: Path) -> dict:
    digest, size = _sha256_file(ffmpeg)
    if not os.access(str(ffmpeg), os.X_OK):
        raise PreparationVerificationError("bundled ffmpeg is not executable")
    with tempfile.TemporaryDirectory(prefix="ffmpeg-smoke-", dir=str(evidence_root)) as raw:
        root = Path(raw)
        segments = []
        for index, color in enumerate(("red", "blue")):
            segment = root / ("segment-%02d.mp4" % index)
            _run_command([
                str(ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "color=c=%s:s=320x240:r=30:d=1" % color,
                "-an", "-c:v", "mpeg4", "-q:v", "3", str(segment),
            ], timeout=120, cwd=root)
            segments.append(segment)
        concat = root / "concat.txt"
        concat.write_text(
            "".join("file '%s'\n" % segment.name for segment in segments),
            encoding="utf-8",
        )
        output = root / "joined.mp4"
        _run_command([
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "concat",
            "-safe", "1", "-i", concat.name, "-map", "0:v:0", "-c", "copy",
            "-movflags", "+faststart", output.name,
        ], timeout=120, env=_safe_environment(), cwd=root)
        _run_command([
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(output),
            "-f", "null", "-",
        ], timeout=120)
        output_digest, output_size = _sha256_file(output)
    return {
        "ffmpeg_sha256": digest,
        "ffmpeg_size": size,
        "joined_sha256": output_digest,
        "joined_size": output_size,
        "status": "PASS",
    }


def _execute_build(
    *, plan: CandidatePlan, source_lock_path: Path,
    source_lock: Mapping[str, object], source_marker: Mapping[str, object],
    source_snapshot: Mapping[str, object],
    gate_source: Path, gate_manifest: Mapping[str, object],
    opencv_source: Path, opencv_manifest: Mapping[str, object],
    conda_executable: Path, code_root: Path, v9_checkpoint: Path,
    posterior_gate_checkpoint: Path, allow_repair_download: bool,
) -> dict:
    reserve_candidate(plan)
    script = Path(__file__).resolve()
    source_root = Path(plan.recipe["source_environment"]["path"])
    expected_source_evidence = {
        "fingerprint_sha256": source_snapshot["source_fingerprint_sha256"],
        "snapshot_sha256": source_snapshot_sha256(source_snapshot),
    }

    def verify_live_source(evidence: Mapping[str, object]) -> None:
        activation._verify_tree_immutable(source_root)
        observed_snapshot = collect_source_snapshot(
            lock_path=source_lock_path, environment_root=source_root,
        )
        if dict(observed_snapshot) != dict(source_snapshot):
            raise PreparationVerificationError(
                "source snapshot changed after recipe selection"
            )
        observed_evidence = {
            "fingerprint_sha256": observed_snapshot[
                "source_fingerprint_sha256"
            ],
            "snapshot_sha256": source_snapshot_sha256(observed_snapshot),
        }
        if dict(evidence) != observed_evidence:
            raise PreparationVerificationError(
                "source snapshot evidence drifted"
            )
        if (
            observed_evidence["snapshot_sha256"]
            != plan.recipe["source_environment"]["snapshot_sha256"]
        ):
            raise PreparationVerificationError(
                "source snapshot evidence drifted"
            )

    def source_action() -> dict:
        return dict(expected_source_evidence)

    source_evidence = run_resumable_phase(
        plan, "verify-source", action=source_action,
        verify=verify_live_source,
    )

    snapshots = {}
    for purpose, source, manifest in (
        ("gate-additions", gate_source, gate_manifest),
        ("opencv-repair", opencv_source, opencv_manifest),
    ):
        def snapshot_action(purpose=purpose, source=source, manifest=manifest):
            path, observed = _snapshot_wheelhouse(
                plan, purpose=purpose, source=source, manifest=manifest
            )
            return {"manifest": observed, "path": str(path)}

        def snapshot_verify(row, purpose=purpose):
            if not isinstance(row.get("path"), str):
                raise PreparationVerificationError("wheelhouse snapshot path is invalid")
            expected_path = plan.journal_root / "inputs" / purpose
            if Path(row["path"]) != expected_path:
                raise PreparationVerificationError("wheelhouse snapshot path drift")
            validated = validate_wheelhouse_manifest(
                row["manifest"], purpose=purpose
            )
            if inventory_wheelhouse(expected_path, purpose=purpose) != validated:
                raise PreparationVerificationError("wheelhouse snapshot bytes drift")

        evidence = run_resumable_phase(
            plan, "snapshot-" + purpose,
            action=snapshot_action,
            verify=snapshot_verify,
        )
        snapshots[purpose] = (Path(evidence["path"]), evidence["manifest"])

    def repair_action() -> dict:
        destination = plan.journal_root / "inputs/runtime-repairs"
        if not destination.exists():
            destination.mkdir(mode=0o700, parents=True)
        else:
            destination.chmod(0o700)
        for row in publisher.EXACT_REPAIR_WHEELS:
            download_verified_wheel(
                row, destination, allow_download=allow_repair_download
            )
        destination.chmod(0o500)
        return {"manifest": exact_repair_wheelhouse_manifest(), "path": str(destination)}

    repair_evidence = run_resumable_phase(
        plan, "acquire-runtime-repairs", action=repair_action,
        verify=lambda row: (
            _validate_repair_manifest(row["manifest"]),
            [
                _verify_wheel(Path(row["path"]) / wheel["filename"], wheel)
                for wheel in publisher.EXACT_REPAIR_WHEELS
            ],
        ),
    )
    repair_root = Path(repair_evidence["path"])

    def clone_action() -> dict:
        verify_live_source(source_evidence)
        python = plan.candidate / "bin/python"
        if not python.exists():
            _run_command([
                str(conda_executable), "create", "--prefix", str(plan.candidate),
                "--clone", str(source_root), "--copy", "--offline", "--yes",
            ], timeout=3600)
        python_digest, python_size = _sha256_file(
            _resolve_candidate_python(python)
        )
        history_digest, history_size = _sha256_file(plan.candidate / "conda-meta/history")
        return {
            "history_sha256": history_digest, "history_size": history_size,
            "python_sha256": python_digest, "python_size": python_size,
        }

    def clone_verify(row: Mapping[str, object]) -> None:
        python_digest, python_size = _sha256_file(
            _resolve_candidate_python(plan.candidate / "bin/python")
        )
        history_digest, history_size = _sha256_file(
            plan.candidate / "conda-meta/history"
        )
        if dict(row) != {
            "history_sha256": history_digest,
            "history_size": history_size,
            "python_sha256": python_digest,
            "python_size": python_size,
        }:
            raise PreparationVerificationError("cloned environment evidence drift")

    clone_evidence = run_resumable_phase(
        plan, "clone", action=clone_action,
        verify=clone_verify,
    )
    run_resumable_phase(
        plan, "sanitize-inherited-metadata",
        action=lambda: sanitize_inherited_environment_metadata(plan),
        verify=lambda row: verify_sanitized_inherited_environment_metadata(
            plan, row
        ),
    )
    candidate_python = plan.candidate / "bin/python"

    def uninstall_action() -> dict:
        return ensure_distribution_absent_resumable(
            python=candidate_python,
            distribution="opencv-python-headless",
        )

    run_resumable_phase(
        plan, "remove-opencv-python-headless", action=uninstall_action,
        verify=lambda row: verify_distribution_absent_evidence(
            candidate_python, row
        ),
    )

    for purpose in ("opencv-repair", "gate-additions"):
        wheel_root, manifest = snapshots[purpose]

        def install_action(
            wheel_root=wheel_root, manifest=manifest, purpose=purpose,
        ):
            return install_wheelhouse_resumable(
                python=candidate_python,
                wheelhouse=wheel_root,
                manifest=manifest,
                skip_if_satisfied=(purpose != "opencv-repair"),
            )

        run_resumable_phase(
            plan, "install-" + purpose, action=install_action,
            verify=lambda row: verify_installed_profiles_evidence(
                candidate_python, row
            ),
        )

    repair_manifest = _repair_install_manifest()

    def install_repair_action() -> dict:
        pip_wheel = repair_root / "pip-26.1.2-py3-none-any.whl"
        return install_wheelhouse_resumable(
            python=candidate_python,
            wheelhouse=repair_root,
            manifest=repair_manifest,
            bootstrap_pip_wheel=pip_wheel,
            skip_if_satisfied=False,
        )

    run_resumable_phase(
        plan, "install-runtime-repairs", action=install_repair_action,
        verify=lambda row: verify_installed_profiles_evidence(
            candidate_python, row
        ),
    )

    def omit_pyc_action() -> dict:
        pyc = plan.candidate / environment_manifest.GENERATED_PYC_OMISSION
        inherited = None
        if not os.path.lexists(str(pyc)):
            inherited = load_source_pyc_omission_evidence(plan, source_lock)
        return persist_and_remove_numpy_generated_pyc(
            plan.candidate,
            evidence_path=plan.journal_root / "PYC_OMISSION_EVIDENCE.json",
            inherited_evidence=inherited,
        )

    pyc_evidence = run_resumable_phase(
        plan, "omit-generated-pyc",
        action=omit_pyc_action,
        verify=lambda row: (
            _validate_persisted_pyc_omission(plan.candidate, row),
            None
            if not os.path.lexists(
                str(plan.candidate / environment_manifest.GENERATED_PYC_OMISSION)
            )
            else (_ for _ in ()).throw(PreparationVerificationError("generated pyc omission drift")),
        ),
    )

    def smoke_action() -> dict:
        pip_check = _run_command([str(candidate_python), "-m", "pip", "check"])
        model = _internal_json(
            candidate_python, script,
            [
                "internal-model-smoke", "--code-root", str(code_root),
                "--evidence-root", str(plan.journal_root),
                "--v9-checkpoint", str(v9_checkpoint),
                "--posterior-gate-checkpoint", str(posterior_gate_checkpoint),
            ],
            timeout=3600,
        )
        ffmpeg = plan.candidate / publisher.BUNDLED_FFMPEG_RELATIVE_PATH
        concat = _ffmpeg_concat_smoke(ffmpeg, plan.journal_root)
        return {"concat": concat, "model": model, "pip_check": pip_check}

    run_resumable_phase(
        plan, "smoke", action=smoke_action,
        verify=lambda row: (
            None
            if row.get("model", {}).get("status") == "PASS"
            and row.get("concat", {}).get("status") == "PASS"
            and row.get("pip_check", {}).get("returncode") == 0
            else (_ for _ in ()).throw(PreparationVerificationError("runtime smoke evidence invalid"))
        ),
    )

    def finalize_action() -> dict:
        immutable = _make_tree_immutable(plan.candidate)
        spec = {
            "build_recipe": plan.recipe,
            "modules": {name: {} for name in publisher.RELOCATION_AWARE_SUPPORTED_MODULES},
            "packages": [
                {"name": name}
                for name in sorted(publisher.RELOCATION_AWARE_SUPPORTED_PACKAGES)
            ],
            "provisioning": {
                "generated_pyc_omission": pyc_evidence["manifest_evidence"]
            },
            "schema": publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA,
        }
        spec_path = plan.journal_root / "COLLECTION_SPEC.json"
        contract._atomic_write(spec_path, contract.canonical_json_bytes(spec))
        observation = _internal_json(
            candidate_python, script,
            [
                "internal-collect", "--spec", str(spec_path),
                "--environment-root", str(plan.candidate),
            ],
            timeout=3600,
        )
        lock = build_lock_from_observation(
            recipe=plan.recipe,
            observation=observation,
            generated_pyc_omission=pyc_evidence["manifest_evidence"],
            source_environment_fingerprint_sha256=source_evidence[
                "fingerprint_sha256"
            ],
        )
        marker = _completion_marker(plan.candidate, lock)
        with _temporarily_writable_root(plan.candidate):
            completion = write_lock_and_completion_marker(
                candidate=plan.candidate, lock=lock, marker=marker
            )
        activation._verify_tree_immutable(plan.candidate)
        report = _internal_json(
            candidate_python,
            Path(__file__).resolve().with_name("verify_perception_environment.py"),
            [
                "--lock", str(plan.candidate / STRONG_LOCK_NAME),
                "--environment-root", str(plan.candidate),
            ],
            timeout=3600,
        )
        return {
            "completion": completion,
            "immutable": immutable,
            "lock_sha256": completion["lock_sha256"],
            "verification_sha256": hashlib.sha256(
                contract.canonical_json_bytes(report)
            ).hexdigest(),
        }

    final_evidence = run_resumable_phase(
        plan, "finalize", action=finalize_action,
        verify=lambda row: (
            None
            if (
                _require_sha256(row.get("lock_sha256"), "final lock")
                and _require_sha256(row.get("verification_sha256"), "final verification")
                and (plan.candidate / verifier.CLONE_MARKER_NAME).is_file()
            )
            else (_ for _ in ()).throw(PreparationVerificationError("final completion evidence invalid"))
        ),
    )
    with _RootBuildLock(plan.root):
        journal = _load_journal(plan)
        journal["status"] = "COMPLETE"
        _write_journal(plan, journal)
    return completed_inactive_result(
        plan=plan,
        final_evidence=final_evidence,
        source_evidence=source_evidence,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one inactive relocation-aware Linbox perception environment",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    internal_collect = subparsers.add_parser("internal-collect", help=argparse.SUPPRESS)
    internal_collect.add_argument("--spec", required=True)
    internal_collect.add_argument("--environment-root", required=True)
    internal_smoke = subparsers.add_parser("internal-model-smoke", help=argparse.SUPPRESS)
    internal_smoke.add_argument("--code-root", required=True)
    internal_smoke.add_argument("--evidence-root", required=True)
    internal_smoke.add_argument("--v9-checkpoint", required=True)
    internal_smoke.add_argument("--posterior-gate-checkpoint", required=True)

    for name in ("plan", "execute"):
        operation = subparsers.add_parser(name)
        operation.add_argument("--source-environment-id", required=True)
        operation.add_argument("--source-lock", required=True)
        operation.add_argument("--gate-wheelhouse", required=True)
        operation.add_argument("--opencv-wheelhouse", required=True)
        operation.add_argument("--conda-executable")
        operation.add_argument("--code-root")
        operation.add_argument("--v9-checkpoint")
        operation.add_argument("--posterior-gate-checkpoint")
        operation.add_argument("--allow-official-repair-download", action="store_true")
        operation.add_argument("--confirm-environment-id")
        operation.add_argument("--confirm-recipe-sha256")
    return parser


def _emit(stream, value: object) -> None:
    stream.buffer.write(contract.canonical_json_bytes(value))
    stream.flush()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.command == "internal-collect":
        return _internal_collect(args)
    if args.command == "internal-model-smoke":
        return _internal_model_smoke(args)
    try:
        root = LINBOX_PERCEPTION_ROOT
        source_lock_path = normalized_absolute_path(
            args.source_lock, "source lock"
        )
        gate_wheelhouse_path = normalized_absolute_path(
            args.gate_wheelhouse, "gate wheelhouse"
        )
        opencv_wheelhouse_path = normalized_absolute_path(
            args.opencv_wheelhouse, "OpenCV wheelhouse"
        )
        identity, source_lock, source_marker, source_snapshot = _source_identity(
            root=root,
            environment_id=args.source_environment_id,
            lock_path=source_lock_path,
        )
        gate = inventory_wheelhouse(
            gate_wheelhouse_path, purpose="gate-additions"
        )
        opencv = inventory_wheelhouse(
            opencv_wheelhouse_path, purpose="opencv-repair"
        )
        if not gate["wheels"]:
            raise PreparationPolicyError("gate additions wheelhouse is empty")
        expected_gate_profiles = [
            {"name": name, "version": publisher.RELOCATION_AWARE_SUPPORTED_PACKAGES[name]}
            for name in (
                "polars", "polars-runtime-32", "ultralytics", "ultralytics-thop"
            )
        ]
        if wheel_distribution_profiles(
            wheelhouse=gate_wheelhouse_path, manifest=gate
        ) != expected_gate_profiles:
            raise PreparationPolicyError(
                "gate additions wheelhouse does not match the exact package profile"
            )
        if wheel_distribution_profiles(
            wheelhouse=opencv_wheelhouse_path, manifest=opencv
        ) != [{"name": "opencv-python", "version": "4.11.0.86"}]:
            raise PreparationPolicyError(
                "OpenCV repair wheelhouse package profile is invalid"
            )
        if (
            len(opencv["wheels"]) != 1
            or opencv["wheels"][0]["sha256"] != OPENCV_REPAIR_WHEEL_SHA256
        ):
            raise PreparationPolicyError("OpenCV repair wheelhouse is not the pinned repair")
        recipe = build_recipe(
            source_environment=identity,
            gate_wheelhouse_manifest=gate,
            opencv_wheelhouse_manifest=opencv,
            repair_wheelhouse_manifest=exact_repair_wheelhouse_manifest(),
        )
        plan = candidate_plan(root=root, recipe=recipe)
        if args.command == "plan":
            _emit(sys.stdout, {
                "candidate": str(plan.candidate),
                "environment_id": plan.environment_id,
                "journal_path": str(plan.journal_path),
                "recipe": plan.recipe,
                "schema": PREPARATION_RESULT_SCHEMA,
                "status": "PLANNED_READ_ONLY",
            })
            return 0
        if (
            args.confirm_environment_id != plan.environment_id
            or args.confirm_recipe_sha256 != recipe["sha256"]
        ):
            raise PreparationPolicyError(
                "execute confirmations do not match the derived environment and recipe"
            )
        required_paths = {
            "conda executable": args.conda_executable,
            "code root": args.code_root,
            "v9 checkpoint": args.v9_checkpoint,
            "posterior gate checkpoint": args.posterior_gate_checkpoint,
        }
        if any(value is None for value in required_paths.values()):
            raise PreparationPolicyError("execute requires conda, code, and model paths")
        conda_executable = normalized_absolute_path(
            args.conda_executable, "conda executable"
        )
        code_root = normalized_absolute_path(args.code_root, "code root")
        v9_checkpoint = normalized_absolute_path(
            args.v9_checkpoint, "v9 checkpoint"
        )
        posterior_checkpoint = normalized_absolute_path(
            args.posterior_gate_checkpoint, "posterior gate checkpoint"
        )
        _sha256_file(conda_executable)
        _require_real_directory(code_root, "code root")
        if _sha256_file(v9_checkpoint)[0] != V9_CHECKPOINT_SHA256:
            raise PreparationPolicyError("v9 checkpoint is not the pinned model")
        if _sha256_file(posterior_checkpoint)[0] != POSTERIOR_GATE_CHECKPOINT_SHA256:
            raise PreparationPolicyError(
                "posterior gate checkpoint is not the pinned model"
            )
        result = _execute_build(
            plan=plan,
            source_lock_path=source_lock_path,
            source_lock=source_lock,
            source_marker=source_marker,
            source_snapshot=source_snapshot,
            gate_source=gate_wheelhouse_path,
            gate_manifest=gate,
            opencv_source=opencv_wheelhouse_path,
            opencv_manifest=opencv,
            conda_executable=conda_executable,
            code_root=code_root,
            v9_checkpoint=v9_checkpoint,
            posterior_gate_checkpoint=posterior_checkpoint,
            allow_repair_download=args.allow_official_repair_download,
        )
    except PreparationError as exc:
        _emit(sys.stderr, {
            "error": str(exc), "schema": PREPARATION_RESULT_SCHEMA,
            "status": "FAILED",
        })
        return 1
    except (publisher.PerceptionRuntimeError, verifier.EnvironmentVerificationError) as exc:
        _emit(sys.stderr, {
            "error": str(exc), "schema": PREPARATION_RESULT_SCHEMA,
            "status": "FAILED",
        })
        return 1
    except Exception as exc:  # pragma: no cover - canonical fail-closed boundary
        _emit(sys.stderr, {
            "error": "unexpected preparation failure: %s" % type(exc).__name__,
            "schema": PREPARATION_RESULT_SCHEMA, "status": "FAILED",
        })
        return 1
    _emit(sys.stdout, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
