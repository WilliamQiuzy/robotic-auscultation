"""Fail-closed provenance gate for Lingbox hand-eye commissioning.

This module deliberately emits identifiers, versions, and SHA-256 digests only.
It never serializes process arguments, environment variables, host paths, or
credentials into calibration evidence.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import types
from typing import Dict, Mapping, Tuple


SCHEMA = "wam_handeye_provenance_v1"
CODE_ARTIFACT_KIND = "handeye_calibration_code"
CODE_SOURCE_ROLE = "mac"
CODE_TARGET_ROLE = "linbox"
ENVIRONMENT_LOCK_NAME = "PERCEPTION_ENVIRONMENT_STRONG_LOCK.json"
ENVIRONMENT_REPORT_NAME = "PERCEPTION_ENVIRONMENT_CLONE_COMPLETE.json"
ENVIRONMENT_LOCK_SCHEMA = "wam_perception_environment_lock_v3"
ENVIRONMENT_REPORT_SCHEMA = "wam_perception_environment_clone_marker_v2"
SOURCE_LINE_HASH_ALGORITHM = "sha256-concat-sha256-source-lines-with-endings-v1"
ARTIFACT_CONTRACT_PATH = "auscultation/scripts/deployment/artifact_contract.py"
SOURCE_PATHS = {
    "camera_stream": "auscultation/ausc/perception/camera_stream.py",
    "handeye": "auscultation/scripts/perception/handeye_calibrate.py",
    "orbbec_preflight": "auscultation/scripts/perception/handeye_orbbec_preflight.py",
    "provenance": "auscultation/scripts/perception/handeye_provenance.py",
    "runtime_paths": "auscultation/scripts/perception/runtime_paths.py",
}
PACKAGE_DISTRIBUTIONS = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "pyorbbecsdk2": "pyorbbecsdk2",
    "scipy": "scipy",
    "yourdfpy": "yourdfpy",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENT_ID = re.compile(r"^linbox-perception-[0-9a-f]{20}$")


class HandeyeProvenanceError(RuntimeError):
    """The running calibration process is not exactly reproducible."""


def _identity(info: os.stat_result) -> Tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
        getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000)),
    )


def _direct_path(path: Path, *, label: str, directory: bool) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise HandeyeProvenanceError("%s must be a direct absolute path" % label)
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise HandeyeProvenanceError("%s is missing" % label) from exc
        except OSError as exc:
            raise HandeyeProvenanceError("%s cannot be inspected" % label) from exc
        if stat.S_ISLNK(info.st_mode):
            raise HandeyeProvenanceError("%s path contains a symlink" % label)
        final = index == len(parts) - 1
        if not final and not stat.S_ISDIR(info.st_mode):
            raise HandeyeProvenanceError("%s path is not direct" % label)
        if final:
            expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
            if not expected:
                kind = "directory" if directory else "regular file"
                raise HandeyeProvenanceError("%s is not a direct %s" % (label, kind))
    return path


def _read_direct_regular(path: Path, *, label: str) -> bytes:
    candidate = _direct_path(path, label=label, directory=False)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(candidate), flags)
    except OSError as exc:
        raise HandeyeProvenanceError("%s cannot be opened directly" % label) from exc
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after):
        raise HandeyeProvenanceError("%s changed while being read" % label)
    return b"".join(chunks)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise HandeyeProvenanceError("%s must be an exact lowercase SHA-256" % label)
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _load_canonical_json(data: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandeyeProvenanceError("%s is not valid JSON" % label) from exc
    if not isinstance(value, dict) or data != _canonical_json_bytes(value):
        raise HandeyeProvenanceError("%s is not canonical JSON" % label)
    return value


def _release_root(provenance: Path) -> Path:
    relative = PurePosixPath(SOURCE_PATHS["provenance"])
    root = provenance
    for _part in relative.parts:
        root = root.parent
    if (
        root.parent.name != "releases"
        or provenance.relative_to(root).as_posix() != relative.as_posix()
    ):
        raise HandeyeProvenanceError(
            "hand-eye preflight is not running from a direct completed code release"
        )
    return _direct_path(root, label="hand-eye code release", directory=True)


def _load_release_contract(release: Path):
    path = release / ARTIFACT_CONTRACT_PATH
    source = _read_direct_regular(path, label="release artifact verifier")
    name = "_wam_handeye_artifact_contract_%s" % _sha256(source)[:20]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except Exception as exc:
        raise HandeyeProvenanceError("release artifact verifier cannot execute") from exc
    verifier = getattr(module, "verify_release", None)
    if not callable(verifier):
        raise HandeyeProvenanceError("release artifact verifier is incomplete")
    return module


def _verify_code_release(release: Path):
    contract = _load_release_contract(release)
    try:
        envelope = contract.verify_release(release, require_complete=True)
    except Exception as exc:
        raise HandeyeProvenanceError("completed code release verification failed: %s" % exc) from exc
    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    if not isinstance(payload, dict):
        raise HandeyeProvenanceError("completed code release payload is invalid")
    if payload.get("artifact_kind") != CODE_ARTIFACT_KIND:
        raise HandeyeProvenanceError("code release artifact kind is invalid")
    if payload.get("source_role") != CODE_SOURCE_ROLE:
        raise HandeyeProvenanceError("code release source role must be mac")
    if envelope.get("release_id") != release.name:
        raise HandeyeProvenanceError("code release id does not match its direct directory")
    complete_bytes = _read_direct_regular(
        release / contract.COMPLETE_NAME, label="code transfer completion marker",
    )
    complete = _load_canonical_json(complete_bytes, label="code transfer completion marker")
    if complete.get("target_role") != CODE_TARGET_ROLE:
        raise HandeyeProvenanceError("code release target must be linbox")
    manifest_bytes = _read_direct_regular(
        release / contract.MANIFEST_NAME, label="code release manifest",
    )
    return contract, envelope, payload, manifest_bytes, complete_bytes


def _line_digest(data: bytes) -> Tuple[int, str]:
    lines = data.splitlines(keepends=True)
    aggregate = hashlib.sha256()
    for line in lines:
        aggregate.update(hashlib.sha256(line).digest())
    return len(lines), aggregate.hexdigest()


def _source_evidence(release: Path, payload: Mapping[str, object]) -> Dict[str, object]:
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise HandeyeProvenanceError("code release file manifest is invalid")
    by_path = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise HandeyeProvenanceError("code release file row is invalid")
        if row["path"] in by_path:
            raise HandeyeProvenanceError("code release file path is duplicated")
        by_path[row["path"]] = row
    evidence: Dict[str, object] = {}
    for label, relative in sorted(SOURCE_PATHS.items()):
        row = by_path.get(relative)
        if not isinstance(row, dict):
            raise HandeyeProvenanceError("required %s source is absent from release" % label)
        data = _read_direct_regular(release / relative, label="%s source" % label)
        digest = _sha256(data)
        if digest != row.get("sha256"):
            raise HandeyeProvenanceError("required %s source hash mismatch" % label)
        line_count, line_digest = _line_digest(data)
        evidence[label] = {
            "line_count": line_count,
            "line_sha256": line_digest,
            "path": relative,
            "sha256": digest,
        }
    if ARTIFACT_CONTRACT_PATH not in by_path:
        raise HandeyeProvenanceError("release artifact verifier is absent from manifest")
    return evidence


def _environment_root(executable: Path) -> Path:
    executable = _direct_path(executable, label="Python executable", directory=False)
    if not os.access(str(executable), os.X_OK):
        raise HandeyeProvenanceError("Python executable is not executable")
    if (
        executable.parent.name != "bin"
        or re.fullmatch(r"python(?:3(?:\.[0-9]+)?)?", executable.name) is None
    ):
        raise HandeyeProvenanceError(
            "Python is outside a direct perception environment release"
        )
    root = executable.parent.parent
    if root.parent.name != "releases" or not _ENVIRONMENT_ID.fullmatch(root.name):
        raise HandeyeProvenanceError(
            "Python is outside a direct perception environment release"
        )
    return _direct_path(root, label="perception environment release", directory=True)


def _package_versions(lock: Mapping[str, object]) -> Dict[str, str]:
    rows = lock.get("packages")
    if not isinstance(rows, list):
        raise HandeyeProvenanceError("perception environment package lock is invalid")
    locked: Dict[str, str] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or not isinstance(row.get("version"), str)
            or not row["version"]
            or row["name"] in locked
        ):
            raise HandeyeProvenanceError("perception environment package row is invalid")
        locked[row["name"]] = row["version"]
    observed: Dict[str, str] = {}
    for audit_name, distribution in sorted(PACKAGE_DISTRIBUTIONS.items()):
        expected = locked.get(distribution)
        if expected is None:
            raise HandeyeProvenanceError("required package is absent from environment lock")
        try:
            actual = metadata.version(distribution)
        except Exception as exc:
            raise HandeyeProvenanceError("required package version cannot be observed") from exc
        if not isinstance(actual, str) or actual != expected:
            raise HandeyeProvenanceError("required package version does not match lock")
        observed[audit_name] = actual
    return observed


def collect_handeye_provenance(
    *, environment_lock_sha256: str, environment_report_sha256: str,
) -> Dict[str, object]:
    """Verify and return non-secret provenance for one calibration process.

    The expected hashes must come from the separately selected, completed
    Lingbox perception environment deployment. Paths are intentionally derived
    from ``__file__`` and ``sys.executable`` and cannot be redirected by caller.
    """
    expected_lock = _required_sha256(
        environment_lock_sha256, "perception environment lock SHA-256",
    )
    expected_report = _required_sha256(
        environment_report_sha256, "perception verification report SHA-256",
    )

    preflight = _direct_path(Path(__file__), label="hand-eye preflight", directory=False)
    release = _release_root(preflight)
    contract, envelope, payload, manifest_bytes, complete_bytes = _verify_code_release(release)
    sources = _source_evidence(release, payload)

    executable = _direct_path(Path(sys.executable), label="Python executable", directory=False)
    environment = _environment_root(executable)
    lock_path = environment / ENVIRONMENT_LOCK_NAME
    report_path = environment / ENVIRONMENT_REPORT_NAME
    lock_bytes = _read_direct_regular(lock_path, label="perception environment lock")
    report_bytes = _read_direct_regular(report_path, label="perception clone marker")
    if _sha256(lock_bytes) != expected_lock:
        raise HandeyeProvenanceError("perception environment lock hash mismatch")
    if _sha256(report_bytes) != expected_report:
        raise HandeyeProvenanceError("perception clone marker hash mismatch")
    lock = _load_canonical_json(lock_bytes, label="perception environment lock")
    report = _load_canonical_json(report_bytes, label="perception clone marker")
    if lock.get("schema") != ENVIRONMENT_LOCK_SCHEMA:
        raise HandeyeProvenanceError("perception environment lock schema is not v3")
    if lock.get("environment_id") != environment.name:
        raise HandeyeProvenanceError("perception environment lock id mismatch")

    executable_bytes = _read_direct_regular(executable, label="Python executable")
    executable_digest = _sha256(executable_bytes)
    python_lock = lock.get("python")
    if not isinstance(python_lock, dict):
        raise HandeyeProvenanceError("perception environment Python lock is invalid")
    python_version = platform.python_version()
    python_implementation = platform.python_implementation()
    if (
        python_lock.get("executable_sha256") != executable_digest
        or python_lock.get("version") != python_version
        or python_lock.get("implementation") != python_implementation
    ):
        raise HandeyeProvenanceError("running Python does not match environment lock")
    package_versions = _package_versions(lock)

    if (
        report.get("schema") != ENVIRONMENT_REPORT_SCHEMA
        or report.get("complete") is not True
        or report.get("environment_id") != environment.name
        or report.get("environment_root") != str(environment)
        or report.get("lock_sha256") != expected_lock
        or not isinstance(report.get("observed_environment_tree"), dict)
        or not isinstance(report.get("observed_record_observation"), dict)
        or report.get("required_service_environment", {}).get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise HandeyeProvenanceError(
            "perception clone marker does not bind the running environment"
        )

    # Rehash all mutable observations before returning so an in-call change
    # cannot be silently represented as the evidence read earlier.
    try:
        contract.verify_release(release, require_complete=True)
    except Exception as exc:
        raise HandeyeProvenanceError("code release changed during provenance check") from exc
    if (
        _sha256(_read_direct_regular(lock_path, label="perception environment lock"))
        != expected_lock
        or _sha256(_read_direct_regular(report_path, label="perception clone marker"))
        != expected_report
        or _sha256(_read_direct_regular(executable, label="Python executable"))
        != executable_digest
    ):
        raise HandeyeProvenanceError("environment evidence changed during provenance check")

    return {
        "code_release": {
            "artifact_kind": payload["artifact_kind"],
            "manifest_sha256": _sha256(manifest_bytes),
            "payload_sha256": envelope["payload_sha256"],
            "release_id": envelope["release_id"],
            "source_line_hash_algorithm": SOURCE_LINE_HASH_ALGORITHM,
            "source_role": payload["source_role"],
            "sources": sources,
            "target_role": CODE_TARGET_ROLE,
            "transfer_complete_sha256": _sha256(complete_bytes),
        },
        "environment": {
            "environment_id": environment.name,
            "lock_sha256": expected_lock,
            "package_versions": package_versions,
            "python": {
                "executable_sha256": executable_digest,
                "implementation": python_implementation,
                "version": python_version,
            },
            "clone_marker_sha256": expected_report,
        },
        "schema": SCHEMA,
        "status": "VERIFIED",
    }
