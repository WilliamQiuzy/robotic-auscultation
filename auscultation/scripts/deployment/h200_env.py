#!/usr/bin/env python3
"""Fail-closed H200 environment release manager.

The manager intentionally uses only the Python standard library and is importable
on Python 3.9.  It provisions a CPython 3.12, Linux x86_64, PyTorch cu128
environment from a hash-locked requirements file and an already materialized
wheelhouse.  Network resolution is deliberately outside the build operation.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union
import uuid


PYPI_INDEX = "https://pypi.org/simple"
PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"
APPROVED_INDEXES = (PYPI_INDEX, PYTORCH_CU128_INDEX)
H200_BASE_PYTHON = (
    "/home/ssh-ziyue/.local/share/uv/python/"
    "cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
)

IDENTITY_PAYLOAD_SCHEMA = "wam_h200_environment_identity_v1"
IDENTITY_ENVELOPE_SCHEMA = "wam_h200_environment_identity_envelope_v1"
SYSTEM_LIBRARIES_SCHEMA = "wam_h200_system_libraries_v1"
SMOKE_SCHEMA = "wam_h200_smoke_v1"
COMPLETE_PAYLOAD_SCHEMA = "wam_h200_environment_complete_v1"
COMPLETE_ENVELOPE_SCHEMA = "wam_h200_environment_complete_envelope_v1"
INSTALLED_TREE_SCHEMA = "wam_h200_installed_tree_v1"
LOCK_BLOCKER_SCHEMA = "wam_h200_lock_blocker_v1"
PRODUCTION_ACTIVATION_REQUEST_SCHEMA = "wam_h200_production_activation_request_v1"
PRODUCTION_ACTIVATION_EVIDENCE_SCHEMA = "wam_h200_production_activation_evidence_v1"
CODE_MANIFEST_NAME = "RELEASE_MANIFEST.json"
CODE_COMPLETE_NAME = "TRANSFER_COMPLETE.json"
CODE_PAYLOAD_SCHEMA = "wam_artifact_release_v1"
CODE_ENVELOPE_SCHEMA = "wam_artifact_manifest_envelope_v1"
CODE_COMPLETE_SCHEMA = "wam_transfer_complete_v1"
LOCK_COMPILE_COMMAND = (
    "uv pip compile auscultation/requirements-h200.in "
    "--output-file auscultation/requirements-h200.lock.txt --generate-hashes "
    "--python-version 3.12 --python-platform x86_64-manylinux_2_31 "
    "--torch-backend cu128 --only-binary :all: --no-sources --emit-index-url "
    "--no-annotate --index-strategy unsafe-best-match"
)

IDENTITY_NAME = "ENVIRONMENT_IDENTITY.json"
SMOKE_NAME = "H200_SMOKE.json"
COMPLETE_NAME = "ENVIRONMENT_COMPLETE.json"
BUILD_OWNERSHIP_NAME = ".H200_BUILD_OWNERSHIP.json"
BUILD_OWNERSHIP_SCHEMA = "wam_h200_environment_build_ownership_v1"

_HASH = re.compile(r"^[0-9a-f]{64}$")
_CODE_RELEASE_ID = re.compile(r"^code_release-([0-9a-f]{20})$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PINNED = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)$")
_PYTHON_VERSION = re.compile(r"^3\.12\.13$")
_FORBIDDEN_REQUIREMENTS = frozenset(("openpi", "openpi-client"))
_CORE_VERSIONS = {
    "lerobot": "0.3.3",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "torchvision": "0.22.1",
}
_PYTORCH_VERSIONS = {
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "torchvision": "0.22.1",
}
_RUNTIME_ENV_ALLOWLIST = {
    "CUDA_VISIBLE_DEVICES",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "TMPDIR",
    "USER",
}
H200_PRODUCTION_HOME = Path("/home/ssh-ziyue")


class H200EnvironmentError(RuntimeError):
    """Base class for H200 environment contract failures."""


class LockValidationError(H200EnvironmentError):
    """The requirements lock is not an unambiguous hash lock."""


class IdentityValidationError(H200EnvironmentError):
    """A bound environment input does not match the H200 target."""


class EnvironmentConflictError(H200EnvironmentError):
    """A managed path or lock conflicts with pre-existing state."""


class EnvironmentVerificationError(H200EnvironmentError):
    """A completed environment does not verify exactly."""


class SmokeValidationError(H200EnvironmentError):
    """The H200 smoke report is missing or failed."""


class PointerPolicyError(H200EnvironmentError):
    """A pointer operation would escape the isolated canary policy."""


class PointerConflictError(H200EnvironmentError):
    """A canary compare-and-swap precondition failed."""


@dataclass(frozen=True)
class LockedRequirement:
    name: str
    version: str
    hashes: Tuple[str, ...]


@dataclass(frozen=True)
class HashLockedRequirements:
    indexes: Tuple[str, ...]
    lock_sha256: str
    lock_size: int
    requirements: Tuple[LockedRequirement, ...]


@dataclass(frozen=True)
class EnvironmentInputSnapshot:
    name: str
    root_path: Path
    lock_path: Path
    wheelhouse_path: Path
    home_path: Path
    parent_fd: int
    root_fd: int
    lock_fd: int
    wheelhouse_fd: int


def canonical_json_bytes(value: object) -> bytes:
    """Return the sole canonical JSON encoding accepted by this contract."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _runtime_process_env(home: Path) -> Dict[str, str]:
    result = {
        key: value
        for key, value in os.environ.items()
        if key in _RUNTIME_ENV_ALLOWLIST and value
    }
    result["HOME"] = str(home)
    result["PIP_NO_INDEX"] = "1"
    return result


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _read_regular_bytes(path: Union[str, Path], label: str) -> bytes:
    candidate = Path(path)
    try:
        info = candidate.lstat()
    except FileNotFoundError as exc:
        raise LockValidationError("%s is missing" % label) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise LockValidationError("%s must be a regular, non-symlink file" % label)
    before = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    data = candidate.read_bytes()
    after_info = candidate.lstat()
    after = (
        after_info.st_dev,
        after_info.st_ino,
        after_info.st_size,
        after_info.st_mtime_ns,
    )
    if before != after:
        raise LockValidationError("%s changed while being read" % label)
    return data


def _logical_lock_lines(text: str) -> List[str]:
    logical: List[str] = []
    pending = ""
    for physical in text.splitlines():
        content = physical.split("#", 1)[0].strip()
        if not content:
            continue
        continued = content.endswith("\\")
        if continued:
            content = content[:-1].rstrip()
        pending = (pending + " " + content).strip() if pending else content
        if not continued:
            logical.append(pending)
            pending = ""
    if pending:
        raise LockValidationError("requirements lock ends with an incomplete continuation")
    return logical


def _parse_index(line: str) -> Tuple[str, str]:
    try:
        tokens = shlex.split(line, posix=True)
    except ValueError as exc:
        raise LockValidationError("requirements index directive is malformed") from exc
    if len(tokens) != 2 or tokens[0] not in ("--index-url", "--extra-index-url"):
        raise LockValidationError("unsupported requirements directive")
    url = tokens[1]
    if url not in APPROVED_INDEXES:
        raise LockValidationError("requirements index is not approved: %s" % url)
    expected_option = "--index-url" if url == PYPI_INDEX else "--extra-index-url"
    if tokens[0] != expected_option:
        raise LockValidationError("requirements index uses the wrong directive")
    return tokens[0], url


def _parse_requirement(line: str) -> LockedRequirement:
    lowered = line.lower()
    if lowered.startswith("-e ") or lowered.startswith("--editable "):
        raise LockValidationError("editable requirements are forbidden")
    if ";" in line:
        raise LockValidationError("environment marker ambiguity is forbidden")
    if " @ " in line or "://" in line or any(
        token in lowered for token in ("git+", "hg+", "svn+", "bzr+")
    ):
        raise LockValidationError("URL and VCS requirements are forbidden")
    if line.startswith("-"):
        raise LockValidationError("unsupported requirements directive")
    try:
        tokens = shlex.split(line, posix=True)
    except ValueError as exc:
        raise LockValidationError("requirement line is malformed") from exc
    if not tokens:
        raise LockValidationError("empty requirement line")
    pinned = _PINNED.fullmatch(tokens[0])
    if pinned is None:
        raise LockValidationError("every requirement must be exactly pinned with ==")
    name = _normalize_name(pinned.group(1))
    version = pinned.group(2)
    if name in _FORBIDDEN_REQUIREMENTS or name.startswith("openpi-"):
        raise LockValidationError("openpi is outside the WAM-core environment")
    hashes: List[str] = []
    for token in tokens[1:]:
        prefix = "--hash=sha256:"
        if not token.startswith(prefix):
            raise LockValidationError("only --hash=sha256 digests may follow a pin")
        value = token[len(prefix) :]
        if not _HASH.fullmatch(value):
            raise LockValidationError("sha256 digest must be 64 lowercase hex characters")
        if value in hashes:
            raise LockValidationError("duplicate sha256 digest on requirement %s" % name)
        hashes.append(value)
    if not hashes:
        raise LockValidationError("every installable requirement needs a sha256 hash")
    return LockedRequirement(name=name, version=version, hashes=tuple(sorted(hashes)))


def parse_hash_lock(lock_path: Union[str, Path]) -> HashLockedRequirements:
    """Strictly parse a pip/uv hash lock and reject ambiguous requirement forms."""
    raw = _read_regular_bytes(lock_path, "requirements lock")
    if b"\x00" in raw:
        raise LockValidationError("requirements lock contains NUL bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LockValidationError("requirements lock must be UTF-8") from exc

    indexes: List[str] = []
    requirements: List[LockedRequirement] = []
    seen = set()
    for line in _logical_lock_lines(text):
        if line.startswith("--"):
            _option, url = _parse_index(line)
            if url in indexes:
                raise LockValidationError("duplicate requirements index")
            indexes.append(url)
            continue
        row = _parse_requirement(line)
        if row.name in seen:
            raise LockValidationError("duplicate requirement after name normalization: %s" % row.name)
        seen.add(row.name)
        requirements.append(row)
    if not requirements:
        raise LockValidationError("requirements lock contains no installable requirements")
    return HashLockedRequirements(
        indexes=tuple(indexes),
        lock_sha256=_sha256_bytes(raw),
        lock_size=len(raw),
        requirements=tuple(requirements),
    )


def read_blocked_lock_metadata(
    lock_path: Union[str, Path], requirements_in: Union[str, Path]
) -> Dict[str, str]:
    """Validate a non-installable resolver-blocker record bound to its input."""
    raw = _read_regular_bytes(lock_path, "requirements lock blocker metadata")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LockValidationError("lock blocker metadata must be UTF-8") from exc
    fields: Dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        if not line.startswith("# ") or ": " not in line[2:]:
            raise LockValidationError("lock blocker metadata contains installable or malformed text")
        key, value = line[2:].split(": ", 1)
        if not key or not value or key in fields:
            raise LockValidationError("lock blocker metadata field is invalid or duplicated")
        fields[key] = value
    expected_fields = {
        "blocker",
        "command",
        "remediation",
        "requirements_in_sha256",
        "resolver",
        "schema",
        "status",
        "target",
    }
    if set(fields) != expected_fields:
        raise LockValidationError("lock blocker metadata fields are not exact")
    if fields["schema"] != LOCK_BLOCKER_SCHEMA or fields["status"] != "blocked":
        raise LockValidationError("lock blocker metadata schema or status is invalid")
    if fields["target"] != "cpython-3.12-linux-x86_64-manylinux_2_31-cu128":
        raise LockValidationError("lock blocker target is invalid")
    if fields["resolver"] != "uv 0.10.12":
        raise LockValidationError("lock blocker resolver is invalid")
    if fields["command"] != LOCK_COMPILE_COMMAND:
        raise LockValidationError("lock blocker rerun command is invalid")
    input_raw = _read_regular_bytes(requirements_in, "requirements input")
    if fields["requirements_in_sha256"] != _sha256_bytes(input_raw):
        raise LockValidationError("lock blocker does not bind requirements input")
    if not _HASH.fullmatch(fields["requirements_in_sha256"]):
        raise LockValidationError("lock blocker input hash is invalid")
    if "evdev" not in fields["blocker"].lower() or "wheel" not in fields["blocker"].lower():
        raise LockValidationError("lock blocker reason is not the observed wheel failure")
    if not fields["remediation"]:
        raise LockValidationError("lock blocker remediation is missing")
    return fields


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def _validate_core_lock(lock: HashLockedRequirements) -> Dict[str, LockedRequirement]:
    by_name = {row.name: row for row in lock.requirements}
    for name, expected in _CORE_VERSIONS.items():
        row = by_name.get(name)
        if row is None:
            raise IdentityValidationError("WAM-core lock is missing %s" % name)
        if _base_version(row.version) != expected:
            raise IdentityValidationError(
                "WAM-core %s must be %s (found %s)" % (name, expected, row.version)
            )
    if lock.indexes != APPROVED_INDEXES:
        raise IdentityValidationError(
            "lock indexes must be exactly %r" % (APPROVED_INDEXES,)
        )
    return by_name


def _validate_python_report(report: Mapping[str, object]) -> Dict[str, str]:
    expected_keys = {"abi", "implementation", "machine", "platform", "version"}
    if set(report) != expected_keys:
        raise IdentityValidationError("python report must have exact fields %r" % sorted(expected_keys))
    normalized = {key: report[key] for key in sorted(expected_keys)}
    for key, value in normalized.items():
        if not isinstance(value, str) or not value:
            raise IdentityValidationError("python %s must be a non-empty string" % key)
    if normalized["implementation"] != "CPython":
        raise IdentityValidationError("python implementation must be CPython")
    if not _PYTHON_VERSION.fullmatch(normalized["version"]):
        raise IdentityValidationError("python version must be CPython 3.12.13")
    if normalized["abi"] != "cp312":
        raise IdentityValidationError("python abi must be cp312")
    if normalized["platform"] != "linux":
        raise IdentityValidationError("python platform must be linux")
    if normalized["machine"] != "x86_64":
        raise IdentityValidationError("python machine must be x86_64")
    return normalized


def _validate_system_report(report: Mapping[str, object]) -> Dict[str, object]:
    expected_keys = {
        "cuda_device_name",
        "glibc_version",
        "libraries",
        "nvidia_driver_version",
        "schema",
    }
    if set(report) != expected_keys:
        raise IdentityValidationError(
            "system library report must have exact fields %r" % sorted(expected_keys)
        )
    if report.get("schema") != SYSTEM_LIBRARIES_SCHEMA:
        raise IdentityValidationError("system library report schema is invalid")
    device_name = report.get("cuda_device_name")
    if not isinstance(device_name, str) or "H200" not in device_name.upper():
        raise IdentityValidationError("system CUDA device is not an H200")
    driver = report.get("nvidia_driver_version")
    if not isinstance(driver, str) or not driver:
        raise IdentityValidationError("nvidia driver version is missing")
    glibc_version = report.get("glibc_version")
    if not isinstance(glibc_version, str) or not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", glibc_version):
        raise IdentityValidationError("glibc version is invalid")
    if tuple(int(part) for part in glibc_version.split(".")[:2]) < (2, 31):
        raise IdentityValidationError("glibc version is below the manylinux_2_31 target")
    source_libraries = report.get("libraries")
    if not isinstance(source_libraries, list) or not source_libraries:
        raise IdentityValidationError("system library report needs at least one library")
    libraries: List[Dict[str, object]] = []
    seen = set()
    for row in source_libraries:
        if not isinstance(row, dict) or set(row) != {
            "device",
            "inode",
            "name",
            "path",
            "sha256",
        }:
            raise IdentityValidationError("system library row has invalid fields")
        device = row.get("device")
        inode = row.get("inode")
        name = row.get("name")
        path = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(device, int)
            or isinstance(device, bool)
            or device < 0
            or not isinstance(inode, int)
            or isinstance(inode, bool)
            or inode <= 0
        ):
            raise IdentityValidationError("system library inode identity is invalid")
        if not isinstance(name, str) or not name:
            raise IdentityValidationError("system library name is invalid")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise IdentityValidationError("system library path must be absolute")
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise IdentityValidationError("system library sha256 is invalid")
        key = (name, path)
        if key in seen:
            raise IdentityValidationError("duplicate system library row")
        seen.add(key)
        libraries.append(
            {
                "device": device,
                "inode": inode,
                "name": name,
                "path": path,
                "sha256": digest,
            }
        )
    libraries.sort(key=lambda item: (str(item["name"]), str(item["path"])))
    return {
        "cuda_device_name": device_name,
        "glibc_version": glibc_version,
        "libraries": libraries,
        "nvidia_driver_version": driver,
        "schema": SYSTEM_LIBRARIES_SCHEMA,
    }


def resolve_base_python(python_executable: str) -> str:
    candidate = shutil.which(python_executable)
    if candidate is None and "/" in python_executable:
        candidate = python_executable
    if candidate is None:
        raise IdentityValidationError("base interpreter cannot be resolved")
    try:
        resolved = Path(candidate).expanduser().resolve(strict=True)
        info = resolved.lstat()
    except OSError as exc:
        raise IdentityValidationError("base interpreter cannot be resolved safely") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(str(resolved), os.X_OK):
        raise IdentityValidationError("base interpreter is not an executable regular file")
    return resolved.as_posix()


def open_base_python(path: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise IdentityValidationError("base interpreter cannot be opened safely") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o111:
        os.close(descriptor)
        raise IdentityValidationError("base interpreter descriptor is not executable")
    return descriptor


def _verify_base_python_path(path: Union[str, Path], descriptor: int) -> None:
    try:
        observed = Path(path).lstat()
        held = os.fstat(descriptor)
    except OSError as exc:
        raise IdentityValidationError("base interpreter path changed") from exc
    if (
        not stat.S_ISREG(observed.st_mode)
        or not observed.st_mode & 0o111
        or (observed.st_dev, observed.st_ino) != (held.st_dev, held.st_ino)
    ):
        raise IdentityValidationError("base interpreter path changed")


def _descriptor_path(descriptor: int) -> str:
    if sys.platform.startswith("linux"):
        return "/proc/self/fd/%s" % descriptor
    if sys.platform == "darwin":
        return "/dev/fd/%s" % descriptor
    raise IdentityValidationError("descriptor-bound execution is unsupported on this platform")


class _BoundExecutableCommand(list):
    def __init__(self, argv: List[str], *, executable: str) -> None:
        super().__init__(argv)
        self.executable = executable


def _verify_bound_base_python(
    launcher: Union[str, Path], base_python_fd: int,
) -> None:
    path = Path(launcher)
    try:
        observed = path.lstat()
        held = os.fstat(base_python_fd)
    except OSError as exc:
        raise IdentityValidationError("bound base interpreter cannot be verified") from exc
    if (
        path.name != "python3.12"
        or not stat.S_ISREG(observed.st_mode)
        or not observed.st_mode & 0o111
        or (observed.st_dev, observed.st_ino) != (held.st_dev, held.st_ino)
    ):
        raise IdentityValidationError("bound base interpreter identity changed")


def _bound_base_python_command(
    base_python_fd: int,
    launcher_home: Path,
    arguments: List[str],
) -> _BoundExecutableCommand:
    launcher = launcher_home / "python3.12"
    try:
        home_info = launcher_home.lstat()
        if not stat.S_ISDIR(home_info.st_mode):
            raise IdentityValidationError("bound base interpreter home is unsafe")
        if sys.platform.startswith("linux"):
            source = _descriptor_path(base_python_fd)
            libc = ctypes.CDLL(None, use_errno=True)
            linkat = getattr(libc, "linkat", None)
            if linkat is None:
                raise IdentityValidationError(
                    "descriptor-bound base interpreter hardlink is unavailable"
                )
            linkat.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            linkat.restype = ctypes.c_int
            result = linkat(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(launcher),
                0x400,
            )
            if result != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), str(launcher))
        elif sys.platform == "darwin":
            raw_path = fcntl.fcntl(base_python_fd, 50, b"\0" * 1024)
            source = raw_path.split(b"\0", 1)[0].decode()
            os.link(source, str(launcher), follow_symlinks=True)
        else:
            raise IdentityValidationError(
                "descriptor-bound execution is unsupported on this platform"
            )
        _verify_bound_base_python(launcher, base_python_fd)
        _fsync_directory(launcher_home)
    except (H200EnvironmentError, OSError):
        try:
            launcher.unlink()
        except FileNotFoundError:
            pass
        raise
    executable = (
        _descriptor_path(base_python_fd)
        if sys.platform.startswith("linux")
        else str(launcher)
    )
    return _BoundExecutableCommand(
        [str(launcher)] + list(arguments), executable=executable,
    )


def probe_runtime(
    python_executable: str, *, pass_fds: Tuple[int, ...] = ()
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """Probe the exact base interpreter and H200 system used for the build."""
    with tempfile.TemporaryDirectory(prefix="wam-h200-probe-home-") as raw_home:
        return _probe_runtime_with_env(
            python_executable,
            pass_fds=pass_fds,
            process_env=_runtime_process_env(Path(raw_home)),
        )


def _probe_runtime_with_env(
    python_executable: str,
    *,
    pass_fds: Tuple[int, ...],
    process_env: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, object]]:
    script = (
        "import json,platform,re,sys;"
        "tag=getattr(sys.implementation,'cache_tag','') or '';"
        "match=re.fullmatch(r'cpython-(\\d{3})',tag);"
        "print(json.dumps({'python':{'abi':'cp'+match.group(1) if match else tag,"
        "'implementation':platform.python_implementation(),"
        "'machine':platform.machine().lower(),'platform':sys.platform,"
        "'version':platform.python_version()},'glibc_version':platform.libc_ver()[1]},"
        "sort_keys=True,separators=(',',':')))"
    )
    try:
        completed = subprocess.run(
            [python_executable, "-I", "-S", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            env=process_env,
            pass_fds=pass_fds,
        )
    except OSError as exc:
        raise IdentityValidationError("base interpreter probe could not start") from exc
    if completed.returncode != 0:
        raise IdentityValidationError(
            "base interpreter probe failed: %s" % completed.stderr.strip()
        )
    try:
        runtime = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise IdentityValidationError("base interpreter probe emitted invalid JSON") from exc
    if not isinstance(runtime, dict) or set(runtime) != {"glibc_version", "python"}:
        raise IdentityValidationError("base interpreter probe fields are invalid")
    python_value = runtime.get("python")
    if not isinstance(python_value, dict):
        raise IdentityValidationError("base interpreter probe is invalid")
    python_report = _validate_python_report(python_value)

    try:
        nvidia = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            env=process_env,
            pass_fds=pass_fds,
        )
    except OSError as exc:
        raise IdentityValidationError("NVIDIA system probe could not start") from exc
    if nvidia.returncode != 0:
        raise IdentityValidationError(
            "NVIDIA system probe failed: %s" % nvidia.stderr.strip()
        )
    gpu_rows = []
    for line in nvidia.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2 or not all(parts):
            raise IdentityValidationError("NVIDIA system probe output is invalid")
        gpu_rows.append((parts[0], parts[1]))
    if not gpu_rows or len(set(gpu_rows)) != 1 or "H200" not in gpu_rows[0][0].upper():
        raise IdentityValidationError("system probe did not find one consistent H200 target")

    try:
        ldconfig = subprocess.run(
            ["ldconfig", "-p"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            env=process_env,
            pass_fds=pass_fds,
        )
    except OSError as exc:
        raise IdentityValidationError("system library probe could not start") from exc
    if ldconfig.returncode != 0:
        raise IdentityValidationError(
            "system library probe failed: %s" % ldconfig.stderr.strip()
        )
    candidates = []
    for line in ldconfig.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("libcuda.so.1 ") or " => " not in stripped:
            continue
        candidate = Path(stripped.split(" => ", 1)[1]).resolve(strict=True)
        if candidate.is_file() and not candidate.is_symlink():
            candidates.append(candidate)
    unique_candidates = sorted(set(candidates), key=lambda path: path.as_posix())
    if len(unique_candidates) != 1:
        raise IdentityValidationError("system probe did not resolve exactly one libcuda.so.1")
    library = unique_candidates[0]
    library_info = library.stat()
    system_report = _validate_system_report(
        {
            "cuda_device_name": gpu_rows[0][0],
            "glibc_version": runtime.get("glibc_version"),
            "libraries": [
                {
                    "device": library_info.st_dev,
                    "inode": library_info.st_ino,
                    "name": "libcuda.so.1",
                    "path": library.as_posix(),
                    "sha256": _sha256_file(library),
                }
            ],
            "nvidia_driver_version": gpu_rows[0][1],
            "schema": SYSTEM_LIBRARIES_SCHEMA,
        }
    )
    return python_report, system_report


def _wheel_name_version(filename: str) -> Tuple[str, str]:
    if not filename.endswith(".whl"):
        raise IdentityValidationError("wheel inventory contains a non-wheel file")
    parts = filename[:-4].split("-")
    if len(parts) < 5 or not _NAME.fullmatch(parts[0]) or not parts[1]:
        raise IdentityValidationError("wheel filename is not canonical: %s" % filename)
    return _normalize_name(parts[0]), parts[1]


def _wheel_inventory(
    wheelhouse: Union[str, Path],
    locked: Mapping[str, LockedRequirement],
) -> List[Dict[str, object]]:
    root = Path(wheelhouse)
    if not root.is_dir() or root.is_symlink():
        raise IdentityValidationError("wheelhouse must be a real directory")
    rows: List[Dict[str, object]] = []
    seen = set()
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise IdentityValidationError("wheel inventory contains a non-regular entry")
        name, wheel_version = _wheel_name_version(path.name)
        if name in seen:
            raise IdentityValidationError("duplicate wheel for locked package %s" % name)
        seen.add(name)
        requirement = locked.get(name)
        if requirement is None:
            raise IdentityValidationError("wheel is not present in the lock: %s" % path.name)
        if _base_version(wheel_version) != _base_version(requirement.version):
            raise IdentityValidationError("wheel version does not match the lock: %s" % path.name)
        digest = _sha256_file(path)
        if digest not in requirement.hashes:
            raise IdentityValidationError("wheel hash is not authorized by the lock: %s" % path.name)
        rows.append(
            {
                "filename": path.name,
                "name": name,
                "sha256": digest,
                "size": info.st_size,
                "version": wheel_version,
            }
        )
    missing = sorted(set(locked) - seen)
    if missing:
        raise IdentityValidationError("wheel inventory is incomplete; missing=%r" % missing)
    return rows


def build_environment_identity(
    lock_path: Union[str, Path],
    wheelhouse: Union[str, Path],
    *,
    python_report: Mapping[str, object],
    system_library_report: Mapping[str, object],
) -> Dict[str, object]:
    """Build the canonical content identity for one installable H200 environment."""
    lock = parse_hash_lock(lock_path)
    locked = _validate_core_lock(lock)
    python = _validate_python_report(python_report)
    systems = _validate_system_report(system_library_report)
    wheels = _wheel_inventory(wheelhouse, locked)
    payload: Dict[str, object] = {
        "cuda": {
            "device_name": systems["cuda_device_name"],
            "runtime": "12.8",
            "torch_build": "cu128",
        },
        "indexes": list(lock.indexes),
        "lock": {"sha256": lock.lock_sha256, "size": lock.lock_size},
        "platform": {"machine": "x86_64", "system": "linux"},
        "python": python,
        "pytorch": dict(_PYTORCH_VERSIONS),
        "schema": IDENTITY_PAYLOAD_SCHEMA,
        "system_libraries": systems,
        "wheels": wheels,
    }
    payload_sha256 = _sha256_bytes(canonical_json_bytes(payload))
    return {
        "environment_id": "wam-%s" % payload_sha256[:20],
        "payload": payload,
        "payload_sha256": payload_sha256,
        "schema": IDENTITY_ENVELOPE_SCHEMA,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            os.fchmod(handle.fileno(), mode)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _rename_noreplace_at(
    source_fd: int, source_name: str, destination_fd: int, destination_name: str
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise EnvironmentConflictError("renameat2 is required for environment publication")
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(source_fd, source, destination_fd, destination, 1)
    elif sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        if operation is None:
            raise EnvironmentConflictError("renameatx_np is required for environment publication")
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(source_fd, source, destination_fd, destination, 0x00000004)
    else:
        raise EnvironmentConflictError(
            "atomic no-replace rename is unsupported on %s" % sys.platform
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise EnvironmentConflictError("immutable final environment already exists")
    raise OSError(error, os.strerror(error), destination_name)


def _build_ownership(identity: Mapping[str, object]) -> Dict[str, object]:
    return {
        "environment_id": identity["environment_id"],
        "identity_sha256": _sha256_bytes(canonical_json_bytes(identity)),
        "schema": BUILD_OWNERSHIP_SCHEMA,
    }


def _directory_descriptor(path: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise EnvironmentConflictError("managed environment directory is unsafe") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise EnvironmentConflictError("managed environment path is not a directory")
    return descriptor


def _open_input_descriptor(path: Union[str, Path], *, directory: bool) -> int:
    absolute = Path(os.path.abspath(str(Path(path).expanduser())))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for component in absolute.parts[1:-1]:
            next_descriptor = os.open(
                component, directory_flags, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        flags = directory_flags if directory else (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        final_descriptor = os.open(
            absolute.name, flags, dir_fd=descriptor
        )
    except OSError as exc:
        raise IdentityValidationError(
            "environment input path contains a symlink or unsafe entry"
        ) from exc
    finally:
        os.close(descriptor)
    info = os.fstat(final_descriptor)
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(info.st_mode):
        os.close(final_descriptor)
        raise IdentityValidationError("environment input has the wrong file type")
    return final_descriptor


def _copy_regular_descriptor(
    source_fd: int, destination_parent_fd: int, destination_name: str
) -> None:
    before = os.fstat(source_fd)
    if not stat.S_ISREG(before.st_mode):
        raise IdentityValidationError("environment input is not a regular file")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    destination_fd = os.open(
        destination_name, flags, 0o400, dir_fd=destination_parent_fd
    )
    try:
        os.lseek(source_fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fchmod(destination_fd, 0o400)
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)
    after = os.fstat(source_fd)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise IdentityValidationError("environment input changed while being snapshotted")


def _regular_descriptor_inventory(descriptor: int) -> Dict[str, object]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise IdentityValidationError("environment snapshot entry is not a regular file")
    digest = hashlib.sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.lseek(descriptor, 0, os.SEEK_SET)
    after = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != before.st_size:
        raise IdentityValidationError(
            "environment snapshot entry changed while being hashed"
        )
    return {"sha256": digest.hexdigest(), "size": size}


def _snapshot_descriptor_inventory(
    snapshot: EnvironmentInputSnapshot,
) -> Dict[str, object]:
    before = os.fstat(snapshot.wheelhouse_fd)
    names = sorted(os.listdir(snapshot.wheelhouse_fd))
    wheels: List[Dict[str, object]] = []
    for name in names:
        if not name or "/" in name or name in (".", ".."):
            raise IdentityValidationError("wheel snapshot entry name is unsafe")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(name, flags, dir_fd=snapshot.wheelhouse_fd)
        except OSError as exc:
            raise IdentityValidationError(
                "wheel snapshot entry cannot be opened safely"
            ) from exc
        try:
            row = _regular_descriptor_inventory(descriptor)
        finally:
            os.close(descriptor)
        row["filename"] = name
        wheels.append(row)
    after = os.fstat(snapshot.wheelhouse_fd)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        names,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        sorted(os.listdir(snapshot.wheelhouse_fd)),
    ):
        raise IdentityValidationError(
            "wheel snapshot changed while being inventoried"
        )
    return {
        "lock": _regular_descriptor_inventory(snapshot.lock_fd),
        "wheels": wheels,
    }


def _identity_input_inventory(identity: Mapping[str, object]) -> Dict[str, object]:
    payload = identity.get("payload")
    if not isinstance(payload, dict):
        raise IdentityValidationError("environment identity payload is invalid")
    lock = payload.get("lock")
    wheels = payload.get("wheels")
    if not isinstance(lock, dict) or not isinstance(wheels, list):
        raise IdentityValidationError("environment identity inputs are invalid")
    return {
        "lock": {"sha256": lock.get("sha256"), "size": lock.get("size")},
        "wheels": [
            {
                "filename": row.get("filename"),
                "sha256": row.get("sha256"),
                "size": row.get("size"),
            }
            for row in wheels
            if isinstance(row, dict)
        ],
    }


@contextmanager
def _environment_input_snapshot(
    lock_path: Union[str, Path],
    wheelhouse: Union[str, Path],
    parent: Path,
) -> Iterator[EnvironmentInputSnapshot]:
    parent_fd = _directory_descriptor(parent)
    name = ".h200-input-%s" % uuid.uuid4().hex
    root_path = parent / name
    root_fd = -1
    wheelhouse_fd = -1
    lock_fd = -1
    source_lock_fd = -1
    source_wheelhouse_fd = -1
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        root_fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        os.mkdir("wheelhouse", mode=0o700, dir_fd=root_fd)
        os.mkdir("home", mode=0o700, dir_fd=root_fd)
        wheelhouse_fd = os.open(
            "wheelhouse",
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        source_lock_fd = _open_input_descriptor(lock_path, directory=False)
        _copy_regular_descriptor(
            source_lock_fd, root_fd, "requirements-h200.lock.txt"
        )
        source_wheelhouse_fd = _open_input_descriptor(wheelhouse, directory=True)
        for filename in sorted(os.listdir(source_wheelhouse_fd)):
            if not filename or "/" in filename or filename in (".", ".."):
                raise IdentityValidationError("wheelhouse entry name is unsafe")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                source_wheel_fd = os.open(
                    filename, flags, dir_fd=source_wheelhouse_fd
                )
            except OSError as exc:
                raise IdentityValidationError(
                    "wheelhouse entry cannot be opened safely"
                ) from exc
            try:
                _copy_regular_descriptor(
                    source_wheel_fd, wheelhouse_fd, filename
                )
            finally:
                os.close(source_wheel_fd)
        os.fchmod(wheelhouse_fd, 0o500)
        os.fchmod(root_fd, 0o500)
        os.fsync(wheelhouse_fd)
        os.fsync(root_fd)
        lock_fd = os.open(
            "requirements-h200.lock.txt",
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        snapshot = EnvironmentInputSnapshot(
            name=name,
            root_path=root_path,
            lock_path=root_path / "requirements-h200.lock.txt",
            wheelhouse_path=root_path / "wheelhouse",
            home_path=root_path / "home",
            parent_fd=parent_fd,
            root_fd=root_fd,
            lock_fd=lock_fd,
            wheelhouse_fd=wheelhouse_fd,
        )
        yield snapshot
    finally:
        try:
            if source_lock_fd >= 0:
                os.close(source_lock_fd)
            if source_wheelhouse_fd >= 0:
                os.close(source_wheelhouse_fd)
            if root_fd >= 0:
                try:
                    _remove_open_directory_at(parent_fd, name, root_fd)
                finally:
                    if lock_fd >= 0:
                        os.close(lock_fd)
                    if wheelhouse_fd >= 0:
                        os.close(wheelhouse_fd)
                    os.close(root_fd)
        finally:
            os.close(parent_fd)


def _create_owned_final(
    releases: Path, final: Path, identity: Mapping[str, object]
) -> Tuple[int, int]:
    releases_fd = _directory_descriptor(releases)
    preparation_name = ".h200-owner-%s" % uuid.uuid4().hex
    preparation = releases / preparation_name
    created = False
    try:
        os.mkdir(preparation_name, mode=0o755, dir_fd=releases_fd)
        created = True
        _atomic_write(
            preparation / BUILD_OWNERSHIP_NAME,
            canonical_json_bytes(_build_ownership(identity)),
            mode=0o444,
        )
        _rename_noreplace_at(
            releases_fd, preparation_name, releases_fd, final.name
        )
        created = False
        os.fsync(releases_fd)
        info = os.stat(final.name, dir_fd=releases_fd, follow_symlinks=False)
        return info.st_dev, info.st_ino
    finally:
        os.close(releases_fd)
        if created:
            shutil.rmtree(str(preparation), ignore_errors=True)


def _read_owned_candidate(final_fd: int, identity: Mapping[str, object]) -> None:
    try:
        marker = os.stat(COMPLETE_NAME, dir_fd=final_fd, follow_symlinks=False)
    except FileNotFoundError:
        marker = None
    if marker is not None:
        raise EnvironmentConflictError("completed or conflicting final environment exists")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(BUILD_OWNERSHIP_NAME, flags, dir_fd=final_fd)
    except OSError as exc:
        raise EnvironmentConflictError(
            "existing final environment exists and is not an owned incomplete candidate"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise EnvironmentConflictError("incomplete candidate ownership is unsafe")
        chunks = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise EnvironmentConflictError("incomplete candidate ownership changed while read")
    raw = b"".join(chunks)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentConflictError("incomplete candidate ownership is invalid") from exc
    expected = _build_ownership(identity)
    if value != expected or raw != canonical_json_bytes(expected):
        raise EnvironmentConflictError(
            "existing final environment ownership does not match its identity name"
        )


def _clear_directory_descriptor(descriptor: int) -> None:
    os.fchmod(descriptor, 0o700)
    for name in sorted(os.listdir(descriptor)):
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            child_fd = os.open(name, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child_fd)
                if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    raise EnvironmentConflictError(
                        "quarantine child changed while being opened"
                    )
                os.fchmod(child_fd, 0o700)
                tombstone = ".h200-delete-%s" % uuid.uuid4().hex
                _rename_noreplace_at(
                    descriptor, name, descriptor, tombstone
                )
                os.fsync(descriptor)
                renamed = os.stat(
                    tombstone, dir_fd=descriptor, follow_symlinks=False
                )
                if (opened.st_dev, opened.st_ino) != (
                    renamed.st_dev,
                    renamed.st_ino,
                ):
                    raise EnvironmentConflictError(
                        "quarantine child changed during tombstone rename"
                    )
                _clear_directory_descriptor(child_fd)
                observed = os.stat(
                    tombstone, dir_fd=descriptor, follow_symlinks=False
                )
                if (opened.st_dev, opened.st_ino) != (
                    observed.st_dev,
                    observed.st_ino,
                ):
                    raise EnvironmentConflictError(
                        "quarantine child identity changed before removal"
                    )
                os.rmdir(tombstone, dir_fd=descriptor)
            finally:
                os.close(child_fd)
        else:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            if stat.S_ISLNK(info.st_mode):
                if hasattr(os, "O_PATH"):
                    flags = (
                        getattr(os, "O_PATH")
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                elif hasattr(os, "O_SYMLINK"):
                    flags |= getattr(os, "O_SYMLINK")
                else:
                    raise EnvironmentConflictError(
                        "descriptor-bound symlink removal is unsupported"
                    )
            try:
                child_fd = os.open(name, flags, dir_fd=descriptor)
            except OSError as exc:
                raise EnvironmentConflictError(
                    "quarantine entry cannot be opened safely"
                ) from exc
            try:
                opened = os.fstat(child_fd)
                if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    raise EnvironmentConflictError(
                        "quarantine entry changed while being opened"
                    )
                tombstone = ".h200-delete-%s" % uuid.uuid4().hex
                _rename_noreplace_at(
                    descriptor, name, descriptor, tombstone
                )
                os.fsync(descriptor)
                renamed = os.stat(
                    tombstone, dir_fd=descriptor, follow_symlinks=False
                )
                if (opened.st_dev, opened.st_ino) != (
                    renamed.st_dev,
                    renamed.st_ino,
                ):
                    raise EnvironmentConflictError(
                        "quarantine entry changed during tombstone rename"
                    )
                observed = os.stat(
                    tombstone, dir_fd=descriptor, follow_symlinks=False
                )
                if (opened.st_dev, opened.st_ino) != (
                    observed.st_dev,
                    observed.st_ino,
                ):
                    raise EnvironmentConflictError(
                        "quarantine entry identity changed before removal"
                    )
                os.unlink(tombstone, dir_fd=descriptor)
            finally:
                os.close(child_fd)
    os.fsync(descriptor)


def _remove_open_directory_at(
    parent_fd: int, name: str, descriptor: int
) -> None:
    expected = os.fstat(descriptor)
    _clear_directory_descriptor(descriptor)
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (expected.st_dev, expected.st_ino) != (observed.st_dev, observed.st_ino):
        raise EnvironmentConflictError(
            "quarantine root identity changed before removal"
        )
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _recover_incomplete_candidate(
    releases: Path,
    final: Path,
    identity: Mapping[str, object],
    *,
    expected_inode: Optional[Tuple[int, int]] = None,
) -> None:
    releases_fd = _directory_descriptor(releases)
    final_fd = -1
    quarantine_name = ".h200-incomplete-%s-%s" % (
        identity["environment_id"],
        uuid.uuid4().hex,
    )
    quarantine = releases / quarantine_name
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            final_fd = os.open(final.name, flags, dir_fd=releases_fd)
        except OSError as exc:
            raise EnvironmentConflictError(
                "existing final environment is not a safe directory"
            ) from exc
        expected = os.fstat(final_fd)
        if expected_inode is not None and (
            expected.st_dev,
            expected.st_ino,
        ) != expected_inode:
            raise EnvironmentConflictError(
                "existing final environment was replaced during this build"
            )
        _read_owned_candidate(final_fd, identity)
        _rename_noreplace_at(
            releases_fd, final.name, releases_fd, quarantine_name
        )
        os.fsync(releases_fd)
        observed = os.stat(
            quarantine_name, dir_fd=releases_fd, follow_symlinks=False
        )
        if (expected.st_dev, expected.st_ino) != (observed.st_dev, observed.st_ino):
            raise EnvironmentConflictError("quarantined environment identity changed")
        _remove_open_directory_at(releases_fd, quarantine_name, final_fd)
    finally:
        if final_fd >= 0:
            os.close(final_fd)
        os.close(releases_fd)


@contextmanager
def exclusive_lock(lock_path: Union[str, Path]) -> Iterator[None]:
    """Hold a non-blocking process-exclusive, persistent deployment lock."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT | nofollow, 0o600)
    except OSError as exc:
        raise EnvironmentConflictError("cannot open H200 environment lock") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise EnvironmentConflictError("H200 environment lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise EnvironmentConflictError("H200 environment lock is already held") from exc
            raise
        os.ftruncate(descriptor, 0)
        os.write(descriptor, ("%s\n" % os.getpid()).encode("ascii"))
        os.fsync(descriptor)
        _fsync_directory(path.parent)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_identity_envelope(identity: object) -> Dict[str, object]:
    if not isinstance(identity, dict) or set(identity) != {
        "environment_id",
        "payload",
        "payload_sha256",
        "schema",
    }:
        raise EnvironmentVerificationError("environment identity fields are invalid")
    if identity.get("schema") != IDENTITY_ENVELOPE_SCHEMA:
        raise EnvironmentVerificationError("environment identity schema is invalid")
    payload = identity.get("payload")
    if not isinstance(payload, dict) or payload.get("schema") != IDENTITY_PAYLOAD_SCHEMA:
        raise EnvironmentVerificationError("environment identity payload is invalid")
    expected_payload_fields = {
        "cuda",
        "indexes",
        "lock",
        "platform",
        "python",
        "pytorch",
        "schema",
        "system_libraries",
        "wheels",
    }
    if set(payload) != expected_payload_fields:
        raise EnvironmentVerificationError("environment identity payload fields are invalid")
    if payload.get("indexes") != list(APPROVED_INDEXES):
        raise EnvironmentVerificationError("environment identity indexes are invalid")
    if payload.get("platform") != {"machine": "x86_64", "system": "linux"}:
        raise EnvironmentVerificationError("environment identity platform is invalid")
    cuda = payload.get("cuda")
    if (
        not isinstance(cuda, dict)
        or set(cuda) != {"device_name", "runtime", "torch_build"}
        or not isinstance(cuda.get("device_name"), str)
        or "H200" not in cuda["device_name"].upper()
        or cuda.get("runtime") != "12.8"
        or cuda.get("torch_build") != "cu128"
    ):
        raise EnvironmentVerificationError("environment identity CUDA target is invalid")
    if payload.get("pytorch") != _PYTORCH_VERSIONS:
        raise EnvironmentVerificationError("environment identity PyTorch target is invalid")
    lock = payload.get("lock")
    if (
        not isinstance(lock, dict)
        or set(lock) != {"sha256", "size"}
        or not isinstance(lock.get("sha256"), str)
        or not _HASH.fullmatch(lock["sha256"])
        or not isinstance(lock.get("size"), int)
        or lock["size"] <= 0
    ):
        raise EnvironmentVerificationError("environment identity lock binding is invalid")
    try:
        _validate_python_report(payload.get("python", {}))  # type: ignore[arg-type]
        _validate_system_report(payload.get("system_libraries", {}))  # type: ignore[arg-type]
    except IdentityValidationError as exc:
        raise EnvironmentVerificationError("environment identity runtime report is invalid") from exc
    wheels = payload.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise EnvironmentVerificationError("environment identity wheel inventory is invalid")
    previous_filename = None
    seen_names = set()
    wheel_versions: Dict[str, str] = {}
    for row in wheels:
        if not isinstance(row, dict) or set(row) != {
            "filename",
            "name",
            "sha256",
            "size",
            "version",
        }:
            raise EnvironmentVerificationError("environment identity wheel row is invalid")
        filename = row.get("filename")
        name = row.get("name")
        version = row.get("version")
        digest_value = row.get("sha256")
        size = row.get("size")
        if (
            not isinstance(filename, str)
            or not isinstance(name, str)
            or not isinstance(version, str)
            or not isinstance(digest_value, str)
            or not _HASH.fullmatch(digest_value)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise EnvironmentVerificationError("environment identity wheel value is invalid")
        parsed_name, parsed_version = _wheel_name_version(filename)
        if parsed_name != name or parsed_version != version:
            raise EnvironmentVerificationError("environment identity wheel filename is inconsistent")
        if name in seen_names:
            raise EnvironmentVerificationError("environment identity has duplicate wheel packages")
        if previous_filename is not None and filename <= previous_filename:
            raise EnvironmentVerificationError("environment identity wheels are not sorted")
        previous_filename = filename
        seen_names.add(name)
        wheel_versions[name] = version
    for name, expected in _CORE_VERSIONS.items():
        if _base_version(wheel_versions.get(name, "")) != expected:
            raise EnvironmentVerificationError("environment identity WAM-core version is invalid")
    digest = _sha256_bytes(canonical_json_bytes(payload))
    if identity.get("payload_sha256") != digest:
        raise EnvironmentVerificationError("environment identity hash mismatch")
    if identity.get("environment_id") != "wam-%s" % digest[:20]:
        raise EnvironmentVerificationError("environment id does not bind its payload")
    return identity


def _validate_smoke_report(smoke: object, identity: Mapping[str, object]) -> Dict[str, object]:
    if not isinstance(smoke, dict) or set(smoke) != {
        "checks",
        "environment_id",
        "packages",
        "runtime",
        "schema",
        "versions",
    }:
        raise SmokeValidationError("H200 smoke report fields are invalid")
    if smoke.get("schema") != SMOKE_SCHEMA:
        raise SmokeValidationError("H200 smoke report schema is invalid")
    if smoke.get("environment_id") != identity.get("environment_id"):
        raise SmokeValidationError("H200 smoke report binds the wrong environment")
    identity_payload = identity.get("payload")
    if not isinstance(identity_payload, dict):
        raise SmokeValidationError("H200 smoke identity payload is invalid")
    cuda = identity_payload.get("cuda")
    if not isinstance(cuda, dict):
        raise SmokeValidationError("H200 smoke identity CUDA target is invalid")
    expected_runtime = {
        "cuda_device_name": cuda.get("device_name"),
        "python": identity_payload.get("python"),
        "system_libraries": identity_payload.get("system_libraries"),
        "torch_cuda": cuda.get("runtime"),
    }
    if smoke.get("runtime") != expected_runtime:
        raise SmokeValidationError("H200 smoke runtime does not match environment identity")
    checks = smoke.get("checks")
    expected_checks = {
        "annotation_ui",
        "audio",
        "cuda_h200",
        "dataset_reader",
        "gpu_tensor",
        "perception",
        "reward_model",
    }
    if not isinstance(checks, dict) or set(checks) != expected_checks:
        raise SmokeValidationError("H200 smoke checks are incomplete")
    if any(checks.get(name) is not True for name in expected_checks):
        raise SmokeValidationError("H200 smoke check failed")
    if smoke.get("versions") != _CORE_VERSIONS:
        raise SmokeValidationError("H200 smoke package versions are invalid")
    wheels = identity.get("payload", {}).get("wheels", [])  # type: ignore[union-attr]
    expected_packages = [
        {"name": row["name"], "version": row["version"]}
        for row in wheels
    ]
    expected_packages.sort(key=lambda row: _normalize_name(row["name"]))
    if smoke.get("packages") != expected_packages:
        raise SmokeValidationError("H200 smoke installed package inventory drifted")
    return smoke


def _canonical_document(path: Path, label: str) -> Dict[str, object]:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise EnvironmentVerificationError("%s is not a regular file" % label)
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        raise EnvironmentVerificationError("%s is missing or invalid" % label) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise EnvironmentVerificationError("%s is not canonical JSON" % label)
    return value


def _installed_tree_inventory(root: Path) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for directory, dirnames, filenames in os.walk(str(root), followlinks=False):
        base = Path(directory)
        base_info = base.lstat()
        if (
            stat.S_ISLNK(base_info.st_mode)
            or not stat.S_ISDIR(base_info.st_mode)
            or base_info.st_mode & 0o222
        ):
            raise EnvironmentVerificationError(
                "installed tree contains an unsafe or writable directory"
            )
        dirnames.sort()
        filenames.sort()
        for name in dirnames:
            candidate = base / name
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise EnvironmentVerificationError(
                    "installed tree contains a symlink or non-directory entry"
                )
        for name in filenames:
            candidate = base / name
            relative = candidate.relative_to(root).as_posix()
            if relative == COMPLETE_NAME:
                continue
            before = candidate.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise EnvironmentVerificationError(
                    "installed tree contains a symlink or non-regular entry"
                )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(str(candidate), flags)
            except OSError as exc:
                raise EnvironmentVerificationError(
                    "installed tree file cannot be opened safely"
                ) from exc
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                ):
                    raise EnvironmentVerificationError(
                        "installed tree file changed while being opened"
                    )
                digest = hashlib.sha256()
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            observed = candidate.lstat()
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                stat.S_IMODE(before.st_mode),
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                stat.S_IMODE(after.st_mode),
            )
            identity_observed = (
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                observed.st_mtime_ns,
                stat.S_IMODE(observed.st_mode),
            )
            if identity_before != identity_after or identity_before != identity_observed:
                raise EnvironmentVerificationError(
                    "installed tree file changed while being inventoried"
                )
            rows.append(
                {
                    "mode": stat.S_IMODE(opened.st_mode),
                    "path": relative,
                    "sha256": digest.hexdigest(),
                    "size": opened.st_size,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "rows": rows,
        "schema": INSTALLED_TREE_SCHEMA,
        "sha256": _sha256_bytes(canonical_json_bytes(rows)),
    }


def _completion_marker(
    identity: Mapping[str, object],
    smoke: Mapping[str, object],
    installed_tree: Mapping[str, object],
) -> Dict[str, object]:
    payload = {
        "environment_id": identity["environment_id"],
        "identity_sha256": _sha256_bytes(canonical_json_bytes(identity)),
        "installed_tree": dict(installed_tree),
        "lock_sha256": identity["payload"]["lock"]["sha256"],  # type: ignore[index]
        "schema": COMPLETE_PAYLOAD_SCHEMA,
        "smoke_sha256": _sha256_bytes(canonical_json_bytes(smoke)),
    }
    return {
        "payload": payload,
        "payload_sha256": _sha256_bytes(canonical_json_bytes(payload)),
        "schema": COMPLETE_ENVELOPE_SCHEMA,
    }


def verify_complete_environment(environment_dir: Union[str, Path]) -> Dict[str, object]:
    """Verify canonical identity, smoke, and exact completion-marker bindings."""
    root = Path(environment_dir)
    if not root.is_dir() or root.is_symlink():
        raise EnvironmentVerificationError("complete environment directory is missing or unsafe")
    identity = _validate_identity_envelope(
        _canonical_document(root / IDENTITY_NAME, "environment identity")
    )
    if root.name != identity["environment_id"]:
        raise EnvironmentVerificationError("complete environment path does not match its id")
    smoke = _validate_smoke_report(
        _canonical_document(root / SMOKE_NAME, "H200 smoke report"), identity
    )
    marker = _canonical_document(root / COMPLETE_NAME, "environment complete marker")
    if (root / COMPLETE_NAME).lstat().st_mode & 0o222:
        raise EnvironmentVerificationError("environment complete marker is writable")
    installed_tree = _installed_tree_inventory(root)
    marker_payload = marker.get("payload")
    if (
        not isinstance(marker_payload, dict)
        or marker_payload.get("installed_tree") != installed_tree
    ):
        raise EnvironmentVerificationError("environment installed tree does not match completion")
    expected_marker = _completion_marker(identity, smoke, installed_tree)
    if marker != expected_marker:
        raise EnvironmentVerificationError("environment complete marker hash or schema mismatch")
    if root.stat().st_mode & 0o222:
        raise EnvironmentVerificationError("complete environment root is not immutable")
    return identity


def _run_command(
    argv: List[str],
    *,
    process_env: Optional[Dict[str, str]] = None,
    pass_fds: Tuple[int, ...] = (),
) -> None:
    subprocess.run(
        argv,
        executable=getattr(argv, "executable", None),
        check=True,
        env=process_env,
        pass_fds=pass_fds,
    )


def _normalize_pyvenv_config(
    root: Path,
    *,
    launcher: Path,
    base_python: Path,
    python_version: str,
) -> None:
    path = root / "pyvenv.cfg"
    if (
        not base_python.is_absolute()
        or base_python.name != "python3.12"
        or python_version != "3.12.13"
    ):
        raise EnvironmentVerificationError(
            "canonical base interpreter config identity is invalid"
        )
    generated = (
        "home = %s\n"
        "include-system-site-packages = false\n"
        "version = %s\n"
        "executable = %s\n"
        "command = %s -m venv --copies --without-pip %s\n"
        % (
            launcher.parent,
            python_version,
            launcher,
            launcher,
            root,
        )
    ).encode("utf-8")
    try:
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise EnvironmentVerificationError("venv config is missing or unsafe") from exc
    if not stat.S_ISREG(info.st_mode) or raw != generated:
        raise EnvironmentVerificationError("venv config does not match the bound build")
    canonical = (
        "home = %s\n"
        "include-system-site-packages = false\n"
        "version = %s\n"
        "executable = %s\n"
        "command = %s -m venv --copies --without-pip %s\n"
        % (
            base_python.parent,
            python_version,
            base_python,
            base_python,
            root,
        )
    ).encode("utf-8")
    _atomic_write(path, canonical)


def _raise_smoke_process_failure(
    completed: object, *, root: Path, identity: Mapping[str, object],
) -> None:
    returncode = getattr(completed, "returncode", None)
    if (
        type(returncode) is not int
        or returncode == 0
        or returncode < -255
        or returncode > 255
    ):
        raise SmokeValidationError("H200 smoke command failed with invalid exit status")
    stderr = getattr(completed, "stderr", None)
    if not isinstance(stderr, str):
        diagnostic = "<non-text stderr>"
    else:
        diagnostic = stderr.strip() or "<no stderr>"
        environment_id = identity.get("environment_id")
        if isinstance(environment_id, str) and environment_id:
            diagnostic = diagnostic.replace(environment_id, "<environment-id>")
        diagnostic = diagnostic.replace(
            str(root / IDENTITY_NAME), "<identity-path>"
        )
        marker = "...[truncated]"
        encoded = diagnostic.encode("utf-8")
        byte_budget = 2048 - len(marker.encode("utf-8"))
        if len(encoded) > 2048:
            diagnostic = encoded[:byte_budget].decode("utf-8", "ignore") + marker
    raise SmokeValidationError(
        "H200 smoke command failed (exit %s): %s" % (returncode, diagnostic)
    )


def _default_smoke_runner(root: Path, identity: Dict[str, object]) -> Dict[str, object]:
    script = Path(__file__).with_name("h200_smoke.py")
    with tempfile.TemporaryDirectory(prefix="wam-h200-smoke-home-") as raw_home:
        completed = subprocess.run(
            [
                str(root / "bin" / "python"),
                "-B",
                "-I",
                str(script),
                "--environment-id",
                str(identity["environment_id"]),
                "--identity",
                str(root / IDENTITY_NAME),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_runtime_process_env(Path(raw_home)),
        )
    returncode = getattr(completed, "returncode", None)
    if (
        type(returncode) is not int
        or returncode < -255
        or returncode > 255
    ):
        raise SmokeValidationError("H200 smoke command failed with invalid exit status")
    if returncode != 0:
        _raise_smoke_process_failure(completed, root=root, identity=identity)
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SmokeValidationError("H200 smoke command did not emit JSON") from exc
    if not isinstance(value, dict):
        raise SmokeValidationError("H200 smoke command emitted a non-object")
    return value


def _make_tree_read_only(root: Path) -> None:
    directories: List[Path] = []
    for directory, dirnames, filenames in os.walk(str(root), followlinks=False):
        base = Path(directory)
        directories.append(base)
        for name in filenames:
            path = base / name
            if path.is_symlink():
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(mode & ~0o222)
        for name in list(dirnames):
            path = base / name
            if path.is_symlink():
                dirnames.remove(name)
    for directory in reversed(directories):
        mode = stat.S_IMODE(directory.stat().st_mode)
        directory.chmod(mode & ~0o222)


def _make_tree_owner_writable(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for directory, dirnames, filenames in os.walk(str(root), topdown=True, followlinks=False):
        base = Path(directory)
        try:
            base.chmod(stat.S_IMODE(base.stat().st_mode) | stat.S_IWUSR | stat.S_IXUSR)
        except FileNotFoundError:
            continue
        for name in filenames:
            path = base / name
            if not path.is_symlink():
                try:
                    path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
                except FileNotFoundError:
                    pass
        for name in list(dirnames):
            if (base / name).is_symlink():
                dirnames.remove(name)


def _ensure_canonical_directory(
    path: Path, error_type: type, label: str
) -> None:
    absolute = Path(os.path.abspath(str(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            created = False
            try:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            except OSError as exc:
                raise error_type("%s is unsafe" % label) from exc
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise error_type("%s contains a symlink or non-directory ancestor" % label) from exc
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise error_type("%s contains a non-directory ancestor" % label)
            if created:
                os.fsync(descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    finally:
        os.close(descriptor)


def _assert_managed_root(releases_root: Union[str, Path]) -> Path:
    root = Path(os.path.abspath(str(Path(releases_root).expanduser())))
    protected = {
        (Path.home() / "envs" / "wam").absolute(),
        (Path.home() / "envs" / "wam-current").absolute(),
    }
    if any(root == item or item in root.parents for item in protected):
        raise PointerPolicyError("legacy and production WAM paths are forbidden")
    _ensure_canonical_directory(
        root, EnvironmentConflictError, "environment releases root"
    )
    return root


def build_environment(
    lock_path: Union[str, Path],
    wheelhouse: Union[str, Path],
    *,
    releases_root: Union[str, Path],
    python_report: Mapping[str, object],
    system_library_report: Mapping[str, object],
    python_executable: str = H200_BASE_PYTHON,
    command_runner: Callable[..., None] = _run_command,
    smoke_runner: Callable[[Path, Dict[str, object]], Dict[str, object]] = _default_smoke_runner,
) -> Path:
    """Build directly in an immutable final release path and mark only after smoke."""
    base_python = resolve_base_python(python_executable)
    base_python_fd = open_base_python(base_python)
    try:
        _verify_base_python_path(base_python, base_python_fd)
        executable = _descriptor_path(base_python_fd)
        reported_python = _validate_python_report(python_report)
        reported_system = _validate_system_report(system_library_report)
        probed_python, probed_system = probe_runtime(
            executable, pass_fds=(base_python_fd,)
        )
        _verify_base_python_path(base_python, base_python_fd)
        if reported_python != probed_python:
            raise IdentityValidationError(
                "caller Python report does not match the probed interpreter"
            )
        if reported_system != probed_system:
            raise IdentityValidationError(
                "caller system report does not match the probed H200 host"
            )
        return _build_environment_with_base(
            lock_path,
            wheelhouse,
            releases_root=releases_root,
            probed_python=probed_python,
            probed_system=probed_system,
            base_python=base_python,
            base_python_fd=base_python_fd,
            command_runner=command_runner,
            smoke_runner=smoke_runner,
        )
    finally:
        os.close(base_python_fd)


def _build_environment_with_base(
    lock_path: Union[str, Path],
    wheelhouse: Union[str, Path],
    *,
    releases_root: Union[str, Path],
    probed_python: Mapping[str, object],
    probed_system: Mapping[str, object],
    base_python: str,
    base_python_fd: int,
    command_runner: Callable[..., None],
    smoke_runner: Callable[[Path, Dict[str, object]], Dict[str, object]],
) -> Path:
    releases = _assert_managed_root(releases_root)
    lock_file = releases.parent / ".h200-env.lock"
    with exclusive_lock(lock_file):
        with _environment_input_snapshot(
            lock_path, wheelhouse, releases.parent
        ) as snapshot:
            return _build_candidate_from_snapshot(
                snapshot,
                releases=releases,
                probed_python=probed_python,
                probed_system=probed_system,
                base_python=base_python,
                base_python_fd=base_python_fd,
                command_runner=command_runner,
                smoke_runner=smoke_runner,
            )


def _build_candidate_from_snapshot(
    snapshot: EnvironmentInputSnapshot,
    *,
    releases: Path,
    probed_python: Mapping[str, object],
    probed_system: Mapping[str, object],
    base_python: str,
    base_python_fd: int,
    command_runner: Callable[..., None],
    smoke_runner: Callable[[Path, Dict[str, object]], Dict[str, object]],
) -> Path:
    descriptor_inputs = _snapshot_descriptor_inventory(snapshot)
    identity = build_environment_identity(
        snapshot.lock_path,
        snapshot.wheelhouse_path,
        python_report=probed_python,
        system_library_report=probed_system,
    )
    if _identity_input_inventory(identity) != descriptor_inputs:
        raise IdentityValidationError(
            "environment identity does not match its descriptor snapshot"
        )
    final = releases / str(identity["environment_id"])
    created_inode: Optional[Tuple[int, int]] = None
    if os.path.lexists(str(final)):
        _recover_incomplete_candidate(releases, final, identity)
    created_inode = _create_owned_final(releases, final, identity)
    try:
        process_env = _runtime_process_env(snapshot.home_path)
        venv_command = _bound_base_python_command(
            base_python_fd,
            snapshot.home_path,
            [
                "-I",
                "-S",
                "-m",
                "venv",
                "--copies",
                "--without-pip",
                str(final),
            ],
        )
        command_runner(
            venv_command,
            process_env=process_env,
            pass_fds=(base_python_fd,),
        )
        _verify_bound_base_python(venv_command[0], base_python_fd)
        _verify_base_python_path(base_python, base_python_fd)
        _normalize_pyvenv_config(
            final,
            launcher=Path(venv_command[0]),
            base_python=Path(base_python),
            python_version=str(probed_python["version"]),
        )
        ensurepip_python = final / "bin" / "python3.12"
        if ensurepip_python.is_symlink() or not ensurepip_python.is_file():
            raise EnvironmentVerificationError(
                "normalized venv did not create bin/python3.12"
            )
        command_runner(
            [
                str(ensurepip_python),
                "-I",
                "-m",
                "ensurepip",
                "--upgrade",
                "--default-pip",
            ],
            process_env=process_env,
            pass_fds=(),
        )
        environment_pip = final / "bin" / "pip"
        if environment_pip.is_symlink() or not environment_pip.is_file():
            raise EnvironmentVerificationError("ensurepip did not create bin/pip")
        lib64 = final / "lib64"
        if os.path.lexists(str(lib64)):
            info = lib64.lstat()
            if (
                not stat.S_ISLNK(info.st_mode)
                or os.readlink(str(lib64)) != "lib"
                or (final / "lib").is_symlink()
                or not (final / "lib").is_dir()
            ):
                raise EnvironmentVerificationError(
                    "venv lib64 entry is not the expected local lib symlink"
                )
            lib64.unlink()
            _fsync_directory(final)
        environment_python = final / "bin" / "python"
        if not environment_python.exists():
            raise EnvironmentVerificationError("venv did not create bin/python")
        command_runner(
            [
                str(environment_python),
                "-I",
                "-m",
                "pip",
                "--isolated",
                "install",
                "--no-index",
                "--find-links",
                _descriptor_path(snapshot.wheelhouse_fd),
                "--require-hashes",
                "-r",
                _descriptor_path(snapshot.lock_fd),
            ],
            process_env=process_env,
            pass_fds=(snapshot.wheelhouse_fd, snapshot.lock_fd),
        )
        if _snapshot_descriptor_inventory(snapshot) != descriptor_inputs:
            raise IdentityValidationError(
                "environment input snapshot changed during installation"
            )
        _verify_base_python_path(base_python, base_python_fd)
        _atomic_write(final / IDENTITY_NAME, canonical_json_bytes(identity))
        smoke = _validate_smoke_report(smoke_runner(final, identity), identity)
        _atomic_write(final / SMOKE_NAME, canonical_json_bytes(smoke))
        _make_tree_read_only(final)
        installed_tree = _installed_tree_inventory(final)
        marker = _completion_marker(identity, smoke, installed_tree)
        final_mode = stat.S_IMODE(final.stat().st_mode)
        final.chmod(final_mode | stat.S_IWUSR)
        try:
            _atomic_write(
                final / COMPLETE_NAME,
                canonical_json_bytes(marker),
                mode=0o444,
            )
        finally:
            final.chmod(final_mode)
        _fsync_directory(final)
        verify_complete_environment(final)
        return final
    except BaseException:
        if created_inode is not None and os.path.lexists(str(final)):
            try:
                _recover_incomplete_candidate(
                    releases,
                    final,
                    identity,
                    expected_inode=created_inode,
                )
            except (H200EnvironmentError, OSError):
                pass
        raise


def _assert_canary_pointer(pointer: Union[str, Path]) -> Path:
    path = Path(os.path.abspath(str(Path(pointer).expanduser())))
    protected = {
        (Path.home() / "envs" / "wam").absolute(),
        (Path.home() / "envs" / "wam-current").absolute(),
    }
    if (
        any(path == item or item in path.parents for item in protected)
        or path.name in ("wam", "wam-current")
    ):
        raise PointerPolicyError("legacy ~/envs/wam and production wam-current are forbidden")
    if "canary" not in path.parts and ".deploy-canary" not in path.parts:
        raise PointerPolicyError("environment pointer must be inside an isolated canary path")
    return path


def _reject_symlink_ancestors(
    path: Path,
    error_type: type,
    label: str,
) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise error_type("%s contains a symlinked ancestor" % label)


def _verified_release_target(candidate: Union[str, Path], releases: Path) -> Path:
    path = Path(candidate).expanduser().resolve(strict=True)
    if path.parent != releases.resolve():
        raise PointerPolicyError("canary target is outside the environment release root")
    try:
        verify_complete_environment(path)
    except EnvironmentVerificationError as exc:
        raise EnvironmentVerificationError(
            "canary target is not a complete environment"
        ) from exc
    return path


def _current_target(current: Path, releases: Path) -> Optional[Path]:
    if not os.path.lexists(str(current)):
        return None
    if not current.is_symlink():
        raise PointerConflictError("canary current path is not a symlink")
    try:
        resolved = current.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PointerConflictError("canary current symlink is dangling") from exc
    if resolved.parent != releases.resolve():
        raise PointerPolicyError("canary current target is outside releases")
    verify_complete_environment(resolved)
    return resolved


def _expected_target(
    expected: Optional[Union[str, Path]], releases: Path
) -> Optional[Path]:
    if expected is None:
        return None
    return _verified_release_target(expected, releases)


def _replace_symlink(current: Path, target: Path) -> None:
    temporary = current.parent / (".%s.switch-%s" % (current.name, uuid.uuid4().hex))
    temporary.symlink_to(os.path.relpath(str(target), str(current.parent)))
    try:
        os.replace(str(temporary), str(current))
        _fsync_directory(current.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _restore_canary_pointer(current: Path, previous: Optional[Path]) -> None:
    if previous is None:
        try:
            current.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(current.parent)
        return
    _replace_symlink(current, previous)


def _switch_canary(
    desired: Optional[Union[str, Path]],
    current_link: Union[str, Path],
    *,
    releases_root: Union[str, Path],
    expected_current: Optional[Union[str, Path]],
) -> Optional[Path]:
    current = _assert_canary_pointer(current_link)
    releases = _assert_managed_root(releases_root)
    desired_path = None if desired is None else _verified_release_target(desired, releases)
    expected_path = _expected_target(expected_current, releases)
    _ensure_canonical_directory(
        current.parent, PointerPolicyError, "canary pointer parent"
    )
    lock_file = releases.parent / ".h200-env.lock"
    with exclusive_lock(lock_file):
        previous = _current_target(current, releases)
        if previous != expected_path:
            raise PointerConflictError(
                "canary compare-and-swap mismatch: expected=%r actual=%r"
                % (expected_path, previous)
            )
        if desired_path is None:
            if previous is not None:
                current.unlink()
                _fsync_directory(current.parent)
        else:
            _verified_release_target(desired_path, releases)
            try:
                _replace_symlink(current, desired_path)
                active = _current_target(current, releases)
                if active != desired_path:
                    raise EnvironmentVerificationError("canary pointer verification failed")
            except BaseException:
                _restore_canary_pointer(current, previous)
                raise
        return previous


def activate_canary(
    candidate: Union[str, Path],
    current_link: Union[str, Path],
    *,
    releases_root: Union[str, Path],
    expected_current: Optional[Union[str, Path]],
) -> Optional[Path]:
    """CAS-activate a completed release only on an isolated canary pointer."""
    return _switch_canary(
        candidate,
        current_link,
        releases_root=releases_root,
        expected_current=expected_current,
    )


def rollback_canary(
    current_link: Union[str, Path],
    previous_target: Optional[Union[str, Path]],
    *,
    releases_root: Union[str, Path],
    expected_current: Optional[Union[str, Path]],
) -> Optional[Path]:
    """CAS-rollback a canary pointer to its verified previous target (or absent)."""
    return _switch_canary(
        previous_target,
        current_link,
        releases_root=releases_root,
        expected_current=expected_current,
    )


def _descriptor_identity(info: os.stat_result) -> Tuple[int, int]:
    return info.st_dev, info.st_ino


def _open_real_directory_at(parent_fd: int, name: str, label: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise PointerPolicyError("%s is missing, symlinked, or unsafe" % label) from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise PointerPolicyError("%s is not a real directory" % label)
    return descriptor


def _open_absolute_real_directory(path: Path, label: str) -> int:
    absolute = Path(os.path.abspath(str(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(absolute.anchor, flags)
        try:
            for component in absolute.parts[1:]:
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except BaseException:
            os.close(descriptor)
            raise
    except OSError as exc:
        raise PointerPolicyError("%s contains a symlink or unsafe ancestor" % label) from exc
    return descriptor


def _read_regular_snapshot_at(
    parent_fd: int, name: str, label: str
) -> Tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise PointerPolicyError("%s is missing or unsafe" % label) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _descriptor_identity(opened) != _descriptor_identity(before)
        ):
            raise PointerPolicyError("%s changed while being opened" % label)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            stat.S_IMODE(value.st_mode),
        )
        if identity(opened) != identity(after) or identity(opened) != identity(observed):
            raise PointerPolicyError("%s changed while being read" % label)
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def _read_regular_at(parent_fd: int, name: str, label: str) -> bytes:
    return _read_regular_snapshot_at(parent_fd, name, label)[0]


def _canonical_json_at(parent_fd: int, name: str, label: str) -> Tuple[Dict[str, object], bytes]:
    raw = _read_regular_at(parent_fd, name, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PointerPolicyError("%s is invalid JSON" % label) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise PointerPolicyError("%s is not canonical JSON" % label)
    return value, raw


def _canonical_artifact_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise PointerPolicyError("frozen code manifest path is invalid")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or any(part in ("", ".", "..") for part in parsed.parts)
        or parsed.as_posix() != value
        or value in (CODE_MANIFEST_NAME, CODE_COMPLETE_NAME)
    ):
        raise PointerPolicyError("frozen code manifest path is not canonical")
    return value


def _regular_tree_inventory(root_fd: int) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}

    def walk(directory_fd: int, prefix: str) -> None:
        before = os.fstat(directory_fd)
        if not stat.S_ISDIR(before.st_mode):
            raise PointerPolicyError("frozen code release contains an unsafe directory")
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise PointerPolicyError("frozen code release cannot be inventoried") from exc
        for name in names:
            if name in (".", "..") or "/" in name:
                raise PointerPolicyError("frozen code release entry is invalid")
            relative = name if not prefix else prefix + "/" + name
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise PointerPolicyError("frozen code release changed during inventory") from exc
            if stat.S_ISDIR(info.st_mode):
                child = _open_real_directory_at(
                    directory_fd, name, "frozen code directory"
                )
                try:
                    if _descriptor_identity(os.fstat(child)) != _descriptor_identity(info):
                        raise PointerPolicyError(
                            "frozen code directory changed while being opened"
                        )
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                data, file_info = _read_regular_snapshot_at(
                    directory_fd, name, "frozen code file"
                )
                rows[relative] = {
                    "mode": format(stat.S_IMODE(file_info.st_mode), "04o"),
                    "sha256": _sha256_bytes(data),
                    "size": len(data),
                }
            else:
                raise PointerPolicyError(
                    "frozen code release contains a symlink or non-regular entry"
                )
        after = os.fstat(directory_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise PointerPolicyError("frozen code directory changed during inventory")

    walk(root_fd, "")
    return rows


def _verify_frozen_code_release(
    release_fd: int,
    release_name: str,
    frozen_release_id: str,
    frozen_payload_sha256: str,
) -> Dict[str, object]:
    manifest, manifest_raw = _canonical_json_at(
        release_fd, CODE_MANIFEST_NAME, "frozen code release manifest"
    )
    if set(manifest) != {"payload", "payload_sha256", "release_id", "schema"}:
        raise PointerPolicyError("frozen code manifest fields are invalid")
    payload = manifest.get("payload")
    if (
        manifest.get("schema") != CODE_ENVELOPE_SCHEMA
        or not isinstance(payload, dict)
        or set(payload) != {"artifact_kind", "files", "schema", "source_role"}
        or payload.get("schema") != CODE_PAYLOAD_SCHEMA
        or payload.get("artifact_kind") != "code_release"
        or payload.get("source_role") != "mac"
    ):
        raise PointerPolicyError("frozen code release manifest schema is invalid")
    payload_hash = _sha256_bytes(canonical_json_bytes(payload))
    expected_id = "code_release-" + payload_hash[:20]
    if (
        manifest.get("payload_sha256") != payload_hash
        or manifest.get("release_id") != expected_id
        or release_name != expected_id
        or frozen_release_id != expected_id
        or frozen_payload_sha256 != payload_hash
    ):
        raise PointerPolicyError("frozen code manifest identity does not match confirmation")
    files = payload.get("files")
    if not isinstance(files, list):
        raise PointerPolicyError("frozen code manifest file list is invalid")
    expected: Dict[str, Dict[str, object]] = {}
    for row in files:
        if not isinstance(row, dict) or set(row) != {"mode", "path", "sha256", "size"}:
            raise PointerPolicyError("frozen code manifest file row is invalid")
        relative = _canonical_artifact_path(row.get("path"))
        if relative in expected:
            raise PointerPolicyError("frozen code manifest contains a duplicate path")
        if (
            not isinstance(row.get("mode"), str)
            or not re.fullmatch(r"0[0-7]{3}", str(row["mode"]))
            or not isinstance(row.get("sha256"), str)
            or not _HASH.fullmatch(str(row["sha256"]))
            or not isinstance(row.get("size"), int)
            or isinstance(row.get("size"), bool)
            or int(row["size"]) < 0
        ):
            raise PointerPolicyError("frozen code manifest file metadata is invalid")
        expected[relative] = {
            "mode": row["mode"],
            "sha256": row["sha256"],
            "size": row["size"],
        }
    inventory = _regular_tree_inventory(release_fd)
    completion, completion_raw = _canonical_json_at(
        release_fd, CODE_COMPLETE_NAME, "frozen code transfer completion marker"
    )
    payload_inventory = {
        key: value
        for key, value in inventory.items()
        if key not in (CODE_MANIFEST_NAME, CODE_COMPLETE_NAME)
    }
    if payload_inventory != expected or set(inventory) != set(expected) | {
        CODE_MANIFEST_NAME,
        CODE_COMPLETE_NAME,
    }:
        raise PointerPolicyError("frozen code release exact file set or hash is invalid")
    expected_completion = {
        "artifact_kind": "code_release",
        "manifest_sha256": _sha256_bytes(manifest_raw),
        "payload_sha256": payload_hash,
        "release_id": expected_id,
        "schema": CODE_COMPLETE_SCHEMA,
        "source_role": "mac",
        "target_role": "h200",
        "verified_files": len(expected),
    }
    if completion != expected_completion:
        raise PointerPolicyError("frozen code completion marker does not bind the manifest")
    return {
        "completion_sha256": _sha256_bytes(completion_raw),
        "manifest_sha256": _sha256_bytes(manifest_raw),
        "payload_sha256": payload_hash,
        "release_id": expected_id,
    }


@dataclass
class _LockedProductionCode:
    root_path: Path
    root_fd: int
    releases_fd: int
    release_fd: int
    lock_fd: int
    release_name: str
    current_text: str
    evidence: Dict[str, object]


@contextmanager
def _locked_production_code(
    home: Path,
    frozen_release_id: str,
    frozen_payload_sha256: str,
) -> Iterator[_LockedProductionCode]:
    root_path = home / "wam_ausc"
    root_fd = _open_absolute_real_directory(root_path, "production code root")
    releases_fd = -1
    release_fd = -1
    lock_fd = -1
    try:
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            lock_fd = os.open(".wam-production.lock", flags, 0o600, dir_fd=root_fd)
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise PointerPolicyError("production code lock is not a regular file")
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise EnvironmentConflictError("production code lock cannot be acquired") from exc
        releases_fd = _open_real_directory_at(
            root_fd, ".releases", "production code releases root"
        )
        try:
            current_info = os.stat("current", dir_fd=root_fd, follow_symlinks=False)
            if not stat.S_ISLNK(current_info.st_mode):
                raise PointerPolicyError("production code current is not a symlink")
            current_text = os.readlink("current", dir_fd=root_fd)
        except OSError as exc:
            raise PointerPolicyError("production code current pointer is missing") from exc
        expected_text = ".releases/" + frozen_release_id
        if current_text != expected_text:
            raise PointerPolicyError("production code current pointer is not canonical or frozen")
        release_fd = _open_real_directory_at(
            releases_fd, frozen_release_id, "frozen code release"
        )
        evidence = _verify_frozen_code_release(
            release_fd,
            frozen_release_id,
            frozen_release_id,
            frozen_payload_sha256,
        )
        _assert_path_maps_to_descriptor(
            root_path, root_fd, "production code root"
        )
        _assert_child_maps_to_descriptor(
            root_fd,
            ".releases",
            releases_fd,
            "production code releases root",
        )
        _assert_child_maps_to_descriptor(
            releases_fd,
            frozen_release_id,
            release_fd,
            "frozen code release",
        )
        try:
            final_current_info = os.stat(
                "current", dir_fd=root_fd, follow_symlinks=False
            )
            final_current_text = os.readlink("current", dir_fd=root_fd)
        except OSError as exc:
            raise PointerPolicyError(
                "production code current changed before activation"
            ) from exc
        if (
            not stat.S_ISLNK(final_current_info.st_mode)
            or final_current_text != current_text
        ):
            raise PointerPolicyError(
                "production code current changed before activation"
            )
        yield _LockedProductionCode(
            root_path=root_path,
            root_fd=root_fd,
            releases_fd=releases_fd,
            release_fd=release_fd,
            lock_fd=lock_fd,
            release_name=frozen_release_id,
            current_text=current_text,
            evidence=evidence,
        )
    finally:
        if release_fd >= 0:
            os.close(release_fd)
        if releases_fd >= 0:
            os.close(releases_fd)
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(root_fd)


def _reverify_locked_production_code(
    locked: _LockedProductionCode,
    frozen_release_id: str,
    frozen_payload_sha256: str,
) -> Dict[str, object]:
    _assert_path_maps_to_descriptor(
        locked.root_path, locked.root_fd, "production code root"
    )
    _assert_child_maps_to_descriptor(
        locked.root_fd,
        ".releases",
        locked.releases_fd,
        "production code releases root",
    )
    _assert_child_maps_to_descriptor(
        locked.releases_fd,
        locked.release_name,
        locked.release_fd,
        "frozen code release",
    )
    try:
        current_info = os.stat("current", dir_fd=locked.root_fd, follow_symlinks=False)
        current_text = os.readlink("current", dir_fd=locked.root_fd)
    except OSError as exc:
        raise PointerPolicyError("production code current changed during activation") from exc
    if not stat.S_ISLNK(current_info.st_mode) or current_text != locked.current_text:
        raise PointerPolicyError("production code current changed during activation")
    evidence = _verify_frozen_code_release(
        locked.release_fd,
        locked.release_name,
        frozen_release_id,
        frozen_payload_sha256,
    )
    if evidence != locked.evidence:
        raise PointerPolicyError("frozen code evidence changed during activation")
    return evidence


def _assert_path_maps_to_descriptor(path: Path, descriptor: int, label: str) -> None:
    fresh = _open_absolute_real_directory(path, label)
    try:
        if _descriptor_identity(os.fstat(fresh)) != _descriptor_identity(
            os.fstat(descriptor)
        ):
            raise PointerPolicyError("%s changed during activation" % label)
    finally:
        os.close(fresh)


def _assert_child_maps_to_descriptor(
    parent_fd: int,
    name: str,
    descriptor: int,
    label: str,
) -> None:
    try:
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise PointerPolicyError("%s changed during activation" % label) from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or _descriptor_identity(observed) != _descriptor_identity(os.fstat(descriptor))
    ):
        raise PointerPolicyError("%s changed during activation" % label)


def _environment_tree_inventory_fd(root_fd: int) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []

    def walk(directory_fd: int, prefix: str) -> None:
        before = os.fstat(directory_fd)
        if not stat.S_ISDIR(before.st_mode) or before.st_mode & 0o222:
            raise EnvironmentVerificationError(
                "production environment contains an unsafe or writable directory"
            )
        for name in sorted(os.listdir(directory_fd)):
            relative = name if not prefix else prefix + "/" + name
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise EnvironmentVerificationError(
                    "production environment changed during inventory"
                ) from exc
            if stat.S_ISDIR(info.st_mode):
                child = _open_real_directory_at(
                    directory_fd, name, "production environment directory"
                )
                try:
                    if _descriptor_identity(os.fstat(child)) != _descriptor_identity(info):
                        raise EnvironmentVerificationError(
                            "production environment directory changed while opening"
                        )
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                if relative == COMPLETE_NAME:
                    continue
                data, file_info = _read_regular_snapshot_at(
                    directory_fd, name, "production environment file"
                )
                rows.append(
                    {
                        "mode": stat.S_IMODE(file_info.st_mode),
                        "path": relative,
                        "sha256": _sha256_bytes(data),
                        "size": len(data),
                    }
                )
            else:
                raise EnvironmentVerificationError(
                    "production environment contains a symlink or non-regular entry"
                )
        after = os.fstat(directory_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise EnvironmentVerificationError(
                "production environment directory changed during inventory"
            )

    walk(root_fd, "")
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "rows": rows,
        "schema": INSTALLED_TREE_SCHEMA,
        "sha256": _sha256_bytes(canonical_json_bytes(rows)),
    }


def _verify_complete_environment_fd(
    release_fd: int, release_name: str
) -> Dict[str, object]:
    identity_doc, _identity_raw = _canonical_json_at(
        release_fd, IDENTITY_NAME, "production environment identity"
    )
    identity = _validate_identity_envelope(identity_doc)
    if identity.get("environment_id") != release_name:
        raise EnvironmentVerificationError(
            "production environment path does not match its identity"
        )
    smoke_doc, _smoke_raw = _canonical_json_at(
        release_fd, SMOKE_NAME, "production H200 smoke report"
    )
    smoke = _validate_smoke_report(smoke_doc, identity)
    marker, _marker_raw = _canonical_json_at(
        release_fd, COMPLETE_NAME, "production environment completion marker"
    )
    marker_info = os.stat(COMPLETE_NAME, dir_fd=release_fd, follow_symlinks=False)
    if marker_info.st_mode & 0o222:
        raise EnvironmentVerificationError(
            "production environment completion marker is writable"
        )
    installed_tree = _environment_tree_inventory_fd(release_fd)
    marker_payload = marker.get("payload")
    if (
        not isinstance(marker_payload, dict)
        or marker_payload.get("installed_tree") != installed_tree
        or marker != _completion_marker(identity, smoke, installed_tree)
    ):
        raise EnvironmentVerificationError(
            "production environment completion does not bind its exact tree"
        )
    if os.fstat(release_fd).st_mode & 0o222:
        raise EnvironmentVerificationError("production environment root is writable")
    return identity


@dataclass
class _LockedProductionEnvironment:
    envs_path: Path
    releases_path: Path
    envs_fd: int
    releases_fd: int
    desired_fd: int
    expected_fd: int
    lock_fd: int
    desired_name: str
    expected_name: Optional[str]
    desired_identity: Dict[str, object]
    previous_pointer_identity: Optional[Tuple[int, int, int, int, int]] = None
    activated_pointer_identity: Optional[Tuple[int, int, int, int, int]] = None


def _canonical_environment_name(
    candidate: Union[str, Path], releases_path: Path, label: str
) -> str:
    path = Path(os.path.abspath(str(Path(candidate).expanduser())))
    if path.parent != releases_path or not _NAME.fullmatch(path.name):
        raise PointerPolicyError("%s is not a direct canonical release" % label)
    return path.name


@contextmanager
def _locked_production_environment(
    home: Path,
    candidate: Union[str, Path],
    expected_current: Optional[Union[str, Path]],
) -> Iterator[_LockedProductionEnvironment]:
    envs_path = home / "envs"
    releases_path = envs_path / "releases"
    desired_name = _canonical_environment_name(
        candidate, releases_path, "production environment"
    )
    expected_name = (
        None
        if expected_current is None
        else _canonical_environment_name(
            expected_current, releases_path, "expected production environment"
        )
    )
    envs_fd = _open_absolute_real_directory(envs_path, "production environment root")
    releases_fd = -1
    desired_fd = -1
    expected_fd = -1
    lock_fd = -1
    try:
        releases_fd = _open_real_directory_at(
            envs_fd, "releases", "production environment releases root"
        )
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            lock_fd = os.open(".h200-env.lock", flags, 0o600, dir_fd=envs_fd)
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise PointerPolicyError("H200 environment lock is not a regular file")
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise EnvironmentConflictError("H200 environment lock is already held") from exc
        desired_fd = _open_real_directory_at(
            releases_fd, desired_name, "production environment release"
        )
        desired_identity = _verify_complete_environment_fd(desired_fd, desired_name)
        if expected_name is not None:
            if expected_name == desired_name:
                expected_fd = os.dup(desired_fd)
            else:
                expected_fd = _open_real_directory_at(
                    releases_fd,
                    expected_name,
                    "expected production environment release",
                )
                _verify_complete_environment_fd(expected_fd, expected_name)
        _assert_path_maps_to_descriptor(envs_path, envs_fd, "production environment root")
        _assert_path_maps_to_descriptor(
            releases_path, releases_fd, "production environment releases root"
        )
        yield _LockedProductionEnvironment(
            envs_path=envs_path,
            releases_path=releases_path,
            envs_fd=envs_fd,
            releases_fd=releases_fd,
            desired_fd=desired_fd,
            expected_fd=expected_fd,
            lock_fd=lock_fd,
            desired_name=desired_name,
            expected_name=expected_name,
            desired_identity=desired_identity,
        )
    finally:
        if expected_fd >= 0:
            os.close(expected_fd)
        if desired_fd >= 0:
            os.close(desired_fd)
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        if releases_fd >= 0:
            os.close(releases_fd)
        os.close(envs_fd)


def _pointer_identity_at(
    parent_fd: int, name: str
) -> Optional[Tuple[int, int, int, int, int]]:
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _pointer_state_at(
    parent_fd: int, name: str
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int, int]]]:
    before = _pointer_identity_at(parent_fd, name)
    if before is None:
        return None, None
    if before[2] != stat.S_IFLNK:
        raise PointerConflictError("production wam-current path is not a symlink")
    try:
        text = os.readlink(name, dir_fd=parent_fd)
    except OSError as exc:
        raise PointerConflictError("production wam-current is unsafe") from exc
    after = _pointer_identity_at(parent_fd, name)
    if after != before:
        raise PointerConflictError("production wam-current identity changed while read")
    return text, before


def _production_environment_pointer_state(
    locked: _LockedProductionEnvironment,
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int, int]]]:
    text, identity = _pointer_state_at(locked.envs_fd, "wam-current")
    if text is None:
        return None, None
    parsed = PurePosixPath(text)
    if parsed.is_absolute() or len(parsed.parts) != 2 or parsed.parts[0] != "releases":
        raise PointerPolicyError("production wam-current symlink is not canonical")
    name = parsed.parts[1]
    if not _NAME.fullmatch(name):
        raise PointerPolicyError("production wam-current target is invalid")
    return name, identity


def _production_environment_link_name(
    locked: _LockedProductionEnvironment,
) -> Optional[str]:
    return _production_environment_pointer_state(locked)[0]


def _verify_environment_child(
    locked: _LockedProductionEnvironment,
    name: str,
    descriptor: Optional[int] = None,
) -> Dict[str, object]:
    owned_fd = -1
    if descriptor is None:
        owned_fd = _open_real_directory_at(
            locked.releases_fd, name, "active production environment"
        )
        descriptor = owned_fd
    try:
        _assert_child_maps_to_descriptor(
            locked.releases_fd,
            name,
            descriptor,
            "production environment release",
        )
        return _verify_complete_environment_fd(descriptor, name)
    finally:
        if owned_fd >= 0:
            os.close(owned_fd)


def _exchange_pointer_at(parent_fd: int, first: str, second: str) -> None:
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
        raise PointerConflictError("atomic production pointer exchange is unsupported")
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    result = operation(
        parent_fd,
        os.fsencode(first),
        parent_fd,
        os.fsencode(second),
        flag,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), second)


def _replace_production_pointer(
    locked: _LockedProductionEnvironment,
    target_name: str,
    *,
    expected_previous_name: Optional[str],
    expected_previous_identity: Optional[Tuple[int, int, int, int, int]],
) -> Tuple[int, int, int, int, int]:
    temporary = ".wam-current.production-switch-%s" % uuid.uuid4().hex
    os.symlink("releases/" + target_name, temporary, dir_fd=locked.envs_fd)
    temporary_text, temporary_identity = _pointer_state_at(
        locked.envs_fd, temporary
    )
    if temporary_text != "releases/" + target_name or temporary_identity is None:
        raise PointerConflictError("prepared production pointer is invalid")
    remove_temporary = True
    try:
        observed_name, observed_identity = _production_environment_pointer_state(
            locked
        )
        if (
            observed_name != expected_previous_name
            or observed_identity != expected_previous_identity
        ):
            raise PointerConflictError(
                "production pointer CAS identity changed before activation"
            )
        if expected_previous_name is None:
            try:
                _rename_noreplace_at(
                    locked.envs_fd,
                    temporary,
                    locked.envs_fd,
                    "wam-current",
                )
            except EnvironmentConflictError as exc:
                raise PointerConflictError(
                    "production absent-pointer CAS was raced"
                ) from exc
            remove_temporary = False
        else:
            _exchange_pointer_at(locked.envs_fd, temporary, "wam-current")
            displaced_name, displaced_identity = _pointer_state_at(
                locked.envs_fd, temporary
            )
            displaced_environment = (
                None
                if displaced_name is None
                else PurePosixPath(displaced_name).name
            )
            if (
                displaced_environment != expected_previous_name
                or displaced_identity is None
                or expected_previous_identity is None
                or displaced_identity[:3] != expected_previous_identity[:3]
            ):
                current_text, current_identity = _pointer_state_at(
                    locked.envs_fd, "wam-current"
                )
                if (
                    current_text != "releases/" + target_name
                    or current_identity is None
                    or current_identity[:3] != temporary_identity[:3]
                ):
                    remove_temporary = False
                    raise PointerConflictError(
                        "production pointer diverged during CAS restoration"
                    )
                _exchange_pointer_at(
                    locked.envs_fd, temporary, "wam-current"
                )
                os.fsync(locked.envs_fd)
                raise PointerConflictError(
                    "production pointer CAS identity changed during activation"
                )
            os.unlink(temporary, dir_fd=locked.envs_fd)
            remove_temporary = False
        os.fsync(locked.envs_fd)
        installed_text, installed_identity = _pointer_state_at(
            locked.envs_fd, "wam-current"
        )
        if installed_text != "releases/" + target_name or installed_identity is None:
            raise PointerConflictError("production pointer activation did not persist")
        return installed_identity
    except BaseException:
        raise
    finally:
        if remove_temporary:
            try:
                os.unlink(temporary, dir_fd=locked.envs_fd)
            except FileNotFoundError:
                pass


def _postverify_production_environment(
    locked: _LockedProductionEnvironment,
) -> Dict[str, object]:
    _assert_path_maps_to_descriptor(
        locked.envs_path, locked.envs_fd, "production environment root"
    )
    _assert_path_maps_to_descriptor(
        locked.releases_path,
        locked.releases_fd,
        "production environment releases root",
    )
    active_name, active_identity = _production_environment_pointer_state(locked)
    if (
        active_name != locked.desired_name
        or locked.activated_pointer_identity is not None
        and active_identity != locked.activated_pointer_identity
    ):
        raise EnvironmentVerificationError(
            "production environment pointer post-verification failed"
        )
    return _verify_environment_child(
        locked, locked.desired_name, locked.desired_fd
    )


def _remove_production_pointer_cas(
    locked: _LockedProductionEnvironment,
    expected_name: str,
    expected_identity: Tuple[int, int, int, int, int],
) -> None:
    observed_name, observed_identity = _production_environment_pointer_state(locked)
    if observed_name != expected_name or observed_identity != expected_identity:
        raise PointerConflictError(
            "production pointer removal CAS identity changed"
        )
    quarantine = ".wam-current.production-remove-%s" % uuid.uuid4().hex
    try:
        _rename_noreplace_at(
            locked.envs_fd,
            "wam-current",
            locked.envs_fd,
            quarantine,
        )
    except EnvironmentConflictError as exc:
        raise PointerConflictError("production pointer removal CAS failed") from exc
    moved_text, moved_identity = _pointer_state_at(locked.envs_fd, quarantine)
    moved_name = None if moved_text is None else PurePosixPath(moved_text).name
    if (
        moved_name != expected_name
        or moved_identity is None
        or moved_identity[:3] != expected_identity[:3]
    ):
        try:
            _rename_noreplace_at(
                locked.envs_fd,
                quarantine,
                locked.envs_fd,
                "wam-current",
            )
            os.fsync(locked.envs_fd)
        except EnvironmentConflictError:
            pass
        raise PointerConflictError(
            "production pointer changed during removal CAS"
        )
    os.unlink(quarantine, dir_fd=locked.envs_fd)
    os.fsync(locked.envs_fd)


def _rollback_production_environment(
    locked: _LockedProductionEnvironment,
    previous_name: Optional[str],
) -> None:
    active, active_identity = _production_environment_pointer_state(locked)
    activated_identity = locked.activated_pointer_identity
    if activated_identity is None:
        if (
            active == previous_name
            and active_identity == locked.previous_pointer_identity
        ):
            return
        raise PointerConflictError("production rollback lacks activated pointer identity")
    try:
        _assert_path_maps_to_descriptor(
            locked.envs_path, locked.envs_fd, "production environment root"
        )
        _assert_path_maps_to_descriptor(
            locked.releases_path,
            locked.releases_fd,
            "production environment releases root",
        )
    except PointerPolicyError:
        if active == locked.desired_name and active_identity == activated_identity:
            _remove_production_pointer_cas(
                locked, locked.desired_name, activated_identity
            )
            return
        raise PointerConflictError(
            "production rollback refused a divergent pointer after root change"
        )
    if active == previous_name:
        if active_identity == locked.previous_pointer_identity:
            return
        raise PointerConflictError(
            "production rollback previous pointer identity changed"
        )
    if active != locked.desired_name or active_identity != activated_identity:
        raise PointerConflictError("production rollback compare-and-swap mismatch")
    if previous_name is None:
        _remove_production_pointer_cas(
            locked, locked.desired_name, activated_identity
        )
    else:
        _replace_production_pointer(
            locked,
            previous_name,
            expected_previous_name=locked.desired_name,
            expected_previous_identity=activated_identity,
        )
    if _production_environment_link_name(locked) != previous_name:
        raise EnvironmentVerificationError("production rollback verification failed")
    if previous_name is not None:
        _verify_environment_child(
            locked,
            previous_name,
            locked.expected_fd if previous_name == locked.expected_name else None,
        )


def _validated_frozen_code_identity(
    frozen_release_id: str,
    frozen_payload_sha256: str,
    confirmed_release_id: str,
    confirmed_payload_sha256: str,
) -> Tuple[str, str]:
    match = _CODE_RELEASE_ID.fullmatch(frozen_release_id)
    if (
        match is None
        or not _HASH.fullmatch(frozen_payload_sha256)
        or match.group(1) != frozen_payload_sha256[:20]
    ):
        raise PointerPolicyError("frozen code release identity is invalid")
    if (
        confirmed_release_id != frozen_release_id
        or confirmed_payload_sha256 != frozen_payload_sha256
    ):
        raise PointerPolicyError("frozen code release double confirmation does not match")
    return frozen_release_id, frozen_payload_sha256


def _activate_production_environment_at_home(
    candidate: Union[str, Path],
    *,
    production_home: Path,
    expected_current: Optional[Union[str, Path]],
    frozen_code_release_id: str,
    frozen_code_payload_sha256: str,
    confirm_code_release_id: str,
    confirm_code_payload_sha256: str,
) -> Dict[str, object]:
    code_release_id, code_payload_sha256 = _validated_frozen_code_identity(
        frozen_code_release_id,
        frozen_code_payload_sha256,
        confirm_code_release_id,
        confirm_code_payload_sha256,
    )
    home = Path(os.path.abspath(str(production_home)))
    _reject_symlink_ancestors(home, PointerPolicyError, "production home")
    with _locked_production_code(
        home, code_release_id, code_payload_sha256
    ) as locked_code:
        with _locked_production_environment(
            home, candidate, expected_current
        ) as locked_environment:
            try:
                legacy_info = os.stat(
                    "wam", dir_fd=locked_environment.envs_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                legacy_info = None
            if legacy_info is not None and stat.S_ISLNK(legacy_info.st_mode):
                raise PointerPolicyError("legacy ~/envs/wam must not be a symlink")
            previous_name, previous_pointer_identity = (
                _production_environment_pointer_state(locked_environment)
            )
            locked_environment.previous_pointer_identity = previous_pointer_identity
            if previous_name != locked_environment.expected_name:
                raise PointerConflictError(
                    "production compare-and-swap mismatch: expected=%r actual=%r"
                    % (locked_environment.expected_name, previous_name)
                )
            if previous_name is not None:
                _verify_environment_child(
                    locked_environment,
                    previous_name,
                    locked_environment.expected_fd,
                )
            desired_identity = locked_environment.desired_identity
            request: Dict[str, object] = {
                "environment": {
                    "environment_id": desired_identity["environment_id"],
                    "payload_sha256": desired_identity["payload_sha256"],
                },
                "expected_current_environment_id": previous_name,
                "frozen_code": dict(locked_code.evidence),
                "operation": "activate_production_environment",
                "schema": PRODUCTION_ACTIVATION_REQUEST_SCHEMA,
            }
            changed = previous_name != locked_environment.desired_name
            if not changed:
                locked_environment.activated_pointer_identity = (
                    previous_pointer_identity
                )
            try:
                if changed:
                    locked_environment.activated_pointer_identity = (
                        _replace_production_pointer(
                            locked_environment,
                            locked_environment.desired_name,
                            expected_previous_name=previous_name,
                            expected_previous_identity=previous_pointer_identity,
                        )
                    )
                _postverify_production_environment(locked_environment)
                _reverify_locked_production_code(
                    locked_code, code_release_id, code_payload_sha256
                )
                _postverify_production_environment(locked_environment)
            except BaseException:
                if changed:
                    _rollback_production_environment(
                        locked_environment, previous_name
                    )
                raise
            return {
                "previous_environment_id": (
                    previous_name
                ),
                "request": request,
                "request_sha256": _sha256_bytes(canonical_json_bytes(request)),
                "schema": PRODUCTION_ACTIVATION_EVIDENCE_SCHEMA,
                "status": "active",
            }


def activate_production_environment(
    candidate: Union[str, Path],
    *,
    expected_current: Optional[Union[str, Path]],
    frozen_code_release_id: str,
    frozen_code_payload_sha256: str,
    confirm_code_release_id: str,
    confirm_code_payload_sha256: str,
) -> Dict[str, object]:
    """Activate only the fixed H200 production environment pointer."""
    return _activate_production_environment_at_home(
        candidate,
        production_home=H200_PRODUCTION_HOME,
        expected_current=expected_current,
        frozen_code_release_id=frozen_code_release_id,
        frozen_code_payload_sha256=frozen_code_payload_sha256,
        confirm_code_release_id=confirm_code_release_id,
        confirm_code_payload_sha256=confirm_code_payload_sha256,
    )
