"""Fail-closed linbox LeRobot converter environment and publication manager."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import pwd
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from typing import Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import uuid


LEROBOT_VERSION = "0.3.3"
PYAV_VERSION = "15.1.0"
ENVIRONMENT_COMPLETE = "ENVIRONMENT_COMPLETE.json"
_HASH = re.compile(r"(?:^|\s)--hash=sha256:([0-9a-f]{64})(?=\s|$)")
_STRICT_PIN = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.-]*)=="
    r"([A-Za-z0-9][A-Za-z0-9_.+!]*)"
    r"((?: --hash=sha256:[0-9a-f]{64})+)$"
)
_WHEEL_NAME = re.compile(
    r"^(?P<distribution>[A-Za-z0-9_]+)-"
    r"(?P<version>[A-Za-z0-9_.+!]+)"
    r"(?:-(?P<build>[0-9][A-Za-z0-9_.]*))?-"
    r"(?P<python>[A-Za-z0-9_.]+)-(?P<abi>[A-Za-z0-9_.]+)-"
    r"(?P<platform>[A-Za-z0-9_.]+)\.whl$"
)
_CONVERTER_RELATIVE = "auscultation/scripts/session_to_lerobot_v2.py"
_CONVERSION_OPERATION_NAME = "CONVERSION_OPERATION.json"
LINBOX_PRODUCTION_RELEASES_ROOT = Path("/home/robot/envs/releases")
LINBOX_PRODUCTION_CURRENT = Path("/home/robot/envs/lerobot-current")
LINBOX_PRODUCTION_CODE_ROOT = Path("/home/robot/Desktop/stethoscope")
_CODE_RELEASE_ID = re.compile(r"^code_release-[0-9a-f]{20}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ConverterPolicyError(RuntimeError):
    """A converter environment or conversion request is outside policy."""


class ConverterExecutionError(RuntimeError):
    """A validated converter subprocess or smoke gate failed."""


def _open_real_directory(path: Path, *, create: bool) -> int:
    pure = PurePosixPath(path.as_posix())
    if not pure.is_absolute() or pure.as_posix() != path.as_posix() or ".." in pure.parts:
        raise ConverterPolicyError("canonical converter directory path is invalid")
    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(pure.anchor, flags)
    try:
        for part in pure.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise ConverterPolicyError("canonical converter directory is missing")
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ConverterPolicyError(
                    "canonical converter directory has a symlink ancestor"
                ) from exc
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                raise ConverterPolicyError(
                    "canonical converter directory has a symlink ancestor"
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


@contextmanager
def _namespace_lock(namespace: Path, name: str):
    _ensure_real_directory(namespace)
    parent = _open_real_directory(namespace, create=False)
    descriptor = None
    try:
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ConverterPolicyError("converter namespace lock is unsafe")
        os.fsync(parent)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(parent)


@dataclass(frozen=True)
class LockAudit:
    lock_sha256: str
    packages: Dict[str, str]
    allowed_hashes: FrozenSet[str]
    package_hashes: Dict[str, FrozenSet[str]]


@dataclass(frozen=True)
class CandidatePlan:
    candidate: Path
    descriptor: Dict[str, object]
    environment_id: str
    lock_path: Path
    wheelhouse: Path


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


_canonical_json_bytes = canonical_json_bytes


def audit_lock(path: Union[str, Path]) -> LockAudit:
    lock = Path(path)
    parent = _open_real_directory(lock.parent, create=False)
    descriptor = None
    try:
        descriptor = os.open(
            lock.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ConverterPolicyError("converter lock is not a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            raw = handle.read()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConverterPolicyError("converter lock is missing or invalid") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)
    packages: Dict[str, str] = {}
    allowed_hashes = set()
    package_hashes: Dict[str, FrozenSet[str]] = {}
    logical = ""
    lines = []
    for physical in text.splitlines():
        stripped = physical.strip()
        if not stripped or stripped.startswith("#"):
            continue
        logical = (logical + " " + stripped).strip()
        if logical.endswith("\\"):
            logical = logical[:-1].rstrip()
            continue
        lines.append(logical)
        logical = ""
    if logical:
        raise ConverterPolicyError("converter lock has an incomplete continuation")
    for line in lines:
        match = _STRICT_PIN.fullmatch(line)
        if match is None:
            if "--hash=sha256:" not in line:
                raise ConverterPolicyError(
                    "every converter lock pin requires a SHA-256 hash"
                )
            raise ConverterPolicyError("converter lock must contain only exact package pins")
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        version = match.group(2)
        hashes = _HASH.findall(match.group(3))
        if not hashes:
            raise ConverterPolicyError("every converter lock pin requires a SHA-256 hash")
        if name in packages:
            raise ConverterPolicyError("converter lock contains a duplicate package pin")
        packages[name] = version
        allowed_hashes.update(hashes)
        package_hashes[name] = frozenset(hashes)
    if not packages:
        raise ConverterPolicyError("converter lock is unresolved or empty")
    if packages.get("lerobot") != LEROBOT_VERSION:
        raise ConverterPolicyError("converter lock must pin exactly lerobot==0.3.3")
    if packages.get("av") != PYAV_VERSION:
        raise ConverterPolicyError("converter lock must pin exactly av==15.1.0")
    return LockAudit(
        lock_sha256=hashlib.sha256(raw).hexdigest(),
        packages=packages,
        allowed_hashes=frozenset(allowed_hashes),
        package_hashes=package_hashes,
    )


def inventory_wheelhouse(
    path: Union[str, Path], audit: LockAudit
) -> Tuple[Dict[str, object], ...]:
    wheelhouse = Path(path)
    wheelhouse_descriptor = _open_real_directory(wheelhouse, create=False)
    os.close(wheelhouse_descriptor)
    rows: List[Dict[str, object]] = []
    seen = set()
    for wheel in sorted(wheelhouse.iterdir()):
        if wheel.is_symlink() or not wheel.is_file() or wheel.suffix != ".whl":
            raise ConverterPolicyError("converter wheelhouse contains a non-wheel entry")
        parsed = _WHEEL_NAME.fullmatch(wheel.name)
        if parsed is None:
            raise ConverterPolicyError("converter wheel filename is invalid")
        package = re.sub(
            r"[-_.]+", "-", parsed.group("distribution")
        ).lower()
        version = parsed.group("version").replace("_", "-")
        if package not in audit.packages or audit.packages[package] != version:
            raise ConverterPolicyError("converter wheel is not an exact locked package")
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if digest not in audit.package_hashes[package]:
            raise ConverterPolicyError("converter wheel hash is not authorized by the lock")
        if package in seen:
            raise ConverterPolicyError("converter wheelhouse has multiple wheels for one package")
        seen.add(package)
        rows.append({
            "name": wheel.name,
            "package": package,
            "sha256": digest,
            "size": wheel.stat().st_size,
            "version": version,
        })
    if seen != set(audit.packages):
        raise ConverterPolicyError("converter wheelhouse does not exactly cover the lock")
    return tuple(rows)


def _validate_runtime(runtime: Mapping[str, object]) -> Dict[str, str]:
    expected_fields = {
        "abi", "implementation", "machine", "python_version", "system"
    }
    if not isinstance(runtime, Mapping) or set(runtime) != expected_fields:
        raise ConverterPolicyError("runtime identity fields are invalid")
    value = {key: runtime[key] for key in sorted(expected_fields)}
    if (
        value["implementation"] != "CPython"
        or not isinstance(value["python_version"], str)
        or not value["python_version"].startswith("3.10.")
        or value["abi"] != "cp310"
        or value["system"] != "Linux"
        or value["machine"] != "x86_64"
    ):
        raise ConverterPolicyError(
            "converter candidate requires Python 3.10 Linux x86_64"
        )
    return value  # type: ignore[return-value]


def plan_candidate(
    *,
    host_role: str,
    lock_path: Union[str, Path],
    releases_root: Union[str, Path],
    runtime: Mapping[str, object],
    wheelhouse: Union[str, Path],
    external_binaries: Sequence[Mapping[str, object]],
) -> CandidatePlan:
    if host_role != "linbox":
        raise ConverterPolicyError("converter environments may be built only on linbox")
    audit = audit_lock(lock_path)
    wheels = inventory_wheelhouse(wheelhouse, audit)
    runtime_identity = _validate_runtime(runtime)
    binaries = []
    for row in external_binaries:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "version"}:
            raise ConverterPolicyError("external binary inventory is invalid")
        if not all(isinstance(row[key], str) and row[key] for key in row):
            raise ConverterPolicyError("external binary inventory is invalid")
        binaries.append({key: row[key] for key in sorted(row)})
    descriptor: Dict[str, object] = {
        "lock_sha256": audit.lock_sha256,
        "media_runtime": {
            "external_binaries": sorted(binaries, key=lambda row: row["path"]),
            "pyav_version": PYAV_VERSION,
        },
        "packages": dict(sorted(audit.packages.items())),
        "runtime": runtime_identity,
        "schema": "wam_linbox_converter_environment_descriptor_v1",
        "wheels": list(wheels),
    }
    _validate_environment_descriptor(descriptor)
    identity = hashlib.sha256(_canonical_json_bytes(descriptor)).hexdigest()
    environment_id = "lerobot-" + identity[:20]
    root = Path(releases_root)
    return CandidatePlan(
        candidate=root / environment_id,
        descriptor=descriptor,
        environment_id=environment_id,
        lock_path=Path(lock_path),
        wheelhouse=Path(wheelhouse),
    )


_RUNTIME_PROBE = r'''import json, platform, sys
value = {
    "abi": "cp%d%d" % sys.version_info[:2],
    "implementation": platform.python_implementation(),
    "machine": platform.machine(),
    "python_version": platform.python_version(),
    "system": platform.system(),
}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
'''


_SMOKE_SCRIPT = r'''import av, importlib.metadata, inspect, json, os, tempfile
from pathlib import Path
import numpy as np
_runtime_cache = tempfile.TemporaryDirectory(prefix="wam-lerobot-runtime-cache-")
os.environ["HOME"] = _runtime_cache.name
os.environ["HF_HOME"] = str(Path(_runtime_cache.name) / "huggingface")
os.environ["HF_DATASETS_CACHE"] = str(Path(_runtime_cache.name) / "huggingface" / "datasets")
os.environ["XDG_CACHE_HOME"] = str(Path(_runtime_cache.name) / "xdg")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

if importlib.metadata.version("lerobot") != "0.3.3":
    raise RuntimeError("unexpected LeRobot version")
if importlib.metadata.version("av") != "15.1.0":
    raise RuntimeError("unexpected PyAV version")
create_ok = all(name in inspect.signature(LeRobotDataset.create).parameters for name in (
    "repo_id", "fps", "features", "root", "robot_type", "use_videos"
))
add_frame_task = "task" in inspect.signature(LeRobotDataset.add_frame).parameters
save_episode_ok = callable(getattr(LeRobotDataset, "save_episode", None))
if not (create_ok and add_frame_task and save_episode_ok):
    raise RuntimeError("LeRobot 0.3.3 converter API contract is unavailable")
with tempfile.TemporaryDirectory(prefix="wam-lerobot-smoke-") as temporary:
    root = Path(temporary) / "dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint_1"]},
        "observation.images.top": {
            "dtype": "video", "shape": [64, 64, 3],
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (1,), "names": ["joint_1"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="wam/task4-smoke", fps=2, features=features, root=root,
        robot_type="wam_smoke", use_videos=True,
    )
    for value in (0.0, 1.0):
        frame = {
            "observation.state": np.asarray([value], dtype=np.float32),
            "observation.images.top": np.full((64, 64, 3), int(value), dtype=np.uint8),
            "action": np.asarray([value], dtype=np.float32),
        }
        dataset.add_frame(frame, task="smoke")
    dataset.save_episode()
    reopened = LeRobotDataset("wam/task4-smoke", root=root)
    frames = len(reopened)
    episodes = getattr(reopened, "num_episodes", None)
    if episodes is None:
        episodes = getattr(getattr(reopened, "meta", None), "total_episodes", None)
    media_files = len([path for path in root.rglob("*.mp4") if path.is_file()])
    if frames != 2 or episodes != 1 or media_files < 1:
        raise RuntimeError("tiny LeRobot dataset did not reopen exactly")
libraries = {
    name: ".".join(str(part) for part in version)
    for name, version in sorted(av.library_versions.items())
}
result = {
    "api": {
        "add_frame_task": add_frame_task, "create": create_ok,
        "reopen": True, "save_episode": save_episode_ok,
    },
    "lerobot_version": "0.3.3",
    "pyav_libraries": libraries,
    "pyav_version": "15.1.0",
    "schema": "wam_lerobot_033_smoke_v1",
    "tiny_dataset": {"episodes": episodes, "frames": frames, "media_files": media_files},
}
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
'''


_REOPEN_SCRIPT = r'''import json, sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
repo_id, raw_root = sys.argv[1:3]
root = Path(raw_root)
dataset = LeRobotDataset(repo_id, root=root)
frames = len(dataset)
episodes = getattr(dataset, "num_episodes", None)
if episodes is None:
    episodes = getattr(getattr(dataset, "meta", None), "total_episodes", None)
media_files = len([path for path in root.rglob("*.mp4") if path.is_file()])
if not isinstance(frames, int) or frames <= 0:
    raise RuntimeError("reopened dataset has no frames")
if not isinstance(episodes, int) or episodes <= 0:
    raise RuntimeError("reopened dataset has no episodes")
if media_files <= 0:
    raise RuntimeError("reopened dataset has no encoded media")
value = {
    "episodes": episodes, "frames": frames, "media_files": media_files,
    "repo_id": repo_id, "schema": "wam_lerobot_dataset_reopen_v1",
}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
'''


def _completed_process(
    runner, argv: Sequence[str], *, label: str,
    environment: Union[Mapping[str, str], None] = None,
) -> object:
    kwargs = {
        "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "check": False,
    }
    if environment is not None:
        kwargs["env"] = dict(environment)
    try:
        completed = runner(list(argv), **kwargs)
    except OSError as exc:
        raise ConverterExecutionError("%s could not start" % label) from exc
    if completed.returncode != 0:
        stderr = bytes(completed.stderr or b"").decode("utf-8", "replace").strip()
        raise ConverterExecutionError(
            "%s failed with exit %s: %s" % (label, completed.returncode, stderr)
        )
    return completed


def _isolated_candidate_environment(
    candidate: Path, external_binaries: Sequence[Mapping[str, object]]
) -> Dict[str, str]:
    paths = [str(candidate / "bin")]
    paths.extend(sorted({str(Path(row["path"]).parent) for row in external_binaries}))
    return {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.pathsep.join(paths),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }


def probe_runtime(base_python: Union[str, Path], *, runner=subprocess.run) -> Dict[str, str]:
    completed = _completed_process(
        runner, (str(base_python), "-c", _RUNTIME_PROBE), label="runtime probe"
    )
    raw = bytes(completed.stdout or b"")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConverterExecutionError("runtime probe returned invalid JSON") from exc
    if raw not in (canonical_json_bytes(value), canonical_json_bytes(value).rstrip(b"\n")):
        raise ConverterExecutionError("runtime probe returned non-canonical JSON")
    return _validate_runtime(value)


@contextmanager
def _environment_deploy_lock(releases_root: Path):
    with _namespace_lock(releases_root.parent, ".lerobot-deploy.lock"):
        yield


def _atomic_json(path: Path, value: object) -> None:
    _ensure_real_directory(path.parent)
    if os.path.lexists(str(path)):
        raise ConverterPolicyError("refusing to replace an existing immutable marker")
    descriptor, temporary = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
        parent = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_smoke(value: object) -> Dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "api", "lerobot_version", "pyav_libraries", "pyav_version", "schema",
        "tiny_dataset",
    }:
        raise ConverterExecutionError("converter smoke response fields are invalid")
    if (
        value.get("schema") != "wam_lerobot_033_smoke_v1"
        or value.get("lerobot_version") != LEROBOT_VERSION
        or value.get("pyav_version") != PYAV_VERSION
        or value.get("api") != {
            "add_frame_task": True, "create": True, "reopen": True,
            "save_episode": True,
        }
        or value.get("tiny_dataset") != {"episodes": 1, "frames": 2, "media_files": 1}
        or not isinstance(value.get("pyav_libraries"), dict)
        or not value["pyav_libraries"]
        or any(not isinstance(key, str) or not isinstance(item, str)
               for key, item in value["pyav_libraries"].items())
    ):
        raise ConverterExecutionError("converter smoke did not prove the exact API")
    return value


def _validate_environment_descriptor(value: object) -> Dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "lock_sha256", "media_runtime", "packages", "runtime", "schema", "wheels"
    }:
        raise ConverterPolicyError("converter candidate descriptor fields are invalid")
    if (
        value.get("schema") != "wam_linbox_converter_environment_descriptor_v1"
        or not isinstance(value.get("lock_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", value["lock_sha256"])
    ):
        raise ConverterPolicyError("converter candidate descriptor identity is invalid")
    _validate_runtime(value.get("runtime", {}))
    packages = value.get("packages")
    if (
        not isinstance(packages, dict) or not packages
        or any(not isinstance(key, str) or not isinstance(item, str)
               for key, item in packages.items())
        or packages.get("lerobot") != LEROBOT_VERSION
        or packages.get("av") != PYAV_VERSION
    ):
        raise ConverterPolicyError("converter candidate descriptor packages are invalid")
    media = value.get("media_runtime")
    if not isinstance(media, dict) or set(media) != {
        "external_binaries", "pyav_version"
    } or media.get("pyav_version") != PYAV_VERSION or not isinstance(
        media.get("external_binaries"), list
    ):
        raise ConverterPolicyError("converter candidate media descriptor is invalid")
    external_paths = set()
    for row in media["external_binaries"]:
        if (
            not isinstance(row, dict) or set(row) != {"path", "sha256", "version"}
            or not isinstance(row.get("path"), str) or not Path(row["path"]).is_absolute()
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or not isinstance(row.get("version"), str) or not row["version"]
            or row["path"] in external_paths
        ):
            raise ConverterPolicyError("converter candidate external binary is invalid")
        external_paths.add(row["path"])
    wheels = value.get("wheels")
    if not isinstance(wheels, list):
        raise ConverterPolicyError("converter candidate wheel descriptor is invalid")
    wheel_packages = set()
    for row in wheels:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "package", "sha256", "size", "version"}
            or not isinstance(row.get("name"), str)
            or _WHEEL_NAME.fullmatch(row["name"]) is None
            or not isinstance(row.get("package"), str)
            or row["package"] in wheel_packages
            or packages.get(row["package"]) != row.get("version")
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or type(row.get("size")) is not int or row["size"] <= 0
        ):
            raise ConverterPolicyError("converter candidate wheel descriptor is invalid")
        wheel_packages.add(row["package"])
    if wheel_packages != set(packages):
        raise ConverterPolicyError(
            "converter candidate wheels do not exactly cover its packages"
        )
    return value


def _installed_tree_inventory(candidate: Path) -> Tuple[Dict[str, object], ...]:
    rows: List[Dict[str, object]] = []
    for directory, dirnames, filenames in os.walk(str(candidate), followlinks=False):
        base = Path(directory)
        for name in dirnames:
            path = base / name
            if path.is_symlink() or not path.is_dir():
                raise ConverterPolicyError("converter environment tree contains a symlink")
        for name in filenames:
            path = base / name
            if path.parent == candidate and path.name == ENVIRONMENT_COMPLETE:
                continue
            if path.is_symlink() or not path.is_file():
                raise ConverterPolicyError("converter environment tree contains an unsafe file")
            info = path.stat()
            rows.append({
                "mode": format(info.st_mode & 0o7777, "04o"),
                "path": path.relative_to(candidate).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": info.st_size,
            })
    if not rows:
        raise ConverterPolicyError("converter environment tree is empty")
    return tuple(sorted(rows, key=lambda row: row["path"]))


def _installed_tree_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(rows))).hexdigest()


def _relocate_candidate_tree(candidate: Path, final: Path) -> None:
    source_bytes = os.fsencode(str(candidate))
    destination_bytes = os.fsencode(str(final))
    for directory, dirnames, filenames in os.walk(str(candidate), followlinks=False):
        base = Path(directory)
        for name in dirnames:
            if (base / name).is_symlink():
                raise ConverterPolicyError("converter environment tree contains a symlink")
        for name in filenames:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ConverterPolicyError("converter environment tree contains an unsafe file")
            raw = path.read_bytes()
            if source_bytes not in raw:
                continue
            if b"\x00" in raw:
                raise ConverterPolicyError(
                    "converter environment binary embeds the staging path"
                )
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ConverterPolicyError(
                    "converter environment non-text file embeds the staging path"
                ) from exc
            with path.open("wb") as handle:
                handle.write(raw.replace(source_bytes, destination_bytes))
                handle.flush()
                os.fsync(handle.fileno())


def _remove_rebuildable_bytecode(candidate: Path) -> None:
    for directory, dirnames, filenames in os.walk(
        str(candidate), topdown=False, followlinks=False
    ):
        base = Path(directory)
        for name in dirnames:
            path = base / name
            if not os.path.lexists(str(path)):
                continue
            if path.is_symlink() or not path.is_dir():
                raise ConverterPolicyError(
                    "converter environment tree contains an unsafe directory"
                )
        for name in filenames:
            if not name.endswith((".pyc", ".pyo")):
                continue
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ConverterPolicyError(
                    "converter environment bytecode cache is unsafe"
                )
            path.unlink()
        if base.name == "__pycache__":
            if any(base.iterdir()):
                raise ConverterPolicyError(
                    "converter __pycache__ contains a non-bytecode artifact"
                )
            base.rmdir()


def _remove_standard_venv_symlink(candidate: Path) -> None:
    lib64 = candidate / "lib64"
    if not os.path.lexists(str(lib64)):
        return
    info = lib64.lstat()
    if (
        not stat.S_ISLNK(info.st_mode)
        or os.readlink(str(lib64)) != "lib"
        or not (candidate / "lib").is_dir()
        or (candidate / "lib").is_symlink()
    ):
        raise ConverterPolicyError(
            "converter venv lib64 is not the expected local lib symlink"
        )
    lib64.unlink()


def build_candidate(
    plan: CandidatePlan,
    *,
    base_python: Union[str, Path],
    host_role: str,
    runner=subprocess.run,
) -> Path:
    if host_role != "linbox":
        raise ConverterPolicyError("converter environments may be built only on linbox")
    runtime = probe_runtime(base_python, runner=runner)
    fresh = plan_candidate(
        host_role=host_role,
        lock_path=plan.lock_path,
        releases_root=plan.candidate.parent,
        runtime=runtime,
        wheelhouse=plan.wheelhouse,
        external_binaries=plan.descriptor["media_runtime"]["external_binaries"],
    )
    if fresh.environment_id != plan.environment_id or fresh.descriptor != plan.descriptor:
        raise ConverterPolicyError("converter candidate inputs changed after planning")
    releases_root = plan.candidate.parent
    with _environment_deploy_lock(releases_root):
        if os.path.lexists(str(plan.candidate)):
            verify_candidate(plan.candidate)
            return plan.candidate
        _ensure_real_directory(releases_root)
        staging_root = releases_root.parent / ".staging"
        _ensure_real_directory(staging_root)
        candidate = staging_root / (
            ".%s-%s" % (plan.environment_id, uuid.uuid4().hex)
        )
        input_context = _build_input_snapshot(
            plan, staging_root / (".inputs-%s" % uuid.uuid4().hex)
        )
        inputs = input_context.__enter__()
        try:
            _completed_process(
                runner,
                (str(base_python), "-m", "venv", "--copies", str(candidate)),
                label="converter venv creation",
            )
            candidate_python = candidate / "bin" / "python"
            if not candidate_python.is_file():
                raise ConverterExecutionError("converter candidate Python is missing")
            _completed_process(
                runner,
                (
                    str(candidate_python), "-m", "pip", "install", "--no-index",
                    "--find-links", str(inputs["snapshot_wheelhouse"]),
                    "--require-hashes", "--no-compile", "-r",
                    str(inputs["snapshot_lock"]),
                ),
                label="converter hash-locked install",
            )
            completed = _completed_process(
                runner, (str(candidate_python), "-c", _SMOKE_SCRIPT),
                label="converter smoke",
                environment=_isolated_candidate_environment(
                    candidate,
                    plan.descriptor["media_runtime"]["external_binaries"],
                ),
            )
            raw = bytes(completed.stdout or b"")
            try:
                smoke = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ConverterExecutionError("converter smoke returned invalid JSON") from exc
            if raw not in (canonical_json_bytes(smoke), canonical_json_bytes(smoke).rstrip(b"\n")):
                raise ConverterExecutionError("converter smoke returned non-canonical JSON")
            smoke = _validate_smoke(smoke)
            _verify_build_input_snapshot(inputs, plan)
            _remove_standard_venv_symlink(candidate)
            _remove_rebuildable_bytecode(candidate)
            _relocate_candidate_tree(candidate, plan.candidate)
            descriptor_sha256 = hashlib.sha256(
                canonical_json_bytes(plan.descriptor)
            ).hexdigest()
            installed_files = _installed_tree_inventory(candidate)
            marker = {
                "descriptor": plan.descriptor,
                "descriptor_sha256": descriptor_sha256,
                "environment_id": plan.environment_id,
                "installed_files": list(installed_files),
                "installed_tree_sha256": _installed_tree_sha256(installed_files),
                "schema": "wam_linbox_converter_environment_complete_v1",
                "smoke": smoke,
            }
            _atomic_json(candidate / ENVIRONMENT_COMPLETE, marker)
            verify_candidate(candidate, allow_staging=True)
            import artifact_contract

            try:
                artifact_contract._rename_noreplace(candidate, plan.candidate)
                parent = os.open(str(releases_root), os.O_RDONLY)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
            except artifact_contract.ConflictError:
                verify_candidate(plan.candidate)
                return plan.candidate
            verify_candidate(plan.candidate)
            return plan.candidate
        finally:
            try:
                input_context.__exit__(*sys.exc_info())
            finally:
                if os.path.lexists(str(candidate)):
                    shutil.rmtree(str(candidate), ignore_errors=True)


def verify_candidate(
    candidate_path: Union[str, Path], *, allow_staging: bool = False
) -> Dict[str, object]:
    candidate = Path(candidate_path)
    if (
        candidate.is_symlink() or not candidate.is_dir()
        or not allow_staging and (
            candidate.parent.name != "releases"
            or candidate.parent.parent.name != "envs"
            or not re.fullmatch(r"lerobot-[0-9a-f]{20}", candidate.name)
        )
    ):
        raise ConverterPolicyError("converter candidate is not an immutable release path")
    marker_path = candidate / ENVIRONMENT_COMPLETE
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ConverterPolicyError("converter candidate completion marker is missing")
    raw = marker_path.read_bytes()
    try:
        marker = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConverterPolicyError("converter candidate marker is invalid") from exc
    if not isinstance(marker, dict) or raw != canonical_json_bytes(marker) or set(marker) != {
        "descriptor", "descriptor_sha256", "environment_id", "installed_files",
        "installed_tree_sha256", "schema", "smoke",
    }:
        raise ConverterPolicyError("converter candidate marker is not canonical")
    descriptor = _validate_environment_descriptor(marker.get("descriptor"))
    digest = hashlib.sha256(canonical_json_bytes(descriptor)).hexdigest()
    if (
        marker.get("schema") != "wam_linbox_converter_environment_complete_v1"
        or marker.get("descriptor_sha256") != digest
        or marker.get("environment_id") != "lerobot-" + digest[:20]
        or not allow_staging and marker.get("environment_id") != candidate.name
    ):
        raise ConverterPolicyError("converter candidate identity does not match its descriptor")
    _validate_smoke(marker.get("smoke"))
    if not (candidate / "bin" / "python").is_file():
        raise ConverterPolicyError("converter candidate Python is missing")
    installed_files = marker.get("installed_files")
    if not isinstance(installed_files, list) or installed_files != list(
        _installed_tree_inventory(candidate)
    ) or marker.get("installed_tree_sha256") != _installed_tree_sha256(
        installed_files
    ):
        raise ConverterPolicyError("converter environment tree does not match its marker")
    return marker


def _canary_pointer(
    current_link: Union[str, Path], canary_root: Union[str, Path]
) -> Tuple[Path, Path]:
    current = Path(current_link)
    root = Path(canary_root)
    if root.name != ".lerobot-canary" or current.parent != root or current.name != "current":
        raise ConverterPolicyError("converter pointer is not the isolated canary pointer")
    _ensure_real_directory(root)
    if os.path.lexists(str(current)) and not current.is_symlink():
        raise ConverterPolicyError("converter canary current is not a symlink")
    return current, root


def _switch_pointer(current: Path, target: Union[str, Path]) -> None:
    temporary = current.parent / (".current-%s" % uuid.uuid4().hex)
    temporary.symlink_to(target)
    os.replace(str(temporary), str(current))
    descriptor = os.open(str(current.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _pointer_lock(canary_root: Path):
    with _namespace_lock(canary_root.parent, ".lerobot-pointer.lock"):
        yield


def activate_canary(
    candidate_path: Union[str, Path], *, current_link: Union[str, Path],
    canary_root: Union[str, Path], expected_current: Union[str, None],
) -> Union[str, None]:
    candidate = Path(candidate_path)
    verify_candidate(candidate)
    root = Path(canary_root)
    with _pointer_lock(root):
        current, _root = _canary_pointer(current_link, root)
        actual = os.readlink(str(current)) if current.is_symlink() else None
        if actual != expected_current:
            raise ConverterPolicyError("converter canary compare-and-swap mismatch")
        target = os.path.relpath(str(candidate), str(current.parent))
        _switch_pointer(current, target)
        if current.resolve() != candidate.resolve():
            raise ConverterPolicyError("converter canary activation resolved incorrectly")
        return actual


def rollback_canary(
    *, current_link: Union[str, Path], canary_root: Union[str, Path],
    expected_current: Union[str, None], previous: Union[str, None],
) -> None:
    root = Path(canary_root)
    with _pointer_lock(root):
        current, _root = _canary_pointer(current_link, root)
        actual = os.readlink(str(current)) if current.is_symlink() else None
        if actual != expected_current:
            raise ConverterPolicyError("converter canary compare-and-swap mismatch")
        if previous is None:
            current.unlink(missing_ok=True)
            descriptor = os.open(str(current.parent), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            relative = Path(previous)
            if relative.is_absolute() or ".." in relative.parts:
                raise ConverterPolicyError("converter canary rollback target is unsafe")
            _switch_pointer(current, previous)


def _confirmed_code_identity(
    frozen_release_id: str,
    frozen_payload_sha256: str,
    confirm_release_id: str,
    confirm_payload_sha256: str,
) -> Tuple[str, str]:
    if (
        not isinstance(frozen_release_id, str)
        or not _CODE_RELEASE_ID.fullmatch(frozen_release_id)
        or not isinstance(frozen_payload_sha256, str)
        or not _HEX_SHA256.fullmatch(frozen_payload_sha256)
        or frozen_release_id != "code_release-" + frozen_payload_sha256[:20]
    ):
        raise ConverterPolicyError("frozen code release identity is invalid")
    if (
        confirm_release_id != frozen_release_id
        or confirm_payload_sha256 != frozen_payload_sha256
    ):
        raise ConverterPolicyError("frozen code double confirmation does not match")
    return frozen_release_id, frozen_payload_sha256


@dataclass
class _LockedProductionCode:
    root: Path
    root_fd: int
    release: Path
    release_fd: int
    lock_fd: int
    current_text: str
    evidence: Dict[str, str]


def _code_evidence(
    code_root: Path, release_id: str, payload_sha256: str,
) -> Dict[str, str]:
    import artifact_contract

    release = code_root / ".wam-releases" / release_id
    envelope = artifact_contract.verify_release(release)
    payload = envelope.get("payload")
    if (
        envelope.get("release_id") != release_id
        or envelope.get("payload_sha256") != payload_sha256
        or not isinstance(payload, dict)
        or payload.get("artifact_kind") != "code_release"
        or payload.get("source_role") != "mac"
    ):
        raise ConverterPolicyError("active frozen code release identity is invalid")
    _active_converter_entrypoint(code_root.parents[1], release_id)
    return {
        "payload_sha256": payload_sha256,
        "release_id": release_id,
    }


@contextmanager
def _locked_production_code(
    code_root: Path, release_id: str, payload_sha256: str,
) -> Iterator[_LockedProductionCode]:
    root_fd = _open_real_directory(code_root, create=False)
    release_fd = -1
    lock_fd = -1
    try:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        lock_fd = os.open(".wam-production.lock", flags, 0o600, dir_fd=root_fd)
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise ConverterPolicyError("production code lock is unsafe")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ConverterPolicyError("production code lock is already held") from exc
        current_info = os.stat(".wam-current", dir_fd=root_fd, follow_symlinks=False)
        current_text = os.readlink(".wam-current", dir_fd=root_fd)
        if (
            not stat.S_ISLNK(current_info.st_mode)
            or current_text != ".wam-releases/%s" % release_id
        ):
            raise ConverterPolicyError("active frozen code pointer is not exact")
        releases_fd = os.open(
            ".wam-releases",
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        try:
            release_fd = os.open(
                release_id,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=releases_fd,
            )
        finally:
            os.close(releases_fd)
        evidence = _code_evidence(code_root, release_id, payload_sha256)
        yield _LockedProductionCode(
            root=code_root,
            root_fd=root_fd,
            release=code_root / ".wam-releases" / release_id,
            release_fd=release_fd,
            lock_fd=lock_fd,
            current_text=current_text,
            evidence=evidence,
        )
    finally:
        if release_fd >= 0:
            os.close(release_fd)
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(root_fd)


def _reverify_production_code(
    locked: _LockedProductionCode, release_id: str, payload_sha256: str,
) -> None:
    current_info = os.stat(
        ".wam-current", dir_fd=locked.root_fd, follow_symlinks=False,
    )
    if (
        not stat.S_ISLNK(current_info.st_mode)
        or os.readlink(".wam-current", dir_fd=locked.root_fd) != locked.current_text
        or _code_evidence(locked.root, release_id, payload_sha256) != locked.evidence
    ):
        raise ConverterPolicyError("active frozen code changed during activation")
    releases_fd = os.open(
        ".wam-releases",
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=locked.root_fd,
    )
    try:
        observed = os.stat(
            release_id, dir_fd=releases_fd, follow_symlinks=False,
        )
    finally:
        os.close(releases_fd)
    opened = os.fstat(locked.release_fd)
    if (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino):
        raise ConverterPolicyError("active frozen code release inode changed")


def _path_identity_at(directory_fd: int, name: str) -> Tuple[int, int, int, int, int]:
    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _pointer_identity_document(identity: Tuple[int, ...]) -> Dict[str, int]:
    if len(identity) < 3 or identity[2] != stat.S_IFLNK:
        raise ConverterPolicyError("production converter pointer identity is not a symlink")
    return {"device": int(identity[0]), "inode": int(identity[1])}


def _pointer_identity_tuple(value: object, label: str) -> Tuple[int, int, int]:
    if not isinstance(value, dict) or set(value) != {"device", "inode"}:
        raise ConverterPolicyError("%s fields are invalid" % label)
    if any(type(value[key]) is not int or value[key] < 0 for key in value):
        raise ConverterPolicyError("%s values are invalid" % label)
    return value["device"], value["inode"], stat.S_IFLNK


def _same_pointer_identity(observed: Tuple[int, ...], expected: Tuple[int, ...]) -> bool:
    return observed[:3] == expected[:3]


def _read_environment_pointer_at(
    envs_fd: int, current_name: str,
) -> Tuple[Optional[str], Optional[str], Optional[Tuple[int, int, int, int, int]]]:
    try:
        identity = _path_identity_at(envs_fd, current_name)
    except FileNotFoundError:
        return None, None, None
    if identity[2] != stat.S_IFLNK:
        raise ConverterPolicyError("production converter current is not a symlink")
    text = os.readlink(current_name, dir_fd=envs_fd)
    prefix = "releases/"
    environment_id = text[len(prefix):] if text.startswith(prefix) else ""
    if text != prefix + environment_id or not re.fullmatch(
        r"lerobot-[0-9a-f]{20}", environment_id
    ):
        raise ConverterPolicyError("production converter current is not canonical")
    return environment_id, text, identity


def _current_environment_id_at(envs_fd: int, current_name: str) -> Optional[str]:
    environment_id, _text, _identity = _read_environment_pointer_at(
        envs_fd, current_name,
    )
    return environment_id


def _rename_noreplace_at(
    directory_fd: int,
    source_name: str,
    destination_name: str,
    *,
    expected_identity: Tuple[int, ...],
) -> None:
    if not _same_pointer_identity(
        _path_identity_at(directory_fd, source_name), expected_identity,
    ):
        raise ConverterPolicyError("production converter prepared pointer inode changed")
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        flag = 1
    elif sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        flag = 0x00000004
    else:
        raise ConverterPolicyError("atomic production converter no-replace is unsupported")
    if function is None:
        raise ConverterPolicyError("atomic production converter no-replace is unavailable")
    function.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        directory_fd, os.fsencode(source_name),
        directory_fd, os.fsencode(destination_name), flag,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise ConverterPolicyError("production converter compare-and-swap mismatch")
        raise OSError(error, os.strerror(error), destination_name)


def _exchange_at(directory_fd: int, first: str, second: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        flag = 2
    elif sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        flag = 0x00000002
    else:
        raise ConverterPolicyError("atomic production converter exchange is unsupported")
    if function is None:
        raise ConverterPolicyError("atomic production converter exchange is unavailable")
    function.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        directory_fd, os.fsencode(first),
        directory_fd, os.fsencode(second), flag,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), second)


def _prepare_environment_pointer_at(
    envs_fd: int, current_name: str, environment_id: str,
) -> Tuple[str, str, Tuple[int, int, int, int, int]]:
    temporary = ".%s.activate-%s" % (current_name, uuid.uuid4().hex)
    target = "releases/%s" % environment_id
    try:
        os.symlink(target, temporary, dir_fd=envs_fd)
    except FileExistsError as exc:
        raise ConverterPolicyError("prepared production converter pointer already exists") from exc
    identity = _path_identity_at(envs_fd, temporary)
    if (
        identity[2] != stat.S_IFLNK
        or os.readlink(temporary, dir_fd=envs_fd) != target
    ):
        raise ConverterPolicyError("prepared production converter pointer is invalid")
    os.fsync(envs_fd)
    return temporary, target, identity


def _remove_owned_pointer_at(
    envs_fd: int,
    name: str,
    *,
    target: str,
    identity: Tuple[int, ...],
) -> None:
    try:
        observed = _path_identity_at(envs_fd, name)
    except FileNotFoundError:
        return
    if (
        not _same_pointer_identity(observed, identity)
        or os.readlink(name, dir_fd=envs_fd) != target
    ):
        raise ConverterPolicyError("production converter owned pointer inode changed")
    os.unlink(name, dir_fd=envs_fd)
    os.fsync(envs_fd)


def _switch_environment_at(envs_fd: int, current_name: str, environment_id: str) -> None:
    journal = _validate_production_journal(
        _read_json_at(envs_fd, _PRODUCTION_JOURNAL)
    )
    request = journal["payload"]["request"]
    environment = request["environment"]
    prepared = request["prepared_pointer"]
    previous = request["previous_pointer"]
    if environment["environment_id"] != environment_id:
        raise ConverterPolicyError("production converter journal request does not match switch")
    prepared_identity = _pointer_identity_tuple(
        prepared["identity"], "prepared production pointer identity",
    )
    temporary = prepared["name"]
    target = prepared["target"]
    observed_prepared = _path_identity_at(envs_fd, temporary)
    if (
        not _same_pointer_identity(observed_prepared, prepared_identity)
        or os.readlink(temporary, dir_fd=envs_fd) != target
    ):
        raise ConverterPolicyError("prepared production converter pointer inode changed")
    if previous is None:
        _rename_noreplace_at(
            envs_fd, temporary, current_name,
            expected_identity=prepared_identity,
        )
    else:
        previous_identity = _pointer_identity_tuple(
            previous["identity"], "previous production pointer identity",
        )
        current_id, current_target, current_identity = _read_environment_pointer_at(
            envs_fd, current_name,
        )
        if (
            current_id != request["previous_environment_id"]
            or current_target != previous["target"]
            or current_identity is None
            or not _same_pointer_identity(current_identity, previous_identity)
        ):
            raise ConverterPolicyError(
                "production converter compare-and-swap pointer inode mismatch"
            )
        _exchange_at(envs_fd, temporary, current_name)
        observed_previous = _path_identity_at(envs_fd, temporary)
        observed_current = _path_identity_at(envs_fd, current_name)
        mismatch = (
            not _same_pointer_identity(observed_previous, previous_identity)
            or os.readlink(temporary, dir_fd=envs_fd) != previous["target"]
            or not _same_pointer_identity(observed_current, prepared_identity)
            or os.readlink(current_name, dir_fd=envs_fd) != target
        )
        if mismatch:
            if (
                _same_pointer_identity(observed_current, prepared_identity)
                and os.readlink(current_name, dir_fd=envs_fd) == target
            ):
                _exchange_at(envs_fd, temporary, current_name)
                os.fsync(envs_fd)
            raise ConverterPolicyError("production converter compare-and-swap inode mismatch")
    os.fsync(envs_fd)
    installed = _path_identity_at(envs_fd, current_name)
    if (
        not _same_pointer_identity(installed, prepared_identity)
        or os.readlink(current_name, dir_fd=envs_fd) != target
    ):
        raise ConverterPolicyError("production converter activation pointer did not persist")


def _restore_environment_at(
    envs_fd: int, current_name: str, previous_id: Optional[str], desired_id: str,
) -> None:
    actual = _current_environment_id_at(envs_fd, current_name)
    if actual == previous_id:
        return
    if actual != desired_id:
        raise ConverterPolicyError("production converter rollback CAS mismatch")
    if previous_id is None:
        os.unlink(current_name, dir_fd=envs_fd)
        os.fsync(envs_fd)
    else:
        temporary = ".%s.rollback-%s" % (current_name, uuid.uuid4().hex)
        previous_target = "releases/%s" % previous_id
        os.symlink(previous_target, temporary, dir_fd=envs_fd)
        temporary_identity = _path_identity_at(envs_fd, temporary)
        try:
            _exchange_at(envs_fd, temporary, current_name)
            replaced_id, replaced_target, _replaced_identity = (
                _read_environment_pointer_at(envs_fd, temporary)
            )
            if replaced_id != desired_id or replaced_target != "releases/%s" % desired_id:
                _exchange_at(envs_fd, temporary, current_name)
                os.fsync(envs_fd)
                raise ConverterPolicyError("production converter rollback CAS mismatch")
            os.unlink(temporary, dir_fd=envs_fd)
            installed_id, installed_target, installed_identity = (
                _read_environment_pointer_at(envs_fd, current_name)
            )
            if (
                installed_id != previous_id
                or installed_target != previous_target
                or installed_identity is None
                or not _same_pointer_identity(installed_identity, temporary_identity)
            ):
                raise ConverterPolicyError("production converter rollback did not persist")
            os.fsync(envs_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=envs_fd)
            except FileNotFoundError:
                pass


def _production_checkpoint(_stage: str) -> None:
    """Test hook at durable converter production transaction boundaries."""


_PRODUCTION_JOURNAL = ".lerobot-production-activation.json"


def _write_json_at(directory_fd: int, name: str, value: Mapping[str, object]) -> None:
    temporary = ".%s-%s" % (name, uuid.uuid4().hex)
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        body = canonical_json_bytes(dict(value))
        os.write(descriptor, body)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(
            temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _read_json_at(directory_fd: int, name: str) -> Dict[str, object]:
    raw, _info = _read_regular_at(directory_fd, name)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConverterPolicyError("production activation journal is invalid") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ConverterPolicyError("production activation journal is not canonical")
    return value


def _remove_at(directory_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        return
    os.fsync(directory_fd)


def _production_journal(
    *,
    desired_id: str,
    previous_id: Optional[str],
    previous_target: Optional[str],
    previous_identity: Optional[Tuple[int, ...]],
    prepared_name: str,
    prepared_target: str,
    prepared_identity: Tuple[int, ...],
    marker: Mapping[str, object],
    code_release_id: str,
    code_payload_sha256: str,
    stage: str,
) -> Dict[str, object]:
    previous_pointer = None
    if previous_id is not None:
        if previous_target is None or previous_identity is None:
            raise ConverterPolicyError("previous production pointer evidence is missing")
        previous_pointer = {
            "identity": _pointer_identity_document(previous_identity),
            "target": previous_target,
        }
    elif previous_target is not None or previous_identity is not None:
        raise ConverterPolicyError("absent production pointer has unexpected evidence")
    request = {
        "code": {
            "payload_sha256": code_payload_sha256,
            "release_id": code_release_id,
        },
        "environment": {
            "descriptor_sha256": marker["descriptor_sha256"],
            "environment_id": desired_id,
        },
        "prepared_pointer": {
            "identity": _pointer_identity_document(prepared_identity),
            "name": prepared_name,
            "target": prepared_target,
        },
        "previous_environment_id": previous_id,
        "previous_pointer": previous_pointer,
        "schema": "wam_linbox_converter_production_request_v1",
    }
    payload = {
        "request": request,
        "request_sha256": hashlib.sha256(canonical_json_bytes(request)).hexdigest(),
        "schema": "wam_linbox_converter_production_transaction_v2",
        "stage": stage,
    }
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        "schema": "wam_linbox_converter_production_transaction_envelope_v1",
    }


def _production_journal_stage(
    journal: Mapping[str, object], stage: str,
) -> Dict[str, object]:
    checked = _validate_production_journal(dict(journal))
    payload = dict(checked["payload"])
    payload["stage"] = stage
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        "schema": checked["schema"],
    }


def _validate_production_journal(value: object) -> Dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "payload", "payload_sha256", "schema",
    }:
        raise ConverterPolicyError("production activation journal fields are invalid")
    payload = value.get("payload")
    if not isinstance(payload, dict) or set(payload) != {
        "request", "request_sha256", "schema", "stage",
    }:
        raise ConverterPolicyError("production activation journal payload fields are invalid")
    request = payload.get("request")
    if not isinstance(request, dict) or set(request) != {
        "code", "environment", "prepared_pointer", "previous_environment_id",
        "previous_pointer", "schema",
    }:
        raise ConverterPolicyError("production activation journal request fields are invalid")
    code = request.get("code")
    environment = request.get("environment")
    prepared = request.get("prepared_pointer")
    previous_id = request.get("previous_environment_id")
    previous = request.get("previous_pointer")
    if (
        not isinstance(code, dict)
        or set(code) != {"payload_sha256", "release_id"}
        or not isinstance(environment, dict)
        or set(environment) != {"descriptor_sha256", "environment_id"}
        or not isinstance(prepared, dict)
        or set(prepared) != {"identity", "name", "target"}
    ):
        raise ConverterPolicyError("production activation journal identity fields are invalid")
    desired_id = environment.get("environment_id")
    if (
        value.get("schema")
        != "wam_linbox_converter_production_transaction_envelope_v1"
        or payload.get("schema") != "wam_linbox_converter_production_transaction_v2"
        or request.get("schema") != "wam_linbox_converter_production_request_v1"
        or payload.get("stage") not in {"prepared", "switched", "committed"}
        or not isinstance(desired_id, str)
        or not re.fullmatch(r"lerobot-[0-9a-f]{20}", desired_id)
        or (
            previous_id is not None
            and (
                not isinstance(previous_id, str)
                or not re.fullmatch(r"lerobot-[0-9a-f]{20}", previous_id)
            )
        )
        or not isinstance(environment.get("descriptor_sha256"), str)
        or not _HEX_SHA256.fullmatch(environment["descriptor_sha256"])
        or not isinstance(code.get("release_id"), str)
        or not _CODE_RELEASE_ID.fullmatch(code["release_id"])
        or not isinstance(code.get("payload_sha256"), str)
        or not _HEX_SHA256.fullmatch(code["payload_sha256"])
        or code["release_id"] != "code_release-" + code["payload_sha256"][:20]
        or not isinstance(prepared.get("name"), str)
        or not re.fullmatch(r"\.lerobot-current\.activate-[0-9a-f]{32}", prepared["name"])
        or prepared.get("target") != "releases/%s" % desired_id
    ):
        raise ConverterPolicyError("production activation journal values are invalid")
    _pointer_identity_tuple(
        prepared.get("identity"), "prepared production pointer identity",
    )
    if previous_id is None:
        if previous is not None:
            raise ConverterPolicyError(
                "production activation journal previous pointer is inconsistent"
            )
    else:
        if (
            not isinstance(previous, dict)
            or set(previous) != {"identity", "target"}
            or previous.get("target") != "releases/%s" % previous_id
        ):
            raise ConverterPolicyError(
                "production activation journal previous pointer is inconsistent"
            )
        _pointer_identity_tuple(
            previous.get("identity"), "previous production pointer identity",
        )
    request_digest = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
    payload_digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if (
        payload.get("request_sha256") != request_digest
        or value.get("payload_sha256") != payload_digest
    ):
        raise ConverterPolicyError("production activation journal integrity hash mismatch")
    return value


def _journal_request(journal: Mapping[str, object]) -> Dict[str, object]:
    return dict(journal["payload"]["request"])


def _cleanup_exchanged_previous_at(
    envs_fd: int, journal: Mapping[str, object],
) -> None:
    request = _journal_request(journal)
    previous = request["previous_pointer"]
    if previous is None:
        return
    prepared = request["prepared_pointer"]
    _remove_owned_pointer_at(
        envs_fd,
        prepared["name"],
        target=previous["target"],
        identity=_pointer_identity_tuple(
            previous["identity"], "previous production pointer identity",
        ),
    )


def _rollback_journal_pointer_at(
    envs_fd: int, current_name: str, journal: Mapping[str, object],
) -> None:
    request = _journal_request(journal)
    environment = request["environment"]
    desired_id = environment["environment_id"]
    desired_target = "releases/%s" % desired_id
    prepared = request["prepared_pointer"]
    prepared_name = prepared["name"]
    prepared_identity = _pointer_identity_tuple(
        prepared["identity"], "prepared production pointer identity",
    )
    previous_id = request["previous_environment_id"]
    previous = request["previous_pointer"]
    current_id, current_target, current_identity = _read_environment_pointer_at(
        envs_fd, current_name,
    )
    current_is_desired = (
        current_id == desired_id
        and current_target == desired_target
        and current_identity is not None
        and _same_pointer_identity(current_identity, prepared_identity)
    )
    if previous is None:
        if current_is_desired:
            os.unlink(current_name, dir_fd=envs_fd)
            os.fsync(envs_fd)
        elif current_id is not None:
            raise ConverterPolicyError("production converter rollback pointer inode changed")
        try:
            _remove_owned_pointer_at(
                envs_fd,
                prepared_name,
                target=prepared["target"],
                identity=prepared_identity,
            )
        except FileNotFoundError:
            pass
        return

    previous_identity = _pointer_identity_tuple(
        previous["identity"], "previous production pointer identity",
    )
    current_is_previous = (
        current_id == previous_id
        and current_target == previous["target"]
        and current_identity is not None
        and _same_pointer_identity(current_identity, previous_identity)
    )
    if current_is_previous:
        _remove_owned_pointer_at(
            envs_fd,
            prepared_name,
            target=prepared["target"],
            identity=prepared_identity,
        )
        return
    if not current_is_desired:
        raise ConverterPolicyError("production converter rollback pointer inode changed")
    try:
        held_target = os.readlink(prepared_name, dir_fd=envs_fd)
        held_identity = _path_identity_at(envs_fd, prepared_name)
    except FileNotFoundError:
        _restore_environment_at(
            envs_fd, current_name, previous_id, desired_id,
        )
    else:
        if (
            held_target != previous["target"]
            or not _same_pointer_identity(held_identity, previous_identity)
        ):
            raise ConverterPolicyError("production converter rollback saved pointer changed")
        _exchange_at(envs_fd, prepared_name, current_name)
        os.fsync(envs_fd)
        restored_id, restored_target, restored_identity = (
            _read_environment_pointer_at(envs_fd, current_name)
        )
        if (
            restored_id != previous_id
            or restored_target != previous["target"]
            or restored_identity is None
            or not _same_pointer_identity(restored_identity, previous_identity)
        ):
            raise ConverterPolicyError("production converter rollback did not restore previous inode")
        _remove_owned_pointer_at(
            envs_fd,
            prepared_name,
            target=prepared["target"],
            identity=prepared_identity,
        )


def _remove_prepared_if_owned_at(
    envs_fd: int, journal: Mapping[str, object],
) -> None:
    prepared = _journal_request(journal)["prepared_pointer"]
    _remove_owned_pointer_at(
        envs_fd,
        prepared["name"],
        target=prepared["target"],
        identity=_pointer_identity_tuple(
            prepared["identity"], "prepared production pointer identity",
        ),
    )


def _activate_production_environment_at_paths(
    candidate_path: Union[str, Path],
    *,
    releases_root: Union[str, Path],
    current_link: Union[str, Path],
    code_root: Union[str, Path],
    expected_current: Optional[str],
    frozen_code_release_id: str,
    frozen_code_payload_sha256: str,
    confirm_code_release_id: str,
    confirm_code_payload_sha256: str,
) -> Dict[str, object]:
    code_release_id, code_payload_sha256 = _confirmed_code_identity(
        frozen_code_release_id,
        frozen_code_payload_sha256,
        confirm_code_release_id,
        confirm_code_payload_sha256,
    )
    releases = Path(releases_root)
    current = Path(current_link)
    candidate = Path(candidate_path)
    if (
        current != releases.parent / "lerobot-current"
        or releases.name != "releases"
        or releases.parent.name != "envs"
        or candidate.parent != releases
        or not re.fullmatch(r"lerobot-[0-9a-f]{20}", candidate.name)
    ):
        raise ConverterPolicyError("production converter path is not canonical")
    marker = verify_candidate(candidate)
    with _locked_production_code(
        Path(code_root), code_release_id, code_payload_sha256,
    ) as locked_code:
        with _environment_deploy_lock(releases):
            envs_fd = _open_real_directory(releases.parent, create=False)
            releases_fd = -1
            candidate_fd = -1
            previous_fd = -1
            switched = False
            journal_written = False
            previous_id: Optional[str] = None
            previous_target: Optional[str] = None
            previous_identity: Optional[Tuple[int, ...]] = None
            prepared_name: Optional[str] = None
            prepared_target: Optional[str] = None
            prepared_identity: Optional[Tuple[int, ...]] = None
            journal: Optional[Dict[str, object]] = None
            try:
                releases_fd = os.open(
                    releases.name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=envs_fd,
                )
                candidate_fd = os.open(
                    candidate.name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=releases_fd,
                )
                opened = os.fstat(candidate_fd)
                observed = os.stat(
                    candidate.name, dir_fd=releases_fd, follow_symlinks=False,
                )
                if (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino):
                    raise ConverterPolicyError("production candidate inode changed")
                verify_candidate(candidate)
                previous_id, previous_target, previous_identity = (
                    _read_environment_pointer_at(envs_fd, current.name)
                )
                if previous_id != expected_current:
                    raise ConverterPolicyError("production converter compare-and-swap mismatch")
                if previous_id is not None:
                    verify_candidate(releases / previous_id)
                    previous_fd = os.open(
                        previous_id,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=releases_fd,
                    )
                    previous_opened = os.fstat(previous_fd)
                    previous_observed = os.stat(
                        previous_id, dir_fd=releases_fd, follow_symlinks=False,
                    )
                    if (previous_opened.st_dev, previous_opened.st_ino) != (
                        previous_observed.st_dev, previous_observed.st_ino,
                    ):
                        raise ConverterPolicyError(
                            "previous converter environment inode changed"
                        )
                try:
                    os.stat(_PRODUCTION_JOURNAL, dir_fd=envs_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ConverterPolicyError(
                        "production converter activation requires reconciliation"
                    )
                prepared_name, prepared_target, prepared_identity = (
                    _prepare_environment_pointer_at(
                        envs_fd, current.name, candidate.name,
                    )
                )
                journal = _production_journal(
                    desired_id=candidate.name,
                    previous_id=previous_id,
                    previous_target=previous_target,
                    previous_identity=previous_identity,
                    prepared_name=prepared_name,
                    prepared_target=prepared_target,
                    prepared_identity=prepared_identity,
                    marker=marker,
                    code_release_id=code_release_id,
                    code_payload_sha256=code_payload_sha256,
                    stage="prepared",
                )
                _write_json_at(envs_fd, _PRODUCTION_JOURNAL, journal)
                journal_written = True
                _switch_environment_at(envs_fd, current.name, candidate.name)
                current_id, current_target, current_identity = (
                    _read_environment_pointer_at(envs_fd, current.name)
                )
                if (
                    current_id != candidate.name
                    or current_target != prepared_target
                    or current_identity is None
                    or not _same_pointer_identity(current_identity, prepared_identity)
                ):
                    raise ConverterPolicyError(
                        "production converter activation pointer inode changed"
                    )
                switched = True
                journal = _production_journal_stage(journal, "switched")
                _write_json_at(envs_fd, _PRODUCTION_JOURNAL, journal)
                _cleanup_exchanged_previous_at(envs_fd, journal)
                _production_checkpoint("after_switch")
                if _current_environment_id_at(envs_fd, current.name) != candidate.name:
                    raise ConverterPolicyError("production converter post-verification failed")
                root_observed = releases.parent.lstat()
                releases_observed = releases.lstat()
                candidate_observed = os.stat(
                    candidate.name, dir_fd=releases_fd, follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(root_observed.st_mode)
                    or not stat.S_ISDIR(releases_observed.st_mode)
                    or (root_observed.st_dev, root_observed.st_ino)
                    != (os.fstat(envs_fd).st_dev, os.fstat(envs_fd).st_ino)
                    or (releases_observed.st_dev, releases_observed.st_ino)
                    != (os.fstat(releases_fd).st_dev, os.fstat(releases_fd).st_ino)
                    or (candidate_observed.st_dev, candidate_observed.st_ino)
                    != (opened.st_dev, opened.st_ino)
                ):
                    raise ConverterPolicyError("production converter root changed during activation")
                final_marker = verify_candidate(candidate)
                if final_marker != marker:
                    raise ConverterPolicyError("production candidate completion changed")
                _reverify_production_code(
                    locked_code, code_release_id, code_payload_sha256,
                )
                _production_checkpoint("after_postverify")
                journal = _production_journal_stage(journal, "committed")
                _write_json_at(envs_fd, _PRODUCTION_JOURNAL, journal)
                _production_checkpoint("after_commit_record")
                _remove_at(envs_fd, _PRODUCTION_JOURNAL)
                journal_written = False
                return {
                    "descriptor_sha256": marker["descriptor_sha256"],
                    "environment_id": candidate.name,
                    "frozen_code": dict(locked_code.evidence),
                    "previous_environment_id": previous_id,
                    "schema": "wam_linbox_converter_production_activation_v1",
                }
            except BaseException:
                if journal_written and journal is not None:
                    request = _journal_request(journal)
                    desired_pointer = request["prepared_pointer"]
                    try:
                        actual_id, actual_target, actual_identity = (
                            _read_environment_pointer_at(envs_fd, current.name)
                        )
                    except BaseException:
                        raise
                    installed_identity = _pointer_identity_tuple(
                        desired_pointer["identity"],
                        "prepared production pointer identity",
                    )
                    installed = (
                        actual_id == candidate.name
                        and actual_target == desired_pointer["target"]
                        and actual_identity is not None
                        and _same_pointer_identity(actual_identity, installed_identity)
                    )
                else:
                    installed = False
                if switched or installed:
                    if previous_id is not None:
                        previous_observed = os.stat(
                            previous_id,
                            dir_fd=releases_fd,
                            follow_symlinks=False,
                        )
                        previous_opened = os.fstat(previous_fd)
                        if (previous_observed.st_dev, previous_observed.st_ino) != (
                            previous_opened.st_dev, previous_opened.st_ino,
                        ):
                            raise ConverterPolicyError(
                                "previous converter environment changed during rollback"
                            )
                        verify_candidate(releases / previous_id)
                    _rollback_journal_pointer_at(envs_fd, current.name, journal)
                    restored_id = _current_environment_id_at(envs_fd, current.name)
                    if restored_id != previous_id:
                        raise ConverterPolicyError(
                            "production converter rollback did not resolve to previous release"
                        )
                    if previous_id is not None:
                        verify_candidate(releases / previous_id)
                elif journal_written and journal is not None:
                    _remove_prepared_if_owned_at(envs_fd, journal)
                if journal_written:
                    _remove_at(envs_fd, _PRODUCTION_JOURNAL)
                raise
            finally:
                if (
                    not journal_written
                    and prepared_name is not None
                    and prepared_target is not None
                    and prepared_identity is not None
                ):
                    try:
                        _remove_owned_pointer_at(
                            envs_fd,
                            prepared_name,
                            target=prepared_target,
                            identity=prepared_identity,
                        )
                    except FileNotFoundError:
                        pass
                if candidate_fd >= 0:
                    os.close(candidate_fd)
                if previous_fd >= 0:
                    os.close(previous_fd)
                if releases_fd >= 0:
                    os.close(releases_fd)
                os.close(envs_fd)


def activate_production_environment(
    environment_id: str,
    *,
    expected_current: Optional[str],
    frozen_code_release_id: str,
    frozen_code_payload_sha256: str,
    confirm_code_release_id: str,
    confirm_code_payload_sha256: str,
) -> Dict[str, object]:
    """Activate one converter environment only in the live linbox namespace."""
    return _activate_production_environment_at_paths(
        LINBOX_PRODUCTION_RELEASES_ROOT / environment_id,
        releases_root=LINBOX_PRODUCTION_RELEASES_ROOT,
        current_link=LINBOX_PRODUCTION_CURRENT,
        code_root=LINBOX_PRODUCTION_CODE_ROOT,
        expected_current=expected_current,
        frozen_code_release_id=frozen_code_release_id,
        frozen_code_payload_sha256=frozen_code_payload_sha256,
        confirm_code_release_id=confirm_code_release_id,
        confirm_code_payload_sha256=confirm_code_payload_sha256,
    )


def _reconcile_production_environment_at_paths(
    *,
    releases_root: Union[str, Path],
    current_link: Union[str, Path],
    code_root: Union[str, Path],
    frozen_code_release_id: str,
    frozen_code_payload_sha256: str,
    confirm_code_release_id: str,
    confirm_code_payload_sha256: str,
) -> Dict[str, object]:
    code_release_id, code_payload_sha256 = _confirmed_code_identity(
        frozen_code_release_id,
        frozen_code_payload_sha256,
        confirm_code_release_id,
        confirm_code_payload_sha256,
    )
    releases = Path(releases_root)
    current = Path(current_link)
    if (
        current != releases.parent / "lerobot-current"
        or releases.name != "releases"
        or releases.parent.name != "envs"
    ):
        raise ConverterPolicyError("production converter path is not canonical")
    with _locked_production_code(
        Path(code_root), code_release_id, code_payload_sha256,
    ) as locked_code:
        with _environment_deploy_lock(releases):
            envs_fd = _open_real_directory(releases.parent, create=False)
            releases_fd = -1
            desired_fd = -1
            previous_fd = -1
            try:
                releases_info = os.stat(
                    releases.name, dir_fd=envs_fd, follow_symlinks=False,
                )
                if not stat.S_ISDIR(releases_info.st_mode):
                    raise ConverterPolicyError(
                        "production converter releases root is a symlink or unsafe"
                    )
                releases_fd = os.open(
                    releases.name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=envs_fd,
                )
                try:
                    journal = _validate_production_journal(
                        _read_json_at(envs_fd, _PRODUCTION_JOURNAL)
                    )
                except FileNotFoundError:
                    return {
                        "schema": "wam_linbox_converter_production_reconcile_v1",
                        "status": "clean",
                    }
                request = _journal_request(journal)
                code = request["code"]
                environment = request["environment"]
                if (
                    code["release_id"] != code_release_id
                    or code["payload_sha256"] != code_payload_sha256
                ):
                    raise ConverterPolicyError(
                        "production journal frozen code identity does not match confirmation"
                    )
                desired_id = str(environment["environment_id"])
                previous = request["previous_environment_id"]
                desired = releases / desired_id
                desired_fd = os.open(
                    desired_id,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=releases_fd,
                )
                desired_opened = os.fstat(desired_fd)
                desired_observed = os.stat(
                    desired_id, dir_fd=releases_fd, follow_symlinks=False,
                )
                if (desired_opened.st_dev, desired_opened.st_ino) != (
                    desired_observed.st_dev, desired_observed.st_ino,
                ):
                    raise ConverterPolicyError(
                        "production journal candidate inode changed"
                    )
                marker = verify_candidate(desired)
                if marker["descriptor_sha256"] != environment["descriptor_sha256"]:
                    raise ConverterPolicyError(
                        "production journal candidate completion does not match"
                    )
                if previous is not None:
                    previous_path = releases / previous
                    previous_fd = os.open(
                        previous,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=releases_fd,
                    )
                    previous_opened = os.fstat(previous_fd)
                    previous_observed = os.stat(
                        previous, dir_fd=releases_fd, follow_symlinks=False,
                    )
                    if (previous_opened.st_dev, previous_opened.st_ino) != (
                        previous_observed.st_dev, previous_observed.st_ino,
                    ):
                        raise ConverterPolicyError(
                            "previous converter environment inode changed"
                        )
                    verify_candidate(previous_path)
                stage = journal["payload"]["stage"]
                if stage == "committed":
                    current_id, current_target, current_identity = (
                        _read_environment_pointer_at(envs_fd, current.name)
                    )
                    prepared = request["prepared_pointer"]
                    if (
                        current_id != desired_id
                        or current_target != prepared["target"]
                        or current_identity is None
                        or not _same_pointer_identity(
                            current_identity,
                            _pointer_identity_tuple(
                                prepared["identity"],
                                "prepared production pointer identity",
                            ),
                        )
                    ):
                        raise ConverterPolicyError(
                            "committed production converter pointer is inconsistent"
                        )
                    _reverify_production_code(
                        locked_code, code_release_id, code_payload_sha256,
                    )
                    _cleanup_exchanged_previous_at(envs_fd, journal)
                    status = "committed"
                else:
                    _rollback_journal_pointer_at(envs_fd, current.name, journal)
                    restored_id, restored_target, _restored_identity = (
                        _read_environment_pointer_at(envs_fd, current.name)
                    )
                    if restored_id != previous or (
                        previous is not None
                        and restored_target != "releases/%s" % previous
                    ):
                        raise ConverterPolicyError(
                            "production converter rollback did not resolve to previous release"
                        )
                    if previous is not None:
                        verify_candidate(releases / previous)
                        observed_after = os.stat(
                            previous, dir_fd=releases_fd, follow_symlinks=False,
                        )
                        opened_previous = os.fstat(previous_fd)
                        if (observed_after.st_dev, observed_after.st_ino) != (
                            opened_previous.st_dev, opened_previous.st_ino,
                        ):
                            raise ConverterPolicyError(
                                "previous converter environment changed during rollback"
                            )
                    status = "rolled_back"
                root_observed = releases.parent.lstat()
                releases_observed = releases.lstat()
                if (
                    not stat.S_ISDIR(root_observed.st_mode)
                    or not stat.S_ISDIR(releases_observed.st_mode)
                    or (root_observed.st_dev, root_observed.st_ino)
                    != (os.fstat(envs_fd).st_dev, os.fstat(envs_fd).st_ino)
                    or (releases_observed.st_dev, releases_observed.st_ino)
                    != (os.fstat(releases_fd).st_dev, os.fstat(releases_fd).st_ino)
                ):
                    raise ConverterPolicyError(
                        "production converter root changed during reconciliation"
                    )
                _remove_at(envs_fd, _PRODUCTION_JOURNAL)
                return {
                    "environment_id": desired_id,
                    "schema": "wam_linbox_converter_production_reconcile_v1",
                    "status": status,
                }
            finally:
                if desired_fd >= 0:
                    os.close(desired_fd)
                if previous_fd >= 0:
                    os.close(previous_fd)
                if releases_fd >= 0:
                    os.close(releases_fd)
                os.close(envs_fd)


def reconcile_production_environment(
    *,
    frozen_code_release_id: str,
    frozen_code_payload_sha256: str,
    confirm_code_release_id: str,
    confirm_code_payload_sha256: str,
) -> Dict[str, object]:
    """Reconcile only the canonical live linbox converter namespace."""
    return _reconcile_production_environment_at_paths(
        releases_root=LINBOX_PRODUCTION_RELEASES_ROOT,
        current_link=LINBOX_PRODUCTION_CURRENT,
        code_root=LINBOX_PRODUCTION_CODE_ROOT,
        frozen_code_release_id=frozen_code_release_id,
        frozen_code_payload_sha256=frozen_code_payload_sha256,
        confirm_code_release_id=confirm_code_release_id,
        confirm_code_payload_sha256=confirm_code_payload_sha256,
    )


def verify_sealed_session(
    session_path: Union[str, Path], *, sessions_root: Union[str, Path], host_role: str,
    trusted_finalization_registry: Union[str, Path],
    trusted_snapshot_root: Union[str, Path],
    trusted_session_id: Optional[str] = None,
) -> Dict[str, object]:
    if host_role != "linbox":
        raise ConverterPolicyError("session conversion is allowed only on linbox")
    session = Path(session_path)
    root = Path(sessions_root)
    if (
        session.is_symlink() or not session.is_dir()
        or session.parent.as_posix() != root.as_posix()
    ):
        raise ConverterPolicyError("sealed session is outside the linbox sessions root")
    try:
        import artifact_contract
        import remote_agent

        selector = trusted_session_id if trusted_session_id is not None else session.name
        trusted_record = artifact_contract.load_trusted_training_finalization_record(
            trusted_finalization_registry, session_id=selector,
        )
        verified = remote_agent.validate_raw_session(
            session,
            allowed_sessions_root=root,
            trusted_finalization_registry=trusted_finalization_registry,
            trusted_snapshot_root=trusted_snapshot_root,
            expected_finalization_sha256=trusted_record["artifact_sha256"][
                "finalization"
            ],
        )
    except Exception as exc:
        if isinstance(exc, ConverterPolicyError):
            raise
        raise ConverterPolicyError("sealed session failed full artifact rehash") from exc
    raw_manifest_sha256 = str(verified["manifest_sha256"])
    raw_finalization_sha256 = str(verified["finalization_sha256"])
    raw_authenticity_sha256 = str(verified["training_authenticity_sha256"])
    raw_payload_sha256 = str(verified["raw_payload_sha256"])
    return {
        "promotion_contract_version": "wam_promotion_v2",
        "raw_finalization_sha256": raw_finalization_sha256,
        "raw_manifest_sha256": raw_manifest_sha256,
        "raw_payload_sha256": raw_payload_sha256,
        "raw_release_id": "raw_session-" + raw_manifest_sha256[:20],
        "raw_session_id": verified["session_id"],
        "raw_training_authenticity_sha256": raw_authenticity_sha256,
        "verified_snapshot_path": verified["verified_snapshot_path"],
        "verified_components": verified["verified_components"],
        "verified_raw_files": verified["verified_raw_files"],
    }


def _parse_canonical_response(completed: object, *, label: str) -> Dict[str, object]:
    raw = bytes(completed.stdout or b"")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConverterExecutionError("%s returned invalid JSON" % label) from exc
    if not isinstance(value, dict) or raw not in (
        canonical_json_bytes(value), canonical_json_bytes(value).rstrip(b"\n")
    ):
        raise ConverterExecutionError("%s returned non-canonical JSON" % label)
    return value


def _dataset_inventory(dataset_root: Path) -> List[Dict[str, object]]:
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ConverterExecutionError("converter did not create a real dataset root")
    rows: List[Dict[str, object]] = []
    for directory, dirnames, filenames in os.walk(str(dataset_root), followlinks=False):
        base = Path(directory)
        for name in dirnames:
            if (base / name).is_symlink():
                raise ConverterExecutionError("converted dataset contains a symlink")
        for name in filenames:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ConverterExecutionError("converted dataset contains an unsafe file")
            relative = path.relative_to(dataset_root).as_posix()
            rows.append({
                "path": "dataset/" + relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            })
    if not rows:
        raise ConverterExecutionError("converted dataset is empty")
    return sorted(rows, key=lambda row: row["path"])


def _verify_external_binaries(marker: Mapping[str, object]) -> None:
    descriptor = marker["descriptor"]
    rows = descriptor["media_runtime"]["external_binaries"]
    for row in rows:
        path = Path(row["path"])
        if (
            not path.is_absolute() or path.is_symlink() or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
        ):
            raise ConverterPolicyError("converter external binary inventory changed")


def _active_converter_entrypoint(
    home: Path, converter_code_release_id: str
) -> Path:
    if not re.fullmatch(
        r"code_release-[0-9a-f]{20}", converter_code_release_id
    ):
        raise ConverterPolicyError("converter code release id is invalid")
    root = home / "Desktop" / "stethoscope"
    descriptor = _open_real_directory(root, create=False)
    os.close(descriptor)
    releases = root / ".wam-releases"
    current = root / ".wam-current"
    if not current.is_symlink():
        raise ConverterPolicyError("active Task2 code pointer is missing")
    raw_target = os.readlink(str(current))
    target = Path(raw_target)
    if target.is_absolute() or ".." in target.parts:
        raise ConverterPolicyError("active Task2 code pointer is unsafe")
    release = (current.parent / target).resolve()
    if (
        release.parent.resolve() != releases.resolve()
        or release.name != converter_code_release_id
        or current.resolve() != release
        or release.is_symlink()
    ):
        raise ConverterPolicyError("active Task2 code release identity is invalid")
    compatibility = root / "auscultation"
    if (
        not compatibility.is_symlink()
        or os.readlink(compatibility) != ".wam-current/auscultation"
        or compatibility.resolve(strict=True)
        != (release / "auscultation").resolve(strict=True)
    ):
        raise ConverterPolicyError("active Task2 code compatibility path is invalid")
    import artifact_contract

    envelope = artifact_contract.verify_release(release)
    payload = envelope["payload"]
    if (
        envelope.get("release_id") != converter_code_release_id
        or payload.get("artifact_kind") != "code_release"
        or payload.get("source_role") != "mac"
    ):
        raise ConverterPolicyError("active Task2 code manifest is invalid")
    completion = json.loads(
        (release / artifact_contract.COMPLETE_NAME).read_text(encoding="utf-8")
    )
    if completion.get("target_role") != "linbox":
        raise ConverterPolicyError("active Task2 code release targets the wrong host")
    paths = {
        row.get("path") for row in payload.get("files", [])
        if isinstance(row, dict)
    }
    if _CONVERTER_RELATIVE not in paths:
        raise ConverterPolicyError("active Task2 release lacks the converter entry point")
    script = release / Path(*_CONVERTER_RELATIVE.split("/"))
    if script.is_symlink() or not script.is_file():
        raise ConverterPolicyError("active Task2 converter entry point is unsafe")
    return script


def _read_regular_at(directory_fd: int, name: str) -> Tuple[bytes, os.stat_result]:
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ConverterPolicyError("converter code release contains an unsafe file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        )
        if identity(before) != identity(after):
            raise ConverterPolicyError("converter code release changed while snapshotting")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def _snapshot_directory_fd(
    source_fd: int, destination: Path, *, prefix: str = "",
) -> Tuple[Dict[str, object], ...]:
    destination.mkdir(mode=0o700, exist_ok=False)
    rows: List[Dict[str, object]] = []
    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for name in sorted(os.listdir(source_fd)):
        info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        relative = "%s/%s" % (prefix, name) if prefix else name
        target = destination / name
        if stat.S_ISDIR(info.st_mode):
            child = os.open(name, flags, dir_fd=source_fd)
            try:
                rows.extend(_snapshot_directory_fd(child, target, prefix=relative))
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ConverterPolicyError("converter code release contains an unsafe entry")
        raw, opened = _read_regular_at(source_fd, name)
        with target.open("xb") as handle:
            handle.write(raw)
            os.fchmod(handle.fileno(), stat.S_IMODE(opened.st_mode))
            handle.flush()
            os.fsync(handle.fileno())
        rows.append({
            "mode": format(stat.S_IMODE(opened.st_mode), "04o"),
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        })
    descriptor = os.open(str(destination), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return tuple(sorted(rows, key=lambda row: row["path"]))


def _directory_fd_inventory(
    source_fd: int, *, prefix: str = "",
) -> Tuple[Dict[str, object], ...]:
    rows: List[Dict[str, object]] = []
    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for name in sorted(os.listdir(source_fd)):
        info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        relative = "%s/%s" % (prefix, name) if prefix else name
        if stat.S_ISDIR(info.st_mode):
            child = os.open(name, flags, dir_fd=source_fd)
            try:
                rows.extend(_directory_fd_inventory(child, prefix=relative))
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ConverterPolicyError("converter code release contains an unsafe entry")
        raw, opened = _read_regular_at(source_fd, name)
        rows.append({
            "mode": format(stat.S_IMODE(opened.st_mode), "04o"),
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        })
    return tuple(sorted(rows, key=lambda row: row["path"]))


def _read_open_regular_descriptor(descriptor: int) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ConverterPolicyError("converter build input is not a regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise ConverterPolicyError("converter build inputs changed while reading")
    return b"".join(chunks)


def _wheelhouse_fd_inventory(
    descriptor: int, expected: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, object], ...]:
    by_name = {str(row["name"]): row for row in expected}
    if set(os.listdir(descriptor)) != set(by_name):
        raise ConverterPolicyError("converter build inputs changed in the wheelhouse")
    rows = []
    for name in sorted(by_name):
        raw, info = _read_regular_at(descriptor, name)
        row = {
            "name": name,
            "package": by_name[name]["package"],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": info.st_size,
            "version": by_name[name]["version"],
        }
        if row != dict(by_name[name]):
            raise ConverterPolicyError("converter build inputs changed in the wheelhouse")
        rows.append(row)
    return tuple(rows)


@contextmanager
def _build_input_snapshot(plan: CandidatePlan, snapshot_root: Path):
    lock_parent = _open_real_directory(plan.lock_path.parent, create=False)
    lock_fd = None
    wheelhouse_fd = None
    try:
        lock_fd = os.open(
            plan.lock_path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=lock_parent,
        )
        lock_info = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_info.st_mode):
            raise ConverterPolicyError("converter lock is not a regular file")
        wheelhouse_fd = _open_real_directory(plan.wheelhouse, create=False)
        wheelhouse_info = os.fstat(wheelhouse_fd)
        raw_lock = _read_open_regular_descriptor(lock_fd)
        if hashlib.sha256(raw_lock).hexdigest() != plan.descriptor["lock_sha256"]:
            raise ConverterPolicyError("converter build inputs changed before snapshot")
        wheels = _wheelhouse_fd_inventory(
            wheelhouse_fd, plan.descriptor["wheels"]
        )

        snapshot_root.mkdir(mode=0o700, exist_ok=False)
        snapshot_lock = snapshot_root / "requirements.lock.txt"
        with snapshot_lock.open("xb") as handle:
            handle.write(raw_lock)
            os.fchmod(handle.fileno(), 0o400)
            handle.flush()
            os.fsync(handle.fileno())
        snapshot_wheelhouse = snapshot_root / "wheelhouse"
        snapshot_wheelhouse.mkdir(mode=0o700)
        for row in wheels:
            raw, _info = _read_regular_at(wheelhouse_fd, row["name"])
            target = snapshot_wheelhouse / row["name"]
            with target.open("xb") as handle:
                handle.write(raw)
                os.fchmod(handle.fileno(), 0o400)
                handle.flush()
                os.fsync(handle.fileno())
        snapshot_wheelhouse.chmod(0o500)
        for directory in (snapshot_wheelhouse, snapshot_root):
            directory_fd = os.open(str(directory), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        snapshot_audit = audit_lock(snapshot_lock)
        snapshot_wheels = inventory_wheelhouse(snapshot_wheelhouse, snapshot_audit)
        if (
            snapshot_audit.lock_sha256 != plan.descriptor["lock_sha256"]
            or list(snapshot_wheels) != plan.descriptor["wheels"]
        ):
            raise ConverterPolicyError("converter build input snapshot is inconsistent")
        yield {
            "lock_fd": lock_fd,
            "lock_info": lock_info,
            "lock_parent_fd": lock_parent,
            "raw_lock": raw_lock,
            "snapshot_lock": snapshot_lock,
            "snapshot_wheelhouse": snapshot_wheelhouse,
            "wheelhouse_fd": wheelhouse_fd,
            "wheelhouse_info": wheelhouse_info,
        }
    finally:
        if wheelhouse_fd is not None:
            os.close(wheelhouse_fd)
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(lock_parent)
        shutil.rmtree(str(snapshot_root), ignore_errors=True)


def _verify_build_input_snapshot(
    inputs: Mapping[str, object], plan: CandidatePlan,
) -> None:
    lock_fd = inputs["lock_fd"]
    lock_parent_fd = inputs["lock_parent_fd"]
    current_lock = os.stat(
        plan.lock_path.name, dir_fd=lock_parent_fd, follow_symlinks=False,
    )
    opened_lock = os.fstat(lock_fd)
    if (
        not stat.S_ISREG(current_lock.st_mode)
        or (current_lock.st_dev, current_lock.st_ino)
        != (opened_lock.st_dev, opened_lock.st_ino)
        or _read_open_regular_descriptor(lock_fd) != inputs["raw_lock"]
    ):
        raise ConverterPolicyError("converter build inputs changed during install")
    wheelhouse_fd = inputs["wheelhouse_fd"]
    current_wheelhouse = plan.wheelhouse.stat()
    opened_wheelhouse = os.fstat(wheelhouse_fd)
    if (
        (current_wheelhouse.st_dev, current_wheelhouse.st_ino)
        != (opened_wheelhouse.st_dev, opened_wheelhouse.st_ino)
        or list(_wheelhouse_fd_inventory(
            wheelhouse_fd, plan.descriptor["wheels"]
        )) != plan.descriptor["wheels"]
    ):
        raise ConverterPolicyError("converter build inputs changed during install")
    snapshot_audit = audit_lock(inputs["snapshot_lock"])
    snapshot_wheels = inventory_wheelhouse(
        inputs["snapshot_wheelhouse"], snapshot_audit
    )
    if (
        snapshot_audit.lock_sha256 != plan.descriptor["lock_sha256"]
        or list(snapshot_wheels) != plan.descriptor["wheels"]
    ):
        raise ConverterPolicyError("converter build input snapshot changed during install")


@contextmanager
def _active_converter_snapshot(
    home: Path, converter_code_release_id: str, requested_script: Path,
    snapshot: Path,
):
    script = _active_converter_entrypoint(home, converter_code_release_id)
    if requested_script.resolve() != script.resolve():
        raise ConverterPolicyError(
            "converter script is not from the active Task2 code release"
        )
    release = script.parents[2]
    release_fd = _open_real_directory(release, create=False)
    try:
        opened = os.fstat(release_fd)
        current = home / "Desktop" / "stethoscope" / ".wam-current"
        current_info = current.stat()
        if (opened.st_dev, opened.st_ino) != (current_info.st_dev, current_info.st_ino):
            raise ConverterPolicyError("active Task2 code pointer changed before snapshot")
        inventory = _snapshot_directory_fd(release_fd, snapshot)
        import artifact_contract

        envelope = artifact_contract.verify_release(snapshot)
        completion = json.loads(
            (snapshot / artifact_contract.COMPLETE_NAME).read_text(encoding="utf-8")
        )
        if (
            envelope.get("release_id") != converter_code_release_id
            or envelope["payload"].get("artifact_kind") != "code_release"
            or envelope["payload"].get("source_role") != "mac"
            or completion.get("target_role") != "linbox"
        ):
            raise ConverterPolicyError("converter code snapshot identity is invalid")
        snapshot_script = snapshot / Path(*_CONVERTER_RELATIVE.split("/"))
        if snapshot_script.is_symlink() or not snapshot_script.is_file():
            raise ConverterPolicyError("converter code snapshot lacks its entry point")
        yield {
            "inventory": inventory,
            "release_fd": release_fd,
            "script": snapshot_script,
            "snapshot": snapshot,
        }
    finally:
        os.close(release_fd)


def _verify_converter_snapshot(snapshot: Mapping[str, object]) -> None:
    if _directory_fd_inventory(snapshot["release_fd"]) != snapshot["inventory"]:
        raise ConverterPolicyError("converter code release changed during conversion")
    import artifact_contract

    try:
        artifact_contract.verify_release(snapshot["snapshot"])
    except Exception as exc:
        raise ConverterPolicyError("converter code snapshot changed during conversion") from exc


def _conversion_operation(
    *, repo_id: str, raw_identity: Mapping[str, object],
    marker: Mapping[str, object], converter_code_release_id: str,
) -> Dict[str, object]:
    return {
        "converter_code_release_id": converter_code_release_id,
        "converter_environment_id": marker["environment_id"],
        "converter_lock_sha256": marker["descriptor"]["lock_sha256"],
        "promotion_contract_version": raw_identity["promotion_contract_version"],
        "raw_finalization_sha256": raw_identity["raw_finalization_sha256"],
        "raw_manifest_sha256": raw_identity["raw_manifest_sha256"],
        "raw_payload_sha256": raw_identity["raw_payload_sha256"],
        "raw_release_id": raw_identity["raw_release_id"],
        "raw_session_id": raw_identity["raw_session_id"],
        "raw_training_authenticity_sha256": raw_identity[
            "raw_training_authenticity_sha256"
        ],
        "repo_id": repo_id,
        "schema": "wam_linbox_conversion_operation_v2",
    }


def _load_conversion_result(
    path: Path, *, operation: Mapping[str, object], releases_root: Path
) -> Dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ConverterPolicyError("conversion operation marker is unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConverterPolicyError("conversion operation marker is invalid") from exc
    if (
        not isinstance(value, dict)
        or raw != canonical_json_bytes(value)
        or set(value) != {"operation", "release_id", "reopen", "schema"}
        or value.get("schema") != "wam_linbox_conversion_complete_v1"
        or value.get("operation") != dict(operation)
        or not isinstance(value.get("release_id"), str)
        or not re.fullmatch(r"lerobot_dataset-[0-9a-f]{20}", value["release_id"])
        or not isinstance(value.get("reopen"), dict)
    ):
        raise ConverterPolicyError("conversion operation marker is outside policy")
    reopen = value["reopen"]
    if (
        set(reopen) != {"episodes", "frames", "media_files", "repo_id", "schema"}
        or reopen.get("schema") != "wam_lerobot_dataset_reopen_v1"
        or reopen.get("repo_id") != operation["repo_id"]
        or any(
            type(reopen.get(field)) is not int or reopen[field] <= 0
            for field in ("episodes", "frames", "media_files")
        )
    ):
        raise ConverterPolicyError("conversion operation reopen evidence is invalid")
    import remote_agent

    release = releases_root / value["release_id"]
    installed = remote_agent.validate_converted_dataset(
        release, allowed_releases_root=releases_root
    )
    if _load_operation_record(
        release / _CONVERSION_OPERATION_NAME
    ) != _operation_record(operation):
        raise ConverterPolicyError(
            "conversion result does not bind its exact operation"
        )
    for field in (
        "converter_code_release_id", "converter_environment_id",
        "converter_lock_sha256", "promotion_contract_version",
        "raw_finalization_sha256", "raw_manifest_sha256", "raw_payload_sha256",
        "raw_release_id", "raw_session_id", "raw_training_authenticity_sha256",
    ):
        if installed[field] != operation[field]:
            raise ConverterPolicyError(
                "conversion result provenance does not match its operation"
            )
    return {
        "environment_id": operation["converter_environment_id"],
        "release_id": value["release_id"],
        "release_path": release.as_posix(),
        "reopen": value["reopen"],
        "schema": "wam_linbox_conversion_result_v1",
    }


def _validate_reopen(value: object, *, repo_id: str) -> Dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value) != {
            "episodes", "frames", "media_files", "repo_id", "schema"
        }
        or value.get("schema") != "wam_lerobot_dataset_reopen_v1"
        or value.get("repo_id") != repo_id
        or any(
            type(value.get(field)) is not int or value[field] <= 0
            for field in ("episodes", "frames", "media_files")
        )
    ):
        raise ConverterExecutionError("converted dataset reopen evidence is invalid")
    return value


def _operation_record(operation: Mapping[str, object]) -> Dict[str, object]:
    return {
        "operation": dict(operation),
        "operation_id": hashlib.sha256(canonical_json_bytes(operation)).hexdigest(),
        "schema": "wam_linbox_conversion_operation_record_v1",
    }


def _load_operation_record(path: Path) -> Dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ConverterPolicyError("converted release operation record is unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConverterPolicyError("converted release operation record is invalid") from exc
    if (
        not isinstance(value, dict)
        or raw != canonical_json_bytes(value)
        or set(value) != {"operation", "operation_id", "schema"}
        or value.get("schema") != "wam_linbox_conversion_operation_record_v1"
        or not isinstance(value.get("operation"), dict)
        or value.get("operation_id") != hashlib.sha256(
            canonical_json_bytes(value["operation"])
        ).hexdigest()
    ):
        raise ConverterPolicyError("converted release operation record is invalid")
    return value


def _reconcile_published_operation(
    *, operation: Mapping[str, object], operation_path: Path,
    releases_root: Path, candidate: Path,
    isolated_environment: Mapping[str, str], runner,
) -> Union[Dict[str, object], None]:
    expected = _operation_record(operation)
    matches = []
    for release in sorted(releases_root.iterdir()):
        if release.is_symlink() or not release.is_dir():
            raise ConverterPolicyError("converted release namespace contains an unsafe entry")
        if not re.fullmatch(r"lerobot_dataset-[0-9a-f]{20}", release.name):
            raise ConverterPolicyError("converted release namespace contains an invalid identity")
        record_path = release / _CONVERSION_OPERATION_NAME
        if not os.path.lexists(str(record_path)):
            continue
        record = _load_operation_record(record_path)
        if record == expected:
            matches.append(release)
    if not matches:
        return None
    if len(matches) != 1:
        raise ConverterPolicyError("conversion operation maps to multiple releases")
    release = matches[0]
    import remote_agent

    remote_agent.validate_converted_dataset(
        release, allowed_releases_root=releases_root
    )
    completed = _completed_process(
        runner,
        (
            str(candidate / "bin" / "python"), "-c", _REOPEN_SCRIPT,
            str(operation["repo_id"]), str(release / "dataset"),
        ),
        label="published converted dataset reconcile reopen",
        environment=isolated_environment,
    )
    reopen = _validate_reopen(
        _parse_canonical_response(
            completed, label="published converted dataset reconcile reopen"
        ),
        repo_id=str(operation["repo_id"]),
    )
    _atomic_json(operation_path, {
        "operation": dict(operation),
        "release_id": release.name,
        "reopen": reopen,
        "schema": "wam_linbox_conversion_complete_v1",
    })
    return _load_conversion_result(
        operation_path, operation=operation, releases_root=releases_root
    )


def convert_and_publish(
    *,
    session_path: Union[str, Path],
    repo_id: str,
    candidate_path: Union[str, Path],
    converter_code_release_id: str,
    home: Union[str, Path],
    converter_script: Union[str, Path],
    host_role: str,
    trusted_finalization_registry: Union[str, Path],
    trusted_snapshot_root: Union[str, Path],
    runner=subprocess.run,
) -> Dict[str, object]:
    if host_role != "linbox":
        raise ConverterPolicyError("session conversion is allowed only on linbox")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}", repo_id):
        raise ConverterPolicyError("converter repo id is invalid")
    home_path = Path(home)
    sessions_root = home_path / "wam_data" / "sessions"
    session = Path(session_path)
    candidate = Path(candidate_path)
    if candidate.parent != home_path / "envs" / "releases":
        raise ConverterPolicyError("converter must use an inactive exact environment release")
    candidate_parent = _open_real_directory(candidate.parent, create=False)
    os.close(candidate_parent)
    marker = verify_candidate(candidate)
    _verify_external_binaries(marker)
    runtime = probe_runtime(candidate / "bin" / "python", runner=runner)
    if runtime != marker["descriptor"]["runtime"]:
        raise ConverterPolicyError("converter candidate runtime changed after completion")
    isolated_environment = _isolated_candidate_environment(
        candidate, marker["descriptor"]["media_runtime"]["external_binaries"]
    )
    smoke_completed = _completed_process(
        runner, (str(candidate / "bin" / "python"), "-c", _SMOKE_SCRIPT),
        label="converter preflight smoke", environment=isolated_environment,
    )
    live_smoke = _validate_smoke(
        _parse_canonical_response(smoke_completed, label="converter preflight smoke")
    )
    if live_smoke != marker["smoke"]:
        raise ConverterPolicyError("converter PyAV library identity changed after completion")
    script = _active_converter_entrypoint(home_path, converter_code_release_id)
    requested_script = Path(converter_script)
    if requested_script.resolve() != script.resolve():
        raise ConverterPolicyError(
            "converter script is not from the active Task2 code release"
        )

    data_root = home_path / "wam_data" / "lerobot"
    _ensure_real_directory(data_root)
    with _namespace_lock(data_root, ".conversion.lock"):
        raw_identity = verify_sealed_session(
            session,
            sessions_root=sessions_root,
            host_role=host_role,
            trusted_finalization_registry=trusted_finalization_registry,
            trusted_snapshot_root=trusted_snapshot_root,
        )
        session = Path(raw_identity["verified_snapshot_path"])
        conversion_root = data_root / ".conversion"
        releases_root = data_root / "releases"
        operations_root = data_root / ".operations"
        _ensure_real_directory(conversion_root)
        _ensure_real_directory(releases_root)
        _ensure_real_directory(operations_root)
        operation = _conversion_operation(
            repo_id=repo_id, raw_identity=raw_identity, marker=marker,
            converter_code_release_id=converter_code_release_id,
        )
        operation_id = hashlib.sha256(canonical_json_bytes(operation)).hexdigest()
        operation_path = operations_root / (operation_id + ".json")
        if os.path.lexists(str(operation_path)):
            return _load_conversion_result(
                operation_path, operation=operation, releases_root=releases_root
            )
        reconciled = _reconcile_published_operation(
            operation=operation, operation_path=operation_path,
            releases_root=releases_root, candidate=candidate,
            isolated_environment=isolated_environment, runner=runner,
        )
        if reconciled is not None:
            return reconciled
        workspace = Path(tempfile.mkdtemp(prefix="convert-", dir=str(conversion_root)))
        sealed_snapshot = session
        snapshot_identity = verify_sealed_session(
            sealed_snapshot,
            sessions_root=Path(trusted_snapshot_root),
            host_role=host_role,
            trusted_finalization_registry=trusted_finalization_registry,
            trusted_snapshot_root=trusted_snapshot_root,
            trusted_session_id=str(raw_identity["raw_session_id"]),
        )
        if snapshot_identity != raw_identity or verify_sealed_session(
            session,
            sessions_root=Path(trusted_snapshot_root),
            host_role=host_role,
            trusted_finalization_registry=trusted_finalization_registry,
            trusted_snapshot_root=trusted_snapshot_root,
            trusted_session_id=str(raw_identity["raw_session_id"]),
        ) != raw_identity:
            raise ConverterPolicyError("sealed session changed while snapshotting")
        dataset_root = workspace / "dataset"
        try:
            with _active_converter_snapshot(
                home_path, converter_code_release_id, requested_script,
                workspace / "code",
            ) as code_snapshot:
                _completed_process(
                    runner,
                    (
                        str(candidate / "bin" / "python"),
                        str(code_snapshot["script"]),
                        str(sealed_snapshot), "--repo-id", repo_id,
                        "--root", str(dataset_root),
                        "--trusted-finalization-registry",
                        str(trusted_finalization_registry),
                        "--trusted-snapshot-root", str(trusted_snapshot_root),
                        "--trusted-session-id", str(raw_identity["raw_session_id"]),
                    ),
                    label="canonical LeRobot conversion",
                    environment=isolated_environment,
                )
                completed = _completed_process(
                    runner,
                    (
                        str(candidate / "bin" / "python"), "-c", _REOPEN_SCRIPT,
                        repo_id, str(dataset_root),
                    ),
                    label="converted dataset reopen",
                    environment=isolated_environment,
                )
                reopen = _parse_canonical_response(
                    completed, label="converted dataset reopen"
                )
                reopen = _validate_reopen(reopen, repo_id=repo_id)
                if (
                    verify_sealed_session(
                        sealed_snapshot,
                        sessions_root=Path(trusted_snapshot_root),
                        host_role=host_role,
                        trusted_finalization_registry=trusted_finalization_registry,
                        trusted_snapshot_root=trusted_snapshot_root,
                        trusted_session_id=str(raw_identity["raw_session_id"]),
                    ) != raw_identity
                    or verify_sealed_session(
                        session,
                        sessions_root=Path(trusted_snapshot_root),
                        host_role=host_role,
                        trusted_finalization_registry=trusted_finalization_registry,
                        trusted_snapshot_root=trusted_snapshot_root,
                        trusted_session_id=str(raw_identity["raw_session_id"]),
                    ) != raw_identity
                ):
                    raise ConverterPolicyError("sealed session changed during conversion")
                if verify_candidate(candidate) != marker:
                    raise ConverterPolicyError(
                        "converter environment changed during conversion"
                    )
                _verify_converter_snapshot(code_snapshot)

            dataset_manifest = {
                "files": _dataset_inventory(dataset_root),
                "schema": "wam_dataset_payload_manifest_v1",
            }
            dataset_manifest_bytes = canonical_json_bytes(dataset_manifest)
            (workspace / "DATASET_MANIFEST.json").write_bytes(dataset_manifest_bytes)
            provenance = {
                "converter_code_release_id": converter_code_release_id,
                "converter_environment_id": marker["environment_id"],
                "converter_lock_sha256": marker["descriptor"]["lock_sha256"],
                "dataset_manifest_sha256": hashlib.sha256(
                    dataset_manifest_bytes
                ).hexdigest(),
                "dataset_schema_version": "lerobot_v2",
                "lerobot_version": LEROBOT_VERSION,
                "promotion_contract_version": raw_identity[
                    "promotion_contract_version"
                ],
                "raw_finalization_sha256": raw_identity["raw_finalization_sha256"],
                "raw_manifest_sha256": raw_identity["raw_manifest_sha256"],
                "raw_payload_sha256": raw_identity["raw_payload_sha256"],
                "raw_release_id": raw_identity["raw_release_id"],
                "raw_session_id": raw_identity["raw_session_id"],
                "raw_training_authenticity_sha256": raw_identity[
                    "raw_training_authenticity_sha256"
                ],
                "schema": "wam_converted_dataset_provenance_v2",
            }
            (workspace / "DATASET_PROVENANCE.json").write_bytes(
                canonical_json_bytes(provenance)
            )
            (workspace / _CONVERSION_OPERATION_NAME).write_bytes(
                canonical_json_bytes(_operation_record(operation))
            )
            relative_paths = [row["path"] for row in dataset_manifest["files"]]
            relative_paths.extend(
                (
                    "DATASET_MANIFEST.json", "DATASET_PROVENANCE.json",
                    _CONVERSION_OPERATION_NAME,
                )
            )
            import artifact_contract

            envelope = artifact_contract.build_manifest(
                workspace, relative_paths,
                artifact_kind="lerobot_dataset", source_role="linbox",
            )
            final = releases_root / envelope["release_id"]
            if not os.path.lexists(str(final)):
                try:
                    artifact_contract.write_release(
                        workspace, envelope, final, target_role="linbox"
                    )
                except artifact_contract.ConflictError:
                    pass
            import remote_agent

            installed = remote_agent.validate_converted_dataset(
                final, allowed_releases_root=releases_root
            )
            if installed["release_id"] != envelope["release_id"]:
                raise ConverterPolicyError(
                    "converted dataset final identity does not match this conversion"
                )
            _atomic_json(operation_path, {
                "operation": operation,
                "release_id": envelope["release_id"],
                "reopen": reopen,
                "schema": "wam_linbox_conversion_complete_v1",
            })
            return _load_conversion_result(
                operation_path, operation=operation, releases_root=releases_root
            )
        finally:
            shutil.rmtree(str(workspace), ignore_errors=True)


def _probe_actual_host_identity() -> Tuple[Path, str]:
    return Path(pwd.getpwuid(os.geteuid()).pw_dir), socket.gethostname()


def _require_actual_linbox_host(
    *, environ: Mapping[str, str], actual_home: Path, actual_hostname: str,
) -> None:
    if environ.get("WAM_HOST_ROLE") != "linbox":
        raise ConverterPolicyError("converter manager may run only on linbox")
    if actual_home.as_posix() != "/home/robot" or actual_hostname != "robot-G914":
        raise ConverterPolicyError("converter actual host identity is not linbox")


def main(
    argv=None, *, environ: Mapping[str, str] = os.environ,
    home: Union[str, Path, None] = None,
    host_identity_probe=_probe_actual_host_identity,
) -> int:
    parser = argparse.ArgumentParser(
        description="Manage the fail-closed linbox LeRobot 0.3.3 converter"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--base-python", required=True)
    build.add_argument("--lock", required=True)
    build.add_argument("--wheelhouse", required=True)
    convert = commands.add_parser("convert")
    convert.add_argument("--session", required=True)
    convert.add_argument("--repo-id", required=True)
    convert.add_argument("--environment", required=True)
    convert.add_argument("--code-release-id", required=True)
    convert.add_argument("--trusted-finalization-registry", default=None)
    convert.add_argument("--trusted-snapshot-root", default=None)
    args = parser.parse_args(argv)
    actual_home, actual_hostname = host_identity_probe()
    actual_home = Path(actual_home)
    home_path = actual_home if home is None else Path(home)
    _require_actual_linbox_host(
        environ=environ, actual_home=actual_home,
        actual_hostname=str(actual_hostname),
    )
    host_role = "linbox"
    if args.command == "build":
        runtime = probe_runtime(args.base_python)
        plan = plan_candidate(
            host_role=host_role,
            lock_path=Path(args.lock),
            releases_root=home_path / "envs" / "releases",
            runtime=runtime,
            wheelhouse=Path(args.wheelhouse),
            external_binaries=[],
        )
        candidate = build_candidate(
            plan, base_python=args.base_python, host_role=host_role
        )
        result = {
            "environment_id": plan.environment_id,
            "environment_path": candidate.as_posix(),
            "schema": "wam_linbox_converter_build_result_v1",
        }
    elif args.command == "convert":
        trusted_finalization_registry = (
            args.trusted_finalization_registry
            or environ.get("WAM_TRAINING_FINALIZATION_REGISTRY")
        )
        trusted_snapshot_root = (
            args.trusted_snapshot_root
            or environ.get("WAM_TRAINING_SNAPSHOT_ROOT")
        )
        if not trusted_finalization_registry or not trusted_snapshot_root:
            raise ConverterPolicyError(
                "converter trusted finalization registry and snapshot root are required"
            )
        active_script = _active_converter_entrypoint(
            home_path, args.code_release_id
        )
        result = convert_and_publish(
            session_path=Path(args.session),
            repo_id=args.repo_id,
            candidate_path=Path(args.environment),
            converter_code_release_id=args.code_release_id,
            home=home_path,
            converter_script=active_script,
            host_role=host_role,
            trusted_finalization_registry=trusted_finalization_registry,
            trusted_snapshot_root=trusted_snapshot_root,
        )
    else:
        raise ConverterPolicyError("unsupported converter manager command")
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.buffer.write(canonical_json_bytes({
            "error": str(exc),
            "error_type": type(exc).__name__,
            "schema": "wam_linbox_converter_error_v1",
        }))
        raise SystemExit(1)
