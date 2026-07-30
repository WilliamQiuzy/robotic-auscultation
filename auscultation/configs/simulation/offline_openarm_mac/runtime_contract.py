from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.tags import Tag, sys_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_DIR.parents[3]
EXPECTED_ENV_ROOT = WORKSPACE_ROOT / ".wam-build/envs/offline-openarm-mac-v1"
CONFIG_PATH = PROJECT_DIR / "pyproject.toml"
LOCK_PATH = PROJECT_DIR / "uv.lock"
MANIFEST_PATH = (
    WORKSPACE_ROOT
    / "docs/runs/2026-07-15-offline-official-mesh-occlusion/raw/runtime_manifest.json"
)
FORBIDDEN_MJX_ROOT = Path.home() / "envs/mjx"

REQUIRES_PYTHON = "==3.11.*"
EXPECTED_DIRECT_DISTRIBUTIONS = {
    "numpy": "2.4.2",
    "mujoco": "3.10.0",
    "usd-core": "24.8",
    "trimesh": "4.12.2",
    "pycollada": "0.9.3",
    "rtree": "1.4.1",
    "scipy": "1.17.1",
    "opencv-python-headless": "5.0.0.93",
    "imageio": "2.37.3",
    "imageio-ffmpeg": "0.6.0",
    "pytest": "9.1.1",
}

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class RuntimeContractError(RuntimeError):
    """Raised when the offline runtime does not satisfy its fail-closed contract."""


@dataclass(frozen=True)
class RuntimeIdentity:
    sys_platform: str
    platform_system: str
    platform_machine: str
    uname_machine: str
    arch_command: str
    sysconfig_platform: str
    python_version: tuple[int, int, int]
    python_version_string: str
    executable: str
    prefix: str
    rosetta_translated: bool


def _realpath(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_identity_command(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeContractError(
            f"runtime identity command failed ({completed.returncode}): {' '.join(argv)}"
        )
    return completed.stdout.strip()


def capture_runtime_identity() -> RuntimeIdentity:
    translated = _run_identity_command(
        ["/usr/sbin/sysctl", "-in", "sysctl.proc_translated"]
    )
    if translated not in {"0", "1"}:
        raise RuntimeContractError(
            f"unexpected sysctl.proc_translated value: {translated!r}"
        )
    version = tuple(sys.version_info[:3])
    return RuntimeIdentity(
        sys_platform=sys.platform,
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        uname_machine=os.uname().machine,
        arch_command=_run_identity_command(["/usr/bin/arch"]),
        sysconfig_platform=sysconfig.get_platform(),
        python_version=(int(version[0]), int(version[1]), int(version[2])),
        python_version_string=platform.python_version(),
        executable=sys.executable,
        prefix=sys.prefix,
        rosetta_translated=translated == "1",
    )


def validate_runtime_identity(identity: RuntimeIdentity) -> dict[str, Any]:
    if identity.sys_platform != "darwin" or identity.platform_system != "Darwin":
        raise RuntimeContractError(
            "offline OpenArm runtime requires native macOS (darwin/Darwin)"
        )
    if identity.rosetta_translated:
        raise RuntimeContractError("Rosetta translation is forbidden")

    architecture_values = {
        "platform.machine": identity.platform_machine,
        "os.uname().machine": identity.uname_machine,
        "/usr/bin/arch": identity.arch_command,
    }
    if any(value != "arm64" for value in architecture_values.values()):
        raise RuntimeContractError(
            f"native arm64 is required, observed {architecture_values}"
        )
    if "arm64" not in identity.sysconfig_platform or "x86_64" in identity.sysconfig_platform:
        raise RuntimeContractError(
            f"native arm64 Python platform is required: {identity.sysconfig_platform!r}"
        )
    if identity.python_version[:2] != (3, 11):
        raise RuntimeContractError(
            f"Python 3.11 is required, observed {identity.python_version_string}"
        )

    env_root = _realpath(EXPECTED_ENV_ROOT)
    forbidden_root = _realpath(FORBIDDEN_MJX_ROOT)
    executable = _realpath(identity.executable)
    prefix = _realpath(identity.prefix)
    if _inside(executable, forbidden_root) or _inside(prefix, forbidden_root):
        raise RuntimeContractError(f"forbidden mjx runtime detected: {forbidden_root}")
    if not _inside(executable, env_root):
        raise RuntimeContractError(
            f"interpreter realpath must be inside {env_root}, observed {executable}"
        )
    if prefix != env_root:
        raise RuntimeContractError(
            f"runtime prefix must equal isolated env {env_root}, observed {prefix}"
        )

    return {
        "arch_command": identity.arch_command,
        "executable": identity.executable,
        "executable_realpath": str(executable),
        "native_arm64": True,
        "platform_machine": identity.platform_machine,
        "platform_system": identity.platform_system,
        "prefix": identity.prefix,
        "prefix_realpath": str(prefix),
        "python_version": list(identity.python_version),
        "python_version_string": identity.python_version_string,
        "rosetta_translated": identity.rosetta_translated,
        "sys_platform": identity.sys_platform,
        "sysconfig_platform": identity.sysconfig_platform,
        "uname_machine": identity.uname_machine,
    }


def _reject_mjx_paths() -> None:
    forbidden_root = _realpath(FORBIDDEN_MJX_ROOT)
    cwd = _realpath(Path.cwd())
    candidates: list[tuple[str, str]] = [
        ("sys.executable", sys.executable),
        ("sys.prefix", sys.prefix),
    ]
    candidates.extend(("sys.path", entry or str(cwd)) for entry in sys.path)
    for variable in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME"):
        value = os.environ.get(variable)
        if value:
            candidates.append((variable, value))
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath is not None:
        candidates.extend(
            ("PYTHONPATH", entry or str(cwd))
            for entry in pythonpath.split(os.pathsep)
        )
    candidates.append(("cwd", str(cwd)))
    for source, candidate in candidates:
        if _inside(_realpath(candidate), forbidden_root):
            raise RuntimeContractError(
                f"forbidden mjx path detected in {source}: {candidate}"
            )


def _validate_artifact_hashes(packages: list[dict[str, Any]]) -> None:
    for package in packages:
        source = package.get("source", {})
        if "registry" not in source:
            continue
        artifacts: list[dict[str, Any]] = []
        if "sdist" in package:
            artifacts.append(package["sdist"])
        artifacts.extend(package.get("wheels", []))
        if not artifacts:
            raise RuntimeContractError(
                f"locked registry package has no hashed artifact: {package['name']}"
            )
        for artifact in artifacts:
            artifact_hash = artifact.get("hash", "")
            if not _SHA256_RE.fullmatch(artifact_hash):
                raise RuntimeContractError(
                    f"invalid lock artifact hash for {package['name']}: {artifact_hash!r}"
                )


def validate_project_contract() -> dict[str, Any]:
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    project = config.get("project", {})
    expected_specs = [
        f"{name}=={version}"
        for name, version in EXPECTED_DIRECT_DISTRIBUTIONS.items()
    ]
    if project.get("requires-python") != REQUIRES_PYTHON:
        raise RuntimeContractError("pyproject requires-python contract drift")
    if project.get("dependencies") != expected_specs:
        raise RuntimeContractError("pyproject direct dependency contract drift")

    lock = tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("version") != 1:
        raise RuntimeContractError(
            f"uv.lock version must be 1, observed {lock.get('version')!r}"
        )
    if lock.get("revision") != 3:
        raise RuntimeContractError(
            f"uv.lock revision must be 3, observed {lock.get('revision')!r}"
        )
    if lock.get("requires-python") != REQUIRES_PYTHON:
        raise RuntimeContractError("uv.lock requires-python contract drift")
    packages = lock.get("package", [])
    _validate_artifact_hashes(packages)

    root_packages = [
        package for package in packages if package.get("name") == "offline-openarm-mac"
    ]
    if len(root_packages) != 1:
        raise RuntimeContractError("uv.lock must contain one offline-openarm-mac root")
    requires_dist = root_packages[0].get("metadata", {}).get("requires-dist", [])
    locked_direct = {
        requirement["name"]: requirement.get("specifier", "").removeprefix("==")
        for requirement in requires_dist
    }
    if locked_direct != EXPECTED_DIRECT_DISTRIBUTIONS:
        raise RuntimeContractError("uv.lock root dependency metadata drift")

    packages_by_name = {package["name"]: package for package in packages}
    for name, expected_version in EXPECTED_DIRECT_DISTRIBUTIONS.items():
        observed = packages_by_name.get(name, {}).get("version")
        if observed != expected_version:
            raise RuntimeContractError(
                f"uv.lock version drift for {name}: expected {expected_version}, got {observed}"
            )

    return {
        "config_path": str(CONFIG_PATH.relative_to(WORKSPACE_ROOT)),
        "config_sha256": _sha256(CONFIG_PATH),
        "direct_dependencies": dict(EXPECTED_DIRECT_DISTRIBUTIONS),
        "lock_path": str(LOCK_PATH.relative_to(WORKSPACE_ROOT)),
        "lock_revision": lock.get("revision"),
        "lock_sha256": _sha256(LOCK_PATH),
        "lock_version": lock.get("version"),
        "locked_package_count": len(packages),
        "requires_python": REQUIRES_PYTHON,
    }


def _strict_distribution_path(
    name: str,
    path: Path,
    env_root: Path,
) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise RuntimeContractError(
            f"distribution {name} has a missing or unresolvable file: {path}"
        ) from error
    if not _inside(resolved, env_root):
        raise RuntimeContractError(
            f"distribution {name} file is outside isolated env: {resolved}"
        )
    if not resolved.is_file() and not resolved.is_dir():
        raise RuntimeContractError(
            f"distribution {name} path is not a regular file or directory: {resolved}"
        )
    return resolved


def _is_allowed_unhashed_record_entry(path: str, record_path: str) -> bool:
    if path == record_path:
        return True
    parsed = PurePosixPath(path)
    return parsed.suffix == ".pyc" and "__pycache__" in parsed.parts


def _decode_record_sha256(name: str, record_path: str, encoded: str) -> bytes:
    if not encoded.startswith("sha256="):
        raise RuntimeContractError(
            f"unsupported RECORD hash for {name}:{record_path}: {encoded!r}"
        )
    payload = encoded.removeprefix("sha256=")
    try:
        digest = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    except (ValueError, TypeError) as error:
        raise RuntimeContractError(
            f"invalid RECORD sha256 for {name}:{record_path}"
        ) from error
    if len(digest) != hashlib.sha256().digest_size:
        raise RuntimeContractError(
            f"invalid RECORD sha256 for {name}:{record_path}"
        )
    return digest


def validate_distribution_provenance(
    name: str,
    distribution: Any,
    *,
    env_root: Path,
    compatible_locked_artifacts: list[dict[str, str]],
) -> dict[str, Any]:
    resolved_env = env_root.resolve(strict=True)
    location = _strict_distribution_path(
        name,
        Path(distribution.locate_file("")),
        resolved_env,
    )
    files = list(distribution.files or [])
    if not files:
        raise RuntimeContractError(f"distribution {name} exposes no RECORD files")
    if not compatible_locked_artifacts:
        raise RuntimeContractError(
            f"distribution {name} has no compatible locked wheel artifact"
        )
    for artifact in compatible_locked_artifacts:
        if not artifact.get("filename", "").endswith(".whl") or not re.fullmatch(
            r"[0-9a-f]{64}", artifact.get("sha256", "")
        ):
            raise RuntimeContractError(
                f"distribution {name} has invalid compatible locked artifact metadata"
            )

    file_paths = [str(file) for file in files]
    if len(file_paths) != len(set(file_paths)):
        raise RuntimeContractError(f"distribution {name} RECORD contains duplicate paths")
    located_files: dict[str, Path] = {}
    for file, record_path in zip(files, file_paths, strict=True):
        located_files[record_path] = _strict_distribution_path(
            name,
            Path(distribution.locate_file(file)),
            resolved_env,
        )

    metadata_paths = [
        path for path in file_paths if path.endswith(".dist-info/METADATA")
    ]
    record_paths = [path for path in file_paths if path.endswith(".dist-info/RECORD")]
    if len(metadata_paths) != 1 or len(record_paths) != 1:
        raise RuntimeContractError(
            f"distribution {name} must expose exactly one METADATA and RECORD file"
        )
    metadata_file = located_files[metadata_paths[0]]
    record_file = located_files[record_paths[0]]
    metadata_dir = metadata_file.parent.resolve(strict=True)
    if record_file.parent.resolve(strict=True) != metadata_dir:
        raise RuntimeContractError(
            f"distribution {name} METADATA and RECORD directories disagree"
        )

    with record_file.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    parsed_rows: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in parsed_rows:
            raise RuntimeContractError(f"distribution {name} has malformed RECORD rows")
        parsed_rows[row[0]] = (row[1], row[2])
    if set(parsed_rows) != set(file_paths):
        raise RuntimeContractError(
            f"distribution {name} files do not exactly match its RECORD"
        )

    canonical_files: list[dict[str, Any]] = []
    allowed_unhashed_paths: list[str] = []
    for record_path in sorted(parsed_rows):
        hash_field, size_field = parsed_rows[record_path]
        actual_path = located_files[record_path]
        if not actual_path.is_file():
            raise RuntimeContractError(
                f"distribution {name} RECORD path is not a file: {actual_path}"
            )
        actual_sha256 = _sha256(actual_path)
        actual_size = actual_path.stat().st_size
        if hash_field:
            expected_digest = _decode_record_sha256(name, record_path, hash_field)
            if not hmac.compare_digest(bytes.fromhex(actual_sha256), expected_digest):
                raise RuntimeContractError(
                    f"RECORD sha256 mismatch for {name}:{record_path}"
                )
        elif _is_allowed_unhashed_record_entry(record_path, record_paths[0]):
            allowed_unhashed_paths.append(record_path)
        else:
            raise RuntimeContractError(
                f"unhashed RECORD entry is not allowed for {name}:{record_path}"
            )
        if size_field:
            try:
                expected_size = int(size_field)
            except ValueError as error:
                raise RuntimeContractError(
                    f"invalid RECORD size for {name}:{record_path}"
                ) from error
            if expected_size != actual_size:
                raise RuntimeContractError(
                    f"RECORD size mismatch for {name}:{record_path}"
                )
        canonical_files.append(
            {"path": record_path, "sha256": actual_sha256, "size": actual_size}
        )

    record_tree_sha256 = hashlib.sha256(
        canonical_json_bytes(canonical_files)
    ).hexdigest()
    return {
        "compatible_locked_artifacts": compatible_locked_artifacts,
        "location_realpath": str(location),
        "metadata_dir_realpath": str(metadata_dir),
        "record_file_count": len(canonical_files),
        "record_tree_sha256": record_tree_sha256,
        "record_unhashed_allowed": allowed_unhashed_paths,
        "version": distribution.version,
    }


def _compatible_locked_artifacts(
    name: str,
    expected_version: str,
    locked_package: dict[str, Any],
    compatible_tags: set[Tag],
) -> list[dict[str, str]]:
    artifacts: set[tuple[str, str]] = set()
    for wheel in locked_package.get("wheels", []):
        filename = Path(
            urllib.parse.unquote(urllib.parse.urlsplit(wheel.get("url", "")).path)
        ).name
        try:
            wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(filename)
        except InvalidWheelFilename as error:
            raise RuntimeContractError(
                f"invalid locked wheel filename for {name}: {filename!r}"
            ) from error
        if canonicalize_name(wheel_name) != canonicalize_name(name):
            raise RuntimeContractError(
                f"locked wheel name mismatch for {name}: {filename!r}"
            )
        if str(wheel_version) != expected_version:
            raise RuntimeContractError(
                f"locked wheel version mismatch for {name}: {filename!r}"
            )
        artifact_hash = wheel.get("hash", "")
        if not _SHA256_RE.fullmatch(artifact_hash):
            raise RuntimeContractError(
                f"invalid locked wheel hash for {name}: {artifact_hash!r}"
            )
        if wheel_tags & compatible_tags:
            artifacts.add((filename, artifact_hash.removeprefix("sha256:")))
    if not artifacts:
        raise RuntimeContractError(
            f"no compatible locked wheel artifact exists for {name} {expected_version}"
        )
    return [
        {"filename": filename, "sha256": artifact_hash}
        for filename, artifact_hash in sorted(artifacts)
    ]


def validate_direct_distributions() -> dict[str, dict[str, Any]]:
    env_root = _realpath(EXPECTED_ENV_ROOT)
    lock = tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("version") != 1 or lock.get("revision") != 3:
        raise RuntimeContractError("uv.lock schema must be version 1 revision 3")
    packages = {package["name"]: package for package in lock.get("package", [])}
    compatible_tags = set(sys_tags())
    records: dict[str, dict[str, Any]] = {}
    for name, expected_version in EXPECTED_DIRECT_DISTRIBUTIONS.items():
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeContractError(f"required distribution is missing: {name}") from error
        if distribution.version != expected_version:
            raise RuntimeContractError(
                f"installed version drift for {name}: "
                f"expected {expected_version}, got {distribution.version}"
            )
        locked_package = packages.get(name)
        if not locked_package or locked_package.get("version") != expected_version:
            raise RuntimeContractError(
                f"locked package metadata is missing or stale for {name}"
            )
        artifacts = _compatible_locked_artifacts(
            name,
            expected_version,
            locked_package,
            compatible_tags,
        )
        records[name] = validate_distribution_provenance(
            name,
            distribution,
            env_root=env_root,
            compatible_locked_artifacts=artifacts,
        )
    return records


def build_runtime_manifest() -> dict[str, Any]:
    _reject_mjx_paths()
    return {
        "contract": validate_project_contract(),
        "direct_distributions": validate_direct_distributions(),
        "runtime": validate_runtime_identity(capture_runtime_identity()),
        "schema": "wam.offline_openarm_mac.runtime.v1",
        "status": "validated",
    }


def canonical_json_bytes(payload: Any) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"payload is not valid canonical JSON: {error}") from error
    return encoded.encode("utf-8")


def write_runtime_manifest(output_path: str | Path) -> dict[str, Any]:
    manifest = build_runtime_manifest()
    payload = canonical_json_bytes(manifest)
    destination = Path(output_path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return manifest


if __name__ == "__main__":
    write_runtime_manifest(MANIFEST_PATH)
    print(MANIFEST_PATH)
