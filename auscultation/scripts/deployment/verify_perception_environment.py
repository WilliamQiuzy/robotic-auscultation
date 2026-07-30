"""Fail-closed verification of the live Linbox perception interpreter.

Legacy locks are accepted only for read-only audits and report explicit schema
gaps.  A production v2 lock hashes the actual bytes and sizes of every regular
file declared by every installed distribution's ``RECORD``.  It also hashes
the complete tree of each critical imported module, including nested Python
files and native extensions.  Its environment id binds that complete target
identity rather than merely the mutable clone source.

All identities that *are* reproducible are checked directly.  In addition, a
completed clone must contain ``PERCEPTION_ENVIRONMENT_CLONE_COMPLETE.json`` at
its environment root.  That canonical marker binds the root, interpreter,
lock, source fingerprint, and inventory digest.  The strong inventory uses:

1. normalize distribution names with ``[-_.]+ -> '-'`` and lowercase them;
2. parse the on-disk ``RECORD`` through a held environment-root descriptor;
3. reject missing, escaping, conflicting spelling, symlink, or non-regular
   entries while binding the exact raw RECORD bytes (including duplicate rows);
4. aggregate shared installed paths under sorted normalized owner sets;
5. hash each actual path's environment-relative path, bytes, and size once; and
6. SHA-256 the artifact contract's canonical JSON bytes for sorted identities.

The verifier never creates that marker.  The trusted clone/publish step owns
creation; verification is read-only and fails when it is absent or mismatched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import sysconfig
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import artifact_contract as contract
import perception_environment_manifest as environment_manifest
import publish_perception_runtime as publisher


VERIFICATION_SCHEMA = "wam_perception_environment_verification_v1"
CLONE_MARKER_SCHEMA = "wam_perception_environment_clone_marker_v1"
RELOCATION_AWARE_CLONE_MARKER_SCHEMA = (
    "wam_perception_environment_clone_marker_v2"
)
CLONE_MARKER_NAME = "PERCEPTION_ENVIRONMENT_CLONE_COMPLETE.json"
DISTRIBUTION_INVENTORY_ALGORITHM = (
    "sha256-canonical-json-normalized-name-version-record-and-installed-path-bytes-size-sorted-owners-v3"
)
LEGACY_BYTE_DISTRIBUTION_INVENTORY_ALGORITHM = (
    "sha256-canonical-json-normalized-name-version-record-and-installed-files-bytes-size-v2"
)
LEGACY_DISTRIBUTION_INVENTORY_ALGORITHM = (
    "sha256-canonical-json-normalized-name-version-record-text-v1"
)
MODULE_TREE_ALGORITHM = "sha256-canonical-json-environment-relative-module-tree-v1"
DEFAULT_LOCK = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "perception_runtime"
    / "linbox-ausc-live.lock.json"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NORMALIZE_NAME = re.compile(r"[-_.]+")
CRITICAL_MODULES = tuple(sorted(publisher.SUPPORTED_MODULES))


def _lock_schema_gaps(lock: Mapping[str, object]) -> list[str]:
    schema = lock.get("schema")
    if not isinstance(schema, str) or not schema:
        raise EnvironmentVerificationError("environment lock schema is invalid")
    if (
        schema == publisher.ENVIRONMENT_LOCK_SCHEMA
        and publisher.is_strong_environment_lock(dict(lock))
    ):
        return []
    if schema == publisher.ENVIRONMENT_LOCK_SCHEMA:
        if lock.get("distribution_inventory", {}).get("algorithm") == (
            LEGACY_BYTE_DISTRIBUTION_INVENTORY_ALGORITHM
        ):
            return [
                "installed-path identity lacks deterministic shared-owner sets",
            ]
        return [
            "distribution inventory locks RECORD text but not installed file bytes",
            "module identities lock only entrypoint files, not complete module trees",
        ]
    return [
        "distribution_inventory.sha256 lacks an algorithm and row projection in %s"
        % schema,
        "provisioning.source_environment_fingerprint_sha256 lacks an algorithm in %s"
        % schema,
    ]


class EnvironmentVerificationError(RuntimeError):
    """The running process does not satisfy the selected environment lock."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise EnvironmentVerificationError("%s must be a lowercase SHA-256" % label)
    return value


def record_observation_summary(value: object) -> dict:
    try:
        observation = environment_manifest.validate_record_observation(value)
    except environment_manifest.ManifestError as exc:
        raise EnvironmentVerificationError(
            "RECORD observation is invalid: %s" % exc
        ) from exc
    return {
        "algorithm": observation["algorithm"],
        "claim_count": observation["claim_count"],
        "distribution_count": observation["distribution_count"],
        "sha256": observation["sha256"],
        "status_counts": dict(observation["status_counts"]),
    }


def environment_tree_summary(value: object) -> dict:
    try:
        tree = environment_manifest.validate_environment_tree(value)
    except environment_manifest.ManifestError as exc:
        raise EnvironmentVerificationError(
            "environment tree is invalid: %s" % exc
        ) from exc
    return {
        "algorithm": tree["algorithm"],
        "directories": tree["directories"],
        "entries_count": tree["entries_count"],
        "regular_files": tree["regular_files"],
        "sha256": tree["sha256"],
        "symlinks": tree["symlinks"],
    }


def _exact_keys(value: object, expected: set, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EnvironmentVerificationError("%s must be an object" % label)
    actual = set(value)
    if actual != expected:
        raise EnvironmentVerificationError(
            "%s fields mismatch; missing=%r extra=%r"
            % (label, sorted(expected - actual), sorted(actual - expected))
        )
    return value


def _load_canonical_object(path: Union[str, Path], label: str) -> Tuple[dict, bytes]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise EnvironmentVerificationError("%s must be a regular file" % label)
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentVerificationError("%s must be valid UTF-8 JSON" % label) from exc
    if not isinstance(value, dict) or raw != contract.canonical_json_bytes(value):
        raise EnvironmentVerificationError("%s must be a canonical JSON object" % label)
    return value, raw


def load_environment_lock(path: Union[str, Path]) -> Tuple[dict, str]:
    """Load canonical bytes and validate semantics through the publisher."""
    value, raw = _load_canonical_object(path, "environment lock")
    try:
        publisher.validate_environment_lock(value)
    except publisher.PerceptionRuntimeError as exc:
        raise EnvironmentVerificationError("environment lock is invalid: %s" % exc) from exc
    return value, hashlib.sha256(raw).hexdigest()


def _normalize_distribution_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise EnvironmentVerificationError("distribution name must be a non-empty string")
    normalized = _NORMALIZE_NAME.sub("-", value).lower()
    if not normalized:
        raise EnvironmentVerificationError("distribution name normalizes to empty")
    return normalized


def canonical_distribution_inventory(
    rows: Iterable[Mapping[str, object]],
    installed_paths: Mapping[str, Mapping[str, object]],
) -> dict:
    """Hash distributions plus each actual installed path once with all owners."""
    normalized = []
    seen = set()
    expected_owners = {}
    for row in rows:
        _exact_keys(
            row,
            {
                "name", "version", "record_sha256", "record_entries_count",
                "installed_paths",
            },
            "distribution row",
        )
        name = _normalize_distribution_name(row["name"])
        if name in seen:
            raise EnvironmentVerificationError("duplicate normalized distribution: %s" % name)
        seen.add(name)
        version = row["version"]
        if not isinstance(version, str) or not version:
            raise EnvironmentVerificationError("distribution version is invalid: %s" % name)
        record_entries_count = row["record_entries_count"]
        if type(record_entries_count) is not int or record_entries_count <= 0:
            raise EnvironmentVerificationError(
                "distribution RECORD entry count is invalid: %s" % name
            )
        paths = row["installed_paths"]
        if (
            not isinstance(paths, list)
            or not paths
            or paths != sorted(set(paths))
            or any(not isinstance(path, str) or not path for path in paths)
        ):
            raise EnvironmentVerificationError(
                "distribution installed paths are invalid: %s" % name
            )
        for path in paths:
            expected_owners.setdefault(path, set()).add(name)
        normalized.append({
            "installed_paths_count": len(paths),
            "name": name,
            "record_sha256": _require_sha256(
                row["record_sha256"], "%s RECORD" % name,
            ),
            "record_entries_count": record_entries_count,
            "version": version,
        })
    normalized.sort(key=lambda row: row["name"])
    if set(installed_paths) != set(expected_owners):
        raise EnvironmentVerificationError(
            "installed path ownership drift: path set mismatch"
        )
    canonical_paths = []
    for path in sorted(installed_paths):
        claim = _exact_keys(
            installed_paths[path],
            {
                "device", "inode", "owners", "path", "record_path",
                "sha256", "size",
            },
            "installed path claim",
        )
        if claim["path"] != path:
            raise EnvironmentVerificationError(
                "installed path claim key mismatch"
            )
        owners_value = claim["owners"]
        if not isinstance(owners_value, (set, frozenset, list, tuple)):
            raise EnvironmentVerificationError("installed path owners are invalid")
        if any(
            not isinstance(owner, str)
            or _normalize_distribution_name(owner) != owner
            for owner in owners_value
        ):
            raise EnvironmentVerificationError("installed path owners are invalid")
        owners = sorted(owners_value)
        if len(owners) != len(set(owners)) or set(owners) != expected_owners[path]:
            raise EnvironmentVerificationError(
                "installed path ownership drift: %s" % path
            )
        record_path = claim["record_path"]
        if not isinstance(record_path, str) or not record_path:
            raise EnvironmentVerificationError(
                "installed path RECORD spelling is invalid"
            )
        if type(claim["device"]) is not int or type(claim["inode"]) is not int:
            raise EnvironmentVerificationError(
                "installed path inode observation is invalid"
            )
        if type(claim["size"]) is not int or claim["size"] < 0:
            raise EnvironmentVerificationError("installed path size is invalid")
        canonical_paths.append({
            "owners": owners,
            "path": path,
            "record_path": record_path,
            "sha256": _require_sha256(
                claim["sha256"], "installed path %s" % path,
            ),
            "size": claim["size"],
        })
    return {
        "algorithm": DISTRIBUTION_INVENTORY_ALGORITHM,
        "count": len(normalized),
        "sha256": hashlib.sha256(contract.canonical_json_bytes({
            "distributions": normalized,
            "installed_paths": canonical_paths,
        })).hexdigest(),
    }


def legacy_canonical_distribution_inventory(
    rows: Iterable[Mapping[str, object]],
) -> dict:
    """Reconstruct the old RECORD-text-only identity for read-only audits."""
    projected = []
    seen = set()
    for row in rows:
        name = _normalize_distribution_name(row.get("name"))
        if name in seen:
            raise EnvironmentVerificationError(
                "duplicate normalized distribution: %s" % name
            )
        seen.add(name)
        version = row.get("version")
        if not isinstance(version, str) or not version:
            raise EnvironmentVerificationError(
                "distribution version is invalid: %s" % name
            )
        projected.append({
            "name": name,
            "record_sha256": _require_sha256(
                row.get("record_sha256"), "%s RECORD" % name,
            ),
            "version": version,
        })
    projected.sort(key=lambda row: row["name"])
    return {
        "algorithm": LEGACY_DISTRIBUTION_INVENTORY_ALGORITHM,
        "count": len(projected),
        "sha256": hashlib.sha256(
            contract.canonical_json_bytes(projected)
        ).hexdigest(),
    }


def legacy_byte_canonical_distribution_inventory(
    rows: Iterable[Mapping[str, object]],
    installed_paths: Mapping[str, Mapping[str, object]],
) -> dict:
    """Reconstruct the superseded per-distribution byte identity read-only."""
    projected = []
    for row in rows:
        name = _normalize_distribution_name(row.get("name"))
        paths = row.get("installed_paths")
        if not isinstance(paths, list) or paths != sorted(set(paths)):
            raise EnvironmentVerificationError(
                "legacy byte inventory installed paths are invalid"
            )
        files = []
        for path in paths:
            claim = installed_paths.get(path)
            if not isinstance(claim, Mapping):
                raise EnvironmentVerificationError(
                    "legacy byte inventory installed path is missing"
                )
            files.append({
                "path": path,
                "sha256": _require_sha256(
                    claim.get("sha256"), "legacy installed path %s" % path,
                ),
                "size": claim.get("size"),
            })
        projected.append({
            "files_count": len(files),
            "files_sha256": hashlib.sha256(
                contract.canonical_json_bytes(files)
            ).hexdigest(),
            "name": name,
            "record_sha256": _require_sha256(
                row.get("record_sha256"), "%s RECORD" % name,
            ),
            "version": row.get("version"),
        })
    projected.sort(key=lambda row: row["name"])
    return {
        "algorithm": LEGACY_BYTE_DISTRIBUTION_INVENTORY_ALGORITHM,
        "count": len(projected),
        "sha256": hashlib.sha256(
            contract.canonical_json_bytes(projected)
        ).hexdigest(),
    }


def _environment_relative_path(
    path: Union[str, Path], environment_root: Path, label: str,
) -> Tuple[Path, str]:
    candidate = Path(os.path.abspath(str(path)))
    root = Path(os.path.abspath(str(environment_root)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise EnvironmentVerificationError("%s escapes the environment root" % label) from exc
    if not relative.parts:
        raise EnvironmentVerificationError("%s resolves to the environment root" % label)
    canonical = PurePosixPath(*relative.parts).as_posix()
    return candidate, canonical


def _read_regular_file_at(
    environment_fd: int, relative: str, label: str,
) -> Tuple[bytes, os.stat_result]:
    parts = PurePosixPath(relative).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise EnvironmentVerificationError("%s path is not canonical" % label)
    descriptors = []
    file_fd = -1
    try:
        descriptors.append(os.dup(environment_fd))
        for part in parts[:-1]:
            info = os.stat(part, dir_fd=descriptors[-1], follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise EnvironmentVerificationError("%s traverses a symlink" % label)
            if not stat.S_ISDIR(info.st_mode):
                raise EnvironmentVerificationError("%s parent is not a directory" % label)
            descriptors.append(os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptors[-1],
            ))
        info = os.stat(parts[-1], dir_fd=descriptors[-1], follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise EnvironmentVerificationError("%s is a symlink" % label)
        if not stat.S_ISREG(info.st_mode):
            raise EnvironmentVerificationError("%s is not a regular file" % label)
        file_fd = os.open(
            parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptors[-1],
        )
        before = os.fstat(file_fd)
        if (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino):
            raise EnvironmentVerificationError("%s inode changed while opening" % label)
        chunks = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        if (
            before.st_dev, before.st_ino, before.st_size,
            getattr(before, "st_mtime_ns", int(before.st_mtime * 1_000_000_000)),
        ) != (
            after.st_dev, after.st_ino, after.st_size,
            getattr(after, "st_mtime_ns", int(after.st_mtime * 1_000_000_000)),
        ):
            raise EnvironmentVerificationError("%s changed while hashing" % label)
        return b"".join(chunks), after
    except FileNotFoundError as exc:
        raise EnvironmentVerificationError("%s is missing" % label) from exc
    except OSError as exc:
        raise EnvironmentVerificationError("%s cannot be opened without following links" % label) from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def collect_bundled_ffmpeg_identity(
    imageio_ffmpeg_module: object, *, environment_root: Union[str, Path],
) -> dict:
    """Resolve and hash the exact imageio-ffmpeg bundled Linux executable."""

    root_input = Path(environment_root)
    if root_input.is_symlink() or not root_input.is_dir():
        raise EnvironmentVerificationError("environment root must be a real directory")
    root = root_input.resolve(strict=True)
    resolver = getattr(imageio_ffmpeg_module, "get_ffmpeg_exe", None)
    if not callable(resolver):
        raise EnvironmentVerificationError(
            "imageio_ffmpeg.get_ffmpeg_exe is unavailable"
        )
    try:
        raw_path = resolver()
    except Exception as exc:
        raise EnvironmentVerificationError("bundled ffmpeg resolution failed") from exc
    if not isinstance(raw_path, str) or not raw_path:
        raise EnvironmentVerificationError("bundled ffmpeg path is invalid")
    absolute, relative = _environment_relative_path(
        raw_path, root, "bundled ffmpeg",
    )
    if relative != publisher.BUNDLED_FFMPEG_RELATIVE_PATH:
        raise EnvironmentVerificationError(
            "bundled ffmpeg path does not match the pinned x86_64 wheel"
        )
    environment_fd = os.open(
        str(root),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        data, info = _read_regular_file_at(
            environment_fd, relative, "bundled ffmpeg",
        )
    finally:
        os.close(environment_fd)
    mode = stat.S_IMODE(info.st_mode)
    if not mode & 0o111:
        raise EnvironmentVerificationError("bundled ffmpeg is not executable")
    return {
        "absolute_path": str(absolute),
        "distribution": "imageio-ffmpeg",
        "mode": format(mode, "04o"),
        "path": relative,
        "resolver": "imageio_ffmpeg.get_ffmpeg_exe",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": info.st_size,
        "version": "0.6.0",
    }


def collect_distribution_identity(
    distribution: importlib.metadata.Distribution,
    *,
    environment_root: Union[str, Path],
    environment_fd: int,
    claimed_paths: dict,
    observe_installed_paths: bool = True,
) -> dict:
    """Bind RECORD rows and aggregate safely shared actual installed paths."""
    if not isinstance(claimed_paths, dict):
        raise EnvironmentVerificationError("installed path ledger must be an object")
    root = Path(environment_root)
    name = _normalize_distribution_name(distribution.metadata.get("Name"))
    version = distribution.version
    if not isinstance(version, str) or not version:
        raise EnvironmentVerificationError("distribution version is invalid: %s" % name)
    location = _distribution_location(distribution)
    _record_path, record_relative = _environment_relative_path(
        location / "RECORD", root, "%s RECORD" % name,
    )
    record_bytes, _record_info = _read_regular_file_at(
        environment_fd, record_relative, "%s RECORD" % name,
    )
    try:
        record_text = record_bytes.decode("utf-8")
        record_rows = list(csv.reader(io.StringIO(record_text, newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise EnvironmentVerificationError("%s RECORD is invalid UTF-8 CSV" % name) from exc
    if not record_rows:
        raise EnvironmentVerificationError("%s RECORD is empty" % name)
    local_paths = {}
    for index, row in enumerate(record_rows, 1):
        record_path = row[0] if row else None
        if (
            len(row) != 3
            or not isinstance(record_path, str)
            or not record_path
            or "\x00" in record_path
            or "\\" in record_path
            or PurePosixPath(record_path).is_absolute()
            or PurePosixPath(record_path).as_posix() != record_path
        ):
            raise EnvironmentVerificationError(
                "%s RECORD row %d is invalid" % (name, index)
            )
        located = distribution.locate_file(record_path)
        _absolute, relative = _environment_relative_path(
            located, root, "%s RECORD row %d" % (name, index),
        )
        if relative in local_paths:
            if local_paths[relative] != record_path:
                raise EnvironmentVerificationError(
                    "%s RECORD uses conflicting spellings for installed path: %s"
                    % (name, relative)
                )
            # Duplicate rows from one RECORD are intentional in some real wheels.
            # The exact row multiset, including differing hash/size columns, is
            # already bound by record_sha256.  Observe the actual path only once.
            continue
        local_paths[relative] = record_path
        if not observe_installed_paths:
            continue
        data, info = _read_regular_file_at(
            environment_fd, relative, "%s installed file %s" % (name, relative),
        )
        observed = {
            "device": info.st_dev,
            "inode": info.st_ino,
            "owners": {name},
            "path": relative,
            "record_path": record_path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": info.st_size,
        }
        previous = claimed_paths.get(relative)
        if previous is None:
            claimed_paths[relative] = observed
            continue
        if previous.get("record_path") != record_path:
            raise EnvironmentVerificationError(
                "shared installed path uses conflicting RECORD spelling: %s"
                % relative
            )
        comparable = ("device", "inode", "path", "sha256", "size")
        if any(previous.get(key) != observed[key] for key in comparable):
            raise EnvironmentVerificationError(
                "shared installed path observation conflicts: %s" % relative
            )
        owners = previous.get("owners")
        if not isinstance(owners, set) or name in owners:
            raise EnvironmentVerificationError(
                "installed path owner ledger is inconsistent: %s" % relative
            )
        owners.add(name)
    return {
        "installed_paths": sorted(local_paths),
        "name": name,
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "record_entries_count": len(record_rows),
        "version": version,
    }


def _open_directory_at(environment_fd: int, relative: str, label: str) -> int:
    parts = PurePosixPath(relative).parts
    descriptors = []
    try:
        descriptors.append(os.dup(environment_fd))
        for part in parts:
            info = os.stat(part, dir_fd=descriptors[-1], follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise EnvironmentVerificationError("%s traverses a symlink" % label)
            if not stat.S_ISDIR(info.st_mode):
                raise EnvironmentVerificationError("%s is not a directory" % label)
            descriptors.append(os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptors[-1],
            ))
        result = descriptors.pop()
        return result
    except FileNotFoundError as exc:
        raise EnvironmentVerificationError("%s is missing" % label) from exc
    except OSError as exc:
        raise EnvironmentVerificationError("%s cannot be opened safely" % label) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _module_tree_rows(
    environment_fd: int, directory_fd: int, prefix: str, label: str,
) -> list:
    before = os.fstat(directory_fd)
    rows = []
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise EnvironmentVerificationError("%s cannot be enumerated" % label) from exc
    for name in names:
        if not name or "/" in name or name in (".", ".."):
            raise EnvironmentVerificationError("%s contains an invalid entry" % label)
        relative = "%s/%s" % (prefix, name) if prefix else name
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise EnvironmentVerificationError(
                "%s changed while enumerating" % label
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise EnvironmentVerificationError(
                "%s module tree contains a symlink: %s" % (label, relative)
            )
        if stat.S_ISDIR(info.st_mode):
            try:
                child_fd = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise EnvironmentVerificationError(
                    "%s module directory cannot be opened safely" % label
                ) from exc
            try:
                rows.extend(_module_tree_rows(
                    environment_fd, child_fd, relative, label,
                ))
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(info.st_mode):
            raise EnvironmentVerificationError(
                "%s module tree contains a non-regular file" % label
            )
        data, opened = _read_regular_file_at(
            environment_fd, relative, "%s module file %s" % (label, relative),
        )
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise EnvironmentVerificationError(
                "%s module file inode changed" % label
            )
        rows.append({
            "path": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": opened.st_size,
        })
    after = os.fstat(directory_fd)
    if (
        before.st_dev, before.st_ino,
        getattr(before, "st_mtime_ns", int(before.st_mtime * 1_000_000_000)),
    ) != (
        after.st_dev, after.st_ino,
        getattr(after, "st_mtime_ns", int(after.st_mtime * 1_000_000_000)),
    ):
        raise EnvironmentVerificationError("%s changed while hashing" % label)
    return rows


def _module_paths(
    name: str, module: object, environment_root: Path,
) -> Tuple[str, str, bool]:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise EnvironmentVerificationError("critical module has no file: %s" % name)
    _absolute_file, file_relative = _environment_relative_path(
        module_file, environment_root, "module %s file" % name,
    )
    spec = getattr(module, "__spec__", None)
    locations = getattr(spec, "submodule_search_locations", None)
    if locations is None:
        return file_relative, file_relative, False
    location_values = list(locations)
    if len(location_values) != 1:
        raise EnvironmentVerificationError(
            "critical module package root is ambiguous: %s" % name
        )
    _absolute_root, root_relative = _environment_relative_path(
        location_values[0], environment_root, "module %s package root" % name,
    )
    return file_relative, root_relative, True


def collect_module_tree_identity(
    name: str,
    module: object,
    *,
    environment_root: Union[str, Path],
    environment_fd: int,
) -> dict:
    """Hash the complete imported package tree, including nested native code."""
    root = Path(environment_root)
    _entrypoint_relative, tree_relative, is_package = _module_paths(
        name, module, root,
    )
    if is_package:
        directory_fd = _open_directory_at(
            environment_fd, tree_relative, "module %s" % name,
        )
        try:
            rows = _module_tree_rows(
                environment_fd, directory_fd, tree_relative, "module %s" % name,
            )
        finally:
            os.close(directory_fd)
    else:
        data, info = _read_regular_file_at(
            environment_fd, tree_relative, "module %s" % name,
        )
        rows = [{
            "path": tree_relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": info.st_size,
        }]
    if not rows:
        raise EnvironmentVerificationError("module tree is empty: %s" % name)
    rows.sort(key=lambda row: row["path"])
    return {
        "algorithm": MODULE_TREE_ALGORITHM,
        "count": len(rows),
        "sha256": hashlib.sha256(
            contract.canonical_json_bytes(rows)
        ).hexdigest(),
    }


def _path_within(path: Union[str, Path], root: Union[str, Path], label: str) -> Path:
    candidate = Path(path)
    base = Path(root)
    if not candidate.is_absolute() or not base.is_absolute():
        raise EnvironmentVerificationError("%s must use absolute paths" % label)
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise EnvironmentVerificationError("%s escapes the environment root" % label) from exc
    return candidate


def _validate_marker(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping) and value.get("schema") == (
        RELOCATION_AWARE_CLONE_MARKER_SCHEMA
    ):
        marker = _exact_keys(value, {
            "schema", "complete", "environment_id", "environment_root",
            "python_executable_relative", "lock_sha256",
            "source_environment_fingerprint_sha256",
            "observed_record_observation", "observed_environment_tree",
            "required_service_environment",
        }, "relocation-aware clone marker")
        if marker["complete"] is not True:
            raise EnvironmentVerificationError(
                "relocation-aware clone marker is not complete"
            )
        for field in (
            "environment_id", "environment_root", "python_executable_relative",
        ):
            if not isinstance(marker[field], str) or not marker[field]:
                raise EnvironmentVerificationError(
                    "clone marker %s is invalid" % field
                )
        _require_sha256(marker["lock_sha256"], "clone marker lock_sha256")
        _require_sha256(
            marker["source_environment_fingerprint_sha256"],
            "clone marker source fingerprint",
        )
        record = _exact_keys(marker["observed_record_observation"], {
            "algorithm", "claim_count", "distribution_count", "sha256",
            "status_counts",
        }, "clone marker RECORD summary")
        if record["algorithm"] != environment_manifest.RECORD_OBSERVATION_ALGORITHM:
            raise EnvironmentVerificationError(
                "clone marker RECORD algorithm is unsupported"
            )
        for field in ("claim_count", "distribution_count"):
            if type(record[field]) is not int or record[field] <= 0:
                raise EnvironmentVerificationError(
                    "clone marker RECORD count is invalid"
                )
        _require_sha256(record["sha256"], "clone marker RECORD SHA-256")
        counts = record["status_counts"]
        if (
            not isinstance(counts, Mapping)
            or set(counts) != set(environment_manifest.RECORD_STATUSES)
            or any(type(count) is not int or count < 0 for count in counts.values())
            or sum(counts.values()) != record["claim_count"]
        ):
            raise EnvironmentVerificationError(
                "clone marker RECORD status counts are invalid"
            )
        tree = _exact_keys(marker["observed_environment_tree"], {
            "algorithm", "directories", "entries_count", "regular_files",
            "sha256", "symlinks",
        }, "clone marker tree summary")
        if tree["algorithm"] != environment_manifest.ENVIRONMENT_TREE_ALGORITHM:
            raise EnvironmentVerificationError(
                "clone marker tree algorithm is unsupported"
            )
        for field in (
            "directories", "entries_count", "regular_files", "symlinks",
        ):
            if type(tree[field]) is not int or tree[field] < 0:
                raise EnvironmentVerificationError(
                    "clone marker tree count is invalid"
                )
        if tree["directories"] <= 0 or tree["entries_count"] <= 0:
            raise EnvironmentVerificationError(
                "clone marker tree count is invalid"
            )
        if (
            tree["directories"] + tree["regular_files"] + tree["symlinks"]
            != tree["entries_count"]
        ):
            raise EnvironmentVerificationError(
                "clone marker tree count total is invalid"
            )
        _require_sha256(tree["sha256"], "clone marker tree SHA-256")
        if marker["required_service_environment"] != {
            "PYTHONDONTWRITEBYTECODE": "1"
        }:
            raise EnvironmentVerificationError(
                "clone marker service environment policy is invalid"
            )
        return marker
    marker = _exact_keys(value, {
        "schema", "complete", "environment_id", "environment_root",
        "python_executable_relative", "lock_sha256",
        "source_environment_fingerprint_sha256",
        "observed_distribution_inventory",
    }, "clone marker")
    if marker["schema"] != CLONE_MARKER_SCHEMA or marker["complete"] is not True:
        raise EnvironmentVerificationError("clone marker is not complete")
    for field in ("environment_id", "environment_root", "python_executable_relative"):
        if not isinstance(marker[field], str) or not marker[field]:
            raise EnvironmentVerificationError("clone marker %s is invalid" % field)
    _require_sha256(marker["lock_sha256"], "clone marker lock_sha256")
    _require_sha256(
        marker["source_environment_fingerprint_sha256"],
        "clone marker source fingerprint",
    )
    inventory = _exact_keys(
        marker["observed_distribution_inventory"],
        {"algorithm", "count", "sha256"},
        "clone marker observed inventory",
    )
    if inventory["algorithm"] not in {
        DISTRIBUTION_INVENTORY_ALGORITHM,
        LEGACY_BYTE_DISTRIBUTION_INVENTORY_ALGORITHM,
        LEGACY_DISTRIBUTION_INVENTORY_ALGORITHM,
    }:
        raise EnvironmentVerificationError("clone marker inventory algorithm is unsupported")
    if type(inventory["count"]) is not int or inventory["count"] <= 0:
        raise EnvironmentVerificationError("clone marker inventory count is invalid")
    _require_sha256(inventory["sha256"], "clone marker inventory SHA-256")
    return marker


def load_clone_marker(path: Union[str, Path]) -> Tuple[dict, str]:
    value, raw = _load_canonical_object(path, "environment clone marker")
    _validate_marker(value)
    return value, hashlib.sha256(raw).hexdigest()


def _python_abi() -> str:
    cache_tag = getattr(sys.implementation, "cache_tag", "") or ""
    match = re.fullmatch(r"cpython-(\d{3})", cache_tag)
    return "cp%s" % match.group(1) if match else cache_tag


def _distribution_location(distribution: importlib.metadata.Distribution) -> Path:
    private_path = getattr(distribution, "_path", None)
    if private_path is not None:
        return Path(private_path)
    return Path(distribution.locate_file(""))


def _import_modules_quietly(
    names: Iterable[str], *, importer=importlib.import_module,
) -> dict:
    """Import native extensions without allowing C-level output on stdout/stderr."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    previous_dont_write = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        return {name: importer(name) for name in names}
    finally:
        sys.dont_write_bytecode = previous_dont_write
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(devnull)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _collect_relocation_aware_current_environment(
    *, lock: Mapping[str, object], root: Path, executable: Path,
) -> dict:
    """Collect the complete v3 RECORD/tree identity at its permanent path."""

    try:
        distributions = list(importlib.metadata.distributions())
        omission = lock["provisioning"]["generated_pyc_omission"]
        record = environment_manifest.collect_record_observation(
            distributions,
            environment_root=root,
            relocation_source_root=lock["build_recipe"]["source_environment"][
                "path"
            ],
            omitted_generated_pyc={
                environment_manifest.GENERATED_PYC_OMISSION: omission,
            },
        )
        environment_manifest.validate_record_observation(
            record, require_resolved=True,
        )
    except (KeyError, TypeError, environment_manifest.ManifestError) as exc:
        raise EnvironmentVerificationError(
            "complete RECORD observation failed: %s" % exc
        ) from exc

    by_name = {row["name"]: row for row in record["distributions"]}
    expected_package_names = [row["name"] for row in lock["packages"]]
    missing = sorted(set(expected_package_names) - set(by_name))
    if missing:
        raise EnvironmentVerificationError(
            "critical distributions are missing: %r" % missing
        )
    packages = [
        {
            "name": by_name[name]["name"],
            "record_sha256": by_name[name]["record_sha256"],
            "version": by_name[name]["version"],
        }
        for name in expected_package_names
    ]

    try:
        imported = _import_modules_quietly(tuple(sorted(lock["modules"])))
    except Exception as exc:
        raise EnvironmentVerificationError(
            "critical module import failed: %s" % type(exc).__name__
        ) from exc

    environment_fd = os.open(
        str(root),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        modules = {}
        for name, module in imported.items():
            entrypoint_relative, tree_relative, _is_package = _module_paths(
                name, module, root,
            )
            entrypoint_data, _entrypoint_info = _read_regular_file_at(
                environment_fd,
                entrypoint_relative,
                "module %s entrypoint" % name,
            )
            tree_identity = collect_module_tree_identity(
                name,
                module,
                environment_root=root,
                environment_fd=environment_fd,
            )
            modules[name] = {
                "algorithm": tree_identity["algorithm"],
                "count": tree_identity["count"],
                "entrypoint_sha256": hashlib.sha256(
                    entrypoint_data
                ).hexdigest(),
                "path": str(root / tree_relative),
                "sha256": tree_identity["sha256"],
            }
    finally:
        os.close(environment_fd)

    ffmpeg = collect_bundled_ffmpeg_identity(
        imported["imageio_ffmpeg"], environment_root=root,
    )
    try:
        environment_tree = environment_manifest.collect_environment_tree(root)
    except environment_manifest.ManifestError as exc:
        raise EnvironmentVerificationError(
            "complete environment tree observation failed: %s" % exc
        ) from exc

    torch = imported["torch"]
    try:
        cuda_build = str(torch.version.cuda)
        architectures = list(torch.cuda.get_arch_list())
        runtime_available = bool(torch.cuda.is_available())
        device_capabilities = []
        if runtime_available:
            for index in range(int(torch.cuda.device_count())):
                major, minor = torch.cuda.get_device_capability(index)
                device_capabilities.append("sm_%d%d" % (major, minor))
    except Exception as exc:
        raise EnvironmentVerificationError("CUDA runtime probe failed") from exc

    return {
        "cuda": {
            "architectures": architectures,
            "build": cuda_build,
            "device_capabilities": sorted(set(device_capabilities)),
            "runtime_available": runtime_available,
        },
        "environment_root": str(root),
        "environment_tree": environment_tree,
        "ffmpeg": ffmpeg,
        "modules": modules,
        "packages": packages,
        "python": {
            "abi": _python_abi(),
            "executable": str(executable),
            "executable_sha256": _sha256_file(executable),
            "implementation": platform.python_implementation(),
            "platform": sysconfig.get_platform(),
            "version": platform.python_version(),
        },
        "record_observation": record,
        "runtime_policy": {
            "PYTHONDONTWRITEBYTECODE": os.environ.get(
                "PYTHONDONTWRITEBYTECODE"
            ),
            "sys_dont_write_bytecode": sys.dont_write_bytecode,
        },
    }


def collect_current_environment(
    *, lock: Mapping[str, object], environment_root: Union[str, Path],
) -> dict:
    """Probe the interpreter executing this module; no subprocess is trusted."""
    root_input = Path(environment_root)
    if root_input.is_symlink() or not root_input.is_dir():
        raise EnvironmentVerificationError("environment root must be a real directory")
    root = root_input.resolve(strict=True)
    try:
        current_prefix = Path(sys.prefix).resolve(strict=True)
    except OSError as exc:
        raise EnvironmentVerificationError("running sys.prefix cannot be resolved") from exc
    if current_prefix != root:
        raise EnvironmentVerificationError("running sys.prefix does not match environment root")

    executable = Path(sys.executable)
    if not executable.is_absolute() or not executable.exists():
        raise EnvironmentVerificationError("running Python executable is invalid")
    try:
        executable.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise EnvironmentVerificationError("running Python executable escapes environment root") from exc

    if lock.get("schema") == publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA:
        return _collect_relocation_aware_current_environment(
            lock=lock, root=root, executable=executable,
        )

    environment_fd = os.open(
        str(root),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        all_rows = []
        by_name = {}
        claimed_paths = {}
        inventory_algorithm = lock.get("distribution_inventory", {}).get(
            "algorithm"
        )
        observe_installed_paths = inventory_algorithm in {
            DISTRIBUTION_INVENTORY_ALGORITHM,
            LEGACY_BYTE_DISTRIBUTION_INVENTORY_ALGORITHM,
        }
        for distribution in importlib.metadata.distributions():
            row = collect_distribution_identity(
                distribution,
                environment_root=root,
                environment_fd=environment_fd,
                claimed_paths=claimed_paths,
                observe_installed_paths=observe_installed_paths,
            )
            name = row["name"]
            if name in by_name:
                raise EnvironmentVerificationError(
                    "duplicate normalized distribution: %s" % name
                )
            all_rows.append(row)
            by_name[name] = row

        expected_package_names = [row["name"] for row in lock["packages"]]
        missing = sorted(set(expected_package_names) - set(by_name))
        if missing:
            raise EnvironmentVerificationError(
                "critical distributions are missing: %r" % missing
            )
        packages = [
            {
                "name": by_name[name]["name"],
                "record_sha256": by_name[name]["record_sha256"],
                "version": by_name[name]["version"],
            }
            for name in expected_package_names
        ]

        try:
            # pyorbbecsdk prints its extension path during import; keep stdout strictly
            # reserved for the canonical verification document.
            critical_modules = tuple(sorted(lock["modules"]))
            imported = _import_modules_quietly(critical_modules)
        except Exception as exc:
            raise EnvironmentVerificationError(
                "critical module import failed: %s" % type(exc).__name__
            ) from exc
        modules = {}
        for name, module in imported.items():
            entrypoint_relative, tree_relative, _is_package = _module_paths(
                name, module, root,
            )
            entrypoint_data, _entrypoint_info = _read_regular_file_at(
                environment_fd,
                entrypoint_relative,
                "module %s entrypoint" % name,
            )
            tree = collect_module_tree_identity(
                name,
                module,
                environment_root=root,
                environment_fd=environment_fd,
            )
            modules[name] = {
                "algorithm": tree["algorithm"],
                "count": tree["count"],
                "entrypoint_sha256": hashlib.sha256(entrypoint_data).hexdigest(),
                "path": str(root / tree_relative),
                "sha256": tree["sha256"],
            }
    finally:
        os.close(environment_fd)

    torch = imported["torch"]
    try:
        cuda_build = str(torch.version.cuda)
        architectures = list(torch.cuda.get_arch_list())
        runtime_available = bool(torch.cuda.is_available())
        device_capabilities = []
        if runtime_available:
            for index in range(int(torch.cuda.device_count())):
                major, minor = torch.cuda.get_device_capability(index)
                device_capabilities.append("sm_%d%d" % (major, minor))
    except Exception as exc:
        raise EnvironmentVerificationError("CUDA runtime probe failed") from exc

    return {
        "environment_root": str(root),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "abi": _python_abi(),
            "platform": sysconfig.get_platform(),
            "executable": str(executable),
            "executable_sha256": _sha256_file(executable),
        },
        "distribution_inventory": (
            canonical_distribution_inventory(all_rows, claimed_paths)
            if lock.get("distribution_inventory", {}).get("algorithm")
            == DISTRIBUTION_INVENTORY_ALGORITHM
            else (
                legacy_byte_canonical_distribution_inventory(
                    all_rows, claimed_paths,
                )
                if lock.get("distribution_inventory", {}).get("algorithm")
                == LEGACY_BYTE_DISTRIBUTION_INVENTORY_ALGORITHM
                else legacy_canonical_distribution_inventory(all_rows)
            )
        ),
        "packages": packages,
        "modules": modules,
        "cuda": {
            "build": cuda_build,
            "architectures": architectures,
            "runtime_available": runtime_available,
            "device_capabilities": sorted(set(device_capabilities)),
        },
    }


def _verify_relocation_aware_observation(
    *, lock: Mapping[str, object], lock_sha256: str,
    marker: Mapping[str, object], marker_sha256: str,
    marker_path: Union[str, Path], observation: Mapping[str, object],
) -> dict:
    lock_digest = _require_sha256(lock_sha256, "environment lock SHA-256")
    marker_digest = _require_sha256(marker_sha256, "clone marker SHA-256")
    marker_value = _validate_marker(marker)
    if marker_value["schema"] != RELOCATION_AWARE_CLONE_MARKER_SCHEMA:
        raise EnvironmentVerificationError(
            "relocation-aware lock requires the v2 clone marker"
        )
    observed = _exact_keys(observation, {
        "environment_root", "python", "record_observation", "environment_tree",
        "packages", "modules", "ffmpeg", "cuda", "runtime_policy",
    }, "relocation-aware runtime observation")
    root_value = observed["environment_root"]
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise EnvironmentVerificationError("observed environment root is invalid")
    root = Path(root_value)
    expected_root = publisher.expected_final_environment_root(lock)
    if root != expected_root:
        raise EnvironmentVerificationError(
            "observed environment is not at its recipe-selected final release path"
        )
    expected_marker_path = root / CLONE_MARKER_NAME
    if Path(marker_path) != expected_marker_path:
        raise EnvironmentVerificationError(
            "clone marker path is not at the environment root"
        )
    if marker_value["environment_root"] != str(root):
        raise EnvironmentVerificationError("clone marker environment root mismatch")
    if marker_value["environment_id"] != lock["environment_id"]:
        raise EnvironmentVerificationError("clone marker environment_id mismatch")
    if marker_value["lock_sha256"] != lock_digest:
        raise EnvironmentVerificationError("clone marker lock SHA-256 mismatch")
    if marker_value["source_environment_fingerprint_sha256"] != lock[
        "provisioning"
    ]["source_environment_fingerprint_sha256"]:
        raise EnvironmentVerificationError("clone marker source fingerprint mismatch")
    relative_executable = marker_value["python_executable_relative"]
    relative_path = Path(relative_executable)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise EnvironmentVerificationError("clone marker Python path is unsafe")

    python = _exact_keys(observed["python"], {
        "implementation", "version", "abi", "platform", "executable",
        "executable_sha256",
    }, "observed Python")
    if {key: python[key] for key in lock["python"]} != dict(lock["python"]):
        raise EnvironmentVerificationError("live python identity mismatch")
    if python["executable"] != str(root / relative_path):
        raise EnvironmentVerificationError(
            "live Python executable does not match clone marker"
        )

    runtime_policy = _exact_keys(observed["runtime_policy"], {
        "PYTHONDONTWRITEBYTECODE", "sys_dont_write_bytecode",
    }, "runtime bytecode policy")
    if dict(runtime_policy) != {
        "PYTHONDONTWRITEBYTECODE": "1",
        "sys_dont_write_bytecode": True,
    }:
        raise EnvironmentVerificationError(
            "PYTHONDONTWRITEBYTECODE runtime policy is not enforced"
        )

    try:
        record = environment_manifest.validate_record_observation(
            observed["record_observation"], require_resolved=True,
        )
        tree = environment_manifest.validate_environment_tree(
            observed["environment_tree"]
        )
        environment_manifest.cross_check_record_tree(record, tree)
    except environment_manifest.ManifestError as exc:
        raise EnvironmentVerificationError(
            "live complete environment manifest is invalid: %s" % exc
        ) from exc
    if record != lock["record_observation"]:
        raise EnvironmentVerificationError(
            "live complete RECORD observation mismatch"
        )
    if tree != lock["environment_tree"]:
        raise EnvironmentVerificationError("live complete environment tree mismatch")
    record_summary = record_observation_summary(record)
    tree_summary = environment_tree_summary(tree)
    if marker_value["observed_record_observation"] != record_summary:
        raise EnvironmentVerificationError(
            "clone marker complete RECORD observation mismatch"
        )
    if marker_value["observed_environment_tree"] != tree_summary:
        raise EnvironmentVerificationError(
            "clone marker complete environment tree mismatch"
        )
    if observed["packages"] != lock["packages"]:
        raise EnvironmentVerificationError("live critical distributions mismatch")

    modules = _exact_keys(
        observed["modules"], set(lock["modules"]), "observed modules",
    )
    for name, expected in lock["modules"].items():
        identity = _exact_keys(
            modules[name],
            {"algorithm", "count", "entrypoint_sha256", "path", "sha256"},
            "module %s" % name,
        )
        _path_within(identity["path"], root, "module %s path" % name)
        actual = {
            key: identity[key] for key in ("algorithm", "count", "sha256")
        }
        if actual != expected:
            raise EnvironmentVerificationError(
                "live module files mismatch: %s" % name
            )

    ffmpeg = _exact_keys(observed["ffmpeg"], {
        "absolute_path", "distribution", "mode", "path", "resolver",
        "sha256", "size", "version",
    }, "observed bundled ffmpeg")
    if ffmpeg["absolute_path"] != str(root / lock["ffmpeg"]["path"]):
        raise EnvironmentVerificationError("bundled ffmpeg absolute path mismatch")
    if {key: ffmpeg[key] for key in lock["ffmpeg"]} != dict(lock["ffmpeg"]):
        raise EnvironmentVerificationError("bundled ffmpeg identity mismatch")

    cuda = _exact_keys(observed["cuda"], {
        "build", "architectures", "runtime_available", "device_capabilities",
    }, "observed CUDA")
    if (
        cuda["build"] != lock["cuda"]["build"]
        or cuda["architectures"] != lock["cuda"]["architectures"]
    ):
        raise EnvironmentVerificationError("live CUDA build/architectures mismatch")
    if cuda["runtime_available"] is not True or "sm_120" not in cuda[
        "device_capabilities"
    ]:
        raise EnvironmentVerificationError("live sm_120 runtime device is unavailable")

    return {
        "checks": {
            "bundled_ffmpeg": "VERIFIED",
            "clone_marker": "VERIFIED",
            "complete_environment_tree": "VERIFIED",
            "complete_record_observation": "VERIFIED",
            "critical_distributions": "VERIFIED",
            "cuda": "VERIFIED",
            "module_files": "VERIFIED",
            "python": "VERIFIED",
            "runtime_bytecode_policy": "VERIFIED",
        },
        "clone_marker": {
            "path": str(marker_path),
            "sha256": marker_digest,
        },
        "cuda": dict(cuda),
        "environment_id": lock["environment_id"],
        "environment_root": str(root),
        "environment_tree": tree_summary,
        "ffmpeg": dict(ffmpeg),
        "lock_sha256": lock_digest,
        "modules": {name: dict(modules[name]) for name in sorted(modules)},
        "packages": list(observed["packages"]),
        "python": dict(python),
        "record_observation": record_summary,
        "runtime_policy": dict(runtime_policy),
        "schema": VERIFICATION_SCHEMA,
        "schema_gaps": [],
        "status": "VERIFIED",
    }


def verify_observation(
    *, lock: Mapping[str, object], lock_sha256: str,
    marker: Mapping[str, object], marker_sha256: str,
    marker_path: Union[str, Path], observation: Mapping[str, object],
) -> dict:
    """Compare one live observation to the canonical lock and clone marker."""
    publisher.validate_environment_lock(dict(lock))
    if lock.get("schema") == publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA:
        return _verify_relocation_aware_observation(
            lock=lock,
            lock_sha256=lock_sha256,
            marker=marker,
            marker_sha256=marker_sha256,
            marker_path=marker_path,
            observation=observation,
        )
    lock_digest = _require_sha256(lock_sha256, "environment lock SHA-256")
    marker_digest = _require_sha256(marker_sha256, "clone marker SHA-256")
    marker_value = _validate_marker(marker)
    observed = _exact_keys(observation, {
        "environment_root", "python", "distribution_inventory", "packages",
        "modules", "cuda",
    }, "runtime observation")

    root_value = observed["environment_root"]
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise EnvironmentVerificationError("observed environment root is invalid")
    root = Path(root_value)
    expected_marker_path = root / CLONE_MARKER_NAME
    if Path(marker_path) != expected_marker_path:
        raise EnvironmentVerificationError("clone marker path is not at the environment root")
    if marker_value["environment_root"] != str(root):
        raise EnvironmentVerificationError("clone marker environment root mismatch")
    if marker_value["environment_id"] != lock["environment_id"]:
        raise EnvironmentVerificationError("clone marker environment_id mismatch")
    if marker_value["lock_sha256"] != lock_digest:
        raise EnvironmentVerificationError("clone marker lock SHA-256 mismatch")
    if marker_value["source_environment_fingerprint_sha256"] != lock["provisioning"][
        "source_environment_fingerprint_sha256"
    ]:
        raise EnvironmentVerificationError("clone marker source fingerprint mismatch")
    relative_executable = marker_value["python_executable_relative"]
    relative_path = Path(relative_executable)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise EnvironmentVerificationError("clone marker Python path is unsafe")

    python = _exact_keys(observed["python"], {
        "implementation", "version", "abi", "platform", "executable",
        "executable_sha256",
    }, "observed Python")
    expected_python = dict(lock["python"])
    actual_python = {key: python[key] for key in expected_python}
    if actual_python != expected_python:
        raise EnvironmentVerificationError("live python identity mismatch")
    if python["executable"] != str(root / relative_path):
        raise EnvironmentVerificationError("live Python executable does not match clone marker")

    inventory = _exact_keys(observed["distribution_inventory"], {
        "algorithm", "count", "sha256",
    }, "observed distribution inventory")
    expected_inventory_algorithm = lock["distribution_inventory"].get(
        "algorithm", LEGACY_DISTRIBUTION_INVENTORY_ALGORITHM,
    )
    if inventory["algorithm"] != expected_inventory_algorithm:
        raise EnvironmentVerificationError("observed inventory algorithm is unsupported")
    if inventory["count"] != lock["distribution_inventory"]["count"]:
        raise EnvironmentVerificationError("live distribution inventory count mismatch")
    _require_sha256(inventory["sha256"], "observed inventory SHA-256")
    gate_profile = lock["schema"] == publisher.ENVIRONMENT_LOCK_SCHEMA
    strong_profile = gate_profile and publisher.is_strong_environment_lock(dict(lock))
    if gate_profile and dict(inventory) != dict(lock["distribution_inventory"]):
        raise EnvironmentVerificationError("live distribution inventory identity mismatch")
    if marker_value["observed_distribution_inventory"] != inventory:
        raise EnvironmentVerificationError("clone marker observed inventory mismatch")

    if observed["packages"] != lock["packages"]:
        raise EnvironmentVerificationError("live critical distributions mismatch")

    modules = _exact_keys(observed["modules"], set(lock["modules"]), "observed modules")
    for name, expected in lock["modules"].items():
        identity = _exact_keys(
            modules[name],
            {"algorithm", "count", "entrypoint_sha256", "path", "sha256"},
            "module %s" % name,
        )
        _path_within(identity["path"], root, "module %s path" % name)
        actual_module = (
            {key: identity[key] for key in ("algorithm", "count", "sha256")}
            if strong_profile else {"sha256": identity["entrypoint_sha256"]}
        )
        if actual_module != expected:
            raise EnvironmentVerificationError("live module files mismatch: %s" % name)

    cuda = _exact_keys(observed["cuda"], {
        "build", "architectures", "runtime_available", "device_capabilities",
    }, "observed CUDA")
    if (
        cuda["build"] != lock["cuda"]["build"]
        or cuda["architectures"] != lock["cuda"]["architectures"]
    ):
        raise EnvironmentVerificationError("live CUDA build/architectures mismatch")
    if cuda["runtime_available"] is not True or "sm_120" not in cuda["device_capabilities"]:
        raise EnvironmentVerificationError("live sm_120 runtime device is unavailable")

    schema_gaps = _lock_schema_gaps(lock)
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "VERIFIED" if not schema_gaps else "VERIFIED_WITH_SCHEMA_GAPS",
        "environment_id": lock["environment_id"],
        "environment_root": str(root),
        "lock_sha256": lock_digest,
        "clone_marker": {
            "path": str(marker_path),
            "sha256": marker_digest,
        },
        "python": dict(python),
        "distribution_inventory": {
            "lock": dict(lock["distribution_inventory"]),
            "observed": dict(inventory),
        },
        "packages": list(observed["packages"]),
        "modules": {name: dict(modules[name]) for name in sorted(modules)},
        "cuda": dict(cuda),
        "checks": {
            "python": "VERIFIED",
            "critical_distributions": "VERIFIED",
            "module_files": "VERIFIED",
            "cuda": "VERIFIED",
            "clone_marker": "VERIFIED",
            "lock_distribution_inventory_sha256": (
                "VERIFIED" if gate_profile else "OPAQUE_BOUND_BY_CLONE_MARKER"
            ),
            "source_environment_fingerprint_sha256": (
                "BOUND_IN_ENVIRONMENT_ID"
                if gate_profile else "OPAQUE_BOUND_BY_CLONE_MARKER"
            ),
        },
        "schema_gaps": schema_gaps,
    }


def verify_current_environment(
    *, lock_path: Union[str, Path], environment_root: Optional[Union[str, Path]] = None,
) -> dict:
    """Verify the interpreter running this command and return a canonical report value."""
    lock, lock_sha256 = load_environment_lock(lock_path)
    root = Path(sys.prefix if environment_root is None else environment_root)
    if root.is_symlink() or not root.is_dir():
        raise EnvironmentVerificationError("environment root must be a real directory")
    root = root.resolve(strict=True)
    marker_path = root / CLONE_MARKER_NAME
    marker, marker_sha256 = load_clone_marker(marker_path)
    observation = collect_current_environment(lock=lock, environment_root=root)
    return verify_observation(
        lock=lock,
        lock_sha256=lock_sha256,
        marker=marker,
        marker_sha256=marker_sha256,
        marker_path=marker_path,
        observation=observation,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed verification of the running Linbox perception environment",
    )
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument(
        "--environment-root",
        help="expected real environment root (defaults to this interpreter's sys.prefix)",
    )
    return parser


def _emit(stream, value: object) -> None:
    stream.write(contract.canonical_json_bytes(value).decode("utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        report = verify_current_environment(
            lock_path=args.lock,
            environment_root=args.environment_root,
        )
    except EnvironmentVerificationError as exc:
        _emit(sys.stderr, {
            "error": str(exc),
            "schema": VERIFICATION_SCHEMA,
            "status": "FAILED",
        })
        return 1
    except Exception as exc:  # pragma: no cover - last-resort canonical fail-close
        _emit(sys.stderr, {
            "error": "unexpected verifier failure: %s" % type(exc).__name__,
            "schema": VERIFICATION_SCHEMA,
            "status": "FAILED",
        })
        return 1
    _emit(sys.stdout, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
