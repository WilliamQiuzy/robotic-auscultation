"""Install one exact, content-addressed openpi source candidate on H200.

This module deliberately stops at the upstream source boundary.  It does not
claim that the WAM observation/action adapter is implemented or training-ready.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
import uuid

import artifact_contract as contract


LOCK_SCHEMA = "wam_external_source_lock_v1"
PROVENANCE_SCHEMA = "wam_external_source_provenance_v1"
SOURCE_INVENTORY_SCHEMA = "wam_external_source_inventory_v1"
LOCK_PATH = Path(__file__).resolve().parents[2] / "configs" / "policy" / "openpi.source.lock.json"
_LOCK_FIELDS = {
    "commit", "environment_boundary", "files", "name", "policy_adapter_status",
    "releases_root", "repository", "schema", "submodules", "target_role", "tree",
}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_LOCK_NAME = ".openpi-source.lock"
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


class SourceLockError(RuntimeError):
    """The source lock, checkout, or completed source release is unsafe."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8") + b"\n"


def sha256_file(path: Union[str, Path]) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise SourceLockError("source identity file is missing, non-regular, or a symlink")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: object, field: str = "path") -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise SourceLockError("%s is invalid" % field)
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise SourceLockError("%s is unsafe" % field)
    return value


def _repository(value: object) -> str:
    if not isinstance(value, str) or not (
        value.startswith("https://github.com/") and value.endswith(".git")
        or value.startswith("file:///")
    ):
        raise SourceLockError("repository must be a fixed HTTPS GitHub or local test URL")
    if any(character in value for character in ("\n", "\r", "\x00")):
        raise SourceLockError("repository contains unsafe characters")
    return value


def _repository_identity(value: object, field: str) -> Tuple[str, str]:
    if not isinstance(value, str) or any(
        character in value for character in ("\n", "\r", "\x00")
    ):
        raise SourceLockError("%s is invalid" % field)
    if value.startswith("https://github.com/"):
        normalized = value[:-4] if value.endswith(".git") else value
        if normalized == "https://github.com/" or normalized.endswith("/"):
            raise SourceLockError("%s is invalid" % field)
        return ("github", normalized)
    if value.startswith("file:///"):
        return ("file", value)
    raise SourceLockError("%s is not an allowed repository URL" % field)


def _same_repository(observed: object, expected: object, field: str) -> bool:
    return _repository_identity(observed, field) == _repository_identity(expected, field)


def validate_source_lock(value: object) -> Dict[str, object]:
    if not isinstance(value, dict) or set(value) != _LOCK_FIELDS:
        raise SourceLockError("source lock fields are invalid")
    if value.get("schema") != LOCK_SCHEMA or value.get("name") != "openpi":
        raise SourceLockError("source lock schema or name is invalid")
    if value.get("target_role") != "h200":
        raise SourceLockError("source target role must be h200")
    if value.get("environment_boundary") != "separate_openpi_uv_lock":
        raise SourceLockError("source environment boundary is invalid")
    if value.get("policy_adapter_status") != "blocked_pending_tracked_wam_adapter":
        raise SourceLockError("source policy adapter status must fail closed")
    _repository(value.get("repository"))
    for field in ("commit", "tree"):
        if not isinstance(value.get(field), str) or not _HEX40.fullmatch(value[field]):
            raise SourceLockError("source %s is invalid" % field)
    releases_root = value.get("releases_root")
    if (
        not isinstance(releases_root, str)
        or not PurePosixPath(releases_root).is_absolute()
        or PurePosixPath(releases_root).as_posix() != releases_root
        or ".." in PurePosixPath(releases_root).parts
    ):
        raise SourceLockError("source releases root is invalid")

    rows = value.get("files")
    if not isinstance(rows, list):
        raise SourceLockError("source file rows are invalid")
    file_paths = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise SourceLockError("source file row fields are invalid")
        file_paths.append(_safe_relative(row.get("path")))
        if not isinstance(row.get("sha256"), str) or not _HEX64.fullmatch(row["sha256"]):
            raise SourceLockError("source file sha256 is invalid")
    if len(set(file_paths)) != len(file_paths):
        raise SourceLockError("source file paths contain a duplicate")
    if set(file_paths) != {".gitmodules", "pyproject.toml", "uv.lock"}:
        raise SourceLockError("source identity file set is invalid")

    modules = value.get("submodules")
    if not isinstance(modules, list):
        raise SourceLockError("source submodules are invalid")
    module_paths = []
    for row in modules:
        if not isinstance(row, dict) or set(row) != {"commit", "path", "repository"}:
            raise SourceLockError("source submodule fields are invalid")
        module_paths.append(_safe_relative(row.get("path"), "submodule path"))
        _repository(row.get("repository"))
        if not isinstance(row.get("commit"), str) or not _HEX40.fullmatch(row["commit"]):
            raise SourceLockError("source submodule commit is invalid")
    if len(set(module_paths)) != len(module_paths):
        raise SourceLockError("source submodule paths contain a duplicate")
    return value


def load_source_lock(path: Union[str, Path] = LOCK_PATH) -> Dict[str, object]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceLockError("source lock is not valid UTF-8 JSON") from exc
    validated = validate_source_lock(value)
    if raw != canonical_json_bytes(validated):
        raise SourceLockError("source lock is not canonical JSON")
    return validated


def _git(checkout: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout)] + list(arguments),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True,
    )
    if completed.returncode != 0:
        raise SourceLockError(
            "git %s failed: %s" % (arguments[0], completed.stderr.strip())
        )
    return completed.stdout


def _git_bytes(checkout: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(checkout)] + list(arguments),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        stderr = bytes(completed.stderr or b"").decode("utf-8", "replace").strip()
        raise SourceLockError(
            "git %s failed: %s" % (arguments[0], stderr)
        )
    return bytes(completed.stdout)


def _git_config_values(checkout: Path, key: str) -> List[str]:
    completed = subprocess.run(
        ["git", "-C", str(checkout), "config", "--get-all", key],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True,
    )
    if completed.returncode == 1 and not completed.stdout:
        return []
    if completed.returncode != 0:
        raise SourceLockError(
            "git config failed: %s" % completed.stderr.strip()
        )
    return completed.stdout.splitlines()


def _gitmodules_repositories(checkout: Path) -> Dict[str, str]:
    completed = subprocess.run(
        [
            "git", "-C", str(checkout), "config", "-z", "--file", ".gitmodules",
            "--get-regexp", r"^submodule\..*\.(path|url)$",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode == 1 and not completed.stdout:
        return {}
    if completed.returncode != 0:
        stderr = bytes(completed.stderr or b"").decode("utf-8", "replace").strip()
        raise SourceLockError("source .gitmodules is invalid: %s" % stderr)
    records: Dict[str, Dict[str, str]] = {}
    try:
        chunks = [chunk for chunk in completed.stdout.split(b"\x00") if chunk]
        for chunk in chunks:
            key_raw, value_raw = chunk.split(b"\n", 1)
            key = key_raw.decode("utf-8")
            item = value_raw.decode("utf-8")
            suffix = ".path" if key.endswith(".path") else ".url" if key.endswith(".url") else None
            if suffix is None or not key.startswith("submodule."):
                raise ValueError("unexpected .gitmodules key")
            section = key[:-len(suffix)]
            field = suffix[1:]
            row = records.setdefault(section, {})
            if field in row:
                raise ValueError("duplicate .gitmodules field")
            row[field] = item
    except (UnicodeDecodeError, ValueError) as exc:
        raise SourceLockError("source .gitmodules entries are invalid") from exc
    repositories: Dict[str, str] = {}
    for row in records.values():
        if set(row) != {"path", "url"}:
            raise SourceLockError("source .gitmodules entries are incomplete")
        path = _safe_relative(row["path"], "submodule path")
        if path in repositories:
            raise SourceLockError("source .gitmodules contains a duplicate path")
        _repository_identity(row["url"], "submodule repository")
        repositories[path] = row["url"]
    return repositories


def _require_origin(checkout: Path, expected: str, field: str) -> str:
    origins = _git_config_values(checkout, "remote.origin.url")
    if len(origins) != 1 or not _same_repository(origins[0], expected, field):
        raise SourceLockError("source %s does not match the lock" % field)
    return origins[0]


def verify_checkout(
    checkout: Union[str, Path],
    lock: Dict[str, object],
    *,
    require_submodules: bool = True,
) -> Dict[str, object]:
    value = validate_source_lock(lock)
    root = Path(checkout)
    if root.is_symlink() or not root.is_dir():
        raise SourceLockError("source checkout must be a real directory")
    commit = _git(root, ("rev-parse", "HEAD")).strip()
    tree = _git(root, ("rev-parse", "HEAD^{tree}")).strip()
    if commit != value["commit"]:
        raise SourceLockError("source checkout commit does not match the lock")
    if tree != value["tree"]:
        raise SourceLockError("source checkout tree does not match the lock")
    if _git(
        root,
        (
            "status", "--porcelain=v1", "--untracked-files=all",
            "--ignore-submodules=all",
        ),
    ).strip():
        raise SourceLockError("source checkout is dirty or has untracked files")
    _require_origin(root, value["repository"], "origin repository")

    hashes = {}
    for row in value["files"]:
        digest = sha256_file(root / Path(*PurePosixPath(row["path"]).parts))
        if digest != row["sha256"]:
            raise SourceLockError("source identity file hash does not match the lock")
        hashes[row["path"]] = digest

    expected = {row["path"]: row["commit"] for row in value["submodules"]}
    expected_repositories = {
        row["path"]: row["repository"] for row in value["submodules"]
    }
    declared_repositories = _gitmodules_repositories(root)
    if set(declared_repositories) != set(expected_repositories) or any(
        not _same_repository(
            declared_repositories[path], expected_repositories[path],
            "submodule repository",
        )
        for path in expected_repositories
    ):
        raise SourceLockError("source submodule repository does not match the lock")
    observed = {}
    missing = []
    output = _git(root, ("submodule", "status", "--recursive"))
    for line in output.splitlines():
        match = re.fullmatch(r"(.)([0-9a-f]{40}) ([^ ]+)(?: .*)?", line)
        if match is None:
            raise SourceLockError("source submodule status is invalid")
        state, revision, path = match.groups()
        _safe_relative(path, "submodule path")
        if path not in expected or revision != expected[path]:
            raise SourceLockError("source submodule identity does not match the lock")
        if state == "-":
            missing.append(path)
        elif state != " ":
            raise SourceLockError("source submodule checkout is modified or conflicted")
        else:
            submodule = root / Path(*PurePosixPath(path).parts)
            if submodule.is_symlink() or not submodule.is_dir():
                raise SourceLockError("source initialized submodule is not a real directory")
            _require_origin(
                submodule, expected_repositories[path], "submodule origin repository"
            )
            if _git(
                submodule,
                (
                    "status", "--porcelain=v1", "--untracked-files=all",
                    "--ignore-submodules=all",
                ),
            ).strip():
                raise SourceLockError(
                    "source submodule checkout is dirty or has untracked files: %s"
                    % path
                )
        observed[path] = revision
    if set(observed) != set(expected):
        raise SourceLockError("source submodule set does not match the lock")
    missing.sort()
    if require_submodules and missing:
        raise SourceLockError("source submodule checkout is not initialized: %s" % missing)
    return {
        "commit": commit,
        "file_sha256": hashes,
        "missing_submodules": missing,
        "repository": value["repository"],
        "schema": "wam_external_source_checkout_v1",
        "submodules": observed,
        "tree": tree,
    }


def snapshot_paths(checkout: Union[str, Path]) -> Tuple[str, ...]:
    root = Path(checkout)
    if root.is_symlink() or not root.is_dir():
        raise SourceLockError("source checkout must be a real directory")
    selected: List[str] = []
    for directory, dirnames, filenames in os.walk(str(root), followlinks=False):
        base = Path(directory)
        kept = []
        for name in dirnames:
            candidate = base / name
            if name == ".git":
                continue
            if candidate.is_symlink():
                raise SourceLockError("source snapshot contains a symlink directory")
            if not candidate.is_dir():
                raise SourceLockError("source snapshot contains a non-directory")
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            if name == ".git":
                continue
            candidate = base / name
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise SourceLockError("source snapshot contains a symlink")
            if not stat.S_ISREG(info.st_mode):
                raise SourceLockError("source snapshot contains a non-regular file")
            selected.append(candidate.relative_to(root).as_posix())
    return tuple(sorted(selected))


def _safe_git_tree_path(value: str) -> str:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SourceLockError("source Git tree path contains control characters")
    relative = _safe_relative(value, "Git tree path")
    parts = PurePosixPath(relative).parts
    if not parts or any(part in ("", ".", "..", ".git") for part in parts):
        raise SourceLockError("source Git tree path is unsafe")
    return relative


def _git_tree_entries(checkout: Path, commit: str) -> List[Tuple[str, str, str, str]]:
    raw = _git_bytes(
        checkout, ("ls-tree", "-r", "-z", "--full-tree", commit)
    )
    entries: List[Tuple[str, str, str, str]] = []
    seen = set()
    try:
        for record in (item for item in raw.split(b"\x00") if item):
            header, path_raw = record.split(b"\t", 1)
            mode_raw, kind_raw, object_raw = header.split(b" ", 2)
            mode = mode_raw.decode("ascii")
            kind = kind_raw.decode("ascii")
            object_id = object_raw.decode("ascii")
            path = _safe_git_tree_path(path_raw.decode("utf-8"))
            if not _HEX40.fullmatch(object_id):
                raise ValueError("invalid Git object id")
            if path in seen:
                raise ValueError("duplicate Git tree path")
            seen.add(path)
            entries.append((mode, kind, object_id, path))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SourceLockError("source Git tree listing is invalid") from exc
    return sorted(entries, key=lambda row: row[3])


def _join_git_path(prefix: str, relative: str) -> str:
    return relative if not prefix else "%s/%s" % (prefix, relative)


def _git_blob(checkout: Path, object_id: str) -> bytes:
    data = _git_bytes(checkout, ("cat-file", "blob", object_id))
    digest = hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\x00" + data
    ).hexdigest()
    if digest != object_id:
        raise SourceLockError("source Git blob content does not match its object id")
    return data


def materialize_pinned_source(
    checkout: Union[str, Path],
    destination: Union[str, Path],
    lock: Dict[str, object],
) -> Dict[str, object]:
    """Materialize only bytes addressed by locked Git commit objects."""
    value = validate_source_lock(lock)
    root = Path(checkout)
    target_root = Path(destination) / "source"
    if target_root.exists() or target_root.is_symlink():
        raise SourceLockError("source snapshot destination already exists")
    target_root.mkdir(mode=0o755)
    modules = {row["path"]: row for row in value["submodules"]}
    visited = set()
    expected_rows: List[Dict[str, object]] = []

    def materialize_repository(repository: Path, commit: str, prefix: str) -> None:
        if repository.is_symlink() or not repository.is_dir():
            raise SourceLockError("source Git object repository is unsafe")
        for mode, kind, object_id, relative in _git_tree_entries(repository, commit):
            full_path = _join_git_path(prefix, relative)
            if mode == "160000":
                if kind != "commit":
                    raise SourceLockError("source Gitlink has the wrong object type")
                module = modules.get(full_path)
                if module is None or module["commit"] != object_id:
                    raise SourceLockError(
                        "source Gitlink does not match the locked submodule"
                    )
                if full_path in visited:
                    raise SourceLockError("source submodule Gitlink is duplicated")
                visited.add(full_path)
                materialize_repository(
                    root / Path(*PurePosixPath(full_path).parts),
                    object_id,
                    full_path,
                )
                continue
            if mode == "120000":
                raise SourceLockError("source Git tree contains a symlink")
            if mode not in ("100644", "100755") or kind != "blob":
                raise SourceLockError(
                    "source Git tree contains an unsupported file mode or object type"
                )
            target = target_root / Path(*PurePosixPath(full_path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = _git_blob(repository, object_id)
            with target.open("xb") as handle:
                handle.write(data)
                os.fchmod(handle.fileno(), 0o755 if mode == "100755" else 0o644)
                handle.flush()
                os.fsync(handle.fileno())
            expected_rows.append({
                "mode": "0755" if mode == "100755" else "0644",
                "path": full_path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            })

    materialize_repository(root, value["commit"], "")
    if visited != set(modules):
        raise SourceLockError("source locked submodule set does not match Gitlinks")
    rows = sorted(expected_rows, key=lambda row: str(row["path"]))
    return {
        "files": rows,
        "schema": SOURCE_INVENTORY_SCHEMA,
        "sha256": hash_source_inventory(rows),
    }


def hash_source_inventory(rows: object) -> str:
    if not isinstance(rows, list):
        raise SourceLockError("source inventory files are invalid")
    previous = None
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "mode", "path", "sha256", "size",
        }:
            raise SourceLockError("source inventory row is invalid")
        path = _safe_relative(row.get("path"), "source inventory path")
        if previous is not None and path <= previous:
            raise SourceLockError("source inventory paths are not unique and sorted")
        previous = path
        if (
            row.get("mode") not in ("0644", "0755")
            or not isinstance(row.get("sha256"), str)
            or not _HEX64.fullmatch(row["sha256"])
            or not isinstance(row.get("size"), int)
            or row["size"] < 0
        ):
            raise SourceLockError("source inventory value is invalid")
    payload = {"files": rows, "schema": SOURCE_INVENTORY_SCHEMA}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_source_inventory(source_root: Union[str, Path]) -> Dict[str, object]:
    root = Path(source_root)
    paths = snapshot_paths(root)
    manifest = contract.build_manifest(
        root,
        paths,
        artifact_kind="openpi_source_inventory",
        source_role="git_objects",
    )
    rows = manifest["payload"]["files"]
    return {
        "files": rows,
        "schema": SOURCE_INVENTORY_SCHEMA,
        "sha256": hash_source_inventory(rows),
    }


def _manifest_source_inventory(manifest: Dict[str, object]) -> Dict[str, object]:
    payload = manifest.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise SourceLockError("source artifact manifest is invalid")
    rows = []
    for row in payload["files"]:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise SourceLockError("source artifact manifest file row is invalid")
        if not row["path"].startswith("source/"):
            continue
        source_row = dict(row)
        source_row["path"] = row["path"][len("source/"):]
        rows.append(source_row)
    return {
        "files": rows,
        "schema": SOURCE_INVENTORY_SCHEMA,
        "sha256": hash_source_inventory(rows),
    }


def verify_source_release(
    release: Union[str, Path], lock: Dict[str, object]
) -> Dict[str, object]:
    value = validate_source_lock(lock)
    root = Path(release)
    envelope = contract.verify_release(root)
    payload = envelope["payload"]
    if payload.get("artifact_kind") != "openpi_source" or payload.get("source_role") != "mac":
        raise SourceLockError("openpi source artifact role or kind is invalid")
    locked_raw = (root / "OPENPI_SOURCE_LOCK.json").read_bytes()
    if locked_raw != canonical_json_bytes(value):
        raise SourceLockError("openpi source release does not contain the exact lock")
    provenance_raw = (root / "SOURCE_PROVENANCE.json").read_bytes()
    try:
        provenance = json.loads(provenance_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceLockError("openpi source provenance is invalid") from exc
    source_inventory = build_source_inventory(root / "source")
    expected_provenance = {
        "commit": value["commit"],
        "environment_boundary": value["environment_boundary"],
        "lock_sha256": hashlib.sha256(locked_raw).hexdigest(),
        "name": "openpi",
        "policy_adapter_status": value["policy_adapter_status"],
        "repository": value["repository"],
        "schema": PROVENANCE_SCHEMA,
        "source_inventory": source_inventory,
        "submodules": {row["path"]: row["commit"] for row in value["submodules"]},
        "tree": value["tree"],
    }
    if provenance != expected_provenance or provenance_raw != canonical_json_bytes(provenance):
        raise SourceLockError("openpi source provenance does not bind the lock")
    for row in value["files"]:
        if sha256_file(root / "source" / Path(*PurePosixPath(row["path"]).parts)) != row["sha256"]:
            raise SourceLockError("openpi released source identity file is invalid")
    if any(path.name == ".git" for path in root.rglob(".git")):
        raise SourceLockError("openpi source release contains Git administrative state")
    marker = json.loads((root / contract.COMPLETE_NAME).read_text(encoding="utf-8"))
    if marker.get("target_role") != "h200":
        raise SourceLockError("openpi source completion marker has the wrong target role")
    return envelope


def _directory_error(path: object, exc: OSError) -> SourceLockError:
    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
        return SourceLockError(
            "openpi source path contains a symlink or non-directory ancestor: %s" % path
        )
    return SourceLockError("openpi source directory is unsafe: %s" % path)


def _open_directory_at(parent_fd: int, name: str, *, create: bool) -> int:
    if not name or "/" in name or name in (".", ".."):
        raise SourceLockError("openpi source directory component is invalid")
    created = False
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise _directory_error(name, exc) from exc
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise _directory_error(name, exc) from exc
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise SourceLockError("openpi source path component is not a directory")
    if created:
        os.fsync(parent_fd)
    return descriptor


@contextmanager
def _open_canonical_directory(
    path: Path, *, create: bool
) -> Iterator[int]:
    pure = PurePosixPath(path.as_posix())
    if not pure.is_absolute() or pure.as_posix() != path.as_posix():
        raise SourceLockError("openpi source directory must be canonical and absolute")
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in pure.parts[1:]:
            next_descriptor = _open_directory_at(
                descriptor, component, create=create
            )
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor
    finally:
        os.close(descriptor)


def _descriptor_path(descriptor: int) -> Path:
    if sys.platform.startswith("linux"):
        try:
            return Path(os.readlink("/proc/self/fd/%s" % descriptor))
        except OSError as exc:
            raise SourceLockError("cannot resolve the verified source directory") from exc
    if sys.platform == "darwin":
        try:
            raw = fcntl.fcntl(descriptor, 50, b"\x00" * 1024)
            return Path(raw.split(b"\x00", 1)[0].decode("utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise SourceLockError("cannot resolve the verified source directory") from exc
    raise SourceLockError(
        "descriptor-bound source installation is unsupported on %s" % sys.platform
    )


@contextmanager
def _source_deployment_lock(source_root_fd: int) -> Iterator[None]:
    flags = (
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(
            _SOURCE_LOCK_NAME, flags, 0o600, dir_fd=source_root_fd
        )
    except OSError as exc:
        raise SourceLockError("openpi source deployment lock is unsafe") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SourceLockError("openpi source deployment lock is not a private file")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _rename_noreplace_at(
    source_fd: int, source_name: str, destination_fd: int, destination_name: str
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise SourceLockError("renameat2 is required for source publication")
        operation.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(source_fd, source, destination_fd, destination, 1)
    elif sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        if operation is None:
            raise SourceLockError("renameatx_np is required for source publication")
        operation.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(source_fd, source, destination_fd, destination, 0x00000004)
    else:
        raise SourceLockError(
            "atomic source publication is unsupported on %s" % sys.platform
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise contract.ConflictError("source release appeared during publication")
    raise OSError(error, os.strerror(error), destination_name)


def _try_open_release(releases_fd: int, release_id: str) -> Optional[int]:
    try:
        return _open_directory_at(releases_fd, release_id, create=False)
    except SourceLockError as exc:
        cause = exc.__cause__
        if isinstance(cause, OSError) and cause.errno == errno.ENOENT:
            return None
        raise


def _remove_work_directory(source_root_fd: int, name: str, descriptor: int) -> None:
    path = _descriptor_path(descriptor)
    for child in path.iterdir():
        if child.is_symlink() or not child.is_dir():
            child.unlink(missing_ok=True)
        else:
            shutil.rmtree(str(child), ignore_errors=True)
    expected = os.fstat(descriptor)
    observed = os.stat(name, dir_fd=source_root_fd, follow_symlinks=False)
    os.close(descriptor)
    if (expected.st_dev, expected.st_ino) != (observed.st_dev, observed.st_ino):
        raise SourceLockError("openpi source work directory identity changed")
    os.rmdir(name, dir_fd=source_root_fd)


def install_source_candidate(lock: Dict[str, object]) -> Path:
    value = validate_source_lock(lock)
    releases = Path(value["releases_root"])
    source_root = releases.parent
    with _open_canonical_directory(source_root, create=True) as source_root_fd:
        with _source_deployment_lock(source_root_fd):
            releases_fd = _open_directory_at(
                source_root_fd, releases.name, create=True
            )
            work_name = ".openpi-source-build-%s" % uuid.uuid4().hex
            os.mkdir(work_name, mode=0o700, dir_fd=source_root_fd)
            work_fd = _open_directory_at(
                source_root_fd, work_name, create=False
            )
            try:
                work = _descriptor_path(work_fd)
                checkout = work / "checkout"
                snapshot = work / "snapshot"
                completed_release = work / "completed-release"
                snapshot.mkdir()
                cloned = subprocess.run(
                    [
                        "git", "clone", "--no-checkout", "--",
                        value["repository"], str(checkout),
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=False, text=True,
                )
                if cloned.returncode != 0:
                    raise SourceLockError(
                        "openpi clone failed: %s" % cloned.stderr.strip()
                    )
                _git(checkout, ("checkout", "--detach", value["commit"]))
                if value["submodules"]:
                    _git(
                        checkout,
                        ("submodule", "update", "--init", "--recursive"),
                    )
                identity = verify_checkout(
                    checkout, value, require_submodules=True
                )
                pinned_inventory = materialize_pinned_source(
                    checkout, snapshot, value
                )
                if build_source_inventory(snapshot / "source") != pinned_inventory:
                    raise SourceLockError(
                        "source snapshot inventory does not match pinned Git objects"
                    )
                lock_raw = canonical_json_bytes(value)
                (snapshot / "OPENPI_SOURCE_LOCK.json").write_bytes(lock_raw)
                provenance = {
                    "commit": identity["commit"],
                    "environment_boundary": value["environment_boundary"],
                    "lock_sha256": hashlib.sha256(lock_raw).hexdigest(),
                    "name": "openpi",
                    "policy_adapter_status": value["policy_adapter_status"],
                    "repository": value["repository"],
                    "schema": PROVENANCE_SCHEMA,
                    "source_inventory": pinned_inventory,
                    "submodules": identity["submodules"],
                    "tree": identity["tree"],
                }
                (snapshot / "SOURCE_PROVENANCE.json").write_bytes(
                    canonical_json_bytes(provenance)
                )
                paths = snapshot_paths(snapshot)
                manifest = contract.build_manifest(
                    snapshot, paths, artifact_kind="openpi_source",
                    source_role="mac",
                )
                if _manifest_source_inventory(manifest) != pinned_inventory:
                    raise SourceLockError(
                        "source artifact inventory does not match pinned Git objects"
                    )
                release_id = manifest["release_id"]
                final_fd = _try_open_release(releases_fd, release_id)
                if final_fd is None:
                    contract.write_release(
                        snapshot, manifest, completed_release,
                        target_role="h200",
                    )
                    verify_source_release(completed_release, value)
                    try:
                        _rename_noreplace_at(
                            work_fd, completed_release.name,
                            releases_fd, release_id,
                        )
                        os.fsync(releases_fd)
                    except contract.ConflictError:
                        pass
                    final_fd = _try_open_release(releases_fd, release_id)
                    if final_fd is None:
                        raise SourceLockError(
                            "openpi source publication did not create the release"
                        )
                try:
                    verify_source_release(_descriptor_path(final_fd), value)
                finally:
                    os.close(final_fd)
                with _open_canonical_directory(
                    releases, create=False
                ) as observed_releases_fd:
                    expected_info = os.fstat(releases_fd)
                    observed_info = os.fstat(observed_releases_fd)
                    if (
                        expected_info.st_dev, expected_info.st_ino
                    ) != (
                        observed_info.st_dev, observed_info.st_ino
                    ):
                        raise SourceLockError(
                            "openpi canonical release root changed during installation"
                        )
                return releases / release_id
            finally:
                try:
                    _remove_work_directory(source_root_fd, work_name, work_fd)
                finally:
                    os.close(releases_fd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify or install the locked openpi source")
    parser.add_argument("--lock", default=str(LOCK_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify-checkout")
    verify_parser.add_argument("--checkout", required=True)
    subparsers.add_parser("install")
    args = parser.parse_args(argv)
    lock = load_source_lock(args.lock)
    if args.command == "verify-checkout":
        result = verify_checkout(args.checkout, lock, require_submodules=True)
    else:
        release = install_source_candidate(lock)
        envelope = verify_source_release(release, lock)
        result = {
            "final_release": release.as_posix(),
            "payload_sha256": envelope["payload_sha256"],
            "release_id": envelope["release_id"],
            "schema": "wam_openpi_source_install_response_v1",
            "target_role": "h200",
        }
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
