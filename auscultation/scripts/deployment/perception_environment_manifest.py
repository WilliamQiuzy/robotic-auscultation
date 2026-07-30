"""Canonical, fail-closed manifests for cloned Linbox Python environments.

The RECORD observation intentionally preserves every CSV row.  It never
pretends that a conda-cloned environment is equivalent to a clean wheel
install: expected wheel metadata and observed installed bytes are separate
fields, and any accepted console-script relocation carries a byte-level proof.

The tree manifest is independent of RECORD ownership and covers the complete
environment namespace.  Only two circular metadata files are omitted from the
tree hash.  The one audited NumPy bytecode cache is not ignored: it is required
to be absent and is represented explicitly by the RECORD observation policy.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import artifact_contract as contract


RECORD_OBSERVATION_ALGORITHM = (
    "sha256-canonical-json-normalized-distributions-record-claims-expected-actual-v1"
)
ENVIRONMENT_TREE_ALGORITHM = (
    "sha256-canonical-json-complete-environment-tree-bytes-mode-symlinks-v1"
)
SELF_REFERENCE_EXCLUSIONS = (
    "PERCEPTION_ENVIRONMENT_CLONE_COMPLETE.json",
    "PERCEPTION_ENVIRONMENT_STRONG_LOCK.json",
)
GENERATED_PYC_OMISSION = (
    "lib/python3.11/site-packages/numpy/distutils/__pycache__/"
    "conv_template.cpython-311.pyc"
)
RECORD_STATUSES = (
    "exact",
    "metadata_mismatch",
    "missing",
    "omitted_generated_pyc",
    "record_unhashed",
    "relocation_mismatch",
)

_NORMALIZE_NAME = re.compile(r"[-_.]+")


class ManifestError(RuntimeError):
    """An environment cannot be represented by the strict manifest contract."""


def canonical_json_bytes(value: object) -> bytes:
    """Expose the repository's canonical JSON encoding to callers and tests."""

    return contract.canonical_json_bytes(value)


def _normalize_distribution_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError("distribution name must be a non-empty string")
    normalized = _NORMALIZE_NAME.sub("-", value).lower()
    if not normalized:
        raise ManifestError("distribution name normalizes to empty")
    return normalized


def _canonical_relative(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
    ):
        raise ManifestError("%s is not a canonical relative POSIX path" % label)
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or candidate.as_posix() != value:
        raise ManifestError("%s is not a canonical relative POSIX path" % label)
    return value


def _environment_relative(
    path: Union[str, Path], root: Path, label: str,
) -> Tuple[Path, str]:
    candidate = Path(os.path.abspath(str(path)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError("%s escapes the environment root" % label) from exc
    if not relative.parts:
        raise ManifestError("%s resolves to the environment root" % label)
    canonical = PurePosixPath(*relative.parts).as_posix()
    return candidate, canonical


def _read_regular(path: Path, label: str) -> Tuple[bytes, os.stat_result]:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ManifestError("%s is missing" % label) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ManifestError("%s is not a regular file" % label)
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ManifestError("%s inode changed while opening" % label)
            chunks = []
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                chunks.append(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ManifestError("%s cannot be read" % label) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        getattr(before, "st_mtime_ns", int(before.st_mtime * 1_000_000_000)),
        stat.S_IMODE(before.st_mode),
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        getattr(after, "st_mtime_ns", int(after.st_mtime * 1_000_000_000)),
        stat.S_IMODE(after.st_mode),
    )
    if before_identity != after_identity:
        raise ManifestError("%s changed while hashing" % label)
    return b"".join(chunks), after


def _decode_record_sha256(hash_field: str) -> Optional[str]:
    if hash_field == "":
        return None
    if not hash_field.startswith("sha256="):
        return None
    encoded = hash_field[len("sha256="):]
    if not encoded:
        return None
    try:
        decoded = base64.urlsafe_b64decode(
            encoded.encode("ascii") + b"=" * (-len(encoded) % 4)
        )
    except (UnicodeEncodeError, ValueError):
        return None
    if len(decoded) != hashlib.sha256().digest_size:
        return None
    return decoded.hex()


def _parse_record_size(size_field: str) -> Optional[int]:
    if size_field == "":
        return None
    if not size_field.isascii() or not size_field.isdigit():
        return None
    value = int(size_field)
    return value if value >= 0 else None


def generated_pyc_omission_evidence(
    *, path: str, source_path: str, source_sha256: str,
    inspection_sha256: str,
) -> dict:
    """Build the small, canonical identity used to authorize one omission."""

    if path != GENERATED_PYC_OMISSION:
        raise ManifestError("generated pyc omission path is unsupported")
    _canonical_relative(source_path, "generated pyc source path")
    for value, label in (
        (source_sha256, "generated pyc source SHA-256"),
        (inspection_sha256, "generated pyc inspection SHA-256"),
    ):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ManifestError("%s is invalid" % label)
    return {
        "inspection_sha256": inspection_sha256,
        "path": path,
        "required_service_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
        "source_path": source_path,
        "source_sha256": source_sha256,
    }


def _strict_relocation_proof(
    *, record_index: int, record_path: str, hash_field: str, size_field: str,
    installed_path: str, actual: bytes, expected_sha256: Optional[str],
    expected_size: Optional[int], environment_root: Path,
    relocation_source_root: Optional[Path],
    source_record_rows: Optional[Sequence[Sequence[str]]],
    source_record_sha256: Optional[str],
) -> Optional[dict]:
    if relocation_source_root is None:
        return None
    parts = PurePosixPath(installed_path).parts
    if len(parts) < 2 or parts[0] != "bin":
        return None
    if expected_sha256 is None or expected_size is None:
        return None
    if (
        source_record_rows is None
        or source_record_sha256 is None
        or record_index < 1
        or record_index > len(source_record_rows)
        or list(source_record_rows[record_index - 1])
        != [record_path, hash_field, size_field]
    ):
        return None
    source_path = relocation_source_root / Path(*parts)
    try:
        source, source_info = _read_regular(
            source_path, "relocation source %s" % installed_path,
        )
    except ManifestError:
        return None
    source_first, source_separator, source_rest = source.partition(b"\n")
    actual_first, actual_separator, actual_rest = actual.partition(b"\n")
    if source_separator != b"\n" or actual_separator != b"\n":
        return None
    old_prefix = os.fsencode(str(relocation_source_root))
    new_prefix = os.fsencode(str(environment_root))
    shebang_prefix = b"#!" + old_prefix
    if not source_first.startswith(shebang_prefix):
        return None
    if source_first.count(old_prefix) != 1:
        return None
    suffix = source_first[len(shebang_prefix):]
    if not suffix.startswith(b"/bin/") or b"\x00" in suffix or b"\r" in suffix:
        return None
    if actual_first != b"#!" + new_prefix + suffix:
        return None
    if actual_first.count(new_prefix) != 1 or actual_rest != source_rest:
        return None
    return {
        "interpreter_suffix": os.fsdecode(suffix),
        "new_prefix": str(environment_root),
        "old_prefix": str(relocation_source_root),
        "source_record_index": record_index,
        "source_record_sha256": source_record_sha256,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_size": source_info.st_size,
    }


def _observe_claim(
    *, record_index: int, record_path: str, hash_field: str,
    size_field: str, installed_path: str, environment_root: Path,
    relocation_source_root: Optional[Path],
    source_record_rows: Optional[Sequence[Sequence[str]]],
    source_record_sha256: Optional[str],
    omitted_generated_pyc: Mapping[str, Mapping[str, object]],
) -> dict:
    expected_sha256 = _decode_record_sha256(hash_field)
    expected_size = _parse_record_size(size_field)
    expected = {
        "hash_field": hash_field,
        "sha256": expected_sha256,
        "size": expected_size,
        "size_field": size_field,
    }
    absolute = environment_root / Path(*PurePosixPath(installed_path).parts)
    if not os.path.lexists(str(absolute)):
        claim = {
            "actual": {"sha256": None, "size": None},
            "expected": expected,
            "installed_path": installed_path,
            "record_index": record_index,
            "record_path": record_path,
            "status": "missing",
        }
        evidence = omitted_generated_pyc.get(installed_path)
        if evidence is not None:
            if installed_path != GENERATED_PYC_OMISSION:
                raise ManifestError("generated pyc omission path is unsupported")
            expected_evidence = generated_pyc_omission_evidence(
                path=evidence.get("path"),
                source_path=evidence.get("source_path"),
                source_sha256=evidence.get("source_sha256"),
                inspection_sha256=evidence.get("inspection_sha256"),
            )
            if dict(evidence) != expected_evidence:
                raise ManifestError("generated pyc omission evidence is noncanonical")
            source = environment_root / Path(
                *PurePosixPath(expected_evidence["source_path"]).parts
            )
            source_bytes, _source_info = _read_regular(
                source, "generated pyc source",
            )
            if hashlib.sha256(source_bytes).hexdigest() != expected_evidence[
                "source_sha256"
            ]:
                raise ManifestError("generated pyc source SHA-256 drift")
            claim["status"] = "omitted_generated_pyc"
            claim["omission_evidence_sha256"] = hashlib.sha256(
                canonical_json_bytes(expected_evidence)
            ).hexdigest()
        return claim
    data, info = _read_regular(absolute, "installed path %s" % installed_path)
    actual_sha256 = hashlib.sha256(data).hexdigest()
    actual = {"sha256": actual_sha256, "size": info.st_size}
    if hash_field == "" and size_field == "":
        status = "record_unhashed"
        relocation = None
    elif (
        expected_sha256 is not None
        and expected_size is not None
        and actual_sha256 == expected_sha256
        and info.st_size == expected_size
    ):
        status = "exact"
        relocation = None
    else:
        relocation = _strict_relocation_proof(
            record_index=record_index,
            record_path=record_path,
            hash_field=hash_field,
            size_field=size_field,
            installed_path=installed_path,
            actual=data,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            environment_root=environment_root,
            relocation_source_root=relocation_source_root,
            source_record_rows=source_record_rows,
            source_record_sha256=source_record_sha256,
        )
        status = "relocation_mismatch" if relocation is not None else "metadata_mismatch"
    claim = {
        "actual": actual,
        "expected": expected,
        "installed_path": installed_path,
        "record_index": record_index,
        "record_path": record_path,
        "status": status,
    }
    if relocation is not None:
        claim["relocation"] = relocation
    return claim


def collect_record_observation(
    distributions: Iterable[object], *, environment_root: Union[str, Path],
    relocation_source_root: Optional[Union[str, Path]] = None,
    omitted_generated_pyc: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> dict:
    """Observe every RECORD row without treating clone drift as wheel identity."""

    root_input = Path(environment_root)
    if root_input.is_symlink() or not root_input.is_dir():
        raise ManifestError("environment root must be a real directory")
    root = root_input.resolve(strict=True)
    source_root = None
    if relocation_source_root is not None:
        source_input = Path(relocation_source_root)
        if source_input.is_symlink() or not source_input.is_dir():
            raise ManifestError("relocation source root must be a real directory")
        source_root = source_input.resolve(strict=True)
    omissions = {} if omitted_generated_pyc is None else dict(omitted_generated_pyc)
    if set(omissions) - {GENERATED_PYC_OMISSION}:
        raise ManifestError("generated pyc omission set is unsupported")

    distribution_rows = []
    seen_names = set()
    for distribution in distributions:
        name = _normalize_distribution_name(distribution.metadata.get("Name"))
        if name in seen_names:
            raise ManifestError("duplicate normalized distribution: %s" % name)
        seen_names.add(name)
        version = distribution.version
        if not isinstance(version, str) or not version:
            raise ManifestError("distribution version is invalid: %s" % name)
        private_path = getattr(distribution, "_path", None)
        location = (
            Path(private_path)
            if private_path is not None
            else Path(distribution.locate_file(""))
        )
        record_path = location / "RECORD"
        _record_absolute, record_relative = _environment_relative(
            record_path, root, "%s RECORD" % name,
        )
        record_bytes, _record_info = _read_regular(
            record_path, "%s RECORD" % name,
        )
        try:
            record_text = record_bytes.decode("utf-8")
            csv_rows = list(csv.reader(io.StringIO(record_text, newline="")))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise ManifestError("%s RECORD is not valid UTF-8 CSV" % name) from exc
        if not csv_rows:
            raise ManifestError("%s RECORD is empty" % name)
        source_record_rows = None
        source_record_sha256 = None
        if source_root is not None:
            source_record = source_root / Path(
                *PurePosixPath(record_relative).parts
            )
            try:
                source_record_bytes, _source_record_info = _read_regular(
                    source_record, "%s source RECORD" % name,
                )
                source_record_text = source_record_bytes.decode("utf-8")
                source_record_rows = list(csv.reader(
                    io.StringIO(source_record_text, newline="")
                ))
                if not source_record_rows or any(
                    len(row) != 3 for row in source_record_rows
                ):
                    source_record_rows = None
                else:
                    source_record_sha256 = hashlib.sha256(
                        source_record_bytes
                    ).hexdigest()
            except (ManifestError, UnicodeDecodeError, csv.Error):
                source_record_rows = None
                source_record_sha256 = None
        claims = []
        for index, row in enumerate(csv_rows, 1):
            if len(row) != 3:
                raise ManifestError("%s RECORD row %d is invalid" % (name, index))
            raw_path, hash_field, size_field = row
            _canonical_relative(raw_path, "%s RECORD row %d" % (name, index))
            located = distribution.locate_file(raw_path)
            _absolute, installed_relative = _environment_relative(
                located, root, "%s RECORD row %d" % (name, index),
            )
            claims.append(_observe_claim(
                record_index=index,
                record_path=raw_path,
                hash_field=hash_field,
                size_field=size_field,
                installed_path=installed_relative,
                environment_root=root,
                relocation_source_root=source_root,
                source_record_rows=source_record_rows,
                source_record_sha256=source_record_sha256,
                omitted_generated_pyc=omissions,
            ))
        distribution_rows.append({
            "claims": claims,
            "name": name,
            "record_entries_count": len(csv_rows),
            "record_path": record_relative,
            "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
            "version": version,
        })
    distribution_rows.sort(key=lambda row: row["name"])
    status_counts = {status: 0 for status in RECORD_STATUSES}
    for row in distribution_rows:
        for claim in row["claims"]:
            status_counts[claim["status"]] += 1
    projection = {
        "distributions": distribution_rows,
        "status_counts": status_counts,
    }
    return {
        "algorithm": RECORD_OBSERVATION_ALGORITHM,
        "claim_count": sum(status_counts.values()),
        "distribution_count": len(distribution_rows),
        "distributions": distribution_rows,
        "sha256": hashlib.sha256(canonical_json_bytes(projection)).hexdigest(),
        "status_counts": status_counts,
    }


def _mode(info: os.stat_result) -> str:
    return format(stat.S_IMODE(info.st_mode), "04o")


def _relative_from_root(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return PurePosixPath(*relative.parts).as_posix() if relative.parts else "."


def _symlink_row(path: Path, root: Path, info: os.stat_result) -> dict:
    relative = _relative_from_root(path, root)
    try:
        target = os.readlink(str(path))
    except OSError as exc:
        raise ManifestError("environment symlink cannot be read: %s" % relative) from exc
    if not target or os.path.isabs(target):
        raise ManifestError("environment symlink uses an absolute or empty target: %s" % relative)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ManifestError("environment symlink is dangling or cyclic: %s" % relative) from exc
    try:
        resolved_relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ManifestError("environment symlink escapes its root: %s" % relative) from exc
    resolved_path = (
        PurePosixPath(*resolved_relative.parts).as_posix()
        if resolved_relative.parts else "."
    )
    return {
        "kind": "symlink",
        "mode": _mode(info),
        "path": relative,
        "resolved_path": resolved_path,
        "target": target,
    }


def collect_environment_tree(
    environment_root: Union[str, Path], *, allow_generated_pyc: bool = False,
    allow_hardlinks: bool = False,
) -> dict:
    """Hash every allowed entry in an environment, including mode and links."""

    root_input = Path(environment_root)
    if root_input.is_symlink() or not root_input.is_dir():
        raise ManifestError("environment root must be a real directory")
    root = root_input.resolve(strict=True)
    if not allow_generated_pyc:
        for relative in (GENERATED_PYC_OMISSION,):
            if os.path.lexists(str(root / Path(*PurePosixPath(relative).parts))):
                raise ManifestError("required absent generated cache is present: %s" % relative)
    self_references = set(SELF_REFERENCE_EXCLUSIONS)
    rows = []
    root_before = root.lstat()
    rows.append({"kind": "directory", "mode": _mode(root_before), "path": "."})
    inode_rows = set()
    hardlink_rows = {}
    for raw_directory, dirnames, filenames in os.walk(
        str(root), topdown=True, followlinks=False,
    ):
        directory = Path(raw_directory)
        before = directory.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ManifestError("tree directory changed while traversing")
        names = sorted(set(dirnames + filenames))
        if len(names) != len(dirnames) + len(filenames):
            raise ManifestError("tree contains conflicting entry names")
        for name in names:
            path = directory / name
            relative = _relative_from_root(path, root)
            try:
                info = path.lstat()
            except FileNotFoundError as exc:
                raise ManifestError("tree changed while traversing: %s" % relative) from exc
            if relative in self_references:
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    raise ManifestError("self-reference exclusion is not a regular file")
                if name in dirnames:
                    dirnames.remove(name)
                continue
            if stat.S_ISLNK(info.st_mode):
                if name in dirnames:
                    dirnames.remove(name)
                rows.append(_symlink_row(path, root, info))
                continue
            if stat.S_ISDIR(info.st_mode):
                rows.append({
                    "kind": "directory",
                    "mode": _mode(info),
                    "path": relative,
                })
                continue
            if stat.S_ISREG(info.st_mode):
                if not allow_hardlinks and info.st_nlink != 1:
                    raise ManifestError(
                        "environment regular file uses a hardlink: %s" % relative
                    )
                data, opened = _read_regular(path, "tree file %s" % relative)
                inode = (opened.st_dev, opened.st_ino)
                if info.st_nlink != opened.st_nlink:
                    raise ManifestError(
                        "environment hardlink count changed while hashing: %s"
                        % relative
                    )
                if not allow_hardlinks and inode in inode_rows:
                    raise ManifestError("environment tree has duplicate hardlink inodes")
                inode_rows.add(inode)
                digest = hashlib.sha256(data).hexdigest()
                if allow_hardlinks:
                    hardlink = hardlink_rows.setdefault(inode, {
                        "identity": (_mode(opened), opened.st_size, digest),
                        "link_count": opened.st_nlink,
                        "paths": [],
                    })
                    if hardlink["link_count"] != opened.st_nlink:
                        raise ManifestError(
                            "environment hardlink count changed while traversing"
                        )
                    if hardlink["identity"] != (
                        _mode(opened), opened.st_size, digest,
                    ):
                        raise ManifestError(
                            "environment hardlink identity changed while traversing"
                        )
                    hardlink["paths"].append(relative)
                rows.append({
                    "kind": "regular",
                    "mode": _mode(opened),
                    "path": relative,
                    "sha256": digest,
                    "size": opened.st_size,
                })
                continue
            kind = (
                "FIFO" if stat.S_ISFIFO(info.st_mode)
                else "socket" if stat.S_ISSOCK(info.st_mode)
                else "device" if stat.S_ISCHR(info.st_mode) or stat.S_ISBLK(info.st_mode)
                else "non-regular"
            )
            raise ManifestError("environment tree contains a %s entry: %s" % (kind, relative))
        after = directory.lstat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            getattr(before, "st_mtime_ns", int(before.st_mtime * 1_000_000_000)),
            stat.S_IMODE(before.st_mode),
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            getattr(after, "st_mtime_ns", int(after.st_mtime * 1_000_000_000)),
            stat.S_IMODE(after.st_mode),
        )
        if before_identity != after_identity:
            raise ManifestError("tree directory changed while hashing: %s" % directory)
    root_after = root.lstat()
    if (
        root_before.st_dev,
        root_before.st_ino,
        stat.S_IMODE(root_before.st_mode),
    ) != (
        root_after.st_dev,
        root_after.st_ino,
        stat.S_IMODE(root_after.st_mode),
    ):
        raise ManifestError("environment root changed while hashing")
    rows.sort(key=lambda row: row["path"])
    exclusions = {
        "required_absent_generated_cache": (
            [] if allow_generated_pyc else [GENERATED_PYC_OMISSION]
        ),
        "self_referential": list(SELF_REFERENCE_EXCLUSIONS),
    }
    projection = {"entries": rows, "exclusions": exclusions}
    hardlinks = sorted(
        (
            {
                "link_count": row["link_count"],
                "paths": sorted(row["paths"]),
            }
            for row in hardlink_rows.values()
            if row["link_count"] > 1 or len(row["paths"]) > 1
        ),
        key=lambda row: row["paths"],
    )
    if allow_hardlinks:
        projection["hardlinks"] = hardlinks
    result = {
        "algorithm": ENVIRONMENT_TREE_ALGORITHM,
        "directories": sum(row["kind"] == "directory" for row in rows),
        "entries": rows,
        "entries_count": len(rows),
        "exclusions": exclusions,
        "regular_files": sum(row["kind"] == "regular" for row in rows),
        "sha256": hashlib.sha256(canonical_json_bytes(projection)).hexdigest(),
        "symlinks": sum(row["kind"] == "symlink" for row in rows),
    }
    if allow_hardlinks:
        result["hardlinks"] = hardlinks
    return result


def _exact_keys(value: object, expected: set, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ManifestError("%s must be an object" % label)
    actual = set(value)
    if actual != expected:
        raise ManifestError(
            "%s fields mismatch; missing=%r extra=%r"
            % (label, sorted(expected - actual), sorted(actual - expected))
        )
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ManifestError("%s is not a lowercase SHA-256" % label)
    return value


def validate_record_observation(
    value: object, *, require_resolved: bool = False,
) -> Mapping[str, object]:
    """Validate and recompute a complete per-claim RECORD observation."""

    observation = _exact_keys(value, {
        "algorithm", "claim_count", "distribution_count", "distributions",
        "sha256", "status_counts",
    }, "RECORD observation")
    if observation["algorithm"] != RECORD_OBSERVATION_ALGORITHM:
        raise ManifestError("RECORD observation algorithm is unsupported")
    distributions = observation["distributions"]
    if not isinstance(distributions, list) or not distributions:
        raise ManifestError("RECORD observation distributions must be a non-empty list")
    validated_distributions = []
    seen = set()
    for distribution_index, raw_distribution in enumerate(distributions, 1):
        distribution = _exact_keys(raw_distribution, {
            "claims", "name", "record_entries_count", "record_path",
            "record_sha256", "version",
        }, "RECORD distribution")
        name = _normalize_distribution_name(distribution["name"])
        if name != distribution["name"] or name in seen:
            raise ManifestError("RECORD distribution names are not canonical and unique")
        seen.add(name)
        version = distribution["version"]
        if not isinstance(version, str) or not version:
            raise ManifestError("RECORD distribution version is invalid")
        _canonical_relative(distribution["record_path"], "distribution RECORD path")
        _require_sha256(distribution["record_sha256"], "distribution RECORD SHA-256")
        claims = distribution["claims"]
        if not isinstance(claims, list) or not claims:
            raise ManifestError("RECORD distribution claims must be non-empty")
        if distribution["record_entries_count"] != len(claims):
            raise ManifestError("RECORD distribution entry count mismatch")
        for expected_index, raw_claim in enumerate(claims, 1):
            base_keys = {
                "actual", "expected", "installed_path", "record_index",
                "record_path", "status",
            }
            if not isinstance(raw_claim, Mapping):
                raise ManifestError("RECORD claim must be an object")
            status = raw_claim.get("status")
            expected_keys = set(base_keys)
            if status == "relocation_mismatch":
                expected_keys.add("relocation")
            if status == "omitted_generated_pyc":
                expected_keys.add("omission_evidence_sha256")
            claim = _exact_keys(raw_claim, expected_keys, "RECORD claim")
            if status not in RECORD_STATUSES:
                raise ManifestError("RECORD claim status is unsupported")
            if claim["record_index"] != expected_index:
                raise ManifestError("RECORD claim index is not sequential")
            _canonical_relative(claim["record_path"], "RECORD claim path")
            installed = _canonical_relative(
                claim["installed_path"], "installed RECORD path",
            )
            if any(part in (".", "..") for part in PurePosixPath(installed).parts):
                raise ManifestError("installed RECORD path is not environment-relative")
            expected = _exact_keys(claim["expected"], {
                "hash_field", "sha256", "size", "size_field",
            }, "RECORD expected metadata")
            if not isinstance(expected["hash_field"], str) or not isinstance(
                expected["size_field"], str
            ):
                raise ManifestError("RECORD expected raw fields are invalid")
            if expected["sha256"] != _decode_record_sha256(expected["hash_field"]):
                raise ManifestError("RECORD expected SHA-256 parsing drift")
            if expected["size"] != _parse_record_size(expected["size_field"]):
                raise ManifestError("RECORD expected size parsing drift")
            actual = _exact_keys(
                claim["actual"], {"sha256", "size"}, "RECORD actual metadata",
            )
            absent = status in {"missing", "omitted_generated_pyc"}
            if absent:
                if actual != {"sha256": None, "size": None}:
                    raise ManifestError("absent RECORD claim has actual metadata")
            else:
                _require_sha256(actual["sha256"], "RECORD actual SHA-256")
                if type(actual["size"]) is not int or actual["size"] < 0:
                    raise ManifestError("RECORD actual size is invalid")
            if status == "exact" and (
                expected["sha256"] != actual["sha256"]
                or expected["size"] != actual["size"]
            ):
                raise ManifestError("exact RECORD status contradicts metadata")
            if status == "record_unhashed" and (
                expected["hash_field"] != "" or expected["size_field"] != ""
            ):
                raise ManifestError("unhashed RECORD status contradicts metadata")
            if status == "relocation_mismatch":
                relocation = _exact_keys(claim["relocation"], {
                    "interpreter_suffix", "new_prefix", "old_prefix",
                    "source_record_index", "source_record_sha256",
                    "source_sha256", "source_size",
                }, "RECORD relocation proof")
                if (
                    not isinstance(relocation["old_prefix"], str)
                    or not Path(relocation["old_prefix"]).is_absolute()
                    or not isinstance(relocation["new_prefix"], str)
                    or not Path(relocation["new_prefix"]).is_absolute()
                    or not isinstance(relocation["interpreter_suffix"], str)
                    or not relocation["interpreter_suffix"].startswith("/bin/")
                ):
                    raise ManifestError("RECORD relocation prefixes are invalid")
                _require_sha256(
                    relocation["source_sha256"], "RECORD relocation source SHA-256",
                )
                _require_sha256(
                    relocation["source_record_sha256"],
                    "RECORD relocation source RECORD SHA-256",
                )
                if (
                    relocation["source_record_index"] != claim["record_index"]
                    or type(relocation["source_record_index"]) is not int
                    or type(relocation["source_size"]) is not int
                    or relocation["source_size"] < 0
                ):
                    raise ManifestError("RECORD relocation source identity is invalid")
            if status == "omitted_generated_pyc":
                if installed != GENERATED_PYC_OMISSION:
                    raise ManifestError("omitted generated pyc path is unsupported")
                _require_sha256(
                    claim["omission_evidence_sha256"],
                    "generated pyc omission evidence SHA-256",
                )
        validated_distributions.append(raw_distribution)
    if [row["name"] for row in distributions] != sorted(seen):
        raise ManifestError("RECORD distributions are not name-sorted")
    if observation["distribution_count"] != len(distributions):
        raise ManifestError("RECORD distribution count mismatch")
    status_counts = observation["status_counts"]
    if not isinstance(status_counts, Mapping) or set(status_counts) != set(RECORD_STATUSES):
        raise ManifestError("RECORD status counts have the wrong fields")
    recomputed_counts = {status: 0 for status in RECORD_STATUSES}
    for distribution in distributions:
        for claim in distribution["claims"]:
            recomputed_counts[claim["status"]] += 1
    if dict(status_counts) != recomputed_counts:
        raise ManifestError("RECORD status counts mismatch")
    if observation["claim_count"] != sum(recomputed_counts.values()):
        raise ManifestError("RECORD claim count mismatch")
    expected_fingerprint = hashlib.sha256(canonical_json_bytes({
        "distributions": distributions,
        "status_counts": recomputed_counts,
    })).hexdigest()
    if observation["sha256"] != expected_fingerprint:
        raise ManifestError("RECORD observation fingerprint mismatch")
    if require_resolved and (
        recomputed_counts["missing"] != 0
        or recomputed_counts["metadata_mismatch"] != 0
    ):
        raise ManifestError("unresolved RECORD statuses remain")
    return observation


def validate_environment_tree(value: object) -> Mapping[str, object]:
    """Validate and recompute a complete environment-tree manifest."""

    tree = _exact_keys(value, {
        "algorithm", "directories", "entries", "entries_count", "exclusions",
        "regular_files", "sha256", "symlinks",
    }, "environment tree")
    if tree["algorithm"] != ENVIRONMENT_TREE_ALGORITHM:
        raise ManifestError("environment tree algorithm is unsupported")
    exclusions = _exact_keys(tree["exclusions"], {
        "required_absent_generated_cache", "self_referential",
    }, "environment tree exclusions")
    expected_exclusions = {
        "required_absent_generated_cache": [GENERATED_PYC_OMISSION],
        "self_referential": list(SELF_REFERENCE_EXCLUSIONS),
    }
    if dict(exclusions) != expected_exclusions:
        raise ManifestError("environment tree exclusion policy is unsupported")
    entries = tree["entries"]
    if not isinstance(entries, list) or not entries:
        raise ManifestError("environment tree entries must be non-empty")
    paths = []
    counts = {"directory": 0, "regular": 0, "symlink": 0}
    for raw_row in entries:
        if not isinstance(raw_row, Mapping):
            raise ManifestError("environment tree row must be an object")
        kind = raw_row.get("kind")
        keys = {"kind", "mode", "path"}
        if kind == "regular":
            keys.update({"sha256", "size"})
        elif kind == "symlink":
            keys.update({"resolved_path", "target"})
        elif kind != "directory":
            raise ManifestError("environment tree row kind is unsupported")
        row = _exact_keys(raw_row, keys, "environment tree row")
        path = row["path"]
        if path != ".":
            _canonical_relative(path, "environment tree path")
            if any(part in (".", "..") for part in PurePosixPath(path).parts):
                raise ManifestError("environment tree path is not canonical")
        if not isinstance(row["mode"], str) or not re.fullmatch(r"0[0-7]{3}", row["mode"]):
            raise ManifestError("environment tree mode is invalid")
        if kind == "regular":
            _require_sha256(row["sha256"], "environment tree file SHA-256")
            if type(row["size"]) is not int or row["size"] < 0:
                raise ManifestError("environment tree file size is invalid")
        elif kind == "symlink":
            if (
                not isinstance(row["target"], str)
                or not row["target"]
                or os.path.isabs(row["target"])
            ):
                raise ManifestError("environment tree symlink target is unsafe")
            resolved = row["resolved_path"]
            if resolved != ".":
                _canonical_relative(resolved, "environment tree resolved symlink")
                if any(part in (".", "..") for part in PurePosixPath(resolved).parts):
                    raise ManifestError("environment tree resolved symlink is unsafe")
        paths.append(path)
        counts[kind] += 1
    if paths != sorted(paths) or len(paths) != len(set(paths)) or paths[0] != ".":
        raise ManifestError("environment tree paths are not sorted and unique")
    if tree["entries_count"] != len(entries):
        raise ManifestError("environment tree entry count mismatch")
    if (
        tree["directories"] != counts["directory"]
        or tree["regular_files"] != counts["regular"]
        or tree["symlinks"] != counts["symlink"]
    ):
        raise ManifestError("environment tree kind counts mismatch")
    expected_fingerprint = hashlib.sha256(canonical_json_bytes({
        "entries": entries,
        "exclusions": expected_exclusions,
    })).hexdigest()
    if tree["sha256"] != expected_fingerprint:
        raise ManifestError("environment tree fingerprint mismatch")
    return tree


def cross_check_record_tree(
    record_observation: object, environment_tree: object,
) -> None:
    """Require every existing RECORD claim to match the full-tree byte row."""

    try:
        record = validate_record_observation(record_observation)
        tree = validate_environment_tree(environment_tree)
    except ManifestError as exc:
        raise ManifestError("RECORD/tree input validation failed: %s" % exc) from exc
    tree_rows = {
        row["path"]: row for row in tree["entries"] if row["kind"] == "regular"
    }
    for distribution in record["distributions"]:
        for claim in distribution["claims"]:
            if claim["status"] in {"missing", "omitted_generated_pyc"}:
                continue
            row = tree_rows.get(claim["installed_path"])
            if (
                row is None
                or row["sha256"] != claim["actual"]["sha256"]
                or row["size"] != claim["actual"]["size"]
            ):
                raise ManifestError(
                    "RECORD/tree installed-file identity mismatch: %s"
                    % claim["installed_path"]
                )
