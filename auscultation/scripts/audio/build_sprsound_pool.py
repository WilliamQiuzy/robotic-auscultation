"""Build a versioned, exact-count SPRSound annotation-pool candidate on H200."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import random
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import uuid

try:
    from scripts.deployment import artifact_contract as contract
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deployment"))
    import artifact_contract as contract


SPRSOUND_REPOSITORY = "https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git"
SPRSOUND_COMMIT = "874eeb8736ddb78937c2fb5332fc7e7293d0f0ca"
SPRSOUND_TREE = "d414f930e98f3f6b4f7153489223e18c3173dd10"
SPRSOUND_PAIR_COUNT = 2683
DEFAULT_QUOTAS = {
    "Poor Quality": 120,
    "Normal": 70,
    "CAS": 70,
    "DAS": 70,
    "CAS & DAS": 70,
}
MANIFEST_NAME = "pool_manifest.jsonl"
SOURCE_NAME = "POOL_SOURCE.json"
COMPLETE_NAME = "POOL_COMPLETE.json"
_RELEASE_ID = re.compile(r"^sprsound_pool-[0-9a-f]{20}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ANNOTATION_PRODUCTION_HOME = Path("/home/ssh-ziyue")


class PoolContractError(RuntimeError):
    """The source, selection, release, or pointer violates the pool contract."""


@dataclass(frozen=True)
class SourceRecord:
    audio_path: Path
    content_sha256: str
    metadata_path: Path
    metadata_sha256: str
    size: int
    source_relative: str
    metadata_relative: str
    stratum: str


@dataclass(frozen=True)
class GitSnapshotEvidence:
    root: Path
    repository_root: Path
    commit: str
    tree: str
    entries: Mapping[str, Tuple[str, str]]


@dataclass(frozen=True)
class ProductionPoolPaths:
    home: Path
    data: Path
    root: Path
    releases: Path
    current: Path
    legacy: Path
    journal: Path
    lock: Path


@dataclass(frozen=True)
class LegacyPoolMigration:
    backup: Optional[Path]
    digest: Optional[str]
    previous_kind: str


@dataclass
class _LegacyPoolEvidence:
    migration: LegacyPoolMigration
    identity: Optional[Tuple[int, int, int, int, int, int]] = None
    parent_fd: int = -1
    root_fd: int = -1

    def close(self) -> None:
        if self.root_fd >= 0:
            os.close(self.root_fd)
            self.root_fd = -1
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1


def _production_paths(
    production_home: Union[str, Path],
) -> ProductionPoolPaths:
    home = Path(production_home)
    if not home.is_absolute():
        raise PoolContractError("production home must be absolute")
    data = home / "wam_ausc" / "data"
    root = data / "annotation_pools" / "sprsound"
    return ProductionPoolPaths(
        home=home,
        data=data,
        root=root,
        releases=root / "releases",
        current=root / "current",
        legacy=data / "annotation_pool_sprsound",
        journal=root / ".production-activation.json",
        lock=root / ".production-activation.lock",
    )


def _ensure_real_directory_chain(path: Union[str, Path], *, mode: int = 0o755) -> Path:
    """Create/open an absolute directory chain without following a symlink component."""
    target = Path(os.path.abspath(os.path.expanduser(str(path))))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for part in target.parts[1:]:
            try:
                os.mkdir(part, mode=mode, dir_fd=descriptor)
            except FileExistsError:
                pass
            info = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise PoolContractError(
                    "annotation pool path must contain only real directory components"
                )
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        raise PoolContractError(
            "annotation pool path contains a symlink or unsafe directory"
        ) from exc
    finally:
        os.close(descriptor)
    return target


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise PoolContractError("source artifact is missing, non-regular, or a symlink")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PoolContractError("SPRSound artifact is outside the corpus") from exc
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != relative:
        raise PoolContractError("SPRSound relative path is unsafe")
    return relative


def _canonical_stratum(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    compact = " ".join(value.strip().split()).lower().replace("and", "&")
    compact = re.sub(r"\s*&\s*", "&", compact)
    mapping = {
        "poor quality": "Poor Quality",
        "normal": "Normal",
        "cas": "CAS",
        "das": "DAS",
        "cas&das": "CAS & DAS",
    }
    return mapping.get(compact)


def _inventory(corpus_root: Union[str, Path]) -> Tuple[Path, Dict[str, Path], Dict[str, Path]]:
    root = Path(corpus_root)
    if root.is_symlink() or not root.is_dir():
        raise PoolContractError("SPRSound corpus must be a real directory")
    wavs: Dict[str, Path] = {}
    metadata: Dict[str, Path] = {}
    for directory, dirnames, filenames in os.walk(str(root), followlinks=False):
        base = Path(directory)
        kept = []
        for name in dirnames:
            candidate = base / name
            if candidate.is_symlink():
                raise PoolContractError("SPRSound corpus contains a symlink directory")
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            candidate = base / name
            if candidate.is_symlink() or not stat.S_ISREG(candidate.lstat().st_mode):
                raise PoolContractError("SPRSound corpus contains a non-regular file")
            suffix = candidate.suffix.lower()
            if suffix not in (".wav", ".json"):
                continue
            relative = _safe_relative(candidate, root)
            stem = relative[: -len(suffix)]
            target = wavs if suffix == ".wav" else metadata
            if stem in target:
                raise PoolContractError("SPRSound corpus contains a duplicate relative stem")
            target[stem] = candidate
    return root, wavs, metadata


def _pair_inventory(
    wavs: Mapping[str, Path], metadata: Mapping[str, Path],
) -> List[Tuple[Path, Path]]:
    """Pair same-layout fixtures by relative stem and official SPRSound by unique basename."""
    pairs: List[Tuple[Path, Path]] = []
    exact = set(wavs) & set(metadata)
    for stem in sorted(exact):
        pairs.append((wavs[stem], metadata[stem]))
    remaining_wavs = {key: value for key, value in wavs.items() if key not in exact}
    remaining_metadata = {key: value for key, value in metadata.items() if key not in exact}

    def unique_basenames(values: Mapping[str, Path], label: str) -> Dict[str, Path]:
        result: Dict[str, Path] = {}
        for stem, path in values.items():
            basename = PurePosixPath(stem).name
            if basename in result:
                raise PoolContractError(
                    "SPRSound %s contains an ambiguous repeated basename" % label
                )
            result[basename] = path
        return result

    wav_by_name = unique_basenames(remaining_wavs, "audio")
    metadata_by_name = unique_basenames(remaining_metadata, "metadata")
    if set(wav_by_name) != set(metadata_by_name):
        raise PoolContractError("SPRSound WAV/JSON pairing is incomplete")
    for basename in sorted(wav_by_name):
        pairs.append((wav_by_name[basename], metadata_by_name[basename]))
    return pairs


def discover_records(corpus_root: Union[str, Path]) -> List[SourceRecord]:
    root, wavs, metadata = _inventory(corpus_root)
    records: List[SourceRecord] = []
    for audio_path, meta_path in _pair_inventory(wavs, metadata):
        try:
            value = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PoolContractError("SPRSound metadata is invalid") from exc
        if not isinstance(value, dict):
            raise PoolContractError("SPRSound metadata must be an object")
        stratum = _canonical_stratum(value.get("record_annotation"))
        if stratum is None:
            continue
        records.append(SourceRecord(
            audio_path=audio_path,
            content_sha256=_sha256(audio_path),
            metadata_path=meta_path,
            metadata_sha256=_sha256(meta_path),
            size=audio_path.stat().st_size,
            source_relative=_safe_relative(audio_path, root),
            metadata_relative=_safe_relative(meta_path, root),
            stratum=stratum,
        ))
    return records


def deduplicate_records(records: Iterable[SourceRecord]) -> List[SourceRecord]:
    by_hash: Dict[str, List[SourceRecord]] = {}
    for record in records:
        by_hash.setdefault(record.content_sha256, []).append(record)
    selected = []
    for digest, rows in sorted(by_hash.items()):
        strata = {row.stratum for row in rows}
        if len(strata) != 1:
            raise PoolContractError(
                "one audio content hash appears in multiple strata: %s" % digest
            )
        selected.append(min(rows, key=lambda row: row.source_relative))
    return sorted(selected, key=lambda row: (row.stratum, row.source_relative))


def _validate_quotas(quotas: Mapping[str, object]) -> Dict[str, int]:
    if set(quotas) != set(DEFAULT_QUOTAS):
        raise PoolContractError("pool quotas must name the five exact strata")
    value: Dict[str, int] = {}
    for key in DEFAULT_QUOTAS:
        count = quotas[key]
        if type(count) is not int or count <= 0:
            raise PoolContractError("pool quota must be a positive integer")
        value[key] = count
    return value


def _validate_release_quota_contract(
    source: Mapping[str, object], quotas: Mapping[str, object],
) -> None:
    """Keep every 400-clip release on the one production contract."""
    wanted = _validate_quotas(quotas)
    production_total = sum(DEFAULT_QUOTAS.values())
    if sum(wanted.values()) == production_total and wanted != DEFAULT_QUOTAS:
        raise PoolContractError(
            "production pool quotas must match the exact 400-clip contract"
        )
    if (
        wanted == DEFAULT_QUOTAS
        and source.get("schema") != "wam_sprsound_git_source_identity_v2"
    ):
        raise PoolContractError("production pool source is not Git-object verified")
    if (
        source.get("schema") == "wam_sprsound_git_source_identity_v2"
        and wanted != DEFAULT_QUOTAS
    ):
        raise PoolContractError(
            "Git-object production source requires exact production quotas"
        )


def select_records(
    records: Sequence[SourceRecord], quotas: Mapping[str, object], *, seed: int,
) -> List[SourceRecord]:
    wanted = _validate_quotas(quotas)
    rng = random.Random(seed)
    result: List[SourceRecord] = []
    for stratum in DEFAULT_QUOTAS:
        available = sorted(
            (row for row in records if row.stratum == stratum),
            key=lambda row: row.source_relative,
        )
        if len(available) < wanted[stratum]:
            raise PoolContractError(
                "%s has %d available unique clips, requires %d"
                % (stratum, len(available), wanted[stratum])
            )
        result.extend(rng.sample(available, wanted[stratum]))
    if len({row.content_sha256 for row in result}) != sum(wanted.values()):
        raise PoolContractError("selected pool contains a repeated content hash")
    return sorted(result, key=lambda row: (row.stratum, row.source_relative))


def _validate_source_identity(value: object) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise PoolContractError("SPRSound source identity fields are invalid")
    schema = value.get("schema")
    fields = {"commit", "corpus", "pair_count", "repository", "schema"}
    if schema == "wam_sprsound_git_source_identity_v2":
        fields = fields | {"tree"}
    if set(value) != fields:
        raise PoolContractError("SPRSound source identity fields are invalid")
    if (
        schema not in (
            "wam_sprsound_source_identity_v1", "wam_sprsound_git_source_identity_v2",
        )
        or value.get("repository") != SPRSOUND_REPOSITORY
        or value.get("commit") != SPRSOUND_COMMIT
        or value.get("corpus") != "BioCAS2022"
        or type(value.get("pair_count")) is not int
        or value["pair_count"] <= 0
    ):
        raise PoolContractError("SPRSound source identity is invalid")
    if schema == "wam_sprsound_git_source_identity_v2" and value.get("tree") != SPRSOUND_TREE:
        raise PoolContractError("SPRSound source tree is invalid")
    if (
        schema == "wam_sprsound_git_source_identity_v2"
        and value.get("pair_count") != SPRSOUND_PAIR_COUNT
    ):
        raise PoolContractError("SPRSound Git source pair count is invalid")
    return value


def _destination(record: SourceRecord) -> str:
    slug = record.stratum.lower().replace(" & ", "-").replace(" ", "-")
    source_id = hashlib.sha256(record.source_relative.encode("utf-8")).hexdigest()[:16]
    return "%s-%s-%s.wav" % (slug, source_id, record.content_sha256[:16])


def _row(record: SourceRecord) -> Dict[str, object]:
    return {
        "content_sha256": record.content_sha256,
        "destination": _destination(record),
        "metadata_relative": record.metadata_relative,
        "metadata_sha256": record.metadata_sha256,
        "size": record.size,
        "source_id": hashlib.sha256(
            record.source_relative.encode("utf-8")
        ).hexdigest(),
        "source_relative": record.source_relative,
        "stratum": record.stratum,
    }


def _validate_manifest_row(value: object) -> Dict[str, object]:
    fields = {
        "content_sha256", "destination", "metadata_relative", "metadata_sha256",
        "size", "source_id", "source_relative", "stratum",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise PoolContractError("pool manifest row fields are invalid")
    row = value
    text_fields = (
        "content_sha256", "destination", "metadata_relative", "metadata_sha256",
        "source_id", "source_relative", "stratum",
    )
    if any(not isinstance(row[field], str) for field in text_fields):
        raise PoolContractError("pool manifest row field types are invalid")
    if (
        not _SHA256.fullmatch(row["content_sha256"])
        or not _SHA256.fullmatch(row["metadata_sha256"])
        or not _SHA256.fullmatch(row["source_id"])
        or type(row["size"]) is not int
        or row["size"] < 0
        or row["stratum"] not in DEFAULT_QUOTAS
    ):
        raise PoolContractError("pool manifest row values are invalid")
    for field, suffix in (("source_relative", ".wav"), ("metadata_relative", ".json")):
        relative = PurePosixPath(row[field])
        if (
            relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)
            or relative.as_posix() != row[field]
            or not row[field].lower().endswith(suffix)
        ):
            raise PoolContractError("pool manifest row source path is invalid")
    destination = PurePosixPath(row["destination"])
    if (
        destination.is_absolute()
        or len(destination.parts) != 1
        or destination.as_posix() != row["destination"]
        or not row["destination"].endswith(".wav")
    ):
        raise PoolContractError("pool manifest row destination is invalid")
    expected_source_id = hashlib.sha256(row["source_relative"].encode("utf-8")).hexdigest()
    slug = row["stratum"].lower().replace(" & ", "-").replace(" ", "-")
    expected_destination = "%s-%s-%s.wav" % (
        slug, expected_source_id[:16], row["content_sha256"][:16],
    )
    if row["source_id"] != expected_source_id or row["destination"] != expected_destination:
        raise PoolContractError("pool manifest row identity is invalid")
    return row


def _identity_projection(
    rows: Sequence[Mapping[str, object]], source: Mapping[str, object],
    quotas: Mapping[str, int], seed: int,
) -> Dict[str, object]:
    return {
        "quotas": dict(quotas),
        "rows": list(rows),
        "schema": "wam_sprsound_pool_payload_v1",
        "seed": seed,
        "source": dict(source),
    }


def _copy_exact(record: SourceRecord, destination: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(record.audio_path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PoolContractError("SPRSound audio source is not regular")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != record.content_sha256 or before.st_size != record.size:
            raise PoolContractError("SPRSound audio changed before copy")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with destination.open("xb") as target:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
        ):
            raise PoolContractError("SPRSound audio changed during copy")
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for directory, _dirnames, _filenames in os.walk(str(root), topdown=False):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def build_pool(
    corpus_root: Union[str, Path],
    pool_root: Union[str, Path],
    *,
    source_identity: Mapping[str, object],
    quotas: Mapping[str, object] = DEFAULT_QUOTAS,
    seed: int = 0,
    git_snapshot: Optional[GitSnapshotEvidence] = None,
) -> Path:
    source = _validate_source_identity(dict(source_identity))
    wanted = _validate_quotas(quotas)
    _validate_release_quota_contract(source, wanted)
    production_pool = wanted == DEFAULT_QUOTAS
    if production_pool and (
        source.get("schema") != "wam_sprsound_git_source_identity_v2"
        or git_snapshot is None
    ):
        raise PoolContractError(
            "production SPRSound pool requires verified Git-object snapshot evidence"
        )
    corpus, wavs, metadata = _inventory(corpus_root)
    pair_count = len(_pair_inventory(wavs, metadata))
    if pair_count != source["pair_count"]:
        raise PoolContractError(
            "SPRSound pair inventory mismatch: expected %s, found %s"
            % (source["pair_count"], pair_count)
        )
    if production_pool:
        assert git_snapshot is not None
        if (
            git_snapshot.commit != source["commit"]
            or git_snapshot.tree != source["tree"]
            or corpus.absolute() != (git_snapshot.root / "BioCAS2022").absolute()
        ):
            raise PoolContractError("production SPRSound snapshot identity is inconsistent")
        _verify_git_snapshot(git_snapshot)
    chosen = select_records(
        deduplicate_records(discover_records(corpus)), wanted, seed=seed,
    )
    rows = sorted((_row(record) for record in chosen), key=lambda row: row["destination"])
    projection = _identity_projection(rows, source, wanted, seed)
    release_id = "sprsound_pool-" + hashlib.sha256(
        contract.canonical_json_bytes(projection)
    ).hexdigest()[:20]
    root = _ensure_real_directory_chain(pool_root)
    releases = root / "releases"
    final = releases / release_id
    if os.path.lexists(str(final)):
        verify_pool_release(final, expected_quotas=wanted)
        if git_snapshot is not None:
            _verify_git_snapshot(git_snapshot)
        return final
    staging = root / (".staging-%s" % uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    by_relative = {record.source_relative: record for record in chosen}
    try:
        for row in rows:
            _copy_exact(by_relative[row["source_relative"]], staging / row["destination"])
        manifest_bytes = b"".join(contract.canonical_json_bytes(row) for row in rows)
        (staging / MANIFEST_NAME).write_bytes(manifest_bytes)
        (staging / SOURCE_NAME).write_bytes(contract.canonical_json_bytes(source))
        marker = {
            "counts": {key: sum(row["stratum"] == key for row in rows) for key in DEFAULT_QUOTAS},
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "quotas": wanted,
            "release_id": release_id,
            "schema": "wam_sprsound_pool_complete_v1",
            "seed": seed,
            "source_commit": source["commit"],
            "total": len(rows),
        }
        (staging / COMPLETE_NAME).write_bytes(contract.canonical_json_bytes(marker))
        verify_pool_release(staging, expected_quotas=wanted, expected_release_id=release_id)
        _fsync_tree(staging)
        _ensure_real_directory_chain(releases)
        try:
            contract._rename_noreplace(staging, final)
        except contract.ConflictError:
            verify_pool_release(final, expected_quotas=wanted)
        contract._fsync_directory(releases)
        verify_pool_release(final, expected_quotas=wanted)
        if git_snapshot is not None:
            _verify_git_snapshot(git_snapshot)
        return final
    finally:
        shutil.rmtree(str(staging), ignore_errors=True)


def verify_pool_release(
    release_path: Union[str, Path],
    *,
    expected_quotas: Optional[Mapping[str, object]] = DEFAULT_QUOTAS,
    expected_release_id: Optional[str] = None,
) -> Dict[str, object]:
    root = Path(release_path)
    if root.is_symlink() or not root.is_dir():
        raise PoolContractError("annotation pool release must be a real directory")
    raw_rows = (root / MANIFEST_NAME).read_bytes()
    rows = []
    for line in raw_rows.splitlines(keepends=True):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PoolContractError("pool manifest row is invalid") from exc
        if line != contract.canonical_json_bytes(row):
            raise PoolContractError("pool manifest row is not canonical")
        rows.append(_validate_manifest_row(row))
    try:
        source_raw = (root / SOURCE_NAME).read_bytes()
        source = json.loads(source_raw.decode("utf-8"))
        marker_raw = (root / COMPLETE_NAME).read_bytes()
        marker = json.loads(marker_raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoolContractError("pool release metadata is missing or invalid") from exc
    source = _validate_source_identity(source)
    wanted = _validate_quotas(
        marker.get("quotas", {}) if expected_quotas is None else expected_quotas
    )
    _validate_release_quota_contract(source, wanted)
    if source_raw != contract.canonical_json_bytes(source) or marker_raw != contract.canonical_json_bytes(marker):
        raise PoolContractError("pool release metadata is not canonical")
    projection = _identity_projection(rows, source, wanted, marker.get("seed"))
    release_id = "sprsound_pool-" + hashlib.sha256(
        contract.canonical_json_bytes(projection)
    ).hexdigest()[:20]
    if expected_release_id is not None and release_id != expected_release_id:
        raise PoolContractError("pool release identity does not match the build")
    if expected_release_id is None and root.name != release_id:
        raise PoolContractError("pool release directory does not match its identity")
    counts = {key: sum(row.get("stratum") == key for row in rows) for key in DEFAULT_QUOTAS}
    expected_marker = {
        "counts": counts,
        "manifest_sha256": hashlib.sha256(raw_rows).hexdigest(),
        "quotas": wanted,
        "release_id": release_id,
        "schema": "wam_sprsound_pool_complete_v1",
        "seed": marker.get("seed"),
        "source_commit": source["commit"],
        "total": len(rows),
    }
    if marker != expected_marker or counts != wanted or len(rows) != sum(wanted.values()):
        raise PoolContractError("pool final count or completion marker is invalid")
    destinations = [row["destination"] for row in rows]
    hashes = [row["content_sha256"] for row in rows]
    if len(set(destinations)) != len(rows) or len(set(hashes)) != len(rows):
        raise PoolContractError("pool destinations or content hashes are not unique")
    expected_files = set(destinations) | {MANIFEST_NAME, SOURCE_NAME, COMPLETE_NAME}
    actual_files = set()
    for item in root.iterdir():
        if item.is_symlink() or not item.is_file():
            raise PoolContractError("pool release contains an unsafe entry")
        actual_files.add(item.name)
    if actual_files != expected_files:
        raise PoolContractError("pool release exact file set is invalid")
    for row in rows:
        clip = root / row["destination"]
        if clip.stat().st_size != row["size"] or _sha256(clip) != row["content_sha256"]:
            raise PoolContractError("pool clip hash or size is invalid")
    return {"counts": counts, "release_id": release_id, "rows": rows, "total": len(rows)}


def verify_production_pool_release(
    release_path: Union[str, Path], *, expected_release_id: str,
) -> Dict[str, object]:
    """Verify the exact official v2, 400-clip production release contract."""
    root = Path(release_path)
    if (
        not isinstance(expected_release_id, str)
        or not _RELEASE_ID.fullmatch(expected_release_id)
        or root.name != expected_release_id
    ):
        raise PoolContractError("production release id is not exact")
    verified = verify_pool_release(
        root,
        expected_quotas=DEFAULT_QUOTAS,
        expected_release_id=expected_release_id,
    )
    source = json.loads((root / SOURCE_NAME).read_text(encoding="utf-8"))
    expected_source = {
        "commit": SPRSOUND_COMMIT,
        "corpus": "BioCAS2022",
        "pair_count": SPRSOUND_PAIR_COUNT,
        "repository": SPRSOUND_REPOSITORY,
        "schema": "wam_sprsound_git_source_identity_v2",
        "tree": SPRSOUND_TREE,
    }
    if source != expected_source:
        raise PoolContractError("production pool source is not the exact official Git-object v2")
    if verified.get("counts") != DEFAULT_QUOTAS or verified.get("total") != 400:
        raise PoolContractError("production pool does not have the exact 400-clip contract")
    return verified


def _production_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _descriptor_identity(info: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
        getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000)),
    )


def _path_identity_at(parent_fd: int, name: str) -> Tuple[int, int, int, int, int, int]:
    return _descriptor_identity(
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    )


def _identity_document(identity: Tuple[int, ...]) -> Dict[str, int]:
    if len(identity) != 6:
        raise PoolContractError("production path identity is incomplete")
    return {
        "ctime_ns": int(identity[5]),
        "device": int(identity[0]),
        "inode": int(identity[1]),
        "mode": int(identity[2]),
        "mtime_ns": int(identity[4]),
        "size": int(identity[3]),
    }


def _identity_tuple(value: object, label: str) -> Tuple[int, int, int, int, int, int]:
    fields = {"ctime_ns", "device", "inode", "mode", "mtime_ns", "size"}
    if not isinstance(value, dict) or set(value) != fields:
        raise PoolContractError("%s fields are invalid" % label)
    if any(type(value[key]) is not int or value[key] < 0 for key in fields):
        raise PoolContractError("%s values are invalid" % label)
    return (
        value["device"], value["inode"], value["mode"], value["size"],
        value["mtime_ns"], value["ctime_ns"],
    )


def _read_pointer_at(
    parent_fd: int, name: str,
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int, int, int]]]:
    try:
        before = _path_identity_at(parent_fd, name)
    except FileNotFoundError:
        return None, None
    if before[2] != stat.S_IFLNK:
        raise PoolContractError("production pointer is not a symlink")
    target = os.readlink(name, dir_fd=parent_fd)
    after = _path_identity_at(parent_fd, name)
    if after != before:
        raise PoolContractError("production pointer identity changed while read")
    return target, before


def _rename_noreplace_at(
    parent_fd: int, source_name: str, destination_name: str,
    *, expected_identity: Tuple[int, ...],
) -> None:
    if _path_identity_at(parent_fd, source_name) != expected_identity:
        raise PoolContractError("production rename source inode changed")
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        flag = 1
    elif sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        flag = 0x00000004
    else:
        raise PoolContractError("atomic production no-replace is unsupported")
    if function is None:
        raise PoolContractError("atomic production no-replace is unavailable")
    function.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        parent_fd, os.fsencode(source_name),
        parent_fd, os.fsencode(destination_name), flag,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise PoolContractError("production destination already exists")
        raise OSError(error, os.strerror(error), destination_name)


def _rename_noreplace_between_at(
    source_fd: int, source_name: str, destination_fd: int, destination_name: str,
    *, expected_identity: Tuple[int, ...],
) -> None:
    if _path_identity_at(source_fd, source_name) != expected_identity:
        raise PoolContractError("production publish source inode changed")
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        flag = 1
    elif sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        flag = 0x00000004
    else:
        raise PoolContractError("atomic production journal publish is unsupported")
    if function is None:
        raise PoolContractError("atomic production journal publish is unavailable")
    function.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        source_fd, os.fsencode(source_name),
        destination_fd, os.fsencode(destination_name), flag,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise PoolContractError("production activation journal already exists")
        raise OSError(error, os.strerror(error), destination_name)


def _exchange_at(
    parent_fd: int,
    first: str,
    second: str,
    *,
    expected_first_identity: Tuple[int, ...],
    expected_second_identity: Tuple[int, ...],
) -> None:
    def precheck() -> None:
        if (
            _path_identity_at(parent_fd, first) != expected_first_identity
            or _path_identity_at(parent_fd, second) != expected_second_identity
        ):
            raise PoolContractError(
                "production exchange source identity changed before syscall"
            )

    precheck()
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        flag = 2
    elif sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        flag = 0x00000002
    else:
        raise PoolContractError("atomic production pointer exchange is unsupported")
    if function is None:
        raise PoolContractError("atomic production pointer exchange is unavailable")
    function.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    precheck()
    result = function(
        parent_fd, os.fsencode(first), parent_fd, os.fsencode(second), flag,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), second)


def _hash_regular_descriptor(descriptor: int) -> Tuple[str, int]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise PoolContractError("legacy annotation pool contains a non-regular file")
    digest = hashlib.sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    after = os.fstat(descriptor)
    if _descriptor_identity(before) != _descriptor_identity(after) or size != before.st_size:
        raise PoolContractError("legacy annotation pool changed while being inventoried")
    return digest.hexdigest(), size


def _legacy_inventory_rows(
    descriptor: int, *, prefix: str = "",
) -> List[Dict[str, object]]:
    before = os.fstat(descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise PoolContractError("legacy annotation pool is not an ordinary directory")
    names = sorted(os.listdir(descriptor))
    rows: List[Dict[str, object]] = []
    for name in names:
        if not name or "/" in name or name in (".", ".."):
            raise PoolContractError("legacy annotation pool entry name is unsafe")
        relative = "%s/%s" % (prefix, name) if prefix else name
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            child_fd = os.open(name, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child_fd)
                if _descriptor_identity(opened) != _descriptor_identity(info):
                    raise PoolContractError("legacy directory changed while being opened")
                rows.append({
                    "mode": stat.S_IMODE(opened.st_mode),
                    "path": relative,
                    "type": "directory",
                })
                rows.extend(_legacy_inventory_rows(child_fd, prefix=relative))
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if _descriptor_identity(opened) != _descriptor_identity(observed):
                    raise PoolContractError("legacy directory changed during inventory")
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(info.st_mode):
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            child_fd = os.open(name, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child_fd)
                if _descriptor_identity(opened) != _descriptor_identity(info):
                    raise PoolContractError("legacy file changed while being opened")
                digest, size = _hash_regular_descriptor(child_fd)
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if _descriptor_identity(opened) != _descriptor_identity(observed):
                    raise PoolContractError("legacy file changed during inventory")
                rows.append({
                    "mode": stat.S_IMODE(opened.st_mode),
                    "path": relative,
                    "sha256": digest,
                    "size": size,
                    "type": "file",
                })
            finally:
                os.close(child_fd)
        else:
            raise PoolContractError(
                "legacy annotation pool contains a symlink or special entry"
            )
    after = os.fstat(descriptor)
    if (
        _descriptor_identity(before) != _descriptor_identity(after)
        or names != sorted(os.listdir(descriptor))
    ):
        raise PoolContractError("legacy annotation pool changed during inventory")
    return rows


def _legacy_inventory_digest(descriptor: int) -> str:
    document = {
        "rows": _legacy_inventory_rows(descriptor),
        "schema": "wam_sprsound_legacy_inventory_v1",
    }
    return hashlib.sha256(contract.canonical_json_bytes(document)).hexdigest()


_JOURNAL_OWNER = re.compile(r"^\.production-activation-owner-([0-9a-f]{64})$")


def _journal_request(journal: Mapping[str, object]) -> Dict[str, object]:
    return dict(journal["payload"]["request"])


def _production_journal(
    request: Mapping[str, object], *, stage: str,
    activated_pointer_identity: Optional[Tuple[int, ...]] = None,
    compatibility_pointer_identity: Optional[Tuple[int, ...]] = None,
    legacy_backup_identity: Optional[Tuple[int, ...]] = None,
    saved_previous_pointer_identity: Optional[Tuple[int, ...]] = None,
) -> Dict[str, object]:
    request_value = dict(request)
    payload = {
        "activated_pointer_identity": (
            None if activated_pointer_identity is None
            else _identity_document(activated_pointer_identity)
        ),
        "compatibility_pointer_identity": (
            None if compatibility_pointer_identity is None
            else _identity_document(compatibility_pointer_identity)
        ),
        "legacy_backup_identity": (
            None if legacy_backup_identity is None
            else _identity_document(legacy_backup_identity)
        ),
        "request": request_value,
        "request_sha256": hashlib.sha256(
            contract.canonical_json_bytes(request_value)
        ).hexdigest(),
        "saved_previous_pointer_identity": (
            None if saved_previous_pointer_identity is None
            else _identity_document(saved_previous_pointer_identity)
        ),
        "schema": "wam_sprsound_production_transaction_v2",
        "stage": stage,
    }
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(
            contract.canonical_json_bytes(payload)
        ).hexdigest(),
        "schema": "wam_sprsound_production_transaction_envelope_v1",
    }


def _production_journal_stage(
    paths: ProductionPoolPaths,
    journal: Mapping[str, object],
    stage: str,
    **identities: Optional[Tuple[int, ...]],
) -> Dict[str, object]:
    checked = _validate_production_journal(dict(journal), paths)
    payload = checked["payload"]
    values = {
        "activated_pointer_identity": payload["activated_pointer_identity"],
        "compatibility_pointer_identity": payload["compatibility_pointer_identity"],
        "legacy_backup_identity": payload["legacy_backup_identity"],
        "saved_previous_pointer_identity": payload["saved_previous_pointer_identity"],
    }
    for name, identity in identities.items():
        if name not in values:
            raise PoolContractError("production activation journal identity update is invalid")
        values[name] = None if identity is None else _identity_document(identity)
    request = payload["request"]
    result = _production_journal(request, stage=stage)
    result_payload = result["payload"]
    result_payload.update(values)
    result["payload_sha256"] = hashlib.sha256(
        contract.canonical_json_bytes(result_payload)
    ).hexdigest()
    return result


def _remove_owned_journal_states(paths: ProductionPoolPaths) -> None:
    root_fd = os.open(
        str(paths.root),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for name in sorted(os.listdir(root_fd)):
            if _JOURNAL_OWNER.fullmatch(name) is None:
                continue
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise PoolContractError("production journal owner path is unsafe")
            owner_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            try:
                for child in os.listdir(owner_fd):
                    child_info = os.stat(child, dir_fd=owner_fd, follow_symlinks=False)
                    if (
                        not child.startswith(".pending-")
                        or not stat.S_ISREG(child_info.st_mode)
                    ):
                        raise PoolContractError(
                            "production journal owner contains an unsafe entry"
                        )
                    os.unlink(child, dir_fd=owner_fd)
                os.fsync(owner_fd)
            finally:
                os.close(owner_fd)
            os.rmdir(name, dir_fd=root_fd)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


def _write_production_journal(
    paths: ProductionPoolPaths, journal: Mapping[str, object],
) -> None:
    checked = _validate_production_journal(dict(journal), paths)
    request_sha256 = str(checked["payload"]["request_sha256"])
    owner_name = ".production-activation-owner-%s" % request_sha256
    root_fd = os.open(
        str(paths.root),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    owner_fd = -1
    pending_name = ".pending-%s" % uuid.uuid4().hex
    descriptor = -1
    try:
        try:
            os.mkdir(owner_name, 0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        except FileExistsError:
            pass
        owner_fd = os.open(
            owner_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        descriptor = os.open(
            pending_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=owner_fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(contract.canonical_json_bytes(checked))
            handle.flush()
            os.fsync(handle.fileno())
        pending_identity = _path_identity_at(owner_fd, pending_name)
        if os.path.lexists(str(paths.journal)):
            existing = os.stat(paths.journal.name, dir_fd=root_fd, follow_symlinks=False)
            if not stat.S_ISREG(existing.st_mode):
                raise PoolContractError("production activation journal is unsafe")
            os.replace(
                pending_name, paths.journal.name,
                src_dir_fd=owner_fd, dst_dir_fd=root_fd,
            )
        else:
            _rename_noreplace_between_at(
                owner_fd, pending_name, root_fd, paths.journal.name,
                expected_identity=pending_identity,
            )
        os.fsync(owner_fd)
        os.fsync(root_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if owner_fd >= 0:
            try:
                os.unlink(pending_name, dir_fd=owner_fd)
            except FileNotFoundError:
                pass
            try:
                os.rmdir(owner_name, dir_fd=root_fd)
            except OSError:
                pass
            os.close(owner_fd)
        os.close(root_fd)


def _remove_production_journal(paths: ProductionPoolPaths) -> None:
    try:
        paths.journal.unlink()
    except FileNotFoundError:
        return
    contract._fsync_directory(paths.root)


def _validate_production_journal(
    value: object, paths: ProductionPoolPaths,
) -> Dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "payload", "payload_sha256", "schema",
    }:
        raise PoolContractError("production activation journal fields are invalid")
    payload = value.get("payload")
    payload_fields = {
        "activated_pointer_identity", "compatibility_pointer_identity",
        "legacy_backup_identity", "request", "request_sha256",
        "saved_previous_pointer_identity", "schema", "stage",
    }
    if not isinstance(payload, dict) or set(payload) != payload_fields:
        raise PoolContractError("production activation journal payload fields are invalid")
    request = payload.get("request")
    request_fields = {
        "activated_pointer", "compatibility", "legacy", "previous_current",
        "previous_pointer", "release", "schema",
    }
    if not isinstance(request, dict) or set(request) != request_fields:
        raise PoolContractError("production activation journal request fields are invalid")
    request_hash = hashlib.sha256(contract.canonical_json_bytes(request)).hexdigest()
    payload_hash = hashlib.sha256(contract.canonical_json_bytes(payload)).hexdigest()
    if (
        payload.get("request_sha256") != request_hash
        or value.get("payload_sha256") != payload_hash
    ):
        raise PoolContractError("production activation journal integrity hash mismatch")
    if (
        value.get("schema") != "wam_sprsound_production_transaction_envelope_v1"
        or payload.get("schema") != "wam_sprsound_production_transaction_v2"
        or request.get("schema") != "wam_sprsound_production_request_v1"
        or payload.get("stage") not in {
            "prepared", "legacy_migrated", "current_switched",
            "compat_switched", "committed",
        }
    ):
        raise PoolContractError("production activation journal schema or stage is invalid")
    release = request.get("release")
    activated = request.get("activated_pointer")
    previous_id = request.get("previous_current")
    previous = request.get("previous_pointer")
    legacy = request.get("legacy")
    compatibility = request.get("compatibility")
    if (
        not isinstance(release, dict)
        or set(release) != {"complete_sha256", "release_id"}
        or not isinstance(release.get("release_id"), str)
        or not _RELEASE_ID.fullmatch(release["release_id"])
        or not isinstance(release.get("complete_sha256"), str)
        or not _SHA256.fullmatch(release["complete_sha256"])
        or not isinstance(activated, dict)
        or set(activated) != {"identity", "name", "target"}
        or activated.get("target") != "releases/%s" % release["release_id"]
        or not isinstance(activated.get("name"), str)
        or not re.fullmatch(r"\.current\.activate-[0-9a-f]{32}", activated["name"])
    ):
        raise PoolContractError("production activation journal release request is invalid")
    activated_identity = _identity_tuple(
        activated["identity"], "production journal activated pointer identity",
    )
    if activated_identity[2] != stat.S_IFLNK:
        raise PoolContractError("production activation journal pointer is not a symlink")
    if previous_id is not None and (
        not isinstance(previous_id, str) or not _RELEASE_ID.fullmatch(previous_id)
    ):
        raise PoolContractError("production activation journal previous release is invalid")
    if previous_id is None:
        if previous is not None:
            raise PoolContractError("production activation journal previous pointer is inconsistent")
    elif (
        not isinstance(previous, dict)
        or set(previous) != {"identity", "target"}
        or previous.get("target") != "releases/%s" % previous_id
        or _identity_tuple(
            previous.get("identity"), "production journal previous pointer identity",
        )[2] != stat.S_IFLNK
    ):
        raise PoolContractError("production activation journal previous pointer is inconsistent")
    if not isinstance(legacy, dict) or set(legacy) != {
        "backup_name", "identity", "legacy_kind", "sha256",
    } or legacy.get("legacy_kind") not in {"absent", "compatibility", "directory"}:
        raise PoolContractError("production activation journal legacy request is invalid")
    kind = legacy["legacy_kind"]
    if kind == "directory":
        digest = legacy.get("sha256")
        backup_name = legacy.get("backup_name")
        if (
            not isinstance(digest, str) or not _SHA256.fullmatch(digest)
            or not isinstance(backup_name, str)
            or not re.fullmatch(
                r"annotation_pool_sprsound\.legacy-\d{8}T\d{6}Z-" + digest,
                backup_name,
            )
            or _identity_tuple(
                legacy.get("identity"), "production journal legacy identity",
            )[2] != stat.S_IFDIR
        ):
            raise PoolContractError("production activation backup journal is invalid")
    elif kind == "compatibility":
        if legacy.get("backup_name") is not None or legacy.get("sha256") is not None:
            raise PoolContractError("production activation journal legacy state is invalid")
        if _identity_tuple(
            legacy.get("identity"), "production journal legacy identity",
        )[2] != stat.S_IFLNK:
            raise PoolContractError("production activation journal legacy pointer is invalid")
    elif any(legacy.get(key) is not None for key in ("backup_name", "identity", "sha256")):
        raise PoolContractError("production activation journal has an unexpected backup")
    if not isinstance(compatibility, dict) or set(compatibility) != {
        "changed", "existing_identity", "prepared_identity", "prepared_name", "target",
    } or type(compatibility.get("changed")) is not bool or compatibility.get(
        "target"
    ) != "annotation_pools/sprsound/current":
        raise PoolContractError("production activation journal compatibility request is invalid")
    if compatibility["changed"]:
        if (
            compatibility.get("existing_identity") is not None
            or not isinstance(compatibility.get("prepared_name"), str)
            or not re.fullmatch(
                r"\.annotation_pool_sprsound\.activate-[0-9a-f]{32}",
                compatibility["prepared_name"],
            )
            or _identity_tuple(
                compatibility.get("prepared_identity"),
                "production journal compatibility identity",
            )[2] != stat.S_IFLNK
        ):
            raise PoolContractError("production activation journal compatibility state is invalid")
    elif (
        compatibility.get("prepared_name") is not None
        or compatibility.get("prepared_identity") is not None
        or _identity_tuple(
            compatibility.get("existing_identity"),
            "production journal compatibility identity",
        )[2] != stat.S_IFLNK
    ):
        raise PoolContractError("production activation journal compatibility state is invalid")
    for key in (
        "activated_pointer_identity", "compatibility_pointer_identity",
        "legacy_backup_identity", "saved_previous_pointer_identity",
    ):
        if payload[key] is not None:
            _identity_tuple(payload[key], "production journal %s" % key)
    return dict(value)


def _read_production_journal(paths: ProductionPoolPaths) -> Dict[str, object]:
    try:
        info = paths.journal.lstat()
        raw = paths.journal.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoolContractError("production activation journal is invalid") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PoolContractError("production activation journal is unsafe")
    if raw != contract.canonical_json_bytes(value):
        raise PoolContractError("production activation journal is not canonical")
    if isinstance(value, dict) and isinstance(value.get("backup_name"), str) and (
        "/" in value["backup_name"] or ".." in value["backup_name"]
    ):
        raise PoolContractError("production activation backup journal is invalid")
    return _validate_production_journal(value, paths)


def _verify_legacy_backup(
    paths: ProductionPoolPaths, backup_name: str, expected_digest: str,
) -> Path:
    backup = paths.data / backup_name
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(str(backup), flags)
    except OSError as exc:
        raise PoolContractError("legacy backup is missing or unsafe") from exc
    try:
        observed = _legacy_inventory_digest(descriptor)
    finally:
        os.close(descriptor)
    if observed != expected_digest:
        raise PoolContractError("legacy backup content does not match its journal")
    return backup


def _inspect_legacy_pool(paths: ProductionPoolPaths) -> _LegacyPoolEvidence:
    if not os.path.lexists(str(paths.legacy)):
        return _LegacyPoolEvidence(LegacyPoolMigration(None, None, "absent"))
    if paths.legacy.is_symlink():
        if os.readlink(paths.legacy) != "annotation_pools/sprsound/current":
            raise PoolContractError("legacy compatibility symlink target is unsafe")
        parent_fd = os.open(
            str(paths.data),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            identity = _path_identity_at(parent_fd, paths.legacy.name)
        finally:
            os.close(parent_fd)
        return _LegacyPoolEvidence(
            LegacyPoolMigration(None, None, "compatibility"), identity=identity,
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open(str(paths.data), flags)
    root_fd = -1
    try:
        before = os.stat(paths.legacy.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            raise PoolContractError("legacy annotation pool must be an ordinary directory")
        root_fd = os.open(paths.legacy.name, flags, dir_fd=parent_fd)
        opened = os.fstat(root_fd)
        if _descriptor_identity(opened) != _descriptor_identity(before):
            raise PoolContractError("legacy annotation pool changed while being opened")
        digest = _legacy_inventory_digest(root_fd)
        backup = paths.data / (
            "annotation_pool_sprsound.legacy-%s-%s"
            % (_production_timestamp(), digest)
        )
        return _LegacyPoolEvidence(
            LegacyPoolMigration(backup, digest, "directory"),
            identity=_descriptor_identity(opened),
            parent_fd=parent_fd,
            root_fd=root_fd,
        )
    except BaseException:
        if root_fd >= 0:
            os.close(root_fd)
        os.close(parent_fd)
        raise


def _migrate_legacy_pool(
    paths: ProductionPoolPaths, evidence: _LegacyPoolEvidence,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    migration = evidence.migration
    if migration.previous_kind == "absent":
        if os.path.lexists(str(paths.legacy)):
            raise PoolContractError("absent legacy path changed before migration")
        return None
    if migration.previous_kind == "compatibility":
        parent_fd = os.open(
            str(paths.data),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            if (
                evidence.identity is None
                or _path_identity_at(parent_fd, paths.legacy.name) != evidence.identity
                or os.readlink(paths.legacy.name, dir_fd=parent_fd)
                != "annotation_pools/sprsound/current"
            ):
                raise PoolContractError("legacy compatibility identity changed")
        finally:
            os.close(parent_fd)
        return None
    if (
        migration.previous_kind != "directory"
        or migration.backup is None
        or migration.digest is None
        or evidence.identity is None
        or evidence.parent_fd < 0
        or evidence.root_fd < 0
    ):
        raise PoolContractError("legacy migration evidence is incomplete")
    if (
        _path_identity_at(evidence.parent_fd, paths.legacy.name) != evidence.identity
        or _descriptor_identity(os.fstat(evidence.root_fd)) != evidence.identity
        or _legacy_inventory_digest(evidence.root_fd) != migration.digest
    ):
        raise PoolContractError("legacy annotation pool changed before pathname rename")
    _rename_noreplace_at(
        evidence.parent_fd,
        paths.legacy.name,
        migration.backup.name,
        expected_identity=evidence.identity,
    )
    opened_after = _descriptor_identity(os.fstat(evidence.root_fd))
    observed_after = _path_identity_at(evidence.parent_fd, migration.backup.name)
    if (
        opened_after != observed_after
        or opened_after[:3] != evidence.identity[:3]
        or _legacy_inventory_digest(evidence.root_fd) != migration.digest
    ):
        raise PoolContractError("legacy backup inode or hash changed after rename")
    os.fsync(evidence.parent_fd)
    return opened_after


def _production_checkpoint(_stage: str) -> None:
    """Test hook at durable production transaction boundaries."""


def _prepare_pointer_at(
    parent_fd: int, name: str, target: str,
) -> Tuple[str, Tuple[int, int, int, int, int, int]]:
    prepared_name = ".%s.activate-%s" % (name, uuid.uuid4().hex)
    os.symlink(target, prepared_name, dir_fd=parent_fd)
    identity = _path_identity_at(parent_fd, prepared_name)
    observed_target, observed_identity = _read_pointer_at(parent_fd, prepared_name)
    if observed_target != target or observed_identity != identity:
        raise PoolContractError("prepared production pointer did not persist")
    os.fsync(parent_fd)
    return prepared_name, identity


def _production_current_state(
    paths: ProductionPoolPaths, root_fd: int,
) -> Tuple[Optional[str], Optional[str], Optional[Tuple[int, int, int, int, int, int]]]:
    target, identity = _read_pointer_at(root_fd, paths.current.name)
    if target is None:
        return None, None, None
    prefix = "releases/"
    release_id = target[len(prefix):] if target.startswith(prefix) else ""
    if target != prefix + release_id or not _RELEASE_ID.fullmatch(release_id):
        raise PoolContractError("production current target is not canonical")
    verify_production_pool_release(
        paths.releases / release_id, expected_release_id=release_id,
    )
    return release_id, target, identity


def _switch_production_current(
    paths: ProductionPoolPaths, journal: Mapping[str, object],
) -> Tuple[
    Tuple[int, int, int, int, int, int],
    Optional[Tuple[int, int, int, int, int, int]],
]:
    request = _journal_request(journal)
    activated = request["activated_pointer"]
    previous = request["previous_pointer"]
    root_fd = os.open(
        str(paths.root),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        prepared_identity = _identity_tuple(
            activated["identity"], "prepared production current identity",
        )
        prepared_target, observed_prepared = _read_pointer_at(
            root_fd, activated["name"],
        )
        if prepared_target != activated["target"] or observed_prepared != prepared_identity:
            raise PoolContractError("prepared production current identity changed")
        _previous_id, current_target, current_identity = _production_current_state(
            paths, root_fd,
        )
        if previous is None:
            if current_target is not None or current_identity is not None:
                raise PoolContractError("production current compare-and-swap identity changed")
            _rename_noreplace_at(
                root_fd, activated["name"], paths.current.name,
                expected_identity=prepared_identity,
            )
            saved_previous = None
        else:
            expected_previous = _identity_tuple(
                previous["identity"], "expected production current identity",
            )
            if current_target != previous["target"] or current_identity != expected_previous:
                raise PoolContractError("production current compare-and-swap inode changed")
            _exchange_at(
                root_fd,
                activated["name"],
                paths.current.name,
                expected_first_identity=prepared_identity,
                expected_second_identity=expected_previous,
            )
            saved_target, saved_previous = _read_pointer_at(
                root_fd, activated["name"],
            )
            installed_target, installed = _read_pointer_at(root_fd, paths.current.name)
            if (
                saved_target != previous["target"]
                or saved_previous is None
                or saved_previous[:3] != expected_previous[:3]
                or installed_target != activated["target"]
                or installed is None
                or installed[:3] != prepared_identity[:3]
            ):
                if (
                    installed_target == activated["target"]
                    and installed is not None
                    and installed[:3] == prepared_identity[:3]
                ):
                    _exchange_at(
                        root_fd,
                        activated["name"],
                        paths.current.name,
                        expected_first_identity=saved_previous,
                        expected_second_identity=installed,
                    )
                    os.fsync(root_fd)
                raise PoolContractError("production current compare-and-swap identity mismatch")
        os.fsync(root_fd)
        installed_target, installed_identity = _read_pointer_at(
            root_fd, paths.current.name,
        )
        if installed_target != activated["target"] or installed_identity is None:
            raise PoolContractError("production current switch did not persist")
        return installed_identity, saved_previous
    finally:
        os.close(root_fd)


def _remove_exact_pointer_at(
    parent_fd: int, name: str, target: str, identity: Tuple[int, ...],
) -> None:
    observed_target, observed_identity = _read_pointer_at(parent_fd, name)
    if observed_target != target or observed_identity != identity:
        raise PoolContractError("production rollback pointer identity changed")
    os.unlink(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _cleanup_saved_previous_pointer(
    paths: ProductionPoolPaths, journal: Mapping[str, object],
) -> None:
    request = _journal_request(journal)
    previous = request["previous_pointer"]
    if previous is None:
        return
    saved_raw = journal["payload"]["saved_previous_pointer_identity"]
    if saved_raw is None:
        return
    root_fd = os.open(
        str(paths.root),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        _remove_exact_pointer_at(
            root_fd,
            request["activated_pointer"]["name"],
            previous["target"],
            _identity_tuple(saved_raw, "saved previous production pointer identity"),
        )
    finally:
        os.close(root_fd)


def _rollback_production_current(
    paths: ProductionPoolPaths, journal: Mapping[str, object],
) -> None:
    request = _journal_request(journal)
    payload = journal["payload"]
    activated = request["activated_pointer"]
    previous = request["previous_pointer"]
    root_fd = os.open(
        str(paths.root),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        current_target, current_identity = _read_pointer_at(root_fd, paths.current.name)
        if previous is None:
            if current_target is None:
                try:
                    _remove_exact_pointer_at(
                        root_fd, activated["name"], activated["target"],
                        _identity_tuple(
                            activated["identity"], "prepared production current identity",
                        ),
                    )
                except FileNotFoundError:
                    pass
                return
        else:
            previous_identity = _identity_tuple(
                previous["identity"], "previous production current identity",
            )
            if current_target == previous["target"] and current_identity == previous_identity:
                _remove_exact_pointer_at(
                    root_fd, activated["name"], activated["target"],
                    _identity_tuple(
                        activated["identity"], "prepared production current identity",
                    ),
                )
                return
        activated_raw = payload["activated_pointer_identity"]
        if activated_raw is None:
            raise PoolContractError("production rollback lacks activated current identity")
        activated_identity = _identity_tuple(
            activated_raw, "activated production current identity",
        )
        if current_target != activated["target"] or current_identity != activated_identity:
            raise PoolContractError("production rollback current identity changed")
        if previous is None:
            _remove_exact_pointer_at(
                root_fd, paths.current.name, activated["target"], activated_identity,
            )
            return
        saved_identity = _identity_tuple(
            payload["saved_previous_pointer_identity"],
            "saved previous production current identity",
        )
        saved_target, observed_saved = _read_pointer_at(root_fd, activated["name"])
        if saved_target != previous["target"] or observed_saved != saved_identity:
            raise PoolContractError("saved previous production current identity changed")
        _exchange_at(
            root_fd,
            activated["name"],
            paths.current.name,
            expected_first_identity=observed_saved,
            expected_second_identity=current_identity,
        )
        os.fsync(root_fd)
        restored_target, restored_identity = _read_pointer_at(root_fd, paths.current.name)
        moved_target, moved_identity = _read_pointer_at(root_fd, activated["name"])
        if (
            restored_target != previous["target"]
            or restored_identity is None
            or restored_identity[:3] != saved_identity[:3]
            or moved_target != activated["target"]
            or moved_identity is None
            or moved_identity[:3] != activated_identity[:3]
        ):
            raise PoolContractError("previous production current was not restored")
        os.unlink(activated["name"], dir_fd=root_fd)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


def _rollback_production_activation(
    paths: ProductionPoolPaths, journal: Mapping[str, object],
) -> None:
    request = _journal_request(journal)
    payload = journal["payload"]
    legacy_request = request["legacy"]
    compatibility = request["compatibility"]
    kind = legacy_request["legacy_kind"]
    backup_name = legacy_request["backup_name"]
    previous_current = request["previous_current"]
    legacy_already_restored = False
    data_fd = os.open(
        str(paths.data),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if compatibility["changed"]:
            published_raw = payload["compatibility_pointer_identity"]
            if published_raw is not None:
                _remove_exact_pointer_at(
                    data_fd,
                    paths.legacy.name,
                    compatibility["target"],
                    _identity_tuple(
                        published_raw, "activated legacy compatibility identity",
                    ),
                )
            else:
                try:
                    _remove_exact_pointer_at(
                        data_fd,
                        compatibility["prepared_name"],
                        compatibility["target"],
                        _identity_tuple(
                            compatibility["prepared_identity"],
                            "prepared legacy compatibility identity",
                        ),
                    )
                except FileNotFoundError:
                    pass
        elif (
            _path_identity_at(data_fd, paths.legacy.name)
            != _identity_tuple(
                compatibility["existing_identity"],
                "existing legacy compatibility identity",
            )
            or os.readlink(paths.legacy.name, dir_fd=data_fd) != compatibility["target"]
        ):
            raise PoolContractError("legacy compatibility identity changed during rollback")
    finally:
        os.close(data_fd)
    _rollback_production_current(paths, journal)
    if kind == "directory":
        digest = legacy_request["sha256"]
        expected_before = _identity_tuple(
            legacy_request["identity"], "legacy pre-migration identity",
        )
        backup = paths.data / backup_name
        if os.path.lexists(str(paths.legacy)):
            if paths.legacy.is_symlink() or not paths.legacy.is_dir():
                raise PoolContractError("legacy path changed during rollback")
            descriptor = os.open(
                str(paths.legacy),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                observed = _descriptor_identity(os.fstat(descriptor))
                if (
                    observed[:3] != expected_before[:3]
                    or _legacy_inventory_digest(descriptor) != digest
                ):
                    raise PoolContractError("legacy path changed during rollback")
            finally:
                os.close(descriptor)
            legacy_already_restored = True
        if not legacy_already_restored and os.path.lexists(str(backup)):
            _verify_legacy_backup(paths, backup_name, digest)
            if os.path.lexists(str(paths.legacy)):
                raise PoolContractError("legacy path is occupied during rollback")
            parent_fd = os.open(
                str(paths.data),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                backup_identity = _path_identity_at(parent_fd, backup.name)
                expected_backup_raw = payload["legacy_backup_identity"]
                if expected_backup_raw is not None and backup_identity != _identity_tuple(
                    expected_backup_raw, "legacy backup identity",
                ):
                    raise PoolContractError("legacy backup identity changed during rollback")
                _rename_noreplace_at(
                    parent_fd, backup.name, paths.legacy.name,
                    expected_identity=backup_identity,
                )
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        if paths.legacy.is_symlink() or not paths.legacy.is_dir():
            raise PoolContractError("legacy ordinary directory was not restored")
    elif kind == "absent":
        if os.path.lexists(str(paths.legacy)):
            raise PoolContractError("absent legacy path was not restored")
    elif kind == "compatibility":
        if (
            not paths.legacy.is_symlink()
            or os.readlink(paths.legacy) != "annotation_pools/sprsound/current"
        ):
            raise PoolContractError("legacy compatibility symlink was not restored")
    else:
        raise PoolContractError("production rollback journal legacy kind is invalid")
    _remove_production_journal(paths)


def _production_current_id(paths: ProductionPoolPaths) -> Optional[str]:
    if not os.path.lexists(str(paths.current)):
        return None
    if not paths.current.is_symlink():
        raise PoolContractError("production current is not a symlink")
    raw = os.readlink(paths.current)
    prefix = "releases/"
    release_id = raw[len(prefix):] if raw.startswith(prefix) else ""
    if raw != prefix + release_id or not _RELEASE_ID.fullmatch(release_id):
        raise PoolContractError("production current target is not canonical")
    release = paths.releases / release_id
    if paths.current.resolve(strict=True) != release.resolve(strict=True):
        raise PoolContractError("production current resolves outside canonical releases")
    verify_production_pool_release(release, expected_release_id=release_id)
    return release_id


def _atomic_symlink(target: str, destination: Path, *, replace: bool) -> None:
    temporary = destination.parent / (".%s-%s" % (destination.name, uuid.uuid4().hex))
    temporary.symlink_to(target)
    try:
        if replace:
            os.replace(str(temporary), str(destination))
        else:
            contract._rename_noreplace(temporary, destination)
        contract._fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _verify_production_links(paths: ProductionPoolPaths, release_id: str) -> None:
    if _production_current_id(paths) != release_id:
        raise PoolContractError("production current post-verification failed")
    if (
        not paths.legacy.is_symlink()
        or os.readlink(paths.legacy) != "annotation_pools/sprsound/current"
        or paths.legacy.resolve(strict=True)
        != (paths.releases / release_id).resolve(strict=True)
    ):
        raise PoolContractError("legacy compatibility post-verification failed")


@contextmanager
def _pointer_lock(parent: Path):
    parent = _ensure_real_directory_chain(parent)
    descriptor = os.open(
        str(parent / ".deploy.lock"),
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _production_lock(paths: ProductionPoolPaths):
    _ensure_real_directory_chain(paths.root)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(paths.lock), flags, 0o600)
    except OSError as exc:
        raise PoolContractError("production activation lock is unsafe") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PoolContractError("production activation lock is not regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise PoolContractError("production activation lock is already held") from exc
            raise
        os.ftruncate(descriptor, 0)
        os.write(descriptor, ("%s\n" % os.getpid()).encode("ascii"))
        os.fsync(descriptor)
        contract._fsync_directory(paths.root)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _reconcile_pool_production_at_home(
    *, production_home: Union[str, Path],
) -> Dict[str, object]:
    """Reconcile a durable interrupted production transaction fail-closed."""
    paths = _production_paths(production_home)
    _ensure_real_directory_chain(paths.root)
    with _production_lock(paths):
        if not os.path.lexists(str(paths.journal)):
            _remove_owned_journal_states(paths)
            return {"schema": "wam_sprsound_production_reconcile_v1", "status": "clean"}
        journal = _read_production_journal(paths)
        request = _journal_request(journal)
        release_request = request["release"]
        release_id = str(release_request["release_id"])
        release = paths.releases / release_id
        verify_production_pool_release(release, expected_release_id=release_id)
        if _sha256(release / COMPLETE_NAME) != release_request["complete_sha256"]:
            raise PoolContractError("production journal release identity changed")
        if journal["payload"]["stage"] == "committed":
            _verify_production_links(paths, release_id)
            root_fd = os.open(
                str(paths.root),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                target, identity = _read_pointer_at(root_fd, paths.current.name)
                if (
                    target != request["activated_pointer"]["target"]
                    or identity != _identity_tuple(
                        journal["payload"]["activated_pointer_identity"],
                        "committed production current identity",
                    )
                ):
                    raise PoolContractError("committed production current identity changed")
            finally:
                os.close(root_fd)
            _cleanup_saved_previous_pointer(paths, journal)
            _remove_production_journal(paths)
            status = "committed"
        else:
            _rollback_production_activation(paths, journal)
            status = "rolled_back"
        _remove_owned_journal_states(paths)
        return {
            "release_id": release_id,
            "schema": "wam_sprsound_production_reconcile_v1",
            "status": status,
        }


def reconcile_pool_production() -> Dict[str, object]:
    """Reconcile only the canonical live H200 annotation pool namespace."""
    return _reconcile_pool_production_at_home(
        production_home=ANNOTATION_PRODUCTION_HOME,
    )


def _activate_pool_production_at_home(
    release_id: str,
    *,
    confirm_release_id: str,
    expected_current: Optional[str],
    production_home: Union[str, Path],
) -> Dict[str, object]:
    """Migrate legacy data and activate one exact official production pool."""
    if (
        not isinstance(release_id, str)
        or not _RELEASE_ID.fullmatch(release_id)
        or confirm_release_id != release_id
    ):
        raise PoolContractError("production activation requires explicit release confirmation")
    if expected_current is not None and not _RELEASE_ID.fullmatch(expected_current):
        raise PoolContractError("production expected current release id is invalid")
    paths = _production_paths(production_home)
    _ensure_real_directory_chain(paths.releases)
    release = paths.releases / release_id
    verify_production_pool_release(release, expected_release_id=release_id)
    with _production_lock(paths):
        if os.path.lexists(str(paths.journal)):
            raise PoolContractError("production activation requires reconciliation")
        owner_names = [
            name for name in os.listdir(paths.root)
            if _JOURNAL_OWNER.fullmatch(name)
        ]
        if owner_names:
            raise PoolContractError("production activation requires pending reconciliation")
        marker = verify_production_pool_release(release, expected_release_id=release_id)
        root_fd = os.open(
            str(paths.root),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        data_fd = os.open(
            str(paths.data),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        legacy_evidence: Optional[_LegacyPoolEvidence] = None
        journal: Optional[Dict[str, object]] = None
        try:
            previous_current, previous_target, previous_identity = (
                _production_current_state(paths, root_fd)
            )
            if previous_current != expected_current:
                raise PoolContractError("production current compare-and-swap mismatch")
            legacy_evidence = _inspect_legacy_pool(paths)
            migration = legacy_evidence.migration
            activated_target = "releases/%s" % release_id
            activated_name, activated_prepared_identity = _prepare_pointer_at(
                root_fd, paths.current.name, activated_target,
            )
            compatibility_target = "annotation_pools/sprsound/current"
            if migration.previous_kind == "compatibility":
                compatibility_changed = False
                compatibility_name = None
                compatibility_prepared_identity = None
                compatibility_existing_identity = legacy_evidence.identity
            else:
                compatibility_changed = True
                compatibility_name, compatibility_prepared_identity = _prepare_pointer_at(
                    data_fd, paths.legacy.name, compatibility_target,
                )
                compatibility_existing_identity = None
            request = {
                "activated_pointer": {
                    "identity": _identity_document(activated_prepared_identity),
                    "name": activated_name,
                    "target": activated_target,
                },
                "compatibility": {
                    "changed": compatibility_changed,
                    "existing_identity": (
                        None if compatibility_existing_identity is None
                        else _identity_document(compatibility_existing_identity)
                    ),
                    "prepared_identity": (
                        None if compatibility_prepared_identity is None
                        else _identity_document(compatibility_prepared_identity)
                    ),
                    "prepared_name": compatibility_name,
                    "target": compatibility_target,
                },
                "legacy": {
                    "backup_name": (
                        None if migration.backup is None else migration.backup.name
                    ),
                    "identity": (
                        None if legacy_evidence.identity is None
                        else _identity_document(legacy_evidence.identity)
                    ),
                    "legacy_kind": migration.previous_kind,
                    "sha256": migration.digest,
                },
                "previous_current": previous_current,
                "previous_pointer": (
                    None if previous_identity is None else {
                        "identity": _identity_document(previous_identity),
                        "target": previous_target,
                    }
                ),
                "release": {
                    "complete_sha256": _sha256(release / COMPLETE_NAME),
                    "release_id": marker["release_id"],
                },
                "schema": "wam_sprsound_production_request_v1",
            }
            journal = _production_journal(request, stage="prepared")
            _write_production_journal(paths, journal)
            legacy_backup_identity = _migrate_legacy_pool(paths, legacy_evidence)
            journal = _production_journal_stage(
                paths, journal, "legacy_migrated",
                legacy_backup_identity=legacy_backup_identity,
            )
            _write_production_journal(paths, journal)
            _production_checkpoint("after_legacy_migration")
            activated_identity, saved_previous_identity = _switch_production_current(
                paths, journal,
            )
            journal = _production_journal_stage(
                paths, journal, "current_switched",
                activated_pointer_identity=activated_identity,
                saved_previous_pointer_identity=saved_previous_identity,
            )
            _write_production_journal(paths, journal)
            _production_checkpoint("after_current_switch")
            if compatibility_changed:
                assert compatibility_name is not None
                assert compatibility_prepared_identity is not None
                if os.path.lexists(str(paths.legacy)):
                    raise PoolContractError("legacy compatibility destination is occupied")
                _rename_noreplace_at(
                    data_fd,
                    compatibility_name,
                    paths.legacy.name,
                    expected_identity=compatibility_prepared_identity,
                )
                compatibility_identity = _path_identity_at(data_fd, paths.legacy.name)
            else:
                compatibility_identity = compatibility_existing_identity
            journal = _production_journal_stage(
                paths, journal, "compat_switched",
                compatibility_pointer_identity=compatibility_identity,
            )
            _write_production_journal(paths, journal)
            _production_checkpoint("after_compat_switch")
            _verify_production_links(paths, release_id)
            current_target, current_identity = _read_pointer_at(root_fd, paths.current.name)
            if current_target != activated_target or current_identity != activated_identity:
                raise PoolContractError("production current full identity changed after activation")
            _production_checkpoint("after_postverify")
            journal = _production_journal_stage(paths, journal, "committed")
            _write_production_journal(paths, journal)
            _production_checkpoint("after_commit_record")
            _cleanup_saved_previous_pointer(paths, journal)
            _remove_production_journal(paths)
            _remove_owned_journal_states(paths)
            return {
                "legacy_backup": (
                    None if migration.backup is None else migration.backup.as_posix()
                ),
                "legacy_sha256": migration.digest,
                "previous_current": previous_current,
                "release_id": release_id,
                "schema": "wam_sprsound_production_activation_v1",
            }
        except BaseException as original:
            if os.path.lexists(str(paths.journal)) and journal is not None:
                try:
                    _rollback_production_activation(paths, journal)
                except BaseException as rollback_error:
                    raise PoolContractError(
                        "production activation failed and rollback did not converge: %s"
                        % rollback_error
                    ) from original
            raise
        finally:
            if legacy_evidence is not None:
                legacy_evidence.close()
            os.close(data_fd)
            os.close(root_fd)


def activate_pool_production(
    release_id: str,
    *,
    confirm_release_id: str,
    expected_current: Optional[str],
) -> Dict[str, object]:
    """Activate only under /home/ssh-ziyue/wam_ausc/data."""
    return _activate_pool_production_at_home(
        release_id,
        confirm_release_id=confirm_release_id,
        expected_current=expected_current,
        production_home=ANNOTATION_PRODUCTION_HOME,
    )


def activate_pool_canary(
    release_path: Union[str, Path], *, current_link: Union[str, Path],
    expected_current: Optional[str],
) -> Optional[str]:
    release = Path(release_path)
    verify_pool_release(release, expected_quotas=None)
    current = Path(current_link)
    if current.parent.name != ".pool-canary" or current.name != "current":
        raise PoolContractError("pool activation is restricted to an isolated canary pointer")
    with _pointer_lock(current.parent):
        if os.path.lexists(str(current)) and not current.is_symlink():
            raise PoolContractError("pool canary current is not a symlink")
        actual = os.readlink(str(current)) if current.is_symlink() else None
        if actual != expected_current:
            raise PoolContractError("pool canary compare-and-swap mismatch")
        target = os.path.relpath(str(release), str(current.parent))
        temporary = current.parent / (".current-%s" % uuid.uuid4().hex)
        temporary.symlink_to(target)
        os.replace(str(temporary), str(current))
        contract._fsync_directory(current.parent)
        if current.resolve() != release.resolve():
            raise PoolContractError("pool canary activation resolved incorrectly")
        return actual


def rollback_pool_canary(
    *, current_link: Union[str, Path], expected_current: Optional[str],
    previous: Optional[str],
) -> None:
    current = Path(current_link)
    if current.parent.name != ".pool-canary" or current.name != "current":
        raise PoolContractError("pool rollback is restricted to an isolated canary pointer")
    with _pointer_lock(current.parent):
        actual = os.readlink(str(current)) if current.is_symlink() else None
        if actual != expected_current:
            raise PoolContractError("pool canary compare-and-swap mismatch")
        if previous is None:
            current.unlink(missing_ok=True)
        else:
            relative = PurePosixPath(previous)
            if relative.is_absolute() or ".." in relative.parts:
                raise PoolContractError("pool rollback target is unsafe")
            temporary = current.parent / (".current-%s" % uuid.uuid4().hex)
            temporary.symlink_to(previous)
            os.replace(str(temporary), str(current))
        contract._fsync_directory(current.parent)


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if completed.returncode != 0:
        raise PoolContractError("SPRSound Git identity check failed")
    return completed.stdout.strip()


def _git_tree_entries(root: Path, commit: str) -> Dict[str, Tuple[str, str]]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "-z", "--full-tree", commit,
         "BioCAS2022"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise PoolContractError("SPRSound Git tree inventory failed")
    entries: Dict[str, Tuple[str, str]] = {}
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            header, path_raw = raw.split(b"\t", 1)
            mode, kind, object_id = header.decode("ascii").split(" ")
            path = path_raw.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise PoolContractError("SPRSound Git tree entry is invalid") from exc
        if kind != "blob" or mode not in ("100644", "100755"):
            raise PoolContractError("SPRSound Git tree contains an unsafe object")
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != path:
            raise PoolContractError("SPRSound Git tree path is unsafe")
        entries[path] = (mode, object_id)
    if not entries:
        raise PoolContractError("SPRSound Git tree is empty")
    return entries


def materialize_git_snapshot(
    repository_root: Union[str, Path], destination: Union[str, Path], *,
    commit: str, tree: str,
) -> GitSnapshotEvidence:
    """Materialize the locked corpus from Git objects, never from the mutable worktree."""
    root = Path(repository_root)
    if _git_output(root, "rev-parse", "%s^{tree}" % commit) != tree:
        raise PoolContractError("SPRSound Git tree does not match the requested identity")
    expected = _git_tree_entries(root, commit)
    target = Path(destination)
    target.mkdir(mode=0o700)
    process = subprocess.Popen(
        ["git", "-C", str(root), "archive", "--format=tar", commit, "BioCAS2022"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    seen = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise PoolContractError("SPRSound Git archive path is unsafe")
                if member.isdir():
                    (target / relative).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or member.name not in expected:
                    raise PoolContractError("SPRSound Git archive contains an unsafe entry")
                source = archive.extractfile(member)
                if source is None:
                    raise PoolContractError("SPRSound Git archive file is unreadable")
                body = source.read()
                mode, object_id = expected[member.name]
                git_digest = hashlib.sha1(
                    b"blob " + str(len(body)).encode("ascii") + b"\0" + body
                ).hexdigest()
                if git_digest != object_id:
                    raise PoolContractError("SPRSound Git archive blob identity mismatch")
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as handle:
                    handle.write(body)
                    handle.flush()
                    os.fsync(handle.fileno())
                output.chmod(0o755 if mode == "100755" else 0o644)
                seen.add(member.name)
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.wait() != 0:
            raise PoolContractError("SPRSound Git archive failed: %s" % stderr.decode(errors="replace"))
    except BaseException:
        process.kill()
        process.wait()
        raise
    if seen != set(expected):
        raise PoolContractError("SPRSound Git archive file set is incomplete")
    evidence = GitSnapshotEvidence(
        root=target, repository_root=root, commit=commit, tree=tree,
        entries=dict(expected),
    )
    _verify_git_snapshot(evidence)
    return evidence


def _verify_git_snapshot(evidence: GitSnapshotEvidence) -> None:
    root = evidence.root
    if root.is_symlink() or not root.is_dir():
        raise PoolContractError("SPRSound Git snapshot root is unsafe")
    repository = evidence.repository_root
    if repository.is_symlink() or not repository.is_dir():
        raise PoolContractError("SPRSound Git object repository is unsafe")
    if _git_output(repository, "rev-parse", "%s^{tree}" % evidence.commit) != evidence.tree:
        raise PoolContractError("SPRSound Git snapshot tree is not backed by its object database")
    object_entries = _git_tree_entries(repository, evidence.commit)
    if object_entries != dict(evidence.entries):
        raise PoolContractError("SPRSound Git snapshot entries are not from the locked tree")
    if evidence.commit == SPRSOUND_COMMIT:
        if evidence.tree != SPRSOUND_TREE:
            raise PoolContractError("SPRSound production tree is not pinned")
        if _git_output(repository, "remote", "get-url", "origin").rstrip("/") != SPRSOUND_REPOSITORY:
            raise PoolContractError("SPRSound production object repository origin is not pinned")
    actual = set()
    for directory, dirnames, filenames in os.walk(str(root), followlinks=False):
        base = Path(directory)
        for name in dirnames:
            if (base / name).is_symlink():
                raise PoolContractError("SPRSound Git snapshot contains a symlink")
        for name in filenames:
            path = base / name
            if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                raise PoolContractError("SPRSound Git snapshot contains an unsafe file")
            relative = path.relative_to(root).as_posix()
            if relative not in evidence.entries:
                raise PoolContractError("SPRSound Git snapshot has an extra file")
            body = path.read_bytes()
            mode, object_id = evidence.entries[relative]
            digest = hashlib.sha1(
                b"blob " + str(len(body)).encode("ascii") + b"\0" + body
            ).hexdigest()
            observed_mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
            if digest != object_id or observed_mode != mode:
                raise PoolContractError("SPRSound Git snapshot blob or mode drifted")
            actual.add(relative)
    if actual != set(evidence.entries):
        raise PoolContractError("SPRSound Git snapshot exact file set is incomplete")


def verify_sprsound_checkout(repository_root: Union[str, Path]) -> Tuple[Path, Dict[str, object]]:
    root = Path(repository_root)
    if root.is_symlink() or not root.is_dir():
        raise PoolContractError("SPRSound checkout must be a real directory")
    if _git_output(root, "rev-parse", "HEAD") != SPRSOUND_COMMIT:
        raise PoolContractError("SPRSound checkout commit is not pinned")
    if _git_output(root, "rev-parse", "HEAD^{tree}") != SPRSOUND_TREE:
        raise PoolContractError("SPRSound checkout tree is not pinned")
    if _git_output(root, "remote", "get-url", "origin").rstrip("/") != SPRSOUND_REPOSITORY:
        raise PoolContractError("SPRSound checkout origin is not pinned")
    if _git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise PoolContractError("SPRSound checkout is dirty or has untracked files")
    entries = _git_tree_entries(root, SPRSOUND_COMMIT)
    wav_names = [PurePosixPath(path).stem for path in entries if path.lower().endswith(".wav")]
    metadata_names = [PurePosixPath(path).stem for path in entries if path.lower().endswith(".json")]
    if len(wav_names) != len(set(wav_names)) or len(metadata_names) != len(set(metadata_names)):
        raise PoolContractError("SPRSound Git tree contains ambiguous source basenames")
    pairs = len(set(wav_names) & set(metadata_names))
    if pairs != SPRSOUND_PAIR_COUNT:
        raise PoolContractError("SPRSound BioCAS2022 pair inventory is not 2683")
    return root, {
        "commit": SPRSOUND_COMMIT,
        "corpus": "BioCAS2022",
        "pair_count": pairs,
        "repository": SPRSOUND_REPOSITORY,
        "schema": "wam_sprsound_git_source_identity_v2",
        "tree": SPRSOUND_TREE,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build an inactive exact SPRSound pool candidate")
    parser.add_argument("--sprsound", required=True, help="pinned SPRSound repository checkout")
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    repository, source = verify_sprsound_checkout(args.sprsound)
    with tempfile.TemporaryDirectory(prefix="wam-sprsound-object-snapshot-") as temporary:
        snapshot = materialize_git_snapshot(
            repository, Path(temporary) / "source", commit=SPRSOUND_COMMIT,
            tree=SPRSOUND_TREE,
        )
        release = build_pool(
            snapshot.root / "BioCAS2022", args.pool_root,
            source_identity=source, seed=args.seed, git_snapshot=snapshot,
        )
    verified = verify_pool_release(release)
    print(contract.canonical_json_bytes({
        "counts": verified["counts"],
        "release_path": release.as_posix(),
        "release_id": verified["release_id"],
        "schema": "wam_sprsound_pool_build_result_v1",
        "total": verified["total"],
    }).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
