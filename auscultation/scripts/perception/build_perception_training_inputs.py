"""Fail-closed builders for immutable perception-training input releases.

The public entry points pin production identities.  Underscored helpers expose
only deterministic filesystem roots and expected identities for unit tests; no
public API accepts a test switch, alternate hash, or verifier callback.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Dict, Iterator, Mapping, Sequence, Tuple, Union

from ausc.training import perception_canary_contract as canary_contract
from scripts.deployment import artifact_contract


DEFAULT_STAGING_ROOT = Path("/Users/williamqiu/.wam/perception-inputs/.staging")
DEFAULT_H200_STAGING_ROOT = Path(
    "/home/ssh-ziyue/wam_ausc/data/.perception-inputs-staging"
)
POINT_ORDER = ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEQUENCE_ID = re.compile(r"^p[0-9]{6}_a[0-9]{6}$")
_LABEL_FRAME_INDICES = (
    60,
    83,
    107,
    131,
    155,
    179,
    203,
    227,
    251,
    275,
    299,
    322,
    346,
    370,
    394,
    418,
    442,
    466,
    490,
    514,
    538,
    562,
)
_LABEL_META_SCHEMA = "R1,L1..R5,L5 (R=图右 L=图左)"
_LABEL_META_SPACE = "FULL native frame pixels"
_SELECTED_MEMBER_ATTESTATION_SCHEMA = (
    "wam_perception_selected_member_attestation_v1"
)

_GEOMETRY_SUPPORT_OUTPUT_TO_SOURCE = {
    "densepose/UV_Processed.mat": "densepose_coco/uv_data/UV_Processed.mat",
    "smpl/SMPL_NEUTRAL.pkl": "smpl/SMPL_NEUTRAL.pkl",
}
_GEOMETRY_LICENSE_NAMES = {
    "licenses/DENSEPOSE_SOURCE_LICENSE.txt": "DENSEPOSE_SOURCE_LICENSE.txt",
    "licenses/HUMMAN_LICENSE.txt": "HUMMAN_LICENSE.txt",
    "licenses/SMPL_MODEL_LICENSE.html": "SMPL_MODEL_LICENSE.html",
}


class InputBuildError(RuntimeError):
    """An input cannot be proven to satisfy the immutable release contract."""


@dataclass(frozen=True)
class _SessionSpec:
    annotation_artifact_kind: str
    diagnostic_png_count: int
    garment: str
    height_px: int
    intrinsics_sha256: str
    label_domain: str
    label_path: str
    label_sha256: str
    projection_artifact_kind: str
    rgb_sha256: str
    session_id: str
    session_number: int
    source_frames_sha256: str
    subject_pseudonym: str
    width_px: int


@dataclass(frozen=True)
class _ProjectionPins:
    coordinate_metadata_sha256: str
    ordered_depth_list_sha256: str
    partition_list_sha256: Mapping[str, str]
    partition_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InputBuildError("value is not finite canonical JSON") from exc


def _reject_duplicate_object(pairs):  # noqa: ANN001, ANN202
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


def _parse_canonical_json_object(payload: bytes, field: str) -> dict:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise InputBuildError(field + " is not valid canonical JSON") from exc
    if type(value) is not dict or payload != _canonical_json_bytes(value):
        raise InputBuildError(field + " is not a canonical JSON object")
    return value


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise InputBuildError(field + " is not a SHA-256 digest")
    return value


def _safe_root(path: Union[str, Path], field: str, *, create: bool = False) -> Path:
    root = Path(path)
    if not root.is_absolute():
        raise InputBuildError(field + " must be absolute")
    try:
        if create:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise InputBuildError(field + " is missing or unsafe") from exc
    if (
        resolved != root
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise InputBuildError(field + " must be a canonical real directory")
    return root


def _relative_path(value: object, field: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise InputBuildError(field + " is not a canonical relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise InputBuildError(field + " is not a canonical relative path")
    return value


@contextmanager
def _open_regular_no_follow(root: Path, relative: str, field: str) -> Iterator[int]:
    parts = PurePosixPath(_relative_path(relative, field)).parts
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptors = []
    descriptor = None
    try:
        descriptors.append(os.open(str(root), directory_flags))
        for part in parts[:-1]:
            info = os.stat(part, dir_fd=descriptors[-1], follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise InputBuildError(field + " contains a symlink or non-directory")
            descriptors.append(
                os.open(part, directory_flags, dir_fd=descriptors[-1])
            )
        info = os.stat(parts[-1], dir_fd=descriptors[-1], follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise InputBuildError(field + " is not a regular file")
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptors[-1],
        )
        opened = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
            raise InputBuildError(field + " changed while opening")
        yield descriptor
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            getattr(opened, "st_mtime_ns", 0),
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            getattr(after, "st_mtime_ns", 0),
        ):
            raise InputBuildError(field + " changed while reading")
    except InputBuildError:
        raise
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise InputBuildError(field + " is missing or unsafe") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for directory in reversed(descriptors):
            os.close(directory)


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_regular(
    root: Path,
    relative: str,
    field: str,
    *,
    maximum_size: Union[int, None] = None,
) -> bytes:
    with _open_regular_no_follow(root, relative, field) as descriptor:
        if maximum_size is not None:
            if os.fstat(descriptor).st_size > maximum_size:
                raise InputBuildError(
                    field + " exceeds the bounded reader hard limit"
                )
        payload = _read_fd(descriptor)
    if maximum_size is not None and len(payload) > maximum_size:
        raise InputBuildError(field + " exceeds the bounded reader hard limit")
    return payload


def _strict_equal(left: object, right: object) -> bool:
    """Deep equality that never conflates JSON scalar types.

    Python's ``==`` treats ``False == 0`` and ``64 == 64.0`` as equal, so a
    resealed semantic manifest could substitute numeric types without
    detection.  Contract comparison therefore requires the exact type at
    every node in addition to the value.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        # Key sets must match INCLUDING key types: plain set equality hashes
        # True == 1 and 1 == 1.0 together, so a key-type substitution would
        # otherwise pass silently (2026-08-30 adversarial discovery).
        if (
            {(type(key), key) for key in left}
            != {(type(key), key) for key in right}
        ):
            return False
        return all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(
            _strict_equal(item, other) for item, other in zip(left, right)
        )
    return left == right


def _reject_nested_output_root(
    output_root: Union[str, Path],
    source_roots: Sequence[Union[str, Path]],
    field: str,
) -> None:
    """Fail before any write when an output root nests inside a source root.

    Sealing into a source tree would let the output mutate (or be captured
    by) the very bytes it attests, so the rejection must happen before the
    output directory is even created.
    """
    output = Path(output_root)
    if not output.is_absolute():
        raise InputBuildError(field + " must be absolute")
    resolved_output = output.resolve()
    for source in source_roots:
        resolved_source = Path(source).resolve()
        try:
            resolved_output.relative_to(resolved_source)
        except ValueError:
            continue
        raise InputBuildError(
            field + " must not be nested inside a source root"
        )


def _write_new_file(root: Path, relative: str, payload: bytes) -> None:
    relative = _relative_path(relative, "staged output path")
    path = root.joinpath(*PurePosixPath(relative).parts)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise InputBuildError("short write while staging " + relative)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_new_file(
    source_root: Path,
    source_relative: str,
    staging: Path,
    output_relative: str,
    field: str,
) -> Tuple[str, int]:
    with _open_regular_no_follow(source_root, source_relative, field) as descriptor:
        payload = _read_fd(descriptor)
    _write_new_file(staging, output_relative, payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


@contextmanager
def _owned_staging(staging_root: Union[str, Path], prefix: str) -> Iterator[Path]:
    root = _safe_root(staging_root, "staging root", create=True)
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=str(root)))
    try:
        yield staging
    finally:
        shutil.rmtree(str(staging), ignore_errors=True)


def _semantic_member(
    *,
    path: str,
    role: str,
    content_origin: str,
    sha256: str,
    size: int,
) -> dict:
    return {
        "content_origin": content_origin,
        "path": path,
        "role": role,
        "sha256": _require_sha256(sha256, "member SHA-256"),
        "size": size,
    }


def _release_identity(release: Path, envelope: dict) -> dict:
    manifest_bytes = _read_regular(
        release,
        artifact_contract.MANIFEST_NAME,
        "release manifest",
    )
    semantic_bytes = _read_regular(
        release,
        canary_contract.PERCEPTION_INPUT_NAME,
        "perception semantic manifest",
    )
    return {
        "payload_sha256": envelope["payload_sha256"],
        "release_id": envelope["release_id"],
        "release_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "semantic_manifest_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
    }


def _verify_geometry_expected_values(
    expected_support_sha256: Mapping[str, str],
    expected_smpl_manifest_sha256: str,
    expected_source_terms: Mapping[str, object],
) -> Dict[str, str]:
    required = frozenset(
        set(_GEOMETRY_SUPPORT_OUTPUT_TO_SOURCE) | set(_GEOMETRY_LICENSE_NAMES)
    )
    if type(expected_support_sha256) is not dict or frozenset(
        expected_support_sha256
    ) != required:
        raise InputBuildError("geometry support hash closure is not exact")
    normalized = {
        path: _require_sha256(digest, "geometry support SHA-256")
        for path, digest in expected_support_sha256.items()
    }
    _require_sha256(
        expected_smpl_manifest_sha256,
        "HuMMan SMPL member manifest SHA-256",
    )
    if type(expected_source_terms) is not dict:
        raise InputBuildError("geometry SOURCE_TERMS expectation is not an object")
    return normalized


def _geometry_sequence_paths(data_root: Path) -> Tuple[Tuple[str, str], ...]:
    base = data_root / "humman" / "smpl_params"
    try:
        base_info = base.lstat()
        if stat.S_ISLNK(base_info.st_mode) or not stat.S_ISDIR(base_info.st_mode):
            raise InputBuildError("HuMMan SMPL source is unsafe")
        names = sorted(os.listdir(base))
    except OSError as exc:
        raise InputBuildError("HuMMan SMPL source is missing") from exc
    if len(names) != 907 or any(_SEQUENCE_ID.fullmatch(name) is None for name in names):
        raise InputBuildError("HuMMan source must contain exactly 907 canonical sequences")
    rows = []
    for sequence_id in names:
        sequence = base / sequence_id
        try:
            metadata = sequence.lstat()
            entries = os.listdir(sequence)
        except OSError as exc:
            raise InputBuildError("HuMMan sequence is missing or unsafe") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or entries != ["smpl_params.npz"]
        ):
            raise InputBuildError(
                "HuMMan sequence closure must contain only smpl_params.npz"
            )
        source = "humman/smpl_params/%s/smpl_params.npz" % sequence_id
        output = source
        rows.append((source, output))
    return tuple(rows)


def _build_geometry_source_release(
    *,
    data_root: Union[str, Path],
    licenses_root: Union[str, Path],
    releases_root: Union[str, Path],
    staging_root: Union[str, Path],
    expected_support_sha256: Mapping[str, str],
    expected_smpl_manifest_sha256: str,
    expected_source_terms: Mapping[str, object],
) -> Path:
    data = _safe_root(data_root, "geometry data root")
    licenses = _safe_root(licenses_root, "geometry license cache")
    _reject_nested_output_root(
        releases_root, (data, licenses), "geometry releases root",
    )
    _reject_nested_output_root(
        staging_root, (data, licenses), "geometry staging root",
    )
    expected_support = _verify_geometry_expected_values(
        expected_support_sha256,
        expected_smpl_manifest_sha256,
        expected_source_terms,
    )
    sequence_paths = _geometry_sequence_paths(data)

    with _owned_staging(staging_root, ".geometry-") as staging:
        members = []
        smpl_manifest_rows = []
        for source_relative, output_relative in sequence_paths:
            digest, size = _copy_new_file(
                data,
                source_relative,
                staging,
                output_relative,
                "HuMMan SMPL parameter",
            )
            sequence_id = PurePosixPath(output_relative).parts[2]
            smpl_manifest_rows.append(
                ("%s/smpl_params.npz" % sequence_id, digest)
            )
            members.append(
                _semantic_member(
                    path=output_relative,
                    role="smpl_params",
                    content_origin="humman_smpl_params",
                    sha256=digest,
                    size=size,
                )
            )
        smpl_manifest_bytes = "".join(
            "%s  %s\n" % (digest, path)
            for path, digest in sorted(smpl_manifest_rows)
        ).encode("ascii")
        if hashlib.sha256(smpl_manifest_bytes).hexdigest() != (
            expected_smpl_manifest_sha256
        ):
            raise InputBuildError("HuMMan SMPL parameter member manifest is not exact")

        support_specs = []
        for output, source in _GEOMETRY_SUPPORT_OUTPUT_TO_SOURCE.items():
            role = "densepose_uv" if output.startswith("densepose/") else "smpl_neutral"
            origin = "densepose_uv" if role == "densepose_uv" else "smpl_neutral"
            support_specs.append((data, source, output, role, origin))
        for output, source in _GEOMETRY_LICENSE_NAMES.items():
            support_specs.append((licenses, source, output, "license", "license"))
        for source_root, source, output, role, origin in support_specs:
            digest, size = _copy_new_file(
                source_root,
                source,
                staging,
                output,
                "geometry support file",
            )
            if digest != expected_support[output]:
                raise InputBuildError("geometry support hash is not exact: " + output)
            members.append(
                _semantic_member(
                    path=output,
                    role=role,
                    content_origin=origin,
                    sha256=digest,
                    size=size,
                )
            )

        source_terms_bytes = _canonical_json_bytes(expected_source_terms)
        source_terms_path = "provenance/SOURCE_TERMS.json"
        _write_new_file(staging, source_terms_path, source_terms_bytes)
        members.append(
            _semantic_member(
                path=source_terms_path,
                role="provenance",
                content_origin="provenance",
                sha256=hashlib.sha256(source_terms_bytes).hexdigest(),
                size=len(source_terms_bytes),
            )
        )
        semantic = {
            "artifact_kind": "perception_geometry_source",
            "contract": {
                "densepose_uv_count": 1,
                "license_count": 3,
                "provenance_count": 1,
                "raw_rgb_count": 0,
                "smpl_neutral_count": 1,
                "smpl_params_count": 907,
            },
            "members": sorted(members, key=lambda row: row["path"]),
            "resolvable_dependencies": [],
            "schema": canary_contract.PERCEPTION_INPUT_SCHEMA,
            "self_contained": True,
            "session_ids": [],
            "source_attestations": [],
            "source_role": "mac",
        }
        semantic_bytes = _canonical_json_bytes(semantic)
        _write_new_file(
            staging,
            canary_contract.PERCEPTION_INPUT_NAME,
            semantic_bytes,
        )
        paths = [row["path"] for row in members] + [
            canary_contract.PERCEPTION_INPUT_NAME
        ]
        try:
            manifest = artifact_contract.build_manifest(
                staging,
                paths,
                artifact_kind="perception_geometry_source",
                source_role="mac",
            )
            releases = _safe_root(
                releases_root,
                "geometry releases root",
                create=True,
            )
            destination = releases / manifest["release_id"]
            if not os.path.lexists(str(destination)):
                # Content-addressed reuse: an identical rebuild lands on the
                # same release ID and the immutable existing directory is
                # re-verified below instead of being rewritten.
                artifact_contract.write_release(
                    staging,
                    manifest,
                    destination,
                    target_role="mac",
                )
        except (artifact_contract.ArtifactError, OSError) as exc:
            raise InputBuildError("geometry release could not be sealed") from exc
    _verify_geometry_source_release(
        destination,
        expected_support_sha256=expected_support,
        expected_smpl_manifest_sha256=expected_smpl_manifest_sha256,
        expected_source_terms=expected_source_terms,
    )
    return destination


def _verify_geometry_source_release(
    release_dir: Union[str, Path],
    *,
    expected_support_sha256: Mapping[str, str],
    expected_smpl_manifest_sha256: str,
    expected_source_terms: Mapping[str, object],
) -> dict:
    expected_support = _verify_geometry_expected_values(
        expected_support_sha256,
        expected_smpl_manifest_sha256,
        expected_source_terms,
    )
    release = Path(release_dir)
    try:
        envelope = artifact_contract.verify_release(release)
    except (artifact_contract.ArtifactError, OSError) as exc:
        raise InputBuildError("geometry release artifact verification failed") from exc
    if release.name != envelope["release_id"]:
        raise InputBuildError(
            "geometry release directory was renamed away from its content ID"
        )
    payload = envelope["payload"]
    if (
        payload.get("artifact_kind") != "perception_geometry_source"
        or payload.get("source_role") != "mac"
    ):
        raise InputBuildError("geometry release kind/source role is not exact")
    semantic_bytes = _read_regular(
        release,
        canary_contract.PERCEPTION_INPUT_NAME,
        "geometry semantic manifest",
    )
    semantic = _parse_canonical_json_object(
        semantic_bytes,
        "geometry semantic manifest",
    )
    identity = _release_identity(release, envelope)
    try:
        canary_contract._validate_perception_input_manifest(
            semantic,
            payload["files"],
            input_name="geometry_source",
            current_identity=identity,
            expected_partition_sha256=(
                canary_contract.HISTORICAL_MIXED_EVAL_PARTITION_SHA256
            ),
            expected_partition_list_sha256=(
                canary_contract._PRODUCTION_PARTITION_LIST_SHA256
            ),
            expected_train_member_sha256=(
                canary_contract._PRODUCTION_TRAIN_MEMBER_SHA256
            ),
            expected_ordered_depth_list_sha256=(
                canary_contract.HISTORICAL_SESSION2_ORDERED_DEPTH_LIST_SHA256
            ),
            expected_geometry_support_sha256=expected_support,
            expected_geometry_smpl_manifest_sha256=(
                expected_smpl_manifest_sha256
            ),
        )
    except canary_contract.CanaryContractError as exc:
        raise InputBuildError("geometry semantic manifest is invalid") from exc
    source_terms_bytes = _read_regular(
        release,
        "provenance/SOURCE_TERMS.json",
        "geometry SOURCE_TERMS",
    )
    _parse_canonical_json_object(
        source_terms_bytes,
        "geometry SOURCE_TERMS",
    )
    if source_terms_bytes != _canonical_json_bytes(expected_source_terms):
        raise InputBuildError("geometry SOURCE_TERMS provenance is not exact")
    sequence_count = sum(
        member["role"] == "smpl_params" for member in semantic["members"]
    )
    return {
        **identity,
        "file_count": len(payload["files"]),
        "sequence_count": sequence_count,
        "total_bytes": sum(row["size"] for row in payload["files"]),
    }


def _production_geometry_support_sha256() -> dict:
    return {
        "densepose/UV_Processed.mat": (
            canary_contract.GEOMETRY_DENSEPOSE_UV_SHA256
        ),
        "licenses/DENSEPOSE_SOURCE_LICENSE.txt": (
            canary_contract.GEOMETRY_DENSEPOSE_LICENSE_SHA256
        ),
        "licenses/HUMMAN_LICENSE.txt": (
            canary_contract.GEOMETRY_HUMMAN_LICENSE_SHA256
        ),
        "licenses/SMPL_MODEL_LICENSE.html": (
            canary_contract.GEOMETRY_SMPL_LICENSE_SHA256
        ),
        "smpl/SMPL_NEUTRAL.pkl": canary_contract.GEOMETRY_SMPL_NEUTRAL_SHA256,
    }


def _production_geometry_source_terms() -> dict:
    return canary_contract._build_geometry_source_terms_expected(
        _production_geometry_support_sha256(),
        canary_contract.HUMMAN_SMPL_PARAMS_MANIFEST_SHA256,
    )


def build_geometry_source_release(
    data_root: Union[str, Path],
    licenses_root: Union[str, Path],
    releases_root: Union[str, Path],
) -> Path:
    """Build the production Mac-authored geometry source release."""
    return _build_geometry_source_release(
        data_root=data_root,
        licenses_root=licenses_root,
        releases_root=releases_root,
        staging_root=DEFAULT_STAGING_ROOT,
        expected_support_sha256=_production_geometry_support_sha256(),
        expected_smpl_manifest_sha256=(
            canary_contract.HUMMAN_SMPL_PARAMS_MANIFEST_SHA256
        ),
        expected_source_terms=_production_geometry_source_terms(),
    )


def verify_geometry_source_release(release_dir: Union[str, Path]) -> dict:
    """Verify the exact production geometry source identity and closure."""
    return _verify_geometry_source_release(
        release_dir,
        expected_support_sha256=_production_geometry_support_sha256(),
        expected_smpl_manifest_sha256=(
            canary_contract.HUMMAN_SMPL_PARAMS_MANIFEST_SHA256
        ),
        expected_source_terms=_production_geometry_source_terms(),
    )


def _parse_json_object(payload: bytes, field: str) -> dict:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise InputBuildError(field + " is not a valid JSON object") from exc
    if type(value) is not dict:
        raise InputBuildError(field + " is not a JSON object")
    return value


def _exact_keys(value: object, expected: frozenset, field: str) -> dict:
    if type(value) is not dict or frozenset(value) != expected:
        raise InputBuildError(field + " keys are not exact")
    return value


def _finite_pixel_pair(
    value: object,
    *,
    width: int,
    height: int,
    field: str,
) -> Tuple[float, float]:
    if type(value) is not list or len(value) != 2:
        raise InputBuildError(field + " must be an [x,y] pair")
    x, y = value
    if (
        type(x) not in (int, float)
        or type(y) not in (int, float)
        or not math.isfinite(float(x))
        or not math.isfinite(float(y))
        or not 0 <= float(x) < width
        or not 0 <= float(y) < height
    ):
        raise InputBuildError(field + " is non-finite or outside the native frame")
    return float(x), float(y)


def _validate_label_bytes(payload: bytes, spec: object) -> Tuple[dict, Tuple[int, ...]]:
    """Validate the exact 22-frame, native-pixel local annotation schema."""
    document = _parse_json_object(payload, "annotation label")
    if len(document) != len(_LABEL_FRAME_INDICES) + 1 or "_meta" not in document:
        raise InputBuildError("annotation label must contain exactly 22 frames and _meta")
    meta = _exact_keys(
        document["_meta"],
        frozenset(("annotator", "domain", "schema", "space")),
        "annotation _meta",
    )
    if (
        meta["annotator"] != "william"
        or meta["domain"] != spec.label_domain
        or meta["schema"] != _LABEL_META_SCHEMA
        or meta["space"] != _LABEL_META_SPACE
    ):
        raise InputBuildError("annotation _meta does not match the fixed local schema")

    prefix = "orb" if spec.session_id == "orbbec_session2" else "orb3"
    expected_row_keys = frozenset(("anchors", "clip", "frame", "h", "points", "w"))
    seen_frames = set()
    for key, row_value in document.items():
        if key == "_meta":
            continue
        row = _exact_keys(row_value, expected_row_keys, "annotation row " + key)
        frame = row["frame"]
        if type(frame) is not int or frame < 0 or frame >= 602:
            raise InputBuildError("annotation frame is outside the 602-frame video")
        if frame in seen_frames:
            raise InputBuildError("annotation contains a duplicate frame")
        seen_frames.add(frame)
        if key != "%s_%04d" % (prefix, frame):
            raise InputBuildError("annotation row name does not bind its frame")
        if (
            row["clip"] != spec.session_id
            or type(row["w"]) is not int
            or type(row["h"]) is not int
            or row["w"] != spec.width_px
            or row["h"] != spec.height_px
        ):
            raise InputBuildError("annotation row video binding is not exact")
        anchors = row["anchors"]
        if type(anchors) is not list or len(anchors) != 3:
            raise InputBuildError("annotation row must contain exactly three anchors")
        for index, pair in enumerate(anchors):
            _finite_pixel_pair(
                pair,
                width=spec.width_px,
                height=spec.height_px,
                field="annotation anchor %d" % index,
            )
        points = row["points"]
        if type(points) is not dict or tuple(points) != POINT_ORDER:
            raise InputBuildError("annotation point order is not exact")
        for point_name in POINT_ORDER:
            _finite_pixel_pair(
                points[point_name],
                width=spec.width_px,
                height=spec.height_px,
                field="annotation point " + point_name,
            )
    frames = tuple(sorted(seen_frames))
    if frames != _LABEL_FRAME_INDICES:
        raise InputBuildError("annotation labeled-frame set is not exact")
    return document, frames


def _production_mixed_eval_identity() -> dict:
    return {
        "payload_sha256": canary_contract.HISTORICAL_MIXED_EVAL_PAYLOAD_SHA256,
        "release_id": canary_contract.HISTORICAL_MIXED_EVAL_RELEASE_ID,
        "release_manifest_sha256": (
            canary_contract.HISTORICAL_MIXED_EVAL_MANIFEST_SHA256
        ),
    }


def _mixed_eval_identity(release: Path, envelope: dict) -> dict:
    manifest = _read_regular(
        release,
        artifact_contract.MANIFEST_NAME,
        "mixed-eval release manifest",
    )
    return {
        "payload_sha256": envelope["payload_sha256"],
        "release_id": envelope["release_id"],
        "release_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def _verify_expected_identity(value: Mapping[str, str], field: str) -> dict:
    expected = _exact_keys(
        value,
        frozenset(("payload_sha256", "release_id", "release_manifest_sha256")),
        field,
    )
    _require_sha256(expected["payload_sha256"], field + " payload SHA-256")
    _require_sha256(
        expected["release_manifest_sha256"],
        field + " manifest SHA-256",
    )
    if type(expected["release_id"]) is not str or not expected["release_id"]:
        raise InputBuildError(field + " release ID is invalid")
    return dict(expected)


def _session_source_path_closure(
    rows: Sequence[dict],
    session_number: int,
    diagnostic_png_count: int,
) -> Tuple[set, Tuple[str, ...]]:
    base = "orbbec_session%d" % session_number
    expected = {
        base + "/frames.csv",
        base + "/intrinsics.json",
        base + "/rgb.mp4",
    }
    expected.update(
        base + "/depth/d%04d.npy" % index for index in range(602)
    )
    diagnostics = tuple(
        sorted(
            row["path"]
            for row in rows
            if row["path"].startswith(base + "/")
            and row["path"].endswith(".png")
        )
    )
    if len(diagnostics) != diagnostic_png_count:
        raise InputBuildError(
            "mixed-eval Session-%d diagnostic PNG closure is not exact"
            % session_number
        )
    expected.update(diagnostics)
    return expected, diagnostics


# Bounded read for a detached RELEASE_MANIFEST.json snapshot: the production
# mixed-eval manifest is a few hundred kilobytes, so anything past this bound
# is not a manifest and must never be slurped into memory.
_MIXED_EVAL_MANIFEST_MAX_BYTES = 8 * 1024 * 1024


def _load_manifest_snapshot(
    manifest_path: Union[str, Path],
    field: str,
) -> Tuple[dict, str]:
    """Parse a detached release manifest and prove its internal binding.

    A snapshot carries no payload tree, so trust reduces to (a) the envelope
    binding its own canonical payload and (b) the caller pinning the file
    hash through an expected immutable identity.
    """
    candidate = Path(manifest_path)
    if not candidate.is_absolute() or candidate.name in ("", ".", ".."):
        raise InputBuildError(field + " path must be absolute")
    parent = _safe_root(candidate.parent, field + " parent")
    manifest_bytes = _read_regular(
        parent,
        candidate.name,
        field,
        maximum_size=_MIXED_EVAL_MANIFEST_MAX_BYTES,
    )
    envelope = _parse_json_object(manifest_bytes, field)
    payload = envelope.get("payload")
    if type(payload) is not dict:
        raise InputBuildError(field + " payload is invalid")
    payload_sha256 = hashlib.sha256(
        artifact_contract.canonical_json_bytes(payload)
    ).hexdigest()
    artifact_kind = payload.get("artifact_kind")
    if type(artifact_kind) is not str or not artifact_kind:
        raise InputBuildError(field + " artifact kind is invalid")
    if (
        envelope.get("schema") != artifact_contract.ENVELOPE_SCHEMA
        or envelope.get("payload_sha256") != payload_sha256
        or envelope.get("release_id")
        != "%s-%s" % (artifact_kind, payload_sha256[:20])
    ):
        raise InputBuildError(field + " envelope does not bind its payload")
    return envelope, hashlib.sha256(manifest_bytes).hexdigest()


def _verify_mixed_eval_payload(
    payload: Mapping[str, object],
    actual_identity: Mapping[str, str],
    *,
    spec: _SessionSpec,
    expected: Mapping[str, str],
) -> Tuple[dict, Tuple[dict, ...]]:
    rows = payload.get("files")
    if (
        payload.get("artifact_kind") != "perception_eval"
        or payload.get("source_role") != "linbox"
        or type(rows) is not list
        or len(rows) != 1215
    ):
        raise InputBuildError("mixed-eval kind, role, or 1215-file closure is not exact")
    if dict(actual_identity) != dict(expected):
        raise InputBuildError("mixed-eval immutable identity does not match")

    session2_paths, _ = _session_source_path_closure(rows, 2, 2)
    session3_paths, _ = _session_source_path_closure(rows, 3, 3)
    if set(row["path"] for row in rows) != session2_paths | session3_paths:
        raise InputBuildError("mixed-eval source path closure is not exact")
    by_path = {row["path"]: row for row in rows}
    if len(by_path) != 1215:
        raise InputBuildError("mixed-eval source contains duplicate paths")
    session2_intrinsics = by_path["orbbec_session2/intrinsics.json"]["sha256"]
    session3_intrinsics = by_path["orbbec_session3/intrinsics.json"]["sha256"]
    if session2_intrinsics != session3_intrinsics:
        raise InputBuildError("mixed-eval intrinsics are not the exact shared content")

    base = spec.session_id
    required_pins = {
        base + "/frames.csv": spec.source_frames_sha256,
        base + "/intrinsics.json": spec.intrinsics_sha256,
        base + "/rgb.mp4": spec.rgb_sha256,
    }
    for path, digest in required_pins.items():
        _require_sha256(digest, "Session-%d source pin" % spec.session_number)
        if by_path[path]["sha256"] != digest:
            raise InputBuildError("mixed-eval selected source hash is not exact: " + path)
    selected = tuple(
        dict(row)
        for row in rows
        if row["path"].startswith(base + "/")
        and not row["path"].endswith(".png")
    )
    if len(selected) != 605 or tuple(row["path"] for row in selected) != tuple(
        sorted(row["path"] for row in selected)
    ):
        raise InputBuildError("mixed-eval selected member list is not exact")
    return dict(actual_identity), selected


def _verify_mixed_eval_release(
    release_dir: Union[str, Path],
    *,
    spec: _SessionSpec,
    expected_identity: Mapping[str, str],
) -> Tuple[dict, Tuple[dict, ...]]:
    expected = _verify_expected_identity(expected_identity, "expected mixed-eval identity")
    release = _safe_root(release_dir, "mixed-eval release")
    try:
        envelope = artifact_contract.verify_release(release)
    except (artifact_contract.ArtifactError, OSError) as exc:
        raise InputBuildError("mixed-eval artifact verification failed") from exc
    actual_identity = _mixed_eval_identity(release, envelope)
    return _verify_mixed_eval_payload(
        envelope["payload"],
        actual_identity,
        spec=spec,
        expected=expected,
    )


def _verify_mixed_eval_manifest_snapshot(
    mixed_eval_manifest: Union[str, Path],
    *,
    spec: _SessionSpec,
    expected_identity: Mapping[str, str],
) -> Tuple[dict, Tuple[dict, ...]]:
    """Verify mixed-eval membership from a detached manifest snapshot.

    The annotation stage runs on hosts that hold the label and the manifest
    but not the multi-gigabyte mixed-eval payload tree; the expected
    immutable identity (payload hash, release ID, manifest hash) pins the
    snapshot to the exact sealed release without touching payload bytes.
    """
    expected = _verify_expected_identity(expected_identity, "expected mixed-eval identity")
    envelope, manifest_sha256 = _load_manifest_snapshot(
        mixed_eval_manifest,
        "mixed-eval manifest snapshot",
    )
    actual_identity = {
        "payload_sha256": envelope["payload_sha256"],
        "release_id": envelope["release_id"],
        "release_manifest_sha256": manifest_sha256,
    }
    return _verify_mixed_eval_payload(
        envelope["payload"],
        actual_identity,
        spec=spec,
        expected=expected,
    )


def _annotation_semantic(
    *,
    label_sha256: str,
    label_size: int,
    frames: Sequence[int],
    source_identity: Mapping[str, str],
    selected_members: Sequence[dict],
    spec: _SessionSpec,
) -> dict:
    selected = [dict(row) for row in selected_members]
    return {
        "artifact_kind": spec.annotation_artifact_kind,
        "contract": {
            "garment": spec.garment,
            "height_px": spec.height_px,
            "label_domain": spec.label_domain,
            "label_path": spec.label_path,
            "label_sha256": label_sha256,
            "labeled_frame_indices": list(frames),
            "landmark_order": list(POINT_ORDER),
            "pixel_axis_order": "xy",
            "pixel_index_base": 0,
            "pixel_origin": "top_left",
            "selected_member_count": 605,
            "session_id": spec.session_id,
            "subject_pseudonym": spec.subject_pseudonym,
            "width_px": spec.width_px,
        },
        "members": [
            _semantic_member(
                path=spec.label_path,
                role="labels",
                content_origin="local_annotation",
                sha256=label_sha256,
                size=label_size,
            )
        ],
        "resolvable_dependencies": [],
        "schema": canary_contract.PERCEPTION_INPUT_SCHEMA,
        "self_contained": True,
        "session_ids": [spec.session_id],
        "source_attestations": [
            {
                "artifact_kind": "perception_eval",
                "schema": _SELECTED_MEMBER_ATTESTATION_SCHEMA,
                "selected_members": selected,
                "selected_members_sha256": hashlib.sha256(
                    _canonical_json_bytes(selected)
                ).hexdigest(),
                "session_id": spec.session_id,
                "source_manifest_sha256": source_identity[
                    "release_manifest_sha256"
                ],
                "source_payload_sha256": source_identity["payload_sha256"],
                "source_release_id": source_identity["release_id"],
            }
        ],
        "source_role": "mac",
    }


def _read_absolute_regular_file(path: Union[str, Path], field: str) -> bytes:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.name in ("", ".", ".."):
        raise InputBuildError(field + " path must be absolute")
    parent = _safe_root(candidate.parent, field + " parent")
    return _read_regular(parent, candidate.name, field)


def _build_annotation_release(
    *,
    label_path: Union[str, Path],
    mixed_eval_manifest: Union[str, Path],
    releases_root: Union[str, Path],
    staging_root: Union[str, Path],
    spec: _SessionSpec,
    expected_mixed_eval_identity: Mapping[str, str],
) -> Path:
    label_bytes = _read_absolute_regular_file(label_path, "annotation label")
    label_sha256 = hashlib.sha256(label_bytes).hexdigest()
    if label_sha256 != spec.label_sha256:
        raise InputBuildError("annotation label SHA-256 is not the fixed local label")
    _, frames = _validate_label_bytes(label_bytes, spec)
    source_identity, selected = _verify_mixed_eval_manifest_snapshot(
        mixed_eval_manifest,
        spec=spec,
        expected_identity=expected_mixed_eval_identity,
    )
    semantic = _annotation_semantic(
        label_sha256=label_sha256,
        label_size=len(label_bytes),
        frames=frames,
        source_identity=source_identity,
        selected_members=selected,
        spec=spec,
    )
    semantic_bytes = _canonical_json_bytes(semantic)
    freshly_written = False
    with _owned_staging(staging_root, ".annotation-%d-" % spec.session_number) as staging:
        _write_new_file(staging, spec.label_path, label_bytes)
        _write_new_file(
            staging,
            canary_contract.PERCEPTION_INPUT_NAME,
            semantic_bytes,
        )
        try:
            manifest = artifact_contract.build_manifest(
                staging,
                (spec.label_path, canary_contract.PERCEPTION_INPUT_NAME),
                artifact_kind=spec.annotation_artifact_kind,
                source_role="mac",
            )
            releases = _safe_root(
                releases_root,
                "annotation releases root",
                create=True,
            )
            destination = releases / manifest["release_id"]
            if not os.path.lexists(str(destination)):
                # A repeat build of identical content lands on the same
                # content-addressed ID; the sealed release is immutable, so
                # the existing directory is simply re-verified below and
                # returned instead of being rewritten.
                artifact_contract.write_release(
                    staging,
                    manifest,
                    destination,
                    target_role="mac",
                )
                freshly_written = True
        except (artifact_contract.ArtifactError, OSError) as exc:
            raise InputBuildError("annotation release could not be sealed") from exc
    try:
        _verify_annotation_release(
            destination,
            mixed_eval_manifest=mixed_eval_manifest,
            spec=spec,
            expected_mixed_eval_identity=expected_mixed_eval_identity,
        )
    except InputBuildError:
        if freshly_written:
            # Specialized verification is part of the seal: a release this
            # build created but could not fully verify is never published.
            # Pre-existing directories are left untouched as evidence.
            shutil.rmtree(str(destination), ignore_errors=True)
        raise
    return destination


def _verify_annotation_release(
    release_dir: Union[str, Path],
    *,
    mixed_eval_manifest: Union[str, Path],
    spec: _SessionSpec,
    expected_mixed_eval_identity: Mapping[str, str],
) -> dict:
    release = _safe_root(release_dir, "annotation release")
    try:
        envelope = artifact_contract.verify_release(release)
    except (artifact_contract.ArtifactError, OSError) as exc:
        raise InputBuildError("annotation artifact verification failed") from exc
    if release.name != envelope["release_id"]:
        raise InputBuildError(
            "annotation release directory was renamed away from its content ID"
        )
    payload = envelope["payload"]
    rows = payload.get("files")
    if (
        payload.get("artifact_kind") != spec.annotation_artifact_kind
        or payload.get("source_role") != "mac"
        or type(rows) is not list
        or {row["path"] for row in rows}
        != {spec.label_path, canary_contract.PERCEPTION_INPUT_NAME}
    ):
        raise InputBuildError("annotation release kind, role, or file closure is not exact")
    label_bytes = _read_regular(release, spec.label_path, "sealed annotation label")
    label_sha256 = hashlib.sha256(label_bytes).hexdigest()
    if label_sha256 != spec.label_sha256:
        raise InputBuildError("sealed annotation label SHA-256 is not exact")
    _, frames = _validate_label_bytes(label_bytes, spec)
    source_identity, selected = _verify_mixed_eval_manifest_snapshot(
        mixed_eval_manifest,
        spec=spec,
        expected_identity=expected_mixed_eval_identity,
    )
    semantic_bytes = _read_regular(
        release,
        canary_contract.PERCEPTION_INPUT_NAME,
        "annotation semantic manifest",
    )
    semantic = _parse_canonical_json_object(
        semantic_bytes,
        "annotation semantic manifest",
    )
    expected_semantic = _annotation_semantic(
        label_sha256=label_sha256,
        label_size=len(label_bytes),
        frames=frames,
        source_identity=source_identity,
        selected_members=selected,
        spec=spec,
    )
    if not _strict_equal(semantic, expected_semantic):
        raise InputBuildError("annotation semantic manifest contract is not exact")
    identity = _release_identity(release, envelope)
    return {
        **identity,
        "file_count": len(rows),
        "labeled_frame_count": len(frames),
        "selected_member_count": len(selected),
        "total_bytes": sum(row["size"] for row in rows),
    }


def _production_session2_spec() -> _SessionSpec:
    return _SessionSpec(
        annotation_artifact_kind="perception_labels_train",
        diagnostic_png_count=2,
        garment="dark",
        height_px=1080,
        intrinsics_sha256=canary_contract.SESSION2_INTRINSICS_SHA256,
        label_domain="clothed-orbbec",
        label_path="labels/dark.json",
        label_sha256=canary_contract.SESSION2_DARK_LABEL_SHA256,
        projection_artifact_kind="perception_session2_train",
        rgb_sha256=canary_contract.SESSION2_RGB_VIDEO_SHA256,
        session_id="orbbec_session2",
        session_number=2,
        source_frames_sha256=canary_contract.SESSION2_SOURCE_FRAMES_SHA256,
        subject_pseudonym="same-subject-001",
        width_px=1920,
    )


def _production_session3_spec() -> _SessionSpec:
    return _SessionSpec(
        annotation_artifact_kind="perception_labels_holdout",
        diagnostic_png_count=3,
        garment="yellow",
        height_px=1080,
        intrinsics_sha256=canary_contract.SESSION2_INTRINSICS_SHA256,
        label_domain="clothed-orbbec-yellow",
        label_path="labels/yellow.json",
        label_sha256=(
            "075671526052c4cfd0cb59706254dd8b29f011aa5c33b268022a37df3cdbee37"
        ),
        projection_artifact_kind="perception_session3_holdout",
        rgb_sha256=(
            "9f7e3ee5ada60f918551407a0b3d8bb34a1fc196e016f523b2d24367fb51214f"
        ),
        session_id="orbbec_session3",
        session_number=3,
        source_frames_sha256=(
            "8499809fcfd2ce096264a574f97121fd8904ef36dc804dc37ff104d06ff5dd9f"
        ),
        # 2026-08-29: Session 3 is the SAME subject in a different garment
        # (within-subject cross-garment holdout), not a held-out person.
        subject_pseudonym="same-subject-001",
        width_px=1920,
    )


def build_session2_annotation_release(
    label_path: Union[str, Path],
    mixed_eval_manifest: Union[str, Path],
    releases_root: Union[str, Path],
) -> Path:
    """Build the fixed Session-2 dark annotation release on the Mac."""
    return _build_annotation_release(
        label_path=label_path,
        mixed_eval_manifest=mixed_eval_manifest,
        releases_root=releases_root,
        staging_root=DEFAULT_STAGING_ROOT,
        spec=_production_session2_spec(),
        expected_mixed_eval_identity=_production_mixed_eval_identity(),
    )


def build_session3_annotation_release(
    label_path: Union[str, Path],
    mixed_eval_manifest: Union[str, Path],
    releases_root: Union[str, Path],
) -> Path:
    """Build the fixed Session-3 yellow annotation release on the Mac."""
    return _build_annotation_release(
        label_path=label_path,
        mixed_eval_manifest=mixed_eval_manifest,
        releases_root=releases_root,
        staging_root=DEFAULT_STAGING_ROOT,
        spec=_production_session3_spec(),
        expected_mixed_eval_identity=_production_mixed_eval_identity(),
    )


def verify_session2_annotation_release(
    release_dir: Union[str, Path],
    mixed_eval_manifest: Union[str, Path],
) -> dict:
    """Verify the fixed Session-2 annotation and historical source attestation."""
    return _verify_annotation_release(
        release_dir,
        mixed_eval_manifest=mixed_eval_manifest,
        spec=_production_session2_spec(),
        expected_mixed_eval_identity=_production_mixed_eval_identity(),
    )


def verify_session3_annotation_release(
    release_dir: Union[str, Path],
    mixed_eval_manifest: Union[str, Path],
) -> dict:
    """Verify the fixed Session-3 annotation and historical source attestation."""
    return _verify_annotation_release(
        release_dir,
        mixed_eval_manifest=mixed_eval_manifest,
        spec=_production_session3_spec(),
        expected_mixed_eval_identity=_production_mixed_eval_identity(),
    )


_PARTITION_LIST_FIELDS = frozenset(
    (
        "session2_decoded_frame_sha256",
        "session2_selected_sha256",
        "session3_decoded_frame_sha256",
        "session3_denied_sha256",
        "shared_intrinsics_sha256",
    )
)


def _validate_projection_pins(pins: _ProjectionPins) -> _ProjectionPins:
    if type(pins) is not _ProjectionPins:
        raise InputBuildError("projection pins are not the fixed pin object")
    _require_sha256(
        pins.coordinate_metadata_sha256,
        "coordinate metadata SHA-256 pin",
    )
    _require_sha256(
        pins.ordered_depth_list_sha256,
        "ordered depth-list SHA-256 pin",
    )
    _require_sha256(pins.partition_sha256, "content partition SHA-256 pin")
    partition_pins = _exact_keys(
        pins.partition_list_sha256,
        _PARTITION_LIST_FIELDS,
        "content partition list pins",
    )
    for field, digest in partition_pins.items():
        _require_sha256(digest, "content partition %s pin" % field)
    return pins


def _copy_fd_to_temporary(descriptor: int) -> str:
    temporary_descriptor, name = tempfile.mkstemp(prefix="wam-video-", suffix=".mp4")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(temporary_descriptor, view)
                if written <= 0:
                    raise InputBuildError("short write while pinning video for decode")
                view = view[written:]
        os.fsync(temporary_descriptor)
    except BaseException:
        os.close(temporary_descriptor)
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise
    os.close(temporary_descriptor)
    return name


def _decoded_frame_sha256(frame: object, spec: _SessionSpec) -> str:
    if (
        str(getattr(frame, "dtype", "")) != "uint8"
        or tuple(getattr(frame, "shape", ()))
        != (spec.height_px, spec.width_px, 3)
        or not bool(getattr(getattr(frame, "flags", None), "c_contiguous", False))
    ):
        raise InputBuildError("decoded RGB frame shape/dtype/layout is not exact")
    try:
        body = frame.tobytes(order="C")
    except Exception as exc:
        raise InputBuildError("decoded RGB frame is not contiguous BGR8") from exc
    header = _canonical_json_bytes(
        {
            "dtype": "uint8",
            "shape": [spec.height_px, spec.width_px, 3],
            "version": canary_contract.DECODED_FRAME_HASH_VERSION,
        }
    )
    return hashlib.sha256(header + body).hexdigest()


def _decode_video_fd(descriptor: int, spec: _SessionSpec) -> Tuple[str, ...]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise InputBuildError("OpenCV is required to verify RGB video bytes") from exc

    temporary = None
    capture = None
    try:
        if sys.platform.startswith("linux"):
            os.lseek(descriptor, 0, os.SEEK_SET)
            capture = cv2.VideoCapture(
                "/proc/self/fd/%d" % descriptor,
                cv2.CAP_FFMPEG,
            )
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            temporary = _copy_fd_to_temporary(descriptor)
            capture = cv2.VideoCapture(temporary, cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise InputBuildError("RGB video could not be decoded")
        hashes = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if len(hashes) >= 602:
                raise InputBuildError("RGB video contains more than 602 frames")
            hashes.append(_decoded_frame_sha256(frame, spec))
        if len(hashes) != 602:
            raise InputBuildError("RGB video must decode exactly 602 frames")
        return tuple(hashes)
    finally:
        if capture is not None:
            capture.release()
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _decode_video_from_root(
    root: Path,
    relative: str,
    spec: _SessionSpec,
) -> Tuple[str, ...]:
    with _open_regular_no_follow(root, relative, "Session RGB video") as descriptor:
        return _decode_video_fd(descriptor, spec)


def _build_content_partition(
    mixed_eval_release: Union[str, Path],
    *,
    session2_spec: _SessionSpec,
    session3_spec: _SessionSpec,
    expected_mixed_eval_identity: Mapping[str, str],
) -> dict:
    identity2, selected2_rows = _verify_mixed_eval_release(
        mixed_eval_release,
        spec=session2_spec,
        expected_identity=expected_mixed_eval_identity,
    )
    identity3, _ = _verify_mixed_eval_release(
        mixed_eval_release,
        spec=session3_spec,
        expected_identity=expected_mixed_eval_identity,
    )
    if identity2 != identity3:
        raise InputBuildError("mixed-eval identity changed between session verification")
    mixed = _safe_root(mixed_eval_release, "mixed-eval release")
    try:
        envelope = artifact_contract.verify_release(mixed)
    except (artifact_contract.ArtifactError, OSError) as exc:
        raise InputBuildError("mixed-eval changed before content partitioning") from exc
    rows = envelope["payload"]["files"]
    session3_rows = tuple(
        row
        for row in rows
        if row["path"].startswith(session3_spec.session_id + "/")
    )
    session2_selected = sorted({row["sha256"] for row in selected2_rows})
    session3_denied = sorted({row["sha256"] for row in session3_rows})
    if len(session2_selected) != 605 or len(session3_denied) != 608:
        raise InputBuildError("mixed-eval source content hashes are degenerate")
    shared = session2_spec.intrinsics_sha256
    if (
        shared != session3_spec.intrinsics_sha256
        or set(session2_selected) & set(session3_denied) != {shared}
    ):
        raise InputBuildError(
            "cross-session raw content overlap is not exactly shared intrinsics"
        )
    decoded2 = _decode_video_from_root(
        mixed,
        session2_spec.session_id + "/rgb.mp4",
        session2_spec,
    )
    decoded3 = _decode_video_from_root(
        mixed,
        session3_spec.session_id + "/rgb.mp4",
        session3_spec,
    )
    if set(decoded2) & set(decoded3):
        raise InputBuildError("cross-session decoded RGB frame collision detected")
    source_domain2 = (set(session2_selected) - {shared}) | set(decoded2)
    source_domain3 = (set(session3_denied) - {shared}) | set(decoded3)
    if source_domain2 & source_domain3:
        raise InputBuildError("cross-session source/decoded content collision detected")
    return {
        "decoded_frame_hash_version": canary_contract.DECODED_FRAME_HASH_VERSION,
        "schema": canary_contract.MIXED_EVAL_CONTENT_PARTITION_SCHEMA,
        "session2_decoded_frame_sha256": list(decoded2),
        "session2_selected_sha256": session2_selected,
        "session3_decoded_frame_sha256": list(decoded3),
        "session3_denied_sha256": session3_denied,
        "shared_intrinsics_sha256": [shared],
        "source_manifest_sha256": identity2["release_manifest_sha256"],
        "source_payload_sha256": identity2["payload_sha256"],
        "source_release_id": identity2["release_id"],
    }


def _validate_content_partition_pins(partition: dict, pins: _ProjectionPins) -> None:
    _validate_projection_pins(pins)
    if hashlib.sha256(_canonical_json_bytes(partition)).hexdigest() != (
        pins.partition_sha256
    ):
        raise InputBuildError("mixed-eval content partition SHA-256 is not exact")
    for field in sorted(_PARTITION_LIST_FIELDS):
        digest = hashlib.sha256(
            _canonical_json_bytes(partition[field])
        ).hexdigest()
        if digest != pins.partition_list_sha256[field]:
            raise InputBuildError("content partition %s SHA-256 is not exact" % field)


def _validate_source_frame_index_bytes(payload: bytes) -> None:
    try:
        rows = list(
            csv.reader(
                io.StringIO(payload.decode("utf-8", errors="strict"), newline=""),
                strict=True,
            )
        )
    except (UnicodeDecodeError, csv.Error) as exc:
        raise InputBuildError("source frame index is not strict UTF-8 CSV") from exc
    if len(rows) != 603 or rows[0] != [
        "idx",
        "t",
        "cam_index",
        "has_depth",
        "depth_scale",
    ]:
        raise InputBuildError("source frame index schema/count is not exact")
    previous_t = None
    previous_cam_index = None
    for expected_index, row in enumerate(rows[1:]):
        if len(row) != 5 or row[0] != str(expected_index):
            raise InputBuildError("source frame indices are not contiguous")
        if re.fullmatch(r"(?:0|[1-9][0-9]*)", row[2]) is None:
            raise InputBuildError("source cam_index is not canonical")
        try:
            timestamp = float(row[1])
            cam_index = int(row[2])
            depth_scale = float(row[4])
        except ValueError as exc:
            raise InputBuildError("source frame numeric value is malformed") from exc
        if (
            not math.isfinite(timestamp)
            or not math.isfinite(depth_scale)
            or row[3] != "1"
            or depth_scale.hex() != float(1.0).hex()
            or (
                previous_t is not None
                and (timestamp <= previous_t or cam_index <= previous_cam_index)
            )
        ):
            raise InputBuildError("source frame timing/depth metadata is not exact")
        previous_t = timestamp
        previous_cam_index = cam_index


def _coordinate_metadata(spec: _SessionSpec) -> dict:
    return {
        "decoded_frame_hash_version": canary_contract.DECODED_FRAME_HASH_VERSION,
        "depth_unit": "metres",
        "garment": spec.garment,
        "height_px": spec.height_px,
        "landmark_order": list(POINT_ORDER),
        "pixel_axis_order": "xy",
        "pixel_index_base": 0,
        "pixel_origin": "top_left",
        "rgb_depth_registration": "registered",
        "schema": "wam_perception_session%d_coordinate_metadata_v1"
        % spec.session_number,
        "session_id": spec.session_id,
        "subject_pseudonym": spec.subject_pseudonym,
        "width_px": spec.width_px,
    }


def _projection_output_specs(spec: _SessionSpec) -> Tuple[Tuple[str, str, str, str], ...]:
    rows = [
        (
            spec.session_id + "/intrinsics.json",
            "camera/intrinsics.json",
            "intrinsics",
            "shared_intrinsics",
        ),
        (
            spec.session_id + "/frames.csv",
            "source/frames.csv",
            "source_frame_index",
            spec.session_id,
        ),
        (
            spec.session_id + "/rgb.mp4",
            "video/session%d.mp4" % spec.session_number,
            "rgb_video",
            spec.session_id,
        ),
    ]
    rows.extend(
        (
            spec.session_id + "/depth/d%04d.npy" % index,
            "depth/depth_%06d.npy" % index,
            "metric_depth",
            spec.session_id,
        )
        for index in range(602)
    )
    return tuple(rows)


def _copy_new_file_streaming(
    source_root: Path,
    source_relative: str,
    staging: Path,
    output_relative: str,
    field: str,
) -> Tuple[str, int]:
    output_relative = _relative_path(output_relative, "staged output path")
    target = staging.joinpath(*PurePosixPath(output_relative).parts)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with _open_regular_no_follow(source_root, source_relative, field) as source_fd:
        target_fd = os.open(
            str(target),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        try:
            os.lseek(source_fd, 0, os.SEEK_SET)
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    if written <= 0:
                        raise InputBuildError("short write while staging " + output_relative)
                    view = view[written:]
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
    return digest.hexdigest(), total


def _projection_contract(
    spec: _SessionSpec,
    *,
    decoded: Sequence[str],
    ordered_depth: Sequence[str],
) -> dict:
    return {
        "decoded_frame_fingerprints": list(decoded),
        "depth_frame_count": 602,
        "frame_count": 602,
        "frame_ledger_path": "ledgers/frames.json",
        "garment": spec.garment,
        "intrinsics_path": "camera/intrinsics.json",
        "labels_path": spec.label_path,
        "ordered_depth_fingerprints": list(ordered_depth),
        "rgb_video_path": "video/session%d.mp4" % spec.session_number,
        "source_frames_path": "source/frames.csv",
    }


def _projection_semantic(
    *,
    members: Sequence[dict],
    partition: dict,
    spec: _SessionSpec,
) -> dict:
    ordered_depth = [
        next(
            member["sha256"]
            for member in members
            if member["path"] == "depth/depth_%06d.npy" % index
        )
        for index in range(602)
    ]
    decoded = partition[
        "session2_decoded_frame_sha256"
        if spec.session_number == 2
        else "session3_decoded_frame_sha256"
    ]
    return {
        "artifact_kind": spec.projection_artifact_kind,
        "contract": _projection_contract(
            spec,
            decoded=decoded,
            ordered_depth=ordered_depth,
        ),
        "members": sorted((dict(row) for row in members), key=lambda row: row["path"]),
        "resolvable_dependencies": [],
        "schema": canary_contract.PERCEPTION_INPUT_SCHEMA,
        "self_contained": True,
        "session_ids": [spec.session_id],
        "source_attestations": [
            {
                "artifact_kind": "perception_eval",
                "content_partition": deepcopy(partition),
                "payload_sha256": partition["source_payload_sha256"],
                "release_id": partition["source_release_id"],
                "release_manifest_sha256": partition[
                    "source_manifest_sha256"
                ],
            }
        ],
        "source_role": "h200",
    }


def _frame_ledger(
    *,
    decoded: Sequence[str],
    ordered_depth: Sequence[str],
    spec: _SessionSpec,
) -> dict:
    return {
        "frames": [
            {
                "depth_path": "depth/depth_%06d.npy" % index,
                "depth_sha256": ordered_depth[index],
                "frame_index": index,
                "rgb_sha256": decoded[index],
            }
            for index in range(602)
        ],
        "schema": "wam_perception_session%d_frame_ledger_v1"
        % spec.session_number,
    }


@dataclass(frozen=True)
class _H200AuthoringAuthorization:
    """Explicit acknowledgement that a projection claims source_role=h200.

    Projection releases attest H200 authorship; emitting that role from any
    other host would forge provenance. Production acquires this token only
    when the environment states the H200 host role; the underscored test
    constructor exists so unit tests can exercise the sealing logic without
    the H200 host (no public API accepts it).
    """

    authority: str

    _AUTHORITIES = ("wam_host_role_environment", "test_fixture")

    def __post_init__(self) -> None:
        if self.authority not in self._AUTHORITIES:
            raise InputBuildError("projection authoring authority is unknown")


def _require_h200_authorization(value: object) -> "_H200AuthoringAuthorization":
    if not isinstance(value, _H200AuthoringAuthorization):
        raise InputBuildError(
            "projection authoring requires an explicit H200 authorization"
        )
    return value


def _production_h200_authorization() -> "_H200AuthoringAuthorization":
    if os.environ.get("WAM_HOST_ROLE") != "h200":
        raise InputBuildError(
            "projection authoring requires the H200 host role environment"
        )
    return _H200AuthoringAuthorization(authority="wam_host_role_environment")


def _test_h200_authorization() -> "_H200AuthoringAuthorization":
    return _H200AuthoringAuthorization(authority="test_fixture")


def _build_projection_release(
    *,
    mixed_eval_release: Union[str, Path],
    annotation_release: Union[str, Path],
    releases_root: Union[str, Path],
    staging_root: Union[str, Path],
    spec: _SessionSpec,
    peer_spec: _SessionSpec,
    expected_mixed_eval_identity: Mapping[str, str],
    pins: _ProjectionPins,
    authoring_authorization: "_H200AuthoringAuthorization",
) -> Path:
    _require_h200_authorization(authoring_authorization)
    _validate_projection_pins(pins)
    if {spec.session_number, peer_spec.session_number} != {2, 3}:
        raise InputBuildError("projection requires exact Session-2/Session-3 specs")
    _reject_nested_output_root(
        releases_root,
        (mixed_eval_release, annotation_release),
        "projection releases root",
    )
    _reject_nested_output_root(
        staging_root,
        (mixed_eval_release, annotation_release),
        "projection staging root",
    )
    _verify_annotation_release(
        annotation_release,
        mixed_eval_manifest=(
            Path(mixed_eval_release) / artifact_contract.MANIFEST_NAME
        ),
        spec=spec,
        expected_mixed_eval_identity=expected_mixed_eval_identity,
    )
    partition = _build_content_partition(
        mixed_eval_release,
        session2_spec=spec if spec.session_number == 2 else peer_spec,
        session3_spec=spec if spec.session_number == 3 else peer_spec,
        expected_mixed_eval_identity=expected_mixed_eval_identity,
    )
    _validate_content_partition_pins(partition, pins)
    denied = set(
        partition[
            "session3_denied_sha256"
            if spec.session_number == 2
            else "session2_selected_sha256"
        ]
    ) | set(
        partition[
            "session3_decoded_frame_sha256"
            if spec.session_number == 2
            else "session2_decoded_frame_sha256"
        ]
    )
    if spec.label_sha256 in denied:
        raise InputBuildError("projection label collides with the opposite session")

    mixed = _safe_root(mixed_eval_release, "mixed-eval release")
    annotation = _safe_root(annotation_release, "annotation release")
    try:
        source_envelope = artifact_contract.verify_release(mixed)
    except (artifact_contract.ArtifactError, OSError) as exc:
        raise InputBuildError("mixed-eval changed before projection copy") from exc
    source_by_path = {
        row["path"]: row for row in source_envelope["payload"]["files"]
    }
    decoded = partition[
        "session2_decoded_frame_sha256"
        if spec.session_number == 2
        else "session3_decoded_frame_sha256"
    ]
    ordered_depth = [
        source_by_path[spec.session_id + "/depth/d%04d.npy" % index]["sha256"]
        for index in range(602)
    ]
    if hashlib.sha256(_canonical_json_bytes(ordered_depth)).hexdigest() != (
        pins.ordered_depth_list_sha256
    ):
        raise InputBuildError("ordered metric-depth list SHA-256 is not exact")
    source_frames_bytes = _read_regular(
        mixed,
        spec.session_id + "/frames.csv",
        "source frame index",
    )
    _validate_source_frame_index_bytes(source_frames_bytes)
    coordinate_bytes = _canonical_json_bytes(_coordinate_metadata(spec))
    if hashlib.sha256(coordinate_bytes).hexdigest() != (
        pins.coordinate_metadata_sha256
    ):
        raise InputBuildError("coordinate metadata SHA-256 is not exact")
    ledger_bytes = _canonical_json_bytes(
        _frame_ledger(decoded=decoded, ordered_depth=ordered_depth, spec=spec)
    )

    with _owned_staging(staging_root, ".projection-%d-" % spec.session_number) as staging:
        members = []
        for source_path, output_path, role, origin in _projection_output_specs(spec):
            digest, size = _copy_new_file_streaming(
                mixed,
                source_path,
                staging,
                output_path,
                "selected mixed-eval member",
            )
            source_row = source_by_path[source_path]
            if digest != source_row["sha256"] or size != source_row["size"]:
                raise InputBuildError("selected source member changed while projecting")
            members.append(
                _semantic_member(
                    path=output_path,
                    role=role,
                    content_origin=origin,
                    sha256=digest,
                    size=size,
                )
            )
        label_digest, label_size = _copy_new_file_streaming(
            annotation,
            spec.label_path,
            staging,
            spec.label_path,
            "verified annotation label",
        )
        if label_digest != spec.label_sha256:
            raise InputBuildError("annotation label changed while projecting")
        members.append(
            _semantic_member(
                path=spec.label_path,
                role="labels",
                content_origin=(
                    "session2_dark_annotation"
                    if spec.session_number == 2
                    else "session3_yellow_annotation"
                ),
                sha256=label_digest,
                size=label_size,
            )
        )
        generated = (
            (
                "ledgers/frames.json",
                ledger_bytes,
                "frame_ledger",
            ),
            (
                "metadata/coordinates.json",
                coordinate_bytes,
                "coordinate_metadata",
            ),
        )
        for path, payload, role in generated:
            _write_new_file(staging, path, payload)
            members.append(
                _semantic_member(
                    path=path,
                    role=role,
                    content_origin="builder_metadata",
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size=len(payload),
                )
            )
        semantic = _projection_semantic(
            members=members,
            partition=partition,
            spec=spec,
        )
        _write_new_file(
            staging,
            canary_contract.PERCEPTION_INPUT_NAME,
            _canonical_json_bytes(semantic),
        )
        paths = [row["path"] for row in members] + [
            canary_contract.PERCEPTION_INPUT_NAME
        ]
        try:
            manifest = artifact_contract.build_manifest(
                staging,
                paths,
                artifact_kind=spec.projection_artifact_kind,
                source_role="h200",
            )
            releases = _safe_root(
                releases_root,
                "projection releases root",
                create=True,
            )
            destination = releases / manifest["release_id"]
            freshly_written = not os.path.lexists(str(destination))
            if freshly_written:
                artifact_contract.write_release(
                    staging,
                    manifest,
                    destination,
                    target_role="h200",
                )
        except (artifact_contract.ArtifactError, OSError) as exc:
            raise InputBuildError("projection release could not be sealed") from exc
    try:
        _verify_projection_release(
            destination,
            mixed_eval_release=mixed_eval_release,
            annotation_release=annotation_release,
            spec=spec,
            peer_spec=peer_spec,
            expected_mixed_eval_identity=expected_mixed_eval_identity,
            pins=pins,
        )
    except InputBuildError:
        if freshly_written:
            # Specialized verification is part of the seal: a projection this
            # build created but could not fully verify is never published
            # (2026-08-29 adversarial review: a mixed-eval mutation in the
            # post-seal window previously left a completion-marked release
            # behind). Pre-existing directories stay as evidence.
            shutil.rmtree(str(destination), ignore_errors=True)
        raise
    return destination


def _verify_projection_release(
    release_dir: Union[str, Path],
    *,
    mixed_eval_release: Union[str, Path],
    annotation_release: Union[str, Path],
    spec: _SessionSpec,
    peer_spec: _SessionSpec,
    expected_mixed_eval_identity: Mapping[str, str],
    pins: _ProjectionPins,
) -> dict:
    _validate_projection_pins(pins)
    if {spec.session_number, peer_spec.session_number} != {2, 3}:
        raise InputBuildError("projection verifier requires exact Session-2/Session-3 specs")
    _verify_annotation_release(
        annotation_release,
        mixed_eval_manifest=(
            Path(mixed_eval_release) / artifact_contract.MANIFEST_NAME
        ),
        spec=spec,
        expected_mixed_eval_identity=expected_mixed_eval_identity,
    )
    partition = _build_content_partition(
        mixed_eval_release,
        session2_spec=spec if spec.session_number == 2 else peer_spec,
        session3_spec=spec if spec.session_number == 3 else peer_spec,
        expected_mixed_eval_identity=expected_mixed_eval_identity,
    )
    _validate_content_partition_pins(partition, pins)

    release = _safe_root(release_dir, "projection release")
    try:
        envelope = artifact_contract.verify_release(release)
    except (artifact_contract.ArtifactError, OSError) as exc:
        raise InputBuildError("projection artifact verification failed") from exc
    if release.name != envelope["release_id"]:
        raise InputBuildError(
            "projection release directory was renamed away from its content ID"
        )
    payload = envelope["payload"]
    rows = payload.get("files")
    if (
        payload.get("artifact_kind") != spec.projection_artifact_kind
        or payload.get("source_role") != "h200"
        or type(rows) is not list
        or len(rows) != 609
    ):
        raise InputBuildError("projection kind, role, or 609-file closure is not exact")
    marker = _parse_canonical_json_object(
        _read_regular(
            release,
            artifact_contract.COMPLETE_NAME,
            "projection completion marker",
        ),
        "projection completion marker",
    )
    if marker.get("target_role") != "h200":
        raise InputBuildError("projection completion target role is not H200")

    expected_paths = {
        canary_contract.PERCEPTION_INPUT_NAME,
        spec.label_path,
        "ledgers/frames.json",
        "metadata/coordinates.json",
    }
    expected_paths.update(output for _, output, _, _ in _projection_output_specs(spec))
    if {row["path"] for row in rows} != expected_paths:
        raise InputBuildError("projection payload path closure is not exact")
    by_path = {row["path"]: row for row in rows}
    if len(by_path) != 609:
        raise InputBuildError("projection payload contains duplicate paths")

    mixed = _safe_root(mixed_eval_release, "mixed-eval release")
    annotation = _safe_root(annotation_release, "annotation release")
    try:
        source_envelope = artifact_contract.verify_release(mixed)
    except (artifact_contract.ArtifactError, OSError) as exc:
        raise InputBuildError("mixed-eval changed during projection verification") from exc
    source_by_path = {
        row["path"]: row for row in source_envelope["payload"]["files"]
    }
    members = []
    for source_path, output_path, role, origin in _projection_output_specs(spec):
        source_row = source_by_path[source_path]
        output_row = by_path[output_path]
        if (
            output_row["sha256"] != source_row["sha256"]
            or output_row["size"] != source_row["size"]
        ):
            raise InputBuildError("projection copied member does not match source attestation")
        members.append(
            _semantic_member(
                path=output_path,
                role=role,
                content_origin=origin,
                sha256=output_row["sha256"],
                size=output_row["size"],
            )
        )
    source_frames_bytes = _read_regular(
        release,
        "source/frames.csv",
        "projected source frame index",
    )
    _validate_source_frame_index_bytes(source_frames_bytes)
    if hashlib.sha256(source_frames_bytes).hexdigest() != spec.source_frames_sha256:
        raise InputBuildError("projected source frame index SHA-256 is not exact")

    label_bytes = _read_regular(release, spec.label_path, "projected annotation label")
    if hashlib.sha256(label_bytes).hexdigest() != spec.label_sha256:
        raise InputBuildError("projected label SHA-256 is not exact")
    _validate_label_bytes(label_bytes, spec)
    annotation_label = _read_regular(
        annotation,
        spec.label_path,
        "source annotation label",
    )
    if label_bytes != annotation_label:
        raise InputBuildError("projected label bytes differ from verified annotation")
    label_row = by_path[spec.label_path]
    members.append(
        _semantic_member(
            path=spec.label_path,
            role="labels",
            content_origin=(
                "session2_dark_annotation"
                if spec.session_number == 2
                else "session3_yellow_annotation"
            ),
            sha256=label_row["sha256"],
            size=label_row["size"],
        )
    )

    coordinate_bytes = _read_regular(
        release,
        "metadata/coordinates.json",
        "projection coordinate metadata",
    )
    coordinate = _parse_canonical_json_object(
        coordinate_bytes,
        "projection coordinate metadata",
    )
    if (
        coordinate != _coordinate_metadata(spec)
        or hashlib.sha256(coordinate_bytes).hexdigest()
        != pins.coordinate_metadata_sha256
    ):
        raise InputBuildError("projection coordinate metadata is not exact")
    coordinate_row = by_path["metadata/coordinates.json"]
    members.append(
        _semantic_member(
            path="metadata/coordinates.json",
            role="coordinate_metadata",
            content_origin="builder_metadata",
            sha256=coordinate_row["sha256"],
            size=coordinate_row["size"],
        )
    )

    ordered_depth = [
        by_path["depth/depth_%06d.npy" % index]["sha256"]
        for index in range(602)
    ]
    if hashlib.sha256(_canonical_json_bytes(ordered_depth)).hexdigest() != (
        pins.ordered_depth_list_sha256
    ):
        raise InputBuildError("projected ordered depth-list SHA-256 is not exact")
    decoded = partition[
        "session2_decoded_frame_sha256"
        if spec.session_number == 2
        else "session3_decoded_frame_sha256"
    ]
    sealed_decoded = _decode_video_from_root(
        release,
        "video/session%d.mp4" % spec.session_number,
        spec,
    )
    if list(sealed_decoded) != decoded:
        raise InputBuildError("projected RGB decode differs from source partition")

    ledger_bytes = _read_regular(
        release,
        "ledgers/frames.json",
        "projection frame ledger",
    )
    ledger = _parse_canonical_json_object(ledger_bytes, "projection frame ledger")
    expected_ledger = _frame_ledger(
        decoded=decoded,
        ordered_depth=ordered_depth,
        spec=spec,
    )
    if not _strict_equal(ledger, expected_ledger):
        raise InputBuildError("projection frame ledger is not exact")
    ledger_row = by_path["ledgers/frames.json"]
    members.append(
        _semantic_member(
            path="ledgers/frames.json",
            role="frame_ledger",
            content_origin="builder_metadata",
            sha256=ledger_row["sha256"],
            size=ledger_row["size"],
        )
    )

    semantic_bytes = _read_regular(
        release,
        canary_contract.PERCEPTION_INPUT_NAME,
        "projection semantic manifest",
    )
    semantic = _parse_canonical_json_object(
        semantic_bytes,
        "projection semantic manifest",
    )
    expected_semantic = _projection_semantic(
        members=members,
        partition=partition,
        spec=spec,
    )
    if not _strict_equal(semantic, expected_semantic):
        raise InputBuildError("projection semantic manifest is not exact")
    denied = set(
        partition[
            "session3_denied_sha256"
            if spec.session_number == 2
            else "session2_selected_sha256"
        ]
    ) | set(
        partition[
            "session3_decoded_frame_sha256"
            if spec.session_number == 2
            else "session2_decoded_frame_sha256"
        ]
    )
    shared = spec.intrinsics_sha256
    for member in members:
        if member["sha256"] not in denied:
            continue
        if not (
            member["sha256"] == shared
            and member["role"] == "intrinsics"
            and member["content_origin"] == "shared_intrinsics"
        ):
            raise InputBuildError("projection contains opposite-session content")
    identity = _release_identity(release, envelope)
    return {
        **identity,
        "file_count": len(rows),
        "member_content_sha256": sorted({row["sha256"] for row in members}),
        "decoded_frame_sha256": list(decoded),
        "total_bytes": sum(row["size"] for row in rows),
    }


def _verify_projection_pair(
    *,
    session2_release: Union[str, Path],
    session3_release: Union[str, Path],
    mixed_eval_release: Union[str, Path],
    session2_annotation_release: Union[str, Path],
    session3_annotation_release: Union[str, Path],
    session2_spec: _SessionSpec,
    session3_spec: _SessionSpec,
    expected_mixed_eval_identity: Mapping[str, str],
    session2_pins: _ProjectionPins,
    session3_pins: _ProjectionPins,
) -> dict:
    report2 = _verify_projection_release(
        session2_release,
        mixed_eval_release=mixed_eval_release,
        annotation_release=session2_annotation_release,
        spec=session2_spec,
        peer_spec=session3_spec,
        expected_mixed_eval_identity=expected_mixed_eval_identity,
        pins=session2_pins,
    )
    report3 = _verify_projection_release(
        session3_release,
        mixed_eval_release=mixed_eval_release,
        annotation_release=session3_annotation_release,
        spec=session3_spec,
        peer_spec=session2_spec,
        expected_mixed_eval_identity=expected_mixed_eval_identity,
        pins=session3_pins,
    )
    members2 = set(report2["member_content_sha256"])
    members3 = set(report3["member_content_sha256"])
    shared = session2_spec.intrinsics_sha256
    if shared != session3_spec.intrinsics_sha256 or members2 & members3 != {shared}:
        raise InputBuildError("projection member overlap is not exactly shared intrinsics")
    full2 = (members2 - {shared}) | set(report2["decoded_frame_sha256"])
    full3 = (members3 - {shared}) | set(report3["decoded_frame_sha256"])
    if full2 & full3:
        raise InputBuildError("projection pair has raw/decoded/label content collision")
    return {
        "session2_file_count": report2["file_count"],
        "session2_release_id": report2["release_id"],
        "session3_file_count": report3["file_count"],
        "session3_release_id": report3["release_id"],
        "shared_content_sha256": [shared],
    }


def _production_projection_pins(
    spec: _SessionSpec,
    mixed_eval_release: Union[str, Path],
) -> _ProjectionPins:
    mixed = _safe_root(mixed_eval_release, "mixed-eval release")
    _verify_mixed_eval_release(
        mixed,
        spec=spec,
        expected_identity=_production_mixed_eval_identity(),
    )
    try:
        envelope = artifact_contract.verify_release(mixed)
    except (artifact_contract.ArtifactError, OSError) as exc:
        raise InputBuildError("mixed-eval changed while deriving fixed projection pins") from exc
    by_path = {row["path"]: row for row in envelope["payload"]["files"]}
    ordered_depth = [
        by_path[spec.session_id + "/depth/d%04d.npy" % index]["sha256"]
        for index in range(602)
    ]
    ordered_depth_sha256 = hashlib.sha256(
        _canonical_json_bytes(ordered_depth)
    ).hexdigest()
    coordinate_sha256 = hashlib.sha256(
        _canonical_json_bytes(_coordinate_metadata(spec))
    ).hexdigest()
    if spec.session_number == 2 and (
        ordered_depth_sha256
        != canary_contract.HISTORICAL_SESSION2_ORDERED_DEPTH_LIST_SHA256
        or coordinate_sha256
        != canary_contract.SESSION2_COORDINATE_METADATA_SHA256
    ):
        raise InputBuildError("production Session-2 projection pins drifted")
    return _ProjectionPins(
        coordinate_metadata_sha256=coordinate_sha256,
        ordered_depth_list_sha256=ordered_depth_sha256,
        partition_list_sha256=dict(
            canary_contract._PRODUCTION_PARTITION_LIST_SHA256
        ),
        partition_sha256=canary_contract.HISTORICAL_MIXED_EVAL_PARTITION_SHA256,
    )


def build_session2_projection_release(
    mixed_eval_release: Union[str, Path],
    annotation_release: Union[str, Path],
    releases_root: Union[str, Path],
) -> Path:
    """Build the fixed self-contained Session-2 training projection on H200."""
    spec = _production_session2_spec()
    return _build_projection_release(
        mixed_eval_release=mixed_eval_release,
        annotation_release=annotation_release,
        releases_root=releases_root,
        staging_root=DEFAULT_H200_STAGING_ROOT,
        spec=spec,
        peer_spec=_production_session3_spec(),
        expected_mixed_eval_identity=_production_mixed_eval_identity(),
        pins=_production_projection_pins(spec, mixed_eval_release),
        authoring_authorization=_production_h200_authorization(),
    )


def build_session3_projection_release(
    mixed_eval_release: Union[str, Path],
    annotation_release: Union[str, Path],
    releases_root: Union[str, Path],
) -> Path:
    """Build the fixed self-contained Session-3 holdout projection on H200."""
    spec = _production_session3_spec()
    return _build_projection_release(
        mixed_eval_release=mixed_eval_release,
        annotation_release=annotation_release,
        releases_root=releases_root,
        staging_root=DEFAULT_H200_STAGING_ROOT,
        spec=spec,
        peer_spec=_production_session2_spec(),
        expected_mixed_eval_identity=_production_mixed_eval_identity(),
        pins=_production_projection_pins(spec, mixed_eval_release),
        authoring_authorization=_production_h200_authorization(),
    )


def verify_session2_projection_release(
    release_dir: Union[str, Path],
    mixed_eval_release: Union[str, Path],
    annotation_release: Union[str, Path],
) -> dict:
    """Verify the fixed Session-2 projection and every local source byte."""
    spec = _production_session2_spec()
    return _verify_projection_release(
        release_dir,
        mixed_eval_release=mixed_eval_release,
        annotation_release=annotation_release,
        spec=spec,
        peer_spec=_production_session3_spec(),
        expected_mixed_eval_identity=_production_mixed_eval_identity(),
        pins=_production_projection_pins(spec, mixed_eval_release),
    )


def verify_session3_projection_release(
    release_dir: Union[str, Path],
    mixed_eval_release: Union[str, Path],
    annotation_release: Union[str, Path],
) -> dict:
    """Verify the fixed Session-3 projection and every local source byte."""
    spec = _production_session3_spec()
    return _verify_projection_release(
        release_dir,
        mixed_eval_release=mixed_eval_release,
        annotation_release=annotation_release,
        spec=spec,
        peer_spec=_production_session2_spec(),
        expected_mixed_eval_identity=_production_mixed_eval_identity(),
        pins=_production_projection_pins(spec, mixed_eval_release),
    )


def verify_projection_pair(
    session2_release: Union[str, Path],
    session3_release: Union[str, Path],
    mixed_eval_release: Union[str, Path],
    session2_annotation_release: Union[str, Path],
    session3_annotation_release: Union[str, Path],
) -> dict:
    """Verify the production pair and reject any non-shared cross-session content."""
    session2_spec = _production_session2_spec()
    session3_spec = _production_session3_spec()
    return _verify_projection_pair(
        session2_release=session2_release,
        session3_release=session3_release,
        mixed_eval_release=mixed_eval_release,
        session2_annotation_release=session2_annotation_release,
        session3_annotation_release=session3_annotation_release,
        session2_spec=session2_spec,
        session3_spec=session3_spec,
        expected_mixed_eval_identity=_production_mixed_eval_identity(),
        session2_pins=_production_projection_pins(
            session2_spec,
            mixed_eval_release,
        ),
        session3_pins=_production_projection_pins(
            session3_spec,
            mixed_eval_release,
        ),
    )


def _emit_report(report: Mapping[str, object]) -> None:
    sys.stdout.write(
        artifact_contract.canonical_json_bytes(report).decode("utf-8")
    )


def main(argv: Union[Sequence[str], None] = None) -> int:
    """CLI over the pinned public builders; emits one canonical report."""
    parser = argparse.ArgumentParser(
        prog="build_perception_training_inputs",
        description=(
            "Build or verify immutable perception-training input releases"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build_geometry = commands.add_parser(
        "build-geometry", help="seal the geometry source release",
    )
    build_geometry.add_argument("--data-root", required=True)
    build_geometry.add_argument("--licenses-root", required=True)
    build_geometry.add_argument("--releases-root", required=True)

    verify_geometry = commands.add_parser(
        "verify-geometry", help="verify a sealed geometry source release",
    )
    verify_geometry.add_argument("--release-dir", required=True)

    for session in (2, 3):
        build_annotation = commands.add_parser(
            "build-session%d-annotation" % session,
            help="seal the Session-%d annotation release" % session,
        )
        build_annotation.add_argument("--label-path", required=True)
        build_annotation.add_argument("--mixed-eval-manifest", required=True)
        build_annotation.add_argument("--releases-root", required=True)

        verify_annotation = commands.add_parser(
            "verify-session%d-annotation" % session,
            help="verify a sealed Session-%d annotation release" % session,
        )
        verify_annotation.add_argument("--release-dir", required=True)
        verify_annotation.add_argument("--mixed-eval-manifest", required=True)

        build_projection = commands.add_parser(
            "build-session%d-projection" % session,
            help="seal the Session-%d projection release (H200 only)" % session,
        )
        build_projection.add_argument("--mixed-eval-release", required=True)
        build_projection.add_argument("--annotation-release", required=True)
        build_projection.add_argument("--releases-root", required=True)

        verify_projection = commands.add_parser(
            "verify-session%d-projection" % session,
            help="verify a sealed Session-%d projection release" % session,
        )
        verify_projection.add_argument("--release-dir", required=True)
        verify_projection.add_argument("--mixed-eval-release", required=True)
        verify_projection.add_argument("--annotation-release", required=True)

    verify_pair = commands.add_parser(
        "verify-projection-pair",
        help="verify the sealed Session-2/Session-3 projection pair",
    )
    verify_pair.add_argument("--session2-release", required=True)
    verify_pair.add_argument("--session3-release", required=True)
    verify_pair.add_argument("--mixed-eval-release", required=True)
    verify_pair.add_argument("--session2-annotation-release", required=True)
    verify_pair.add_argument("--session3-annotation-release", required=True)

    args = parser.parse_args(argv)
    if args.command == "build-geometry":
        release = build_geometry_source_release(
            args.data_root, args.licenses_root, args.releases_root,
        )
        _emit_report(verify_geometry_source_release(release))
    elif args.command == "verify-geometry":
        _emit_report(verify_geometry_source_release(args.release_dir))
    elif args.command in ("build-session2-annotation", "build-session3-annotation"):
        session = 2 if args.command == "build-session2-annotation" else 3
        build = (
            build_session2_annotation_release
            if session == 2
            else build_session3_annotation_release
        )
        verify = (
            verify_session2_annotation_release
            if session == 2
            else verify_session3_annotation_release
        )
        release = build(
            args.label_path, args.mixed_eval_manifest, args.releases_root,
        )
        _emit_report(verify(release, args.mixed_eval_manifest))
    elif args.command in ("verify-session2-annotation", "verify-session3-annotation"):
        verify = (
            verify_session2_annotation_release
            if args.command == "verify-session2-annotation"
            else verify_session3_annotation_release
        )
        _emit_report(verify(args.release_dir, args.mixed_eval_manifest))
    elif args.command in ("build-session2-projection", "build-session3-projection"):
        session = 2 if args.command == "build-session2-projection" else 3
        build = (
            build_session2_projection_release
            if session == 2
            else build_session3_projection_release
        )
        verify = (
            verify_session2_projection_release
            if session == 2
            else verify_session3_projection_release
        )
        release = build(
            args.mixed_eval_release,
            args.annotation_release,
            args.releases_root,
        )
        _emit_report(
            verify(release, args.mixed_eval_release, args.annotation_release)
        )
    elif args.command in ("verify-session2-projection", "verify-session3-projection"):
        verify = (
            verify_session2_projection_release
            if args.command == "verify-session2-projection"
            else verify_session3_projection_release
        )
        _emit_report(
            verify(
                args.release_dir,
                args.mixed_eval_release,
                args.annotation_release,
            )
        )
    else:
        _emit_report(
            verify_projection_pair(
                args.session2_release,
                args.session3_release,
                args.mixed_eval_release,
                args.session2_annotation_release,
                args.session3_annotation_release,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
