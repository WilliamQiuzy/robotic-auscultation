"""Content-addressed, fail-closed release artifact primitives.

The module intentionally uses only the Python standard library and remains compatible
with Python 3.9 so the exact same verifier can run on the Mac and both Linux hosts.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union
import uuid


MANIFEST_NAME = "RELEASE_MANIFEST.json"
COMPLETE_NAME = "TRANSFER_COMPLETE.json"
PAYLOAD_SCHEMA = "wam_artifact_release_v1"
ENVELOPE_SCHEMA = "wam_artifact_manifest_envelope_v1"
COMPLETE_SCHEMA = "wam_transfer_complete_v1"
_RESERVED = frozenset((MANIFEST_NAME, COMPLETE_NAME))
_ANY_CURRENT = object()
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class ArtifactError(RuntimeError):
    """The requested artifact violates the release contract."""


class VerificationError(ArtifactError):
    """A release does not exactly match its signed manifest."""


class ConflictError(ArtifactError):
    """A caller attempted to overwrite or race existing managed state."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the only JSON representation accepted by the contract."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: Union[str, Path], *, allow_reserved: bool = False) -> str:
    raw = str(value)
    if not raw or raw in (".", "..") or "\\" in raw or "\x00" in raw:
        raise ArtifactError("artifact path must be a normalized relative POSIX path")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts):
        raise ArtifactError("artifact path must not be absolute or traverse parents")
    normalized = candidate.as_posix()
    if raw != normalized:
        raise ArtifactError("artifact path is not canonical POSIX form")
    if not allow_reserved and normalized in _RESERVED:
        raise ArtifactError("artifact payload uses a reserved metadata filename")
    return normalized


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ArtifactError("%s must match %s" % (field, _IDENTIFIER.pattern))
    return value


@contextmanager
def _open_regular_fd(root: Union[str, Path], relative: str) -> Iterator[int]:
    """Open a regular file by descriptor-relative, no-follow traversal."""
    root_path = Path(root)
    parts = PurePosixPath(_relative_path(relative, allow_reserved=True)).parts
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    directory_flags = os.O_RDONLY | nofollow | directory
    descriptors: List[int] = []
    file_descriptor: Optional[int] = None
    try:
        descriptors.append(os.open(str(root_path), directory_flags))
        for part in parts[:-1]:
            entry = os.stat(part, dir_fd=descriptors[-1], follow_symlinks=False)
            if stat.S_ISLNK(entry.st_mode):
                raise ArtifactError("artifact source contains a symlink: %s" % relative)
            descriptors.append(os.open(part, directory_flags, dir_fd=descriptors[-1]))
        entry = os.stat(parts[-1], dir_fd=descriptors[-1], follow_symlinks=False)
        if stat.S_ISLNK(entry.st_mode):
            raise ArtifactError("artifact source contains a symlink: %s" % relative)
        file_descriptor = os.open(
            parts[-1], os.O_RDONLY | nofollow, dir_fd=descriptors[-1]
        )
        info = os.fstat(file_descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactError("artifact source is not a regular file: %s" % relative)
        yield file_descriptor
    except FileNotFoundError as exc:
        raise ArtifactError("artifact source file is missing: %s" % relative) from exc
    except NotADirectoryError as exc:
        raise ArtifactError(
            "artifact source contains a symlink or non-directory: %s" % relative
        ) from exc
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise ArtifactError("artifact source contains a symlink: %s" % relative) from exc
        raise
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _file_identity(info: os.stat_result) -> tuple:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
        getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000)),
    )


def _sha256_fd(descriptor: int) -> str:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    after = os.fstat(descriptor)
    if _file_identity(before) != _file_identity(after):
        raise ArtifactError("artifact file changed while hashing")
    return digest.hexdigest()


def _read_regular_bytes(root: Path, relative: str) -> bytes:
    with _open_regular_fd(root, relative) as descriptor:
        before = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise ArtifactError("artifact file changed while reading")
        return b"".join(chunks)


TRAINING_AUTHENTICITY_NAME = "TRAINING_AUTHENTICITY.json"
TRAINING_SESSION_MANIFEST_NAME = "manifest.json"
TRAINING_SESSION_FINALIZATION_NAME = "FINALIZATION.json"
TRAINING_PROMOTION_CONTRACT = "wam_promotion_v2"
TRAINING_AUTHENTICITY_SCHEMA = "wam_training_authenticity_v2"
TRAINING_COMPONENT_SCHEMA = "wam_training_component_authenticity_v2"
TRAINING_FINALIZATION_SCHEMA = "wam_session_finalization_v2"
TRAINING_TRUSTED_FINALIZATION_SCHEMA = "wam_trusted_training_finalization_v1"
TRAINING_TRUSTED_FINALIZATION_REGISTRY_SCHEMA = (
    "wam_trusted_training_finalization_registry_v1"
)
_TRAINING_COMPONENTS = frozenset(("audio", "force", "geometry", "quality", "robot"))
_TRAINING_REQUIRED_RAW = frozenset((
    "FORCE_CAPTURE_COMPLETE.json",
    "audio.wav",
    "audio_capture.json",
    "audio_input_times.jsonl",
    "audio_times.jsonl",
    "depth/depth_times.csv",
    "force.csv",
    "force_calibration.json",
    "force_capture.json",
    "geometry_frame_authenticity.json",
    "geometry_frame_packets.jsonl",
    "geometry/patient_depth_authenticity.json",
    "geometry/patient_depth_evidence.jsonl",
    "obs.csv",
    "quality_inference_packets.jsonl",
    "quality_inference_authenticity.json",
    "rgb.mp4",
    "rgb_times.csv",
    "robot_capture_summary.json",
    "robot_control_envelopes.jsonl",
    "robot_exact_control.csv",
    "robot_frame_index.jsonl",
    "targets.jsonl",
    "tracker.jsonl",
))
#: Immutable capture kinds this outer contract will verify a training snapshot
#: for.  MUST stay in agreement with
#: ``training_authenticity_finalizer.PRODUCTION_CAPTURE_KINDS`` and
#: ``session_to_lerobot_v2.PRODUCTION_CAPTURE_KINDS``.  A missing, diagnostic, or
#: forged ``production_*`` kind is refused.
_TRAINING_PRODUCTION_CAPTURE_KINDS = frozenset({"production_teleop_v4"})
_TRAINING_COMPONENT_REQUIRED_RAW = {
    "audio": frozenset((
        "audio.wav", "audio_capture.json", "audio_input_times.jsonl",
        "audio_times.jsonl",
    )),
    "force": frozenset((
        "FORCE_CAPTURE_COMPLETE.json", "force.csv", "force_calibration.json",
        "force_capture.json",
    )),
    "geometry": frozenset((
        "depth/depth_times.csv", "geometry_frame_authenticity.json",
        "geometry_frame_packets.jsonl", "geometry/patient_depth_authenticity.json",
        "geometry/patient_depth_evidence.jsonl", "obs.csv", "rgb.mp4",
        "rgb_times.csv", "targets.jsonl", "tracker.jsonl",
    )),
    "quality": frozenset((
        "audio.wav", "audio_times.jsonl", "force.csv",
        "quality_inference_authenticity.json", "quality_inference_packets.jsonl",
    )),
    "robot": frozenset((
        "robot_capture_summary.json", "robot_control_envelopes.jsonl",
        "robot_exact_control.csv", "robot_frame_index.jsonl",
    )),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_RELEASE_ID = re.compile(r"^[a-z][a-z0-9_-]*-[0-9a-f]{20}$")
_TRAINING_DYNAMIC_RAW_PATTERNS = {
    # Capture depth is stored losslessly as ".npy.zlib" (depth_codec.DEPTH_SUFFIX);
    # legacy raw ".npy" sessions stay contract-valid.
    "depth/depth_*.npy": re.compile(r"^depth/depth_[0-9]{6}\.npy(?:\.zlib)?$"),
    "geometry/model_inputs/<sha256>.npy": re.compile(
        r"^geometry/model_inputs/[0-9a-f]{64}\.npy$"
    ),
    "geometry/evidence_arrays/back_<sha256>.npy": re.compile(
        r"^geometry/evidence_arrays/back_[0-9a-f]{64}\.npy$"
    ),
    "geometry/evidence_arrays/robot_<sha256>.npy": re.compile(
        r"^geometry/evidence_arrays/robot_[0-9a-f]{64}\.npy$"
    ),
    "quality_model_inputs/input_<sequence>.f32le": re.compile(
        r"^quality_model_inputs/input_[0-9]{20}\.f32le$"
    ),
}
_TRAINING_DYNAMIC_DIRECTORY_RULES = {
    "depth": (
        frozenset(("depth/depth_times.csv",)),
        ("depth/depth_*.npy",),
    ),
    "geometry/model_inputs": (
        frozenset(),
        ("geometry/model_inputs/<sha256>.npy",),
    ),
    "geometry/evidence_arrays": (
        frozenset(),
        (
            "geometry/evidence_arrays/back_<sha256>.npy",
            "geometry/evidence_arrays/robot_<sha256>.npy",
        ),
    ),
    "quality_model_inputs": (
        frozenset(),
        ("quality_model_inputs/input_<sequence>.f32le",),
    ),
}


def _training_duplicate_safe_object(pairs):  # noqa: ANN001, ANN202
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _training_reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number is forbidden: %s" % value)


def _training_json(root: Path, relative: str, label: str) -> tuple:
    try:
        raw = _read_regular_bytes(root, relative)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_training_duplicate_safe_object,
            parse_constant=_training_reject_constant,
        )
    except (ArtifactError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationError("%s is missing, unsafe, or invalid JSON" % label) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise VerificationError("%s is not a canonical JSON object" % label)
    return value, raw


def _training_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise VerificationError("%s SHA-256 is invalid" % label)
    return value


def _training_release_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _CONTENT_RELEASE_ID.fullmatch(value):
        raise VerificationError("%s content-addressed release id is invalid" % label)
    return value


def _verify_trusted_training_finalization_record(value: object) -> tuple:
    """Pin one finalization to deployment-owned, out-of-session metadata.

    The caller is responsible for loading this record from a controlled registry,
    never from the session being verified.  Static public release hashes alone do
    not prove who produced a session, so the record also binds the exact per-session
    manifest, authenticity receipt, and finalization seal digests.
    """
    expected_fields = {
        "artifact_sha256", "component_verifiers", "finalizer",
        "promotion_contract_version", "schema_version", "session_id",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise VerificationError(
            "training trusted finalization record is missing or has invalid fields"
        )
    if (
        value.get("schema_version") != TRAINING_TRUSTED_FINALIZATION_SCHEMA
        or value.get("promotion_contract_version") != TRAINING_PROMOTION_CONTRACT
    ):
        raise VerificationError("training trusted finalization record is unsupported")
    session_id = value.get("session_id")
    if (
        not isinstance(session_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", session_id)
    ):
        raise VerificationError(
            "training trusted finalization record session identity is invalid"
        )

    artifact_sha256 = value.get("artifact_sha256")
    if (
        not isinstance(artifact_sha256, dict)
        or set(artifact_sha256)
        != {"finalization", "manifest", "training_authenticity"}
    ):
        raise VerificationError(
            "training trusted finalization record artifact identities are incomplete"
        )
    for label in ("finalization", "manifest", "training_authenticity"):
        _training_sha(
            artifact_sha256.get(label),
            "trusted training %s" % label,
        )

    finalizer = value.get("finalizer")
    expected_finalizer_fields = {
        "code_manifest_sha256", "code_release_id", "environment_id",
        "environment_lock_sha256", "module_sha256",
    }
    if not isinstance(finalizer, dict) or set(finalizer) != expected_finalizer_fields:
        raise VerificationError(
            "training trusted finalization record finalizer identity is incomplete"
        )
    finalizer_code_release = _training_release_id(
        finalizer.get("code_release_id"), "trusted finalizer code"
    )
    finalizer_environment = _training_release_id(
        finalizer.get("environment_id"), "trusted finalizer environment"
    )
    for field in ("code_manifest_sha256", "environment_lock_sha256", "module_sha256"):
        _training_sha(finalizer.get(field), "trusted finalizer %s" % field)
    if not finalizer_code_release.endswith(
        "-" + finalizer["code_manifest_sha256"][:20]
    ) or not finalizer_environment.endswith(
        "-" + finalizer["environment_lock_sha256"][:20]
    ):
        raise VerificationError(
            "training trusted finalizer identity is not content-addressed"
        )

    component_verifiers = value.get("component_verifiers")
    if (
        not isinstance(component_verifiers, dict)
        or set(component_verifiers) != _TRAINING_COMPONENTS
    ):
        raise VerificationError(
            "training trusted finalization record component identities are incomplete"
        )
    for component in sorted(_TRAINING_COMPONENTS):
        verifier = component_verifiers[component]
        if (
            not isinstance(verifier, dict)
            or set(verifier)
            != {"code_release_id", "module_path", "module_sha256", "policy_sha256"}
        ):
            raise VerificationError(
                "training trusted component verifier identity is invalid: %s" % component
            )
        component_code_release = _training_release_id(
            verifier.get("code_release_id"),
            "trusted component verifier code",
        )
        if component_code_release != finalizer_code_release:
            raise VerificationError(
                "training trusted component verifier code release is not the trusted "
                "finalizer release: %s" % component
            )
        try:
            _relative_path(verifier.get("module_path"), allow_reserved=True)
        except ArtifactError as exc:
            raise VerificationError(
                "training trusted component verifier module path is unsafe"
            ) from exc
        _training_sha(verifier.get("module_sha256"), "trusted component verifier module")
        _training_sha(verifier.get("policy_sha256"), "trusted component verifier policy")

    try:
        record_raw = canonical_json_bytes(value)
        record = json.loads(
            record_raw.decode("utf-8"),
            object_pairs_hook=_training_duplicate_safe_object,
            parse_constant=_training_reject_constant,
        )
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationError(
            "training trusted finalization record is not canonical JSON data"
        ) from exc
    if record != value:
        raise VerificationError(
            "training trusted finalization record is not canonical JSON data"
        )
    return record, record_raw


def load_trusted_training_finalization_record(
    registry_path: object, *, session_id: object = None,
    finalization_sha256: object = None,
) -> Dict[str, object]:
    """Load one record from an immutable, verifier-owned registry file.

    This loader intentionally accepts no session-relative path and never inspects a
    producer session to discover trust metadata.  A converter selects by its
    externally known session id; the H200 archive agent selects by the finalization
    digest already carried in its authenticated deployment request.
    """
    if not isinstance(registry_path, (str, Path)):
        raise VerificationError(
            "training trusted finalization registry path is required"
        )
    registry = Path(registry_path)
    if not registry.is_absolute():
        raise VerificationError(
            "training trusted finalization registry path must be absolute"
        )
    try:
        resolved = registry.resolve(strict=True)
        metadata = registry.lstat()
        parent_resolved = registry.parent.resolve(strict=True)
        parent_metadata = registry.parent.lstat()
    except OSError as exc:
        raise VerificationError(
            "training trusted finalization registry is missing or unsafe"
        ) from exc
    if resolved != registry or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(
        metadata.st_mode
    ):
        raise VerificationError(
            "training trusted finalization registry is missing or unsafe"
        )
    if (
        parent_resolved != registry.parent
        or stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
    ):
        raise VerificationError(
            "training trusted finalization registry parent is unsafe"
        )
    if metadata.st_mode & 0o022 or parent_metadata.st_mode & 0o022:
        raise VerificationError(
            "training trusted finalization registry and parent must not be "
            "group/world writable"
        )
    geteuid = getattr(os, "geteuid", None)
    if geteuid is not None and (
        metadata.st_uid != geteuid() or parent_metadata.st_uid != geteuid()
    ):
        raise VerificationError(
            "training trusted finalization registry is not owned by the verifier"
        )

    selected_session_id: Optional[str] = None
    if session_id is not None:
        if (
            not isinstance(session_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", session_id)
        ):
            raise VerificationError(
                "training trusted finalization registry session selector is invalid"
            )
        selected_session_id = session_id
    selected_finalization: Optional[str] = None
    if finalization_sha256 is not None:
        selected_finalization = _training_sha(
            finalization_sha256,
            "training trusted finalization registry selector",
        )
    if selected_session_id is None and selected_finalization is None:
        raise VerificationError(
            "training trusted finalization registry selector is required"
        )

    try:
        raw = _read_regular_bytes(registry.parent, registry.name)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_training_duplicate_safe_object,
            parse_constant=_training_reject_constant,
        )
    except (ArtifactError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationError(
            "training trusted finalization registry is invalid JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"records", "schema_version"}
        or value.get("schema_version")
        != TRAINING_TRUSTED_FINALIZATION_REGISTRY_SCHEMA
        or not isinstance(value.get("records"), list)
        or raw != canonical_json_bytes(value)
    ):
        raise VerificationError(
            "training trusted finalization registry schema or encoding is invalid"
        )

    records: List[Dict[str, object]] = []
    seen_session_ids = set()
    seen_finalizations = set()
    for candidate in value["records"]:
        record, _record_raw = _verify_trusted_training_finalization_record(candidate)
        candidate_session_id = record["session_id"]
        candidate_finalization = record["artifact_sha256"]["finalization"]
        if (
            candidate_session_id in seen_session_ids
            or candidate_finalization in seen_finalizations
        ):
            raise VerificationError(
                "training trusted finalization registry contains duplicate identities"
            )
        seen_session_ids.add(candidate_session_id)
        seen_finalizations.add(candidate_finalization)
        if (
            (selected_session_id is None or candidate_session_id == selected_session_id)
            and (
                selected_finalization is None
                or candidate_finalization == selected_finalization
            )
        ):
            records.append(record)
    if len(records) != 1:
        raise VerificationError(
            "training trusted finalization registry has no unique matching record"
        )
    return dict(records[0])


def _verify_training_raw_files(
    session: Path, files: object,
) -> Dict[str, str]:
    if not isinstance(files, dict) or not files:
        raise VerificationError("training raw artifact inventory is empty or invalid")
    hashes: Dict[str, str] = {}
    for untrusted_relative, metadata in files.items():
        try:
            relative = _relative_path(untrusted_relative, allow_reserved=True)
        except ArtifactError as exc:
            raise VerificationError("training raw artifact path is unsafe") from exc
        if relative in {
            TRAINING_SESSION_MANIFEST_NAME,
            TRAINING_SESSION_FINALIZATION_NAME,
            TRAINING_AUTHENTICITY_NAME,
        } or relative.startswith("authenticity/"):
            raise VerificationError("training raw inventory includes finalization metadata")
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"sha256", "size_bytes"}
            or not isinstance(metadata.get("size_bytes"), int)
            or isinstance(metadata.get("size_bytes"), bool)
            or metadata["size_bytes"] <= 0
        ):
            raise VerificationError("training raw artifact metadata is invalid")
        expected_hash = _training_sha(metadata.get("sha256"), "raw artifact")
        try:
            with _open_regular_fd(session, relative) as descriptor:
                info = os.fstat(descriptor)
                if info.st_size != metadata["size_bytes"]:
                    raise VerificationError(
                        "raw artifact size does not match inventory: %s" % relative
                    )
                if _sha256_fd(descriptor) != expected_hash:
                    raise VerificationError(
                        "raw artifact SHA-256 does not match inventory: %s" % relative
                    )
        except VerificationError:
            raise
        except (ArtifactError, OSError) as exc:
            raise VerificationError(
                "raw artifact is missing or unsafe: %s" % relative
            ) from exc
        hashes[relative] = expected_hash
    return hashes


def _verify_training_dynamic_raw_layout(
    raw_hashes: Dict[str, str],
) -> Dict[str, set]:
    """Require exact, lossless dynamic-stream layouts and reject directory debris."""
    observed = {
        label: {
            relative
            for relative in raw_hashes
            if pattern.fullmatch(relative)
        }
        for label, pattern in _TRAINING_DYNAMIC_RAW_PATTERNS.items()
    }
    malformed = []
    for directory, rule in _TRAINING_DYNAMIC_DIRECTORY_RULES.items():
        allowed_static, pattern_labels = rule
        for relative in raw_hashes:
            if relative != directory and not relative.startswith(directory + "/"):
                continue
            if relative in allowed_static:
                continue
            if not any(
                _TRAINING_DYNAMIC_RAW_PATTERNS[label].fullmatch(relative)
                for label in pattern_labels
            ):
                malformed.append(relative)
    if malformed:
        raise VerificationError(
            "training promotion contains noncanonical dynamic raw paths: %s"
            % sorted(malformed)
        )
    return observed


def _verify_training_session_snapshot(
    session_root: Union[str, Path], *, trusted_finalization_record: object = None,
) -> Dict[str, object]:
    """Verify one verifier-controlled snapshot, not a mutable producer path.

    Component-specific finalizers must independently recompute their evidence from the
    raw streams before emitting the five component receipts.  This standard-library
    verifier then binds those receipts, every raw byte, the finalizer release identity,
    the session manifest, and the external finalization seal into one exact contract.
    """
    trusted_record, trusted_record_raw = (
        _verify_trusted_training_finalization_record(trusted_finalization_record)
    )
    trusted_artifact_sha256 = trusted_record["artifact_sha256"]
    trusted_finalizer = trusted_record["finalizer"]
    trusted_component_verifiers = trusted_record["component_verifiers"]

    session = Path(session_root)
    if not session.is_dir() or session.is_symlink():
        raise VerificationError("training session must be a real directory")

    manifest, manifest_raw = _training_json(
        session, TRAINING_SESSION_MANIFEST_NAME, "training manifest"
    )
    if _sha256_bytes(manifest_raw) != trusted_artifact_sha256["manifest"]:
        raise VerificationError(
            "training trusted finalization record does not bind manifest"
        )
    if (
        manifest.get("schema_version") != "wam_session_v2"
        or manifest.get("promotion_contract_version") != TRAINING_PROMOTION_CONTRACT
    ):
        raise VerificationError("training manifest promotion contract is unsupported")
    if (
        manifest.get("status") != "complete"
        or manifest.get("training_eligible") is not True
        or manifest.get("rehearsal_overrides") != []
    ):
        raise VerificationError("training manifest is not fail-closed complete")
    # Outer (third-layer) production-writer gate.  ``training_eligible`` cannot
    # distinguish a diagnostic capture from a production one, so the immutable
    # capture_kind must positively name an allowlisted production kind here as
    # well, never merely be a ``production_*``-shaped or absent string.
    if manifest.get("capture_kind") not in _TRAINING_PRODUCTION_CAPTURE_KINDS:
        raise VerificationError(
            "training manifest capture_kind is not an allowlisted production kind"
        )
    session_id = manifest.get("session_id")
    if (
        not isinstance(session_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", session_id)
    ):
        raise VerificationError("training manifest session identity is invalid")
    if session_id != trusted_record["session_id"]:
        raise VerificationError(
            "training trusted finalization record does not bind session identity"
        )

    inventory = manifest.get("artifact_hashes")
    if (
        not isinstance(inventory, dict)
        or set(inventory) != {"algorithm", "files", "schema_version"}
        or inventory.get("algorithm") != "sha256"
        or inventory.get("schema_version") != "wam_session_artifact_hashes_v1"
    ):
        raise VerificationError("training raw artifact inventory is invalid")
    raw_hashes = _verify_training_raw_files(session, inventory.get("files"))
    dynamic_raw = _verify_training_dynamic_raw_layout(raw_hashes)
    missing_raw = sorted(_TRAINING_REQUIRED_RAW.difference(raw_hashes))
    missing_dynamic = sorted(
        label for label, paths in dynamic_raw.items() if not paths
    )
    if missing_raw or missing_dynamic:
        raise VerificationError(
            "training promotion lacks required raw streams: %s"
            % (missing_raw + missing_dynamic)
        )

    authenticity_ref = manifest.get("training_authenticity")
    if (
        not isinstance(authenticity_ref, dict)
        or set(authenticity_ref) != {"path", "schema_version", "sha256"}
        or authenticity_ref.get("path") != TRAINING_AUTHENTICITY_NAME
        or authenticity_ref.get("schema_version") != TRAINING_AUTHENTICITY_SCHEMA
    ):
        raise VerificationError("training authenticity manifest reference is invalid")
    authenticity_expected_sha = _training_sha(
        authenticity_ref.get("sha256"), "training authenticity"
    )
    authenticity, authenticity_raw = _training_json(
        session, TRAINING_AUTHENTICITY_NAME, "training authenticity receipt"
    )
    authenticity_sha = _sha256_bytes(authenticity_raw)
    if (
        authenticity_sha
        != trusted_artifact_sha256["training_authenticity"]
    ):
        raise VerificationError(
            "training trusted finalization record does not bind authenticity"
        )
    if authenticity_sha != authenticity_expected_sha:
        raise VerificationError("training authenticity SHA-256 does not match manifest")
    expected_authenticity_fields = {
        "components", "finalizer", "promotion_contract_version",
        "raw_artifact_inventory_sha256", "schema_version", "session_id", "status",
        "training_eligible",
    }
    if set(authenticity) != expected_authenticity_fields:
        raise VerificationError("training authenticity fields are invalid")
    if (
        authenticity.get("schema_version") != TRAINING_AUTHENTICITY_SCHEMA
        or authenticity.get("promotion_contract_version") != TRAINING_PROMOTION_CONTRACT
        or authenticity.get("status") != "PASS"
        or authenticity.get("training_eligible") is not True
        or authenticity.get("session_id") != session_id
    ):
        raise VerificationError("training authenticity receipt is not eligible and bound")
    expected_inventory_sha = _sha256_bytes(canonical_json_bytes(inventory))
    if (
        _training_sha(
            authenticity.get("raw_artifact_inventory_sha256"),
            "training raw inventory",
        )
        != expected_inventory_sha
    ):
        raise VerificationError("training authenticity does not bind the raw inventory")

    finalizer = authenticity.get("finalizer")
    expected_finalizer_fields = {
        "code_manifest_sha256", "code_release_id", "environment_id",
        "environment_lock_sha256", "module_sha256",
    }
    if not isinstance(finalizer, dict) or set(finalizer) != expected_finalizer_fields:
        raise VerificationError("training finalizer identity is incomplete")
    if finalizer != trusted_finalizer:
        raise VerificationError(
            "training finalizer identity does not match trusted deployment record"
        )
    finalizer_release = _training_release_id(
        finalizer.get("code_release_id"), "training finalizer code"
    )
    _training_release_id(finalizer.get("environment_id"), "training finalizer environment")
    for field in ("code_manifest_sha256", "environment_lock_sha256", "module_sha256"):
        _training_sha(finalizer.get(field), "training finalizer %s" % field)

    components = authenticity.get("components")
    if not isinstance(components, dict) or set(components) != _TRAINING_COMPONENTS:
        raise VerificationError("training authenticity components are incomplete")
    component_paths = set()
    covered_raw = set()
    for component in sorted(_TRAINING_COMPONENTS):
        reference = components[component]
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise VerificationError("training component reference is invalid: %s" % component)
        expected_path = "authenticity/%s.json" % component
        if reference.get("path") != expected_path:
            raise VerificationError("training component receipt path is invalid: %s" % component)
        expected_receipt_sha = _training_sha(
            reference.get("sha256"), "training component receipt"
        )
        receipt, receipt_raw = _training_json(
            session, expected_path, "training component receipt"
        )
        if _sha256_bytes(receipt_raw) != expected_receipt_sha:
            raise VerificationError(
                "training component receipt SHA-256 mismatch: %s" % component
            )
        expected_receipt_fields = {
            "component", "evidence", "policy_version", "raw_artifacts",
            "schema_version", "status", "training_signal_eligible", "verifier",
        }
        if set(receipt) != expected_receipt_fields:
            raise VerificationError("training component receipt fields are invalid")
        if (
            receipt.get("schema_version") != TRAINING_COMPONENT_SCHEMA
            or receipt.get("component") != component
            or receipt.get("status") != "PASS"
            or receipt.get("training_signal_eligible") is not True
            or not isinstance(receipt.get("evidence"), dict)
            or not receipt["evidence"]
        ):
            raise VerificationError(
                "training component is not independently PASS: %s" % component
            )
        try:
            _identifier(receipt.get("policy_version"), "component policy_version")
        except ArtifactError as exc:
            raise VerificationError("training component policy is invalid") from exc
        verifier = receipt.get("verifier")
        if (
            not isinstance(verifier, dict)
            or set(verifier)
            != {"code_release_id", "module_path", "module_sha256", "policy_sha256"}
            or verifier.get("code_release_id") != finalizer_release
        ):
            raise VerificationError("training component verifier identity is invalid")
        if verifier != trusted_component_verifiers[component]:
            raise VerificationError(
                "training component verifier identity does not match trusted deployment "
                "record: %s" % component
            )
        try:
            _relative_path(verifier.get("module_path"), allow_reserved=True)
        except ArtifactError as exc:
            raise VerificationError("training component verifier module path is unsafe") from exc
        _training_sha(verifier.get("module_sha256"), "component verifier module")
        _training_sha(verifier.get("policy_sha256"), "component verifier policy")
        claimed_raw = receipt.get("raw_artifacts")
        if not isinstance(claimed_raw, dict) or not claimed_raw:
            raise VerificationError("training component has no raw evidence")
        claimed_paths = set()
        for untrusted_relative, claimed_sha in claimed_raw.items():
            try:
                relative = _relative_path(untrusted_relative, allow_reserved=True)
            except ArtifactError as exc:
                raise VerificationError("training component raw path is unsafe") from exc
            if relative not in raw_hashes or claimed_sha != raw_hashes[relative]:
                raise VerificationError(
                    "training component raw binding is invalid: %s" % component
                )
            claimed_paths.add(relative)
            covered_raw.add(relative)
        missing_component_raw = _TRAINING_COMPONENT_REQUIRED_RAW[
            component
        ].difference(claimed_paths)
        missing_component_dynamic = set()
        if component == "geometry":
            geometry_dynamic = set().union(*(
                dynamic_raw[label]
                for label in (
                    "depth/depth_*.npy",
                    "geometry/model_inputs/<sha256>.npy",
                    "geometry/evidence_arrays/back_<sha256>.npy",
                    "geometry/evidence_arrays/robot_<sha256>.npy",
                )
            ))
            missing_component_dynamic = geometry_dynamic.difference(claimed_paths)
        elif component == "quality":
            missing_component_dynamic = dynamic_raw[
                "quality_model_inputs/input_<sequence>.f32le"
            ].difference(claimed_paths)
        if missing_component_raw or missing_component_dynamic:
            raise VerificationError(
                "training component raw requirements are incomplete: %s" % component
            )
        component_paths.add(expected_path)
    if covered_raw != set(raw_hashes):
        raise VerificationError("training component receipts do not cover every raw artifact")

    finalization, finalization_raw = _training_json(
        session, TRAINING_SESSION_FINALIZATION_NAME, "training finalization seal"
    )
    if _sha256_bytes(finalization_raw) != trusted_artifact_sha256["finalization"]:
        raise VerificationError(
            "training trusted finalization record does not bind finalization"
        )
    if set(finalization) != {
        "manifest_sha256", "promotion_contract_version", "schema_version",
        "training_authenticity_sha256",
    }:
        raise VerificationError("training finalization seal fields are invalid")
    if (
        finalization.get("schema_version") != TRAINING_FINALIZATION_SCHEMA
        or finalization.get("promotion_contract_version") != TRAINING_PROMOTION_CONTRACT
        or finalization.get("manifest_sha256") != _sha256_bytes(manifest_raw)
        or finalization.get("training_authenticity_sha256") != authenticity_sha
    ):
        raise VerificationError("training finalization seal does not bind authenticity")

    expected_files = set(raw_hashes) | component_paths | {
        TRAINING_SESSION_MANIFEST_NAME,
        TRAINING_SESSION_FINALIZATION_NAME,
        TRAINING_AUTHENTICITY_NAME,
    }
    found_files = set()
    for directory, dirnames, filenames in os.walk(str(session), followlinks=False):
        base = Path(directory)
        for name in dirnames:
            if (base / name).is_symlink():
                raise VerificationError("training session exact file set contains a symlink")
        for name in filenames:
            path = base / name
            if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                raise VerificationError("training session exact file set is unsafe")
            found_files.add(path.relative_to(session).as_posix())
    if found_files != expected_files:
        raise VerificationError("training session exact file set does not match contract")

    # A session is a multi-file transaction.  Per-file stable descriptor reads
    # prevent torn reads within one file; this second full pass prevents a
    # producer from changing an already-verified stream while later receipts
    # are being checked.
    try:
        if _verify_training_raw_files(session, inventory.get("files")) != raw_hashes:
            raise VerificationError("training raw inventory changed")
        observed_manifest, observed_manifest_raw = _training_json(
            session, TRAINING_SESSION_MANIFEST_NAME, "training manifest"
        )
        observed_authenticity, observed_authenticity_raw = _training_json(
            session, TRAINING_AUTHENTICITY_NAME, "training authenticity receipt"
        )
        observed_finalization, observed_finalization_raw = _training_json(
            session, TRAINING_SESSION_FINALIZATION_NAME, "training finalization seal"
        )
        if (
            observed_manifest != manifest
            or observed_manifest_raw != manifest_raw
            or observed_authenticity != authenticity
            or observed_authenticity_raw != authenticity_raw
            or observed_finalization != finalization
            or observed_finalization_raw != finalization_raw
        ):
            raise VerificationError("training metadata changed")
        for component in sorted(_TRAINING_COMPONENTS):
            relative = components[component]["path"]
            _, receipt_raw = _training_json(
                session, relative, "training component receipt"
            )
            if _sha256_bytes(receipt_raw) != components[component]["sha256"]:
                raise VerificationError("training component receipt changed")
    except VerificationError as exc:
        raise VerificationError(
            "training session changed during verification"
        ) from exc

    raw_payload_sha256 = _sha256_bytes(canonical_json_bytes({
        "artifact_hashes": inventory,
        "promotion_contract_version": TRAINING_PROMOTION_CONTRACT,
        "schema": "wam_raw_session_payload_identity_v2",
        "training_authenticity_sha256": authenticity_sha,
        "trusted_finalization_record_sha256": _sha256_bytes(trusted_record_raw),
    }))
    return {
        "finalization_sha256": _sha256_bytes(finalization_raw),
        "manifest_sha256": _sha256_bytes(manifest_raw),
        "promotion_contract_version": TRAINING_PROMOTION_CONTRACT,
        "raw_payload_sha256": raw_payload_sha256,
        "session_id": session_id,
        "training_authenticity_sha256": authenticity_sha,
        "trusted_finalization_record_sha256": _sha256_bytes(trusted_record_raw),
        "verified_components": sorted(_TRAINING_COMPONENTS),
        "verified_raw_files": len(raw_hashes),
    }


def _training_source_file_paths(session: Path) -> List[str]:
    paths: List[str] = []
    try:
        for directory, dirnames, filenames in os.walk(
            str(session), followlinks=False
        ):
            base = Path(directory)
            for name in dirnames:
                candidate = base / name
                if candidate.is_symlink():
                    raise VerificationError(
                        "training source snapshot contains a symlinked directory"
                    )
            for name in filenames:
                candidate = base / name
                try:
                    metadata = candidate.lstat()
                except OSError as exc:
                    raise VerificationError(
                        "training source snapshot changed during enumeration"
                    ) from exc
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise VerificationError(
                        "training source snapshot contains an unsafe file"
                    )
                try:
                    relative = _relative_path(
                        candidate.relative_to(session).as_posix(),
                        allow_reserved=True,
                    )
                except (ArtifactError, ValueError) as exc:
                    raise VerificationError(
                        "training source snapshot path is unsafe"
                    ) from exc
                paths.append(relative)
    except VerificationError:
        raise
    except OSError as exc:
        raise VerificationError(
            "training source snapshot could not be enumerated"
        ) from exc
    if len(paths) != len(set(paths)):
        raise VerificationError("training source snapshot has duplicate paths")
    return sorted(paths)


def _copy_regular_file_to_snapshot(
    source: Path, relative: str, destination: Path,
) -> None:
    target = destination / relative
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    target_descriptor: Optional[int] = None
    try:
        with _open_regular_fd(source, relative) as source_descriptor:
            before = os.fstat(source_descriptor)
            target_descriptor = os.open(
                str(target),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                0o600,
            )
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                offset = 0
                while offset < len(chunk):
                    written = os.write(target_descriptor, chunk[offset:])
                    if written <= 0:
                        raise OSError(errno.EIO, "short snapshot write")
                    offset += written
            os.fsync(target_descriptor)
            after = os.fstat(source_descriptor)
            if _file_identity(before) != _file_identity(after):
                raise VerificationError(
                    "training source file changed while snapshotting: %s" % relative
                )
    except VerificationError:
        raise
    except (ArtifactError, OSError) as exc:
        raise VerificationError(
            "training source file is missing, unsafe, or unstable: %s" % relative
        ) from exc
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)


def _materialize_training_snapshot(source: Path, staging: Path) -> None:
    try:
        staging.mkdir(mode=0o700)
        for relative in _training_source_file_paths(source):
            _copy_regular_file_to_snapshot(source, relative, staging)
        _fsync_tree_directories(staging)
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise


def _make_training_snapshot_read_only(snapshot: Path) -> None:
    for directory, _dirnames, filenames in os.walk(
        str(snapshot), topdown=False, followlinks=False
    ):
        base = Path(directory)
        for name in filenames:
            os.chmod(str(base / name), 0o400, follow_symlinks=False)
        os.chmod(str(base), 0o500, follow_symlinks=False)


def _trusted_training_snapshot_root(
    value: object, *, source: Path,
) -> Path:
    if not isinstance(value, (str, Path)):
        raise VerificationError("training trusted snapshot root is required")
    root = Path(value)
    source_resolved = source.resolve()
    try:
        root_parent = root.parent
        if not root_parent.is_dir() or root_parent.is_symlink():
            raise VerificationError(
                "training trusted snapshot parent is missing or unsafe"
            )
        root.mkdir(mode=0o700, exist_ok=True)
        metadata = root.lstat()
    except VerificationError:
        raise
    except OSError as exc:
        raise VerificationError(
            "training trusted snapshot root cannot be prepared"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise VerificationError("training trusted snapshot root is unsafe")
    if metadata.st_mode & 0o022:
        raise VerificationError(
            "training trusted snapshot root must not be group/world writable"
        )
    geteuid = getattr(os, "geteuid", None)
    if geteuid is not None and metadata.st_uid != geteuid():
        raise VerificationError(
            "training trusted snapshot root is not owned by the verifier"
        )
    root_resolved = root.resolve()
    try:
        if os.path.commonpath((str(source_resolved), str(root_resolved))) == str(
            source_resolved
        ):
            raise VerificationError(
                "training trusted snapshot root must be outside the source session"
            )
    except ValueError as exc:
        raise VerificationError("training trusted snapshot root is invalid") from exc
    return root_resolved


def verify_training_session_contract(
    session_root: Union[str, Path], *, trusted_finalization_record: object = None,
    trusted_snapshot_root: object = None,
) -> Dict[str, object]:
    """Verify and publish a stable, content-addressed training-session snapshot.

    ``trusted_finalization_record`` must come from a verifier-controlled deployment
    registry, never from ``session_root``.  ``trusted_snapshot_root`` must likewise
    be controlled by the verifier and inaccessible to the session producer.  A
    successful caller must consume ``verified_snapshot_path`` instead of the mutable
    source path; public static release hashes without this per-session record are not
    proof that the trusted finalizer produced the session.
    """
    trusted_record, trusted_record_raw = (
        _verify_trusted_training_finalization_record(trusted_finalization_record)
    )
    source = Path(session_root)
    if not source.is_dir() or source.is_symlink():
        raise VerificationError("training session must be a real directory")
    snapshots = _trusted_training_snapshot_root(
        trusted_snapshot_root,
        source=source,
    )
    trusted_record_sha256 = _sha256_bytes(trusted_record_raw)
    snapshot_id = "training_session-%s" % trusted_record_sha256[:20]
    final = snapshots / snapshot_id
    lock_path = snapshots / ".training-snapshot-publish.lock"

    # A content-addressed cache is a destination, never an identity oracle for
    # a newly supplied path.  Revalidate the current source even when the same
    # trusted record already has a cached snapshot; otherwise an unrelated or
    # corrupted directory could borrow an older snapshot's identity simply by
    # presenting that record.
    source_result = _verify_training_session_snapshot(
        source,
        trusted_finalization_record=trusted_record,
    )

    with deployment_lock(lock_path):
        if os.path.lexists(str(final)):
            if final.is_symlink() or not final.is_dir():
                raise ConflictError("conflicting trusted training snapshot path")
            result = _verify_training_session_snapshot(
                final,
                trusted_finalization_record=trusted_record,
            )
            if result != source_result:
                raise ConflictError(
                    "cached trusted training snapshot conflicts with current source"
                )
            # Heal a snapshot that a crash left published-but-writable between
            # the atomic rename and the read-only chmod below.  Re-applying is
            # idempotent for an already read-only snapshot, so every retry
            # converges the tree to read-only rather than trusting the cache.
            _make_training_snapshot_read_only(final)
            _fsync_tree_directories(final)
        else:
            staging = snapshots / (
                ".staging-%s-%s" % (snapshot_id, uuid.uuid4().hex)
            )
            try:
                _materialize_training_snapshot(source, staging)
                result = _verify_training_session_snapshot(
                    staging,
                    trusted_finalization_record=trusted_record,
                )
                if result != source_result:
                    raise VerificationError(
                        "training source changed while materializing trusted snapshot"
                    )
                _fsync_tree_directories(staging)
                _rename_noreplace(staging, final)
                _make_training_snapshot_read_only(final)
                _fsync_tree_directories(final)
                _fsync_directory(snapshots)
            except BaseException:
                shutil.rmtree(str(staging), ignore_errors=True)
                raise

    result = dict(result)
    result["verified_snapshot_id"] = snapshot_id
    result["verified_snapshot_path"] = str(final)
    return result


def build_manifest(
    source_root: Union[str, Path],
    relative_paths: Iterable[Union[str, Path]],
    *,
    artifact_kind: str,
    source_role: str,
) -> Dict[str, object]:
    """Build a deterministic manifest envelope without touching destination state."""
    root = Path(source_root)
    if not root.is_dir() or root.is_symlink():
        raise ArtifactError("artifact source root must be a real directory")
    artifact_kind = _identifier(artifact_kind, "artifact_kind")
    source_role = _identifier(source_role, "source_role")

    normalized: List[str] = []
    seen = set()
    for item in relative_paths:
        relative = _relative_path(item)
        if relative in seen:
            raise ArtifactError("duplicate artifact path: %s" % relative)
        seen.add(relative)
        normalized.append(relative)

    files: List[Dict[str, object]] = []
    for relative in sorted(normalized):
        with _open_regular_fd(root, relative) as descriptor:
            info = os.fstat(descriptor)
            files.append({
                "mode": format(stat.S_IMODE(info.st_mode), "04o"),
                "path": relative,
                "sha256": _sha256_fd(descriptor),
                "size": info.st_size,
            })

    payload: Dict[str, object] = {
        "artifact_kind": artifact_kind,
        "files": files,
        "schema": PAYLOAD_SCHEMA,
        "source_role": source_role,
    }
    payload_sha256 = _sha256_bytes(canonical_json_bytes(payload))
    return {
        "payload": payload,
        "payload_sha256": payload_sha256,
        "release_id": "%s-%s" % (artifact_kind, payload_sha256[:20]),
        "schema": ENVELOPE_SCHEMA,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
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


def _load_envelope(release: Path) -> tuple:
    manifest_path = release / MANIFEST_NAME
    try:
        raw = _read_regular_bytes(release, MANIFEST_NAME)
        envelope = json.loads(raw.decode("utf-8"))
    except (ArtifactError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("release manifest is missing or invalid") from exc
    if not isinstance(envelope, dict) or envelope.get("schema") != ENVELOPE_SCHEMA:
        raise VerificationError("release manifest envelope schema is invalid")
    if raw != canonical_json_bytes(envelope):
        raise VerificationError("release manifest is not canonical JSON")
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or payload.get("schema") != PAYLOAD_SCHEMA:
        raise VerificationError("release payload schema is invalid")
    try:
        _identifier(payload.get("artifact_kind"), "artifact_kind")
        _identifier(payload.get("source_role"), "source_role")
    except ArtifactError as exc:
        raise VerificationError("release payload identifier is invalid") from exc
    payload_sha256 = _sha256_bytes(canonical_json_bytes(payload))
    if envelope.get("payload_sha256") != payload_sha256:
        raise VerificationError("release payload hash mismatch")
    expected_release_id = "%s-%s" % (payload.get("artifact_kind"), payload_sha256[:20])
    if envelope.get("release_id") != expected_release_id:
        raise VerificationError("release id mismatch")
    return envelope, raw


def _tree_files(release: Path) -> set:
    found = set()
    for directory, dirnames, filenames in os.walk(str(release), followlinks=False):
        base = Path(directory)
        for name in list(dirnames):
            candidate = base / name
            if candidate.is_symlink():
                raise VerificationError("release contains a symlink: %s" % candidate)
        for name in filenames:
            candidate = base / name
            relative = candidate.relative_to(release).as_posix()
            try:
                info = candidate.lstat()
            except FileNotFoundError as exc:
                raise VerificationError("release changed during verification") from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise VerificationError("release contains a non-regular file: %s" % relative)
            found.add(relative)
    return found


def _fsync_tree_directories(root: Path) -> None:
    directories = []
    for directory, dirnames, _filenames in os.walk(str(root), followlinks=False):
        base = Path(directory)
        for name in dirnames:
            candidate = base / name
            if candidate.is_symlink():
                raise VerificationError("release contains a symlink: %s" % candidate)
        directories.append(base)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _load_completion_marker(release: Path) -> Dict[str, object]:
    try:
        raw = _read_regular_bytes(release, COMPLETE_NAME)
        marker = json.loads(raw.decode("utf-8"))
    except (ArtifactError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("release completion marker is missing or invalid") from exc
    if not isinstance(marker, dict) or raw != canonical_json_bytes(marker):
        raise VerificationError("release completion marker is not canonical JSON")
    return marker


def verify_release(
    release_dir: Union[str, Path], *, require_complete: bool = True
) -> Dict[str, object]:
    """Verify envelope, exact payload set, bytes, modes, and completion marker."""
    release = Path(release_dir)
    if not release.is_dir() or release.is_symlink():
        raise VerificationError("release directory is missing or is a symlink")
    envelope, manifest_bytes = _load_envelope(release)
    payload = envelope["payload"]
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise VerificationError("release file manifest is invalid")

    expected = {MANIFEST_NAME}
    if require_complete:
        expected.add(COMPLETE_NAME)
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise VerificationError("release file row is invalid")
        try:
            relative = _relative_path(row["path"])
        except (ArtifactError, KeyError) as exc:
            raise VerificationError("release file path is invalid") from exc
        if relative in seen:
            raise VerificationError("release manifest contains a duplicate path")
        seen.add(relative)
        expected.add(relative)
        try:
            with _open_regular_fd(release, relative) as descriptor:
                info = os.fstat(descriptor)
                digest = _sha256_fd(descriptor)
        except (ArtifactError, OSError) as exc:
            raise VerificationError("release payload file is missing: %s" % relative) from exc
        if info.st_size != row.get("size"):
            raise VerificationError("release payload size mismatch: %s" % relative)
        if format(stat.S_IMODE(info.st_mode), "04o") != row.get("mode"):
            raise VerificationError("release payload mode mismatch: %s" % relative)
        if digest != row.get("sha256"):
            raise VerificationError("release payload hash mismatch: %s" % relative)

    found = _tree_files(release)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if require_complete and COMPLETE_NAME in missing:
            raise VerificationError("release completion marker is missing")
        raise VerificationError("release exact-set mismatch; missing=%r extra=%r" % (missing, extra))

    if require_complete:
        marker_path = release / COMPLETE_NAME
        marker = _load_completion_marker(release)
        try:
            _identifier(marker.get("target_role"), "target_role")
        except ArtifactError as exc:
            raise VerificationError("release completion target role is invalid") from exc
        expected_marker = {
            "artifact_kind": payload["artifact_kind"],
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "payload_sha256": envelope["payload_sha256"],
            "release_id": envelope["release_id"],
            "schema": COMPLETE_SCHEMA,
            "source_role": payload["source_role"],
            "target_role": marker.get("target_role"),
            "verified_files": len(rows),
        }
        if not marker.get("target_role") or marker != expected_marker:
            raise VerificationError("release completion marker does not bind the payload")
    return envelope


def write_release(
    source_root: Union[str, Path],
    manifest: Dict[str, object],
    destination: Union[str, Path],
    *,
    target_role: str,
) -> Path:
    """Materialize, verify, mark, and atomically publish one completed release."""
    source = Path(source_root)
    destination_path = Path(destination)
    target_role = _identifier(target_role, "target_role")
    if os.path.lexists(str(destination_path)):
        raise ConflictError("release destination already exists")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = destination_path.parent / (".staging-write-%s" % uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    try:
        payload = manifest.get("payload") if isinstance(manifest, dict) else None
        if not isinstance(payload, dict):
            raise ArtifactError("manifest payload is invalid")
        expected_hash = _sha256_bytes(canonical_json_bytes(payload))
        if manifest.get("schema") != ENVELOPE_SCHEMA or manifest.get("payload_sha256") != expected_hash:
            raise ArtifactError("manifest envelope does not bind its payload")
        expected_id = "%s-%s" % (payload.get("artifact_kind"), expected_hash[:20])
        if manifest.get("release_id") != expected_id:
            raise ArtifactError("manifest release id is invalid")

        for row in payload.get("files", []):
            relative = _relative_path(row["path"])
            with _open_regular_fd(source, relative) as source_descriptor:
                source_info = os.fstat(source_descriptor)
                digest = _sha256_fd(source_descriptor)
                if source_info.st_size != row.get("size") or digest != row.get("sha256"):
                    raise ArtifactError("source changed after manifest construction: %s" % relative)
                if format(stat.S_IMODE(source_info.st_mode), "04o") != row.get("mode"):
                    raise ArtifactError("source mode changed after manifest construction: %s" % relative)
                target = staging / Path(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                before_copy = _file_identity(os.fstat(source_descriptor))
                os.lseek(source_descriptor, 0, os.SEEK_SET)
                with target.open("xb") as target_handle:
                    while True:
                        chunk = os.read(source_descriptor, 1024 * 1024)
                        if not chunk:
                            break
                        target_handle.write(chunk)
                    os.fchmod(target_handle.fileno(), int(row["mode"], 8))
                    target_handle.flush()
                    os.fsync(target_handle.fileno())
                if before_copy != _file_identity(os.fstat(source_descriptor)):
                    raise ArtifactError("source changed while copying: %s" % relative)

        manifest_bytes = canonical_json_bytes(manifest)
        _atomic_write(staging / MANIFEST_NAME, manifest_bytes)
        verify_release(staging, require_complete=False)
        marker = {
            "artifact_kind": payload["artifact_kind"],
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "payload_sha256": manifest["payload_sha256"],
            "release_id": manifest["release_id"],
            "schema": COMPLETE_SCHEMA,
            "source_role": payload["source_role"],
            "target_role": target_role,
            "verified_files": len(payload["files"]),
        }
        _atomic_write(staging / COMPLETE_NAME, canonical_json_bytes(marker))
        verify_release(staging)
        _fsync_tree_directories(staging)
        _rename_noreplace(staging, destination_path)
        _fsync_directory(destination_path.parent)
        return destination_path
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise


@contextmanager
def deployment_lock(lock_path: Union[str, Path]) -> Iterator[None]:
    """Acquire a process-independent exclusive lock without stale lock stealing."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ConflictError("deployment lock is already held: %s" % path) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("%s\n" % os.getpid())
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
        yield
    finally:
        try:
            path.unlink()
            _fsync_directory(path.parent)
        except FileNotFoundError:
            pass


def _restore_pointer(current: Path, previous: Optional[str]) -> None:
    if previous is None:
        try:
            current.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(current.parent)
        return
    temporary = current.parent / (".%s.rollback-%s" % (current.name, uuid.uuid4().hex))
    temporary.symlink_to(previous)
    os.replace(str(temporary), str(current))
    _fsync_directory(current.parent)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(str(source))
    destination_bytes = os.fsencode(str(destination))
    result: int
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise ArtifactError("renameat2 is required for atomic no-replace publication")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise ArtifactError("renamex_np is required for atomic no-replace publication")
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
    else:
        raise ArtifactError("atomic no-replace publication is unsupported on %s" % sys.platform)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise ConflictError("release destination appeared during publication")
    raise OSError(error, os.strerror(error), str(destination))


def _single_component(value: object, label: str) -> str:
    if not isinstance(value, (str, Path)):
        raise ArtifactError("%s must be one canonical path component" % label)
    relative = _relative_path(value, allow_reserved=True)
    if len(PurePosixPath(relative).parts) != 1:
        raise ArtifactError("%s must be one canonical path component" % label)
    return relative


def _require_directory_fd(descriptor: int, label: str) -> None:
    try:
        info = os.fstat(descriptor)
    except OSError as exc:
        raise ArtifactError("%s descriptor is invalid" % label) from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ArtifactError("%s descriptor is not a directory" % label)


@contextmanager
def _open_regular_fd_at(root_descriptor: int, relative: str) -> Iterator[int]:
    """Open a regular file beneath an already pinned directory descriptor."""
    _require_directory_fd(root_descriptor, "artifact root")
    parts = PurePosixPath(_relative_path(relative, allow_reserved=True)).parts
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | nofollow | getattr(os, "O_DIRECTORY", 0)
    descriptors: List[int] = [os.dup(root_descriptor)]
    file_descriptor: Optional[int] = None
    try:
        for part in parts[:-1]:
            info = os.stat(part, dir_fd=descriptors[-1], follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ArtifactError(
                    "artifact source contains a symlink or non-directory: %s"
                    % relative
                )
            descriptors.append(os.open(
                part, directory_flags, dir_fd=descriptors[-1],
            ))
        info = os.stat(parts[-1], dir_fd=descriptors[-1], follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise ArtifactError("artifact source contains a symlink: %s" % relative)
        file_descriptor = os.open(
            parts[-1], os.O_RDONLY | nofollow, dir_fd=descriptors[-1],
        )
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            raise ArtifactError("artifact source is not a regular file: %s" % relative)
        yield file_descriptor
    except FileNotFoundError as exc:
        raise ArtifactError("artifact source file is missing: %s" % relative) from exc
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK, errno.ENOTDIR):
            raise ArtifactError("artifact source contains a symlink: %s" % relative) from exc
        raise
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_regular_bytes_at(root_descriptor: int, relative: str) -> bytes:
    with _open_regular_fd_at(root_descriptor, relative) as descriptor:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise ArtifactError("artifact file changed while reading")
        return b"".join(chunks)


def _open_directory_at(parent_descriptor: int, name: str) -> int:
    component = _single_component(name, "directory name")
    _require_directory_fd(parent_descriptor, "parent")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        info = os.stat(
            component, dir_fd=parent_descriptor, follow_symlinks=False,
        )
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ArtifactError("directory entry is a symlink or non-directory")
        descriptor = os.open(component, flags, dir_fd=parent_descriptor)
    except FileNotFoundError as exc:
        raise ArtifactError("directory entry is missing") from exc
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK, errno.ENOTDIR):
            raise ArtifactError("directory entry is a symlink or non-directory") from exc
        raise
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ArtifactError("directory entry is not a directory")
    return descriptor


def _tree_files_fd(root_descriptor: int, prefix: str = "") -> set:
    found = set()
    for name in os.listdir(root_descriptor):
        info = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        relative = "%s/%s" % (prefix, name) if prefix else name
        if stat.S_ISLNK(info.st_mode):
            raise VerificationError("release contains a symlink: %s" % relative)
        if stat.S_ISDIR(info.st_mode):
            child = _open_directory_at(root_descriptor, name)
            try:
                found.update(_tree_files_fd(child, relative))
            finally:
                os.close(child)
        elif stat.S_ISREG(info.st_mode):
            found.add(relative)
        else:
            raise VerificationError(
                "release contains a non-regular file: %s" % relative
            )
    return found


def _load_envelope_fd(release_descriptor: int) -> tuple:
    try:
        raw = _read_regular_bytes_at(release_descriptor, MANIFEST_NAME)
        envelope = json.loads(raw.decode("utf-8"))
    except (ArtifactError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("release manifest is missing or invalid") from exc
    if not isinstance(envelope, dict) or envelope.get("schema") != ENVELOPE_SCHEMA:
        raise VerificationError("release manifest envelope schema is invalid")
    if raw != canonical_json_bytes(envelope):
        raise VerificationError("release manifest is not canonical JSON")
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or payload.get("schema") != PAYLOAD_SCHEMA:
        raise VerificationError("release payload schema is invalid")
    try:
        _identifier(payload.get("artifact_kind"), "artifact_kind")
        _identifier(payload.get("source_role"), "source_role")
    except ArtifactError as exc:
        raise VerificationError("release payload identifier is invalid") from exc
    payload_sha256 = _sha256_bytes(canonical_json_bytes(payload))
    if envelope.get("payload_sha256") != payload_sha256:
        raise VerificationError("release payload hash mismatch")
    expected_release_id = "%s-%s" % (payload.get("artifact_kind"), payload_sha256[:20])
    if envelope.get("release_id") != expected_release_id:
        raise VerificationError("release id mismatch")
    return envelope, raw


def _load_completion_marker_fd(release_descriptor: int) -> Dict[str, object]:
    try:
        raw = _read_regular_bytes_at(release_descriptor, COMPLETE_NAME)
        marker = json.loads(raw.decode("utf-8"))
    except (ArtifactError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("release completion marker is missing or invalid") from exc
    if not isinstance(marker, dict) or raw != canonical_json_bytes(marker):
        raise VerificationError("release completion marker is not canonical JSON")
    return marker


def verify_release_fd(
    release_descriptor: int, *, require_complete: bool = True,
) -> Dict[str, object]:
    """Verify a release while its directory identity remains pinned by an FD."""
    _require_directory_fd(release_descriptor, "release")
    envelope, manifest_bytes = _load_envelope_fd(release_descriptor)
    payload = envelope["payload"]
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise VerificationError("release file manifest is invalid")
    expected = {MANIFEST_NAME}
    if require_complete:
        expected.add(COMPLETE_NAME)
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise VerificationError("release file row is invalid")
        try:
            relative = _relative_path(row["path"])
        except (ArtifactError, KeyError) as exc:
            raise VerificationError("release file path is invalid") from exc
        if relative in seen:
            raise VerificationError("release manifest contains a duplicate path")
        seen.add(relative)
        expected.add(relative)
        try:
            with _open_regular_fd_at(release_descriptor, relative) as descriptor:
                info = os.fstat(descriptor)
                digest = _sha256_fd(descriptor)
        except (ArtifactError, OSError) as exc:
            raise VerificationError(
                "release payload file is missing: %s" % relative
            ) from exc
        if info.st_size != row.get("size"):
            raise VerificationError("release payload size mismatch: %s" % relative)
        if format(stat.S_IMODE(info.st_mode), "04o") != row.get("mode"):
            raise VerificationError("release payload mode mismatch: %s" % relative)
        if digest != row.get("sha256"):
            raise VerificationError("release payload hash mismatch: %s" % relative)
    found = _tree_files_fd(release_descriptor)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if require_complete and COMPLETE_NAME in missing:
            raise VerificationError("release completion marker is missing")
        raise VerificationError(
            "release exact-set mismatch; missing=%r extra=%r" % (missing, extra)
        )
    if require_complete:
        marker = _load_completion_marker_fd(release_descriptor)
        try:
            _identifier(marker.get("target_role"), "target_role")
        except ArtifactError as exc:
            raise VerificationError("release completion target role is invalid") from exc
        expected_marker = {
            "artifact_kind": payload["artifact_kind"],
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "payload_sha256": envelope["payload_sha256"],
            "release_id": envelope["release_id"],
            "schema": COMPLETE_SCHEMA,
            "source_role": payload["source_role"],
            "target_role": marker.get("target_role"),
            "verified_files": len(rows),
        }
        if not marker.get("target_role") or marker != expected_marker:
            raise VerificationError("release completion marker does not bind the payload")
    return envelope


def verify_release_at(
    parent_descriptor: int,
    release_name: str,
    *,
    require_complete: bool = True,
) -> Dict[str, object]:
    """Open and verify one direct-child release without resolving parent paths."""
    name = _single_component(release_name, "release name")
    try:
        release_descriptor = _open_directory_at(parent_descriptor, name)
    except ArtifactError as exc:
        raise VerificationError("release directory is missing or unsafe") from exc
    try:
        return verify_release_fd(
            release_descriptor, require_complete=require_complete,
        )
    finally:
        os.close(release_descriptor)


def _target_parent_fd(root_descriptor: int, parts: tuple) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for part in parts:
            try:
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            child = _open_directory_at(descriptor, part)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_new_regular_at(
    root_descriptor: int, relative: str, data: bytes, mode: int,
) -> None:
    parts = PurePosixPath(_relative_path(relative, allow_reserved=True)).parts
    parent = _target_parent_fd(root_descriptor, parts[:-1])
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
            dir_fd=parent,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            os.fchmod(handle.fileno(), mode)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _copy_manifest_file_at(
    source_root: Union[str, Path, int],
    target_descriptor: int,
    row: Dict[str, object],
) -> None:
    relative = _relative_path(row["path"])
    opener = (
        _open_regular_fd_at(source_root, relative)
        if isinstance(source_root, int) and not isinstance(source_root, bool)
        else _open_regular_fd(source_root, relative)
    )
    with opener as source_descriptor:
        source_info = os.fstat(source_descriptor)
        digest = _sha256_fd(source_descriptor)
        if source_info.st_size != row.get("size") or digest != row.get("sha256"):
            raise ArtifactError("source changed after manifest construction: %s" % relative)
        if format(stat.S_IMODE(source_info.st_mode), "04o") != row.get("mode"):
            raise ArtifactError("source mode changed after manifest construction: %s" % relative)
        try:
            target_mode = int(str(row["mode"]), 8)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError("manifest file mode is invalid: %s" % relative) from exc
        parts = PurePosixPath(relative).parts
        parent = _target_parent_fd(target_descriptor, parts[:-1])
        target: Optional[int] = None
        try:
            before_copy = _file_identity(os.fstat(source_descriptor))
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            target = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                target_mode,
                dir_fd=parent,
            )
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(target, view)
                    if written <= 0:
                        raise ArtifactError(
                            "short write while copying artifact: %s" % relative
                        )
                    view = view[written:]
            os.fchmod(target, target_mode)
            os.fsync(target)
            if before_copy != _file_identity(os.fstat(source_descriptor)):
                raise ArtifactError("source changed while copying: %s" % relative)
            os.fsync(parent)
        finally:
            if target is not None:
                os.close(target)
            os.close(parent)


def _fsync_tree_fd(root_descriptor: int) -> None:
    for name in os.listdir(root_descriptor):
        info = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise VerificationError("release contains a symlink: %s" % name)
        if stat.S_ISDIR(info.st_mode):
            child = _open_directory_at(root_descriptor, name)
            try:
                _fsync_tree_fd(child)
            finally:
                os.close(child)
    os.fsync(root_descriptor)


def _remove_entry_at(parent_descriptor: int, name: str) -> None:
    component = _single_component(name, "entry name")
    try:
        info = os.stat(component, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        child = _open_directory_at(parent_descriptor, component)
        try:
            for nested in os.listdir(child):
                _remove_entry_at(child, nested)
        finally:
            os.close(child)
        os.rmdir(component, dir_fd=parent_descriptor)
    else:
        os.unlink(component, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


def remove_release_at(parent_descriptor: int, release_name: str) -> None:
    """Remove one direct child through its pinned parent, never following links."""
    _require_directory_fd(parent_descriptor, "release parent")
    _remove_entry_at(parent_descriptor, _single_component(release_name, "release name"))


def _rename_noreplace_at(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
) -> None:
    """Descriptor-relative atomic rename with no-replace semantics."""
    source = _single_component(source_name, "source name")
    destination = _single_component(destination_name, "destination name")
    _require_directory_fd(source_descriptor, "source parent")
    _require_directory_fd(destination_descriptor, "destination parent")
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise ArtifactError("renameat2 is required for atomic no-replace publication")
        operation.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            source_descriptor, os.fsencode(source),
            destination_descriptor, os.fsencode(destination), 1,
        )
    elif sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        if operation is None:
            raise ArtifactError("renameatx_np is required for atomic no-replace publication")
        operation.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            source_descriptor, os.fsencode(source),
            destination_descriptor, os.fsencode(destination), 0x00000004,
        )
    else:
        raise ArtifactError(
            "atomic descriptor-relative publication is unsupported on %s"
            % sys.platform
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise ConflictError("release destination appeared during publication")
    raise OSError(error, os.strerror(error), destination)


def write_release_at(
    source_root: Union[str, Path, int],
    manifest: Dict[str, object],
    destination_parent_descriptor: int,
    destination_name: str,
    *,
    target_role: str,
) -> str:
    """Materialize and atomically publish beneath a pinned parent descriptor."""
    _require_directory_fd(destination_parent_descriptor, "destination parent")
    destination = _single_component(destination_name, "release destination")
    target_role = _identifier(target_role, "target_role")
    try:
        os.stat(
            destination,
            dir_fd=destination_parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    else:
        raise ConflictError("release destination already exists")

    temporary = ".staging-write-%s" % uuid.uuid4().hex
    os.mkdir(temporary, mode=0o700, dir_fd=destination_parent_descriptor)
    os.fsync(destination_parent_descriptor)
    staging_descriptor: Optional[int] = None
    try:
        staging_descriptor = _open_directory_at(
            destination_parent_descriptor, temporary,
        )
        payload = manifest.get("payload") if isinstance(manifest, dict) else None
        if not isinstance(payload, dict):
            raise ArtifactError("manifest payload is invalid")
        expected_hash = _sha256_bytes(canonical_json_bytes(payload))
        if (
            manifest.get("schema") != ENVELOPE_SCHEMA
            or manifest.get("payload_sha256") != expected_hash
        ):
            raise ArtifactError("manifest envelope does not bind its payload")
        expected_id = "%s-%s" % (payload.get("artifact_kind"), expected_hash[:20])
        if manifest.get("release_id") != expected_id:
            raise ArtifactError("manifest release id is invalid")
        rows = payload.get("files")
        if not isinstance(rows, list):
            raise ArtifactError("manifest file list is invalid")
        for row in rows:
            if not isinstance(row, dict):
                raise ArtifactError("manifest file row is invalid")
            _copy_manifest_file_at(source_root, staging_descriptor, row)

        manifest_bytes = canonical_json_bytes(manifest)
        _write_new_regular_at(
            staging_descriptor, MANIFEST_NAME, manifest_bytes, 0o644,
        )
        verify_release_fd(staging_descriptor, require_complete=False)
        marker = {
            "artifact_kind": payload["artifact_kind"],
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "payload_sha256": manifest["payload_sha256"],
            "release_id": manifest["release_id"],
            "schema": COMPLETE_SCHEMA,
            "source_role": payload["source_role"],
            "target_role": target_role,
            "verified_files": len(rows),
        }
        _write_new_regular_at(
            staging_descriptor,
            COMPLETE_NAME,
            canonical_json_bytes(marker),
            0o644,
        )
        verify_release_fd(staging_descriptor)
        _fsync_tree_fd(staging_descriptor)
        _rename_noreplace_at(
            destination_parent_descriptor,
            temporary,
            destination_parent_descriptor,
            destination,
        )
        os.fsync(destination_parent_descriptor)
        return destination
    except BaseException:
        _remove_entry_at(destination_parent_descriptor, temporary)
        raise
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)


def install_release(
    source_release: Union[str, Path],
    releases_root: Union[str, Path],
    current_link: Union[str, Path],
    *,
    target_role: str,
    expected_current: object = _ANY_CURRENT,
    post_activate: Optional[Callable[[Path], None]] = None,
) -> Path:
    """Install a completed release and atomically activate it with rollback."""
    source = Path(source_release)
    source_manifest = verify_release(source)
    target_role = _identifier(target_role, "target_role")
    releases = Path(releases_root)
    current = Path(current_link)
    lock_path = releases.parent / ".deploy.lock"

    with deployment_lock(lock_path):
        releases.mkdir(parents=True, exist_ok=True)
        current.parent.mkdir(parents=True, exist_ok=True)
        final = releases / str(source_manifest["release_id"])
        if final.parent.resolve() != releases.resolve():
            raise ConflictError("release id does not resolve to a direct child")
        if os.path.lexists(str(final)):
            if not final.is_dir() or final.is_symlink():
                raise ConflictError("conflicting existing release path")
            try:
                installed_manifest = verify_release(final)
            except VerificationError as exc:
                raise ConflictError("conflicting existing release is invalid") from exc
            if installed_manifest != source_manifest:
                raise ConflictError("conflicting existing release has different content")
            installed_marker = _load_completion_marker(final)
            if installed_marker.get("target_role") != target_role:
                raise ConflictError("conflicting existing release has the wrong target role")
        else:
            staging = releases / (".staging-%s-%s" % (source_manifest["release_id"], uuid.uuid4().hex))
            try:
                write_release(source, source_manifest, staging, target_role=target_role)
                _rename_noreplace(staging, final)
                _fsync_directory(releases)
            except BaseException:
                shutil.rmtree(str(staging), ignore_errors=True)
                raise

        previous: Optional[str]
        if os.path.lexists(str(current)):
            if not current.is_symlink():
                raise ConflictError("refusing to replace a non-symlink current path")
            previous = os.readlink(str(current))
        else:
            previous = None
        if expected_current is not _ANY_CURRENT and previous != expected_current:
            raise ConflictError(
                "compare-and-swap current mismatch: expected=%r actual=%r"
                % (expected_current, previous)
            )

        temporary = current.parent / (".%s.activate-%s" % (current.name, uuid.uuid4().hex))
        link_target = os.path.relpath(str(final), str(current.parent))
        temporary.symlink_to(link_target)
        try:
            os.replace(str(temporary), str(current))
            _fsync_directory(current.parent)
            verify_release(current.resolve())
            if current.resolve() != final.resolve():
                raise VerificationError("active release pointer resolves to the wrong release")
            if post_activate is not None:
                post_activate(final)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            _restore_pointer(current, previous)
            raise
        return final
