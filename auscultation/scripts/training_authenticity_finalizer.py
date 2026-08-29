"""Fail-closed promotion-v2 training authenticity finalizer.

This module is deliberately an offline verifier, not another capture producer.  It
never turns a producer's ``training_eligible`` flag or small summary receipt into a
PASS.  Every component verifier is replayed from persisted raw artifacts and the
outer :func:`deployment.artifact_contract.verify_training_session_contract` is the
last gate.

The current capture layout is intentionally reported ``NOT_READY``.  In particular,
the lossy ``rgb.mp4`` cannot reproduce the exact uint8 tensor hash carried by a
geometry packet, and PCM16 ``audio.wav`` cannot reproduce the exact float32 window
hash carried by a Q packet.  The future capture contract therefore has to persist the
actual model inputs plus patient-frame/depth-patch evidence.  Missing evidence is a
blocker; it is never approximated from MP4 or inferred from a self-reported hash.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from typing import Any, Callable
import wave

import numpy as np

from ausc.perception import depth_codec


PROMOTION_CONTRACT_VERSION = "wam_promotion_v2"
SESSION_SCHEMA_VERSION = "wam_session_v2"
#: Immutable capture kinds a production writer may stamp into the session
#: manifest and that this finalizer is willing to promote.  A diagnostic capture
#: is never a member.  This allowlist MUST stay in agreement with
#: ``session_to_lerobot_v2.PRODUCTION_CAPTURE_KINDS``.
PRODUCTION_CAPTURE_KINDS = frozenset({"production_teleop_v4"})
COMPONENT_SCHEMA_VERSION = "wam_training_component_authenticity_v2"
AUTHENTICITY_SCHEMA_VERSION = "wam_training_authenticity_v2"
FINALIZATION_SCHEMA_VERSION = "wam_session_finalization_v2"

MANIFEST_NAME = "manifest.json"


def _session_recording_frame_divisor(session: "Path") -> int:
    """Recorded camera indices stride by this divisor (15 fps recording,
    2026-08-12): read from the captured manifest metadata; absent -> 1."""

    import json as _json

    try:
        with open(session / MANIFEST_NAME, "r", encoding="utf-8") as handle:
            manifest = _json.load(handle)
        value = manifest.get("metadata", {}).get("recording_frame_divisor", 1)
    except (OSError, ValueError):
        return 1
    if type(value) is not int or value < 1:
        raise TrainingFinalizationError(
            "recording frame divisor in the session manifest is invalid"
        )
    return value
AUTHENTICITY_NAME = "TRAINING_AUTHENTICITY.json"
FINALIZATION_NAME = "FINALIZATION.json"
COMPONENT_DIR = "authenticity"

# This is intentionally the same minimum set as deployment.artifact_contract.
# Additional lossless evidence below is required by this finalizer and will be
# included in the manifest inventory once the capture producer persists it.
REQUIRED_BASE_RAW_ARTIFACTS = frozenset(
    {
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
    }
)

LOSSLESS_EVIDENCE_ARTIFACTS = {
    "geometry/patient_depth_evidence.jsonl": (
        "geometry_patient_frame_evidence_missing"
    ),
    "geometry/patient_depth_authenticity.json": (
        "geometry_depth_patch_evidence_missing"
    ),
}

_ROBOT_FILES = frozenset(
    {
        "robot_capture_summary.json",
        "robot_control_envelopes.jsonl",
        "robot_exact_control.csv",
        "robot_frame_index.jsonl",
    }
)
_FORCE_FILES = frozenset(
    {
        "FORCE_CAPTURE_COMPLETE.json",
        "force.csv",
        "force_calibration.json",
        "force_capture.json",
    }
)
_AUDIO_FILES = frozenset(
    {
        "audio.wav",
        "audio_capture.json",
        "audio_input_times.jsonl",
        "audio_times.jsonl",
    }
)
_QUALITY_FILES = frozenset(
    {
        "audio.wav",
        "audio_times.jsonl",
        "force.csv",
        "quality_inference_packets.jsonl",
        "quality_inference_authenticity.json",
    }
)
_GEOMETRY_FILES = frozenset(
    {
        "depth/depth_times.csv",
        "geometry_frame_authenticity.json",
        "geometry_frame_packets.jsonl",
        "geometry/patient_depth_evidence.jsonl",
        "geometry/patient_depth_authenticity.json",
        "obs.csv",
        "rgb.mp4",
        "rgb_times.csv",
        "targets.jsonl",
        "tracker.jsonl",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

DEPTH_AUTHENTICITY_POLICY_VERSION = "depth_session_conservative_v2"
_DEPTH_AUTHENTICITY_POLICY = {
    "minimum_valid_fraction_per_frame": 0.05,
    "minimum_frame_count": 30,
    "minimum_duration_s": 1.0,
    "minimum_changed_fraction": 1.0e-4,
    "frozen_min_consecutive_frames": 60,
    "frozen_min_duration_s": 2.0,
    "maximum_device_host_duration_error_fraction": 0.05,
    "maximum_device_host_duration_error_floor_s": 0.05,
}


class TrainingFinalizationError(RuntimeError):
    """A raw artifact or finalization operation violated the v2 contract."""


class TrainingFinalizationNotReady(TrainingFinalizationError):
    """The capture is diagnostic-only and cannot receive a PASS receipt."""

    def __init__(self, report: Mapping[str, object]):
        self.report = json.loads(json.dumps(report, allow_nan=False))
        blockers = self.report.get("blockers") or ["unknown"]
        super().__init__("training finalization is NOT_READY: " + ", ".join(blockers))


@dataclass(frozen=True)
class ActivatedHandEyeBinding:
    """Policy identity reconstructed only from the official runtime loader."""

    artifact_sha256: str
    capability_sha256: str
    release_id: str
    capability_schema: str
    calibration_tier: str
    policy_sha256: str
    calibration_status: str
    provisional: bool
    quality_gate_passed: bool | None
    failed_quality_gate_m: float | None
    authorized_scopes: tuple[str, ...]
    workspace_1sigma_median_m: float | None
    workspace_1sigma_max_m: float | None
    covariance_bound_transform: str
    covariance_scope: str
    systematic_bias_status: str
    systematic_bias_bound_m: float | None
    autonomous_targeting_authorized: bool
    provisional_policy_allowance_m: float

    def __post_init__(self) -> None:
        for value, label in (
            (self.artifact_sha256, "hand-eye activation artifact"),
            (self.capability_sha256, "hand-eye capability"),
            (self.policy_sha256, "hand-eye policy"),
        ):
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise TrainingFinalizationError(
                    f"trusted {label} SHA-256 is invalid"
                )
        if (
            type(self.release_id) is not str
            or not self.release_id
            or type(self.capability_schema) is not str
            or not self.capability_schema
            or type(self.calibration_tier) is not str
            or not self.calibration_tier
        ):
            raise TrainingFinalizationError(
                "trusted hand-eye capability identity is invalid"
            )
        provisional = self.capability_schema == (
            "wam_provisional_robot_mask_handeye_capability_v1"
        )
        if provisional:
            valid = (
                self.release_id
                == "handeye_activation-" + self.artifact_sha256[:20]
                and self.artifact_sha256 == self.capability_sha256
                and self.calibration_tier
                == "USER_ACCEPTED_BELOW_GATE_5MM"
                and self.calibration_status == "USER_ACCEPTED_BELOW_GATE"
                and self.provisional is True
                and self.quality_gate_passed is False
                and self.failed_quality_gate_m == 0.003
                and self.authorized_scopes
                == ("bootstrap_teleop_logging", "conservative_robot_mask")
                and self.workspace_1sigma_median_m is not None
                and self.workspace_1sigma_max_m is not None
                and 0.0 < self.workspace_1sigma_median_m
                <= self.workspace_1sigma_max_m <= 0.0065
                and self.covariance_bound_transform
                == "transform_follower_base_from_camera"
                and self.covariance_scope == "random_only"
                and self.systematic_bias_status == "NOT_CHARACTERIZED"
                and self.systematic_bias_bound_m is None
                and self.autonomous_targeting_authorized is False
                and self.provisional_policy_allowance_m == 0.0065
            )
        else:
            valid = (
                self.capability_schema == "wam_openarm_handeye_v1"
                and self.calibration_tier == "QUALITY_GATE_COMPLETE"
                and self.release_id
                == "handeye-" + self.capability_sha256[:20]
                and self.calibration_status == "COMPLETE"
                and self.provisional is False
                and self.quality_gate_passed is None
                and self.failed_quality_gate_m is None
                and self.authorized_scopes == ()
                and self.workspace_1sigma_median_m is None
                and self.workspace_1sigma_max_m is None
                and self.covariance_bound_transform
                == "transform_camera_from_follower_base"
                and self.covariance_scope == "legacy_production"
                and self.systematic_bias_status == "LEGACY_NOT_APPLICABLE"
                and self.systematic_bias_bound_m is None
                and self.autonomous_targeting_authorized is True
                and self.provisional_policy_allowance_m == 0.0
            )
        if not valid:
            raise TrainingFinalizationError(
                "trusted hand-eye capability policy is invalid"
            )

    def frame_evidence_policy(self) -> dict[str, object]:
        return {
            "handeye_autonomous_targeting_authorized": (
                self.autonomous_targeting_authorized
            ),
            "handeye_calibration_status": self.calibration_status,
            "handeye_calibration_tier": self.calibration_tier,
            "handeye_capability_schema": self.capability_schema,
            "handeye_covariance_bound_transform": (
                self.covariance_bound_transform
            ),
            "handeye_covariance_scope": self.covariance_scope,
            "handeye_policy_sha256": self.policy_sha256,
            "handeye_provisional": self.provisional,
            "handeye_provisional_policy_allowance_m": (
                self.provisional_policy_allowance_m
            ),
            "handeye_quality_gate_passed": self.quality_gate_passed,
            "handeye_systematic_bias_bound_m": self.systematic_bias_bound_m,
            "handeye_systematic_bias_status": self.systematic_bias_status,
            "handeye_workspace_1sigma_max_m": self.workspace_1sigma_max_m,
        }


@dataclass(frozen=True)
class ActivatedCaptureContracts:
    """Read-only identities independently reconstructed from one activation."""

    robot_contract: object
    depth_scale_mm: float
    activation_path: str
    activation_sha256: str
    camera_id: str
    point_names: tuple[str, ...]
    handeye: ActivatedHandEyeBinding


def canonical_json_bytes(value: object) -> bytes:
    """Return the only representation written by the finalizer."""

    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrainingFinalizationError(
            "finalization payload is not canonical JSON data"
        ) from exc


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise TrainingFinalizationError(
                f"raw artifact is not a regular non-symlink file: {path}"
            )
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            opened_before = os.fstat(descriptor)
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = path.lstat()
    except TrainingFinalizationError:
        raise
    except OSError as exc:
        raise TrainingFinalizationError(f"raw artifact cannot be hashed: {path}") from exc
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        getattr(item, "st_mtime_ns", int(item.st_mtime * 1_000_000_000)),
        getattr(item, "st_ctime_ns", int(item.st_ctime * 1_000_000_000)),
    )
    if (
        identity(before) != identity(opened_before)
        or identity(opened_before) != identity(opened_after)
        or identity(opened_after) != identity(after)
    ):
        raise TrainingFinalizationError(f"raw artifact changed while hashing: {path}")
    return digest.hexdigest()


def _stable_regular_bytes(
    path: Path, *, label: str, maximum_bytes: int = 64 * 1024 * 1024
) -> bytes:
    """Read one bounded regular file through a no-follow stable descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = path.lstat()
        if (
            stat.S_ISLNK(before_path.st_mode)
            or not stat.S_ISREG(before_path.st_mode)
            or before_path.st_size <= 0
            or before_path.st_size > maximum_bytes
        ):
            raise TrainingFinalizationError(f"{label} has an invalid file type/size")
        descriptor = os.open(str(path), flags)
    except TrainingFinalizationError:
        raise
    except OSError as exc:
        raise TrainingFinalizationError(f"{label} is missing or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum_bytes:
                raise TrainingFinalizationError(f"{label} exceeds its size bound")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        getattr(item, "st_mtime_ns", int(item.st_mtime * 1_000_000_000)),
        getattr(item, "st_ctime_ns", int(item.st_ctime * 1_000_000_000)),
    )
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise TrainingFinalizationError(f"{label} changed while reading") from exc
    if (
        identity(before_path) != identity(before)
        or identity(before) != identity(after)
        or identity(after) != identity(after_path)
        or total != before.st_size
    ):
        raise TrainingFinalizationError(f"{label} changed while reading")
    return b"".join(chunks)


def module_sha256() -> str:
    """Content identity used in the finalizer activation record."""

    return _sha256_file(Path(__file__))


def _duplicate_safe_object(pairs):  # noqa: ANN001, ANN202
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingFinalizationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TrainingFinalizationError(f"non-finite JSON number is forbidden: {value}")


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = _stable_regular_bytes(path, label=label)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant,
        )
    except TrainingFinalizationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingFinalizationError(f"{label} is missing or invalid JSON") from exc
    if not isinstance(value, dict):
        raise TrainingFinalizationError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, object]]:
    try:
        raw = _stable_regular_bytes(path, label=label, maximum_bytes=512 * 1024 * 1024)
    except TrainingFinalizationError:
        raise
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise TrainingFinalizationError(f"{label} has invalid JSONL framing")
    rows = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        try:
            value = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_duplicate_safe_object,
                parse_constant=_reject_constant,
            )
        except TrainingFinalizationError:
            raise
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise TrainingFinalizationError(
                f"{label} row {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict) or line != canonical_json_bytes(value):
            raise TrainingFinalizationError(
                f"{label} row {line_number} is not canonical JSON"
            )
        rows.append(value)
    if not rows:
        raise TrainingFinalizationError(f"{label} is empty")
    return rows


def _iter_stable_jsonl(
    path: Path,
    *,
    label: str,
    maximum_line_bytes: int = 16 * 1024 * 1024,
) -> Iterator[dict[str, object]]:
    """Stream canonical JSONL through one stable, no-follow descriptor.

    Unlike :func:`_read_jsonl`, this reader deliberately has no whole-file size
    bound.  It is used for long-running frame ledgers, where the safe bound is
    per record rather than per session.
    """

    if maximum_line_bytes <= 0:
        raise TrainingFinalizationError(f"{label} has an invalid line-size bound")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        getattr(item, "st_mtime_ns", int(item.st_mtime * 1_000_000_000)),
        getattr(item, "st_ctime_ns", int(item.st_ctime * 1_000_000_000)),
    )
    try:
        before_path = path.lstat()
        if (
            stat.S_ISLNK(before_path.st_mode)
            or not stat.S_ISREG(before_path.st_mode)
            or before_path.st_size <= 0
        ):
            raise TrainingFinalizationError(f"{label} has an invalid file type/size")
        descriptor = os.open(str(path), flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or identity(before_path) != identity(before)
        ):
            raise TrainingFinalizationError(f"{label} changed while opening")

        buffer = bytearray()
        row_count = 0
        remaining = before.st_size
        while remaining > 0:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise TrainingFinalizationError(f"{label} changed while reading")
            remaining -= len(block)
            if b"\r" in block:
                raise TrainingFinalizationError(f"{label} has invalid JSONL framing")
            buffer.extend(block)
            while True:
                newline = buffer.find(b"\n")
                if newline < 0:
                    if len(buffer) > maximum_line_bytes:
                        raise TrainingFinalizationError(
                            f"{label} row {row_count + 1} exceeds its size bound"
                        )
                    break
                line_length = newline + 1
                if line_length > maximum_line_bytes:
                    raise TrainingFinalizationError(
                        f"{label} row {row_count + 1} exceeds its size bound"
                    )
                line = bytes(buffer[:line_length])
                del buffer[:line_length]
                row_count += 1
                try:
                    value = json.loads(
                        line.decode("utf-8"),
                        object_pairs_hook=_duplicate_safe_object,
                        parse_constant=_reject_constant,
                    )
                except TrainingFinalizationError:
                    raise
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise TrainingFinalizationError(
                        f"{label} row {row_count} is invalid JSON"
                    ) from exc
                if not isinstance(value, dict) or line != canonical_json_bytes(value):
                    raise TrainingFinalizationError(
                        f"{label} row {row_count} is not canonical JSON"
                    )
                yield value

        if os.read(descriptor, 1):
            raise TrainingFinalizationError(f"{label} changed while reading")
        if buffer:
            raise TrainingFinalizationError(f"{label} has invalid JSONL framing")
        if row_count == 0:
            raise TrainingFinalizationError(f"{label} is empty")
        after = os.fstat(descriptor)
        try:
            after_path = path.lstat()
        except OSError as exc:
            raise TrainingFinalizationError(f"{label} changed while reading") from exc
        if (
            identity(before_path) != identity(before)
            or identity(before) != identity(after)
            or identity(after) != identity(after_path)
        ):
            raise TrainingFinalizationError(f"{label} changed while reading")
    except TrainingFinalizationError:
        raise
    except OSError as exc:
        raise TrainingFinalizationError(f"{label} is missing or unsafe") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_canonical_json(path: str | Path, value: object) -> None:
    """Durably publish canonical JSON without ever replacing an existing file."""

    destination = Path(path)
    parent = destination.parent
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise TrainingFinalizationError(
            f"finalization parent directory is unavailable: {parent}"
        ) from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
        parent_metadata.st_mode
    ):
        raise TrainingFinalizationError(
            f"finalization parent must be a real directory: {parent}"
        )
    if destination.exists() or destination.is_symlink():
        raise TrainingFinalizationError(
            f"finalization artifact already exists: {destination.name}"
        )
    raw = canonical_json_bytes(value)
    staging = parent / (
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.finalizing"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(str(staging), flags, 0o640)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive JSON write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(staging, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise TrainingFinalizationError(
                f"finalization artifact already exists: {destination.name}"
            ) from exc
        _fsync_directory(parent)
        staging.unlink()
        _fsync_directory(parent)
    except TrainingFinalizationError:
        if descriptor is not None:
            os.close(descriptor)
        staging.unlink(missing_ok=True)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        staging.unlink(missing_ok=True)
        raise TrainingFinalizationError(
            f"exclusive finalization write failed: {destination.name}"
        ) from exc


def _identity_check(value: object) -> tuple[bool, dict[str, object]]:
    required = {
        "code_manifest_sha256",
        "code_release_id",
        "environment_id",
        "environment_lock_sha256",
        "module_sha256",
    }
    details: dict[str, object] = {"status": "FAIL"}
    if not isinstance(value, Mapping) or set(value) != required:
        details["reason"] = "finalizer identity has an invalid exact schema"
        return False, details
    code_sha = value.get("code_manifest_sha256")
    env_sha = value.get("environment_lock_sha256")
    module_sha = value.get("module_sha256")
    if any(not isinstance(item, str) or _SHA256.fullmatch(item) is None for item in (
        code_sha,
        env_sha,
        module_sha,
    )):
        details["reason"] = "finalizer identity contains an invalid SHA-256"
        return False, details
    code_release = value.get("code_release_id")
    environment = value.get("environment_id")
    if (
        not isinstance(code_release, str)
        or not code_release.endswith("-" + str(code_sha)[:20])
        or not isinstance(environment, str)
        or not environment.endswith("-" + str(env_sha)[:20])
    ):
        details["reason"] = "finalizer release identity is not content-addressed"
        return False, details
    actual_module = module_sha256()
    if module_sha != actual_module:
        details.update(
            reason="activated finalizer module SHA-256 does not match loaded code",
            expected_module_sha256=actual_module,
        )
        return False, details
    details.update(
        status="PASS",
        code_release_id=code_release,
        environment_id=environment,
        module_sha256=module_sha,
    )
    return True, details


def _regular_nonempty(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(
        metadata.st_mode
    ) and metadata.st_size > 0


def _safe_session(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)


def _manifest_check(value: object) -> tuple[list[str], dict[str, object]]:
    blockers: list[str] = []
    details: dict[str, object] = {"status": "FAIL"}
    if not isinstance(value, Mapping):
        return ["manifest_template_invalid"], {
            "status": "FAIL",
            "reason": "manifest template is not an object",
        }
    if value.get("promotion_contract_version") != PROMOTION_CONTRACT_VERSION:
        blockers.append("unsupported_promotion_contract")
    if value.get("schema_version") != SESSION_SCHEMA_VERSION:
        blockers.append("manifest_template_invalid")
    session_id = value.get("session_id")
    if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
        blockers.append("manifest_template_invalid")
    if value.get("rehearsal_overrides") != []:
        blockers.append("rehearsal_override_present")
    if value.get("training_eligible") is not False:
        blockers.append("source_manifest_not_fail_closed")
    if blockers:
        details["reason"] = "manifest template is not a fail-closed v2 capture"
    else:
        details.update(status="PASS", session_id=session_id)
    return blockers, details


def _safe_relative_artifact_reference(
    session: Path,
    value: object,
    *,
    label: str,
    path_field: str = "calibration_path",
    sha_field: str = "calibration_sha256",
) -> tuple[bool, dict[str, object]]:
    """Verify a manifest reference against same-session bytes, never a bare SHA."""

    result: dict[str, object] = {"status": "FAIL"}
    if not isinstance(value, Mapping):
        result["reason"] = f"{label} manifest reference is missing"
        return False, result
    relative = value.get(path_field)
    claimed_sha = value.get(sha_field)
    if (
        not isinstance(relative, str)
        or not relative
        or not isinstance(claimed_sha, str)
        or _SHA256.fullmatch(claimed_sha) is None
    ):
        result["reason"] = f"{label} path/SHA reference is invalid"
        return False, result
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        result["reason"] = f"{label} must be snapshotted inside the session"
        return False, result
    path = session / relative
    if not _regular_nonempty(path):
        result.update(reason=f"{label} artifact is missing", path=relative)
        return False, result
    try:
        actual_sha = _sha256_file(path)
    except TrainingFinalizationError as exc:
        result.update(reason=str(exc), path=relative)
        return False, result
    if actual_sha != claimed_sha:
        result.update(
            reason=f"{label} artifact SHA-256 mismatch",
            path=relative,
            expected_sha256=claimed_sha,
            actual_sha256=actual_sha,
        )
        return False, result
    result.update(status="PASS", path=relative, sha256=actual_sha)
    return True, result


def _canonical_point_names() -> tuple[str, ...]:
    try:
        from ausc.perception.tracking.detector import LUNG_ORDER
    except ImportError as exc:
        raise TrainingFinalizationError(
            "canonical auscultation point order is unavailable"
        ) from exc
    names = tuple(LUNG_ORDER)
    if (
        len(names) != 10
        or len(set(names)) != 10
        or any(type(name) is not str or not name for name in names)
    ):
        raise TrainingFinalizationError(
            "canonical auscultation point order is invalid"
        )
    return names


def _binding_from_runtime_handeye(
    handeye: object,
    *,
    artifact_sha256: str,
) -> ActivatedHandEyeBinding:
    """Freeze hand-eye policy from an official, fully validated capability."""

    try:
        from ausc.perception.robot.robot_mask_provider import (
            ActivatedHandEye,
            ProvisionalRobotMaskHandEye,
        )
    except ImportError as exc:
        raise TrainingFinalizationError(
            "official hand-eye capability classes are unavailable"
        ) from exc
    if isinstance(handeye, ProvisionalRobotMaskHandEye):
        return ActivatedHandEyeBinding(
            artifact_sha256=artifact_sha256,
            capability_sha256=handeye.handeye_sha256,
            release_id=handeye.handeye_release_id,
            capability_schema=handeye.capability_schema,
            calibration_tier=handeye.calibration_tier,
            policy_sha256=handeye.policy_sha256,
            calibration_status=handeye.calibration_status,
            provisional=handeye.provisional,
            quality_gate_passed=handeye.quality_gate_passed,
            failed_quality_gate_m=handeye.failed_quality_gate_m,
            authorized_scopes=tuple(handeye.authorized_scopes),
            workspace_1sigma_median_m=handeye.workspace_1sigma_median,
            workspace_1sigma_max_m=handeye.workspace_1sigma_max,
            covariance_bound_transform=handeye.covariance_bound_transform,
            covariance_scope=handeye.covariance_scope,
            systematic_bias_status=handeye.systematic_bias_status,
            systematic_bias_bound_m=handeye.systematic_bias_bound_m,
            autonomous_targeting_authorized=(
                handeye.autonomous_targeting_authorized
            ),
            provisional_policy_allowance_m=(
                handeye.provisional_policy_allowance_m
            ),
        )
    if isinstance(handeye, ActivatedHandEye):
        if handeye.status != "COMPLETE":
            raise TrainingFinalizationError(
                "official legacy hand-eye capability is not COMPLETE"
            )
        return ActivatedHandEyeBinding(
            artifact_sha256=artifact_sha256,
            capability_sha256=handeye.handeye_sha256,
            release_id=handeye.handeye_release_id,
            capability_schema=handeye.capability_schema,
            calibration_tier=handeye.calibration_tier,
            policy_sha256=handeye.policy_sha256,
            calibration_status=handeye.status,
            provisional=False,
            quality_gate_passed=None,
            failed_quality_gate_m=None,
            authorized_scopes=(),
            workspace_1sigma_median_m=None,
            workspace_1sigma_max_m=None,
            covariance_bound_transform="transform_camera_from_follower_base",
            covariance_scope="legacy_production",
            systematic_bias_status="LEGACY_NOT_APPLICABLE",
            systematic_bias_bound_m=None,
            autonomous_targeting_authorized=True,
            provisional_policy_allowance_m=0.0,
        )
    raise TrainingFinalizationError(
        "hardware activation did not yield an official hand-eye capability"
    )


def _activation_handeye_artifact_sha256(
    activation_path: Path,
    *,
    expected_activation_sha256: str,
) -> str:
    raw = _stable_regular_bytes(
        activation_path,
        label="production runtime activation",
    )
    if hashlib.sha256(raw).hexdigest() != expected_activation_sha256:
        raise TrainingFinalizationError(
            "production runtime activation SHA-256 changed before replay"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant,
        )
    except TrainingFinalizationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingFinalizationError(
            "production runtime activation is invalid JSON"
        ) from exc
    artifacts = payload.get("artifacts") if type(payload) is dict else None
    handeye = artifacts.get("handeye") if type(artifacts) is dict else None
    if (
        type(handeye) is not dict
        or set(handeye) != {"complete_path", "complete_sha256", "path", "sha256"}
        or type(handeye.get("sha256")) is not str
        or _SHA256.fullmatch(handeye["sha256"]) is None
    ):
        raise TrainingFinalizationError(
            "production runtime activation has no exact hand-eye artifact pin"
        )
    return handeye["sha256"]


def load_activated_capture_contracts(
    session_dir: str | Path,
    manifest: Mapping[str, object],
) -> ActivatedCaptureContracts:
    """Load the same-session production activation without starting hardware.

    The session manifest supplies only routing identity.  Robot bounds, robot
    release identity, and the depth scale are reconstructed by the official
    production runtime activation loader; none are accepted from capture
    summaries.  The returned assembly is always closed before this function
    returns, and receiver/CAN preparation is never invoked.
    """

    session = Path(session_dir)
    if not _safe_session(session):
        raise TrainingFinalizationError(
            "production runtime activation requires a real session directory"
        )
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != SESSION_SCHEMA_VERSION
        or manifest.get("promotion_contract_version")
        != PROMOTION_CONTRACT_VERSION
    ):
        raise TrainingFinalizationError(
            "production runtime activation requires the exact v2 manifest schema"
        )

    activation = manifest.get("hardware_activation")
    if type(activation) is not dict or set(activation) != {"path", "sha256"}:
        raise TrainingFinalizationError(
            "hardware activation must contain exact path/SHA fields"
        )
    reference_ok, reference = _safe_relative_artifact_reference(
        session,
        activation,
        label="hardware activation",
        path_field="path",
        sha_field="sha256",
    )
    if not reference_ok:
        raise TrainingFinalizationError(str(reference.get("reason")))

    camera = manifest.get("camera")
    timing = camera.get("timing") if isinstance(camera, Mapping) else None
    camera_id = camera.get("camera_id") if isinstance(camera, Mapping) else None
    capture_mode = (
        timing.get("capture_mode") if isinstance(timing, Mapping) else None
    )
    if (
        type(camera_id) is not str
        or not camera_id
        or camera_id.strip() != camera_id
        or type(capture_mode) is not dict
        or not capture_mode
    ):
        raise TrainingFinalizationError(
            "manifest camera identity/capture_mode is missing or invalid"
        )
    try:
        frozen_capture_mode = json.loads(
            canonical_json_bytes(capture_mode).decode("ascii")
        )
    except (TrainingFinalizationError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingFinalizationError(
            "manifest camera identity/capture_mode is not canonical JSON"
        ) from exc
    if type(frozen_capture_mode) is not dict:
        raise TrainingFinalizationError(
            "manifest camera identity/capture_mode is invalid"
        )

    expected_point_names = _canonical_point_names()
    point_names = manifest.get("point_names")
    if type(point_names) is not list or tuple(point_names) != expected_point_names:
        raise TrainingFinalizationError(
            "manifest point_names do not match the canonical 10-point order"
        )

    relative = activation["path"]
    claimed_sha = activation["sha256"]
    # ``_safe_relative_artifact_reference`` established the canonical relative
    # path.  Resolve only the session root: the official loader will reject any
    # symlink in the activation path or its remaining parent components.
    session_root = session.resolve(strict=True)
    activation_path = session_root / PurePosixPath(relative)
    handeye_artifact_sha256 = _activation_handeye_artifact_sha256(
        activation_path,
        expected_activation_sha256=claimed_sha,
    )
    assembly = None
    close_error: Exception | None = None
    result: ActivatedCaptureContracts | None = None
    try:
        from ausc.perception.production_runtime_activation import (
            load_production_runtime_assembly,
        )
        from ausc.perception.robot.robot_state_stream import RobotStateContract

        assembly = load_production_runtime_assembly(
            activation_path,
            expected_activation_sha256=claimed_sha,
            point_names=expected_point_names,
            expected_camera_id=camera_id,
            expected_capture_mode=frozen_capture_mode,
        )
        robot_contract = getattr(
            getattr(assembly, "robot_capture_coordinator", None),
            "contract",
            None,
        )
        if not isinstance(robot_contract, RobotStateContract):
            raise TrainingFinalizationError(
                "production activation did not yield a RobotStateContract"
            )
        depth_scale = getattr(
            getattr(
                getattr(assembly, "patient_pose_provider", None),
                "intrinsics",
                None,
            ),
            "depth_scale",
            None,
        )
        if (
            isinstance(depth_scale, bool)
            or not isinstance(depth_scale, (int, float))
            or not math.isfinite(float(depth_scale))
            or float(depth_scale) <= 0.0
        ):
            raise TrainingFinalizationError(
                "production activation did not yield a valid camera depth scale"
            )
        activated_camera_id = getattr(
            getattr(assembly, "camera_exposure_contract", None),
            "camera_id",
            None,
        )
        activated_point_names = tuple(
            getattr(
                getattr(assembly, "occlusion_runtime", None),
                "point_names",
                (),
            )
        )
        if (
            activated_camera_id != camera_id
            or activated_point_names != expected_point_names
        ):
            raise TrainingFinalizationError(
                "production activation camera/point identity does not match manifest"
            )
        handeye = getattr(
            getattr(assembly, "robot_mask_provider", None),
            "handeye",
            None,
        )
        handeye_binding = _binding_from_runtime_handeye(
            handeye,
            artifact_sha256=handeye_artifact_sha256,
        )
        result = ActivatedCaptureContracts(
            robot_contract=robot_contract,
            depth_scale_mm=float(depth_scale),
            activation_path=str(activation_path),
            activation_sha256=claimed_sha,
            camera_id=camera_id,
            point_names=expected_point_names,
            handeye=handeye_binding,
        )
    except TrainingFinalizationError:
        raise
    except Exception as exc:
        raise TrainingFinalizationError(
            f"production runtime activation replay failed: {exc}"
        ) from exc
    finally:
        if assembly is not None:
            try:
                assembly.close()
            except Exception as exc:  # pragma: no cover - defensive hardware boundary
                close_error = exc
        if close_error is not None:
            raise TrainingFinalizationError(
                f"production runtime assembly could not close safely: {close_error}"
            ) from close_error
    if result is None:  # pragma: no cover - all construction failures raise above
        raise TrainingFinalizationError(
            "production runtime activation replay produced no contracts"
        )
    return result


def verify_recorded_handeye_frame_evidence(
    session_dir: str | Path,
    handeye: ActivatedHandEyeBinding,
) -> dict[str, object]:
    """Revalidate the policy identity recorded on every captured RGB frame."""

    session = Path(session_dir)
    if not _safe_session(session) or not isinstance(
        handeye, ActivatedHandEyeBinding
    ):
        raise TrainingFinalizationError(
            "tracker hand-eye evidence requires a trusted activation binding"
        )
    try:
        rgb_raw = _stable_regular_bytes(
            session / "rgb_times.csv",
            label="RGB timestamp timeline",
            maximum_bytes=512 * 1024 * 1024,
        )
        rgb_text = rgb_raw.decode("utf-8")
    except UnicodeError as exc:
        raise TrainingFinalizationError(
            "tracker hand-eye evidence RGB timeline is not UTF-8"
        ) from exc
    if not rgb_text.endswith("\n") or "\r" in rgb_text:
        raise TrainingFinalizationError(
            "tracker hand-eye evidence RGB timeline framing is invalid"
        )
    reader = csv.DictReader(io.StringIO(rgb_text, newline=""))
    if reader.fieldnames != [
        "frame_index",
        "host_arrival_t_s",
        "device_t_ms",
        "time_source",
    ]:
        raise TrainingFinalizationError(
            "tracker hand-eye evidence RGB timeline schema is invalid"
        )
    rgb_indexes: list[int] = []
    try:
        for row in reader:
            if None in row or row.get("frame_index") is None:
                raise ValueError("invalid RGB row")
            rgb_indexes.append(int(row["frame_index"]))
    except (TypeError, ValueError) as exc:
        raise TrainingFinalizationError(
            "tracker hand-eye evidence RGB frame index is invalid"
        ) from exc
    stride = _session_recording_frame_divisor(session)
    if (
        not rgb_indexes
        or len(set(rgb_indexes)) != len(rgb_indexes)
        or any(
            current != previous + stride
            for previous, current in zip(rgb_indexes, rgb_indexes[1:])
        )
    ):
        raise TrainingFinalizationError(
            "tracker hand-eye evidence RGB frame coverage is invalid"
        )

    rows = _iter_stable_jsonl(
        session / "tracker.jsonl",
        label="tracker hand-eye evidence",
    )
    policy = handeye.frame_evidence_policy()
    row_count = 0
    try:
        for row_number, row in enumerate(rows, start=1):
            row_count = row_number
            if row_number > len(rgb_indexes):
                raise TrainingFinalizationError(
                    "tracker hand-eye evidence does not cover every RGB frame"
                )
            expected_index = rgb_indexes[row_number - 1]
            if (
                type(row.get("frame_index")) is not int
                or row["frame_index"] != expected_index
            ):
                raise TrainingFinalizationError(
                    f"tracker hand-eye evidence row {row_number} frame binding is invalid"
                )
            ledger = row.get("frame_evidence")
            if type(ledger) is not dict or set(ledger) != {
                "bundle_id",
                "canonical_json",
                "canonical_sha256",
                "schema",
            }:
                raise TrainingFinalizationError(
                    f"tracker hand-eye evidence row {row_number} ledger is missing"
                )
            canonical_json = ledger.get("canonical_json")
            canonical_sha256 = ledger.get("canonical_sha256")
            ledger_bundle_id = ledger.get("bundle_id")
            if (
                ledger.get("schema") != "wam_frame_evidence_ledger_record_v1"
                or type(canonical_json) is not str
                or type(canonical_sha256) is not str
                or _SHA256.fullmatch(canonical_sha256) is None
                or type(ledger_bundle_id) is not str
                or ledger_bundle_id != "frame_evidence-" + canonical_sha256
                or row.get("frame_evidence_bundle_id") != ledger_bundle_id
            ):
                raise TrainingFinalizationError(
                    f"tracker hand-eye evidence row {row_number} ledger identity is invalid"
                )
            try:
                canonical_raw = canonical_json.encode("ascii")
                metadata = json.loads(
                    canonical_json,
                    object_pairs_hook=_duplicate_safe_object,
                    parse_constant=_reject_constant,
                )
            except TrainingFinalizationError:
                raise
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise TrainingFinalizationError(
                    f"tracker hand-eye evidence row {row_number} ledger JSON is invalid"
                ) from exc
            if (
                hashlib.sha256(canonical_raw).hexdigest() != canonical_sha256
                or type(metadata) is not dict
                or json.dumps(
                    metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii") != canonical_raw
                or metadata.get("schema") != "wam_frame_evidence_bundle_v3"
            ):
                raise TrainingFinalizationError(
                    f"tracker hand-eye evidence row {row_number} ledger hash/schema is invalid"
                )
            camera_frame = metadata.get("camera_frame")
            camera_exposure = metadata.get("camera_exposure")
            robot_mask = metadata.get("robot_mask")
            if (
                type(camera_frame) is not dict
                or camera_frame.get("index") != expected_index
                or type(camera_exposure) is not dict
                or camera_exposure.get("frame_index") != expected_index
                or type(robot_mask) is not dict
            ):
                raise TrainingFinalizationError(
                    f"tracker hand-eye evidence row {row_number} ledger frame is invalid"
                )
            required = {
                **policy,
                "frame_index": expected_index,
                "handeye_release_id": handeye.release_id,
                "handeye_sha256": handeye.capability_sha256,
            }
            for field, expected in required.items():
                observed = robot_mask.get(field, object())
                canonical_expected = (
                    {"float64_hex": expected.hex()}
                    if type(expected) is float else expected
                )
                if (
                    type(observed) is not type(canonical_expected)
                    or observed != canonical_expected
                ):
                    raise TrainingFinalizationError(
                        "tracker hand-eye evidence row "
                        f"{row_number} mismatches {field}"
                    )
            mask_sha256 = robot_mask.get("mask_sha256")
            if type(mask_sha256) is not str or _SHA256.fullmatch(mask_sha256) is None:
                raise TrainingFinalizationError(
                    f"tracker hand-eye evidence row {row_number} mask SHA-256 is invalid"
                )
            derived = row.get("robot_mask")
            if type(derived) is not dict:
                raise TrainingFinalizationError(
                    "tracker hand-eye evidence derived robot-mask metadata is missing"
                )
            for field, expected in {**required, "mask_sha256": mask_sha256}.items():
                observed = derived.get(field, object())
                if type(observed) is not type(expected) or observed != expected:
                    raise TrainingFinalizationError(
                        "tracker hand-eye evidence derived robot-mask metadata "
                        f"row {row_number} mismatches {field}"
                    )
    finally:
        rows.close()
    if row_count != len(rgb_indexes):
        raise TrainingFinalizationError(
            "tracker hand-eye evidence does not cover every RGB frame"
        )
    return {
        "status": "PASS",
        "frame_count": row_count,
        "first_frame_index": rgb_indexes[0],
        "last_frame_index": rgb_indexes[-1],
        "release_id": handeye.release_id,
        "capability_sha256": handeye.capability_sha256,
        "policy_sha256": handeye.policy_sha256,
    }


def _production_artifact_binding_checks(
    session: Path,
    manifest: Mapping[str, object],
    *,
    activated_contracts: ActivatedCaptureContracts | None = None,
) -> tuple[list[str], dict[str, object]]:
    blockers: list[str] = []
    details: dict[str, object] = {}

    sync_ok, details["physical_audio_force_sync"] = (
        _safe_relative_artifact_reference(
            session,
            manifest.get("sync"),
            label="physical audio-force sync calibration",
        )
    )
    if not sync_ok:
        blockers.append("physical_audio_force_sync_artifact_missing")

    camera = manifest.get("camera")
    timing = camera.get("timing") if isinstance(camera, Mapping) else None
    camera_ok, details["camera_led_latency"] = (
        _safe_relative_artifact_reference(
            session,
            timing,
            label="camera LED latency calibration",
        )
    )
    if (
        camera_ok
        and isinstance(timing, Mapping)
        # hardware timing: LED flash or host-synced device timestamp (device_timer_host_sync)
        and timing.get("measurement_method")
        not in ("hardware_led_flash", "device_timer_host_sync")
    ):
        camera_ok = False
        details["camera_led_latency"] = {
            "status": "FAIL",
            "reason": "camera latency was not commissioned with a hardware timing method",
        }
    if not camera_ok:
        blockers.append("camera_led_latency_artifact_missing")

    handeye = manifest.get("handeye")
    handeye_ok, details["handeye"] = _safe_relative_artifact_reference(
        session,
        handeye,
        label="hand-eye calibration",
    )
    if handeye_ok and isinstance(handeye, Mapping):
        expected_keys = {
            "calibration_path",
            "calibration_sha256",
            "capability_sha256",
            "policy_sha256",
            "release_id",
        }
        if set(handeye) != expected_keys:
            handeye_ok = False
            details["handeye"] = {
                "status": "FAIL",
                "reason": "hand-eye manifest binding has an invalid exact schema",
            }
        elif activated_contracts is None:
            handeye_ok = False
            details["handeye"] = {
                "status": "FAIL",
                "reason": (
                    "hand-eye is not bound to the trusted hardware activation"
                ),
            }
        else:
            trusted = activated_contracts.handeye
            exact_identity = {
                "calibration_sha256": trusted.artifact_sha256,
                "capability_sha256": trusted.capability_sha256,
                "policy_sha256": trusted.policy_sha256,
                "release_id": trusted.release_id,
            }
            if any(handeye.get(key) != value for key, value in exact_identity.items()):
                handeye_ok = False
                details["handeye"] = {
                    "status": "FAIL",
                    "reason": (
                        "hand-eye manifest identity/policy does not exactly match "
                        "the trusted hardware activation"
                    ),
                }
            else:
                details["handeye"].update({
                    "status": (
                        "PASS_PROVISIONAL" if trusted.provisional else "PASS"
                    ),
                    "artifact_sha256": trusted.artifact_sha256,
                    "capability_sha256": trusted.capability_sha256,
                    "release_id": trusted.release_id,
                    "capability_schema": trusted.capability_schema,
                    "calibration_tier": trusted.calibration_tier,
                    "policy_sha256": trusted.policy_sha256,
                    "calibration_status": trusted.calibration_status,
                    "authorized_scope": (
                        "bootstrap_teleop_logging"
                        if trusted.provisional else "production"
                    ),
                    "autonomous_targeting_authorized": (
                        trusted.autonomous_targeting_authorized
                    ),
                    "workspace_1sigma_median_m": (
                        trusted.workspace_1sigma_median_m
                    ),
                    "workspace_1sigma_max_m": (
                        trusted.workspace_1sigma_max_m
                    ),
                    "systematic_bias_status": trusted.systematic_bias_status,
                })
                try:
                    details["handeye_frame_evidence"] = (
                        verify_recorded_handeye_frame_evidence(
                            session,
                            trusted,
                        )
                    )
                except TrainingFinalizationError as exc:
                    blockers.append("handeye_frame_evidence_invalid")
                    details["handeye_frame_evidence"] = {
                        "status": "FAIL",
                        "reason": str(exc),
                    }
    if not handeye_ok:
        blockers.append("handeye_robot_base_artifact_missing")

    hardware_ok, details["hardware_activation"] = (
        _safe_relative_artifact_reference(
            session,
            manifest.get("hardware_activation"),
            label="hardware activation",
            path_field="path",
            sha_field="sha256",
        )
    )
    if not hardware_ok:
        blockers.append("hardware_activation_artifact_missing")

    model_activations = manifest.get("model_activations")
    model_details: dict[str, object] = {}
    model_ok = isinstance(model_activations, Mapping) and set(model_activations) == {
        "independent_geometry",
        "perception_geometry",
        "quality",
    }
    if model_ok:
        for name in (
            "independent_geometry",
            "perception_geometry",
            "quality",
        ):
            activation = model_activations[name]
            ok, item = _safe_relative_artifact_reference(
                session,
                activation,
                label=f"{name} model activation",
                path_field="path",
                sha_field="sha256",
            )
            if ok and isinstance(activation, Mapping):
                release_id = activation.get("release_id")
                digest = activation.get("model_sha256")
                if (
                    not isinstance(digest, str)
                    or _SHA256.fullmatch(digest) is None
                    or not isinstance(release_id, str)
                    or not release_id.endswith("-" + digest[:20])
                ):
                    ok = False
                    item = {
                        "status": "FAIL",
                        "reason": f"{name} model release is not content-addressed",
                    }
            model_ok = model_ok and ok
            model_details[name] = item
    details["model_activations"] = model_details
    if not model_ok:
        blockers.append("model_activation_artifacts_missing")

    details["status"] = "PASS" if not blockers else "FAIL"
    return blockers, details


def inspect_training_finalization_readiness(
    session_dir: str | Path,
    *,
    manifest_template: Mapping[str, object],
    finalizer_identity: Mapping[str, object],
) -> dict[str, object]:
    """Return structured blockers without writing any promotion metadata."""

    session = Path(session_dir)
    blockers: list[str] = []
    checks: dict[str, object] = {}
    activated_contracts: ActivatedCaptureContracts | None = None

    if not _safe_session(session):
        blockers.append("session_directory_invalid")
        checks["session"] = {
            "status": "FAIL",
            "reason": "session must be a real non-symlink directory",
        }
    else:
        checks["session"] = {"status": "PASS", "path": str(session.resolve())}

    manifest_blockers, manifest_details = _manifest_check(manifest_template)
    blockers.extend(manifest_blockers)
    checks["manifest_template"] = manifest_details

    # Immutable production-writer gate.  A diagnostic capture is *also*
    # fail-closed (``training_eligible`` False), so that flag alone cannot
    # distinguish a production writer from a diagnostic one.  Promotion requires
    # the manifest -- which the source-manifest check below binds by SHA to the
    # exact on-disk bytes -- to positively declare an *allowlisted* production
    # ``capture_kind``.  An allowlist (not a mere ``production_*`` prefix) means
    # a forged arbitrary ``production_anything`` string cannot pass either.  It
    # is verified here, never inferred from the absence of a diagnostic flag or
    # trusted from a producer's summary receipt.
    declared_capture_kind = (
        manifest_template.get("capture_kind")
        if isinstance(manifest_template, Mapping)
        else None
    )
    if declared_capture_kind in PRODUCTION_CAPTURE_KINDS:
        checks["production_capture_kind"] = {
            "status": "PASS",
            "capture_kind": declared_capture_kind,
        }
    else:
        blockers.append("capture_kind_not_production")
        checks["production_capture_kind"] = {
            "status": "FAIL",
            "capture_kind": declared_capture_kind,
            "reason": (
                "promotion requires an allowlisted production capture_kind; a "
                "diagnostic or unrecognized capture kind can never be upgraded"
            ),
        }

    identity_ok, identity_details = _identity_check(finalizer_identity)
    if not identity_ok:
        blockers.append("finalizer_identity_invalid")
    checks["finalizer_identity"] = identity_details

    if _safe_session(session):
        try:
            expected_point_names = list(_canonical_point_names())
        except TrainingFinalizationError:
            expected_point_names = []
        raw_point_names = manifest_template.get("point_names")
        observed_point_names = (
            list(raw_point_names) if type(raw_point_names) is list else None
        )
        try:
            activated_contracts = load_activated_capture_contracts(
                session, manifest_template
            )
        except TrainingFinalizationError as exc:
            blockers.append(
                "production_runtime_activation_revalidation_failed"
            )
            checks["production_runtime_activation"] = {
                "status": "FAIL",
                "reason": str(exc),
                "observed_point_names": observed_point_names,
                "expected_point_names": expected_point_names,
            }
        else:
            checks["production_runtime_activation"] = {
                "status": "PASS",
                "activation_path": activated_contracts.activation_path,
                "activation_sha256": activated_contracts.activation_sha256,
                "camera_id": activated_contracts.camera_id,
                "depth_scale_mm": activated_contracts.depth_scale_mm,
                "observed_point_names": list(
                    activated_contracts.point_names
                ),
                "expected_point_names": expected_point_names,
                "receiver_prepared": False,
                "can_opened": False,
                "assembly_closed": True,
            }

        source_manifest_path = session / MANIFEST_NAME
        if source_manifest_path.exists() or source_manifest_path.is_symlink():
            try:
                source_manifest = _read_json_object(
                    source_manifest_path, label="source capture manifest"
                )
                source_sha = _sha256_file(source_manifest_path)
            except TrainingFinalizationError as exc:
                blockers.append("source_manifest_mismatch")
                checks["source_manifest"] = {
                    "status": "FAIL",
                    "reason": str(exc),
                }
            else:
                if source_manifest != dict(manifest_template):
                    blockers.append("source_manifest_mismatch")
                    checks["source_manifest"] = {
                        "status": "FAIL",
                        "reason": "source manifest differs from supplied capture template",
                        "sha256": source_sha,
                    }
                else:
                    checks["source_manifest"] = {
                        "status": "PASS",
                        "sha256": source_sha,
                        "transaction": (
                            "preserve_exact_original_then_atomic_controlled_replace"
                        ),
                    }
                    blockers.append(
                        "source_manifest_controlled_replacement_not_activated"
                    )
        else:
            # No on-disk source manifest means NOTHING binds the supplied
            # manifest_template to real captured bytes by SHA -- the production
            # capture_kind / schema gates above would then be decided purely from
            # caller-supplied input. A trusted promotion REQUIRES a SHA-bound on-disk
            # source manifest (every real captured session writes manifest.json at
            # finalize), so its absence is a hard blocker, never a PASS.
            blockers.append("source_manifest_absent")
            checks["source_manifest"] = {
                "status": "FAIL",
                "reason": (
                    "no on-disk source manifest to bind the capture template by SHA; "
                    "promotion requires the original captured manifest.json"
                ),
            }

        committed = [
            name
            for name in (AUTHENTICITY_NAME, FINALIZATION_NAME, COMPONENT_DIR)
            if (session / name).exists() or (session / name).is_symlink()
        ]
        if committed:
            blockers.append("finalization_metadata_already_present")
            checks["exclusive_output"] = {
                "status": "FAIL",
                "present": committed,
            }
        else:
            checks["exclusive_output"] = {"status": "PASS"}

        missing = sorted(
            relative
            for relative in REQUIRED_BASE_RAW_ARTIFACTS
            if not _regular_nonempty(session / relative)
        )
        depth_frames = (sorted((session / "depth").glob(f"depth_*{depth_codec.DEPTH_SUFFIX}"))
                        or sorted((session / "depth").glob("depth_*.npy")))
        valid_depth_paths = [
            path for path in depth_frames if _regular_nonempty(path)
        ]
        if missing or not valid_depth_paths:
            blockers.append("required_raw_artifacts_missing")
        checks["base_raw_artifacts"] = {
            "status": "PASS" if not missing and valid_depth_paths else "FAIL",
            "missing": missing + ([] if valid_depth_paths else ["depth/depth_*.npy"]),
            "depth_frame_count": len(valid_depth_paths),
        }

        if (
            not _regular_nonempty(session / "depth" / "depth_times.csv")
            and _regular_nonempty(session / "depth_times.csv")
        ):
            blockers.append("depth_timing_path_contract_mismatch")
            checks["depth_timing_layout"] = {
                "status": "FAIL",
                "observed": "depth_times.csv",
                "required": "depth/depth_times.csv",
            }
        else:
            checks["depth_timing_layout"] = {
                "status": (
                    "PASS"
                    if _regular_nonempty(session / "depth" / "depth_times.csv")
                    else "FAIL"
                ),
                "required": "depth/depth_times.csv",
            }

        evidence_checks: dict[str, object] = {}
        model_input_paths = (
            sorted((session / "geometry" / "model_inputs").glob("*.npy"))
            if (session / "geometry" / "model_inputs").is_dir()
            else []
        )
        model_input_paths = [
            path for path in model_input_paths if _regular_nonempty(path)
        ]
        if not model_input_paths:
            blockers.append("geometry_lossless_model_inputs_missing")
        evidence_checks["geometry/model_inputs/*.npy"] = {
            "status": "PASS" if model_input_paths else "FAIL",
            "count": len(model_input_paths),
            "reason": (
                "lossless model-consumed uint8 tensors are present"
                if model_input_paths
                else "lossy rgb.mp4 cannot reproduce geometry input hashes"
            ),
        }
        for relative, blocker in LOSSLESS_EVIDENCE_ARTIFACTS.items():
            present = _regular_nonempty(session / relative)
            if not present:
                blockers.append(blocker)
            evidence_checks[relative] = {
                "status": "PASS" if present else "FAIL",
                "reason": (
                    "persisted raw model input/evidence is present"
                    if present
                    else "receipt hash alone cannot be independently revalidated"
                ),
            }
        if model_input_paths and all(
            _regular_nonempty(session / relative)
            for relative in LOSSLESS_EVIDENCE_ARTIFACTS
        ):
            try:
                geometry_evidence = verify_geometry_packet_component(session)
            except TrainingFinalizationError as exc:
                blockers.append("geometry_raw_revalidation_failed")
                evidence_checks["geometry_exact_evidence_verifier"] = {
                    "status": "FAIL",
                    "reason": str(exc),
                }
            else:
                # A self-consistent numeric packet chain is not production truth.
                # Only a valid replay attestation (the activated model re-run on
                # the persisted inputs) flips ``production_authorized``; until
                # then this is a blocker, never a PASS.
                if geometry_evidence.get("production_authorized") is True:
                    evidence_checks["geometry_exact_evidence_verifier"] = {
                        "status": "PASS",
                        "evidence": geometry_evidence,
                    }
                else:
                    blockers.append("geometry_component_not_production_authorized")
                    evidence_checks["geometry_exact_evidence_verifier"] = {
                        "status": "FAIL",
                        "evidence": geometry_evidence,
                        "reason": (
                            geometry_evidence.get("blocker")
                            or "geometry component is diagnostic-only "
                            "(production_authorized is False)"
                        ),
                    }
        quality_input_dir = session / "quality_model_inputs"
        quality_inputs_present = False
        try:
            quality_inputs_present = (
                quality_input_dir.is_dir()
                and not quality_input_dir.is_symlink()
                and any(
                    _regular_nonempty(path)
                    for path in quality_input_dir.iterdir()
                )
            )
        except OSError:
            quality_inputs_present = False
        if not quality_inputs_present:
            blockers.append("quality_lossless_model_inputs_missing")
            evidence_checks["quality_exact_input_verifier"] = {
                "status": "FAIL",
                "reason": "quality_model_inputs/ is missing or empty",
            }
        elif all(
            _regular_nonempty(session / relative)
            for relative in (
                "force.csv",
                "quality_inference_authenticity.json",
                "quality_inference_packets.jsonl",
            )
        ):
            try:
                quality_evidence = verify_quality_packet_component(session)
            except TrainingFinalizationError as exc:
                blockers.append("quality_raw_revalidation_failed")
                evidence_checks["quality_exact_input_verifier"] = {
                    "status": "FAIL",
                    "reason": str(exc),
                }
            else:
                # A self-consistent packet chain is *not* production truth.
                # verify_quality_packet_component() reports
                # ``production_authorized=False`` until the activated quality
                # weights are re-run against lossless audio windows; that
                # verdict must form a blocker here, never be recorded as PASS.
                if quality_evidence.get("production_authorized") is True:
                    evidence_checks["quality_exact_input_verifier"] = {
                        "status": "PASS",
                        "evidence": quality_evidence,
                    }
                else:
                    blockers.append("quality_component_not_production_authorized")
                    evidence_checks["quality_exact_input_verifier"] = {
                        "status": "FAIL",
                        "evidence": quality_evidence,
                        "reason": (
                            quality_evidence.get("blocker")
                            or "quality component is diagnostic-only "
                            "(production_authorized is False)"
                        ),
                    }
        else:
            blockers.append("quality_raw_revalidation_failed")
            evidence_checks["quality_exact_input_verifier"] = {
                "status": "FAIL",
                "reason": "quality packet log/receipt or force.csv is missing",
            }
        checks["lossless_model_evidence"] = evidence_checks

        component_checks: dict[str, object] = {}
        if all(_regular_nonempty(session / relative) for relative in _AUDIO_FILES):
            try:
                component_checks["audio"] = verify_audio_component(session)
            except TrainingFinalizationError as exc:
                blockers.append("audio_raw_revalidation_failed")
                component_checks["audio"] = {
                    "status": "FAIL",
                    "reason": str(exc),
                }
        else:
            component_checks["audio"] = {
                "status": "FAIL",
                "reason": "audio WAV/metadata/timelines are incomplete",
            }

        if all(_regular_nonempty(session / relative) for relative in _FORCE_FILES):
            try:
                component_checks["force"] = verify_force_component(session)
            except TrainingFinalizationError as exc:
                blockers.append("force_raw_revalidation_failed")
                component_checks["force"] = {
                    "status": "FAIL",
                    "reason": str(exc),
                }
        else:
            component_checks["force"] = {
                "status": "FAIL",
                "reason": "force CSV/metadata/completion/calibration are incomplete",
            }

        if all(_regular_nonempty(session / relative) for relative in _ROBOT_FILES):
            if activated_contracts is None:
                blockers.append("robot_raw_revalidation_failed")
                component_checks["robot"] = {
                    "status": "FAIL",
                    "reason": (
                        "independently activated RobotStateContract is unavailable"
                    ),
                }
            else:
                try:
                    component_checks["robot"] = (
                        _verify_robot_component_against_contract(
                            session,
                            manifest_template,
                            activated_contracts.robot_contract,
                        )
                    )
                except TrainingFinalizationError as exc:
                    blockers.append("robot_raw_revalidation_failed")
                    component_checks["robot"] = {
                        "status": "FAIL",
                        "reason": str(exc),
                    }
        else:
            component_checks["robot"] = {
                "status": "FAIL",
                "reason": "four v4 robot raw artifacts are incomplete",
            }

        if (
            _regular_nonempty(session / "depth" / "depth_times.csv")
            and valid_depth_paths
        ):
            if activated_contracts is None:
                blockers.append("depth_raw_revalidation_failed")
                component_checks["depth_session"] = {
                    "status": "FAIL",
                    "reason": "activated camera depth scale is unavailable",
                }
            else:
                try:
                    component_checks["depth_session"] = (
                        verify_depth_session_authenticity(
                            session,
                            expected_depth_scale_mm=(
                                activated_contracts.depth_scale_mm
                            ),
                        )
                    )
                except TrainingFinalizationError as exc:
                    blockers.append("depth_raw_revalidation_failed")
                    component_checks["depth_session"] = {
                        "status": "FAIL",
                        "reason": str(exc),
                    }
        else:
            component_checks["depth_session"] = {
                "status": "FAIL",
                "reason": "depth timeline/NPY frame set is incomplete",
            }

        if all(
            _regular_nonempty(session / relative)
            for relative in (
                "geometry_frame_authenticity.json",
                "geometry_frame_packets.jsonl",
            )
        ):
            try:
                component_checks["geometry_packet_chain"] = (
                    verify_geometry_packet_component(session)
                )
            except TrainingFinalizationError as exc:
                blockers.append("geometry_packet_revalidation_failed")
                component_checks["geometry_packet_chain"] = {
                    "status": "FAIL",
                    "reason": str(exc),
                }
        else:
            component_checks["geometry_packet_chain"] = {
                "status": "FAIL",
                "reason": "geometry numeric packet log/receipt are incomplete",
            }
        checks["component_revalidation"] = component_checks

        binding_blockers, binding_checks = _production_artifact_binding_checks(
            session,
            manifest_template,
            activated_contracts=activated_contracts,
        )
        blockers.extend(binding_blockers)
        checks["production_artifact_bindings"] = binding_checks

    # Preserve a stable, human- and machine-readable order without duplicates.
    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": "wam_training_finalization_readiness_v1",
        "promotion_contract_version": PROMOTION_CONTRACT_VERSION,
        "status": "READY" if not blockers else "NOT_READY",
        "training_eligible": False,
        "blockers": blockers,
        "checks": checks,
    }


def _verify_audio_timelines(
    session: Path, metadata: Mapping[str, object]
) -> dict[str, object]:
    input_rows = _read_jsonl(
        session / "audio_input_times.jsonl", label="audio input timeline"
    )
    model_rows = _read_jsonl(
        session / "audio_times.jsonl", label="audio model timeline"
    )
    expected_input_fields = {
        "input_sample_offset",
        "input_sample_count",
        "t_start",
        "t_end",
        "source_block_start_t",
        "source_block_end_t",
    }
    input_rate = metadata.get("input_sample_rate")
    input_count = metadata.get("partial_n_samples")
    audio_start = metadata.get("audio_t0_monotonic")
    if (
        type(input_rate) is not int
        or input_rate <= 0
        or type(input_count) is not int
        or input_count <= 0
        or isinstance(audio_start, bool)
        or not isinstance(audio_start, (int, float))
        or not math.isfinite(float(audio_start))
    ):
        raise TrainingFinalizationError("audio input timeline metadata is invalid")
    expected_offset = 0
    previous_source_end: float | None = None
    pending_source_row: tuple[int, int, float, float] | None = None
    source_gap_tolerance = max(2.0 / float(input_rate), 0.010)
    for row_number, row in enumerate(input_rows, start=1):
        if set(row) != expected_input_fields:
            raise TrainingFinalizationError(
                f"audio input timeline row {row_number} schema is invalid"
            )
        offset = row.get("input_sample_offset")
        count = row.get("input_sample_count")
        if (
            type(offset) is not int
            or offset != expected_offset
            or type(count) is not int
            or count <= 0
        ):
            raise TrainingFinalizationError(
                f"audio input timeline row {row_number} is discontinuous"
            )
        times = []
        for field in (
            "t_start",
            "t_end",
            "source_block_start_t",
            "source_block_end_t",
        ):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TrainingFinalizationError(
                    f"audio input timeline row {row_number} time is invalid"
                )
            times.append(float(value))
        start, end, source_start, source_end = times
        expected_start = float(audio_start) + offset / float(input_rate)
        expected_end = float(audio_start) + (offset + count) / float(input_rate)
        tolerance = 0.5 / float(input_rate)
        if (
            abs(start - expected_start) > tolerance
            or abs(end - expected_end) > tolerance
            or end <= start
            or source_end <= source_start
        ):
            raise TrainingFinalizationError(
                f"audio input timeline row {row_number} timestamps are discontinuous"
            )
        if previous_source_end is None:
            if abs(source_start - float(audio_start)) > tolerance:
                raise TrainingFinalizationError(
                    "audio input timeline source clock does not start at audio t0"
                )
        elif abs(source_start - previous_source_end) > source_gap_tolerance:
            raise TrainingFinalizationError(
                "audio input timeline source clock is discontinuous"
            )
        if pending_source_row is not None:
            prior_line, prior_count, prior_start, prior_end = pending_source_row
            expected_duration = prior_count / float(input_rate)
            if abs((prior_end - prior_start) - expected_duration) > tolerance:
                raise TrainingFinalizationError(
                    f"audio input timeline source duration is invalid at row {prior_line}"
                )
        pending_source_row = (
            row_number,
            count,
            source_start,
            source_end,
        )
        previous_source_end = source_end
        expected_offset += count
    if pending_source_row is not None:
        final_line, final_count, final_start, final_end = pending_source_row
        minimum_duration = final_count / float(input_rate)
        duration_tolerance = max(1e-9, 1.0 / (float(input_rate) * 1_000.0))
        if final_end - final_start < minimum_duration - duration_tolerance:
            raise TrainingFinalizationError(
                f"audio input timeline final source row {final_line} is too short"
            )
    if expected_offset != input_count:
        raise TrainingFinalizationError(
            "audio input timeline sample count does not match capture metadata"
        )

    expected_model_fields = {
        "sample_offset",
        "sample_count",
        "t_start",
        "t_end",
        "source_input_sample_count",
        "input_sample_rate",
        "sample_rate",
    }
    sample_rate = metadata.get("sample_rate")
    sample_count = metadata.get("n_samples")
    audio_end = metadata.get("audio_t_end_monotonic")
    if type(sample_rate) is not int or sample_rate <= 0 or type(sample_count) is not int:
        raise TrainingFinalizationError("audio model timeline metadata is invalid")
    model_offset = 0
    source_input_total = 0
    for row_number, model in enumerate(model_rows, start=1):
        if set(model) != expected_model_fields:
            raise TrainingFinalizationError(
                f"audio model timeline row {row_number} schema is invalid"
            )
        count = model.get("sample_count")
        source_count = model.get("source_input_sample_count")
        if (
            model.get("sample_offset") != model_offset
            or type(count) is not int
            or count <= 0
            or type(source_count) is not int
            or source_count <= 0
            or model.get("input_sample_rate") != input_rate
            or model.get("sample_rate") != sample_rate
        ):
            raise TrainingFinalizationError(
                f"audio model timeline row {row_number} count/rate is invalid"
            )
        expected_start = float(audio_start) + model_offset / float(sample_rate)
        expected_end = expected_start + count / float(sample_rate)
        for field, expected in (("t_start", expected_start), ("t_end", expected_end)):
            value = model.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or abs(float(value) - expected) > 0.5 / float(sample_rate)
            ):
                raise TrainingFinalizationError(
                    f"audio model timeline row {row_number} {field} is invalid"
                )
        model_offset += count
        source_input_total += source_count
    if (
        model_offset != sample_count
        or source_input_total != input_count
        or abs(
            float(audio_start) + model_offset / float(sample_rate) - float(audio_end)
        )
        > 0.5 / float(sample_rate)
    ):
        raise TrainingFinalizationError(
            "audio model timeline totals do not match capture metadata"
        )
    for field, expected in (("t_start", float(audio_start)), ("t_end", float(audio_end))):
        value = model_rows[0 if field == "t_start" else -1].get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or abs(float(value) - expected) > 0.5 / float(sample_rate)
        ):
            raise TrainingFinalizationError(
                f"audio model timeline {field} does not match capture metadata"
            )
    return {
        "input_timeline_rows": len(input_rows),
        "timeline_sha256": {
            "audio_input_times.jsonl": _sha256_file(
                session / "audio_input_times.jsonl"
            ),
            "audio_times.jsonl": _sha256_file(session / "audio_times.jsonl"),
        },
    }


def verify_depth_session_authenticity(
    session_dir: str | Path,
    *,
    expected_depth_scale_mm: float | None = None,
) -> dict[str, object]:
    """Stream-replay aligned depth and reject empty or frozen capture channels.

    This is a session-level channel-liveness gate.  It does not replace the
    per-auscultation-point 5x5 patch replay performed by the geometry evidence
    verifier.
    """

    if (
        isinstance(expected_depth_scale_mm, bool)
        or not isinstance(expected_depth_scale_mm, (int, float))
        or not math.isfinite(float(expected_depth_scale_mm))
        or float(expected_depth_scale_mm) <= 0.0
    ):
        raise TrainingFinalizationError(
            "depth session requires an activated camera depth scale"
        )
    activated_depth_scale = float(expected_depth_scale_mm)
    session = Path(session_dir)
    depth_stride = _session_recording_frame_divisor(session)
    timeline_path = session / "depth" / "depth_times.csv"
    timeline_raw = _stable_regular_bytes(
        timeline_path, label="depth session timeline", maximum_bytes=64 * 1024 * 1024
    )
    try:
        reader = csv.DictReader(io.StringIO(timeline_raw.decode("utf-8")))
        fieldnames = reader.fieldnames
    except (UnicodeError, csv.Error) as exc:
        raise TrainingFinalizationError("depth session timeline is invalid CSV") from exc
    if not isinstance(fieldnames, list):
        raise TrainingFinalizationError("depth session timeline is invalid")
    production_fields = {
        "frame_index",
        "host_arrival_t_s",
        "device_t_ms",
        "depth_scale",
        "path",
    }
    fields = set(fieldnames)
    if fields != production_fields:
        raise TrainingFinalizationError(
            "depth session timeline must use the exact production schema"
        )

    previous_index: int | None = None
    previous_time: float | None = None
    previous_device_time_ms: float | None = None
    first_time: float | None = None
    first_device_time_ms: float | None = None
    observed_depth_scale: float | None = None
    previous_depth: np.ndarray | None = None
    previous_shape: tuple[int, int] | None = None
    frozen_run_start_time: float | None = None
    frozen_run_frames = 1
    longest_frozen_frames = 1
    longest_frozen_duration = 0.0
    observed_paths: set[str] = set()
    frame_count = 0
    valid_fractions: list[float] = []
    first_frame_sha256: str | None = None
    last_frame_sha256: str | None = None
    minimum_fraction = float(
        _DEPTH_AUTHENTICITY_POLICY["minimum_valid_fraction_per_frame"]
    )

    for row_number, row in enumerate(reader, start=1):
        try:
            frame_index = int(row["frame_index"])
            host_time = float(row["host_arrival_t_s"])
            device_time_ms = float(row["device_t_ms"])
            depth_scale = float(row["depth_scale"])
            relative = str(row["path"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingFinalizationError(
                f"depth session timeline row {row_number} is invalid"
            ) from exc
        expected_relatives = (f"depth/depth_{frame_index:06d}{depth_codec.DEPTH_SUFFIX}",
                              f"depth/depth_{frame_index:06d}.npy")
        if (
            frame_index < 0
            or (
                previous_index is not None
                and frame_index != previous_index + depth_stride
            )
            or not math.isfinite(host_time)
            or (previous_time is not None and host_time <= previous_time)
            or not math.isfinite(device_time_ms)
            or not math.isfinite(depth_scale)
            or depth_scale <= 0.0
            or relative not in expected_relatives
            or relative in observed_paths
        ):
            raise TrainingFinalizationError(
                f"depth session timeline row {row_number} is discontinuous"
            )
        if (
            previous_device_time_ms is not None
            and device_time_ms <= previous_device_time_ms
        ):
            raise TrainingFinalizationError(
                f"depth session device clock froze or regressed at row {row_number}"
            )
        if (
            depth_scale.hex() != activated_depth_scale.hex()
            or (
                observed_depth_scale is not None
                and depth_scale.hex() != observed_depth_scale.hex()
            )
        ):
            raise TrainingFinalizationError(
                f"depth session depth scale does not match activation at row {row_number}"
            )
        path = session / relative
        raw_npy = _stable_regular_bytes(
            path,
            label=f"depth frame {frame_index}",
            maximum_bytes=512 * 1024 * 1024,
        )
        try:
            depth = depth_codec.decode_depth(raw_npy)
        except (OSError, ValueError, TypeError) as exc:
            raise TrainingFinalizationError(
                f"depth frame {frame_index} is invalid NPY"
            ) from exc
        if (
            not isinstance(depth, np.ndarray)
            or depth.dtype != np.uint16
            or depth.ndim != 2
            or depth.size <= 0
            or (previous_shape is not None and tuple(depth.shape) != previous_shape)
        ):
            raise TrainingFinalizationError(
                f"depth frame {frame_index} dtype/shape is invalid"
            )
        depth_m = depth.astype(np.float64) * depth_scale / 1000.0
        valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 5.0)
        valid_count = int(np.count_nonzero(valid))
        valid_fraction = valid_count / float(depth.size)
        if valid_count == 0:
            raise TrainingFinalizationError(
                f"depth frame {frame_index} is all-zero/invalid"
            )
        if valid_fraction < minimum_fraction:
            raise TrainingFinalizationError(
                f"depth frame {frame_index} is near-empty "
                f"({valid_fraction:.6f} valid fraction)"
            )
        values = depth_m[valid]
        p01, p50, p99 = np.quantile(values, (0.01, 0.50, 0.99))
        if not np.isfinite((p01, p50, p99)).all() or not 0.0 < p01 <= p50 <= p99:
            raise TrainingFinalizationError(
                f"depth frame {frame_index} percentiles are invalid"
            )

        current = np.ascontiguousarray(depth)
        frozen_like = False
        if previous_depth is not None:
            delta = np.abs(
                current.astype(np.int32) - previous_depth.astype(np.int32)
            )
            changed_fraction = float(np.count_nonzero(delta)) / float(delta.size)
            frozen_like = bool(
                np.array_equal(current, previous_depth)
                or changed_fraction
                < float(
                    _DEPTH_AUTHENTICITY_POLICY[
                        "minimum_changed_fraction"
                    ]
                )
            )
            if frozen_like:
                if frozen_run_frames == 1:
                    frozen_run_start_time = previous_time
                frozen_run_frames += 1
                duration = host_time - float(frozen_run_start_time)
                longest_frozen_frames = max(longest_frozen_frames, frozen_run_frames)
                longest_frozen_duration = max(longest_frozen_duration, duration)
                if (
                    frozen_run_frames
                    >= int(
                        _DEPTH_AUTHENTICITY_POLICY[
                            "frozen_min_consecutive_frames"
                        ]
                    )
                    and duration
                    >= float(_DEPTH_AUTHENTICITY_POLICY["frozen_min_duration_s"])
                ):
                    raise TrainingFinalizationError(
                        "depth session contains a long bit-identical/near-zero-change run"
                    )
            else:
                frozen_run_frames = 1
                frozen_run_start_time = None

        # The verifier intentionally retains only bounded per-session state;
        # each frame is fully decoded/recomputed and then released.
        frame_sha256 = _sha256_bytes(raw_npy)
        if first_frame_sha256 is None:
            first_frame_sha256 = frame_sha256
        last_frame_sha256 = frame_sha256
        valid_fractions.append(round(valid_fraction, 9))
        frame_count += 1
        observed_paths.add(relative)
        previous_index = frame_index
        previous_time = host_time
        previous_device_time_ms = device_time_ms
        if first_time is None:
            first_time = host_time
            first_device_time_ms = device_time_ms
        if observed_depth_scale is None:
            observed_depth_scale = depth_scale
        previous_depth = current.copy()
        previous_shape = tuple(int(value) for value in depth.shape)

    if frame_count <= 0:
        raise TrainingFinalizationError("depth session timeline is empty")
    host_duration_s = float(previous_time) - float(first_time)
    device_duration_s = (
        float(previous_device_time_ms) - float(first_device_time_ms)
    ) / 1000.0
    if (
        frame_count < int(_DEPTH_AUTHENTICITY_POLICY["minimum_frame_count"])
        or host_duration_s
        < float(_DEPTH_AUTHENTICITY_POLICY["minimum_duration_s"])
    ):
        raise TrainingFinalizationError(
            "depth session does not meet the minimum frame count/duration"
        )
    maximum_clock_error_s = max(
        float(
            _DEPTH_AUTHENTICITY_POLICY[
                "maximum_device_host_duration_error_floor_s"
            ]
        ),
        host_duration_s
        * float(
            _DEPTH_AUTHENTICITY_POLICY[
                "maximum_device_host_duration_error_fraction"
            ]
        ),
    )
    if abs(device_duration_s - host_duration_s) > maximum_clock_error_s:
        raise TrainingFinalizationError(
            "depth session device clock duration disagrees with host time"
        )

    on_disk = {
        path.relative_to(session).as_posix()
        for path in ((session / "depth").glob(f"depth_*{depth_codec.DEPTH_SUFFIX}")
                     if any((session / "depth").glob(f"depth_*{depth_codec.DEPTH_SUFFIX}"))
                     else (session / "depth").glob("depth_*.npy"))
        if _regular_nonempty(path)
    }
    if on_disk != observed_paths:
        raise TrainingFinalizationError(
            "depth session timeline does not cover the exact NPY frame set"
        )
    return {
        "schema_version": "wam_depth_session_authenticity_v2",
        "policy_version": DEPTH_AUTHENTICITY_POLICY_VERSION,
        "status": "PASS",
        "policy": dict(_DEPTH_AUTHENTICITY_POLICY),
        "frame_count": frame_count,
        "host_duration_s": round(host_duration_s, 9),
        "device_duration_s": round(device_duration_s, 9),
        "depth_scale_mm": observed_depth_scale,
        "depth_scale_activation_bound": True,
        "shape": list(previous_shape or ()),
        "minimum_valid_fraction": min(valid_fractions),
        "median_valid_fraction": float(np.median(valid_fractions)),
        "longest_frozen_frame_run": longest_frozen_frames,
        "longest_frozen_duration_s": round(longest_frozen_duration, 9),
        "timeline_sha256": _sha256_bytes(timeline_raw),
        "first_frame_file_sha256": first_frame_sha256,
        "last_frame_file_sha256": last_frame_sha256,
        "per_point_depth_gate": "required_from_verify_production_evidence_capture",
    }


def verify_audio_component(session_dir: str | Path) -> dict[str, object]:
    """Decode WAV bytes and recompute the producer's signal-authenticity result."""

    session = Path(session_dir)
    wav_path = session / "audio.wav"
    metadata_path = session / "audio_capture.json"
    metadata = _read_json_object(metadata_path, label="audio capture metadata")
    try:
        with wave.open(str(wav_path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            n_samples = source.getnframes()
            compression = source.getcomptype()
            raw = source.readframes(n_samples)
            if source.readframes(1):
                raise TrainingFinalizationError("audio WAV has trailing PCM frames")
    except TrainingFinalizationError:
        raise
    except (OSError, EOFError, wave.Error) as exc:
        raise TrainingFinalizationError("audio WAV is missing or invalid") from exc
    if (
        channels != 1
        or sample_width != 2
        or compression != "NONE"
        or sample_rate <= 0
        or n_samples <= 0
        or len(raw) != n_samples * 2
    ):
        raise TrainingFinalizationError("audio WAV must be non-empty PCM16 mono")
    expected_fields = {
        "schema_version",
        "status",
        "training_eligible",
        "sample_rate",
        "input_sample_rate",
        "partial_sample_rate",
        "partial_encoding",
        "t0_monotonic",
        "audio_t0_monotonic",
        "audio_t_end_monotonic",
        "partial_n_samples",
        "n_samples",
        "sha256",
        "signal_authenticity",
    }
    if (
        set(metadata) != expected_fields
        or metadata.get("schema_version") != "wam_audio_stream_capture_v1"
        or metadata.get("status") != "sensors_finalized"
        or metadata.get("training_eligible") is not False
        or metadata.get("sample_rate") != sample_rate
        or metadata.get("n_samples") != n_samples
        or metadata.get("partial_encoding") != "signed_pcm_s16le_mono"
        or metadata.get("sha256") != _sha256_file(wav_path)
    ):
        raise TrainingFinalizationError("audio metadata does not bind the WAV")
    for field in ("t0_monotonic", "audio_t0_monotonic", "audio_t_end_monotonic"):
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise TrainingFinalizationError("audio metadata timeline is invalid")
    if not (
        float(metadata["audio_t0_monotonic"])
        >= float(metadata["t0_monotonic"])
        and float(metadata["audio_t_end_monotonic"])
        > float(metadata["audio_t0_monotonic"])
    ):
        raise TrainingFinalizationError("audio metadata timeline is invalid")
    try:
        from ausc.audio.audio_stream import AudioSignalAuthenticityAccumulator

        analyzer = AudioSignalAuthenticityAccumulator(sample_rate)
        analyzer.update(np.frombuffer(raw, dtype="<i2"))
        recomputed = analyzer.finish()
    except Exception as exc:
        raise TrainingFinalizationError(
            f"audio authenticity recomputation failed: {exc}"
        ) from exc
    if metadata.get("signal_authenticity") != recomputed:
        raise TrainingFinalizationError(
            "audio authenticity receipt does not match decoded WAV"
        )
    timeline = _verify_audio_timelines(session, metadata)
    return {
        "schema_version": "wam_audio_component_revalidation_v1",
        "status": "PASS",
        "sample_rate": sample_rate,
        "n_samples": n_samples,
        "wav_sha256": _sha256_file(wav_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "signal_authenticity": recomputed,
        **timeline,
    }


def _verify_robot_component_against_contract(
    session_dir: str | Path,
    manifest_template: Mapping[str, object],
    activated_contract: object,
) -> dict[str, object]:
    """Replay v4 bytes against a contract reconstructed by official activation."""

    try:
        from ausc.perception.robot.robot_state_stream import RobotStateContract
    except ImportError as exc:
        raise TrainingFinalizationError(
            "RobotStateContract implementation is unavailable"
        ) from exc
    if not isinstance(activated_contract, RobotStateContract):
        raise TrainingFinalizationError(
            "independently activated RobotStateContract is unavailable"
        )
    session = Path(session_dir)
    summary = _read_json_object(
        session / "robot_capture_summary.json", label="robot capture summary"
    )
    manifest_robot = manifest_template.get("robot")
    if not isinstance(manifest_robot, Mapping) or dict(manifest_robot) != summary:
        raise TrainingFinalizationError(
            "manifest robot identity does not exactly match raw v4 summary"
        )
    activated_identity = {
        "schema": "wam_openarm_exact_control_capture_summary_v4",
        "side": activated_contract.side,
        "controller_mode": activated_contract.controller_mode,
        "joint_names": list(activated_contract.joint_names),
        "joint_position_limits_rad": [
            list(pair) for pair in activated_contract.joint_position_limits_rad
        ],
        "leader_can_interface": activated_contract.leader_can_interface,
        "follower_can_interface": activated_contract.follower_can_interface,
        "urdf_sha256": activated_contract.urdf_sha256,
        "joint_mapper_sha256": activated_contract.joint_mapper_sha256,
        "camera_profile_sha256": activated_contract.camera_profile_sha256,
        "timing_artifact_sha256": activated_contract.timing_artifact_sha256,
        "pair_contract_sha256": activated_contract.pair_contract_sha256,
        "controller_binary_sha256": activated_contract.controller_binary_sha256,
        "controller_source_manifest_sha256": (
            activated_contract.controller_source_manifest_sha256
        ),
        "follower_config_sha256": activated_contract.follower_config_sha256,
        "openarm_sdk_sha256": activated_contract.openarm_sdk_sha256,
        "max_abs_joint_velocity_rad_s": (
            activated_contract.max_abs_joint_velocity_rad_s
        ),
        "gripper_motor_position_limits_rad": list(
            activated_contract.gripper_motor_position_limits_rad
        ),
        "buffer_duration_ns": activated_contract.buffer_duration_ns,
        "max_packets": activated_contract.max_packets,
        "max_gap_ns": activated_contract.max_gap_ns,
        "max_staleness_ns": activated_contract.max_staleness_ns,
        "max_exposure_interval_ns": (
            activated_contract.max_exposure_interval_ns
        ),
    }
    mismatched_fields = [
        field
        for field, expected in activated_identity.items()
        if canonical_json_bytes(summary.get(field))
        != canonical_json_bytes(expected)
    ]
    if mismatched_fields:
        raise TrainingFinalizationError(
            "robot summary does not match independent activation fields: "
            + ", ".join(mismatched_fields)
        )
    try:
        from ausc.perception.robot.robot_session_capture import (
            validate_robot_capture_artifacts,
        )

        replay = validate_robot_capture_artifacts(session, activated_contract)
    except Exception as exc:
        raise TrainingFinalizationError(f"robot v4 raw replay failed: {exc}") from exc
    if dict(replay) != summary or replay.get("training_eligible") is not True:
        raise TrainingFinalizationError(
            "robot v4 replay is not feedback-live and trajectory-authentic"
        )
    return {
        "schema_version": "wam_robot_component_revalidation_v1",
        "status": "PASS",
        "summary_schema": summary["schema"],
        "frame_count": summary["frame_count"],
        "control_tick_count": summary["control_tick_count"],
        "feedback_clean": summary.get("feedback_clean"),
        "trajectory_nondegenerate": summary.get("trajectory_nondegenerate"),
        "command_response_consistent": summary.get("command_response_consistent"),
        "activated_pair_contract_sha256": (
            activated_contract.pair_contract_sha256
        ),
        "activated_policy": {
            "buffer_duration_ns": activated_contract.buffer_duration_ns,
            "max_packets": activated_contract.max_packets,
            "max_gap_ns": activated_contract.max_gap_ns,
            "max_staleness_ns": activated_contract.max_staleness_ns,
            "max_exposure_interval_ns": (
                activated_contract.max_exposure_interval_ns
            ),
        },
    }


def verify_robot_component(
    session_dir: str | Path, manifest_template: Mapping[str, object]
) -> dict[str, object]:
    """Replay v4 robot bytes against the same-session production activation."""

    activated = load_activated_capture_contracts(
        session_dir, manifest_template
    )
    return _verify_robot_component_against_contract(
        session_dir, manifest_template, activated.robot_contract
    )


def verify_force_component(session_dir: str | Path) -> dict[str, object]:
    """Recompute force CSV values from the pinned calibration artifact."""

    try:
        from mechanical_sensor.force_feedback.force_capture import verify_force_capture

        summary = verify_force_capture(Path(session_dir))
    except Exception as exc:
        raise TrainingFinalizationError(f"force raw replay failed: {exc}") from exc
    if summary.status != "OK" or summary.signal_authenticity.get("status") not in (
        "PASS",
        "PASS_DEGRADED",
    ):
        raise TrainingFinalizationError("force raw replay is not authenticity PASS")
    return {
        "schema_version": "wam_force_component_revalidation_v1",
        "status": "PASS",
        "n_samples": summary.n_samples,
        "csv_sha256": summary.sha256,
        "calibration_sha256": summary.calibration_sha256,
        "device_identity_sha256": summary.device_identity_sha256,
        "signal_authenticity": summary.signal_authenticity,
    }


def _unconfigured_replay_trust_root(_record: object) -> bool:
    """The default replay-attestation trust root: refuse everything.

    No signing/registry trust root for replay attestations is wired into the
    finalizer yet, so a present attestation can never authorize promotion.  This
    makes the gate EXPLICITLY refuse (a session-local JSON file is not proof)
    rather than relying on the downstream NOT_READY lock to hide a forgeable
    authorization.  A real trust verifier is injected once it exists.
    """

    return False


def _apply_replay_attestation(
    session_dir: str | Path,
    base: dict[str, object],
    *,
    component: str,
    recorded_identity: Mapping[str, object],
    recorded_item_count: object,
    trust_verifier: "Callable[[object], bool]" = _unconfigured_replay_trust_root,
) -> dict[str, object]:
    """Fold a present, valid replay attestation into a component's verdict.

    Absent attestation -> return ``base`` unchanged (``production_authorized``
    stays False and the diagnostic blocker remains).  A present attestation must
    pass the mandatory ``trust_verifier`` AND verify against this exact session and
    model identity, or the whole component fails closed via
    :class:`TrainingFinalizationError` — a present-but-invalid or un-vouched
    attestation is never silently downgraded to the diagnostic state.
    """

    from ausc.perception.replay_attestation import (
        ReplayAttestationError,
        load_session_attestation,
        verify_replay_attestation,
    )

    try:
        raw = load_session_attestation(session_dir, component)
    except ReplayAttestationError as exc:
        raise TrainingFinalizationError(
            f"{component} replay attestation path is unsafe: {exc}"
        ) from exc
    if raw is None:
        return dict(base)
    try:
        evidence = verify_replay_attestation(
            session_dir,
            raw,
            component=component,
            recorded_identity=recorded_identity,
            recorded_item_count=recorded_item_count,
            trust_verifier=trust_verifier,
        )
    except ReplayAttestationError as exc:
        raise TrainingFinalizationError(
            f"{component} replay attestation is invalid: {exc}"
        ) from exc
    authorized = dict(base)
    authorized["production_authorized"] = True
    authorized.pop("blocker", None)
    authorized["replay_attestation"] = evidence
    return authorized


def verify_geometry_packet_component(
    session_dir: str | Path,
    *,
    trust_verifier: "Callable[[object], bool]" = _unconfigured_replay_trust_root,
) -> dict[str, object]:
    """Replay every numeric packet pair; a valid attestation authorizes promotion."""

    try:
        from ausc.perception.tracking.geometry_channel_packet import (
            verify_geometry_packet_log_files,
        )

        entries, receipt = verify_geometry_packet_log_files(Path(session_dir))
    except Exception as exc:
        raise TrainingFinalizationError(
            f"geometry numeric packet replay failed: {exc}"
        ) from exc
    base = {
        "schema_version": "wam_geometry_packet_component_revalidation_v1",
        "status": "PASS",
        "packet_count": len(entries),
        "packet_log_sha256": receipt.packet_log_sha256,
        "receipt_sha256": receipt.receipt_sha256,
        "episode_id": receipt.episode_id,
        "perception_release_id": receipt.perception_release_id,
        "perception_model_sha256": receipt.perception_model_sha256,
        "independent_release_id": receipt.independent_release_id,
        "independent_model_sha256": receipt.independent_model_sha256,
        "production_authorized": False,
        "blocker": "lossless patient-frame and depth-patch replay still required",
    }
    # Authorize promotion only when the deployment-activated model has been
    # independently re-run on the exact persisted inputs and its outputs
    # reproduce the recorded packets.  Absent attestation -> stay diagnostic;
    # present-but-invalid attestation -> hard failure (never silently PASS).
    result = _apply_replay_attestation(
        session_dir,
        base,
        component="geometry",
        recorded_identity={
            "release_id": receipt.perception_release_id,
            "model_sha256": receipt.perception_model_sha256,
        },
        recorded_item_count=len(entries),
        trust_verifier=trust_verifier,
    )
    if result.get("production_authorized") is True:
        # The perception channel replayed.  The independent metric patient-pose
        # channel must ALSO be independently replayed (its own attestation), or it
        # is merely caller-self-reported — so promotion requires BOTH channels.
        independent = _apply_replay_attestation(
            session_dir,
            {"production_authorized": False,
             "blocker": "independent geometry channel replay still required"},
            component="geometry_independent",
            recorded_identity={
                "release_id": receipt.independent_release_id,
                "model_sha256": receipt.independent_model_sha256,
            },
            recorded_item_count=len(entries),
            trust_verifier=trust_verifier,
        )
        if independent.get("production_authorized") is not True:
            raise TrainingFinalizationError(
                "geometry independent-channel replay attestation is missing; both "
                "the perception and the independent channels must be replayed"
            )
        result["independent_replay_attestation"] = independent["replay_attestation"]
    if result.get("production_authorized") is True:
        # The attestation binds the packet log + ledger, but the ledger only
        # *references* the RGB/depth/mask model-input files by hash.  Re-verify
        # each referenced array against the ledger so a post-attestation tamper of
        # a model input can never survive: reload_for_inference re-hashes every
        # file and raises on any mismatch.
        try:
            reverified = _reverify_geometry_model_inputs(session_dir)
        except Exception as exc:
            raise TrainingFinalizationError(
                f"geometry model-input re-verification failed: {exc}"
            ) from exc
        result["model_inputs_reverified"] = reverified
    return result


def _reverify_geometry_model_inputs(session_dir: str | Path) -> int:
    """Re-open and hash-verify every ledger-referenced geometry model input.

    Returns the number of frames re-verified.  Raises if the ledger is malformed
    or any referenced RGB/depth/mask array no longer matches its recorded hash.
    """

    from ausc.perception.tracking.production_evidence_capture import (
        read_geometry_model_input_ledger,
    )

    records = read_geometry_model_input_ledger(Path(session_dir))
    for record in records:
        record.reload_for_inference()  # re-hashes rgb/depth/masks; raises on tamper
    return len(records)


def verify_quality_force_rows(
    session_dir: str | Path, packets: object
) -> dict[str, object]:
    """Prove every Q force sample is an exact row from verified ``force.csv``."""

    session = Path(session_dir)
    path = session / "force.csv"
    raw = _stable_regular_bytes(
        path, label="quality force.csv binding", maximum_bytes=512 * 1024 * 1024
    )
    fields = (
        "host_t_s",
        "raw",
        "hall_v",
        "displacement_mm",
        "force_n",
        "status",
        "device_t_ms",
    )
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
        if tuple(reader.fieldnames or ()) != fields:
            raise TrainingFinalizationError(
                "quality force.csv has an invalid exact header"
            )
        available: set[tuple[object, ...]] = set()
        for row_number, row in enumerate(reader, start=2):
            if set(row) != set(fields) or None in row:
                raise TrainingFinalizationError(
                    f"quality force.csv row {row_number} is malformed"
                )
            parsed = (
                float(row["host_t_s"]),
                int(row["raw"]),
                float(row["hall_v"]),
                float(row["displacement_mm"]),
                float(row["force_n"]),
                row["status"],
                float(row["device_t_ms"]),
            )
            numeric = parsed[:5] + parsed[6:]
            if (
                parsed[1] < 0
                or any(not math.isfinite(float(value)) for value in numeric)
                # P0-4: an over-threshold reading is a VALID physician-reference
                # measurement, not a session-rejecting fault; only a non-native status
                # is invalid (genuineness travels as per-frame contact_force_valid).
                or parsed[5] not in {"OK", "WARN", "SCREENED_LIMIT", "HARD_LIMIT"}
            ):
                raise TrainingFinalizationError(
                    f"quality force.csv row {row_number} has a non-native status"
                )
            available.add(parsed)
    except TrainingFinalizationError:
        raise
    except (UnicodeError, csv.Error, TypeError, ValueError, KeyError) as exc:
        raise TrainingFinalizationError("quality force.csv cannot be parsed") from exc
    if not available:
        raise TrainingFinalizationError("quality force.csv is empty")

    if isinstance(packets, (str, bytes, bytearray, Mapping)):
        raise TrainingFinalizationError("quality packet collection is invalid")
    try:
        packet_values = tuple(packets)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TrainingFinalizationError("quality packet collection is invalid") from exc
    if not packet_values:
        raise TrainingFinalizationError("quality packet collection is empty")
    matched_rows = []
    for index, packet in enumerate(packet_values):
        force = packet.get("force_sample") if isinstance(packet, Mapping) else None
        if not isinstance(force, Mapping) or set(force) != set(fields):
            raise TrainingFinalizationError(
                f"quality packet {index} force evidence schema is invalid"
            )
        try:
            candidate = (
                float(force["host_t_s"]),
                force["raw"],
                float(force["hall_v"]),
                float(force["displacement_mm"]),
                float(force["force_n"]),
                force["status"],
                float(force["device_t_ms"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise TrainingFinalizationError(
                f"quality packet {index} force evidence is invalid"
            ) from exc
        if type(candidate[1]) is not int or candidate not in available:
            raise TrainingFinalizationError(
                f"quality packet {index} force sample does not exist in force.csv"
            )
        matched_rows.append({field: force[field] for field in fields})
    return {
        "schema_version": "wam_quality_force_csv_binding_v1",
        "status": "PASS",
        "matched_packet_count": len(matched_rows),
        "force_csv_sha256": _sha256_bytes(raw),
        "matched_rows_sha256": _sha256_bytes(
            b"".join(canonical_json_bytes(row) for row in matched_rows)
        ),
    }


def verify_quality_packet_component(
    session_dir: str | Path,
    *,
    trust_verifier: "Callable[[object], bool]" = _unconfigured_replay_trust_root,
) -> dict[str, object]:
    """Replay the entire Q chain; lossless model-input replay remains separate."""

    try:
        from ausc.realtime_q import read_quality_packet_log, verify_quality_packet_log

        receipt = verify_quality_packet_log(Path(session_dir))
        packets = read_quality_packet_log(
            Path(session_dir) / "quality_inference_packets.jsonl"
        )
    except Exception as exc:
        raise TrainingFinalizationError(f"quality packet replay failed: {exc}") from exc
    chain = receipt.get("chain_verification")
    authenticity = receipt.get("quality_authenticity")
    if (
        not isinstance(chain, Mapping)
        or chain.get("training_signal_eligible") is not True
        or not isinstance(authenticity, Mapping)
    ):
        raise TrainingFinalizationError("quality full-chain replay is not eligible")
    model = authenticity.get("model")
    if not isinstance(model, Mapping):
        raise TrainingFinalizationError("quality model activation identity is missing")
    force_binding = verify_quality_force_rows(session_dir, packets)
    base = {
        "schema_version": "wam_quality_packet_component_revalidation_v1",
        "status": "PASS",
        "packet_count": receipt["packet_count"],
        "packet_log_sha256": receipt["sha256"],
        "last_packet_chain_sha256": receipt["last_packet_chain_sha256"],
        "model_release_id": model.get("release_id"),
        "model_sha256": model.get("model_sha256"),
        "policy_sha256": authenticity.get("policy_sha256"),
        "model_input_inventory": receipt.get("model_input_inventory"),
        "force_input_identity": receipt.get("force_input_identity"),
        "force_csv_binding": force_binding,
        # The packet chain proves that persisted inputs, force rows, policy EMA,
        # and recorded outputs are internally consistent.  On its own it does NOT
        # load the deployment-activated quality weights and independently re-run
        # the model on audio windows reconstructed from canonical ``audio.wav``
        # sample offsets; a self-consistent forged model/output pair could
        # otherwise be mislabeled as production truth.  Promotion is authorized
        # only by a valid replay attestation covering exactly this session.
        "production_authorized": False,
        "blocker": (
            "activated quality model replay against lossless audio windows is required"
        ),
    }
    return _apply_replay_attestation(
        session_dir,
        base,
        component="quality",
        recorded_identity={
            "release_id": model.get("release_id"),
            "model_sha256": model.get("model_sha256"),
        },
        recorded_item_count=receipt["packet_count"],
        trust_verifier=trust_verifier,
    )


def finalize_training_session(
    session_dir: str | Path,
    *,
    manifest_template: Mapping[str, object],
    finalizer_identity: Mapping[str, object],
) -> dict[str, object]:
    """Finalize only a fully replayable promotion-v2 session.

    At the present capture contract the function intentionally raises
    :class:`TrainingFinalizationNotReady` before writing anything.  Once the
    required lossless ledgers are produced, their independent validators can
    be connected here; no diagnostic receipt is silently upgraded meanwhile.
    """

    report = inspect_training_finalization_readiness(
        session_dir,
        manifest_template=manifest_template,
        finalizer_identity=finalizer_identity,
    )
    if report["status"] != "READY":
        raise TrainingFinalizationNotReady(report)
    # Readiness now includes the per-component replay attestations: the geometry
    # and quality components each reached ``production_authorized`` only via a
    # valid attestation covering exactly this session's persisted inputs.  The
    # sole remaining gap is the authenticity/finalization publish path that mints
    # a training-eligible session; it is intentionally not yet activated here
    # (production promotion, Step 4).  A diagnostic receipt is never silently
    # upgraded meanwhile.
    hardened = dict(report)
    hardened["status"] = "NOT_READY"
    hardened["blockers"] = ["production_publish_not_activated"]
    hardened["checks"] = dict(report["checks"])
    hardened["checks"]["production_publish"] = {
        "status": "PENDING",
        "reason": (
            "replay attestations verified; the training authenticity and "
            "finalization publish path is not yet activated"
        ),
    }
    raise TrainingFinalizationNotReady(hardened)


__all__ = [
    "ActivatedCaptureContracts",
    "AUTHENTICITY_NAME",
    "FINALIZATION_NAME",
    "LOSSLESS_EVIDENCE_ARTIFACTS",
    "MANIFEST_NAME",
    "PROMOTION_CONTRACT_VERSION",
    "REQUIRED_BASE_RAW_ARTIFACTS",
    "TrainingFinalizationError",
    "TrainingFinalizationNotReady",
    "canonical_json_bytes",
    "finalize_training_session",
    "inspect_training_finalization_readiness",
    "load_activated_capture_contracts",
    "module_sha256",
    "verify_audio_component",
    "verify_depth_session_authenticity",
    "verify_force_component",
    "verify_geometry_packet_component",
    "verify_quality_force_rows",
    "verify_quality_packet_component",
    "verify_robot_component",
    "write_exclusive_canonical_json",
]
