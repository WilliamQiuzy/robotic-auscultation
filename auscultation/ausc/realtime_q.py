"""Authenticated streaming quality packets (realtime multimodal pipeline §4).

Production is fail-closed unless callers inject a hash-pinned scorer plus an exact
policy declaring its model role and output semantics. A clinical-usability Q
(``P(usable)``) and a signal-artifact gate (often ``P(poor)``) are separate roles;
the latter must be explicitly transformed and must not silently replace the former.

``band_energy_proxy`` exists only for structural diagnostics. The CLI labels its
output DIAGNOSTIC-ONLY and cannot issue a production authenticity receipt.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
from collections.abc import Iterable, Mapping
from numbers import Real

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, "audio"), os.path.join(ROOT, "mechanical_sensor", "force_feedback")]
from health import audio_status  # noqa: E402


QUALITY_PACKET_SCHEMA_VERSION = "wam_quality_inference_packet_v2"
QUALITY_PACKET_HASH_CONTRACT = "wam_quality_packet_hash_v2"
QUALITY_AUTHENTICITY_SCHEMA_VERSION = "wam_quality_authenticity_v2"
QUALITY_AUTHENTICITY_POLICY_VERSION = "quality_packet_liveness_v3"
QUALITY_CHAIN_VERIFICATION_SCHEMA_VERSION = (
    "wam_quality_packet_chain_verification_v2"
)
QUALITY_OUTPUT_NONDEGENERACY_POLICY_VERSION = (
    "quality_output_nondegeneracy_v2"
)
QUALITY_OUTPUT_MINIMUM_RANGE = 1e-4
QUALITY_OUTPUT_MINIMUM_STD = 5e-5
QUALITY_OUTPUT_MINIMUM_PACKET_COUNT = 3
# Match the conservative PCM16 End-of-session floor.  One-LSB digital dither
# must not become a "live" Q model input merely because it is non-zero.
QUALITY_MODEL_INPUT_MINIMUM_RMS = 2.0 / 32_767.0
QUALITY_MODEL_INPUT_MINIMUM_PEAK = 2.0 / 32_767.0
QUALITY_MODEL_INPUT_MAXIMUM_BYTES = 64 * 1024 * 1024
QUALITY_POLICY_SCHEMA_VERSION = "wam_quality_policy_v1"
QUALITY_PROMOTION_BLOCKER = (
    "quality packet log receipt alone is not bound by the final multi-channel "
    "training finalization"
)
_QUALITY_POLICY_FIELDS = frozenset({
    "schema_version",
    "policy_id",
    "force_band_n",
    "window_s",
    "ema",
    "max_audio_age_s",
    "max_inference_age_s",
    "max_force_age_s",
    "min_packets_for_receipt",
    "output_contract",
})
_OUTPUT_CONTRACT_FIELDS = frozenset({
    "model_role",
    "output_semantics",
    "positive_class",
    "calibration",
    "transform_to_p_usable",
    "output_range",
})
_CALIBRATION_FIELDS = frozenset({"method", "release_id", "sha256"})
_FORCE_SAMPLE_STATUSES = frozenset({
    "OK",
    "WARN",
    "SCREENED_LIMIT",
    "HARD_LIMIT",
    "STALE",
    "NO_DATA",
})
_QUALITY_PACKET_FIELDS = frozenset({
    "schema_version",
    "scorer_scope",
    "production_eligible",
    "model",
    "inference_sequence",
    "inference_monotonic_s",
    "observed_monotonic_s",
    "audio_window",
    "force_sample",
    "result",
    "hash_contract",
    "previous_packet_chain_sha256",
    "packet_sha256",
    "packet_chain_sha256",
})
_QUALITY_MODEL_FIELDS = frozenset({
    "release_id",
    "model_sha256",
    "policy_id",
    "policy_sha256",
    "output_contract",
})
_QUALITY_AUDIO_BASE_FIELDS = frozenset({
    "t_start_monotonic_s",
    "t_end_monotonic_s",
    "sample_rate",
    "sample_count",
    "content_sha256",
    "encoding",
    "status",
})
_QUALITY_PRODUCTION_AUDIO_FIELDS = (
    _QUALITY_AUDIO_BASE_FIELDS | {"model_input", "raw_source"}
)
#: Provenance that ties a production Q window to the session's physical audio.
#: Without it, a real-model replay only proves the persisted f32 window and the
#: Q output are self-consistent — NOT that the window was actually derived from
#: this session's canonical ``audio.wav``.  These fields name the exact PCM16
#: sample range, the source channel, and the resampler used, so the derivation
#: is reproducible.
_QUALITY_RAW_SOURCE_FIELDS = frozenset({
    "source_channel",
    "source_sample_start",
    "source_sample_end",
    "source_sample_rate",
    "resampler_identity",
})
_QUALITY_RESAMPLER_IDENTITY_FIELDS = frozenset({
    "algorithm",
    "input_sample_rate",
    "output_sample_rate",
})
QUALITY_AUDIO_RAW_SOURCE_SCHEMA_VERSION = "wam_quality_audio_raw_source_v1"
_QUALITY_FORCE_FIELDS = frozenset({
    "host_t_s",
    "raw",
    "hall_v",
    "displacement_mm",
    "force_n",
    "status",
    "device_t_ms",
})
_PRODUCTION_RESULT_FIELDS = frozenset({
    "force_gate_open",
    "q_raw",
    "p_usable_raw",
    "p_usable",
})
_DIAGNOSTIC_RESULT_FIELDS = frozenset({
    "force_gate_open",
    "q_raw",
    "diagnostic_score_raw",
    "diagnostic_score_smoothed",
})
QUALITY_PACKET_LOG_BASENAME = "quality_inference_packets.jsonl"
QUALITY_PACKET_LOG_STAGING_BASENAME = (
    ".quality_inference_packets.jsonl.inprogress"
)
QUALITY_PACKET_LOG_AUTHENTICITY_SCHEMA_VERSION = (
    "wam_quality_packet_log_authenticity_v3"
)
QUALITY_PACKET_LOG_RECEIPT_BASENAME = "quality_inference_authenticity.json"
QUALITY_MODEL_INPUT_SCHEMA_VERSION = "wam_quality_model_input_v1"
QUALITY_MODEL_INPUT_INVENTORY_SCHEMA_VERSION = (
    "wam_quality_model_input_inventory_v1"
)
QUALITY_MODEL_INPUT_INVENTORY_HASH_CONTRACT = (
    "canonical_quality_model_input_claims_json_sha256_v1"
)
QUALITY_MODEL_INPUT_STAGING_DIRNAME = ".quality_model_inputs.inprogress"
QUALITY_MODEL_INPUT_DIRNAME = "quality_model_inputs"
QUALITY_FORCE_INPUT_IDENTITY_SCHEMA_VERSION = (
    "wam_quality_force_input_identity_v2"
)
QUALITY_FORCE_INPUT_HASH_CONTRACT = (
    "canonical_exact_matched_force_csv_rows_jsonl_sha256_v2"
)
QUALITY_FORCE_CSV_MATCH_CONTRACT = (
    "exact_seven_fields_unique_host_and_device_timestamp_row_v1"
)
QUALITY_FORCE_CSV_BASENAME = "force.csv"
QUALITY_FORCE_CSV_FIELDS = (
    "host_t_s",
    "raw",
    "hall_v",
    "displacement_mm",
    "force_n",
    "status",
    "device_t_ms",
)
_QUALITY_MODEL_INPUT_FIELDS = frozenset({
    "schema_version",
    "path",
    "inference_sequence",
    "dtype",
    "shape",
    "bytes",
    "sha256",
})
_QUALITY_MODEL_INPUT_INVENTORY_FIELDS = frozenset({
    "schema_version",
    "status",
    "directory",
    "hash_contract",
    "file_count",
    "total_bytes",
    "files",
    "inventory_sha256",
})
_QUALITY_PACKET_LOG_RECEIPT_FIELDS = frozenset({
    "schema_version",
    "status",
    "path",
    "receipt_path",
    "bytes",
    "sha256",
    "packet_count",
    "last_packet_chain_sha256",
    "verified_at_monotonic_s",
    "quality_authenticity",
    "chain_verification",
    "model_input_inventory",
    "force_input_identity",
    "promotion_eligible",
    "promotion_blockers",
})


class QualityNotReadyError(RuntimeError):
    """A production Q result was requested without a complete pinned runtime."""


class QualityPacketError(RuntimeError):
    """A Q inference packet failed identity, freshness, or progress checks."""


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_clone(payload: object):  # noqa: ANN202 - preserves JSON container shape
    return json.loads(_canonical_json_bytes(payload))


def _quality_packet_preimage(packet: Mapping[str, object]) -> bytes:
    """Canonical packet preimage: compact sorted JSON + LF, without final hashes."""

    preimage = _json_clone(packet)
    preimage.pop("packet_sha256", None)
    preimage.pop("packet_chain_sha256", None)
    return _canonical_json_bytes(preimage)


def _quality_packet_hashes(packet: Mapping[str, object]) -> tuple[str, str]:
    previous = packet.get("previous_packet_chain_sha256")
    if previous is None:
        previous_raw = bytes(32)
    else:
        previous_raw = bytes.fromhex(
            _require_sha256(previous, "quality previous packet chain SHA-256")
        )
    packet_sha256 = _sha256_bytes(_quality_packet_preimage(packet))
    chain_sha256 = _sha256_bytes(previous_raw + bytes.fromhex(packet_sha256))
    return packet_sha256, chain_sha256


def verify_quality_inference_packet(
    packet: Mapping[str, object],
) -> dict[str, object]:
    """Independently verify the canonical preimage, packet hash, and chain link."""

    if not isinstance(packet, Mapping):
        raise QualityPacketError("quality inference packet must be an object")
    candidate = _json_clone(packet)
    if set(candidate) != _QUALITY_PACKET_FIELDS:
        raise QualityPacketError("quality inference packet has an invalid exact schema")
    if candidate.get("schema_version") != QUALITY_PACKET_SCHEMA_VERSION:
        raise QualityPacketError("quality inference packet schema is invalid")
    if candidate.get("hash_contract") != QUALITY_PACKET_HASH_CONTRACT:
        raise QualityPacketError("quality packet hash contract is invalid")
    model = candidate.get("model")
    audio = candidate.get("audio_window")
    force = candidate.get("force_sample")
    result = candidate.get("result")
    production_eligible = candidate.get("production_eligible") is True
    expected_result_fields = (
        _PRODUCTION_RESULT_FIELDS
        if production_eligible
        else _DIAGNOSTIC_RESULT_FIELDS
    )
    expected_audio_fields = (
        _QUALITY_PRODUCTION_AUDIO_FIELDS
        if production_eligible
        else _QUALITY_AUDIO_BASE_FIELDS
    )
    if (
        not isinstance(model, Mapping)
        or set(model) != _QUALITY_MODEL_FIELDS
        or not isinstance(audio, Mapping)
        or set(audio) != expected_audio_fields
        or not isinstance(force, Mapping)
        or set(force) != _QUALITY_FORCE_FIELDS
        or not isinstance(result, Mapping)
        or set(result) != expected_result_fields
    ):
        raise QualityPacketError("quality inference packet nested schema is invalid")
    if production_eligible:
        try:
            _normalize_model_input_claim(audio.get("model_input"))
        except (TypeError, ValueError) as exc:
            raise QualityPacketError(
                f"quality inference packet model input claim is invalid: {exc}"
            ) from exc
        # A production window must prove where it came from in the physical
        # audio: exact PCM16 sample range + channel + resampler identity, with
        # the resampler output rate equal to this window's sample rate.
        try:
            _normalize_raw_source_claim(
                audio.get("raw_source"),
                window_sample_rate=audio.get("sample_rate"),
            )
        except (TypeError, ValueError) as exc:
            raise QualityPacketError(
                f"quality inference packet raw-source claim is invalid: {exc}"
            ) from exc
    try:
        stored_packet_sha = _require_sha256(
            candidate.get("packet_sha256"), "quality packet SHA-256"
        )
        stored_chain_sha = _require_sha256(
            candidate.get("packet_chain_sha256"), "quality packet chain SHA-256"
        )
        computed_packet_sha, computed_chain_sha = _quality_packet_hashes(candidate)
    except (TypeError, ValueError) as exc:
        raise QualityPacketError(str(exc)) from exc
    if stored_packet_sha != computed_packet_sha:
        raise QualityPacketError(
            "quality packet SHA-256 does not match canonical preimage"
        )
    if stored_chain_sha != computed_chain_sha:
        raise QualityPacketError("quality packet chain SHA-256 does not match")
    return candidate


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("quality packet log write made no progress")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_regular_path(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QualityPacketError(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise QualityPacketError(f"{label} must be a regular non-symlink file")
    return metadata


def _regular_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _sha256_regular_file(
    path: Path,
    *,
    label: str = "quality packet log",
) -> tuple[str, int]:
    metadata = _require_regular_path(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise QualityPacketError(f"{label} cannot be opened: {exc}") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        opened_before = os.fstat(fd)
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(fd)
        try:
            current = path.lstat()
        except OSError as exc:
            raise QualityPacketError(f"{label} changed while hashing") from exc
    finally:
        os.close(fd)
    if (
        _regular_file_identity(metadata) != _regular_file_identity(opened_before)
        or _regular_file_identity(opened_before) != _regular_file_identity(after)
        or _regular_file_identity(after) != _regular_file_identity(current)
        or stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or total != metadata.st_size
    ):
        raise QualityPacketError(f"{label} changed while hashing")
    return digest.hexdigest(), total


def _read_small_regular_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = 8 * 1024 * 1024,
) -> bytes:
    metadata = _require_regular_path(path, label)
    if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
        raise QualityPacketError(f"{label} has an invalid size")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise QualityPacketError(f"{label} cannot be opened: {exc}") from exc
    chunks: list[bytes] = []
    total = 0
    try:
        opened_before = os.fstat(fd)
        while True:
            block = os.read(fd, min(65_536, maximum_bytes + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum_bytes:
                raise QualityPacketError(f"{label} exceeds its size limit")
        after = os.fstat(fd)
        try:
            current = path.lstat()
        except OSError as exc:
            raise QualityPacketError(f"{label} changed while reading") from exc
    finally:
        os.close(fd)
    if (
        _regular_file_identity(metadata) != _regular_file_identity(opened_before)
        or _regular_file_identity(opened_before) != _regular_file_identity(after)
        or _regular_file_identity(after) != _regular_file_identity(current)
        or stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or total != metadata.st_size
    ):
        raise QualityPacketError(f"{label} changed while reading")
    return b"".join(chunks)


def _validate_quality_model_input_signal(
    raw: bytes,
    *,
    label: str,
) -> dict[str, float]:
    """Decode exact f32-le scorer bytes and reject dead/non-finite audio."""

    if (
        not isinstance(raw, bytes)
        or not raw
        or len(raw) % 4 != 0
        or len(raw) > QUALITY_MODEL_INPUT_MAXIMUM_BYTES
    ):
        raise QualityPacketError(
            f"{label} must be bounded non-empty float32-le bytes"
        )
    samples = np.frombuffer(raw, dtype="<f4")
    if not bool(np.all(np.isfinite(samples))):
        raise QualityPacketError(f"{label} contains non-finite samples")
    samples64 = samples.astype(np.float64, copy=False)
    peak = float(np.max(np.abs(samples64)))
    rms = float(np.sqrt(np.mean(np.square(samples64))))
    if (
        peak < QUALITY_MODEL_INPUT_MINIMUM_PEAK
        or rms < QUALITY_MODEL_INPUT_MINIMUM_RMS
    ):
        raise QualityPacketError(
            f"{label} signal is silent or below the authenticity floor"
        )
    return {"peak": peak, "rms": rms}


def _write_exclusive_regular_file(path: Path, raw: bytes, *, label: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags, 0o640)
    except OSError as exc:
        raise QualityPacketError(f"{label} cannot be created: {exc}") from exc
    try:
        _write_all(fd, raw)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)
    _fsync_directory(path.parent)


def read_quality_packet_log(path: str | Path) -> list[dict[str, object]]:
    """Securely parse and hash-verify every canonical JSONL packet."""

    selected = Path(path)
    _require_regular_path(selected, "quality packet log")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(selected), flags)
    except OSError as exc:
        raise QualityPacketError(f"quality packet log cannot be opened: {exc}") from exc
    packets: list[dict[str, object]] = []
    try:
        with os.fdopen(fd, "rb", closefd=True) as stream:
            for line_number, raw in enumerate(stream, start=1):
                if len(raw) > 8 * 1024 * 1024:
                    raise QualityPacketError(
                        f"quality packet log line {line_number} is too large"
                    )
                if not raw.endswith(b"\n") or not raw.strip():
                    raise QualityPacketError(
                        f"quality packet log line {line_number} is incomplete"
                    )
                try:
                    parsed = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise QualityPacketError(
                        f"quality packet log line {line_number} is invalid JSON"
                    ) from exc
                packets.append(verify_quality_inference_packet(parsed))
    except BaseException:
        # fdopen owns and closes fd; only the exception needs to propagate.
        raise
    if not packets:
        raise QualityPacketError("quality packet log is empty")
    return packets


def _verify_quality_model_input_inventory(
    packet_log: Iterable[Mapping[str, object]],
    model_input_dir: str | Path,
) -> dict[str, object]:
    root = Path(model_input_dir)
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise QualityPacketError(
            f"quality model input directory is unavailable: {exc}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise QualityPacketError(
            "quality model input directory must be a real non-symlink directory"
        )
    claims: list[dict[str, object]] = []
    expected_names: set[str] = set()
    previous_sequence: int | None = None
    for packet_index, packet in enumerate(packet_log):
        try:
            sequence = packet.get("inference_sequence")
            claim = _normalize_model_input_claim(
                packet["audio_window"].get("model_input")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QualityPacketError(
                f"quality model input claim {packet_index} is invalid: {exc}"
            ) from exc
        if (
            type(sequence) is not int
            or claim["inference_sequence"] != sequence
            or (
                previous_sequence is not None
                and sequence != previous_sequence + 1
            )
        ):
            raise QualityPacketError(
                "quality model input inventory has duplicate/replay/gap sequence"
            )
        audio = packet["audio_window"]
        if (
            claim["shape"] != [audio.get("sample_count")]
            or claim["bytes"] != int(audio.get("sample_count", -1)) * 4
            or claim["sha256"] != audio.get("content_sha256")
            or audio.get("encoding") != "float32_le_mono"
        ):
            raise QualityPacketError(
                "quality model input claim does not match packet audio evidence"
            )
        basename = Path(str(claim["path"])).name
        if basename in expected_names:
            raise QualityPacketError(
                "quality model input inventory contains a duplicate path"
            )
        expected_names.add(basename)
        path = root / basename
        raw = _read_small_regular_file(
            path,
            label="quality model input",
            maximum_bytes=QUALITY_MODEL_INPUT_MAXIMUM_BYTES,
        )
        actual_sha256 = _sha256_bytes(raw)
        actual_bytes = len(raw)
        if (
            actual_sha256 != claim["sha256"]
            or actual_bytes != claim["bytes"]
        ):
            raise QualityPacketError(
                f"quality model input content mismatch for {claim['path']}"
            )
        _validate_quality_model_input_signal(
            raw,
            label=f"quality model input {claim['path']}",
        )
        claims.append(claim)
        previous_sequence = sequence
    if not claims:
        raise QualityPacketError("quality model input inventory is empty")
    try:
        actual_entries = list(root.iterdir())
    except OSError as exc:
        raise QualityPacketError(
            f"quality model input directory cannot be enumerated: {exc}"
        ) from exc
    actual_names: set[str] = set()
    for entry in actual_entries:
        try:
            entry_metadata = entry.lstat()
        except OSError as exc:
            raise QualityPacketError(
                f"quality model input entry is unavailable: {exc}"
            ) from exc
        if stat.S_ISLNK(entry_metadata.st_mode) or not stat.S_ISREG(
            entry_metadata.st_mode
        ):
            raise QualityPacketError(
                "quality model input inventory contains a symlink/non-file entry"
            )
        if entry.name in actual_names:
            raise QualityPacketError(
                "quality model input inventory contains duplicate entries"
            )
        actual_names.add(entry.name)
    if actual_names != expected_names:
        raise QualityPacketError(
            "quality model input inventory has missing or extra files"
        )
    inventory_sha256 = _sha256_bytes(_canonical_json_bytes(claims))
    return {
        "schema_version": QUALITY_MODEL_INPUT_INVENTORY_SCHEMA_VERSION,
        "status": "PASS",
        "directory": QUALITY_MODEL_INPUT_DIRNAME,
        "hash_contract": QUALITY_MODEL_INPUT_INVENTORY_HASH_CONTRACT,
        "file_count": len(claims),
        "total_bytes": sum(int(claim["bytes"]) for claim in claims),
        "files": claims,
        "inventory_sha256": inventory_sha256,
    }


def _force_csv_number(value: object, label: str) -> float:
    if not isinstance(value, str) or not value:
        raise QualityPacketError(f"{label} is missing")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise QualityPacketError(f"{label} is not numeric") from exc
    if not math.isfinite(parsed):
        raise QualityPacketError(f"{label} is not finite")
    return parsed


def _normalize_force_row(
    value: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(QUALITY_FORCE_CSV_FIELDS):
        raise QualityPacketError(f"{label} has an invalid exact seven-field schema")
    raw = value.get("raw")
    if type(raw) is not int or raw < 0:
        raise QualityPacketError(f"{label} raw ADC is invalid")
    status = value.get("status")
    if status not in _FORCE_SAMPLE_STATUSES:
        raise QualityPacketError(f"{label} status is invalid")
    normalized: dict[str, object] = {
        "host_t_s": None,
        "raw": raw,
        "hall_v": None,
        "displacement_mm": None,
        "force_n": None,
        "status": status,
        "device_t_ms": None,
    }
    for field in (
        "host_t_s",
        "hall_v",
        "displacement_mm",
        "force_n",
        "device_t_ms",
    ):
        try:
            normalized[field] = _finite(value.get(field), f"{label} {field}")
        except ValueError as exc:
            raise QualityPacketError(str(exc)) from exc
    return normalized


def _read_stable_force_csv(
    force_csv_path: str | Path,
) -> tuple[list[tuple[int, dict[str, object]]], str, int]:
    """Stream, hash and parse the complete force.csv through one stable fd."""

    path = Path(force_csv_path)
    if path.name != QUALITY_FORCE_CSV_BASENAME:
        raise QualityPacketError("quality force input path must be exactly force.csv")
    before = _require_regular_path(path, "force.csv")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise QualityPacketError(f"force.csv cannot be opened: {exc}") from exc
    digest = hashlib.sha256()
    total = 0
    rows: list[tuple[int, dict[str, object]]] = []
    previous_host: float | None = None
    previous_device: float | None = None
    host_times: set[float] = set()
    device_times: set[float] = set()
    try:
        with os.fdopen(fd, "rb", closefd=False) as binary:
            def decoded_lines():  # noqa: ANN202 - csv iterable boundary
                nonlocal total
                while True:
                    raw_line = binary.readline(1024 * 1024 + 1)
                    if not raw_line:
                        break
                    if len(raw_line) > 1024 * 1024:
                        raise QualityPacketError("force.csv contains an oversized row")
                    if not raw_line.endswith(b"\n"):
                        raise QualityPacketError("force.csv contains an incomplete row")
                    digest.update(raw_line)
                    total += len(raw_line)
                    try:
                        yield raw_line.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise QualityPacketError("force.csv is not UTF-8") from exc

            reader = csv.DictReader(decoded_lines())
            if tuple(reader.fieldnames or ()) != QUALITY_FORCE_CSV_FIELDS:
                raise QualityPacketError("force.csv has an invalid exact header")
            for row_number, row in enumerate(reader, start=2):
                if set(row) != set(QUALITY_FORCE_CSV_FIELDS) or None in row:
                    raise QualityPacketError(
                        f"force.csv row {row_number} has invalid columns"
                    )
                raw_text = row["raw"]
                try:
                    raw_adc = int(raw_text)
                except (TypeError, ValueError) as exc:
                    raise QualityPacketError(
                        f"force.csv row {row_number} raw ADC is invalid"
                    ) from exc
                if raw_adc < 0 or str(raw_adc) != raw_text:
                    raise QualityPacketError(
                        f"force.csv row {row_number} raw ADC is invalid"
                    )
                status = row["status"]
                if status not in _FORCE_SAMPLE_STATUSES:
                    raise QualityPacketError(
                        f"force.csv row {row_number} status is invalid"
                    )
                normalized = {
                    "host_t_s": _force_csv_number(
                        row["host_t_s"], f"force.csv row {row_number} host_t_s",
                    ),
                    "raw": raw_adc,
                    "hall_v": _force_csv_number(
                        row["hall_v"], f"force.csv row {row_number} hall_v",
                    ),
                    "displacement_mm": _force_csv_number(
                        row["displacement_mm"],
                        f"force.csv row {row_number} displacement_mm",
                    ),
                    "force_n": _force_csv_number(
                        row["force_n"], f"force.csv row {row_number} force_n",
                    ),
                    "status": status,
                    "device_t_ms": _force_csv_number(
                        row["device_t_ms"],
                        f"force.csv row {row_number} device_t_ms",
                    ),
                }
                host_t = float(normalized["host_t_s"])
                device_t = float(normalized["device_t_ms"])
                if host_t in host_times or device_t in device_times:
                    raise QualityPacketError(
                        "force.csv contains duplicate/ambiguous host or device timestamp"
                    )
                if (
                    previous_host is not None
                    and (host_t <= previous_host or device_t <= previous_device)
                ):
                    raise QualityPacketError(
                        "force.csv host/device timestamp regressed or repeated"
                    )
                host_times.add(host_t)
                device_times.add(device_t)
                previous_host = host_t
                previous_device = device_t
                rows.append((row_number, normalized))
        after = os.fstat(fd)
        try:
            current = path.lstat()
        except OSError as exc:
            raise QualityPacketError("force.csv changed while reading") from exc
    finally:
        os.close(fd)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        not rows
        or total != before.st_size
        or before_identity != after_identity
        or (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        != before_identity
        or stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
    ):
        if not rows:
            raise QualityPacketError("force.csv has no data rows")
        raise QualityPacketError("force.csv changed while reading")
    return rows, digest.hexdigest(), total


def verify_quality_force_csv_binding(
    packet_log: Iterable[Mapping[str, object]],
    force_csv_path: str | Path,
) -> dict[str, object]:
    """Bind every production Q packet to one exact, unique force.csv row."""

    if isinstance(packet_log, (str, bytes, bytearray, Mapping)):
        raise QualityPacketError("quality packet log must be an iterable of packets")
    try:
        packets = [verify_quality_inference_packet(packet) for packet in packet_log]
    except TypeError as exc:
        raise QualityPacketError("quality packet log is not iterable") from exc
    if not packets:
        raise QualityPacketError("quality force input identity is empty")
    csv_rows, csv_sha256, csv_bytes = _read_stable_force_csv(force_csv_path)
    by_host = {float(row["host_t_s"]): (number, row) for number, row in csv_rows}
    matched_rows: list[dict[str, object]] = []
    matches: list[dict[str, object]] = []
    matched_numbers: set[int] = set()
    previous_number: int | None = None
    previous_chain: str | None = None
    previous_sequence: int | None = None
    for index, packet in enumerate(packets):
        if (
            packet.get("production_eligible") is not True
            or packet.get("scorer_scope") != "pinned_model"
        ):
            raise QualityPacketError(
                "diagnostic proxy cannot issue a production force input receipt"
            )
        if packet.get("previous_packet_chain_sha256") != previous_chain:
            raise QualityPacketError(
                f"quality packet {index} force binding chain is not continuous"
            )
        sequence = packet.get("inference_sequence")
        if (
            type(sequence) is not int
            or (
                previous_sequence is not None
                and sequence != previous_sequence + 1
            )
        ):
            raise QualityPacketError(
                f"quality packet {index} force binding sequence has a gap/replay"
            )
        force = _normalize_force_row(
            packet["force_sample"],
            label=f"quality packet {index} force evidence",
        )
        host_t = float(force["host_t_s"])
        if host_t < float(csv_rows[0][1]["host_t_s"]) or host_t > float(
            csv_rows[-1][1]["host_t_s"]
        ):
            raise QualityPacketError(
                f"quality packet {index} force timestamp lies outside force.csv"
            )
        selected = by_host.get(host_t)
        if selected is None or selected[1] != force:
            raise QualityPacketError(
                f"quality packet {index} force.csv exact seven-field match is missing"
            )
        row_number, row = selected
        if (
            row_number in matched_numbers
            or (previous_number is not None and row_number <= previous_number)
        ):
            raise QualityPacketError(
                f"quality packet {index} force.csv row is duplicate/replayed"
            )
        matched_rows.append(row)
        matches.append({
            "inference_sequence": sequence,
            "csv_row_number": row_number,
            "host_t_s": row["host_t_s"],
            "device_t_ms": row["device_t_ms"],
            "row_sha256": _sha256_bytes(_canonical_json_bytes(row)),
        })
        matched_numbers.add(row_number)
        previous_number = row_number
        previous_chain = packet["packet_chain_sha256"]
        previous_sequence = sequence
    matched_raw = b"".join(_canonical_json_bytes(row) for row in matched_rows)
    return {
        "schema_version": QUALITY_FORCE_INPUT_IDENTITY_SCHEMA_VERSION,
        "status": "PASS",
        "path": QUALITY_FORCE_CSV_BASENAME,
        "fields": list(QUALITY_FORCE_CSV_FIELDS),
        "match_contract": QUALITY_FORCE_CSV_MATCH_CONTRACT,
        "hash_contract": QUALITY_FORCE_INPUT_HASH_CONTRACT,
        "force_csv_bytes": csv_bytes,
        "force_csv_sha256": csv_sha256,
        "force_csv_row_count": len(csv_rows),
        "sample_count": len(matched_rows),
        "first_host_t_s": matched_rows[0]["host_t_s"],
        "last_host_t_s": matched_rows[-1]["host_t_s"],
        "first_device_t_ms": matched_rows[0]["device_t_ms"],
        "last_device_t_ms": matched_rows[-1]["device_t_ms"],
        "matches": matches,
        "canonical_jsonl_sha256": _sha256_bytes(matched_raw),
    }


class QualityPacketLogWriter:
    """Append-only, fsynced packet JSONL staging with fail-closed publication."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        try:
            metadata = self.session_dir.lstat()
        except OSError as exc:
            raise QualityPacketError(f"quality session directory is unavailable: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise QualityPacketError(
                "quality session directory must be a real non-symlink directory"
            )
        self.staging_path = self.session_dir / QUALITY_PACKET_LOG_STAGING_BASENAME
        self.final_path = self.session_dir / QUALITY_PACKET_LOG_BASENAME
        self.receipt_path = self.session_dir / QUALITY_PACKET_LOG_RECEIPT_BASENAME
        self.input_staging_path = (
            self.session_dir / QUALITY_MODEL_INPUT_STAGING_DIRNAME
        )
        self.input_final_path = self.session_dir / QUALITY_MODEL_INPUT_DIRNAME
        if (
            self.final_path.exists()
            or self.final_path.is_symlink()
            or self.receipt_path.exists()
            or self.receipt_path.is_symlink()
            or self.staging_path.exists()
            or self.staging_path.is_symlink()
            or self.input_staging_path.exists()
            or self.input_staging_path.is_symlink()
            or self.input_final_path.exists()
            or self.input_final_path.is_symlink()
        ):
            raise QualityPacketError("quality packet artifacts already exist")
        try:
            os.mkdir(self.input_staging_path, 0o750)
            _fsync_directory(self.session_dir)
        except OSError as exc:
            raise QualityPacketError(
                f"quality model input staging cannot be created: {exc}"
            ) from exc
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            self._fd = os.open(str(self.staging_path), flags, 0o640)
        except OSError as exc:
            try:
                self.input_staging_path.rmdir()
                _fsync_directory(self.session_dir)
            except OSError:
                pass
            raise QualityPacketError(f"quality packet staging cannot be created: {exc}") from exc
        self._closed = False
        self._published = False
        self._count = 0
        self._last_chain: str | None = None
        self._last_sequence: int | None = None
        self._last_inference_t: float | None = None
        self._audio_hashes: set[str] = set()
        self._model: dict[str, object] | None = None
        self._input_claims: dict[int, dict[str, object]] = {}

    def persist_model_input(
        self,
        *,
        inference_sequence: int,
        raw: bytes,
    ) -> dict[str, object]:
        """Durably persist the exact immutable float32-le bytes sent to Q."""

        if self._closed or self._published:
            raise QualityPacketError("quality packet writer is closed")
        if type(inference_sequence) is not int or inference_sequence < 0:
            raise QualityPacketError("quality model input sequence is invalid")
        if not isinstance(raw, bytes) or not raw or len(raw) % 4 != 0:
            raise QualityPacketError(
                "quality model input must be non-empty immutable float32 bytes"
            )
        _validate_quality_model_input_signal(raw, label="quality model input")
        if inference_sequence in self._input_claims:
            raise QualityPacketError("quality model input sequence is duplicate/replayed")
        if (
            self._last_sequence is not None
            and inference_sequence != self._last_sequence + 1
        ):
            raise QualityPacketError("quality model input sequence has a gap")
        basename = f"input_{inference_sequence:020d}.f32le"
        staged_path = self.input_staging_path / basename
        _write_exclusive_regular_file(
            staged_path,
            raw,
            label="quality model input",
        )
        actual_sha256, actual_bytes = _sha256_regular_file(
            staged_path,
            label="quality model input",
        )
        if actual_bytes != len(raw) or actual_sha256 != _sha256_bytes(raw):
            raise QualityPacketError("quality model input changed after persistence")
        claim = {
            "schema_version": QUALITY_MODEL_INPUT_SCHEMA_VERSION,
            "path": f"{QUALITY_MODEL_INPUT_DIRNAME}/{basename}",
            "inference_sequence": inference_sequence,
            "dtype": "<f4",
            "shape": [len(raw) // 4],
            "bytes": len(raw),
            "sha256": actual_sha256,
        }
        self._input_claims[inference_sequence] = claim
        return _json_clone(claim)

    def append(self, packet: Mapping[str, object]) -> None:
        if self._closed or self._published:
            raise QualityPacketError("quality packet writer is closed")
        candidate = verify_quality_inference_packet(packet)
        if (
            candidate.get("production_eligible") is not True
            or candidate.get("scorer_scope") != "pinned_model"
        ):
            raise QualityPacketError("quality packet writer accepts production packets only")
        if candidate.get("previous_packet_chain_sha256") != self._last_chain:
            raise QualityPacketError("quality packet writer chain is not continuous")
        sequence = candidate.get("inference_sequence")
        if type(sequence) is not int or sequence < 0:
            raise QualityPacketError("quality packet writer sequence is invalid")
        if self._last_sequence is not None and sequence != self._last_sequence + 1:
            raise QualityPacketError("quality packet writer sequence is not continuous")
        try:
            inference_t = _finite(
                candidate.get("inference_monotonic_s"),
                "quality packet writer inference timestamp",
            )
            audio_sha256 = _require_sha256(
                candidate["audio_window"].get("content_sha256"),
                "quality packet writer audio SHA-256",
            )
            input_claim = _normalize_model_input_claim(
                candidate["audio_window"].get("model_input")
            )
        except (TypeError, ValueError) as exc:
            raise QualityPacketError(str(exc)) from exc
        if self._last_inference_t is not None and inference_t <= self._last_inference_t:
            raise QualityPacketError("quality packet writer timestamp did not advance")
        if audio_sha256 in self._audio_hashes:
            raise QualityPacketError("quality packet writer audio input hash is frozen")
        if (
            input_claim.get("inference_sequence") != sequence
            or input_claim.get("sha256") != audio_sha256
            or input_claim.get("shape")
            != [candidate["audio_window"].get("sample_count")]
            or self._input_claims.get(sequence) != input_claim
        ):
            raise QualityPacketError(
                "quality packet model input does not match its durable ledger claim"
            )
        staged_input = self.input_staging_path / Path(
            str(input_claim["path"])
        ).name
        input_sha256, input_bytes = _sha256_regular_file(
            staged_input,
            label="quality model input",
        )
        if (
            input_sha256 != input_claim["sha256"]
            or input_bytes != input_claim["bytes"]
        ):
            raise QualityPacketError(
                "quality packet model input content does not match its claim"
            )
        model = candidate["model"]
        if self._model is not None and model != self._model:
            raise QualityPacketError("quality packet writer model/policy identity changed")
        raw = _canonical_json_bytes(candidate)
        try:
            _write_all(self._fd, raw)
            os.fsync(self._fd)
        except OSError as exc:
            raise QualityPacketError(f"quality packet log append failed: {exc}") from exc
        self._count += 1
        self._last_chain = candidate["packet_chain_sha256"]
        self._last_sequence = sequence
        self._last_inference_t = inference_t
        self._audio_hashes.add(audio_sha256)
        if self._model is None:
            self._model = _json_clone(model)

    def finalize(
        self,
        receipt: Mapping[str, object],
        *,
        now_monotonic_s: float | None = None,
    ) -> dict[str, object]:
        if self._closed or self._published:
            raise QualityPacketError("quality packet writer is closed")
        try:
            os.fsync(self._fd)
        except OSError as exc:
            raise QualityPacketError(f"quality packet staging fsync failed: {exc}") from exc
        verified_at = (
            time.monotonic()
            if now_monotonic_s is None
            else _finite(now_monotonic_s, "quality log verification timestamp")
        )
        packets = read_quality_packet_log(self.staging_path)
        quality_authenticity = verify_quality_authenticity(
            receipt,
            now_monotonic_s=verified_at,
        )
        chain_verification = verify_quality_packet_chain(
            packets,
            quality_authenticity,
            now_monotonic_s=verified_at,
            model_input_dir=self.input_staging_path,
            force_csv_path=self.session_dir / QUALITY_FORCE_CSV_BASENAME,
        )
        if chain_verification.get("output_nondegeneracy", {}).get("status") != "PASS":
            raise QualityPacketError(
                "quality packet log output is not nondegenerate; publication refused"
            )
        staging_sha256, staging_bytes = _sha256_regular_file(self.staging_path)
        if staging_sha256 != chain_verification.get("packet_log_sha256"):
            raise QualityPacketError("quality packet staging hash does not match verification")
        if len(self._input_claims) != self._count:
            raise QualityPacketError(
                "quality model input ledger count does not match packet count"
            )
        try:
            os.link(
                self.staging_path,
                self.final_path,
                follow_symlinks=False,
            )
            _fsync_directory(self.session_dir)
            self.staging_path.unlink()
            _fsync_directory(self.session_dir)
        except OSError as exc:
            raise QualityPacketError(f"quality packet log publication failed: {exc}") from exc
        final_sha256, final_bytes = _sha256_regular_file(self.final_path)
        if final_sha256 != staging_sha256 or final_bytes != staging_bytes:
            raise QualityPacketError("published quality packet log changed")
        try:
            os.mkdir(self.input_final_path, 0o750)
            _fsync_directory(self.session_dir)
            for claim in chain_verification["model_input_inventory"]["files"]:
                basename = Path(str(claim["path"])).name
                os.link(
                    self.input_staging_path / basename,
                    self.input_final_path / basename,
                    follow_symlinks=False,
                )
            _fsync_directory(self.input_final_path)
            for entry in self.input_staging_path.iterdir():
                entry.unlink()
            self.input_staging_path.rmdir()
            _fsync_directory(self.session_dir)
        except OSError as exc:
            raise QualityPacketError(
                f"quality model input publication failed: {exc}"
            ) from exc
        published_inventory = _verify_quality_model_input_inventory(
            packets,
            self.input_final_path,
        )
        if published_inventory != chain_verification["model_input_inventory"]:
            raise QualityPacketError(
                "published quality model input inventory changed"
            )
        log_receipt = {
            "schema_version": QUALITY_PACKET_LOG_AUTHENTICITY_SCHEMA_VERSION,
            "status": "PASS",
            "path": self.final_path.name,
            "receipt_path": self.receipt_path.name,
            "bytes": final_bytes,
            "sha256": final_sha256,
            "packet_count": self._count,
            "last_packet_chain_sha256": self._last_chain,
            "verified_at_monotonic_s": verified_at,
            "quality_authenticity": quality_authenticity,
            "chain_verification": chain_verification,
            "model_input_inventory": published_inventory,
            "force_input_identity": chain_verification["force_input_identity"],
            "promotion_eligible": False,
            "promotion_blockers": [QUALITY_PROMOTION_BLOCKER],
        }
        _write_exclusive_regular_file(
            self.receipt_path,
            _canonical_json_bytes(log_receipt),
            label="quality packet log authenticity receipt",
        )
        os.close(self._fd)
        self._closed = True
        self._published = True
        return verify_quality_packet_log(self.session_dir)

    def close(self) -> None:
        if not self._closed:
            os.close(self._fd)
            self._closed = True


def verify_quality_packet_log(session_dir: str | Path) -> dict[str, object]:
    """Independently verify the persisted receipt and every canonical Q packet."""

    session = Path(session_dir)
    try:
        session_metadata = session.lstat()
    except OSError as exc:
        raise QualityPacketError(
            f"quality session directory is unavailable: {exc}"
        ) from exc
    if stat.S_ISLNK(session_metadata.st_mode) or not stat.S_ISDIR(
        session_metadata.st_mode
    ):
        raise QualityPacketError(
            "quality session directory must be a real non-symlink directory"
        )
    staging_path = session / QUALITY_PACKET_LOG_STAGING_BASENAME
    input_staging_path = session / QUALITY_MODEL_INPUT_STAGING_DIRNAME
    if (
        staging_path.exists()
        or staging_path.is_symlink()
        or input_staging_path.exists()
        or input_staging_path.is_symlink()
    ):
        raise QualityPacketError(
            "quality packet/model-input staging still exists beside finalized artifacts"
        )
    receipt_path = session / QUALITY_PACKET_LOG_RECEIPT_BASENAME
    receipt_raw = _read_small_regular_file(
        receipt_path,
        label="quality packet log authenticity receipt",
    )
    try:
        candidate = json.loads(receipt_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualityPacketError(
            "quality packet log authenticity receipt is invalid JSON"
        ) from exc
    if not isinstance(candidate, dict):
        raise QualityPacketError(
            "quality packet log authenticity receipt must be an object"
        )
    if receipt_raw != _canonical_json_bytes(candidate):
        raise QualityPacketError(
            "quality packet log authenticity receipt is not canonical JSON"
        )
    if set(candidate) != _QUALITY_PACKET_LOG_RECEIPT_FIELDS:
        raise QualityPacketError(
            "quality packet log authenticity receipt has an invalid exact schema"
        )
    if (
        candidate.get("schema_version")
        != QUALITY_PACKET_LOG_AUTHENTICITY_SCHEMA_VERSION
        or candidate.get("status") != "PASS"
        or candidate.get("path") != QUALITY_PACKET_LOG_BASENAME
        or candidate.get("receipt_path") != QUALITY_PACKET_LOG_RECEIPT_BASENAME
        or candidate.get("promotion_eligible") is not False
        or candidate.get("promotion_blockers") != [QUALITY_PROMOTION_BLOCKER]
    ):
        raise QualityPacketError(
            "quality packet log authenticity receipt status/scope is invalid"
        )
    packet_count = candidate.get("packet_count")
    byte_count = candidate.get("bytes")
    if (
        type(packet_count) is not int
        or packet_count <= 0
        or type(byte_count) is not int
        or byte_count <= 0
    ):
        raise QualityPacketError(
            "quality packet log authenticity receipt count/size is invalid"
        )
    try:
        claimed_sha256 = _require_sha256(
            candidate.get("sha256"),
            "quality packet log SHA-256",
        )
        claimed_last_chain = _require_sha256(
            candidate.get("last_packet_chain_sha256"),
            "quality packet log last chain SHA-256",
        )
        verified_at = _finite(
            candidate.get("verified_at_monotonic_s"),
            "quality packet log verification timestamp",
        )
    except ValueError as exc:
        raise QualityPacketError(str(exc)) from exc
    raw_authenticity = candidate.get("quality_authenticity")
    if not isinstance(raw_authenticity, Mapping):
        raise QualityPacketError(
            "quality packet log authenticity receipt lacks Q authenticity"
        )
    quality_authenticity = verify_quality_authenticity(
        raw_authenticity,
        now_monotonic_s=verified_at,
    )
    log_path = session / QUALITY_PACKET_LOG_BASENAME
    actual_sha256, actual_bytes = _sha256_regular_file(log_path)
    if actual_sha256 != claimed_sha256 or actual_bytes != byte_count:
        raise QualityPacketError(
            "quality packet log content identity does not match its receipt"
        )
    packets = read_quality_packet_log(log_path)
    chain_verification = verify_quality_packet_chain(
        packets,
        quality_authenticity,
        now_monotonic_s=verified_at,
        model_input_dir=session / QUALITY_MODEL_INPUT_DIRNAME,
        force_csv_path=session / QUALITY_FORCE_CSV_BASENAME,
    )
    if (
        chain_verification.get("status") != "PASS"
        or chain_verification.get("chain_integrity") != "PASS"
        or chain_verification.get("training_signal_eligible") is not True
        or chain_verification.get("output_nondegeneracy", {}).get("status")
        != "PASS"
        or candidate.get("chain_verification") != chain_verification
        or candidate.get("model_input_inventory")
        != chain_verification.get("model_input_inventory")
        or candidate.get("force_input_identity")
        != chain_verification.get("force_input_identity")
        or packet_count != len(packets)
        or packet_count != chain_verification.get("packet_count")
        or claimed_sha256 != chain_verification.get("packet_log_sha256")
        or claimed_last_chain
        != chain_verification.get("last_packet_chain_sha256")
    ):
        raise QualityPacketError(
            "quality packet log full-chain verification does not match its receipt"
        )
    return _json_clone(candidate)


def quality_packet_log_summary(
    log_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Return the small exact boundary object bound by capture finalization."""

    if not isinstance(log_receipt, Mapping):
        raise QualityPacketError("quality packet log receipt must be an object")
    candidate = _json_clone(log_receipt)
    if set(candidate) != _QUALITY_PACKET_LOG_RECEIPT_FIELDS:
        raise QualityPacketError("quality packet log receipt schema is invalid")
    if candidate.get("schema_version") != QUALITY_PACKET_LOG_AUTHENTICITY_SCHEMA_VERSION:
        raise QualityPacketError("quality packet log receipt version is invalid")
    quality_authenticity = candidate.get("quality_authenticity")
    if not isinstance(quality_authenticity, Mapping):
        raise QualityPacketError("quality packet log receipt model identity is missing")
    model = quality_authenticity.get("model")
    if not isinstance(model, Mapping):
        raise QualityPacketError("quality packet log receipt model identity is missing")
    return {
        "path": QUALITY_PACKET_LOG_BASENAME,
        "receipt_path": QUALITY_PACKET_LOG_RECEIPT_BASENAME,
        "status": "PASS",
        "bytes": candidate.get("bytes"),
        "sha256": candidate.get("sha256"),
        "packet_count": candidate.get("packet_count"),
        "last_packet_chain_sha256": candidate.get(
            "last_packet_chain_sha256"
        ),
        "model_release_id": model.get("release_id"),
        "model_sha256": model.get("model_sha256"),
        "policy_sha256": quality_authenticity.get("policy_sha256"),
    }


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite JSON number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_content_addressed_release_id(
    value: object,
    sha256: str,
    label: str,
) -> str:
    suffix = f"-{sha256[:20]}"
    if (
        not isinstance(value, str)
        or value != value.strip()
        or len(value) <= len(suffix)
        or not value.endswith(suffix)
    ):
        raise ValueError(
            f"{label} must be content-addressed with suffix {suffix}"
        )
    return value


def _normalize_model_input_claim(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _QUALITY_MODEL_INPUT_FIELDS:
        raise ValueError("quality model input claim has an invalid exact schema")
    sequence = value.get("inference_sequence")
    shape = value.get("shape")
    byte_count = value.get("bytes")
    if type(sequence) is not int or sequence < 0:
        raise ValueError("quality model input sequence is invalid")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or type(shape[0]) is not int
        or shape[0] <= 0
    ):
        raise ValueError("quality model input shape must be one positive dimension")
    if type(byte_count) is not int or byte_count != shape[0] * 4:
        raise ValueError("quality model input byte count does not match float32 shape")
    expected_path = (
        f"{QUALITY_MODEL_INPUT_DIRNAME}/input_{sequence:020d}.f32le"
    )
    if (
        value.get("schema_version") != QUALITY_MODEL_INPUT_SCHEMA_VERSION
        or value.get("path") != expected_path
        or value.get("dtype") != "<f4"
    ):
        raise ValueError("quality model input identity/path/dtype is invalid")
    sha256 = _require_sha256(
        value.get("sha256"), "quality model input SHA-256"
    )
    return {
        "schema_version": QUALITY_MODEL_INPUT_SCHEMA_VERSION,
        "path": expected_path,
        "inference_sequence": sequence,
        "dtype": "<f4",
        "shape": [shape[0]],
        "bytes": byte_count,
        "sha256": sha256,
    }


def _normalize_raw_source_claim(
    value: object, *, window_sample_rate: object = None,
) -> dict[str, object]:
    """Validate a Q window's provenance into the session's physical audio.

    Requires an exact PCM16 sample range ``[start, end)`` on a named channel of a
    source at ``source_sample_rate``, plus a ``resampler_identity`` whose input
    rate equals the source rate.  When ``window_sample_rate`` is supplied, the
    resampler's output rate must equal it — so the persisted f32 window's rate is
    provably the source audio resampled by the named path, not an unrelated
    buffer.
    """

    if not isinstance(value, Mapping):
        raise ValueError("quality audio raw-source claim has an invalid exact schema")
    # Accept both the raw builder input (core fields only) and the stored form
    # (which additionally carries schema_version); validate the version when
    # present and always return the canonical stored form.
    if "schema_version" in value:
        if value["schema_version"] != QUALITY_AUDIO_RAW_SOURCE_SCHEMA_VERSION:
            raise ValueError("quality audio raw-source schema version is invalid")
        core = {key: value[key] for key in value if key != "schema_version"}
    else:
        core = dict(value)
    if set(core) != _QUALITY_RAW_SOURCE_FIELDS:
        raise ValueError("quality audio raw-source claim has an invalid exact schema")
    value = core
    channel = value.get("source_channel")
    start = value.get("source_sample_start")
    end = value.get("source_sample_end")
    source_rate = value.get("source_sample_rate")
    identity = value.get("resampler_identity")
    if type(channel) is not int or channel < 0:
        raise ValueError("quality raw-source channel is invalid")
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
    ):
        raise ValueError("quality raw-source sample range is invalid")
    if type(source_rate) is not int or source_rate <= 0:
        raise ValueError("quality raw-source sample rate is invalid")
    if (
        not isinstance(identity, Mapping)
        or set(identity) != _QUALITY_RESAMPLER_IDENTITY_FIELDS
    ):
        raise ValueError("quality resampler identity has an invalid exact schema")
    algorithm = identity.get("algorithm")
    input_rate = identity.get("input_sample_rate")
    output_rate = identity.get("output_sample_rate")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("quality resampler algorithm is invalid")
    if type(input_rate) is not int or input_rate <= 0:
        raise ValueError("quality resampler input sample rate is invalid")
    if type(output_rate) is not int or output_rate <= 0:
        raise ValueError("quality resampler output sample rate is invalid")
    if input_rate != source_rate:
        raise ValueError(
            "quality resampler input rate must equal the source sample rate"
        )
    if window_sample_rate is not None and output_rate != window_sample_rate:
        raise ValueError(
            "quality resampler output rate must equal the window sample rate"
        )
    return {
        "schema_version": QUALITY_AUDIO_RAW_SOURCE_SCHEMA_VERSION,
        "source_channel": channel,
        "source_sample_start": start,
        "source_sample_end": end,
        "source_sample_rate": source_rate,
        "resampler_identity": {
            "algorithm": algorithm,
            "input_sample_rate": input_rate,
            "output_sample_rate": output_rate,
        },
    }


def _normalize_output_contract(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _OUTPUT_CONTRACT_FIELDS:
        raise ValueError("quality output_contract has an invalid exact schema")
    model_role = value.get("model_role")
    semantics = value.get("output_semantics")
    positive_class = value.get("positive_class")
    transform = value.get("transform_to_p_usable")
    output_range = value.get("output_range")
    calibration = value.get("calibration")
    if not isinstance(output_range, (list, tuple)) or len(output_range) != 2:
        raise ValueError("quality output_range must contain two values")
    output_lo = _finite(output_range[0], "quality output_range lower bound")
    output_hi = _finite(output_range[1], "quality output_range upper bound")
    if [output_lo, output_hi] != [0.0, 1.0]:
        raise ValueError("quality output_range must be exactly [0.0, 1.0]")
    if not isinstance(calibration, Mapping) or set(calibration) != _CALIBRATION_FIELDS:
        raise ValueError("quality calibration has an invalid exact schema")
    method = calibration.get("method")
    release_id = calibration.get("release_id")
    calibration_sha256 = calibration.get("sha256")
    if semantics == "class_probability":
        if model_role not in {"clinical_usability_q", "signal_artifact_gate"}:
            raise ValueError("quality production model_role is invalid")
        if positive_class not in {"usable", "poor"}:
            raise ValueError("quality positive_class must be usable or poor")
        expected_transform = "identity" if positive_class == "usable" else "one_minus"
        if transform != expected_transform:
            raise ValueError(
                "quality transform_to_p_usable does not match positive_class"
            )
        if not isinstance(method, str) or not method.strip():
            raise ValueError("quality calibration method must be non-empty")
        if not isinstance(release_id, str) or not release_id.strip():
            raise ValueError("quality calibration release_id must be non-empty")
        calibration_sha256 = _require_sha256(
            calibration_sha256, "quality calibration sha256"
        )
        release_id = _require_content_addressed_release_id(
            release_id,
            calibration_sha256,
            "quality calibration release_id",
        )
    elif semantics == "diagnostic_score":
        if (
            model_role != "diagnostic_proxy"
            or positive_class != "not_applicable"
            or transform != "identity_diagnostic_only"
            or method != "none"
            or release_id != "diagnostic-none"
            or calibration_sha256 is not None
        ):
            raise ValueError("diagnostic quality output_contract is invalid")
    else:
        raise ValueError("quality output_semantics is invalid")
    return {
        "model_role": model_role,
        "output_semantics": semantics,
        "positive_class": positive_class,
        "calibration": {
            "method": method,
            "release_id": release_id,
            "sha256": calibration_sha256,
        },
        "transform_to_p_usable": transform,
        "output_range": [output_lo, output_hi],
    }


def _normalize_quality_policy(policy: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(policy, Mapping) or set(policy) != _QUALITY_POLICY_FIELDS:
        raise ValueError("quality policy has an invalid exact schema")
    if policy.get("schema_version") != QUALITY_POLICY_SCHEMA_VERSION:
        raise ValueError("quality policy schema_version is invalid")
    policy_id = policy.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError("quality policy_id must be non-empty")
    force_band = policy.get("force_band_n")
    if not isinstance(force_band, (list, tuple)) or len(force_band) != 2:
        raise ValueError("quality policy force_band_n must contain two values")
    force_lo = _finite(force_band[0], "quality force lower bound")
    force_hi = _finite(force_band[1], "quality force upper bound")
    if force_lo >= force_hi:
        raise ValueError("quality force band must be increasing")
    window_s = _finite(policy.get("window_s"), "quality window_s")
    ema = _finite(policy.get("ema"), "quality ema")
    max_audio_age_s = _finite(
        policy.get("max_audio_age_s"), "quality max_audio_age_s"
    )
    max_inference_age_s = _finite(
        policy.get("max_inference_age_s"), "quality max_inference_age_s"
    )
    max_force_age_s = _finite(
        policy.get("max_force_age_s"), "quality max_force_age_s"
    )
    minimum = policy.get("min_packets_for_receipt")
    output_contract = _normalize_output_contract(policy.get("output_contract"))
    if type(minimum) is not int or minimum < 2:
        raise ValueError("quality min_packets_for_receipt must be an integer >= 2")
    if window_s <= 0 or not 0.0 <= ema <= 1.0:
        raise ValueError("quality window_s/ema policy is invalid")
    if max_audio_age_s <= 0 or max_inference_age_s <= 0 or max_force_age_s <= 0:
        raise ValueError("quality freshness policy must be positive")
    return {
        "schema_version": QUALITY_POLICY_SCHEMA_VERSION,
        "policy_id": policy_id,
        "force_band_n": [force_lo, force_hi],
        "window_s": window_s,
        "ema": ema,
        "max_audio_age_s": max_audio_age_s,
        "max_inference_age_s": max_inference_age_s,
        "max_force_age_s": max_force_age_s,
        "min_packets_for_receipt": minimum,
        "output_contract": output_contract,
    }


def diagnostic_quality_policy(
    *, window_s: float, force_band: tuple[float, float], ema: float = 0.5,
) -> dict[str, object]:
    """Explicit non-production policy for the command-line spectral proxy."""

    return _normalize_quality_policy({
        "schema_version": QUALITY_POLICY_SCHEMA_VERSION,
        "policy_id": "diagnostic-band-energy-proxy-v1",
        "force_band_n": list(force_band),
        "window_s": float(window_s),
        "ema": float(ema),
        "max_audio_age_s": 1.0,
        "max_inference_age_s": 2.0,
        "max_force_age_s": 1.0,
        "min_packets_for_receipt": 2,
        "output_contract": {
            "model_role": "diagnostic_proxy",
            "output_semantics": "diagnostic_score",
            "positive_class": "not_applicable",
            "calibration": {
                "method": "none",
                "release_id": "diagnostic-none",
                "sha256": None,
            },
            "transform_to_p_usable": "identity_diagnostic_only",
            "output_range": [0.0, 1.0],
        },
    })


def safe_force_value(sample) -> float | None:  # noqa: ANN001
    """Never open the contact gate from absent, stale, or over-limit force."""
    if sample is None or getattr(sample, "status", None) not in ("OK", "WARN"):
        return None
    return float(sample.force_n)


def band_energy_proxy(samples: np.ndarray, sr: int = 16000, lo: float = 90.0, hi: float = 1100.0) -> float:
    """Diagnostic-only spectral proxy; never a production or default Q scorer."""
    if samples.size < sr // 4:
        return 0.0
    x = samples - samples.mean()
    f = np.fft.rfftfreq(x.size, 1.0 / sr)
    p = np.abs(np.fft.rfft(x)) ** 2
    tot = p[(f >= 20) & (f <= sr / 2)].sum()
    return float(p[(f >= lo) & (f <= hi)].sum() / tot) if tot > 0 else 0.0


band_energy_proxy._wam_quality_scope = "diagnostic_only"  # type: ignore[attr-defined]


class StreamingQuality:
    """Pinned per-hop Q inference with packet-based liveness evidence.

    A missing production scorer/model/policy leaves the object explicitly
    ``NOT_READY``.  No fallback value is emitted.  Diagnostic proxy use must be
    selected with ``mode="diagnostic"`` and can never issue a production
    authenticity receipt.
    """

    def __init__(
        self,
        scorer=None,  # noqa: ANN001 - pluggable inference callable
        *,
        mode: str = "production",
        model_release_id: str | None = None,
        model_sha256: str | None = None,
        policy: Mapping[str, object] | None = None,
        packet_log_writer: QualityPacketLogWriter | None = None,
    ) -> None:
        if mode not in {"production", "diagnostic"}:
            raise ValueError("quality mode must be production or diagnostic")
        if (
            mode == "production"
            and getattr(scorer, "_wam_quality_scope", None) == "diagnostic_only"
        ):
            raise ValueError(
                "band_energy_proxy is diagnostic-only and forbidden in production"
            )
        self.mode = mode
        self.scorer = scorer
        self.model_release_id = model_release_id
        self.model_sha256 = model_sha256
        if packet_log_writer is not None and not isinstance(
            packet_log_writer, QualityPacketLogWriter
        ):
            raise TypeError("packet_log_writer must be a QualityPacketLogWriter")
        if mode == "diagnostic" and packet_log_writer is not None:
            raise ValueError(
                "diagnostic proxy cannot own a production quality input ledger"
            )
        self.packet_log_writer = packet_log_writer
        self.policy = (
            None if policy is None else _normalize_quality_policy(policy)
        )
        self.policy_sha256 = (
            None
            if self.policy is None
            else _sha256_bytes(_canonical_json_bytes(self.policy))
        )
        self.output_contract = (
            None if self.policy is None else self.policy["output_contract"]
        )
        self._blockers: list[str] = []
        if scorer is None:
            self._blockers.append(
                "pinned production Q scorer"
                if mode == "production" else "explicit diagnostic Q scorer"
            )
        elif not callable(scorer):
            self._blockers.append(
                "callable pinned production Q scorer"
                if mode == "production" else "callable diagnostic Q scorer"
            )
        if self.policy is None:
            self._blockers.append("pinned production Q policy")
        elif (
            mode == "production"
            and self.output_contract["output_semantics"] != "class_probability"
        ):
            raise ValueError("production Q requires class_probability output semantics")
        elif (
            mode == "diagnostic"
            and self.output_contract["output_semantics"] != "diagnostic_score"
        ):
            raise ValueError("diagnostic Q requires diagnostic_score output semantics")
        if mode == "production":
            if packet_log_writer is None:
                self._blockers.append(
                    "durable production Q model-input and packet ledger"
                )
            if not isinstance(model_release_id, str) or not model_release_id.strip():
                self._blockers.append("pinned production Q model release_id")
            if model_sha256 is None:
                self._blockers.append("pinned production Q model SHA-256")
            else:
                self.model_sha256 = _require_sha256(
                    model_sha256, "quality model_sha256"
                )
                if isinstance(model_release_id, str) and model_release_id.strip():
                    self.model_release_id = _require_content_addressed_release_id(
                        model_release_id,
                        self.model_sha256,
                        "quality model release_id",
                    )
        elif model_sha256 is not None:
            self.model_sha256 = _require_sha256(
                model_sha256, "diagnostic quality model_sha256"
            )
        if self.policy is None:
            self.force_lo = self.force_hi = self.window_s = self.ema = None
        else:
            self.force_lo, self.force_hi = self.policy["force_band_n"]
            self.window_s = float(self.policy["window_s"])
            self.ema = float(self.policy["ema"])
        self.p_smoothed = 0.0
        self._packet_count = 0
        self._first_sequence: int | None = None
        self._last_sequence: int | None = None
        self._first_inference_t: float | None = None
        self._last_inference_t: float | None = None
        self._last_audio_sha256: str | None = None
        self._last_packet_sha256: str | None = None
        self._packet_chain_sha256: str | None = None
        self._last_packet: dict[str, object] | None = None

    def readiness(self) -> dict[str, object]:
        ready = not self._blockers
        return {
            "status": "READY" if ready else "NOT_READY",
            "ready": ready,
            "production_ready": ready and self.mode == "production",
            "mode": self.mode,
            "blockers": list(self._blockers),
            "model_release_id": self.model_release_id,
            "model_sha256": self.model_sha256,
            "policy_sha256": self.policy_sha256,
            "packet_log_configured": self.packet_log_writer is not None,
            "promotion_ready": False,
        }

    def _require_ready(self) -> None:
        if self._blockers:
            raise QualityNotReadyError(
                "production Q is not ready: " + "; ".join(self._blockers)
            )

    def _audio_window_evidence(
        self, audio_window,  # noqa: ANN001 - AudioWindow-compatible boundary
    ) -> tuple[np.ndarray, bytes, int, dict[str, object]]:
        try:
            source = np.asarray(audio_window.samples, dtype="<f4").reshape(-1)
            immutable_raw = source.tobytes(order="C")
            samples = np.frombuffer(immutable_raw, dtype="<f4")
            sample_rate = int(audio_window.sr)
            t_start = _finite(audio_window.t_start, "quality audio window Start")
            t_end = _finite(audio_window.t_end, "quality audio window End")
            status = str(audio_window.status)
        except (AttributeError, TypeError, ValueError) as exc:
            raise QualityPacketError("quality audio window is incomplete") from exc
        if sample_rate <= 0 or samples.size <= 0 or status != "OK":
            raise QualityPacketError("quality audio window is not current OK data")
        if t_end <= t_start:
            raise QualityPacketError("quality audio window interval is invalid")
        expected_duration = samples.size / float(sample_rate)
        tolerance = 1.0 / float(sample_rate)
        assert self.window_s is not None
        if (
            abs((t_end - t_start) - expected_duration) > tolerance
            or abs(expected_duration - self.window_s) > tolerance
        ):
            raise QualityPacketError(
                "quality audio window sample count/timestamps do not match policy"
            )
        if samples.flags.writeable:
            raise QualityPacketError("quality model input is not immutable")
        content_sha256 = _sha256_bytes(immutable_raw)
        return samples, immutable_raw, sample_rate, {
            "t_start_monotonic_s": t_start,
            "t_end_monotonic_s": t_end,
            "sample_rate": sample_rate,
            "sample_count": int(samples.size),
            "content_sha256": content_sha256,
            "encoding": "float32_le_mono",
            "status": status,
        }

    def _force_sample_evidence(
        self, force_sample, *, inference_t: float,  # noqa: ANN001
    ) -> dict[str, object]:
        def required(name: str, label: str):  # noqa: ANN202
            try:
                return getattr(force_sample, name)
            except AttributeError as exc:
                raise QualityPacketError(f"quality force {label} is missing") from exc

        try:
            force_t = _finite(
                required("t", "timestamp"), "quality force timestamp",
            )
            status = str(required("status", "status"))
            raw = required("raw", "raw ADC")
            if type(raw) is not int or raw < 0:
                raise ValueError("quality force raw ADC is invalid")
            hall_v = _finite(
                required("hall_v", "Hall voltage"),
                "quality force Hall voltage",
            )
            displacement_mm = _finite(
                required("displacement_mm", "displacement"),
                "quality force displacement",
            )
            force_n = _finite(
                required("force_n", "value"), "quality force value",
            )
            device_t_ms = _finite(
                required("device_t_ms", "device timestamp"),
                "quality force device timestamp",
            )
        except ValueError as exc:
            raise QualityPacketError(str(exc)) from exc
        if status not in _FORCE_SAMPLE_STATUSES:
            raise QualityPacketError("quality force status is invalid")
        assert self.policy is not None
        age = inference_t - force_t
        if age < -1e-9 or age > float(self.policy["max_force_age_s"]) + 1e-12:
            raise QualityPacketError("quality force sample is stale or from the future")
        return {
            "host_t_s": force_t,
            "raw": raw,
            "hall_v": hall_v,
            "displacement_mm": displacement_mm,
            "force_n": force_n,
            "status": status,
            "device_t_ms": device_t_ms,
        }

    def _accept_packet(
        self,
        packet: dict[str, object],
    ) -> None:
        sequence = int(packet["inference_sequence"])
        inference_t = float(packet["inference_monotonic_s"])
        audio_sha256 = str(packet["audio_window"]["content_sha256"])  # type: ignore[index]
        packet["hash_contract"] = QUALITY_PACKET_HASH_CONTRACT
        packet["previous_packet_chain_sha256"] = self._packet_chain_sha256
        packet_sha256, chain_sha256 = _quality_packet_hashes(packet)
        packet["packet_sha256"] = packet_sha256
        packet["packet_chain_sha256"] = chain_sha256
        if self.packet_log_writer is not None:
            self.packet_log_writer.append(packet)
        self._packet_count += 1
        if self._first_sequence is None:
            self._first_sequence = sequence
            self._first_inference_t = inference_t
        self._last_sequence = sequence
        self._last_inference_t = inference_t
        self._last_audio_sha256 = audio_sha256
        self._last_packet_sha256 = packet_sha256
        self._packet_chain_sha256 = chain_sha256
        self._last_packet = _json_clone(packet)

    def _validate_progress(
        self,
        *,
        sequence: int,
        inference_t: float,
        observed_t: float,
        audio_sha256: str,
        audio_end_t: float,
    ) -> None:
        """Reject frozen/stale evidence before invoking the pinned model."""

        assert self.policy is not None
        age = observed_t - inference_t
        if age < -1e-9 or age > float(self.policy["max_inference_age_s"]) + 1e-12:
            raise QualityPacketError("quality inference packet is stale or from the future")
        if self._last_sequence is not None and sequence != self._last_sequence + 1:
            raise QualityPacketError("quality inference sequence did not advance exactly")
        if self._last_inference_t is not None and inference_t <= self._last_inference_t:
            raise QualityPacketError("quality inference timestamp did not advance")
        if self._last_audio_sha256 is not None and audio_sha256 == self._last_audio_sha256:
            raise QualityPacketError("quality audio input hash is frozen")
        audio_age = inference_t - audio_end_t
        if (
            audio_age < -1e-9
            or audio_age > float(self.policy["max_audio_age_s"]) + 1e-12
        ):
            raise QualityPacketError("quality audio window is stale or from the future")

    def update(
        self,
        *,
        audio_window,  # noqa: ANN001 - AudioWindow-compatible boundary
        force_sample,  # noqa: ANN001 - ForceSample-compatible boundary
        inference_sequence: int,
        inference_monotonic_s: float,
        observed_monotonic_s: float | None = None,
    ) -> dict[str, object]:
        self._require_ready()
        if type(inference_sequence) is not int or inference_sequence < 0:
            raise QualityPacketError("quality inference sequence must be a nonnegative integer")
        inference_t = _finite(
            inference_monotonic_s, "quality inference monotonic timestamp"
        )
        observed_t = (
            time.monotonic()
            if observed_monotonic_s is None
            else _finite(observed_monotonic_s, "quality packet observation timestamp")
        )
        samples, immutable_raw, sample_rate, audio_evidence = (
            self._audio_window_evidence(audio_window)
        )
        force_evidence = self._force_sample_evidence(
            force_sample, inference_t=inference_t,
        )
        self._validate_progress(
            sequence=inference_sequence,
            inference_t=inference_t,
            observed_t=observed_t,
            audio_sha256=str(audio_evidence["content_sha256"]),
            audio_end_t=float(audio_evidence["t_end_monotonic_s"]),
        )
        if self.mode == "production":
            if self.packet_log_writer is None:
                raise QualityNotReadyError(
                    "production Q has no durable model-input ledger"
                )
            audio_evidence["model_input"] = (
                self.packet_log_writer.persist_model_input(
                    inference_sequence=inference_sequence,
                    raw=immutable_raw,
                )
            )
            # Bind the window to the session's physical audio: the exact PCM16
            # sample range, channel, and resampler identity travel with the
            # packet so a later real-model replay can prove the window derives
            # from this session's canonical audio.wav, not just that the f32 file
            # and the Q output are self-consistent.
            try:
                audio_evidence["raw_source"] = _normalize_raw_source_claim(
                    getattr(audio_window, "raw_source", None),
                    window_sample_rate=sample_rate,
                )
            except (TypeError, ValueError) as exc:
                raise QualityPacketError(
                    f"production Q raw-source provenance is invalid: {exc}"
                ) from exc
        assert callable(self.scorer)
        before_scorer_sha256 = _sha256_bytes(samples.tobytes(order="C"))
        if before_scorer_sha256 != audio_evidence["content_sha256"]:
            raise QualityPacketError(
                "quality model input changed before scorer invocation"
            )
        try:
            model_output = self.scorer(samples, sample_rate)
        except Exception as exc:  # noqa: BLE001 - model boundary must fail closed
            if _sha256_bytes(samples.tobytes(order="C")) != before_scorer_sha256:
                raise QualityPacketError(
                    "quality model input was mutated by the scorer"
                ) from exc
            raise QualityPacketError("quality scorer output was not produced") from exc
        if _sha256_bytes(samples.tobytes(order="C")) != before_scorer_sha256:
            raise QualityPacketError("quality model input was mutated by the scorer")
        try:
            q_raw = _finite(model_output, "quality scorer output")
        except ValueError as exc:
            raise QualityPacketError(str(exc)) from exc
        assert self.output_contract is not None
        output_lo, output_hi = self.output_contract["output_range"]
        if not float(output_lo) <= q_raw <= float(output_hi):
            raise QualityPacketError(
                "quality scorer output is outside the pinned output_range"
            )
        assert self.force_lo is not None and self.force_hi is not None
        gate = (
            force_evidence["status"] in {"OK", "WARN"}
            and self.force_lo <= float(force_evidence["force_n"]) <= self.force_hi
        )
        if self.mode == "production":
            p_usable_raw = (
                q_raw
                if self.output_contract["transform_to_p_usable"] == "identity"
                else 1.0 - q_raw
            )
            gated = p_usable_raw if gate else 0.0
        else:
            p_usable_raw = None
            gated = q_raw if gate else 0.0
        assert self.ema is not None
        next_smoothed = self.ema * self.p_smoothed + (1.0 - self.ema) * gated
        model = {
            "release_id": (
                self.model_release_id
                if self.mode == "production" else "diagnostic-band-energy-proxy-v1"
            ),
            "model_sha256": self.model_sha256,
            "policy_id": self.policy["policy_id"],  # type: ignore[index]
            "policy_sha256": self.policy_sha256,
            "output_contract": _json_clone(self.output_contract),
        }
        result_evidence: dict[str, object] = {
            "force_gate_open": gate,
            "q_raw": q_raw,
        }
        if self.mode == "production":
            result_evidence.update(
                p_usable_raw=p_usable_raw,
                p_usable=next_smoothed,
            )
        else:
            result_evidence.update(
                diagnostic_score_raw=q_raw,
                diagnostic_score_smoothed=next_smoothed,
            )
        packet: dict[str, object] = {
            "schema_version": QUALITY_PACKET_SCHEMA_VERSION,
            "scorer_scope": (
                "pinned_model" if self.mode == "production" else "diagnostic_only"
            ),
            "production_eligible": self.mode == "production",
            "model": model,
            "inference_sequence": inference_sequence,
            "inference_monotonic_s": inference_t,
            "observed_monotonic_s": observed_t,
            "audio_window": audio_evidence,
            "force_sample": force_evidence,
            "result": result_evidence,
        }
        self._accept_packet(packet)
        self.p_smoothed = next_smoothed
        result: dict[str, object] = {
            "force_n": force_evidence["force_n"],
            "gate": gate,
            "q_raw": q_raw,
            "packet": packet,
        }
        if self.mode == "production":
            result.update(p_usable_raw=p_usable_raw, p_usable=next_smoothed)
        else:
            result.update(
                diagnostic_score_raw=q_raw,
                diagnostic_score_smoothed=next_smoothed,
            )
        return result

    def quality_authenticity(
        self, *, now_monotonic_s: float | None = None,
    ) -> dict[str, object]:
        self._require_ready()
        if self.mode != "production":
            raise QualityNotReadyError(
                "diagnostic-only Q cannot issue a production authenticity receipt"
            )
        assert self.policy is not None
        minimum = int(self.policy["min_packets_for_receipt"])
        if self._packet_count < minimum or self._last_inference_t is None:
            raise QualityPacketError(
                f"quality liveness requires at least {minimum} inference packets"
            )
        now = (
            time.monotonic()
            if now_monotonic_s is None
            else _finite(now_monotonic_s, "quality receipt timestamp")
        )
        age = now - self._last_inference_t
        if age < -1e-9 or age > float(self.policy["max_inference_age_s"]) + 1e-12:
            raise QualityPacketError("quality inference liveness is stale")
        assert self._first_sequence is not None
        assert self._first_inference_t is not None
        assert self._last_sequence is not None
        assert self._last_audio_sha256 is not None
        assert self._last_packet_sha256 is not None
        assert self._packet_chain_sha256 is not None
        assert self._last_packet is not None
        receipt = {
            "schema_version": QUALITY_AUTHENTICITY_SCHEMA_VERSION,
            "policy_version": QUALITY_AUTHENTICITY_POLICY_VERSION,
            "status": "PASS",
            "promotion_eligible": False,
            "promotion_blockers": [QUALITY_PROMOTION_BLOCKER],
            "model": {
                "release_id": self.model_release_id,
                "model_sha256": self.model_sha256,
                "output_contract": _json_clone(self.output_contract),
            },
            "policy": _json_clone(self.policy),
            "policy_sha256": self.policy_sha256,
            "packet_count": self._packet_count,
            "first_inference_sequence": self._first_sequence,
            "last_inference_sequence": self._last_sequence,
            "first_inference_monotonic_s": self._first_inference_t,
            "last_inference_monotonic_s": self._last_inference_t,
            "last_audio_content_sha256": self._last_audio_sha256,
            "last_packet_sha256": self._last_packet_sha256,
            "packet_chain_sha256": self._packet_chain_sha256,
            "last_packet": _json_clone(self._last_packet),
            "checks": {
                "audio_input_progress": "PASS",
                "lossless_model_input_binding": "PASS",
                "force_raw_evidence": "PASS",
                "inference_freshness": "PASS",
                "inference_sequence_progress": "PASS",
                "inference_timestamp_progress": "PASS",
                "model_identity": "PASS",
                "output_semantics": "PASS",
                "packet_hash_chain": "PASS",
            },
        }
        return verify_quality_authenticity(receipt, now_monotonic_s=now)


def build_diagnostic_quality_runtime(
    *, window_s: float, force_band: tuple[float, float], ema: float = 0.5,
) -> StreamingQuality:
    """Build the CLI demo's proxy runtime with an unambiguous diagnostic scope."""

    return StreamingQuality(
        scorer=band_energy_proxy,
        mode="diagnostic",
        policy=diagnostic_quality_policy(
            window_s=window_s,
            force_band=force_band,
            ema=ema,
        ),
    )


def verify_quality_authenticity(
    receipt: Mapping[str, object],
    *,
    now_monotonic_s: float | None = None,
) -> dict[str, object]:
    """Verify a production Q receipt without trusting ``StreamingQuality`` state."""

    if not isinstance(receipt, Mapping):
        raise QualityPacketError("quality authenticity receipt must be an object")
    candidate = _json_clone(receipt)
    if candidate.get("schema_version") != QUALITY_AUTHENTICITY_SCHEMA_VERSION:
        raise QualityPacketError("quality authenticity schema is invalid")
    if candidate.get("policy_version") != QUALITY_AUTHENTICITY_POLICY_VERSION:
        raise QualityPacketError("quality authenticity policy_version is invalid")
    if candidate.get("status") != "PASS":
        raise QualityPacketError("quality authenticity receipt is not PASS")
    if (
        candidate.get("promotion_eligible") is not False
        or candidate.get("promotion_blockers") != [QUALITY_PROMOTION_BLOCKER]
    ):
        raise QualityPacketError(
            "quality authenticity receipt must remain non-promotable without a "
            "finalization-bound packet log"
        )
    try:
        policy = _normalize_quality_policy(candidate.get("policy"))
    except (TypeError, ValueError) as exc:
        raise QualityPacketError(f"quality receipt policy is invalid: {exc}") from exc
    policy_sha256 = _sha256_bytes(_canonical_json_bytes(policy))
    if candidate.get("policy_sha256") != policy_sha256:
        raise QualityPacketError("quality receipt policy SHA-256 does not match")

    raw_packet = candidate.get("last_packet")
    if not isinstance(raw_packet, Mapping):
        raise QualityPacketError("quality receipt last_packet is missing")
    packet = verify_quality_inference_packet(raw_packet)
    if packet.get("production_eligible") is not True:
        raise QualityPacketError("quality receipt packet is not production eligible")
    if packet.get("scorer_scope") != "pinned_model":
        raise QualityPacketError("quality receipt packet scorer scope is invalid")
    packet_model = packet.get("model")
    receipt_model = candidate.get("model")
    if not isinstance(packet_model, Mapping) or not isinstance(receipt_model, Mapping):
        raise QualityPacketError("quality receipt model identity is incomplete")
    expected_output_contract = policy["output_contract"]
    if (
        packet_model.get("release_id") != receipt_model.get("release_id")
        or packet_model.get("model_sha256") != receipt_model.get("model_sha256")
        or packet_model.get("policy_id") != policy["policy_id"]
        or packet_model.get("policy_sha256") != policy_sha256
        or packet_model.get("output_contract") != expected_output_contract
        or receipt_model.get("output_contract") != expected_output_contract
    ):
        raise QualityPacketError("quality receipt model/output identity does not match")
    try:
        _require_sha256(packet_model.get("model_sha256"), "quality model SHA-256")
    except ValueError as exc:
        raise QualityPacketError(str(exc)) from exc
    try:
        release_id = _require_content_addressed_release_id(
            packet_model.get("release_id"),
            _require_sha256(
                packet_model.get("model_sha256"),
                "quality model SHA-256",
            ),
            "quality model release_id",
        )
    except ValueError as exc:
        raise QualityPacketError(str(exc)) from exc

    if (
        candidate.get("last_packet_sha256") != packet.get("packet_sha256")
        or candidate.get("packet_chain_sha256")
        != packet.get("packet_chain_sha256")
    ):
        raise QualityPacketError("quality receipt packet hash binding does not match")
    integer_progress = (
        candidate.get("packet_count"),
        candidate.get("first_inference_sequence"),
        candidate.get("last_inference_sequence"),
        packet.get("inference_sequence"),
    )
    if any(type(value) is not int for value in integer_progress):
        raise QualityPacketError("quality receipt progress evidence is invalid")
    try:
        packet_count = candidate["packet_count"]
        first_sequence = candidate["first_inference_sequence"]
        last_sequence = candidate["last_inference_sequence"]
        packet_sequence = packet["inference_sequence"]
        first_inference_t = _finite(
            candidate.get("first_inference_monotonic_s"),
            "quality first inference timestamp",
        )
        last_inference_t = _finite(
            candidate.get("last_inference_monotonic_s"),
            "quality last inference timestamp",
        )
        packet_inference_t = _finite(
            packet.get("inference_monotonic_s"),
            "quality packet inference timestamp",
        )
    except (TypeError, ValueError) as exc:
        raise QualityPacketError(f"quality receipt progress evidence is invalid: {exc}") from exc
    minimum = int(policy["min_packets_for_receipt"])
    if (
        packet_count < minimum
        or first_sequence < 0
        or last_sequence != first_sequence + packet_count - 1
        or packet_sequence != last_sequence
        or last_inference_t != packet_inference_t
        or first_inference_t > last_inference_t
    ):
        raise QualityPacketError("quality receipt sequence/timestamp progress is invalid")
    now = (
        time.monotonic()
        if now_monotonic_s is None
        else _finite(now_monotonic_s, "quality verifier timestamp")
    )
    inference_age = now - last_inference_t
    if (
        inference_age < -1e-9
        or inference_age > float(policy["max_inference_age_s"]) + 1e-12
    ):
        raise QualityPacketError("quality receipt inference packet is stale")

    audio = packet.get("audio_window")
    force = packet.get("force_sample")
    result = packet.get("result")
    if not isinstance(audio, Mapping) or not isinstance(force, Mapping) or not isinstance(result, Mapping):
        raise QualityPacketError("quality receipt input/result evidence is incomplete")
    try:
        audio_start = _finite(audio.get("t_start_monotonic_s"), "quality audio start")
        audio_end = _finite(audio.get("t_end_monotonic_s"), "quality audio end")
        sample_rate = int(audio.get("sample_rate"))
        sample_count = int(audio.get("sample_count"))
        model_input = _normalize_model_input_claim(audio.get("model_input"))
        force_t = _finite(force.get("host_t_s"), "quality force timestamp")
        raw_adc = force.get("raw")
        if type(raw_adc) is not int or raw_adc < 0:
            raise ValueError("quality force raw ADC is invalid")
        _finite(force.get("hall_v"), "quality force Hall voltage")
        _finite(force.get("displacement_mm"), "quality force displacement")
        force_n = _finite(force.get("force_n"), "quality force value")
        _finite(force.get("device_t_ms"), "quality force device timestamp")
        q_raw = _finite(result.get("q_raw"), "quality scorer output")
        p_usable_raw = _finite(result.get("p_usable_raw"), "quality p_usable_raw")
    except (TypeError, ValueError) as exc:
        raise QualityPacketError(f"quality receipt packet evidence is invalid: {exc}") from exc
    try:
        audio_sha256 = _require_sha256(
            audio.get("content_sha256"), "quality audio content SHA-256"
        )
    except ValueError as exc:
        raise QualityPacketError(str(exc)) from exc
    if candidate.get("last_audio_content_sha256") != audio_sha256:
        raise QualityPacketError("quality receipt audio content binding does not match")
    if (
        model_input["inference_sequence"] != packet_sequence
        or model_input["shape"] != [sample_count]
        or model_input["bytes"] != sample_count * 4
        or model_input["sha256"] != audio_sha256
    ):
        raise QualityPacketError(
            "quality receipt lossless model input binding does not match"
        )
    if sample_rate <= 0 or sample_count <= 0 or audio.get("encoding") != "float32_le_mono":
        raise QualityPacketError("quality receipt audio shape/encoding is invalid")
    tolerance = 1.0 / float(sample_rate)
    if (
        abs((audio_end - audio_start) - sample_count / float(sample_rate)) > tolerance
        or abs(sample_count / float(sample_rate) - float(policy["window_s"])) > tolerance
        or packet_inference_t - audio_end < -1e-9
        or packet_inference_t - audio_end
        > float(policy["max_audio_age_s"]) + 1e-12
    ):
        raise QualityPacketError("quality receipt audio freshness/interval is invalid")
    if (
        packet_inference_t - force_t < -1e-9
        or packet_inference_t - force_t
        > float(policy["max_force_age_s"]) + 1e-12
    ):
        raise QualityPacketError("quality receipt force sample is stale")
    force_status = force.get("status")
    if force_status not in _FORCE_SAMPLE_STATUSES:
        raise QualityPacketError("quality receipt force status is invalid")
    force_lo, force_hi = policy["force_band_n"]
    expected_gate = (
        force_status in {"OK", "WARN"}
        and float(force_lo) <= force_n <= float(force_hi)
    )
    if result.get("force_gate_open") is not expected_gate:
        raise QualityPacketError("quality receipt force gate evidence does not match")
    output_contract = policy["output_contract"]
    output_lo, output_hi = output_contract["output_range"]
    if not float(output_lo) <= q_raw <= float(output_hi):
        raise QualityPacketError("quality receipt scorer output is outside output_range")
    expected_p_usable = (
        q_raw
        if output_contract["transform_to_p_usable"] == "identity"
        else 1.0 - q_raw
    )
    if abs(p_usable_raw - expected_p_usable) > 1e-12:
        raise QualityPacketError("quality receipt output semantic transform does not match")
    expected_checks = {
        "audio_input_progress": "PASS",
        "lossless_model_input_binding": "PASS",
        "force_raw_evidence": "PASS",
        "inference_freshness": "PASS",
        "inference_sequence_progress": "PASS",
        "inference_timestamp_progress": "PASS",
        "model_identity": "PASS",
        "output_semantics": "PASS",
        "packet_hash_chain": "PASS",
    }
    if candidate.get("checks") != expected_checks:
        raise QualityPacketError("quality receipt checks are incomplete")
    return candidate


def verify_quality_packet_chain(
    packet_log: Iterable[Mapping[str, object]],
    receipt: Mapping[str, object],
    *,
    now_monotonic_s: float | None = None,
    model_input_dir: str | Path | None = None,
    force_csv_path: str | Path | None = None,
) -> dict[str, object]:
    """Verify every packet in one in-memory/log iterable against its receipt.

    Passing this verifier is still not promotion evidence: the caller must
    durably persist the canonical JSONL and bind its hash into finalization.
    """

    verified_receipt = verify_quality_authenticity(
        receipt,
        now_monotonic_s=now_monotonic_s,
    )
    if isinstance(packet_log, (str, bytes, bytearray, Mapping)):
        raise QualityPacketError("quality packet log must be an iterable of packets")
    try:
        raw_packets = list(packet_log)
    except TypeError as exc:
        raise QualityPacketError("quality packet log is not iterable") from exc
    if not raw_packets:
        raise QualityPacketError("quality packet log is empty")
    if model_input_dir is None:
        raise QualityPacketError(
            "quality packet log verifier requires the lossless model-input directory"
        )
    if force_csv_path is None:
        raise QualityPacketError(
            "quality packet log verifier requires the exact force.csv source"
        )

    policy = _normalize_quality_policy(verified_receipt["policy"])
    expected_model = verified_receipt["last_packet"]["model"]
    verified_packets: list[dict[str, object]] = []
    audio_hashes: set[str] = set()
    q_raw_values: list[float] = []
    p_usable_raw_values: list[float] = []
    previous_chain: str | None = None
    previous_sequence: int | None = None
    previous_inference_t: float | None = None
    previous_observed_t: float | None = None
    previous_audio_start: float | None = None
    previous_audio_end: float | None = None
    previous_force_t: float | None = None
    previous_force_device_t: float | None = None
    expected_p_usable_smoothed = 0.0

    for index, raw_packet in enumerate(raw_packets):
        packet = verify_quality_inference_packet(raw_packet)
        if (
            packet.get("production_eligible") is not True
            or packet.get("scorer_scope") != "pinned_model"
            or packet.get("model") != expected_model
        ):
            raise QualityPacketError(
                f"quality packet {index} model/policy identity changed"
            )
        if packet.get("previous_packet_chain_sha256") != previous_chain:
            raise QualityPacketError(
                f"quality packet {index} previous chain is not continuous"
            )
        sequence = packet.get("inference_sequence")
        if type(sequence) is not int or sequence < 0:
            raise QualityPacketError(f"quality packet {index} sequence is invalid")
        if previous_sequence is not None and sequence != previous_sequence + 1:
            raise QualityPacketError(
                f"quality packet {index} sequence is not continuous"
            )
        try:
            inference_t = _finite(
                packet.get("inference_monotonic_s"),
                f"quality packet {index} inference timestamp",
            )
            observed_t = _finite(
                packet.get("observed_monotonic_s"),
                f"quality packet {index} observed timestamp",
            )
        except ValueError as exc:
            raise QualityPacketError(str(exc)) from exc
        if previous_inference_t is not None and inference_t <= previous_inference_t:
            raise QualityPacketError(
                f"quality packet {index} inference time did not advance"
            )
        if previous_observed_t is not None and observed_t <= previous_observed_t:
            raise QualityPacketError(
                f"quality packet {index} observation time did not advance"
            )
        inference_age = observed_t - inference_t
        if (
            inference_age < -1e-9
            or inference_age > float(policy["max_inference_age_s"]) + 1e-12
        ):
            raise QualityPacketError(
                f"quality packet {index} inference was stale when observed"
            )

        audio = packet["audio_window"]
        force = packet["force_sample"]
        result = packet["result"]
        try:
            audio_start = _finite(
                audio.get("t_start_monotonic_s"),
                f"quality packet {index} audio start",
            )
            audio_end = _finite(
                audio.get("t_end_monotonic_s"),
                f"quality packet {index} audio end",
            )
            sample_rate = audio.get("sample_rate")
            sample_count = audio.get("sample_count")
            if type(sample_rate) is not int or type(sample_count) is not int:
                raise ValueError("audio sample_rate/sample_count must be integers")
            audio_sha256 = _require_sha256(
                audio.get("content_sha256"),
                f"quality packet {index} audio content SHA-256",
            )
            model_input = _normalize_model_input_claim(
                audio.get("model_input")
            )
            force_t = _finite(
                force.get("host_t_s"),
                f"quality packet {index} force timestamp",
            )
            raw_adc = force.get("raw")
            if type(raw_adc) is not int or raw_adc < 0:
                raise ValueError("force raw ADC must be a nonnegative integer")
            _finite(
                force.get("hall_v"),
                f"quality packet {index} force Hall voltage",
            )
            _finite(
                force.get("displacement_mm"),
                f"quality packet {index} force displacement",
            )
            force_n = _finite(
                force.get("force_n"),
                f"quality packet {index} force value",
            )
            force_device_t = _finite(
                force.get("device_t_ms"),
                f"quality packet {index} force device timestamp",
            )
            q_raw = _finite(
                result.get("q_raw"),
                f"quality packet {index} scorer output",
            )
            p_usable_raw = _finite(
                result.get("p_usable_raw"),
                f"quality packet {index} p_usable_raw",
            )
            p_usable = _finite(
                result.get("p_usable"),
                f"quality packet {index} p_usable",
            )
        except (TypeError, ValueError) as exc:
            raise QualityPacketError(
                f"quality packet {index} evidence is invalid: {exc}"
            ) from exc
        if (
            sample_rate <= 0
            or sample_count <= 0
            or audio.get("encoding") != "float32_le_mono"
            or audio.get("status") != "OK"
        ):
            raise QualityPacketError(f"quality packet {index} audio shape/status is invalid")
        tolerance = 1.0 / float(sample_rate)
        if (
            abs((audio_end - audio_start) - sample_count / float(sample_rate))
            > tolerance
            or abs(sample_count / float(sample_rate) - float(policy["window_s"]))
            > tolerance
            or inference_t - audio_end < -1e-9
            or inference_t - audio_end
            > float(policy["max_audio_age_s"]) + 1e-12
        ):
            raise QualityPacketError(
                f"quality packet {index} audio interval/freshness is invalid"
            )
        if (
            previous_audio_start is not None
            and (
                audio_start <= previous_audio_start
                or audio_end <= previous_audio_end
            )
        ):
            raise QualityPacketError(
                f"quality packet {index} audio time did not advance"
            )
        if audio_sha256 in audio_hashes:
            raise QualityPacketError(
                f"quality packet {index} audio input hash is frozen/reused"
            )
        if (
            model_input["inference_sequence"] != sequence
            or model_input["sha256"] != audio_sha256
            or model_input["shape"] != [sample_count]
            or model_input["bytes"] != sample_count * 4
        ):
            raise QualityPacketError(
                f"quality packet {index} model input binding is invalid"
            )
        if (
            inference_t - force_t < -1e-9
            or inference_t - force_t
            > float(policy["max_force_age_s"]) + 1e-12
            or force.get("status") not in _FORCE_SAMPLE_STATUSES
        ):
            raise QualityPacketError(
                f"quality packet {index} force sample is invalid or stale"
            )
        if (
            previous_force_t is not None
            and (
                force_t <= previous_force_t
                or force_device_t <= previous_force_device_t
            )
        ):
            raise QualityPacketError(
                f"quality packet {index} force evidence is duplicate/replayed"
            )
        force_lo, force_hi = policy["force_band_n"]
        expected_gate = (
            force.get("status") in {"OK", "WARN"}
            and float(force_lo) <= force_n <= float(force_hi)
        )
        if result.get("force_gate_open") is not expected_gate:
            raise QualityPacketError(
                f"quality packet {index} force gate evidence does not match"
            )
        output_contract = policy["output_contract"]
        output_lo, output_hi = output_contract["output_range"]
        if not float(output_lo) <= q_raw <= float(output_hi):
            raise QualityPacketError(
                f"quality packet {index} scorer output is outside output_range"
            )
        expected_p_usable = (
            q_raw
            if output_contract["transform_to_p_usable"] == "identity"
            else 1.0 - q_raw
        )
        expected_p_usable_smoothed = (
            float(policy["ema"]) * expected_p_usable_smoothed
            + (1.0 - float(policy["ema"]))
            * (expected_p_usable if expected_gate else 0.0)
        )
        if (
            abs(p_usable_raw - expected_p_usable) > 1e-12
            or not 0.0 <= p_usable <= 1.0
        ):
            raise QualityPacketError(
                f"quality packet {index} output semantic transform is invalid"
            )
        if abs(p_usable - expected_p_usable_smoothed) > 1e-12:
            raise QualityPacketError(
                f"quality packet {index} p_usable does not match policy EMA"
            )

        verified_packets.append(packet)
        audio_hashes.add(audio_sha256)
        q_raw_values.append(q_raw)
        p_usable_raw_values.append(p_usable_raw)
        previous_chain = packet["packet_chain_sha256"]
        previous_sequence = sequence
        previous_inference_t = inference_t
        previous_observed_t = observed_t
        previous_audio_start = audio_start
        previous_audio_end = audio_end
        previous_force_t = force_t
        previous_force_device_t = force_device_t

    first = verified_packets[0]
    last = verified_packets[-1]
    if (
        len(verified_packets) != verified_receipt.get("packet_count")
        or first.get("inference_sequence")
        != verified_receipt.get("first_inference_sequence")
        or last.get("inference_sequence")
        != verified_receipt.get("last_inference_sequence")
        or first.get("inference_monotonic_s")
        != verified_receipt.get("first_inference_monotonic_s")
        or last.get("inference_monotonic_s")
        != verified_receipt.get("last_inference_monotonic_s")
        or last != verified_receipt.get("last_packet")
        or last.get("packet_sha256")
        != verified_receipt.get("last_packet_sha256")
        or last.get("packet_chain_sha256")
        != verified_receipt.get("packet_chain_sha256")
    ):
        raise QualityPacketError(
            "quality packet log count/first/last does not match receipt"
        )
    packet_log_bytes = b"".join(
        _canonical_json_bytes(packet) for packet in verified_packets
    )
    q_raw_range = max(q_raw_values) - min(q_raw_values)
    p_usable_raw_range = max(p_usable_raw_values) - min(p_usable_raw_values)
    q_raw_std = float(np.std(np.asarray(q_raw_values, dtype=np.float64)))
    p_usable_raw_std = float(
        np.std(np.asarray(p_usable_raw_values, dtype=np.float64))
    )
    observed_duration_s = float(
        last["inference_monotonic_s"] - first["inference_monotonic_s"]
    )
    minimum_duration_s = float(policy["window_s"])
    output_nondegenerate = (
        q_raw_range > QUALITY_OUTPUT_MINIMUM_RANGE
        and p_usable_raw_range > QUALITY_OUTPUT_MINIMUM_RANGE
        and q_raw_std > QUALITY_OUTPUT_MINIMUM_STD
        and p_usable_raw_std > QUALITY_OUTPUT_MINIMUM_STD
        and len(verified_packets) >= QUALITY_OUTPUT_MINIMUM_PACKET_COUNT
        and observed_duration_s >= minimum_duration_s
    )
    promotion_blockers = [QUALITY_PROMOTION_BLOCKER]
    if not output_nondegenerate:
        promotion_blockers.append(
            "quality output is constant or below the nondegeneracy range"
        )
    model_input_inventory = _verify_quality_model_input_inventory(
        verified_packets,
        model_input_dir,
    )
    force_input_identity = verify_quality_force_csv_binding(
        verified_packets,
        force_csv_path,
    )
    return {
        "schema_version": QUALITY_CHAIN_VERIFICATION_SCHEMA_VERSION,
        "status": "PASS" if output_nondegenerate else "FAIL",
        "chain_integrity": "PASS",
        "inference_liveness": {
            "status": "PASS",
            "packet_count": len(verified_packets),
            "sequence_progress": "PASS",
            "timestamp_progress": "PASS",
            "audio_input_progress": "PASS",
        },
        "output_nondegeneracy": {
            "policy_version": QUALITY_OUTPUT_NONDEGENERACY_POLICY_VERSION,
            "status": "PASS" if output_nondegenerate else "FAIL",
            "minimum_range": QUALITY_OUTPUT_MINIMUM_RANGE,
            "minimum_std": QUALITY_OUTPUT_MINIMUM_STD,
            "minimum_packet_count": QUALITY_OUTPUT_MINIMUM_PACKET_COUNT,
            "minimum_duration_s": minimum_duration_s,
            "packet_count": len(verified_packets),
            "observed_duration_s": observed_duration_s,
            "q_raw_range": q_raw_range,
            "p_usable_raw_range": p_usable_raw_range,
            "q_raw_std": q_raw_std,
            "p_usable_raw_std": p_usable_raw_std,
        },
        "packet_log_hash_contract": "canonical_packet_jsonl_sha256_v2",
        "packet_log_sha256": _sha256_bytes(packet_log_bytes),
        "packet_count": len(verified_packets),
        "first_inference_sequence": first["inference_sequence"],
        "last_inference_sequence": last["inference_sequence"],
        "last_packet_chain_sha256": last["packet_chain_sha256"],
        "model_input_inventory": model_input_inventory,
        "force_input_identity": force_input_identity,
        "training_signal_eligible": output_nondegenerate,
        "promotion_eligible": False,
        "promotion_blockers": promotion_blockers,
    }


def resolve_hall_calibration(selected: str | Path | None) -> Path:
    """Select the persistent linbox calibration and fail closed on drift/missing data."""
    if selected is not None:
        candidate = Path(selected).expanduser()
    elif os.environ.get("WAM_HALL_CALIBRATION"):
        candidate = Path(os.environ["WAM_HALL_CALIBRATION"]).expanduser()
    else:
        home = Path(os.environ.get("HOME", str(Path.home()))).expanduser()
        candidate = home / "wam_data" / "config" / "calibration" / \
            "force_feedback_calibration_veo6_tared.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Hall calibration does not exist: {candidate}")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"Hall calibration must be a real regular file: {candidate}")
    return candidate.resolve()


def run_live(seconds: float = 12.0, hop_s: float = 1.0, window_s: float = 8.0,
             force_band=(3.0, 5.0), audio_device: str = "Scarlett Solo USB", audio_channel: int | None = 1,
             hall_port: str | None = None, hall_cal: str | None = None) -> None:
    from audio_stream import AudioStream
    from force_sensor import ForceCalibration, ForceSensor

    hall_port = hall_port or (sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*")) or [None])[0]
    hall_cal_path = resolve_hall_calibration(hall_cal)
    calibration = ForceCalibration.from_json(str(hall_cal_path))
    sq = build_diagnostic_quality_runtime(
        force_band=force_band,
        window_s=window_s,
    )
    from contextlib import ExitStack
    with ExitStack() as st:
        au = st.enter_context(AudioStream(audio_device, pick_channel=audio_channel, buffer_seconds=window_s + 3))
        fs = None
        try:
            fs = st.enter_context(ForceSensor(hall_port, calibration))
        except Exception as e:  # noqa: BLE001
            print(f"[force] not available ({e}); gate will remain closed")
        print(
            f">>> DIAGNOSTIC-ONLY spectral proxy @ {1/hop_s:.0f} Hz; "
            "not production Q <<<",
            flush=True,
        )
        time.sleep(window_s * 0.3)
        t0 = time.monotonic()
        inference_sequence = 0
        while time.monotonic() - t0 < seconds:
            w = au.latest_window(window_s)
            ah = audio_status(w.samples, w.sr)            # fault detection: stethoscope off?
            force_sample = fs.read() if fs else None
            if force_sample is None:
                print("  Q skipped: no physical force sample", flush=True)
                time.sleep(hop_s)
                continue
            inference_t = time.monotonic()
            try:
                out = sq.update(
                    audio_window=w,
                    force_sample=force_sample,
                    inference_sequence=inference_sequence,
                    inference_monotonic_s=inference_t,
                    observed_monotonic_s=time.monotonic(),
                )
            except QualityPacketError as exc:
                print(f"  Q skipped: {exc}", flush=True)
                time.sleep(hop_s)
                continue
            inference_sequence += 1
            score = float(out["diagnostic_score_smoothed"])
            bar = "#" * int(score * 30)
            tag = "" if ah["status"] == "OK" else f"  ⚠️ audio {ah['status']} ({ah['peak_db']}dB: stethoscope off?)"
            print(f"  diagnostic force={out['force_n']:.2f}N gate={'OPEN ' if out['gate'] else 'shut '} "
                  f"band score={score:.2f} |{bar}{tag}", flush=True)
            time.sleep(hop_s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--force-lo", type=float, default=3.0)
    ap.add_argument("--force-hi", type=float, default=5.0)
    a = ap.parse_args()
    run_live(seconds=a.seconds, hop_s=a.hop, force_band=(a.force_lo, a.force_hi))
