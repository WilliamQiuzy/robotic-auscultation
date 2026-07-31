"""Content-addressed per-frame geometry model output packets.

The packet carries an explicit producer channel and immutable model/runtime
identity.  Production geometry authorization consumes one perception packet
and one separately identified independent-patient-model packet.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .patient_frame_evidence import PatientFrameEvidence


GEOMETRY_CHANNEL_PACKET_SCHEMA = "wam_geometry_channel_packet_v1"
GEOMETRY_INFERENCE_ID_PREFIX = "geometry_inference-"
GEOMETRY_FRAME_PACKET_LOG_ENTRY_SCHEMA = "wam_geometry_frame_packet_log_entry_v1"
GEOMETRY_SESSION_RECEIPT_SCHEMA = "wam_geometry_session_receipt_v1"
GEOMETRY_PACKET_LOG_HASH_CONTRACT = "canonical_geometry_frame_packet_jsonl_v1"
GEOMETRY_PACKET_LOG_BASENAME = "geometry_frame_packets.jsonl"
GEOMETRY_PACKET_LOG_RECEIPT_BASENAME = "geometry_frame_authenticity.json"
GEOMETRY_RECEIPT_AUTHORIZATION_BLOCKER = (
    "geometry packet receipt alone is not finalization-bound for production"
)
PERCEPTION_OBSERVATION_CHANNEL = "perception_observation"
INDEPENDENT_PATIENT_MODEL_CHANNEL = "independent_patient_model"
_CHANNELS = frozenset(
    {PERCEPTION_OBSERVATION_CHANNEL, INDEPENDENT_PATIENT_MODEL_CHANNEL}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Underscores are permitted so the real activated model ids (e.g.
# ``patient_pose-<sha20>``) validate; the content-address suffix is unchanged.
_RELEASE_ID = re.compile(r"^[a-z][a-z0-9_-]{2,96}-[0-9a-f]{20}$")
_PRODUCER_ID = re.compile(r"^[a-z][a-z0-9-]{2,63}$")


class GeometryChannelPacketError(ValueError):
    """A model packet is mutable, malformed, or not frame-bound."""


class GeometryPacketLogError(ValueError):
    """A geometry packet-pair log or its session receipt is invalid."""


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _packet_metadata(packet: "GeometryChannelPacket") -> dict[str, object]:
    return {
        "schema": GEOMETRY_CHANNEL_PACKET_SCHEMA,
        "channel": packet.channel,
        "producer_id": packet.producer_id,
        "release_id": packet.release_id,
        "model_sha256": packet.model_sha256,
        "frame_bundle_id": packet.frame_bundle_id,
        "patient_frame_evidence_id": packet.patient_frame_evidence_id,
        "episode_id": packet.episode_id,
        "frame_index": packet.frame_index,
        "camera_host_monotonic_ns": packet.camera_host_monotonic_ns,
        "sequence": packet.sequence,
        "input_rgb_sha256": packet.input_rgb_sha256,
        "input_depth_sha256": packet.input_depth_sha256,
        "point_names": list(packet.point_names),
        "points_sha256": packet.points_sha256,
        "covariances_sha256": packet.covariances_sha256,
    }


def _readonly_float_array(value: object) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GeometryChannelPacket:
    """Immutable output from exactly one declared geometry producer."""

    schema: str
    channel: str
    producer_id: str
    release_id: str
    model_sha256: str
    frame_bundle_id: str
    patient_frame_evidence_id: str
    episode_id: str
    frame_index: int
    camera_host_monotonic_ns: int
    sequence: int
    input_rgb_sha256: str
    input_depth_sha256: str
    point_names: tuple[str, ...]
    points_m: np.ndarray
    covariances_m2: np.ndarray
    points_sha256: str
    covariances_sha256: str
    inference_id: str

    def __post_init__(self) -> None:
        if self.schema != GEOMETRY_CHANNEL_PACKET_SCHEMA:
            raise GeometryChannelPacketError("geometry packet schema is unsupported")
        if self.channel not in _CHANNELS:
            raise GeometryChannelPacketError("geometry packet channel is invalid")
        if not isinstance(self.producer_id, str) or not _PRODUCER_ID.fullmatch(
            self.producer_id
        ):
            raise GeometryChannelPacketError("geometry producer identity is invalid")
        if not isinstance(self.model_sha256, str) or not _SHA256.fullmatch(
            self.model_sha256
        ):
            raise GeometryChannelPacketError("geometry model SHA-256 is invalid")
        if (
            not isinstance(self.release_id, str)
            or not _RELEASE_ID.fullmatch(self.release_id)
            or not self.release_id.endswith(self.model_sha256[:20])
        ):
            raise GeometryChannelPacketError(
                "geometry release identity does not bind the model"
            )
        for identity, prefix, label in (
            (self.frame_bundle_id, "frame_evidence-", "frame bundle"),
            (
                self.patient_frame_evidence_id,
                "patient_frame_evidence-",
                "patient frame evidence",
            ),
        ):
            if (
                not isinstance(identity, str)
                or not identity.startswith(prefix)
                or _SHA256.fullmatch(identity[len(prefix) :]) is None
            ):
                raise GeometryChannelPacketError(f"{label} identity is invalid")
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise GeometryChannelPacketError("geometry episode identity is missing")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise GeometryChannelPacketError("geometry frame index is invalid")
        if (
            type(self.camera_host_monotonic_ns) is not int
            or self.camera_host_monotonic_ns <= 0
        ):
            raise GeometryChannelPacketError("geometry frame timestamp is invalid")
        if type(self.sequence) is not int or self.sequence <= 0:
            raise GeometryChannelPacketError("geometry sequence is invalid")
        for digest, label in (
            (self.input_rgb_sha256, "input RGB"),
            (self.input_depth_sha256, "input depth"),
            (self.points_sha256, "points"),
            (self.covariances_sha256, "covariances"),
        ):
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise GeometryChannelPacketError(f"{label} SHA-256 is invalid")
        if (
            not isinstance(self.point_names, tuple)
            or not self.point_names
            or len(set(self.point_names)) != len(self.point_names)
            or any(not isinstance(name, str) or not name for name in self.point_names)
        ):
            raise GeometryChannelPacketError("geometry point names are invalid")

        points = _readonly_float_array(self.points_m)
        covariances = _readonly_float_array(self.covariances_m2)
        count = len(self.point_names)
        if points.shape != (count, 3) or not np.isfinite(points).all():
            raise GeometryChannelPacketError("geometry points are invalid")
        if covariances.shape != (count, 3, 3) or not np.isfinite(
            covariances
        ).all():
            raise GeometryChannelPacketError("geometry covariances are invalid")
        symmetric = 0.5 * (covariances + np.swapaxes(covariances, -1, -2))
        if not np.allclose(covariances, symmetric, rtol=0.0, atol=1.0e-15):
            raise GeometryChannelPacketError("geometry covariances are not symmetric")
        eigenvalues = np.linalg.eigvalsh(symmetric)
        if (
            not np.isfinite(eigenvalues).all()
            or np.any(eigenvalues < -1.0e-15)
            or np.any(eigenvalues[:, -1] <= 0.0)
        ):
            raise GeometryChannelPacketError(
                "geometry covariances are not positive semidefinite and nonzero"
            )
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "covariances_m2", covariances)
        if self.points_sha256 != _array_sha256(points):
            raise GeometryChannelPacketError("geometry points SHA-256 mismatch")
        if self.covariances_sha256 != _array_sha256(covariances):
            raise GeometryChannelPacketError("geometry covariance SHA-256 mismatch")
        expected_inference_id = GEOMETRY_INFERENCE_ID_PREFIX + hashlib.sha256(
            _canonical_json(_packet_metadata(self))
        ).hexdigest()
        if self.inference_id != expected_inference_id:
            raise GeometryChannelPacketError("geometry inference identity mismatch")

    @classmethod
    def build(
        cls,
        *,
        channel: str,
        producer_id: str,
        release_id: str,
        model_sha256: str,
        patient_frame_evidence: PatientFrameEvidence,
        episode_id: str,
        sequence: int,
        point_names: tuple[str, ...],
        points_m: np.ndarray,
        covariances_m2: np.ndarray,
    ) -> "GeometryChannelPacket":
        from .patient_frame_evidence import (
            PatientFrameEvidence,
            PatientFrameEvidenceError,
        )

        if not isinstance(patient_frame_evidence, PatientFrameEvidence):
            raise GeometryChannelPacketError("patient frame evidence is required")
        try:
            patient_frame_evidence.__post_init__()
        except PatientFrameEvidenceError as exc:
            raise GeometryChannelPacketError(str(exc)) from exc
        camera = patient_frame_evidence.frame_bundle.camera_frame
        points = np.asarray(points_m, dtype=np.float64)
        covariances = np.asarray(covariances_m2, dtype=np.float64)
        values = {
            "schema": GEOMETRY_CHANNEL_PACKET_SCHEMA,
            "channel": channel,
            "producer_id": producer_id,
            "release_id": release_id,
            "model_sha256": model_sha256,
            "frame_bundle_id": patient_frame_evidence.frame_bundle.bundle_id,
            "patient_frame_evidence_id": patient_frame_evidence.evidence_id,
            "episode_id": episode_id,
            "frame_index": camera.index,
            "camera_host_monotonic_ns": camera.camera_host_monotonic_ns,
            "sequence": sequence,
            "input_rgb_sha256": camera.rgb_sha256,
            "input_depth_sha256": camera.depth_sha256,
            "point_names": tuple(point_names),
            "points_m": points,
            "covariances_m2": covariances,
            "points_sha256": _array_sha256(points),
            "covariances_sha256": _array_sha256(covariances),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        inference_id = GEOMETRY_INFERENCE_ID_PREFIX + hashlib.sha256(
            _canonical_json(_packet_metadata(provisional))
        ).hexdigest()
        return cls(**values, inference_id=inference_id)

    @classmethod
    def from_log_payload(cls, value: Mapping[str, object]) -> "GeometryChannelPacket":
        """Reconstruct and re-hash one fully numeric persisted model packet."""

        expected = {
            "schema", "channel", "producer_id", "release_id", "model_sha256",
            "frame_bundle_id", "patient_frame_evidence_id", "episode_id",
            "frame_index", "camera_host_monotonic_ns", "sequence",
            "input_rgb_sha256", "input_depth_sha256", "point_names",
            "points_m", "covariances_m2", "points_sha256",
            "covariances_sha256", "inference_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise GeometryPacketLogError(
                "persisted geometry packet fields are incomplete"
            )
        point_names = value.get("point_names")
        if not isinstance(point_names, list):
            raise GeometryPacketLogError(
                "persisted geometry packet point order is invalid"
            )
        try:
            return cls(
                schema=value["schema"],
                channel=value["channel"],
                producer_id=value["producer_id"],
                release_id=value["release_id"],
                model_sha256=value["model_sha256"],
                frame_bundle_id=value["frame_bundle_id"],
                patient_frame_evidence_id=value["patient_frame_evidence_id"],
                episode_id=value["episode_id"],
                frame_index=value["frame_index"],
                camera_host_monotonic_ns=value["camera_host_monotonic_ns"],
                sequence=value["sequence"],
                input_rgb_sha256=value["input_rgb_sha256"],
                input_depth_sha256=value["input_depth_sha256"],
                point_names=tuple(point_names),
                points_m=np.asarray(value["points_m"], dtype=np.float64),
                covariances_m2=np.asarray(
                    value["covariances_m2"], dtype=np.float64
                ),
                points_sha256=value["points_sha256"],
                covariances_sha256=value["covariances_sha256"],
                inference_id=value["inference_id"],
            )
        except (KeyError, TypeError, ValueError, GeometryChannelPacketError) as exc:
            raise GeometryPacketLogError(
                f"persisted geometry packet is invalid: {exc}"
            ) from exc


def geometry_channels_are_independent(
    perception: GeometryChannelPacket,
    independent: GeometryChannelPacket,
) -> bool:
    """Return whether two packets declare distinct production identities."""

    return bool(
        perception.channel == PERCEPTION_OBSERVATION_CHANNEL
        and independent.channel == INDEPENDENT_PATIENT_MODEL_CHANNEL
        and perception.producer_id != independent.producer_id
        and perception.release_id != independent.release_id
        and perception.model_sha256 != independent.model_sha256
        and perception.inference_id != independent.inference_id
    )


def _packet_log_metadata(packet: GeometryChannelPacket) -> dict[str, object]:
    metadata = _packet_metadata(packet)
    metadata["points_m"] = packet.points_m.tolist()
    metadata["covariances_m2"] = packet.covariances_m2.tolist()
    metadata["inference_id"] = packet.inference_id
    return metadata


def _frame_packet_log_entry_payload(
    entry: "GeometryFramePacketLogEntry",
) -> dict[str, object]:
    return {
        "schema": GEOMETRY_FRAME_PACKET_LOG_ENTRY_SCHEMA,
        "perception_observation_packet": _packet_log_metadata(
            entry.perception_observation_packet
        ),
        "independent_model_packet": _packet_log_metadata(
            entry.independent_model_packet
        ),
        "previous_entry_chain_sha256": entry.previous_entry_chain_sha256,
    }


def _frame_packet_log_entry_hashes(
    entry: "GeometryFramePacketLogEntry",
) -> tuple[str, str]:
    entry_sha256 = hashlib.sha256(
        _canonical_json(_frame_packet_log_entry_payload(entry))
    ).hexdigest()
    previous = entry.previous_entry_chain_sha256
    previous_raw = bytes(32) if previous is None else bytes.fromhex(previous)
    chain_sha256 = hashlib.sha256(
        previous_raw + bytes.fromhex(entry_sha256)
    ).hexdigest()
    return entry_sha256, chain_sha256


def _frame_packet_log_entry_bytes(
    entry: "GeometryFramePacketLogEntry",
) -> bytes:
    value = _frame_packet_log_entry_payload(entry)
    value.update(
        {
            "entry_sha256": entry.entry_sha256,
            "entry_chain_sha256": entry.entry_chain_sha256,
        }
    )
    return _canonical_json(value) + b"\n"


@dataclass(frozen=True)
class GeometryFramePacketLogEntry:
    """One chained same-frame pair of perception and independent outputs."""

    schema: str
    perception_observation_packet: GeometryChannelPacket
    independent_model_packet: GeometryChannelPacket
    previous_entry_chain_sha256: str | None
    entry_sha256: str
    entry_chain_sha256: str

    def __post_init__(self) -> None:
        if self.schema != GEOMETRY_FRAME_PACKET_LOG_ENTRY_SCHEMA:
            raise GeometryPacketLogError("geometry packet log schema is invalid")
        perception = self.perception_observation_packet
        independent = self.independent_model_packet
        if not isinstance(perception, GeometryChannelPacket) or not isinstance(
            independent, GeometryChannelPacket
        ):
            raise GeometryPacketLogError("geometry packet log pair is missing")
        try:
            perception.__post_init__()
            independent.__post_init__()
        except GeometryChannelPacketError as exc:
            raise GeometryPacketLogError(
                "geometry packet log pair contains an invalid packet"
            ) from exc
        if not geometry_channels_are_independent(perception, independent):
            raise GeometryPacketLogError(
                "geometry packet log channels are not independent"
            )
        perception_context = (
            perception.frame_bundle_id,
            perception.patient_frame_evidence_id,
            perception.episode_id,
            perception.frame_index,
            perception.camera_host_monotonic_ns,
            perception.sequence,
            perception.input_rgb_sha256,
            perception.input_depth_sha256,
            perception.point_names,
        )
        independent_context = (
            independent.frame_bundle_id,
            independent.patient_frame_evidence_id,
            independent.episode_id,
            independent.frame_index,
            independent.camera_host_monotonic_ns,
            independent.sequence,
            independent.input_rgb_sha256,
            independent.input_depth_sha256,
            independent.point_names,
        )
        if perception_context != independent_context:
            raise GeometryPacketLogError(
                "geometry packet log pair is not bound to one frame"
            )
        if self.previous_entry_chain_sha256 is not None and (
            not isinstance(self.previous_entry_chain_sha256, str)
            or _SHA256.fullmatch(self.previous_entry_chain_sha256) is None
        ):
            raise GeometryPacketLogError(
                "geometry packet previous chain SHA-256 is invalid"
            )
        for digest, label in (
            (self.entry_sha256, "entry"),
            (self.entry_chain_sha256, "entry chain"),
        ):
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise GeometryPacketLogError(
                    f"geometry packet {label} SHA-256 is invalid"
                )
        expected_entry, expected_chain = _frame_packet_log_entry_hashes(self)
        if self.entry_sha256 != expected_entry:
            raise GeometryPacketLogError(
                "geometry packet log entry SHA-256 does not match"
            )
        if self.entry_chain_sha256 != expected_chain:
            raise GeometryPacketLogError(
                "geometry packet log entry chain SHA-256 does not match"
            )

    @classmethod
    def build(
        cls,
        *,
        perception_observation_packet: GeometryChannelPacket,
        independent_model_packet: GeometryChannelPacket,
        previous_entry: "GeometryFramePacketLogEntry | None" = None,
    ) -> "GeometryFramePacketLogEntry":
        if previous_entry is not None:
            if not isinstance(previous_entry, cls):
                raise GeometryPacketLogError(
                    "previous geometry packet log entry is invalid"
                )
            previous_entry.__post_init__()
        values = {
            "schema": GEOMETRY_FRAME_PACKET_LOG_ENTRY_SCHEMA,
            "perception_observation_packet": perception_observation_packet,
            "independent_model_packet": independent_model_packet,
            "previous_entry_chain_sha256": (
                previous_entry.entry_chain_sha256
                if previous_entry is not None
                else None
            ),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        entry_sha256, entry_chain_sha256 = _frame_packet_log_entry_hashes(
            provisional
        )
        return cls(
            **values,
            entry_sha256=entry_sha256,
            entry_chain_sha256=entry_chain_sha256,
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "GeometryFramePacketLogEntry":
        expected = {
            "schema", "perception_observation_packet",
            "independent_model_packet", "previous_entry_chain_sha256",
            "entry_sha256", "entry_chain_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise GeometryPacketLogError(
                "persisted geometry log entry fields are incomplete"
            )
        try:
            return cls(
                schema=value["schema"],
                perception_observation_packet=(
                    GeometryChannelPacket.from_log_payload(
                        value["perception_observation_packet"]
                    )
                ),
                independent_model_packet=GeometryChannelPacket.from_log_payload(
                    value["independent_model_packet"]
                ),
                previous_entry_chain_sha256=value[
                    "previous_entry_chain_sha256"
                ],
                entry_sha256=value["entry_sha256"],
                entry_chain_sha256=value["entry_chain_sha256"],
            )
        except (KeyError, TypeError, ValueError, GeometryPacketLogError) as exc:
            if isinstance(exc, GeometryPacketLogError):
                raise
            raise GeometryPacketLogError(
                "persisted geometry log entry is invalid"
            ) from exc


def _validated_geometry_packet_log(
    packet_log: Iterable[GeometryFramePacketLogEntry],
) -> tuple[GeometryFramePacketLogEntry, ...]:
    if isinstance(packet_log, (str, bytes, bytearray, Mapping)):
        raise GeometryPacketLogError(
            "geometry packet log must be an iterable of entries"
        )
    try:
        entries = tuple(packet_log)
    except TypeError as exc:
        raise GeometryPacketLogError("geometry packet log is not iterable") from exc
    if len(entries) < 2:
        raise GeometryPacketLogError(
            "geometry packet log requires at least 2 packet pairs"
        )

    first = entries[0]
    if not isinstance(first, GeometryFramePacketLogEntry):
        raise GeometryPacketLogError("geometry packet log entry 0 is invalid")
    first.__post_init__()
    if first.previous_entry_chain_sha256 is not None:
        raise GeometryPacketLogError(
            "geometry packet log first previous chain must be empty"
        )
    first_perception = first.perception_observation_packet
    first_independent = first.independent_model_packet
    session_identity = (
        first_perception.episode_id,
        first_perception.point_names,
        first_perception.producer_id,
        first_perception.release_id,
        first_perception.model_sha256,
        first_independent.producer_id,
        first_independent.release_id,
        first_independent.model_sha256,
    )
    seen_frame_bundle_ids: set[str] = set()
    seen_patient_frame_evidence_ids: set[str] = set()
    seen_inference_ids: set[str] = set()
    previous: GeometryFramePacketLogEntry | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, GeometryFramePacketLogEntry):
            raise GeometryPacketLogError(
                f"geometry packet log entry {index} is invalid"
            )
        try:
            entry.__post_init__()
        except (GeometryChannelPacketError, GeometryPacketLogError) as exc:
            raise GeometryPacketLogError(
                f"geometry packet log entry {index} failed validation: {exc}"
            ) from exc
        perception = entry.perception_observation_packet
        independent = entry.independent_model_packet
        observed_identity = (
            perception.episode_id,
            perception.point_names,
            perception.producer_id,
            perception.release_id,
            perception.model_sha256,
            independent.producer_id,
            independent.release_id,
            independent.model_sha256,
        )
        if observed_identity != session_identity:
            raise GeometryPacketLogError(
                f"geometry packet log entry {index} session identity changed"
            )
        if previous is not None:
            previous_packet = previous.perception_observation_packet
            if entry.previous_entry_chain_sha256 != previous.entry_chain_sha256:
                raise GeometryPacketLogError(
                    f"geometry packet log entry {index} chain is not continuous"
                )
            if perception.sequence != previous_packet.sequence + 1:
                raise GeometryPacketLogError(
                    f"geometry packet log entry {index} sequence is not continuous"
                )
            if (
                perception.frame_index <= previous_packet.frame_index
                or perception.camera_host_monotonic_ns
                <= previous_packet.camera_host_monotonic_ns
            ):
                raise GeometryPacketLogError(
                    f"geometry packet log entry {index} frame time did not advance"
                )
        packet_inference_ids = {
            perception.inference_id,
            independent.inference_id,
        }
        if (
            perception.frame_bundle_id in seen_frame_bundle_ids
            or perception.patient_frame_evidence_id
            in seen_patient_frame_evidence_ids
            or not packet_inference_ids.isdisjoint(seen_inference_ids)
        ):
            raise GeometryPacketLogError(
                f"geometry packet log entry {index} replays frame evidence"
            )
        seen_frame_bundle_ids.add(perception.frame_bundle_id)
        seen_patient_frame_evidence_ids.add(perception.patient_frame_evidence_id)
        seen_inference_ids.update(packet_inference_ids)
        previous = entry
    return entries


def _geometry_session_receipt_payload(
    receipt: "GeometrySessionReceipt",
) -> dict[str, object]:
    return {
        "schema": GEOMETRY_SESSION_RECEIPT_SCHEMA,
        "status": receipt.status,
        "hash_contract": receipt.hash_contract,
        "production_authorized": receipt.production_authorized,
        "authorization_blockers": list(receipt.authorization_blockers),
        "episode_id": receipt.episode_id,
        "packet_count": receipt.packet_count,
        "first_sequence": receipt.first_sequence,
        "last_sequence": receipt.last_sequence,
        "first_frame_index": receipt.first_frame_index,
        "last_frame_index": receipt.last_frame_index,
        "first_camera_host_monotonic_ns": (
            receipt.first_camera_host_monotonic_ns
        ),
        "last_camera_host_monotonic_ns": receipt.last_camera_host_monotonic_ns,
        "point_names": list(receipt.point_names),
        "perception_producer_id": receipt.perception_producer_id,
        "perception_release_id": receipt.perception_release_id,
        "perception_model_sha256": receipt.perception_model_sha256,
        "independent_producer_id": receipt.independent_producer_id,
        "independent_release_id": receipt.independent_release_id,
        "independent_model_sha256": receipt.independent_model_sha256,
        "first_entry_sha256": receipt.first_entry_sha256,
        "last_entry_sha256": receipt.last_entry_sha256,
        "last_entry_chain_sha256": receipt.last_entry_chain_sha256,
        "packet_log_sha256": receipt.packet_log_sha256,
    }


@dataclass(frozen=True)
class GeometrySessionReceipt:
    """Self-hashed summary that is valid only with its full packet-pair log."""

    schema: str
    status: str
    hash_contract: str
    production_authorized: bool
    authorization_blockers: tuple[str, ...]
    episode_id: str
    packet_count: int
    first_sequence: int
    last_sequence: int
    first_frame_index: int
    last_frame_index: int
    first_camera_host_monotonic_ns: int
    last_camera_host_monotonic_ns: int
    point_names: tuple[str, ...]
    perception_producer_id: str
    perception_release_id: str
    perception_model_sha256: str
    independent_producer_id: str
    independent_release_id: str
    independent_model_sha256: str
    first_entry_sha256: str
    last_entry_sha256: str
    last_entry_chain_sha256: str
    packet_log_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.schema != GEOMETRY_SESSION_RECEIPT_SCHEMA:
            raise GeometryPacketLogError("geometry session receipt schema is invalid")
        if self.status != "PASS":
            raise GeometryPacketLogError("geometry session receipt is not PASS")
        if self.hash_contract != GEOMETRY_PACKET_LOG_HASH_CONTRACT:
            raise GeometryPacketLogError(
                "geometry session receipt hash contract is invalid"
            )
        if (
            self.production_authorized is not False
            or self.authorization_blockers
            != (GEOMETRY_RECEIPT_AUTHORIZATION_BLOCKER,)
        ):
            raise GeometryPacketLogError(
                "geometry session receipt cannot authorize production"
            )
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise GeometryPacketLogError(
                "geometry session receipt episode is invalid"
            )
        integer_fields = (
            self.packet_count,
            self.first_sequence,
            self.last_sequence,
            self.first_frame_index,
            self.last_frame_index,
            self.first_camera_host_monotonic_ns,
            self.last_camera_host_monotonic_ns,
        )
        if any(type(value) is not int for value in integer_fields):
            raise GeometryPacketLogError(
                "geometry session receipt progress is invalid"
            )
        if (
            self.packet_count < 2
            or self.first_sequence <= 0
            or self.last_sequence != self.first_sequence + self.packet_count - 1
            or self.first_frame_index < 0
            or self.last_frame_index <= self.first_frame_index
            or self.first_camera_host_monotonic_ns <= 0
            or self.last_camera_host_monotonic_ns
            <= self.first_camera_host_monotonic_ns
        ):
            raise GeometryPacketLogError(
                "geometry session receipt progress is not continuous"
            )
        if (
            not isinstance(self.point_names, tuple)
            or not self.point_names
            or len(set(self.point_names)) != len(self.point_names)
            or any(not isinstance(name, str) or not name for name in self.point_names)
        ):
            raise GeometryPacketLogError(
                "geometry session receipt point order is invalid"
            )
        source_identities = (
            (
                self.perception_producer_id,
                self.perception_release_id,
                self.perception_model_sha256,
            ),
            (
                self.independent_producer_id,
                self.independent_release_id,
                self.independent_model_sha256,
            ),
        )
        for producer_id, release_id, model_sha256 in source_identities:
            if (
                not isinstance(producer_id, str)
                or _PRODUCER_ID.fullmatch(producer_id) is None
                or not isinstance(model_sha256, str)
                or _SHA256.fullmatch(model_sha256) is None
                or not isinstance(release_id, str)
                or _RELEASE_ID.fullmatch(release_id) is None
                or not release_id.endswith(model_sha256[:20])
            ):
                raise GeometryPacketLogError(
                    "geometry session receipt source identity is invalid"
                )
        if len({item[0] for item in source_identities}) != 2 or len(
            {item[1] for item in source_identities}
        ) != 2 or len({item[2] for item in source_identities}) != 2:
            raise GeometryPacketLogError(
                "geometry session receipt channels are not independent"
            )
        for digest in (
            self.first_entry_sha256,
            self.last_entry_sha256,
            self.last_entry_chain_sha256,
            self.packet_log_sha256,
            self.receipt_sha256,
        ):
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise GeometryPacketLogError(
                    "geometry session receipt SHA-256 is invalid"
                )
        expected_receipt_sha256 = hashlib.sha256(
            _canonical_json(_geometry_session_receipt_payload(self))
        ).hexdigest()
        if self.receipt_sha256 != expected_receipt_sha256:
            raise GeometryPacketLogError(
                "geometry session receipt SHA-256 does not match"
            )

    @classmethod
    def build(
        cls,
        packet_log: Iterable[GeometryFramePacketLogEntry],
    ) -> "GeometrySessionReceipt":
        entries = _validated_geometry_packet_log(packet_log)
        first = entries[0]
        last = entries[-1]
        first_perception = first.perception_observation_packet
        last_perception = last.perception_observation_packet
        first_independent = first.independent_model_packet
        values = {
            "schema": GEOMETRY_SESSION_RECEIPT_SCHEMA,
            "status": "PASS",
            "hash_contract": GEOMETRY_PACKET_LOG_HASH_CONTRACT,
            "production_authorized": False,
            "authorization_blockers": (
                GEOMETRY_RECEIPT_AUTHORIZATION_BLOCKER,
            ),
            "episode_id": first_perception.episode_id,
            "packet_count": len(entries),
            "first_sequence": first_perception.sequence,
            "last_sequence": last_perception.sequence,
            "first_frame_index": first_perception.frame_index,
            "last_frame_index": last_perception.frame_index,
            "first_camera_host_monotonic_ns": (
                first_perception.camera_host_monotonic_ns
            ),
            "last_camera_host_monotonic_ns": (
                last_perception.camera_host_monotonic_ns
            ),
            "point_names": first_perception.point_names,
            "perception_producer_id": first_perception.producer_id,
            "perception_release_id": first_perception.release_id,
            "perception_model_sha256": first_perception.model_sha256,
            "independent_producer_id": first_independent.producer_id,
            "independent_release_id": first_independent.release_id,
            "independent_model_sha256": first_independent.model_sha256,
            "first_entry_sha256": first.entry_sha256,
            "last_entry_sha256": last.entry_sha256,
            "last_entry_chain_sha256": last.entry_chain_sha256,
            "packet_log_sha256": hashlib.sha256(
                b"".join(_frame_packet_log_entry_bytes(entry) for entry in entries)
            ).hexdigest(),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        receipt_sha256 = hashlib.sha256(
            _canonical_json(_geometry_session_receipt_payload(provisional))
        ).hexdigest()
        return cls(**values, receipt_sha256=receipt_sha256)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "GeometrySessionReceipt":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != expected:
            raise GeometryPacketLogError(
                "persisted geometry receipt fields are incomplete"
            )
        blockers = value.get("authorization_blockers")
        point_names = value.get("point_names")
        if not isinstance(blockers, list) or not isinstance(point_names, list):
            raise GeometryPacketLogError(
                "persisted geometry receipt lists are invalid"
            )
        fields = dict(value)
        fields["authorization_blockers"] = tuple(blockers)
        fields["point_names"] = tuple(point_names)
        try:
            return cls(**fields)
        except (TypeError, ValueError, GeometryPacketLogError) as exc:
            if isinstance(exc, GeometryPacketLogError):
                raise
            raise GeometryPacketLogError(
                "persisted geometry receipt is invalid"
            ) from exc


def verify_geometry_packet_log(
    packet_log: Iterable[GeometryFramePacketLogEntry],
    receipt: GeometrySessionReceipt,
) -> GeometrySessionReceipt:
    """Verify every packet pair and bind the exact canonical log to a receipt."""

    if not isinstance(receipt, GeometrySessionReceipt):
        raise GeometryPacketLogError("geometry session receipt is required")
    receipt.__post_init__()
    entries = _validated_geometry_packet_log(packet_log)
    expected = GeometrySessionReceipt.build(entries)
    if receipt != expected:
        raise GeometryPacketLogError(
            "geometry packet log count/identity/chain does not match receipt"
        )
    return receipt


def _strict_json_object(pairs):  # noqa: ANN001, ANN202
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise GeometryPacketLogError(
                f"duplicate persisted geometry JSON field: {key}"
            )
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise GeometryPacketLogError(
        f"non-finite persisted geometry JSON number is forbidden: {value}"
    )


def _stable_regular_bytes(
    path: Path, *, label: str, maximum_bytes: int = 256 * 1024 * 1024,
) -> bytes:
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise GeometryPacketLogError(f"{label} is missing") from exc
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise GeometryPacketLogError(f"{label} must be a regular non-symlink file")
    if before_path.st_size <= 0 or before_path.st_size > maximum_bytes:
        raise GeometryPacketLogError(f"{label} has an invalid size")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise GeometryPacketLogError(f"{label} cannot be opened") from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
            if size > maximum_bytes:
                raise GeometryPacketLogError(f"{label} exceeds its size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (  # noqa: E731
        item.st_dev, item.st_ino, item.st_mode, item.st_size,
        getattr(item, "st_mtime_ns", int(item.st_mtime * 1_000_000_000)),
        getattr(item, "st_ctime_ns", int(item.st_ctime * 1_000_000_000)),
    )
    if identity(before) != identity(after) or size != before.st_size:
        raise GeometryPacketLogError(f"{label} changed while reading")
    return b"".join(chunks)


def _decode_canonical_json(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GeometryPacketLogError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_json(value) + b"\n":
        raise GeometryPacketLogError(f"{label} is not canonical JSON")
    return value


def read_geometry_packet_log(
    path: str | Path,
) -> tuple[GeometryFramePacketLogEntry, ...]:
    raw = _stable_regular_bytes(Path(path), label="geometry packet log")
    entries = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or len(line) > 8 * 1024 * 1024:
            raise GeometryPacketLogError(
                f"geometry packet log line {line_number} is incomplete or too large"
            )
        value = _decode_canonical_json(
            line, label=f"geometry packet log line {line_number}"
        )
        entries.append(GeometryFramePacketLogEntry.from_dict(value))
    return _validated_geometry_packet_log(entries)


def read_geometry_session_receipt(path: str | Path) -> GeometrySessionReceipt:
    raw = _stable_regular_bytes(
        Path(path), label="geometry packet log receipt", maximum_bytes=1024 * 1024
    )
    return GeometrySessionReceipt.from_dict(
        _decode_canonical_json(raw, label="geometry packet log receipt")
    )


def _write_exclusive_regular(path: Path, raw: bytes, *, label: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags, 0o640)
    except OSError as exc:
        raise GeometryPacketLogError(f"{label} cannot be created") from exc
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("geometry artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_geometry_packet_log_files(
    session_dir: str | Path,
    packet_log: Iterable[GeometryFramePacketLogEntry],
) -> GeometrySessionReceipt:
    """Exclusively persist the complete numeric packet chain and its receipt."""

    session = Path(session_dir)
    try:
        session_metadata = session.lstat()
    except OSError as exc:
        raise GeometryPacketLogError("geometry session directory is missing") from exc
    if stat.S_ISLNK(session_metadata.st_mode) or not stat.S_ISDIR(
        session_metadata.st_mode
    ):
        raise GeometryPacketLogError(
            "geometry session directory must be a real directory"
        )
    entries = _validated_geometry_packet_log(packet_log)
    receipt = GeometrySessionReceipt.build(entries)
    log_raw = b"".join(_frame_packet_log_entry_bytes(entry) for entry in entries)
    if hashlib.sha256(log_raw).hexdigest() != receipt.packet_log_sha256:
        raise GeometryPacketLogError(
            "geometry packet log serialization disagrees with receipt"
        )
    receipt_value = _geometry_session_receipt_payload(receipt)
    receipt_value["receipt_sha256"] = receipt.receipt_sha256
    receipt_raw = _canonical_json(receipt_value) + b"\n"
    log_path = session / GEOMETRY_PACKET_LOG_BASENAME
    receipt_path = session / GEOMETRY_PACKET_LOG_RECEIPT_BASENAME
    _write_exclusive_regular(log_path, log_raw, label="geometry packet log")
    try:
        _write_exclusive_regular(
            receipt_path, receipt_raw, label="geometry packet log receipt"
        )
    except BaseException:
        log_path.unlink(missing_ok=True)
        _fsync_directory(session)
        raise
    _fsync_directory(session)
    _, verified = verify_geometry_packet_log_files(session)
    return verified


def verify_geometry_packet_log_files(
    session_dir: str | Path,
) -> tuple[tuple[GeometryFramePacketLogEntry, ...], GeometrySessionReceipt]:
    """Re-read every numeric packet and reject missing/tampered intermediate frames."""

    session = Path(session_dir)
    entries = read_geometry_packet_log(session / GEOMETRY_PACKET_LOG_BASENAME)
    receipt = read_geometry_session_receipt(
        session / GEOMETRY_PACKET_LOG_RECEIPT_BASENAME
    )
    verify_geometry_packet_log(entries, receipt)
    log_raw = _stable_regular_bytes(
        session / GEOMETRY_PACKET_LOG_BASENAME, label="geometry packet log"
    )
    if hashlib.sha256(log_raw).hexdigest() != receipt.packet_log_sha256:
        raise GeometryPacketLogError(
            "geometry packet log bytes do not match receipt"
        )
    observed_receipt = read_geometry_session_receipt(
        session / GEOMETRY_PACKET_LOG_RECEIPT_BASENAME
    )
    if observed_receipt != receipt:
        raise GeometryPacketLogError(
            "geometry packet log receipt changed during verification"
        )
    return entries, receipt


__all__ = [
    "GEOMETRY_FRAME_PACKET_LOG_ENTRY_SCHEMA",
    "GEOMETRY_CHANNEL_PACKET_SCHEMA",
    "GEOMETRY_INFERENCE_ID_PREFIX",
    "GEOMETRY_PACKET_LOG_HASH_CONTRACT",
    "GEOMETRY_PACKET_LOG_BASENAME",
    "GEOMETRY_PACKET_LOG_RECEIPT_BASENAME",
    "GEOMETRY_RECEIPT_AUTHORIZATION_BLOCKER",
    "GEOMETRY_SESSION_RECEIPT_SCHEMA",
    "INDEPENDENT_PATIENT_MODEL_CHANNEL",
    "PERCEPTION_OBSERVATION_CHANNEL",
    "GeometryChannelPacket",
    "GeometryChannelPacketError",
    "GeometryFramePacketLogEntry",
    "GeometryPacketLogError",
    "GeometrySessionReceipt",
    "geometry_channels_are_independent",
    "read_geometry_packet_log",
    "read_geometry_session_receipt",
    "verify_geometry_packet_log",
    "verify_geometry_packet_log_files",
    "write_geometry_packet_log_files",
]
