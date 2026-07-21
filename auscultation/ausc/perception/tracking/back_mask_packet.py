"""Immutable same-frame evidence for independent posterior segmentation.

The ten-point landmark model is deliberately not allowed to manufacture this
mask.  A caller must first snapshot the RGB-D camera frame, run the separately
released back-mask service on that exact RGB image, and build this packet before
the result is consumed by patient-pose or geometry code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Any, Mapping

import numpy as np

from ausc.perception.robot.frame_evidence_bundle import (
    FrameEvidenceBundleError,
    ImmutableCameraFrame,
)


BACK_MASK_PACKET_SCHEMA = "wam_back_mask_packet_v2"
BACK_MASK_PACKET_ID_PREFIX = "back_mask_packet-"
BACK_MASK_SOURCE = "independent_posterior_segmentation"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[a-z][a-z0-9_]*-[0-9a-f]{20}$")
_OPERATOR_GATE_DECISION_ID = re.compile(
    r"^operator_back_gate-[0-9a-f]{64}$"
)


class BackMaskPacketError(ValueError):
    """Back-mask evidence cannot be bound to the claimed camera frame."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BackMaskPacketError(message)


def _hard_readonly_bool(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.bool_)
    result = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.bool_
    ).reshape(contiguous.shape)
    result.setflags(write=False)
    return result


def _mask_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.uint8).tobytes(order="C")
    ).hexdigest()


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return int(value)
    if isinstance(value, (float, np.floating)):
        _require(
            math.isfinite(float(value)),
            "back-mask metadata contains a non-finite number",
        )
        return {"float64_hex": float(value).hex()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise BackMaskPacketError(
        f"back-mask metadata contains unsupported type: {type(value).__name__}"
    )


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True)
class BackMaskPacket:
    """One independent bool mask cryptographically bound to one RGB frame."""

    schema: str
    camera_frame: ImmutableCameraFrame
    frame_index: int
    camera_host_monotonic_ns: int
    camera_device_t_ms: float
    camera_id: str
    camera_profile_sha256: str
    alignment_sha256: str
    rgb_sha256: str
    depth_sha256: str
    operator_gate_decision_id: str
    back_mask_release_id: str
    back_mask_model_sha256: str
    source: str
    mask: np.ndarray
    mask_sha256: str
    masked_pixel_count: int
    status: str
    packet_id: str

    def _metadata(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "frame_index": self.frame_index,
            "camera_host_monotonic_ns": self.camera_host_monotonic_ns,
            "camera_device_t_ms": self.camera_device_t_ms,
            "camera_id": self.camera_id,
            "camera_profile_sha256": self.camera_profile_sha256,
            "alignment_sha256": self.alignment_sha256,
            "rgb_sha256": self.rgb_sha256,
            "depth_sha256": self.depth_sha256,
            "operator_gate_decision_id": self.operator_gate_decision_id,
            "back_mask_release_id": self.back_mask_release_id,
            "back_mask_model_sha256": self.back_mask_model_sha256,
            "source": self.source,
            "mask": {
                "shape": tuple(int(value) for value in self.mask.shape),
                "dtype": self.mask.dtype.str,
                "sha256": self.mask_sha256,
            },
            "masked_pixel_count": self.masked_pixel_count,
            "status": self.status,
        }

    def __post_init__(self) -> None:
        _require(
            self.schema == BACK_MASK_PACKET_SCHEMA,
            "back-mask packet schema is unsupported",
        )
        _require(
            isinstance(self.camera_frame, ImmutableCameraFrame),
            "immutable camera frame is required",
        )
        frame = self.camera_frame
        try:
            frame.__post_init__()
        except FrameEvidenceBundleError as exc:
            raise BackMaskPacketError(str(exc)) from exc
        _require(
            self.frame_index == frame.index,
            "back-mask frame index does not match the immutable camera frame",
        )
        _require(
            self.camera_host_monotonic_ns
            == frame.camera_host_monotonic_ns,
            "back-mask host timestamp does not match the immutable camera frame",
        )
        _require(
            isinstance(self.camera_device_t_ms, (int, float))
            and not isinstance(self.camera_device_t_ms, bool)
            and math.isfinite(float(self.camera_device_t_ms))
            and float(self.camera_device_t_ms)
            == float(frame.device_t_ms),
            "back-mask device timestamp does not match the immutable camera frame",
        )
        _require(
            self.camera_id == frame.camera_id,
            "back-mask camera identity does not match the immutable camera frame",
        )
        _require(
            self.camera_profile_sha256 == frame.camera_profile_sha256,
            "back-mask camera profile does not match the immutable camera frame",
        )
        _require(
            self.alignment_sha256 == frame.alignment_sha256,
            "back-mask alignment does not match the immutable camera frame",
        )
        _require(
            self.rgb_sha256 == frame.rgb_sha256,
            "back-mask RGB SHA-256 does not match the immutable camera frame",
        )
        _require(
            self.depth_sha256 == frame.depth_sha256,
            "back-mask depth SHA-256 does not match the immutable camera frame",
        )
        _require(
            isinstance(self.operator_gate_decision_id, str)
            and _OPERATOR_GATE_DECISION_ID.fullmatch(
                self.operator_gate_decision_id
            )
            is not None,
            "back-mask operator gate decision ID is invalid",
        )
        _require(
            isinstance(self.back_mask_release_id, str)
            and _RELEASE_ID.fullmatch(self.back_mask_release_id) is not None,
            "back-mask release identity is invalid",
        )
        _require(
            isinstance(self.back_mask_model_sha256, str)
            and _SHA256.fullmatch(self.back_mask_model_sha256) is not None,
            "back-mask model SHA-256 is invalid",
        )
        _require(
            self.source == BACK_MASK_SOURCE,
            "back-mask source must be independent of the landmark model",
        )
        _require(
            isinstance(self.mask, np.ndarray)
            and self.mask.dtype == np.bool_,
            "back-mask must use bool pixels",
        )
        _require(
            self.mask.ndim == 2
            and tuple(self.mask.shape) == tuple(frame.frame.shape[:2]),
            "back-mask shape does not match the immutable camera frame",
        )
        frozen = _hard_readonly_bool(self.mask)
        object.__setattr__(self, "mask", frozen)
        expected_mask_sha256 = _mask_sha256(frozen)
        _require(
            self.mask_sha256 == expected_mask_sha256,
            "back-mask mask SHA-256 does not match mask bytes",
        )
        expected_count = int(np.count_nonzero(frozen))
        _require(
            type(self.masked_pixel_count) is int
            and self.masked_pixel_count == expected_count,
            "back-mask pixel count does not match mask bytes",
        )
        expected_status = "OK" if expected_count > 0 else "NO_BACK"
        _require(
            self.status == expected_status,
            "back-mask status does not match mask content",
        )
        expected_packet_id = BACK_MASK_PACKET_ID_PREFIX + hashlib.sha256(
            _canonical_json(self._metadata())
        ).hexdigest()
        _require(
            self.packet_id == expected_packet_id,
            "back-mask packet ID does not match canonical evidence",
        )

    @classmethod
    def build(
        cls,
        *,
        camera_frame: ImmutableCameraFrame,
        mask: np.ndarray,
        back_mask_release_id: str,
        back_mask_model_sha256: str,
        operator_gate_decision_id: str,
    ) -> "BackMaskPacket":
        """Snapshot a separately produced mask for one immutable RGB frame."""

        _require(
            isinstance(camera_frame, ImmutableCameraFrame),
            "immutable camera frame is required",
        )
        _require(
            isinstance(mask, np.ndarray) and mask.dtype == np.bool_,
            "back-mask must use bool pixels",
        )
        _require(
            mask.ndim == 2
            and tuple(mask.shape) == tuple(camera_frame.frame.shape[:2]),
            "back-mask shape does not match the immutable camera frame",
        )
        frame_snapshot = replace(camera_frame)
        mask_snapshot = _hard_readonly_bool(mask)
        mask_digest = _mask_sha256(mask_snapshot)
        count = int(np.count_nonzero(mask_snapshot))
        values: dict[str, object] = {
            "schema": BACK_MASK_PACKET_SCHEMA,
            "camera_frame": frame_snapshot,
            "frame_index": frame_snapshot.index,
            "camera_host_monotonic_ns": (
                frame_snapshot.camera_host_monotonic_ns
            ),
            "camera_device_t_ms": frame_snapshot.device_t_ms,
            "camera_id": frame_snapshot.camera_id,
            "camera_profile_sha256": frame_snapshot.camera_profile_sha256,
            "alignment_sha256": frame_snapshot.alignment_sha256,
            "rgb_sha256": frame_snapshot.rgb_sha256,
            "depth_sha256": frame_snapshot.depth_sha256,
            "operator_gate_decision_id": operator_gate_decision_id,
            "back_mask_release_id": back_mask_release_id,
            "back_mask_model_sha256": back_mask_model_sha256,
            "source": BACK_MASK_SOURCE,
            "mask": mask_snapshot,
            "mask_sha256": mask_digest,
            "masked_pixel_count": count,
            "status": "OK" if count > 0 else "NO_BACK",
        }
        temporary = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(temporary, key, value)
        packet_id = BACK_MASK_PACKET_ID_PREFIX + hashlib.sha256(
            _canonical_json(temporary._metadata())
        ).hexdigest()
        return cls(packet_id=packet_id, **values)

    def canonical_metadata(self) -> dict[str, object]:
        """Return a detached serializable representation hashed by packet_id."""

        return self._metadata()


__all__ = [
    "BACK_MASK_PACKET_ID_PREFIX",
    "BACK_MASK_PACKET_SCHEMA",
    "BACK_MASK_SOURCE",
    "BackMaskPacket",
    "BackMaskPacketError",
]
