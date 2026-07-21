"""Single-frame evidence boundary for patient pose and geometry.

This wrapper is the only object that should cross from the collection fanout
into the independent patient-pose provider.  It joins a complete RGB-D/robot
frame bundle with a separately released back mask and rejects any cross-frame
association before raw arrays are exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

from ausc.perception.robot.frame_evidence_bundle import (
    FrameEvidenceBundle,
    FrameEvidenceBundleError,
)
from ausc.perception.tracking.back_mask_packet import (
    BackMaskPacket,
    BackMaskPacketError,
)


PATIENT_FRAME_EVIDENCE_SCHEMA = "wam_patient_frame_evidence_v2"
PATIENT_FRAME_EVIDENCE_ID_PREFIX = "patient_frame_evidence-"


class PatientFrameEvidenceError(ValueError):
    """Camera, robot, or back-mask evidence is not the same physical frame."""


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _identity_metadata(
    *,
    frame_bundle: FrameEvidenceBundle,
    back_mask_packet: BackMaskPacket,
) -> dict[str, object]:
    return {
        "schema": PATIENT_FRAME_EVIDENCE_SCHEMA,
        "frame_bundle_id": frame_bundle.bundle_id,
        "back_mask_packet_id": back_mask_packet.packet_id,
        "frame_index": frame_bundle.camera_frame.index,
        "camera_host_monotonic_ns": (
            frame_bundle.camera_frame.camera_host_monotonic_ns
        ),
        "camera_id": frame_bundle.camera_frame.camera_id,
        "camera_profile_sha256": (
            frame_bundle.camera_frame.camera_profile_sha256
        ),
        "alignment_sha256": frame_bundle.camera_frame.alignment_sha256,
        "rgb_sha256": frame_bundle.camera_frame.rgb_sha256,
        "depth_sha256": frame_bundle.camera_frame.depth_sha256,
        "back_mask_depth_sha256": back_mask_packet.depth_sha256,
        "operator_gate_decision_id": (
            back_mask_packet.operator_gate_decision_id
        ),
        "robot_mask_sha256": frame_bundle.robot_mask_packet.mask_sha256,
        "back_mask_sha256": back_mask_packet.mask_sha256,
    }


@dataclass(frozen=True)
class PatientFrameEvidence:
    """Exact RGB-D, robot mask, and independent back mask for one frame."""

    schema: str
    frame_bundle: FrameEvidenceBundle
    back_mask_packet: BackMaskPacket
    evidence_id: str

    def __post_init__(self) -> None:
        if self.schema != PATIENT_FRAME_EVIDENCE_SCHEMA:
            raise PatientFrameEvidenceError(
                "patient frame evidence schema is unsupported"
            )
        if not isinstance(self.frame_bundle, FrameEvidenceBundle):
            raise PatientFrameEvidenceError("frame evidence bundle is missing")
        if not isinstance(self.back_mask_packet, BackMaskPacket):
            raise PatientFrameEvidenceError("back-mask packet is missing")
        try:
            self.frame_bundle.__post_init__()
            self.back_mask_packet.__post_init__()
        except (FrameEvidenceBundleError, BackMaskPacketError) as exc:
            raise PatientFrameEvidenceError(str(exc)) from exc

        camera = self.frame_bundle.camera_frame
        back = self.back_mask_packet
        comparisons = (
            ("frame index", camera.index, back.frame_index),
            (
                "host timestamp",
                camera.camera_host_monotonic_ns,
                back.camera_host_monotonic_ns,
            ),
            ("camera identity", camera.camera_id, back.camera_id),
            (
                "camera profile",
                camera.camera_profile_sha256,
                back.camera_profile_sha256,
            ),
            ("alignment", camera.alignment_sha256, back.alignment_sha256),
            ("RGB", camera.rgb_sha256, back.rgb_sha256),
            ("depth", camera.depth_sha256, back.depth_sha256),
        )
        for label, expected, observed in comparisons:
            if observed != expected:
                raise PatientFrameEvidenceError(
                    f"back-mask {label} does not match frame evidence bundle"
                )
        if tuple(back.mask.shape) != tuple(
            self.frame_bundle.robot_mask_packet.mask.shape
        ):
            raise PatientFrameEvidenceError(
                "back and robot mask shapes do not match"
            )
        expected_id = PATIENT_FRAME_EVIDENCE_ID_PREFIX + hashlib.sha256(
            _canonical_json(
                _identity_metadata(
                    frame_bundle=self.frame_bundle,
                    back_mask_packet=back,
                )
            )
        ).hexdigest()
        if self.evidence_id != expected_id:
            raise PatientFrameEvidenceError(
                "patient frame evidence ID does not match bound evidence"
            )

    @classmethod
    def build(
        cls,
        *,
        frame_bundle: FrameEvidenceBundle,
        back_mask_packet: BackMaskPacket,
    ) -> "PatientFrameEvidence":
        if not isinstance(frame_bundle, FrameEvidenceBundle):
            raise PatientFrameEvidenceError("frame evidence bundle is missing")
        if not isinstance(back_mask_packet, BackMaskPacket):
            raise PatientFrameEvidenceError("back-mask packet is missing")
        try:
            frame_bundle.__post_init__()
            back_mask_packet.__post_init__()
        except (FrameEvidenceBundleError, BackMaskPacketError) as exc:
            raise PatientFrameEvidenceError(str(exc)) from exc
        metadata = _identity_metadata(
            frame_bundle=frame_bundle,
            back_mask_packet=back_mask_packet,
        )
        return cls(
            schema=PATIENT_FRAME_EVIDENCE_SCHEMA,
            frame_bundle=frame_bundle,
            back_mask_packet=back_mask_packet,
            evidence_id=PATIENT_FRAME_EVIDENCE_ID_PREFIX
            + hashlib.sha256(_canonical_json(metadata)).hexdigest(),
        )

    @property
    def index(self) -> int:
        return self.frame_bundle.camera_frame.index

    @property
    def t(self) -> float:
        return self.frame_bundle.camera_frame.t

    @property
    def rgb(self) -> np.ndarray:
        return self.frame_bundle.camera_frame.frame

    @property
    def aligned_depth(self) -> np.ndarray:
        return self.frame_bundle.camera_frame.depth

    @property
    def back_mask(self) -> np.ndarray:
        return self.back_mask_packet.mask

    @property
    def robot_mask(self) -> np.ndarray:
        return self.frame_bundle.robot_mask_packet.mask

    def canonical_metadata(self) -> dict[str, object]:
        return _identity_metadata(
            frame_bundle=self.frame_bundle,
            back_mask_packet=self.back_mask_packet,
        )


__all__ = [
    "PATIENT_FRAME_EVIDENCE_ID_PREFIX",
    "PATIENT_FRAME_EVIDENCE_SCHEMA",
    "PatientFrameEvidence",
    "PatientFrameEvidenceError",
]
