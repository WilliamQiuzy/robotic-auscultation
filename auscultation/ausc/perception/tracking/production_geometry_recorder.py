"""Session-level recorder for the production geometry write-side.

The live diagnostic capture already writes video, audio, force, depth and robot
artifacts.  The production delta is the geometry evidence, and this is the single
component that produces it.  Per frame it:

* persists the exact RGB-D + mask model inputs and appends the strictly-ordered
  model-input ledger (Step 2a: :class:`ProductionGeometryEvidenceCapture` +
  :func:`append_geometry_model_input_ledger`), and
* builds the dual-channel geometry packet — the perception observation and the
  independent patient-model prediction of the same frame's metric landmarks —
  and hash-chains it into the frame packet log.

:meth:`finalize` writes ``geometry_frame_packets.jsonl`` +
``geometry_frame_authenticity.json`` from the accumulated, chained entries and
returns the session receipt the finalizer replays against.

It is framework-free (no camera or model): the caller supplies the per-frame
:class:`PatientFrameEvidence` and both channels' ``points_m``/``covariances_m2``,
so the recorder's ordering, chaining, and ledger/packet alignment are fully
unit-tested with synthetic inputs.  The perception and independent points come
from two genuinely independent producers upstream; the recorder enforces that
they are declared independent but does not itself run either model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry_channel_packet import (
    INDEPENDENT_PATIENT_MODEL_CHANNEL,
    PERCEPTION_OBSERVATION_CHANNEL,
    GeometryChannelPacket,
    GeometryFramePacketLogEntry,
    GeometrySessionReceipt,
    write_geometry_packet_log_files,
)
from .patient_frame_evidence import PatientFrameEvidence
from .production_evidence_capture import (
    ProductionGeometryEvidenceCapture,
    append_geometry_model_input_ledger,
)


class ProductionGeometryRecorderError(RuntimeError):
    """The production geometry recorder cannot record a trustworthy session."""


@dataclass(frozen=True)
class GeometryChannelIdentity:
    """The producing model identity for one geometry channel."""

    producer_id: str
    release_id: str
    model_sha256: str


class ProductionGeometrySessionRecorder:
    """Compose per-frame model-input persistence with the geometry packet log."""

    def __init__(
        self,
        session_dir: "str | Path",
        *,
        intrinsics,
        geometry_profile,
        point_names: Sequence[str],
        perception_channel: GeometryChannelIdentity,
        independent_channel: GeometryChannelIdentity,
    ) -> None:
        if not isinstance(perception_channel, GeometryChannelIdentity) or not isinstance(
            independent_channel, GeometryChannelIdentity
        ):
            raise ProductionGeometryRecorderError("channel identities are required")
        # The two channels must be genuinely independent producers, or the frame
        # packet log's independence check would reject every entry.  Fail early.
        if (
            perception_channel.producer_id == independent_channel.producer_id
            or perception_channel.release_id == independent_channel.release_id
            or perception_channel.model_sha256 == independent_channel.model_sha256
        ):
            raise ProductionGeometryRecorderError(
                "perception and independent channels must be independent"
            )
        self._capture = ProductionGeometryEvidenceCapture(
            session_dir,
            intrinsics=intrinsics,
            geometry_profile=geometry_profile,
            point_names=point_names,
        )
        self._session_dir = Path(session_dir).absolute()
        self._point_names = tuple(point_names)
        self._perception = perception_channel
        self._independent = independent_channel
        self._entries: list[GeometryFramePacketLogEntry] = []
        self._last_entry: GeometryFramePacketLogEntry | None = None
        self._last_sequence = 0
        self._episode_id: str | None = None
        self._finalized = False

    @property
    def frame_count(self) -> int:
        return len(self._entries)

    @property
    def finalized(self) -> bool:
        return self._finalized

    def record_frame(
        self,
        patient_frame_evidence: PatientFrameEvidence,
        *,
        episode_id: str,
        sequence: int,
        perception_points_m: np.ndarray,
        perception_covariances_m2: np.ndarray,
        independent_points_m: np.ndarray,
        independent_covariances_m2: np.ndarray,
    ) -> None:
        """Persist one frame's model inputs, ledger line, and packet pair."""

        if self._finalized:
            raise ProductionGeometryRecorderError("recorder is already finalized")
        if not isinstance(episode_id, str) or not episode_id:
            raise ProductionGeometryRecorderError("episode id is required")
        if self._episode_id is None:
            self._episode_id = episode_id
        elif episode_id != self._episode_id:
            raise ProductionGeometryRecorderError("episode id changed mid-session")
        if type(sequence) is not int or sequence != self._last_sequence + 1:
            raise ProductionGeometryRecorderError(
                "sequence must be contiguous from 1: expected "
                f"{self._last_sequence + 1}, got {sequence!r}"
            )

        # Build both channel packets and the chained entry in memory first: no
        # disk write happens until the frame is proven well-formed, so a bad
        # frame can never leave an orphaned ledger line ahead of its packet.
        perception_packet = GeometryChannelPacket.build(
            channel=PERCEPTION_OBSERVATION_CHANNEL,
            producer_id=self._perception.producer_id,
            release_id=self._perception.release_id,
            model_sha256=self._perception.model_sha256,
            patient_frame_evidence=patient_frame_evidence,
            episode_id=episode_id,
            sequence=sequence,
            point_names=self._point_names,
            points_m=perception_points_m,
            covariances_m2=perception_covariances_m2,
        )
        independent_packet = GeometryChannelPacket.build(
            channel=INDEPENDENT_PATIENT_MODEL_CHANNEL,
            producer_id=self._independent.producer_id,
            release_id=self._independent.release_id,
            model_sha256=self._independent.model_sha256,
            patient_frame_evidence=patient_frame_evidence,
            episode_id=episode_id,
            sequence=sequence,
            point_names=self._point_names,
            points_m=independent_points_m,
            covariances_m2=independent_covariances_m2,
        )
        entry = GeometryFramePacketLogEntry.build(
            perception_observation_packet=perception_packet,
            independent_model_packet=independent_packet,
            previous_entry=self._last_entry,
        )

        # Disk side effects only after in-memory validation succeeds.  The
        # ledger reader's 1..N contiguity requirement is the fail-closed net if a
        # crash lands between the model-input write and the ledger append.
        record = self._capture.stage_model_input(
            patient_frame_evidence, episode_id=episode_id, sequence=sequence,
        )
        append_geometry_model_input_ledger(self._session_dir, record)

        self._entries.append(entry)
        self._last_entry = entry
        self._last_sequence = sequence

    def finalize(self) -> GeometrySessionReceipt:
        """Write and verify the geometry packet log; return the session receipt."""

        if self._finalized:
            raise ProductionGeometryRecorderError("recorder is already finalized")
        if len(self._entries) < 2:
            raise ProductionGeometryRecorderError(
                "a production geometry session requires at least two frames"
            )
        receipt = write_geometry_packet_log_files(self._session_dir, self._entries)
        self._finalized = True
        return receipt


__all__ = [
    "GeometryChannelIdentity",
    "ProductionGeometryRecorderError",
    "ProductionGeometrySessionRecorder",
]
