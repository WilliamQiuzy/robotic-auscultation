"""Session bundle replay (replay rig v2): recorded evidence -> sealed bundles.

Reconstructs real :class:`FrameEvidenceBundle` objects from a recorded
session's persisted artifacts so the production perception path
(``TrackerProcessor``'s production-evidence branch) can be re-driven offline
frame by frame — the standing pre-train gate: no perception train ships
without an offline replay over real session data.

Every reconstruction step re-runs a PRODUCTION component and fails closed:

1. control envelopes re-parse through ``parse_control_envelope_datagram``
   under the activated :class:`RobotStateContract` (identity + schema);
2. the exposure interval re-derives through the activated
   ``CameraExposureContract.for_frame`` and must reproduce the recorded
   frame-index ledger interval EXACTLY (nanosecond equality) — proving the
   latency artifact and host times used offline are the ones used live;
3. each frame's record bracket revalidates through the production
   ``_validate_streamed_frame_bracket`` (bracket completeness, gap bounds,
   recorded trajectory/action SHA-256, nominal interpolation identity);
4. the bundle re-seals through ``FrameEvidenceBundle.build`` whose
   ``__post_init__`` re-runs the full evidence-chain validation;
5. where the session recorded the sealed bundle ledger (model frames), the
   reconstructed canonical metadata can additionally be required to hash to
   the recorded ``canonical_sha256`` (byte-exact reconstruction proof; only
   meaningful when the mask is rebuilt with the semantics that were active
   at record time, e.g. ``depth_veto=False`` for pre-veto sessions).

Framework-free: camera pixels enter through an injected ``frame_loader`` so
the reconstruction, ordering, and fail-closed logic are unit-testable on any
host; the box-side driver supplies the RGB/depth decoders.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
)

from ausc.perception.robot.camera_exposure_evidence import (
    CameraExposureContract,
)
from ausc.perception.robot.frame_evidence_bundle import FrameEvidenceBundle
from ausc.perception.robot.robot_session_capture import (
    _validate_streamed_frame_bracket,
)
from ausc.perception.robot.robot_state_stream import (
    BracketedExactControlTrajectory,
    ExactControlRecord,
    RobotControlEnvelope,
    RobotStateContract,
    RobotTelemetryError,
    parse_control_envelope_datagram,
)


class SessionBundleReplayError(RuntimeError):
    """The session evidence cannot be reconstructed faithfully."""


# Session artifact names appear finalized, or suffixed while a session is
# still open / was aborted; replay accepts them in this strict order so a
# finalized artifact always wins over its in-progress shadow.
_ARTIFACT_SUFFIXES = ("", ".partial", ".inprogress")


def session_artifact_path(session_dir: Path, name: str) -> Path:
    session = Path(session_dir)
    for suffix in _ARTIFACT_SUFFIXES:
        candidate = session / (name + suffix)
        if candidate.is_file():
            return candidate
    raise SessionBundleReplayError(
        "session artifact is missing: %s" % name
    )


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                # A final line without its newline is a truncated write
                # (production finalize rejects exactly this); replaying a
                # silently shortened ledger would pass an incomplete
                # session off as covered.
                raise SessionBundleReplayError(
                    "%s line %d is truncated (missing trailing newline)"
                    % (path.name, line_number)
                )
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SessionBundleReplayError(
                    "%s line %d is not valid JSON" % (path.name, line_number)
                ) from exc
            if not isinstance(row, dict):
                raise SessionBundleReplayError(
                    "%s line %d is not a JSON object" % (path.name, line_number)
                )
            yield row


def iter_control_envelopes(
    session_dir: Path,
    contract: RobotStateContract,
) -> Iterator[RobotControlEnvelope]:
    """Re-parse every recorded envelope through the production parser."""
    path = session_artifact_path(session_dir, "robot_control_envelopes.jsonl")
    for row in iter_jsonl(path):
        try:
            yield parse_control_envelope_datagram(
                json.dumps(row).encode("utf-8"), contract,
            )
        except RobotTelemetryError as exc:
            raise SessionBundleReplayError(
                "recorded control envelope failed production parsing"
            ) from exc


def iter_frame_index_rows(session_dir: Path) -> Iterator[dict]:
    path = session_artifact_path(session_dir, "robot_frame_index.jsonl")
    return iter_jsonl(path)


class ExactRecordWindow:
    """Sliding tick-indexed window over the session's exact control records.

    Frame-index rows arrive in increasing tick order, so records older than
    the last consumed bracket are evicted; a bracket whose ticks are not all
    present is a hard error (the recording persisted an incomplete stream).
    """

    def __init__(self, envelopes: Iterable[RobotControlEnvelope]):
        self._envelopes = iter(envelopes)
        self._records: "OrderedDict[int, ExactControlRecord]" = OrderedDict()
        self._highest_tick: Optional[int] = None
        self._exhausted = False

    def _pull(self) -> bool:
        try:
            envelope = next(self._envelopes)
        except StopIteration:
            self._exhausted = True
            return False
        for record in envelope.records:
            tick = record.control_tick_seq
            if self._highest_tick is not None and tick <= self._highest_tick:
                # Publishers may retransmit an envelope boundary record;
                # identical resends are tolerated, regressions are not.
                existing = self._records.get(tick)
                if existing == record:
                    continue
                raise SessionBundleReplayError(
                    "exact control stream regressed at tick %d" % tick
                )
            self._records[tick] = record
            self._highest_tick = tick
        return True

    def bracket(
        self, first_tick: int, last_tick: int,
    ) -> Tuple[ExactControlRecord, ...]:
        if first_tick > last_tick:
            raise SessionBundleReplayError("bracket tick range is inverted")
        while (
            self._highest_tick is None or self._highest_tick < last_tick
        ) and not self._exhausted:
            self._pull()
        try:
            bracket = tuple(
                self._records[tick]
                for tick in range(first_tick, last_tick + 1)
            )
        except KeyError as exc:
            raise SessionBundleReplayError(
                "recorded exact control stream does not cover the bracket "
                "[%d, %d]" % (first_tick, last_tick)
            ) from exc
        for tick in list(self._records):
            if tick >= first_tick:
                break
            del self._records[tick]
        return bracket


@dataclass(frozen=True)
class ReconstructedFrameEvidence:
    """One faithfully reconstructed, fully revalidated evidence bundle."""

    frame_index: int
    bundle: FrameEvidenceBundle
    ledger_canonical_sha256: Optional[str]
    ledger_match: Optional[bool]


def reconstruct_session_bundles(
    *,
    frame_index_rows: Iterable[Mapping[str, object]],
    envelopes: Iterable[RobotControlEnvelope],
    contract: RobotStateContract,
    exposure_contract: CameraExposureContract,
    activated_identity: object,
    mask_provider: object,
    frame_loader: Callable[[Mapping[str, object]], object],
    depth_veto: bool = True,
    recorded_ledger_sha256: Optional[Mapping[int, str]] = None,
    require_ledger_match: bool = False,
    minimum_frames: int = 1,
) -> Iterator[ReconstructedFrameEvidence]:
    """Yield sealed production bundles for every recorded frame, in order.

    ``frame_loader(row)`` must return the recorded camera frame (RGB +
    aligned depth + host arrival time + device timestamp) for the row's
    ``frame_index``. ``depth_veto`` controls whether the rebuilt robot mask
    consumes the frame's aligned depth (live semantics) or reproduces the
    record-time pure-geometric mask (byte-fidelity comparisons against
    pre-veto sessions).
    """
    if require_ledger_match and recorded_ledger_sha256 is None:
        raise SessionBundleReplayError(
            "ledger matching requires the recorded ledger hashes"
        )
    window = ExactRecordWindow(envelopes)
    previous_index: Optional[int] = None
    reconstructed_count = 0
    for row in frame_index_rows:
        frame_index = int(row["frame_index"])
        if previous_index is not None and frame_index <= previous_index:
            raise SessionBundleReplayError(
                "frame index ledger is not strictly increasing"
            )
        previous_index = frame_index
        frame = frame_loader(row)
        if int(getattr(frame, "index", -1)) != frame_index:
            raise SessionBundleReplayError(
                "frame loader returned a mismatched frame index"
            )
        if getattr(frame, "frame", None) is None or getattr(
            frame, "depth", None
        ) is None:
            raise SessionBundleReplayError(
                "recorded frame is missing RGB or aligned depth pixels"
            )
        exposure = exposure_contract.for_frame(
            frame_index=frame_index,
            camera_host_t_s=float(frame.t),
            camera_device_t_ms=float(frame.device_t_ms),
        )
        if (
            exposure.exposure_lower_monotonic_ns
            != int(row["exposure_lower_ns"])
            or exposure.exposure_nominal_monotonic_ns
            != int(row["exposure_nominal_ns"])
            or exposure.exposure_upper_monotonic_ns
            != int(row["exposure_upper_ns"])
        ):
            raise SessionBundleReplayError(
                "re-derived exposure interval does not reproduce the "
                "recorded ledger interval for frame %d" % frame_index
            )
        bracket = window.bracket(
            int(row["exact_action_tick_first"]),
            int(row["exact_action_tick_last"]),
        )
        try:
            _validate_streamed_frame_bracket(row, bracket, contract)
        except Exception as exc:
            raise SessionBundleReplayError(
                "production bracket validation rejected frame %d"
                % frame_index
            ) from exc
        trajectory = BracketedExactControlTrajectory(
            exposure_lower_ns=exposure.exposure_lower_monotonic_ns,
            exposure_nominal_ns=exposure.exposure_nominal_monotonic_ns,
            exposure_upper_ns=exposure.exposure_upper_monotonic_ns,
            records=bracket,
        )
        mask_packet = mask_provider.build(
            exposure=exposure,
            trajectory=trajectory,
            aligned_depth=(frame.depth if depth_veto else None),
            depth_scale=(
                float(frame.depth_scale) if depth_veto else 1.0
            ),
        )
        bundle = FrameEvidenceBundle.build(
            camera_frame=frame,
            camera_id=str(row["camera_id"]),
            camera_profile_sha256=str(row["camera_profile_sha256"]),
            alignment_sha256=exposure.alignment_sha256,
            timing_artifact_sha256=str(row["timing_artifact_sha256"]),
            exposure=exposure,
            robot_trajectory=trajectory,
            robot_mask_packet=mask_packet,
            activated_identity=activated_identity,
        )
        ledger_sha: Optional[str] = None
        ledger_match: Optional[bool] = None
        if recorded_ledger_sha256 is not None:
            ledger_sha = recorded_ledger_sha256.get(frame_index)
            if ledger_sha is not None:
                reconstructed = hashlib.sha256(
                    bundle.canonical_metadata_bytes()
                ).hexdigest()
                ledger_match = reconstructed == ledger_sha
                if require_ledger_match and not ledger_match:
                    raise SessionBundleReplayError(
                        "reconstructed bundle metadata does not hash to the "
                        "recorded ledger for frame %d" % frame_index
                    )
        reconstructed_count += 1
        yield ReconstructedFrameEvidence(
            frame_index=frame_index,
            bundle=bundle,
            ledger_canonical_sha256=ledger_sha,
            ledger_match=ledger_match,
        )
    if reconstructed_count < minimum_frames:
        # A replay that "succeeds" over zero (or too few) frames is not a
        # gate at all — production finalize likewise hard-fails an empty
        # session (2026-08-29 adversarial review).
        raise SessionBundleReplayError(
            "session replay reconstructed %d frame(s); at least %d required"
            % (reconstructed_count, minimum_frames)
        )


def load_recorded_ledger_hashes(session_dir: Path) -> Dict[int, str]:
    """Map frame_index -> recorded bundle canonical SHA-256 (model frames).

    The inference stream records the sealed bundle ledger only for frames
    the model consumed; frames without a ledger simply cannot be
    byte-compared and reconstruct under the other four anchors alone.
    """
    hashes: Dict[int, str] = {}
    path = session_artifact_path(session_dir, "inference.jsonl")
    for row in iter_jsonl(path):
        # Recorded inference rows carry the ledger at the top level; the
        # live preview tracking dict nests it under "tracking".
        containers = [row]
        tracking = row.get("tracking")
        if isinstance(tracking, dict):
            containers.append(tracking)
        for container in containers:
            ledger = container.get("frame_evidence")
            if not isinstance(ledger, dict):
                continue
            digest = ledger.get("canonical_sha256")
            frame_index = row.get("frame_index", container.get("frame_index"))
            if isinstance(digest, str) and frame_index is not None:
                hashes[int(frame_index)] = digest
                break
    return hashes


__all__ = [
    "ExactRecordWindow",
    "ReconstructedFrameEvidence",
    "SessionBundleReplayError",
    "iter_control_envelopes",
    "iter_frame_index_rows",
    "iter_jsonl",
    "load_recorded_ledger_hashes",
    "reconstruct_session_bundles",
    "session_artifact_path",
]
