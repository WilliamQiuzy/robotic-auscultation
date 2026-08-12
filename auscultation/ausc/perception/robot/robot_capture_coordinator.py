"""Production ordering for left OpenArm telemetry and camera-frame evidence.

The coordinator is the single sink attached to :class:`RobotSocketReceiver`.
It continuously validates the controller stream, keeps the exact recent wire
records needed for Start preflight, and fans new records into one
``RobotSessionCapture`` only while a session is active.

It does not arm a camera or control the robot.  Instead, the collector must:

1. start this receiver and pass Start preflight;
2. allocate the session and call :meth:`start_session`;
3. wait for :meth:`authorize_camera_arm`;
4. arm the camera only after that method returns;
5. call :meth:`build_frame_evidence_inputs` before each frame is fanned out;
6. call :meth:`finish_session` after camera delivery has stopped.

Any telemetry/capture integrity failure is permanently latched.  Readiness
failures (for example, asking for camera-arm authorization before pre-roll is
complete) are retryable and do not contaminate the session.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
import time
from types import MappingProxyType
from typing import Callable, Deque, Mapping, Optional, Tuple

from .camera_exposure_evidence import (
    CameraExposureContract,
    CameraExposureEvidence,
)
from .robot_session_capture import (
    CapturedExactControlFrameEvidence,
    RobotCaptureError,
    RobotSessionCapture,
)
from .robot_socket_receiver import RobotSocketReceiver
from .robot_state_stream import (
    MAX_CONTROL_TICK_GAP_NS,
    CONTROL_ENVELOPE_SCHEMA_V4,
    BracketedExactControlTrajectory,
    ExactControlRecord,
    ExactControlStateStream,
    RobotControlEnvelope,
    RobotStateContract,
    RobotTelemetryError,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_CAMERA_TO_EXPOSURE_NS = 100_000_000
_MAX_EXPOSURE_INTERVAL_NS = 40_000_000


class RobotCaptureCoordinatorError(RuntimeError):
    """The production robot capture cannot safely continue."""


class RobotCaptureNotReady(RobotCaptureCoordinatorError):
    """A retryable ordering/readiness gate has not yet passed."""


@dataclass(frozen=True)
class RobotCameraArmPermit:
    """Proof that recorded post-Start telemetry covers the camera pre-roll."""

    session_dir: str
    pair: str
    authorized_monotonic_ns: int
    first_recorded_envelope_seq: int
    last_recorded_envelope_seq: int
    first_recorded_control_tick_seq: int
    last_recorded_control_tick_seq: int
    first_recorded_monotonic_ns: int
    last_recorded_monotonic_ns: int
    recorded_envelope_count: int
    recorded_control_tick_count: int
    minimum_preroll_ns: int


@dataclass(frozen=True)
class RobotFrameEvidenceInputs:
    """Camera exposure plus the exact persisted robot trajectory for fanout."""

    camera_exposure: CameraExposureEvidence
    robot_trajectory: BracketedExactControlTrajectory
    persisted_index: Mapping[str, object]


@dataclass(frozen=True)
class RobotLiveFrameEvidenceInputs:
    """Non-persisted preview exposure plus its exact robot trajectory."""

    camera_exposure: CameraExposureEvidence
    robot_trajectory: BracketedExactControlTrajectory


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class ProductionRobotCaptureCoordinator:
    """Fail-closed Start/frame/End coordinator for left can1 -> can3."""

    # Production collection is deliberately v4-only.  The lower-level parser
    # still understands v2/v3 for commissioning and historical diagnostics,
    # but neither schema proves that every follower motor is returning fresh
    # decoded STATE feedback.
    telemetry_schema = CONTROL_ENVELOPE_SCHEMA_V4
    telemetry_schemas = frozenset({CONTROL_ENVELOPE_SCHEMA_V4})

    def __init__(
        self,
        *,
        contract: RobotStateContract,
        camera_exposure_contract: CameraExposureContract,
        socket_path: object,
        runtime_dir: object = None,
        receiver_factory: Callable[..., object] = RobotSocketReceiver,
        capture_factory: Callable[..., object] = RobotSessionCapture,
        controller_process_attestation: object = None,
        deferred_controller_attestation: bool = False,
        process_inspector: object = None,
        abort_callback: Optional[Callable[[str], None]] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        preflight_min_envelopes: int = 11,
        preflight_duration_ns: int = 100_000_000,
        warmup_min_new_envelopes: int = 2,
    ) -> None:
        if not isinstance(contract, RobotStateContract):
            raise TypeError("contract must be RobotStateContract")
        if (
            contract.side != "left"
            or contract.controller_mode != "unilateral"
            or contract.leader_can_interface != "can1"
            or contract.follower_can_interface != "can3"
        ):
            raise ValueError(
                "production robot capture requires the left OpenArm "
                "can1 leader -> can3 follower pair"
            )
        for digest, label in (
            (
                contract.controller_source_manifest_sha256,
                "controller source manifest",
            ),
            (contract.follower_config_sha256, "follower config"),
            (contract.openarm_sdk_sha256, "OpenArm SDK"),
        ):
            if (
                not isinstance(digest, str)
                or _SHA256_RE.fullmatch(digest) is None
            ):
                raise ValueError(
                    f"production robot capture requires activated {label} SHA-256"
                )
        if not isinstance(camera_exposure_contract, CameraExposureContract):
            raise TypeError(
                "camera_exposure_contract must be CameraExposureContract"
            )
        camera_contract = camera_exposure_contract
        if camera_contract.camera_profile_sha256 != contract.camera_profile_sha256:
            raise ValueError(
                "camera profile does not match the activated robot contract"
            )
        if (
            camera_contract.camera_latency_sha256
            != contract.timing_artifact_sha256
        ):
            raise ValueError(
                "camera timing artifact does not match the activated robot contract"
            )
        for digest, label in (
            (camera_contract.camera_latency_sha256, "camera timing artifact"),
            (camera_contract.camera_profile_sha256, "camera profile"),
            (camera_contract.alignment_sha256, "camera alignment"),
        ):
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{label} SHA-256 is invalid")
        if (
            not isinstance(camera_contract.camera_id, str)
            or not camera_contract.camera_id
            # hardware timing methods: LED flash or the camera's host-synced hardware timestamp
            # (device_timer_host_sync); see camera_exposure_evidence._ACCEPTED_TIMING_METHODS.
            or camera_contract.measurement_method
            not in ("hardware_led_flash", "device_timer_host_sync")
            or not _is_int(camera_contract.n_samples)
            or camera_contract.n_samples < 20
        ):
            raise ValueError(
                "camera exposure contract is not hardware commissioned"
            )
        if (
            not _is_int(camera_contract.latency_ns)
            or not _is_int(camera_contract.latency_uncertainty_ns)
            or camera_contract.latency_ns <= 0
            or camera_contract.latency_uncertainty_ns <= 0
            or camera_contract.latency_ns
            <= camera_contract.latency_uncertainty_ns
            or camera_contract.latency_ns
            + camera_contract.latency_uncertainty_ns
            > _MAX_CAMERA_TO_EXPOSURE_NS
            or 2 * camera_contract.latency_uncertainty_ns
            > _MAX_EXPOSURE_INTERVAL_NS
        ):
            raise ValueError("camera exposure timing envelope is unsafe")
        if (
            not _is_int(preflight_min_envelopes)
            or preflight_min_envelopes < 2
            or not _is_int(preflight_duration_ns)
            or preflight_duration_ns <= 0
            or preflight_duration_ns > contract.buffer_duration_ns
        ):
            raise ValueError(
                "robot preflight must fit inside the telemetry buffer"
            )
        if (
            not _is_int(warmup_min_new_envelopes)
            or warmup_min_new_envelopes < 2
        ):
            raise ValueError(
                "post-Start warmup requires at least two new packets"
            )
        if not callable(receiver_factory):
            raise TypeError("receiver_factory must be callable")
        if not callable(capture_factory):
            raise TypeError("capture_factory must be callable")
        if abort_callback is not None and not callable(abort_callback):
            raise TypeError("abort_callback must be callable")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")

        self.contract = contract
        self.camera_exposure_contract = camera_contract
        self._capture_factory = capture_factory
        self._abort_callback = abort_callback
        self._clock_ns = clock_ns
        self._preflight_min_envelopes = preflight_min_envelopes
        self._preflight_duration_ns = preflight_duration_ns
        self._warmup_min_new_envelopes = warmup_min_new_envelopes
        self._minimum_preroll_ns = (
            camera_contract.latency_ns
            + camera_contract.latency_uncertainty_ns
        )

        self._lock = threading.RLock()
        self._preflight_stream = ExactControlStateStream(contract, absorb_bounded_holes=True)
        self._preflight_reseed_count = 0
        self._last_preflight_reseed_reason = None
        self._recent_wire: Deque[
            Tuple[RobotControlEnvelope, bytes]
        ] = deque()
        self._active_capture: Optional[RobotSessionCapture] = None
        self._active_session_dir: Optional[Path] = None
        self._active_first_envelope: Optional[RobotControlEnvelope] = None
        self._active_last_envelope: Optional[RobotControlEnvelope] = None
        self._active_first_record: Optional[ExactControlRecord] = None
        self._active_last_record: Optional[ExactControlRecord] = None
        self._active_envelope_count = 0
        self._active_control_tick_count = 0
        self._camera_arm_permit: Optional[RobotCameraArmPermit] = None
        self._pending_exposure: Optional[CameraExposureEvidence] = None
        self._last_frame_inputs: Optional[RobotFrameEvidenceInputs] = None
        self._last_summary: Optional[Mapping[str, object]] = None
        self._receiver_started = False
        self._fault_reason = ""
        self._abort_callback_invoked = False

        # The receiver is SINGLE-USE (its stop() permanently consumes it), but the
        # coordinator outlives bring-up attempts. Keep the factory + construction
        # arguments so ``renew_receiver`` can build a fresh receiver for each new
        # bring-up (safe-stop test retries, and real bring-up after a safe-stop).
        self._receiver_factory = receiver_factory
        self._receiver_kwargs = dict(
            socket_path=socket_path,
            runtime_dir=runtime_dir,
            controller_process_attestation=controller_process_attestation,
            deferred_controller_attestation=deferred_controller_attestation,
            process_inspector=process_inspector,
            preflight_min_envelopes=preflight_min_envelopes,
            preflight_duration_ns=preflight_duration_ns,
        )
        self._receiver = self._construct_receiver()

    def _construct_receiver(self) -> object:
        """Build and validate one fresh single-use receiver from the stored args."""
        contract = self.contract
        try:
            receiver = self._receiver_factory(
                sink=self,
                abort_callback=self._receiver_fault,
                # strict reference-provenance bound only while actually
                # recording; advisory otherwise (see robot_socket_receiver)
                reference_strict_supplier=lambda: bool(self.recording),
                **self._receiver_kwargs,
            )
        except Exception as exc:
            raise RobotCaptureCoordinatorError(
                f"robot telemetry receiver construction failed: {exc}"
            ) from exc
        if getattr(receiver, "contract", None) is not contract:
            try:
                receiver.stop()
            except Exception:
                pass
            raise RobotCaptureCoordinatorError(
                "robot telemetry receiver did not preserve the activated contract"
            )
        for method in (
            "start",
            "stop",
            "require_start_preflight",
            "health",
        ):
            if not callable(getattr(receiver, method, None)):
                try:
                    receiver.stop()
                except Exception:
                    pass
                raise RobotCaptureCoordinatorError(
                    f"robot telemetry receiver is missing {method}()"
                )
        return receiver

    def renew_receiver(self) -> None:
        """Replace a consumed single-use receiver ahead of a fresh bring-up attempt.

        ``RobotSocketReceiver.stop()`` permanently consumes the object, so a second
        bring-up on the same coordinator would fail at ``prepare_receiver_socket``
        ("cannot prepare a socket after reception has started"). Callers invoke this
        only from the bring-up path AFTER the prior controller is confirmed
        stopped/disabled; it refuses during an active capture. The preflight stream
        and fault latches reset so stale ticks or a dead attempt's fault can never
        satisfy or poison the new attempt.
        """
        with self._lock:
            if self._active_capture is not None:
                raise RobotCaptureCoordinatorError(
                    "cannot renew the telemetry receiver during an active capture"
                )
            old = self._receiver
            try:
                old.stop()
            except Exception:
                pass
            self._receiver_started = False
            self._fault_reason = ""
            self._abort_callback_invoked = False
            self._preflight_stream = ExactControlStateStream(self.contract, absorb_bounded_holes=True)
            self._recent_wire.clear()
            self._receiver = self._construct_receiver()

    @property
    def receiver(self) -> object:
        """Expose the read-only receiver handle for lifecycle/health integration."""

        return self._receiver

    @property
    def faulted(self) -> bool:
        with self._lock:
            return bool(self._fault_reason)

    @property
    def fault_reason(self) -> str:
        with self._lock:
            return self._fault_reason

    @property
    def recording(self) -> bool:
        with self._lock:
            capture = self._active_capture
            return bool(capture is not None and capture.recording)

    @property
    def camera_arm_authorized(self) -> bool:
        with self._lock:
            return self._camera_arm_permit is not None

    @property
    def last_summary(self) -> Optional[Mapping[str, object]]:
        with self._lock:
            return self._last_summary

    def _now_ns(self, value: Optional[int] = None) -> int:
        try:
            result = self._clock_ns() if value is None else value
        except Exception as exc:
            raise RobotCaptureCoordinatorError(
                f"CLOCK_MONOTONIC source failed: {exc}"
            ) from exc
        if not _is_int(result) or result <= 0:
            raise RobotCaptureCoordinatorError(
                "CLOCK_MONOTONIC time must be positive integer ns"
            )
        return result

    def _ensure_not_faulted_locked(self) -> None:
        if self._fault_reason:
            raise RobotCaptureCoordinatorError(
                f"robot capture coordinator is faulted: {self._fault_reason}"
            )

    def _invoke_abort_once(self, reason: str) -> None:
        callback = None
        with self._lock:
            if not self._abort_callback_invoked:
                self._abort_callback_invoked = True
                callback = self._abort_callback
        if callback is not None:
            try:
                callback(reason)
            except Exception:
                # The integrity fault remains latched even when the platform's
                # higher-level abort handler itself raises.
                pass

    def _latch_fault(self, reason: str) -> None:
        text = str(reason).strip() or "unknown robot capture fault"
        capture = None
        first_fault = False
        with self._lock:
            if not self._fault_reason:
                self._fault_reason = text
                first_fault = True
                capture = self._active_capture
        if not first_fault:
            return
        if capture is not None and not bool(
            getattr(capture, "faulted", False)
        ):
            try:
                capture.abort(text)
            except RobotCaptureError:
                pass
            except Exception:
                pass
        self._invoke_abort_once(text)

    def _receiver_fault(self, reason: str) -> None:
        self._latch_fault(f"robot telemetry receiver fault: {reason}")

    def prepare_receiver_socket(self) -> None:
        """Bind the telemetry socket before the OpenArm controller is launched."""

        with self._lock:
            self._ensure_not_faulted_locked()
            if self._receiver_started:
                raise RobotCaptureCoordinatorError(
                    "cannot prepare robot socket after reception starts"
                )
        prepare = getattr(self._receiver, "prepare_socket", None)
        if not callable(prepare):
            raise RobotCaptureCoordinatorError(
                "robot telemetry receiver lacks two-phase socket preparation"
            )
        try:
            prepare()
        except Exception as exc:
            try:
                self._receiver.stop()
            except Exception:
                pass
            self._latch_fault(
                f"robot telemetry socket preparation failed: {exc}"
            )
            raise RobotCaptureCoordinatorError(
                f"robot telemetry socket preparation failed: {exc}"
            ) from exc

    def attach_controller_process(self, attestation: object) -> None:
        """Attach the independently captured final controller exec identity."""

        with self._lock:
            self._ensure_not_faulted_locked()
            if self._receiver_started:
                raise RobotCaptureCoordinatorError(
                    "cannot attach controller after reception starts"
                )
        attach = getattr(
            self._receiver, "attach_controller_process", None
        )
        if not callable(attach):
            raise RobotCaptureCoordinatorError(
                "robot telemetry receiver lacks process attachment"
            )
        try:
            attach(attestation)
        except Exception as exc:
            try:
                self._receiver.stop()
            except Exception:
                pass
            self._latch_fault(
                f"robot controller process attachment failed: {exc}"
            )
            raise RobotCaptureCoordinatorError(
                f"robot controller process attachment failed: {exc}"
            ) from exc

    def start_receiver(self) -> None:
        """Start the credential-bound socket before the operator presses Start."""

        with self._lock:
            self._ensure_not_faulted_locked()
            if self._receiver_started:
                return
        try:
            self._receiver.start()
        except Exception as exc:
            try:
                self._receiver.stop()
            except Exception:
                pass
            self._latch_fault(f"robot telemetry receiver start failed: {exc}")
            raise RobotCaptureCoordinatorError(
                f"robot telemetry receiver start failed: {exc}"
            ) from exc
        with self._lock:
            self._receiver_started = True
            immediate_fault = self._fault_reason
        if immediate_fault:
            try:
                self._receiver.stop()
            finally:
                with self._lock:
                    self._receiver_started = False
            raise RobotCaptureCoordinatorError(
                f"robot telemetry receiver faulted during start: {immediate_fault}"
            )

    def stop_receiver(self) -> None:
        """Stop idle reception; stopping during capture is a fatal truncation."""

        with self._lock:
            active = self._active_capture is not None
            started = self._receiver_started
        if active:
            self._latch_fault(
                "robot telemetry receiver stopped during active capture"
            )
        if started:
            try:
                self._receiver.stop()
            except Exception as exc:
                self._latch_fault(
                    f"robot telemetry receiver stop failed: {exc}"
                )
                raise RobotCaptureCoordinatorError(
                    f"robot telemetry receiver stop failed: {exc}"
                ) from exc
            finally:
                with self._lock:
                    self._receiver_started = False

    def receiver_health(
        self, *, now_monotonic_ns: Optional[int] = None
    ) -> object:
        """Return the receiver's immutable health snapshot."""

        now_ns = self._now_ns(now_monotonic_ns)
        health = self._receiver.health(now_monotonic_ns=now_ns)
        if bool(getattr(self._receiver, "faulted", False)):
            reason = str(
                getattr(self._receiver, "fault_reason", "")
            ) or str(getattr(health, "error", "")) or "unknown receiver fault"
            self._receiver_fault(reason)
        return health

    def ingest_datagram(self, payload: bytes) -> RobotControlEnvelope:
        """Sink used by ``RobotSocketReceiver``; preserve exact wire bytes."""

        try:
            with self._lock:
                self._ensure_not_faulted_locked()
                if not isinstance(payload, bytes):
                    raise RobotTelemetryError(
                        "robot-state datagram must be bytes"
                    )
                if b"\n" in payload or b"\r" in payload:
                    raise RobotTelemetryError(
                        "robot-state datagram must be one JSONL-safe line"
                    )
                try:
                    envelope = self._preflight_stream.ingest_datagram(payload)
                except Exception as preflight_exc:
                    if self._active_capture is not None:
                        # Recording: the strict session stream is the
                        # authority and a preflight refusal must abort.
                        raise
                    # Advisory-only refusal (2026-08-11 incident, twice): the
                    # receiver validated and forwarded this datagram but the
                    # preflight stream refused it — a one-envelope state
                    # divergence that permanently latched the coordinator
                    # every few minutes of arm-up. Reseed the advisory stream
                    # and revalidate from this very datagram; a payload the
                    # fresh stream also refuses is genuinely malformed and
                    # latches below. History restarts honestly: the preflight
                    # continuity window is cleared, so bring-up/session
                    # preflight waits for a fresh continuous run.
                    self._preflight_stream = ExactControlStateStream(
                        self.contract, absorb_bounded_holes=True,
                    )
                    self._recent_wire.clear()
                    self._preflight_reseed_count += 1
                    reason = f"{type(preflight_exc).__name__}: {preflight_exc}"
                    if (
                        reason != self._last_preflight_reseed_reason
                        or self._preflight_reseed_count % 50 == 1
                    ):
                        print(
                            "[robot] advisory preflight stream reseed "
                            f"#{self._preflight_reseed_count} "
                            f"(receiver kept): {reason}",
                            flush=True,
                        )
                    self._last_preflight_reseed_reason = reason
                    envelope = self._preflight_stream.ingest_datagram(payload)
                capture = self._active_capture
                if capture is not None:
                    captured_envelope = capture.ingest_datagram(payload)
                    if captured_envelope != envelope:
                        raise RobotCaptureError(
                            "session capture changed the validated control envelope"
                        )
                    if self._active_first_envelope is None:
                        self._active_first_envelope = envelope
                        self._active_first_record = envelope.records[0]
                    self._active_last_envelope = envelope
                    self._active_last_record = envelope.records[-1]
                    self._active_envelope_count += 1
                    self._active_control_tick_count += len(envelope.records)

                self._recent_wire.append((envelope, payload))
                cutoff = (
                    envelope.records[-1].observation_monotonic_ns
                    - self.contract.buffer_duration_ns
                )
                while self._recent_wire and (
                    self._recent_wire[0][0]
                    .records[-1]
                    .observation_monotonic_ns
                    < cutoff
                    or len(self._recent_wire) > self.contract.max_packets
                ):
                    self._recent_wire.popleft()
                return envelope
        except RobotCaptureCoordinatorError:
            raise
        except Exception as exc:
            self._latch_fault(f"robot telemetry ingest failed: {exc}")
            raise RobotCaptureCoordinatorError(
                f"robot telemetry ingest failed: {exc}"
            ) from exc

    def _preflight_failure_locked(self, now_ns: int) -> str:
        if len(self._recent_wire) < self._preflight_min_envelopes:
            return "robot preflight has too few continuous v4 envelopes"
        envelopes = tuple(item[0] for item in self._recent_wire)
        records = tuple(
            record
            for envelope in envelopes
            for record in envelope.records
        )
        if any(
            envelope.schema != CONTROL_ENVELOPE_SCHEMA_V4
            for envelope in envelopes
        ):
            return (
                "robot production preflight requires per-motor feedback "
                "liveness schema v4"
            )
        if any(
            not record.action_send_status_verified for record in records
        ):
            return (
                "robot production preflight contains a SocketCAN TX enqueue "
                "failure"
            )
        if any(not record.feedback_rx_all_fresh for record in records):
            return (
                "robot production preflight contains stale or missing "
                "per-motor STATE feedback"
            )
        latest = records[-1]
        if now_ns < latest.observation_monotonic_ns:
            return "robot preflight query clock precedes telemetry"
        if (
            now_ns - latest.observation_monotonic_ns
            > self.contract.max_staleness_ns
        ):
            return "robot preflight telemetry is stale"
        if (
            latest.observation_monotonic_ns
            - records[0].observation_monotonic_ns
            < self._preflight_duration_ns
        ):
            return "robot preflight continuous duration is too short"
        first_envelope = envelopes[0]
        for left, right in zip(envelopes, envelopes[1:]):
            if (
                right.envelope_seq != left.envelope_seq + 1
                or right.publisher_monotonic_ns
                <= left.publisher_monotonic_ns
                or right.publisher_monotonic_ns
                - left.publisher_monotonic_ns
                > self.contract.max_gap_ns
                or right.process_identity
                != first_envelope.process_identity
            ):
                return (
                    "robot preflight is not one continuous 100 Hz "
                    "controller envelope stream"
                )
        for left, right in zip(records, records[1:]):
            if (
                right.control_tick_seq != left.control_tick_seq + 1
                or right.observation_monotonic_ns
                <= left.observation_monotonic_ns
                or right.observation_monotonic_ns
                - left.observation_monotonic_ns
                > MAX_CONTROL_TICK_GAP_NS
                or right.process_identity != records[0].process_identity
            ):
                return (
                    "robot preflight is not one continuous exact 1 kHz "
                    "control stream"
                )
        if any(
            record.side != "left"
            or record.controller_mode != "unilateral"
            or record.leader_can_interface != "can1"
            or record.follower_can_interface != "can3"
            or record.controller_source_manifest_sha256
            != self.contract.controller_source_manifest_sha256
            or record.follower_config_sha256
            != self.contract.follower_config_sha256
            or record.openarm_sdk_sha256
            != self.contract.openarm_sdk_sha256
            for record in records
        ):
            return "robot preflight is not left can1 leader -> can3 follower"
        return ""

    def start_session(
        self,
        session_dir: object,
        *,
        now_monotonic_ns: Optional[int] = None,
    ) -> None:
        """Open robot partials for an already allocated sensor session."""

        with self._lock:
            now_ns = self._now_ns(now_monotonic_ns)
            self._ensure_not_faulted_locked()
            if not self._receiver_started:
                raise RobotCaptureNotReady(
                    "robot telemetry receiver is not running"
                )
            if self._active_capture is not None:
                raise RobotCaptureCoordinatorError(
                    "a robot session capture is already active"
                )
        if bool(getattr(self._receiver, "faulted", False)):
            reason = str(
                getattr(self._receiver, "fault_reason", "")
            ) or "unknown receiver fault"
            self._receiver_fault(reason)
            raise RobotCaptureCoordinatorError(
                f"robot telemetry receiver is faulted: {reason}"
            )
        try:
            self._receiver.require_start_preflight(
                now_monotonic_ns=now_ns
            )
        except Exception as exc:
            if bool(getattr(self._receiver, "faulted", False)):
                reason = str(
                    getattr(self._receiver, "fault_reason", "")
                ) or str(exc)
                self._receiver_fault(reason)
                raise RobotCaptureCoordinatorError(
                    f"robot telemetry receiver is faulted: {reason}"
                ) from exc
            raise RobotCaptureNotReady(
                f"robot receiver preflight is not ready: {exc}"
            ) from exc

        try:
            with self._lock:
                # Re-sample after receiver preflight because telemetry may
                # have arrived while that independent health check ran.
                now_ns = self._now_ns(now_monotonic_ns)
                self._ensure_not_faulted_locked()
                failure = self._preflight_failure_locked(now_ns)
                if failure:
                    raise RobotCaptureNotReady(failure)
                path = Path(os.fspath(session_dir))
                capture = self._capture_factory(
                    path,
                    self.contract,
                    preflight_min_envelopes=(
                        self._preflight_min_envelopes
                    ),
                    preflight_duration_ns=self._preflight_duration_ns,
                )
                if not isinstance(capture, RobotSessionCapture):
                    required = (
                        "ingest_datagram",
                        "start",
                        "record_frame_evidence",
                        "finish",
                        "abort",
                    )
                    if any(
                        not callable(getattr(capture, name, None))
                        for name in required
                    ):
                        raise RobotCaptureCoordinatorError(
                            "capture factory returned an incompatible object"
                        )
                self._active_capture = capture
                self._active_session_dir = path
                self._active_first_envelope = None
                self._active_last_envelope = None
                self._active_first_record = None
                self._active_last_record = None
                self._active_envelope_count = 0
                self._active_control_tick_count = 0
                self._camera_arm_permit = None
                self._pending_exposure = None
                self._last_frame_inputs = None
                for _envelope, payload in tuple(self._recent_wire):
                    capture.ingest_datagram(payload)
                capture.start(now_monotonic_ns=now_ns)
                self._last_summary = None
        except RobotCaptureNotReady:
            raise
        except Exception as exc:
            self._latch_fault(f"robot session Start failed: {exc}")
            raise RobotCaptureCoordinatorError(
                f"robot session Start failed: {exc}"
            ) from exc

    def authorize_camera_arm(
        self,
        *,
        now_monotonic_ns: Optional[int] = None,
    ) -> RobotCameraArmPermit:
        """Authorize camera arm only after recorded post-Start pre-roll."""

        with self._lock:
            # Sample after acquiring the same lock used by ingestion.  Sampling
            # before the lock would let a newly ingested packet appear to come
            # from the future relative to this readiness query.
            now_ns = self._now_ns(now_monotonic_ns)
            self._ensure_not_faulted_locked()
            capture = self._active_capture
            if capture is None or not capture.recording:
                raise RobotCaptureNotReady(
                    "robot session partials are not recording"
                )
            if self._camera_arm_permit is not None:
                return self._camera_arm_permit
            first_envelope = self._active_first_envelope
            last_envelope = self._active_last_envelope
            first = self._active_first_record
            last = self._active_last_record
            if (
                first_envelope is None
                or last_envelope is None
                or first is None
                or last is None
                or self._active_envelope_count
                < self._warmup_min_new_envelopes
            ):
                raise RobotCaptureNotReady(
                    "robot post-start warmup has too few new recorded v4 envelopes"
                )
            if now_ns < last.observation_monotonic_ns:
                raise RobotCaptureNotReady(
                    "camera-arm query clock precedes robot telemetry"
                )
            if (
                now_ns - last.observation_monotonic_ns
                > self.contract.max_gap_ns
            ):
                raise RobotCaptureNotReady(
                    "camera-arm telemetry is not fresh within the 20 ms gap bound"
                )
            if (
                first.observation_monotonic_ns
                > now_ns - self._minimum_preroll_ns
            ):
                raise RobotCaptureNotReady(
                    "robot post-start camera pre-roll is not yet complete"
                )
            permit = RobotCameraArmPermit(
                session_dir=os.fspath(self._active_session_dir),
                pair="left:can1->can3",
                authorized_monotonic_ns=now_ns,
                first_recorded_envelope_seq=(
                    first_envelope.envelope_seq
                ),
                last_recorded_envelope_seq=last_envelope.envelope_seq,
                first_recorded_control_tick_seq=(
                    first.control_tick_seq
                ),
                last_recorded_control_tick_seq=last.control_tick_seq,
                first_recorded_monotonic_ns=(
                    first.observation_monotonic_ns
                ),
                last_recorded_monotonic_ns=(
                    last.observation_monotonic_ns
                ),
                recorded_envelope_count=self._active_envelope_count,
                recorded_control_tick_count=(
                    self._active_control_tick_count
                ),
                minimum_preroll_ns=self._minimum_preroll_ns,
            )
            self._camera_arm_permit = permit
            return permit

    def build_live_frame_evidence_inputs(
        self,
        *,
        frame_index: int,
        camera_host_t_s: float,
        camera_device_t_ms: float,
        now_monotonic_ns: Optional[int] = None,
    ) -> RobotLiveFrameEvidenceInputs:
        """Query preview evidence without opening or contaminating a session."""

        try:
            exposure = self.camera_exposure_contract.for_frame(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                camera_device_t_ms=camera_device_t_ms,
            )
            with self._lock:
                now_ns = self._now_ns(now_monotonic_ns)
                self._ensure_not_faulted_locked()
                if not self._receiver_started:
                    raise RobotCaptureNotReady(
                        "robot telemetry receiver is not running"
                    )
                if self._active_capture is not None:
                    raise RobotCaptureNotReady(
                        "recording frames must use persisted robot evidence"
                    )
                failure = self._preflight_failure_locked(now_ns)
                if failure:
                    raise RobotCaptureNotReady(failure)
                if now_ns < exposure.camera_host_monotonic_ns:
                    raise RobotCaptureNotReady(
                        "preview query clock precedes camera host arrival"
                    )
                envelopes = tuple(
                    envelope for envelope, _payload in self._recent_wire
                )
                if not envelopes:
                    raise RobotCaptureNotReady(
                        "preview robot telemetry buffer is empty"
                    )
                first = envelopes[0].records[0]
                last = envelopes[-1].records[-1]
                if (
                    first.observation_monotonic_ns
                    > exposure.exposure_lower_monotonic_ns
                ):
                    raise RobotCaptureNotReady(
                        "preview exposure lower bracket is unavailable"
                    )
                if (
                    last.observation_monotonic_ns
                    < exposure.exposure_upper_monotonic_ns
                ):
                    raise RobotCaptureNotReady(
                        "robot telemetry is still waiting for the preview "
                        "exposure upper bracket"
                    )
                trajectory = self._preflight_stream.query_exposure_interval(
                    exposure_lower_ns=(
                        exposure.exposure_lower_monotonic_ns
                    ),
                    exposure_nominal_ns=(
                        exposure.exposure_nominal_monotonic_ns
                    ),
                    exposure_upper_ns=(
                        exposure.exposure_upper_monotonic_ns
                    ),
                    now_monotonic_ns=now_ns,
                    camera_profile_sha256=(
                        exposure.camera_profile_sha256
                    ),
                    timing_artifact_sha256=(
                        exposure.camera_latency_sha256
                    ),
                )
                return RobotLiveFrameEvidenceInputs(
                    camera_exposure=exposure,
                    robot_trajectory=trajectory,
                )
        except RobotCaptureNotReady:
            raise
        except RobotTelemetryError as exc:
            raise RobotCaptureNotReady(
                f"live robot frame bracket is not ready: {exc}"
            ) from exc
        except (ValueError, TypeError) as exc:
            raise RobotCaptureCoordinatorError(
                f"live camera evidence is invalid: {exc}"
            ) from exc

    def build_frame_evidence_inputs(
        self,
        *,
        frame_index: int,
        camera_host_t_s: float,
        camera_device_t_ms: float,
        now_monotonic_ns: Optional[int] = None,
    ) -> RobotFrameEvidenceInputs:
        """Derive exposure, persist its exact bracket, and return fanout inputs."""

        try:
            exposure = self.camera_exposure_contract.for_frame(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                camera_device_t_ms=camera_device_t_ms,
            )
            with self._lock:
                # See authorize_camera_arm(): the query timestamp and stream
                # snapshot must be ordered by the ingestion lock.
                now_ns = self._now_ns(now_monotonic_ns)
                self._ensure_not_faulted_locked()
                capture = self._active_capture
                permit = self._camera_arm_permit
                if capture is None or not capture.recording:
                    raise RobotCaptureCoordinatorError(
                        "robot session capture is not recording"
                    )
                if permit is None:
                    raise RobotCaptureCoordinatorError(
                        "camera arm has not been authorized"
                    )
                if (
                    self._pending_exposure is not None
                    and exposure != self._pending_exposure
                ):
                    raise RobotCaptureCoordinatorError(
                        "the pending camera frame must be retried before any later frame"
                    )
                if (
                    exposure.camera_host_monotonic_ns
                    < permit.authorized_monotonic_ns
                ):
                    raise RobotCaptureCoordinatorError(
                        "camera frame arrived before camera arm authorization"
                    )
                if now_ns < exposure.camera_host_monotonic_ns:
                    raise RobotCaptureCoordinatorError(
                        "frame query clock precedes camera host arrival"
                    )
                if (
                    now_ns - exposure.camera_host_monotonic_ns
                    > self.contract.max_staleness_ns
                ):
                    raise RobotCaptureCoordinatorError(
                        "camera frame was not bracketed within the live staleness bound"
                    )
                first = self._active_first_record
                last = self._active_last_record
                if (
                    first is None
                    or first.observation_monotonic_ns
                    > exposure.exposure_lower_monotonic_ns
                ):
                    raise RobotCaptureCoordinatorError(
                        "camera exposure begins before recorded post-Start robot telemetry"
                    )
                if (
                    last is None
                    or last.observation_monotonic_ns
                    < exposure.exposure_upper_monotonic_ns
                ):
                    # The exposure already happened, but a legal 20 ms robot
                    # packet phase can leave its final bracket in flight for a
                    # few milliseconds.  Do not fan out or contaminate the
                    # capture; the collector must retry this same frame.
                    self._pending_exposure = exposure
                    raise RobotCaptureNotReady(
                        "robot telemetry is still waiting for the frame exposure upper bracket"
                    )
                captured: CapturedExactControlFrameEvidence = (
                    capture.record_frame_evidence(
                        camera_id=exposure.camera_id,
                        frame_index=exposure.frame_index,
                        exposure_lower_ns=(
                            exposure.exposure_lower_monotonic_ns
                        ),
                        exposure_nominal_ns=(
                            exposure.exposure_nominal_monotonic_ns
                        ),
                        exposure_upper_ns=(
                            exposure.exposure_upper_monotonic_ns
                        ),
                        now_monotonic_ns=now_ns,
                        camera_profile_sha256=(
                            exposure.camera_profile_sha256
                        ),
                        timing_artifact_sha256=(
                            exposure.camera_latency_sha256
                        ),
                    )
                )
                inputs = RobotFrameEvidenceInputs(
                    camera_exposure=exposure,
                    robot_trajectory=captured.trajectory,
                    persisted_index=MappingProxyType(dict(captured.index)),
                )
                self._pending_exposure = None
                self._last_frame_inputs = inputs
                return inputs
        except RobotCaptureNotReady:
            raise
        except RobotCaptureCoordinatorError as exc:
            self._latch_fault(str(exc))
            raise
        except (RobotCaptureError, RobotTelemetryError, ValueError, TypeError) as exc:
            self._latch_fault(f"robot frame evidence failed: {exc}")
            raise RobotCaptureCoordinatorError(
                f"robot frame evidence failed: {exc}"
            ) from exc
        except Exception as exc:
            self._latch_fault(f"robot frame evidence failed: {exc}")
            raise RobotCaptureCoordinatorError(
                f"robot frame evidence failed: {exc}"
            ) from exc

    def finish_session(
        self,
        *,
        now_monotonic_ns: Optional[int] = None,
    ) -> Mapping[str, object]:
        """Finalize only after the last camera bracket can be revalidated."""

        with self._lock:
            # Prevent an in-flight receiver update from overtaking the End
            # query clock and causing a false clock-rollback failure.
            now_ns = self._now_ns(now_monotonic_ns)
            self._ensure_not_faulted_locked()
            capture = self._active_capture
            if capture is None or not capture.recording:
                raise RobotCaptureNotReady(
                    "robot session capture is not recording"
                )
            if self._pending_exposure is not None:
                raise RobotCaptureNotReady(
                    "robot capture cannot End with a pending camera frame bracket"
                )
            last_inputs = self._last_frame_inputs
            if last_inputs is None:
                raise RobotCaptureNotReady(
                    "robot capture cannot End without a camera frame"
                )
            if (
                now_ns
                < last_inputs.camera_exposure.exposure_upper_monotonic_ns
            ):
                raise RobotCaptureNotReady(
                    "End precedes the last camera exposure upper bound"
                )
            try:
                summary = capture.finish(now_monotonic_ns=now_ns)
            except Exception as exc:
                failure = exc
            else:
                immutable = MappingProxyType(dict(summary))
                self._last_summary = immutable
                self._active_capture = None
                self._active_session_dir = None
                self._active_first_envelope = None
                self._active_last_envelope = None
                self._active_first_record = None
                self._active_last_record = None
                self._active_envelope_count = 0
                self._active_control_tick_count = 0
                self._camera_arm_permit = None
                self._pending_exposure = None
                self._last_frame_inputs = None
                return immutable
        self._latch_fault(f"robot session End failed: {failure}")
        raise RobotCaptureCoordinatorError(
            f"robot session End failed: {failure}"
        ) from failure

    def abort_session(self, reason: object) -> bool:
        """Close one incomplete sidecar without poisoning idle reception.

        A camera, audio, or operator abort invalidates the whole sensor
        session, but it does not imply that the independently attested
        controller process or telemetry socket is corrupt.  Keep the receiver
        alive so a later, newly allocated session can pass preflight again.
        RobotSessionCapture intentionally raises RobotCaptureError from
        ``abort()`` even after a clean close; that exception is the aborted
        session's terminal evidence, not a coordinator integrity fault.
        """

        text = str(reason).strip()
        if not text:
            raise ValueError("robot session abort reason is required")
        with self._lock:
            capture = self._active_capture
            if capture is None:
                return False
            try:
                capture.abort(text)
            except RobotCaptureError:
                pass
            finally:
                self._active_capture = None
                self._active_session_dir = None
                self._active_first_envelope = None
                self._active_last_envelope = None
                self._active_first_record = None
                self._active_last_record = None
                self._active_envelope_count = 0
                self._active_control_tick_count = 0
                self._camera_arm_permit = None
                self._pending_exposure = None
                self._last_frame_inputs = None
                self._last_summary = None
            return True

    def __enter__(self) -> "ProductionRobotCaptureCoordinator":
        self.start_receiver()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop_receiver()


__all__ = [
    "ProductionRobotCaptureCoordinator",
    "RobotCameraArmPermit",
    "RobotCaptureCoordinatorError",
    "RobotCaptureNotReady",
    "RobotFrameEvidenceInputs",
    "RobotLiveFrameEvidenceInputs",
]
