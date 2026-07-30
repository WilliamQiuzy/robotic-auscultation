"""Fail-closed control plane for the Linbox collection console.

This module deliberately owns no camera.  The single Linbox camera/inference loop
injects each immutable ``CameraFrame`` through :meth:`CollectionPlatform.ingest_frame`;
the exact same object is handed to the active diagnostic writer.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol


class CollectionState(str, Enum):
    IDLE = "IDLE"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    READY = "READY"
    RECORDING = "RECORDING"
    FINALIZING = "FINALIZING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"
    ERROR = "ERROR"


class CaptureMode(str, Enum):
    """The operational mode a control plane is constructed for.

    The mode is fixed at construction; it is *not* a runtime toggle.  It exists
    so a single Start authority (:meth:`CollectionPlatform.start_authorization`)
    can reason about what a session is allowed to do instead of the intent being
    split across an HTTP handler, a page button, and session allocation.
    """

    #: Live perception/preview only.  Recording can never be authorized.
    PREVIEW_ONLY = "PREVIEW_ONLY"
    #: Records immutable diagnostic evidence that can be kept and inspected but
    #: can *never* be promoted to a training corpus.
    DIAGNOSTIC_CAPTURE = "DIAGNOSTIC_CAPTURE"
    #: Records training-intent evidence produced by a real production writer.
    #: Requires production preflight to be clear before a session is allocated.
    PRODUCTION_CAPTURE = "PRODUCTION_CAPTURE"


#: Writer class names that are diagnostic-only by construction.  A production
#: capture can never be produced by one of these; the contract rejects the pair
#: before a session directory is ever allocated.
_DIAGNOSTIC_WRITER_IDENTITIES = frozenset({"DiagnosticPerceptionCapture"})


class WriterIdentityMismatch(RuntimeError):
    """The live capture writer does not match the contract's declared writer.

    The contract's ``writer_identity`` is only a *declaration*.  This is raised
    when the object actually returned by ``capture_factory`` is a different
    writer, so a diagnostic writer can never be recorded under a production
    contract by simply mislabelling the contract string.
    """


def _resolve_writer_identity(capture: "_Capture") -> str:
    """Return the capture's authoritative writer identity: its concrete class
    name.

    A *self-reported* ``writer_identity`` string is deliberately NOT trusted --
    any object could set one, so it is forgeable.  The class name is used
    instead, and when a ``writer_types`` registry is supplied the check binds to
    the exact class *object* (below), which cannot be forged by naming alone.
    """

    return type(capture).__name__


@dataclass(frozen=True)
class CaptureContract:
    """Immutable binding of a session's mode, writer, intent, and schemas.

    The contract is the single source of truth that a session's operational
    ``mode``, the ``capture_kind`` persisted into its manifest, whether it may
    ever promote (``training_intent``), and the ``writer_identity`` that
    produces it are mutually consistent.  It is validated at construction so an
    inconsistent triple (e.g. a production mode fronting a diagnostic writer)
    can never reach the Start gate.
    """

    mode: CaptureMode
    capture_kind: str
    training_intent: bool
    writer_identity: str
    manifest_schema: str
    finalization_schema: str

    def __post_init__(self) -> None:
        kind = str(self.capture_kind)
        is_production_kind = kind.startswith("production_")
        if self.mode is CaptureMode.PREVIEW_ONLY:
            if self.training_intent:
                raise ValueError(
                    "preview-only capture can never carry training intent"
                )
            if is_production_kind:
                raise ValueError(
                    "preview-only capture cannot claim a production capture_kind"
                )
        elif self.mode is CaptureMode.DIAGNOSTIC_CAPTURE:
            if self.training_intent:
                raise ValueError(
                    "diagnostic capture can never carry training intent"
                )
            if is_production_kind:
                raise ValueError(
                    "diagnostic capture cannot claim a production capture_kind"
                )
        elif self.mode is CaptureMode.PRODUCTION_CAPTURE:
            if not self.training_intent:
                raise ValueError(
                    "production capture must carry training intent"
                )
            if not is_production_kind:
                raise ValueError(
                    "production capture_kind must start with 'production_'"
                )
            if self.writer_identity in _DIAGNOSTIC_WRITER_IDENTITIES:
                raise ValueError(
                    "production capture cannot be produced by a diagnostic writer"
                )
        else:  # pragma: no cover - exhaustive over the enum
            raise ValueError(f"unknown capture mode: {self.mode!r}")

    @property
    def records(self) -> bool:
        """Whether this mode is allowed to record at all."""

        return self.mode is not CaptureMode.PREVIEW_ONLY

    @property
    def training_eligible(self) -> bool:
        """Whether a session under this contract may ever promote to training."""

        return self.mode is CaptureMode.PRODUCTION_CAPTURE and self.training_intent


#: The console's default: single-camera realtime perception writer, diagnostic
#: only, structurally un-promotable.
DIAGNOSTIC_PERCEPTION_CONTRACT = CaptureContract(
    mode=CaptureMode.DIAGNOSTIC_CAPTURE,
    capture_kind="diagnostic_perception",
    training_intent=False,
    writer_identity="DiagnosticPerceptionCapture",
    manifest_schema="wam_session_v2",
    finalization_schema="wam_promotion_v2",
)

#: Commissioning preview: same writer identity, but recording is never allowed.
PREVIEW_ONLY_CONTRACT = CaptureContract(
    mode=CaptureMode.PREVIEW_ONLY,
    capture_kind="diagnostic_perception_preview",
    training_intent=False,
    writer_identity="DiagnosticPerceptionCapture",
    manifest_schema="wam_session_v2",
    finalization_schema="wam_promotion_v2",
)


class _Capture(Protocol):
    def start(self) -> None: ...
    def record_event(self, payload: dict) -> None: ...
    def write_frame(self, frame, tracker_output: dict) -> None: ...  # noqa: ANN001
    def finalize(self) -> dict: ...
    def abort(self, reason: str) -> None: ...


_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")


def make_session_id(
    *,
    now_ns: int | None = None,
    uuid_value: uuid.UUID | None = None,
) -> str:
    """Return a human-readable UTC-nanosecond ID with a UUID collision guard."""

    value = time.time_ns() if now_ns is None else int(now_ns)
    if value < 0:
        raise ValueError("UTC nanoseconds must be non-negative")
    stamp = datetime.fromtimestamp(value // 1_000_000_000, tz=timezone.utc)
    stamp_text = stamp.strftime("%Y%m%dT%H%M%S") + f"{value % 1_000_000_000:09d}Z"
    unique = uuid.uuid4() if uuid_value is None else uuid_value
    return f"diagnostic_perception_{stamp_text}_{unique.hex[-12:]}"


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_session_directory(root: str | Path, session_id: str) -> Path:
    """Create one contained session directory with no-replace semantics."""

    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise ValueError(f"invalid session id: {session_id!r}")
    resolved_root = Path(root).expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    session = resolved_root / session_id
    if session.parent != resolved_root:
        raise ValueError("session id escapes configured session root")
    session.mkdir(mode=0o750, exist_ok=False)
    _fsync_directory(resolved_root)
    return session


def _atomic_json(path: Path, payload: dict) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def _read_json_object(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def reconcile_orphan_diagnostic_sessions(
    session_root: str | Path,
    *,
    wall_time_ns: Callable[[], int] = time.time_ns,
    video_recoverer: Callable[[Path], dict] | None = None,
) -> list[str]:
    """Fail closed any unsealed diagnostic session left active by a restart.

    Partial media are preserved for forensic recovery, but the function never
    manufactures a completion seal or training eligibility.
    """
    root = Path(session_root).expanduser().resolve()
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"diagnostic session root is not a real directory: {root}")
    if video_recoverer is None:
        from perception_capture import recover_orphan_diagnostic_video

        video_recoverer = recover_orphan_diagnostic_video
    reconciled: list[str] = []
    active_states = {"RECORDING", "FINALIZING"}
    active_statuses = {"recording", "finalizing"}
    for session in sorted(root.iterdir(), key=lambda path: path.name):
        if not session.name.startswith("diagnostic_perception_"):
            continue
        if session.is_symlink() or not session.is_dir():
            raise RuntimeError(f"unsafe diagnostic session entry: {session}")
        if (session / "DIAGNOSTIC_FINALIZATION.json").is_file():
            continue
        manifest_path = session / "manifest.json"
        lifecycle_path = session / "lifecycle.json"
        manifest = _read_json_object(manifest_path)
        lifecycle = _read_json_object(lifecycle_path)
        if (
            str(manifest.get("status", "")) not in active_statuses
            and str(lifecycle.get("state", "")) not in active_states
        ):
            continue
        recovered_ns = int(wall_time_ns())
        reason = "service restarted before diagnostic finalization completed"
        try:
            media_recovery = dict(video_recoverer(session))
        except Exception as exc:
            # A corrupt fragment must never prevent the control plane from
            # recording a durable fail-closed recovery decision.  Leave every
            # source byte in place for offline repair and surface the error.
            media_recovery = {
                "status": "video_recovery_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        manifest.update({
            "status": "diagnostic_aborted",
            "training_eligible": False,
            "production_finalization_permitted": False,
            "abort_reason": reason,
            "ended_utc_ns": recovered_ns,
        })
        lifecycle.update({
            "state": "ABORTED",
            "training_eligible": False,
            "finalization_active": False,
            "error": reason,
        })
        recovery = {
            "schema_version": "wam_diagnostic_restart_recovery_v1",
            "status": "diagnostic_aborted_on_restart",
            "training_eligible": False,
            "production_finalization_permitted": False,
            "reason": reason,
            "recovered_utc_ns": recovered_ns,
            "media_recovery": media_recovery,
        }
        _atomic_json(manifest_path, manifest)
        _atomic_json(lifecycle_path, lifecycle)
        _atomic_json(session / "RECOVERY.json", recovery)
        reconciled.append(str(session))
    return reconciled


class _DurableJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd = os.open(
            str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_APPEND, 0o640,
        )

    def append(self, payload: dict) -> None:
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        view = memoryview(raw)
        while view:
            written = os.write(self._fd, view)
            view = view[written:]
        os.fsync(self._fd)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class _Rejected(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CollectionPlatform:
    """Session-bound controls and lifecycle for diagnostic perception capture."""

    CAPTURE_KIND = "diagnostic_perception"

    def __init__(
        self,
        *,
        session_root: str | Path,
        point_names: Iterable[str],
        capture_factory: Callable[[Path], _Capture],
        contract: CaptureContract | None = None,
        writer_types: "Mapping[str, type] | None" = None,
        diagnostic_blockers: Iterable[str] = (),
        production_blockers: Iterable[str] = (),
        completion_validator: Callable[[dict, Path], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_time_ns: Callable[[], int] = time.time_ns,
        uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
    ) -> None:
        self.session_root = Path(session_root).expanduser().resolve()
        self.point_names = tuple(str(name) for name in point_names)
        if not self.point_names:
            raise ValueError("point_names cannot be empty")
        self.capture_factory = capture_factory
        self.contract = contract or DIAGNOSTIC_PERCEPTION_CONTRACT
        self.mode = self.contract.mode
        # Optional non-forgeable writer binding: identity string -> exact class
        # object.  When supplied, a session may only record if the live writer
        # *is* the registered class for the contract's identity (not merely a
        # class that shares its name).  When absent, the check falls back to the
        # class name, which is stronger than a self-reported attribute but not
        # forgery-proof; production callers must supply the registry.
        self._writer_types = dict(writer_types) if writer_types else None
        self.diagnostic_blockers = tuple(str(item) for item in diagnostic_blockers)
        self.production_blockers = tuple(str(item) for item in production_blockers)
        self.completion_validator = completion_validator
        self._clock = clock
        self._wall_time_ns = wall_time_ns
        self._uuid_factory = uuid_factory
        self._lock = threading.RLock()

        self.state = (
            CollectionState.PREFLIGHT_FAILED
            if self.diagnostic_blockers
            else CollectionState.READY
        )
        self.session_id: str | None = None
        self.session_dir: Path | None = None
        self.capture: _Capture | None = None
        self._journal: _DurableJournal | None = None
        self._finalization: dict | None = None
        self._finalize_thread: threading.Thread | None = None
        self._finalization_active = False
        self._end_requested = False
        # Whether the live writer's identity was verified against the contract
        # at capture creation, and the training eligibility the writer actually
        # reported at finalize.  The snapshot's ``training_eligible`` is gated by
        # both so a contract's declared intent can never be reported as real
        # eligibility unless a matching writer produced a matching result.
        self._capture_writer_verified = False
        self._finalized_training_eligible: bool | None = None
        self.point_index = 0
        self.attempt = 1
        self.completed: list[int] = []
        self.revision = 0
        self.action_ack: str | None = None
        self.last_error: str | None = None
        self.back_present = False
        self.initialized = False
        self.target_valid = False
        self.tracking_reason = "no tracking result"
        self.tracking_points: list[dict] = []
        self.frame_index: int | None = None
        self.frame_host_t: float | None = None
        self.frame_device_t: float | None = None

    def state_snapshot(self) -> dict:
        with self._lock:
            # Training eligibility is *realized*, never *declared*.  It is true
            # only when the session has reached COMPLETE AND the writer's own
            # finalize result was independently validated as eligible.  Every
            # other state — READY, RECORDING, FINALIZING, ERROR, ABORTED, a
            # failed Start, or a failed finalize — reports False.  The contract's
            # training_intent is only a *possibility*, never evidence of a
            # finished, verified session, so a mislabelled contract or a crashed
            # Start can never surface as eligibility.
            training_eligible = (
                self.state is CollectionState.COMPLETE
                and self._finalized_training_eligible is True
            )
            # Single Start authority: expose the exact verdict the control
            # plane will enforce so the HTTP layer and the page cannot compute a
            # divergent "can start" from raw fields.
            start_rejection = self.start_authorization()
            return {
                "state": self.state.value,
                "revision": self.revision,
                "action_ack": self.action_ack,
                "error": self.last_error,
                "session_id": self.session_id,
                "session_dir": str(self.session_dir) if self.session_dir else None,
                "capture_kind": self.contract.capture_kind,
                "operational_mode": self.mode.value,
                "training_eligible": training_eligible,
                "start_authorized": start_rejection is None,
                "start_rejection": start_rejection,
                "training_blockers": list(self.production_blockers),
                "diagnostic_blockers": list(self.diagnostic_blockers),
                "point_index": self.point_index,
                "point": self.point_names[self.point_index],
                "point_names": list(self.point_names),
                "attempt": self.attempt,
                "completed": list(self.completed),
                "back_present": self.back_present,
                "initialized": self.initialized,
                "target_valid": self.target_valid,
                "tracking_reason": self.tracking_reason,
                "points": [dict(point) for point in self.tracking_points],
                "frame_index": self.frame_index,
                "frame_host_t": self.frame_host_t,
                "frame_device_t": self.frame_device_t,
                "finalization_active": self._finalization_active,
                "finalization": (
                    None
                    if self._finalization is None
                    else json.loads(json.dumps(self._finalization))
                ),
            }

    def wait_for_terminal(self, timeout: float | None = None) -> dict:
        """Join the one finalizer thread for tests/shutdown without blocking state polling."""

        with self._lock:
            thread = self._finalize_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        return self.state_snapshot()

    def set_diagnostic_blockers(self, blockers: Iterable[str]) -> dict:
        """Update live camera/model preflight without mutating an active session."""
        with self._lock:
            normalized = tuple(str(item) for item in blockers if str(item))
            self.diagnostic_blockers = normalized
            if self.state in {CollectionState.READY, CollectionState.PREFLIGHT_FAILED}:
                self.state = (
                    CollectionState.PREFLIGHT_FAILED
                    if normalized
                    else CollectionState.READY
                )
                self.action_ack = (
                    "Diagnostic preflight ready"
                    if not normalized
                    else "Diagnostic preflight blocked: " + "; ".join(normalized)
                )
                self.revision += 1
            return self.state_snapshot()

    def control(self, action: object) -> dict:
        with self._lock:
            normalized = action.strip().lower() if isinstance(action, str) else ""
            handlers = {
                "start": self._start,
                "end": self._end,
                "stop": self._end,
                "confirm": self._confirm,
                "redo": self._redo,
                "next": lambda: self._move(1),
                "prev": lambda: self._move(-1),
            }
            if normalized not in handlers:
                return self._failure(
                    "invalid_action", f"unknown control action: {action}",
                )
            try:
                return handlers[normalized]()
            except _Rejected as exc:
                return self._failure(exc.code, str(exc))
            except Exception as exc:  # fail visibly; never return a false success
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.action_ack = None
                self.state = CollectionState.ERROR
                self._finalization_active = False
                self.revision += 1
                if self._journal is not None:
                    try:
                        self._journal.append({
                            "event": "action_failed",
                            "revision": self.revision,
                            "host_t": self._clock(),
                            "utc_ns": self._wall_time_ns(),
                            "state": self.state.value,
                            "session_id": self.session_id,
                            "action": normalized,
                            "error": self.last_error,
                        })
                    except Exception:
                        pass
                if self.capture is not None:
                    try:
                        self.capture.abort(self.last_error)
                    except Exception:
                        pass
                self._write_lifecycle()
                self._close_journal()
                return self._failure("action_failed", self.last_error)

    def _apply_tracking_locked(self, frame, tracking: dict) -> None:  # noqa: ANN001
        tracking = dict(tracking or {})
        self.back_present = bool(tracking.get("back_present", False))
        self.initialized = bool(tracking.get("initialized", self.back_present))
        self.tracking_reason = str(tracking.get("reason", ""))
        raw_points = tracking.get("points") or []
        self.tracking_points = [dict(point) for point in raw_points]
        active_name = self.point_names[self.point_index]
        active_present = any(
            str(point.get("name", "")) == active_name
            and str(point.get("status", "LOST")) != "LOST"
            for point in self.tracking_points
        )
        inferred_target = self.back_present and self.initialized and active_present
        # An upstream explicit False may tighten the gate.  Explicit True can
        # never bypass the independently inferred live-landmark requirement.
        requested_target = bool(tracking.get("target_valid", True))
        self.target_valid = inferred_target and requested_target
        self.frame_index = tracking.get("frame_index", getattr(frame, "index", None))
        self.frame_host_t = tracking.get("host_t", getattr(frame, "t", None))
        self.frame_device_t = tracking.get(
            "device_t", tracking.get("device_t_ms", getattr(frame, "device_t_ms", None)),
        )

    def update_tracking(self, tracking: dict, *, frame=None) -> None:
        """Update only the live control gate; never write a capture frame."""
        with self._lock:
            self._apply_tracking_locked(frame, tracking)
            self.revision += 1

    def ingest_frame(
        self, frame, tracking: dict, *, update_control_state: bool = True,
    ) -> None:  # noqa: ANN001
        """Accept the already-acquired frame shared by preview, inference and capture."""

        with self._lock:
            tracking = dict(tracking or {})
            if update_control_state:
                self._apply_tracking_locked(frame, tracking)
            self.revision += 1
            if self.state is CollectionState.RECORDING and self.capture is not None:
                try:
                    self.capture.write_frame(frame, tracking)
                except Exception as exc:
                    self.last_error = f"capture write failed: {type(exc).__name__}: {exc}"
                    self.state = CollectionState.ERROR
                    self._emit("capture_write_failed", error=self.last_error, mirror_capture=False)
                    try:
                        self.capture.abort(self.last_error)
                    finally:
                        self._write_lifecycle()
                    raise

    def abort(self, reason: str) -> dict:
        with self._lock:
            if self.state is CollectionState.ABORTED:
                return self._success(
                    ack=self.action_ack or "Diagnostic capture is already aborted",
                    idempotent=True,
                )
            if self.state not in {CollectionState.RECORDING, CollectionState.FINALIZING,
                                  CollectionState.ERROR}:
                return self._failure("invalid_state", f"cannot abort from {self.state.value}")
            primary_reason = (
                self.last_error
                if self.state is CollectionState.ERROR and self.last_error
                else str(reason)
            )
            # Capture may already have fail-closed itself inside write_frame.
            # Never mirror an abort event into a non-recording writer.
            self._emit(
                "abort_requested", reason=primary_reason, mirror_capture=False,
            )
            if self.capture is not None:
                try:
                    self.capture.abort(primary_reason)
                except Exception:
                    # Lifecycle ABORTED evidence is authoritative; retain the
                    # first capture error rather than losing it to a second abort.
                    pass
            self.state = CollectionState.ABORTED
            self.action_ack = f"Diagnostic capture aborted: {primary_reason}"
            self.last_error = primary_reason
            self.revision += 1
            self._emit("capture_aborted", reason=primary_reason, mirror_capture=False)
            self._write_lifecycle()
            self._close_journal()
            return self._success(ack=self.action_ack)

    def start_authorization(self) -> str | None:
        """Single Start authority for every mode.

        Returns a human-readable rejection reason, or ``None`` when a Start may
        proceed to allocate a session.  Every entry point that can begin a
        recording — the HTTP control handler, the page Start button, and the
        internal :meth:`_start` allocation — defers to this one decision so the
        per-mode contract cannot be bypassed by any single path.

        - ``PREVIEW_ONLY``: recording is never authorized.
        - ``DIAGNOSTIC_CAPTURE``: only diagnostic preflight must be clear; the
          session is structurally un-promotable regardless.
        - ``PRODUCTION_CAPTURE``: diagnostic *and* production preflight must be
          clear, the contract must be training-eligible, and the writer must be
          a real production writer — a diagnostic writer is refused before any
          session directory is allocated.
        """

        if self.mode is CaptureMode.PREVIEW_ONLY:
            return "preview-only mode never records"
        if self.diagnostic_blockers:
            return "diagnostic preflight failed: " + "; ".join(
                self.diagnostic_blockers
            )
        if self.mode is CaptureMode.PRODUCTION_CAPTURE:
            if not self.contract.training_eligible:
                return "production capture contract is not training-eligible"
            if self.contract.writer_identity in _DIAGNOSTIC_WRITER_IDENTITIES:
                return (
                    "production capture requires a real production writer, "
                    "not a diagnostic writer"
                )
            if self.production_blockers:
                return "production preflight failed: " + "; ".join(
                    self.production_blockers
                )
        # Startability is part of the authority, not a separate check: a Start
        # can only proceed from a startable state, so ``start_authorized`` is
        # never reported true while RECORDING/FINALIZING.  This is the same
        # startable set ``_start`` re-arms below (a finished/aborted session, or
        # a clean terminal error, resets to READY for a fresh no-replace start).
        if not self._is_startable_state():
            return f"cannot start from {self.state.value}"
        return None

    def _is_startable_state(self) -> bool:
        retryable_error = (
            self.state is CollectionState.ERROR
            and not self._finalization_active
            and self._journal is None
        )
        return (
            self.state
            in {
                CollectionState.READY,
                CollectionState.COMPLETE,
                CollectionState.ABORTED,
            }
            or retryable_error
        )

    def _start(self) -> dict:
        # start_authorization() is the single authority: it already rejects an
        # unstartable state, so this is also the exact gate the snapshot's
        # start_authorized/start_rejection reflect.
        rejection = self.start_authorization()
        if rejection is not None:
            raise _Rejected("invalid_state", rejection)
        if self.state in {
            CollectionState.COMPLETE,
            CollectionState.ABORTED,
            CollectionState.ERROR,
        }:
            # The prior immutable session remains on disk; only the in-memory
            # control plane is armed for a fresh, no-replace session.
            self.state = CollectionState.READY
            self.capture = None
            self._journal = None
        if self.state is not CollectionState.READY:
            raise _Rejected("invalid_state", f"cannot start from {self.state.value}")

        session_id = make_session_id(
            now_ns=self._wall_time_ns(), uuid_value=self._uuid_factory(),
        )
        session_dir = create_session_directory(self.session_root, session_id)
        # Bind the no-replace allocation immediately. If construction or start
        # fails, the outer control handler can still write lifecycle evidence to
        # the exact directory that was allocated for this attempt.
        self.session_id = session_id
        self.session_dir = session_dir
        self._journal = None
        self.capture = None
        self._finalization = None
        self._finalize_thread = None
        self._finalization_active = False
        self._end_requested = False
        self._capture_writer_verified = False
        self._finalized_training_eligible = None
        self.point_index = 0
        self.attempt = 1
        self.completed = []
        self.back_present = False
        self.initialized = False
        self.target_valid = False
        self.tracking_reason = "waiting for tracker"
        self.tracking_points = []
        self.frame_index = None
        self.frame_host_t = None
        self.frame_device_t = None
        self.last_error = None
        try:
            self._journal = _DurableJournal(session_dir / "control_journal.jsonl")
            self.capture = self.capture_factory(session_dir)
            # Bind the *declared* contract writer to the *actual* writer before
            # a single frame is recorded.  With a registry the binding is to the
            # exact class object (non-forgeable); without one it falls back to
            # the class name.  Either way a mismatch is fail-closed: the session
            # never enters RECORDING, so a diagnostic writer can never be
            # recorded under a production contract.
            if self._writer_types is not None:
                expected_type = self._writer_types.get(self.contract.writer_identity)
                identity_ok = (
                    expected_type is not None
                    and isinstance(self.capture, expected_type)
                )
            else:
                identity_ok = (
                    _resolve_writer_identity(self.capture)
                    == self.contract.writer_identity
                )
            if not identity_ok:
                raise WriterIdentityMismatch(
                    "capture writer %r does not match contract writer %r"
                    % (_resolve_writer_identity(self.capture),
                       self.contract.writer_identity)
                )
            self._capture_writer_verified = True
            self.capture.start()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            _atomic_json(session_dir / "START_FAILURE.json", {
                "schema_version": "wam_diagnostic_start_failure_v1",
                "capture_mode": self.contract.capture_kind,
                "status": "diagnostic_start_failed",
                "training_eligible": False,
                "production_finalization_permitted": False,
                "session_id": session_id,
                "error": error,
                "failed_utc_ns": self._wall_time_ns(),
            })
            if self._journal is not None:
                try:
                    self._journal.append({
                        "event": "start_failed",
                        "revision": self.revision,
                        "host_t": self._clock(),
                        "utc_ns": self._wall_time_ns(),
                        "state": self.state.value,
                        "session_id": self.session_id,
                        "action": "start",
                        "error": error,
                    })
                except Exception:
                    # START_FAILURE.json is the primary durable evidence. A
                    # damaged journal must not replace the original start error.
                    pass
            raise
        self.state = CollectionState.RECORDING
        self.action_ack = "Diagnostic recording started"
        self.revision += 1
        self._emit("capture_started")
        self._emit("attempt_started")
        self._write_lifecycle()
        return self._success(ack=self.action_ack, session_dir=str(session_dir))

    def _end(self) -> dict:
        if self._end_requested and self.state in {
            CollectionState.FINALIZING,
            CollectionState.COMPLETE,
            CollectionState.ERROR,
            CollectionState.ABORTED,
        }:
            return self._success(
                ack=self.action_ack or f"Diagnostic finalization is {self.state.value.lower()}",
                idempotent=True,
                accepted=False,
            )
        if self.state is not CollectionState.RECORDING:
            raise _Rejected("invalid_state", f"cannot end from {self.state.value}")
        if self.capture is None:
            raise RuntimeError("active capture is missing")

        self._emit("attempt_closed", outcome="ended_unconfirmed")
        self._end_requested = True
        self._finalization_active = True
        self.state = CollectionState.FINALIZING
        self.action_ack = "End accepted; diagnostic recording is finalizing"
        self.revision += 1
        self._emit("finalization_started")
        self._write_lifecycle()
        thread = threading.Thread(
            target=self._finalize_capture,
            name=f"diagnostic-finalize-{self.session_id}",
            daemon=False,
        )
        self._finalize_thread = thread
        thread.start()
        return self._success(ack=self.action_ack, accepted=True)

    def _finalize_capture(self) -> None:
        capture = self.capture
        if capture is None:
            self._finalization_failed(RuntimeError("active capture disappeared"))
            return
        try:
            result = capture.finalize()
            if not isinstance(result, dict):
                raise RuntimeError("diagnostic finalize result is not a mapping")
            if result.get("status") != "diagnostic_complete":
                raise RuntimeError("diagnostic finalize result has no complete status")
            if result.get("training_eligible") is not False:
                raise RuntimeError("diagnostic capture attempted to become training eligible")
            # Record what the writer actually finalized so the snapshot reports
            # the real eligibility, not the contract's declared intent.  The
            # current finalize path is diagnostic-only, so this is always False.
            self._finalized_training_eligible = bool(result.get("training_eligible"))
            if self.session_dir is None:
                raise RuntimeError("diagnostic session path disappeared before validation")
            if self.completion_validator is not None:
                self.completion_validator(result, self.session_dir)
        except Exception as exc:  # background failures must become durable, visible state
            self._finalization_failed(exc)
            return

        with self._lock:
            if self.state is not CollectionState.FINALIZING:
                return
            self._finalization = dict(result)
            self._finalization_active = False
            self.state = CollectionState.COMPLETE
            self.action_ack = "Diagnostic recording saved and verified (not training eligible)"
            self.last_error = None
            self.revision += 1
            self._emit("capture_completed", finalization=result, mirror_capture=False)
            self._write_lifecycle()
            self._close_journal()

    def _finalization_failed(self, exc: Exception) -> None:
        with self._lock:
            self.last_error = f"finalization failed: {type(exc).__name__}: {exc}"
            self.action_ack = "Diagnostic finalization failed"
            self._finalization_active = False
            self.state = CollectionState.ERROR
            self.revision += 1
            if self._journal is not None:
                try:
                    self._journal.append({
                        "event": "finalization_failed",
                        "revision": self.revision,
                        "host_t": self._clock(),
                        "utc_ns": self._wall_time_ns(),
                        "state": self.state.value,
                        "session_id": self.session_id,
                        "error": self.last_error,
                    })
                except Exception:
                    pass
            if self.capture is not None:
                try:
                    self.capture.abort(self.last_error)
                except Exception:
                    pass
            self._write_lifecycle()
            self._close_journal()

    def _require_tracked_attempt(self) -> None:
        if self.state is not CollectionState.RECORDING:
            raise _Rejected("not_recording", "control requires an active recording")
        if not self.back_present or not self.initialized or not self.target_valid:
            raise _Rejected("back_not_tracked", "a stable posterior back target is not tracked")

    def _redo(self) -> dict:
        if self.state is not CollectionState.RECORDING:
            raise _Rejected("not_recording", "redo requires an active recording")
        point = self.point_names[self.point_index]
        previous = self.attempt
        self._emit("attempt_closed", outcome="redo")
        if self.point_index in self.completed:
            self.completed.remove(self.point_index)
        self.attempt += 1
        self._emit("attempt_started")
        self.action_ack = f"Redo saved {point} attempt {previous}; attempt {self.attempt} is active"
        self.revision += 1
        self._write_lifecycle()
        return self._success(ack=self.action_ack)

    def _confirm(self) -> dict:
        self._require_tracked_attempt()
        point = self.point_names[self.point_index]
        previous = self.attempt
        self._emit("attempt_closed", outcome="confirmed")
        if self.point_index not in self.completed:
            self.completed.append(self.point_index)
        self.point_index = (self.point_index + 1) % len(self.point_names)
        self.attempt = 1
        self._emit("attempt_started")
        self.action_ack = (
            f"Confirmed {point} attempt {previous}; {self.point_names[self.point_index]} is active"
        )
        self.revision += 1
        self._write_lifecycle()
        return self._success(ack=self.action_ack)

    def _move(self, delta: int) -> dict:
        if self.state is not CollectionState.RECORDING:
            raise _Rejected("not_recording", "point navigation requires an active recording")
        self._emit("attempt_closed", outcome="skipped")
        self.point_index = (self.point_index + int(delta)) % len(self.point_names)
        self.attempt = 1
        self._emit("attempt_started")
        self.action_ack = f"Selected {self.point_names[self.point_index]} attempt 1"
        self.revision += 1
        self._write_lifecycle()
        return self._success(ack=self.action_ack)

    def _emit(self, event: str, *, mirror_capture: bool = True, **extra: object) -> dict:
        self.revision += 1
        payload = {
            "event": str(event),
            "revision": self.revision,
            "host_t": self._clock(),
            "utc_ns": self._wall_time_ns(),
            "state": self.state.value,
            "session_id": self.session_id,
            "point_index": self.point_index,
            "point": self.point_names[self.point_index],
            "attempt": self.attempt,
            "frame_index": self.frame_index,
            **extra,
        }
        if self._journal is not None:
            self._journal.append(payload)
        if mirror_capture and self.capture is not None:
            self.capture.record_event(payload)
        return payload

    def _write_lifecycle(self) -> None:
        if self.session_dir is None:
            return
        payload = self.state_snapshot()
        payload.update(
            schema_version="wam_collection_lifecycle_v1",
            capture_mode=self.contract.capture_kind,
            finalization=self._finalization,
        )
        _atomic_json(self.session_dir / "lifecycle.json", payload)

    def _close_journal(self) -> None:
        if self._journal is not None:
            self._journal.close()
            self._journal = None

    def _success(self, *, ack: str, **extra: object) -> dict:
        return {"ok": True, "state": self.state.value, "ack": ack, **extra}

    def _failure(self, code: str, message: str) -> dict:
        return {
            "ok": False,
            "code": str(code),
            "message": str(message),
            "state": self.state.value,
        }
