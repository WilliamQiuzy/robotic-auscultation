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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Protocol


class CollectionState(str, Enum):
    IDLE = "IDLE"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    READY = "READY"
    RECORDING = "RECORDING"
    FINALIZING = "FINALIZING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"
    ERROR = "ERROR"


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
            return {
                "state": self.state.value,
                "revision": self.revision,
                "action_ack": self.action_ack,
                "error": self.last_error,
                "session_id": self.session_id,
                "session_dir": str(self.session_dir) if self.session_dir else None,
                "capture_kind": self.CAPTURE_KIND,
                "training_eligible": False,
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

    def _start(self) -> dict:
        if self.diagnostic_blockers:
            raise _Rejected(
                "invalid_state",
                "diagnostic preflight failed: " + "; ".join(self.diagnostic_blockers),
            )
        retryable_error = (
            self.state is CollectionState.ERROR
            and not self._finalization_active
            and self._journal is None
        )
        if self.state in {
            CollectionState.COMPLETE,
            CollectionState.ABORTED,
        } or retryable_error:
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
            self.capture.start()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            _atomic_json(session_dir / "START_FAILURE.json", {
                "schema_version": "wam_diagnostic_start_failure_v1",
                "capture_mode": self.CAPTURE_KIND,
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
            capture_mode=self.CAPTURE_KIND,
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
