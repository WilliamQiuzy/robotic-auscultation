"""Bounded, fail-closed Hall-force recording sidecar.

``ForceCaptureStream`` owns exactly one :class:`force_sensor.ForceSensor`.  The
serial reader remains open for live monitoring, while Start/End only attach and
detach a bounded disk spool.  Every accepted firmware sample is submitted from
the native reader thread; the recorder never polls ``ForceSensor.read()`` and
therefore cannot silently duplicate or skip intermediate samples.

The success contract is deliberately stronger than the legacy ``log_csv``
option.  ``force.csv.inprogress`` and ``force_capture.json.inprogress`` are
durable crash evidence.  End publishes ``force.csv`` and ``force_capture.json``
atomically, then writes ``FORCE_CAPTURE_COMPLETE.json`` last.  Consumers must
call :func:`verify_force_capture`; final-looking files without that marker are
not a successful capture.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import queue
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

try:  # supports both package imports and the existing direct-path scripts
    from .force_sensor import ForceCalibration, ForceSample, ForceSensor
except ImportError:  # pragma: no cover - exercised by legacy direct import tests
    from force_sensor import ForceCalibration, ForceSample, ForceSensor


SCHEMA_VERSION = "wam_hall_force_capture_v1"
COMPLETE_SCHEMA_VERSION = "wam_hall_force_capture_complete_v1"
SIGNAL_AUTHENTICITY_SCHEMA_VERSION = "wam_force_signal_authenticity_v1"
SIGNAL_AUTHENTICITY_POLICY_VERSION = "force_signal_conservative_v3"
CALIBRATION_ARTIFACT_SCHEMA_VERSION = "wam_force_calibration_json_v1"
COMPLETE_NAME = "FORCE_CAPTURE_COMPLETE.json"
CSV_NAME = "force.csv"
CSV_INPROGRESS_NAME = "force.csv.inprogress"
METADATA_NAME = "force_capture.json"
STATE_NAME = "force_capture.json.inprogress"
CALIBRATION_ARTIFACT_NAME = "force_calibration.json"
CSV_COLUMNS = (
    "host_t_s",
    "raw",
    "hall_v",
    "displacement_mm",
    "force_n",
    "status",
    "device_t_ms",
)
_NATIVE_SAMPLE_STATUS = {
    "OK", "WARN", "SCREENED_LIMIT", "HARD_LIMIT",
}
_SAFE_CAPTURE_STATUS = {"OK", "WARN"}
_SENTINEL = object()
_AUTHENTICITY_POLICY = {
    "distinct_value_count_cap": 1_024,
    "minimum_evidence_duration_s": 5.0,
    "minimum_evidence_samples": 250,
    "freeze_min_duration_s": 5.0,
    "freeze_min_samples": 250,
    "conversion_min_duration_s": 5.0,
    "conversion_min_samples": 250,
    "conversion_min_raw_span": 64,
    "conversion_min_distinct_raw": 16,
    "near_constant_min_duration_s": 5.0,
    "near_constant_min_samples": 250,
    "near_constant_min_raw_span": 16,
    "near_constant_min_force_span_n": 0.01,
    "near_zero_max_abs_force_n": 0.02,
    "calibration_hall_v_tolerance": 0.000051,
    "calibration_displacement_tolerance_mm": 0.00051,
    "calibration_force_tolerance_n": 0.00051,
}


class ForceCaptureError(RuntimeError):
    """The current Hall-force recording cannot be sealed without data loss."""


class ForceCaptureVerificationError(ForceCaptureError):
    """A published Hall-force sidecar failed its immutable contract."""


@dataclass(frozen=True)
class ForceCaptureSummary:
    path: str
    t_start: float
    t_end: float
    n_samples: int
    status: str
    sha256: str
    device_t_start_ms: float
    device_t_end_ms: float
    calibration_sha256: str
    device_identity_sha256: str
    signal_authenticity: dict[str, object]


@dataclass(frozen=True)
class _CsvAudit:
    n_samples: int
    t_start: float
    t_end: float
    device_t_start_ms: float
    device_t_end_ms: float
    signal_authenticity: dict[str, object]


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("Hall-force artifact write made no progress")
        view = view[written:]


def _atomic_bytes(path: Path, raw: bytes) -> None:
    temp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(temp), flags, 0o640)
    try:
        _write_all(descriptor, raw)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temp.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    try:
        os.replace(temp, path)
        _fsync_directory(path.parent)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: object) -> bytes:
    raw = _canonical_json_bytes(payload)
    _atomic_bytes(path, raw)
    return raw


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _json_identity(value: Mapping[str, object], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty JSON object")
    copied = dict(value)
    try:
        raw = _canonical_json_bytes(copied)
        decoded = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc
    if not isinstance(decoded, dict) or not decoded:
        raise ValueError(f"{label} must be a non-empty JSON object")
    return decoded


def _validate_calibration_identity(value: Mapping[str, object]) -> dict[str, object]:
    identity = _json_identity(value, "calibration_identity")
    _require_sha256(identity.get("sha256"), "calibration_identity.sha256")
    if type(identity.get("calibrated")) is not bool:
        raise ValueError("calibration_identity.calibrated must be a JSON boolean")
    source = identity.get("source_name")
    if not isinstance(source, str) or not source or "/" in source or "\\" in source:
        raise ValueError("calibration_identity.source_name must be a file name")
    artifact_schema = identity.get("artifact_schema_version")
    release_id = identity.get("release_id")
    if (artifact_schema is None) != (release_id is None):
        raise ValueError(
            "calibration_identity artifact schema and release_id must appear together"
        )
    if artifact_schema is not None:
        if artifact_schema != CALIBRATION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("calibration_identity artifact schema is invalid")
        suffix = f"-{identity['sha256'][:20]}"
        if (
            not isinstance(release_id, str)
            or len(release_id) <= len(suffix)
            or not release_id.endswith(suffix)
        ):
            raise ValueError(
                "calibration_identity release_id is not content-addressed"
            )
    return identity


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ForceCaptureError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ForceCaptureError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise ForceCaptureError(f"{label} is not finite")
    return result


def _format_float(value: float) -> str:
    result = float(value)
    if not math.isfinite(result):
        raise ForceCaptureError("Hall-force sample contains a non-finite value")
    return format(result, ".17g")


def _sample_row(sample: ForceSample) -> bytes:
    if type(sample.raw) is not int or sample.raw < 0:
        raise ForceCaptureError("Hall-force sample raw ADC must be a nonnegative integer")
    if sample.status not in _NATIVE_SAMPLE_STATUS:
        raise ForceCaptureError(
            f"Hall-force native sample has an invalid status: {sample.status!r}"
        )
    device = "" if sample.device_t_ms is None else _format_float(sample.device_t_ms)
    values = (
        _format_float(sample.t),
        str(sample.raw),
        _format_float(sample.hall_v),
        _format_float(sample.displacement_mm),
        _format_float(sample.force_n),
        sample.status,
        device,
    )
    return (",".join(values) + "\n").encode("utf-8")


class ForceCaptureRecorder:
    """One bounded disk spool for one Start/End interval.

    ``submit`` is safe for the serial reader callback: it only validates the
    immutable sample and performs a non-blocking bounded queue insertion.  If
    storage cannot keep up, overflow is latched and End fails closed; no sample
    is silently dropped from a successful capture.
    """

    def __init__(
        self,
        session_dir: str | Path,
        *,
        t0: float | None,
        calibration_identity: Mapping[str, object],
        device_identity: Mapping[str, object],
        queue_samples: int = 2_048,
        checkpoint_samples: int = 100,
        checkpoint_interval_s: float = 1.0,
        max_device_gap_ms: float = 50.0,
        max_host_gap_ms: float = 50.0,
        calibration_artifact: bytes | None = None,
    ) -> None:
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.t0 = (
            None if t0 is None else _finite(t0, "force capture Start timestamp")
        )
        self.calibration_identity = _validate_calibration_identity(calibration_identity)
        self.calibration_artifact: bytes | None = None
        self.calibration: ForceCalibration | None = None
        if calibration_artifact is not None:
            if not isinstance(calibration_artifact, bytes) or not calibration_artifact:
                raise ValueError("calibration_artifact must be non-empty bytes")
            if (
                self.calibration_identity.get("artifact_schema_version")
                != CALIBRATION_ARTIFACT_SCHEMA_VERSION
            ):
                raise ValueError(
                    "calibration_artifact requires a versioned calibration identity"
                )
            if _sha256_bytes(calibration_artifact) != self.calibration_identity["sha256"]:
                raise ValueError("calibration_artifact SHA-256 does not match identity")
            try:
                parsed_calibration = ForceCalibration.from_bytes(calibration_artifact)
            except ValueError as exc:
                raise ValueError("calibration_artifact is invalid") from exc
            if parsed_calibration.calibrated is not self.calibration_identity["calibrated"]:
                raise ValueError("calibration_artifact calibrated flag does not match identity")
            self.calibration_artifact = bytes(calibration_artifact)
            self.calibration = parsed_calibration
        self.device_identity = _json_identity(device_identity, "device_identity")
        self.calibration_sha256 = str(self.calibration_identity["sha256"])
        self.device_identity_sha256 = _sha256_bytes(
            _canonical_json_bytes(self.device_identity)
        )
        if type(queue_samples) is not int or queue_samples < 2:
            raise ValueError("force queue_samples must be an integer >= 2")
        if type(checkpoint_samples) is not int or checkpoint_samples <= 0:
            raise ValueError("force checkpoint_samples must be a positive integer")
        self.checkpoint_samples = checkpoint_samples
        self.checkpoint_interval_s = _finite(
            checkpoint_interval_s, "force checkpoint_interval_s"
        )
        self.max_device_gap_ms = _finite(
            max_device_gap_ms, "force max_device_gap_ms"
        )
        self.max_host_gap_ms = _finite(max_host_gap_ms, "force max_host_gap_ms")
        if self.checkpoint_interval_s <= 0:
            raise ValueError("force checkpoint_interval_s must be > 0")
        if self.max_device_gap_ms <= 0 or self.max_host_gap_ms <= 0:
            raise ValueError("force clock gap limits must be > 0")

        self.csv_inprogress_path = self.session_dir / CSV_INPROGRESS_NAME
        self.csv_path = self.session_dir / CSV_NAME
        self.state_path = self.session_dir / STATE_NAME
        self.metadata_path = self.session_dir / METADATA_NAME
        self.complete_path = self.session_dir / COMPLETE_NAME
        self.calibration_artifact_path = (
            self.session_dir / CALIBRATION_ARTIFACT_NAME
        )
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_samples)
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._started = False
        self._accepting = False
        self._overflow = False
        self._overflow_t: float | None = None
        self._worker_error: BaseException | None = None
        self._first_reader_error: tuple[BaseException, float] | None = None
        self._last_prebind_reader_error: tuple[BaseException, float] | None = None
        self._unsafe_failure: tuple[str, float] | None = None
        self._last_prebind_unsafe: tuple[str, float] | None = None
        self._written_samples = 0
        self._durable_samples = 0
        self._checkpoint_count = 0

    @property
    def queue_capacity(self) -> int:
        return self._queue.maxsize

    def _state_payload(self, *, status: str) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "training_eligible": False,
            "t0_monotonic": self.t0,
            "durable_n_samples": self._durable_samples,
            "queue_capacity": self.queue_capacity,
            "max_device_gap_ms": self.max_device_gap_ms,
            "max_host_gap_ms": self.max_host_gap_ms,
            "calibration": self.calibration_identity,
            "calibration_artifact": self._calibration_artifact_claim(),
            "device_identity": self.device_identity,
            "device_identity_sha256": self.device_identity_sha256,
        }

    def _calibration_artifact_claim(self) -> dict[str, object] | None:
        if self.calibration_artifact is None:
            return None
        return {
            "schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
            "path": CALIBRATION_ARTIFACT_NAME,
            "sha256": self.calibration_identity["sha256"],
            "bytes": len(self.calibration_artifact),
            "release_id": self.calibration_identity["release_id"],
        }

    def start(self) -> None:
        if self.calibration_identity["calibrated"] is not True:
            raise ForceCaptureError(
                "formal Hall-force capture requires calibration_identity.calibrated=true"
            )
        with self._state_lock:
            if self._started:
                raise RuntimeError("Hall-force recorder is already started")
            self._started = True
            self._accepting = True
        self.session_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.csv_inprogress_path,
            self.csv_path,
            self.state_path,
            self.metadata_path,
            self.complete_path,
            self.calibration_artifact_path,
        ):
            if path.exists() or path.is_symlink():
                with self._state_lock:
                    self._accepting = False
                raise FileExistsError(f"Hall-force capture artifact already exists: {path}")
        if self.calibration_artifact is not None:
            _atomic_bytes(self.calibration_artifact_path, self.calibration_artifact)
        _atomic_json(
            self.state_path,
            self._state_payload(
                status="recording" if self.t0 is not None else "prearmed"
            ),
        )
        self._thread = threading.Thread(
            target=self._run, name="wam-force-spool", daemon=True,
        )
        self._thread.start()

    def submit(self, sample: ForceSample) -> None:
        if not isinstance(sample, ForceSample):
            with self._state_lock:
                self._overflow = True
                self._overflow_t = time.monotonic()
            return
        with self._state_lock:
            if not self._accepting or self._overflow:
                return
            sample_t = float(sample.t)
            # P0-4: force is an advisory physician reference. An over-threshold
            # ("unsafe") reading is exactly the datum the physician needs -- record it
            # like any other sample: NEVER stop the recorder, NEVER drop it. The
            # occurrence is noted only so the summary can honestly report it.
            if sample.status not in _SAFE_CAPTURE_STATUS:
                if self.t0 is None:
                    self._last_prebind_unsafe = (sample.status, sample_t)
                elif sample_t >= self.t0:
                    self._unsafe_failure = (sample.status, sample_t)
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            with self._state_lock:
                self._overflow = True
                self._overflow_t = float(sample.t)

    def report_reader_error(self, error: BaseException, *, t: float) -> None:
        observed = _finite(t, "serial reader error timestamp")
        with self._state_lock:
            if not self._accepting:
                return
            if self.t0 is None:
                self._last_prebind_reader_error = (error, observed)
            elif observed >= self.t0 and self._first_reader_error is None:
                self._first_reader_error = (error, observed)
                self._accepting = False

    def bind_start(self, *, t0: float) -> None:
        """Lock the camera Start boundary after the disk spool was pre-armed."""

        start = _finite(t0, "force capture Start timestamp")
        failure: ForceCaptureError | None = None
        with self._state_lock:
            if not self._started:
                raise RuntimeError("Hall-force recorder is not pre-armed")
            if self.t0 is not None:
                raise RuntimeError("Hall-force Start boundary is already bound")
            self.t0 = start
            unsafe = self._last_prebind_unsafe
            reader_error = self._last_prebind_reader_error
            if unsafe is not None and unsafe[1] >= start:
                # P0-4: an over-threshold reading at the Start boundary is advisory, not
                # a Start failure -- keep recording (the sample is preserved in the CSV).
                self._unsafe_failure = unsafe
            if reader_error is not None and reader_error[1] >= start:
                self._first_reader_error = reader_error
                self._accepting = False
                failure = ForceCaptureError(
                    f"Hall-force serial reader failed at Start boundary: {reader_error[0]}"
                )
            self._last_prebind_unsafe = None
            self._last_prebind_reader_error = None
        _atomic_json(
            self.state_path,
            self._state_payload(status="recording" if failure is None else "failed"),
        )
        if failure is not None:
            raise failure

    def _checkpoint(self, descriptor: int) -> None:
        os.fsync(descriptor)
        self._durable_samples = self._written_samples
        self._checkpoint_count += 1
        _atomic_json(
            self.state_path,
            self._state_payload(
                status="recording" if self.t0 is not None else "prearmed"
            ),
        )

    def _run(self) -> None:
        descriptor: int | None = None
        try:
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(str(self.csv_inprogress_path), flags, 0o640)
            _write_all(descriptor, (",".join(CSV_COLUMNS) + "\n").encode("ascii"))
            _fsync_directory(self.session_dir)
            last_checkpoint = time.monotonic()
            since_checkpoint = 0
            while True:
                item = self._queue.get()
                try:
                    if item is _SENTINEL:
                        break
                    assert isinstance(item, ForceSample)
                    _write_all(descriptor, _sample_row(item))
                    self._written_samples += 1
                    since_checkpoint += 1
                    now = time.monotonic()
                    if (
                        since_checkpoint >= self.checkpoint_samples
                        or now - last_checkpoint >= self.checkpoint_interval_s
                    ):
                        self._checkpoint(descriptor)
                        since_checkpoint = 0
                        last_checkpoint = now
                finally:
                    self._queue.task_done()
            self._checkpoint(descriptor)
        except BaseException as exc:
            self._worker_error = exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    if self._worker_error is None:
                        self._worker_error = exc

    def _stop_accepting_and_drain(self) -> None:
        with self._state_lock:
            self._accepting = False
        thread = self._thread
        if thread is None:
            return
        inserted = False
        while thread.is_alive() and not inserted:
            try:
                self._queue.put(_SENTINEL, timeout=0.1)
                inserted = True
            except queue.Full:
                continue
        thread.join(timeout=10.0)
        if thread.is_alive():
            raise ForceCaptureError("Hall-force spool did not stop within 10 seconds")

    def _mark_interrupted(self, reason: str | None = None) -> None:
        payload = self._state_payload(status="interrupted")
        if reason:
            payload["failure"] = reason
        try:
            _atomic_json(self.state_path, payload)
        except BaseException:
            pass
        try:
            self.complete_path.unlink(missing_ok=True)
            _fsync_directory(self.session_dir)
        except BaseException:
            pass

    def abort(self) -> None:
        try:
            self._stop_accepting_and_drain()
        finally:
            self._mark_interrupted("capture aborted")

    def finish(self, *, t_end: float) -> ForceCaptureSummary:
        try:
            self._stop_accepting_and_drain()
            end = _finite(t_end, "force capture End timestamp")
            start = self.t0
            if start is None:
                raise ForceCaptureError("Hall-force Start boundary was never bound")
            if end <= start:
                raise ValueError("force capture End timestamp must follow Start")
            if self._worker_error is not None:
                raise ForceCaptureError(
                    f"Hall-force spool failed: {self._worker_error}"
                ) from self._worker_error
            with self._state_lock:
                overflow = self._overflow
                reader_error = self._first_reader_error
                unsafe_failure = self._unsafe_failure
            if overflow:
                raise ForceCaptureError(
                    "Hall-force spool queue overflow; capture is incomplete"
                )
            # P0-4: an over-threshold ("unsafe") force reading is advisory -- it was
            # recorded to the CSV like any other sample and NEVER fails the capture; the
            # physician's over-threshold datum is preserved, not dropped.
            del unsafe_failure
            if (
                reader_error is not None
                and start <= reader_error[1] <= end
            ):
                raise ForceCaptureError(
                    f"Hall-force serial reader failed: {reader_error[0]}"
                ) from reader_error[0]
            return self._publish(end)
        except BaseException as exc:
            self._mark_interrupted(str(exc))
            raise

    def _publish(self, end: float) -> ForceCaptureSummary:
        start = self.t0
        if start is None:
            raise ForceCaptureError("Hall-force Start boundary was never bound")
        temp = self.csv_path.with_name(
            f".{self.csv_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        flags |= getattr(os, "O_NOFOLLOW", 0)
        output_descriptor = os.open(str(temp), flags, 0o640)
        try:
            with self.csv_inprogress_path.open(
                "r", encoding="utf-8", newline=""
            ) as source, os.fdopen(
                output_descriptor, "w", encoding="utf-8", newline="", closefd=True
            ) as output:
                output_descriptor = -1
                audit = _copy_and_validate_csv(
                    source,
                    output,
                    t0=start,
                    t_end=end,
                    max_device_gap_ms=self.max_device_gap_ms,
                    max_host_gap_ms=self.max_host_gap_ms,
                    calibration=self.calibration,
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, self.csv_path)
            _fsync_directory(self.session_dir)
        except BaseException:
            if output_descriptor >= 0:
                os.close(output_descriptor)
            temp.unlink(missing_ok=True)
            raise

        csv_sha256 = _sha256_regular_file(self.csv_path)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "status": "sensors_finalized",
            "training_eligible": False,
            "completion_marker": COMPLETE_NAME,
            "csv_path": CSV_NAME,
            "csv_columns": list(CSV_COLUMNS),
            "csv_sha256": csv_sha256,
            "n_samples": audit.n_samples,
            "t0_monotonic": start,
            "requested_t_end_monotonic": end,
            "force_t_start_monotonic": audit.t_start,
            "force_t_end_monotonic": audit.t_end,
            "device_t_start_ms": audit.device_t_start_ms,
            "device_t_end_ms": audit.device_t_end_ms,
            "max_device_gap_ms": self.max_device_gap_ms,
            "max_host_gap_ms": self.max_host_gap_ms,
            "queue_capacity": self.queue_capacity,
            "checkpoint_samples": self.checkpoint_samples,
            "durable_checkpoints": self._checkpoint_count,
            "signal_authenticity": audit.signal_authenticity,
            "calibration": self.calibration_identity,
            "calibration_artifact": self._calibration_artifact_claim(),
            "device_identity": self.device_identity,
            "device_identity_sha256": self.device_identity_sha256,
        }
        metadata_raw = _atomic_json(self.metadata_path, metadata)
        marker = {
            "schema_version": COMPLETE_SCHEMA_VERSION,
            "capture_schema_version": SCHEMA_VERSION,
            "force_csv_sha256": csv_sha256,
            "force_capture_json_sha256": _sha256_bytes(metadata_raw),
            "n_samples": audit.n_samples,
            "t0_monotonic": start,
            "requested_t_end_monotonic": end,
            "calibration_artifact_sha256": (
                None
                if self.calibration_artifact is None
                else self.calibration_identity["sha256"]
            ),
        }
        for path in (self.csv_inprogress_path, self.state_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(self.session_dir)
        # The marker is the final durable mutation in a successful sidecar.
        _atomic_json(self.complete_path, marker)
        return verify_force_capture(self.session_dir)


def _copy_and_validate_csv(
    source: io.TextIOBase,
    output: io.TextIOBase | None,
    *,
    t0: float,
    t_end: float,
    max_device_gap_ms: float,
    max_host_gap_ms: float,
    calibration: ForceCalibration | None = None,
) -> _CsvAudit:
    reader = csv.DictReader(source)
    if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
        raise ForceCaptureError("force.csv has an invalid exact header")
    writer = None
    if output is not None:
        writer = csv.DictWriter(
            output, fieldnames=CSV_COLUMNS, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()

    count = 0
    first_host: float | None = None
    last_host: float | None = None
    first_device: float | None = None
    last_device: float | None = None
    all_zero_raw = True
    raw_min: int | None = None
    raw_max: int | None = None
    force_min: float | None = None
    force_max: float | None = None
    raw_values: set[int] = set()
    force_values: set[str] = set()
    previous_signature: tuple[str, str] | None = None
    identical_run_samples = 0
    identical_run_start: float | None = None
    longest_identical_run_samples = 0
    longest_identical_run_s = 0.0
    host_limit_s = max_host_gap_ms / 1_000.0
    tolerance = 1e-12
    for row_number, row in enumerate(reader, start=2):
        if set(row) != set(CSV_COLUMNS) or None in row:
            raise ForceCaptureError(f"force.csv row {row_number} has invalid columns")
        host = _finite(row["host_t_s"], f"force.csv row {row_number} host_t_s")
        if host < t0 - tolerance or host > t_end + tolerance:
            if output is not None:
                continue
            raise ForceCaptureError(
                f"force.csv row {row_number} lies outside the Start/End interval"
            )
        raw_text = row["raw"]
        try:
            raw = int(raw_text)
        except (TypeError, ValueError) as exc:
            raise ForceCaptureError(f"force.csv row {row_number} raw is invalid") from exc
        if raw < 0 or str(raw) != raw_text:
            raise ForceCaptureError(f"force.csv row {row_number} raw is invalid")
        numeric = {
            name: _finite(row[name], f"force.csv row {row_number} {name}")
            for name in ("hall_v", "displacement_mm", "force_n")
        }
        if row["status"] not in _NATIVE_SAMPLE_STATUS:
            raise ForceCaptureError(f"force.csv row {row_number} status is invalid")
        # P0-4: over-threshold ("unsafe") rows are advisory physician-reference data and
        # are PRESERVED in force.csv, never rejected -- only a non-native status is
        # invalid. The row's status is recorded verbatim for the physician.
        if not row["device_t_ms"]:
            raise ForceCaptureError(
                f"force.csv row {row_number} is missing device_t_ms"
            )
        device = _finite(
            row["device_t_ms"], f"force.csv row {row_number} device_t_ms"
        )
        if calibration is not None:
            expected = calibration.sample_from_raw(
                raw,
                host,
                device_t_ms=device,
            )
            calibration_tolerances = {
                "hall_v": float(_AUTHENTICITY_POLICY["calibration_hall_v_tolerance"]),
                "displacement_mm": float(
                    _AUTHENTICITY_POLICY["calibration_displacement_tolerance_mm"]
                ),
                "force_n": float(
                    _AUTHENTICITY_POLICY["calibration_force_tolerance_n"]
                ),
            }
            for field, expected_value in (
                ("hall_v", expected.hall_v),
                ("displacement_mm", expected.displacement_mm),
                ("force_n", expected.force_n),
            ):
                if abs(numeric[field] - float(expected_value)) > calibration_tolerances[field]:
                    raise ForceCaptureError(
                        f"force.csv row {row_number} calibration recomputation "
                        f"mismatch for {field}"
                    )
            if row["status"] != expected.status:
                raise ForceCaptureError(
                    f"force.csv row {row_number} calibration recomputation "
                    "mismatch for status"
                )
        if last_host is not None:
            host_delta = host - last_host
            if host_delta <= 0:
                raise ForceCaptureError("Hall-force host clock regressed or repeated")
            if host_delta > host_limit_s + tolerance:
                raise ForceCaptureError(
                    f"Hall-force host clock gap {host_delta * 1_000.0:.3f} ms "
                    f"exceeds {max_host_gap_ms:.3f} ms"
                )
        if last_device is not None:
            device_delta = device - last_device
            if device_delta <= 0:
                raise ForceCaptureError("Hall-force device clock regressed or repeated")
            if device_delta > max_device_gap_ms + tolerance:
                raise ForceCaptureError(
                    f"Hall-force device clock gap {device_delta:.3f} ms "
                    f"exceeds {max_device_gap_ms:.3f} ms"
                )
        if first_host is None:
            first_host = host
            first_device = device
        signature = (raw_text, row["force_n"])
        if signature == previous_signature:
            identical_run_samples += 1
        else:
            identical_run_samples = 1
            identical_run_start = host
            previous_signature = signature
        assert identical_run_start is not None
        run_duration = host - identical_run_start
        if (
            identical_run_samples > longest_identical_run_samples
            or (
                identical_run_samples == longest_identical_run_samples
                and run_duration > longest_identical_run_s
            )
        ):
            longest_identical_run_samples = identical_run_samples
            longest_identical_run_s = run_duration
        all_zero_raw = all_zero_raw and raw == 0
        raw_min = raw if raw_min is None else min(raw_min, raw)
        raw_max = raw if raw_max is None else max(raw_max, raw)
        force_min = (
            numeric["force_n"]
            if force_min is None
            else min(force_min, numeric["force_n"])
        )
        force_max = (
            numeric["force_n"]
            if force_max is None
            else max(force_max, numeric["force_n"])
        )
        value_cap = int(_AUTHENTICITY_POLICY["distinct_value_count_cap"])
        if len(raw_values) < value_cap:
            raw_values.add(raw)
        if len(force_values) < value_cap:
            force_values.add(row["force_n"])
        last_host = host
        last_device = device
        count += 1
        if writer is not None:
            writer.writerow(row)

    if count <= 0 or first_host is None or last_host is None:
        raise ForceCaptureError("Hall-force capture contains no samples in Start/End")
    if first_host - t0 > host_limit_s + tolerance:
        raise ForceCaptureError(
            "Hall-force coverage gap at Start exceeds the configured host gap"
        )
    if t_end - last_host > host_limit_s + tolerance:
        raise ForceCaptureError(
            "Hall-force coverage gap at End exceeds the configured host gap"
        )
    assert first_device is not None and last_device is not None
    assert raw_min is not None and raw_max is not None
    assert force_min is not None and force_max is not None
    duration_s = last_host - first_host
    if all_zero_raw:
        raise ForceCaptureError(
            "Hall-force signal authenticity failed: all-zero raw ADC"
        )
    if (
        longest_identical_run_s
        >= float(_AUTHENTICITY_POLICY["freeze_min_duration_s"])
        and longest_identical_run_samples
        >= int(_AUTHENTICITY_POLICY["freeze_min_samples"])
    ):
        raise ForceCaptureError(
            "Hall-force signal authenticity failed: frozen bit-identical raw ADC/force"
        )
    if (
        duration_s >= float(_AUTHENTICITY_POLICY["conversion_min_duration_s"])
        and count >= int(_AUTHENTICITY_POLICY["conversion_min_samples"])
        and raw_max - raw_min
        >= int(_AUTHENTICITY_POLICY["conversion_min_raw_span"])
        and len(raw_values)
        >= int(_AUTHENTICITY_POLICY["conversion_min_distinct_raw"])
        and len(force_values) == 1
    ):
        raise ForceCaptureError(
            "Hall-force signal authenticity failed: conversion output force is pinned "
            "while raw ADC changes"
        )
    near_zero_response = (
        max(abs(force_min), abs(force_max))
        <= float(_AUTHENTICITY_POLICY["near_zero_max_abs_force_n"])
    )
    if near_zero_response:
        raise ForceCaptureError(
            "Hall-force signal authenticity failed: near-zero response over the "
            "full capture"
        )
    if (
        duration_s
        >= float(_AUTHENTICITY_POLICY["near_constant_min_duration_s"])
        and count >= int(_AUTHENTICITY_POLICY["near_constant_min_samples"])
        and (
            raw_max - raw_min
            < int(_AUTHENTICITY_POLICY["near_constant_min_raw_span"])
            or force_max - force_min
            < float(_AUTHENTICITY_POLICY["near_constant_min_force_span_n"])
        )
    ):
        raise ForceCaptureError(
            "Hall-force signal authenticity failed: near-constant/near-zero "
            "response over the full capture"
        )
    calibration_ready = calibration is not None
    evidence_sufficient = (
        duration_s + 1e-9
        >= float(_AUTHENTICITY_POLICY["minimum_evidence_duration_s"])
        and count >= int(_AUTHENTICITY_POLICY["minimum_evidence_samples"])
    )
    promotion_blockers = []
    if not evidence_sufficient:
        promotion_blockers.append("minimum duration/sample evidence")
    if not calibration_ready:
        promotion_blockers.append(
            "pinned calibration artifact unavailable for independent recomputation"
        )
    signal_authenticity = {
        "schema_version": SIGNAL_AUTHENTICITY_SCHEMA_VERSION,
        "policy_version": SIGNAL_AUTHENTICITY_POLICY_VERSION,
        "status": "PASS" if evidence_sufficient else "INSUFFICIENT_EVIDENCE",
        "promotion_eligible": calibration_ready and evidence_sufficient,
        "promotion_blockers": promotion_blockers,
        "checks": {
            "all_zero": "PASS",
            "conversion_response": "PASS",
            "digital_freeze": "PASS",
            "minimum_evidence": (
                "PASS" if evidence_sufficient else "INSUFFICIENT_EVIDENCE"
            ),
            "near_zero_response": "PASS",
            "near_constant_response": "PASS",
            "calibration_recomputation": (
                "PASS" if calibration_ready else "NOT_AVAILABLE"
            ),
        },
        "policy": dict(_AUTHENTICITY_POLICY),
        "evidence": {
            "duration_s": round(duration_s, 9),
            "n_samples": count,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_span": raw_max - raw_min,
            "force_min_n": force_min,
            "force_max_n": force_max,
            "force_span_n": force_max - force_min,
            "distinct_raw_values_capped": len(raw_values),
            "distinct_force_values_capped": len(force_values),
            "longest_identical_run_samples": longest_identical_run_samples,
            "longest_identical_run_s": round(longest_identical_run_s, 9),
        },
    }
    return _CsvAudit(
        n_samples=count,
        t_start=first_host,
        t_end=last_host,
        device_t_start_ms=first_device,
        device_t_end_ms=last_device,
        signal_authenticity=signal_authenticity,
    )


def _open_regular(path: Path, label: str):  # noqa: ANN202
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise ForceCaptureVerificationError(f"{label} is missing or unsafe") from exc
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        os.close(descriptor)
        raise ForceCaptureVerificationError(f"{label} is not a regular file")
    return descriptor, before


def _stable_small_bytes(path: Path, label: str, *, limit: int = 1_048_576) -> bytes:
    descriptor, before = _open_regular(path, label)
    try:
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, min(65_536, limit + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
            if size > limit:
                raise ForceCaptureVerificationError(f"{label} exceeds its size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ForceCaptureVerificationError(f"{label} changed while it was read")
    return b"".join(chunks)


def _decode_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForceCaptureVerificationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ForceCaptureVerificationError(f"{label} must be a JSON object")
    if raw != _canonical_json_bytes(value):
        raise ForceCaptureVerificationError(f"{label} is not canonical JSON")
    return value


def _sha256_regular_file(path: Path) -> str:
    descriptor, before = _open_regular(path, path.name)
    digest = hashlib.sha256()
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ForceCaptureVerificationError(f"{path.name} changed while hashing")
    return digest.hexdigest()


def _hash_and_validate_regular_csv(
    path: Path,
    *,
    expected_sha256: str,
    t0: float,
    t_end: float,
    max_device_gap_ms: float,
    max_host_gap_ms: float,
    calibration: ForceCalibration | None,
) -> tuple[str, _CsvAudit]:
    """Hash and audit one stable, no-follow CSV descriptor.

    Keeping both operations on one descriptor closes the hash-then-reopen race;
    the final ``lstat`` also rejects a path swap while that descriptor is open.
    """

    descriptor, before = _open_regular(path, "Hall-force CSV")
    digest = hashlib.sha256()
    try:
        binary = os.fdopen(descriptor, "rb", closefd=False)
        while True:
            block = binary.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ForceCaptureVerificationError(
                "Hall-force CSV SHA-256 does not match metadata"
            )
        binary.seek(0)
        text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
        try:
            try:
                audit = _copy_and_validate_csv(
                    text,
                    None,
                    t0=t0,
                    t_end=t_end,
                    max_device_gap_ms=max_device_gap_ms,
                    max_host_gap_ms=max_host_gap_ms,
                    calibration=calibration,
                )
            except ForceCaptureVerificationError:
                raise
            except ForceCaptureError as exc:
                raise ForceCaptureVerificationError(str(exc)) from exc
        finally:
            text.detach()
            binary.close()
        after = os.fstat(descriptor)
        try:
            current = os.lstat(path)
        except OSError as exc:
            raise ForceCaptureVerificationError(
                "Hall-force CSV changed while validating"
            ) from exc
    finally:
        os.close(descriptor)
    stable_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if stable_identity != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
        raise ForceCaptureVerificationError("Hall-force CSV changed while validating")
    if not stat.S_ISREG(current.st_mode):
        raise ForceCaptureVerificationError("Hall-force CSV is not a regular file")
    return actual_sha256, audit


def verify_force_capture(session_dir: str | Path) -> ForceCaptureSummary:
    """Verify marker, hashes, identities, exact counts and both clock timelines."""

    session = Path(session_dir).expanduser().resolve()
    lingering_staging = [
        name
        for name in (CSV_INPROGRESS_NAME, STATE_NAME)
        if (
            (path := session / name).exists()
            or path.is_symlink()
        )
    ]
    if lingering_staging:
        raise ForceCaptureVerificationError(
            "Hall-force staging artifacts remain beside finalized data: "
            + ", ".join(lingering_staging)
        )
    marker_raw = _stable_small_bytes(
        session / COMPLETE_NAME, "Hall-force completion marker"
    )
    marker = _decode_json(marker_raw, "Hall-force completion marker")
    metadata_raw = _stable_small_bytes(
        session / METADATA_NAME, "Hall-force capture metadata"
    )
    metadata = _decode_json(metadata_raw, "Hall-force capture metadata")
    if marker.get("schema_version") != COMPLETE_SCHEMA_VERSION:
        raise ForceCaptureVerificationError("Hall-force completion marker schema is invalid")
    if marker.get("capture_schema_version") != SCHEMA_VERSION:
        raise ForceCaptureVerificationError("Hall-force completion marker capture schema is invalid")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ForceCaptureVerificationError("Hall-force capture metadata schema is invalid")
    if metadata.get("status") != "sensors_finalized" or metadata.get("training_eligible") is not False:
        raise ForceCaptureVerificationError("Hall-force capture metadata is not fail-closed finalized")
    if marker.get("force_capture_json_sha256") != _sha256_bytes(metadata_raw):
        raise ForceCaptureVerificationError("Hall-force metadata SHA-256 does not match completion marker")
    csv_path = session / CSV_NAME
    calibration = metadata.get("calibration")
    device_identity = metadata.get("device_identity")
    try:
        calibration_identity = _validate_calibration_identity(calibration)  # type: ignore[arg-type]
        normalized_device = _json_identity(device_identity, "device_identity")  # type: ignore[arg-type]
    except ValueError as exc:
        raise ForceCaptureVerificationError(str(exc)) from exc
    device_sha = _sha256_bytes(_canonical_json_bytes(normalized_device))
    if metadata.get("device_identity_sha256") != device_sha:
        raise ForceCaptureVerificationError("Hall-force device identity SHA-256 is invalid")

    calibration_claim = metadata.get("calibration_artifact")
    calibration_artifact_path = session / CALIBRATION_ARTIFACT_NAME
    calibration_model: ForceCalibration | None = None
    if calibration_claim is None:
        if calibration_artifact_path.exists() or calibration_artifact_path.is_symlink():
            raise ForceCaptureVerificationError(
                "unclaimed Hall-force calibration artifact is present"
            )
        if marker.get("calibration_artifact_sha256") is not None:
            raise ForceCaptureVerificationError(
                "Hall-force calibration marker claim is invalid"
            )
    else:
        expected_claim_fields = {
            "schema_version", "path", "sha256", "bytes", "release_id",
        }
        if (
            not isinstance(calibration_claim, dict)
            or set(calibration_claim) != expected_claim_fields
            or calibration_claim.get("schema_version")
            != CALIBRATION_ARTIFACT_SCHEMA_VERSION
            or calibration_claim.get("path") != CALIBRATION_ARTIFACT_NAME
            or calibration_claim.get("sha256")
            != calibration_identity.get("sha256")
            or calibration_claim.get("release_id")
            != calibration_identity.get("release_id")
            or type(calibration_claim.get("bytes")) is not int
            or calibration_claim["bytes"] <= 0
            or marker.get("calibration_artifact_sha256")
            != calibration_claim.get("sha256")
        ):
            raise ForceCaptureVerificationError(
                "Hall-force calibration artifact claim is invalid"
            )
        calibration_raw = _stable_small_bytes(
            calibration_artifact_path,
            "Hall-force calibration artifact",
            limit=4 * 1024 * 1024,
        )
        if (
            len(calibration_raw) != calibration_claim["bytes"]
            or _sha256_bytes(calibration_raw) != calibration_claim["sha256"]
        ):
            raise ForceCaptureVerificationError(
                "Hall-force calibration artifact content identity is invalid"
            )
        try:
            calibration_model = ForceCalibration.from_bytes(calibration_raw)
        except ValueError as exc:
            raise ForceCaptureVerificationError(
                "Hall-force calibration artifact cannot be parsed"
            ) from exc
        if (
            calibration_model.calibrated is not True
            or calibration_identity.get("calibrated") is not True
            or calibration_identity.get("artifact_schema_version")
            != CALIBRATION_ARTIFACT_SCHEMA_VERSION
        ):
            raise ForceCaptureVerificationError(
                "Hall-force calibration artifact is not production-pinned"
            )

    t0 = _finite(metadata.get("t0_monotonic"), "Hall-force metadata Start")
    end = _finite(
        metadata.get("requested_t_end_monotonic"), "Hall-force metadata End"
    )
    max_device_gap = _finite(
        metadata.get("max_device_gap_ms"), "Hall-force metadata max_device_gap_ms"
    )
    max_host_gap = _finite(
        metadata.get("max_host_gap_ms"), "Hall-force metadata max_host_gap_ms"
    )
    claimed_csv_sha256 = metadata.get("csv_sha256")
    if (
        not isinstance(claimed_csv_sha256, str)
        or claimed_csv_sha256 != marker.get("force_csv_sha256")
    ):
        raise ForceCaptureVerificationError(
            "Hall-force CSV SHA-256 claims do not match"
        )
    csv_sha256, audit = _hash_and_validate_regular_csv(
        csv_path,
        expected_sha256=claimed_csv_sha256,
        t0=t0,
        t_end=end,
        max_device_gap_ms=max_device_gap,
        max_host_gap_ms=max_host_gap,
        calibration=calibration_model,
    )
    expected = {
        "n_samples": audit.n_samples,
        "force_t_start_monotonic": audit.t_start,
        "force_t_end_monotonic": audit.t_end,
        "device_t_start_ms": audit.device_t_start_ms,
        "device_t_end_ms": audit.device_t_end_ms,
    }
    for name, value in expected.items():
        claimed = metadata.get(name)
        if isinstance(value, int):
            if claimed != value:
                raise ForceCaptureVerificationError(f"Hall-force metadata {name} is invalid")
        elif _finite(claimed, f"Hall-force metadata {name}") != value:
            raise ForceCaptureVerificationError(f"Hall-force metadata {name} is invalid")
    if marker.get("n_samples") != audit.n_samples:
        raise ForceCaptureVerificationError("Hall-force completion marker count is invalid")
    if _finite(marker.get("t0_monotonic"), "completion Start") != t0:
        raise ForceCaptureVerificationError("Hall-force completion marker Start is invalid")
    if _finite(marker.get("requested_t_end_monotonic"), "completion End") != end:
        raise ForceCaptureVerificationError("Hall-force completion marker End is invalid")
    if metadata.get("signal_authenticity") != audit.signal_authenticity:
        raise ForceCaptureVerificationError(
            "Hall-force signal authenticity receipt does not match the CSV"
        )
    return ForceCaptureSummary(
        path=str(csv_path),
        t_start=audit.t_start,
        t_end=audit.t_end,
        n_samples=audit.n_samples,
        status="OK",
        sha256=csv_sha256,
        device_t_start_ms=audit.device_t_start_ms,
        device_t_end_ms=audit.device_t_end_ms,
        calibration_sha256=str(calibration_identity["sha256"]),
        device_identity_sha256=device_sha,
        signal_authenticity=audit.signal_authenticity,
    )


class ForceCaptureStream:
    """One live Hall serial owner with attachable Start/End disk recordings."""

    def __init__(
        self,
        port: str,
        calibration_path: str | Path,
        *,
        baud: int = 115200,
        device_identity: Mapping[str, object] | None = None,
        on_safety_event: Callable[[ForceSample], None] | None = None,
        on_hard_limit: Callable[[ForceSample], None] | None = None,
        on_capture_fault: Callable[[BaseException, float], None] | None = None,
        sensor_factory: Callable[..., ForceSensor] = ForceSensor,
        queue_samples: int = 2_048,
        checkpoint_samples: int = 100,
        checkpoint_interval_s: float = 1.0,
        max_device_gap_ms: float = 50.0,
        max_host_gap_ms: float = 50.0,
    ) -> None:
        calibration_file = Path(calibration_path).expanduser().resolve()
        calibration_raw = _stable_small_bytes(
            calibration_file, "Hall-force calibration", limit=4 * 1024 * 1024
        )
        calibration = ForceCalibration.from_bytes(calibration_raw)
        calibration_sha256 = _sha256_bytes(calibration_raw)
        self.calibration_artifact = calibration_raw
        self.calibration_identity = {
            "sha256": calibration_sha256,
            "calibrated": calibration.calibrated,
            "source_name": calibration_file.name,
            "artifact_schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
            "release_id": f"hall-force-calibration-{calibration_sha256[:20]}",
        }
        identity = {"port": str(port), "baud": int(baud)}
        if device_identity is not None:
            identity.update(dict(device_identity))
        if identity.get("port") != str(port) or identity.get("baud") != int(baud):
            raise ValueError("Hall-force device identity must match port and baud")
        self.device_identity = _json_identity(identity, "device_identity")
        self._settings = {
            "queue_samples": queue_samples,
            "checkpoint_samples": checkpoint_samples,
            "checkpoint_interval_s": checkpoint_interval_s,
            "max_device_gap_ms": max_device_gap_ms,
            "max_host_gap_ms": max_host_gap_ms,
        }
        self._lock = threading.Lock()
        self._on_capture_fault = on_capture_fault
        self._recording: ForceCaptureRecorder | None = None
        self._prepared = False
        if on_safety_event is not None and on_hard_limit is not None:
            raise ValueError("provide only one Hall-force safety callback alias")
        safety_callback = (
            on_safety_event if on_safety_event is not None else on_hard_limit
        )
        self._sensor = sensor_factory(
            str(port),
            calibration,
            baud=int(baud),
            on_hard_limit=safety_callback,
            on_sample=self._on_sample,
            on_reader_error=self._on_reader_error,
        )

    def prepare(self) -> "ForceCaptureStream":
        with self._lock:
            if self._prepared:
                return self
            self._sensor.start()
            self._prepared = True
        return self

    start = prepare

    def stop(self) -> None:
        errors: list[BaseException] = []
        try:
            self.abort_recording()
        except BaseException as exc:
            errors.append(exc)
        with self._lock:
            prepared = self._prepared
            self._prepared = False
        if prepared:
            try:
                self._sensor.stop()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> "ForceCaptureStream":
        return self.prepare()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def read(self) -> ForceSample:
        return self._sensor.read()

    def _on_sample(self, sample: ForceSample) -> None:
        with self._lock:
            recording = self._recording
        if recording is not None:
            recording.submit(sample)

    def _on_reader_error(self, error: BaseException, observed_t: float) -> None:
        with self._lock:
            recording = self._recording
        if recording is not None:
            recording.report_reader_error(error, t=observed_t)
        if self._on_capture_fault is not None:
            try:
                self._on_capture_fault(error, observed_t)
            except Exception:
                pass

    def prepare_recording(self, session_dir: str | Path) -> None:
        """Pre-arm the spool before camera Start so the boundary has no attach gap."""

        with self._lock:
            if not self._prepared:
                raise RuntimeError("Hall-force stream is not prepared")
            if self._recording is not None:
                raise RuntimeError("Hall-force recording is already active")
        recorder = ForceCaptureRecorder(
            session_dir,
            t0=None,
            calibration_identity=self.calibration_identity,
            calibration_artifact=self.calibration_artifact,
            device_identity=self.device_identity,
            **self._settings,
        )
        recorder.start()
        with self._lock:
            if self._recording is not None:
                recorder.abort()
                raise RuntimeError("Hall-force recording is already active")
            self._recording = recorder

    def begin_recording(
        self, session_dir: str | Path | None = None, *, t0: float,
    ) -> None:
        """Bind Start on a pre-armed spool.

        Passing ``session_dir`` remains a convenience for non-camera tools.  A
        synchronized collector must call ``prepare_recording(session)`` before
        camera Start, then call ``begin_recording(t0=...)`` at the shared
        barrier so native force samples between camera Start and this call are
        already on the spool.
        """

        if session_dir is not None:
            self.prepare_recording(session_dir)
        with self._lock:
            recorder = self._recording
        if recorder is None:
            raise RuntimeError("Hall-force recording is not pre-armed")
        try:
            recorder.bind_start(t0=t0)
        except BaseException:
            with self._lock:
                if self._recording is recorder:
                    self._recording = None
            recorder.abort()
            raise

    def end_recording(self, *, t_end: float) -> ForceCaptureSummary:
        with self._lock:
            recorder = self._recording
            self._recording = None
        if recorder is None:
            raise RuntimeError("Hall-force recording is not active")
        return recorder.finish(t_end=t_end)

    def abort_recording(self) -> bool:
        with self._lock:
            recorder = self._recording
            self._recording = None
        if recorder is None:
            return False
        recorder.abort()
        return True

    @property
    def recording_queue_capacity(self) -> int:
        with self._lock:
            return 0 if self._recording is None else self._recording.queue_capacity
