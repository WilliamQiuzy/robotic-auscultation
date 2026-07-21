"""Durable, explicitly non-training RGB-D perception capture.

The caller remains the sole camera owner.  It injects the same immutable camera
frame used by preview and inference; this writer never imports or opens an Orbbec
device.  Successful finalization emits only ``DIAGNOSTIC_FINALIZATION.json``.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import queue
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
import wave
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from ausc.audio.audio_stream import (
    AudioSignalAuthenticityAccumulator,
    AudioSignalAuthenticityError,
)
from ausc.realtime_q import (
    QUALITY_MODEL_INPUT_DIRNAME,
    QUALITY_MODEL_INPUT_STAGING_DIRNAME,
    QualityPacketError,
    quality_packet_log_summary,
    verify_quality_packet_log,
)


_FORCE_FEEDBACK_DIR = (
    Path(__file__).resolve().parents[1]
    / "mechanical_sensor"
    / "force_feedback"
)
if str(_FORCE_FEEDBACK_DIR) not in sys.path:
    sys.path.insert(0, str(_FORCE_FEEDBACK_DIR))
from force_capture import (  # noqa: E402
    ForceCaptureVerificationError,
    verify_force_capture,
)


DIAGNOSTIC_SCHEMA_VERSION = "wam_diagnostic_perception_v1"
DIAGNOSTIC_FINALIZATION_SCHEMA_VERSION = "wam_diagnostic_finalization_v1"
CAPTURE_MODE = "diagnostic_perception"
_REQUIRED_PROVENANCE = ("camera", "model", "runtime")
_FINAL_VIDEO_RESERVE_MULTIPLIER = 2
_FINAL_VIDEO_RESERVE_OVERHEAD_BYTES = 64 * 1024**2
_DIAGNOSTIC_AUDIO_SAMPLE_RATE = 16_000
_ROBOT_CAPTURE_SUMMARY_SCHEMA_V3 = "wam_openarm_exact_control_capture_summary_v3"
_ROBOT_CAPTURE_SUMMARY_SCHEMA_V4 = "wam_openarm_exact_control_capture_summary_v4"
# Compatibility name used by the historical v3 diagnostic verifier below.
_ROBOT_CAPTURE_SUMMARY_SCHEMA = _ROBOT_CAPTURE_SUMMARY_SCHEMA_V3
_ROBOT_TX_STATUS_REASON = "openarm_can_1_2_2_full_frame_write_results"
_ROBOT_TX_STATUS_SEMANTICS = (
    "linux_socketcan_full_frame_write_tx_enqueue_not_motor_ack"
)
_ROBOT_MOTOR_ACK_REASON = (
    "socketcan_tx_enqueue_does_not_acknowledge_motor_receipt_or_execution"
)
_ROBOT_FEEDBACK_LIVENESS_REASON = (
    "v3_has_no_per_motor_feedback_rx_counters_or_timestamps"
)
_ROBOT_FINAL_ARTIFACT_NAMES = (
    "robot_control_envelopes.jsonl",
    "robot_exact_control.csv",
    "robot_frame_index.jsonl",
    "robot_capture_summary.json",
)
_ROBOT_DATA_ARTIFACT_NAMES = _ROBOT_FINAL_ARTIFACT_NAMES[:-1]
_FORCE_FINAL_ARTIFACT_NAMES = (
    "force.csv",
    "force_capture.json",
    "FORCE_CAPTURE_COMPLETE.json",
)
_FORCE_OPTIONAL_ARTIFACT_NAMES = ("force_calibration.json",)
_FORCE_STAGING_ARTIFACT_NAMES = (
    "force.csv.inprogress",
    "force_capture.json.inprogress",
)
_QUALITY_FINAL_ARTIFACT_NAMES = (
    "quality_inference_packets.jsonl",
    "quality_inference_authenticity.json",
)
_QUALITY_STAGING_ARTIFACT_NAMES = (
    ".quality_inference_packets.jsonl.inprogress",
)
_MAX_ROBOT_FRAME_INDEX_LINE_BYTES = 65_536


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, raw: bytes) -> None:
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


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _atomic_save_npy(path: Path, values: np.ndarray) -> dict[str, object]:
    """Publish one depth frame and bind its one-time content hash/identity."""

    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.inprogress")
    try:
        with partial.open("xb") as fp:
            np.save(fp, values, allow_pickle=False)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
        before = path.stat()
        digest = _sha256(path)
        after = path.stat()
        identity_before = (
            int(before.st_dev), int(before.st_ino), int(before.st_size),
            int(before.st_mtime_ns), int(before.st_ctime_ns),
        )
        identity_after = (
            int(after.st_dev), int(after.st_ino), int(after.st_size),
            int(after.st_mtime_ns), int(after.st_ctime_ns),
        )
        if identity_before != identity_after:
            raise RuntimeError(f"depth frame changed while hashing: {path}")
        return {
            "bytes": int(after.st_size),
            "sha256": digest,
            "identity": list(identity_after),
        }
    finally:
        partial.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audio_time_matches(actual: float, expected: float, sample_rate: int) -> bool:
    tolerance = max(1e-9, 1.0 / (float(sample_rate) * 1_000.0))
    return bool(np.isclose(actual, expected, rtol=0.0, atol=tolerance))


def _exact_audio_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"diagnostic audio {label} must be an exact integer")
    return value


def _strict_robot_json_object(pairs):  # noqa: ANN001
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"duplicate robot JSON field: {key}")
        value[key] = item
    return value


def _reject_robot_json_constant(value: str) -> None:
    raise RuntimeError(f"non-finite robot JSON constant is forbidden: {value}")


def _stream_audio_jsonl(path: Path, *, label: str):
    """Yield validated objects without loading a long recording timeline."""

    try:
        with path.open("r", encoding="utf-8") as fp:
            seen = False
            for line_number, line in enumerate(fp, start=1):
                if not line.strip():
                    raise RuntimeError(
                        f"diagnostic audio {label} has an empty row at line {line_number}"
                    )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"diagnostic audio {label} is invalid at line {line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"diagnostic audio {label} row {line_number} is not an object"
                    )
                seen = True
                yield line_number, row
            if not seen:
                raise RuntimeError(f"diagnostic audio {label} is empty")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"diagnostic audio {label} is unreadable") from exc


def _default_writer_factory(path: Path, fps: float, size: tuple[int, int]):
    import cv2

    return cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), tuple(size),
    )


def _default_video_validator(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"finalized video could not be reopened: {path}")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"video decoded an empty frame at index {count}")
            count += 1
    finally:
        capture.release()
    return count


def _default_video_mode_probe(path: Path) -> dict[str, object]:
    """Read the encoded playback timebase and dimensions from the final MP4."""

    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"finalized video mode could not be read: {path}")
    try:
        return {
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        }
    finally:
        capture.release()


class _PinnedFfmpegConcatStitcher:
    """Bounded MP4 concat-copy using one explicitly identified executable."""

    def __init__(self, executable: str | Path, expected_sha256: str) -> None:
        if executable is None or not str(executable).strip():
            raise ValueError("ffmpeg concat-copy executable path is required")
        expected = str(expected_sha256 or "").strip().lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ValueError("ffmpeg concat-copy executable SHA-256 is required")
        self.requested_path = Path(executable).expanduser()
        self.expected_sha256 = expected
        self.resolved_path: Path | None = None
        self._identity: tuple[int, int, int, int, int] | None = None
        self.provenance: dict[str, object] | None = None

    def prepare(self) -> dict[str, object]:
        resolved = self.requested_path.resolve(strict=True)
        info = resolved.stat()
        if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
            raise RuntimeError(f"ffmpeg concat-copy executable is not executable: {resolved}")
        actual = _sha256(resolved)
        if actual != self.expected_sha256:
            raise RuntimeError(
                f"ffmpeg concat-copy SHA-256 mismatch: expected {self.expected_sha256}, "
                f"got {actual}"
            )
        identity = (
            int(info.st_dev), int(info.st_ino), int(info.st_size),
            int(info.st_mtime_ns), int(info.st_ctime_ns),
        )
        self.resolved_path = resolved
        self._identity = identity
        self.provenance = {
            "method": "ffmpeg_concat_demuxer_stream_copy",
            "requested_path": str(self.requested_path),
            "resolved_path": str(resolved),
            "sha256": actual,
            "identity": list(identity),
            "shell": False,
        }
        return dict(self.provenance)

    def _verify_identity(self) -> None:
        if self.resolved_path is None or self._identity is None:
            self.prepare()
        info = self.resolved_path.stat()
        current = (
            int(info.st_dev), int(info.st_ino), int(info.st_size),
            int(info.st_mtime_ns), int(info.st_ctime_ns),
        )
        if current != self._identity or _sha256(self.resolved_path) != self.expected_sha256:
            raise RuntimeError("ffmpeg concat-copy executable changed after preflight")

    def __call__(
        self,
        segment_paths: list[Path],
        output_path: Path,
        fps: float,
        size: tuple[int, int],
    ) -> dict[str, object]:
        del fps, size
        self._verify_identity()
        if not segment_paths:
            raise RuntimeError("ffmpeg concat-copy requires at least one segment")
        output = Path(output_path).resolve()
        session = output.parent
        segment_dir = (session / "rgb_segments").resolve()
        lines = []
        total_bytes = 0
        expected_frames = 0
        for index, raw_path in enumerate(segment_paths):
            path = Path(raw_path).resolve(strict=True)
            expected_name = f"segment_{index:06d}.mp4"
            if path.parent != segment_dir or path.name != expected_name:
                raise RuntimeError("ffmpeg concat-copy segment path/index is invalid")
            # Only fixed recorder-generated relative paths enter the demuxer file.
            lines.append(f"file 'rgb_segments/{expected_name}'\n")
            total_bytes += int(path.stat().st_size)
            metadata = _default_video_mode_probe(path)
            expected_frames += int(metadata["frame_count"])
        if output.exists() or output.parent != session:
            raise RuntimeError("ffmpeg concat-copy output path is invalid or already exists")
        concat_path = session / f".rgb.concat.{uuid.uuid4().hex}.txt"
        _atomic_bytes(concat_path, "".join(lines).encode("utf-8"))
        timeout_s = max(60.0, total_bytes / (10 * 1024**2) + 30.0)
        argv = [
            str(self.resolved_path), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "1", "-i", concat_path.name,
            "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart",
            "-n", output.name,
        ]
        try:
            completed = subprocess.run(
                argv, cwd=str(session), check=False, shell=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ffmpeg concat-copy exceeded bounded timeout {timeout_s:.1f}s"
            ) from exc
        finally:
            concat_path.unlink(missing_ok=True)
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                f"ffmpeg concat-copy failed with exit {completed.returncode}: {error}"
            )
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("ffmpeg concat-copy produced no output bytes")
        metadata = _default_video_mode_probe(output)
        if int(metadata["frame_count"]) != expected_frames:
            raise RuntimeError(
                "ffmpeg concat-copy frame metadata mismatch: "
                f"{metadata['frame_count']} != {expected_frames}"
            )
        decode_argv = [
            str(self.resolved_path), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-progress", "pipe:1", "-nostats",
            "-i", output.name, "-map", "0:v:0", "-f", "null", "-",
        ]
        try:
            decode_probe = subprocess.run(
                decode_argv, cwd=str(session), check=False, shell=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ffmpeg final-container decode validation exceeded {timeout_s:.1f}s"
            ) from exc
        if decode_probe.returncode != 0:
            error = decode_probe.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                "ffmpeg final-container decode validation failed with exit "
                f"{decode_probe.returncode}: {error}"
            )
        decoded_frames = None
        for raw_line in decode_probe.stdout.decode("utf-8", errors="replace").splitlines():
            key, separator, value = raw_line.partition("=")
            if separator and key.strip() == "frame":
                try:
                    decoded_frames = int(value.strip())
                except ValueError:
                    decoded_frames = None
        if decoded_frames != expected_frames:
            raise RuntimeError(
                "ffmpeg final-container decoded frame count mismatch: "
                f"{decoded_frames} != {expected_frames}"
            )
        return {
            "frame_count": expected_frames,
            "provenance": dict(self.provenance or {}),
            "output_metadata": metadata,
            "packet_validation": {
                "verified": True,
                "decoded_frame_count": decoded_frames,
                "expected_frame_count": expected_frames,
                "method": "ffmpeg_decode_null_muxer",
            },
            "input_bytes": total_bytes,
            "output_bytes": int(output.stat().st_size),
            "timeout_s": timeout_s,
        }


class _SegmentValidationWorker:
    """Validate sealed five-second segments without blocking camera disk ingress."""

    def __init__(self, validator, mode_probe, negotiated_mode, *, capacity=2):  # noqa: ANN001
        self._validator = validator
        self._mode_probe = mode_probe
        self._negotiated_mode = negotiated_mode
        self._queue: queue.Queue = queue.Queue(maxsize=int(capacity))
        self._sentinel = object()
        self._lock = threading.Lock()
        self._results: list[dict[str, object]] = []
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="wam-segment-validator", daemon=True,
        )
        self._thread.start()

    def submit(self, payload: dict[str, object]) -> None:
        self.raise_if_error()
        try:
            self._queue.put_nowait(dict(payload))
        except queue.Full:
            raise RuntimeError(
                "video segment validation worker fell behind its bounded queue"
            ) from None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._sentinel:
                    return
                path = Path(item["path"])
                expected_count = int(item["frame_count"])
                decoded = int(self._validator(path))
                if decoded != expected_count:
                    raise RuntimeError(
                        f"video segment decoded {decoded} frames but recorded "
                        f"{expected_count}: {path}"
                    )
                mode = None
                if self._negotiated_mode is not None:
                    mode = dict(self._mode_probe(path))
                    expected_fps = float(self._negotiated_mode["fps"])
                    actual_fps = float(mode["fps"])
                    expected_size = (
                        int(self._negotiated_mode["width"]),
                        int(self._negotiated_mode["height"]),
                    )
                    actual_size = (int(mode["width"]), int(mode["height"]))
                    if actual_size != expected_size:
                        raise RuntimeError(
                            "video segment encoded shape differs from negotiated camera mode"
                        )
                    if not np.isclose(
                        actual_fps, expected_fps, rtol=0.0,
                        atol=max(0.01, expected_fps * 0.001),
                    ):
                        raise RuntimeError(
                            "video segment encoded FPS differs from negotiated camera mode"
                        )
                result = {
                    "segment_index": int(item["segment_index"]),
                    "path": str(item["relative_path"]),
                    "frame_count": expected_count,
                    "decoded_frame_count": decoded,
                    "encoded_mode": mode,
                    "sha256": str(item["sha256"]),
                    "bytes": int(item["bytes"]),
                }
                with self._lock:
                    self._results.append(result)
            except BaseException as exc:
                with self._lock:
                    self._error = exc
            finally:
                self._queue.task_done()

    def raise_if_error(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(
                f"video segment validation failed: {type(error).__name__}: {error}"
            ) from error

    def finish(self, *, timeout_s=120.0) -> list[dict[str, object]]:
        deadline = time.monotonic() + float(timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("video segment validation did not drain before timeout")
            try:
                self._queue.put(self._sentinel, timeout=min(0.1, remaining))
                break
            except queue.Full:
                self.raise_if_error()
        self._thread.join(max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise RuntimeError("video segment validation worker did not stop")
        self.raise_if_error()
        with self._lock:
            return sorted(
                (dict(item) for item in self._results),
                key=lambda item: int(item["segment_index"]),
            )

    def abort(self, *, timeout_s=5.0) -> bool:
        if not self._thread.is_alive():
            return True
        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("video segment validation abort timeout must be positive")
        deadline = time.monotonic() + timeout
        # Aborted sessions never consume queued validation results. Remove all
        # queued work so a full bounded queue cannot prevent the stop sentinel.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("video segment validation abort timed out")
        self._queue.put(self._sentinel, timeout=remaining)
        self._thread.join(max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise RuntimeError("video segment validation worker leaked after abort")
        return True


def _default_video_stitcher(
    segment_paths: list[Path],
    output_path: Path,
    fps: float,
    size: tuple[int, int],
) -> int:
    """Decode sealed MP4 segments and publish one conventional MP4 stream."""
    import cv2

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), tuple(size),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"stitched diagnostic video writer failed to open: {output_path}")
    count = 0
    try:
        for path in segment_paths:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"diagnostic video segment could not be opened: {path}")
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame is None or frame.size == 0:
                        raise RuntimeError(
                            f"diagnostic video segment decoded an empty frame: {path}"
                        )
                    height, width = frame.shape[:2]
                    if (width, height) != tuple(size):
                        raise RuntimeError(
                            f"diagnostic video segment shape {(width, height)} != {size}"
                        )
                    writer.write(frame)
                    count += 1
            finally:
                capture.release()
    finally:
        writer.release()
    return count


class _DurableJsonl:
    """Append-only JSONL whose acknowledged records have reached stable storage."""

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


def _read_segment_journal(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"video segment journal is corrupt at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"video segment journal line {line_number} is not an object"
                )
            records.append(value)
    return records


def recover_orphan_diagnostic_video(
    session_dir: str | Path,
    *,
    video_validator: Callable[[Path], int] | None = None,
    video_stitcher: Callable | None = None,
    ffmpeg_path: str | Path | None = None,
    ffmpeg_sha256: str | None = None,
) -> dict:
    """Recover the durable sealed prefix of an unsealed diagnostic recording.

    The original segments and any active ``*.inprogress.mp4`` are never removed.
    Recovery emits ``rgb.recovered.mp4`` only; it never creates a completion seal
    or changes training eligibility.
    """
    session = Path(session_dir).expanduser().resolve()
    if session.is_symlink() or not session.is_dir():
        raise RuntimeError(f"diagnostic recovery path is not a real directory: {session}")
    if (session / "DIAGNOSTIC_FINALIZATION.json").exists():
        return {"status": "already_finalized", "recovered_frame_count": 0}
    validator = video_validator or _default_video_validator
    stitcher = video_stitcher
    if stitcher is None:
        stitcher = _PinnedFfmpegConcatStitcher(ffmpeg_path, ffmpeg_sha256)
        stitcher.prepare()
    journal_path = session / "video_segments.jsonl"
    records = _read_segment_journal(journal_path)
    by_index: dict[int, dict] = {}
    for record in records:
        try:
            index = int(record["segment_index"])
            rel = str(record["path"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("video segment journal record is incomplete") from exc
        expected_rel = f"rgb_segments/segment_{index:06d}.mp4"
        if rel != expected_rel or index in by_index:
            raise RuntimeError("video segment journal has a conflicting path/index")
        by_index[index] = record

    segment_paths: list[Path] = []
    recovered_frames = 0
    width = height = None
    fps = None
    index = 0
    while index in by_index:
        record = by_index[index]
        path = session / str(record["path"])
        if path.resolve().parent != (session / "rgb_segments").resolve():
            raise RuntimeError("video segment path escapes the diagnostic session")
        if path.is_symlink() or not path.is_file():
            break
        expected_bytes = int(record.get("bytes", -1))
        expected_sha = str(record.get("sha256", ""))
        expected_count = int(record.get("frame_count", -1))
        if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha:
            raise RuntimeError(f"sealed video segment integrity mismatch: {path}")
        decoded = int(validator(path))
        if decoded != expected_count or decoded <= 0:
            raise RuntimeError(
                f"sealed video segment decoded {decoded} frames, expected {expected_count}: {path}"
            )
        record_size = (int(record.get("width", 0)), int(record.get("height", 0)))
        record_fps = float(record.get("fps", 0.0))
        if min(record_size) <= 0 or not np.isfinite(record_fps) or record_fps <= 0:
            raise RuntimeError("sealed video segment has invalid mode metadata")
        if width is None:
            width, height = record_size
            fps = record_fps
        elif record_size != (width, height) or record_fps != fps:
            raise RuntimeError("sealed video segments do not share one camera mode")
        segment_paths.append(path)
        recovered_frames += decoded
        index += 1

    if not segment_paths:
        return {
            "status": "no_recoverable_video_segments",
            "recovered_frame_count": 0,
            "source_segments": [],
        }

    output = session / "rgb.recovered.mp4"
    if output.is_file():
        decoded = int(validator(output))
        if decoded != recovered_frames:
            raise RuntimeError(
                f"existing recovered video decoded {decoded}, expected {recovered_frames}"
            )
    else:
        partial = session / "rgb.recovering.mp4"
        if partial.exists():
            if partial.is_symlink() or not partial.is_file():
                raise RuntimeError(
                    f"previous diagnostic recovery output is not a regular file: {partial}"
                )
            partial_decoded = int(validator(partial))
            if partial_decoded != recovered_frames:
                raise RuntimeError(
                    "previous diagnostic recovery output is incomplete: "
                    f"decoded {partial_decoded}, expected {recovered_frames}"
                )
        else:
            raw_stitch = stitcher(segment_paths, partial, float(fps), (width, height))
            stitched = int(
                raw_stitch.get("frame_count", -1)
                if isinstance(raw_stitch, Mapping) else raw_stitch
            )
            if stitched != recovered_frames:
                raise RuntimeError(
                    f"recovered video stitched {stitched} frames, expected {recovered_frames}"
                )
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise RuntimeError("recovered video writer produced no bytes")
        _fsync_file(partial)
        os.replace(partial, output)
        _fsync_directory(session)
        decoded = int(validator(output))
        if decoded != recovered_frames:
            raise RuntimeError(
                f"recovered video decoded {decoded} frames, expected {recovered_frames}"
            )
    return {
        "status": "recovered_video",
        "recovered_frame_count": recovered_frames,
        "source_segments": [path.relative_to(session).as_posix() for path in segment_paths],
        "output": output.relative_to(session).as_posix(),
        "sha256": _sha256(output),
    }


def _validated_provenance(value: Mapping[str, object]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("diagnostic provenance must be a mapping")
    missing = [
        name for name in _REQUIRED_PROVENANCE
        if not isinstance(value.get(name), Mapping) or not value.get(name)
    ]
    if missing:
        raise ValueError("diagnostic provenance is missing non-empty " + ", ".join(missing))
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("diagnostic provenance must be JSON-serializable") from exc


def _validated_boundary(value, *, label):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    try:
        result = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable") from exc
    return result


class DiagnosticPerceptionCapture:
    """Write and verify a short camera/model commissioning capture."""

    def __init__(
        self,
        session_dir: str | Path,
        *,
        fps: float,
        provenance: Mapping[str, object],
        writer_factory: Callable | None = None,
        video_validator: Callable[[Path], int] | None = None,
        video_mode_probe: Callable[[Path], Mapping[str, object]] | None = None,
        video_stitcher: Callable | None = None,
        ffmpeg_path: str | Path | None = None,
        ffmpeg_sha256: str | None = None,
        disk_usage: Callable = shutil.disk_usage,
        min_free_bytes: int = 5 * 1024**3,
        depth_chunk_frames: int = 30,
        checkpoint_every_frames: int = 30,
        video_segment_frames: int | None = None,
        require_depth: bool = True,
        require_robot: bool = True,
        require_force: bool = False,
        require_quality: bool = False,
        recording_boundary: Mapping[str, object] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.provenance = _validated_provenance(provenance)
        negotiated_mode = self.provenance["camera"].get("negotiated_mode")
        self.negotiated_mode = (
            dict(negotiated_mode) if isinstance(negotiated_mode, Mapping) else None
        )
        if isinstance(negotiated_mode, Mapping) and "fps" in negotiated_mode:
            try:
                negotiated_fps = float(negotiated_mode["fps"])
            except (TypeError, ValueError) as exc:
                raise ValueError("negotiated camera FPS must be finite and positive") from exc
            if not np.isfinite(negotiated_fps) or negotiated_fps <= 0:
                raise ValueError("negotiated camera FPS must be finite and positive")
            if not np.isclose(self.fps, negotiated_fps, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"capture fps {self.fps} does not match negotiated camera FPS "
                    f"{negotiated_fps}"
                )
        self._writer_factory = writer_factory or _default_writer_factory
        self._video_validator = video_validator or _default_video_validator
        self._video_mode_probe = video_mode_probe or _default_video_mode_probe
        self._bounded_concat = video_stitcher is None
        self._video_stitcher = video_stitcher
        self._ffmpeg_path = ffmpeg_path
        self._ffmpeg_sha256 = ffmpeg_sha256
        self._video_stitch_provenance: dict[str, object] | None = None
        self._disk_usage = disk_usage
        self.min_free_bytes = int(min_free_bytes)
        self.depth_chunk_frames = int(depth_chunk_frames)
        self.checkpoint_every_frames = int(checkpoint_every_frames)
        self.video_segment_frames = (
            max(1, int(round(self.fps * 5.0)))
            if video_segment_frames is None
            else int(video_segment_frames)
        )
        if (
            self.depth_chunk_frames <= 0
            or self.checkpoint_every_frames <= 0
            or self.video_segment_frames <= 0
        ):
            raise ValueError("depth/checkpoint/video segment frame intervals must be positive")
        self.require_depth = bool(require_depth)
        self.require_robot = bool(require_robot)
        self.require_force = bool(require_force)
        self.require_quality = bool(require_quality)
        self.recording_boundary = _validated_boundary(
            recording_boundary, label="recording boundary",
        )
        self._finalization_context: dict | None = None
        self._audio_summary: dict | None = None
        self._audio_authenticity: dict | None = None
        self._robot_summary: dict | None = None
        self._robot_contract = None
        self._robot_validation_receipt: dict | None = None
        self._force_summary: dict | None = None
        self._force_authenticity: dict | None = None
        self._quality_summary: dict | None = None
        self._quality_authenticity: dict | None = None
        self._clock = clock
        self._wall_time_ns = wall_time_ns

        self._state = "new"
        self._manifest: dict = {}
        self._writer = None
        self._video_size: tuple[int, int] | None = None
        self._first_frame_index: int | None = None
        self._last_frame_index: int | None = None
        self._first_host_t: float | None = None
        self._last_host_t: float | None = None
        self._max_host_gap_s = 0.0
        self._missing_camera_indexes = 0
        self._tracking_association_counts = {
            "same_camera_frame": 0,
            "carried_forward": 0,
            "unavailable": 0,
        }
        self._max_tracking_age_s = 0.0
        self._n_rgb = 0
        self._n_depth = 0
        self._depth_bytes = 0
        self._start_free_bytes: int | None = None
        self._video_segment_index = 0
        self._video_segment_frame_count = 0
        self._video_segment_first_frame_index: int | None = None
        self._video_segment_last_frame_index: int | None = None
        self._sealed_video_frames = 0
        self._sealed_video_bytes = 0
        self._sealed_video_segments_validated = False
        self._active_video_partial: Path | None = None
        self._segment_journal: _DurableJsonl | None = None
        self._segment_validator: _SegmentValidationWorker | None = None
        self._segment_validation_results: list[dict[str, object]] = []
        self._rgb_fp = None
        self._rgb_csv = None
        self._tracker_fp = None
        self._inference_fp = None
        self._last_persisted_source_tracking_index: int | None = None
        self._events_fp = None
        self._depth_fp = None
        self._depth_csv = None
        self._depth_integrity_fp = None
        self._depth_integrity_count = 0

        self._segment_dir = self.session_dir / "rgb_segments"
        self._segment_journal_path = self.session_dir / "video_segments.jsonl"
        self._segment_validation_path = (
            self.session_dir / "video_segment_validation.jsonl"
        )
        self._video_final = self.session_dir / "rgb.mp4"
        self._video_stitch_partial = self.session_dir / "rgb.finalizing.mp4"
        self._rgb_partial = self.session_dir / "rgb_times.csv.inprogress"
        self._tracker_partial = self.session_dir / "tracker.jsonl.inprogress"
        self._inference_partial = self.session_dir / "inference.jsonl.inprogress"
        self._events_partial = self.session_dir / "events.jsonl.inprogress"
        self._depth_dir = self.session_dir / "depth"
        self._depth_partial = self._depth_dir / "depth_times.csv.inprogress"
        self._depth_integrity_partial = (
            self.session_dir / "depth_integrity.jsonl.inprogress"
        )

    def start(self) -> None:
        if self._state != "new":
            raise RuntimeError(f"cannot start diagnostic capture from {self._state}")
        if not self.session_dir.is_dir():
            raise FileNotFoundError(f"session directory does not exist: {self.session_dir}")
        if self._bounded_concat:
            stitcher = _PinnedFfmpegConcatStitcher(
                self._ffmpeg_path, self._ffmpeg_sha256,
            )
            self._video_stitch_provenance = stitcher.prepare()
            self._video_stitcher = stitcher
        free = int(self._disk_usage(self.session_dir).free)
        self._start_free_bytes = free
        if free < self.min_free_bytes:
            raise RuntimeError(
                f"insufficient free disk for diagnostic capture: {free} < {self.min_free_bytes} bytes"
            )
        forbidden = (
            self.session_dir / "FINALIZATION.json",
            self.session_dir / "DIAGNOSTIC_FINALIZATION.json",
            self.session_dir / "manifest.json",
            *(self.session_dir / name for name in _ROBOT_FINAL_ARTIFACT_NAMES),
            *(self.session_dir / name for name in _FORCE_FINAL_ARTIFACT_NAMES),
            *(self.session_dir / name for name in _FORCE_OPTIONAL_ARTIFACT_NAMES),
            *(self.session_dir / name for name in _FORCE_STAGING_ARTIFACT_NAMES),
            *(self.session_dir / name for name in _QUALITY_FINAL_ARTIFACT_NAMES),
            *(self.session_dir / name for name in _QUALITY_STAGING_ARTIFACT_NAMES),
            self.session_dir / QUALITY_MODEL_INPUT_DIRNAME,
            self.session_dir / QUALITY_MODEL_INPUT_STAGING_DIRNAME,
        )
        existing = [path.name for path in forbidden if path.exists()]
        if existing:
            raise FileExistsError("diagnostic session already contains " + ", ".join(existing))

        self._segment_dir.mkdir(mode=0o750, exist_ok=False)
        self._depth_dir.mkdir(mode=0o750, exist_ok=False)
        _fsync_directory(self.session_dir)
        self._segment_journal = _DurableJsonl(self._segment_journal_path)
        self._segment_validator = _SegmentValidationWorker(
            self._video_validator, self._video_mode_probe, self.negotiated_mode,
            capacity=2,
        )
        self._rgb_fp = self._rgb_partial.open("x", newline="", encoding="utf-8")
        self._rgb_csv = csv.writer(self._rgb_fp)
        self._rgb_csv.writerow(
            ["frame_index", "host_arrival_t_s", "device_t_ms", "time_source"]
        )
        self._tracker_fp = self._tracker_partial.open("x", encoding="utf-8")
        self._inference_fp = self._inference_partial.open("x", encoding="utf-8")
        self._events_fp = self._events_partial.open("x", encoding="utf-8")
        self._depth_fp = self._depth_partial.open("x", newline="", encoding="utf-8")
        self._depth_csv = csv.writer(self._depth_fp)
        self._depth_csv.writerow(
            ["frame_index", "host_arrival_t_s", "device_t_ms", "depth_scale", "path"]
        )
        self._depth_integrity_fp = self._depth_integrity_partial.open(
            "x", encoding="utf-8",
        )
        self._state = "recording"
        self._manifest = {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "capture_mode": CAPTURE_MODE,
            "status": "recording",
            "training_eligible": False,
            "production_finalization_permitted": False,
            "started_host_t": self._clock(),
            "started_utc_ns": self._wall_time_ns(),
            "requested_fps": self.fps,
            "depth_storage": "formal_per_frame_npy_v1",
            "depth_contract": {
                "dtype": "uint16",
                "bytes_per_pixel": int(np.dtype(np.uint16).itemsize),
                "shape": (
                    [
                        int(self.negotiated_mode["height"]),
                        int(self.negotiated_mode["width"]),
                    ]
                    if isinstance(self.negotiated_mode, dict)
                    and "height" in self.negotiated_mode
                    and "width" in self.negotiated_mode
                    else None
                ),
                "one_file_per_camera_frame": True,
            },
            "require_depth": self.require_depth,
            "require_robot": self.require_robot,
            "require_force": self.require_force,
            "require_quality": self.require_quality,
            "provenance": self.provenance,
            "recording_boundary": self.recording_boundary,
            "abort_reason": None,
        }
        self._write_manifest()

    def record_event(self, payload: dict) -> None:
        if self._state != "recording" or self._events_fp is None:
            raise RuntimeError(f"cannot record event while diagnostic capture is {self._state}")
        event = dict(payload)
        event.setdefault("host_t", self._clock())
        event.setdefault("utc_ns", self._wall_time_ns())
        self._events_fp.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        self._flush_file(self._events_fp)

    def set_recording_boundary(self, payload: Mapping[str, object]) -> None:
        """Bind the camera Start boundary after durable writer initialization.

        Camera delivery cannot be armed before the files are ready without
        reintroducing the slow-Start overwrite race. The boundary is therefore a
        one-time recording-state binding and is persisted before any frame is
        accepted by the collector.
        """

        if self._state != "recording":
            raise RuntimeError(
                f"cannot bind recording boundary while diagnostic capture is {self._state}"
            )
        if self.recording_boundary is not None:
            raise RuntimeError("diagnostic recording boundary is already bound")
        boundary = _validated_boundary(payload, label="recording boundary")
        if boundary is None:
            raise ValueError("recording boundary cannot be null")
        self.recording_boundary = boundary
        self._manifest["recording_boundary"] = boundary
        self._write_manifest()

    def set_finalization_context(self, payload: Mapping[str, object]) -> None:
        if self._state != "recording":
            raise RuntimeError(
                f"cannot set finalization context while diagnostic capture is {self._state}"
            )
        self._finalization_context = _validated_boundary(
            payload, label="finalization context",
        )

    def set_audio_summary(self, payload: Mapping[str, object]) -> None:
        """Bind one already-finalized, same-session audio stream before End.

        The collector owns the sound card and closes audio at the camera End
        barrier. This capture then hashes those exact files into the same
        diagnostic manifest and seal as RGB-D; an unsealed sidecar is never
        presented as a complete recording.
        """

        if self._state != "recording":
            raise RuntimeError(
                f"cannot bind audio while diagnostic capture is {self._state}"
            )
        if self._audio_summary is not None:
            raise RuntimeError("diagnostic audio summary is already bound")
        summary = _validated_boundary(payload, label="audio summary")
        if summary is None:
            raise ValueError("audio summary cannot be null")
        self._audio_summary = summary
        try:
            self._validate_audio_artifacts()
        except BaseException:
            self._audio_summary = None
            self._audio_authenticity = None
            raise

    def set_robot_summary(
        self, payload: Mapping[str, object], *, contract=None,
    ) -> None:
        """Bind one finalized left-OpenArm sidecar before camera End.

        Historical v3 is retained only for diagnostic replay.  Production v4
        must carry the actual frozen activation contract owned by the live
        coordinator; policy fields omitted from a summary are never guessed.
        """

        if self._state != "recording":
            raise RuntimeError(
                f"cannot bind robot capture while diagnostic capture is {self._state}"
            )
        if self._robot_summary is not None:
            raise RuntimeError("diagnostic robot summary is already bound")
        summary = _validated_boundary(payload, label="robot summary")
        if summary is None:
            raise ValueError("robot summary cannot be null")
        schema = summary.get("schema")
        if (
            schema not in {
                _ROBOT_CAPTURE_SUMMARY_SCHEMA_V3,
                _ROBOT_CAPTURE_SUMMARY_SCHEMA_V4,
            }
            or summary.get("side") != "left"
            or summary.get("controller_mode") != "unilateral"
            or summary.get("leader_can_interface") != "can1"
            or summary.get("follower_can_interface") != "can3"
        ):
            raise RuntimeError(
                "robot capture requires v3 diagnostic or v4 production left "
                "can1 leader -> can3 follower"
            )
        if schema == _ROBOT_CAPTURE_SUMMARY_SCHEMA_V4:
            from ausc.perception.robot.robot_state_stream import RobotStateContract

            frozen = getattr(type(contract), "__dataclass_params__", None)
            if (
                type(contract) is not RobotStateContract
                or frozen is None
                or frozen.frozen is not True
            ):
                raise RuntimeError(
                    "v4 robot capture requires the actual frozen RobotStateContract"
                )
            self._robot_contract = contract
        else:
            self._robot_contract = None
        self._robot_summary = summary
        self._robot_validation_receipt = None
        try:
            self._validate_robot_artifacts()
        except BaseException:
            self._robot_summary = None
            self._robot_contract = None
            self._robot_validation_receipt = None
            raise

    def set_force_summary(self, payload: Mapping[str, object]) -> None:
        """Bind one finalized Hall-force sidecar before camera finalization."""

        if self._state != "recording":
            raise RuntimeError(
                f"cannot bind Hall-force capture while diagnostic capture is {self._state}"
            )
        if self._force_summary is not None:
            raise RuntimeError("diagnostic Hall-force summary is already bound")
        summary = _validated_boundary(payload, label="Hall-force summary")
        if summary is None:
            raise ValueError("Hall-force summary cannot be null")
        self._force_summary = summary
        try:
            self._validate_force_artifacts()
        except BaseException:
            self._force_summary = None
            self._force_authenticity = None
            raise

    def _validate_force_artifacts(self) -> dict | None:
        summary = self._force_summary
        if summary is None:
            self._force_authenticity = None
            present = [
                name
                for name in (
                    *_FORCE_FINAL_ARTIFACT_NAMES,
                    *_FORCE_OPTIONAL_ARTIFACT_NAMES,
                    *_FORCE_STAGING_ARTIFACT_NAMES,
                )
                if (
                    (path := self.session_dir / name).exists()
                    or path.is_symlink()
                )
            ]
            if present:
                raise RuntimeError(
                    "unbound Hall-force artifacts are present: "
                    + ", ".join(present)
                )
            if self.require_force:
                raise RuntimeError("required Hall-force capture sidecar is missing")
            return None
        try:
            verified = verify_force_capture(self.session_dir)
        except (ForceCaptureVerificationError, OSError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Hall-force capture verification failed: {exc}") from exc
        canonical = {
            "path": "force.csv",
            "t_start": float(verified.t_start),
            "t_end": float(verified.t_end),
            "n_samples": int(verified.n_samples),
            "status": str(verified.status),
            "sha256": str(verified.sha256),
            "device_t_start_ms": float(verified.device_t_start_ms),
            "device_t_end_ms": float(verified.device_t_end_ms),
            "calibration_sha256": str(verified.calibration_sha256),
            "device_identity_sha256": str(verified.device_identity_sha256),
        }
        if summary != canonical:
            raise RuntimeError(
                "Hall-force summary does not match the verified capture artifacts"
            )
        self._force_authenticity = dict(verified.signal_authenticity)
        if (
            self.require_force
            and self._force_authenticity.get("status") != "PASS"
        ):
            raise RuntimeError(
                "required Hall-force capture lacks minimum authenticity evidence"
            )
        return canonical

    def set_quality_summary(self, payload: Mapping[str, object]) -> None:
        """Bind one published, full-chain-verified Q packet log before End."""

        if self._state != "recording":
            raise RuntimeError(
                f"cannot bind quality packet log while diagnostic capture is {self._state}"
            )
        if self._quality_summary is not None:
            raise RuntimeError("diagnostic quality packet log is already bound")
        summary = _validated_boundary(payload, label="quality packet log summary")
        if summary is None:
            raise ValueError("quality packet log summary cannot be null")
        self._quality_summary = summary
        try:
            self._validate_quality_artifacts()
        except BaseException:
            self._quality_summary = None
            self._quality_authenticity = None
            raise

    def _validate_quality_artifacts(self) -> dict | None:
        summary = self._quality_summary
        if summary is None:
            self._quality_authenticity = None
            present = [
                name
                for name in (
                    *_QUALITY_FINAL_ARTIFACT_NAMES,
                    *_QUALITY_STAGING_ARTIFACT_NAMES,
                    QUALITY_MODEL_INPUT_DIRNAME,
                    QUALITY_MODEL_INPUT_STAGING_DIRNAME,
                )
                if (
                    (path := self.session_dir / name).exists()
                    or path.is_symlink()
                )
            ]
            if present:
                raise RuntimeError(
                    "unbound quality packet log artifacts are present: "
                    + ", ".join(present)
                )
            if self.require_quality:
                raise RuntimeError("required quality packet log sidecar is missing")
            return None
        try:
            log_receipt = verify_quality_packet_log(self.session_dir)
            canonical = quality_packet_log_summary(log_receipt)
        except (QualityPacketError, OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"quality packet log verification failed: {exc}"
            ) from exc
        if summary != canonical:
            raise RuntimeError(
                "quality packet log summary does not match the verified artifacts"
            )
        if self._force_summary is None:
            raise RuntimeError(
                "quality packet log is not bound to a verified Hall-force capture"
            )
        force_identity = log_receipt.get("force_input_identity")
        if (
            not isinstance(force_identity, Mapping)
            or force_identity.get("path") != "force.csv"
            or force_identity.get("force_csv_sha256")
            != self._force_summary.get("sha256")
        ):
            raise RuntimeError(
                "quality packet log force identity does not match the verified "
                "Hall-force capture"
            )
        self._quality_authenticity = json.loads(json.dumps(log_receipt))
        return canonical

    def _validate_audio_artifacts(self) -> dict | None:
        summary = self._audio_summary
        if summary is None:
            self._audio_authenticity = None
            return None
        try:
            sample_rate = _exact_audio_int(
                summary["sample_rate"], label="summary sample_rate",
            )
            input_sample_rate = _exact_audio_int(
                summary["input_sample_rate"], label="summary input_sample_rate",
            )
            n_samples = _exact_audio_int(
                summary["n_samples"], label="summary n_samples",
            )
            t_start = float(summary["t_start"])
            t_end = float(summary["t_end"])
            peak_db = float(summary["peak_db"])
            sha256 = str(summary["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("diagnostic audio summary is incomplete") from exc
        if (
            summary.get("path") != "audio.wav"
            or summary.get("status") != "OK"
            or sample_rate != _DIAGNOSTIC_AUDIO_SAMPLE_RATE
            or input_sample_rate <= 0
            or n_samples <= 0
            or not np.isfinite(t_start)
            or not np.isfinite(t_end)
            or t_end <= t_start
            or not np.isfinite(peak_db)
            or len(sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in sha256)
        ):
            if sample_rate != _DIAGNOSTIC_AUDIO_SAMPLE_RATE:
                raise RuntimeError("diagnostic audio model sample rate must be 16000")
            raise RuntimeError("diagnostic audio summary failed validation")
        expected_end = t_start + n_samples / float(sample_rate)
        if not _audio_time_matches(t_end, expected_end, sample_rate):
            raise RuntimeError(
                "diagnostic audio summary duration does not match its sample count"
            )
        required = (
            "audio.wav", "audio_capture.json", "audio_input_times.jsonl",
            "audio_times.jsonl",
        )
        for name in required:
            path = self.session_dir / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"diagnostic audio artifact is missing: {name}")
        wav_path = self.session_dir / "audio.wav"
        if _sha256(wav_path) != sha256:
            raise RuntimeError("diagnostic audio WAV SHA-256 does not match its summary")
        wav_peak_db, signal_authenticity = self._validate_audio_wav(
            wav_path, sample_rate=sample_rate, n_samples=n_samples,
        )
        if not np.isclose(peak_db, wav_peak_db, rtol=0.0, atol=1e-6):
            raise RuntimeError(
                "diagnostic audio peak summary does not match decoded WAV PCM"
            )
        try:
            metadata = json.loads(
                (self.session_dir / "audio_capture.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("diagnostic audio metadata is unreadable") from exc
        try:
            metadata_sample_rate = _exact_audio_int(
                metadata["sample_rate"], label="metadata sample_rate",
            )
            metadata_input_rate = _exact_audio_int(
                metadata["input_sample_rate"], label="metadata input_sample_rate",
            )
            partial_sample_rate = _exact_audio_int(
                metadata["partial_sample_rate"], label="metadata partial_sample_rate",
            )
            metadata_samples = _exact_audio_int(
                metadata["n_samples"], label="metadata n_samples",
            )
            partial_samples = _exact_audio_int(
                metadata["partial_n_samples"], label="metadata partial_n_samples",
            )
            recording_t0 = float(metadata["t0_monotonic"])
            metadata_t_start = float(metadata["audio_t0_monotonic"])
            metadata_t_end = float(metadata["audio_t_end_monotonic"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("diagnostic audio metadata is incomplete") from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != "wam_audio_stream_capture_v1"
            or metadata.get("status") != "sensors_finalized"
            or metadata.get("training_eligible") is not False
            or metadata_sample_rate != sample_rate
            or metadata_input_rate != input_sample_rate
            or partial_sample_rate != input_sample_rate
            or metadata.get("partial_encoding") != "signed_pcm_s16le_mono"
            or metadata_samples != n_samples
            or partial_samples <= 0
            or not np.isfinite(recording_t0)
            or not np.isfinite(metadata_t_start)
            or not np.isfinite(metadata_t_end)
            or not _audio_time_matches(metadata_t_start, t_start, sample_rate)
            or not _audio_time_matches(metadata_t_end, t_end, sample_rate)
            or metadata.get("sha256") != sha256
            or metadata.get("signal_authenticity") != signal_authenticity
        ):
            raise RuntimeError("diagnostic audio metadata does not match its summary")
        start_tolerance = max(
            1e-9, 1.0 / (float(input_sample_rate) * 1_000.0),
        )
        if (
            metadata_t_start < recording_t0 - start_tolerance
            or metadata_t_start > recording_t0 + 0.050 + start_tolerance
        ):
            raise RuntimeError(
                "diagnostic audio start is outside the producer 50 ms t0 coverage"
            )
        self._validate_audio_model_timeline(
            sample_rate=sample_rate,
            input_sample_rate=input_sample_rate,
            n_samples=n_samples,
            input_samples=partial_samples,
            t_start=t_start,
            t_end=t_end,
        )
        self._validate_audio_input_timeline(
            input_sample_rate=input_sample_rate,
            input_samples=partial_samples,
            t_start=metadata_t_start,
            t_end=metadata_t_end,
            output_sample_rate=sample_rate,
        )
        self._audio_authenticity = signal_authenticity
        return summary

    def _validate_robot_artifacts(self) -> dict | None:
        summary = self._robot_summary
        if summary is None:
            if self.require_robot:
                raise RuntimeError("required robot capture sidecar is missing")
            return None
        if summary.get("schema") == _ROBOT_CAPTURE_SUMMARY_SCHEMA_V4:
            return self._validate_v4_robot_artifacts(summary)
        if (
            summary.get("schema") != _ROBOT_CAPTURE_SUMMARY_SCHEMA
            or summary.get("status")
            != "exact_control_capture_validated_socketcan_tx_enqueue"
            or summary.get("side") != "left"
            or summary.get("controller_mode") != "unilateral"
            or summary.get("leader_can_interface") != "can1"
            or summary.get("follower_can_interface") != "can3"
        ):
            raise RuntimeError(
                "production robot capture requires v3 left can1 leader -> can3 follower"
            )
        for field, label in (
            ("urdf_sha256", "URDF"),
            ("joint_mapper_sha256", "joint mapper"),
            ("camera_profile_sha256", "camera profile"),
            ("timing_artifact_sha256", "timing artifact"),
            ("pair_contract_sha256", "pair contract"),
            ("controller_binary_sha256", "controller binary"),
            ("controller_source_manifest_sha256", "controller source manifest"),
            ("follower_config_sha256", "follower config"),
            ("openarm_sdk_sha256", "OpenArm SDK"),
        ):
            digest = summary.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
            ):
                raise RuntimeError(f"robot {label} SHA-256 is invalid")
        canonical_joints = [f"openarm_left_joint{index}" for index in range(1, 8)]
        if summary.get("joint_names") != canonical_joints:
            raise RuntimeError("robot summary joint order is not canonical left OpenArm")
        controller_pid = summary.get("controller_pid")
        controller_start_token = summary.get("controller_start_token")
        if (
            type(controller_pid) is not int
            or not 0 < controller_pid < 2**31
            or not isinstance(controller_start_token, str)
            or not controller_start_token
            or len(controller_start_token) > 128
        ):
            raise RuntimeError("robot controller process identity is invalid")
        expected_publisher_identity = (
            f"pid={controller_pid};start={controller_start_token};"
            "mode=unilateral;leader_can=can1;follower_can=can3;"
            f"urdf={summary['urdf_sha256']};"
            f"mapper={summary['joint_mapper_sha256']};"
            f"pair={summary['pair_contract_sha256']};"
            f"binary={summary['controller_binary_sha256']};"
            f"source={summary['controller_source_manifest_sha256']};"
            f"config={summary['follower_config_sha256']};"
            f"sdk={summary['openarm_sdk_sha256']}"
        )
        if summary.get("publisher_identity") != expected_publisher_identity:
            raise RuntimeError("robot publisher identity does not match source closure")
        for field in (
            "ring_overflow_count",
            "publisher_rejected_record_count",
            "socket_send_failure_count",
        ):
            if type(summary.get(field)) is not int or summary[field] != 0:
                raise RuntimeError(f"robot {field} must remain zero")
        if summary.get("send_status_observable") is not True:
            raise RuntimeError("robot SocketCAN TX status must be observable")
        if summary.get("send_status_reason") != _ROBOT_TX_STATUS_REASON:
            raise RuntimeError("robot send status reason does not match OpenArm CAN 1.2.2")
        if summary.get("send_status_semantics") != _ROBOT_TX_STATUS_SEMANTICS:
            raise RuntimeError("robot send status is not SocketCAN TX enqueue evidence")
        if summary.get("motor_ack_observable") is not False:
            raise RuntimeError("robot summary must not claim a motor acknowledgement")
        if summary.get("motor_ack_reason") != _ROBOT_MOTOR_ACK_REASON:
            raise RuntimeError("robot motor acknowledgement limitation is missing")
        if summary.get("feedback_liveness_observable") is not False:
            raise RuntimeError(
                "robot v3 feedback liveness must remain unobservable"
            )
        if (
            summary.get("feedback_liveness_reason")
            != _ROBOT_FEEDBACK_LIVENESS_REASON
        ):
            raise RuntimeError("robot feedback liveness reason is invalid")
        if summary.get("training_eligible") is not False:
            raise RuntimeError("diagnostic robot summary must be training_eligible=false")
        blockers = summary.get("eligibility_blockers")
        if blockers != ["robot_feedback_liveness_unobservable"]:
            raise RuntimeError(
                "diagnostic robot summary must retain the feedback liveness blocker"
            )
        if summary.get("action_semantics") != "exact_mitparam_consumed_at_control_tick":
            raise RuntimeError("robot action semantics must retain exact MIT parameters")
        if summary.get("action_interpolation") != "forbidden":
            raise RuntimeError("robot action interpolation must remain forbidden")
        integer_fields = (
            "envelope_count", "control_tick_count", "control_tick_first",
            "control_tick_last", "frame_count",
        )
        if any(type(summary.get(field)) is not int for field in integer_fields):
            raise RuntimeError("robot summary count and range fields must be exact integers")
        if (
            summary["envelope_count"] <= 0
            or summary["control_tick_count"] <= 0
            or summary["frame_count"] <= 0
            or summary["control_tick_first"] < 0
            or summary["control_tick_last"] < summary["control_tick_first"]
            or summary["control_tick_last"] - summary["control_tick_first"] + 1
            != summary["control_tick_count"]
        ):
            raise RuntimeError("robot summary count and timeline ranges are not closed")
        tx_counts = (
            "arm_tx_expected_frame_count",
            "arm_tx_full_frame_write_count",
            "arm_tx_failed_frame_write_count",
            "gripper_tx_expected_frame_count",
            "gripper_tx_full_frame_write_count",
            "gripper_tx_failed_frame_write_count",
            "socketcan_tx_enqueue_failure_tick_count",
        )
        if any(type(summary.get(field)) is not int for field in tx_counts):
            raise RuntimeError("robot SocketCAN TX counts must be exact integers")
        tick_count = summary["control_tick_count"]
        if (
            summary["arm_tx_expected_frame_count"] != tick_count * 7
            or summary["arm_tx_full_frame_write_count"] != tick_count * 7
            or summary["arm_tx_failed_frame_write_count"] != 0
            or summary["gripper_tx_expected_frame_count"] != tick_count
            or summary["gripper_tx_full_frame_write_count"] != tick_count
            or summary["gripper_tx_failed_frame_write_count"] != 0
            or summary["socketcan_tx_enqueue_failure_tick_count"] != 0
        ):
            raise RuntimeError(
                "robot SocketCAN TX counts do not prove every arm and gripper frame queued"
            )
        inventory = summary.get("hash_inventory")
        if (
            not isinstance(inventory, dict)
            or set(inventory) != set(_ROBOT_DATA_ARTIFACT_NAMES)
        ):
            raise RuntimeError("robot capture hash inventory is incomplete")
        for name, item in inventory.items():
            if not isinstance(item, dict):
                raise RuntimeError(f"robot capture hash inventory is invalid for {name}")
            expected_size = item.get("size_bytes")
            expected_sha = item.get("sha256")
            if (
                type(expected_size) is not int
                or expected_size <= 0
                or not isinstance(expected_sha, str)
                or len(expected_sha) != 64
                or any(ch not in "0123456789abcdef" for ch in expected_sha)
            ):
                raise RuntimeError(f"robot capture hash inventory is invalid for {name}")

        if self._robot_validation_receipt is None:
            self._bind_robot_artifact_receipt(summary)
        else:
            self._validate_robot_artifact_receipt(summary)
        return summary

    @staticmethod
    def _robot_stat_identity(info) -> tuple[int, ...]:
        return (
            int(info.st_dev), int(info.st_ino), int(info.st_size),
            int(info.st_mtime_ns), int(info.st_ctime_ns), int(info.st_mode),
        )

    @classmethod
    def _robot_artifact_identity(cls, path: Path) -> tuple[int, ...]:
        try:
            info = path.lstat()
        except OSError as exc:
            raise RuntimeError(f"robot capture artifact is missing: {path.name}") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise RuntimeError(
                f"robot capture artifact is not a non-empty regular file: {path.name}"
            )
        return cls._robot_stat_identity(info)

    def _robot_artifact_identities(self) -> dict[str, tuple[int, ...]]:
        return {
            name: self._robot_artifact_identity(self.session_dir / name)
            for name in _ROBOT_FINAL_ARTIFACT_NAMES
        }

    def _open_robot_artifact_descriptors(self) -> dict[str, int]:
        descriptors: dict[str, int] = {}
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            for name in _ROBOT_FINAL_ARTIFACT_NAMES:
                descriptor = os.open(str(self.session_dir / name), flags)
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                    os.close(descriptor)
                    raise RuntimeError(
                        f"robot capture artifact is not a non-empty regular file: {name}"
                    )
                descriptors[name] = descriptor
        except BaseException:
            for descriptor in descriptors.values():
                os.close(descriptor)
            raise
        return descriptors

    def _robot_descriptor_identities(
        self, descriptors: Mapping[str, int],
    ) -> dict[str, tuple[int, ...]]:
        return {
            name: self._robot_stat_identity(os.fstat(descriptor))
            for name, descriptor in descriptors.items()
        }

    def _robot_descriptor_sha256(self, descriptor: int) -> str:
        before = self._robot_stat_identity(os.fstat(descriptor))
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        after = self._robot_stat_identity(os.fstat(descriptor))
        if before != after:
            raise RuntimeError("robot artifact changed while hashing pinned descriptor")
        return digest.hexdigest()

    @staticmethod
    def _robot_contract_from_summary(summary: Mapping[str, object]):  # noqa: ANN205
        """Reconstruct the hash-pinned parser contract for independent replay.

        The activated physical bounds are persisted in the sidecar and reused
        verbatim here.  The independent replay must reject values that the
        recording controller's activated contract would have rejected.
        """

        from ausc.perception.robot.robot_state_stream import RobotStateContract

        try:
            return RobotStateContract(
                side="left",
                controller_mode="unilateral",
                joint_names=tuple(summary["joint_names"]),
                joint_position_limits_rad=tuple(
                    tuple(pair)
                    for pair in summary["joint_position_limits_rad"]
                ),
                leader_can_interface="can1",
                follower_can_interface="can3",
                urdf_sha256=str(summary["urdf_sha256"]),
                joint_mapper_sha256=str(summary["joint_mapper_sha256"]),
                camera_profile_sha256=str(summary["camera_profile_sha256"]),
                timing_artifact_sha256=str(summary["timing_artifact_sha256"]),
                pair_contract_sha256=str(summary["pair_contract_sha256"]),
                controller_binary_sha256=str(summary["controller_binary_sha256"]),
                controller_source_manifest_sha256=str(
                    summary["controller_source_manifest_sha256"]
                ),
                follower_config_sha256=str(summary["follower_config_sha256"]),
                openarm_sdk_sha256=str(summary["openarm_sdk_sha256"]),
                max_abs_joint_velocity_rad_s=summary[
                    "max_abs_joint_velocity_rad_s"
                ],
                gripper_motor_position_limits_rad=tuple(
                    summary["gripper_motor_position_limits_rad"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "robot summary activated safety bounds are invalid"
            ) from exc

    def _validate_robot_camera_coverage(
        self, summary: Mapping[str, object], *, artifact_fd: int | None = None,
    ) -> None:
        """Close the robot frame index on this camera capture in O(1) memory."""

        if self._first_frame_index is None or self._last_frame_index is None:
            raise RuntimeError("robot frame index cannot bind an empty camera capture")
        path = self.session_dir / "robot_frame_index.jsonl"
        expected_index = self._first_frame_index
        seen = 0
        descriptor = None
        try:
            descriptor = (
                os.open(
                    str(path),
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                if artifact_fd is None
                else os.dup(artifact_fd)
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeError("robot frame index must be a regular file")
            os.lseek(descriptor, 0, os.SEEK_SET)
            handle = os.fdopen(descriptor, "rb")
            descriptor = None
            with handle:
                while True:
                    raw = handle.readline(_MAX_ROBOT_FRAME_INDEX_LINE_BYTES + 1)
                    if not raw:
                        break
                    if (
                        len(raw) > _MAX_ROBOT_FRAME_INDEX_LINE_BYTES
                        or not raw.endswith(b"\n")
                        or b"\r" in raw
                        or raw == b"\n"
                    ):
                        raise RuntimeError("robot frame index has invalid bounded JSONL framing")
                    try:
                        row = json.loads(
                            raw[:-1].decode("utf-8"),
                            object_pairs_hook=_strict_robot_json_object,
                            parse_constant=_reject_robot_json_constant,
                        )
                    except RuntimeError:
                        raise
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise RuntimeError(
                            f"robot frame index row {seen + 1} is invalid"
                        ) from exc
                    if not isinstance(row, dict):
                        raise RuntimeError("robot frame index row must be an object")
                    frame_index = row.get("frame_index")
                    if type(frame_index) is not int or frame_index != expected_index:
                        raise RuntimeError(
                            "robot frame index does not close on camera capture"
                        )
                    expected_index += 1
                    seen += 1
        except RuntimeError:
            raise
        except OSError as exc:
            raise RuntimeError("robot frame index is unreadable") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if (
            seen != summary["frame_count"]
            or seen != self._n_rgb
            or expected_index - 1 != self._last_frame_index
        ):
            raise RuntimeError("robot frame index count does not close on camera capture")

    def _bind_robot_artifact_receipt(
        self, summary: Mapping[str, object],
    ) -> None:
        """Run the one semantic replay and bind immutable final-file identities."""

        is_v4 = summary.get("schema") == _ROBOT_CAPTURE_SUMMARY_SCHEMA_V4
        descriptors = None
        before = self._robot_artifact_identities()
        contract = (
            self._robot_contract
            if is_v4
            else self._robot_contract_from_summary(summary)
        )
        try:
            from ausc.perception.robot import robot_session_capture

            if is_v4:
                descriptors = self._open_robot_artifact_descriptors()
                if self._robot_descriptor_identities(descriptors) != before:
                    raise RuntimeError(
                        "robot artifact path changed while pinning descriptors"
                    )
                validated = robot_session_capture.validate_robot_capture_artifacts(
                    self.session_dir, contract, artifact_fds=descriptors,
                )
            else:
                validated = robot_session_capture.validate_robot_capture_artifacts(
                    self.session_dir, contract,
                )
        except Exception as exc:  # canonical validator owns its domain exception
            if descriptors is not None:
                for descriptor in descriptors.values():
                    os.close(descriptor)
                descriptors = None
            raise RuntimeError(f"robot canonical replay failed: {exc}") from exc
        try:
            if validated != summary:
                raise RuntimeError(
                    "robot canonical replay summary differs from bound summary"
                )
            self._validate_robot_camera_coverage(
                summary,
                artifact_fd=(
                    None
                    if descriptors is None
                    else descriptors["robot_frame_index.jsonl"]
                ),
            )

            summary_sha256 = (
                _sha256(self.session_dir / "robot_capture_summary.json")
                if descriptors is None
                else self._robot_descriptor_sha256(
                    descriptors["robot_capture_summary.json"]
                )
            )
            after = self._robot_artifact_identities()
            if before != after or (
                descriptors is not None
                and self._robot_descriptor_identities(descriptors) != after
            ):
                raise RuntimeError("robot artifacts changed during canonical replay")
            inventory = summary["hash_inventory"]
            artifacts = {}
            for name in _ROBOT_DATA_ARTIFACT_NAMES:
                item = inventory[name]
                artifacts[name] = {
                    "bytes": int(item["size_bytes"]),
                    "sha256": str(item["sha256"]),
                    "identity": list(after[name]),
                }
            artifacts["robot_capture_summary.json"] = {
                "bytes": int(after["robot_capture_summary.json"][2]),
                "sha256": summary_sha256,
                "identity": list(after["robot_capture_summary.json"]),
            }
            self._robot_validation_receipt = {
                "summary": dict(summary),
                "artifacts": artifacts,
            }
        finally:
            if descriptors is not None:
                for descriptor in descriptors.values():
                    os.close(descriptor)

    def _validate_v4_robot_artifacts(
        self, summary: Mapping[str, object],
    ) -> dict:
        """Fail closed on v4 receipts, then delegate truth to raw replay."""

        from ausc.perception.robot.robot_state_stream import RobotStateContract

        contract = self._robot_contract
        if type(contract) is not RobotStateContract:
            raise RuntimeError(
                "v4 robot capture requires the actual frozen RobotStateContract"
            )
        if (
            summary.get("status")
            != "exact_control_capture_validated_feedback_liveness"
            or summary.get("side") != "left"
            or summary.get("controller_mode") != "unilateral"
            or summary.get("leader_can_interface") != "can1"
            or summary.get("follower_can_interface") != "can3"
        ):
            raise RuntimeError(
                "v4 robot capture lacks clean left OpenArm feedback liveness"
            )
        if (
            summary.get("feedback_liveness_observable") is not True
            or summary.get("feedback_liveness_reason")
            != "openarm_can_1_2_2_matching_motor_id_valid_state_decode"
            or summary.get("feedback_liveness_semantics")
            != "counter_generation_increment_only_after_matching_motor_id_valid_state_decode"
            or summary.get("feedback_counter_progress_all_motors") is not True
            or summary.get("feedback_all_ticks_fresh") is not True
            or summary.get("feedback_gap_clean") is not True
            or summary.get("feedback_clean") is not True
            or summary.get("feedback_gap_violation_count_delta") != [0] * 8
        ):
            raise RuntimeError("v4 robot feedback liveness receipt is not PASS")
        if (
            summary.get("trajectory_validation_schema")
            != "wam_openarm_trajectory_validation_v1"
            or summary.get("trajectory_nondegenerate") is not True
            or summary.get("command_response_consistent") is not True
            or summary.get("command_response_violation_joint_names") != []
            or summary.get("training_eligible") is not True
            or summary.get("eligibility_blockers") != []
        ):
            raise RuntimeError(
                "v4 robot summary is not training eligible or has a degenerate trajectory"
            )
        policy_fields = {
            "buffer_duration_ns": contract.buffer_duration_ns,
            "max_packets": contract.max_packets,
            "max_gap_ns": contract.max_gap_ns,
            "max_staleness_ns": contract.max_staleness_ns,
            "max_exposure_interval_ns": contract.max_exposure_interval_ns,
        }
        if any(summary.get(name) != value for name, value in policy_fields.items()):
            raise RuntimeError(
                "v4 robot summary does not bind the activated contract policy"
            )
        if self._robot_validation_receipt is None:
            self._bind_robot_artifact_receipt(summary)
        else:
            self._validate_robot_artifact_receipt(summary)
        return dict(summary)

    def _validate_robot_artifact_receipt(
        self, summary: Mapping[str, object],
    ) -> None:
        """Perform cheap final identity/commit closure without replay or rehash."""

        receipt = self._robot_validation_receipt
        if not isinstance(receipt, dict) or receipt.get("summary") != summary:
            raise RuntimeError("robot canonical validation receipt is missing or changed")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(
            _ROBOT_FINAL_ARTIFACT_NAMES
        ):
            raise RuntimeError("robot canonical validation receipt is incomplete")
        identities = self._robot_artifact_identities()
        for name, item in artifacts.items():
            if (
                not isinstance(item, dict)
                or item.get("identity") != list(identities[name])
                or item.get("bytes") != identities[name][2]
            ):
                raise RuntimeError(
                    f"robot artifact identity changed after binding: {name}"
                )
        summary_path = self.session_dir / "robot_capture_summary.json"
        try:
            disk_summary = json.loads(
                summary_path.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_robot_json_object,
                parse_constant=_reject_robot_json_constant,
            )
        except RuntimeError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("robot capture summary commit marker is unreadable") from exc
        if disk_summary != summary:
            raise RuntimeError("robot capture summary commit marker changed after binding")


    @staticmethod
    def _validate_audio_wav(
        path: Path, *, sample_rate: int, n_samples: int,
    ) -> tuple[float, dict[str, object]]:
        try:
            with wave.open(str(path), "rb") as wav:
                if wav.getnchannels() != 1:
                    raise RuntimeError("diagnostic audio WAV must be mono")
                if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                    raise RuntimeError("diagnostic audio WAV must be PCM16")
                if wav.getframerate() != sample_rate:
                    raise RuntimeError(
                        "diagnostic audio WAV sample rate does not match its summary"
                    )
                if wav.getnframes() != n_samples:
                    raise RuntimeError(
                        "diagnostic audio WAV frame count does not match its summary"
                    )
                decoded = 0
                peak = 0
                authenticity = AudioSignalAuthenticityAccumulator(sample_rate)
                while decoded < n_samples:
                    raw = wav.readframes(min(65_536, n_samples - decoded))
                    if not raw or len(raw) % 2:
                        break
                    pcm = np.frombuffer(raw, dtype="<i2")
                    decoded += int(pcm.size)
                    if pcm.size:
                        widened = pcm.astype(np.int32)
                        peak = max(
                            peak, int(np.max(np.abs(widened))),
                        )
                        authenticity.update(pcm)
                if decoded != n_samples:
                    raise RuntimeError(
                        "diagnostic audio WAV decoded frame count is incomplete"
                    )
                signal_authenticity = authenticity.finish()
        except AudioSignalAuthenticityError as exc:
            if "all-zero PCM" in str(exc):
                raise RuntimeError(
                    f"diagnostic audio WAV is silent; {exc}"
                ) from exc
            raise RuntimeError(str(exc)) from exc
        except RuntimeError:
            raise
        except (EOFError, OSError, wave.Error) as exc:
            raise RuntimeError("diagnostic audio WAV could not be decoded") from exc
        return (
            float(20.0 * np.log10(max(peak / 32_767.0, 1e-9))),
            signal_authenticity,
        )

    def _validate_audio_model_timeline(
        self, *, sample_rate: int, input_sample_rate: int, n_samples: int,
        input_samples: int, t_start: float, t_end: float,
    ) -> None:
        expected_offset = 0
        source_input_samples = 0
        path = self.session_dir / "audio_times.jsonl"
        for line_number, row in _stream_audio_jsonl(path, label="timeline"):
            try:
                offset = _exact_audio_int(
                    row["sample_offset"], label="timeline sample_offset",
                )
                count = _exact_audio_int(
                    row["sample_count"], label="timeline sample_count",
                )
                row_sample_rate = _exact_audio_int(
                    row["sample_rate"], label="timeline sample_rate",
                )
                row_input_rate = _exact_audio_int(
                    row["input_sample_rate"], label="timeline input_sample_rate",
                )
                source_count = _exact_audio_int(
                    row["source_input_sample_count"],
                    label="timeline source_input_sample_count",
                )
                row_start = float(row["t_start"])
                row_end = float(row["t_end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"diagnostic audio timeline row {line_number} is incomplete"
                ) from exc
            expected_start = t_start + expected_offset / float(sample_rate)
            expected_end = expected_start + count / float(sample_rate)
            if (
                offset != expected_offset
                or count <= 0
                or source_count <= 0
                or row_sample_rate != sample_rate
                or row_input_rate != input_sample_rate
                or not np.isfinite(row_start)
                or not np.isfinite(row_end)
                or not _audio_time_matches(row_start, expected_start, sample_rate)
                or not _audio_time_matches(row_end, expected_end, sample_rate)
            ):
                raise RuntimeError(
                    f"diagnostic audio timeline row {line_number} is inconsistent"
                )
            expected_offset += count
            source_input_samples += source_count
        if (
            expected_offset != n_samples
            or not _audio_time_matches(
                t_start + expected_offset / float(sample_rate), t_end, sample_rate,
            )
        ):
            raise RuntimeError("diagnostic audio timeline totals do not match summary")
        if source_input_samples != input_samples:
            raise RuntimeError(
                "diagnostic audio metadata input count does not match audio timeline"
            )

    def _validate_audio_input_timeline(
        self, *, input_sample_rate: int, input_samples: int, t_start: float,
        t_end: float, output_sample_rate: int,
    ) -> None:
        expected_offset = 0
        previous_source_end: float | None = None
        pending_source_row: tuple[int, int, float, float] | None = None
        source_gap_tolerance = max(2.0 / float(input_sample_rate), 0.010)
        path = self.session_dir / "audio_input_times.jsonl"
        for line_number, row in _stream_audio_jsonl(path, label="input timeline"):
            try:
                offset = _exact_audio_int(
                    row["input_sample_offset"],
                    label="input timeline input_sample_offset",
                )
                count = _exact_audio_int(
                    row["input_sample_count"],
                    label="input timeline input_sample_count",
                )
                row_start = float(row["t_start"])
                row_end = float(row["t_end"])
                source_start = float(row["source_block_start_t"])
                source_end = float(row["source_block_end_t"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"diagnostic audio input timeline row {line_number} is incomplete"
                ) from exc
            expected_start = t_start + expected_offset / float(input_sample_rate)
            expected_end = expected_start + count / float(input_sample_rate)
            if (
                offset != expected_offset
                or count <= 0
                or not np.isfinite(row_start)
                or not np.isfinite(row_end)
                or not np.isfinite(source_start)
                or not np.isfinite(source_end)
                or source_end <= source_start
                or not _audio_time_matches(
                    row_start, expected_start, input_sample_rate,
                )
                or not _audio_time_matches(row_end, expected_end, input_sample_rate)
            ):
                raise RuntimeError(
                    f"diagnostic audio input timeline row {line_number} is inconsistent"
                )
            if previous_source_end is None:
                if not _audio_time_matches(
                    source_start, t_start, input_sample_rate,
                ):
                    raise RuntimeError(
                        "diagnostic audio input source clock does not start at audio t0"
                    )
            elif abs(source_start - previous_source_end) > source_gap_tolerance:
                raise RuntimeError(
                    "diagnostic audio input source clock discontinuity exceeds "
                    f"{source_gap_tolerance:.6f} s"
                )
            if pending_source_row is not None:
                prior_line, prior_count, prior_start, prior_end = pending_source_row
                expected_duration = prior_count / float(input_sample_rate)
                if not _audio_time_matches(
                    prior_end - prior_start, expected_duration, input_sample_rate,
                ):
                    raise RuntimeError(
                        "diagnostic audio input source clock duration is invalid at "
                        f"line {prior_line}"
                    )
            pending_source_row = (line_number, count, source_start, source_end)
            previous_source_end = source_end
            expected_offset += count
        if pending_source_row is not None:
            final_line, final_count, final_start, final_end = pending_source_row
            minimum_duration = final_count / float(input_sample_rate)
            duration_tolerance = max(
                1e-9, 1.0 / (float(input_sample_rate) * 1_000.0),
            )
            if final_end - final_start < minimum_duration - duration_tolerance:
                raise RuntimeError(
                    "diagnostic audio input source clock duration is shorter than "
                    f"the retained final row at line {final_line}"
                )
        input_end = t_start + expected_offset / float(input_sample_rate)
        boundary_tolerance = (
            1.0 / float(input_sample_rate)
            + 1.0 / float(output_sample_rate)
            + 1e-9
        )
        if (
            expected_offset != input_samples
            or not np.isclose(input_end, t_end, rtol=0.0, atol=boundary_tolerance)
        ):
            raise RuntimeError(
                "diagnostic audio input timeline totals do not match metadata"
            )

    def write_frame(self, frame, tracker_output: dict) -> bool:  # noqa: ANN001
        if self._state != "recording":
            raise RuntimeError(f"cannot write frame while diagnostic capture is {self._state}")
        try:
            return self._write_frame(frame, tracker_output)
        except Exception as exc:
            self.abort(f"frame write failed: {type(exc).__name__}: {exc}")
            raise

    def _write_frame(self, frame, tracker_output: dict) -> bool:  # noqa: ANN001
        if self._segment_validator is not None:
            self._segment_validator.raise_if_error()
        index = int(getattr(frame, "index"))
        if self._last_frame_index is not None:
            if index == self._last_frame_index:
                return False
            if index < self._last_frame_index:
                raise RuntimeError(
                    f"camera frame index regressed from {self._last_frame_index} to {index}"
                )
        if str(getattr(frame, "status", "NO_DATA")) != "OK":
            raise RuntimeError(f"camera frame {index} is not live: {getattr(frame, 'status', None)}")
        rgb = getattr(frame, "frame", None)
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise RuntimeError(f"camera frame {index} is not an RGB uint8 array")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise RuntimeError(f"camera frame {index} has invalid RGB shape {getattr(rgb, 'shape', None)}")
        host_t = float(getattr(frame, "t"))
        if not np.isfinite(host_t):
            raise RuntimeError(f"camera frame {index} has non-finite host timestamp")
        if self._last_host_t is not None:
            gap = host_t - self._last_host_t
            if not np.isfinite(gap) or gap <= 0:
                raise RuntimeError(
                    f"camera host timestamp did not increase: {self._last_host_t} -> {host_t}"
                )
            self._max_host_gap_s = max(self._max_host_gap_s, gap)
        if self._last_frame_index is not None:
            self._missing_camera_indexes += max(0, index - self._last_frame_index - 1)
        if self._first_host_t is None:
            self._first_host_t = host_t
            self._first_frame_index = index
        self._last_host_t = host_t
        tracking = dict(tracker_output or {})
        tracked_index = tracking.get("frame_index", index)
        if int(tracked_index) != index:
            raise RuntimeError(
                f"tracker frame index {tracked_index} does not match camera frame {index}"
            )
        association = str(tracking.get("tracking_association", ""))
        if association not in {"same_camera_frame", "carried_forward", "unavailable"}:
            raise RuntimeError(
                f"camera frame {index} has invalid explicit tracking association {association!r}"
            )
        source_raw = tracking.get("source_tracking_frame_index")
        if association == "unavailable":
            if source_raw is not None:
                raise RuntimeError(
                    f"unavailable tracking for camera frame {index} declares a source frame"
                )
            source_index = None
        else:
            if source_raw is None:
                raise RuntimeError(
                    f"camera frame {index} is missing source_tracking_frame_index"
                )
            source_index = int(source_raw)
            if source_index > index:
                raise RuntimeError(
                    f"future tracking frame {source_index} cannot annotate camera frame {index}"
                )
            if association == "same_camera_frame" and source_index != index:
                raise RuntimeError(
                    f"same-frame tracking source {source_index} does not match camera frame {index}"
                )
            if association == "carried_forward" and source_index >= index:
                raise RuntimeError(
                    f"carried-forward tracking source {source_index} is not older than camera frame {index}"
                )
            try:
                tracking_age = float(tracking["tracking_age_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"camera frame {index} has no finite tracking_age_s"
                ) from exc
            if not np.isfinite(tracking_age) or tracking_age < 0:
                raise RuntimeError(
                    f"camera frame {index} has invalid tracking_age_s {tracking_age}"
                )
            self._max_tracking_age_s = max(self._max_tracking_age_s, tracking_age)
        self._tracking_association_counts[association] += 1

        h, w = rgb.shape[:2]
        if self._writer is None:
            if self._video_size is None:
                self._video_size = (w, h)
            elif self._video_size != (w, h):
                raise RuntimeError(
                    f"RGB dimensions changed from {self._video_size} to {(w, h)}"
                )
            self._open_video_segment()
        elif self._video_size != (w, h):
            raise RuntimeError(
                f"RGB dimensions changed from {self._video_size} to {(w, h)}"
            )
        result = self._writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))
        if result is False:
            raise RuntimeError(f"video writer rejected camera frame {index}")
        # From this point the frame may already be inside the encoder even if a
        # later metadata/depth fsync fails.  Track the physical video commit
        # immediately so fail-closed abort can seal a segment whose journal
        # count matches the bytes actually handed to VideoWriter.  `_n_rgb`
        # remains the fully aligned RGB+metadata commit count and is advanced
        # only after every companion artifact succeeds below.
        if self._video_segment_frame_count == 0:
            self._video_segment_first_frame_index = index
        self._video_segment_last_frame_index = index
        self._video_segment_frame_count += 1

        device_t = getattr(frame, "device_t_ms", None)
        if device_t is not None:
            device_t = float(device_t)
            if not np.isfinite(device_t):
                device_t = None
        time_source = str(getattr(frame, "time_source", "host_arrival"))
        self._rgb_csv.writerow([index, f"{host_t:.9f}", "" if device_t is None else device_t,
                                time_source])
        tracker_row = {
            **tracking,
            "frame_index": index,
            "host_t": host_t,
            "device_t": device_t,
            "points": [dict(point) for point in (tracking.get("points") or [])],
        }
        self._tracker_fp.write(
            json.dumps(tracker_row, sort_keys=True, separators=(",", ":")) + "\n"
        )
        if source_index is not None and source_index != self._last_persisted_source_tracking_index:
            if (
                self._last_persisted_source_tracking_index is not None
                and source_index < self._last_persisted_source_tracking_index
            ):
                raise RuntimeError(
                    "source tracking frame index regressed from "
                    f"{self._last_persisted_source_tracking_index} to {source_index}"
                )
            inference_row = {
                **tracking,
                "source_tracking_frame_index": source_index,
                "first_associated_camera_frame_index": index,
                "points": [dict(point) for point in (tracking.get("points") or [])],
            }
            self._inference_fp.write(
                json.dumps(inference_row, sort_keys=True, separators=(",", ":")) + "\n"
            )
            self._last_persisted_source_tracking_index = source_index

        depth = getattr(frame, "depth", None)
        if depth is None:
            if self.require_depth:
                raise RuntimeError(f"Orbbec diagnostic frame {index} is missing depth")
        else:
            if (
                not isinstance(depth, np.ndarray)
                or depth.dtype != np.uint16
                or depth.ndim != 2
                or not depth.size
            ):
                raise RuntimeError(f"camera frame {index} has invalid depth")
            if depth.shape != (h, w):
                raise RuntimeError(
                    f"RGB/depth shape mismatch at frame {index}: {(h, w)} vs {depth.shape}"
                )
            if int(np.count_nonzero(depth)) == 0:
                raise RuntimeError(
                    f"camera frame {index} has no valid depth pixels"
                )
            depth_scale = float(getattr(frame, "depth_scale", 1.0))
            if not np.isfinite(depth_scale) or depth_scale <= 0:
                raise RuntimeError(f"camera frame {index} has invalid depth scale")
            storage = self.storage_health()
            next_depth_bytes = int(depth.nbytes) + 512
            if int(storage["usable_bytes_before_safety_reserve"]) < next_depth_bytes:
                raise RuntimeError(
                    "insufficient live storage reserve for next aligned depth frame: "
                    f"{storage['usable_bytes_before_safety_reserve']} < {next_depth_bytes} bytes"
                )
            depth_name = f"depth_{index:06d}.npy"
            depth_path = self.session_dir / "depth" / depth_name
            if depth_path.exists():
                raise FileExistsError(f"diagnostic depth frame already exists: {depth_path}")
            depth_integrity = _atomic_save_npy(
                depth_path, np.ascontiguousarray(depth),
            )
            self._depth_bytes += int(depth_integrity["bytes"])
            integrity_row = {
                "schema_version": "wam_depth_frame_integrity_v1",
                "frame_index": index,
                "path": f"depth/{depth_name}",
                "dtype": "uint16",
                "shape": [int(depth.shape[0]), int(depth.shape[1])],
                **depth_integrity,
            }
            self._depth_integrity_fp.write(
                json.dumps(
                    integrity_row, sort_keys=True, separators=(",", ":"),
                ) + "\n"
            )
            self._depth_integrity_count += 1
            self._depth_csv.writerow([
                index, f"{host_t:.9f}", "" if device_t is None else device_t,
                depth_scale, f"depth/{depth_name}",
            ])
            self._n_depth += 1

        self._last_frame_index = index
        self._n_rgb += 1
        if self._video_segment_frame_count >= self.video_segment_frames:
            self._seal_video_segment()
        if self._n_rgb % self.checkpoint_every_frames == 0:
            self._checkpoint()
        return True

    def _open_video_segment(self) -> None:
        if self._writer is not None or self._video_size is None:
            raise RuntimeError("diagnostic video segment state is inconsistent")
        partial = self._segment_dir / (
            f"segment_{self._video_segment_index:06d}.inprogress.mp4"
        )
        final = self._segment_dir / f"segment_{self._video_segment_index:06d}.mp4"
        if partial.exists() or final.exists():
            raise FileExistsError(f"diagnostic video segment already exists: {partial}")
        writer = self._writer_factory(partial, self.fps, self._video_size)
        opened = writer is not None and bool(writer.isOpened())
        if not opened:
            if writer is not None:
                writer.release()
            raise RuntimeError("diagnostic video writer failed to open")
        self._active_video_partial = partial
        self._writer = writer

    def _seal_video_segment(self) -> Path:
        if self._writer is None or self._active_video_partial is None:
            raise RuntimeError("no active diagnostic video segment to seal")
        if self._video_segment_frame_count <= 0:
            raise RuntimeError("cannot seal an empty diagnostic video segment")
        writer = self._writer
        partial = self._active_video_partial
        expected_count = self._video_segment_frame_count
        first_index = self._video_segment_first_frame_index
        last_index = self._video_segment_last_frame_index
        writer.release()
        self._writer = None
        self._active_video_partial = None
        if not partial.is_file() or partial.stat().st_size <= 0:
            raise RuntimeError("diagnostic video segment writer produced no bytes")
        _fsync_file(partial)
        final = self._segment_dir / f"segment_{self._video_segment_index:06d}.mp4"
        os.replace(partial, final)
        _fsync_directory(self._segment_dir)
        if self._segment_journal is None or self._video_size is None:
            raise RuntimeError("diagnostic video segment journal is unavailable")
        width, height = self._video_size
        final_bytes = final.stat().st_size
        segment_sha256 = _sha256(final)
        journal_record = {
            "segment_index": self._video_segment_index,
            "path": final.relative_to(self.session_dir).as_posix(),
            "frame_count": expected_count,
            "first_frame_index": first_index,
            "last_frame_index": last_index,
            "fps": self.fps,
            "width": width,
            "height": height,
            "bytes": final_bytes,
            "sha256": segment_sha256,
            "validation": "pending_async",
        }
        self._segment_journal.append(journal_record)
        if self._segment_validator is None:
            raise RuntimeError("video segment validation worker is unavailable")
        self._segment_validator.submit({
            **journal_record,
            "path": str(final),
            "relative_path": journal_record["path"],
        })
        self._sealed_video_frames += expected_count
        self._sealed_video_bytes += final_bytes
        self._sealed_video_segments_validated = False
        self._video_segment_index += 1
        self._video_segment_frame_count = 0
        self._video_segment_first_frame_index = None
        self._video_segment_last_frame_index = None
        self._check_final_video_reserve()
        return final

    def finalize(self) -> dict:
        if self._state != "recording":
            raise RuntimeError(f"cannot finalize diagnostic capture from {self._state}")
        self._state = "finalizing"
        try:
            return self._finalize()
        except Exception as exc:
            self._abort_internal(f"finalization failed: {type(exc).__name__}: {exc}")
            raise

    def _finalize(self) -> dict:
        if self._n_rgb <= 0:
            raise RuntimeError("diagnostic capture has no RGB frames")
        self._validate_finalization_boundary_indexes()
        if self._missing_camera_indexes:
            raise RuntimeError(
                "camera frame index coverage is incomplete: "
                f"missing {self._missing_camera_indexes} camera frame(s)"
            )
        if self.require_depth and self._n_depth != self._n_rgb:
            raise RuntimeError(
                f"RGB/depth frame mismatch: {self._n_rgb} RGB vs {self._n_depth} depth"
            )
        if self._writer is not None:
            self._seal_video_segment()
        if self._sealed_video_frames != self._n_rgb:
            raise RuntimeError(
                f"sealed video has {self._sealed_video_frames} frames but capture has {self._n_rgb}"
            )
        segment_validation = self._finish_segment_validation()
        self._checkpoint()
        self._close_and_publish_artifacts()
        depth_integrity = self._validate_depth_integrity()
        segment_paths = self._sealed_segment_paths()
        self._checkpoint()
        if not segment_paths or self._video_size is None:
            raise RuntimeError("diagnostic capture has no sealed video segments")
        if self._video_stitch_partial.exists() or self._video_final.exists():
            raise FileExistsError("diagnostic final video output already exists")
        raw_stitch = self._video_stitcher(
            segment_paths, self._video_stitch_partial, self.fps, self._video_size,
        )
        if isinstance(raw_stitch, Mapping):
            stitched = int(raw_stitch.get("frame_count", -1))
            stitch_provenance = raw_stitch.get("provenance")
            if isinstance(stitch_provenance, Mapping):
                self._video_stitch_provenance = dict(stitch_provenance)
            stitch_summary = json.loads(json.dumps(raw_stitch))
        else:
            stitched = int(raw_stitch)
            stitch_summary = {
                "frame_count": stitched,
                "provenance": self._video_stitch_provenance,
            }
        if stitched != self._n_rgb:
            raise RuntimeError(
                f"final video stitched {stitched} frames but recorded {self._n_rgb}"
            )
        if (
            not self._video_stitch_partial.is_file()
            or self._video_stitch_partial.stat().st_size <= 0
        ):
            raise RuntimeError("diagnostic final video writer produced no bytes")
        _fsync_file(self._video_stitch_partial)
        os.replace(self._video_stitch_partial, self._video_final)
        _fsync_directory(self.session_dir)
        if self._bounded_concat:
            decoded = int(self._video_mode_probe(self._video_final)["frame_count"])
        else:
            decoded = int(self._video_validator(self._video_final))
        if decoded != self._n_rgb:
            raise RuntimeError(
                f"finalized video decoded {decoded} frames but recorded {self._n_rgb}"
            )
        self._validate_indexes()
        video_timebase = self._validate_video_timebase()
        audio_summary = self._validate_audio_artifacts()
        robot_summary = self._validate_robot_artifacts()
        force_summary = self._validate_force_artifacts()
        quality_summary = self._validate_quality_artifacts()
        audio_authenticity = (
            None if self._audio_authenticity is None
            else json.loads(json.dumps(self._audio_authenticity))
        )
        force_authenticity = (
            None if self._force_authenticity is None
            else json.loads(json.dumps(self._force_authenticity))
        )
        quality_authenticity = (
            None if self._quality_authenticity is None
            else json.loads(json.dumps(self._quality_authenticity))
        )
        first = self._artifact_inventory()
        second = self._artifact_inventory()
        depth_integrity_second = self._validate_depth_integrity()
        audio_summary_second = self._validate_audio_artifacts()
        robot_summary_second = self._validate_robot_artifacts()
        force_summary_second = self._validate_force_artifacts()
        quality_summary_second = self._validate_quality_artifacts()
        if (
            first != second
            or depth_integrity_second != depth_integrity
            or audio_summary_second != audio_summary
            or robot_summary_second != robot_summary
            or force_summary_second != force_summary
            or quality_summary_second != quality_summary
            or self._audio_authenticity != audio_authenticity
            or self._force_authenticity != force_authenticity
            or self._quality_authenticity != quality_authenticity
        ):
            raise RuntimeError("diagnostic artifacts changed while finalization hashed them")
        if (self.session_dir / "FINALIZATION.json").exists():
            raise RuntimeError("production FINALIZATION.json is forbidden in diagnostic capture")

        timeline = self._timeline_summary()
        frame_index_range = {
            "first_frame_index": self._first_frame_index,
            "last_frame_index": self._last_frame_index,
        }
        self._manifest.update(
            status="diagnostic_complete",
            training_eligible=False,
            production_finalization_permitted=False,
            ended_host_t=self._clock(),
            ended_utc_ns=self._wall_time_ns(),
            n_rgb_frames=self._n_rgb,
            n_depth_frames=self._n_depth,
            decoded_rgb_frames=decoded,
            timeline=timeline,
            video_timebase=video_timebase,
            video_stitch=stitch_summary,
            segment_validation=segment_validation,
            depth_integrity=depth_integrity,
            storage=self.storage_health(),
            capture_boundary=self._finalization_context,
            frame_index_range=frame_index_range,
            artifacts=first,
            audio=audio_summary,
            audio_authenticity=audio_authenticity,
            robot=robot_summary,
            force=force_summary,
            force_authenticity=force_authenticity,
            quality=quality_summary,
            quality_authenticity=quality_authenticity,
            abort_reason=None,
        )
        self._write_manifest()
        manifest_raw = (self.session_dir / "manifest.json").read_bytes()
        seal = {
            "schema_version": DIAGNOSTIC_FINALIZATION_SCHEMA_VERSION,
            "capture_mode": CAPTURE_MODE,
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "artifacts": first,
            "sealed_utc_ns": self._wall_time_ns(),
            "capture_boundary": self._finalization_context,
            "frame_index_range": frame_index_range,
            "video_timebase": video_timebase,
            "video_stitch": stitch_summary,
            "segment_validation": segment_validation,
            "depth_integrity": depth_integrity,
            "storage": self.storage_health(),
            "provenance": self.provenance,
            "audio": audio_summary,
            "audio_authenticity": audio_authenticity,
            "robot": robot_summary,
            "force": force_summary,
            "force_authenticity": force_authenticity,
            "quality": quality_summary,
            "quality_authenticity": quality_authenticity,
        }
        _atomic_json(self.session_dir / "DIAGNOSTIC_FINALIZATION.json", seal)
        self._state = "complete"
        return {
            "status": "diagnostic_complete",
            "training_eligible": False,
            "session_dir": str(self.session_dir),
            "n_rgb_frames": self._n_rgb,
            "decoded_rgb_frames": decoded,
            "n_depth_frames": self._n_depth,
            "timeline": timeline,
            "video_timebase": video_timebase,
            "video_stitch": stitch_summary,
            "segment_validation": segment_validation,
            "depth_integrity": depth_integrity,
            "storage": self.storage_health(),
            "audio": audio_summary,
            "audio_authenticity": audio_authenticity,
            "robot": robot_summary,
            "force": force_summary,
            "force_authenticity": force_authenticity,
            "quality": quality_summary,
            "quality_authenticity": quality_authenticity,
            "capture_boundary": self._finalization_context,
            "frame_index_range": frame_index_range,
            "diagnostic_seal": {
                "schema_version": DIAGNOSTIC_FINALIZATION_SCHEMA_VERSION,
                "path": "DIAGNOSTIC_FINALIZATION.json",
                "manifest_sha256": seal["manifest_sha256"],
                "artifact_count": len(first),
            },
        }

    def _validate_finalization_boundary_indexes(self) -> None:
        context = self._finalization_context
        if context is None:
            return
        try:
            submitted = int(context["last_submitted_camera_index"])
            completed = int(context["last_completed_camera_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "capture boundary is missing its final submitted/completed index"
            ) from exc
        if (
            self._last_frame_index is None
            or submitted != self._last_frame_index
            or completed != self._last_frame_index
        ):
            raise RuntimeError(
                "capture boundary does not close on the last written frame: "
                f"submitted={submitted}, completed={completed}, "
                f"written={self._last_frame_index}"
            )
        start_raw = context.get("camera_start_boundary_index")
        if start_raw is not None and self._first_frame_index != int(start_raw) + 1:
            raise RuntimeError(
                "capture boundary does not begin on the first written frame"
            )

    def abort(self, reason: str) -> None:
        if self._state == "aborted":
            return
        if self._state == "complete":
            raise RuntimeError("cannot abort a completed diagnostic capture")
        self._abort_internal(str(reason))

    def _abort_internal(self, reason: str) -> None:
        if self._writer is not None and self._video_segment_frame_count > 0:
            try:
                self._seal_video_segment()
                self._checkpoint()
            except Exception:
                # Preserve the original partial bytes even when they cannot be
                # proven decodable; recovery never deletes an in-progress file.
                pass
        if self._segment_validator is not None:
            self._segment_validator.abort()
            self._segment_validator = None
        self._close_partial_resources()
        robot_quarantine = self._quarantine_robot_artifacts()
        force_quarantine = self._quarantine_force_artifacts()
        quality_quarantine = self._quarantine_quality_artifacts()
        self._state = "aborted"
        if self._manifest:
            self._manifest.update(
                status="diagnostic_aborted",
                training_eligible=False,
                production_finalization_permitted=False,
                abort_reason=str(reason),
                ended_host_t=self._clock(),
                ended_utc_ns=self._wall_time_ns(),
                n_rgb_frames=self._n_rgb,
                n_depth_frames=self._n_depth,
                robot_abort_quarantine=robot_quarantine,
                force_abort_quarantine=force_quarantine,
                quality_abort_quarantine=quality_quarantine,
            )
            self._write_manifest()

    def _quarantine_force_artifacts(self) -> dict[str, object]:
        """Remove the Hall completion marker before preserving aborted evidence."""

        moved: dict[str, str] = {}
        errors: list[str] = []
        ordered = (
            "FORCE_CAPTURE_COMPLETE.json",
            "force_capture.json",
            "force.csv",
            "force_calibration.json",
        )
        for name in ordered:
            source = self.session_dir / name
            if not source.exists() and not source.is_symlink():
                continue
            destination = self.session_dir / f"{name}.aborted"
            if destination.exists() or destination.is_symlink():
                destination = self.session_dir / f"{name}.aborted.{uuid.uuid4().hex}"
            try:
                os.replace(source, destination)
                moved[name] = destination.name
                _fsync_directory(self.session_dir)
            except OSError as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        marker = self.session_dir / "FORCE_CAPTURE_COMPLETE.json"
        return {
            "schema_version": "wam_force_abort_quarantine_v1",
            "commit_marker_removed": not marker.exists() and not marker.is_symlink(),
            "moved": moved,
            "errors": errors,
            "partials_preserved": True,
        }

    def _quarantine_quality_artifacts(self) -> dict[str, object]:
        """Remove Q final names while preserving failed evidence for audit."""

        moved: dict[str, str] = {}
        errors: list[str] = []
        ordered = (
            "quality_inference_authenticity.json",
            "quality_inference_packets.jsonl",
            QUALITY_MODEL_INPUT_DIRNAME,
            ".quality_inference_packets.jsonl.inprogress",
            QUALITY_MODEL_INPUT_STAGING_DIRNAME,
        )
        for name in ordered:
            source = self.session_dir / name
            if not source.exists() and not source.is_symlink():
                continue
            destination = self.session_dir / f"{name}.aborted"
            if destination.exists() or destination.is_symlink():
                destination = self.session_dir / f"{name}.aborted.{uuid.uuid4().hex}"
            try:
                os.replace(source, destination)
                moved[name] = destination.name
                _fsync_directory(self.session_dir)
            except OSError as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        marker = self.session_dir / "quality_inference_authenticity.json"
        return {
            "schema_version": "wam_quality_abort_quarantine_v1",
            "commit_marker_removed": not marker.exists() and not marker.is_symlink(),
            "moved": moved,
            "errors": errors,
            "partials_preserved": True,
        }

    def _quarantine_robot_artifacts(self) -> dict[str, object]:
        """Remove finalized robot names without destroying aborted evidence.

        ``robot_capture_summary.json`` is the robot sidecar commit marker.  It
        must disappear first, then every other finalized name is moved out of
        the admissible namespace.  Existing ``*.partial`` evidence is retained
        untouched for forensic recovery.
        """

        moved: dict[str, str] = {}
        errors: list[str] = []
        ordered = (
            "robot_capture_summary.json",
            "robot_control_envelopes.jsonl",
            "robot_exact_control.csv",
            "robot_frame_index.jsonl",
        )
        for name in ordered:
            source = self.session_dir / name
            if not source.exists() and not source.is_symlink():
                continue
            destination = self.session_dir / f"{name}.aborted"
            if destination.exists() or destination.is_symlink():
                destination = self.session_dir / f"{name}.aborted.{uuid.uuid4().hex}"
            try:
                os.replace(source, destination)
                moved[name] = destination.name
                _fsync_directory(self.session_dir)
            except OSError as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        commit_marker = self.session_dir / "robot_capture_summary.json"
        return {
            "schema_version": "wam_robot_abort_quarantine_v1",
            "commit_marker_removed": not commit_marker.exists()
            and not commit_marker.is_symlink(),
            "moved": moved,
            "errors": errors,
            "partials_preserved": True,
        }

    def _checkpoint(self) -> None:
        # A checkpoint describes bytes that have actually reached storage.  The
        # active MP4 remains bounded and explicitly at risk until its segment is
        # closed; all previously sealed segments have trailers, hashes and an
        # fsynced journal record.
        for fp in (
            self._rgb_fp,
            self._tracker_fp,
            self._inference_fp,
            self._events_fp,
            self._depth_fp,
            self._depth_integrity_fp,
        ):
            if fp is not None:
                self._flush_file(fp)
        if self._active_video_partial is not None and self._active_video_partial.is_file():
            _fsync_file(self._active_video_partial)
        reserve = self._check_final_video_reserve()
        free = reserve["free_bytes"]
        _atomic_json(self.session_dir / "capture_checkpoint.json", {
            "schema_version": "wam_diagnostic_checkpoint_v1",
            "capture_mode": CAPTURE_MODE,
            "status": self._state,
            "n_rgb_frames": self._n_rgb,
            "n_depth_frames": self._n_depth,
            "last_frame_index": self._last_frame_index,
            "free_bytes": free,
            "final_rgb_mp4_reserve": reserve,
            "timeline": self._timeline_summary(),
            "storage": self.storage_health(),
            "indexes_and_depth_durable": True,
            "video_container_recoverable": self._sealed_video_frames > 0,
            "video_segments_durable": True,
            "sealed_video_segments_validated": self._sealed_video_segments_validated,
            "video_segment_frames_limit": self.video_segment_frames,
            "sealed_video_frames": self._sealed_video_frames,
            "at_risk_video_frames": self._video_segment_frame_count,
            "video_size": (
                None if self._video_size is None else list(self._video_size)
            ),
            "updated_host_t": self._clock(),
        })

    def _check_final_video_reserve(self) -> dict:
        """Keep enough free space to create rgb.mp4 beside retained segments.

        The final OpenCV pass decodes and re-encodes all sealed segments.  Its
        exact variable-bit-rate size is not known in advance, so retain twice
        the bytes already produced by the same codec plus a fixed container
        margin.  The active segment may still be buffered inside VideoWriter;
        cover it by its uncompressed frame bytes when that is larger than the
        visible partial file.  Crossing this budget aborts during recording,
        rather than surprising the operator with ENOSPC only after End.
        """

        active_file_bytes = 0
        if self._active_video_partial is not None and self._active_video_partial.is_file():
            active_file_bytes = int(self._active_video_partial.stat().st_size)
        active_raw_bytes = 0
        if self._video_size is not None and self._video_segment_frame_count:
            width, height = self._video_size
            active_raw_bytes = (
                int(self._video_segment_frame_count) * int(width) * int(height) * 3
            )
        has_video = bool(
            self._sealed_video_frames or self._video_segment_frame_count
        )
        active_reserve = max(
            _FINAL_VIDEO_RESERVE_MULTIPLIER * active_file_bytes,
            active_raw_bytes,
        )
        final_copy_reserve = (
            _FINAL_VIDEO_RESERVE_MULTIPLIER * int(self._sealed_video_bytes)
            + int(active_reserve)
            + (_FINAL_VIDEO_RESERVE_OVERHEAD_BYTES if has_video else 0)
        )
        free = int(self._disk_usage(self.session_dir).free)
        required = self.min_free_bytes + final_copy_reserve
        if free < required:
            raise RuntimeError(
                "insufficient final rgb.mp4 copy reserve: "
                f"free {free} < safety watermark {self.min_free_bytes} + "
                f"final-copy reserve {final_copy_reserve} bytes"
            )
        return {
            "policy": "2x_sealed_codec_bytes_plus_active_raw_and_64MiB",
            "free_bytes": free,
            "safety_watermark_bytes": self.min_free_bytes,
            "sealed_segment_bytes": int(self._sealed_video_bytes),
            "active_segment_file_bytes": active_file_bytes,
            "active_segment_raw_bytes": active_raw_bytes,
            "final_copy_reserve_bytes": final_copy_reserve,
            "required_free_bytes": required,
        }

    def _finish_segment_validation(self) -> dict[str, object]:
        if self._segment_validator is None:
            raise RuntimeError("video segment validation worker is unavailable")
        results = self._segment_validator.finish(timeout_s=120.0)
        self._segment_validator = None
        if len(results) != self._video_segment_index:
            raise RuntimeError(
                f"video segment validation count {len(results)} != "
                f"{self._video_segment_index}"
            )
        raw = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            for row in results
        )
        _atomic_bytes(self._segment_validation_path, raw)
        self._segment_validation_results = results
        return {
            "schema_version": "wam_video_segment_validation_index_v1",
            "path": self._segment_validation_path.name,
            "count": len(results),
            "first_segment_index": None if not results else 0,
            "last_segment_index": None if not results else len(results) - 1,
            "frame_count": sum(int(row["frame_count"]) for row in results),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "validation_worker_bounded_capacity": 2,
        }

    def _validate_depth_integrity(self) -> dict[str, object]:
        path = self.session_dir / "depth_integrity.jsonl"
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("depth integrity index is missing")
        count = 0
        first_index = last_index = None
        total_bytes = 0
        chain = bytes(32)
        with path.open("r", encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, start=1):
                try:
                    row = json.loads(line)
                    index = int(row["frame_index"])
                    rel = str(row["path"])
                    size = int(row["bytes"])
                    digest = str(row["sha256"])
                    identity = tuple(int(value) for value in row["identity"])
                    shape = tuple(int(value) for value in row["shape"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"depth integrity index is invalid at line {line_number}"
                    ) from exc
                if (
                    row.get("schema_version") != "wam_depth_frame_integrity_v1"
                    or row.get("dtype") != "uint16"
                    or len(identity) != 5
                    or len(digest) != 64
                    or any(ch not in "0123456789abcdef" for ch in digest)
                    or rel != f"depth/depth_{index:06d}.npy"
                    or (last_index is not None and index != last_index + 1)
                ):
                    raise RuntimeError(
                        f"depth integrity contract failed at line {line_number}"
                    )
                depth_path = self.session_dir / rel
                info = depth_path.lstat()
                current_identity = (
                    int(info.st_dev), int(info.st_ino), int(info.st_size),
                    int(info.st_mtime_ns), int(info.st_ctime_ns),
                )
                if (
                    depth_path.is_symlink()
                    or not stat.S_ISREG(info.st_mode)
                    or current_identity != identity
                    or size != int(info.st_size)
                ):
                    raise RuntimeError(f"depth frame identity changed: {rel}")
                if self._video_size is not None:
                    width, height = self._video_size
                    if shape != (height, width):
                        raise RuntimeError(f"depth frame shape proof is invalid: {rel}")
                chain = hashlib.sha256(chain + bytes.fromhex(digest)).digest()
                if first_index is None:
                    first_index = index
                last_index = index
                total_bytes += size
                count += 1
        if count != self._n_depth or count != self._depth_integrity_count:
            raise RuntimeError(
                f"depth integrity count {count} != recorded {self._n_depth}"
            )
        return {
            "schema_version": "wam_depth_integrity_index_v1",
            "path": path.name,
            "count": count,
            "first_frame_index": first_index,
            "last_frame_index": last_index,
            "total_depth_bytes": total_bytes,
            "index_sha256": _sha256(path),
            "content_hash_chain_sha256": chain.hex(),
            "dtype": "uint16",
            "one_file_per_camera_frame": True,
            "identity_revalidated_without_content_reread": True,
        }

    def storage_health(self) -> dict[str, object]:
        """Expose bounded raw-depth growth and a conservative remaining runtime."""

        active_video_bytes = 0
        if self._active_video_partial is not None and self._active_video_partial.is_file():
            active_video_bytes = int(self._active_video_partial.stat().st_size)
        active_raw_bytes = 0
        if self._video_size is not None and self._video_segment_frame_count:
            width, height = self._video_size
            active_raw_bytes = (
                int(self._video_segment_frame_count) * int(width) * int(height) * 3
            )
        has_video = bool(self._sealed_video_frames or self._video_segment_frame_count)
        final_copy_reserve = (
            _FINAL_VIDEO_RESERVE_MULTIPLIER * int(self._sealed_video_bytes)
            + max(
                _FINAL_VIDEO_RESERVE_MULTIPLIER * active_video_bytes,
                active_raw_bytes,
            )
            + (_FINAL_VIDEO_RESERVE_OVERHEAD_BYTES if has_video else 0)
        )
        free = int(self._disk_usage(self.session_dir).free)
        usable = max(0, free - self.min_free_bytes - final_copy_reserve)
        duration = (
            0.0 if self._first_host_t is None or self._last_host_t is None
            else max(0.0, self._last_host_t - self._first_host_t)
        )
        effective_duration = max(duration, self._n_rgb / self.fps if self._n_rgb else 0.0)
        persisted_bytes = (
            int(self._depth_bytes) + int(self._sealed_video_bytes) + active_video_bytes
        )
        measured_rate = (
            0.0 if effective_duration <= 0 else persisted_bytes / effective_duration
        )
        mode = self.negotiated_mode or {}
        try:
            width = int(mode.get("width", self._video_size[0] if self._video_size else 0))
            height = int(mode.get("height", self._video_size[1] if self._video_size else 0))
        except (TypeError, ValueError):
            width = height = 0
        expected_depth_rate = (
            float(width * height * np.dtype(np.uint16).itemsize) * self.fps
            if width > 0 and height > 0 and self.require_depth else 0.0
        )
        external_consumption_rate = 0.0
        if self._start_free_bytes is not None and effective_duration > 0:
            external_consumption_rate = max(
                0.0, (self._start_free_bytes - free) / effective_duration,
            )
        conservative_rate = max(
            expected_depth_rate, measured_rate, external_consumption_rate,
        )
        remaining_s = (
            None if conservative_rate <= 0 else usable / conservative_rate
        )
        return {
            "status": "OK" if usable > 0 else "RESERVE_EXHAUSTED",
            "depth_format": "formal_per_frame_npy_v1",
            "free_bytes": free,
            "safety_watermark_bytes": self.min_free_bytes,
            "final_rgb_copy_reserve_bytes": int(final_copy_reserve),
            "usable_bytes_before_safety_reserve": int(usable),
            "persisted_depth_bytes": int(self._depth_bytes),
            "persisted_capture_bytes_estimate": int(persisted_bytes),
            "expected_raw_depth_bytes_per_second": expected_depth_rate,
            "measured_capture_bytes_per_second": measured_rate,
            "conservative_bytes_per_second": conservative_rate,
            "estimated_remaining_seconds": remaining_s,
        }

    def _sealed_segment_paths(self) -> list[Path]:
        records = _read_segment_journal(self._segment_journal_path)
        validations = {
            int(row["segment_index"]): row
            for row in self._segment_validation_results
        }
        paths: list[Path] = []
        frame_count = 0
        for expected_index, record in enumerate(records):
            if int(record.get("segment_index", -1)) != expected_index:
                raise RuntimeError("video segment journal indexes are not contiguous")
            rel = str(record.get("path", ""))
            expected_rel = f"rgb_segments/segment_{expected_index:06d}.mp4"
            if rel != expected_rel:
                raise RuntimeError("video segment journal path is invalid")
            path = self.session_dir / rel
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"sealed diagnostic video segment disappeared: {path}")
            if (
                path.stat().st_size != int(record.get("bytes", -1))
                or _sha256(path) != str(record.get("sha256", ""))
            ):
                raise RuntimeError(f"sealed diagnostic video segment changed: {path}")
            count = int(record.get("frame_count", -1))
            if count <= 0:
                raise RuntimeError("video segment journal has an invalid frame count")
            validation = validations.get(expected_index)
            if not isinstance(validation, dict):
                raise RuntimeError("video segment has no asynchronous validation proof")
            decoded = int(validation.get("decoded_frame_count", -1))
            if decoded != count:
                raise RuntimeError(
                    f"video segment durable decode proof {decoded} != {count}: {path}"
                )
            if (
                validation.get("path") != rel
                or validation.get("sha256") != record.get("sha256")
                or int(validation.get("bytes", -1)) != int(record.get("bytes", -2))
            ):
                raise RuntimeError("video segment validation proof disagrees with journal")
            frame_count += count
            paths.append(path)
        if frame_count != self._sealed_video_frames:
            raise RuntimeError(
                f"video segment journal has {frame_count} frames, expected "
                f"{self._sealed_video_frames}"
            )
        self._sealed_video_segments_validated = True
        return paths

    def _validate_video_timebase(self) -> dict[str, object]:
        """Fail closed if MP4 playback speed/shape disagrees with camera time."""

        timeline = self._timeline_summary()
        expected = self.negotiated_mode
        if not isinstance(expected, dict):
            return {
                "verified": False,
                "reason": "negotiated_camera_mode_unavailable",
                "requested_fps": self.fps,
                "encoded_fps": None,
                "encoded_width": None,
                "encoded_height": None,
                "observed_timeline_fps": timeline["observed_fps"],
            }
        try:
            expected_fps = float(expected["fps"])
            expected_width = int(expected["width"])
            expected_height = int(expected["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("negotiated camera mode lacks valid FPS/shape") from exc
        raw_probe = self._video_mode_probe(self._video_final)
        if not isinstance(raw_probe, Mapping):
            raise RuntimeError("final video mode probe did not return a mapping")
        try:
            encoded_fps = float(raw_probe["fps"])
            encoded_width = int(raw_probe["width"])
            encoded_height = int(raw_probe["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("final video mode probe is incomplete") from exc
        fps_atol = max(0.01, expected_fps * 0.001)
        if (
            not np.isfinite(encoded_fps)
            or not np.isclose(encoded_fps, expected_fps, rtol=0.0, atol=fps_atol)
        ):
            raise RuntimeError(
                f"final video encoded FPS {encoded_fps} != negotiated {expected_fps}"
            )
        if (encoded_width, encoded_height) != (expected_width, expected_height):
            raise RuntimeError(
                "final video encoded shape "
                f"{encoded_width}x{encoded_height} != negotiated "
                f"{expected_width}x{expected_height}"
            )
        observed = float(timeline["observed_fps"])
        observed_tolerance = max(0.5, expected_fps * 0.05)
        if self._n_rgb < 2 or abs(observed - expected_fps) > observed_tolerance:
            raise RuntimeError(
                f"RGB timeline observed FPS {observed:.6f} is not within "
                f"{observed_tolerance:.6f} of negotiated {expected_fps:.6f}"
            )
        max_gap_tolerance = 3.0 / expected_fps
        if float(timeline["max_host_gap_s"]) > max_gap_tolerance:
            raise RuntimeError(
                "RGB timeline maximum host gap "
                f"{timeline['max_host_gap_s']:.6f}s exceeds "
                f"{max_gap_tolerance:.6f}s"
            )
        return {
            "verified": True,
            "negotiated_fps": expected_fps,
            "encoded_fps": encoded_fps,
            "fps_absolute_tolerance": fps_atol,
            "negotiated_width": expected_width,
            "negotiated_height": expected_height,
            "encoded_width": encoded_width,
            "encoded_height": encoded_height,
            "observed_timeline_fps": observed,
            "observed_fps_tolerance": observed_tolerance,
            "max_host_gap_s": timeline["max_host_gap_s"],
            "max_host_gap_tolerance_s": max_gap_tolerance,
            "camera_indexes_contiguous": timeline["camera_index_coverage_complete"],
        }

    def _timeline_summary(self) -> dict:
        duration = (
            0.0 if self._first_host_t is None or self._last_host_t is None
            else max(0.0, self._last_host_t - self._first_host_t)
        )
        observed_fps = (
            0.0 if self._n_rgb <= 1 or duration <= 0
            else (self._n_rgb - 1) / duration
        )
        result = {
            "first_host_t": self._first_host_t,
            "last_host_t": self._last_host_t,
            "duration_s": duration,
            "observed_fps": observed_fps,
            "max_host_gap_s": self._max_host_gap_s,
            "missing_camera_indexes": self._missing_camera_indexes,
            "camera_index_coverage_complete": self._missing_camera_indexes == 0,
            "model_gap_evidence": {
                "same_frame_count": int(
                    self._tracking_association_counts["same_camera_frame"]
                ),
                "carried_forward_count": int(
                    self._tracking_association_counts["carried_forward"]
                ),
                "unavailable_count": int(
                    self._tracking_association_counts["unavailable"]
                ),
                "max_tracking_age_s": float(self._max_tracking_age_s),
            },
        }
        if self.recording_boundary is not None:
            result["recording_boundary"] = self.recording_boundary
        if self._finalization_context is not None:
            result["end_boundary"] = self._finalization_context
        return result

    @staticmethod
    def _flush_file(fp) -> None:  # noqa: ANN001
        fp.flush()
        os.fsync(fp.fileno())

    def _close_and_publish_artifacts(self) -> None:
        if self._writer is not None:
            raise RuntimeError("cannot publish metadata while a video segment is active")

        publications = (
            (self._rgb_fp, self._rgb_partial, self.session_dir / "rgb_times.csv"),
            (self._tracker_fp, self._tracker_partial, self.session_dir / "tracker.jsonl"),
            (
                self._inference_fp,
                self._inference_partial,
                self.session_dir / "inference.jsonl",
            ),
            (self._events_fp, self._events_partial, self.session_dir / "events.jsonl"),
            (
                self._depth_fp,
                self._depth_partial,
                self._depth_dir / "depth_times.csv",
            ),
            (
                self._depth_integrity_fp,
                self._depth_integrity_partial,
                self.session_dir / "depth_integrity.jsonl",
            ),
        )
        for fp, partial, final in publications:
            if fp is not None:
                self._flush_file(fp)
                fp.close()
            os.replace(partial, final)
            _fsync_directory(final.parent)
        self._rgb_fp = self._tracker_fp = self._inference_fp = None
        self._events_fp = self._depth_fp = None
        self._depth_integrity_fp = None
        if self._segment_journal is not None:
            self._segment_journal.close()
            self._segment_journal = None

    def _close_partial_resources(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            finally:
                self._writer = None
        for name in (
            "_rgb_fp",
            "_tracker_fp",
            "_inference_fp",
            "_events_fp",
            "_depth_fp",
            "_depth_integrity_fp",
        ):
            fp = getattr(self, name)
            if fp is not None:
                try:
                    self._flush_file(fp)
                finally:
                    fp.close()
                    setattr(self, name, None)
        if self._segment_journal is not None:
            self._segment_journal.close()
            self._segment_journal = None

    def _validate_indexes(self) -> None:
        def csv_indexes(path: Path):
            with path.open(newline="", encoding="utf-8") as fp:
                for row_number, row in enumerate(csv.DictReader(fp), start=1):
                    try:
                        yield int(row["frame_index"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"{path.name} has an invalid frame index at row {row_number}"
                        ) from exc

        def json_rows(path: Path):
            with path.open("r", encoding="utf-8") as fp:
                for line_number, line in enumerate(fp, start=1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"{path.name} is corrupt at line {line_number}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise RuntimeError(
                            f"{path.name} line {line_number} is not an object"
                        )
                    yield row

        missing = object()
        rgb_count = 0
        first_rgb = None
        last_rgb = None
        rgb_path = self.session_dir / "rgb_times.csv"
        tracker_path = self.session_dir / "tracker.jsonl"
        for rgb_index, tracker_row in itertools.zip_longest(
            csv_indexes(rgb_path), json_rows(tracker_path), fillvalue=missing,
        ):
            if rgb_index is missing or tracker_row is missing:
                raise RuntimeError(
                    "RGB timestamp/tracker frame indexes are incomplete or misaligned"
                )
            try:
                tracker_index = int(tracker_row["frame_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("tracker frame index is invalid") from exc
            if tracker_index != rgb_index:
                raise RuntimeError(
                    "RGB timestamp/tracker frame indexes are incomplete or misaligned"
                )
            if last_rgb is not None and rgb_index != last_rgb + 1:
                raise RuntimeError("RGB frame indexes are not contiguous")
            if first_rgb is None:
                first_rgb = rgb_index
            last_rgb = rgb_index
            rgb_count += 1
        if rgb_count != self._n_rgb:
            raise RuntimeError(
                "RGB timestamp/tracker frame indexes are incomplete or misaligned"
            )

        depth_path = self._depth_dir / "depth_times.csv"
        if self.require_depth:
            for rgb_index, depth_index in itertools.zip_longest(
                csv_indexes(rgb_path), csv_indexes(depth_path), fillvalue=missing,
            ):
                if (
                    rgb_index is missing
                    or depth_index is missing
                    or rgb_index != depth_index
                ):
                    raise RuntimeError(
                        "RGB/depth frame indexes are incomplete or misaligned"
                    )

        previous_source = None
        for row in json_rows(self.session_dir / "inference.jsonl"):
            try:
                source = int(row["source_tracking_frame_index"])
                associated = int(row["first_associated_camera_frame_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("inference-to-camera frame association is invalid") from exc
            if previous_source is not None and source <= previous_source:
                raise RuntimeError(
                    "inference source frame indexes are duplicated or regressed"
                )
            if (
                first_rgb is None
                or last_rgb is None
                or not first_rgb <= associated <= last_rgb
                or source > associated
            ):
                raise RuntimeError("inference-to-camera frame association is invalid")
            previous_source = source

        def depth_file_indexes():
            for path in sorted((self.session_dir / "depth").glob("depth_*.npy")):
                try:
                    index = int(path.stem.removeprefix("depth_"))
                    depth = np.load(path, mmap_mode="r", allow_pickle=False)
                except (OSError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"depth frame is invalid: {path}") from exc
                if depth.dtype != np.uint16 or depth.ndim != 2 or not depth.size:
                    raise RuntimeError(f"depth frame has invalid array metadata: {path}")
                if self._video_size is not None:
                    width, height = self._video_size
                    if tuple(depth.shape) != (height, width):
                        raise RuntimeError(
                            f"depth frame shape {depth.shape} != {(height, width)}: {path}"
                        )
                yield index

        for depth_index, file_index in itertools.zip_longest(
            csv_indexes(depth_path), depth_file_indexes(), fillvalue=missing,
        ):
            if (
                depth_index is missing
                or file_index is missing
                or depth_index != file_index
            ):
                raise RuntimeError(
                    "depth NPY frame indexes do not match depth_times.csv"
                )
        with depth_path.open(newline="", encoding="utf-8") as fp:
            for row_number, row in enumerate(csv.DictReader(fp), start=1):
                try:
                    index = int(row["frame_index"])
                    rel = str(row["path"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"depth_times.csv path is invalid at row {row_number}"
                    ) from exc
                if rel != f"depth/depth_{index:06d}.npy":
                    raise RuntimeError(
                        f"depth_times.csv path is invalid at row {row_number}"
                    )

    def _artifact_inventory(self) -> dict:
        inventory = {}
        robot_receipt_artifacts = None
        if self._robot_summary is not None:
            self._validate_robot_artifact_receipt(self._robot_summary)
            robot_receipt_artifacts = self._robot_validation_receipt["artifacts"]
        paths = [
            self._video_final,
            self.session_dir / "rgb_times.csv",
            self.session_dir / "tracker.jsonl",
            self.session_dir / "inference.jsonl",
            self.session_dir / "events.jsonl",
            self._depth_dir / "depth_times.csv",
            self.session_dir / "depth_integrity.jsonl",
            self.session_dir / "capture_checkpoint.json",
            self._segment_journal_path,
            self._segment_validation_path,
        ]
        paths.extend(sorted(self._segment_dir.glob("segment_*.mp4")))
        paths.extend(sorted(self._depth_dir.glob("depth_*.npy")))
        if self._audio_summary is not None:
            paths.extend(
                self.session_dir / name
                for name in (
                    "audio.wav", "audio_capture.json",
                    "audio_input_times.jsonl", "audio_times.jsonl",
                )
            )
        if self._robot_summary is not None:
            paths.extend(
                self.session_dir / name for name in _ROBOT_FINAL_ARTIFACT_NAMES
            )
        if self._force_summary is not None:
            paths.extend(
                self.session_dir / name for name in _FORCE_FINAL_ARTIFACT_NAMES
            )
            paths.extend(
                path
                for name in _FORCE_OPTIONAL_ARTIFACT_NAMES
                if (path := self.session_dir / name).exists()
            )
        if self._quality_summary is not None:
            paths.extend(
                self.session_dir / name for name in _QUALITY_FINAL_ARTIFACT_NAMES
            )
            try:
                quality_inputs = self._quality_authenticity[
                    "model_input_inventory"
                ]["files"]
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    "verified quality receipt has no model-input inventory"
                ) from exc
            for claim in quality_inputs:
                if not isinstance(claim, Mapping):
                    raise RuntimeError(
                        "verified quality model-input inventory is invalid"
                    )
                rel = claim.get("path")
                if (
                    not isinstance(rel, str)
                    or not rel.startswith(f"{QUALITY_MODEL_INPUT_DIRNAME}/input_")
                    or Path(rel).as_posix() != rel
                ):
                    raise RuntimeError(
                        "verified quality model-input path is invalid"
                    )
                paths.append(self.session_dir / rel)
        for path in sorted(set(paths), key=lambda item: item.as_posix()):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"diagnostic artifact disappeared: {path}")
            rel = path.relative_to(self.session_dir).as_posix()
            if robot_receipt_artifacts is not None and rel in robot_receipt_artifacts:
                receipt = robot_receipt_artifacts[rel]
                inventory[rel] = {
                    "bytes": receipt["bytes"],
                    "sha256": receipt["sha256"],
                }
            else:
                inventory[rel] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        return inventory

    def _write_manifest(self) -> None:
        if self._manifest.get("training_eligible") is not False:
            raise RuntimeError("diagnostic manifest must remain training_eligible=false")
        if "promotion_contract_version" in self._manifest:
            raise RuntimeError("diagnostic manifest cannot declare a promotion contract")
        _atomic_json(self.session_dir / "manifest.json", self._manifest)
