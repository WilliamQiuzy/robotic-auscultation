#!/usr/bin/env python3
"""Concurrent, fail-closed supervisor for arm teleop + multimodal recording.

The recorder starts first and proves calibrated sensor readiness.  Only then is
teleop launched and the shared start barrier released.  This process owns every
signal/direct-disable action; the recorder only writes ``SAFETY_ABORT.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import wave
from bisect import bisect_right
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ausc.perception import depth_codec


REQUIRED_SESSION_ARTIFACTS = (
    "audio.wav", "rgb.mp4", "robot.csv", "force.csv", "rgb_times.csv", "obs.csv",
    "targets.jsonl",
)
ARTIFACT_HASH_SCHEMA_VERSION = "wam_session_artifact_hashes_v1"
PROMOTION_CONTRACT_VERSION = "wam_promotion_v1"
FINALIZATION_SCHEMA_VERSION = "wam_session_finalization_v1"
CAPTURE_RELEASE_PROVENANCE_SCHEMA_VERSION = "wam_capture_release_provenance_v1"
CODE_RELEASE_ID_PATTERN = re.compile(r"^code_release-[0-9a-f]{20}$")
RUNTIME_RELEASE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*-[0-9a-f]{20}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TRAINING_ELIGIBLE_STOP_OUTCOMES = frozenset({
    "normal_stop_direct_disable", "normal_stop_fallback_disable",
})


class SupervisorSignal(BaseException):
    """Convert SIGINT/SIGTERM into the supervisor's mandatory cleanup path."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"supervisor received signal {signum}")
        self.signum = int(signum)


def _send_process_group_signal(process, signum: int) -> None:  # noqa: ANN001
    """Signal an isolated child process group, falling back for test doubles."""
    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(os.getpgid(pid), signum)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    process.send_signal(signum)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash_inventory(session_dir: str | Path) -> dict:
    """Hash every promoted raw input, excluding the self-referential manifest."""
    session = Path(session_dir).resolve()
    paths = [session / name for name in REQUIRED_SESSION_ARTIFACTS]
    missing = [path.relative_to(session).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"session missing required artifacts: {missing}")
    depth_dir = session / "depth"
    depth_paths = sorted(path for path in depth_dir.rglob("*") if path.is_file())
    depth_names = {path.relative_to(session).as_posix() for path in depth_paths}
    if "depth/depth_times.csv" not in depth_names or not any(
        name.startswith("depth/depth_")
        and name.endswith((depth_codec.DEPTH_SUFFIX, depth_codec.LEGACY_DEPTH_SUFFIX))
        for name in depth_names
    ):
        raise ValueError("session is missing the depth timeline or depth frame artifacts")
    paths.extend(depth_paths)

    files: dict[str, dict] = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"session artifact must not be a symlink: {path}")
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(session).as_posix()
        except ValueError as exc:
            raise ValueError(f"session artifact escapes session directory: {path}") from exc
        files[relative] = {
            "sha256": _sha256(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    return {
        "schema_version": ARTIFACT_HASH_SCHEMA_VERSION,
        "algorithm": "sha256",
        "files": files,
    }


def _require_training_eligible_stop(outcome: str) -> None:
    if outcome not in TRAINING_ELIGIBLE_STOP_OUTCOMES:
        raise ValueError(f"teleop stop was not confirmed for normal completion: {outcome}")


def _require_observable_robot_feedback_liveness() -> None:
    """Keep the legacy CSV supervisor diagnostic-only.

    ``robot.csv`` contains numeric samples but no receive-side evidence from the
    OpenArm motors.  A frozen/all-zero column can therefore be indistinguishable
    from a valid stationary sample.  Numeric range or variance checks cannot
    close that observability gap and must never authorize training promotion.
    """

    raise ValueError(
        "legacy robot.csv cannot prove follower feedback liveness; production "
        "promotion requires per-motor decoded CAN RX counters and update "
        "timestamps"
    )


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: str | Path, raw: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("atomic metadata write made no progress")
            view = view[written:]
        os.fsync(fd)
    except BaseException:
        try:
            os.close(fd)
        finally:
            tmp.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)
    try:
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_json(path: str | Path, payload: dict) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def _publish_finalized_manifest(session_dir: str | Path, manifest: dict) -> None:
    """Seal a legacy diagnostic manifest as one crash-safe commit.

    Before the manifest replacement, the seal deliberately mismatches the old
    sensors-complete manifest, so conversion remains fail-closed.  This retired
    publisher explicitly rejects training eligibility; a future v4 promotion
    contract must use a separate authenticity-bound publisher.
    """
    if manifest.get("training_eligible") is not False:
        raise ValueError(
            "legacy finalization publisher cannot publish training eligibility"
        )
    session = Path(session_dir)
    manifest_bytes = _json_bytes(manifest)
    _atomic_json(session / "FINALIZATION.json", {
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "promotion_contract_version": PROMOTION_CONTRACT_VERSION,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    })
    _atomic_bytes(session / "manifest.json", manifest_bytes)


def validate_recorder_command(command: Sequence[str]) -> None:
    forbidden = {"--allow-uncalibrated-force", "--allow-missing-streams"}.intersection(command)
    if forbidden:
        raise ValueError(f"arm-connected supervisor forbids rehearsal flags: {sorted(forbidden)}")


def _run_disable(
    stop_cmd: Sequence[str], run: Callable[..., subprocess.CompletedProcess], *, timeout_s: float,
) -> None:
    if not stop_cmd:
        raise RuntimeError("direct motor-disable command is required")
    try:
        result = run(stop_cmd, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"direct motor-disable timed out after {timeout_s:.3f}s") from exc
    if result.returncode != 0:
        raise RuntimeError(f"direct motor-disable failed with exit {result.returncode}")


def _terminate_and_reap(process, *, timeout_s: float) -> str:  # noqa: ANN001
    """Escalate TERM -> KILL and do not return while a live child is unreaped."""
    if process is None or process.poll() is not None:
        return "already_stopped"
    if isinstance(getattr(process, "pid", None), int):
        _send_process_group_signal(process, signal.SIGTERM)
    else:  # preserve subprocess-like test doubles and compatibility wrappers
        process.terminate()
    try:
        process.wait(timeout=timeout_s)
        return "terminated"
    except subprocess.TimeoutExpired:
        if isinstance(getattr(process, "pid", None), int):
            try:
                _send_process_group_signal(process, signal.SIGKILL)
            except (ProcessLookupError, AttributeError):
                process.kill()
        else:
            process.kill()
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("teleop survived SIGINT, SIGTERM, and SIGKILL") from exc
        return "killed"


def stop_teleop(
    process,
    *,
    severity: str,
    stop_cmd: Sequence[str],
    timeout_s: float = 0.250,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    severity_probe: Callable[[], str | None] | None = None,
) -> str:  # noqa: ANN001
    """Stop teleop with severity-specific, independently confirmed energy isolation."""
    if severity not in {"hard", "screened", "normal"}:
        raise ValueError(f"unsupported stop severity: {severity!r}")

    def observed_safety_severity() -> str | None:
        if severity_probe is None:
            return None
        try:
            observed = severity_probe()
        except Exception:
            return "hard"
        if observed is None or observed in {"hard", "screened"}:
            return observed
        return "hard"

    def confirmed_normal_disable() -> str:
        # Process exit/SIGINT acknowledgement is not proof that the motor bus
        # reached disabled state.  The independently commissioned command is
        # mandatory on every terminal path.
        _run_disable(stop_cmd, run, timeout_s=timeout_s)
        return "normal_stop_direct_disable"

    if severity in {"hard", "screened"}:
        # A screened event has no independently verified retract/ack interface in
        # the collection teleop.  Start energy isolation immediately; never wait
        # for process exit and mistake it for hold or force-relief evidence.
        effective_severity = severity
        if severity == "screened":
            if observed_safety_severity() == "hard":
                effective_severity = "hard"
        try:
            was_running = process is not None and process.poll() is None
        except Exception:
            was_running = process is not None
        if was_running:
            try:
                _send_process_group_signal(process, signal.SIGINT)
            except ProcessLookupError:
                pass  # process raced to exit; independent direct disable is still required
            except Exception:
                pass  # signal transport failure must never skip independent direct disable
        disable_error: Exception | None = None
        try:
            _run_disable(stop_cmd, run, timeout_s=timeout_s)
        except Exception as exc:  # preserve failure, but first guarantee child shutdown
            disable_error = exc
        if severity == "screened":
            if observed_safety_severity() == "hard":
                effective_severity = "hard"
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _terminate_and_reap(process, timeout_s=timeout_s)
        if disable_error is not None:
            raise disable_error
        return ("hard_limit_emergency_disable" if effective_severity == "hard"
                else "screened_limit_direct_disable")

    if process is None or process.poll() is not None:
        return confirmed_normal_disable()

    try:
        _send_process_group_signal(process, signal.SIGINT)  # request graceful process shutdown first
    except ProcessLookupError:
        observed = observed_safety_severity()
        if observed in {"hard", "screened"}:
            return stop_teleop(
                process, severity=observed, stop_cmd=stop_cmd, timeout_s=timeout_s, run=run,
                severity_probe=severity_probe,
            )
        return confirmed_normal_disable()
    deadline = time.monotonic() + timeout_s
    while True:
        observed = observed_safety_severity()
        if observed in {"hard", "screened"}:
            return stop_teleop(
                process, severity=observed, stop_cmd=stop_cmd, timeout_s=timeout_s, run=run,
                severity_probe=severity_probe,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            process.wait(timeout=min(0.010, remaining))
        except subprocess.TimeoutExpired:
            continue
        observed = observed_safety_severity()
        if observed in {"hard", "screened"}:
            return stop_teleop(
                process, severity=observed, stop_cmd=stop_cmd, timeout_s=timeout_s, run=run,
                severity_probe=severity_probe,
            )
        return confirmed_normal_disable()
    disable_error: Exception | None = None
    try:
        _run_disable(stop_cmd, run, timeout_s=timeout_s)
    except Exception as exc:
        disable_error = exc
    _terminate_and_reap(process, timeout_s=timeout_s)
    if disable_error is not None:
        raise disable_error
    return "normal_stop_fallback_disable"


def validate_asof_coverage(robot_t_s, camera_t_s, *, max_age_s: float) -> None:  # noqa: ANN001
    robot = np.asarray(robot_t_s, dtype=np.float64)
    camera = np.asarray(camera_t_s, dtype=np.float64)
    if (robot.size == 0 or camera.size == 0 or not np.isfinite(robot).all()
            or not np.isfinite(camera).all() or np.any(np.diff(robot) <= 0)):
        raise ValueError("robot/camera timestamps must be finite, non-empty, and robot strictly increasing")
    values = robot.tolist()
    for t in camera:
        index = bisect_right(values, float(t)) - 1
        if index < 0:
            raise ValueError(f"no past robot sample for camera t={t:.6f}")
        age = float(t - robot[index])
        if age > max_age_s + 1e-12:
            raise ValueError(
                f"robot coverage failed at camera t={t:.6f}: age={age:.6f}s > {max_age_s:.6f}s"
            )


def validate_internal_gaps(values, *, max_gap_s: float, stream: str) -> None:  # noqa: ANN001
    times = np.asarray(values, dtype=np.float64)
    if (times.size < 2 or not np.isfinite(times).all() or max_gap_s <= 0
            or np.any(np.diff(times) <= 0)):
        raise ValueError(f"{stream} timestamps/gap contract is invalid")
    largest = float(np.max(np.diff(times)))
    if largest > max_gap_s + 1e-12:
        raise ValueError(f"{stream} internal gap {largest:.6f}s > {max_gap_s:.6f}s")


def _capture_release_provenance(manifest: dict) -> dict:
    provenance = manifest.get("release_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("release provenance is missing or is not an object")
    code = provenance.get("code")
    runtime = provenance.get("runtime")
    if (provenance.get("schema_version") != CAPTURE_RELEASE_PROVENANCE_SCHEMA_VERSION
            or not isinstance(code, dict) or not isinstance(runtime, dict)
            or CODE_RELEASE_ID_PATTERN.fullmatch(str(code.get("release_id", ""))) is None
            or RUNTIME_RELEASE_ID_PATTERN.fullmatch(
                str(runtime.get("release_id", ""))
            ) is None
            or SHA256_PATTERN.fullmatch(str(runtime.get("lock_sha256", ""))) is None):
        raise ValueError("release provenance does not bind versioned code and runtime locks")
    return provenance


def _capture_durability(manifest: dict) -> dict:
    """Prove both lossless FIFOs and their common atomic Start/End interval."""

    def fail(reason: str) -> None:
        raise ValueError(f"capture durability proof is invalid: {reason}")

    def mapping(value, label: str) -> dict:  # noqa: ANN001
        if not isinstance(value, dict):
            fail(f"{label} is missing")
        return value

    def integer(value, label: str, *, minimum: int = 0) -> int:  # noqa: ANN001
        if type(value) is not int or value < minimum:
            fail(f"{label} must be an integer >= {minimum}")
        return value

    def sha(value, label: str) -> str:  # noqa: ANN001
        if SHA256_PATTERN.fullmatch(str(value or "")) is None:
            fail(f"{label} is not a lowercase SHA-256")
        return str(value)

    def finite(value, label: str) -> float:  # noqa: ANN001
        try:
            result = float(value)
        except (TypeError, ValueError):
            fail(f"{label} is not finite")
        if not np.isfinite(result):
            fail(f"{label} is not finite")
        return result

    durability = mapping(manifest.get("capture_durability"), "capture_durability")
    if durability.get("schema_version") != "wam_streaming_capture_v1":
        fail("schema version is unsupported")
    if durability.get("crash_recovery_policy") != (
        "diagnostic_only_never_training_eligible"
    ):
        fail("crash recovery policy is not fail-closed")

    audio = mapping(durability.get("audio"), "audio proof")
    if (audio.get("streamed_to_disk") is not True
            or integer(audio.get("n_samples"), "audio samples", minimum=1) <= 0
            or integer(audio.get("sample_rate"), "audio sample rate", minimum=1) <= 0
            or audio.get("timeline") != "audio_times.jsonl"):
        fail("audio was not durably streamed")
    sha(audio.get("sha256"), "audio SHA-256")

    video = mapping(durability.get("video"), "video proof")
    frame_count = integer(video.get("frame_count"), "video frame count", minimum=1)
    if (video.get("path") != "rgb.mp4"
            or integer(video.get("decoded_frame_count"), "decoded frame count", minimum=1)
            != frame_count):
        fail("video frame/decode counts do not match")
    sha(video.get("sha256"), "video SHA-256")
    camera_meta = mapping(manifest.get("camera"), "camera manifest")
    if (integer(camera_meta.get("n_rgb_frames"), "manifest RGB count", minimum=1)
            != frame_count
            or integer(camera_meta.get("n_depth_frames"), "manifest depth count", minimum=1)
            != frame_count
            or camera_meta.get("has_depth") is not True):
        fail("manifest RGB-D counts do not match decoded video")

    fifo = mapping(durability.get("capture_fifo"), "formal writer FIFO")
    accepted = integer(fifo.get("accepted_frames"), "accepted frame count", minimum=1)
    if accepted != frame_count or integer(
        fifo.get("processed_frames"), "processed frame count", minimum=1,
    ) != accepted:
        fail("formal writer accepted/processed counts do not match video")
    for field in (
        "overflow_frames", "missing_camera_frames", "boundary_rejections", "queue_depth",
    ):
        if integer(fifo.get(field), f"formal writer {field}") != 0:
            fail(f"formal writer {field} is non-zero")
    if (fifo.get("sequence_policy") != "strict_contiguous_camera_index"
            or fifo.get("camera_backend") != "orbbec"
            or fifo.get("start_boundary_verified") is not True
            or fifo.get("drained") is not True
            or fifo.get("worker_error") is not None
            or fifo.get("fatal_reason") is not None):
        fail("formal writer did not close as a strict drained Orbbec FIFO")
    start_index = integer(
        fifo.get("start_boundary_frame_index"), "Start boundary frame index",
    )
    first_index = integer(
        fifo.get("first_accepted_frame_index"), "first accepted frame index", minimum=1,
    )
    if (integer(fifo.get("first_expected_frame_index"), "first expected frame index", minimum=1)
            != first_index or first_index != start_index + 1):
        fail("first formal frame is not the exact Start-boundary successor")
    last_index = integer(
        fifo.get("last_submitted_frame_index"), "last submitted frame index", minimum=1,
    )
    if (integer(fifo.get("last_processed_frame_index"), "last processed frame index", minimum=1)
            != last_index or last_index != first_index + frame_count - 1):
        fail("formal writer last-frame proof is not contiguous")

    delivery = mapping(durability.get("camera_delivery_fifo"), "camera delivery FIFO")
    interval_id = integer(delivery.get("interval_id"), "delivery interval id", minimum=1)
    capacity = integer(delivery.get("capacity"), "delivery capacity", minimum=1)
    if (delivery.get("status") != "OK"
            or delivery.get("interval_active") is not False
            or delivery.get("accepting") is not False
            or delivery.get("closing") is not False):
        fail("camera delivery FIFO did not close cleanly")
    for field in (
        "queue_depth", "overflow_frames", "interval_overflow_frames",
        "rejected_after_overflow",
    ):
        if integer(delivery.get(field), f"camera delivery {field}") != 0:
            fail(f"camera delivery {field} is non-zero")
    if (integer(delivery.get("interval_enqueued_frames"), "delivery enqueued frames")
            != frame_count
            or integer(delivery.get("interval_delivered_frames"), "delivery delivered frames")
            != frame_count
            or integer(delivery.get("first_enqueued_frame_index"), "first delivered frame", minimum=1)
            != first_index
            or integer(delivery.get("last_enqueued_frame_index"), "last enqueued frame", minimum=1)
            != last_index
            or integer(delivery.get("last_delivered_frame_index"), "last delivered frame", minimum=1)
            != last_index):
        fail("camera delivery FIFO count/index proof does not match formal writer")
    if integer(delivery.get("max_queue_depth"), "delivery maximum queue depth") > capacity:
        fail("camera delivery FIFO exceeded its fixed capacity")

    start = mapping(durability.get("camera_start_boundary"), "camera Start proof")
    if (start.get("boundary_latched_atomically") is not True
            or integer(start.get("interval_id"), "Start interval id", minimum=1) != interval_id
            or integer(start.get("start_boundary_frame_index"), "Start proof frame index")
            != start_index
            or integer(start.get("capacity"), "Start proof capacity", minimum=1) != capacity
            or integer(start.get("overflow_baseline"), "Start overflow baseline") != 0
            or type(start.get("inflight_pre_boundary_seen")) is not bool
            or start.get("inflight_pre_boundary_completed") is not True
            or type(start.get("inflight_pre_boundary_produced_frame")) is not bool):
        fail("camera Start proof does not bind the empty lossless interval")
    if abs(
        finite(start.get("start_boundary_host_t"), "Start boundary host timestamp")
        - finite(fifo.get("start_boundary_frame_t"), "formal Start boundary timestamp")
    ) > 1e-6:
        fail("camera Start timestamp does not match formal writer boundary")

    end = mapping(durability.get("camera_end_boundary"), "camera End proof")
    if (end.get("boundary_latched_atomically") is not True
            or end.get("post_boundary_overwrite_prevented") is not True
            or integer(end.get("interval_id"), "End interval id", minimum=1) != interval_id
            or integer(end.get("delivery_capacity"), "End delivery capacity", minimum=1)
            != capacity
            or integer(end.get("delivery_overflow_frames"), "End delivery overflow") != 0
            or integer(end.get("delivery_enqueued_frames"), "End enqueued count")
            != frame_count
            or integer(end.get("delivery_first_frame_index"), "End first frame", minimum=1)
            != first_index
            or integer(end.get("delivery_last_frame_index"), "End last frame", minimum=1)
            != last_index):
        fail("camera End proof does not bind the same lossless interval")
    end_delivered = integer(end.get("delivery_delivered_frames"), "End delivered count")
    end_depth = integer(end.get("delivery_queue_depth"), "End delivery queue depth")
    if end_delivered > frame_count or end_delivered + end_depth != frame_count:
        fail("camera End delivery snapshot count is inconsistent")
    if (type(end.get("inflight_pre_boundary_seen")) is not bool
            or end.get("inflight_pre_boundary_completed") is not True
            or type(end.get("inflight_pre_boundary_produced_frame")) is not bool):
        fail("camera End in-flight proof is incomplete")
    if end.get("inflight_pre_boundary_seen") is True and (
        end.get("inflight_pre_boundary_completed") is not True
        or end.get("inflight_pre_boundary_produced_frame") is not True
    ):
        fail("camera End in-flight pre-boundary frame was not completed")
    end_t = finite(end.get("t_end_monotonic"), "End boundary timestamp")
    if abs(end_t - finite(fifo.get("t_end_monotonic"), "formal FIFO End timestamp")) > 1e-6:
        fail("camera End timestamp does not match formal writer End")
    snapshot_t = finite(end.get("snapshot_host_t"), "End snapshot timestamp")
    if snapshot_t > end_t + 1e-9:
        fail("camera End snapshot arrived after the sealed boundary")
    if (integer(end.get("snapshot_frame_index"), "End snapshot frame", minimum=1)
            != last_index
            or end.get("tail_frame_already_submitted") is not True
            or end.get("tail_frame_submitted") is not False
            or end.get("tail_frame_precedes_start") is not False
            or integer(end.get("fifo_last_submitted_frame_index"), "End FIFO submitted", minimum=1)
            != last_index
            or integer(end.get("fifo_last_processed_frame_index"), "End FIFO processed", minimum=1)
            != last_index):
        fail("camera End snapshot is not the exact drained tail")

    timelines = mapping(durability.get("timelines"), "timeline proof")
    expected_rows = {
        "rgb_times.csv": frame_count,
        "depth/depth_times.csv": frame_count,
        "obs.csv": integer(manifest.get("n_obs"), "manifest observation count", minimum=1),
    }
    for name, rows in expected_rows.items():
        timeline = mapping(timelines.get(name), f"{name} timeline")
        if integer(timeline.get("rows"), f"{name} rows", minimum=1) != rows:
            fail(f"{name} row count does not match manifest")
        sha(timeline.get("sha256"), f"{name} SHA-256")
    return durability


def _validate_capture_durability_artifact_hashes(
    durability: dict, artifact_hashes: dict,
) -> None:
    files = artifact_hashes.get("files") if isinstance(artifact_hashes, dict) else None
    if not isinstance(files, dict):
        raise ValueError("capture durability artifact hashes are unavailable")
    bindings = {
        "audio.wav": durability["audio"]["sha256"],
        "rgb.mp4": durability["video"]["sha256"],
        "rgb_times.csv": durability["timelines"]["rgb_times.csv"]["sha256"],
        "obs.csv": durability["timelines"]["obs.csv"]["sha256"],
        "depth/depth_times.csv": durability["timelines"][
            "depth/depth_times.csv"
        ]["sha256"],
    }
    for relative, expected in bindings.items():
        actual = files.get(relative)
        if not isinstance(actual, dict) or actual.get("sha256") != expected:
            raise ValueError(
                f"capture durability artifact SHA-256 mismatch for {relative}"
            )


def validate_promotion_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != "wam_session_v2" or manifest.get("status") != "sensors_complete":
        raise ValueError("recorder did not finish a wam_session_v2 sensors_complete session")
    if manifest.get("training_eligible") is not False:
        raise ValueError("sensor recorder cannot pre-claim training eligibility")
    if manifest.get("rehearsal_overrides"):
        raise ValueError("rehearsal overrides can never be promoted to training data")
    if not {"force", "audio", "cam"}.issubset(manifest.get("streams") or []):
        raise ValueError("required sensor streams are incomplete")
    if (manifest.get("force") or {}).get("calibrated") is not True:
        raise ValueError("force calibration is not calibrated=true")
    sync = manifest.get("sync") or {}
    if sync.get("applied") is not False or len(str(sync.get("calibration_sha256", ""))) != 64:
        raise ValueError("sync metadata is invalid or already applied")
    try:
        sync_std_ms = float(sync["offset_std_ms"])
        sync_matches = int(sync["n_matched"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("sync calibration quality metadata is incomplete") from exc
    if (not np.isfinite(sync_std_ms) or not 0 <= sync_std_ms <= 5.0 or sync_matches < 4
            or sync.get("unique_matches") is not True
            or sync.get("measurement_method") != "physical_diaphragm_taps"):
        raise ValueError("sync calibration quality gate failed")
    camera = manifest.get("camera") or {}
    if (camera.get("backend") != "orbbec"
            or not isinstance(camera.get("camera_id"), str)
            or not camera["camera_id"].strip()
            or camera.get("camera_id_source") != "verified_orbbec_serial"):
        raise ValueError(
            "arm training promotion requires a serial-verifiable Orbbec camera identity"
        )
    camera_timing = camera.get("timing") or {}
    if (camera_timing.get("applied") is not False
            or len(str(camera_timing.get("calibration_sha256", ""))) != 64):
        raise ValueError("camera latency metadata is invalid or already applied")
    try:
        camera_latency_ms = float(camera_timing["latency_ms"])
        camera_std_ms = float(camera_timing["latency_std_ms"])
        camera_samples = int(camera_timing["n_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("camera latency quality metadata is incomplete") from exc
    if (not np.isfinite(camera_latency_ms) or not 0 < camera_latency_ms <= 500.0
            or not np.isfinite(camera_std_ms) or not 0 <= camera_std_ms <= 5.0
            or camera_samples < 20
            # hardware timing: LED flash or host-synced device timestamp (device_timer_host_sync)
            or camera_timing.get("measurement_method")
            not in ("hardware_led_flash", "device_timer_host_sync")):
        raise ValueError("camera latency quality gate failed")
    health = manifest.get("health") or {}
    unhealthy = [
        name for name in ("force", "audio", "camera")
        if (health.get(name) or {}).get("status") != "OK"
    ]
    if unhealthy:
        raise ValueError(f"required stream health is not OK: {unhealthy}")
    _capture_release_provenance(manifest)
    _capture_durability(manifest)


def _robot_metadata(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    if len(rows) < 2 or rows[0][0] not in ("t", "host_t_s"):
        raise ValueError("robot.csv must have timestamp plus q*/a* header groups")
    body = rows[0][1:]
    q_cols = [name for name in body if name.startswith("q")]
    a_cols = [name for name in body if name.startswith("a")]
    if not q_cols or len(q_cols) != len(a_cols) or body != q_cols + a_cols:
        raise ValueError("robot.csv q*/a* header groups do not match")
    suffix = lambda name: name[2:] if len(name) > 1 and name[1] == "_" else name[1:]
    if [suffix(name) for name in q_cols] != [suffix(name) for name in a_cols]:
        raise ValueError("robot.csv q*/a* joint identities differ")
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError("robot.csv has no data rows")
    expected_width = len(rows[0])
    if any(len(row) != expected_width for row in data_rows):
        raise ValueError("robot.csv contains a malformed row width")
    try:
        values = np.asarray([[float(value) for value in row] for row in data_rows], dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("robot.csv contains non-numeric data") from exc
    if not np.isfinite(values).all():
        raise ValueError("robot.csv contains non-finite data")
    times = values[:, 0]
    if np.any(np.diff(times) <= 0):
        raise ValueError("robot.csv timestamps are not finite and strictly increasing")
    names = [f"joint_{suffix(name)}" for name in q_cols]
    return times, names


def _camera_host_times(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp); rows = list(reader)
    if not rows:
        raise ValueError("rgb_times.csv has no rows")
    field = "host_arrival_t_s" if "host_arrival_t_s" in rows[0] else "t"
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def _validate_interval_coverage(
    values, *, t0: float, duration_s: float, max_edge_gap_s: float, stream: str,
) -> None:  # noqa: ANN001
    times = np.asarray(values, dtype=np.float64)
    end = float(t0) + float(duration_s)
    if (times.size == 0 or not np.isfinite(times).all() or not np.isfinite(end)
            or duration_s <= 0):
        raise ValueError(f"{stream} coverage inputs are invalid")
    if times[0] > float(t0) + max_edge_gap_s or times[-1] < end - max_edge_gap_s:
        raise ValueError(
            f"{stream} does not cover capture interval [{t0:.6f}, {end:.6f}] "
            f"within {max_edge_gap_s:.3f}s"
        )


def _validate_audio_artifact(path: Path, manifest: dict) -> dict:
    """Decode WAV metadata/payload and prove it covers the timed capture."""
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            payload = wav.readframes(n_frames)
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"audio.wav is not a decodable WAV: {exc}") from exc
    expected_rate = int((manifest.get("audio") or {}).get("sr", 0))
    if (channels != 1 or sample_width != 2 or sample_rate <= 0 or sample_rate != expected_rate
            or n_frames <= 0 or len(payload) != n_frames * channels * sample_width):
        raise ValueError("audio.wav format/payload does not match mono int16 manifest audio")
    audio_t0 = float((manifest.get("audio") or {}).get("t0_monotonic", np.nan))
    capture_t0 = float(manifest.get("t0_monotonic", np.nan))
    duration_s = float(manifest.get("duration_s", np.nan))
    audio_end = audio_t0 + n_frames / float(sample_rate)
    tolerance = 0.050
    if (not np.isfinite([audio_t0, capture_t0, duration_s, audio_end]).all()
            or audio_t0 > capture_t0 + tolerance
            or audio_end < capture_t0 + duration_s - tolerance):
        raise ValueError("audio.wav timestamps/duration do not cover the capture interval")
    return {"sample_rate": sample_rate, "n_frames": n_frames, "duration_s": n_frames / sample_rate}


def _validate_video_artifact(
    path: Path, *, expected_frames: int, expected_width: int | None = None,
    expected_height: int | None = None, expected_fps: float | None = None,
) -> dict:
    """Actually decode every RGB frame; container metadata alone is not sufficient."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    count = 0
    shape = None
    encoded_fps = float("nan")
    try:
        if not cap.isOpened():
            raise ValueError("rgb.mp4 is not a decodable video")
        encoded_fps = float(cap.get(cv2.CAP_PROP_FPS))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("rgb.mp4 contains an invalid decoded frame")
            current_shape = tuple(int(value) for value in frame.shape)
            if shape is None:
                shape = current_shape
            elif current_shape != shape:
                raise ValueError("rgb.mp4 frame shape changes within the recording")
            count += 1
    finally:
        cap.release()
    if expected_frames <= 0 or count != int(expected_frames):
        raise ValueError(f"rgb.mp4 decoded {count} frames but rgb_times.csv has {expected_frames}")
    if (expected_width is not None or expected_height is not None):
        expected_shape = (int(expected_height), int(expected_width), 3)
        if shape != expected_shape:
            raise ValueError(f"rgb.mp4 shape {shape!r} does not match capture mode {expected_shape!r}")
    if expected_fps is not None:
        expected_fps = float(expected_fps)
        if (not np.isfinite(encoded_fps)
                or abs(encoded_fps - expected_fps) > max(0.5, expected_fps * 0.02)):
            raise ValueError(
                f"rgb.mp4 encoded FPS {encoded_fps!r} does not match capture mode {expected_fps!r}"
            )
    return {"n_frames": count, "shape": shape, "encoded_fps": encoded_fps}


def _validate_depth_artifacts(session: Path, camera, camera_meta: dict) -> dict:  # noqa: ANN001
    """Verify one finite depth array and matching timestamp for every RGB-D frameset."""
    timing = camera_meta.get("timing") or {}
    mode = timing.get("capture_mode") or {}
    depth_mode = mode.get("depth_mode")
    if not isinstance(depth_mode, dict):
        raise ValueError("Orbbec capture_mode depth_mode is missing")
    if camera_meta.get("has_depth") is not True:
        raise ValueError("Orbbec session manifest does not confirm depth capture")
    expected_count = len(camera.frame_index)
    if int(camera_meta.get("n_depth_frames", -1)) != expected_count:
        raise ValueError("manifest depth frame count does not match RGB frames")
    depth_dir = session / "depth"
    index_path = depth_dir / "depth_times.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"depth timing index missing: {index_path}")
    expected_shape = (
        (int(mode["height"]), int(mode["width"]))
        if mode.get("align_mode") == "software_depth_to_color"
        else (int(depth_mode["height"]), int(depth_mode["width"]))
    )
    minimum_valid_fraction = 1.0
    previous_device_time_ms: float | None = None
    # Stream the timeline instead of list(reader): a multi-hour session has
    # hundreds of thousands of rows and must not be materialized in memory.
    with index_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames != [
            "frame_index", "host_arrival_t_s", "device_t_ms", "depth_scale", "path",
        ]:
            raise ValueError("depth_times.csv header is invalid")
        for frame_index, host_t in zip(camera.frame_index, camera.host_arrival_t_s):
            try:
                row = next(reader)
            except StopIteration:
                raise ValueError("depth_times.csv row count does not match RGB frames")
            try:
                index = int(row["frame_index"])
                timestamp = float(row["host_arrival_t_s"])
                device_time_ms = float(row["device_t_ms"])
                scale = float(row["depth_scale"])
                relative = str(row["path"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("depth_times.csv contains invalid values") from exc
            expected_relative = f"depth/depth_{index:06d}{depth_codec.DEPTH_SUFFIX}"
            if (index != int(frame_index) or not np.isfinite(timestamp)
                    or abs(timestamp - float(host_t)) > 0.001
                    or not np.isfinite(device_time_ms)
                    or relative != expected_relative
                    or not np.isfinite(scale) or scale <= 0):
                raise ValueError("depth timing/index/scale/path does not match RGB timeline")
            # Fail fast on a frozen/regressing device clock (matches the writer and
            # finalizer) rather than passing it through to a late finalize rejection.
            if previous_device_time_ms is not None and device_time_ms <= previous_device_time_ms:
                raise ValueError(
                    f"depth session device clock froze or regressed at frame {index}"
                )
            previous_device_time_ms = device_time_ms
            path = session / relative
            if not path.is_file():
                raise FileNotFoundError(f"depth frame missing: {path}")
            values = depth_codec.load_depth(path)
            if values.dtype != np.uint16:
                raise ValueError(f"depth frame {index} must be uint16")
            if values.shape != expected_shape or values.ndim != 2:
                raise ValueError(f"depth frame {index} has invalid shape")
            valid_pixels = int(np.count_nonzero(values))
            if valid_pixels == 0:
                raise ValueError(f"depth frame {index} has no valid depth pixels")
            minimum_valid_fraction = min(
                minimum_valid_fraction,
                valid_pixels / float(values.size),
            )
        try:
            next(reader)
        except StopIteration:
            pass
        else:
            raise ValueError("depth_times.csv row count does not match RGB frames")
    # Count frames without building a list of hundreds of thousands of paths.
    file_count = sum(1 for _ in depth_dir.glob(f"depth_*{depth_codec.DEPTH_SUFFIX}"))
    if file_count != expected_count:
        raise ValueError("depth directory contains an unexpected number of frames")
    return {
        "n_frames": expected_count,
        "shape": expected_shape,
        "dtype": "uint16",
        "minimum_nonzero_depth_fraction": minimum_valid_fraction,
    }


def _validate_obs_artifact(
    path: Path, *, t0: float, duration_s: float, control_hz: float,
    camera_frame_index=None, camera_host_t_s=None,
) -> dict:
    required = [
        "t", "force_n", "force_status", "audio_rms_db", "audio_peak_db", "cam_index", "cam_t",
    ]
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames != required:
            raise ValueError(f"obs.csv header must be {required}")
        rows = list(reader)
    if not rows:
        raise ValueError("obs.csv has no rows")
    try:
        numeric = np.asarray([
            [float(row[name]) for name in ("t", "force_n", "audio_rms_db", "audio_peak_db",
                                           "cam_index", "cam_t")]
            for row in rows
        ], dtype=np.float64)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("obs.csv contains non-numeric fields") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("obs.csv numeric values must be finite")
    times = numeric[:, 0]
    if np.any(np.diff(times) <= 0):
        raise ValueError("obs.csv timestamps must be strictly increasing")
    validate_internal_gaps(
        times, max_gap_s=max(0.050, 2.0 / float(control_hz)), stream="obs.csv",
    )
    if any(row["force_status"] not in ("OK", "WARN") for row in rows):
        raise ValueError("obs.csv contains an unsafe force status")
    cam_indices = numeric[:, 4]
    cam_times = numeric[:, 5]
    if (np.any(cam_indices < 0) or np.any(cam_indices != np.floor(cam_indices))
            or np.any(cam_times > times + 1e-9) or np.any(times - cam_times > 0.550)):
        raise ValueError("obs.csv camera references are invalid, future, or stale")
    if camera_frame_index is not None or camera_host_t_s is not None:
        indices = np.asarray(camera_frame_index, dtype=np.int64)
        host_times = np.asarray(camera_host_t_s, dtype=np.float64)
        if (indices.shape != host_times.shape or indices.size == 0
                or not np.isfinite(host_times).all() or len(set(indices.tolist())) != len(indices)):
            raise ValueError("camera reference timeline is invalid")
        by_index = {int(index): float(host_t) for index, host_t in zip(indices, host_times)}
        for index, cam_t in zip(cam_indices.astype(np.int64), cam_times):
            if int(index) not in by_index or abs(float(cam_t) - by_index[int(index)]) > 0.001:
                raise ValueError("obs.csv camera reference does not match rgb_times.csv")
    max_gap = max(0.050, 2.0 / float(control_hz))
    _validate_interval_coverage(
        times, t0=t0, duration_s=duration_s, max_edge_gap_s=max_gap, stream="obs.csv",
    )
    minimum_rows = max(1, int(np.floor(duration_s * control_hz * 0.8)))
    if len(rows) < minimum_rows:
        raise ValueError(f"obs.csv has {len(rows)} rows; expected at least {minimum_rows}")
    return {"n_rows": len(rows)}


def mark_aborted(session_dir: str | Path, reason: str, **extra) -> None:
    (Path(session_dir) / "FINALIZATION.json").unlink(missing_ok=True)
    path = Path(session_dir) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {"schema_version": "wam_session_v2"}
    manifest.update(status="aborted", training_eligible=False, abort_reason=reason, **extra)
    _atomic_json(path, manifest)


def finalize_session(session_dir: str | Path, *, stop_outcome: str) -> dict:
    session = Path(session_dir)
    if (session / "SAFETY_ABORT.json").is_file():
        raise ValueError("session has a safety sentinel and cannot be promoted")
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    interrupted_final_manifest = None
    if (
        manifest.get("status") == "complete"
        and manifest.get("training_eligible") is True
        and not (session / "FINALIZATION.json").exists()
    ):
        # Compatibility recovery for the former manifest-then-seal publisher.
        # Nothing is trusted merely because it claimed eligibility: normalize
        # the in-memory copy and rerun every calibration/artifact validator.
        if (manifest.get("promotion_contract_version") != PROMOTION_CONTRACT_VERSION
                or manifest.get("teleop_stop_outcome") != stop_outcome):
            raise ValueError("interrupted finalization contract/stop outcome mismatch")
        interrupted_final_manifest = json.loads(json.dumps(manifest))
        manifest = json.loads(json.dumps(manifest))
        manifest["status"] = "sensors_complete"
        manifest["training_eligible"] = False
    validate_promotion_manifest(manifest)
    from session_to_lerobot_v2 import (
        fit_device_clock_checked, read_camera_timeline, read_force, read_robot, read_targets,
        verify_force_calibration_metadata, verify_sync_metadata,
    )
    # Report invalid/missing calibrations before downstream artifact noise: a
    # session cannot be interpreted safely without its timing contract.
    verify_sync_metadata(session, manifest)
    missing_artifacts = [
        name for name in REQUIRED_SESSION_ARTIFACTS if not (session / name).is_file()
    ]
    if missing_artifacts:
        raise ValueError(f"session missing required artifacts: {missing_artifacts}")
    artifact_hashes_before = _artifact_hash_inventory(session)
    if (interrupted_final_manifest is not None
            and interrupted_final_manifest.get("artifact_hashes") != artifact_hashes_before):
        raise ValueError("interrupted finalization artifacts no longer match the manifest")
    force_calibration = verify_force_calibration_metadata(session, manifest)
    # read_robot enforces exact row width, finite q/a values, matching joint
    # identities, units, action type, side, and manifest DOF.
    robot = read_robot(session, manifest)
    force = read_force(session)
    if force.device_t_ms is None:
        raise ValueError("force.csv lacks device_t_ms required for drift validation")
    force_fit = fit_device_clock_checked(force.t_host_s, force.device_t_ms, stream="force")
    if any(status not in ("OK", "WARN") for status in force.status):
        raise ValueError("force.csv contains missing or unsafe status values")
    force_t = (force_fit["host_intercept_s"]
               + force_fit["host_s_per_device_ms"] * force.device_t_ms)
    camera = read_camera_timeline(session, manifest)
    if np.any(np.diff(camera.frame_index) <= 0):
        raise ValueError("rgb_times.csv frame_index must be strictly increasing")
    t0 = float(manifest.get("t0_monotonic", np.nan))
    duration_s = float(manifest.get("duration_s", np.nan))
    control_hz = float(manifest.get("control_hz", np.nan))
    if not np.isfinite([t0, duration_s, control_hz]).all() or duration_s <= 0 or control_hz <= 0:
        raise ValueError("manifest capture timing is invalid")
    validate_internal_gaps(robot.t_s, max_gap_s=0.050, stream="robot")
    validate_internal_gaps(force_t, max_gap_s=0.050, stream="force")
    camera_dt = np.diff(camera.t_exposure_s)
    camera_max_gap = min(0.500, max(0.150, 3.0 * float(np.median(camera_dt))))
    validate_internal_gaps(camera.t_exposure_s, max_gap_s=camera_max_gap, stream="camera")
    _validate_interval_coverage(
        camera.t_exposure_s, t0=t0, duration_s=duration_s, max_edge_gap_s=0.150,
        stream="camera exposure",
    )
    _validate_interval_coverage(
        force_t, t0=t0, duration_s=duration_s, max_edge_gap_s=0.050, stream="force",
    )
    validate_asof_coverage(robot.t_s, camera.t_exposure_s, max_age_s=0.050)
    validate_asof_coverage(force_t, camera.t_exposure_s, max_age_s=0.050)
    targets = read_targets(session, camera)
    if any(point is None or frame != "robot_base"
           for point, frame in zip(targets.xyz, targets.xyz_frames)):
        raise ValueError("every target must contain xyz in robot_base frame")
    validate_asof_coverage(targets.t_s, camera.t_exposure_s, max_age_s=0.250)
    capture_mode = manifest["camera"]["timing"]["capture_mode"]
    artifact_validation = {
        "audio": _validate_audio_artifact(session / "audio.wav", manifest),
        "video": _validate_video_artifact(
            session / "rgb.mp4", expected_frames=len(camera.frame_index),
            expected_width=int(capture_mode["width"]),
            expected_height=int(capture_mode["height"]),
            expected_fps=float(capture_mode["fps"]),
        ),
        "depth": _validate_depth_artifacts(session, camera, manifest["camera"]),
        "observations": _validate_obs_artifact(
            session / "obs.csv", t0=t0, duration_s=duration_s, control_hz=control_hz,
            camera_frame_index=camera.frame_index, camera_host_t_s=camera.host_arrival_t_s,
        ),
    }
    artifact_hashes = _artifact_hash_inventory(session)
    if artifact_hashes != artifact_hashes_before:
        raise RuntimeError("session artifacts changed while finalization was validating them")
    durability = _capture_durability(manifest)
    _validate_capture_durability_artifact_hashes(durability, artifact_hashes)
    if interrupted_final_manifest is not None:
        previous_validation = interrupted_final_manifest.get("artifact_validation")
        current_validation = json.loads(json.dumps(artifact_validation))
        if previous_validation != current_validation:
            raise ValueError(
                "interrupted finalization artifact validation no longer reproduces"
            )
    _require_training_eligible_stop(stop_outcome)
    _require_observable_robot_feedback_liveness()
    raise AssertionError(
        "legacy supervisor feedback-liveness guard returned unexpectedly"
    )


def _wait_for_file(path: Path, process, timeout_s: float) -> dict:  # noqa: ANN001
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        if process.poll() is not None:
            raise RuntimeError(f"recorder exited before readiness (exit {process.returncode})")
        time.sleep(0.01)
    raise TimeoutError(f"timed out waiting for recorder readiness: {path}")


def read_capture_ended_ack(path: str | Path) -> dict:
    """Validate the recorder's durable frame-boundary sampling acknowledgement."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "wam_capture_ended_v1":
        raise ValueError("capture-ended acknowledgement schema is invalid")
    try:
        t0 = float(payload["t0_monotonic"])
        t_end = float(payload["t_end_monotonic"])
        duration = float(payload["actual_duration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("capture-ended timing is incomplete") from exc
    if (
        not np.isfinite([t0, t_end, duration]).all()
        or t_end < t0
        or duration < 0
        or abs(duration - (t_end - t0)) > 1e-6
    ):
        raise ValueError("capture-ended actual duration is inconsistent")
    if payload.get("reason") not in {"operator_end", "max_duration"}:
        raise ValueError("capture-ended reason is invalid")
    return payload


def supervise(
    session_dir: str | Path,
    recorder_cmd: Sequence[str],
    teleop_cmd: Sequence[str],
    stop_cmd: Sequence[str],
    *,
    readiness_timeout_s: float = 15.0,
    stop_timeout_s: float = 0.250,
    allow_preallocated_session: bool = False,
) -> int:
    """Run both children with sensor-first enable ordering and concurrent safety response."""
    validate_recorder_command(recorder_cmd)
    if not teleop_cmd or not stop_cmd:
        raise ValueError("teleop and direct-disable commands are required")
    session = Path(session_dir)
    if allow_preallocated_session:
        session.mkdir(parents=True, exist_ok=True)
    else:
        session.parent.mkdir(parents=True, exist_ok=True)
        # Raw session identities are write-once.  Never clean or append to an
        # existing complete, aborted, or partial directory.
        session.mkdir(exist_ok=False)
    ready = session / "SENSORS_READY.json"
    start = session / "START_CAPTURE"
    operator_stop = session / "STOP_CAPTURE.json"
    sentinel = session / "SAFETY_ABORT.json"
    capture_ended = session / "CAPTURE_ENDED.json"
    teleop_stopped = session / "TELEOP_STOPPED.json"
    for path in (ready, start, sentinel):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    recorder = subprocess.Popen(list(recorder_cmd), start_new_session=True)
    teleop = None
    teleop_stop_outcome = None
    teleop_stop_trigger = None
    teleop_stopped_t = None
    capture_ended_ack = None

    def publish_teleop_stopped() -> None:
        _atomic_json(teleop_stopped, {
            "schema_version": "wam_teleop_stopped_v1",
            "trigger": teleop_stop_trigger,
            "capture_ended": capture_ended_ack,
            "teleop_stop_outcome": teleop_stop_outcome,
            "t_supervisor_monotonic": teleop_stopped_t,
        })

    def current_safety_severity() -> str | None:
        if not sentinel.is_file():
            return None
        payload = json.loads(sentinel.read_text(encoding="utf-8"))
        return "hard" if payload.get("severity") == "hard" else "screened"

    try:
        readiness = _wait_for_file(ready, recorder, readiness_timeout_s)
        if readiness.get("ready") is not True or readiness.get("force_calibrated") is not True:
            raise RuntimeError(f"recorder readiness was not safe: {readiness}")
        env = dict(os.environ)
        env["REC_CSV"] = str(session / "robot.csv")
        env["WAM_START_FILE"] = str(start)
        teleop = subprocess.Popen(list(teleop_cmd), env=env, start_new_session=True)
        time.sleep(0.05)
        if teleop.poll() is not None:
            raise RuntimeError(f"teleop exited during startup (exit {teleop.returncode})")
        _atomic_json(start, {"start": True, "t_supervisor_monotonic": time.monotonic()})

        while True:
            if sentinel.is_file():
                event = json.loads(sentinel.read_text(encoding="utf-8"))
                severity = "hard" if event.get("severity") == "hard" else "screened"
                outcome = stop_teleop(
                    teleop, severity=severity, stop_cmd=stop_cmd, timeout_s=stop_timeout_s,
                    severity_probe=current_safety_severity,
                )
                hard_disabled = outcome == "hard_limit_emergency_disable"
                event = json.loads(sentinel.read_text(encoding="utf-8"))
                if event.get("severity") == "hard" and not hard_disabled:
                    outcome = stop_teleop(
                        teleop, severity="hard", stop_cmd=stop_cmd, timeout_s=stop_timeout_s,
                    )
                    hard_disabled = True
                # Do not mark aborted until the recorder is reaped: otherwise a
                # late recorder finalizer can overwrite the abort manifest.
                deadline = time.monotonic() + 2.0
                while recorder.poll() is None and time.monotonic() < deadline:
                    if current_safety_severity() == "hard" and not hard_disabled:
                        outcome = stop_teleop(
                            teleop, severity="hard", stop_cmd=stop_cmd,
                            timeout_s=stop_timeout_s,
                        )
                        hard_disabled = True
                    time.sleep(0.01)
                if recorder.poll() is None:
                    _terminate_and_reap(recorder, timeout_s=max(0.250, stop_timeout_s))
                event = json.loads(sentinel.read_text(encoding="utf-8"))
                if event.get("severity") == "hard" and not hard_disabled:
                    outcome = stop_teleop(
                        teleop, severity="hard", stop_cmd=stop_cmd, timeout_s=stop_timeout_s,
                    )
                mark_aborted(session, event.get("reason", "force safety stop"),
                             teleop_stop_outcome=outcome, safety_event=event)
                return 2
            if operator_stop.is_file() and teleop_stop_outcome is None:
                # The recorder may be blocked in camera or disk I/O and unable
                # to publish CAPTURE_ENDED yet.  The operator marker therefore
                # owns an independent, bounded-latency shutdown path.
                teleop_stop_outcome = stop_teleop(
                    teleop, severity="normal", stop_cmd=stop_cmd,
                    timeout_s=stop_timeout_s, severity_probe=current_safety_severity,
                )
                teleop_stop_trigger = "operator_stop_marker"
                teleop_stopped_t = time.monotonic()
                publish_teleop_stopped()
                # Recorder remains alive to acknowledge the last sampled frame,
                # flush every stream, and publish its durable manifest.
                continue
            if capture_ended.is_file() and capture_ended_ack is None:
                capture_ended_ack = read_capture_ended_ack(capture_ended)
                if teleop_stop_outcome is None:
                    teleop_stop_outcome = stop_teleop(
                        teleop, severity="normal", stop_cmd=stop_cmd,
                        timeout_s=stop_timeout_s, severity_probe=current_safety_severity,
                    )
                    teleop_stop_trigger = "capture_ended_ack"
                    teleop_stopped_t = time.monotonic()
                publish_teleop_stopped()
                # Recorder may now spend time flushing audio/timelines.  Teleop
                # has already been independently disabled and reaped.
                continue
            recorder_code = recorder.poll()
            teleop_code = teleop.poll()
            if (
                teleop_code is not None
                and recorder_code is None
                and teleop_stop_outcome is None
            ):
                outcome = stop_teleop(
                    teleop, severity="hard", stop_cmd=stop_cmd, timeout_s=stop_timeout_s,
                )
                _terminate_and_reap(recorder, timeout_s=max(0.250, stop_timeout_s))
                mark_aborted(session, f"teleop exited early with code {teleop_code}",
                             teleop_stop_outcome=outcome)
                return 3
            if recorder_code is not None:
                if recorder_code != 0:
                    outcome = stop_teleop(
                        teleop, severity="hard", stop_cmd=stop_cmd, timeout_s=stop_timeout_s,
                    )
                    mark_aborted(session, f"recorder failed with code {recorder_code}",
                                 teleop_stop_outcome=outcome)
                    return 4
                outcome = teleop_stop_outcome
                if teleop_stop_trigger == "operator_stop_marker" and capture_ended_ack is None:
                    mark_aborted(
                        session,
                        "recorder exited without CAPTURE_ENDED after operator stop",
                        teleop_stop_outcome=outcome,
                    )
                    return 5
                if outcome is None:
                    outcome = stop_teleop(
                        teleop, severity="normal", stop_cmd=stop_cmd,
                        timeout_s=stop_timeout_s, severity_probe=current_safety_severity,
                    )
                if sentinel.is_file():
                    event = json.loads(sentinel.read_text(encoding="utf-8"))
                    if event.get("severity") == "hard" and outcome != "hard_limit_emergency_disable":
                        outcome = stop_teleop(
                            teleop, severity="hard", stop_cmd=stop_cmd,
                            timeout_s=stop_timeout_s,
                        )
                    mark_aborted(
                        session, event.get("reason", "force safety stop during shutdown"),
                        teleop_stop_outcome=outcome, safety_event=event,
                    )
                    return 2
                finalize_session(session, stop_outcome=outcome)
                return 0
            time.sleep(0.01)
    except BaseException as exc:
        stop_error = None
        if teleop is not None:
            try:
                stop_teleop(teleop, severity="hard", stop_cmd=stop_cmd, timeout_s=stop_timeout_s)
            except Exception as emergency_exc:  # preserve evidence that direct disable was unconfirmed
                stop_error = emergency_exc
        recorder_error = None
        if recorder.poll() is None:
            try:
                _terminate_and_reap(recorder, timeout_s=max(0.250, stop_timeout_s))
            except Exception as reap_exc:
                recorder_error = reap_exc
        extra = {}
        if stop_error is not None:
            extra["emergency_stop_unconfirmed"] = str(stop_error)
        if recorder_error is not None:
            extra["recorder_reap_failed"] = str(recorder_error)
        mark_aborted(session, str(exc), **extra)
        if stop_error is not None or recorder_error is not None:
            raise RuntimeError(
                f"{exc}; emergency cleanup failed: stop={stop_error}, recorder={recorder_error}"
            ) from exc
        raise


def main(argv=None) -> int:  # noqa: ANN001
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--recorder-cmd-json", required=True)
    parser.add_argument("--teleop-cmd-json", required=True)
    parser.add_argument("--stop-cmd-json", required=True)
    parser.add_argument("--readiness-timeout-s", type=float, default=15.0)
    args = parser.parse_args(argv)
    previous_handlers = {}

    def raise_for_signal(signum, _frame):  # noqa: ANN001
        raise SupervisorSignal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, raise_for_signal)
    try:
        return supervise(
            args.session_dir, json.loads(args.recorder_cmd_json), json.loads(args.teleop_cmd_json),
            json.loads(args.stop_cmd_json), readiness_timeout_s=args.readiness_timeout_s,
        )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
